"""Sample smoothed node predictions and certify joint edge/feature budgets.

CPU model evaluators live in local_sampling, gcn_sampling, gcn_global and
bwgnn_sampling. The CUDA batch evaluators are in batched.py.
"""

import time
from typing import Any, Dict, List
import os
import numpy as np
import scipy.sparse as sp
import torch
from ..data import normalize, normalize_adj, sparse_mx_to_torch_sparse_tensor
from hybrid_smoothing.utils_hybrid import clopper_pearson_lcb, clopper_pearson_ucb
from hybrid_smoothing.certif_hybrid import (
    certificate_bound_pair,
    hybrid_certificate_grid,
    hybrid_certificate_discrete_grid,
    sequential_np_certificate_from_discrete_bounds,
)
from .local_sampling import LocalSampling


class SmoothHybrid(LocalSampling):
    """Joint target-incident topology and fixed-support Gaussian smoothing."""

    _VALID_FEATURE_NOISE_SCOPES = {"target", "all"}
    _VALID_ADJ_NORM_TYPES = {"row", "sym"}

    def __init__(
        self,
        base_classifier: torch.nn.Module,
        num_classes: int,
        prob: float,
        adj: torch.Tensor,
        features: torch.Tensor,
        sigma: float,
        cuda: bool,
        feature_noise_scope: str = "target",
        adj_norm_type: str = "row",
        adj_norm_clean: torch.Tensor | None = None,
    ):
        self.base_classifier = base_classifier
        self.num_classes = int(num_classes)
        self.prob = float(prob)
        self.sigma = float(sigma)
        self.adj = adj.coalesce()
        self.features = features
        self.cuda = bool(cuda)
        self.feature_noise_scope = self._validate_feature_noise_scope(feature_noise_scope)
        self.adj_norm_type = self._validate_adj_norm_type(adj_norm_type)
        self.device = torch.device("cuda") if self.cuda else torch.device("cpu")
        self._sample_device = torch.device("cuda") if self.cuda else torch.device("cpu")
        self.n_nodes = int(self.features.shape[0])
        self.n_features = int(self.features.shape[1])

        indices = self.adj.indices().cpu()
        keep_upper = indices[0] < indices[1]
        self.upper_i = indices[0][keep_upper].numpy()
        self.upper_j = indices[1][keep_upper].numpy()

        # Neighbor list for each node (undirected, no self loops).
        self.neighbors = [[] for _ in range(self.n_nodes)]
        for i, j in zip(self.upper_i.tolist(), self.upper_j.tolist()):
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)

        self._adj_csr_clean = sp.csr_matrix(
            (
                np.ones(self.upper_i.shape[0] * 2, dtype=np.float32),
                (
                    np.concatenate([self.upper_i, self.upper_j]),
                    np.concatenate([self.upper_j, self.upper_i]),
                ),
            ),
            shape=(self.n_nodes, self.n_nodes),
            dtype=np.float32,
        ).tocsr()
        self._adj_csr_clean = self._adj_csr_clean.maximum(self._adj_csr_clean.T).tocsr()
        self._adj_lil_clean = self._adj_csr_clean.tolil()

        self._topology_smoothing_active = self.prob < (1.0 - 1e-12)
        if adj_norm_clean is None:
            adj_norm_clean_sp = self._normalize_adj_with_self_loops(self._adj_csr_clean)
            adj_norm_clean = sparse_mx_to_torch_sparse_tensor(
                sp.coo_matrix(adj_norm_clean_sp)
            ).coalesce()
            if self.cuda:
                adj_norm_clean = adj_norm_clean.to(device="cuda")
        self.adj_norm_clean = adj_norm_clean.coalesce()

        self._target_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._fast_local_supported = False
        self._fast_local_cache: Dict[str, Any] = {}
        self._fast_local_reason = "not initialized"
        self._path_logged = False
        self._dense_batch_mode_logged = False
        # Runtime toggles for benchmarking/ablation of local optimizations.
        self._local_all_opt = os.environ.get("HYBRID_LOCAL_ALL_OPT", "1") != "0"
        self._sparse_diff_opt = os.environ.get("HYBRID_SPARSE_DIFF_OPT", "1") != "0"
        self._indexed_noise_opt = os.environ.get("HYBRID_INDEXED_NOISE_OPT", "1") != "0"
        self._setup_fast_local_path()

    @classmethod
    def _validate_feature_noise_scope(cls, scope: str) -> str:
        scope = str(scope).strip().lower()
        if scope not in cls._VALID_FEATURE_NOISE_SCOPES:
            raise ValueError(
                f"Unsupported feature_noise_scope '{scope}'. "
                f"Valid values: {sorted(cls._VALID_FEATURE_NOISE_SCOPES)}."
            )
        return scope

    def _feature_noise_indices_for_target(self, node_idx: int) -> np.ndarray:
        if self.feature_noise_scope == "all":
            return np.arange(self.n_nodes, dtype=np.int64)
        return np.array([int(node_idx)], dtype=np.int64)

    @classmethod
    def _validate_adj_norm_type(cls, mode: str) -> str:
        mode = str(mode).strip().lower()
        if mode not in cls._VALID_ADJ_NORM_TYPES:
            raise ValueError(
                f"Unsupported adj_norm_type '{mode}'. "
                f"Valid values: {sorted(cls._VALID_ADJ_NORM_TYPES)}."
            )
        return mode

    def _normalize_adj_with_self_loops(self, adj_sp: sp.spmatrix) -> sp.spmatrix:
        adj_sp = sp.csr_matrix(adj_sp)
        adj_plus_i = adj_sp + sp.eye(adj_sp.shape[0], dtype=np.float32, format="csr")
        if self.adj_norm_type == "sym":
            return normalize_adj(adj_plus_i)
        return normalize(adj_plus_i)

    def _sample_noisy_adj_norm_sparse_for_target(self, node_idx: int) -> torch.Tensor:
        cache = self._get_target_cache(node_idx)
        orig_row = cache["orig_row"]
        keep = (torch.rand((self.n_nodes,)) < self.prob).cpu().numpy()
        rand_bits = (
            torch.randint(0, 2, (self.n_nodes,), dtype=torch.int8)
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )
        noisy_row = np.where(keep, orig_row, rand_bits).astype(np.float32, copy=False)
        noisy_row[int(node_idx)] = 0.0

        adj_lil = self._adj_lil_clean.copy()
        adj_lil[:, int(node_idx)] = noisy_row.reshape((-1, 1))
        adj_lil[int(node_idx), :] = noisy_row.reshape((1, -1))
        adj_lil[int(node_idx), int(node_idx)] = 0.0
        adj_norm = self._normalize_adj_with_self_loops(adj_lil.tocsr())
        adj_norm_t = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj_norm)).coalesce()
        if self.cuda:
            adj_norm_t = adj_norm_t.to(device="cuda")
        return adj_norm_t

    def _sample_noisy_features_full(
        self,
        base_feat: torch.Tensor,
        node_idx: int,
        sigma: float | None = None,
    ) -> torch.Tensor:
        sigma_val = self.sigma if sigma is None else float(sigma)
        if sigma_val <= 0.0:
            return base_feat
        if self.feature_noise_scope == "all":
            return base_feat + torch.randn_like(base_feat) * sigma_val

        noisy_indices = self._feature_noise_indices_for_target(node_idx)
        if noisy_indices.size == 0:
            return base_feat

        x = base_feat.clone()
        idx_t = torch.as_tensor(noisy_indices, dtype=torch.long, device=x.device)
        x[idx_t] = x[idx_t] + torch.randn_like(x[idx_t]) * sigma_val
        return x

    def _sample_noisy_row_batch(self, cache: Dict[str, np.ndarray], batch_size: int) -> np.ndarray:
        # Overwrite noise: keep original w.p. prob, otherwise random Bernoulli(0.5).
        clean = cache["orig_row_torch"].unsqueeze(0).expand(batch_size, -1)
        keep = torch.rand((batch_size, clean.shape[1]), device=self._sample_device) < self.prob
        rand_bits = torch.randint(
            0, 2, (batch_size, clean.shape[1]), device=self._sample_device, dtype=torch.int8
        ).float()
        sampled = torch.where(keep, clean, rand_bits)
        # Keep diagonal fixed at 0; +I self-loop is injected separately in the predictor.
        node_idx = int(cache["node_idx"][0])
        sampled[:, node_idx] = 0.0
        return sampled.cpu().numpy().astype(np.float32, copy=False)

    def _sample_votes_target_fallback(
        self, node_idx: int, n_samples: int, batch_size: int
    ) -> np.ndarray:
        # Generic exact fallback (slow):
        # sparse row overwrite + full normalize + full model forward per sample.
        # Kept intentionally for unsupported model/configuration fast-path cases.
        node_idx = int(node_idx)
        n_samples = int(n_samples)
        batch_size = max(1, int(batch_size))
        counts = np.zeros(self.num_classes, dtype=np.int64)

        base_feat = self.features

        self.base_classifier.eval()
        with torch.no_grad():
            remaining = n_samples
            while remaining > 0:
                cur_bs = min(remaining, batch_size)
                for _ in range(cur_bs):
                    x = self._sample_noisy_features_full(base_feat=base_feat, node_idx=node_idx)
                    if self._topology_smoothing_active:
                        adj_norm = self._sample_noisy_adj_norm_sparse_for_target(node_idx)
                    else:
                        adj_norm = self.adj_norm_clean
                    logits = self.base_classifier(x, adj_norm)
                    pred = int(logits.argmax(1)[node_idx].item())
                    counts[pred] += 1
                remaining -= cur_bs
        return counts

    def sample_votes_target(self, node_idx: int, n_samples: int, batch_size: int) -> np.ndarray:
        if not self._path_logged:
            if self._fast_local_supported:
                print("[hybrid-fastpath] enabled (local incremental path).")
            else:
                print(f"[hybrid-fastpath] disabled, using fallback path: {self._fast_local_reason}")
            self._path_logged = True
        if self._fast_local_supported:
            return self._sample_votes_target_local(
                node_idx=node_idx, n_samples=n_samples, batch_size=batch_size
            )
        return self._sample_votes_target_fallback(
            node_idx=node_idx, n_samples=n_samples, batch_size=batch_size
        )

    def certify_node_hybrid(
        self,
        node_idx: int,
        n0: int,
        n: int,
        alpha: float,
        batch_size: int,
        d_list: List[int],
        eps_values: np.ndarray,
        p_minus: float,
        p_plus: float,
        compute_seq: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Select a class independently, bound its votes, then evaluate the grids."""
        started = time.time()
        node_idx, n = int(node_idx), int(n)
        pilot = self.sample_votes_target(
            node_idx=node_idx, n_samples=int(n0), batch_size=batch_size
        )
        best = int(pilot.argmax())
        counts = self.sample_votes_target(node_idx=node_idx, n_samples=n, batch_size=batch_size)
        rivals = np.where(np.arange(self.num_classes) != best, counts, -1)
        second = int(rivals.argmax()) if self.num_classes > 1 else best
        n_best, n_second = int(counts[best]), int(counts[second])
        alpha_per_class = float(alpha) / max(1, self.num_classes)
        lower = float(clopper_pearson_lcb(n, n_best, alpha_per_class))
        upper = float(clopper_pearson_ucb(n, n_second, alpha_per_class))

        cache = self._get_target_cache(node_idx)
        m1, m0 = int(cache["m1"][0]), int(cache["m0"][0])
        radii = np.asarray(eps_values, dtype=np.float64)
        budgets = np.asarray(d_list, dtype=np.int64)
        parameters = dict(
            pf_plus=p_plus,
            pf_minus=p_minus,
            max_d=int(max(budgets, default=0)),
            m1=m1,
            m0=m0,
            progress_bar=False,
        )
        grid_lower, grid_upper = certificate_bound_pair(
            hybrid_certificate_grid,
            lower,
            upper,
            sigma=self.sigma,
            eps_values=radii,
            **parameters,
        )
        grid_lower, grid_upper = grid_lower[0, budgets], grid_upper[0, budgets]
        grid_half, grid_tight = grid_lower > 0.5, grid_lower > grid_upper
        if compute_seq:
            discrete_lower, discrete_upper = certificate_bound_pair(
                hybrid_certificate_discrete_grid,
                lower,
                upper,
                **parameters,
            )
            discrete_lower, discrete_upper = discrete_lower[:, budgets], discrete_upper[:, budgets]
            sequential = sequential_np_certificate_from_discrete_bounds(
                discrete_lower,
                discrete_upper,
                radii,
                self.sigma,
            )
            seq_lower, seq_upper, seq_half, seq_tight = (grid[0] for grid in sequential)
            discrete_lower, discrete_upper = discrete_lower[0], discrete_upper[0]
        else:
            discrete_lower, discrete_upper = np.full(len(budgets), lower), np.full(
                len(budgets), upper
            )
            seq_lower, seq_upper = grid_lower.copy(), grid_upper.copy()
            seq_half, seq_tight = grid_half.copy(), grid_tight.copy()

        integers = dict(
            node_idx=node_idx,
            pred_class=best,
            second_class=second,
            n_best=n_best,
            n_second=n_second,
            n=n,
            m1=m1,
            m0=m0,
        )
        result = {key: np.array([value], dtype=np.int64) for key, value in integers.items()}
        result.update(pA_lower=np.array([lower]), pB_upper=np.array([upper]))
        result.update(
            grid_lower_disc=discrete_lower,
            grid_upper_disc=discrete_upper,
            grid_lower=grid_lower,
            grid_upper=grid_upper,
            grid_half=grid_half,
            grid_tight=grid_tight,
            grid_lower_seq=seq_lower,
            grid_upper_seq=seq_upper,
            grid_half_seq=seq_half,
            grid_tight_seq=seq_tight,
            seconds=np.array([time.time() - started]),
        )
        return result

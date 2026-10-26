"""Shared CPU sampling and per-target caches. Mixins share SmoothHybrid state."""

from typing import Dict
import numpy as np
import torch
from ..architecture import BWGNN
from .gcn_sampling import GCNSampling
from .bwgnn_sampling import BWGNNSampling


class LocalSampling(GCNSampling, BWGNNSampling):
    def _setup_fast_local_path(self):
        if self.adj_norm_type != "row":
            self._fast_local_reason = (
                f"fast local path requires row normalization; got {self.adj_norm_type}"
            )
        elif isinstance(self.base_classifier, BWGNN):
            self._setup_fast_local_bwgnn_path()
        else:
            self._setup_fast_local_gcn_path()

    def _get_target_cache(self, node_idx: int) -> Dict[str, np.ndarray]:
        node_idx = int(node_idx)
        if node_idx in self._target_cache:
            return self._target_cache[node_idx]

        state = self._fast_local_cache
        feature_indices = self._feature_noise_indices_for_target(node_idx)
        feature_mask = np.zeros(self.n_nodes, dtype=bool)
        feature_mask[feature_indices] = True
        if "adj_csr" in state:
            adjacency = state["adj_csr"]
            row = adjacency.getrow(node_idx).toarray().ravel().astype(np.float32, copy=False)
            column = adjacency.getcol(node_idx).toarray().ravel().astype(np.float32, copy=False)
        else:
            # Generic fallback still needs incident-slot counts for certification.
            row = np.zeros(self.n_nodes, dtype=np.float32)
            row[self.neighbors[node_idx]] = 1.0
            column = row.copy()
        # Self loops are deterministic model inputs, outside the smoothing slots.
        row[node_idx] = column[node_idx] = 0.0
        row_tensor = torch.from_numpy(row).to(device=self._sample_device)
        m1 = int(row.sum())
        cache = {
            "node_idx": np.array([node_idx], dtype=np.int64),
            "orig_row": row,
            "orig_row_torch": row_tensor,
            "edge_to_u": column,
            "m1": np.array([m1], dtype=np.int64),
            "m0": np.array([max(0, self.n_nodes - 1 - m1)], dtype=np.int64),
            "feature_noise_indices": feature_indices,
            "feature_noise_mask": feature_mask,
        }
        if state.get("model_type") == "bwgnn":
            cache.update(
                h0_u_clean=state["h0_clean"][node_idx].astype(np.float32, copy=False),
                sum_h_u_clean=state["sum_h_clean"][node_idx].astype(np.float32, copy=False),
                deg_b_u=float(state["deg_b"][node_idx]),
            )
        elif "adj_csr" in state:
            cache.update(
                support1_u_clean=state["support1_clean"][node_idx].astype(np.float32, copy=False),
                sum_b_u_clean=state["sum_b_clean"][node_idx].astype(np.float32, copy=False),
                deg_b_u=float(state["deg_b"][node_idx]),
            )
        else:
            empty = np.zeros(0, dtype=np.float32)
            cache.update(support1_u_clean=empty, sum_b_u_clean=empty, deg_b_u=float(m1 + 1))
        self._target_cache[node_idx] = cache
        return cache

    def _effective_mc_batch_size(self, requested_batch_size: int):
        requested_batch_size = max(1, int(requested_batch_size))
        if self._fast_local_supported:
            n_nodes = int(self._fast_local_cache["n_nodes"])
            hdim = int(self._fast_local_cache["hidden_dim"])
            n_classes = int(self.num_classes)
            if self._fast_local_cache.get("model_type") == "bwgnn":
                bytes_per_sample = n_nodes * 4
                target_bytes = 384 * 1024 * 1024
                max_by_mem = max(1, target_bytes // max(1, bytes_per_sample))
                return max(1, min(requested_batch_size, int(max_by_mem), 1024))
            indexed_enabled = bool(self._indexed_noise_opt and self.sigma > 0)

            # Memory model for batched local predictor.
            # Base buffers: sampled rows / diff / degrees / coeff + dense local tensors.
            if self.feature_noise_scope == "all" and self._local_all_opt:
                # Optimized scalar-local "all" path:
                # sampled row + one hidden-noise matrix; avoids full [bs, n, h] sum_delta materialization.
                if indexed_enabled:
                    bytes_per_sample = 2 * n_nodes * 4  # sampled row + temporary diff buffers
                    target_bytes = 384 * 1024 * 1024
                    max_by_mem = max(1, target_bytes // max(1, bytes_per_sample))
                    return max(1, min(requested_batch_size, int(max_by_mem), 512))
                bytes_per_sample = (n_nodes * 4) + (  # sampled row
                    n_nodes * hdim * 4
                )  # delta_support1
                target_bytes = 512 * 1024 * 1024
                max_by_mem = max(1, target_bytes // max(1, bytes_per_sample))
                return max(1, min(requested_batch_size, int(max_by_mem), 256))

            bytes_per_sample = (
                (4 * n_nodes * 4)  # sampled row, diff, deg, coeff
                + (2 * n_nodes * hdim * 4)  # num_all + h1_all
                + (n_nodes * n_classes * 4)  # s2_all
            )
            if self.feature_noise_scope == "all" and self.sigma > 0:
                bytes_per_sample += 2 * n_nodes * hdim * 4  # delta_support1 + sum_delta
                target_bytes = 512 * 1024 * 1024
            elif self.feature_noise_scope == "target" and self.sigma > 0:
                bytes_per_sample += n_nodes * hdim * 4  # sum_delta_target
                target_bytes = 384 * 1024 * 1024
            else:
                target_bytes = 384 * 1024 * 1024

            max_by_mem = max(1, target_bytes // max(1, bytes_per_sample))
            return max(1, min(requested_batch_size, int(max_by_mem), 128))
        feature_bytes_per_graph = int(self.features.numel()) * int(self.features.element_size())
        target_feature_bytes = 256 * 1024 * 1024
        max_by_features = max(1, target_feature_bytes // max(1, feature_bytes_per_graph))
        return max(1, min(requested_batch_size, int(max_by_features), 8))

    def _should_use_dense_batch_predict(self, sampled_rows: np.ndarray) -> bool:
        """
        Heuristic selector for dense batched predictor.
        On PubMed row-targeted settings, rows are usually sparse enough that
        the scalar local path is faster than materializing full [bs, n, h] tensors.
        """
        n_nodes = int(self._fast_local_cache["n_nodes"])
        if sampled_rows.size == 0:
            return False
        mean_nnz = float(sampled_rows.sum(axis=1).mean() + 1.0)  # +I self term
        density = mean_nnz / float(max(1, n_nodes))
        # Dense batch wins mainly for high-density rows.
        return density >= 0.35

    def _sample_votes_target_local(
        self, node_idx: int, n_samples: int, batch_size: int
    ) -> np.ndarray:
        node_idx = int(node_idx)
        cache = self._get_target_cache(node_idx)
        counts = np.zeros(self.num_classes, dtype=np.int64)
        remaining = int(n_samples)
        eff_bs = self._effective_mc_batch_size(batch_size)

        if self._fast_local_cache.get("model_type") == "bwgnn":
            if not self._dense_batch_mode_logged:
                print("[hybrid-fastpath] predictor mode for MC votes: bwgnn-row-scalar")
                self._dense_batch_mode_logged = True
            while remaining > 0:
                cur_bs = min(remaining, eff_bs)
                sampled_rows = self._sample_noisy_row_batch(cache, cur_bs)
                preds = np.empty(cur_bs, dtype=np.int64)
                for i in range(cur_bs):
                    preds[i] = self._predict_target_bwgnn_row_sample(
                        node_idx=node_idx,
                        e_row=sampled_rows[i],
                        cache=cache,
                    )
                counts += np.bincount(preds, minlength=self.num_classes).astype(
                    np.int64, copy=False
                )
                remaining -= cur_bs
            return counts

        while remaining > 0:
            cur_bs = min(remaining, eff_bs)
            sampled_rows = self._sample_noisy_row_batch(cache, cur_bs)
            use_dense_batch_predict = self._should_use_dense_batch_predict(sampled_rows)
            indexed_enabled = bool(self._indexed_noise_opt and self.sigma > 0)
            if not self._dense_batch_mode_logged:
                mode = "dense-batch" if use_dense_batch_predict else "scalar-local"
                if self.feature_noise_scope == "all":
                    mode += "-opt" if self._local_all_opt else "-legacy"
                    if self._local_all_opt:
                        mode += "-indexed" if indexed_enabled else "-fullnoise"
                print(f"[hybrid-fastpath] predictor mode for MC votes: {mode}")
                self._dense_batch_mode_logged = True
            if self.feature_noise_scope == "all":
                if self._local_all_opt and not use_dense_batch_predict:
                    preds = np.empty(cur_bs, dtype=np.int64)
                    if indexed_enabled:
                        for i in range(cur_bs):
                            preds[i] = self._predict_target_local_sample_all_indexed(
                                node_idx=node_idx,
                                e_row=sampled_rows[i],
                                cache=cache,
                            )
                    else:
                        delta_batch = self._sample_feature_delta_all_batch(cur_bs)
                        for i in range(cur_bs):
                            preds[i] = self._predict_target_local_sample_all_sparse(
                                node_idx=node_idx,
                                e_row=sampled_rows[i],
                                cache=cache,
                                delta_support1=delta_batch[i],
                            )
                    counts += np.bincount(preds, minlength=self.num_classes).astype(
                        np.int64, copy=False
                    )
                    remaining -= cur_bs
                    continue
                delta_batch, sum_batch = self._sample_feature_state_all_batch(cur_bs)
                fstate_batch = {
                    "type": "all",
                    "delta_support1_batch": delta_batch,
                    "sum_delta_batch": sum_batch,
                }
            else:
                fstate_batch = self._sample_feature_state_target_batch(cur_bs)

            if use_dense_batch_predict:
                preds = self._predict_target_local_batch(
                    node_idx=node_idx,
                    sampled_rows=sampled_rows,
                    cache=cache,
                    fstate_batch=fstate_batch,
                )
            else:
                preds = np.empty(cur_bs, dtype=np.int64)
                if fstate_batch["type"] == "all":
                    for i in range(cur_bs):
                        fstate = {
                            "type": "all",
                            "delta_support1": fstate_batch["delta_support1_batch"][i],
                            "sum_delta": fstate_batch["sum_delta_batch"][i],
                        }
                        preds[i] = self._predict_target_local_sample(
                            node_idx=node_idx, e_row=sampled_rows[i], cache=cache, fstate=fstate
                        )
                else:
                    for i in range(cur_bs):
                        fstate = {
                            "type": "target",
                            "delta_u": fstate_batch["delta_u_batch"][i],
                        }
                        preds[i] = self._predict_target_local_sample(
                            node_idx=node_idx, e_row=sampled_rows[i], cache=cache, fstate=fstate
                        )
            counts += np.bincount(preds, minlength=self.num_classes).astype(np.int64, copy=False)
            remaining -= cur_bs
        return counts

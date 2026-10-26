"""Target-row asymmetric topology sampling, from the binary-comparison pipeline."""
from __future__ import annotations
import types
from typing import Any, Dict
import numpy as np
import scipy.sparse as sp
import torch
from hybrid_graph.data import sparse_mx_to_torch_sparse_tensor

def _sample_noisy_row_batch_asym(self, cache: Dict[str, np.ndarray], batch_size: int) -> np.ndarray:
    """Bernoulli(p_minus) deletions on edges, Bernoulli(p_plus) insertions on non-edges."""
    clean = cache["orig_row_torch"].unsqueeze(0).expand(batch_size, -1)
    u = torch.rand((batch_size, clean.shape[1]), device=self._sample_device)
    flip_prob = torch.where(clean > 0.5, self._p_minus_t, self._p_plus_t)
    flip = u < flip_prob
    sampled = torch.where(flip, 1.0 - clean, clean)
    node_idx = int(cache["node_idx"][0])
    sampled[:, node_idx] = 0.0
    return sampled.cpu().numpy().astype(np.float32, copy=False)

def _sample_noisy_adj_norm_sparse_for_target_asym(self, node_idx: int) -> torch.Tensor:
    cache = self._get_target_cache(node_idx)
    orig_row = cache["orig_row"]
    flip_prob = np.where(orig_row > 0.5, float(self.p_minus), float(self.p_plus)).astype(np.float32)
    flip = np.random.rand(self.n_nodes).astype(np.float32) < flip_prob
    noisy_row = np.where(flip, 1.0 - orig_row, orig_row).astype(np.float32, copy=False)
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

def enable_asymmetric_kernel(smoother: Any, *, p_minus: float, p_plus: float) -> Any:
    """Rebind `smoother`'s topology sampler to the asymmetric (p_minus, p_plus) kernel."""
    p_minus = float(p_minus)
    p_plus = float(p_plus)
    if not (0.0 <= p_minus < 1.0) or not (0.0 <= p_plus < 1.0):
        raise ValueError(f"p_minus/p_plus must lie in [0, 1), got {p_minus}, {p_plus}")
    smoother.p_minus = p_minus
    smoother.p_plus = p_plus
    smoother._p_minus_t = torch.tensor(p_minus, device=smoother._sample_device, dtype=torch.float32)
    smoother._p_plus_t = torch.tensor(p_plus, device=smoother._sample_device, dtype=torch.float32)
    smoother._topology_smoothing_active = (p_minus > 1e-12) or (p_plus > 1e-12)
    smoother._sample_noisy_row_batch = types.MethodType(_sample_noisy_row_batch_asym, smoother)
    smoother._sample_noisy_adj_norm_sparse_for_target = types.MethodType(
        _sample_noisy_adj_norm_sparse_for_target_asym, smoother
    )
    return smoother

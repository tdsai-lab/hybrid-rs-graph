"""Exact global-feature GCN evaluation on the noisy rows needed by a draw."""

from typing import Dict
import numpy as np


class GlobalGCNSampling:
    """CPU evaluator methods sharing the state initialized by SmoothHybrid."""

    def _predict_target_local_sample_all_sparse(
        self,
        node_idx: int,
        e_row: np.ndarray,
        cache: Dict[str, np.ndarray],
        delta_support1: np.ndarray,
    ) -> int:
        return self._predict_global_features(node_idx, e_row, cache, delta_support1)

    def _predict_target_local_sample_all_indexed(
        self,
        node_idx: int,
        e_row: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> int:
        return self._predict_global_features(node_idx, e_row, cache)

    def _predict_global_features(self, node_idx, e_row, cache, delta_support1=None):
        """Share the GCN arithmetic between full and lazily drawn feature noise."""
        state = self._fast_local_cache
        support = state["support1_clean"]
        sums = state["sum_b_clean"]
        degrees = state["deg_b"]
        bias1, bias2 = state["b1"], state["b2"]
        node = int(node_idx)
        difference = e_row - cache["orig_row"]
        if self._sparse_diff_opt:
            changed = np.flatnonzero(difference != 0.0)
            changed_weights = difference[changed] if changed.size else None
            difference_sum = float(changed_weights.sum()) if changed.size else 0.0
        else:
            changed = np.arange(len(difference), dtype=np.int64)
            changed_weights = difference
            difference_sum = float(difference.sum())
        degree = max(float(cache["deg_b_u"]) + difference_sum, 1.0)

        coefficients = e_row.copy()
        coefficients[node] += 1.0
        selected = np.flatnonzero(coefficients > 0.0)
        if not selected.size:
            return int(np.argmax(bias2))
        rows = state["adj_plus_i_csr"][selected]

        if delta_support1 is None:
            needed = np.unique(rows.indices.astype(np.int64, copy=False))
            if changed.size:
                needed = np.union1d(needed, changed)
            if node not in needed:
                needed = np.union1d(needed, np.array([node], dtype=np.int64))
            noise = self._sample_feature_delta_all_indexed(needed)
            noise_sums = rows[:, needed].dot(noise).astype(np.float32, copy=False)
            target_noise = noise[int(np.searchsorted(needed, node))]
            changed_noise = noise[np.searchsorted(needed, changed)]
        else:
            noise_sums = rows.dot(delta_support1).astype(np.float32, copy=False)
            target_noise = delta_support1[node]
            changed_noise = delta_support1[changed]

        target_support = cache["support1_u_clean"] + target_noise
        target_sum = cache["sum_b_u_clean"] + noise_sums[int(np.searchsorted(selected, node))]
        if changed.size:
            target_sum = (
                target_sum + changed_weights @ support[changed] + changed_weights @ changed_noise
            )
        target_hidden = np.maximum(target_sum / degree + bias1, 0.0)

        is_target = selected == node
        hidden = np.empty((len(selected), int(state["hidden_dim"])), dtype=np.float32)
        hidden[is_target] = target_hidden
        others = selected[~is_target]
        if others.size:
            edge_difference = (e_row[others] - cache["edge_to_u"][others]).astype(
                np.float32, copy=False
            )
            other_degrees = np.maximum(degrees[others] + edge_difference, 1.0)
            other_sums = (
                sums[others]
                + noise_sums[~is_target]
                + edge_difference[:, None] * target_support[None, :]
            )
            hidden[~is_target] = np.maximum(
                other_sums / other_degrees[:, None] + bias1[None, :], 0.0
            )

        second_support = hidden @ state["w2"]
        logits = (coefficients[selected][:, None] * second_support).sum(axis=0) / degree + bias2
        return int(np.argmax(logits))

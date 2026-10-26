"""BWGNN CPU setup and exact target-row Bernstein-filter evaluation."""

import inspect
from typing import Dict
import numpy as np
import scipy.sparse as sp
import torch


class BWGNNSampling:
    def _linear_params_np(self, layer: torch.nn.Module, name: str):
        if not hasattr(layer, "weight"):
            self._fast_local_reason = f"BWGNN fast path missing {name}.weight"
            return None
        weight = layer.weight.detach().cpu().numpy().astype(np.float32, copy=False)
        bias_t = getattr(layer, "bias", None)
        if bias_t is None:
            bias = np.zeros(weight.shape[0], dtype=np.float32)
        else:
            bias = bias_t.detach().cpu().numpy().astype(np.float32, copy=False)
        return weight, bias

    def _setup_fast_local_bwgnn_path(self):
        if self.features.dim() != 2:
            self._fast_local_reason = f"features dim={self.features.dim()} (expected 2)"
            return
        if self.sigma > 0.0 and self.feature_noise_scope != "target":
            self._fast_local_reason = (
                "BWGNN fast local path currently supports Gaussian feature noise "
                f"only for feature_noise_scope=target; got {self.feature_noise_scope}"
            )
            return

        order = int(getattr(self.base_classifier, "order", -1))
        if order < 0 or order > 2:
            self._fast_local_reason = (
                "BWGNN fast local path currently supports Bernstein order <= 2; "
                f"got order={order}"
            )
            return
        try:
            sig = inspect.signature(self.base_classifier.forward)
            params = list(sig.parameters.keys())
            if params != ["x", "adj"]:
                self._fast_local_reason = f"unsupported BWGNN forward signature: {params}"
                return
        except Exception:
            self._fast_local_reason = "cannot inspect BWGNN forward signature"
            return

        linear = getattr(self.base_classifier, "linear", None)
        linear2 = getattr(self.base_classifier, "linear2", None)
        linear3 = getattr(self.base_classifier, "linear3", None)
        linear4 = getattr(self.base_classifier, "linear4", None)
        if linear is None or linear2 is None or linear3 is None or linear4 is None:
            self._fast_local_reason = (
                "BWGNN fast path requires linear/linear2/linear3/linear4 layers"
            )
            return

        params = {
            "w0b0": self._linear_params_np(linear, "linear"),
            "w1b1": self._linear_params_np(linear2, "linear2"),
            "w3b3": self._linear_params_np(linear3, "linear3"),
            "w4b4": self._linear_params_np(linear4, "linear4"),
        }
        if any(v is None for v in params.values()):
            return
        w0, b0 = params["w0b0"]
        w1, b1 = params["w1b1"]
        w3, b3 = params["w3b3"]
        w4, b4 = params["w4b4"]

        theta = getattr(self.base_classifier, "theta", None)
        if theta is None:
            self._fast_local_reason = "BWGNN fast path missing theta coefficients"
            return
        theta_np = theta.detach().cpu().numpy().astype(np.float32, copy=False)
        if theta_np.ndim != 2 or theta_np.shape[0] != order + 1 or theta_np.shape[1] < order + 1:
            self._fast_local_reason = f"unexpected BWGNN theta shape: {tuple(theta_np.shape)}"
            return

        adj = self.adj.coalesce().cpu()
        idx = adj.indices().numpy()
        n_nodes = int(adj.size(0))
        adj_csr = sp.csr_matrix(
            (np.ones(idx.shape[1], dtype=np.float32), (idx[0], idx[1])),
            shape=(n_nodes, n_nodes),
            dtype=np.float32,
        )
        adj_csr.data[:] = 1.0
        adj_csr = adj_csr.maximum(adj_csr.T).tocsr()
        adj_plus_i = (adj_csr + sp.eye(n_nodes, dtype=np.float32, format="csr")).tocsr()

        x_np = self.features.detach().cpu().numpy().astype(np.float32, copy=False)
        if w0.ndim != 2 or w1.ndim != 2 or w3.ndim != 2 or w4.ndim != 2:
            self._fast_local_reason = "BWGNN linear weights are not 2D"
            return
        if x_np.shape[1] != w0.shape[1]:
            self._fast_local_reason = "incompatible BWGNN input feature/linear dimensions"
            return
        if w0.shape[0] != w1.shape[1]:
            self._fast_local_reason = "incompatible BWGNN linear/linear2 dimensions"
            return
        hidden_dim = int(w1.shape[0])
        if w3.shape[1] != hidden_dim * (order + 1):
            self._fast_local_reason = "incompatible BWGNN filter/linear3 dimensions"
            return
        if w3.shape[0] != w4.shape[1]:
            self._fast_local_reason = "incompatible BWGNN linear3/linear4 dimensions"
            return

        h = np.maximum(x_np @ w0.T + b0[None, :], 0.0)
        h0_clean = np.maximum(h @ w1.T + b1[None, :], 0.0).astype(np.float32, copy=False)
        sum_h_clean = adj_plus_i.dot(h0_clean).astype(np.float32, copy=False)
        deg_b = np.asarray(adj_plus_i.sum(axis=1)).ravel().astype(np.float32, copy=False)

        self._fast_local_cache = {
            "model_type": "bwgnn",
            "n_nodes": n_nodes,
            "adj_csr": adj_csr,
            "adj_plus_i": adj_plus_i,
            "adj_plus_i_csr": adj_plus_i.tocsr(),
            "deg_b": deg_b,
            "hidden_dim": hidden_dim,
            "bwgnn_order": order,
            "theta": theta_np[:, : order + 1],
            "x_clean": x_np,
            "h0_clean": h0_clean,
            "sum_h_clean": sum_h_clean,
            "bwgnn_w0": w0,
            "bwgnn_b0": b0,
            "bwgnn_w1": w1,
            "bwgnn_b1": b1,
            "bwgnn_w3": w3,
            "bwgnn_b3": b3,
            "bwgnn_w4": w4,
            "bwgnn_b4": b4,
        }
        self._fast_local_supported = True
        self._fast_local_reason = f"ok-bwgnn-row-target-order{order}"

    def _sample_bwgnn_target_h0(self, node_idx: int) -> np.ndarray:
        fc = self._fast_local_cache
        idx = int(node_idx)
        if self.sigma <= 0.0:
            return fc["h0_clean"][idx].astype(np.float32, copy=False)

        x_u = (
            fc["x_clean"][idx]
            + np.random.randn(fc["x_clean"].shape[1]).astype(np.float32) * self.sigma
        )
        h = np.maximum(x_u @ fc["bwgnn_w0"].T + fc["bwgnn_b0"], 0.0)
        h0_u = np.maximum(h @ fc["bwgnn_w1"].T + fc["bwgnn_b1"], 0.0)
        return h0_u.astype(np.float32, copy=False)

    def _logits_target_bwgnn_row_sample(
        self,
        node_idx: int,
        e_row: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Exact target-row BWGNN predictor for row-normalized adjacency and order <= 2.

        BWGNN applies Bernstein filters to h0 = MLP(features). For the target node
        and row normalization, order 2 only needs P h0 and P^2 h0 on that target row.
        When the target row/column is resampled, only the target row and the selected
        neighbors' rows change, so we update those sums locally.
        """
        fc = self._fast_local_cache
        h0_clean = fc["h0_clean"]
        sum_h_clean = fc["sum_h_clean"]
        deg_b = fc["deg_b"]
        theta = fc["theta"]
        order = int(fc["bwgnn_order"])
        hidden_dim = int(fc["hidden_dim"])

        idx = int(node_idx)
        orig_row = cache["orig_row"]
        edge_to_u = cache["edge_to_u"]
        h_u_clean = cache["h0_u_clean"]
        sum_h_u_clean = cache["sum_h_u_clean"]
        deg_b_u = float(cache["deg_b_u"])

        h_u = self._sample_bwgnn_target_h0(idx)
        delta_h_u = h_u - h_u_clean

        diff = e_row - orig_row
        if self._sparse_diff_opt:
            changed = np.flatnonzero(diff != 0.0)
            diff_changed = diff[changed] if changed.size > 0 else None
            diff_sum = float(diff_changed.sum()) if changed.size > 0 else 0.0
        else:
            changed = np.arange(diff.shape[0], dtype=np.int64)
            diff_changed = diff
            diff_sum = float(diff.sum())

        deg_u = deg_b_u + diff_sum
        if deg_u <= 0:
            deg_u = 1.0

        if changed.size > 0:
            p_h_num_u = sum_h_u_clean + delta_h_u + diff_changed @ h0_clean[changed]
        else:
            p_h_num_u = sum_h_u_clean + delta_h_u
        p_h_u = p_h_num_u / deg_u

        if order >= 2:
            coeff = e_row.copy()
            coeff[idx] += 1.0
            selected = np.flatnonzero(coeff > 0.0).astype(np.int64, copy=False)
            if selected.size == 0:
                p2_h_u = np.zeros(hidden_dim, dtype=np.float32)
            else:
                p_h_selected = np.empty((selected.shape[0], hidden_dim), dtype=np.float32)
                is_u = selected == idx
                if is_u.any():
                    p_h_selected[is_u] = p_h_u

                others = selected[~is_u]
                if others.size > 0:
                    delta_edge = (e_row[others] - edge_to_u[others]).astype(np.float32, copy=False)
                    deg_other = np.maximum(deg_b[others] + delta_edge, 1.0)
                    p_h_num_other = (
                        sum_h_clean[others]
                        + edge_to_u[others, None] * delta_h_u[None, :]
                        + delta_edge[:, None] * h_u[None, :]
                    )
                    p_h_selected[~is_u] = p_h_num_other / deg_other[:, None]

                p2_h_u = (coeff[selected][:, None] * p_h_selected).sum(axis=0) / deg_u

        lap_terms = [h_u]
        if order >= 1:
            lap_terms.append(h_u - p_h_u)
        if order >= 2:
            lap_terms.append(h_u - 2.0 * p_h_u + p2_h_u)

        filters = []
        for theta_i in theta:
            out = np.zeros(hidden_dim, dtype=np.float32)
            for k in range(order + 1):
                out = out + theta_i[k] * lap_terms[k]
            filters.append(out)
        h_filters = np.concatenate(filters, axis=0)
        h = np.maximum(h_filters @ fc["bwgnn_w3"].T + fc["bwgnn_b3"], 0.0)
        logits = h @ fc["bwgnn_w4"].T + fc["bwgnn_b4"]
        return logits.astype(np.float32, copy=False)

    def _predict_target_bwgnn_row_sample(
        self,
        node_idx: int,
        e_row: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> int:
        logits = self._logits_target_bwgnn_row_sample(
            node_idx=node_idx,
            e_row=e_row,
            cache=cache,
        )
        return int(np.argmax(logits))

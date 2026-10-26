"""Two-layer GCN CPU setup, Gaussian draws and dense/target evaluators."""

import inspect
from typing import Any, Dict
import numpy as np
import scipy.sparse as sp
from ..architecture import GCN, GraphConvolution
from .gcn_global import GlobalGCNSampling


class GCNSampling(GlobalGCNSampling):
    def _setup_fast_local_gcn_path(self):
        if int(getattr(self.base_classifier, "num_layers", 2)) != 2:
            self._fast_local_reason = (
                "fast local path currently supports only 2-layer GCNs; "
                f"got num_layers={getattr(self.base_classifier, 'num_layers', 'unknown')}"
            )
            return
        gc1 = getattr(self.base_classifier, "gc1", None)
        gc2 = getattr(self.base_classifier, "gc2", None)
        if not isinstance(self.base_classifier, GCN):
            self._fast_local_reason = (
                f"unsupported model type: {type(self.base_classifier).__name__}"
            )
            return
        if gc1 is None or gc2 is None:
            self._fast_local_reason = "missing gc1/gc2 layers"
            return
        if not isinstance(gc1, GraphConvolution) or not isinstance(gc2, GraphConvolution):
            self._fast_local_reason = "gc1/gc2 are not GraphConvolution"
            return
        for attr in ("weight",):
            if not hasattr(gc1, attr) or not hasattr(gc2, attr):
                self._fast_local_reason = "missing gc layer weights"
                return
        if self.features.dim() != 2:
            self._fast_local_reason = f"features dim={self.features.dim()} (expected 2)"
            return
        try:
            sig = inspect.signature(self.base_classifier.forward)
            params = list(sig.parameters.keys())
            if params != ["x", "adj"]:
                self._fast_local_reason = f"unsupported forward signature: {params}"
                return
        except Exception:
            self._fast_local_reason = "cannot inspect forward signature"
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
        w1 = gc1.weight.detach().cpu().numpy().astype(np.float32, copy=False)
        w2 = gc2.weight.detach().cpu().numpy().astype(np.float32, copy=False)
        if w1.ndim != 2 or w2.ndim != 2:
            self._fast_local_reason = "gc weights are not 2D"
            return
        if x_np.shape[1] != w1.shape[0] or w1.shape[1] != w2.shape[0]:
            self._fast_local_reason = "incompatible feature/weight dimensions"
            return
        b1 = (
            gc1.bias.detach().cpu().numpy().astype(np.float32, copy=False)
            if getattr(gc1, "bias", None) is not None
            else np.zeros(w1.shape[1], dtype=np.float32)
        )
        b2 = (
            gc2.bias.detach().cpu().numpy().astype(np.float32, copy=False)
            if getattr(gc2, "bias", None) is not None
            else np.zeros(w2.shape[1], dtype=np.float32)
        )

        support1_clean = x_np @ w1
        sum_b_clean = adj_plus_i.dot(support1_clean).astype(np.float32, copy=False)
        deg_b = np.asarray(adj_plus_i.sum(axis=1)).ravel().astype(np.float32, copy=False)

        cov_h = (w1.T @ w1).astype(np.float64, copy=False)
        jitter = 1e-8
        chol_h = None
        for _ in range(6):
            try:
                chol_h = np.linalg.cholesky(
                    cov_h + np.eye(cov_h.shape[0], dtype=np.float64) * jitter
                )
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        if chol_h is None:
            eigvals, eigvecs = np.linalg.eigh(cov_h)
            eigvals = np.maximum(eigvals, 1e-10)
            chol_h = (eigvecs * np.sqrt(eigvals)[None, :]).astype(np.float64, copy=False)
        chol_h = (self.sigma * chol_h).astype(np.float32, copy=False)

        self._fast_local_cache = {
            "model_type": "gcn",
            "n_nodes": n_nodes,
            "adj_csr": adj_csr,
            "adj_plus_i": adj_plus_i,
            "adj_plus_i_csr": adj_plus_i.tocsr(),
            "support1_clean": support1_clean,
            "sum_b_clean": sum_b_clean,
            "deg_b": deg_b,
            "w2": w2,
            "b1": b1,
            "b2": b2,
            "hidden_dim": int(w1.shape[1]),
            "chol_h": chol_h,
        }
        self._fast_local_supported = True
        self._fast_local_reason = "ok"

    def _sample_feature_state_all_batch(self, batch_size: int):
        fc = self._fast_local_cache
        n_nodes = int(fc["n_nodes"])
        hdim = int(fc["hidden_dim"])

        if self.sigma <= 0:
            delta_batch = np.zeros((batch_size, n_nodes, hdim), dtype=np.float32)
            sum_batch = np.zeros((batch_size, n_nodes, hdim), dtype=np.float32)
            return delta_batch, sum_batch

        # Chunked generation/aggregation to keep memory bounded on PubMed while
        # still using batched sparse multiplication.
        delta_batch = np.empty((batch_size, n_nodes, hdim), dtype=np.float32)
        sum_batch = np.empty((batch_size, n_nodes, hdim), dtype=np.float32)
        chunk = max(1, min(batch_size, 16))
        for start in range(0, batch_size, chunk):
            end = min(batch_size, start + chunk)
            cur = end - start

            # Layout [n, cur, h] avoids an extra full transpose copy before CSR dot.
            z_nch = np.random.randn(n_nodes, cur, hdim).astype(np.float32, copy=False)
            delta_nch = np.matmul(z_nch, fc["chol_h"].T).astype(np.float32, copy=False)

            delta_2d = delta_nch.reshape(n_nodes, cur * hdim)
            sum_2d = fc["adj_plus_i"].dot(delta_2d).astype(np.float32, copy=False)
            sum_nch = sum_2d.reshape(n_nodes, cur, hdim)

            # moveaxis is a view; assignment writes into preallocated batch buffers.
            delta_batch[start:end] = np.moveaxis(delta_nch, 1, 0)
            sum_batch[start:end] = np.moveaxis(sum_nch, 1, 0)
        return delta_batch, sum_batch

    def _sample_feature_delta_all_batch(self, batch_size: int):
        """Generate hidden-space Gaussian feature noise for all nodes, without full aggregation."""
        fc = self._fast_local_cache
        n_nodes = int(fc["n_nodes"])
        hdim = int(fc["hidden_dim"])

        if self.sigma <= 0:
            return np.zeros((batch_size, n_nodes, hdim), dtype=np.float32)

        delta_batch = np.empty((batch_size, n_nodes, hdim), dtype=np.float32)
        chunk = max(1, min(batch_size, 16))
        for start in range(0, batch_size, chunk):
            end = min(batch_size, start + chunk)
            cur = end - start
            z_nch = np.random.randn(n_nodes, cur, hdim).astype(np.float32, copy=False)
            delta_nch = np.matmul(z_nch, fc["chol_h"].T).astype(np.float32, copy=False)
            delta_batch[start:end] = np.moveaxis(delta_nch, 1, 0)
        return delta_batch

    def _sample_feature_delta_all_indexed(self, indices: np.ndarray):
        """Generate hidden-space Gaussian feature noise only for requested node indices."""
        fc = self._fast_local_cache
        hdim = int(fc["hidden_dim"])
        n_sel = int(indices.shape[0])
        if n_sel <= 0:
            return np.zeros((0, hdim), dtype=np.float32)
        if self.sigma <= 0:
            return np.zeros((n_sel, hdim), dtype=np.float32)
        z = np.random.randn(n_sel, hdim).astype(np.float32, copy=False)
        return np.matmul(z, fc["chol_h"].T).astype(np.float32, copy=False)

    def _sample_feature_state_target_batch(self, batch_size: int):
        fc = self._fast_local_cache
        hdim = int(fc["hidden_dim"])
        if self.sigma <= 0:
            delta_u_batch = np.zeros((batch_size, hdim), dtype=np.float32)
        else:
            z = np.random.randn(batch_size, hdim).astype(np.float32, copy=False)
            delta_u_batch = np.matmul(z, fc["chol_h"].T).astype(np.float32, copy=False)
        return {"type": "target", "delta_u_batch": delta_u_batch}

    def _predict_target_local_sample(
        self, node_idx: int, e_row: np.ndarray, cache: Dict[str, np.ndarray], fstate: Dict[str, Any]
    ):
        fc = self._fast_local_cache
        support1_clean = fc["support1_clean"]
        sum_b_clean = fc["sum_b_clean"]
        deg_b = fc["deg_b"]
        w2 = fc["w2"]
        b1 = fc["b1"]
        b2 = fc["b2"]
        hidden_dim = int(fc["hidden_dim"])

        idx = int(node_idx)
        orig_row = cache["orig_row"]
        edge_to_u = cache["edge_to_u"]
        support1_u_clean = cache["support1_u_clean"]
        sum_b_u_clean = cache["sum_b_u_clean"]
        deg_b_u = float(cache["deg_b_u"])

        diff = e_row - orig_row
        diff_sum = float(diff.sum())
        deg_u = deg_b_u + diff_sum
        if deg_u <= 0:
            deg_u = 1.0

        if fstate["type"] == "all":
            delta_support1 = fstate["delta_support1"]
            sum_delta = fstate["sum_delta"]
            support1_noisy = support1_clean + delta_support1
            support1_u_noisy = support1_noisy[idx]
            num_u = sum_b_u_clean + sum_delta[idx] + diff @ support1_noisy
        else:
            delta_u = fstate["delta_u"]
            support1_u_noisy = support1_u_clean + delta_u
            if self._sparse_diff_opt:
                changed = np.flatnonzero(diff != 0.0)
                if changed.size > 0:
                    diff_changed = diff[changed]
                    num_u = (
                        sum_b_u_clean
                        + delta_u
                        + diff_changed @ support1_clean[changed]
                        + diff[idx] * delta_u
                    )
                else:
                    num_u = sum_b_u_clean + delta_u + diff[idx] * delta_u
            else:
                num_u = sum_b_u_clean + delta_u + diff @ support1_clean + diff[idx] * delta_u

        h1_u = np.maximum(num_u / deg_u + b1, 0.0)

        coeff = e_row.copy()
        coeff[idx] += 1.0
        selected = np.flatnonzero(coeff > 0.0)
        if selected.size == 0:
            return int(np.argmax(b2))

        is_u = selected == idx
        h1_sel = np.empty((selected.shape[0], hidden_dim), dtype=np.float32)
        if is_u.any():
            h1_sel[is_u] = h1_u

        others = selected[~is_u]
        if others.size > 0:
            delta_edge = (e_row[others] - edge_to_u[others]).astype(np.float32, copy=False)
            deg_other = np.maximum(deg_b[others] + delta_edge, 1.0)

            if fstate["type"] == "all":
                sum_delta_other = fstate["sum_delta"][others]
            else:
                sum_delta_other = edge_to_u[others, None] * fstate["delta_u"][None, :]
            num_other = (
                sum_b_clean[others]
                + sum_delta_other
                + delta_edge[:, None] * support1_u_noisy[None, :]
            )
            h1_other = np.maximum(num_other / deg_other[:, None] + b1[None, :], 0.0)
            h1_sel[~is_u] = h1_other

        s2_sel = h1_sel @ w2
        logits_u = (coeff[selected][:, None] * s2_sel).sum(axis=0) / deg_u + b2
        return int(np.argmax(logits_u))

    def _predict_target_local_batch(
        self,
        node_idx: int,
        sampled_rows: np.ndarray,
        cache: Dict[str, np.ndarray],
        fstate_batch: Dict[str, Any],
        return_logits: bool = False,
    ) -> np.ndarray:
        """
        Batched local incremental predictor.
        Exact same logic as scalar local path, but applied to a full MC batch.
        """
        fc = self._fast_local_cache
        support1_clean = fc["support1_clean"]
        sum_b_clean = fc["sum_b_clean"]
        deg_b = fc["deg_b"]
        w2 = fc["w2"]
        b1 = fc["b1"]
        b2 = fc["b2"]

        idx = int(node_idx)
        orig_row = cache["orig_row"]
        edge_to_u = cache["edge_to_u"]
        support1_u_clean = cache["support1_u_clean"]
        sum_b_u_clean = cache["sum_b_u_clean"]
        deg_b_u = float(cache["deg_b_u"])

        e_rows = sampled_rows.astype(np.float32, copy=False)
        bs = int(e_rows.shape[0])
        diff = e_rows - orig_row[None, :]
        deg_u = np.maximum(deg_b_u + diff.sum(axis=1), 1.0).astype(np.float32, copy=False)

        coeff = e_rows.copy()
        coeff[:, idx] += 1.0

        deg_all = np.maximum(deg_b[None, :] + diff, 1.0).astype(np.float32, copy=False)

        if fstate_batch["type"] == "all":
            delta_support1 = fstate_batch["delta_support1_batch"]
            sum_delta = fstate_batch["sum_delta_batch"]
            support1_u_noisy = support1_u_clean[None, :] + delta_support1[:, idx, :]
            # num_u = sum_b_u + sum_delta_u + diff @ (support1_clean + delta_support1)
            num_u = (
                sum_b_u_clean[None, :]
                + sum_delta[:, idx, :]
                + diff @ support1_clean
                + np.einsum("bn,bnh->bh", diff, delta_support1, optimize=True)
            )
            num_all = (
                sum_b_clean[None, :, :]
                + sum_delta
                + diff[:, :, None] * support1_u_noisy[:, None, :]
            )
        else:
            delta_u_batch = fstate_batch["delta_u_batch"]
            support1_u_noisy = support1_u_clean[None, :] + delta_u_batch
            num_u = (
                sum_b_u_clean[None, :]
                + delta_u_batch
                + diff @ support1_clean
                + diff[:, idx][:, None] * delta_u_batch
            )
            sum_delta_target = edge_to_u[None, :, None] * delta_u_batch[:, None, :]
            num_all = (
                sum_b_clean[None, :, :]
                + sum_delta_target
                + diff[:, :, None] * support1_u_noisy[:, None, :]
            )

        h1_all = np.maximum(num_all / deg_all[:, :, None] + b1[None, None, :], 0.0)
        h1_u = np.maximum(num_u / deg_u[:, None] + b1[None, :], 0.0)
        h1_all[:, idx, :] = h1_u

        s2_all = np.einsum("bnh,hc->bnc", h1_all, w2, optimize=True)
        logits = np.einsum("bn,bnc->bc", coeff, s2_all, optimize=True)
        logits = logits / deg_u[:, None] + b2[None, :]
        if return_logits:
            return logits.astype(np.float32, copy=False)
        return np.asarray(logits.argmax(axis=1), dtype=np.int64)

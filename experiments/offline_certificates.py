"""Hybrid and sequential NP grids from saved sampling statistics."""

from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from scipy.stats import norm as _norm
from hybrid_smoothing.certif_hybrid import (
    certificate_bound_pair,
    hybrid_certificate_grid,
    hybrid_certificate_discrete_grid,
)
from hybrid_smoothing.utils_hybrid import clopper_pearson_lcb, clopper_pearson_ucb


def confidence_bounds(
    n_best: int,
    n_second: int,
    n: int,
    num_classes: int,
    alpha: float = 1e-3,
) -> Tuple[float, float]:
    n = int(n)
    nb, ns = int(n_best), int(n_second)
    alpha_adj = float(alpha) / float(max(1, int(num_classes)))
    return (
        float(clopper_pearson_lcb(n, nb, alpha_adj)),
        float(clopper_pearson_ucb(n, ns, alpha_adj)),
    )


def _gauss_lower(p: np.ndarray, eps: np.ndarray, sigma: float) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return _norm.cdf(_norm.ppf(p) - eps / float(sigma))


def _gauss_upper(p: np.ndarray, eps: np.ndarray, sigma: float) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return _norm.cdf(_norm.ppf(p) + eps / float(sigma))


def node_grids(
    *,
    pA_lower: float,
    pB_upper: float,
    m1: int,
    m0: int,
    sigma: float,
    p_minus: float,
    p_plus: float,
    d_list: Sequence[int],
    eps_values: Sequence[float],
    variants: Sequence[str] = ("hybrid", "seq_topo_feat", "seq_feat_topo", "seq_max"),
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {variant: {"lower": [n_d, n_eps], "upper": [...]}} for one node."""
    eps_values = np.asarray(eps_values, dtype=np.float64)
    d_list = [int(d) for d in d_list]
    max_d = int(max(d_list)) if d_list else 0
    select = np.asarray(d_list, dtype=np.int64)
    out: Dict[str, Dict[str, np.ndarray]] = {}

    need_disc = any(v in variants for v in ("seq_topo_feat", "seq_feat_topo", "seq_max"))

    if "hybrid" in variants:
        gl, gu = certificate_bound_pair(
            hybrid_certificate_grid,
            pA_lower,
            pB_upper,
            sigma=sigma,
            pf_plus=p_plus,
            pf_minus=p_minus,
            eps_values=eps_values,
            max_d=max_d,
            m1=m1,
            m0=m0,
            progress_bar=False,
        )
        out["hybrid"] = {"lower": gl[0, select, :], "upper": gu[0, select, :]}

    if need_disc:
        dl, du = certificate_bound_pair(
            hybrid_certificate_discrete_grid,
            pA_lower,
            pB_upper,
            pf_plus=p_plus,
            pf_minus=p_minus,
            max_d=max_d,
            m1=m1,
            m0=m0,
            progress_bar=False,
        )
        disc_l = dl[0, select]  # [n_d]
        disc_u = du[0, select]

        # order 1: topology first, then Gaussian
        if any(v in variants for v in ("seq_topo_feat", "seq_max")):
            lo = _gauss_lower(disc_l[:, None], eps_values[None, :], sigma)
            up = _gauss_upper(disc_u[:, None], eps_values[None, :], sigma)
            out["seq_topo_feat"] = {"lower": lo, "upper": up}

        # order 2: Gaussian first, then the discrete NP slice on the degraded bound
        if any(v in variants for v in ("seq_feat_topo", "seq_max")):
            pa_eps = _gauss_lower(np.full(eps_values.shape, pA_lower), eps_values, sigma)
            pb_eps = _gauss_upper(np.full(eps_values.shape, pB_upper), eps_values, sigma)
            gl2, gu2 = certificate_bound_pair(
                hybrid_certificate_discrete_grid,
                pa_eps,
                pb_eps,
                pf_plus=p_plus,
                pf_minus=p_minus,
                max_d=max_d,
                m1=m1,
                m0=m0,
                progress_bar=False,
            )
            out["seq_feat_topo"] = {
                "lower": gl2[:, select].T,  # [n_d, n_eps]
                "upper": gu2[:, select].T,
            }

        if "seq_max" in variants:
            a, b = out["seq_topo_feat"], out["seq_feat_topo"]
            # Both bounds are sound; take the best (largest lower / smallest upper).
            out["seq_max"] = {
                "lower": np.maximum(a["lower"], b["lower"]),
                "upper": np.minimum(a["upper"], b["upper"]),
            }

    return out


def restricted_grids(
    *,
    pA_lower: float,
    pB_upper: float,
    m1: int,
    m0: int,
    sigma: float,
    p_minus: float,
    p_plus: float,
    d_list: Sequence[int],
    eps_values: Sequence[float],
    max_d_plus: Optional[int] = 0,
    max_d_minus: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Hybrid NP grid under a *restricted* topology threat model.

    The shipped grid takes the envelope over every split of the budget d into
    (deletions, insertions). Some baselines only certify one direction: a
    sparse-smoothing channel with `pf_plus_adj = 0` cannot certify insertions at
    all, so comparing against the full both-directions envelope would be unfair.

    `max_d_plus = 0` restricts the adversary to deletions, `max_d_minus = 0` to
    insertions; `None` leaves that direction unrestricted.
    """
    from hybrid_smoothing.utils_hybrid import (
        _U_sigma_batch,
        _V_sigma_batch,
        _safe_log_gamma,
        grouped_edge_flip,
        solve_log_s_star_batch,
        solve_log_t_star_batch,
    )

    eps_values = np.asarray(eps_values, dtype=np.float64)
    d_list = [int(d) for d in d_list]
    n_eps = int(eps_values.size)
    lower = np.zeros((len(d_list), n_eps), dtype=np.float64)
    upper = np.zeros((len(d_list), n_eps), dtype=np.float64)
    eps_eval = np.where(eps_values > 1e-12, eps_values, 1e-8)

    for i, d in enumerate(d_list):
        cap_m = int(min(d, m1 if max_d_minus is None else min(m1, max_d_minus)))
        best_lo = np.ones(n_eps)
        best_up = np.zeros(n_eps)
        for dm in range(cap_m + 1):
            dp_cap = d - dm
            if max_d_plus is not None:
                dp_cap = min(dp_cap, int(max_d_plus))
            dp_cap = int(min(dp_cap, m0))
            for dp in range(dp_cap + 1):
                if dm + dp == 0:
                    lo = _gauss_lower(np.full(n_eps, pA_lower), eps_values, sigma)
                    up = _gauss_upper(np.full(n_eps, pB_upper), eps_values, sigma)
                else:
                    w_cln, w_adv, gamma = grouped_edge_flip(dm, dp, p_minus, p_plus)
                    lg = _safe_log_gamma(gamma)
                    log_t = solve_log_t_star_batch(pA_lower, eps_eval, sigma, w_cln, lg)
                    lo = _V_sigma_batch(log_t, eps_eval, sigma, w_adv, lg)
                    log_s = solve_log_s_star_batch(pB_upper, eps_eval, sigma, w_cln, lg)
                    up = _U_sigma_batch(log_s, eps_eval, sigma, w_adv, lg)
                best_lo = np.minimum(best_lo, lo)
                best_up = np.maximum(best_up, up)
        lower[i] = best_lo
        upper[i] = best_up
    return {"lower": lower, "upper": upper}

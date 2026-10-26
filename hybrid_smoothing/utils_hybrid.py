"""Edge-flip likelihood ratios, exact confidence bounds and vectorized hybrid NP primitives."""

import math
import numpy as np
from scipy.stats import norm as _norm, binom as _binom, beta as _beta_dist

_Phi = _norm.cdf


# ============================================================
# Clopper–Pearson  (implemented from scipy.stats.beta)
# ============================================================

def clopper_pearson_lcb(n: int, nA: int, alpha: float) -> float:
    """
    One-sided lower confidence bound on Binomial parameter p.
    P(p >= lcb) >= 1 - alpha.

    Uses the exact Clopper-Pearson method:
      lcb = Beta.ppf(alpha, nA, n - nA + 1)
    """
    n, nA = int(n), int(nA)
    if n <= 0:
        return float("nan")
    if nA == 0:
        return 0.0
    return float(_beta_dist.ppf(alpha, nA, n - nA + 1))


def clopper_pearson_ucb(n: int, nA: int, alpha: float) -> float:
    """
    One-sided upper confidence bound on Binomial parameter p.
    P(p <= ucb) >= 1 - alpha.

    Uses: ucb = Beta.ppf(1 - alpha, nA + 1, n - nA)
    """
    n, nA = int(n), int(nA)
    if n <= 0:
        return float("nan")
    if nA == n:
        return 1.0
    return float(_beta_dist.ppf(1.0 - alpha, nA + 1, n - nA))


# ============================================================
# Discrete kernels  ->  (w_cln, w_adv, gamma) float64 arrays
# ============================================================
# Convention:
#   w_cln[k] = P(atom k | clean)        sums to 1
#   w_adv[k] = P(atom k | adversary)    sums to 1
#   gamma[k] = likelihood ratio value    = w_adv[k] / w_cln[k]
#
# For gamma=0 atoms: w_adv[k]=0 by absolute continuity.

def grouped_edge_flip(d_minus, d_plus, p_minus, p_plus):
    """
    Homogeneous i.i.d. per-edge corruption on candidate set S.

    Each (i,j) in S independently:
      - existing edge: deleted w.p. p_minus
      - non-edge:      added   w.p. p_plus

    Adversary: d_minus deletions, d_plus additions.

    log gamma_1 = (alpha - beta) * K + d_minus * beta - d_plus * alpha
    where
      alpha = log((1 - p_plus) / p_minus)
      beta  = log(p_plus / (1 - p_minus))
      K = K_- + K_+
      Clean:  K_- ~ Bin(d_-, p_-),   K_+ ~ Bin(d_+, p_+)
      Adv:    K_- ~ Bin(d_-, 1-p_+), K_+ ~ Bin(d_+, 1-p_-)

    Returns (w_cln, w_adv, gamma) each of length d_- + d_+ + 1.
    """
    assert 0 < p_minus < 1 and 0 < p_plus < 1
    assert d_minus >= 0 and d_plus >= 0

    alpha = math.log((1.0 - p_plus) / p_minus)
    beta_ = math.log(p_plus / (1.0 - p_minus))
    max_K = d_minus + d_plus

    w_cln = np.zeros(max_K + 1)
    w_adv = np.zeros(max_K + 1)
    gamma = np.zeros(max_K + 1)

    for k in range(max_K + 1):
        lam_k = (alpha - beta_) * k + d_minus * beta_ - d_plus * alpha
        gamma[k] = math.exp(lam_k)

        j_lo = max(0, k - d_plus)
        j_hi = min(k, d_minus)
        wc, wa = 0.0, 0.0
        for j in range(j_lo, j_hi + 1):
            wc += (_binom.pmf(j, d_minus, p_minus)
                   * _binom.pmf(k - j, d_plus, p_plus))
            wa += (_binom.pmf(j, d_minus, 1.0 - p_plus)
                   * _binom.pmf(k - j, d_plus, 1.0 - p_minus))
        w_cln[k] = wc
        w_adv[k] = wa


    mask = w_cln > 0
    rel = np.max(np.abs(np.log(w_adv[mask]) - np.log(w_cln[mask]) - np.log(gamma[mask])))
    assert rel < 1e-10, f"LR mismatch: max log-error {rel}"

    return w_cln, w_adv, gamma


# ============================================================
# Safe log-gamma
# ============================================================

def _safe_log_gamma(gamma):
    """log(gamma), mapping gamma=0 -> -1e15 (avoids -inf in arithmetic)."""
    with np.errstate(divide="ignore"):
        lg = np.log(np.asarray(gamma, dtype=np.float64))
    lg[np.isneginf(lg)] = -1e15
    return lg


def _fractional_knapsack_adv_mass(w_cln, w_adv, p, order):
    """
    Given atom masses under clean/adv and a clean-mass budget p in [0,1],
    solve fractional knapsack exactly:
      choose x_k in [0,1], sum x_k w_cln[k] = p,
      optimize sum x_k w_adv[k].
    """
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0:
        return 0.0
    w_cln = np.asarray(w_cln, dtype=np.float64)
    w_adv = np.asarray(w_adv, dtype=np.float64)

    remain = p
    total = 0.0
    for k in order:
        c = float(w_cln[k])
        if c <= 0:
            continue
        if c <= remain + 1e-18:
            total += float(w_adv[k])
            remain -= c
            if remain <= 1e-18:
                break
        else:
            frac = remain / c
            total += frac * float(w_adv[k])
            remain = 0.0
            break
    return float(np.clip(total, 0.0, 1.0))


def compute_V_discrete_at_counts(p, d_minus, d_plus, p_minus, p_plus):
    """
    Exact discrete NP slice at epsilon=0 (lower bound): minimize adversarial mass
    at fixed clean mass p by ordering atoms with increasing likelihood ratio gamma.
    """
    if d_minus + d_plus == 0:
        return float(np.clip(p, 0.0, 1.0))
    w_cln, w_adv, gamma = grouped_edge_flip(d_minus, d_plus, p_minus, p_plus)
    order = np.argsort(gamma, kind='mergesort')  # increasing gamma for lower envelope
    return _fractional_knapsack_adv_mass(w_cln, w_adv, p, order)


def compute_U_discrete_at_counts(p, d_minus, d_plus, p_minus, p_plus):
    """
    Exact discrete NP slice at epsilon=0 (upper bound): maximize adversarial mass
    at fixed clean mass p by ordering atoms with decreasing likelihood ratio gamma.
    """
    if d_minus + d_plus == 0:
        return float(np.clip(p, 0.0, 1.0))
    w_cln, w_adv, gamma = grouped_edge_flip(d_minus, d_plus, p_minus, p_plus)
    order = np.argsort(-gamma, kind='mergesort')  # decreasing gamma for upper envelope
    return _fractional_knapsack_adv_mass(w_cln, w_adv, p, order)


# ============================================================
# NP primitives  (ALL in log-space: log_t = log(t), log_gamma = log(gamma))
# ============================================================
#
# Phi argument for F:
#   a_k = r/(2*sigma) + sigma*(log_t - log_gamma_k)/r
#
# Phi argument for V:
#   a_k = r/(2*sigma) + sigma*(log_t - log_gamma_k)/r  -  r/sigma
#
# Phi_bar = 1 - Phi  for the reverse (multiclass upper bound).


# Vectorized capacity maps and bisection across feature radii.

def _F_sigma_batch(log_t_vec, r_vec, sigma, w_cln, log_gamma):
    """
    Vectorized F_sigma: evaluate at multiple (log_t, r) pairs simultaneously.

    log_t_vec : array [M]
    r_vec     : array [M]  (the eps values)
    w_cln     : array [K]  (kernel atoms)
    log_gamma : array [K]

    Returns array [M].
    """
    # args shape: [M, K]
    r_col = r_vec[:, None]           # [M, 1]
    lt_col = log_t_vec[:, None]      # [M, 1]
    lg_row = log_gamma[None, :]      # [1, K]

    a = r_col / (2.0 * sigma) + sigma * (lt_col - lg_row) / r_col  # [M, K]
    phi_vals = _Phi(a)                                               # [M, K]
    return phi_vals @ w_cln                                          # [M]


def _V_sigma_batch(log_t_vec, r_vec, sigma, w_adv, log_gamma):
    """Vectorized V_sigma."""
    r_col = r_vec[:, None]
    lt_col = log_t_vec[:, None]
    lg_row = log_gamma[None, :]

    a = r_col / (2.0 * sigma) + sigma * (lt_col - lg_row) / r_col - r_col / sigma
    phi_vals = _Phi(a)
    return phi_vals @ w_adv


def _U_sigma_batch(log_s_vec, r_vec, sigma, w_adv, log_gamma):
    """Vectorized U_sigma (reverse NP value)."""
    r_col = r_vec[:, None]
    ls_col = log_s_vec[:, None]
    lg_row = log_gamma[None, :]

    a = r_col / (2.0 * sigma) + sigma * (ls_col - lg_row) / r_col - r_col / sigma
    phi_bar_vals = 1.0 - _Phi(a)
    return phi_bar_vals @ w_adv


def _Fbar_sigma_batch(log_s_vec, r_vec, sigma, w_cln, log_gamma):
    """Vectorized F_bar_sigma."""
    r_col = r_vec[:, None]
    ls_col = log_s_vec[:, None]
    lg_row = log_gamma[None, :]

    a = r_col / (2.0 * sigma) + sigma * (ls_col - lg_row) / r_col
    phi_bar_vals = 1.0 - _Phi(a)
    return phi_bar_vals @ w_cln


def solve_log_t_star_batch(pA, r_vec, sigma, w_cln, log_gamma,
                           tol=1e-10, max_iter=120):
    """
    Vectorized bisection: solve F(exp(u); r_j) = pA for all r_j simultaneously.

    r_vec     : array [M] of eps values (must be > 0)
    Returns   : array [M] of log(t*) values
    """
    r_vec = np.asarray(r_vec, dtype=np.float64)
    M = len(r_vec)
    lo = np.full(M, -100.0)
    hi = np.full(M, 100.0)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        converged = (hi - lo) < tol
        if converged.all():
            break
        f_vals = _F_sigma_batch(mid, r_vec, sigma, w_cln, log_gamma)
        go_left = f_vals >= pA
        hi = np.where(go_left & ~converged, mid, hi)
        lo = np.where(~go_left & ~converged, mid, lo)

    return 0.5 * (lo + hi)


def solve_log_s_star_batch(pB, r_vec, sigma, w_cln, log_gamma,
                           tol=1e-10, max_iter=120):
    """
    Vectorized bisection for the reverse: solve F_bar(exp(u); r_j) = pB.
    F_bar is decreasing in u.
    """
    r_vec = np.asarray(r_vec, dtype=np.float64)
    M = len(r_vec)
    lo = np.full(M, -100.0)
    hi = np.full(M, 100.0)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        converged = (hi - lo) < tol
        if converged.all():
            break
        f_vals = _Fbar_sigma_batch(mid, r_vec, sigma, w_cln, log_gamma)
        go_right = f_vals >= pB    # F_bar decreasing: if too large, move right
        lo = np.where(go_right & ~converged, mid, lo)
        hi = np.where(~go_right & ~converged, mid, hi)

    return 0.5 * (lo + hi)

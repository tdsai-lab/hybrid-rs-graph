"""Hybrid and sequential NP bounds over edge budgets and feature radii."""

import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm

from .utils_hybrid import (
    _safe_log_gamma,
    _U_sigma_batch,
    _V_sigma_batch,
    compute_U_discrete_at_counts,
    compute_V_discrete_at_counts,
    grouped_edge_flip,
    solve_log_s_star_batch,
    solve_log_t_star_batch,
)


def _edge_counts(budget, m1, m0):
    """Feasible (deletions, insertions) with total at most budget."""
    deletions = budget if m1 is None else min(budget, m1)
    for deleted in range(deletions + 1):
        additions = budget - deleted if m0 is None else min(budget - deleted, m0)
        for added in range(additions + 1):
            yield deleted, added


def certificate_bound_pair(grid_function, lower, upper, **parameters):
    """Apply a grid to selected-class LCBs and competing-class UCBs."""
    return tuple(
        grid_function(p_emps=np.atleast_1d(probabilities), reverse=reverse, **parameters)[0]
        for probabilities, reverse in ((lower, False), (upper, True))
    )


def hybrid_certificate_grid(
    p_emps,
    sigma,
    pf_plus,
    pf_minus,
    eps_values,
    max_d,
    m1=None,
    m0=None,
    reverse=False,
    progress_bar=True,
):
    """Return bounds of shape [nodes, max_d + 1, radii] and max_d.

    Lower bounds minimize over feasible edge counts; upper bounds maximize.
    Positive radii share vectorized log-space bisection. The existing numerical
    topology limit uses radius 1e-8 at d>0; d=0 keeps zero-radius values exact.
    """
    probabilities = np.asarray(p_emps, dtype=np.float64)
    radii = np.asarray(eps_values, dtype=np.float64)
    positive = radii > 1e-12
    evaluation_radii = np.where(positive, radii, 1e-8)
    grid = np.zeros((len(probabilities), max_d + 1, len(radii)))
    combine = np.maximum if reverse else np.minimum
    solve = solve_log_s_star_batch if reverse else solve_log_t_star_batch
    evaluate = _U_sigma_batch if reverse else _V_sigma_batch
    direction = 1 if reverse else -1

    kernels = {}
    for deleted, added in _edge_counts(max_d, m1, m0):
        if deleted + added:
            clean, adversarial, ratio = grouped_edge_flip(deleted, added, pf_minus, pf_plus)
            kernels[deleted, added] = clean, adversarial, _safe_log_gamma(ratio)

    for node in tqdm(range(len(probabilities)), desc="nodes", disable=not progress_bar):
        probability = probabilities[node]
        if probability <= 0:
            continue
        quantile = norm.ppf(probability)
        grid[node, 0, ~positive] = probability
        grid[node, 0, positive] = norm.cdf(quantile + direction * radii[positive] / sigma)
        gaussian = norm.cdf(quantile + direction * evaluation_radii / sigma)

        for budget in range(1, max_d + 1):
            best = np.zeros(len(radii)) if reverse else np.ones(len(radii))
            for deleted, added in _edge_counts(budget, m1, m0):
                if deleted + added == 0:
                    values = gaussian
                else:
                    clean, adversarial, log_ratio = kernels[deleted, added]
                    threshold = solve(probability, evaluation_radii, sigma, clean, log_ratio)
                    values = evaluate(threshold, evaluation_radii, sigma, adversarial, log_ratio)
                best = combine(best, values)
            grid[node, budget] = best
    return grid, max_d


def hybrid_certificate_discrete_grid(
    p_emps,
    pf_plus,
    pf_minus,
    max_d=None,
    m1=None,
    m0=None,
    reverse=False,
    progress_bar=True,
):
    """Return exact epsilon=0 NP bounds of shape [nodes, max_d + 1]."""
    probabilities = np.asarray(p_emps, dtype=np.float64)
    if max_d is None:
        max_d = int(m1 + m0) if m1 is not None and m0 is not None else 20
    max_d = int(max_d)
    grid = np.zeros((len(probabilities), max_d + 1), dtype=np.float64)
    combine = max if reverse else min
    evaluate = compute_U_discrete_at_counts if reverse else compute_V_discrete_at_counts

    for node in tqdm(range(len(probabilities)), desc="nodes", disable=not progress_bar):
        probability = float(np.clip(probabilities[node], 0, 1))
        grid[node, 0] = probability
        for budget in range(1, max_d + 1):
            best = 0.0 if reverse else 1.0
            for deleted, added in _edge_counts(budget, m1, m0):
                value = (
                    probability
                    if deleted + added == 0
                    else evaluate(probability, deleted, added, pf_minus, pf_plus)
                )
                best = combine(best, value)
            grid[node, budget] = float(np.clip(best, 0, 1))
    return grid, max_d


def sequential_np_certificate_from_discrete_bounds(
    p_lower_disc_grid,
    p_upper_disc_grid,
    eps_values,
    sigma,
):
    """Compose topology bounds with Gaussian bounds; preserve epsilon=0 exactly."""
    lower = np.asarray(p_lower_disc_grid, dtype=np.float64)
    upper = np.asarray(p_upper_disc_grid, dtype=np.float64)
    radii = np.asarray(eps_values, dtype=np.float64)
    if lower.shape != upper.shape:
        raise ValueError("lower/upper discrete grids must have same shape")
    if lower.ndim != 2:
        raise ValueError("expected discrete grids of shape [n_nodes, max_d+1]")

    positive = radii > 1e-12
    shifted = []
    for bounds, direction in ((lower, -1), (upper, 1)):
        grid = np.repeat(bounds[:, :, None], len(radii), axis=2)
        quantile = norm.ppf(np.clip(bounds[:, :, None], 1e-12, 1 - 1e-12))
        grid[:, :, positive] = norm.cdf(
            quantile + direction * radii[positive][None, None, :] / float(sigma)
        )
        shifted.append(grid)
    lower, upper = shifted
    return lower, upper, lower > 0.5, lower > upper

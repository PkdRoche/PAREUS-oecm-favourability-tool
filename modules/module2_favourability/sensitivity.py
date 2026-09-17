"""Monte Carlo sensitivity analysis for MCE weight uncertainty.

Runs inter-group aggregation N times with weights sampled from Dirichlet
distributions centred on the user-specified base values.

Key design: this module works on **pre-computed group scores** (A, B, C)
returned by mce_engine.compute_favourability — NOT on individual criterion
arrays.  Perturbing only inter-group weights is sufficient to capture the
dominant uncertainty, avoids re-running the full criterion normalisation /
land-use recoding pipeline, and guarantees consistency with the main MCE code
path.  Intra-group weight perturbation can optionally be enabled via
``perturb_intra``, which falls back to perturbing the inter-group alphas with
a secondary Dirichlet sample (since the group scores are already aggregated,
intra-group re-aggregation would require the raw criterion arrays — not stored
here; instead we model their combined uncertainty through the concentration
parameter).

The concentration parameter controls how tightly samples cluster around the
base weights:
  - concentration =  5  → wide spread, high uncertainty
  - concentration = 20  → moderate spread (default)
  - concentration = 100 → tight spread, low uncertainty

Returns
-------
stability_map : float32 array [0, 1]
    Fraction of runs where pixel score is >= threshold.
    1.0 = always above, 0.0 = never above, NaN = eliminated pixel.
std_map : float32 array
    Standard deviation of scores across runs (NaN for eliminated pixels).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dirichlet weight sampling
# ---------------------------------------------------------------------------

def _sample_dirichlet(
    base: np.ndarray,
    concentration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a weight vector from Dirichlet(base * concentration).

    Ensures non-zero alpha values and that the result sums to 1.
    """
    alpha = np.maximum(base * concentration, 0.1)
    return rng.dirichlet(alpha).astype(np.float32)


# ---------------------------------------------------------------------------
# Fast inter-group geometric mean (mirrors mce_engine logic exactly)
# ---------------------------------------------------------------------------

_EPS = np.float32(1e-9)


def _inter_group_geometric_mean(
    score_a: np.ndarray,
    score_b: np.ndarray,
    score_c: np.ndarray,
    w: np.ndarray,          # shape (3,)  already sums to 1
) -> np.ndarray:
    """Weighted geometric mean of three group score arrays.

    Matches the aggregation in mce_engine.compute_favourability exactly:
    - log-space summation
    - NaN propagation for eliminated pixels
    """
    arrays = [score_a, score_b, score_c]
    log_sum = np.zeros(score_a.shape, dtype=np.float32)
    nan_mask = np.zeros(score_a.shape, dtype=bool)

    for arr, wi in zip(arrays, w):
        safe = np.where(np.isnan(arr), _EPS, np.maximum(arr, _EPS))
        log_sum += wi * np.log(safe)
        nan_mask |= np.isnan(arr)

    result = np.exp(log_sum)
    result[nan_mask] = np.nan
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sensitivity(
    group_scores: dict[str, np.ndarray],
    base_weights: dict,
    eliminatory_mask: np.ndarray,
    threshold: float,
    n_runs: int = 200,
    concentration: float = 20.0,
    perturb_intra: bool = True,
    seed: Optional[int] = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo inter-group weight sensitivity analysis.

    Parameters
    ----------
    group_scores : dict
        Pre-computed group score arrays from mce_engine, keyed 'A', 'B', 'C'.
        All arrays must have the same shape as ``eliminatory_mask``.
    base_weights : dict
        Weight dict with key ``inter_group_weights`` → {W_A, W_B, W_C}.
        (group_a_weights / group_c_weights are accepted but ignored because
        the group scores are already aggregated.)
    eliminatory_mask : np.ndarray
        Boolean mask: True = pixel survived Group D (eligible).
    threshold : float
        Score threshold for stability computation.
    n_runs : int
        Number of Monte Carlo iterations.
    concentration : float
        Dirichlet concentration parameter (higher → tighter spread around base).
    perturb_intra : bool
        If True, use a slightly lower concentration to simulate combined intra +
        inter uncertainty.  If False, only inter-group weights are perturbed.
    seed : int or None
        Random seed for reproducibility.
    progress_callback : callable(current_run: int, total_runs: int) or None
        Called after each run; use for UI progress bars.

    Returns
    -------
    stability_map : np.ndarray float32
        Fraction of runs where pixel score >= threshold. Shape = raster shape.
    std_map : np.ndarray float32
        Standard deviation of pixel scores across runs. Shape = raster shape.
    """
    rng   = np.random.default_rng(seed)
    shape = eliminatory_mask.shape

    # ── Extract group scores ─────────────────────────────────────────────────
    score_a = group_scores.get('A')
    score_b = group_scores.get('B')
    score_c = group_scores.get('C')

    if score_a is None or score_b is None or score_c is None:
        raise ValueError(
            "group_scores must contain keys 'A', 'B', 'C'. "
            f"Got: {list(group_scores.keys())}"
        )

    # ── Base inter-group weight vector ───────────────────────────────────────
    iw = base_weights['inter_group_weights']
    base_inter = np.array(
        [iw['W_A'], iw['W_B'], iw['W_C']], dtype=np.float32
    )
    base_inter /= base_inter.sum()   # normalise (safety)

    # When perturb_intra is True we effectively model combined uncertainty by
    # slightly widening the inter-group Dirichlet (lower concentration).
    effective_concentration = concentration * (0.7 if perturb_intra else 1.0)

    # ── Accumulators ─────────────────────────────────────────────────────────
    above_count = np.zeros(shape, dtype=np.float32)
    score_sum   = np.zeros(shape, dtype=np.float32)
    score_sq    = np.zeros(shape, dtype=np.float32)
    n_valid_per_pixel = np.zeros(shape, dtype=np.int32)

    logger.info(
        "Running sensitivity analysis: %d runs, concentration=%.1f, "
        "perturb_intra=%s, effective_concentration=%.1f",
        n_runs, concentration, perturb_intra, effective_concentration
    )

    for i in range(n_runs):
        # ── Sample inter-group weights ────────────────────────────────────
        w_inter = _sample_dirichlet(base_inter, effective_concentration, rng)

        # ── Inter-group geometric mean ────────────────────────────────────
        run_score = _inter_group_geometric_mean(score_a, score_b, score_c, w_inter)

        # ── Apply eliminatory mask ────────────────────────────────────────
        run_score = np.where(eliminatory_mask, run_score, np.nan)

        # ── Accumulate ───────────────────────────────────────────────────
        valid = ~np.isnan(run_score)
        above_count[valid & (run_score >= threshold)] += 1.0
        score_sum[valid]        += run_score[valid]
        score_sq[valid]         += run_score[valid] ** 2
        n_valid_per_pixel[valid] += 1

        if progress_callback is not None:
            progress_callback(i + 1, n_runs)

    # ── Stability map ─────────────────────────────────────────────────────────
    stability_map = np.where(
        eliminatory_mask, above_count / n_runs, np.nan
    ).astype(np.float32)

    # ── Std-dev map ───────────────────────────────────────────────────────────
    # Use per-pixel valid run count to avoid bias from masked pixels
    n_runs_f = np.where(n_valid_per_pixel > 0, n_valid_per_pixel, 1).astype(np.float32)
    mean_scores = np.where(eliminatory_mask, score_sum / n_runs_f, np.nan)
    variance    = np.where(
        eliminatory_mask,
        score_sq / n_runs_f - mean_scores ** 2,
        np.nan
    )
    std_map = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    n_stable   = int(np.nansum(stability_map >= 0.8))
    n_ambig    = int(np.nansum((stability_map >= 0.4) & (stability_map < 0.6)))
    n_unstable = int(np.nansum(stability_map < 0.5))
    n_eligible = int(np.sum(eliminatory_mask))

    logger.info(
        "Sensitivity complete: %d eligible pixels — "
        "%d stable (≥80%%), %d ambiguous (40-60%%), %d unstable (<50%%)",
        n_eligible, n_stable, n_ambig, n_unstable
    )

    return stability_map, std_map

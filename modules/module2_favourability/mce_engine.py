"""Multi-criteria evaluation engine.

This module implements the core MCE aggregation functions for OECM favourability
analysis. Only weighted geometric mean and Yager OWA aggregation methods are
supported. Weighted Linear Combination (WLC) is explicitly forbidden per
SPECIFICATIONS.md section 4.4.

Functions
---------
weighted_geometric_mean
    Compute weighted geometric mean of criteria arrays.
yager_owa
    Compute Yager OWA aggregation with configurable orness parameter.
compute_favourability
    Full MCE pipeline producing favourability scores and masks.
"""

import functools
import logging
import numpy as np
import yaml
from pathlib import Path
from typing import Optional

from . import criteria_manager
from . import raster_preprocessing

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named inter-group weight scenarios for stakeholder-facing comparison
# ("virtual laboratory" exploration of trade-offs between ecological
# integrity, co-benefits and production function). Only W_A/W_B/W_C vary
# between presets — intra-group weights are left at whatever the analyst has
# set, so a preset changes the trade-off axis without requiring the full
# weight set to be re-specified.
# ---------------------------------------------------------------------------
SCENARIO_PRESETS: dict[str, dict[str, float]] = {
    'Biodiversity priority': {'W_A': 0.70, 'W_B': 0.10, 'W_C': 0.20},
    'Services priority':     {'W_A': 0.25, 'W_B': 0.25, 'W_C': 0.50},
    'Compromise':            {'W_A': 0.45, 'W_B': 0.20, 'W_C': 0.35},
}


def weighted_geometric_mean(
    arrays: list[np.ndarray],
    weights: list[float]
) -> np.ndarray:
    """Compute weighted geometric mean of criteria arrays.

    Formula: S = prod(array_i ^ w_i) for all i, where sum(w_i) = 1.

    Computation is performed in log-space to avoid numerical underflow:
    S = exp(sum(w_i * log(array_i)))

    Parameters
    ----------
    arrays : list[np.ndarray]
        List of numpy arrays containing criteria values. All arrays must have
        the same shape. Values should be in [0, 1].
    weights : list[float]
        List of weights corresponding to each array. Weights must sum to 1.0.

    Returns
    -------
    np.ndarray
        Aggregated score array with values in [0, 1]. Same shape as input arrays.

    Raises
    ------
    ValueError
        If weights do not sum to 1.0 (tolerance 1e-6).
        If number of arrays does not match number of weights.
        If arrays have inconsistent shapes.

    Notes
    -----
    - NaN propagation: if any input criterion is NaN, output is NaN.
    - Zero handling: exact 0 inputs are floored to 1e-9 before log-space
      computation, driving the output very close to 0 without nullifying it.
      Hard pixel elimination is handled upstream by the Group D mask.
    - The function is strongly non-compensatory: a near-zero criterion pulls
      the total score toward 0, but only Group D eliminates pixels entirely.

    Examples
    --------
    >>> import numpy as np
    >>> arrays = [np.array([0.8, 0.6]), np.array([0.5, 0.9])]
    >>> weights = [0.6, 0.4]
    >>> result = weighted_geometric_mean(arrays, weights)
    >>> # result[0] = 0.8^0.6 * 0.5^0.4 = 0.6822...
    >>> # result[1] = 0.6^0.6 * 0.9^0.4 = 0.7192...
    """
    logger.info(f"Computing weighted geometric mean for {len(arrays)} criteria")

    # Validate inputs
    if len(arrays) != len(weights):
        raise ValueError(
            f"Number of arrays ({len(arrays)}) must match number of weights ({len(weights)})"
        )

    if len(arrays) == 0:
        raise ValueError("At least one array and weight must be provided")

    # Check weights sum to 1.0
    weight_sum = sum(weights)
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"Weights must sum to 1.0, got {weight_sum:.6f}"
        )

    # Check all arrays have same shape
    reference_shape = arrays[0].shape
    for i, arr in enumerate(arrays[1:], start=1):
        if arr.shape != reference_shape:
            raise ValueError(
                f"Array {i} shape {arr.shape} does not match reference shape {reference_shape}"
            )

    # Use float32 — sufficient precision for [0,1] scores, half the memory of float64
    arrays_float = [arr.astype(np.float32) for arr in arrays]
    weights_arr = np.array(weights, dtype=np.float32)

    EPS = np.float32(1e-9)

    # Build NaN mask once, then compute log-sum in a single accumulator (no per-array temporaries)
    nan_mask = np.zeros(reference_shape, dtype=bool)
    for arr in arrays_float:
        nan_mask |= np.isnan(arr)

    valid_mask = ~nan_mask

    # Compute weighted geometric mean in log-space
    result = np.full(reference_shape, np.nan, dtype=np.float32)
    if np.any(valid_mask):
        log_sum = np.zeros(reference_shape, dtype=np.float32)
        for arr, w in zip(arrays_float, weights_arr):
            # Floor to EPS only where valid; elsewhere leave as-is (won't enter result)
            np.maximum(arr, EPS, out=arr)
            np.log(arr, out=arr)
            arr *= w
            log_sum += arr
        np.exp(log_sum, out=log_sum)
        np.clip(log_sum, 0.0, 1.0, out=log_sum)
        result[valid_mask] = log_sum[valid_mask]

    logger.info(f"Geometric mean complete. Range: [{np.nanmin(result):.4f}, {np.nanmax(result):.4f}]")
    return result


def yager_owa(
    arrays: list[np.ndarray],
    weights: list[float],
    alpha: float
) -> np.ndarray:
    """Compute Yager OWA aggregation with orness parameter.

    Ordered Weighted Averaging (OWA) provides a family of aggregation operators
    ranging from AND logic (minimum) to OR logic (maximum) controlled by the
    orness parameter alpha.

    Parameters
    ----------
    arrays : list[np.ndarray]
        List of numpy arrays containing criteria values. All arrays must have
        the same shape. Values should be in [0, 1].
    weights : list[float]
        List of criterion importance weights. These represent the relative
        importance of each criterion, NOT the OWA position weights.
        Weights must sum to 1.0.
    alpha : float
        Orness parameter in [0, 1]:
        - alpha = 0: pure AND logic (minimum value)
        - alpha = 0.5: balanced partial compensation
        - alpha = 1: pure OR logic (maximum value)
        Recommended default: 0.25 (near AND logic, conservative)

    Returns
    -------
    np.ndarray
        Aggregated score array with values in [0, 1]. Same shape as input arrays.

    Raises
    ------
    ValueError
        If alpha is not in [0, 1].
        If weights do not sum to 1.0 (tolerance 1e-6).
        If number of arrays does not match number of weights.
        If arrays have inconsistent shapes.

    Notes
    -----
    The algorithm follows the specification for combining criterion importance
    weights with OWA position weights:

    1. Stack raw criterion values and sort per pixel in descending order -> b_j.
    2. Compute OWA position weights v_j from alpha using Yager's formula:
       v_j = (j/n)^(1-alpha) - ((j-1)/n)^(1-alpha)
       where n is the number of criteria and j = 1, ..., n.
    3. Compute final score: S = sum(v_j * b_j)

    Implements Weighted OWA (WOWA): criterion importance weights and OWA
    position weights are combined so that both the rank order of values AND
    the user-assigned importance of each criterion influence the final score.
    For each pixel, the combined weight at position j is proportional to
    owa_position_weight_j × importance_weight_of_criterion_ranked_j,
    normalised to sum to 1. This ensures W_A/W_B/W_C settings are respected.

    For alpha=0 (AND): result equals the value of the lowest-ranked criterion,
    weighted by its importance. For alpha=1 (OR): result equals the maximum.

    Examples
    --------
    >>> import numpy as np
    >>> arrays = [np.array([0.8]), np.array([0.5]), np.array([0.2])]
    >>> weights = [1/3, 1/3, 1/3]
    >>> # alpha=0 -> minimum = 0.2
    >>> result = yager_owa(arrays, weights, alpha=0.0)
    """
    logger.info(f"Computing Yager OWA for {len(arrays)} criteria with alpha={alpha}")

    # Validate alpha
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

    # Validate inputs
    if len(arrays) != len(weights):
        raise ValueError(
            f"Number of arrays ({len(arrays)}) must match number of weights ({len(weights)})"
        )

    if len(arrays) == 0:
        raise ValueError("At least one array and weight must be provided")

    # Check weights sum to 1.0
    weight_sum = sum(weights)
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"Weights must sum to 1.0, got {weight_sum:.6f}"
        )

    # Check all arrays have same shape
    reference_shape = arrays[0].shape
    for i, arr in enumerate(arrays[1:], start=1):
        if arr.shape != reference_shape:
            raise ValueError(
                f"Array {i} shape {arr.shape} does not match reference shape {reference_shape}"
            )

    n = len(arrays)
    # Use float32 — halves memory vs float64 for (H×W×N) stack
    arrays_float = [arr.astype(np.float32) for arr in arrays]
    weights_arr = np.array(weights, dtype=np.float32)

    # Build NaN mask
    nan_mask = np.zeros(reference_shape, dtype=bool)
    for arr in arrays_float:
        nan_mask |= np.isnan(arr)

    # Step 1: Stack values (float32) and sort per pixel in descending order.
    # np.stack creates (H×W×N) — largest single allocation; float32 halves its size.
    stacked = np.stack(arrays_float, axis=-1)  # shape (..., n), float32
    sort_idx = np.argsort(stacked, axis=-1)[..., ::-1]  # descending
    sorted_values = np.take_along_axis(stacked, sort_idx, axis=-1)
    del stacked  # free immediately — no longer needed

    # Step 2: OWA position weights (Yager's formula)
    if alpha == 0.0:
        owa_weights = np.zeros(n, dtype=np.float32)
        owa_weights[-1] = 1.0   # all weight on minimum
    elif alpha == 1.0:
        owa_weights = np.zeros(n, dtype=np.float32)
        owa_weights[0] = 1.0    # all weight on maximum
    else:
        exponent = 1.0 - alpha
        owa_weights = np.array(
            [(j / n) ** exponent - ((j - 1) / n) ** exponent for j in range(1, n + 1)],
            dtype=np.float32
        )

    logger.debug(f"OWA position weights: {owa_weights}")

    # Step 3: Weighted OWA
    sorted_importance = weights_arr[sort_idx]            # shape (..., n)
    combined = owa_weights * sorted_importance           # element-wise, shape (..., n)
    sum_combined = combined.sum(axis=-1, keepdims=True)
    combined_norm = np.where(sum_combined > 0, combined / sum_combined, owa_weights)
    del combined, sorted_importance  # free before the final sum

    result = np.sum(sorted_values * combined_norm, axis=-1)
    result = np.clip(result, 0.0, 1.0)
    result[nan_mask] = np.nan

    logger.info(f"OWA complete. Range: [{np.nanmin(result):.4f}, {np.nanmax(result):.4f}]")
    return result


@functools.lru_cache(maxsize=1)
def _load_criteria_config() -> dict:
    """Load criteria defaults configuration (cached after first read)."""
    config_path = Path(__file__).parent.parent.parent / "config" / "criteria_defaults.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def _load_transformation_config() -> dict:
    """Load transformation function configuration (cached after first read)."""
    config_path = Path(__file__).parent.parent.parent / "config" / "transformation_functions.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def _load_landuse_config() -> dict:
    """Load land use compatibility configuration (cached after first read)."""
    config_path = Path(__file__).parent.parent.parent / "config" / "land_use_compatibility.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_favourability(
    ecosystem_condition: np.ndarray,
    regulating_es: np.ndarray,
    cultural_es: np.ndarray,
    provisioning_es: np.ndarray,
    anthropogenic_pressure: np.ndarray,
    landuse: np.ndarray,
    weights: dict,
    method: str = "geometric",
    alpha: float = 0.25,
    threshold_pressure: float = 150.0,
    gap_bonus: float = 0.0,
    gap_mask: Optional[np.ndarray] = None,
    percentile_norm: bool = False,
    proximity_bonus: float = 0.0,
    proximity_decay_km: float = 10.0,
    pa_proximity_raster: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Compute full MCE favourability pipeline.

    Implements the complete multi-criteria evaluation for OECM favourability,
    following the strict order: Group D (eliminatory mask) -> Group A -> B -> C.

    Parameters
    ----------
    ecosystem_condition : np.ndarray
        Ecosystem condition layer, values in [0, 1].
    regulating_es : np.ndarray
        Regulating ecosystem services capacity, values in [0, 1].
    cultural_es : np.ndarray
        Cultural ecosystem services capacity, values in [0, 1].
    provisioning_es : np.ndarray
        Provisioning ecosystem services capacity, values in [0, 1].
        Will be transformed using Gaussian normalisation.
    anthropogenic_pressure : np.ndarray
        Raw anthropogenic pressure layer (e.g., population density).
        Dual role: values > threshold_max -> Group D elimination;
        values <= threshold_max -> inverted linear score in Group A.
    landuse : np.ndarray
        Categorical land use layer (CLC/OSO integer codes).
        Dual role: incompatible classes -> Group D elimination;
        compatible classes -> ordinal recoding for Group C.
    weights : dict
        Weight configuration dictionary with keys:
        - 'inter_group_weights': {'W_A': float, 'W_B': float, 'W_C': float}
        - 'group_a_weights': {'ecosystem_condition': float, 'regulating_es': float, 'low_pressure': float}
        - 'group_b_weights': {'cultural_es': float}
        - 'group_c_weights': {'provisioning_es': float, 'compatible_landuse': float}
    method : str, optional
        Aggregation method: 'geometric' or 'owa'. Default is 'geometric'.
    alpha : float, optional
        Orness parameter for OWA method, in [0, 1]. Default is 0.25.
    gap_bonus : float, optional
        Bonus multiplier for pixels identified as conservation gaps by
        Module 1 gap analysis. Applied as S_final = S × (1 + gap_bonus).
        Must be in [0, 0.20]. Default is 0.0 (no bonus).
    gap_mask : np.ndarray or None, optional
        Boolean array (same shape as input layers) where True indicates
        pixels within conservation gap zones from Module 1. If None or
        gap_bonus == 0.0, no bonus is applied.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:
        - 'score': Favourability score [0-1], NaN where ineligible.
        - 'oecm_mask': Boolean array, True = OECM favourable.
        - 'classical_pa_mask': Boolean array, True = classical PA preferable.
        - 'eliminatory_mask': Boolean array, True = eligible (passed Group D).

    Raises
    ------
    ValueError
        If method is not 'geometric' or 'owa'.
        If input arrays have inconsistent shapes.
        If required weight keys are missing.

    Notes
    -----
    Pipeline steps:
    1. Step 0: Apply Group D mask (build_eliminatory_mask) - incompatible land use
       and excessive pressure eliminate pixels.
    2. Step 1: Normalise layers using appropriate transformation functions:
       - pressure: inverted linear (low pressure = high score)
       - provisioning_es: Gaussian (non-monotone, optimum at mean)
       - others: sigmoid or linear per config
    3. Step 2: Compute intra-group scores (A, B, C) using chosen method.
    4. Step 3: Compute inter-group score using chosen method.
    5. Step 4: Check use presence (Group C threshold) to flag classical_pa_preferable.
    6. Step 5: Set score = NaN where eliminatory_mask = False.

    All configuration parameters are loaded from config/ files.
    """
    logger.info(f"Computing favourability with method='{method}', alpha={alpha}")

    # Validate method
    if method not in ('geometric', 'owa'):
        raise ValueError(
            f"Method must be 'geometric' or 'owa', got '{method}'"
        )

    # Validate all arrays have same shape
    reference_shape = ecosystem_condition.shape
    input_arrays = {
        'ecosystem_condition': ecosystem_condition,
        'regulating_es': regulating_es,
        'cultural_es': cultural_es,
        'provisioning_es': provisioning_es,
        'anthropogenic_pressure': anthropogenic_pressure,
        'landuse': landuse
    }
    for name, arr in input_arrays.items():
        if arr.shape != reference_shape:
            raise ValueError(
                f"Array '{name}' shape {arr.shape} does not match reference shape {reference_shape}"
            )

    # Load configurations
    criteria_config = _load_criteria_config()
    transform_config = _load_transformation_config()
    landuse_config = _load_landuse_config()

    # Extract thresholds — use caller-supplied pressure threshold (from sidebar slider)
    max_pressure = threshold_pressure
    min_use_threshold = criteria_config['use_presence']['min_group_c_score']

    # Build incompatible classes list from landuse config
    incompatible_classes = []
    for code, info in landuse_config.get('clc_compatibility', {}).items():
        if info.get('status') == 'eliminatory':
            # Handle both string codes (e.g., "1.1") and integer codes
            # Convert CLC codes to integers for the mask
            # CLC codes like "1", "1.1", "2.1" need to be matched against integer categories
            incompatible_classes.append(code)

    logger.info(f"Eliminatory thresholds: max_pressure={max_pressure}, "
                f"incompatible_classes={incompatible_classes}")

    # =========================================================================
    # Step 0: Apply Group D mask
    # =========================================================================
    eliminatory_mask = criteria_manager.build_eliminatory_mask(
        pressure_array=anthropogenic_pressure,
        landuse_array=landuse,
        threshold_max_pressure=max_pressure,
        incompatible_classes=incompatible_classes
    )
    logger.info(f"Group D mask: {np.sum(eliminatory_mask)} eligible pixels out of {eliminatory_mask.size}")

    # =========================================================================
    # Step 1: Normalise layers
    # =========================================================================
    # Ecosystem condition - sigmoid
    eco_params = transform_config['ecosystem_condition']
    if eco_params['type'] == 'sigmoid':
        eco_score = raster_preprocessing.normalize_sigmoid(
            ecosystem_condition,
            inflection=eco_params['inflection'],
            slope=eco_params['slope']
        )
    else:
        eco_score = raster_preprocessing.normalize_layer(
            ecosystem_condition, 'ecosystem_condition', eco_params,
            percentile_norm=percentile_norm
        )

    # Regulating ES - sigmoid
    reg_params = transform_config['regulating_es']
    if reg_params['type'] == 'sigmoid':
        _reg = raster_preprocessing.percentile_clip(regulating_es)[0] \
               if percentile_norm else regulating_es
        reg_score = raster_preprocessing.normalize_sigmoid(
            _reg,
            inflection=reg_params['inflection'],
            slope=reg_params['slope']
        )
    else:
        reg_score = raster_preprocessing.normalize_layer(
            regulating_es, 'regulating_es', reg_params,
            percentile_norm=percentile_norm
        )

    # Cultural ES - linear
    cult_params = transform_config['cultural_es']
    cult_score = raster_preprocessing.normalize_layer(
        cultural_es, 'cultural_es', cult_params,
        percentile_norm=percentile_norm
    )

    # Provisioning ES - Gaussian (non-monotone, optimum at mean)
    prov_params = transform_config['provisioning_es']
    prov_score = raster_preprocessing.normalize_gaussian(
        provisioning_es,
        mean=prov_params['mean'],
        std=prov_params['std']
    )

    # Anthropogenic pressure - inverted linear, normalised against [0, threshold_pressure].
    # Reference range is FIXED to [0, max_pressure]:
    #   pressure = 0           → score 1.0 (no pressure, always)
    #   pressure = max_pressure → score 0.0 (at the eliminatory limit, always)
    # Using the runtime eligible range [p_min, p_max] would make the score non-stationary:
    # the same pixel would receive a different score if the threshold is changed, because
    # the eligible population changes and therefore p_max changes. A fixed reference
    # ensures that pressure scores are comparable across different threshold settings.
    pressure_for_score = anthropogenic_pressure.astype(np.float32)
    pressure_for_score[~eliminatory_mask] = np.nan  # Exclude eliminated pixels

    valid_pressure = pressure_for_score[~np.isnan(pressure_for_score)]
    if len(valid_pressure) > 0:
        pressure_params = transform_config['anthropogenic_pressure'].copy()
        pressure_params['vmin'] = 0.0           # absolute minimum (no pressure)
        pressure_params['vmax'] = float(max_pressure)  # threshold = upper limit of eligible range
        pressure_score = raster_preprocessing.normalize_layer(
            pressure_for_score,
            'anthropogenic_pressure',
            pressure_params
        )
    else:
        # No valid pressure values (all pixels eliminated)
        pressure_score = np.full(pressure_for_score.shape, np.nan)

    # Recode land use for compatible classes
    landuse_score = criteria_manager.recode_landuse(
        landuse,
        landuse_config['clc_compatibility']
    )

    # =========================================================================
    # Step 2: Compute intra-group scores
    # =========================================================================
    # Extract weights
    group_a_weights = weights.get('group_a_weights', criteria_config['group_a_weights'])
    group_b_weights = weights.get('group_b_weights', criteria_config['group_b_weights'])
    group_c_weights = weights.get('group_c_weights', criteria_config['group_c_weights'])

    # Group A: ecological integrity (ecosystem_condition, regulating_es, low_pressure)
    group_a_arrays = {
        'ecosystem_condition': eco_score,
        'regulating_es': reg_score,
        'low_pressure': pressure_score
    }
    group_a_score = criteria_manager.compute_group_score(
        criteria_arrays=group_a_arrays,
        weights=group_a_weights,
        method=method,
        alpha=alpha
    )
    logger.info(f"Group A score range: [{np.nanmin(group_a_score):.4f}, {np.nanmax(group_a_score):.4f}]")

    # Group B: co-benefits (cultural_es)
    group_b_arrays = {
        'cultural_es': cult_score
    }
    group_b_score = criteria_manager.compute_group_score(
        criteria_arrays=group_b_arrays,
        weights=group_b_weights,
        method=method,
        alpha=alpha
    )
    logger.info(f"Group B score range: [{np.nanmin(group_b_score):.4f}, {np.nanmax(group_b_score):.4f}]")

    # Group C: use function (provisioning_es, compatible_landuse)
    group_c_arrays = {
        'provisioning_es': prov_score,
        'compatible_landuse': landuse_score
    }
    group_c_score = criteria_manager.compute_group_score(
        criteria_arrays=group_c_arrays,
        weights=group_c_weights,
        method=method,
        alpha=alpha
    )
    logger.info(f"Group C score range: [{np.nanmin(group_c_score):.4f}, {np.nanmax(group_c_score):.4f}]")

    # =========================================================================
    # Step 3: Compute inter-group score
    # =========================================================================
    inter_weights = weights.get('inter_group_weights', criteria_config['inter_group_weights'])

    # Build arrays and weights for inter-group aggregation
    inter_arrays = {
        'A': group_a_score,
        'B': group_b_score,
        'C': group_c_score
    }
    inter_weight_values = {
        'A': inter_weights['W_A'],
        'B': inter_weights['W_B'],
        'C': inter_weights['W_C']
    }

    final_score = criteria_manager.compute_group_score(
        criteria_arrays=inter_arrays,
        weights=inter_weight_values,
        method=method,
        alpha=alpha
    )
    logger.info(f"Final score range: [{np.nanmin(final_score):.4f}, {np.nanmax(final_score):.4f}]")

    # =========================================================================
    # Step 4: Check use presence (Group C threshold)
    # =========================================================================
    oecm_mask, classical_pa_mask = criteria_manager.check_use_presence(
        group_c_score=group_c_score,
        min_use_threshold=min_use_threshold
    )
    logger.info(f"OECM favourable: {np.sum(oecm_mask)} pixels")
    logger.info(f"Classical PA preferable: {np.sum(classical_pa_mask)} pixels")

    # =========================================================================
    # Step 5: Set score = NaN where eliminated or classical PA
    # =========================================================================
    # Apply eliminatory mask
    final_score[~eliminatory_mask] = np.nan

    # Pixels flagged as classical_pa_preferable should have score but be excluded from OECM
    # They are kept in the score array but oecm_mask = False for them

    # =========================================================================
    # Step 6: Apply gap analysis bonus (optional)
    # S_final = S × (1 + gap_bonus) for pixels within conservation gaps
    # =========================================================================
    if gap_bonus > 0.0 and gap_mask is not None:
        if gap_mask.shape != reference_shape:
            logger.warning(
                f"gap_mask shape {gap_mask.shape} != reference {reference_shape}, "
                f"skipping gap bonus"
            )
        else:
            # Apply bonus only to eligible, non-NaN pixels within gap zones
            bonus_pixels = gap_mask & eliminatory_mask & ~np.isnan(final_score)
            n_bonus = np.sum(bonus_pixels)
            if n_bonus > 0:
                final_score[bonus_pixels] *= (1.0 + gap_bonus)
                # Clamp to [0, 1]
                final_score[bonus_pixels] = np.clip(final_score[bonus_pixels], 0.0, 1.0)
                logger.info(
                    f"Gap bonus applied: {n_bonus} pixels boosted by "
                    f"factor {1.0 + gap_bonus:.2f}"
                )
            else:
                logger.info("Gap bonus: no eligible pixels in gap zones")

    # =========================================================================
    # Step 7: Apply PA proximity bonus (optional)
    # S_final = S × (1 + proximity_bonus × exp(-d / decay_km)) for eligible pixels
    # The bonus decays exponentially with distance from the PA network.
    # A pixel adjacent to an existing PA receives the full proximity_bonus;
    # pixels further away receive a proportionally smaller boost.
    # =========================================================================
    if proximity_bonus > 0.0 and pa_proximity_raster is not None:
        if pa_proximity_raster.shape != reference_shape:
            logger.warning(
                f"pa_proximity_raster shape {pa_proximity_raster.shape} != "
                f"reference {reference_shape}, skipping proximity bonus"
            )
        else:
            eligible = eliminatory_mask & ~np.isnan(final_score)
            decay_m  = proximity_decay_km * 1000.0   # km → metres (EPSG:3035)
            bonus_factor = proximity_bonus * np.exp(
                -pa_proximity_raster.astype(np.float32) / decay_m
            )
            final_score[eligible] = np.clip(
                final_score[eligible] * (1.0 + bonus_factor[eligible]),
                0.0, 1.0
            )
            n_boosted = int(np.sum(eligible & (bonus_factor > 0.01)))
            logger.info(
                f"Proximity bonus applied: {n_boosted} pixels boosted "
                f"(max_bonus={proximity_bonus:.2f}, decay={proximity_decay_km:.0f} km)"
            )

    # Return results.
    # group_scores are used by sensitivity analysis so it re-runs only the
    # inter-group (and optionally intra-group) aggregation — not the full
    # normalization pipeline — guaranteeing consistency with the main MCE.
    return {
        'score':             final_score,
        'oecm_mask':         oecm_mask & eliminatory_mask,
        'classical_pa_mask': classical_pa_mask & eliminatory_mask,
        'eliminatory_mask':  eliminatory_mask,
        'group_scores': {
            'A': group_a_score,
            'B': group_b_score,
            'C': group_c_score,
        },
        'normalised_arrays': {
            'ecosystem_condition': eco_score,
            'regulating_es':       reg_score,
            'low_pressure':        pressure_score,
            'cultural_es':         cult_score,
            'provisioning_es':     prov_score,
            'compatible_landuse':  landuse_score,
        },
    }

"""Criteria layer management and validation.

This module provides functions for loading criteria configurations, building
eliminatory masks, checking use presence thresholds, recoding land use
categories, and computing group scores for the OECM Favourability Tool.

Functions
---------
load_criteria_config
    Load and validate criteria configuration from YAML file.
build_eliminatory_mask
    Build Group D eliminatory mask from pressure and land use layers.
check_use_presence
    Check Group C score against use presence threshold.
recode_landuse
    Recode categorical land use to ordinal scores.
compute_group_score
    Aggregate criteria within a group using geometric mean or OWA.
"""

import logging
import numpy as np
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def load_criteria_config(config_path: str) -> dict:
    """Load criteria configuration from YAML file and validate required keys.

    Parameters
    ----------
    config_path : str
        Path to the criteria_defaults.yaml configuration file.

    Returns
    -------
    dict
        Validated configuration dictionary containing:
        - inter_group_weights: {W_A, W_B, W_C}
        - group_a_weights: {ecosystem_condition, regulating_es, low_pressure}
        - group_b_weights: {cultural_es}
        - group_c_weights: {provisioning_es, compatible_landuse}
        - aggregation: {default_method, default_alpha}
        - eliminatory: {max_anthropogenic_pressure, gap_bonus_max}
        - use_presence: {min_group_c_score}

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If required configuration keys are missing.

    Examples
    --------
    >>> config = load_criteria_config("config/criteria_defaults.yaml")
    >>> config['inter_group_weights']['W_A']
    0.5
    """
    logger.info(f"Loading criteria configuration from {config_path}")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Define required keys
    required_keys = [
        'inter_group_weights',
        'group_a_weights',
        'group_b_weights',
        'group_c_weights',
        'aggregation',
        'eliminatory',
        'use_presence'
    ]

    # Validate required top-level keys
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required configuration keys: {missing_keys}"
        )

    # Validate inter_group_weights
    inter_group_required = ['W_A', 'W_B', 'W_C']
    missing_inter = [k for k in inter_group_required if k not in config['inter_group_weights']]
    if missing_inter:
        raise ValueError(f"Missing inter_group_weights keys: {missing_inter}")

    # Validate group_a_weights
    group_a_required = ['ecosystem_condition', 'regulating_es', 'low_pressure']
    missing_a = [k for k in group_a_required if k not in config['group_a_weights']]
    if missing_a:
        raise ValueError(f"Missing group_a_weights keys: {missing_a}")

    # Validate group_b_weights
    if 'cultural_es' not in config['group_b_weights']:
        raise ValueError("Missing group_b_weights key: cultural_es")

    # Validate group_c_weights
    group_c_required = ['provisioning_es', 'compatible_landuse']
    missing_c = [k for k in group_c_required if k not in config['group_c_weights']]
    if missing_c:
        raise ValueError(f"Missing group_c_weights keys: {missing_c}")

    # Validate aggregation
    if 'default_method' not in config['aggregation']:
        raise ValueError("Missing aggregation key: default_method")
    if 'default_alpha' not in config['aggregation']:
        raise ValueError("Missing aggregation key: default_alpha")

    # Validate eliminatory
    if 'max_anthropogenic_pressure' not in config['eliminatory']:
        raise ValueError("Missing eliminatory key: max_anthropogenic_pressure")

    # Validate use_presence
    if 'min_group_c_score' not in config['use_presence']:
        raise ValueError("Missing use_presence key: min_group_c_score")

    logger.info("Configuration validated successfully")
    return config


def build_eliminatory_mask(
    pressure_array: np.ndarray,
    landuse_array: np.ndarray,
    threshold_max_pressure: float,
    incompatible_classes: list
) -> np.ndarray:
    """Build Group D eliminatory mask from pressure and land use layers.

    Combines eliminatory criteria via logical AND. A pixel is eligible (True)
    only if:
    - Pressure is at or below the maximum threshold, AND
    - Land use is not in the incompatible classes list.

    Parameters
    ----------
    pressure_array : np.ndarray
        Anthropogenic pressure layer (e.g., population density in inhabitants/km2).
    landuse_array : np.ndarray
        Categorical land use layer (integer or string codes matching CLC format).
    threshold_max_pressure : float
        Maximum allowed anthropogenic pressure. Pixels exceeding this are eliminated.
    incompatible_classes : list
        List of incompatible land use class codes (can be strings like "1.1" or integers).
        Pixels with these land use classes are eliminated.

    Returns
    -------
    np.ndarray
        Boolean mask where True = eligible pixel, False = eliminated pixel.
        Same shape as input arrays.

    Raises
    ------
    ValueError
        If input arrays have different shapes.

    Notes
    -----
    Group D criteria are applied FIRST, before any score calculation.
    The mask combines:
    - Pressure > threshold_max -> False (eliminated)
    - Land use in incompatible_classes -> False (eliminated)

    NaN values in pressure_array are treated as invalid (eliminated).

    Examples
    --------
    >>> pressure = np.array([[50, 200], [100, 75]])
    >>> landuse = np.array([[31, 11], [23, 31]])  # 31=forest, 11=urban, 23=pasture
    >>> incompatible = [11, 12]  # Urban classes
    >>> mask = build_eliminatory_mask(pressure, landuse, 150.0, incompatible)
    >>> # mask[0,0] = True (pressure OK, landuse OK)
    >>> # mask[0,1] = False (pressure > 150)
    >>> # mask[1,0] = True (pressure OK, landuse OK)
    >>> # mask[1,1] = True (pressure OK, landuse OK)
    """
    logger.info("Building Group D eliminatory mask")

    # Validate shapes
    if pressure_array.shape != landuse_array.shape:
        raise ValueError(
            f"Shape mismatch: pressure_array {pressure_array.shape} vs "
            f"landuse_array {landuse_array.shape}"
        )

    # Initialize mask as all eligible
    mask = np.ones(pressure_array.shape, dtype=bool)

    # Apply pressure threshold: pressure > threshold -> eliminated
    pressure_eliminated = pressure_array > threshold_max_pressure
    mask &= ~pressure_eliminated
    logger.info(f"Pressure threshold eliminates {np.sum(pressure_eliminated)} pixels")

    # Handle NaN in pressure: treat as eliminated
    pressure_nan = np.isnan(pressure_array)
    mask &= ~pressure_nan
    logger.info(f"Pressure NaN eliminates {np.sum(pressure_nan)} pixels")

    # Apply land use incompatibility
    # Handle both string and numeric codes
    landuse_eliminated = np.zeros(landuse_array.shape, dtype=bool)

    for clc_code in incompatible_classes:
        # Try to match as integer first (for categorical raster with integer codes)
        try:
            # Handle CLC codes like "1", "1.1", "2.1" etc.
            # For a categorical raster, the code might be stored as:
            # - Integer levels (1, 2, 3, ...)
            # - Or derived from CLC string codes
            if isinstance(clc_code, str):
                # Parse CLC code: "1" -> level 1, "1.1" -> match patterns
                # For level 1 codes like "1", "2", "3", we match all sub-codes
                if '.' not in clc_code:
                    # Level 1 code: match all classes starting with this level
                    level1 = int(clc_code)
                    # Match values in range [level1*10, (level1+1)*10) for 2-digit codes
                    # or [level1*100, (level1+1)*100) for 3-digit codes
                    # This depends on the encoding used in the landuse raster
                    # Common encoding: CLC 1.1.1 -> 111, CLC 2.3 -> 23, CLC 3 -> 3
                    landuse_eliminated |= (
                        ((landuse_array >= level1 * 10) & (landuse_array < (level1 + 1) * 10)) |
                        ((landuse_array >= level1 * 100) & (landuse_array < (level1 + 1) * 100)) |
                        (landuse_array == level1)
                    )
                else:
                    # Level 2 code like "1.1", "2.3"
                    parts = clc_code.split('.')
                    if len(parts) == 2:
                        level2_code = int(parts[0]) * 10 + int(parts[1])
                        landuse_eliminated |= (
                            (landuse_array == level2_code) |
                            ((landuse_array >= level2_code * 10) & (landuse_array < (level2_code + 1) * 10))
                        )
            else:
                # Direct integer match
                landuse_eliminated |= (landuse_array == clc_code)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse land use code: {clc_code}")
            continue

    mask &= ~landuse_eliminated
    logger.info(f"Incompatible land use eliminates {np.sum(landuse_eliminated)} pixels")

    total_eligible = np.sum(mask)
    total_eliminated = mask.size - total_eligible
    logger.info(f"Eliminatory mask: {total_eligible} eligible, {total_eliminated} eliminated")

    return mask


def check_use_presence(
    group_c_score: np.ndarray,
    min_use_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Check Group C score against use presence threshold.

    Pixels with Group C score below the minimum threshold are flagged as
    "classical PA preferable" - they have insufficient use function to
    qualify as OECMs but may be candidates for classical protected areas.

    Parameters
    ----------
    group_c_score : np.ndarray
        Aggregated Group C (use function) score, values in [0, 1].
    min_use_threshold : float
        Minimum threshold for Group C score. Pixels below this are flagged
        as classical PA preferable.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing:
        - oecm_mask: Boolean array, True = OECM favourable (score >= threshold)
        - classical_pa_mask: Boolean array, True = classical PA preferable (score < threshold)

    Notes
    -----
    The two masks are mutually exclusive for valid pixels:
    - oecm_mask[i] = True implies classical_pa_mask[i] = False
    - classical_pa_mask[i] = True implies oecm_mask[i] = False

    NaN values in group_c_score result in both masks being False.

    Examples
    --------
    >>> scores = np.array([0.5, 0.08, 0.3, np.nan])
    >>> oecm, classical = check_use_presence(scores, 0.1)
    >>> oecm  # [True, False, True, False]
    >>> classical  # [False, True, False, False]
    """
    logger.info(f"Checking use presence with threshold {min_use_threshold}")

    # Handle NaN values
    valid_mask = ~np.isnan(group_c_score)

    # OECM favourable: score >= threshold
    oecm_mask = valid_mask & (group_c_score >= min_use_threshold)

    # Classical PA preferable: score < threshold (but not NaN)
    classical_pa_mask = valid_mask & (group_c_score < min_use_threshold)

    logger.info(f"OECM favourable: {np.sum(oecm_mask)} pixels")
    logger.info(f"Classical PA preferable: {np.sum(classical_pa_mask)} pixels")

    return oecm_mask, classical_pa_mask


def recode_landuse(
    landuse_array: np.ndarray,
    compatibility_table: dict
) -> np.ndarray:
    """Recode categorical land use to ordinal scores.

    Converts CLC/OSO integer land use codes to ordinal compatibility scores
    in [0.0, 1.0] based on the compatibility table from configuration.

    Parameters
    ----------
    landuse_array : np.ndarray
        Categorical land use layer with integer codes.
    compatibility_table : dict
        Compatibility table from land_use_compatibility.yaml.
        Keys are CLC codes (strings like "3.1", "2.3").
        Values are dicts with 'status' and 'score' keys.

    Returns
    -------
    np.ndarray
        Array of ordinal scores in [0.0, 1.0]. Same shape as input.
        Incompatible and unknown classes receive score 0.0.

    Notes
    -----
    The compatibility table uses CLC hierarchical codes:
    - Level 1: "1", "2", "3", "4", "5"
    - Level 2: "1.1", "1.2", "2.3", etc.
    - Level 3: "1.1.1", "2.3.1", etc.

    Matching priority: Level 3 > Level 2 > Level 1.
    If no match found, score defaults to 0.0.

    Examples
    --------
    >>> landuse = np.array([[31, 23], [11, 41]])  # Forest, Pasture, Urban, Wetland
    >>> table = {
    ...     "3.1": {"status": "compatible", "score": 0.85},
    ...     "2.3": {"status": "compatible", "score": 0.75},
    ...     "1.1": {"status": "eliminatory", "score": None},
    ...     "4.1": {"status": "compatible", "score": 0.90}
    ... }
    >>> scores = recode_landuse(landuse, table)
    >>> # scores ≈ [[0.85, 0.75], [0.0, 0.90]]
    """
    logger.info("Recoding land use to ordinal scores")

    # Build lookup from integer code to score
    code_to_score = {}
    for clc_code, info in compatibility_table.items():
        score = 0.0 if (info.get('status') == 'eliminatory' or info.get('score') is None) \
                else float(info.get('score', 0.0))
        if isinstance(clc_code, str) and '.' in clc_code:
            parts = clc_code.split('.')
            if len(parts) == 2:
                code_to_score[int(parts[0]) * 10 + int(parts[1])] = score
            elif len(parts) == 3:
                code_to_score[int(parts[0]) * 100 + int(parts[1]) * 10 + int(parts[2])] = score
        else:
            code_to_score[int(clc_code)] = score

    # Resolve each unique raster value once (with hierarchical fallback), then
    # apply via fancy indexing — avoids one full-array comparison per CLC class.
    unique_vals = np.unique(landuse_array[np.isfinite(landuse_array.astype(float))])
    val_to_score = {}
    for val in unique_vals:
        v = int(val)
        if v in code_to_score:
            val_to_score[v] = code_to_score[v]
        else:
            level2 = v // 10
            if level2 in code_to_score:
                val_to_score[v] = code_to_score[level2]
            else:
                level1 = v // 100 if v >= 100 else v // 10
                val_to_score[v] = code_to_score.get(level1, 0.0)

    # Build a vectorised lookup array: map integer raster value → score in one pass
    if len(unique_vals) > 0:
        max_code = int(unique_vals.max()) + 1
        lut = np.zeros(max_code, dtype=np.float32)
        for v, s in val_to_score.items():
            if 0 <= v < max_code:
                lut[v] = s
        land_int = landuse_array.astype(np.int32)
        valid = (land_int >= 0) & (land_int < max_code)
        scores = np.zeros(landuse_array.shape, dtype=np.float32)
        scores[valid] = lut[land_int[valid]]
    else:
        scores = np.zeros(landuse_array.shape, dtype=np.float32)

    logger.info(f"Recoded land use. Score range: [{scores.min():.2f}, {scores.max():.2f}]")
    return scores


def compute_group_score(
    criteria_arrays: dict[str, np.ndarray],
    weights: dict[str, float],
    method: str,
    alpha: float = 0.25
) -> np.ndarray:
    """Aggregate criteria within a group using geometric mean or OWA.

    Parameters
    ----------
    criteria_arrays : dict[str, np.ndarray]
        Dictionary mapping criterion names to numpy arrays.
        All arrays must have the same shape.
    weights : dict[str, float]
        Dictionary mapping criterion names to weights.
        Weights must sum to 1.0.
    method : str
        Aggregation method: 'geometric' or 'owa'.
    alpha : float, optional
        Orness parameter for OWA method, in [0, 1]. Default is 0.25.

    Returns
    -------
    np.ndarray
        Aggregated group score array with values in [0, 1].

    Raises
    ------
    ValueError
        If method is not 'geometric' or 'owa'.
        If weights do not sum to 1.0.
        If criterion names in arrays and weights do not match.

    Notes
    -----
    This function dispatches to weighted_geometric_mean or yager_owa
    from mce_engine module.

    Weighted Linear Combination (WLC) is NOT supported per specifications.

    Examples
    --------
    >>> arrays = {
    ...     'criterion_a': np.array([0.8, 0.6]),
    ...     'criterion_b': np.array([0.5, 0.9])
    ... }
    >>> weights = {'criterion_a': 0.6, 'criterion_b': 0.4}
    >>> score = compute_group_score(arrays, weights, method='geometric')
    """
    logger.info(f"Computing group score with method='{method}', {len(criteria_arrays)} criteria")

    # Import mce_engine functions (avoid circular import)
    from . import mce_engine

    # Validate method
    if method not in ('geometric', 'owa'):
        raise ValueError(
            f"Method must be 'geometric' or 'owa', got '{method}'. "
            "Note: WLC (weighted linear combination) is forbidden."
        )

    # Match criteria names
    array_names = set(criteria_arrays.keys())
    weight_names = set(weights.keys())

    if array_names != weight_names:
        missing_weights = array_names - weight_names
        missing_arrays = weight_names - array_names
        raise ValueError(
            f"Criterion name mismatch. Missing weights: {missing_weights}, "
            f"Missing arrays: {missing_arrays}"
        )

    # Check weights sum to 1.0
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"Weights must sum to 1.0, got {weight_sum:.6f}"
        )

    # Build ordered lists
    names = sorted(criteria_arrays.keys())
    arrays_list = [criteria_arrays[name] for name in names]
    weights_list = [weights[name] for name in names]

    # Dispatch to appropriate method
    if method == 'geometric':
        return mce_engine.weighted_geometric_mean(arrays_list, weights_list)
    else:  # method == 'owa'
        return mce_engine.yager_owa(arrays_list, weights_list, alpha)


# Legacy stubs for backward compatibility
def load_criteria_layers(layer_paths, settings_config, transformation_config):
    """
    Load and preprocess all required criterion layers.

    Args:
        layer_paths: Dictionary mapping criterion names to file paths
        settings_config: Configuration from settings.yaml
        transformation_config: Configuration from transformation_functions.yaml

    Returns:
        Dictionary of preprocessed xarray DataArrays
    """
    raise NotImplementedError("Criteria layer loading not yet implemented")


def validate_criteria_stack(criteria_dict, required_criteria):
    """
    Validate that all required criteria are present and spatially aligned.

    Args:
        criteria_dict: Dictionary of criterion DataArrays
        required_criteria: List of required criterion names

    Returns:
        Boolean validation status and list of any missing/misaligned layers
    """
    raise NotImplementedError("Criteria validation not yet implemented")

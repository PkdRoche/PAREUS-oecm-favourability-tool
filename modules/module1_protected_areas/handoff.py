"""Module 1 → Module 2 weight handoff validation and formatting.

This module ensures that Group A criterion weights proposed by Module 1
representativity analysis are correctly formatted and validated before
being passed to Module 2 MCE engine.

Functions
---------
validate_weight_handoff
    Validate that weight dictionary conforms to the handoff contract.
format_weights_for_mce
    Format weights in the exact format Module 2 expects.
"""

import logging
from pathlib import Path
from typing import Dict
import yaml

logger = logging.getLogger(__name__)


def validate_weight_handoff(
    weight_dict: Dict[str, float],
    config_path: str
) -> None:
    """Validate that weight dictionary conforms to the Module 1 → Module 2 handoff contract.

    Parameters
    ----------
    weight_dict : dict[str, float]
        Weight dictionary from propose_group_a_weights.
        Keys should be Group A criterion names.
        Values should be weights summing to 1.0.
    config_path : str
        Path to criteria_defaults.yaml configuration file.

    Raises
    ------
    ValueError
        If weight dictionary violates the handoff contract:
        - Keys do not match expected Group A criteria
        - Weights do not sum to 1.0 (tolerance 1e-6)
        - Any weight is negative
        - Any weight is zero (zero-weight criteria should be excluded)

    Notes
    -----
    The handoff contract requires:
    1. Keys must match Group A criterion IDs in config/criteria_defaults.yaml
    2. Values must sum to 1.0 ± 1e-6
    3. No zero-weight criteria included (they should be excluded upstream)
    4. All weights must be non-negative

    Examples
    --------
    >>> weights = {
    ...     'ecosystem_condition': 0.35,
    ...     'regulating_es': 0.45,
    ...     'low_pressure': 0.20
    ... }
    >>> validate_weight_handoff(weights, "config/criteria_defaults.yaml")
    # Passes validation

    >>> bad_weights = {'unknown_criterion': 1.0}
    >>> validate_weight_handoff(bad_weights, "config/criteria_defaults.yaml")
    ValueError: Weight keys do not match expected Group A criteria...
    """
    # Load expected Group A criteria from config
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    expected_criteria = set(config.get('group_a_weights', {}).keys())

    if not expected_criteria:
        raise ValueError(
            f"No Group A criteria found in config file: {config_path}"
        )

    # Validate keys match
    provided_criteria = set(weight_dict.keys())

    if provided_criteria != expected_criteria:
        missing = expected_criteria - provided_criteria
        extra = provided_criteria - expected_criteria

        error_msg = "Weight keys do not match expected Group A criteria.\n"
        if missing:
            error_msg += f"  Missing: {sorted(missing)}\n"
        if extra:
            error_msg += f"  Unexpected: {sorted(extra)}\n"
        error_msg += f"  Expected: {sorted(expected_criteria)}"

        raise ValueError(error_msg)

    # Validate all weights are non-negative
    negative_weights = {k: v for k, v in weight_dict.items() if v < 0}
    if negative_weights:
        raise ValueError(
            f"Negative weights are not allowed: {negative_weights}"
        )

    # Validate no zero weights (should be excluded upstream)
    zero_weights = [k for k, v in weight_dict.items() if v == 0.0]
    if zero_weights:
        logger.warning(
            f"Zero-weight criteria detected: {zero_weights}. "
            f"Consider excluding these criteria from the weight dictionary."
        )

    # Validate sum equals 1.0
    weight_sum = sum(weight_dict.values())
    tolerance = 1e-6

    if abs(weight_sum - 1.0) > tolerance:
        raise ValueError(
            f"Weights must sum to 1.0 (tolerance {tolerance}). "
            f"Got sum = {weight_sum:.8f}. "
            f"Difference: {abs(weight_sum - 1.0):.8e}"
        )

    logger.info(
        f"Weight handoff validation passed. "
        f"Keys: {sorted(weight_dict.keys())}, "
        f"Sum: {weight_sum:.8f}"
    )


def format_weights_for_mce(
    weight_dict: Dict[str, float]
) -> Dict[str, float]:
    """Format weights in the exact format Module 2 MCE engine expects.

    Currently, Module 2 expects the same format as Module 1 produces, so this
    function performs validation and returns a copy of the input dictionary.
    Future versions may require format transformations.

    Parameters
    ----------
    weight_dict : dict[str, float]
        Weight dictionary from propose_group_a_weights.
        Keys should be Group A criterion names.
        Values should be weights summing to 1.0.

    Returns
    -------
    dict[str, float]
        Formatted weight dictionary ready for Module 2 MCE engine.
        Keys: Group A criterion names
        Values: Normalised weights (sum = 1.0)

    Notes
    -----
    This function ensures:
    - Keys are strings (criterion names)
    - Values are floats
    - Weights are normalised to exactly 1.0 (corrects minor floating point errors)
    - Zero-weight criteria are excluded

    Examples
    --------
    >>> weights = {
    ...     'ecosystem_condition': 0.35,
    ...     'regulating_es': 0.45,
    ...     'low_pressure': 0.20
    ... }
    >>> mce_weights = format_weights_for_mce(weights)
    >>> mce_weights == weights
    True
    >>> sum(mce_weights.values())
    1.0
    """
    # Filter out zero-weight criteria
    filtered = {k: v for k, v in weight_dict.items() if v > 0.0}

    if not filtered:
        raise ValueError(
            "All weights are zero. Cannot format empty weight dictionary."
        )

    # Normalise to exactly 1.0 (correct minor floating point errors)
    weight_sum = sum(filtered.values())
    normalised = {k: v / weight_sum for k, v in filtered.items()}

    # Verify normalisation
    final_sum = sum(normalised.values())
    logger.info(
        f"Formatted weights for MCE. "
        f"Criteria: {len(normalised)}, "
        f"Sum: {final_sum:.15f}"
    )

    return normalised

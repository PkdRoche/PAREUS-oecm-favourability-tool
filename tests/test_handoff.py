"""Tests for Module 1 → Module 2 weight handoff validation."""

import pytest
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'modules'))

from module1_protected_areas.handoff import (
    validate_weight_handoff,
    format_weights_for_mce
)


@pytest.fixture
def config_path():
    """Path to criteria_defaults.yaml configuration file."""
    return str(Path(__file__).parent.parent / 'config' / 'criteria_defaults.yaml')


def test_valid_handoff_passes_validation(config_path):
    """Test that a valid weight dictionary passes validation."""
    # Valid weights for Group A criteria
    weights = {
        'ecosystem_condition': 0.35,
        'regulating_es': 0.45,
        'low_pressure': 0.20
    }

    # Should not raise any exception
    validate_weight_handoff(weights, config_path)

    # Format for MCE should succeed
    mce_weights = format_weights_for_mce(weights)

    # Verify output
    assert isinstance(mce_weights, dict)
    assert set(mce_weights.keys()) == set(weights.keys())
    assert abs(sum(mce_weights.values()) - 1.0) < 1e-10


def test_mismatched_keys_raises_ValueError(config_path):
    """Test that mismatched criterion keys raise ValueError."""
    # Missing 'low_pressure', has unknown 'unknown_criterion'
    weights = {
        'ecosystem_condition': 0.5,
        'regulating_es': 0.3,
        'unknown_criterion': 0.2
    }

    with pytest.raises(ValueError) as excinfo:
        validate_weight_handoff(weights, config_path)

    # Verify error message contains helpful information
    error_msg = str(excinfo.value)
    assert 'do not match' in error_msg
    assert 'Missing' in error_msg or 'Unexpected' in error_msg


def test_weights_not_summing_to_one_raises_ValueError(config_path):
    """Test that weights not summing to 1.0 raise ValueError."""
    # Sum = 0.95 (off by 0.05)
    weights = {
        'ecosystem_condition': 0.35,
        'regulating_es': 0.40,
        'low_pressure': 0.20
    }

    with pytest.raises(ValueError) as excinfo:
        validate_weight_handoff(weights, config_path)

    error_msg = str(excinfo.value)
    assert 'sum to 1.0' in error_msg.lower()


def test_negative_weights_raise_ValueError(config_path):
    """Test that negative weights raise ValueError."""
    weights = {
        'ecosystem_condition': 0.5,
        'regulating_es': 0.6,
        'low_pressure': -0.1  # Negative weight
    }

    with pytest.raises(ValueError) as excinfo:
        validate_weight_handoff(weights, config_path)

    error_msg = str(excinfo.value)
    assert 'negative' in error_msg.lower()


def test_zero_weight_criteria_excluded_from_mce_format():
    """Test that zero-weight criteria are excluded by format_weights_for_mce."""
    weights = {
        'ecosystem_condition': 0.5,
        'regulating_es': 0.5,
        'low_pressure': 0.0  # Zero weight
    }

    mce_weights = format_weights_for_mce(weights)

    # Zero-weight criterion should be excluded
    assert 'low_pressure' not in mce_weights
    assert len(mce_weights) == 2

    # Remaining weights should be renormalised to sum to 1.0
    assert abs(sum(mce_weights.values()) - 1.0) < 1e-10


def test_format_normalises_to_exactly_one():
    """Test that format_weights_for_mce normalises to exactly 1.0."""
    # Weights with minor floating point error
    weights = {
        'ecosystem_condition': 0.333333,
        'regulating_es': 0.333333,
        'low_pressure': 0.333333
    }

    mce_weights = format_weights_for_mce(weights)

    # Sum should be exactly 1.0 (within machine epsilon)
    weight_sum = sum(mce_weights.values())
    assert abs(weight_sum - 1.0) < 1e-15, f"Sum should be exactly 1.0, got {weight_sum}"


def test_all_zero_weights_raise_ValueError():
    """Test that a dictionary with all zero weights raises ValueError."""
    weights = {
        'ecosystem_condition': 0.0,
        'regulating_es': 0.0,
        'low_pressure': 0.0
    }

    with pytest.raises(ValueError) as excinfo:
        format_weights_for_mce(weights)

    error_msg = str(excinfo.value)
    assert 'zero' in error_msg.lower() or 'empty' in error_msg.lower()


def test_missing_criteria_in_config_raises_error(tmp_path):
    """Test that missing Group A criteria in config raises error."""
    # Create invalid config without group_a_weights
    invalid_config = tmp_path / "invalid_config.yaml"
    invalid_config.write_text("inter_group_weights:\n  W_A: 0.5\n")

    weights = {
        'ecosystem_condition': 0.35,
        'regulating_es': 0.45,
        'low_pressure': 0.20
    }

    with pytest.raises(ValueError) as excinfo:
        validate_weight_handoff(weights, str(invalid_config))

    error_msg = str(excinfo.value)
    assert 'No Group A criteria' in error_msg


def test_handoff_contract_integration(config_path):
    """Integration test: verify complete handoff pipeline."""
    # Simulate Module 1 output
    module1_weights = {
        'ecosystem_condition': 0.33,
        'regulating_es': 0.47,
        'low_pressure': 0.20
    }

    # Step 1: Validate handoff contract
    validate_weight_handoff(module1_weights, config_path)

    # Step 2: Format for MCE
    mce_weights = format_weights_for_mce(module1_weights)

    # Step 3: Verify MCE-ready weights meet Module 2 expectations
    # Module 2 expects dict[str, float] with keys matching Group A criteria
    assert isinstance(mce_weights, dict)
    assert all(isinstance(k, str) for k in mce_weights.keys())
    assert all(isinstance(v, float) for v in mce_weights.values())

    # Keys should match Group A criteria
    expected_keys = {'ecosystem_condition', 'regulating_es', 'low_pressure'}
    assert set(mce_weights.keys()) == expected_keys

    # Sum should be exactly 1.0
    assert abs(sum(mce_weights.values()) - 1.0) < 1e-10

    # All weights should be positive
    assert all(v > 0 for v in mce_weights.values())


def test_format_preserves_keys_and_values():
    """Test that format_weights_for_mce preserves keys and relative values."""
    weights = {
        'ecosystem_condition': 0.2,
        'regulating_es': 0.5,
        'low_pressure': 0.3
    }

    mce_weights = format_weights_for_mce(weights)

    # Keys should be preserved
    assert set(mce_weights.keys()) == set(weights.keys())

    # Relative ordering should be preserved
    assert mce_weights['regulating_es'] > mce_weights['low_pressure']
    assert mce_weights['low_pressure'] > mce_weights['ecosystem_condition']


def test_validate_with_floating_point_tolerance(config_path):
    """Test that validation accepts weights summing to 1.0 within tolerance."""
    # Sum = 1.0000000001 (within 1e-6 tolerance)
    weights = {
        'ecosystem_condition': 0.33333333,
        'regulating_es': 0.33333333,
        'low_pressure': 0.33333334
    }

    # Should pass validation (sum within tolerance)
    validate_weight_handoff(weights, config_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

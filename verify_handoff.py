"""Quick verification script for Module 1 → Module 2 weight handoff.

Run this script to verify the handoff implementation works correctly:
    python verify_handoff.py
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / 'modules'))

from module1_protected_areas.handoff import (
    validate_weight_handoff,
    format_weights_for_mce
)


def test_valid_handoff():
    """Test 1: Valid handoff passes validation."""
    print("Test 1: Valid handoff passes validation")
    config_path = str(Path(__file__).parent / 'config' / 'criteria_defaults.yaml')

    weights = {
        'ecosystem_condition': 0.35,
        'regulating_es': 0.45,
        'low_pressure': 0.20
    }

    try:
        validate_weight_handoff(weights, config_path)
        mce_weights = format_weights_for_mce(weights)
        assert abs(sum(mce_weights.values()) - 1.0) < 1e-10
        print("  ✓ PASS: Valid weights accepted")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_mismatched_keys():
    """Test 2: Mismatched keys raise ValueError."""
    print("Test 2: Mismatched keys raise ValueError")
    config_path = str(Path(__file__).parent / 'config' / 'criteria_defaults.yaml')

    weights = {
        'ecosystem_condition': 0.5,
        'regulating_es': 0.3,
        'unknown_criterion': 0.2
    }

    try:
        validate_weight_handoff(weights, config_path)
        print("  ✗ FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        if 'do not match' in str(e):
            print("  ✓ PASS: Correctly rejected mismatched keys")
            return True
        else:
            print(f"  ✗ FAIL: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"  ✗ FAIL: Unexpected error: {e}")
        return False


def test_weights_not_summing_to_one():
    """Test 3: Weights not summing to 1 raise ValueError."""
    print("Test 3: Weights not summing to 1 raise ValueError")
    config_path = str(Path(__file__).parent / 'config' / 'criteria_defaults.yaml')

    weights = {
        'ecosystem_condition': 0.35,
        'regulating_es': 0.40,
        'low_pressure': 0.20
    }

    try:
        validate_weight_handoff(weights, config_path)
        print("  ✗ FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        if 'sum to 1.0' in str(e).lower():
            print("  ✓ PASS: Correctly rejected weights not summing to 1")
            return True
        else:
            print(f"  ✗ FAIL: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"  ✗ FAIL: Unexpected error: {e}")
        return False


def test_format_normalises():
    """Test 4: format_weights_for_mce normalises to exactly 1.0."""
    print("Test 4: format_weights_for_mce normalises to exactly 1.0")

    weights = {
        'ecosystem_condition': 0.333333,
        'regulating_es': 0.333333,
        'low_pressure': 0.333333
    }

    try:
        mce_weights = format_weights_for_mce(weights)
        weight_sum = sum(mce_weights.values())

        if abs(weight_sum - 1.0) < 1e-15:
            print(f"  ✓ PASS: Normalised to sum = {weight_sum:.15f}")
            return True
        else:
            print(f"  ✗ FAIL: Sum = {weight_sum}, expected 1.0")
            return False
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_zero_weight_exclusion():
    """Test 5: Zero-weight criteria are excluded."""
    print("Test 5: Zero-weight criteria are excluded")

    weights = {
        'ecosystem_condition': 0.5,
        'regulating_es': 0.5,
        'low_pressure': 0.0
    }

    try:
        mce_weights = format_weights_for_mce(weights)

        if 'low_pressure' not in mce_weights and len(mce_weights) == 2:
            print("  ✓ PASS: Zero-weight criterion excluded")
            return True
        else:
            print(f"  ✗ FAIL: Expected 2 criteria, got {len(mce_weights)}")
            return False
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Module 1 → Module 2 Weight Handoff Verification")
    print("=" * 60)
    print()

    tests = [
        test_valid_handoff,
        test_mismatched_keys,
        test_weights_not_summing_to_one,
        test_format_normalises,
        test_zero_weight_exclusion
    ]

    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
        print()

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} test(s) failed")

    print("=" * 60)

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

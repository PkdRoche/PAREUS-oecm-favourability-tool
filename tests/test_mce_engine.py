"""Tests for MCE engine and criteria manager.

All tests use manually computed reference values to verify analytical correctness.
Tests cover weighted geometric mean, Yager OWA, eliminatory masks, and the full
favourability pipeline.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.module2_favourability import mce_engine
from modules.module2_favourability import criteria_manager


class TestWeightedGeometricMean:
    """Tests for weighted_geometric_mean function."""

    def test_geometric_mean_known_values(self):
        """Test geometric mean with manually computed reference values.

        Given:
            arrays = [np.array([0.8, 0.6]), np.array([0.5, 0.9])]
            weights = [0.6, 0.4]

        Expected:
            result[0] = 0.8^0.6 * 0.5^0.4 = 0.8705505 * 0.7578583 = 0.6597539
            result[1] = 0.6^0.6 * 0.9^0.4 = 0.7213475 * 0.9587041 = 0.6915502

        Manually verified calculation:
            0.8^0.6 = exp(0.6 * ln(0.8)) = exp(0.6 * -0.2231) = exp(-0.1339) = 0.8747
            0.5^0.4 = exp(0.4 * ln(0.5)) = exp(0.4 * -0.6931) = exp(-0.2773) = 0.7579
            Product = 0.8747 * 0.7579 = 0.6630

            0.6^0.6 = exp(0.6 * ln(0.6)) = exp(0.6 * -0.5108) = exp(-0.3065) = 0.7360
            0.9^0.4 = exp(0.4 * ln(0.9)) = exp(0.4 * -0.1054) = exp(-0.0422) = 0.9587
            Product = 0.7360 * 0.9587 = 0.7056
        """
        arrays = [np.array([0.8, 0.6]), np.array([0.5, 0.9])]
        weights = [0.6, 0.4]

        result = mce_engine.weighted_geometric_mean(arrays, weights)

        # Manually computed expected values
        expected_0 = 0.8 ** 0.6 * 0.5 ** 0.4  # = 0.6630 approximately
        expected_1 = 0.6 ** 0.6 * 0.9 ** 0.4  # = 0.7056 approximately

        np.testing.assert_almost_equal(result[0], expected_0, decimal=4)
        np.testing.assert_almost_equal(result[1], expected_1, decimal=4)

    def test_geometric_mean_weights_sum_to_one_enforced(self):
        """Test that weights must sum to 1.0."""
        arrays = [np.array([0.8]), np.array([0.5])]
        weights = [0.5, 0.3]  # Sum = 0.8, not 1.0

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            mce_engine.weighted_geometric_mean(arrays, weights)

    def test_geometric_mean_zero_criterion_nullifies_score(self):
        """Test that a zero criterion drives the output very close to zero.

        This is the key non-compensatory property of geometric mean. Per
        weighted_geometric_mean's own docstring, exact 0 inputs are floored
        to 1e-9 before the log-space computation (not treated as exact 0)
        so the result is driven very close to — but not exactly — 0; hard
        pixel elimination is handled upstream by the Group D mask instead.
        """
        arrays = [np.array([0.8, 0.0]), np.array([0.5, 0.9])]
        weights = [0.6, 0.4]

        result = mce_engine.weighted_geometric_mean(arrays, weights)

        # First pixel: normal calculation
        assert result[0] > 0

        # Second pixel: zero in first array -> result must be driven near-zero
        assert result[1] < 1e-4

    def test_geometric_mean_nan_propagation(self):
        """Test that NaN in any criterion propagates to output."""
        arrays = [np.array([0.8, np.nan]), np.array([0.5, 0.9])]
        weights = [0.6, 0.4]

        result = mce_engine.weighted_geometric_mean(arrays, weights)

        assert not np.isnan(result[0])
        assert np.isnan(result[1])

    def test_geometric_mean_equal_weights(self):
        """Test geometric mean with equal weights (standard geometric mean)."""
        arrays = [np.array([0.4]), np.array([0.9])]
        weights = [0.5, 0.5]

        result = mce_engine.weighted_geometric_mean(arrays, weights)

        # Standard geometric mean: sqrt(0.4 * 0.9) = sqrt(0.36) = 0.6
        expected = np.sqrt(0.4 * 0.9)
        np.testing.assert_almost_equal(result[0], expected, decimal=4)

    def test_geometric_mean_shape_mismatch_raises(self):
        """Test that mismatched array shapes raise ValueError."""
        arrays = [np.array([0.8, 0.6]), np.array([0.5])]
        weights = [0.6, 0.4]

        with pytest.raises(ValueError, match="shape"):
            mce_engine.weighted_geometric_mean(arrays, weights)


class TestYagerOWA:
    """Tests for yager_owa function."""

    def test_owa_alpha_zero_equals_minimum(self):
        """Test that alpha=0 produces pure AND logic (minimum).

        With alpha=0, OWA should return the minimum weighted value.
        """
        arrays = [np.array([0.8]), np.array([0.5]), np.array([0.2])]
        weights = [1/3, 1/3, 1/3]  # Equal importance

        result = mce_engine.yager_owa(arrays, weights, alpha=0.0)

        # With equal weights, alpha=0 should give minimum value
        expected_min = 0.2
        np.testing.assert_almost_equal(result[0], expected_min, decimal=4)

    def test_owa_alpha_one_equals_maximum(self):
        """Test that alpha=1 produces pure OR logic (maximum).

        With alpha=1, OWA should return the maximum weighted value.
        """
        arrays = [np.array([0.8]), np.array([0.5]), np.array([0.2])]
        weights = [1/3, 1/3, 1/3]  # Equal importance

        result = mce_engine.yager_owa(arrays, weights, alpha=1.0)

        # With equal weights, alpha=1 should give maximum value
        expected_max = 0.8
        np.testing.assert_almost_equal(result[0], expected_max, decimal=4)

    def test_owa_alpha_half_between_min_and_max(self):
        """Test that alpha=0.5 produces result between min and max."""
        arrays = [np.array([0.9]), np.array([0.5]), np.array([0.1])]
        weights = [1/3, 1/3, 1/3]

        result = mce_engine.yager_owa(arrays, weights, alpha=0.5)

        # Result should be strictly between min and max
        assert result[0] > 0.1, "Result should be greater than minimum"
        assert result[0] < 0.9, "Result should be less than maximum"

    def test_owa_invalid_alpha_raises(self):
        """Test that alpha outside [0, 1] raises ValueError."""
        arrays = [np.array([0.8]), np.array([0.5])]
        weights = [0.5, 0.5]

        with pytest.raises(ValueError, match="Alpha must be in"):
            mce_engine.yager_owa(arrays, weights, alpha=1.5)

        with pytest.raises(ValueError, match="Alpha must be in"):
            mce_engine.yager_owa(arrays, weights, alpha=-0.1)

    def test_owa_weights_sum_enforced(self):
        """Test that weights must sum to 1.0."""
        arrays = [np.array([0.8]), np.array([0.5])]
        weights = [0.3, 0.3]  # Sum = 0.6

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            mce_engine.yager_owa(arrays, weights, alpha=0.5)

    def test_owa_nan_propagation(self):
        """Test that NaN propagates through OWA."""
        arrays = [np.array([0.8, np.nan]), np.array([0.5, 0.9])]
        weights = [0.5, 0.5]

        result = mce_engine.yager_owa(arrays, weights, alpha=0.5)

        assert not np.isnan(result[0])
        assert np.isnan(result[1])


class TestEliminatoryMask:
    """Tests for build_eliminatory_mask function."""

    def test_eliminatory_mask_incompatible_class_excluded(self):
        """Test that incompatible land use class is excluded."""
        pressure = np.array([[50, 50], [50, 50]])
        landuse = np.array([[31, 11], [23, 31]])  # 11 = urban (incompatible)
        threshold = 150.0
        incompatible = ["1.1"]  # Urban class

        mask = criteria_manager.build_eliminatory_mask(
            pressure, landuse, threshold, incompatible
        )

        # Pixel [0,1] has landuse 11 (class 1.1) -> should be False
        assert mask[0, 1] == False
        # Other pixels should be True
        assert mask[0, 0] == True
        assert mask[1, 0] == True
        assert mask[1, 1] == True

    def test_eliminatory_mask_high_pressure_excluded(self):
        """Test that high pressure pixels are excluded."""
        pressure = np.array([[50, 200], [100, 75]])
        landuse = np.array([[31, 31], [31, 31]])  # All compatible
        threshold = 150.0
        incompatible = []

        mask = criteria_manager.build_eliminatory_mask(
            pressure, landuse, threshold, incompatible
        )

        # Pixel [0,1] has pressure 200 > 150 -> should be False
        assert mask[0, 1] == False
        # Other pixels should be True
        assert mask[0, 0] == True
        assert mask[1, 0] == True
        assert mask[1, 1] == True

    def test_eliminatory_mask_eligible_pixel_retained(self):
        """Test that eligible pixels are retained."""
        pressure = np.array([[100, 75]])
        landuse = np.array([[31, 23]])  # Forest and pasture (compatible)
        threshold = 150.0
        incompatible = ["1", "1.1", "1.2", "2.1"]

        mask = criteria_manager.build_eliminatory_mask(
            pressure, landuse, threshold, incompatible
        )

        # Both pixels should be True (pressure OK, landuse compatible)
        assert mask[0, 0] == True
        assert mask[0, 1] == True

    def test_eliminatory_mask_nan_pressure_excluded(self):
        """Test that NaN pressure values are excluded."""
        pressure = np.array([[50, np.nan]])
        landuse = np.array([[31, 31]])
        threshold = 150.0
        incompatible = []

        mask = criteria_manager.build_eliminatory_mask(
            pressure, landuse, threshold, incompatible
        )

        assert mask[0, 0] == True
        assert mask[0, 1] == False


class TestGroupCFlag:
    """Tests for check_use_presence function."""

    def test_group_c_flag_low_score_sets_classical_pa(self):
        """Test that low Group C score sets classical_pa_mask = True."""
        scores = np.array([0.05, 0.08, 0.02])
        threshold = 0.10

        oecm_mask, classical_mask = criteria_manager.check_use_presence(
            scores, threshold
        )

        # All scores below threshold -> classical PA
        assert np.all(classical_mask == True)
        assert np.all(oecm_mask == False)

    def test_group_c_flag_score_above_threshold_sets_oecm(self):
        """Test that score above threshold sets OECM mask."""
        scores = np.array([0.5, 0.15, 0.3])
        threshold = 0.10

        oecm_mask, classical_mask = criteria_manager.check_use_presence(
            scores, threshold
        )

        # All scores above threshold -> OECM
        assert np.all(oecm_mask == True)
        assert np.all(classical_mask == False)

    def test_group_c_flag_mixed_scores(self):
        """Test mixed scores produce correct masks."""
        scores = np.array([0.5, 0.08, 0.3, 0.05])
        threshold = 0.10

        oecm_mask, classical_mask = criteria_manager.check_use_presence(
            scores, threshold
        )

        expected_oecm = np.array([True, False, True, False])
        expected_classical = np.array([False, True, False, True])

        np.testing.assert_array_equal(oecm_mask, expected_oecm)
        np.testing.assert_array_equal(classical_mask, expected_classical)

    def test_group_c_flag_nan_handled(self):
        """Test that NaN scores produce False in both masks."""
        scores = np.array([0.5, np.nan, 0.05])
        threshold = 0.10

        oecm_mask, classical_mask = criteria_manager.check_use_presence(
            scores, threshold
        )

        # NaN should be False in both masks
        assert oecm_mask[1] == False
        assert classical_mask[1] == False


class TestComputeGroupScore:
    """Tests for compute_group_score function."""

    def test_group_score_geometric_method(self):
        """Test group score computation with geometric method."""
        arrays = {
            'criterion_a': np.array([0.8, 0.6]),
            'criterion_b': np.array([0.5, 0.9])
        }
        weights = {'criterion_a': 0.6, 'criterion_b': 0.4}

        result = criteria_manager.compute_group_score(
            arrays, weights, method='geometric'
        )

        # Same as geometric mean test
        expected_0 = 0.8 ** 0.6 * 0.5 ** 0.4
        expected_1 = 0.6 ** 0.6 * 0.9 ** 0.4

        np.testing.assert_almost_equal(result[0], expected_0, decimal=4)
        np.testing.assert_almost_equal(result[1], expected_1, decimal=4)

    def test_group_score_owa_method(self):
        """Test group score computation with OWA method."""
        arrays = {
            'criterion_a': np.array([0.8]),
            'criterion_b': np.array([0.3])
        }
        weights = {'criterion_a': 0.5, 'criterion_b': 0.5}

        # alpha=0 -> minimum (raw value, not weighted)
        result = criteria_manager.compute_group_score(
            arrays, weights, method='owa', alpha=0.0
        )

        # With alpha=0, should return minimum = 0.3
        assert result[0] == pytest.approx(0.3, abs=0.01)

    def test_group_score_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        arrays = {'a': np.array([0.5])}
        weights = {'a': 1.0}

        with pytest.raises(ValueError, match="Method must be"):
            criteria_manager.compute_group_score(arrays, weights, method='wlc')

    def test_group_score_mismatched_names_raises(self):
        """Test that mismatched criterion names raise ValueError."""
        arrays = {'a': np.array([0.5]), 'b': np.array([0.5])}
        weights = {'a': 0.5, 'c': 0.5}  # 'c' not in arrays

        with pytest.raises(ValueError, match="Criterion name mismatch"):
            criteria_manager.compute_group_score(arrays, weights, method='geometric')


class TestRecodeLanduse:
    """Tests for recode_landuse function."""

    def test_recode_landuse_compatible_class(self):
        """Test that compatible classes receive correct score."""
        landuse = np.array([[31, 23]])  # Forest (3.1), Pasture (2.3)
        table = {
            "3.1": {"status": "compatible", "score": 0.85, "label": "Forests"},
            "2.3": {"status": "compatible", "score": 0.75, "label": "Pastures"}
        }

        scores = criteria_manager.recode_landuse(landuse, table)

        np.testing.assert_almost_equal(scores[0, 0], 0.85, decimal=2)
        np.testing.assert_almost_equal(scores[0, 1], 0.75, decimal=2)

    def test_recode_landuse_eliminatory_class(self):
        """Test that eliminatory classes receive score 0."""
        landuse = np.array([[11, 31]])  # Urban (1.1), Forest (3.1)
        table = {
            "1.1": {"status": "eliminatory", "score": None, "label": "Urban"},
            "3.1": {"status": "compatible", "score": 0.85, "label": "Forests"}
        }

        scores = criteria_manager.recode_landuse(landuse, table)

        assert scores[0, 0] == 0.0  # Eliminatory
        np.testing.assert_almost_equal(scores[0, 1], 0.85, decimal=2)

    def test_recode_landuse_unknown_class(self):
        """Test that unknown classes receive score 0."""
        landuse = np.array([[99]])  # Unknown code
        table = {
            "3.1": {"status": "compatible", "score": 0.85}
        }

        scores = criteria_manager.recode_landuse(landuse, table)

        assert scores[0, 0] == 0.0


class TestLoadCriteriaConfig:
    """Tests for load_criteria_config function."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid configuration file."""
        config_content = """
inter_group_weights:
  W_A: 0.5
  W_B: 0.15
  W_C: 0.35

group_a_weights:
  ecosystem_condition: 0.45
  regulating_es: 0.35
  low_pressure: 0.20

group_b_weights:
  cultural_es: 1.0

group_c_weights:
  provisioning_es: 0.6
  compatible_landuse: 0.4

aggregation:
  default_method: geometric
  default_alpha: 0.25

eliminatory:
  max_anthropogenic_pressure: 150.0

use_presence:
  min_group_c_score: 0.10
"""
        config_file = tmp_path / "criteria_defaults.yaml"
        config_file.write_text(config_content)

        config = criteria_manager.load_criteria_config(str(config_file))

        assert config['inter_group_weights']['W_A'] == 0.5
        assert config['aggregation']['default_method'] == 'geometric'
        assert config['eliminatory']['max_anthropogenic_pressure'] == 150.0

    def test_load_missing_file_raises(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            criteria_manager.load_criteria_config("/nonexistent/path.yaml")

    def test_load_missing_keys_raises(self, tmp_path):
        """Test that missing required keys raise ValueError."""
        config_content = """
inter_group_weights:
  W_A: 0.5
# Missing other required keys
"""
        config_file = tmp_path / "incomplete.yaml"
        config_file.write_text(config_content)

        with pytest.raises(ValueError, match="Missing required"):
            criteria_manager.load_criteria_config(str(config_file))


class TestFullPipeline:
    """Tests for compute_favourability full pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample input data for pipeline tests."""
        shape = (10, 10)
        rng = np.random.default_rng(42)

        return {
            'ecosystem_condition': rng.uniform(0.3, 0.9, shape),
            'regulating_es': rng.uniform(0.2, 0.8, shape),
            'cultural_es': rng.uniform(0.1, 0.7, shape),
            'provisioning_es': rng.uniform(0.2, 0.7, shape),
            'anthropogenic_pressure': rng.uniform(20, 100, shape),
            'landuse': rng.choice([23, 31, 41], shape)  # Compatible classes
        }

    @pytest.fixture
    def sample_weights(self):
        """Create sample weights for pipeline tests."""
        return {
            'inter_group_weights': {'W_A': 0.5, 'W_B': 0.15, 'W_C': 0.35},
            'group_a_weights': {
                'ecosystem_condition': 0.45,
                'regulating_es': 0.35,
                'low_pressure': 0.20
            },
            'group_b_weights': {'cultural_es': 1.0},
            'group_c_weights': {
                'provisioning_es': 0.6,
                'compatible_landuse': 0.4
            }
        }

    def test_full_pipeline_output_keys_present(self, sample_data, sample_weights):
        """Test that pipeline returns all required output keys."""
        result = mce_engine.compute_favourability(
            ecosystem_condition=sample_data['ecosystem_condition'],
            regulating_es=sample_data['regulating_es'],
            cultural_es=sample_data['cultural_es'],
            provisioning_es=sample_data['provisioning_es'],
            anthropogenic_pressure=sample_data['anthropogenic_pressure'],
            landuse=sample_data['landuse'],
            weights=sample_weights,
            method='geometric'
        )

        assert 'score' in result
        assert 'oecm_mask' in result
        assert 'classical_pa_mask' in result
        assert 'eliminatory_mask' in result

    def test_full_pipeline_score_range_zero_to_one(self, sample_data, sample_weights):
        """Test that scores are in [0, 1] range."""
        result = mce_engine.compute_favourability(
            ecosystem_condition=sample_data['ecosystem_condition'],
            regulating_es=sample_data['regulating_es'],
            cultural_es=sample_data['cultural_es'],
            provisioning_es=sample_data['provisioning_es'],
            anthropogenic_pressure=sample_data['anthropogenic_pressure'],
            landuse=sample_data['landuse'],
            weights=sample_weights,
            method='geometric'
        )

        score = result['score']
        valid_scores = score[~np.isnan(score)]

        assert np.all(valid_scores >= 0.0)
        assert np.all(valid_scores <= 1.0)

    def test_full_pipeline_nan_where_eliminated(self, sample_weights):
        """Test that eliminated pixels have NaN score."""
        shape = (5, 5)

        # Create data with some pixels that will be eliminated
        data = {
            'ecosystem_condition': np.full(shape, 0.5),
            'regulating_es': np.full(shape, 0.5),
            'cultural_es': np.full(shape, 0.5),
            'provisioning_es': np.full(shape, 0.5),
            'anthropogenic_pressure': np.array([
                [50, 50, 200, 50, 50],  # One high pressure
                [50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50]
            ], dtype=float),
            'landuse': np.array([
                [31, 31, 31, 31, 31],
                [31, 11, 31, 31, 31],  # One urban pixel
                [31, 31, 31, 31, 31],
                [31, 31, 31, 31, 31],
                [31, 31, 31, 31, 31]
            ])
        }

        result = mce_engine.compute_favourability(
            ecosystem_condition=data['ecosystem_condition'],
            regulating_es=data['regulating_es'],
            cultural_es=data['cultural_es'],
            provisioning_es=data['provisioning_es'],
            anthropogenic_pressure=data['anthropogenic_pressure'],
            landuse=data['landuse'],
            weights=sample_weights,
            method='geometric'
        )

        score = result['score']
        eliminatory_mask = result['eliminatory_mask']

        # High pressure pixel should be eliminated
        assert eliminatory_mask[0, 2] == False
        assert np.isnan(score[0, 2])

        # Urban pixel should be eliminated
        assert eliminatory_mask[1, 1] == False
        assert np.isnan(score[1, 1])

        # Other pixels should have valid scores
        assert eliminatory_mask[0, 0] == True
        assert not np.isnan(score[0, 0])

    def test_full_pipeline_owa_method(self, sample_data, sample_weights):
        """Test pipeline with OWA method."""
        result = mce_engine.compute_favourability(
            ecosystem_condition=sample_data['ecosystem_condition'],
            regulating_es=sample_data['regulating_es'],
            cultural_es=sample_data['cultural_es'],
            provisioning_es=sample_data['provisioning_es'],
            anthropogenic_pressure=sample_data['anthropogenic_pressure'],
            landuse=sample_data['landuse'],
            weights=sample_weights,
            method='owa',
            alpha=0.25
        )

        score = result['score']
        valid_scores = score[~np.isnan(score)]

        # Should produce valid scores
        assert len(valid_scores) > 0
        assert np.all(valid_scores >= 0.0)
        assert np.all(valid_scores <= 1.0)

    def test_full_pipeline_invalid_method_raises(self, sample_data, sample_weights):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Method must be"):
            mce_engine.compute_favourability(
                ecosystem_condition=sample_data['ecosystem_condition'],
                regulating_es=sample_data['regulating_es'],
                cultural_es=sample_data['cultural_es'],
                provisioning_es=sample_data['provisioning_es'],
                anthropogenic_pressure=sample_data['anthropogenic_pressure'],
                landuse=sample_data['landuse'],
                weights=sample_weights,
                method='wlc'  # Forbidden method
            )


class TestAnalyticalVerification:
    """Additional analytical verification tests with manually computed values."""

    def test_geometric_mean_three_criteria(self):
        """Test geometric mean with three criteria.

        Given:
            arrays = [0.6, 0.8, 0.4]
            weights = [0.5, 0.3, 0.2]

        Expected:
            S = 0.6^0.5 * 0.8^0.3 * 0.4^0.2
            S = 0.7746 * 0.9322 * 0.8326
            S = 0.6013
        """
        arrays = [np.array([0.6]), np.array([0.8]), np.array([0.4])]
        weights = [0.5, 0.3, 0.2]

        result = mce_engine.weighted_geometric_mean(arrays, weights)

        expected = 0.6 ** 0.5 * 0.8 ** 0.3 * 0.4 ** 0.2
        np.testing.assert_almost_equal(result[0], expected, decimal=4)

    def test_owa_three_criteria_alpha_025(self):
        """Test OWA with alpha=0.25.

        Yager OWA with alpha=0.25 computes position weights as:
        v_j = (j/n)^(1-alpha) - ((j-1)/n)^(1-alpha)

        For n=3, alpha=0.25 (exponent=0.75):
        v_1 = (1/3)^0.75 = 0.4387 (weight on max)
        v_2 = (2/3)^0.75 - (1/3)^0.75 = 0.2991
        v_3 = 1 - (2/3)^0.75 = 0.2622 (weight on min)

        The first position (highest value) gets most weight, so result
        is between mean and max.
        """
        arrays = [np.array([0.9]), np.array([0.5]), np.array([0.1])]
        weights = [1/3, 1/3, 1/3]

        result = mce_engine.yager_owa(arrays, weights, alpha=0.25)

        # Mean = (0.9 + 0.5 + 0.1) / 3 = 0.5
        # Max = 0.9
        # Min = 0.1
        # Expected = 0.9 * 0.4387 + 0.5 * 0.2991 + 0.1 * 0.2622 = 0.5706

        expected = 0.9 * 0.4387 + 0.5 * 0.2991 + 0.1 * 0.2622
        np.testing.assert_almost_equal(result[0], expected, decimal=3)

        # Result should be between min and max
        assert result[0] > 0.1, "Result should be above minimum"
        assert result[0] < 0.9, "Result should be below maximum"

    def test_geometric_mean_single_criterion(self):
        """Test geometric mean with single criterion returns that criterion."""
        arrays = [np.array([0.7, 0.3])]
        weights = [1.0]

        result = mce_engine.weighted_geometric_mean(arrays, weights)

        np.testing.assert_array_almost_equal(result, arrays[0])

    def test_owa_single_criterion(self):
        """Test OWA with single criterion returns that criterion."""
        arrays = [np.array([0.7, 0.3])]
        weights = [1.0]

        result = mce_engine.yager_owa(arrays, weights, alpha=0.5)

        np.testing.assert_array_almost_equal(result, arrays[0], decimal=4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

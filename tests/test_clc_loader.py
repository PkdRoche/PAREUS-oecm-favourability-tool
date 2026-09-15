"""Tests for Corine Land Cover (CLC) loader module."""

import pytest
import numpy as np
from shapely.geometry import box
import rasterio
from rasterio.transform import from_origin
import yaml
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'modules'))

from utils.clc_loader import (
    load_clc,
    reclassify_clc,
    load_and_reclassify_clc,
    get_clc_legend
)


# Synthetic CLC raster parameters
CLC_SHAPE = (50, 50)  # 50×50 pixels
CLC_RESOLUTION = 100.0  # 100m
CLC_ORIGIN = (3000000.0, 2000000.0)  # EPSG:3035 origin (top-left)
CLC_CRS = "EPSG:3035"

# Realistic CLC codes (Level 3)
# Format: XYZ where X=level1, Y=level2_digit, Z=level3_digit
REALISTIC_CLC_CODES = [
    111,  # Continuous urban fabric
    112,  # Discontinuous urban fabric
    121,  # Industrial or commercial units
    211,  # Non-irrigated arable land
    231,  # Pastures
    311,  # Broad-leaved forest
    312,  # Coniferous forest
    313,  # Mixed forest
    321,  # Natural grasslands
    322,  # Moors and heathland
    411,  # Inland marshes
    511   # Water courses
]


@pytest.fixture
def synthetic_clc_raster(tmp_path):
    """
    Create a synthetic CLC raster (50×50 pixels, EPSG:3035, 100m resolution).

    Returns path to the created GeoTIFF.
    Uses realistic CLC level 3 codes with nodata=0.
    """
    # Generate random CLC values from realistic codes
    np.random.seed(42)
    clc_data = np.random.choice(
        REALISTIC_CLC_CODES,
        size=CLC_SHAPE,
        replace=True
    ).astype('int16')

    # Inject nodata patch (5×5 pixels in corner)
    clc_data[0:5, 0:5] = 0  # CLC nodata value

    # Create raster profile
    transform = from_origin(CLC_ORIGIN[0], CLC_ORIGIN[1],
                           CLC_RESOLUTION, CLC_RESOLUTION)

    profile = {
        'driver': 'GTiff',
        'dtype': 'int16',
        'width': CLC_SHAPE[1],
        'height': CLC_SHAPE[0],
        'count': 1,
        'crs': CLC_CRS,
        'transform': transform,
        'nodata': 0
    }

    # Write raster
    raster_path = tmp_path / 'synthetic_clc.tif'
    with rasterio.open(raster_path, 'w', **profile) as dst:
        dst.write(clc_data, 1)

    return str(raster_path)


@pytest.fixture
def synthetic_clc_raster_wrong_crs(tmp_path):
    """
    Create a synthetic CLC raster in EPSG:4326 (wrong CRS) for error testing.
    """
    np.random.seed(42)
    clc_data = np.random.choice(
        REALISTIC_CLC_CODES,
        size=CLC_SHAPE,
        replace=True
    ).astype('int16')

    # Create raster in WGS84 (EPSG:4326)
    transform = from_origin(2.0, 48.0, 0.001, 0.001)  # Paris area

    profile = {
        'driver': 'GTiff',
        'dtype': 'int16',
        'width': CLC_SHAPE[1],
        'height': CLC_SHAPE[0],
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': transform,
        'nodata': 0
    }

    raster_path = tmp_path / 'synthetic_clc_wrong_crs.tif'
    with rasterio.open(raster_path, 'w', **profile) as dst:
        dst.write(clc_data, 1)

    return str(raster_path)


@pytest.fixture
def study_area_geom():
    """
    Return a shapely box covering the entire synthetic CLC raster extent.

    Raster extent:
    - Origin (top-left): (3000000, 2000000)
    - Width: 50 pixels × 100m = 5000m
    - Height: 50 pixels × 100m = 5000m
    - Bottom-right: (3005000, 1995000)
    """
    xmin = CLC_ORIGIN[0]
    ymax = CLC_ORIGIN[1]
    xmax = xmin + (CLC_SHAPE[1] * CLC_RESOLUTION)
    ymin = ymax - (CLC_SHAPE[0] * CLC_RESOLUTION)

    return box(xmin, ymin, xmax, ymax)


@pytest.fixture
def sample_reclassification_table():
    """
    Return a dictionary mapping CLC codes to scores [0.0, 1.0].

    Maps the 12 realistic CLC codes used in synthetic data.
    """
    return {
        111: 0.0,   # Continuous urban fabric (eliminatory)
        112: 0.1,   # Discontinuous urban fabric (low)
        121: 0.0,   # Industrial (eliminatory)
        211: 0.2,   # Arable land (low)
        231: 0.75,  # Pastures (compatible)
        311: 0.90,  # Broad-leaved forest (high)
        312: 0.85,  # Coniferous forest (high)
        313: 0.88,  # Mixed forest (high)
        321: 0.80,  # Natural grasslands (high)
        322: 0.78,  # Moors and heathland (high)
        411: 0.95,  # Inland marshes (very high)
        511: 0.70   # Water courses (compatible)
    }


@pytest.fixture
def sample_reclassification_config(tmp_path, sample_reclassification_table):
    """
    Create a temporary YAML config file with reclassification table.

    Returns path to the config file.
    """
    config_data = {
        'reclassification': sample_reclassification_table
    }

    config_path = tmp_path / 'clc_reclassification.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)

    return str(config_path)


# ============================================================================
# Test: load_clc
# ============================================================================

def test_load_clc_returns_correct_shape(synthetic_clc_raster, study_area_geom):
    """Test that load_clc returns array with expected shape."""
    array, profile = load_clc(
        synthetic_clc_raster,
        study_area_geom,
        target_resolution=CLC_RESOLUTION
    )

    # Output should match the study area extent at target resolution
    # Since study area covers the entire raster, shape should be CLC_SHAPE
    assert array.shape == CLC_SHAPE, (
        f"Expected shape {CLC_SHAPE}, got {array.shape}"
    )

    # Profile should be a dictionary
    assert hasattr(profile, '__getitem__'), "Profile must be a mapping"

    # Profile should contain key metadata
    assert 'crs' in profile
    assert 'transform' in profile
    assert 'width' in profile
    assert 'height' in profile


def test_load_clc_raises_on_wrong_crs(synthetic_clc_raster_wrong_crs, study_area_geom):
    """Test that load_clc raises ValueError when raster is not EPSG:3035."""
    with pytest.raises(ValueError, match="EPSG:3035"):
        load_clc(
            synthetic_clc_raster_wrong_crs,
            study_area_geom,
            target_resolution=CLC_RESOLUTION
        )


def test_load_clc_profile_is_epsg3035(synthetic_clc_raster, study_area_geom):
    """Test that output profile CRS is EPSG:3035."""
    array, profile = load_clc(
        synthetic_clc_raster,
        study_area_geom,
        target_resolution=CLC_RESOLUTION
    )

    # CRS should be EPSG:3035
    assert profile['crs'].to_string() == 'EPSG:3035', (
        f"Expected CRS EPSG:3035, got {profile['crs'].to_string()}"
    )


# ============================================================================
# Test: reclassify_clc
# ============================================================================

def test_reclassify_clc_scores_in_range(synthetic_clc_raster,
                                         study_area_geom,
                                         sample_reclassification_table):
    """Test that reclassified scores are in [0.0, 1.0] range."""
    # Load raw CLC data
    array, profile = load_clc(
        synthetic_clc_raster,
        study_area_geom,
        target_resolution=CLC_RESOLUTION
    )

    # Reclassify
    reclassified = reclassify_clc(array, sample_reclassification_table)

    # Extract non-NaN values
    valid_scores = reclassified[~np.isnan(reclassified)]

    # All valid scores should be in [0.0, 1.0]
    assert (valid_scores >= 0.0).all(), "Found scores < 0.0"
    assert (valid_scores <= 1.0).all(), "Found scores > 1.0"


def test_reclassify_clc_unknown_codes_are_nan(sample_reclassification_table):
    """Test that pixels with unmapped CLC codes become NaN."""
    # Create array with one known code and one unknown code
    array = np.array([
        [311, 999],  # 311=forest(mapped), 999=unknown(unmapped)
        [211, 888]   # 211=arable(mapped), 888=unknown(unmapped)
    ], dtype='int16')

    # Reclassify
    reclassified = reclassify_clc(array, sample_reclassification_table)

    # Known codes should have valid scores
    assert not np.isnan(reclassified[0, 0]), "Code 311 should be mapped"
    assert not np.isnan(reclassified[1, 0]), "Code 211 should be mapped"

    # Unknown codes should be NaN
    assert np.isnan(reclassified[0, 1]), "Code 999 should be NaN"
    assert np.isnan(reclassified[1, 1]), "Code 888 should be NaN"


def test_reclassify_clc_nodata_is_nan(sample_reclassification_table):
    """Test that pixels with nodata values (0 or 128) become NaN."""
    # Create array with nodata values
    array = np.array([
        [311, 0],    # 0 = standard CLC nodata
        [211, 128]   # 128 = alternative CLC nodata
    ], dtype='int16')

    # Reclassify
    reclassified = reclassify_clc(array, sample_reclassification_table)

    # Valid code should have score
    assert not np.isnan(reclassified[0, 0]), "Code 311 should be mapped"
    assert not np.isnan(reclassified[1, 0]), "Code 211 should be mapped"

    # Nodata values should be NaN
    assert np.isnan(reclassified[0, 1]), "Nodata value 0 should be NaN"
    assert np.isnan(reclassified[1, 1]), "Nodata value 128 should be NaN"


def test_reclassify_clc_output_dtype_float32(synthetic_clc_raster,
                                               study_area_geom,
                                               sample_reclassification_table):
    """Test that reclassified output is float32."""
    # Load raw CLC data
    array, profile = load_clc(
        synthetic_clc_raster,
        study_area_geom,
        target_resolution=CLC_RESOLUTION
    )

    # Reclassify
    reclassified = reclassify_clc(array, sample_reclassification_table)

    # Check dtype
    assert reclassified.dtype == np.float32, (
        f"Expected dtype float32, got {reclassified.dtype}"
    )


# ============================================================================
# Test: get_clc_legend
# ============================================================================

def test_get_clc_legend_has_44_classes():
    """Test that CLC legend dictionary has exactly 44 classes (CLC level 3)."""
    legend = get_clc_legend()

    # CLC has 44 level-3 classes
    assert len(legend) == 44, (
        f"Expected 44 CLC classes, got {len(legend)}"
    )


def test_get_clc_legend_all_codes_valid():
    """Test that all CLC legend keys are integers in valid range [111-523]."""
    legend = get_clc_legend()

    for code in legend.keys():
        # All codes should be integers
        assert isinstance(code, int), f"Code {code} is not an integer"

        # All codes should be in range 111-523
        assert 111 <= code <= 523, (
            f"Code {code} outside valid CLC range [111-523]"
        )

        # First digit should be 1-5 (level 1: artificial, agri, forest, wetland, water)
        first_digit = code // 100
        assert 1 <= first_digit <= 5, (
            f"Code {code} has invalid first digit {first_digit}"
        )


# ============================================================================
# Test: load_and_reclassify_clc (end-to-end)
# ============================================================================

def test_load_and_reclassify_end_to_end(synthetic_clc_raster,
                                         study_area_geom,
                                         sample_reclassification_config):
    """Test the full pipeline: load + reclassify in one function."""
    # Run full pipeline
    reclassified, profile = load_and_reclassify_clc(
        synthetic_clc_raster,
        study_area_geom,
        sample_reclassification_config,
        target_resolution=CLC_RESOLUTION
    )

    # Output should be float32 array
    assert reclassified.dtype == np.float32, (
        f"Expected dtype float32, got {reclassified.dtype}"
    )

    # Extract valid (non-NaN) scores
    valid_scores = reclassified[~np.isnan(reclassified)]

    # Should have at least some valid scores
    assert len(valid_scores) > 0, "No valid scores found in output"

    # All valid scores should be in [0.0, 1.0]
    assert (valid_scores >= 0.0).all(), "Found scores < 0.0"
    assert (valid_scores <= 1.0).all(), "Found scores > 1.0"

    # Profile should be valid
    assert hasattr(profile, '__getitem__')
    assert profile['crs'].to_string() == 'EPSG:3035'
    assert profile['dtype'] == 'float32'


def test_load_and_reclassify_with_nested_config(synthetic_clc_raster,
                                                  study_area_geom,
                                                  tmp_path):
    """Test that load_and_reclassify_clc handles nested config format."""
    # Create nested config format (like real clc_reclassification.yaml)
    nested_config = {
        'reclassification': {
            111: {'score': 0.0, 'label': 'Continuous urban fabric'},
            112: {'score': 0.1, 'label': 'Discontinuous urban fabric'},
            311: {'score': 0.90, 'label': 'Broad-leaved forest'},
            411: {'score': 0.95, 'label': 'Inland marshes'}
        }
    }

    config_path = tmp_path / 'nested_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(nested_config, f)

    # Run full pipeline
    reclassified, profile = load_and_reclassify_clc(
        synthetic_clc_raster,
        study_area_geom,
        str(config_path),
        target_resolution=CLC_RESOLUTION
    )

    # Should succeed and produce valid output
    assert reclassified.dtype == np.float32
    valid_scores = reclassified[~np.isnan(reclassified)]
    assert len(valid_scores) > 0
    assert (valid_scores >= 0.0).all()
    assert (valid_scores <= 1.0).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

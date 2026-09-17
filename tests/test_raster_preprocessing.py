"""
Tests for raster preprocessing functions.

All tests use synthetic data from tests/synthetic_data/.
Run generate_synthetic_data.py first if synthetic_data/ does not exist.
"""

import pytest
import numpy as np
import rasterio
from pathlib import Path
from shapely.geometry import box

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.module2_favourability.raster_preprocessing import (
    load_raster,
    reproject_raster,
    resample_raster,
    align_rasters,
    apply_nodata_mask,
    normalize_linear,
    normalize_sigmoid,
    normalize_gaussian,
    normalize_layer,
    derive_grid_from_geometry,
    validate_and_rescale_layer,
    validate_and_rescale_all_layers
)


# Test data directory
SYNTHETIC_DATA_DIR = Path(__file__).parent / "synthetic_data"


@pytest.fixture
def ecosystem_condition_path():
    """Path to ecosystem condition test raster."""
    return str(SYNTHETIC_DATA_DIR / "ecosystem_condition.tif")


@pytest.fixture
def regulating_es_path():
    """Path to regulating ES test raster."""
    return str(SYNTHETIC_DATA_DIR / "regulating_es.tif")


@pytest.fixture
def cultural_es_path():
    """Path to cultural ES test raster."""
    return str(SYNTHETIC_DATA_DIR / "cultural_es.tif")


@pytest.fixture
def provisioning_es_path():
    """Path to provisioning ES test raster."""
    return str(SYNTHETIC_DATA_DIR / "provisioning_es.tif")


@pytest.fixture
def anthropogenic_pressure_path():
    """Path to anthropogenic pressure test raster."""
    return str(SYNTHETIC_DATA_DIR / "anthropogenic_pressure.tif")


@pytest.fixture
def land_use_path():
    """Path to land use test raster."""
    return str(SYNTHETIC_DATA_DIR / "land_use.tif")


def test_load_raster_returns_array_and_profile(ecosystem_condition_path):
    """Test that load_raster returns array and profile."""
    array, profile = load_raster(ecosystem_condition_path)

    assert isinstance(array, np.ndarray)
    assert isinstance(profile, dict)
    assert 'crs' in profile
    assert 'transform' in profile
    assert 'width' in profile
    assert 'height' in profile


def test_load_raster_shape_matches_file(ecosystem_condition_path):
    """Test that loaded array shape matches file dimensions."""
    array, profile = load_raster(ecosystem_condition_path)

    # Verify against direct rasterio read
    with rasterio.open(ecosystem_condition_path) as src:
        expected_shape = src.shape
        expected_dtype = src.dtypes[0]

    assert array.shape == expected_shape
    assert array.shape == (profile['height'], profile['width'])
    assert str(array.dtype) == expected_dtype


def test_reproject_output_crs_matches_target(ecosystem_condition_path):
    """Test that reprojected raster has correct target CRS."""
    array, profile = load_raster(ecosystem_condition_path)

    # Reproject from EPSG:3035 to EPSG:4326
    target_crs = "EPSG:4326"
    reprojected_array, reprojected_profile = reproject_raster(
        array, profile, target_crs
    )

    assert str(reprojected_profile['crs']).upper() == target_crs.upper()
    assert isinstance(reprojected_array, np.ndarray)
    assert reprojected_array.size > 0


def test_reproject_invalid_crs_raises_value_error(ecosystem_condition_path):
    """Test that invalid CRS raises ValueError."""
    array, profile = load_raster(ecosystem_condition_path)

    with pytest.raises(ValueError, match="Invalid target CRS"):
        reproject_raster(array, profile, "INVALID:CRS:12345")


def test_align_rasters_identical_extents():
    """Test that aligned rasters have identical extents."""
    # Load multiple rasters
    paths = {
        "ecosystem_condition": str(SYNTHETIC_DATA_DIR / "ecosystem_condition.tif"),
        "regulating_es": str(SYNTHETIC_DATA_DIR / "regulating_es.tif"),
        "cultural_es": str(SYNTHETIC_DATA_DIR / "cultural_es.tif")
    }

    raster_dict = {}
    for name, path in paths.items():
        array, profile = load_raster(path)
        raster_dict[name] = (array, profile)

    # Align rasters
    aligned = align_rasters(raster_dict)

    # Get reference bounds (first sorted name)
    ref_name = sorted(raster_dict.keys())[0]
    ref_profile = aligned[ref_name][1]
    ref_bounds = rasterio.transform.array_bounds(
        ref_profile['height'],
        ref_profile['width'],
        ref_profile['transform']
    )

    # Check all rasters have same bounds
    for name, (array, profile) in aligned.items():
        bounds = rasterio.transform.array_bounds(
            profile['height'],
            profile['width'],
            profile['transform']
        )
        assert np.allclose(bounds, ref_bounds), f"{name} bounds differ from reference"


def test_align_rasters_identical_resolution():
    """Test that aligned rasters have identical resolution."""
    # Load multiple rasters
    paths = {
        "ecosystem_condition": str(SYNTHETIC_DATA_DIR / "ecosystem_condition.tif"),
        "regulating_es": str(SYNTHETIC_DATA_DIR / "regulating_es.tif"),
        "cultural_es": str(SYNTHETIC_DATA_DIR / "cultural_es.tif")
    }

    raster_dict = {}
    for name, path in paths.items():
        array, profile = load_raster(path)
        raster_dict[name] = (array, profile)

    # Align rasters
    aligned = align_rasters(raster_dict)

    # Get reference resolution
    ref_name = sorted(raster_dict.keys())[0]
    ref_profile = aligned[ref_name][1]
    ref_res_x = abs(ref_profile['transform'][0])
    ref_res_y = abs(ref_profile['transform'][4])

    # Check all rasters have same resolution
    for name, (array, profile) in aligned.items():
        res_x = abs(profile['transform'][0])
        res_y = abs(profile['transform'][4])
        assert np.isclose(res_x, ref_res_x), f"{name} x-resolution differs from reference"
        assert np.isclose(res_y, ref_res_y), f"{name} y-resolution differs from reference"


def test_normalize_linear_range_zero_to_one(ecosystem_condition_path):
    """Test that linear normalization produces values in [0, 1]."""
    array, profile = load_raster(ecosystem_condition_path)

    # Apply nodata mask first
    array_masked = apply_nodata_mask(array, profile.get('nodata'))

    # Normalize
    vmin, vmax = 0.0, 1.0
    normalized = normalize_linear(array_masked, vmin, vmax, invert=False)

    # Check range (ignoring NaN)
    valid_values = normalized[~np.isnan(normalized)]
    assert np.all(valid_values >= 0.0), "Some values below 0"
    assert np.all(valid_values <= 1.0), "Some values above 1"
    assert len(valid_values) > 0, "No valid values after normalization"


def test_normalize_linear_inverted_monotone_decreasing(anthropogenic_pressure_path):
    """Test that inverted linear normalization is monotone decreasing."""
    array, profile = load_raster(anthropogenic_pressure_path)

    # Apply nodata mask
    array_masked = apply_nodata_mask(array, profile.get('nodata'))

    # Get min/max for normalization
    vmin = np.nanmin(array_masked)
    vmax = np.nanmax(array_masked)

    # Normalize with invert=True
    normalized = normalize_linear(array_masked, vmin, vmax, invert=True)

    # Check that higher input values produce lower output values
    # Find non-NaN values
    valid_mask = ~np.isnan(array_masked)
    if np.sum(valid_mask) < 2:
        pytest.skip("Not enough valid values to test monotonicity")

    # Get a sample of input-output pairs
    input_vals = array_masked[valid_mask]
    output_vals = normalized[valid_mask]

    # Check monotonicity: sort by input and verify output is decreasing
    sorted_indices = np.argsort(input_vals)
    sorted_output = output_vals[sorted_indices]

    # Compute differences (should be <= 0 for monotone decreasing)
    diffs = np.diff(sorted_output)
    # Allow small numerical errors
    assert np.all(diffs <= 1e-10), "Inverted normalization is not monotone decreasing"


def test_normalize_sigmoid_range_zero_to_one(ecosystem_condition_path):
    """Test that sigmoid normalization produces values in [0, 1]."""
    array, profile = load_raster(ecosystem_condition_path)

    # Apply nodata mask
    array_masked = apply_nodata_mask(array, profile.get('nodata'))

    # Normalize with sigmoid
    inflection = 0.5
    slope = 8.0
    normalized = normalize_sigmoid(array_masked, inflection, slope)

    # Check range (ignoring NaN)
    valid_values = normalized[~np.isnan(normalized)]
    assert np.all(valid_values >= 0.0), "Some sigmoid values below 0"
    assert np.all(valid_values <= 1.0), "Some sigmoid values above 1"
    assert len(valid_values) > 0, "No valid values after sigmoid normalization"


def test_normalize_gaussian_maximum_at_mean(provisioning_es_path):
    """Test that Gaussian normalization has maximum at specified mean."""
    array, profile = load_raster(provisioning_es_path)

    # Apply nodata mask
    array_masked = apply_nodata_mask(array, profile.get('nodata'))

    # Normalize with Gaussian
    mean = 0.45
    std = 0.20
    normalized = normalize_gaussian(array_masked, mean, std)

    # The normalized value should be maximum (=1.0) at input values equal to mean
    # Find values closest to mean
    valid_mask = ~np.isnan(array_masked)
    if np.sum(valid_mask) == 0:
        pytest.skip("No valid values to test")

    distances = np.abs(array_masked - mean)
    distances[~valid_mask] = np.inf

    closest_idx = np.unravel_index(np.argmin(distances), distances.shape)
    max_normalized_value = normalized[closest_idx]

    # Should be close to 1.0
    assert max_normalized_value >= 0.9, f"Maximum value {max_normalized_value} not close to 1.0"

    # Also verify that the maximum normalized value occurs at an input near the mean
    max_norm_idx = np.unravel_index(np.nanargmax(normalized), normalized.shape)
    input_at_max = array_masked[max_norm_idx]
    assert np.abs(input_at_max - mean) < std, f"Maximum occurs at {input_at_max}, far from mean {mean}"


def test_normalize_gaussian_symmetric_around_mean(provisioning_es_path):
    """Test that Gaussian normalization is symmetric around mean."""
    array, profile = load_raster(provisioning_es_path)

    # Apply nodata mask
    array_masked = apply_nodata_mask(array, profile.get('nodata'))

    mean = 0.45
    std = 0.20

    # Create test points symmetric around mean
    test_points = np.array([mean - 0.1, mean, mean + 0.1])
    expected = normalize_gaussian(test_points, mean, std)

    # Values at mean - delta and mean + delta should be equal
    assert np.isclose(expected[0], expected[2]), "Gaussian not symmetric around mean"
    # Value at mean should be maximum (1.0)
    assert np.isclose(expected[1], 1.0), "Value at mean not equal to 1.0"


def test_apply_nodata_mask_converts_to_nan(ecosystem_condition_path):
    """Test that nodata values are converted to np.nan."""
    array, profile = load_raster(ecosystem_condition_path)

    nodata_value = profile.get('nodata')
    if nodata_value is None:
        pytest.skip("No nodata value in test file")

    # Count nodata values before masking
    nodata_count_before = np.sum(array == nodata_value)

    # Apply mask
    masked = apply_nodata_mask(array, nodata_value)

    # Count NaN values after masking
    nan_count_after = np.sum(np.isnan(masked))

    # Should be equal
    assert nan_count_after == nodata_count_before, \
        f"Expected {nodata_count_before} NaN values, got {nan_count_after}"

    # No original nodata values should remain
    assert np.sum(masked == nodata_value) == 0, "Some nodata values not converted to NaN"


def test_apply_nodata_mask_none_nodata_is_safe():
    """Test that apply_nodata_mask handles None nodata gracefully."""
    # Create test array
    test_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # Apply with None nodata
    masked = apply_nodata_mask(test_array, None)

    # Should return array unchanged (except dtype conversion)
    assert masked.shape == test_array.shape
    assert np.all(masked == test_array)
    assert np.sum(np.isnan(masked)) == 0, "Unexpected NaN values when nodata is None"


def test_normalize_layer_dispatcher_linear():
    """Test normalize_layer dispatcher with linear transformation."""
    test_array = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float64)

    params = {
        'type': 'linear',
        'vmin': 0.0,
        'vmax': 1.0,
        'invert': False
    }

    normalized = normalize_layer(test_array, 'test_layer', params)

    expected = np.array([[0.0, 0.5], [1.0, 1.0]])  # 1.5 clipped to 1.0
    assert np.allclose(normalized, expected)


def test_normalize_layer_dispatcher_sigmoid():
    """Test normalize_layer dispatcher with sigmoid transformation."""
    test_array = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float64)

    params = {
        'type': 'sigmoid',
        'inflection': 0.5,
        'slope': 8.0
    }

    normalized = normalize_layer(test_array, 'test_layer', params)

    # Check range
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)
    # Value at inflection should be ~0.5
    assert np.isclose(normalized[0, 1], 0.5, atol=0.01)


def test_normalize_layer_dispatcher_gaussian():
    """Test normalize_layer dispatcher with gaussian transformation."""
    test_array = np.array([[0.0, 0.45], [0.9, 1.5]], dtype=np.float64)

    params = {
        'type': 'gaussian',
        'mean': 0.45,
        'std': 0.20
    }

    normalized = normalize_layer(test_array, 'test_layer', params)

    # Check range
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)
    # Value at mean should be 1.0
    assert np.isclose(normalized[0, 1], 1.0)


def test_normalize_layer_dispatcher_inverted_linear():
    """Test normalize_layer dispatcher with inverted_linear transformation."""
    test_array = np.array([[0.0, 50.0], [100.0, 150.0]], dtype=np.float64)

    params = {
        'type': 'inverted_linear',
        'vmin': 0.0,
        'vmax': 100.0
    }

    normalized = normalize_layer(test_array, 'test_layer', params)

    # Check that higher values map to lower normalized values
    assert normalized[0, 0] > normalized[1, 0]  # 0.0 > 100.0 after inversion
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)


def test_normalize_layer_missing_type_raises_error():
    """Test that missing 'type' in params raises ValueError."""
    test_array = np.array([[0.0, 1.0]], dtype=np.float64)

    params = {'vmin': 0.0, 'vmax': 1.0}  # Missing 'type'

    with pytest.raises(ValueError, match="missing 'type' key"):
        normalize_layer(test_array, 'test_layer', params)


def test_normalize_layer_unknown_type_raises_error():
    """Test that unknown transformation type raises ValueError."""
    test_array = np.array([[0.0, 1.0]], dtype=np.float64)

    params = {'type': 'unknown_transform'}

    with pytest.raises(ValueError, match="Unknown transformation type"):
        normalize_layer(test_array, 'test_layer', params)


def test_resample_raster_changes_resolution():
    """Test that resample_raster produces correct output resolution."""
    # Load a test raster
    path = str(SYNTHETIC_DATA_DIR / "ecosystem_condition.tif")
    array, profile = load_raster(path)

    # Original resolution is 100m, resample to 200m
    target_resolution = 200.0
    resampled_array, resampled_profile = resample_raster(
        array, profile, target_resolution, method='bilinear'
    )

    # Check resolution
    res_x = abs(resampled_profile['transform'][0])
    res_y = abs(resampled_profile['transform'][4])

    assert np.isclose(res_x, target_resolution), f"X resolution {res_x} != {target_resolution}"
    assert np.isclose(res_y, target_resolution), f"Y resolution {res_y} != {target_resolution}"

    # Shape should be approximately halved
    expected_width = profile['width'] // 2
    expected_height = profile['height'] // 2

    assert abs(resampled_profile['width'] - expected_width) <= 1
    assert abs(resampled_profile['height'] - expected_height) <= 1


def test_derive_grid_from_geometry():
    """Test that derive_grid_from_geometry produces correct grid parameters."""
    # Create a test geometry (100km x 100km square in EPSG:3035 coordinates)
    # Center around arbitrary European coordinates
    minx, miny = 4000000, 3000000
    maxx, maxy = 4100000, 3100000
    test_geom = box(minx, miny, maxx, maxy)

    resolution = 100.0
    profile = derive_grid_from_geometry(test_geom, resolution=resolution, crs="EPSG:3035")

    # Check basic properties
    assert profile['crs'] == "EPSG:3035"
    assert profile['width'] == 1000  # 100km / 100m
    assert profile['height'] == 1000
    assert profile['dtype'] == 'float64'
    assert profile['count'] == 1

    # Check transform resolution
    transform = profile['transform']
    assert abs(transform[0]) == resolution  # pixel width
    assert abs(transform[4]) == resolution  # pixel height (negative)

    # Check bounds alignment (should snap to grid)
    bounds_minx = transform[2]
    bounds_maxy = transform[5]
    assert bounds_minx % resolution == 0  # Aligned to grid
    assert bounds_maxy % resolution == 0


def test_align_rasters_with_nuts2_geometry():
    """Test that align_rasters with geometry produces grid matching derived grid."""
    # Load test rasters
    paths = {
        "ecosystem_condition": str(SYNTHETIC_DATA_DIR / "ecosystem_condition.tif"),
        "regulating_es": str(SYNTHETIC_DATA_DIR / "regulating_es.tif"),
        "cultural_es": str(SYNTHETIC_DATA_DIR / "cultural_es.tif")
    }

    raster_dict = {}
    for name, path in paths.items():
        array, profile = load_raster(path)
        raster_dict[name] = (array, profile)

    # Create a study area geometry that covers part of the raster extent
    # Get bounds from first raster
    first_array, first_profile = list(raster_dict.values())[0]
    first_bounds = rasterio.transform.array_bounds(
        first_profile['height'],
        first_profile['width'],
        first_profile['transform']
    )

    # Create geometry covering central 80% of first raster
    margin = 0.1
    width = first_bounds[2] - first_bounds[0]
    height = first_bounds[3] - first_bounds[1]
    study_minx = first_bounds[0] + width * margin
    study_miny = first_bounds[1] + height * margin
    study_maxx = first_bounds[2] - width * margin
    study_maxy = first_bounds[3] - height * margin
    study_area_geom = box(study_minx, study_miny, study_maxx, study_maxy)

    resolution = 100.0
    crs = "EPSG:3035"

    # Align with study area geometry
    aligned = align_rasters(
        raster_dict,
        study_area_geom=study_area_geom,
        resolution=resolution,
        crs=crs
    )

    # Derive expected grid
    expected_profile = derive_grid_from_geometry(study_area_geom, resolution=resolution, crs=crs)

    # Check all aligned rasters match expected grid
    for name, (array, profile) in aligned.items():
        assert profile['width'] == expected_profile['width'], f"{name} width mismatch"
        assert profile['height'] == expected_profile['height'], f"{name} height mismatch"
        assert profile['crs'] == expected_profile['crs'], f"{name} CRS mismatch"

        # Check transform matches (within floating point tolerance)
        for i in range(6):
            assert np.isclose(profile['transform'][i], expected_profile['transform'][i]), \
                f"{name} transform[{i}] mismatch"

        # Check array shape matches profile
        assert array.shape == (profile['height'], profile['width']), f"{name} array shape mismatch"


def test_align_rasters_warns_on_low_coverage(caplog):
    """Test that align_rasters logs warning when layer covers <80% of study area."""
    import logging

    # Load a test raster
    path = str(SYNTHETIC_DATA_DIR / "ecosystem_condition.tif")
    array, profile = load_raster(path)

    # Create a study area geometry much larger than the raster
    # This will cause low coverage
    raster_bounds = rasterio.transform.array_bounds(
        profile['height'],
        profile['width'],
        profile['transform']
    )

    # Create study area 3x larger than raster
    center_x = (raster_bounds[0] + raster_bounds[2]) / 2
    center_y = (raster_bounds[1] + raster_bounds[3]) / 2
    width = (raster_bounds[2] - raster_bounds[0]) * 3
    height = (raster_bounds[3] - raster_bounds[1]) * 3

    large_study_area = box(
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2
    )

    raster_dict = {"test_layer": (array, profile)}

    # Set logging level to capture warnings
    with caplog.at_level(logging.WARNING):
        aligned = align_rasters(
            raster_dict,
            study_area_geom=large_study_area,
            resolution=100.0,
            crs="EPSG:3035"
        )

    # Check that warning was logged
    warning_messages = [record.message for record in caplog.records if record.levelname == "WARNING"]
    assert any("covers only" in msg and "80%" in msg for msg in warning_messages), \
        "Expected low coverage warning not found in logs"


def test_validate_no_rescale_needed():
    """Test validate_and_rescale_layer with values already in [0, 1]."""
    # Create test array in [0, 1] range
    test_array = np.array([
        [0.0, 0.3, 0.5],
        [0.7, 0.9, 1.0],
        [np.nan, 0.4, 0.6]
    ], dtype=np.float64)

    test_profile = {
        'dtype': 'float64',
        'crs': 'EPSG:3035',
        'transform': rasterio.transform.from_bounds(0, 0, 300, 300, 3, 3),
        'width': 3,
        'height': 3,
        'count': 1
    }

    array_out, profile_out, report = validate_and_rescale_layer(
        array=test_array,
        profile=test_profile,
        criterion_key='ecosystem_condition'
    )

    # Should not rescale
    assert report['rescaled'] is False
    assert report['method'] == 'none'
    assert report['expected_min'] == 0.0
    assert report['expected_max'] == 1.0
    assert report['original_min'] == 0.0
    assert report['original_max'] == 1.0

    # Output should be float32
    assert profile_out['dtype'] == 'float32'
    assert array_out.dtype == np.float32

    # Values should be unchanged (except dtype)
    assert np.allclose(array_out, test_array, equal_nan=True)


def test_validate_rescales_percentage():
    """Test validate_and_rescale_layer with percentage values [0, 100]."""
    # Create test array with percentage values
    test_array = np.array([
        [0.0, 30.0, 50.0],
        [70.0, 90.0, 100.0],
        [np.nan, 40.0, 60.0]
    ], dtype=np.float64)

    test_profile = {
        'dtype': 'float64',
        'crs': 'EPSG:3035',
        'transform': rasterio.transform.from_bounds(0, 0, 300, 300, 3, 3),
        'width': 3,
        'height': 3,
        'count': 1
    }

    array_out, profile_out, report = validate_and_rescale_layer(
        array=test_array,
        profile=test_profile,
        criterion_key='regulating_es'
    )

    # Should rescale
    assert report['rescaled'] is True
    assert report['method'] == 'linear_rescale'
    assert report['original_min'] == 0.0
    assert report['original_max'] == 100.0

    # Output should be in [0, 1]
    valid_values = array_out[~np.isnan(array_out)]
    assert np.all(valid_values >= 0.0)
    assert np.all(valid_values <= 1.0)

    # Check specific values (divided by 100)
    assert np.isclose(array_out[0, 1], 0.30)
    assert np.isclose(array_out[1, 2], 1.00)


def test_validate_rescales_arbitrary_range():
    """Test validate_and_rescale_layer with arbitrary value range."""
    # Create test array with values in [-5, 200]
    test_array = np.array([
        [-5.0, 50.0, 100.0],
        [150.0, 200.0, np.nan],
        [25.0, 75.0, 125.0]
    ], dtype=np.float64)

    test_profile = {
        'dtype': 'float64',
        'crs': 'EPSG:3035',
        'transform': rasterio.transform.from_bounds(0, 0, 300, 300, 3, 3),
        'width': 3,
        'height': 3,
        'count': 1
    }

    array_out, profile_out, report = validate_and_rescale_layer(
        array=test_array,
        profile=test_profile,
        criterion_key='cultural_es'
    )

    # Should rescale
    assert report['rescaled'] is True
    assert report['method'] == 'linear_rescale'
    # original_min/max recorded BEFORE negative masking
    assert report['original_min'] == -5.0
    assert report['original_max'] == 200.0

    # Negative value (-5) must be NaN in output (masked as NoData)
    assert np.isnan(array_out[0, 0])

    # Output valid values should be in [0, 1]
    valid_values = array_out[~np.isnan(array_out)]
    assert np.all(valid_values >= 0.0)
    assert np.all(valid_values <= 1.0)

    # Max value (200) should map to 1.0
    assert np.isclose(array_out[1, 1], 1.0)

    # Check intermediate value: 50 should map to (50 - (-5)) / (200 - (-5)) = 55/205
    expected_mid = (50.0 - (-5.0)) / (200.0 - (-5.0))
    assert np.isclose(array_out[0, 1], expected_mid)


def test_validate_pressure_warns_if_normalised():
    """Test validate_and_rescale_layer warns for pre-normalized pressure."""
    # Create test array with values already in [0, 1]
    test_array = np.array([
        [0.0, 0.3, 0.5],
        [0.7, 0.9, 1.0],
        [np.nan, 0.4, 0.6]
    ], dtype=np.float64)

    test_profile = {
        'dtype': 'float64',
        'crs': 'EPSG:3035',
        'transform': rasterio.transform.from_bounds(0, 0, 300, 300, 3, 3),
        'width': 3,
        'height': 3,
        'count': 1
    }

    array_out, profile_out, report = validate_and_rescale_layer(
        array=test_array,
        profile=test_profile,
        criterion_key='anthropogenic_pressure'
    )

    # Should not rescale but should warn
    assert report['rescaled'] is False
    assert report['method'] == 'warn_only'
    assert report['warning'] is not None
    assert 'already normalised' in report['warning'].lower()

    # Values should be unchanged (except dtype)
    assert np.allclose(array_out, test_array, equal_nan=True)


def test_validate_landuse_warns_if_float():
    """Test validate_and_rescale_layer warns for float land use values."""
    # Create test array with float values in (0, 1] — avoid 0.0 which is CLC NoData sentinel
    test_array = np.array([
        [0.1, 0.3, 0.5],
        [0.7, 0.9, 1.0],
        [np.nan, 0.4, 0.6]
    ], dtype=np.float64)

    test_profile = {
        'dtype': 'float64',
        'crs': 'EPSG:3035',
        'transform': rasterio.transform.from_bounds(0, 0, 300, 300, 3, 3),
        'width': 3,
        'height': 3,
        'count': 1
    }

    array_out, profile_out, report = validate_and_rescale_layer(
        array=test_array,
        profile=test_profile,
        criterion_key='landuse'
    )

    # Should not rescale but should warn
    assert report['rescaled'] is False
    assert report['method'] == 'warn_only'
    assert report['warning'] is not None
    assert 'pre-normalised' in report['warning'].lower()

    # Values should be unchanged (except dtype)
    assert np.allclose(array_out, test_array, equal_nan=True)


def test_validate_landuse_integer_codes():
    """Test validate_and_rescale_layer with valid integer land use codes."""
    # Create test array with valid CLC codes
    test_array = np.array([
        [111, 211, 312],
        [523, 142, 231],
        [0, 324, 411]  # 0 = NoData in CLC
    ], dtype=np.int16)

    test_profile = {
        'dtype': 'int16',
        'crs': 'EPSG:3035',
        'transform': rasterio.transform.from_bounds(0, 0, 300, 300, 3, 3),
        'width': 3,
        'height': 3,
        'count': 1,
        'nodata': 0
    }

    # First apply nodata mask
    test_array_masked = test_array.astype(np.float64)
    test_array_masked[test_array == 0] = np.nan

    array_out, profile_out, report = validate_and_rescale_layer(
        array=test_array_masked,
        profile=test_profile,
        criterion_key='landuse'
    )

    # Should not rescale and no warning
    assert report['rescaled'] is False
    assert report['warning'] is None
    assert report['expected_min'] == 111
    assert report['expected_max'] == 523


def test_validate_and_rescale_all_layers():
    """Test validate_and_rescale_all_layers batch function."""
    # Create test dictionary with multiple layers
    raster_dict = {
        'ecosystem_condition': (
            np.array([[0.0, 0.5, 1.0]], dtype=np.float64),
            {'dtype': 'float64', 'crs': 'EPSG:3035', 'width': 3, 'height': 1, 'count': 1,
             'transform': rasterio.transform.from_bounds(0, 0, 300, 100, 3, 1)}
        ),
        'regulating_es': (
            np.array([[0.0, 50.0, 100.0]], dtype=np.float64),  # Percentage
            {'dtype': 'float64', 'crs': 'EPSG:3035', 'width': 3, 'height': 1, 'count': 1,
             'transform': rasterio.transform.from_bounds(0, 0, 300, 100, 3, 1)}
        ),
        'anthropogenic_pressure': (
            np.array([[10.0, 50.0, 150.0]], dtype=np.float64),  # hab/km²
            {'dtype': 'float64', 'crs': 'EPSG:3035', 'width': 3, 'height': 1, 'count': 1,
             'transform': rasterio.transform.from_bounds(0, 0, 300, 100, 3, 1)}
        )
    }

    updated_dict, reports = validate_and_rescale_all_layers(raster_dict)

    # Should have 3 reports
    assert len(reports) == 3

    # Check ecosystem_condition - no rescale
    assert reports[0]['criterion'] == 'ecosystem_condition'
    assert reports[0]['rescaled'] is False

    # Check regulating_es - should rescale from percentage
    assert reports[1]['criterion'] == 'regulating_es'
    assert reports[1]['rescaled'] is True
    assert reports[1]['method'] == 'linear_rescale'

    # Check anthropogenic_pressure - no rescale, no warning for raw values
    assert reports[2]['criterion'] == 'anthropogenic_pressure'
    assert reports[2]['rescaled'] is False

    # Check updated arrays
    assert 'ecosystem_condition' in updated_dict
    assert 'regulating_es' in updated_dict
    assert 'anthropogenic_pressure' in updated_dict

    # Regulating ES should be rescaled to [0, 1]
    reg_es_array = updated_dict['regulating_es'][0]
    assert np.allclose(reg_es_array, [0.0, 0.5, 1.0])


def test_validate_invalid_criterion_raises_error():
    """Test that invalid criterion_key raises ValueError."""
    test_array = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    test_profile = {
        'dtype': 'float64',
        'crs': 'EPSG:3035',
        'width': 3,
        'height': 1,
        'count': 1,
        'transform': rasterio.transform.from_bounds(0, 0, 300, 100, 3, 1)
    }

    with pytest.raises(ValueError, match="Invalid criterion_key"):
        validate_and_rescale_layer(test_array, test_profile, 'invalid_criterion')


def test_validate_all_nan_raises_error():
    """Test that array with all NaN values raises ValueError."""
    test_array = np.array([[np.nan, np.nan, np.nan]], dtype=np.float64)
    test_profile = {
        'dtype': 'float64',
        'crs': 'EPSG:3035',
        'width': 3,
        'height': 1,
        'count': 1,
        'transform': rasterio.transform.from_bounds(0, 0, 300, 100, 3, 1)
    }

    with pytest.raises(ValueError, match="only NaN/NoData values"):
        validate_and_rescale_layer(test_array, test_profile, 'ecosystem_condition')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
End-to-end integration test suite for OECM Favourability Tool.

Tests the complete pipeline from NUTS2 loading through MCE computation
and export using only synthetic data (no real network calls).

Phase 7 — Integration test gate.
"""

import pytest
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from pathlib import Path
from shapely.geometry import box
from unittest.mock import patch, MagicMock
import tempfile
import os

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Import modules to test
from modules.utils import nuts2_loader
from modules.module1_protected_areas import (
    wdpa_loader,
    coverage_stats,
    representativity,
    gap_analysis,
    zonal_stats
)
from modules.module2_favourability import (
    raster_preprocessing,
    mce_engine,
    export
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_nuts2_gdf():
    """Create a synthetic NUTS2 region GeoDataFrame."""
    # Create a polygon in EPSG:3035 coordinates (valid ETRS89/LAEA)
    # Using coordinates around the synthetic data origin (3000000, 2000000)
    polygon = box(
        3000000.0,  # minx
        2000000.0,  # miny
        3010000.0,  # maxx (10 km extent)
        2010000.0   # maxy (10 km extent)
    )

    gdf = gpd.GeoDataFrame(
        {
            'NUTS_ID': ['TEST01'],
            'CNTR_CODE': ['TS'],
            'NUTS_NAME': ['Test Region'],
            'LEVL_CODE': [2]
        },
        geometry=[polygon],
        crs='EPSG:3035'
    )

    return gdf


@pytest.fixture
def synthetic_wdpa_gdf(synthetic_nuts2_gdf):
    """Create synthetic WDPA protected areas overlapping NUTS2 region."""
    # Get NUTS2 bounds
    bounds = synthetic_nuts2_gdf.total_bounds
    minx, miny, maxx, maxy = bounds

    # Create 3 protected areas with different IUCN classes
    pa1 = box(minx + 1000, miny + 1000, minx + 3000, miny + 3000)  # Strict core
    pa2 = box(minx + 3500, miny + 3500, minx + 6000, miny + 6000)  # Regulatory
    pa3 = box(minx + 6500, miny + 1000, minx + 9000, miny + 3500)  # Contractual

    gdf = gpd.GeoDataFrame(
        {
            'WDPA_PID': [100001, 100002, 100003],
            'NAME': ['Test PA 1', 'Test PA 2', 'Test PA 3'],
            'IUCN_MAX': ['Ia', 'III', 'V'],
            'DESIG': ['National Park', 'Natural Monument', 'Protected Landscape'],
            'DESIG_TYPE': ['National', 'National', 'National'],
            'STATUS': ['Designated', 'Designated', 'Designated'],
            'GIS_AREA': [400.0, 625.0, 625.0],
            'protection_class': ['strict_core', 'regulatory', 'contractual']
        },
        geometry=[pa1, pa2, pa3],
        crs='EPSG:3035'
    )

    return gdf


@pytest.fixture
def synthetic_raster_paths():
    """Return paths to synthetic raster data."""
    data_dir = PROJECT_ROOT / "tests" / "synthetic_data"

    return {
        'ecosystem_condition': str(data_dir / "ecosystem_condition.tif"),
        'regulating_es': str(data_dir / "regulating_es.tif"),
        'cultural_es': str(data_dir / "cultural_es.tif"),
        'provisioning_es': str(data_dir / "provisioning_es.tif"),
        'anthropogenic_pressure': str(data_dir / "anthropogenic_pressure.tif"),
        'land_use': str(data_dir / "land_use.tif")
    }


@pytest.fixture
def synthetic_ecosystem_layer(synthetic_nuts2_gdf):
    """Create synthetic ecosystem type layer for representativity analysis."""
    # Get NUTS2 bounds
    bounds = synthetic_nuts2_gdf.total_bounds
    minx, miny, maxx, maxy = bounds

    # Create 3 ecosystem types covering different parts of the region
    forest = box(minx, miny, minx + 5000, maxy)
    wetland = box(minx + 5000, miny, maxx, miny + 5000)
    grassland = box(minx + 5000, miny + 5000, maxx, maxy)

    gdf = gpd.GeoDataFrame(
        {
            'ecosystem_type': ['forests', 'wetlands', 'grasslands']
        },
        geometry=[forest, wetland, grassland],
        crs='EPSG:3035'
    )

    return gdf


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Step 1: NUTS2 Loader
# ============================================================================

def test_step1_nuts2_loader_mock(synthetic_nuts2_gdf):
    """Test NUTS2 loader with mocked Eurostat API call."""
    # Mock the requests.get call to avoid real network access
    with patch('modules.utils.nuts2_loader.requests.get') as mock_get:
        # Configure mock response. load_nuts2() reads the already-downloaded
        # bytes via gpd.read_file(io.BytesIO(response.content)) — .content
        # must be real bytes so io.BytesIO() doesn't choke on a MagicMock,
        # even though gpd.read_file itself is mocked below.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Mock gpd.read_file to return our synthetic data
        with patch('modules.utils.nuts2_loader.gpd.read_file') as mock_read:
            mock_read.return_value = synthetic_nuts2_gdf

            # Call the function
            result = nuts2_loader.load_nuts2(year=2021, scale="20M")

            # Verify result
            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == 1
            assert result.crs.to_epsg() == 3035
            assert 'NUTS_ID' in result.columns
            assert 'LEVL_CODE' in result.columns
            assert result.iloc[0]['LEVL_CODE'] == 2


def test_step1_nuts2_helper_functions(synthetic_nuts2_gdf):
    """Test NUTS2 helper functions (get_countries, get_nuts2_for_country, etc)."""
    # Test get_countries
    countries = nuts2_loader.get_countries(synthetic_nuts2_gdf)
    assert isinstance(countries, list)
    assert 'TS' in countries

    # Test get_nuts2_for_country
    filtered = nuts2_loader.get_nuts2_for_country(synthetic_nuts2_gdf, 'TS')
    assert isinstance(filtered, gpd.GeoDataFrame)
    assert len(filtered) == 1

    # Test get_nuts2_geometry
    geom = nuts2_loader.get_nuts2_geometry(synthetic_nuts2_gdf, 'TEST01')
    assert geom is not None
    assert geom.is_valid


# ============================================================================
# Step 2: WDPA Loader
# ============================================================================

def test_step2_wdpa_loader_synthetic(synthetic_wdpa_gdf):
    """Test WDPA loader with synthetic data."""
    # Test filter_to_extent — pass GeoSeries (with CRS) so function detects EPSG:3035
    territory_geom = box(3000000, 2000000, 3010000, 2010000)
    territory = gpd.GeoSeries([territory_geom], crs="EPSG:3035")

    clipped = wdpa_loader.filter_to_extent(synthetic_wdpa_gdf, territory)

    assert isinstance(clipped, gpd.GeoDataFrame)
    assert len(clipped) > 0
    assert clipped.crs == 'EPSG:3035'
    assert all(clipped.intersects(territory.iloc[0]))


def test_step2_wdpa_classify_iucn(synthetic_wdpa_gdf):
    """Test IUCN classification."""
    # Simple classification table
    classification = {
        'classes': {
            'strict_core': {'iucn_cats': ['Ia', 'Ib', 'II'], 'desig_keywords': []},
            'regulatory': {'iucn_cats': ['III', 'IV'], 'desig_keywords': []},
            'contractual': {'iucn_cats': ['V', 'VI'], 'desig_keywords': ['Natura 2000']},
            'unassigned': {'iucn_cats': ['Not Reported'], 'desig_keywords': []}
        }
    }

    # Remove protection_class to test classification
    wdpa = synthetic_wdpa_gdf.copy()
    if 'protection_class' in wdpa.columns:
        wdpa = wdpa.drop(columns=['protection_class'])

    classified = wdpa_loader.classify_iucn(wdpa, classification)

    assert 'protection_class' in classified.columns
    assert classified.iloc[0]['protection_class'] == 'strict_core'
    assert classified.iloc[1]['protection_class'] == 'regulatory'
    assert classified.iloc[2]['protection_class'] == 'contractual'


# ============================================================================
# Step 3: Coverage Stats
# ============================================================================

def test_step3_coverage_by_class(synthetic_wdpa_gdf, synthetic_nuts2_gdf):
    """Test coverage statistics computation."""
    # Calculate territory area
    territory_area_ha = synthetic_nuts2_gdf.geometry.area.sum() / 10000.0

    # Compute coverage
    coverage = coverage_stats.coverage_by_class(synthetic_wdpa_gdf, territory_area_ha)

    assert isinstance(coverage, pd.DataFrame)
    assert 'protection_class' in coverage.columns
    assert 'area_ha' in coverage.columns
    assert 'pct_territory' in coverage.columns
    assert 'n_sites' in coverage.columns

    # Check TOTAL row exists
    assert 'TOTAL' in coverage['protection_class'].values

    # Check all percentages are non-negative
    assert all(coverage['pct_territory'] >= 0)


def test_step3_compute_net_area(synthetic_wdpa_gdf, synthetic_nuts2_gdf):
    """Test net area computation with deduplication."""
    territory = synthetic_nuts2_gdf.geometry.iloc[0]

    net_area = coverage_stats.compute_net_area(synthetic_wdpa_gdf, territory)

    assert isinstance(net_area, float)
    assert net_area > 0

    # Net area should be less than or equal to sum of individual areas
    # (due to potential overlaps)
    sum_individual = synthetic_wdpa_gdf['GIS_AREA'].sum()
    assert net_area <= sum_individual * 1.1  # Allow 10% margin for rounding


# ============================================================================
# Step 4: Representativity
# ============================================================================

def test_step4_representativity_index(synthetic_wdpa_gdf, synthetic_ecosystem_layer):
    """Test representativity index computation."""
    # Cross PAs with ecosystems
    coverage_df = representativity.cross_with_ecosystem_types(
        synthetic_wdpa_gdf,
        synthetic_ecosystem_layer,
        type_column='ecosystem_type'
    )

    assert isinstance(coverage_df, pd.DataFrame)
    assert 'ecosystem_type' in coverage_df.columns
    assert 'pa_class' in coverage_df.columns
    assert 'area_ha' in coverage_df.columns

    # Compute territory totals
    territory_totals = {
        'forests': synthetic_ecosystem_layer[
            synthetic_ecosystem_layer['ecosystem_type'] == 'forests'
        ].geometry.area.sum() / 10000.0,
        'wetlands': synthetic_ecosystem_layer[
            synthetic_ecosystem_layer['ecosystem_type'] == 'wetlands'
        ].geometry.area.sum() / 10000.0,
        'grasslands': synthetic_ecosystem_layer[
            synthetic_ecosystem_layer['ecosystem_type'] == 'grasslands'
        ].geometry.area.sum() / 10000.0
    }

    # Compute RI
    ri_df = representativity.representativity_index(
        coverage_df,
        territory_totals,
        target_threshold=0.30
    )

    assert isinstance(ri_df, pd.DataFrame)
    assert 'ecosystem_type' in ri_df.columns
    assert 'RI' in ri_df.columns
    assert 'gap_ha' in ri_df.columns

    # RI should be in [0, 1]
    assert all(ri_df['RI'] >= 0)
    assert all(ri_df['RI'] <= 1.0)


def test_step4_propose_weights(synthetic_wdpa_gdf, synthetic_ecosystem_layer):
    """Test weight proposal from representativity deficits."""
    # Setup coverage
    coverage_df = representativity.cross_with_ecosystem_types(
        synthetic_wdpa_gdf,
        synthetic_ecosystem_layer,
        type_column='ecosystem_type'
    )

    territory_totals = {
        'forests': 5000 * 10000 / 10000.0,
        'wetlands': 2500 * 10000 / 10000.0,
        'grasslands': 2500 * 10000 / 10000.0
    }

    ri_df = representativity.representativity_index(
        coverage_df,
        territory_totals,
        target_threshold=0.30
    )

    # Propose weights
    criterion_mapping = {
        'ecosystem_condition': 'all',
        'regulating_es': 'wetlands',
        'low_pressure': 'all'
    }

    weights = representativity.propose_group_a_weights(ri_df, criterion_mapping)

    assert isinstance(weights, dict)
    assert len(weights) == 3
    assert all(isinstance(v, float) for v in weights.values())

    # Weights must sum to 1.0 ± 1e-6
    weight_sum = sum(weights.values())
    assert abs(weight_sum - 1.0) < 1e-6


# ============================================================================
# Step 5: Gap Analysis
# ============================================================================

def test_step5_strict_gaps(synthetic_wdpa_gdf, synthetic_nuts2_gdf):
    """Test strict gap identification."""
    territory = synthetic_nuts2_gdf.geometry.iloc[0]

    gaps = gap_analysis.strict_gaps(synthetic_wdpa_gdf, territory)

    assert isinstance(gaps, gpd.GeoDataFrame)
    assert gaps.crs == 'EPSG:3035'

    if len(gaps) > 0:
        assert 'gap_type' in gaps.columns
        assert all(gaps['gap_type'] == 'strict')


def test_step5_export_gap_raster(synthetic_wdpa_gdf, synthetic_nuts2_gdf, temp_output_dir, synthetic_raster_paths):
    """Test gap mask rasterization."""
    territory = synthetic_nuts2_gdf.geometry.iloc[0]
    gaps = gap_analysis.strict_gaps(synthetic_wdpa_gdf, territory)

    # Load reference raster profile
    with rasterio.open(synthetic_raster_paths['ecosystem_condition']) as src:
        reference_profile = src.profile

    # Export gap masks
    gap_layers = {'strict_gaps': gaps}

    output_paths = gap_analysis.export_gap_masks_as_raster(
        gap_layers,
        reference_profile,
        temp_output_dir
    )

    assert isinstance(output_paths, dict)

    if len(gaps) > 0:
        assert 'strict_gaps' in output_paths
        assert Path(output_paths['strict_gaps']).exists()

        # Verify output raster
        with rasterio.open(output_paths['strict_gaps']) as dst:
            assert dst.crs == reference_profile['crs']
            assert dst.shape == (reference_profile['height'], reference_profile['width'])


# ============================================================================
# Step 6: Zonal Stats
# ============================================================================

def test_step6_zonal_stats_by_pa_class(synthetic_wdpa_gdf, synthetic_raster_paths):
    """Test zonal statistics computation."""
    # Select subset of rasters
    raster_subset = {
        'ecosystem_condition': synthetic_raster_paths['ecosystem_condition'],
        'regulating_es': synthetic_raster_paths['regulating_es']
    }

    stats_df = zonal_stats.zonal_stats_by_pa_class(
        synthetic_wdpa_gdf,
        raster_subset,
        nodata=-9999.0
    )

    assert isinstance(stats_df, pd.DataFrame)
    assert 'criterion' in stats_df.columns
    assert 'iucn_cat' in stats_df.columns
    assert 'mean' in stats_df.columns
    assert 'median' in stats_df.columns
    assert 'std' in stats_df.columns
    assert 'min' in stats_df.columns
    assert 'max' in stats_df.columns
    assert 'pixel_count' in stats_df.columns

    # Should have stats for each criterion × PA class combination
    assert len(stats_df) >= len(raster_subset)


# ============================================================================
# Step 7: Raster Preprocessing
# ============================================================================

def test_step7_load_and_normalize_rasters(synthetic_raster_paths):
    """Test raster loading and normalization."""
    # Load ecosystem condition raster
    array, profile = raster_preprocessing.load_raster(
        synthetic_raster_paths['ecosystem_condition']
    )

    assert isinstance(array, np.ndarray)
    assert isinstance(profile, dict)
    assert array.shape == (100, 100)
    assert profile['crs'] is not None

    # Apply nodata mask
    array_masked = raster_preprocessing.apply_nodata_mask(array, profile['nodata'])
    assert array_masked.dtype == np.float64

    # Test linear normalization
    normalized = raster_preprocessing.normalize_linear(array_masked, 0.0, 1.0, invert=False)
    assert normalized.shape == array.shape
    assert np.nanmin(normalized) >= 0.0
    assert np.nanmax(normalized) <= 1.0


def test_step7_align_rasters(synthetic_raster_paths):
    """Test raster alignment to common grid."""
    # Load multiple rasters
    raster_dict = {}
    for name in ['ecosystem_condition', 'regulating_es']:
        array, profile = raster_preprocessing.load_raster(synthetic_raster_paths[name])
        raster_dict[name] = (array, profile)

    # Align rasters
    aligned = raster_preprocessing.align_rasters(raster_dict)

    assert len(aligned) == len(raster_dict)

    # Check all aligned rasters have same shape and CRS
    reference_array, reference_profile = aligned['ecosystem_condition']
    for name, (array, profile) in aligned.items():
        assert array.shape == reference_array.shape
        assert profile['crs'] == reference_profile['crs']
        assert profile['transform'] == reference_profile['transform']


# ============================================================================
# Step 8: MCE Engine
# ============================================================================

def test_step8_weighted_geometric_mean():
    """Test weighted geometric mean aggregation."""
    arrays = [
        np.array([[0.8, 0.6], [0.5, 0.9]]),
        np.array([[0.5, 0.7], [0.4, 0.8]])
    ]
    weights = [0.6, 0.4]

    result = mce_engine.weighted_geometric_mean(arrays, weights)

    assert result.shape == arrays[0].shape
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)

    # Manual verification for one pixel
    # result[0,0] = 0.8^0.6 * 0.5^0.4 ≈ 0.6822
    expected_00 = 0.8**0.6 * 0.5**0.4
    assert abs(result[0, 0] - expected_00) < 1e-4


def test_step8_yager_owa():
    """Test Yager OWA aggregation."""
    arrays = [
        np.array([[0.8, 0.6]]),
        np.array([[0.5, 0.7]]),
        np.array([[0.2, 0.9]])
    ]
    weights = [1/3, 1/3, 1/3]

    # Test alpha=0 (minimum)
    result_min = mce_engine.yager_owa(arrays, weights, alpha=0.0)
    assert result_min[0, 0] == 0.2  # Minimum of 0.8, 0.5, 0.2
    assert result_min[0, 1] == 0.6  # Minimum of 0.6, 0.7, 0.9

    # Test alpha=1 (maximum)
    result_max = mce_engine.yager_owa(arrays, weights, alpha=1.0)
    assert result_max[0, 0] == 0.8  # Maximum of 0.8, 0.5, 0.2
    assert result_max[0, 1] == 0.9  # Maximum of 0.6, 0.7, 0.9


def test_step8_compute_favourability_pipeline(synthetic_raster_paths):
    """Test full MCE favourability pipeline."""
    # Load all required layers
    eco_cond, _ = raster_preprocessing.load_raster(synthetic_raster_paths['ecosystem_condition'])
    reg_es, _ = raster_preprocessing.load_raster(synthetic_raster_paths['regulating_es'])
    cult_es, _ = raster_preprocessing.load_raster(synthetic_raster_paths['cultural_es'])
    prov_es, _ = raster_preprocessing.load_raster(synthetic_raster_paths['provisioning_es'])
    pressure, _ = raster_preprocessing.load_raster(synthetic_raster_paths['anthropogenic_pressure'])
    landuse, profile = raster_preprocessing.load_raster(synthetic_raster_paths['land_use'])

    # Prepare weights
    weights = {
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

    # Run MCE
    result = mce_engine.compute_favourability(
        ecosystem_condition=eco_cond,
        regulating_es=reg_es,
        cultural_es=cult_es,
        provisioning_es=prov_es,
        anthropogenic_pressure=pressure,
        landuse=landuse,
        weights=weights,
        method='geometric',
        alpha=0.25
    )

    # Verify output structure
    assert isinstance(result, dict)
    assert 'score' in result
    assert 'oecm_mask' in result
    assert 'classical_pa_mask' in result
    assert 'eliminatory_mask' in result

    # Verify shapes
    assert result['score'].shape == eco_cond.shape
    assert result['oecm_mask'].shape == eco_cond.shape
    assert result['classical_pa_mask'].shape == eco_cond.shape
    assert result['eliminatory_mask'].shape == eco_cond.shape

    # Verify score range (excluding NaN)
    valid_scores = result['score'][~np.isnan(result['score'])]
    if len(valid_scores) > 0:
        assert np.all(valid_scores >= 0.0)
        assert np.all(valid_scores <= 1.0)


# ============================================================================
# Step 9: Export
# ============================================================================

def test_step9_export_geotiff(temp_output_dir, synthetic_raster_paths):
    """Test GeoTIFF export."""
    # Load a raster
    array, profile = raster_preprocessing.load_raster(
        synthetic_raster_paths['ecosystem_condition']
    )

    # Export
    output_path = os.path.join(temp_output_dir, "test_output.tif")
    export.export_geotiff(array, profile, output_path)

    # Verify output exists
    assert Path(output_path).exists()

    # Verify output is valid GeoTIFF
    with rasterio.open(output_path) as src:
        assert src.crs == profile['crs']
        assert src.shape == (profile['height'], profile['width'])
        assert src.dtypes[0] == 'float32'


# ============================================================================
# Full End-to-End Integration Test
# ============================================================================

def test_full_e2e_pipeline(
    synthetic_nuts2_gdf,
    synthetic_wdpa_gdf,
    synthetic_ecosystem_layer,
    synthetic_raster_paths,
    temp_output_dir
):
    """
    Full end-to-end integration test covering all 9 steps.

    This test validates the complete pipeline:
    1. NUTS2 region (mocked)
    2. WDPA PAs (synthetic)
    3. Coverage stats
    4. Representativity analysis
    5. Gap analysis
    6. Zonal stats
    7. Raster preprocessing
    8. MCE engine
    9. Export
    """
    # Step 1: NUTS2 region (use synthetic)
    nuts2_region = synthetic_nuts2_gdf.iloc[0].geometry
    territory_area_ha = synthetic_nuts2_gdf.geometry.area.sum() / 10000.0

    # Step 2: Protected areas (use synthetic)
    pa_gdf = synthetic_wdpa_gdf
    assert len(pa_gdf) == 3
    assert 'protection_class' in pa_gdf.columns

    # Step 3: Coverage statistics
    coverage = coverage_stats.coverage_by_class(pa_gdf, territory_area_ha)
    assert 'TOTAL' in coverage['protection_class'].values
    net_area = coverage_stats.compute_net_area(pa_gdf, nuts2_region)
    assert net_area > 0

    # Step 4: Representativity
    coverage_df = representativity.cross_with_ecosystem_types(
        pa_gdf,
        synthetic_ecosystem_layer,
        type_column='ecosystem_type'
    )

    territory_totals = {
        eco: synthetic_ecosystem_layer[
            synthetic_ecosystem_layer['ecosystem_type'] == eco
        ].geometry.area.sum() / 10000.0
        for eco in synthetic_ecosystem_layer['ecosystem_type'].unique()
    }

    ri_df = representativity.representativity_index(
        coverage_df,
        territory_totals,
        target_threshold=0.30
    )

    criterion_mapping = {
        'ecosystem_condition': 'all',
        'regulating_es': 'wetlands',
        'low_pressure': 'all'
    }

    proposed_weights = representativity.propose_group_a_weights(ri_df, criterion_mapping)
    assert abs(sum(proposed_weights.values()) - 1.0) < 1e-6

    # Step 5: Gap analysis
    gaps = gap_analysis.strict_gaps(pa_gdf, nuts2_region)

    with rasterio.open(synthetic_raster_paths['ecosystem_condition']) as src:
        reference_profile = src.profile

    gap_layers = {'strict_gaps': gaps}
    gap_paths = gap_analysis.export_gap_masks_as_raster(
        gap_layers,
        reference_profile,
        temp_output_dir
    )

    # Step 6: Zonal statistics
    raster_subset = {
        'ecosystem_condition': synthetic_raster_paths['ecosystem_condition'],
        'regulating_es': synthetic_raster_paths['regulating_es']
    }

    zonal_df = zonal_stats.zonal_stats_by_pa_class(
        pa_gdf,
        raster_subset,
        nodata=-9999.0
    )

    assert len(zonal_df) > 0
    assert 'criterion' in zonal_df.columns
    assert 'iucn_cat' in zonal_df.columns

    # Step 7: Raster preprocessing
    raster_dict = {}
    for name in ['ecosystem_condition', 'regulating_es', 'cultural_es']:
        array, profile = raster_preprocessing.load_raster(synthetic_raster_paths[name])
        raster_dict[name] = (array, profile)

    aligned = raster_preprocessing.align_rasters(raster_dict)
    assert len(aligned) == 3

    # Step 8: MCE engine
    eco_cond, _ = raster_preprocessing.load_raster(synthetic_raster_paths['ecosystem_condition'])
    reg_es, _ = raster_preprocessing.load_raster(synthetic_raster_paths['regulating_es'])
    cult_es, _ = raster_preprocessing.load_raster(synthetic_raster_paths['cultural_es'])
    prov_es, _ = raster_preprocessing.load_raster(synthetic_raster_paths['provisioning_es'])
    pressure, _ = raster_preprocessing.load_raster(synthetic_raster_paths['anthropogenic_pressure'])
    landuse, profile = raster_preprocessing.load_raster(synthetic_raster_paths['land_use'])

    weights = {
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

    mce_result = mce_engine.compute_favourability(
        ecosystem_condition=eco_cond,
        regulating_es=reg_es,
        cultural_es=cult_es,
        provisioning_es=prov_es,
        anthropogenic_pressure=pressure,
        landuse=landuse,
        weights=weights,
        method='geometric',
        alpha=0.25
    )

    assert 'score' in mce_result
    assert 'oecm_mask' in mce_result
    assert 'classical_pa_mask' in mce_result
    assert 'eliminatory_mask' in mce_result

    # Step 9: Export
    output_path = os.path.join(temp_output_dir, "favourability_e2e.tif")
    export.export_geotiff(mce_result['score'], profile, output_path)

    # Verify export
    assert Path(output_path).exists()

    with rasterio.open(output_path) as src:
        assert src.crs.to_epsg() == 3035
        assert src.dtypes[0] == 'float32'
        exported_data = src.read(1)
        assert exported_data.shape == (100, 100)

    # All steps completed successfully
    print("\n=== END-TO-END INTEGRATION TEST PASSED ===")
    print("✓ Step 1: NUTS2 loader (mocked)")
    print("✓ Step 2: WDPA loader (synthetic)")
    print(f"✓ Step 3: Coverage stats — {len(coverage)} classes")
    print(f"✓ Step 4: Representativity — proposed weights sum to {sum(proposed_weights.values()):.6f}")
    print(f"✓ Step 5: Gap analysis — exported to {len(gap_paths)} file(s)")
    print(f"✓ Step 6: Zonal stats — {len(zonal_df)} records")
    print(f"✓ Step 7: Raster preprocessing — {len(aligned)} layers aligned")
    print(f"✓ Step 8: MCE engine — score range [{np.nanmin(mce_result['score']):.4f}, {np.nanmax(mce_result['score']):.4f}]")
    print(f"✓ Step 9: Export — {output_path}")
    print("="*50)

"""Tests for zonal statistics module."""

import pytest
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import rasterio
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'modules'))

from module1_protected_areas.zonal_stats import (
    zonal_stats_by_pa_class,
    criterion_coverage_summary
)


# Synthetic data parameters (from generate_synthetic_data.py)
SYNTHETIC_DATA_DIR = Path(__file__).parent / 'synthetic_data'
ORIGIN = (3000000.0, 2000000.0)  # EPSG:3035 origin (top-left)
RESOLUTION = 100.0
RASTER_SHAPE = (100, 100)  # 100×100 pixels = 10km × 10km


@pytest.fixture
def sample_pa_gdf():
    """
    Create sample protected areas GeoDataFrame in EPSG:3035.

    Creates 3 rectangular polygons covering parts of the 100×100 synthetic rasters:
    - PA1 (strict_core): 2km × 2km in lower-left quadrant
    - PA2 (regulatory): 3km × 2km in center
    - PA3 (contractual): 2km × 2km in upper-right quadrant

    Coordinates consistent with synthetic rasters:
    - Origin: (3000000, 2000000) — top-left corner
    - Raster extends 10km east and 10km south
    - Y-axis increases northward, so raster bottom = origin_y - 10000
    """
    # PA1: lower-left area (2km × 2km)
    # East: 1-3km from origin, South: 7-9km from origin
    pa1 = box(
        3000000 + 1000,          # xmin: 1km east
        2000000 - 9000,          # ymin: 9km south (lower edge)
        3000000 + 3000,          # xmax: 3km east
        2000000 - 7000           # ymax: 7km south
    )

    # PA2: center area (3km × 2km)
    # East: 3-6km from origin, South: 4-6km from origin
    pa2 = box(
        3000000 + 3000,
        2000000 - 6000,
        3000000 + 6000,
        2000000 - 4000
    )

    # PA3: upper-right area (2km × 2km)
    # East: 7-9km from origin, South: 1-3km from origin
    pa3 = box(
        3000000 + 7000,
        2000000 - 3000,
        3000000 + 9000,
        2000000 - 1000
    )

    gdf = gpd.GeoDataFrame({
        'WDPA_PID': [1, 2, 3],
        'NAME': ['PA1', 'PA2', 'PA3'],
        'IUCN_CAT': ['II', 'IV', 'V'],
        'protection_class': ['strict_core', 'regulatory', 'contractual'],
        'geometry': [pa1, pa2, pa3]
    }, crs='EPSG:3035')

    return gdf


@pytest.fixture
def sample_raster_paths():
    """
    Dictionary mapping criterion names to synthetic raster paths.

    Uses existing synthetic rasters from tests/synthetic_data/:
    - ecosystem_condition: [0-1] continuous
    - regulating_es: [0-1] continuous
    - cultural_es: [0-1] continuous
    - provisioning_es: [0-1] continuous
    """
    return {
        'ecosystem_condition': str(SYNTHETIC_DATA_DIR / 'ecosystem_condition.tif'),
        'regulating_es': str(SYNTHETIC_DATA_DIR / 'regulating_es.tif'),
        'cultural_es': str(SYNTHETIC_DATA_DIR / 'cultural_es.tif'),
        'provisioning_es': str(SYNTHETIC_DATA_DIR / 'provisioning_es.tif')
    }






def test_zonal_stats_returns_dataframe(sample_pa_gdf, sample_raster_paths):
    """Test that zonal_stats_by_pa_class returns DataFrame with correct columns."""
    result = zonal_stats_by_pa_class(sample_pa_gdf, sample_raster_paths)

    # Verify output is DataFrame
    assert isinstance(result, pd.DataFrame), "Output must be a pandas DataFrame"

    # Verify required columns
    required_cols = ['criterion', 'iucn_cat', 'mean', 'median', 'std', 'min', 'max', 'pixel_count']
    for col in required_cols:
        assert col in result.columns, f"Missing column: {col}"

    # Verify non-empty result
    assert len(result) > 0, "Result should contain at least one row"

    # Verify we have stats for multiple criteria and classes
    # 4 criteria × (3 IUCN categories + 1 outside) = 16 rows expected
    n_criteria = len(sample_raster_paths)
    expected_min_rows = n_criteria  # At least one row per criterion
    assert len(result) >= expected_min_rows, f"Expected at least {expected_min_rows} rows"


def test_zonal_stats_pixel_count_positive(sample_pa_gdf, sample_raster_paths):
    """Test that pixel_count > 0 for all rows."""
    result = zonal_stats_by_pa_class(sample_pa_gdf, sample_raster_paths)

    # All pixel counts should be positive
    assert (result['pixel_count'] > 0).all(), (
        "All rows should have positive pixel_count. "
        f"Found zero counts in:\n{result[result['pixel_count'] <= 0]}"
    )


def test_zonal_stats_mean_in_valid_range(sample_pa_gdf, sample_raster_paths):
    """Test that mean values are within the raster's min-max range."""
    result = zonal_stats_by_pa_class(sample_pa_gdf, sample_raster_paths)

    # For each criterion, verify mean is between global min and max
    for criterion, raster_path in sample_raster_paths.items():
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            nodata = src.nodata

            # Filter out nodata
            if nodata is not None:
                valid_data = raster_data[raster_data != nodata]
            else:
                valid_data = raster_data[~np.isnan(raster_data)]

            global_min = float(valid_data.min())
            global_max = float(valid_data.max())

        # Check all rows for this criterion
        criterion_rows = result[result['criterion'] == criterion]

        for _, row in criterion_rows.iterrows():
            mean_val = row['mean']
            assert global_min <= mean_val <= global_max, (
                f"Mean {mean_val} for {criterion}/{row['iucn_cat']} outside valid range "
                f"[{global_min}, {global_max}]"
            )

            # Also verify min ≤ mean ≤ max within each zone
            assert row['min'] <= mean_val <= row['max'], (
                f"Mean {mean_val} not between min {row['min']} and max {row['max']} "
                f"for {criterion}/{row['iucn_cat']}"
            )


def test_zonal_stats_outside_class_present(sample_pa_gdf, sample_raster_paths):
    """Test that 'outside' class is always present in output."""
    result = zonal_stats_by_pa_class(sample_pa_gdf, sample_raster_paths)

    # Verify 'outside' class exists for each criterion
    for criterion in sample_raster_paths.keys():
        outside_rows = result[(result['criterion'] == criterion) & (result['iucn_cat'] == 'outside')]
        assert len(outside_rows) > 0, f"'outside' class missing for criterion: {criterion}"

        # Verify 'outside' has positive pixel count
        assert outside_rows.iloc[0]['pixel_count'] > 0, (
            f"Outside class should have pixels for {criterion}"
        )


def test_zonal_stats_non_epsg3035_raises(sample_raster_paths):
    """Test that passing a GeoDataFrame not in EPSG:3035 raises ValueError."""
    # Create PA GeoDataFrame in WGS84 (EPSG:4326)
    pa_wgs84 = gpd.GeoDataFrame({
        'WDPA_PID': [1],
        'protection_class': ['strict_core'],
        'geometry': [box(2.0, 48.0, 2.1, 48.1)]  # Paris area in WGS84
    }, crs='EPSG:4326')

    # Should raise ValueError
    with pytest.raises(ValueError, match="EPSG:3035"):
        zonal_stats_by_pa_class(pa_wgs84, sample_raster_paths)


def test_criterion_coverage_summary_shape(sample_pa_gdf, sample_raster_paths):
    """Test that pivot table has correct shape (n_classes × n_criteria)."""
    # Compute zonal stats
    zonal_df = zonal_stats_by_pa_class(sample_pa_gdf, sample_raster_paths)

    # Create summary pivot
    summary = criterion_coverage_summary(zonal_df)

    # Verify output is DataFrame
    assert isinstance(summary, pd.DataFrame)

    # Expected dimensions
    unique_classes = zonal_df['iucn_cat'].unique()
    unique_criteria = zonal_df['criterion'].unique()

    expected_n_rows = len(unique_classes)
    expected_n_cols = len(unique_criteria)

    assert summary.shape[0] == expected_n_rows, (
        f"Pivot should have {expected_n_rows} rows (classes), got {summary.shape[0]}"
    )
    assert summary.shape[1] == expected_n_cols, (
        f"Pivot should have {expected_n_cols} columns (criteria), got {summary.shape[1]}"
    )


def test_criterion_coverage_summary_values_match_zonal_means(sample_pa_gdf, sample_raster_paths):
    """Test that pivot values match the mean column of the zonal DataFrame."""
    # Compute zonal stats
    zonal_df = zonal_stats_by_pa_class(sample_pa_gdf, sample_raster_paths)

    # Create summary pivot
    summary = criterion_coverage_summary(zonal_df)

    # Verify each cell matches the corresponding mean value
    for iucn_cat in summary.index:
        for criterion in summary.columns:
            pivot_value = summary.loc[iucn_cat, criterion]

            # Find corresponding row in zonal_df
            matching_row = zonal_df[
                (zonal_df['iucn_cat'] == iucn_cat) &
                (zonal_df['criterion'] == criterion)
            ]

            if len(matching_row) > 0:
                zonal_mean = matching_row.iloc[0]['mean']

                # Values should match (allowing for floating-point precision)
                assert abs(pivot_value - zonal_mean) < 1e-6, (
                    f"Pivot value {pivot_value} does not match zonal mean {zonal_mean} "
                    f"for {iucn_cat}/{criterion}"
                )




if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

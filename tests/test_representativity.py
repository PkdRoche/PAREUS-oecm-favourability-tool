"""Tests for representativity module and coverage statistics."""

import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'modules'))

from module1_protected_areas.representativity import (
    cross_with_ecosystem_types,
    representativity_index,
    propose_group_a_weights
)
from module1_protected_areas.coverage_stats import (
    compute_net_area,
    coverage_by_class
)


@pytest.fixture
def sample_territory():
    """Create a sample territory geometry (100 km x 100 km = 1,000,000 ha)."""
    # In EPSG:3035, create a 100km x 100km box
    # Reference point: approximately central Europe
    territory = box(4000000, 3000000, 4100000, 3100000)
    return territory


@pytest.fixture
def sample_pa_gdf():
    """Create sample protected areas GeoDataFrame."""
    # Two overlapping squares (each 10km x 10km = 10,000 ha)
    # Overlap area: 5km x 10km = 5,000 ha
    # Net area should be 15,000 ha (not 20,000 ha)
    pa1 = box(4010000, 3010000, 4020000, 3020000)
    pa2 = box(4015000, 3010000, 4025000, 3020000)

    gdf = gpd.GeoDataFrame({
        'WDPA_PID': [1, 2],
        'NAME': ['PA1', 'PA2'],
        'IUCN_CAT': ['II', 'IV'],
        'protection_class': ['strict_core', 'regulatory'],
        'geometry': [pa1, pa2]
    }, crs='EPSG:3035')

    return gdf


@pytest.fixture
def sample_ecosystem_gdf():
    """Create sample ecosystem types GeoDataFrame."""
    # Three ecosystem patches
    forest = box(4010000, 3010000, 4030000, 3030000)  # 20km x 20km = 40,000 ha
    wetland = box(4030000, 3010000, 4050000, 3030000)  # 20km x 20km = 40,000 ha
    grassland = box(4050000, 3010000, 4070000, 3030000)  # 20km x 20km = 40,000 ha

    gdf = gpd.GeoDataFrame({
        'ecosystem_type': ['forests', 'wetlands', 'grasslands'],
        'geometry': [forest, wetland, grassland]
    }, crs='EPSG:3035')

    return gdf


def test_ri_equals_one_when_all_types_at_threshold():
    """Test that RI = 1.0 when all ecosystem types reach target threshold."""
    # Coverage at exactly 30% for all types
    coverage_df = pd.DataFrame({
        'ecosystem_type': ['forests', 'wetlands', 'grasslands'],
        'pa_class': ['strict_core', 'strict_core', 'strict_core'],
        'area_ha': [3000.0, 3000.0, 3000.0]
    })

    territory_totals = {
        'forests': 10000.0,
        'wetlands': 10000.0,
        'grasslands': 10000.0
    }

    ri_df = representativity_index(coverage_df, territory_totals, target_threshold=0.30)

    # All RI values should be 1.0
    assert (ri_df['RI'] == 1.0).all(), "RI should be 1.0 when coverage equals threshold"

    # Synthetic RI should be 1.0
    synthetic_ri = ri_df['RI'].mean()
    assert synthetic_ri == 1.0, f"Synthetic RI should be 1.0, got {synthetic_ri}"


def test_ri_equals_zero_when_no_coverage():
    """Test that RI = 0.0 when no ecosystem type has any coverage."""
    # No coverage for any type
    coverage_df = pd.DataFrame({
        'ecosystem_type': [],
        'pa_class': [],
        'area_ha': []
    })

    territory_totals = {
        'forests': 10000.0,
        'wetlands': 10000.0,
        'grasslands': 10000.0
    }

    ri_df = representativity_index(coverage_df, territory_totals, target_threshold=0.30)

    # All RI values should be 0.0
    assert (ri_df['RI'] == 0.0).all(), "RI should be 0.0 when coverage is zero"

    # Synthetic RI should be 0.0
    synthetic_ri = ri_df['RI'].mean()
    assert synthetic_ri == 0.0, f"Synthetic RI should be 0.0, got {synthetic_ri}"


def test_ri_capped_at_one_when_overcovered():
    """Test that RI is capped at 1.0 even when coverage exceeds threshold."""
    # Coverage at 60% (double the 30% threshold)
    coverage_df = pd.DataFrame({
        'ecosystem_type': ['forests', 'wetlands'],
        'pa_class': ['strict_core', 'strict_core'],
        'area_ha': [6000.0, 6000.0]
    })

    territory_totals = {
        'forests': 10000.0,
        'wetlands': 10000.0
    }

    ri_df = representativity_index(coverage_df, territory_totals, target_threshold=0.30)

    # RI should be capped at 1.0
    assert (ri_df['RI'] == 1.0).all(), "RI should be capped at 1.0 even with overcoverage"

    # Coverage percentage should show the actual value
    assert (ri_df['coverage_pct'] == 60.0).all(), "Coverage percentage should show actual value"


def test_proposed_weights_sum_to_one():
    """Test that proposed Group A weights sum to exactly 1.0."""
    # Create representativity data with deficits
    ri_df = pd.DataFrame({
        'ecosystem_type': ['forests', 'wetlands', 'grasslands'],
        'total_ha': [10000.0, 10000.0, 10000.0],
        'protected_ha': [2000.0, 1000.0, 500.0],
        'coverage_pct': [20.0, 10.0, 5.0],
        'RI': [0.67, 0.33, 0.17],
        'gap_ha': [1000.0, 2000.0, 2500.0]
    })

    criterion_mapping = {
        'ecosystem_condition': 'all',
        'regulating_es': 'wetlands',
        'low_pressure': 'grasslands'
    }

    weights = propose_group_a_weights(ri_df, criterion_mapping)

    # Verify sum equals 1.0
    weight_sum = sum(weights.values())
    assert abs(weight_sum - 1.0) < 1e-6, f"Weights should sum to 1.0, got {weight_sum}"

    # Verify all weights are non-negative
    assert all(w >= 0 for w in weights.values()), "All weights should be non-negative"


def test_proposed_weights_zero_deficit_excluded():
    """Test weight proposal when all types are at target (zero deficit)."""
    # All types at target = no deficit
    ri_df = pd.DataFrame({
        'ecosystem_type': ['forests', 'wetlands'],
        'total_ha': [10000.0, 10000.0],
        'protected_ha': [3000.0, 3000.0],
        'coverage_pct': [30.0, 30.0],
        'RI': [1.0, 1.0],
        'gap_ha': [0.0, 0.0]
    })

    criterion_mapping = {
        'ecosystem_condition': 'all',
        'regulating_es': 'wetlands'
    }

    weights = propose_group_a_weights(ri_df, criterion_mapping)

    # Should return equal weights when no deficit
    weight_sum = sum(weights.values())
    assert abs(weight_sum - 1.0) < 1e-6, f"Weights should sum to 1.0, got {weight_sum}"

    # Equal weights expected (0.5 each)
    assert abs(weights['ecosystem_condition'] - 0.5) < 1e-6
    assert abs(weights['regulating_es'] - 0.5) < 1e-6


def test_net_area_less_than_or_equal_sum_of_areas(sample_pa_gdf, sample_territory):
    """Test that net deduplicated area is ≤ sum of individual areas."""
    # Compute net area via unary_union
    net_area = compute_net_area(sample_pa_gdf, sample_territory)

    # Compute sum of individual areas
    sum_areas = sample_pa_gdf.geometry.area.sum() / 10000.0  # Convert to hectares

    # Net area must be ≤ sum (due to deduplication)
    assert net_area <= sum_areas, (
        f"Net area ({net_area} ha) should be ≤ sum of individual areas ({sum_areas} ha)"
    )

    # For this specific test case: two 10km x 10km squares with 5km overlap
    # Expected: net = 15,000 ha, sum = 20,000 ha
    expected_net = 15000.0
    expected_sum = 20000.0

    assert abs(net_area - expected_net) < 100, (
        f"Expected net area ~{expected_net} ha, got {net_area} ha"
    )
    assert abs(sum_areas - expected_sum) < 100, (
        f"Expected sum area ~{expected_sum} ha, got {sum_areas} ha"
    )


def test_net_area_deduplication_verified(sample_territory):
    """Test net area deduplication with two overlapping polygons."""
    # Create two overlapping squares
    # Square 1: 10km x 10km = 10,000 ha
    # Square 2: 10km x 10km = 10,000 ha
    # Overlap: 5km x 10km = 5,000 ha
    # Net area: 15,000 ha

    pa1 = box(4010000, 3010000, 4020000, 3020000)
    pa2 = box(4015000, 3010000, 4025000, 3020000)

    gdf = gpd.GeoDataFrame({
        'WDPA_PID': [1, 2],
        'protection_class': ['strict_core', 'strict_core'],
        'geometry': [pa1, pa2]
    }, crs='EPSG:3035')

    # Compute net area
    net_area = compute_net_area(gdf, sample_territory)

    # Compute sum of individual areas
    sum_areas = gdf.geometry.area.sum() / 10000.0

    # Verify deduplication occurred
    assert net_area < sum_areas, (
        f"Net area ({net_area} ha) should be less than sum ({sum_areas} ha) "
        f"due to overlap"
    )

    # Verify expected values
    expected_net = 15000.0
    expected_sum = 20000.0

    assert abs(net_area - expected_net) < 100, (
        f"Expected net area {expected_net} ha, got {net_area} ha"
    )
    assert abs(sum_areas - expected_sum) < 100, (
        f"Expected sum area {expected_sum} ha, got {sum_areas} ha"
    )

    # Verify deduplication percentage
    dedup_pct = ((sum_areas - net_area) / sum_areas) * 100
    expected_dedup_pct = 25.0  # 5000/20000 = 25%

    assert abs(dedup_pct - expected_dedup_pct) < 1.0, (
        f"Expected {expected_dedup_pct}% deduplication, got {dedup_pct}%"
    )


def test_coverage_by_class_deduplication(sample_territory):
    """Test that coverage_by_class correctly deduplicates within and across classes."""
    # Create PAs with overlaps
    # Class 1: two overlapping squares (net 15,000 ha)
    pa1 = box(4010000, 3010000, 4020000, 3020000)  # 10,000 ha
    pa2 = box(4015000, 3010000, 4025000, 3020000)  # 10,000 ha

    # Class 2: one separate square (10,000 ha)
    pa3 = box(4030000, 3010000, 4040000, 3020000)  # 10,000 ha

    gdf = gpd.GeoDataFrame({
        'WDPA_PID': [1, 2, 3],
        'protection_class': ['strict_core', 'strict_core', 'regulatory'],
        'geometry': [pa1, pa2, pa3]
    }, crs='EPSG:3035')

    territory_area_ha = 1000000.0  # 1,000,000 ha

    coverage = coverage_by_class(gdf, territory_area_ha)

    # Verify strict_core deduplication
    strict_row = coverage[coverage['protection_class'] == 'strict_core']
    assert len(strict_row) == 1
    strict_area = strict_row.iloc[0]['area_ha']
    assert abs(strict_area - 15000.0) < 100, (
        f"Expected strict_core area 15,000 ha, got {strict_area} ha"
    )

    # Verify regulatory (no overlap)
    reg_row = coverage[coverage['protection_class'] == 'regulatory']
    assert len(reg_row) == 1
    reg_area = reg_row.iloc[0]['area_ha']
    assert abs(reg_area - 10000.0) < 100

    # Verify total (no cross-class overlap in this case)
    total_row = coverage[coverage['protection_class'] == 'TOTAL']
    assert len(total_row) == 1
    total_area = total_row.iloc[0]['area_ha']
    expected_total = 25000.0
    assert abs(total_area - expected_total) < 100, (
        f"Expected total area {expected_total} ha, got {total_area} ha"
    )


def test_cross_with_ecosystem_types(sample_pa_gdf, sample_ecosystem_gdf):
    """Test spatial cross-analysis between PAs and ecosystem types."""
    coverage_df = cross_with_ecosystem_types(sample_pa_gdf, sample_ecosystem_gdf)

    # Verify output structure
    assert 'ecosystem_type' in coverage_df.columns
    assert 'pa_class' in coverage_df.columns
    assert 'area_ha' in coverage_df.columns

    # Verify non-negative areas
    assert (coverage_df['area_ha'] >= 0).all()

    # Verify forests have coverage (PAs overlap with forests)
    forest_coverage = coverage_df[coverage_df['ecosystem_type'] == 'forests']
    assert len(forest_coverage) > 0, "Should find PA coverage in forests"


def test_ri_calculation_formula():
    """Test RI formula: min(coverage / threshold, 1.0)."""
    # Test various coverage levels
    test_cases = [
        {'coverage': 0.0, 'threshold': 0.30, 'expected_ri': 0.0},
        {'coverage': 0.15, 'threshold': 0.30, 'expected_ri': 0.5},
        {'coverage': 0.30, 'threshold': 0.30, 'expected_ri': 1.0},
        {'coverage': 0.60, 'threshold': 0.30, 'expected_ri': 1.0},  # Capped
    ]

    for case in test_cases:
        coverage_df = pd.DataFrame({
            'ecosystem_type': ['test'],
            'pa_class': ['strict_core'],
            'area_ha': [case['coverage'] * 10000.0]
        })

        territory_totals = {'test': 10000.0}

        ri_df = representativity_index(
            coverage_df,
            territory_totals,
            target_threshold=case['threshold']
        )

        actual_ri = ri_df.iloc[0]['RI']
        assert abs(actual_ri - case['expected_ri']) < 1e-6, (
            f"Coverage {case['coverage']}, threshold {case['threshold']}: "
            f"expected RI {case['expected_ri']}, got {actual_ri}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

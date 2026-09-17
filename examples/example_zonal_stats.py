"""
Example: Zonal Statistics for Protected Areas

This script demonstrates how to compute zonal statistics of MCE criterion rasters
within existing WDPA protected areas. This helps analyze whether high-value areas
are already protected and identifies potential gaps.

Requirements:
- WDPA shapefile in EPSG:3035
- MCE criterion rasters (output from Module 2 preprocessing)
- IUCN classification YAML config
"""

import sys
from pathlib import Path
import yaml

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'modules'))

from module1_protected_areas import (
    load_wdpa_local,
    classify_iucn,
    zonal_stats_by_pa_class,
    criterion_coverage_summary
)


def main():
    """Run zonal statistics example workflow."""

    # Configuration
    wdpa_path = "data/synthetic_wdpa.shp"
    raster_dir = Path("output/rasters")
    config_path = "config/iucn_classification.yaml"

    print("=" * 80)
    print("ZONAL STATISTICS: MCE Criterion Analysis within Protected Areas")
    print("=" * 80)

    # Step 1: Load and classify WDPA data
    print("\n[1/4] Loading WDPA data...")
    try:
        pa_gdf = load_wdpa_local(wdpa_path)
        print(f"  Loaded {len(pa_gdf)} protected areas")
    except FileNotFoundError:
        print(f"  ERROR: WDPA file not found at {wdpa_path}")
        print("  Please ensure WDPA data is available or update the path.")
        return

    # Reproject to EPSG:3035
    print("\n[2/4] Reprojecting to EPSG:3035...")
    if pa_gdf.crs != 'EPSG:3035':
        pa_gdf = pa_gdf.to_crs('EPSG:3035')
        print(f"  Reprojected from {pa_gdf.crs} to EPSG:3035")
    else:
        print("  Already in EPSG:3035")

    # Classify IUCN categories
    print("\n[3/4] Classifying IUCN categories...")
    try:
        with open(config_path) as f:
            classification = yaml.safe_load(f)
        pa_gdf = classify_iucn(pa_gdf, classification)

        # Show classification distribution
        print("  Protection class distribution:")
        for class_name, count in pa_gdf['protection_class'].value_counts().items():
            print(f"    {class_name}: {count}")

    except FileNotFoundError:
        print(f"  ERROR: Config file not found at {config_path}")
        return

    # Step 2: Define criterion rasters
    print("\n[4/4] Computing zonal statistics...")

    # Check which rasters are available
    available_rasters = {}
    criterion_files = {
        "ecosystem_condition": "ecosystem_condition.tif",
        "connectivity": "connectivity.tif",
        "species_richness": "species_richness.tif",
        "low_pressure": "low_pressure.tif"
    }

    for criterion, filename in criterion_files.items():
        raster_path = raster_dir / filename
        if raster_path.exists():
            available_rasters[criterion] = str(raster_path)
            print(f"  Found: {criterion}")
        else:
            print(f"  Missing: {criterion} (skipping)")

    if not available_rasters:
        print("\n  ERROR: No criterion rasters found in output/rasters/")
        print("  Please run Module 2 preprocessing first to generate criterion rasters.")
        return

    # Compute zonal statistics
    print(f"\n  Computing statistics for {len(available_rasters)} criteria...")
    zonal_stats_df = zonal_stats_by_pa_class(pa_gdf, available_rasters)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS: Detailed Zonal Statistics")
    print("=" * 80)
    print(zonal_stats_df.to_string(index=False))

    # Create summary pivot table
    print("\n" + "=" * 80)
    print("SUMMARY: Mean Criterion Values by PA Class")
    print("=" * 80)
    summary = criterion_coverage_summary(zonal_stats_df)
    print(summary.to_string())

    # Export results
    output_dir = Path("output/zonal_stats")
    output_dir.mkdir(parents=True, exist_ok=True)

    zonal_stats_path = output_dir / "zonal_statistics.csv"
    zonal_stats_df.to_csv(zonal_stats_path, index=False)
    print(f"\n  Detailed stats saved to: {zonal_stats_path}")

    summary_path = output_dir / "summary_pivot.csv"
    summary.to_csv(summary_path)
    print(f"  Summary table saved to: {summary_path}")

    # Analysis insights
    print("\n" + "=" * 80)
    print("INSIGHTS")
    print("=" * 80)

    for criterion in available_rasters.keys():
        criterion_data = zonal_stats_df[zonal_stats_df['criterion'] == criterion]

        # Compare PA vs outside
        pa_data = criterion_data[criterion_data['pa_class'] != 'outside']
        outside_data = criterion_data[criterion_data['pa_class'] == 'outside']

        if len(pa_data) > 0 and len(outside_data) > 0:
            pa_mean = pa_data['mean'].mean()
            outside_mean = outside_data['mean'].values[0]

            print(f"\n{criterion.upper()}:")
            print(f"  PA mean: {pa_mean:.3f}")
            print(f"  Outside mean: {outside_mean:.3f}")

            if pa_mean > outside_mean:
                diff_pct = ((pa_mean - outside_mean) / outside_mean) * 100
                print(f"  → PAs contain {diff_pct:.1f}% higher values (GOOD)")
            else:
                diff_pct = ((outside_mean - pa_mean) / outside_mean) * 100
                print(f"  → Outside areas have {diff_pct:.1f}% higher values (GAP)")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

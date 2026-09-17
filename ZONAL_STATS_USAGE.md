# Zonal Statistics Module Usage

## Overview

The `zonal_stats.py` module provides zonal statistics computation for MCE criterion rasters within existing WDPA protected areas. This allows you to analyze how different criterion values (e.g., ecosystem condition, connectivity, species richness) vary across different protection classes.

## Functions

### `zonal_stats_by_pa_class(pa_gdf, raster_paths, nodata=None)`

Computes summary statistics (mean, median, std, min, max, pixel_count) for each combination of criterion raster and PA protection class.

**Parameters:**
- `pa_gdf`: GeoDataFrame with 'protection_class' column (output of `classify_iucn()`)
  - MUST be in EPSG:3035
- `raster_paths`: dict mapping criterion name to GeoTIFF path
  - Example: `{"ecosystem_condition": "data/rasters/condition.tif"}`
- `nodata`: optional custom nodata value (otherwise uses raster metadata)

**Returns:**
- Tidy DataFrame with columns: `criterion`, `pa_class`, `mean`, `median`, `std`, `min`, `max`, `pixel_count`

**Key Features:**
- Automatically computes stats for areas **outside all PAs** (class = "outside")
- Handles CRS mismatch by reprojecting on-the-fly (with warning)
- Uses geometric union to avoid counting overlapping PA areas multiple times

### `criterion_coverage_summary(zonal_df)`

Converts the tidy zonal statistics DataFrame into a pivot table for easy comparison.

**Parameters:**
- `zonal_df`: output from `zonal_stats_by_pa_class()`

**Returns:**
- Pivot table with rows = PA classes, columns = criteria, values = mean

## Example Workflow

```python
from modules.module1_protected_areas import (
    load_wdpa_local,
    classify_iucn,
    zonal_stats_by_pa_class,
    criterion_coverage_summary
)
import yaml

# Step 1: Load and classify WDPA data
pa_gdf = load_wdpa_local("data/WDPA_France.shp")
pa_gdf = pa_gdf.to_crs('EPSG:3035')

with open('config/iucn_classification.yaml') as f:
    classification = yaml.safe_load(f)
pa_gdf = classify_iucn(pa_gdf, classification)

# Step 2: Define criterion rasters (from Module 2 preprocessing)
raster_paths = {
    "ecosystem_condition": "output/rasters/ecosystem_condition.tif",
    "connectivity": "output/rasters/connectivity.tif",
    "species_richness": "output/rasters/species_richness.tif",
    "low_pressure": "output/rasters/low_pressure.tif"
}

# Step 3: Compute zonal statistics
zonal_stats_df = zonal_stats_by_pa_class(pa_gdf, raster_paths)

# Step 4: Create summary pivot table
summary = criterion_coverage_summary(zonal_stats_df)

print(summary)
# Output:
#                    ecosystem_condition  connectivity  species_richness  low_pressure
# pa_class
# strict_core                    0.75          0.82              0.68          0.70
# regulatory                     0.65          0.74              0.55          0.62
# contractual                    0.58          0.68              0.50          0.55
# outside                        0.45          0.52              0.38          0.40
```

## Interpretation

The zonal statistics reveal:

1. **PA effectiveness**: Are higher-quality areas (higher criterion values) protected?
   - If `mean(PA classes) > mean(outside)`, PAs are effectively targeting high-value areas

2. **Class differentiation**: Do stricter protection classes contain higher-value areas?
   - Expected pattern: strict_core > regulatory > contractual > outside

3. **Gap identification**: Which criteria are underrepresented in PAs?
   - Low mean values across all PA classes suggest need for targeted expansion

## Integration with Module 2 (MCE)

The zonal statistics can inform:

- **Criterion weight adjustment**: Criteria poorly represented in existing PAs may warrant higher weights in the MCE
- **Threshold calibration**: Understanding current PA criterion values helps set realistic target thresholds
- **Validation**: Compare MCE output high-suitability areas with existing PA criterion profiles

## CRS Requirements

- Input `pa_gdf` MUST be in EPSG:3035 (raises ValueError otherwise)
- Raster CRS mismatch triggers a warning but is handled automatically via on-the-fly reprojection
- All area calculations use the raster's native grid (no resampling of raster data)

## Performance Notes

- For large rasters (>1GB), processing may take several minutes per criterion
- Memory usage scales with raster size (entire raster is loaded for "outside" stats computation)
- Consider spatial subsetting for preliminary analysis if performance is an issue

## Testing

Run the test suite:

```bash
cd tests
python -m pytest test_zonal_stats.py -v
```

Tests verify:
- CRS validation
- Output structure correctness
- Statistical validity (min ≤ mean ≤ max, etc.)
- "outside" class computation
- Pivot table generation

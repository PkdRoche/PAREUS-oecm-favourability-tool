# Corine Land Cover (CLC) Loader Module

## Overview

The `clc_loader.py` module provides functions for loading, clipping, resampling, and reclassifying Corine Land Cover (CLC) raster data for the OECM Favourability Tool.

## Module Location

**File**: `C:\Users\phroche\IONOS HiDrive Next\Mobilité_Travail\Claude_Code\PareusProg\oecm-favourability-tool\modules\utils\clc_loader.py`

## Functions

### 1. `load_clc(filepath, study_area_geom, target_resolution=100)`

Load a local CLC GeoTIFF, clip to study area, and optionally resample.

**Parameters**:
- `filepath` (str): Path to CLC GeoTIFF (must be EPSG:3035)
- `study_area_geom` (shapely.geometry): Study area geometry in EPSG:3035
- `target_resolution` (float): Target resolution in meters (default: 100)

**Returns**:
- `tuple[np.ndarray, dict]`: (array, profile)

**Raises**:
- `ValueError` if CRS is not EPSG:3035

### 2. `reclassify_clc(array, reclassification_table)`

Reclassify CLC integer codes to continuous scores in [0, 1].

**Parameters**:
- `array` (np.ndarray): Raw CLC int16 array (codes 111-523)
- `reclassification_table` (dict[int, float]): Mapping from CLC code to score

**Returns**:
- `np.ndarray`: Float32 array with scores in [0, 1]

**Notes**:
- NoData values (0, 128) are converted to np.nan
- Unmapped codes are set to np.nan

### 3. `load_and_reclassify_clc(filepath, study_area_geom, config_path, target_resolution=100)`

Convenience wrapper: load CLC and apply reclassification in one step.

**Parameters**:
- `filepath` (str): Path to CLC GeoTIFF
- `study_area_geom` (shapely.geometry): Study area geometry in EPSG:3035
- `config_path` (str): Path to YAML config with reclassification table
- `target_resolution` (float): Target resolution in meters (default: 100)

**Returns**:
- `tuple[np.ndarray, dict]`: (scored_array, profile)

**Config Format**:
Supports both simple and nested formats:

```yaml
# Simple format
reclassification:
  111: 0.0
  112: 0.1
  311: 0.9

# Nested format (with metadata)
reclassification:
  111:
    score: 0.0
    label: "Continuous urban fabric"
    notes: "Eliminatory"
  311:
    score: 0.9
    label: "Broad-leaved forest"
```

### 4. `get_clc_legend()`

Return the full CLC 2018 44-class legend.

**Returns**:
- `dict[int, dict[str, str]]`: Mapping of CLC code to metadata

**Example**:
```python
{
    111: {
        "label": "Continuous urban fabric",
        "level1": "Artificial surfaces",
        "level2": "Urban fabric",
        "level3": "Continuous urban fabric"
    },
    ...
}
```

## Configuration File

**Location**: `C:\Users\phroche\IONOS HiDrive Next\Mobilité_Travail\Claude_Code\PareusProg\oecm-favourability-tool\config\clc_reclassification.yaml`

The config file maps all 44 CLC 2018 classes to favourability scores [0.0, 1.0].

**Scoring rationale**:
- **0.90-1.00**: Natural/semi-natural habitats (forests, wetlands)
- **0.70-0.89**: Semi-natural areas (transitional woodland, natural grasslands)
- **0.30-0.60**: Extensive agriculture, heterogeneous areas
- **0.10-0.29**: Intensive agriculture
- **0.00-0.09**: Artificial surfaces (eliminatory)

## Usage Example

```python
from modules.utils.clc_loader import load_and_reclassify_clc
from shapely.geometry import box

# Define study area (EPSG:3035)
study_area = box(3000000, 2000000, 3100000, 2100000)

# Load and reclassify CLC
scored_array, profile = load_and_reclassify_clc(
    filepath='data/clc2018_epsg3035.tif',
    study_area_geom=study_area,
    config_path='config/clc_reclassification.yaml',
    target_resolution=100
)

# scored_array is now a float32 array with values in [0, 1]
print(f"Score range: {np.nanmin(scored_array):.2f} - {np.nanmax(scored_array):.2f}")
```

## Technical Details

- **CRS**: EPSG:3035 (LAEA Europe) - mandatory
- **Native Resolution**: 100m
- **Data Type**: int16 (input), float32 (output after reclassification)
- **NoData Values**: 0 and 128
- **Resampling Method**: Nearest-neighbour (CLC is categorical data)
- **CLC Codes**: 111-523 (44 classes)

## Testing

**Test file**: `tests/test_clc_loader.py`

Run tests:
```bash
cd tests
python -m pytest test_clc_loader.py -v
```

Test coverage:
- CRS validation (raises error if not EPSG:3035)
- Clipping to study area geometry
- Resampling to target resolution
- Reclassification to [0, 1] range
- NoData handling (0, 128 → np.nan)
- Unmapped codes → np.nan
- Simple and nested config format support
- Output dtype (float32)

## References

- **CLC 2018 Documentation**: https://land.copernicus.eu/user-corner/technical-library/corine-land-cover-nomenclature-guidelines/html/
- **Data Source**: Copernicus Land Monitoring Service
- **CRS**: EPSG:3035 (ETRS89-extended / LAEA Europe)

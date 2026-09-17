# NUTS2 Region Selector

## Overview

The NUTS2 loader module provides functionality to download and use NUTS (Nomenclature of Territorial Units for Statistics) Level 2 administrative boundaries from Eurostat for study area selection.

## Features

- **Automatic download**: Fetches NUTS2 GeoJSON directly from Eurostat GISCO services
- **Caching**: Uses Streamlit's `@st.cache_data` to avoid repeated downloads
- **Two-step selector**: Country → NUTS2 Region hierarchy
- **Fallback mode**: Manual bounding box input if download fails
- **EPSG:3035 native**: All geometries returned in LAEA Europe projection

## Data Source

URL pattern:
```
https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_{scale}_{year}_3035.geojson
```

Default parameters:
- Year: 2021
- Scale: 20M (1:20 million, ~500KB file)

Available scales:
- `03M`: 1:3 million (high detail, ~10MB)
- `10M`: 1:10 million (medium detail, ~2MB)
- `20M`: 1:20 million (low detail, ~500KB) — **default**
- `60M`: 1:60 million (very low detail, ~100KB)

## Functions

### `load_nuts2(year=2021, scale="20M")`
Downloads and returns NUTS2 boundaries as GeoDataFrame.

**Returns**: `gpd.GeoDataFrame` with columns:
- `NUTS_ID`: Region identifier (e.g., "FR10", "DE21")
- `CNTR_CODE`: 2-letter country code
- `NUTS_NAME`: Region name
- `LEVL_CODE`: NUTS level (always 2)
- `geometry`: Polygon/MultiPolygon in EPSG:3035

### `get_countries(nuts2_gdf)`
Returns sorted list of unique country codes.

### `get_nuts2_for_country(nuts2_gdf, country_code)`
Filters and returns regions for specified country, sorted by name.

### `get_nuts2_geometry(nuts2_gdf, nuts_id)`
Returns geometry for specific NUTS_ID.

## Usage in Sidebar

The sidebar now returns:
```python
{
    'study_area_nuts_id': str,       # e.g., "FR10" or "CUSTOM"
    'study_area_name': str,          # Region name
    'study_area_geometry': BaseGeometry,  # Shapely geometry in EPSG:3035
    ...  # other parameters
}
```

## Error Handling

If Eurostat download fails (network error, service unavailable):
- Warning displayed in sidebar
- Fallback to manual bounding box input (4 fields: xmin, ymin, xmax, ymax)
- `study_area_nuts_id` set to `"CUSTOM"`

## Dependencies

- `geopandas`
- `requests`
- `shapely`
- `streamlit` (for caching decorator)

## Testing

Run the test script to verify connectivity:
```bash
python test_nuts2_loader.py
```

Expected output:
- Number of NUTS2 regions loaded
- List of countries
- Sample regions for France
- Geometry details for first region

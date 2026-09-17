# NUTS2 Region Selector Implementation

## Summary

Replaced the simple ISO3 country code input with a two-step hierarchical selector (Country → NUTS2 Region) using official Eurostat boundaries.

## Files Created

### 1. `modules/utils/__init__.py`
Empty package initializer for utils module.

### 2. `modules/utils/nuts2_loader.py`
Core functionality for NUTS2 boundary loading:
- `load_nuts2(year, scale)` — Downloads NUTS2 GeoJSON from Eurostat with caching
- `get_countries(nuts2_gdf)` — Returns sorted list of country codes
- `get_nuts2_for_country(nuts2_gdf, country_code)` — Filters regions by country
- `get_nuts2_geometry(nuts2_gdf, nuts_id)` — Retrieves geometry for specific region

**Data source**:
```
https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_20M_2021_3035.geojson
```

### 3. `test_nuts2_loader.py`
Standalone test script to verify NUTS2 loader functionality (not integrated with Streamlit).

### 4. `modules/utils/README_NUTS2.md`
Documentation for NUTS2 module including usage, data sources, and API reference.

## Files Modified

### `ui/sidebar.py`

**Imports added:**
```python
from modules.utils.nuts2_loader import (
    load_nuts2,
    get_countries,
    get_nuts2_for_country,
    get_nuts2_geometry
)
```

**Section 1 (Study Area) completely rewritten:**

**Before:**
- Single text input: ISO3 country code (e.g., "FRA")
- Returned: `'iso3': str`

**After:**
- Two-step selectbox hierarchy:
  1. Country (2-letter code from NUTS2 data)
  2. NUTS2 Region (filtered by selected country)
- Display: Region name with NUTS_ID and area in km²
- Fallback: Manual bounding box input if Eurostat download fails
- Returns:
  ```python
  'study_area_nuts_id': str        # e.g., "FR10" or "CUSTOM"
  'study_area_name': str           # Region display name
  'study_area_geometry': BaseGeometry  # Shapely polygon in EPSG:3035
  ```

**Docstring updated:**
Return parameter documentation now reflects new study area fields.

### `ui/tab_module1.py`

**Line 59 updated:**
- Old: "Set ISO3 country code in sidebar (e.g., 'FRA' for France)"
- New: "Select study area (Country → NUTS2 Region) in sidebar"

## User Experience

### Normal Operation
1. User opens sidebar
2. Eurostat NUTS2 data loads automatically (cached after first load)
3. User selects country from dropdown (sorted alphabetically)
4. User selects NUTS2 region from dropdown (sorted by region name)
5. Display shows: `FR10 — 12,012 km²` (example)
6. Geometry ready for use in modules

### Fallback Mode (network failure)
1. Warning displayed: "Could not load NUTS2 boundaries from Eurostat"
2. Manual input fields appear: X min/max, Y min/max (EPSG:3035)
3. Bounding box geometry created from coordinates
4. Display shows: `CUSTOM — 1,000,000 km²` (example)

## Technical Details

### Caching Strategy
- `@st.cache_data` on `load_nuts2()` function
- GeoJSON (~500KB) downloaded once per session
- No disk I/O; pure in-memory caching

### Coordinate Reference System
- All geometries in **EPSG:3035** (LAEA Europe)
- Area calculations in square meters
- Displayed in km² (divided by 1,000,000)

### Data Characteristics
- NUTS 2021 edition
- Scale: 1:20M (simplified boundaries)
- ~280 NUTS2 regions across Europe
- Covers EU27 + UK, Norway, Switzerland, Iceland, etc.

## Breaking Changes

### For downstream code:
- `params['iso3']` **removed**
- **New fields** available:
  - `params['study_area_nuts_id']`
  - `params['study_area_name']`
  - `params['study_area_geometry']`

### Migration guide:
If any code previously used `params['iso3']`:
- Replace with `params['study_area_nuts_id']` (region code) or
- Use `params['study_area_name']` (human-readable name)
- Use `params['study_area_geometry']` for spatial operations

## Dependencies

No new dependencies added (all already in project):
- `geopandas` (existing)
- `requests` (existing)
- `shapely` (existing)
- `streamlit` (existing)

## Testing

To test NUTS2 loader independently:
```bash
python test_nuts2_loader.py
```

Expected output:
```
Testing NUTS2 loader...
------------------------------------------------------------

1. Loading NUTS2 boundaries from Eurostat...
   SUCCESS: Loaded 242 NUTS2 regions
   CRS: EPSG:3035
   Columns: ['NUTS_ID', 'CNTR_CODE', 'NUTS_NAME', 'LEVL_CODE', 'geometry']

2. Getting list of countries...
   Found 31 countries
   First 10: ['AT', 'BE', 'BG', 'CH', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL']

3. Getting NUTS2 regions for France (FR)...
   Found 27 French NUTS2 regions

   First 5 regions:
      - Alsace (FRF1)
      - Aquitaine (FRI1)
      - Auvergne (FRK1)
      - Basse-Normandie (FRD1)
      - Bourgogne (FRC1)

4. Getting geometry for a specific region...
   Region: FRF1
   Geometry type: MultiPolygon
   Area: 8,280.45 km²
   Bounds: (3924571.23, 2958634.56, 4156782.91, 3134521.77)

============================================================
All tests completed successfully!
```

## Future Enhancements

Potential improvements (not implemented):
1. NUTS3 level selector (even finer resolution)
2. Multi-region selection (combine multiple NUTS2 areas)
3. Custom polygon upload alternative to bounding box
4. Display region on mini-map in sidebar
5. Filter by country group (e.g., "EU27 only")

## Eurostat Data License

NUTS boundaries sourced from Eurostat GISCO:
- License: CC BY 4.0
- Attribution: "© EuroGeographics for the administrative boundaries"
- Free to use for any purpose with attribution

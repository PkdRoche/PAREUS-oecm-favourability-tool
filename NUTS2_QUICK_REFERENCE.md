# NUTS2 Region Selector — Quick Reference

## For End Users

### Selecting a Study Area

1. Open sidebar (should be visible by default)
2. Look for **"1. Study Area"** section
3. Select country from dropdown (e.g., "FR" for France)
4. Select NUTS2 region from second dropdown (e.g., "Île de France (FR10)")
5. See region ID and area displayed below: `FR10 — 12,012 km²`

### If Network Error Occurs

If you see a warning about Eurostat connection:
1. Enter bounding box coordinates manually
2. Use EPSG:3035 coordinates (meters)
3. Example for France region:
   - X min: 2500000
   - Y min: 1500000
   - X max: 3500000
   - Y max: 2500000

## For Developers

### Accessing Study Area in Code

```python
# In any function that receives params dict from sidebar:
params = render_sidebar()

# Access study area
nuts_id = params['study_area_nuts_id']      # e.g., "FR10" or "CUSTOM"
name = params['study_area_name']            # e.g., "Île de France"
geometry = params['study_area_geometry']    # Shapely polygon in EPSG:3035

# Use geometry for spatial operations
area_km2 = geometry.area / 1_000_000
bounds = geometry.bounds  # (minx, miny, maxx, maxy)
```

### Migration from old `iso3` field

**Old code:**
```python
country_code = params['iso3']  # "FRA"
```

**New code:**
```python
# Option 1: Use NUTS region ID
region_id = params['study_area_nuts_id']  # "FR10"

# Option 2: Extract country from NUTS ID
country_code = params['study_area_nuts_id'][:2]  # "FR"

# Option 3: Use geometry directly
geometry = params['study_area_geometry']
```

### Loading NUTS2 Data Programmatically

```python
from modules.utils.nuts2_loader import (
    load_nuts2,
    get_countries,
    get_nuts2_for_country,
    get_nuts2_geometry
)

# Load all NUTS2 regions (cached)
nuts2_gdf = load_nuts2()

# Get list of countries
countries = get_countries(nuts2_gdf)  # ['AT', 'BE', 'BG', ...]

# Get regions for specific country
french_regions = get_nuts2_for_country(nuts2_gdf, 'FR')
# Returns GeoDataFrame sorted by region name

# Get geometry for specific region
geometry = get_nuts2_geometry(nuts2_gdf, 'FR10')
# Returns Shapely geometry in EPSG:3035
```

### Common Use Cases

#### 1. Clip raster to study area
```python
import rasterio
from rasterio.mask import mask

# Get study area geometry from params
geom = params['study_area_geometry']

# Clip raster
with rasterio.open('input.tif') as src:
    clipped, transform = mask(src, [geom], crop=True)
```

#### 2. Filter vector data to study area
```python
import geopandas as gpd

# Load vector data
data_gdf = gpd.read_file('protected_areas.shp')

# Ensure same CRS
if data_gdf.crs != 'EPSG:3035':
    data_gdf = data_gdf.to_crs('EPSG:3035')

# Filter to study area
geom = params['study_area_geometry']
filtered = data_gdf[data_gdf.intersects(geom)]
```

#### 3. Add study area to metadata
```python
metadata = {
    'study_area_id': params['study_area_nuts_id'],
    'study_area_name': params['study_area_name'],
    'study_area_epsg': 3035,
    'study_area_bounds': params['study_area_geometry'].bounds,
    'analysis_date': datetime.now().isoformat()
}
```

## NUTS2 Data Specifications

### Geometry Properties
- **CRS**: EPSG:3035 (LAEA Europe)
- **Unit**: Meters
- **Type**: Polygon or MultiPolygon
- **Simplification**: 1:20M scale (medium detail)

### NUTS ID Format
- **Pattern**: 2-letter country + 1-2 digits
- **Examples**:
  - `FR10` — Île de France (France)
  - `DE21` — Oberbayern (Germany)
  - `ES30` — Comunidad de Madrid (Spain)
  - `CUSTOM` — Manual bounding box (fallback)

### Coverage
- All EU27 member states
- EFTA countries (Norway, Switzerland, Iceland, Liechtenstein)
- UK (post-Brexit data still available)
- Candidate countries (Turkey, Serbia, etc.)

## Troubleshooting

### Problem: "Could not load NUTS2 boundaries"
**Cause**: Network error or Eurostat server unavailable
**Solution**: Use manual bounding box fallback (appears automatically)

### Problem: Selected region not appearing
**Cause**: Region name filtering issue
**Solution**: Try different country or check browser console for errors

### Problem: Area calculation seems wrong
**Cause**: CRS unit confusion (meters vs degrees)
**Solution**: All EPSG:3035 areas are in square meters. Divide by 1,000,000 for km²

### Problem: Geometry not compatible with other data
**Cause**: CRS mismatch
**Solution**: Reproject other data to EPSG:3035 before spatial operations

## API Reference

### `load_nuts2(year=2021, scale="20M")`
Downloads NUTS2 GeoJSON from Eurostat.

**Parameters:**
- `year` (int): NUTS edition year (default: 2021)
- `scale` (str): Map scale ("03M", "10M", "20M", "60M")

**Returns:** `gpd.GeoDataFrame` with NUTS2 regions

**Raises:** `requests.HTTPError` if download fails

---

### `get_countries(nuts2_gdf)`
Extract unique country codes from NUTS2 data.

**Parameters:**
- `nuts2_gdf` (gpd.GeoDataFrame): NUTS2 regions

**Returns:** `list[str]` of 2-letter country codes (sorted)

---

### `get_nuts2_for_country(nuts2_gdf, country_code)`
Filter NUTS2 regions for specific country.

**Parameters:**
- `nuts2_gdf` (gpd.GeoDataFrame): NUTS2 regions
- `country_code` (str): 2-letter country code (e.g., "FR")

**Returns:** `gpd.GeoDataFrame` with filtered regions (sorted by name)

---

### `get_nuts2_geometry(nuts2_gdf, nuts_id)`
Retrieve geometry for specific NUTS ID.

**Parameters:**
- `nuts2_gdf` (gpd.GeoDataFrame): NUTS2 regions
- `nuts_id` (str): NUTS identifier (e.g., "FR10")

**Returns:** `shapely.geometry.base.BaseGeometry` or `None`

## Performance Notes

- **First load**: 2-5 seconds (downloads ~500KB GeoJSON)
- **Cached loads**: <100ms (in-memory cache)
- **Region switching**: Instant (no re-download)
- **Memory usage**: ~10MB for full NUTS2 dataset

## Data Sources

- **Provider**: Eurostat GISCO
- **License**: CC BY 4.0
- **Attribution**: © EuroGeographics
- **URL**: https://gisco-services.ec.europa.eu/
- **Update frequency**: Annual (NUTS revisions)

## Support

For issues or questions:
1. Check `modules/utils/README_NUTS2.md` for detailed docs
2. Run `python test_nuts2_loader.py` to verify setup
3. Check Streamlit logs for error details
4. Review `NUTS2_IMPLEMENTATION.md` for technical details

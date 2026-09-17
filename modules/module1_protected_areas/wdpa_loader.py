"""WDPA data acquisition and loading functionality."""

import logging
from typing import Optional
import requests
import geopandas as gpd
import shapely.geometry.base
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch_wdpa_api(
    iso3: str,
    token: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Fetch WDPA data for a country via Protected Planet API.

    Falls back to local shapefile if API unavailable. Attempts to download
    WDPA data from the Protected Planet API. If the API is unreachable or
    returns an error, attempts to load from a local fallback file specified
    in config/settings.yaml.

    Parameters
    ----------
    iso3 : str
        ISO3 country code (e.g., 'FRA', 'DEU', 'ESP').
    token : str or None, optional
        Protected Planet API authentication token. If None, attempts to
        fetch without authentication (may be rate-limited).

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with standardised column names including:
        - WDPA_PID : unique protected area identifier
        - NAME : protected area name
        - IUCN_CAT : IUCN category code
        - DESIG : designation type
        - DESIG_TYPE : national/international designation
        - STATUS : establishment status
        - GIS_AREA : area in hectares
        - geometry : polygon geometry

    Raises
    ------
    RuntimeError
        If both API fetch and local fallback fail.

    Notes
    -----
    API endpoint: https://api.protectedplanet.net/v3/protected_areas
    Response is GeoJSON format, converted to GeoDataFrame.

    Examples
    --------
    >>> gdf = fetch_wdpa_api('FRA', token='my_api_token')
    >>> gdf.head()
    """
    # Attempt API fetch
    try:
        from config import settings
        base_url = settings.get('wdpa_api_base', 'https://api.protectedplanet.net/v3')
        endpoint = f"{base_url}/protected_areas"

        params = {'country': iso3}
        headers = {}
        if token:
            headers['Authorization'] = f'Token {token}'

        logger.info(f"Fetching WDPA data for {iso3} from API...")
        response = requests.get(endpoint, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Convert GeoJSON response to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(data['protected_areas'])

        # Standardise column names
        gdf = _standardise_columns(gdf)

        logger.info(f"Successfully fetched {len(gdf)} protected areas from API")
        return gdf

    except Exception as e:
        logger.warning(f"API fetch failed: {e}. Attempting local fallback...")

        # Attempt local fallback
        try:
            from config import settings
            local_path = settings.get('wdpa_local_path')

            if local_path is None:
                raise RuntimeError(
                    "API fetch failed and no local WDPA path configured in settings.yaml"
                )

            logger.warning(f"Loading WDPA from local file: {local_path}")
            return load_wdpa_local(local_path)

        except Exception as local_error:
            raise RuntimeError(
                f"Both API fetch and local fallback failed. API error: {e}. "
                f"Local error: {local_error}"
            )


def load_wdpa_local(path: str) -> gpd.GeoDataFrame:
    """
    Load WDPA from local shapefile or GDB.

    Supports multiple formats: shapefile (.shp), file geodatabase (.gdb),
    GeoPackage (.gpkg), and GeoJSON (.geojson).

    Parameters
    ----------
    path : str
        Path to local WDPA file. Can be:
        - Shapefile: '/path/to/WDPA_polygons.shp'
        - GDB: '/path/to/WDPA.gdb' (specify layer if needed)
        - GeoPackage: '/path/to/WDPA.gpkg'
        - GeoJSON: '/path/to/WDPA.geojson'

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with standardised WDPA columns (same as fetch_wdpa_api).

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the file format is not supported or cannot be read.

    Examples
    --------
    >>> gdf = load_wdpa_local('data/WDPA_FRA_polygons.shp')
    >>> gdf.crs
    <Geographic 2D CRS: EPSG:4326>
    """
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"WDPA file not found: {path}")

    logger.info(f"Loading WDPA from local file: {path}")

    # Allow GDAL to reconstruct a missing .shx for shapefiles uploaded via browser
    import os
    os.environ.setdefault('SHAPE_RESTORE_SHX', 'YES')

    suffix = path_obj.suffix.lower()

    # ZIP: extract to a temp dir and recurse on the main vector file
    if suffix == '.zip':
        import zipfile, tempfile, os
        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(path, 'r') as zf:
            zf.extractall(tmp_dir)
        for ext in ['.gpkg', '.shp', '.geojson', '.gdb']:
            matches = [
                os.path.join(tmp_dir, fn)
                for fn in os.listdir(tmp_dir)
                if fn.lower().endswith(ext)
            ]
            if matches:
                logger.info(f"Extracted from ZIP: {matches[0]}")
                return load_wdpa_local(matches[0])
        raise ValueError(f"No recognised vector file found inside ZIP: {path}")

    # For multi-layer formats (GDB, GPKG) pick the polygon layer explicitly
    if suffix in ('.gdb', '.gpkg'):
        try:
            import pyogrio
            layers = pyogrio.list_layers(path)
            # layers is an ndarray of (name, geometry_type) pairs
            layer_names = [row[0] for row in layers]
        except Exception:
            try:
                import fiona
                layer_names = fiona.listlayers(path)
            except Exception:
                layer_names = []

        if not layer_names:
            raise ValueError(f"No layers found in {suffix.upper()} file: {path}")

        # Prefer any layer whose name contains 'poly'
        polygon_layer = next(
            (l for l in layer_names if 'poly' in l.lower()),
            layer_names[0]
        )
        if polygon_layer != layer_names[0]:
            logger.info(f"Multi-layer file — using polygon layer: {polygon_layer}")
        gdf = gpd.read_file(path, layer=polygon_layer)
    else:
        # Standard single-layer formats (.shp, .geojson, …)
        gdf = gpd.read_file(path)

    # Standardise column names
    gdf = _standardise_columns(gdf)

    logger.info(f"Loaded {len(gdf)} protected areas from local file")
    return gdf


def filter_to_extent(
    gdf: gpd.GeoDataFrame,
    extent_geom: shapely.geometry.base.BaseGeometry
) -> gpd.GeoDataFrame:
    """
    Clip protected areas to study area extent.

    Performs spatial clipping and reprojection if CRS differs between
    the protected areas and the extent geometry. Logs a warning if
    reprojection is required.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Protected areas GeoDataFrame to clip.
    extent_geom : shapely.geometry.base.BaseGeometry
        Study area boundary geometry. Must be a valid Polygon or MultiPolygon.

    Returns
    -------
    gpd.GeoDataFrame
        Clipped GeoDataFrame in the same CRS as extent_geom.

    Notes
    -----
    - If gdf and extent_geom have different CRS, gdf is reprojected to match extent_geom.
    - Clipping uses spatial intersection.
    - Empty geometries after clipping are removed.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> extent = box(2.0, 48.0, 3.0, 49.0)  # Bounding box for study area
    >>> extent_gdf = gpd.GeoDataFrame([1], geometry=[extent], crs='EPSG:4326')
    >>> clipped = filter_to_extent(wdpa_gdf, extent_gdf.geometry[0])
    """
    # Create GeoDataFrame from extent geometry if needed
    if hasattr(extent_geom, 'crs'):
        extent_crs = extent_geom.crs
        extent_geom_single = extent_geom.iloc[0] if hasattr(extent_geom, 'iloc') else extent_geom
    else:
        # Default to EPSG:3035 (tool working CRS — NUTS2 geometries are always in 3035)
        extent_crs = 'EPSG:3035'
        extent_geom_single = extent_geom

    # Check CRS match
    if gdf.crs != extent_crs:
        logger.warning(
            f"CRS mismatch detected. Reprojecting PA data from {gdf.crs} to {extent_crs}"
        )
        gdf = gdf.to_crs(extent_crs)

    # Spatial clip
    logger.info("Clipping protected areas to study extent...")
    clipped = gdf[gdf.intersects(extent_geom_single)].copy()

    # Clip geometries to exact extent
    clipped['geometry'] = clipped.geometry.intersection(extent_geom_single)

    # Remove empty geometries
    clipped = clipped[~clipped.geometry.is_empty].copy()

    logger.info(f"Retained {len(clipped)} protected areas within extent")
    return clipped


def classify_iucn(
    gdf: gpd.GeoDataFrame,
    classification_table: dict
) -> gpd.GeoDataFrame:
    """
    Recode IUCN_MAX and DESIG_TYPE to 5 display classes.

    Classification table loaded from config/iucn_classification.yaml.
    Adds a 'protection_class' column with standardised values based on
    IUCN categories and designation keywords.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Protected areas GeoDataFrame with IUCN_MAX and DESIG columns.
    classification_table : dict
        Classification rules from iucn_classification.yaml. Expected structure:
        {
            'classes': {
                'strict_core': {'iucn_cats': [...], 'desig_keywords': [...]},
                'regulatory': {'iucn_cats': [...], 'desig_keywords': [...]},
                ...
            }
        }

    Returns
    -------
    gpd.GeoDataFrame
        Input GeoDataFrame with added 'protection_class' column containing:
        - 'strict_core' : IUCN Ia, Ib, II
        - 'regulatory' : IUCN III, IV
        - 'contractual' : IUCN V, VI, Natura 2000
        - 'unassigned' : Not Reported / Not Assigned
        - 'oecm' : WD-OECM source

    Notes
    -----
    Classification logic:
    1. Check IUCN_MAX first
    2. If not matched, check DESIG field for keywords
    3. Default to 'unassigned' if no match

    Examples
    --------
    >>> import yaml
    >>> with open('config/iucn_classification.yaml') as f:
    ...     classification = yaml.safe_load(f)
    >>> classified = classify_iucn(wdpa_gdf, classification)
    >>> classified['protection_class'].value_counts()
    """
    gdf = gdf.copy()

    # Ensure required columns exist
    if 'IUCN_MAX' not in gdf.columns:
        if 'IUCN_CAT' in gdf.columns:
            logger.info("IUCN_MAX column not found — falling back to IUCN_CAT")
            gdf['IUCN_MAX'] = gdf['IUCN_CAT']
        else:
            logger.warning("IUCN_MAX column not found, creating empty column")
            gdf['IUCN_MAX'] = ''

    # Accept DESIG or DESIG_ENG (WDPA uses DESIG_ENG in most exports)
    if 'DESIG' not in gdf.columns and 'DESIG_ENG' in gdf.columns:
        gdf['DESIG'] = gdf['DESIG_ENG']
        logger.info("Using DESIG_ENG as DESIG column")
    elif 'DESIG' not in gdf.columns:
        logger.warning("Neither DESIG nor DESIG_ENG found, creating empty column")
        gdf['DESIG'] = ''

    # Extract classes from configuration
    classes = classification_table.get('classes', {})

    # Pre-build flat IUCN-cat → class_name lookup for O(1) matching
    iucn_lookup: dict = {}
    for class_name, class_config in classes.items():
        for cat in class_config.get('iucn_cats', []):
            iucn_lookup.setdefault(cat, class_name)   # first match wins

    # Pre-lowercase designation keywords to avoid repeated .lower() calls
    desig_rules = [
        (class_name, [kw.lower() for kw in class_config.get('desig_keywords', [])])
        for class_name, class_config in classes.items()
    ]

    def _classify(iucn_cat: str, desig: str) -> str:
        """Classify a single PA row. Called via list-comprehension (no iterrows)."""
        # 1. Fast IUCN category lookup
        result = iucn_lookup.get(iucn_cat)
        if result:
            return result
        # 2. Designation keyword scan
        desig_lower = desig.lower()
        for class_name, keywords in desig_rules:
            for kw in keywords:
                if kw in desig_lower:
                    return class_name
        return 'unassigned'

    # Vectorised classification — list comprehension avoids per-row Series overhead
    iucn_cats = gdf['IUCN_MAX'].astype(str).str.strip().tolist()
    desigs    = gdf['DESIG'].astype(str).str.strip().tolist()
    gdf['protection_class'] = [_classify(ic, d) for ic, d in zip(iucn_cats, desigs)]

    logger.info(
        f"Classification complete. Distribution:\n"
        f"{gdf['protection_class'].value_counts().to_dict()}"
    )

    return gdf


def _standardise_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardise WDPA column names.

    Ensures consistent column naming across different WDPA sources.
    Handles variations in column names from different API versions or
    local file formats.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Raw WDPA GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with standardised column names.
    """
    # Column name mapping (API name → standard name)
    column_mapping = {
        'wdpa_pid': 'WDPA_PID',
        'wdpaid': 'WDPA_PID',
        'name': 'NAME',
        'iucn_cat': 'IUCN_CAT',
        'iucn_category': 'IUCN_CAT',
        'iucn_max': 'IUCN_MAX',
        'desig': 'DESIG',
        'designation': 'DESIG',
        'desig_type': 'DESIG_TYPE',
        'status': 'STATUS',
        'gis_area': 'GIS_AREA',
        'rep_area': 'GIS_AREA',
        'parent_iso': 'PARENT_ISO',
        'iso3': 'PARENT_ISO',
    }

    # Apply mapping (case-insensitive)
    cols_lower = {col.lower(): col for col in gdf.columns}
    rename_dict = {}

    for old_name, new_name in column_mapping.items():
        if old_name in cols_lower:
            rename_dict[cols_lower[old_name]] = new_name

    if rename_dict:
        gdf = gdf.rename(columns=rename_dict)

    return gdf

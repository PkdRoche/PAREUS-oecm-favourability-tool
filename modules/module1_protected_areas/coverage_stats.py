"""Protected area coverage statistics computation."""

import logging
from typing import Dict
import geopandas as gpd
import pandas as pd
import shapely.geometry.base
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def compute_net_area(
    gdf: gpd.GeoDataFrame,
    territory_geom: shapely.geometry.base.BaseGeometry
) -> float:
    """
    Compute net deduplicated protected area in hectares.

    Uses geometric union to avoid overestimation due to overlapping polygons.
    NEVER sums individual GIS_AREA fields directly.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Protected areas GeoDataFrame. Must be in a projected CRS (EPSG:3035).
    territory_geom : shapely.geometry.base.BaseGeometry
        Territory boundary geometry (used to clip union if needed).

    Returns
    -------
    float
        Net deduplicated area in hectares.

    Raises
    ------
    ValueError
        If GeoDataFrame is not in EPSG:3035.

    Notes
    -----
    - All geometries must be in EPSG:3035 before calling this function.
    - Area is computed via unary_union, then measured in square meters and
      converted to hectares (÷ 10,000).
    - This is the ONLY valid method for computing net protected area.

    Examples
    --------
    >>> gdf_proj = gdf.to_crs('EPSG:3035')
    >>> net_area = compute_net_area(gdf_proj, territory_geom)
    >>> print(f"Net protected area: {net_area:.2f} ha")
    """
    # Verify CRS
    if gdf.crs != 'EPSG:3035':
        raise ValueError(
            f"GeoDataFrame must be in EPSG:3035 before area calculation. "
            f"Current CRS: {gdf.crs}"
        )

    if len(gdf) == 0:
        return 0.0

    # Compute union
    logger.debug("Computing geometric union for net area calculation...")
    union_geom = unary_union(gdf.geometry)

    # Measure area in square meters, convert to hectares
    area_m2 = union_geom.area
    area_ha = area_m2 / 10000.0

    logger.debug(f"Net area computed: {area_ha:.2f} ha")
    return area_ha


def coverage_by_class(
    gdf: gpd.GeoDataFrame,
    territory_area_ha: float
) -> pd.DataFrame:
    """
    Compute surface area and percentage coverage per protection class.

    Includes net deduplicated total across all classes. Each class is
    deduplicated independently, then a total net area is computed across
    all classes.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Protected areas GeoDataFrame with 'protection_class' column.
        Must be in EPSG:3035.
    territory_area_ha : float
        Total territory area in hectares (for percentage calculation).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - protection_class : class name
        - area_ha : deduplicated area in hectares
        - pct_territory : percentage of territory
        - n_sites : number of sites/polygons

        Includes a final row with protection_class='TOTAL' for net area
        across all classes.

    Raises
    ------
    ValueError
        If GeoDataFrame is not in EPSG:3035 or missing protection_class column.

    Examples
    --------
    >>> coverage = coverage_by_class(gdf_proj, territory_area_ha=50000)
    >>> print(coverage)
      protection_class  area_ha  pct_territory  n_sites
    0      strict_core  12500.0           25.0      150
    1       regulatory   8000.0           16.0      200
    2      contractual   5000.0           10.0      300
    3       unassigned   2000.0            4.0       50
    4            TOTAL  15000.0           30.0      700
    """
    # Verify CRS
    if gdf.crs != 'EPSG:3035':
        raise ValueError(
            f"GeoDataFrame must be in EPSG:3035. Current CRS: {gdf.crs}"
        )

    # Verify protection_class column
    if 'protection_class' not in gdf.columns:
        raise ValueError("GeoDataFrame must have 'protection_class' column")

    if len(gdf) == 0:
        return pd.DataFrame(columns=['protection_class', 'area_ha', 'pct_territory', 'n_sites'])

    results = []

    # Compute per-class statistics
    for class_name in gdf['protection_class'].unique():
        class_gdf = gdf[gdf['protection_class'] == class_name]

        # Deduplicated area for this class
        class_union = unary_union(class_gdf.geometry)
        area_ha = class_union.area / 10000.0
        pct = (area_ha / territory_area_ha * 100.0) if territory_area_ha > 0 else 0.0
        n_sites = len(class_gdf)

        results.append({
            'protection_class': class_name,
            'area_ha': area_ha,
            'pct_territory': pct,
            'n_sites': n_sites
        })

    # Compute total net area (across all classes)
    total_union = unary_union(gdf.geometry)
    total_area_ha = total_union.area / 10000.0
    total_pct = (total_area_ha / territory_area_ha * 100.0) if territory_area_ha > 0 else 0.0

    results.append({
        'protection_class': 'TOTAL',
        'area_ha': total_area_ha,
        'pct_territory': total_pct,
        'n_sites': len(gdf)
    })

    df = pd.DataFrame(results)

    logger.info(
        f"Coverage statistics computed:\n"
        f"Total net area: {total_area_ha:.2f} ha ({total_pct:.2f}%)"
    )

    return df


def fragmentation_index(
    gdf: gpd.GeoDataFrame
) -> Dict[str, float]:
    """
    Compute fragmentation index per protection class.

    Fragmentation index = number of patches / total protected area (ha).
    Higher values indicate more fragmented protection.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Protected areas GeoDataFrame with 'protection_class' column.
        Must be in EPSG:3035.

    Returns
    -------
    dict
        Dictionary mapping protection_class to fragmentation index.
        {class_name: patches_per_1000_ha}

    Raises
    ------
    ValueError
        If GeoDataFrame is not in EPSG:3035 or missing protection_class column.

    Notes
    -----
    Fragmentation index formula:
        FI = (n_patches / total_area_ha) × 1000

    Expressed as patches per 1000 hectares for interpretability.

    Examples
    --------
    >>> frag = fragmentation_index(gdf_proj)
    >>> print(frag)
    {'strict_core': 12.0, 'regulatory': 25.0, 'contractual': 60.0}
    """
    # Verify CRS
    if gdf.crs != 'EPSG:3035':
        raise ValueError(
            f"GeoDataFrame must be in EPSG:3035. Current CRS: {gdf.crs}"
        )

    # Verify protection_class column
    if 'protection_class' not in gdf.columns:
        raise ValueError("GeoDataFrame must have 'protection_class' column")

    if len(gdf) == 0:
        return {}

    results = {}

    for class_name in gdf['protection_class'].unique():
        class_gdf = gdf[gdf['protection_class'] == class_name]

        # Number of patches
        n_patches = len(class_gdf)

        # Total area (deduplicated)
        class_union = unary_union(class_gdf.geometry)
        area_ha = class_union.area / 10000.0

        # Fragmentation index (patches per 1000 ha)
        if area_ha > 0:
            frag_index = (n_patches / area_ha) * 1000.0
        else:
            frag_index = 0.0

        results[class_name] = frag_index

    logger.debug(f"Fragmentation indices: {results}")
    return results


def kmgbf_indicator(
    gdf: gpd.GeoDataFrame,
    territory_area_ha: float,
    classes: list | None = None,
) -> float:
    """
    Compute KMGBF Target 3 indicator: percentage of territory under effective
    area-based conservation.

    Per CBD COP15 decision 15/4 (Kunming-Montreal GBF), Target 3 counts:
      - All IUCN categories I–VI  (strict_core + regulatory + contractual)
      - Recorded OECMs            (oecm)
    Areas with unassigned/not-reported IUCN status are excluded by default
    because their legal effectiveness cannot be verified.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Protected areas GeoDataFrame with 'protection_class' column.
        Must be in EPSG:3035.
    territory_area_ha : float
        Total territory area in hectares.
    classes : list of str, optional
        Protection classes to include. Default: all IUCN I–VI + OECMs
        ['strict_core', 'regulatory', 'contractual', 'oecm'].
        Pass ['strict_core'] for the old strict-only sub-indicator.

    Returns
    -------
    float
        Percentage of territory under effective protection (0–100).

    Notes
    -----
    Reference: CBD COP15 decision 15/4, Target 3 operative paragraph 2:
    "at least 30 per cent of terrestrial and inland water areas … are
    effectively conserved and managed through … protected areas and other
    effective area-based conservation measures".
    All IUCN categories I–VI and recorded OECMs qualify.  Unassigned areas
    (IUCN 'Not Reported' / 'Not Applicable') are excluded.

    Examples
    --------
    >>> full_pct   = kmgbf_indicator(gdf_proj, territory_area_ha=50000)
    >>> strict_pct = kmgbf_indicator(gdf_proj, territory_area_ha=50000,
    ...                              classes=['strict_core'])
    """
    if gdf.crs != 'EPSG:3035':
        raise ValueError(
            f"GeoDataFrame must be in EPSG:3035. Current CRS: {gdf.crs}"
        )
    if 'protection_class' not in gdf.columns:
        raise ValueError("GeoDataFrame must have 'protection_class' column")
    if len(gdf) == 0 or territory_area_ha == 0:
        return 0.0

    # Default: all classes that qualify under KMGBF Target 3
    if classes is None:
        classes = ['strict_core', 'regulatory', 'contractual', 'oecm']

    filtered = gdf[gdf['protection_class'].isin(classes)]
    if len(filtered) == 0:
        logger.warning("No qualifying protection areas found for KMGBF indicator")
        return 0.0

    area_ha = unary_union(filtered.geometry).area / 10_000.0
    pct = (area_ha / territory_area_ha) * 100.0

    logger.info(
        "KMGBF Target 3 indicator: %.2f%% (%.0f ha / %.0f ha) — classes: %s",
        pct, area_ha, territory_area_ha, classes
    )
    return pct

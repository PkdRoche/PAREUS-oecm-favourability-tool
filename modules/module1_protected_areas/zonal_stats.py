"""Zonal statistics of MCE criterion rasters within protected areas."""

import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, box
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# Canonical IUCN protection-strictness order (least to most permissive),
# used to sort category axes/tables instead of plain alphabetical sort
# (which incorrectly interleaves e.g. "Ia" between "IV" and "V").
_IUCN_CATEGORY_ORDER = [
    'I', 'Ia', 'Ib', 'II', 'III', 'IV', 'V', 'VI',
    'Not Reported', 'Not Applicable', 'Not Assigned',
]


def iucn_category_sort_key(category: str):
    """Sort key placing IUCN categories in canonical order (I, Ia, Ib, II, ...).

    Categories not in the standard list are sorted alphabetically after the
    known ones. 'outside' is not included here — callers append it last.
    """
    try:
        return (0, _IUCN_CATEGORY_ORDER.index(category))
    except ValueError:
        return (1, str(category))


# Default CLC Level 1 labels (first digit of the CLC integer code)
CLC_LEVEL1_LABELS = {
    1: "Urban",
    2: "Agricultural",
    3: "Shrublands / Mixed",
    4: "Forest",
    5: "Water",
}


def clc_composition_by_iucn_category(
    landuse_path: str,
    pa_gdf: gpd.GeoDataFrame,
    level1_labels: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Compute the % composition of CLC Level 1 categories for the whole raster
    extent and within each IUCN protection category.

    Shared by the Module 1 UI (Data Diagnostic tab) and the DOCX report
    generator so the breakdown is computed once and reused, rather than
    re-read/re-rasterized for each consumer.

    Parameters
    ----------
    landuse_path : str
        Path to the CLC (or CLC-coded) raster used as the 'landuse' criterion.
    pa_gdf : gpd.GeoDataFrame
        Protected areas GeoDataFrame with an IUCN_MAX or IUCN_CAT column.
    level1_labels : dict, optional
        Mapping of CLC Level 1 integer code → label. Defaults to
        ``CLC_LEVEL1_LABELS``.

    Returns
    -------
    pd.DataFrame
        Columns: 'level1' (int), 'category' (str), 'Whole Study Area' (float %),
        then one float % column per IUCN category present in ``pa_gdf``, in
        canonical IUCN order. Values are raw percentages, not pre-formatted.
    """
    from rasterio.features import rasterize as _rasterize

    labels = level1_labels if level1_labels is not None else CLC_LEVEL1_LABELS

    with rasterio.open(landuse_path) as src:
        lu_array = src.read(1).astype(float)
        lu_nodata = src.nodata
        lu_transform = src.transform
        lu_crs = src.crs
    if lu_nodata is not None:
        lu_array[lu_array == lu_nodata] = np.nan

    h_lu, w_lu = lu_array.shape
    valid_mask_all = ~np.isnan(lu_array)
    level1_all = np.full(lu_array.shape, -1, dtype=int)
    level1_all[valid_mask_all] = lu_array[valid_mask_all].astype(int) // 100

    def _pct_by_level1(zone_mask: np.ndarray) -> Dict[int, float]:
        selected = level1_all[zone_mask & valid_mask_all]
        if selected.size == 0:
            return {}
        vals, counts = np.unique(selected, return_counts=True)
        return {int(v): 100.0 * c / selected.size for v, c in zip(vals, counts)}

    columns_pct = {'Whole Study Area': _pct_by_level1(np.ones((h_lu, w_lu), dtype=bool))}

    iucn_col = 'IUCN_MAX' if 'IUCN_MAX' in pa_gdf.columns else 'IUCN_CAT'
    if iucn_col in pa_gdf.columns:
        pa_gdf_lu = pa_gdf.to_crs(lu_crs) if pa_gdf.crs != lu_crs else pa_gdf
        cats_sorted = sorted(pa_gdf_lu[iucn_col].unique(), key=iucn_category_sort_key)
        for cat in cats_sorted:
            grp = pa_gdf_lu[pa_gdf_lu[iucn_col] == cat]
            geoms = [(g, 1) for g in grp.geometry if g is not None and not g.is_empty]
            if not geoms:
                continue
            cat_mask = _rasterize(
                shapes=geoms, out_shape=(h_lu, w_lu), transform=lu_transform,
                fill=0, dtype='uint8'
            ).astype(bool)
            columns_pct[str(cat)] = _pct_by_level1(cat_mask)

    all_l1_codes = sorted({code for pct_map in columns_pct.values() for code in pct_map})
    rows = []
    for l1 in all_l1_codes:
        row = {'level1': l1, 'category': labels.get(l1, f"Unknown ({l1})")}
        for col_name, pct_map in columns_pct.items():
            row[col_name] = pct_map.get(l1, 0.0)
        rows.append(row)

    return pd.DataFrame(rows)


def zonal_stats_by_pa_class(
    pa_gdf: gpd.GeoDataFrame,
    raster_paths: Dict[str, str],
    nodata: Optional[float] = None,
    iucn_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute zonal statistics of MCE criterion rasters within protected areas,
    grouped by IUCN category.

    For each (criterion × IUCN category) combination, masks the raster to the
    matching PA polygons and computes summary statistics: mean, median, std, min,
    max, and pixel count. Also computes statistics for areas outside all PAs.

    Parameters
    ----------
    pa_gdf : gpd.GeoDataFrame
        Protected areas GeoDataFrame. Must be in EPSG:3035.
    raster_paths : dict
        Dictionary mapping criterion name (str) to GeoTIFF file path.
        Example: {"ecosystem_condition": "path/to/condition.tif"}
    nodata : float or None, optional
        Custom nodata value to use for all rasters. If None, uses the nodata
        value specified in each raster's metadata.
    iucn_col : str or None, optional
        Column name to use for IUCN category grouping.  When None the function
        auto-selects: 'IUCN_MAX' if present (preferred — matches WDPA best
        designation), else 'IUCN_CAT'.  Pass an explicit value to override.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns:
        - criterion : criterion name
        - iucn_cat : IUCN category (or 'outside' for unprotected areas)
        - mean, median, std, min, max, pixel_count
    """
    # Verify CRS
    if pa_gdf.crs != 'EPSG:3035':
        raise ValueError(
            f"PA GeoDataFrame must be in EPSG:3035 for zonal statistics. "
            f"Current CRS: {pa_gdf.crs}"
        )

    # Resolve which IUCN column to use — same logic as coverage table
    if iucn_col is None:
        iucn_col = 'IUCN_MAX' if 'IUCN_MAX' in pa_gdf.columns else 'IUCN_CAT'

    if iucn_col not in pa_gdf.columns:
        raise ValueError(
            f"Column '{iucn_col}' not found in PA GeoDataFrame. "
            f"Available columns: {list(pa_gdf.columns)}"
        )

    logger.info(f"Grouping zonal stats by column: '{iucn_col}'")

    if len(pa_gdf) == 0:
        logger.warning("PA GeoDataFrame is empty, returning empty statistics")
        return pd.DataFrame(columns=[
            'criterion', 'iucn_cat', 'mean', 'median', 'std', 'min', 'max', 'pixel_count'
        ])

    results = []

    # Process each criterion raster
    for criterion_name, raster_path in raster_paths.items():
        logger.info(f"Processing criterion: {criterion_name} ({raster_path})")

        try:
            with rasterio.open(raster_path) as src:
                # Check CRS and reproject if needed
                raster_crs = src.crs
                if raster_crs is None:
                    logger.warning(
                        f"Raster {raster_path} has no CRS defined. "
                        f"Assuming EPSG:3035."
                    )
                    raster_crs = 'EPSG:3035'

                # Determine nodata value
                nodata_value = nodata if nodata is not None else src.nodata

                # Get IUCN categories
                iucn_categories = pa_gdf[iucn_col].unique()

                # Reproject entire PA GeoDataFrame once (not once per category)
                if str(raster_crs).upper() != 'EPSG:3035':
                    logger.warning(
                        f"Raster CRS ({raster_crs}) differs from EPSG:3035. "
                        f"Reprojecting PA geometries once for all categories."
                    )
                    pa_gdf_masked = pa_gdf.to_crs(raster_crs)
                else:
                    pa_gdf_masked = pa_gdf

                # Process each IUCN category
                for iucn_cat in iucn_categories:
                    class_gdf = pa_gdf_masked[pa_gdf_masked[iucn_col] == iucn_cat]

                    # Union all geometries for this class to avoid overlap
                    union_geom = unary_union(class_gdf.geometry)

                    # Skip empty unions early
                    if getattr(union_geom, 'is_empty', False):
                        logger.warning(
                            f"Empty geometry union for class {iucn_cat} ({criterion_name}); skipping"
                        )
                        continue

                    # If the union is a GeometryCollection, folium/mapping can behave oddly;
                    # flatten to its unary union of parts.
                    try:
                        if union_geom.geom_type == 'GeometryCollection':
                            union_geom = unary_union(getattr(union_geom, 'geoms', []))
                    except Exception:
                        pass

                    # Fast extent-based intersection test against raster bounds.
                    # This avoids rasterio.mask raising ValueError("Input shapes do not overlap raster")
                    # and provides actionable logs.
                    try:
                        rb = src.bounds
                        raster_bounds_geom = box(rb.left, rb.bottom, rb.right, rb.top)
                        if not union_geom.intersects(raster_bounds_geom):
                            logger.warning(
                                f"No raster overlap for {criterion_name} in class {iucn_cat}. "
                                f"Raster bounds={rb}, geom bounds={getattr(union_geom, 'bounds', None)}"
                            )
                            continue
                    except Exception:
                        pass

                    # Mask raster to this IUCN category
                    try:
                        masked_array, _ = mask(
                            src,
                            [mapping(union_geom)],
                            crop=True,
                            nodata=nodata_value,
                            filled=True
                        )

                        # Extract valid pixels (first band)
                        pixels = masked_array[0].astype(np.float64)
                        if nodata_value is not None:
                            valid_pixels = pixels[pixels != nodata_value]
                        else:
                            valid_pixels = pixels
                        valid_pixels = valid_pixels[~np.isnan(valid_pixels)]

                        if len(valid_pixels) > 0:
                            results.append({
                                'criterion': criterion_name,
                                'iucn_cat': iucn_cat,
                                'mean': float(np.mean(valid_pixels)),
                                'median': float(np.median(valid_pixels)),
                                'std': float(np.std(valid_pixels)),
                                'min': float(np.min(valid_pixels)),
                                'max': float(np.max(valid_pixels)),
                                'pixel_count': int(len(valid_pixels))
                            })
                        else:
                            logger.warning(
                                f"No valid pixels for {criterion_name} in IUCN cat {iucn_cat}"
                            )

                    except ValueError as e:
                        logger.warning(
                            f"Failed to mask {criterion_name} for IUCN cat {iucn_cat}: {e}"
                        )
                        continue

                # Compute statistics for areas OUTSIDE all PAs
                # Use a separate try/except so errors here don't swallow per-class results
                try:
                    logger.info(f"Computing outside-PA statistics for {criterion_name}")
                    all_pa_union = unary_union(pa_gdf_masked.geometry)

                    if getattr(all_pa_union, 'is_empty', True):
                        logger.warning(f"Empty PA union for {criterion_name}; skipping outside stats")
                    else:
                        from rasterio.features import geometry_mask
                        raster_data = src.read(1)
                        outside_mask = geometry_mask(
                            [mapping(all_pa_union)],
                            transform=src.transform,
                            invert=False,   # True = outside all PAs
                            out_shape=raster_data.shape
                        )
                        outside_pixels = raster_data[outside_mask].astype(np.float64)
                        if nodata_value is not None:
                            outside_pixels = outside_pixels[outside_pixels != nodata_value]
                        outside_pixels = outside_pixels[~np.isnan(outside_pixels)]

                        if len(outside_pixels) > 0:
                            results.append({
                                'criterion': criterion_name,
                                'iucn_cat': 'outside',
                                'mean': float(np.mean(outside_pixels)),
                                'median': float(np.median(outside_pixels)),
                                'std': float(np.std(outside_pixels)),
                                'min': float(np.min(outside_pixels)),
                                'max': float(np.max(outside_pixels)),
                                'pixel_count': int(len(outside_pixels))
                            })
                        else:
                            logger.warning(
                                f"No valid pixels outside PAs for {criterion_name} "
                                f"(nodata={nodata_value}, outside pixel count before filter="
                                f"{np.sum(outside_mask)})"
                            )
                except Exception as e:
                    logger.error(f"Outside-PA stats failed for {criterion_name}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Failed to process raster {raster_path}: {e}", exc_info=True)
            continue

    # Convert to DataFrame
    if len(results) == 0:
        logger.warning("No zonal statistics computed")
        return pd.DataFrame(columns=[
            'criterion', 'iucn_cat', 'mean', 'median', 'std', 'min', 'max', 'pixel_count'
        ])

    df = pd.DataFrame(results)
    logger.info(
        f"Zonal statistics computed for {len(df)} (criterion × IUCN category) combinations"
    )
    return df


def criterion_coverage_summary(
    zonal_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create pivot table of mean criterion values per PA class.

    Transforms the tidy output of zonal_stats_by_pa_class() into a wide-format
    table for easy comparison across criteria and PA classes.

    Parameters
    ----------
    zonal_df : pd.DataFrame
        Output from zonal_stats_by_pa_class() with columns:
        criterion, pa_class, mean, median, std, min, max, pixel_count

    Returns
    -------
    pd.DataFrame
        Pivot table with:
        - Index: pa_class
        - Columns: criterion names
        - Values: mean criterion values

    Examples
    --------
    >>> summary = criterion_coverage_summary(zonal_df)
    >>> print(summary)
                      ecosystem_condition  connectivity  species_richness
    pa_class
    strict_core                    0.75          0.82              0.68
    regulatory                     0.65          0.74              0.55
    contractual                    0.58          0.68              0.50
    outside                        0.45          0.52              0.38
    """
    if len(zonal_df) == 0:
        logger.warning("Input DataFrame is empty, returning empty summary")
        return pd.DataFrame()

    # Verify required columns
    required_cols = ['criterion', 'iucn_cat', 'mean']
    missing_cols = [col for col in required_cols if col not in zonal_df.columns]
    if missing_cols:
        raise ValueError(
            f"Input DataFrame missing required columns: {missing_cols}"
        )

    # Create pivot table
    pivot = zonal_df.pivot(
        index='iucn_cat',
        columns='criterion',
        values='mean'
    )

    # Sort index in canonical IUCN order (I, Ia, Ib, II, III, IV, V, VI, ...),
    # with 'outside' last if present.
    if 'outside' in pivot.index:
        other_classes = sorted(
            [idx for idx in pivot.index if idx != 'outside'],
            key=iucn_category_sort_key
        )
        pivot = pivot.reindex(other_classes + ['outside'])
    else:
        pivot = pivot.reindex(sorted(pivot.index, key=iucn_category_sort_key))

    logger.info(
        f"Coverage summary created: {len(pivot)} IUCN categories × "
        f"{len(pivot.columns)} criteria"
    )

    return pivot

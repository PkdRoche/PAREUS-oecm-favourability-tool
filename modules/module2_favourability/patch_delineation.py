"""Patch delineation, MMU filtering and candidate OECM site ranking.

Pipeline
--------
1. Threshold the MCE score array → binary raster
2. Connected-component labelling (8-connectivity)
3. Remove patches below the Minimum Mapping Unit (MMU)
4. Vectorise surviving patches → GeoDataFrame in EPSG:3035
5. Compute per-patch attributes:
     area_ha, mean_score, max_score, compactness, dist_to_pa_km,
     gap_overlap_pct, rank_score
6. Rank by rank_score (descending)

Compactness = 4π·area / perimeter²  (Polsby-Popper index, 1 = circle)
Proximity to PA = distance of patch centroid to nearest WDPA geometry (km)
Gap overlap = % of patch area intersecting strict gap layer
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import geopandas as gpd
import pandas as pd
from rasterio.features import shapes as _rasterio_shapes
from shapely.geometry import shape as _shape
from shapely.ops import unary_union
from scipy.ndimage import label as _ndlabel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _smooth_geometry(geom, pixel_size: float):
    """Smooth a raster-derived polygon outline.

    Two-pass approach:
    1. Morphological closing — buffer outward by ~0.6 × pixel_size then inward
       by the same amount.  This rounds concave staircase notches and fills
       single-pixel indentations without eroding the overall shape.
    2. Douglas-Peucker simplification — removes co-linear vertices that were
       produced by the pixel grid, reducing vertex count by ~70-80 %.

    Parameters
    ----------
    geom : shapely geometry
    pixel_size : float
        Pixel side length in the geometry's CRS units (metres for EPSG:3035).
    """
    if geom is None or geom.is_empty:
        return geom
    r   = pixel_size * 0.6          # slightly more than half-pixel
    tol = pixel_size * 0.4          # simplification tolerance
    try:
        smoothed = (
            geom
            .buffer(r, resolution=2)   # resolution=2 keeps buffers fast
            .buffer(-r, resolution=2)
            .simplify(tol, preserve_topology=True)
        )
        return smoothed if not smoothed.is_empty else geom
    except Exception:
        return geom


def _remove_holes(geom, min_hole_area_m2: float = 0.0):
    """Remove interior rings (holes) from polygons.

    Parameters
    ----------
    geom : shapely Polygon or MultiPolygon
    min_hole_area_m2 : float
        Only remove holes whose area is *below* this threshold.
        0 (default) removes **all** holes.
    """
    from shapely.geometry import Polygon, MultiPolygon

    def _fill(poly: Polygon) -> Polygon:
        if min_hole_area_m2 <= 0:
            return Polygon(poly.exterior)
        kept = [r for r in poly.interiors if Polygon(r).area >= min_hole_area_m2]
        return Polygon(poly.exterior, kept)

    if isinstance(geom, Polygon):
        return _fill(geom)
    if isinstance(geom, MultiPolygon):
        filled = [_fill(p) for p in geom.geoms]
        return MultiPolygon([p for p in filled if not p.is_empty])
    return geom


def _polsby_popper(geom) -> float:
    """Polsby-Popper compactness index [0, 1]. 1 = perfect circle."""
    import math
    if geom is None or geom.is_empty:
        return 0.0
    area = geom.area
    perim = geom.length
    if perim == 0:
        return 0.0
    return float(4 * math.pi * area / (perim ** 2))


def _dist_to_pa_km(centroid, pa_union_geom) -> float:
    """Distance in km from centroid to nearest PA boundary (EPSG:3035 metres)."""
    if pa_union_geom is None or pa_union_geom.is_empty:
        return np.nan
    return float(centroid.distance(pa_union_geom)) / 1000.0


def _gap_overlap_pct(patch_geom, gap_geom) -> float:
    """% of patch area that overlaps the strict-gap layer."""
    if gap_geom is None or gap_geom.is_empty:
        return 0.0
    try:
        inter = patch_geom.intersection(gap_geom)
        if inter.is_empty:
            return 0.0
        return float(inter.area / patch_geom.area * 100.0)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def delineate_patches(
    score_array: np.ndarray,
    profile: dict,
    threshold: float,
    mmu_ha: float = 100.0,
    pa_gdf: Optional[gpd.GeoDataFrame] = None,
    strict_gaps_gdf: Optional[gpd.GeoDataFrame] = None,
    smooth: bool = True,
    min_hole_area_ha: float = 1.0,
) -> gpd.GeoDataFrame:
    """
    Delineate candidate OECM sites from the MCE score raster.

    Parameters
    ----------
    score_array : np.ndarray
        MCE favourability score (NaN = eliminated).
    profile : dict
        Rasterio profile (transform, crs, width, height).
    threshold : float
        Minimum score for a pixel to be included in a candidate patch.
    mmu_ha : float
        Minimum Mapping Unit in hectares. Patches smaller than this are
        discarded.
    pa_gdf : GeoDataFrame, optional
        WDPA protected area polygons (EPSG:3035). Used to compute
        distance-to-PA attribute.
    strict_gaps_gdf : GeoDataFrame, optional
        Strict gap layer from Module 1 gap analysis (EPSG:3035). Used to
        compute gap overlap attribute.
    smooth : bool, default True
        Apply morphological closing + Douglas-Peucker simplification to
        remove staircase pixel edges from the vectorised outlines.
    min_hole_area_ha : float, default 1.0
        Remove interior holes whose area is below this threshold (ha).
        Set to 0 to remove *all* holes; set to a large value to keep them.

    Returns
    -------
    GeoDataFrame
        One row per candidate site, columns:
        patch_id, area_ha, mean_score, max_score, compactness,
        dist_to_pa_km, gap_overlap_pct, rank_score, geometry (EPSG:3035)
        Sorted by rank_score descending.
    """
    transform     = profile['transform']
    crs           = profile['crs']
    pixel_size    = abs(transform[0])                            # metres (EPSG:3035)
    pixel_area_ha = abs(transform[0] * transform[4]) / 10000.0
    min_hole_m2   = min_hole_area_ha * 10_000.0

    # ------------------------------------------------------------------
    # 1. Binary mask at threshold
    # ------------------------------------------------------------------
    valid     = ~np.isnan(score_array)
    binary    = (valid & (score_array >= threshold)).astype(np.uint8)

    if binary.sum() == 0:
        logger.warning("No pixels above threshold — empty patch GeoDataFrame.")
        return gpd.GeoDataFrame(
            columns=['patch_id', 'area_ha', 'mean_score', 'max_score',
                     'compactness', 'dist_to_pa_km', 'gap_overlap_pct',
                     'rank_score', 'geometry'],
            geometry='geometry',
            crs=crs,
        )

    # ------------------------------------------------------------------
    # 2. Connected components (8-connectivity)
    # ------------------------------------------------------------------
    struct   = np.ones((3, 3), dtype=int)   # 8-connectivity
    labelled, n_patches = _ndlabel(binary, structure=struct)
    logger.info(f"Found {n_patches} raw patches before MMU filter")

    # ------------------------------------------------------------------
    # 3. MMU filter — remove patches below mmu_ha
    # ------------------------------------------------------------------
    mmu_pixels = mmu_ha / pixel_area_ha
    patch_sizes = np.bincount(labelled.ravel())  # index 0 = background

    keep_ids = np.where(patch_sizes >= mmu_pixels)[0]
    keep_ids = keep_ids[keep_ids > 0]            # exclude background (0)

    if len(keep_ids) == 0:
        logger.warning(
            f"All {n_patches} patches smaller than MMU ({mmu_ha:.0f} ha). "
            "Lower the MMU or threshold."
        )
        return gpd.GeoDataFrame(
            columns=['patch_id', 'area_ha', 'mean_score', 'max_score',
                     'compactness', 'dist_to_pa_km', 'gap_overlap_pct',
                     'rank_score', 'geometry'],
            geometry='geometry',
            crs=crs,
        )

    # Zero-out small patches for vectorisation
    keep_mask = np.isin(labelled, keep_ids)
    filtered_labels = np.where(keep_mask, labelled, 0).astype(np.int32)

    logger.info(
        f"{len(keep_ids)} patches survive MMU filter "
        f"({n_patches - len(keep_ids)} removed)"
    )

    # ------------------------------------------------------------------
    # 4. Pre-compute spatial context (once, outside patch loop)
    # ------------------------------------------------------------------
    pa_union = None
    if pa_gdf is not None and len(pa_gdf) > 0:
        try:
            pa_union = pa_gdf.geometry.union_all()
        except Exception as e:
            logger.warning(f"Could not compute PA union for proximity: {e}")

    gap_union = None
    if strict_gaps_gdf is not None and len(strict_gaps_gdf) > 0:
        try:
            gap_union = strict_gaps_gdf.geometry.union_all()
        except Exception as e:
            logger.warning(f"Could not compute gap union: {e}")

    # ------------------------------------------------------------------
    # 5. Vectorise patches and compute attributes
    # ------------------------------------------------------------------
    records = []

    for pid in keep_ids:
        patch_binary = (filtered_labels == pid).astype(np.uint8)

        # --- Vectorise via rasterio.features.shapes ---
        geoms = [
            _shape(geom_dict)
            for geom_dict, val in _rasterio_shapes(
                patch_binary, mask=patch_binary, transform=transform
            )
            if val == 1
        ]
        if not geoms:
            continue
        patch_geom = unary_union(geoms)
        if patch_geom.is_empty:
            continue

        # --- Post-vectorisation geometry improvement ---
        # Remove holes first (before smoothing, so small pixel-gaps don't
        # get locked in by the buffer pass)
        patch_geom = _remove_holes(patch_geom, min_hole_area_m2=min_hole_m2)
        # Smooth staircase pixel edges
        if smooth:
            patch_geom = _smooth_geometry(patch_geom, pixel_size=pixel_size)
        if patch_geom is None or patch_geom.is_empty:
            continue

        # --- Score statistics ---
        pixel_scores = score_array[filtered_labels == pid]
        pixel_scores = pixel_scores[~np.isnan(pixel_scores)]
        if len(pixel_scores) == 0:
            continue

        area_ha   = len(pixel_scores) * pixel_area_ha
        mean_sc   = float(np.mean(pixel_scores))
        max_sc    = float(np.max(pixel_scores))

        # --- Spatial attributes ---
        compact   = _polsby_popper(patch_geom)
        dist_pa   = _dist_to_pa_km(patch_geom.centroid, pa_union)
        gap_pct   = _gap_overlap_pct(patch_geom, gap_union)

        records.append({
            'patch_id':        int(pid),
            'area_ha':         round(area_ha, 1),
            'mean_score':      round(mean_sc, 4),
            'max_score':       round(max_sc, 4),
            'compactness':     round(compact, 3),
            'dist_to_pa_km':   round(dist_pa, 2) if not np.isnan(dist_pa) else None,
            'gap_overlap_pct': round(gap_pct, 1),
            'geometry':        patch_geom,
        })

    if not records:
        return gpd.GeoDataFrame(
            columns=['patch_id', 'area_ha', 'mean_score', 'max_score',
                     'compactness', 'dist_to_pa_km', 'gap_overlap_pct',
                     'rank_score', 'geometry'],
            geometry='geometry',
            crs=crs,
        )

    gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=crs)

    # ------------------------------------------------------------------
    # 6. Rank score
    #
    # Composite of:
    #   - 50 % mean favourability score
    #   - 20 % gap overlap (high gap overlap = high OECM complementarity)
    #   - 20 % area (log-scaled, normalised)
    #   - 10 % PA proximity (closer = better — distance inverted)
    # ------------------------------------------------------------------
    def _norm01(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(np.ones(len(series)), index=series.index)
        return (series - mn) / (mx - mn)

    score_norm    = _norm01(gdf['mean_score'])
    gap_norm      = _norm01(gdf['gap_overlap_pct'])
    area_norm     = _norm01(np.log1p(gdf['area_ha']))

    # Proximity: closer is better → invert distance
    if gdf['dist_to_pa_km'].notna().any():
        prox_norm = _norm01(-gdf['dist_to_pa_km'].fillna(gdf['dist_to_pa_km'].max()))
    else:
        prox_norm = pd.Series(np.zeros(len(gdf)), index=gdf.index)

    gdf['rank_score'] = (
        0.50 * score_norm +
        0.20 * gap_norm   +
        0.20 * area_norm  +
        0.10 * prox_norm
    ).round(4)

    gdf = gdf.sort_values('rank_score', ascending=False).reset_index(drop=True)
    gdf['patch_id'] = gdf.index + 1   # renumber 1-based after sort

    logger.info(
        f"Delineated {len(gdf)} candidate OECM sites. "
        f"Top site: {gdf.iloc[0]['area_ha']:.0f} ha, "
        f"score={gdf.iloc[0]['mean_score']:.3f}"
    )
    return gdf


def evaluate_external_sites(
    sites_gdf: gpd.GeoDataFrame,
    score_array: np.ndarray,
    profile: dict,
    oecm_mask: Optional[np.ndarray] = None,
    pa_gdf: Optional[gpd.GeoDataFrame] = None,
    strict_gaps_gdf: Optional[gpd.GeoDataFrame] = None,
    name_col: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Evaluate externally-imported candidate site polygons against the
    already-computed MCE favourability score.

    Applies the same methodology and attributes as delineate_patches() —
    including the identical rank_score formula, for direct comparability —
    but to user-supplied geometries instead of raster-threshold-derived
    patches. This is how an analyst checks the relevance of a proposed
    OECM site against the exact same MCE criteria (weights, aggregation,
    eliminatory thresholds) used everywhere else in Module 2.

    Parameters
    ----------
    sites_gdf : GeoDataFrame
        Imported candidate site polygons, any CRS (reprojected to the score
        raster's CRS internally).
    score_array : np.ndarray
        MCE favourability score from compute_favourability() (NaN = pixel
        eliminated by Group D — high pressure or incompatible land use).
    profile : dict
        Rasterio profile matching score_array (transform, crs, width, height).
    oecm_mask : np.ndarray, optional
        Boolean mask from compute_favourability() — True where Group C score
        clears the OECM-favourable threshold. Used to report what fraction
        of each site is OECM-favourable vs. better suited to a classical PA.
    pa_gdf : GeoDataFrame, optional
        WDPA protected area polygons — distance-to-PA attribute.
    strict_gaps_gdf : GeoDataFrame, optional
        Strict gap layer from Module 1 — gap overlap attribute.
    name_col : str, optional
        Column in sites_gdf to use as the site label (e.g. a name/ID field
        from the imported file). Falls back to "Site 1", "Site 2", ... if
        not given or not present.

    Returns
    -------
    GeoDataFrame
        One row per input site (original row order preserved as `site_id`),
        columns: site_id, site_name, area_ha, mean_score, max_score,
        min_score, pct_eliminated, pct_oecm_favourable, compactness,
        dist_to_pa_km, gap_overlap_pct, rank_score, geometry (score CRS).
        A site with no overlap with the aligned raster grid gets NaN score
        columns rather than being dropped, so it still shows up as a
        "no data / outside study area" result.
    """
    from rasterio.features import rasterize as _rasterize

    transform = profile['transform']
    crs       = profile['crs']
    h, w      = profile['height'], profile['width']

    if sites_gdf.crs is not None and str(sites_gdf.crs) != str(crs):
        sites_gdf = sites_gdf.to_crs(crs)
    elif sites_gdf.crs is None:
        logger.warning(
            "Imported sites have no CRS — assuming it already matches the "
            "score raster's CRS (%s). Verify this is correct.", crs
        )

    pa_union = None
    if pa_gdf is not None and len(pa_gdf) > 0:
        try:
            pa_union = pa_gdf.geometry.union_all()
        except Exception as e:
            logger.warning(f"Could not compute PA union for proximity: {e}")

    gap_union = None
    if strict_gaps_gdf is not None and len(strict_gaps_gdf) > 0:
        try:
            gap_union = strict_gaps_gdf.geometry.union_all()
        except Exception as e:
            logger.warning(f"Could not compute gap union: {e}")

    records = []
    for i, row in sites_gdf.reset_index(drop=True).iterrows():
        geom = row.geometry
        site_name = (
            str(row[name_col]) if name_col and name_col in sites_gdf.columns
            else f"Site {i + 1}"
        )

        if geom is None or geom.is_empty:
            records.append({
                'site_id': i + 1, 'site_name': site_name, 'area_ha': None,
                'mean_score': np.nan, 'max_score': np.nan, 'min_score': np.nan,
                'pct_eliminated': np.nan, 'pct_oecm_favourable': np.nan,
                'compactness': np.nan, 'dist_to_pa_km': None,
                'gap_overlap_pct': np.nan, 'geometry': geom,
            })
            continue

        site_mask = _rasterize(
            shapes=[(geom, 1)], out_shape=(h, w), transform=transform,
            fill=0, dtype='uint8'
        ).astype(bool)
        n_overlap = int(site_mask.sum())

        if n_overlap == 0:
            logger.warning(
                "Site '%s' has no overlap with the aligned raster grid — "
                "likely outside the study area.", site_name
            )
            pixel_scores = np.array([])
        else:
            pixel_scores = score_array[site_mask]

        valid_scores = pixel_scores[~np.isnan(pixel_scores)] if n_overlap else pixel_scores

        if n_overlap > 0:
            pct_eliminated = 100.0 * (1.0 - len(valid_scores) / n_overlap)
        else:
            pct_eliminated = np.nan

        if len(valid_scores) > 0:
            mean_sc, max_sc, min_sc = (
                float(np.mean(valid_scores)), float(np.max(valid_scores)),
                float(np.min(valid_scores)),
            )
        else:
            mean_sc = max_sc = min_sc = np.nan

        if oecm_mask is not None and n_overlap > 0:
            pct_oecm = 100.0 * float(oecm_mask[site_mask].sum()) / n_overlap
        else:
            pct_oecm = np.nan

        compact = _polsby_popper(geom)
        dist_pa = _dist_to_pa_km(geom.centroid, pa_union)
        gap_pct = _gap_overlap_pct(geom, gap_union)

        records.append({
            'site_id':             i + 1,
            'site_name':           site_name,
            'area_ha':             round(geom.area / 10_000.0, 1),
            'mean_score':          round(mean_sc, 4) if not np.isnan(mean_sc) else np.nan,
            'max_score':           round(max_sc, 4) if not np.isnan(max_sc) else np.nan,
            'min_score':           round(min_sc, 4) if not np.isnan(min_sc) else np.nan,
            'pct_eliminated':      round(pct_eliminated, 1) if not np.isnan(pct_eliminated) else np.nan,
            'pct_oecm_favourable': round(pct_oecm, 1) if not np.isnan(pct_oecm) else np.nan,
            'compactness':         round(compact, 3),
            'dist_to_pa_km':       round(dist_pa, 2) if not np.isnan(dist_pa) else None,
            'gap_overlap_pct':     round(gap_pct, 1),
            'geometry':            geom,
        })

    gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=crs)

    # Same rank_score formula as delineate_patches(), for direct
    # comparability with auto-delineated candidate sites. Sites with no
    # valid score (outside the study area / fully eliminated) get rank 0
    # rather than being excluded from normalisation.
    def _norm01(series: pd.Series) -> pd.Series:
        s = series.dropna()
        if len(s) == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(np.where(series.notna(), 1.0, 0.0), index=series.index)
        return ((series - mn) / (mx - mn)).fillna(0.0)

    score_norm = _norm01(gdf['mean_score'])
    gap_norm   = _norm01(gdf['gap_overlap_pct'])
    area_norm  = _norm01(np.log1p(gdf['area_ha'].fillna(0)))

    if gdf['dist_to_pa_km'].notna().any():
        prox_norm = _norm01(-gdf['dist_to_pa_km'].fillna(gdf['dist_to_pa_km'].max()))
    else:
        prox_norm = pd.Series(np.zeros(len(gdf)), index=gdf.index)

    gdf['rank_score'] = (
        0.50 * score_norm +
        0.20 * gap_norm   +
        0.20 * area_norm  +
        0.10 * prox_norm
    ).round(4)

    gdf = gdf.sort_values('rank_score', ascending=False).reset_index(drop=True)

    logger.info(f"Evaluated {len(gdf)} externally-imported candidate site(s).")
    return gdf

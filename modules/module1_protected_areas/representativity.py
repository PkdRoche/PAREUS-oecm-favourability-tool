"""Ecosystem representativity assessment (KMGBF Target 3)."""

import logging
from typing import Dict, Optional
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLC → broad ecosystem-type groupings (for raster-based RI)
# Artificial surfaces (1xx) are excluded — not conservation targets.
# ---------------------------------------------------------------------------
_CLC_ECOSYSTEM_GROUPS: dict[str, frozenset[int]] = {
    'Forests':                frozenset({311, 312, 313}),
    'Grasslands & heathland': frozenset({321, 322, 323, 324}),
    'Wetlands':               frozenset({411, 412, 421, 422, 423}),
    'Water bodies':           frozenset({511, 512, 521, 522, 523}),
    'Semi-natural open land': frozenset({331, 332, 333, 334, 335}),
    'Agricultural areas':     frozenset({211, 212, 213, 221, 222, 223,
                                          231, 241, 242, 243, 244}),
}


def cross_with_ecosystem_types(
    pa_gdf: gpd.GeoDataFrame,
    ecosystem_layer: gpd.GeoDataFrame,
    type_column: str = "ecosystem_type"
) -> pd.DataFrame:
    """
    Spatial intersection of PA polygons with ecosystem type polygons.

    Computes the area of each ecosystem type covered by each protection class.

    Parameters
    ----------
    pa_gdf : gpd.GeoDataFrame
        Protected areas GeoDataFrame with 'protection_class' column.
        Must be in EPSG:3035.
    ecosystem_layer : gpd.GeoDataFrame
        Ecosystem types GeoDataFrame with ecosystem type classification column.
        Must be in EPSG:3035.
    type_column : str, default='ecosystem_type'
        Column name in ecosystem_layer containing ecosystem type labels.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ecosystem_type : ecosystem type label
        - pa_class : protection class
        - area_ha : area of intersection in hectares

    Raises
    ------
    ValueError
        If input GeoDataFrames are not in EPSG:3035 or missing required columns.

    Notes
    -----
    - Both input GeoDataFrames must be in EPSG:3035 before calling.
    - Performs spatial overlay (intersection) between PA and ecosystem layers.
    - Areas are computed in square meters and converted to hectares.

    Examples
    --------
    >>> coverage_df = cross_with_ecosystem_types(
    ...     pa_gdf=protected_areas,
    ...     ecosystem_layer=ecosystems,
    ...     type_column='eunis_class'
    ... )
    >>> coverage_df.head()
      ecosystem_type    pa_class  area_ha
    0        forests strict_core   1250.5
    1        forests  regulatory    850.3
    2       wetlands strict_core    320.1
    """
    # Verify CRS
    if pa_gdf.crs != 'EPSG:3035':
        raise ValueError(f"PA GeoDataFrame must be in EPSG:3035. Current: {pa_gdf.crs}")

    if ecosystem_layer.crs != 'EPSG:3035':
        raise ValueError(f"Ecosystem layer must be in EPSG:3035. Current: {ecosystem_layer.crs}")

    # Verify required columns
    if 'protection_class' not in pa_gdf.columns:
        raise ValueError("PA GeoDataFrame must have 'protection_class' column")

    if type_column not in ecosystem_layer.columns:
        raise ValueError(f"Ecosystem layer must have '{type_column}' column")

    logger.info("Computing spatial intersection between PAs and ecosystem types...")

    # Perform spatial overlay
    overlay = gpd.overlay(
        pa_gdf[['protection_class', 'geometry']],
        ecosystem_layer[[type_column, 'geometry']],
        how='intersection'
    )

    if len(overlay) == 0:
        logger.warning("No spatial overlap found between PAs and ecosystem layer")
        return pd.DataFrame(columns=['ecosystem_type', 'pa_class', 'area_ha'])

    # Compute area for each intersection
    overlay['area_ha'] = overlay.geometry.area / 10000.0

    # Group by ecosystem type and protection class
    result = overlay.groupby([type_column, 'protection_class'])['area_ha'].sum().reset_index()
    result.columns = ['ecosystem_type', 'pa_class', 'area_ha']

    logger.info(f"Computed coverage for {len(result)} ecosystem-PA class combinations")
    return result


def representativity_index(
    coverage_df: pd.DataFrame,
    territory_totals: Dict[str, float],
    target_threshold: float = 0.30
) -> pd.DataFrame:
    """
    Compute representativity index (RI) per ecosystem type.

    Formula: RI_e = min(coverage_e / target_threshold_e, 1.0)
    Synthetic RI: mean of all RI_e values.

    Parameters
    ----------
    coverage_df : pd.DataFrame
        Output from cross_with_ecosystem_types with columns:
        ecosystem_type, pa_class, area_ha.
    territory_totals : dict
        Dictionary mapping ecosystem_type to total area (ha) in territory.
        Example: {'forests': 10000.0, 'wetlands': 5000.0, 'grasslands': 8000.0}
    target_threshold : float, default=0.30
        Target protection percentage (0.30 = 30% KMGBF Target 3).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ecosystem_type : ecosystem type label
        - total_ha : total area of this type in territory
        - protected_ha : area under protection (all classes)
        - coverage_pct : percentage protected (0-100)
        - RI : representativity index (0-1, capped at 1.0)
        - gap_ha : remaining area needed to reach target

    Notes
    -----
    RI interpretation:
    - RI = 1.0 : target threshold achieved or exceeded
    - RI < 1.0 : under-represented ecosystem type
    - RI = 0.0 : no protection

    The synthetic RI is the mean of all individual RI_e values and represents
    overall network representativity.

    Examples
    --------
    >>> territory_totals = {'forests': 10000, 'wetlands': 5000, 'grasslands': 8000}
    >>> ri_df = representativity_index(coverage_df, territory_totals, target_threshold=0.30)
    >>> print(ri_df)
      ecosystem_type  total_ha  protected_ha  coverage_pct    RI   gap_ha
    0        forests   10000.0        3500.0          35.0  1.00      0.0
    1       wetlands    5000.0         800.0          16.0  0.53    700.0
    2     grasslands    8000.0        1200.0          15.0  0.50   1200.0
    >>> synthetic_ri = ri_df['RI'].mean()
    >>> print(f"Synthetic RI: {synthetic_ri:.2f}")
    """
    if len(coverage_df) == 0:
        logger.warning("Empty coverage DataFrame — returning RI=0.0 for all ecosystem types")
        if not territory_totals:
            return pd.DataFrame(
                columns=['ecosystem_type', 'total_ha', 'protected_ha', 'coverage_pct', 'RI', 'gap_ha']
            )
        return pd.DataFrame([{
            'ecosystem_type': eco_type,
            'total_ha': total_ha,
            'protected_ha': 0.0,
            'coverage_pct': 0.0,
            'RI': 0.0,
            'gap_ha': total_ha * target_threshold,
        } for eco_type, total_ha in territory_totals.items()])

    # Aggregate protected area by ecosystem type (sum across all PA classes)
    protected = coverage_df.groupby('ecosystem_type')['area_ha'].sum().to_dict()

    results = []

    for eco_type, total_ha in territory_totals.items():
        protected_ha = protected.get(eco_type, 0.0)

        # Coverage percentage
        coverage_pct = (protected_ha / total_ha * 100.0) if total_ha > 0 else 0.0

        # Representativity index (capped at 1.0)
        coverage_fraction = protected_ha / total_ha if total_ha > 0 else 0.0
        ri = min(coverage_fraction / target_threshold, 1.0)

        # Gap to target
        target_ha = total_ha * target_threshold
        gap_ha = max(0.0, target_ha - protected_ha)

        results.append({
            'ecosystem_type': eco_type,
            'total_ha': total_ha,
            'protected_ha': protected_ha,
            'coverage_pct': coverage_pct,
            'RI': ri,
            'gap_ha': gap_ha
        })

    df = pd.DataFrame(results)

    # Compute synthetic RI
    synthetic_ri = df['RI'].mean()

    logger.info(
        f"Representativity analysis complete. Synthetic RI: {synthetic_ri:.3f}\n"
        f"Ecosystem types at target: {(df['RI'] >= 1.0).sum()} / {len(df)}"
    )

    return df


def representativity_from_clc_raster(
    clc_path: str,
    pa_gdf: gpd.GeoDataFrame,
    target_threshold: float = 0.30,
    ecosystem_groups: Optional[dict] = None,
    study_area_geom=None,
) -> pd.DataFrame:
    """
    Compute ecosystem representativity directly from the CLC raster.

    Avoids expensive vector overlay by working entirely in raster space:
    1. Load CLC raster (integer CLC codes 111–523).
    2. Rasterize the study area geometry to restrict analysis to the territory.
    3. Rasterize PA polygons onto the same grid.
    4. For each ecosystem group, count total pixels and PA-covered pixels
       **within the study area only**.
    5. Convert pixel counts to hectares and compute RI.

    Parameters
    ----------
    clc_path : str
        Path to the CLC GeoTIFF (EPSG:3035, integer CLC codes).
    pa_gdf : gpd.GeoDataFrame
        Protected areas GeoDataFrame (any CRS — reprojected internally).
    target_threshold : float
        KMGBF protection target as a fraction (default 0.30 = 30%).
    ecosystem_groups : dict, optional
        Mapping of ecosystem-type label → set of CLC integer codes.
        Defaults to the built-in ``_CLC_ECOSYSTEM_GROUPS`` (6 categories).
    study_area_geom : shapely geometry, optional
        Territory polygon used to restrict pixel counting to the study area.
        **Without this, totals are computed over the entire CLC raster extent
        (all of Europe), producing near-zero coverage percentages.**
        Pass ``st.session_state.get('study_area_geometry')`` from the UI.

    Returns
    -------
    pd.DataFrame
        Same schema as ``representativity_index``:
        ecosystem_type, total_ha, protected_ha, coverage_pct, RI, gap_ha.
        Sorted by coverage_pct ascending (most under-represented first).
    """
    import rasterio
    from rasterio.features import rasterize as _rasterize

    groups = ecosystem_groups if ecosystem_groups is not None else _CLC_ECOSYSTEM_GROUPS

    # ── Load CLC raster — windowed read when study area is available ─────────
    # EU-wide CLC at 100 m is ~37 M pixels. Reading only the study-area window
    # reduces I/O and memory by ~100× for a country-sized territory.
    def _load_clc():
        if study_area_geom is not None and not study_area_geom.is_empty:
            try:
                from modules.module2_favourability.raster_preprocessing import (
                    load_raster_windowed as _lw,
                )
                return _lw(clc_path, clip_geom=study_area_geom, geom_crs='EPSG:3035')
            except Exception as _e:
                logger.warning("Windowed CLC read failed (%s) — using full read", _e)
        with rasterio.open(clc_path) as _s:
            return _s.read(1), dict(_s.profile)

    clc_array, _clc_prof = _load_clc()

    with rasterio.open(clc_path) as src:
        nodata = src.nodata or 0
        crs    = _clc_prof.get('crs') or src.crs

    transform = _clc_prof['transform']

    pixel_area_ha = abs(transform[0]) * abs(transform[4]) / 10_000.0
    h, w = clc_array.shape

    # ── Rasterize study area → territory mask ────────────────────────────────
    # CRITICAL: without this mask, total_px counts the entire CLC extent
    # (all of Europe), making coverage appear near 0% for any local territory.
    if study_area_geom is not None and not study_area_geom.is_empty:
        try:
            import geopandas as _gpd
            # Reproject study area to raster CRS
            sa_gdf   = _gpd.GeoDataFrame(geometry=[study_area_geom], crs='EPSG:3035')
            sa_repr  = sa_gdf.to_crs(crs).geometry.iloc[0]
            sa_union = unary_union(
                [g for g in ([sa_repr] if sa_repr.geom_type != 'GeometryCollection'
                             else sa_repr.geoms)
                 if g is not None and not g.is_empty]
            )
            territory_mask = _rasterize(
                shapes=[(sa_union, 1)],
                out_shape=(h, w),
                transform=transform,
                fill=0,
                dtype='uint8',
            ).astype(bool)
            logger.info("Study area mask applied: %d territory pixels", territory_mask.sum())
        except Exception as _e:
            logger.warning("Could not rasterize study area — using full raster extent: %s", _e)
            territory_mask = np.ones((h, w), dtype=bool)
    else:
        logger.warning(
            "No study_area_geom provided to representativity_from_clc_raster. "
            "Totals will cover the full CLC raster extent (likely all of Europe). "
            "Pass study_area_geom=st.session_state.get('study_area_geometry')."
        )
        territory_mask = np.ones((h, w), dtype=bool)

    # ── Rasterize PA polygons onto the CLC grid ───────────────────────────────
    # Dissolve to a single (Multi)Polygon before rasterizing — much faster than
    # passing thousands of individual polygon shapes to rasterio.features.rasterize.
    if len(pa_gdf) > 0:
        pa_repr   = pa_gdf.to_crs(crs)
        dissolved = unary_union(
            [g for g in pa_repr.geometry if g is not None and not g.is_empty]
        )
        if dissolved is not None and not dissolved.is_empty:
            pa_mask = _rasterize(
                shapes=[(dissolved, 1)],
                out_shape=(h, w),
                transform=transform,
                fill=0,
                dtype='uint8',
            ).astype(bool)
        else:
            pa_mask = np.zeros((h, w), dtype=bool)
    else:
        pa_mask = np.zeros((h, w), dtype=bool)

    # ── Per-protection-class PA masks (for breakdown table) ──────────────────
    _PA_CLASSES = ['strict_core', 'regulatory', 'contractual', 'unassigned']
    class_masks: dict[str, np.ndarray] = {}
    if 'protection_class' in pa_gdf.columns:
        for _cls in _PA_CLASSES:
            _sub = pa_gdf[pa_gdf['protection_class'] == _cls]
            if len(_sub) > 0:
                _sub_repr = _sub.to_crs(crs)
                _sub_diss = unary_union(
                    [g for g in _sub_repr.geometry if g is not None and not g.is_empty]
                )
                if _sub_diss is not None and not _sub_diss.is_empty:
                    class_masks[_cls] = _rasterize(
                        shapes=[(_sub_diss, 1)],
                        out_shape=(h, w),
                        transform=transform,
                        fill=0,
                        dtype='uint8',
                    ).astype(bool)
                else:
                    class_masks[_cls] = np.zeros((h, w), dtype=bool)
            else:
                class_masks[_cls] = np.zeros((h, w), dtype=bool)

    # ── Pixel-count RI per ecosystem group (within territory only) ────────────
    valid = (clc_array != nodata) & (clc_array != 0) & territory_mask
    results       = []
    class_results = []

    for eco_type, code_set in groups.items():
        codes    = np.array(list(code_set), dtype=clc_array.dtype)
        eco_mask = np.isin(clc_array, codes) & valid

        total_px     = int(eco_mask.sum())
        protected_px = int((eco_mask & pa_mask).sum())

        total_ha     = total_px     * pixel_area_ha
        protected_ha = protected_px * pixel_area_ha

        if total_ha == 0:
            continue                         # ecosystem type absent from territory

        coverage_pct = protected_ha / total_ha * 100.0
        ri           = min(coverage_pct / (target_threshold * 100.0), 1.0)
        gap_ha       = max(0.0, total_ha * target_threshold - protected_ha)

        results.append({
            'ecosystem_type': eco_type,
            'total_ha':       round(total_ha,     1),
            'protected_ha':   round(protected_ha, 1),
            'coverage_pct':   round(coverage_pct, 2),
            'RI':             round(ri,            3),
            'gap_ha':         round(gap_ha,        1),
        })

        # Per-class breakdown row
        if class_masks:
            _row: dict = {'ecosystem_type': eco_type, 'total_ha': round(total_ha, 1)}
            for _cls in _PA_CLASSES:
                _m = class_masks.get(_cls)
                if _m is not None:
                    _cls_ha = int((eco_mask & _m).sum()) * pixel_area_ha
                else:
                    _cls_ha = 0.0
                _row[_cls] = round(_cls_ha / total_ha * 100.0, 1)
            class_results.append(_row)

    df = pd.DataFrame(results).sort_values('coverage_pct').reset_index(drop=True)
    class_df = (
        pd.DataFrame(class_results)
        .sort_values('total_ha', ascending=False)
        .reset_index(drop=True)
        if class_results else pd.DataFrame()
    )

    if len(df) > 0:
        logger.info(
            "CLC-based RI: synthetic RI = %.3f  (%d / %d ecosystem types at target)",
            df['RI'].mean(),
            (df['RI'] >= 1.0).sum(),
            len(df),
        )
    return df, class_df


def propose_group_a_weights(
    ri_df: pd.DataFrame,
    criterion_ecosystem_mapping: Dict[str, str]
) -> Dict[str, float]:
    """
    Derive default Group A criterion weights from representativity deficits.

    Formula: w_i = deficit_e / Σ deficit_e
    where deficit_e = max(0, target_threshold_e - current_coverage_e)

    Only informs intra-Group A weights. Returns normalised dict (Σ = 1.0).

    Parameters
    ----------
    ri_df : pd.DataFrame
        Output from representativity_index with columns:
        ecosystem_type, total_ha, protected_ha, coverage_pct, RI, gap_ha.
    criterion_ecosystem_mapping : dict
        Mapping from Group A criterion name to ecosystem type.
        From config/criteria_defaults.yaml.
        Example: {
            'ecosystem_condition': 'all',
            'regulating_es': 'wetlands',
            'low_pressure': 'all'
        }

    Returns
    -------
    dict
        Normalised weights for Group A criteria (sum = 1.0).
        Example: {
            'ecosystem_condition': 0.35,
            'regulating_es': 0.45,
            'low_pressure': 0.20
        }

    Warnings
    --------
    This function ONLY proposes weights for Group A criteria (ecological
    integrity). It CANNOT inform inter-group weights (W_A, W_B, W_C) which
    require user value judgement.

    The mapping between criteria and ecosystem types must be defined manually
    in config/criteria_defaults.yaml and reflects expert judgement about which
    criteria best capture conservation priorities for each ecosystem.

    Notes
    -----
    - If an ecosystem type is mapped to 'all', the average deficit across all
      types is used.
    - If total deficit is zero (all types at target), equal weights are returned.
    - Weights are normalised to sum to 1.0.

    Examples
    --------
    >>> criterion_mapping = {
    ...     'ecosystem_condition': 'all',
    ...     'regulating_es': 'wetlands',
    ...     'low_pressure': 'all'
    ... }
    >>> weights = propose_group_a_weights(ri_df, criterion_mapping)
    >>> print(weights)
    {'ecosystem_condition': 0.33, 'regulating_es': 0.47, 'low_pressure': 0.20}
    >>> assert abs(sum(weights.values()) - 1.0) < 1e-6
    """
    logger.warning(
        "Weight proposal is valid ONLY for Group A criteria. "
        "Inter-group weights (W_A, W_B, W_C) require user value judgement and "
        "cannot be derived from representativity deficits alone."
    )

    if len(ri_df) == 0:
        logger.warning("Empty representativity DataFrame, returning equal weights")
        n_criteria = len(criterion_ecosystem_mapping)
        equal_weight = 1.0 / n_criteria if n_criteria > 0 else 0.0
        return {crit: equal_weight for crit in criterion_ecosystem_mapping.keys()}

    # Build deficit lookup by ecosystem type
    deficit_by_type = ri_df.set_index('ecosystem_type')['gap_ha'].to_dict()

    # Compute average deficit for 'all' mapping
    avg_deficit = ri_df['gap_ha'].mean()

    # Map criteria to deficits
    criterion_deficits = {}

    for criterion, eco_type in criterion_ecosystem_mapping.items():
        if eco_type == 'all':
            deficit = avg_deficit
        else:
            deficit = deficit_by_type.get(eco_type, 0.0)

        criterion_deficits[criterion] = deficit

    # Normalise weights
    total_deficit = sum(criterion_deficits.values())

    if total_deficit == 0:
        logger.info("All ecosystem types at target, returning equal weights")
        n_criteria = len(criterion_ecosystem_mapping)
        equal_weight = 1.0 / n_criteria
        weights = {crit: equal_weight for crit in criterion_ecosystem_mapping.keys()}
    else:
        weights = {
            crit: deficit / total_deficit
            for crit, deficit in criterion_deficits.items()
        }

    logger.info(f"Proposed Group A weights (from representativity): {weights}")

    # Verify sum
    weight_sum = sum(weights.values())
    assert abs(weight_sum - 1.0) < 1e-6, f"Weights do not sum to 1.0: {weight_sum}"

    return weights


def clc_groups_for_level(clc_level: int) -> dict[str, set]:
    """
    Build {group_label: set(CLC codes)} from the full official CLC 2018
    nomenclature (``modules.utils.clc_loader.get_clc_legend``) at the
    requested hierarchy level.

    Parameters
    ----------
    clc_level : int
        1 -> 5 broad classes (e.g. "Forest and semi-natural areas")
        2 -> 15 classes (e.g. "Forests", "Pastures")
        3 -> 44 classes, the full CLC nomenclature (e.g. "311 - Broad-leaved forest")

    Returns
    -------
    dict[str, set[int]]
        Group label -> set of CLC integer codes belonging to it, in CLC
        code order. Unlike ``_CLC_ECOSYSTEM_GROUPS`` (used for the
        conservation-representativity index), this covers **all** land
        cover, including artificial surfaces — appropriate for a complete
        extent account rather than a conservation-target subset.
    """
    from modules.utils.clc_loader import get_clc_legend

    level_key = {1: 'level1', 2: 'level2', 3: 'label'}.get(clc_level)
    if level_key is None:
        raise ValueError(f"clc_level must be 1, 2 or 3 (got {clc_level})")

    legend = get_clc_legend()
    groups: dict[str, set] = {}
    for code in sorted(legend.keys()):
        meta = legend[code]
        label = f"{code} - {meta[level_key]}" if clc_level == 3 else meta[level_key]
        groups.setdefault(label, set()).add(code)
    return groups


def ecosystem_account_table(
    aligned_arrays: dict,
    profile: dict,
    clc_level: int = 1,
    ecosystem_groups: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a habitat-level ecosystem account: extent, condition and service
    capacity per broad ecosystem type — the "Habitats x Surface x Condition x
    Capacite en services" table (WP5 ecosystem-accounting data stream,
    complementing the socio-ecological mapping already produced elsewhere).

    Mirrors the SEEA Ecosystem Accounting logic (extent account + condition
    account + services account), computed directly from the same aligned
    criterion rasters used by the MCE engine, on the same pixel grid — so the
    table is exactly consistent with whatever favourability run it accompanies,
    with no separate re-sampling step.

    Parameters
    ----------
    aligned_arrays : dict
        Mapping criterion name -> np.ndarray, all on the same grid as
        ``profile`` (i.e. st.session_state['_aligned_arrays'] in the UI).
        Must include 'landuse', 'ecosystem_condition', 'regulating_es',
        'cultural_es' and 'provisioning_es'.
    profile : dict
        Rasterio profile matching the aligned arrays (transform, crs).
    clc_level : int, default 1
        Corine Land Cover hierarchy level to group habitats by — 1 (5
        classes), 2 (15 classes) or 3 (44 classes, full nomenclature).
        Ignored if ``ecosystem_groups`` is given explicitly.
    ecosystem_groups : dict, optional
        Mapping ecosystem-type label -> set of CLC integer codes, overriding
        ``clc_level`` entirely. Pass ``_CLC_ECOSYSTEM_GROUPS`` for the
        6-category, conservation-target-only taxonomy used elsewhere in
        Module 1 (excludes artificial surfaces).

    Returns
    -------
    pd.DataFrame
        One row per ecosystem type plus a "Whole study area" total row.
        Columns: ecosystem_type, area_ha, pct_area, condition_mean,
        regulating_es_mean, cultural_es_mean, provisioning_es_mean.
        Condition/service columns are NaN for a type with zero valid pixels.
    """
    groups = ecosystem_groups if ecosystem_groups is not None else clc_groups_for_level(clc_level)

    required = ['landuse', 'ecosystem_condition', 'regulating_es', 'cultural_es', 'provisioning_es']
    missing = [k for k in required if k not in aligned_arrays]
    if missing:
        raise ValueError(f"aligned_arrays is missing required layer(s): {missing}")

    landuse = aligned_arrays['landuse']
    transform = profile['transform']
    pixel_area_ha = abs(transform[0] * transform[4]) / 10_000.0

    valid_lu = ~np.isnan(landuse)
    total_valid_px = int(valid_lu.sum())

    def _service_means(mask: np.ndarray) -> dict:
        out = {}
        for key, col in [
            ('condition_mean', 'ecosystem_condition'),
            ('regulating_es_mean', 'regulating_es'),
            ('cultural_es_mean', 'cultural_es'),
            ('provisioning_es_mean', 'provisioning_es'),
        ]:
            arr = aligned_arrays[col][mask]
            arr = arr[~np.isnan(arr)]
            out[key] = float(np.mean(arr)) if arr.size > 0 else np.nan
        return out

    rows = []
    for label, codes in groups.items():
        mask = valid_lu & np.isin(landuse, list(codes))
        n_px = int(mask.sum())
        row = {
            'ecosystem_type': label,
            'area_ha': round(n_px * pixel_area_ha, 1),
            'pct_area': round(100.0 * n_px / total_valid_px, 1) if total_valid_px > 0 else 0.0,
        }
        row.update(_service_means(mask))
        rows.append(row)

    # Whole-study-area total row, for reference
    total_row = {
        'ecosystem_type': 'Whole study area',
        'area_ha': round(total_valid_px * pixel_area_ha, 1),
        'pct_area': 100.0,
    }
    total_row.update(_service_means(valid_lu))
    rows.append(total_row)

    df = pd.DataFrame(rows)
    logger.info(f"Ecosystem account built: {len(groups)} habitat types, {total_valid_px} valid pixels")
    return df

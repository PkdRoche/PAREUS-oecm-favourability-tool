"""Gap analysis and weight adjustment for Module 2."""

import logging
from typing import Dict, List
from pathlib import Path
import geopandas as gpd
import shapely.geometry.base
from shapely.ops import unary_union
import rasterio
from rasterio.features import rasterize

logger = logging.getLogger(__name__)


def strict_gaps(
    pa_gdf: gpd.GeoDataFrame,
    territory_geom: shapely.geometry.base.BaseGeometry
) -> gpd.GeoDataFrame:
    """
    Identify areas with no PA coverage of any class.

    These are strict conservation gaps where no protected area exists.

    Parameters
    ----------
    pa_gdf : gpd.GeoDataFrame
        Protected areas GeoDataFrame. Must be in EPSG:3035.
    territory_geom : shapely.geometry.base.BaseGeometry
        Territory boundary geometry. Must be in EPSG:3035.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame of gap polygons (territory - PA union).

    Raises
    ------
    ValueError
        If GeoDataFrame is not in EPSG:3035.

    Notes
    -----
    Strict gaps = territory minus (union of all PA polygons)

    These areas are priority candidates for new PA or OECM designation.

    Examples
    --------
    >>> gaps = strict_gaps(pa_gdf, territory_boundary)
    >>> gap_area = gaps.geometry.area.sum() / 10000  # hectares
    >>> print(f"Strict gap area: {gap_area:.2f} ha")
    """
    # Verify CRS
    if pa_gdf.crs != 'EPSG:3035':
        raise ValueError(f"PA GeoDataFrame must be in EPSG:3035. Current: {pa_gdf.crs}")

    logger.info("Computing strict gaps (no PA coverage)...")

    if len(pa_gdf) == 0:
        # No PAs = entire territory is a gap
        logger.warning("No protected areas found, entire territory is a strict gap")
        return gpd.GeoDataFrame([{'gap_type': 'strict'}], geometry=[territory_geom], crs='EPSG:3035')

    # Union all PA polygons
    pa_union = unary_union(pa_gdf.geometry)

    # Compute difference
    gap_geom = territory_geom.difference(pa_union)

    # Convert to GeoDataFrame
    if gap_geom.is_empty:
        logger.info("No strict gaps found (full territorial coverage)")
        return gpd.GeoDataFrame(columns=['gap_type', 'geometry'], crs='EPSG:3035')

    gaps_gdf = gpd.GeoDataFrame([{'gap_type': 'strict'}], geometry=[gap_geom], crs='EPSG:3035')

    gap_area_ha = gap_geom.area / 10000.0
    logger.info(f"Strict gaps identified: {gap_area_ha:.2f} ha")

    return gaps_gdf


def qualitative_gaps(
    pa_gdf: gpd.GeoDataFrame,
    territory_geom: shapely.geometry.base.BaseGeometry,
    weak_classes: List[str] = None
) -> gpd.GeoDataFrame:
    """
    Identify areas covered only by weak protection classes.

    Qualitative gaps are areas where protection exists but is insufficient
    (e.g., only contractual management, no regulatory protection).

    Parameters
    ----------
    pa_gdf : gpd.GeoDataFrame
        Protected areas GeoDataFrame with 'protection_class' column.
        Must be in EPSG:3035.
    territory_geom : shapely.geometry.base.BaseGeometry
        Territory boundary geometry. Must be in EPSG:3035.
    weak_classes : list of str, optional
        List of protection classes considered weak.
        Default: ['contractual', 'unassigned']

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame of qualitative gap polygons.

    Raises
    ------
    ValueError
        If GeoDataFrame is not in EPSG:3035 or missing protection_class column.

    Notes
    -----
    Qualitative gaps = areas covered ONLY by weak classes
                     = territory minus (union of strong PA classes)

    These areas may benefit from OECM designation to complement existing
    weak protection.

    Examples
    --------
    >>> qual_gaps = qualitative_gaps(
    ...     pa_gdf,
    ...     territory_boundary,
    ...     weak_classes=['contractual', 'unassigned']
    ... )
    >>> gap_area = qual_gaps.geometry.area.sum() / 10000
    """
    # Verify CRS
    if pa_gdf.crs != 'EPSG:3035':
        raise ValueError(f"PA GeoDataFrame must be in EPSG:3035. Current: {pa_gdf.crs}")

    # Verify protection_class column
    if 'protection_class' not in pa_gdf.columns:
        raise ValueError("PA GeoDataFrame must have 'protection_class' column")

    # Default weak classes
    if weak_classes is None:
        weak_classes = ['contractual', 'unassigned']

    logger.info(f"Computing qualitative gaps (weak protection only: {weak_classes})...")

    # Filter to strong protection classes (exclude weak)
    strong_gdf = pa_gdf[~pa_gdf['protection_class'].isin(weak_classes)]

    if len(strong_gdf) == 0:
        # No strong protection = entire territory is qualitative gap
        logger.warning("No strong protection found, entire territory is a qualitative gap")
        return gpd.GeoDataFrame(
            [{'gap_type': 'qualitative'}],
            geometry=[territory_geom],
            crs='EPSG:3035'
        )

    # Union strong PA polygons
    strong_union = unary_union(strong_gdf.geometry)

    # Compute difference
    gap_geom = territory_geom.difference(strong_union)

    # Convert to GeoDataFrame
    if gap_geom.is_empty:
        logger.info("No qualitative gaps found (full strong protection coverage)")
        return gpd.GeoDataFrame(columns=['gap_type', 'geometry'], crs='EPSG:3035')

    gaps_gdf = gpd.GeoDataFrame(
        [{'gap_type': 'qualitative'}],
        geometry=[gap_geom],
        crs='EPSG:3035'
    )

    gap_area_ha = gap_geom.area / 10000.0
    logger.info(f"Qualitative gaps identified: {gap_area_ha:.2f} ha")

    return gaps_gdf


def potential_corridors(
    pa_gdf: gpd.GeoDataFrame,
    territory_geom: shapely.geometry.base.BaseGeometry,
    max_gap_m: float = 5000.0
) -> gpd.GeoDataFrame:
    """
    Identify potential ecological corridors between PA patches.

    Simplified corridor identification: unprotected areas within max_gap_m
    of two or more PA patches. Uses buffer intersection method.

    Parameters
    ----------
    pa_gdf : gpd.GeoDataFrame
        Protected areas GeoDataFrame. Must be in EPSG:3035.
    territory_geom : shapely.geometry.base.BaseGeometry
        Territory boundary geometry. Must be in EPSG:3035.
    max_gap_m : float, default=5000.0
        Maximum distance (meters) between PA patches to consider as corridor.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame of potential corridor polygons.

    Raises
    ------
    ValueError
        If GeoDataFrame is not in EPSG:3035.

    Notes
    -----
    Corridor identification algorithm:
    1. Buffer each PA patch by max_gap_m / 2
    2. Find areas where buffers from different patches overlap
    3. Exclude existing PA areas
    4. Return unprotected overlap areas as potential corridors

    This is a simplified approach. More sophisticated methods (cost-distance,
    Circuitscape) are recommended for dedicated connectivity analysis.

    Examples
    --------
    >>> corridors = potential_corridors(
    ...     pa_gdf,
    ...     territory_boundary,
    ...     max_gap_m=5000.0
    ... )
    >>> corridor_area = corridors.geometry.area.sum() / 10000
    >>> print(f"Potential corridor area: {corridor_area:.2f} ha")
    """
    # Verify CRS
    if pa_gdf.crs != 'EPSG:3035':
        raise ValueError(f"PA GeoDataFrame must be in EPSG:3035. Current: {pa_gdf.crs}")

    logger.info(f"Identifying potential corridors (max gap: {max_gap_m} m)...")

    if len(pa_gdf) < 2:
        logger.warning("Need at least 2 PA patches for corridor analysis")
        return gpd.GeoDataFrame(columns=['corridor_type', 'geometry'], crs='EPSG:3035')

    # Buffer each PA patch
    buffer_distance = max_gap_m / 2.0
    pa_buffers = pa_gdf.copy()
    pa_buffers['geometry'] = pa_buffers.geometry.buffer(buffer_distance)
    pa_buffers = pa_buffers.reset_index(drop=True)

    # Find overlapping buffer pairs using spatial index (R-tree).
    # Replaces O(n²) brute-force loop with O(n·k) where k = avg neighbours.
    corridor_candidates = []
    sindex = pa_buffers.sindex
    geoms = pa_buffers.geometry

    for i, geom_i in enumerate(geoms):
        # R-tree candidate query — returns indices whose bounding boxes overlap
        candidate_idxs = sindex.query(geom_i, predicate='intersects')
        for j in candidate_idxs:
            if j <= i:          # Skip self and already-processed pairs
                continue
            intersection = geom_i.intersection(geoms.iloc[j])
            if not intersection.is_empty:
                corridor_candidates.append(intersection)

    if not corridor_candidates:
        logger.info("No potential corridors found")
        return gpd.GeoDataFrame(columns=['corridor_type', 'geometry'], crs='EPSG:3035')

    # Union all candidate corridors
    corridor_union = unary_union(corridor_candidates)

    # Exclude existing PA areas
    pa_union = unary_union(pa_gdf.geometry)
    corridor_gaps = corridor_union.difference(pa_union)

    # Clip to territory boundary
    corridor_final = corridor_gaps.intersection(territory_geom)

    if corridor_final.is_empty:
        logger.info("No corridor gaps found (all potential corridors already protected)")
        return gpd.GeoDataFrame(columns=['corridor_type', 'geometry'], crs='EPSG:3035')

    # Convert to GeoDataFrame
    corridors_gdf = gpd.GeoDataFrame(
        [{'corridor_type': 'potential'}],
        geometry=[corridor_final],
        crs='EPSG:3035'
    )

    corridor_area_ha = corridor_final.area / 10000.0
    logger.info(f"Potential corridors identified: {corridor_area_ha:.2f} ha")

    return corridors_gdf


def export_gap_masks_as_raster(
    gap_layers: Dict[str, gpd.GeoDataFrame],
    reference_profile: dict,
    output_dir: str
) -> Dict[str, str]:
    """
    Rasterise gap vector layers to match Module 2 raster grid.

    Converts gap analysis vector outputs to raster masks compatible with
    Module 2 MCE workflow.

    Parameters
    ----------
    gap_layers : dict
        Dictionary mapping gap type to GeoDataFrame.
        Example: {
            'strict_gaps': strict_gaps_gdf,
            'qualitative_gaps': qual_gaps_gdf,
            'corridors': corridors_gdf
        }
    reference_profile : dict
        Rasterio profile (metadata) from a Module 2 reference raster.
        Must include: crs, transform, width, height, dtype, nodata.
        Example: reference_raster.profile
    output_dir : str
        Directory path for output GeoTIFF files.

    Returns
    -------
    dict
        Dictionary mapping gap type to output GeoTIFF path.
        Example: {
            'strict_gaps': '/path/to/strict_gaps.tif',
            'qualitative_gaps': '/path/to/qualitative_gaps.tif',
            'corridors': '/path/to/corridors.tif'
        }

    Raises
    ------
    ValueError
        If reference_profile is missing required keys.
    OSError
        If output_dir cannot be created.

    Notes
    -----
    - Gap areas are encoded as 1.0, non-gap as 0.0
    - Output rasters match the CRS, resolution, and extent of reference_profile
    - Output format: GeoTIFF, float32, LZW compression

    Examples
    --------
    >>> import rasterio
    >>> with rasterio.open('data/reference_raster.tif') as ref:
    ...     profile = ref.profile
    >>> gap_layers = {
    ...     'strict_gaps': strict_gaps_gdf,
    ...     'qualitative_gaps': qual_gaps_gdf
    ... }
    >>> output_paths = export_gap_masks_as_raster(
    ...     gap_layers,
    ...     profile,
    ...     'outputs/gap_masks'
    ... )
    """
    # Verify required profile keys
    required_keys = ['crs', 'transform', 'width', 'height']
    missing = [k for k in required_keys if k not in reference_profile]
    if missing:
        raise ValueError(f"Reference profile missing required keys: {missing}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Rasterising {len(gap_layers)} gap layers to {output_dir}...")

    output_paths = {}

    for gap_type, gdf in gap_layers.items():
        if len(gdf) == 0 or gdf.geometry.is_empty.all():
            logger.warning(f"Skipping empty gap layer: {gap_type}")
            continue

        # Reproject to match reference CRS if needed
        target_crs = reference_profile['crs']
        if gdf.crs != target_crs:
            logger.info(f"Reprojecting {gap_type} from {gdf.crs} to {target_crs}")
            gdf = gdf.to_crs(target_crs)

        # Create output file path
        output_file = output_path / f"{gap_type}.tif"

        # Prepare shapes for rasterization
        shapes = [(geom, 1.0) for geom in gdf.geometry]

        # Create raster array
        raster_array = rasterize(
            shapes=shapes,
            out_shape=(reference_profile['height'], reference_profile['width']),
            transform=reference_profile['transform'],
            fill=0.0,
            dtype='float32'
        )

        # Write to GeoTIFF
        output_profile = reference_profile.copy()
        output_profile.update({
            'dtype': 'float32',
            'count': 1,
            'nodata': -9999.0,
            'compress': 'lzw'
        })

        with rasterio.open(output_file, 'w', **output_profile) as dst:
            dst.write(raster_array, 1)

        output_paths[gap_type] = str(output_file)
        logger.info(f"Exported {gap_type} to {output_file}")

    logger.info(f"Gap mask rasterization complete. {len(output_paths)} files created.")
    return output_paths

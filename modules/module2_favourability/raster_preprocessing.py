"""Raster data preprocessing and harmonisation.

This module provides functions for loading, reprojecting, resampling, aligning,
and normalising raster data for the OECM Favourability Tool.
"""

import logging
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import yaml
from pathlib import Path
from typing import Optional
from shapely.geometry import mapping
import math

logger = logging.getLogger(__name__)


def _load_config():
    """Load configuration from settings.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_transformation_config():
    """Load transformation function configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "transformation_functions.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raster(path: str) -> tuple[np.ndarray, dict]:
    """Load a GeoTIFF raster file.

    Parameters
    ----------
    path : str
        Path to the GeoTIFF file.

    Returns
    -------
    tuple[np.ndarray, dict]
        A tuple containing:
        - array : np.ndarray
            2D numpy array of raster values (first band).
        - profile : dict
            Rasterio profile dictionary containing metadata (CRS, transform, etc.).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the raster is empty or cannot be read.
    """
    logger.info(f"Loading raster from {path}")

    if not Path(path).exists():
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as src:
        array = src.read(1)
        profile = dict(src.profile)

    if array.size == 0:
        raise ValueError(f"Empty raster array loaded from {path}")

    # If CRS is missing, assume EPSG:3035 (tool requirement) and warn
    if profile.get('crs') is None:
        logger.warning(
            f"Raster '{Path(path).name}' has no embedded CRS. "
            "Assuming EPSG:3035 — please verify this is correct."
        )
        profile['crs'] = 'EPSG:3035'

    logger.info(f"Loaded raster with shape {array.shape}, dtype {array.dtype}, CRS {profile['crs']}")
    return array, profile


def load_raster_windowed(
    path: str,
    clip_geom,
    geom_crs: str = 'EPSG:3035',
    buffer_pixels: int = 10,
) -> tuple[np.ndarray, dict]:
    """Load only the portion of a GeoTIFF that overlaps a geometry's bounding box.

    Instead of reading the entire file and then discarding pixels outside the
    study area, this function:
      1. Reprojects the geometry bounds into the raster's own CRS.
      2. Computes the corresponding pixel window.
      3. Reads only that window from disk.
      4. Returns an (array, profile) with an adjusted transform.

    For EU-wide rasters (e.g. CLC at 100 m, ~37 M pixels) clipped to a
    country-sized NUTS territory (~300 k pixels), this reduces I/O and memory
    by ~100×.

    A ``buffer_pixels`` margin is added around the bounding box to ensure full
    coverage after the subsequent reprojection step in ``align_rasters``.

    Parameters
    ----------
    path : str
        Path to the GeoTIFF file.
    clip_geom : shapely geometry
        Study area geometry used to derive the read window.
    geom_crs : str
        CRS of ``clip_geom``. Default ``'EPSG:3035'``.
    buffer_pixels : int
        Extra pixel margin added around the bounding box. Default 10.

    Returns
    -------
    tuple[np.ndarray, dict]
        (array, profile) covering at least the bounding box of ``clip_geom``.
        Falls back to a full ``load_raster`` if windowed read fails.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Raster file not found: {path}")

    try:
        from pyproj import Transformer
        from rasterio.windows import from_bounds as _win_from_bounds

        with rasterio.open(path) as src:
            raster_crs = src.crs
            src_transform = src.transform

            # ── Reproject geometry bounds into raster CRS ─────────────────
            if str(raster_crs) != geom_crs:
                _t = Transformer.from_crs(geom_crs, raster_crs, always_xy=True)
                minx, miny, maxx, maxy = clip_geom.bounds
                xs = [minx, maxx, minx, maxx]
                ys = [miny, miny, maxy, maxy]
                txs, tys = _t.transform(xs, ys)
                win_bounds = (min(txs), min(tys), max(txs), max(tys))
            else:
                minx, miny, maxx, maxy = clip_geom.bounds
                win_bounds = (minx, miny, maxx, maxy)

            # ── Compute pixel window + buffer ────────────────────────────
            win = _win_from_bounds(*win_bounds, transform=src_transform)
            win = win.round_offsets().round_lengths()
            # Add buffer and clamp to raster extent
            col_off = max(0, int(win.col_off) - buffer_pixels)
            row_off = max(0, int(win.row_off) - buffer_pixels)
            col_end = min(src.width,  int(win.col_off + win.width)  + buffer_pixels)
            row_end = min(src.height, int(win.row_off + win.height) + buffer_pixels)
            buffered_win = rasterio.windows.Window(
                col_off, row_off,
                col_end - col_off,
                row_end - row_off,
            )

            array   = src.read(1, window=buffered_win)
            profile = dict(src.profile)
            profile['width']     = buffered_win.width
            profile['height']    = buffered_win.height
            profile['transform'] = src.window_transform(buffered_win)

        if array.size == 0:
            raise ValueError("Windowed read produced empty array — falling back to full read")

        if profile.get('crs') is None:
            profile['crs'] = geom_crs

        orig_h = rasterio.open(path).height
        orig_w = rasterio.open(path).width
        reduction = (orig_h * orig_w) / max(array.size, 1)
        logger.info(
            "Windowed read '%s': %dx%d pixels (%.0f× smaller than full raster)",
            Path(path).name, array.shape[1], array.shape[0], reduction,
        )
        return array, profile

    except Exception as exc:
        logger.warning(
            "Windowed read failed for '%s' (%s) — falling back to full load",
            Path(path).name, exc,
        )
        return load_raster(path)


def reproject_raster(
    array: np.ndarray,
    src_profile: dict,
    target_crs: str,
    categorical: bool = False
) -> tuple[np.ndarray, dict]:
    """Reproject raster to target CRS using rasterio.warp.

    Parameters
    ----------
    array : np.ndarray
        Source raster array.
    src_profile : dict
        Source raster profile containing CRS and transform.
    target_crs : str
        Target coordinate reference system (e.g., 'EPSG:3035').

    Returns
    -------
    tuple[np.ndarray, dict]
        A tuple containing:
        - array : np.ndarray
            Reprojected raster array.
        - profile : dict
            Updated profile with new CRS and transform.

    Raises
    ------
    ValueError
        If target_crs is invalid or CRS information is missing.
    """
    logger.info(f"Reprojecting from {src_profile['crs']} to {target_crs}")

    if src_profile.get('crs') is None:
        raise ValueError("Source profile missing CRS information")

    if not target_crs:
        raise ValueError("Target CRS must be specified")

    # Check if already in target CRS
    if str(src_profile['crs']).upper() == target_crs.upper():
        logger.info("Raster already in target CRS, skipping reprojection")
        return array, src_profile.copy()

    # Calculate transform and dimensions for target CRS
    try:
        transform, width, height = calculate_default_transform(
            src_profile['crs'],
            target_crs,
            src_profile['width'],
            src_profile['height'],
            *rasterio.transform.array_bounds(
                src_profile['height'],
                src_profile['width'],
                src_profile['transform']
            )
        )
    except Exception as e:
        raise ValueError(f"Invalid target CRS '{target_crs}': {e}")

    # Create destination array
    dst_array = np.empty((height, width), dtype=array.dtype)

    # Reproject — use nearest neighbour for categorical layers (e.g. CLC land use)
    _resampling = Resampling.nearest if categorical else Resampling.bilinear
    reproject(
        source=array,
        destination=dst_array,
        src_transform=src_profile['transform'],
        src_crs=src_profile['crs'],
        dst_transform=transform,
        dst_crs=target_crs,
        resampling=_resampling
    )

    # Update profile
    dst_profile = src_profile.copy()
    dst_profile.update({
        'crs': target_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    logger.info(f"Reprojected to shape {dst_array.shape}")
    return dst_array, dst_profile


def resample_raster(
    array: np.ndarray,
    profile: dict,
    target_resolution: float,
    method: str = "bilinear"
) -> tuple[np.ndarray, dict]:
    """Resample raster to target resolution.

    Parameters
    ----------
    array : np.ndarray
        Input raster array.
    profile : dict
        Raster profile containing transform and bounds.
    target_resolution : float
        Target resolution in units of the CRS (typically meters).
    method : str, optional
        Resampling method: 'bilinear', 'nearest', or 'cubic'.
        Default is 'bilinear'.

    Returns
    -------
    tuple[np.ndarray, dict]
        A tuple containing:
        - array : np.ndarray
            Resampled raster array.
        - profile : dict
            Updated profile with new transform and dimensions.

    Raises
    ------
    ValueError
        If resampling method is invalid.
    """
    logger.info(f"Resampling to {target_resolution}m resolution using {method}")

    # Map method string to Resampling enum
    resampling_methods = {
        'bilinear': Resampling.bilinear,
        'nearest': Resampling.nearest,
        'cubic': Resampling.cubic
    }

    if method not in resampling_methods:
        raise ValueError(
            f"Invalid resampling method '{method}'. "
            f"Must be one of {list(resampling_methods.keys())}"
        )

    resampling_enum = resampling_methods[method]

    # Calculate current resolution
    transform = profile['transform']
    current_res_x = abs(transform[0])
    current_res_y = abs(transform[4])

    # Check if already at target resolution
    if np.isclose(current_res_x, target_resolution) and np.isclose(current_res_y, target_resolution):
        logger.info("Raster already at target resolution, skipping resampling")
        return array, profile.copy()

    # Calculate new dimensions
    scale_x = current_res_x / target_resolution
    scale_y = current_res_y / target_resolution

    new_width = int(np.round(profile['width'] * scale_x))
    new_height = int(np.round(profile['height'] * scale_y))

    # Get bounds
    bounds = rasterio.transform.array_bounds(
        profile['height'],
        profile['width'],
        transform
    )

    # Create new transform
    new_transform = from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3],
        new_width, new_height
    )

    # Create destination array
    dst_array = np.empty((new_height, new_width), dtype=array.dtype)

    # Resample
    reproject(
        source=array,
        destination=dst_array,
        src_transform=transform,
        src_crs=profile['crs'],
        dst_transform=new_transform,
        dst_crs=profile['crs'],
        resampling=resampling_enum
    )

    # Update profile
    dst_profile = profile.copy()
    dst_profile.update({
        'transform': new_transform,
        'width': new_width,
        'height': new_height
    })

    logger.info(f"Resampled from {array.shape} to {dst_array.shape}")
    return dst_array, dst_profile


def derive_grid_from_geometry(geom, resolution=100.0, crs="EPSG:3035") -> dict:
    """Derive a rasterio profile (transform, width, height, crs) from a shapely geometry.

    Parameters
    ----------
    geom : shapely.geometry
        Shapely geometry in the target CRS (typically NUTS2 boundary).
    resolution : float, optional
        Target grid resolution in units of the CRS (typically meters).
        Default is 100.0.
    crs : str, optional
        Target coordinate reference system. Default is 'EPSG:3035'.

    Returns
    -------
    dict
        Rasterio profile dictionary containing:
        - 'crs': Target CRS
        - 'transform': Affine transform for the grid
        - 'width': Grid width in pixels
        - 'height': Grid height in pixels
        - 'dtype': Data type (float64)
        - 'count': Number of bands (1)

    Notes
    -----
    Grid bounds are snapped to clean multiples of the resolution to ensure
    alignment with standard grid systems.
    """
    logger.info(f"Deriving grid from geometry: resolution={resolution}m, CRS={crs}")

    bounds = geom.bounds  # (minx, miny, maxx, maxy)

    # Snap bounds to clean grid multiples
    minx = math.floor(bounds[0] / resolution) * resolution
    miny = math.floor(bounds[1] / resolution) * resolution
    maxx = math.ceil(bounds[2] / resolution) * resolution
    maxy = math.ceil(bounds[3] / resolution) * resolution

    # Calculate dimensions
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

    # Create transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    profile = {
        'crs': crs,
        'transform': transform,
        'width': width,
        'height': height,
        'dtype': 'float64',
        'count': 1
    }

    logger.info(f"Derived grid: {width}x{height} pixels, bounds=({minx}, {miny}, {maxx}, {maxy})")
    return profile


def align_rasters(
    raster_dict: dict[str, tuple[np.ndarray, dict]],
    study_area_geom=None,
    resolution: float = 100.0,
    crs: str = "EPSG:3035"
) -> dict[str, tuple[np.ndarray, dict]]:
    """Align all rasters to a common grid (extent, resolution, CRS).

    When study_area_geom is provided, the reference grid is derived from the
    NUTS2 study area geometry bounds. All layers are clipped to the study area
    before alignment. This ensures all outputs are masked to the analysis extent.

    When study_area_geom is None (backward compatibility), the first layer after
    sorting by name is used as the reference grid.

    Parameters
    ----------
    raster_dict : dict[str, tuple[np.ndarray, dict]]
        Dictionary mapping layer names to (array, profile) tuples.
    study_area_geom : shapely.geometry, optional
        Study area geometry (typically NUTS2 boundary) in target CRS.
        If provided, this defines the reference grid extent. Default is None.
    resolution : float, optional
        Target resolution in units of the CRS (typically meters).
        Only used when study_area_geom is provided. Default is 100.0.
    crs : str, optional
        Target coordinate reference system. Only used when study_area_geom
        is provided. Default is 'EPSG:3035'.

    Returns
    -------
    dict[str, tuple[np.ndarray, dict]]
        Dictionary with aligned rasters, all sharing the same grid.

    Raises
    ------
    ValueError
        If raster_dict is empty or contains incompatible grids.

    Warnings
    --------
    Logs a warning if a layer covers less than 80% of the study area grid cells.

    Notes
    -----
    - When study_area_geom is provided:
      - Uses nearest-neighbor resampling for integer/categorical layers
      - Uses bilinear resampling for float layers
      - Fills NoData with np.nan for float output (not 0)
      - Clips each layer to study area before alignment
    - When study_area_geom is None:
      - Legacy behavior: uses first sorted layer as reference
      - Logs deprecation warning
    """
    logger.info(f"Aligning {len(raster_dict)} rasters to common grid")

    if not raster_dict:
        raise ValueError("Cannot align empty raster dictionary")

    # ===================================================================
    # Case 1: study_area_geom provided — derive grid from geometry
    # ===================================================================
    if study_area_geom is not None:
        logger.info("Using study area geometry to derive reference grid")

        # Derive reference grid from geometry
        ref_profile = derive_grid_from_geometry(geom=study_area_geom, resolution=resolution, crs=crs)

        # Build geometry mask ONCE for all layers — True = outside geometry (set to NaN/0)
        # This replaces the per-layer MemoryFile roundtrip.
        from rasterio.features import geometry_mask as _geometry_mask
        try:
            outside_mask = _geometry_mask(
                [mapping(study_area_geom)],
                out_shape=(ref_profile['height'], ref_profile['width']),
                transform=ref_profile['transform'],
                all_touched=False,
                invert=False   # True outside the geometry
            )
        except Exception as e:
            logger.warning(f"Could not build geometry mask: {e}. Layers will not be clipped.")
            outside_mask = None

        aligned = {}

        for name, (src_array, src_profile) in raster_dict.items():
            logger.info(f"Clipping and aligning '{name}' to study area grid")

            # 'landuse' is always categorical regardless of stored dtype.
            is_categorical = (name == 'landuse') or np.issubdtype(src_array.dtype, np.integer)
            resampling_method = Resampling.nearest if is_categorical else Resampling.bilinear

            logger.info(f"  dtype={src_array.dtype}, resampling={resampling_method.name}")

            # Normalise the source array's NoData representation to a single
            # value and pass it explicitly as src_nodata. `source` here is a
            # plain numpy array (not an open rasterio dataset), so reproject()
            # has no file handle to read nodata metadata from — without this,
            # it treats every source pixel as valid data, and bilinear
            # resampling can blend a NoData sentinel (or an already-NaN pixel
            # from upstream validation) into neighbouring valid pixels,
            # producing spurious values — including negative scores for
            # continuous [0,1] criteria — near any NoData edge.
            if is_categorical:
                src_nodata = src_profile.get('nodata')
                if src_nodata is None:
                    src_nodata = 0
            else:
                src_array = src_array.astype(np.float32, copy=True)
                _declared_nodata = src_profile.get('nodata')
                if _declared_nodata is not None and not (
                    isinstance(_declared_nodata, float) and np.isnan(_declared_nodata)
                ):
                    src_array[src_array == _declared_nodata] = np.nan
                src_nodata = np.nan

            # Use float32 (not float64) for continuous layers — halves memory, sufficient precision.
            dst_dtype = src_array.dtype if is_categorical else np.float32
            dst_nodata = np.nan if not is_categorical else 0
            dst_array = np.full(
                (ref_profile['height'], ref_profile['width']),
                fill_value=dst_nodata,
                dtype=dst_dtype
            )

            # Reproject and resample to reference grid
            reproject(
                source=src_array,
                destination=dst_array,
                src_transform=src_profile['transform'],
                src_crs=src_profile['crs'],
                dst_transform=ref_profile['transform'],
                dst_crs=ref_profile['crs'],
                resampling=resampling_method,
                src_nodata=src_nodata,
                dst_nodata=dst_nodata
            )

            # Defensive clip for the four [0,1]-bounded criteria: bilinear
            # resampling of correctly-declared NoData should no longer
            # overshoot the source range, but this guards against any
            # residual floating-point overshoot at sharp edges without
            # masking genuinely out-of-range upstream data on unbounded
            # layers (anthropogenic_pressure) or the categorical landuse codes.
            if name in ('ecosystem_condition', 'regulating_es', 'cultural_es', 'provisioning_es'):
                _valid = ~np.isnan(dst_array)
                dst_array[_valid] = np.clip(dst_array[_valid], 0.0, 1.0)

            # Apply geometry mask directly via numpy — no MemoryFile needed
            if outside_mask is not None:
                if is_categorical:
                    dst_array[outside_mask] = 0
                else:
                    dst_array[outside_mask] = np.nan

            # Check coverage (warn if < 80%)
            valid_count = np.sum(~np.isnan(dst_array)) if not is_categorical else np.sum(dst_array != 0)
            total_count = dst_array.size
            coverage = valid_count / total_count if total_count > 0 else 0

            if coverage < 0.80:
                logger.warning(
                    f"Layer '{name}' covers only {coverage*100:.1f}% of the study area grid. "
                    f"Expected coverage ≥80%. Check layer extent."
                )

            # Create aligned profile
            dst_profile = ref_profile.copy()
            dst_profile['dtype'] = str(dst_dtype)

            aligned[name] = (dst_array, dst_profile)
            logger.info(f"Aligned '{name}' to shape {dst_array.shape}, coverage={coverage*100:.1f}%")

        logger.info("All rasters aligned to study area grid")
        return aligned

    # ===================================================================
    # Case 2: No study_area_geom — legacy behavior (alphabetical reference)
    # ===================================================================
    else:
        logger.warning(
            "align_rasters() called without study_area_geom. "
            "Using legacy behavior (first sorted layer as reference). "
            "This is deprecated — please provide study_area_geom for correct grid alignment."
        )

        # Sort by name and use first as reference
        sorted_names = sorted(raster_dict.keys())
        reference_name = sorted_names[0]
        ref_array, ref_profile = raster_dict[reference_name]

        logger.info(f"Using '{reference_name}' as reference grid")
        logger.info(f"Reference: shape={ref_array.shape}, CRS={ref_profile['crs']}, "
                    f"transform={ref_profile['transform']}")

        aligned = {}
        aligned[reference_name] = (ref_array.copy(), ref_profile.copy())

        # Align all other rasters to reference
        for name in sorted_names[1:]:
            src_array, src_profile = raster_dict[name]
            logger.info(f"Aligning '{name}' to reference grid")

            # Create destination array matching reference
            dst_array = np.empty(
                (ref_profile['height'], ref_profile['width']),
                dtype=src_array.dtype
            )

            # Reproject to match reference
            reproject(
                source=src_array,
                destination=dst_array,
                src_transform=src_profile['transform'],
                src_crs=src_profile['crs'],
                dst_transform=ref_profile['transform'],
                dst_crs=ref_profile['crs'],
                resampling=Resampling.bilinear
            )

            # Create aligned profile
            dst_profile = src_profile.copy()
            dst_profile.update({
                'crs': ref_profile['crs'],
                'transform': ref_profile['transform'],
                'width': ref_profile['width'],
                'height': ref_profile['height']
            })

            aligned[name] = (dst_array, dst_profile)
            logger.info(f"Aligned '{name}' to shape {dst_array.shape}")

        logger.info("All rasters aligned to common grid")
        return aligned


def apply_nodata_mask(
    array: np.ndarray,
    nodata_value: Optional[float]
) -> np.ndarray:
    """Replace nodata values with np.nan.

    Parameters
    ----------
    array : np.ndarray
        Input raster array.
    nodata_value : float or None
        Value representing missing data. If None, array is returned unchanged.

    Returns
    -------
    np.ndarray
        Array with nodata values replaced by np.nan. Returned as float dtype.

    Notes
    -----
    Handles None nodata_value gracefully by returning a copy of the array
    converted to float dtype.
    """
    # Ensure float dtype for np.nan
    array_float = array.astype(np.float64, copy=True)

    if nodata_value is None:
        logger.debug("No nodata value specified, returning array as-is")
        return array_float

    # Replace nodata with nan
    mask = array_float == nodata_value
    nodata_count = np.sum(mask)

    if nodata_count > 0:
        array_float[mask] = np.nan
        logger.info(f"Masked {nodata_count} nodata values to np.nan")

    return array_float


def normalize_linear(
    array: np.ndarray,
    vmin: float,
    vmax: float,
    invert: bool = False
) -> np.ndarray:
    """Linear normalisation to [0, 1].

    Parameters
    ----------
    array : np.ndarray
        Input array to normalize.
    vmin : float
        Minimum value for normalization range.
    vmax : float
        Maximum value for normalization range.
    invert : bool, optional
        If True, invert the normalized values (1 - normalized).
        Used for pressure layers. Default is False.

    Returns
    -------
    np.ndarray
        Normalized array with values in [0, 1]. NaN values are preserved.

    Notes
    -----
    Values below vmin are clipped to 0, values above vmax are clipped to 1.
    """
    logger.info(f"Applying linear normalization (vmin={vmin}, vmax={vmax}, invert={invert})")

    if vmax <= vmin:
        raise ValueError(f"vmax ({vmax}) must be greater than vmin ({vmin})")

    # Normalize to [0, 1]
    normalized = (array - vmin) / (vmax - vmin)

    # Clip to [0, 1]
    normalized = np.clip(normalized, 0, 1)

    # Invert if requested
    if invert:
        normalized = 1.0 - normalized

    logger.info(f"Linear normalization complete, range: [{np.nanmin(normalized):.3f}, {np.nanmax(normalized):.3f}]")
    return normalized


def normalize_sigmoid(
    array: np.ndarray,
    inflection: float,
    slope: float
) -> np.ndarray:
    """Sigmoid normalization to [0, 1].

    Parameters
    ----------
    array : np.ndarray
        Input array to normalize.
    inflection : float
        Inflection point of sigmoid curve (input value at which output = 0.5).
    slope : float
        Steepness of the sigmoid curve. Higher values create sharper transitions.

    Returns
    -------
    np.ndarray
        Normalized array with values in [0, 1]. NaN values are preserved.

    Notes
    -----
    Uses the logistic function: f(x) = 1 / (1 + exp(-slope * (x - inflection)))
    """
    logger.info(f"Applying sigmoid normalization (inflection={inflection}, slope={slope})")

    # Sigmoid transformation: 1 / (1 + exp(-slope * (x - inflection)))
    normalized = 1.0 / (1.0 + np.exp(-slope * (array - inflection)))

    logger.info(f"Sigmoid normalization complete, range: [{np.nanmin(normalized):.3f}, {np.nanmax(normalized):.3f}]")
    return normalized


def normalize_gaussian(
    array: np.ndarray,
    mean: float,
    std: float
) -> np.ndarray:
    """Gaussian normalization - non-monotone, optimum at mean.

    This function is used EXCLUSIVELY for provisioning ES capacity (Group C),
    where an optimal intermediate value is desired (e.g., moderate use is best).

    Parameters
    ----------
    array : np.ndarray
        Input array to normalize.
    mean : float
        Optimal value (peak of Gaussian curve).
    std : float
        Standard deviation controlling spread around optimum.

    Returns
    -------
    np.ndarray
        Normalized array with values in [0, 1], maximum at mean. NaN values are preserved.

    Notes
    -----
    Uses the Gaussian (normal distribution) formula:
    f(x) = exp(-((x - mean)^2) / (2 * std^2))

    WARNING: This is a non-monotone transformation. Do NOT apply to any layer
    other than provisioning ES without explicit instruction.
    """
    logger.info(f"Applying Gaussian normalization (mean={mean}, std={std})")
    logger.warning("Gaussian normalization is non-monotone - use only for provisioning ES")

    if std <= 0:
        raise ValueError(f"Standard deviation must be positive, got {std}")

    # Gaussian transformation: exp(-((x - mean)^2) / (2 * std^2))
    normalized = np.exp(-((array - mean) ** 2) / (2 * std ** 2))

    logger.info(f"Gaussian normalization complete, range: [{np.nanmin(normalized):.3f}, {np.nanmax(normalized):.3f}]")
    return normalized


def percentile_clip(
    array: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> tuple[np.ndarray, float, float]:
    """Clip array to [low_pct, high_pct] percentiles of valid values.

    Returns the clipped array together with the percentile bounds used,
    so callers can log or reuse them as vmin/vmax.

    Parameters
    ----------
    array : np.ndarray
        Input raster array (may contain NaN).
    low_pct : float
        Lower percentile cutoff (default 2).
    high_pct : float
        Upper percentile cutoff (default 98).

    Returns
    -------
    clipped : np.ndarray
        Array with values outside [p_low, p_high] clipped to those bounds.
    p_low : float
        Value at low_pct percentile.
    p_high : float
        Value at high_pct percentile.
    """
    valid = array[~np.isnan(array)]
    if len(valid) == 0:
        return array.copy(), 0.0, 1.0
    p_low  = float(np.percentile(valid, low_pct))
    p_high = float(np.percentile(valid, high_pct))
    if p_high <= p_low:
        logger.warning(
            f"percentile_clip: p{high_pct}={p_high} <= p{low_pct}={p_low}; "
            "skipping clip."
        )
        return array.copy(), p_low, p_high
    clipped = np.where(np.isnan(array), array, np.clip(array, p_low, p_high))
    logger.info(
        f"Percentile clip [{low_pct}–{high_pct}%]: "
        f"vmin={p_low:.4g}, vmax={p_high:.4g} "
        f"(was [{float(np.nanmin(array)):.4g}, {float(np.nanmax(array)):.4g}])"
    )
    return clipped, p_low, p_high


def normalize_layer(
    array: np.ndarray,
    layer_name: str,
    params: dict,
    percentile_norm: bool = False,
) -> np.ndarray:
    """Dispatcher: apply normalization based on configuration.

    Reads the transformation type from the params dictionary and applies
    the appropriate normalization function.

    Parameters
    ----------
    array : np.ndarray
        Input array to normalize.
    layer_name : str
        Name of the layer being normalized (for logging).
    params : dict
        Transformation parameters from configuration, must include 'type' key.
        Additional keys depend on the transformation type:
        - 'linear': vmin, vmax, invert (optional)
        - 'inverted_linear': vmin, vmax
        - 'sigmoid': inflection, slope
        - 'gaussian': mean, std
    percentile_norm : bool
        When True, clip the array to its 2nd–98th percentile range before
        normalization.  This makes the result robust to outliers.  The
        clipped bounds replace config-derived vmin/vmax for linear /
        inverted_linear transforms.  Sigmoid and Gaussian transforms
        receive the clipped array directly.

    Returns
    -------
    np.ndarray
        Normalized array with values in [0, 1].

    Raises
    ------
    ValueError
        If transformation type is unknown or required parameters are missing.
    """
    logger.info(
        f"Normalizing layer '{layer_name}' "
        f"(percentile_norm={percentile_norm})"
    )

    if 'type' not in params:
        raise ValueError(f"Transformation parameters for '{layer_name}' missing 'type' key")

    transform_type = params['type']

    # Optionally pre-clip to percentile range
    if percentile_norm:
        array, p_low, p_high = percentile_clip(array)
    else:
        p_low  = float(np.nanmin(array)) if np.any(~np.isnan(array)) else 0.0
        p_high = float(np.nanmax(array)) if np.any(~np.isnan(array)) else 1.0

    if transform_type == 'linear':
        # When percentile_norm is active, override config vmin/vmax with
        # percentile bounds so the full [p2, p98] range maps to [0, 1].
        vmin = p_low  if percentile_norm else params.get('vmin')
        vmax = p_high if percentile_norm else params.get('vmax')
        if vmin is None or vmax is None:
            raise ValueError("Linear transformation requires 'vmin' and 'vmax' parameters")
        return normalize_linear(array, vmin=vmin, vmax=vmax,
                                invert=params.get('invert', False))

    elif transform_type == 'inverted_linear':
        if not np.any(~np.isnan(array)):
            raise ValueError(
                f"Layer '{layer_name}': all values are NaN — cannot derive vmin/vmax"
            )
        vmin = p_low  if percentile_norm else params.get('vmin', float(np.nanmin(array)))
        vmax = p_high if percentile_norm else params.get('vmax', float(np.nanmax(array)))
        logger.info(f"Inverted linear: using vmin={vmin}, vmax={vmax}")
        return normalize_linear(array, vmin=vmin, vmax=vmax, invert=True)

    elif transform_type == 'sigmoid':
        if 'inflection' not in params or 'slope' not in params:
            raise ValueError("Sigmoid transformation requires 'inflection' and 'slope' parameters")
        return normalize_sigmoid(array,
                                 inflection=params['inflection'],
                                 slope=params['slope'])

    elif transform_type == 'gaussian':
        if 'mean' not in params or 'std' not in params:
            raise ValueError("Gaussian transformation requires 'mean' and 'std' parameters")
        return normalize_gaussian(array,
                                  mean=params['mean'],
                                  std=params['std'])

    else:
        raise ValueError(
            f"Unknown transformation type '{transform_type}' for layer '{layer_name}'. "
            f"Must be one of: linear, inverted_linear, sigmoid, gaussian"
        )


def validate_and_rescale_layer(
    array: np.ndarray,
    profile: dict,
    criterion_key: str,
    config_path: Optional[str] = None
) -> tuple[np.ndarray, dict, dict]:
    """Validate and auto-rescale a raster layer to its expected value range.

    Parameters
    ----------
    array : np.ndarray
        Input raster array (may contain np.nan for NoData).
    profile : dict
        Rasterio profile dict.
    criterion_key : str
        One of: 'ecosystem_condition', 'regulating_es', 'cultural_es',
        'provisioning_es', 'anthropogenic_pressure', 'landuse'
    config_path : str, optional
        Path to criteria_defaults.yaml. Uses default if None.

    Returns
    -------
    array_out : np.ndarray
        Rescaled array (float32), or original if already in range.
    profile_out : dict
        Updated profile (dtype float32 for continuous, int16 for landuse).
    report : dict
        {
            'criterion': str,
            'original_min': float,
            'original_max': float,
            'expected_min': float,
            'expected_max': float,
            'rescaled': bool,
            'method': str,   # 'none', 'linear_rescale', 'warn_only'
            'warning': str or None
        }

    Raises
    ------
    ValueError
        If criterion_key is invalid or config cannot be loaded.

    Notes
    -----
    Rescaling rules:
    - Continuous [0-1] layers (ecosystem_condition, regulating_es, cultural_es,
      provisioning_es): rescale if needed
    - Anthropogenic pressure: accept any positive values, warn if already normalized
    - Land use (landuse): expect integer codes in [111, 523], warn if float
    """
    logger.info(f"Validating and rescaling layer for criterion '{criterion_key}'")

    # Define criterion groups
    continuous_criteria = ['ecosystem_condition', 'regulating_es', 'cultural_es', 'provisioning_es']

    # Validate criterion_key
    valid_criteria = continuous_criteria + ['anthropogenic_pressure', 'landuse']
    if criterion_key not in valid_criteria:
        raise ValueError(
            f"Invalid criterion_key '{criterion_key}'. "
            f"Must be one of: {valid_criteria}"
        )

    # Strip common NoData sentinel values BEFORE range check:
    # -99999, -9999 (float layers), -32768, -32767 (int16), 0 (CLC nodata)
    NODATA_SENTINELS = {-99999.0, -9999.0, -32768.0, -32767.0}
    array_work = array.astype(np.float64)
    for sentinel in NODATA_SENTINELS:
        array_work[array_work == sentinel] = np.nan
    # Also mask float32 min/max sentinels (GDAL convention: ±3.4028e+38)
    f32_max = float(np.finfo(np.float32).max)
    array_work[np.abs(array_work) >= f32_max * 0.9] = np.nan
    # For CLC, 0 and 128 are also NoData
    if criterion_key == 'landuse':
        array_work[array_work == 0] = np.nan
        array_work[array_work == 128] = np.nan

    # Calculate raw min/max BEFORE negative masking (for reporting)
    valid_mask = ~np.isnan(array_work)
    if not np.any(valid_mask):
        raise ValueError(f"Array for '{criterion_key}' contains only NaN/NoData values")

    original_min = float(np.nanmin(array_work))
    original_max = float(np.nanmax(array_work))

    # For continuous [0-1] layers, mask negative values as NoData AFTER recording original range
    continuous_criteria = {'ecosystem_condition', 'regulating_es', 'cultural_es', 'provisioning_es'}
    if criterion_key in continuous_criteria:
        n_negative = int(np.sum(array_work < 0))
        if n_negative > 0:
            array_work[array_work < 0] = np.nan
            logger.warning(
                f"Masked {n_negative} negative pixels to NaN in '{criterion_key}' "
                "(negative values are invalid for [0-1] integrity layers)"
            )

    logger.info(f"Original value range: [{original_min:.4f}, {original_max:.4f}]")

    # Initialize report
    report = {
        'criterion': criterion_key,
        'original_min': original_min,
        'original_max': original_max,
        'expected_min': None,
        'expected_max': None,
        'rescaled': False,
        'method': 'none',
        'warning': None
    }

    # Copy array_work (already masked) as the output base — not the raw array
    array_out = array_work.astype(np.float32)
    profile_out = profile.copy()
    profile_out['dtype'] = 'float32'

    # ===================================================================
    # CONTINUOUS [0-1] LAYERS
    # ===================================================================
    if criterion_key in continuous_criteria:
        report['expected_min'] = 0.0
        report['expected_max'] = 1.0

        # Case 1: Already in [0, 1]
        if original_min >= 0.0 and original_max <= 1.0:
            logger.info("Values already in [0, 1] range, no rescaling needed")
            report['method'] = 'none'
            profile_out['dtype'] = 'float32'
            array_out = array_out.astype(np.float32)

        # Case 2: Percentage scale [0, 100]
        elif original_min >= 0.0 and original_max > 1.0 and original_max <= 100.0:
            logger.info("Values appear to be percentages, dividing by 100")
            array_out = array_out / 100.0
            array_out = np.clip(array_out, 0.0, 1.0)
            report['rescaled'] = True
            report['method'] = 'linear_rescale'
            profile_out['dtype'] = 'float32'
            array_out = array_out.astype(np.float32)

        # Case 3: Arbitrary range - linear min-max rescale
        else:
            logger.info("Values outside [0, 100], applying min-max rescaling")
            # Linear rescale: (x - min) / (max - min)
            if original_max > original_min:
                array_out = (array_out - original_min) / (original_max - original_min)
                array_out = np.clip(array_out, 0.0, 1.0)
            else:
                # Edge case: all values are the same
                logger.warning(f"All values are identical ({original_min}), setting to 0.5")
                array_out = np.where(valid_mask, 0.5, np.nan)

            report['rescaled'] = True
            report['method'] = 'linear_rescale'
            profile_out['dtype'] = 'float32'
            array_out = array_out.astype(np.float32)

    # ===================================================================
    # ANTHROPOGENIC PRESSURE
    # ===================================================================
    elif criterion_key == 'anthropogenic_pressure':
        report['expected_min'] = 0.0
        report['expected_max'] = None  # No upper limit

        # Accept any positive values
        if original_min < 0.0:
            logger.warning(f"Negative values found in pressure layer (min={original_min})")
            report['warning'] = f"Negative values found (min={original_min}). Expected positive values only."

        # Warn if already normalized
        if original_min >= 0.0 and original_max <= 1.0:
            warning_msg = (
                "Values appear already normalised to [0, 1]. "
                "MCE will apply sigmoid/linear transform which may compress range further."
            )
            logger.warning(warning_msg)
            report['warning'] = warning_msg
            report['method'] = 'warn_only'

        profile_out['dtype'] = 'float32'
        array_out = array_out.astype(np.float32)

    # ===================================================================
    # LAND USE / CLC
    # ===================================================================
    elif criterion_key == 'landuse':
        report['expected_min'] = 111
        report['expected_max'] = 523

        # Warn if float values in [0, 1] (pre-normalized)
        if array.dtype in [np.float32, np.float64]:
            if original_min >= 0.0 and original_max <= 1.0:
                warning_msg = (
                    "Values are floats in [0, 1], which looks like a pre-normalised layer, "
                    "not raw CLC codes [111-523]."
                )
                logger.warning(warning_msg)
                report['warning'] = warning_msg
                report['method'] = 'warn_only'

        # Warn if outside expected range
        if original_min < 0.0 or original_max > 999:
            warning_msg = f"Unexpected value range for land use: [{original_min}, {original_max}]. Expected [111-523]."
            logger.warning(warning_msg)
            report['warning'] = warning_msg
            report['method'] = 'warn_only'

        # Keep as-is (reclassification happens separately)
        # Use int16 for categorical data
        if np.issubdtype(array.dtype, np.integer):
            profile_out['dtype'] = 'int16'
            array_out = array_out.astype(np.int16)
        else:
            # If it's float, keep as float32
            profile_out['dtype'] = 'float32'
            array_out = array_out.astype(np.float32)

    logger.info(f"Validation complete: rescaled={report['rescaled']}, method={report['method']}")
    return array_out, profile_out, report


def validate_and_rescale_all_layers(
    raster_dict: dict[str, tuple[np.ndarray, dict]],
    config_path: Optional[str] = None
) -> tuple[dict[str, tuple[np.ndarray, dict]], list[dict]]:
    """Run validate_and_rescale_layer on all layers.

    Parameters
    ----------
    raster_dict : dict[str, tuple[np.ndarray, dict]]
        Dictionary mapping criterion keys to (array, profile) tuples.
        Keys must be valid criterion names: 'ecosystem_condition', 'regulating_es',
        'cultural_es', 'provisioning_es', 'anthropogenic_pressure', 'landuse'.
    config_path : str, optional
        Path to criteria_defaults.yaml. Uses default if None.

    Returns
    -------
    updated_dict : dict[str, tuple[np.ndarray, dict]]
        Dictionary with validated and potentially rescaled rasters.
    reports : list[dict]
        List of validation reports, one per layer.

    Raises
    ------
    ValueError
        If any criterion key is invalid.

    Notes
    -----
    Logs warnings for any layers that require attention (e.g., pre-normalized
    pressure layers, float land use values).
    """
    logger.info(f"Validating and rescaling {len(raster_dict)} layers")

    updated_dict = {}
    reports = []

    for criterion_key, (array, profile) in raster_dict.items():
        logger.info(f"Processing '{criterion_key}'...")

        try:
            array_out, profile_out, report = validate_and_rescale_layer(
                array=array,
                profile=profile,
                criterion_key=criterion_key,
                config_path=config_path
            )

            updated_dict[criterion_key] = (array_out, profile_out)
            reports.append(report)

            # Log report summary
            if report['rescaled']:
                logger.info(
                    f"  ✓ Rescaled from [{report['original_min']:.4f}, {report['original_max']:.4f}] "
                    f"to [{report['expected_min']:.4f}, {report['expected_max']:.4f}]"
                )
            elif report['warning']:
                logger.warning(f"  ⚠ {report['warning']}")
            else:
                logger.info("  ✓ No rescaling needed")

        except Exception as e:
            logger.error(f"Failed to validate '{criterion_key}': {e}")
            raise

    logger.info(f"All layers validated. {sum(r['rescaled'] for r in reports)} rescaled, "
                f"{sum(1 for r in reports if r['warning']) } warnings.")

    return updated_dict, reports


# Legacy function stub for backward compatibility
def harmonise_raster(input_raster, target_crs, target_resolution, resampling_method="bilinear"):
    """
    Reproject and resample input raster to common grid.

    Args:
        input_raster: Path to input raster or xarray DataArray
        target_crs: Target CRS (e.g., 'EPSG:3035')
        target_resolution: Target resolution in meters
        resampling_method: Resampling algorithm ('bilinear', 'nearest', 'cubic')

    Returns:
        xarray DataArray in target CRS and resolution
    """
    raise NotImplementedError("Use load_raster, reproject_raster, and resample_raster instead")


# Legacy function stub for backward compatibility
def apply_transformation_function(raster_array, transform_config):
    """
    Apply transformation function to normalise criterion values to [0,1].

    Args:
        raster_array: Input xarray DataArray
        transform_config: Parameters from transformation_functions.yaml

    Returns:
        Transformed xarray DataArray with values in [0,1]
    """
    raise NotImplementedError("Use normalize_layer instead")

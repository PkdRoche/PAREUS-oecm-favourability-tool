"""Corine Land Cover (CLC) loader and reclassification module.

This module provides functions for loading, clipping, resampling, and reclassifying
Corine Land Cover (CLC) raster data for the OECM Favourability Tool.
"""

import logging
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import yaml
from pathlib import Path
import shapely.geometry

logger = logging.getLogger(__name__)


def load_clc(
    filepath: str,
    study_area_geom: shapely.geometry.base.BaseGeometry,
    target_resolution: float = 100
) -> tuple[np.ndarray, dict]:
    """Load a local CLC GeoTIFF, clip to study area, and optionally resample.

    Parameters
    ----------
    filepath : str
        Path to the CLC GeoTIFF file (must be in EPSG:3035).
    study_area_geom : shapely.geometry.base.BaseGeometry
        Study area geometry in EPSG:3035 for clipping.
    target_resolution : float, optional
        Target resolution in meters. Default is 100 (CLC native resolution).
        Uses nearest-neighbour resampling for categorical data.

    Returns
    -------
    tuple[np.ndarray, dict]
        A tuple containing:
        - array : np.ndarray
            2D numpy array of CLC codes (int16), clipped and resampled.
        - profile : dict
            Rasterio profile dictionary containing metadata (CRS, transform, etc.).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the CLC raster is not in EPSG:3035 or cannot be read.

    Notes
    -----
    CLC is a categorical dataset with integer codes (111-523). Native resolution
    is 100m in EPSG:3035. NoData values are typically 0 or 128.
    """
    logger.info(f"Loading CLC raster from {filepath}")

    if not Path(filepath).exists():
        raise FileNotFoundError(f"CLC file not found: {filepath}")

    with rasterio.open(filepath) as src:
        # Validate CRS is EPSG:3035
        if src.crs is None:
            raise ValueError(f"CLC raster has no CRS information: {filepath}")

        src_crs_code = src.crs.to_epsg()
        if src_crs_code != 3035:
            raise ValueError(
                f"CLC raster must be in EPSG:3035, found EPSG:{src_crs_code}. "
                f"Please reproject the input file to EPSG:3035 before loading."
            )

        # Clip to study area geometry
        logger.info("Clipping CLC to study area geometry")
        try:
            clipped_array, clipped_transform = rio_mask(
                src,
                [study_area_geom],
                crop=True,
                filled=True,
                nodata=0
            )
        except Exception as e:
            raise ValueError(f"Failed to clip CLC raster to study area: {e}")

        # Extract first band (CLC is single-band)
        clipped_array = clipped_array[0]

        # Build clipped profile
        clipped_profile = src.profile.copy()
        clipped_profile.update({
            'height': clipped_array.shape[0],
            'width': clipped_array.shape[1],
            'transform': clipped_transform
        })

    logger.info(f"Clipped CLC to shape {clipped_array.shape}")

    # Resample if target resolution differs from current
    current_res = abs(clipped_profile['transform'][0])

    if not np.isclose(current_res, target_resolution, rtol=0.01):
        logger.info(f"Resampling CLC from {current_res}m to {target_resolution}m using nearest-neighbour")
        clipped_array, clipped_profile = _resample_clc(
            clipped_array,
            clipped_profile,
            target_resolution
        )
    else:
        logger.info(f"CLC already at target resolution ({target_resolution}m), skipping resampling")

    logger.info(f"Loaded CLC raster: shape={clipped_array.shape}, dtype={clipped_array.dtype}")
    return clipped_array, clipped_profile


def _resample_clc(
    array: np.ndarray,
    profile: dict,
    target_resolution: float
) -> tuple[np.ndarray, dict]:
    """Resample CLC raster to target resolution using nearest-neighbour.

    Parameters
    ----------
    array : np.ndarray
        Input CLC array (int16).
    profile : dict
        Raster profile containing transform and bounds.
    target_resolution : float
        Target resolution in meters.

    Returns
    -------
    tuple[np.ndarray, dict]
        Resampled array and updated profile.

    Notes
    -----
    CLC is categorical data, so nearest-neighbour resampling is mandatory
    to preserve class codes.
    """
    # Calculate current resolution
    transform = profile['transform']
    current_res_x = abs(transform[0])
    current_res_y = abs(transform[4])

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

    # Resample using nearest-neighbour (mandatory for categorical data)
    reproject(
        source=array,
        destination=dst_array,
        src_transform=transform,
        src_crs=profile['crs'],
        dst_transform=new_transform,
        dst_crs=profile['crs'],
        resampling=Resampling.nearest
    )

    # Update profile
    dst_profile = profile.copy()
    dst_profile.update({
        'transform': new_transform,
        'width': new_width,
        'height': new_height
    })

    logger.info(f"Resampled CLC from {array.shape} to {dst_array.shape}")
    return dst_array, dst_profile


def reclassify_clc(
    array: np.ndarray,
    reclassification_table: dict[int, float]
) -> np.ndarray:
    """Reclassify CLC codes to continuous scores in [0, 1].

    Parameters
    ----------
    array : np.ndarray
        Raw CLC int16 array with codes 111-523 (nodata=0 or 128).
    reclassification_table : dict[int, float]
        Mapping from CLC integer code to float score in [0.0, 1.0].
        Example: {111: 0.1, 112: 0.2, ...}

    Returns
    -------
    np.ndarray
        Float32 array with values in [0, 1]. Pixels with no mapping become np.nan.

    Notes
    -----
    - NoData values (0 and 128) are automatically converted to np.nan.
    - CLC codes not present in reclassification_table are also set to np.nan.
    - Output dtype is always float32 to support np.nan.
    """
    logger.info(f"Reclassifying CLC codes using {len(reclassification_table)} class mappings")

    # Create output array as float32
    output = np.full(array.shape, np.nan, dtype=np.float32)

    # Track statistics
    mapped_count = 0
    unmapped_count = 0
    nodata_count = 0

    # Get unique values in input array
    unique_values = np.unique(array)

    for clc_code in unique_values:
        # Skip nodata values (0 and 128)
        if clc_code in [0, 128]:
            nodata_count += np.sum(array == clc_code)
            continue

        # Apply mapping if exists
        if clc_code in reclassification_table:
            score = reclassification_table[clc_code]
            if not (0.0 <= score <= 1.0):
                logger.warning(
                    f"Score {score} for CLC code {clc_code} is outside [0, 1] range. Clipping."
                )
                score = np.clip(score, 0.0, 1.0)

            mask = array == clc_code
            output[mask] = score
            mapped_count += np.sum(mask)
        else:
            # Code not in table -> remains np.nan
            unmapped_count += np.sum(array == clc_code)
            logger.debug(f"CLC code {clc_code} not in reclassification table, setting to NaN")

    logger.info(
        f"Reclassification complete: {mapped_count} pixels mapped, "
        f"{unmapped_count} unmapped (NaN), {nodata_count} nodata (NaN)"
    )

    return output


def load_and_reclassify_clc(
    filepath: str,
    study_area_geom: shapely.geometry.base.BaseGeometry,
    config_path: str,
    target_resolution: float = 100
) -> tuple[np.ndarray, dict]:
    """Convenience wrapper: load CLC and apply reclassification in one step.

    Parameters
    ----------
    filepath : str
        Path to the CLC GeoTIFF file (must be in EPSG:3035).
    study_area_geom : shapely.geometry.base.BaseGeometry
        Study area geometry in EPSG:3035 for clipping.
    config_path : str
        Path to YAML configuration file containing reclassification table.
        Expected structure: {reclassification: {111: 0.5, 112: 0.6, ...}}
        or {reclassification: {111: {score: 0.5, ...}, 112: {score: 0.6, ...}, ...}}
    target_resolution : float, optional
        Target resolution in meters. Default is 100.

    Returns
    -------
    tuple[np.ndarray, dict]
        A tuple containing:
        - scored_array : np.ndarray
            Float32 array with reclassified scores in [0, 1].
        - profile : dict
            Rasterio profile dictionary.

    Raises
    ------
    FileNotFoundError
        If filepath or config_path does not exist.
    ValueError
        If CLC is not in EPSG:3035 or config is invalid.
    KeyError
        If config file does not contain 'reclassification' key.

    Notes
    -----
    This is a convenience function that chains load_clc() and reclassify_clc().
    Config file supports both simple format (code: score) and nested format
    (code: {score: value, label: ..., ...}).
    """
    logger.info(f"Loading and reclassifying CLC from {filepath}")

    # Load configuration
    if not Path(config_path).exists():
        raise FileNotFoundError(f"CLC reclassification config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'reclassification' not in config:
        raise KeyError(
            f"Config file {config_path} must contain 'reclassification' key "
            f"with mapping of CLC codes to scores"
        )

    reclassification_raw = config['reclassification']

    # Validate reclassification table
    if not isinstance(reclassification_raw, dict):
        raise ValueError(
            f"'reclassification' must be a dict, found {type(reclassification_raw)}"
        )

    # Extract scores from nested or simple format
    reclassification_table = {}
    for clc_code, value in reclassification_raw.items():
        if isinstance(value, dict):
            # Nested format: {111: {score: 0.5, label: "...", ...}}
            if 'score' not in value:
                raise ValueError(
                    f"CLC code {clc_code} uses nested format but missing 'score' key"
                )
            reclassification_table[clc_code] = value['score']
        else:
            # Simple format: {111: 0.5}
            reclassification_table[clc_code] = value

    logger.info(f"Loaded {len(reclassification_table)} CLC class mappings from config")

    # Load CLC raster
    clc_array, clc_profile = load_clc(filepath, study_area_geom, target_resolution)

    # Reclassify
    scored_array = reclassify_clc(clc_array, reclassification_table)

    # Update profile to reflect reclassified output dtype
    clc_profile = dict(clc_profile)
    clc_profile['dtype'] = 'float32'
    clc_profile['nodata'] = float('nan')

    logger.info(
        f"CLC loaded and reclassified: "
        f"score range [{np.nanmin(scored_array):.3f}, {np.nanmax(scored_array):.3f}]"
    )

    return scored_array, clc_profile


def get_clc_legend() -> dict[int, dict[str, str]]:
    """Return the full CLC 2018 44-class legend.

    Returns
    -------
    dict[int, dict[str, str]]
        Dictionary mapping CLC code to metadata:
        {
            111: {
                "label": "Continuous urban fabric",
                "level1": "Artificial surfaces",
                "level2": "Urban fabric",
                "level3": "Continuous urban fabric"
            },
            ...
        }

    Notes
    -----
    CLC 2018 nomenclature has 44 classes organized in 3 hierarchical levels:
    - Level 1: 5 major categories (1-5)
    - Level 2: 15 subcategories
    - Level 3: 44 classes (codes 111-523)

    Reference: https://land.copernicus.eu/user-corner/technical-library/corine-land-cover-nomenclature-guidelines/html/
    """
    legend = {
        # 1. Artificial surfaces
        111: {
            "label": "Continuous urban fabric",
            "level1": "Artificial surfaces",
            "level2": "Urban fabric",
            "level3": "Continuous urban fabric"
        },
        112: {
            "label": "Discontinuous urban fabric",
            "level1": "Artificial surfaces",
            "level2": "Urban fabric",
            "level3": "Discontinuous urban fabric"
        },
        121: {
            "label": "Industrial or commercial units",
            "level1": "Artificial surfaces",
            "level2": "Industrial, commercial and transport units",
            "level3": "Industrial or commercial units"
        },
        122: {
            "label": "Road and rail networks and associated land",
            "level1": "Artificial surfaces",
            "level2": "Industrial, commercial and transport units",
            "level3": "Road and rail networks and associated land"
        },
        123: {
            "label": "Port areas",
            "level1": "Artificial surfaces",
            "level2": "Industrial, commercial and transport units",
            "level3": "Port areas"
        },
        124: {
            "label": "Airports",
            "level1": "Artificial surfaces",
            "level2": "Industrial, commercial and transport units",
            "level3": "Airports"
        },
        131: {
            "label": "Mineral extraction sites",
            "level1": "Artificial surfaces",
            "level2": "Mine, dump and construction sites",
            "level3": "Mineral extraction sites"
        },
        132: {
            "label": "Dump sites",
            "level1": "Artificial surfaces",
            "level2": "Mine, dump and construction sites",
            "level3": "Dump sites"
        },
        133: {
            "label": "Construction sites",
            "level1": "Artificial surfaces",
            "level2": "Mine, dump and construction sites",
            "level3": "Construction sites"
        },
        141: {
            "label": "Green urban areas",
            "level1": "Artificial surfaces",
            "level2": "Artificial, non-agricultural vegetated areas",
            "level3": "Green urban areas"
        },
        142: {
            "label": "Sport and leisure facilities",
            "level1": "Artificial surfaces",
            "level2": "Artificial, non-agricultural vegetated areas",
            "level3": "Sport and leisure facilities"
        },

        # 2. Agricultural areas
        211: {
            "label": "Non-irrigated arable land",
            "level1": "Agricultural areas",
            "level2": "Arable land",
            "level3": "Non-irrigated arable land"
        },
        212: {
            "label": "Permanently irrigated land",
            "level1": "Agricultural areas",
            "level2": "Arable land",
            "level3": "Permanently irrigated land"
        },
        213: {
            "label": "Rice fields",
            "level1": "Agricultural areas",
            "level2": "Arable land",
            "level3": "Rice fields"
        },
        221: {
            "label": "Vineyards",
            "level1": "Agricultural areas",
            "level2": "Permanent crops",
            "level3": "Vineyards"
        },
        222: {
            "label": "Fruit trees and berry plantations",
            "level1": "Agricultural areas",
            "level2": "Permanent crops",
            "level3": "Fruit trees and berry plantations"
        },
        223: {
            "label": "Olive groves",
            "level1": "Agricultural areas",
            "level2": "Permanent crops",
            "level3": "Olive groves"
        },
        231: {
            "label": "Pastures",
            "level1": "Agricultural areas",
            "level2": "Pastures",
            "level3": "Pastures"
        },
        241: {
            "label": "Annual crops associated with permanent crops",
            "level1": "Agricultural areas",
            "level2": "Heterogeneous agricultural areas",
            "level3": "Annual crops associated with permanent crops"
        },
        242: {
            "label": "Complex cultivation patterns",
            "level1": "Agricultural areas",
            "level2": "Heterogeneous agricultural areas",
            "level3": "Complex cultivation patterns"
        },
        243: {
            "label": "Land principally occupied by agriculture with significant areas of natural vegetation",
            "level1": "Agricultural areas",
            "level2": "Heterogeneous agricultural areas",
            "level3": "Land principally occupied by agriculture with significant areas of natural vegetation"
        },
        244: {
            "label": "Agro-forestry areas",
            "level1": "Agricultural areas",
            "level2": "Heterogeneous agricultural areas",
            "level3": "Agro-forestry areas"
        },

        # 3. Forest and semi-natural areas
        311: {
            "label": "Broad-leaved forest",
            "level1": "Forest and semi-natural areas",
            "level2": "Forests",
            "level3": "Broad-leaved forest"
        },
        312: {
            "label": "Coniferous forest",
            "level1": "Forest and semi-natural areas",
            "level2": "Forests",
            "level3": "Coniferous forest"
        },
        313: {
            "label": "Mixed forest",
            "level1": "Forest and semi-natural areas",
            "level2": "Forests",
            "level3": "Mixed forest"
        },
        321: {
            "label": "Natural grasslands",
            "level1": "Forest and semi-natural areas",
            "level2": "Scrub and/or herbaceous vegetation associations",
            "level3": "Natural grasslands"
        },
        322: {
            "label": "Moors and heathland",
            "level1": "Forest and semi-natural areas",
            "level2": "Scrub and/or herbaceous vegetation associations",
            "level3": "Moors and heathland"
        },
        323: {
            "label": "Sclerophyllous vegetation",
            "level1": "Forest and semi-natural areas",
            "level2": "Scrub and/or herbaceous vegetation associations",
            "level3": "Sclerophyllous vegetation"
        },
        324: {
            "label": "Transitional woodland-shrub",
            "level1": "Forest and semi-natural areas",
            "level2": "Scrub and/or herbaceous vegetation associations",
            "level3": "Transitional woodland-shrub"
        },
        331: {
            "label": "Beaches, dunes, sands",
            "level1": "Forest and semi-natural areas",
            "level2": "Open spaces with little or no vegetation",
            "level3": "Beaches, dunes, sands"
        },
        332: {
            "label": "Bare rocks",
            "level1": "Forest and semi-natural areas",
            "level2": "Open spaces with little or no vegetation",
            "level3": "Bare rocks"
        },
        333: {
            "label": "Sparsely vegetated areas",
            "level1": "Forest and semi-natural areas",
            "level2": "Open spaces with little or no vegetation",
            "level3": "Sparsely vegetated areas"
        },
        334: {
            "label": "Burnt areas",
            "level1": "Forest and semi-natural areas",
            "level2": "Open spaces with little or no vegetation",
            "level3": "Burnt areas"
        },
        335: {
            "label": "Glaciers and perpetual snow",
            "level1": "Forest and semi-natural areas",
            "level2": "Open spaces with little or no vegetation",
            "level3": "Glaciers and perpetual snow"
        },

        # 4. Wetlands
        411: {
            "label": "Inland marshes",
            "level1": "Wetlands",
            "level2": "Inland wetlands",
            "level3": "Inland marshes"
        },
        412: {
            "label": "Peat bogs",
            "level1": "Wetlands",
            "level2": "Inland wetlands",
            "level3": "Peat bogs"
        },
        421: {
            "label": "Salt marshes",
            "level1": "Wetlands",
            "level2": "Maritime wetlands",
            "level3": "Salt marshes"
        },
        422: {
            "label": "Salines",
            "level1": "Wetlands",
            "level2": "Maritime wetlands",
            "level3": "Salines"
        },
        423: {
            "label": "Intertidal flats",
            "level1": "Wetlands",
            "level2": "Maritime wetlands",
            "level3": "Intertidal flats"
        },

        # 5. Water bodies
        511: {
            "label": "Water courses",
            "level1": "Water bodies",
            "level2": "Inland waters",
            "level3": "Water courses"
        },
        512: {
            "label": "Water bodies",
            "level1": "Water bodies",
            "level2": "Inland waters",
            "level3": "Water bodies"
        },
        521: {
            "label": "Coastal lagoons",
            "level1": "Water bodies",
            "level2": "Marine waters",
            "level3": "Coastal lagoons"
        },
        522: {
            "label": "Estuaries",
            "level1": "Water bodies",
            "level2": "Marine waters",
            "level3": "Estuaries"
        },
        523: {
            "label": "Sea and ocean",
            "level1": "Water bodies",
            "level2": "Marine waters",
            "level3": "Sea and ocean"
        }
    }

    logger.debug(f"CLC legend contains {len(legend)} classes")
    return legend

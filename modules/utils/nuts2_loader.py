"""NUTS2 region loader for Eurostat administrative boundaries."""
import io
import streamlit as st
import geopandas as gpd
import requests
from typing import Optional
import shapely.geometry


@st.cache_data(show_spinner="Loading NUTS2 boundaries from Eurostat...")
def load_nuts2(year: int = 2021, scale: str = "20M") -> gpd.GeoDataFrame:
    """
    Download NUTS2 boundaries from Eurostat and return as GeoDataFrame (EPSG:3035).
    Caches result in memory using @st.cache_data.

    Parameters
    ----------
    year : int, optional
        NUTS reference year (default 2021).
    scale : str, optional
        Map scale resolution: "03M", "10M", "20M", "60M" (default "20M").

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with columns: NUTS_ID, CNTR_CODE, NUTS_NAME, LEVL_CODE, geometry
        Filtered to LEVL_CODE == 2 (NUTS2 only), in EPSG:3035.

    Raises
    ------
    requests.HTTPError
        If download fails or URL is invalid.
    ValueError
        If the returned GeoJSON contains no NUTS2 regions.

    Notes
    -----
    URL pattern:
    https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_{scale}_{year}_3035.geojson
    """
    url = (
        f"https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
        f"NUTS_RG_{scale}_{year}_3035.geojson"
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise requests.HTTPError(
            f"Failed to download NUTS2 boundaries from Eurostat: {e}"
        )

    # Load GeoJSON into GeoDataFrame from the already-downloaded bytes
    # (avoids re-downloading the same URL a second time via gpd.read_file(url))
    gdf = gpd.read_file(io.BytesIO(response.content))

    # Filter to NUTS2 level only (LEVL_CODE == 2)
    gdf_nuts2 = gdf[gdf['LEVL_CODE'] == 2].copy()

    if len(gdf_nuts2) == 0:
        raise ValueError(
            f"No NUTS2 regions found in downloaded data for year {year}, scale {scale}"
        )

    # Ensure CRS is EPSG:3035
    if gdf_nuts2.crs is None:
        gdf_nuts2.set_crs("EPSG:3035", inplace=True)
    elif gdf_nuts2.crs.to_epsg() != 3035:
        gdf_nuts2 = gdf_nuts2.to_crs("EPSG:3035")

    return gdf_nuts2


def get_countries(nuts2_gdf: gpd.GeoDataFrame) -> list[str]:
    """
    Return sorted list of unique country codes from NUTS2 GeoDataFrame.

    Parameters
    ----------
    nuts2_gdf : gpd.GeoDataFrame
        GeoDataFrame with NUTS2 regions containing CNTR_CODE column.

    Returns
    -------
    list[str]
        Sorted list of unique 2-letter ISO country codes.
    """
    countries = sorted(nuts2_gdf['CNTR_CODE'].unique().tolist())
    return countries


def get_nuts2_for_country(
    nuts2_gdf: gpd.GeoDataFrame,
    country_code: str
) -> gpd.GeoDataFrame:
    """
    Return NUTS2 regions for a specific country, sorted by name.

    Parameters
    ----------
    nuts2_gdf : gpd.GeoDataFrame
        GeoDataFrame with NUTS2 regions.
    country_code : str
        2-letter ISO country code (e.g., "FR", "DE").

    Returns
    -------
    gpd.GeoDataFrame
        Filtered GeoDataFrame containing only regions for the specified country,
        sorted by NUTS_NAME.
    """
    filtered = nuts2_gdf[nuts2_gdf['CNTR_CODE'] == country_code].copy()
    filtered = filtered.sort_values('NUTS_NAME').reset_index(drop=True)
    return filtered


def get_nuts2_geometry(
    nuts2_gdf: gpd.GeoDataFrame,
    nuts_id: str
) -> Optional[shapely.geometry.base.BaseGeometry]:
    """
    Return the geometry for the given NUTS_ID.

    Parameters
    ----------
    nuts2_gdf : gpd.GeoDataFrame
        GeoDataFrame with NUTS2 regions.
    nuts_id : str
        NUTS identifier (e.g., "FR10", "DE21").

    Returns
    -------
    shapely.geometry.base.BaseGeometry or None
        Geometry in EPSG:3035. Returns None if NUTS_ID not found.
    """
    row = nuts2_gdf[nuts2_gdf['NUTS_ID'] == nuts_id]

    if len(row) == 0:
        return None

    return row.iloc[0].geometry

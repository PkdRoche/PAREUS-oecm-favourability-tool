"""Phase 4 gate verification: GeoTIFF export functionality."""

import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.transform import from_origin
import sys
from pathlib import Path
import tempfile

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.module1_protected_areas.gap_analysis import strict_gaps, export_gap_masks_as_raster


def test_gap_geotiff_export():
    """Verify that export_gap_masks_as_raster produces valid GeoTIFF files."""
    # Create synthetic territory and PA
    territory_geom = box(3000000, 1990000, 3010000, 2000000)
    territory_gdf = gpd.GeoDataFrame({'geometry': [territory_geom]}, crs='EPSG:3035')

    # PA covering top-left quarter only
    pa_geom = box(3000000, 1995000, 3005000, 2000000)
    pa_gdf = gpd.GeoDataFrame({
        'geometry': [pa_geom],
        'protection_class': ['strict_core']
    }, crs='EPSG:3035')

    # Compute strict gaps
    gaps = strict_gaps(pa_gdf, territory_geom)
    print(f"Strict gaps computed: {len(gaps)} features, area = {gaps.geometry.area.sum()/10000:.1f} ha")

    assert len(gaps) > 0, "Should have at least one gap feature"
    gap_area = gaps.geometry.area.sum() / 10000
    assert gap_area > 0, "Gap area should be positive"

    # Export to GeoTIFF
    reference_profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': 100,
        'height': 100,
        'count': 1,
        'crs': 'EPSG:3035',
        'transform': from_origin(3000000, 2000000, 100, 100),
        'nodata': 255
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = export_gap_masks_as_raster({'strict_gaps': gaps}, reference_profile, tmpdir)

        # Verify output paths
        assert 'strict_gaps' in paths, "Should return path for strict_gaps"
        assert Path(paths['strict_gaps']).exists(), "GeoTIFF file should exist"

        # Verify raster properties
        for name, path in paths.items():
            with rasterio.open(path) as src:
                print(f"GeoTIFF {name}: shape={src.shape}, crs={src.crs}, dtype={src.dtypes[0]}")

                # Check CRS
                assert src.crs.to_epsg() == 3035, f"Expected CRS EPSG:3035, got {src.crs}"

                # Check dimensions
                assert src.shape == (100, 100), f"Expected shape (100, 100), got {src.shape}"

                # Check dtype
                assert src.dtypes[0] in ['float32', 'uint8'], f"Unexpected dtype: {src.dtypes[0]}"

                # Check data range (should be 0.0 or 1.0 for gaps)
                data = src.read(1)
                unique_vals = np.unique(data)
                print(f"  Unique values in raster: {unique_vals}")

                # Gap areas should be marked as 1.0
                assert 1.0 in unique_vals or 1 in unique_vals, "Gap areas should be marked as 1"

    print("Phase 4 gate: gap GeoTIFFs VERIFIED")


if __name__ == '__main__':
    test_gap_geotiff_export()

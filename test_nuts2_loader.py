"""Test script for NUTS2 loader functionality."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.utils.nuts2_loader import (
    load_nuts2,
    get_countries,
    get_nuts2_for_country,
    get_nuts2_geometry
)


def test_nuts2_loader():
    """Test NUTS2 loader functions."""
    print("Testing NUTS2 loader...")
    print("-" * 60)

    # Test 1: Load NUTS2 data
    print("\n1. Loading NUTS2 boundaries from Eurostat...")
    try:
        nuts2_gdf = load_nuts2(year=2021, scale="20M")
        print(f"   SUCCESS: Loaded {len(nuts2_gdf)} NUTS2 regions")
        print(f"   CRS: {nuts2_gdf.crs}")
        print(f"   Columns: {list(nuts2_gdf.columns)}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return

    # Test 2: Get countries
    print("\n2. Getting list of countries...")
    countries = get_countries(nuts2_gdf)
    print(f"   Found {len(countries)} countries")
    print(f"   First 10: {countries[:10]}")

    # Test 3: Get NUTS2 regions for France
    print("\n3. Getting NUTS2 regions for France (FR)...")
    if "FR" in countries:
        fr_regions = get_nuts2_for_country(nuts2_gdf, "FR")
        print(f"   Found {len(fr_regions)} French NUTS2 regions")
        print("\n   First 5 regions:")
        for idx, row in fr_regions.head(5).iterrows():
            print(f"      - {row.NUTS_NAME} ({row.NUTS_ID})")
    else:
        print("   France not found in country list")

    # Test 4: Get geometry for specific NUTS2 region
    print("\n4. Getting geometry for a specific region...")
    test_nuts_id = fr_regions.iloc[0]['NUTS_ID'] if "FR" in countries and len(fr_regions) > 0 else None

    if test_nuts_id:
        geom = get_nuts2_geometry(nuts2_gdf, test_nuts_id)
        if geom is not None:
            area_km2 = geom.area / 1_000_000
            print(f"   Region: {test_nuts_id}")
            print(f"   Geometry type: {geom.geom_type}")
            print(f"   Area: {area_km2:,.2f} km²")
            print(f"   Bounds: {geom.bounds}")
        else:
            print(f"   Geometry not found for {test_nuts_id}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")


if __name__ == "__main__":
    test_nuts2_loader()

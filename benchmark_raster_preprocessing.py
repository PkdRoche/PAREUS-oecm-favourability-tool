"""Performance benchmarks for raster_preprocessing.py.

Measures execution time of key functions using synthetic test data.
Flags operations exceeding 500ms threshold.
"""

import time
import numpy as np
from pathlib import Path
import sys

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.module2_favourability.raster_preprocessing import (
    load_raster,
    reproject_raster,
    resample_raster,
    align_rasters,
    apply_nodata_mask,
    normalize_linear,
    normalize_sigmoid,
    normalize_gaussian
)


def benchmark(func, *args, **kwargs):
    """Run function and measure execution time in milliseconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
    return result, elapsed


def format_time(ms):
    """Format time with warning flag if > 500ms."""
    flag = " [SLOW!]" if ms > 500 else ""
    return f"{ms:.2f}ms{flag}"


def main():
    """Run performance diagnostics."""
    print("=" * 70)
    print("RASTER PREPROCESSING PERFORMANCE DIAGNOSTICS")
    print("=" * 70)
    print()

    # Test data paths
    test_data_dir = Path(__file__).parent / "tests" / "synthetic_data"
    test_files = [
        "anthropogenic_pressure.tif",
        "cultural_es.tif",
        "ecosystem_condition.tif",
        "provisioning_es.tif",
        "regulating_es.tif"
    ]

    # Verify test data exists
    if not test_data_dir.exists():
        print(f"ERROR: Test data directory not found: {test_data_dir}")
        return 1

    print(f"Test data directory: {test_data_dir}")
    print(f"Testing with {len(test_files)} synthetic 100×100 GeoTIFFs")
    print()

    # -------------------------------------------------------------------------
    # 1. Benchmark load_raster
    # -------------------------------------------------------------------------
    print("1. load_raster()")
    print("-" * 70)

    load_times = []
    rasters = {}

    for filename in test_files:
        filepath = test_data_dir / filename
        if not filepath.exists():
            print(f"   SKIP: {filename} (not found)")
            continue

        (array, profile), elapsed = benchmark(load_raster, str(filepath))
        load_times.append(elapsed)
        rasters[filename] = (array, profile)

        print(f"   {filename:30s}  {format_time(elapsed):>15s}  shape={array.shape}")

    if load_times:
        avg_load = np.mean(load_times)
        print(f"\n   Average load time: {format_time(avg_load)}")
    print()

    # -------------------------------------------------------------------------
    # 2. Benchmark reproject_raster
    # -------------------------------------------------------------------------
    print("2. reproject_raster() [to EPSG:4326]")
    print("-" * 70)

    reproject_times = []
    reprojected_rasters = {}

    for filename, (array, profile) in rasters.items():
        (rep_array, rep_profile), elapsed = benchmark(
            reproject_raster, array, profile, "EPSG:4326"
        )
        reproject_times.append(elapsed)
        reprojected_rasters[filename] = (rep_array, rep_profile)

        print(f"   {filename:30s}  {format_time(elapsed):>15s}  "
              f"{profile['crs']} -> EPSG:4326")

    if reproject_times:
        avg_reproject = np.mean(reproject_times)
        print(f"\n   Average reproject time: {format_time(avg_reproject)}")
    print()

    # -------------------------------------------------------------------------
    # 3. Benchmark resample_raster
    # -------------------------------------------------------------------------
    print("3. resample_raster() [to 50m resolution]")
    print("-" * 70)

    resample_times = []

    for filename, (array, profile) in rasters.items():
        (res_array, res_profile), elapsed = benchmark(
            resample_raster, array, profile, target_resolution=50.0
        )
        resample_times.append(elapsed)

        old_shape = array.shape
        new_shape = res_array.shape
        print(f"   {filename:30s}  {format_time(elapsed):>15s}  "
              f"{old_shape} -> {new_shape}")

    if resample_times:
        avg_resample = np.mean(resample_times)
        print(f"\n   Average resample time: {format_time(avg_resample)}")
    print()

    # -------------------------------------------------------------------------
    # 4. Benchmark align_rasters
    # -------------------------------------------------------------------------
    print("4. align_rasters() [aligning all layers to common grid]")
    print("-" * 70)

    (aligned, elapsed) = benchmark(align_rasters, rasters)
    print(f"   Aligned {len(rasters)} rasters: {format_time(elapsed)}")

    # Verify alignment
    reference_name = sorted(rasters.keys())[0]
    ref_profile = aligned[reference_name][1]
    print(f"   Reference grid: {reference_name}")
    print(f"   Reference shape: {aligned[reference_name][0].shape}")

    all_aligned = True
    for name, (arr, prof) in aligned.items():
        if arr.shape != aligned[reference_name][0].shape:
            print(f"   WARNING: {name} shape mismatch!")
            all_aligned = False

    if all_aligned:
        print(f"   All rasters aligned successfully")
    print()

    # -------------------------------------------------------------------------
    # 5. Benchmark normalization functions
    # -------------------------------------------------------------------------
    print("5. Normalization functions")
    print("-" * 70)

    # Use first raster for normalization tests
    test_filename = list(rasters.keys())[0]
    test_array, test_profile = rasters[test_filename]

    # Convert to float and add some nodata
    test_array_float = test_array.astype(np.float64)
    test_array_float[0:10, 0:10] = -9999.0

    # apply_nodata_mask
    (masked_array, elapsed) = benchmark(apply_nodata_mask, test_array_float, -9999.0)
    print(f"   apply_nodata_mask:    {format_time(elapsed):>15s}")

    # normalize_linear
    (norm_linear, elapsed) = benchmark(normalize_linear, masked_array, 0.0, 1.0)
    print(f"   normalize_linear:     {format_time(elapsed):>15s}")

    # normalize_sigmoid
    (norm_sigmoid, elapsed) = benchmark(normalize_sigmoid, masked_array, 0.5, 10.0)
    print(f"   normalize_sigmoid:    {format_time(elapsed):>15s}")

    # normalize_gaussian
    (norm_gaussian, elapsed) = benchmark(normalize_gaussian, masked_array, 0.5, 0.2)
    print(f"   normalize_gaussian:   {format_time(elapsed):>15s}")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_times = load_times + reproject_times + resample_times + [elapsed]
    max_time = max(all_times) if all_times else 0
    slow_operations = [t for t in all_times if t > 500]

    if slow_operations:
        print(f"WARNING: {len(slow_operations)} operation(s) exceeded 500ms threshold")
        print(f"Maximum operation time: {format_time(max_time)}")
        print()
        print("RECOMMENDATION: Consider optimization for operations > 500ms")
    else:
        print("All operations completed within acceptable time (< 500ms)")
        print(f"Maximum operation time: {format_time(max_time)}")

    print()
    print("=" * 70)

    return 1 if slow_operations else 0


if __name__ == "__main__":
    sys.exit(main())

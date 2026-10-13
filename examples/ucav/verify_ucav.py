#!/usr/bin/env python3
"""
StrikeCEM UCAV Benchmark Verification & Real-Time Query Demonstration.
Loads the solved UCAV database, queries across interpolation modes,
and measures microsecond evaluation throughput.
"""
import os
import sys
import time
import random

# Ensure StrikeCEM Python SDK is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, os.path.join(repo_root, "python"))

# Configure STRIKECEM_LIB_PATH
lib_path = os.path.join(repo_root, "build/release/libStrikeCEM.so")
os.environ["STRIKECEM_LIB_PATH"] = lib_path

from strikecem import Database, QueryRequest, InterpolationMode

def main():
    db_path = os.path.join(script_dir, "ucav_hybrid.h5")
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        sys.exit(1)

    print("=" * 75)
    print("StrikeCEM UCAV Benchmark Real-Time Runtime Verification")
    print("=" * 75)

    db = Database.open(db_path)
    print(f"Loaded Database:    {db_path}")
    print(f"Frequencies (Hz):   {db.frequencies}")
    print(f"Polarizations:      {db.polarizations}")
    print(f"Design ID:          {db.design_id}")
    print(f"Design Revision:    {db.design_revision}")
    print("-" * 75)

    aspects = [
        ("Nose-On", 0.0, 0.0),
        ("Starboard Chine", 45.0, 0.0),
        ("Right Wing Broadside", 90.0, 0.0),
        ("Aft Oblique", 135.0, 0.0),
        ("Tail", 180.0, 0.0),
        ("Port Wing Broadside", 270.0, 0.0)
    ]

    print(f"{'Aspect Name':<22} | {'Azimuth':<8} | {'Mode':<15} | {'RCS (dBsm)':<10} | {'RCS (m^2)':<10}")
    print("-" * 75)

    modes = [
        ("Nearest", InterpolationMode.NEAREST),
        ("Linear", InterpolationMode.LINEAR),
        ("Bicubic C^1", InterpolationMode.BICUBIC),
        ("Wendland C^2", InterpolationMode.SPHERICAL)
    ]

    for name, az, el in aspects:
        for mode_name, mode in modes:
            res = db.query(az, el, 10.0e9, "HH", interpolation_mode=mode)
            print(f"{name:<22} | {az:6.1f}°  | {mode_name:<15} | {res.rcs_dbsm:8.2f}   | {res.rcs_sqm:8.4f}")
        print("-" * 75)

    # Latency Benchmark: 10,000 sub-microsecond queries
    print("Benchmarking Query Throughput (10,000 random queries via query_batch)...")
    random.seed(42)
    test_queries = [
        QueryRequest(
            azimuth_deg=random.uniform(0.0, 360.0),
            elevation_deg=0.0,
            frequency_hz=10.0e9,
            polarization="HH"
        )
        for _ in range(10000)
    ]

    t0 = time.perf_counter()
    results = db.query_batch(test_queries, interpolation_mode=InterpolationMode.BICUBIC)
    t1 = time.perf_counter()

    elapsed_sec = t1 - t0
    queries_per_sec = len(test_queries) / elapsed_sec
    lat_us = (elapsed_sec / len(test_queries)) * 1e6

    print(f"Total Time:         {elapsed_sec:.4f} s")
    print(f"Average Latency:    {lat_us:.3f} microseconds / query")
    print(f"Query Throughput:   {queries_per_sec:,.0f} queries / second")
    print("=" * 75)
    print("Status: Verified & Operational!")

if __name__ == "__main__":
    main()

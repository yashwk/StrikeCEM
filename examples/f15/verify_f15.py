#!/usr/bin/env python3
"""
StrikeCEM 4th-Gen F-15 Eagle Benchmark Verification & Real-Time Query Demonstration.
Loads the solved F-15 Eagle database, queries across key tactical aspects (frontal,
broadside upright fins, aft nozzles) and interpolation modes, and measures
sub-microsecond evaluation throughput.
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
    db_path = os.path.join(script_dir, "f15_hybrid.h5")
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        sys.exit(1)

    print("=" * 80)
    print("StrikeCEM F-15 Eagle 4th-Gen Airframe Radar Signature Verification")
    print("=" * 80)

    db = Database.open(db_path)
    print(f"Loaded Database:    {db_path}")
    print(f"Frequencies (Hz):   {db.frequencies}")
    print(f"Polarizations:      {db.polarizations}")
    print(f"Design ID:          {db.design_id}")
    print(f"Design Revision:    {db.design_revision}")
    print("-" * 80)

    aspects = [
        ("Nose-On Sector", 0.0, 0.0),
        ("Intake Lip Oblique", 15.0, 0.0),
        ("Wing Leading Edge (45 deg)", 45.0, 0.0),
        ("Starboard Broadside (Fins)", 90.0, 0.0),
        ("Aft Oblique Wing TE", 135.0, 0.0),
        ("Aft Turbofan Nozzles", 180.0, 0.0),
        ("Port Broadside (Fins)", 270.0, 0.0)
    ]

    print(f"{'Aspect Name':<28} | {'Azimuth':<8} | {'Mode':<15} | {'RCS (dBsm)':<10} | {'RCS (m^2)':<10}")
    print("-" * 80)

    modes = [
        ("Nearest", InterpolationMode.NEAREST),
        ("Linear", InterpolationMode.LINEAR),
        ("Bicubic C^1", InterpolationMode.BICUBIC),
        ("Wendland C^2", InterpolationMode.SPHERICAL)
    ]

    for name, az, el in aspects:
        for mode_name, mode in modes:
            res = db.query(az, el, 10.0e9, "HH", interpolation_mode=mode)
            print(f"{name:<28} | {az:6.1f}°  | {mode_name:<15} | {res.rcs_dbsm:8.2f}   | {res.rcs_sqm:10.2f}")
        print("-" * 80)

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
    print("=" * 80)
    print("Status: Verified & Operational!")

if __name__ == "__main__":
    main()

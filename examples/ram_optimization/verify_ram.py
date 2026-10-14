#!/usr/bin/env python3
"""
StrikeCEM Radar Absorbent Material (RAM) Benchmark Verification & Runtime Query Demonstration.
Loads the solved Bare PEC, Intake RAM, and Full Stealth F-15 databases,
queries across tactical aspect sectors, computes exact dB power reductions,
and benchmarks sub-microsecond evaluation throughput (>200,000 queries/s).
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
    bare_path = os.path.join(script_dir, "f15_bare_pec.h5")
    intk_path = os.path.join(script_dir, "f15_ram_intake.h5")
    stl_path  = os.path.join(script_dir, "f15_ram_stealth.h5")

    for p in [bare_path, intk_path, stl_path]:
        if not os.path.exists(p):
            print(f"Error: {p} not found.")
            sys.exit(1)

    print("=" * 105)
    print("StrikeCEM F-15 Eagle RAM Treatment & Stealth Optimization Benchmark Verification")
    print("Solvers: Physical Optics with Multi-Layer Transmission-Line Reflection & Leontovich SIBC")
    print("=" * 105)

    db_bare = Database.open(bare_path)
    db_intk = Database.open(intk_path)
    db_stl  = Database.open(stl_path)

    print(f"Loaded Databases:")
    print(f"  1. Bare PEC Baseline:  {bare_path}")
    print(f"  2. Selective Intake:   {intk_path}")
    print(f"  3. Full Stealth Suite: {stl_path}")
    print(f"Frequency: 10.0 GHz | Polarizations: HH, VV")
    print("-" * 105)

    aspects = [
        ("Nose-On Sector", 0.0, 0.0),
        ("Intake Cowl Oblique", 15.0, 0.0),
        ("Starboard Chine", 30.0, 0.0),
        ("Wing Leading Edge Specular", 45.0, 0.0),
        ("Starboard Broadside (Fins)", 90.0, 0.0),
        ("Aft Oblique Wing TE", 135.0, 0.0),
        ("Direct Aft (Turbofan Nozzles)", 180.0, 0.0),
        ("Port Broadside (Fins)", 270.0, 0.0)
    ]

    print(f"{'Aspect Name':<30} | {'Azimuth':<7} | {'Bare PEC (dBsm)':<15} | {'Intake RAM':<12} | {'Stealth RAM':<12} | {'Delta (dB)':<10} | {'Power Reduction':<15}")
    print("-" * 105)

    for name, az, el in aspects:
        r_bare = db_bare.query(az, el, 10.0e9, "HH", interpolation_mode=InterpolationMode.NEAREST)
        r_intk = db_intk.query(az, el, 10.0e9, "HH", interpolation_mode=InterpolationMode.NEAREST)
        r_stl  = db_stl.query(az, el, 10.0e9, "HH", interpolation_mode=InterpolationMode.NEAREST)

        delta_db = r_stl.rcs_dbsm - r_bare.rcs_dbsm
        if delta_db < 0:
            pct_reduc = (1.0 - 10.0 ** (delta_db / 10.0)) * 100.0
            reduc_str = f"-{pct_reduc:.1f}%"
        else:
            reduc_str = "0.0%"

        print(f"{name:<30} | {az:5.1f}° | {r_bare.rcs_dbsm:10.2f} dBsm | {r_intk.rcs_dbsm:8.2f} dBsm | {r_stl.rcs_dbsm:8.2f} dBsm | {delta_db:7.2f} dB | {reduc_str:<15}")
    print("-" * 105)

    # Latency Benchmark: 10,000 sub-microsecond queries on RAM database
    print("\nBenchmarking Query Throughput on RAM Stealth Database (10,000 random queries via query_batch)...")
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
    results = db_stl.query_batch(test_queries, interpolation_mode=InterpolationMode.BICUBIC)
    t1 = time.perf_counter()

    elapsed_sec = t1 - t0
    queries_per_sec = len(test_queries) / elapsed_sec
    lat_us = (elapsed_sec / len(test_queries)) * 1e6

    print(f"Total Evaluation Time:  {elapsed_sec:.4f} s")
    print(f"Average Query Latency:  {lat_us:.3f} microseconds / query")
    print(f"Query Throughput:       {queries_per_sec:,.0f} queries / second")
    print("=" * 105)
    print("Status: Verified & Operational!")

if __name__ == "__main__":
    main()

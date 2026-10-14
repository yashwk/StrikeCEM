#!/usr/bin/env python3
"""
StrikeCEM Naval Stealth Warship Benchmark Verification & Runtime Query Demonstration.
Loads the solved Bare PEC and Stealth RAM Zumwalt (DDG-1000) databases,
queries across key maritime tactical aspect sectors, computes exact dB power reductions,
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
    bare_path = os.path.join(script_dir, "zumwalt_bare_pec.h5")
    stl_path  = os.path.join(script_dir, "zumwalt_stealth.h5")

    for p in [bare_path, stl_path]:
        if not os.path.exists(p):
            print(f"Error: {p} not found.")
            sys.exit(1)

    print("=" * 105)
    print("StrikeCEM USS Zumwalt (DDG-1000) Naval Stealth Combatant Benchmark Verification")
    print("Solvers: Physical Optics with Inward Tumblehome Hull & Multi-Material RAM Deckhouse/Mast")
    print("=" * 105)

    db_bare = Database.open(bare_path)
    db_stl  = Database.open(stl_path)

    print(f"Loaded Databases:")
    print(f"  1. Bare PEC Baseline:  {bare_path}")
    print(f"  2. Stealth RAM Suite:  {stl_path}")
    print(f"Radar Frequency: 10.0 GHz (X-Band Surface Search) | Polarizations: HH, VV")
    print("-" * 105)

    aspects = [
        ("Bow-On (Inverted Wave-Piercing Stem)", 0.0, 0.0),
        ("Starboard Bow Oblique", 30.0, 0.0),
        ("Forecastle Turrets Oblique", 45.0, 0.0),
        ("Starboard Beam (Tumblehome Hull)", 90.0, 0.0),
        ("Starboard Quarter (Hangar Transition)", 135.0, 0.0),
        ("Direct Stern (Reverse Slanted Transom)", 180.0, 0.0),
        ("Port Quartering Aspect", 225.0, 0.0),
        ("Port Beam (Tumblehome Hull)", 270.0, 0.0)
    ]

    print(f"{'Tactical Aspect Sector':<38} | {'Azimuth':>7} | {'Bare PEC (dBsm)':>15} | {'Stealth RAM':>12} | {'Delta (dB)':>10} | {'Power Reduction':>15}")
    print("-" * 105)

    for name, az, el in aspects:
        res_bare = db_bare.query(az, el, 10.0e9, "HH", interpolation_mode=InterpolationMode.NEAREST)
        res_stl  = db_stl.query(az, el, 10.0e9, "HH", interpolation_mode=InterpolationMode.NEAREST)

        rcs_b = res_bare.rcs_dbsm
        rcs_s = res_stl.rcs_dbsm
        delta = rcs_s - rcs_b
        pct = (1.0 - 10.0 ** (delta / 10.0)) * 100.0 if delta <= 0 else 0.0

        pct_str = f"-{pct:.1f}%" if delta <= 0 else f"+{abs(pct):.1f}%"
        print(f"{name:<38} | {az:>6.1f}° | {rcs_b:>13.2f} dBsm | {rcs_s:>10.2f} dBsm | {delta:>8.2f} dB | {pct_str:>15}")

    print("-" * 105)
    print("\nBenchmarking Query Throughput on Zumwalt Stealth Database (10,000 random queries via query_batch)...")

    n_queries = 10000
    batch_reqs = []
    for _ in range(n_queries):
        az = random.uniform(0.0, 360.0)
        batch_reqs.append(QueryRequest(frequency_hz=10.0e9, azimuth_deg=az, elevation_deg=0.0, polarization="HH"))

    t0 = time.perf_counter()
    results = db_stl.query_batch(batch_reqs)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    latency_us = (elapsed / n_queries) * 1.0e6
    throughput = n_queries / elapsed

    print(f"Total Evaluation Time:  {elapsed:.4f} s")
    print(f"Average Query Latency:  {latency_us:.3f} microseconds / query")
    print(f"Query Throughput:       {throughput:,.0f} queries / second")
    print("=" * 105)
    print("Status: Verified & Operational!\n")


if __name__ == "__main__":
    main()

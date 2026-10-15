#!/usr/bin/env python3
"""
StrikeCEM Next-Gen Stealth Flying-Wing Bomber (B-21 Raider) Benchmark Verification.
Loads the solved Bare PEC and Stealth RAM B-21 databases,
queries across key tactical aspect sectors, computes exact dB power reductions,
and benchmarks sub-microsecond evaluation throughput (>250,000 queries/s).
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
    bare_path = os.path.join(script_dir, "b21_bare_pec.h5")
    stl_path  = os.path.join(script_dir, "b21_stealth.h5")

    for p in [bare_path, stl_path]:
        if not os.path.exists(p):
            print(f"Error: {p} not found.")
            sys.exit(1)

    print("=" * 110)
    print("StrikeCEM Next-Gen Stealth Flying-Wing Bomber (B-21 Raider) Benchmark Verification")
    print("Solvers: CUDA Physical Optics + PTD Edge Fringe Diffraction + Fock Curvature + RAM Coatings")
    print("=" * 110)

    db_bare = Database.open(bare_path)
    db_stl  = Database.open(stl_path)

    print(f"Loaded Databases:")
    print(f"  1. Bare PEC Baseline:  {bare_path}")
    print(f"  2. Stealth RAM Suite:  {stl_path}")
    print(f"Radar Frequency: 10.0 GHz (X-Band Air-to-Air / Ground Fire Control) | Polarizations: HH, VV")
    print("-" * 110)

    aspects = [
        ("Nose-On (Apex & ITO Canopy)", 0.0, 5.0),
        ("Starboard Nose Oblique", 15.0, 5.0),
        ("Starboard Leading-Edge Flash", 35.0, 5.0),
        ("Starboard Leading-Edge Normal", 55.0, 5.0),
        ("Starboard Wingtip Profile", 90.0, 5.0),
        ("Starboard Trailing-Edge Chevron", 125.0, 5.0),
        ("Direct Stern (Beavertail Notch)", 180.0, 5.0),
        ("Port Trailing-Edge Chevron", 235.0, 5.0),
        ("Port Leading-Edge Normal", 305.0, 5.0),
        ("Port Leading-Edge Flash", 325.0, 5.0)
    ]

    print(f"{'Tactical Aspect Sector':<36} | {'Azimuth':>7} | {'Bare PEC (dBsm)':>15} | {'Stealth RAM':>12} | {'Delta (dB)':>10} | {'Power Reduction':>15}")
    print("-" * 110)

    for name, az, el in aspects:
        res_bare = db_bare.query(az, el, 10.0e9, "HH", interpolation_mode=InterpolationMode.NEAREST)
        res_stl  = db_stl.query(az, el, 10.0e9, "HH", interpolation_mode=InterpolationMode.NEAREST)

        rcs_b = res_bare.rcs_dbsm
        rcs_s = res_stl.rcs_dbsm
        delta = rcs_s - rcs_b
        pct = (1.0 - 10.0 ** (delta / 10.0)) * 100.0 if delta <= 0 else 0.0

        print(f"{name:<36} | {az:>6.1f}° | {rcs_b:>15.2f} | {rcs_s:>12.2f} | {delta:>+9.2f} dB | {pct:>14.2f}%")

    print("-" * 110)

    # Throughput benchmark
    N = 100000
    print(f"Running in-memory spatial query throughput benchmark ({N:,} queries)...")
    start = time.perf_counter()
    for _ in range(N):
        az = random.uniform(0.0, 360.0)
        db_stl.query(az, 0.0, 10.0e9, "HH", interpolation_mode=InterpolationMode.NEAREST)
    elapsed = time.perf_counter() - start
    qps = N / elapsed
    us_per_q = (elapsed / N) * 1e6
    print(f"Benchmark Result: {qps:,.0f} queries/sec ({us_per_q:.2f} μs/query)")
    print("=" * 110)


if __name__ == "__main__":
    main()

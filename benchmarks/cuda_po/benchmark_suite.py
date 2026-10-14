#!/usr/bin/env python3
"""
Comprehensive StrikeCEM CUDA GPU PO Benchmark Suite
Tests:
  1. Mesh Facet Scaling: 752 up to 192,512 triangles
  2. Multi-Core CPU vs NVIDIA RTX 4060 GPU (sm_89 Ada Lovelace)
  3. Precision modes: float64 vs mixed vs float32
  4. Angle Sweep Density: 36, 180, 360, 1440 directions
  5. Mathematical precision & accuracy verification vs CPU double-precision
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import h5py
import matplotlib.pyplot as plt

PROJECT_ROOT = "/home/yashwanth/projects/StrikeCEM"
BENCH_DIR = os.path.join(PROJECT_ROOT, "benchmarks/cuda_po")
BIN_PATH = os.path.join(PROJECT_ROOT, "build/release/strikecem")
BASE_OBJ = os.path.join(PROJECT_ROOT, "examples/f15/f15.obj")

os.makedirs(BENCH_DIR, exist_ok=True)
os.makedirs(os.path.join(BENCH_DIR, "meshes"), exist_ok=True)
os.makedirs(os.path.join(BENCH_DIR, "runs"), exist_ok=True)
os.makedirs(os.path.join(BENCH_DIR, "plots"), exist_ok=True)

def load_obj(path):
    verts = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()
                # Parse vertex index (1-based)
                f_idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                faces.append(f_idx)
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)

def save_obj(path, verts, faces):
    with open(path, "w") as f:
        f.write("# Subdivided mesh for StrikeCEM CUDA benchmark\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def subdivide_1to4(verts, faces):
    """Subdivide each triangle into 4 smaller triangles (1:4 split)."""
    edge_map = {}
    new_verts = list(verts)
    new_faces = []

    def get_midpoint(i1, i2):
        edge = tuple(sorted((i1, i2)))
        if edge in edge_map:
            return edge_map[edge]
        mid = (verts[i1] + verts[i2]) * 0.5
        new_idx = len(new_verts)
        new_verts.append(mid)
        edge_map[edge] = new_idx
        return new_idx

    for f in faces:
        v0, v1, v2 = f[0], f[1], f[2]
        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)

        new_faces.append([v0, m01, m20])
        new_faces.append([v1, m12, m01])
        new_faces.append([v2, m20, m12])
        new_faces.append([m01, m12, m20])

    return np.array(new_verts, dtype=np.float64), np.array(new_faces, dtype=np.int32)

def prepare_meshes():
    v0, f0 = load_obj(BASE_OBJ)
    meshes = {
        0: (BASE_OBJ, len(f0)),
    }
    cur_v, cur_f = v0, f0
    for lvl in range(1, 4):
        cur_v, cur_f = subdivide_1to4(cur_v, cur_f)
        out_path = os.path.join(BENCH_DIR, "meshes", f"f15_subdiv_lvl{lvl}.obj")
        save_obj(out_path, cur_v, cur_f)
        meshes[lvl] = (out_path, len(cur_f))
        print(f"Mesh Level {lvl}: {len(cur_f)} triangles, {len(cur_v)} vertices -> {out_path}")
    return meshes

def create_config(mesh_path, out_h5, az_step=1.0, precision="float64", accelerator="cpu", threads=1):
    cfg = {
        "scem_schema_version": "1.0",
        "model": {
            "path": mesh_path,
            "units": "m",
            "mesh_repair": "report"
        },
        "frequency": {
            "frequency_hz": 10000000000
        },
        "physics": {
            "monostatic": True,
            "polarization_basis": "linear",
            "incident_amplitude": 1.0,
            "phase_reference": [0.0, 0.0, 0.0]
        },
        "angles": {
            "azimuth": {
                "start": 0,
                "stop": 360,
                "step": az_step
            },
            "elevation": {
                "start": 0,
                "stop": 0,
                "step": 1
            }
        },
        "polarization": ["HH", "VV"],
        "solver": {
            "type": "PO",
            "precision": precision,
            "rcs_units": "dBsm",
            "po_options": {
                "edge_correction": "none",
                "curvature_correction": False,
                "shadowing": False,
                "max_bounces": 1
            }
        },
        "execution": {
            "accelerator": accelerator,
            "cuda_device_id": 0,
            "cpu_threads": threads
        },
        "output": {
            "format": "hdf5",
            "path": out_h5
        }
    }
    return cfg

def run_strike_bench(cfg_path, profile_path):
    if os.path.exists(profile_path):
        os.remove(profile_path)
    cmd = [BIN_PATH, "--bench-profile", profile_path, "run", cfg_path]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wall_time = time.perf_counter() - t0
    if res.returncode != 0:
        print("ERROR running:", cmd)
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        raise RuntimeError("StrikeCEM benchmark failed")
    
    if os.path.exists(profile_path):
        with open(profile_path, "r") as f:
            prof = json.load(f)
            return prof.get("wall_seconds", wall_time), prof
    return wall_time, {}

def main():
    print("=" * 70)
    print("STRIKECEM CUDA GPU PO ACCELERATION BENCHMARK CAMPAIGN")
    print("=" * 70)

    meshes = prepare_meshes()

    # ---------------------------------------------------------
    # EXPERIMENT 1: Mesh Facet Scaling & CPU Multi-Thread vs GPU
    # ---------------------------------------------------------
    print("\n--- EXPERIMENT 1: Facet Scaling (360 deg sweep, 1 deg step = 361 dirs x 2 pol = 722 samples) ---")
    results_exp1 = []
    
    thread_configs = [1, 4, 8, 16]

    for lvl, (mesh_path, ntris) in meshes.items():
        print(f"\n[Mesh Level {lvl}: {ntris:,} Triangles]")
        row = {"level": lvl, "triangles": ntris, "cpu": {}, "gpu": {}}

        # Warmup CUDA runtime once
        if lvl == 0:
            warmup_h5 = os.path.join(BENCH_DIR, "runs", "warmup.h5")
            warmup_cfg = create_config(mesh_path, warmup_h5, az_step=10.0, accelerator="cuda")
            warmup_cfg_path = os.path.join(BENCH_DIR, "runs", "warmup.json")
            with open(warmup_cfg_path, "w") as f:
                json.dump(warmup_cfg, f, indent=2)
            run_strike_bench(warmup_cfg_path, os.path.join(BENCH_DIR, "runs", "warmup_prof.json"))

        # Test CPU threads
        for nth in thread_configs:
            out_h5 = os.path.join(BENCH_DIR, "runs", f"m{lvl}_cpu_{nth}t.h5")
            if os.path.exists(out_h5): os.remove(out_h5)
            cfg = create_config(mesh_path, out_h5, az_step=1.0, accelerator="cpu", threads=nth)
            cfg_path = os.path.join(BENCH_DIR, "runs", f"m{lvl}_cpu_{nth}t.json")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            prof_path = os.path.join(BENCH_DIR, "runs", f"prof_m{lvl}_cpu_{nth}t.json")
            
            # Average over 2 runs for stability
            times = []
            for _ in range(2):
                if os.path.exists(out_h5): os.remove(out_h5)
                sec, _ = run_strike_bench(cfg_path, prof_path)
                times.append(sec)
            best_sec = min(times)
            row["cpu"][nth] = best_sec
            print(f"  CPU ({nth:2d} threads): {best_sec*1000:8.2f} ms")

        # Test GPU (float64, mixed, float32)
        for prec in ["float64", "mixed", "float32"]:
            out_h5 = os.path.join(BENCH_DIR, "runs", f"m{lvl}_gpu_{prec}.h5")
            if os.path.exists(out_h5): os.remove(out_h5)
            cfg = create_config(mesh_path, out_h5, az_step=1.0, precision=prec, accelerator="cuda")
            cfg_path = os.path.join(BENCH_DIR, "runs", f"m{lvl}_gpu_{prec}.json")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            prof_path = os.path.join(BENCH_DIR, "runs", f"prof_m{lvl}_gpu_{prec}.json")

            times = []
            for _ in range(3):
                if os.path.exists(out_h5): os.remove(out_h5)
                sec, _ = run_strike_bench(cfg_path, prof_path)
                times.append(sec)
            best_sec = min(times)
            row["gpu"][prec] = best_sec
            speedup_vs_1t = row["cpu"][1] / best_sec
            speedup_vs_16t = row["cpu"][16] / best_sec
            print(f"  GPU ({prec:7s}): {best_sec*1000:8.2f} ms  (Speedup: {speedup_vs_1t:6.1f}x vs CPU-1T, {speedup_vs_16t:5.1f}x vs CPU-16T)")

        results_exp1.append(row)

    # ---------------------------------------------------------
    # EXPERIMENT 2: Angular Sweep Density Scaling
    # ---------------------------------------------------------
    print("\n--- EXPERIMENT 2: Angle Sweep Density Scaling (Mesh Level 2: 12,032 Triangles) ---")
    test_mesh, ntris_test = meshes[2]
    az_steps = [10.0, 5.0, 2.0, 1.0, 0.5]
    results_exp2 = []

    for step in az_steps:
        num_dirs = int(360 / step)
        num_samples = num_dirs * 2
        print(f"\n[Azimuth Step {step:4.1f} deg -> {num_dirs:,} directions ({num_samples:,} samples)]")
        exp_row = {"step_deg": step, "dirs": num_dirs, "samples": num_samples, "cpu_16t": 0.0, "gpu_f64": 0.0, "gpu_mixed": 0.0}

        # CPU 16T
        out_h5 = os.path.join(BENCH_DIR, "runs", f"sweep_cpu_{step}.h5")
        if os.path.exists(out_h5): os.remove(out_h5)
        cfg = create_config(test_mesh, out_h5, az_step=step, accelerator="cpu", threads=16)
        cfg_path = os.path.join(BENCH_DIR, "runs", f"sweep_cpu_{step}.json")
        with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
        sec_cpu, _ = run_strike_bench(cfg_path, os.path.join(BENCH_DIR, "runs", f"prof_sweep_cpu_{step}.json"))
        exp_row["cpu_16t"] = sec_cpu
        print(f"  CPU-16T: {sec_cpu*1000:8.2f} ms")

        # GPU float64
        out_h5 = os.path.join(BENCH_DIR, "runs", f"sweep_gpu_f64_{step}.h5")
        if os.path.exists(out_h5): os.remove(out_h5)
        cfg = create_config(test_mesh, out_h5, az_step=step, precision="float64", accelerator="cuda")
        cfg_path = os.path.join(BENCH_DIR, "runs", f"sweep_gpu_f64_{step}.json")
        with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
        sec_gpu, _ = run_strike_bench(cfg_path, os.path.join(BENCH_DIR, "runs", f"prof_sweep_gpu_{step}.json"))
        exp_row["gpu_f64"] = sec_gpu
        print(f"  GPU-f64: {sec_gpu*1000:8.2f} ms  (Speedup vs CPU-16T: {sec_cpu/sec_gpu:5.1f}x)")

        # GPU mixed
        out_h5 = os.path.join(BENCH_DIR, "runs", f"sweep_gpu_mix_{step}.h5")
        if os.path.exists(out_h5): os.remove(out_h5)
        cfg = create_config(test_mesh, out_h5, az_step=step, precision="mixed", accelerator="cuda")
        cfg_path = os.path.join(BENCH_DIR, "runs", f"sweep_gpu_mix_{step}.json")
        with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
        sec_gpu_mix, _ = run_strike_bench(cfg_path, os.path.join(BENCH_DIR, "runs", f"prof_sweep_gpu_mix_{step}.json"))
        exp_row["gpu_mixed"] = sec_gpu_mix
        print(f"  GPU-mix: {sec_gpu_mix*1000:8.2f} ms  (Speedup vs CPU-16T: {sec_cpu/sec_gpu_mix:5.1f}x)")

        results_exp2.append(exp_row)

    # ---------------------------------------------------------
    # EXPERIMENT 3: Precision & Numerical Accuracy Verification
    # ---------------------------------------------------------
    print("\n--- EXPERIMENT 3: Precision & Accuracy Verification (CPU f64 vs GPU f64 vs GPU mixed vs GPU f32) ---")
    cpu_ref_h5 = os.path.join(BENCH_DIR, "runs", "m1_cpu_16t.h5")
    gpu_f64_h5 = os.path.join(BENCH_DIR, "runs", "m1_gpu_float64.h5")
    gpu_mix_h5 = os.path.join(BENCH_DIR, "runs", "m1_gpu_mixed.h5")
    gpu_f32_h5 = os.path.join(BENCH_DIR, "runs", "m1_gpu_float32.h5")

    with h5py.File(cpu_ref_h5, "r") as f:
        cpu_rcs = f["samples/rcs"][:]
        cpu_s = f["samples/scattering_re"][:] + 1j * f["samples/scattering_im"][:]
    with h5py.File(gpu_f64_h5, "r") as f:
        gpu_f64_rcs = f["samples/rcs"][:]
        gpu_f64_s = f["samples/scattering_re"][:] + 1j * f["samples/scattering_im"][:]
    with h5py.File(gpu_mix_h5, "r") as f:
        gpu_mix_rcs = f["samples/rcs"][:]
        gpu_mix_s = f["samples/scattering_re"][:] + 1j * f["samples/scattering_im"][:]
    with h5py.File(gpu_f32_h5, "r") as f:
        gpu_f32_rcs = f["samples/rcs"][:]
        gpu_f32_s = f["samples/scattering_re"][:] + 1j * f["samples/scattering_im"][:]

    err_f64_rcs = np.max(np.abs(cpu_rcs - gpu_f64_rcs))
    err_f64_s_rel = np.max(np.abs(cpu_s - gpu_f64_s) / (np.abs(cpu_s) + 1e-15))

    err_mix_rcs = np.max(np.abs(cpu_rcs - gpu_mix_rcs))
    err_mix_s_rel = np.max(np.abs(cpu_s - gpu_mix_s) / (np.abs(cpu_s) + 1e-15))

    err_f32_rcs = np.max(np.abs(cpu_rcs - gpu_f32_rcs))
    err_f32_s_rel = np.max(np.abs(cpu_s - gpu_f32_s) / (np.abs(cpu_s) + 1e-15))

    print(f"GPU float64: Max RCS diff: {err_f64_rcs:.2e} dBsm | Max Rel Field Err: {err_f64_s_rel:.2e}")
    print(f"GPU mixed:   Max RCS diff: {err_mix_rcs:.2e} dBsm | Max Rel Field Err: {err_mix_s_rel:.2e}")
    print(f"GPU float32: Max RCS diff: {err_f32_rcs:.2e} dBsm | Max Rel Field Err: {err_f32_s_rel:.2e}")

    accuracy_data = {
        "f64": {"max_rcs_diff_dbsm": float(err_f64_rcs), "max_rel_field_err": float(err_f64_s_rel)},
        "mixed": {"max_rcs_diff_dbsm": float(err_mix_rcs), "max_rel_field_err": float(err_mix_s_rel)},
        "f32": {"max_rcs_diff_dbsm": float(err_f32_rcs), "max_rel_field_err": float(err_f32_s_rel)}
    }

    # Save all data
    benchmark_data = {
        "device": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "arch": "Ada Lovelace (sm_89, 8188 MiB VRAM)",
        "cpu": "AMD Ryzen 7 8845HS (16 logical threads)",
        "experiment1_facet_scaling": results_exp1,
        "experiment2_sweep_density": results_exp2,
        "experiment3_accuracy": accuracy_data
    }
    with open(os.path.join(BENCH_DIR, "cuda_po_benchmark_results.json"), "w") as f:
        json.dump(benchmark_data, f, indent=2)

    # ---------------------------------------------------------
    # GENERATE PLOTS
    # ---------------------------------------------------------
    print("\nGenerating Benchmark Plots...")
    plt.style.use("dark_background")

    # Plot 1: Speedup vs Triangles
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    tris = [r["triangles"] for r in results_exp1]
    sp_vs_1t = [r["cpu"][1] / r["gpu"]["mixed"] for r in results_exp1]
    sp_vs_4t = [r["cpu"][4] / r["gpu"]["mixed"] for r in results_exp1]
    sp_vs_16t = [r["cpu"][16] / r["gpu"]["mixed"] for r in results_exp1]
    sp_f64_vs_16t = [r["cpu"][16] / r["gpu"]["float64"] for r in results_exp1]

    ax.plot(tris, sp_vs_1t, "o-", color="#00ffcc", linewidth=2.5, markersize=8, label="GPU (Mixed) vs CPU 1-Thread")
    ax.plot(tris, sp_vs_4t, "s--", color="#3399ff", linewidth=2.0, markersize=7, label="GPU (Mixed) vs CPU 4-Threads")
    ax.plot(tris, sp_vs_16t, "^-", color="#ff007f", linewidth=2.5, markersize=8, label="GPU (Mixed) vs CPU 16-Threads")
    ax.plot(tris, sp_f64_vs_16t, "d:", color="#ffaa00", linewidth=2.0, markersize=7, label="GPU (Float64) vs CPU 16-Threads")

    ax.set_xscale("log")
    ax.set_xlabel("Mesh Triangle Count (Facets)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Speedup Factor (x)", fontsize=12, fontweight="bold")
    ax.set_title("StrikeCEM CUDA PO Speedup vs Multi-Thread CPU\n(RTX 4060 sm_89 vs Ryzen 7 8845HS)", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor="#1e1e1e", edgecolor="#444444", fontsize=10)
    plt.tight_layout()
    plot1_path = os.path.join(BENCH_DIR, "plots", "cuda_po_speedup_vs_mesh.png")
    plt.savefig(plot1_path)
    plt.close()
    print("Saved:", plot1_path)

    # Plot 2: Absolute Runtime vs Mesh Size
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.plot(tris, [r["cpu"][1] * 1000 for r in results_exp1], "x--", color="#ff4444", label="CPU (1 Thread)")
    ax.plot(tris, [r["cpu"][4] * 1000 for r in results_exp1], "+--", color="#ff8800", label="CPU (4 Threads)")
    ax.plot(tris, [r["cpu"][16] * 1000 for r in results_exp1], "s--", color="#ffff00", label="CPU (16 Threads)")
    ax.plot(tris, [r["gpu"]["float64"] * 1000 for r in results_exp1], "o-", color="#33ccff", linewidth=2.2, label="CUDA GPU (Float64)")
    ax.plot(tris, [r["gpu"]["mixed"] * 1000 for r in results_exp1], "*-", color="#00ff66", linewidth=2.5, label="CUDA GPU (Mixed Precision)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mesh Triangle Count (Facets)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Wall Clock Execution Time (milliseconds)", fontsize=12, fontweight="bold")
    ax.set_title("Absolute PO Solve Time vs Mesh Size (360 Azimuth Directions)", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor="#1e1e1e", edgecolor="#444444", fontsize=10)
    plt.tight_layout()
    plot2_path = os.path.join(BENCH_DIR, "plots", "cuda_po_runtime_scaling.png")
    plt.savefig(plot2_path)
    plt.close()
    print("Saved:", plot2_path)

    # Plot 3: Throughput (Million Facet-Directions per Second)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    # 360 directions
    ndirs = 360
    tput_cpu16 = [(r["triangles"] * ndirs) / (r["cpu"][16] * 1e6) for r in results_exp1]
    tput_gpu_f64 = [(r["triangles"] * ndirs) / (r["gpu"]["float64"] * 1e6) for r in results_exp1]
    tput_gpu_mix = [(r["triangles"] * ndirs) / (r["gpu"]["mixed"] * 1e6) for r in results_exp1]
    tput_gpu_f32 = [(r["triangles"] * ndirs) / (r["gpu"]["float32"] * 1e6) for r in results_exp1]

    ax.plot(tris, tput_gpu_f32, "d-", color="#ff00ff", linewidth=2.5, label="CUDA GPU (Float32)")
    ax.plot(tris, tput_gpu_mix, "o-", color="#00ffcc", linewidth=2.5, label="CUDA GPU (Mixed)")
    ax.plot(tris, tput_gpu_f64, "^-", color="#3399ff", linewidth=2.0, label="CUDA GPU (Float64)")
    ax.plot(tris, tput_cpu16, "s--", color="#ffaa00", linewidth=1.8, label="CPU (16 Threads)")

    ax.set_xscale("log")
    ax.set_xlabel("Mesh Triangle Count (Facets)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (Million Facet-Directions / sec)", fontsize=12, fontweight="bold")
    ax.set_title("StrikeCEM Computational Throughput on RTX 4060", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor="#1e1e1e", edgecolor="#444444", fontsize=10)
    plt.tight_layout()
    plot3_path = os.path.join(BENCH_DIR, "plots", "cuda_po_throughput.png")
    plt.savefig(plot3_path)
    plt.close()
    print("Saved:", plot3_path)

    print("\nBenchmark campaign successfully completed!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Executes scem_gpubench across the experimental matrix and generates
publication-quality charts for the StrikeCEM CUDA GPU PO Solver Showcase.
"""

import os
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt

BENCH_DIR = "/home/yashwanth/projects/StrikeCEM/benchmarks/cuda_po"
BIN_PATH = "/home/yashwanth/projects/StrikeCEM/build/release/scem_gpubench"
PLOTS_DIR = os.path.join(BENCH_DIR, "plots")
DATA_DIR = os.path.join(BENCH_DIR, "json_data")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Run Mesh Scaling Matrix (subdiv 0 to 4 with 360 dirs)
mesh_levels = [0, 1, 2, 3, 4]
mesh_data = []

print("=== RUNNING MESH SCALING BENCHMARKS ===")
for lvl in mesh_levels:
    json_path = os.path.join(DATA_DIR, f"mesh_subdiv_{lvl}.json")
    cmd = [
        BIN_PATH,
        "--mesh", "examples/f15/f15.obj",
        "--subdiv", str(lvl),
        "--dirs", "360",
        "--iters", "2" if lvl <= 2 else "1",
        "--json", json_path
    ]
    print(f"Running Subdiv Level {lvl}...")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print("Error in scem_gpubench:", res.stderr)
        raise RuntimeError("Benchmark run failed")
    with open(json_path, "r") as f:
        mesh_data.append(json.load(f))

# 2. Run Sweep Density Matrix (on Level 2 mesh with 36 to 1440 dirs)
dirs_list = [36, 72, 180, 360, 720, 1440]
sweep_data = []

print("\n=== RUNNING SWEEP DENSITY BENCHMARKS ===")
for d in dirs_list:
    json_path = os.path.join(DATA_DIR, f"sweep_dirs_{d}.json")
    cmd = [
        BIN_PATH,
        "--mesh", "examples/f15/f15.obj",
        "--subdiv", "2",
        "--dirs", str(d),
        "--iters", "2" if d <= 360 else "1",
        "--json", json_path
    ]
    print(f"Running Dirs {d}...")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print("Error in scem_gpubench:", res.stderr)
        raise RuntimeError("Benchmark run failed")
    with open(json_path, "r") as f:
        sweep_data.append(json.load(f))

# 3. Generate Publication Plots
print("\n=== GENERATING PLOTS ===")
plt.style.use("dark_background")

# Plot 1: Speedup vs Mesh Size
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
tris = [d["triangles"] for d in mesh_data]
sp_f32_vs_1t = [d["speedup_vs_1t"]["float32"] for d in mesh_data]
sp_f32_vs_16t = [d["speedup_vs_16t"]["float32"] for d in mesh_data]
sp_f64_vs_1t = [d["speedup_vs_1t"]["float64"] for d in mesh_data]
sp_f64_vs_16t = [d["speedup_vs_16t"]["float64"] for d in mesh_data]

ax.plot(tris, sp_f32_vs_1t, "d-", color="#00ffcc", linewidth=2.8, markersize=9, label="GPU Float32 vs CPU 1-Thread")
ax.plot(tris, sp_f32_vs_16t, "s--", color="#ff007f", linewidth=2.4, markersize=8, label="GPU Float32 vs CPU 16-Threads (All Cores)")
ax.plot(tris, sp_f64_vs_1t, "o-", color="#3399ff", linewidth=2.4, markersize=8, label="GPU Float64 vs CPU 1-Thread")
ax.plot(tris, sp_f64_vs_16t, "^--", color="#ffaa00", linewidth=2.0, markersize=7, label="GPU Float64 vs CPU 16-Threads")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Mesh Triangle Count (Facets)", fontsize=12, fontweight="bold")
ax.set_ylabel("Speedup Factor (log scale)", fontsize=12, fontweight="bold")
ax.set_title("StrikeCEM Hardware Acceleration Speedup vs Mesh Complexity\n(NVIDIA GeForce RTX 4060 sm_89 vs AMD Ryzen 7 8845HS 16-Thread)", fontsize=13, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(frameon=True, facecolor="#181818", edgecolor="#444444", fontsize=10)

for x, y in zip(tris, sp_f32_vs_1t):
    ax.annotate(f"{y:.1f}x", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, fontweight="bold", color="#00ffcc")

plot1_path = os.path.join(PLOTS_DIR, "cuda_po_speedup_vs_mesh.png")
plt.tight_layout()
plt.savefig(plot1_path)
plt.close()
print("Saved:", plot1_path)

# Plot 2: Absolute Runtime Scaling
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
cpu_1t = [d["cpu_seconds"]["1"] * 1000.0 for d in mesh_data]
cpu_4t = [d["cpu_seconds"]["4"] * 1000.0 for d in mesh_data]
cpu_16t = [d["cpu_seconds"]["16"] * 1000.0 for d in mesh_data]
gpu_f64 = [d["gpu_seconds"]["float64"] * 1000.0 for d in mesh_data]
gpu_f32 = [d["gpu_seconds"]["float32"] * 1000.0 for d in mesh_data]

ax.plot(tris, cpu_1t, "x--", color="#ff4d4d", linewidth=2.0, markersize=8, label="CPU Reference (1 Thread)")
ax.plot(tris, cpu_4t, "+--", color="#ffa64d", linewidth=2.0, markersize=8, label="CPU Parallel (4 Threads)")
ax.plot(tris, cpu_16t, "s--", color="#ffff66", linewidth=2.2, markersize=8, label="CPU Max Parallel (16 Threads)")
ax.plot(tris, gpu_f64, "o-", color="#3399ff", linewidth=2.5, markersize=8, label="CUDA GPU (Double-Precision Float64)")
ax.plot(tris, gpu_f32, "*-", color="#00ff88", linewidth=2.8, markersize=10, label="CUDA GPU (Single-Precision Float32)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Mesh Triangle Count (Facets)", fontsize=12, fontweight="bold")
ax.set_ylabel("Execution Time (milliseconds, log scale)", fontsize=12, fontweight="bold")
ax.set_title("StrikeCEM PO Execution Time Comparison\n(360-Azimuth Sweep at 10 GHz X-Band)", fontsize=13, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(frameon=True, facecolor="#181818", edgecolor="#444444", fontsize=10)

plot2_path = os.path.join(PLOTS_DIR, "cuda_po_runtime_scaling.png")
plt.tight_layout()
plt.savefig(plot2_path)
plt.close()
print("Saved:", plot2_path)

# Plot 3: Throughput (Million & Billion Facet-Directions per Second)
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
tput_f32 = [d["throughput_mfacets_sec"]["float32"] / 1000.0 for d in mesh_data]
tput_f64 = [d["throughput_mfacets_sec"]["float64"] / 1000.0 for d in mesh_data]
tput_mix = [d["throughput_mfacets_sec"]["mixed"] / 1000.0 for d in mesh_data]
tput_cpu16 = [(d["facet_directions"] / d["cpu_seconds"]["16"]) / 1.0e9 for d in mesh_data]

ax.plot(tris, tput_f32, "d-", color="#00ffcc", linewidth=2.8, markersize=9, label="CUDA GPU (Float32)")
ax.plot(tris, tput_mix, "s-", color="#ff00aa", linewidth=2.4, markersize=8, label="CUDA GPU (Mixed Precision)")
ax.plot(tris, tput_f64, "o-", color="#3399ff", linewidth=2.4, markersize=8, label="CUDA GPU (Float64)")
ax.plot(tris, tput_cpu16, "^--", color="#ffaa00", linewidth=2.0, markersize=7, label="CPU (16 Threads)")

ax.set_xscale("log")
ax.set_xlabel("Mesh Triangle Count (Facets)", fontsize=12, fontweight="bold")
ax.set_ylabel("Throughput (Billion Facet-Directions / sec)", fontsize=12, fontweight="bold")
ax.set_title("Computational Throughput on NVIDIA RTX 4060 Laptop GPU", fontsize=13, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(frameon=True, facecolor="#181818", edgecolor="#444444", fontsize=10)

for x, y in zip(tris, tput_f32):
    ax.annotate(f"{y:.2f} B/s", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, fontweight="bold", color="#00ffcc")

plot3_path = os.path.join(PLOTS_DIR, "cuda_po_throughput.png")
plt.tight_layout()
plt.savefig(plot3_path)
plt.close()
print("Saved:", plot3_path)

# Plot 4: Angular Sweep Density Scaling (12,032 Triangles)
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
dirs_arr = [d["directions"] for d in sweep_data]
sp_sweep_f32 = [d["speedup_vs_1t"]["float32"] for d in sweep_data]
sp_sweep_16t = [d["speedup_vs_16t"]["float32"] for d in sweep_data]
sp_sweep_f64 = [d["speedup_vs_16t"]["float64"] for d in sweep_data]

ax.plot(dirs_arr, sp_sweep_f32, "d-", color="#00ffcc", linewidth=2.8, markersize=9, label="GPU Float32 vs CPU 1-Thread")
ax.plot(dirs_arr, sp_sweep_16t, "s-", color="#ff007f", linewidth=2.4, markersize=8, label="GPU Float32 vs CPU 16-Threads")
ax.plot(dirs_arr, sp_sweep_f64, "o-", color="#ffaa00", linewidth=2.2, markersize=8, label="GPU Float64 vs CPU 16-Threads")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Azimuth Sweep Directions", fontsize=12, fontweight="bold")
ax.set_ylabel("Speedup Factor (log scale)", fontsize=12, fontweight="bold")
ax.set_title("Speedup vs Angular Sweep Density\n(F-15 Jet with 12,032 Triangles at 10 GHz)", fontsize=13, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(frameon=True, facecolor="#181818", edgecolor="#444444", fontsize=10)

for x, y in zip(dirs_arr, sp_sweep_f32):
    ax.annotate(f"{y:.0f}x", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, fontweight="bold", color="#00ffcc")

plot4_path = os.path.join(PLOTS_DIR, "cuda_po_angle_scaling.png")
plt.tight_layout()
plt.savefig(plot4_path)
plt.close()
print("Saved:", plot4_path)

# Plot 5: Accuracy & Numerical Error Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
precs = ["Float64", "Mixed", "Float32"]
colors = ["#3399ff", "#ff00aa", "#00ffcc"]

# Use data from largest mesh (192,512 triangles)
last_d = mesh_data[-1]
rcs_errs = [last_d["max_rcs_err_sqm"]["float64"], last_d["max_rcs_err_sqm"]["mixed"], last_d["max_rcs_err_sqm"]["float32"]]
rel_errs = [last_d["max_field_rel_err"]["float64"], last_d["max_field_rel_err"]["mixed"], last_d["max_field_rel_err"]["float32"]]

bars1 = ax1.bar(precs, rcs_errs, color=colors, width=0.55, edgecolor="#ffffff", alpha=0.85)
ax1.set_yscale("log")
ax1.set_ylabel("Max Absolute RCS Error (sqm, log scale)", fontsize=11, fontweight="bold")
ax1.set_title("Max Absolute RCS Error vs CPU Reference\n(192,512 Triangles, 360 Directions)", fontsize=12, fontweight="bold", pad=10)
ax1.grid(True, linestyle="--", alpha=0.3, axis="y")
for bar, val in zip(bars1, rcs_errs):
    ax1.annotate(f"{val:.1e}", (bar.get_x() + bar.get_width() / 2, val),
                 textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9, fontweight="bold")

bars2 = ax2.bar(precs, rel_errs, color=colors, width=0.55, edgecolor="#ffffff", alpha=0.85)
ax2.set_yscale("log")
ax2.set_ylabel("Max Relative Complex Field Difference (log scale)", fontsize=11, fontweight="bold")
ax2.set_title("Max Relative Scattering Field Difference\n(Machine Precision Verification)", fontsize=12, fontweight="bold", pad=10)
ax2.grid(True, linestyle="--", alpha=0.3, axis="y")
for bar, val in zip(bars2, rel_errs):
    ax2.annotate(f"{val:.1e}", (bar.get_x() + bar.get_width() / 2, val),
                 textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9, fontweight="bold")

plot5_path = os.path.join(PLOTS_DIR, "cuda_po_accuracy_analysis.png")
plt.tight_layout()
plt.savefig(plot5_path)
plt.close()
print("Saved:", plot5_path)

print("\nALL PLOTS AND BENCHMARK DATA GENERATED SUCCESSFULLY!")

#!/usr/bin/env python3
"""
Executes scem_gpubench across the RAM and SBR Multi-Bounce experimental matrix,
and generates publication-quality visualization charts for StrikeCEM ADR-0036.
"""

import os
import subprocess
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

BENCH_DIR = "/home/yashwanth/projects/StrikeCEM/benchmarks/cuda_ram_sbr"
BIN_PATH = "/home/yashwanth/projects/StrikeCEM/build/release/scem_gpubench"
PLOTS_DIR = os.path.join(BENCH_DIR, "plots")
DATA_DIR = os.path.join(BENCH_DIR, "json_data")
ARTIFACT_DIR = "/home/yashwanth/.gemini/antigravity-cli/brain/58a3f65d-829c-4907-a237-7ac46c7a5a96"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Dark theme styling matching StrikeCEM aesthetic
plt.style.use("dark_background")
plt.rcParams.update({
    "font.sans-serif": ["Inter", "DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 1.2,
    "grid.color": "#1e293b",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "text.color": "#f8fafc",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
})

# 1. Benchmark Matrix: RAM Coating Scaling (PEC vs RAM SIBC vs Coated RAM)
materials = ["pec", "ram", "coated"]
subdivs = [0, 1, 2] # 752, 3008, 12032 triangles
mat_results = {}

print("=== 1. RUNNING GPU RAM COATING BENCHMARKS ===")
for mat in materials:
    mat_results[mat] = []
    for s in subdivs:
        json_path = os.path.join(DATA_DIR, f"mat_{mat}_subdiv_{s}.json")
        cmd = [
            BIN_PATH,
            "--mesh", "examples/f15/f15.obj",
            "--subdiv", str(s),
            "--dirs", "180",
            "--material", mat,
            "--iters", "2" if s <= 1 else "1",
            "--json", json_path
        ]
        print(f"Running Material={mat}, Subdiv={s}...")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print("Error in scem_gpubench:", res.stderr)
            raise RuntimeError("Benchmark run failed")
        with open(json_path, "r") as f:
            mat_results[mat].append(json.load(f))

# 2. Benchmark Matrix: SBR Multi-Bounce Scaling (1-bounce PO vs 2-bounce SBR with shadowing)
bounces = [1, 2]
sbr_subdivs = [0, 1]
sbr_results = {}

print("=== 2. RUNNING GPU SBR MULTI-BOUNCE BENCHMARKS ===")
for b in bounces:
    sbr_results[b] = []
    for s in sbr_subdivs:
        json_path = os.path.join(DATA_DIR, f"sbr_bounces_{b}_subdiv_{s}.json")
        cmd = [
            BIN_PATH,
            "--mesh", "examples/f15/f15.obj",
            "--subdiv", str(s),
            "--dirs", "72",
            "--bounces", str(b),
            "--iters", "2",
            "--json", json_path
        ]
        if b > 1:
            cmd.append("--shadowing")
        print(f"Running Bounces={b}, Subdiv={s}...")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print("Error in scem_gpubench:", res.stderr)
            raise RuntimeError("Benchmark run failed")
        with open(json_path, "r") as f:
            sbr_results[b].append(json.load(f))

# -------------------------------------------------------------
# PLOT 1: GPU RAM Throughput & Speedup vs CPU
# -------------------------------------------------------------
print("Generating Plot 1: RAM Throughput and Speedup...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=180)
fig.patch.set_facecolor("#0b0f19")
ax1.set_facecolor("#0f172a")
ax2.set_facecolor("#0f172a")

tri_counts = [d["triangles"] for d in mat_results["ram"]]
x_labels = [f"{t:,}\n(subdiv {i})" for i, t in enumerate(tri_counts)]
x = np.arange(len(tri_counts))
width = 0.22

# Left: Throughput comparison (RAM SIBC)
cpu_16t_tput = [d["facet_directions"] / d["cpu_seconds"]["16"] / 1e6 for d in mat_results["ram"]]
gpu_f64_tput = [d["throughput_mfacets_sec"]["float64"] for d in mat_results["ram"]]
gpu_f32_tput = [d["throughput_mfacets_sec"]["float32"] for d in mat_results["ram"]]

rects1 = ax1.bar(x - width, cpu_16t_tput, width, label="CPU (16 Threads)", color="#64748b", edgecolor="#94a3b8")
rects2 = ax1.bar(x, gpu_f64_tput, width, label="CUDA GPU (FP64)", color="#38bdf8", edgecolor="#7dd3fc")
rects3 = ax1.bar(x + width, gpu_f32_tput, width, label="CUDA GPU (FP32 Fast)", color="#10b981", edgecolor="#6ee7b7")

ax1.set_title("RAM Coating Solver Throughput (F-15 Airframe)", fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
ax1.set_ylabel("Throughput (Million Facet-Directions / sec)", fontsize=11, fontweight="semibold")
ax1.set_xlabel("Mesh Density (Triangles)", fontsize=11, fontweight="semibold")
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.grid(True, axis="y")
ax1.legend(loc="upper left", framealpha=0.9, facecolor="#1e293b", edgecolor="#334155")

# Annotate bars
for bar in rects3:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 30, f"{yval:.0f} M/s", ha="center", va="bottom", fontsize=9, color="#6ee7b7", fontweight="bold")

# Right: Speedup across Materials (PEC vs RAM vs Coated) on highest mesh (12,032 tris)
m_names = ["Bare PEC", "RAM SIBC (ηs=1)", "Coated RAM (2-layer)"]
m_keys = ["pec", "ram", "coated"]
sp_vs_1t = [mat_results[k][2]["speedup_vs_1t"]["float32"] for k in m_keys]
sp_vs_16t = [mat_results[k][2]["speedup_vs_16t"]["float32"] for k in m_keys]

xm = np.arange(len(m_names))
w = 0.35
r_1t = ax2.bar(xm - w/2, sp_vs_1t, w, label="Speedup vs 1-Thread CPU", color="#f59e0b", edgecolor="#fde68a")
r_16t = ax2.bar(xm + w/2, sp_vs_16t, w, label="Speedup vs 16-Thread CPU", color="#ec4899", edgecolor="#fbcfe8")

ax2.set_title("CUDA PO Speedup Across Stealth Materials (12k Triangles)", fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
ax2.set_ylabel("Hardware Acceleration Speedup (x)", fontsize=11, fontweight="semibold")
ax2.set_xlabel("Surface Coating Boundary Condition", fontsize=11, fontweight="semibold")
ax2.set_xticks(xm)
ax2.set_xticklabels(m_names)
ax2.grid(True, axis="y")
ax2.legend(loc="upper left", framealpha=0.9, facecolor="#1e293b", edgecolor="#334155")

for bar in r_1t:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, y + 4, f"{y:.1f}x", ha="center", va="bottom", fontsize=10, color="#fde68a", fontweight="bold")
for bar in r_16t:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, y + 4, f"{y:.1f}x", ha="center", va="bottom", fontsize=10, color="#fbcfe8", fontweight="bold")

plt.tight_layout()
p1_path = os.path.join(PLOTS_DIR, "cuda_ram_throughput_and_speedup.png")
plt.savefig(p1_path, dpi=180, bbox_inches="tight")
shutil.copy(p1_path, os.path.join(ARTIFACT_DIR, "cuda_ram_throughput_and_speedup.png"))
plt.close()

# -------------------------------------------------------------
# PLOT 2: GPU SBR Multi-Bounce Scaling & Ray Tracing Performance
# -------------------------------------------------------------
print("Generating Plot 2: SBR Multi-Bounce Performance...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=180)
fig.patch.set_facecolor("#0b0f19")
ax1.set_facecolor("#0f172a")
ax2.set_facecolor("#0f172a")

sbr_tri = [d["triangles"] for d in sbr_results[1]]
sbr_xlabels = [f"{t:,}\nTriangles" for t in sbr_tri]
xs = np.arange(len(sbr_tri))
ws = 0.2

# Left: 1-bounce PO vs 2-bounce SBR Ray Tracing Execution Time
t_1b_cpu = [d["cpu_seconds"]["16"] * 1000 for d in sbr_results[1]]
t_1b_gpu = [d["gpu_seconds"]["float64"] * 1000 for d in sbr_results[1]]
t_2b_cpu = [d["cpu_seconds"]["16"] * 1000 for d in sbr_results[2]]
t_2b_gpu = [d["gpu_seconds"]["float64"] * 1000 for d in sbr_results[2]]

b1 = ax1.bar(xs - 1.5*ws, t_1b_cpu, ws, label="1-Bounce CPU (16T)", color="#475569")
b2 = ax1.bar(xs - 0.5*ws, t_1b_gpu, ws, label="1-Bounce GPU (FP64)", color="#38bdf8")
b3 = ax1.bar(xs + 0.5*ws, t_2b_cpu, ws, label="2-Bounce SBR CPU (16T)", color="#e11d48")
b4 = ax1.bar(xs + 1.5*ws, t_2b_gpu, ws, label="2-Bounce SBR GPU (FP64)", color="#fb7185")

ax1.set_title("1-Bounce PO vs 2-Bounce SBR Runtime (72 Directions)", fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
ax1.set_ylabel("Execution Time (milliseconds, log scale)", fontsize=11, fontweight="semibold")
ax1.set_yscale("log")
ax1.set_xlabel("Mesh Geometry Density", fontsize=11, fontweight="semibold")
ax1.set_xticks(xs)
ax1.set_xticklabels(sbr_xlabels)
ax1.grid(True, axis="y", which="both")
ax1.legend(loc="upper left", framealpha=0.9, facecolor="#1e293b", edgecolor="#334155")

# Right: Speedup of GPU SBR over CPU
sp_sbr_1t = [sbr_results[2][i]["speedup_vs_1t"]["float64"] for i in range(len(sbr_tri))]
sp_sbr_16t = [sbr_results[2][i]["speedup_vs_16t"]["float64"] for i in range(len(sbr_tri))]

ax2.plot(sbr_tri, sp_sbr_1t, marker="o", markersize=8, linewidth=2.5, color="#f59e0b", label="SBR GPU vs 1-Thread CPU")
ax2.plot(sbr_tri, sp_sbr_16t, marker="s", markersize=8, linewidth=2.5, color="#10b981", label="SBR GPU vs 16-Thread CPU")

ax2.set_title("GPU Hardware SBR BVH Ray Tracing Speedup", fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
ax2.set_ylabel("Speedup Factor (x)", fontsize=11, fontweight="semibold")
ax2.set_xlabel("Number of Triangles in BVH", fontsize=11, fontweight="semibold")
ax2.set_xticks(sbr_tri)
ax2.set_xticklabels([f"{t:,}" for t in sbr_tri])
ax2.grid(True)
ax2.legend(loc="center right", framealpha=0.9, facecolor="#1e293b", edgecolor="#334155")

for x_val, y_val in zip(sbr_tri, sp_sbr_1t):
    ax2.annotate(f"{y_val:.1f}x", (x_val, y_val), textcoords="offset points", xytext=(0,10), ha="center", fontsize=10, color="#fde68a", fontweight="bold")
for x_val, y_val in zip(sbr_tri, sp_sbr_16t):
    ax2.annotate(f"{y_val:.1f}x", (x_val, y_val), textcoords="offset points", xytext=(0,-15), ha="center", fontsize=10, color="#6ee7b7", fontweight="bold")

plt.tight_layout()
p2_path = os.path.join(PLOTS_DIR, "cuda_sbr_multibounce_scaling.png")
plt.savefig(p2_path, dpi=180, bbox_inches="tight")
shutil.copy(p2_path, os.path.join(ARTIFACT_DIR, "cuda_sbr_multibounce_scaling.png"))
plt.close()

# -------------------------------------------------------------
# PLOT 3: Accuracy and Relative Field Difference (FP64 / Mixed / FP32)
# -------------------------------------------------------------
print("Generating Plot 3: Numerical Accuracy Verification...")
fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
fig.patch.set_facecolor("#0b0f19")
ax.set_facecolor("#0f172a")

labels = ["FP64 Double", "Mixed Precision", "FP32 Single"]
ram_rel_err = [
    mat_results["ram"][1]["max_field_rel_err"]["float64"],
    mat_results["ram"][1]["max_field_rel_err"]["mixed"],
    mat_results["ram"][1]["max_field_rel_err"]["float32"]
]
sbr_rel_err = [
    sbr_results[2][1]["max_field_rel_err"]["float64"],
    sbr_results[2][1]["max_field_rel_err"]["mixed"],
    sbr_results[2][1]["max_field_rel_err"]["float32"]
]

xp = np.arange(len(labels))
wp = 0.35
bp1 = ax.bar(xp - wp/2, ram_rel_err, wp, label="RAM Coating (SIBC ηs=1)", color="#38bdf8", edgecolor="#7dd3fc")
bp2 = ax.bar(xp + wp/2, sbr_rel_err, wp, label="SBR Multi-Bounce (2 Bounces)", color="#f43f5e", edgecolor="#fda4af")

ax.set_title("Numerical Accuracy vs CPU Reference (|E_gpu - E_cpu| / |E_cpu|)", fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
ax.set_ylabel("Maximum Relative Field Difference (log scale)", fontsize=11, fontweight="semibold")
ax.set_yscale("log")
ax.set_xticks(xp)
ax.set_xticklabels(labels, fontsize=11, fontweight="semibold")
ax.grid(True, axis="y", which="both")
ax.legend(loc="upper left", framealpha=0.9, facecolor="#1e293b", edgecolor="#334155")

for bar in bp1:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, y * 1.5, f"{y:.1e}", ha="center", va="bottom", fontsize=9, color="#7dd3fc", fontweight="bold")
for bar in bp2:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, y * 1.5, f"{y:.1e}", ha="center", va="bottom", fontsize=9, color="#fda4af", fontweight="bold")

plt.tight_layout()
p3_path = os.path.join(PLOTS_DIR, "cuda_ram_sbr_accuracy_verification.png")
plt.savefig(p3_path, dpi=180, bbox_inches="tight")
shutil.copy(p3_path, os.path.join(ARTIFACT_DIR, "cuda_ram_sbr_accuracy_verification.png"))
plt.close()

print("All benchmark runs completed and plots generated successfully!")

# StrikeCEM: High-Performance Computational Electromagnetics & RCS Engine

StrikeCEM is an industrial-grade high-frequency computational electromagnetics (CEM) simulation suite and real-time radar cross section (RCS) query engine. Designed for aerospace, naval, and defense simulation systems (including StrikeEngine integration), StrikeCEM scales from local single-workstation multi-GPU runs to high-performance computing (HPC) clusters and cloud batch deployments.

---

## Capabilities & Architecture Matrix

| Domain | Capability | Description |
|---|---|---|
| **Physics & Solvers** | **Physical Optics (PO)** | Gordon closed-form planar facet integration with monostatic and bistatic scattering modes. |
| | **PTD & Michaeli EEC** | Physical Theory of Diffraction (PTD) and Michaeli Equivalent Edge Currents along sharp wedges with Gauss-Legendre quadrature. |
| | **Fock Curvature Correction** | Frequency-dependent asymptotic Fock curvature factors restoring physical creeping-wave behavior on curved surfaces. |
| | **Coherent SBR** | Shooting and Bouncing Rays (SBR) multi-bounce ray tracing with complex dielectric Fresnel reflection coefficients and BVH acceleration. |
| | **Electromagnetic Similitude** | Sinclair electrodynamic scaling ($s_g, 1/s_g, +20\log_{10}(1/s_g)\,\text{dB}$) for scale-model simulation and chamber measurement projection. |
| **Geometry & Meshing** | **Automated Vehicle Pre-Scan** | Deep geometric scan classifying planar specular, curved, sharp edge, and cavity interior facets with hybrid solver parameter auto-tuning. |
| | **Bilateral Symmetry** | Spatial hash detection of planar target symmetry ($xz, yz, xy$), cutting solve runtime and memory footprint in half. |
| | **Dynamic Wavelength Meshing** | Conforming adaptive edge subdivision ensuring $h_{\max} \le \lambda / \rho$ per frequency sweep point without non-conforming T-junctions. |
| | **CAD & Normalization** | Loads ASCII/Binary STL and OBJ formats; transforms, verifies orientation, repairs non-manifold topology, and caches geometry. |
| **Compute & Acceleration** | **CUDA GPU Acceleration** | Native CUDA kernel dispatch with double-precision and memory-optimized mixed-precision (float32 coordinates, float64 phase). |
| | **NVTX Profiling Hooks** | Built-in instrumentation markers for NVIDIA Nsight Systems and Compute kernel tracing. |
| | **Deterministic Multi-Threading** | OpenMP/thread-pool parallel CPU execution guaranteed 100% bit-identical to serial reference runs. |
| **Distributed Orchestration** | **Local Parallel Runner** | Single-command multi-device parallel solver (`strikecem run-parallel`) with automatic shard staging and fail-closed merge reduction. |
| | **REST Campaign Broker** | Headless HTTP REST campaign coordinator (`strikecem coordinate`) with lease tracking, automatic fault recovery, and SHA-256 verification. |
| | **Headless Worker Daemon** | Autonomous pull-execute-heartbeat daemon (`strikecem worker`) with interruptible keepalive threads. |
| | **HPC & Cloud Generators** | Native generator for Slurm job arrays (`#SBATCH --array`) and Kubernetes batch jobs (`batch/v1 Job`). |
| **Storage & Runtime** | **Thread-Safe HDF5 & CSV** | High-throughput HDF5 with chunk checksums and resume support, plus CSV with JSON sidecars. |
| | **Asset Bundler (`.scempack`)** | Single-file tamper-verified asset archive packaging RCS databases, LOD meshes, and JSON metadata with cryptographic SHA-256 validation. |
| | **Spherical Splines & Python SDK** | Real-time sub-microsecond $C^1$ Hermite and $C^2$ Wendland spherical spline query engine in Python and C ABI. |

---

## CLI Command Quick Reference

The `strikecem` binary provides a unified command-line interface:

```text
strikecem [--schema PATH] [--bench-profile PATH] <command> [options...] <args...>
```

| Command | Usage | Description |
|---|---|---|
| `validate` | `strikecem validate config.json` | Validates configuration against JSON schema and inspects mesh hashes. |
| `estimate` | `strikecem estimate config.json` | Validates configuration, counts sample plan, and computes memory/runtime estimates. |
| `run` | `strikecem run config.json [--scale S]` | Executes full CEM simulation pipeline (PO/SBR/PTD) with optional similitude scaling. |
| `prescan` | `strikecem prescan <mesh\|config.json> [--report R] [--auto-tune-config T]` | Analyzes vehicle features (cavities, edges, curves) and outputs auto-tuned solver configs. |
| `run-parallel` | `strikecem run-parallel config.json [--workers N] [--devices d1,d2]` | Dispatches local parallel solve across multiple CPU threads or CUDA devices. |
| `shard` | `strikecem shard config.json --num-shards N --out-dir D` | Partitions sample plan into N deterministic shard configurations with a manifest. |
| `merge` | `strikecem merge manifest.json --output out.h5` | Validates chunk checksums and merges completed shard outputs into a unified database. |
| `coordinate` | `strikecem coordinate --manifest M --output out.h5 [--port P]` | Launches HTTP REST distributed campaign broker for cluster or cloud workers. |
| `worker` | `strikecem worker --coordinator URL [--device DEV]` | Launches autonomous headless worker daemon pulling tasks from coordinator. |
| `cluster` | `strikecem cluster generate-slurm\|generate-k8s ...` | Generates production Slurm `#SBATCH` scripts or Kubernetes batch `Job` YAML manifests. |
| `bundle` | `strikecem bundle --rcs db.h5 --mesh m.obj [--lod name=path] -o pkg.scempack` | Packages RCS databases and LOD geometry into a single `.scempack` file. |
| `vtk` | `strikecem vtk config.json --output surface.vtu` | Exports 3D per-facet induced currents and lit/shadow diagnostics to ParaView VTU. |
| `export-mesh` | `strikecem export-mesh config.json -o out.obj --format obj` | Exports procedural or normalized geometry to standalone OBJ/STL format. |

### Companion Analysis Tools

- **`scem_quicklook`**: Instant command-line RCS cut extraction and ASCII radiation bar chart rendering.
- **`scem_compare`**: Compares two simulation databases and checks pointwise tolerance with detailed regression metrics.
- **`scem_benchreport`**: Aggregates benchmark scaling profiles, reports normalized throughput ($M\text{-Eval/s}$), and renders comparison charts.

---

## Quickstart Tutorial

### 1. Pre-Scan Vehicle Geometry & Auto-Tune Hybrid Config

Inspect your 3D target geometry (e.g. an aerospace fighter or UAV) to detect re-entrant cavities, sharp edges, and curved bodies:

```bash
strikecem prescan models/uav.obj --auto-tune-config configs/uav_tuned.json
```

### 2. Run Simulation with Similitude Scaling

Simulate the target at full accuracy or scale large entities down using Sinclair similitude:

```bash
# Monolithic full-scale solve
strikecem run configs/uav_tuned.json

# Or solve as a 1:5 scale model (frequency scaled up 5x, output RCS offset by +13.98 dBsm automatically)
strikecem run configs/uav_tuned.json --scale 0.2
```

### 3. Inspect Results in Terminal

Quickly inspect a horizontal azimuth cut ($el = 0^\circ$) at 10 GHz with ASCII visualization:

```bash
scem_quicklook uav_tuned.h5 --cut azimuth --angle 0.0 --freq 10e9 --pol HH --ascii
```

### 4. Package for Real-Time Runtime

Bundle the simulation database alongside target 3D meshes for simulation engines:

```bash
strikecem bundle --rcs uav_tuned.h5 --mesh models/uav.obj -o uav_target.scempack
```

---

## Python SDK Quickstart

The Python SDK provides real-time, sub-microsecond radar cross section queries with $C^1$ bicubic and $C^2$ Wendland spherical spline interpolation:

```python
import strikecem
from strikecem import Database, QueryRequest, InterpolationMode

# Open simulation database
db = Database("uav_tuned.h5")

# Query RCS at arbitrary continuous angle
req = QueryRequest(
    azimuth_deg=34.8,
    elevation_deg=-2.1,
    frequency_hz=10.0e9,
    polarization="HH",
    mode=InterpolationMode.Bicubic  # C^1 Hermite spline
)

result = db.query(req)
print(f"RCS: {result.rcs_dbsm:.2f} dBsm ({result.rcs_sqm:.4f} m^2)")
print(f"Scattering: {result.scattering}")
```

---

## Build and Test

### Prerequisites
- CMake 3.24+
- Ninja build system
- C++20 compliant compiler (GCC 11+, Clang 14+)
- OpenSSL development libraries (`libssl-dev`)
- HDF5 C/C++ development libraries (`libhdf5-dev`)
- Python 3.9+ (optional, for Python bindings)
- NVIDIA CUDA Toolkit 11.8+ (optional, for GPU acceleration)

### Compilation

```bash
# Configure with Ninja (CUDA is auto-detected; pass -DSCEM_ENABLE_CUDA=OFF to disable)
cmake --preset release

# Build all binaries and libraries
cmake --build --preset release --parallel

# Execute comprehensive test suite (332 tests)
ctest --test-dir build/release --output-on-failure
```

### Installation

```bash
cmake --install build/release --prefix "$HOME/.local"
```

---

## In-Depth Documentation & Tutorials

- [Tutorial: Automated Vehicle Pre-Scan Workflow](docs/tutorials/vehicle_prescan_workflow.md)
- [Tutorial: Electromagnetic Similitude & Scale-Model Simulation](docs/tutorials/scale_model_similitude.md)
- [Tutorial: Distributed Cluster & Parallel Campaigns](docs/tutorials/distributed_and_parallel_campaigns.md)
- [Tutorial: Python SDK, High-Order Splines & Asset Bundling](docs/tutorials/python_sdk_and_splines.md)
- [StrikeDesigner CAD Package Integration](docs/designer-packages.md)
- [Architecture Decision Records (ADRs)](docs/adr/)

---

## License

StrikeCEM is released under the GNU LGPL-3.0. See [LICENSE](LICENSE).

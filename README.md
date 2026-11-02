# StrikeCEM: High-Performance Computational Electromagnetics & RCS Engine

StrikeCEM is a high-frequency computational electromagnetics (CEM) simulator and radar cross section (RCS) query engine. It calculates scattering signatures from 3D surface meshes and supports local CPU/CUDA runs, distributed campaigns, and runtime database queries.

---

## Capabilities & Architecture Matrix

| Domain | Capability | Description |
|---|---|---|
| **Physics & Solvers** | **Physical Optics (PO)** | Gordon closed-form planar facet integration with monostatic and bistatic scattering modes. |
| | **PTD & Michaeli EEC** | Physical Theory of Diffraction (PTD) and Michaeli Equivalent Edge Currents along sharp wedges with Gauss-Legendre quadrature. |
| | **Fock Curvature Correction** | Frequency-dependent asymptotic Fock curvature factors restoring physical creeping-wave behavior on curved surfaces. |
| | **Coherent multi-bounce** | The `SBR` configuration enables coherent two- and three-bounce contributions in the shared PO solver path. It does not dispatch to a separate general-purpose ray-tracing solver. |
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
| | **RCS query SDK** | C ABI and Python bindings query stored signatures using nearest, linear, bicubic, or spherical interpolation. |

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
| `run` | `strikecem run config.json [--scale S]` | Runs the PO-centered high-frequency solver with configured edge, curvature, material, and multi-bounce options. |
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
| `isar` | `strikecem isar database.h5 --image isar.png` | Forms a 2D range-Doppler ISAR image from a frequency-azimuth sweep. |
| `sar` | `strikecem sar config.json --output sar.png` | Synthesizes a SAR image using the selected platform and scene settings. |
| `render` | `strikecem render config.json --mode currents` | Renders surface currents, lit/shadow state, or shaded geometry. |
| `report` | `strikecem report database.h5 -o report.html` | Creates a standalone HTML radar-signature report. |

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

The Python SDK queries stored databases through the native C ABI. Choose an interpolation mode for each query:

```python
from strikecem import Database, InterpolationMode

with Database.open("uav_tuned.h5") as db:
    result = db.query(
        az_deg=34.8,
        el_deg=-2.1,
        freq_hz=10.0e9,
        pol="HH",
        interpolation_mode=InterpolationMode.BICUBIC,
    )
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

# Execute the configured test suite
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

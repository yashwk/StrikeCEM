# Tutorial: Python SDK, High-Order Splines & Asset Bundling

This tutorial explains how to consume StrikeCEM simulation outputs in Python and game/simulation engines using the StrikeCEM Python SDK ([ADR-0027](file:///home/yashwanth/projects/StrikeCEM/docs/adr/ADR-0027-high-order-spherical-rcs-splines.md)) and the `.scempack` asset bundler ([ADR-0026](file:///home/yashwanth/projects/StrikeCEM/docs/adr/ADR-0026-production-strikeengine-asset-bundling.md)).

---

## 1. Setting Up the Python SDK

The Python SDK wraps `strikecem_lib` via ctypes with zero overhead.

Ensure your environment includes `strikecem_lib`:

```bash
export PYTHONPATH=/path/to/StrikeCEM/python:$PYTHONPATH
export STRIKECEM_LIB_PATH=/path/to/StrikeCEM/build/release/libStrikeCEM.so
```

Test the import:

```python
import strikecem
print(strikecem.__version__)
```

---

## 2. Querying an RCS Database in Real Time

Open any StrikeCEM HDF5 database:

```python
from strikecem import Database, QueryRequest, InterpolationMode

db = Database("naval_vessel.h5")

# Inspect available channels and grid
print("Frequencies:", db.frequencies_hz)
print("Polarizations:", db.polarizations)
print("Angle bounds: Azimuth", db.azimuth_range_deg, "Elevation", db.elevation_range_deg)
```

### Performing Sub-Microsecond Queries

```python
req = QueryRequest(
    azimuth_deg=45.2,
    elevation_deg=-2.5,
    frequency_hz=10.0e9,
    polarization="HH",
    mode=InterpolationMode.Bicubic  # C^1 Hermite/Catmull-Rom spline
)

result = db.query(req)
print(f"RCS: {result.rcs_dbsm:.2f} dBsm ({result.rcs_sqm:.4f} m^2)")
print(f"Complex Scattering Amplitude: {result.scattering}")
```

---

## 3. Selecting the Optimal Interpolation Mode

StrikeCEM provides four interpolation algorithms tuned for real-time radar signatures:

| Mode | Mathematical Formulation | Continuity | Latency | Recommended Use |
|---|---|---|---|---|
| `InterpolationMode.Nearest` | Closest angular sample | $C^0$ discontinuous | $< 0.1\,\mu\text{s}$ | Debugging, rapid testing |
| `InterpolationMode.Bilinear` | 4-point bilinear on grid | $C^0$ continuous | $< 0.2\,\mu\text{s}$ | Fast flight simulators |
| `InterpolationMode.Bicubic` | 16-point Hermite / Catmull-Rom | $C^1$ smooth | $< 0.6\,\mu\text{s}$ | High-fidelity RF simulation |
| `InterpolationMode.SphericalSpline` | Wendland $C^2$ Compact Radial Basis | $C^2$ smooth | $< 1.8\,\mu\text{s}$ | Whole-sphere Fibonacci / unstructured grids |

```python
# Query with smooth C^2 Wendland spherical spline
result_smooth = db.query(QueryRequest(
    azimuth_deg=120.4,
    elevation_deg=15.1,
    frequency_hz=10.0e9,
    polarization="VV",
    mode=InterpolationMode.SphericalSpline
))
```

---

## 4. Packaging Production Bundles (`.scempack`)

To ship RCS databases alongside 3D target geometry and Level of Detail (LOD) models in a single, tamper-verified binary file for game or simulation runtime:

```bash
strikecem bundle \
  --rcs uav_sweep.h5 \
  --mesh models/uav_lod0.obj \
  --lod lod1=models/uav_lod1.obj \
  --lod lod2=models/uav_lod2.obj \
  --output packages/uav_target.scempack
```

### Bundle Specifications:
- Fixed 256-byte binary header with magic bytes `SCEMPACK1`.
- Cryptographic SHA-256 checksum verification across all embedded payloads.
- JSON metadata table declaring LOD meshes, target dimensions, and provenance.
- Directly loadable by StrikeEngine and game engines without external asset dependencies.

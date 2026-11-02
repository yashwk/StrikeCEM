# Tutorial: Python SDK, High-Order Splines & Asset Bundling

This tutorial explains how to consume StrikeCEM simulation outputs in Python and game/simulation engines using the StrikeCEM Python SDK ([ADR-0027](file:///home/yashwanth/projects/StrikeCEM/docs/adr/ADR-0027-high-order-spherical-rcs-splines.md)) and the `.scempack` asset bundler ([ADR-0026](file:///home/yashwanth/projects/StrikeCEM/docs/adr/ADR-0026-production-strikeengine-asset-bundling.md)).

---

## 1. Setting Up the Python SDK

The Python SDK wraps `strikecem_lib` with ctypes and opens databases through the native library.

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

## 2. Querying an RCS Database

Open a StrikeCEM database and inspect its frequency and polarization channels:

```python
from strikecem import Database, InterpolationMode

with Database.open("naval_vessel.h5") as db:
    print("Frequencies:", db.frequencies)
    print("Polarizations:", db.polarizations)
```

### Querying RCS

```python
with Database.open("naval_vessel.h5") as db:
    result = db.query(
        az_deg=45.2,
        el_deg=-2.5,
        freq_hz=10.0e9,
        pol="HH",
        interpolation_mode=InterpolationMode.BICUBIC,
    )
    print(f"RCS: {result.rcs_dbsm:.2f} dBsm ({result.rcs_sqm:.4f} m^2)")
    print(f"Complex scattering amplitude: {result.scattering}")
```

---

## 3. Choosing an Interpolation Mode

The Python enum exposes four interpolation modes:

| Mode | Behavior |
|---|---|
| `InterpolationMode.NEAREST` | Returns the nearest stored direction. |
| `InterpolationMode.LINEAR` | Uses linear interpolation on regular angle grids and spherical interpolation for unstructured directions. |
| `InterpolationMode.BICUBIC` | Uses bicubic interpolation on regular angle grids and spherical interpolation for unstructured directions. |
| `InterpolationMode.SPHERICAL` | Uses the spherical interpolation path. |

`QueryRequest` is used for batch queries. For a single query, pass the coordinates and channel directly to `Database.query` as shown above. Query latency depends on the database and machine; benchmark with the intended workload.

```python
from strikecem import Database, InterpolationMode, QueryRequest

request = QueryRequest(
    azimuth_deg=120.4,
    elevation_deg=15.1,
    frequency_hz=10.0e9,
    polarization="VV",
)
with Database.open("naval_vessel.h5") as db:
    results = db.query_batch([request], interpolation_mode=InterpolationMode.SPHERICAL)
    print(results[0].rcs_dbsm)
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

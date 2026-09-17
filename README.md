# StrikeCEM

**StrikeCEM** (Strike Computational Electromagnetics) generates radar
cross-section (RCS) databases from surface meshes for offline use by
StrikeEngine.

## Stable v1 scope

* Geometry: STL/OBJ triangle meshes (m, mm, in) with rigid transforms
* Physics: PEC, plane-wave illumination, monostatic RCS
* Solver: Physical Optics (PO) on CPU, float32/float64
* Sampling: regular azimuth/elevation grid or explicit direction list
* Polarization: HH, VV, HV, VH, RHCP, LHCP (default HH + VV)
* Output: CSV with JSON sidecar, or flat-row HDF5
* CLI: `strikecem validate | estimate | run`

The machine-readable contract is
[config.schema.json](schemas/config.schema.json) – the v1 configuration
schema. Unsupported features (other solvers, GPU execution, materials,
bistatic output, CAD input) are rejected, never silently substituted.

---

## Status

Early scaffold (Phase 0): source layout and build exist; configuration
loader, mesh pipeline, solver, output writer, and CLI are in progress.

---

## Schemas

StrikeCEM uses JSON Schema for defining simulation configuration.
* [config.schema.json](schemas/config.schema.json) – core simulation setup.
---

## License
StrikeCEM is released under the **GNU LGPL-3.0** license.  
You are free to use, modify, and redistribute under the terms of the LGPL, while keeping modifications to StrikeCEM itself open-source.  
See the [LICENSE](../StrikeCEM/LICENSE) file for the full text.

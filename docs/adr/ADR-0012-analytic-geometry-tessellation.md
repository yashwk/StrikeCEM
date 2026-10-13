# ADR-0012: Analytic Aerospace Geometry and Parametric Tessellation Engine

## Status
Accepted

## Context
Upstream specification (`FULL_IMPLEMENTATION.md` §4, `BACKLOG.md` item 45, Capability Stage E) requires:
"CAD tessellation and preservation of part names/tags. Parametric surfaces and analytic primitives for high-frequency meshing. CAD support must preserve stable part names, material tags, coordinate systems, and tessellation parameters in the geometry hash."

Aerospace targets (missiles, UAVs, fighter configurations) are fundamentally composed of canonical geometric components:
- Tangent ogive radomes / nose cones;
- Cylindrical fuselage bodies;
- Spheroidal domes and benchmark scatterers;
- Trapezoidal wings, canards, and cruciform stabilizers.

Directly generating high-fidelity tessellations from analytic primitives ensures exact surface curvature, eliminates polygonal approximation errors from external meshers, guarantees watertight manifold topology, and attaches component identities (`part_name`, `part_id`) and electromagnetic `material_tag` attributes directly to each triangle facet.

## Decision
1. **Analytic Aerospace Primitives**:
   Implement parametric generators in `include/strikecem/geometry/AnalyticGeometry.hpp`:
   - `SpherePrimitive`: radius, center.
   - `CylinderPrimitive`: radius, length, origin, axis, end caps.
   - `TangentOgivePrimitive`: base radius $R$, nose length $L$, ogive radius $\rho = \frac{R^2 + L^2}{2R}$, base cap.
   - `FinPrimitive`: root chord, tip chord, span, sweep angle, thickness, origin.
2. **Curvature-Aware Adaptive Tessellation**:
   Given maximum edge length $\delta$ or wavelength criteria ($\lambda / 10$):
   Radial segments: $N_{\text{radial}} \ge \max(12, \lceil 2\pi R / \delta \rceil)$.
   Axial segments: $N_{\text{axial}} \ge \max(4, \lceil L / \delta \rceil)$.
3. **Component Identity and Material Preservation**:
   Every primitive assigns its `part_name` into the mesh group table and `triangle_tags` per facet. This directly interfaces with `resolve_materials()` in `Material.cpp` for multi-material coatings and dielectric penetration, as well as VTK diagnostics.
4. **Schema Integration**:
   Extend `schemas/config.schema.json` under `model`:
   Allow either `model.path` or `model.primitives` array.

## Consequences
- Enables procedural aerospace target modeling without requiring external CAD software or third-party mesh converters.
- Seamlessly propagates component identity and material tags into `NormalizedMesh`, PO solver, and output files.
- Provides mathematically exact normal vectors and facet centroids for high-frequency PO integration.

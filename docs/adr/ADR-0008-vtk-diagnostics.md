# ADR-0008: Diagnostic & Field Visualization Export (VTK .vtu)

## Status
Accepted

## Context
Per `FULL_IMPLEMENTATION.md` §12.2:
Per-facet diagnostics are critical for engineers evaluating electromagnetic stealth treatments, material edge discontinuities, specular flashpoints, and shadowing:
- Surface current density $\vec{J}_s$ magnitude and phase.
- Lit vs. shadowed geometric occlusion flags.
- Material assignment tags and indices.
- Surface normal and geometric area distributions.

Per §12.2: "Diagnostic files are not interchangeable with the runtime RCS database." They provide rich 3D field spatial visualization inspectable in ParaView, VisIt, and StrikeEngine diagnostic viewports.

## Decision
1. **Format Standards:**
   Use the XML-based VTK UnstructuredGrid format (`.vtu`, version 1.0, LittleEndian).
   XML `.vtu` is natively supported across open-source and commercial visualization platforms without external proprietary library dependencies.

2. **Per-Facet Cell Data Fields:**
   - `LitStatus` (`UInt8`): 1 if illuminated by the incident wave, 0 if geometric backface or ray-traced shadow.
   - `CurrentMagnitude` (`Float64`): $|\vec{J}_s|$ surface current density magnitude in amperes per meter ($A/m$).
   - `CurrentPhaseDeg` (`Float64`): Phase $\arg(\vec{J}_s)$ in degrees $[-180^\circ, 180^\circ]$.
   - `SurfaceCurrentReal` (`Float64`, 3 components): Vector components $(\text{Re}(J_x), \text{Re}(J_y), \text{Re}(J_z))$.
   - `SurfaceCurrentImag` (`Float64`, 3 components): Vector components $(\text{Im}(J_x), \text{Im}(J_y), \text{Im}(J_z))$.
   - `MaterialId` (`UInt32`): Per-facet material assignment index.
   - `FacetArea` (`Float64`): Facet surface area in $m^2$.
   - `FacetNormal` (`Float64`, 3 components): Unit outward normal $\hat{n}$.

3. **Mesh Geometry & Topology:**
   - `<Points>`: 3D coordinates $(x, y, z)$ in meters.
   - `<Cells>`: Triangle vertex connectivity, contiguous offsets, cell type 5 (`VTK_TRIANGLE`).

4. **Integration:**
   - Configurable in `output.diagnostics.vtk_export_path` in `config.json`.
   - Accessible via CLI subcommand: `strikecem vtk [--output diag.vtu] config.json`.

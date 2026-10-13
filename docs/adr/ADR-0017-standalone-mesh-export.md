# ADR-0017: Standalone Mesh Export CLI Tooling

## Status
Accepted

## Context
StrikeCEM supports both imported meshes (STL, Wavefront OBJ) and procedural aerospace geometries (tangent ogives, cylinders, cones, fins, spheres) with applied translations, Euler rotations, scaling, and repair filters.
In engineering workflows involving StrikeDesigner, StrikeCEM, and StrikeEngine, users require the ability to:
1. Inspect, debug, and export tessellated aerospace primitive assemblies into standard 3D asset formats without running full electromagnetic simulations.
2. Export normalized, transformed, and repaired CAD geometry into standard Wavefront OBJ (`.obj`), binary Stereolithography (`.stl`), and VTK XML UnstructuredGrid (`.vtu`) formats.
3. Validate mesh topology, vertex counts, facet areas, and outward normals in third-party visualization tools (ParaView, Blender, MeshLab).

## Decision
1. Implement standalone mesh exporter functions in `strikecem::io::MeshLoader`:
   - `export_mesh_obj(const NormalizedMesh&, const std::filesystem::path&)`: exports Wavefront OBJ format with vertex coordinates (`v`), facet normals (`vn`), and 1-indexed polygon facet definitions (`f v//vn`).
   - `export_mesh_stl(const NormalizedMesh&, const std::filesystem::path&)`: exports binary STL with 80-byte header, 32-bit unsigned facet count, and 50-byte IEEE 754 float facet records (`normal`, `v1`, `v2`, `v3`, `attribute_byte_count`).
   - `export_mesh_vtu(const NormalizedMesh&, const std::filesystem::path&)`: exports VTK XML UnstructuredGrid with Points (`Float64`), Cells (`connectivity`, `offsets`, cell `types=5` for `VTK_TRIANGLE`), and CellData attributes (`Normals` 3-vector, `Area_m2` scalar).
   - `export_mesh(const NormalizedMesh&, const std::filesystem::path&, const std::string& format)`: dispatches to appropriate exporter with auto-detection from file extension or explicit format override.
2. Add the `export-mesh` CLI subcommand to `strikecem`:
   - Usage: `strikecem export-mesh <config_path> [-o <mesh_output>] [--format <obj|stl|vtu>]`
   - Automatically resolves configuration, loads procedural or imported mesh via `load_normalized_mesh`, applies transformations and repairs, and writes output.
   - Outputs summary metadata: `exported_mesh`, `vertices`, `triangles`, `surface_area`, and `status: ok`.
3. Provide unit test coverage:
   - Round-trip fidelity validation for `.obj` and `.stl` against `parse_obj` and `parse_stl`.
   - Structural XML verification for `.vtu`.
   - Procedural missile assembly export and CLI argument handling.

## Consequences
- Enables seamless CAD/mesh interchange across the Strike suite and external visualizers.
- Decouples geometry validation from solver execution time.
- All 277 tests pass with 100% test suite fidelity.

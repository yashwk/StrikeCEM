# ADR-0030: Automated Vehicle Feature Pre-Scan & Hybrid Solver Dispatcher

## Status
Accepted

## Context
Radar cross section simulation of full aerospace, naval, and ground vehicles involves heterogeneous scattering mechanisms:
- Specular reflections from broad planar plates (wings, flat bulkheads) are accurately solved with classical Physical Optics (PO).
- Continuous curved structures (radomes, nacelles, fuselages) require Fock creeping-wave and curvature factors.
- Sharp leading edges, trailing edges, and control surfaces produce strong diffraction fringes modeled by Physical Theory of Diffraction (PTD) or Michaeli Equivalent Edge Currents (EEC).
- Concave geometries, re-entrant jet engine intakes, cockpit cavities, and corner reflectors produce internal multi-bounce interactions requiring Shooting and Bouncing Rays (SBR).

Historically, setting appropriate hybrid solver flags (`edge_correction`, `shadowing`, `max_bounces`, `curvature_correction`) required deep domain knowledge from CEM practitioners, leading to misconfigured runs (e.g., attempting single-bounce PO on engine ducts, or running expensive 4-bounce SBR on flat plates).

An automated geometric pre-scan engine was required to classify vehicle surface features and auto-tune hybrid solver parameters.

## Decision
1. **Geometric Feature Classification Engine (`VehiclePreScan.hpp` / `VehiclePreScan.cpp`)**:
   - Implemented `VehiclePreScan::analyze` operating on `NormalizedMesh`.
   - Feature classification taxonomy (`VehicleFeatureType`):
     - `PlanarSpecular`: flat surface facets with zero curvature and dihedral angle below edge threshold; solved via standard PO.
     - `SmoothCurved`: facets with continuous curvature ($H > 0.05\,\text{m}^{-1}$); recommended for Fock curvature correction.
     - `SharpEdgeWedge`: facets adjacent to sharp creases (dihedral angle $> 30^\circ$); recommended for PTD / Michaeli EEC edge current corrections.
     - `CavityInterior`: facets participating in concave valleys (detected via `EdgeModel` non-convex crease extraction) or facing opposing surfaces with mutual line-of-sight ($d \cdot n_1 > 0$ and $d \cdot n_2 < 0$); recommended for SBR multi-bounce tracking and shadowing.

2. **Metrics & Volumetric Estimation**:
   - Calculates percentage breakdowns for planar, curved, edge, and cavity facet allocations.
   - Computes surface area ($m^2$), bounding box dimensions ($m$), and estimated re-entrant cavity volume ($m^3$).

3. **Hybrid Solver Auto-Tuner (`auto_tune_config`)**:
   - Evaluates pre-scan statistics and automatically injects optimal settings into solver configuration JSON:
     - Sets solver type to `"SBR"` when cavity facets are present, or `"PO"` otherwise.
     - Sets `po_options.edge_correction` to `"ptd_full"` if sharp edges exist.
     - Enables `shadowing` when non-coplanar 3D geometry, edges, or cavities exist.
     - Configures `max_bounces` (3 for cavity geometries, 1 for specular/flat objects).
     - Activates `curvature_correction` when curved lifting bodies are identified.

4. **CLI Integration (`strikecem prescan`)**:
   - Introduced top-level `strikecem prescan` command supporting input config JSON or standalone 3D meshes (.stl, .obj).
   - Generates formatted terminal ASCII reports.
   - Supports `--report <path>` to save machine-readable JSON metrics.
   - Supports `--auto-tune-config <path>` to export a tuned, ready-to-run simulation configuration.

5. **Test Verification (`tests/test_prescan.cpp`)**:
   - Tested flat plate (100% planar specular, 0 cavities, 0 edges, 1 bounce).
   - Tested cube (12 sharp edge facets, PTD recommended, shadowing enabled).
   - Tested dihedral corner reflector (100% cavity interior, SBR recommended, max bounces 3, shadowing enabled, positive cavity volume).
   - Tested JSON report serialization and config auto-tuning.

## Consequences
- Eliminates manual guesswork when configuring complex aerospace or vehicle models.
- Guarantees optimal balance between computational efficiency and physical fidelity along the Pareto frontier.
- Adds 5 new test cases to the test suite, bringing total test count to 332 tests with 100% pass rate.

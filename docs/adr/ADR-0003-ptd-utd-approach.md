# ADR-0003 — PTD/UTD approach: EEC fringe correction first, schema locked

- **Status:** accepted (approach)
- **Date:** 2026-09-17
- **Context:** the roadmap stages PTD/UTD after GPU PO, and the schema
  rejects edge correction until edge benchmarks pass. Diffraction lives
  or dies on wedge geometry: wrong wedge angles or phantom edges on
  smooth meshes silently corrupt every channel, so extraction comes
  before any fringe physics.

## Decision

1. Edge extraction first (`solvers/EdgeModel`): boundary rims plus
   interior creases above a dihedral threshold (default 0.1 rad), with
   exterior wedge parameter `n`, convexity, tangent, and adjacent faces.
   A dihedral threshold alone cannot separate coarse-smooth meshes from
   creases (e.g. an icosahedron reads as all-creases), so explicit
   feature tags stay a documented future extension per the design docs;
   no curvature-based selection, ever.
2. Diffraction core second: Kouyoumjian-Pathak coefficients with the
   Fresnel transition function, validated against the Sommerfeld exact
   half-plane solution (face-zero, shadow decay, two-distance agreement).
   This supersedes the EEC-first ordering: EEC closed forms reproduced
   from memory are unverifiable, while KP+Sommerfeld cross-checks an
   independent exact solution. The EEC line-integral idea returns in the
   3D extension as validated-2D-pattern × analytic along-edge sinc.
3. Benchmark order: Sommerfeld half-plane exact solution, then plate
   rim, then dihedral. Schema keeps `edge_correction: const none`
   until all three pass; only then is a backward-compatible extension
   proposed. No schema change in these slices.
4. Non-manifold input never reaches the edge model (the loader rejects
   it); the model throws on violation rather than counting the
   impossible.

## Consequences

- The edge model is solver-independent and tested on plate (rims only),
  cube (12 convex creases), dihedral (1 valley + rims), and sphere
  (zero phantom edges).
- Fringe physics must consume `MeshEdge` as-is; any extra per-edge data
  it needs is a new field with a test, not a parallel structure.
- Slice C (3D fringe geometry, done): `MeshEdge.face0_dir` (unit, into
  face tri0, perpendicular to tangent) feeds `solvers/EdgeFringe` — rim
  transverse frame (`phi`/`phi_prime`/`sin_beta0`) plus the analytic
  along-edge integral in PO phase convention, cross-checked against
  brute-force quadrature with a predicted sinc null. Plate-rim
  structural benchmark passes (edge-on: PO lights nothing, the two
  transverse rims stay valid with finite KP coefficients).
- Deferred to slice D: coupling validated-2D-pattern × integral,
  absolute spreading calibration, `n != 2` wedge-frame mapping, end-on
  cones. Schema stays locked (`edge_correction: const none`).
- Slice D1 (fringe amplitude, done): `MeshEdge.face1_dir` plus
  `fringe_amplitude` = D_pol(k_t, rho = L; phi, phi_prime) × I with
  k_t = k·sin_beta0. Interior creases use the wedge frame (phi = 0 on
  face0_dir, phi = wedge_n·pi on the face-tri1 ray; labeling-flip
  symmetry tested on the coefficient). Dihedral valley benchmark:
  frame lands at the predicted angle inside the exterior cone, fringe
  is finite with soft/hard split, and an exterior arc sweep tracks the
  look direction continuously. rho_ref = L stays provisional; no solver
  wiring, schema still locked.
- Remaining (slice D2): polarization-to-channel mapping, solver
  integration behind the flag, plate/dihedral RCS benchmarks, then the
  backward-compatible schema extension.
- Slice D2d (schema + wiring, done): `edge_correction` extends to
  `{"none", "fringe"}` (default `none`; GPU phase-3 pattern for
  backward-compatible 1.0 extensions). CLI builds the edge model and
  threads `FringeOptions` through solve, resume, and both writers;
  CUDA + fringe fails closed (exit 4). Provenance records
  `edge_correction` + `fringe_edges` in the CSV sidecar and HDF5 attrs.
  `config_hash` covers `edge_correction`, so fringe/none resume mixing
  is rejected by the existing identity check. All three ADR benchmarks
  pass (Sommerfeld exact, plate rim, dihedral valley): PTD/UTD track
  complete, schema unlocked.

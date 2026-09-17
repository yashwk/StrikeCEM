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

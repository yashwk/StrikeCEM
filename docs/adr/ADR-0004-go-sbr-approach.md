# ADR-0004 — GO/SBR approach: ray kernel first, schema locked

- **Status:** accepted (approach)
- **Date:** 2026-09-17
- **Context:** v1 PO is single-bounce with a per-facet facing test but no
  occlusion: every front-facing facet contributes even when another plate
  stands between it and the radar. The dihedral double-bounce return —
  the canonical corner-reflector RCS — is entirely missing. GO/SBR lives
  or dies on the ray kernel: false occlusions silently zero real returns
  while missed occlusions inflate shadowed ones, so intersection comes
  before any bounce physics.

## Decision

1. Ray kernel first (`solvers/Ray`): double-precision Möller–Trumbore,
   double-sided (zero-thickness PEC blocks both ways), closest-hit mesh
   query with source-facet skip, and a segment-occlusion predicate with
   an explicit `t_min` floor. No bounce logic, no solver changes.
2. GO shadowing second: PO restricted to unoccluded facets. Reduction
   test required: with nothing shadowed, results must equal v1 PO
   exactly (the new path reproduces the old numbers, not approximately).
3. Two-bounce specular chains third, benchmarked against the analytic
   square-dihedral double-bounce RCS (sigma = 8·pi·a²·b²/lambda²).
   Trihedral and curved-surface divergence are explicitly out of scope
   for this track (documented future work, not placeholders).
4. Benchmark order: kernel unit cases, shadowing reduction + shadowed
   plate, dihedral analytic. Schema keeps `shadowing: const false` and
   gains no bounce knob until all three pass; only then is a
   backward-compatible extension proposed. No schema change in these
   slices.

## Consequences

- Slice 1 (done): kernel + occlusion predicate + tests only.
- Slice 2 (done): GO shadowing behind `ShadowOptions` (default off, no
  CLI/schema changes). Reduction pins: lone plate and steep-dihedral
  looks reproduce v1 PO bit-identically; stacked plates resolve exactly
  to the upper plate alone. `lit_facets` counts truly illuminated
  (facing and unoccluded) facets. O(n²) per unit marked in code; the
  upgrade path is a BVH when meshes grow.
- The `t_min` floor is absolute (1e-9 of the mesh bbox diagonal, computed
  by the caller from the mesh report); wavelength-relative flooring is
  the documented upgrade path if electrically tiny features ever arrive.

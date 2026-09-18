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
   square-dihedral double-bounce RCS (sigma = 8·pi·a²·b²/lambda²) and
   the analytic trihedral return (FULL §9.4 requires both; trihedral is
   therefore in scope for the unlock slice, curved-surface divergence
   stays out as documented future work).
4. Benchmark order: kernel unit cases, shadowing reduction + shadowed
   plate, dihedral + trihedral analytic. Schema keeps
   `shadowing: const false` and gains no bounce knob until all pass;
   only then is a backward-compatible extension proposed. No schema
   change in these slices.
5. Unlock additionally requires (IMPLEMENTATION §12.3, FULL §9): a BVH
   (or equivalent) acceleration structure replacing the O(n²) scan,
   ray/bounce provenance metadata (density, max bounces, thresholds),
   and a GO output warning identifying edge/shadow limitations
   (FULL §9.4: GO does not replace diffraction).

## Consequences

- Slice 1 (done): kernel + occlusion predicate + tests only.
- Slice 2 (done): GO shadowing behind `ShadowOptions` (default off, no
  CLI/schema changes). Reduction pins: lone plate and steep-dihedral
  looks reproduce v1 PO bit-identically; stacked plates resolve exactly
  to the upper plate alone. `lit_facets` counts truly illuminated
  (facing and unoccluded) facets. O(n²) per unit; the
  upgrade path is a BVH when meshes grow.
- Slice 3 (done): two-bounce specular chains behind `GoOptions`
  (`shadowing`/`two_bounce`, default off; `ShadowOptions` renamed).
  Single-pair hand computation passes exactly with both orders firing;
  non-retroreflective and exit-blocked pairs contribute exactly zero;
  CUDA + any GO flag fails closed. Dihedral benchmark at optimal 45°
  lands within ±20% of 8πa²/λ² with HH/VV agreement, converging under
  refinement. dihedral.obj's vertical normal faces away from the pocket,
  so it cannot double-bounce single-sided; benchmarks use a
  pocket-facing mesh instead.
- Slice 4 (done): `GoOptions.max_bounces` in {1, 2, 3} (validated, else
  throw). Triples share the pair transport/radiate code with one more
  hop. max_bounces=2 on the trihedral is bit-identical to PO (no
  2-bounce retroreflection exists when the look has an along-edge
  component for every valley). Trihedral benchmark at boresight lands
  within ±20% of 12πa⁴/λ² both channels. Triple hand values are not
  pinned individually; the evidence stack is pair-exact transport plus
  max2-exact plus analytic agreement.
- Slice 5 (done): median-split AABB BVH backing every solver ray query
  (nearest-hit with lowest-index tie-break, any-hit occlusion). BVH vs
  brute force agree bit-exactly on 2000 plate + 200 sphere rays; solver
  results are unchanged by construction. Measured 383x per-ray speedup
  on the 20k-triangle sphere (109us to 0.3us). The O(n²) ceiling is
  retired.
- The `t_min` floor is absolute (1e-9 of the mesh bbox diagonal, computed
  by the caller from the mesh report); wavelength-relative flooring is
  the documented upgrade path if electrically tiny features ever arrive.

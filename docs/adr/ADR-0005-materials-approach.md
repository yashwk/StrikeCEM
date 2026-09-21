# ADR-0005 — Materials approach: Fresnel core first, schema locked

- **Status:** accepted (approach)
- **Date:** 2026-09-18
- **Context:** v1 physics is PEC with a homogeneous background medium; the
  schema carries `epsilon_r`/`mu_r`/`sigma` for the background but no wall
  properties. SPEC §4 gates materials on Fresnel tests, and FULL §8
  requires a versioned material model, the $e^{+j\omega t}$ complex
  convention, TE/TM basis rules, and dielectric/coating/PEC-degenerate
  tests. A wrong loss sign silently creates gain, so the convention comes
  before any model or solver wiring.

## Decision

1. Convention first: $e^{+j\omega t}$ throughout; passive complex
   relative permittivity $\epsilon_r = \epsilon' - j\epsilon''$
   ($\epsilon'' \ge 0$, folding conductivity as $\sigma/\omega$);
   $n = \sqrt{\epsilon_r\mu_r}$ and $\eta = \eta_0\sqrt{\mu_r/\epsilon_r}$
   on principal branches. Energy conservation pins the signs, not
   memory of textbook $e^{-j\omega t}$ forms.
2. Fresnel core second (`solvers/Fresnel`): single-interface amplitude
   coefficients in impedance form (exact with magnetic contrast,
   reducing to Hecht $r_s$/$r_p$ when $\mu_1 = \mu_2$), complex Snell,
   plus transmission for the energy check. Benchmark order: normal
   incidence analytic, Brewster zero, PEC limit, lossless energy,
   total internal reflection.
3. Deferred: `MaterialSample`/`MaterialModel` tables, frequency
   interpolation, coatings/transfer-matrix, tag policy, and all solver
   integration. No schema change in these slices; no solver changes.
4. Recorded for unlock time (FULL §8.3): PEC-derived fringe corrections
   are not automatically valid for dielectric edges — fringe + dielectric
   walls will need a warning or formulation gate.

## Consequences

- Slice 1 (done): convention helpers + Fresnel core + tests only.
- Slice 2 (done): `MaterialModel` tables + validation policy + linear
  frequency interpolation + conductivity folding. Single-entry tables
  are constant; multi-entry tables interpolate inside and throw
  outside; PEC is bare (empty table/layers) and unevaluatable;
  dielectric needs a table and no layers; coated needs layers.
- Slice 3 (done): coated-PEC recursion reusing `fresnel()` stage by
  stage (top-down Snell chain, bottom-up combination). Pinned by
  half-wave absentee, thin-degenerate PEC recovery, an exact asymmetric
  two-layer case (which caught a top-down combination bug: the
  single-layer tests cannot see recursion order), lossless unit
  magnitude, and empty-stack PEC return. Layer order is outer-to-PEC.
- Slice 4a (done): Fresnel-weighted facet currents behind
  `MaterialOptions` (default off, uniform wall, no schema change).
  Per lit facet the incident field is split into TE/TM on the local
  plane of incidence (`e_te = n x k_hat`, `e_tm = k x e_te`, reflected
  basis `k_ref x e_te` matching the Hecht `R_TM` sign in `fresnel()`);
  equivalent currents `J = n x (H_inc + H_ref)` and
  `M = -n x (E_inc + E_ref)` radiate as
  `F = (jk/4pi) A [eta r x (r x J) + r x M]`. Coated walls reuse
  `coated_pec_reflection()` per facet with the vacuum wavelength.
  PEC walls (or disabled) keep the legacy path bit-identically.
  Solver pins: PEC bit-identity; normal dielectric power exactly
  |R|^2 = 1/9 with null cross-pol; oblique HH/VV track |R_TE|/|R_TM|
  hand values with null cross-pol; lossy matches `fresnel()` with
  passive |R| < 1 (loss can raise |R| vs lossless by moving the match,
  so no reduction is asserted); coated thin/half-wave recover PEC;
  background-matched wall returns ~0 (pins the M sign); float32 tracks
  float64. Gates: non-PEC + CUDA throws, non-PEC + multi-bounce throws,
  coated frequency tables throw, model/wall type mismatch throws,
  out-of-range table frequency throws at solve time, fringe + non-PEC
  warns (edge returns stay PEC-based). Shadowing composes (geometric
  only). Remaining for unlock (4b/4c): per-face tag mapping in the mesh
  loader (FR-12 forbids tag-driven behavior until it exists), then the
  materials schema block with the tag-coverage policy, provenance, and
  estimator treatment.
- TE means E perpendicular to the local plane of incidence (s-pol);
  TM means E in the plane (p-pol), with Hecht's reflected-basis sign
  ($R_{\mathrm{TM}} = -R_{\mathrm{TE}}$ at normal incidence; PEC limit
  $R_{\mathrm{TE}} \to -1$, $R_{\mathrm{TM}} \to +1$ as the reflected
  p-basis flips with the ray — both satisfy the tangential flip).

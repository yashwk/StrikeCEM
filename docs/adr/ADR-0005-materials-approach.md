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
- TE means E perpendicular to the local plane of incidence (s-pol);
  TM means E in the plane (p-pol), with Hecht's reflected-basis sign
  ($R_{\mathrm{TM}} = -R_{\mathrm{TE}}$ at normal incidence; PEC limit
  $R_{\mathrm{TE}} \to -1$, $R_{\mathrm{TM}} \to +1$ as the reflected
  p-basis flips with the ray — both satisfy the tangential flip).

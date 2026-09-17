# ADR-0001 — PO far-field uses the transverse projector r̂×(r̂×J)

- **Status:** accepted (implementation); diverges from design-doc SPEC §3
  until the design docs are amended
- **Date:** 2026-09-17
- **Context:** the design documents state the v1 PO facet contribution as
  `F_m = (jkη/4π)·A·[r̂×J_m]·e^{+jkr̂·rc}` (SPEC §3, IMPLEMENTATION §7.3).
  The scattered far field of an electric surface current requires the
  transverse projection `r̂×(r̂×J)`; a single `r̂×J` is the far *magnetic*
  field direction, not the electric far-field amplitude.

## Hand computation (single triangle)

Triangle in the z=0 plane (normal +ẑ, area 0.5, centroid phase 0), wave
from above (k̂=(0,0,−1)), H-transmit (E₀ along ŷ by the pole basis rule):

- `H_inc = (E₀/η)·x̂`, so `J = 2n×H = (0, 2E₀/η, 0)`
- Design-doc form: `r̂×J = (0,0,1)×(0,2E₀/η,0) = (−2E₀/η, 0, 0)`
  → `S_HH = 0`, all broadside return lands in cross-pol.
- Physical truth: a PEC plate at normal incidence returns co-polarized
  field (`E_ref = −E_inc` tangential), cross-pol exactly zero by symmetry.
  The broadside `4πA²/λ²` peak (SPEC §7.1) is a co-pol statement, so the
  single-cross form contradicts the spec's own acceptance case.
- Corrected form: `r̂×(r̂×J) = (0,−2E₀/η,0)` → `S_HH = −jk/4π`,
  `S_VH = 0`, `σ = 4π|S_HH|² = π/λ² = 4πA²/λ²` — the textbook result.

RCS magnitude is unaffected (same vector norm); only the vector direction
— channel assignment and coherent phase — was wrong.

## Decision

Implement `F_m = (jkη/4π)·A·[r̂×(r̂×J_m)]·e^{+jkr̂·rc}` under the
$e^{+j\omega t}$ convention. The single-triangle closed-form test
(`Mesh-independent S_HH = −jk/4π·e^{+jkr̂·rc}`, null cross-pol) pins this
behavior and fails under the design-doc form.

## Consequences

- This implementation intentionally diverges from design-doc SPEC §3 and
  IMPLEMENTATION §7.3 until those are amended upstream.
- Golden tests encode the corrected behavior; do not "fix" them to match
  the single-cross form.
- If upstream keeps the single-cross form, this ADR must be revisited —
  the two forms disagree on every off-trivial channel assignment.

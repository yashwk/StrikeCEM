# ADR-0022: Full PTD Wedge and Equivalent Edge Current (EEC) Integration

## Status
Accepted

## Context
Upstream specification `FULL_IMPLEMENTATION.md` §10.2 states:
"PTD adds equivalent edge currents or fringe fields to the PO solution. `PTD-lite` and `PTD-full` must have separately documented equations, validity assumptions, and cost models.
The edge strip mesh width is a numerical support region, not a substitute for an edge diffraction coefficient. Non-PEC edges require a warning unless a material-aware formulation exists."

Prior to this change:
- `edge_correction` was restricted to `"none"` or `"fringe"`.
- The edge fringe evaluator used a single midpoint evaluation with a closed-form $\operatorname{sinc}$ integral, ignoring edge illumination shadowing across adjacent wedge faces.
- Non-PEC edges were solved without diagnostic warnings, ignoring the PEC wedge assumption of canonical UTD/PTD.

## Decision
1. Schema Extension (`schemas/config.schema.json`):
   - Expanded `solver.po_options.edge_correction` enum to `["none", "fringe", "ptd_lite", "ptd_full"]`.
   - Maintained backward compatibility where `"fringe"` aliases `"ptd_lite"`.

2. Dual PTD Execution Modes (`include/strikecem/solvers/EdgeFringe.hpp`, `src/strikecem/solvers/EdgeFringe.cpp`):
   - Defined `enum class PtdMethod { Lite, Full };`.
   - **`PTD-lite` (`fringe`)**:
     - Fast evaluation: $O(N_{\text{edges}})$, 1 evaluation per edge.
     - Closed-form analytical $\operatorname{sinc}$ along-edge integral:
       $$I_{\text{edge}} = L \operatorname{sinc}\left(\frac{k L (\hat{r}-\hat{s})\cdot\hat{t}}{2}\right) e^{j k (\hat{r}-\hat{s})\cdot(\mathbf{c} - \mathbf{r}_{\text{ref}})}$$
     - Midpoint Keller/UTD diffraction evaluation.
   - **`PTD-full` (Michaeli EEC)**:
     - 5-point Gauss-Legendre numerical line quadrature along the edge contour:
       $$I_{\text{edge}} = \frac{L}{2} \sum_{i=0}^4 w_i e^{j k (\hat{r}-\hat{s})\cdot(\mathbf{c} + x_i \frac{L}{2}\hat{t} - \mathbf{r}_{\text{ref}})}$$
     - Rigorous illumination state masking for wedge faces:
       - Evaluates local face illumination: $\hat{s}\cdot\mathbf{n}_0 < 0$ and $\hat{s}\cdot\mathbf{n}_1 < 0$.
       - If both adjacent wedge faces are geometrically shadowed from the incoming wave, the fringe diffraction vanishes (`std::nullopt`), preventing unphysical fringe leakage across shadowed wedges.
   - Non-PEC Material Diagnostic:
     - In `PhysicalOptics.cpp`, detects whether extracted edges border triangles with non-PEC tags.
     - Injects warning: `"edge diffraction enabled with non-PEC faces; PTD assumes PEC wedge"`.

3. Test Verification (`tests/test_fringe.cpp`):
   - `QuadratureMatchesAnalyticalAtBroadside`: verifies Gauss-Legendre quadrature matches exact analytical sinc to $< 10^{-12}$.
   - `QuadratureAccurateAtObliqueIncidence`: verifies quadrature convergence under oblique illumination.
   - `PtdFullShadowedWedgeZeroFringe`: verifies shadowed wedges evaluate to null fringe response.
   - `PtdNonPecFaceWarning`: verifies automatic detection and warning for non-PEC edge configurations.

## Consequences
- Physical Theory of Diffraction (PTD) is fully supported with explicit lite and full tiers.
- Accurate wedge shadowing and phase tracking without unphysical fringe radiation from dark faces.
- Material compliance contracts enforced with informative warnings.

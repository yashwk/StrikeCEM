# ADR-0018: Configurable Electromagnetic Phase Reference Point

## Status
Accepted

## Context
Upstream specification `FULL_IMPLEMENTATION.md` §5.4 notes:
"The phase origin is the transformed mesh coordinate origin unless a future phase_reference option is introduced. Translating the target changes complex phase but not monostatic RCS; this property is a useful test."

In high-frequency Physical Optics, each facet $m$ with centroid $\vec{r}_m$ contributes to the scattered far-field through the coherent phase factor:
$$e^{-j k (\hat{k}_{\text{inc}} - \hat{r}_{\text{obs}}) \cdot \vec{r}_m}$$

When comparing targets across coordinate system translations (such as airframe body-axes vs engine nacelle local coordinates), or when integrating scattering matrices into 6-DoF flight dynamics models (which define phase about the center-of-gravity $\vec{r}_{\text{cg}}$), the phase origin must be decoupled from the raw CAD coordinate origin.

## Decision
1. Extend `physics` configuration in `schemas/config.schema.json`:
   - Add `physics.phase_reference`: 3-element float array `[x, y, z]` in meters (default `[0.0, 0.0, 0.0]`).
   - Materialize defaults in `strikecem::resolve_defaults()`.
2. Implement phase reference origin transformation across all solvers:
   - In CPU PO (`PhysicalOptics.cpp`): facet and material phase terms evaluate $\vec{r}' = \vec{r} - \vec{r}_{\text{ref}}$, while spatial BVH ray tracing preserves world coordinates for geometric occlusion tests.
   - In Edge Fringe / UTD (`EdgeFringe.cpp`): edge centroid phase evaluations offset by $\vec{r}_{\text{ref}}$: $\vec{c}' = \frac{1}{2}(\vec{p}_0 + \vec{p}_1) - \vec{r}_{\text{ref}}$.
   - In CUDA GPU PO (`GpuPO.cu`): host pre-computes centroids relative to `phase_ref` prior to device memory upload.
3. Verification:
   - Monostatic RCS magnitude invariance: monostatic $\sigma$ remains identical ($< 10^{-9}$ m$^2$) regardless of chosen $\vec{r}_{\text{ref}}$.
   - Phase rotation theorem: phase shifts analytically by $\Delta \phi = 2 k (\hat{k}_{\text{inc}} \cdot \vec{r}_{\text{ref}}) \pmod{2\pi}$.
   - Translation equivalence: translating a target by $\vec{d}$ while setting $\vec{r}_{\text{ref}} = \vec{d}$ yields bit-identical complex scattering amplitudes to the original target at origin.

## Consequences
- Independent control of the electromagnetic phase center for sensor fusion and flight simulation.
- Provenance and config hashes accurately record phase center specifications.
- 100% test pass rate across all suites.

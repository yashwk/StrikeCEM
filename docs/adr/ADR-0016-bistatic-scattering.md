# ADR-0016: Bistatic Radar Scattering Physical Optics Engine

## Status
Accepted

## Context
Upstream specifications (FULL_IMPLEMENTATION.md §§2.2, 5.2, 17) demand first-class support for bistatic radar scattering scenarios where transmitter and receiver directions are decoupled.
In bistatic radar:
1. Transmitter emits plane waves along incident vector $\hat{k}_{\text{inc}}$: $\vec{E}_{\text{inc}}(\vec{r}) = \vec{E}_0 e^{-j k \hat{k}_{\text{inc}} \cdot \vec{r}}$.
2. Observer measures scattered far-field in observation direction $\hat{r}_{\text{obs}}$.
3. The bistatic angle is defined as $\beta = \arccos(-\hat{k}_{\text{inc}} \cdot \hat{r}_{\text{obs}})$, with $\beta = 0^\circ$ denoting monostatic backscatter and $\beta = 180^\circ$ denoting forward scatter.
4. The Physical Optics surface integral must decouple transmit phase and receive phase: $\vec{w} = \hat{k}_{\text{inc}} - \hat{r}_{\text{obs}}$, yielding phase factor $e^{-j k (\hat{k}_{\text{inc}} - \hat{r}_{\text{obs}}) \cdot \vec{r}'}$.
5. Receiver polarization basis must be evaluated in the observation plane $(\hat{h}_{\text{rx}}, \hat{v}_{\text{rx}})$ defined by `polarization_basis(r_obs)`.

## Decision
1. Extend `Direction` in `strikecem::Direction`:
   - Add `is_bistatic`, `bistatic_azimuth_deg`, `bistatic_elevation_deg`, `r_obs_hat`, and `bistatic_angle_deg`.
   - Provide default initialization that sets $\hat{r}_{\text{obs}} = -\hat{k}_{\text{inc}}$ and $\beta = 0^\circ$ for monostatic configurations.
2. Update `schemas/config.schema.json`:
   - Allow `physics.monostatic: boolean` (default: true).
   - Add `angles.bistatic` object supporting modes: `fixed_offset`, `fixed_receiver`, `fixed_transmitter`, and `bistatic_angle`.
   - Support 4-tuples `[tx_az, tx_el, rx_az, rx_el]` in `angles.list`.
3. In `PhysicalOptics.cpp` and `GpuPO.cu`:
   - Use $\hat{n} \cdot (-\hat{k}_{\text{inc}}) > 0$ for transmitter illumination.
   - Project far-field radiation onto observation vector $\hat{r}_{\text{obs}}$.
   - Project received linear channels onto receiver polarization basis $(\hat{h}_{\text{rx}}, \hat{v}_{\text{rx}})$.
   - Preserve bit-for-bit equivalence when $\hat{r}_{\text{obs}} = -\hat{k}_{\text{inc}}$.

## Consequences
- Full support for bistatic simulations, forward-scatter shadow lobe validation ($\sigma_{\text{forward}} = \frac{4 \pi A^2}{\lambda^2}$), and multi-static radar networks.
- 100% backward compatibility and bit-identical parity with all monostatic baselines.

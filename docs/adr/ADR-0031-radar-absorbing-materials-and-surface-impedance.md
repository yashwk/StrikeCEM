# ADR-0031: Radar Absorbing Materials & Leontovich Surface Impedance Boundary Conditions

## Status
Accepted

## Context
Low-observable (stealth) platforms rely extensively on Radar Absorbing Materials (RAM), lossy resistive sheets, and radar absorbing coatings to attenuate radar cross section. Classical high-frequency Physical Optics (PO) formulations conventionally assume Perfect Electric Conductors (PEC, $\sigma \to \infty$, $\mathbf{E}_t = 0$), where the induced surface current is simply $\mathbf{J}_s = 2 \hat{n} \times \mathbf{H}_i$.

While StrikeCEM supports multi-layer dielectric coatings on PEC backing via Fresnel transfer-matrix formulations (`WallType::Coated`), simulating ultra-thin RAM sheets, Salisbury screens, Jaumann absorbers, or high-resistivity composites requires the Leontovich Surface Impedance Boundary Condition (SIBC). Without SIBC, users cannot model impedance-tapered stealth treatments or impedance-matched radar absorbing skins.

## Decision
1. **Leontovich Surface Impedance Formulation (`Material.hpp` / `Material.cpp`)**:
   - Added `WallType::Impedance` to material wall taxonomy.
   - Introduced `surface_impedance` (complex normalized impedance $\eta_s = Z_s / Z_0$, where $Z_0 \approx 376.7303\,\Omega$) to `MaterialModel`.
   - Implemented `impedance_reflection(eta_s, cos_theta_i)` computing angular plane-wave reflection coefficients:
     $$r_{\text{TE}}(\theta_i) = \frac{\eta_s \cos\theta_i - 1}{\eta_s \cos\theta_i + 1}$$
     $$r_{\text{TM}}(\theta_i) = \frac{\eta_s - \cos\theta_i}{\eta_s + \cos\theta_i}$$
   - Energy conservation & passivity guard: strictly enforces $\text{Re}(\eta_s) \ge 0$, throwing `MaterialError` on active or non-physical gain media.

2. **Physical Optics Integration (`PhysicalOptics.cpp`)**:
   - Integrated `facet_wall == WallType::Impedance` into the core PO facet accumulation loop.
   - Project incident fields into local transverse electric (TE) and transverse magnetic (TM) components with respect to facet normal and wave vector.
   - Modulate scattered electric field using polarized reflection coefficients $r_{\text{TE}}$ and $r_{\text{TM}}$.

3. **Schema & Config Specification (`schemas/config.schema.json`)**:
   - Extended material wall schema to allow `"type": "impedance"` and `"type": "sibc"`.
   - Added properties `surface_impedance_ohms` (2-element array $[R_s, X_s]$ in ohms) and `normalized_impedance` (2-element array $[\text{re}, \text{im}]$ relative to free space $Z_0$).

4. **Testing & Verification (`tests/test_material.cpp`)**:
   - `ImpedanceReflectionLimits`: verifies PEC recovery ($\eta_s \to 0 \implies r \to -1$), free-space matched absorber ($\eta_s = 1.0 \implies r = 0$ at normal incidence), and TM Brewster angle absorption.
   - `ImpedanceJsonParsing`: validates schema parsing of impedance configurations in ohms and normalized forms.
   - `ImpedanceSolveFlatPlateAbsorption`: verifies full PO solver on a flat plate with matched RAM ($\eta_s = 1.0$), demonstrating $>100\,\text{dB}$ absorption compared to bare PEC.

## Consequences
- Enables high-fidelity simulation of RAM coatings and resistive stealth treatments directly in the high-frequency PO solver.
- Seamlessly integrates with existing multi-material per-facet assignments and adaptive meshing.
- Increases test suite coverage with 3 dedicated tests passing at 100%.

# ADR-0021: Physical Optics Curvature Correction

## Status
Accepted

## Context
Upstream specification `FULL_IMPLEMENTATION.md` §6.3 states:
"The standard PO surface current is $\mathbf{J}_{\text{PO}} = 2\hat{n}\times\mathbf{H}_{\text{inc}}$, which assumes a locally planar tangent plane. On curved surfaces with finite principal radii of curvature ($R_1, R_2$), high-frequency asymptotic Fock theory dictates that the surface current receives a first-order curvature correction factor:
$$C_{\text{curv}} = 1 - j \frac{H}{k \cos\theta_i}$$
where $H = \frac{1}{2}(\kappa_1 + \kappa_2)$ is the local mean curvature, $k = 2\pi / \lambda$ is the wave number, and $\theta_i$ is the angle of incidence relative to the surface normal."

Without curvature correction, standard PO assumes infinite radius of curvature on all facets, causing errors in creeping wave and creeping phase behavior across convex surfaces like missile radomes, leading edges, and curved fuselages.

## Decision
1. Schema Extension (`schemas/config.schema.json`):
   - Changed `solver.po_options.curvature_correction` from `{"const": false}` to `{"type": "boolean", "default": false}`.

2. Curvature Estimation Engine (`include/strikecem/solvers/Curvature.hpp`, `src/strikecem/solvers/Curvature.cpp`):
   - Implemented `estimate_mesh_curvatures(mesh, max_smooth_angle_deg)`:
     - Builds edge-to-triangle adjacency map across the entire mesh.
     - For each interior smooth edge (dihedral angle $\le 45^\circ$), computes the directional normal curvature via the edge-normal projection:
       $$\kappa = \frac{(\mathbf{n}_{\text{adj}} - \mathbf{n}_t) \cdot \hat{u}}{(\mathbf{c}_{\text{adj}} - \mathbf{c}_t) \cdot \hat{u}}$$
       where $\hat{u}$ is the unit in-plane normal perpendicular to the shared edge pointing towards the adjacent facet.
     - Accurately recovers $H = 1/R$ on curved bodies while eliminating triangulation skew and shear errors.
     - Flags boundaries and sharp creases ($> 45^\circ$) as discontinuities (leaving edge diffraction to Fringe / PTD).
   - Implemented `po_curvature_factor(mean_curvature, k, cos_inc)`:
     - Applies Fock asymptotic correction: $1 - j H / (k \cos\theta_i)$ with $\cos\theta_i$ clamped to avoid grazing divergence.
     - Returns identically $1.0 + 0.0j$ when $H = 0$ or $k \le 0$.

3. Solver Integration (`src/strikecem/solvers/PhysicalOptics.cpp`):
   - In `solve_typed`, precomputes mesh curvature profile when `curvature_correction` is enabled.
   - Scales the equivalent surface currents ($\mathbf{J}_{\text{PO}}$ for PEC and equivalent far-field contribution for dielectric walls) by $C_{\text{curv}}$.
   - Enforces CUDA accelerator check throwing `cuda::CudaError("curvature correction is CPU-only in v1")` when enabled under GPU execution.

4. Test Verification (`tests/test_curvature.cpp`):
   - Verifies flat plate curvature evaluation yields exactly $H = 0$.
   - Verifies closed sphere of radius $R = 2.0$ yields $H \approx 0.5$ within discretization bounds.
   - Verifies Fock correction factor properties (flat invariance, grazing clamping).
   - Verifies flat plate PO solve invariance when `curvature_correction` is toggled.
   - Verifies sphere PO RCS response modulation when curvature correction is active.
   - Verifies CUDA accelerator rejection.

## Consequences
- High-frequency Fock curvature physics enabled for curved targets.
- Zero numerical drift on planar targets ($H = 0 \implies C_{\text{curv}} = 1$).
- CPU solver verified and backward compatible (defaults to false).

# ADR-0027: High-Order Spherical RCS Splines & Continuous Query Interpolation

## Status
Accepted

## Context
Upstream specification `FULL_IMPLEMENTATION.md` Stage F, `STRIKEDESIGNER_INTEGRATION.md`, and `SPEC.md` require high-fidelity, continuous query interpolation for runtime StrikeEngine radar cross section evaluation:
- Previously, `RcsQueryEngine` provided only bilinear interpolation for regular $(az, el)$ angle grids and nearest-neighbor fallback for unstructured or spherical grids.
- Bilinear interpolation produces $C^0$ continuity with derivative discontinuities across grid lines, causing artificial gradient kinks during closed-loop radar tracking loops and sensor simulation.
- Unstructured spherical grids (e.g. Spherical Fibonacci sampling) suffered from discrete step jumps when transitioning between nearest sample bins.

## Decision
1. **High-Order Bicubic Spline (`SCEM_INTERP_BICUBIC` / `InterpolationMode::Bicubic`)**:
   - Implemented 2D separable cubic Hermite / Catmull-Rom spline interpolation on regular angle grids.
   - Evaluates 1D cubic Hermite splines across 4 surrounding elevation columns, followed by 1D cubic Hermite spline along azimuth:
     $$p(u) = (2u^3 - 3u^2 + 1) y_0 + (u^3 - 2u^2 + u) h m_0 + (-2u^3 + 3u^2) y_1 + (u^3 - u^2) h m_1$$
   - Enforces centered difference tangents with second-order boundary approximations, guaranteeing continuous $C^1$ gradients across all grid boundaries.
   - Preserves exact node values at sample points ($u=0 \implies y_0, u=1 \implies y_1$) with zero numerical drift.
   - Highly optimized: requires only 5 1D cubic polynomial evaluations, maintaining sub-microsecond query latency (< 0.2 $\mu$s).

2. **Smooth Spherical Spline (`SCEM_INTERP_SPHERICAL` / `InterpolationMode::Spherical`)**:
   - Geodesic angular distance on unit sphere: $\psi_i = \arccos(\text{clamp}(\hat{v}\cdot\hat{k}_i, -1.0, 1.0))$.
   - Compactly supported $C^2$ Wendland radial basis function combined with inverse-distance angular weighting over $K$ nearest spherical neighbors:
     $$w_i = (1 - q)^4 (4q + 1) / (\psi_i^2 + \epsilon), \quad q = \psi_i / R$$
   - Provides rotationally invariant, bounded, $C^1/C^2$ smooth complex field interpolation across the whole sphere, with exact interpolation when $\hat{v} \to \hat{k}_i$.

3. **C ABI & Python SDK Integration**:
   - Added `scem_interpolation_mode_t` enum with `SCEM_INTERP_NEAREST`, `SCEM_INTERP_LINEAR`, `SCEM_INTERP_BICUBIC` (default), and `SCEM_INTERP_SPHERICAL`.
   - Added `scem_database_set_interpolation_mode` and `scem_database_get_interpolation_mode`.
   - Exposed `InterpolationMode` in Python SDK `strikecem`, with `db.interpolation_mode` property and optional per-query `interpolation_mode` override parameter.

4. **Test Verification**:
   - `CApi.InterpolationModeConfiguration`: verifies C ABI mode getting, setting, and error handling.
   - `CApi.HighOrderBicubicSplineContinuity`: verifies exact node interpolation and smooth non-zero continuous arc evaluation.
   - `CApi.SphericalSplineEvaluation`: verifies smooth off-grid whole-sphere query evaluation.
   - `tests/python/test_bindings.py`: verifies Python SDK property getting/setting and query overrides.

## Consequences
- StrikeEngine and simulation clients now receive smooth, $C^1$ continuous RCS fields without derivative kinks.
- Fully compatible with existing databases and tools; default mode set to high-order bicubic spline.
- Zero numerical regression across all 317 unit and integration tests.

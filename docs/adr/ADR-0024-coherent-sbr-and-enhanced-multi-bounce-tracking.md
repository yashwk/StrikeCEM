# ADR-0024: Coherent SBR and Enhanced Multi-Bounce Ray Tracking

## Status
Accepted

## Context
Upstream specification `FULL_IMPLEMENTATION.md` §9.2–§9.3 states:
- "The ray engine must support coherent path accumulation when phase relationships are meaningful."
- "At each hit: 1. Determine local normal and material; 2. Resolve TE/TM polarization; 3. Apply Fresnel reflection/refraction; 4. Update direction, complex amplitude, phase, and polarization state; 5. Terminate on power, bounce, or scene-boundary criteria."

Prior to this change:
- Multi-bounce in `PhysicalOptics.cpp` iterated across all $N$ triangles in nested loops (`ia`, `ib`, `ic`), making $O(N^2)$ and $O(N^3)$ redundant BVH query calls even though `ray_bvh` directly returns the exact struck triangle.
- Configurable phase reference center `phase_ref` was ignored in multi-bounce propagation phases (only applied to single bounce and edge fringe).
- No structured SBR ray-tube data model or TE/TM polarization decomposition existed in `Ray.hpp` / `Ray.cpp`.

## Decision
1. Dedicated Coherent SBR Ray-Tube Tracking (`include/strikecem/solvers/Ray.hpp`, `src/strikecem/solvers/Ray.cpp`):
   - Defined `reflect_direction`: specular direction reflection $\mathbf{d} - 2(\mathbf{d}\cdot\mathbf{n})\mathbf{n}$.
   - Defined `reflect_electric_field`: PEC vector boundary condition $2(\mathbf{E}\cdot\mathbf{n})\mathbf{n} - \mathbf{E}$, reversing tangential E-field ($180^\circ$ phase shift) and preserving normal E-field.
   - Defined `decompose_polarization`: resolves local incident ray and normal into orthonormal TE/TM basis ($\hat{\mathbf{u}}_{\text{TE}} \propto \hat{\mathbf{k}}\times\mathbf{n}$, $\hat{\mathbf{u}}_{\text{TM}} = \hat{\mathbf{u}}_{\text{TE}}\times\hat{\mathbf{k}}$) with conserved power.
   - Defined `struct SbrRayHop` and `struct SbrRayPath` to record path trajectories, inter-facet segment lengths, intermediate field vectors, retroreflection status, and coherent path phase.
   - Implemented `trace_sbr_path` for general SBR tube propagation.

2. Multi-Bounce Performance & Phase Reference Fix (`src/strikecem/solvers/PhysicalOptics.cpp`):
   - Replaced nested $O(N^2)$ and $O(N^3)$ loops with direct BVH hit lookups:
     - Hop 1 strikes facet `ib = hit->tri`.
     - Hop 2 strikes facet `ic = hit2->tri`.
     This drops multi-bounce complexity to $O(N \log N)$ without altering numerical output.
   - Incorporated `phase_ref` into the entry, inter-bounce propagation, and exit phases:
     $$\phi = -k \hat{s}\cdot(\mathbf{c}_a - \mathbf{r}_{\text{ref}}) - k \sum \hat{d}_m\cdot(\mathbf{c}_{m+1} - \mathbf{c}_m) + k \hat{r}\cdot(\mathbf{c}_{\text{last}} - \mathbf{r}_{\text{ref}})$$
     Ensuring exact phase invariance under arbitrary coordinate translation.

3. Test Verification (`tests/test_ray.cpp`):
   - `Sbr.ReflectDirection`: verifies vector specular reflection at normal and 45-degree incidence.
   - `Sbr.ReflectElectricField`: validates tangential E-field reversal and normal E-field preservation.
   - `Sbr.PolarizationDecomposition`: verifies orthogonality and energy conservation ($|E_{\text{TE}}|^2 + |E_{\text{TM}}|^2 = \|\mathbf{E}\|^2$).
   - `Sbr.TraceDihedralTwoBounce`: validates 2-bounce corner-reflector retroreflection and exit visibility.
   - `Sbr.MultiBouncePhaseReference`: validates that shifting `phase_ref` shifts the multi-bounce phase by exactly $e^{-j k (\hat{r} - \hat{s})\cdot \mathbf{r}_{\text{ref}}}$.
   - All 305 tests passing including `Benchmark.TwoBounceDihedralAnalytic` and `Benchmark.TwoBounceTrihedralAnalytic`.

## Consequences
- Multi-bounce SBR tracing is now order-of-magnitude faster via direct BVH dispatch.
- Coherent phase referencing is consistent across single-bounce PO, fringe diffraction, and multi-bounce reflection paths.
- Foundation for advanced multi-bounce ray tubes is established with rigorous polarization decomposition.

# ADR-0020: Wavelength-Dependent Dynamic Meshing and Frequency Sweep Policy

## Status
Accepted

## Context
Upstream specification `FULL_IMPLEMENTATION.md` §4.4 states:
"For a wavelength-aware mesh, the target size is:
$\lambda = \frac{c}{f\sqrt{\epsilon_r\mu_r}}$,
$h_{\mathrm{smooth}} \le \alpha_{\mathrm{smooth}}\lambda$,
$h_{\mathrm{edge}} \le \alpha_{\mathrm{edge}}\lambda$.
Typical starting values are $\alpha_{\mathrm{smooth}} \in [1/10, 1/6]$ and $\alpha_{\mathrm{edge}} \approx 1/20$.
The frequency-sweep policy must be explicit:
- `mesh_at_max_frequency`: one mesh for the entire sweep;
- `mesh_per_frequency`: separate meshes and hashes;
- `user_mesh`: use supplied triangles and report resolution only."

In computational electromagnetics, evaluating Physical Optics on triangles that are electrically too coarse ($h > \lambda / 4$) fails to capture phase variation across the facet, producing inaccurate sidelobes and nulls. Conversely, requiring manual CAD remeshing for multi-octave frequency sweeps is labor-intensive and error-prone.

## Decision
1. Schema Extension (`schemas/config.schema.json`):
   - Added `mesh.frequency_policy` enum `["user_mesh", "mesh_at_max_frequency", "mesh_per_frequency"]` (default `"user_mesh"`).
   - Added `mesh.wavelength_adaptive` configuration:
     - `alpha_smooth`: double, default $1/6 \approx 0.16667$
     - `alpha_edge`: double, default $1/20 = 0.05$
     - `curvature_factor`: double, default $0.1$
   - Materialized defaults in `strikecem::resolve_defaults()`.

2. Conforming Remeshing Engine (`include/strikecem/geometry/Remesh.hpp`, `src/strikecem/geometry/Remesh.cpp`):
   - Implemented `compute_wavelength_m(freq_hz, eps_r, mu_r)` calculating medium wavelength $\lambda = c / (f \sqrt{\epsilon_r \mu_r})$.
   - Implemented conforming adaptive triangle subdivision `remesh_target_edge_length(mesh, h_target, max_depth)`:
     - Detects edges exceeding $h_{\text{target}}$.
     - Inserts shared edge midpoints preventing cracks and T-junctions.
     - Performs 1-to-4 red subdivision and 1-to-2/1-to-3 green bisection.
     - Strictly conserves total surface area ($< 10^{-12} \text{ m}^2$ error) and surface coplanarity.
     - Updates bounding box, max edge length metric, and canonical geometry SHA-256 hash.
   - Implemented `remesh_wavelength(mesh, resolved, frequencies_hz)` resolving sweep policy.

3. Integration in Mesh Loading Pipeline (`MeshLoader.cpp`):
   - Automatically applies wavelength-adaptive remeshing to file models (STL, OBJ) and procedural primitive assemblies upon load and cache retrieval when `frequency_policy` is active.

4. Test Suite (`tests/test_remesh.cpp`):
   - Wavelength calculations in free space and dielectric media.
   - Conforming subdivision resolution and strict area conservation.
   - Policy dispatch (`user_mesh` vs `mesh_at_max_frequency`).
   - End-to-end Physical Optics solution verifying flat plate theoretical broadside RCS $\sigma = 4\pi A^2 / \lambda^2$ within 5%.

## Consequences
- Guaranteed electromagnetic phase resolution across electrically large models.
- Fully automated frequency-dependent mesh refinement without requiring external CAD re-export.
- Complete backward compatibility: existing configurations default to `user_mesh`.

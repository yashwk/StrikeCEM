# ADR-0011: Spherical Equal-Area Sampling via Fibonacci Lattice and Quadrature Weights

## Status
Accepted

## Context
Upstream specification (`FULL_IMPLEMENTATION.md` §11.3) notes:
"Uniform angle steps oversample near the poles and do not produce equal-area samples. A future equal-area mode should support spherical quadrature weights when integrating total scattered energy or detecting global peaks."

Standard regular grids ($\Delta\phi \times \Delta\theta$) suffer from severe area distortion: facet differential area $\sin\theta\,d\theta\,d\phi$ shrinks to 0 at the poles, causing polar oversampling by orders of magnitude while under-representing equatorial directions. Consequently, direct summation of regular grid samples skews total scattered power and statistical metrics (median, mean, dynamic range).

## Decision
1. **Fibonacci Lattice Generator**:
   We implement a deterministic equal-area Fibonacci spiral lattice on the unit sphere $S^2$:
   $$z_i = z_{\max} - \frac{2i + 1}{2N} (z_{\max} - z_{\min}), \quad i \in [0, N-1]$$
   $$\phi_i = 2\pi \frac{i}{\phi_{\text{golden}}} \pmod{2\pi}, \quad \phi_{\text{golden}} = \frac{1 + \sqrt{5}}{2}$$
   $$\text{elevation}_i = \arcsin(z_i) \times \frac{180}{\pi}, \quad \text{azimuth}_i = \text{atan2}(y_i, x_i) \times \frac{180}{\pi}$$
2. **Quadrature Weights**:
   Every sample point is assigned an explicit equal-area quadrature weight:
   $$w_i = \frac{\Delta\Omega}{N} = \frac{2\pi (z_{\max} - z_{\min})}{N} \text{ steradians}$$
   For full sphere ($z \in [-1, 1]$), $\sum_{i=0}^{N-1} w_i = 4\pi$.
   For hemisphere ($z \in [0, 1]$), $\sum_{i=0}^{N-1} w_i = 2\pi$.
3. **Configuration Schema**:
   Extend `angles` in `schemas/config.schema.json` with a third variant `spherical`:
   ```json
   "spherical": {
     "type": "fibonacci",
     "num_points": 1000,
     "elevation_min_deg": -90.0,
     "elevation_max_deg": 90.0
   }
   ```
4. **Scattered Power Integration**:
   Provide `integrate_scattered_power()` and `compute_spherical_stats()` to compute total scattered cross section $\sigma_{\text{tot}} = \sum_i w_i \sigma_i$ and sphere-weighted average RCS $\bar{\sigma} = \frac{1}{4\pi} \sum_i w_i \sigma_i$.

## Consequences
- Enables unbiased whole-sphere and hemispherical RCS characterization without polar clustering.
- Enables accurate total energy and radar cross section integration on $S^2$.
- Fully compatible with the existing PO solver, pipeline, CSV/HDF5/SCEM output formats, and CLI runner.

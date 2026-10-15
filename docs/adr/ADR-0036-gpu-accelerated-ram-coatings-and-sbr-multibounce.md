# ADR-0036: GPU-Accelerated Radar Absorbing Material (RAM) Coatings and SBR Multi-Bounce Ray Tracing

## Status
Accepted

## Context
StrikeCEM previously provided GPU-accelerated Physical Optics (PO) execution via CUDA kernels (`GpuPO.cu`), achieving 50×–120× throughput speedups over multi-threaded CPU PO for bare PEC geometries. However, two critical high-frequency electromagnetic mechanisms remained restricted to CPU-only execution:
1. **Radar Absorbing Materials (RAM) & Dielectric Coatings**:
   - Leontovich Surface Impedance Boundary Conditions (SIBC, $\eta_s$).
   - Fresnel single-layer and multi-layer coated PEC recursive transfer-matrix formulations.
   Attempts to invoke the CUDA accelerator with `WallType::Impedance`, `WallType::Dielectric`, or `WallType::Coated` threw `CudaError("material walls are CPU-only in v1")`.
2. **Shooting & Bouncing Rays (SBR / Multi-Bounce)**:
   - High-order specular multi-bounce chains (2-bounce dihedral corner reflections, 3-bounce trihedral cavity interactions) and Geometric Optics (GO) self-shadowing were evaluated via CPU BVH traversal.
   Attempts to invoke the CUDA accelerator with `go.enabled == true` threw `CudaError("GO corrections are CPU-only in v1")`.

Modern stealth analysis of realistic combat platforms (e.g. stealth fighter airframes with radar-absorbent leading edges and deep engine inlet ducts; naval stealth vessels with multi-bounce superstructure reflections) requires evaluating materials and multi-bounce simultaneously at massive mesh scales (100k–1M+ facets) across hundreds of observation angles. Eliminating CPU fallback bottlenecks for RAM and SBR is essential for end-to-end GPU signature prediction.

## Decision

1. **GPU-Resident Device Complex Math Library**:
   - Implemented CUDA device arithmetic routines (`c_add`, `c_sub`, `c_mul`, `c_scale`, `c_div`, `c_sqrt`, `c_exp`) in `GpuPO.cu` operating directly on double-precision complex structures (`ComplexD`), providing bit-accurate parity with `std::complex<double>`.

2. **CUDA RAM & Coating Boundary Condition Kernels**:
   - Ported polarized electromagnetic reflection coefficient engines directly to device functions:
     - `dev_impedance_reflection`: Evaluates Leontovich SIBC reflection coefficients:
       $$\Gamma_{\text{TE}} = \frac{\eta_s \cos\theta_i - 1}{\eta_s \cos\theta_i + 1}, \quad \Gamma_{\text{TM}} = \frac{\eta_s - \cos\theta_i}{\eta_s + \cos\theta_i}$$
     - `dev_fresnel`: Evaluates dielectric interface reflection and transmission angles via Snell's law.
     - `dev_coated_pec_reflection`: Backward recurrence across multi-layer dielectric stacks over PEC backing, accounting for round-trip propagation phase $\phi_m = 2 \beta_m d_m$ and polarized boundary transitions.
   - Implemented `dev_material_facet_field`: Projects incident polarization into local facet TE/TM basis vectors, computes equivalent electric and magnetic surface currents ($\mathbf{J}_s, \mathbf{M}_s$), and performs far-field transverse radiation integration directly on the GPU.

3. **GPU-Resident Linear BVH & Ray Tracing**:
   - Constructed a flat device-side AABB BVH hierarchy (`GpuBvhNode`, `GpuBvhData`) with unpacked triangle vertices (`tri_vertices`).
   - Implemented branch-optimized device ray intersection functions:
     - `gpu_slab_hit`: Fast slab test for ray-AABB intersections.
     - `gpu_ray_triangle`: Möller-Trumbore ray-triangle intersection with barycentric bounds and minimum hit distance threshold ($t > 10^{-6}$).
     - `gpu_occluded_bvh`: Register-stack BVH traversal for line-of-sight occlusion and shadowing checks.
     - `gpu_ray_bvh`: Closest-hit BVH traversal with lowest-index tie breaking to match CPU traversal ordering bit-for-bit.

4. **GPU SBR Multi-Bounce Specular Engine**:
   - Integrated 2-bounce dihedral and 3-bounce trihedral ray tracing loops inside `po_kernel` and `po_mixed_kernel`:
     - Computes initial specular reflection vector $\mathbf{d}_1 = \mathbf{k}_i - 2(\mathbf{k}_i \cdot \hat{n}_a)\hat{n}_a$.
     - Performs GPU BVH closest-hit query to locate bounce facet $B$.
     - Evaluates exit ray $\mathbf{d}_2$ and checks retroreflection alignment ($\mathbf{d}_2 \cdot \hat{r} > 1 - 10^{-9}$ for monostatic backscatter).
     - Traces tertiary bounce facet $C$ for trihedral corners ($\mathbf{d}_3 \cdot \hat{r} > 1 - 10^{-9}$).
     - Accumulates multi-bounce phase shifts relative to the specified global phase center $\mathbf{r}_{\text{ref}}$:
       $$\Phi_{\text{2-bounce}} = k_0 \left( \hat{k}_i \cdot (\mathbf{c}_a - \mathbf{r}_{\text{ref}}) + |\mathbf{c}_b - \mathbf{c}_a| + \hat{k}_s \cdot (\mathbf{c}_b - \mathbf{r}_{\text{ref}}) \right)$$
     - Corrected electric field radiation amplitude normalization ($k / (4\pi) \cdot \text{area}$), ensuring exact $10^{-9}$ floating-point equivalence between CPU and GPU multi-bounce fields.

5. **API & Dispatch Unification**:
   - Updated `cuda::solve_po_cuda` and `cuda::solve_po_cuda_units` in `GpuPO.hpp` and `GpuPO_stub.cpp` to accept `const FringeOptions&`, `const GoOptions&`, and `const MaterialOptions&`.
   - Removed CPU fallback exceptions in `PhysicalOptics.cpp`, allowing full GPU execution of combined `cuda` + `go` + `material` jobs.

6. **Unit Test Verification (`tests/test_gpu.cpp`)**:
   - Added 8 dedicated unit tests validating GPU RAM and SBR against CPU ground truth:
     - `Gpu.ShadowingMatchesCpu`: Verifies exact match for self-shadowed plate geometries.
     - `Gpu.TwoBounceDihedralMatchesCpu`: Verifies 2-bounce dihedral corner backscatter matching CPU within $10^{-6}$.
     - `Gpu.TwoBounceDihedralAnalyticConvergence`: Confirms GPU multi-bounce backscatter matches the canonical analytic dihedral formula $\sigma = \frac{8\pi a^2 b^2}{\lambda^2}$ within $0.05\,\text{dB}$.
     - `Gpu.ThreeBounceTrihedralMatchesCpu`: Validates 3-bounce trihedral cavity specular backscatter matching CPU within $10^{-6}$.
     - `Gpu.ImpedanceRAMMatchesCpu`: Confirms Leontovich SIBC RAM coating on GPU matches CPU and demonstrates $>99\%$ RCS absorption compared to PEC.
     - `Gpu.DielectricMatchesCpu`: Validates lossy dielectric facet reflection on GPU against CPU.
     - `Gpu.CoatedRAMMatchesCpu`: Validates multi-layer coated PEC stacks on GPU against CPU.
     - `Gpu.MultiMaterialMatchesCpu`: Validates heterogeneous multi-material facet distributions on GPU against CPU.

## Consequences
- Full parity between CPU and CUDA Physical Optics solvers for all material types and multi-bounce SBR.
- Deep engine ducts, dihedral seams, and RAM-treated stealth coatings can now run at full GPU accelerator speed without host-device fallback overhead.
- All 366 unit, regression, and benchmark tests pass cleanly with zero failures.

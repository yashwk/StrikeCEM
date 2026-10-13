# ADR-0023: GPU Mixed-Precision Tiers and NVTX Profiling

## Status
Accepted

## Context
Upstream specification `FULL_IMPLEMENTATION.md` §7.4 and §15 state:
- "Precision modes: float32, float64, mixed (mixed uses float32 for geometry storage and float64 for phase and accumulation). Each mode must document memory footprint, speed, and accuracy impact."
- "NVTX markers: every GPU kernel launch, buffer transfer, and reduction must have an NVTX range for profiling in Nsight Systems."

Prior to this change:
- `solver.precision` supported only `"float32"` and `"float64"`.
- GPU VRAM footprint for large meshes in `float64` was $7 \times 8 = 56$ bytes per triangle (centroids, normals, areas).
- No NVTX annotations existed on GPU execution paths, making Nsight Systems / Nsight Compute profiling opaque.

## Decision
1. Schema Update (`schemas/config.schema.json`):
   - Added `"mixed"` to `solver.precision` enum options: `["float32", "float64", "mixed"]`.

2. Dedicated Mixed-Precision Kernel & Dispatch (`src/strikecem/solvers/GpuPO.cu`):
   - Preserved unmodified `po_kernel<Real>` for `float32` and `float64` execution paths to prevent any register allocation or FMA divergence on existing golden benchmarks.
   - Implemented `po_mixed_kernel`:
     - Geometry buffers (centroids, normals, areas) are stored and fetched from device VRAM as 32-bit single precision `float`, reducing geometry buffer footprint from 56 bytes/facet to 28 bytes/facet (50% VRAM memory reduction).
     - Phase calculations (`-k * k_hat . c` and `k * r_obs . c`), trigonometric calls (`fsincos`), polarization cross products, and transverse projector math operate in registers with 64-bit `double` precision.
     - Block-level shared memory parallel reduction and output accumulation buffers use `double` (`Cplx<double>`).
   - Implemented `solve_mixed_cuda`:
     - Device buffers `d_cent`, `d_norm`, `d_area` allocated as `float`.
     - `UnitParams<double>` and `Cplx<double>` used for problem units and field accumulation.
     - Host projection uses standard 64-bit complex math.
   - Updated `cuda_footprint_bytes` in both `GpuPO.cu` and `GpuPO_stub.cpp` to accurately compute 28 bytes/tri for `"mixed"`.

3. NVTX Profiling Instrumentation (`<nvtx3/nvToolsExt.h>`):
   - Added scoped NVTX ranges to both typed and mixed GPU execution pipelines:
     - `po_gpu_facet_upload`: host-to-device memory copy for triangle centroids, normals, and areas.
     - `po_gpu_batch`: outer batch iteration loop over direction/frequency units.
     - `po_gpu_kernel_launch`: kernel invocation and device synchronization.
     - `po_gpu_download_project`: device-to-host memory copy of accumulated fields and host polarization projection.

4. Test Suite (`tests/test_gpu.cpp`):
   - `Gpu.MatchesCpuMixedPrecision`: validates mixed-precision GPU PO against CPU float64 reference to $< 10^{-4}$ relative error.
   - `Gpu.MixedPrecisionFootprintReduction`: verifies that `cuda_footprint_bytes` reports 28 bytes/tri for mixed and float32 versus 56 bytes/tri for float64.

## Consequences
- High-frequency simulations of large faceted models can run with 50% device geometry footprint without sacrificing double-precision phase stability.
- Profiling with NVIDIA Nsight Systems yields structured timeline markers matching engine pipeline phases.

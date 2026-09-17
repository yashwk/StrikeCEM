# ADR-0002 — GPU PO as an opt-in accelerator backend

- **Status:** accepted (implementation)
- **Date:** 2026-09-17
- **Context:** the v1 contract is CPU-only; the roadmap promotes GPU PO
  with tolerance gates, device-specific benchmarks, explicit memory
  planning, and no silent precision/backend changes. The v1 schema can be
  extended without a version bump only by backward-compatible addition.

## Decision

1. Schema stays `scem_schema_version: 1.0`. `execution` gains three
   optional keys, all defaulted, so every v1.0 config still validates:
   - `accelerator`: `"cpu"` (default) or `"cuda"`;
   - `cuda_device_id`: integer >= 0, default 0;
   - `cuda_batch_units`: integer >= 0, default 0 meaning automatic batch
     selection from measured device memory at run time.
2. The GPU path consumes the same normalized mesh and sample plan as the
   CPU reference and accumulates in the configured precision. Requesting
   `cuda` when the binary lacks CUDA support, or when no device answers,
   is an explicit exit-4 refusal — never a silent CPU fallback, never a
   silent precision change.
3. Correctness is gated on complex-field agreement with the CPU reference
   (relative tolerance 1e-9 float64, 1e-3 float32, plus an absolute floor
   near nulls), not on RCS alone and never bitwise.
4. Provenance records the effective backend and device name in both the
   CSV sidecar and the HDF5 attributes. Benchmark profiles record backend
   and device; the estimator only matches profiles of the requested
   backend.
5. CI builds and tests the CPU path. The CUDA backend compiles when a
   toolkit is found (`SCEM_ENABLE_CUDA`, default auto) and its tests run
   wherever a CUDA device exists, skipping loudly otherwise. GPU
   verification hardware: RTX 4060 Laptop (sm_89), CUDA 13.4.
6. Quicklook plotting is not part of the core release; if wanted it lives
   in a separate tool reading the CSV/HDF5 output, never in the solver.

## Consequences

- Old configs are unaffected (defaults preserve CPU behavior bit-for-bit,
  covered by the determinism tests).
- A future multi-GPU or streams policy needs a new ADR; device selection
  stays a single index.

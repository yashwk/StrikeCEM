# ADR-0025: Benchmark Report Aggregation CLI Tool (`scem_benchreport`)

## Status
Accepted

## Context
Upstream specification `FULL_IMPLEMENTATION.md` §2.2 and §14.3 define:
- Tool scaffold in `tools/benchmark_report/` alongside `scem_quicklook` and `scem_compare`.
- §14.3 Scaling measurements: "Every backend gets scaling tests for: fixed facets, increasing samples; fixed samples, increasing facets; increasing polarization count; increasing frequency count; memory cap and batch-size changes; single-thread versus multi-thread; CPU versus accelerator where applicable. Performance claims are published as ranges tied to benchmark profiles."

Prior to this change:
- Profile generation existed via `strikecem run --bench-profile <path>`, but no aggregation or visualization tooling existed to process multiple profiles, compare CPU vs CUDA throughput, or generate scaling throughput curves.

## Decision
1. Core Benchmark Aggregation Engine (`include/strikecem/analysis/BenchmarkReport.hpp`, `src/strikecem/analysis/BenchmarkReport.cpp`):
   - Defined `BenchmarkEntry`: binds profile data with normalized evaluation throughput:
     $$\text{Throughput} = \frac{N_{\text{triangles}} \times N_{\text{samples}}}{t_{\text{wall}} \times 10^6} \quad (\text{M-Eval/s})$$
   - Defined `BenchmarkAggregation`: computes aggregate metrics including CPU average throughput, CUDA average throughput, peak throughput, and backend speedup ratios.
   - Implemented `render_benchmark_ascii`: renders ASCII horizontal bar chart comparison normalized across all profiles.
   - Implemented `benchmark_aggregation_to_json`: exports machine-readable JSON summary for CI/CD and regression gating.

2. CLI Executable (`tools/benchmark_report/main.cpp` -> `scem_benchreport`):
   - CLI flags: `--json`, `--ascii` (default), `--dir <path>`, and positional directory/file arguments.
   - Integrated into CMake build and install targets alongside `strikecem`, `scem_quicklook`, and `scem_compare`.

3. Test Verification (`tests/test_benchmarks.cpp`):
   - `BenchmarkReport.AggregateEmpty`: handles empty profile sets gracefully.
   - `BenchmarkReport.AggregateMetrics`: verifies mathematical throughput calculations and speedup ratios.
   - `BenchmarkReport.RenderAsciiAndJson`: validates ASCII visual chart and JSON serialization.
   - `BenchmarkReport.LoadFromDirectory`: verifies real profile parsing from `tools/benchmark_report/profiles`.

## Consequences
- Developers and CI pipelines have a native, single-command utility (`scem_benchreport`) to monitor solver throughput scaling across hardware and algorithms.
- Full parity achieved with upstream `tools/benchmark_report/` architectural requirement.

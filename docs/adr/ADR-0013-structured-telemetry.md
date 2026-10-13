# ADR-0013: Structured Diagnostics and Telemetry Profiling

## Status
Accepted

## Context
Upstream specification (`FULL_IMPLEMENTATION.md` §15) requires structured diagnostics and execution telemetry:
"Every run records:
- resolved plan and hashes;
- per-stage wall time;
- per-frequency and per-batch time;
- mesh and cache hit/miss information;
- memory high-water mark;
- warnings and fallback/refusal decisions;
- output commit progress."

In automated CI pipelines, cluster worker fleets, and mission-level RCS validation workflows, operators need machine-readable execution traces that record exact stage breakdown (mesh normalization, sample planning, PO execution, and file I/O), memory footprint (`VmHWM` / `ru_maxrss`), computational throughput (facet-evaluations/sec, directions/sec), and all convergence/warning annotations without perturbing numerical determinism.

## Decision
1. **Telemetry Collector (`TelemetryCollector`)**:
   Implement `strikecem::telemetry::TelemetryCollector` in `include/strikecem/telemetry/Telemetry.hpp`:
   - Hierarchical stage timers (`start_stage`, `end_stage`) with microsecond resolution using `std::chrono::steady_clock`;
   - OS-level memory profiling (`query_memory_metrics()`) querying `/proc/self/status` (`VmHWM`, `VmRSS`) and `getrusage()`;
   - Throughput counters (`samples_per_second`, `facets_per_second`, `facet_samples_per_second`);
   - Warning and provenance metadata capture;
   - Export structured JSON report conforming to the diagnostics contract.
2. **Schema Integration**:
   Extend `schemas/config.schema.json` under `output.telemetry`:
   ```json
   "telemetry": {
     "enabled": true,
     "path": "telemetry.json",
     "record_memory": true
   }
   ```
3. **CLI Integration**:
   When configured or run via CLI, telemetry is recorded through the pipeline and written atomically at solver completion.

## Consequences
- Produces complete, machine-actionable performance and memory telemetry.
- Zero numerical overhead in the hot PO computation loops.
- Provides hard memory bounds and throughput metrics for HPC cluster provisioning.

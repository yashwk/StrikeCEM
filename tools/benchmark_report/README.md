# Benchmark profiles

`estimate` reports a calibrated runtime range only when a profile matches
the planned run: same solver precision, CPU backend, and triangle/sample
counts each within 4x. Otherwise it reports an uncalibrated operation
count and says so.

## Recording a profile

```sh
strikecem run --bench-profile tools/benchmark_report/profiles/<name>.json <config>
```

The profile records solver wall time, triangle/sample counts, precision,
CPU brand, and tool identity (`scem-bench-1` format).

Record profiles from representative lit runs: a fully shadowed run
measures loop overhead, not solver throughput, and miscalibrates anything
it matches. Prefer electrically large cases with several thousand lit
facets and at least tens of milliseconds of solve time. GPU profiles work
the same way (`accelerator: cuda`); the estimator only matches profiles of
the requested backend, and each profile records its device.

## Lookup order

1. `$SCEM_BENCHMARK_DIR/*.json`
2. `./tools/benchmark_report/profiles/*.json` (this directory, committed seeds)

Unparseable files are skipped. A profile never changes solver behavior;
it only feeds the advisory estimator.

# ADR-0015: Python SDK and Runtime Bindings

## Status
Accepted

## Context
Upstream specifications (FULL_IMPLEMENTATION.md §§12, 17) require seamless integration between StrikeCEM's high-performance C ABI and Python-based aerospace mission simulation workflows (flight mechanics, mission trajectory simulators, 6-DOF models, RF telemetry).
The Python runtime interface must satisfy:
1. Pure standard library compatibility (`ctypes`) without requiring compilation of external Cython or heavy binary wheels on target deployment nodes.
2. Direct capability to open and query both binary transport `.scem`/`.scemdb` and scientific `.h5` databases.
3. Design verification matching design ID and revision against target model definitions.
4. High-throughput batch query execution supporting both standard Python lists of requests/tuples and vectorized NumPy arrays.
5. Fail-closed error handling with an explicit exception hierarchy (`DesignMismatchError`, `ChannelNotFoundError`, `DatabaseOpenError`, etc.).

## Decision
1. Implement the `strikecem` package under `python/strikecem`:
   - `strikecem.bindings`: loads `libStrikeCEM.so` (linked with `WHOLE_ARCHIVE` to export pure C ABI symbols defined in `strikecem_c_api.h`).
   - `strikecem.errors`: maps `scem_status_t` error codes into typed Python exceptions.
   - `strikecem.database`: exposes the `Database` class with context manager support, properties for design metadata, frequencies, polarizations, `verify_design`, `query`, `query_batch`, and `query_batch_numpy`.
2. Configure `pyproject.toml` with optional NumPy dependencies and setuptools packaging.
3. Integrate automatic Python SDK discovery and execution into CMake's `ctest` harness (`python_bindings` test).

## Consequences
- Aerospace simulation suites and data science scripts can interactively query and analyze RCS databases with microsecond latency.
- Seamless compatibility with both pure Python environments and scientific NumPy/SciPy computing pipelines.
- Standardized package structure ready for `pip install .`.

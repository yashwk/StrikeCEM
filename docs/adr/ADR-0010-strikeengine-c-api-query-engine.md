# ADR-0010: StrikeEngine Real-Time RCS Query Engine & C-API

## Status
Accepted

## Context
Per `FULL_IMPLEMENTATION.md` §18, `STRIKEDESIGNER_INTEGRATION.md`, and `OUTPUT_FORMAT.md`:
StrikeCEM generates offline electromagnetic databases (`.scem` binary and `.h5` HDF5). StrikeEngine (the simulation runtime powering missile seekers, radar systems, and 6-DOF vehicle dynamics) must consume these databases at simulation runtime:
1. **Low-Latency In-Memory Query:** A real-time radar seeker or EW simulation running at 100 Hz to 1 kHz requires sub-microsecond RCS lookups for arbitrary illumination angles $(\text{az}, \text{el})$, frequency $f$, and polarization channel.
2. **Stable, Portable C ABI:** StrikeEngine or external simulators (in C, C++, Rust, or Python) cannot rely on C++ compiler-dependent ABI symbol mangling or internal solver headers. A zero-dependency, C99-compatible header (`strikecem_c_api.h`) is mandatory.
3. **Asset Verification:** StrikeEngine must verify that the loaded database's `design_id` and `design_revision` match the target asset, rejecting stale or mismatched databases.
4. **Coherent Interpolation:** Coherent interpolation must operate on the complex scattering amplitude $(\text{Re}, \text{Im})$ before computing $\sigma_{\text{sqm}} = \text{Re}^2 + \text{Im}^2$ and $\sigma_{\text{dBsm}} = 10 \log_{10}(\sigma_{\text{sqm}})$, avoiding unphysical power sum errors near interference nulls.

## Decision
1. **Public C ABI Header:**
   Define `include/strikecem/api/strikecem_c_api.h` exporting pure C functions with `extern "C"` and portable visibility (`SCEM_API`):
   - `scem_database_open(const char* path, scem_database_t** out_db)`
   - `scem_database_close(scem_database_t* db)`
   - `scem_database_verify_design(scem_database_t* db, const char* expected_design_id, const char* expected_revision)`
   - `scem_database_get_frequencies(scem_database_t* db, double* out_freqs, size_t* count)`
   - `scem_database_get_polarizations(scem_database_t* db, char pols[][8], size_t* count)`
   - `scem_database_query(scem_database_t* db, double az_deg, double el_deg, double freq_hz, const char* pol, scem_query_result_t* out_result)`
   - `scem_database_query_batch(scem_database_t* db, const scem_query_request_t* requests, size_t count, scem_query_result_t* out_results)`
   - `scem_get_last_error()`

2. **In-Memory Spatial Indexing:**
   Implement `RcsQueryEngine` in C++ (`include/strikecem/api/RcsQueryEngine.hpp` / `src/strikecem/api/RcsQueryEngine.cpp`):
   - Fast channel lookup indexed by `(frequency, polarization)`.
   - Direction grid index detecting regular angular grids for $O(1)$ bilinear cell indexing.
   - Spatial KD-tree on unit direction vectors $\hat{k}$ on the sphere for irregular/adaptive sample distributions, executing in $O(\log N)$ time.
   - Bilinear angular interpolation on complex far-field components $(\text{Re}, \text{Im})$, followed by recalculation of $\sigma_{\text{sqm}}$ and $\sigma_{\text{dBsm}}$.

3. **Format Support & Provenance:**
   - Supports direct loading of both `.scem` (zero-copy binary transport) and `.h5` (HDF5).
   - Validates database completeness and integrity before allowing queries.
   - Extracts design identity metadata (`design_id`, `design_revision`, `export_id`).

4. **Integration:**
   - Exported directly from the shared library `libStrikeCEM.so`.

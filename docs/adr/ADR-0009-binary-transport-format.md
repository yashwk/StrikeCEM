# ADR-0009: Compact Binary Transport Format (.scem)

## Status
Accepted

## Context
Per `FULL_IMPLEMENTATION.md` §12.4 and §16.3:
While HDF5 provides an extensible hierarchical storage container and CSV provides portable plain-text inspectability, high-throughput distributed parameter sweeps, edge HPC clusters, and real-time StrikeEngine flight simulations require an ultra-compact, zero-overhead, contiguous binary transport format:
- Instantaneous deserialization without complex third-party runtime link overhead or CSV string parsing bottlenecks.
- Deterministic endianness, header magic, schema versioning, and payload integrity checksums (FNV-1a 64).
- Exact bit-level numerical round-trip preservation and numerical parity with HDF5 and CSV outputs.
- Comprehensive metadata encapsulation (JSON provenance, unit strings, hashes, solver parameters).

## Decision
1. **File Extension & Identification:**
   - Extension: `.scem`
   - Magic bytes: `SCEM` (`0x53, 0x43, 0x45, 0x4D`).
   - Format version: `1` (`uint16_t`).
   - Endianness marker: `0x01020304` (`uint32_t`). Reader verifies byte ordering.

2. **Binary Header Layout:**
   Fixed 256-byte header structure (`ScemHeader`, packed):
   - `magic[4]`: ASCII "SCEM"
   - `version`: uint16_t (1)
   - `header_size`: uint16_t (sizeof(ScemHeader) = 256 bytes)
   - `endianness_marker`: uint32_t (0x01020304)
   - `dtype`: uint8_t (0 for float32, 1 for float64)
   - `flags`: uint8_t
   - `reserved_u16`: uint16_t
   - `sample_count`: uint64_t
   - `frequency_count`: uint32_t
   - `direction_count`: uint32_t
   - `polarization_count`: uint16_t
   - `reserved_u16_2`: uint16_t
   - Offsets and byte counts:
     - `frequencies_offset`, `frequencies_bytes`: uint64_t
     - `directions_offset`, `directions_bytes`: uint64_t
     - `polarizations_offset`, `polarizations_bytes`: uint64_t
     - `metadata_offset`, `metadata_bytes`: uint64_t
     - `samples_offset`, `samples_bytes`: uint64_t
   - Checksums:
     - `header_checksum`: uint64_t (FNV-1a 64 over header fields up to header_checksum)
     - `data_checksum`: uint64_t (FNV-1a 64 over entire payload after header)
   - `reserved[124]`: padding to 256 bytes.

3. **Index & Metadata Tables:**
   - **Frequencies Table:** Contiguous `double` array of frequency values in Hz.
   - **Directions Table:** Contiguous records of `ScemDirectionRecord { double azimuth_deg; double elevation_deg; double k_x; double k_y; double k_z; }`.
   - **Polarizations Table:** Contiguous fixed 8-byte ASCII records `char name[8]`, null-padded.
   - **Metadata Block:** Canonical UTF-8 JSON payload encoding schema version, config hash, geometry/mesh hashes, unit definitions, timestamps, and solver configuration.

4. **Sample Data Rows:**
   Packed contiguous sample array ordered by `sample_id`:
   - Float64 row (`ScemSampleRowF64`, 52 bytes):
     - `sample_id`: uint64_t
     - `direction_index`: uint32_t
     - `frequency_index`: uint32_t
     - `polarization_index`: uint16_t
     - `valid`: uint8_t (1 = valid, 0 = invalid/pending)
     - `status_code`: uint8_t (0 = pending, 1 = ok, 2 = warning, 3 = error)
     - `scattering_real`: double
     - `scattering_imag`: double
     - `rcs_sqm`: double
     - `rcs_dbsm`: double
   - Float32 row (`ScemSampleRowF32`, 36 bytes):
     - Same index fields, with float32 values for `scattering_real`, `scattering_imag`, `rcs_sqm`, `rcs_dbsm`.

5. **Reader Verification & Fail-Closed Semantics:**
   - Reader verifies magic bytes, major version, endianness marker, and header checksum.
   - Reader computes FNV-1a 64 over payload and verifies exact match against `data_checksum`.
   - Rejects corrupted checksums, truncated files, mismatched sample counts, non-sequential IDs, or incomplete results.

6. **Integration:**
   - Config format option: `"format": "scem"` in `config.json`.
   - CLI command support for direct run and shard merges (`strikecem merge`).
   - Round-trip bit-level and numerical parity testing against HDF5 and CSV references.

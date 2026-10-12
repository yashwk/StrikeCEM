# ADR-0006 — Distributed Shards and Batch Merger

- **Status:** accepted (approach)
- **Date:** 2026-10-13
- **Context:** Large industrial radar cross section campaigns encompass tens of
  thousands of directions and frequencies. Running these as a single monolithic
  process is vulnerable to node failures and underutilizes compute clusters.
  `FULL_IMPLEMENTATION.md` §13.2 and `BACKLOG.md` require deterministically
  partitioned batch manifests based on global sample IDs and checksums, with a
  fail-closed merge utility. Floating-point angle strings must never serve as
  merge keys; numerical identity is pinned to global `sample_id` and canonical
  direction vectors.

## Decision

1. **Deterministic Unit Partitioning**:
   The sample plan of $N = N_f \times N_d \times N_p$ samples comprises $U = N_f \times N_d$
   independent solve units. Given `num_shards` $S$ and `shard_id` $s \in [0, S-1]$,
   units are partitioned deterministically:
   $$\text{base} = \lfloor U / S \rfloor, \quad \text{rem} = U \pmod S$$
   Shard $s$ receives count $C_s = \text{base} + (s < \text{rem} ? 1 : 0)$ and
   offset $O_s = s \cdot \text{base} + \min(s, \text{rem})$.
   Global `sample_id` values remain $(u \cdot N_p + p_i)$, preserving global
   identity and index arithmetic.

2. **Configuration Contract**:
   A `run.shard` block is added to `schemas/config.schema.json`:
   ```json
   "shard": {
     "type": "object",
     "required": ["shard_id", "num_shards"],
     "properties": {
       "shard_id": {"type": "integer", "minimum": 0},
       "num_shards": {"type": "integer", "minimum": 1},
       "run_id": {"type": "string"},
       "worker_identity": {"type": "string"}
     }
   }
   ```
   When omitted, `shard_id = 0, num_shards = 1` represents an unpartitioned run.

3. **Manifest Format (`shard_manifest.json`)**:
   Produced by `strikecem shard`:
   - `scem_manifest_version`: `"1.0"`
   - `run_id`: UUID or stable run identifier
   - `config_hash`, `geometry_hash`, `normalized_mesh_hash`, `sample_plan_hash`
   - `total_samples`, `total_units`, `num_shards`
   - Array of shard descriptors with `shard_id`, `unit_start`, `unit_count`,
     `sample_id_start`, `sample_id_count`, `output_path`, `output_format`.

4. **Fail-Closed Merger (`strikecem merge`)**:
   `strikecem merge` accepts either a manifest file or a list of shard output files.
   It verifies:
   - Complete coverage: every sample ID in $[0, N-1]$ must be present.
   - Non-overlapping: duplicate sample IDs trigger immediate error.
   - Provenance compatibility: matching `config_hash`, `geometry_hash`,
     `normalized_mesh_hash`, and `sample_plan_hash`.
   - Data integrity: committed chunk checksums are re-verified.
   - If any shard is missing, corrupt, or mismatched, the merger fails closed.

5. **Output Provenance**:
   - Shard outputs record `shard_id`, `num_shards`, `run_id`, `sample_id_start`,
     and `sample_id_count` in sidecar JSON and HDF5 attributes.
   - Merged outputs record `merged_shards` count and `run_id` in provenance.

## Consequences

- Backward compatibility: unpartitioned runs are unaffected. Direct runs produce
  bit-identical results to merged shard runs.
- Shards can be distributed across Slurm, Kubernetes, or multi-node clusters.
- Worker identity and run IDs provide distributed traceability.

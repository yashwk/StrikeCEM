# ADR-0028: Distributed Cluster Orchestrator & Local Multi-Device Parallel Runner

## Status
Accepted

## Context
Industrial radar cross section campaigns encompass tens of thousands of directions, multiple frequency bands, and polarizations. Running monolithic single-thread or single-device processes fails to leverage multi-core CPU architectures, multi-GPU workstations, and distributed high-performance computing (HPC) clusters or cloud environments:
- ADR-0006 established the deterministic sample-plan partitioning math, shard manifest schema, and fail-closed merger.
- Full platform roadmap completion (`FULL_IMPLEMENTATION.md` Stage D and `BACKLOG.md`) required:
  1. A unified local multi-device parallel runner (`strikecem run-parallel`) with automatic shard generation, device scheduling, and reduction.
  2. An autonomous HTTP REST campaign coordinator broker (`strikecem coordinate`).
  3. A headless, self-healing worker daemon (`strikecem worker`) supporting heartbeat leases and SHA-256 result verification.
  4. Native scheduler script generators for Slurm job arrays and Kubernetes batch jobs (`strikecem cluster generate-slurm` and `generate-k8s`).

## Decision
1. **Local Multi-Device Parallel Runner (`strikecem run-parallel`)**:
   - Implemented `LocalParallelRunner` (`strikecem::run_local_parallel`).
   - Automatically computes deterministic solve-unit partitioning across configured CPU threads or CUDA devices (e.g. `--devices cuda:0,cuda:1`).
   - Stages temporary shard configurations, dispatches parallel workers in a concurrent execution pool, and applies fail-closed merge reduction into the final target database.
   - Guaranteed bit-identical outputs to monolithic single-process executions.

2. **Distributed Campaign Coordinator Broker (`strikecem coordinate`)**:
   - Implemented lightweight, self-contained POSIX socket HTTP REST coordinator server (`CoordinatorServer`) with zero third-party networking library dependencies.
   - Exposes REST endpoints:
     - `GET /api/v1/health`: Broker status and campaign run identifier.
     - `GET /api/v1/status`: Comprehensive campaign telemetry (pending, leased, completed shard counts, progress percentage).
     - `POST /api/v1/task/acquire`: Leases next pending shard to requesting worker with configurable lease duration.
     - `POST /api/v1/task/heartbeat`: Renews active shard leases during long-running solves.
     - `POST /api/v1/task/complete`: Records shard completion with SHA-256 checksum verification and triggers automated fail-closed reduction upon 100% completion.
     - `POST /api/v1/task/fail`: Resets task state for resilient retry upon worker or node crash.
   - Fault tolerance: automatically reclaims expired leases when worker nodes disconnect or fail to send heartbeats.
   - Supports ephemeral port binding (`port = 0`) via `getsockname()` to prevent port collisions in parallel CI environments.

3. **Headless Worker Daemon (`strikecem worker`)**:
   - Implemented `WorkerDaemon` (`WorkerDaemon::run()`).
   - Executes autonomous pull-execute-commit loop with background heartbeat thread synchronized via condition variable.
   - Calculates cryptographic SHA-256 digest of generated shard output database using OpenSSL EVP before committing task completion to coordinator.
   - Supports device overrides (e.g. assigning `--device cuda:1` or `cpu`).

4. **Cluster Scheduler Generators (`strikecem cluster`)**:
   - Implemented `ClusterScheduler` generators:
     - `strikecem cluster generate-slurm`: generates production `#SBATCH --array=0-N` Slurm submission scripts with partition, CPU, GPU, and time limit directives.
     - `strikecem cluster generate-k8s`: generates Kubernetes `batch/v1` `Job` YAML manifests with specified container image, coordinator URL, and worker parallelism.

5. **Thread-Safe HDF5 Serialization**:
   - Created `Hdf5Lock` (`hdf5_global_mutex()`) to serialize HDF5 C/C++ library operations during multi-threaded shard output creation and reading, protecting platforms where system `libhdf5` is not compiled with `--enable-threadsafe`.

6. **Test Verification**:
   - `LocalParallelRunner.DirectVersusParallelExecutionMatch`: asserts 3-worker parallel run produces bit-identical sample values to monolithic run.
   - `LocalParallelRunner.CliCommandDispatch`: validates CLI argument parsing and execution.
   - `Coordinator.HttpRestEndpoints`: validates `/health`, `/status`, task acquire, heartbeat, and completion.
   - `Coordinator.WorkerHeartbeatAndLeaseExpiry`: verifies expired leases reset to Pending state.
   - `Worker.AutonomousExecutionLoop`: verifies multi-worker autonomous task consumption, heartbeat keepalive, and auto-merge.
   - `ClusterScheduler.SlurmAndK8sGeneration`: verifies generated Slurm job arrays and Kubernetes Job manifests.

## Consequences
- Full closure of Domain 4 and complete fulfillment of `FULL_IMPLEMENTATION.md` and `BACKLOG.md` distributed computing specifications.
- Enables single-command local multi-GPU runs (`strikecem run-parallel`) as well as scalable multi-node cluster campaigns.
- 100% test pass rate across 323 test cases in the test suite with zero regressions.

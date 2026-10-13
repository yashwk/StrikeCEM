# Tutorial: Distributed Cluster & Parallel Orchestration Campaigns

This tutorial explains how to scale StrikeCEM across multi-core CPUs, multi-GPU workstations, HPC clusters (Slurm), and cloud infrastructure (Kubernetes) ([ADR-0028](file:///home/yashwanth/projects/StrikeCEM/docs/adr/ADR-0028-distributed-cluster-and-parallel-orchestration.md)).

---

## 1. Local Multi-Device Parallel Execution (`run-parallel`)

When running large campaigns on a single workstation or server with multiple CPU cores or multiple GPUs, use `strikecem run-parallel`:

```bash
strikecem run-parallel configs/full_sphere_xband.json \
  --output final_database.h5 \
  --workers 4 \
  --devices cuda:0,cuda:1 \
  --work-dir /tmp/scem_shards
```

### What Happens Automatically:
1. **Deterministic Partitioning**: Partitions the sample plan into balanced units across the 4 workers.
2. **Device Assignment**: Schedules workers across the specified CUDA devices (`cuda:0` and `cuda:1`) or CPU threads.
3. **Concurrent Execution**: Dispatches worker processes in a concurrent pool.
4. **Fail-Closed Merge Reduction**: Automatically validates chunk checksums and merges completed shard databases into `final_database.h5`.
5. **Bit-Identical Guarantee**: Monolithic runs and parallel runs produce 100% bit-identical results.

---

## 2. Distributed HTTP REST Campaign Coordinator (`coordinate` & `worker`)

For multi-node cluster networks without a shared filesystem or centralized job queue, StrikeCEM provides a lightweight, self-contained HTTP REST coordinator and headless worker daemon.

```
       [Worker Node 1] ─── POST /api/v1/task/acquire ───┐
       [Worker Node 2] ─── POST /api/v1/task/heartbeat ─┼──> [Coordinator Broker]
       [Worker Node N] ─── POST /api/v1/task/complete ──┘       (:8080)
```

### Step 1: Create Shards & Manifest

```bash
strikecem shard configs/campaign.json --num-shards 64 --out-dir ./shards
```

This creates 64 shard configs and `shards/manifest.json`.

### Step 2: Start the Coordinator Broker

On the coordinator master node:

```bash
strikecem coordinate --manifest ./shards/manifest.json \
  --output ./campaign_merged.h5 \
  --port 8080 \
  --lease 300 \
  --auto-exit
```

- `--lease 300`: Each shard is leased for 300 seconds. If a worker node crashes or disconnects, the lease expires and the shard is automatically returned to the pending queue.
- `--auto-exit`: Coordinator terminates cleanly after 100% of shards are solved and merged.

### Step 3: Launch Headless Workers

On any worker machines or container instances:

```bash
strikecem worker --coordinator http://master-ip:8080 --device cuda:0
```

The worker autonomously:
1. Pulls the next pending shard.
2. Runs the simulation with an interruptible heartbeat keepalive thread.
3. Computes cryptographic SHA-256 checksums of output data.
4. Reports completion back to the coordinator.
5. Loops until all tasks are finished.

---

## 3. Slurm HPC Cluster Submission (`cluster generate-slurm`)

On traditional supercomputing clusters managed by Slurm, StrikeCEM can generate production `#SBATCH` job array scripts:

```bash
strikecem cluster generate-slurm \
  --manifest shards/manifest.json \
  --output submit_campaign.slurm \
  --partition gpu \
  --cpus 8 \
  --gpus 1 \
  --time 02:00:00 \
  --job-name scem_sweep
```

Submit directly to Slurm:

```bash
sbatch submit_campaign.slurm
```

When all array tasks finish, merge the output with `strikecem merge`:

```bash
strikecem merge shards/manifest.json --output final_campaign.h5
```

---

## 4. Kubernetes Cloud Batch Jobs (`cluster generate-k8s`)

For cloud-native deployments (AWS EKS, GCP GKE, Azure AKS), generate a standard Kubernetes `batch/v1` `Job` manifest:

```bash
strikecem cluster generate-k8s \
  --manifest shards/manifest.json \
  --output k8s_job.yaml \
  --image strikecem:latest \
  --coordinator http://scem-coordinator.default.svc.cluster.local:8080 \
  --parallelism 16 \
  --completions 64
```

Apply to your cluster:

```bash
kubectl apply -f k8s_job.yaml
```

Kubernetes spins up 16 parallel worker pods that consume tasks from the coordinator until all 64 shards are resolved and merged.

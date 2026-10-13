#pragma once
// Cluster scheduler job script and manifest generators (Domain 4).
// Supports Slurm job arrays and Kubernetes Jobs for StrikeCEM distributed campaigns.
#include <string>

namespace strikecem {

struct SlurmOptions {
    std::string manifest_path;
    std::string output_path;
    std::string partition = "compute";
    int cpus_per_task = 4;
    std::string gpus_per_task = "";
    std::string time_limit = "04:00:00";
    std::string job_name = "strikecem_campaign";
    std::string strikecem_bin = "strikecem";
};

struct K8sOptions {
    std::string manifest_path;
    std::string output_path;
    std::string image_name = "strikecem:latest";
    std::string coordinator_url = "http://strikecem-coordinator:8080";
    int parallelism = 4;
    int completions = 0; // 0 = match manifest shards or default
    std::string job_name = "strikecem-worker-job";
    std::string cpu_request = "2";
    std::string memory_request = "4Gi";
};

// Generates a Slurm batch array submission script (.slurm or .sh) from a shard manifest.
void generate_slurm_script(const SlurmOptions& options);

// Generates a Kubernetes Job YAML manifest (.yaml) for distributed worker execution.
void generate_k8s_manifest(const K8sOptions& options);

} // namespace strikecem

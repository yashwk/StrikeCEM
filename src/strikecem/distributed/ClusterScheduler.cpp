#include "strikecem/distributed/ClusterScheduler.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

#include "strikecem/io/OutputError.hpp"

namespace fs = std::filesystem;

namespace strikecem {

void generate_slurm_script(const SlurmOptions& options) {
    if (options.manifest_path.empty()) {
        throw OutputError("Slurm generator requires --manifest path");
    }
    if (options.output_path.empty()) {
        throw OutputError("Slurm generator requires --output path");
    }
    if (!fs::exists(options.manifest_path)) {
        throw OutputError("Manifest file does not exist: " + options.manifest_path);
    }

    std::ifstream manifest_in(options.manifest_path);
    if (!manifest_in) {
        throw OutputError("Cannot open manifest file: " + options.manifest_path);
    }

    nlohmann::json manifest_json;
    try {
        manifest_in >> manifest_json;
    } catch (const std::exception& e) {
        throw OutputError("Failed to parse manifest JSON: " + std::string(e.what()));
    }

    const size_t num_shards = manifest_json.value("num_shards", size_t(1));
    const std::string run_id = manifest_json.value("run_id", options.job_name);
    const size_t max_idx = (num_shards > 0) ? (num_shards - 1) : 0;

    const fs::path abs_manifest = fs::absolute(options.manifest_path);
    const fs::path manifest_dir = abs_manifest.parent_path();

    std::ofstream out(options.output_path);
    if (!out) {
        throw OutputError("Cannot create Slurm output script: " + options.output_path);
    }

    out << "#!/usr/bin/env bash\n"
        << "# ==============================================================================\n"
        << "# StrikeCEM Slurm Job Array Script\n"
        << "# Run ID: " << run_id << "\n"
        << "# Shards: " << num_shards << " (array indices 0-" << max_idx << ")\n"
        << "# Generated automatically by strikecem cluster generate-slurm\n"
        << "# ==============================================================================\n"
        << "#SBATCH --job-name=" << (options.job_name.empty() ? run_id : options.job_name) << "\n"
        << "#SBATCH --array=0-" << max_idx << "\n"
        << "#SBATCH --partition=" << options.partition << "\n"
        << "#SBATCH --cpus-per-task=" << options.cpus_per_task << "\n"
        << "#SBATCH --time=" << options.time_limit << "\n"
        << "#SBATCH --output=" << run_id << "_%A_%a.log\n"
        << "#SBATCH --error=" << run_id << "_%A_%a.err\n";

    if (!options.gpus_per_task.empty()) {
        out << "#SBATCH --gres=gpu:" << options.gpus_per_task << "\n";
    }

    out << "\nset -euo pipefail\n\n"
        << "MANIFEST_DIR=\"" << manifest_dir.string() << "\"\n"
        << "STRIKECEM_BIN=\"" << options.strikecem_bin << "\"\n\n"
        << "SHARD_INDEX=$(printf \"%04d\" ${SLURM_ARRAY_TASK_ID})\n"
        << "SHARD_CONFIG=\"${MANIFEST_DIR}/shard_${SHARD_INDEX}.json\"\n\n"
        << "echo \"[StrikeCEM Slurm] Starting Shard ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)\"\n"
        << "echo \"[StrikeCEM Slurm] Config: ${SHARD_CONFIG}\"\n\n"
        << "if [ ! -f \"${SHARD_CONFIG}\" ]; then\n"
        << "    echo \"[StrikeCEM Slurm] Error: Shard config not found: ${SHARD_CONFIG}\" >&2\n"
        << "    exit 1\n"
        << "fi\n\n"
        << "\"${STRIKECEM_BIN}\" run \"${SHARD_CONFIG}\"\n\n"
        << "echo \"[StrikeCEM Slurm] Shard ${SLURM_ARRAY_TASK_ID} completed successfully at $(date)\"\n";

    out.close();
}

void generate_k8s_manifest(const K8sOptions& options) {
    if (options.output_path.empty()) {
        throw OutputError("Kubernetes generator requires --output path");
    }

    size_t completions = options.completions;
    if (completions == 0 && !options.manifest_path.empty() && fs::exists(options.manifest_path)) {
        try {
            std::ifstream mf(options.manifest_path);
            nlohmann::json j;
            mf >> j;
            completions = j.value("num_shards", size_t(options.parallelism));
        } catch (...) {
            completions = options.parallelism;
        }
    }
    if (completions == 0) {
        completions = static_cast<size_t>(options.parallelism);
    }

    std::ofstream out(options.output_path);
    if (!out) {
        throw OutputError("Cannot create Kubernetes output manifest: " + options.output_path);
    }

    out << "apiVersion: batch/v1\n"
        << "kind: Job\n"
        << "metadata:\n"
        << "  name: " << options.job_name << "\n"
        << "  labels:\n"
        << "    app.kubernetes.io/name: strikecem-worker\n"
        << "    app.kubernetes.io/component: batch-worker\n"
        << "spec:\n"
        << "  parallelism: " << options.parallelism << "\n"
        << "  completions: " << completions << "\n"
        << "  backoffLimit: 4\n"
        << "  template:\n"
        << "    metadata:\n"
        << "      labels:\n"
        << "        app.kubernetes.io/name: strikecem-worker\n"
        << "    spec:\n"
        << "      restartPolicy: OnFailure\n"
        << "      containers:\n"
        << "      - name: strikecem-worker\n"
        << "        image: " << options.image_name << "\n"
        << "        args:\n"
        << "        - \"worker\"\n"
        << "        - \"--coordinator\"\n"
        << "        - \"" << options.coordinator_url << "\"\n"
        << "        resources:\n"
        << "          requests:\n"
        << "            cpu: \"" << options.cpu_request << "\"\n"
        << "            memory: \"" << options.memory_request << "\"\n";

    out.close();
}

} // namespace strikecem

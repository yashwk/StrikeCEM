#include "strikecem/distributed/LocalParallelRunner.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>

#include "strikecem/app/CLI.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/io/OutputError.hpp"

namespace fs = std::filesystem;

namespace strikecem {

LocalParallelResult run_local_parallel(const LocalParallelOptions& options,
                                      const std::string& schema_path) {
    const auto start_time = std::chrono::steady_clock::now();
    LocalParallelResult result;

    if (!fs::exists(options.config_path)) {
        result.error_message = "Config file does not exist: " + options.config_path;
        return result;
    }

    try {
        const ResolvedConfig rc = load_config(options.config_path, schema_path);
        const SamplePlan plan = build_sample_plan(rc.value);
        const fs::path cfg_dir = fs::path(options.config_path).parent_path();
        const NormalizedMesh mesh = load_normalized_mesh(rc.value, cfg_dir, rc.schema_version);

        // Determine final output path
        std::string final_out = options.output_path;
        if (final_out.empty()) {
            final_out = rc.value["output"]["path"].get<std::string>();
        }
        result.final_output_path = final_out;

        // Determine number of workers
        size_t workers = options.num_workers;
        if (workers == 0) {
            if (!options.devices.empty()) {
                workers = options.devices.size();
            } else {
                workers = std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
            }
        }
        // Shard count cannot exceed total units
        const size_t total_units = plan.frequencies_hz.size() * plan.directions.size();
        workers = std::max<size_t>(1, std::min(workers, total_units));

        // Create staging directory for shards
        std::ostringstream ss_run_id;
        ss_run_id << "parallel_" << ::getpid() << "_"
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         start_time.time_since_epoch()).count();
        const std::string run_id = ss_run_id.str();

        fs::path staging_dir;
        if (!options.work_dir.empty()) {
            staging_dir = fs::path(options.work_dir) / run_id;
        } else {
            staging_dir = fs::temp_directory_path() / run_id;
        }
        fs::create_directories(staging_dir);

        // Partition into deterministic shards
        const ShardManifest manifest = create_shards(
            rc, plan, mesh, options.config_path, workers, staging_dir.string(), run_id);
        result.num_shards = manifest.shards.size();

        // If specific device assignments were requested, update each shard config
        if (!options.devices.empty()) {
            for (size_t s = 0; s < manifest.shards.size(); ++s) {
                const std::string& dev = options.devices[s % options.devices.size()];
                std::ifstream in(manifest.shards[s].config_path);
                if (in) {
                    nlohmann::json j;
                    in >> j;
                    in.close();

                    if (dev.rfind("cuda:", 0) == 0) {
                        j["execution"]["accelerator"] = "cuda";
                        j["execution"]["cuda_device_id"] = std::stoi(dev.substr(5));
                    } else if (dev == "cuda") {
                        j["execution"]["accelerator"] = "cuda";
                        j["execution"]["cuda_device_id"] = 0;
                    } else {
                        j["execution"]["accelerator"] = "cpu";
                    }

                    std::ofstream out(manifest.shards[s].config_path);
                    out << j.dump(2) << "\n";
                }
            }
        }

        // Spawn parallel workers
        std::vector<std::thread> thread_pool;
        std::mutex mtx;
        std::exception_ptr first_exc = nullptr;

        for (size_t s = 0; s < manifest.shards.size(); ++s) {
            const std::string shard_cfg = manifest.shards[s].config_path;
            thread_pool.emplace_back([shard_cfg, schema_path, &mtx, &first_exc, &result]() {
                try {
                    std::vector<std::string> args = {"strikecem", "run", shard_cfg};
                    if (!schema_path.empty()) {
                        args.push_back("--schema");
                        args.push_back(schema_path);
                    }
                    std::vector<char*> c_argv;
                    for (auto& a : args) c_argv.push_back(a.data());

                    const int rc_code = cli::run(static_cast<int>(c_argv.size()), c_argv.data());
                    if (rc_code != 0) {
                        throw OutputError("Shard execution failed with exit code " + std::to_string(rc_code));
                    }
                    std::lock_guard<std::mutex> lock(mtx);
                    result.completed_shards++;
                } catch (...) {
                    std::lock_guard<std::mutex> lock(mtx);
                    if (!first_exc) {
                        first_exc = std::current_exception();
                    }
                }
            });
        }

        for (auto& th : thread_pool) {
            if (th.joinable()) th.join();
        }

        if (first_exc) {
            std::rethrow_exception(first_exc);
        }

        // Merge shards into target final output
        const fs::path manifest_path = staging_dir / "shard_manifest.json";
        merge_shards(manifest_path.string(), {}, final_out, schema_path);

        // Clean up temporary shards unless requested to keep
        if (!options.keep_shards) {
            std::error_code ec;
            fs::remove_all(staging_dir, ec);
        }

        result.success = true;
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    const auto end_time = std::chrono::steady_clock::now();
    result.wall_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    return result;
}

} // namespace strikecem

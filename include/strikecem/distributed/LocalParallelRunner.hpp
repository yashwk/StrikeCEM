#pragma once
// Local multi-device / multi-worker parallel runner for StrikeCEM (Domain 4).
// Automatically creates deterministic shards, executes across local CPU/GPU workers,
// and merges into a unified final output database.
#include <cstddef>
#include <string>
#include <vector>

#include "strikecem/distributed/Shard.hpp"

namespace strikecem {

struct LocalParallelOptions {
    std::string config_path;
    std::string output_path;
    size_t num_workers = 0; // 0 = auto-detect hardware concurrency
    std::vector<std::string> devices; // Optional specific devices (e.g. "cuda:0", "cpu")
    std::string work_dir; // Optional staging directory for shards
    bool keep_shards = false;
};

struct LocalParallelResult {
    bool success = false;
    size_t num_shards = 0;
    size_t completed_shards = 0;
    double wall_time_seconds = 0.0;
    std::string final_output_path;
    std::string error_message;
};

LocalParallelResult run_local_parallel(const LocalParallelOptions& options,
                                      const std::string& schema_path = "");

} // namespace strikecem

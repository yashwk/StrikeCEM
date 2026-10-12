#pragma once
// Distributed batch execution: deterministic shard partitioning and fail-closed merge.
// Follows FULL_IMPLEMENTATION.md §13.2 and BACKLOG.md.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace strikecem {

struct ShardDescriptor {
    size_t shard_id = 0;
    size_t unit_start = 0;
    size_t unit_count = 0;
    uint64_t sample_id_start = 0;
    uint64_t sample_id_count = 0;
    std::string config_path;
    std::string output_path;
};

struct ShardManifest {
    std::string version = "1.0";
    std::string run_id;
    std::string config_path;
    std::string config_hash;
    std::string geometry_hash;
    std::string normalized_mesh_hash;
    std::string sample_plan_hash;
    std::string output_format;
    size_t total_units = 0;
    uint64_t total_samples = 0;
    size_t num_shards = 0;
    std::vector<ShardDescriptor> shards;
};

// Generates N shard configuration files and a master manifest.
ShardManifest create_shards(const ResolvedConfig& rc, const SamplePlan& plan,
                            const NormalizedMesh& mesh, const std::string& config_path,
                            size_t num_shards, const std::string& out_dir,
                            const std::string& run_id = "");

// Merges completed shard files (from manifest or file list) into a unified final database.
// Throws OutputError on missing shards, duplicate sample IDs, or hash mismatch.
void merge_shards(const std::string& manifest_or_shard_path,
                  const std::vector<std::string>& additional_shards,
                  const std::string& output_path,
                  const std::string& schema_path = "");

} // namespace strikecem

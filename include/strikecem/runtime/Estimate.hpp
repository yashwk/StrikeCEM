#pragma once
// v1 resource estimator (SPEC FR-7): advisory memory/operation counts from
// actual mesh and sample counts, plus benchmark-calibrated runtime ranges.
// The solver re-checks limits after loading the mesh; estimate never refuses.
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"

namespace strikecem {

struct BenchmarkProfile {
    std::string id; // file stem
    std::string precision;
    std::string backend = "cpu";
    uint64_t triangles = 0;
    uint64_t samples = 0; // output rows
    double wall_seconds = 0.0;
    std::string cpu;
};

// Loads *.json profiles from $SCEM_BENCHMARK_DIR and
// ./tools/benchmark_report/profiles; unreadable files are skipped.
std::vector<BenchmarkProfile> load_benchmark_profiles();

// Records a profile for a completed run (see `run --bench-profile`).
void write_benchmark_profile(const std::string& path, const ResolvedConfig& rc,
                             uint64_t triangles, uint64_t samples, double wall_seconds,
                             const std::string& backend, const std::string& device);

struct ResourceEstimate {
    uint64_t mesh_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t thread_bytes = 0;
    uint64_t total_bytes = 0;
    double limit_bytes = 0.0;
    bool fits = true;
    uint64_t operation_count = 0;
    double electrical_size_wavelengths = 0.0; // bbox diagonal / min wavelength
    std::string backend = "cpu";
    uint64_t device_bytes = 0; // CUDA resident footprint (0 for CPU runs)
    bool calibrated = false;
    double runtime_lo_s = 0.0;
    double runtime_hi_s = 0.0;
    std::string profile_id;
    std::vector<std::string> warnings;
};

ResourceEstimate estimate_resources(const nlohmann::json& resolved, const SamplePlan& plan,
                                    const NormalizedMesh& mesh,
                                    const std::vector<BenchmarkProfile>& profiles);

} // namespace strikecem

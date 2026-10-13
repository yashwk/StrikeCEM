#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "strikecem/runtime/Estimate.hpp"

namespace strikecem::analysis {

struct BenchmarkEntry {
    BenchmarkProfile profile;
    double throughput_meval_per_sec = 0.0; // (triangles * samples) / (wall_seconds * 1e6)
};

struct BenchmarkAggregation {
    std::vector<BenchmarkEntry> entries;
    double cpu_avg_throughput = 0.0;
    double cuda_avg_throughput = 0.0;
    double max_throughput = 0.0;
    double speedup_ratio = 1.0; // cuda_avg / cpu_avg
};

// Aggregates a set of BenchmarkProfile objects and computes throughput metrics.
BenchmarkAggregation aggregate_profiles(const std::vector<BenchmarkProfile>& profiles);

// Collects profiles from specified paths or default locations, then aggregates them.
BenchmarkAggregation load_and_aggregate(const std::vector<std::string>& search_paths = {});

// Renders an ASCII throughput scaling bar chart.
std::string render_benchmark_ascii(const BenchmarkAggregation& agg, int bar_width = 30);

// Exports aggregation results to JSON format.
nlohmann::json benchmark_aggregation_to_json(const BenchmarkAggregation& agg);

} // namespace strikecem::analysis

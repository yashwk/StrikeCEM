#pragma once
// Master-coordinated adaptive coarse-to-refine sampling.
// Implements ADR-0007, FULL_IMPLEMENTATION.md §11.4, and BACKLOG.md item 46.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/EdgeFringe.hpp"
#include "strikecem/solvers/Material.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"
#include "strikecem/solvers/Ray.hpp"

namespace strikecem {

struct AdaptiveParams {
    double coarse_step_deg = 1.0;
    double refine_step_deg = 0.1;
    std::optional<double> peak_threshold_dbsm;
    double gradient_threshold_db_per_deg = 2.0;
    double window_half_width_deg = 5.0;
    int max_refinement_stages = 1;
};

AdaptiveParams parse_adaptive_params(const nlohmann::json& resolved);

struct Interval {
    double start_deg = 0.0;
    double stop_deg = 0.0;
};

// Detects local peaks and high gradient regions on coarse results.
// Expands intervals by window_half_width_deg and merges overlapping intervals into disjoint windows.
std::vector<Interval> detect_and_merge_windows(
    const SamplePlan& coarse_plan,
    const PoResult& coarse_result,
    const AdaptiveParams& params,
    double domain_min_deg,
    double domain_max_deg,
    size_t* out_peaks = nullptr,
    size_t* out_gradients = nullptr);

// Generates a deduplicated refinement plan from disjoint windows, omitting directions
// that are already present in coarse_plan.
SamplePlan build_refinement_plan(
    const ResolvedConfig& rc,
    const SamplePlan& coarse_plan,
    const std::vector<Interval>& windows,
    double refine_step_deg);

// Combines coarse and refinement sample results into a unified sparse dataset with canonical IDs.
struct AdaptiveResult {
    SamplePlan coarse_plan;
    PoResult coarse_result;
    SamplePlan refinement_plan;
    PoResult refinement_result;
    SamplePlan merged_plan;
    PoResult merged_result;
    std::vector<Interval> windows;
    size_t peaks_detected = 0;
    size_t gradient_triggers = 0;
};

AdaptiveResult solve_adaptive(
    const NormalizedMesh& mesh,
    const ResolvedConfig& rc,
    const FringeOptions& fringe = FringeOptions{},
    const GoOptions& go = GoOptions{},
    const MaterialOptions& material = MaterialOptions{});

} // namespace strikecem

#include "strikecem/adaptive/AdaptiveSampling.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>

#include "strikecem/io/OutputError.hpp"

namespace strikecem {

AdaptiveParams parse_adaptive_params(const nlohmann::json& resolved) {
    AdaptiveParams p;
    if (!resolved.contains("angles") || !resolved["angles"].contains("adaptive")) {
        return p;
    }
    const auto& a = resolved["angles"]["adaptive"];
    p.coarse_step_deg = a["coarse_step_deg"].get<double>();
    p.refine_step_deg = a["refine_step_deg"].get<double>();
    if (a.contains("peak_threshold_dbsm") && !a["peak_threshold_dbsm"].is_null()) {
        p.peak_threshold_dbsm = a["peak_threshold_dbsm"].get<double>();
    }
    p.gradient_threshold_db_per_deg = a.value("gradient_threshold_db_per_deg", 2.0);
    p.window_half_width_deg = a.value("window_half_width_deg", 5.0);
    p.max_refinement_stages = a.value("max_refinement_stages", 1);
    return p;
}

std::vector<Interval> detect_and_merge_windows(
    const SamplePlan& coarse_plan,
    const PoResult& coarse_result,
    const AdaptiveParams& params,
    double domain_min_deg,
    double domain_max_deg,
    size_t* out_peaks,
    size_t* out_gradients) {
    const size_t ndir = coarse_plan.directions.size();
    if (ndir == 0) return {};

    // Group max RCS (in dBsm) per direction
    std::vector<double> max_rcs_db(ndir, -1000.0);
    for (const auto& sample : coarse_result.samples) {
        if (sample.direction_id < ndir) {
            const double db = 10.0 * std::log10(std::max(1e-12, sample.rcs_sqm));
            if (db > max_rcs_db[sample.direction_id]) {
                max_rcs_db[sample.direction_id] = db;
            }
        }
    }

    std::vector<double> trigger_angles;
    size_t peaks = 0;
    size_t gradients = 0;

    // 1. Peak detection
    for (size_t i = 0; i < ndir; ++i) {
        const double val = max_rcs_db[i];
        if (params.peak_threshold_dbsm.has_value() && val < params.peak_threshold_dbsm.value()) {
            continue;
        }
        bool is_peak = false;
        if (ndir == 1) {
            is_peak = true;
        } else if (i == 0) {
            is_peak = (val > max_rcs_db[1]);
        } else if (i == ndir - 1) {
            is_peak = (val > max_rcs_db[ndir - 2]);
        } else {
            is_peak = (val > max_rcs_db[i - 1] && val > max_rcs_db[i + 1]);
        }
        if (is_peak) {
            trigger_angles.push_back(coarse_plan.directions[i].azimuth_deg);
            ++peaks;
        }
    }

    // 2. High-gradient detection
    for (size_t i = 0; i + 1 < ndir; ++i) {
        const double az0 = coarse_plan.directions[i].azimuth_deg;
        const double az1 = coarse_plan.directions[i + 1].azimuth_deg;
        const double delta = std::abs(az1 - az0);
        if (delta > 1e-6) {
            const double grad = std::abs(max_rcs_db[i + 1] - max_rcs_db[i]) / delta;
            if (grad >= params.gradient_threshold_db_per_deg) {
                trigger_angles.push_back(0.5 * (az0 + az1));
                ++gradients;
            }
        }
    }

    if (out_peaks) *out_peaks = peaks;
    if (out_gradients) *out_gradients = gradients;

    if (trigger_angles.empty()) return {};

    // 3. Expand windows
    std::vector<Interval> intervals;
    intervals.reserve(trigger_angles.size());
    const double w = params.window_half_width_deg;
    for (double theta : trigger_angles) {
        Interval iv;
        iv.start_deg = std::max(domain_min_deg, theta - w);
        iv.stop_deg = std::min(domain_max_deg, theta + w);
        intervals.push_back(iv);
    }

    // 4. Merge overlapping intervals
    std::sort(intervals.begin(), intervals.end(),
              [](const Interval& a, const Interval& b) {
                  return a.start_deg < b.start_deg;
              });

    std::vector<Interval> merged;
    for (const auto& iv : intervals) {
        if (merged.empty()) {
            merged.push_back(iv);
        } else {
            auto& back = merged.back();
            if (iv.start_deg <= back.stop_deg + 1e-6) {
                back.stop_deg = std::max(back.stop_deg, iv.stop_deg);
            } else {
                merged.push_back(iv);
            }
        }
    }
    return merged;
}

SamplePlan build_refinement_plan(
    const ResolvedConfig& rc,
    const SamplePlan& coarse_plan,
    const std::vector<Interval>& windows,
    double refine_step_deg) {
    SamplePlan plan;
    plan.frequencies_hz = coarse_plan.frequencies_hz;
    plan.polarizations = coarse_plan.polarizations;

    // Collect all coarse azimuths for deduplication
    std::vector<double> coarse_az;
    for (const auto& d : coarse_plan.directions) {
        coarse_az.push_back(d.azimuth_deg);
    }
    std::sort(coarse_az.begin(), coarse_az.end());

    auto is_coarse_point = [&](double a) {
        auto it = std::lower_bound(coarse_az.begin(), coarse_az.end(), a - 1e-6);
        return it != coarse_az.end() && std::abs(*it - a) < 1e-6;
    };

    std::vector<double> refine_az;
    for (const auto& win : windows) {
        const double tol = 1e-9 * std::max(1.0, std::abs(win.stop_deg));
        const long long n = static_cast<long long>(std::floor((win.stop_deg - win.start_deg) / refine_step_deg + tol)) + 1;
        for (long long i = 0; i < n; ++i) {
            const double val = win.start_deg + i * refine_step_deg;
            if (val <= win.stop_deg + 1e-6 && !is_coarse_point(val)) {
                refine_az.push_back(val);
            }
        }
    }

    // Sort and deduplicate
    std::sort(refine_az.begin(), refine_az.end());
    refine_az.erase(std::unique(refine_az.begin(), refine_az.end(),
                                [](double a, double b) { return std::abs(a - b) < 1e-6; }),
                    refine_az.end());

    // Expand across elevation (using the first elevation from coarse or domain)
    double elev = 0.0;
    if (!coarse_plan.directions.empty()) {
        elev = coarse_plan.directions.front().elevation_deg;
    }

    uint32_t id = 0;
    for (double az : refine_az) {
        plan.directions.push_back({id++, az, elev, direction_from_az_el(az, elev)});
    }

    // Canonical plan hash
    std::ostringstream plan_bytes;
    plan_bytes << std::setprecision(std::numeric_limits<double>::max_digits10);
    plan_bytes << "parent:" << coarse_plan.hash << ";";
    for (double f : plan.frequencies_hz) plan_bytes << "f" << f << ";";
    for (const auto& d : plan.directions)
        plan_bytes << "d" << d.azimuth_deg << "," << d.elevation_deg << ";";
    for (const auto& p : plan.polarizations) plan_bytes << "p" << p << ";";
    plan.hash = sha256_hex(plan_bytes.str());

    return plan;
}

AdaptiveResult solve_adaptive(
    const NormalizedMesh& mesh,
    const ResolvedConfig& rc,
    const FringeOptions& fringe,
    const GoOptions& go,
    const MaterialOptions& material) {
    AdaptiveResult res;
    const AdaptiveParams params = parse_adaptive_params(rc.value);

    // 1. Stage 0: Solve coarse plan
    res.coarse_plan = build_sample_plan(rc.value);
    res.coarse_result = solve_po(mesh, res.coarse_plan, rc.value, fringe, go, material);

    const double domain_min = rc.value["angles"]["azimuth"]["start"].get<double>();
    const double domain_max = rc.value["angles"]["azimuth"]["stop"].get<double>();

    // 2. Detect peaks and gradients, expand windows
    res.windows = detect_and_merge_windows(
        res.coarse_plan, res.coarse_result, params,
        domain_min, domain_max,
        &res.peaks_detected, &res.gradient_triggers);

    if (res.windows.empty()) {
        res.merged_plan = res.coarse_plan;
        res.merged_result = res.coarse_result;
        return res;
    }

    // 3. Stage 1: Build refinement plan
    res.refinement_plan = build_refinement_plan(
        rc, res.coarse_plan, res.windows, params.refine_step_deg);

    if (res.refinement_plan.directions.empty()) {
        res.merged_plan = res.coarse_plan;
        res.merged_result = res.coarse_result;
        return res;
    }

    res.refinement_result = solve_po(mesh, res.refinement_plan, rc.value, fringe, go, material);

    // 4. Merge plans and results
    // Union of all directions
    std::vector<Direction> all_dirs = res.coarse_plan.directions;
    for (const auto& d : res.refinement_plan.directions) {
        all_dirs.push_back(d);
    }
    std::sort(all_dirs.begin(), all_dirs.end(), [](const Direction& a, const Direction& b) {
        if (std::abs(a.azimuth_deg - b.azimuth_deg) > 1e-6)
            return a.azimuth_deg < b.azimuth_deg;
        return a.elevation_deg < b.elevation_deg;
    });

    res.merged_plan.frequencies_hz = res.coarse_plan.frequencies_hz;
    res.merged_plan.polarizations = res.coarse_plan.polarizations;
    uint32_t dir_id = 0;
    for (auto& d : all_dirs) {
        d.id = dir_id++;
        res.merged_plan.directions.push_back(d);
    }

    // Hash merged plan
    std::ostringstream plan_bytes;
    plan_bytes << std::setprecision(std::numeric_limits<double>::max_digits10);
    plan_bytes << "adaptive_merged;";
    for (double f : res.merged_plan.frequencies_hz) plan_bytes << "f" << f << ";";
    for (const auto& d : res.merged_plan.directions)
        plan_bytes << "d" << d.azimuth_deg << "," << d.elevation_deg << ";";
    for (const auto& p : res.merged_plan.polarizations) plan_bytes << "p" << p << ";";
    res.merged_plan.hash = sha256_hex(plan_bytes.str());

    // Map samples from coarse and refinement by (frequency_index, direction key, polarization_index)
    // Direction key: quantized azimuth string or rounded angle
    auto dir_key = [](double az, double el) {
        const int64_t az_i = static_cast<int64_t>(std::round(az * 1000000.0));
        const int64_t el_i = static_cast<int64_t>(std::round(el * 1000000.0));
        return std::make_pair(az_i, el_i);
    };

    std::map<std::tuple<uint32_t, std::pair<int64_t, int64_t>, uint32_t>, PoSampleResult> sample_lookup;

    for (const auto& s : res.coarse_result.samples) {
        const auto& d = res.coarse_plan.directions[s.direction_id];
        sample_lookup[{s.frequency_id, dir_key(d.azimuth_deg, d.elevation_deg), s.pol_id}] = s;
    }
    for (const auto& s : res.refinement_result.samples) {
        const auto& d = res.refinement_plan.directions[s.direction_id];
        sample_lookup[{s.frequency_id, dir_key(d.azimuth_deg, d.elevation_deg), s.pol_id}] = s;
    }

    // Reconstruct contiguous merged result in merged_plan order
    const size_t nfreq = res.merged_plan.frequencies_hz.size();
    const size_t ndir = res.merged_plan.directions.size();
    const size_t npol = res.merged_plan.polarizations.size();
    const uint64_t total_samples = static_cast<uint64_t>(nfreq) * ndir * npol;

    res.merged_result.samples.reserve(total_samples);
    uint64_t global_id = 0;
    for (uint32_t fi = 0; fi < nfreq; ++fi) {
        for (uint32_t di = 0; di < ndir; ++di) {
            const auto& d = res.merged_plan.directions[di];
            const auto dk = dir_key(d.azimuth_deg, d.elevation_deg);
            for (uint32_t pi = 0; pi < npol; ++pi) {
                auto it = sample_lookup.find({fi, dk, pi});
                if (it != sample_lookup.end()) {
                    PoSampleResult row = it->second;
                    row.sample_id = global_id++;
                    row.frequency_id = fi;
                    row.direction_id = di;
                    row.pol_id = pi;
                    res.merged_result.samples.push_back(row);
                } else {
                    throw OutputError("internal error: missing sample in adaptive plan reconstruction");
                }
            }
        }
    }

    for (const auto& w : res.coarse_result.warnings) res.merged_result.warnings.push_back(w);
    for (const auto& w : res.refinement_result.warnings) res.merged_result.warnings.push_back(w);

    return res;
}

} // namespace strikecem

#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/api/RcsQueryEngine.hpp"

namespace strikecem::analysis {

struct DatabaseSummary {
    std::string design_id;
    std::string design_revision;
    std::vector<double> frequencies_hz;
    std::vector<std::string> polarizations;
    size_t direction_count = 0;
    double min_rcs_dbsm = 0.0;
    double max_rcs_dbsm = -999.0;
    double mean_rcs_dbsm = -999.0;
    double peak_azimuth_deg = 0.0;
    double peak_elevation_deg = 0.0;
    double peak_freq_hz = 0.0;
    std::string peak_polarization;
};

struct CutSample {
    double angle_deg = 0.0;
    double rcs_sqm = 0.0;
    double rcs_dbsm = -999.0;
};

struct DatabaseCut {
    std::string cut_type; // "azimuth" or "elevation"
    double fixed_angle_deg = 0.0;
    double frequency_hz = 0.0;
    std::string polarization;
    std::vector<CutSample> samples;
};

struct ComparisonMetrics {
    size_t sample_count = 0;
    double max_absolute_error_dbsm = 0.0;
    double mean_absolute_error_dbsm = 0.0;
    double rmse_dbsm = 0.0;
    double complex_correlation = 1.0;
    double peak_rcs_delta_dbsm = 0.0;
    bool passed = true;
};

// Generates statistical overview of a database
DatabaseSummary summarize_database(const RcsQueryEngine& engine);

// Extracts a 1D angle cut (azimuth cut at fixed elevation, or elevation cut at fixed azimuth)
DatabaseCut extract_cut(const RcsQueryEngine& engine,
                        const std::string& cut_type,
                        double fixed_angle_deg,
                        double frequency_hz,
                        const std::string& polarization,
                        double step_deg = 1.0);

// Compares two databases across common grid points
ComparisonMetrics compare_databases(const RcsQueryEngine& engine_a,
                                    const RcsQueryEngine& engine_b,
                                    double tolerance_dbsm = 1.0);

// ASCII visualization of a 1D cut
std::string render_ascii_cut(const DatabaseCut& cut, int width = 60);

nlohmann::json summary_to_json(const DatabaseSummary& summary);
nlohmann::json comparison_to_json(const ComparisonMetrics& metrics);

} // namespace strikecem::analysis

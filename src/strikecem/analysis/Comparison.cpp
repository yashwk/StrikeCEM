#include "strikecem/analysis/Comparison.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace strikecem::analysis {

DatabaseSummary summarize_database(const RcsQueryEngine& engine) {
    DatabaseSummary sum;
    sum.design_id = engine.design_id();
    sum.design_revision = engine.design_revision();
    sum.frequencies_hz = engine.frequencies();
    sum.polarizations = engine.polarizations();
    sum.direction_count = engine.directions().size();

    if (sum.frequencies_hz.empty() || sum.polarizations.empty() || engine.directions().empty()) {
        return sum;
    }

    double max_rcs = -1.0;
    double min_rcs = 1e30;
    double total_rcs = 0.0;
    size_t count = 0;

    for (double f : sum.frequencies_hz) {
        for (const auto& pol : sum.polarizations) {
            for (const auto& dir : engine.directions()) {
                const auto res = engine.query(dir.azimuth_deg, dir.elevation_deg, f, pol);
                total_rcs += res.rcs_sqm;
                ++count;

                if (res.rcs_sqm > max_rcs) {
                    max_rcs = res.rcs_sqm;
                    sum.peak_azimuth_deg = dir.azimuth_deg;
                    sum.peak_elevation_deg = dir.elevation_deg;
                    sum.peak_freq_hz = f;
                    sum.peak_polarization = pol;
                }
                if (res.rcs_sqm < min_rcs) {
                    min_rcs = res.rcs_sqm;
                }
            }
        }
    }

    sum.max_rcs_dbsm = (max_rcs > 1e-30) ? (10.0 * std::log10(max_rcs)) : -999.0;
    sum.min_rcs_dbsm = (min_rcs > 1e-30) ? (10.0 * std::log10(min_rcs)) : -999.0;
    const double mean_sqm = (count > 0) ? (total_rcs / count) : 0.0;
    sum.mean_rcs_dbsm = (mean_sqm > 1e-30) ? (10.0 * std::log10(mean_sqm)) : -999.0;

    return sum;
}

DatabaseCut extract_cut(const RcsQueryEngine& engine,
                        const std::string& cut_type,
                        double fixed_angle_deg,
                        double frequency_hz,
                        const std::string& polarization,
                        double step_deg) {
    if (step_deg <= 0.0) throw std::invalid_argument("step_deg must be positive");

    DatabaseCut cut;
    cut.cut_type = cut_type;
    cut.fixed_angle_deg = fixed_angle_deg;
    cut.frequency_hz = frequency_hz;
    cut.polarization = polarization;

    if (cut_type == "azimuth") {
        for (double az = 0.0; az <= 360.0 + 1e-6; az += step_deg) {
            const auto res = engine.query(az, fixed_angle_deg, frequency_hz, polarization);
            cut.samples.push_back({az, res.rcs_sqm, res.rcs_dbsm});
        }
    } else if (cut_type == "elevation") {
        for (double el = -90.0; el <= 90.0 + 1e-6; el += step_deg) {
            const auto res = engine.query(fixed_angle_deg, el, frequency_hz, polarization);
            cut.samples.push_back({el, res.rcs_sqm, res.rcs_dbsm});
        }
    } else {
        throw std::invalid_argument("unknown cut_type (must be 'azimuth' or 'elevation'): " + cut_type);
    }

    return cut;
}

ComparisonMetrics compare_databases(const RcsQueryEngine& engine_a,
                                    const RcsQueryEngine& engine_b,
                                    double tolerance_dbsm) {
    ComparisonMetrics m;
    const auto& freqs = engine_a.frequencies();
    const auto& pols = engine_a.polarizations();
    const auto& dirs = engine_a.directions();

    if (freqs.empty() || pols.empty() || dirs.empty()) {
        return m;
    }

    double sum_abs_err = 0.0;
    double sum_sq_err = 0.0;
    double max_err = 0.0;

    std::complex<double> sum_ab(0.0, 0.0);
    double sum_a2 = 0.0;
    double sum_b2 = 0.0;

    double max_rcs_a = -1e30;
    double max_rcs_b = -1e30;

    for (double f : freqs) {
        for (const auto& pol : pols) {
            for (const auto& dir : dirs) {
                const auto ra = engine_a.query(dir.azimuth_deg, dir.elevation_deg, f, pol);
                const auto rb = engine_b.query(dir.azimuth_deg, dir.elevation_deg, f, pol);

                const double err = std::abs(ra.rcs_dbsm - rb.rcs_dbsm);
                sum_abs_err += err;
                sum_sq_err += err * err;
                max_err = std::max(max_err, err);

                const std::complex<double> sa(ra.scattering_real, ra.scattering_imag);
                const std::complex<double> sb(rb.scattering_real, rb.scattering_imag);

                sum_ab += sa * std::conj(sb);
                sum_a2 += std::norm(sa);
                sum_b2 += std::norm(sb);

                max_rcs_a = std::max(max_rcs_a, ra.rcs_dbsm);
                max_rcs_b = std::max(max_rcs_b, rb.rcs_dbsm);

                ++m.sample_count;
            }
        }
    }

    if (m.sample_count > 0) {
        m.max_absolute_error_dbsm = max_err;
        m.mean_absolute_error_dbsm = sum_abs_err / static_cast<double>(m.sample_count);
        m.rmse_dbsm = std::sqrt(sum_sq_err / static_cast<double>(m.sample_count));
        const double denom = std::sqrt(sum_a2 * sum_b2);
        m.complex_correlation = (denom > 1e-15) ? (std::abs(sum_ab) / denom) : 1.0;
        m.peak_rcs_delta_dbsm = std::abs(max_rcs_a - max_rcs_b);
        m.passed = (m.max_absolute_error_dbsm <= tolerance_dbsm);
    }

    return m;
}

std::string render_ascii_cut(const DatabaseCut& cut, int width) {
    if (cut.samples.empty()) return "(empty cut)\n";

    double min_val = 1e30;
    double max_val = -1e30;
    for (const auto& s : cut.samples) {
        min_val = std::min(min_val, s.rcs_dbsm);
        max_val = std::max(max_val, s.rcs_dbsm);
    }

    if (max_val <= min_val) max_val = min_val + 1.0;

    std::ostringstream ss;
    ss << "=== 1D " << cut.cut_type << " Cut at " << cut.fixed_angle_deg
       << " deg | Freq: " << (cut.frequency_hz / 1e9) << " GHz | Pol: " << cut.polarization << " ===\n";
    ss << "Range: [" << std::fixed << std::setprecision(1) << min_val << " dBsm, " << max_val << " dBsm]\n\n";

    for (const auto& s : cut.samples) {
        const double norm = std::clamp((s.rcs_dbsm - min_val) / (max_val - min_val), 0.0, 1.0);
        const int bar_len = static_cast<int>(norm * width);

        ss << std::setw(6) << std::fixed << std::setprecision(1) << s.angle_deg << " deg | "
           << std::setw(7) << std::setprecision(2) << s.rcs_dbsm << " dBsm | "
           << std::string(bar_len, '#') << "\n";
    }

    return ss.str();
}

nlohmann::json summary_to_json(const DatabaseSummary& sum) {
    nlohmann::json j;
    j["design_id"] = sum.design_id;
    j["design_revision"] = sum.design_revision;
    j["frequencies_hz"] = sum.frequencies_hz;
    j["polarizations"] = sum.polarizations;
    j["direction_count"] = sum.direction_count;
    j["min_rcs_dbsm"] = sum.min_rcs_dbsm;
    j["max_rcs_dbsm"] = sum.max_rcs_dbsm;
    j["mean_rcs_dbsm"] = sum.mean_rcs_dbsm;
    j["peak"] = {
        {"azimuth_deg", sum.peak_azimuth_deg},
        {"elevation_deg", sum.peak_elevation_deg},
        {"frequency_hz", sum.peak_freq_hz},
        {"polarization", sum.peak_polarization}
    };
    return j;
}

nlohmann::json comparison_to_json(const ComparisonMetrics& m) {
    nlohmann::json j;
    j["sample_count"] = m.sample_count;
    j["max_absolute_error_dbsm"] = m.max_absolute_error_dbsm;
    j["mean_absolute_error_dbsm"] = m.mean_absolute_error_dbsm;
    j["rmse_dbsm"] = m.rmse_dbsm;
    j["complex_correlation"] = m.complex_correlation;
    j["peak_rcs_delta_dbsm"] = m.peak_rcs_delta_dbsm;
    j["passed"] = m.passed;
    return j;
}

} // namespace strikecem::analysis

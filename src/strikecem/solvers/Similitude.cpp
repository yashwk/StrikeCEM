#include "strikecem/solvers/Similitude.hpp"

#include <cmath>
#include <iostream>

#include "strikecem/io/OutputError.hpp"

namespace strikecem {

double scale_frequency(double freq_hz, double geometric_scale, SimilitudeDirection dir) {
    if (geometric_scale <= 0.0) {
        throw OutputError("Similitude geometric scale must be positive");
    }
    if (dir == SimilitudeDirection::FullToModel) {
        return freq_hz / geometric_scale;
    } else {
        return freq_hz * geometric_scale;
    }
}

double scale_wavelength(double wavelength_m, double geometric_scale, SimilitudeDirection dir) {
    if (geometric_scale <= 0.0) {
        throw OutputError("Similitude geometric scale must be positive");
    }
    if (dir == SimilitudeDirection::FullToModel) {
        return wavelength_m * geometric_scale;
    } else {
        return wavelength_m / geometric_scale;
    }
}

double rescale_rcs_sqm(double rcs_sqm, double geometric_scale, SimilitudeDirection dir) {
    if (geometric_scale <= 0.0) {
        throw OutputError("Similitude geometric scale must be positive");
    }
    const double factor = 1.0 / (geometric_scale * geometric_scale);
    return rcs_sqm * factor;
}

double rescale_rcs_dbsm(double rcs_dbsm, double geometric_scale, SimilitudeDirection dir) {
    if (geometric_scale <= 0.0) {
        throw OutputError("Similitude geometric scale must be positive");
    }
    const double offset_db = 20.0 * std::log10(1.0 / geometric_scale);
    return rcs_dbsm + offset_db;
}

std::complex<double> rescale_scattering_amplitude(std::complex<double> S, double geometric_scale,
                                                 SimilitudeDirection dir) {
    if (geometric_scale <= 0.0) {
        throw OutputError("Similitude geometric scale must be positive");
    }
    const double factor = 1.0 / geometric_scale;
    return S * factor;
}

NormalizedMesh scale_mesh(const NormalizedMesh& mesh, double geometric_scale, SimilitudeDirection dir) {
    if (geometric_scale <= 0.0) {
        throw OutputError("Similitude geometric scale must be positive");
    }
    const double factor = (dir == SimilitudeDirection::FullToModel) ? geometric_scale : (1.0 / geometric_scale);

    NormalizedMesh scaled = mesh;
    scaled.report.bbox_min = geom::Vec3d{1e30, 1e30, 1e30};
    scaled.report.bbox_max = geom::Vec3d{-1e30, -1e30, -1e30};
    scaled.report.total_area_m2 = 0.0;

    for (auto& v : scaled.vertices) {
        v.x *= factor;
        v.y *= factor;
        v.z *= factor;
        scaled.report.bbox_min.x = std::min(scaled.report.bbox_min.x, v.x);
        scaled.report.bbox_min.y = std::min(scaled.report.bbox_min.y, v.y);
        scaled.report.bbox_min.z = std::min(scaled.report.bbox_min.z, v.z);
        scaled.report.bbox_max.x = std::max(scaled.report.bbox_max.x, v.x);
        scaled.report.bbox_max.y = std::max(scaled.report.bbox_max.y, v.y);
        scaled.report.bbox_max.z = std::max(scaled.report.bbox_max.z, v.z);
    }

    for (auto& a : scaled.areas) {
        a *= (factor * factor);
        scaled.report.total_area_m2 += a;
    }

    return scaled;
}

SimilitudeConfig parse_similitude_config(const nlohmann::json& config_json) {
    SimilitudeConfig cfg;
    if (!config_json.contains("similitude")) {
        return cfg;
    }

    const auto& s = config_json["similitude"];
    cfg.enabled = s.value("enabled", false);
    cfg.geometric_scale = s.value("geometric_scale", 1.0);
    cfg.auto_rescale_rcs = s.value("auto_rescale_rcs", true);

    const std::string mode_str = s.value("mode", s.value("direction", "full_to_model"));
    if (mode_str == "model_to_full") {
        cfg.direction = SimilitudeDirection::ModelToFull;
    } else {
        cfg.direction = SimilitudeDirection::FullToModel;
    }

    return cfg;
}

SimilitudeSummary apply_similitude(ResolvedConfig& rc, SamplePlan& plan, NormalizedMesh& mesh,
                                   const SimilitudeConfig& sim_cfg) {
    SimilitudeSummary sum;
    sum.geometric_scale = sim_cfg.geometric_scale;
    sum.rcs_scale_factor = 1.0 / (sim_cfg.geometric_scale * sim_cfg.geometric_scale);
    sum.rcs_offset_dbsm = 20.0 * std::log10(1.0 / sim_cfg.geometric_scale);

    const auto& bmin = mesh.report.bbox_min;
    const auto& bmax = mesh.report.bbox_max;
    sum.input_bbox_diagonal_m = std::sqrt(
        (bmax.x - bmin.x) * (bmax.x - bmin.x) +
        (bmax.y - bmin.y) * (bmax.y - bmin.y) +
        (bmax.z - bmin.z) * (bmax.z - bmin.z));

    if (!plan.frequencies_hz.empty()) {
        sum.input_frequency_hz = plan.frequencies_hz.front();
    }

    if (!sim_cfg.enabled || sim_cfg.geometric_scale == 1.0) {
        sum.scaled_frequency_hz = sum.input_frequency_hz;
        sum.scaled_bbox_diagonal_m = sum.input_bbox_diagonal_m;
        sum.frequency_scale = 1.0;
        return sum;
    }

    if (sim_cfg.direction == SimilitudeDirection::FullToModel) {
        sum.frequency_scale = 1.0 / sim_cfg.geometric_scale;
        mesh = scale_mesh(mesh, sim_cfg.geometric_scale, sim_cfg.direction);

        for (auto& f : plan.frequencies_hz) {
            f = scale_frequency(f, sim_cfg.geometric_scale, sim_cfg.direction);
        }
    } else {
        sum.frequency_scale = 1.0; // Solve model as-is, rescale output only
    }

    const auto& sbmin = mesh.report.bbox_min;
    const auto& sbmax = mesh.report.bbox_max;
    sum.scaled_bbox_diagonal_m = std::sqrt(
        (sbmax.x - sbmin.x) * (sbmax.x - sbmin.x) +
        (sbmax.y - sbmin.y) * (sbmax.y - sbmin.y) +
        (sbmax.z - sbmin.z) * (sbmax.z - sbmin.z));

    if (!plan.frequencies_hz.empty()) {
        sum.scaled_frequency_hz = plan.frequencies_hz.front();
    }

    return sum;
}

void rescale_po_result(PoResult& result, const SimilitudeConfig& sim_cfg) {
    if (!sim_cfg.enabled || !sim_cfg.auto_rescale_rcs || sim_cfg.geometric_scale == 1.0) {
        return;
    }

    const double sqm_factor = 1.0 / (sim_cfg.geometric_scale * sim_cfg.geometric_scale);
    const double s_factor = 1.0 / sim_cfg.geometric_scale;

    for (auto& sample : result.samples) {
        sample.rcs_sqm *= sqm_factor;
        sample.scattering *= s_factor;
    }
}

} // namespace strikecem

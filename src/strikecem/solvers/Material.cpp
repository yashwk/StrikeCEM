#include "strikecem/solvers/Material.hpp"

#include <cmath>
#include <map>
#include <numbers>
#include <set>
#include <stdexcept>

#include "strikecem/core/Config.hpp"
#include "strikecem/core/Conventions.hpp"
#include "strikecem/io/MeshLoader.hpp"

namespace strikecem {
namespace {

constexpr double kPassiveTol = 1e-12;

bool finite_c(const std::complex<double>& z) { return std::isfinite(z.real() + z.imag()); }

void check_passive(const ComplexMedium& m) {
    if (!finite_c(m.eps_r) || !finite_c(m.mu_r))
        throw std::invalid_argument("material constants must be finite");
    if (m.eps_r.imag() > kPassiveTol || m.mu_r.imag() > kPassiveTol)
        throw std::invalid_argument("material must be passive");
}

} // namespace

void validate_material(const MaterialModel& model) {
    if (model.tag.empty()) throw std::invalid_argument("material tag must be non-empty");
    for (const auto& layer : model.layers) {
        if (!(layer.thickness_m > 0.0) || !std::isfinite(layer.thickness_m))
            throw std::invalid_argument("coating thickness must be positive and finite");
        check_passive(layer.medium);
    }
    if (model.type == WallType::Pec) {
        if (!model.table.empty() || !model.layers.empty())
            throw std::invalid_argument("PEC is bare by definition");
        return;
    }
    if (model.type == WallType::Impedance) {
        if (model.surface_impedance.real() < -1e-12 || !std::isfinite(model.surface_impedance.real()) ||
            !std::isfinite(model.surface_impedance.imag()))
            throw std::invalid_argument("surface impedance real part must be non-negative and finite");
        return;
    }
    if (model.type == WallType::Dielectric && !model.layers.empty())
        throw std::invalid_argument("layered walls use the coated type");
    if (model.type == WallType::Coated && model.layers.empty())
        throw std::invalid_argument("coated walls need at least one layer");
    if (model.type == WallType::Dielectric && model.table.empty())
        throw std::invalid_argument("dielectric walls need a property table");
    for (size_t i = 0; i < model.table.size(); ++i) {
        const auto& e = model.table[i];
        if (!(e.frequency_hz > 0.0) || !std::isfinite(e.frequency_hz))
            throw std::invalid_argument("table frequencies must be positive and finite");
        if (i > 0 && !(e.frequency_hz > model.table[i - 1].frequency_hz))
            throw std::invalid_argument("table frequencies must be strictly ascending");
        check_passive(e.medium);
        if (!(e.sigma >= 0.0) || !std::isfinite(e.sigma))
            throw std::invalid_argument("conductivity must be non-negative and finite");
    }
}

ComplexMedium effective_medium(const MaterialEntry& entry, double frequency_hz) {
    if (!(frequency_hz > 0.0) || !std::isfinite(frequency_hz))
        throw std::invalid_argument("frequency must be positive and finite");
    check_passive(entry.medium);
    if (!(entry.sigma >= 0.0) || !std::isfinite(entry.sigma))
        throw std::invalid_argument("conductivity must be non-negative and finite");
    const double eps0 = 1.0 / (kEta0 * kSpeedOfLight);
    ComplexMedium out = entry.medium;
    out.eps_r -= std::complex<double>(0.0, entry.sigma / (2.0 * std::numbers::pi *
                                                           frequency_hz * eps0));
    return out;
}

ComplexMedium evaluate_material(const MaterialModel& model, double frequency_hz) {
    validate_material(model);
    if (!(frequency_hz > 0.0) || !std::isfinite(frequency_hz))
        throw std::invalid_argument("frequency must be positive and finite");
    if (model.table.empty())
        throw std::invalid_argument("no property table to evaluate");
    if (model.table.size() == 1) return effective_medium(model.table[0], frequency_hz);
    if (frequency_hz < model.table.front().frequency_hz ||
        frequency_hz > model.table.back().frequency_hz)
        throw std::invalid_argument("frequency outside material table range");
    size_t hi = 1;
    while (hi + 1 < model.table.size() && model.table[hi].frequency_hz < frequency_hz) ++hi;
    const MaterialEntry& lo = model.table[hi - 1];
    const MaterialEntry& up = model.table[hi];
    const double t = (frequency_hz - lo.frequency_hz) / (up.frequency_hz - lo.frequency_hz);
    MaterialEntry mix;
    mix.frequency_hz = frequency_hz;
    mix.medium.eps_r = lo.medium.eps_r + (up.medium.eps_r - lo.medium.eps_r) * t;
    mix.medium.mu_r = lo.medium.mu_r + (up.medium.mu_r - lo.medium.mu_r) * t;
    mix.sigma = lo.sigma + (up.sigma - lo.sigma) * t;
    return effective_medium(mix, frequency_hz);
}

CoatedResult coated_pec_reflection(const ComplexMedium& incident,
                                   const std::vector<CoatingLayer>& outer_to_pec,
                                   double cos_theta_i, double wavelength_m) {
    if (!(wavelength_m > 0.0) || !std::isfinite(wavelength_m))
        throw std::invalid_argument("wavelength must be positive and finite");
    if (!(cos_theta_i >= 0.0) || !(cos_theta_i <= 1.0) || !std::isfinite(cos_theta_i))
        throw std::invalid_argument("cos_theta_i must be in [0, 1]");
    if (!finite_c(incident.eps_r) || !finite_c(incident.mu_r))
        throw std::invalid_argument("incident medium must be finite");
    for (const auto& layer : outer_to_pec) {
        if (!(layer.thickness_m > 0.0) || !std::isfinite(layer.thickness_m))
            throw std::invalid_argument("coating thickness must be positive and finite");
        check_passive(layer.medium);
    }
    const double k0 = 2.0 * std::numbers::pi / wavelength_m;
    std::vector<std::complex<double>> rs_te;
    std::vector<std::complex<double>> rs_tm;
    std::vector<std::complex<double>> phis;
    rs_te.reserve(outer_to_pec.size());
    rs_tm.reserve(outer_to_pec.size());
    phis.reserve(outer_to_pec.size());
    ComplexMedium above = incident;
    std::complex<double> cos_above{cos_theta_i, 0.0};
    for (const auto& layer : outer_to_pec) {
        const FresnelResult f = fresnel(above, layer.medium, cos_above);
        const std::complex<double> n = refractive_index(layer.medium);
        const std::complex<double> beta = k0 * n * layer.thickness_m * f.cos_theta_t;
        rs_te.push_back(f.r_te);
        rs_tm.push_back(f.r_tm);
        phis.push_back(std::exp(std::complex<double>(0.0, -2.0) * beta));
        above = layer.medium;
        cos_above = f.cos_theta_t;
    }
    std::complex<double> r_te{-1.0, 0.0};
    std::complex<double> r_tm{1.0, 0.0};
    for (size_t j = outer_to_pec.size(); j-- > 0;) {
        r_te = (rs_te[j] + r_te * phis[j]) / (1.0 + rs_te[j] * r_te * phis[j]);
        r_tm = (rs_tm[j] + r_tm * phis[j]) / (1.0 + rs_tm[j] * r_tm * phis[j]);
    }
    return {r_te, r_tm};
}

ImpedanceResult impedance_reflection(const std::complex<double>& eta_s, double cos_theta_i) {
    if (cos_theta_i < 0.0 || cos_theta_i > 1.0 || !std::isfinite(cos_theta_i))
        throw std::invalid_argument("cos_theta_i must be in [0, 1]");
    if (eta_s.real() < -1e-12 || !std::isfinite(eta_s.real()) || !std::isfinite(eta_s.imag()))
        throw std::invalid_argument("surface impedance real part must be non-negative and finite");

    // Leontovich surface impedance boundary condition
    // For TE (perpendicular): r_te = (eta_s * cos_theta_i - 1) / (eta_s * cos_theta_i + 1)
    std::complex<double> num_te = eta_s * cos_theta_i - 1.0;
    std::complex<double> den_te = eta_s * cos_theta_i + 1.0;
    std::complex<double> r_te = (std::abs(den_te) < 1e-14) ? std::complex<double>(-1.0, 0.0) : num_te / den_te;

    // For TM (parallel): r_tm = (eta_s - cos_theta_i) / (eta_s + cos_theta_i)
    std::complex<double> num_tm = eta_s - cos_theta_i;
    std::complex<double> den_tm = eta_s + cos_theta_i;
    std::complex<double> r_tm = (std::abs(den_tm) < 1e-14) ? std::complex<double>(1.0, 0.0) : num_tm / den_tm;

    return {r_te, r_tm};
}

MaterialModel material_from_json(const nlohmann::json& j) {
    if (!j.contains("tag") || !j["tag"].is_string() || j["tag"].get<std::string>().empty())
        throw ConfigError("material definition requires non-empty string 'tag'");
    MaterialModel m;
    m.tag = j["tag"].get<std::string>();
    if (!j.contains("type") || !j["type"].is_string())
        throw ConfigError("material definition '" + m.tag + "' requires string 'type'");
    const std::string type_str = j["type"].get<std::string>();
    if (type_str == "pec") {
        m.type = WallType::Pec;
    } else if (type_str == "dielectric") {
        m.type = WallType::Dielectric;
    } else if (type_str == "coated") {
        m.type = WallType::Coated;
    } else if (type_str == "impedance" || type_str == "sibc") {
        m.type = WallType::Impedance;
    } else {
        throw ConfigError("unknown material type: '" + type_str + "'");
    }

    auto parse_complex = [](const nlohmann::json& val, const char* name) -> std::complex<double> {
        if (val.is_number()) {
            return {val.get<double>(), 0.0};
        } else if (val.is_array()) {
            if (val.empty() || val.size() > 2)
                throw ConfigError(std::string(name) + " must have 1 or 2 components");
            double r = val[0].get<double>();
            double i = val.size() > 1 ? val[1].get<double>() : 0.0;
            return {r, i};
        }
        throw ConfigError(std::string(name) + " must be a number or [real, imag] array");
    };

    if (j.contains("frequency_table")) {
        for (const auto& entry_j : j["frequency_table"]) {
            MaterialEntry entry;
            if (!entry_j.contains("frequency_hz") || !entry_j["frequency_hz"].is_number())
                throw ConfigError("frequency_table entry requires number 'frequency_hz'");
            entry.frequency_hz = entry_j["frequency_hz"].get<double>();
            if (entry_j.contains("epsilon_r")) {
                entry.medium.eps_r = parse_complex(entry_j["epsilon_r"], "epsilon_r");
            } else {
                entry.medium.eps_r = {1.0, 0.0};
            }
            if (entry_j.contains("mu_r")) {
                entry.medium.mu_r = parse_complex(entry_j["mu_r"], "mu_r");
            } else {
                entry.medium.mu_r = {1.0, 0.0};
            }
            if (entry_j.contains("sigma")) {
                entry.sigma = entry_j["sigma"].get<double>();
            }
            m.table.push_back(entry);
        }
    }

    if (j.contains("layers")) {
        for (const auto& layer_j : j["layers"]) {
            CoatingLayer layer;
            if (!layer_j.contains("thickness_m") || !layer_j["thickness_m"].is_number())
                throw ConfigError("coating layer requires number 'thickness_m'");
            layer.thickness_m = layer_j["thickness_m"].get<double>();
            if (layer_j.contains("epsilon_r")) {
                layer.medium.eps_r = parse_complex(layer_j["epsilon_r"], "epsilon_r");
            } else {
                layer.medium.eps_r = {1.0, 0.0};
            }
            if (layer_j.contains("mu_r")) {
                layer.medium.mu_r = parse_complex(layer_j["mu_r"], "mu_r");
            } else {
                layer.medium.mu_r = {1.0, 0.0};
            }
            m.layers.push_back(layer);
        }
    }

    if (m.type == WallType::Impedance) {
        if (j.contains("surface_impedance_ohms")) {
            auto z = parse_complex(j["surface_impedance_ohms"], "surface_impedance_ohms");
            m.surface_impedance = z / 376.73031346177066;
        } else if (j.contains("normalized_impedance")) {
            m.surface_impedance = parse_complex(j["normalized_impedance"], "normalized_impedance");
        } else {
            m.surface_impedance = {1.0, 0.0};
        }
    }

    try {
        validate_material(m);
    } catch (const std::exception& e) {
        throw ConfigError(std::string("invalid material '") + m.tag + "': " + e.what());
    }
    return m;
}

ResolvedMaterials resolve_materials(const nlohmann::json& resolved, const NormalizedMesh& mesh) {
    ResolvedMaterials res;
    if (!resolved.contains("materials")) {
        return res;
    }
    const auto& mat_conf = resolved["materials"];
    res.default_material = mat_conf.value("default_material", "pec");
    res.tag_coverage = mat_conf.value("tag_coverage", "warn");

    // Index 0 is always PEC
    MaterialModel pec_model;
    pec_model.tag = "pec";
    pec_model.type = WallType::Pec;
    res.models.push_back(pec_model);

    std::map<std::string, uint32_t> tag_to_model;
    tag_to_model["pec"] = 0;

    if (mat_conf.contains("definitions")) {
        for (const auto& def_j : mat_conf["definitions"]) {
            MaterialModel model = material_from_json(def_j);
            if (tag_to_model.find(model.tag) != tag_to_model.end() && model.tag != "pec") {
                throw ConfigError("duplicate material tag definition: " + model.tag);
            }
            if (model.tag == "pec") {
                res.models[0] = model;
            } else {
                uint32_t idx = static_cast<uint32_t>(res.models.size());
                res.models.push_back(model);
                tag_to_model[model.tag] = idx;
            }
        }
    }

    auto default_it = tag_to_model.find(res.default_material);
    if (default_it == tag_to_model.end()) {
        throw ConfigError("default_material '" + res.default_material + "' is not defined");
    }
    const uint32_t default_idx = default_it->second;

    const size_t n_tris = mesh.triangles.size();
    res.facet_material_indices.resize(n_tris, default_idx);

    bool had_untagged = false;
    std::set<std::string> unmatched_tags;

    for (size_t t = 0; t < n_tris; ++t) {
        const std::string& tag = (t < mesh.triangle_tags.size()) ? mesh.triangle_tags[t] : "";
        if (tag.empty()) {
            if (res.tag_coverage == "strict") {
                throw ConfigError("mesh contains untagged facet(s) under strict material tag coverage");
            }
            had_untagged = true;
            res.facet_material_indices[t] = default_idx;
        } else {
            auto it = tag_to_model.find(tag);
            if (it != tag_to_model.end()) {
                res.facet_material_indices[t] = it->second;
            } else {
                if (res.tag_coverage == "strict") {
                    throw ConfigError("mesh tag '" + tag + "' has no material definition under strict coverage");
                }
                unmatched_tags.insert(tag);
                res.facet_material_indices[t] = default_idx;
            }
        }
    }

    if (res.tag_coverage == "warn") {
        if (had_untagged) {
            res.warnings.push_back("mesh contains untagged facets; using default material '" + res.default_material + "'");
        }
        for (const auto& ut : unmatched_tags) {
            res.warnings.push_back("mesh tag '" + ut + "' not found in material definitions; using default material '" + res.default_material + "'");
        }
    }

    for (size_t t = 0; t < n_tris; ++t) {
        uint32_t idx = res.facet_material_indices[t];
        if (res.models[idx].type != WallType::Pec) {
            res.has_materials = true;
            ++res.non_pec_facets;
        }
    }

    return res;
}

} // namespace strikecem

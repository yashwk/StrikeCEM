// v1 configuration loader implementation.
#include "strikecem/core/Config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <openssl/sha.h>
#include <set>
#include <sstream>

#include <nlohmann/json-schema.hpp>

#include "strikecem/sampling/SphericalSampling.hpp"
#include "strikecem/solvers/Material.hpp"

namespace strikecem {
namespace {

namespace fs = std::filesystem;
using nlohmann::json;
using nlohmann::json_schema::json_validator;

json read_json_file(const std::string& path, const char* what) {
    std::ifstream in(path);
    if (!in) throw ConfigError(std::string("cannot open ") + what + ": " + path);
    try {
        return json::parse(in);
    } catch (const std::exception& e) {
        throw ConfigError(std::string("invalid JSON in ") + what + " " + path + ": " + e.what());
    }
}

void require_finite(double v, const char* name) {
    if (!std::isfinite(v)) throw ConfigError(std::string("non-finite value for ") + name);
}

// Apply v1 schema defaults into a new resolved object.
json resolve_defaults(const json& raw) {
    json r = raw;
    auto& model = r["model"];
    if (!model.contains("format")) model["format"] = "mesh";
    if (!model.contains("units")) model["units"] = "m";
    if (!model.contains("mesh_repair")) model["mesh_repair"] = "report";

    auto& freq = r["frequency"];
    if (!freq.contains("medium")) freq["medium"] = json::object();
    auto& medium = freq["medium"];
    if (!medium.contains("epsilon_r")) medium["epsilon_r"] = 1.0;
    if (!medium.contains("mu_r")) medium["mu_r"] = 1.0;
    if (!medium.contains("sigma")) medium["sigma"] = 0.0;

    if (!r.contains("physics")) r["physics"] = json::object();
    auto& physics = r["physics"];
    if (!physics.contains("monostatic")) physics["monostatic"] = true;
    if (!physics.contains("polarization_basis")) physics["polarization_basis"] = "linear";
    if (!physics.contains("incident_amplitude")) physics["incident_amplitude"] = 1.0;
    if (!physics.contains("phase_reference")) physics["phase_reference"] = json::array({0.0, 0.0, 0.0});

    auto& angles = r["angles"];
    if (!angles.contains("deduplicate_periodic_endpoints"))
        angles["deduplicate_periodic_endpoints"] = true;
    if (angles.contains("spherical")) {
        auto& sph = angles["spherical"];
        if (!sph.contains("type")) sph["type"] = "fibonacci";
        if (!sph.contains("elevation_min_deg")) sph["elevation_min_deg"] = -90.0;
        if (!sph.contains("elevation_max_deg")) sph["elevation_max_deg"] = 90.0;
    }

    if (!r.contains("polarization")) r["polarization"] = json{"HH", "VV"};

    auto& solver = r["solver"];
    if (!solver.contains("type")) solver["type"] = "PO";
    if (!solver.contains("precision")) solver["precision"] = "float64";
    if (!solver.contains("rcs_units")) solver["rcs_units"] = "dBsm";
    if (!solver.contains("po_options")) solver["po_options"] = json::object();
    auto& po = solver["po_options"];
    if (!po.contains("edge_correction")) po["edge_correction"] = "none";
    if (!po.contains("illumination_model")) po["illumination_model"] = "hard";
    if (!po.contains("curvature_correction")) po["curvature_correction"] = false;
    if (!po.contains("shadowing")) po["shadowing"] = false;
    if (!po.contains("max_bounces")) po["max_bounces"] = 1;

    if (!r.contains("mesh")) r["mesh"] = json::object();
    auto& mesh = r["mesh"];
    if (!mesh.contains("mode")) mesh["mode"] = "input";
    if (!mesh.contains("validate_orientation")) mesh["validate_orientation"] = true;
    if (!mesh.contains("max_aspect_ratio")) mesh["max_aspect_ratio"] = 20.0;
    if (!mesh.contains("symmetry")) mesh["symmetry"] = json::object();
    auto& sym = mesh["symmetry"];
    if (!sym.contains("enabled")) sym["enabled"] = false;
    if (!sym.contains("plane")) sym["plane"] = "xz";
    if (!sym.contains("tolerance_m")) sym["tolerance_m"] = 1e-4;
    if (!mesh.contains("frequency_policy")) mesh["frequency_policy"] = "user_mesh";
    if (!mesh.contains("wavelength_adaptive")) mesh["wavelength_adaptive"] = json::object();
    auto& wa = mesh["wavelength_adaptive"];
    if (!wa.contains("alpha_smooth")) wa["alpha_smooth"] = 1.0 / 6.0;
    if (!wa.contains("alpha_edge")) wa["alpha_edge"] = 1.0 / 20.0;
    if (!wa.contains("curvature_factor")) wa["curvature_factor"] = 0.1;

    if (!r.contains("execution")) r["execution"] = json::object();
    auto& exec = r["execution"];
    if (!exec.contains("cpu_threads")) exec["cpu_threads"] = 1;
    if (!exec.contains("batch_samples")) exec["batch_samples"] = 4096;
    if (!exec.contains("max_memory_mb")) exec["max_memory_mb"] = 16384;
    if (!exec.contains("deterministic")) exec["deterministic"] = true;
    if (!exec.contains("accelerator")) exec["accelerator"] = "cpu";
    if (!exec.contains("cuda_device_id")) exec["cuda_device_id"] = 0;
    if (!exec.contains("cuda_batch_units")) exec["cuda_batch_units"] = 0;

    auto& output = r["output"];
    if (!output.contains("format")) output["format"] = "hdf5";
    if (!output.contains("precision")) output["precision"] = 6;
    if (!output.contains("compression")) output["compression"] = false;
    if (!output.contains("include_metadata")) output["include_metadata"] = true;

    if (r.contains("materials")) {
        auto& materials = r["materials"];
        if (!materials.contains("default_material")) materials["default_material"] = "pec";
        if (!materials.contains("tag_coverage")) materials["tag_coverage"] = "warn";
        if (!materials.contains("definitions")) materials["definitions"] = json::array();
    }

    if (!r.contains("run")) r["run"] = json::object();
    auto& run = r["run"];
    if (!run.contains("checkpoint_every_n_samples")) run["checkpoint_every_n_samples"] = 1000;
    if (!run.contains("log_level")) run["log_level"] = "info";
    return r;
}

bool has_extension_ci(const std::string& path, const char* ext) {
    if (path.size() < std::strlen(ext)) return false;
    return std::equal(ext, ext + std::strlen(ext), path.end() - std::strlen(ext),
                      [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

// Cross-field rules not expressible in JSON Schema.
void validate_semantics(const json& r, const fs::path& config_dir) {
    if (r["model"].contains("path")) {
        const std::string model_path = r["model"]["path"].get<std::string>();
        if (!has_extension_ci(model_path, ".stl") && !has_extension_ci(model_path, ".obj"))
            throw MeshError("unsupported mesh extension (v1 accepts .stl/.obj): " + model_path);
        const fs::path mesh = model_path.empty() ? fs::path{}
                                                 : (fs::path(model_path).is_absolute()
                                                        ? fs::path(model_path)
                                                        : config_dir / model_path);
        if (!fs::exists(mesh)) throw MeshError("mesh file not found: " + mesh.string());
    }

    if (r["frequency"].contains("sweep")) {
        const auto& s = r["frequency"]["sweep"];
        const double start = s["start"].get<double>(), stop = s["stop"].get<double>(),
                     step = s["step"].get<double>();
        require_finite(start, "frequency.sweep.start");
        require_finite(stop, "frequency.sweep.stop");
        require_finite(step, "frequency.sweep.step");
        if (!(stop > start)) throw ConfigError("frequency.sweep requires stop > start");
        // Integer step indices avoid floating-point drift.
        const double steps = (stop - start) / step;
        if (std::abs(steps - std::llround(steps)) > 1e-9)
            throw ConfigError("frequency.sweep requires an integral number of steps");
    } else {
        require_finite(r["frequency"]["frequency_hz"].get<double>(), "frequency.frequency_hz");
    }

    const auto& medium = r["frequency"]["medium"];
    require_finite(medium["epsilon_r"].get<double>(), "medium.epsilon_r");
    require_finite(medium["mu_r"].get<double>(), "medium.mu_r");
    require_finite(medium["sigma"].get<double>(), "medium.sigma");
    require_finite(r["physics"]["incident_amplitude"].get<double>(),
                   "physics.incident_amplitude");

    if (r["model"].contains("transform")) {
        for (const char* key : {"translate", "rotate_euler_deg"})
            if (r["model"]["transform"].contains(key))
                for (const auto& v : r["model"]["transform"][key])
                    require_finite(v.get<double>(), "model.transform value");
    }
    if (r["mesh"]["symmetry"].value("enabled", false)) {
        const std::string plane = r["mesh"]["symmetry"].value("plane", "xz");
        const double tol = r["mesh"]["symmetry"].value("tolerance_m", 1e-4);
        const auto& pr = r["physics"]["phase_reference"];
        double coord = 0.0;
        if (plane == "xz") coord = pr[1].get<double>();
        else if (plane == "yz") coord = pr[0].get<double>();
        else if (plane == "xy") coord = pr[2].get<double>();
        if (std::abs(coord) > tol) {
            throw ConfigError("mesh.symmetry requires physics.phase_reference to lie on the symmetry plane");
        }
    }

    const auto& angles = r["angles"];
    if (angles.contains("azimuth")) {
        const auto& az = angles["azimuth"];
        if (!(az["stop"].get<double>() >= az["start"].get<double>()))
            throw ConfigError("angles.azimuth requires stop >= start");
        const auto& el = angles["elevation"];
        if (!(el["stop"].get<double>() >= el["start"].get<double>()))
            throw ConfigError("angles.elevation requires stop >= start");
    } else if (angles.contains("spherical")) {
        const auto& sph = angles["spherical"];
        require_finite(sph["num_points"].get<double>(), "angles.spherical.num_points");
        if (sph["num_points"].get<int>() <= 0)
            throw ConfigError("angles.spherical num_points must be positive");
        const double el_min = sph.value("elevation_min_deg", -90.0);
        const double el_max = sph.value("elevation_max_deg", 90.0);
        require_finite(el_min, "angles.spherical.elevation_min_deg");
        require_finite(el_max, "angles.spherical.elevation_max_deg");
        if (el_min < -90.0 || el_max > 90.0 || el_min > el_max)
            throw ConfigError("angles.spherical invalid elevation range");
    } else {
        for (const auto& entry : angles["list"]) {
            require_finite(entry[0].get<double>(), "angles.list azimuth");
            require_finite(entry[1].get<double>(), "angles.list elevation");
            const double el = entry[1].get<double>();
            if (el < -90.0 || el > 90.0)
                throw ConfigError("angles.list elevation out of [-90, 90]");
            if (entry.size() >= 4) {
                require_finite(entry[2].get<double>(), "angles.list obs azimuth");
                require_finite(entry[3].get<double>(), "angles.list obs elevation");
                const double obs_el = entry[3].get<double>();
                if (obs_el < -90.0 || obs_el > 90.0)
                    throw ConfigError("angles.list obs elevation out of [-90, 90]");
            }
        }
    }

    if (angles.contains("bistatic")) {
        const auto& bi = angles["bistatic"];
        const std::string mode = bi.value("mode", "fixed_offset");
        if (mode == "fixed_receiver") {
            if (!bi.contains("receiver_azimuth_deg") || !bi.contains("receiver_elevation_deg")) {
                throw ConfigError("angles.bistatic mode 'fixed_receiver' requires receiver_azimuth_deg and receiver_elevation_deg");
            }
        } else if (mode == "fixed_transmitter") {
            if (!bi.contains("transmitter_azimuth_deg") || !bi.contains("transmitter_elevation_deg")) {
                throw ConfigError("angles.bistatic mode 'fixed_transmitter' requires transmitter_azimuth_deg and transmitter_elevation_deg");
            }
        } else if (mode == "bistatic_angle") {
            if (!bi.contains("bistatic_angle_deg")) {
                throw ConfigError("angles.bistatic mode 'bistatic_angle' requires bistatic_angle_deg");
            }
        }
    }

    if (angles.contains("adaptive")) {
        if (!angles.contains("azimuth") || !angles.contains("elevation")) {
            throw ConfigError("angles.adaptive requires bounded angles.azimuth and angles.elevation ranges");
        }
        const auto& ad = angles["adaptive"];
        const double c_step = ad["coarse_step_deg"].get<double>();
        const double r_step = ad["refine_step_deg"].get<double>();
        if (!(r_step < c_step)) {
            throw ConfigError("angles.adaptive requires refine_step_deg < coarse_step_deg");
        }
        if (ad.contains("window_half_width_deg") && ad["window_half_width_deg"].get<double>() <= 0) {
            throw ConfigError("angles.adaptive window_half_width_deg must be positive");
        }
        if (ad.contains("gradient_threshold_db_per_deg") && ad["gradient_threshold_db_per_deg"].get<double>() <= 0) {
            throw ConfigError("angles.adaptive gradient_threshold_db_per_deg must be positive");
        }
    }

    if (r.contains("materials")) {
        const auto& mat_conf = r["materials"];
        const std::string def_mat = mat_conf.value("default_material", "pec");
        std::set<std::string> tags;
        tags.insert("pec");
        if (mat_conf.contains("definitions")) {
            for (const auto& def_j : mat_conf["definitions"]) {
                MaterialModel m = material_from_json(def_j);
                if (tags.find(m.tag) != tags.end() && m.tag != "pec")
                    throw ConfigError("duplicate material tag: " + m.tag);
                tags.insert(m.tag);
            }
        }
        if (tags.find(def_mat) == tags.end())
            throw ConfigError("default_material '" + def_mat + "' is not defined in materials");
    }

    if (r["run"].contains("shard")) {
        const auto& sh = r["run"]["shard"];
        const int shard_id = sh["shard_id"].get<int>();
        const int num_shards = sh["num_shards"].get<int>();
        if (num_shards <= 0) throw ConfigError("run.shard.num_shards must be at least 1");
        if (shard_id < 0 || shard_id >= num_shards)
            throw ConfigError("run.shard.shard_id must be in [0, num_shards - 1]");
    }

    const std::string output_path = r["output"]["path"].get<std::string>();
    const fs::path parent = fs::path(output_path).parent_path();
    if (!parent.empty() && !fs::is_directory(parent))
        throw ConfigError("output directory does not exist: " + parent.string());
}

// Axis values start + i*step while <= stop (+tolerance); never drifts.
std::vector<double> expand_axis(double start, double stop, double step) {
    const double tol = 1e-9 * std::max(1.0, std::abs(stop));
    const long long n = static_cast<long long>(std::floor((stop - start) / step + tol)) + 1;
    std::vector<double> values;
    for (long long i = 0; i < n; ++i) values.push_back(start + i * step);
    return values;
}

} // namespace

std::string canonical_json(const json& value) { return value.dump(); }

std::string sha256_hex(const std::string& bytes) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), digest);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char c : digest) out << std::setw(2) << static_cast<int>(c);
    return out.str();
}

ResolvedConfig load_config(const std::string& config_path, const std::string& schema_path) {
    const json raw = read_json_file(config_path, "config");
    const json schema = read_json_file(schema_path, "schema");
    try {
        json_validator validator;
        validator.set_root_schema(schema);
        validator.validate(raw);
    } catch (const ConfigError&) {
        throw;
    } catch (const std::exception& e) {
        throw ConfigError(std::string("schema validation failed: ") + e.what());
    }
    json resolved = resolve_defaults(raw);
    ResolvedConfig rc;
    if (resolved["run"].contains("resume_from_checkpoint")) {
        rc.resume_from_checkpoint =
            resolved["run"]["resume_from_checkpoint"].get<std::string>();
        resolved["run"].erase("resume_from_checkpoint");
    }
    const fs::path config_dir = fs::path(config_path).parent_path();
    validate_semantics(resolved, config_dir.empty() ? fs::current_path() : config_dir);
    rc.value = std::move(resolved);
    rc.hash = sha256_hex(canonical_json(rc.value));
    return rc;
}

SamplePlan build_sample_plan(const json& resolved) {
    SamplePlan plan;
    if (resolved["frequency"].contains("sweep")) {
        const auto& s = resolved["frequency"]["sweep"];
        const double start = s["start"].get<double>(), stop = s["stop"].get<double>(),
                     step = s["step"].get<double>();
        const long long n = std::llround((stop - start) / step) + 1;
        for (long long i = 0; i < n; ++i) plan.frequencies_hz.push_back(start + i * step);
    } else {
        plan.frequencies_hz.push_back(resolved["frequency"]["frequency_hz"].get<double>());
    }

    const bool monostatic = resolved.contains("physics") && resolved["physics"].contains("monostatic")
                                ? resolved["physics"]["monostatic"].get<bool>()
                                : true;
    const bool has_bistatic_config = resolved["angles"].contains("bistatic") || !monostatic;

    struct DirCandidate {
        double tx_az = 0.0;
        double tx_el = 0.0;
        double rx_az = 0.0;
        double rx_el = 0.0;
        bool is_bistatic = false;

        bool operator<(const DirCandidate& o) const {
            if (std::abs(tx_az - o.tx_az) > 1e-9) return tx_az < o.tx_az;
            if (std::abs(tx_el - o.tx_el) > 1e-9) return tx_el < o.tx_el;
            if (std::abs(rx_az - o.rx_az) > 1e-9) return rx_az < o.rx_az;
            return rx_el < o.rx_el;
        }
        bool operator==(const DirCandidate& o) const {
            return std::abs(tx_az - o.tx_az) < 1e-9 &&
                   std::abs(tx_el - o.tx_el) < 1e-9 &&
                   std::abs(rx_az - o.rx_az) < 1e-9 &&
                   std::abs(rx_el - o.rx_el) < 1e-9;
        }
    };

    auto map_bistatic = [&](double in_az, double in_el) -> DirCandidate {
        const double mono_rx_az = normalize_azimuth(in_az + 180.0);
        const double mono_rx_el = -in_el;
        DirCandidate c;
        c.tx_az = in_az;
        c.tx_el = in_el;
        c.rx_az = mono_rx_az;
        c.rx_el = mono_rx_el;
        c.is_bistatic = has_bistatic_config;
        if (!resolved["angles"].contains("bistatic")) {
            return c;
        }
        const auto& bi = resolved["angles"]["bistatic"];
        const std::string mode = bi.value("mode", "fixed_offset");
        if (mode == "fixed_receiver") {
            c.rx_az = normalize_azimuth(bi.value("receiver_azimuth_deg", 0.0));
            c.rx_el = bi.value("receiver_elevation_deg", 0.0);
        } else if (mode == "fixed_transmitter") {
            c.tx_az = normalize_azimuth(bi.value("transmitter_azimuth_deg", 0.0));
            c.tx_el = bi.value("transmitter_elevation_deg", 0.0);
            c.rx_az = in_az;
            c.rx_el = in_el;
        } else if (mode == "bistatic_angle") {
            const double beta = bi.value("bistatic_angle_deg", 0.0);
            c.rx_az = normalize_azimuth(mono_rx_az + beta);
            c.rx_el = mono_rx_el;
        } else { // fixed_offset
            c.rx_az = normalize_azimuth(mono_rx_az + bi.value("azimuth_offset_deg", 0.0));
            c.rx_el = std::clamp(mono_rx_el + bi.value("elevation_offset_deg", 0.0), -90.0, 90.0);
        }
        return c;
    };

    const auto& angles = resolved["angles"];
    if (angles.contains("spherical")) {
        const auto& sph = angles["spherical"];
        const size_t num_pts = sph["num_points"].get<size_t>();
        const double el_min = sph.value("elevation_min_deg", -90.0);
        const double el_max = sph.value("elevation_max_deg", 90.0);
        plan.directions = generate_fibonacci_sphere(num_pts, el_min, el_max);
        if (has_bistatic_config) {
            for (auto& d : plan.directions) {
                DirCandidate c = map_bistatic(d.azimuth_deg, d.elevation_deg);
                d.is_bistatic = true;
                d.bistatic_azimuth_deg = c.rx_az;
                d.bistatic_elevation_deg = c.rx_el;
                d.r_obs_hat = direction_from_az_el(c.rx_az, c.rx_el);
                const double cos_beta = std::clamp(-geom::dot(d.k_hat, d.r_obs_hat), -1.0, 1.0);
                d.bistatic_angle_deg = rad_to_deg(std::acos(cos_beta));
            }
        }
    } else {
        std::vector<DirCandidate> candidates;
        if (angles.contains("azimuth")) {
            const auto& az = angles["azimuth"], el = angles["elevation"];
            const double az_step = angles.contains("adaptive")
                                       ? angles["adaptive"]["coarse_step_deg"].get<double>()
                                       : az["step"].get<double>();
            auto az_values = expand_axis(az["start"].get<double>(), az["stop"].get<double>(), az_step);
            auto el_values = expand_axis(el["start"].get<double>(), el["stop"].get<double>(),
                                         el["step"].get<double>());
            for (double& a : az_values) a = normalize_azimuth(a);
            std::sort(az_values.begin(), az_values.end());
            std::sort(el_values.begin(), el_values.end());
            for (double a : az_values)
                for (double e : el_values) candidates.push_back(map_bistatic(a, e));
        } else {
            for (const auto& p : angles["list"]) {
                if (p.size() >= 4) {
                    DirCandidate c;
                    c.tx_az = normalize_azimuth(p[0].get<double>());
                    c.tx_el = p[1].get<double>();
                    c.rx_az = normalize_azimuth(p[2].get<double>());
                    c.rx_el = p[3].get<double>();
                    c.is_bistatic = true;
                    candidates.push_back(c);
                } else {
                    candidates.push_back(map_bistatic(normalize_azimuth(p[0].get<double>()), p[1].get<double>()));
                }
            }
        }
        // Canonical order, stable IDs, duplicate removal with warning (FR-4).
        std::sort(candidates.begin(), candidates.end());
        std::vector<DirCandidate> unique;
        for (const auto& c : candidates) {
            if (!unique.empty() && unique.back() == c)
                continue;
            unique.push_back(c);
        }
        if (unique.size() != candidates.size())
            plan.warnings.push_back("removed " + std::to_string(candidates.size() - unique.size()) +
                                    " duplicate direction(s)");
        uint32_t id = 0;
        for (const auto& c : unique) {
            Direction d;
            d.id = id++;
            d.azimuth_deg = c.tx_az;
            d.elevation_deg = c.tx_el;
            d.k_hat = direction_from_az_el(c.tx_az, c.tx_el);
            if (c.is_bistatic) {
                d.is_bistatic = true;
                d.bistatic_azimuth_deg = c.rx_az;
                d.bistatic_elevation_deg = c.rx_el;
                d.r_obs_hat = direction_from_az_el(c.rx_az, c.rx_el);
                const double cos_beta = std::clamp(-geom::dot(d.k_hat, d.r_obs_hat), -1.0, 1.0);
                d.bistatic_angle_deg = rad_to_deg(std::acos(cos_beta));
            } else {
                d.is_bistatic = false;
                d.bistatic_azimuth_deg = c.tx_az;
                d.bistatic_elevation_deg = c.tx_el;
                d.r_obs_hat = -d.k_hat;
                d.bistatic_angle_deg = 0.0;
            }
            plan.directions.push_back(d);
        }
    }

    for (const auto& p : resolved["polarization"]) plan.polarizations.push_back(p.get<std::string>());

    // Canonical plan encoding for resume identity (full round-trip precision).
    std::ostringstream plan_bytes;
    plan_bytes << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (double f : plan.frequencies_hz) plan_bytes << "f" << f << ";";
    for (const auto& d : plan.directions) {
        plan_bytes << "d" << d.azimuth_deg << "," << d.elevation_deg;
        if (d.is_bistatic) {
            plan_bytes << ":b" << d.bistatic_azimuth_deg << "," << d.bistatic_elevation_deg;
        }
        plan_bytes << ";";
    }
    for (const auto& p : plan.polarizations) plan_bytes << "p" << p << ";";
    plan.hash = sha256_hex(plan_bytes.str());
    return plan;
}

} // namespace strikecem

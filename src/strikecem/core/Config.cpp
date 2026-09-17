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
#include <sstream>

#include <nlohmann/json-schema.hpp>

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

    auto& angles = r["angles"];
    if (!angles.contains("deduplicate_periodic_endpoints"))
        angles["deduplicate_periodic_endpoints"] = true;

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

    if (!r.contains("mesh")) r["mesh"] = json::object();
    auto& mesh = r["mesh"];
    if (!mesh.contains("mode")) mesh["mode"] = "input";
    if (!mesh.contains("validate_orientation")) mesh["validate_orientation"] = true;
    if (!mesh.contains("max_aspect_ratio")) mesh["max_aspect_ratio"] = 20.0;

    if (!r.contains("execution")) r["execution"] = json::object();
    auto& exec = r["execution"];
    if (!exec.contains("cpu_threads")) exec["cpu_threads"] = 1;
    if (!exec.contains("batch_samples")) exec["batch_samples"] = 4096;
    if (!exec.contains("max_memory_mb")) exec["max_memory_mb"] = 16384;
    if (!exec.contains("deterministic")) exec["deterministic"] = true;

    auto& output = r["output"];
    if (!output.contains("format")) output["format"] = "hdf5";
    if (!output.contains("precision")) output["precision"] = 6;
    if (!output.contains("compression")) output["compression"] = false;
    if (!output.contains("include_metadata")) output["include_metadata"] = true;

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
    const std::string model_path = r["model"]["path"].get<std::string>();
    if (!has_extension_ci(model_path, ".stl") && !has_extension_ci(model_path, ".obj"))
        throw MeshError("unsupported mesh extension (v1 accepts .stl/.obj): " + model_path);
    const fs::path mesh = model_path.empty() ? fs::path{}
                                             : (fs::path(model_path).is_absolute()
                                                    ? fs::path(model_path)
                                                    : config_dir / model_path);
    if (!fs::exists(mesh)) throw MeshError("mesh file not found: " + mesh.string());

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

    const auto& angles = r["angles"];
    if (angles.contains("azimuth")) {
        const auto& az = angles["azimuth"];
        if (!(az["stop"].get<double>() >= az["start"].get<double>()))
            throw ConfigError("angles.azimuth requires stop >= start");
        const auto& el = angles["elevation"];
        if (!(el["stop"].get<double>() >= el["start"].get<double>()))
            throw ConfigError("angles.elevation requires stop >= start");
    } else {
        for (const auto& pair : angles["list"]) {
            require_finite(pair[0].get<double>(), "angles.list azimuth");
            require_finite(pair[1].get<double>(), "angles.list elevation");
            const double el = pair[1].get<double>();
            if (el < -90.0 || el > 90.0)
                throw ConfigError("angles.list elevation out of [-90, 90]");
        }
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

    std::vector<std::pair<double, double>> pairs;
    const auto& angles = resolved["angles"];
    if (angles.contains("azimuth")) {
        const auto& az = angles["azimuth"], el = angles["elevation"];
        auto az_values = expand_axis(az["start"].get<double>(), az["stop"].get<double>(),
                                     az["step"].get<double>());
        auto el_values = expand_axis(el["start"].get<double>(), el["stop"].get<double>(),
                                     el["step"].get<double>());
        for (double& a : az_values) a = normalize_azimuth(a);
        std::sort(az_values.begin(), az_values.end());
        std::sort(el_values.begin(), el_values.end());
        for (double a : az_values)
            for (double e : el_values) pairs.emplace_back(a, e);
    } else {
        for (const auto& p : angles["list"])
            pairs.emplace_back(normalize_azimuth(p[0].get<double>()), p[1].get<double>());
        std::sort(pairs.begin(), pairs.end());
    }
    // Canonical order, stable IDs, duplicate removal with warning (FR-4).
    std::vector<std::pair<double, double>> unique;
    for (const auto& p : pairs) {
        if (!unique.empty() && std::abs(unique.back().first - p.first) < 1e-9 &&
            std::abs(unique.back().second - p.second) < 1e-9)
            continue;
        unique.push_back(p);
    }
    if (unique.size() != pairs.size())
        plan.warnings.push_back("removed " + std::to_string(pairs.size() - unique.size()) +
                                " duplicate direction(s)");
    uint32_t id = 0;
    for (const auto& [az, el] : unique)
        plan.directions.push_back({id++, az, el, direction_from_az_el(az, el)});

    for (const auto& p : resolved["polarization"]) plan.polarizations.push_back(p.get<std::string>());

    // Canonical plan encoding for resume identity (full round-trip precision).
    std::ostringstream plan_bytes;
    plan_bytes << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (double f : plan.frequencies_hz) plan_bytes << "f" << f << ";";
    for (const auto& d : plan.directions)
        plan_bytes << "d" << d.azimuth_deg << "," << d.elevation_deg << ";";
    for (const auto& p : plan.polarizations) plan_bytes << "p" << p << ";";
    plan.hash = sha256_hex(plan_bytes.str());
    return plan;
}

} // namespace strikecem

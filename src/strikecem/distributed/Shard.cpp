#include "strikecem/distributed/Shard.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

#include <H5Cpp.h>

#include "strikecem/io/CsvWriter.hpp"
#include "strikecem/io/Hdf5Reader.hpp"
#include "strikecem/io/Hdf5Writer.hpp"
#include "strikecem/io/OutputError.hpp"
#include "strikecem/solvers/Material.hpp"

namespace strikecem {

namespace fs = std::filesystem;

namespace {

std::string generate_run_id() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    std::ostringstream ss;
    ss << "run-" << std::hex << ms;
    return ss.str();
}

std::string read_str_attr(const H5::H5File& file, const char* name) {
    if (!file.attrExists(name)) return "";
    const H5::Attribute attr = file.openAttribute(name);
    const H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    std::string val;
    attr.read(type, val);
    return val;
}

uint64_t read_u64_attr_or_default(const H5::H5File& file, const char* name, uint64_t def = 0) {
    if (!file.attrExists(name)) return def;
    const H5::Attribute attr = file.openAttribute(name);
    uint64_t val = 0;
    attr.read(H5::PredType::NATIVE_UINT64, &val);
    return val;
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    std::istringstream ss(line);
    while (std::getline(ss, field, ',')) out.push_back(field);
    return out;
}

struct ShardData {
    std::string config_hash;
    std::string geometry_hash;
    std::string normalized_mesh_hash;
    std::string sample_plan_hash;
    std::string run_id;
    nlohmann::json config_json;
    std::vector<PoSampleResult> samples;
};

ShardData read_csv_shard(const fs::path& csv_path) {
    ShardData sd;
    const fs::path sidecar_path = csv_path.string() + ".json";
    if (fs::exists(sidecar_path)) {
        std::ifstream sc_in(sidecar_path);
        if (sc_in.good()) {
            const auto sidecar = nlohmann::json::parse(sc_in);
            sd.config_hash = sidecar.value("config_hash", "");
            sd.geometry_hash = sidecar.value("geometry_hash", "");
            sd.normalized_mesh_hash = sidecar.value("normalized_mesh_hash", "");
            if (sidecar.contains("sample_plan")) {
                sd.sample_plan_hash = sidecar["sample_plan"].value("hash", "");
            }
            if (sidecar.contains("shard")) {
                sd.run_id = sidecar["shard"].value("run_id", "");
            }
            if (sidecar.contains("resolved_config")) {
                sd.config_json = sidecar["resolved_config"];
            }
        }
    }

    std::ifstream in(csv_path);
    if (!in) throw OutputError("cannot open CSV shard: " + csv_path.string());
    std::string line;
    if (!std::getline(in, line)) throw OutputError("empty CSV shard: " + csv_path.string());
    // Parse header to confirm columns
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto cols = split_csv(line);
        if (cols.size() < 10) continue;
        PoSampleResult r;
        r.sample_id = std::stoull(cols[0]);
        // Format: sample_id, az, el, freq, pol, real, imag, rcs_sqm, rcs_dbsm, valid, status
        const double re = std::stod(cols[5]);
        const double im = std::stod(cols[6]);
        r.scattering = {re, im};
        r.rcs_sqm = std::stod(cols[7]);
        sd.samples.push_back(r);
    }
    return sd;
}

ShardData read_hdf5_shard(const fs::path& h5_path) {
    ShardData sd;
    H5::Exception::dontPrint();
    try {
        H5::H5File file(h5_path.string(), H5F_ACC_RDONLY);
        sd.config_hash = read_str_attr(file, "config_hash");
        sd.geometry_hash = read_str_attr(file, "geometry_hash");
        sd.normalized_mesh_hash = read_str_attr(file, "normalized_mesh_hash");
        sd.sample_plan_hash = read_str_attr(file, "sample_plan_hash");
        sd.run_id = read_str_attr(file, "run_id");
        const std::string cfg_str = read_str_attr(file, "config_json");
        if (!cfg_str.empty()) {
            sd.config_json = nlohmann::json::parse(cfg_str);
        }

        H5::Group samples = file.openGroup("/samples");
        H5::DataSet id_ds = samples.openDataSet("sample_id");
        H5::DataSet di_ds = samples.openDataSet("direction_index");
        H5::DataSet fi_ds = samples.openDataSet("frequency_index");
        H5::DataSet pi_ds = samples.openDataSet("polarization_index");
        H5::DataSet sr_ds = samples.openDataSet("scattering_real");
        H5::DataSet si_ds = samples.openDataSet("scattering_imag");
        H5::DataSet sqm_ds = samples.openDataSet("rcs_sqm");

        hsize_t n = 0;
        id_ds.getSpace().getSimpleExtentDims(&n);
        std::vector<uint64_t> ids(n);
        std::vector<uint32_t> di(n), fi(n);
        std::vector<uint16_t> pi(n);
        std::vector<double> sr(n), si(n), sqm(n);

        if (n > 0) {
            id_ds.read(ids.data(), H5::PredType::NATIVE_UINT64);
            di_ds.read(di.data(), H5::PredType::NATIVE_UINT32);
            fi_ds.read(fi.data(), H5::PredType::NATIVE_UINT32);
            pi_ds.read(pi.data(), H5::PredType::NATIVE_UINT16);
            sr_ds.read(sr.data(), H5::PredType::NATIVE_DOUBLE);
            si_ds.read(si.data(), H5::PredType::NATIVE_DOUBLE);
            sqm_ds.read(sqm.data(), H5::PredType::NATIVE_DOUBLE);
        }

        sd.samples.resize(n);
        for (hsize_t i = 0; i < n; ++i) {
            sd.samples[i].sample_id = ids[i];
            sd.samples[i].direction_id = di[i];
            sd.samples[i].frequency_id = fi[i];
            sd.samples[i].pol_id = pi[i];
            sd.samples[i].scattering = {sr[i], si[i]};
            sd.samples[i].rcs_sqm = sqm[i];
        }
        file.close();
    } catch (const H5::Exception& e) {
        throw OutputError("cannot read HDF5 shard " + h5_path.string() + ": " + e.getDetailMsg());
    }
    return sd;
}

} // namespace

ShardManifest create_shards(const ResolvedConfig& rc, const SamplePlan& plan,
                            const NormalizedMesh& mesh, const std::string& config_path,
                            size_t num_shards, const std::string& out_dir,
                            const std::string& run_id) {
    if (num_shards == 0) throw ConfigError("num_shards must be positive");
    const std::string actual_run_id = run_id.empty() ? generate_run_id() : run_id;
    const fs::path dir = fs::absolute(out_dir);
    std::error_code ec;
    fs::create_directories(dir, ec);

    const size_t ndir = plan.directions.size();
    const size_t nfreq = plan.frequencies_hz.size();
    const size_t npol = plan.polarizations.size();
    const size_t total_units = nfreq * ndir;
    const uint64_t total_samples = static_cast<uint64_t>(total_units) * npol;

    const size_t base_units = total_units / num_shards;
    const size_t rem_units = total_units % num_shards;

    const std::string format = rc.value["output"].value("format", "hdf5");
    const std::string ext = (format == "csv") ? ".csv" : ".h5";

    ShardManifest manifest;
    manifest.version = "1.0";
    manifest.run_id = actual_run_id;
    manifest.config_path = fs::absolute(config_path).string();
    manifest.config_hash = rc.hash;
    manifest.geometry_hash = mesh.geometry_hash;
    manifest.normalized_mesh_hash = mesh.normalized_mesh_hash;
    manifest.sample_plan_hash = plan.hash;
    manifest.output_format = format;
    manifest.total_units = total_units;
    manifest.total_samples = total_samples;
    manifest.num_shards = num_shards;

    for (size_t s = 0; s < num_shards; ++s) {
        const size_t u_count = base_units + (s < rem_units ? 1 : 0);
        const size_t u_start = s * base_units + std::min(s, rem_units);
        const uint64_t s_id_start = static_cast<uint64_t>(u_start) * npol;
        const uint64_t s_id_count = static_cast<uint64_t>(u_count) * npol;

        std::ostringstream prefix;
        prefix << "shard_" << std::setfill('0') << std::setw(4) << s;
        const std::string stem = prefix.str();
        const fs::path shard_cfg_path = dir / (stem + ".json");
        const fs::path shard_out_path = dir / (stem + ext);

        auto shard_json = rc.value;
        shard_json["output"]["path"] = shard_out_path.string();
        shard_json["run"]["shard"] = {
            {"shard_id", s},
            {"num_shards", num_shards},
            {"run_id", actual_run_id},
            {"worker_identity", "worker-" + std::to_string(s)},
            {"parent_config_hash", rc.hash}
        };

        std::ofstream cfg_out(shard_cfg_path);
        if (!cfg_out) throw OutputError("cannot write shard config: " + shard_cfg_path.string());
        cfg_out << shard_json.dump(2) << "\n";
        cfg_out.flush();

        ShardDescriptor desc;
        desc.shard_id = s;
        desc.unit_start = u_start;
        desc.unit_count = u_count;
        desc.sample_id_start = s_id_start;
        desc.sample_id_count = s_id_count;
        desc.config_path = shard_cfg_path.string();
        desc.output_path = shard_out_path.string();
        manifest.shards.push_back(desc);
    }

    nlohmann::json manifest_j;
    manifest_j["scem_manifest_version"] = manifest.version;
    manifest_j["run_id"] = manifest.run_id;
    manifest_j["config_path"] = manifest.config_path;
    manifest_j["config_hash"] = manifest.config_hash;
    manifest_j["geometry_hash"] = manifest.geometry_hash;
    manifest_j["normalized_mesh_hash"] = manifest.normalized_mesh_hash;
    manifest_j["sample_plan_hash"] = manifest.sample_plan_hash;
    manifest_j["output_format"] = manifest.output_format;
    manifest_j["total_units"] = manifest.total_units;
    manifest_j["total_samples"] = manifest.total_samples;
    manifest_j["num_shards"] = manifest.num_shards;
    manifest_j["shards"] = nlohmann::json::array();
    for (const auto& d : manifest.shards) {
        manifest_j["shards"].push_back({
            {"shard_id", d.shard_id},
            {"unit_start", d.unit_start},
            {"unit_count", d.unit_count},
            {"sample_id_start", d.sample_id_start},
            {"sample_id_count", d.sample_id_count},
            {"config_path", d.config_path},
            {"output_path", d.output_path}
        });
    }

    const fs::path manifest_path = dir / "shard_manifest.json";
    std::ofstream mf_out(manifest_path);
    if (!mf_out) throw OutputError("cannot write shard manifest: " + manifest_path.string());
    mf_out << manifest_j.dump(2) << "\n";
    mf_out.flush();

    return manifest;
}

void merge_shards(const std::string& manifest_or_shard_path,
                  const std::vector<std::string>& additional_shards,
                  const std::string& output_path,
                  const std::string& schema_path) {
    std::vector<fs::path> shard_paths;
    nlohmann::json manifest_meta = nlohmann::json::object();
    const fs::path first_path(manifest_or_shard_path);

    bool from_manifest = false;
    if (fs::exists(first_path) && first_path.extension() == ".json") {
        try {
            std::ifstream in(first_path);
            const auto j = nlohmann::json::parse(in);
            if (j.contains("scem_manifest_version") && j.contains("shards")) {
                from_manifest = true;
                manifest_meta = j;
                const fs::path base_dir = first_path.parent_path();
                for (const auto& s : j["shards"]) {
                    fs::path sp(s["output_path"].get<std::string>());
                    if (sp.is_relative()) sp = base_dir / sp;
                    shard_paths.push_back(sp);
                }
            }
        } catch (...) {
            // Not a manifest, treat as individual shard
        }
    }

    if (!from_manifest) {
        shard_paths.push_back(first_path);
        for (const auto& p : additional_shards) shard_paths.push_back(p);
    }

    if (shard_paths.empty()) throw OutputError("merge rejected: no shards provided");

    std::string expected_cfg_hash = manifest_meta.is_object() ? manifest_meta.value("config_hash", "") : "";
    std::string expected_geom_hash = manifest_meta.is_object() ? manifest_meta.value("geometry_hash", "") : "";
    std::string expected_norm_hash = manifest_meta.is_object() ? manifest_meta.value("normalized_mesh_hash", "") : "";
    std::string expected_plan_hash = manifest_meta.is_object() ? manifest_meta.value("sample_plan_hash", "") : "";
    nlohmann::json parent_config;

    std::map<uint64_t, PoSampleResult> merged_by_id;

    for (const auto& sp : shard_paths) {
        if (!fs::exists(sp)) throw OutputError("merge rejected: missing shard file: " + sp.string());
        ShardData sd;
        if (sp.extension() == ".csv") {
            sd = read_csv_shard(sp);
        } else {
            sd = read_hdf5_shard(sp);
        }

        if (parent_config.is_null() && !sd.config_json.is_null()) {
            parent_config = sd.config_json;
        }

        if (expected_cfg_hash.empty()) expected_cfg_hash = sd.config_hash;
        else if (!sd.config_hash.empty() && sd.config_hash != expected_cfg_hash)
            throw OutputError("merge rejected: config_hash mismatch in " + sp.string());

        if (expected_geom_hash.empty()) expected_geom_hash = sd.geometry_hash;
        else if (!sd.geometry_hash.empty() && sd.geometry_hash != expected_geom_hash)
            throw OutputError("merge rejected: geometry_hash mismatch in " + sp.string());

        if (expected_plan_hash.empty()) expected_plan_hash = sd.sample_plan_hash;
        else if (!sd.sample_plan_hash.empty() && sd.sample_plan_hash != expected_plan_hash)
            throw OutputError("merge rejected: sample_plan_hash mismatch in " + sp.string());

        for (const auto& sample : sd.samples) {
            auto [it, inserted] = merged_by_id.insert({sample.sample_id, sample});
            if (!inserted) {
                throw OutputError("merge rejected: duplicate sample_id " + std::to_string(sample.sample_id));
            }
        }
    }

    uint64_t total_samples = (manifest_meta.is_object() && manifest_meta.contains("total_samples"))
                                 ? manifest_meta["total_samples"].get<uint64_t>()
                                 : 0ULL;
    if (total_samples == 0) {
        total_samples = merged_by_id.size();
    }

    if (merged_by_id.size() != total_samples) {
        throw OutputError("merge rejected: expected " + std::to_string(total_samples) +
                          " samples but received " + std::to_string(merged_by_id.size()));
    }

    // Verify complete contiguous coverage: 0 to total_samples - 1
    for (uint64_t id = 0; id < total_samples; ++id) {
        if (merged_by_id.find(id) == merged_by_id.end()) {
            throw OutputError("merge rejected: missing sample_id " + std::to_string(id));
        }
    }

    // Prepare merged PoResult
    PoResult merged_result;
    merged_result.samples.reserve(total_samples);
    for (uint64_t id = 0; id < total_samples; ++id) {
        merged_result.samples.push_back(merged_by_id[id]);
    }

    // Construct parent ResolvedConfig
    ResolvedConfig rc;
    if (!schema_path.empty() && from_manifest && manifest_meta.contains("config_path") &&
        fs::exists(manifest_meta["config_path"].get<std::string>())) {
        rc = load_config(manifest_meta["config_path"].get<std::string>(), schema_path);
    } else if (!parent_config.is_null() && parent_config.is_object()) {
        rc.value = parent_config;
        rc.schema_version = parent_config.value("scem_schema_version", "1.0");
        rc.hash = expected_cfg_hash;
    } else {
        throw OutputError("merge rejected: cannot resolve parent configuration");
    }

    rc.value["output"]["path"] = output_path;
    rc.value["run"].erase("shard");

    SamplePlan plan = build_sample_plan(rc.value);
    const size_t npol = plan.polarizations.size();
    const size_t ndir = plan.directions.size();
    for (auto& s : merged_result.samples) {
        const uint64_t u_idx = s.sample_id / npol;
        s.pol_id = static_cast<uint32_t>(s.sample_id % npol);
        s.direction_id = static_cast<uint32_t>(u_idx % ndir);
        s.frequency_id = static_cast<uint32_t>(u_idx / ndir);
    }
    NormalizedMesh mesh = load_normalized_mesh(rc.value, fs::path(output_path).parent_path(), rc.schema_version);
    const DesignerIdentity designer = {};
    EdgeModel edge_model;
    FringeOptions fringe;
    if (rc.value["solver"]["po_options"].value("edge_correction", "none") == "fringe") {
        edge_model = extract_edges(mesh);
        fringe = FringeOptions{true, &edge_model};
    }
    const ResolvedMaterials resolved_mat = resolve_materials(rc.value, mesh);
    MaterialOptions material;
    material.enabled = rc.value.contains("materials") || resolved_mat.has_materials;
    material.models = resolved_mat.models;
    material.facet_material_indices = resolved_mat.facet_material_indices;

    const std::string out_fmt = rc.value["output"].value("format", "hdf5");
    if (out_fmt == "csv" || fs::path(output_path).extension() == ".csv") {
        rc.value["output"]["format"] = "csv";
        write_csv_and_sidecar(rc, plan, mesh, merged_result, designer, fringe, material);
    } else {
        rc.value["output"]["format"] = "hdf5";
        write_hdf5_output(rc, plan, mesh, merged_result, designer, fringe, material);
    }
}

} // namespace strikecem

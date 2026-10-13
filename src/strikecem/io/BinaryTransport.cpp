#include "strikecem/io/BinaryTransport.hpp"

#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#ifdef __linux__
#include <unistd.h>
#endif

#include "strikecem/core/Conventions.hpp"
#include "strikecem/io/Checksum.hpp"

namespace strikecem {
namespace {

namespace fs = std::filesystem;

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string hostname() {
#ifdef __linux__
    char buf[256] = {};
    if (::gethostname(buf, sizeof(buf) - 1) == 0) return buf;
#endif
    return "unknown";
}

} // namespace

void write_scem_output(const ResolvedConfig& rc, const SamplePlan& plan,
                       const NormalizedMesh& mesh, const PoResult& result,
                       const DesignerIdentity& designer,
                       const FringeOptions& fringe,
                       const MaterialOptions& material) {
    const auto& output = rc.value["output"];
    const fs::path out_path(output["path"].get<std::string>());
    const std::string completed = utc_now();

    const bool is_shard = rc.value["run"].contains("shard");
    const std::string parent_cfg_hash =
        (is_shard && rc.value["run"]["shard"].contains("parent_config_hash"))
            ? rc.value["run"]["shard"]["parent_config_hash"].get<std::string>()
            : rc.hash;

    nlohmann::json metadata;
    metadata["scem_schema_version"] = rc.schema_version;
    metadata["output_format_version"] = "1.0";
    metadata["tool"] = {{"name", "strikecem"},
                        {"version", SCEM_VERSION},
                        {"git_revision", SCEM_GIT_REVISION},
                        {"build_type", SCEM_BUILD_TYPE},
                        {"compiler", std::string(SCEM_COMPILER_ID) + " " + SCEM_COMPILER_VERSION}};
    metadata["host"] = {{"hostname", hostname()},
                        {"device", rc.value["execution"].value("accelerator", "cpu")}};
    metadata["completed_utc"] = completed;
    metadata["config_hash"] = parent_cfg_hash;
    if (is_shard) {
        metadata["shard_config_hash"] = rc.hash;
        metadata["shard"] = rc.value["run"]["shard"];
    }
    if (designer.packaged && !designer.design_id.empty()) {
        metadata["design_id"] = designer.design_id;
        metadata["design_revision"] = designer.revision;
        metadata["export_id"] = designer.export_id;
    } else if (rc.value.contains("integration") && rc.value["integration"].contains("designer")) {
        metadata["design_id"] = rc.value["integration"]["designer"].value("design_id", "");
        metadata["design_revision"] = rc.value["integration"]["designer"].value("revision", "");
        metadata["export_id"] = rc.value["integration"]["designer"].value("export_id", "");
    }
    metadata["geometry_hash"] = mesh.geometry_hash;
    metadata["normalized_mesh_hash"] = mesh.normalized_mesh_hash;
    metadata["sample_plan"] = {{"frequencies", plan.frequencies_hz.size()},
                               {"directions", plan.directions.size()},
                               {"polarizations", plan.polarizations},
                               {"samples", plan.sample_count()},
                               {"hash", plan.hash}};
    metadata["solver"] = {{"type", rc.value["solver"]["type"]},
                          {"precision", rc.value["solver"]["precision"]},
                          {"edge_correction",
                           rc.value["solver"]["po_options"].value("edge_correction", "none")},
                          {"fringe_edges", fringe.enabled && fringe.edges != nullptr
                                               ? fringe.edges->edges.size()
                                               : 0}};
    if (material.enabled) {
        metadata["materials"] = {{"enabled", true}, {"models_count", material.models.size()}};
    }
    metadata["units"] = {
        {"azimuth", "deg"},
        {"elevation", "deg"},
        {"frequency", "Hz"},
        {"rcs_sqm", "m^2"},
        {"rcs_dbsm", "dBsm"},
        {"coordinates", "m"}
    };
    metadata["resolved_config"] = rc.value;

    const std::string meta_str = metadata.dump();

    // Table data
    const std::vector<double>& freqs = plan.frequencies_hz;

    std::vector<ScemDirectionRecord> dir_records;
    dir_records.reserve(plan.directions.size());
    for (const auto& d : plan.directions) {
        ScemDirectionRecord rec;
        rec.azimuth_deg = d.azimuth_deg;
        rec.elevation_deg = d.elevation_deg;
        rec.k_x = d.k_hat.x;
        rec.k_y = d.k_hat.y;
        rec.k_z = d.k_hat.z;
        dir_records.push_back(rec);
    }

    std::vector<ScemPolarizationRecord> pol_records;
    pol_records.reserve(plan.polarizations.size());
    for (const auto& p : plan.polarizations) {
        ScemPolarizationRecord rec;
        std::memset(rec.name, 0, sizeof(rec.name));
        std::strncpy(rec.name, p.c_str(), sizeof(rec.name) - 1);
        pol_records.push_back(rec);
    }

    const bool float32 = (rc.value["solver"]["precision"].get<std::string>() == "float32");
    std::vector<ScemSampleRowF64> rows64;
    std::vector<ScemSampleRowF32> rows32;
    if (float32) {
        rows32.reserve(result.samples.size());
        for (const auto& s : result.samples) {
            ScemSampleRowF32 r;
            r.sample_id = s.sample_id;
            r.direction_index = s.direction_id;
            r.frequency_index = s.frequency_id;
            r.polarization_index = static_cast<uint16_t>(s.pol_id);
            r.valid = 1;
            r.status_code = 1; // ok
            r.scattering_real = static_cast<float>(s.scattering.real());
            r.scattering_imag = static_cast<float>(s.scattering.imag());
            r.rcs_sqm = static_cast<float>(s.rcs_sqm);
            r.rcs_dbsm = static_cast<float>(rcs_sqm_to_dbsm(s.rcs_sqm));
            rows32.push_back(r);
        }
    } else {
        rows64.reserve(result.samples.size());
        for (const auto& s : result.samples) {
            ScemSampleRowF64 r;
            r.sample_id = s.sample_id;
            r.direction_index = s.direction_id;
            r.frequency_index = s.frequency_id;
            r.polarization_index = static_cast<uint16_t>(s.pol_id);
            r.valid = 1;
            r.status_code = 1; // ok
            r.scattering_real = s.scattering.real();
            r.scattering_imag = s.scattering.imag();
            r.rcs_sqm = s.rcs_sqm;
            r.rcs_dbsm = rcs_sqm_to_dbsm(s.rcs_sqm);
            rows64.push_back(r);
        }
    }

    ScemHeader header;
    header.version = 1;
    header.header_size = sizeof(ScemHeader);
    header.endianness_marker = 0x01020304;
    header.dtype = float32 ? 0 : 1;
    header.flags = 0;
    header.sample_count = result.samples.size();
    header.frequency_count = static_cast<uint32_t>(freqs.size());
    header.direction_count = static_cast<uint32_t>(dir_records.size());
    header.polarization_count = static_cast<uint16_t>(pol_records.size());

    uint64_t offset = sizeof(ScemHeader);
    header.frequencies_offset = offset;
    header.frequencies_bytes = freqs.size() * sizeof(double);
    offset += header.frequencies_bytes;

    header.directions_offset = offset;
    header.directions_bytes = dir_records.size() * sizeof(ScemDirectionRecord);
    offset += header.directions_bytes;

    header.polarizations_offset = offset;
    header.polarizations_bytes = pol_records.size() * sizeof(ScemPolarizationRecord);
    offset += header.polarizations_bytes;

    header.metadata_offset = offset;
    header.metadata_bytes = meta_str.size();
    offset += header.metadata_bytes;

    header.samples_offset = offset;
    header.samples_bytes = float32 ? (rows32.size() * sizeof(ScemSampleRowF32))
                                   : (rows64.size() * sizeof(ScemSampleRowF64));
    offset += header.samples_bytes;

    // Checksums
    uint64_t data_hash = 14695981039346656037ULL;
    if (header.frequencies_bytes > 0)
        data_hash = fnv1a_64(freqs.data(), header.frequencies_bytes, data_hash);
    if (header.directions_bytes > 0)
        data_hash = fnv1a_64(dir_records.data(), header.directions_bytes, data_hash);
    if (header.polarizations_bytes > 0)
        data_hash = fnv1a_64(pol_records.data(), header.polarizations_bytes, data_hash);
    if (header.metadata_bytes > 0)
        data_hash = fnv1a_64(meta_str.data(), header.metadata_bytes, data_hash);
    if (header.samples_bytes > 0) {
        if (float32)
            data_hash = fnv1a_64(rows32.data(), header.samples_bytes, data_hash);
        else
            data_hash = fnv1a_64(rows64.data(), header.samples_bytes, data_hash);
    }
    header.data_checksum = data_hash;
    header.header_checksum = fnv1a_64(&header, offsetof(ScemHeader, header_checksum));

    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) throw OutputError("cannot open output file: " + out_path.string());

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (header.frequencies_bytes > 0)
        out.write(reinterpret_cast<const char*>(freqs.data()), header.frequencies_bytes);
    if (header.directions_bytes > 0)
        out.write(reinterpret_cast<const char*>(dir_records.data()), header.directions_bytes);
    if (header.polarizations_bytes > 0)
        out.write(reinterpret_cast<const char*>(pol_records.data()), header.polarizations_bytes);
    if (header.metadata_bytes > 0)
        out.write(meta_str.data(), header.metadata_bytes);
    if (header.samples_bytes > 0) {
        if (float32)
            out.write(reinterpret_cast<const char*>(rows32.data()), header.samples_bytes);
        else
            out.write(reinterpret_cast<const char*>(rows64.data()), header.samples_bytes);
    }
    out.flush();
    if (!out) throw OutputError("failed writing scem binary file: " + out_path.string());
}

ScemDatabase read_scem_database(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw ReaderError("cannot open scem file: " + path);

    const std::streamsize file_size = in.tellg();
    if (file_size < static_cast<std::streamsize>(sizeof(ScemHeader))) {
        throw ReaderError("file smaller than SCEM header: " + path);
    }

    in.seekg(0, std::ios::beg);
    ScemHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in) throw ReaderError("failed reading scem header: " + path);

    if (std::memcmp(header.magic, "SCEM", 4) != 0) {
        throw ReaderError("invalid SCEM magic header: " + path);
    }

    if (header.version != 1) {
        throw ReaderError("unsupported SCEM format version: " + std::to_string(header.version));
    }

    if (header.endianness_marker == 0x04030201) {
        throw ReaderError("opposite endianness detected; byte swap not yet supported: " + path);
    } else if (header.endianness_marker != 0x01020304) {
        throw ReaderError("corrupted endianness marker in scem file: " + path);
    }

    const uint64_t expected_hdr_cs = fnv1a_64(&header, offsetof(ScemHeader, header_checksum));
    if (expected_hdr_cs != header.header_checksum) {
        throw ReaderError("corrupted scem header checksum in " + path);
    }

    const uint64_t total_expected_bytes = header.samples_offset + header.samples_bytes;
    if (total_expected_bytes > static_cast<uint64_t>(file_size)) {
        throw ReaderError("scem payload truncated or table offset out of bounds: " + path);
    }

    // Verify payload data checksum
    std::vector<char> payload(file_size - sizeof(ScemHeader));
    in.seekg(sizeof(ScemHeader), std::ios::beg);
    if (!payload.empty()) {
        in.read(payload.data(), payload.size());
        if (!in) throw ReaderError("failed reading scem payload bytes: " + path);
        const uint64_t computed_data_cs = fnv1a_64(payload.data(), payload.size());
        if (computed_data_cs != header.data_checksum) {
            throw ReaderError("corrupted scem data checksum in " + path);
        }
    }

    ScemDatabase db;
    db.version = header.version;
    db.dtype = header.dtype;
    db.payload_checksum = header.data_checksum;

    // Frequencies
    db.frequencies_hz.resize(header.frequency_count);
    if (header.frequencies_bytes > 0) {
        std::memcpy(db.frequencies_hz.data(),
                    payload.data() + (header.frequencies_offset - sizeof(ScemHeader)),
                    header.frequencies_bytes);
    }

    // Directions
    std::vector<ScemDirectionRecord> dir_recs(header.direction_count);
    if (header.directions_bytes > 0) {
        std::memcpy(dir_recs.data(),
                    payload.data() + (header.directions_offset - sizeof(ScemHeader)),
                    header.directions_bytes);
    }
    db.directions.resize(header.direction_count);
    for (size_t i = 0; i < header.direction_count; ++i) {
        db.directions[i].id = static_cast<uint32_t>(i);
        db.directions[i].azimuth_deg = dir_recs[i].azimuth_deg;
        db.directions[i].elevation_deg = dir_recs[i].elevation_deg;
        db.directions[i].k_hat = geom::Vec3d{dir_recs[i].k_x, dir_recs[i].k_y, dir_recs[i].k_z};
    }

    // Polarizations
    std::vector<ScemPolarizationRecord> pol_recs(header.polarization_count);
    if (header.polarizations_bytes > 0) {
        std::memcpy(pol_recs.data(),
                    payload.data() + (header.polarizations_offset - sizeof(ScemHeader)),
                    header.polarizations_bytes);
    }
    db.polarizations.resize(header.polarization_count);
    for (size_t i = 0; i < header.polarization_count; ++i) {
        char name[9] = {0};
        std::memcpy(name, pol_recs[i].name, sizeof(pol_recs[i].name));
        db.polarizations[i] = std::string(name);
    }

    // Metadata
    if (header.metadata_bytes > 0) {
        const std::string meta_str(
            payload.data() + (header.metadata_offset - sizeof(ScemHeader)),
            header.metadata_bytes);
        try {
            db.metadata = nlohmann::json::parse(meta_str);
        } catch (const std::exception& e) {
            throw ReaderError(std::string("failed parsing scem metadata JSON: ") + e.what());
        }
    }

    // Samples
    auto status_from_code = [](uint8_t c) -> std::string {
        switch (c) {
            case 0: return "pending";
            case 1: return "ok";
            case 2: return "warning";
            case 3: return "error";
            default: return "unknown";
        }
    };

    db.rows.resize(header.sample_count);
    if (header.dtype == 0) { // float32
        const auto* raw_rows = reinterpret_cast<const ScemSampleRowF32*>(
            payload.data() + (header.samples_offset - sizeof(ScemHeader)));
        for (size_t i = 0; i < header.sample_count; ++i) {
            db.rows[i].sample_id = raw_rows[i].sample_id;
            db.rows[i].direction_index = raw_rows[i].direction_index;
            db.rows[i].frequency_index = raw_rows[i].frequency_index;
            db.rows[i].polarization_index = raw_rows[i].polarization_index;
            db.rows[i].scattering_real = raw_rows[i].scattering_real;
            db.rows[i].scattering_imag = raw_rows[i].scattering_imag;
            db.rows[i].rcs_sqm = raw_rows[i].rcs_sqm;
            db.rows[i].rcs_dbsm = raw_rows[i].rcs_dbsm;
            db.rows[i].valid = (raw_rows[i].valid != 0);
            db.rows[i].status = status_from_code(raw_rows[i].status_code);
        }
    } else { // float64
        const auto* raw_rows = reinterpret_cast<const ScemSampleRowF64*>(
            payload.data() + (header.samples_offset - sizeof(ScemHeader)));
        for (size_t i = 0; i < header.sample_count; ++i) {
            db.rows[i].sample_id = raw_rows[i].sample_id;
            db.rows[i].direction_index = raw_rows[i].direction_index;
            db.rows[i].frequency_index = raw_rows[i].frequency_index;
            db.rows[i].polarization_index = raw_rows[i].polarization_index;
            db.rows[i].scattering_real = raw_rows[i].scattering_real;
            db.rows[i].scattering_imag = raw_rows[i].scattering_imag;
            db.rows[i].rcs_sqm = raw_rows[i].rcs_sqm;
            db.rows[i].rcs_dbsm = raw_rows[i].rcs_dbsm;
            db.rows[i].valid = (raw_rows[i].valid != 0);
            db.rows[i].status = status_from_code(raw_rows[i].status_code);
        }
    }

    return db;
}

void validate_scem_complete(const ScemDatabase& db) {
    const size_t expected_count =
        db.frequencies_hz.size() * db.directions.size() * db.polarizations.size();
    if (db.rows.size() != expected_count) {
        throw ReaderError("database incomplete: expected " + std::to_string(expected_count) +
                          " samples, got " + std::to_string(db.rows.size()));
    }
    for (size_t i = 0; i < db.rows.size(); ++i) {
        if (db.rows[i].sample_id != i) {
            throw ReaderError("non-sequential sample_id at index " + std::to_string(i));
        }
        if (!db.rows[i].valid) {
            throw ReaderError("invalid row at index " + std::to_string(i));
        }
        if (db.rows[i].status != "ok") {
            throw ReaderError("non-ok row status at index " + std::to_string(i));
        }
    }
}

bool check_existing_scem(const ResolvedConfig& rc, const SamplePlan& plan,
                         const NormalizedMesh& mesh, const std::string& path) {
    try {
        const ScemDatabase db = read_scem_database(path);
        validate_scem_complete(db);
        if (db.metadata.value("config_hash", "") != rc.hash) return false;
        if (db.metadata.value("geometry_hash", "") != mesh.geometry_hash) return false;
        if (db.metadata.value("normalized_mesh_hash", "") != mesh.normalized_mesh_hash) return false;
        if (db.metadata.contains("sample_plan") &&
            db.metadata["sample_plan"].value("hash", "") != plan.hash) return false;
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace strikecem

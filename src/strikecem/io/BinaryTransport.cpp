#include "strikecem/io/BinaryTransport.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#ifdef __linux__
#include <unistd.h>
#endif

#include "strikecem/core/Conventions.hpp"
#include "strikecem/io/Checksum.hpp"

namespace strikecem {
namespace {

namespace fs = std::filesystem;

bool checked_product(uint64_t count, uint64_t factor, uint64_t& product) {
    if (factor != 0 && count > std::numeric_limits<uint64_t>::max() / factor) {
        return false;
    }
    product = count * factor;
    return true;
}

bool region_within_file(uint64_t offset, uint64_t bytes, uint64_t file_size) {
    return offset >= sizeof(ScemHeader) && offset <= file_size && bytes <= file_size - offset;
}

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
    metadata["solver"] = configured_solver_dispatch(rc.value);
    metadata["solver"]["precision"] = rc.value["solver"]["precision"];
    metadata["solver"]["edge_correction"] =
        rc.value["solver"]["po_options"].value("edge_correction", "none");
    metadata["solver"]["fringe_edges"] = fringe.enabled && fringe.edges != nullptr
                                               ? fringe.edges->edges.size() : 0;
    if (!rc.provenance.empty()) metadata["provenance"] = rc.provenance;
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
    if (header.header_size != sizeof(ScemHeader)) {
        throw ReaderError("invalid SCEM header size: " + path);
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

    if (header.dtype != 0 && header.dtype != 1) {
        throw ReaderError("unsupported SCEM sample dtype: " + std::to_string(header.dtype));
    }
    if (header.metadata_bytes == 0) {
        throw ReaderError("SCEM metadata section must not be empty: " + path);
    }

    uint64_t expected_frequencies_bytes = 0;
    uint64_t expected_directions_bytes = 0;
    uint64_t expected_polarizations_bytes = 0;
    uint64_t expected_samples_bytes = 0;
    const uint64_t sample_record_size = header.dtype == 0
                                            ? sizeof(ScemSampleRowF32)
                                            : sizeof(ScemSampleRowF64);
    if (!checked_product(header.frequency_count, sizeof(double), expected_frequencies_bytes) ||
        !checked_product(header.direction_count, sizeof(ScemDirectionRecord),
                         expected_directions_bytes) ||
        !checked_product(header.polarization_count, sizeof(ScemPolarizationRecord),
                         expected_polarizations_bytes) ||
        !checked_product(header.sample_count, sample_record_size, expected_samples_bytes)) {
        throw ReaderError("SCEM table size overflow: " + path);
    }
    if (header.frequencies_bytes != expected_frequencies_bytes ||
        header.directions_bytes != expected_directions_bytes ||
        header.polarizations_bytes != expected_polarizations_bytes ||
        header.samples_bytes != expected_samples_bytes) {
        throw ReaderError("SCEM table byte counts do not match record counts: " + path);
    }

    const uint64_t file_size_u64 = static_cast<uint64_t>(file_size);
    uint64_t expected_offset = sizeof(ScemHeader);
    auto validate_section = [&](uint64_t offset, uint64_t bytes, const char* name) {
        if (offset != expected_offset ||
            !region_within_file(offset, bytes, file_size_u64)) {
            throw ReaderError(std::string("invalid or non-contiguous SCEM ") + name +
                              " range: " + path);
        }
        expected_offset += bytes;
    };
    validate_section(header.frequencies_offset, header.frequencies_bytes, "frequency table");
    validate_section(header.directions_offset, header.directions_bytes, "direction table");
    validate_section(header.polarizations_offset, header.polarizations_bytes, "polarization table");
    validate_section(header.metadata_offset, header.metadata_bytes, "metadata");
    validate_section(header.samples_offset, header.samples_bytes, "sample table");
    if (expected_offset != file_size_u64) {
        throw ReaderError("SCEM file has unreferenced payload bytes: " + path);
    }

    ScemDatabase db;
    if (header.frequency_count > std::vector<double>().max_size() ||
        header.direction_count > std::vector<Direction>().max_size() ||
        header.polarization_count > std::vector<std::string>().max_size() ||
        header.sample_count > std::vector<Hdf5SampleRow>().max_size() ||
        header.metadata_bytes > std::string().max_size()) {
        throw ReaderError("SCEM counts exceed supported in-memory limits: " + path);
    }

    // Stream the checksum so an input file does not need a second full-size buffer.
    uint64_t computed_data_cs = 14695981039346656037ULL;
    in.seekg(sizeof(ScemHeader), std::ios::beg);
    char checksum_buffer[64 * 1024];
    uint64_t checksum_remaining = file_size_u64 - sizeof(ScemHeader);
    while (checksum_remaining > 0) {
        const auto count = static_cast<std::streamsize>(
            std::min<uint64_t>(checksum_remaining, sizeof(checksum_buffer)));
        in.read(checksum_buffer, count);
        if (in.gcount() != count) throw ReaderError("failed reading SCEM payload: " + path);
        computed_data_cs = fnv1a_64(checksum_buffer, static_cast<size_t>(count), computed_data_cs);
        checksum_remaining -= static_cast<uint64_t>(count);
    }
    if (computed_data_cs != header.data_checksum) {
        throw ReaderError("corrupted scem data checksum in " + path);
    }

    db.version = header.version;
    db.dtype = header.dtype;
    db.payload_checksum = header.data_checksum;

    // Frequencies
    db.frequencies_hz.resize(header.frequency_count);
    if (header.frequencies_bytes > 0) {
        in.clear();
        in.seekg(static_cast<std::streamoff>(header.frequencies_offset), std::ios::beg);
        in.read(reinterpret_cast<char*>(db.frequencies_hz.data()),
                static_cast<std::streamsize>(header.frequencies_bytes));
        if (in.gcount() != static_cast<std::streamsize>(header.frequencies_bytes))
            throw ReaderError("truncated SCEM frequency table: " + path);
    }

    // Directions
    db.directions.resize(header.direction_count);
    in.clear();
    in.seekg(static_cast<std::streamoff>(header.directions_offset), std::ios::beg);
    std::vector<ScemDirectionRecord> direction_chunk(4096);
    for (size_t base = 0; base < header.direction_count;) {
        const size_t count = std::min(direction_chunk.size(),
                                      static_cast<size_t>(header.direction_count) - base);
        const auto bytes = static_cast<std::streamsize>(count * sizeof(ScemDirectionRecord));
        in.read(reinterpret_cast<char*>(direction_chunk.data()), bytes);
        if (in.gcount() != bytes) throw ReaderError("truncated SCEM direction table: " + path);
        for (size_t j = 0; j < count; ++j) {
            const auto& rec = direction_chunk[j];
            auto& direction = db.directions[base + j];
            direction.id = static_cast<uint32_t>(base + j);
            direction.azimuth_deg = rec.azimuth_deg;
            direction.elevation_deg = rec.elevation_deg;
            direction.k_hat = geom::Vec3d{rec.k_x, rec.k_y, rec.k_z};
        }
        base += count;
    }

    // Polarizations
    db.polarizations.resize(header.polarization_count);
    in.clear();
    in.seekg(static_cast<std::streamoff>(header.polarizations_offset), std::ios::beg);
    std::vector<ScemPolarizationRecord> polarization_records(header.polarization_count);
    if (header.polarizations_bytes > 0) {
        in.read(reinterpret_cast<char*>(polarization_records.data()),
                static_cast<std::streamsize>(header.polarizations_bytes));
        if (in.gcount() != static_cast<std::streamsize>(header.polarizations_bytes))
            throw ReaderError("truncated SCEM polarization table: " + path);
    }
    for (size_t i = 0; i < header.polarization_count; ++i) {
        char name[9] = {0};
        std::memcpy(name, polarization_records[i].name, sizeof(polarization_records[i].name));
        db.polarizations[i] = std::string(name);
    }

    // Metadata
    if (header.metadata_bytes > 0) {
        std::string meta_str(static_cast<size_t>(header.metadata_bytes), '\0');
        in.clear();
        in.seekg(static_cast<std::streamoff>(header.metadata_offset), std::ios::beg);
        in.read(meta_str.data(), static_cast<std::streamsize>(header.metadata_bytes));
        if (in.gcount() != static_cast<std::streamsize>(header.metadata_bytes))
            throw ReaderError("truncated SCEM metadata: " + path);
        try {
            db.metadata = nlohmann::json::parse(meta_str);
        } catch (const std::exception& e) {
            throw ReaderError(std::string("failed parsing scem metadata JSON: ") + e.what());
        }
        if (!db.metadata.is_object())
            throw ReaderError("SCEM metadata must be a JSON object: " + path);
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

    db.rows.resize(static_cast<size_t>(header.sample_count));
    in.clear();
    in.seekg(static_cast<std::streamoff>(header.samples_offset), std::ios::beg);
    auto copy_row = [&](const auto& source, Hdf5SampleRow& destination) {
        destination.sample_id = source.sample_id;
        destination.direction_index = source.direction_index;
        destination.frequency_index = source.frequency_index;
        destination.polarization_index = source.polarization_index;
        destination.scattering_real = source.scattering_real;
        destination.scattering_imag = source.scattering_imag;
        destination.rcs_sqm = source.rcs_sqm;
        destination.rcs_dbsm = source.rcs_dbsm;
        destination.valid = source.valid != 0;
        destination.status = status_from_code(source.status_code);
        if (destination.direction_index >= db.directions.size() ||
            destination.frequency_index >= db.frequencies_hz.size() ||
            destination.polarization_index >= db.polarizations.size()) {
            throw ReaderError("SCEM sample index is outside its table: " + path);
        }
    };

    constexpr size_t kRowsPerChunk = 8192;
    if (header.dtype == 0) {
        std::vector<ScemSampleRowF32> rows(std::min<size_t>(kRowsPerChunk, db.rows.size()));
        for (size_t base = 0; base < db.rows.size();) {
            const size_t count = std::min(rows.size(), db.rows.size() - base);
            const auto bytes = static_cast<std::streamsize>(count * sizeof(ScemSampleRowF32));
            in.read(reinterpret_cast<char*>(rows.data()), bytes);
            if (in.gcount() != bytes) throw ReaderError("truncated SCEM sample table: " + path);
            for (size_t i = 0; i < count; ++i) copy_row(rows[i], db.rows[base + i]);
            base += count;
        }
    } else {
        std::vector<ScemSampleRowF64> rows(std::min<size_t>(kRowsPerChunk, db.rows.size()));
        for (size_t base = 0; base < db.rows.size();) {
            const size_t count = std::min(rows.size(), db.rows.size() - base);
            const auto bytes = static_cast<std::streamsize>(count * sizeof(ScemSampleRowF64));
            in.read(reinterpret_cast<char*>(rows.data()), bytes);
            if (in.gcount() != bytes) throw ReaderError("truncated SCEM sample table: " + path);
            for (size_t i = 0; i < count; ++i) copy_row(rows[i], db.rows[base + i]);
            base += count;
        }
    }

    return db;
}

void validate_scem_complete(const ScemDatabase& db) {
    uint64_t expected_count = 0;
    uint64_t frequency_direction_count = 0;
    if (!checked_product(db.frequencies_hz.size(), db.directions.size(),
                         frequency_direction_count) ||
        !checked_product(frequency_direction_count, db.polarizations.size(), expected_count)) {
        throw ReaderError("database sample count overflow");
    }
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
        if (!rc.provenance.empty() &&
            db.metadata.value("provenance", nlohmann::json::object()) != rc.provenance)
            return false;
        if (db.metadata.contains("sample_plan") &&
            db.metadata["sample_plan"].value("hash", "") != plan.hash) return false;
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace strikecem

// Flat-row HDF5 writer with checksummed chunk commits (OUTPUT_FORMAT.md).
//
// Fresh runs create tables, provenance, and zeroed progress, then commit
// every chunk: data, flush, read-back checksum, flush, progress flag, flush.
// Resume verifies identity hashes, re-verifies committed checksums, solves
// only missing units, and commits touched chunks. Partial files stay
// queryable; resume never recomputes committed rows.
#include "strikecem/io/Hdf5Writer.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <map>

#include <H5Cpp.h>

#include "strikecem/io/Checksum.hpp"
#include "strikecem/io/DesignerPackage.hpp"
#include "strikecem/solvers/GpuPO.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

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

void write_str_attr(H5::H5Object& obj, const char* name, const std::string& value) {
    const H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    const H5::DataSpace space(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, type, space);
    attr.write(type, value);
}

void write_u64_attr(H5::H5Object& obj, const char* name, uint64_t value) {
    const H5::DataSpace space(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, H5::PredType::NATIVE_UINT64, space);
    attr.write(H5::PredType::NATIVE_UINT64, &value);
}

uint64_t read_u64_attr(H5::H5Object& obj, const char* name) {
    const H5::Attribute attr = obj.openAttribute(name);
    uint64_t value = 0;
    attr.read(H5::PredType::NATIVE_UINT64, &value);
    return value;
}

H5::DataSet create_vector(H5::Group& group, const char* name, size_t n,
                          const H5::PredType& type) {
    const hsize_t dims[1] = {n};
    const H5::DataSpace space(1, dims);
    return group.createDataSet(name, type, space);
}

H5::DataSet create_strings(H5::Group& group, const char* name, size_t n) {
    const H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    const hsize_t dims[1] = {n};
    const H5::DataSpace space(1, dims);
    return group.createDataSet(name, type, space);
}

void write_string_slice(H5::DataSet& ds, const std::vector<std::string>& values, size_t offset) {
    const H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    std::vector<const char*> ptrs;
    ptrs.reserve(values.size());
    for (const auto& s : values) ptrs.push_back(s.c_str());
    const hsize_t count[1] = {values.size()};
    const hsize_t start[1] = {offset};
    const H5::DataSpace mem(1, count);
    H5::DataSpace file_space = ds.getSpace();
    file_space.selectHyperslab(H5S_SELECT_SET, count, start);
    ds.write(ptrs.data(), type, mem, file_space);
}

void write_strings(H5::Group& group, const char* name, const std::vector<std::string>& values) {
    H5::DataSet ds = create_strings(group, name, values.size());
    write_string_slice(ds, values, 0);
}

void write_units(H5::DataSet& ds, const char* units) { write_str_attr(ds, "units", units); }

struct OpenDb {
    H5::H5File file;
    H5::DataSet id_ds, di_ds, fi_ds, pi_ds;
    H5::DataSet sr_ds, si_ds, sqm_ds, db_ds;
    H5::DataSet valid_ds, status_ds;
    H5::DataSet prog_ds, cksum_ds;
    size_t chunk_rows = 1000;
    size_t n_chunks = 0;
    size_t n_rows = 0;
    size_t n_pols = 0;
    bool float32 = false;
};

H5::PredType float_type(bool float32) {
    return float32 ? H5::PredType::NATIVE_FLOAT : H5::PredType::NATIVE_DOUBLE;
}

void open_datasets(OpenDb& db) {
    H5::Group samples = db.file.openGroup("/samples");
    H5::Group progress = db.file.openGroup("/progress");
    db.id_ds = samples.openDataSet("sample_id");
    db.di_ds = samples.openDataSet("direction_index");
    db.fi_ds = samples.openDataSet("frequency_index");
    db.pi_ds = samples.openDataSet("polarization_index");
    db.sr_ds = samples.openDataSet("scattering_real");
    db.si_ds = samples.openDataSet("scattering_imag");
    db.sqm_ds = samples.openDataSet("rcs_sqm");
    db.db_ds = samples.openDataSet("rcs_dBsm");
    db.valid_ds = samples.openDataSet("valid");
    db.status_ds = samples.openDataSet("status");
    db.prog_ds = progress.openDataSet("completed_chunks");
    db.cksum_ds = progress.openDataSet("chunk_checksum");
    db.chunk_rows = static_cast<size_t>(read_u64_attr(progress, "chunk_rows"));
}

std::vector<uint8_t> read_flags(OpenDb& db) {
    std::vector<uint8_t> flags(db.n_chunks);
    if (!flags.empty()) db.prog_ds.read(flags.data(), H5::PredType::NATIVE_UINT8);
    return flags;
}

std::vector<uint64_t> read_checksums(OpenDb& db) {
    std::vector<uint64_t> sums(db.n_chunks);
    if (!sums.empty()) db.cksum_ds.read(sums.data(), H5::PredType::NATIVE_UINT64);
    return sums;
}

std::vector<double> read_float_slice(H5::DataSet& ds, size_t begin, size_t n, bool float32) {
    const hsize_t count[1] = {n};
    const hsize_t start[1] = {begin};
    const H5::DataSpace mem(1, count);
    H5::DataSpace file_space = ds.getSpace();
    file_space.selectHyperslab(H5S_SELECT_SET, count, start);
    std::vector<double> out(n);
    if (float32) {
        std::vector<float> tmp(n);
        ds.read(tmp.data(), H5::PredType::NATIVE_FLOAT, mem, file_space);
        for (size_t i = 0; i < n; ++i) out[i] = tmp[i];
    } else {
        ds.read(out.data(), H5::PredType::NATIVE_DOUBLE, mem, file_space);
    }
    return out;
}

template <typename T>
std::vector<T> read_int_slice(H5::DataSet& ds, const H5::PredType& type, size_t begin,
                              size_t n) {
    const hsize_t count[1] = {n};
    const hsize_t start[1] = {begin};
    const H5::DataSpace mem(1, count);
    H5::DataSpace file_space = ds.getSpace();
    file_space.selectHyperslab(H5S_SELECT_SET, count, start);
    std::vector<T> out(n);
    ds.read(out.data(), type, mem, file_space);
    return out;
}

std::vector<std::string> read_string_slice(H5::DataSet& ds, size_t begin, size_t n) {
    const H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    const hsize_t count[1] = {n};
    const hsize_t start[1] = {begin};
    const H5::DataSpace mem(1, count);
    H5::DataSpace file_space = ds.getSpace();
    file_space.selectHyperslab(H5S_SELECT_SET, count, start);
    std::vector<char*> raw(n);
    ds.read(raw.data(), type, mem, file_space);
    std::vector<std::string> out;
    out.reserve(n);
    for (char* s : raw) out.emplace_back(s ? s : "");
    H5Dvlen_reclaim(type.getId(), mem.getId(), H5P_DEFAULT, raw.data());
    return out;
}

// Checksum over exactly the stored bytes of a chunk's rows.
uint64_t checksum_chunk(OpenDb& db, size_t chunk) {
    const size_t begin = chunk * db.chunk_rows;
    const size_t n = std::min(db.chunk_rows, db.n_rows - begin);
    const auto ids = read_int_slice<uint64_t>(db.id_ds, H5::PredType::NATIVE_UINT64, begin, n);
    const auto di = read_int_slice<uint32_t>(db.di_ds, H5::PredType::NATIVE_UINT32, begin, n);
    const auto fi = read_int_slice<uint32_t>(db.fi_ds, H5::PredType::NATIVE_UINT32, begin, n);
    const auto pi = read_int_slice<uint16_t>(db.pi_ds, H5::PredType::NATIVE_UINT16, begin, n);
    const auto sr = read_float_slice(db.sr_ds, begin, n, db.float32);
    const auto si = read_float_slice(db.si_ds, begin, n, db.float32);
    const auto sqm = read_float_slice(db.sqm_ds, begin, n, db.float32);
    const auto dbv = read_float_slice(db.db_ds, begin, n, db.float32);
    const auto valid = read_int_slice<uint8_t>(db.valid_ds, H5::PredType::NATIVE_UINT8, begin, n);
    const auto status = read_string_slice(db.status_ds, begin, n);
    std::vector<ChecksumRow> rows(n);
    for (size_t i = 0; i < n; ++i) {
        rows[i].sample_id = ids[i];
        rows[i].direction_index = di[i];
        rows[i].frequency_index = fi[i];
        rows[i].polarization_index = pi[i];
        rows[i].scattering_real = sr[i];
        rows[i].scattering_imag = si[i];
        rows[i].rcs_sqm = sqm[i];
        rows[i].rcs_dbsm = dbv[i];
        rows[i].valid = valid[i];
        rows[i].status = status[i];
    }
    return chunk_checksum(rows.data(), n);
}

void commit_chunk(OpenDb& db, size_t chunk) {
    db.file.flush(H5F_SCOPE_GLOBAL);
    const uint64_t sum = checksum_chunk(db, chunk);
    auto flags = read_flags(db);
    auto sums = read_checksums(db);
    sums[chunk] = sum;
    db.cksum_ds.write(sums.data(), H5::PredType::NATIVE_UINT64);
    db.file.flush(H5F_SCOPE_GLOBAL);
    flags[chunk] = 1;
    db.prog_ds.write(flags.data(), H5::PredType::NATIVE_UINT8);
    db.file.flush(H5F_SCOPE_GLOBAL);
}

void commit_units(OpenDb& db, const SamplePlan& plan, const PoResult& result,
                  const std::vector<std::pair<uint32_t, uint32_t>>& units) {
    const size_t npol = plan.polarizations.size();
    const size_t ndir = plan.directions.size();
    // Rows keyed by global sample_id (subset results arrive in unit order).
    std::map<uint64_t, const PoSampleResult*> by_id;
    for (const auto& row : result.samples) by_id[row.sample_id] = &row;
    // Group solved units into maximal contiguous row runs per chunk commit.
    std::vector<bool> touched(db.n_chunks, false);
    for (const auto& [fi, di] : units) {
        const size_t base = (static_cast<size_t>(fi) * ndir + di) * npol;
        // Gather this unit's P rows by global sample_id.
        const hsize_t count[1] = {npol};
        const hsize_t start[1] = {base};
        const H5::DataSpace mem(1, count);
        auto slab = [&](H5::DataSet& ds) {
            H5::DataSpace file_space = ds.getSpace();
            file_space.selectHyperslab(H5S_SELECT_SET, count, start);
            return file_space;
        };
        auto put = [&](H5::DataSet& ds, const void* data, const H5::PredType& t) {
            ds.write(data, t, mem, slab(ds));
        };
        std::vector<uint64_t> ids(npol);
        std::vector<uint32_t> di_v(npol), fi_v(npol);
        std::vector<uint16_t> pi_v(npol);
        std::vector<double> sr_d(npol), si_d(npol), sqm_d(npol), db_d(npol);
        std::vector<float> sr_f(npol), si_f(npol), sqm_f(npol), db_f(npol);
        std::vector<uint8_t> valid(npol, 1);
        std::vector<std::string> status(npol, "ok");
        for (size_t pi = 0; pi < npol; ++pi) {
            const uint64_t id = base + pi;
            const auto it = by_id.find(id);
            if (it == by_id.end())
                throw OutputError("solver did not return sample " + std::to_string(id));
            const PoSampleResult& row = *it->second;
            ids[pi] = row.sample_id;
            di_v[pi] = row.direction_id;
            fi_v[pi] = row.frequency_id;
            pi_v[pi] = static_cast<uint16_t>(row.pol_id);
            if (db.float32) {
                sr_f[pi] = static_cast<float>(row.scattering.real());
                si_f[pi] = static_cast<float>(row.scattering.imag());
                sqm_f[pi] = static_cast<float>(row.rcs_sqm);
                db_f[pi] = static_cast<float>(rcs_sqm_to_dbsm(row.rcs_sqm));
            } else {
                sr_d[pi] = row.scattering.real();
                si_d[pi] = row.scattering.imag();
                sqm_d[pi] = row.rcs_sqm;
                db_d[pi] = rcs_sqm_to_dbsm(row.rcs_sqm);
            }
        }
        const H5::PredType ftype = float_type(db.float32);
        put(db.id_ds, ids.data(), H5::PredType::NATIVE_UINT64);
        put(db.di_ds, di_v.data(), H5::PredType::NATIVE_UINT32);
        put(db.fi_ds, fi_v.data(), H5::PredType::NATIVE_UINT32);
        put(db.pi_ds, pi_v.data(), H5::PredType::NATIVE_UINT16);
        if (db.float32) {
            put(db.sr_ds, sr_f.data(), ftype);
            put(db.si_ds, si_f.data(), ftype);
            put(db.sqm_ds, sqm_f.data(), ftype);
            put(db.db_ds, db_f.data(), ftype);
        } else {
            put(db.sr_ds, sr_d.data(), ftype);
            put(db.si_ds, si_d.data(), ftype);
            put(db.sqm_ds, sqm_d.data(), ftype);
            put(db.db_ds, db_d.data(), ftype);
        }
        put(db.valid_ds, valid.data(), H5::PredType::NATIVE_UINT8);
        write_string_slice(db.status_ds, status, base);
        for (size_t r = base; r < base + npol; ++r) touched[r / db.chunk_rows] = true;
    }
    for (size_t c = 0; c < db.n_chunks; ++c)
        if (touched[c]) commit_chunk(db, c);
}

std::string read_version_major(const H5::H5File& file) {
    const H5::Attribute attr = file.openAttribute("output_format_version");
    const H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    std::string version;
    attr.read(type, version);
    const size_t dot = version.find('.');
    return dot == std::string::npos ? version : version.substr(0, dot);
}

std::string read_attr_str(const H5::H5File& file, const char* name) {
    const H5::Attribute attr = file.openAttribute(name);
    const H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    std::string value;
    attr.read(type, value);
    return value;
}

// Throws OutputError naming the first identity mismatch.
void check_resume_identity(H5::H5File& file, const ResolvedConfig& rc, const SamplePlan& plan,
                           const NormalizedMesh& mesh) {
    if (read_version_major(file) != "1")
        throw OutputError("resume rejected: unsupported output-format version");
    auto expect = [&](const char* attr, const std::string& want, const char* what) {
        std::string got;
        try {
            got = read_attr_str(file, attr);
        } catch (const H5::Exception&) {
            throw OutputError(std::string("resume rejected: database lacks ") + what);
        }
        if (got != want)
            throw OutputError(std::string("resume rejected: ") + what + " changed");
    };
    expect("scem_schema_version", rc.schema_version, "schema version");
    expect("config_hash", rc.hash, "resolved configuration");
    expect("geometry_hash", mesh.geometry_hash, "geometry");
    expect("normalized_mesh_hash", mesh.normalized_mesh_hash, "normalized mesh");
    expect("sample_plan_hash", plan.hash, "sample plan");
    expect("solver_type", rc.value["solver"]["type"].get<std::string>(), "solver type");
    expect("solver_precision", rc.value["solver"]["precision"].get<std::string>(),
           "solver precision");
}

} // namespace

void write_hdf5_output(const ResolvedConfig& rc, const SamplePlan& plan,
                       const NormalizedMesh& mesh, const PoResult& result,
                       const DesignerIdentity& designer, const FringeOptions& fringe) {
    H5::Exception::dontPrint();
    const fs::path out_path(rc.value["output"]["path"].get<std::string>());
    const bool float32 = rc.value["solver"]["precision"].get<std::string>() == "float32";
    const size_t chunk_rows = static_cast<size_t>(
        rc.value["run"]["checkpoint_every_n_samples"].get<int>());
    const size_t n_rows = result.samples.size();
    const size_t n_chunks = n_rows == 0 ? 0 : (n_rows + chunk_rows - 1) / chunk_rows;

    try {
        H5::H5File file(out_path.string(), H5F_ACC_TRUNC);
        OpenDb db;
        db.file = file;
        db.float32 = float32;
        db.chunk_rows = chunk_rows;
        db.n_chunks = n_chunks;
        db.n_rows = n_rows;
        db.n_pols = plan.polarizations.size();

        H5::Group directions = file.createGroup("/directions");
        H5::Group frequencies = file.createGroup("/frequencies");
        H5::Group samples = file.createGroup("/samples");
        H5::Group progress = file.createGroup("/progress");

        std::vector<double> az, el;
        for (const auto& d : plan.directions) {
            az.push_back(d.azimuth_deg);
            el.push_back(d.elevation_deg);
        }
        auto az_ds = create_vector(directions, "azimuth_deg", az.size(),
                                   H5::PredType::NATIVE_DOUBLE);
        az_ds.write(az.data(), H5::PredType::NATIVE_DOUBLE);
        write_units(az_ds, "degree");
        auto el_ds = create_vector(directions, "elevation_deg", el.size(),
                                   H5::PredType::NATIVE_DOUBLE);
        el_ds.write(el.data(), H5::PredType::NATIVE_DOUBLE);
        write_units(el_ds, "degree");
        auto hz_ds = create_vector(frequencies, "hz", plan.frequencies_hz.size(),
                                   H5::PredType::NATIVE_DOUBLE);
        hz_ds.write(plan.frequencies_hz.data(), H5::PredType::NATIVE_DOUBLE);
        write_units(hz_ds, "hertz");
        write_strings(file, "polarization", plan.polarizations);

        db.id_ds = create_vector(samples, "sample_id", n_rows, H5::PredType::NATIVE_UINT64);
        db.di_ds = create_vector(samples, "direction_index", n_rows, H5::PredType::NATIVE_UINT32);
        db.fi_ds = create_vector(samples, "frequency_index", n_rows, H5::PredType::NATIVE_UINT32);
        db.pi_ds = create_vector(samples, "polarization_index", n_rows,
                                 H5::PredType::NATIVE_UINT16);
        const H5::PredType ftype = float_type(float32);
        db.sr_ds = create_vector(samples, "scattering_real", n_rows, ftype);
        db.si_ds = create_vector(samples, "scattering_imag", n_rows, ftype);
        db.sqm_ds = create_vector(samples, "rcs_sqm", n_rows, ftype);
        db.db_ds = create_vector(samples, "rcs_dBsm", n_rows, ftype);
        db.valid_ds = create_vector(samples, "valid", n_rows, H5::PredType::NATIVE_UINT8);
        db.status_ds = create_strings(samples, "status", n_rows);
        db.prog_ds = create_vector(progress, "completed_chunks", n_chunks,
                                   H5::PredType::NATIVE_UINT8);
        db.cksum_ds = create_vector(progress, "chunk_checksum", n_chunks,
                                    H5::PredType::NATIVE_UINT64);
        write_u64_attr(progress, "chunk_rows", chunk_rows);
        write_units(db.sr_ds, "m");
        write_units(db.si_ds, "m");
        write_units(db.sqm_ds, "m^2");
        write_units(db.db_ds, "dBsm");

        write_str_attr(file, "scem_schema_version", rc.schema_version);
        write_str_attr(file, "output_format_version", "1.0");
        write_str_attr(file, "config_json", canonical_json(rc.value));
        write_str_attr(file, "config_hash", rc.hash);
        write_str_attr(file, "geometry_hash", mesh.geometry_hash);
        write_str_attr(file, "normalized_mesh_hash", mesh.normalized_mesh_hash);
        write_str_attr(file, "sample_plan_hash", plan.hash);
        write_str_attr(file, "solver_type", rc.value["solver"]["type"].get<std::string>());
        write_str_attr(file, "edge_correction",
                       rc.value["solver"]["po_options"].value("edge_correction", "none"));
        write_u64_attr(file, "fringe_edges",
                       fringe.enabled && fringe.edges != nullptr ? fringe.edges->edges.size()
                                                                : 0);
        write_str_attr(file, "design_id", designer.design_id);
        write_str_attr(file, "design_revision", designer.revision);
        write_str_attr(file, "export_id", designer.export_id);
        write_str_attr(file, "export_manifest_hash", designer.manifest_hash);
        write_str_attr(file, "tool_version", SCEM_VERSION);
        write_str_attr(file, "git_revision", SCEM_GIT_REVISION);
        write_str_attr(file, "build_type", SCEM_BUILD_TYPE);
        write_str_attr(file, "compiler",
                       std::string(SCEM_COMPILER_ID) + " " + SCEM_COMPILER_VERSION);
        write_str_attr(file, "solver_precision",
                       rc.value["solver"]["precision"].get<std::string>());
        write_str_attr(file, "solver_backend",
                       rc.value["execution"].value("accelerator", "cpu"));
        write_str_attr(file, "cuda_device", cuda::active_device_name(rc.value));
        write_str_attr(file, "completed_utc", utc_now());
        std::string warnings;
        for (const auto& w : plan.warnings) warnings += w + "\n";
        for (const auto& w : result.warnings) warnings += w + "\n";
        write_str_attr(file, "warnings", warnings);
        file.flush(H5F_SCOPE_GLOBAL);

        // Commit every unit, then every chunk.
        std::vector<std::pair<uint32_t, uint32_t>> units;
        for (uint32_t fi = 0; fi < plan.frequencies_hz.size(); ++fi)
            for (uint32_t di = 0; di < plan.directions.size(); ++di)
                units.emplace_back(fi, di);
        commit_units(db, plan, result, units);
        file.flush(H5F_SCOPE_GLOBAL);
        file.close();
    } catch (const H5::Exception& e) {
        std::error_code ec;
        fs::remove(out_path, ec);
        throw OutputError(std::string("HDF5 write failed: ") + e.getDetailMsg());
    }
}

namespace {

void open_resume_db(OpenDb& db, const std::string& path, const SamplePlan& plan) {
    db.file = H5::H5File(path, H5F_ACC_RDWR);
    db.float32 = false; // reset below from stored datatype
    open_datasets(db);
    db.n_rows = plan.sample_count();
    db.n_pols = plan.polarizations.size();
    // Row count must match the plan; the config hash already pins content.
    H5::DataSpace space = db.id_ds.getSpace();
    hsize_t n = 0;
    space.getSimpleExtentDims(&n);
    if (n != db.n_rows)
        throw OutputError("resume rejected: database row count does not match plan");
    // Direction/frequency/polarization table sizes pin the axes.
    auto extent = [&](H5::Group& group, const char* name) {
        H5::DataSet ds = group.openDataSet(name);
        H5::DataSpace sp = ds.getSpace();
        hsize_t m = 0;
        sp.getSimpleExtentDims(&m);
        return static_cast<size_t>(m);
    };
    H5::Group directions = db.file.openGroup("/directions");
    H5::Group frequencies = db.file.openGroup("/frequencies");
    if (extent(directions, "azimuth_deg") != plan.directions.size() ||
        extent(frequencies, "hz") != plan.frequencies_hz.size())
        throw OutputError("resume rejected: database layout does not match plan");
    // Float width decides the checksum read-back path.
    db.float32 = db.sr_ds.getFloatType().getSize() == 4;
    H5::DataSpace prog_space = db.prog_ds.getSpace();
    hsize_t nc = 0;
    prog_space.getSimpleExtentDims(&nc);
    db.n_chunks = static_cast<size_t>(nc);
    if (read_checksums(db).size() != db.n_chunks || (db.n_chunks == 0 && db.n_rows > 0))
        throw OutputError("resume rejected: progress bookkeeping is corrupt");
}

} // namespace

size_t resume_hdf5_output(const ResolvedConfig& rc, const SamplePlan& plan,
                          const NormalizedMesh& mesh, const std::string& path,
                          const FringeOptions& fringe) {
    H5::Exception::dontPrint();
    try {
        OpenDb db;
        open_resume_db(db, path, plan);
        check_resume_identity(db.file, rc, plan, mesh);
        auto flags = read_flags(db);
        const auto sums = read_checksums(db);
        const size_t npol = plan.polarizations.size();
        const size_t ndir = plan.directions.size();
        const size_t nunits = plan.frequencies_hz.size() * ndir;
        std::vector<bool> unit_missing(nunits, false);
        for (size_t c = 0; c < db.n_chunks; ++c) {
            bool missing = flags[c] == 0;
            if (!missing && checksum_chunk(db, c) != sums[c]) missing = true;
            if (!missing) continue;
            const size_t begin = c * db.chunk_rows;
            const size_t end = std::min(begin + db.chunk_rows, db.n_rows);
            for (size_t r = begin; r < end; ++r) unit_missing[r / npol] = true;
        }
        std::vector<std::pair<uint32_t, uint32_t>> units;
        for (size_t u = 0; u < nunits; ++u)
            if (unit_missing[u])
                units.emplace_back(static_cast<uint32_t>(u / ndir),
                                   static_cast<uint32_t>(u % ndir));
        if (units.empty()) {
            db.file.close();
            return 0;
        }
        PoResult subset = solve_po_units(mesh, plan, rc.value, units, fringe);
        commit_units(db, plan, subset, units);
        db.file.flush(H5F_SCOPE_GLOBAL);
        db.file.close();
        return subset.samples.size();
    } catch (const H5::Exception& e) {
        throw OutputError(std::string("HDF5 resume failed: ") + e.getDetailMsg());
    }
}

bool check_existing_hdf5(const ResolvedConfig& rc, const SamplePlan& plan,
                         const NormalizedMesh& mesh, const std::string& path) {
    H5::Exception::dontPrint();
    try {
        OpenDb db;
        open_resume_db(db, path, plan);
        try {
            check_resume_identity(db.file, rc, plan, mesh);
        } catch (const OutputError& e) {
            throw OutputError(std::string("existing output was not produced by this "
                                          "configuration (") +
                              e.what() + "); remove it or resume it explicitly");
        }
        const auto flags = read_flags(db);
        const auto sums = read_checksums(db);
        for (size_t c = 0; c < db.n_chunks; ++c) {
            if (flags[c] == 0)
                throw OutputError("existing output is incomplete; rerun with "
                                  "run.resume_from_checkpoint pointing at it");
            if (checksum_chunk(db, c) != sums[c])
                throw OutputError("existing output is corrupt; rerun with "
                                  "run.resume_from_checkpoint pointing at it");
        }
        db.file.close();
        return true;
    } catch (const H5::Exception& e) {
        throw OutputError(std::string("cannot inspect existing output: ") + e.getDetailMsg());
    }
}

} // namespace strikecem

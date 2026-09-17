// Flat-row HDF5 writer (OUTPUT_FORMAT.md).
#include "strikecem/io/Hdf5Writer.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>

#include <H5Cpp.h>

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

template <typename T> H5::PredType native_type();
template <> H5::PredType native_type<double>() { return H5::PredType::NATIVE_DOUBLE; }
template <> H5::PredType native_type<uint64_t>() { return H5::PredType::NATIVE_UINT64; }
template <> H5::PredType native_type<uint32_t>() { return H5::PredType::NATIVE_UINT32; }
template <> H5::PredType native_type<uint16_t>() { return H5::PredType::NATIVE_UINT16; }
template <> H5::PredType native_type<uint8_t>() { return H5::PredType::NATIVE_UINT8; }

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

} // namespace

void write_hdf5_output(const ResolvedConfig& rc, const SamplePlan& plan,
                       const NormalizedMesh& mesh, const PoResult& result) {
    H5::Exception::dontPrint();
    const fs::path out_path(rc.value["output"]["path"].get<std::string>());
    const bool float32 = rc.value["solver"]["precision"].get<std::string>() == "float32";
    const size_t chunk_rows = static_cast<size_t>(
        rc.value["run"]["checkpoint_every_n_samples"].get<int>());
    const size_t n_chunks =
        result.samples.empty() ? 0 : (result.samples.size() + chunk_rows - 1) / chunk_rows;

    try {
        H5::H5File file(out_path.string(), H5F_ACC_TRUNC);

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

        const size_t k = result.samples.size();
        std::vector<uint64_t> ids(k);
        std::vector<uint32_t> dir_idx(k), freq_idx(k);
        std::vector<uint16_t> pol_idx(k);
        std::vector<double> sr_d(k), si_d(k), sqm_d(k), db_d(k);
        std::vector<float> sr_f(k), si_f(k), sqm_f(k), db_f(k);
        std::vector<uint8_t> valid(k, 1);
        std::vector<std::string> status(k, "ok");
        for (size_t i = 0; i < k; ++i) {
            const auto& row = result.samples[i];
            ids[i] = row.sample_id;
            dir_idx[i] = row.direction_id;
            freq_idx[i] = row.frequency_id;
            pol_idx[i] = static_cast<uint16_t>(row.pol_id);
            if (float32) {
                sr_f[i] = static_cast<float>(row.scattering.real());
                si_f[i] = static_cast<float>(row.scattering.imag());
                sqm_f[i] = static_cast<float>(row.rcs_sqm);
                db_f[i] = static_cast<float>(rcs_sqm_to_dbsm(row.rcs_sqm));
            } else {
                sr_d[i] = row.scattering.real();
                si_d[i] = row.scattering.imag();
                sqm_d[i] = row.rcs_sqm;
                db_d[i] = rcs_sqm_to_dbsm(row.rcs_sqm);
            }
        }
        auto id_ds = create_vector(samples, "sample_id", k, H5::PredType::NATIVE_UINT64);
        auto di_ds = create_vector(samples, "direction_index", k, H5::PredType::NATIVE_UINT32);
        auto fi_ds = create_vector(samples, "frequency_index", k, H5::PredType::NATIVE_UINT32);
        auto pi_ds = create_vector(samples, "polarization_index", k, H5::PredType::NATIVE_UINT16);
        const H5::PredType ftype = float32 ? H5::PredType::NATIVE_FLOAT : H5::PredType::NATIVE_DOUBLE;
        auto sr_ds = create_vector(samples, "scattering_real", k, ftype);
        auto si_ds = create_vector(samples, "scattering_imag", k, ftype);
        auto sqm_ds = create_vector(samples, "rcs_sqm", k, ftype);
        auto db_ds = create_vector(samples, "rcs_dBsm", k, ftype);
        auto valid_ds = create_vector(samples, "valid", k, H5::PredType::NATIVE_UINT8);
        auto status_ds = create_strings(samples, "status", k);

        auto prog_ds = create_vector(progress, "completed_chunks", n_chunks,
                                     H5::PredType::NATIVE_UINT8);

        // Incremental chunk commit: data, flush, then progress flag. A chunk
        // counts only after its flag lands.
        std::vector<uint8_t> flags(n_chunks, 0);
        for (size_t c = 0; c < n_chunks; ++c) {
            const size_t begin = c * chunk_rows;
            const hsize_t start[1] = {begin};
            const hsize_t count[1] = {std::min(chunk_rows, k - begin)};
            const H5::DataSpace mem(1, count);
            auto slab = [&](H5::DataSet& ds) {
                H5::DataSpace file_space = ds.getSpace();
                file_space.selectHyperslab(H5S_SELECT_SET, count, start);
                return file_space;
            };
            auto put = [&](H5::DataSet& ds, const void* data, const H5::PredType& t) {
                ds.write(data, t, mem, slab(ds));
            };
            put(id_ds, ids.data() + begin, H5::PredType::NATIVE_UINT64);
            put(di_ds, dir_idx.data() + begin, H5::PredType::NATIVE_UINT32);
            put(fi_ds, freq_idx.data() + begin, H5::PredType::NATIVE_UINT32);
            put(pi_ds, pol_idx.data() + begin, H5::PredType::NATIVE_UINT16);
            if (float32) {
                put(sr_ds, sr_f.data() + begin, ftype);
                put(si_ds, si_f.data() + begin, ftype);
                put(sqm_ds, sqm_f.data() + begin, ftype);
                put(db_ds, db_f.data() + begin, ftype);
            } else {
                put(sr_ds, sr_d.data() + begin, ftype);
                put(si_ds, si_d.data() + begin, ftype);
                put(sqm_ds, sqm_d.data() + begin, ftype);
                put(db_ds, db_d.data() + begin, ftype);
            }
            put(valid_ds, valid.data() + begin, H5::PredType::NATIVE_UINT8);
            write_string_slice(status_ds,
                               std::vector<std::string>(status.begin() + begin,
                                                        status.begin() + begin + count[0]),
                               begin);
            file.flush(H5F_SCOPE_GLOBAL);
            flags[c] = 1;
            prog_ds.write(flags.data(), H5::PredType::NATIVE_UINT8);
            file.flush(H5F_SCOPE_GLOBAL);
        }
        write_units(sr_ds, "m"); // normalized amplitude F/E0 carries metres
        write_units(si_ds, "m");
        write_units(sqm_ds, "m^2");
        write_units(db_ds, "dBsm");

        write_str_attr(file, "scem_schema_version", rc.schema_version);
        write_str_attr(file, "output_format_version", "1.0");
        write_str_attr(file, "config_json", canonical_json(rc.value));
        write_str_attr(file, "config_hash", rc.hash);
        write_str_attr(file, "geometry_hash", mesh.geometry_hash);
        write_str_attr(file, "normalized_mesh_hash", mesh.normalized_mesh_hash);
        if (rc.value.contains("integration") && rc.value["integration"].contains("designer")) {
            const auto& d = rc.value["integration"]["designer"];
            write_str_attr(file, "design_id", d["design_id"].get<std::string>());
            write_str_attr(file, "design_revision", d["revision"].get<std::string>());
            write_str_attr(file, "export_id",
                           d.value("export_id", std::string()));
        } else {
            write_str_attr(file, "design_id", "");
            write_str_attr(file, "design_revision", "");
            write_str_attr(file, "export_id", "");
        }
        write_str_attr(file, "tool_version", SCEM_VERSION);
        write_str_attr(file, "git_revision", SCEM_GIT_REVISION);
        write_str_attr(file, "build_type", SCEM_BUILD_TYPE);
        write_str_attr(file, "compiler",
                       std::string(SCEM_COMPILER_ID) + " " + SCEM_COMPILER_VERSION);
        write_str_attr(file, "solver_precision",
                       rc.value["solver"]["precision"].get<std::string>());
        write_str_attr(file, "completed_utc", utc_now());
        std::string warnings;
        for (const auto& w : plan.warnings) warnings += w + "\n";
        for (const auto& w : result.warnings) warnings += w + "\n";
        write_str_attr(file, "warnings", warnings);
        file.flush(H5F_SCOPE_GLOBAL);
        file.close();
    } catch (const H5::Exception& e) {
        std::error_code ec;
        fs::remove(out_path, ec);
        throw OutputError(std::string("HDF5 write failed: ") + e.getDetailMsg());
    }
}

} // namespace strikecem

// Reference HDF5 reader (SPEC section 10).
#include "strikecem/io/Hdf5Reader.hpp"

#include <H5Cpp.h>

namespace strikecem {
namespace {

std::string read_str_attr(const H5::H5Object& obj, const char* name) {
    try {
        const H5::Attribute attr = obj.openAttribute(name);
        const H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
        std::string value;
        attr.read(type, value);
        return value;
    } catch (const H5::Exception& e) {
        throw ReaderError(std::string("missing provenance attribute '") + name +
                          "': " + e.getDetailMsg());
    }
}

template <typename T> std::vector<T> read_vector(H5::Group& group, const char* name) {
    H5::DataSet ds;
    try {
        ds = group.openDataSet(name);
    } catch (const H5::Exception& e) {
        throw ReaderError(std::string("missing dataset '") + name + "': " + e.getDetailMsg());
    }
    const H5::DataSpace space = ds.getSpace();
    if (space.getSimpleExtentNdims() != 1)
        throw ReaderError(std::string("dataset '") + name + "' is not rank 1");
    hsize_t n = 0;
    space.getSimpleExtentDims(&n);
    std::vector<T> values(n);
    try {
        if constexpr (std::is_same_v<T, double>)
            ds.read(values.data(), H5::PredType::NATIVE_DOUBLE);
        else if constexpr (std::is_same_v<T, uint64_t>)
            ds.read(values.data(), H5::PredType::NATIVE_UINT64);
        else if constexpr (std::is_same_v<T, uint32_t>)
            ds.read(values.data(), H5::PredType::NATIVE_UINT32);
        else if constexpr (std::is_same_v<T, uint16_t>)
            ds.read(values.data(), H5::PredType::NATIVE_UINT16);
        else if constexpr (std::is_same_v<T, uint8_t>)
            ds.read(values.data(), H5::PredType::NATIVE_UINT8);
        else if constexpr (std::is_same_v<T, float>)
            ds.read(values.data(), H5::PredType::NATIVE_FLOAT);
    } catch (const H5::Exception& e) {
        throw ReaderError(std::string("cannot read dataset '") + name +
                          "': " + e.getDetailMsg());
    }
    return values;
}

// Reads a float-or-double dataset into doubles.
std::vector<double> read_float_dataset(H5::DataSet& ds) {
    const H5::DataSpace space = ds.getSpace();
    hsize_t n = 0;
    space.getSimpleExtentDims(&n);
    const H5::FloatType ftype = ds.getFloatType();
    const size_t size = ftype.getSize();
    std::vector<double> out(n);
    if (size == 8) {
        ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
    } else if (size == 4) {
        std::vector<float> tmp(n);
        ds.read(tmp.data(), H5::PredType::NATIVE_FLOAT);
        for (size_t i = 0; i < n; ++i) out[i] = tmp[i];
    } else {
        throw ReaderError("unexpected float width in dataset");
    }
    return out;
}

std::vector<std::string> read_strings(H5::Group& group, const char* name) {
    H5::DataSet ds;
    try {
        ds = group.openDataSet(name);
    } catch (const H5::Exception& e) {
        throw ReaderError(std::string("missing dataset '") + name + "': " + e.getDetailMsg());
    }
    const H5::DataSpace space = ds.getSpace();
    hsize_t n = 0;
    space.getSimpleExtentDims(&n);
    const H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    std::vector<char*> raw(n);
    try {
        ds.read(raw.data(), type);
    } catch (const H5::Exception& e) {
        throw ReaderError(std::string("cannot read dataset '") + name +
                          "': " + e.getDetailMsg());
    }
    std::vector<std::string> out;
    out.reserve(n);
    for (char* s : raw) out.emplace_back(s ? s : "");
    H5Dvlen_reclaim(type.getId(), space.getId(), H5P_DEFAULT, raw.data());
    return out;
}

} // namespace

Hdf5Database read_hdf5_database(const std::string& path) {
    H5::Exception::dontPrint();
    Hdf5Database db;
    try {
        H5::H5File file(path, H5F_ACC_RDONLY);
        H5::Group directions = file.openGroup("/directions");
        H5::Group frequencies = file.openGroup("/frequencies");
        H5::Group samples = file.openGroup("/samples");
        H5::Group progress = file.openGroup("/progress");
        db.azimuth_deg = read_vector<double>(directions, "azimuth_deg");
        db.elevation_deg = read_vector<double>(directions, "elevation_deg");
        db.frequencies_hz = read_vector<double>(frequencies, "hz");
        db.polarizations = read_strings(file, "polarization");
        const auto ids = read_vector<uint64_t>(samples, "sample_id");
        const auto dir_idx = read_vector<uint32_t>(samples, "direction_index");
        const auto freq_idx = read_vector<uint32_t>(samples, "frequency_index");
        const auto pol_idx = read_vector<uint16_t>(samples, "polarization_index");
        H5::DataSet sr_ds = samples.openDataSet("scattering_real");
        H5::DataSet si_ds = samples.openDataSet("scattering_imag");
        H5::DataSet sqm_ds = samples.openDataSet("rcs_sqm");
        H5::DataSet db_ds = samples.openDataSet("rcs_dBsm");
        const auto sr = read_float_dataset(sr_ds);
        const auto si = read_float_dataset(si_ds);
        const auto sqm = read_float_dataset(sqm_ds);
        const auto dbsm = read_float_dataset(db_ds);
        const auto valid = read_vector<uint8_t>(samples, "valid");
        const auto status = read_strings(samples, "status");
        db.completed_chunks = read_vector<uint8_t>(progress, "completed_chunks");
        const size_t k = ids.size();
        db.rows.reserve(k);
        for (size_t i = 0; i < k; ++i) {
            Hdf5SampleRow row;
            row.sample_id = ids[i];
            row.direction_index = dir_idx[i];
            row.frequency_index = freq_idx[i];
            row.polarization_index = pol_idx[i];
            row.scattering_real = sr[i];
            row.scattering_imag = si[i];
            row.rcs_sqm = sqm[i];
            row.rcs_dbsm = dbsm[i];
            row.valid = valid[i] != 0;
            row.status = status[i];
            db.rows.push_back(std::move(row));
        }
        db.output_format_version = read_str_attr(file, "output_format_version");
        db.scem_schema_version = read_str_attr(file, "scem_schema_version");
        db.config_hash = read_str_attr(file, "config_hash");
        db.solver_precision = read_str_attr(file, "solver_precision");
        file.close();
    } catch (const ReaderError&) {
        throw;
    } catch (const H5::Exception& e) {
        throw ReaderError(std::string("cannot open database: ") + e.getDetailMsg());
    }
    return db;
}

void validate_hdf5_complete(const Hdf5Database& db) {
    const std::string& version = db.output_format_version;
    const size_t dot = version.find('.');
    const std::string major = dot == std::string::npos ? version : version.substr(0, dot);
    if (major != "1") throw ReaderError("unsupported output-format version: " + version);
    const uint64_t expected =
        static_cast<uint64_t>(db.azimuth_deg.size()) * db.frequencies_hz.size() *
        db.polarizations.size();
    if (db.rows.size() != expected)
        throw ReaderError("incomplete database: rows != directions*frequencies*polarizations");
    for (size_t i = 0; i < db.rows.size(); ++i) {
        if (db.rows[i].sample_id != i)
            throw ReaderError("non-sequential sample_id: coverage gap or reorder");
        if (!db.rows[i].valid) throw ReaderError("invalid sample present");
        if (db.rows[i].status == "error") throw ReaderError("error sample present");
    }
    for (uint8_t flag : db.completed_chunks)
        if (flag == 0) throw ReaderError("incomplete progress: uncommitted chunk");
    if (db.completed_chunks.empty() && !db.rows.empty())
        throw ReaderError("incomplete progress: no committed chunks");
}

} // namespace strikecem

// Reference HDF5 reader for the flat sample-table format.
#include "strikecem/io/Hdf5Reader.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <utility>

#include <H5Cpp.h>

#include "strikecem/io/Checksum.hpp"

namespace strikecem {
namespace {

size_t checked_count(hsize_t n, size_t max_size, const char* name) {
    if (n > std::numeric_limits<size_t>::max() || n > max_size)
        throw ReaderError(std::string("dataset '") + name + "' is too large");
    return static_cast<size_t>(n);
}

size_t dataset_length(H5::DataSet& ds, const char* name) {
    const H5::DataSpace space = ds.getSpace();
    if (space.getSimpleExtentNdims() != 1)
        throw ReaderError(std::string("dataset '") + name + "' is not rank 1");
    hsize_t n = 0;
    space.getSimpleExtentDims(&n);
    if (n > std::numeric_limits<size_t>::max())
        throw ReaderError(std::string("dataset '") + name + "' is too large");
    return static_cast<size_t>(n);
}

size_t checked_product(size_t a, size_t b, const char* what) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
        throw ReaderError(std::string("database ") + what + " count overflows");
    return a * b;
}

size_t expected_chunks(size_t rows, uint64_t chunk_rows) {
    if (chunk_rows == 0 || chunk_rows > std::numeric_limits<size_t>::max())
        throw ReaderError("invalid progress chunk_rows");
    const size_t chunk = static_cast<size_t>(chunk_rows);
    return rows / chunk + (rows % chunk != 0);
}

void validate_polarization_axis(const std::vector<std::string>& pols) {
    for (size_t i = 0; i < pols.size(); ++i) {
        const std::string& pol = pols[i];
        if (pol != "HH" && pol != "VV" && pol != "HV" && pol != "VH" && pol != "RHCP" &&
            pol != "LHCP")
            throw ReaderError("invalid polarization axis value");
        if (std::find(pols.begin(), pols.begin() + i, pol) != pols.begin() + i)
            throw ReaderError("duplicate polarization axis value");
    }
}

template <typename T> void check_vector_type(H5::DataSet& ds, const char* name) {
    const H5::DataType type = ds.getDataType();
    const H5T_class_t expected_class = std::is_floating_point_v<T> ? H5T_FLOAT : H5T_INTEGER;
    if (type.getClass() != expected_class || type.getSize() != sizeof(T))
        throw ReaderError(std::string("dataset '") + name + "' has an unexpected type");
    if constexpr (std::is_integral_v<T>) {
        const H5T_sign_t expected_sign = std::is_signed_v<T> ? H5T_SGN_2 : H5T_SGN_NONE;
        if (H5Tget_sign(type.getId()) != expected_sign)
            throw ReaderError(std::string("dataset '") + name + "' has an unexpected sign");
    }
}

void check_dataset_length(H5::DataSet& ds, const char* name, size_t expected) {
    if (dataset_length(ds, name) != expected)
        throw ReaderError(std::string("dataset '") + name + "' has the wrong length");
}

void check_integer_dataset(H5::DataSet& ds, const char* name, size_t expected_length,
                           size_t width) {
    check_dataset_length(ds, name, expected_length);
    const H5::DataType type = ds.getDataType();
    if (type.getClass() != H5T_INTEGER || type.getSize() != width ||
        H5Tget_sign(type.getId()) != H5T_SGN_NONE)
        throw ReaderError(std::string("dataset '") + name + "' has an unexpected type");
}

size_t check_float_dataset(H5::DataSet& ds, const char* name, size_t expected_length) {
    check_dataset_length(ds, name, expected_length);
    const H5::DataType type = ds.getDataType();
    if (type.getClass() != H5T_FLOAT ||
        (type.getSize() != sizeof(float) && type.getSize() != sizeof(double)))
        throw ReaderError(std::string("dataset '") + name + "' has an unexpected float type");
    return type.getSize();
}

void check_string_dataset(H5::DataSet& ds, const char* name, size_t expected_length) {
    check_dataset_length(ds, name, expected_length);
    if (ds.getDataType().getClass() != H5T_STRING)
        throw ReaderError(std::string("dataset '") + name + "' is not a string dataset");
}

std::string read_str_attr(const H5::H5Object& obj, const char* name) {
    try {
        const H5::Attribute attr = obj.openAttribute(name);
        if (attr.getSpace().getSimpleExtentType() != H5S_SCALAR ||
            attr.getDataType().getClass() != H5T_STRING)
            throw ReaderError(std::string("invalid provenance attribute '") + name + "'");
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
    const size_t n = dataset_length(ds, name);
    check_vector_type<T>(ds, name);
    std::vector<T> values;
    const size_t count = checked_count(n, values.max_size(), name);
    values.resize(count);
    try {
        if (!values.empty()) {
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
        }
    } catch (const H5::Exception& e) {
        throw ReaderError(std::string("cannot read dataset '") + name +
                          "': " + e.getDetailMsg());
    }
    return values;
}

// Reads a float-or-double dataset into doubles.
std::vector<double> read_float_dataset(H5::DataSet& ds, const char* name) {
    const H5::DataSpace space = ds.getSpace();
    if (space.getSimpleExtentNdims() != 1)
        throw ReaderError(std::string("dataset '") + name + "' is not rank 1");
    hsize_t extent = 0;
    space.getSimpleExtentDims(&extent);
    std::vector<double> out;
    const size_t n = checked_count(extent, out.max_size(), name);
    if (ds.getDataType().getClass() != H5T_FLOAT)
        throw ReaderError(std::string("dataset '") + name + "' is not floating-point");
    const H5::FloatType ftype = ds.getFloatType();
    const size_t size = ftype.getSize();
    if (size != sizeof(double) && size != sizeof(float))
        throw ReaderError(std::string("dataset '") + name + "' has an unexpected float width");
    out.resize(n);
    if (size == 8) {
        if (n) ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
    } else if (size == 4) {
        std::vector<float> tmp;
        tmp.resize(checked_count(extent, tmp.max_size(), name));
        if (n) ds.read(tmp.data(), H5::PredType::NATIVE_FLOAT);
        for (size_t i = 0; i < n; ++i) out[i] = tmp[i];
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
    const size_t n = dataset_length(ds, name);
    if (ds.getDataType().getClass() != H5T_STRING)
        throw ReaderError(std::string("dataset '") + name + "' is not a string dataset");
    const H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    std::vector<char*> raw;
    raw.resize(checked_count(n, raw.max_size(), name));
    struct Reclaim {
        hid_t type;
        hid_t space;
        std::vector<char*>& values;
        ~Reclaim() { H5Dvlen_reclaim(type, space, H5P_DEFAULT, values.data()); }
    } reclaim{type.getId(), space.getId(), raw};
    try {
        if (n) ds.read(raw.data(), type);
    } catch (const H5::Exception& e) {
        throw ReaderError(std::string("cannot read dataset '") + name +
                          "': " + e.getDetailMsg());
    }
    std::vector<std::string> out;
    out.reserve(n);
    for (char* s : raw) out.emplace_back(s ? s : "");
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
        validate_polarization_axis(db.polarizations);
        if (db.azimuth_deg.size() != db.elevation_deg.size())
            throw ReaderError("direction azimuth/elevation lengths do not match");
        const size_t expected_samples = checked_product(
            checked_product(db.azimuth_deg.size(), db.frequencies_hz.size(), "sample"),
            db.polarizations.size(), "sample");
        auto check_sample_integer = [&](const char* name, size_t width) {
            H5::DataSet ds = samples.openDataSet(name);
            check_integer_dataset(ds, name, expected_samples, width);
        };
        auto check_sample_float = [&](const char* name) {
            H5::DataSet ds = samples.openDataSet(name);
            return check_float_dataset(ds, name, expected_samples);
        };
        check_sample_integer("sample_id", sizeof(uint64_t));
        check_sample_integer("direction_index", sizeof(uint32_t));
        check_sample_integer("frequency_index", sizeof(uint32_t));
        check_sample_integer("polarization_index", sizeof(uint16_t));
        check_sample_integer("valid", sizeof(uint8_t));
        const size_t float_width = check_sample_float("scattering_real");
        for (const char* name : {"scattering_imag", "rcs_sqm", "rcs_dBsm"})
            if (check_sample_float(name) != float_width)
                throw ReaderError("sample floating-point datasets use mixed precisions");
        {
            H5::DataSet ds = samples.openDataSet("status");
            check_string_dataset(ds, "status", expected_samples);
        }
        const auto ids = read_vector<uint64_t>(samples, "sample_id");
        const auto dir_idx = read_vector<uint32_t>(samples, "direction_index");
        const auto freq_idx = read_vector<uint32_t>(samples, "frequency_index");
        const auto pol_idx = read_vector<uint16_t>(samples, "polarization_index");
        H5::DataSet sr_ds = samples.openDataSet("scattering_real");
        H5::DataSet si_ds = samples.openDataSet("scattering_imag");
        H5::DataSet sqm_ds = samples.openDataSet("rcs_sqm");
        H5::DataSet db_ds = samples.openDataSet("rcs_dBsm");
        const auto sr = read_float_dataset(sr_ds, "scattering_real");
        const auto si = read_float_dataset(si_ds, "scattering_imag");
        const auto sqm = read_float_dataset(sqm_ds, "rcs_sqm");
        const auto dbsm = read_float_dataset(db_ds, "rcs_dBsm");
        const auto valid = read_vector<uint8_t>(samples, "valid");
        const auto status = read_strings(samples, "status");
        try {
            const H5::Attribute chunk_rows = progress.openAttribute("chunk_rows");
            if (chunk_rows.getSpace().getSimpleExtentType() != H5S_SCALAR)
                throw ReaderError("invalid progress chunk_rows attribute shape");
            const H5::DataType chunk_type = chunk_rows.getDataType();
            if (chunk_type.getClass() != H5T_INTEGER ||
                chunk_type.getSize() != sizeof(uint64_t) ||
                H5Tget_sign(chunk_type.getId()) != H5T_SGN_NONE)
                throw ReaderError("invalid progress chunk_rows attribute type");
            chunk_rows.read(H5::PredType::NATIVE_UINT64, &db.chunk_rows);
        } catch (const ReaderError&) {
            throw;
        } catch (const H5::Exception& e) {
            throw ReaderError(std::string("missing chunk_rows bookkeeping: ") + e.getDetailMsg());
        }
        const size_t expected_progress = expected_chunks(expected_samples, db.chunk_rows);
        {
            H5::DataSet ds = progress.openDataSet("completed_chunks");
            check_integer_dataset(ds, "completed_chunks", expected_progress, sizeof(uint8_t));
        }
        {
            H5::DataSet ds = progress.openDataSet("chunk_checksum");
            check_integer_dataset(ds, "chunk_checksum", expected_progress, sizeof(uint64_t));
        }
        db.completed_chunks = read_vector<uint8_t>(progress, "completed_chunks");
        db.chunk_checksums = read_vector<uint64_t>(progress, "chunk_checksum");
        for (uint8_t flag : db.completed_chunks)
            if (flag > 1) throw ReaderError("invalid progress completion flag");
        const size_t k = ids.size();
        for (const auto& [name, length] : std::initializer_list<std::pair<const char*, size_t>>{
                 {"direction_index", dir_idx.size()}, {"frequency_index", freq_idx.size()},
                 {"polarization_index", pol_idx.size()}, {"scattering_real", sr.size()},
                 {"scattering_imag", si.size()}, {"rcs_sqm", sqm.size()},
                 {"rcs_dBsm", dbsm.size()}, {"valid", valid.size()}, {"status", status.size()}}) {
            if (length != k)
                throw ReaderError(std::string("sample dataset '") + name +
                                  "' length does not match sample_id");
        }
        if (db.azimuth_deg.size() != db.elevation_deg.size())
            throw ReaderError("direction azimuth/elevation lengths do not match");
        for (double az : db.azimuth_deg)
            if (!std::isfinite(az) || az < 0.0 || az >= 360.0)
                throw ReaderError("invalid azimuth axis value");
        for (double el : db.elevation_deg)
            if (!std::isfinite(el) || el < -90.0 || el > 90.0)
                throw ReaderError("invalid elevation axis value");
        for (double f : db.frequencies_hz)
            if (!std::isfinite(f) || !(f > 0.0))
                throw ReaderError("invalid frequency axis value");
        if (db.polarizations.empty()) throw ReaderError("empty polarization axis");
        const size_t expected = checked_product(
            checked_product(db.azimuth_deg.size(), db.frequencies_hz.size(), "sample"),
            db.polarizations.size(), "sample");
        if (k != expected) throw ReaderError("sample dataset lengths do not match axes");
        if (db.chunk_rows == 0 || db.chunk_checksums.size() != db.completed_chunks.size() ||
            db.completed_chunks.size() != expected_chunks(k, db.chunk_rows))
            throw ReaderError("corrupt progress bookkeeping");
        db.rows.reserve(k);
        for (size_t i = 0; i < k; ++i) {
            if (dir_idx[i] >= db.azimuth_deg.size() || freq_idx[i] >= db.frequencies_hz.size() ||
                pol_idx[i] >= db.polarizations.size())
                throw ReaderError("sample index outside axis bounds");
            if (valid[i] > 1) throw ReaderError("invalid sample valid flag");
            if (!std::isfinite(sr[i]) || !std::isfinite(si[i]) || !std::isfinite(sqm[i]) ||
                sqm[i] < 0.0 || std::isnan(dbsm[i]) || dbsm[i] ==
                    std::numeric_limits<double>::infinity())
                throw ReaderError("non-finite or invalid sample value");
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
        if ((db.solver_precision == "float32" && float_width != sizeof(float)) ||
            (db.solver_precision == "float64" && float_width != sizeof(double)) ||
            (db.solver_precision != "float32" && db.solver_precision != "float64"))
            throw ReaderError("sample float type does not match solver_precision");
        file.close();
    } catch (const ReaderError&) {
        throw;
    } catch (const H5::Exception& e) {
        throw ReaderError(std::string("cannot open database: ") + e.getDetailMsg());
    } catch (const std::exception& e) {
        throw ReaderError(std::string("cannot read database: ") + e.what());
    }
    return db;
}

void validate_hdf5_complete(const Hdf5Database& db) {
    const std::string& version = db.output_format_version;
    const size_t dot = version.find('.');
    const std::string major = dot == std::string::npos ? version : version.substr(0, dot);
    if (major != "1") throw ReaderError("unsupported output-format version: " + version);
    if (db.azimuth_deg.empty() || db.elevation_deg.size() != db.azimuth_deg.size() ||
        db.frequencies_hz.empty() || db.polarizations.empty())
        throw ReaderError("empty or inconsistent database axes");
    validate_polarization_axis(db.polarizations);
    for (double az : db.azimuth_deg)
        if (!std::isfinite(az) || az < 0.0 || az >= 360.0)
            throw ReaderError("invalid azimuth axis value");
    for (double el : db.elevation_deg)
        if (!std::isfinite(el) || el < -90.0 || el > 90.0)
            throw ReaderError("invalid elevation axis value");
    for (double f : db.frequencies_hz)
        if (!std::isfinite(f) || !(f > 0.0)) throw ReaderError("invalid frequency axis value");
    const size_t expected = checked_product(
        checked_product(db.azimuth_deg.size(), db.frequencies_hz.size(), "sample"),
        db.polarizations.size(), "sample");
    if (db.rows.size() != expected)
        throw ReaderError("incomplete database: rows != directions*frequencies*polarizations");
    for (size_t i = 0; i < db.rows.size(); ++i) {
        if (db.rows[i].sample_id != i)
            throw ReaderError("non-sequential sample_id: coverage gap or reorder");
        const size_t npol = db.polarizations.size();
        const size_t ndir = db.azimuth_deg.size();
        if (db.rows[i].polarization_index != i % npol ||
            db.rows[i].direction_index != (i / npol) % ndir ||
            db.rows[i].frequency_index != (i / npol) / ndir)
            throw ReaderError("sample indices do not match sample_id order");
        if (!std::isfinite(db.rows[i].scattering_real) ||
            !std::isfinite(db.rows[i].scattering_imag) ||
            !std::isfinite(db.rows[i].rcs_sqm) || db.rows[i].rcs_sqm < 0.0 ||
            std::isnan(db.rows[i].rcs_dbsm) ||
            db.rows[i].rcs_dbsm == std::numeric_limits<double>::infinity())
            throw ReaderError("non-finite or invalid sample value");
        if (!db.rows[i].valid) throw ReaderError("invalid sample present");
        if (db.rows[i].status == "error") throw ReaderError("error sample present");
    }
    const size_t n_chunks = expected_chunks(db.rows.size(), db.chunk_rows);
    if (db.completed_chunks.size() != n_chunks || db.chunk_checksums.size() != n_chunks)
        throw ReaderError("corrupt progress bookkeeping: chunk coverage mismatch");
    for (uint8_t flag : db.completed_chunks) {
        if (flag > 1) throw ReaderError("corrupt progress: invalid completion flag");
        if (flag == 0) throw ReaderError("incomplete progress: uncommitted chunk");
    }
    if (db.completed_chunks.empty() && !db.rows.empty())
        throw ReaderError("incomplete progress: no committed chunks");
    // Checksum over exactly the stored bytes, chunk by chunk.
    for (size_t c = 0; c < db.completed_chunks.size(); ++c) {
        const size_t begin = c * db.chunk_rows;
        const size_t end = begin + std::min(db.chunk_rows, db.rows.size() - begin);
        std::vector<ChecksumRow> rows;
        rows.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
            const Hdf5SampleRow& r = db.rows[i];
            ChecksumRow cr;
            cr.sample_id = r.sample_id;
            cr.direction_index = r.direction_index;
            cr.frequency_index = r.frequency_index;
            cr.polarization_index = r.polarization_index;
            cr.scattering_real = r.scattering_real;
            cr.scattering_imag = r.scattering_imag;
            cr.rcs_sqm = r.rcs_sqm;
            cr.rcs_dbsm = r.rcs_dbsm;
            cr.valid = r.valid ? 1 : 0;
            cr.status = r.status;
            rows.push_back(std::move(cr));
        }
        if (chunk_checksum(rows.data(), rows.size()) != db.chunk_checksums[c])
            throw ReaderError("chunk checksum mismatch: data corruption");
    }
}

} // namespace strikecem

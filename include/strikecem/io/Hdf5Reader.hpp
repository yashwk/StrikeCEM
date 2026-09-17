#pragma once
// Reference HDF5 reader (StrikeEngine contract, SPEC section 10): loads the
// flat sample table and rejects anything that is not a complete, compatible
// database. Throws ReaderError.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strikecem {

struct ReaderError : public std::runtime_error {
    explicit ReaderError(const std::string& msg) : std::runtime_error(msg) {}
};

struct Hdf5SampleRow {
    uint64_t sample_id = 0;
    uint32_t direction_index = 0;
    uint32_t frequency_index = 0;
    uint16_t polarization_index = 0;
    double scattering_real = 0.0;
    double scattering_imag = 0.0;
    double rcs_sqm = 0.0;
    double rcs_dbsm = 0.0;
    bool valid = false;
    std::string status;
};

struct Hdf5Database {
    std::vector<double> azimuth_deg;
    std::vector<double> elevation_deg;
    std::vector<double> frequencies_hz;
    std::vector<std::string> polarizations;
    std::vector<Hdf5SampleRow> rows; // sample_id order
    std::vector<uint8_t> completed_chunks;
    std::string output_format_version;
    std::string scem_schema_version;
    std::string config_hash;
    std::string solver_precision;
};

Hdf5Database read_hdf5_database(const std::string& path);

// Rejects: unknown major format version, missing samples, non-sequential
// IDs, invalid rows, error rows, incomplete progress.
void validate_hdf5_complete(const Hdf5Database& db);

} // namespace strikecem

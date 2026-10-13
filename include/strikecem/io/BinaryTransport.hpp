#pragma once
// Compact zero-overhead binary transport format (.scem)
// Explicit endianness, header magic, schema versioning, index tables,
// payload integrity checksums (FNV-1a 64), and flat sample rows.
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/DesignerPackage.hpp"
#include "strikecem/io/Hdf5Reader.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/io/OutputError.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace strikecem {

#pragma pack(push, 1)
struct ScemHeader {
    char magic[4] = {'S', 'C', 'E', 'M'};
    uint16_t version = 1;
    uint16_t header_size = 256;
    uint32_t endianness_marker = 0x01020304;
    uint8_t dtype = 1; // 0 = float32, 1 = float64
    uint8_t flags = 0;
    uint16_t reserved_u16 = 0;
    uint64_t sample_count = 0;
    uint32_t frequency_count = 0;
    uint32_t direction_count = 0;
    uint16_t polarization_count = 0;
    uint16_t reserved_u16_2 = 0;
    uint64_t frequencies_offset = 0;
    uint64_t frequencies_bytes = 0;
    uint64_t directions_offset = 0;
    uint64_t directions_bytes = 0;
    uint64_t polarizations_offset = 0;
    uint64_t polarizations_bytes = 0;
    uint64_t metadata_offset = 0;
    uint64_t metadata_bytes = 0;
    uint64_t samples_offset = 0;
    uint64_t samples_bytes = 0;
    uint64_t header_checksum = 0;
    uint64_t data_checksum = 0;
    uint8_t reserved[124] = {0};
};
static_assert(sizeof(ScemHeader) == 256, "ScemHeader must be exactly 256 bytes");

struct ScemDirectionRecord {
    double azimuth_deg = 0.0;
    double elevation_deg = 0.0;
    double k_x = 0.0;
    double k_y = 0.0;
    double k_z = 0.0;
};
static_assert(sizeof(ScemDirectionRecord) == 40, "ScemDirectionRecord must be exactly 40 bytes");

struct ScemPolarizationRecord {
    char name[8] = {0};
};
static_assert(sizeof(ScemPolarizationRecord) == 8, "ScemPolarizationRecord must be exactly 8 bytes");

struct ScemSampleRowF64 {
    uint64_t sample_id = 0;
    uint32_t direction_index = 0;
    uint32_t frequency_index = 0;
    uint16_t polarization_index = 0;
    uint8_t valid = 0;
    uint8_t status_code = 0; // 0=pending, 1=ok, 2=warning, 3=error
    double scattering_real = 0.0;
    double scattering_imag = 0.0;
    double rcs_sqm = 0.0;
    double rcs_dbsm = 0.0;
};
static_assert(sizeof(ScemSampleRowF64) == 52, "ScemSampleRowF64 must be exactly 52 bytes");

struct ScemSampleRowF32 {
    uint64_t sample_id = 0;
    uint32_t direction_index = 0;
    uint32_t frequency_index = 0;
    uint16_t polarization_index = 0;
    uint8_t valid = 0;
    uint8_t status_code = 0;
    float scattering_real = 0.0f;
    float scattering_imag = 0.0f;
    float rcs_sqm = 0.0f;
    float rcs_dbsm = 0.0f;
};
static_assert(sizeof(ScemSampleRowF32) == 36, "ScemSampleRowF32 must be exactly 36 bytes");
#pragma pack(pop)

struct ScemDatabase {
    uint16_t version = 0;
    uint8_t dtype = 1;
    std::vector<double> frequencies_hz;
    std::vector<Direction> directions;
    std::vector<std::string> polarizations;
    nlohmann::json metadata;
    std::vector<Hdf5SampleRow> rows;
    uint64_t payload_checksum = 0;
};

void write_scem_output(const ResolvedConfig& rc, const SamplePlan& plan,
                       const NormalizedMesh& mesh, const PoResult& result,
                       const DesignerIdentity& designer = {},
                       const FringeOptions& fringe = {},
                       const MaterialOptions& material = {});

ScemDatabase read_scem_database(const std::string& path);

void validate_scem_complete(const ScemDatabase& db);

bool check_existing_scem(const ResolvedConfig& rc, const SamplePlan& plan,
                         const NormalizedMesh& mesh, const std::string& path);

} // namespace strikecem

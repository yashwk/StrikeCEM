#pragma once
// FNV-1a 64 checksums over canonical sample-row bytes. The writer checksums
// read-back chunk data; resume and the reader verify identically, so the
// hash must cover exactly the stored values (floats as stored, doubles as
// stored) in sample_id order.
#include <cstdint>
#include <string>
#include <vector>

namespace strikecem {

inline uint64_t fnv1a_64(const void* data, size_t len, uint64_t hash = 14695981039346656037ULL) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

struct ChecksumRow {
    uint64_t sample_id = 0;
    uint32_t direction_index = 0;
    uint32_t frequency_index = 0;
    uint16_t polarization_index = 0;
    double scattering_real = 0.0;
    double scattering_imag = 0.0;
    double rcs_sqm = 0.0;
    double rcs_dbsm = 0.0;
    uint8_t valid = 0;
    std::string status;
};

inline void checksum_append(uint64_t& hash, const void* data, size_t len) {
    hash = fnv1a_64(data, len, hash);
}

inline uint64_t chunk_checksum(const ChecksumRow* rows, size_t n) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < n; ++i) {
        const ChecksumRow& r = rows[i];
        checksum_append(hash, &r.sample_id, sizeof(r.sample_id));
        checksum_append(hash, &r.direction_index, sizeof(r.direction_index));
        checksum_append(hash, &r.frequency_index, sizeof(r.frequency_index));
        checksum_append(hash, &r.polarization_index, sizeof(r.polarization_index));
        checksum_append(hash, &r.scattering_real, sizeof(r.scattering_real));
        checksum_append(hash, &r.scattering_imag, sizeof(r.scattering_imag));
        checksum_append(hash, &r.rcs_sqm, sizeof(r.rcs_sqm));
        checksum_append(hash, &r.rcs_dbsm, sizeof(r.rcs_dbsm));
        checksum_append(hash, &r.valid, sizeof(r.valid));
        checksum_append(hash, r.status.data(), r.status.size());
    }
    return hash;
}

} // namespace strikecem

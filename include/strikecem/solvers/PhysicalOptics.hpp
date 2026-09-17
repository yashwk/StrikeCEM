#pragma once
// Deterministic CPU Physical Optics reference solver (SPEC FR-6).
// Implements ADR-0001: F_m = (jkη/4π)·A·[r̂×(r̂×J_m)]·e^{+jkr̂·rc},
// monostatic (r̂ = −k̂), hard illumination, PEC, single-threaded.
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"

namespace strikecem {

struct PoChannelSample {
    std::complex<double> scattering{0.0, 0.0}; // normalized F/E0
    double rcs_sqm = 0.0;
};

struct PoSampleResult {
    // One row per (frequency, direction, polarization) of the output table.
    uint64_t sample_id = 0; // ((frequency_id * N) + direction_id) * P + pol_id
    uint32_t frequency_id = 0;
    uint32_t direction_id = 0;
    uint32_t pol_id = 0;
    std::complex<double> scattering{0.0, 0.0}; // normalized F/E0
    double rcs_sqm = 0.0;
    size_t lit_facets = 0;
};

struct PoResult {
    std::vector<PoSampleResult> samples; // plan order
    std::vector<std::string> warnings;
};

constexpr double kSpeedOfLight = 299792458.0;
constexpr double kEta0 = 376.73031346177066; // μ0·c, ohms

PoResult solve_po(const NormalizedMesh& mesh, const SamplePlan& plan,
                  const nlohmann::json& resolved);

} // namespace strikecem

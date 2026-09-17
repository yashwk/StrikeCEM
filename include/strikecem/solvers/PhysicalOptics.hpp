#pragma once
// Deterministic CPU Physical Optics reference solver (SPEC FR-6).
// Implements ADR-0001: F_m = (jkη/4π)·A·[r̂×(r̂×J_m)]·e^{+jkr̂·rc},
// monostatic (r̂ = −k̂), hard illumination, PEC. Units are independent
// rows, so results are thread-count invariant by construction.
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/EdgeFringe.hpp"
#include "strikecem/solvers/Ray.hpp"

namespace strikecem {

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

// Fringe (edge-diffraction) correction, off by default. Enabled only via
// this API until the schema exposes edge_correction (slice D2d); the
// schema still rejects anything but "none".
struct FringeOptions {
    bool enabled = false;
    const EdgeModel* edges = nullptr; // owned by the caller; required when enabled
};

PoResult solve_po(const NormalizedMesh& mesh, const SamplePlan& plan,
                  const nlohmann::json& resolved,
                  const FringeOptions& fringe = FringeOptions{},
                  const ShadowOptions& shadow = ShadowOptions{});

// Solve only the given (frequency_id, direction_id) units with correct
// global sample_ids (resume path for missing chunks).
PoResult solve_po_units(const NormalizedMesh& mesh, const SamplePlan& plan,
                        const nlohmann::json& resolved,
                        const std::vector<std::pair<uint32_t, uint32_t>>& units,
                        const FringeOptions& fringe = FringeOptions{},
                        const ShadowOptions& shadow = ShadowOptions{});

} // namespace strikecem

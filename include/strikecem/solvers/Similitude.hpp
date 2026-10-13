#pragma once
// Electromagnetic similitude and scale-model transformation engine (ADR-0029).
// Applies electrodynamic scaling theorems (Sinclair/Stratton-Chu) to simulate
// large vehicles via scale models at proportionately upscaled frequencies,
// exactly renormalizing output radar cross sections by +20*log10(scale) dB.
#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace strikecem {

enum class SimilitudeDirection {
    FullToModel = 0, // Input is full-scale; scale geometry down by s_g and frequency up by 1/s_g for solve
    ModelToFull = 1  // Input is scale model; solve as given and project up to full-scale vehicle
};

struct SimilitudeConfig {
    bool enabled = false;
    double geometric_scale = 1.0; // s_g = L_model / L_full (e.g. 0.1 for 1:10 scale)
    SimilitudeDirection direction = SimilitudeDirection::FullToModel;
    bool auto_rescale_rcs = true; // Renormalize output RCS to target full-scale domain
};

struct SimilitudeSummary {
    double geometric_scale = 1.0;
    double frequency_scale = 1.0;
    double rcs_scale_factor = 1.0; // Linear sqm multiplier
    double rcs_offset_dbsm = 0.0;  // dBsm offset: +20 * log10(1 / s_g)
    double input_frequency_hz = 0.0;
    double scaled_frequency_hz = 0.0;
    double input_bbox_diagonal_m = 0.0;
    double scaled_bbox_diagonal_m = 0.0;
};

// Pure scaling math functions
double scale_frequency(double freq_hz, double geometric_scale,
                       SimilitudeDirection dir = SimilitudeDirection::FullToModel);

double scale_wavelength(double wavelength_m, double geometric_scale,
                        SimilitudeDirection dir = SimilitudeDirection::FullToModel);

double rescale_rcs_sqm(double rcs_sqm, double geometric_scale,
                       SimilitudeDirection dir = SimilitudeDirection::FullToModel);

double rescale_rcs_dbsm(double rcs_dbsm, double geometric_scale,
                        SimilitudeDirection dir = SimilitudeDirection::FullToModel);

std::complex<double> rescale_scattering_amplitude(std::complex<double> S, double geometric_scale,
                                                 SimilitudeDirection dir = SimilitudeDirection::FullToModel);

// Transforms mesh coordinates by geometric scale
NormalizedMesh scale_mesh(const NormalizedMesh& mesh, double geometric_scale,
                          SimilitudeDirection dir = SimilitudeDirection::FullToModel);

// Parses similitude block from config JSON
SimilitudeConfig parse_similitude_config(const nlohmann::json& config_json);

// Applies similitude scaling to config, sample plan, and mesh prior to solve
SimilitudeSummary apply_similitude(ResolvedConfig& rc, SamplePlan& plan, NormalizedMesh& mesh,
                                   const SimilitudeConfig& sim_cfg);

// Rescales PoResult back to target domain values
void rescale_po_result(PoResult& result, const SimilitudeConfig& sim_cfg);

} // namespace strikecem

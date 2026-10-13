#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <nlohmann/json.hpp>

#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/math/Geometry.hpp"

namespace strikecem {

struct RemeshOptions {
    std::string frequency_policy = "user_mesh"; // "user_mesh", "mesh_at_max_frequency", "mesh_per_frequency"
    double alpha_smooth = 1.0 / 6.0;            // h_smooth <= alpha_smooth * lambda
    double alpha_edge = 1.0 / 20.0;             // h_edge <= alpha_edge * lambda
    double curvature_factor = 0.1;              // curvature sensitivity
    size_t max_refinement_depth = 5;            // max recursive splits
};

// Compute free-space or medium wavelength (lambda = c / (f * sqrt(eps_r * mu_r)))
double compute_wavelength_m(double freq_hz, double eps_r = 1.0, double mu_r = 1.0);

// Conforming adaptive subdivision refining edges longer than h_target
NormalizedMesh remesh_target_edge_length(const NormalizedMesh& mesh, double h_target,
                                         size_t max_depth = 5);

// Apply wavelength-dependent remeshing given solver plan frequencies and options
NormalizedMesh remesh_wavelength(const NormalizedMesh& mesh, const nlohmann::json& resolved,
                                 const std::vector<double>& frequencies_hz);

} // namespace strikecem

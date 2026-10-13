#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

#include "strikecem/core/Config.hpp"
#include "strikecem/math/Geometry.hpp"

namespace strikecem {

struct SphericalStats {
    double total_power_sqm = 0.0;
    double mean_rcs_sqm = 0.0;
    double mean_rcs_dbsm = -999.0;
    double min_rcs_dbsm = 0.0;
    double max_rcs_dbsm = -999.0;
    size_t peak_direction_index = 0;
};

// Generates an equal-area Fibonacci spiral distribution on S^2 or a spherical cap/band.
// Samples are ordered deterministically from elevation_min to elevation_max.
// Each direction is assigned an equal solid angle weight w_i = DeltaOmega / N.
std::vector<Direction> generate_fibonacci_sphere(
    size_t num_points,
    double el_min_deg = -90.0,
    double el_max_deg = 90.0);

// Spherical quadrature integration: integral_{Omega} sigma(Omega) dOmega = sum_i w_i * sigma_i
double integrate_scattered_power(
    const std::vector<Direction>& directions,
    const std::vector<double>& rcs_sqm);

// Computes solid-angle weighted statistics over the spherical sample set
SphericalStats compute_spherical_stats(
    const std::vector<Direction>& directions,
    const std::vector<double>& rcs_sqm);

} // namespace strikecem

#include "strikecem/sampling/SphericalSampling.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "strikecem/core/Conventions.hpp"

namespace strikecem {

std::vector<Direction> generate_fibonacci_sphere(
    size_t num_points,
    double el_min_deg,
    double el_max_deg) {
    if (num_points == 0) {
        throw std::invalid_argument("num_points must be at least 1");
    }
    if (el_min_deg < -90.0 || el_max_deg > 90.0 || el_min_deg > el_max_deg) {
        throw std::invalid_argument("invalid elevation range for spherical sampling");
    }

    const double deg_to_rad = std::numbers::pi / 180.0;
    const double rad_to_deg = 180.0 / std::numbers::pi;
    const double golden_angle = std::numbers::pi * (3.0 - std::sqrt(5.0)); // ~2.3999632 rad

    const double z_min = std::sin(el_min_deg * deg_to_rad);
    const double z_max = std::sin(el_max_deg * deg_to_rad);
    const double delta_omega = 2.0 * std::numbers::pi * (z_max - z_min);
    const double weight = delta_omega / static_cast<double>(num_points);

    std::vector<Direction> directions;
    directions.reserve(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        const double t = (static_cast<double>(i) + 0.5) / static_cast<double>(num_points);
        double z = z_min + t * (z_max - z_min);
        z = std::clamp(z, -1.0, 1.0);

        const double r_xy = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double phi = std::fmod(static_cast<double>(i) * golden_angle, 2.0 * std::numbers::pi);

        const double x = r_xy * std::cos(phi);
        const double y = r_xy * std::sin(phi);

        Direction dir;
        dir.id = static_cast<uint32_t>(i);
        dir.k_hat = {x, y, z};
        dir.k_hat.normalize();

        const double el_deg = std::asin(std::clamp(dir.k_hat.z, -1.0, 1.0)) * rad_to_deg;
        const double az_deg = normalize_azimuth(std::atan2(dir.k_hat.y, dir.k_hat.x) * rad_to_deg);

        dir.azimuth_deg = az_deg;
        dir.elevation_deg = el_deg;
        dir.weight = weight;

        directions.push_back(dir);
    }

    return directions;
}

double integrate_scattered_power(
    const std::vector<Direction>& directions,
    const std::vector<double>& rcs_sqm) {
    if (directions.size() != rcs_sqm.size()) {
        throw std::invalid_argument("directions and rcs_sqm sizes do not match");
    }

    double total = 0.0;
    for (size_t i = 0; i < directions.size(); ++i) {
        total += directions[i].weight * rcs_sqm[i];
    }
    return total;
}

SphericalStats compute_spherical_stats(
    const std::vector<Direction>& directions,
    const std::vector<double>& rcs_sqm) {
    if (directions.empty() || directions.size() != rcs_sqm.size()) {
        throw std::invalid_argument("invalid inputs to compute_spherical_stats");
    }

    double total_power = 0.0;
    double total_weight = 0.0;
    double max_rcs = -1.0;
    double min_rcs = 1e30;
    size_t peak_idx = 0;

    for (size_t i = 0; i < directions.size(); ++i) {
        const double val = rcs_sqm[i];
        total_power += directions[i].weight * val;
        total_weight += directions[i].weight;

        if (val > max_rcs) {
            max_rcs = val;
            peak_idx = i;
        }
        if (val < min_rcs) {
            min_rcs = val;
        }
    }

    SphericalStats stats;
    stats.total_power_sqm = total_power;
    stats.mean_rcs_sqm = (total_weight > 0.0) ? (total_power / total_weight) : 0.0;
    stats.mean_rcs_dbsm = (stats.mean_rcs_sqm > 1e-30) ? (10.0 * std::log10(stats.mean_rcs_sqm)) : -999.0;
    stats.min_rcs_dbsm = (min_rcs > 1e-30) ? (10.0 * std::log10(min_rcs)) : -999.0;
    stats.max_rcs_dbsm = (max_rcs > 1e-30) ? (10.0 * std::log10(max_rcs)) : -999.0;
    stats.peak_direction_index = peak_idx;

    return stats;
}

} // namespace strikecem

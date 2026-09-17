#pragma once
// StrikeCEM v1 physical conventions (see design docs: CONVENTIONS.md).
// Time dependence e^{+j\omega t}. Angles in degrees at the API boundary;
// radians only inside numerical kernels.

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

#include "strikecem/math/Geometry.hpp"

namespace strikecem {

inline double deg_to_rad(double deg) { return deg * std::numbers::pi / 180.0; }
inline double rad_to_deg(double rad) { return rad * 180.0 / std::numbers::pi; }

// Fold to [0, 360). 360 maps to 0 and is never emitted twice.
inline double normalize_azimuth(double azimuth_deg) {
    double a = std::fmod(azimuth_deg, 360.0);
    if (a < 0.0) a += 360.0;
    if (a == 0.0) return 0.0; // force +0.0 for canonical output
    return a;
}

// k_hat from (az, el): az about +z from +x toward +y, el from xy-plane to +z.
inline geom::Vec3d direction_from_az_el(double azimuth_deg, double elevation_deg) {
    const double az = deg_to_rad(azimuth_deg);
    const double el = deg_to_rad(elevation_deg);
    return {std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el)};
}

inline void az_el_from_direction(const geom::Vec3d& k, double& azimuth_deg,
                                 double& elevation_deg) {
    const double n = k.length();
    if (!(n > 0.0)) throw std::invalid_argument("zero direction vector");
    const double z = std::max(-1.0, std::min(1.0, k.z / n));
    elevation_deg = rad_to_deg(std::asin(z));
    azimuth_deg = normalize_azimuth(rad_to_deg(std::atan2(k.y / n, k.x / n)));
}

struct PolarizationBasis {
    geom::Vec3d v; // local vertical
    geom::Vec3d h; // local horizontal, h = v x k
};

// v is the normalized projection of +z onto the plane perpendicular to k_hat.
// At the poles +x is the deterministic reference axis.
inline PolarizationBasis polarization_basis(const geom::Vec3d& k_hat) {
    geom::Vec3d v;
    if (std::abs(std::abs(k_hat.z) - 1.0) < 1e-12) {
        v = {1.0, 0.0, 0.0};
    } else {
        v = geom::Vec3d{0.0, 0.0, 1.0} - k_hat * k_hat.z;
        v.normalize();
    }
    return {v, geom::cross(v, k_hat)};
}

using Complex2x2 = std::array<std::array<std::complex<double>, 2>, 2>;

// Linear (H,V) scattering matrix to circular (R,L) under the IEEE radar
// convention with e^{+jwt}: S_circ = U S_lin U^T, rows R=(H-jV)/sqrt2,
// L=(H+jV)/sqrt2. Physical pin: an odd-bounce target (sphere, S_lin = I)
// has null co-circular return (RR = LL = 0) and unit cross-circular
// (RL = LR = 1); handedness flips on reflection.
inline Complex2x2 linear_to_circular(const Complex2x2& s) {
    const std::complex<double> j(0.0, 1.0);
    const double inv = 1.0 / std::sqrt(2.0);
    // U rows: R = (1, -j)/sqrt2, L = (1, +j)/sqrt2 over (H, V).
    const std::array<std::array<std::complex<double>, 2>, 2> u = {
        std::array<std::complex<double>, 2>{{inv, -inv * j}},
        std::array<std::complex<double>, 2>{{inv, inv * j}}};
    Complex2x2 out{};
    for (int i = 0; i < 2; ++i)
        for (int m = 0; m < 2; ++m)
            for (int k = 0; k < 2; ++k)
                for (int l = 0; l < 2; ++l)
                    out[i][m] += u[i][k] * s[k][l] * u[m][l];
    return out;
}

// RCS conversions. Exact zero maps to -inf dBsm; an uncomputed sample is
// valid=false, never NaN. Negative or NaN square-metre input is rejected.
inline double rcs_sqm_to_dbsm(double sqm) {
    if (std::isnan(sqm) || sqm < 0.0)
        throw std::invalid_argument("RCS in square metres must be non-negative");
    if (sqm == 0.0) return -std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(sqm);
}

inline double rcs_dbsm_to_sqm(double dbsm) {
    if (dbsm == -std::numeric_limits<double>::infinity()) return 0.0;
    return std::pow(10.0, dbsm / 10.0);
}

inline double units_to_metres(const std::string& units) {
    if (units == "m") return 1.0;
    if (units == "mm") return 1e-3;
    if (units == "in") return 0.0254;
    throw std::invalid_argument("unknown length units: " + units);
}

// Mesh transform order: scale, then active right-handed Rx, Ry, Rz,
// then translate.
inline geom::Vec3d apply_transform(const geom::Vec3d& p, double scale,
                                   const geom::Vec3d& euler_deg,
                                   const geom::Vec3d& translate) {
    geom::Vec3d q = p * scale;
    const double rx = deg_to_rad(euler_deg.x);
    const double ry = deg_to_rad(euler_deg.y);
    const double rz = deg_to_rad(euler_deg.z);
    { // Rx
        const double c = std::cos(rx), s = std::sin(rx);
        q = {q.x, c * q.y - s * q.z, s * q.y + c * q.z};
    }
    { // Ry
        const double c = std::cos(ry), s = std::sin(ry);
        q = {c * q.x + s * q.z, q.y, -s * q.x + c * q.z};
    }
    { // Rz
        const double c = std::cos(rz), s = std::sin(rz);
        q = {c * q.x - s * q.y, s * q.x + c * q.y, q.z};
    }
    return q + translate;
}

} // namespace strikecem

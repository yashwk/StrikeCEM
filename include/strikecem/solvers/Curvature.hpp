#pragma once

#include <vector>
#include <cstdint>
#include <complex>

#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/math/Geometry.hpp"

namespace strikecem {

struct CurvatureInfo {
    double mean_curvature = 0.0;     // H = (kappa_1 + kappa_2) / 2 [1/m]
    double max_curvature = 0.0;      // max(|kappa_1|, |kappa_2|) [1/m]
    bool is_discontinuous = false;   // edge discontinuity / crease
};

// Estimate per-triangle principal and mean curvatures from adjacent facet normals
std::vector<CurvatureInfo> estimate_mesh_curvatures(const NormalizedMesh& mesh,
                                                    double max_smooth_angle_deg = 45.0);

// Compute asymptotic Fock/Ufimtsev Physical Optics curvature correction multiplier
std::complex<double> po_curvature_factor(double mean_curvature, double k, double cos_inc);

} // namespace strikecem

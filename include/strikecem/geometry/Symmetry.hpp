#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "strikecem/math/Geometry.hpp"
#include "strikecem/io/MeshLoader.hpp"

namespace strikecem {

struct SymmetryOptions {
    bool enabled = false;
    std::string plane = "xz"; // "xz" (y -> -y), "yz" (x -> -x), "xy" (z -> -z)
    double tolerance_m = 1e-4;
};

// Coordinate accessor along normal axis of symmetry plane
double get_symmetry_coord(const geom::Vec3d& p, const std::string& plane);

// Reflect a 3D coordinate point across the chosen symmetry plane
geom::Vec3d reflect_point(const geom::Vec3d& p, const std::string& plane);

// Reflect a normal vector across the chosen symmetry plane
geom::Vec3d reflect_normal(const geom::Vec3d& n, const std::string& plane);

// Verify if a mesh possesses bilateral symmetry across the chosen plane
bool is_symmetric_mesh(const NormalizedMesh& mesh, const SymmetryOptions& opts);

struct HalfMeshResult {
    NormalizedMesh mesh;
    std::vector<uint32_t> original_indices;
    std::vector<uint32_t> reflected_indices; // original index of reflection in full mesh, or UINT32_MAX if boundary or none
};

// Extract the half-domain mesh (positive half-space of the symmetry axis)
HalfMeshResult extract_half_mesh(const NormalizedMesh& mesh, const SymmetryOptions& opts);

} // namespace strikecem

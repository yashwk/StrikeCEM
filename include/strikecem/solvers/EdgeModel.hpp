#pragma once
// Straight-edge diffraction model (ADR-0003): boundary rims plus interior
// creases above a dihedral threshold, with exterior wedge parameter,
// convexity, and adjacent faces. Consumed by the PTD/UTD solvers; no
// fringe physics here.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "strikecem/io/MeshLoader.hpp"

namespace strikecem {

struct MeshEdge {
    geom::Vec3d p0, p1;   // endpoints, lower vertex index first
    geom::Vec3d tangent;  // unit, p0 -> p1
    double length = 0.0;
    geom::Vec3d n0, n1;   // adjacent outward face normals (n1 == n0 on rims)
    geom::Vec3d face0_dir; // unit, perpendicular to tangent, into face tri0
    geom::Vec3d face1_dir; // unit, perpendicular to tangent, into face tri1 (= face0 on rims)
    int32_t tri0 = -1;    // adjacent triangle (lower index)
    int32_t tri1 = -1;    // second triangle, -1 on boundary rims
    bool boundary = false;
    double wedge_n = 2.0; // exterior angle / pi (half-plane rim = 2)
    bool convex = true;   // false = re-entrant (valley) wedge
};

struct EdgeModel {
    std::vector<MeshEdge> edges; // selected diffracting edges
    size_t smooth_skipped = 0;   // coplanar interior pairs below threshold
};

// Throws std::invalid_argument on non-manifold edges (the loader rejects
// those before the edge model ever runs).
EdgeModel extract_edges(const NormalizedMesh& mesh,
                        double dihedral_threshold_rad = 0.1);

} // namespace strikecem

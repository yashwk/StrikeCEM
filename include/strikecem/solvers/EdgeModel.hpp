#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "strikecem/io/MeshLoader.hpp"

namespace strikecem {

struct MeshEdge {
    geom::Vec3d p0, p1;
    geom::Vec3d tangent;
    double length = 0.0;
    geom::Vec3d n0, n1;
    geom::Vec3d face0_dir;
    geom::Vec3d face1_dir;
    int32_t tri0 = -1;
    int32_t tri1 = -1;
    bool boundary = false;
    double wedge_n = 2.0;
    bool convex = true;
};

struct EdgeModel {
    std::vector<MeshEdge> edges;
    size_t smooth_skipped = 0;
};

EdgeModel extract_edges(const NormalizedMesh& mesh,
                        double dihedral_threshold_rad = 0.1);

}

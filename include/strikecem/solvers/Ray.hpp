#pragma once
#include <cstdint>
#include <optional>

#include "strikecem/io/MeshLoader.hpp"

namespace strikecem {

struct RayHit {
    double t = 0.0;
    uint32_t tri = 0;
    double bary_u = 0.0;
    double bary_v = 0.0;
};

struct GoOptions {
    bool shadowing = false;
    int max_bounces = 1;
};

struct BvhNode {
    geom::Vec3d bmin;
    geom::Vec3d bmax;
    int left = -1;
    int right = -1;
    uint32_t start = 0;
    uint32_t count = 0;
};

struct Bvh {
    std::vector<BvhNode> nodes;
    std::vector<uint32_t> order;
};

Bvh build_bvh(const NormalizedMesh& mesh);

std::optional<RayHit> ray_bvh(const Bvh& bvh, const NormalizedMesh& mesh,
                               const geom::Vec3d& origin, const geom::Vec3d& dir, double t_min,
                               uint32_t skip_tri);

bool occluded_bvh(const Bvh& bvh, const NormalizedMesh& mesh, const geom::Vec3d& p,
                  const geom::Vec3d& s, double max_t, uint32_t self_tri);

std::optional<RayHit> ray_triangle(const geom::Vec3d& origin, const geom::Vec3d& dir,
                                   const geom::Vec3d& a, const geom::Vec3d& b,
                                   const geom::Vec3d& c, double t_min, uint32_t tri = 0);

std::optional<RayHit> ray_mesh(const NormalizedMesh& mesh, const geom::Vec3d& origin,
                               const geom::Vec3d& dir, double t_min, uint32_t skip_tri);

bool occluded(const NormalizedMesh& mesh, const geom::Vec3d& p, const geom::Vec3d& s,
              double max_t, uint32_t self_tri);

}

#include "strikecem/solvers/Ray.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace strikecem {
namespace {

constexpr double kDetEps = 1e-18;
constexpr double kBaryTol = 1e-9;

bool is_unit(const geom::Vec3d& v) {
    if (!std::isfinite(v.x + v.y + v.z)) return false;
    const double n = v.length();
    return std::isfinite(n) && std::abs(n - 1.0) <= 1e-6;
}

}

std::optional<RayHit> ray_triangle(const geom::Vec3d& origin, const geom::Vec3d& dir,
                                   const geom::Vec3d& a, const geom::Vec3d& b,
                                   const geom::Vec3d& c, double t_min, uint32_t tri) {
    const geom::Vec3d e1 = b - a;
    const geom::Vec3d e2 = c - a;
    const geom::Vec3d p = geom::cross(dir, e2);
    const double det = geom::dot(e1, p);
    if (std::abs(det) < kDetEps) return std::nullopt;
    const double inv = 1.0 / det;
    const geom::Vec3d q = origin - a;
    const double bary_u = geom::dot(q, p) * inv;
    if (bary_u < -kBaryTol || bary_u > 1.0 + kBaryTol) return std::nullopt;
    const geom::Vec3d r = geom::cross(q, e1);
    const double bary_v = geom::dot(dir, r) * inv;
    if (bary_v < -kBaryTol || bary_u + bary_v > 1.0 + kBaryTol) return std::nullopt;
    const double t = geom::dot(e2, r) * inv;
    if (!(t > t_min) || !std::isfinite(t)) return std::nullopt;
    return RayHit{t, tri, bary_u, bary_v};
}

std::optional<RayHit> ray_mesh(const NormalizedMesh& mesh, const geom::Vec3d& origin,
                               const geom::Vec3d& dir, double t_min, uint32_t skip_tri) {
    if (!is_unit(dir)) throw std::invalid_argument("ray direction must be unit");
    if (!(t_min >= 0.0) || !std::isfinite(t_min))
        throw std::invalid_argument("t_min must be non-negative and finite");
    std::optional<RayHit> best;
    for (uint32_t t = 0; t < mesh.triangles.size(); ++t) {
        if (t == skip_tri) continue;
        const auto& tri = mesh.triangles[t];
        const auto hit = ray_triangle(origin, dir, mesh.vertices[tri[0]], mesh.vertices[tri[1]],
                                      mesh.vertices[tri[2]], t_min, t);
        if (hit && (!best || hit->t < best->t)) best = hit;
    }
    return best;
}

bool occluded(const NormalizedMesh& mesh, const geom::Vec3d& p, const geom::Vec3d& s,
              double max_t, uint32_t self_tri) {
    if (!is_unit(s)) throw std::invalid_argument("occlusion direction must be unit");
    if (!(max_t > 0.0) || !std::isfinite(max_t))
        throw std::invalid_argument("max_t must be positive and finite");
    const geom::Vec3d diag = mesh.report.bbox_max - mesh.report.bbox_min;
    const double t_min = 1e-9 * diag.length();
    for (uint32_t t = 0; t < mesh.triangles.size(); ++t) {
        if (t == self_tri) continue;
        const auto& tri = mesh.triangles[t];
        const auto hit = ray_triangle(p, s, mesh.vertices[tri[0]], mesh.vertices[tri[1]],
                                      mesh.vertices[tri[2]], t_min, t);
        if (hit && hit->t < max_t) return true;
    }
    return false;
}

}

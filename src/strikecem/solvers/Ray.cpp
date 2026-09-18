#include "strikecem/solvers/Ray.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

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

namespace {

constexpr uint32_t kMaxLeafTris = 8;

struct BuildItem {
    uint32_t tri;
    geom::Vec3d centroid;
    geom::Vec3d bmin;
    geom::Vec3d bmax;
};

BuildItem make_item(const NormalizedMesh& mesh, uint32_t t) {
    const auto& tri = mesh.triangles[t];
    const geom::Vec3d a = mesh.vertices[tri[0]];
    const geom::Vec3d b = mesh.vertices[tri[1]];
    const geom::Vec3d c = mesh.vertices[tri[2]];
    BuildItem item;
    item.tri = t;
    item.centroid = (a + b + c) * (1.0 / 3.0);
    item.bmin = {std::min({a.x, b.x, c.x}), std::min({a.y, b.y, c.y}),
                 std::min({a.z, b.z, c.z})};
    item.bmax = {std::max({a.x, b.x, c.x}), std::max({a.y, b.y, c.y}),
                 std::max({a.z, b.z, c.z})};
    return item;
}

int longest_axis(const geom::Vec3d& lo, const geom::Vec3d& hi) {
    const geom::Vec3d e = hi - lo;
    return (e.x >= e.y && e.x >= e.z) ? 0 : ((e.y >= e.z) ? 1 : 2);
}

double axis_of(const geom::Vec3d& v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

uint32_t build_node(Bvh& bvh, std::vector<BuildItem>& items, uint32_t begin, uint32_t end) {
    const uint32_t node_id = uint32_t(bvh.nodes.size());
    bvh.nodes.push_back({});
    geom::Vec3d bmin = items[begin].bmin;
    geom::Vec3d bmax = items[begin].bmax;
    geom::Vec3d clo = items[begin].centroid;
    geom::Vec3d chi = items[begin].centroid;
    for (uint32_t i = begin + 1; i < end; ++i) {
        bmin = {std::min(bmin.x, items[i].bmin.x), std::min(bmin.y, items[i].bmin.y),
                std::min(bmin.z, items[i].bmin.z)};
        bmax = {std::max(bmax.x, items[i].bmax.x), std::max(bmax.y, items[i].bmax.y),
                std::max(bmax.z, items[i].bmax.z)};
        clo = {std::min(clo.x, items[i].centroid.x), std::min(clo.y, items[i].centroid.y),
               std::min(clo.z, items[i].centroid.z)};
        chi = {std::max(chi.x, items[i].centroid.x), std::max(chi.y, items[i].centroid.y),
               std::max(chi.z, items[i].centroid.z)};
    }
    bvh.nodes[node_id].bmin = bmin;
    bvh.nodes[node_id].bmax = bmax;
    if (end - begin <= kMaxLeafTris) {
        bvh.nodes[node_id].start = uint32_t(bvh.order.size());
        bvh.nodes[node_id].count = end - begin;
        for (uint32_t i = begin; i < end; ++i) bvh.order.push_back(items[i].tri);
        return node_id;
    }
    const int axis = longest_axis(clo, chi);
    const uint32_t mid = begin + (end - begin) / 2;
    std::nth_element(items.begin() + begin, items.begin() + mid, items.begin() + end,
                     [axis](const BuildItem& a, const BuildItem& b) {
                         return axis_of(a.centroid, axis) < axis_of(b.centroid, axis);
                     });
    bvh.nodes[node_id].left = int(build_node(bvh, items, begin, mid));
    bvh.nodes[node_id].right = int(build_node(bvh, items, mid, end));
    return node_id;
}

bool slab_hit(const BvhNode& node, const geom::Vec3d& o, const geom::Vec3d& d, double t_min,
              double& entry) {
    double t0 = t_min;
    double t1 = std::numeric_limits<double>::infinity();
    const double oo[3] = {o.x, o.y, o.z};
    const double dd[3] = {d.x, d.y, d.z};
    const double lo[3] = {node.bmin.x, node.bmin.y, node.bmin.z};
    const double hi[3] = {node.bmax.x, node.bmax.y, node.bmax.z};
    for (int i = 0; i < 3; ++i) {
        if (dd[i] == 0.0) {
            if (oo[i] < lo[i] || oo[i] > hi[i]) return false;
        } else {
            const double inv = 1.0 / dd[i];
            double a = (lo[i] - oo[i]) * inv;
            double b = (hi[i] - oo[i]) * inv;
            if (a > b) std::swap(a, b);
            t0 = std::max(t0, a);
            t1 = std::min(t1, b);
            if (t0 > t1) return false;
        }
    }
    entry = t0;
    return true;
}

} // namespace

Bvh build_bvh(const NormalizedMesh& mesh) {
    Bvh bvh;
    if (mesh.triangles.empty()) return bvh;
    std::vector<BuildItem> items;
    items.reserve(mesh.triangles.size());
    for (uint32_t t = 0; t < mesh.triangles.size(); ++t) items.push_back(make_item(mesh, t));
    bvh.nodes.reserve(2 * items.size());
    bvh.order.reserve(items.size());
    build_node(bvh, items, 0, uint32_t(items.size()));
    return bvh;
}

std::optional<RayHit> ray_bvh(const Bvh& bvh, const NormalizedMesh& mesh,
                               const geom::Vec3d& origin, const geom::Vec3d& dir, double t_min,
                               uint32_t skip_tri) {
    if (!is_unit(dir)) throw std::invalid_argument("ray direction must be unit");
    if (!(t_min >= 0.0) || !std::isfinite(t_min))
        throw std::invalid_argument("t_min must be non-negative and finite");
    if (bvh.nodes.empty()) return std::nullopt;
    std::optional<RayHit> best;
    std::array<int, 256> stack;
    size_t sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const BvhNode& node = bvh.nodes[stack[--sp]];
        double entry = 0.0;
        if (!slab_hit(node, origin, dir, t_min, entry)) continue;
        if (best && entry > best->t) continue;
        if (node.count > 0) {
            for (uint32_t k = 0; k < node.count; ++k) {
                const uint32_t t = bvh.order[node.start + k];
                if (t == skip_tri) continue;
                const auto& tri = mesh.triangles[t];
                const auto hit =
                    ray_triangle(origin, dir, mesh.vertices[tri[0]], mesh.vertices[tri[1]],
                                 mesh.vertices[tri[2]], t_min, t);
                if (hit && (!best || hit->t < best->t ||
                            (hit->t == best->t && hit->tri < best->tri)))
                    best = hit;
            }
        } else {
            if (sp + 2 > stack.size()) throw std::logic_error("bvh traversal overflow");
            stack[sp++] = node.left;
            stack[sp++] = node.right;
        }
    }
    return best;
}

bool occluded_bvh(const Bvh& bvh, const NormalizedMesh& mesh, const geom::Vec3d& p,
                  const geom::Vec3d& s, double max_t, uint32_t self_tri) {
    if (!is_unit(s)) throw std::invalid_argument("occlusion direction must be unit");
    if (!(max_t > 0.0) || !std::isfinite(max_t))
        throw std::invalid_argument("max_t must be positive and finite");
    if (bvh.nodes.empty()) return false;
    const geom::Vec3d diag = mesh.report.bbox_max - mesh.report.bbox_min;
    const double t_min = 1e-9 * diag.length();
    std::array<int, 256> stack;
    size_t sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const BvhNode& node = bvh.nodes[stack[--sp]];
        double entry = 0.0;
        if (!slab_hit(node, p, s, t_min, entry)) continue;
        if (!(entry < max_t)) continue;
        if (node.count > 0) {
            for (uint32_t k = 0; k < node.count; ++k) {
                const uint32_t t = bvh.order[node.start + k];
                if (t == self_tri) continue;
                const auto& tri = mesh.triangles[t];
                const auto hit = ray_triangle(p, s, mesh.vertices[tri[0]], mesh.vertices[tri[1]],
                                              mesh.vertices[tri[2]], t_min, t);
                if (hit && hit->t < max_t) return true;
            }
        } else {
            if (sp + 2 > stack.size()) throw std::logic_error("bvh traversal overflow");
            stack[sp++] = node.left;
            stack[sp++] = node.right;
        }
    }
    return false;
}

}

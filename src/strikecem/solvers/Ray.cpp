#include "strikecem/solvers/Ray.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
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

geom::Vec3d reflect_direction(const geom::Vec3d& d, const geom::Vec3d& n) {
    const double p = geom::dot(d, n);
    return geom::Vec3d(d.x - 2.0 * p * n.x, d.y - 2.0 * p * n.y, d.z - 2.0 * p * n.z);
}

std::array<std::complex<double>, 3> reflect_electric_field(
    const std::array<std::complex<double>, 3>& e, const geom::Vec3d& n) {
    const std::complex<double> dot = e[0] * n.x + e[1] * n.y + e[2] * n.z;
    return {2.0 * dot * n.x - e[0], 2.0 * dot * n.y - e[1], 2.0 * dot * n.z - e[2]};
}

PolarizationDecomposition decompose_polarization(
    const std::array<std::complex<double>, 3>& e_in,
    const geom::Vec3d& dir,
    const geom::Vec3d& normal) {
    PolarizationDecomposition result;
    geom::Vec3d te = geom::cross(dir, normal);
    const double te_len = te.length();
    if (te_len > 1e-12) {
        te = te * (1.0 / te_len);
    } else {
        geom::Vec3d arb = (std::abs(dir.z) < 0.9) ? geom::Vec3d(0, 0, 1) : geom::Vec3d(1, 0, 0);
        te = geom::cross(dir, arb);
        te.normalize();
    }
    geom::Vec3d tm = geom::cross(te, dir);
    tm.normalize();

    result.te_hat = te;
    result.tm_hat = tm;
    result.e_te = e_in[0] * te.x + e_in[1] * te.y + e_in[2] * te.z;
    result.e_tm = e_in[0] * tm.x + e_in[1] * tm.y + e_in[2] * tm.z;
    return result;
}

SbrRayPath trace_sbr_path(
    const Bvh& bvh, const NormalizedMesh& mesh,
    uint32_t first_triangle,
    const geom::Vec3d& initial_k,
    const std::array<std::complex<double>, 3>& initial_e,
    const geom::Vec3d& obs_r,
    double k_wave,
    const geom::Vec3d& phase_ref,
    int max_bounces) {
    SbrRayPath path;
    if (first_triangle >= mesh.triangles.size() || max_bounces < 2) return path;

    const geom::Vec3d diag = mesh.report.bbox_max - mesh.report.bbox_min;
    const double scene = 2.0 * diag.length();
    const double tmin = 1e-9 * diag.length();

    const uint32_t ia = first_triangle;
    const geom::Vec3d na = mesh.normals[ia];
    if (!(geom::dot(na, obs_r) > 0.0)) return path;

    const auto& ta = mesh.triangles[ia];
    const geom::Vec3d ca =
        (mesh.vertices[ta[0]] + mesh.vertices[ta[1]] + mesh.vertices[ta[2]]) * (1.0 / 3.0);
    const geom::Vec3d d1 = reflect_direction(initial_k, na);
    const auto e1 = reflect_electric_field(initial_e, na);

    // Hop 0
    path.hops.push_back({ia, ca, na, initial_k, d1, 0.0,
                         std::exp(std::complex<double>(0.0, -k_wave * geom::dot(initial_k, ca - phase_ref))),
                         e1});

    // Hop 1 (Second bounce)
    const auto hit1 = ray_bvh(bvh, mesh, ca, d1, tmin, ia);
    if (!hit1) return path;
    const uint32_t ib = hit1->tri;
    if (ib == ia) return path;
    const geom::Vec3d nb = mesh.normals[ib];
    if (!(geom::dot(nb, d1) < 0.0)) return path;

    const auto& tb = mesh.triangles[ib];
    const geom::Vec3d cb =
        (mesh.vertices[tb[0]] + mesh.vertices[tb[1]] + mesh.vertices[tb[2]]) * (1.0 / 3.0);
    const geom::Vec3d d2 = reflect_direction(d1, nb);
    const auto e2 = reflect_electric_field(e1, nb);
    const double seg1 = (cb - ca).length();
    const auto phase1 = std::exp(std::complex<double>(0.0, -k_wave * geom::dot(d1, cb - ca)));
    path.hops.push_back({ib, cb, nb, d1, d2, seg1, phase1, e2});

    if (geom::dot(d2, obs_r) > 1.0 - 1e-9) {
        const auto exit = ray_bvh(bvh, mesh, cb, obs_r, tmin, ib);
        path.exit_occluded = (exit && exit->t < scene);
        path.is_retroreflective = !path.exit_occluded;
        if (path.is_retroreflective) {
            path.total_phase = path.hops[0].phase_factor * phase1 *
                std::exp(std::complex<double>(0.0, k_wave * geom::dot(obs_r, cb - phase_ref)));
            return path;
        }
    }

    // Hop 2 (Third bounce)
    if (max_bounces >= 3) {
        const auto hit2 = ray_bvh(bvh, mesh, cb, d2, tmin, ib);
        if (!hit2) return path;
        const uint32_t ic = hit2->tri;
        if (ic == ib) return path;
        const geom::Vec3d nc = mesh.normals[ic];
        if (!(geom::dot(nc, d2) < 0.0)) return path;

        const auto& tc = mesh.triangles[ic];
        const geom::Vec3d cc =
            (mesh.vertices[tc[0]] + mesh.vertices[tc[1]] + mesh.vertices[tc[2]]) * (1.0 / 3.0);
        const geom::Vec3d d3 = reflect_direction(d2, nc);
        const auto e3 = reflect_electric_field(e2, nc);
        const double seg2 = (cc - cb).length();
        const auto phase2 = std::exp(std::complex<double>(0.0, -k_wave * geom::dot(d2, cc - cb)));
        path.hops.push_back({ic, cc, nc, d2, d3, seg2, phase2, e3});

        if (geom::dot(d3, obs_r) > 1.0 - 1e-9) {
            const auto exit = ray_bvh(bvh, mesh, cc, obs_r, tmin, ic);
            path.exit_occluded = (exit && exit->t < scene);
            path.is_retroreflective = !path.exit_occluded;
            if (path.is_retroreflective) {
                path.total_phase = path.hops[0].phase_factor * phase1 * phase2 *
                    std::exp(std::complex<double>(0.0, k_wave * geom::dot(obs_r, cc - phase_ref)));
            }
        }
    }
    return path;
}

}

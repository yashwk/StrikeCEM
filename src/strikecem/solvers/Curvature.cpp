#include "strikecem/solvers/Curvature.hpp"

#include <cmath>
#include <map>
#include <algorithm>
#include <numbers>

namespace strikecem {
namespace {

struct EdgePair {
    uint32_t v0, v1;
    bool operator<(const EdgePair& o) const {
        return v0 != o.v0 ? v0 < o.v0 : v1 < o.v1;
    }
};

EdgePair make_edge(uint32_t a, uint32_t b) {
    return a < b ? EdgePair{a, b} : EdgePair{b, a};
}

} // namespace

std::vector<CurvatureInfo> estimate_mesh_curvatures(const NormalizedMesh& mesh,
                                                    double max_smooth_angle_deg) {
    const size_t ntris = mesh.triangles.size();
    std::vector<CurvatureInfo> result(ntris);
    if (ntris == 0) return result;

    const double min_cos_smooth = std::cos(max_smooth_angle_deg * std::numbers::pi / 180.0);

    // Build edge to adjacent triangle map
    std::map<EdgePair, std::vector<uint32_t>> edge_to_tris;
    std::vector<geom::Vec3d> centroids(ntris);
    for (size_t t = 0; t < ntris; ++t) {
        const auto& tri = mesh.triangles[t];
        centroids[t] = (mesh.vertices[tri[0]] + mesh.vertices[tri[1]] + mesh.vertices[tri[2]]) * (1.0 / 3.0);
        edge_to_tris[make_edge(tri[0], tri[1])].push_back(static_cast<uint32_t>(t));
        edge_to_tris[make_edge(tri[1], tri[2])].push_back(static_cast<uint32_t>(t));
        edge_to_tris[make_edge(tri[2], tri[0])].push_back(static_cast<uint32_t>(t));
    }

    for (size_t t = 0; t < ntris; ++t) {
        const auto& tri = mesh.triangles[t];
        const uint32_t edges[3][2] = {
            {tri[0], tri[1]},
            {tri[1], tri[2]},
            {tri[2], tri[0]}
        };

        double sum_kappa = 0.0;
        double max_k = 0.0;
        int smooth_count = 0;
        bool has_discontinuity = false;

        for (int e = 0; e < 3; ++e) {
            const auto it = edge_to_tris.find(make_edge(edges[e][0], edges[e][1]));
            if (it == edge_to_tris.end() || it->second.size() < 2) {
                // Open boundary edge
                has_discontinuity = true;
                continue;
            }

            for (uint32_t adj : it->second) {
                if (adj == t) continue;
                const double cos_dihedral = geom::dot(mesh.normals[t], mesh.normals[adj]);
                if (cos_dihedral < min_cos_smooth) {
                    // Sharp edge / crease
                    has_discontinuity = true;
                    continue;
                }

                const geom::Vec3d dr = centroids[adj] - centroids[t];
                const geom::Vec3d e_vec = mesh.vertices[edges[e][1]] - mesh.vertices[edges[e][0]];
                const double e_len = e_vec.length();
                if (e_len > 1e-12) {
                    geom::Vec3d u = geom::cross(e_vec * (1.0 / e_len), mesh.normals[t]);
                    const double u_len = u.length();
                    if (u_len > 1e-12) {
                        u = u * (1.0 / u_len);
                        if (geom::dot(u, dr) < 0.0) {
                            u = u * -1.0;
                        }
                        const double d_perp = geom::dot(dr, u);
                        if (d_perp > 1e-6) {
                            const geom::Vec3d dn = mesh.normals[adj] - mesh.normals[t];
                            const double kappa = geom::dot(dn, u) / d_perp;
                            sum_kappa += kappa;
                            max_k = std::max(max_k, std::abs(kappa));
                            ++smooth_count;
                        }
                    }
                }
            }
        }

        result[t].is_discontinuous = has_discontinuity;
        if (smooth_count > 0) {
            result[t].mean_curvature = sum_kappa / static_cast<double>(smooth_count);
            result[t].max_curvature = max_k;
        } else {
            result[t].mean_curvature = 0.0;
            result[t].max_curvature = 0.0;
        }
    }

    return result;
}

std::complex<double> po_curvature_factor(double mean_curvature, double k, double cos_inc) {
    if (std::abs(mean_curvature) < 1e-9 || k <= 0.0) {
        return {1.0, 0.0};
    }
    const double cos_clamped = std::max(0.1, std::abs(cos_inc));
    // Asymptotic Fock correction: 1 - j * H / (k * cos_inc)
    const double imag_part = -mean_curvature / (k * cos_clamped);
    return {1.0, imag_part};
}

} // namespace strikecem

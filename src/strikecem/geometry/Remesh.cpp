#include "strikecem/geometry/Remesh.hpp"

#include <cmath>
#include <map>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <openssl/sha.h>

#include "strikecem/core/Conventions.hpp"

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

std::string compute_sha256(const std::string& input) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(input.data()), input.size(), hash);
    std::ostringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return ss.str();
}

} // namespace

double compute_wavelength_m(double freq_hz, double eps_r, double mu_r) {
    if (freq_hz <= 0.0) throw std::invalid_argument("frequency must be positive");
    if (eps_r <= 0.0 || mu_r <= 0.0) throw std::invalid_argument("permittivity and permeability must be positive");
    return kSpeedOfLight / (freq_hz * std::sqrt(eps_r * mu_r));
}

NormalizedMesh remesh_target_edge_length(const NormalizedMesh& mesh, double h_target,
                                         size_t max_depth) {
    if (h_target <= 0.0) throw std::invalid_argument("h_target must be positive");
    if (mesh.triangles.empty()) return mesh;

    NormalizedMesh cur = mesh;

    for (size_t depth = 0; depth < max_depth; ++depth) {
        // Step 1: Detect edges that exceed h_target
        std::map<EdgePair, uint32_t> edge_midpoints;
        for (const auto& tri : cur.triangles) {
            const uint32_t idx[3] = {tri[0], tri[1], tri[2]};
            for (int e = 0; e < 3; ++e) {
                const uint32_t a = idx[e];
                const uint32_t b = idx[(e + 1) % 3];
                const double len = (cur.vertices[a] - cur.vertices[b]).length();
                if (len > h_target) {
                    edge_midpoints[make_edge(a, b)] = 0; // mark for splitting
                }
            }
        }

        if (edge_midpoints.empty()) break; // all edges satisfy target resolution

        // Step 2: Create midpoint vertices
        for (auto& [edge, mid_idx] : edge_midpoints) {
            mid_idx = static_cast<uint32_t>(cur.vertices.size());
            geom::Vec3d mid = (cur.vertices[edge.v0] + cur.vertices[edge.v1]) * 0.5;
            cur.vertices.push_back(mid);
        }

        // Step 3: Conforming triangle subdivision
        std::vector<std::array<uint32_t, 3>> next_triangles;
        std::vector<geom::Vec3d> next_normals;
        std::vector<double> next_areas;
        std::vector<std::string> next_tags;

        for (size_t t = 0; t < cur.triangles.size(); ++t) {
            const auto& tri = cur.triangles[t];
            const std::string tag = (t < cur.triangle_tags.size()) ? cur.triangle_tags[t] : "";

            const uint32_t v0 = tri[0];
            const uint32_t v1 = tri[1];
            const uint32_t v2 = tri[2];

            auto it01 = edge_midpoints.find(make_edge(v0, v1));
            auto it12 = edge_midpoints.find(make_edge(v1, v2));
            auto it20 = edge_midpoints.find(make_edge(v2, v0));

            const bool s01 = (it01 != edge_midpoints.end());
            const bool s12 = (it12 != edge_midpoints.end());
            const bool s20 = (it20 != edge_midpoints.end());

            const uint32_t m01 = s01 ? it01->second : 0;
            const uint32_t m12 = s12 ? it12->second : 0;
            const uint32_t m20 = s20 ? it20->second : 0;

            const int split_count = (s01 ? 1 : 0) + (s12 ? 1 : 0) + (s20 ? 1 : 0);

            if (split_count == 3) {
                // 1-to-4 red subdivision
                next_triangles.push_back({v0, m01, m20});
                next_triangles.push_back({m01, v1, m12});
                next_triangles.push_back({m20, m12, v2});
                next_triangles.push_back({m01, m12, m20});
                for (int k = 0; k < 4; ++k) next_tags.push_back(tag);
            } else if (split_count == 2) {
                // 1-to-3 green bisection
                if (s01 && s12) {
                    next_triangles.push_back({v0, m01, v2});
                    next_triangles.push_back({m01, m12, v2});
                    next_triangles.push_back({m01, v1, m12});
                } else if (s12 && s20) {
                    next_triangles.push_back({v1, m12, v0});
                    next_triangles.push_back({m12, m20, v0});
                    next_triangles.push_back({m12, v2, m20});
                } else { // s20 && s01
                    next_triangles.push_back({v2, m20, v1});
                    next_triangles.push_back({m20, m01, v1});
                    next_triangles.push_back({m20, v0, m01});
                }
                for (int k = 0; k < 3; ++k) next_tags.push_back(tag);
            } else if (split_count == 1) {
                // 1-to-2 green bisection
                if (s01) {
                    next_triangles.push_back({v0, m01, v2});
                    next_triangles.push_back({m01, v1, v2});
                } else if (s12) {
                    next_triangles.push_back({v1, m12, v0});
                    next_triangles.push_back({m12, v2, v0});
                } else { // s20
                    next_triangles.push_back({v2, m20, v1});
                    next_triangles.push_back({m20, v0, v1});
                }
                for (int k = 0; k < 2; ++k) next_tags.push_back(tag);
            } else {
                // 0 splits
                next_triangles.push_back(tri);
                next_tags.push_back(tag);
            }
        }

        cur.triangles = std::move(next_triangles);
        cur.triangle_tags = std::move(next_tags);
    }

    // Step 4: Recompute geometry metrics
    cur.normals.resize(cur.triangles.size());
    cur.areas.resize(cur.triangles.size());
    double total_area = 0.0;
    double max_edge = 0.0;

    for (size_t t = 0; t < cur.triangles.size(); ++t) {
        const auto& tri = cur.triangles[t];
        const auto& p0 = cur.vertices[tri[0]];
        const auto& p1 = cur.vertices[tri[1]];
        const auto& p2 = cur.vertices[tri[2]];

        const geom::Vec3d e0 = p1 - p0;
        const geom::Vec3d e1 = p2 - p0;
        const geom::Vec3d e2 = p2 - p1;

        max_edge = std::max({max_edge, e0.length(), e1.length(), e2.length()});

        geom::Vec3d n = geom::cross(e0, e1);
        const double n_len = n.length();
        const double a = 0.5 * n_len;
        if (n_len > 1e-12) {
            n.normalize();
        } else {
            n = {0.0, 0.0, 1.0};
        }
        cur.normals[t] = n;
        cur.areas[t] = a;
        total_area += a;
    }

    cur.report.vertex_count = cur.vertices.size();
    cur.report.triangle_count = cur.triangles.size();
    cur.report.total_area_m2 = total_area;
    cur.report.max_edge_length_m = max_edge;

    if (!cur.vertices.empty()) {
        cur.report.bbox_min = cur.vertices.front();
        cur.report.bbox_max = cur.vertices.front();
        for (const auto& v : cur.vertices) {
            cur.report.bbox_min.x = std::min(cur.report.bbox_min.x, v.x);
            cur.report.bbox_min.y = std::min(cur.report.bbox_min.y, v.y);
            cur.report.bbox_min.z = std::min(cur.report.bbox_min.z, v.z);
            cur.report.bbox_max.x = std::max(cur.report.bbox_max.x, v.x);
            cur.report.bbox_max.y = std::max(cur.report.bbox_max.y, v.y);
            cur.report.bbox_max.z = std::max(cur.report.bbox_max.z, v.z);
        }
    }

    std::ostringstream bytes;
    for (const auto& v : cur.vertices) bytes << "v" << v.x << "," << v.y << "," << v.z << ";";
    for (const auto& t : cur.triangles) bytes << "t" << t[0] << "," << t[1] << "," << t[2] << ";";
    cur.geometry_hash = compute_sha256(bytes.str());
    cur.normalized_mesh_hash = cur.geometry_hash;

    return cur;
}

NormalizedMesh remesh_wavelength(const NormalizedMesh& mesh, const nlohmann::json& resolved,
                                 const std::vector<double>& frequencies_hz) {
    std::string policy = "user_mesh";
    double alpha_smooth = 1.0 / 6.0;
    double alpha_edge = 1.0 / 20.0;

    if (resolved.contains("mesh")) {
        policy = resolved["mesh"].value("frequency_policy", "user_mesh");
        if (resolved["mesh"].contains("wavelength_adaptive")) {
            const auto& wa = resolved["mesh"]["wavelength_adaptive"];
            alpha_smooth = wa.value("alpha_smooth", 1.0 / 6.0);
            alpha_edge = wa.value("alpha_edge", 1.0 / 20.0);
        }
    }

    if (policy == "user_mesh" || frequencies_hz.empty()) {
        return mesh;
    }

    double max_f = *std::max_element(frequencies_hz.begin(), frequencies_hz.end());
    double eps_r = 1.0;
    double mu_r = 1.0;
    if (resolved.contains("frequency") && resolved["frequency"].contains("medium")) {
        const auto& med = resolved["frequency"]["medium"];
        eps_r = med.value("epsilon_r", 1.0);
        mu_r = med.value("mu_r", 1.0);
    }

    double lambda = compute_wavelength_m(max_f, eps_r, mu_r);
    double h_target = alpha_smooth * lambda;

    return remesh_target_edge_length(mesh, h_target);
}

} // namespace strikecem

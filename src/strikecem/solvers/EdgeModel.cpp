// Straight-edge diffraction model (ADR-0003).
#include "strikecem/solvers/EdgeModel.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace strikecem {
namespace {

struct EdgeKey {
    uint32_t v0, v1; // sorted vertex indices
    bool operator<(const EdgeKey& o) const {
        return v0 != o.v0 ? v0 < o.v0 : v1 < o.v1;
    }
};

} // namespace

EdgeModel extract_edges(const NormalizedMesh& mesh, double dihedral_threshold_rad) {
    EdgeModel model;
    std::map<EdgeKey, std::vector<std::pair<uint32_t, bool>>> uses; // tri, forward
    for (uint32_t t = 0; t < mesh.triangles.size(); ++t) {
        const auto& tri = mesh.triangles[t];
        for (int e = 0; e < 3; ++e) {
            const uint32_t a = tri[e], b = tri[(e + 1) % 3];
            uses[{std::min(a, b), std::max(a, b)}].push_back({t, a < b});
        }
    }
    auto push_edge = [&](uint32_t va, uint32_t vb, const geom::Vec3d& n0,
                         const geom::Vec3d& n1, int32_t t0, int32_t t1, bool boundary,
                         double wedge_n, bool convex) {
        MeshEdge edge;
        edge.p0 = mesh.vertices[va];
        edge.p1 = mesh.vertices[vb];
        const geom::Vec3d span = edge.p1 - edge.p0;
        edge.length = span.length();
        if (!(edge.length > 0.0)) throw std::invalid_argument("zero-length mesh edge");
        edge.tangent = span * (1.0 / edge.length);
        // Interior directions of the adjacent faces, projected
        // perpendicular to the edge: the screen reference for rims
        // (face1 duplicates face0 there), the wedge-frame rays for
        // interior creases.
        const geom::Vec3d m = (mesh.vertices[va] + mesh.vertices[vb]) * 0.5;
        auto inward_of = [&](int32_t t) {
            const auto& ft = mesh.triangles[static_cast<uint32_t>(t)];
            const geom::Vec3d c =
                (mesh.vertices[ft[0]] + mesh.vertices[ft[1]] + mesh.vertices[ft[2]]) *
                (1.0 / 3.0);
            geom::Vec3d inward = (c - m) - edge.tangent * geom::dot(c - m, edge.tangent);
            const double len = inward.length();
            if (!(len > 1e-15 * edge.length))
                throw std::invalid_argument("degenerate edge-adjacent face");
            return inward * (1.0 / len);
        };
        edge.face0_dir = inward_of(t0);
        edge.face1_dir = (t1 < 0) ? edge.face0_dir : inward_of(t1);
        edge.n0 = n0;
        edge.n1 = n1;
        edge.tri0 = t0;
        edge.tri1 = t1;
        edge.boundary = boundary;
        edge.wedge_n = wedge_n;
        edge.convex = convex;
        model.edges.push_back(edge);
    };
    for (const auto& [key, tris] : uses) {
        if (tris.size() == 1) {
            const geom::Vec3d n = mesh.normals[tris[0].first];
            push_edge(key.v0, key.v1, n, n, tris[0].first, -1, true, 2.0, true);
        } else if (tris.size() == 2) {
            const geom::Vec3d& n0 = mesh.normals[tris[0].first];
            const geom::Vec3d& n1 = mesh.normals[tris[1].first];
            const double cos_a =
                std::max(-1.0, std::min(1.0, geom::dot(n0, n1)));
            const double deviation = std::acos(cos_a);
            if (deviation < dihedral_threshold_rad) {
                ++model.smooth_skipped;
                continue;
            }
            // Exterior wedge parameter: half-plane rim n = 2.
            const double wedge_n = 2.0 - deviation / 3.141592653589793;
            // Convexity from the material-side test: w points from the edge
            // midpoint toward the adjacent triangle centroids; b is the
            // exterior bisector. dot < 0 (beyond fp tolerance) = convex
            // ridge; a symmetric right-angle valley tests exactly zero and
            // classifies concave, which is the correct side.
            const auto& t0 = mesh.triangles[tris[0].first];
            const auto& t1 = mesh.triangles[tris[1].first];
            const geom::Vec3d c0 =
                (mesh.vertices[t0[0]] + mesh.vertices[t0[1]] + mesh.vertices[t0[2]]) *
                (1.0 / 3.0);
            const geom::Vec3d c1 =
                (mesh.vertices[t1[0]] + mesh.vertices[t1[1]] + mesh.vertices[t1[2]]) *
                (1.0 / 3.0);
            const geom::Vec3d m = (mesh.vertices[key.v0] + mesh.vertices[key.v1]) * 0.5;
            const geom::Vec3d w = (c0 - m) + (c1 - m);
            const geom::Vec3d b = n0 + n1;
            const double side = geom::dot(w, b);
            const double scale = w.length() * b.length();
            const bool convex = side < -1e-12 * (scale > 0.0 ? scale : 1.0);
            push_edge(key.v0, key.v1, n0, n1, tris[0].first, tris[1].first, false, wedge_n,
                      convex);
        } else {
            throw std::invalid_argument("non-manifold edge in edge model input");
        }
    }
    return model;
}

} // namespace strikecem

#include "strikecem/geometry/Symmetry.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>

namespace strikecem {

geom::Vec3d reflect_point(const geom::Vec3d& p, const std::string& plane) {
    if (plane == "xz") {
        return {p.x, -p.y, p.z};
    } else if (plane == "yz") {
        return {-p.x, p.y, p.z};
    } else if (plane == "xy") {
        return {p.x, p.y, -p.z};
    }
    throw std::invalid_argument("unknown symmetry plane: " + plane + " (valid: xz, yz, xy)");
}

geom::Vec3d reflect_normal(const geom::Vec3d& n, const std::string& plane) {
    return reflect_point(n, plane);
}

double get_symmetry_coord(const geom::Vec3d& p, const std::string& plane) {
    if (plane == "xz") return p.y;
    if (plane == "yz") return p.x;
    if (plane == "xy") return p.z;
    throw std::invalid_argument("unknown symmetry plane: " + plane);
}

namespace {

struct CellKey {
    int64_t x, y, z;
    bool operator==(const CellKey& o) const {
        return x == o.x && y == o.y && z == o.z;
    }
};

struct CellHash {
    size_t operator()(const CellKey& k) const {
        size_t h = std::hash<int64_t>()(k.x);
        h ^= std::hash<int64_t>()(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>()(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

class CentroidGrid {
public:
    explicit CentroidGrid(double cell_size) : cell_size_(std::max(cell_size, 1e-6)) {}

    void insert(uint32_t idx, const geom::Vec3d& pt) {
        grid_[to_key(pt)].push_back(idx);
    }

    uint32_t find_matching(const geom::Vec3d& target, const geom::Vec3d& expected_normal,
                          const std::vector<geom::Vec3d>& normals,
                          const std::vector<geom::Vec3d>& centroids,
                          double tol, uint32_t self_idx) const {
        const int64_t cx = static_cast<int64_t>(std::floor(target.x / cell_size_));
        const int64_t cy = static_cast<int64_t>(std::floor(target.y / cell_size_));
        const int64_t cz = static_cast<int64_t>(std::floor(target.z / cell_size_));

        for (int64_t dx = -1; dx <= 1; ++dx) {
            for (int64_t dy = -1; dy <= 1; ++dy) {
                for (int64_t dz = -1; dz <= 1; ++dz) {
                    auto it = grid_.find({cx + dx, cy + dy, cz + dz});
                    if (it == grid_.end()) continue;
                    for (uint32_t cand_idx : it->second) {
                        if (cand_idx == self_idx) continue;
                        if ((centroids[cand_idx] - target).length() <= tol) {
                            if ((normals[cand_idx] - expected_normal).length() <= 0.1) {
                                return cand_idx;
                            }
                        }
                    }
                }
            }
        }
        return UINT32_MAX;
    }

private:
    double cell_size_;
    std::unordered_map<CellKey, std::vector<uint32_t>, CellHash> grid_;

    CellKey to_key(const geom::Vec3d& pt) const {
        return {static_cast<int64_t>(std::floor(pt.x / cell_size_)),
                static_cast<int64_t>(std::floor(pt.y / cell_size_)),
                static_cast<int64_t>(std::floor(pt.z / cell_size_))};
    }
};

} // namespace

bool is_symmetric_mesh(const NormalizedMesh& mesh, const SymmetryOptions& opts) {
    if (mesh.triangles.empty()) return true;
    const double tol = opts.tolerance_m;
    const size_t ntris = mesh.triangles.size();

    std::vector<geom::Vec3d> centroids(ntris);
    CentroidGrid grid(tol * 2.0);
    for (size_t i = 0; i < ntris; ++i) {
        const auto& tri = mesh.triangles[i];
        centroids[i] = (mesh.vertices[tri[0]] + mesh.vertices[tri[1]] + mesh.vertices[tri[2]]) * (1.0 / 3.0);
        grid.insert(static_cast<uint32_t>(i), centroids[i]);
    }

    for (size_t i = 0; i < ntris; ++i) {
        const double coord = get_symmetry_coord(centroids[i], opts.plane);
        if (std::abs(coord) <= tol) continue; // on symmetry boundary

        const geom::Vec3d c_refl = reflect_point(centroids[i], opts.plane);
        const geom::Vec3d n_refl = reflect_normal(mesh.normals[i], opts.plane);

        uint32_t match = grid.find_matching(c_refl, n_refl, mesh.normals, centroids, tol, static_cast<uint32_t>(i));
        if (match == UINT32_MAX) return false;
    }
    return true;
}

HalfMeshResult extract_half_mesh(const NormalizedMesh& mesh, const SymmetryOptions& opts) {
    HalfMeshResult res;
    NormalizedMesh& half = res.mesh;
    half.geometry_hash = mesh.geometry_hash + "_half";
    half.normalized_mesh_hash = mesh.normalized_mesh_hash + "_half";
    const double tol = opts.tolerance_m;
    const size_t ntris = mesh.triangles.size();

    std::vector<geom::Vec3d> centroids(ntris);
    CentroidGrid grid(tol * 2.0);
    for (size_t i = 0; i < ntris; ++i) {
        const auto& tri = mesh.triangles[i];
        centroids[i] = (mesh.vertices[tri[0]] + mesh.vertices[tri[1]] + mesh.vertices[tri[2]]) * (1.0 / 3.0);
        grid.insert(static_cast<uint32_t>(i), centroids[i]);
    }

    std::unordered_map<uint32_t, uint32_t> old_to_new_vertex;
    auto add_vertex = [&](uint32_t old_idx) -> uint32_t {
        auto it = old_to_new_vertex.find(old_idx);
        if (it != old_to_new_vertex.end()) return it->second;
        uint32_t new_idx = static_cast<uint32_t>(half.vertices.size());
        half.vertices.push_back(mesh.vertices[old_idx]);
        old_to_new_vertex[old_idx] = new_idx;
        return new_idx;
    };

    double total_area = 0.0;
    for (size_t i = 0; i < ntris; ++i) {
        const auto& tri = mesh.triangles[i];
        const double coord = get_symmetry_coord(centroids[i], opts.plane);

        // Keep facets strictly in positive half-space or lying on boundary
        if (coord < -tol) continue;

        uint32_t v0 = add_vertex(tri[0]);
        uint32_t v1 = add_vertex(tri[1]);
        uint32_t v2 = add_vertex(tri[2]);
        half.triangles.push_back({v0, v1, v2});
        half.normals.push_back(mesh.normals[i]);

        double a = mesh.areas[i];
        half.areas.push_back(a);
        total_area += a;

        if (i < mesh.triangle_tags.size()) {
            half.triangle_tags.push_back(mesh.triangle_tags[i]);
        }
        res.original_indices.push_back(static_cast<uint32_t>(i));

        if (std::abs(coord) <= tol) {
            res.reflected_indices.push_back(UINT32_MAX);
        } else {
            const geom::Vec3d c_refl = reflect_point(centroids[i], opts.plane);
            const geom::Vec3d n_refl = reflect_normal(mesh.normals[i], opts.plane);
            uint32_t match = grid.find_matching(c_refl, n_refl, mesh.normals, centroids, tol, static_cast<uint32_t>(i));
            res.reflected_indices.push_back(match);
        }
    }

    half.report.vertex_count = half.vertices.size();
    half.report.triangle_count = half.triangles.size();
    half.report.total_area_m2 = total_area;
    if (!half.vertices.empty()) {
        half.report.bbox_min = half.vertices.front();
        half.report.bbox_max = half.vertices.front();
        for (const auto& v : half.vertices) {
            half.report.bbox_min.x = std::min(half.report.bbox_min.x, v.x);
            half.report.bbox_min.y = std::min(half.report.bbox_min.y, v.y);
            half.report.bbox_min.z = std::min(half.report.bbox_min.z, v.z);
            half.report.bbox_max.x = std::max(half.report.bbox_max.x, v.x);
            half.report.bbox_max.y = std::max(half.report.bbox_max.y, v.y);
            half.report.bbox_max.z = std::max(half.report.bbox_max.z, v.z);
        }
    }
    return res;
}

} // namespace strikecem

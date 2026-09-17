#pragma once
// STL/OBJ triangle mesh loading, normalization, inspection, repair, cache.
// SPEC FR-2, IMPLEMENTATION.md section 6. Open meshes are legal (plates);
// degenerate triangles and non-manifold edges are fatal.
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/math/Geometry.hpp"

namespace strikecem {

struct RawTriangle {
    geom::Vec3d a, b, c;
};

// Parsed without units/transform applied; exposed for unit tests.
std::vector<RawTriangle> parse_stl(const std::filesystem::path& path);
std::vector<RawTriangle> parse_obj(const std::filesystem::path& path);

struct MeshReport {
    size_t vertex_count = 0;
    size_t triangle_count = 0;
    geom::Vec3d bbox_min;
    geom::Vec3d bbox_max;
    double total_area_m2 = 0.0;
    size_t degenerate_count = 0;
    size_t inconsistent_winding_count = 0;
    size_t open_edge_count = 0;
    size_t nonmanifold_edge_count = 0;
    double aspect_min = 0.0;
    double aspect_max = 0.0;
    double aspect_mean = 0.0;
    size_t aspect_over_limit_count = 0;
    double max_edge_length_m = 0.0;
};

struct NormalizedMesh {
    std::vector<geom::Vec3d> vertices; // metres, transformed
    std::vector<std::array<uint32_t, 3>> triangles;
    std::vector<geom::Vec3d> normals;
    std::vector<double> areas;
    MeshReport report;
    bool repaired = false;
    MeshReport report_before; // valid only when repaired
    bool cache_hit = false;
    std::string geometry_hash; // raw bytes + units/transform/repair/schema
    std::string normalized_mesh_hash;
};

struct MeshLoadError : public std::runtime_error {
    explicit MeshLoadError(const std::string& msg) : std::runtime_error(msg) {}
};

// Full pipeline: parse, convert units, transform, weld, inspect, repair per
// model.mesh_repair, then cache lookup/store. Throws MeshLoadError.
NormalizedMesh load_normalized_mesh(const nlohmann::json& resolved,
                                    const std::filesystem::path& config_dir,
                                    const std::string& schema_version);

} // namespace strikecem

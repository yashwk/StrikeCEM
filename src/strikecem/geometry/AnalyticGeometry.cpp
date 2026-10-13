#include "strikecem/geometry/AnalyticGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <set>
#include <sstream>
#include <stdexcept>

#include <openssl/sha.h>

namespace strikecem {
namespace {

std::string compute_sha256(const std::string& input) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(input.data()), input.size(), hash);
    std::ostringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return ss.str();
}

void build_orthonormal_basis(const geom::Vec3d& dir, geom::Vec3d& u, geom::Vec3d& v) {
    geom::Vec3d a = dir;
    a.normalize();
    geom::Vec3d up = (std::abs(a.z) < 0.9) ? geom::Vec3d{0.0, 0.0, 1.0} : geom::Vec3d{0.0, 1.0, 0.0};
    u = geom::cross(a, up);
    u.normalize();
    v = geom::cross(a, u);
    v.normalize();
}

void finalize_mesh(NormalizedMesh& mesh) {
    mesh.normals.clear();
    mesh.areas.clear();
    mesh.normals.reserve(mesh.triangles.size());
    mesh.areas.reserve(mesh.triangles.size());

    double total_area = 0.0;
    geom::Vec3d bb_min{1e30, 1e30, 1e30};
    geom::Vec3d bb_max{-1e30, -1e30, -1e30};

    for (const auto& v : mesh.vertices) {
        bb_min.x = std::min(bb_min.x, v.x);
        bb_min.y = std::min(bb_min.y, v.y);
        bb_min.z = std::min(bb_min.z, v.z);
        bb_max.x = std::max(bb_max.x, v.x);
        bb_max.y = std::max(bb_max.y, v.y);
        bb_max.z = std::max(bb_max.z, v.z);
    }

    double max_edge = 0.0;
    for (const auto& tri : mesh.triangles) {
        const auto& p0 = mesh.vertices[tri[0]];
        const auto& p1 = mesh.vertices[tri[1]];
        const auto& p2 = mesh.vertices[tri[2]];

        const geom::Vec3d e0 = p1 - p0;
        const geom::Vec3d e1 = p2 - p0;
        const geom::Vec3d e2 = p2 - p1;

        max_edge = std::max({max_edge, e0.length(), e1.length(), e2.length()});

        geom::Vec3d n = geom::cross(e0, e1);
        const double n_len = n.length();
        const double area = 0.5 * n_len;
        if (n_len > 1e-12) {
            n.normalize();
        } else {
            n = {0.0, 0.0, 1.0};
        }
        mesh.normals.push_back(n);
        mesh.areas.push_back(area);
        total_area += area;
    }

    mesh.report.vertex_count = mesh.vertices.size();
    mesh.report.triangle_count = mesh.triangles.size();
    mesh.report.bbox_min = bb_min;
    mesh.report.bbox_max = bb_max;
    mesh.report.total_area_m2 = total_area;
    mesh.report.max_edge_length_m = max_edge;

    std::ostringstream bytes;
    for (const auto& v : mesh.vertices) bytes << "v" << v.x << "," << v.y << "," << v.z << ";";
    for (const auto& t : mesh.triangles) bytes << "t" << t[0] << "," << t[1] << "," << t[2] << ";";
    for (const auto& tag : mesh.triangle_tags) bytes << "g" << tag << ";";
    mesh.geometry_hash = compute_sha256(bytes.str());
    mesh.normalized_mesh_hash = mesh.geometry_hash;
}

} // namespace

NormalizedMesh tessellate_sphere(const AnalyticSphere& sphere,
                                 const PrimitiveTessellationParams& params) {
    if (sphere.radius <= 0.0) throw std::invalid_argument("sphere radius must be positive");

    const double delta = params.max_edge_length;
    const size_t n_phi = std::max(params.min_radial_segments,
                                  static_cast<size_t>(std::ceil(2.0 * std::numbers::pi * sphere.radius / delta)));
    const size_t n_theta = std::max(params.min_axial_segments,
                                    static_cast<size_t>(std::ceil(std::numbers::pi * sphere.radius / delta)));

    NormalizedMesh mesh;
    mesh.groups.push_back(sphere.name);

    // North pole (index 0)
    mesh.vertices.push_back({sphere.center.x, sphere.center.y, sphere.center.z + sphere.radius});

    // Intermediate rings: j = 1 ... n_theta - 1
    for (size_t j = 1; j < n_theta; ++j) {
        const double theta = std::numbers::pi * static_cast<double>(j) / static_cast<double>(n_theta);
        const double z = sphere.radius * std::cos(theta);
        const double r = sphere.radius * std::sin(theta);
        for (size_t i = 0; i < n_phi; ++i) {
            const double phi = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(n_phi);
            const double x = r * std::cos(phi);
            const double y = r * std::sin(phi);
            mesh.vertices.push_back({sphere.center.x + x, sphere.center.y + y, sphere.center.z + z});
        }
    }

    // South pole
    const uint32_t south_pole_idx = static_cast<uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({sphere.center.x, sphere.center.y, sphere.center.z - sphere.radius});

    // North pole fan
    for (size_t i = 0; i < n_phi; ++i) {
        const uint32_t i0 = 1 + static_cast<uint32_t>(i);
        const uint32_t i1 = 1 + static_cast<uint32_t>((i + 1) % n_phi);
        mesh.triangles.push_back({0, i0, i1});
        mesh.triangle_tags.push_back(sphere.material_tag);
    }

    // Intermediate quads
    for (size_t j = 1; j < n_theta - 1; ++j) {
        const uint32_t ring0 = 1 + static_cast<uint32_t>((j - 1) * n_phi);
        const uint32_t ring1 = 1 + static_cast<uint32_t>(j * n_phi);
        for (size_t i = 0; i < n_phi; ++i) {
            const uint32_t next_i = static_cast<uint32_t>((i + 1) % n_phi);
            const uint32_t a = ring0 + static_cast<uint32_t>(i);
            const uint32_t b = ring0 + next_i;
            const uint32_t c = ring1 + next_i;
            const uint32_t d = ring1 + static_cast<uint32_t>(i);

            mesh.triangles.push_back({a, c, b});
            mesh.triangle_tags.push_back(sphere.material_tag);
            mesh.triangles.push_back({a, d, c});
            mesh.triangle_tags.push_back(sphere.material_tag);
        }
    }

    // South pole fan
    const uint32_t last_ring = 1 + static_cast<uint32_t>((n_theta - 2) * n_phi);
    for (size_t i = 0; i < n_phi; ++i) {
        const uint32_t i0 = last_ring + static_cast<uint32_t>(i);
        const uint32_t i1 = last_ring + static_cast<uint32_t>((i + 1) % n_phi);
        mesh.triangles.push_back({south_pole_idx, i1, i0});
        mesh.triangle_tags.push_back(sphere.material_tag);
    }

    finalize_mesh(mesh);
    return mesh;
}

NormalizedMesh tessellate_cylinder(const AnalyticCylinder& cylinder,
                                   const PrimitiveTessellationParams& params) {
    if (cylinder.radius <= 0.0 || cylinder.length <= 0.0) {
        throw std::invalid_argument("cylinder radius and length must be positive");
    }

    geom::Vec3d u, v;
    geom::Vec3d axis = cylinder.axis;
    axis.normalize();
    build_orthonormal_basis(axis, u, v);

    const double delta = params.max_edge_length;
    const size_t n_phi = std::max(params.min_radial_segments,
                                  static_cast<size_t>(std::ceil(2.0 * std::numbers::pi * cylinder.radius / delta)));
    const size_t n_z = std::max(params.min_axial_segments,
                                static_cast<size_t>(std::ceil(cylinder.length / delta)));

    NormalizedMesh mesh;
    mesh.groups.push_back(cylinder.name);

    for (size_t k = 0; k <= n_z; ++k) {
        const double s = static_cast<double>(k) / static_cast<double>(n_z) * cylinder.length;
        const geom::Vec3d center_k = cylinder.origin + axis * s;
        for (size_t i = 0; i < n_phi; ++i) {
            const double phi = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(n_phi);
            const geom::Vec3d p = center_k + (u * std::cos(phi) + v * std::sin(phi)) * cylinder.radius;
            mesh.vertices.push_back(p);
        }
    }

    // Cylinder lateral surface
    for (size_t k = 0; k < n_z; ++k) {
        const uint32_t ring0 = static_cast<uint32_t>(k * n_phi);
        const uint32_t ring1 = static_cast<uint32_t>((k + 1) * n_phi);
        for (size_t i = 0; i < n_phi; ++i) {
            const uint32_t next_i = static_cast<uint32_t>((i + 1) % n_phi);
            const uint32_t a = ring0 + static_cast<uint32_t>(i);
            const uint32_t b = ring0 + next_i;
            const uint32_t c = ring1 + next_i;
            const uint32_t d = ring1 + static_cast<uint32_t>(i);

            mesh.triangles.push_back({a, b, c});
            mesh.triangle_tags.push_back(cylinder.material_tag);
            mesh.triangles.push_back({a, c, d});
            mesh.triangle_tags.push_back(cylinder.material_tag);
        }
    }

    // Start cap
    if (cylinder.cap_start) {
        const uint32_t cap0_idx = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(cylinder.origin);
        for (size_t i = 0; i < n_phi; ++i) {
            const uint32_t i0 = static_cast<uint32_t>(i);
            const uint32_t i1 = static_cast<uint32_t>((i + 1) % n_phi);
            mesh.triangles.push_back({cap0_idx, i1, i0});
            mesh.triangle_tags.push_back(cylinder.material_tag);
        }
    }

    // End cap
    if (cylinder.cap_end) {
        const uint32_t cap1_idx = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(cylinder.origin + axis * cylinder.length);
        const uint32_t ring_end = static_cast<uint32_t>(n_z * n_phi);
        for (size_t i = 0; i < n_phi; ++i) {
            const uint32_t i0 = ring_end + static_cast<uint32_t>(i);
            const uint32_t i1 = ring_end + static_cast<uint32_t>((i + 1) % n_phi);
            mesh.triangles.push_back({cap1_idx, i0, i1});
            mesh.triangle_tags.push_back(cylinder.material_tag);
        }
    }

    finalize_mesh(mesh);
    return mesh;
}

NormalizedMesh tessellate_tangent_ogive(const AnalyticTangentOgive& ogive,
                                       const PrimitiveTessellationParams& params) {
    if (ogive.base_radius <= 0.0 || ogive.length <= 0.0) {
        throw std::invalid_argument("ogive radius and length must be positive");
    }

    geom::Vec3d u, v;
    geom::Vec3d axis = ogive.axis;
    axis.normalize();
    build_orthonormal_basis(axis, u, v);

    const double R = ogive.base_radius;
    const double L = ogive.length;
    // Ogive radius of curvature rho = (R^2 + L^2) / (2R)
    const double rho = (R * R + L * L) / (2.0 * R);

    const double delta = params.max_edge_length;
    const size_t n_phi = std::max(params.min_radial_segments,
                                  static_cast<size_t>(std::ceil(2.0 * std::numbers::pi * R / delta)));
    const size_t n_x = std::max(params.min_axial_segments,
                                static_cast<size_t>(std::ceil(L / delta)));

    NormalizedMesh mesh;
    mesh.groups.push_back(ogive.name);

    // Tip at origin (index 0)
    mesh.vertices.push_back(ogive.origin);

    // Intermediate rings k = 1 ... n_x
    for (size_t k = 1; k <= n_x; ++k) {
        const double x = static_cast<double>(k) / static_cast<double>(n_x) * L;
        const double term = rho * rho - (L - x) * (L - x);
        const double r = std::sqrt(std::max(0.0, term)) + R - rho;
        const geom::Vec3d center_k = ogive.origin + axis * x;

        for (size_t i = 0; i < n_phi; ++i) {
            const double phi = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(n_phi);
            const geom::Vec3d p = center_k + (u * std::cos(phi) + v * std::sin(phi)) * r;
            mesh.vertices.push_back(p);
        }
    }

    // Tip fan
    for (size_t i = 0; i < n_phi; ++i) {
        const uint32_t i0 = 1 + static_cast<uint32_t>(i);
        const uint32_t i1 = 1 + static_cast<uint32_t>((i + 1) % n_phi);
        mesh.triangles.push_back({0, i0, i1});
        mesh.triangle_tags.push_back(ogive.material_tag);
    }

    // Intermediate quads
    for (size_t k = 1; k < n_x; ++k) {
        const uint32_t ring0 = 1 + static_cast<uint32_t>((k - 1) * n_phi);
        const uint32_t ring1 = 1 + static_cast<uint32_t>(k * n_phi);
        for (size_t i = 0; i < n_phi; ++i) {
            const uint32_t next_i = static_cast<uint32_t>((i + 1) % n_phi);
            const uint32_t a = ring0 + static_cast<uint32_t>(i);
            const uint32_t b = ring0 + next_i;
            const uint32_t c = ring1 + next_i;
            const uint32_t d = ring1 + static_cast<uint32_t>(i);

            mesh.triangles.push_back({a, b, c});
            mesh.triangle_tags.push_back(ogive.material_tag);
            mesh.triangles.push_back({a, c, d});
            mesh.triangle_tags.push_back(ogive.material_tag);
        }
    }

    // Base cap
    if (ogive.base_cap) {
        const uint32_t base_center_idx = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(ogive.origin + axis * L);
        const uint32_t ring_base = 1 + static_cast<uint32_t>((n_x - 1) * n_phi);
        for (size_t i = 0; i < n_phi; ++i) {
            const uint32_t i0 = ring_base + static_cast<uint32_t>(i);
            const uint32_t i1 = ring_base + static_cast<uint32_t>((i + 1) % n_phi);
            mesh.triangles.push_back({base_center_idx, i0, i1});
            mesh.triangle_tags.push_back(ogive.material_tag);
        }
    }

    finalize_mesh(mesh);
    return mesh;
}

NormalizedMesh tessellate_fin(const AnalyticFin& fin,
                              const PrimitiveTessellationParams& /*params*/) {
    if (fin.root_chord <= 0.0 || fin.tip_chord <= 0.0 || fin.span <= 0.0) {
        throw std::invalid_argument("fin chords and span must be positive");
    }

    geom::Vec3d c_dir = fin.chord_dir;
    c_dir.normalize();
    geom::Vec3d s_dir = fin.span_dir;
    s_dir.normalize();
    geom::Vec3d n_dir = geom::cross(c_dir, s_dir);
    n_dir.normalize();

    const double sweep_rad = fin.sweep_angle_deg * std::numbers::pi / 180.0;
    const double sweep_offset = fin.span * std::tan(sweep_rad);

    const geom::Vec3d r_le = fin.origin;
    const geom::Vec3d r_te = fin.origin + c_dir * fin.root_chord;
    const geom::Vec3d t_le = fin.origin + s_dir * fin.span + c_dir * sweep_offset;
    const geom::Vec3d t_te = t_le + c_dir * fin.tip_chord;

    NormalizedMesh mesh;
    mesh.groups.push_back(fin.name);

    if (fin.thickness <= 1e-6) {
        // Flat planar plate
        mesh.vertices = {r_le, r_te, t_te, t_le};
        // Top face
        mesh.triangles.push_back({0, 1, 2});
        mesh.triangle_tags.push_back(fin.material_tag);
        mesh.triangles.push_back({0, 2, 3});
        mesh.triangle_tags.push_back(fin.material_tag);
    } else {
        // Prismatic fin with thickness
        const geom::Vec3d off = n_dir * (0.5 * fin.thickness);
        // Top vertices (0..3)
        mesh.vertices.push_back(r_le + off);
        mesh.vertices.push_back(r_te + off);
        mesh.vertices.push_back(t_te + off);
        mesh.vertices.push_back(t_le + off);
        // Bottom vertices (4..7)
        mesh.vertices.push_back(r_le - off);
        mesh.vertices.push_back(r_te - off);
        mesh.vertices.push_back(t_te - off);
        mesh.vertices.push_back(t_le - off);

        // Top surface
        mesh.triangles.push_back({0, 1, 2});
        mesh.triangle_tags.push_back(fin.material_tag);
        mesh.triangles.push_back({0, 2, 3});
        mesh.triangle_tags.push_back(fin.material_tag);

        // Bottom surface
        mesh.triangles.push_back({4, 6, 5});
        mesh.triangle_tags.push_back(fin.material_tag);
        mesh.triangles.push_back({4, 7, 6});
        mesh.triangle_tags.push_back(fin.material_tag);

        // Leading edge
        mesh.triangles.push_back({0, 3, 7});
        mesh.triangle_tags.push_back(fin.material_tag);
        mesh.triangles.push_back({0, 7, 4});
        mesh.triangle_tags.push_back(fin.material_tag);

        // Trailing edge
        mesh.triangles.push_back({1, 5, 6});
        mesh.triangle_tags.push_back(fin.material_tag);
        mesh.triangles.push_back({1, 6, 2});
        mesh.triangle_tags.push_back(fin.material_tag);

        // Tip edge
        mesh.triangles.push_back({3, 2, 6});
        mesh.triangle_tags.push_back(fin.material_tag);
        mesh.triangles.push_back({3, 6, 7});
        mesh.triangle_tags.push_back(fin.material_tag);

        // Root edge
        mesh.triangles.push_back({0, 4, 5});
        mesh.triangle_tags.push_back(fin.material_tag);
        mesh.triangles.push_back({0, 5, 1});
        mesh.triangle_tags.push_back(fin.material_tag);
    }

    finalize_mesh(mesh);
    return mesh;
}

NormalizedMesh merge_meshes(const std::vector<NormalizedMesh>& meshes) {
    if (meshes.empty()) return {};
    if (meshes.size() == 1) return meshes.front();

    NormalizedMesh merged;
    std::set<std::string> group_set;

    for (const auto& m : meshes) {
        const uint32_t v_offset = static_cast<uint32_t>(merged.vertices.size());
        merged.vertices.insert(merged.vertices.end(), m.vertices.begin(), m.vertices.end());

        for (const auto& tri : m.triangles) {
            merged.triangles.push_back({tri[0] + v_offset, tri[1] + v_offset, tri[2] + v_offset});
        }
        merged.triangle_tags.insert(merged.triangle_tags.end(),
                                    m.triangle_tags.begin(), m.triangle_tags.end());

        for (const auto& g : m.groups) {
            if (group_set.insert(g).second) {
                merged.groups.push_back(g);
            }
        }
    }

    finalize_mesh(merged);
    return merged;
}

NormalizedMesh tessellate_aerospace_assembly(const nlohmann::json& primitives_json,
                                             double max_edge_length) {
    std::vector<NormalizedMesh> parts;
    const auto& prim_arr = primitives_json.is_array() ? primitives_json
                                                      : primitives_json.value("primitives", nlohmann::json::array());

    PrimitiveTessellationParams params;
    params.max_edge_length = max_edge_length;

    for (const auto& item : prim_arr) {
        const std::string type = item.value("type", "");
        const std::string name = item.value("name", type);
        const std::string mat = item.value("material_tag", "pec");

        if (type == "sphere") {
            AnalyticSphere s;
            s.name = name;
            s.material_tag = mat;
            s.radius = item.value("radius", 1.0);
            if (item.contains("center")) {
                s.center = {item["center"][0].get<double>(), item["center"][1].get<double>(),
                            item["center"][2].get<double>()};
            }
            parts.push_back(tessellate_sphere(s, params));
        } else if (type == "cylinder") {
            AnalyticCylinder c;
            c.name = name;
            c.material_tag = mat;
            c.radius = item.value("radius", 0.5);
            c.length = item.value("length", 2.0);
            c.cap_start = item.value("cap_start", true);
            c.cap_end = item.value("cap_end", true);
            if (item.contains("origin")) {
                c.origin = {item["origin"][0].get<double>(), item["origin"][1].get<double>(),
                            item["origin"][2].get<double>()};
            }
            if (item.contains("axis")) {
                c.axis = {item["axis"][0].get<double>(), item["axis"][1].get<double>(),
                          item["axis"][2].get<double>()};
            }
            parts.push_back(tessellate_cylinder(c, params));
        } else if (type == "ogive") {
            AnalyticTangentOgive o;
            o.name = name;
            o.material_tag = mat;
            o.base_radius = item.value("base_radius", 0.5);
            o.length = item.value("length", 1.5);
            o.base_cap = item.value("base_cap", true);
            if (item.contains("origin")) {
                o.origin = {item["origin"][0].get<double>(), item["origin"][1].get<double>(),
                            item["origin"][2].get<double>()};
            }
            if (item.contains("axis")) {
                o.axis = {item["axis"][0].get<double>(), item["axis"][1].get<double>(),
                          item["axis"][2].get<double>()};
            }
            parts.push_back(tessellate_tangent_ogive(o, params));
        } else if (type == "fin") {
            AnalyticFin f;
            f.name = name;
            f.material_tag = mat;
            f.root_chord = item.value("root_chord", 0.8);
            f.tip_chord = item.value("tip_chord", 0.4);
            f.span = item.value("span", 0.6);
            f.sweep_angle_deg = item.value("sweep_angle_deg", 30.0);
            f.thickness = item.value("thickness", 0.02);
            if (item.contains("origin")) {
                f.origin = {item["origin"][0].get<double>(), item["origin"][1].get<double>(),
                            item["origin"][2].get<double>()};
            }
            if (item.contains("chord_dir")) {
                f.chord_dir = {item["chord_dir"][0].get<double>(), item["chord_dir"][1].get<double>(),
                               item["chord_dir"][2].get<double>()};
            }
            if (item.contains("span_dir")) {
                f.span_dir = {item["span_dir"][0].get<double>(), item["span_dir"][1].get<double>(),
                              item["span_dir"][2].get<double>()};
            }
            parts.push_back(tessellate_fin(f, params));
        } else {
            throw std::invalid_argument("unknown primitive type: " + type);
        }
    }

    return merge_meshes(parts);
}

} // namespace strikecem

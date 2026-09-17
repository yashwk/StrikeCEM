// STL/OBJ loading, normalization, inspection, repair, and cache.
#include "strikecem/io/MeshLoader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <sstream>

#include "strikecem/core/Conventions.hpp"
#include "strikecem/core/Config.hpp" // sha256_hex, canonical_json

namespace strikecem {
namespace {

namespace fs = std::filesystem;

float read_f32_le(const char*& p) {
    float v;
    std::memcpy(&v, p, sizeof(v));
    p += sizeof(v);
    return v;
}

uint32_t read_u32_le(const char*& p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    p += sizeof(v);
    return v;
}

std::string read_file_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw MeshLoadError("cannot open mesh file: " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

geom::Vec3d get_vertex(const std::vector<geom::Vec3d>& verts, long long idx,
                       const fs::path& path) {
    const long long n = static_cast<long long>(verts.size());
    if (idx > 0) {
        if (idx > n) throw MeshLoadError("OBJ face index out of range in " + path.string());
        return verts[static_cast<size_t>(idx - 1)];
    }
    if (idx < 0) {
        if (-idx > n) throw MeshLoadError("OBJ face index out of range in " + path.string());
        return verts[static_cast<size_t>(n + idx)];
    }
    throw MeshLoadError("OBJ face index 0 is invalid in " + path.string());
}

struct BitwiseVec3 {
    // Exact-duplicate welding only; tolerance welding is future work.
    bool operator()(const geom::Vec3d& a, const geom::Vec3d& b) const {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return a.z < b.z;
    }
};

struct EdgeKey {
    uint32_t v0, v1; // unordered endpoints
    bool operator<(const EdgeKey& o) const {
        return v0 != o.v0 ? v0 < o.v0 : v1 < o.v1;
    }
};

struct EdgeUse {
    uint32_t tri = 0;
    bool forward = true; // tri traverses v0 -> v1
};

// Longest-edge over shortest-altitude ratio: L^2 / (2A).
double aspect_ratio(const geom::Vec3d& a, const geom::Vec3d& b, const geom::Vec3d& c,
                    double area) {
    const double lab = (b - a).lengthSquared();
    const double lbc = (c - b).lengthSquared();
    const double lca = (a - c).lengthSquared();
    const double lmax2 = std::max({lab, lbc, lca});
    if (!(area > 0.0)) return std::numeric_limits<double>::infinity();
    return lmax2 / (2.0 * area);
}

struct BuiltMesh {
    std::vector<geom::Vec3d> vertices;
    std::vector<std::array<uint32_t, 3>> triangles;
    std::vector<geom::Vec3d> normals;
    std::vector<double> areas;
    MeshReport report;
};

BuiltMesh build_mesh(const std::vector<RawTriangle>& raw, double max_aspect_ratio) {
    BuiltMesh m;
    std::map<geom::Vec3d, uint32_t, BitwiseVec3> index;
    auto vertex_id = [&](const geom::Vec3d& v) {
        auto it = index.find(v);
        if (it != index.end()) return it->second;
        const uint32_t id = static_cast<uint32_t>(m.vertices.size());
        m.vertices.push_back(v);
        index[v] = id;
        return id;
    };
    std::map<EdgeKey, std::vector<EdgeUse>> edges;
    double aspect_sum = 0.0;
    size_t aspect_n = 0;
    double max_edge2 = 0.0;
    m.report.aspect_min = std::numeric_limits<double>::infinity();
    for (const auto& t : raw) {
        const double area = geom::triangleArea(t.a, t.b, t.c);
        max_edge2 = std::max({max_edge2, (t.b - t.a).lengthSquared(),
                              (t.c - t.b).lengthSquared(), (t.a - t.c).lengthSquared()});
        const uint32_t tri = static_cast<uint32_t>(m.triangles.size());
        m.triangles.push_back({vertex_id(t.a), vertex_id(t.b), vertex_id(t.c)});
        m.normals.push_back(geom::triangleNormal(t.a, t.b, t.c));
        m.areas.push_back(area);
        if (area == 0.0) {
            ++m.report.degenerate_count;
            continue;
        }
        const double ar = aspect_ratio(t.a, t.b, t.c, area);
        m.report.aspect_min = std::min(m.report.aspect_min, ar);
        m.report.aspect_max = std::max(m.report.aspect_max, ar);
        aspect_sum += ar;
        ++aspect_n;
        if (ar > max_aspect_ratio) ++m.report.aspect_over_limit_count;
        const auto& tri_idx = m.triangles.back();
        const uint32_t v[3] = {tri_idx[0], tri_idx[1], tri_idx[2]};
        for (int e = 0; e < 3; ++e) {
            const uint32_t a = v[e], b = v[(e + 1) % 3];
            const EdgeKey key{std::min(a, b), std::max(a, b)};
            edges[key].push_back({tri, a == key.v0});
        }
    }
    for (const auto& [key, uses] : edges) {
        if (uses.size() == 1) {
            ++m.report.open_edge_count;
        } else if (uses.size() == 2) {
            if (uses[0].forward == uses[1].forward) ++m.report.inconsistent_winding_count;
        } else {
            ++m.report.nonmanifold_edge_count;
        }
    }
    m.report.vertex_count = m.vertices.size();
    m.report.triangle_count = m.triangles.size();
    m.report.max_edge_length_m = std::sqrt(max_edge2);
    if (aspect_n == 0) m.report.aspect_min = 0.0;
    m.report.aspect_mean = aspect_n > 0 ? aspect_sum / aspect_n : 0.0;
    if (!m.vertices.empty()) {
        m.report.bbox_min = m.report.bbox_max = m.vertices[0];
        for (const auto& v : m.vertices) {
            m.report.bbox_min.x = std::min(m.report.bbox_min.x, v.x);
            m.report.bbox_min.y = std::min(m.report.bbox_min.y, v.y);
            m.report.bbox_min.z = std::min(m.report.bbox_min.z, v.z);
            m.report.bbox_max.x = std::max(m.report.bbox_max.x, v.x);
            m.report.bbox_max.y = std::max(m.report.bbox_max.y, v.y);
            m.report.bbox_max.z = std::max(m.report.bbox_max.z, v.z);
        }
    }
    m.report.total_area_m2 = 0.0;
    for (double a : m.areas) m.report.total_area_m2 += a;
    return m;
}

std::string hash_normalized(const BuiltMesh& m) {
    std::string bytes;
    bytes.reserve(m.vertices.size() * 24 + m.triangles.size() * 12);
    for (const auto& v : m.vertices)
        bytes.append(reinterpret_cast<const char*>(&v), sizeof(v));
    for (const auto& t : m.triangles)
        bytes.append(reinterpret_cast<const char*>(t.data()), sizeof(uint32_t) * 3);
    return sha256_hex(bytes);
}

// Cache layout v3: magic + key-input echo + counts + mesh hash +
// repair metadata + payload.
constexpr char kMagic[] = "SCEMMESH04";

void append_u64(std::string& out, uint64_t v) {
    out.append(reinterpret_cast<const char*>(&v), sizeof(v));
}

void append_f64(std::string& out, double v) {
    out.append(reinterpret_cast<const char*>(&v), sizeof(v));
}

void append_report(std::string& out, const MeshReport& r) {
    append_u64(out, r.vertex_count);
    append_u64(out, r.triangle_count);
    append_f64(out, r.bbox_min.x);
    append_f64(out, r.bbox_min.y);
    append_f64(out, r.bbox_min.z);
    append_f64(out, r.bbox_max.x);
    append_f64(out, r.bbox_max.y);
    append_f64(out, r.bbox_max.z);
    append_f64(out, r.total_area_m2);
    append_u64(out, r.degenerate_count);
    append_u64(out, r.inconsistent_winding_count);
    append_u64(out, r.open_edge_count);
    append_u64(out, r.nonmanifold_edge_count);
    append_f64(out, r.aspect_min);
    append_f64(out, r.aspect_max);
    append_f64(out, r.aspect_mean);
    append_u64(out, r.aspect_over_limit_count);
    append_f64(out, r.max_edge_length_m);
}

constexpr size_t kReportBytes = 8 + 8 + 6 * 8 + 8 + 5 * 8 + 3 * 8 + 8 + 8;

uint64_t take_u64(const char*& p, const char* end, const char* what) {
    if (p + sizeof(uint64_t) > end) throw MeshLoadError(std::string("corrupt mesh cache: ") + what);
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    p += sizeof(v);
    return v;
}

double take_f64(const char*& p, const char* end, const char* what) {
    if (p + sizeof(double) > end) throw MeshLoadError(std::string("corrupt mesh cache: ") + what);
    double v;
    std::memcpy(&v, p, sizeof(v));
    p += sizeof(v);
    return v;
}

MeshReport take_report(const char*& p, const char* end) {
    MeshReport r;
    r.vertex_count = static_cast<size_t>(take_u64(p, end, "report"));
    r.triangle_count = static_cast<size_t>(take_u64(p, end, "report"));
    r.bbox_min.x = take_f64(p, end, "report");
    r.bbox_min.y = take_f64(p, end, "report");
    r.bbox_min.z = take_f64(p, end, "report");
    r.bbox_max.x = take_f64(p, end, "report");
    r.bbox_max.y = take_f64(p, end, "report");
    r.bbox_max.z = take_f64(p, end, "report");
    r.total_area_m2 = take_f64(p, end, "report");
    r.degenerate_count = static_cast<size_t>(take_u64(p, end, "report"));
    r.inconsistent_winding_count = static_cast<size_t>(take_u64(p, end, "report"));
    r.open_edge_count = static_cast<size_t>(take_u64(p, end, "report"));
    r.nonmanifold_edge_count = static_cast<size_t>(take_u64(p, end, "report"));
    r.aspect_min = take_f64(p, end, "report");
    r.aspect_max = take_f64(p, end, "report");
    r.aspect_mean = take_f64(p, end, "report");
    r.aspect_over_limit_count = static_cast<size_t>(take_u64(p, end, "report"));
    r.max_edge_length_m = take_f64(p, end, "report");
    return r;
}

std::string cache_path_for(const std::string& key, const fs::path& cache_dir) {
    return (cache_dir / (key + ".bin")).string();
}

bool try_load_cache(const std::string& key, const std::string& key_echo,
                    const fs::path& cache_dir, double max_aspect_ratio, NormalizedMesh& out) {
    std::error_code ec;
    if (!fs::exists(cache_path_for(key, cache_dir), ec)) return false;
    std::string bytes;
    try {
        bytes = read_file_bytes(cache_path_for(key, cache_dir));
    } catch (const MeshLoadError&) {
        return false;
    }
    try {
        const char* p = bytes.data();
        const char* end = p + bytes.size();
        if (bytes.size() < sizeof(kMagic) ||
            std::memcmp(p, kMagic, sizeof(kMagic)) != 0)
            return false;
        p += sizeof(kMagic);
        const uint64_t echo_len = take_u64(p, end, "echo length");
        if (p + echo_len > end || std::string(p, echo_len) != key_echo) return false;
        p += echo_len;
        const uint64_t nverts = take_u64(p, end, "vertex count");
        const uint64_t ntris = take_u64(p, end, "triangle count");
        const uint64_t hash_len = take_u64(p, end, "hash length");
        if (p + hash_len > end) return false;
        const std::string stored_hash(p, hash_len);
        p += hash_len;
        const uint64_t repaired_flag = take_u64(p, end, "repair flag");
        const MeshReport stored_before = take_report(p, end);
        const uint64_t ngroups = take_u64(p, end, "group count");
        if (ngroups > 100'000) return false;
        std::vector<std::string> groups;
        for (uint64_t i = 0; i < ngroups; ++i) {
            const uint64_t len = take_u64(p, end, "group name");
            if (len > 4096 || p + len > end) return false;
            groups.emplace_back(p, len);
            p += len;
        }
        if (nverts > 100'000'000 || ntris > 100'000'000) return false;
        const size_t need = nverts * sizeof(geom::Vec3d) + ntris * 3 * sizeof(uint32_t) +
                            ntris * sizeof(geom::Vec3d) + ntris * sizeof(double);
        if (static_cast<size_t>(end - p) != need) return false;
        out.vertices.resize(nverts);
        std::memcpy(out.vertices.data(), p, nverts * sizeof(geom::Vec3d));
        p += nverts * sizeof(geom::Vec3d);
        out.triangles.resize(ntris);
        std::memcpy(out.triangles.data(), p, ntris * 3 * sizeof(uint32_t));
        p += ntris * 3 * sizeof(uint32_t);
        out.normals.resize(ntris);
        std::memcpy(out.normals.data(), p, ntris * sizeof(geom::Vec3d));
        p += ntris * sizeof(geom::Vec3d);
        out.areas.resize(ntris);
        std::memcpy(out.areas.data(), p, ntris * sizeof(double));
        out.groups = std::move(groups);
        // Rebuild the report from payload so resume never trusts a stale one.
        std::vector<RawTriangle> raw;
        raw.reserve(ntris);
        for (const auto& t : out.triangles)
            raw.push_back({out.vertices[t[0]], out.vertices[t[1]], out.vertices[t[2]]});
        BuiltMesh rebuilt = build_mesh(raw, max_aspect_ratio);
        out.report = rebuilt.report;
        BuiltMesh check;
        check.vertices = out.vertices;
        check.triangles = out.triangles;
        if (hash_normalized(check) != stored_hash) return false;
        out.normalized_mesh_hash = stored_hash;
        out.repaired = (repaired_flag != 0);
        out.report_before = stored_before;
        out.cache_hit = true;
        return true;
    } catch (const MeshLoadError&) {
        return false;
    }
}

void store_cache(const std::string& key, const std::string& key_echo, const fs::path& cache_dir,
                 const NormalizedMesh& mesh) {
    std::error_code ec;
    fs::create_directories(cache_dir, ec);
    if (ec) return; // cache is advisory; a missing cache never fails the run
    std::string bytes(kMagic, sizeof(kMagic));
    append_u64(bytes, key_echo.size());
    bytes += key_echo;
    append_u64(bytes, mesh.vertices.size());
    append_u64(bytes, mesh.triangles.size());
    append_u64(bytes, mesh.normalized_mesh_hash.size());
    bytes += mesh.normalized_mesh_hash;
    append_u64(bytes, mesh.repaired ? 1 : 0);
    append_report(bytes, mesh.report_before);
    append_u64(bytes, mesh.groups.size());
    for (const auto& g : mesh.groups) {
        append_u64(bytes, g.size());
        bytes += g;
    }
    bytes.append(reinterpret_cast<const char*>(mesh.vertices.data()),
                 mesh.vertices.size() * sizeof(geom::Vec3d));
    bytes.append(reinterpret_cast<const char*>(mesh.triangles.data()),
                 mesh.triangles.size() * 3 * sizeof(uint32_t));
    bytes.append(reinterpret_cast<const char*>(mesh.normals.data()),
                 mesh.normals.size() * sizeof(geom::Vec3d));
    bytes.append(reinterpret_cast<const char*>(mesh.areas.data()),
                 mesh.areas.size() * sizeof(double));
    const fs::path tmp = cache_dir / (key + ".tmp");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return;
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        out.flush();
        if (!out) return;
    }
    fs::rename(tmp, cache_path_for(key, cache_dir), ec);
}

} // namespace

std::vector<RawTriangle> parse_stl(const fs::path& path) {
    const std::string bytes = read_file_bytes(path);
    std::vector<RawTriangle> tris;
    // Binary STL: 84-byte header + 50 bytes per facet.
    if (bytes.size() >= 84) {
        const char* p = bytes.data() + 80;
        const uint32_t count = read_u32_le(p);
        if (bytes.size() == 84 + static_cast<size_t>(count) * 50) {
            for (uint32_t i = 0; i < count; ++i) {
                p += 12; // facet normal (recomputed; never trusted)
                geom::Vec3d v[3];
                for (auto& corner : v) {
                    corner.x = read_f32_le(p);
                    corner.y = read_f32_le(p);
                    corner.z = read_f32_le(p);
                }
                p += 2; // attribute byte count
                tris.push_back({v[0], v[1], v[2]});
            }
            return tris;
        }
    }
    // ASCII STL.
    std::istringstream in(bytes);
    std::string word;
    geom::Vec3d v[3];
    int corner = -1;
    auto need = [&](const std::string& expected) {
        if (!(in >> word) || word != expected)
            throw MeshLoadError("malformed ASCII STL in " + path.string());
    };
    if (!(in >> word) || word != "solid")
        throw MeshLoadError("unrecognized STL content in " + path.string());
    std::getline(in, word); // rest of the solid line (name is free-form)
    while (in >> word) {
        if (word == "endsolid") return tris;
        if (word != "facet")
            throw MeshLoadError("malformed ASCII STL in " + path.string());
        need("normal");
        double nx, ny, nz;
        if (!(in >> nx >> ny >> nz)) throw MeshLoadError("malformed ASCII STL normal");
        (void)nx;
        (void)ny;
        (void)nz;
        need("outer");
        need("loop");
        for (corner = 0; corner < 3; ++corner) {
            need("vertex");
            if (!(in >> v[corner].x >> v[corner].y >> v[corner].z))
                throw MeshLoadError("malformed ASCII STL vertex");
        }
        need("endloop");
        need("endfacet");
        tris.push_back({v[0], v[1], v[2]});
    }
    throw MeshLoadError("ASCII STL missing endsolid in " + path.string());
}

ObjMesh parse_obj(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw MeshLoadError("cannot open mesh file: " + path.string());
    std::vector<geom::Vec3d> verts;
    ObjMesh mesh;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string tag;
        if (!(ls >> tag) || tag[0] == '#') continue;
        if (tag == "v") {
            geom::Vec3d v;
            if (!(ls >> v.x >> v.y >> v.z))
                throw MeshLoadError("malformed OBJ vertex in " + path.string());
            verts.push_back(v);
        } else if (tag == "g" || tag == "o") {
            std::string name;
            if (ls >> name &&
                std::find(mesh.groups.begin(), mesh.groups.end(), name) == mesh.groups.end())
                mesh.groups.push_back(name);
        } else if (tag == "f") {
            std::vector<geom::Vec3d> face;
            std::string token;
            while (ls >> token) {
                const size_t slash = token.find('/');
                const std::string idx = token.substr(0, slash);
                face.push_back(get_vertex(verts, std::stoll(idx), path));
            }
            if (face.size() < 3)
                throw MeshLoadError("OBJ face with fewer than 3 vertices in " + path.string());
            // Fan triangulation for polygonal faces.
            for (size_t i = 1; i + 1 < face.size(); ++i)
                mesh.triangles.push_back({face[0], face[i], face[i + 1]});
        }
    }
    return mesh;
}

NormalizedMesh load_normalized_mesh(const nlohmann::json& resolved, const fs::path& config_dir,
                                    const std::string& schema_version) {
    const auto& model = resolved["model"];
    const std::string rel = model["path"].get<std::string>();
    const fs::path mesh_path =
        fs::path(rel).is_absolute() ? fs::path(rel) : config_dir / rel;
    const std::string raw_bytes = read_file_bytes(mesh_path);
    const std::string units = model["units"].get<std::string>();
    const std::string repair = model["mesh_repair"].get<std::string>();
    const std::string transform_json =
        model.contains("transform") ? model["transform"].dump() : "{}";
    const std::string key_echo =
        units + "|" + transform_json + "|" + repair + "|" + schema_version;
    const std::string key_input =
        raw_bytes + "|" + key_echo + "|" + std::to_string(raw_bytes.size());
    const std::string geometry_hash = sha256_hex(key_input);

    fs::path cache_dir = config_dir / ".scem_cache";
    if (resolved["mesh"].contains("cache_dir")) {
        const fs::path custom(resolved["mesh"]["cache_dir"].get<std::string>());
        cache_dir = custom.is_absolute() ? custom : config_dir / custom;
    }
    NormalizedMesh mesh;
    mesh.geometry_hash = geometry_hash;
    const double max_ar = resolved["mesh"]["max_aspect_ratio"].get<double>();
    if (try_load_cache(geometry_hash, key_echo, cache_dir, max_ar, mesh)) return mesh;

    const std::string lower = rel;
    std::vector<RawTriangle> raw;
    auto ends_with = [&](const char* ext) {
        const size_t n = std::strlen(ext);
        if (lower.size() < n) return false;
        return std::equal(ext, ext + n, lower.end() - n,
                          [](char a, char b) { return std::tolower(a) == std::tolower(b); });
    };
    try {
        if (ends_with(".stl")) {
            raw = parse_stl(mesh_path);
        } else if (ends_with(".obj")) {
            ObjMesh obj = parse_obj(mesh_path);
            raw = std::move(obj.triangles);
            mesh.groups = std::move(obj.groups);
        } else {
            throw MeshLoadError("unsupported mesh extension (v1 accepts .stl/.obj): " + rel);
        }
    } catch (const MeshLoadError&) {
        throw;
    } catch (const std::exception& e) {
        throw MeshLoadError("failed to parse mesh " + mesh_path.string() + ": " + e.what());
    }
    if (raw.empty()) throw MeshLoadError("mesh contains no triangles: " + mesh_path.string());
    const double to_m = units_to_metres(units);
    geom::Vec3d euler{0, 0, 0}, translate{0, 0, 0};
    double scale = 1.0;
    if (model.contains("transform")) {
        const auto& t = model["transform"];
        if (t.contains("translate")) {
            translate = {t["translate"][0].get<double>(), t["translate"][1].get<double>(),
                         t["translate"][2].get<double>()};
        }
        if (t.contains("rotate_euler_deg")) {
            euler = {t["rotate_euler_deg"][0].get<double>(), t["rotate_euler_deg"][1].get<double>(),
                     t["rotate_euler_deg"][2].get<double>()};
        }
        if (t.contains("scale")) scale = t["scale"].get<double>();
    }
    for (auto& tri : raw) {
        tri.a = apply_transform(tri.a * to_m, scale, euler, translate);
        tri.b = apply_transform(tri.b * to_m, scale, euler, translate);
        tri.c = apply_transform(tri.c * to_m, scale, euler, translate);
    }

    BuiltMesh built = build_mesh(raw, max_ar);
    const bool fatal =
        built.report.degenerate_count > 0 || built.report.nonmanifold_edge_count > 0;
    if (repair == "repair") {
        mesh.report_before = built.report;
        std::vector<RawTriangle> kept;
        kept.reserve(raw.size());
        for (const auto& tri : raw)
            if (geom::triangleArea(tri.a, tri.b, tri.c) != 0.0) kept.push_back(tri);
        if (kept.size() != raw.size()) {
            mesh.repaired = true;
            built = build_mesh(kept, max_ar);
        }
        if (built.report.nonmanifold_edge_count > 0)
            throw MeshLoadError("repair could not resolve non-manifold edges");
        if (built.report.degenerate_count > 0)
            throw MeshLoadError("repair could not resolve degenerate triangles");
    } else if (fatal) {
        throw MeshLoadError("mesh has fatal topology errors (degenerate triangles or "
                            "non-manifold edges); set model.mesh_repair=repair for an explicit "
                            "repair pass");
    }

    mesh.vertices = std::move(built.vertices);
    mesh.triangles = std::move(built.triangles);
    mesh.normals = std::move(built.normals);
    mesh.areas = std::move(built.areas);
    mesh.report = built.report;
    BuiltMesh check;
    check.vertices = mesh.vertices;
    check.triangles = mesh.triangles;
    mesh.normalized_mesh_hash = hash_normalized(check);
    store_cache(geometry_hash, key_echo, cache_dir, mesh);
    return mesh;
}

} // namespace strikecem

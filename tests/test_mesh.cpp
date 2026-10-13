// Mesh pipeline tests: STL/OBJ parsing, reports, transforms, repair
// modes, hashes, and the geometry cache.
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <gtest/gtest.h>

#include "strikecem/app/CLI.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"

namespace {

namespace fs = std::filesystem;

std::string fixture(const std::string& name) { return std::string(SCEM_FIXTURE_DIR) + "/" + name; }
std::string example(const std::string& name) {
    return std::string(SCEM_FIXTURE_DIR) + "/../../examples/" + name;
}

strikecem::NormalizedMesh load_fixture_mesh(const std::string& name) {
    const auto rc = strikecem::load_config(fixture(name), SCEM_SCHEMA_PATH);
    return strikecem::load_normalized_mesh(rc.value, SCEM_FIXTURE_DIR, rc.schema_version);
}

TEST(Mesh, ParseAsciiStlPlate) {
    const auto tris = strikecem::parse_stl(example("plate.stl"));
    ASSERT_EQ(tris.size(), 2u);
    EXPECT_DOUBLE_EQ(tris[0].a.x, 0.0);
    EXPECT_DOUBLE_EQ(tris[0].c.x, 1.0);
    EXPECT_DOUBLE_EQ(tris[0].c.y, 1.0);
}

TEST(Mesh, ParseBinaryStl) {
    // One facet: right triangle in z=0, normal +z.
    const fs::path tmp = fs::temp_directory_path() / "scem_binary_test.stl";
    std::string bytes(80, '\0');
    const uint32_t count = 1;
    bytes.append(reinterpret_cast<const char*>(&count), 4);
    const float facet[12] = {0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0};
    bytes.append(reinterpret_cast<const char*>(facet), sizeof(facet));
    const char attr[2] = {0, 0};
    bytes.append(attr, 2);
    { std::ofstream out(tmp, std::ios::binary); out.write(bytes.data(), bytes.size()); }
    const auto tris = strikecem::parse_stl(tmp);
    ASSERT_EQ(tris.size(), 1u);
    EXPECT_FLOAT_EQ(tris[0].b.x, 1.0f);
    EXPECT_FLOAT_EQ(tris[0].c.y, 1.0f);
    std::error_code ec;
    fs::remove(tmp, ec);
}

TEST(Mesh, ParseObjTriangle) {
    const auto obj = strikecem::parse_obj(example("triangle.obj"));
    ASSERT_EQ(obj.triangles.size(), 1u);
    EXPECT_DOUBLE_EQ(obj.triangles[0].a.x, 0.0);
    EXPECT_DOUBLE_EQ(obj.triangles[0].b.x, 1.0);
    EXPECT_DOUBLE_EQ(obj.triangles[0].c.y, 1.0);
}

TEST(Mesh, PlateReport) {
    const auto mesh = load_fixture_mesh("valid_minimal.json");
    EXPECT_EQ(mesh.report.vertex_count, 4u);
    EXPECT_EQ(mesh.report.triangle_count, 2u);
    EXPECT_EQ(mesh.report.degenerate_count, 0u);
    EXPECT_EQ(mesh.report.nonmanifold_edge_count, 0u);
    EXPECT_EQ(mesh.report.open_edge_count, 4u); // square boundary
    EXPECT_DOUBLE_EQ(mesh.report.total_area_m2, 1.0);
    EXPECT_DOUBLE_EQ(mesh.report.bbox_min.x, 0.0);
    EXPECT_DOUBLE_EQ(mesh.report.bbox_max.x, 1.0);
    EXPECT_FALSE(mesh.repaired);
    EXPECT_EQ(mesh.geometry_hash.size(), 64u);
    EXPECT_EQ(mesh.normalized_mesh_hash.size(), 64u);
}

TEST(Mesh, TransformIsApplied) {
    const auto mesh = load_fixture_mesh("valid_mesh_transform.json");
    EXPECT_DOUBLE_EQ(mesh.report.bbox_min.x, 1.0);
    EXPECT_DOUBLE_EQ(mesh.report.bbox_min.y, 2.0);
    EXPECT_DOUBLE_EQ(mesh.report.bbox_min.z, 3.0);
    EXPECT_DOUBLE_EQ(mesh.report.bbox_max.x, 2.0);
    EXPECT_DOUBLE_EQ(mesh.report.bbox_max.y, 3.0);
    EXPECT_DOUBLE_EQ(mesh.report.bbox_max.z, 3.0);
}

TEST(Mesh, ObjLoadsThroughPipeline) {
    const auto mesh = load_fixture_mesh("valid_mesh_obj.json");
    EXPECT_EQ(mesh.report.triangle_count, 1u);
    EXPECT_EQ(mesh.report.vertex_count, 3u);
    EXPECT_EQ(mesh.report.open_edge_count, 3u);
}

TEST(Mesh, RepairOffFailsOnDegenerate) {
    EXPECT_THROW(load_fixture_mesh("invalid_repair_off.json"), strikecem::MeshLoadError);
}

TEST(Mesh, RepairDropsDegenerateFacets) {
    const auto mesh = load_fixture_mesh("valid_repair_on.json");
    EXPECT_TRUE(mesh.repaired);
    EXPECT_EQ(mesh.report.triangle_count, 2u);
    EXPECT_EQ(mesh.report.degenerate_count, 0u);
    EXPECT_EQ(mesh.report_before.triangle_count, 3u);
}

TEST(Mesh, NonManifoldFailsEvenWithRepair) {
    EXPECT_THROW(load_fixture_mesh("invalid_nonmanifold_off.json"), strikecem::MeshLoadError);
    try {
        load_fixture_mesh("invalid_nonmanifold_repair.json");
        FAIL() << "expected MeshLoadError";
    } catch (const strikecem::MeshLoadError& e) {
        EXPECT_NE(std::string(e.what()).find("non-manifold"), std::string::npos);
    }
}

TEST(Mesh, CacheHitAndStableHashes) {
    const auto first = load_fixture_mesh("valid_minimal.json");
    const auto second = load_fixture_mesh("valid_minimal.json");
    EXPECT_TRUE(second.cache_hit);
    EXPECT_EQ(first.geometry_hash, second.geometry_hash);
    EXPECT_EQ(first.normalized_mesh_hash, second.normalized_mesh_hash);
    EXPECT_EQ(first.report.triangle_count, second.report.triangle_count);
}

TEST(Mesh, CorruptCacheDerivedDataIsRebuiltAndBadIndicesMiss) {
    const fs::path dir = fs::temp_directory_path() / "scem_mesh_cache_integrity";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    struct Cleanup {
        fs::path path;
        ~Cleanup() {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    } cleanup{dir};

    nlohmann::json config;
    {
        std::ifstream in(fixture("valid_minimal.json"));
        in >> config;
    }
    config["model"]["path"] = fs::absolute(example("plate.stl")).string();
    config["output"]["path"] = (dir / "out.h5").string();
    config["mesh"]["cache_dir"] = "cache";
    const fs::path config_path = dir / "config.json";
    {
        std::ofstream out(config_path);
        out << config.dump(2);
    }
    const auto rc = strikecem::load_config(config_path.string(), SCEM_SCHEMA_PATH);
    const auto first = strikecem::load_normalized_mesh(rc.value, dir, rc.schema_version);
    ASSERT_EQ(first.areas.size(), 2u);
    const auto cache_entry = fs::directory_iterator(dir / "cache");
    ASSERT_NE(cache_entry, fs::directory_iterator{});
    const fs::path cache_path = cache_entry->path();
    auto read_cache = [&] {
        std::ifstream in(cache_path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    };
    auto write_cache = [&](const std::string& bytes) {
        std::ofstream out(cache_path, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    };

    std::string bytes = read_cache();
    ASSERT_GE(bytes.size(), sizeof(double));
    const double wrong_area = 50.0;
    std::memcpy(bytes.data() + bytes.size() - sizeof(double), &wrong_area, sizeof(wrong_area));
    write_cache(bytes);
    const auto rebuilt = strikecem::load_normalized_mesh(rc.value, dir, rc.schema_version);
    EXPECT_TRUE(rebuilt.cache_hit);
    EXPECT_EQ(rebuilt.geometry_hash, first.geometry_hash);
    EXPECT_EQ(rebuilt.areas, first.areas);
    EXPECT_DOUBLE_EQ(rebuilt.report.total_area_m2, first.report.total_area_m2);

    bytes = read_cache();
    const size_t payload_bytes = rebuilt.vertices.size() * sizeof(geom::Vec3d) +
                                 rebuilt.triangles.size() * 3 * sizeof(uint32_t) +
                                 rebuilt.triangles.size() * sizeof(geom::Vec3d) +
                                 rebuilt.triangles.size() * sizeof(double);
    ASSERT_GE(bytes.size(), payload_bytes);
    const size_t index_offset = bytes.size() - payload_bytes +
                                rebuilt.vertices.size() * sizeof(geom::Vec3d);
    const uint32_t invalid_index = UINT32_MAX;
    std::memcpy(bytes.data() + index_offset, &invalid_index, sizeof(invalid_index));
    write_cache(bytes);
    const auto fallback = strikecem::load_normalized_mesh(rc.value, dir, rc.schema_version);
    EXPECT_FALSE(fallback.cache_hit);
    EXPECT_EQ(fallback.normalized_mesh_hash, first.normalized_mesh_hash);
    EXPECT_EQ(fallback.areas, first.areas);
}

TEST(Mesh, ObjGroupsPopulateTriangleTags) {
    const fs::path tmp = fs::temp_directory_path() / "scem_tagged_mesh.obj";
    {
        std::ofstream out(tmp);
        out << "v 0 0 0\n"
            << "v 1 0 0\n"
            << "v 0 1 0\n"
            << "v 1 1 0\n"
            << "g fin_left\n"
            << "f 1 2 3\n"
            << "g fin_right\n"
            << "f 2 4 3\n";
    }
    const auto obj = strikecem::parse_obj(tmp);
    ASSERT_EQ(obj.triangles.size(), 2u);
    ASSERT_EQ(obj.groups.size(), 2u);
    EXPECT_EQ(obj.groups[0], "fin_left");
    EXPECT_EQ(obj.groups[1], "fin_right");
    ASSERT_EQ(obj.triangle_tags.size(), 2u);
    EXPECT_EQ(obj.triangle_tags[0], "fin_left");
    EXPECT_EQ(obj.triangle_tags[1], "fin_right");
    std::error_code ec;
    fs::remove(tmp, ec);
}

TEST(Mesh, LoadNormalizedMeshPreservesTagsAndCache) {
    const fs::path dir = fs::temp_directory_path() / "scem_mesh_tags_cache";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    struct Cleanup {
        fs::path path;
        ~Cleanup() {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    } cleanup{dir};
    const fs::path mesh_path = dir / "two_fins.obj";
    {
        std::ofstream out(mesh_path);
        out << "v 0 0 0\n"
            << "v 1 0 0\n"
            << "v 0 1 0\n"
            << "v 1 1 0\n"
            << "g fin_left\n"
            << "f 1 2 3\n"
            << "g fin_right\n"
            << "f 2 4 3\n";
    }
    nlohmann::json config;
    {
        std::ifstream in(fixture("valid_minimal.json"));
        in >> config;
    }
    config["model"]["path"] = mesh_path.string();
    config["output"]["path"] = (dir / "out.h5").string();
    config["mesh"]["cache_dir"] = "cache";
    const fs::path config_path = dir / "config.json";
    {
        std::ofstream out(config_path);
        out << config.dump(2);
    }
    const auto rc = strikecem::load_config(config_path.string(), SCEM_SCHEMA_PATH);
    const auto first = strikecem::load_normalized_mesh(rc.value, dir, rc.schema_version);
    ASSERT_EQ(first.triangle_tags.size(), 2u);
    EXPECT_EQ(first.triangle_tags[0], "fin_left");
    EXPECT_EQ(first.triangle_tags[1], "fin_right");
    EXPECT_FALSE(first.cache_hit);

    const auto cached = strikecem::load_normalized_mesh(rc.value, dir, rc.schema_version);
    EXPECT_TRUE(cached.cache_hit);
    ASSERT_EQ(cached.triangle_tags.size(), 2u);
    EXPECT_EQ(cached.triangle_tags[0], "fin_left");
    EXPECT_EQ(cached.triangle_tags[1], "fin_right");
}

int call_cli(const std::vector<std::string>& arg_list) {
    std::vector<std::string> storage = {"strikecem"};
    for (const auto& a : arg_list) storage.push_back(a);
    std::vector<char*> argv;
    for (auto& s : storage) argv.push_back(s.data());
    return strikecem::cli::run(static_cast<int>(argv.size()), argv.data());
}

TEST(MeshExport, ObjRoundTrip) {
    const fs::path tmp = fs::temp_directory_path() / "scem_export_test.obj";
    std::error_code ec;
    fs::remove(tmp, ec);

    const auto mesh = load_fixture_mesh("valid_minimal.json");
    strikecem::export_mesh_obj(mesh, tmp);
    ASSERT_TRUE(fs::exists(tmp));

    const auto obj = strikecem::parse_obj(tmp);
    EXPECT_EQ(obj.triangles.size(), mesh.triangles.size());

    fs::remove(tmp, ec);
}

TEST(MeshExport, StlRoundTrip) {
    const fs::path tmp = fs::temp_directory_path() / "scem_export_test.stl";
    std::error_code ec;
    fs::remove(tmp, ec);

    const auto mesh = load_fixture_mesh("valid_minimal.json");
    strikecem::export_mesh_stl(mesh, tmp);
    ASSERT_TRUE(fs::exists(tmp));

    const auto tris = strikecem::parse_stl(tmp);
    EXPECT_EQ(tris.size(), mesh.triangles.size());

    fs::remove(tmp, ec);
}

TEST(MeshExport, VtuStructure) {
    const fs::path tmp = fs::temp_directory_path() / "scem_export_test.vtu";
    std::error_code ec;
    fs::remove(tmp, ec);

    const auto mesh = load_fixture_mesh("valid_minimal.json");
    strikecem::export_mesh_vtu(mesh, tmp);
    ASSERT_TRUE(fs::exists(tmp));

    std::ifstream in(tmp);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("<VTKFile type=\"UnstructuredGrid\""), std::string::npos);
    EXPECT_NE(content.find("NumberOfPoints=\"4\""), std::string::npos);
    EXPECT_NE(content.find("NumberOfCells=\"2\""), std::string::npos);
    EXPECT_NE(content.find("Name=\"Points\""), std::string::npos);
    EXPECT_NE(content.find("Name=\"Normals\""), std::string::npos);
    EXPECT_NE(content.find("Name=\"Area_m2\""), std::string::npos);

    fs::remove(tmp, ec);
}

TEST(MeshExport, AutoDetectAndExplicitFormat) {
    const fs::path dir = fs::temp_directory_path() / "scem_export_dispatcher";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    const auto mesh = load_fixture_mesh("valid_minimal.json");

    // Auto-detect obj
    const fs::path p_obj = dir / "test.obj";
    strikecem::export_mesh(mesh, p_obj);
    EXPECT_TRUE(fs::exists(p_obj));

    // Auto-detect stl
    const fs::path p_stl = dir / "test.stl";
    strikecem::export_mesh(mesh, p_stl);
    EXPECT_TRUE(fs::exists(p_stl));

    // Auto-detect vtu
    const fs::path p_vtu = dir / "test.vtu";
    strikecem::export_mesh(mesh, p_vtu);
    EXPECT_TRUE(fs::exists(p_vtu));

    // Explicit format override
    const fs::path p_custom = dir / "test.custom";
    strikecem::export_mesh(mesh, p_custom, "stl");
    EXPECT_TRUE(fs::exists(p_custom));
    const auto tris = strikecem::parse_stl(p_custom);
    EXPECT_EQ(tris.size(), mesh.triangles.size());

    // Invalid format throws MeshLoadError
    EXPECT_THROW(strikecem::export_mesh(mesh, p_custom, "invalid_fmt"), strikecem::MeshLoadError);

    fs::remove_all(dir, ec);
}

TEST(MeshExport, ProceduralAssemblyExport) {
    const fs::path dir = fs::temp_directory_path() / "scem_export_procedural";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    nlohmann::json cfg;
    cfg["scem_schema_version"] = "1.0";
    cfg["model"] = {
        {"primitives", {
            {{"type", "ogive"}, {"name", "nose"}, {"base_radius", 0.2}, {"length", 0.6}},
            {{"type", "cylinder"}, {"name", "fuselage"}, {"radius", 0.2}, {"length", 1.2}, {"origin", {0.6, 0.0, 0.0}}}
        }},
        {"max_edge_length", 0.1}
    };
    cfg["frequency"] = {{"frequency_hz", 10e9}};
    cfg["angles"]["azimuth"] = {{"start", 0}, {"stop", 0}, {"step", 1}};
    cfg["angles"]["elevation"] = {{"start", 0}, {"stop", 0}, {"step", 1}};
    cfg["polarization"] = {"HH"};
    cfg["solver"] = {{"type", "PO"}};
    cfg["output"] = {{"path", (dir / "out.h5").string()}, {"format", "hdf5"}};

    const fs::path cfg_file = dir / "config.json";
    {
        std::ofstream out(cfg_file);
        out << cfg.dump(2);
    }

    const auto rc = strikecem::load_config(cfg_file.string(), SCEM_SCHEMA_PATH);
    const auto mesh = strikecem::load_normalized_mesh(rc.value, dir, rc.schema_version);
    EXPECT_GT(mesh.vertices.size(), 50u);
    EXPECT_GT(mesh.triangles.size(), 50u);

    const fs::path out_obj = dir / "missile.obj";
    strikecem::export_mesh(mesh, out_obj);
    ASSERT_TRUE(fs::exists(out_obj));

    const auto obj = strikecem::parse_obj(out_obj);
    EXPECT_EQ(obj.triangles.size(), mesh.triangles.size());

    // CLI export-mesh test
    const fs::path cli_stl = dir / "cli_missile.stl";
    int ret = call_cli({"export-mesh", cfg_file.string(), "-o", cli_stl.string(), "--format", "stl"});
    EXPECT_EQ(ret, 0);
    ASSERT_TRUE(fs::exists(cli_stl));
    const auto stl_tris = strikecem::parse_stl(cli_stl);
    EXPECT_EQ(stl_tris.size(), mesh.triangles.size());

    fs::remove_all(dir, ec);
}

TEST(MeshExport, CLIInvalidFormatFails) {
    const fs::path dir = fs::temp_directory_path() / "scem_export_invalid";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    int ret = call_cli({"export-mesh", fixture("valid_minimal.json"), "-o", (dir / "bad.xyz").string(), "--format", "unsupported_xyz"});
    EXPECT_NE(ret, 0);

    fs::remove_all(dir, ec);
}

} // namespace

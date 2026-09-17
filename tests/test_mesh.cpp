// Mesh pipeline tests: STL/OBJ parsing, reports, transforms, repair
// modes, hashes, and the geometry cache.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

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

} // namespace

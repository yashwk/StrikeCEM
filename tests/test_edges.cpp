// Edge model tests (ADR-0003): rim/crease selection, wedge parameter,
// convexity, threshold policy, and determinism.
#include <cmath>
#include <gtest/gtest.h>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/EdgeModel.hpp"

namespace {

std::string fixture(const std::string& name) { return std::string(SCEM_FIXTURE_DIR) + "/" + name; }

strikecem::NormalizedMesh load_mesh(const std::string& name) {
    const std::string config = fixture(name);
    auto rc = strikecem::load_config(config, SCEM_SCHEMA_PATH);
    return strikecem::load_normalized_mesh(rc.value, SCEM_FIXTURE_DIR, rc.schema_version);
}

TEST(Edges, PlateHasRimsOnly) {
    const auto mesh = load_mesh("valid_minimal.json");
    const auto model = strikecem::extract_edges(mesh);
    ASSERT_EQ(model.edges.size(), 4u);
    EXPECT_EQ(model.smooth_skipped, 1u); // coplanar diagonal
    for (const auto& e : model.edges) {
        EXPECT_TRUE(e.boundary);
        EXPECT_DOUBLE_EQ(e.wedge_n, 2.0);
        EXPECT_TRUE(e.convex);
        EXPECT_DOUBLE_EQ(e.length, 1.0);
        EXPECT_EQ(e.tri1, -1);
        EXPECT_NEAR(e.tangent.length(), 1.0, 1e-12);
    }
}

TEST(Edges, CubeHasTwelveConvexCreases) {
    const auto mesh = load_mesh("valid_cube.json");
    const auto model = strikecem::extract_edges(mesh);
    ASSERT_EQ(model.edges.size(), 12u);
    EXPECT_EQ(model.smooth_skipped, 6u); // face diagonals
    for (const auto& e : model.edges) {
        EXPECT_FALSE(e.boundary);
        EXPECT_TRUE(e.convex);
        EXPECT_NEAR(e.wedge_n, 1.5, 1e-9);
        EXPECT_DOUBLE_EQ(e.length, 1.0);
        EXPECT_GE(e.tri1, 0);
    }
}

TEST(Edges, DihedralValleyIsConcave) {
    const auto mesh = load_mesh("valid_dihedral.json");
    const auto model = strikecem::extract_edges(mesh);
    size_t interior = 0, rims = 0;
    for (const auto& e : model.edges) {
        if (e.boundary) {
            ++rims;
            EXPECT_DOUBLE_EQ(e.wedge_n, 2.0);
        } else {
            ++interior;
            EXPECT_FALSE(e.convex);
            EXPECT_NEAR(e.wedge_n, 1.5, 1e-9);
        }
    }
    EXPECT_EQ(interior, 1u);
    EXPECT_EQ(rims, 6u);
    EXPECT_EQ(model.smooth_skipped, 2u);
}

TEST(Edges, SphereHasNoPhantomEdges) {
    const auto mesh = load_mesh("valid_sphere.json");
    const auto model = strikecem::extract_edges(mesh);
    EXPECT_TRUE(model.edges.empty());
}

TEST(Edges, ThresholdSuppressesCreases) {
    const auto mesh = load_mesh("valid_cube.json");
    const auto model = strikecem::extract_edges(mesh, 2.0);
    EXPECT_TRUE(model.edges.empty());
    EXPECT_EQ(model.smooth_skipped, 18u); // 12 creases + 6 diagonals
}

TEST(Edges, DeterministicAcrossRuns) {
    const auto mesh = load_mesh("valid_cube.json");
    const auto a = strikecem::extract_edges(mesh);
    const auto b = strikecem::extract_edges(mesh);
    ASSERT_EQ(a.edges.size(), b.edges.size());
    for (size_t i = 0; i < a.edges.size(); ++i) {
        EXPECT_EQ(a.edges[i].p0.x, b.edges[i].p0.x);
        EXPECT_EQ(a.edges[i].p1.x, b.edges[i].p1.x);
        EXPECT_DOUBLE_EQ(a.edges[i].wedge_n, b.edges[i].wedge_n);
    }
}

TEST(Edges, NonManifoldThrows) {
    strikecem::NormalizedMesh mesh;
    mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}};
    mesh.triangles = {{0, 1, 2}, {0, 1, 3}, {0, 1, 4}};
    mesh.normals = {{0, 0, 1}, {0, 0, -1}, {0, -1, 0}};
    mesh.areas = {0.5, 0.5, 0.5};
    EXPECT_THROW(strikecem::extract_edges(mesh), std::invalid_argument);
}

} // namespace

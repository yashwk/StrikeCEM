#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/Ray.hpp"

namespace {

std::string fixture(const std::string& name) { return std::string(SCEM_FIXTURE_DIR) + "/" + name; }

uint64_t lcg_next(uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return s;
}

double lcg_unit(uint64_t& s) { return double(lcg_next(s) >> 11) / double(1ULL << 53); }

geom::Vec3d lcg_dir(uint64_t& s) {
    geom::Vec3d d(lcg_unit(s) - 0.5, lcg_unit(s) - 0.5, lcg_unit(s) - 0.5);
    if (d.length() < 1e-9) d = {1, 0, 0};
    d.normalize();
    return d;
}

geom::Vec3d v(double x, double y, double z) { return {x, y, z}; }

TEST(Ray, HitsInteriorWithBarycentrics) {
    const auto hit = strikecem::ray_triangle(v(0.25, 0.25, 1), v(0, 0, -1), v(0, 0, 0),
                                             v(1, 0, 0), v(0, 1, 0), 0.0);
    ASSERT_TRUE(hit.has_value());
    EXPECT_DOUBLE_EQ(hit->t, 1.0);
    EXPECT_DOUBLE_EQ(hit->bary_u, 0.25);
    EXPECT_DOUBLE_EQ(hit->bary_v, 0.25);
}

TEST(Ray, DoubleSided) {
    const auto hit = strikecem::ray_triangle(v(0.25, 0.25, -2), v(0, 0, 1), v(0, 0, 0),
                                             v(1, 0, 0), v(0, 1, 0), 0.0);
    ASSERT_TRUE(hit.has_value());
    EXPECT_DOUBLE_EQ(hit->t, 2.0);
}

TEST(Ray, Misses) {
    const geom::Vec3d a(0, 0, 0), b(1, 0, 0), c(0, 1, 0);
    EXPECT_FALSE(strikecem::ray_triangle(v(0.25, 0.25, 1), v(0, 0, 1), a, b, c, 0.0).has_value());
    EXPECT_FALSE(
        strikecem::ray_triangle(v(2, 2, 1), v(0, 0, -1), a, b, c, 0.0).has_value());
    EXPECT_FALSE(strikecem::ray_triangle(v(0.25, 0.25, 1), v(1, 0, 0), a, b, c, 0.0).has_value());
    EXPECT_FALSE(
        strikecem::ray_triangle(v(0.25, 0.25, -1), v(0, 0, -1), a, b, c, 0.0).has_value());
}

TEST(Ray, EdgeVertexAndTMin) {
    const geom::Vec3d a(0, 0, 0), b(1, 0, 0), c(0, 1, 0);
    EXPECT_TRUE(strikecem::ray_triangle(v(0.5, 0.5, 1), v(0, 0, -1), a, b, c, 0.0).has_value());
    EXPECT_TRUE(strikecem::ray_triangle(v(0, 0, 1), v(0, 0, -1), a, b, c, 0.0).has_value());
    EXPECT_FALSE(strikecem::ray_triangle(v(0.25, 0.25, 1), v(0, 0, -1), a, b, c, 1.0).has_value());
    EXPECT_TRUE(strikecem::ray_triangle(v(0.25, 0.25, 1), v(0, 0, -1), a, b, c, 0.999).has_value());
    const auto deg =
        strikecem::ray_triangle(v(0.25, 0.25, 1), v(0, 0, -1), a, a, c, 0.0);
    EXPECT_FALSE(deg.has_value());
}

strikecem::NormalizedMesh two_plates() {
    strikecem::NormalizedMesh m;
    m.vertices = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                  {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    m.triangles = {{0, 1, 2}, {0, 2, 3}, {4, 6, 5}, {4, 7, 6}};
    m.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, -1}, {0, 0, -1}};
    m.areas.assign(4, 0.5);
    m.report.bbox_min = {0, 0, 0};
    m.report.bbox_max = {1, 1, 1};
    return m;
}

TEST(Ray, MeshClosestHitAndSkip) {
    const auto mesh = two_plates();
    const auto hit = strikecem::ray_mesh(mesh, v(0.5, 0.5, 2), v(0, 0, -1), 0.0, UINT32_MAX);
    ASSERT_TRUE(hit.has_value());
    EXPECT_DOUBLE_EQ(hit->t, 1.0);
    EXPECT_EQ(hit->tri, 2u);
    const auto mid = strikecem::ray_mesh(mesh, v(0.5, 0.5, 0.5), v(0, 0, -1), 0.0, UINT32_MAX);
    ASSERT_TRUE(mid.has_value());
    EXPECT_DOUBLE_EQ(mid->t, 0.5);
    EXPECT_EQ(mid->tri, 0u);
    const auto skipped = strikecem::ray_mesh(mesh, v(0.5, 0.5, 0.5), v(0, 0, -1), 0.0, 0u);
    ASSERT_TRUE(skipped.has_value());
    EXPECT_DOUBLE_EQ(skipped->t, 0.5);
    EXPECT_EQ(skipped->tri, 1u);
    EXPECT_THROW(strikecem::ray_mesh(mesh, v(0, 0, 2), v(0, 0, -2), 0.0, UINT32_MAX),
                 std::invalid_argument);
    EXPECT_THROW(strikecem::ray_mesh(mesh, v(0, 0, 2), v(0, 0, -1), -1.0, UINT32_MAX),
                 std::invalid_argument);
}

TEST(Ray, OcclusionPredicate) {
    const auto mesh = two_plates();
    EXPECT_TRUE(strikecem::occluded(mesh, v(0.5, 0.5, 0), v(0, 0, 1), 10.0, 0u));
    EXPECT_FALSE(strikecem::occluded(mesh, v(0.5, 0.5, 0), v(0, 0, -1), 10.0, 0u));
    EXPECT_FALSE(strikecem::occluded(mesh, v(0.5, 0.5, 0), v(0, 0, 1), 0.5, 0u));
    EXPECT_FALSE(strikecem::occluded(mesh, v(1.0 / 3, 1.0 / 3, 1), v(0, 0, 1), 10.0, 2u));
    EXPECT_FALSE(strikecem::occluded(mesh, v(1.0 / 3, 1.0 / 3, 1), v(0, 0, 1), 10.0, 99u));
    EXPECT_THROW(strikecem::occluded(mesh, v(0, 0, 0), v(0, 0, 2), 1.0, 0u),
                 std::invalid_argument);
}

TEST(Ray, BvhBuildSanity) {
    const auto mesh = two_plates();
    const auto bvh = strikecem::build_bvh(mesh);
    EXPECT_LE(bvh.nodes.size(), 2u * mesh.triangles.size());
    EXPECT_EQ(bvh.order.size(), mesh.triangles.size());
    ASSERT_FALSE(bvh.nodes.empty());
    EXPECT_LE(bvh.nodes[0].bmin.x, 0.0);
    EXPECT_LE(bvh.nodes[0].bmin.y, 0.0);
    EXPECT_LE(bvh.nodes[0].bmin.z, 0.0);
    EXPECT_GE(bvh.nodes[0].bmax.x, 1.0);
    EXPECT_GE(bvh.nodes[0].bmax.y, 1.0);
    EXPECT_GE(bvh.nodes[0].bmax.z, 1.0);
    const strikecem::NormalizedMesh empty;
    const auto none = strikecem::build_bvh(empty);
    EXPECT_TRUE(none.nodes.empty());
    EXPECT_FALSE(
        strikecem::ray_bvh(none, empty, v(0, 0, 2), v(0, 0, -1), 0.0, UINT32_MAX).has_value());
    EXPECT_FALSE(strikecem::occluded_bvh(none, empty, v(0, 0, 0), v(0, 0, 1), 1.0, 0u));
}

void check_bvh_matches_brute(const strikecem::NormalizedMesh& mesh, int rays, uint64_t seed) {
    const auto bvh = strikecem::build_bvh(mesh);
    const geom::Vec3d lo = mesh.report.bbox_min;
    const geom::Vec3d span = mesh.report.bbox_max - mesh.report.bbox_min;
    uint64_t s = seed;
    for (int i = 0; i < rays; ++i) {
        const geom::Vec3d o(lo.x - span.x + 3.0 * span.x * lcg_unit(s),
                            lo.y - span.y + 3.0 * span.y * lcg_unit(s),
                            lo.z - span.z + 3.0 * span.z * lcg_unit(s));
        const geom::Vec3d d = lcg_dir(s);
        const uint32_t skip = (i % 3 == 0) ? UINT32_MAX : uint32_t(i % mesh.triangles.size());
        const auto a = strikecem::ray_mesh(mesh, o, d, 0.0, skip);
        const auto b = strikecem::ray_bvh(bvh, mesh, o, d, 0.0, skip);
        EXPECT_EQ(a.has_value(), b.has_value()) << "i=" << i;
        if (a && b) {
            EXPECT_EQ(a->tri, b->tri) << "i=" << i;
            EXPECT_EQ(a->t, b->t) << "i=" << i;
            EXPECT_EQ(a->bary_u, b->bary_u) << "i=" << i;
            EXPECT_EQ(a->bary_v, b->bary_v) << "i=" << i;
        }
        EXPECT_EQ(strikecem::occluded(mesh, o, d, 10.0, skip),
                  strikecem::occluded_bvh(bvh, mesh, o, d, 10.0, skip))
            << "i=" << i;
    }
}

TEST(Ray, BvhMatchesBrutePlates) { check_bvh_matches_brute(two_plates(), 2000, 12345u); }

TEST(Ray, BvhMatchesBruteSphere) {
    auto rc = strikecem::load_config(fixture("valid_sphere.json"), SCEM_SCHEMA_PATH);
    const auto mesh =
        strikecem::load_normalized_mesh(rc.value, SCEM_FIXTURE_DIR, rc.schema_version);
    check_bvh_matches_brute(mesh, 200, 777u);
}

TEST(Ray, BvhBadInputsThrow) {
    const auto mesh = two_plates();
    const auto bvh = strikecem::build_bvh(mesh);
    EXPECT_THROW(strikecem::ray_bvh(bvh, mesh, v(0, 0, 2), v(0, 0, -2), 0.0, UINT32_MAX),
                 std::invalid_argument);
    EXPECT_THROW(strikecem::ray_bvh(bvh, mesh, v(0, 0, 2), v(0, 0, -1), -1.0, UINT32_MAX),
                 std::invalid_argument);
    EXPECT_THROW(strikecem::occluded_bvh(bvh, mesh, v(0, 0, 0), v(0, 0, 2), 1.0, 0u),
                 std::invalid_argument);
}

}

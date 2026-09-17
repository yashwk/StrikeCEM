#include <cmath>
#include <gtest/gtest.h>

#include "strikecem/solvers/Ray.hpp"

namespace {

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

}

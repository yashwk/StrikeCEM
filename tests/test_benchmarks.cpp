#include <cmath>
#include <complex>
#include <gtest/gtest.h>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/EdgeModel.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace {

constexpr double kPi = 3.141592653589793;

std::string fixture(const std::string& name) { return std::string(SCEM_FIXTURE_DIR) + "/" + name; }

strikecem::NormalizedMesh grid_plate(int n) {
    strikecem::NormalizedMesh m;
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) m.vertices.push_back({double(i) / n, double(j) / n, 0.0});
    auto vid = [&](int i, int j) { return uint32_t(j * (n + 1) + i); };
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            m.triangles.push_back({vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)});
            m.triangles.push_back({vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)});
        }
    m.normals.assign(m.triangles.size(), {0, 0, 1});
    m.areas.assign(m.triangles.size(), 0.5 / (n * n));
    return m;
}

strikecem::ResolvedConfig load_rc(const std::string& name) {
    return strikecem::load_config(fixture(name), SCEM_SCHEMA_PATH);
}

strikecem::SamplePlan single_sample(double freq, double az, double el,
                                    const std::vector<std::string>& pols) {
    strikecem::SamplePlan plan;
    plan.frequencies_hz = {freq};
    plan.directions.push_back({0, az, el, strikecem::direction_from_az_el(az, el)});
    plan.polarizations = pols;
    return plan;
}

TEST(Benchmark, PlateBroadsideFringeIsSmallCorrection) {
    const auto rc = load_rc("valid_minimal.json");
    const auto mesh = grid_plate(100);
    const auto model = strikecem::extract_edges(mesh);
    ASSERT_EQ(model.edges.size(), 400u);
    const strikecem::FringeOptions on{true, &model};
    for (const auto& pol : {"HH", "VV"}) {
        const auto plan = single_sample(3e9, 0.0, -90.0, {pol});
        const auto po = strikecem::solve_po(mesh, plan, rc.value);
        const auto total = strikecem::solve_po(mesh, plan, rc.value, on);
        EXPECT_NEAR(std::abs(po.samples[0].scattering), 1.0 / (strikecem::kSpeedOfLight / 3e9), 0.05);
        EXPECT_LT(std::abs(total.samples[0].scattering - po.samples[0].scattering),
                  0.05 * std::abs(po.samples[0].scattering));
    }
}

TEST(Benchmark, PlateFirstNullFilled) {
    const auto rc = load_rc("valid_minimal.json");
    const auto mesh = grid_plate(100);
    const auto model = strikecem::extract_edges(mesh);
    const strikecem::FringeOptions on{true, &model};
    const double lambda = strikecem::kSpeedOfLight / 3e9;
    const double null_el = -90.0 + strikecem::rad_to_deg(std::asin(lambda / 1.0));
    const auto plan = single_sample(3e9, 0.0, null_el, {"HH"});
    const auto po = strikecem::solve_po(mesh, plan, rc.value);
    const auto total = strikecem::solve_po(mesh, plan, rc.value, on);
    const auto broad = strikecem::solve_po(mesh, single_sample(3e9, 0.0, -90.0, {"HH"}), rc.value);
    const double ref = std::abs(broad.samples[0].scattering);
    EXPECT_LT(std::abs(po.samples[0].scattering), 0.01 * ref);
    EXPECT_GT(std::abs(total.samples[0].scattering), 0.0);
    EXPECT_LT(std::abs(po.samples[0].scattering), 0.5 * std::abs(total.samples[0].scattering));
}

TEST(Benchmark, PlateObliqueCrossPolVanishes) {
    const auto rc = load_rc("valid_minimal.json");
    const auto mesh = grid_plate(100);
    const auto model = strikecem::extract_edges(mesh);
    const strikecem::FringeOptions on{true, &model};
    const auto plan = single_sample(3e9, 0.0, -60.0, {"HH", "VV", "HV", "VH"});
    const auto total = strikecem::solve_po(mesh, plan, rc.value, on);
    ASSERT_EQ(total.samples.size(), 4u);
    const double co = std::max(std::abs(total.samples[0].scattering),
                               std::abs(total.samples[1].scattering));
    EXPECT_GT(co, 0.0);
    EXPECT_LT(std::abs(total.samples[2].scattering), 1e-6 * co);
    EXPECT_LT(std::abs(total.samples[3].scattering), 1e-6 * co);
}

TEST(Benchmark, DihedralYMirrorSymmetric) {
    const auto rc = load_rc("valid_dihedral.json");
    strikecem::NormalizedMesh mesh;
    mesh.vertices = {{0, -0.5, 0}, {1, -0.5, 0}, {1, 0.5, 0}, {0, 0.5, 0}, {0.5, 0, 0},
                     {0, -0.5, 1}, {0, 0.5, 1}, {0, 0, 0.5}};
    mesh.triangles = {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4},
                      {0, 3, 7}, {3, 6, 7}, {6, 5, 7}, {5, 0, 7}};
    mesh.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
                    {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}};
    mesh.areas.assign(8, 0.25);
    const auto model = strikecem::extract_edges(mesh);
    size_t interior = 0;
    for (const auto& e : model.edges) interior += e.boundary ? 0 : 1;
    ASSERT_EQ(interior, 1u);
    const strikecem::FringeOptions on{true, &model};
    auto plan_for = [](double ky) {
        geom::Vec3d k_hat(0.5, ky, -0.7);
        k_hat.normalize();
        double az, el;
        strikecem::az_el_from_direction(k_hat, az, el);
        strikecem::SamplePlan plan;
        plan.frequencies_hz = {10e9};
        plan.directions.push_back({0, az, el, k_hat});
        plan.polarizations = {"HH", "VV"};
        return plan;
    };
    const auto a = strikecem::solve_po(mesh, plan_for(0.3), rc.value, on);
    const auto b = strikecem::solve_po(mesh, plan_for(-0.3), rc.value, on);
    ASSERT_EQ(a.samples.size(), b.samples.size());
    for (size_t i = 0; i < a.samples.size(); ++i) {
        const double ref = std::max(std::abs(a.samples[i].scattering), 1e-300);
        EXPECT_NEAR(std::abs(b.samples[i].scattering), std::abs(a.samples[i].scattering),
                    1e-9 * ref)
            << "i=" << i;
        EXPECT_NEAR(std::abs(b.samples[i].scattering - a.samples[i].scattering), 0.0,
                    1e-9 * ref)
            << "i=" << i;
    }
}

strikecem::NormalizedMesh dihedral_grids(int n) {
    strikecem::NormalizedMesh m;
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) m.vertices.push_back({double(i) / n, double(j) / n, 0.0});
    auto h = [&](int i, int j) { return uint32_t(j * (n + 1) + i); };
    const uint32_t vbase = uint32_t(m.vertices.size());
    for (int j = 0; j <= n; ++j)
        for (int k = 0; k <= n; ++k) m.vertices.push_back({0.0, double(j) / n, double(k) / n});
    auto w = [&](int j, int k) { return vbase + uint32_t(j * (n + 1) + k); };
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            m.triangles.push_back({h(i, j), h(i + 1, j), h(i + 1, j + 1)});
            m.triangles.push_back({h(i, j), h(i + 1, j + 1), h(i, j + 1)});
            m.normals.push_back({0, 0, 1});
            m.normals.push_back({0, 0, 1});
            m.areas.push_back(0.5 / (n * n));
            m.areas.push_back(0.5 / (n * n));
        }
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k) {
            m.triangles.push_back({w(j, k), w(j + 1, k), w(j + 1, k + 1)});
            m.triangles.push_back({w(j, k), w(j + 1, k + 1), w(j, k + 1)});
            m.normals.push_back({1, 0, 0});
            m.normals.push_back({1, 0, 0});
            m.areas.push_back(0.5 / (n * n));
            m.areas.push_back(0.5 / (n * n));
        }
    m.report.bbox_min = {0, 0, 0};
    m.report.bbox_max = {1, 1, 1};
    return m;
}

TEST(Benchmark, TwoBounceDihedralAnalytic) {
    const auto rc = load_rc("valid_dihedral.json");
    const auto mesh = dihedral_grids(10);
    const auto plan = single_sample(1e9, 180.0, -45.0, {"HH", "VV"});
    const auto po = strikecem::solve_po(mesh, plan, rc.value);
    const strikecem::GoOptions go{false, true};
    const auto total = strikecem::solve_po(mesh, plan, rc.value, {}, go);
    ASSERT_EQ(total.samples.size(), 2u);
    const double lambda = strikecem::kSpeedOfLight / 1e9;
    const double analytic = 8.0 * kPi / (lambda * lambda);
    for (size_t i = 0; i < 2; ++i) {
        const double ratio = total.samples[i].rcs_sqm / analytic;
        EXPECT_GT(ratio, 0.8);
        EXPECT_LT(ratio, 1.2);
        EXPECT_LT(std::abs(po.samples[i].scattering), 0.1 * std::abs(total.samples[i].scattering));
    }
    const double mean = 0.5 * (total.samples[0].rcs_sqm + total.samples[1].rcs_sqm);
    EXPECT_LT(std::abs(total.samples[0].rcs_sqm - total.samples[1].rcs_sqm), 0.3 * mean);
}

}

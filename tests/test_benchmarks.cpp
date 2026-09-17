// Plate/dihedral RCS benchmarks (ADR-0003 slice D2c): broadside fringe is
// a small correction, the predicted first PO null is filled by fringe,
// oblique cross-pol vanishes by symmetry, and the dihedral is y-mirror
// symmetric with fringe on. Bounds carry 4x+ headroom over probed values;
// null locations come from aperture theory, not from the implementation.
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

// 1x1 m plate in z=0, n x n quads, normal +z (probe-measured fixture).
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
        // Aperture check: |s| = A/lambda at broadside.
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
    const double lambda = strikecem::kSpeedOfLight / 3e9; // plate is 10 lambda wide
    const double null_el = -90.0 + strikecem::rad_to_deg(std::asin(lambda / 1.0));
    const auto plan = single_sample(3e9, 0.0, null_el, {"HH"});
    const auto po = strikecem::solve_po(mesh, plan, rc.value);
    const auto total = strikecem::solve_po(mesh, plan, rc.value, on);
    const auto broad = strikecem::solve_po(mesh, single_sample(3e9, 0.0, -90.0, {"HH"}), rc.value);
    const double ref = std::abs(broad.samples[0].scattering);
    EXPECT_LT(std::abs(po.samples[0].scattering), 0.01 * ref); // null is deep
    EXPECT_GT(std::abs(total.samples[0].scattering), 0.0);
    // Fringe is the majority of the return at the null.
    EXPECT_LT(std::abs(po.samples[0].scattering), 0.5 * std::abs(total.samples[0].scattering));
}

TEST(Benchmark, PlateObliqueCrossPolVanishes) {
    // y-mirror symmetry kills depolarization, PO and fringe alike: a sign
    // error in the vector assembly would leak here.
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
    // Valley runs along y: ky -> -ky looks are identical, PO and fringe.
    // The mesh below is exactly y-mirror symmetric as sets (each unit
    // quad split into 4 via its fixed center vertex), so any asymmetry
    // would be a solver bug, not mesh quadrature: the diagonal split of
    // dihedral.obj is NOT mirror symmetric, hence a dedicated fixture.
    // Plates span y in [-0.5, 0.5] so the mirror plane runs through the
    // origin: otherwise the plane offset injects a global phase and only
    // magnitudes match.
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
    ASSERT_EQ(interior, 1u); // the valley; rims plus coplanar spokes skipped
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

} // namespace

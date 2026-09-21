#include <cmath>
#include <complex>
#include <gtest/gtest.h>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/EdgeModel.hpp"
#include "strikecem/solvers/GpuPO.hpp"
#include "strikecem/solvers/Material.hpp"
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
    const strikecem::GoOptions go{false, 2};
    const auto total = strikecem::solve_po(mesh, plan, rc.value, {}, go);
    ASSERT_EQ(total.samples.size(), 2u);
    EXPECT_GT(total.chains_fired, 0u);
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

strikecem::NormalizedMesh trihedral_grids(int n) {
    strikecem::NormalizedMesh m;
    auto grid = [&](double ox, double oy, double oz, double ax, double ay, double az, double bx,
                    double by, double bz) {
        const uint32_t base = uint32_t(m.vertices.size());
        for (int j = 0; j <= n; ++j)
            for (int i = 0; i <= n; ++i)
                m.vertices.push_back({ox + (ax * i + bx * j) / n, oy + (ay * i + by * j) / n,
                                      oz + (az * i + bz * j) / n});
        return base;
    };
    auto quad = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d, geom::Vec3d nrm) {
        m.triangles.push_back({a, b, c});
        m.triangles.push_back({a, c, d});
        m.normals.push_back(nrm);
        m.normals.push_back(nrm);
        m.areas.push_back(0.5 / (n * n));
        m.areas.push_back(0.5 / (n * n));
    };
    const uint32_t h = grid(0, 0, 0, 1, 0, 0, 0, 1, 0);
    const uint32_t x = grid(0, 0, 0, 0, 1, 0, 0, 0, 1);
    const uint32_t y = grid(0, 0, 0, 1, 0, 0, 0, 0, 1);
    auto id = [&](uint32_t base, int i, int j) { return base + uint32_t(j * (n + 1) + i); };
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            quad(id(h, i, j), id(h, i + 1, j), id(h, i + 1, j + 1), id(h, i, j + 1), {0, 0, 1});
            quad(id(x, i, j), id(x, i + 1, j), id(x, i + 1, j + 1), id(x, i, j + 1), {1, 0, 0});
            quad(id(y, i, j), id(y, i + 1, j), id(y, i + 1, j + 1), id(y, i, j + 1), {0, 1, 0});
        }
    m.report.bbox_min = {0, 0, 0};
    m.report.bbox_max = {1, 1, 1};
    return m;
}

TEST(Benchmark, TwoBounceTrihedralAnalytic) {
    const auto rc = load_rc("valid_dihedral.json");
    const auto mesh = trihedral_grids(6);
    const double q = 1.0 / std::sqrt(3.0);
    const geom::Vec3d k_hat(-q, -q, -q);
    strikecem::SamplePlan plan;
    plan.frequencies_hz = {1e9};
    plan.directions.push_back({0, 225.0, -35.264389682754654, k_hat});
    plan.polarizations = {"HH", "VV"};
    const auto po = strikecem::solve_po(mesh, plan, rc.value);
    const strikecem::GoOptions two{false, 2};
    const auto pairs = strikecem::solve_po(mesh, plan, rc.value, {}, two);
    ASSERT_EQ(pairs.samples.size(), po.samples.size());
    for (size_t i = 0; i < po.samples.size(); ++i)
        EXPECT_EQ(pairs.samples[i].scattering, po.samples[i].scattering);
    const strikecem::GoOptions three{false, 3};
    const auto total = strikecem::solve_po(mesh, plan, rc.value, {}, three);
    ASSERT_EQ(total.samples.size(), 2u);
    EXPECT_GT(total.chains_fired, 0u);
    const double lambda = strikecem::kSpeedOfLight / 1e9;
    const double analytic = 12.0 * kPi / (lambda * lambda);
    for (size_t i = 0; i < 2; ++i) {
        const double ratio = total.samples[i].rcs_sqm / analytic;
        EXPECT_GT(ratio, 0.8);
        EXPECT_LT(ratio, 1.2);
        EXPECT_LT(std::abs(po.samples[i].scattering), 0.01 * std::abs(total.samples[i].scattering));
    }
}

strikecem::MaterialModel glass_model() {
    strikecem::MaterialModel m;
    m.tag = "glass";
    m.type = strikecem::WallType::Dielectric;
    m.table.push_back({1e9, {{4.0, 0.0}, {1.0, 0.0}}, 0.0});
    return m;
}

TEST(MaterialSolver, PecWallBitIdentical) {
    const auto rc = load_rc("valid_minimal.json");
    const auto mesh = grid_plate(8);
    const strikecem::MaterialOptions pec{true, strikecem::WallType::Pec, {}};
    for (const auto& [az, el] : {std::pair{0.0, -90.0}, std::pair{30.0, -60.0}}) {
        const auto plan = single_sample(3e9, az, el, {"HH", "VV", "HV", "VH"});
        const auto ref = strikecem::solve_po(mesh, plan, rc.value);
        const auto got = strikecem::solve_po(mesh, plan, rc.value, {}, {}, pec);
        ASSERT_EQ(got.samples.size(), ref.samples.size());
        for (size_t i = 0; i < ref.samples.size(); ++i) {
            EXPECT_EQ(got.samples[i].scattering, ref.samples[i].scattering);
            EXPECT_EQ(got.samples[i].rcs_sqm, ref.samples[i].rcs_sqm);
            EXPECT_EQ(got.samples[i].lit_facets, ref.samples[i].lit_facets);
        }
    }
    const auto plan = single_sample(3e9, 0.0, -90.0, {"HH"});
    EXPECT_TRUE(strikecem::solve_po(mesh, plan, rc.value, {}, {}, pec).warnings.empty());
}

TEST(MaterialSolver, DielectricNormalMatchesFresnelPower) {
    const auto rc = load_rc("valid_minimal.json");
    const auto mesh = grid_plate(8);
    const strikecem::MaterialOptions glass{true, strikecem::WallType::Dielectric, glass_model()};
    const auto plan = single_sample(1e9, 0.0, -90.0, {"HH", "VV", "HV", "VH"});
    const auto pec = strikecem::solve_po(mesh, plan, rc.value);
    const auto got = strikecem::solve_po(mesh, plan, rc.value, {}, {}, glass);
    ASSERT_EQ(got.samples.size(), 4u);
    const double co = std::max(std::abs(pec.samples[0].scattering),
                               std::abs(pec.samples[1].scattering));
    EXPECT_NEAR(std::norm(got.samples[0].scattering) / std::norm(pec.samples[0].scattering),
                1.0 / 9.0, 1e-9);
    EXPECT_NEAR(std::norm(got.samples[1].scattering) / std::norm(pec.samples[1].scattering),
                1.0 / 9.0, 1e-9);
    EXPECT_LT(std::abs(got.samples[2].scattering), 1e-9 * co);
    EXPECT_LT(std::abs(got.samples[3].scattering), 1e-9 * co);
}

TEST(MaterialSolver, DielectricObliqueSplitsTeTm) {
    const auto rc = load_rc("valid_minimal.json");
    const auto mesh = grid_plate(8);
    const strikecem::MaterialOptions glass{true, strikecem::WallType::Dielectric, glass_model()};
    const auto plan = single_sample(1e9, 0.0, -60.0, {"HH", "VV", "HV", "VH"});
    const auto pec = strikecem::solve_po(mesh, plan, rc.value);
    const auto got = strikecem::solve_po(mesh, plan, rc.value, {}, {}, glass);
    ASSERT_EQ(got.samples.size(), 4u);
    const double c = std::sqrt(3.0) / 2.0;
    const double ct = std::sqrt(15.0) / 4.0;
    const double r_te = (0.5 * c - ct) / (0.5 * c + ct);
    const double r_tm = (c - 0.5 * ct) / (c + 0.5 * ct);
    const double hh_ratio =
        std::abs(got.samples[0].scattering) / std::abs(pec.samples[0].scattering);
    const double vv_ratio =
        std::abs(got.samples[1].scattering) / std::abs(pec.samples[1].scattering);
    EXPECT_NEAR(hh_ratio, std::abs(r_te), 1e-9);
    EXPECT_NEAR(vv_ratio, std::abs(r_tm), 1e-9);
    const strikecem::ComplexMedium air;
    const strikecem::ComplexMedium wall{{4.0, 0.0}, {1.0, 0.0}};
    const auto f = strikecem::fresnel(air, wall, c);
    EXPECT_NEAR(hh_ratio, std::abs(f.r_te), 1e-12);
    EXPECT_NEAR(vv_ratio, std::abs(f.r_tm), 1e-12);
    const double co = std::max(std::abs(got.samples[0].scattering),
                               std::abs(got.samples[1].scattering));
    EXPECT_LT(std::abs(got.samples[2].scattering), 1e-6 * co);
    EXPECT_LT(std::abs(got.samples[3].scattering), 1e-6 * co);
}

TEST(MaterialSolver, LossyWallMatchesFresnel) {
    const auto rc = load_rc("valid_minimal.json");
    const auto mesh = grid_plate(8);
    strikecem::MaterialModel lossy = glass_model();
    lossy.table[0].medium.eps_r = {4.0, -1.0};
    const strikecem::MaterialOptions lossy_wall{true, strikecem::WallType::Dielectric, lossy};
    const auto plan = single_sample(1e9, 0.0, -90.0, {"HH"});
    const auto pec = strikecem::solve_po(mesh, plan, rc.value);
    const auto got = strikecem::solve_po(mesh, plan, rc.value, {}, {}, lossy_wall);
    const strikecem::ComplexMedium air;
    const auto f = strikecem::fresnel(air, lossy.table[0].medium, 1.0);
    const double ratio =
        std::norm(got.samples[0].scattering) / std::norm(pec.samples[0].scattering);
    EXPECT_NEAR(ratio, std::norm(f.r_te), 1e-9);
    EXPECT_GT(ratio, 0.0);
    EXPECT_LT(ratio, 1.0);
    strikecem::MaterialModel slight = glass_model();
    slight.table[0].medium.eps_r = {4.0, -0.01};
    const strikecem::MaterialOptions slight_wall{true, strikecem::WallType::Dielectric, slight};
    const auto near = strikecem::solve_po(mesh, plan, rc.value, {}, {}, slight_wall);
    const double near_ratio =
        std::norm(near.samples[0].scattering) / std::norm(pec.samples[0].scattering);
    EXPECT_NEAR(near_ratio, 1.0 / 9.0, 0.01 * (1.0 / 9.0));
}

TEST(MaterialSolver, CoatedThinRecoversPec) {
    const auto rc = load_rc("valid_minimal.json");
    const auto mesh = grid_plate(8);
    strikecem::MaterialModel paint;
    paint.tag = "paint";
    paint.type = strikecem::WallType::Coated;
    paint.layers.push_back({1e-6, {{6.25, -0.09}, {1.0, 0.0}}});
    const strikecem::MaterialOptions coated{true, strikecem::WallType::Coated, paint};
    const auto plan = single_sample(1e9, 0.0, -90.0, {"HH", "VV"});
    const auto pec = strikecem::solve_po(mesh, plan, rc.value);
    const auto got = strikecem::solve_po(mesh, plan, rc.value, {}, {}, coated);
    ASSERT_EQ(got.samples.size(), pec.samples.size());
    for (size_t i = 0; i < pec.samples.size(); ++i)
        EXPECT_LT(std::abs(got.samples[i].scattering - pec.samples[i].scattering),
                  1e-3 * std::abs(pec.samples[i].scattering));
}

TEST(MaterialSolver, CoatedHalfWaveAbsentee) {
    const auto rc = load_rc("valid_minimal.json");
    const auto mesh = grid_plate(8);
    strikecem::MaterialModel paint;
    paint.tag = "paint";
    paint.type = strikecem::WallType::Coated;
    paint.layers.push_back({0.25, {{4.0, 0.0}, {1.0, 0.0}}});
    const strikecem::MaterialOptions coated{true, strikecem::WallType::Coated, paint};
    const auto plan = single_sample(strikecem::kSpeedOfLight, 0.0, -90.0, {"HH", "VV"});
    const auto pec = strikecem::solve_po(mesh, plan, rc.value);
    const auto got = strikecem::solve_po(mesh, plan, rc.value, {}, {}, coated);
    ASSERT_EQ(got.samples.size(), pec.samples.size());
    for (size_t i = 0; i < pec.samples.size(); ++i)
        EXPECT_NEAR(std::abs(got.samples[i].scattering) / std::abs(pec.samples[i].scattering),
                    1.0, 1e-9);
}

TEST(MaterialSolver, MatchedWallGivesZero) {
    const auto rc = load_rc("valid_minimal.json");
    const auto mesh = grid_plate(8);
    strikecem::MaterialModel air_wall;
    air_wall.tag = "air";
    air_wall.type = strikecem::WallType::Dielectric;
    air_wall.table.push_back({1e9, {{1.0, 0.0}, {1.0, 0.0}}, 0.0});
    const strikecem::MaterialOptions matched{true, strikecem::WallType::Dielectric, air_wall};
    const auto plan = single_sample(1e9, 0.0, -90.0, {"HH", "VV"});
    const auto pec = strikecem::solve_po(mesh, plan, rc.value);
    const auto got = strikecem::solve_po(mesh, plan, rc.value, {}, {}, matched);
    ASSERT_EQ(got.samples.size(), pec.samples.size());
    for (size_t i = 0; i < pec.samples.size(); ++i)
        EXPECT_LT(std::abs(got.samples[i].scattering),
                  1e-9 * std::abs(pec.samples[i].scattering));
}

TEST(MaterialSolver, Float32TracksFloat64) {
    const auto rc = load_rc("valid_minimal.json");
    const auto mesh = grid_plate(8);
    const strikecem::MaterialOptions glass{true, strikecem::WallType::Dielectric, glass_model()};
    const auto plan = single_sample(1e9, 30.0, -60.0, {"HH", "VV"});
    const auto ref = strikecem::solve_po(mesh, plan, rc.value, {}, {}, glass);
    auto single = rc.value;
    single["solver"]["precision"] = "float32";
    const auto got = strikecem::solve_po(mesh, plan, single, {}, {}, glass);
    ASSERT_EQ(got.samples.size(), ref.samples.size());
    for (size_t i = 0; i < ref.samples.size(); ++i) {
        const double expected = std::abs(ref.samples[i].scattering);
        EXPECT_LT(std::abs(got.samples[i].scattering - ref.samples[i].scattering),
                  1e-5 * std::max(expected, 1e-300));
    }
}

TEST(MaterialSolver, NonPecGuards) {
    const auto rc = load_rc("valid_minimal.json");
    auto mesh = grid_plate(4);
    mesh.report.bbox_min = {0.0, 0.0, 0.0};
    mesh.report.bbox_max = {1.0, 1.0, 0.0};
    const strikecem::MaterialOptions glass{true, strikecem::WallType::Dielectric, glass_model()};
    const auto plan = single_sample(1e9, 0.0, -90.0, {"HH"});
    auto cuda_cfg = rc.value;
    cuda_cfg["execution"]["accelerator"] = "cuda";
    EXPECT_THROW(strikecem::solve_po(mesh, plan, cuda_cfg, {}, {}, glass),
                 strikecem::cuda::CudaError);
    const strikecem::GoOptions chains{false, 2};
    EXPECT_THROW(strikecem::solve_po(mesh, plan, rc.value, {}, chains, glass),
                 std::invalid_argument);
    const strikecem::GoOptions shade{true, 1};
    EXPECT_NO_THROW(strikecem::solve_po(mesh, plan, rc.value, {}, shade, glass));
    const auto edges = strikecem::extract_edges(mesh);
    const strikecem::FringeOptions fringe{true, &edges};
    const auto fringed = strikecem::solve_po(mesh, plan, rc.value, fringe, {}, glass);
    bool warned = false;
    for (const auto& w : fringed.warnings)
        if (w.find("PEC-derived") != std::string::npos) warned = true;
    EXPECT_TRUE(warned);
    strikecem::MaterialModel narrow = glass_model();
    narrow.table.push_back({2e9, {{4.0, 0.0}, {1.0, 0.0}}, 0.0});
    const strikecem::MaterialOptions ranged{true, strikecem::WallType::Dielectric, narrow};
    EXPECT_THROW(strikecem::solve_po(mesh, single_sample(3e9, 0.0, -90.0, {"HH"}), rc.value,
                                     {}, {}, ranged),
                 std::invalid_argument);
    strikecem::MaterialModel mismatched = glass_model();
    mismatched.type = strikecem::WallType::Pec;
    const strikecem::MaterialOptions bad{true, strikecem::WallType::Dielectric, mismatched};
    EXPECT_THROW(strikecem::solve_po(mesh, plan, rc.value, {}, {}, bad), std::invalid_argument);
    strikecem::MaterialModel tabled_coat;
    tabled_coat.tag = "tabled";
    tabled_coat.type = strikecem::WallType::Coated;
    tabled_coat.table.push_back({1e9, {{4.0, 0.0}, {1.0, 0.0}}, 0.0});
    tabled_coat.layers.push_back({0.01, {{4.0, 0.0}, {1.0, 0.0}}});
    const strikecem::MaterialOptions future{true, strikecem::WallType::Coated, tabled_coat};
    EXPECT_THROW(strikecem::solve_po(mesh, plan, rc.value, {}, {}, future),
                 std::invalid_argument);
}

}

// PO reference solver tests: shadowing, symmetry, reciprocity, the ADR-0001
// single-triangle closed form, precision behavior, and determinism.
#include <cmath>
#include <gtest/gtest.h>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"
#include "strikecem/solvers/Ray.hpp"

namespace {

std::string fixture(const std::string& name) { return std::string(SCEM_FIXTURE_DIR) + "/" + name; }

struct Case {
    strikecem::ResolvedConfig rc;
    strikecem::NormalizedMesh mesh;
};

Case load_case(const std::string& name) {
    Case c;
    c.rc = strikecem::load_config(fixture(name), SCEM_SCHEMA_PATH);
    c.mesh = strikecem::load_normalized_mesh(c.rc.value, SCEM_FIXTURE_DIR, c.rc.schema_version);
    return c;
}

strikecem::SamplePlan make_plan(const std::vector<double>& freqs,
                                const std::vector<std::pair<double, double>>& dirs,
                                const std::vector<std::string>& pols) {
    strikecem::SamplePlan plan;
    plan.frequencies_hz = freqs;
    uint32_t id = 0;
    for (const auto& [az, el] : dirs)
        plan.directions.push_back(
            {id++, az, el, strikecem::direction_from_az_el(az, el)});
    plan.polarizations = pols;
    return plan;
}

TEST(PoSolver, DarkSideIsExactlyZero) {
    const Case c = load_case("valid_minimal.json");
    const auto plan = make_plan({10e9}, {{0.0, 90.0}}, {"HH", "VV", "HV", "VH"});
    const auto result = strikecem::solve_po(c.mesh, plan, c.rc.value);
    ASSERT_EQ(result.samples.size(), 4u);
    for (const auto& row : result.samples) {
        EXPECT_EQ(row.lit_facets, 0u);
        EXPECT_DOUBLE_EQ(row.scattering.real(), 0.0);
        EXPECT_DOUBLE_EQ(row.scattering.imag(), 0.0);
        EXPECT_DOUBLE_EQ(row.rcs_sqm, 0.0);
    }
}

TEST(PoSolver, BroadsideLitCountAndPositiveRcs) {
    const Case c = load_case("valid_minimal.json");
    const auto plan = make_plan({10e9}, {{0.0, -90.0}}, {"HH"});
    const auto result = strikecem::solve_po(c.mesh, plan, c.rc.value);
    ASSERT_EQ(result.samples.size(), 1u);
    EXPECT_EQ(result.samples[0].lit_facets, 2u);
    EXPECT_GT(result.samples[0].rcs_sqm, 0.0);
}

TEST(PoSolver, SquarePlateHHEqualsVV) {
    const Case c = load_case("valid_minimal.json");
    const auto plan = make_plan({10e9}, {{0.0, -90.0}}, {"HH", "VV"});
    const auto result = strikecem::solve_po(c.mesh, plan, c.rc.value);
    ASSERT_EQ(result.samples.size(), 2u);
    const auto hh = result.samples[0].scattering;
    const auto vv = result.samples[1].scattering;
    EXPECT_NEAR(std::abs(hh - vv), 0.0, 1e-9 * std::abs(hh));
}

TEST(PoSolver, MonostaticReciprocityHvVh) {
    const Case c = load_case("valid_minimal.json");
    const auto plan = make_plan({10e9}, {{30.0, 20.0}}, {"HV", "VH"});
    const auto result = strikecem::solve_po(c.mesh, plan, c.rc.value);
    ASSERT_EQ(result.samples.size(), 2u);
    const auto hv = result.samples[0].scattering;
    const auto vh = result.samples[1].scattering;
    const double scale = std::max(std::abs(hv), 1e-300);
    EXPECT_NEAR(std::abs(hv - vh), 0.0, 1e-9 * scale);
}

TEST(PoSolver, SingleTriangleClosedForm) {
    // ADR-0001 pin: 0.5 m^2 triangle, broadside, 1 GHz.
    // S_HH = -j*k/4pi, cross-pol null. Fails under the single-cross form.
    const Case c = load_case("valid_mesh_obj.json");
    const double freq = 1e9;
    const double k = 2.0 * 3.141592653589793 * freq / strikecem::kSpeedOfLight;
    const auto plan = make_plan({freq}, {{0.0, -90.0}}, {"HH", "VH"});
    const auto result = strikecem::solve_po(c.mesh, plan, c.rc.value);
    ASSERT_EQ(result.samples.size(), 2u);
    EXPECT_NEAR(result.samples[0].scattering.real(), 0.0, 1e-9);
    EXPECT_NEAR(result.samples[0].scattering.imag(), -k / (4.0 * 3.141592653589793), 1e-9);
    EXPECT_NEAR(std::abs(result.samples[1].scattering), 0.0, 1e-9);
    EXPECT_NEAR(result.samples[0].rcs_sqm, k * k / (4.0 * 3.141592653589793),
                1e-9 * result.samples[0].rcs_sqm);
}

TEST(PoSolver, Float32AgreesWithFloat64) {
    const Case c = load_case("valid_minimal.json");
    const auto plan = make_plan({10e9}, {{30.0, 20.0}}, {"HH", "VV", "HV", "RHCP"});
    const auto ref = strikecem::solve_po(c.mesh, plan, c.rc.value);
    nlohmann::json single = c.rc.value;
    single["solver"]["precision"] = "float32";
    const auto result = strikecem::solve_po(c.mesh, plan, single);
    ASSERT_EQ(result.samples.size(), ref.samples.size());
    for (size_t i = 0; i < ref.samples.size(); ++i) {
        const double expected = std::abs(ref.samples[i].scattering);
        const double err = std::abs(result.samples[i].scattering - ref.samples[i].scattering);
        EXPECT_LT(err, 1e-5 * std::max(expected, 1e-300));
        EXPECT_EQ(result.samples[i].lit_facets, ref.samples[i].lit_facets);
    }
}

TEST(PoSolver, DeterministicAcrossRuns) {
    const Case c = load_case("valid_minimal.json");
    const auto plan = make_plan({10e9}, {{30.0, 20.0}}, {"HH", "VV"});
    const auto a = strikecem::solve_po(c.mesh, plan, c.rc.value);
    const auto b = strikecem::solve_po(c.mesh, plan, c.rc.value);
    ASSERT_EQ(a.samples.size(), b.samples.size());
    for (size_t i = 0; i < a.samples.size(); ++i)
        EXPECT_EQ(a.samples[i].scattering, b.samples[i].scattering);
}

TEST(PoSolver, SampleIdsFollowPlanOrder) {
    const Case c = load_case("valid_minimal.json");
    const auto plan = make_plan({10e9, 11e9}, {{0.0, 0.0}, {90.0, 0.0}}, {"HH", "VV"});
    const auto result = strikecem::solve_po(c.mesh, plan, c.rc.value);
    ASSERT_EQ(result.samples.size(), 8u);
    for (uint64_t i = 0; i < 8; ++i) {
        EXPECT_EQ(result.samples[i].sample_id, i);
        EXPECT_EQ(result.samples[i].frequency_id, i / 4);
        EXPECT_EQ(result.samples[i].direction_id, (i / 2) % 2);
        EXPECT_EQ(result.samples[i].pol_id, i % 2);
    }
}

TEST(PoSolver, ConductivityWarns) {
    const Case c = load_case("valid_minimal.json");
    nlohmann::json lossy = c.rc.value;
    lossy["frequency"]["medium"]["sigma"] = 1.0;
    const auto plan = make_plan({10e9}, {{0.0, -90.0}}, {"HH"});
    EXPECT_TRUE(strikecem::solve_po(c.mesh, plan, lossy).warnings.size() == 1);
    EXPECT_TRUE(strikecem::solve_po(c.mesh, plan, c.rc.value).warnings.empty());
}

// GO shadowing (ADR-0004 slice 2): occlusion-gated PO behind ShadowOptions.
// Schema stays locked (shadowing const false); no CLI changes.
TEST(PoSolver, ShadowReductionPlate) {
    // Nothing to shadow on a lone plate: bit-identical to v1 PO.
    const Case c = load_case("valid_minimal.json");
    const auto plan = make_plan({10e9}, {{0.0, -90.0}, {45.0, -45.0}}, {"HH", "VV"});
    const auto po = strikecem::solve_po(c.mesh, plan, c.rc.value);
    const strikecem::ShadowOptions shadow{true};
    const auto go = strikecem::solve_po(c.mesh, plan, c.rc.value, {}, shadow);
    ASSERT_EQ(po.samples.size(), go.samples.size());
    for (size_t i = 0; i < po.samples.size(); ++i) {
        EXPECT_EQ(go.samples[i].scattering, po.samples[i].scattering);
        EXPECT_EQ(go.samples[i].rcs_sqm, po.samples[i].rcs_sqm);
        EXPECT_EQ(go.samples[i].lit_facets, po.samples[i].lit_facets);
    }
}

TEST(PoSolver, ShadowReductionDihedral) {
    // Steep Q1 look (az 0, el -78.5) lights all four facets with no
    // occlusion: horizontal rays clear the vertical plate above z = 1 and
    // vertical rays climb away from z = 0. Bit-identical to v1 PO.
    const Case c = load_case("valid_dihedral.json");
    const auto plan = make_plan({10e9}, {{0.0, -78.5}}, {"HH"});
    const auto po = strikecem::solve_po(c.mesh, plan, c.rc.value);
    ASSERT_EQ(po.samples[0].lit_facets, 4u);
    const strikecem::ShadowOptions shadow{true};
    const auto go = strikecem::solve_po(c.mesh, plan, c.rc.value, {}, shadow);
    ASSERT_EQ(po.samples.size(), go.samples.size());
    for (size_t i = 0; i < po.samples.size(); ++i)
        EXPECT_EQ(go.samples[i].scattering, po.samples[i].scattering);
}

strikecem::NormalizedMesh stacked_plates() {
    // 1x1 plate at z=1 (tris 0,1) over one at z=0 (tris 2,3), both normal
    // +z: a look from +z lights all four by facing, but the lower pair
    // sits in the upper plate's shadow.
    strikecem::NormalizedMesh m;
    m.vertices = {{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
                  {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    m.triangles = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7}};
    m.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    m.areas.assign(4, 0.5);
    m.report.bbox_min = {0, 0, 0};
    m.report.bbox_max = {1, 1, 1};
    return m;
}

TEST(PoSolver, ShadowBlocksLowerPlate) {
    const Case c = load_case("valid_minimal.json"); // resolved physics/medium only
    const auto mesh = stacked_plates();
    const auto plan = make_plan({10e9}, {{0.0, -90.0}}, {"HH"});
    const auto po = strikecem::solve_po(mesh, plan, c.rc.value);
    ASSERT_EQ(po.samples[0].lit_facets, 4u);
    const strikecem::ShadowOptions shadow{true};
    const auto go = strikecem::solve_po(mesh, plan, c.rc.value, {}, shadow);
    EXPECT_EQ(go.samples[0].lit_facets, 2u); // only the upper pair is lit
    EXPECT_LT(std::abs(go.samples[0].scattering), std::abs(po.samples[0].scattering));
    // Exact: shadowed stack == upper plate alone without shadowing.
    strikecem::NormalizedMesh upper;
    upper.vertices = {{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    upper.triangles = {{0, 2, 1}, {0, 3, 2}};
    upper.normals = {{0, 0, 1}, {0, 0, 1}};
    upper.areas.assign(2, 0.5);
    upper.report.bbox_min = {0, 0, 1};
    upper.report.bbox_max = {1, 1, 1};
    const auto ref = strikecem::solve_po(upper, plan, c.rc.value);
    EXPECT_EQ(go.samples[0].scattering, ref.samples[0].scattering);
}

} // namespace

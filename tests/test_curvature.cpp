#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

#include "strikecem/solvers/Curvature.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"
#include "strikecem/solvers/GpuPO.hpp"
#include "strikecem/geometry/AnalyticGeometry.hpp"
#include "strikecem/core/Config.hpp"

using namespace strikecem;

namespace {

NormalizedMesh make_flat_quad() {
    NormalizedMesh m;
    m.vertices = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0},
        {0.0, 1.0, 0.0}
    };
    m.triangles = {
        {0, 1, 2},
        {0, 2, 3}
    };
    m.normals = {
        {0.0, 0.0, 1.0},
        {0.0, 0.0, 1.0}
    };
    m.areas = {0.5, 0.5};
    m.triangle_tags = {"pec", "pec"};
    m.report.vertex_count = 4;
    m.report.triangle_count = 2;
    m.report.total_area_m2 = 1.0;
    m.report.max_edge_length_m = std::sqrt(2.0);
    m.report.bbox_min = {0.0, 0.0, 0.0};
    m.report.bbox_max = {1.0, 1.0, 0.0};
    return m;
}

SamplePlan make_simple_plan() {
    SamplePlan plan;
    plan.frequencies_hz = {1.0e9}; // 1 GHz
    plan.polarizations = {"VV", "HH"};
    plan.directions.push_back({0, 0.0, 90.0, direction_from_az_el(0.0, 90.0)});
    return plan;
}

ResolvedConfig base_resolved() {
    return load_config(std::string(SCEM_FIXTURE_DIR) + "/valid_minimal.json", SCEM_SCHEMA_PATH);
}

} // namespace

TEST(Curvature, FlatPlateZeroCurvature) {
    NormalizedMesh plate = make_flat_quad();
    auto curvs = estimate_mesh_curvatures(plate);
    ASSERT_EQ(curvs.size(), 2u);
    for (const auto& c : curvs) {
        EXPECT_NEAR(c.mean_curvature, 0.0, 1e-12);
        EXPECT_NEAR(c.max_curvature, 0.0, 1e-12);
        // Notice boundaries are open edges, so is_discontinuous is true
        EXPECT_TRUE(c.is_discontinuous);
    }
}

TEST(Curvature, SphereCurvatureEstimation) {
    AnalyticSphere sphere_spec;
    sphere_spec.radius = 2.0;
    PrimitiveTessellationParams params;
    params.max_edge_length = 0.2;
    params.min_radial_segments = 32;
    params.min_axial_segments = 16;
    NormalizedMesh sphere = tessellate_sphere(sphere_spec, params);

    auto curvs = estimate_mesh_curvatures(sphere);
    ASSERT_EQ(curvs.size(), sphere.triangles.size());

    // Theoretical curvature H = 1 / R = 1 / 2.0 = 0.5
    double expected_H = 1.0 / sphere_spec.radius;
    size_t valid_count = 0;
    double avg_H = 0.0;
    for (const auto& c : curvs) {
        if (!c.is_discontinuous) {
            avg_H += c.mean_curvature;
            ++valid_count;
        }
    }
    ASSERT_GT(valid_count, 0u);
    avg_H /= static_cast<double>(valid_count);

    // On a closed tessellated sphere, discrete estimation approximates 1/R closely
    EXPECT_NEAR(avg_H, expected_H, 0.05);
}

TEST(Curvature, FockCurvatureFactor) {
    double k = 2.0 * std::numbers::pi * 1.0e9 / 299792458.0; // ~20.958 rad/m
    // Flat: H = 0
    auto f_flat = po_curvature_factor(0.0, k, 1.0);
    EXPECT_DOUBLE_EQ(f_flat.real(), 1.0);
    EXPECT_DOUBLE_EQ(f_flat.imag(), 0.0);

    // Curved: H = 0.5, cos_inc = 1.0
    auto f_curv = po_curvature_factor(0.5, k, 1.0);
    EXPECT_DOUBLE_EQ(f_curv.real(), 1.0);
    EXPECT_NEAR(f_curv.imag(), -0.5 / k, 1e-12);

    // Grazing incidence: cos_inc near 0 clamped to 0.1
    auto f_grazing = po_curvature_factor(0.5, k, 0.001);
    EXPECT_DOUBLE_EQ(f_grazing.real(), 1.0);
    EXPECT_NEAR(f_grazing.imag(), -0.5 / (k * 0.1), 1e-12);
}

TEST(Curvature, FlatPlateSolveInvariance) {
    NormalizedMesh plate = make_flat_quad();
    SamplePlan plan = make_simple_plan();

    nlohmann::json base_cfg = base_resolved().value;
    base_cfg["execution"]["accelerator"] = "cpu";
    base_cfg["solver"]["po_options"]["curvature_correction"] = false;

    FringeOptions fringe;
    GoOptions go;
    MaterialOptions mat;

    PoResult res_nocorrection = solve_po(plate, plan, base_cfg, fringe, go, mat);

    nlohmann::json curv_cfg = base_cfg;
    curv_cfg["solver"]["po_options"]["curvature_correction"] = true;

    PoResult res_correction = solve_po(plate, plan, curv_cfg, fringe, go, mat);

    ASSERT_EQ(res_nocorrection.samples.size(), res_correction.samples.size());
    for (size_t i = 0; i < res_nocorrection.samples.size(); ++i) {
        EXPECT_NEAR(res_nocorrection.samples[i].rcs_sqm,
                    res_correction.samples[i].rcs_sqm, 1e-12);
    }
}

TEST(Curvature, SphereSolveCurvatureCorrectionEffect) {
    AnalyticSphere sphere_spec;
    sphere_spec.radius = 1.0;
    PrimitiveTessellationParams params;
    params.max_edge_length = 0.25;
    params.min_radial_segments = 24;
    params.min_axial_segments = 12;
    NormalizedMesh sphere = tessellate_sphere(sphere_spec, params);
    SamplePlan plan = make_simple_plan();

    nlohmann::json cfg_off = base_resolved().value;
    cfg_off["execution"]["accelerator"] = "cpu";
    cfg_off["solver"]["po_options"]["curvature_correction"] = false;

    FringeOptions fringe;
    GoOptions go;
    MaterialOptions mat;

    PoResult res_off = solve_po(sphere, plan, cfg_off, fringe, go, mat);

    nlohmann::json cfg_on = cfg_off;
    cfg_on["solver"]["po_options"]["curvature_correction"] = true;

    PoResult res_on = solve_po(sphere, plan, cfg_on, fringe, go, mat);

    ASSERT_FALSE(res_off.samples.empty());
    ASSERT_FALSE(res_on.samples.empty());
    // Curvature correction modifies RCS due to asymptotic Fock term
    EXPECT_GT(res_on.samples[0].rcs_sqm, 0.0);
    EXPECT_NE(res_on.samples[0].rcs_sqm, res_off.samples[0].rcs_sqm);
}

TEST(Curvature, CudaAcceleratorRejection) {
    NormalizedMesh plate = make_flat_quad();
    SamplePlan plan = make_simple_plan();

    nlohmann::json cfg = base_resolved().value;
    cfg["execution"]["accelerator"] = "cuda";
    cfg["solver"]["po_options"]["curvature_correction"] = true;

    FringeOptions fringe;
    GoOptions go;
    MaterialOptions mat;

    EXPECT_THROW(solve_po(plate, plan, cfg, fringe, go, mat), cuda::CudaError);
}

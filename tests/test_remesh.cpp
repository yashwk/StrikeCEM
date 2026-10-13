#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

#include "strikecem/geometry/Remesh.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"
#include "strikecem/core/Config.hpp"

using namespace strikecem;

namespace {

NormalizedMesh make_coarse_plate() {
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
    m.report.max_edge_length_m = std::sqrt(2.0); // hypotenuse
    m.report.bbox_min = {0.0, 0.0, 0.0};
    m.report.bbox_max = {1.0, 1.0, 0.0};
    return m;
}

} // namespace

TEST(Remesh, WavelengthCalculation) {
    // 1 GHz: c / 1e9 = ~0.299792458 m
    double lambda_1g = compute_wavelength_m(1.0e9);
    EXPECT_NEAR(lambda_1g, 0.299792458, 1e-6);

    // 10 GHz in free space: ~0.0299792458 m
    double lambda_10g = compute_wavelength_m(10.0e9);
    EXPECT_NEAR(lambda_10g, 0.0299792458, 1e-7);

    // In dielectric with eps_r = 4.0: wavelength halved
    double lambda_diel = compute_wavelength_m(1.0e9, 4.0, 1.0);
    EXPECT_NEAR(lambda_diel, lambda_1g / 2.0, 1e-6);

    EXPECT_THROW(compute_wavelength_m(-1.0), std::invalid_argument);
    EXPECT_THROW(compute_wavelength_m(1.0e9, -1.0, 1.0), std::invalid_argument);
}

TEST(Remesh, ConformingAdaptiveSubdivision) {
    NormalizedMesh plate = make_coarse_plate();
    EXPECT_EQ(plate.triangles.size(), 2u);
    EXPECT_NEAR(plate.report.total_area_m2, 1.0, 1e-12);

    // Target edge length 0.8 (hypotenuse ~1.414 exceeds 0.8)
    NormalizedMesh subdivided = remesh_target_edge_length(plate, 0.8, 5);
    EXPECT_GT(subdivided.triangles.size(), 2u);
    EXPECT_NEAR(subdivided.report.total_area_m2, 1.0, 1e-12); // Exact area conservation
    EXPECT_LE(subdivided.report.max_edge_length_m, 0.8 + 1e-6);

    // Finer target 0.4
    NormalizedMesh finer = remesh_target_edge_length(plate, 0.4, 5);
    EXPECT_GT(finer.triangles.size(), subdivided.triangles.size());
    EXPECT_NEAR(finer.report.total_area_m2, 1.0, 1e-12);
    EXPECT_LE(finer.report.max_edge_length_m, 0.4 + 1e-6);
}

TEST(Remesh, PolicyIntegration) {
    NormalizedMesh plate = make_coarse_plate();

    // Policy: user_mesh (default) -> mesh unchanged
    nlohmann::json cfg_user = {
        {"mesh", {
            {"frequency_policy", "user_mesh"}
        }}
    };
    NormalizedMesh res_user = remesh_wavelength(plate, cfg_user, {10.0e9});
    EXPECT_EQ(res_user.triangles.size(), plate.triangles.size());
    EXPECT_DOUBLE_EQ(res_user.report.max_edge_length_m, plate.report.max_edge_length_m);

    // Policy: mesh_at_max_frequency at 3 GHz
    // lambda = 0.1 m, alpha_smooth = 1/6 -> h_target = 0.01667 m
    nlohmann::json cfg_wld = {
        {"mesh", {
            {"frequency_policy", "mesh_at_max_frequency"},
            {"wavelength_adaptive", {
                {"alpha_smooth", 0.5}, // h_target = 0.5 * 0.1 = 0.05 m
                {"alpha_edge", 0.1}
            }}
        }}
    };
    NormalizedMesh res_wld = remesh_wavelength(plate, cfg_wld, {3.0e9});
    EXPECT_GT(res_wld.triangles.size(), plate.triangles.size());
    EXPECT_NEAR(res_wld.report.total_area_m2, 1.0, 1e-12);
    EXPECT_LE(res_wld.report.max_edge_length_m, 0.05 + 1e-5);
}

TEST(Remesh, EndToEndPoSolveWithSubdivision) {
    NormalizedMesh plate = make_coarse_plate();

    SamplePlan plan;
    plan.frequencies_hz = {3.0e9};
    plan.polarizations = {"HH"};
    Direction dir;
    dir.azimuth_deg = 0.0;
    dir.elevation_deg = -90.0; // normal incidence on front face
    dir.k_hat = direction_from_az_el(0.0, -90.0);
    plan.directions.push_back(dir);

    // Remesh to lambda / 4
    double lambda = compute_wavelength_m(3.0e9);
    NormalizedMesh fine_plate = remesh_target_edge_length(plate, lambda * 0.25);

    nlohmann::json cfg = {
        {"scem_schema_version", "1.0"},
        {"solver", {{"type", "PO"}, {"precision", "float64"}}},
        {"frequency", {{"medium", {{"epsilon_r", 1.0}, {"mu_r", 1.0}, {"sigma", 0.0}}}}},
        {"physics", {{"incident_amplitude", 1.0}, {"phase_reference", {0.0, 0.0, 0.0}}}},
        {"execution", {{"accelerator", "cpu"}, {"cpu_threads", 1}}}
    };

    FringeOptions fringe;
    fringe.enabled = false;
    GoOptions go;
    go.shadowing = false;
    go.max_bounces = 1;
    MaterialOptions mat;
    mat.enabled = false;

    PoResult res = solve_po(fine_plate, plan, cfg, fringe, go, mat);
    ASSERT_EQ(res.samples.size(), 1u);

    // Theoretical normal incidence plate RCS: sigma = 4 * pi * A^2 / lambda^2
    // A = 1.0 m^2, lambda = 0.09993 m -> sigma = 4 * pi * 1.0 / (0.09993^2) ~= 1258.4 m^2
    double sigma_theory = 4.0 * std::numbers::pi * (1.0 * 1.0) / (lambda * lambda);
    EXPECT_NEAR(res.samples[0].rcs_sqm, sigma_theory, sigma_theory * 0.05); // Within 5%
}

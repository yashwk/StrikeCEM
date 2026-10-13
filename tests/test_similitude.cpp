#include <gtest/gtest.h>
#include <cmath>
#include <complex>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"
#include "strikecem/solvers/Similitude.hpp"

using namespace strikecem;

namespace {

std::string fixture(const std::string& name) {
    return std::string(SCEM_FIXTURE_DIR) + "/" + name;
}

NormalizedMesh make_flat_plate(double width, double height) {
    NormalizedMesh m;
    m.vertices = {
        {0.0, 0.0, 0.0},
        {width, 0.0, 0.0},
        {width, height, 0.0},
        {0.0, height, 0.0}
    };
    m.triangles = {
        {0, 1, 2},
        {0, 2, 3}
    };
    m.normals = {
        {0.0, 0.0, 1.0},
        {0.0, 0.0, 1.0}
    };
    double half_area = 0.5 * width * height;
    m.areas = {half_area, half_area};
    m.triangle_tags = {"pec", "pec"};
    m.report.vertex_count = 4;
    m.report.triangle_count = 2;
    m.report.total_area_m2 = width * height;
    m.report.max_edge_length_m = std::sqrt(width * width + height * height);
    m.report.bbox_min = {0.0, 0.0, 0.0};
    m.report.bbox_max = {width, height, 0.0};
    return m;
}

SamplePlan make_plan(double freq_hz, double az, double el) {
    SamplePlan plan;
    plan.frequencies_hz = {freq_hz};
    plan.polarizations = {"HH", "VV"};
    plan.directions.push_back({0, az, el, direction_from_az_el(az, el)});
    return plan;
}

} // namespace

TEST(Similitude, ScalingMath) {
    const double sg = 0.5; // 1:2 scale model
    const double f_full = 10.0e9; // 10 GHz
    const double lambda_full = 299792458.0 / f_full;

    // Frequency scaling: FullToModel -> f_model = f_full / sg = 20 GHz
    EXPECT_DOUBLE_EQ(scale_frequency(f_full, sg, SimilitudeDirection::FullToModel), 20.0e9);
    // ModelToFull -> f_full = f_model * sg = 10 GHz
    EXPECT_DOUBLE_EQ(scale_frequency(20.0e9, sg, SimilitudeDirection::ModelToFull), 10.0e9);

    // Wavelength scaling: FullToModel -> lambda_model = lambda_full * sg
    EXPECT_DOUBLE_EQ(scale_wavelength(lambda_full, sg, SimilitudeDirection::FullToModel), lambda_full * 0.5);
    // ModelToFull -> lambda_full = lambda_model / sg
    EXPECT_DOUBLE_EQ(scale_wavelength(lambda_full * 0.5, sg, SimilitudeDirection::ModelToFull), lambda_full);

    // RCS sqm scaling: FullToModel -> sigma_full = sigma_model / (sg^2) = sigma_model * 4
    double rcs_model_sqm = 2.5;
    EXPECT_DOUBLE_EQ(rescale_rcs_sqm(rcs_model_sqm, sg, SimilitudeDirection::FullToModel), 10.0);
    EXPECT_DOUBLE_EQ(rescale_rcs_sqm(rcs_model_sqm, sg, SimilitudeDirection::ModelToFull), 10.0);

    // RCS dBsm scaling: +20 * log10(1 / 0.5) = +20 * log10(2) approx +6.0205999 dB
    double rcs_model_dbsm = 10.0;
    double expected_offset = 20.0 * std::log10(2.0);
    EXPECT_NEAR(rescale_rcs_dbsm(rcs_model_dbsm, sg, SimilitudeDirection::FullToModel),
                rcs_model_dbsm + expected_offset, 1e-9);

    // Complex scattering amplitude scaling: S_full = S_model / sg = S_model * 2
    std::complex<double> s_model(3.0, 4.0);
    auto s_full = rescale_scattering_amplitude(s_model, sg, SimilitudeDirection::FullToModel);
    EXPECT_DOUBLE_EQ(s_full.real(), 6.0);
    EXPECT_DOUBLE_EQ(s_full.imag(), 8.0);
}

TEST(Similitude, ScaleMesh) {
    NormalizedMesh full_mesh = make_flat_plate(2.0, 1.0);
    double sg = 0.5;

    NormalizedMesh scaled = scale_mesh(full_mesh, sg, SimilitudeDirection::FullToModel);

    EXPECT_EQ(scaled.vertices.size(), 4u);
    EXPECT_DOUBLE_EQ(scaled.vertices[1].x, 1.0);
    EXPECT_DOUBLE_EQ(scaled.vertices[2].x, 1.0);
    EXPECT_DOUBLE_EQ(scaled.vertices[2].y, 0.5);

    // Area should scale as sg^2 = 0.25
    EXPECT_DOUBLE_EQ(scaled.areas[0], full_mesh.areas[0] * 0.25);
    EXPECT_DOUBLE_EQ(scaled.areas[1], full_mesh.areas[1] * 0.25);
    EXPECT_DOUBLE_EQ(scaled.report.total_area_m2, full_mesh.report.total_area_m2 * 0.25);

    // Bounding box
    EXPECT_DOUBLE_EQ(scaled.report.bbox_max.x, 1.0);
    EXPECT_DOUBLE_EQ(scaled.report.bbox_max.y, 0.5);
}

TEST(Similitude, ConfigParsing) {
    nlohmann::json j = {
        {"similitude", {
            {"enabled", true},
            {"geometric_scale", 0.2},
            {"direction", "full_to_model"},
            {"auto_rescale_rcs", true}
        }}
    };

    SimilitudeConfig cfg = parse_similitude_config(j);
    EXPECT_TRUE(cfg.enabled);
    EXPECT_DOUBLE_EQ(cfg.geometric_scale, 0.2);
    EXPECT_EQ(cfg.direction, SimilitudeDirection::FullToModel);
    EXPECT_TRUE(cfg.auto_rescale_rcs);

    // Missing block returns disabled
    nlohmann::json empty_j = nlohmann::json::object();
    SimilitudeConfig empty_cfg = parse_similitude_config(empty_j);
    EXPECT_FALSE(empty_cfg.enabled);
}

TEST(Similitude, EndToEndEquivalence) {
    // Sinclair electrodynamic similitude theorem test:
    // A 1m x 1m flat plate at 10 GHz broadside has identical rescaled RCS
    // as a 0.5m x 0.5m model plate at 20 GHz.
    auto rc = load_config(fixture("valid_minimal.json"), SCEM_SCHEMA_PATH);
    NormalizedMesh full_plate = make_flat_plate(1.0, 1.0);
    SamplePlan full_plan = make_plan(10.0e9, 0.0, -90.0); // broadside lit

    PoResult full_result = solve_po(full_plate, full_plan, rc.value);
    ASSERT_EQ(full_result.samples.size(), 2u);

    // Now set up scale model
    SimilitudeConfig sim_cfg;
    sim_cfg.enabled = true;
    sim_cfg.geometric_scale = 0.5;
    sim_cfg.direction = SimilitudeDirection::FullToModel;
    sim_cfg.auto_rescale_rcs = true;

    NormalizedMesh model_mesh = scale_mesh(full_plate, sim_cfg.geometric_scale, sim_cfg.direction);
    SamplePlan model_plan = make_plan(scale_frequency(10.0e9, sim_cfg.geometric_scale, sim_cfg.direction),
                                      0.0, -90.0);

    PoResult model_result = solve_po(model_mesh, model_plan, rc.value);
    ASSERT_EQ(model_result.samples.size(), 2u);

    // Raw model RCS should be 1/4 of full-scale RCS
    EXPECT_NEAR(model_result.samples[0].rcs_sqm, full_result.samples[0].rcs_sqm / 4.0, 1e-6);

    // Rescale model result back to full-scale domain
    rescale_po_result(model_result, sim_cfg);

    // Rescaled model RCS must match full-scale RCS exactly!
    for (size_t i = 0; i < full_result.samples.size(); ++i) {
        EXPECT_NEAR(model_result.samples[i].rcs_sqm, full_result.samples[i].rcs_sqm, 1e-6);
        EXPECT_NEAR(model_result.samples[i].scattering.real(),
                    full_result.samples[i].scattering.real(), 1e-6);
        EXPECT_NEAR(model_result.samples[i].scattering.imag(),
                    full_result.samples[i].scattering.imag(), 1e-6);
    }
}

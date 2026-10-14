#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>

#include "strikecem/analysis/SarImaging.hpp"
#include "strikecem/io/MeshLoader.hpp"

namespace fs = std::filesystem;
using namespace strikecem;
using namespace strikecem::analysis;

TEST(SarImaging, PresetConfigurations) {
    auto sat_ka = SarProcessor::make_preset("satellite-ka");
    EXPECT_EQ(sat_ka.platform, SarPlatform::Spaceborne);
    EXPECT_EQ(sat_ka.mode, SarMode::Spotlight);
    EXPECT_DOUBLE_EQ(sat_ka.center_frequency_hz, 35.0e9);
    EXPECT_DOUBLE_EQ(sat_ka.bandwidth_hz, 1.5e9);
    EXPECT_DOUBLE_EQ(sat_ka.altitude_m, 500.0e3);
    EXPECT_DOUBLE_EQ(sat_ka.incidence_angle_deg, 40.0);
    EXPECT_TRUE(sat_ka.enable_shadow);
    EXPECT_EQ(sat_ka.colormap, "gray");

    auto sat_x = SarProcessor::make_preset("satellite-x");
    EXPECT_EQ(sat_x.platform, SarPlatform::Spaceborne);
    EXPECT_DOUBLE_EQ(sat_x.center_frequency_hz, 9.6e9);
    EXPECT_DOUBLE_EQ(sat_x.bandwidth_hz, 600.0e6);

    auto air_ka = SarProcessor::make_preset("airborne-ka");
    EXPECT_EQ(air_ka.platform, SarPlatform::Airborne);
    EXPECT_DOUBLE_EQ(air_ka.center_frequency_hz, 35.0e9);
    EXPECT_DOUBLE_EQ(air_ka.altitude_m, 8.0e3);

    auto air_x = SarProcessor::make_preset("airborne-x");
    EXPECT_EQ(air_x.platform, SarPlatform::Airborne);
    EXPECT_DOUBLE_EQ(air_x.center_frequency_hz, 10.0e9);
    EXPECT_DOUBLE_EQ(air_x.altitude_m, 12.0e3);
}

TEST(SarImaging, PhysicalResolutions) {
    SarConfig cfg = SarProcessor::make_preset("satellite-ka");
    NormalizedMesh mesh;
    // Single triangle
    mesh.vertices = {{0,0,1}, {1,0,1}, {0,1,1}};
    mesh.triangles = {{0, 1, 2}};
    mesh.normals = {{0, 0, 1}};
    mesh.areas = {0.5};
    mesh.report.bbox_min = {0,0,0};
    mesh.report.bbox_max = {1,1,1};

    SarImage img = SarProcessor::compute(mesh, cfg);

    // Slant range resolution: c / (2 * 1.5e9) = 299792458 / 3.0e9 = 0.09993 m (~10 cm)
    EXPECT_NEAR(img.slant_range_resolution_m, 0.10, 0.005);

    // Ground range resolution: slant_res / sin(40 deg) = 0.10 / 0.64278 = 0.155 m
    EXPECT_NEAR(img.ground_range_resolution_m, 0.155, 0.01);

    // Azimuth resolution: lambda / (2 * delta_theta)
    // lambda = c / 35e9 = 0.008565 m. delta_theta = 2.5 deg = 0.04363 rad.
    // az_res = 0.008565 / (2 * 0.04363) = 0.098 m (~10 cm)
    EXPECT_NEAR(img.azimuth_resolution_m, 0.098, 0.01);
}

TEST(SarImaging, GroundShadowProjectionGeometry) {
    NormalizedMesh mesh;
    // Elevated flat plate at z = 2.0 m, spanning x in [-1, 1], y in [-1, 1]
    mesh.vertices = {
        {-1.0, -1.0, 2.0},
        { 1.0, -1.0, 2.0},
        { 1.0,  1.0, 2.0},
        {-1.0,  1.0, 2.0}
    };
    mesh.triangles = {
        {0, 1, 2},
        {0, 2, 3}
    };
    mesh.report.bbox_min = {-1.0, -1.0, 2.0};
    mesh.report.bbox_max = { 1.0,  1.0, 2.0};

    // Look vector from negative x at 45 deg incidence: k_hat = [sin(45), 0, -cos(45)]
    double s45 = std::sin(M_PI / 4.0);
    geom::Vec3d k_hat{s45, 0.0, -s45};
    double ground_z = 0.0;

    // Shadow cast: delta_z = 2.0 m.
    // L_shadow = delta_z * tan(45) = 2.0 m in +x direction.
    // Plate shadow on ground should span x in [-1 + 2, 1 + 2] = [1, 3], y in [-1, 1]
    size_t nx = 100, ny = 100;
    auto mask = SarProcessor::compute_ground_shadow_mask(mesh, k_hat, ground_z,
                                                         -2.0, 6.0, nx,
                                                         -2.0, 2.0, ny);

    // Point at (x=2.0, y=0.0) should definitely be inside shadow!
    int i_shadow = static_cast<int>((2.0 - (-2.0)) / (8.0 / nx));
    int j_shadow = static_cast<int>((0.0 - (-2.0)) / (4.0 / ny));
    EXPECT_TRUE(mask[i_shadow][j_shadow]);

    // Point at (x=-1.5, y=0.0) should be outside shadow (facing radar)
    int i_lit = static_cast<int>((-1.5 - (-2.0)) / (8.0 / nx));
    EXPECT_FALSE(mask[i_lit][j_shadow]);
}

TEST(SarImaging, SarImageGenerationAndJson) {
    NormalizedMesh mesh;
    // Simple box-like target
    mesh.vertices = {
        {-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0},
        {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}
    };
    mesh.triangles = {
        {0, 1, 2}, {0, 2, 3}, // bottom
        {4, 5, 6}, {4, 6, 7}, // top
        {0, 1, 5}, {0, 5, 4}, // front
        {2, 3, 7}, {2, 7, 6}  // back
    };
    mesh.normals.resize(mesh.triangles.size(), {0, 0, 1});
    mesh.areas.resize(mesh.triangles.size(), 2.0);
    mesh.report.bbox_min = {-1, -1, 0};
    mesh.report.bbox_max = { 1,  1, 1};

    SarConfig cfg = SarProcessor::make_preset("satellite-ka");
    cfg.grid_samples_range = 128;
    cfg.grid_samples_azimuth = 128;

    SarImage img = SarProcessor::compute(mesh, cfg, "BoxVehicle");

    EXPECT_EQ(img.intensity_db.size(), 128);
    EXPECT_EQ(img.intensity_db[0].size(), 128);
    EXPECT_GT(img.shadow_area_fraction, 0.0);
    EXPECT_DOUBLE_EQ(img.peak_rcs_dbsm, 10.0 * std::log10(std::pow(10.0, img.peak_rcs_dbsm / 10.0)));

    nlohmann::json j = img.to_json();
    EXPECT_EQ(j["vehicle_name"], "BoxVehicle");
    EXPECT_EQ(j["platform"], "spaceborne");
    EXPECT_TRUE(j.contains("resolutions"));
    EXPECT_TRUE(j.contains("peak_scatterer"));

    const fs::path json_tmp = fs::temp_directory_path() / "test_sar.json";
    img.write_json(json_tmp.string());
    EXPECT_TRUE(fs::exists(json_tmp));
    fs::remove(json_tmp);
}

TEST(SarImaging, ImageRenderingAndFileExport) {
    NormalizedMesh mesh;
    mesh.vertices = {{-2, -1, 0}, {2, -1, 0}, {0, 2, 1}};
    mesh.triangles = {{0, 1, 2}};
    mesh.normals = {{0, 0, 1}};
    mesh.areas = {4.0};
    mesh.report.bbox_min = {-2, -1, 0};
    mesh.report.bbox_max = { 2,  2, 1};

    SarConfig cfg = SarProcessor::make_preset("satellite-ka");
    cfg.grid_samples_range = 64;
    cfg.grid_samples_azimuth = 64;

    SarImage img = SarProcessor::compute(mesh, cfg);

    // Test military satellite grayscale rendering
    rendering::ImageBuffer gray_buf = img.render_image(400, 300, "gray");
    EXPECT_EQ(gray_buf.width(), 400);
    EXPECT_EQ(gray_buf.height(), 300);

    // Test scientific turbo colormap rendering
    rendering::ImageBuffer turbo_buf = img.render_image(400, 300, "turbo");
    EXPECT_EQ(turbo_buf.width(), 400);
    EXPECT_EQ(turbo_buf.height(), 300);

    const fs::path png_tmp = fs::temp_directory_path() / "test_sar_render.png";
    img.write_image(png_tmp.string(), 400, 300, "gray");
    EXPECT_TRUE(fs::exists(png_tmp));
    EXPECT_GT(fs::file_size(png_tmp), 500);
    fs::remove(png_tmp);
}

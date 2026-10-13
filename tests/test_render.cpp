#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include "strikecem/analysis/IsarImaging.hpp"
#include "strikecem/diagnostics/VtkWriter.hpp"
#include "strikecem/geometry/AnalyticGeometry.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/rendering/Colormap.hpp"
#include "strikecem/rendering/Image.hpp"
#include "strikecem/rendering/ModelRenderer.hpp"

namespace strikecem {
namespace {

namespace fs = std::filesystem;

} // namespace

TEST(Rendering, ImageBufferPrimitivesAndPngExport) {
    rendering::ImageBuffer img(200, 150, {20, 20, 30, 255});
    EXPECT_EQ(img.width(), 200u);
    EXPECT_EQ(img.height(), 150u);

    // Draw primitives
    img.draw_rect(10, 10, 50, 40, {255, 0, 0, 255}, true);
    img.draw_circle(100, 75, 25, {0, 255, 0, 255}, true);
    img.draw_line(0, 0, 199, 149, {0, 0, 255, 255});
    img.draw_text(10, 120, "STRIKECEM TEST", {255, 255, 255, 255}, 1);

    // Pixel checks
    const rendering::Rgba p_rect = img.get_pixel(20, 20);
    EXPECT_EQ(p_rect.r, 255);
    EXPECT_EQ(p_rect.g, 0);

    const rendering::Rgba p_circ = img.get_pixel(110, 75);
    EXPECT_EQ(p_circ.r, 0);
    EXPECT_EQ(p_circ.g, 255);

    // Export PNG
    const fs::path png_path = fs::temp_directory_path() / "test_render.png";
    img.write_png(png_path.string());
    EXPECT_TRUE(fs::exists(png_path));
    EXPECT_GT(fs::file_size(png_path), 100u);

    // Verify PNG magic bytes
    std::ifstream in(png_path, std::ios::binary);
    uint8_t sig[8] = {0};
    in.read(reinterpret_cast<char*>(sig), 8);
    EXPECT_EQ(sig[0], 0x89);
    EXPECT_EQ(sig[1], 'P');
    EXPECT_EQ(sig[2], 'N');
    EXPECT_EQ(sig[3], 'G');

    fs::remove(png_path);
}

TEST(Rendering, BmpAndPpmExport) {
    rendering::ImageBuffer img(64, 64, {100, 150, 200, 255});
    img.draw_rect(16, 16, 32, 32, {255, 255, 0, 255}, true);

    const fs::path bmp_path = fs::temp_directory_path() / "test_render.bmp";
    img.write_bmp(bmp_path.string());
    EXPECT_TRUE(fs::exists(bmp_path));
    EXPECT_GT(fs::file_size(bmp_path), 54u);

    std::ifstream in_bmp(bmp_path, std::ios::binary);
    char bm[2] = {0};
    in_bmp.read(bm, 2);
    EXPECT_EQ(bm[0], 'B');
    EXPECT_EQ(bm[1], 'M');
    fs::remove(bmp_path);

    const fs::path ppm_path = fs::temp_directory_path() / "test_render.ppm";
    img.write_ppm(ppm_path.string());
    EXPECT_TRUE(fs::exists(ppm_path));
    EXPECT_GT(fs::file_size(ppm_path), 15u);
    fs::remove(ppm_path);
}

TEST(Rendering, ColormapsEvaluation) {
    const auto c_blue = rendering::evaluate_colormap(rendering::ColormapType::Turbo, 0.2);
    const auto c_green = rendering::evaluate_colormap(rendering::ColormapType::Turbo, 0.5);
    const auto c_red = rendering::evaluate_colormap(rendering::ColormapType::Turbo, 0.9);

    // Blue at 0.2, Greenish at 0.5, Red/Maroon at 0.9
    EXPECT_GT(c_blue.b, c_blue.r);
    EXPECT_GT(c_green.g, c_green.b);
    EXPECT_GT(c_red.r, c_red.b);

    const auto j = rendering::evaluate_colormap(rendering::ColormapType::Jet, 0.5);
    EXPECT_GT(j.g, 100);

    const auto v = rendering::evaluate_colormap(rendering::ColormapType::Viridis, 0.5);
    EXPECT_GT(v.g, 100);
}

TEST(Rendering, IsarHeatmapImageGeneration) {
    const size_t M = 16;
    std::vector<double> freqs(M);
    for (size_t i = 0; i < M; ++i) freqs[i] = 10.0e9 + i * 1e8;

    const size_t N = 16;
    std::vector<double> az(N);
    for (size_t i = 0; i < N; ++i) az[i] = -4.0 + i * 0.5;

    std::vector<std::vector<std::complex<double>>> S(M, std::vector<std::complex<double>>(N, {1.0, 0.0}));
    const analysis::IsarImage isar = analysis::IsarProcessor::compute_from_matrix(freqs, az, S);

    analysis::IsarRenderOptions ropts;
    ropts.width = 400;
    ropts.height = 400;
    ropts.colormap = rendering::ColormapType::Turbo;

    const rendering::ImageBuffer img = isar.render_image(ropts);
    EXPECT_EQ(img.width(), 400u);
    EXPECT_EQ(img.height(), 400u);

    const fs::path out_path = fs::temp_directory_path() / "test_isar_heatmap.png";
    isar.write_image(out_path.string(), ropts);
    EXPECT_TRUE(fs::exists(out_path));
    EXPECT_GT(fs::file_size(out_path), 500u);
    fs::remove(out_path);
}

TEST(Rendering, ModelRendererVisualizations) {
    AnalyticSphere sph;
    sph.radius = 1.0;
    const NormalizedMesh mesh = tessellate_sphere(sph);

    Direction dir;
    dir.azimuth_deg = 0.0;
    dir.elevation_deg = 30.0;
    dir.k_hat = direction_from_az_el(0.0, 30.0);

    const VtkMeshDiagnostics diag = compute_facet_diagnostics(mesh, dir, 10.0e9, "HH");

    rendering::ModelRenderConfig cfg;
    cfg.width = 400;
    cfg.height = 300;

    // 1. Shaded CAD mode
    cfg.mode = rendering::RenderShadingMode::ShadedCad;
    const rendering::ImageBuffer img_cad = rendering::ModelRenderer::render_diagnostics(
        mesh, diag, dir, 10.0e9, "HH", cfg);
    EXPECT_EQ(img_cad.width(), 400u);
    EXPECT_EQ(img_cad.height(), 300u);

    // 2. Lit/Shadow mode
    cfg.mode = rendering::RenderShadingMode::LitShadow;
    const rendering::ImageBuffer img_lit = rendering::ModelRenderer::render_diagnostics(
        mesh, diag, dir, 10.0e9, "HH", cfg);
    EXPECT_EQ(img_lit.width(), 400u);

    // 3. Surface Currents mode
    cfg.mode = rendering::RenderShadingMode::SurfaceCurrents;
    const fs::path out_cur = fs::temp_directory_path() / "test_model_currents.png";
    rendering::ModelRenderer::render_and_save(out_cur.string(), mesh, diag, dir, 10.0e9, "HH", cfg);
    EXPECT_TRUE(fs::exists(out_cur));
    EXPECT_GT(fs::file_size(out_cur), 1000u);
    fs::remove(out_cur);
}

} // namespace strikecem

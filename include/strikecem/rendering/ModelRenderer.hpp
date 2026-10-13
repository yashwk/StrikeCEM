#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "strikecem/core/Config.hpp"
#include "strikecem/diagnostics/VtkWriter.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/rendering/Colormap.hpp"
#include "strikecem/rendering/Image.hpp"

namespace strikecem::rendering {

enum class RenderShadingMode {
    ShadedCad,       // Diffuse + ambient shaded 3D CAD geometry
    LitShadow,       // Illumination map: bright gold for lit, dark charcoal for shadow
    SurfaceCurrents  // Surface current magnitude |Js| mapped to scientific colormap
};

struct RenderCamera {
    geom::Vec3d eye{5.0, 5.0, 3.0};
    geom::Vec3d target{0.0, 0.0, 0.0};
    geom::Vec3d up{0.0, 0.0, 1.0};
    bool orthographic = true;

    static RenderCamera radar_view(const Direction& dir, const NormalizedMesh& mesh);
    static RenderCamera isometric_view(const NormalizedMesh& mesh, double az_deg = 45.0, double el_deg = 25.0);
};

struct ModelRenderConfig {
    size_t width = 1024;
    size_t height = 768;
    RenderShadingMode mode = RenderShadingMode::SurfaceCurrents;
    ColormapType colormap = ColormapType::Turbo;
    RenderCamera camera;
    bool camera_radar_aligned = true;
    bool draw_wireframe = false;
    bool draw_hud = true;
    Rgba background_color = {15, 20, 30, 255}; // Deep slate aerospace background
};

class ModelRenderer {
public:
    // Render 3D mesh with facet diagnostics (illumination, surface currents)
    static ImageBuffer render_diagnostics(
        const NormalizedMesh& mesh,
        const VtkMeshDiagnostics& diag,
        const Direction& dir,
        double frequency_hz,
        const std::string& polarization = "HH",
        const ModelRenderConfig& config = {});

    // Render pure CAD mesh (shaded or wireframe)
    static ImageBuffer render_mesh(
        const NormalizedMesh& mesh,
        const ModelRenderConfig& config = {});

    // Convenience function to render and directly write to disk
    static void render_and_save(
        const std::string& output_path,
        const NormalizedMesh& mesh,
        const VtkMeshDiagnostics& diag,
        const Direction& dir,
        double frequency_hz,
        const std::string& polarization = "HH",
        const ModelRenderConfig& config = {});
};

} // namespace strikecem::rendering

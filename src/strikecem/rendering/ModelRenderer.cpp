#include "strikecem/rendering/ModelRenderer.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <sstream>

#include "strikecem/core/Conventions.hpp"

namespace strikecem::rendering {

namespace {

struct CameraFrame {
    geom::Vec3d eye;
    geom::Vec3d target;
    geom::Vec3d forward;
    geom::Vec3d right;
    geom::Vec3d up;
    double scale = 1.0;
};

CameraFrame build_camera_frame(const RenderCamera& cam, const NormalizedMesh& mesh, double aspect) {
    CameraFrame frame;
    frame.eye = cam.eye;
    frame.target = cam.target;

    geom::Vec3d f = frame.target - frame.eye;
    const double f_len = f.length();
    if (f_len > 1e-9) {
        frame.forward = f * (1.0 / f_len);
    } else {
        frame.forward = {0.0, 1.0, 0.0};
    }

    geom::Vec3d r = geom::cross(frame.forward, cam.up);
    const double r_len = r.length();
    if (r_len > 1e-9) {
        frame.right = r * (1.0 / r_len);
    } else {
        frame.right = {1.0, 0.0, 0.0};
    }

    frame.up = geom::cross(frame.right, frame.forward);

    // Compute bounding radius around target
    double max_dist = 0.1;
    for (const auto& v : mesh.vertices) {
        const double d = (v - frame.target).length();
        if (d > max_dist) max_dist = d;
    }
    frame.scale = max_dist * 1.25; // 25% margin
    if (aspect > 1.0) {
        frame.scale *= 1.0;
    }
    return frame;
}

struct ProjectedVertex {
    int x = 0;
    int y = 0;
    double z = 0.0;
};

ProjectedVertex project_point(const geom::Vec3d& p, const CameraFrame& frame, int width, int height) {
    const geom::Vec3d v = p - frame.target;
    const double x_cam = geom::dot(v, frame.right);
    const double y_cam = geom::dot(v, frame.up);
    const double z_cam = geom::dot(v, frame.forward); // larger is closer to target / deeper into scene

    const double half_h = 0.5 * static_cast<double>(height);
    const double half_w = 0.5 * static_cast<double>(width);

    ProjectedVertex pv;
    pv.x = static_cast<int>(half_w + half_h * (x_cam / frame.scale));
    pv.y = static_cast<int>(half_h - half_h * (y_cam / frame.scale));
    pv.z = z_cam;
    return pv;
}

} // namespace

RenderCamera RenderCamera::radar_view(const Direction& dir, const NormalizedMesh& mesh) {
    const geom::Vec3d center = (mesh.report.bbox_min + mesh.report.bbox_max) * 0.5;
    const double diag = (mesh.report.bbox_max - mesh.report.bbox_min).length();
    const double dist = std::max(2.0, diag * 2.5);

    RenderCamera cam;
    cam.target = center;
    // Eye is along the direction to the radar (dir.k_hat)
    cam.eye = center + dir.k_hat * dist;
    cam.up = {0.0, 0.0, 1.0};
    if (std::abs(dir.k_hat.z) > 0.99) {
        cam.up = {1.0, 0.0, 0.0};
    }
    cam.orthographic = true;
    return cam;
}

RenderCamera RenderCamera::isometric_view(const NormalizedMesh& mesh, double az_deg, double el_deg) {
    const geom::Vec3d center = (mesh.report.bbox_min + mesh.report.bbox_max) * 0.5;
    const double diag = (mesh.report.bbox_max - mesh.report.bbox_min).length();
    const double dist = std::max(2.0, diag * 2.5);

    const geom::Vec3d dir = direction_from_az_el(az_deg, el_deg);
    RenderCamera cam;
    cam.target = center;
    cam.eye = center + dir * dist;
    cam.up = {0.0, 0.0, 1.0};
    cam.orthographic = true;
    return cam;
}

ImageBuffer ModelRenderer::render_diagnostics(
    const NormalizedMesh& mesh,
    const VtkMeshDiagnostics& diag,
    const Direction& dir,
    double frequency_hz,
    const std::string& polarization,
    const ModelRenderConfig& config) {

    const size_t W = config.width > 0 ? config.width : 1024;
    const size_t H = config.height > 0 ? config.height : 768;
    ImageBuffer img(W, H, config.background_color);

    // Subtle dark gradient background
    for (size_t y = 0; y < H; ++y) {
        const double factor = 1.0 - 0.4 * (static_cast<double>(y) / static_cast<double>(H));
        const Rgba row_bg = {
            static_cast<uint8_t>(config.background_color.r * factor),
            static_cast<uint8_t>(config.background_color.g * factor),
            static_cast<uint8_t>(config.background_color.b * factor),
            255
        };
        for (size_t x = 0; x < W; ++x) {
            img.set_pixel(static_cast<int>(x), static_cast<int>(y), row_bg);
        }
    }

    RenderCamera cam = config.camera;
    if (config.camera_radar_aligned) {
        cam = RenderCamera::radar_view(dir, mesh);
    }
    const double aspect = static_cast<double>(W) / static_cast<double>(H);
    const CameraFrame frame = build_camera_frame(cam, mesh, aspect);

    // Z-buffer (initialized to -infinity; larger z is closer along look ray)
    std::vector<double> z_buffer(W * H, -std::numeric_limits<double>::infinity());

    // Find maximum current magnitude for colormap normalization
    double max_current = 0.0;
    size_t lit_count = 0;
    for (const auto& fd : diag.facets) {
        if (fd.lit) ++lit_count;
        if (fd.current_magnitude > max_current) {
            max_current = fd.current_magnitude;
        }
    }
    if (max_current <= 1e-12) max_current = 1.0;

    // Directional light vector for shaded CAD mode (from top-right-front in camera space)
    const geom::Vec3d light_dir = (frame.right * 0.5 + frame.up * 0.8 - frame.forward * 0.6).normalized();

    // Project all vertices
    const size_t n_verts = mesh.vertices.size();
    std::vector<ProjectedVertex> proj(n_verts);
    for (size_t i = 0; i < n_verts; ++i) {
        proj[i] = project_point(mesh.vertices[i], frame, static_cast<int>(W), static_cast<int>(H));
    }

    // Rasterize all triangles
    const size_t n_tri = mesh.triangles.size();
    for (size_t t = 0; t < n_tri; ++t) {
        const auto& tri = mesh.triangles[t];
        const auto& p0 = proj[tri[0]];
        const auto& p1 = proj[tri[1]];
        const auto& p2 = proj[tri[2]];

        const geom::Vec3d& normal = mesh.normals[t];

        // Determine facet color based on shading mode
        Rgba facet_color{180, 190, 205, 255};
        if (config.mode == RenderShadingMode::ShadedCad) {
            const double diff = std::max(0.12, geom::dot(normal, light_dir));
            const double spec = std::pow(std::max(0.0, geom::dot(normal, light_dir)), 16) * 0.3;
            const double intensity = std::clamp(diff + spec, 0.0, 1.0);
            facet_color = {
                static_cast<uint8_t>(std::clamp(170.0 * intensity + 30.0, 0.0, 255.0)),
                static_cast<uint8_t>(std::clamp(185.0 * intensity + 35.0, 0.0, 255.0)),
                static_cast<uint8_t>(std::clamp(210.0 * intensity + 45.0, 0.0, 255.0)),
                255
            };
        } else if (config.mode == RenderShadingMode::LitShadow) {
            const bool is_lit = (t < diag.facets.size()) && (diag.facets[t].lit > 0);
            if (is_lit) {
                const double cos_i = std::clamp(-geom::dot(dir.k_hat, normal), 0.1, 1.0);
                facet_color = {
                    static_cast<uint8_t>(std::clamp(255.0 * (0.3 + 0.7 * cos_i), 0.0, 255.0)),
                    static_cast<uint8_t>(std::clamp(195.0 * (0.3 + 0.7 * cos_i), 0.0, 255.0)),
                    static_cast<uint8_t>(std::clamp(35.0 * cos_i, 0.0, 255.0)),
                    255
                }; // Amber gold for lit facets
            } else {
                facet_color = {35, 45, 65, 255}; // Dark navy charcoal for shadowed facets
            }
        } else if (config.mode == RenderShadingMode::SurfaceCurrents) {
            const double j_mag = (t < diag.facets.size()) ? diag.facets[t].current_magnitude : 0.0;
            const bool is_lit = (t < diag.facets.size()) && (diag.facets[t].lit > 0);
            if (is_lit) {
                const double norm_val = std::clamp(j_mag / max_current, 0.0, 1.0);
                facet_color = evaluate_colormap(config.colormap, norm_val);
            } else {
                facet_color = {25, 30, 42, 255}; // Shadowed facets stay dim dark
            }
        }

        // Bounding box of triangle in screen space
        int min_x = std::max(0, std::min({p0.x, p1.x, p2.x}));
        int max_x = std::min(static_cast<int>(W) - 1, std::max({p0.x, p1.x, p2.x}));
        int min_y = std::max(0, std::min({p0.y, p1.y, p2.y}));
        int max_y = std::min(static_cast<int>(H) - 1, std::max({p0.y, p1.y, p2.y}));

        const double denom = static_cast<double>((p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y));
        if (std::abs(denom) < 1e-5) continue; // Degenerate triangle

        for (int py = min_y; py <= max_y; ++py) {
            for (int px = min_x; px <= max_x; ++px) {
                const double w0 = static_cast<double>((p1.y - p2.y) * (px - p2.x) + (p2.x - p1.x) * (py - p2.y)) / denom;
                const double w1 = static_cast<double>((p2.y - p0.y) * (px - p2.x) + (p0.x - p2.x) * (py - p2.y)) / denom;
                const double w2 = 1.0 - w0 - w1;

                if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) {
                    const double z = w0 * p0.z + w1 * p1.z + w2 * p2.z;
                    const size_t pixel_idx = static_cast<size_t>(py) * W + static_cast<size_t>(px);
                    if (z > z_buffer[pixel_idx]) {
                        z_buffer[pixel_idx] = z;
                        img.set_pixel(px, py, facet_color);
                    }
                }
            }
        }

        // Optional subtle wireframe edge overlay
        if (config.draw_wireframe) {
            const Rgba wire_color{60, 80, 110, 180};
            img.draw_line(p0.x, p0.y, p1.x, p1.y, wire_color);
            img.draw_line(p1.x, p1.y, p2.x, p2.y, wire_color);
            img.draw_line(p2.x, p2.y, p0.x, p0.y, wire_color);
        }
    }

    // HUD / Annotations
    if (config.draw_hud) {
        // Top banner
        img.draw_rect(0, 0, static_cast<int>(W), 42, {10, 14, 22, 220}, true);
        img.draw_line(0, 42, static_cast<int>(W), 42, {45, 65, 95, 255});

        std::string mode_str = "SURFACE CURRENTS |Js|";
        if (config.mode == RenderShadingMode::LitShadow) mode_str = "LIT / SHADOW ILLUMINATION";
        if (config.mode == RenderShadingMode::ShadedCad) mode_str = "3D CAD SHADED GEOMETRY";

        img.draw_text(16, 12, "STRIKECEM 3D SIMULATION VISUALIZER - " + mode_str, {255, 255, 255, 255}, 1);

        std::ostringstream top_right;
        top_right << "Facets: " << n_tri << " | Lit: " << lit_count << " ("
                  << std::fixed << std::setprecision(1) << (n_tri > 0 ? (100.0 * lit_count / n_tri) : 0.0) << "%)";
        img.draw_text(static_cast<int>(W) - 340, 12, top_right.str(), {160, 200, 250, 255}, 1);

        // Bottom status bar
        const int bot_y = static_cast<int>(H) - 34;
        img.draw_rect(0, bot_y, static_cast<int>(W), 34, {10, 14, 22, 220}, true);
        img.draw_line(0, bot_y, static_cast<int>(W), bot_y, {45, 65, 95, 255});

        std::ostringstream bot_left;
        bot_left << "Aspect: Az " << std::fixed << std::setprecision(1) << dir.azimuth_deg
                 << " deg, El " << dir.elevation_deg << " deg | Freq: "
                 << std::setprecision(3) << (frequency_hz * 1e-9) << " GHz | Pol: " << polarization;
        img.draw_text(16, bot_y + 10, bot_left.str(), {220, 230, 245, 255}, 1);

        // Model dimensions on bottom right
        const geom::Vec3d dims = mesh.report.bbox_max - mesh.report.bbox_min;
        std::ostringstream bot_right;
        bot_right << "BBox: " << std::setprecision(2) << dims.x << "x" << dims.y << "x" << dims.z << " m";
        img.draw_text(static_cast<int>(W) - 220, bot_y + 10, bot_right.str(), {160, 180, 200, 255}, 1);

        // Colorbar on right if in SurfaceCurrents mode
        if (config.mode == RenderShadingMode::SurfaceCurrents) {
            const int cb_x = static_cast<int>(W) - 40;
            const int cb_y = 60;
            const int cb_w = 16;
            const int cb_h = 200;

            img.draw_rect(cb_x - 1, cb_y - 1, cb_w + 2, cb_h + 2, {255, 255, 255, 180}, false);
            for (int j = 0; j < cb_h; ++j) {
                const double t_val = 1.0 - (static_cast<double>(j) / static_cast<double>(cb_h));
                const Rgba c = evaluate_colormap(config.colormap, t_val);
                for (int i = 0; i < cb_w; ++i) {
                    img.set_pixel(cb_x + i, cb_y + j, c);
                }
            }
            img.draw_text(cb_x - 30, cb_y - 12, "Peak", {255, 255, 255, 255}, 1);
            img.draw_text(cb_x - 30, cb_y + cb_h + 4, "Zero", {200, 200, 200, 255}, 1);
        }

        // Orientation Triad in bottom left
        const int triad_cx = 50;
        const int triad_cy = static_cast<int>(H) - 80;
        const double triad_len = 32.0;

        const geom::Vec3d ax_x{1.0, 0.0, 0.0}; // +X (forward) -> Red
        const geom::Vec3d ax_y{0.0, 1.0, 0.0}; // +Y (starboard) -> Green
        const geom::Vec3d ax_z{0.0, 0.0, 1.0}; // +Z (up) -> Blue

        auto project_triad = [&](const geom::Vec3d& axis) {
            const double px = geom::dot(axis, frame.right);
            const double py = geom::dot(axis, frame.up);
            return std::pair<int, int>{
                triad_cx + static_cast<int>(px * triad_len),
                triad_cy - static_cast<int>(py * triad_len)
            };
        };

        const auto [xx, xy] = project_triad(ax_x);
        const auto [yx, yy] = project_triad(ax_y);
        const auto [zx, zy] = project_triad(ax_z);

        img.draw_line(triad_cx, triad_cy, xx, xy, {255, 60, 60, 255});
        img.draw_text(xx + 2, xy - 4, "X", {255, 80, 80, 255}, 1);

        img.draw_line(triad_cx, triad_cy, yx, yy, {60, 255, 60, 255});
        img.draw_text(yx + 2, yy - 4, "Y", {80, 255, 80, 255}, 1);

        img.draw_line(triad_cx, triad_cy, zx, zy, {60, 140, 255, 255});
        img.draw_text(zx + 2, zy - 4, "Z", {80, 160, 255, 255}, 1);
    }

    return img;
}

ImageBuffer ModelRenderer::render_mesh(
    const NormalizedMesh& mesh,
    const ModelRenderConfig& config) {

    // Dummy diagnostic to render purely geometric shaded mesh
    VtkMeshDiagnostics dummy_diag;
    dummy_diag.facets.resize(mesh.triangles.size());
    Direction dummy_dir;
    dummy_dir.k_hat = {1.0, 0.0, 0.0};
    ModelRenderConfig cfg = config;
    cfg.mode = RenderShadingMode::ShadedCad;
    return render_diagnostics(mesh, dummy_diag, dummy_dir, 10.0e9, "HH", cfg);
}

void ModelRenderer::render_and_save(
    const std::string& output_path,
    const NormalizedMesh& mesh,
    const VtkMeshDiagnostics& diag,
    const Direction& dir,
    double frequency_hz,
    const std::string& polarization,
    const ModelRenderConfig& config) {

    const ImageBuffer img = render_diagnostics(mesh, diag, dir, frequency_hz, polarization, config);
    img.write_image(output_path);
}

} // namespace strikecem::rendering

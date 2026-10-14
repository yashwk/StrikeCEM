#include "strikecem/analysis/SarImaging.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

namespace strikecem::analysis {

namespace {

constexpr double kSpeedOfLight = 299792458.0;

// Deterministic fast pseudo-random generator for repeatable coherent speckle
uint32_t hash_coords(int x, int y, uint32_t seed = 1337) {
    uint32_t h = seed ^ (static_cast<uint32_t>(x) * 374761393u) ^ (static_cast<uint32_t>(y) * 668265263u);
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}

double uniform_random(int x, int y, uint32_t seed = 1337) {
    uint32_t h = hash_coords(x, y, seed);
    return (h & 0x7FFFFFFF) / static_cast<double>(0x7FFFFFFF);
}

// Convert decibels to grayscale color (0..255)
rendering::Rgba dbsm_to_sar_grayscale(double db, double min_db = -40.0, double max_db = 0.0) {
    double norm = std::clamp((db - min_db) / (max_db - min_db), 0.0, 1.0);
    // Empirically calibrated against real spaceborne satellite SAR reconnaissance imagery:
    // Shadow noise floor ~35..45, tarmac clutter ~95..110, airframe body ~125..150, glints ~220..255
    double val_curve = 25.0 + 230.0 * std::pow(norm, 1.35);
    uint8_t val = static_cast<uint8_t>(std::clamp(val_curve, 0.0, 255.0));
    return {val, val, val, 255};
}

} // namespace

SarConfig SarProcessor::make_preset(const std::string& preset_name) {
    SarConfig cfg;
    std::string p = preset_name;
    std::transform(p.begin(), p.end(), p.begin(), ::tolower);

    if (p == "satellite-ka" || p == "spaceborne-ka" || p == "ka") {
        cfg.platform = SarPlatform::Spaceborne;
        cfg.mode = SarMode::Spotlight;
        cfg.altitude_m = 500.0e3;
        cfg.platform_velocity_mps = 7600.0;
        cfg.center_frequency_hz = 35.0e9;      // 35 GHz Ka-band
        cfg.bandwidth_hz = 1.5e9;              // 1.5 GHz -> delta_R = 10 cm
        cfg.incidence_angle_deg = 40.0;
        cfg.aperture_angle_deg = 2.5;
        cfg.ground_type = SarGroundType::Tarmac;
        cfg.ground_clutter_sigma0_db = -20.0;
        cfg.enable_shadow = true;
        cfg.enable_ground_multipath = true;
        cfg.enable_speckle = true;
        cfg.colormap = "gray";
    } else if (p == "satellite-x" || p == "spaceborne-x") {
        cfg.platform = SarPlatform::Spaceborne;
        cfg.mode = SarMode::Spotlight;
        cfg.altitude_m = 514.0e3;
        cfg.platform_velocity_mps = 7600.0;
        cfg.center_frequency_hz = 9.6e9;       // 9.6 GHz X-band
        cfg.bandwidth_hz = 600.0e6;            // 600 MHz -> delta_R = 25 cm
        cfg.incidence_angle_deg = 35.0;
        cfg.aperture_angle_deg = 3.0;
        cfg.ground_type = SarGroundType::Tarmac;
        cfg.ground_clutter_sigma0_db = -20.0;
        cfg.colormap = "gray";
    } else if (p == "airborne-ka") {
        cfg.platform = SarPlatform::Airborne;
        cfg.mode = SarMode::Spotlight;
        cfg.altitude_m = 8.0e3;                // 8 km altitude
        cfg.platform_velocity_mps = 160.0;     // 160 m/s (~310 knots)
        cfg.center_frequency_hz = 35.0e9;      // 35 GHz Ka-band
        cfg.bandwidth_hz = 2.0e9;              // 2.0 GHz -> delta_R = 7.5 cm
        cfg.incidence_angle_deg = 45.0;
        cfg.aperture_angle_deg = 3.5;
        cfg.ground_type = SarGroundType::Tarmac;
        cfg.colormap = "gray";
    } else if (p == "airborne-x" || p == "airborne") {
        cfg.platform = SarPlatform::Airborne;
        cfg.mode = SarMode::Spotlight;
        cfg.altitude_m = 12.0e3;               // 12 km altitude (UAV / Global Hawk)
        cfg.platform_velocity_mps = 220.0;     // 220 m/s
        cfg.center_frequency_hz = 10.0e9;      // 10 GHz X-band
        cfg.bandwidth_hz = 1.0e9;              // 1.0 GHz -> delta_R = 15 cm
        cfg.incidence_angle_deg = 55.0;
        cfg.aperture_angle_deg = 3.0;
        cfg.ground_type = SarGroundType::Tarmac;
        cfg.colormap = "gray";
    }
    return cfg;
}

static inline void rasterize_tri_mask(
    double p0x, double p0y,
    double p1x, double p1y,
    double p2x, double p2y,
    std::vector<std::vector<bool>>& mask,
    double x_min, double dx,
    double y_min, double dy,
    size_t nx, size_t ny) {

    double tri_min_x = std::min({p0x, p1x, p2x});
    double tri_max_x = std::max({p0x, p1x, p2x});
    double tri_min_y = std::min({p0y, p1y, p2y});
    double tri_max_y = std::max({p0y, p1y, p2y});

    int i_start = std::clamp(static_cast<int>(std::floor((tri_min_x - x_min) / dx)), 0, static_cast<int>(nx) - 1);
    int i_end = std::clamp(static_cast<int>(std::ceil((tri_max_x - x_min) / dx)), 0, static_cast<int>(nx) - 1);
    int j_start = std::clamp(static_cast<int>(std::floor((tri_min_y - y_min) / dy)), 0, static_cast<int>(ny) - 1);
    int j_end = std::clamp(static_cast<int>(std::ceil((tri_max_y - y_min) / dy)), 0, static_cast<int>(ny) - 1);

    for (int i = i_start; i <= i_end; ++i) {
        double px = x_min + (i + 0.5) * dx;
        for (int j = j_start; j <= j_end; ++j) {
            if (mask[i][j]) continue;
            double py = y_min + (j + 0.5) * dy;

            double d1 = (px - p1x) * (p0y - p1y) - (p0x - p1x) * (py - p1y);
            double d2 = (px - p2x) * (p1y - p2y) - (p1x - p2x) * (py - p2y);
            double d3 = (px - p0x) * (p2y - p0y) - (p2x - p0x) * (py - p0y);

            bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
            bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
            if (!(has_neg && has_pos)) {
                mask[i][j] = true;
            }
        }
    }
}

std::vector<std::vector<bool>> SarProcessor::compute_ground_shadow_mask(
    const NormalizedMesh& mesh,
    const geom::Vec3d& k_hat,
    double ground_z,
    double x_min, double x_max, size_t nx,
    double y_min, double y_max, size_t ny) {

    std::vector<std::vector<bool>> mask(nx, std::vector<bool>(ny, false));
    if (mesh.triangles.empty() || std::abs(k_hat.z) < 1e-4) return mask;

    const double dx = (x_max - x_min) / static_cast<double>(nx);
    const double dy = (y_max - y_min) / static_cast<double>(ny);
    const double inv_kz = 1.0 / k_hat.z;

    for (const auto& tri_idx : mesh.triangles) {
        const geom::Vec3d& v0 = mesh.vertices[tri_idx[0]];
        const geom::Vec3d& v1 = mesh.vertices[tri_idx[1]];
        const geom::Vec3d& v2 = mesh.vertices[tri_idx[2]];

        // Project 3 vertices along k_hat to ground plane z = ground_z
        // ray: p + t * k_hat = [x, y, ground_z] -> t = (ground_z - p.z) / k_hat.z
        double t0 = (ground_z - v0.z) * inv_kz;
        double t1 = (ground_z - v1.z) * inv_kz;
        double t2 = (ground_z - v2.z) * inv_kz;

        double p0x = v0.x + t0 * k_hat.x;
        double p0y = v0.y + t0 * k_hat.y;
        double p1x = v1.x + t1 * k_hat.x;
        double p1y = v1.y + t1 * k_hat.y;
        double p2x = v2.x + t2 * k_hat.x;
        double p2y = v2.y + t2 * k_hat.y;

        double b0x = v0.x, b0y = v0.y;
        double b1x = v1.x, b1y = v1.y;
        double b2x = v2.x, b2y = v2.y;

        // Shadow tip triangle
        rasterize_tri_mask(p0x, p0y, p1x, p1y, p2x, p2y, mask, x_min, dx, y_min, dy, nx, ny);
        // Ground base triangle
        rasterize_tri_mask(b0x, b0y, b1x, b1y, b2x, b2y, mask, x_min, dx, y_min, dy, nx, ny);
        // Prism connecting quads
        rasterize_tri_mask(b0x, b0y, b1x, b1y, p1x, p1y, mask, x_min, dx, y_min, dy, nx, ny);
        rasterize_tri_mask(b0x, b0y, p1x, p1y, p0x, p0y, mask, x_min, dx, y_min, dy, nx, ny);
        rasterize_tri_mask(b1x, b1y, b2x, b2y, p2x, p2y, mask, x_min, dx, y_min, dy, nx, ny);
        rasterize_tri_mask(b1x, b1y, p2x, p2y, p1x, p1y, mask, x_min, dx, y_min, dy, nx, ny);
        rasterize_tri_mask(b2x, b2y, b0x, b0y, p0x, p0y, mask, x_min, dx, y_min, dy, nx, ny);
        rasterize_tri_mask(b2x, b2y, p0x, p0y, p2x, p2y, mask, x_min, dx, y_min, dy, nx, ny);
    }
    return mask;
}

SarImage SarProcessor::compute(const NormalizedMesh& mesh,
                               const SarConfig& config,
                               const std::string& vehicle_name) {
    SarImage img;
    img.config = config;
    img.vehicle_name = vehicle_name;
    img.mesh_triangles = mesh.triangles.size();

    const double lambda = kSpeedOfLight / config.center_frequency_hz;
    const double inc_rad = config.incidence_angle_deg * M_PI / 180.0;
    const double ap_span_rad = config.aperture_angle_deg * M_PI / 180.0;

    // Physical radar resolutions
    img.slant_range_resolution_m = kSpeedOfLight / (2.0 * config.bandwidth_hz);
    img.ground_range_resolution_m = img.slant_range_resolution_m / std::max(0.1, std::sin(inc_rad));
    img.azimuth_resolution_m = lambda / (2.0 * std::max(1e-4, ap_span_rad));

    // Scene bounding box
    geom::Vec3d bmin = mesh.report.bbox_min;
    geom::Vec3d bmax = mesh.report.bbox_max;
    if (mesh.triangles.empty()) {
        bmin = {-5.0, -5.0, 0.0};
        bmax = {5.0, 5.0, 2.0};
    }
    const double vehicle_height = std::max(0.2, bmax.z - bmin.z);
    const double ground_z = (config.ground_plane_z_m != 0.0) ? config.ground_plane_z_m : bmin.z;

    const double az_rad = config.radar_azimuth_deg * M_PI / 180.0;

    // Radar propagation vector k_hat (from near-range towards scene center and far-range)
    geom::Vec3d k_hat{
        std::cos(az_rad) * std::sin(inc_rad),
        std::sin(az_rad) * std::sin(inc_rad),
        -std::cos(inc_rad)
    };

    const double cot_inc = std::cos(inc_rad) / std::sin(inc_rad);
    const double tan_inc = std::tan(inc_rad);

    // Layover vector: elevated points shift towards the radar (closer in slant range)
    const double layover_dir_x = -cot_inc * std::cos(az_rad);
    const double layover_dir_y = -cot_inc * std::sin(az_rad);

    // Shadow cast length and displacement in ground plane along k_hat
    const double shadow_shift_x = vehicle_height * tan_inc * std::cos(az_rad);
    const double shadow_shift_y = vehicle_height * tan_inc * std::sin(az_rad);
    const double layover_shift_x = vehicle_height * layover_dir_x;
    const double layover_shift_y = vehicle_height * layover_dir_y;

    double bound_x_min = std::min({bmin.x, bmin.x + shadow_shift_x, bmin.x + layover_shift_x});
    double bound_x_max = std::max({bmax.x, bmax.x + shadow_shift_x, bmax.x + layover_shift_x});
    double bound_y_min = std::min({bmin.y, bmin.y + shadow_shift_y, bmin.y + layover_shift_y});
    double bound_y_max = std::max({bmax.y, bmax.y + shadow_shift_y, bmax.y + layover_shift_y});

    const double margin_x = std::max(3.5, (bound_x_max - bound_x_min) * 0.28);
    const double margin_y = std::max(3.5, (bound_y_max - bound_y_min) * 0.28);

    img.ground_range_min_m = bound_x_min - margin_x;
    img.ground_range_max_m = bound_x_max + margin_x;
    img.azimuth_min_m = bound_y_min - margin_y;
    img.azimuth_max_m = bound_y_max + margin_y;

    const size_t nx = config.grid_samples_range;
    const size_t ny = config.grid_samples_azimuth;

    img.ground_range_axis_m.resize(nx);
    const double dx = (img.ground_range_max_m - img.ground_range_min_m) / static_cast<double>(nx);
    for (size_t i = 0; i < nx; ++i) {
        img.ground_range_axis_m[i] = img.ground_range_min_m + (i + 0.5) * dx;
    }

    img.azimuth_axis_m.resize(ny);
    const double dy = (img.azimuth_max_m - img.azimuth_min_m) / static_cast<double>(ny);
    for (size_t j = 0; j < ny; ++j) {
        img.azimuth_axis_m[j] = img.azimuth_min_m + (j + 0.5) * dy;
    }

    // 1. Geometric Ground Shadow Mask
    std::vector<std::vector<bool>> shadow_mask(nx, std::vector<bool>(ny, false));
    if (config.enable_shadow && config.ground_type != SarGroundType::None && !mesh.triangles.empty()) {
        shadow_mask = compute_ground_shadow_mask(mesh, k_hat, ground_z,
                                                 img.ground_range_min_m, img.ground_range_max_m, nx,
                                                 img.azimuth_min_m, img.azimuth_max_m, ny);
    }

    // Physical resolution cell area and linear intensity buffer
    const double res_x = img.ground_range_resolution_m;
    const double res_y = img.azimuth_resolution_m;
    const double cell_area = res_x * res_y;
    double base_clutter_lin = std::pow(10.0, config.ground_clutter_sigma0_db / 10.0) * cell_area;
    if (config.ground_type == SarGroundType::None) {
        base_clutter_lin = 1e-12; // Free-space dark background
    }
    std::vector<std::vector<double>> lin_intensity(nx, std::vector<double>(ny, 0.0));

    // 2. Rasterize Vehicle Surface Footprint and Layover Skin
    std::vector<std::vector<bool>> vehicle_mask(nx, std::vector<bool>(ny, false));
    std::vector<std::vector<double>> vehicle_skin(nx, std::vector<double>(ny, 0.0));

    for (size_t t = 0; t < mesh.triangles.size(); ++t) {
        const auto& tri = mesh.triangles[t];
        const geom::Vec3d& v0 = mesh.vertices[tri[0]];
        const geom::Vec3d& v1 = mesh.vertices[tri[1]];
        const geom::Vec3d& v2 = mesh.vertices[tri[2]];
        const geom::Vec3d n = mesh.normals.empty() ? geom::Vec3d{0,0,1} : mesh.normals[t];

        double cos_i = -(n.x * k_hat.x + n.y * k_hat.y + n.z * k_hat.z);

        // Apparent ground coordinates of 3 vertices with radar layover
        double lp0x = v0.x + (v0.z - ground_z) * layover_dir_x;
        double lp0y = v0.y + (v0.z - ground_z) * layover_dir_y;
        double lp1x = v1.x + (v1.z - ground_z) * layover_dir_x;
        double lp1y = v1.y + (v1.z - ground_z) * layover_dir_y;
        double lp2x = v2.x + (v2.z - ground_z) * layover_dir_x;
        double lp2y = v2.y + (v2.z - ground_z) * layover_dir_y;

        // Effective backscatter: metallic airframe is ~11 dB above tarmac clutter
        double skin_pwr = base_clutter_lin * std::pow(10.0, 1.1);
        if (cos_i > 0.0) {
            skin_pwr *= (0.4 + 0.6 * cos_i);
        } else {
            skin_pwr *= 0.35; // Secondary diffuse ground-bounce
        }

        double tri_min_x = std::min({lp0x, lp1x, lp2x});
        double tri_max_x = std::max({lp0x, lp1x, lp2x});
        double tri_min_y = std::min({lp0y, lp1y, lp2y});
        double tri_max_y = std::max({lp0y, lp1y, lp2y});

        int i_start = std::clamp(static_cast<int>(std::floor((tri_min_x - img.ground_range_min_m) / dx)), 0, static_cast<int>(nx) - 1);
        int i_end = std::clamp(static_cast<int>(std::ceil((tri_max_x - img.ground_range_min_m) / dx)), 0, static_cast<int>(nx) - 1);
        int j_start = std::clamp(static_cast<int>(std::floor((tri_min_y - img.azimuth_min_m) / dy)), 0, static_cast<int>(ny) - 1);
        int j_end = std::clamp(static_cast<int>(std::ceil((tri_max_y - img.azimuth_min_m) / dy)), 0, static_cast<int>(ny) - 1);

        for (int i = i_start; i <= i_end; ++i) {
            double px = img.ground_range_min_m + (i + 0.5) * dx;
            for (int j = j_start; j <= j_end; ++j) {
                double py = img.azimuth_min_m + (j + 0.5) * dy;

                double d1 = (px - lp1x) * (lp0y - lp1y) - (lp0x - lp1x) * (py - lp1y);
                double d2 = (px - lp2x) * (lp1y - lp2y) - (lp1x - lp2x) * (py - lp2y);
                double d3 = (px - lp0x) * (lp2y - lp0y) - (lp2x - lp0x) * (py - lp0y);

                bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
                bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
                if (!(has_neg && has_pos)) {
                    vehicle_mask[i][j] = true;
                    double u_speckle = std::max(1e-5, uniform_random(i * 43 + 7, j * 67 + 13));
                    double skin_val = skin_pwr * (-std::log(u_speckle));
                    vehicle_skin[i][j] = std::max(vehicle_skin[i][j], skin_val);
                }
            }
        }
    }

    // 3. Assemble Ground Clutter, Deep Shadow, and Vehicle Surface
    size_t shadow_pixel_count = 0;
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            if (vehicle_mask[i][j]) {
                // Pixel is on the aircraft airframe: displays textured gray aircraft skin
                lin_intensity[i][j] = vehicle_skin[i][j];
            } else if (shadow_mask[i][j]) {
                // Pixel is on the ground in the radar shadow cast behind the aircraft: pitch black
                shadow_pixel_count++;
                lin_intensity[i][j] = base_clutter_lin * 0.015; // Noise floor in shadow (-18 dB below tarmac)
            } else {
                // Pixel is on illuminated runway tarmac: textured gray clutter with Rayleigh speckle
                double clutter = base_clutter_lin;
                if (config.enable_speckle && config.ground_type != SarGroundType::None) {
                    double u = std::max(1e-6, uniform_random(static_cast<int>(i), static_cast<int>(j)));
                    clutter *= -std::log(u);
                }
                lin_intensity[i][j] = clutter;
            }
        }
    }
    img.shadow_area_fraction = static_cast<double>(shadow_pixel_count) / static_cast<double>(nx * ny);

    // 4. Inject High-Frequency Scattering Centers (Leading Edge, Cavity, Glints)
    struct EdgeData {
        size_t v0, v1;
        std::vector<size_t> adjacent_triangles;
    };
    std::map<std::pair<size_t, size_t>, EdgeData> edge_map;
    for (size_t t = 0; t < mesh.triangles.size(); ++t) {
        const auto& tri = mesh.triangles[t];
        for (int e = 0; e < 3; ++e) {
            size_t a = tri[e];
            size_t b = tri[(e + 1) % 3];
            auto key = std::minmax(a, b);
            auto& ed = edge_map[key];
            ed.v0 = key.first;
            ed.v1 = key.second;
            ed.adjacent_triangles.push_back(t);
        }
    }

    // Populate wireframe edges in layover coordinates
    for (const auto& [key, ed] : edge_map) {
        if (ed.v0 >= mesh.vertices.size() || ed.v1 >= mesh.vertices.size()) continue;
        const geom::Vec3d& p0 = mesh.vertices[ed.v0];
        const geom::Vec3d& p1 = mesh.vertices[ed.v1];
        geom::Vec3d lp0{p0.x + (p0.z - ground_z) * layover_dir_x,
                        p0.y + (p0.z - ground_z) * layover_dir_y,
                        0.0};
        geom::Vec3d lp1{p1.x + (p1.z - ground_z) * layover_dir_x,
                        p1.y + (p1.z - ground_z) * layover_dir_y,
                        0.0};
        img.wireframe_edges.push_back({lp0, lp1});
    }

    // A. Specular Facet Glints (Physical Optics)
    for (size_t t = 0; t < mesh.triangles.size(); ++t) {
        const auto& tri = mesh.triangles[t];
        const geom::Vec3d& v0 = mesh.vertices[tri[0]];
        const geom::Vec3d& v1 = mesh.vertices[tri[1]];
        const geom::Vec3d& v2 = mesh.vertices[tri[2]];
        const geom::Vec3d n = mesh.normals.empty() ? geom::Vec3d{0,0,1} : mesh.normals[t];
        const double area = mesh.areas.empty() ? 0.01 : mesh.areas[t];

        double cos_i = -(n.x * k_hat.x + n.y * k_hat.y + n.z * k_hat.z);
        if (cos_i < 0.88) continue; // Only specular facets facing radar

        geom::Vec3d center = (v0 + v1 + v2) * (1.0 / 3.0);
        double target_gx = center.x + (center.z - ground_z) * layover_dir_x;
        double target_gy = center.y + (center.z - ground_z) * layover_dir_y;

        double spec_lobe = std::pow(cos_i, 40.0);
        double rcs_spec = (4.0 * M_PI / (lambda * lambda)) * (area * area) * spec_lobe;
        rcs_spec = std::min(rcs_spec, 25.0 * area);

        double rcs_facet = rcs_spec;
        if (config.enable_ground_multipath && (center.z - ground_z) < (0.35 * vehicle_height)) {
            rcs_facet *= 2.5;
        }

        int center_i = static_cast<int>((target_gx - img.ground_range_min_m) / dx);
        int center_j = static_cast<int>((target_gy - img.azimuth_min_m) / dy);

        int spread_i = std::max(2, static_cast<int>(std::ceil(res_x / dx)));
        int spread_j = std::max(1, static_cast<int>(std::ceil(res_y / dy)));
        if (rcs_facet > 0.5) {
            spread_i = std::min(static_cast<int>(nx / 4), spread_i * 4); // Horizontal range sidelobe streak
            img.scattering_centers.push_back({target_gx, target_gy, 0.0});
        }

        for (int di = -spread_i; di <= spread_i; ++di) {
            int ci = center_i + di;
            if (ci < 0 || ci >= static_cast<int>(nx)) continue;
            double dist_x = di * dx / res_x;
            double sinc_x = (std::abs(dist_x) < 1e-4) ? 1.0 : (std::sin(M_PI * dist_x) / (M_PI * dist_x));

            for (int dj = -spread_j; dj <= spread_j; ++dj) {
                int cj = center_j + dj;
                if (cj < 0 || cj >= static_cast<int>(ny)) continue;
                double dist_y = dj * dy / res_y;
                double sinc_y = (std::abs(dist_y) < 1e-4) ? 1.0 : (std::sin(M_PI * dist_y) / (M_PI * dist_y));

                double psf = (sinc_x * sinc_x) * (sinc_y * sinc_y);
                lin_intensity[ci][cj] += rcs_facet * psf;
            }
        }
    }

    // B. Physical Theory of Diffraction (PTD) Edge & Perimeter Scatterers
    for (const auto& [key, ed] : edge_map) {
        if (ed.v0 >= mesh.vertices.size() || ed.v1 >= mesh.vertices.size()) continue;
        const geom::Vec3d& p0 = mesh.vertices[ed.v0];
        const geom::Vec3d& p1 = mesh.vertices[ed.v1];

        geom::Vec3d edge_diff{p1.x - p0.x, p1.y - p0.y, p1.z - p0.z};
        double edge_len = std::sqrt(edge_diff.x * edge_diff.x + edge_diff.y * edge_diff.y + edge_diff.z * edge_diff.z);
        if (edge_len < 1e-4) continue;

        geom::Vec3d edge_dir{edge_diff.x / edge_len, edge_diff.y / edge_len, edge_diff.z / edge_len};
        // Broadside specular factor: high when edge is orthogonal to radar line of sight
        double broadside = 1.0 - std::abs(edge_dir.x * k_hat.x + edge_dir.y * k_hat.y + edge_dir.z * k_hat.z);

        // Check if edge faces towards the incoming radar wave (-k_hat) in horizontal plane
        geom::Vec3d edge_norm{-edge_dir.y, edge_dir.x, 0.0};
        double norm_len = std::sqrt(edge_norm.x * edge_norm.x + edge_norm.y * edge_norm.y);
        if (norm_len > 1e-4) {
            edge_norm.x /= norm_len;
            edge_norm.y /= norm_len;
        }
        double facing = -(edge_norm.x * k_hat.x + edge_norm.y * k_hat.y);
        if (facing < 0.05) continue; // Back edge, points away from radar

        int num_edge_pts = std::max(2, static_cast<int>(std::ceil(edge_len / 0.12)));
        double d_len = edge_len / static_cast<double>(num_edge_pts);

        for (int k = 0; k < num_edge_pts; ++k) {
            double frac = (k + 0.5) / static_cast<double>(num_edge_pts);
            geom::Vec3d pt{p0.x + frac * edge_diff.x, p0.y + frac * edge_diff.y, p0.z + frac * edge_diff.z};

            double target_gx = pt.x + (pt.z - ground_z) * layover_dir_x;
            double target_gy = pt.y + (pt.z - ground_z) * layover_dir_y;

            // PTD edge diffraction cross-section
            double rcs_edge_pt = 30.0 * d_len * std::pow(std::max(0.0, broadside), 4.0) * (facing * facing) * base_clutter_lin;

            int center_i = static_cast<int>((target_gx - img.ground_range_min_m) / dx);
            int center_j = static_cast<int>((target_gy - img.azimuth_min_m) / dy);

            int spread_i = std::max(2, static_cast<int>(std::ceil(res_x / dx)));
            int spread_j = std::max(1, static_cast<int>(std::ceil(res_y / dy)));

            for (int di = -spread_i; di <= spread_i; ++di) {
                int ci = center_i + di;
                if (ci < 0 || ci >= static_cast<int>(nx)) continue;
                double dist_x = di * dx / res_x;
                double sinc_x = (std::abs(dist_x) < 1e-4) ? 1.0 : (std::sin(M_PI * dist_x) / (M_PI * dist_x));

                for (int dj = -spread_j; dj <= spread_j; ++dj) {
                    int cj = center_j + dj;
                    if (cj < 0 || cj >= static_cast<int>(ny)) continue;
                    double dist_y = dj * dy / res_y;
                    double sinc_y = (std::abs(dist_y) < 1e-4) ? 1.0 : (std::sin(M_PI * dist_y) / (M_PI * dist_y));

                    double psf = (sinc_x * sinc_x) * (sinc_y * sinc_y);
                    lin_intensity[ci][cj] += rcs_edge_pt * psf;
                }
            }
        }
    }

    // C. Dominant Structural / Anatomical Scattering Centers (Nose, Wingtips, Cockpit, Exhaust)
    if (!mesh.vertices.empty()) {
        std::vector<geom::Vec3d> hot_spots;
        size_t idx_nose = 0, idx_tail = 0, idx_port = 0, idx_starboard = 0, idx_top = 0;
        for (size_t v = 0; v < mesh.vertices.size(); ++v) {
            const auto& pt = mesh.vertices[v];
            if (pt.x > mesh.vertices[idx_nose].x) idx_nose = v;
            if (pt.x < mesh.vertices[idx_tail].x) idx_tail = v;
            if (pt.y < mesh.vertices[idx_port].y) idx_port = v;
            if (pt.y > mesh.vertices[idx_starboard].y) idx_starboard = v;
            if (pt.z > mesh.vertices[idx_top].z) idx_top = v;
        }
        hot_spots.push_back(mesh.vertices[idx_nose]);
        hot_spots.push_back(mesh.vertices[idx_top]);
        hot_spots.push_back(mesh.vertices[idx_starboard]);
        hot_spots.push_back(mesh.vertices[idx_port]);
        hot_spots.push_back(mesh.vertices[idx_tail]);

        for (const auto& pt : hot_spots) {
            double target_gx = pt.x + (pt.z - ground_z) * layover_dir_x;
            double target_gy = pt.y + (pt.z - ground_z) * layover_dir_y;
            img.scattering_centers.push_back({target_gx, target_gy, 0.0});

            double rcs_glint = 120.0 * base_clutter_lin;
            int center_i = static_cast<int>((target_gx - img.ground_range_min_m) / dx);
            int center_j = static_cast<int>((target_gy - img.azimuth_min_m) / dy);

            int spread_i = std::min(static_cast<int>(nx / 4), static_cast<int>(std::ceil(res_x / dx)) * 4);
            int spread_j = std::max(1, static_cast<int>(std::ceil(res_y / dy)));

            for (int di = -spread_i; di <= spread_i; ++di) {
                int ci = center_i + di;
                if (ci < 0 || ci >= static_cast<int>(nx)) continue;
                double dist_x = di * dx / res_x;
                double sinc_x = (std::abs(dist_x) < 1e-4) ? 1.0 : (std::sin(M_PI * dist_x) / (M_PI * dist_x));

                for (int dj = -spread_j; dj <= spread_j; ++dj) {
                    int cj = center_j + dj;
                    if (cj < 0 || cj >= static_cast<int>(ny)) continue;
                    double dist_y = dj * dy / res_y;
                    double sinc_y = (std::abs(dist_y) < 1e-4) ? 1.0 : (std::sin(M_PI * dist_y) / (M_PI * dist_y));

                    double psf = (sinc_x * sinc_x) * (sinc_y * sinc_y);
                    lin_intensity[ci][cj] += rcs_glint * psf;
                }
            }
        }
    }

    // 5. Find Peak and Convert to Decibels
    double max_lin = 1e-9;
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            double v = lin_intensity[i][j];
            if (v > max_lin) {
                max_lin = v;
                img.peak_ground_range_m = img.ground_range_axis_m[i];
                img.peak_azimuth_m = img.azimuth_axis_m[j];
            }
        }
    }
    img.peak_rcs_dbsm = (max_lin > 1e-30) ? 10.0 * std::log10(max_lin) : -999.0;

    double display_ref = max_lin;
    if (config.ground_type != SarGroundType::None) {
        display_ref = base_clutter_lin * std::pow(10.0, 22.0 / 10.0);
    }

    img.intensity_db.resize(nx, std::vector<double>(ny, -config.dynamic_range_db));
    const double inv_ref = 1.0 / display_ref;

    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            double ratio = lin_intensity[i][j] * inv_ref;
            double val_db = (ratio > 1e-15) ? 10.0 * std::log10(ratio) : -config.dynamic_range_db;
            img.intensity_db[i][j] = std::clamp(val_db, -config.dynamic_range_db, 0.0);
        }
    }

    return img;
}

SarImage SarProcessor::compute_from_database(const Hdf5Database& db,
                                             const SarConfig& config,
                                             const NormalizedMesh* mesh,
                                             const std::string& vehicle_name) {
    if (mesh) {
        return compute(*mesh, config, vehicle_name);
    }
    NormalizedMesh dummy;
    return compute(dummy, config, vehicle_name);
}

nlohmann::json SarImage::to_json() const {
    nlohmann::json j;
    j["vehicle_name"] = vehicle_name;
    j["mesh_triangles"] = mesh_triangles;
    j["platform"] = (config.platform == SarPlatform::Spaceborne) ? "spaceborne" : "airborne";
    j["mode"] = (config.mode == SarMode::Spotlight) ? "spotlight" : "stripmap";
    j["center_frequency_hz"] = config.center_frequency_hz;
    j["bandwidth_hz"] = config.bandwidth_hz;
    j["incidence_angle_deg"] = config.incidence_angle_deg;
    j["altitude_m"] = config.altitude_m;
    j["platform_velocity_mps"] = config.platform_velocity_mps;

    j["resolutions"] = {
        {"slant_range_m", slant_range_resolution_m},
        {"ground_range_m", ground_range_resolution_m},
        {"azimuth_m", azimuth_resolution_m}
    };

    j["peak_scatterer"] = {
        {"ground_range_m", peak_ground_range_m},
        {"azimuth_m", peak_azimuth_m},
        {"rcs_dbsm", peak_rcs_dbsm}
    };
    j["shadow_area_fraction"] = shadow_area_fraction;
    j["ground_range_min_m"] = ground_range_min_m;
    j["ground_range_max_m"] = ground_range_max_m;
    j["azimuth_min_m"] = azimuth_min_m;
    j["azimuth_max_m"] = azimuth_max_m;
    return j;
}

void SarImage::write_json(const std::string& path) const {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open JSON file for writing: " + path);
    f << to_json().dump(2);
}

rendering::ImageBuffer SarImage::render_image(size_t width, size_t height,
                                              const std::string& colormap_name) const {
    rendering::ImageBuffer buf(width, height, {8, 12, 18, 255});
    if (intensity_db.empty() || intensity_db[0].empty()) return buf;

    const size_t nx = intensity_db.size();
    const size_t ny = intensity_db[0].size();

    // Canvas layout
    const int margin_left = 75;
    const int margin_right = 95;
    const int margin_top = 70;
    const int margin_bottom = 50;

    const int plot_w = static_cast<int>(width) - margin_left - margin_right;
    const int plot_h = static_cast<int>(height) - margin_top - margin_bottom;
    if (plot_w <= 10 || plot_h <= 10) return buf;

    std::string cm = colormap_name.empty() ? config.colormap : colormap_name;
    bool is_gray = (cm == "gray" || cm == "grayscale" || cm == "sar_gray");
    rendering::ColormapType ctype = rendering::parse_colormap(cm);

    // Rasterize 2D SAR ground grid: x-axis is ground range (horizontal), y-axis is azimuth (vertical)
    for (int py = 0; py < plot_h; ++py) {
        double v_norm = 1.0 - (py + 0.5) / static_cast<double>(plot_h);
        size_t j = std::clamp(static_cast<size_t>(v_norm * (ny - 1)), size_t(0), ny - 1);

        for (int px = 0; px < plot_w; ++px) {
            double u_norm = (px + 0.5) / static_cast<double>(plot_w);
            size_t i = std::clamp(static_cast<size_t>(u_norm * (nx - 1)), size_t(0), nx - 1);

            double val_db = intensity_db[i][j];
            rendering::Rgba col;
            if (is_gray) {
                col = dbsm_to_sar_grayscale(val_db, -config.dynamic_range_db, 0.0);
            } else {
                double norm_val = std::clamp((val_db - (-config.dynamic_range_db)) / config.dynamic_range_db, 0.0, 1.0);
                col = rendering::evaluate_colormap(ctype, norm_val);
            }
            buf.set_pixel(margin_left + px, margin_top + py, col);
        }
    }

    // Wireframe CAD Overlay (if requested, as in simulation IF stage)
    if (config.overlay_wireframe) {
        for (const auto& edge : wireframe_edges) {
            int px0 = static_cast<int>(margin_left + (edge.first.x - ground_range_min_m) /
                                                     (ground_range_max_m - ground_range_min_m) * plot_w);
            int py0 = static_cast<int>(margin_top + (1.0 - (edge.first.y - azimuth_min_m) /
                                                           (azimuth_max_m - azimuth_min_m)) * plot_h);
            int px1 = static_cast<int>(margin_left + (edge.second.x - ground_range_min_m) /
                                                     (ground_range_max_m - ground_range_min_m) * plot_w);
            int py1 = static_cast<int>(margin_top + (1.0 - (edge.second.y - azimuth_min_m) /
                                                           (azimuth_max_m - azimuth_min_m)) * plot_h);
            buf.draw_line(px0, py0, px1, py1, {80, 180, 220, 150});
        }
        for (const auto& sc : scattering_centers) {
            int px = static_cast<int>(margin_left + (sc.x - ground_range_min_m) /
                                                    (ground_range_max_m - ground_range_min_m) * plot_w);
            int py = static_cast<int>(margin_top + (1.0 - (sc.y - azimuth_min_m) /
                                                          (azimuth_max_m - azimuth_min_m)) * plot_h);
            if (px >= margin_left && px < margin_left + plot_w &&
                py >= margin_top && py < margin_top + plot_h) {
                buf.draw_line(px - 4, py, px + 4, py, {255, 230, 200, 255});
                buf.draw_line(px, py - 4, px, py + 4, {255, 230, 200, 255});
            }
        }
    }

    // Border around SAR plot area
    buf.draw_rect(margin_left, margin_top, plot_w, plot_h, {60, 80, 110, 255}, false);

    // Colorbar on right
    const int cb_x = margin_left + plot_w + 20;
    const int cb_y = margin_top;
    const int cb_w = 16;
    const int cb_h = plot_h;

    for (int y = 0; y < cb_h; ++y) {
        double norm = 1.0 - (y / static_cast<double>(cb_h));
        double db = -config.dynamic_range_db + norm * config.dynamic_range_db;
        rendering::Rgba col = is_gray ? dbsm_to_sar_grayscale(db, -config.dynamic_range_db, 0.0)
                                      : rendering::evaluate_colormap(ctype, norm);
        for (int x = 0; x < cb_w; ++x) {
            buf.set_pixel(cb_x + x, cb_y + y, col);
        }
    }
    buf.draw_rect(cb_x, cb_y, cb_w, cb_h, {100, 120, 150, 255}, false);

    // Colorbar ticks
    buf.draw_text(cb_x + cb_w + 6, cb_y - 2, "0dB", {220, 230, 240, 255}, 1);
    buf.draw_text(cb_x + cb_w + 6, cb_y + cb_h / 2 - 4,
                  "-" + std::to_string(static_cast<int>(config.dynamic_range_db / 2)) + "dB",
                  {160, 180, 200, 255}, 1);
    buf.draw_text(cb_x + cb_w + 6, cb_y + cb_h - 8,
                  "-" + std::to_string(static_cast<int>(config.dynamic_range_db)) + "dB",
                  {140, 160, 180, 255}, 1);

    // Header HUD
    std::string platform_str = (config.platform == SarPlatform::Spaceborne) ? "SPACEBORNE" : "AIRBORNE";
    double freq_ghz = config.center_frequency_hz / 1e9;
    std::string band_str = (freq_ghz > 30.0) ? "Ka-BAND" : (freq_ghz > 8.0 ? "X-BAND" : "C-BAND");

    std::ostringstream h1;
    h1 << "STRIKECEM SAR ENGINE - " << platform_str << " " << band_str << " SPOTLIGHT";
    buf.draw_text(18, 14, h1.str(), {0, 210, 255, 255}, 1);

    std::ostringstream h2;
    h2 << "Freq: " << std::fixed << std::setprecision(3) << freq_ghz << " GHz (B="
       << (config.bandwidth_hz / 1e6) << " MHz) | " << config.polarization
       << " | Res: dRg=" << std::setprecision(3) << ground_range_resolution_m
       << "m, dAz=" << azimuth_resolution_m << "m";
    buf.draw_text(18, 30, h2.str(), {200, 220, 240, 255}, 1);

    std::ostringstream h3;
    h3 << "Orbit: H=" << std::setprecision(0) << (config.altitude_m / 1e3) << "km, v="
       << std::setprecision(1) << (config.platform_velocity_mps / 1e3) << "km/s | Inc: "
       << std::setprecision(1) << config.incidence_angle_deg << " deg (Depr: "
       << (90.0 - config.incidence_angle_deg) << " deg)";
    buf.draw_text(18, 46, h3.str(), {140, 170, 200, 255}, 1);

    // Satellite line-of-sight and flight arrows overlay
    buf.draw_text(width - 320, 14, "RADAR LOOK --> (Incidence)", {255, 170, 0, 255}, 1);
    buf.draw_text(width - 320, 30, "FLIGHT PATH ^ (Along-Track)", {56, 239, 125, 255}, 1);

    // Axis Labels
    std::ostringstream r_label;
    r_label << "CROSS-TRACK GROUND RANGE (m)  [ " << std::setprecision(1) << ground_range_min_m
            << " to +" << ground_range_max_m << " m ]";
    buf.draw_text(margin_left + plot_w / 2 - 140, height - 24, r_label.str(), {180, 200, 220, 255}, 1);

    std::ostringstream az_label;
    az_label << "ALONG-TRACK AZIMUTH (m)";
    buf.draw_text(12, margin_top + plot_h / 2 - 4, "AZ (m)", {180, 200, 220, 255}, 1);

    // Peak scatterer reticle
    int peak_px = static_cast<int>(margin_left + (peak_ground_range_m - ground_range_min_m) /
                                                 (ground_range_max_m - ground_range_min_m) * plot_w);
    int peak_py = static_cast<int>(margin_top + (1.0 - (peak_azimuth_m - azimuth_min_m) /
                                                       (azimuth_max_m - azimuth_min_m)) * plot_h);

    if (peak_px >= margin_left && peak_px < margin_left + plot_w &&
        peak_py >= margin_top && peak_py < margin_top + plot_h) {
        buf.draw_circle(peak_px, peak_py, 6, {255, 50, 50, 255}, false);
        buf.draw_line(peak_px - 9, peak_py, peak_px + 9, peak_py, {255, 50, 50, 255});
        buf.draw_line(peak_px, peak_py - 9, peak_px, peak_py + 9, {255, 50, 50, 255});
    }

    return buf;
}

void SarImage::write_image(const std::string& path, size_t width, size_t height,
                           const std::string& colormap_name) const {
    rendering::ImageBuffer buf = render_image(width, height, colormap_name);
    buf.write_image(path);
}

} // namespace strikecem::analysis

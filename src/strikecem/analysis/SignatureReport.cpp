#include "strikecem/analysis/SignatureReport.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#include "strikecem/rendering/Image.hpp"
#include "strikecem/rendering/ModelRenderer.hpp"
#include "strikecem/diagnostics/VtkWriter.hpp"

namespace strikecem::analysis {

namespace fs = std::filesystem;

namespace {

double linear_power_mean_dbsm(const std::vector<double>& values_db) {
    if (values_db.empty()) return -999.0;
    double sum_lin = 0.0;
    for (double v : values_db) {
        sum_lin += std::pow(10.0, v / 10.0);
    }
    double mean_lin = sum_lin / static_cast<double>(values_db.size());
    if (mean_lin <= 1e-30) return -999.0;
    return 10.0 * std::log10(mean_lin);
}

double median_of(std::vector<double> vals) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    size_t mid = vals.size() / 2;
    if (vals.size() % 2 == 0) {
        return (vals[mid - 1] + vals[mid]) * 0.5;
    }
    return vals[mid];
}

SectorStats compute_sector_stats(const std::vector<double>& dbsm_values) {
    SectorStats s;
    if (dbsm_values.empty()) return s;
    s.sample_count = dbsm_values.size();
    s.min_dbsm = *std::min_element(dbsm_values.begin(), dbsm_values.end());
    s.max_dbsm = *std::max_element(dbsm_values.begin(), dbsm_values.end());
    s.mean_dbsm = linear_power_mean_dbsm(dbsm_values);
    s.median_dbsm = median_of(dbsm_values);
    return s;
}

std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }
    return {};
}

} // namespace

std::string SignatureReportGenerator::base64_encode(const std::vector<uint8_t>& data) {
    static const char kChars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    for (size_t i = 0; i < data.size(); i += 3) {
        uint32_t val = (static_cast<uint32_t>(data[i]) << 16);
        if (i + 1 < data.size()) val |= (static_cast<uint32_t>(data[i + 1]) << 8);
        if (i + 2 < data.size()) val |= static_cast<uint32_t>(data[i + 2]);
        out.push_back(kChars[(val >> 18) & 0x3F]);
        out.push_back(kChars[(val >> 12) & 0x3F]);
        out.push_back((i + 1 < data.size()) ? kChars[(val >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < data.size()) ? kChars[val & 0x3F] : '=');
    }
    return out;
}

std::string SignatureReportGenerator::base64_encode(const std::string& data) {
    std::vector<uint8_t> bytes(data.begin(), data.end());
    return base64_encode(bytes);
}

SignatureAnalysisSummary SignatureReportGenerator::analyze(const Hdf5Database& db) {
    SignatureAnalysisSummary sum;
    if (db.rows.empty()) return sum;

    std::vector<double> all_dbsm;
    all_dbsm.reserve(db.rows.size());

    std::vector<double> nose_vals, stbd_vals, port_vals, stern_vals;

    // Find principal elevation (closest to 0.0)
    size_t target_el_idx = 0;
    double best_el_diff = 1e9;
    for (size_t i = 0; i < db.elevation_deg.size(); ++i) {
        double diff = std::abs(db.elevation_deg[i]);
        if (diff < best_el_diff) {
            best_el_diff = diff;
            target_el_idx = i;
        }
    }

    // Find principal frequency (first frequency)
    const size_t target_freq_idx = 0;

    for (const auto& row : db.rows) {
        if (!row.valid) continue;
        all_dbsm.push_back(row.rcs_dbsm);

        if (row.rcs_dbsm > sum.peak_rcs_dbsm) {
            sum.peak_rcs_dbsm = row.rcs_dbsm;
            if (row.direction_index < db.azimuth_deg.size()) {
                sum.peak_azimuth_deg = db.azimuth_deg[row.direction_index];
            }
            if (row.direction_index < db.elevation_deg.size()) {
                sum.peak_elevation_deg = db.elevation_deg[row.direction_index];
            }
            if (row.frequency_index < db.frequencies_hz.size()) {
                sum.peak_frequency_hz = db.frequencies_hz[row.frequency_index];
            }
        }

        // Categorize into aerospace sectors based on azimuth
        if (row.direction_index < db.azimuth_deg.size()) {
            double az = db.azimuth_deg[row.direction_index];
            // Normalize azimuth to [0, 360)
            while (az < 0.0) az += 360.0;
            while (az >= 360.0) az -= 360.0;

            if (az <= 15.0 || az >= 345.0) {
                nose_vals.push_back(row.rcs_dbsm);
            } else if (az >= 75.0 && az <= 105.0) {
                stbd_vals.push_back(row.rcs_dbsm);
            } else if (az >= 255.0 && az <= 285.0) {
                port_vals.push_back(row.rcs_dbsm);
            } else if (az >= 165.0 && az <= 195.0) {
                stern_vals.push_back(row.rcs_dbsm);
            }
        }

        // Collect polar cut for target elevation and target frequency
        if (row.frequency_index == target_freq_idx &&
            row.polarization_index == 0 &&
            row.direction_index < db.azimuth_deg.size() &&
            row.direction_index < db.elevation_deg.size() &&
            std::abs(db.elevation_deg[row.direction_index] - db.elevation_deg[target_el_idx]) < 1e-4) {
            sum.polar_azimuth_deg.push_back(db.azimuth_deg[row.direction_index]);
            sum.polar_rcs_dbsm.push_back(row.rcs_dbsm);
        }
    }

    sum.overall_mean_dbsm = linear_power_mean_dbsm(all_dbsm);
    sum.median_rcs_dbsm = median_of(all_dbsm);

    sum.nose_sector = compute_sector_stats(nose_vals);
    sum.stbd_beam = compute_sector_stats(stbd_vals);
    sum.port_beam = compute_sector_stats(port_vals);
    sum.stern_sector = compute_sector_stats(stern_vals);

    // Sort polar cut points by azimuth
    if (!sum.polar_azimuth_deg.empty()) {
        std::vector<size_t> indices(sum.polar_azimuth_deg.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return sum.polar_azimuth_deg[a] < sum.polar_azimuth_deg[b];
        });
        std::vector<double> s_az, s_rcs;
        s_az.reserve(indices.size());
        s_rcs.reserve(indices.size());
        for (size_t idx : indices) {
            s_az.push_back(sum.polar_azimuth_deg[idx]);
            s_rcs.push_back(sum.polar_rcs_dbsm[idx]);
        }
        sum.polar_azimuth_deg = std::move(s_az);
        sum.polar_rcs_dbsm = std::move(s_rcs);
    }

    // Collect frequency sweep at target direction (closest to nose azimuth=0, el=0)
    if (db.frequencies_hz.size() > 1) {
        size_t nose_dir_idx = 0;
        double min_dist = 1e9;
        for (size_t d = 0; d < db.azimuth_deg.size(); ++d) {
            double dist = std::hypot(db.azimuth_deg[d], db.elevation_deg[d]);
            if (dist < min_dist) {
                min_dist = dist;
                nose_dir_idx = d;
            }
        }
        for (const auto& row : db.rows) {
            if (row.valid && row.direction_index == nose_dir_idx && row.polarization_index == 0) {
                if (row.frequency_index < db.frequencies_hz.size()) {
                    sum.freq_sweep_ghz.push_back(db.frequencies_hz[row.frequency_index] / 1e9);
                    sum.freq_sweep_dbsm.push_back(row.rcs_dbsm);
                }
            }
        }
    }

    return sum;
}

std::string SignatureReportGenerator::generate_polar_svg(
    const std::vector<double>& az_deg,
    const std::vector<double>& rcs_dbsm,
    double max_db,
    double dyn_range_db,
    int size_px) {
    if (az_deg.empty() || az_deg.size() != rcs_dbsm.size()) return "";

    const double min_db = max_db - dyn_range_db;
    const double cx = size_px / 2.0;
    const double cy = size_px / 2.0;
    const double max_r = (size_px / 2.0) - 45.0;

    std::ostringstream svg;
    svg << "<svg viewBox=\"0 0 " << size_px << " " << size_px
        << "\" class=\"radar-svg\" xmlns=\"http://www.w3.org/2000/svg\">\n";

    // Background circle
    svg << "  <circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << max_r
        << "\" fill=\"#080d16\" stroke=\"#1f2d40\" stroke-width=\"1.5\" />\n";

    // Concentric range circles (every 15 dB)
    const int num_rings = 4;
    for (int ring = 1; ring <= num_rings; ++ring) {
        double r = (max_r / num_rings) * ring;
        double level_db = min_db + (dyn_range_db / num_rings) * ring;
        svg << "  <circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << r
            << "\" fill=\"none\" stroke=\"#1a2a3e\" stroke-width=\"1\" stroke-dasharray=\"3,3\" />\n";
        svg << "  <text x=\"" << (cx + 6) << "\" y=\"" << (cy - r + 12)
            << "\" fill=\"#54708f\" font-size=\"10\" font-family=\"monospace\">"
            << std::fixed << std::setprecision(0) << level_db << " dB</text>\n";
    }

    // Radial spokes (every 30 deg)
    for (int deg = 0; deg < 360; deg += 30) {
        double rad = (deg - 90.0) * M_PI / 180.0;
        double x2 = cx + max_r * std::cos(rad);
        double y2 = cy + max_r * std::sin(rad);
        svg << "  <line x1=\"" << cx << "\" y1=\"" << cy << "\" x2=\"" << x2 << "\" y2=\"" << y2
            << "\" stroke=\"#1a2a3e\" stroke-width=\"1\" />\n";

        // Label around outer rim
        double lx = cx + (max_r + 20.0) * std::cos(rad);
        double ly = cy + (max_r + 20.0) * std::sin(rad) + 4.0;
        std::string label = std::to_string(deg) + "°";
        if (deg == 0) label = "0° (Nose)";
        else if (deg == 90) label = "90° (Stbd)";
        else if (deg == 180) label = "180° (Stern)";
        else if (deg == 270) label = "270° (Port)";

        svg << "  <text x=\"" << lx << "\" y=\"" << ly
            << "\" fill=\"#8da2ba\" font-size=\"11\" font-family=\"sans-serif\" text-anchor=\"middle\">"
            << label << "</text>\n";
    }

    // Polygon trace
    std::ostringstream pts;
    for (size_t i = 0; i < az_deg.size(); ++i) {
        double val = std::clamp(rcs_dbsm[i], min_db, max_db);
        double norm_r = (val - min_db) / dyn_range_db;
        double r = norm_r * max_r;
        double rad = (az_deg[i] - 90.0) * M_PI / 180.0;
        double x = cx + r * std::cos(rad);
        double y = cy + r * std::sin(rad);
        if (i > 0) pts << " ";
        pts << std::fixed << std::setprecision(1) << x << "," << y;
    }

    svg << "  <polygon points=\"" << pts.str()
        << "\" fill=\"rgba(0, 210, 255, 0.22)\" stroke=\"#00d2ff\" stroke-width=\"2\" />\n";

    // Center crosshair
    svg << "  <circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"2\" fill=\"#00d2ff\" />\n";
    svg << "</svg>\n";
    return svg.str();
}

std::string SignatureReportGenerator::generate_rectangular_svg(
    const std::vector<double>& x_data,
    const std::vector<double>& y_dbsm,
    const std::string& x_label,
    const std::string& y_label,
    int width_px,
    int height_px) {
    if (x_data.empty() || x_data.size() != y_dbsm.size()) return "";

    const double ml = 65.0, mr = 25.0, mt = 30.0, mb = 45.0;
    const double pw = width_px - ml - mr;
    const double ph = height_px - mt - mb;

    double min_x = *std::min_element(x_data.begin(), x_data.end());
    double max_x = *std::max_element(x_data.begin(), x_data.end());
    if (std::abs(max_x - min_x) < 1e-4) max_x += 1.0;

    double min_y = *std::min_element(y_dbsm.begin(), y_dbsm.end());
    double max_y = *std::max_element(y_dbsm.begin(), y_dbsm.end());
    min_y = std::floor(min_y / 10.0) * 10.0;
    max_y = std::ceil(max_y / 10.0) * 10.0;
    if (std::abs(max_y - min_y) < 1e-4) max_y += 10.0;

    std::ostringstream svg;
    svg << "<svg viewBox=\"0 0 " << width_px << " " << height_px
        << "\" class=\"rect-svg\" xmlns=\"http://www.w3.org/2000/svg\">\n";

    // Chart plot area background
    svg << "  <rect x=\"" << ml << "\" y=\"" << mt << "\" width=\"" << pw << "\" height=\"" << ph
        << "\" fill=\"#080d16\" stroke=\"#1f2d40\" stroke-width=\"1.5\" />\n";

    // Horizontal Y grid lines
    const int y_ticks = 5;
    for (int i = 0; i <= y_ticks; ++i) {
        double y_val = min_y + (max_y - min_y) * (i / static_cast<double>(y_ticks));
        double y_pos = mt + ph - (ph * (i / static_cast<double>(y_ticks)));
        svg << "  <line x1=\"" << ml << "\" y1=\"" << y_pos << "\" x2=\"" << (ml + pw)
            << "\" y2=\"" << y_pos << "\" stroke=\"#1a2a3e\" stroke-width=\"1\" stroke-dasharray=\"3,3\" />\n";
        svg << "  <text x=\"" << (ml - 8) << "\" y=\"" << (y_pos + 4)
            << "\" fill=\"#8da2ba\" font-size=\"10\" font-family=\"monospace\" text-anchor=\"end\">"
            << std::fixed << std::setprecision(0) << y_val << "</text>\n";
    }

    // Vertical X grid lines
    const int x_ticks = 6;
    for (int i = 0; i <= x_ticks; ++i) {
        double x_val = min_x + (max_x - min_x) * (i / static_cast<double>(x_ticks));
        double x_pos = ml + (pw * (i / static_cast<double>(x_ticks)));
        svg << "  <line x1=\"" << x_pos << "\" y1=\"" << mt << "\" x2=\"" << x_pos
            << "\" y2=\"" << (mt + ph) << "\" stroke=\"#1a2a3e\" stroke-width=\"1\" stroke-dasharray=\"3,3\" />\n";
        svg << "  <text x=\"" << x_pos << "\" y=\"" << (mt + ph + 18)
            << "\" fill=\"#8da2ba\" font-size=\"10\" font-family=\"monospace\" text-anchor=\"middle\">"
            << std::fixed << std::setprecision(0) << x_val << "</text>\n";
    }

    // Polyline points
    std::ostringstream pts;
    for (size_t i = 0; i < x_data.size(); ++i) {
        double norm_x = (x_data[i] - min_x) / (max_x - min_x);
        double norm_y = (y_dbsm[i] - min_y) / (max_y - min_y);
        double px = ml + norm_x * pw;
        double py = mt + ph - norm_y * ph;
        if (i > 0) pts << " ";
        pts << std::fixed << std::setprecision(1) << px << "," << py;
    }

    svg << "  <polyline points=\"" << pts.str()
        << "\" fill=\"none\" stroke=\"#38ef7d\" stroke-width=\"2\" />\n";

    // Axis Labels
    svg << "  <text x=\"" << (ml + pw / 2.0) << "\" y=\"" << (height_px - 8)
        << "\" fill=\"#c5d1de\" font-size=\"11\" font-family=\"sans-serif\" text-anchor=\"middle\">"
        << x_label << "</text>\n";
    svg << "  <text x=\"16\" y=\"" << (mt + ph / 2.0)
        << "\" fill=\"#c5d1de\" font-size=\"11\" font-family=\"sans-serif\" text-anchor=\"middle\" "
        << "transform=\"rotate(-90 16," << (mt + ph / 2.0) << ")\">"
        << y_label << "</text>\n";

    svg << "</svg>\n";
    return svg.str();
}

std::string SignatureReportGenerator::generate_html(
    const Hdf5Database& db,
    const SignatureReportConfig& config,
    const NormalizedMesh* mesh,
    const nlohmann::json* resolved_cfg) {

    const SignatureAnalysisSummary summary = analyze(db);

    // Prepare embedded images (Base64)
    std::string isar_b64;
    std::string model_b64;

    // 1. ISAR image
    if (!config.custom_isar_path.empty()) {
        auto b = read_binary_file(config.custom_isar_path);
        if (!b.empty()) isar_b64 = base64_encode(b);
    } else if (config.generate_isar && db.frequencies_hz.size() >= 3 && db.azimuth_deg.size() >= 3) {
        try {
            IsarConfig icfg;
            icfg.dynamic_range_db = 35.0;
            icfg.range_samples = 256;
            icfg.cross_range_samples = 256;
            IsarImage img = IsarProcessor::compute_from_database(db, icfg);
            IsarRenderOptions ropts;
            ropts.width = 720;
            ropts.height = 540;
            ropts.colormap = rendering::parse_colormap(config.colormap);
            rendering::ImageBuffer buf = img.render_image(ropts);
            auto png_bytes = buf.encode_png();
            isar_b64 = base64_encode(png_bytes);
        } catch (...) {
            // Ignore ISAR computation failures gracefully
        }
    }

    // 2. 3D Model image
    if (!config.custom_image_path.empty()) {
        auto b = read_binary_file(config.custom_image_path);
        if (!b.empty()) model_b64 = base64_encode(b);
    } else if (config.generate_3d_render && mesh && !mesh->triangles.empty()) {
        try {
            rendering::ModelRenderConfig mrc;
            mrc.width = 720;
            mrc.height = 540;
            mrc.colormap = rendering::parse_colormap(config.colormap);
            if (config.render_mode == "lit_shadow") mrc.mode = rendering::RenderShadingMode::LitShadow;
            else if (config.render_mode == "shaded") mrc.mode = rendering::RenderShadingMode::ShadedCad;
            else mrc.mode = rendering::RenderShadingMode::SurfaceCurrents;

            Direction dir;
            dir.azimuth_deg = summary.peak_azimuth_deg;
            dir.elevation_deg = summary.peak_elevation_deg;
            dir.k_hat = direction_from_az_el(dir.azimuth_deg, dir.elevation_deg);

            double freq = db.frequencies_hz.empty() ? 10.0e9 : db.frequencies_hz.front();
            std::string pol = db.polarizations.empty() ? "HH" : db.polarizations.front();

            GoOptions go;
            MaterialOptions mat;
            VtkMeshDiagnostics diag;
            if (mrc.mode != rendering::RenderShadingMode::ShadedCad) {
                diag = compute_facet_diagnostics(*mesh, dir, freq, pol, go, mat);
            } else {
                diag.facets.resize(mesh->triangles.size());
            }

            rendering::ImageBuffer mbuf = rendering::ModelRenderer::render_diagnostics(*mesh, diag, dir, freq, pol, mrc);
            auto png_bytes = mbuf.encode_png();
            model_b64 = base64_encode(png_bytes);
        } catch (...) {
            // Ignore model rendering failures gracefully
        }
    }

    // Generate SVGs
    std::string polar_svg = generate_polar_svg(summary.polar_azimuth_deg, summary.polar_rcs_dbsm,
                                               config.polar_max_db, config.polar_dyn_range_db, 460);

    std::string rect_svg = generate_rectangular_svg(summary.polar_azimuth_deg, summary.polar_rcs_dbsm,
                                                    "Azimuth Angle (deg)", "RCS (dBsm)", 580, 260);

    std::string freq_svg;
    if (summary.freq_sweep_ghz.size() > 1) {
        freq_svg = generate_rectangular_svg(summary.freq_sweep_ghz, summary.freq_sweep_dbsm,
                                            "Frequency (GHz)", "RCS (dBsm)", 580, 240);
    }

    // Build self-contained HTML
    std::ostringstream h;
    h << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
      << "<meta charset=\"UTF-8\">\n"
      << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
      << "<title>" << config.title << " - " << config.vehicle_name << "</title>\n"
      << "<style>\n"
      << "  :root {\n"
      << "    --bg-dark: #0a0e17;\n"
      << "    --panel-bg: #131b29;\n"
      << "    --panel-border: #1e2c40;\n"
      << "    --text-primary: #e6edf5;\n"
      << "    --text-secondary: #8da2ba;\n"
      << "    --accent-cyan: #00d2ff;\n"
      << "    --accent-green: #38ef7d;\n"
      << "    --accent-amber: #ffaa00;\n"
      << "    --accent-red: #ff4757;\n"
      << "  }\n"
      << "  * { box-sizing: border-box; margin: 0; padding: 0; }\n"
      << "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;\n"
      << "         background: var(--bg-dark); color: var(--text-primary); padding: 24px; line-height: 1.5; }\n"
      << "  .container { max-width: 1240px; margin: 0 auto; }\n"
      << "  header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px;\n"
      << "           border-bottom: 1px solid var(--panel-border); padding-bottom: 16px; }\n"
      << "  .title-group h1 { font-size: 24px; font-weight: 700; color: #fff; letter-spacing: -0.5px; }\n"
      << "  .title-group p { font-size: 14px; color: var(--text-secondary); margin-top: 4px; }\n"
      << "  .badge { padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 600;\n"
      << "           letter-spacing: 1px; text-transform: uppercase; background: #1a273a; border: 1px solid #2e4362; }\n"
      << "  .badge.success { color: var(--accent-green); border-color: rgba(56, 239, 125, 0.4); }\n"
      << "  .badge.amber { color: var(--accent-amber); border-color: rgba(255, 170, 0, 0.4); }\n"
      << "  .grid-kpi { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }\n"
      << "  .card { background: var(--panel-bg); border: 1px solid var(--panel-border); border-radius: 8px; padding: 18px; }\n"
      << "  .kpi-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-secondary); margin-bottom: 6px; }\n"
      << "  .kpi-val { font-size: 26px; font-weight: 700; font-family: monospace; color: #fff; }\n"
      << "  .kpi-unit { font-size: 13px; font-weight: 400; color: var(--text-secondary); margin-left: 2px; }\n"
      << "  .kpi-sub { font-size: 11px; color: var(--text-secondary); margin-top: 4px; font-family: monospace; }\n"
      << "  .grid-two { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }\n"
      << "  @media (max-width: 900px) { .grid-two { grid-template-columns: 1fr; } }\n"
      << "  .section-title { font-size: 16px; font-weight: 600; margin-bottom: 14px; color: var(--accent-cyan); display: flex; align-items: center; gap: 8px; }\n"
      << "  .chart-wrapper { display: flex; justify-content: center; align-items: center; min-height: 280px; }\n"
      << "  .radar-svg { max-width: 440px; width: 100%; height: auto; }\n"
      << "  .rect-svg { width: 100%; height: auto; }\n"
      << "  .gallery-img { width: 100%; border-radius: 6px; border: 1px solid var(--panel-border); display: block; }\n"
      << "  table { width: 100%; border-collapse: collapse; font-size: 12px; font-family: monospace; }\n"
      << "  th { text-align: left; padding: 8px; color: var(--text-secondary); border-bottom: 1px solid var(--panel-border); }\n"
      << "  td { padding: 8px; border-bottom: 1px solid rgba(30, 44, 64, 0.5); color: var(--text-primary); }\n"
      << "  footer { text-align: center; font-size: 12px; color: var(--text-secondary); margin-top: 32px;\n"
      << "           padding-top: 16px; border-top: 1px solid var(--panel-border); }\n"
      << "</style>\n"
      << "</head>\n<body>\n"
      << "<div class=\"container\">\n"
      << "  <header>\n"
      << "    <div class=\"title-group\">\n"
      << "      <h1>" << config.title << "</h1>\n"
      << "      <p>Target: <strong>" << config.vehicle_name << "</strong> | Solver: Physical Optics (PO/PTD/SBR) Engine</p>\n"
      << "    </div>\n"
      << "    <div>\n"
      << "      <span class=\"badge success\">" << config.classification << "</span>\n"
      << "    </div>\n"
      << "  </header>\n";

    // KPI Cards
    h << "  <div class=\"grid-kpi\">\n"
      << "    <div class=\"card\">\n"
      << "      <div class=\"kpi-title\">Peak RCS</div>\n"
      << "      <div class=\"kpi-val\">" << std::fixed << std::setprecision(1) << summary.peak_rcs_dbsm << "<span class=\"kpi-unit\">dBsm</span></div>\n"
      << "      <div class=\"kpi-sub\">Az " << summary.peak_azimuth_deg << "°, El " << summary.peak_elevation_deg << "°</div>\n"
      << "    </div>\n"
      << "    <div class=\"card\">\n"
      << "      <div class=\"kpi-title\">Median RCS</div>\n"
      << "      <div class=\"kpi-val\">" << std::fixed << std::setprecision(1) << summary.median_rcs_dbsm << "<span class=\"kpi-unit\">dBsm</span></div>\n"
      << "      <div class=\"kpi-sub\">Mean: " << std::setprecision(1) << summary.overall_mean_dbsm << " dBsm</div>\n"
      << "    </div>\n"
      << "    <div class=\"card\">\n"
      << "      <div class=\"kpi-title\">Nose Sector (±15°)</div>\n"
      << "      <div class=\"kpi-val\">" << std::fixed << std::setprecision(1) << summary.nose_sector.mean_dbsm << "<span class=\"kpi-unit\">dBsm</span></div>\n"
      << "      <div class=\"kpi-sub\">Max: " << std::setprecision(1) << summary.nose_sector.max_dbsm << " dBsm</div>\n"
      << "    </div>\n"
      << "    <div class=\"card\">\n"
      << "      <div class=\"kpi-title\">Beam Mean (Stbd/Port)</div>\n"
      << "      <div class=\"kpi-val\">" << std::fixed << std::setprecision(1)
      << ((summary.stbd_beam.mean_dbsm + summary.port_beam.mean_dbsm) * 0.5) << "<span class=\"kpi-unit\">dBsm</span></div>\n"
      << "      <div class=\"kpi-sub\">Stbd: " << std::setprecision(1) << summary.stbd_beam.mean_dbsm
      << " | Port: " << summary.port_beam.mean_dbsm << "</div>\n"
      << "    </div>\n"
      << "    <div class=\"card\">\n"
      << "      <div class=\"kpi-title\">Stern Sector (±15°)</div>\n"
      << "      <div class=\"kpi-val\">" << std::fixed << std::setprecision(1) << summary.stern_sector.mean_dbsm << "<span class=\"kpi-unit\">dBsm</span></div>\n"
      << "      <div class=\"kpi-sub\">Max: " << std::setprecision(1) << summary.stern_sector.max_dbsm << " dBsm</div>\n"
      << "    </div>\n"
      << "  </div>\n";

    // Section 1: Polar and Rectangular RCS Cuts
    h << "  <div class=\"grid-two\">\n"
      << "    <div class=\"card\">\n"
      << "      <div class=\"section-title\">RCS Polar Cut (Principal Plane)</div>\n"
      << "      <div class=\"chart-wrapper\">\n" << polar_svg << "      </div>\n"
      << "    </div>\n"
      << "    <div class=\"card\">\n"
      << "      <div class=\"section-title\">Azimuth RCS Profile</div>\n"
      << "      <div class=\"chart-wrapper\">\n" << rect_svg << "      </div>\n";
    if (!freq_svg.empty()) {
        h << "      <div class=\"section-title\" style=\"margin-top:20px;\">Frequency Sweep Response</div>\n"
          << "      <div class=\"chart-wrapper\">\n" << freq_svg << "      </div>\n";
    }
    h << "    </div>\n"
      << "  </div>\n";

    // Section 2: Visualizations (ISAR & 3D Model Snapshots)
    if (!isar_b64.empty() || !model_b64.empty()) {
        h << "  <div class=\"grid-two\">\n";
        if (!isar_b64.empty()) {
            h << "    <div class=\"card\">\n"
              << "      <div class=\"section-title\">2D ISAR Radar Reflectivity Image</div>\n"
              << "      <img class=\"gallery-img\" src=\"data:image/png;base64," << isar_b64 << "\" alt=\"2D ISAR Heatmap\" />\n"
              << "    </div>\n";
        }
        if (!model_b64.empty()) {
            h << "    <div class=\"card\">\n"
              << "      <div class=\"section-title\">3D Model Simulation Diagnostic Snapshot</div>\n"
              << "      <img class=\"gallery-img\" src=\"data:image/png;base64," << model_b64 << "\" alt=\"3D Simulation Diagnostic\" />\n"
              << "    </div>\n";
        }
        h << "  </div>\n";
    }

    // Section 3: Technical Campaign Metadata Table
    h << "  <div class=\"card\" style=\"margin-bottom:24px;\">\n"
      << "    <div class=\"section-title\">Campaign & Solver Specifications</div>\n"
      << "    <table>\n"
      << "      <thead>\n"
      << "        <tr><th>Parameter</th><th>Value</th><th>Parameter</th><th>Value</th></tr>\n"
      << "      </thead>\n"
      << "      <tbody>\n"
      << "        <tr><td>Frequency Range</td><td>";
    if (!db.frequencies_hz.empty()) {
        h << std::fixed << std::setprecision(2) << (db.frequencies_hz.front() / 1e9) << " - "
          << (db.frequencies_hz.back() / 1e9) << " GHz (" << db.frequencies_hz.size() << " steps)";
    } else {
        h << "N/A";
    }
    h << "</td><td>Polarizations</td><td>";
    for (size_t i = 0; i < db.polarizations.size(); ++i) {
        if (i > 0) h << ", ";
        h << db.polarizations[i];
    }
    h << "</td></tr>\n"
      << "        <tr><td>Azimuth Range</td><td>";
    if (!db.azimuth_deg.empty()) {
        h << db.azimuth_deg.front() << "° to " << db.azimuth_deg.back() << "° (" << db.azimuth_deg.size() << " cuts)";
    } else {
        h << "N/A";
    }
    h << "</td><td>Total Samples</td><td>" << db.rows.size() << " rows</td></tr>\n"
      << "        <tr><td>Schema Version</td><td>" << db.scem_schema_version << "</td>"
      << "<td>Config Hash</td><td>" << db.config_hash.substr(0, 16) << "...</td></tr>\n";
    if (mesh) {
        h << "        <tr><td>Mesh Triangles</td><td>" << mesh->triangles.size() << " facets</td>"
          << "<td>Mesh Vertices</td><td>" << mesh->vertices.size() << "</td></tr>\n"
          << "        <tr><td>Bounding Box</td><td>"
          << std::fixed << std::setprecision(2)
          << (mesh->report.bbox_max.x - mesh->report.bbox_min.x) << " x "
          << (mesh->report.bbox_max.y - mesh->report.bbox_min.y) << " x "
          << (mesh->report.bbox_max.z - mesh->report.bbox_min.z) << " m</td>"
          << "<td>Surface Area</td><td>" << std::setprecision(2) << mesh->report.total_area_m2 << " m²</td></tr>\n";
    }
    h << "      </tbody>\n"
      << "    </table>\n"
      << "  </div>\n";

    // Footer
    h << "  <footer>\n"
      << "    StrikeCEM High-Frequency Computational Electromagnetics Engine &bull; Generated Automatically\n"
      << "  </footer>\n"
      << "</div>\n"
      << "</body>\n</html>\n";

    return h.str();
}

void SignatureReportGenerator::write_report(
    const std::string& hdf5_path,
    const std::string& output_html,
    const SignatureReportConfig& config,
    const std::string& config_or_mesh_path) {

    Hdf5Database db = read_hdf5_database(hdf5_path);
    NormalizedMesh mesh;
    bool has_mesh = false;

    if (!config_or_mesh_path.empty()) {
        if (config_or_mesh_path.rfind(".json") != std::string::npos) {
            ResolvedConfig rc = load_config(config_or_mesh_path, "schemas/config.schema.json");
            mesh = load_normalized_mesh(rc.value, fs::path(config_or_mesh_path).parent_path(), rc.schema_version);
            has_mesh = true;
        } else {
            const fs::path abs_path = fs::absolute(config_or_mesh_path);
            nlohmann::json dummy_cfg = {
                {"model", {
                    {"path", abs_path.string()},
                    {"units", "m"},
                    {"mesh_repair", "report"}
                }},
                {"mesh", {{"max_aspect_ratio", 100.0}}},
                {"scem_schema_version", "1.0"}
            };
            mesh = load_normalized_mesh(dummy_cfg, abs_path.parent_path(), "1.0");
            has_mesh = true;
        }
    }

    std::string html = generate_html(db, config, has_mesh ? &mesh : nullptr);
    std::ofstream out(output_html);
    if (!out) {
        throw std::runtime_error("Cannot write signature report to: " + output_html);
    }
    out << html;
}

} // namespace strikecem::analysis

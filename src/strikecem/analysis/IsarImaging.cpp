#include "strikecem/analysis/IsarImaging.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>
#include <stdexcept>

#include "strikecem/core/Conventions.hpp"

namespace strikecem::analysis {

namespace {

// Palette of 10 shading characters from noise floor to peak
constexpr const char kShades[] = " .:-=+*#%@";
constexpr size_t kNumShades = sizeof(kShades) - 1;

double unwrap_angle(double prev_rad, double current_rad) {
    double diff = current_rad - prev_rad;
    while (diff > std::numbers::pi) diff -= 2.0 * std::numbers::pi;
    while (diff < -std::numbers::pi) diff += 2.0 * std::numbers::pi;
    return prev_rad + diff;
}

} // namespace

IsarImage IsarProcessor::compute_from_matrix(
    const std::vector<double>& frequencies_hz,
    const std::vector<double>& azimuth_deg,
    const std::vector<std::vector<std::complex<double>>>& s_matrix,
    const IsarConfig& config) {

    const size_t M = frequencies_hz.size();
    const size_t N = azimuth_deg.size();

    if (M < 2) {
        throw std::invalid_argument(
            "ISAR imaging requires at least 2 frequency steps (found " + std::to_string(M) + ")");
    }
    if (N < 2) {
        throw std::invalid_argument(
            "ISAR imaging requires at least 2 azimuth steps (found " + std::to_string(N) + ")");
    }
    if (s_matrix.size() != M) {
        throw std::invalid_argument("Scattering matrix row count does not match frequency count");
    }
    for (size_t m = 0; m < M; ++m) {
        if (s_matrix[m].size() != N) {
            throw std::invalid_argument("Scattering matrix column count at row " +
                                        std::to_string(m) + " does not match azimuth count");
        }
    }

    IsarImage img;
    img.polarization = config.polarization;
    img.elevation_deg = config.target_elevation_deg;
    img.dynamic_range_db = config.dynamic_range_db > 0.0 ? config.dynamic_range_db : 35.0;

    // 1. Frequency axis analysis
    img.num_frequencies = M;
    img.f_min_hz = frequencies_hz.front();
    img.f_max_hz = frequencies_hz.back();
    for (double f : frequencies_hz) {
        img.f_min_hz = std::min(img.f_min_hz, f);
        img.f_max_hz = std::max(img.f_max_hz, f);
    }
    img.f_center_hz = 0.5 * (img.f_min_hz + img.f_max_hz);
    img.bandwidth_hz = img.f_max_hz - img.f_min_hz;
    img.f_step_hz = img.bandwidth_hz / static_cast<double>(M - 1);

    const double lambda_c = kSpeedOfLight / img.f_center_hz;

    // 2. Azimuth angle unwrapping & span
    img.num_azimuths = N;
    std::vector<double> az_rad(N);
    az_rad[0] = deg_to_rad(azimuth_deg[0]);
    for (size_t n = 1; n < N; ++n) {
        az_rad[n] = unwrap_angle(az_rad[n - 1], deg_to_rad(azimuth_deg[n]));
    }

    double rad_min = az_rad[0];
    double rad_max = az_rad[0];
    for (double r : az_rad) {
        rad_min = std::min(rad_min, r);
        rad_max = std::max(rad_max, r);
    }
    const double az_span_rad = rad_max - rad_min;
    const double az_center_rad = 0.5 * (rad_min + rad_max);
    const double az_step_rad = az_span_rad / static_cast<double>(N - 1);

    img.az_min_deg = rad_to_deg(rad_min);
    img.az_max_deg = rad_to_deg(rad_max);
    img.az_center_deg = rad_to_deg(az_center_rad);
    img.az_span_deg = rad_to_deg(az_span_rad);
    img.az_step_deg = rad_to_deg(az_step_rad);

    // 3. Radar resolutions
    img.range_resolution_m = kSpeedOfLight / (2.0 * img.bandwidth_hz);
    img.cross_range_resolution_m = lambda_c / (2.0 * (az_span_rad > 1e-12 ? az_span_rad : 1e-12));
    img.unambiguous_range_m = (img.f_step_hz > 0.0) ? (kSpeedOfLight / (2.0 * img.f_step_hz)) : img.range_resolution_m;
    img.unambiguous_cross_range_m = (az_step_rad > 0.0) ? (lambda_c / (2.0 * az_step_rad)) : img.cross_range_resolution_m;

    // 4. Extents and pixel axes
    const size_t NR = config.range_samples > 0 ? config.range_samples : 64;
    const size_t NCR = config.cross_range_samples > 0 ? config.cross_range_samples : 64;
    img.range_samples = NR;
    img.cross_range_samples = NCR;

    const double r_extent = (config.custom_range_extent_m > 0.0) ? config.custom_range_extent_m : img.unambiguous_range_m;
    const double cr_extent = (config.custom_cross_range_extent_m > 0.0) ? config.custom_cross_range_extent_m : img.unambiguous_cross_range_m;

    img.range_axis_m.resize(NR);
    for (size_t p = 0; p < NR; ++p) {
        img.range_axis_m[p] = -0.5 * r_extent + static_cast<double>(p) * (r_extent / static_cast<double>(NR > 1 ? NR - 1 : 1));
    }

    img.cross_range_axis_m.resize(NCR);
    for (size_t q = 0; q < NCR; ++q) {
        img.cross_range_axis_m[q] = -0.5 * cr_extent + static_cast<double>(q) * (cr_extent / static_cast<double>(NCR > 1 ? NCR - 1 : 1));
    }

    // 5. Window weights (Hann window)
    std::vector<double> w_freq(M, 1.0);
    std::vector<double> w_az(N, 1.0);
    if (config.windowing) {
        for (size_t m = 0; m < M; ++m) {
            w_freq[m] = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(m) / static_cast<double>(M - 1));
        }
        for (size_t n = 0; n < N; ++n) {
            w_az[n] = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(n) / static_cast<double>(N - 1));
        }
    }

    // 6. Separable 2D Inversion (Range-Doppler)
    // Step 1: Range compression for each azimuth angle
    // Intermediate matrix: F_R[range_idx][az_idx] of size NR x N
    std::vector<std::vector<std::complex<double>>> F_R(NR, std::vector<std::complex<double>>(N, {0.0, 0.0}));
    const double four_pi_over_c = 4.0 * std::numbers::pi / kSpeedOfLight;

    for (size_t n = 0; n < N; ++n) {
        for (size_t p = 0; p < NR; ++p) {
            const double y_p = img.range_axis_m[p];
            std::complex<double> sum{0.0, 0.0};
            for (size_t m = 0; m < M; ++m) {
                const double df = frequencies_hz[m] - img.f_center_hz;
                const double phase = four_pi_over_c * df * y_p;
                const std::complex<double> kernel(std::cos(phase), std::sin(phase));
                sum += (w_freq[m] * kernel) * s_matrix[m][n];
            }
            F_R[p][n] = sum;
        }
    }

    // Step 2: Cross-range compression for each range bin
    img.complex_image.assign(NR, std::vector<std::complex<double>>(NCR, {0.0, 0.0}));
    const double norm_scale = 1.0 / static_cast<double>(M * N);
    const double cr_phase_factor = four_pi_over_c * img.f_center_hz;

    double max_mag = 0.0;
    size_t peak_p = 0;
    size_t peak_q = 0;

    for (size_t p = 0; p < NR; ++p) {
        for (size_t q = 0; q < NCR; ++q) {
            const double x_q = img.cross_range_axis_m[q];
            std::complex<double> sum{0.0, 0.0};
            for (size_t n = 0; n < N; ++n) {
                const double d_theta = az_rad[n] - az_center_rad;
                const double phase = cr_phase_factor * d_theta * x_q;
                const std::complex<double> kernel(std::cos(phase), std::sin(phase));
                sum += (w_az[n] * kernel) * F_R[p][n];
            }
            const std::complex<double> val = sum * norm_scale;
            img.complex_image[p][q] = val;
            const double mag = std::abs(val);
            if (mag > max_mag) {
                max_mag = mag;
                peak_p = p;
                peak_q = q;
            }
        }
    }

    img.peak_magnitude = max_mag;
    img.peak_range_m = img.range_axis_m[peak_p];
    img.peak_cross_range_m = img.cross_range_axis_m[peak_q];

    // 7. Decibel normalization and dynamic range clamping
    img.intensity_db.assign(NR, std::vector<double>(NCR, -img.dynamic_range_db));
    const double ref_mag = max_mag > 1e-15 ? max_mag : 1e-15;

    for (size_t p = 0; p < NR; ++p) {
        for (size_t q = 0; q < NCR; ++q) {
            const double mag = std::abs(img.complex_image[p][q]);
            double db = 20.0 * std::log10((mag + 1e-15) / ref_mag);
            if (db < -img.dynamic_range_db) db = -img.dynamic_range_db;
            if (db > 0.0) db = 0.0;
            img.intensity_db[p][q] = db;
        }
    }

    return img;
}

IsarImage IsarProcessor::compute_from_database(const Hdf5Database& db,
                                               const IsarConfig& config) {
    if (db.frequencies_hz.size() < 2) {
        throw std::invalid_argument("Database has fewer than 2 frequencies");
    }
    if (db.azimuth_deg.size() < 2) {
        throw std::invalid_argument("Database has fewer than 2 direction samples");
    }

    // Determine polarization
    uint16_t target_pol_idx = 0;
    bool found_pol = false;
    for (size_t i = 0; i < db.polarizations.size(); ++i) {
        if (db.polarizations[i] == config.polarization) {
            target_pol_idx = static_cast<uint16_t>(i);
            found_pol = true;
            break;
        }
    }
    if (!found_pol) {
        target_pol_idx = 0;
    }

    // Filter directions by elevation closest to target_elevation_deg
    double best_el_diff = std::numeric_limits<double>::infinity();
    double chosen_elevation = 0.0;
    for (double el : db.elevation_deg) {
        double diff = std::abs(el - config.target_elevation_deg);
        if (diff < best_el_diff) {
            best_el_diff = diff;
            chosen_elevation = el;
        }
    }

    struct DirMatch {
        uint32_t dir_idx;
        double az_deg;
    };
    std::vector<DirMatch> matched_dirs;
    for (size_t i = 0; i < db.azimuth_deg.size(); ++i) {
        if (std::abs(db.elevation_deg[i] - chosen_elevation) < 1e-3) {
            matched_dirs.push_back({static_cast<uint32_t>(i), db.azimuth_deg[i]});
        }
    }

    if (matched_dirs.size() < 2) {
        throw std::invalid_argument("Fewer than 2 azimuth directions found at elevation " +
                                    std::to_string(chosen_elevation) + " deg");
    }

    // Sort directions by azimuth
    std::sort(matched_dirs.begin(), matched_dirs.end(),
              [](const DirMatch& a, const DirMatch& b) { return a.az_deg < b.az_deg; });

    // Map direction_index to column index n
    std::vector<int> dir_to_col(db.azimuth_deg.size(), -1);
    std::vector<double> sorted_az(matched_dirs.size());
    for (size_t n = 0; n < matched_dirs.size(); ++n) {
        dir_to_col[matched_dirs[n].dir_idx] = static_cast<int>(n);
        sorted_az[n] = matched_dirs[n].az_deg;
    }

    const size_t M = db.frequencies_hz.size();
    const size_t N = matched_dirs.size();

    // Populate scattering matrix from rows
    std::vector<std::vector<std::complex<double>>> s_matrix(
        M, std::vector<std::complex<double>>(N, {0.0, 0.0}));

    for (const auto& row : db.rows) {
        if (row.polarization_index != target_pol_idx) continue;
        if (row.direction_index >= dir_to_col.size()) continue;
        const int col = dir_to_col[row.direction_index];
        if (col < 0) continue;
        if (row.frequency_index >= M) continue;

        s_matrix[row.frequency_index][col] =
            std::complex<double>(row.scattering_real, row.scattering_imag);
    }

    IsarConfig cfg = config;
    cfg.target_elevation_deg = chosen_elevation;
    if (!found_pol && !db.polarizations.empty()) {
        cfg.polarization = db.polarizations.front();
    }

    return compute_from_matrix(db.frequencies_hz, sorted_az, s_matrix, cfg);
}

IsarImage IsarProcessor::compute_from_file(const std::string& hdf5_path,
                                           const IsarConfig& config) {
    const Hdf5Database db = read_hdf5_database(hdf5_path);
    return compute_from_database(db, config);
}

std::string IsarImage::to_ascii_heatmap(size_t max_rows, size_t max_cols) const {
    if (range_samples == 0 || cross_range_samples == 0) return "[Empty ISAR image]\n";

    std::ostringstream oss;
    oss << "=== StrikeCEM 2D ISAR Radar Image ===\n";
    oss << "  Polarization: " << polarization << " | Elevation: " << elevation_deg << " deg\n";
    oss << "  Frequency: [" << std::fixed << std::setprecision(3) << (f_min_hz * 1e-9) << " - "
        << (f_max_hz * 1e-9) << "] GHz (Center: " << (f_center_hz * 1e-9)
        << " GHz, Bandwidth: " << (bandwidth_hz * 1e-9) << " GHz, " << num_frequencies << " steps)\n";
    oss << "  Azimuth:   [" << std::setprecision(1) << az_min_deg << " to " << az_max_deg
        << "] deg (Span: " << az_span_deg << " deg, " << num_azimuths << " angles)\n";
    oss << "  Resolutions: Down-Range = " << std::setprecision(3) << range_resolution_m
        << " m | Cross-Range = " << cross_range_resolution_m << " m\n";
    oss << "  Peak Scatterer: Range = " << (peak_range_m >= 0 ? "+" : "") << peak_range_m
        << " m, Cross-Range = " << (peak_cross_range_m >= 0 ? "+" : "") << peak_cross_range_m
        << " m (Mag: " << std::scientific << peak_magnitude << std::fixed << ")\n";
    oss << "  Dynamic Range: " << std::setprecision(1) << dynamic_range_db << " dB\n\n";

    // Subsample/bin to display dimensions
    const size_t disp_rows = std::min(range_samples, max_rows);
    const size_t disp_cols = std::min(cross_range_samples, max_cols);

    oss << " Down-Range (m) ^  [+Front / -Back]\n";
    oss << "                |\n";

    // Row loop: from maximum range (top) down to minimum range (bottom)
    for (size_t r = 0; r < disp_rows; ++r) {
        // Map display row r in [0, disp_rows - 1] to original range index p
        // r = 0 -> top (highest range index)
        const size_t p = (disp_rows > 1)
            ? (range_samples - 1 - (r * (range_samples - 1) / (disp_rows - 1)))
            : (range_samples - 1);
        const double r_val = range_axis_m[p];

        oss << std::setw(7) << std::setprecision(2) << std::showpos << r_val << std::noshowpos << " m |";

        for (size_t c = 0; c < disp_cols; ++c) {
            const size_t q = (disp_cols > 1)
                ? (c * (cross_range_samples - 1) / (disp_cols - 1))
                : 0;
            const double val_db = intensity_db[p][q];

            // Normalize val_db in [-D, 0] to [0, kNumShades - 1]
            const double frac = std::clamp((val_db + dynamic_range_db) / dynamic_range_db, 0.0, 1.0);
            const size_t shade_idx = std::clamp(static_cast<size_t>(frac * static_cast<double>(kNumShades - 1) + 0.5),
                                                size_t{0}, kNumShades - 1);
            oss << kShades[shade_idx];
        }
        oss << "|\n";
    }

    oss << "        --------+";
    for (size_t c = 0; c < disp_cols; ++c) oss << "-";
    oss << "+------> Cross-Range (m) [+Starboard / -Port]\n";

    // Cross-range axis annotations
    oss << "                ";
    const double cr_min = cross_range_axis_m.front();
    const double cr_mid = cross_range_axis_m[cross_range_axis_m.size() / 2];
    const double cr_max = cross_range_axis_m.back();

    std::ostringstream tick_oss;
    tick_oss << std::setprecision(2) << std::showpos << cr_min << "m";
    const std::string s_min = tick_oss.str();
    tick_oss.str("");
    tick_oss << std::setprecision(2) << std::showpos << cr_mid << "m";
    const std::string s_mid = tick_oss.str();
    tick_oss.str("");
    tick_oss << std::setprecision(2) << std::showpos << cr_max << "m";
    const std::string s_max = tick_oss.str();

    oss << s_min;
    int pad1 = static_cast<int>(disp_cols / 2) - static_cast<int>(s_min.size()) - static_cast<int>(s_mid.size() / 2);
    for (int i = 0; i < std::max(1, pad1); ++i) oss << " ";
    oss << s_mid;
    int pad2 = static_cast<int>(disp_cols) - (static_cast<int>(disp_cols / 2) + static_cast<int>(s_mid.size() / 2) + static_cast<int>(s_max.size()));
    for (int i = 0; i < std::max(1, pad2); ++i) oss << " ";
    oss << s_max << "\n\n";

    // Legend
    oss << "  Intensity Legend: [ -" << std::setprecision(0) << dynamic_range_db << " dB ] ";
    for (size_t i = 0; i < kNumShades; ++i) oss << kShades[i] << " ";
    oss << "[ 0 dB (Peak) ]\n";

    return oss.str();
}

nlohmann::json IsarImage::to_json() const {
    nlohmann::json root;
    root["scem_version"] = SCEM_VERSION;
    root["polarization"] = polarization;
    root["elevation_deg"] = elevation_deg;

    root["frequency"] = {
        {"min_hz", f_min_hz},
        {"max_hz", f_max_hz},
        {"center_hz", f_center_hz},
        {"bandwidth_hz", bandwidth_hz},
        {"step_hz", f_step_hz},
        {"num_samples", num_frequencies}
    };

    root["azimuth"] = {
        {"min_deg", az_min_deg},
        {"max_deg", az_max_deg},
        {"center_deg", az_center_deg},
        {"span_deg", az_span_deg},
        {"step_deg", az_step_deg},
        {"num_samples", num_azimuths}
    };

    root["resolutions"] = {
        {"range_resolution_m", range_resolution_m},
        {"cross_range_resolution_m", cross_range_resolution_m},
        {"unambiguous_range_m", unambiguous_range_m},
        {"unambiguous_cross_range_m", unambiguous_cross_range_m}
    };

    root["image"] = {
        {"range_samples", range_samples},
        {"cross_range_samples", cross_range_samples},
        {"dynamic_range_db", dynamic_range_db},
        {"peak_range_m", peak_range_m},
        {"peak_cross_range_m", peak_cross_range_m},
        {"peak_magnitude", peak_magnitude},
        {"range_axis_m", range_axis_m},
        {"cross_range_axis_m", cross_range_axis_m},
        {"intensity_db", intensity_db}
    };

    return root;
}

void IsarImage::write_json(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot write ISAR JSON file: " + path);
    }
    out << to_json().dump(2) << "\n";
}

rendering::ImageBuffer IsarImage::render_image(const IsarRenderOptions& opts) const {
    const size_t W = opts.width > 0 ? opts.width : 800;
    const size_t H = opts.height > 0 ? opts.height : 800;
    rendering::ImageBuffer img(W, H, {16, 20, 28, 255});

    if (range_samples == 0 || cross_range_samples == 0) return img;

    const int ML = 70; // Margin left
    const int MR = 90; // Margin right (for colorbar)
    const int MT = 55; // Margin top (for title)
    const int MB = 60; // Margin bottom (for axis label)

    const int plot_x0 = ML;
    const int plot_y0 = MT;
    const int plot_w = static_cast<int>(W) - ML - MR;
    const int plot_h = static_cast<int>(H) - MT - MB;

    if (plot_w <= 0 || plot_h <= 0) return img;

    // Render 2D heatmap with bilinear interpolation
    const double r_min = range_axis_m.front();
    const double r_max = range_axis_m.back();
    const double cr_min = cross_range_axis_m.front();
    const double cr_max = cross_range_axis_m.back();

    for (int py = 0; py < plot_h; ++py) {
        // py = 0 is top (max range), py = plot_h - 1 is bottom (min range)
        const double v = (plot_h > 1) ? (1.0 - static_cast<double>(py) / static_cast<double>(plot_h - 1)) : 0.5;
        const double p_cont = v * static_cast<double>(range_samples - 1);
        const size_t p0 = std::clamp(static_cast<size_t>(p_cont), size_t{0}, range_samples - 1);
        const size_t p1 = std::min(p0 + 1, range_samples - 1);
        const double fp = p_cont - static_cast<double>(p0);

        for (int px = 0; px < plot_w; ++px) {
            // px = 0 is left (min cross-range), px = plot_w - 1 is right (max cross-range)
            const double u = (plot_w > 1) ? (static_cast<double>(px) / static_cast<double>(plot_w - 1)) : 0.5;
            const double q_cont = u * static_cast<double>(cross_range_samples - 1);
            const size_t q0 = std::clamp(static_cast<size_t>(q_cont), size_t{0}, cross_range_samples - 1);
            const size_t q1 = std::min(q0 + 1, cross_range_samples - 1);
            const double fq = q_cont - static_cast<double>(q0);

            const double v00 = intensity_db[p0][q0];
            const double v01 = intensity_db[p0][q1];
            const double v10 = intensity_db[p1][q0];
            const double v11 = intensity_db[p1][q1];

            const double val_db = (1.0 - fp) * (1.0 - fq) * v00 +
                                  (1.0 - fp) * fq * v01 +
                                  fp * (1.0 - fq) * v10 +
                                  fp * fq * v11;

            const double norm_val = std::clamp((val_db + dynamic_range_db) / dynamic_range_db, 0.0, 1.0);
            const rendering::Rgba color = rendering::evaluate_colormap(opts.colormap, norm_val);

            img.set_pixel(plot_x0 + px, plot_y0 + py, color);
        }
    }

    // Grid lines and Axis borders
    if (opts.show_axes) {
        img.draw_rect(plot_x0, plot_y0, plot_w, plot_h, {70, 90, 120, 255}, false);

        // Down-range ticks (vertical axis)
        const int num_ticks = 5;
        for (int i = 0; i < num_ticks; ++i) {
            const double frac = static_cast<double>(i) / static_cast<double>(num_ticks - 1);
            const int ty = plot_y0 + static_cast<int>((1.0 - frac) * (plot_h - 1));
            const double r_val = r_min + frac * (r_max - r_min);

            // Subtle grid line
            if (i > 0 && i < num_ticks - 1) {
                for (int x = plot_x0 + 1; x < plot_x0 + plot_w - 1; x += 4) {
                    img.set_pixel(x, ty, {40, 50, 70, 160});
                }
            }

            img.draw_line(plot_x0 - 5, ty, plot_x0, ty, {180, 190, 210, 255});
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << (r_val >= 0 ? "+" : "") << r_val << "m";
            img.draw_text(plot_x0 - 64, ty - 4, ss.str(), {190, 205, 225, 255}, 1);
        }

        // Cross-range ticks (horizontal axis)
        for (int i = 0; i < num_ticks; ++i) {
            const double frac = static_cast<double>(i) / static_cast<double>(num_ticks - 1);
            const int tx = plot_x0 + static_cast<int>(frac * (plot_w - 1));
            const double cr_val = cr_min + frac * (cr_max - cr_min);

            // Subtle grid line
            if (i > 0 && i < num_ticks - 1) {
                for (int y = plot_y0 + 1; y < plot_y0 + plot_h - 1; y += 4) {
                    img.set_pixel(tx, y, {40, 50, 70, 160});
                }
            }

            img.draw_line(tx, plot_y0 + plot_h, tx, plot_y0 + plot_h + 5, {180, 190, 210, 255});
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << (cr_val >= 0 ? "+" : "") << cr_val << "m";
            img.draw_text(tx - 24, plot_y0 + plot_h + 10, ss.str(), {190, 205, 225, 255}, 1);
        }

        // Axis Titles
        img.draw_text(plot_x0 + plot_w / 2 - 64, plot_y0 + plot_h + 36, "CROSS-RANGE (m)", {220, 230, 245, 255}, 1);
        img.draw_text(10, MT - 18, "DOWN-RANGE (m)", {220, 230, 245, 255}, 1);
    }

    // Peak scatterer reticle marker
    if (opts.show_peak_marker && peak_magnitude > 0.0) {
        const double r_span = r_max - r_min > 1e-9 ? (r_max - r_min) : 1.0;
        const double cr_span = cr_max - cr_min > 1e-9 ? (cr_max - cr_min) : 1.0;

        const double u_pk = std::clamp((peak_cross_range_m - cr_min) / cr_span, 0.0, 1.0);
        const double v_pk = std::clamp((peak_range_m - r_min) / r_span, 0.0, 1.0);

        const int pk_x = plot_x0 + static_cast<int>(u_pk * (plot_w - 1));
        const int pk_y = plot_y0 + static_cast<int>((1.0 - v_pk) * (plot_h - 1));

        // Bright white reticle circle and crosshairs
        img.draw_circle(pk_x, pk_y, 7, {255, 255, 255, 255}, false);
        img.draw_circle(pk_x, pk_y, 8, {0, 0, 0, 180}, false);
        img.draw_line(pk_x - 12, pk_y, pk_x - 9, pk_y, {255, 255, 255, 255});
        img.draw_line(pk_x + 9, pk_y, pk_x + 12, pk_y, {255, 255, 255, 255});
        img.draw_line(pk_x, pk_y - 12, pk_x, pk_y - 9, {255, 255, 255, 255});
        img.draw_line(pk_x, pk_y + 9, pk_x, pk_y + 12, {255, 255, 255, 255});
    }

    // Colorbar on right
    if (opts.show_colorbar) {
        const int cb_x = static_cast<int>(W) - 65;
        const int cb_y = plot_y0;
        const int cb_w = 16;
        const int cb_h = plot_h;

        img.draw_rect(cb_x - 1, cb_y - 1, cb_w + 2, cb_h + 2, {70, 90, 120, 255}, false);

        for (int j = 0; j < cb_h; ++j) {
            const double t_val = 1.0 - (static_cast<double>(j) / static_cast<double>(cb_h - 1));
            const rendering::Rgba c = rendering::evaluate_colormap(opts.colormap, t_val);
            for (int i = 0; i < cb_w; ++i) {
                img.set_pixel(cb_x + i, cb_y + j, c);
            }
        }

        // dB labels
        const int num_cb_labels = 5;
        for (int i = 0; i < num_cb_labels; ++i) {
            const double frac = static_cast<double>(i) / static_cast<double>(num_cb_labels - 1);
            const int ly = cb_y + static_cast<int>(frac * (cb_h - 1));
            const double db_val = -frac * dynamic_range_db;
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(0) << db_val << "dB";
            img.draw_line(cb_x + cb_w, ly, cb_x + cb_w + 4, ly, {180, 190, 210, 255});
            img.draw_text(cb_x + cb_w + 6, ly - 4, ss.str(), {190, 205, 225, 255}, 1);
        }
    }

    // Title banner
    img.draw_rect(0, 0, static_cast<int>(W), 40, {10, 14, 22, 240}, true);
    img.draw_line(0, 40, static_cast<int>(W), 40, {45, 65, 95, 255});

    img.draw_text(16, 10, "STRIKECEM 2D ISAR RADAR HEATMAP", {255, 255, 255, 255}, 1);

    std::ostringstream sub_oss;
    sub_oss << "Band: [" << std::fixed << std::setprecision(2) << (f_min_hz * 1e-9) << "-"
            << (f_max_hz * 1e-9) << "] GHz (B=" << (bandwidth_hz * 1e-9) << " GHz) | Pol: "
            << polarization << " | Res: dR=" << std::setprecision(3) << range_resolution_m
            << "m, dCR=" << cross_range_resolution_m << "m";
    img.draw_text(16, 24, sub_oss.str(), {160, 200, 245, 255}, 1);

    return img;
}

void IsarImage::write_image(const std::string& path, const IsarRenderOptions& opts) const {
    const rendering::ImageBuffer img = render_image(opts);
    img.write_image(path);
}

} // namespace strikecem::analysis

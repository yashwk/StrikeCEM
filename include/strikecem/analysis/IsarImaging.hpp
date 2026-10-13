#pragma once

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/io/Hdf5Reader.hpp"
#include "strikecem/rendering/Image.hpp"

namespace strikecem::analysis {

struct IsarRenderOptions {
    size_t width = 800;
    size_t height = 800;
    rendering::ColormapType colormap = rendering::ColormapType::Turbo;
    bool show_axes = true;
    bool show_colorbar = true;
    bool show_peak_marker = true;
};

struct IsarConfig {
    std::string polarization = "HH";
    double target_elevation_deg = 0.0;
    size_t range_samples = 64;
    size_t cross_range_samples = 64;
    double dynamic_range_db = 35.0;
    bool windowing = true; // Hann windowing on both axes
    double custom_range_extent_m = 0.0;       // 0.0 = auto (full unambiguous range)
    double custom_cross_range_extent_m = 0.0; // 0.0 = auto (full unambiguous cross-range)
};

struct IsarImage {
    std::string polarization;
    double elevation_deg = 0.0;

    // Frequency axis metadata
    double f_min_hz = 0.0;
    double f_max_hz = 0.0;
    double f_center_hz = 0.0;
    double bandwidth_hz = 0.0;
    double f_step_hz = 0.0;
    size_t num_frequencies = 0;

    // Azimuth axis metadata
    double az_min_deg = 0.0;
    double az_max_deg = 0.0;
    double az_center_deg = 0.0;
    double az_span_deg = 0.0;
    double az_step_deg = 0.0;
    size_t num_azimuths = 0;

    // Physical radar resolutions
    double range_resolution_m = 0.0;       // c / (2 * B)
    double cross_range_resolution_m = 0.0; // lambda_c / (2 * delta_theta_rad)
    double unambiguous_range_m = 0.0;      // c / (2 * delta_f)
    double unambiguous_cross_range_m = 0.0;// lambda_c / (2 * delta_theta_step_rad)

    // Image grid and peak
    size_t range_samples = 0;
    size_t cross_range_samples = 0;
    double dynamic_range_db = 35.0;
    double peak_range_m = 0.0;
    double peak_cross_range_m = 0.0;
    double peak_magnitude = 0.0;

    // Coordinates:
    // Range axis (down-range): [ -range_extent/2, +range_extent/2 ]
    // Cross-range axis:        [ -cross_range_extent/2, +cross_range_extent/2 ]
    std::vector<double> range_axis_m;
    std::vector<double> cross_range_axis_m;

    // 2D Normalized dB intensity matrix: intensity_db[range_idx][cross_range_idx]
    // Normalized so peak is 0.0 dB, clipped at -dynamic_range_db.
    std::vector<std::vector<double>> intensity_db;

    // Complex spatial reconstructed field (unnormalized)
    std::vector<std::vector<std::complex<double>>> complex_image;

    // Export & Rendering
    std::string to_ascii_heatmap(size_t max_rows = 25, size_t max_cols = 50) const;
    nlohmann::json to_json() const;
    void write_json(const std::string& path) const;

    // 2D Image rendering (PNG/BMP/PPM heatmap)
    rendering::ImageBuffer render_image(const IsarRenderOptions& opts = {}) const;
    void write_image(const std::string& path, const IsarRenderOptions& opts = {}) const;
};

class IsarProcessor {
public:
    // Compute 2D ISAR image from an open Hdf5Database struct
    static IsarImage compute_from_database(const Hdf5Database& db,
                                           const IsarConfig& config = {});

    // Compute 2D ISAR image directly from an HDF5 database file on disk
    static IsarImage compute_from_file(const std::string& hdf5_path,
                                       const IsarConfig& config = {});

    // Compute 2D ISAR image from frequency/azimuth vectors and complex scattering matrix
    // s_matrix dimensions: [num_frequencies][num_azimuths]
    static IsarImage compute_from_matrix(
        const std::vector<double>& frequencies_hz,
        const std::vector<double>& azimuth_deg,
        const std::vector<std::vector<std::complex<double>>>& s_matrix,
        const IsarConfig& config = {});
};

} // namespace strikecem::analysis

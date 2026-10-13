#pragma once
// Interactive standalone HTML signature report generator.
// Compiles 3D model renders, 2D ISAR radar heatmaps, interactive SVG polar
// and rectangular RCS cuts, and sector statistics into a self-contained,
// offline-capable HTML report.
#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/Hdf5Reader.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/analysis/IsarImaging.hpp"
#include "strikecem/rendering/Colormap.hpp"

namespace strikecem::analysis {

struct SignatureReportConfig {
    std::string title = "StrikeCEM Radar Signature Diagnostic Report";
    std::string vehicle_name = "Target Vehicle";
    std::string classification = "UNCLASSIFIED";
    std::string output_path = "signature_report.html";
    std::string custom_image_path;
    std::string custom_isar_path;
    bool generate_isar = true;
    bool generate_3d_render = true;
    std::string colormap = "turbo";
    std::string render_mode = "currents"; // "currents", "lit_shadow", "shaded"
    double polar_dyn_range_db = 60.0;    // Dynamic range on polar plot (dBsm)
    double polar_max_db = 20.0;           // Outer ring dBsm level
};

struct SectorStats {
    double min_dbsm = 0.0;
    double max_dbsm = 0.0;
    double mean_dbsm = 0.0; // 10*log10(mean(10^(dbsm/10)))
    double median_dbsm = 0.0;
    size_t sample_count = 0;
};

struct SignatureAnalysisSummary {
    double peak_rcs_dbsm = -999.0;
    double peak_azimuth_deg = 0.0;
    double peak_elevation_deg = 0.0;
    double peak_frequency_hz = 0.0;
    double median_rcs_dbsm = 0.0;
    double overall_mean_dbsm = 0.0;

    // Standard aerospace aspect sectors
    SectorStats nose_sector;   // [-15 deg, +15 deg] (azimuth 345..360, 0..15)
    SectorStats stbd_beam;     // [75 deg, 105 deg]
    SectorStats port_beam;     // [255 deg, 285 deg]
    SectorStats stern_sector;  // [165 deg, 195 deg]

    // Principal elevation polar cut (azimuth 0..360 at primary elevation)
    std::vector<double> polar_azimuth_deg;
    std::vector<double> polar_rcs_dbsm;

    // Frequency sweep curve (frequency vs dBsm at nose or peak aspect)
    std::vector<double> freq_sweep_ghz;
    std::vector<double> freq_sweep_dbsm;
};

class SignatureReportGenerator {
public:
    // Extract key statistics and 1D/2D cuts from an HDF5 database
    static SignatureAnalysisSummary analyze(const Hdf5Database& db);

    // Generate standalone self-contained HTML report string
    static std::string generate_html(const Hdf5Database& db,
                                     const SignatureReportConfig& config,
                                     const NormalizedMesh* mesh = nullptr,
                                     const nlohmann::json* resolved_cfg = nullptr);

    // Convenience end-to-end writer from files on disk
    static void write_report(const std::string& hdf5_path,
                             const std::string& output_html,
                             const SignatureReportConfig& config = {},
                             const std::string& config_or_mesh_path = "");

    // Utility: encode binary data to Base64 string
    static std::string base64_encode(const std::vector<uint8_t>& data);
    static std::string base64_encode(const std::string& data);

    // SVG Generators
    static std::string generate_polar_svg(const std::vector<double>& az_deg,
                                          const std::vector<double>& rcs_dbsm,
                                          double max_db = 20.0,
                                          double dyn_range_db = 60.0,
                                          int size_px = 480);

    static std::string generate_rectangular_svg(const std::vector<double>& x_data,
                                                const std::vector<double>& y_dbsm,
                                                const std::string& x_label = "Azimuth (deg)",
                                                const std::string& y_label = "RCS (dBsm)",
                                                int width_px = 640,
                                                int height_px = 280);
};

} // namespace strikecem::analysis

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

#include "strikecem/analysis/IsarImaging.hpp"
#include "strikecem/core/Conventions.hpp"

namespace strikecem {
namespace {

namespace fs = std::filesystem;

// Helper to generate a synthetic scattering matrix from a list of point scatterers
// Each scatterer: (x = down-range, y = cross-range, amplitude)
struct PointScatterer {
    double x; // down-range (m)
    double y; // cross-range (m)
    double amplitude;
};

std::vector<std::vector<std::complex<double>>> generate_point_scatterer_matrix(
    const std::vector<double>& freqs,
    const std::vector<double>& azimuths_deg,
    const std::vector<PointScatterer>& scatterers) {

    const size_t M = freqs.size();
    const size_t N = azimuths_deg.size();
    std::vector<std::vector<std::complex<double>>> S(M, std::vector<std::complex<double>>(N, {0.0, 0.0}));

    const double four_pi_over_c = 4.0 * std::numbers::pi / kSpeedOfLight;

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            const double az_rad = deg_to_rad(azimuths_deg[n]);
            std::complex<double> val{0.0, 0.0};
            for (const auto& sc : scatterers) {
                // Distance along radar line of sight at azimuth angle az_rad:
                // r = x * cos(az) + y * sin(az)
                const double r = sc.x * std::cos(az_rad) + sc.y * std::sin(az_rad);
                const double phase = -four_pi_over_c * freqs[m] * r;
                val += sc.amplitude * std::complex<double>(std::cos(phase), std::sin(phase));
            }
            S[m][n] = val;
        }
    }
    return S;
}

} // namespace

TEST(IsarImaging, PointScattererResolutionAndPeakLocation) {
    // Frequency band: 9 GHz to 11 GHz (X-band, Bandwidth = 2 GHz)
    // Down-range resolution: c / (2 * 2e9) = 0.0749 m (~7.5 cm)
    // M = 64 -> delta_f = 2e9 / 63 = 31.75 MHz -> unambiguous range R_max = c / (2 * delta_f) = 4.72 m
    const size_t M = 64;
    std::vector<double> freqs(M);
    for (size_t i = 0; i < M; ++i) {
        freqs[i] = 9.0e9 + static_cast<double>(i) * (2.0e9 / static_cast<double>(M - 1));
    }

    // Azimuth span: -5 deg to +5 deg (10 deg span = 0.1745 rad)
    // Cross-range resolution: lambda_c / (2 * delta_theta) = 0.03 / (2 * 0.1745) = 0.086 m (~8.6 cm)
    const size_t N = 33;
    std::vector<double> az(N);
    for (size_t i = 0; i < N; ++i) {
        az[i] = -5.0 + static_cast<double>(i) * (10.0 / static_cast<double>(N - 1));
    }

    // Place a single point scatterer at x = 1.2 m, y = -0.5 m (well within R_max/2 = 2.36 m)
    const PointScatterer target{1.2, -0.5, 1.0};
    const auto s_matrix = generate_point_scatterer_matrix(freqs, az, {target});

    analysis::IsarConfig cfg;
    cfg.range_samples = 128;
    cfg.cross_range_samples = 128;
    cfg.windowing = false; // test unwindowed peak accuracy

    const analysis::IsarImage img = analysis::IsarProcessor::compute_from_matrix(freqs, az, s_matrix, cfg);

    EXPECT_NEAR(img.bandwidth_hz, 2.0e9, 1e3);
    EXPECT_NEAR(img.range_resolution_m, kSpeedOfLight / 4.0e9, 1e-4);
    EXPECT_NEAR(img.az_span_deg, 10.0, 1e-4);

    // Peak location should accurately recover target (1.2, -0.5) within resolution cell
    EXPECT_NEAR(img.peak_range_m, target.x, img.range_resolution_m);
    EXPECT_NEAR(img.peak_cross_range_m, target.y, img.cross_range_resolution_m);

    // Peak intensity in normalized dB must be 0.0 dB
    EXPECT_DOUBLE_EQ(img.intensity_db[img.range_samples / 2][img.cross_range_samples / 2] <= 0.0, true);
}

TEST(IsarImaging, MultipleScatterersDetection) {
    const size_t M = 64;
    std::vector<double> freqs(M);
    for (size_t i = 0; i < M; ++i) {
        freqs[i] = 9.0e9 + static_cast<double>(i) * (2.0e9 / static_cast<double>(M - 1));
    }

    const size_t N = 33;
    std::vector<double> az(N);
    for (size_t i = 0; i < N; ++i) {
        az[i] = -6.0 + static_cast<double>(i) * (12.0 / static_cast<double>(N - 1));
    }

    // Two scatterers: Primary at (+1.5, 0.0) with amp 1.0, Secondary at (0.0, +1.0) with amp 0.7
    // Both are within unambiguous range R_max/2 = 2.36 m
    const PointScatterer p1{1.5, 0.0, 1.0};
    const PointScatterer p2{0.0, 1.0, 0.7};
    const auto s_matrix = generate_point_scatterer_matrix(freqs, az, {p1, p2});

    analysis::IsarConfig cfg;
    cfg.range_samples = 64;
    cfg.cross_range_samples = 64;
    cfg.windowing = true; // Hann windowing

    const analysis::IsarImage img = analysis::IsarProcessor::compute_from_matrix(freqs, az, s_matrix, cfg);

    // Primary peak is p1
    EXPECT_NEAR(img.peak_range_m, p1.x, img.range_resolution_m);
    EXPECT_NEAR(img.peak_cross_range_m, p1.y, img.cross_range_resolution_m);

    // Find pixel closest to p2
    size_t p2_r_idx = 0;
    size_t p2_cr_idx = 0;
    double min_dist_r = 1e9;
    double min_dist_cr = 1e9;
    for (size_t p = 0; p < img.range_samples; ++p) {
        if (std::abs(img.range_axis_m[p] - p2.x) < min_dist_r) {
            min_dist_r = std::abs(img.range_axis_m[p] - p2.x);
            p2_r_idx = p;
        }
    }
    for (size_t q = 0; q < img.cross_range_samples; ++q) {
        if (std::abs(img.cross_range_axis_m[q] - p2.y) < min_dist_cr) {
            min_dist_cr = std::abs(img.cross_range_axis_m[q] - p2.y);
            p2_cr_idx = q;
        }
    }

    // p2 relative amplitude in dB is 20 * log10(0.7) ~= -3.1 dB
    const double p2_db = img.intensity_db[p2_r_idx][p2_cr_idx];
    EXPECT_NEAR(p2_db, -3.1, 2.5); // within 2.5 dB tolerance with windowing
}

TEST(IsarImaging, AsciiHeatmapRendering) {
    const size_t M = 16;
    std::vector<double> freqs(M);
    for (size_t i = 0; i < M; ++i) freqs[i] = 10.0e9 + i * 1e8;

    const size_t N = 16;
    std::vector<double> az(N);
    for (size_t i = 0; i < N; ++i) az[i] = -4.0 + i * 0.5;

    const auto s_matrix = generate_point_scatterer_matrix(freqs, az, {{0.0, 0.0, 1.0}});
    const analysis::IsarImage img = analysis::IsarProcessor::compute_from_matrix(freqs, az, s_matrix);

    const std::string ascii = img.to_ascii_heatmap(20, 30);
    EXPECT_FALSE(ascii.empty());
    EXPECT_NE(ascii.find("StrikeCEM 2D ISAR Radar Image"), std::string::npos);
    EXPECT_NE(ascii.find("Down-Range"), std::string::npos);
    EXPECT_NE(ascii.find("Cross-Range"), std::string::npos);
    EXPECT_NE(ascii.find("Intensity Legend"), std::string::npos);
}

TEST(IsarImaging, JsonExportAndFileWriting) {
    const size_t M = 8;
    std::vector<double> freqs(M);
    for (size_t i = 0; i < M; ++i) freqs[i] = 10.0e9 + i * 2e8;

    const size_t N = 8;
    std::vector<double> az(N);
    for (size_t i = 0; i < N; ++i) az[i] = -2.0 + i * 0.5;

    const auto s_matrix = generate_point_scatterer_matrix(freqs, az, {{0.5, -0.2, 1.0}});
    const analysis::IsarImage img = analysis::IsarProcessor::compute_from_matrix(freqs, az, s_matrix);

    const nlohmann::json j = img.to_json();
    EXPECT_TRUE(j.contains("scem_version"));
    EXPECT_TRUE(j.contains("frequency"));
    EXPECT_TRUE(j.contains("azimuth"));
    EXPECT_TRUE(j.contains("resolutions"));
    EXPECT_TRUE(j.contains("image"));
    EXPECT_EQ(j["image"]["range_samples"], 64);
    EXPECT_EQ(j["image"]["cross_range_samples"], 64);

    const fs::path tmp_json = fs::temp_directory_path() / "test_isar_export.json";
    img.write_json(tmp_json.string());
    EXPECT_TRUE(fs::exists(tmp_json));
    EXPECT_GT(fs::file_size(tmp_json), 100u);
    fs::remove(tmp_json);
}

TEST(IsarImaging, ValidationAndErrorGuards) {
    // Fewer than 2 frequencies throws
    EXPECT_THROW(analysis::IsarProcessor::compute_from_matrix({10.0e9}, {0.0, 1.0}, {{{1.0, 0.0}}}),
                 std::invalid_argument);

    // Fewer than 2 azimuths throws
    EXPECT_THROW(analysis::IsarProcessor::compute_from_matrix({10.0e9, 11.0e9}, {0.0}, {{{1.0, 0.0}}, {{1.0, 0.0}}}),
                 std::invalid_argument);

    // Row length mismatch throws
    EXPECT_THROW(analysis::IsarProcessor::compute_from_matrix({10.0e9, 11.0e9}, {0.0, 1.0}, {{{1.0, 0.0}}}),
                 std::invalid_argument);
}

TEST(IsarImaging, DatabaseIntegration) {
    Hdf5Database db;
    db.polarizations = {"HH", "VV"};
    db.frequencies_hz = {9.0e9, 9.5e9, 10.0e9, 10.5e9};
    db.azimuth_deg = {0.0, 1.0, 2.0};
    db.elevation_deg = {0.0, 0.0, 0.0};

    // Populate rows
    uint64_t sid = 0;
    for (uint32_t di = 0; di < db.azimuth_deg.size(); ++di) {
        for (uint32_t fi = 0; fi < db.frequencies_hz.size(); ++fi) {
            for (uint16_t pi = 0; pi < db.polarizations.size(); ++pi) {
                Hdf5SampleRow row;
                row.sample_id = sid++;
                row.direction_index = di;
                row.frequency_index = fi;
                row.polarization_index = pi;
                row.scattering_real = 1.0;
                row.scattering_imag = 0.0;
                row.valid = true;
                db.rows.push_back(row);
            }
        }
    }

    analysis::IsarConfig cfg;
    cfg.polarization = "HH";
    cfg.range_samples = 16;
    cfg.cross_range_samples = 16;

    const analysis::IsarImage img = analysis::IsarProcessor::compute_from_database(db, cfg);
    EXPECT_EQ(img.num_frequencies, 4u);
    EXPECT_EQ(img.num_azimuths, 3u);
    EXPECT_EQ(img.polarization, "HH");
    EXPECT_EQ(img.range_samples, 16u);
    EXPECT_EQ(img.cross_range_samples, 16u);
}

} // namespace strikecem

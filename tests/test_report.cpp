#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <vector>

#include "strikecem/analysis/SignatureReport.hpp"
#include "strikecem/io/Hdf5Reader.hpp"
#include "strikecem/io/Hdf5Writer.hpp"

namespace fs = std::filesystem;
using namespace strikecem;
using namespace strikecem::analysis;

TEST(SignatureReport, Base64Encoding) {
    std::string text = "Hello StrikeCEM Report Engine!";
    std::string b64 = SignatureReportGenerator::base64_encode(text);
    EXPECT_FALSE(b64.empty());
    // Known Base64 properties: length must be multiple of 4
    EXPECT_EQ(b64.size() % 4, 0);

    // Byte vector encoding
    std::vector<uint8_t> bytes = {0x00, 0xFF, 0x7F, 0x80};
    std::string b64_bytes = SignatureReportGenerator::base64_encode(bytes);
    EXPECT_EQ(b64_bytes.size() % 4, 0);
}

TEST(SignatureReport, SectorStatisticsAndAnalysis) {
    Hdf5Database db;
    db.scem_schema_version = "1.0";
    db.config_hash = "abcdef0123456789abcdef0123456789";
    db.frequencies_hz = {10.0e9};
    db.polarizations = {"HH"};

    // 360-degree azimuth sweep (0..359, 1 deg step), el = 0
    for (int az = 0; az < 360; ++az) {
        db.azimuth_deg.push_back(static_cast<double>(az));
        db.elevation_deg.push_back(0.0);

        Hdf5SampleRow row;
        row.sample_id = az;
        row.direction_index = az;
        row.frequency_index = 0;
        row.polarization_index = 0;
        row.valid = true;

        // Model strong specular flash at 90 deg and 270 deg (beam), low at nose
        if (az == 90 || az == 270) {
            row.rcs_dbsm = 25.0;
        } else if (az >= 345 || az <= 15) {
            row.rcs_dbsm = -10.0; // nose sector
        } else {
            row.rcs_dbsm = 0.0;
        }
        row.rcs_sqm = std::pow(10.0, row.rcs_dbsm / 10.0);
        db.rows.push_back(row);
    }

    SignatureAnalysisSummary sum = SignatureReportGenerator::analyze(db);

    EXPECT_DOUBLE_EQ(sum.peak_rcs_dbsm, 25.0);
    EXPECT_EQ(sum.polar_azimuth_deg.size(), 360);
    EXPECT_EQ(sum.polar_rcs_dbsm.size(), 360);

    // Verify nose sector statistics
    EXPECT_GT(sum.nose_sector.sample_count, 0);
    EXPECT_NEAR(sum.nose_sector.mean_dbsm, -10.0, 1e-4);

    // Verify beam sectors have high peak/mean
    EXPECT_GT(sum.stbd_beam.max_dbsm, 20.0);
    EXPECT_GT(sum.port_beam.max_dbsm, 20.0);
}

TEST(SignatureReport, SvgChartGeneration) {
    std::vector<double> az = {0.0, 90.0, 180.0, 270.0};
    std::vector<double> rcs = {-10.0, 15.0, -5.0, 15.0};

    std::string polar_svg = SignatureReportGenerator::generate_polar_svg(az, rcs, 20.0, 50.0, 400);
    EXPECT_FALSE(polar_svg.empty());
    EXPECT_NE(polar_svg.find("<svg"), std::string::npos);
    EXPECT_NE(polar_svg.find("<polygon points="), std::string::npos);
    EXPECT_NE(polar_svg.find("0° (Nose)"), std::string::npos);
    EXPECT_NE(polar_svg.find("90° (Stbd)"), std::string::npos);

    std::string rect_svg = SignatureReportGenerator::generate_rectangular_svg(az, rcs, "Azimuth", "dBsm", 500, 250);
    EXPECT_FALSE(rect_svg.empty());
    EXPECT_NE(rect_svg.find("<svg"), std::string::npos);
    EXPECT_NE(rect_svg.find("<polyline points="), std::string::npos);
    EXPECT_NE(rect_svg.find("Azimuth"), std::string::npos);
}

TEST(SignatureReport, FullHtmlReportGeneration) {
    Hdf5Database db;
    db.scem_schema_version = "1.0";
    db.config_hash = "11223344556677889900aabbccddeeff";
    db.frequencies_hz = {9.0e9, 10.0e9, 11.0e9};
    db.polarizations = {"HH", "VV"};

    for (int az = 0; az < 180; az += 10) {
        db.azimuth_deg.push_back(static_cast<double>(az));
        db.elevation_deg.push_back(0.0);
    }

    uint64_t sid = 0;
    for (size_t f = 0; f < db.frequencies_hz.size(); ++f) {
        for (size_t d = 0; d < db.azimuth_deg.size(); ++d) {
            for (size_t p = 0; p < db.polarizations.size(); ++p) {
                Hdf5SampleRow row;
                row.sample_id = sid++;
                row.direction_index = d;
                row.frequency_index = f;
                row.polarization_index = p;
                row.rcs_dbsm = (d == 9) ? 12.5 : -5.0; // peak at 90 deg
                row.rcs_sqm = std::pow(10.0, row.rcs_dbsm / 10.0);
                row.valid = true;
                db.rows.push_back(row);
            }
        }
    }

    SignatureReportConfig cfg;
    cfg.title = "Stealth Drone Signature Report";
    cfg.vehicle_name = "Experimental UCAV";
    cfg.generate_isar = false;
    cfg.generate_3d_render = false;

    std::string html = SignatureReportGenerator::generate_html(db, cfg);
    EXPECT_FALSE(html.empty());
    EXPECT_NE(html.find("<!DOCTYPE html>"), std::string::npos);
    EXPECT_NE(html.find("Stealth Drone Signature Report"), std::string::npos);
    EXPECT_NE(html.find("Experimental UCAV"), std::string::npos);
    EXPECT_NE(html.find("Peak RCS"), std::string::npos);
    EXPECT_NE(html.find("Nose Sector (±15°)"), std::string::npos);
    EXPECT_NE(html.find("radar-svg"), std::string::npos);
}

TEST(SignatureReport, EndToEndFileReportWriter) {
    const fs::path test_html = fs::temp_directory_path() / "test_report.html";
    const std::string h5_path = "examples/ucav/ucav_baseline_po.h5";
    const std::string cfg_path = "examples/ucav/ucav_baseline_po.json";

    if (fs::exists(h5_path)) {
        SignatureReportConfig cfg;
        cfg.title = "UCAV Baseline Report";
        cfg.vehicle_name = "Strike UCAV";
        cfg.generate_isar = false;
        cfg.generate_3d_render = true;

        SignatureReportGenerator::write_report(h5_path, test_html.string(), cfg, cfg_path);

        EXPECT_TRUE(fs::exists(test_html));
        EXPECT_GT(fs::file_size(test_html), 1000);

        std::string content;
        std::ifstream in(test_html);
        std::string line;
        while (std::getline(in, line)) content += line + "\n";
        EXPECT_NE(content.find("<!DOCTYPE html>"), std::string::npos);
        EXPECT_NE(content.find("Strike UCAV"), std::string::npos);

        fs::remove(test_html);
    }
}

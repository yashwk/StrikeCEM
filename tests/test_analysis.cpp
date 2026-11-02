#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <gtest/gtest.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "strikecem/analysis/Comparison.hpp"
#include "strikecem/api/RcsQueryEngine.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/io/BinaryTransport.hpp"
#include "strikecem/io/Hdf5Writer.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace {

namespace fs = std::filesystem;

struct Workdir {
    fs::path dir = fs::temp_directory_path() / ("scem_analysis_test_" + std::to_string(::getpid()));
    Workdir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
        fs::create_directories(dir, ec);
    }
    ~Workdir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

struct Pipeline {
    strikecem::ResolvedConfig rc;
    strikecem::SamplePlan plan;
    strikecem::NormalizedMesh mesh;
    strikecem::PoResult result;
};

Pipeline run_pipeline(const nlohmann::json& cfg, const fs::path& config_path) {
    {
        std::ofstream out(config_path);
        out << cfg.dump(2);
    }
    Pipeline p;
    p.rc = strikecem::load_config(config_path.string(), SCEM_SCHEMA_PATH);
    p.plan = strikecem::build_sample_plan(p.rc.value);
    p.mesh = strikecem::load_normalized_mesh(p.rc.value, config_path.parent_path(),
                                             p.rc.schema_version);
    p.result = strikecem::solve_po(p.mesh, p.plan, p.rc.value);
    return p;
}

nlohmann::json make_config(const std::string& output_path, const std::string& format) {
    nlohmann::json cfg;
    cfg["scem_schema_version"] = "1.0";
    cfg["model"] = {{"path", std::string(SCEM_FIXTURE_DIR) + "/../../examples/plate.stl"}};
    cfg["frequency"] = {{"frequency_hz", 10e9}};
    cfg["angles"]["azimuth"] = {{"start", 0}, {"stop", 90}, {"step", 30}};
    cfg["angles"]["elevation"] = {{"start", -90}, {"stop", -90}, {"step", 1}};
    cfg["polarization"] = {"HH", "VV"};
    cfg["solver"] = {{"type", "PO"}};
    cfg["output"] = {{"path", output_path}, {"format", format}};
    cfg["integration"] = {
        {"designer", {
            {"design_id", "ANALYSIS-TEST-001"},
            {"revision", "v1.0"}
        }}
    };
    return cfg;
}

} // namespace

TEST(AnalysisTest, SummarizeDatabase) {
    Workdir w;
    fs::path h5_file = w.dir / "out.h5";
    Pipeline p = run_pipeline(make_config(h5_file.string(), "hdf5"), w.dir / "cfg.json");
    strikecem::write_hdf5_output(p.rc, p.plan, p.mesh, p.result);

    auto engine = strikecem::RcsQueryEngine::open_file(h5_file.string());
    ASSERT_NE(engine, nullptr);

    auto summary = strikecem::analysis::summarize_database(*engine);
    EXPECT_EQ(summary.design_id, "ANALYSIS-TEST-001");
    EXPECT_EQ(summary.design_revision, "v1.0");
    EXPECT_EQ(summary.frequencies_hz.size(), 1u);
    EXPECT_DOUBLE_EQ(summary.frequencies_hz[0], 10e9);
    EXPECT_EQ(summary.polarizations.size(), 2u);
    EXPECT_GT(summary.direction_count, 0u);
    EXPECT_GE(summary.max_rcs_dbsm, summary.min_rcs_dbsm);
    EXPECT_GT(summary.mean_rcs_dbsm, -200.0);

    nlohmann::json j = strikecem::analysis::summary_to_json(summary);
    EXPECT_EQ(j["design_id"], "ANALYSIS-TEST-001");
    EXPECT_EQ(j["polarizations"].size(), 2u);
}

TEST(AnalysisTest, ExtractCutAndRenderAscii) {
    Workdir w;
    fs::path scem_file = w.dir / "out.scem";
    Pipeline p = run_pipeline(make_config(scem_file.string(), "scem"), w.dir / "cfg.json");
    strikecem::write_scem_output(p.rc, p.plan, p.mesh, p.result);

    auto engine = strikecem::RcsQueryEngine::open_file(scem_file.string());
    ASSERT_NE(engine, nullptr);

    // Azimuth cut
    auto az_cut = strikecem::analysis::extract_cut(*engine, "azimuth", -90.0, 10e9, "HH", 10.0);
    EXPECT_EQ(az_cut.cut_type, "azimuth");
    EXPECT_FALSE(az_cut.samples.empty());
    EXPECT_DOUBLE_EQ(az_cut.samples.front().angle_deg, 0.0);

    // Elevation cut
    auto el_cut = strikecem::analysis::extract_cut(*engine, "elevation", 0.0, 10e9, "HH", 30.0);
    EXPECT_EQ(el_cut.cut_type, "elevation");
    EXPECT_FALSE(el_cut.samples.empty());
    EXPECT_DOUBLE_EQ(el_cut.samples.front().angle_deg, -90.0);

    // Render ASCII
    std::string ascii = strikecem::analysis::render_ascii_cut(az_cut, 40);
    EXPECT_FALSE(ascii.empty());
    EXPECT_NE(ascii.find("1D azimuth Cut"), std::string::npos);
    EXPECT_NE(ascii.find("#"), std::string::npos);

    // Invalid parameters throw
    EXPECT_THROW(strikecem::analysis::extract_cut(*engine, "invalid_cut", 0.0, 10e9, "HH"), std::invalid_argument);
    EXPECT_THROW(strikecem::analysis::extract_cut(*engine, "azimuth", 0.0, 10e9, "HH", -5.0), std::invalid_argument);
}

TEST(AnalysisTest, CompareDatabasesSelf) {
    Workdir w;
    fs::path h5_file = w.dir / "out.h5";
    Pipeline p = run_pipeline(make_config(h5_file.string(), "hdf5"), w.dir / "cfg.json");
    strikecem::write_hdf5_output(p.rc, p.plan, p.mesh, p.result);

    auto engine_a = strikecem::RcsQueryEngine::open_file(h5_file.string());
    auto engine_b = strikecem::RcsQueryEngine::open_file(h5_file.string());

    auto metrics = strikecem::analysis::compare_databases(*engine_a, *engine_b, 0.5);
    EXPECT_GT(metrics.sample_count, 0u);
    EXPECT_DOUBLE_EQ(metrics.max_absolute_error_dbsm, 0.0);
    EXPECT_DOUBLE_EQ(metrics.mean_absolute_error_dbsm, 0.0);
    EXPECT_DOUBLE_EQ(metrics.rmse_dbsm, 0.0);
    EXPECT_NEAR(metrics.complex_correlation, 1.0, 1e-9);
    EXPECT_DOUBLE_EQ(metrics.peak_rcs_delta_dbsm, 0.0);
    EXPECT_TRUE(metrics.passed);

    nlohmann::json j = strikecem::analysis::comparison_to_json(metrics);
    EXPECT_TRUE(j["passed"]);
    EXPECT_DOUBLE_EQ(j["max_absolute_error_dbsm"], 0.0);
}

TEST(RcsQueryEngine, RegularAzimuthInterpolationWrapsAtTheSeam) {
    strikecem::ScemDatabase db;
    db.frequencies_hz = {10.0e9};
    db.polarizations = {"HH"};
    const std::vector<std::pair<double, double>> samples = {
        {0.0, 0.0}, {180.0, 180.0}, {350.0, 350.0}};
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto [az, value] = samples[i];
        db.directions.emplace_back(static_cast<uint32_t>(i), az, 0.0,
                                   strikecem::direction_from_az_el(az, 0.0));
        strikecem::Hdf5SampleRow row;
        row.direction_index = static_cast<uint32_t>(i);
        row.scattering_real = value;
        db.rows.push_back(row);
    }

    strikecem::RcsQueryEngine engine(std::move(db));
    const auto linear = engine.query(359.0, 0.0, 10.0e9, "HH",
                                     strikecem::InterpolationMode::Linear);
    const auto bicubic = engine.query(359.0, 0.0, 10.0e9, "HH",
                                      strikecem::InterpolationMode::Bicubic);
    const auto nearest = engine.query(359.0, 0.0, 10.0e9, "HH",
                                      strikecem::InterpolationMode::Nearest);
    EXPECT_NEAR(linear.scattering_real, 35.0, 1e-9);
    EXPECT_NEAR(bicubic.scattering_real, 35.0, 1e-9);
    EXPECT_DOUBLE_EQ(nearest.scattering_real, 0.0);
}

TEST(RcsQueryEngine, IrregularNearestUsesAngularDistance) {
    strikecem::ScemDatabase db;
    db.frequencies_hz = {10.0e9};
    db.polarizations = {"HH"};
    const std::vector<std::tuple<double, double, double>> samples = {
        {5.0, 60.0, 200.0}, {90.0, 80.0, 300.0}, {180.0, 0.0, 100.0}};
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto [az, el, value] = samples[i];
        db.directions.emplace_back(static_cast<uint32_t>(i), az, el,
                                   strikecem::direction_from_az_el(az, el));
        strikecem::Hdf5SampleRow row;
        row.direction_index = static_cast<uint32_t>(i);
        row.scattering_real = value;
        db.rows.push_back(row);
    }

    strikecem::RcsQueryEngine engine(std::move(db));
    const auto nearest = engine.query(0.0, 80.0, 10.0e9, "HH",
                                      strikecem::InterpolationMode::Nearest);
    EXPECT_DOUBLE_EQ(nearest.scattering_real, 300.0);
}

TEST(RcsQueryEngine, SphericalIndexMatchesBruteForceInterpolation) {
    strikecem::ScemDatabase db;
    db.frequencies_hz = {10.0e9};
    db.polarizations = {"HH"};
    std::vector<std::pair<geom::Vec3d, double>> directions;
    for (size_t i = 0; i < 31; ++i) {
        const double az = std::fmod(17.0 + 47.0 * i, 360.0);
        const double el = -70.0 + std::fmod(31.0 * i, 140.0);
        const double value = 1.0 + static_cast<double>(i);
        const auto k = strikecem::direction_from_az_el(az, el);
        directions.emplace_back(k, value);
        db.directions.emplace_back(static_cast<uint32_t>(i), az, el, k);
        strikecem::Hdf5SampleRow row;
        row.direction_index = static_cast<uint32_t>(i);
        row.scattering_real = value;
        db.rows.push_back(row);
    }

    const double query_az = 43.0;
    const double query_el = 13.0;
    const auto q = strikecem::direction_from_az_el(query_az, query_el);
    struct Candidate { double dot; double value; };
    std::vector<Candidate> candidates;
    for (const auto& [k, value] : directions)
        candidates.push_back({q.x * k.x + q.y * k.y + q.z * k.z, value});
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.dot > b.dot;
    });
    candidates.resize(8);
    std::array<double, 8> theta{};
    for (size_t i = 0; i < theta.size(); ++i)
        theta[i] = std::acos(std::clamp(candidates[i].dot, -1.0, 1.0));
    const double radius = std::max(1.5 * theta.back(), 1e-4);
    double weight_sum = 0.0;
    double value_sum = 0.0;
    for (size_t i = 0; i < theta.size(); ++i) {
        const double u = std::clamp(theta[i] / radius, 0.0, 1.0);
        const double w = std::pow(1.0 - u, 4.0) * (4.0 * u + 1.0) /
                         (theta[i] * theta[i] + 1e-9);
        weight_sum += w;
        value_sum += w * candidates[i].value;
    }

    strikecem::RcsQueryEngine engine(std::move(db));
    const auto result = engine.query(query_az, query_el, 10.0e9, "HH",
                                     strikecem::InterpolationMode::Spherical);
    EXPECT_NEAR(result.scattering_real, value_sum / weight_sum, 1e-9);
}

#include <filesystem>
#include <fstream>
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

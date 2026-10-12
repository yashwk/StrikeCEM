// Tests for Track 2: Master-coordinated adaptive coarse-to-refine sampling.
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <unistd.h>

#include <H5Cpp.h>
#include <nlohmann/json.hpp>

#include "strikecem/adaptive/AdaptiveSampling.hpp"
#include "strikecem/app/CLI.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/io/Hdf5Reader.hpp"
#include "strikecem/io/MeshLoader.hpp"

namespace {

namespace fs = std::filesystem;

struct Workdir {
    fs::path dir = fs::temp_directory_path() / ("scem_adapt_test_" + std::to_string(::getpid()) + "_" +
                                                std::to_string(reinterpret_cast<uintptr_t>(this)));
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

int call_cli(const std::vector<std::string>& arg_list) {
    std::vector<std::string> storage = {"strikecem"};
    for (const auto& a : arg_list) storage.push_back(a);
    std::vector<char*> argv;
    for (auto& s : storage) argv.push_back(s.data());
    return strikecem::cli::run(static_cast<int>(argv.size()), argv.data());
}

nlohmann::json make_adaptive_config(const fs::path& out_path, const std::string& fmt = "hdf5",
                                    double coarse_step = 10.0, double refine_step = 2.0) {
    nlohmann::json cfg;
    cfg["scem_schema_version"] = "1.0";
    cfg["model"] = {
        {"path", std::string(std::string(SCEM_FIXTURE_DIR) + "/../../examples/plate.stl")},
        {"transform", {{"rotate_euler_deg", {90, 0, 0}}}}
    };
    cfg["frequency"] = {{"frequency_hz", 10e9}};
    cfg["angles"]["azimuth"] = {{"start", 60}, {"stop", 120}, {"step", 10}};
    cfg["angles"]["elevation"] = {{"start", 0}, {"stop", 0}, {"step", 1}};
    cfg["angles"]["adaptive"] = {
        {"coarse_step_deg", coarse_step},
        {"refine_step_deg", refine_step},
        {"gradient_threshold_db_per_deg", 1.5},
        {"window_half_width_deg", 5.0},
        {"max_refinement_stages", 1}
    };
    cfg["polarization"] = {"HH", "VV"};
    cfg["solver"] = {{"type", "PO"}};
    cfg["run"] = {{"checkpoint_every_n_samples", 2}};
    cfg["output"] = {{"path", out_path.string()}, {"format", fmt}};
    return cfg;
}

fs::path write_json(const fs::path& path, const nlohmann::json& j) {
    std::ofstream out(path);
    out << j.dump(2);
    return path;
}

} // namespace

TEST(Adaptive, ParseParams) {
    nlohmann::json j;
    j["angles"]["adaptive"] = {
        {"coarse_step_deg", 5.0},
        {"refine_step_deg", 0.5},
        {"peak_threshold_dbsm", -10.0},
        {"gradient_threshold_db_per_deg", 3.0},
        {"window_half_width_deg", 4.0},
        {"max_refinement_stages", 2}
    };
    const auto p = strikecem::parse_adaptive_params(j);
    EXPECT_DOUBLE_EQ(p.coarse_step_deg, 5.0);
    EXPECT_DOUBLE_EQ(p.refine_step_deg, 0.5);
    ASSERT_TRUE(p.peak_threshold_dbsm.has_value());
    EXPECT_DOUBLE_EQ(*p.peak_threshold_dbsm, -10.0);
    EXPECT_DOUBLE_EQ(p.gradient_threshold_db_per_deg, 3.0);
    EXPECT_DOUBLE_EQ(p.window_half_width_deg, 4.0);
    EXPECT_EQ(p.max_refinement_stages, 2);
}

TEST(Adaptive, WindowExpansionAndMerging) {
    strikecem::SamplePlan coarse_plan;
    coarse_plan.frequencies_hz = {10e9};
    coarse_plan.polarizations = {"HH"};
    for (uint32_t i = 0; i < 7; ++i) {
        coarse_plan.directions.push_back({i, static_cast<double>(i * 10), 0.0, {}});
    }

    strikecem::PoResult coarse_result;
    // Synthetic response with a peak at 20 deg (i = 2) and 30 deg (i = 3)
    // 0 deg: 0 dBsm, 10 deg: 5 dBsm, 20 deg: 20 dBsm, 30 deg: 18 dBsm, 40 deg: -5 dBsm
    const std::vector<double> dbsm = {0.0, 5.0, 20.0, 18.0, -5.0, -20.0, -30.0};
    for (uint32_t i = 0; i < 7; ++i) {
        strikecem::PoSampleResult r;
        r.direction_id = i;
        r.rcs_sqm = std::pow(10.0, dbsm[i] / 10.0);
        coarse_result.samples.push_back(r);
    }

    strikecem::AdaptiveParams params;
    params.coarse_step_deg = 10.0;
    params.refine_step_deg = 1.0;
    params.window_half_width_deg = 5.0;
    params.gradient_threshold_db_per_deg = 1.0; // 20 -> 18 is 0.2 db/deg, but 30 -> 40 is 2.3 db/deg

    size_t peaks = 0, grads = 0;
    const auto windows = strikecem::detect_and_merge_windows(
        coarse_plan, coarse_result, params, 0.0, 60.0, &peaks, &grads);

    EXPECT_GT(peaks, 0u);
    EXPECT_GT(grads, 0u);
    // Overlapping windows around 20 and 35 should merge
    ASSERT_FALSE(windows.empty());
    for (const auto& w : windows) {
        EXPECT_GE(w.start_deg, 0.0);
        EXPECT_LE(w.stop_deg, 60.0);
        EXPECT_LE(w.start_deg, w.stop_deg);
    }
}

TEST(Adaptive, RefinementDeduplicationAgainstCoarse) {
    strikecem::ResolvedConfig rc;
    strikecem::SamplePlan coarse_plan;
    coarse_plan.frequencies_hz = {10e9};
    coarse_plan.polarizations = {"HH"};
    coarse_plan.directions = {
        {0, 0.0, 0.0, {}},
        {1, 10.0, 0.0, {}},
        {2, 20.0, 0.0, {}}
    };
    coarse_plan.hash = "parent_hash_123";

    // Refinement window [5.0, 15.0] with step 5.0 -> would generate 5.0, 10.0, 15.0.
    // 10.0 must be deduplicated out because it's already in coarse_plan!
    std::vector<strikecem::Interval> windows = {{5.0, 15.0}};
    const auto ref_plan = strikecem::build_refinement_plan(rc, coarse_plan, windows, 5.0);

    ASSERT_EQ(ref_plan.directions.size(), 2u);
    EXPECT_DOUBLE_EQ(ref_plan.directions[0].azimuth_deg, 5.0);
    EXPECT_DOUBLE_EQ(ref_plan.directions[1].azimuth_deg, 15.0);
    // Hash must be computed
    EXPECT_FALSE(ref_plan.hash.empty());
}

TEST(Adaptive, EndToEndSolvePlateSpecularLobe) {
    Workdir w;
    const fs::path out_path = w.dir / "adaptive_plate.h5";
    const fs::path cfg_file = write_json(w.dir / "config.json", make_adaptive_config(out_path, "hdf5", 10.0, 2.0));

    const auto rc = strikecem::load_config(cfg_file.string(), SCEM_SCHEMA_PATH);
    const auto mesh = strikecem::load_normalized_mesh(rc.value, w.dir, rc.schema_version);

    const auto res = strikecem::solve_adaptive(mesh, rc);

    // Coarse plan: 0 to 60 deg with 10 deg step = 7 directions * 2 pols = 14 samples
    EXPECT_EQ(res.coarse_result.samples.size(), 14u);
    EXPECT_GT(res.peaks_detected, 0u);
    EXPECT_FALSE(res.windows.empty());
    // Refinement directions generated
    EXPECT_GT(res.refinement_result.samples.size(), 0u);
    // Merged result must contain both coarse and refinement points
    EXPECT_GT(res.merged_result.samples.size(), res.coarse_result.samples.size());
    EXPECT_EQ(res.merged_plan.sample_count(), res.merged_result.samples.size());
}

TEST(Adaptive, CliExecutionEndToEnd) {
    Workdir w;
    const fs::path out_path = w.dir / "cli_adaptive.h5";
    const fs::path cfg_file = write_json(w.dir / "config.json", make_adaptive_config(out_path, "hdf5", 10.0, 2.0));

    ASSERT_EQ(call_cli({"run", cfg_file.string()}), 0);
    ASSERT_TRUE(fs::exists(out_path));

    const auto db = strikecem::read_hdf5_database(out_path.string());
    strikecem::validate_hdf5_complete(db);
    // More samples than just coarse grid
    EXPECT_GT(db.rows.size(), 14u);
}

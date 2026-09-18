// Resource estimator tests: byte accounting, fit/refusal math, resolution
// warnings, benchmark calibration matching, and thread-count invariance.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "strikecem/app/CLI.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/runtime/Estimate.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace {

namespace fs = std::filesystem;

std::string fixture(const std::string& name) { return std::string(SCEM_FIXTURE_DIR) + "/" + name; }

struct Case {
    strikecem::ResolvedConfig rc;
    strikecem::NormalizedMesh mesh;
    strikecem::SamplePlan plan;
};

Case load_case(const std::string& name) {
    Case c;
    c.rc = strikecem::load_config(fixture(name), SCEM_SCHEMA_PATH);
    c.plan = strikecem::build_sample_plan(c.rc.value);
    c.mesh = strikecem::load_normalized_mesh(c.rc.value, SCEM_FIXTURE_DIR, c.rc.schema_version);
    return c;
}

TEST(Estimate, ByteAccountingAndFit) {
    const Case c = load_case("valid_minimal.json"); // 2 tris, 4 verts, 181 rows hdf5
    const auto est = strikecem::estimate_resources(c.rc.value, c.plan, c.mesh, {});
    EXPECT_EQ(est.mesh_bytes, 4u * 24 + 2u * (12 + 24 + 8));
    EXPECT_EQ(est.output_bytes, 181u * (18 + 4 * 8 + 1 + 4));
    EXPECT_GT(est.thread_bytes, 0u);
    EXPECT_EQ(est.total_bytes, est.mesh_bytes + est.output_bytes + est.thread_bytes);
    EXPECT_TRUE(est.fits);
    EXPECT_EQ(est.operation_count, 2u * 181u * 200u);
    EXPECT_FALSE(est.calibrated); // no profiles supplied
}

TEST(Estimate, TinyLimitRefuses) {
    const Case c = load_case("valid_minimal.json");
    nlohmann::json tight = c.rc.value;
    tight["execution"]["max_memory_mb"] = 0.000001;
    const auto est = strikecem::estimate_resources(tight, c.plan, c.mesh, {});
    EXPECT_FALSE(est.fits);
    EXPECT_FALSE(est.warnings.empty());
}

TEST(Estimate, CoarseMeshWarnsAtHighFrequency) {
    const Case c = load_case("valid_minimal.json"); // 1.4 m edges at 10 GHz
    const auto est = strikecem::estimate_resources(c.rc.value, c.plan, c.mesh, {});
    EXPECT_FALSE(est.warnings.empty());
    EXPECT_GT(est.electrical_size_wavelengths, 1.0);
}

TEST(Estimate, CalibrationMatchesInRange) {
    const Case c = load_case("valid_minimal.json");
    strikecem::BenchmarkProfile profile;
    profile.id = "synthetic";
    profile.precision = "float64";
    profile.backend = "cpu";
    profile.triangles = 2;
    profile.samples = 200; // within 4x of 181 rows
    profile.wall_seconds = 0.5;
    const auto est = strikecem::estimate_resources(c.rc.value, c.plan, c.mesh, {profile});
    ASSERT_TRUE(est.calibrated);
    EXPECT_EQ(est.profile_id, "synthetic");
    // Throughput = 2*200*200/0.5 ops/s; predicted = ops/throughput.
    const double expected = (2.0 * 181 * 200) / ((2.0 * 200 * 200) / 0.5);
    EXPECT_NEAR((est.runtime_lo_s + est.runtime_hi_s) / 2.0, expected * 1.25, 1e-9);
    EXPECT_LT(est.runtime_lo_s, est.runtime_hi_s);
}

TEST(Estimate, CalibrationRejectsMismatch) {
    const Case c = load_case("valid_minimal.json");
    strikecem::BenchmarkProfile wrong_precision = {"p", "float32", "cpu", 2, 181, 0.5, ""};
    strikecem::BenchmarkProfile far_size = {"q", "float64", "cpu", 2, 181 * 100, 0.5, ""};
    EXPECT_FALSE(
        strikecem::estimate_resources(c.rc.value, c.plan, c.mesh, {wrong_precision}).calibrated);
    EXPECT_FALSE(strikecem::estimate_resources(c.rc.value, c.plan, c.mesh, {far_size}).calibrated);
}

TEST(Estimate, BenchmarkProfileRoundTrip) {
    const Case c = load_case("valid_minimal.json");
    const fs::path path = fs::temp_directory_path() / "scem_bench_test.json";
    strikecem::write_benchmark_profile(path.string(), c.rc, c.mesh.report.triangle_count,
                                       c.plan.sample_count(), 1.25, "cpu", "");
    std::ifstream in(path);
    const auto j = nlohmann::json::parse(in);
    EXPECT_EQ(j["profile_format"], "scem-bench-1");
    EXPECT_EQ(j["precision"], "float64");
    EXPECT_EQ(j["triangles"], 2u);
    EXPECT_EQ(j["samples"], c.plan.sample_count());
    EXPECT_DOUBLE_EQ(j["wall_seconds"], 1.25);
    std::error_code ec;
    fs::remove(path, ec);
}

TEST(Threads, CountIsNumericallyInvisible) {
    const Case c = load_case("valid_sweep_grid.json"); // 72 rows
    const auto plan = c.plan;
    nlohmann::json one = c.rc.value;
    one["execution"]["cpu_threads"] = 1;
    nlohmann::json many = c.rc.value;
    many["execution"]["cpu_threads"] = 8;
    const auto a = strikecem::solve_po(c.mesh, plan, one);
    const auto b = strikecem::solve_po(c.mesh, plan, many);
    ASSERT_EQ(a.samples.size(), b.samples.size());
    for (size_t i = 0; i < a.samples.size(); ++i) {
        EXPECT_EQ(a.samples[i].sample_id, b.samples[i].sample_id);
        EXPECT_EQ(a.samples[i].scattering, b.samples[i].scattering);
        EXPECT_EQ(a.samples[i].lit_facets, b.samples[i].lit_facets);
    }
}

TEST(Cli, RunRefusesOverLimit) {
    const fs::path dir = fs::temp_directory_path() / "scem_limit_test";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    nlohmann::json cfg;
    cfg["scem_schema_version"] = "1.0";
    cfg["model"] = {{"path", std::string(std::string(SCEM_FIXTURE_DIR) +
                                         "/../../examples/plate.stl")}};
    cfg["frequency"] = {{"frequency_hz", 10e9}};
    cfg["angles"]["azimuth"] = {{"start", 0}, {"stop", 0}, {"step", 1}};
    cfg["angles"]["elevation"] = {{"start", -90}, {"stop", -90}, {"step", 1}};
    cfg["solver"] = {{"type", "PO"}};
    cfg["execution"] = {{"max_memory_mb", 0.000001}};
    cfg["output"] = {{"path", (dir / "out.csv").string()}, {"format", "csv"}};
    const fs::path config_path = dir / "scem.json";
    {
        std::ofstream out(config_path);
        out << cfg.dump(2);
    }
    const std::string argv0 = "strikecem";
    const std::string config = config_path.string();
    char* argv[] = {const_cast<char*>(argv0.c_str()), const_cast<char*>("run"),
                    const_cast<char*>(config.c_str())};
    // Uses the install-tree schema default; run from the source root.
    EXPECT_EQ(strikecem::cli::run(3, argv), static_cast<int>(strikecem::ExitCode::Resource));
    fs::remove_all(dir, ec);
}

TEST(Estimate, GoCorrectionsWarnUncalibrated) {
    const Case c = load_case("valid_minimal.json");
    bool plain_warns = false;
    for (const auto& w : strikecem::estimate_resources(c.rc.value, c.plan, c.mesh, {}).warnings)
        if (w.find("GO corrections") != std::string::npos) plain_warns = true;
    EXPECT_FALSE(plain_warns);
    nlohmann::json go = c.rc.value;
    go["solver"]["po_options"]["shadowing"] = true;
    go["solver"]["po_options"]["max_bounces"] = 3;
    bool found = false;
    for (const auto& w : strikecem::estimate_resources(go, c.plan, c.mesh, {}).warnings)
        if (w.find("GO corrections") != std::string::npos) found = true;
    EXPECT_TRUE(found);
}

} // namespace

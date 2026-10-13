#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include "strikecem/app/CLI.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/telemetry/Telemetry.hpp"

namespace fs = std::filesystem;

namespace {

TEST(Telemetry, StageSpanTimingAndDurations) {
    strikecem::telemetry::TelemetryCollector tc;

    tc.start_stage("stage_alpha");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    tc.end_stage("stage_alpha");

    tc.start_stage("stage_beta");
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    tc.end_stage("stage_beta");

    const auto& stages = tc.stages();
    ASSERT_EQ(stages.size(), 2u);

    EXPECT_EQ(stages[0].name, "stage_alpha");
    EXPECT_GE(stages[0].duration_seconds, 0.008);
    EXPECT_LE(stages[0].start_timestamp_sec, stages[0].end_timestamp_sec);

    EXPECT_EQ(stages[1].name, "stage_beta");
    EXPECT_GE(stages[1].duration_seconds, 0.012);
    EXPECT_GE(stages[1].start_timestamp_sec, stages[0].end_timestamp_sec);
}

TEST(Telemetry, MemoryMetricsQuery) {
    strikecem::telemetry::TelemetryCollector tc;
    const auto mem = tc.query_memory_metrics();

#if defined(__linux__)
    EXPECT_GT(mem.vm_hwm_bytes, 0u);
    EXPECT_GT(mem.vm_rss_bytes, 0u);
    EXPECT_GE(mem.vm_hwm_bytes, mem.vm_rss_bytes);
#endif
}

TEST(Telemetry, ExportJsonStructure) {
    strikecem::telemetry::TelemetryCollector tc;
    tc.record_metadata("solver", "PO");
    tc.record_counter("total_facets", 1500);
    tc.record_metric("throughput_mps", 42.5);
    tc.record_warning("test warning message");

    {
        strikecem::telemetry::ScopedStageTimer timer(tc, "scoped_computation");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    const auto j = tc.export_json();
    EXPECT_EQ(j["telemetry_schema_version"], "1.0");
    EXPECT_EQ(j["metadata"]["solver"], "PO");
    EXPECT_EQ(j["counters"]["total_facets"], 1500u);
    EXPECT_DOUBLE_EQ(j["metrics"]["throughput_mps"], 42.5);
    EXPECT_EQ(j["warnings"].size(), 1u);
    EXPECT_EQ(j["warnings"][0], "test warning message");
    EXPECT_EQ(j["stages"].size(), 1u);
    EXPECT_EQ(j["stages"][0]["name"], "scoped_computation");
    EXPECT_GT(j["memory"]["vm_hwm_bytes"].get<uint64_t>(), 0u);
}

TEST(Telemetry, AtomicFileWriteAndReload) {
    const fs::path dir = fs::temp_directory_path() / ("scem_telem_test_" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    const fs::path telem_path = dir / "report.json";

    strikecem::telemetry::TelemetryCollector tc;
    tc.record_metadata("run_id", "run-42");
    tc.record_counter("samples", 100);
    tc.write_json_report(telem_path.string());

    ASSERT_TRUE(fs::exists(telem_path));
    std::ifstream in(telem_path);
    ASSERT_TRUE(in.is_open());
    const auto j = nlohmann::json::parse(in);
    EXPECT_EQ(j["metadata"]["run_id"], "run-42");
    EXPECT_EQ(j["counters"]["samples"], 100u);

    fs::remove_all(dir, ec);
}

TEST(Telemetry, EndToEndSimulationRunTelemetry) {
    const fs::path dir = fs::temp_directory_path() / ("scem_e2e_telem_" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    const fs::path stl_path = std::string(SCEM_FIXTURE_DIR) + "/../../examples/plate.stl";
    const fs::path cfg_path = dir / "cfg.json";
    const fs::path out_path = dir / "out.csv";
    const fs::path telem_path = dir / "telemetry.json";

    nlohmann::json cfg = {
        {"scem_schema_version", "1.0"},
        {"model", {{"path", stl_path.string()}}},
        {"frequency", {{"frequency_hz", 10e9}}},
        {"angles", {
            {"azimuth", {{"start", 0}, {"stop", 30}, {"step", 30}}},
            {"elevation", {{"start", -90}, {"stop", -90}, {"step", 1}}}
        }},
        {"polarization", {"HH", "VV"}},
        {"solver", {{"type", "PO"}}},
        {"output", {
            {"path", out_path.string()},
            {"format", "csv"},
            {"telemetry", {
                {"enabled", true},
                {"path", telem_path.string()},
                {"record_memory", true}
            }}
        }}
    };

    {
        std::ofstream out(cfg_path);
        out << cfg.dump(2);
    }

    const std::string argv0 = "strikecem";
    char* argv[] = {const_cast<char*>(argv0.c_str()), const_cast<char*>("run"),
                    const_cast<char*>(cfg_path.c_str())};
    EXPECT_EQ(strikecem::cli::run(3, argv), 0);

    ASSERT_TRUE(fs::exists(telem_path));
    std::ifstream in(telem_path);
    ASSERT_TRUE(in.is_open());
    const auto j = nlohmann::json::parse(in);
    EXPECT_EQ(j["telemetry_schema_version"], "1.0");
    EXPECT_EQ(j["metadata"]["solver"], "PO");
    EXPECT_EQ(j["metadata"]["format"], "csv");
    EXPECT_EQ(j["counters"]["samples_computed"], 4u); // 2 dirs x 2 pols
    EXPECT_GT(j["counters"]["facets_count"], 0u);
    EXPECT_GT(j["metrics"]["solve_wall_seconds"], 0.0);
    EXPECT_GT(j["metrics"]["samples_per_second"], 0.0);
    EXPECT_GT(j["memory"]["vm_hwm_bytes"].get<uint64_t>(), 0u);

    fs::remove_all(dir, ec);
}

} // namespace

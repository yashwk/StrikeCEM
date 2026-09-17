// CSV writer + sidecar tests: exact columns, value fidelity, zero-row
// policy, sidecar provenance, and I/O failure behavior.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/CsvWriter.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace {

namespace fs = std::filesystem;

struct Workdir {
    fs::path dir = fs::temp_directory_path() / "scem_output_test";
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

nlohmann::json base_config(const fs::path& dir, double elevation_deg) {
    nlohmann::json cfg;
    cfg["scem_schema_version"] = "1.0";
    cfg["model"] = {{"path", std::string(std::string(SCEM_FIXTURE_DIR) +
                                         "/../../examples/plate.stl")}};
    cfg["frequency"] = {{"frequency_hz", 10e9}};
    cfg["angles"]["azimuth"] = {{"start", 0}, {"stop", 0}, {"step", 1}};
    cfg["angles"]["elevation"] = {
        {"start", elevation_deg}, {"stop", elevation_deg}, {"step", 1}};
    cfg["polarization"] = {"HH"};
    cfg["solver"] = {{"type", "PO"}};
    cfg["output"] = {{"path", (dir / "out.csv").string()}, {"format", "csv"}};
    return cfg;
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream in(line);
    while (std::getline(in, field, ',')) fields.push_back(field);
    return fields;
}

std::vector<std::string> read_lines(const fs::path& path) {
    std::ifstream in(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
    return lines;
}

TEST(Output, CsvAndSidecarRoundTrip) {
    Workdir work;
    const fs::path config_path = work.dir / "scem.json";
    {
        std::ofstream out(config_path);
        out << base_config(work.dir, -90.0).dump(2);
    }
    const auto rc = strikecem::load_config(config_path.string(), SCEM_SCHEMA_PATH);
    const auto plan = strikecem::build_sample_plan(rc.value);
    const auto mesh =
        strikecem::load_normalized_mesh(rc.value, work.dir, rc.schema_version);
    const auto result = strikecem::solve_po(mesh, plan, rc.value);
    ASSERT_EQ(result.samples.size(), 1u);
    strikecem::write_csv_and_sidecar(rc, plan, mesh, result);

    const auto lines = read_lines(work.dir / "out.csv");
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_EQ(lines[0], "sample_id,azimuth_deg,elevation_deg,frequency_hz,polarization,"
                        "scattering_real,scattering_imag,rcs_sqm,rcs_dBsm,valid,status");
    const auto row = split_csv(lines[1]);
    ASSERT_EQ(row.size(), 11u);
    EXPECT_EQ(row[0], "0");
    EXPECT_EQ(row[4], "HH");
    EXPECT_EQ(row[8], row[8]); // present
    EXPECT_EQ(row[9], "true");
    const auto expected = result.samples[0];
    EXPECT_NEAR(std::stod(row[5]), expected.scattering.real(), 5e-7);
    EXPECT_NEAR(std::stod(row[6]), expected.scattering.imag(), 5e-7);
    EXPECT_NEAR(std::stod(row[7]), expected.rcs_sqm, 1e-4 * expected.rcs_sqm);
    EXPECT_NEAR(std::stod(row[8]),
                strikecem::rcs_sqm_to_dbsm(expected.rcs_sqm), 1e-6);
    EXPECT_EQ(row[10], "ok");
    // Six decimals by default.
    EXPECT_NE(row[5].find('.'), std::string::npos);
    EXPECT_EQ(row[5].size() - row[5].find('.') - 1, 6u);

    std::ifstream sidecar_in(work.dir / "out.csv.json");
    ASSERT_TRUE(sidecar_in.good());
    const auto sidecar = nlohmann::json::parse(sidecar_in);
    EXPECT_EQ(sidecar["scem_schema_version"], "1.0");
    EXPECT_EQ(sidecar["output_format_version"], "1.0");
    EXPECT_EQ(sidecar["config_hash"], rc.hash);
    EXPECT_EQ(sidecar["geometry_hash"], mesh.geometry_hash);
    EXPECT_EQ(sidecar["normalized_mesh_hash"], mesh.normalized_mesh_hash);
    EXPECT_EQ(sidecar["sample_plan"]["samples"], 1u);
    EXPECT_EQ(sidecar["tool"]["name"], "strikecem");
    EXPECT_EQ(sidecar["resolved_config"]["solver"]["type"], "PO");
}

TEST(Output, DarkRowFollowsZeroPolicy) {
    Workdir work;
    const fs::path config_path = work.dir / "scem.json";
    {
        std::ofstream out(config_path);
        out << base_config(work.dir, 90.0).dump(2);
    }
    const auto rc = strikecem::load_config(config_path.string(), SCEM_SCHEMA_PATH);
    const auto plan = strikecem::build_sample_plan(rc.value);
    const auto mesh =
        strikecem::load_normalized_mesh(rc.value, work.dir, rc.schema_version);
    const auto result = strikecem::solve_po(mesh, plan, rc.value);
    strikecem::write_csv_and_sidecar(rc, plan, mesh, result);
    const auto lines = read_lines(work.dir / "out.csv");
    ASSERT_EQ(lines.size(), 2u);
    const auto row = split_csv(lines[1]);
    ASSERT_EQ(row.size(), 11u);
    EXPECT_EQ(std::stod(row[7]), 0.0);
    EXPECT_EQ(row[8], "-inf");
    EXPECT_EQ(row[9], "true");
    EXPECT_EQ(row[10], "ok");
}

TEST(Output, UnwritablePathFails) {
    Workdir work;
    auto cfg = base_config(work.dir, -90.0);
    cfg["output"]["path"] = work.dir.string(); // a directory, not a file
    const fs::path config_path = work.dir / "scem.json";
    {
        std::ofstream out(config_path);
        out << cfg.dump(2);
    }
    const auto rc = strikecem::load_config(config_path.string(), SCEM_SCHEMA_PATH);
    const auto plan = strikecem::build_sample_plan(rc.value);
    const auto mesh =
        strikecem::load_normalized_mesh(rc.value, work.dir, rc.schema_version);
    const auto result = strikecem::solve_po(mesh, plan, rc.value);
    EXPECT_THROW(strikecem::write_csv_and_sidecar(rc, plan, mesh, result),
                 strikecem::OutputError);
}

} // namespace

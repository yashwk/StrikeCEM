// StrikeDesigner package tests (SPEC FR-12): manifest validation, hash
// pinning, identity agreement, component warnings, and provenance flow.
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "strikecem/app/CLI.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/io/DesignerPackage.hpp"
#include "strikecem/io/MeshLoader.hpp"

namespace {

namespace fs = std::filesystem;

std::string package(const std::string& name) {
    return std::string(SCEM_FIXTURE_DIR) + "/packages/" + name;
}

strikecem::DesignerIdentity validate_package(const std::string& name) {
    const std::string config = package(name) + "/config.json";
    auto rc = strikecem::load_config(config, SCEM_SCHEMA_PATH);
    return strikecem::validate_designer_package(rc.value, config);
}

int cli_validate(const std::string& config) {
    const std::string argv0 = "strikecem";
    char* argv[] = {const_cast<char*>(argv0.c_str()), const_cast<char*>("validate"),
                    const_cast<char*>(config.c_str())};
    return strikecem::cli::run(3, argv);
}

TEST(Designer, ObjGroupsParsed) {
    const auto plain = strikecem::parse_obj(
        std::string(SCEM_FIXTURE_DIR) + "/../../examples/triangle.obj");
    EXPECT_TRUE(plain.groups.empty());
    const auto grouped =
        strikecem::parse_obj(package("valid") + "/geometry/target.obj");
    ASSERT_EQ(grouped.groups.size(), 1u);
    EXPECT_EQ(grouped.groups[0], "body");
    // Groups survive the normalize + cache round trip.
    auto rc = strikecem::load_config(package("valid") + "/config.json", SCEM_SCHEMA_PATH);
    const auto mesh = strikecem::load_normalized_mesh(
        rc.value, package("valid"), rc.schema_version);
    EXPECT_EQ(mesh.groups, grouped.groups);
    const auto cached = strikecem::load_normalized_mesh(
        rc.value, package("valid"), rc.schema_version);
    EXPECT_TRUE(cached.cache_hit);
    EXPECT_EQ(cached.groups, grouped.groups);
}

TEST(Designer, ValidPackageResolvesIdentity) {
    const auto id = validate_package("valid");
    EXPECT_TRUE(id.packaged);
    EXPECT_EQ(id.design_id, "test-plate");
    EXPECT_EQ(id.revision, "7");
    EXPECT_EQ(id.manifest_hash.size(), 64u);
    EXPECT_TRUE(id.warnings.empty());
    EXPECT_EQ(cli_validate(package("valid") + "/config.json"), 0);
}

TEST(Designer, DirectModePassesThrough) {
    const std::string config = std::string(SCEM_FIXTURE_DIR) + "/valid_minimal.json";
    auto rc = strikecem::load_config(config, SCEM_SCHEMA_PATH);
    const auto id = strikecem::validate_designer_package(rc.value, config);
    EXPECT_FALSE(id.packaged);
    EXPECT_TRUE(id.design_id.empty());
}

TEST(Designer, DirectModeKeepsConfigBlock) {
    const fs::path dir = fs::temp_directory_path() / "scem_designer_test";
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
    cfg["output"] = {{"path", "x.h5"}};
    cfg["integration"] = {{"designer",
                           {{"design_id", "standalone"},
                            {"revision", "3"},
                            {"export_id", "standalone-r3"}}}};
    const fs::path config = dir / "scem.json";
    {
        std::ofstream out(config);
        out << cfg.dump(2);
    }
    auto rc = strikecem::load_config(config.string(), SCEM_SCHEMA_PATH);
    const auto id = strikecem::validate_designer_package(rc.value, config.string());
    EXPECT_FALSE(id.packaged);
    EXPECT_EQ(id.design_id, "standalone");
    EXPECT_EQ(id.revision, "3");
    EXPECT_EQ(id.export_id, "standalone-r3");
    fs::remove_all(dir, ec);
}

TEST(Designer, BadHashFailsClosed) {
    EXPECT_THROW(validate_package("bad_hash"), strikecem::MeshError);
    EXPECT_EQ(cli_validate(package("bad_hash") + "/config.json"), 3);
}

TEST(Designer, BadVersionRejected) {
    EXPECT_THROW(validate_package("bad_version"), strikecem::ConfigError);
    EXPECT_EQ(cli_validate(package("bad_version") + "/config.json"), 2);
}

TEST(Designer, MissingGeometryRejected) {
    EXPECT_THROW(validate_package("missing_geometry"), strikecem::MeshError);
}

TEST(Designer, IdentityMismatchRejected) {
    EXPECT_THROW(validate_package("identity_mismatch"), strikecem::ConfigError);
    EXPECT_EQ(cli_validate(package("identity_mismatch") + "/config.json"), 2);
}

TEST(Designer, UnknownGroupWarnsButValidates) {
    const auto id = validate_package("unknown_group");
    EXPECT_TRUE(id.packaged);
    ASSERT_EQ(id.warnings.size(), 1u);
    EXPECT_NE(id.warnings[0].find("ghost"), std::string::npos);
    EXPECT_EQ(cli_validate(package("unknown_group") + "/config.json"), 0);
}

TEST(Designer, ProvenanceCarriesIdentity) {
    const fs::path dir = fs::temp_directory_path() / "scem_package_run";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::copy(package("valid"), dir, fs::copy_options::recursive, ec);
    ASSERT_FALSE(ec);
    nlohmann::json cfg;
    {
        std::ifstream in(dir / "config.json");
        cfg = nlohmann::json::parse(in);
    }
    cfg["output"] = {{"path", (dir / "out.csv").string()}, {"format", "csv"}};
    {
        std::ofstream out(dir / "config.json");
        out << cfg.dump(2);
    }
    const std::string argv0 = "strikecem";
    const std::string config = (dir / "config.json").string();
    char* argv[] = {const_cast<char*>(argv0.c_str()), const_cast<char*>("run"),
                    const_cast<char*>(config.c_str())};
    ASSERT_EQ(strikecem::cli::run(3, argv), 0);
    std::ifstream sidecar_in(dir / "out.csv.json");
    ASSERT_TRUE(sidecar_in.good());
    const auto sidecar = nlohmann::json::parse(sidecar_in);
    EXPECT_EQ(sidecar["designer"]["design_id"], "test-plate");
    EXPECT_EQ(sidecar["designer"]["revision"], "7");
    EXPECT_EQ(sidecar["designer"]["export_manifest_hash"].get<std::string>().size(), 64u);
    fs::remove_all(dir, ec);
}

} // namespace

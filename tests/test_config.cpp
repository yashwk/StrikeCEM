// Config loader tests: schema accept/reject, default resolution,
// canonical hashing, sweep expansion, and direction-plan rules.
#include <fstream>
#include <gtest/gtest.h>

#include "strikecem/core/Config.hpp"

namespace {

std::string fixture(const std::string& name) { return std::string(SCEM_FIXTURE_DIR) + "/" + name; }

strikecem::ResolvedConfig load_fixture(const std::string& name) {
    return strikecem::load_config(fixture(name), SCEM_SCHEMA_PATH);
}

TEST(Config, ValidMinimalResolvesDefaults) {
    const auto rc = load_fixture("valid_minimal.json");
    EXPECT_EQ(rc.schema_version, "1.0");
    EXPECT_EQ(rc.hash.size(), 64u);
    const auto& v = rc.value;
    EXPECT_EQ(v["polarization"], (nlohmann::json{"HH"}));
    // Absent polarization defaults to HH + VV.
    EXPECT_EQ(load_fixture("valid_list_dup.json").value["polarization"],
              (nlohmann::json{"HH", "VV"}));
    EXPECT_EQ(v["solver"]["precision"], "float64");
    EXPECT_EQ(v["solver"]["type"], "PO");
    EXPECT_EQ(v["solver"]["po_options"]["illumination_model"], "hard");
    EXPECT_EQ(v["execution"]["cpu_threads"], 1);
    EXPECT_TRUE(v["execution"]["deterministic"].get<bool>());
    EXPECT_EQ(v["execution"]["accelerator"], "cpu");
    EXPECT_EQ(v["execution"]["cuda_device_id"], 0);
    EXPECT_EQ(v["execution"]["cuda_batch_units"], 0);
    EXPECT_EQ(v["model"]["units"], "m");
}

TEST(Config, HashIsDeterministicAndOrderIndependent) {
    const auto a = load_fixture("valid_minimal.json");
    const auto b = load_fixture("valid_minimal.json");
    EXPECT_EQ(a.hash, b.hash);
    // Same content, shuffled key order: identical canonical hash.
    const auto c = load_fixture("valid_reordered.json");
    EXPECT_EQ(a.hash, c.hash);
    // Different content: different hash.
    const auto d = load_fixture("valid_sweep_grid.json");
    EXPECT_NE(a.hash, d.hash);
}

TEST(Config, SweepExpandsByIntegerIndices) {
    const auto rc = load_fixture("valid_sweep_grid.json");
    const auto plan = strikecem::build_sample_plan(rc.value);
    ASSERT_EQ(plan.frequencies_hz.size(), 3u);
    EXPECT_DOUBLE_EQ(plan.frequencies_hz[0], 10e9);
    EXPECT_DOUBLE_EQ(plan.frequencies_hz[1], 11e9);
    EXPECT_DOUBLE_EQ(plan.frequencies_hz[2], 12e9);
    EXPECT_EQ(plan.polarizations.size(), 6u);
}

TEST(Config, GridDeduplicatesPeriodicEndpoint) {
    // az 0..360 step 90 emits {0, 90, 180, 270}; el single row.
    const auto rc = load_fixture("valid_sweep_grid.json");
    const auto plan = strikecem::build_sample_plan(rc.value);
    ASSERT_EQ(plan.directions.size(), 4u);
    EXPECT_DOUBLE_EQ(plan.directions[0].azimuth_deg, 0.0);
    EXPECT_DOUBLE_EQ(plan.directions[1].azimuth_deg, 90.0);
    EXPECT_DOUBLE_EQ(plan.directions[2].azimuth_deg, 180.0);
    EXPECT_DOUBLE_EQ(plan.directions[3].azimuth_deg, 270.0);
    for (uint32_t i = 0; i < plan.directions.size(); ++i) EXPECT_EQ(plan.directions[i].id, i);
    EXPECT_EQ(plan.sample_count(), 3u * 4u * 6u);
}

TEST(Config, ExplicitListNormalizesAndDedupes) {
    const auto rc = load_fixture("valid_list_dup.json");
    const auto plan = strikecem::build_sample_plan(rc.value);
    ASSERT_EQ(plan.directions.size(), 2u);
    EXPECT_DOUBLE_EQ(plan.directions[0].azimuth_deg, 0.0);
    EXPECT_DOUBLE_EQ(plan.directions[1].azimuth_deg, 45.0);
    EXPECT_FALSE(plan.warnings.empty());
}

TEST(Config, RejectsUnknownNestedKey) {
    EXPECT_THROW(load_fixture("invalid_unknown_key.json"), strikecem::ConfigError);
}

TEST(Config, RejectsUnsupportedSolver) {
    EXPECT_THROW(load_fixture("invalid_solver.json"), strikecem::ConfigError);
}

TEST(Config, RejectsBadSweep) {
    EXPECT_THROW(load_fixture("invalid_sweep_order.json"), strikecem::ConfigError);
    EXPECT_THROW(load_fixture("invalid_sweep_steps.json"), strikecem::ConfigError);
}

TEST(Config, RejectsBadAnglesAndPolarization) {
    EXPECT_THROW(load_fixture("invalid_angles.json"), strikecem::ConfigError);
    EXPECT_THROW(load_fixture("invalid_polarization.json"), strikecem::ConfigError);
}

TEST(Config, RejectsMissingMeshFile) {
    EXPECT_THROW(load_fixture("invalid_missing_mesh.json"), strikecem::MeshError);
}

TEST(Config, RejectsUnsupportedMeshExtension) {
    EXPECT_THROW(load_fixture("invalid_mesh_ext.json"), strikecem::MeshError);
}

TEST(Config, RejectsMalformedJson) {
    EXPECT_THROW(load_fixture("invalid_malformed.json"), strikecem::ConfigError);
}

} // namespace

#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <numbers>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/core/Conventions.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace {

namespace fs = std::filesystem;

struct Workdir {
    fs::path dir = fs::temp_directory_path() / ("scem_bistatic_test_" + std::to_string(::getpid()));
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

nlohmann::json base_config(const fs::path& work_dir) {
    nlohmann::json cfg;
    cfg["scem_schema_version"] = "1.0";
    cfg["model"] = {{"path", std::string(SCEM_FIXTURE_DIR) + "/../../examples/plate.stl"}};
    cfg["frequency"] = {{"frequency_hz", 1.0e9}};
    cfg["polarization"] = {"HH", "VV"};
    cfg["solver"] = {{"type", "PO"}};
    cfg["output"] = {{"path", (work_dir / "out.h5").string()}, {"format", "hdf5"}};
    return cfg;
}

strikecem::ResolvedConfig load_from_json(const nlohmann::json& j, const fs::path& file_path) {
    {
        std::ofstream out(file_path);
        out << j.dump(2);
    }
    return strikecem::load_config(file_path.string(), SCEM_SCHEMA_PATH);
}

} // namespace

TEST(Bistatic, MonostaticEquivalence) {
    Workdir w;
    // Monostatic config
    auto cfg_mono = base_config(w.dir);
    cfg_mono["angles"] = {
        {"azimuth", {{"start", 0}, {"stop", 60}, {"step", 30}}},
        {"elevation", {{"start", 0}, {"stop", 0}, {"step", 1}}}
    };
    cfg_mono["physics"] = {{"monostatic", true}};

    // Bistatic config with zero offset
    auto cfg_bi = base_config(w.dir);
    cfg_bi["angles"] = {
        {"azimuth", {{"start", 0}, {"stop", 60}, {"step", 30}}},
        {"elevation", {{"start", 0}, {"stop", 0}, {"step", 1}}},
        {"bistatic", {{"mode", "fixed_offset"}, {"azimuth_offset_deg", 0.0}, {"elevation_offset_deg", 0.0}}}
    };
    cfg_bi["physics"] = {{"monostatic", false}};

    auto rc_mono = load_from_json(cfg_mono, w.dir / "mono.json");
    auto rc_bi = load_from_json(cfg_bi, w.dir / "bi.json");

    auto plan_mono = strikecem::build_sample_plan(rc_mono.value);
    auto plan_bi = strikecem::build_sample_plan(rc_bi.value);

    ASSERT_EQ(plan_mono.directions.size(), plan_bi.directions.size());
    for (size_t i = 0; i < plan_mono.directions.size(); ++i) {
        EXPECT_FALSE(plan_mono.directions[i].is_bistatic);
        EXPECT_TRUE(plan_bi.directions[i].is_bistatic);
        // Both transmit and receive vectors align backscatter
        EXPECT_NEAR(plan_bi.directions[i].bistatic_angle_deg, 0.0, 1e-5);
    }

    auto mesh_mono = strikecem::load_normalized_mesh(rc_mono.value, w.dir, rc_mono.schema_version);
    auto mesh_bi = strikecem::load_normalized_mesh(rc_bi.value, w.dir, rc_bi.schema_version);

    auto res_mono = strikecem::solve_po(mesh_mono, plan_mono, rc_mono.value);
    auto res_bi = strikecem::solve_po(mesh_bi, plan_bi, rc_bi.value);

    ASSERT_EQ(res_mono.samples.size(), res_bi.samples.size());
    for (size_t i = 0; i < res_mono.samples.size(); ++i) {
        EXPECT_NEAR(res_mono.samples[i].rcs_sqm, res_bi.samples[i].rcs_sqm, 1e-9);
        EXPECT_NEAR(res_mono.samples[i].scattering.real(), res_bi.samples[i].scattering.real(), 1e-9);
        EXPECT_NEAR(res_mono.samples[i].scattering.imag(), res_bi.samples[i].scattering.imag(), 1e-9);
    }
}

TEST(Bistatic, FlatPlateForwardScatterPeak) {
    Workdir w;
    // 1m x 1m square plate in xy-plane. Area A = 1 m^2.
    // Incident wave from +z to -z (el = -90 deg).
    // In forward scatter, observation direction is also -z (el = -90 deg, beta = 180 deg).
    // Theoretical forward scatter peak: sigma = 4 * pi * A^2 / lambda^2
    const double freq = 1.0e9; // 1 GHz
    const double lambda = strikecem::kSpeedOfLight / freq;
    const double area = 1.0; // 1.0 m^2
    const double sigma_theoretical = 4.0 * std::numbers::pi * (area * area) / (lambda * lambda);
    const double dbsm_theoretical = 10.0 * std::log10(sigma_theoretical);

    auto cfg = base_config(w.dir);
    cfg["frequency"]["frequency_hz"] = freq;
    cfg["angles"]["list"] = {
        // [inc_az, inc_el, obs_az, obs_el]: normal incidence and forward observation
        {0.0, -90.0, 0.0, -90.0}
    };
    cfg["physics"] = {{"monostatic", false}};

    auto rc = load_from_json(cfg, w.dir / "forward.json");
    auto plan = strikecem::build_sample_plan(rc.value);

    ASSERT_EQ(plan.directions.size(), 1u);
    EXPECT_TRUE(plan.directions[0].is_bistatic);
    EXPECT_NEAR(plan.directions[0].bistatic_angle_deg, 180.0, 1e-4);

    auto mesh = strikecem::load_normalized_mesh(rc.value, w.dir, rc.schema_version);
    auto res = strikecem::solve_po(mesh, plan, rc.value);

    ASSERT_FALSE(res.samples.empty());
    const double rcs_sqm = res.samples[0].rcs_sqm;
    const double rcs_dbsm = strikecem::rcs_sqm_to_dbsm(rcs_sqm);

    // Forward scatter peak matches Babinet/optical theorem within 0.15 dB
    EXPECT_NEAR(rcs_sqm, sigma_theoretical, 0.03 * sigma_theoretical);
    EXPECT_NEAR(rcs_dbsm, dbsm_theoretical, 0.15);
}

TEST(Bistatic, BistaticAngleRollOff) {
    Workdir w;
    // Normal incidence, sweep observation angle in elevation from forward (-90 deg) towards broadside
    auto cfg = base_config(w.dir);
    cfg["frequency"]["frequency_hz"] = 3.0e9; // 3 GHz
    cfg["angles"]["list"] = {
        {0.0, -90.0, 0.0, -90.0}, // forward scatter peak (beta = 180 deg)
        {0.0, -90.0, 0.0, -89.0}, // 1 deg off peak
        {0.0, -90.0, 0.0, -88.0}, // 2 deg off peak
        {0.0, -90.0, 0.0, -87.0}  // 3 deg off peak
    };
    cfg["physics"] = {{"monostatic", false}};

    auto rc = load_from_json(cfg, w.dir / "rolloff.json");
    auto plan = strikecem::build_sample_plan(rc.value);
    auto mesh = strikecem::load_normalized_mesh(rc.value, w.dir, rc.schema_version);
    auto res = strikecem::solve_po(mesh, plan, rc.value);

    // RCS must strictly decrease as observation moves down the main forward lobe
    const double rcs_peak = res.samples[0].rcs_sqm;
    const double rcs_1deg = res.samples[2].rcs_sqm;
    const double rcs_2deg = res.samples[4].rcs_sqm;
    const double rcs_3deg = res.samples[6].rcs_sqm;

    EXPECT_GT(rcs_peak, rcs_1deg);
    EXPECT_GT(rcs_1deg, rcs_2deg);
    EXPECT_GT(rcs_2deg, rcs_3deg);
}

TEST(Bistatic, ConfigBistaticAngleMode) {
    Workdir w;
    auto cfg = base_config(w.dir);
    cfg["angles"] = {
        {"azimuth", {{"start", 0}, {"stop", 40}, {"step", 20}}},
        {"elevation", {{"start", 0}, {"stop", 0}, {"step", 1}}},
        {"bistatic", {{"mode", "bistatic_angle"}, {"bistatic_angle_deg", 45.0}}}
    };
    cfg["physics"] = {{"monostatic", false}};

    auto rc = load_from_json(cfg, w.dir / "angle_mode.json");
    auto plan = strikecem::build_sample_plan(rc.value);
    ASSERT_EQ(plan.directions.size(), 3u);
    for (const auto& d : plan.directions) {
        EXPECT_TRUE(d.is_bistatic);
        EXPECT_NEAR(d.bistatic_angle_deg, 45.0, 1e-4);
    }
}

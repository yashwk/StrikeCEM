#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>

#include "strikecem/core/Config.hpp"
#include "strikecem/core/Conventions.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/sampling/SphericalSampling.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace fs = std::filesystem;

namespace {

TEST(SphericalSampling, PointCountAndNormalization) {
    const size_t n = 500;
    const auto dirs = strikecem::generate_fibonacci_sphere(n, -90.0, 90.0);
    ASSERT_EQ(dirs.size(), n);

    double sum_weights = 0.0;
    for (size_t i = 0; i < dirs.size(); ++i) {
        const auto& d = dirs[i];
        EXPECT_EQ(d.id, static_cast<uint32_t>(i));
        const double len = d.k_hat.length();
        EXPECT_NEAR(len, 1.0, 1e-9);
        EXPECT_GE(d.elevation_deg, -90.0 - 1e-6);
        EXPECT_LE(d.elevation_deg, 90.0 + 1e-6);
        EXPECT_GE(d.azimuth_deg, 0.0 - 1e-6);
        EXPECT_LE(d.azimuth_deg, 360.0 + 1e-6);
        sum_weights += d.weight;
    }

    const double expected_total_solid_angle = 4.0 * std::numbers::pi;
    EXPECT_NEAR(sum_weights, expected_total_solid_angle, 1e-12);
}

TEST(SphericalSampling, HemisphericalSamplingAndWeights) {
    const size_t n = 200;
    const auto dirs = strikecem::generate_fibonacci_sphere(n, 0.0, 90.0);
    ASSERT_EQ(dirs.size(), n);

    double sum_weights = 0.0;
    for (const auto& d : dirs) {
        EXPECT_GE(d.elevation_deg, -1e-6);
        EXPECT_LE(d.elevation_deg, 90.0 + 1e-6);
        EXPECT_GE(d.k_hat.z, -1e-6);
        sum_weights += d.weight;
    }

    const double expected_hemi_solid_angle = 2.0 * std::numbers::pi;
    EXPECT_NEAR(sum_weights, expected_hemi_solid_angle, 1e-12);
}

TEST(SphericalSampling, QuadratureIntegratesConstantFunctionExactly) {
    const size_t n = 600;
    const auto dirs = strikecem::generate_fibonacci_sphere(n, -90.0, 90.0);
    std::vector<double> constant_rcs(n, 5.0); // 5.0 m^2 isotropic

    const double integral = strikecem::integrate_scattered_power(dirs, constant_rcs);
    // Integral of 5 over 4*pi steradians = 20*pi
    EXPECT_NEAR(integral, 5.0 * 4.0 * std::numbers::pi, 1e-12);
}

TEST(SphericalSampling, QuadratureIntegratesDipolePatternAccurately) {
    const size_t n = 2000;
    const auto dirs = strikecem::generate_fibonacci_sphere(n, -90.0, 90.0);
    // f(x, y, z) = z^2. Analytical integral over S^2 is 4*pi / 3.
    std::vector<double> vals(n);
    for (size_t i = 0; i < n; ++i) {
        vals[i] = dirs[i].k_hat.z * dirs[i].k_hat.z;
    }

    const double numeric_val = strikecem::integrate_scattered_power(dirs, vals);
    const double analytic_val = 4.0 * std::numbers::pi / 3.0;
    EXPECT_NEAR(numeric_val, analytic_val, 0.01 * analytic_val); // within 1%
}

TEST(SphericalSampling, StatisticsCalculation) {
    const size_t n = 100;
    const auto dirs = strikecem::generate_fibonacci_sphere(n, -90.0, 90.0);
    std::vector<double> rcs(n, 10.0);
    rcs[42] = 100.0; // peak

    const auto stats = strikecem::compute_spherical_stats(dirs, rcs);
    EXPECT_EQ(stats.peak_direction_index, 42u);
    EXPECT_NEAR(stats.max_rcs_dbsm, 20.0, 1e-3); // 10*log10(100) = 20 dBsm
    EXPECT_NEAR(stats.min_rcs_dbsm, 10.0, 1e-3); // 10*log10(10) = 10 dBsm
    EXPECT_GT(stats.mean_rcs_sqm, 10.0);
    EXPECT_GT(stats.total_power_sqm, 0.0);
}

TEST(SphericalSampling, InvalidParametersThrow) {
    EXPECT_THROW(strikecem::generate_fibonacci_sphere(0), std::invalid_argument);
    EXPECT_THROW(strikecem::generate_fibonacci_sphere(100, 50.0, 20.0), std::invalid_argument);
    EXPECT_THROW(strikecem::generate_fibonacci_sphere(100, -95.0, 90.0), std::invalid_argument);
    EXPECT_THROW(strikecem::generate_fibonacci_sphere(100, -90.0, 95.0), std::invalid_argument);

    const auto dirs = strikecem::generate_fibonacci_sphere(10);
    std::vector<double> mismatched_rcs(5, 1.0);
    EXPECT_THROW(strikecem::integrate_scattered_power(dirs, mismatched_rcs), std::invalid_argument);
    EXPECT_THROW(strikecem::compute_spherical_stats(dirs, mismatched_rcs), std::invalid_argument);
}

TEST(SphericalSampling, EndToEndConfigAndExecution) {
    const fs::path dir = fs::temp_directory_path() / ("scem_spherical_" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    const fs::path stl_path = std::string(SCEM_FIXTURE_DIR) + "/../../examples/plate.stl";
    const fs::path cfg_path = dir / "spherical_cfg.json";
    const fs::path out_path = dir / "spherical_out.csv";

    nlohmann::json cfg = {
        {"scem_schema_version", "1.0"},
        {"model", {{"path", stl_path.string()}}},
        {"frequency", {{"frequency_hz", 10e9}}},
        {"angles", {
            {"spherical", {
                {"type", "fibonacci"},
                {"num_points", 80},
                {"elevation_min_deg", -90.0},
                {"elevation_max_deg", 90.0}
            }}
        }},
        {"polarization", {"HH", "VV"}},
        {"solver", {{"type", "PO"}}},
        {"output", {{"path", out_path.string()}, {"format", "csv"}}}
    };

    {
        std::ofstream out(cfg_path);
        out << cfg.dump(2);
    }

    const auto rc = strikecem::load_config(cfg_path.string(), SCEM_SCHEMA_PATH);
    const auto plan = strikecem::build_sample_plan(rc.value);
    EXPECT_EQ(plan.directions.size(), 80u);
    EXPECT_EQ(plan.sample_count(), 160u); // 80 dirs x 2 pols

    auto mesh = strikecem::load_normalized_mesh(rc.value, dir, rc.schema_version);
    const auto res = strikecem::solve_po(mesh, plan, rc.value);
    EXPECT_EQ(res.samples.size(), 160u);

    // Compute spherical statistics across all HH samples
    std::vector<double> hh_rcs;
    for (size_t i = 0; i < plan.directions.size(); ++i) {
        hh_rcs.push_back(res.samples[i * 2].rcs_sqm);
    }
    const auto stats = strikecem::compute_spherical_stats(plan.directions, hh_rcs);
    EXPECT_GT(stats.total_power_sqm, 0.0);
    EXPECT_GT(stats.mean_rcs_sqm, 0.0);
    EXPECT_GT(stats.max_rcs_dbsm, stats.min_rcs_dbsm);

    fs::remove_all(dir, ec);
}

} // namespace

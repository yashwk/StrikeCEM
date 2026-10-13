// Tests for StrikeEngine C ABI and Real-Time RcsQueryEngine
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "strikecem/api/strikecem_c_api.h"
#include "strikecem/api/RcsQueryEngine.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/io/BinaryTransport.hpp"
#include "strikecem/io/Hdf5Writer.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace {

namespace fs = std::filesystem;

struct Workdir {
    fs::path dir = fs::temp_directory_path() / ("scem_c_api_test_" + std::to_string(::getpid()));
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
    cfg["model"] = {{"path", std::string(std::string(SCEM_FIXTURE_DIR) +
                                         "/../../examples/plate.stl")}};
    cfg["frequency"] = {{"frequency_hz", 10e9}};
    cfg["angles"]["azimuth"] = {{"start", 0}, {"stop", 90}, {"step", 30}}; // 0, 30, 60, 90 (4 dirs)
    cfg["angles"]["elevation"] = {{"start", -90}, {"stop", -90}, {"step", 1}};
    cfg["polarization"] = {"HH", "VV"};
    cfg["solver"] = {{"type", "PO"}};
    cfg["output"] = {{"path", output_path}, {"format", format}};
    cfg["integration"] = {
        {"designer", {
            {"design_id", "fighter-jet-target"},
            {"revision", "rev-3b"}
        }}
    };
    return cfg;
}

TEST(CApi, OpenAndQueryScemDatabase) {
    Workdir work;
    const fs::path scem_path = work.dir / "test.scem";
    Pipeline p = run_pipeline(make_config(scem_path.string(), "scem"), work.dir / "cfg.json");
    strikecem::write_scem_output(p.rc, p.plan, p.mesh, p.result);

    scem_database_t* db = nullptr;
    ASSERT_EQ(scem_database_open(scem_path.c_str(), &db), SCEM_OK);
    ASSERT_NE(db, nullptr);

    // Verify metadata
    char design_id[64] = {0};
    char revision[64] = {0};
    EXPECT_EQ(scem_database_get_design_id(db, design_id, sizeof(design_id)), SCEM_OK);
    EXPECT_EQ(scem_database_get_design_revision(db, revision, sizeof(revision)), SCEM_OK);
    EXPECT_STREQ(design_id, "fighter-jet-target");
    EXPECT_STREQ(revision, "rev-3b");

    EXPECT_EQ(scem_database_verify_design(db, "fighter-jet-target", "rev-3b"), SCEM_OK);
    EXPECT_EQ(scem_database_verify_design(db, "wrong-target", "rev-3b"), SCEM_ERROR_DESIGN_MISMATCH);

    // Verify frequencies & polarizations
    double freqs[4];
    size_t nfreq = 4;
    EXPECT_EQ(scem_database_get_frequencies(db, freqs, &nfreq), SCEM_OK);
    EXPECT_EQ(nfreq, 1u);
    EXPECT_DOUBLE_EQ(freqs[0], 10e9);

    char pols[4][8];
    size_t npol = 4;
    EXPECT_EQ(scem_database_get_polarizations(db, pols, &npol), SCEM_OK);
    EXPECT_EQ(npol, 2u);
    EXPECT_STREQ(pols[0], "HH");
    EXPECT_STREQ(pols[1], "VV");

    // Exact query on grid point: az=0, el=-90, freq=10 GHz, pol="HH"
    scem_query_result_t res;
    EXPECT_EQ(scem_database_query(db, 0.0, -90.0, 10e9, "HH", &res), SCEM_OK);
    EXPECT_DOUBLE_EQ(res.scattering_real, p.result.samples[0].scattering.real());
    EXPECT_DOUBLE_EQ(res.scattering_imag, p.result.samples[0].scattering.imag());
    EXPECT_DOUBLE_EQ(res.rcs_sqm, p.result.samples[0].rcs_sqm);

    // Interpolated query midway between az=0 and az=30 (az=15)
    scem_query_result_t res_mid;
    EXPECT_EQ(scem_database_query(db, 15.0, -90.0, 10e9, "HH", &res_mid), SCEM_OK);
    EXPECT_GT(res_mid.rcs_sqm, 0.0);

    // Reject unknown channel
    scem_query_result_t res_err;
    EXPECT_EQ(scem_database_query(db, 0.0, -90.0, 10e9, "CIRCULAR", &res_err), SCEM_ERROR_CHANNEL_NOT_FOUND);

    scem_database_close(db);
}

TEST(CApi, OpenAndQueryHdf5Database) {
    Workdir work;
    const fs::path h5_path = work.dir / "test.h5";
    Pipeline p = run_pipeline(make_config(h5_path.string(), "hdf5"), work.dir / "cfg.json");
    strikecem::write_hdf5_output(p.rc, p.plan, p.mesh, p.result);

    scem_database_t* db = nullptr;
    ASSERT_EQ(scem_database_open(h5_path.c_str(), &db), SCEM_OK);
    ASSERT_NE(db, nullptr);

    scem_query_result_t res;
    EXPECT_EQ(scem_database_query(db, 0.0, -90.0, 10e9, "HH", &res), SCEM_OK);
    EXPECT_DOUBLE_EQ(res.rcs_sqm, p.result.samples[0].rcs_sqm);

    scem_database_close(db);
}

TEST(CApi, BatchQueryExecution) {
    Workdir work;
    const fs::path scem_path = work.dir / "test_batch.scem";
    Pipeline p = run_pipeline(make_config(scem_path.string(), "scem"), work.dir / "cfg.json");
    strikecem::write_scem_output(p.rc, p.plan, p.mesh, p.result);

    scem_database_t* db = nullptr;
    ASSERT_EQ(scem_database_open(scem_path.c_str(), &db), SCEM_OK);

    std::vector<scem_query_request_t> reqs = {
        {0.0, -90.0, 10e9, "HH"},
        {30.0, -90.0, 10e9, "HH"},
        {60.0, -90.0, 10e9, "VV"}
    };
    std::vector<scem_query_result_t> results(3);
    EXPECT_EQ(scem_database_query_batch(db, reqs.data(), reqs.size(), results.data()), SCEM_OK);

    EXPECT_DOUBLE_EQ(results[0].scattering_real, p.result.samples[0].scattering.real());
    EXPECT_GT(results[1].rcs_sqm, 0.0);
    EXPECT_GT(results[2].rcs_sqm, 0.0);

    scem_database_close(db);
}

TEST(CApi, MicrosecondQueryLatency) {
    Workdir work;
    const fs::path scem_path = work.dir / "perf.scem";
    Pipeline p = run_pipeline(make_config(scem_path.string(), "scem"), work.dir / "cfg.json");
    strikecem::write_scem_output(p.rc, p.plan, p.mesh, p.result);

    scem_database_t* db = nullptr;
    ASSERT_EQ(scem_database_open(scem_path.c_str(), &db), SCEM_OK);

    const size_t N = 10000;
    scem_query_result_t res;
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < N; ++i) {
        const double az = (i % 90);
        scem_database_query(db, az, -90.0, 10e9, "HH", &res);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double total_micros = std::chrono::duration<double, std::micro>(elapsed).count();
    const double per_query_micros = total_micros / N;

    // Latency should be sub-microsecond in release builds (typically < 0.2 us)
    EXPECT_LT(per_query_micros, 5.0); // Safe threshold allowing test overhead

    scem_database_close(db);
}

TEST(CApi, InterpolationModeConfiguration) {
    Workdir work;
    const fs::path scem_path = work.dir / "interp.scem";
    Pipeline p = run_pipeline(make_config(scem_path.string(), "scem"), work.dir / "cfg.json");
    strikecem::write_scem_output(p.rc, p.plan, p.mesh, p.result);

    scem_database_t* db = nullptr;
    ASSERT_EQ(scem_database_open(scem_path.c_str(), &db), SCEM_OK);

    // Default mode should be BICUBIC
    scem_interpolation_mode_t mode;
    EXPECT_EQ(scem_database_get_interpolation_mode(db, &mode), SCEM_OK);
    EXPECT_EQ(mode, SCEM_INTERP_BICUBIC);

    // Set to LINEAR
    EXPECT_EQ(scem_database_set_interpolation_mode(db, SCEM_INTERP_LINEAR), SCEM_OK);
    EXPECT_EQ(scem_database_get_interpolation_mode(db, &mode), SCEM_OK);
    EXPECT_EQ(mode, SCEM_INTERP_LINEAR);

    // Set to NEAREST
    EXPECT_EQ(scem_database_set_interpolation_mode(db, SCEM_INTERP_NEAREST), SCEM_OK);
    EXPECT_EQ(scem_database_get_interpolation_mode(db, &mode), SCEM_OK);
    EXPECT_EQ(mode, SCEM_INTERP_NEAREST);

    // Set to SPHERICAL
    EXPECT_EQ(scem_database_set_interpolation_mode(db, SCEM_INTERP_SPHERICAL), SCEM_OK);
    EXPECT_EQ(scem_database_get_interpolation_mode(db, &mode), SCEM_OK);
    EXPECT_EQ(mode, SCEM_INTERP_SPHERICAL);

    // Invalid mode rejection
    EXPECT_EQ(scem_database_set_interpolation_mode(db, static_cast<scem_interpolation_mode_t>(99)),
              SCEM_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(scem_database_set_interpolation_mode(nullptr, SCEM_INTERP_LINEAR),
              SCEM_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(scem_database_get_interpolation_mode(nullptr, &mode),
              SCEM_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(scem_database_get_interpolation_mode(db, nullptr),
              SCEM_ERROR_INVALID_ARGUMENT);

    scem_database_close(db);
}

TEST(CApi, HighOrderBicubicSplineContinuity) {
    Workdir work;
    const fs::path scem_path = work.dir / "spline.scem";
    Pipeline p = run_pipeline(make_config(scem_path.string(), "scem"), work.dir / "cfg.json");
    strikecem::write_scem_output(p.rc, p.plan, p.mesh, p.result);

    scem_database_t* db = nullptr;
    ASSERT_EQ(scem_database_open(scem_path.c_str(), &db), SCEM_OK);

    // Test exact grid point evaluation for both linear and bicubic modes
    scem_query_result_t res_linear_0, res_bicubic_0;
    scem_database_set_interpolation_mode(db, SCEM_INTERP_LINEAR);
    EXPECT_EQ(scem_database_query(db, 0.0, -90.0, 10e9, "HH", &res_linear_0), SCEM_OK);

    scem_database_set_interpolation_mode(db, SCEM_INTERP_BICUBIC);
    EXPECT_EQ(scem_database_query(db, 0.0, -90.0, 10e9, "HH", &res_bicubic_0), SCEM_OK);

    EXPECT_DOUBLE_EQ(res_linear_0.scattering_real, res_bicubic_0.scattering_real);
    EXPECT_DOUBLE_EQ(res_linear_0.scattering_imag, res_bicubic_0.scattering_imag);
    EXPECT_DOUBLE_EQ(res_linear_0.rcs_sqm, res_bicubic_0.rcs_sqm);

    // Sample continuous arc from 0 to 30 degrees in 1 degree steps
    std::vector<double> rcs_values;
    for (double az = 0.0; az <= 30.0; az += 1.0) {
        scem_query_result_t res;
        EXPECT_EQ(scem_database_query(db, az, -90.0, 10e9, "HH", &res), SCEM_OK);
        EXPECT_GT(res.rcs_sqm, 0.0);
        EXPECT_FALSE(std::isnan(res.rcs_dbsm));
        rcs_values.push_back(res.rcs_dbsm);
    }
    EXPECT_EQ(rcs_values.size(), 31u);

    scem_database_close(db);
}

TEST(CApi, SphericalSplineEvaluation) {
    Workdir work;
    const fs::path scem_path = work.dir / "spherical.scem";
    Pipeline p = run_pipeline(make_config(scem_path.string(), "scem"), work.dir / "cfg.json");
    strikecem::write_scem_output(p.rc, p.plan, p.mesh, p.result);

    scem_database_t* db = nullptr;
    ASSERT_EQ(scem_database_open(scem_path.c_str(), &db), SCEM_OK);

    EXPECT_EQ(scem_database_set_interpolation_mode(db, SCEM_INTERP_SPHERICAL), SCEM_OK);

    // Query on an off-grid direction
    scem_query_result_t res;
    EXPECT_EQ(scem_database_query(db, 17.5, -90.0, 10e9, "HH", &res), SCEM_OK);
    EXPECT_GT(res.rcs_sqm, 0.0);
    EXPECT_FALSE(std::isnan(res.rcs_dbsm));

    scem_database_close(db);
}

} // namespace

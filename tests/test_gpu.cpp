// GPU backend tests: device presence, CPU/GPU tolerance gates per
// precision, bitwise rerun determinism, batch override, and refusal paths.
// Device tests skip loudly when no CUDA device answers.
#include <cmath>
#include <gtest/gtest.h>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/GpuPO.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace {

std::string fixture(const std::string& name) { return std::string(SCEM_FIXTURE_DIR) + "/" + name; }

struct Case {
    strikecem::ResolvedConfig rc;
    strikecem::NormalizedMesh mesh;
};

Case load_case(const std::string& name) {
    Case c;
    c.rc = strikecem::load_config(fixture(name), SCEM_SCHEMA_PATH);
    c.mesh = strikecem::load_normalized_mesh(c.rc.value, SCEM_FIXTURE_DIR, c.rc.schema_version);
    return c;
}

strikecem::SamplePlan make_plan(const std::vector<double>& freqs,
                                const std::vector<std::pair<double, double>>& dirs,
                                const std::vector<std::string>& pols) {
    strikecem::SamplePlan plan;
    plan.frequencies_hz = freqs;
    uint32_t id = 0;
    for (const auto& [az, el] : dirs)
        plan.directions.push_back(
            {id++, az, el, strikecem::direction_from_az_el(az, el)});
    plan.polarizations = pols;
    return plan;
}

#ifdef SCEM_ENABLE_CUDA

strikecem::PoResult solve_backend(const Case& c, const strikecem::SamplePlan& plan,
                                  const std::string& backend, const std::string& precision) {
    nlohmann::json cfg = c.rc.value;
    cfg["execution"]["accelerator"] = backend;
    cfg["solver"]["precision"] = precision;
    return strikecem::solve_po(c.mesh, plan, cfg);
}

void expect_close(const strikecem::PoResult& gpu, const strikecem::PoResult& cpu, double rel,
                  double floor) {
    ASSERT_EQ(gpu.samples.size(), cpu.samples.size());
    for (size_t i = 0; i < cpu.samples.size(); ++i) {
        EXPECT_EQ(gpu.samples[i].sample_id, cpu.samples[i].sample_id);
        EXPECT_EQ(gpu.samples[i].lit_facets, cpu.samples[i].lit_facets);
        const double scale = std::abs(cpu.samples[i].scattering);
        EXPECT_NEAR(std::abs(gpu.samples[i].scattering - cpu.samples[i].scattering), 0.0,
                    rel * scale + floor)
            << "row " << i;
    }
}

TEST(Gpu, DevicePresent) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const auto info = strikecem::cuda::cuda_device_info(0);
    EXPECT_FALSE(info.name.empty());
    EXPECT_GT(info.total_bytes, 0u);
    EXPECT_GT(info.free_bytes, 0u);
}

TEST(Gpu, MatchesCpuFloat64) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const Case c = load_case("valid_sweep_grid.json");
    const auto plan = make_plan({10e9, 11e9}, {{30.0, 20.0}, {0.0, -90.0}},
                                {"HH", "VV", "HV", "RHCP"});
    const auto cpu = solve_backend(c, plan, "cpu", "float64");
    const auto gpu = solve_backend(c, plan, "cuda", "float64");
    expect_close(gpu, cpu, 1e-9, 1e-12);
}

TEST(Gpu, MatchesCpuFloat32) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const Case c = load_case("valid_sweep_grid.json");
    const auto plan = make_plan({10e9}, {{30.0, 20.0}}, {"HH", "VV", "HV", "RHCP"});
    const auto cpu = solve_backend(c, plan, "cpu", "float32");
    const auto gpu = solve_backend(c, plan, "cuda", "float32");
    expect_close(gpu, cpu, 1e-3, 1e-6);
}

TEST(Gpu, SphereMatchesCpu) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const Case c = load_case("valid_sphere.json");
    const auto plan = make_plan({1e9}, {{0.0, 0.0}, {45.0, 30.0}}, {"HH", "VV"});
    const auto cpu = solve_backend(c, plan, "cpu", "float64");
    const auto gpu = solve_backend(c, plan, "cuda", "float64");
    expect_close(gpu, cpu, 1e-9, 1e-12);
}

TEST(Gpu, DeterministicRerun) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const Case c = load_case("valid_sweep_grid.json");
    const auto plan = make_plan({10e9}, {{30.0, 20.0}}, {"HH", "VV"});
    const auto a = solve_backend(c, plan, "cuda", "float64");
    const auto b = solve_backend(c, plan, "cuda", "float64");
    ASSERT_EQ(a.samples.size(), b.samples.size());
    for (size_t i = 0; i < a.samples.size(); ++i)
        EXPECT_EQ(a.samples[i].scattering, b.samples[i].scattering);
}

TEST(Gpu, BatchOverrideMatches) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const Case c = load_case("valid_sweep_grid.json");
    const auto plan = make_plan({10e9}, {{0.0, 0.0}, {90.0, 0.0}, {180.0, 0.0}}, {"HH"});
    nlohmann::json one_unit = c.rc.value;
    one_unit["execution"]["accelerator"] = "cuda";
    one_unit["execution"]["cuda_batch_units"] = 1;
    const auto batched = strikecem::solve_po(c.mesh, plan, one_unit);
    const auto cpu = solve_backend(c, plan, "cpu", "float64");
    expect_close(batched, cpu, 1e-9, 1e-12);
}

TEST(Gpu, BadDeviceIdRefuses) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    EXPECT_THROW(strikecem::cuda::cuda_device_info(999), strikecem::cuda::CudaError);
    const Case c = load_case("valid_minimal.json");
    const auto plan = make_plan({10e9}, {{0.0, -90.0}}, {"HH"});
    nlohmann::json bad = c.rc.value;
    bad["execution"]["accelerator"] = "cuda";
    bad["execution"]["cuda_device_id"] = 999;
    EXPECT_THROW(strikecem::solve_po(c.mesh, plan, bad), strikecem::cuda::CudaError);
}

#else

TEST(Gpu, CudaDisabledStubRefuses) {
    EXPECT_FALSE(strikecem::cuda::cuda_available());
    EXPECT_THROW(strikecem::cuda::cuda_device_info(0), strikecem::cuda::CudaError);
    const std::string path = std::string(SCEM_FIXTURE_DIR) + "/valid_minimal.json";
    auto rc = strikecem::load_config(path, SCEM_SCHEMA_PATH);
    auto mesh = strikecem::load_normalized_mesh(rc.value, SCEM_FIXTURE_DIR, rc.schema_version);
    strikecem::SamplePlan plan;
    plan.frequencies_hz = {10e9};
    plan.directions.push_back({0, 0.0, -90.0, strikecem::direction_from_az_el(0.0, -90.0)});
    plan.polarizations = {"HH"};
    nlohmann::json cfg = rc.value;
    cfg["execution"]["accelerator"] = "cuda";
    EXPECT_THROW(strikecem::solve_po(mesh, plan, cfg), strikecem::cuda::CudaError);
}

#endif

} // namespace

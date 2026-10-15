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

TEST(Gpu, MatchesCpuMixedPrecision) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const Case c = load_case("valid_sweep_grid.json");
    const auto plan = make_plan({10e9}, {{30.0, 20.0}}, {"HH", "VV", "HV", "RHCP"});
    const auto cpu = solve_backend(c, plan, "cpu", "float64");
    const auto gpu = solve_backend(c, plan, "cuda", "mixed");
    expect_close(gpu, cpu, 1e-4, 1e-6);
}

TEST(Gpu, MixedPrecisionFootprintReduction) {
    nlohmann::json cfg_f64 = {{"solver", {{"precision", "float64"}}}};
    nlohmann::json cfg_mixed = {{"solver", {{"precision", "mixed"}}}};
    nlohmann::json cfg_f32 = {{"solver", {{"precision", "float32"}}}};

    const size_t ntris = 1000;
    EXPECT_EQ(strikecem::cuda::cuda_footprint_bytes(cfg_f64, ntris), ntris * 7 * 8);
    EXPECT_EQ(strikecem::cuda::cuda_footprint_bytes(cfg_mixed, ntris), ntris * 7 * 4);
    EXPECT_EQ(strikecem::cuda::cuda_footprint_bytes(cfg_f32, ntris), ntris * 7 * 4);
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

TEST(Gpu, BistaticMatchesCpu) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const Case c = load_case("valid_sphere.json");
    strikecem::SamplePlan plan;
    plan.frequencies_hz = {10e9};
    plan.polarizations = {"HH", "VV"};
    strikecem::Direction d;
    d.id = 0;
    d.azimuth_deg = 30.0;
    d.elevation_deg = 15.0;
    d.k_hat = strikecem::direction_from_az_el(30.0, 15.0);
    d.is_bistatic = true;
    d.r_obs_hat = strikecem::direction_from_az_el(45.0, 20.0);
    plan.directions.push_back(d);

    const auto cpu = solve_backend(c, plan, "cpu", "float64");
    const auto gpu = solve_backend(c, plan, "cuda", "float64");
    expect_close(gpu, cpu, 1e-9, 1e-12);
}

TEST(Gpu, DenseAzimuthSweepMatchesCpu) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const Case c = load_case("valid_sphere.json");
    std::vector<std::pair<double, double>> dirs;
    for (double az = 0.0; az < 360.0; az += 15.0) {
        dirs.emplace_back(az, 0.0);
    }
    const auto plan = make_plan({10e9}, dirs, {"HH", "VV"});
    const auto cpu = solve_backend(c, plan, "cpu", "float64");
    const auto gpu = solve_backend(c, plan, "cuda", "float64");
    expect_close(gpu, cpu, 1e-9, 1e-12);
}

strikecem::NormalizedMesh make_test_dihedral(int n) {
    strikecem::NormalizedMesh m;
    auto h = [&](int i, int j) { return uint32_t(j * (n + 1) + i); };
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) m.vertices.push_back({double(i) / n, double(j) / n, 0.0});
    const uint32_t vbase = uint32_t(m.vertices.size());
    for (int j = 0; j <= n; ++j)
        for (int k = 0; k <= n; ++k) m.vertices.push_back({0.0, double(j) / n, double(k) / n});
    auto w = [&](int j, int k) { return vbase + uint32_t(j * (n + 1) + k); };
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            m.triangles.push_back({h(i, j), h(i + 1, j), h(i + 1, j + 1)});
            m.triangles.push_back({h(i, j), h(i + 1, j + 1), h(i, j + 1)});
            m.normals.push_back({0, 0, 1});
            m.normals.push_back({0, 0, 1});
            m.areas.push_back(0.5 / (n * n));
            m.areas.push_back(0.5 / (n * n));
        }
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k) {
            m.triangles.push_back({w(j, k), w(j + 1, k), w(j + 1, k + 1)});
            m.triangles.push_back({w(j, k), w(j + 1, k + 1), w(j, k + 1)});
            m.normals.push_back({1, 0, 0});
            m.normals.push_back({1, 0, 0});
            m.areas.push_back(0.5 / (n * n));
            m.areas.push_back(0.5 / (n * n));
        }
    m.report.bbox_min = {0, 0, 0};
    m.report.bbox_max = {1, 1, 1};
    return m;
}

strikecem::NormalizedMesh make_test_trihedral(int n) {
    strikecem::NormalizedMesh m;
    auto grid = [&](double ox, double oy, double oz, double ax, double ay, double az, double bx,
                    double by, double bz) {
        const uint32_t base = uint32_t(m.vertices.size());
        for (int j = 0; j <= n; ++j)
            for (int i = 0; i <= n; ++i)
                m.vertices.push_back({ox + (ax * i + bx * j) / n, oy + (ay * i + by * j) / n,
                                      oz + (az * i + bz * j) / n});
        return base;
    };
    auto quad = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d, geom::Vec3d nrm) {
        m.triangles.push_back({a, b, c});
        m.triangles.push_back({a, c, d});
        m.normals.push_back(nrm);
        m.normals.push_back(nrm);
        m.areas.push_back(0.5 / (n * n));
        m.areas.push_back(0.5 / (n * n));
    };
    const uint32_t h = grid(0, 0, 0, 1, 0, 0, 0, 1, 0);
    const uint32_t x = grid(0, 0, 0, 0, 1, 0, 0, 0, 1);
    const uint32_t y = grid(0, 0, 0, 1, 0, 0, 0, 0, 1);
    auto id = [&](uint32_t base, int i, int j) { return base + uint32_t(j * (n + 1) + i); };
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            quad(id(h, i, j), id(h, i + 1, j), id(h, i + 1, j + 1), id(h, i, j + 1), {0, 0, 1});
            quad(id(x, i, j), id(x, i + 1, j), id(x, i + 1, j + 1), id(x, i, j + 1), {1, 0, 0});
            quad(id(y, i, j), id(y, i + 1, j), id(y, i + 1, j + 1), id(y, i, j + 1), {0, 1, 0});
        }
    m.report.bbox_min = {0, 0, 0};
    m.report.bbox_max = {1, 1, 1};
    return m;
}

TEST(Gpu, ShadowingMatchesCpu) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const Case c = load_case("valid_dihedral.json");
    const auto plan = make_plan({10e9}, {{0.0, -45.0}, {90.0, -45.0}}, {"HH", "VV"});
    const strikecem::GoOptions shade{true, 1};
    nlohmann::json cpu_cfg = c.rc.value;
    cpu_cfg["execution"]["accelerator"] = "cpu";
    const auto cpu = strikecem::solve_po(c.mesh, plan, cpu_cfg, {}, shade);
    nlohmann::json gpu_cfg = c.rc.value;
    gpu_cfg["execution"]["accelerator"] = "cuda";
    const auto gpu = strikecem::solve_po(c.mesh, plan, gpu_cfg, {}, shade);
    expect_close(gpu, cpu, 1e-9, 1e-12);
}

TEST(Gpu, TwoBounceDihedralMatchesCpu) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const auto c = load_case("valid_dihedral.json");
    const auto mesh = make_test_dihedral(6);
    const auto plan = make_plan({10e9}, {{180.0, -45.0}}, {"HH", "VV"});
    const strikecem::GoOptions go{false, 2};
    nlohmann::json cpu_cfg = c.rc.value;
    cpu_cfg["execution"]["accelerator"] = "cpu";
    const auto cpu = strikecem::solve_po(mesh, plan, cpu_cfg, {}, go);
    nlohmann::json gpu_cfg = c.rc.value;
    gpu_cfg["execution"]["accelerator"] = "cuda";
    const auto gpu = strikecem::solve_po(mesh, plan, gpu_cfg, {}, go);
    expect_close(gpu, cpu, 1e-9, 1e-12);
}

TEST(Gpu, TwoBounceDihedralAnalyticConvergence) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const auto c = load_case("valid_dihedral.json");
    const auto mesh = make_test_dihedral(10);
    const auto plan = make_plan({1e9}, {{180.0, -45.0}}, {"HH", "VV"});
    const strikecem::GoOptions go{false, 2};
    nlohmann::json gpu_cfg = c.rc.value;
    gpu_cfg["execution"]["accelerator"] = "cuda";
    const auto gpu = strikecem::solve_po(mesh, plan, gpu_cfg, {}, go);
    const double lambda = strikecem::kSpeedOfLight / 1e9;
    const double analytic = 8.0 * 3.14159265358979323846 / (lambda * lambda);
    for (size_t i = 0; i < 2; ++i) {
        const double ratio = gpu.samples[i].rcs_sqm / analytic;
        EXPECT_GT(ratio, 0.8);
        EXPECT_LT(ratio, 1.2);
    }
}

TEST(Gpu, ThreeBounceTrihedralMatchesCpu) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const auto c = load_case("valid_dihedral.json");
    const auto mesh = make_test_trihedral(4);
    const double q = 1.0 / std::sqrt(3.0);
    const double az_deg = 180.0 + std::atan2(q, q) * 180.0 / 3.14159265358979323846;
    const double el_deg = -std::asin(q) * 180.0 / 3.14159265358979323846;
    const auto plan = make_plan({1e9}, {{az_deg, el_deg}}, {"HH", "VV"});
    const strikecem::GoOptions go{false, 3};
    nlohmann::json cpu_cfg = c.rc.value;
    cpu_cfg["execution"]["accelerator"] = "cpu";
    const auto cpu = strikecem::solve_po(mesh, plan, cpu_cfg, {}, go);
    nlohmann::json gpu_cfg = c.rc.value;
    gpu_cfg["execution"]["accelerator"] = "cuda";
    const auto gpu = strikecem::solve_po(mesh, plan, gpu_cfg, {}, go);
    expect_close(gpu, cpu, 1e-9, 1e-12);
}

TEST(Gpu, ImpedanceRAMMatchesCpu) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const Case c = load_case("valid_minimal.json");
    const auto plan = make_plan({10e9}, {{0.0, -90.0}, {30.0, -45.0}}, {"HH", "VV"});

    strikecem::MaterialModel ram_model;
    ram_model.tag = "ram_sibc";
    ram_model.type = strikecem::WallType::Impedance;
    ram_model.surface_impedance = {1.0, 0.0}; // Matched normalized impedance
    strikecem::MaterialOptions ram_opts{true, strikecem::WallType::Impedance, ram_model};

    nlohmann::json cpu_cfg = c.rc.value;
    cpu_cfg["execution"]["accelerator"] = "cpu";
    const auto cpu = strikecem::solve_po(c.mesh, plan, cpu_cfg, {}, {}, ram_opts);

    nlohmann::json gpu_cfg = c.rc.value;
    gpu_cfg["execution"]["accelerator"] = "cuda";
    const auto gpu = strikecem::solve_po(c.mesh, plan, gpu_cfg, {}, {}, ram_opts);

    expect_close(gpu, cpu, 1e-9, 1e-12);

    // Verify significant RCS reduction vs bare PEC
    const auto bare_gpu = strikecem::solve_po(c.mesh, plan, gpu_cfg);
    EXPECT_LT(gpu.samples[0].rcs_sqm, 0.01 * bare_gpu.samples[0].rcs_sqm);
}

TEST(Gpu, DielectricMatchesCpu) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const Case c = load_case("valid_minimal.json");
    const auto plan = make_plan({10e9}, {{0.0, -90.0}, {45.0, -30.0}}, {"HH", "VV"});

    strikecem::MaterialModel d_model;
    d_model.tag = "glass";
    d_model.type = strikecem::WallType::Dielectric;
    d_model.table.push_back({10e9, {{4.0, -0.2}, {1.0, 0.0}}, 0.0});
    strikecem::MaterialOptions d_opts{true, strikecem::WallType::Dielectric, d_model};

    nlohmann::json cpu_cfg = c.rc.value;
    cpu_cfg["execution"]["accelerator"] = "cpu";
    const auto cpu = strikecem::solve_po(c.mesh, plan, cpu_cfg, {}, {}, d_opts);

    nlohmann::json gpu_cfg = c.rc.value;
    gpu_cfg["execution"]["accelerator"] = "cuda";
    const auto gpu = strikecem::solve_po(c.mesh, plan, gpu_cfg, {}, {}, d_opts);

    expect_close(gpu, cpu, 1e-9, 1e-12);
}

TEST(Gpu, CoatedRAMMatchesCpu) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const Case c = load_case("valid_minimal.json");
    const auto plan = make_plan({10e9}, {{0.0, -90.0}}, {"HH", "VV"});

    strikecem::MaterialModel coated_model;
    coated_model.tag = "stealth_paint";
    coated_model.type = strikecem::WallType::Coated;
    coated_model.layers.push_back({0.002, {{3.5, -0.5}, {1.2, -0.1}}});
    strikecem::MaterialOptions c_opts{true, strikecem::WallType::Coated, coated_model};

    nlohmann::json cpu_cfg = c.rc.value;
    cpu_cfg["execution"]["accelerator"] = "cpu";
    const auto cpu = strikecem::solve_po(c.mesh, plan, cpu_cfg, {}, {}, c_opts);

    nlohmann::json gpu_cfg = c.rc.value;
    gpu_cfg["execution"]["accelerator"] = "cuda";
    const auto gpu = strikecem::solve_po(c.mesh, plan, gpu_cfg, {}, {}, c_opts);

    expect_close(gpu, cpu, 1e-9, 1e-12);
}

TEST(Gpu, MultiMaterialMatchesCpu) {
    if (!strikecem::cuda::cuda_available()) GTEST_SKIP() << "no CUDA device; backend untested";
    const Case c = load_case("valid_minimal.json");
    const auto plan = make_plan({10e9}, {{0.0, -90.0}}, {"HH", "VV"});

    strikecem::MaterialModel pec_model;
    pec_model.tag = "pec";
    pec_model.type = strikecem::WallType::Pec;

    strikecem::MaterialModel ram_model;
    ram_model.tag = "ram";
    ram_model.type = strikecem::WallType::Impedance;
    ram_model.surface_impedance = {1.0, 0.0};

    strikecem::MaterialOptions mixed_opts;
    mixed_opts.enabled = true;
    mixed_opts.models = {pec_model, ram_model};
    mixed_opts.facet_material_indices = {0, 1}; // 2 triangles in minimal mesh

    nlohmann::json cpu_cfg = c.rc.value;
    cpu_cfg["execution"]["accelerator"] = "cpu";
    const auto cpu = strikecem::solve_po(c.mesh, plan, cpu_cfg, {}, {}, mixed_opts);

    nlohmann::json gpu_cfg = c.rc.value;
    gpu_cfg["execution"]["accelerator"] = "cuda";
    const auto gpu = strikecem::solve_po(c.mesh, plan, gpu_cfg, {}, {}, mixed_opts);

    expect_close(gpu, cpu, 1e-9, 1e-12);
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

#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include <fstream>
#include <filesystem>

#include "strikecem/geometry/Symmetry.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/solvers/GpuPO.hpp"

namespace fs = std::filesystem;
using namespace strikecem;

namespace {

NormalizedMesh make_symmetric_plate() {
    NormalizedMesh m;
    m.vertices = {
        {0.0, 0.0, 0.0},  // 0
        {1.0, 0.0, 0.0},  // 1
        {1.0, 1.0, 0.0},  // 2
        {0.0, 1.0, 0.0},  // 3
        {1.0, -1.0, 0.0}, // 4
        {0.0, -1.0, 0.0}  // 5
    };
    // Triangles on +y side:
    m.triangles = {
        {0, 1, 2},
        {0, 2, 3},
        // Reflected triangles on -y side:
        {0, 4, 1},
        {0, 5, 4}
    };
    for (size_t i = 0; i < 4; ++i) {
        m.normals.push_back({0.0, 0.0, 1.0});
        m.areas.push_back(0.5);
    }
    m.report.vertex_count = m.vertices.size();
    m.report.triangle_count = m.triangles.size();
    m.report.total_area_m2 = 2.0;
    m.report.bbox_min = {0.0, -1.0, 0.0};
    m.report.bbox_max = {1.0, 1.0, 0.0};
    return m;
}

ResolvedConfig base_resolved() {
    return load_config(std::string(SCEM_FIXTURE_DIR) + "/valid_minimal.json", SCEM_SCHEMA_PATH);
}

} // namespace

TEST(Symmetry, ReflectionMath) {
    geom::Vec3d p{1.0, 2.0, 3.0};
    auto r_xz = reflect_point(p, "xz");
    EXPECT_DOUBLE_EQ(r_xz.x, 1.0);
    EXPECT_DOUBLE_EQ(r_xz.y, -2.0);
    EXPECT_DOUBLE_EQ(r_xz.z, 3.0);

    auto r_yz = reflect_point(p, "yz");
    EXPECT_DOUBLE_EQ(r_yz.x, -1.0);
    EXPECT_DOUBLE_EQ(r_yz.y, 2.0);
    EXPECT_DOUBLE_EQ(r_yz.z, 3.0);

    auto r_xy = reflect_point(p, "xy");
    EXPECT_DOUBLE_EQ(r_xy.x, 1.0);
    EXPECT_DOUBLE_EQ(r_xy.y, 2.0);
    EXPECT_DOUBLE_EQ(r_xy.z, -3.0);

    geom::Vec3d n{0.0, 1.0, 0.0};
    auto n_xz = reflect_normal(n, "xz");
    EXPECT_DOUBLE_EQ(n_xz.x, 0.0);
    EXPECT_DOUBLE_EQ(n_xz.y, -1.0);
    EXPECT_DOUBLE_EQ(n_xz.z, 0.0);

    EXPECT_THROW(reflect_point(p, "invalid"), std::invalid_argument);
    EXPECT_THROW(get_symmetry_coord(p, "invalid"), std::invalid_argument);
}

TEST(Symmetry, DetectionAndExtraction) {
    NormalizedMesh plate = make_symmetric_plate();

    SymmetryOptions opts;
    opts.enabled = true;
    opts.plane = "xz";
    opts.tolerance_m = 1e-4;

    EXPECT_TRUE(is_symmetric_mesh(plate, opts));

    // Shift plate so it is no longer symmetric about y=0
    NormalizedMesh shifted = plate;
    for (auto& v : shifted.vertices) v.y += 0.2;
    EXPECT_FALSE(is_symmetric_mesh(shifted, opts));

    // Extract half mesh
    HalfMeshResult half = extract_half_mesh(plate, opts);
    EXPECT_EQ(half.mesh.triangles.size(), 2u);
    EXPECT_EQ(half.original_indices.size(), 2u);
    EXPECT_EQ(half.reflected_indices.size(), 2u);

    EXPECT_EQ(half.original_indices[0], 0u);
    EXPECT_EQ(half.reflected_indices[0], 2u);
    EXPECT_EQ(half.original_indices[1], 1u);
    EXPECT_EQ(half.reflected_indices[1], 3u);
}

TEST(Symmetry, PoSolveEquivalence) {
    NormalizedMesh plate = make_symmetric_plate();

    SamplePlan plan;
    plan.frequencies_hz = {1.0e9};
    plan.polarizations = {"HH", "VV"};
    Direction dir;
    dir.azimuth_deg = 30.0;
    dir.elevation_deg = -60.0;
    const double az_rad = dir.azimuth_deg * std::numbers::pi / 180.0;
    const double el_rad = dir.elevation_deg * std::numbers::pi / 180.0;
    dir.k_hat = {-std::cos(el_rad) * std::cos(az_rad),
                 -std::cos(el_rad) * std::sin(az_rad),
                 -std::sin(el_rad)};
    plan.directions.push_back(dir);

    auto rc = base_resolved();
    rc.value["execution"]["accelerator"] = "cpu";
    rc.value["mesh"]["symmetry"]["enabled"] = false;
    rc.value["mesh"]["symmetry"]["plane"] = "xz";

    FringeOptions fringe;
    fringe.enabled = false;
    GoOptions go;
    go.shadowing = false;
    go.max_bounces = 1;
    MaterialOptions mat;
    mat.enabled = false;

    PoResult res_full = solve_po(plate, plan, rc.value, fringe, go, mat);

    rc.value["mesh"]["symmetry"]["enabled"] = true;
    PoResult res_sym = solve_po(plate, plan, rc.value, fringe, go, mat);

    ASSERT_EQ(res_full.samples.size(), res_sym.samples.size());
    for (size_t i = 0; i < res_full.samples.size(); ++i) {
        EXPECT_NEAR(res_full.samples[i].rcs_sqm, res_sym.samples[i].rcs_sqm, 1e-10);
        EXPECT_NEAR(res_full.samples[i].scattering.real(), res_sym.samples[i].scattering.real(), 1e-10);
        EXPECT_NEAR(res_full.samples[i].scattering.imag(), res_sym.samples[i].scattering.imag(), 1e-10);
    }
}

TEST(Symmetry, PhaseReferenceEnforcement) {
    const fs::path tmp_bad = fs::temp_directory_path() / "scem_sym_bad.json";
    const fs::path tmp_good = fs::temp_directory_path() / "scem_sym_good.json";

    nlohmann::json cfg = {
        {"scem_schema_version", "1.0"},
        {"model", {{"path", fs::canonical("examples/plate.stl").string()}, {"units", "m"}}},
        {"frequency", {{"frequency_hz", 10000000000ULL}}},
        {"mesh", {{"mode", "input"}, {"symmetry", {{"enabled", true}, {"plane", "xz"}, {"tolerance_m", 1e-4}}}}},
        {"physics", {{"incident_amplitude", 1.0}, {"phase_reference", {0.0, 0.5, 0.0}}}},
        {"angles", {{"azimuth", {{"start", 0}, {"stop", 0}, {"step", 1}}},
                    {"elevation", {{"start", 0}, {"stop", 0}, {"step", 1}}}}},
        {"polarization", {"HH"}},
        {"solver", {{"type", "PO"}}},
        {"output", {{"path", "test.h5"}, {"format", "hdf5"}}}
    };

    {
        std::ofstream f(tmp_bad);
        f << cfg.dump(2);
    }
    EXPECT_THROW(load_config(tmp_bad.string(), SCEM_SCHEMA_PATH), ConfigError);

    // Coordinate on symmetry plane (y=0 for "xz") should pass
    cfg["physics"]["phase_reference"] = {0.2, 0.0, -0.3};
    {
        std::ofstream f(tmp_good);
        f << cfg.dump(2);
    }
    EXPECT_NO_THROW(load_config(tmp_good.string(), SCEM_SCHEMA_PATH));

    fs::remove(tmp_bad);
    fs::remove(tmp_good);
}

#if defined(SCEM_ENABLE_CUDA)
TEST(Symmetry, GpuSolveEquivalence) {
    if (!cuda::cuda_available()) GTEST_SKIP() << "CUDA not available";

    NormalizedMesh plate = make_symmetric_plate();

    SamplePlan plan;
    plan.frequencies_hz = {1.0e9};
    plan.polarizations = {"HH", "VV"};
    Direction dir;
    dir.azimuth_deg = 30.0;
    dir.elevation_deg = -60.0;
    const double az_rad = dir.azimuth_deg * std::numbers::pi / 180.0;
    const double el_rad = dir.elevation_deg * std::numbers::pi / 180.0;
    dir.k_hat = {-std::cos(el_rad) * std::cos(az_rad),
                 -std::cos(el_rad) * std::sin(az_rad),
                 -std::sin(el_rad)};
    plan.directions.push_back(dir);

    auto rc = base_resolved();
    rc.value["execution"]["accelerator"] = "cuda";
    rc.value["execution"]["cuda_device_id"] = 0;
    rc.value["mesh"]["symmetry"]["enabled"] = false;
    rc.value["mesh"]["symmetry"]["plane"] = "xz";

    FringeOptions fringe;
    fringe.enabled = false;
    GoOptions go;
    go.shadowing = false;
    go.max_bounces = 1;
    MaterialOptions mat;
    mat.enabled = false;

    PoResult res_full = solve_po(plate, plan, rc.value, fringe, go, mat);

    rc.value["mesh"]["symmetry"]["enabled"] = true;
    PoResult res_sym = solve_po(plate, plan, rc.value, fringe, go, mat);

    ASSERT_EQ(res_full.samples.size(), res_sym.samples.size());
    for (size_t i = 0; i < res_full.samples.size(); ++i) {
        EXPECT_NEAR(res_full.samples[i].rcs_sqm, res_sym.samples[i].rcs_sqm, 1e-8);
    }
}
#endif

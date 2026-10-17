#include <cmath>
#include <fstream>
#include <numbers>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/core/Conventions.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/EdgeModel.hpp"
#include "strikecem/solvers/Material.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace {

constexpr double kSpeedOfLight = 299792458.0;

inline double to_dbsm(double sqm) {
    return (sqm > 0.0) ? 10.0 * std::log10(sqm) : -999.0;
}

strikecem::SamplePlan make_single_plan(double freq_hz, double az_deg, double el_deg,
                                       const std::string& pol) {
    strikecem::SamplePlan plan;
    plan.frequencies_hz = {freq_hz};
    plan.directions.push_back({0, az_deg, el_deg, strikecem::direction_from_az_el(az_deg, el_deg)});
    plan.polarizations = {pol};
    return plan;
}

strikecem::NormalizedMesh make_canonical_dihedral(double a = 0.5, double b = 0.5, int n = 16) {
    strikecem::NormalizedMesh mesh;
    // Plate 1: z=0, x in [0, a], y in [0, b], normal +z
    const uint32_t base1 = static_cast<uint32_t>(mesh.vertices.size());
    for (int j = 0; j <= n; ++j) {
        double y = b * (static_cast<double>(j) / n);
        for (int i = 0; i <= n; ++i) {
            double x = a * (static_cast<double>(i) / n);
            mesh.vertices.push_back({x, y, 0.0});
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            uint32_t p0 = base1 + j * (n + 1) + i;
            uint32_t p1 = base1 + j * (n + 1) + (i + 1);
            uint32_t p2 = base1 + (j + 1) * (n + 1) + (i + 1);
            uint32_t p3 = base1 + (j + 1) * (n + 1) + i;
            mesh.triangles.push_back({p0, p1, p2});
            mesh.triangles.push_back({p0, p2, p3});
            mesh.normals.push_back({0.0, 0.0, 1.0});
            mesh.normals.push_back({0.0, 0.0, 1.0});
            double area = 0.5 * (a / n) * (b / n);
            mesh.areas.push_back(area);
            mesh.areas.push_back(area);
        }
    }
    // Plate 2: x=0, z in [0, a], y in [0, b], normal +x
    const uint32_t base2 = static_cast<uint32_t>(mesh.vertices.size());
    for (int j = 0; j <= n; ++j) {
        double y = b * (static_cast<double>(j) / n);
        for (int k = 0; k <= n; ++k) {
            double z = a * (static_cast<double>(k) / n);
            mesh.vertices.push_back({0.0, y, z});
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
            uint32_t p0 = base2 + j * (n + 1) + k;
            uint32_t p1 = base2 + (j + 1) * (n + 1) + k;
            uint32_t p2 = base2 + (j + 1) * (n + 1) + (k + 1);
            uint32_t p3 = base2 + j * (n + 1) + (k + 1);
            mesh.triangles.push_back({p0, p1, p2});
            mesh.triangles.push_back({p0, p2, p3});
            mesh.normals.push_back({1.0, 0.0, 0.0});
            mesh.normals.push_back({1.0, 0.0, 0.0});
            double area = 0.5 * (a / n) * (b / n);
            mesh.areas.push_back(area);
            mesh.areas.push_back(area);
        }
    }
    mesh.report.vertex_count = mesh.vertices.size();
    mesh.report.triangle_count = mesh.triangles.size();
    mesh.report.bbox_min = {0.0, 0.0, 0.0};
    mesh.report.bbox_max = {a, b, a};
    return mesh;
}

} // namespace

// ============================================================================
// GOLDEN BENCHMARK 2: Canonical 90-Degree Dihedral Corner Reflector
// ============================================================================

TEST(GoldenBenchmarks, DihedralAnalyticalKnottPeak) {
    // E. F. Knott Analytical Formula: sigma_peak = (8 * pi * a^2 * b^2) / lambda^2
    const double a = 0.5, b = 0.5;
    const double freq = 10.0e9;
    const double lambda = kSpeedOfLight / freq;
    const double expected_sigma_sqm = (8.0 * std::numbers::pi * (a * a) * (b * b)) / (lambda * lambda);
    const double expected_sigma_dbsm = 10.0 * std::log10(expected_sigma_sqm); // ~ 32.425 dBsm

    const auto mesh = make_canonical_dihedral(a, b, 16);
    const auto rc = strikecem::load_config(std::string(SCEM_FIXTURE_DIR) + "/valid_minimal.json", SCEM_SCHEMA_PATH);

    strikecem::GoOptions go;
    go.shadowing = true;
    go.max_bounces = 2; // SBR double-bounce

    const auto plan_hh = make_single_plan(freq, 180.0, -45.0, "HH");
    const auto res_hh = strikecem::solve_po(mesh, plan_hh, rc.value, {}, go, {});
    const double hh_dbsm = to_dbsm(res_hh.samples[0].rcs_sqm);

    const auto plan_vv = make_single_plan(freq, 180.0, -45.0, "VV");
    const auto res_vv = strikecem::solve_po(mesh, plan_vv, rc.value, {}, go, {});
    const double vv_dbsm = to_dbsm(res_vv.samples[0].rcs_sqm);

    const double mean_dbsm = 0.5 * (hh_dbsm + vv_dbsm);
    EXPECT_NEAR(mean_dbsm, expected_sigma_dbsm, 1.0)
        << "Mean retroreflection peak (" << mean_dbsm << " dBsm) deviates from Knott ("
        << expected_sigma_dbsm << " dBsm)";
}

TEST(GoldenBenchmarks, DihedralMultiBounceGain) {
    const double a = 0.5, b = 0.5;
    const double freq = 10.0e9;
    const auto mesh = make_canonical_dihedral(a, b, 16);
    const auto rc = strikecem::load_config(std::string(SCEM_FIXTURE_DIR) + "/valid_minimal.json", SCEM_SCHEMA_PATH);

    strikecem::GoOptions go_1b{true, 1};
    strikecem::GoOptions go_2b{true, 2};

    // 1. Peak at -45 deg
    const auto plan_peak = make_single_plan(freq, 180.0, -45.0, "HH");
    const auto res_1b_peak = strikecem::solve_po(mesh, plan_peak, rc.value, {}, go_1b, {});
    const auto res_2b_peak = strikecem::solve_po(mesh, plan_peak, rc.value, {}, go_2b, {});
    const double gain_peak = to_dbsm(res_2b_peak.samples[0].rcs_sqm) - to_dbsm(res_1b_peak.samples[0].rcs_sqm);
    EXPECT_GT(gain_peak, 7.0) << "Expected multi-bounce gain > 7 dB at retroreflection peak";

    // 2. Off-specular at -60 deg
    const auto plan_60 = make_single_plan(freq, 180.0, -60.0, "HH");
    const auto res_1b_60 = strikecem::solve_po(mesh, plan_60, rc.value, {}, go_1b, {});
    const auto res_2b_60 = strikecem::solve_po(mesh, plan_60, rc.value, {}, go_2b, {});
    const double gain_60 = to_dbsm(res_2b_60.samples[0].rcs_sqm) - to_dbsm(res_1b_60.samples[0].rcs_sqm);
    EXPECT_GT(gain_60, 15.0) << "Expected multi-bounce gain > 15 dB at off-specular angle -60 deg";
}

TEST(GoldenBenchmarks, DihedralElevationSymmetry) {
    const double a = 0.5, b = 0.5;
    const double freq = 10.0e9;
    const auto mesh = make_canonical_dihedral(a, b, 16);
    const auto rc = strikecem::load_config(std::string(SCEM_FIXTURE_DIR) + "/valid_minimal.json", SCEM_SCHEMA_PATH);

    strikecem::GoOptions go{true, 2};

    for (double delta : {10.0, 20.0, 30.0}) {
        const double el_minus = -45.0 - delta;
        const double el_plus = -45.0 + delta;

        const auto res_minus = strikecem::solve_po(mesh, make_single_plan(freq, 180.0, el_minus, "HH"), rc.value, {}, go, {});
        const auto res_plus = strikecem::solve_po(mesh, make_single_plan(freq, 180.0, el_plus, "HH"), rc.value, {}, go, {});

        EXPECT_NEAR(to_dbsm(res_minus.samples[0].rcs_sqm), to_dbsm(res_plus.samples[0].rcs_sqm), 1e-4)
            << "Asymmetry detected at delta = " << delta << " deg";
    }
}

TEST(GoldenBenchmarks, DihedralRamCoatingAttenuation) {
    const double a = 0.5, b = 0.5;
    const double freq = 10.0e9;
    const auto mesh = make_canonical_dihedral(a, b, 16);
    const auto rc = strikecem::load_config(std::string(SCEM_FIXTURE_DIR) + "/valid_minimal.json", SCEM_SCHEMA_PATH);

    strikecem::GoOptions go{true, 2};

    // Bare PEC
    const auto res_pec = strikecem::solve_po(mesh, make_single_plan(freq, 180.0, -45.0, "HH"), rc.value, {}, go, {});

    // Lossy RAM Coating
    strikecem::MaterialModel ram_model;
    ram_model.tag = "ram";
    ram_model.type = strikecem::WallType::Coated;
    ram_model.layers.push_back({0.002, {{4.0, -1.5}, {1.2, -0.4}}});

    strikecem::MaterialOptions mat_opts;
    mat_opts.enabled = true;
    mat_opts.models = {ram_model};
    mat_opts.facet_material_indices.assign(mesh.triangles.size(), 0);

    const auto res_ram = strikecem::solve_po(mesh, make_single_plan(freq, 180.0, -45.0, "HH"), rc.value, {}, go, mat_opts);

    const double atten_db = to_dbsm(res_pec.samples[0].rcs_sqm) - to_dbsm(res_ram.samples[0].rcs_sqm);
    EXPECT_GT(atten_db, 1.0) << "Expected RAM attenuation > 1.0 dB on double-bounce retroreflection";
}

// ============================================================================
// GOLDEN BENCHMARK 1: NASA Almond Standard Target (Woo et al., IEEE AP-S 1993)
// ============================================================================

TEST(GoldenBenchmarks, NasaAlmondGeometryConformity) {
    const std::string almond_path = "examples/nasa_almond/nasa_almond.obj";
    std::ifstream f(almond_path);
    if (!f.good()) {
        GTEST_SKIP() << "nasa_almond.obj not generated yet; run generate_almond.py";
    }

    const auto rc = strikecem::load_config("examples/nasa_almond/almond_1_19ghz.json", SCEM_SCHEMA_PATH);
    const auto mesh = strikecem::load_normalized_mesh(rc.value, "examples/nasa_almond", rc.schema_version);

    // Published dimensions: Length d = 0.2524m, Width = 0.0975m, Height = 0.0325m
    const double length = mesh.report.bbox_max.x - mesh.report.bbox_min.x;
    const double width = mesh.report.bbox_max.y - mesh.report.bbox_min.y;
    const double height = mesh.report.bbox_max.z - mesh.report.bbox_min.z;

    EXPECT_NEAR(length, 0.25237, 1e-3);
    EXPECT_NEAR(width, 0.09752, 1e-3);
    EXPECT_NEAR(height, 0.03251, 1e-3);

    // Watertight manifold
    const auto edges = strikecem::extract_edges(mesh);
    size_t boundary_edges = 0;
    for (const auto& e : edges.edges) {
        if (e.boundary) ++boundary_edges;
    }
    EXPECT_EQ(boundary_edges, 0u) << "Almond mesh should be a closed watertight manifold";
}

TEST(GoldenBenchmarks, NasaAlmondBroadsideSpecular) {
    const std::string almond_path = "examples/nasa_almond/nasa_almond.obj";
    std::ifstream f(almond_path);
    if (!f.good()) {
        GTEST_SKIP() << "nasa_almond.obj not generated yet; run generate_almond.py";
    }

    const auto rc = strikecem::load_config("examples/nasa_almond/almond_1_19ghz.json", SCEM_SCHEMA_PATH);
    const auto mesh = strikecem::load_normalized_mesh(rc.value, "examples/nasa_almond", rc.schema_version);
    const auto edge_model = strikecem::extract_edges(mesh);
    const strikecem::FringeOptions fringe{true, &edge_model, "ptd_full"};
    const strikecem::GoOptions go{true, 1};

    // 1.19 GHz Broadside (IEEE MoM ground truth is -10.5 dBsm)
    const auto plan_119 = make_single_plan(1.19e9, 90.0, 0.0, "HH");
    const auto res_119 = strikecem::solve_po(mesh, plan_119, rc.value, fringe, go);
    EXPECT_NEAR(to_dbsm(res_119.samples[0].rcs_sqm), -10.5, 3.5);

    // 9.92 GHz Broadside (IEEE MoM ground truth is -21.0 dBsm)
    const auto plan_992 = make_single_plan(9.92e9, 90.0, 0.0, "HH");
    const auto res_992 = strikecem::solve_po(mesh, plan_992, rc.value, fringe, go);
    EXPECT_NEAR(to_dbsm(res_992.samples[0].rcs_sqm), -21.0, 3.5);
}

TEST(GoldenBenchmarks, NasaAlmondPtdCurvatureEnhancement) {
    const std::string almond_path = "examples/nasa_almond/nasa_almond.obj";
    std::ifstream f(almond_path);
    if (!f.good()) {
        GTEST_SKIP() << "nasa_almond.obj not generated yet; run generate_almond.py";
    }

    auto rc = strikecem::load_config("examples/nasa_almond/almond_1_19ghz.json", SCEM_SCHEMA_PATH);
    const auto mesh = strikecem::load_normalized_mesh(rc.value, "examples/nasa_almond", rc.schema_version);
    const auto edge_model = strikecem::extract_edges(mesh);
    const strikecem::FringeOptions fringe{true, &edge_model, "ptd_full"};
    const strikecem::GoOptions go{true, 1};

    // Pure PO
    rc.value["solver"]["po_options"]["curvature_correction"] = false;
    rc.value["solver"]["po_options"]["edge_correction"] = "none";
    const auto plan = make_single_plan(1.19e9, 90.0, 0.0, "HH");
    const auto res_po = strikecem::solve_po(mesh, plan, rc.value, {}, go);

    // Full PO + Curvature + PTD
    rc.value["solver"]["po_options"]["curvature_correction"] = true;
    rc.value["solver"]["po_options"]["edge_correction"] = "ptd_full";
    const auto res_full = strikecem::solve_po(mesh, plan, rc.value, fringe, go);

    const double enhancement = to_dbsm(res_full.samples[0].rcs_sqm) - to_dbsm(res_po.samples[0].rcs_sqm);
    EXPECT_GT(enhancement, 5.0) << "Curvature and PTD should lift smooth body return by > 5 dB";
}

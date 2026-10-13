#include <gtest/gtest.h>
#include <cmath>

#include "strikecem/analysis/VehiclePreScan.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"

using namespace strikecem;

namespace {

std::string fixture(const std::string& name) {
    return std::string(SCEM_FIXTURE_DIR) + "/" + name;
}

NormalizedMesh load_mesh(const std::string& name) {
    auto rc = load_config(fixture(name), SCEM_SCHEMA_PATH);
    return load_normalized_mesh(rc.value, SCEM_FIXTURE_DIR, rc.schema_version);
}

NormalizedMesh make_flat_plate() {
    NormalizedMesh m;
    m.vertices = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0},
        {0.0, 1.0, 0.0}
    };
    m.triangles = {
        {0, 1, 2},
        {0, 2, 3}
    };
    m.normals = {
        {0.0, 0.0, 1.0},
        {0.0, 0.0, 1.0}
    };
    m.areas = {0.5, 0.5};
    m.triangle_tags = {"pec", "pec"};
    m.report.vertex_count = 4;
    m.report.triangle_count = 2;
    m.report.total_area_m2 = 1.0;
    m.report.max_edge_length_m = std::sqrt(2.0);
    m.report.bbox_min = {0.0, 0.0, 0.0};
    m.report.bbox_max = {1.0, 1.0, 0.0};
    return m;
}

} // namespace

TEST(VehiclePreScan, FlatPlatePurelyPlanar) {
    NormalizedMesh plate = make_flat_plate();
    VehiclePreScanReport report = VehiclePreScan::analyze(plate);

    EXPECT_EQ(report.total_facets, 2u);
    EXPECT_EQ(report.planar_facets, 2u);
    EXPECT_EQ(report.curved_facets, 0u);
    EXPECT_EQ(report.edge_facets, 0u);
    EXPECT_EQ(report.cavity_facets, 0u);
    EXPECT_DOUBLE_EQ(report.planar_percent, 100.0);

    // Flat plate recommendations
    EXPECT_EQ(report.recommended_edge_correction, "none");
    EXPECT_FALSE(report.recommended_shadowing);
    EXPECT_EQ(report.recommended_max_bounces, 1);
    EXPECT_FALSE(report.recommended_curvature_correction);
}

TEST(VehiclePreScan, CubeDetectsSharpEdges) {
    NormalizedMesh cube = load_mesh("valid_cube.json");
    VehiclePreScanReport report = VehiclePreScan::analyze(cube);

    EXPECT_EQ(report.total_facets, 12u);
    EXPECT_GT(report.edge_facets, 0u);
    EXPECT_EQ(report.recommended_edge_correction, "ptd_full");
    EXPECT_TRUE(report.recommended_shadowing);
}

TEST(VehiclePreScan, DihedralDetectsCavityInterior) {
    NormalizedMesh dihedral = load_mesh("valid_dihedral.json");
    VehiclePreScanReport report = VehiclePreScan::analyze(dihedral);

    EXPECT_GT(report.total_facets, 0u);
    EXPECT_GT(report.cavity_facets, 0u);
    EXPECT_GT(report.estimated_cavity_volume_m3, 0.0);
    EXPECT_GE(report.recommended_max_bounces, 2);
    EXPECT_TRUE(report.recommended_shadowing);
}

TEST(VehiclePreScan, ReportSerialization) {
    NormalizedMesh plate = make_flat_plate();
    VehiclePreScanReport report = VehiclePreScan::analyze(plate);

    nlohmann::json j = report.to_json();
    EXPECT_TRUE(j.contains("summary"));
    EXPECT_TRUE(j.contains("recommendations"));
    EXPECT_EQ(j["summary"]["total_facets"], 2);
    EXPECT_EQ(j["recommendations"]["recommended_edge_correction"], "none");

    std::string text = report.to_text_summary();
    EXPECT_NE(text.find("VEHICLE PRE-SCAN FEATURE ANALYSIS REPORT"), std::string::npos);
    EXPECT_NE(text.find("RECOMMENDED HYBRID SOLVER TUNING"), std::string::npos);
}

TEST(VehiclePreScan, AutoTuneConfig) {
    NormalizedMesh dihedral = load_mesh("valid_dihedral.json");
    VehiclePreScanReport report = VehiclePreScan::analyze(dihedral);

    nlohmann::json base_cfg = {
        {"model", {{"path", "tests/fixtures/dihedral.obj"}}},
        {"solver", {
            {"type", "po"},
            {"edge_correction", "none"}
        }}
    };

    nlohmann::json tuned_cfg = VehiclePreScan::auto_tune_config(base_cfg, report);

    EXPECT_EQ(tuned_cfg["solver"]["edge_correction"], report.recommended_edge_correction);
    EXPECT_EQ(tuned_cfg["solver"]["shadowing"], report.recommended_shadowing);
    EXPECT_EQ(tuned_cfg["solver"]["sbr"]["max_bounces"], report.recommended_max_bounces);
    EXPECT_EQ(tuned_cfg["solver"]["po_options"]["curvature_correction"],
              report.recommended_curvature_correction);
}

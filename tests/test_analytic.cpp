#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>

#include "strikecem/core/Config.hpp"
#include "strikecem/geometry/AnalyticGeometry.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace fs = std::filesystem;

namespace {

TEST(AnalyticGeometry, SphereAreaAndNormals) {
    strikecem::AnalyticSphere s;
    s.name = "test_sphere";
    s.material_tag = "pec_sphere";
    s.radius = 1.0;
    s.center = {0.0, 0.0, 0.0};

    strikecem::PrimitiveTessellationParams params;
    params.max_edge_length = 0.2;

    const auto mesh = strikecem::tessellate_sphere(s, params);
    EXPECT_GT(mesh.vertices.size(), 50u);
    EXPECT_GT(mesh.triangles.size(), 100u);

    // Analytic sphere surface area = 4*pi*R^2 ~ 12.56637 m^2
    const double expected_area = 4.0 * std::numbers::pi * s.radius * s.radius;
    EXPECT_NEAR(mesh.report.total_area_m2, expected_area, 0.05 * expected_area); // within 5%

    // All normals should point outward from center
    for (size_t i = 0; i < mesh.triangles.size(); ++i) {
        const auto& tri = mesh.triangles[i];
        const auto& centroid = (mesh.vertices[tri[0]] + mesh.vertices[tri[1]] + mesh.vertices[tri[2]]) * (1.0 / 3.0);
        const auto outward = (centroid - s.center).normalized();
        const double dot = outward.x * mesh.normals[i].x + outward.y * mesh.normals[i].y + outward.z * mesh.normals[i].z;
        EXPECT_GT(dot, 0.9); // outward normal alignment
        EXPECT_EQ(mesh.triangle_tags[i], "pec_sphere");
    }
}

TEST(AnalyticGeometry, CylinderDimensionsAndCaps) {
    strikecem::AnalyticCylinder c;
    c.name = "fuselage_body";
    c.material_tag = "aluminum";
    c.radius = 0.5;
    c.length = 2.0;
    c.origin = {0.0, 0.0, 0.0};
    c.axis = {1.0, 0.0, 0.0};
    c.cap_start = true;
    c.cap_end = true;

    strikecem::PrimitiveTessellationParams params;
    params.max_edge_length = 0.15;

    const auto mesh = strikecem::tessellate_cylinder(c, params);
    EXPECT_GT(mesh.triangles.size(), 100u);

    // Total area = lateral (2*pi*R*L) + 2 caps (2 * pi*R^2)
    // = 2*pi*0.5*2.0 + 2*pi*(0.25) = 2*pi + 0.5*pi = 2.5*pi ~ 7.85398 m^2
    const double expected_area = 2.5 * std::numbers::pi;
    EXPECT_NEAR(mesh.report.total_area_m2, expected_area, 0.05 * expected_area);

    EXPECT_NEAR(mesh.report.bbox_min.x, 0.0, 1e-4);
    EXPECT_NEAR(mesh.report.bbox_max.x, 2.0, 1e-4);
    EXPECT_NEAR(mesh.report.bbox_min.y, -0.5, 1e-2);
    EXPECT_NEAR(mesh.report.bbox_max.y, 0.5, 1e-2);
}

TEST(AnalyticGeometry, TangentOgiveProfile) {
    strikecem::AnalyticTangentOgive o;
    o.name = "radome_ogive";
    o.material_tag = "radome_composite";
    o.base_radius = 0.4;
    o.length = 1.2;
    o.origin = {0.0, 0.0, 0.0};
    o.axis = {1.0, 0.0, 0.0};
    o.base_cap = true;

    strikecem::PrimitiveTessellationParams params;
    params.max_edge_length = 0.1;

    const auto mesh = strikecem::tessellate_tangent_ogive(o, params);
    EXPECT_GT(mesh.triangles.size(), 100u);

    // Nose tip should be at x = 0
    EXPECT_NEAR(mesh.report.bbox_min.x, 0.0, 1e-4);
    EXPECT_NEAR(mesh.report.bbox_max.x, 1.2, 1e-4);

    // Base cap radius should reach ~ 0.4
    EXPECT_NEAR(mesh.report.bbox_max.y, 0.4, 0.05);

    for (const auto& tag : mesh.triangle_tags) {
        EXPECT_EQ(tag, "radome_composite");
    }
}

TEST(AnalyticGeometry, FinPlanarAndThick) {
    strikecem::AnalyticFin fin;
    fin.name = "tail_fin";
    fin.material_tag = "titanium";
    fin.root_chord = 0.6;
    fin.tip_chord = 0.3;
    fin.span = 0.5;
    fin.sweep_angle_deg = 25.0;
    fin.thickness = 0.02; // 2 cm thick

    const auto mesh = strikecem::tessellate_fin(fin);
    EXPECT_EQ(mesh.triangles.size(), 12u); // 2 top + 2 bottom + 8 edges
    EXPECT_GT(mesh.report.total_area_m2, 0.0);

    for (const auto& tag : mesh.triangle_tags) {
        EXPECT_EQ(tag, "titanium");
    }
}

TEST(AnalyticGeometry, AerospaceMissileAssemblyMerge) {
    nlohmann::json assembly = {
        {"primitives", {
            {{"type", "ogive"}, {"name", "nose"}, {"material_tag", "absorber"},
             {"base_radius", 0.3}, {"length", 0.9}, {"origin", {0.0, 0.0, 0.0}}, {"axis", {1.0, 0.0, 0.0}}, {"base_cap", false}},
            {{"type", "cylinder"}, {"name", "body"}, {"material_tag", "pec"},
             {"radius", 0.3}, {"length", 2.0}, {"origin", {0.9, 0.0, 0.0}}, {"axis", {1.0, 0.0, 0.0}}, {"cap_start", false}, {"cap_end", true}},
            {{"type", "fin"}, {"name", "fin_top"}, {"material_tag", "pec"},
             {"root_chord", 0.5}, {"tip_chord", 0.25}, {"span", 0.4}, {"sweep_angle_deg", 30.0}, {"thickness", 0.01},
             {"origin", {2.0, 0.3, 0.0}}, {"chord_dir", {1.0, 0.0, 0.0}}, {"span_dir", {0.0, 1.0, 0.0}}}
        }}
    };

    const auto mesh = strikecem::tessellate_aerospace_assembly(assembly, 0.1);
    EXPECT_EQ(mesh.groups.size(), 3u);
    EXPECT_GT(mesh.triangles.size(), 100u);

    // Verify tag counts
    size_t absorber_count = 0;
    size_t pec_count = 0;
    for (const auto& tag : mesh.triangle_tags) {
        if (tag == "absorber") ++absorber_count;
        if (tag == "pec") ++pec_count;
    }
    EXPECT_GT(absorber_count, 0u);
    EXPECT_GT(pec_count, 0u);
    EXPECT_EQ(absorber_count + pec_count, mesh.triangles.size());
}

TEST(AnalyticGeometry, EndToEndProceduralModelSolve) {
    const fs::path dir = fs::temp_directory_path() / ("scem_proc_" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    const fs::path cfg_path = dir / "proc_cfg.json";
    const fs::path out_path = dir / "proc_out.csv";

    nlohmann::json cfg = {
        {"scem_schema_version", "1.0"},
        {"model", {
            {"primitives", {
                {{"type", "sphere"}, {"name", "nose_radome"}, {"material_tag", "pec"},
                 {"radius", 1.0}, {"center", {0.0, 0.0, 0.0}}}
            }},
            {"max_edge_length", 0.05}
        }},
        {"frequency", {{"frequency_hz", 1e9}}},
        {"angles", {
            {"azimuth", {{"start", 0}, {"stop", 0}, {"step", 1}}},
            {"elevation", {{"start", 0}, {"stop", 0}, {"step", 1}}}
        }},
        {"polarization", {"HH"}},
        {"solver", {{"type", "PO"}}},
        {"output", {{"path", out_path.string()}, {"format", "csv"}}}
    };

    {
        std::ofstream out(cfg_path);
        out << cfg.dump(2);
    }

    const auto rc = strikecem::load_config(cfg_path.string(), SCEM_SCHEMA_PATH);
    const auto plan = strikecem::build_sample_plan(rc.value);
    auto mesh = strikecem::load_normalized_mesh(rc.value, dir, rc.schema_version);

    EXPECT_GT(mesh.triangles.size(), 100u);
    EXPECT_NEAR(mesh.report.total_area_m2, 4.0 * std::numbers::pi, 0.5); // 4*pi ~ 12.56

    const auto res = strikecem::solve_po(mesh, plan, rc.value);
    EXPECT_EQ(res.samples.size(), 1u);
    // Sphere optical PO asymptote: pi * R^2 = pi ~ 3.14159 m^2
    EXPECT_NEAR(res.samples[0].rcs_sqm, std::numbers::pi, 0.35);

    fs::remove_all(dir, ec);
}

} // namespace

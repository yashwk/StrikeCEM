#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/math/Geometry.hpp"

namespace strikecem {

struct PrimitiveTessellationParams {
    double max_edge_length = 0.05; // metres
    size_t min_radial_segments = 16;
    size_t min_axial_segments = 8;
};

struct AnalyticSphere {
    std::string name = "sphere";
    std::string material_tag = "pec";
    double radius = 1.0;
    geom::Vec3d center = {0.0, 0.0, 0.0};
};

struct AnalyticCylinder {
    std::string name = "fuselage";
    std::string material_tag = "pec";
    double radius = 0.5;
    double length = 2.0;
    geom::Vec3d origin = {0.0, 0.0, 0.0}; // Base center
    geom::Vec3d axis = {1.0, 0.0, 0.0};   // Axis vector
    bool cap_start = true;
    bool cap_end = true;
};

struct AnalyticTangentOgive {
    std::string name = "nose_cone";
    std::string material_tag = "pec";
    double base_radius = 0.5;
    double length = 1.5;
    geom::Vec3d origin = {0.0, 0.0, 0.0}; // Nose tip
    geom::Vec3d axis = {1.0, 0.0, 0.0};   // Axis from tip towards base
    bool base_cap = true;
};

struct AnalyticFin {
    std::string name = "fin";
    std::string material_tag = "pec";
    double root_chord = 0.8;
    double tip_chord = 0.4;
    double span = 0.6;
    double sweep_angle_deg = 30.0;
    double thickness = 0.02;
    geom::Vec3d origin = {0.0, 0.0, 0.0};   // Leading edge root
    geom::Vec3d chord_dir = {1.0, 0.0, 0.0}; // Chordwise direction
    geom::Vec3d span_dir = {0.0, 1.0, 0.0};  // Spanwise direction
};

// Generates NormalizedMesh for individual primitives
NormalizedMesh tessellate_sphere(
    const AnalyticSphere& sphere,
    const PrimitiveTessellationParams& params = {});

NormalizedMesh tessellate_cylinder(
    const AnalyticCylinder& cylinder,
    const PrimitiveTessellationParams& params = {});

NormalizedMesh tessellate_tangent_ogive(
    const AnalyticTangentOgive& ogive,
    const PrimitiveTessellationParams& params = {});

NormalizedMesh tessellate_fin(
    const AnalyticFin& fin,
    const PrimitiveTessellationParams& params = {});

// Merges multiple meshes into a single NormalizedMesh preserving group and tag mappings
NormalizedMesh merge_meshes(const std::vector<NormalizedMesh>& meshes);

// Tessellates a complete aerospace assembly from JSON config
NormalizedMesh tessellate_aerospace_assembly(
    const nlohmann::json& primitives_json,
    double max_edge_length = 0.05);

} // namespace strikecem

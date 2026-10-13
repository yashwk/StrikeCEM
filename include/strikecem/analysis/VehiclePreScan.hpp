#pragma once
// Automated vehicle feature pre-scan and hybrid solver configuration engine (ADR-0030).
// Automatically analyzes geometry to detect cavities/ducts, sharp leading edges,
// curved lifting surfaces, and planar specular glints, recommending optimal hybrid solver settings.
#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/math/Geometry.hpp"

namespace strikecem {

enum class VehicleFeatureType {
    PlanarSpecular = 0, // Flat surface; specular flash handled by standard PO
    SmoothCurved = 1,   // Non-zero curvature; benefited by Fock curvature correction
    SharpEdgeWedge = 2, // Sharp wedge (dihedral > 30 deg); benefited by PTD / Michaeli EEC
    CavityInterior = 3  // Concave / re-entrant pocket; requires SBR multi-bounce tracking
};

struct FacetClassification {
    size_t facet_index = 0;
    VehicleFeatureType type = VehicleFeatureType::PlanarSpecular;
    double mean_curvature = 0.0;
    double min_dihedral_deg = 0.0;
    bool is_cavity = false;
};

struct VehiclePreScanReport {
    size_t total_facets = 0;
    size_t planar_facets = 0;
    size_t curved_facets = 0;
    size_t edge_facets = 0;
    size_t cavity_facets = 0;

    double planar_percent = 0.0;
    double curved_percent = 0.0;
    double edge_percent = 0.0;
    double cavity_percent = 0.0;

    geom::Vec3d bbox_dimensions_m{0, 0, 0};
    double surface_area_m2 = 0.0;
    double estimated_cavity_volume_m3 = 0.0;

    // Recommended solver tuning
    std::string recommended_edge_correction = "none"; // "none" or "ptd_full"
    bool recommended_shadowing = false;
    int recommended_max_bounces = 1;
    bool recommended_curvature_correction = false;

    std::vector<FacetClassification> classifications;

    nlohmann::json to_json() const;
    std::string to_text_summary() const;
};

class VehiclePreScan {
public:
    static VehiclePreScanReport analyze(const NormalizedMesh& mesh);

    // Applies pre-scan recommendations to an existing configuration JSON
    static nlohmann::json auto_tune_config(const nlohmann::json& input_config,
                                           const VehiclePreScanReport& report);
};

} // namespace strikecem

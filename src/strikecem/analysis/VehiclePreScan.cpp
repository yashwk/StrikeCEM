#include "strikecem/analysis/VehiclePreScan.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>

#include "strikecem/solvers/EdgeModel.hpp"

namespace strikecem {

namespace {

struct EdgeKey {
    size_t v0 = 0;
    size_t v1 = 0;
    bool operator<(const EdgeKey& o) const {
        if (v0 != o.v0) return v0 < o.v0;
        return v1 < o.v1;
    }
};

EdgeKey make_edge_key(size_t a, size_t b) {
    return (a < b) ? EdgeKey{a, b} : EdgeKey{b, a};
}

geom::Vec3d face_centroid(const NormalizedMesh& mesh, const std::array<uint32_t, 3>& tri) {
    const auto& v0 = mesh.vertices[tri[0]];
    const auto& v1 = mesh.vertices[tri[1]];
    const auto& v2 = mesh.vertices[tri[2]];
    return geom::Vec3d{
        (v0.x + v1.x + v2.x) / 3.0,
        (v0.y + v1.y + v2.y) / 3.0,
        (v0.z + v1.z + v2.z) / 3.0
    };
}

} // namespace

VehiclePreScanReport VehiclePreScan::analyze(const NormalizedMesh& mesh) {
    VehiclePreScanReport report;
    report.total_facets = mesh.triangles.size();
    report.surface_area_m2 = mesh.report.total_area_m2;

    const auto& bmin = mesh.report.bbox_min;
    const auto& bmax = mesh.report.bbox_max;
    report.bbox_dimensions_m = geom::Vec3d{
        std::max(0.0, bmax.x - bmin.x),
        std::max(0.0, bmax.y - bmin.y),
        std::max(0.0, bmax.z - bmin.z)
    };

    const double bbox_diag = std::sqrt(
        report.bbox_dimensions_m.x * report.bbox_dimensions_m.x +
        report.bbox_dimensions_m.y * report.bbox_dimensions_m.y +
        report.bbox_dimensions_m.z * report.bbox_dimensions_m.z);

    if (report.total_facets == 0) {
        return report;
    }

    // 1. Build edge adjacency graph
    std::map<EdgeKey, std::vector<size_t>> edge_to_faces;
    for (size_t f = 0; f < mesh.triangles.size(); ++f) {
        const auto& t = mesh.triangles[f];
        edge_to_faces[make_edge_key(t[0], t[1])].push_back(f);
        edge_to_faces[make_edge_key(t[1], t[2])].push_back(f);
        edge_to_faces[make_edge_key(t[2], t[0])].push_back(f);
    }

    // 2. Compute face centroids
    std::vector<geom::Vec3d> centroids(mesh.triangles.size());
    for (size_t f = 0; f < mesh.triangles.size(); ++f) {
        centroids[f] = face_centroid(mesh, mesh.triangles[f]);
    }

    // 3. Extract edge concavity using EdgeModel
    std::vector<bool> is_cavity_facet(mesh.triangles.size(), false);
    EdgeModel edge_model = extract_edges(mesh, 0.05);
    for (const auto& e : edge_model.edges) {
        if (!e.boundary && !e.convex) {
            if (e.tri0 >= 0 && static_cast<size_t>(e.tri0) < is_cavity_facet.size()) {
                is_cavity_facet[e.tri0] = true;
            }
            if (e.tri1 >= 0 && static_cast<size_t>(e.tri1) < is_cavity_facet.size()) {
                is_cavity_facet[e.tri1] = true;
            }
        }
    }

    // 4. Mutual line-of-sight / opposing faces check
    for (size_t f1 = 0; f1 < mesh.triangles.size(); ++f1) {
        const auto& c1 = centroids[f1];
        const auto& n1 = mesh.normals[f1];

        for (size_t f2 = f1 + 1; f2 < mesh.triangles.size(); ++f2) {
            const auto& c2 = centroids[f2];
            const auto& n2 = mesh.normals[f2];

            geom::Vec3d d12 = c2 - c1;
            double dist = d12.length();
            if (dist > 1e-5 && dist < bbox_diag * 0.9) {
                double r_dot_n1 = geom::dot(d12, n1);
                double r_dot_n2 = geom::dot(d12, n2);

                if (r_dot_n1 > 0.05 * dist && r_dot_n2 < -0.05 * dist) {
                    is_cavity_facet[f1] = true;
                    is_cavity_facet[f2] = true;
                }
            }
        }
    }

    // Propagate cavity classification across coplanar/smooth neighbors
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& [key, faces] : edge_to_faces) {
            if (faces.size() == 2) {
                size_t f0 = faces[0];
                size_t f1 = faces[1];
                if (is_cavity_facet[f0] != is_cavity_facet[f1]) {
                    double dot = geom::dot(mesh.normals[f0], mesh.normals[f1]);
                    if (dot > 0.98) { // nearly coplanar
                        is_cavity_facet[f0] = true;
                        is_cavity_facet[f1] = true;
                        changed = true;
                    }
                }
            }
        }
    }

    // 5. Classify each facet
    report.classifications.resize(mesh.triangles.size());

    for (size_t f = 0; f < mesh.triangles.size(); ++f) {
        FacetClassification fc;
        fc.facet_index = f;
        const auto& tri = mesh.triangles[f];

        // Examine adjacent faces to detect dihedral angles and curvature
        double max_dihedral_deg = 0.0;
        double curvature_est = 0.0;

        auto check_edge = [&](size_t v_a, size_t v_b) {
            auto it = edge_to_faces.find(make_edge_key(v_a, v_b));
            if (it != edge_to_faces.end()) {
                for (size_t other_f : it->second) {
                    if (other_f != f) {
                        const auto& n1 = mesh.normals[f];
                        const auto& n2 = mesh.normals[other_f];
                        double dot = geom::dot(n1, n2);
                        dot = std::max(-1.0, std::min(1.0, dot));
                        double angle_deg = std::acos(dot) * (180.0 / M_PI);
                        max_dihedral_deg = std::max(max_dihedral_deg, angle_deg);

                        double dx = (centroids[f] - centroids[other_f]).length();
                        if (dx > 1e-6) {
                            curvature_est = std::max(curvature_est, (angle_deg * M_PI / 180.0) / dx);
                        }
                    }
                }
            }
        };

        check_edge(tri[0], tri[1]);
        check_edge(tri[1], tri[2]);
        check_edge(tri[2], tri[0]);

        fc.min_dihedral_deg = max_dihedral_deg;
        fc.mean_curvature = curvature_est;
        fc.is_cavity = is_cavity_facet[f];

        // Hierarchy of assignment
        if (fc.is_cavity) {
            fc.type = VehicleFeatureType::CavityInterior;
            report.cavity_facets++;
        } else if (max_dihedral_deg > 30.0) {
            fc.type = VehicleFeatureType::SharpEdgeWedge;
            report.edge_facets++;
        } else if (max_dihedral_deg > 2.0 && curvature_est > 0.05) {
            fc.type = VehicleFeatureType::SmoothCurved;
            report.curved_facets++;
        } else {
            fc.type = VehicleFeatureType::PlanarSpecular;
            report.planar_facets++;
        }

        report.classifications[f] = fc;
    }

    report.planar_percent = 100.0 * report.planar_facets / report.total_facets;
    report.curved_percent = 100.0 * report.curved_facets / report.total_facets;
    report.edge_percent = 100.0 * report.edge_facets / report.total_facets;
    report.cavity_percent = 100.0 * report.cavity_facets / report.total_facets;

    double bbox_vol = std::max(1e-6, report.bbox_dimensions_m.x *
                                      report.bbox_dimensions_m.y *
                                      report.bbox_dimensions_m.z);
    report.estimated_cavity_volume_m3 = (report.cavity_facets > 0)
        ? (report.cavity_percent / 100.0) * bbox_vol
        : 0.0;

    // Recommendations
    report.recommended_edge_correction = (report.edge_facets > 0) ? "ptd_full" : "none";
    report.recommended_shadowing = (report.cavity_facets > 0 || report.curved_facets > 0 || report.edge_facets > 0);
    report.recommended_max_bounces = (report.cavity_facets > 0) ? 3 : 1;
    report.recommended_curvature_correction = (report.curved_facets > 0);

    return report;
}

nlohmann::json VehiclePreScanReport::to_json() const {
    return {
        {"summary", {
            {"total_facets", total_facets},
            {"surface_area_m2", surface_area_m2},
            {"planar_facets", planar_facets},
            {"curved_facets", curved_facets},
            {"edge_facets", edge_facets},
            {"cavity_facets", cavity_facets},
            {"estimated_cavity_volume_m3", estimated_cavity_volume_m3},
            {"bounding_box_m", {
                {"length_x", bbox_dimensions_m.x},
                {"width_y", bbox_dimensions_m.y},
                {"height_z", bbox_dimensions_m.z}
            }}
        }},
        {"feature_breakdown", {
            {"planar_percent", planar_percent},
            {"curved_percent", curved_percent},
            {"edge_percent", edge_percent},
            {"cavity_percent", cavity_percent}
        }},
        {"recommendations", {
            {"recommended_solver", (cavity_facets > 0) ? "SBR" : "PO"},
            {"recommended_edge_correction", recommended_edge_correction},
            {"recommended_shadowing", recommended_shadowing},
            {"recommended_max_bounces", recommended_max_bounces},
            {"recommended_curvature_correction", recommended_curvature_correction}
        }}
    };
}

std::string VehiclePreScanReport::to_text_summary() const {
    std::ostringstream ss;
    ss << "=== VEHICLE PRE-SCAN FEATURE ANALYSIS REPORT ===\n"
       << "Total Facets: " << total_facets << "\n"
       << "Surface Area: " << std::fixed << std::setprecision(4) << surface_area_m2 << " m^2\n"
       << "Bounding Box: " << bbox_dimensions_m.x << " x " << bbox_dimensions_m.y
       << " x " << bbox_dimensions_m.z << " m\n"
       << "Estimated Cavity Volume: " << estimated_cavity_volume_m3 << " m^3\n\n"
       << "Feature Breakdown:\n"
       << "  - Planar Specular Surfaces: " << planar_facets << " (" << planar_percent << "%)\n"
       << "  - Smooth Curved Surfaces:   " << curved_facets << " (" << curved_percent << "%)\n"
       << "  - Sharp Leading Edges/Wedges: " << edge_facets << " (" << edge_percent << "%)\n"
       << "  - Cavity / Duct Interiors:   " << cavity_facets << " (" << cavity_percent << "%)\n\n"
       << "--- RECOMMENDED HYBRID SOLVER TUNING ---\n"
       << "  - Solver Type:             " << ((cavity_facets > 0) ? "SBR (Shooting & Bouncing Rays)" : "PO (Physical Optics)") << "\n"
       << "  - Edge Correction Mode:    " << recommended_edge_correction << "\n"
       << "  - Shadowing Enabled:       " << (recommended_shadowing ? "true" : "false") << "\n"
       << "  - Multi-Bounce Limit:      " << recommended_max_bounces << "\n"
       << "  - Fock Curvature Factor:   " << (recommended_curvature_correction ? "true" : "false") << "\n";
    return ss.str();
}

nlohmann::json VehiclePreScan::auto_tune_config(const nlohmann::json& input_config,
                                                const VehiclePreScanReport& report) {
    nlohmann::json tuned = input_config;

    if (!tuned.contains("solver") || !tuned["solver"].is_object()) {
        tuned["solver"] = nlohmann::json::object();
    }
    if (!tuned["solver"].contains("po_options") || !tuned["solver"]["po_options"].is_object()) {
        tuned["solver"]["po_options"] = nlohmann::json::object();
    }

    tuned["solver"]["type"] = (report.cavity_facets > 0) ? "SBR" : "PO";
    tuned["solver"]["edge_correction"] = report.recommended_edge_correction;
    tuned["solver"]["shadowing"] = report.recommended_shadowing;

    tuned["solver"]["po_options"]["edge_correction"] = report.recommended_edge_correction;
    tuned["solver"]["po_options"]["shadowing"] = report.recommended_shadowing;
    tuned["solver"]["po_options"]["curvature_correction"] = report.recommended_curvature_correction;
    tuned["solver"]["po_options"]["max_bounces"] = report.recommended_max_bounces;

    if (report.cavity_facets > 0) {
        tuned["solver"]["sbr"] = {
            {"max_bounces", report.recommended_max_bounces}
        };
    }

    return tuned;
}

} // namespace strikecem

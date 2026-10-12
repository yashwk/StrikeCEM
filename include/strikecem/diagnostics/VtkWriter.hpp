#pragma once
// Diagnostic & Field Visualization Export (VTK .vtu XML format).
// Implements ADR-0008 and FULL_IMPLEMENTATION.md §12.2.
#include <cstdint>
#include <string>
#include <vector>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"
#include "strikecem/solvers/Ray.hpp"

namespace strikecem {

struct VtkFacetDiagnostic {
    uint8_t lit = 1;
    double current_magnitude = 0.0;
    double current_phase_deg = 0.0;
    geom::Vec3d current_real;
    geom::Vec3d current_imag;
    uint32_t material_id = 0;
    double area_m2 = 0.0;
    geom::Vec3d normal;
};

struct VtkMeshDiagnostics {
    std::vector<VtkFacetDiagnostic> facets;
};

// Computes per-facet surface current and illumination state for an observation direction and frequency.
VtkMeshDiagnostics compute_facet_diagnostics(
    const NormalizedMesh& mesh,
    const Direction& direction,
    double frequency_hz,
    const std::string& polarization = "HH",
    const GoOptions& go = GoOptions{},
    const MaterialOptions& material = MaterialOptions{});

// Writes the diagnostic mesh to a standard XML VTK UnstructuredGrid (.vtu) file.
void write_vtu(
    const std::string& vtu_path,
    const NormalizedMesh& mesh,
    const VtkMeshDiagnostics& diag);

} // namespace strikecem

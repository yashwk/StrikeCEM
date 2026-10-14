#include "strikecem/diagnostics/VtkWriter.hpp"

#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <stdexcept>

#include "strikecem/core/Conventions.hpp"
#include "strikecem/io/OutputError.hpp"
#include "strikecem/solvers/Material.hpp"

namespace strikecem {

namespace fs = std::filesystem;

VtkMeshDiagnostics compute_facet_diagnostics(
    const NormalizedMesh& mesh,
    const Direction& direction,
    double frequency_hz,
    const std::string& polarization,
    const GoOptions& go,
    const MaterialOptions& material) {
    const size_t n_facets = mesh.triangles.size();
    VtkMeshDiagnostics diag;
    diag.facets.resize(n_facets);

    const double k = (2.0 * std::numbers::pi * frequency_hz) / kSpeedOfLight;
    const geom::Vec3d k_hat = direction.k_hat;
    const auto [tx_h, tx_v] = polarization_basis(k_hat);
    const geom::Vec3d e_inc_dir = (polarization == "VV" || polarization == "VH") ? tx_v : tx_h;

    Bvh bvh;
    if (go.shadowing) {
        bvh = build_bvh(mesh);
    }

    for (size_t t = 0; t < n_facets; ++t) {
        auto& fd = diag.facets[t];
        const geom::Vec3d& n = mesh.normals[t];
        fd.normal = n;
        fd.area_m2 = mesh.areas[t];

        if (t < material.facet_material_indices.size()) {
            fd.material_id = material.facet_material_indices[t];
        }

        const auto& tri = mesh.triangles[t];
        const geom::Vec3d centroid = (mesh.vertices[tri[0]] + mesh.vertices[tri[1]] + mesh.vertices[tri[2]]) * (1.0 / 3.0);

        // Illumination condition: dot(n, -k_hat) > 0 => dot(n, k_hat) < 0
        const double cos_inc = -dot(k_hat, n);
        if (cos_inc <= 1e-9) {
            fd.lit = 0;
            continue;
        }

        if (go.shadowing) {
            const geom::Vec3d ray_origin = centroid + n * 1e-6;
            const geom::Vec3d dir_to_source = -k_hat;
            const double max_dist = (mesh.report.bbox_max - mesh.report.bbox_min).length() * 2.0;
            if (occluded_bvh(bvh, mesh, ray_origin, dir_to_source, max_dist > 0.0 ? max_dist : 1e3, static_cast<uint32_t>(t))) {
                fd.lit = 0;
                continue;
            }
        }

        fd.lit = 1;

        // Incident phase at centroid: e^{-j k (k_hat . r_c)}
        const double phase_rad = -k * dot(k_hat, centroid);
        const std::complex<double> phase_factor(std::cos(phase_rad), std::sin(phase_rad));

        // H_inc = (k_hat x E_inc) / eta0
        const geom::Vec3d h_inc = cross(k_hat, e_inc_dir) * (1.0 / kEta0);

        // Surface current: J_po = (1 + Gamma) * (n x H_inc)
        geom::Vec3d j_vec = cross(n, h_inc) * 2.0;
        if (material.enabled && t < material.facet_material_indices.size()) {
            uint32_t midx = material.facet_material_indices[t];
            if (midx < material.models.size()) {
                const auto& model = material.models[midx];
                if (model.type == WallType::Coated) {
                    ComplexMedium incident;
                    double lambda_vac = kSpeedOfLight / frequency_hz;
                    auto cr = coated_pec_reflection(incident, model.layers, std::clamp(cos_inc, 0.05, 1.0), lambda_vac);
                    double refl = std::sqrt(0.5 * (std::norm(cr.r_te) + std::norm(cr.r_tm)));
                    j_vec = j_vec * refl;
                } else if (model.type == WallType::Impedance) {
                    auto ir = impedance_reflection(model.surface_impedance, std::clamp(cos_inc, 0.05, 1.0));
                    double refl = std::sqrt(0.5 * (std::norm(ir.r_te) + std::norm(ir.r_tm)));
                    j_vec = j_vec * refl;
                } else if (model.type == WallType::Dielectric) {
                    ComplexMedium incident;
                    ComplexMedium wmed = evaluate_material(model, frequency_hz);
                    auto fr = fresnel(incident, wmed, std::clamp(cos_inc, 0.05, 1.0));
                    double refl = std::sqrt(0.5 * (std::norm(fr.r_te) + std::norm(fr.r_tm)));
                    j_vec = j_vec * refl;
                }
            }
        }

        const std::complex<double> jx = j_vec.x * phase_factor;
        const std::complex<double> jy = j_vec.y * phase_factor;
        const std::complex<double> jz = j_vec.z * phase_factor;

        fd.current_real = {jx.real(), jy.real(), jz.real()};
        fd.current_imag = {jx.imag(), jy.imag(), jz.imag()};

        const double mag_sq = std::norm(jx) + std::norm(jy) + std::norm(jz);
        fd.current_magnitude = std::sqrt(mag_sq);

        double phase_deg = phase_rad * (180.0 / std::numbers::pi);
        while (phase_deg > 180.0) phase_deg -= 360.0;
        while (phase_deg < -180.0) phase_deg += 360.0;
        fd.current_phase_deg = phase_deg;
    }

    return diag;
}

void write_vtu(
    const std::string& vtu_path,
    const NormalizedMesh& mesh,
    const VtkMeshDiagnostics& diag) {
    const fs::path path(vtu_path);
    const fs::path dir = path.parent_path();
    if (!dir.empty()) {
        std::error_code ec;
        fs::create_directories(dir, ec);
    }

    std::ofstream out(vtu_path, std::ios::trunc);
    if (!out) {
        throw OutputError("cannot open VTU output file: " + vtu_path);
    }

    const size_t n_points = mesh.vertices.size();
    const size_t n_cells = mesh.triangles.size();

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << n_points << "\" NumberOfCells=\"" << n_cells << "\">\n";

    // Points
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    out << std::fixed << std::setprecision(8);
    for (size_t i = 0; i < n_points; ++i) {
        const auto& v = mesh.vertices[i];
        out << "          " << v.x << " " << v.y << " " << v.z << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    // Cells
    out << "      <Cells>\n";
    out << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (size_t t = 0; t < n_cells; ++t) {
        const auto& tri = mesh.triangles[t];
        out << "          " << tri[0] << " " << tri[1] << " " << tri[2] << "\n";
    }
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    for (size_t t = 0; t < n_cells; ++t) {
        out << "          " << ((t + 1) * 3) << "\n";
    }
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (size_t t = 0; t < n_cells; ++t) {
        out << "          5\n"; // VTK_TRIANGLE
    }
    out << "        </DataArray>\n";
    out << "      </Cells>\n";

    // CellData
    out << "      <CellData>\n";

    // LitStatus
    out << "        <DataArray type=\"UInt8\" Name=\"LitStatus\" format=\"ascii\">\n";
    for (size_t t = 0; t < n_cells; ++t) {
        out << "          " << static_cast<unsigned>(t < diag.facets.size() ? diag.facets[t].lit : 0) << "\n";
    }
    out << "        </DataArray>\n";

    // CurrentMagnitude
    out << "        <DataArray type=\"Float64\" Name=\"CurrentMagnitude\" format=\"ascii\">\n";
    for (size_t t = 0; t < n_cells; ++t) {
        out << "          " << (t < diag.facets.size() ? diag.facets[t].current_magnitude : 0.0) << "\n";
    }
    out << "        </DataArray>\n";

    // CurrentPhaseDeg
    out << "        <DataArray type=\"Float64\" Name=\"CurrentPhaseDeg\" format=\"ascii\">\n";
    for (size_t t = 0; t < n_cells; ++t) {
        out << "          " << (t < diag.facets.size() ? diag.facets[t].current_phase_deg : 0.0) << "\n";
    }
    out << "        </DataArray>\n";

    // SurfaceCurrentReal
    out << "        <DataArray type=\"Float64\" Name=\"SurfaceCurrentReal\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (size_t t = 0; t < n_cells; ++t) {
        const auto& cr = (t < diag.facets.size()) ? diag.facets[t].current_real : geom::Vec3d{};
        out << "          " << cr.x << " " << cr.y << " " << cr.z << "\n";
    }
    out << "        </DataArray>\n";

    // SurfaceCurrentImag
    out << "        <DataArray type=\"Float64\" Name=\"SurfaceCurrentImag\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (size_t t = 0; t < n_cells; ++t) {
        const auto& ci = (t < diag.facets.size()) ? diag.facets[t].current_imag : geom::Vec3d{};
        out << "          " << ci.x << " " << ci.y << " " << ci.z << "\n";
    }
    out << "        </DataArray>\n";

    // MaterialId
    out << "        <DataArray type=\"UInt32\" Name=\"MaterialId\" format=\"ascii\">\n";
    for (size_t t = 0; t < n_cells; ++t) {
        out << "          " << (t < diag.facets.size() ? diag.facets[t].material_id : 0) << "\n";
    }
    out << "        </DataArray>\n";

    // FacetArea
    out << "        <DataArray type=\"Float64\" Name=\"FacetArea\" format=\"ascii\">\n";
    for (size_t t = 0; t < n_cells; ++t) {
        out << "          " << (t < diag.facets.size() ? diag.facets[t].area_m2 : 0.0) << "\n";
    }
    out << "        </DataArray>\n";

    // FacetNormal
    out << "        <DataArray type=\"Float64\" Name=\"FacetNormal\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (size_t t = 0; t < n_cells; ++t) {
        const auto& norm = (t < diag.facets.size()) ? diag.facets[t].normal : geom::Vec3d{};
        out << "          " << norm.x << " " << norm.y << " " << norm.z << "\n";
    }
    out << "        </DataArray>\n";

    out << "      </CellData>\n";
    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";

    out.flush();
    if (!out) {
        throw OutputError("failed while writing VTU diagnostic file: " + vtu_path);
    }
}

} // namespace strikecem

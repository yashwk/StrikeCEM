// Deterministic CPU PO reference solver (ADR-0001, SPEC FR-6).
//
// Parallelism is over independent (frequency, direction) units writing
// disjoint, preassigned rows, so thread count cannot change numerics:
// deterministic mode needs no tolerance-based reproducibility gate.
#include "strikecem/solvers/PhysicalOptics.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include <mutex>
#include <numbers>
#include <string>
#include <thread>

#include "strikecem/solvers/Projection.hpp"
#include "strikecem/geometry/Symmetry.hpp"
#include "strikecem/solvers/Curvature.hpp"

#include "strikecem/solvers/GpuPO.hpp"

namespace strikecem {
namespace {

template <typename Real> struct Vec3 {
    Real x{0}, y{0}, z{0};
};
template <typename Real> Vec3<Real> cross(const Vec3<Real>& a, const Vec3<Real>& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
template <typename Real> Real dot(const Vec3<Real>& a, const Vec3<Real>& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// 3-vector over complex scalars for field arithmetic.
template <typename Real> using CVec3 = std::array<std::complex<Real>, 3>;
template <typename Real> CVec3<Real> cross_c(const CVec3<Real>& a, const CVec3<Real>& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
template <typename Real> CVec3<Real> to_complex(const Vec3<Real>& v) {
    return {v.x, v.y, v.z};
}

using D3 = std::array<std::complex<double>, 3>;

D3 cross_d(const geom::Vec3d& a, const D3& b) {
    return {a.y * b[2] - a.z * b[1], a.z * b[0] - a.x * b[2], a.x * b[1] - a.y * b[0]};
}

D3 material_facet_field(const geom::Vec3d& n, const geom::Vec3d& k_hat,
                        const geom::Vec3d& r_hat, const geom::Vec3d& e_te,
                        const geom::Vec3d& tm_inc, const geom::Vec3d& k_ref,
                        const geom::Vec3d& tm_ref, const D3& e_inc, double kd, double eta,
                        double area, const geom::Vec3d& centroid,
                        const std::complex<double>& r_te, const std::complex<double>& r_tm) {
    const std::complex<double> a_te = e_inc[0] * e_te.x + e_inc[1] * e_te.y + e_inc[2] * e_te.z;
    const std::complex<double> a_tm =
        e_inc[0] * tm_inc.x + e_inc[1] * tm_inc.y + e_inc[2] * tm_inc.z;
    const D3 e_ref = {r_te * a_te * e_te.x + r_tm * a_tm * tm_ref.x,
                      r_te * a_te * e_te.y + r_tm * a_tm * tm_ref.y,
                      r_te * a_te * e_te.z + r_tm * a_tm * tm_ref.z};
    D3 h_inc = cross_d(k_hat, e_inc);
    D3 h_ref = cross_d(k_ref, e_ref);
    for (int i = 0; i < 3; ++i) {
        h_inc[i] /= eta;
        h_ref[i] /= eta;
    }
    const D3 h_tot = {h_inc[0] + h_ref[0], h_inc[1] + h_ref[1], h_inc[2] + h_ref[2]};
    const D3 e_tot = {e_inc[0] + e_ref[0], e_inc[1] + e_ref[1], e_inc[2] + e_ref[2]};
    const D3 j = cross_d(n, h_tot);
    D3 m = cross_d(n, e_tot);
    for (int i = 0; i < 3; ++i) m[i] = -m[i];
    const D3 tj = cross_d(r_hat, cross_d(r_hat, j));
    const D3 tm = cross_d(r_hat, m);
    const std::complex<double> eout =
        std::exp(std::complex<double>(0.0, kd * geom::dot(r_hat, centroid)));
    const std::complex<double> amp =
        std::complex<double>(0.0, 1.0) * kd * area * eout / (4.0 * std::numbers::pi);
    D3 out;
    for (int i = 0; i < 3; ++i) out[i] = amp * (eta * tj[i] + tm[i]);
    return out;
}

template <typename Real>
PoResult solve_typed(const NormalizedMesh& mesh, const SamplePlan& plan,
                     const nlohmann::json& resolved, unsigned num_threads,
                     const std::vector<std::pair<uint32_t, uint32_t>>& units,
                     const FringeOptions& fringe, const GoOptions& go,
                     const MaterialOptions& material) {
    using C = std::complex<Real>;
    PoResult out;
    const auto& medium = resolved["frequency"]["medium"];
    const double eps_r = medium["epsilon_r"].get<double>();
    const double mu_r = medium["mu_r"].get<double>();
    if (medium["sigma"].get<double>() != 0.0)
        out.warnings.push_back("medium conductivity is ignored in v1 PO");
    if (go.shadowing)
        out.warnings.push_back(
            "GO shadowing enabled: hard shadow boundaries without edge diffraction; "
            "multi-bounce not included");
    bool has_multi_material = material.enabled && !material.facet_material_indices.empty();
    bool any_non_pec = false;
    if (material.enabled) {
        if (material.wall != WallType::Pec) any_non_pec = true;
        if (has_multi_material) {
            for (uint32_t idx : material.facet_material_indices) {
                if (idx < material.models.size() && material.models[idx].type != WallType::Pec) {
                    any_non_pec = true;
                    break;
                }
            }
        }
    }
    const bool use_material = any_non_pec;
    if (use_material && fringe.enabled)
        out.warnings.push_back("fringe correction is PEC-derived and unvalidated for "
                               "material walls; edge returns stay PEC-based");
    const double eta = kEta0 * std::sqrt(mu_r / eps_r);
    const double e0 = resolved["physics"]["incident_amplitude"].get<double>();
    geom::Vec3d phase_ref{0.0, 0.0, 0.0};
    if (resolved.contains("physics") && resolved["physics"].contains("phase_reference")) {
        const auto& pr = resolved["physics"]["phase_reference"];
        if (pr.is_array() && pr.size() == 3) {
            phase_ref = {pr[0].get<double>(), pr[1].get<double>(), pr[2].get<double>()};
        }
    }
    bool use_symmetry = false;
    std::string sym_plane = "xz";
    double sym_tol = 1e-4;
    if (resolved.contains("mesh") && resolved["mesh"].contains("symmetry")) {
        const auto& sym = resolved["mesh"]["symmetry"];
        use_symmetry = sym.value("enabled", false);
        sym_plane = sym.value("plane", "xz");
        sym_tol = sym.value("tolerance_m", 1e-4);
    }
    const SymmetryOptions sym_opts{use_symmetry, sym_plane, sym_tol};
    const HalfMeshResult half_result = use_symmetry ? extract_half_mesh(mesh, sym_opts) : HalfMeshResult{mesh, {}, {}};
    const NormalizedMesh& effective_mesh = use_symmetry ? half_result.mesh : mesh;

    bool curvature_correction = false;
    if (resolved.contains("solver") && resolved["solver"].contains("po_options")) {
        curvature_correction = resolved["solver"]["po_options"].value("curvature_correction", false);
    }
    const std::vector<CurvatureInfo> curvatures = curvature_correction ? estimate_mesh_curvatures(mesh) : std::vector<CurvatureInfo>{};

    const size_t nfreq = plan.frequencies_hz.size();
    const size_t ndir = plan.directions.size();
    const size_t npol = plan.polarizations.size();
    const size_t nunits = units.size();
    out.samples.resize(nunits * npol);
    std::atomic<size_t> fringe_skipped{0};
    std::atomic<size_t> chains_fired{0};
    const Bvh bvh = build_bvh(mesh);
    const ComplexMedium incident{{eps_r, 0.0}, {mu_r, 0.0}};
    std::vector<ComplexMedium> wall_per_freq;
    std::vector<std::vector<ComplexMedium>> models_wall_per_freq;
    if (use_material) {
        if (!has_multi_material && material.wall == WallType::Dielectric) {
            wall_per_freq.reserve(nfreq);
            for (double f : plan.frequencies_hz)
                wall_per_freq.push_back(evaluate_material(material.model, f));
        } else if (has_multi_material) {
            models_wall_per_freq.resize(material.models.size());
            for (size_t m = 0; m < material.models.size(); ++m) {
                if (material.models[m].type == WallType::Dielectric) {
                    models_wall_per_freq[m].reserve(nfreq);
                    for (double f : plan.frequencies_hz)
                        models_wall_per_freq[m].push_back(
                            evaluate_material(material.models[m], f));
                }
            }
        }
    }

    auto run_unit = [&](size_t pos) {
        const size_t fi = units[pos].first;
        const size_t di = units[pos].second;
        const double freq = plan.frequencies_hz[fi];
        const Direction& dir = plan.directions[di];
        const Real k = static_cast<Real>(2.0 * std::numbers::pi * freq *
                                         std::sqrt(eps_r * mu_r) / kSpeedOfLight);
        const Real eta_r = static_cast<Real>(eta);
        const Real e0_r = static_cast<Real>(e0);
        const C coeff_jk = C(0, 1) * k * eta_r / static_cast<Real>(4.0 * std::numbers::pi);
        const Vec3<Real> k_hat{static_cast<Real>(dir.k_hat.x), static_cast<Real>(dir.k_hat.y),
                               static_cast<Real>(dir.k_hat.z)};
        const Vec3<Real> to_inc{-k_hat.x, -k_hat.y, -k_hat.z};
        const Vec3<Real> r_hat = dir.is_bistatic
            ? Vec3<Real>{static_cast<Real>(dir.r_obs_hat.x), static_cast<Real>(dir.r_obs_hat.y),
                         static_cast<Real>(dir.r_obs_hat.z)}
            : to_inc;
        const auto tx_basis = polarization_basis(dir.k_hat);
        const Vec3<Real> tx_eh{static_cast<Real>(tx_basis.h.x), static_cast<Real>(tx_basis.h.y),
                               static_cast<Real>(tx_basis.h.z)};
        const Vec3<Real> tx_ev{static_cast<Real>(tx_basis.v.x), static_cast<Real>(tx_basis.v.y),
                               static_cast<Real>(tx_basis.v.z)};
        const Vec3<Real> tx_e[2] = {{tx_eh.x * e0_r, tx_eh.y * e0_r, tx_eh.z * e0_r},
                                    {tx_ev.x * e0_r, tx_ev.y * e0_r, tx_ev.z * e0_r}};

        const auto rx_basis = dir.is_bistatic ? polarization_basis(dir.r_obs_hat) : tx_basis;
        const Vec3<Real> rx_eh{static_cast<Real>(rx_basis.h.x), static_cast<Real>(rx_basis.h.y),
                               static_cast<Real>(rx_basis.h.z)};
        const Vec3<Real> rx_ev{static_cast<Real>(rx_basis.v.x), static_cast<Real>(rx_basis.v.y),
                               static_cast<Real>(rx_basis.v.z)};
        const CVec3<Real> r_hat_c = to_complex<Real>(r_hat);
        const CVec3<Real> k_hat_c = to_complex<Real>(k_hat);
        CVec3<Real> f_tx[2] = {};
        size_t lit = 0;

        auto accumulate_facet = [&](const Vec3<Real>& world_centroid, const Vec3<Real>& n,
                                    Real area, uint32_t self_tri, size_t mat_t) {
            if (!(dot(n, to_inc) > Real(0))) return; // hard shadow: n.(-k_inc)
            if (go.shadowing) {
                const geom::Vec3d center_d{static_cast<double>(world_centroid.x),
                                           static_cast<double>(world_centroid.y),
                                           static_cast<double>(world_centroid.z)};
                const geom::Vec3d r_hat_d{static_cast<double>(r_hat.x),
                                          static_cast<double>(r_hat.y),
                                          static_cast<double>(r_hat.z)};
                const geom::Vec3d diag = mesh.report.bbox_max - mesh.report.bbox_min;
                if (occluded_bvh(bvh, mesh, center_d, r_hat_d, 2.0 * diag.length(), self_tri))
                    return;
            }
            ++lit;
            const Vec3<Real> centroid{
                world_centroid.x - static_cast<Real>(phase_ref.x),
                world_centroid.y - static_cast<Real>(phase_ref.y),
                world_centroid.z - static_cast<Real>(phase_ref.z)};
            const C ein_phase = std::exp(C(0, -k * dot(k_hat, centroid)));
            const C eout_phase = std::exp(C(0, k * dot(r_hat, centroid)));
            const C coeff = coeff_jk * area * eout_phase;
            const CVec3<Real> n_c = to_complex<Real>(n);
            WallType facet_wall = WallType::Pec;
            const MaterialModel* facet_model = nullptr;
            if (material.enabled) {
                if (has_multi_material) {
                    if (mat_t < material.facet_material_indices.size()) {
                        uint32_t midx = material.facet_material_indices[mat_t];
                        if (midx < material.models.size()) {
                            facet_model = &material.models[midx];
                            facet_wall = facet_model->type;
                        }
                    }
                } else if (material.wall != WallType::Pec) {
                    facet_wall = material.wall;
                    facet_model = &material.model;
                }
            }

            C curv_c{Real(1), Real(0)};
            if (curvature_correction && mat_t < curvatures.size() && !curvatures[mat_t].is_discontinuous) {
                const auto c_fac = po_curvature_factor(curvatures[mat_t].mean_curvature,
                                                       static_cast<double>(k),
                                                       static_cast<double>(dot(n, to_inc)));
                curv_c = C(static_cast<Real>(c_fac.real()), static_cast<Real>(c_fac.imag()));
            }

            if (facet_wall == WallType::Pec) {
                for (int tx = 0; tx < 2; ++tx) {
                    const CVec3<Real> e_c = to_complex<Real>(tx_e[tx]);
                    CVec3<Real> h_inc = cross_c(k_hat_c, e_c);
                    for (auto& c : h_inc) c *= ein_phase / eta_r;
                    CVec3<Real> j_po = cross_c(n_c, h_inc);
                    for (auto& c : j_po) c *= Real(2) * curv_c;
                    // ADR-0001 transverse projector: r_hat x (r_hat x J).
                    const CVec3<Real> t1 = cross_c(r_hat_c, j_po);
                    const CVec3<Real> t2 = cross_c(r_hat_c, t1);
                    for (int i = 0; i < 3; ++i) f_tx[tx][i] += coeff * t2[i];
                }
            } else {
                const double cos_i = std::min(
                    1.0, std::max(0.0, static_cast<double>(dot(n, r_hat))));
                std::complex<double> r_te{-1.0, 0.0};
                std::complex<double> r_tm{1.0, 0.0};
                if (facet_wall == WallType::Dielectric) {
                    ComplexMedium wmed;
                    if (has_multi_material) {
                        uint32_t midx = material.facet_material_indices[mat_t];
                        wmed = models_wall_per_freq[midx][fi];
                    } else {
                        wmed = wall_per_freq[fi];
                    }
                    const FresnelResult fr = fresnel(incident, wmed, cos_i);
                    r_te = fr.r_te;
                    r_tm = fr.r_tm;
                } else if (facet_wall == WallType::Coated) {
                    const double lambda_vac = kSpeedOfLight / freq;
                    const CoatedResult cr =
                        coated_pec_reflection(incident, facet_model->layers, cos_i, lambda_vac);
                    r_te = cr.r_te;
                    r_tm = cr.r_tm;
                } else if (facet_wall == WallType::Impedance) {
                    const auto ir = impedance_reflection(facet_model->surface_impedance, cos_i);
                    r_te = ir.r_te;
                    r_tm = ir.r_tm;
                }
                const geom::Vec3d nd{static_cast<double>(n.x), static_cast<double>(n.y),
                                     static_cast<double>(n.z)};
                const geom::Vec3d kd{static_cast<double>(k_hat.x), static_cast<double>(k_hat.y),
                                     static_cast<double>(k_hat.z)};
                const geom::Vec3d rd{static_cast<double>(r_hat.x), static_cast<double>(r_hat.y),
                                     static_cast<double>(r_hat.z)};
                const geom::Vec3d cd{static_cast<double>(centroid.x),
                                     static_cast<double>(centroid.y),
                                     static_cast<double>(centroid.z)};
                geom::Vec3d e_te = geom::cross(nd, kd);
                if (e_te.lengthSquared() <= 1e-24) {
                    const geom::Vec3d ax =
                        (std::abs(nd.x) <= std::abs(nd.y) && std::abs(nd.x) <= std::abs(nd.z))
                            ? geom::Vec3d{1.0, 0.0, 0.0}
                        : (std::abs(nd.y) <= std::abs(nd.z) ? geom::Vec3d{0.0, 1.0, 0.0}
                                                            : geom::Vec3d{0.0, 0.0, 1.0});
                    e_te = geom::cross(nd, ax);
                }
                e_te.normalize();
                const geom::Vec3d tm_inc = geom::cross(kd, e_te);
                const double kn = geom::dot(kd, nd);
                const geom::Vec3d k_ref{kd.x - 2.0 * kn * nd.x, kd.y - 2.0 * kn * nd.y,
                                        kd.z - 2.0 * kn * nd.z};
                const geom::Vec3d tm_ref = geom::cross(k_ref, e_te);
                const double kd_d = static_cast<double>(k);
                const double eta_d = static_cast<double>(eta_r);
                const double area_d = static_cast<double>(area);
                const std::complex<double> ein_d =
                    std::exp(std::complex<double>(0.0, -kd_d * geom::dot(kd, cd)));
                for (int tx = 0; tx < 2; ++tx) {
                    const D3 e_inc = {static_cast<double>(tx_e[tx].x) * ein_d,
                                      static_cast<double>(tx_e[tx].y) * ein_d,
                                      static_cast<double>(tx_e[tx].z) * ein_d};
                    const D3 df = material_facet_field(nd, kd, rd, e_te, tm_inc, k_ref, tm_ref,
                                                       e_inc, kd_d, eta_d, area_d, cd, r_te, r_tm);
                    for (int i = 0; i < 3; ++i)
                        f_tx[tx][i] += C(static_cast<Real>(df[i].real()),
                                         static_cast<Real>(df[i].imag())) * curv_c;
                }
            }
        };

        for (size_t t = 0; t < effective_mesh.triangles.size(); ++t) {
            const auto& tri = effective_mesh.triangles[t];
            const Vec3<Real> world_centroid{
                static_cast<Real>((effective_mesh.vertices[tri[0]].x + effective_mesh.vertices[tri[1]].x +
                                   effective_mesh.vertices[tri[2]].x) /
                                  3.0),
                static_cast<Real>((effective_mesh.vertices[tri[0]].y + effective_mesh.vertices[tri[1]].y +
                                   effective_mesh.vertices[tri[2]].y) /
                                  3.0),
                static_cast<Real>((effective_mesh.vertices[tri[0]].z + effective_mesh.vertices[tri[1]].z +
                                   effective_mesh.vertices[tri[2]].z) /
                                  3.0)};
            const Vec3<Real> n{static_cast<Real>(effective_mesh.normals[t].x),
                               static_cast<Real>(effective_mesh.normals[t].y),
                               static_cast<Real>(effective_mesh.normals[t].z)};
            const Real area = static_cast<Real>(effective_mesh.areas[t]);
            const uint32_t orig_idx = use_symmetry ? half_result.original_indices[t] : static_cast<uint32_t>(t);
            accumulate_facet(world_centroid, n, area, orig_idx, orig_idx);

            if (use_symmetry) {
                const geom::Vec3d center_d{static_cast<double>(world_centroid.x),
                                           static_cast<double>(world_centroid.y),
                                           static_cast<double>(world_centroid.z)};
                const double coord = get_symmetry_coord(center_d, sym_plane);
                if (std::abs(coord) > sym_tol) {
                    const geom::Vec3d c_refl_d = reflect_point(center_d, sym_plane);
                    const geom::Vec3d n_refl_d = reflect_normal({n.x, n.y, n.z}, sym_plane);
                    const Vec3<Real> world_centroid_refl{static_cast<Real>(c_refl_d.x),
                                                         static_cast<Real>(c_refl_d.y),
                                                         static_cast<Real>(c_refl_d.z)};
                    const Vec3<Real> n_refl{static_cast<Real>(n_refl_d.x),
                                            static_cast<Real>(n_refl_d.y),
                                            static_cast<Real>(n_refl_d.z)};
                    const uint32_t refl_idx = half_result.reflected_indices[t];
                    const uint32_t refl_mat = (refl_idx != UINT32_MAX) ? refl_idx : orig_idx;
                    accumulate_facet(world_centroid_refl, n_refl, area, refl_idx, refl_mat);
                }
            }
        }
        if (fringe.enabled) {
            const geom::Vec3d s_hat_d{static_cast<double>(k_hat.x),
                                      static_cast<double>(k_hat.y),
                                      static_cast<double>(k_hat.z)};
            const geom::Vec3d r_hat_d{static_cast<double>(r_hat.x),
                                      static_cast<double>(r_hat.y),
                                      static_cast<double>(r_hat.z)};
            const double k_d = static_cast<double>(k);
            const double e0_d = static_cast<double>(e0_r);
            const PtdMethod ptd_m = (fringe.method == "ptd_full") ? PtdMethod::Full : PtdMethod::Lite;
            for (const MeshEdge& edge : fringe.edges->edges) {
                for (int tx = 0; tx < 2; ++tx) {
                    const std::array<std::complex<double>, 3> e_unit = {
                        std::complex<double>(static_cast<double>(tx_e[tx].x) / e0_d, 0.0),
                        std::complex<double>(static_cast<double>(tx_e[tx].y) / e0_d, 0.0),
                        std::complex<double>(static_cast<double>(tx_e[tx].z) / e0_d, 0.0)};
                    const auto fv = fringe_vector(edge, k_d, s_hat_d, r_hat_d, e_unit, phase_ref, ptd_m);
                    if (!fv) {
                        if (tx == 0) ++fringe_skipped;
                        continue;
                    }
                    for (int i = 0; i < 3; ++i)
                        f_tx[tx][i] +=
                            C(static_cast<Real>((*fv)[i].real() * e0_d),
                              static_cast<Real>((*fv)[i].imag() * e0_d));
                }
            }
        }
        if (go.max_bounces >= 2) {
            using D = std::complex<double>;
            using D3 = std::array<D, 3>;
            const geom::Vec3d s{static_cast<double>(k_hat.x), static_cast<double>(k_hat.y),
                                static_cast<double>(k_hat.z)};
            const geom::Vec3d r{static_cast<double>(r_hat.x), static_cast<double>(r_hat.y),
                                static_cast<double>(r_hat.z)};
            const double kd = static_cast<double>(k);
            const double eta_d = static_cast<double>(eta_r);
            const double e0_d = static_cast<double>(e0_r);
            const geom::Vec3d diag = mesh.report.bbox_max - mesh.report.bbox_min;
            const double scene = 2.0 * diag.length();
            const double tmin = 1e-9 * diag.length();
            auto dot3 = [](const D3& a, const geom::Vec3d& b) {
                return a[0] * b.x + a[1] * b.y + a[2] * b.z;
            };
            auto cross3 = [](const geom::Vec3d& a, const D3& b) {
                return D3{a.y * b[2] - a.z * b[1], a.z * b[0] - a.x * b[2],
                          a.x * b[1] - a.y * b[0]};
            };
            auto reflect_e = [&dot3](const D3& e, const geom::Vec3d& n) {
                const D d = dot3(e, n);
                return D3{2.0 * d * n.x - e[0], 2.0 * d * n.y - e[1], 2.0 * d * n.z - e[2]};
            };
            auto reflect_d = [](const geom::Vec3d& d, const geom::Vec3d& n) {
                const double p = geom::dot(d, n);
                return geom::Vec3d(d.x - 2.0 * p * n.x, d.y - 2.0 * p * n.y,
                                   d.z - 2.0 * p * n.z);
            };
            const size_t ntri = mesh.triangles.size();
            for (size_t ia = 0; ia < ntri; ++ia) {
                const geom::Vec3d na = mesh.normals[ia];
                if (!(geom::dot(na, r) > 0.0)) continue;
                const auto& ta = mesh.triangles[ia];
                const geom::Vec3d ca =
                    (mesh.vertices[ta[0]] + mesh.vertices[ta[1]] + mesh.vertices[ta[2]]) *
                    (1.0 / 3.0);
                const geom::Vec3d d1 = reflect_d(s, na);
                const auto hit = ray_bvh(bvh, mesh, ca, d1, tmin, uint32_t(ia));
                if (!hit) continue;
                const size_t ib = hit->tri;
                if (ib == ia) continue;
                const geom::Vec3d nb = mesh.normals[ib];
                if (!(geom::dot(nb, d1) < 0.0)) continue;
                const auto& tb = mesh.triangles[ib];
                const geom::Vec3d cb =
                    (mesh.vertices[tb[0]] + mesh.vertices[tb[1]] + mesh.vertices[tb[2]]) *
                    (1.0 / 3.0);
                if (!(geom::dot(reflect_d(d1, nb), r) > 1.0 - 1e-9)) continue;
                const auto exit = ray_bvh(bvh, mesh, cb, r, tmin, uint32_t(ib));
                if (exit && exit->t < scene) continue;
                ++chains_fired;
                const D path = std::exp(D(0.0, -kd * (geom::dot(s, ca - phase_ref) +
                                                       geom::dot(d1, cb - ca))));
                const D amp = D(0.0, 1.0) * kd * eta_d / (4.0 * std::numbers::pi) *
                              static_cast<double>(mesh.areas[ib]) *
                              std::exp(D(0.0, kd * geom::dot(r, cb - phase_ref)));
                for (int tx = 0; tx < 2; ++tx) {
                    const D3 ein = {tx_e[tx].x / e0_d, tx_e[tx].y / e0_d,
                                    tx_e[tx].z / e0_d};
                    const D3 e1 = reflect_e(ein, na);
                    D3 hb = cross3(d1, e1);
                    for (auto& c : hb) c *= path / eta_d;
                    const D3 jb0 = cross3(nb, hb);
                    const D3 jb{2.0 * jb0[0], 2.0 * jb0[1], 2.0 * jb0[2]};
                    const D3 t1 = cross3(r, jb);
                    const D3 t2 = cross3(r, t1);
                    for (int i = 0; i < 3; ++i)
                        f_tx[tx][i] += C(static_cast<Real>((amp * t2[i]).real() * e0_d),
                                         static_cast<Real>((amp * t2[i]).imag() * e0_d));
                }
            }
            if (go.max_bounces >= 3) {
                for (size_t ia = 0; ia < ntri; ++ia) {
                    const geom::Vec3d na = mesh.normals[ia];
                    if (!(geom::dot(na, r) > 0.0)) continue;
                    const auto& ta = mesh.triangles[ia];
                    const geom::Vec3d ca =
                        (mesh.vertices[ta[0]] + mesh.vertices[ta[1]] + mesh.vertices[ta[2]]) *
                        (1.0 / 3.0);
                    const geom::Vec3d d1 = reflect_d(s, na);
                    const auto hit1 = ray_bvh(bvh, mesh, ca, d1, tmin, uint32_t(ia));
                    if (!hit1) continue;
                    const size_t ib = hit1->tri;
                    if (ib == ia) continue;
                    const geom::Vec3d nb = mesh.normals[ib];
                    if (!(geom::dot(nb, d1) < 0.0)) continue;
                    const auto& tb = mesh.triangles[ib];
                    const geom::Vec3d cb =
                        (mesh.vertices[tb[0]] + mesh.vertices[tb[1]] + mesh.vertices[tb[2]]) *
                        (1.0 / 3.0);
                    const geom::Vec3d d2 = reflect_d(d1, nb);
                    const auto hit2 = ray_bvh(bvh, mesh, cb, d2, tmin, uint32_t(ib));
                    if (!hit2) continue;
                    const size_t ic = hit2->tri;
                    if (ic == ib) continue;
                    const geom::Vec3d nc = mesh.normals[ic];
                    if (!(geom::dot(nc, d2) < 0.0)) continue;
                    const auto& tc = mesh.triangles[ic];
                    const geom::Vec3d cc =
                        (mesh.vertices[tc[0]] + mesh.vertices[tc[1]] +
                         mesh.vertices[tc[2]]) *
                        (1.0 / 3.0);
                    if (!(geom::dot(reflect_d(d2, nc), r) > 1.0 - 1e-9)) continue;
                    const auto exit = ray_bvh(bvh, mesh, cc, r, tmin, uint32_t(ic));
                    if (exit && exit->t < scene) continue;
                    ++chains_fired;
                    const D epath =
                        std::exp(D(0.0, -kd * (geom::dot(s, ca - phase_ref) +
                                               geom::dot(d1, cb - ca) +
                                               geom::dot(d2, cc - cb))));
                    const D amp =
                        D(0.0, 1.0) * kd * eta_d / (4.0 * std::numbers::pi) *
                        static_cast<double>(mesh.areas[ic]) *
                        std::exp(D(0.0, kd * geom::dot(r, cc - phase_ref)));
                    for (int tx = 0; tx < 2; ++tx) {
                        const D3 ein = {tx_e[tx].x / e0_d, tx_e[tx].y / e0_d,
                                        tx_e[tx].z / e0_d};
                        const D3 e2 = reflect_e(reflect_e(ein, na), nb);
                        D3 hc = cross3(d2, e2);
                        for (auto& c : hc) c *= epath / eta_d;
                        const D3 jc0 = cross3(nc, hc);
                        const D3 jc{2.0 * jc0[0], 2.0 * jc0[1], 2.0 * jc0[2]};
                        const D3 t1 = cross3(r, jc);
                        const D3 t2 = cross3(r, t1);
                        for (int i = 0; i < 3; ++i)
                            f_tx[tx][i] +=
                                C(static_cast<Real>((amp * t2[i]).real() * e0_d),
                                  static_cast<Real>((amp * t2[i]).imag() * e0_d));
                    }
                }
            }
        }
        // Linear scattering matrix S[rx][tx], rows/cols (H, V).
        const Vec3<Real> rx_e[2] = {rx_eh, rx_ev};
        std::complex<double> f_tx_d[2][3];
        for (int tx = 0; tx < 2; ++tx)
            for (int i = 0; i < 3; ++i)
                f_tx_d[tx][i] = {static_cast<double>(f_tx[tx][i].real()),
                                 static_cast<double>(f_tx[tx][i].imag())};
        const double eh_d[3] = {static_cast<double>(rx_eh.x), static_cast<double>(rx_eh.y),
                                static_cast<double>(rx_eh.z)};
        const double ev_d[3] = {static_cast<double>(rx_ev.x), static_cast<double>(rx_ev.y),
                                static_cast<double>(rx_ev.z)};
        const auto channels = project_unit_channels(
            f_tx_d, eh_d, ev_d, static_cast<double>(e0_r), plan.polarizations);
        for (size_t pi = 0; pi < npol; ++pi) {
            PoSampleResult& sample = out.samples[pos * npol + pi];
            sample.sample_id = (static_cast<uint64_t>(fi) * ndir + di) * npol + pi;
            sample.frequency_id = static_cast<uint32_t>(fi);
            sample.direction_id = static_cast<uint32_t>(di);
            sample.pol_id = static_cast<uint32_t>(pi);
            sample.scattering = channels[pi].first;
            sample.rcs_sqm = channels[pi].second;
            sample.lit_facets = lit;
        }
    };

    const unsigned workers =
        nunits == 0 ? 1 : std::min<unsigned>(std::max(1u, num_threads), nunits);
    if (workers <= 1) {
        for (size_t pos = 0; pos < nunits; ++pos) run_unit(pos);
    } else {
        std::vector<std::thread> threads;
        std::exception_ptr first_error;
        std::mutex error_mutex;
        const size_t stride = (nunits + workers - 1) / workers;
        for (unsigned w = 0; w < workers; ++w) {
            const size_t begin = w * stride;
            const size_t end = std::min(begin + stride, nunits);
            if (begin >= end) break;
            threads.emplace_back([&, begin, end] {
                try {
                    for (size_t pos = begin; pos < end; ++pos) run_unit(pos);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!first_error) first_error = std::current_exception();
                }
            });
        }
        for (auto& t : threads) t.join();
        if (first_error) std::rethrow_exception(first_error);
    }
    if (fringe.enabled) {
        if (fringe_skipped.load() > 0) {
            out.warnings.push_back("fringe correction skipped " +
                                   std::to_string(fringe_skipped.load()) +
                                   " edge-sample(s): no transverse frame (end-on incidence)");
        }
        if (fringe.edges != nullptr) {
            bool has_non_pec = false;
            for (const auto& edge : fringe.edges->edges) {
                if (edge.tri0 >= 0 && static_cast<size_t>(edge.tri0) < mesh.triangle_tags.size() &&
                    mesh.triangle_tags[edge.tri0] != "pec") {
                    has_non_pec = true;
                    break;
                }
                if (edge.tri1 >= 0 && static_cast<size_t>(edge.tri1) < mesh.triangle_tags.size() &&
                    mesh.triangle_tags[edge.tri1] != "pec") {
                    has_non_pec = true;
                    break;
                }
            }
            if (has_non_pec) {
                out.warnings.push_back("edge diffraction enabled with non-PEC faces; PTD assumes PEC wedge");
            }
        }
    }
    out.chains_fired = chains_fired.load();
    return out;
}

} // namespace

namespace {
std::vector<std::pair<uint32_t, uint32_t>> all_units(const SamplePlan& plan) {
    std::vector<std::pair<uint32_t, uint32_t>> units;
    for (uint32_t fi = 0; fi < plan.frequencies_hz.size(); ++fi)
        for (uint32_t di = 0; di < plan.directions.size(); ++di)
            units.emplace_back(fi, di);
    return units;
}

PoResult solve_backend(const NormalizedMesh& mesh, const SamplePlan& plan,
                       const nlohmann::json& resolved,
                       const std::vector<std::pair<uint32_t, uint32_t>>& units,
                       const FringeOptions& fringe, const GoOptions& go,
                       const MaterialOptions& material) {
    const std::string precision = resolved["solver"].value("precision", "float64");
    const unsigned threads = resolved["execution"].value("cpu_threads", 1u);
    if (fringe.enabled) {
        if (fringe.edges == nullptr)
            throw std::invalid_argument("fringe correction enabled without an edge model");
        if (resolved["execution"].value("accelerator", "cpu") == "cuda")
            throw cuda::CudaError("fringe correction is CPU-only in v1");
    }
    if (resolved.contains("solver") && resolved["solver"].contains("po_options") &&
        resolved["solver"]["po_options"].value("curvature_correction", false) &&
        resolved["execution"].value("accelerator", "cpu") == "cuda") {
        throw cuda::CudaError("curvature correction is CPU-only in v1");
    }
    if (go.max_bounces < 1 || go.max_bounces > 3)
        throw std::invalid_argument("max_bounces must be 1, 2, or 3");
    bool any_non_pec = false;
    if (material.enabled) {
        if (!material.facet_material_indices.empty()) {
            for (uint32_t idx : material.facet_material_indices) {
                if (idx < material.models.size() && material.models[idx].type != WallType::Pec) {
                    any_non_pec = true;
                    break;
                }
            }
        } else if (material.wall != WallType::Pec) {
            any_non_pec = true;
            validate_material(material.model);
            if (material.model.type != material.wall)
                throw std::invalid_argument("material model type must match the active wall");
            if (material.wall == WallType::Coated && !material.model.table.empty())
                throw std::invalid_argument("coated frequency tables are future work");
        }
        if (any_non_pec) {
            if (go.max_bounces > 1)
                throw std::invalid_argument("multi-bounce GO requires PEC walls in v1");
        }
    }
    if (resolved["execution"].value("accelerator", "cpu") == "cuda") {
        const int device = resolved["execution"].value("cuda_device_id", 0);
        return cuda::solve_po_cuda_units(mesh, plan, resolved, device, units, fringe, go, material);
    }
    if (precision == "float32")
        return solve_typed<float>(mesh, plan, resolved, threads, units, fringe, go, material);
    return solve_typed<double>(mesh, plan, resolved, threads, units, fringe, go, material);
}
} // namespace

PoResult solve_po(const NormalizedMesh& mesh, const SamplePlan& plan,
                  const nlohmann::json& resolved, const FringeOptions& fringe,
                  const GoOptions& go, const MaterialOptions& material) {
    std::vector<std::pair<uint32_t, uint32_t>> units;
    for (uint32_t fi = 0; fi < plan.frequencies_hz.size(); ++fi)
        for (uint32_t di = 0; di < plan.directions.size(); ++di) units.emplace_back(fi, di);
    return solve_backend(mesh, plan, resolved, units, fringe, go, material);
}

PoResult solve_po_units(const NormalizedMesh& mesh, const SamplePlan& plan,
                        const nlohmann::json& resolved,
                        const std::vector<std::pair<uint32_t, uint32_t>>& units,
                        const FringeOptions& fringe, const GoOptions& go,
                        const MaterialOptions& material) {
    return solve_backend(mesh, plan, resolved, units, fringe, go, material);
}

} // namespace strikecem

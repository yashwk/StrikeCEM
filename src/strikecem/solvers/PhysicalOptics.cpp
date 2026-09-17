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

template <typename Real>
PoResult solve_typed(const NormalizedMesh& mesh, const SamplePlan& plan,
                     const nlohmann::json& resolved, unsigned num_threads,
                     const std::vector<std::pair<uint32_t, uint32_t>>& units,
                     const FringeOptions& fringe, const GoOptions& go) {
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
    const double eta = kEta0 * std::sqrt(mu_r / eps_r);
    const double e0 = resolved["physics"]["incident_amplitude"].get<double>();

    const size_t nfreq = plan.frequencies_hz.size();
    const size_t ndir = plan.directions.size();
    const size_t npol = plan.polarizations.size();
    const size_t nunits = units.size();
    out.samples.resize(nunits * npol);
    std::atomic<size_t> fringe_skipped{0};

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
        const Vec3<Real> r_hat{-k_hat.x, -k_hat.y, -k_hat.z};
        const auto basis = polarization_basis(dir.k_hat);
        const Vec3<Real> eh{static_cast<Real>(basis.h.x), static_cast<Real>(basis.h.y),
                            static_cast<Real>(basis.h.z)};
        const Vec3<Real> ev{static_cast<Real>(basis.v.x), static_cast<Real>(basis.v.y),
                            static_cast<Real>(basis.v.z)};
        const Vec3<Real> tx_e[2] = {{eh.x * e0_r, eh.y * e0_r, eh.z * e0_r},
                                    {ev.x * e0_r, ev.y * e0_r, ev.z * e0_r}};
        const CVec3<Real> r_hat_c = to_complex<Real>(r_hat);
        const CVec3<Real> k_hat_c = to_complex<Real>(k_hat);
        CVec3<Real> f_tx[2] = {};
        size_t lit = 0;
        for (size_t t = 0; t < mesh.triangles.size(); ++t) {
            const auto& tri = mesh.triangles[t];
            const Vec3<Real> centroid{
                static_cast<Real>((mesh.vertices[tri[0]].x + mesh.vertices[tri[1]].x +
                                   mesh.vertices[tri[2]].x) /
                                  3.0),
                static_cast<Real>((mesh.vertices[tri[0]].y + mesh.vertices[tri[1]].y +
                                   mesh.vertices[tri[2]].y) /
                                  3.0),
                static_cast<Real>((mesh.vertices[tri[0]].z + mesh.vertices[tri[1]].z +
                                   mesh.vertices[tri[2]].z) /
                                  3.0)};
            const Vec3<Real> n{static_cast<Real>(mesh.normals[t].x),
                               static_cast<Real>(mesh.normals[t].y),
                               static_cast<Real>(mesh.normals[t].z)};
            if (!(dot(n, r_hat) > Real(0))) continue; // hard shadow: n.(-k_hat)
            if (go.shadowing) {
                const geom::Vec3d center_d{static_cast<double>(centroid.x),
                                           static_cast<double>(centroid.y),
                                           static_cast<double>(centroid.z)};
                const geom::Vec3d r_hat_d{static_cast<double>(r_hat.x),
                                          static_cast<double>(r_hat.y),
                                          static_cast<double>(r_hat.z)};
                const geom::Vec3d diag = mesh.report.bbox_max - mesh.report.bbox_min;
                if (occluded(mesh, center_d, r_hat_d, 2.0 * diag.length(),
                             static_cast<uint32_t>(t)))
                    continue;
            }
            ++lit;
            const Real area = static_cast<Real>(mesh.areas[t]);
            const C ein_phase = std::exp(C(0, -k * dot(k_hat, centroid)));
            const C eout_phase = std::exp(C(0, k * dot(r_hat, centroid)));
            const C coeff = coeff_jk * area * eout_phase;
            const CVec3<Real> n_c = to_complex<Real>(n);
            for (int tx = 0; tx < 2; ++tx) {
                const CVec3<Real> e_c = to_complex<Real>(tx_e[tx]);
                CVec3<Real> h_inc = cross_c(k_hat_c, e_c);
                for (auto& c : h_inc) c *= ein_phase / eta_r;
                CVec3<Real> j_po = cross_c(n_c, h_inc);
                for (auto& c : j_po) c *= Real(2);
                // ADR-0001 transverse projector: r_hat x (r_hat x J).
                const CVec3<Real> t1 = cross_c(r_hat_c, j_po);
                const CVec3<Real> t2 = cross_c(r_hat_c, t1);
                for (int i = 0; i < 3; ++i) f_tx[tx][i] += coeff * t2[i];
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
            for (const MeshEdge& edge : fringe.edges->edges) {
                for (int tx = 0; tx < 2; ++tx) {
                    const std::array<std::complex<double>, 3> e_unit = {
                        std::complex<double>(static_cast<double>(tx_e[tx].x) / e0_d, 0.0),
                        std::complex<double>(static_cast<double>(tx_e[tx].y) / e0_d, 0.0),
                        std::complex<double>(static_cast<double>(tx_e[tx].z) / e0_d, 0.0)};
                    const auto fv = fringe_vector(edge, k_d, s_hat_d, r_hat_d, e_unit);
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
        if (go.two_bounce) {
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
                for (size_t ib = 0; ib < ntri; ++ib) {
                    if (ib == ia) continue;
                    const geom::Vec3d nb = mesh.normals[ib];
                    if (!(geom::dot(nb, d1) < 0.0)) continue;
                    const auto hit = ray_mesh(mesh, ca, d1, tmin, uint32_t(ia));
                    if (!hit || hit->tri != ib) continue;
                    const auto& tb = mesh.triangles[ib];
                    const geom::Vec3d cb =
                        (mesh.vertices[tb[0]] + mesh.vertices[tb[1]] + mesh.vertices[tb[2]]) *
                        (1.0 / 3.0);
                    if (!(geom::dot(reflect_d(d1, nb), r) > 1.0 - 1e-9)) continue;
                    const auto exit = ray_mesh(mesh, cb, r, tmin, uint32_t(ib));
                    if (exit && exit->t < scene) continue;
                    const D path = std::exp(D(0.0, -kd * (geom::dot(s, ca) +
                                                           geom::dot(d1, cb - ca))));
                    const D amp = D(0.0, 1.0) * kd * eta_d / (4.0 * std::numbers::pi) *
                                  static_cast<double>(mesh.areas[ib]) *
                                  std::exp(D(0.0, kd * geom::dot(r, cb)));
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
            }
        }
        // Linear scattering matrix S[rx][tx], rows/cols (H, V).
        const Vec3<Real> rx_e[2] = {eh, ev};
        std::complex<double> f_tx_d[2][3];
        for (int tx = 0; tx < 2; ++tx)
            for (int i = 0; i < 3; ++i)
                f_tx_d[tx][i] = {static_cast<double>(f_tx[tx][i].real()),
                                 static_cast<double>(f_tx[tx][i].imag())};
        const double eh_d[3] = {static_cast<double>(eh.x), static_cast<double>(eh.y),
                                static_cast<double>(eh.z)};
        const double ev_d[3] = {static_cast<double>(ev.x), static_cast<double>(ev.y),
                                static_cast<double>(ev.z)};
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
    if (fringe.enabled && fringe_skipped.load() > 0)
        out.warnings.push_back("fringe correction skipped " +
                               std::to_string(fringe_skipped.load()) +
                               " edge-sample(s): no transverse frame (end-on incidence)");
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
                       const FringeOptions& fringe, const GoOptions& go) {
    const std::string precision = resolved["solver"].value("precision", "float64");
    const unsigned threads = resolved["execution"].value("cpu_threads", 1u);
    if (fringe.enabled) {
        if (fringe.edges == nullptr)
            throw std::invalid_argument("fringe correction enabled without an edge model");
        if (resolved["execution"].value("accelerator", "cpu") == "cuda")
            throw cuda::CudaError("fringe correction is CPU-only in v1");
    }
    if ((go.shadowing || go.two_bounce) &&
        resolved["execution"].value("accelerator", "cpu") == "cuda")
        throw cuda::CudaError("GO corrections are CPU-only in v1");
    if (resolved["execution"].value("accelerator", "cpu") == "cuda") {
        const int device = resolved["execution"].value("cuda_device_id", 0);
        return cuda::solve_po_cuda_units(mesh, plan, resolved, device, units);
    }
    if (precision == "float32")
        return solve_typed<float>(mesh, plan, resolved, threads, units, fringe, go);
    return solve_typed<double>(mesh, plan, resolved, threads, units, fringe, go);
}
} // namespace

PoResult solve_po(const NormalizedMesh& mesh, const SamplePlan& plan,
                  const nlohmann::json& resolved, const FringeOptions& fringe,
                  const GoOptions& go) {
    std::vector<std::pair<uint32_t, uint32_t>> units;
    for (uint32_t fi = 0; fi < plan.frequencies_hz.size(); ++fi)
        for (uint32_t di = 0; di < plan.directions.size(); ++di) units.emplace_back(fi, di);
    return solve_backend(mesh, plan, resolved, units, fringe, go);
}

PoResult solve_po_units(const NormalizedMesh& mesh, const SamplePlan& plan,
                        const nlohmann::json& resolved,
                        const std::vector<std::pair<uint32_t, uint32_t>>& units,
                        const FringeOptions& fringe, const GoOptions& go) {
    return solve_backend(mesh, plan, resolved, units, fringe, go);
}

} // namespace strikecem

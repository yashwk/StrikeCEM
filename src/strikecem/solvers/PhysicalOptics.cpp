// Deterministic CPU PO reference solver (ADR-0001, SPEC FR-6).
#include "strikecem/solvers/PhysicalOptics.hpp"

#include <array>
#include <cmath>
#include <numbers>

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
                     const nlohmann::json& resolved) {
    using C = std::complex<Real>;
    PoResult out;
    const auto& medium = resolved["frequency"]["medium"];
    const double eps_r = medium["epsilon_r"].get<double>();
    const double mu_r = medium["mu_r"].get<double>();
    if (medium["sigma"].get<double>() != 0.0)
        out.warnings.push_back("medium conductivity is ignored in v1 PO");
    const double eta = kEta0 * std::sqrt(mu_r / eps_r);
    const double e0 = resolved["physics"]["incident_amplitude"].get<double>();

    uint64_t sample_id = 0;
    uint32_t frequency_id = 0;
    for (double freq : plan.frequencies_hz) {
        const Real k = static_cast<Real>(2.0 * std::numbers::pi * freq *
                                         std::sqrt(eps_r * mu_r) / kSpeedOfLight);
        const Real eta_r = static_cast<Real>(eta);
        const Real e0_r = static_cast<Real>(e0);
        const C coeff_jk = C(0, 1) * k * eta_r / static_cast<Real>(4.0 * std::numbers::pi);
        for (size_t di = 0; di < plan.directions.size(); ++di) {
            const Direction& dir = plan.directions[di];
            const Vec3<Real> k_hat{static_cast<Real>(dir.k_hat.x),
                                   static_cast<Real>(dir.k_hat.y),
                                   static_cast<Real>(dir.k_hat.z)};
            const Vec3<Real> r_hat{-k_hat.x, -k_hat.y, -k_hat.z};
            const Vec3<Real> neg_k = r_hat; // -k_hat, for the illumination test
            const auto basis = polarization_basis(dir.k_hat);
            const Vec3<Real> eh{static_cast<Real>(basis.h.x), static_cast<Real>(basis.h.y),
                                static_cast<Real>(basis.h.z)};
            const Vec3<Real> ev{static_cast<Real>(basis.v.x), static_cast<Real>(basis.v.y),
                                static_cast<Real>(basis.v.z)};
            const Vec3<Real> tx_e[2] = {{eh.x * e0_r, eh.y * e0_r, eh.z * e0_r},
                                        {ev.x * e0_r, ev.y * e0_r, ev.z * e0_r}};
            const CVec3<Real> r_hat_c = to_complex<Real>(r_hat);
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
                if (!(dot(n, neg_k) > Real(0))) continue; // hard shadow
                ++lit;
                const Real area = static_cast<Real>(mesh.areas[t]);
                const C ein_phase = std::exp(C(0, -k * dot(k_hat, centroid)));
                const C eout_phase = std::exp(C(0, k * dot(r_hat, centroid)));
                const C coeff = coeff_jk * area * eout_phase;
                const CVec3<Real> n_c = to_complex<Real>(n);
                for (int tx = 0; tx < 2; ++tx) {
                    const CVec3<Real> e_c = to_complex<Real>(tx_e[tx]);
                    const CVec3<Real> k_c = to_complex<Real>(k_hat);
                    CVec3<Real> h_inc = cross_c(k_c, e_c);
                    for (auto& c : h_inc) c *= ein_phase / eta_r;
                    CVec3<Real> j_po = cross_c(n_c, h_inc);
                    for (auto& c : j_po) c *= Real(2);
                    // ADR-0001 transverse projector: r_hat x (r_hat x J).
                    const CVec3<Real> t1 = cross_c(r_hat_c, j_po);
                    const CVec3<Real> t2 = cross_c(r_hat_c, t1);
                    for (int i = 0; i < 3; ++i) f_tx[tx][i] += coeff * t2[i];
                }
            }
            // Linear scattering matrix S[rx][tx], rows/cols (H, V).
            const Vec3<Real> rx_e[2] = {eh, ev};
            std::complex<double> s_lin[2][2];
            for (int rx = 0; rx < 2; ++rx)
                for (int tx = 0; tx < 2; ++tx) {
                    const C s = (f_tx[tx][0] * rx_e[rx].x + f_tx[tx][1] * rx_e[rx].y +
                                 f_tx[tx][2] * rx_e[rx].z) /
                                e0_r;
                    s_lin[rx][tx] = {static_cast<double>(s.real()),
                                     static_cast<double>(s.imag())};
                }
            const Complex2x2 circ = linear_to_circular(
                {std::array<std::complex<double>, 2>{s_lin[0][0], s_lin[0][1]},
                 std::array<std::complex<double>, 2>{s_lin[1][0], s_lin[1][1]}});
            for (size_t pi = 0; pi < plan.polarizations.size(); ++pi) {
                const std::string& pol = plan.polarizations[pi];
                std::complex<double> s{0, 0};
                if (pol == "HH")
                    s = s_lin[0][0];
                else if (pol == "VV")
                    s = s_lin[1][1];
                else if (pol == "HV")
                    s = s_lin[1][0];
                else if (pol == "VH")
                    s = s_lin[0][1];
                else if (pol == "RHCP")
                    s = circ[0][0];
                else if (pol == "LHCP")
                    s = circ[1][1];
                PoSampleResult sample;
                sample.sample_id = sample_id++;
                sample.frequency_id = frequency_id;
                sample.direction_id = static_cast<uint32_t>(di);
                sample.pol_id = static_cast<uint32_t>(pi);
                sample.scattering = s;
                sample.rcs_sqm = 4.0 * std::numbers::pi * std::norm(s);
                sample.lit_facets = lit;
                out.samples.push_back(sample);
            }
        }
        ++frequency_id;
    }
    return out;
}

} // namespace

PoResult solve_po(const NormalizedMesh& mesh, const SamplePlan& plan,
                  const nlohmann::json& resolved) {
    const std::string precision = resolved["solver"].value("precision", "float64");
    if (precision == "float32") return solve_typed<float>(mesh, plan, resolved);
    return solve_typed<double>(mesh, plan, resolved);
}

} // namespace strikecem

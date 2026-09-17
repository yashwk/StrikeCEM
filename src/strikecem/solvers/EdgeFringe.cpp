// 3D straight-edge fringe geometry (ADR-0003 slices C/D1).
#include "strikecem/solvers/EdgeFringe.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

#include "strikecem/solvers/UtdWedge.hpp"

namespace strikecem {
namespace {

constexpr double kPi = std::numbers::pi;

bool is_unit(const geom::Vec3d& v) {
    if (!std::isfinite(v.x + v.y + v.z)) return false;
    const double n = v.length();
    return std::isfinite(n) && std::abs(n - 1.0) <= 1e-6;
}

double norm_2pi(double a) {
    a = std::fmod(a, 2.0 * kPi);
    if (a < 0.0) a += 2.0 * kPi;
    return a;
}

// Unit transverse basis (e1, e2 = e1 x t) from a face direction, or false.
bool transverse_basis(const geom::Vec3d& face_dir, const geom::Vec3d& t, geom::Vec3d& e1,
                      geom::Vec3d& e2) {
    e1 = face_dir - t * geom::dot(face_dir, t);
    const double n = e1.length();
    if (!(n > 1e-12)) return false;
    e1 = e1 * (1.0 / n);
    e2 = geom::cross(e1, t);
    return true;
}

void project_angles(const geom::Vec3d& e1, const geom::Vec3d& e2,
                    const geom::Vec3d& s_hat, const geom::Vec3d& r_hat, double& phi,
                    double& phi_prime) {
    const geom::Vec3d arrival = s_hat * -1.0; // edge -> source
    phi = norm_2pi(std::atan2(geom::dot(r_hat, e2), geom::dot(r_hat, e1)));
    phi_prime = norm_2pi(std::atan2(geom::dot(arrival, e2), geom::dot(arrival, e1)));
}

} // namespace

TransverseAngles edge_transverse_angles(const MeshEdge& edge,
                                        const geom::Vec3d& s_hat,
                                        const geom::Vec3d& r_hat) {
    if (!is_unit(s_hat) || !is_unit(r_hat))
        throw std::invalid_argument("edge fringe directions must be unit vectors");
    TransverseAngles out;
    if (!edge.boundary || std::abs(edge.wedge_n - 2.0) > 1e-9) return out; // ponytail: rims only
    const double t_len = edge.tangent.length();
    if (!(t_len > 0.0)) throw std::invalid_argument("edge tangent must be non-zero");
    const geom::Vec3d t = edge.tangent * (1.0 / t_len);
    geom::Vec3d e1, e2;
    if (!transverse_basis(edge.face0_dir, t, e1, e2)) return out;
    const geom::Vec3d st = s_hat - t * geom::dot(s_hat, t);
    const double sin_beta0 = st.length();
    if (!(sin_beta0 > 1e-12)) return out; // end-on: no transverse plane
    project_angles(e1, e2, s_hat, r_hat, out.phi, out.phi_prime);
    out.sin_beta0 = sin_beta0;
    out.valid = true;
    return out;
}

TransverseAngles wedge_transverse_angles(const MeshEdge& edge,
                                         const geom::Vec3d& s_hat,
                                         const geom::Vec3d& r_hat) {
    if (!is_unit(s_hat) || !is_unit(r_hat))
        throw std::invalid_argument("edge fringe directions must be unit vectors");
    TransverseAngles out;
    if (edge.boundary || edge.wedge_n < 1.0 || edge.wedge_n > 2.0) return out;
    const double t_len = edge.tangent.length();
    if (!(t_len > 0.0)) throw std::invalid_argument("edge tangent must be non-zero");
    const geom::Vec3d t = edge.tangent * (1.0 / t_len);
    geom::Vec3d e1, e2;
    if (!transverse_basis(edge.face0_dir, t, e1, e2)) return out;
    geom::Vec3d f1, dummy;
    if (!transverse_basis(edge.face1_dir, t, f1, dummy)) return out;
    if (geom::dot(f1, e2) > 0.0) e2 = e2 * -1.0; // face-tri1 ray lands at wedge_n * pi
    const geom::Vec3d st = s_hat - t * geom::dot(s_hat, t);
    const double sin_beta0 = st.length();
    if (!(sin_beta0 > 1e-12)) return out; // end-on: no transverse plane
    project_angles(e1, e2, s_hat, r_hat, out.phi, out.phi_prime);
    out.sin_beta0 = sin_beta0;
    out.valid = true;
    return out;
}

std::optional<std::complex<double>> fringe_amplitude(const MeshEdge& edge, double k,
                                                     const geom::Vec3d& s_hat,
                                                     const geom::Vec3d& r_hat, char pol) {
    if (pol != 's' && pol != 'h') throw std::invalid_argument("fringe pol must be 's' or 'h'");
    if (!(k > 0.0) || !std::isfinite(k))
        throw std::invalid_argument("wavenumber must be positive and finite");
    if (!is_unit(s_hat) || !is_unit(r_hat))
        throw std::invalid_argument("edge fringe directions must be unit vectors");
    const TransverseAngles ang =
        edge.boundary ? edge_transverse_angles(edge, s_hat, r_hat)
                      : wedge_transverse_angles(edge, s_hat, r_hat);
    if (!ang.valid) return std::nullopt;
    const double n = edge.boundary ? 2.0 : edge.wedge_n;
    const double kt = k * ang.sin_beta0;
    const auto integral = along_edge_integral(edge, k, s_hat, r_hat); // throws on zero length
    const auto d = utd_coefficient(kt, edge.length, n, ang.phi, ang.phi_prime);
    return (pol == 's' ? d.soft : d.hard) * integral;
}

std::complex<double> along_edge_integral(const MeshEdge& edge, double k,
                                         const geom::Vec3d& s_hat,
                                         const geom::Vec3d& r_hat) {
    if (!(k > 0.0) || !std::isfinite(k))
        throw std::invalid_argument("wavenumber must be positive and finite");
    if (!is_unit(s_hat) || !is_unit(r_hat))
        throw std::invalid_argument("edge fringe directions must be unit vectors");
    if (!(edge.length > 0.0)) throw std::invalid_argument("edge length must be positive");
    const double t_len = edge.tangent.length();
    if (!(t_len > 0.0)) throw std::invalid_argument("edge tangent must be non-zero");
    const geom::Vec3d t = edge.tangent * (1.0 / t_len);
    const geom::Vec3d d = r_hat - s_hat;
    const double a = geom::dot(d, t);
    const geom::Vec3d c = (edge.p0 + edge.p1) * 0.5;
    const double phase = k * geom::dot(d, c);
    const double x = k * edge.length * a / 2.0;
    const double sinc = (std::abs(x) < 1e-8) ? 1.0 - x * x / 6.0 : std::sin(x) / x;
    const double mag = edge.length * sinc;
    return {mag * std::cos(phase), mag * std::sin(phase)};
}

} // namespace strikecem

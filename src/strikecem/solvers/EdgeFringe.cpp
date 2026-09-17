// 3D straight-edge fringe geometry (ADR-0003 slice C).
#include "strikecem/solvers/EdgeFringe.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

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
    geom::Vec3d e1 = edge.face0_dir - t * geom::dot(edge.face0_dir, t);
    const double e1_len = e1.length();
    if (!(e1_len > 1e-12)) return out;
    e1 = e1 * (1.0 / e1_len);
    const geom::Vec3d e2 = geom::cross(e1, t); // unit: e1 ⊥ t, both unit
    const geom::Vec3d st = s_hat - t * geom::dot(s_hat, t);
    const double sin_beta0 = st.length();
    if (!(sin_beta0 > 1e-12)) return out; // end-on: no transverse plane
    const geom::Vec3d arrival = s_hat * -1.0; // edge -> source
    const double au = geom::dot(arrival, e1), av = geom::dot(arrival, e2);
    const double ou = geom::dot(r_hat, e1), ov = geom::dot(r_hat, e2);
    out.phi = norm_2pi(std::atan2(ov, ou));
    out.phi_prime = norm_2pi(std::atan2(av, au));
    out.sin_beta0 = sin_beta0;
    out.valid = true;
    return out;
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

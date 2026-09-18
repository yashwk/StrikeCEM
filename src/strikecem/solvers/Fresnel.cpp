#include "strikecem/solvers/Fresnel.hpp"

#include <cmath>
#include <stdexcept>

#include "strikecem/core/Conventions.hpp"

namespace strikecem {
namespace {

constexpr double kPassiveTol = 1e-12;

bool finite_c(const std::complex<double>& z) {
    return std::isfinite(z.real() + z.imag());
}

}

std::complex<double> refractive_index(const ComplexMedium& m) {
    if (!finite_c(m.eps_r) || !finite_c(m.mu_r))
        throw std::invalid_argument("medium constants must be finite");
    return std::sqrt(m.eps_r * m.mu_r);
}

std::complex<double> wave_impedance(const ComplexMedium& m) {
    if (!finite_c(m.eps_r) || !finite_c(m.mu_r))
        throw std::invalid_argument("medium constants must be finite");
    if (m.eps_r == std::complex<double>{0.0, 0.0})
        throw std::invalid_argument("eps_r must be non-zero");
    return kEta0 * std::sqrt(m.mu_r / m.eps_r);
}

FresnelResult fresnel(const ComplexMedium& medium, const ComplexMedium& wall,
                      double cos_theta_i) {
    if (!(cos_theta_i >= 0.0) || !(cos_theta_i <= 1.0) || !std::isfinite(cos_theta_i))
        throw std::invalid_argument("cos_theta_i must be in [0, 1]");
    return fresnel(medium, wall, std::complex<double>{cos_theta_i, 0.0});
}

FresnelResult fresnel(const ComplexMedium& medium, const ComplexMedium& wall,
                      std::complex<double> cos_theta_i) {
    for (const auto z : {medium.eps_r, medium.mu_r, wall.eps_r, wall.mu_r})
        if (!finite_c(z)) throw std::invalid_argument("medium constants must be finite");
    if (wall.eps_r.imag() > kPassiveTol || wall.mu_r.imag() > kPassiveTol)
        throw std::invalid_argument("wall must be passive (Im(eps), Im(mu) <= 0)");
    if (!finite_c(cos_theta_i))
        throw std::invalid_argument("cos_theta_i must be finite");
    const std::complex<double> n1 = refractive_index(medium);
    const std::complex<double> n2 = refractive_index(wall);
    const std::complex<double> e1 = wave_impedance(medium);
    const std::complex<double> e2 = wave_impedance(wall);
    const std::complex<double> ci = cos_theta_i;
    const std::complex<double> sin_theta_i =
        std::sqrt(std::complex<double>{1.0, 0.0} - ci * ci);
    const std::complex<double> s = n1 * sin_theta_i / n2;
    std::complex<double> ct = std::sqrt(std::complex<double>{1.0, 0.0} - s * s);
    if ((n2 * ct).imag() > 0.0) ct = -ct;
    FresnelResult out;
    out.cos_theta_t = ct;
    const std::complex<double> d_te = e2 * ci + e1 * ct;
    const std::complex<double> d_tm = e1 * ci + e2 * ct;
    if (d_te == std::complex<double>{0.0, 0.0} || d_tm == std::complex<double>{0.0, 0.0})
        throw std::invalid_argument("degenerate Fresnel denominator");
    out.r_te = (e2 * ci - e1 * ct) / d_te;
    out.t_te = 2.0 * e2 * ci / d_te;
    out.r_tm = (e1 * ci - e2 * ct) / d_tm;
    out.t_tm = 2.0 * e2 * ci / d_tm;
    return out;
}

}

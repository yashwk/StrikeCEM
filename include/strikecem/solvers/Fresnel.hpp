#pragma once
// Fresnel core for materials (ADR-0005 slice 1): e^{+jwt} complex-medium
// helpers plus single-interface amplitude coefficients. No material
// tables, no coatings, no solver wiring here.
#include <complex>

namespace strikecem {

// Passive complex relative permittivity: eps'' >= 0 (conductivity
// folds as sigma/omega); mu likewise. Vacuum is (1, 1).
struct ComplexMedium {
    std::complex<double> eps_r{1.0, 0.0};
    std::complex<double> mu_r{1.0, 0.0};
};

// n = sqrt(eps*mu), eta = eta0*sqrt(mu/eps), principal branches.
std::complex<double> refractive_index(const ComplexMedium& m);
std::complex<double> wave_impedance(const ComplexMedium& m); // ohms

struct FresnelResult {
    std::complex<double> r_te; // E perpendicular to incidence plane (s)
    std::complex<double> r_tm; // E in the incidence plane (p, Hecht sign)
    std::complex<double> t_te;
    std::complex<double> t_tm;
    std::complex<double> cos_theta_t; // complex Snell transmission cosine
};

// Single interface medium -> wall at real incidence angle cos_theta_i in
// [0, 1]. Impedance form: exact with magnetic contrast, Hecht r_s/r_p
// when mu1 == mu2. Throws std::invalid_argument on non-passive walls
// (imag(eps_r) > 0 or imag(mu_r) > 0 beyond fp tolerance), cos outside
// [0,1], or non-finite input.
// ponytail: single-interface only; multilayer recursion arrives with the
// coated-PEC slice, which is its upgrade path.
FresnelResult fresnel(const ComplexMedium& medium, const ComplexMedium& wall, double cos_theta_i);

} // namespace strikecem

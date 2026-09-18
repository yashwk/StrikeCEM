#pragma once
#include <complex>

namespace strikecem {

struct ComplexMedium {
    std::complex<double> eps_r{1.0, 0.0};
    std::complex<double> mu_r{1.0, 0.0};
};

std::complex<double> refractive_index(const ComplexMedium& m);
std::complex<double> wave_impedance(const ComplexMedium& m);

struct FresnelResult {
    std::complex<double> r_te;
    std::complex<double> r_tm;
    std::complex<double> t_te;
    std::complex<double> t_tm;
    std::complex<double> cos_theta_t;
};

FresnelResult fresnel(const ComplexMedium& medium, const ComplexMedium& wall, double cos_theta_i);

FresnelResult fresnel(const ComplexMedium& medium, const ComplexMedium& wall,
                      std::complex<double> cos_theta_i);

}

#pragma once
#include <complex>
#include <utility>

namespace strikecem {

double fresnel_c(double x);
double fresnel_s(double x);

std::complex<double> utd_transition(double x);

struct WedgeCoefficients {
    std::complex<double> soft;
    std::complex<double> hard;
};
WedgeCoefficients utd_coefficient(double k, double rho, double n, double phi, double phi_prime);

std::complex<double> utd_total(char pol, double k, double sx, double sy, double x, double y,
                               double e0 = 1.0);

std::complex<double> utd_go_incident(double k, double sx, double sy, double x, double y,
                                     double e0 = 1.0);
std::complex<double> utd_go_reflected(char pol, double k, double sx, double sy, double x,
                                      double y, double e0 = 1.0);
std::complex<double> utd_diffracted(char pol, double k, double sx, double sy, double x,
                                    double y, double e0 = 1.0);

}

#pragma once
// 2D UTD wedge diffraction (ADR-0003): Fresnel integrals, KP transition
// function and diffraction coefficients, and the GO + diffracted total
// field for an exterior wedge, validated against the Sommerfeld exact
// half-plane solution. e^{+jwt} convention, matching the v1 solvers.
#include <complex>
#include <utility>

namespace strikecem {

// Fresnel cosine/sine integrals C(x), S(x).
double fresnel_c(double x);
double fresnel_s(double x);

// KP transition function F(x) = 2j*sqrt(x)*e^{jx} * int_{sqrt(x)}^inf
// e^{-jt^2} dt, x >= 0. F(0) = 0, F -> 1 as x -> infinity.
std::complex<double> utd_transition(double x);

// KP diffraction coefficients for an exterior wedge of angle n*pi,
// 2D normal incidence (sin beta0 = 1). phi in (0, 2*n*pi) excluding the
// screen; phi_prime is the arrival angle in the same frame.
struct WedgeCoefficients {
    std::complex<double> soft; // E parallel to edge (Dirichlet)
    std::complex<double> hard; // H parallel to edge (Neumann)
};
WedgeCoefficients utd_coefficient(double k, double rho, double n, double phi, double phi_prime);

// Total 2D field for a half-plane screen (x >= 0 on y = 0): incident plus
// reflected GO with exact geometric shadowing, plus the UTD edge wave.
// pol 's' = E out of plane (R = -1), 'h' = H out of plane (R = +1).
// (sx, sy) is the incident direction of TRAVEL; (x, y) the observation
// point; the edge sits at the origin (phase reference).
std::complex<double> utd_total(char pol, double k, double sx, double sy, double x, double y,
                               double e0 = 1.0);

// Components, exposed for tests.
std::complex<double> utd_go_incident(double k, double sx, double sy, double x, double y,
                                     double e0 = 1.0);
std::complex<double> utd_go_reflected(char pol, double k, double sx, double sy, double x,
                                      double y, double e0 = 1.0);
std::complex<double> utd_diffracted(char pol, double k, double sx, double sy, double x,
                                    double y, double e0 = 1.0);

} // namespace strikecem

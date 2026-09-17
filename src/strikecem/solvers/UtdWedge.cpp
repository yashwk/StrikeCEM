// 2D UTD wedge diffraction (ADR-0003). Conventions: e^{+jwt}, screen on
// x >= 0 (half-plane), exterior angle phi in (0, 2*pi) CCW from +x.
// Arrival angle phi' is the math angle of (-travel direction).
#include "strikecem/solvers/UtdWedge.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace strikecem {
namespace {

constexpr double kPi = std::numbers::pi;

// Fresnel auxiliary functions, derived by repeated integration by parts
// (all powers verified term-by-term; the crossover test pins truncation).
long double aux_f(long double x) {
    const long double p1 = kPi * x;                 // pi*x
    const long double p3x5 = p1 * p1 * p1 * x * x;  // pi^3*x^5
    const long double p5x9 = p3x5 * p1 * p1 * x * x * x * x; // pi^5*x^9
    const long double p7x13 = p5x9 * p1 * p1 * x * x * x * x; // pi^7*x^13
    return 1.0L / p1 - 3.0L / p3x5 + 105.0L / p5x9 - 10395.0L / p7x13;
}

long double aux_g(long double x) {
    const long double p2x3 = kPi * kPi * x * x * x;             // pi^2*x^3
    const long double p4x7 = p2x3 * kPi * kPi * x * x * x * x;  // pi^4*x^7
    const long double p6x11 = p4x7 * kPi * kPi * x * x * x * x; // pi^6*x^11
    return 1.0L / p2x3 - 15.0L / p4x7 + 945.0L / p6x11;
}

constexpr double kCrossover = 3.0;

void fresnel_cs(long double x, long double& c, long double& s) {
    const long double ax = fabsl(x);
    long double cc, ss;
    if (ax <= kCrossover) {
        // Power series in long double (cancellation-safe headroom).
        const long double x2 = ax * ax;
        const long double p = kPi / 2.0L;
        long double term_c = ax, term_s = p * ax * x2 / 3.0L;
        cc = term_c;
        ss = term_s;
        for (int n = 1; n < 60; ++n) {
            const long double fn = n;
            term_c *= -p * p * x2 * x2 / (((2 * fn) * (2 * fn - 1) * (4 * fn + 1)) / (4 * fn - 3));
            term_s *= -p * p * x2 * x2 / (((2 * fn + 1) * (2 * fn) * (4 * fn + 3)) / (4 * fn - 1));
            cc += term_c;
            ss += term_s;
            if (fabsl(term_c) < 1e-20L && fabsl(term_s) < 1e-20L) break;
        }
    } else {
        const long double arg = kPi * ax * ax / 2.0L;
        const long double sn = sinl(arg), cs = cosl(arg);
        const long double f = aux_f(ax), g = aux_g(ax);
        cc = 0.5L + f * sn - g * cs;
        ss = 0.5L - f * cs - g * sn;
    }
    c = (x < 0) ? -cc : cc;
    s = (x < 0) ? -ss : ss;
}

double norm_angle(double a) {
    a = std::fmod(a, 2.0 * kPi);
    if (a < 0.0) a += 2.0 * kPi;
    if (a == 0.0) return 0.0;
    return a;
}

// Segment [p, p - t*s], t > 0, hits the screen ray x >= 0, y = 0.
bool hits_screen(double px, double py, double sx, double sy) {
    if (std::abs(sy) < 1e-300) return false; // parallel: never crosses
    const double t = py / sy;
    if (!(t > 0.0)) return false;
    return px - t * sx >= 0.0;
}

} // namespace

double fresnel_c(double x) {
    long double c, s;
    fresnel_cs(x, c, s);
    return static_cast<double>(c);
}

double fresnel_s(double x) {
    long double c, s;
    fresnel_cs(x, c, s);
    return static_cast<double>(s);
}

std::complex<double> utd_transition(double x) {
    if (!(x >= 0.0)) throw std::invalid_argument("transition argument must be >= 0");
    if (x == 0.0) return {0.0, 0.0};
    const double t0 = std::sqrt(2.0 * x / kPi);
    const double c = fresnel_c(t0), s = fresnel_s(t0);
    // F = 2j*sqrt(x)*e^{jx}*sqrt(pi/2)*[(1/2 - C) - j*(1/2 - S)]
    const std::complex<double> rest((0.5 - c), -(0.5 - s));
    const std::complex<double> phase(std::cos(x), std::sin(x));
    return std::complex<double>(0.0, 2.0 * std::sqrt(x)) * phase * std::sqrt(kPi / 2.0) * rest;
}

WedgeCoefficients utd_coefficient(double k, double rho, double n, double phi, double phi_prime) {
    const std::complex<double> j(0.0, 1.0);
    const std::complex<double> c0 =
        -std::exp(-j * kPi / 4.0) / (2.0 * n * std::sqrt(2.0 * kPi * k));
    auto term = [&](double beta) {
        const double np = std::round((beta + kPi) / (2.0 * n * kPi));
        const double nm = std::round((beta - kPi) / (2.0 * n * kPi));
        const double ap = 2.0 * std::pow(std::cos((2.0 * n * kPi * np - beta) / 2.0), 2);
        const double am = 2.0 * std::pow(std::cos((2.0 * n * kPi * nm - beta) / 2.0), 2);
        std::complex<double> out{0.0, 0.0};
        const double tp = (kPi + beta) / (2.0 * n);
        const double tm = (kPi - beta) / (2.0 * n);
        const double cp = std::cos(tp) / std::sin(tp);
        const double cm = std::cos(tm) / std::sin(tm);
        // cot/0 with F = 0 is the two-sided average 0 (antisymmetric jump).
        if (std::isfinite(cp)) out += cp * utd_transition(k * rho * ap);
        if (std::isfinite(cm)) out += cm * utd_transition(k * rho * am);
        return out;
    };
    const double beta_m = phi - phi_prime;
    const double beta_p = phi + phi_prime;
    const auto tm = term(beta_m);
    const auto tp = term(beta_p);
    return {c0 * (tm - tp), c0 * (tm + tp)};
}

std::complex<double> utd_go_incident(double k, double sx, double sy, double x, double y,
                                     double e0) {
    if (hits_screen(x, y, sx, sy)) return {0.0, 0.0};
    return e0 * std::exp(std::complex<double>(0.0, -k * (sx * x + sy * y)));
}

std::complex<double> utd_go_reflected(char pol, double k, double sx, double sy, double x,
                                      double y, double e0) {
    if (std::abs(sy) < 1e-300) return {0.0, 0.0}; // grazing: no reflection
    const double t = y / -sy; // travel mirrored direction (sx, -sy) back to y = 0
    if (!(t > 0.0)) return {0.0, 0.0};
    if (x - t * sx < 0.0) return {0.0, 0.0}; // specular point off the screen
    const double r = (pol == 's') ? -1.0 : 1.0;
    return r * e0 * std::exp(std::complex<double>(0.0, -k * (sx * x - sy * y)));
}

std::complex<double> utd_diffracted(char pol, double k, double sx, double sy, double x,
                                    double y, double e0) {
    const double rho = std::hypot(x, y);
    if (!(rho > 0.0)) return {0.0, 0.0};
    const double phi = norm_angle(std::atan2(y, x));
    const double phi_prime = norm_angle(std::atan2(-sy, -sx));
    const auto d = utd_coefficient(k, rho, 2.0, phi, phi_prime);
    const std::complex<double> coeff = (pol == 's') ? d.soft : d.hard;
    return e0 * coeff * std::exp(std::complex<double>(0.0, -k * rho)) / std::sqrt(rho);
}

std::complex<double> utd_total(char pol, double k, double sx, double sy, double x, double y,
                               double e0) {
    return utd_go_incident(k, sx, sy, x, y, e0) +
           utd_go_reflected(pol, k, sx, sy, x, y, e0) +
           utd_diffracted(pol, k, sx, sy, x, y, e0);
}

} // namespace strikecem

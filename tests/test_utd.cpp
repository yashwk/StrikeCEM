#include <cmath>
#include <gtest/gtest.h>

#include "strikecem/solvers/UtdWedge.hpp"

namespace {

constexpr double kPi = 3.141592653589793;

namespace reference {
std::complex<double> fresnel_fc(double xi) {
    const double c = strikecem::fresnel_c(xi);
    const double s = strikecem::fresnel_s(xi);
    const std::complex<double> u((0.5 - c), -(0.5 - s));
    return 1.0 - std::complex<double>(1.0, 1.0) / 2.0 * u;
}

std::complex<double> sommerfeld(char pol, double k, double rho, double phi, double phi_prime,
                                double e0 = 1.0) {
    const double spx = std::cos(phi_prime + kPi);
    const double spy = std::sin(phi_prime + kPi);
    const double x = rho * std::cos(phi), y = rho * std::sin(phi);
    const std::complex<double> j(0.0, 1.0);
    const auto phase_i = std::exp(-j * k * (spx * x + spy * y));
    const auto phase_r = std::exp(-j * k * (spx * x - spy * y));
    const double s = 2.0 * std::sqrt(k * rho / kPi);
    const double xi_i = s * std::cos((phi - phi_prime) / 2.0);
    const double xi_r = s * std::cos((phi + phi_prime) / 2.0);
    const double r = (pol == 's') ? -1.0 : 1.0;
    return e0 * (phase_i * fresnel_fc(xi_i) + r * phase_r * fresnel_fc(xi_r));
}
}

TEST(Utd, FresnelIdentities) {
    EXPECT_DOUBLE_EQ(strikecem::fresnel_c(0.0), 0.0);
    EXPECT_DOUBLE_EQ(strikecem::fresnel_s(0.0), 0.0);
    EXPECT_NEAR(strikecem::fresnel_c(1.0), 0.779893400377, 1e-9);
    EXPECT_NEAR(strikecem::fresnel_s(1.0), 0.438259147390, 1e-9);
    for (double x : {5.0, 10.0, 20.0}) {
        EXPECT_LT(std::abs(strikecem::fresnel_c(x) - 0.5), 2.0 / (kPi * x)) << "x=" << x;
        EXPECT_LT(std::abs(strikecem::fresnel_s(x) - 0.5), 2.0 / (kPi * x)) << "x=" << x;
    }
    EXPECT_NEAR(strikecem::fresnel_c(-1.3), -strikecem::fresnel_c(1.3), 1e-15);
    EXPECT_NEAR(strikecem::fresnel_s(-2.1), -strikecem::fresnel_s(2.1), 1e-15);
    for (double x : {0.5, 1.0, 2.0}) {
        const double h = 1e-6;
        const double dc = (strikecem::fresnel_c(x + h) - strikecem::fresnel_c(x - h)) / (2 * h);
        EXPECT_NEAR(dc, std::cos(kPi * x * x / 2.0), 1e-9) << "x=" << x;
    }
}

TEST(Utd, KnownValues) {
    EXPECT_NEAR(strikecem::fresnel_c(3.0), 0.605720789298, 1e-9);
    EXPECT_NEAR(strikecem::fresnel_s(3.0), 0.496312998967, 1e-9);
    EXPECT_NEAR(strikecem::fresnel_c(2.0), 0.488253406075, 1e-9);
    EXPECT_NEAR(strikecem::fresnel_s(2.0), 0.343415678364, 1e-9);
}

TEST(Utd, TransitionLimits) {
    const auto f0 = strikecem::utd_transition(0.0);
    EXPECT_DOUBLE_EQ(f0.real(), 0.0);
    EXPECT_DOUBLE_EQ(f0.imag(), 0.0);
    EXPECT_NEAR(std::abs(strikecem::utd_transition(100.0) - 1.0), 0.0, 0.02);
    EXPECT_LT(std::abs(strikecem::utd_transition(1.0)), 2.0);
}

TEST(Utd, SommerfeldFaceZero) {
    const double k = 2.0 * kPi, phi_prime = kPi / 3.0;
    for (double rho : {3.0, 10.0}) {
        EXPECT_NEAR(std::abs(reference::sommerfeld('s', k, rho, 0.0, phi_prime)), 0.0,
                    1e-9)
            << "rho=" << rho;
        const double xi_face = 2.0 * std::sqrt(k * rho / kPi) * std::cos(phi_prime / 2.0);
        EXPECT_NEAR(std::abs(reference::sommerfeld('h', k, rho, 0.0, phi_prime)), 2.0,
                    3.0 / (kPi * xi_face))
            << "rho=" << rho;
        EXPECT_LT(std::abs(reference::sommerfeld('s', k, rho, 2.0 * kPi, phi_prime)), 0.01);
        EXPECT_LT(std::abs(reference::sommerfeld('h', k, rho, 2.0 * kPi, phi_prime)),
                  rho < 5.0 ? 0.3 : 0.15);
    }
}

TEST(Utd, SommerfeldShadowDecayAndGoRecovery) {
    const double k = 2.0 * kPi, phi_prime = kPi / 3.0;
    for (char pol : {'s', 'h'}) {
        const double near = std::abs(reference::sommerfeld(pol, k, 20.0 / k, 1.5 * kPi, phi_prime));
        const double far = std::abs(reference::sommerfeld(pol, k, 80.0 / k, 1.5 * kPi, phi_prime));
        EXPECT_LT(near, 0.35) << "pol=" << pol;
        EXPECT_NEAR(near / far, 2.0, 0.5) << "pol=" << pol;
    }
    const double rho = 500.0 / k;
    const double phi = 0.4 * kPi;
    const double x = rho * std::cos(phi), y = rho * std::sin(phi);
    const double spx = std::cos(phi_prime + kPi), spy = std::sin(phi_prime + kPi);
    const std::complex<double> j(0.0, 1.0);
    const auto ei = std::exp(-j * k * (spx * x + spy * y));
    const auto er = std::exp(-j * k * (spx * x - spy * y));
    EXPECT_NEAR(std::abs(reference::sommerfeld('s', k, rho, phi, phi_prime) - (ei - er)), 0.0,
                0.05);
    EXPECT_NEAR(std::abs(reference::sommerfeld('h', k, rho, phi, phi_prime) - (ei + er)), 0.0,
                0.05);
}

double max_neighbor_jump(char pol, double k, double rho, double arrival, int steps,
                         bool utd) {
    double worst = 0.0;
    std::complex<double> prev{0.0, 0.0};
    bool first = true;
    const double sx = std::cos(arrival + kPi), sy = std::sin(arrival + kPi);
    for (int i = 0; i <= steps; ++i) {
        const double phi = 0.02 + (2.0 * kPi - 0.04) * i / steps;
        if (std::abs(phi) < 0.05 || std::abs(phi - 2.0 * kPi) < 0.05) continue;
        const double x = rho * std::cos(phi), y = rho * std::sin(phi);
        const std::complex<double> cur =
            utd ? strikecem::utd_total(pol, k, sx, sy, x, y)
                : reference::sommerfeld(pol, k, rho, phi, arrival);
        if (!first) worst = std::max(worst, std::abs(cur - prev));
        prev = cur;
        first = false;
    }
    return worst;
}

TEST(Utd, FieldsAreContinuousByRefinement) {
    const double k = 2.0 * kPi, arrival = 70.0 * kPi / 180.0, rho = 10.0 / k;
    for (char pol : {'s', 'h'}) {
        const double coarse_s = max_neighbor_jump(pol, k, rho, arrival, 180, false);
        const double fine_s = max_neighbor_jump(pol, k, rho, arrival, 720, false);
        EXPECT_LT(fine_s, 2.0 * coarse_s + 1e-12) << "pol=" << pol;
        const double coarse_u = max_neighbor_jump(pol, k, rho, arrival, 180, true);
        const double fine_u = max_neighbor_jump(pol, k, rho, arrival, 720, true);
        EXPECT_LT(fine_u, 2.0 * coarse_u + 1e-12) << "pol=" << pol;
    }
}

TEST(Utd, MatchesSommerfeldAcrossBoundaries) {
    const double k = 2.0 * kPi, arrival = 70.0 * kPi / 180.0;
    const double sx = std::cos(arrival + kPi), sy = std::sin(arrival + kPi);
    for (double kr : {5.0, 15.0}) {
        const double rho = kr / k;
        for (char pol : {'s', 'h'}) {
            double worst = 0.0;
            for (int i = 0; i < 145; ++i) {
                const double phi = 0.02 + (2.0 * kPi - 0.04) * i / 144;
                const double x = rho * std::cos(phi), y = rho * std::sin(phi);
                const auto got = strikecem::utd_total(pol, k, sx, sy, x, y);
                const auto ref = reference::sommerfeld(pol, k, rho, phi, arrival);
                worst = std::max(worst, std::abs(got - ref));
            }
            EXPECT_LT(worst, 0.02) << "pol=" << pol << " kr=" << kr;
        }
    }
}

TEST(Utd, KellerLimitFarFromBoundaries) {
    const double k = 2.0 * kPi, rho = 60.0 / k, n = 2.0;
    const double phi = 0.3 * kPi, phi_prime = 0.25 * kPi;
    const auto d = strikecem::utd_coefficient(k, rho, n, phi, phi_prime);
    const std::complex<double> j(0.0, 1.0);
    const std::complex<double> c0 = -std::exp(-j * kPi / 4.0) / (2.0 * n * std::sqrt(2 * kPi * k));
    auto keller = [&](double beta) {
        return std::cos((kPi + beta) / (2 * n)) / std::sin((kPi + beta) / (2 * n)) +
               std::cos((kPi - beta) / (2 * n)) / std::sin((kPi - beta) / (2 * n));
    };
    EXPECT_NEAR(std::abs(d.soft - c0 * (keller(phi - phi_prime) - keller(phi + phi_prime))), 0.0,
                0.05 * std::abs(d.soft));
}

TEST(Utd, WedgeCoefficientStaysFinite) {
    const double k = 2.0 * kPi, rho = 10.0 / k, n = 1.5;
    for (int i = 0; i <= 60; ++i) {
        const double phi = 0.05 + (n * kPi - 0.1) * i / 60;
        const auto d = strikecem::utd_coefficient(k, rho, n, phi, 1.0);
        EXPECT_TRUE(std::isfinite(d.soft.real() + d.soft.imag()));
        EXPECT_TRUE(std::isfinite(d.hard.real() + d.hard.imag()));
    }
}

}

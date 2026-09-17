// Fresnel core tests (ADR-0005 slice 1): convention helpers, analytic
// normal incidence, Brewster zero, PEC limit, lossless energy, TIR, and
// the decaying-branch invariant. No fitting: every bound is analytic.
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

#include "strikecem/solvers/Fresnel.hpp"

namespace {

constexpr double kPi = 3.141592653589793;

strikecem::ComplexMedium lossless(double n) {
    return {{n * n, 0.0}, {1.0, 0.0}};
}

TEST(Fresnel, VacuumHelpers) {
    const strikecem::ComplexMedium vac;
    EXPECT_DOUBLE_EQ(strikecem::refractive_index(vac).real(), 1.0);
    EXPECT_DOUBLE_EQ(strikecem::refractive_index(vac).imag(), 0.0);
    EXPECT_DOUBLE_EQ(strikecem::wave_impedance(vac).real(), 376.73031346177066);
    const strikecem::ComplexMedium glass = lossless(1.5);
    EXPECT_DOUBLE_EQ(strikecem::refractive_index(glass).real(), 1.5);
    EXPECT_NEAR(strikecem::wave_impedance(glass).real(), 376.73031346177066 / 1.5, 1e-9);
    EXPECT_THROW(strikecem::wave_impedance({{0.0, 0.0}, {1.0, 0.0}}), std::invalid_argument);
}

TEST(Fresnel, NormalIncidenceAnalytic) {
    // Air -> n=2 at theta=0: Rte = (1-2)/(1+2), Rtm = -Rte (Hecht sign).
    const auto f = strikecem::fresnel({}, lossless(2.0), 1.0);
    EXPECT_NEAR(f.r_te.real(), -1.0 / 3.0, 1e-12);
    EXPECT_NEAR(f.r_te.imag(), 0.0, 1e-12);
    EXPECT_NEAR(f.r_tm.real(), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(f.r_tm.imag(), 0.0, 1e-12);
    EXPECT_NEAR(f.t_te.real(), 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(f.cos_theta_t.real(), 1.0, 1e-12);
}

TEST(Fresnel, BrewsterZero) {
    // Non-magnetic air -> glass: TM vanishes at tan(thetaB) = n2/n1.
    const double theta_b = std::atan(1.5);
    const auto f = strikecem::fresnel({}, lossless(1.5), std::cos(theta_b));
    EXPECT_LT(std::abs(f.r_tm), 1e-9);
    EXPECT_GT(std::abs(f.r_te), 0.0); // TE stays finite: not a no-reflection point
}

TEST(Fresnel, PecLimit) {
    // Huge conductivity: R_te -> -1, R_tm -> +1 (ray-fixed basis: the
    // reflected p-basis flips with the ray, so both satisfy the PEC
    // tangential-flip with opposite amplitude signs). Rate ~1/|n|.
    const strikecem::ComplexMedium metal{{1.0, -1e8}, {1.0, 0.0}};
    const auto f = strikecem::fresnel({}, metal, std::cos(0.3));
    EXPECT_LT(std::abs(f.r_te + 1.0), 1e-3);
    EXPECT_LT(std::abs(f.r_tm - 1.0), 1e-3);
}

TEST(Fresnel, LosslessEnergy) {
    // |R|^2 + (n2 Re(ct) / n1 ci) |T|^2 = 1 per polarization.
    const double ci = std::cos(0.6);
    const auto f = strikecem::fresnel({}, lossless(1.5), ci);
    const double pref = 1.5 * f.cos_theta_t.real() / ci;
    EXPECT_NEAR(std::norm(f.r_te) + pref * std::norm(f.t_te), 1.0, 1e-12);
    EXPECT_NEAR(std::norm(f.r_tm) + pref * std::norm(f.t_tm), 1.0, 1e-12);
}

TEST(Fresnel, TotalInternalReflection) {
    // Glass -> air past critical (asin(1/1.5) = 0.7297): unit reflection,
    // decaying transmitted branch.
    const auto f = strikecem::fresnel(lossless(1.5), {}, std::cos(1.0));
    EXPECT_NEAR(std::abs(f.r_te), 1.0, 1e-12);
    EXPECT_NEAR(std::abs(f.r_tm), 1.0, 1e-12);
    EXPECT_LT(f.cos_theta_t.imag(), 0.0);
}

TEST(Fresnel, DecayingBranchInvariant) {
    // Im(n2*ct) <= 0 across lossless/lossy/metallic walls and
    // transmission/TIR regimes: the transmitted wave never grows.
    for (const auto wall :
         {lossless(1.5), strikecem::ComplexMedium{{2.0, -0.5}, {1.0, 0.0}},
          strikecem::ComplexMedium{{0.2, -3.0}, {1.0, 0.0}}}) {
        const auto n2 = strikecem::refractive_index(wall);
        for (double theta : {0.0, 0.6, 1.0, 1.4}) {
            const auto f = strikecem::fresnel({}, wall, std::cos(theta));
            EXPECT_LE((n2 * f.cos_theta_t).imag(), 1e-12) << "theta=" << theta;
        }
    }
}

TEST(Fresnel, PassiveWallsOnly) {
    // Gain (Im(eps) > 0 under e^{+jwt}) is rejected, not silently run.
    EXPECT_THROW(strikecem::fresnel({}, {{{1.0, 1.0}}, {{1.0, 0.0}}}, 1.0),
                 std::invalid_argument);
    EXPECT_THROW(strikecem::fresnel({}, lossless(1.5), 2.0), std::invalid_argument);
    // Lossy wall reflects without gain.
    const auto f = strikecem::fresnel({}, {{2.0, -0.5}, {1.0, 0.0}}, std::cos(0.6));
    EXPECT_LE(std::abs(f.r_te), 1.0 + 1e-12);
    EXPECT_LE(std::abs(f.r_tm), 1.0 + 1e-12);
}

} // namespace

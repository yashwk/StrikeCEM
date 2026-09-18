#include <cmath>
#include <complex>
#include <limits>
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
    const auto f = strikecem::fresnel({}, lossless(2.0), 1.0);
    EXPECT_NEAR(f.r_te.real(), -1.0 / 3.0, 1e-12);
    EXPECT_NEAR(f.r_te.imag(), 0.0, 1e-12);
    EXPECT_NEAR(f.r_tm.real(), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(f.r_tm.imag(), 0.0, 1e-12);
    EXPECT_NEAR(f.t_te.real(), 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(f.cos_theta_t.real(), 1.0, 1e-12);
}

TEST(Fresnel, BrewsterZero) {
    const double theta_b = std::atan(1.5);
    const auto f = strikecem::fresnel({}, lossless(1.5), std::cos(theta_b));
    EXPECT_LT(std::abs(f.r_tm), 1e-9);
    EXPECT_GT(std::abs(f.r_te), 0.0);
}

TEST(Fresnel, PecLimit) {
    const strikecem::ComplexMedium metal{{1.0, -1e8}, {1.0, 0.0}};
    const auto f = strikecem::fresnel({}, metal, std::cos(0.3));
    EXPECT_LT(std::abs(f.r_te + 1.0), 1e-3);
    EXPECT_LT(std::abs(f.r_tm - 1.0), 1e-3);
}

TEST(Fresnel, LosslessEnergy) {
    const double ci = std::cos(0.6);
    const auto f = strikecem::fresnel({}, lossless(1.5), ci);
    const double pref = 1.5 * f.cos_theta_t.real() / ci;
    EXPECT_NEAR(std::norm(f.r_te) + pref * std::norm(f.t_te), 1.0, 1e-12);
    EXPECT_NEAR(std::norm(f.r_tm) + pref * std::norm(f.t_tm), 1.0, 1e-12);
}

TEST(Fresnel, TotalInternalReflection) {
    const auto f = strikecem::fresnel(lossless(1.5), {}, std::cos(1.0));
    EXPECT_NEAR(std::abs(f.r_te), 1.0, 1e-12);
    EXPECT_NEAR(std::abs(f.r_tm), 1.0, 1e-12);
    EXPECT_LT(f.cos_theta_t.imag(), 0.0);
}

TEST(Fresnel, DecayingBranchInvariant) {
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
    EXPECT_THROW(strikecem::fresnel({}, {{{1.0, 1.0}}, {{1.0, 0.0}}}, 1.0),
                 std::invalid_argument);
    EXPECT_THROW(strikecem::fresnel({}, lossless(1.5), 2.0), std::invalid_argument);
    const auto f = strikecem::fresnel({}, {{2.0, -0.5}, {1.0, 0.0}}, std::cos(0.6));
    EXPECT_LE(std::abs(f.r_te), 1.0 + 1e-12);
    EXPECT_LE(std::abs(f.r_tm), 1.0 + 1e-12);
}

TEST(Fresnel, ComplexPathMatchesReal) {
    const strikecem::ComplexMedium air;
    const strikecem::ComplexMedium glass{{2.25, 0.0}, {1.0, 0.0}};
    for (double cos_i : {1.0, 0.8, 0.2}) {
        const auto a = strikecem::fresnel(air, glass, cos_i);
        const auto b = strikecem::fresnel(air, glass, std::complex<double>{cos_i, 0.0});
        EXPECT_NEAR(std::abs(a.r_te - b.r_te), 0.0, 1e-12);
        EXPECT_NEAR(std::abs(a.r_tm - b.r_tm), 0.0, 1e-12);
        EXPECT_NEAR(std::abs(a.t_te - b.t_te), 0.0, 1e-12);
        EXPECT_NEAR(std::abs(a.cos_theta_t - b.cos_theta_t), 0.0, 1e-12);
    }
    const auto tir_a = strikecem::fresnel(glass, air, std::cos(1.0));
    const auto tir_b =
        strikecem::fresnel(glass, air, std::complex<double>{std::cos(1.0), 0.0});
    EXPECT_NEAR(std::abs(tir_a.r_te - tir_b.r_te), 0.0, 1e-12);
    EXPECT_THROW(strikecem::fresnel(air, glass,
                                      std::complex<double>{
                                          1.0, std::numeric_limits<double>::infinity()}),
                 std::invalid_argument);
}

}

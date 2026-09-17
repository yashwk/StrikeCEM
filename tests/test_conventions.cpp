// v1 conventions tests (CONVENTIONS.md): directions, poles, transforms,
// polarization bases, circular conversion, RCS/zero policy.
#include <cmath>
#include <gtest/gtest.h>

#include "strikecem/core/Conventions.hpp"

using strikecem::az_el_from_direction;
using strikecem::direction_from_az_el;

namespace {

constexpr double kTol = 1e-12;

TEST(Directions, CardinalAxes) {
    auto v = direction_from_az_el(0.0, 0.0);
    EXPECT_NEAR(v.x, 1.0, kTol);
    EXPECT_NEAR(v.y, 0.0, kTol);
    EXPECT_NEAR(v.z, 0.0, kTol);
    v = direction_from_az_el(90.0, 0.0);
    EXPECT_NEAR(v.x, 0.0, kTol);
    EXPECT_NEAR(v.y, 1.0, kTol);
    v = direction_from_az_el(0.0, 90.0);
    EXPECT_NEAR(v.z, 1.0, kTol);
    v = direction_from_az_el(180.0, -90.0);
    EXPECT_NEAR(v.x, 0.0, kTol);
    EXPECT_NEAR(v.z, -1.0, kTol);
}

TEST(Directions, UnitLengthEverywhere) {
    for (double az = 0.0; az < 360.0; az += 7.0)
        for (double el = -90.0; el <= 90.0; el += 5.0)
            EXPECT_NEAR(direction_from_az_el(az, el).length(), 1.0, kTol);
}

TEST(Directions, RoundTrip) {
    for (double az = 0.0; az < 360.0; az += 13.0) {
        for (double el = -90.0; el <= 90.0; el += 11.0) {
            double rt_az = -1.0, rt_el = -999.0;
            az_el_from_direction(direction_from_az_el(az, el), rt_az, rt_el);
            EXPECT_NEAR(rt_el, el, 1e-9);
            if (std::abs(el) < 90.0 - 1e-9) EXPECT_NEAR(rt_az, az, 1e-9);
        }
    }
}

TEST(Directions, NormalizeAzimuth) {
    EXPECT_DOUBLE_EQ(strikecem::normalize_azimuth(360.0), 0.0);
    EXPECT_DOUBLE_EQ(strikecem::normalize_azimuth(-90.0), 270.0);
    EXPECT_DOUBLE_EQ(strikecem::normalize_azimuth(720.0), 0.0);
    EXPECT_DOUBLE_EQ(strikecem::normalize_azimuth(0.0), 0.0);
}

TEST(Polarization, OrthonormalBasis) {
    for (double az = 0.0; az < 360.0; az += 17.0) {
        for (double el = -90.0; el <= 90.0; el += 9.0) {
            const auto k = direction_from_az_el(az, el);
            const auto basis = strikecem::polarization_basis(k);
            EXPECT_NEAR(basis.v.length(), 1.0, kTol);
            EXPECT_NEAR(basis.h.length(), 1.0, kTol);
            EXPECT_NEAR(geom::dot(basis.v, basis.h), 0.0, kTol);
            EXPECT_NEAR(geom::dot(basis.v, k), 0.0, kTol);
            EXPECT_NEAR(geom::dot(basis.h, k), 0.0, kTol);
        }
    }
}

TEST(Polarization, PoleReferenceIsDeterministic) {
    const auto north = strikecem::polarization_basis(direction_from_az_el(0.0, 90.0));
    EXPECT_NEAR(north.v.x, 1.0, kTol);
    EXPECT_NEAR(north.v.y, 0.0, kTol);
    EXPECT_NEAR(north.v.z, 0.0, kTol);
    // h = v x k = x cross z = -y.
    EXPECT_NEAR(north.h.y, -1.0, kTol);
    const auto from_zero = strikecem::polarization_basis(direction_from_az_el(0.0, 90.0));
    const auto from_other = strikecem::polarization_basis(direction_from_az_el(270.0, 90.0));
    EXPECT_NEAR(from_zero.v.x, from_other.v.x, kTol);
    EXPECT_NEAR(from_zero.h.y, from_other.h.y, kTol);
}

TEST(Polarization, SphereHasNullCoCircularReturn) {
    // Odd-bounce reference: handedness flips, so RR = LL = 0, RL = LR = 1.
    const strikecem::Complex2x2 sphere = {
        std::array<std::complex<double>, 2>{{1.0, 0.0}},
        std::array<std::complex<double>, 2>{{0.0, 1.0}}};
    const auto circ = strikecem::linear_to_circular(sphere);
    EXPECT_NEAR(std::abs(circ[0][0]), 0.0, kTol);
    EXPECT_NEAR(std::abs(circ[1][1]), 0.0, kTol);
    EXPECT_NEAR(std::abs(circ[0][1]), 1.0, kTol);
    EXPECT_NEAR(std::abs(circ[1][0]), 1.0, kTol);
}

TEST(Polarization, CircularConversionPreservesPower) {
    const strikecem::Complex2x2 s = {
        std::array<std::complex<double>, 2>{{{0.7, 0.1}, {-0.2, 0.3}}},
        std::array<std::complex<double>, 2>{{{0.4, -0.5}, {0.9, 0.2}}}};
    const auto circ = strikecem::linear_to_circular(s);
    double lin = 0.0, cir = 0.0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            lin += std::norm(s[i][j]);
            cir += std::norm(circ[i][j]);
        }
    EXPECT_NEAR(lin, cir, kTol);
}

TEST(Rcs, ConversionsAndZeroPolicy) {
    EXPECT_DOUBLE_EQ(strikecem::rcs_sqm_to_dbsm(1.0), 0.0);
    EXPECT_NEAR(strikecem::rcs_sqm_to_dbsm(10.0), 10.0, kTol);
    EXPECT_EQ(strikecem::rcs_sqm_to_dbsm(0.0), -std::numeric_limits<double>::infinity());
    EXPECT_THROW(strikecem::rcs_sqm_to_dbsm(-1.0), std::invalid_argument);
    EXPECT_DOUBLE_EQ(strikecem::rcs_dbsm_to_sqm(-std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_NEAR(strikecem::rcs_dbsm_to_sqm(0.0), 1.0, kTol);
}

TEST(Units, ScaleFactors) {
    EXPECT_DOUBLE_EQ(strikecem::units_to_metres("m"), 1.0);
    EXPECT_DOUBLE_EQ(strikecem::units_to_metres("mm"), 1e-3);
    EXPECT_DOUBLE_EQ(strikecem::units_to_metres("in"), 0.0254);
    EXPECT_THROW(strikecem::units_to_metres("cm"), std::invalid_argument);
}

TEST(Transform, ScaleRotateTranslateOrder) {
    // p=(0,1,0), scale 2 -> (0,2,0); Rx(90): y->z -> (0,0,2); +t(1,0,0).
    const auto q = strikecem::apply_transform({0.0, 1.0, 0.0}, 2.0, {90.0, 0.0, 0.0},
                                              {1.0, 0.0, 0.0});
    EXPECT_NEAR(q.x, 1.0, kTol);
    EXPECT_NEAR(q.y, 0.0, kTol);
    EXPECT_NEAR(q.z, 2.0, kTol);
    // Identity transform is a fixed point.
    const auto same = strikecem::apply_transform({1.0, 2.0, 3.0}, 1.0, {0.0, 0.0, 0.0},
                                                 {0.0, 0.0, 0.0});
    EXPECT_NEAR(same.x, 1.0, kTol);
    EXPECT_NEAR(same.y, 2.0, kTol);
    EXPECT_NEAR(same.z, 3.0, kTol);
    // Rz(90) takes +x toward +y (azimuth convention).
    const auto r = strikecem::apply_transform({1.0, 0.0, 0.0}, 1.0, {0.0, 0.0, 90.0},
                                              {0.0, 0.0, 0.0});
    EXPECT_NEAR(r.x, 0.0, kTol);
    EXPECT_NEAR(r.y, 1.0, kTol);
}

} // namespace

#include <cmath>
#include <numbers>
#include <gtest/gtest.h>

#include "strikecem/solvers/Material.hpp"

namespace {

strikecem::MaterialModel dielectric_single() {
    strikecem::MaterialModel m;
    m.tag = "glass";
    m.type = strikecem::WallType::Dielectric;
    m.table.push_back({1e9, {{4.0, 0.0}, {1.0, 0.0}}, 0.0});
    return m;
}

TEST(Material, SingleEntryIsConstant) {
    const auto m = dielectric_single();
    EXPECT_NO_THROW(strikecem::validate_material(m));
    for (double f : {1e6, 1e9, 18e9}) {
        const auto e = strikecem::evaluate_material(m, f);
        EXPECT_DOUBLE_EQ(e.eps_r.real(), 4.0);
        EXPECT_DOUBLE_EQ(e.eps_r.imag(), 0.0);
    }
}

TEST(Material, LinearMidpointExact) {
    strikecem::MaterialModel m;
    m.tag = "interp";
    m.type = strikecem::WallType::Dielectric;
    m.table.push_back({1e9, {{2.0, -0.2}, {1.0, 0.0}}, 0.0});
    m.table.push_back({3e9, {{4.0, -0.6}, {1.0, 0.0}}, 2.0});
    const auto e = strikecem::evaluate_material(m, 2e9);
    EXPECT_DOUBLE_EQ(e.eps_r.real(), 3.0);
    EXPECT_DOUBLE_EQ(e.mu_r.real(), 1.0);
    const double eps0 = 1.0 / (376.73031346177066 * 299792458.0);
    const double folded = 1.0 / (2.0 * std::numbers::pi * 2e9 * eps0);
    EXPECT_DOUBLE_EQ(e.eps_r.imag(), -0.4 - folded);
}

TEST(Material, RangeAndOrderEnforced) {
    auto m = dielectric_single();
    m.table.push_back({0.5e9, {{4.0, 0.0}, {1.0, 0.0}}, 0.0});
    EXPECT_THROW(strikecem::validate_material(m), std::invalid_argument);
    auto ok = dielectric_single();
    ok.table.push_back({2e9, {{4.0, 0.0}, {1.0, 0.0}}, 0.0});
    EXPECT_THROW(strikecem::evaluate_material(ok, 0.5e9), std::invalid_argument);
    EXPECT_THROW(strikecem::evaluate_material(ok, 3e9), std::invalid_argument);
    EXPECT_THROW(strikecem::evaluate_material(ok, 0.0), std::invalid_argument);
    EXPECT_NO_THROW(strikecem::evaluate_material(ok, 1e9));
    EXPECT_NO_THROW(strikecem::evaluate_material(ok, 2e9));
}

TEST(Material, ShapePolicy) {
    EXPECT_THROW(strikecem::validate_material({}), std::invalid_argument);
    strikecem::MaterialModel pec;
    pec.tag = "metal";
    pec.type = strikecem::WallType::Pec;
    EXPECT_NO_THROW(strikecem::validate_material(pec));
    EXPECT_THROW(strikecem::evaluate_material(pec, 1e9), std::invalid_argument);
    auto pec_table = pec;
    pec_table.table.push_back({1e9, {{1.0, 0.0}, {1.0, 0.0}}, 0.0});
    EXPECT_THROW(strikecem::validate_material(pec_table), std::invalid_argument);
    strikecem::MaterialModel coated;
    coated.tag = "paint";
    coated.type = strikecem::WallType::Coated;
    EXPECT_THROW(strikecem::validate_material(coated), std::invalid_argument);
    coated.layers.push_back({0.001, {{3.0, -0.1}, {1.0, 0.0}}});
    EXPECT_NO_THROW(strikecem::validate_material(coated));
    coated.layers.push_back({0.0, {{3.0, 0.0}, {1.0, 0.0}}});
    EXPECT_THROW(strikecem::validate_material(coated), std::invalid_argument);
    auto mixed = dielectric_single();
    mixed.layers.push_back({0.001, {{3.0, 0.0}, {1.0, 0.0}}});
    EXPECT_THROW(strikecem::validate_material(mixed), std::invalid_argument);
}

TEST(Material, GainRejected) {
    auto m = dielectric_single();
    m.table[0].medium.eps_r = {4.0, 0.5};
    EXPECT_THROW(strikecem::validate_material(m), std::invalid_argument);
    auto neg = dielectric_single();
    neg.table[0].sigma = -1.0;
    EXPECT_THROW(strikecem::validate_material(neg), std::invalid_argument);
}

TEST(Material, ConductivityFoldsWithSign) {
    strikecem::MaterialEntry e{1e9, {{1.0, 0.0}, {1.0, 0.0}}, 1.0};
    const auto m = strikecem::effective_medium(e, 1e9);
    EXPECT_DOUBLE_EQ(m.eps_r.real(), 1.0);
    EXPECT_LT(m.eps_r.imag(), 0.0);
    const double eps0 = 1.0 / (376.73031346177066 * 299792458.0);
    EXPECT_DOUBLE_EQ(m.eps_r.imag(), -1.0 / (2.0 * std::numbers::pi * 1e9 * eps0));
}

TEST(Material, BarePecEmptyStack) {
    const auto r = strikecem::coated_pec_reflection({}, {}, 1.0, 1.0);
    EXPECT_DOUBLE_EQ(r.r_te.real(), -1.0);
    EXPECT_DOUBLE_EQ(r.r_te.imag(), 0.0);
    EXPECT_DOUBLE_EQ(r.r_tm.real(), 1.0);
    EXPECT_DOUBLE_EQ(r.r_tm.imag(), 0.0);
}

TEST(Material, HalfWaveAbsentee) {
    const strikecem::CoatingLayer layer{0.25, {{4.0, 0.0}, {1.0, 0.0}}};
    const auto r = strikecem::coated_pec_reflection({}, {layer}, 1.0, 1.0);
    EXPECT_NEAR(std::abs(r.r_te + 1.0), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(r.r_tm - 1.0), 0.0, 1e-12);
}

TEST(Material, ThinDegenerate) {
    const strikecem::CoatingLayer layer{1e-6, {{6.25, -0.09}, {1.0, 0.0}}};
    const auto r = strikecem::coated_pec_reflection({}, {layer}, std::cos(0.3), 1.0);
    EXPECT_LT(std::abs(r.r_te + 1.0), 1e-3);
    EXPECT_LT(std::abs(r.r_tm - 1.0), 1e-3);
}

TEST(Material, AsymmetricTwoLayer) {
    const strikecem::CoatingLayer outer{1.0 / 8.0, {{4.0, 0.0}, {1.0, 0.0}}};
    const strikecem::CoatingLayer inner{1.0 / 24.0, {{9.0, 0.0}, {1.0, 0.0}}};
    const auto r = strikecem::coated_pec_reflection({}, {outer, inner}, 1.0, 1.0);
    EXPECT_NEAR(r.r_te.real(), -0.28, 1e-9);
    EXPECT_NEAR(r.r_te.imag(), -0.96, 1e-9);
    EXPECT_NEAR(r.r_tm.real(), 0.28, 1e-9);
    EXPECT_NEAR(r.r_tm.imag(), 0.96, 1e-9);
    const auto swapped = strikecem::coated_pec_reflection({}, {inner, outer}, 1.0, 1.0);
    EXPECT_GT(std::abs(swapped.r_te - r.r_te), 0.1);
}

TEST(Material, LosslessUnitMagnitude) {
    const strikecem::CoatingLayer a{0.13, {{4.0, 0.0}, {1.0, 0.0}}};
    const strikecem::CoatingLayer b{0.07, {{2.25, 0.0}, {1.0, 0.0}}};
    for (double cos_i : {1.0, 0.8, 0.4}) {
        const auto r = strikecem::coated_pec_reflection({}, {a, b}, cos_i, 1.0);
        EXPECT_NEAR(std::abs(r.r_te), 1.0, 1e-9);
        EXPECT_NEAR(std::abs(r.r_tm), 1.0, 1e-9);
    }
    const strikecem::CoatingLayer lossy{0.13, {{4.0, -1.0}, {1.0, 0.0}}};
    const auto rl = strikecem::coated_pec_reflection({}, {lossy}, 0.9, 1.0);
    EXPECT_LE(std::abs(rl.r_te), 1.0 + 1e-12);
    EXPECT_LE(std::abs(rl.r_tm), 1.0 + 1e-12);
    EXPECT_TRUE(std::isfinite(rl.r_te.real() + rl.r_tm.imag()));
}

TEST(Material, CoatingBadInputsThrow) {
    const strikecem::CoatingLayer good{0.1, {{4.0, 0.0}, {1.0, 0.0}}};
    EXPECT_THROW(strikecem::coated_pec_reflection({}, {good}, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(strikecem::coated_pec_reflection({}, {good}, 2.0, 1.0), std::invalid_argument);
    const strikecem::CoatingLayer flat{0.0, {{4.0, 0.0}, {1.0, 0.0}}};
    EXPECT_THROW(strikecem::coated_pec_reflection({}, {flat}, 1.0, 1.0), std::invalid_argument);
    const strikecem::CoatingLayer gain{0.1, {{4.0, 0.5}, {1.0, 0.0}}};
    EXPECT_THROW(strikecem::coated_pec_reflection({}, {gain}, 1.0, 1.0), std::invalid_argument);
}

} // namespace

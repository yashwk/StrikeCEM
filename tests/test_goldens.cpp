// Analytic golden tests (SPEC section 7.1): flat-plate broadside peak and
// sinc envelope, rotated-plate pattern rotation, and the electrically
// large sphere PO asymptote pi*r^2.
#include <cmath>
#include <gtest/gtest.h>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace {

constexpr double kPi = 3.141592653589793;

std::string fixture(const std::string& name) { return std::string(SCEM_FIXTURE_DIR) + "/" + name; }

struct Case {
    strikecem::ResolvedConfig rc;
    strikecem::NormalizedMesh mesh;
};

Case load_case(const std::string& name) {
    Case c;
    c.rc = strikecem::load_config(fixture(name), SCEM_SCHEMA_PATH);
    c.mesh = strikecem::load_normalized_mesh(c.rc.value, SCEM_FIXTURE_DIR, c.rc.schema_version);
    return c;
}

strikecem::SamplePlan make_plan(const std::vector<double>& freqs,
                                const std::vector<std::pair<double, double>>& dirs,
                                const std::vector<std::string>& pols) {
    strikecem::SamplePlan plan;
    plan.frequencies_hz = freqs;
    uint32_t id = 0;
    for (const auto& [az, el] : dirs)
        plan.directions.push_back(
            {id++, az, el, strikecem::direction_from_az_el(az, el)});
    plan.polarizations = pols;
    return plan;
}

double solve_rcs(const Case& c, double freq, double az, double el, const std::string& pol) {
    const auto plan = make_plan({freq}, {{az, el}}, {pol});
    const auto result = strikecem::solve_po(c.mesh, plan, c.rc.value);
    return result.samples[0].rcs_sqm;
}

double sinc(double u) {
    if (u == 0.0) return 1.0;
    return std::sin(kPi * u) / (kPi * u);
}

TEST(Goldens, PlateBroadsidePeakIsExact) {
    // Flat plate: every facet shares the broadside phase, so the discrete
    // sum reproduces 4*pi*A^2/lambda^2 to floating-point precision.
    const Case c = load_case("valid_minimal.json");
    const double freq = 300e6;
    const double lambda = strikecem::kSpeedOfLight / freq;
    const double expected = 4.0 * kPi * 1.0 / (lambda * lambda); // A = 1 m^2
    EXPECT_NEAR(solve_rcs(c, freq, 0.0, -90.0, "HH"), expected, 1e-9 * expected);
    EXPECT_NEAR(solve_rcs(c, freq, 0.0, -90.0, "VV"), expected, 1e-9 * expected);
}

TEST(Goldens, PlateSincEnvelope) {
    // Off-broadside ratio sigma(theta)/sigma(0) = cos^2(theta) *
    // sinc^2(k*a*sin(theta)/pi) for the square plate (derived in review).
    // Documented discretization tolerance: 1% at 5 deg, 2% at 10 deg for
    // lambda/10 facets at 300 MHz.
    const Case c = load_case("valid_plate_fine.json");
    const double freq = 300e6;
    const double k = 2.0 * kPi * freq / strikecem::kSpeedOfLight;
    const double broadside = solve_rcs(c, freq, 0.0, -90.0, "HH");
    for (const auto& [theta_deg, tol] : {std::pair{5.0, 0.01}, {10.0, 0.02}}) {
        const double theta = theta_deg * kPi / 180.0;
        const double expected_ratio =
            std::cos(theta) * std::cos(theta) * std::pow(sinc(k * std::sin(theta) / kPi), 2);
        const double measured = solve_rcs(c, freq, 0.0, -90.0 + theta_deg, "HH") / broadside;
        EXPECT_NEAR(measured, expected_ratio, tol * expected_ratio)
            << "theta=" << theta_deg << "deg";
    }
}

TEST(Goldens, RotatedPlatePatternRotates) {
    // Ry(90) maps the plate normal +z to +x; the monostatic return at
    // k_hat=-x must equal the unrotated return at k_hat=-z.
    const Case plain = load_case("valid_minimal.json");
    const Case rotated = load_case("valid_rotated.json");
    const double freq = 300e6;
    for (const std::string& pol : {"HH", "VV"}) {
        const double ref = solve_rcs(plain, freq, 0.0, -90.0, pol);
        const double got = solve_rcs(rotated, freq, 180.0, 0.0, pol);
        EXPECT_NEAR(got, ref, 1e-9 * ref) << "pol=" << pol;
    }
}

TEST(Goldens, SphereApproachesPiRSquared) {
    // Electrically large sphere (ka ~ 21 at 1 GHz, r = 1 m) against the
    // high-frequency PO asymptote, not unconditional Mie equality.
    // Facets are ~lambda/7; measured +3.9%, bound documents that
    // discretization with margin.
    const Case c = load_case("valid_sphere.json");
    const double freq = 1e9;
    const double hh = solve_rcs(c, freq, 0.0, 0.0, "HH");
    const double vv = solve_rcs(c, freq, 0.0, 0.0, "VV");
    EXPECT_NEAR(hh, vv, 0.01 * hh); // rotational symmetry about x
    EXPECT_GT(hh, 0.95 * kPi);
    EXPECT_LT(hh, 1.05 * kPi);
}

} // namespace

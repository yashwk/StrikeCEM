// 3D fringe geometry tests (ADR-0003 slice C): face0_dir field, rim
// frame at broadside/edge-on, integral vs quadrature, sinc nulls,
// bounds, and deferred-cone flags. No amplitude physics yet.
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/EdgeFringe.hpp"
#include "strikecem/solvers/EdgeModel.hpp"
#include "strikecem/solvers/UtdWedge.hpp"

namespace {

constexpr double kPi = 3.141592653589793;

std::string fixture(const std::string& name) { return std::string(SCEM_FIXTURE_DIR) + "/" + name; }

strikecem::NormalizedMesh load_mesh(const std::string& name) {
    const std::string config = fixture(name);
    auto rc = strikecem::load_config(config, SCEM_SCHEMA_PATH);
    return strikecem::load_normalized_mesh(rc.value, SCEM_FIXTURE_DIR, rc.schema_version);
}

geom::Vec3d unit(double x, double y, double z) {
    geom::Vec3d v(x, y, z);
    v.normalize();
    return v;
}

strikecem::MeshEdge synthetic_edge(double length) {
    strikecem::MeshEdge e;
    e.p0 = {0, 0, 0};
    e.p1 = {length, 0, 0};
    e.tangent = {1, 0, 0};
    e.length = length;
    e.n0 = e.n1 = {0, 0, 1};
    e.face0_dir = {0, 1, 0};
    e.boundary = true;
    return e;
}

TEST(Fringe, Face0DirPointsIntoFace) {
    for (const char* name : {"valid_minimal.json", "valid_cube.json", "valid_dihedral.json"}) {
        const auto mesh = load_mesh(name);
        const auto model = strikecem::extract_edges(mesh);
        ASSERT_FALSE(model.edges.empty()) << name;
        for (const auto& e : model.edges) {
            EXPECT_NEAR(e.face0_dir.length(), 1.0, 1e-12) << name;
            EXPECT_NEAR(geom::dot(e.face0_dir, e.tangent), 0.0, 1e-12) << name;
            // Into face tri0: positive dot with centroid-minus-midpoint.
            const auto& ft = mesh.triangles[static_cast<uint32_t>(e.tri0)];
            const geom::Vec3d c =
                (mesh.vertices[ft[0]] + mesh.vertices[ft[1]] + mesh.vertices[ft[2]]) * (1.0 / 3.0);
            const geom::Vec3d m = (e.p0 + e.p1) * 0.5;
            EXPECT_GT(geom::dot(e.face0_dir, c - m), 0.0) << name;
        }
    }
}

TEST(Fringe, BroadsideBackscatterSymmetric) {
    // Plate in z=0, wave traveling -z, observer +z: every rim sees the
    // same transverse geometry, integral = L (centers are in-plane).
    const auto mesh = load_mesh("valid_minimal.json");
    const auto model = strikecem::extract_edges(mesh);
    ASSERT_EQ(model.edges.size(), 4u);
    const geom::Vec3d s(0, 0, -1), r(0, 0, 1);
    for (const auto& e : model.edges) {
        const auto ang = strikecem::edge_transverse_angles(e, s, r);
        ASSERT_TRUE(ang.valid);
        EXPECT_NEAR(ang.sin_beta0, 1.0, 1e-12);
        EXPECT_NEAR(ang.phi, ang.phi_prime, 1e-12); // backscatter: same ray
        const auto integral = strikecem::along_edge_integral(e, 2.0 * kPi, s, r);
        EXPECT_NEAR(integral.real(), 1.0, 1e-12);
        EXPECT_NEAR(integral.imag(), 0.0, 1e-12);
    }
}

TEST(Fringe, IntegralMatchesQuadrature) {
    // Independent cross-check: closed-form sinc vs brute-force Simpson of
    // the line integral (pins phase convention and null structure).
    const auto e = synthetic_edge(1.7);
    const double k = 4.0;
    const geom::Vec3d s(0, 0, -1), r = unit(0.6, 0.0, 0.8);
    const geom::Vec3d d = r - s;
    const geom::Vec3d c = (e.p0 + e.p1) * 0.5;
    const int n = 4096; // even
    const double h = e.length / n;
    std::complex<double> num{0.0, 0.0};
    for (int i = 0; i <= n; ++i) {
        const double u = -e.length / 2.0 + i * h;
        const double ph = k * geom::dot(d, c + e.tangent * u);
        const std::complex<double> f(std::cos(ph), std::sin(ph));
        const double w = (i == 0 || i == n) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
        num += (h / 3.0) * w * f;
    }
    const auto closed = strikecem::along_edge_integral(e, k, s, r);
    EXPECT_LT(std::abs(num - closed), 1e-9 * e.length);
}

TEST(Fringe, SincNullAtPredictedAngle) {
    // Monostatic, kL/2 * a = pi with a = (r-s).t = 1: first sinc null.
    const auto e = synthetic_edge(1.0);
    const double k = 2.0 * kPi; // L = lambda
    const geom::Vec3d s = unit(-0.5, 0.0, -std::sqrt(3.0) / 2.0);
    const geom::Vec3d r = s * -1.0;
    const auto integral = strikecem::along_edge_integral(e, k, s, r);
    EXPECT_LT(std::abs(integral), 1e-9 * e.length);
}

TEST(Fringe, BoundAndLengthScaling) {
    const auto e1 = synthetic_edge(1.0);
    const auto e2 = synthetic_edge(2.0);
    const geom::Vec3d s(0, 0, -1);
    for (const auto r :
         {geom::Vec3d(0, 0, 1), unit(0.6, 0.0, 0.8), unit(-0.5, 0.5, 0.7071067811865476)}) {
        EXPECT_LE(std::abs(strikecem::along_edge_integral(e1, 2.0 * kPi, s, r)),
                  e1.length * (1.0 + 1e-12));
        // Broadside doubling: same center phase, sinc = 1 both times.
        const geom::Vec3d rb(0, 0, 1);
        if (r.x == 0.0 && r.y == 0.0) {
            const auto i1 = strikecem::along_edge_integral(e1, 2.0 * kPi, s, rb);
            const auto i2 = strikecem::along_edge_integral(e2, 2.0 * kPi, s, rb);
            EXPECT_NEAR(i2.real(), 2.0 * i1.real(), 1e-12);
        }
        (void)rb;
    }
}

TEST(Fringe, EndOnInvalidButIntegralDefined) {
    const auto mesh = load_mesh("valid_minimal.json");
    const auto model = strikecem::extract_edges(mesh);
    const geom::Vec3d s(1, 0, 0), r(-1, 0, 0); // travel along +x
    bool found_end_on = false;
    for (const auto& e : model.edges) {
        if (std::abs(e.tangent.x) < 0.9) continue;
        EXPECT_FALSE(strikecem::edge_transverse_angles(e, s, r).valid);
        const auto integral = strikecem::along_edge_integral(e, 2.0 * kPi, s, r);
        EXPECT_TRUE(std::isfinite(integral.real() + integral.imag()));
        EXPECT_LE(std::abs(integral), e.length * (1.0 + 1e-12));
        found_end_on = true;
    }
    EXPECT_TRUE(found_end_on);
}

TEST(Fringe, NonRimInvalidButIntegralDefined) {
    const auto mesh = load_mesh("valid_cube.json");
    const auto model = strikecem::extract_edges(mesh);
    ASSERT_EQ(model.edges.size(), 12u);
    const geom::Vec3d s(0, 0, -1), r(0, 0, 1);
    for (const auto& e : model.edges) {
        EXPECT_FALSE(strikecem::edge_transverse_angles(e, s, r).valid) << "n=1.5 deferred";
        EXPECT_TRUE(std::isfinite(strikecem::along_edge_integral(e, 2.0 * kPi, s, r).real()));
    }
}

TEST(Fringe, EdgeOnPlateAssembly) {
    // Plate-rim structural benchmark (ADR-0003 order: plate first).
    // Incidence in-plane: PO lights nothing (all n.r_hat = 0, and the
    // solver's hard-shadow rule needs n.r_hat > 0), while the two rims
    // transverse to incidence stay valid with finite KP coefficients —
    // the fringe is the entire return there.
    const auto mesh = load_mesh("valid_minimal.json");
    const auto model = strikecem::extract_edges(mesh);
    const geom::Vec3d s(1, 0, 0), r(-1, 0, 0);
    for (const auto& n : mesh.normals) EXPECT_LE(geom::dot(n, r), 0.0);
    const double k = 2.0 * kPi;
    int valid = 0;
    bool saw_zero = false, saw_pi = false;
    for (const auto& e : model.edges) {
        const auto ang = strikecem::edge_transverse_angles(e, s, r);
        if (!ang.valid) continue;
        ++valid;
        EXPECT_NEAR(ang.phi, ang.phi_prime, 1e-12);
        if (std::abs(ang.phi) < 1e-9) saw_zero = true;
        if (std::abs(ang.phi - kPi) < 1e-9) saw_pi = true;
        // Grazing shadow-boundary angles through the validated 2D core.
        const auto d = strikecem::utd_coefficient(k, 1.0, 2.0, ang.phi, ang.phi_prime);
        EXPECT_TRUE(std::isfinite(d.soft.real() + d.soft.imag() + d.hard.real() + d.hard.imag()));
        const auto integral = strikecem::along_edge_integral(e, k, s, r);
        EXPECT_TRUE(std::isfinite(integral.real() + integral.imag()));
    }
    EXPECT_EQ(valid, 2); // y-rims diffract; x-rims are end-on (flagged)
    EXPECT_TRUE(saw_zero && saw_pi); // leading/trailing rim pair
}

TEST(Fringe, BadInputsThrow) {
    const auto e = synthetic_edge(1.0);
    const geom::Vec3d s(0, 0, -1), r(0, 0, 1), bad(0, 0, 2);
    EXPECT_THROW(strikecem::along_edge_integral(e, 0.0, s, r), std::invalid_argument);
    EXPECT_THROW(strikecem::along_edge_integral(e, 2.0 * kPi, bad, r), std::invalid_argument);
    EXPECT_THROW(strikecem::edge_transverse_angles(e, bad, r), std::invalid_argument);
    strikecem::MeshEdge zero;
    EXPECT_THROW(strikecem::along_edge_integral(zero, 2.0 * kPi, s, r), std::invalid_argument);
}

} // namespace

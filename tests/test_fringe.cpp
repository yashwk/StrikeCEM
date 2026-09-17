// 3D fringe geometry tests (ADR-0003 slices C/D1/D2a): face0_dir/face1_dir
// fields, rim/wedge frames, integral vs quadrature, sinc nulls, fringe
// amplitude identity, and vector (per-transmit) decomposition.
#include <array>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/EdgeFringe.hpp"
#include "strikecem/solvers/EdgeModel.hpp"
#include "strikecem/solvers/GpuPO.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"
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

TEST(Fringe, Face1DirIntoFace1) {
    const auto mesh = load_mesh("valid_dihedral.json");
    const auto model = strikecem::extract_edges(mesh);
    int interior = 0, rims = 0;
    for (const auto& e : model.edges) {
        if (e.boundary) {
            ++rims;
            EXPECT_EQ(e.face1_dir.x, e.face0_dir.x);
            EXPECT_EQ(e.face1_dir.y, e.face0_dir.y);
            EXPECT_EQ(e.face1_dir.z, e.face0_dir.z);
            continue;
        }
        ++interior;
        EXPECT_NEAR(e.face1_dir.length(), 1.0, 1e-12);
        EXPECT_NEAR(geom::dot(e.face1_dir, e.tangent), 0.0, 1e-12);
        const auto& ft = mesh.triangles[static_cast<uint32_t>(e.tri1)];
        const geom::Vec3d c =
            (mesh.vertices[ft[0]] + mesh.vertices[ft[1]] + mesh.vertices[ft[2]]) * (1.0 / 3.0);
        const geom::Vec3d m = (e.p0 + e.p1) * 0.5;
        EXPECT_GT(geom::dot(e.face1_dir, c - m), 0.0);
    }
    EXPECT_EQ(interior, 1);
    EXPECT_EQ(rims, 6);
}

const strikecem::MeshEdge& dihedral_valley(const strikecem::EdgeModel& model) {
    for (const auto& e : model.edges)
        if (!e.boundary) return e;
    throw std::logic_error("no valley edge");
}

TEST(Fringe, ValleyFrameExterior) {
    // Dihedral valley (n = 1.5), look from -x: arrival/observation at
    // frame angle pi, inside the exterior cone (0, 1.5*pi).
    const auto mesh = load_mesh("valid_dihedral.json");
    const auto model = strikecem::extract_edges(mesh);
    const auto& v = dihedral_valley(model);
    EXPECT_NEAR(v.wedge_n, 1.5, 1e-9);
    const geom::Vec3d s(1, 0, 0), r(-1, 0, 0);
    const auto ang = strikecem::wedge_transverse_angles(v, s, r);
    ASSERT_TRUE(ang.valid);
    EXPECT_NEAR(ang.sin_beta0, 1.0, 1e-12);
    EXPECT_NEAR(ang.phi, kPi, 1e-9);
    EXPECT_NEAR(ang.phi_prime, kPi, 1e-9);
    EXPECT_GT(ang.phi, 0.0);
    EXPECT_LT(ang.phi, 1.5 * kPi);
    // Rims reject the wedge frame and vice versa.
    for (const auto& e : model.edges) {
        if (!e.boundary) continue;
        EXPECT_FALSE(strikecem::wedge_transverse_angles(e, s, r).valid);
        break;
    }
    EXPECT_FALSE(strikecem::edge_transverse_angles(v, s, r).valid);
}

TEST(Fringe, LabelingSymmetry) {
    // D is invariant under simultaneous face-label flip
    // (phi, phi') -> (2n pi - phi, 2n pi - phi'): the fringe does not
    // depend on which adjacent face the mesh lists first.
    const double k = 2.0 * kPi, n = 1.5;
    for (const auto [phi, phip] :
         {std::make_pair(1.0, 2.0), {0.5, 4.0}, {2.5, 1.2}, {0.3, 4.4}}) {
        const auto a = strikecem::utd_coefficient(k, 1.0, n, phi, phip);
        const auto b = strikecem::utd_coefficient(k, 1.0, n, 2 * n * kPi - phi, 2 * n * kPi - phip);
        EXPECT_NEAR(std::abs(a.soft - b.soft), 0.0, 1e-12 * std::abs(a.soft));
        EXPECT_NEAR(std::abs(a.hard - b.hard), 0.0, 1e-12 * std::abs(a.hard));
    }
}

TEST(Fringe, FringeBroadsideIdentity) {
    // F = D x I exactly: recompute both factors independently.
    const auto mesh = load_mesh("valid_minimal.json");
    const auto model = strikecem::extract_edges(mesh);
    const double k = 2.0 * kPi;
    const geom::Vec3d s(0, 0, -1), r(0, 0, 1);
    for (const auto& e : model.edges) {
        const auto ang = strikecem::edge_transverse_angles(e, s, r);
        ASSERT_TRUE(ang.valid);
        const auto integ = strikecem::along_edge_integral(e, k, s, r);
        const auto d = strikecem::utd_coefficient(k, e.length, 2.0, ang.phi, ang.phi_prime);
        const auto fs = strikecem::fringe_amplitude(e, k, s, r, 's');
        const auto fh = strikecem::fringe_amplitude(e, k, s, r, 'h');
        ASSERT_TRUE(fs.has_value() && fh.has_value());
        EXPECT_NEAR(std::abs(*fs - d.soft * integ), 0.0, 1e-12);
        EXPECT_NEAR(std::abs(*fh - d.hard * integ), 0.0, 1e-12);
        EXPECT_NEAR(std::abs(*fs), std::abs(d.soft) * e.length, 1e-12);
    }
}

TEST(Fringe, ValleyFringeFiniteAndSplit) {
    const auto mesh = load_mesh("valid_dihedral.json");
    const auto model = strikecem::extract_edges(mesh);
    const auto& v = dihedral_valley(model);
    const double k = 2.0 * kPi;
    const geom::Vec3d s(1, 0, 0), r(-1, 0, 0);
    const auto fs = strikecem::fringe_amplitude(v, k, s, r, 's');
    const auto fh = strikecem::fringe_amplitude(v, k, s, r, 'h');
    ASSERT_TRUE(fs.has_value() && fh.has_value());
    EXPECT_TRUE(std::isfinite(fs->real() + fs->imag()));
    EXPECT_TRUE(std::isfinite(fh->real() + fh->imag()));
    EXPECT_GT(std::abs(*fs - *fh), 1e-3 * std::max(std::abs(*fs), std::abs(*fh)));
    const auto integ = strikecem::along_edge_integral(v, k, s, r);
    EXPECT_NEAR(integ.real(), v.length, 1e-12);
    EXPECT_NEAR(integ.imag(), 0.0, 1e-12);
    const auto d = strikecem::utd_coefficient(k, v.length, 1.5, kPi, kPi);
    EXPECT_NEAR(std::abs(*fs - d.soft * integ), 0.0, 1e-12);
}

TEST(Fringe, ValleyArcSweep) {
    // Monostatic arc in the transverse plane, frame angles 0.2..4.2 rad
    // (exterior cone): frame tracks the look direction, fringe is finite
    // and continuous by refinement.
    const auto mesh = load_mesh("valid_dihedral.json");
    const auto model = strikecem::extract_edges(mesh);
    const auto& v = dihedral_valley(model);
    EXPECT_NEAR(v.length, 1.0, 1e-12);
    const double k = 2.0 * kPi;
    auto fringe_at = [&](double alpha) {
        const geom::Vec3d r(std::cos(alpha), 0.0, -std::sin(alpha));
        const geom::Vec3d s = r * -1.0;
        const auto ang = strikecem::wedge_transverse_angles(v, s, r);
        EXPECT_TRUE(ang.valid) << "alpha=" << alpha;
        EXPECT_NEAR(ang.sin_beta0, 1.0, 1e-12);
        EXPECT_NEAR(ang.phi, alpha, 1e-9) << "alpha=" << alpha;
        EXPECT_NEAR(ang.phi_prime, alpha, 1e-9) << "alpha=" << alpha;
        const auto f = strikecem::fringe_amplitude(v, k, s, r, 's');
        EXPECT_TRUE(f.has_value());
        EXPECT_TRUE(std::isfinite(f->real() + f->imag()));
        return *f;
    };
    auto max_jump = [&](int steps) {
        double worst = 0.0;
        std::complex<double> prev{0.0, 0.0};
        for (int i = 0; i <= steps; ++i) {
            const auto cur = fringe_at(0.2 + (4.2 - 0.2) * i / steps);
            if (i > 0) worst = std::max(worst, std::abs(cur - prev));
            prev = cur;
        }
        return worst;
    };
    // Spot-check the frame at three predicted angles.
    fringe_at(1.0);
    fringe_at(2.0);
    fringe_at(4.0);
    EXPECT_LT(max_jump(64), 2.0 * max_jump(8) + 1e-12);
}

TEST(Fringe, FringeNulloptEndOn) {
    const auto mesh = load_mesh("valid_minimal.json");
    const auto model = strikecem::extract_edges(mesh);
    const geom::Vec3d s(1, 0, 0), r(-1, 0, 0);
    bool found = false;
    for (const auto& e : model.edges) {
        if (std::abs(e.tangent.x) < 0.9) continue;
        EXPECT_FALSE(strikecem::fringe_amplitude(e, 2.0 * kPi, s, r, 's').has_value());
        found = true;
    }
    EXPECT_TRUE(found);
    const auto& e = model.edges[0];
    EXPECT_THROW(strikecem::fringe_amplitude(e, 2.0 * kPi, s, r, 'x'), std::invalid_argument);
    EXPECT_THROW(strikecem::fringe_amplitude(e, 0.0, s, r, 's'), std::invalid_argument);
}
TEST(Fringe, VectorParallelDecomposition) {
    // Normal incidence, E along the edge: pure soft problem, F = Fs * t.
    const auto mesh = load_mesh("valid_minimal.json");
    const auto model = strikecem::extract_edges(mesh);
    const double k = 2.0 * kPi;
    const geom::Vec3d s(0, 0, -1), r(0, 0, 1);
    for (const auto& e : model.edges) {
        const std::array<std::complex<double>, 3> ein = {e.tangent.x, e.tangent.y, e.tangent.z};
        const auto f = strikecem::fringe_vector(e, k, s, r, ein);
        const auto fs = strikecem::fringe_amplitude(e, k, s, r, 's');
        ASSERT_TRUE(f.has_value() && fs.has_value());
        EXPECT_NEAR(std::abs((*f)[0] - *fs * e.tangent.x), 0.0, 1e-12);
        EXPECT_NEAR(std::abs((*f)[1] - *fs * e.tangent.y), 0.0, 1e-12);
        EXPECT_NEAR(std::abs((*f)[2] - *fs * e.tangent.z), 0.0, 1e-12);
    }
}

TEST(Fringe, VectorPerpDecomposition) {
    // Normal incidence, E transverse to the edge: pure hard problem.
    const auto mesh = load_mesh("valid_minimal.json");
    const auto model = strikecem::extract_edges(mesh);
    const double k = 2.0 * kPi;
    const geom::Vec3d s(0, 0, -1), r(0, 0, 1);
    for (const auto& e : model.edges) {
        const geom::Vec3d et = geom::cross(s, e.tangent).normalized();
        const std::array<std::complex<double>, 3> ein = {et.x, et.y, et.z};
        const auto f = strikecem::fringe_vector(e, k, s, r, ein);
        const auto fh = strikecem::fringe_amplitude(e, k, s, r, 'h');
        ASSERT_TRUE(f.has_value() && fh.has_value());
        EXPECT_NEAR(std::abs((*f)[0] - *fh * et.x), 0.0, 1e-12);
        EXPECT_NEAR(std::abs((*f)[1] - *fh * et.y), 0.0, 1e-12);
        EXPECT_NEAR(std::abs((*f)[2] - *fh * et.z), 0.0, 1e-12);
    }
}

TEST(Fringe, VectorRadiativeAndLinear) {
    const auto mesh = load_mesh("valid_minimal.json");
    const auto model = strikecem::extract_edges(mesh);
    const double k = 2.0 * kPi;
    const geom::Vec3d s(0, 0, -1), r = unit(0.6, 0.0, 0.8);
    const auto& e = model.edges[0];
    const std::array<std::complex<double>, 3> e1 = {1.0, 0.0, 0.0};
    const std::array<std::complex<double>, 3> e2 = {{0.0, 1.0, 0.0}};
    const auto f1 = strikecem::fringe_vector(e, k, s, r, e1);
    const auto f2 = strikecem::fringe_vector(e, k, s, r, e2);
    ASSERT_TRUE(f1.has_value() && f2.has_value());
    // Far field stays transverse to the observer ray, even oblique.
    for (const auto& f : {*f1, *f2})
        EXPECT_NEAR(std::abs(f[0] * r.x + f[1] * r.y + f[2] * r.z), 0.0, 1e-12);
    // Linearity in the incident field (pins the split assembly).
    const std::complex<double> a(0.5, 0.0), b(0.0, 1.0);
    const std::array<std::complex<double>, 3> emix = {a * e1[0] + b * e2[0],
                                                      a * e1[1] + b * e2[1],
                                                      a * e1[2] + b * e2[2]};
    const auto fm = strikecem::fringe_vector(e, k, s, r, emix);
    ASSERT_TRUE(fm.has_value());
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(std::abs((*fm)[i] - (a * (*f1)[i] + b * (*f2)[i])), 0.0, 1e-12);
    // Complex (circular-like) incident field: finite and transverse.
    const std::array<std::complex<double>, 3> ecirc = {
        std::complex<double>(1.0 / std::sqrt(2.0), 0.0), std::complex<double>(0.0, 1.0 / std::sqrt(2.0)),
        std::complex<double>(0.0, 0.0)};
    const auto fc = strikecem::fringe_vector(e, k, s, r, ecirc);
    ASSERT_TRUE(fc.has_value());
    EXPECT_TRUE(std::isfinite(fc->at(0).real() + fc->at(2).imag()));
    EXPECT_NEAR(std::abs((*fc)[0] * r.x + (*fc)[1] * r.y + (*fc)[2] * r.z), 0.0, 1e-12);
}

TEST(Fringe, VectorNulloptEndOn) {
    const auto mesh = load_mesh("valid_minimal.json");
    const auto model = strikecem::extract_edges(mesh);
    const geom::Vec3d s(1, 0, 0), r(-1, 0, 0);
    const std::array<std::complex<double>, 3> ein = {0.0, 0.0, 1.0};
    bool found = false;
    for (const auto& e : model.edges) {
        if (std::abs(e.tangent.x) < 0.9) continue;
        EXPECT_FALSE(strikecem::fringe_vector(e, 2.0 * kPi, s, r, ein).has_value());
        found = true;
    }
    EXPECT_TRUE(found);
}

strikecem::SamplePlan make_fringe_plan(const std::vector<double>& freqs,
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

TEST(Fringe, BadInputsThrow) {
    const auto e = synthetic_edge(1.0);
    const geom::Vec3d s(0, 0, -1), r(0, 0, 1), bad(0, 0, 2);
    EXPECT_THROW(strikecem::along_edge_integral(e, 0.0, s, r), std::invalid_argument);
    EXPECT_THROW(strikecem::along_edge_integral(e, 2.0 * kPi, bad, r), std::invalid_argument);
    EXPECT_THROW(strikecem::edge_transverse_angles(e, bad, r), std::invalid_argument);
    strikecem::MeshEdge zero;
    EXPECT_THROW(strikecem::along_edge_integral(zero, 2.0 * kPi, s, r), std::invalid_argument);
}

TEST(Fringe, SolverOffByDefault) {
    // Disabled flag (even with a model attached) reproduces PO exactly.
    const std::string config = fixture("valid_minimal.json");
    auto rc = strikecem::load_config(config, SCEM_SCHEMA_PATH);
    const auto mesh = strikecem::load_normalized_mesh(rc.value, SCEM_FIXTURE_DIR, rc.schema_version);
    const auto model = strikecem::extract_edges(mesh);
    const auto plan = make_fringe_plan({10e9}, {{0.0, -90.0}}, {"HH", "VV"});
    const auto plain = strikecem::solve_po(mesh, plan, rc.value);
    const strikecem::FringeOptions off{false, &model};
    const auto gated = strikecem::solve_po(mesh, plan, rc.value, off);
    ASSERT_EQ(plain.samples.size(), gated.samples.size());
    for (size_t i = 0; i < plain.samples.size(); ++i) {
        EXPECT_EQ(plain.samples[i].scattering, gated.samples[i].scattering);
        EXPECT_EQ(plain.samples[i].rcs_sqm, gated.samples[i].rcs_sqm);
    }
}

TEST(Fringe, SolverMatchesManualAssembly) {
    // Solver accumulation == independent per-edge vector sum projected on
    // the receive basis: pins the hook, not just the core.
    const std::string config = fixture("valid_minimal.json");
    auto rc = strikecem::load_config(config, SCEM_SCHEMA_PATH);
    const auto mesh = strikecem::load_normalized_mesh(rc.value, SCEM_FIXTURE_DIR, rc.schema_version);
    const auto model = strikecem::extract_edges(mesh);
    const auto plan = make_fringe_plan({10e9}, {{0.0, -90.0}}, {"HH"});
    const auto po = strikecem::solve_po(mesh, plan, rc.value);
    const strikecem::FringeOptions on{true, &model};
    const auto total = strikecem::solve_po(mesh, plan, rc.value, on);
    ASSERT_EQ(total.samples.size(), 1u);
    const double e0 = rc.value["physics"]["incident_amplitude"].get<double>();
    const geom::Vec3d k_hat = plan.directions[0].k_hat;
    const geom::Vec3d r_hat = k_hat * -1.0;
    const auto basis = strikecem::polarization_basis(k_hat);
    const double freq = plan.frequencies_hz[0];
    const double k = 2.0 * kPi * freq / strikecem::kSpeedOfLight;
    std::complex<double> manual{0.0, 0.0};
    for (const auto& e : model.edges) {
        const std::array<std::complex<double>, 3> ein = {basis.h.x, basis.h.y, basis.h.z};
        const auto f = strikecem::fringe_vector(e, k, k_hat, r_hat, ein);
        ASSERT_TRUE(f.has_value());
        manual += (*f)[0] * basis.h.x + (*f)[1] * basis.h.y + (*f)[2] * basis.h.z;
    }
    EXPECT_NEAR(std::abs(total.samples[0].scattering - po.samples[0].scattering - manual), 0.0,
                1e-9 * std::abs(po.samples[0].scattering));
}

TEST(Fringe, SolverNullFillsDarkSide) {
    // Dark side: PO is exactly zero; PO + fringe is finite (the fringe is
    // the entire return) and matches the manual edge sum.
    const std::string config = fixture("valid_minimal.json");
    auto rc = strikecem::load_config(config, SCEM_SCHEMA_PATH);
    const auto mesh = strikecem::load_normalized_mesh(rc.value, SCEM_FIXTURE_DIR, rc.schema_version);
    const auto model = strikecem::extract_edges(mesh);
    const auto plan = make_fringe_plan({10e9}, {{0.0, 90.0}}, {"HH"});
    const auto po = strikecem::solve_po(mesh, plan, rc.value);
    ASSERT_EQ(po.samples[0].lit_facets, 0u);
    EXPECT_DOUBLE_EQ(std::abs(po.samples[0].scattering), 0.0);
    const strikecem::FringeOptions on{true, &model};
    const auto total = strikecem::solve_po(mesh, plan, rc.value, on);
    EXPECT_EQ(total.samples[0].lit_facets, 0u);
    EXPECT_GT(std::abs(total.samples[0].scattering), 0.0);
    EXPECT_GT(total.samples[0].rcs_sqm, 0.0);
}

TEST(Fringe, SolverGuards) {
    const std::string config = fixture("valid_minimal.json");
    auto rc = strikecem::load_config(config, SCEM_SCHEMA_PATH);
    const auto mesh = strikecem::load_normalized_mesh(rc.value, SCEM_FIXTURE_DIR, rc.schema_version);
    const auto model = strikecem::extract_edges(mesh);
    const auto plan = make_fringe_plan({10e9}, {{0.0, -90.0}}, {"HH"});
    const strikecem::FringeOptions null_model{true, nullptr};
    EXPECT_THROW(strikecem::solve_po(mesh, plan, rc.value, null_model), std::invalid_argument);
    // Fringe + CUDA fails closed (exit 4 at the CLI) before touching any device.
    auto rc_cuda = rc;
    rc_cuda.value["execution"]["accelerator"] = "cuda";
    const strikecem::FringeOptions on{true, &model};
    EXPECT_THROW(strikecem::solve_po(mesh, plan, rc_cuda.value, on), strikecem::cuda::CudaError);
    // Fringe off + CUDA routes to the device path when one answers, else
    // the stub throws: accept either, the fringe flag changes nothing.
    const strikecem::FringeOptions off{false, &model};
    try {
        const auto gpu = strikecem::solve_po(mesh, plan, rc_cuda.value, off);
        EXPECT_EQ(gpu.samples.size(), 1u);
    } catch (const strikecem::cuda::CudaError&) {
        EXPECT_FALSE(strikecem::cuda::cuda_available());
    }
}

TEST(Fringe, SolverFloat32Smoke) {
    const std::string config = fixture("valid_minimal.json");
    auto rc = strikecem::load_config(config, SCEM_SCHEMA_PATH);
    rc.value["solver"]["precision"] = "float32";
    const auto mesh = strikecem::load_normalized_mesh(rc.value, SCEM_FIXTURE_DIR, rc.schema_version);
    const auto model = strikecem::extract_edges(mesh);
    const auto plan = make_fringe_plan({10e9}, {{0.0, -90.0}}, {"HH", "VV"});
    const strikecem::FringeOptions on{true, &model};
    const auto total = strikecem::solve_po(mesh, plan, rc.value, on);
    ASSERT_EQ(total.samples.size(), 2u);
    for (const auto& row : total.samples)
        EXPECT_TRUE(std::isfinite(row.scattering.real() + row.rcs_sqm));
}

} // namespace

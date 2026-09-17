#pragma once
// 3D straight-edge fringe geometry (ADR-0003 slice C): transverse wedge
// angles for half-plane rims plus the analytic along-edge line integral.
// No amplitude physics here: coupling validated-2D-pattern x integral and
// its absolute calibration is the next slice with its own benchmark.
// e^{+jwt} convention, matching the v1 solvers.
//
// Directions are unit travel/observer vectors: s_hat = incident direction
// of travel, r_hat = direction from origin to the far observer.
#include <complex>
#include <optional>

#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/EdgeModel.hpp"

namespace strikecem {

struct TransverseAngles {
    double phi = 0.0;       // observation angle, [0, 2pi) from the screen
    double phi_prime = 0.0; // arrival angle, [0, 2pi) from the screen
    double sin_beta0 = 1.0; // |incident transverse part| (1 = normal to edge)
    bool valid = false;     // false for non-rims and end-on incidence
};

// Transverse (phi, phi_prime) for a half-plane rim (boundary, wedge_n ==
// 2), measured from the screen (face0_dir) with e2 = face0_dir x tangent.
// valid=false for interior creases (wedge-frame mapping deferred) and for
// end-on incidence (transverse projection vanishes). Throws
// std::invalid_argument on non-unit/non-finite directions.
// ponytail: single-edge analytic scope; end-on and wedge-n!=2 cones are
// flagged, not modeled — the upgrade path is slice D (D-coupling) which
// decides per-cone handling with its calibration benchmark.
TransverseAngles edge_transverse_angles(const MeshEdge& edge,
                                        const geom::Vec3d& s_hat,
                                        const geom::Vec3d& r_hat);

// Transverse (phi, phi_prime) for an interior wedge crease (1 <= wedge_n
// <= 2), measured from the face-tri0 ray through the exterior: phi = 0 on
// face0_dir and phi = wedge_n * pi on the face-tri1 ray, so the exterior
// cone is (0, wedge_n * pi). e2 = face0_dir x tangent, negated when
// face1_dir.e2 > 0, which makes the frame tangent-sign invariant.
// Observers in the interior cone are unphysical (inside PEC);
// evaluation there is the caller's responsibility. valid=false for rims
// (use the screen frame above), wedge_n outside [1, 2], degenerate face
// directions, and end-on incidence. Throws on non-unit/non-finite
// directions like the screen frame.
TransverseAngles wedge_transverse_angles(const MeshEdge& edge,
                                         const geom::Vec3d& s_hat,
                                         const geom::Vec3d& r_hat);

// Fringe amplitude F = D_pol(k_t, rho = L; phi, phi_prime) x I, with
// transverse wavenumber k_t = k * sin_beta0, edge length L, and I the
// along-edge integral above. pol 's' = E parallel to edge (Dirichlet),
// 'h' = H parallel to edge (Neumann). Rims use the screen frame,
// interior creases the wedge frame; nullopt when the frame is invalid
// (end-on incidence). Throws std::invalid_argument on bad k, pol, or
// directions.
// ponytail: rho_ref = L is the provisional calibration (the only
// intrinsic length; k_t * L >> 1 recovers Keller away from boundaries).
// Absolute spreading and any residual obliquity amplitude ride with the
// solver-integration benchmark (slice D2), which is the upgrade path.
std::optional<std::complex<double>> fringe_amplitude(const MeshEdge& edge, double k,
                                                     const geom::Vec3d& s_hat,
                                                     const geom::Vec3d& r_hat, char pol);

// Along-edge line integral I = int_edge e^{jk(r_hat - s_hat).r'} dl'
// = e^{jk(r_hat - s_hat).c} L sinc(k L a / 2), a = (r_hat - s_hat).t_hat,
// c = edge center. Phase convention matches PhysicalOptics (outgoing
// e^{+jk r.c}, incident e^{-jk s.c}). |I| <= L. Throws on k <= 0,
// non-unit directions, or zero-length edges.
std::complex<double> along_edge_integral(const MeshEdge& edge, double k,
                                         const geom::Vec3d& s_hat,
                                         const geom::Vec3d& r_hat);

} // namespace strikecem

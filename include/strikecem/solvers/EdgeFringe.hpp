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

// Along-edge line integral I = int_edge e^{jk(r_hat - s_hat).r'} dl'
// = e^{jk(r_hat - s_hat).c} L sinc(k L a / 2), a = (r_hat - s_hat).t_hat,
// c = edge center. Phase convention matches PhysicalOptics (outgoing
// e^{+jk r.c}, incident e^{-jk s.c}). |I| <= L. Throws on k <= 0,
// non-unit directions, or zero-length edges.
std::complex<double> along_edge_integral(const MeshEdge& edge, double k,
                                         const geom::Vec3d& s_hat,
                                         const geom::Vec3d& r_hat);

} // namespace strikecem

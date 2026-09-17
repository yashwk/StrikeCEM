#pragma once
// Ray kernel for GO/SBR (ADR-0004 slice 1): double-precision
// Möller–Trumbore intersection, closest-hit mesh query, and segment
// occlusion. Double-sided throughout: zero-thickness PEC blocks rays
// from both sides. No bounce logic here; directions must be unit
// (invalid_argument otherwise).
#include <cstdint>
#include <optional>

#include "strikecem/io/MeshLoader.hpp"

namespace strikecem {

struct RayHit {
    double t = 0.0;     // distance along dir (dir is unit)
    uint32_t tri = 0;   // mesh triangle index
    double u = 0.0;     // barycentric weights: point = (1-u-v)*a + u*b + v*c
    double v = 0.0;
};

// GO shadowing switch. Off by default; enabled only via the solver API
// until the schema unlocks shadowing (ADR-0004). No bounce logic here.
struct ShadowOptions {
    bool enabled = false;
};

// Nullopt on miss, parallel (|det| ~ 0), or degenerate triangle.
// Accepts hits with t > t_min only.
std::optional<RayHit> ray_triangle(const geom::Vec3d& origin, const geom::Vec3d& dir,
                                   const geom::Vec3d& a, const geom::Vec3d& b,
                                   const geom::Vec3d& c, double t_min, uint32_t tri = 0);

// Closest hit over the mesh, skipping skip_tri (pass UINT32_MAX for none).
// Throws invalid_argument on non-unit dir or negative t_min.
std::optional<RayHit> ray_mesh(const NormalizedMesh& mesh, const geom::Vec3d& origin,
                               const geom::Vec3d& dir, double t_min, uint32_t skip_tri);

// True when any facet other than self_tri intersects the segment
// [p, p + s * max_t). The t_min floor is 1e-9 of the mesh bbox diagonal.
// ponytail: absolute floor, not wavelength-relative — the upgrade path is
// a wavelength-relative floor if electrically tiny features ever arrive.
bool occluded(const NormalizedMesh& mesh, const geom::Vec3d& p, const geom::Vec3d& s,
              double max_t, uint32_t self_tri);

} // namespace strikecem

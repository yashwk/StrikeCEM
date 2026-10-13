#pragma once
#include <array>
#include <complex>
#include <cstdint>
#include <optional>

#include "strikecem/io/MeshLoader.hpp"

namespace strikecem {

struct RayHit {
    double t = 0.0;
    uint32_t tri = 0;
    double bary_u = 0.0;
    double bary_v = 0.0;
};

struct GoOptions {
    bool shadowing = false;
    int max_bounces = 1;
};

struct BvhNode {
    geom::Vec3d bmin;
    geom::Vec3d bmax;
    int left = -1;
    int right = -1;
    uint32_t start = 0;
    uint32_t count = 0;
};

struct Bvh {
    std::vector<BvhNode> nodes;
    std::vector<uint32_t> order;
};

Bvh build_bvh(const NormalizedMesh& mesh);

std::optional<RayHit> ray_bvh(const Bvh& bvh, const NormalizedMesh& mesh,
                               const geom::Vec3d& origin, const geom::Vec3d& dir, double t_min,
                               uint32_t skip_tri);

bool occluded_bvh(const Bvh& bvh, const NormalizedMesh& mesh, const geom::Vec3d& p,
                  const geom::Vec3d& s, double max_t, uint32_t self_tri);

std::optional<RayHit> ray_triangle(const geom::Vec3d& origin, const geom::Vec3d& dir,
                                   const geom::Vec3d& a, const geom::Vec3d& b,
                                   const geom::Vec3d& c, double t_min, uint32_t tri = 0);

std::optional<RayHit> ray_mesh(const NormalizedMesh& mesh, const geom::Vec3d& origin,
                               const geom::Vec3d& dir, double t_min, uint32_t skip_tri);

bool occluded(const NormalizedMesh& mesh, const geom::Vec3d& p, const geom::Vec3d& s,
              double max_t, uint32_t self_tri);

// --- SBR (Shooting and Bouncing Rays) & Multi-Bounce Coherent Tracking ---

// Computes specular reflection of vector d off unit normal n: d - 2 * (d . n) * n
geom::Vec3d reflect_direction(const geom::Vec3d& d, const geom::Vec3d& n);

// Computes PEC reflection of complex electric field: 2 * (e . n) * n - e
std::array<std::complex<double>, 3> reflect_electric_field(
    const std::array<std::complex<double>, 3>& e, const geom::Vec3d& n);

// Polarization decomposition into TE (perpendicular to incident plane) and TM (parallel)
struct PolarizationDecomposition {
    std::complex<double> e_te{0.0, 0.0};
    std::complex<double> e_tm{0.0, 0.0};
    geom::Vec3d te_hat;
    geom::Vec3d tm_hat;
};

PolarizationDecomposition decompose_polarization(
    const std::array<std::complex<double>, 3>& e_in,
    const geom::Vec3d& dir,
    const geom::Vec3d& normal);

struct SbrRayHop {
    uint32_t triangle_id = 0;
    geom::Vec3d hit_point;
    geom::Vec3d normal;
    geom::Vec3d incoming_dir;
    geom::Vec3d outgoing_dir;
    double segment_length = 0.0;
    std::complex<double> phase_factor{1.0, 0.0};
    std::array<std::complex<double>, 3> e_field = {};
};

struct SbrRayPath {
    std::vector<SbrRayHop> hops;
    bool is_retroreflective = false;
    bool exit_occluded = false;
    std::complex<double> total_phase{1.0, 0.0};
};

SbrRayPath trace_sbr_path(
    const Bvh& bvh, const NormalizedMesh& mesh,
    uint32_t first_triangle,
    const geom::Vec3d& initial_k,
    const std::array<std::complex<double>, 3>& initial_e,
    const geom::Vec3d& obs_r,
    double k_wave,
    const geom::Vec3d& phase_ref,
    int max_bounces);

}

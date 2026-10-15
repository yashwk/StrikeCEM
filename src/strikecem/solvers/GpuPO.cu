// CUDA Physical Optics backend (ADR-0002) with Hardware-Accelerated
// Shooting & Bouncing Rays (SBR / Multi-Bounce) and Radar Absorbing Materials
// (RAM / Surface Impedance & Dielectric Coatings).
// One block per (frequency, direction) unit; threads stride over facets with
// a deterministic shared-memory tree reduction.
#include "strikecem/solvers/GpuPO.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "strikecem/geometry/Symmetry.hpp"
#include "strikecem/solvers/Fresnel.hpp"
#include "strikecem/solvers/Material.hpp"
#include "strikecem/solvers/Projection.hpp"
#include "strikecem/solvers/Ray.hpp"

namespace strikecem::cuda {
namespace {

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                             \
            (void)cudaGetLastError();                                         \
            throw CudaError(std::string("CUDA failure: ") +                   \
                            cudaGetErrorString(err));                         \
        }                                                                     \
    } while (0)

template <typename T> class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw CudaError("CUDA buffer size overflow");
        if (count != 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (ptr_) (void)cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }

private:
    T* ptr_ = nullptr;
};

template <typename Real> struct Cplx {
    Real re{0}, im{0};
};

// Device complex arithmetic helpers
template <typename Real>
__device__ inline Cplx<Real> c_add(Cplx<Real> a, Cplx<Real> b) {
    return {a.re + b.re, a.im + b.im};
}

template <typename Real>
__device__ inline Cplx<Real> c_sub(Cplx<Real> a, Cplx<Real> b) {
    return {a.re - b.re, a.im - b.im};
}

template <typename Real>
__device__ inline Cplx<Real> c_mul(Cplx<Real> a, Cplx<Real> b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

template <typename Real>
__device__ inline Cplx<Real> c_scale(Cplx<Real> a, Real s) {
    return {a.re * s, a.im * s};
}

template <typename Real>
__device__ inline Cplx<Real> c_div(Cplx<Real> a, Cplx<Real> b) {
    Real d = b.re * b.re + b.im * b.im;
    if (d < Real(1e-30)) return {Real(0), Real(0)};
    Real inv = Real(1) / d;
    return {(a.re * b.re + a.im * b.im) * inv, (a.im * b.re - a.re * b.im) * inv};
}

__device__ inline void fsincos(float a, float* s, float* c) { sincosf(a, s, c); }
__device__ inline void fsincos(double a, double* s, double* c) { sincos(a, s, c); }

template <typename Real>
__device__ inline Cplx<Real> c_sqrt(Cplx<Real> a) {
    Real r = sqrt(a.re * a.re + a.im * a.im);
    Real phi = atan2(a.im, a.re);
    Real sqrt_r = sqrt(r);
    Real s, c;
    fsincos(Real(0.5) * phi, &s, &c);
    return {sqrt_r * c, sqrt_r * s};
}

template <typename Real>
__device__ inline Cplx<Real> c_exp(Cplx<Real> a) {
    Real exp_re = exp(a.re);
    Real s, c;
    fsincos(a.im, &s, &c);
    return {exp_re * c, exp_re * s};
}

// Material definitions for GPU
struct GpuLayer {
    double thickness_m;
    Cplx<double> eps_r;
    Cplx<double> mu_r;
};

struct GpuMaterialModel {
    uint8_t wall_type; // 0=PEC, 1=Dielectric, 2=Coated, 3=Impedance
    uint8_t num_layers;
    Cplx<double> eta_s; // normalized surface impedance
    Cplx<double> eps_r; // for dielectric
    Cplx<double> mu_r;
    GpuLayer layers[8]; // up to 8 layers for coated
};

// BVH structures for GPU ray tracing
struct GpuBvhNode {
    double bmin[3];
    double bmax[3];
    int left;
    int right;
    uint32_t start;
    uint32_t count;
};

struct GpuBvhData {
    const GpuBvhNode* nodes;
    const uint32_t* order;
    const double* tri_vertices;
    int num_nodes;
    double scene_diag;
    double t_min;
    double scene_limit;
    double phase_ref[3];
    bool shadowing;
    int max_bounces;
};

template <typename Real> struct UnitParams {
    Real kx, ky, kz;   // incident propagation direction (unit)
    Real rx, ry, rz;   // backscatter direction (-k_hat)
    Real ehx, ehy, ehz; // H-transmit E0 vector (with amplitude)
    Real evx, evy, evz; // V-transmit E0 vector (with amplitude)
    Real k, eta, e0;
    Real qx, qy, qz;   // wavevector difference: k * (r - k)
    Real h0_h[3];      // (k x Eh) / eta
    Real h0_v[3];      // (k x Ev) / eta
    Real k_eta_4pi;    // k * eta / (4 * pi)
    Real freq_hz;
    Real lambda_m;
    int has_materials;
    int num_materials;
    GpuMaterialModel materials[8];
};

constexpr int kBlock = 256;

// --- Device Material Reflection Functions ---

template <typename Real>
__device__ inline void dev_impedance_reflection(Cplx<Real> eta_s, Real cos_i,
                                                Cplx<Real>& r_te, Cplx<Real>& r_tm) {
    Cplx<Real> num_te = {eta_s.re * cos_i - Real(1), eta_s.im * cos_i};
    Cplx<Real> den_te = {eta_s.re * cos_i + Real(1), eta_s.im * cos_i};
    Real den_te_sq = den_te.re * den_te.re + den_te.im * den_te.im;
    r_te = (den_te_sq < Real(1e-28)) ? Cplx<Real>{Real(-1), Real(0)} : c_div(num_te, den_te);

    Cplx<Real> num_tm = {eta_s.re - cos_i, eta_s.im};
    Cplx<Real> den_tm = {eta_s.re + cos_i, eta_s.im};
    Real den_tm_sq = den_tm.re * den_tm.re + den_tm.im * den_tm.im;
    r_tm = (den_tm_sq < Real(1e-28)) ? Cplx<Real>{Real(1), Real(0)} : c_div(num_tm, den_tm);
}

template <typename Real>
__device__ inline void dev_fresnel(Cplx<Real> eps1, Cplx<Real> mu1,
                                   Cplx<Real> eps2, Cplx<Real> mu2,
                                   Real cos_i,
                                   Cplx<Real>& r_te, Cplx<Real>& r_tm) {
    Cplx<Real> z1 = c_sqrt(c_div(mu1, eps1));
    Cplx<Real> z2 = c_sqrt(c_div(mu2, eps2));
    Cplx<Real> n1_sq = c_mul(eps1, mu1);
    Cplx<Real> n2_sq = c_mul(eps2, mu2);
    Real sin_i_sq = Real(1) - cos_i * cos_i;
    Cplx<Real> sin_t_sq = c_scale(c_div(n1_sq, n2_sq), sin_i_sq);
    Cplx<Real> one = {Real(1), Real(0)};
    Cplx<Real> cos_t = c_sqrt(c_sub(one, sin_t_sq));
    if (cos_t.re < Real(0)) cos_t = c_scale(cos_t, Real(-1));

    Cplx<Real> cos_i_c = {cos_i, Real(0)};
    Cplx<Real> z2_cos_i = c_mul(z2, cos_i_c);
    Cplx<Real> z1_cos_t = c_mul(z1, cos_t);
    r_te = c_div(c_sub(z2_cos_i, z1_cos_t), c_add(z2_cos_i, z1_cos_t));

    Cplx<Real> z1_cos_i = c_mul(z1, cos_i_c);
    Cplx<Real> z2_cos_t = c_mul(z2, cos_t);
    r_tm = c_div(c_sub(z1_cos_i, z2_cos_t), c_add(z1_cos_i, z2_cos_t));
}

template <typename Real>
__device__ inline void dev_coated_pec_reflection(Cplx<Real> eps_inc, Cplx<Real> mu_inc,
                                                const GpuLayer* layers, int num_layers,
                                                Real cos_i, Real wavelength_m,
                                                Cplx<Real>& r_te, Cplx<Real>& r_tm) {
    Real k0 = Real(2.0 * 3.14159265358979323846) / wavelength_m;
    Cplx<Real> rs_te[8];
    Cplx<Real> rs_tm[8];
    Cplx<Real> phis[8];
    Cplx<Real> eps_above = eps_inc;
    Cplx<Real> mu_above = mu_inc;
    Cplx<Real> cos_above = {cos_i, Real(0)};

    for (int l = 0; l < num_layers; ++l) {
        Cplx<Real> eps_l = {Real(layers[l].eps_r.re), Real(layers[l].eps_r.im)};
        Cplx<Real> mu_l = {Real(layers[l].mu_r.re), Real(layers[l].mu_r.im)};
        Real d_l = Real(layers[l].thickness_m);

        Cplx<Real> n = c_sqrt(c_mul(eps_l, mu_l));
        Cplx<Real> n_above_sq = c_mul(eps_above, mu_above);
        Cplx<Real> n_l_sq = c_mul(eps_l, mu_l);
        Cplx<Real> sin_above_sq = c_sub(Cplx<Real>{Real(1), Real(0)}, c_mul(cos_above, cos_above));
        Cplx<Real> sin_t_sq = c_mul(c_div(n_above_sq, n_l_sq), sin_above_sq);
        Cplx<Real> cos_t = c_sqrt(c_sub(Cplx<Real>{Real(1), Real(0)}, sin_t_sq));
        if (cos_t.re < Real(0)) cos_t = c_scale(cos_t, Real(-1));

        Cplx<Real> z_above = c_sqrt(c_div(mu_above, eps_above));
        Cplx<Real> z_l = c_sqrt(c_div(mu_l, eps_l));
        Cplx<Real> z_l_cos_a = c_mul(z_l, cos_above);
        Cplx<Real> z_a_cos_t = c_mul(z_above, cos_t);
        rs_te[l] = c_div(c_sub(z_l_cos_a, z_a_cos_t), c_add(z_l_cos_a, z_a_cos_t));

        Cplx<Real> z_a_cos_a = c_mul(z_above, cos_above);
        Cplx<Real> z_l_cos_t = c_mul(z_l, cos_t);
        rs_tm[l] = c_div(c_sub(z_a_cos_a, z_l_cos_t), c_add(z_a_cos_a, z_l_cos_t));

        Cplx<Real> beta = c_scale(c_mul(n, cos_t), k0 * d_l);
        Cplx<Real> minus_2i_beta = {Real(2) * beta.im, Real(-2) * beta.re};
        phis[l] = c_exp(minus_2i_beta);

        eps_above = eps_l;
        mu_above = mu_l;
        cos_above = cos_t;
    }

    Cplx<Real> cur_te = {Real(-1), Real(0)};
    Cplx<Real> cur_tm = {Real(1), Real(0)};
    for (int j = num_layers - 1; j >= 0; --j) {
        Cplx<Real> num_te = c_add(rs_te[j], c_mul(cur_te, phis[j]));
        Cplx<Real> den_te = c_add(Cplx<Real>{Real(1), Real(0)}, c_mul(rs_te[j], c_mul(cur_te, phis[j])));
        cur_te = c_div(num_te, den_te);

        Cplx<Real> num_tm = c_add(rs_tm[j], c_mul(cur_tm, phis[j]));
        Cplx<Real> den_tm = c_add(Cplx<Real>{Real(1), Real(0)}, c_mul(rs_tm[j], c_mul(cur_tm, phis[j])));
        cur_tm = c_div(num_tm, den_tm);
    }
    r_te = cur_te;
    r_tm = cur_tm;
}

// Device material facet field evaluation (exact match to PhysicalOptics.cpp:52-84)
template <typename Real>
__device__ inline void dev_material_facet_field(
    const Real* n, const Real* k_hat, const Real* r_hat,
    const Cplx<Real>* e_inc, Real k, Real eta, Real area, const Real* centroid,
    Cplx<Real> r_te, Cplx<Real> r_tm,
    Cplx<Real>* out_field) {
    Real e_te[3] = {n[1] * k_hat[2] - n[2] * k_hat[1],
                    n[2] * k_hat[0] - n[0] * k_hat[2],
                    n[0] * k_hat[1] - n[1] * k_hat[0]};
    Real len_sq = e_te[0] * e_te[0] + e_te[1] * e_te[1] + e_te[2] * e_te[2];
    if (len_sq <= Real(1e-24)) {
        Real ax[3] = {(fabs(n[0]) <= fabs(n[1]) && fabs(n[0]) <= fabs(n[2])) ? Real(1) : Real(0),
                      (fabs(n[1]) <= fabs(n[2]) && !(fabs(n[0]) <= fabs(n[1]) && fabs(n[0]) <= fabs(n[2]))) ? Real(1) : Real(0),
                      Real(0)};
        if (ax[0] == Real(0) && ax[1] == Real(0)) ax[2] = Real(1);
        e_te[0] = n[1] * ax[2] - n[2] * ax[1];
        e_te[1] = n[2] * ax[0] - n[0] * ax[2];
        e_te[2] = n[0] * ax[1] - n[1] * ax[0];
        len_sq = e_te[0] * e_te[0] + e_te[1] * e_te[1] + e_te[2] * e_te[2];
    }
    Real inv_len = rsqrt(len_sq);
    e_te[0] *= inv_len; e_te[1] *= inv_len; e_te[2] *= inv_len;

    Real tm_inc[3] = {k_hat[1] * e_te[2] - k_hat[2] * e_te[1],
                      k_hat[2] * e_te[0] - k_hat[0] * e_te[2],
                      k_hat[0] * e_te[1] - k_hat[1] * e_te[0]};

    Real kn = k_hat[0] * n[0] + k_hat[1] * n[1] + k_hat[2] * n[2];
    Real k_ref[3] = {k_hat[0] - Real(2) * kn * n[0],
                     k_hat[1] - Real(2) * kn * n[1],
                     k_hat[2] - Real(2) * kn * n[2]};

    Real tm_ref[3] = {k_ref[1] * e_te[2] - k_ref[2] * e_te[1],
                      k_ref[2] * e_te[0] - k_ref[0] * e_te[2],
                      k_ref[0] * e_te[1] - k_ref[1] * e_te[0]};

    Cplx<Real> a_te = {e_inc[0].re * e_te[0] + e_inc[1].re * e_te[1] + e_inc[2].re * e_te[2],
                       e_inc[0].im * e_te[0] + e_inc[1].im * e_te[1] + e_inc[2].im * e_te[2]};
    Cplx<Real> a_tm = {e_inc[0].re * tm_inc[0] + e_inc[1].re * tm_inc[1] + e_inc[2].re * tm_inc[2],
                       e_inc[0].im * tm_inc[0] + e_inc[1].im * tm_inc[1] + e_inc[2].im * tm_inc[2]};

    Cplx<Real> r_a_te = c_mul(r_te, a_te);
    Cplx<Real> r_a_tm = c_mul(r_tm, a_tm);
    Cplx<Real> e_ref[3];
    for (int i = 0; i < 3; ++i) {
        e_ref[i] = {r_a_te.re * e_te[i] + r_a_tm.re * tm_ref[i],
                    r_a_te.im * e_te[i] + r_a_tm.im * tm_ref[i]};
    }

    Real inv_eta = Real(1) / eta;
    Cplx<Real> h_tot[3], e_tot[3];
    for (int i = 0; i < 3; ++i) {
        e_tot[i] = c_add(e_inc[i], e_ref[i]);
    }
    Cplx<Real> h_inc[3] = {
        c_scale(c_sub(c_scale(e_inc[2], k_hat[1]), c_scale(e_inc[1], k_hat[2])), inv_eta),
        c_scale(c_sub(c_scale(e_inc[0], k_hat[2]), c_scale(e_inc[2], k_hat[0])), inv_eta),
        c_scale(c_sub(c_scale(e_inc[1], k_hat[0]), c_scale(e_inc[0], k_hat[1])), inv_eta)
    };
    Cplx<Real> h_ref[3] = {
        c_scale(c_sub(c_scale(e_ref[2], k_ref[1]), c_scale(e_ref[1], k_ref[2])), inv_eta),
        c_scale(c_sub(c_scale(e_ref[0], k_ref[2]), c_scale(e_ref[2], k_ref[0])), inv_eta),
        c_scale(c_sub(c_scale(e_ref[1], k_ref[0]), c_scale(e_ref[0], k_ref[1])), inv_eta)
    };
    for (int i = 0; i < 3; ++i) {
        h_tot[i] = c_add(h_inc[i], h_ref[i]);
    }

    Cplx<Real> j[3] = {
        c_sub(c_scale(h_tot[2], n[1]), c_scale(h_tot[1], n[2])),
        c_sub(c_scale(h_tot[0], n[2]), c_scale(h_tot[2], n[0])),
        c_sub(c_scale(h_tot[1], n[0]), c_scale(h_tot[0], n[1]))
    };
    Cplx<Real> m[3] = {
        c_sub(c_scale(e_tot[1], n[2]), c_scale(e_tot[2], n[1])),
        c_sub(c_scale(e_tot[2], n[0]), c_scale(e_tot[0], n[2])),
        c_sub(c_scale(e_tot[0], n[1]), c_scale(e_tot[1], n[0]))
    };

    Cplx<Real> r_cross_j[3] = {
        c_sub(c_scale(j[2], r_hat[1]), c_scale(j[1], r_hat[2])),
        c_sub(c_scale(j[0], r_hat[2]), c_scale(j[2], r_hat[0])),
        c_sub(c_scale(j[1], r_hat[0]), c_scale(j[0], r_hat[1]))
    };
    Cplx<Real> tj[3] = {
        c_sub(c_scale(r_cross_j[2], r_hat[1]), c_scale(r_cross_j[1], r_hat[2])),
        c_sub(c_scale(r_cross_j[0], r_hat[2]), c_scale(r_cross_j[2], r_hat[0])),
        c_sub(c_scale(r_cross_j[1], r_hat[0]), c_scale(r_cross_j[0], r_hat[1]))
    };

    Cplx<Real> tm[3] = {
        c_sub(c_scale(m[2], r_hat[1]), c_scale(m[1], r_hat[2])),
        c_sub(c_scale(m[0], r_hat[2]), c_scale(m[2], r_hat[0])),
        c_sub(c_scale(m[1], r_hat[0]), c_scale(m[0], r_hat[1]))
    };

    Real r_dot_c = r_hat[0] * centroid[0] + r_hat[1] * centroid[1] + r_hat[2] * centroid[2];
    Real phi = k * r_dot_c;
    Real s_phi, c_phi;
    fsincos(phi, &s_phi, &c_phi);
    Real k_area_4pi = k * area / Real(4.0 * 3.14159265358979323846);
    Cplx<Real> amp = {-k_area_4pi * s_phi, k_area_4pi * c_phi};

    for (int i = 0; i < 3; ++i) {
        Cplx<Real> bracket = c_add(c_scale(tj[i], eta), tm[i]);
        out_field[i] = c_mul(amp, bracket);
    }
}

// --- GPU Ray Tracing Device Functions ---

__device__ inline bool gpu_slab_hit(const GpuBvhNode& node, const double* o, const double* d,
                                    double t_min, double& entry) {
    double t0 = t_min;
    double t1 = 1e30;
    for (int i = 0; i < 3; ++i) {
        if (d[i] == 0.0) {
            if (o[i] < node.bmin[i] || o[i] > node.bmax[i]) return false;
        } else {
            const double inv = 1.0 / d[i];
            double a = (node.bmin[i] - o[i]) * inv;
            double b = (node.bmax[i] - o[i]) * inv;
            if (a > b) { double tmp = a; a = b; b = tmp; }
            if (a > t0) t0 = a;
            if (b < t1) t1 = b;
            if (t0 > t1) return false;
        }
    }
    entry = t0;
    return true;
}

__device__ inline bool gpu_ray_triangle(const double* o, const double* d,
                                        const double* a, const double* b, const double* c,
                                        double t_min, double& hit_t) {
    double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    double e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    double p[3] = {d[1] * e2[2] - d[2] * e2[1],
                   d[2] * e2[0] - d[0] * e2[2],
                   d[0] * e2[1] - d[1] * e2[0]};
    double det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
    if (fabs(det) < 1e-18) return false;
    double inv = 1.0 / det;
    double q[3] = {o[0] - a[0], o[1] - a[1], o[2] - a[2]};
    double u = (q[0] * p[0] + q[1] * p[1] + q[2] * p[2]) * inv;
    if (u < -1e-9 || u > 1.0 + 1e-9) return false;
    double r[3] = {q[1] * e1[2] - q[2] * e1[1],
                   q[2] * e1[0] - q[0] * e1[2],
                   q[0] * e1[1] - q[1] * e1[0]};
    double v = (d[0] * r[0] + d[1] * r[1] + d[2] * r[2]) * inv;
    if (v < -1e-9 || u + v > 1.0 + 1e-9) return false;
    double t = (e2[0] * r[0] + e2[1] * r[1] + e2[2] * r[2]) * inv;
    if (!(t > t_min)) return false;
    hit_t = t;
    return true;
}

__device__ inline bool gpu_occluded_bvh(const GpuBvhNode* nodes, const uint32_t* order,
                                        const double* tri_verts,
                                        const double* o, const double* d,
                                        double t_min, double max_t, uint32_t self_tri) {
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const GpuBvhNode& node = nodes[stack[--sp]];
        double entry = 0.0;
        if (!gpu_slab_hit(node, o, d, t_min, entry)) continue;
        if (!(entry < max_t)) continue;
        if (node.count > 0) {
            for (uint32_t k = 0; k < node.count; ++k) {
                uint32_t t = order[node.start + k];
                if (t == self_tri) continue;
                const double* a = &tri_verts[9 * t];
                const double* b = &tri_verts[9 * t + 3];
                const double* c = &tri_verts[9 * t + 6];
                double hit_t;
                if (gpu_ray_triangle(o, d, a, b, c, t_min, hit_t) && hit_t < max_t) {
                    return true;
                }
            }
        } else {
            stack[sp++] = node.left;
            stack[sp++] = node.right;
        }
    }
    return false;
}

__device__ inline bool gpu_ray_bvh(const GpuBvhNode* nodes, const uint32_t* order,
                                   const double* tri_verts,
                                   const double* o, const double* d,
                                   double t_min, uint32_t skip_tri,
                                   uint32_t& hit_tri, double& best_t) {
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;
    bool found = false;
    best_t = 1e30;
    while (sp > 0) {
        const GpuBvhNode& node = nodes[stack[--sp]];
        double entry = 0.0;
        if (!gpu_slab_hit(node, o, d, t_min, entry)) continue;
        if (found && entry > best_t) continue;
        if (node.count > 0) {
            for (uint32_t k = 0; k < node.count; ++k) {
                uint32_t t = order[node.start + k];
                if (t == skip_tri) continue;
                const double* a = &tri_verts[9 * t];
                const double* b = &tri_verts[9 * t + 3];
                const double* c = &tri_verts[9 * t + 6];
                double hit_t;
                if (gpu_ray_triangle(o, d, a, b, c, t_min, hit_t)) {
                    if (!found || hit_t < best_t || (hit_t == best_t && t < hit_tri)) {
                        best_t = hit_t;
                        hit_tri = t;
                        found = true;
                    }
                }
            }
        } else {
            stack[sp++] = node.left;
            stack[sp++] = node.right;
        }
    }
    return found;
}

// --- Main GPU Kernel ---

template <typename Real>
__global__ void po_kernel(const Real* __restrict__ centroids, // [3N]
                          const Real* __restrict__ normals,   // [3N]
                          const Real* __restrict__ areas,     // [N]
                          const uint32_t* __restrict__ mat_indices, // [N] or null
                          const UnitParams<Real>* __restrict__ units, // [U]
                          GpuBvhData bvh_data,
                          const double* __restrict__ mesh_centroids, // [3*N_mesh] or null
                          const double* __restrict__ mesh_normals,   // [3*N_mesh] or null
                          const double* __restrict__ mesh_areas,     // [N_mesh] or null
                          Cplx<Real>* __restrict__ out,       // [6U] F per tx
                          int* __restrict__ lit_out,          // [U] lit facets
                          int ntris,
                          int n_mesh_tris) {
    const int u = static_cast<int>(blockIdx.x);
    const UnitParams<Real> p = units[u];
    __shared__ Real sh[2][3][2 * kBlock]; // [tx][xyz][re/im interleave]
    __shared__ int lit_sh[kBlock];
    Cplx<Real> sum[2][3] = {};
    int lit = 0;

    const Real to_inc_x = -p.kx, to_inc_y = -p.ky, to_inc_z = -p.kz;

    for (int t = static_cast<int>(threadIdx.x); t < ntris; t += kBlock) {
        const Real nx = normals[3 * t], ny = normals[3 * t + 1], nz = normals[3 * t + 2];
        const Real cx = centroids[3 * t], cy = centroids[3 * t + 1], cz = centroids[3 * t + 2];

        bool faces = (nx * to_inc_x + ny * to_inc_y + nz * to_inc_z > Real(0));
        bool is_lit = faces;

        // 1. GO Shadowing check
        if (is_lit && bvh_data.shadowing && bvh_data.nodes != nullptr) {
            double orig[3] = {static_cast<double>(cx) + bvh_data.phase_ref[0],
                              static_cast<double>(cy) + bvh_data.phase_ref[1],
                              static_cast<double>(cz) + bvh_data.phase_ref[2]};
            double obs[3] = {static_cast<double>(p.rx), static_cast<double>(p.ry), static_cast<double>(p.rz)};
            if (gpu_occluded_bvh(bvh_data.nodes, bvh_data.order, bvh_data.tri_vertices,
                                 orig, obs, bvh_data.t_min, bvh_data.scene_limit, static_cast<uint32_t>(t))) {
                is_lit = false;
            }
        }

        // 2. Single-bounce PO accumulation if lit
        if (is_lit) {
            ++lit;

            int mat_id = 0;
            uint8_t wall_type = 0; // default PEC
            if (p.has_materials) {
                uint32_t midx = mat_indices ? mat_indices[t] : 0;
                if (midx < static_cast<uint32_t>(p.num_materials)) {
                    mat_id = static_cast<int>(midx);
                    wall_type = p.materials[mat_id].wall_type;
                }
            }

            if (wall_type == 0) {
                // Fast PEC path
                const Real phi = p.qx * cx + p.qy * cy + p.qz * cz;
                Real s_phi, c_phi;
                fsincos(phi, &s_phi, &c_phi);

                const Real w = p.k_eta_4pi * areas[t];
                const Real w_re = -w * s_phi;
                const Real w_im =  w * c_phi;

                // tx = 0: H transmit
                {
                    const Real j0_0 = Real(2) * (ny * p.h0_h[2] - nz * p.h0_h[1]);
                    const Real j0_1 = Real(2) * (nz * p.h0_h[0] - nx * p.h0_h[2]);
                    const Real j0_2 = Real(2) * (nx * p.h0_h[1] - ny * p.h0_h[0]);
                    const Real r_dot_j0 = p.rx * j0_0 + p.ry * j0_1 + p.rz * j0_2;
                    const Real t0_0 = r_dot_j0 * p.rx - j0_0;
                    const Real t0_1 = r_dot_j0 * p.ry - j0_1;
                    const Real t0_2 = r_dot_j0 * p.rz - j0_2;
                    sum[0][0].re += w_re * t0_0;  sum[0][0].im += w_im * t0_0;
                    sum[0][1].re += w_re * t0_1;  sum[0][1].im += w_im * t0_1;
                    sum[0][2].re += w_re * t0_2;  sum[0][2].im += w_im * t0_2;
                }

                // tx = 1: V transmit
                {
                    const Real j0_0 = Real(2) * (ny * p.h0_v[2] - nz * p.h0_v[1]);
                    const Real j0_1 = Real(2) * (nz * p.h0_v[0] - nx * p.h0_v[2]);
                    const Real j0_2 = Real(2) * (nx * p.h0_v[1] - ny * p.h0_v[0]);
                    const Real r_dot_j0 = p.rx * j0_0 + p.ry * j0_1 + p.rz * j0_2;
                    const Real t0_0 = r_dot_j0 * p.rx - j0_0;
                    const Real t0_1 = r_dot_j0 * p.ry - j0_1;
                    const Real t0_2 = r_dot_j0 * p.rz - j0_2;
                    sum[1][0].re += w_re * t0_0;  sum[1][0].im += w_im * t0_0;
                    sum[1][1].re += w_re * t0_1;  sum[1][1].im += w_im * t0_1;
                    sum[1][2].re += w_re * t0_2;  sum[1][2].im += w_im * t0_2;
                }
            } else {
                // Material path (Impedance, Dielectric, Coated)
                const GpuMaterialModel& mat = p.materials[mat_id];
                Real dot_n_r = nx * p.rx + ny * p.ry + nz * p.rz;
                Real cos_i = (dot_n_r < Real(0)) ? Real(0) : ((dot_n_r > Real(1)) ? Real(1) : dot_n_r);

                Cplx<Real> r_te = {Real(-1), Real(0)};
                Cplx<Real> r_tm = {Real(1), Real(0)};

                if (wall_type == 1) { // Dielectric
                    Cplx<Real> eps_inc = {Real(1), Real(0)};
                    Cplx<Real> mu_inc = {Real(1), Real(0)};
                    Cplx<Real> eps_w = {Real(mat.eps_r.re), Real(mat.eps_r.im)};
                    Cplx<Real> mu_w = {Real(mat.mu_r.re), Real(mat.mu_r.im)};
                    dev_fresnel(eps_inc, mu_inc, eps_w, mu_w, cos_i, r_te, r_tm);
                } else if (wall_type == 2) { // Coated
                    Cplx<Real> eps_inc = {Real(1), Real(0)};
                    Cplx<Real> mu_inc = {Real(1), Real(0)};
                    dev_coated_pec_reflection(eps_inc, mu_inc, mat.layers, mat.num_layers,
                                              cos_i, p.lambda_m, r_te, r_tm);
                } else if (wall_type == 3) { // Impedance
                    Cplx<Real> eta_s = {Real(mat.eta_s.re), Real(mat.eta_s.im)};
                    dev_impedance_reflection(eta_s, cos_i, r_te, r_tm);
                }

                Real n_vec[3] = {nx, ny, nz};
                Real k_vec[3] = {p.kx, p.ky, p.kz};
                Real r_vec[3] = {p.rx, p.ry, p.rz};
                Real c_vec[3] = {cx, cy, cz};

                Real ein_phi = -p.k * (p.kx * cx + p.ky * cy + p.kz * cz);
                Real s_ein, c_ein;
                fsincos(ein_phi, &s_ein, &c_ein);
                Cplx<Real> ein_c = {c_ein, s_ein};

                // tx = 0: H transmit
                {
                    Cplx<Real> e_inc[3] = {c_scale(ein_c, p.ehx), c_scale(ein_c, p.ehy), c_scale(ein_c, p.ehz)};
                    Cplx<Real> f_mat[3];
                    dev_material_facet_field(n_vec, k_vec, r_vec, e_inc, p.k, p.eta, areas[t], c_vec, r_te, r_tm, f_mat);
                    for (int i = 0; i < 3; ++i) {
                        sum[0][i].re += f_mat[i].re;
                        sum[0][i].im += f_mat[i].im;
                    }
                }
                // tx = 1: V transmit
                {
                    Cplx<Real> e_inc[3] = {c_scale(ein_c, p.evx), c_scale(ein_c, p.evy), c_scale(ein_c, p.evz)};
                    Cplx<Real> f_mat[3];
                    dev_material_facet_field(n_vec, k_vec, r_vec, e_inc, p.k, p.eta, areas[t], c_vec, r_te, r_tm, f_mat);
                    for (int i = 0; i < 3; ++i) {
                        sum[1][i].re += f_mat[i].re;
                        sum[1][i].im += f_mat[i].im;
                    }
                }
            }
        }

        // 3. Multi-bounce SBR tracking (bounce 2 and bounce 3)
        if (bvh_data.max_bounces >= 2 && bvh_data.nodes != nullptr && t < n_mesh_tris) {
            Real dot_na_r = nx * p.rx + ny * p.ry + nz * p.rz;
            if (dot_na_r > Real(0)) {
                double ca_w[3] = {static_cast<double>(cx) + bvh_data.phase_ref[0],
                                  static_cast<double>(cy) + bvh_data.phase_ref[1],
                                  static_cast<double>(cz) + bvh_data.phase_ref[2]};
                double s_dot_na = static_cast<double>(p.kx * nx + p.ky * ny + p.kz * nz);
                double d1[3] = {static_cast<double>(p.kx) - 2.0 * s_dot_na * static_cast<double>(nx),
                                static_cast<double>(p.ky) - 2.0 * s_dot_na * static_cast<double>(ny),
                                static_cast<double>(p.kz) - 2.0 * s_dot_na * static_cast<double>(nz)};

                uint32_t ib = 0;
                double hit_t1 = 0.0;
                if (gpu_ray_bvh(bvh_data.nodes, bvh_data.order, bvh_data.tri_vertices,
                                ca_w, d1, bvh_data.t_min, static_cast<uint32_t>(t), ib, hit_t1) &&
                    ib != static_cast<uint32_t>(t)) {

                    const double* nb_ptr = (mesh_normals != nullptr) ? &mesh_normals[3 * ib] : &reinterpret_cast<const double*>(normals)[3 * ib];
                    double dot_nb_d1 = nb_ptr[0] * d1[0] + nb_ptr[1] * d1[1] + nb_ptr[2] * d1[2];
                    if (dot_nb_d1 < 0.0) {
                        const double* cb_ptr = (mesh_centroids != nullptr) ? &mesh_centroids[3 * ib] : &reinterpret_cast<const double*>(centroids)[3 * ib];
                        double cb_w[3] = {cb_ptr[0] + bvh_data.phase_ref[0],
                                          cb_ptr[1] + bvh_data.phase_ref[1],
                                          cb_ptr[2] + bvh_data.phase_ref[2]};
                        double area_b = (mesh_areas != nullptr) ? mesh_areas[ib] : static_cast<double>(areas[ib]);

                        double d2[3] = {d1[0] - 2.0 * dot_nb_d1 * nb_ptr[0],
                                        d1[1] - 2.0 * dot_nb_d1 * nb_ptr[1],
                                        d1[2] - 2.0 * dot_nb_d1 * nb_ptr[2]};

                        double dot_d2_r = d2[0] * static_cast<double>(p.rx) +
                                          d2[1] * static_cast<double>(p.ry) +
                                          d2[2] * static_cast<double>(p.rz);

                        bool retro2 = (dot_d2_r > 1.0 - 1e-9);
                        if (retro2) {
                            double r_vec[3] = {static_cast<double>(p.rx), static_cast<double>(p.ry), static_cast<double>(p.rz)};
                            bool exit_occ = gpu_occluded_bvh(bvh_data.nodes, bvh_data.order, bvh_data.tri_vertices,
                                                             cb_w, r_vec, bvh_data.t_min, bvh_data.scene_limit, ib);
                            if (!exit_occ) {
                                double s_dot_ca_ref = static_cast<double>(p.kx) * static_cast<double>(cx) +
                                                      static_cast<double>(p.ky) * static_cast<double>(cy) +
                                                      static_cast<double>(p.kz) * static_cast<double>(cz);
                                double d1_dot_cb_ca = d1[0] * (cb_ptr[0] - static_cast<double>(cx)) +
                                                      d1[1] * (cb_ptr[1] - static_cast<double>(cy)) +
                                                      d1[2] * (cb_ptr[2] - static_cast<double>(cz));
                                double r_dot_cb_ref = static_cast<double>(p.rx) * cb_ptr[0] +
                                                      static_cast<double>(p.ry) * cb_ptr[1] +
                                                      static_cast<double>(p.rz) * cb_ptr[2];

                                double phi_2b = -static_cast<double>(p.k) * (s_dot_ca_ref + d1_dot_cb_ca) +
                                                 static_cast<double>(p.k) * r_dot_cb_ref;
                                Real s_phi2, c_phi2;
                                fsincos(static_cast<Real>(phi_2b), &s_phi2, &c_phi2);

                                Real amp_val = static_cast<Real>(static_cast<double>(p.k) / (4.0 * 3.14159265358979323846) * area_b);
                                Real amp_re = -amp_val * s_phi2;
                                Real amp_im =  amp_val * c_phi2;

                                // tx = 0: H transmit
                                {
                                    Real ein[3] = {p.ehx / p.e0, p.ehy / p.e0, p.ehz / p.e0};
                                    Real dot_e_na = ein[0] * nx + ein[1] * ny + ein[2] * nz;
                                    Real e1[3] = {Real(2) * dot_e_na * nx - ein[0],
                                                  Real(2) * dot_e_na * ny - ein[1],
                                                  Real(2) * dot_e_na * nz - ein[2]};
                                    Real hb[3] = {static_cast<Real>(d1[1]) * e1[2] - static_cast<Real>(d1[2]) * e1[1],
                                                  static_cast<Real>(d1[2]) * e1[0] - static_cast<Real>(d1[0]) * e1[2],
                                                  static_cast<Real>(d1[0]) * e1[1] - static_cast<Real>(d1[1]) * e1[0]};
                                    Real nb_r[3] = {static_cast<Real>(nb_ptr[0]), static_cast<Real>(nb_ptr[1]), static_cast<Real>(nb_ptr[2])};
                                    Real jb[3] = {Real(2) * (nb_r[1] * hb[2] - nb_r[2] * hb[1]),
                                                  Real(2) * (nb_r[2] * hb[0] - nb_r[0] * hb[2]),
                                                  Real(2) * (nb_r[0] * hb[1] - nb_r[1] * hb[0])};
                                    Real r_dot_jb = p.rx * jb[0] + p.ry * jb[1] + p.rz * jb[2];
                                    Real t2[3] = {r_dot_jb * p.rx - jb[0],
                                                  r_dot_jb * p.ry - jb[1],
                                                  r_dot_jb * p.rz - jb[2]};
                                    for (int i = 0; i < 3; ++i) {
                                        Real term = t2[i] * p.e0;
                                        sum[0][i].re += amp_re * term;
                                        sum[0][i].im += amp_im * term;
                                    }
                                }
                                // tx = 1: V transmit
                                {
                                    Real ein[3] = {p.evx / p.e0, p.evy / p.e0, p.evz / p.e0};
                                    Real dot_e_na = ein[0] * nx + ein[1] * ny + ein[2] * nz;
                                    Real e1[3] = {Real(2) * dot_e_na * nx - ein[0],
                                                  Real(2) * dot_e_na * ny - ein[1],
                                                  Real(2) * dot_e_na * nz - ein[2]};
                                    Real hb[3] = {static_cast<Real>(d1[1]) * e1[2] - static_cast<Real>(d1[2]) * e1[1],
                                                  static_cast<Real>(d1[2]) * e1[0] - static_cast<Real>(d1[0]) * e1[2],
                                                  static_cast<Real>(d1[0]) * e1[1] - static_cast<Real>(d1[1]) * e1[0]};
                                    Real nb_r[3] = {static_cast<Real>(nb_ptr[0]), static_cast<Real>(nb_ptr[1]), static_cast<Real>(nb_ptr[2])};
                                    Real jb[3] = {Real(2) * (nb_r[1] * hb[2] - nb_r[2] * hb[1]),
                                                  Real(2) * (nb_r[2] * hb[0] - nb_r[0] * hb[2]),
                                                  Real(2) * (nb_r[0] * hb[1] - nb_r[1] * hb[0])};
                                    Real r_dot_jb = p.rx * jb[0] + p.ry * jb[1] + p.rz * jb[2];
                                    Real t2[3] = {r_dot_jb * p.rx - jb[0],
                                                  r_dot_jb * p.ry - jb[1],
                                                  r_dot_jb * p.rz - jb[2]};
                                    for (int i = 0; i < 3; ++i) {
                                        Real term = t2[i] * p.e0;
                                        sum[1][i].re += amp_re * term;
                                        sum[1][i].im += amp_im * term;
                                    }
                                }
                            }
                        } else if (bvh_data.max_bounces >= 3) {
                            uint32_t ic = 0;
                            double hit_t2 = 0.0;
                            if (gpu_ray_bvh(bvh_data.nodes, bvh_data.order, bvh_data.tri_vertices,
                                            cb_w, d2, bvh_data.t_min, ib, ic, hit_t2) &&
                                ic != ib) {
                                const double* nc_ptr = (mesh_normals != nullptr) ? &mesh_normals[3 * ic] : &reinterpret_cast<const double*>(normals)[3 * ic];
                                double dot_nc_d2 = nc_ptr[0] * d2[0] + nc_ptr[1] * d2[1] + nc_ptr[2] * d2[2];
                                if (dot_nc_d2 < 0.0) {
                                    const double* cc_ptr = (mesh_centroids != nullptr) ? &mesh_centroids[3 * ic] : &reinterpret_cast<const double*>(centroids)[3 * ic];
                                    double cc_w[3] = {cc_ptr[0] + bvh_data.phase_ref[0],
                                                      cc_ptr[1] + bvh_data.phase_ref[1],
                                                      cc_ptr[2] + bvh_data.phase_ref[2]};
                                    double area_c = (mesh_areas != nullptr) ? mesh_areas[ic] : static_cast<double>(areas[ic]);

                                    double d3[3] = {d2[0] - 2.0 * dot_nc_d2 * nc_ptr[0],
                                                    d2[1] - 2.0 * dot_nc_d2 * nc_ptr[1],
                                                    d2[2] - 2.0 * dot_nc_d2 * nc_ptr[2]};

                                    double dot_d3_r = d3[0] * static_cast<double>(p.rx) +
                                                      d3[1] * static_cast<double>(p.ry) +
                                                      d3[2] * static_cast<double>(p.rz);

                                    if (dot_d3_r > 1.0 - 1e-9) {
                                        double r_vec[3] = {static_cast<double>(p.rx), static_cast<double>(p.ry), static_cast<double>(p.rz)};
                                        bool exit_occ3 = gpu_occluded_bvh(bvh_data.nodes, bvh_data.order, bvh_data.tri_vertices,
                                                                         cc_w, r_vec, bvh_data.t_min, bvh_data.scene_limit, ic);
                                        if (!exit_occ3) {
                                            double s_dot_ca_ref = static_cast<double>(p.kx) * static_cast<double>(cx) +
                                                                  static_cast<double>(p.ky) * static_cast<double>(cy) +
                                                                  static_cast<double>(p.kz) * static_cast<double>(cz);
                                            double d1_dot_cb_ca = d1[0] * (cb_ptr[0] - static_cast<double>(cx)) +
                                                                  d1[1] * (cb_ptr[1] - static_cast<double>(cy)) +
                                                                  d1[2] * (cb_ptr[2] - static_cast<double>(cz));
                                            double d2_dot_cc_cb = d2[0] * (cc_ptr[0] - cb_ptr[0]) +
                                                                  d2[1] * (cc_ptr[1] - cb_ptr[1]) +
                                                                  d2[2] * (cc_ptr[2] - cb_ptr[2]);
                                            double r_dot_cc_ref = static_cast<double>(p.rx) * cc_ptr[0] +
                                                                  static_cast<double>(p.ry) * cc_ptr[1] +
                                                                  static_cast<double>(p.rz) * cc_ptr[2];

                                            double phi_3b = -static_cast<double>(p.k) * (s_dot_ca_ref + d1_dot_cb_ca + d2_dot_cc_cb) +
                                                             static_cast<double>(p.k) * r_dot_cc_ref;
                                            Real s_phi3, c_phi3;
                                            fsincos(static_cast<Real>(phi_3b), &s_phi3, &c_phi3);

                                            Real amp_val3 = static_cast<Real>(static_cast<double>(p.k) / (4.0 * 3.14159265358979323846) * area_c);
                                            Real amp_re3 = -amp_val3 * s_phi3;
                                            Real amp_im3 =  amp_val3 * c_phi3;

                                            // tx = 0: H transmit
                                            {
                                                Real ein[3] = {p.ehx / p.e0, p.ehy / p.e0, p.ehz / p.e0};
                                                Real dot_e_na = ein[0] * nx + ein[1] * ny + ein[2] * nz;
                                                Real e1[3] = {Real(2) * dot_e_na * nx - ein[0],
                                                              Real(2) * dot_e_na * ny - ein[1],
                                                              Real(2) * dot_e_na * nz - ein[2]};
                                                Real nb_r[3] = {static_cast<Real>(nb_ptr[0]), static_cast<Real>(nb_ptr[1]), static_cast<Real>(nb_ptr[2])};
                                                Real dot_e1_nb = e1[0] * nb_r[0] + e1[1] * nb_r[1] + e1[2] * nb_r[2];
                                                Real e2[3] = {Real(2) * dot_e1_nb * nb_r[0] - e1[0],
                                                              Real(2) * dot_e1_nb * nb_r[1] - e1[1],
                                                              Real(2) * dot_e1_nb * nb_r[2] - e1[2]};
                                                Real hc[3] = {static_cast<Real>(d2[1]) * e2[2] - static_cast<Real>(d2[2]) * e2[1],
                                                              static_cast<Real>(d2[2]) * e2[0] - static_cast<Real>(d2[0]) * e2[2],
                                                              static_cast<Real>(d2[0]) * e2[1] - static_cast<Real>(d2[1]) * e2[0]};
                                                Real nc_r[3] = {static_cast<Real>(nc_ptr[0]), static_cast<Real>(nc_ptr[1]), static_cast<Real>(nc_ptr[2])};
                                                Real jc[3] = {Real(2) * (nc_r[1] * hc[2] - nc_r[2] * hc[1]),
                                                              Real(2) * (nc_r[2] * hc[0] - nc_r[0] * hc[2]),
                                                              Real(2) * (nc_r[0] * hc[1] - nc_r[1] * hc[0])};
                                                Real r_dot_jc = p.rx * jc[0] + p.ry * jc[1] + p.rz * jc[2];
                                                Real t2[3] = {r_dot_jc * p.rx - jc[0],
                                                              r_dot_jc * p.ry - jc[1],
                                                              r_dot_jc * p.rz - jc[2]};
                                                for (int i = 0; i < 3; ++i) {
                                                    Real term = t2[i] * p.e0;
                                                    sum[0][i].re += amp_re3 * term;
                                                    sum[0][i].im += amp_im3 * term;
                                                }
                                            }
                                            // tx = 1: V transmit
                                            {
                                                Real ein[3] = {p.evx / p.e0, p.evy / p.e0, p.evz / p.e0};
                                                Real dot_e_na = ein[0] * nx + ein[1] * ny + ein[2] * nz;
                                                Real e1[3] = {Real(2) * dot_e_na * nx - ein[0],
                                                              Real(2) * dot_e_na * ny - ein[1],
                                                              Real(2) * dot_e_na * nz - ein[2]};
                                                Real nb_r[3] = {static_cast<Real>(nb_ptr[0]), static_cast<Real>(nb_ptr[1]), static_cast<Real>(nb_ptr[2])};
                                                Real dot_e1_nb = e1[0] * nb_r[0] + e1[1] * nb_r[1] + e1[2] * nb_r[2];
                                                Real e2[3] = {Real(2) * dot_e1_nb * nb_r[0] - e1[0],
                                                              Real(2) * dot_e1_nb * nb_r[1] - e1[1],
                                                              Real(2) * dot_e1_nb * nb_r[2] - e1[2]};
                                                Real hc[3] = {static_cast<Real>(d2[1]) * e2[2] - static_cast<Real>(d2[2]) * e2[1],
                                                              static_cast<Real>(d2[2]) * e2[0] - static_cast<Real>(d2[0]) * e2[2],
                                                              static_cast<Real>(d2[0]) * e2[1] - static_cast<Real>(d2[1]) * e2[0]};
                                                Real nc_r[3] = {static_cast<Real>(nc_ptr[0]), static_cast<Real>(nc_ptr[1]), static_cast<Real>(nc_ptr[2])};
                                                Real jc[3] = {Real(2) * (nc_r[1] * hc[2] - nc_r[2] * hc[1]),
                                                              Real(2) * (nc_r[2] * hc[0] - nc_r[0] * hc[2]),
                                                              Real(2) * (nc_r[0] * hc[1] - nc_r[1] * hc[0])};
                                                Real r_dot_jc = p.rx * jc[0] + p.ry * jc[1] + p.rz * jc[2];
                                                Real t2[3] = {r_dot_jc * p.rx - jc[0],
                                                              r_dot_jc * p.ry - jc[1],
                                                              r_dot_jc * p.rz - jc[2]};
                                                for (int i = 0; i < 3; ++i) {
                                                    Real term = t2[i] * p.e0;
                                                    sum[1][i].re += amp_re3 * term;
                                                    sum[1][i].im += amp_im3 * term;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    const int lane = static_cast<int>(threadIdx.x);
    for (int tx = 0; tx < 2; ++tx)
        for (int i = 0; i < 3; ++i) {
            sh[tx][i][2 * lane] = sum[tx][i].re;
            sh[tx][i][2 * lane + 1] = sum[tx][i].im;
        }
    lit_sh[lane] = lit;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            for (int tx = 0; tx < 2; ++tx)
                for (int i = 0; i < 3; ++i) {
                    sh[tx][i][2 * lane] += sh[tx][i][2 * (lane + stride)];
                    sh[tx][i][2 * lane + 1] += sh[tx][i][2 * (lane + stride) + 1];
                }
            lit_sh[lane] += lit_sh[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        for (int tx = 0; tx < 2; ++tx)
            for (int i = 0; i < 3; ++i) {
                out[6 * u + 3 * tx + i].re = sh[tx][i][0];
                out[6 * u + 3 * tx + i].im = sh[tx][i][1];
            }
        lit_out[u] = lit_sh[0];
    }
}

// Mixed precision kernel
__global__ void po_mixed_kernel(const float* __restrict__ centroids, // [3N]
                                const float* __restrict__ normals,   // [3N]
                                const float* __restrict__ areas,     // [N]
                                const uint32_t* __restrict__ mat_indices,
                                const UnitParams<double>* __restrict__ units, // [U]
                                GpuBvhData bvh_data,
                                const double* __restrict__ mesh_centroids,
                                const double* __restrict__ mesh_normals,
                                const double* __restrict__ mesh_areas,
                                Cplx<double>* __restrict__ out,       // [6U] F per tx
                                int* __restrict__ lit_out,          // [U] lit facets
                                int ntris,
                                int n_mesh_tris) {
    const int u = static_cast<int>(blockIdx.x);
    const UnitParams<double> p = units[u];
    __shared__ double sh[2][3][2 * kBlock];
    __shared__ int lit_sh[kBlock];
    Cplx<double> sum[2][3] = {};
    int lit = 0;

    const double to_inc_x = -p.kx, to_inc_y = -p.ky, to_inc_z = -p.kz;

    for (int t = static_cast<int>(threadIdx.x); t < ntris; t += kBlock) {
        const double nx = static_cast<double>(normals[3 * t]);
        const double ny = static_cast<double>(normals[3 * t + 1]);
        const double nz = static_cast<double>(normals[3 * t + 2]);
        const double cx = static_cast<double>(centroids[3 * t]);
        const double cy = static_cast<double>(centroids[3 * t + 1]);
        const double cz = static_cast<double>(centroids[3 * t + 2]);
        const double area_t = static_cast<double>(areas[t]);

        bool faces = (nx * to_inc_x + ny * to_inc_y + nz * to_inc_z > 0.0);
        bool is_lit = faces;

        if (is_lit && bvh_data.shadowing && bvh_data.nodes != nullptr) {
            double orig[3] = {cx + bvh_data.phase_ref[0],
                              cy + bvh_data.phase_ref[1],
                              cz + bvh_data.phase_ref[2]};
            double obs[3] = {p.rx, p.ry, p.rz};
            if (gpu_occluded_bvh(bvh_data.nodes, bvh_data.order, bvh_data.tri_vertices,
                                 orig, obs, bvh_data.t_min, bvh_data.scene_limit, static_cast<uint32_t>(t))) {
                is_lit = false;
            }
        }

        if (is_lit) {
            ++lit;

            int mat_id = 0;
            uint8_t wall_type = 0;
            if (p.has_materials) {
                uint32_t midx = mat_indices ? mat_indices[t] : 0;
                if (midx < static_cast<uint32_t>(p.num_materials)) {
                    mat_id = static_cast<int>(midx);
                    wall_type = p.materials[mat_id].wall_type;
                }
            }

            if (wall_type == 0) {
                const double phi = p.qx * cx + p.qy * cy + p.qz * cz;
                double s_phi, c_phi;
                fsincos(phi, &s_phi, &c_phi);

                const double w = p.k_eta_4pi * area_t;
                const double w_re = -w * s_phi;
                const double w_im =  w * c_phi;

                // tx = 0: H transmit
                {
                    const double j0_0 = 2.0 * (ny * p.h0_h[2] - nz * p.h0_h[1]);
                    const double j0_1 = 2.0 * (nz * p.h0_h[0] - nx * p.h0_h[2]);
                    const double j0_2 = 2.0 * (nx * p.h0_h[1] - ny * p.h0_h[0]);
                    const double r_dot_j0 = p.rx * j0_0 + p.ry * j0_1 + p.rz * j0_2;
                    const double t0_0 = r_dot_j0 * p.rx - j0_0;
                    const double t0_1 = r_dot_j0 * p.ry - j0_1;
                    const double t0_2 = r_dot_j0 * p.rz - j0_2;
                    sum[0][0].re += w_re * t0_0;  sum[0][0].im += w_im * t0_0;
                    sum[0][1].re += w_re * t0_1;  sum[0][1].im += w_im * t0_1;
                    sum[0][2].re += w_re * t0_2;  sum[0][2].im += w_im * t0_2;
                }

                // tx = 1: V transmit
                {
                    const double j0_0 = 2.0 * (ny * p.h0_v[2] - nz * p.h0_v[1]);
                    const double j0_1 = 2.0 * (nz * p.h0_v[0] - nx * p.h0_v[2]);
                    const double j0_2 = 2.0 * (nx * p.h0_v[1] - ny * p.h0_v[0]);
                    const double r_dot_j0 = p.rx * j0_0 + p.ry * j0_1 + p.rz * j0_2;
                    const double t0_0 = r_dot_j0 * p.rx - j0_0;
                    const double t0_1 = r_dot_j0 * p.ry - j0_1;
                    const double t0_2 = r_dot_j0 * p.rz - j0_2;
                    sum[1][0].re += w_re * t0_0;  sum[1][0].im += w_im * t0_0;
                    sum[1][1].re += w_re * t0_1;  sum[1][1].im += w_im * t0_1;
                    sum[1][2].re += w_re * t0_2;  sum[1][2].im += w_im * t0_2;
                }
            } else {
                const GpuMaterialModel& mat = p.materials[mat_id];
                double dot_n_r = nx * p.rx + ny * p.ry + nz * p.rz;
                double cos_i = (dot_n_r < 0.0) ? 0.0 : ((dot_n_r > 1.0) ? 1.0 : dot_n_r);

                Cplx<double> r_te = {-1.0, 0.0};
                Cplx<double> r_tm = {1.0, 0.0};

                if (wall_type == 1) {
                    Cplx<double> eps_inc = {1.0, 0.0};
                    Cplx<double> mu_inc = {1.0, 0.0};
                    dev_fresnel(eps_inc, mu_inc, mat.eps_r, mat.mu_r, cos_i, r_te, r_tm);
                } else if (wall_type == 2) {
                    Cplx<double> eps_inc = {1.0, 0.0};
                    Cplx<double> mu_inc = {1.0, 0.0};
                    dev_coated_pec_reflection(eps_inc, mu_inc, mat.layers, mat.num_layers,
                                              cos_i, p.lambda_m, r_te, r_tm);
                } else if (wall_type == 3) {
                    dev_impedance_reflection(mat.eta_s, cos_i, r_te, r_tm);
                }

                double n_vec[3] = {nx, ny, nz};
                double k_vec[3] = {p.kx, p.ky, p.kz};
                double r_vec[3] = {p.rx, p.ry, p.rz};
                double c_vec[3] = {cx, cy, cz};

                double ein_phi = -p.k * (p.kx * cx + p.ky * cy + p.kz * cz);
                double s_ein, c_ein;
                fsincos(ein_phi, &s_ein, &c_ein);
                Cplx<double> ein_c = {c_ein, s_ein};

                // tx = 0: H transmit
                {
                    Cplx<double> e_inc[3] = {c_scale(ein_c, p.ehx), c_scale(ein_c, p.ehy), c_scale(ein_c, p.ehz)};
                    Cplx<double> f_mat[3];
                    dev_material_facet_field(n_vec, k_vec, r_vec, e_inc, p.k, p.eta, area_t, c_vec, r_te, r_tm, f_mat);
                    for (int i = 0; i < 3; ++i) {
                        sum[0][i].re += f_mat[i].re;
                        sum[0][i].im += f_mat[i].im;
                    }
                }
                // tx = 1: V transmit
                {
                    Cplx<double> e_inc[3] = {c_scale(ein_c, p.evx), c_scale(ein_c, p.evy), c_scale(ein_c, p.evz)};
                    Cplx<double> f_mat[3];
                    dev_material_facet_field(n_vec, k_vec, r_vec, e_inc, p.k, p.eta, area_t, c_vec, r_te, r_tm, f_mat);
                    for (int i = 0; i < 3; ++i) {
                        sum[1][i].re += f_mat[i].re;
                        sum[1][i].im += f_mat[i].im;
                    }
                }
            }
        }

        // Multi-bounce SBR tracking
        if (bvh_data.max_bounces >= 2 && bvh_data.nodes != nullptr && t < n_mesh_tris) {
            double dot_na_r = nx * p.rx + ny * p.ry + nz * p.rz;
            if (dot_na_r > 0.0) {
                double ca_w[3] = {cx + bvh_data.phase_ref[0],
                                  cy + bvh_data.phase_ref[1],
                                  cz + bvh_data.phase_ref[2]};
                double s_dot_na = p.kx * nx + p.ky * ny + p.kz * nz;
                double d1[3] = {p.kx - 2.0 * s_dot_na * nx,
                                p.ky - 2.0 * s_dot_na * ny,
                                p.kz - 2.0 * s_dot_na * nz};

                uint32_t ib = 0;
                double hit_t1 = 0.0;
                if (gpu_ray_bvh(bvh_data.nodes, bvh_data.order, bvh_data.tri_vertices,
                                ca_w, d1, bvh_data.t_min, static_cast<uint32_t>(t), ib, hit_t1) &&
                    ib != static_cast<uint32_t>(t)) {

                    const double* nb_ptr = (mesh_normals != nullptr) ? &mesh_normals[3 * ib] : &nx;
                    double dot_nb_d1 = nb_ptr[0] * d1[0] + nb_ptr[1] * d1[1] + nb_ptr[2] * d1[2];
                    if (dot_nb_d1 < 0.0) {
                        const double* cb_ptr = (mesh_centroids != nullptr) ? &mesh_centroids[3 * ib] : &cx;
                        double cb_w[3] = {cb_ptr[0] + bvh_data.phase_ref[0],
                                          cb_ptr[1] + bvh_data.phase_ref[1],
                                          cb_ptr[2] + bvh_data.phase_ref[2]};
                        double area_b = (mesh_areas != nullptr) ? mesh_areas[ib] : area_t;

                        double d2[3] = {d1[0] - 2.0 * dot_nb_d1 * nb_ptr[0],
                                        d1[1] - 2.0 * dot_nb_d1 * nb_ptr[1],
                                        d1[2] - 2.0 * dot_nb_d1 * nb_ptr[2]};

                        double dot_d2_r = d2[0] * p.rx + d2[1] * p.ry + d2[2] * p.rz;
                        bool retro2 = (dot_d2_r > 1.0 - 1e-9);
                        if (retro2) {
                            double r_vec[3] = {p.rx, p.ry, p.rz};
                            bool exit_occ = gpu_occluded_bvh(bvh_data.nodes, bvh_data.order, bvh_data.tri_vertices,
                                                             cb_w, r_vec, bvh_data.t_min, bvh_data.scene_limit, ib);
                            if (!exit_occ) {
                                double s_dot_ca_ref = p.kx * cx + p.ky * cy + p.kz * cz;
                                double d1_dot_cb_ca = d1[0] * (cb_ptr[0] - cx) +
                                                      d1[1] * (cb_ptr[1] - cy) +
                                                      d1[2] * (cb_ptr[2] - cz);
                                double r_dot_cb_ref = p.rx * cb_ptr[0] + p.ry * cb_ptr[1] + p.rz * cb_ptr[2];

                                double phi_2b = -p.k * (s_dot_ca_ref + d1_dot_cb_ca) + p.k * r_dot_cb_ref;
                                double s_phi2, c_phi2;
                                fsincos(phi_2b, &s_phi2, &c_phi2);

                                double amp_val = p.k / (4.0 * 3.14159265358979323846) * area_b;
                                double amp_re = -amp_val * s_phi2;
                                double amp_im =  amp_val * c_phi2;

                                // tx = 0: H transmit
                                {
                                    double ein[3] = {p.ehx / p.e0, p.ehy / p.e0, p.ehz / p.e0};
                                    double dot_e_na = ein[0] * nx + ein[1] * ny + ein[2] * nz;
                                    double e1[3] = {2.0 * dot_e_na * nx - ein[0],
                                                  2.0 * dot_e_na * ny - ein[1],
                                                  2.0 * dot_e_na * nz - ein[2]};
                                    double hb[3] = {d1[1] * e1[2] - d1[2] * e1[1],
                                                  d1[2] * e1[0] - d1[0] * e1[2],
                                                  d1[0] * e1[1] - d1[1] * e1[0]};
                                    double nb_r[3] = {nb_ptr[0], nb_ptr[1], nb_ptr[2]};
                                    double jb[3] = {2.0 * (nb_r[1] * hb[2] - nb_r[2] * hb[1]),
                                                  2.0 * (nb_r[2] * hb[0] - nb_r[0] * hb[2]),
                                                  2.0 * (nb_r[0] * hb[1] - nb_r[1] * hb[0])};
                                    double r_dot_jb = p.rx * jb[0] + p.ry * jb[1] + p.rz * jb[2];
                                    double t2[3] = {r_dot_jb * p.rx - jb[0],
                                                  r_dot_jb * p.ry - jb[1],
                                                  r_dot_jb * p.rz - jb[2]};
                                    for (int i = 0; i < 3; ++i) {
                                        double term = t2[i] * p.e0;
                                        sum[0][i].re += amp_re * term;
                                        sum[0][i].im += amp_im * term;
                                    }
                                }
                                // tx = 1: V transmit
                                {
                                    double ein[3] = {p.evx / p.e0, p.evy / p.e0, p.evz / p.e0};
                                    double dot_e_na = ein[0] * nx + ein[1] * ny + ein[2] * nz;
                                    double e1[3] = {2.0 * dot_e_na * nx - ein[0],
                                                  2.0 * dot_e_na * ny - ein[1],
                                                  2.0 * dot_e_na * nz - ein[2]};
                                    double hb[3] = {d1[1] * e1[2] - d1[2] * e1[1],
                                                  d1[2] * e1[0] - d1[0] * e1[2],
                                                  d1[0] * e1[1] - d1[1] * e1[0]};
                                    double nb_r[3] = {nb_ptr[0], nb_ptr[1], nb_ptr[2]};
                                    double jb[3] = {2.0 * (nb_r[1] * hb[2] - nb_r[2] * hb[1]),
                                                  2.0 * (nb_r[2] * hb[0] - nb_r[0] * hb[2]),
                                                  2.0 * (nb_r[0] * hb[1] - nb_r[1] * hb[0])};
                                    double r_dot_jb = p.rx * jb[0] + p.ry * jb[1] + p.rz * jb[2];
                                    double t2[3] = {r_dot_jb * p.rx - jb[0],
                                                  r_dot_jb * p.ry - jb[1],
                                                  r_dot_jb * p.rz - jb[2]};
                                    for (int i = 0; i < 3; ++i) {
                                        double term = t2[i] * p.e0;
                                        sum[1][i].re += amp_re * term;
                                        sum[1][i].im += amp_im * term;
                                    }
                                }
                            }
                        } else if (bvh_data.max_bounces >= 3) {
                            uint32_t ic = 0;
                            double hit_t2 = 0.0;
                            if (gpu_ray_bvh(bvh_data.nodes, bvh_data.order, bvh_data.tri_vertices,
                                            cb_w, d2, bvh_data.t_min, ib, ic, hit_t2) &&
                                ic != ib) {
                                const double* nc_ptr = (mesh_normals != nullptr) ? &mesh_normals[3 * ic] : &nx;
                                double dot_nc_d2 = nc_ptr[0] * d2[0] + nc_ptr[1] * d2[1] + nc_ptr[2] * d2[2];
                                if (dot_nc_d2 < 0.0) {
                                    const double* cc_ptr = (mesh_centroids != nullptr) ? &mesh_centroids[3 * ic] : &cx;
                                    double cc_w[3] = {cc_ptr[0] + bvh_data.phase_ref[0],
                                                      cc_ptr[1] + bvh_data.phase_ref[1],
                                                      cc_ptr[2] + bvh_data.phase_ref[2]};
                                    double area_c = (mesh_areas != nullptr) ? mesh_areas[ic] : area_t;

                                    double d3[3] = {d2[0] - 2.0 * dot_nc_d2 * nc_ptr[0],
                                                    d2[1] - 2.0 * dot_nc_d2 * nc_ptr[1],
                                                    d2[2] - 2.0 * dot_nc_d2 * nc_ptr[2]};

                                    double dot_d3_r = d3[0] * p.rx + d3[1] * p.ry + d3[2] * p.rz;

                                    if (dot_d3_r > 1.0 - 1e-9) {
                                        double r_vec[3] = {p.rx, p.ry, p.rz};
                                        bool exit_occ3 = gpu_occluded_bvh(bvh_data.nodes, bvh_data.order, bvh_data.tri_vertices,
                                                                         cc_w, r_vec, bvh_data.t_min, bvh_data.scene_limit, ic);
                                        if (!exit_occ3) {
                                            double s_dot_ca_ref = p.kx * cx + p.ky * cy + p.kz * cz;
                                            double d1_dot_cb_ca = d1[0] * (cb_ptr[0] - cx) +
                                                                  d1[1] * (cb_ptr[1] - cy) +
                                                                  d1[2] * (cb_ptr[2] - cz);
                                            double d2_dot_cc_cb = d2[0] * (cc_ptr[0] - cb_ptr[0]) +
                                                                  d2[1] * (cc_ptr[1] - cb_ptr[1]) +
                                                                  d2[2] * (cc_ptr[2] - cb_ptr[2]);
                                            double r_dot_cc_ref = p.rx * cc_ptr[0] + p.ry * cc_ptr[1] + p.rz * cc_ptr[2];

                                            double phi_3b = -p.k * (s_dot_ca_ref + d1_dot_cb_ca + d2_dot_cc_cb) +
                                                             p.k * r_dot_cc_ref;
                                            double s_phi3, c_phi3;
                                            fsincos(phi_3b, &s_phi3, &c_phi3);

                                            double amp_val3 = p.k / (4.0 * 3.14159265358979323846) * area_c;
                                            double amp_re3 = -amp_val3 * s_phi3;
                                            double amp_im3 =  amp_val3 * c_phi3;

                                            // tx = 0: H transmit
                                            {
                                                double ein[3] = {p.ehx / p.e0, p.ehy / p.e0, p.ehz / p.e0};
                                                double dot_e_na = ein[0] * nx + ein[1] * ny + ein[2] * nz;
                                                double e1[3] = {2.0 * dot_e_na * nx - ein[0],
                                                              2.0 * dot_e_na * ny - ein[1],
                                                              2.0 * dot_e_na * nz - ein[2]};
                                                double nb_r[3] = {nb_ptr[0], nb_ptr[1], nb_ptr[2]};
                                                double dot_e1_nb = e1[0] * nb_r[0] + e1[1] * nb_r[1] + e1[2] * nb_r[2];
                                                double e2[3] = {2.0 * dot_e1_nb * nb_r[0] - e1[0],
                                                              2.0 * dot_e1_nb * nb_r[1] - e1[1],
                                                              2.0 * dot_e1_nb * nb_r[2] - e1[2]};
                                                double hc[3] = {d2[1] * e2[2] - d2[2] * e2[1],
                                                              d2[2] * e2[0] - d2[0] * e2[2],
                                                              d2[0] * e2[1] - d2[1] * e2[0]};
                                                double nc_r[3] = {nc_ptr[0], nc_ptr[1], nc_ptr[2]};
                                                double jc[3] = {2.0 * (nc_r[1] * hc[2] - nc_r[2] * hc[1]),
                                                              2.0 * (nc_r[2] * hc[0] - nc_r[0] * hc[2]),
                                                              2.0 * (nc_r[0] * hc[1] - nc_r[1] * hc[0])};
                                                double r_dot_jc = p.rx * jc[0] + p.ry * jc[1] + p.rz * jc[2];
                                                double t2[3] = {r_dot_jc * p.rx - jc[0],
                                                              r_dot_jc * p.ry - jc[1],
                                                              r_dot_jc * p.rz - jc[2]};
                                                for (int i = 0; i < 3; ++i) {
                                                    double term = t2[i] * p.e0;
                                                    sum[0][i].re += amp_re3 * term;
                                                    sum[0][i].im += amp_im3 * term;
                                                }
                                            }
                                            // tx = 1: V transmit
                                            {
                                                double ein[3] = {p.evx / p.e0, p.evy / p.e0, p.evz / p.e0};
                                                double dot_e_na = ein[0] * nx + ein[1] * ny + ein[2] * nz;
                                                double e1[3] = {2.0 * dot_e_na * nx - ein[0],
                                                              2.0 * dot_e_na * ny - ein[1],
                                                              2.0 * dot_e_na * nz - ein[2]};
                                                double nb_r[3] = {nb_ptr[0], nb_ptr[1], nb_ptr[2]};
                                                double dot_e1_nb = e1[0] * nb_r[0] + e1[1] * nb_r[1] + e1[2] * nb_r[2];
                                                double e2[3] = {2.0 * dot_e1_nb * nb_r[0] - e1[0],
                                                              2.0 * dot_e1_nb * nb_r[1] - e1[1],
                                                              2.0 * dot_e1_nb * nb_r[2] - e1[2]};
                                                double hc[3] = {d2[1] * e2[2] - d2[2] * e2[1],
                                                              d2[2] * e2[0] - d2[0] * e2[2],
                                                              d2[0] * e2[1] - d2[1] * e2[0]};
                                                double nc_r[3] = {nc_ptr[0], nc_ptr[1], nc_ptr[2]};
                                                double jc[3] = {2.0 * (nc_r[1] * hc[2] - nc_r[2] * hc[1]),
                                                              2.0 * (nc_r[2] * hc[0] - nc_r[0] * hc[2]),
                                                              2.0 * (nc_r[0] * hc[1] - nc_r[1] * hc[0])};
                                                double r_dot_jc = p.rx * jc[0] + p.ry * jc[1] + p.rz * jc[2];
                                                double t2[3] = {r_dot_jc * p.rx - jc[0],
                                                              r_dot_jc * p.ry - jc[1],
                                                              r_dot_jc * p.rz - jc[2]};
                                                for (int i = 0; i < 3; ++i) {
                                                    double term = t2[i] * p.e0;
                                                    sum[1][i].re += amp_re3 * term;
                                                    sum[1][i].im += amp_im3 * term;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    const int lane = static_cast<int>(threadIdx.x);
    for (int tx = 0; tx < 2; ++tx)
        for (int i = 0; i < 3; ++i) {
            sh[tx][i][2 * lane] = sum[tx][i].re;
            sh[tx][i][2 * lane + 1] = sum[tx][i].im;
        }
    lit_sh[lane] = lit;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            for (int tx = 0; tx < 2; ++tx)
                for (int i = 0; i < 3; ++i) {
                    sh[tx][i][2 * lane] += sh[tx][i][2 * (lane + stride)];
                    sh[tx][i][2 * lane + 1] += sh[tx][i][2 * (lane + stride) + 1];
                }
            lit_sh[lane] += lit_sh[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        for (int tx = 0; tx < 2; ++tx)
            for (int i = 0; i < 3; ++i) {
                out[6 * u + 3 * tx + i].re = sh[tx][i][0];
                out[6 * u + 3 * tx + i].im = sh[tx][i][1];
            }
        lit_out[u] = lit_sh[0];
    }
}

} // namespace

bool cuda_available() noexcept {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
        return count > 0;
    } catch (...) {
        return false;
    }
}

CudaDeviceInfo cuda_device_info(int device_id) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (device_id < 0 || device_id >= count)
        throw CudaError("CUDA device " + std::to_string(device_id) + " not found (" +
                        std::to_string(count) + " present)");
    CUDA_CHECK(cudaSetDevice(device_id));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    CudaDeviceInfo info;
    info.name = prop.name;
    info.total_bytes = total_bytes;
    info.free_bytes = free_bytes;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    return info;
}

uint64_t cuda_footprint_bytes(const nlohmann::json& resolved, size_t triangles) {
    const std::string precision = resolved["solver"].value("precision", "float64");
    const bool float32_geom = (precision == "float32" || precision == "mixed");
    return triangles * (3 + 3 + 1) * (float32_geom ? 4 : 8);
}

size_t cuda_select_batch_units(const nlohmann::json& resolved, size_t nunits, size_t triangles,
                               int device_id, size_t override_units) {
    if (nunits == 0) return 0;
    if (override_units > 0) return std::min(override_units, nunits);
    const CudaDeviceInfo info = cuda_device_info(device_id);
    const uint64_t footprint = cuda_footprint_bytes(resolved, triangles);
    if (footprint > info.free_bytes * 4 / 5)
        throw CudaError("device memory insufficient for mesh (" +
                        std::to_string(footprint) + " needed)");
    constexpr uint64_t kPerUnitBytes = 4096; // params + outputs + slack
    const uint64_t budget =
        info.free_bytes / 2 > footprint ? info.free_bytes / 2 - footprint : kPerUnitBytes;
    return std::max<size_t>(1, std::min(nunits, static_cast<size_t>(budget / kPerUnitBytes)));
}

std::string active_device_name(const nlohmann::json& resolved) {
    if (resolved["execution"].value("accelerator", "cpu") != "cuda") return "";
    const int id = resolved["execution"].value("cuda_device_id", 0);
    return cuda_device_info(id).name;
}

namespace {

template <typename Real>
PoResult solve_typed_cuda(const NormalizedMesh& mesh, const SamplePlan& plan,
                          const nlohmann::json& resolved, int device_id,
                          const std::vector<std::pair<uint32_t, uint32_t>>& units,
                          const FringeOptions& fringe,
                          const GoOptions& go,
                          const MaterialOptions& material) {
    (void)fringe;
    PoResult out;
    const auto& medium = resolved["frequency"]["medium"];
    const double eps_r = medium["epsilon_r"].get<double>();
    const double mu_r = medium["mu_r"].get<double>();
    if (medium["sigma"].get<double>() != 0.0)
        out.warnings.push_back("medium conductivity is ignored in v1 PO");
    const double eta = kEta0 * std::sqrt(mu_r / eps_r);
    const double e0 = resolved["physics"]["incident_amplitude"].get<double>();
    const size_t ndir = plan.directions.size();
    const size_t npol = plan.polarizations.size();
    out.samples.resize(units.size() * npol);
    if (units.empty()) return out;

    CUDA_CHECK(cudaSetDevice(device_id));
    geom::Vec3d phase_ref{0.0, 0.0, 0.0};
    if (resolved.contains("physics") && resolved["physics"].contains("phase_reference")) {
        const auto& pr = resolved["physics"]["phase_reference"];
        if (pr.is_array() && pr.size() == 3) {
            phase_ref = {pr[0].get<double>(), pr[1].get<double>(), pr[2].get<double>()};
        }
    }
    bool use_symmetry = false;
    std::string sym_plane = "xz";
    double sym_tol = 1e-4;
    if (resolved.contains("mesh") && resolved["mesh"].contains("symmetry")) {
        const auto& sym = resolved["mesh"]["symmetry"];
        use_symmetry = sym.value("enabled", false);
        sym_plane = sym.value("plane", "xz");
        sym_tol = sym.value("tolerance_m", 1e-4);
    }
    const SymmetryOptions sym_opts{use_symmetry, sym_plane, sym_tol};
    const HalfMeshResult half_result = use_symmetry ? extract_half_mesh(mesh, sym_opts) : HalfMeshResult{mesh, {}, {}};
    const NormalizedMesh& eff_mesh = use_symmetry ? half_result.mesh : mesh;

    std::vector<Real> h_cent, h_norm, h_area;
    h_cent.reserve(eff_mesh.triangles.size() * (use_symmetry ? 6 : 3));
    h_norm.reserve(eff_mesh.triangles.size() * (use_symmetry ? 6 : 3));
    h_area.reserve(eff_mesh.triangles.size() * (use_symmetry ? 2 : 1));

    std::vector<uint32_t> h_mat_idx;
    bool has_materials = (material.enabled && (material.wall != WallType::Pec || !material.facet_material_indices.empty()));
    if (has_materials && !material.facet_material_indices.empty()) {
        h_mat_idx.reserve(eff_mesh.triangles.size() * (use_symmetry ? 2 : 1));
    }

    for (size_t t = 0; t < eff_mesh.triangles.size(); ++t) {
        const auto& tri = eff_mesh.triangles[t];
        const geom::Vec3d c = (eff_mesh.vertices[tri[0]] + eff_mesh.vertices[tri[1]] + eff_mesh.vertices[tri[2]]) * (1.0 / 3.0);
        const geom::Vec3d n = eff_mesh.normals[t];
        const double a = eff_mesh.areas[t];

        h_cent.push_back(static_cast<Real>(c.x - phase_ref.x));
        h_cent.push_back(static_cast<Real>(c.y - phase_ref.y));
        h_cent.push_back(static_cast<Real>(c.z - phase_ref.z));
        h_norm.push_back(static_cast<Real>(n.x));
        h_norm.push_back(static_cast<Real>(n.y));
        h_norm.push_back(static_cast<Real>(n.z));
        h_area.push_back(static_cast<Real>(a));

        uint32_t orig_idx = use_symmetry ? half_result.original_indices[t] : static_cast<uint32_t>(t);
        if (has_materials && !material.facet_material_indices.empty()) {
            uint32_t midx = (orig_idx < material.facet_material_indices.size()) ? material.facet_material_indices[orig_idx] : 0;
            h_mat_idx.push_back(midx);
        }

        if (use_symmetry) {
            const double coord = get_symmetry_coord(c, sym_plane);
            if (std::abs(coord) > sym_tol) {
                const geom::Vec3d c_refl = reflect_point(c, sym_plane);
                const geom::Vec3d n_refl = reflect_normal(n, sym_plane);
                h_cent.push_back(static_cast<Real>(c_refl.x - phase_ref.x));
                h_cent.push_back(static_cast<Real>(c_refl.y - phase_ref.y));
                h_cent.push_back(static_cast<Real>(c_refl.z - phase_ref.z));
                h_norm.push_back(static_cast<Real>(n_refl.x));
                h_norm.push_back(static_cast<Real>(n_refl.y));
                h_norm.push_back(static_cast<Real>(n_refl.z));
                h_area.push_back(static_cast<Real>(a));

                if (has_materials && !material.facet_material_indices.empty()) {
                    uint32_t midx = (orig_idx < material.facet_material_indices.size()) ? material.facet_material_indices[orig_idx] : 0;
                    h_mat_idx.push_back(midx);
                }
            }
        }
    }

    nvtxRangePushA("po_gpu_facet_upload");
    const size_t ntris = h_area.size();
    DeviceBuffer<Real> d_cent(h_cent.size());
    DeviceBuffer<Real> d_norm(h_norm.size());
    DeviceBuffer<Real> d_area(h_area.size());
    CUDA_CHECK(cudaMemcpy(d_cent.get(), h_cent.data(), h_cent.size() * sizeof(Real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norm.get(), h_norm.data(), h_norm.size() * sizeof(Real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_area.get(), h_area.data(), h_area.size() * sizeof(Real),
                          cudaMemcpyHostToDevice));

    DeviceBuffer<uint32_t> d_mat_idx(h_mat_idx.size());
    if (!h_mat_idx.empty()) {
        CUDA_CHECK(cudaMemcpy(d_mat_idx.get(), h_mat_idx.data(), h_mat_idx.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    }
    nvtxRangePop();

    // BVH construction & upload for GO (Shadowing / Multi-Bounce)
    GpuBvhData bvh_data{};
    std::unique_ptr<DeviceBuffer<GpuBvhNode>> d_bvh_nodes;
    std::unique_ptr<DeviceBuffer<uint32_t>> d_bvh_order;
    std::unique_ptr<DeviceBuffer<double>> d_tri_verts;
    std::unique_ptr<DeviceBuffer<double>> d_mesh_cent;
    std::unique_ptr<DeviceBuffer<double>> d_mesh_norm;
    std::unique_ptr<DeviceBuffer<double>> d_mesh_area;

    if (go.shadowing || go.max_bounces >= 2) {
        nvtxRangePushA("po_gpu_bvh_build_upload");
        const Bvh bvh = build_bvh(mesh);
        std::vector<GpuBvhNode> h_bvh_nodes;
        h_bvh_nodes.reserve(bvh.nodes.size());
        for (const auto& n : bvh.nodes) {
            GpuBvhNode gn{};
            gn.bmin[0] = n.bmin.x; gn.bmin[1] = n.bmin.y; gn.bmin[2] = n.bmin.z;
            gn.bmax[0] = n.bmax.x; gn.bmax[1] = n.bmax.y; gn.bmax[2] = n.bmax.z;
            gn.left = n.left;
            gn.right = n.right;
            gn.start = n.start;
            gn.count = n.count;
            h_bvh_nodes.push_back(gn);
        }

        std::vector<double> h_tri_verts;
        h_tri_verts.reserve(mesh.triangles.size() * 9);
        for (const auto& tri : mesh.triangles) {
            const auto& v0 = mesh.vertices[tri[0]];
            const auto& v1 = mesh.vertices[tri[1]];
            const auto& v2 = mesh.vertices[tri[2]];
            h_tri_verts.push_back(v0.x); h_tri_verts.push_back(v0.y); h_tri_verts.push_back(v0.z);
            h_tri_verts.push_back(v1.x); h_tri_verts.push_back(v1.y); h_tri_verts.push_back(v1.z);
            h_tri_verts.push_back(v2.x); h_tri_verts.push_back(v2.y); h_tri_verts.push_back(v2.z);
        }

        std::vector<double> h_mesh_cent, h_mesh_norm, h_mesh_area;
        h_mesh_cent.reserve(mesh.triangles.size() * 3);
        h_mesh_norm.reserve(mesh.triangles.size() * 3);
        h_mesh_area.reserve(mesh.triangles.size());
        for (size_t t = 0; t < mesh.triangles.size(); ++t) {
            const auto& tri = mesh.triangles[t];
            geom::Vec3d c = (mesh.vertices[tri[0]] + mesh.vertices[tri[1]] + mesh.vertices[tri[2]]) * (1.0 / 3.0);
            h_mesh_cent.push_back(c.x - phase_ref.x);
            h_mesh_cent.push_back(c.y - phase_ref.y);
            h_mesh_cent.push_back(c.z - phase_ref.z);
            h_mesh_norm.push_back(mesh.normals[t].x);
            h_mesh_norm.push_back(mesh.normals[t].y);
            h_mesh_norm.push_back(mesh.normals[t].z);
            h_mesh_area.push_back(mesh.areas[t]);
        }

        d_bvh_nodes = std::make_unique<DeviceBuffer<GpuBvhNode>>(h_bvh_nodes.size());
        d_bvh_order = std::make_unique<DeviceBuffer<uint32_t>>(bvh.order.size());
        d_tri_verts = std::make_unique<DeviceBuffer<double>>(h_tri_verts.size());
        d_mesh_cent = std::make_unique<DeviceBuffer<double>>(h_mesh_cent.size());
        d_mesh_norm = std::make_unique<DeviceBuffer<double>>(h_mesh_norm.size());
        d_mesh_area = std::make_unique<DeviceBuffer<double>>(h_mesh_area.size());

        CUDA_CHECK(cudaMemcpy(d_bvh_nodes->get(), h_bvh_nodes.data(), h_bvh_nodes.size() * sizeof(GpuBvhNode), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bvh_order->get(), bvh.order.data(), bvh.order.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tri_verts->get(), h_tri_verts.data(), h_tri_verts.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mesh_cent->get(), h_mesh_cent.data(), h_mesh_cent.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mesh_norm->get(), h_mesh_norm.data(), h_mesh_norm.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mesh_area->get(), h_mesh_area.data(), h_mesh_area.size() * sizeof(double), cudaMemcpyHostToDevice));

        bvh_data.nodes = d_bvh_nodes->get();
        bvh_data.order = d_bvh_order->get();
        bvh_data.tri_vertices = d_tri_verts->get();
        bvh_data.num_nodes = static_cast<int>(h_bvh_nodes.size());
        const geom::Vec3d diag = mesh.report.bbox_max - mesh.report.bbox_min;
        bvh_data.scene_diag = diag.length();
        bvh_data.t_min = 1e-9 * bvh_data.scene_diag;
        bvh_data.scene_limit = 2.0 * bvh_data.scene_diag;
        bvh_data.phase_ref[0] = phase_ref.x;
        bvh_data.phase_ref[1] = phase_ref.y;
        bvh_data.phase_ref[2] = phase_ref.z;
        bvh_data.shadowing = go.shadowing;
        bvh_data.max_bounces = go.max_bounces;
        nvtxRangePop();
    } else {
        bvh_data.nodes = nullptr;
        bvh_data.shadowing = false;
        bvh_data.max_bounces = 1;
    }

    const size_t override_units =
        static_cast<size_t>(resolved["execution"].value("cuda_batch_units", 0u));
    const size_t batch = cuda_select_batch_units(resolved, units.size(), ntris, device_id,
                                                  override_units);
    if (batch > std::numeric_limits<size_t>::max() / 6)
        throw CudaError("CUDA output batch size overflow");
    DeviceBuffer<UnitParams<Real>> d_units(batch);
    DeviceBuffer<Cplx<Real>> d_out(6 * batch);
    DeviceBuffer<int> d_lit(batch);
    std::vector<UnitParams<Real>> h_units(batch);
    std::vector<Cplx<Real>> h_out(6 * batch);
    std::vector<int> h_lit(batch);

    for (size_t b = 0; b < units.size(); b += batch) {
        nvtxRangePushA("po_gpu_batch");
        const size_t count = std::min(batch, units.size() - b);
        for (size_t i = 0; i < count; ++i) {
            const uint32_t fi = units[b + i].first;
            const uint32_t di = units[b + i].second;
            const double freq = plan.frequencies_hz[fi];
            const Direction& dir = plan.directions[di];
            const Real k = static_cast<Real>(2.0 * 3.141592653589793 * freq *
                                             std::sqrt(eps_r * mu_r) / kSpeedOfLight);
            UnitParams<Real>& p = h_units[i];
            p.kx = static_cast<Real>(dir.k_hat.x);
            p.ky = static_cast<Real>(dir.k_hat.y);
            p.kz = static_cast<Real>(dir.k_hat.z);
            if (dir.is_bistatic) {
                p.rx = static_cast<Real>(dir.r_obs_hat.x);
                p.ry = static_cast<Real>(dir.r_obs_hat.y);
                p.rz = static_cast<Real>(dir.r_obs_hat.z);
            } else {
                p.rx = -p.kx;
                p.ry = -p.ky;
                p.rz = -p.kz;
            }
            const auto basis = polarization_basis(dir.k_hat);
            p.ehx = static_cast<Real>(basis.h.x * e0);
            p.ehy = static_cast<Real>(basis.h.y * e0);
            p.ehz = static_cast<Real>(basis.h.z * e0);
            p.evx = static_cast<Real>(basis.v.x * e0);
            p.evy = static_cast<Real>(basis.v.y * e0);
            p.evz = static_cast<Real>(basis.v.z * e0);
            p.k = k;
            p.eta = static_cast<Real>(eta);
            p.e0 = static_cast<Real>(e0);
            p.qx = p.k * (p.rx - p.kx);
            p.qy = p.k * (p.ry - p.ky);
            p.qz = p.k * (p.rz - p.kz);
            p.h0_h[0] = (p.ky * p.ehz - p.kz * p.ehy) / p.eta;
            p.h0_h[1] = (p.kz * p.ehx - p.kx * p.ehz) / p.eta;
            p.h0_h[2] = (p.kx * p.ehy - p.ky * p.ehx) / p.eta;
            p.h0_v[0] = (p.ky * p.evz - p.kz * p.evy) / p.eta;
            p.h0_v[1] = (p.kz * p.evx - p.kx * p.evz) / p.eta;
            p.h0_v[2] = (p.kx * p.evy - p.ky * p.evx) / p.eta;
            p.k_eta_4pi = p.k * p.eta / static_cast<Real>(4.0 * 3.14159265358979323846);

            p.freq_hz = static_cast<Real>(freq);
            p.lambda_m = static_cast<Real>(kSpeedOfLight / freq);
            p.has_materials = has_materials ? 1 : 0;
            p.num_materials = 0;

            if (has_materials) {
                if (!material.models.empty()) {
                    p.num_materials = static_cast<int>(std::min(size_t(8), material.models.size()));
                    for (int m = 0; m < p.num_materials; ++m) {
                        const auto& mod = material.models[m];
                        p.materials[m].wall_type = static_cast<uint8_t>(mod.type);
                        if (mod.type == WallType::Impedance) {
                            p.materials[m].eta_s = {mod.surface_impedance.real(), mod.surface_impedance.imag()};
                        } else if (mod.type == WallType::Dielectric) {
                            ComplexMedium med = evaluate_material(mod, freq);
                            p.materials[m].eps_r = {med.eps_r.real(), med.eps_r.imag()};
                            p.materials[m].mu_r = {med.mu_r.real(), med.mu_r.imag()};
                        } else if (mod.type == WallType::Coated) {
                            p.materials[m].num_layers = static_cast<uint8_t>(std::min(size_t(8), mod.layers.size()));
                            for (uint8_t l = 0; l < p.materials[m].num_layers; ++l) {
                                p.materials[m].layers[l].thickness_m = mod.layers[l].thickness_m;
                                p.materials[m].layers[l].eps_r = {mod.layers[l].medium.eps_r.real(), mod.layers[l].medium.eps_r.imag()};
                                p.materials[m].layers[l].mu_r = {mod.layers[l].medium.mu_r.real(), mod.layers[l].medium.mu_r.imag()};
                            }
                        }
                    }
                } else if (material.wall != WallType::Pec) {
                    p.num_materials = 1;
                    p.materials[0].wall_type = static_cast<uint8_t>(material.wall);
                    if (material.wall == WallType::Impedance) {
                        p.materials[0].eta_s = {material.model.surface_impedance.real(), material.model.surface_impedance.imag()};
                    } else if (material.wall == WallType::Dielectric) {
                        ComplexMedium med = evaluate_material(material.model, freq);
                        p.materials[0].eps_r = {med.eps_r.real(), med.eps_r.imag()};
                        p.materials[0].mu_r = {med.mu_r.real(), med.mu_r.imag()};
                    } else if (material.wall == WallType::Coated) {
                        p.materials[0].num_layers = static_cast<uint8_t>(std::min(size_t(8), material.model.layers.size()));
                        for (uint8_t l = 0; l < p.materials[0].num_layers; ++l) {
                            p.materials[0].layers[l].thickness_m = material.model.layers[l].thickness_m;
                            p.materials[0].layers[l].eps_r = {material.model.layers[l].medium.eps_r.real(), material.model.layers[l].medium.eps_r.imag()};
                            p.materials[0].layers[l].mu_r = {material.model.layers[l].medium.mu_r.real(), material.model.layers[l].medium.mu_r.imag()};
                        }
                    }
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(d_units.get(), h_units.data(), count * sizeof(UnitParams<Real>),
                              cudaMemcpyHostToDevice));
        nvtxRangePushA("po_gpu_kernel_launch");
        po_kernel<Real>
            <<<static_cast<unsigned>(count), kBlock>>>(d_cent.get(), d_norm.get(), d_area.get(),
                                                       d_mat_idx.get(),
                                                       d_units.get(), bvh_data,
                                                       d_mesh_cent ? d_mesh_cent->get() : nullptr,
                                                       d_mesh_norm ? d_mesh_norm->get() : nullptr,
                                                       d_mesh_area ? d_mesh_area->get() : nullptr,
                                                       d_out.get(), d_lit.get(),
                                                       static_cast<int>(ntris),
                                                       static_cast<int>(mesh.triangles.size()));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();

        nvtxRangePushA("po_gpu_download_project");
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.get(), 6 * count * sizeof(Cplx<Real>),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_lit.data(), d_lit.get(), count * sizeof(int),
                              cudaMemcpyDeviceToHost));
        // Project on the host with the shared channel math.
        const size_t npol = plan.polarizations.size();
        const size_t ndir = plan.directions.size();
        for (size_t i = 0; i < count; ++i) {
            const uint32_t fi = units[b + i].first;
            const uint32_t di = units[b + i].second;
            const Direction& dir = plan.directions[di];
            const auto basis = polarization_basis(dir.k_hat);
            const auto rx_basis = dir.is_bistatic ? polarization_basis(dir.r_obs_hat) : basis;
            const double eh[3] = {rx_basis.h.x, rx_basis.h.y, rx_basis.h.z};
            const double ev[3] = {rx_basis.v.x, rx_basis.v.y, rx_basis.v.z};
            std::complex<double> ftx[2][3];
            for (int tx = 0; tx < 2; ++tx)
                for (int c = 0; c < 3; ++c)
                    ftx[tx][c] = {static_cast<double>(h_out[6 * i + 3 * tx + c].re),
                                  static_cast<double>(h_out[6 * i + 3 * tx + c].im)};
            const auto channels = project_unit_channels(ftx, eh, ev, e0, plan.polarizations);
            for (size_t pi = 0; pi < npol; ++pi) {
                PoSampleResult& sample = out.samples[(b + i) * npol + pi];
                sample.sample_id = (static_cast<uint64_t>(fi) * ndir + di) * npol + pi;
                sample.frequency_id = fi;
                sample.direction_id = di;
                sample.pol_id = static_cast<uint32_t>(pi);
                sample.scattering = channels[pi].first;
                sample.rcs_sqm = channels[pi].second;
                sample.lit_facets = static_cast<size_t>(h_lit[i]);
            }
        }
        nvtxRangePop();
        nvtxRangePop(); // po_gpu_batch
    }
    return out;
}

PoResult solve_mixed_cuda(const NormalizedMesh& mesh, const SamplePlan& plan,
                          const nlohmann::json& resolved, int device_id,
                          const std::vector<std::pair<uint32_t, uint32_t>>& units,
                          const FringeOptions& fringe,
                          const GoOptions& go,
                          const MaterialOptions& material) {
    (void)fringe;
    PoResult out;
    const auto& medium = resolved["frequency"]["medium"];
    const double eps_r = medium["epsilon_r"].get<double>();
    const double mu_r = medium["mu_r"].get<double>();
    if (medium["sigma"].get<double>() != 0.0)
        out.warnings.push_back("medium conductivity is ignored in v1 PO");
    const double eta = kEta0 * std::sqrt(mu_r / eps_r);
    const double e0 = resolved["physics"]["incident_amplitude"].get<double>();
    const size_t ndir = plan.directions.size();
    const size_t npol = plan.polarizations.size();
    out.samples.resize(units.size() * npol);
    if (units.empty()) return out;

    CUDA_CHECK(cudaSetDevice(device_id));
    geom::Vec3d phase_ref{0.0, 0.0, 0.0};
    if (resolved.contains("physics") && resolved["physics"].contains("phase_reference")) {
        const auto& pr = resolved["physics"]["phase_reference"];
        if (pr.is_array() && pr.size() == 3) {
            phase_ref = {pr[0].get<double>(), pr[1].get<double>(), pr[2].get<double>()};
        }
    }
    bool use_symmetry = false;
    std::string sym_plane = "xz";
    double sym_tol = 1e-4;
    if (resolved.contains("mesh") && resolved["mesh"].contains("symmetry")) {
        const auto& sym = resolved["mesh"]["symmetry"];
        use_symmetry = sym.value("enabled", false);
        sym_plane = sym.value("plane", "xz");
        sym_tol = sym.value("tolerance_m", 1e-4);
    }
    const SymmetryOptions sym_opts{use_symmetry, sym_plane, sym_tol};
    const HalfMeshResult half_result = use_symmetry ? extract_half_mesh(mesh, sym_opts) : HalfMeshResult{mesh, {}, {}};
    const NormalizedMesh& eff_mesh = use_symmetry ? half_result.mesh : mesh;

    std::vector<float> h_cent, h_norm, h_area;
    h_cent.reserve(eff_mesh.triangles.size() * (use_symmetry ? 6 : 3));
    h_norm.reserve(eff_mesh.triangles.size() * (use_symmetry ? 6 : 3));
    h_area.reserve(eff_mesh.triangles.size() * (use_symmetry ? 2 : 1));

    std::vector<uint32_t> h_mat_idx;
    bool has_materials = (material.enabled && (material.wall != WallType::Pec || !material.facet_material_indices.empty()));
    if (has_materials && !material.facet_material_indices.empty()) {
        h_mat_idx.reserve(eff_mesh.triangles.size() * (use_symmetry ? 2 : 1));
    }

    for (size_t t = 0; t < eff_mesh.triangles.size(); ++t) {
        const auto& tri = eff_mesh.triangles[t];
        const geom::Vec3d c = (eff_mesh.vertices[tri[0]] + eff_mesh.vertices[tri[1]] + eff_mesh.vertices[tri[2]]) * (1.0 / 3.0);
        const geom::Vec3d n = eff_mesh.normals[t];
        const double a = eff_mesh.areas[t];

        h_cent.push_back(static_cast<float>(c.x - phase_ref.x));
        h_cent.push_back(static_cast<float>(c.y - phase_ref.y));
        h_cent.push_back(static_cast<float>(c.z - phase_ref.z));
        h_norm.push_back(static_cast<float>(n.x));
        h_norm.push_back(static_cast<float>(n.y));
        h_norm.push_back(static_cast<float>(n.z));
        h_area.push_back(static_cast<float>(a));

        uint32_t orig_idx = use_symmetry ? half_result.original_indices[t] : static_cast<uint32_t>(t);
        if (has_materials && !material.facet_material_indices.empty()) {
            uint32_t midx = (orig_idx < material.facet_material_indices.size()) ? material.facet_material_indices[orig_idx] : 0;
            h_mat_idx.push_back(midx);
        }

        if (use_symmetry) {
            const double coord = get_symmetry_coord(c, sym_plane);
            if (std::abs(coord) > sym_tol) {
                const geom::Vec3d c_refl = reflect_point(c, sym_plane);
                const geom::Vec3d n_refl = reflect_normal(n, sym_plane);
                h_cent.push_back(static_cast<float>(c_refl.x - phase_ref.x));
                h_cent.push_back(static_cast<float>(c_refl.y - phase_ref.y));
                h_cent.push_back(static_cast<float>(c_refl.z - phase_ref.z));
                h_norm.push_back(static_cast<float>(n_refl.x));
                h_norm.push_back(static_cast<float>(n_refl.y));
                h_norm.push_back(static_cast<float>(n_refl.z));
                h_area.push_back(static_cast<float>(a));

                if (has_materials && !material.facet_material_indices.empty()) {
                    uint32_t midx = (orig_idx < material.facet_material_indices.size()) ? material.facet_material_indices[orig_idx] : 0;
                    h_mat_idx.push_back(midx);
                }
            }
        }
    }

    nvtxRangePushA("po_gpu_facet_upload");
    const size_t ntris = h_area.size();
    DeviceBuffer<float> d_cent(h_cent.size());
    DeviceBuffer<float> d_norm(h_norm.size());
    DeviceBuffer<float> d_area(h_area.size());
    CUDA_CHECK(cudaMemcpy(d_cent.get(), h_cent.data(), h_cent.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norm.get(), h_norm.data(), h_norm.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_area.get(), h_area.data(), h_area.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    DeviceBuffer<uint32_t> d_mat_idx(h_mat_idx.size());
    if (!h_mat_idx.empty()) {
        CUDA_CHECK(cudaMemcpy(d_mat_idx.get(), h_mat_idx.data(), h_mat_idx.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    }
    nvtxRangePop();

    // BVH construction & upload for GO
    GpuBvhData bvh_data{};
    std::unique_ptr<DeviceBuffer<GpuBvhNode>> d_bvh_nodes;
    std::unique_ptr<DeviceBuffer<uint32_t>> d_bvh_order;
    std::unique_ptr<DeviceBuffer<double>> d_tri_verts;
    std::unique_ptr<DeviceBuffer<double>> d_mesh_cent;
    std::unique_ptr<DeviceBuffer<double>> d_mesh_norm;
    std::unique_ptr<DeviceBuffer<double>> d_mesh_area;

    if (go.shadowing || go.max_bounces >= 2) {
        nvtxRangePushA("po_gpu_bvh_build_upload");
        const Bvh bvh = build_bvh(mesh);
        std::vector<GpuBvhNode> h_bvh_nodes;
        h_bvh_nodes.reserve(bvh.nodes.size());
        for (const auto& n : bvh.nodes) {
            GpuBvhNode gn{};
            gn.bmin[0] = n.bmin.x; gn.bmin[1] = n.bmin.y; gn.bmin[2] = n.bmin.z;
            gn.bmax[0] = n.bmax.x; gn.bmax[1] = n.bmax.y; gn.bmax[2] = n.bmax.z;
            gn.left = n.left;
            gn.right = n.right;
            gn.start = n.start;
            gn.count = n.count;
            h_bvh_nodes.push_back(gn);
        }

        std::vector<double> h_tri_verts;
        h_tri_verts.reserve(mesh.triangles.size() * 9);
        for (const auto& tri : mesh.triangles) {
            const auto& v0 = mesh.vertices[tri[0]];
            const auto& v1 = mesh.vertices[tri[1]];
            const auto& v2 = mesh.vertices[tri[2]];
            h_tri_verts.push_back(v0.x); h_tri_verts.push_back(v0.y); h_tri_verts.push_back(v0.z);
            h_tri_verts.push_back(v1.x); h_tri_verts.push_back(v1.y); h_tri_verts.push_back(v1.z);
            h_tri_verts.push_back(v2.x); h_tri_verts.push_back(v2.y); h_tri_verts.push_back(v2.z);
        }

        std::vector<double> h_mesh_cent, h_mesh_norm, h_mesh_area;
        h_mesh_cent.reserve(mesh.triangles.size() * 3);
        h_mesh_norm.reserve(mesh.triangles.size() * 3);
        h_mesh_area.reserve(mesh.triangles.size());
        for (size_t t = 0; t < mesh.triangles.size(); ++t) {
            const auto& tri = mesh.triangles[t];
            geom::Vec3d c = (mesh.vertices[tri[0]] + mesh.vertices[tri[1]] + mesh.vertices[tri[2]]) * (1.0 / 3.0);
            h_mesh_cent.push_back(c.x - phase_ref.x);
            h_mesh_cent.push_back(c.y - phase_ref.y);
            h_mesh_cent.push_back(c.z - phase_ref.z);
            h_mesh_norm.push_back(mesh.normals[t].x);
            h_mesh_norm.push_back(mesh.normals[t].y);
            h_mesh_norm.push_back(mesh.normals[t].z);
            h_mesh_area.push_back(mesh.areas[t]);
        }

        d_bvh_nodes = std::make_unique<DeviceBuffer<GpuBvhNode>>(h_bvh_nodes.size());
        d_bvh_order = std::make_unique<DeviceBuffer<uint32_t>>(bvh.order.size());
        d_tri_verts = std::make_unique<DeviceBuffer<double>>(h_tri_verts.size());
        d_mesh_cent = std::make_unique<DeviceBuffer<double>>(h_mesh_cent.size());
        d_mesh_norm = std::make_unique<DeviceBuffer<double>>(h_mesh_norm.size());
        d_mesh_area = std::make_unique<DeviceBuffer<double>>(h_mesh_area.size());

        CUDA_CHECK(cudaMemcpy(d_bvh_nodes->get(), h_bvh_nodes.data(), h_bvh_nodes.size() * sizeof(GpuBvhNode), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bvh_order->get(), bvh.order.data(), bvh.order.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tri_verts->get(), h_tri_verts.data(), h_tri_verts.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mesh_cent->get(), h_mesh_cent.data(), h_mesh_cent.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mesh_norm->get(), h_mesh_norm.data(), h_mesh_norm.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mesh_area->get(), h_mesh_area.data(), h_mesh_area.size() * sizeof(double), cudaMemcpyHostToDevice));

        bvh_data.nodes = d_bvh_nodes->get();
        bvh_data.order = d_bvh_order->get();
        bvh_data.tri_vertices = d_tri_verts->get();
        bvh_data.num_nodes = static_cast<int>(h_bvh_nodes.size());
        const geom::Vec3d diag = mesh.report.bbox_max - mesh.report.bbox_min;
        bvh_data.scene_diag = diag.length();
        bvh_data.t_min = 1e-9 * bvh_data.scene_diag;
        bvh_data.scene_limit = 2.0 * bvh_data.scene_diag;
        bvh_data.phase_ref[0] = phase_ref.x;
        bvh_data.phase_ref[1] = phase_ref.y;
        bvh_data.phase_ref[2] = phase_ref.z;
        bvh_data.shadowing = go.shadowing;
        bvh_data.max_bounces = go.max_bounces;
        nvtxRangePop();
    } else {
        bvh_data.nodes = nullptr;
        bvh_data.shadowing = false;
        bvh_data.max_bounces = 1;
    }

    const size_t override_units =
        static_cast<size_t>(resolved["execution"].value("cuda_batch_units", 0u));
    const size_t batch = cuda_select_batch_units(resolved, units.size(), ntris, device_id,
                                                  override_units);
    if (batch > std::numeric_limits<size_t>::max() / 6)
        throw CudaError("CUDA output batch size overflow");
    DeviceBuffer<UnitParams<double>> d_units(batch);
    DeviceBuffer<Cplx<double>> d_out(6 * batch);
    DeviceBuffer<int> d_lit(batch);
    std::vector<UnitParams<double>> h_units(batch);
    std::vector<Cplx<double>> h_out(6 * batch);
    std::vector<int> h_lit(batch);

    for (size_t b = 0; b < units.size(); b += batch) {
        nvtxRangePushA("po_gpu_batch");
        const size_t count = std::min(batch, units.size() - b);
        for (size_t i = 0; i < count; ++i) {
            const uint32_t fi = units[b + i].first;
            const uint32_t di = units[b + i].second;
            const double freq = plan.frequencies_hz[fi];
            const Direction& dir = plan.directions[di];
            const double k = 2.0 * 3.141592653589793 * freq *
                             std::sqrt(eps_r * mu_r) / kSpeedOfLight;
            UnitParams<double>& p = h_units[i];
            p.kx = dir.k_hat.x;
            p.ky = dir.k_hat.y;
            p.kz = dir.k_hat.z;
            if (dir.is_bistatic) {
                p.rx = dir.r_obs_hat.x;
                p.ry = dir.r_obs_hat.y;
                p.rz = dir.r_obs_hat.z;
            } else {
                p.rx = -p.kx;
                p.ry = -p.ky;
                p.rz = -p.kz;
            }
            const auto basis = polarization_basis(dir.k_hat);
            p.ehx = basis.h.x * e0;
            p.ehy = basis.h.y * e0;
            p.ehz = basis.h.z * e0;
            p.evx = basis.v.x * e0;
            p.evy = basis.v.y * e0;
            p.evz = basis.v.z * e0;
            p.k = k;
            p.eta = eta;
            p.e0 = e0;
            p.qx = p.k * (p.rx - p.kx);
            p.qy = p.k * (p.ry - p.ky);
            p.qz = p.k * (p.rz - p.kz);
            p.h0_h[0] = (p.ky * p.ehz - p.kz * p.ehy) / p.eta;
            p.h0_h[1] = (p.kz * p.ehx - p.kx * p.ehz) / p.eta;
            p.h0_h[2] = (p.kx * p.ehy - p.ky * p.ehx) / p.eta;
            p.h0_v[0] = (p.ky * p.evz - p.kz * p.evy) / p.eta;
            p.h0_v[1] = (p.kz * p.evx - p.kx * p.evz) / p.eta;
            p.h0_v[2] = (p.kx * p.evy - p.ky * p.evx) / p.eta;
            p.k_eta_4pi = p.k * p.eta / (4.0 * 3.14159265358979323846);

            p.freq_hz = freq;
            p.lambda_m = kSpeedOfLight / freq;
            p.has_materials = has_materials ? 1 : 0;
            p.num_materials = 0;

            if (has_materials) {
                if (!material.models.empty()) {
                    p.num_materials = static_cast<int>(std::min(size_t(8), material.models.size()));
                    for (int m = 0; m < p.num_materials; ++m) {
                        const auto& mod = material.models[m];
                        p.materials[m].wall_type = static_cast<uint8_t>(mod.type);
                        if (mod.type == WallType::Impedance) {
                            p.materials[m].eta_s = {mod.surface_impedance.real(), mod.surface_impedance.imag()};
                        } else if (mod.type == WallType::Dielectric) {
                            ComplexMedium med = evaluate_material(mod, freq);
                            p.materials[m].eps_r = {med.eps_r.real(), med.eps_r.imag()};
                            p.materials[m].mu_r = {med.mu_r.real(), med.mu_r.imag()};
                        } else if (mod.type == WallType::Coated) {
                            p.materials[m].num_layers = static_cast<uint8_t>(std::min(size_t(8), mod.layers.size()));
                            for (uint8_t l = 0; l < p.materials[m].num_layers; ++l) {
                                p.materials[m].layers[l].thickness_m = mod.layers[l].thickness_m;
                                p.materials[m].layers[l].eps_r = {mod.layers[l].medium.eps_r.real(), mod.layers[l].medium.eps_r.imag()};
                                p.materials[m].layers[l].mu_r = {mod.layers[l].medium.mu_r.real(), mod.layers[l].medium.mu_r.imag()};
                            }
                        }
                    }
                } else if (material.wall != WallType::Pec) {
                    p.num_materials = 1;
                    p.materials[0].wall_type = static_cast<uint8_t>(material.wall);
                    if (material.wall == WallType::Impedance) {
                        p.materials[0].eta_s = {material.model.surface_impedance.real(), material.model.surface_impedance.imag()};
                    } else if (material.wall == WallType::Dielectric) {
                        ComplexMedium med = evaluate_material(material.model, freq);
                        p.materials[0].eps_r = {med.eps_r.real(), med.eps_r.imag()};
                        p.materials[0].mu_r = {med.mu_r.real(), med.mu_r.imag()};
                    } else if (material.wall == WallType::Coated) {
                        p.materials[0].num_layers = static_cast<uint8_t>(std::min(size_t(8), material.model.layers.size()));
                        for (uint8_t l = 0; l < p.materials[0].num_layers; ++l) {
                            p.materials[0].layers[l].thickness_m = material.model.layers[l].thickness_m;
                            p.materials[0].layers[l].eps_r = {material.model.layers[l].medium.eps_r.real(), material.model.layers[l].medium.eps_r.imag()};
                            p.materials[0].layers[l].mu_r = {material.model.layers[l].medium.mu_r.real(), material.model.layers[l].medium.mu_r.imag()};
                        }
                    }
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(d_units.get(), h_units.data(), count * sizeof(UnitParams<double>),
                              cudaMemcpyHostToDevice));
        nvtxRangePushA("po_gpu_kernel_launch");
        po_mixed_kernel
            <<<static_cast<unsigned>(count), kBlock>>>(d_cent.get(), d_norm.get(), d_area.get(),
                                                       d_mat_idx.get(),
                                                       d_units.get(), bvh_data,
                                                       d_mesh_cent ? d_mesh_cent->get() : nullptr,
                                                       d_mesh_norm ? d_mesh_norm->get() : nullptr,
                                                       d_mesh_area ? d_mesh_area->get() : nullptr,
                                                       d_out.get(), d_lit.get(),
                                                       static_cast<int>(ntris),
                                                       static_cast<int>(mesh.triangles.size()));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();

        nvtxRangePushA("po_gpu_download_project");
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.get(), 6 * count * sizeof(Cplx<double>),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_lit.data(), d_lit.get(), count * sizeof(int),
                              cudaMemcpyDeviceToHost));
        const size_t npol = plan.polarizations.size();
        const size_t ndir = plan.directions.size();
        for (size_t i = 0; i < count; ++i) {
            const uint32_t fi = units[b + i].first;
            const uint32_t di = units[b + i].second;
            const Direction& dir = plan.directions[di];
            const auto basis = polarization_basis(dir.k_hat);
            const auto rx_basis = dir.is_bistatic ? polarization_basis(dir.r_obs_hat) : basis;
            const double eh[3] = {rx_basis.h.x, rx_basis.h.y, rx_basis.h.z};
            const double ev[3] = {rx_basis.v.x, rx_basis.v.y, rx_basis.v.z};
            std::complex<double> ftx[2][3];
            for (int tx = 0; tx < 2; ++tx)
                for (int c = 0; c < 3; ++c)
                    ftx[tx][c] = {h_out[6 * i + 3 * tx + c].re,
                                  h_out[6 * i + 3 * tx + c].im};
            const auto channels = project_unit_channels(ftx, eh, ev, e0, plan.polarizations);
            for (size_t pi = 0; pi < npol; ++pi) {
                PoSampleResult& sample = out.samples[(b + i) * npol + pi];
                sample.sample_id = (static_cast<uint64_t>(fi) * ndir + di) * npol + pi;
                sample.frequency_id = fi;
                sample.direction_id = di;
                sample.pol_id = static_cast<uint32_t>(pi);
                sample.scattering = channels[pi].first;
                sample.rcs_sqm = channels[pi].second;
                sample.lit_facets = static_cast<size_t>(h_lit[i]);
            }
        }
        nvtxRangePop();
        nvtxRangePop(); // po_gpu_batch
    }
    return out;
}

} // namespace

PoResult solve_po_cuda(const NormalizedMesh& mesh, const SamplePlan& plan,
                       const nlohmann::json& resolved, int device_id,
                       const FringeOptions& fringe,
                       const GoOptions& go,
                       const MaterialOptions& material) {
    std::vector<std::pair<uint32_t, uint32_t>> units;
    for (uint32_t fi = 0; fi < plan.frequencies_hz.size(); ++fi)
        for (uint32_t di = 0; di < plan.directions.size(); ++di) units.emplace_back(fi, di);
    return solve_po_cuda_units(mesh, plan, resolved, device_id, units, fringe, go, material);
}

PoResult solve_po_cuda_units(const NormalizedMesh& mesh, const SamplePlan& plan,
                             const nlohmann::json& resolved, int device_id,
                             const std::vector<std::pair<uint32_t, uint32_t>>& units,
                             const FringeOptions& fringe,
                             const GoOptions& go,
                             const MaterialOptions& material) {
    const std::string precision = resolved["solver"].value("precision", "float64");
    if (precision == "float32")
        return solve_typed_cuda<float>(mesh, plan, resolved, device_id, units, fringe, go, material);
    if (precision == "mixed")
        return solve_mixed_cuda(mesh, plan, resolved, device_id, units, fringe, go, material);
    return solve_typed_cuda<double>(mesh, plan, resolved, device_id, units, fringe, go, material);
}

} // namespace strikecem::cuda

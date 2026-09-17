// CUDA Physical Optics backend (ADR-0002). One block per (frequency,
// direction) unit; threads stride over facets with a deterministic shared-
// memory tree reduction. Same equation, mesh, plan, and channel math as
// the CPU reference; only the accumulation order differs (gated, never
// bitwise-compared).
#include "strikecem/solvers/GpuPO.hpp"

#include <cmath>

#include <cuda_runtime.h>

#include "strikecem/solvers/Projection.hpp"

namespace strikecem::cuda {
namespace {

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err = (call);                                       \
        if (err != cudaSuccess)                                               \
            throw CudaError(std::string("CUDA failure: ") +                   \
                            cudaGetErrorString(err));                         \
    } while (0)

template <typename Real> struct Cplx {
    Real re{0}, im{0};
};

template <typename Real> struct UnitParams {
    Real kx, ky, kz;   // incident propagation direction (unit)
    Real rx, ry, rz;   // backscatter direction (-k_hat)
    Real ehx, ehy, ehz; // H-transmit E0 vector (with amplitude)
    Real evx, evy, evz; // V-transmit E0 vector (with amplitude)
    Real k, eta, e0;
};

__device__ inline void fsincos(float a, float* s, float* c) { sincosf(a, s, c); }
__device__ inline void fsincos(double a, double* s, double* c) { sincos(a, s, c); }

constexpr int kBlock = 256;

template <typename Real>
__global__ void po_kernel(const Real* __restrict__ centroids, // [3N]
                          const Real* __restrict__ normals,   // [3N]
                          const Real* __restrict__ areas,     // [N]
                          const UnitParams<Real>* __restrict__ units, // [U]
                          Cplx<Real>* __restrict__ out,       // [6U] F per tx
                          int* __restrict__ lit_out,          // [U] lit facets
                          int ntris) {
    const int u = static_cast<int>(blockIdx.x);
    const UnitParams<Real> p = units[u];
    __shared__ Real sh[2][3][2 * kBlock]; // [tx][xyz][re/im interleave]
    __shared__ int lit_sh[kBlock];
    Cplx<Real> sum[2][3] = {};
    int lit = 0;
    for (int t = static_cast<int>(threadIdx.x); t < ntris; t += kBlock) {
        const Real nx = normals[3 * t], ny = normals[3 * t + 1], nz = normals[3 * t + 2];
        if (!(nx * -p.kx + ny * -p.ky + nz * -p.kz > Real(0))) continue; // hard shadow
        ++lit;
        const Real cx = centroids[3 * t], cy = centroids[3 * t + 1], cz = centroids[3 * t + 2];
        Real s_in, c_in, s_out, c_out;
        fsincos(-p.k * (p.kx * cx + p.ky * cy + p.kz * cz), &s_in, &c_in);
        fsincos(p.k * (p.rx * cx + p.ry * cy + p.rz * cz), &s_out, &c_out);
        // Per-transmit incident H = (k x E0)/eta, phased at the centroid.
        const Real e0v[2][3] = {{p.ehx, p.ehy, p.ehz}, {p.evx, p.evy, p.evz}};
        for (int tx = 0; tx < 2; ++tx) {
            const Real* e = e0v[tx];
            // k x E0
            const Real kxx[3] = {p.ky * e[2] - p.kz * e[1], p.kz * e[0] - p.kx * e[2],
                                 p.kx * e[1] - p.ky * e[0]};
            // H complex = (k x E0)/eta * (c_in + j s_in)
            Real h_re[3], h_im[3];
            for (int i = 0; i < 3; ++i) {
                h_re[i] = kxx[i] / p.eta * c_in;
                h_im[i] = kxx[i] / p.eta * s_in;
            }
            // J = 2 n x H
            const Real nrm[3] = {nx, ny, nz};
            Real j_re[3], j_im[3];
            for (int i = 0; i < 3; ++i) {
                const int j = (i + 1) % 3, k3 = (i + 2) % 3;
                j_re[i] = 2 * (nrm[j] * h_re[k3] - nrm[k3] * h_re[j]);
                j_im[i] = 2 * (nrm[j] * h_im[k3] - nrm[k3] * h_im[j]);
            }
            // T = r x (r x J), ADR-0001 transverse projector.
            const Real rv[3] = {p.rx, p.ry, p.rz};
            Real t1_re[3], t1_im[3], t2_re[3], t2_im[3];
            for (int i = 0; i < 3; ++i) {
                const int j = (i + 1) % 3, k3 = (i + 2) % 3;
                t1_re[i] = rv[j] * j_re[k3] - rv[k3] * j_re[j];
                t1_im[i] = rv[j] * j_im[k3] - rv[k3] * j_im[j];
            }
            for (int i = 0; i < 3; ++i) {
                const int j = (i + 1) % 3, k3 = (i + 2) % 3;
                t2_re[i] = rv[j] * t1_re[k3] - rv[k3] * t1_re[j];
                t2_im[i] = rv[j] * t1_im[k3] - rv[k3] * t1_im[j];
            }
            // coeff = j*k*eta/4pi * A, times output phase.
            const Real a = areas[t];
            const Real ck_re = -p.k * p.eta / Real(4 * 3.141592653589793) * a * s_out;
            const Real ck_im = p.k * p.eta / Real(4 * 3.141592653589793) * a * c_out;
            for (int i = 0; i < 3; ++i) {
                sum[tx][i].re += ck_re * t2_re[i] - ck_im * t2_im[i];
                sum[tx][i].im += ck_re * t2_im[i] + ck_im * t2_re[i];
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
    const bool float32 = resolved["solver"].value("precision", "float64") == "float32";
    return triangles * (3 + 3 + 1) * (float32 ? 4 : 8);
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
    constexpr uint64_t kPerUnitBytes = 512; // params + outputs + slack
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
                          const std::vector<std::pair<uint32_t, uint32_t>>& units) {
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
    const size_t ntris = mesh.triangles.size();
    std::vector<Real> h_cent(3 * ntris), h_norm(3 * ntris), h_area(ntris);
    for (size_t t = 0; t < ntris; ++t) {
        const auto& tri = mesh.triangles[t];
        h_cent[3 * t] = static_cast<Real>((mesh.vertices[tri[0]].x + mesh.vertices[tri[1]].x +
                                           mesh.vertices[tri[2]].x) /
                                          3.0);
        h_cent[3 * t + 1] = static_cast<Real>((mesh.vertices[tri[0]].y + mesh.vertices[tri[1]].y +
                                               mesh.vertices[tri[2]].y) /
                                              3.0);
        h_cent[3 * t + 2] = static_cast<Real>((mesh.vertices[tri[0]].z + mesh.vertices[tri[1]].z +
                                               mesh.vertices[tri[2]].z) /
                                              3.0);
        h_norm[3 * t] = static_cast<Real>(mesh.normals[t].x);
        h_norm[3 * t + 1] = static_cast<Real>(mesh.normals[t].y);
        h_norm[3 * t + 2] = static_cast<Real>(mesh.normals[t].z);
        h_area[t] = static_cast<Real>(mesh.areas[t]);
    }
    Real *d_cent = nullptr, *d_norm = nullptr, *d_area = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cent, h_cent.size() * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_norm, h_norm.size() * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_area, h_area.size() * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_cent, h_cent.data(), h_cent.size() * sizeof(Real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norm, h_norm.data(), h_norm.size() * sizeof(Real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_area, h_area.data(), h_area.size() * sizeof(Real),
                          cudaMemcpyHostToDevice));

    const size_t override_units =
        static_cast<size_t>(resolved["execution"].value("cuda_batch_units", 0u));
    const size_t batch = cuda_select_batch_units(resolved, units.size(), ntris, device_id,
                                                 override_units);
    UnitParams<Real>* d_units = nullptr;
    Cplx<Real>* d_out = nullptr;
    int* d_lit = nullptr;
    CUDA_CHECK(cudaMalloc(&d_units, batch * sizeof(UnitParams<Real>)));
    CUDA_CHECK(cudaMalloc(&d_out, 6 * batch * sizeof(Cplx<Real>)));
    CUDA_CHECK(cudaMalloc(&d_lit, batch * sizeof(int)));
    std::vector<UnitParams<Real>> h_units(batch);
    std::vector<Cplx<Real>> h_out(6 * batch);
    std::vector<int> h_lit(batch);

    for (size_t b = 0; b < units.size(); b += batch) {
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
            p.rx = -p.kx;
            p.ry = -p.ky;
            p.rz = -p.kz;
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
        }
        CUDA_CHECK(cudaMemcpy(d_units, h_units.data(), count * sizeof(UnitParams<Real>),
                              cudaMemcpyHostToDevice));
        po_kernel<Real>
            <<<static_cast<unsigned>(count), kBlock>>>(d_cent, d_norm, d_area, d_units, d_out,
                                                       d_lit, static_cast<int>(ntris));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, 6 * count * sizeof(Cplx<Real>),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_lit.data(), d_lit, count * sizeof(int),
                              cudaMemcpyDeviceToHost));
        // Project on the host with the shared channel math.
        const size_t npol = plan.polarizations.size();
        const size_t ndir = plan.directions.size();
        for (size_t i = 0; i < count; ++i) {
            const uint32_t fi = units[b + i].first;
            const uint32_t di = units[b + i].second;
            const Direction& dir = plan.directions[di];
            const auto basis = polarization_basis(dir.k_hat);
            const double eh[3] = {basis.h.x, basis.h.y, basis.h.z};
            const double ev[3] = {basis.v.x, basis.v.y, basis.v.z};
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
    }
    CUDA_CHECK(cudaFree(d_cent));
    CUDA_CHECK(cudaFree(d_norm));
    CUDA_CHECK(cudaFree(d_area));
    CUDA_CHECK(cudaFree(d_units));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_lit));
    return out;
}

} // namespace

PoResult solve_po_cuda(const NormalizedMesh& mesh, const SamplePlan& plan,
                       const nlohmann::json& resolved, int device_id) {
    std::vector<std::pair<uint32_t, uint32_t>> units;
    for (uint32_t fi = 0; fi < plan.frequencies_hz.size(); ++fi)
        for (uint32_t di = 0; di < plan.directions.size(); ++di) units.emplace_back(fi, di);
    return solve_po_cuda_units(mesh, plan, resolved, device_id, units);
}

PoResult solve_po_cuda_units(const NormalizedMesh& mesh, const SamplePlan& plan,
                             const nlohmann::json& resolved, int device_id,
                             const std::vector<std::pair<uint32_t, uint32_t>>& units) {
    const std::string precision = resolved["solver"].value("precision", "float64");
    if (precision == "float32")
        return solve_typed_cuda<float>(mesh, plan, resolved, device_id, units);
    return solve_typed_cuda<double>(mesh, plan, resolved, device_id, units);
}

} // namespace strikecem::cuda

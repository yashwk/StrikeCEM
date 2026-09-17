#pragma once
// CUDA Physical Optics backend (ADR-0002): same mesh, plan, equation, and
// channel math as the CPU reference; only the facet accumulation moves to
// the device. No CUDA headers here, so CPU translation units include this
// freely; without SCEM_ENABLE_CUDA the stub throws CudaError.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace strikecem::cuda {

struct CudaError : public std::runtime_error {
    explicit CudaError(const std::string& msg) : std::runtime_error(msg) {}
};

struct CudaDeviceInfo {
    std::string name;
    uint64_t total_bytes = 0;
    uint64_t free_bytes = 0;
    int compute_major = 0;
    int compute_minor = 0;
};

// False when built without CUDA or when no device answers. Never throws.
bool cuda_available() noexcept;

// Throws CudaError when the device is missing.
CudaDeviceInfo cuda_device_info(int device_id);

// Device footprint of one run's resident facet buffers (centroids, normals,
// areas) in the configured precision. Needs no device.
uint64_t cuda_footprint_bytes(const nlohmann::json& resolved, size_t triangles);

// Batch size in (frequency, direction) units: override when > 0, else the
// largest batch fitting half of free device memory. Throws CudaError when
// even one unit cannot run (exit 4 at the CLI).
size_t cuda_select_batch_units(const nlohmann::json& resolved, size_t nunits, size_t triangles,
                               int device_id, size_t override_units);

// "" for CPU runs; device name for CUDA runs (throws when unavailable).
std::string active_device_name(const nlohmann::json& resolved);

// Full-plan GPU solve with correct global sample_ids.
PoResult solve_po_cuda(const NormalizedMesh& mesh, const SamplePlan& plan,
                       const nlohmann::json& resolved, int device_id);

// Subset GPU solve for resume (same contract as solve_po_units).
PoResult solve_po_cuda_units(const NormalizedMesh& mesh, const SamplePlan& plan,
                             const nlohmann::json& resolved, int device_id,
                             const std::vector<std::pair<uint32_t, uint32_t>>& units);

} // namespace strikecem::cuda

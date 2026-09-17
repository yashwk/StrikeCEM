// Stub when built without CUDA: every device operation refuses explicitly.
#include "strikecem/solvers/GpuPO.hpp"

namespace strikecem::cuda {

bool cuda_available() noexcept { return false; }

CudaDeviceInfo cuda_device_info(int) { throw CudaError("built without CUDA support"); }

uint64_t cuda_footprint_bytes(const nlohmann::json& resolved, size_t triangles) {
    const bool float32 = resolved["solver"].value("precision", "float64") == "float32";
    return triangles * (3 + 3 + 1) * (float32 ? 4 : 8);
}

size_t cuda_select_batch_units(const nlohmann::json&, size_t, size_t, int, size_t) {
    throw CudaError("built without CUDA support");
}

std::string active_device_name(const nlohmann::json& resolved) {
    if (resolved["execution"].value("accelerator", "cpu") != "cuda") return "";
    throw CudaError("built without CUDA support");
}

PoResult solve_po_cuda(const NormalizedMesh&, const SamplePlan&, const nlohmann::json&, int) {
    throw CudaError("built without CUDA support");
}

PoResult solve_po_cuda_units(const NormalizedMesh&, const SamplePlan&, const nlohmann::json&, int,
                             const std::vector<std::pair<uint32_t, uint32_t>>&) {
    throw CudaError("built without CUDA support");
}

} // namespace strikecem::cuda

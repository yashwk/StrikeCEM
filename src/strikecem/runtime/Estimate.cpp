// v1 resource estimator (SPEC FR-7).
#include "strikecem/runtime/Estimate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "strikecem/core/Conventions.hpp"
#include "strikecem/solvers/GpuPO.hpp"

namespace strikecem {
namespace {

namespace fs = std::filesystem;

// Documented rough cost: one triangle-sample-channel evaluation.
constexpr uint64_t kOpsPerTriSample = 200;
// Thread-local scratch per worker (row buffers, reduction temps).
constexpr uint64_t kThreadScratchBytes = 65536;
// CSV row width heuristic (fixed columns, default precision).
constexpr uint64_t kCsvBytesPerRow = 110;

std::string cpu_brand() {
#ifdef __linux__
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line))
        if (line.rfind("model name", 0) == 0) {
            const size_t colon = line.find(':');
            if (colon != std::string::npos) return line.substr(colon + 2);
        }
#endif
    return "unknown";
}

void collect_profiles(const fs::path& dir, std::vector<BenchmarkProfile>& out) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (entry.path().extension() != ".json") continue;
        std::ifstream in(entry.path());
        if (!in) continue;
        nlohmann::json j;
        try {
            in >> j;
        } catch (const std::exception&) {
            continue;
        }
        if (j.value("profile_format", "") != "scem-bench-1") continue;
        if (!j.contains("precision") || !j.contains("triangles") || !j.contains("samples") ||
            !j.contains("wall_seconds"))
            continue;
        BenchmarkProfile p;
        p.id = entry.path().stem().string();
        p.precision = j["precision"].get<std::string>();
        p.backend = j.value("backend", "cpu");
        p.triangles = j["triangles"].get<uint64_t>();
        p.samples = j["samples"].get<uint64_t>();
        p.wall_seconds = j["wall_seconds"].get<double>();
        p.cpu = j.value("cpu", "unknown");
        if (p.wall_seconds > 0.0 && p.triangles > 0 && p.samples > 0) out.push_back(p);
    }
}

} // namespace

std::vector<BenchmarkProfile> load_benchmark_profiles() {
    std::vector<BenchmarkProfile> profiles;
    if (const char* env = std::getenv("SCEM_BENCHMARK_DIR"))
        collect_profiles(fs::path(env), profiles);
    collect_profiles(fs::path("tools/benchmark_report/profiles"), profiles);
    return profiles;
}

void write_benchmark_profile(const std::string& path, const ResolvedConfig& rc,
                             uint64_t triangles, uint64_t samples, double wall_seconds,
                             const std::string& backend, const std::string& device) {
    nlohmann::json j;
    j["profile_format"] = "scem-bench-1";
    j["precision"] = rc.value["solver"]["precision"];
    j["backend"] = backend;
    j["device"] = device;
    j["triangles"] = triangles;
    j["samples"] = samples;
    j["wall_seconds"] = wall_seconds;
    j["cpu"] = cpu_brand();
    j["tool_version"] = SCEM_VERSION;
    j["git_revision"] = SCEM_GIT_REVISION;
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open benchmark profile: " + path);
    out << j.dump(2) << "\n";
    out.flush();
    if (!out) throw std::runtime_error("failed writing benchmark profile: " + path);
}

ResourceEstimate estimate_resources(const nlohmann::json& resolved, const SamplePlan& plan,
                                    const NormalizedMesh& mesh,
                                    const std::vector<BenchmarkProfile>& profiles) {
    ResourceEstimate est;
    const uint64_t tris = mesh.report.triangle_count;
    const uint64_t rows = plan.sample_count();
    est.mesh_bytes = mesh.report.vertex_count * 24 + tris * (12 + 24 + 8);
    const bool float32 = resolved["solver"]["precision"].get<std::string>() == "float32";
    const uint64_t float_bytes = float32 ? 4 : 8;
    if (resolved["output"]["format"].get<std::string>() == "csv")
        est.output_bytes = rows * kCsvBytesPerRow;
    else
        est.output_bytes = rows * (18 + 4 * float_bytes + 1 + 4);
    const unsigned workers =
        std::max(1u, resolved["execution"]["cpu_threads"].get<unsigned>());
    est.thread_bytes = static_cast<uint64_t>(workers) * kThreadScratchBytes;
    est.total_bytes = est.mesh_bytes + est.output_bytes + est.thread_bytes;
    est.limit_bytes = resolved["execution"]["max_memory_mb"].get<double>() * 1048576.0;
    est.fits = static_cast<double>(est.total_bytes) <= est.limit_bytes;
    est.operation_count = tris * rows * kOpsPerTriSample;
    if (resolved["solver"]["po_options"].value("shadowing", false) ||
        resolved["solver"]["po_options"].value("max_bounces", 1) > 1)
        est.warnings.emplace_back("GO corrections are excluded from the operation count; "
                                  "treat calibrated runtime ranges as lower bounds");
    est.backend = resolved["execution"].value("accelerator", "cpu");
    if (est.backend == "cuda") est.device_bytes = cuda::cuda_footprint_bytes(resolved, tris);

    double max_freq = 0.0;
    for (double f : plan.frequencies_hz) max_freq = std::max(max_freq, f);
    const double lambda_min = kSpeedOfLight / max_freq;
    const auto diag = mesh.report.bbox_max - mesh.report.bbox_min;
    est.electrical_size_wavelengths = diag.length() / lambda_min;
    for (double f : plan.frequencies_hz) {
        const double lambda = kSpeedOfLight / f;
        if (mesh.report.max_edge_length_m > lambda / 10.0) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "mesh may be under-resolved at %.6g Hz: max edge %.6g m vs "
                          "lambda/10 = %.6g m",
                          f, mesh.report.max_edge_length_m, lambda / 10.0);
            est.warnings.emplace_back(buf);
        }
    }
    if (!est.fits) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "estimated %llu bytes exceed configured limit of %.6g MB",
                      static_cast<unsigned long long>(est.total_bytes),
                      est.limit_bytes / 1048576.0);
        est.warnings.emplace_back(buf);
    }

    // Calibration: same precision and backend, triangle and sample counts
    // each within 4x; nearest in log space wins. Predicted range is 2x each
    // way; anything else stays explicitly uncalibrated.
    const BenchmarkProfile* best = nullptr;
    double best_score = 0.0;
    for (const auto& p : profiles) {
        if (p.precision != resolved["solver"]["precision"].get<std::string>()) continue;
        if (p.backend != est.backend) continue;
        const double rt = static_cast<double>(tris) / p.triangles;
        const double rs = static_cast<double>(rows) / p.samples;
        if (rt < 0.25 || rt > 4.0 || rs < 0.25 || rs > 4.0) continue;
        const double score = std::max({rt, 1.0 / rt, rs, 1.0 / rs});
        if (!best || score < best_score) {
            best = &p;
            best_score = score;
        }
    }
    if (best) {
        const double throughput =
            static_cast<double>(best->triangles) * best->samples * kOpsPerTriSample /
            best->wall_seconds;
        const double predicted = static_cast<double>(est.operation_count) / throughput;
        est.calibrated = true;
        est.runtime_lo_s = predicted / 2.0;
        est.runtime_hi_s = predicted * 2.0;
        est.profile_id = best->id;
    }
    return est;
}

} // namespace strikecem

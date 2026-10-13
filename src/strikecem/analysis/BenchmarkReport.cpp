#include "strikecem/analysis/BenchmarkReport.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace strikecem::analysis {

namespace fs = std::filesystem;

BenchmarkAggregation aggregate_profiles(const std::vector<BenchmarkProfile>& profiles) {
    BenchmarkAggregation agg;
    if (profiles.empty()) return agg;

    double cpu_sum = 0.0;
    size_t cpu_count = 0;
    double cuda_sum = 0.0;
    size_t cuda_count = 0;

    for (const auto& p : profiles) {
        if (p.wall_seconds <= 0.0 || p.triangles == 0 || p.samples == 0) continue;
        const double meval_per_sec =
            (static_cast<double>(p.triangles) * static_cast<double>(p.samples)) /
            (p.wall_seconds * 1.0e6);

        BenchmarkEntry entry;
        entry.profile = p;
        entry.throughput_meval_per_sec = meval_per_sec;
        agg.entries.push_back(entry);

        agg.max_throughput = std::max(agg.max_throughput, meval_per_sec);

        if (p.backend == "cuda") {
            cuda_sum += meval_per_sec;
            ++cuda_count;
        } else {
            cpu_sum += meval_per_sec;
            ++cpu_count;
        }
    }

    if (cpu_count > 0) agg.cpu_avg_throughput = cpu_sum / static_cast<double>(cpu_count);
    if (cuda_count > 0) agg.cuda_avg_throughput = cuda_sum / static_cast<double>(cuda_count);

    if (agg.cpu_avg_throughput > 0.0 && agg.cuda_avg_throughput > 0.0) {
        agg.speedup_ratio = agg.cuda_avg_throughput / agg.cpu_avg_throughput;
    }

    return agg;
}

namespace {

void collect_from_path(const fs::path& p, std::vector<BenchmarkProfile>& out) {
    std::error_code ec;
    if (fs::is_regular_file(p, ec) && p.extension() == ".json") {
        std::ifstream in(p);
        if (!in) return;
        nlohmann::json j;
        try { in >> j; } catch (...) { return; }
        if (j.value("profile_format", "") != "scem-bench-1") return;
        if (!j.contains("precision") || !j.contains("triangles") || !j.contains("samples") ||
            !j.contains("wall_seconds"))
            return;
        BenchmarkProfile prof;
        prof.id = p.stem().string();
        prof.precision = j["precision"].get<std::string>();
        prof.backend = j.value("backend", "cpu");
        prof.triangles = j["triangles"].get<uint64_t>();
        prof.samples = j["samples"].get<uint64_t>();
        prof.wall_seconds = j["wall_seconds"].get<double>();
        prof.cpu = j.value("cpu", "unknown");
        if (prof.wall_seconds > 0.0 && prof.triangles > 0 && prof.samples > 0)
            out.push_back(prof);
    } else if (fs::is_directory(p, ec)) {
        for (const auto& entry : fs::directory_iterator(p, ec)) {
            collect_from_path(entry.path(), out);
        }
    }
}

} // namespace

BenchmarkAggregation load_and_aggregate(const std::vector<std::string>& search_paths) {
    std::vector<BenchmarkProfile> profiles;
    if (search_paths.empty()) {
        profiles = load_benchmark_profiles();
    } else {
        for (const auto& sp : search_paths) {
            collect_from_path(fs::path(sp), profiles);
        }
    }
    return aggregate_profiles(profiles);
}

std::string render_benchmark_ascii(const BenchmarkAggregation& agg, int bar_width) {
    std::ostringstream ss;
    ss << "================================================================================\n";
    ss << "StrikeCEM Benchmark Scaling & Throughput Report\n";
    ss << "================================================================================\n";
    ss << std::left << std::setw(28) << "ID"
       << std::setw(8)  << "Backend"
       << std::setw(10) << "Prec"
       << std::right << std::setw(12) << "M-Eval/s"
       << "  Throughput Bar\n";
    ss << "--------------------------------------------------------------------------------\n";

    if (agg.entries.empty()) {
        ss << "  (no benchmark profiles loaded)\n";
    } else {
        const double max_val = agg.max_throughput > 0.0 ? agg.max_throughput : 1.0;
        for (const auto& e : agg.entries) {
            std::string id_str = e.profile.id;
            if (id_str.length() > 26) id_str = id_str.substr(0, 23) + "...";

            int filled = static_cast<int>(std::round((e.throughput_meval_per_sec / max_val) * bar_width));
            filled = std::max(0, std::min(bar_width, filled));
            std::string bar = "[" + std::string(filled, '#') + std::string(bar_width - filled, ' ') + "]";

            ss << std::left << std::setw(28) << id_str
               << std::setw(8)  << e.profile.backend
               << std::setw(10) << e.profile.precision
               << std::right << std::setw(12) << std::fixed << std::setprecision(2)
               << e.throughput_meval_per_sec
               << "  " << bar << "\n";
        }
    }

    ss << "--------------------------------------------------------------------------------\n";
    ss << "Summary:\n";
    ss << "  Profiles aggregated: " << agg.entries.size() << "\n";
    ss << "  CPU Avg Throughput:  " << std::fixed << std::setprecision(2)
       << agg.cpu_avg_throughput << " M-Eval/s\n";
    ss << "  CUDA Avg Throughput: " << std::fixed << std::setprecision(2)
       << agg.cuda_avg_throughput << " M-Eval/s\n";
    if (agg.cuda_avg_throughput > 0.0 && agg.cpu_avg_throughput > 0.0) {
        ss << "  CUDA vs CPU Speedup: " << std::fixed << std::setprecision(2)
           << agg.speedup_ratio << "x\n";
    }
    ss << "================================================================================\n";
    return ss.str();
}

nlohmann::json benchmark_aggregation_to_json(const BenchmarkAggregation& agg) {
    nlohmann::json root;
    root["profile_count"] = agg.entries.size();
    root["cpu_avg_meval_per_sec"] = agg.cpu_avg_throughput;
    root["cuda_avg_meval_per_sec"] = agg.cuda_avg_throughput;
    root["speedup_ratio"] = agg.speedup_ratio;
    root["max_meval_per_sec"] = agg.max_throughput;

    nlohmann::json items = nlohmann::json::array();
    for (const auto& e : agg.entries) {
        nlohmann::json item;
        item["id"] = e.profile.id;
        item["backend"] = e.profile.backend;
        item["precision"] = e.profile.precision;
        item["triangles"] = e.profile.triangles;
        item["samples"] = e.profile.samples;
        item["wall_seconds"] = e.profile.wall_seconds;
        item["cpu"] = e.profile.cpu;
        item["throughput_meval_per_sec"] = e.throughput_meval_per_sec;
        items.push_back(item);
    }
    root["profiles"] = items;
    return root;
}

} // namespace strikecem::analysis

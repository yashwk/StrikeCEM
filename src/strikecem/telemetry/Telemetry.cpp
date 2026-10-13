#include "strikecem/telemetry/Telemetry.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#if defined(__linux__)
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace strikecem::telemetry {

namespace fs = std::filesystem;

TelemetryCollector::TelemetryCollector()
    : global_start_(std::chrono::steady_clock::now()) {}

void TelemetryCollector::start_stage(const std::string& name) {
    active_stages_[name] = std::chrono::steady_clock::now();
}

void TelemetryCollector::end_stage(const std::string& name) {
    auto it = active_stages_.find(name);
    if (it == active_stages_.end()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double dur = std::chrono::duration<double>(now - it->second).count();
    const double start_sec = std::chrono::duration<double>(it->second - global_start_).count();
    const double end_sec = std::chrono::duration<double>(now - global_start_).count();

    StageSpan span;
    span.name = name;
    span.duration_seconds = dur;
    span.start_timestamp_sec = start_sec;
    span.end_timestamp_sec = end_sec;
    completed_stages_.push_back(span);

    active_stages_.erase(it);
}

void TelemetryCollector::record_metric(const std::string& key, double value) {
    metrics_[key] = value;
}

void TelemetryCollector::record_counter(const std::string& key, uint64_t count) {
    counters_[key] = count;
}

void TelemetryCollector::record_metadata(const std::string& key, const std::string& value) {
    metadata_[key] = value;
}

void TelemetryCollector::record_warning(const std::string& warning) {
    warnings_.push_back(warning);
}

MemoryMetrics TelemetryCollector::query_memory_metrics() const {
    MemoryMetrics mem;

#if defined(__linux__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // On Linux, ru_maxrss is reported in kilobytes
        mem.max_rss_rusage_bytes = static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
    }

    std::ifstream status_file("/proc/self/status");
    if (status_file.is_open()) {
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.rfind("VmHWM:", 0) == 0) {
                std::istringstream iss(line.substr(6));
                uint64_t val_kb = 0;
                if (iss >> val_kb) {
                    mem.vm_hwm_bytes = val_kb * 1024ULL;
                }
            } else if (line.rfind("VmRSS:", 0) == 0) {
                std::istringstream iss(line.substr(6));
                uint64_t val_kb = 0;
                if (iss >> val_kb) {
                    mem.vm_rss_bytes = val_kb * 1024ULL;
                }
            }
        }
    }
#endif

    if (mem.vm_hwm_bytes == 0 && mem.max_rss_rusage_bytes > 0) {
        mem.vm_hwm_bytes = mem.max_rss_rusage_bytes;
    }

    return mem;
}

nlohmann::json TelemetryCollector::export_json() const {
    const auto now = std::chrono::steady_clock::now();
    const double total_duration = std::chrono::duration<double>(now - global_start_).count();

    nlohmann::json root;
    root["telemetry_schema_version"] = "1.0";
    root["total_wall_time_seconds"] = total_duration;

    nlohmann::json meta_json = nlohmann::json::object();
    for (const auto& [k, v] : metadata_) {
        meta_json[k] = v;
    }
    root["metadata"] = meta_json;

    nlohmann::json stages_json = nlohmann::json::array();
    for (const auto& st : completed_stages_) {
        nlohmann::json s;
        s["name"] = st.name;
        s["duration_seconds"] = st.duration_seconds;
        s["start_seconds"] = st.start_timestamp_sec;
        s["end_seconds"] = st.end_timestamp_sec;
        s["percent_total"] = (total_duration > 0.0) ? (st.duration_seconds / total_duration * 100.0) : 0.0;
        stages_json.push_back(s);
    }
    root["stages"] = stages_json;

    nlohmann::json metrics_json = nlohmann::json::object();
    for (const auto& [k, v] : metrics_) {
        metrics_json[k] = v;
    }
    root["metrics"] = metrics_json;

    nlohmann::json counters_json = nlohmann::json::object();
    for (const auto& [k, v] : counters_) {
        counters_json[k] = v;
    }
    root["counters"] = counters_json;

    const MemoryMetrics mem = query_memory_metrics();
    nlohmann::json mem_json;
    mem_json["vm_hwm_bytes"] = mem.vm_hwm_bytes;
    mem_json["vm_hwm_mb"] = static_cast<double>(mem.vm_hwm_bytes) / (1024.0 * 1024.0);
    mem_json["vm_rss_bytes"] = mem.vm_rss_bytes;
    mem_json["vm_rss_mb"] = static_cast<double>(mem.vm_rss_bytes) / (1024.0 * 1024.0);
    mem_json["max_rss_rusage_bytes"] = mem.max_rss_rusage_bytes;
    root["memory"] = mem_json;

    root["warnings"] = warnings_;

    return root;
}

void TelemetryCollector::write_json_report(const std::string& filepath) const {
    const fs::path target(filepath);
    if (!target.parent_path().empty()) {
        std::error_code ec;
        fs::create_directories(target.parent_path(), ec);
    }

    const fs::path tmp_path = target.string() + ".tmp." + std::to_string(::getpid());
    {
        std::ofstream out(tmp_path);
        if (!out) {
            throw std::runtime_error("failed to open telemetry file for writing: " + filepath);
        }
        out << export_json().dump(2) << "\n";
    }

    std::error_code ec;
    fs::rename(tmp_path, target, ec);
    if (ec) {
        fs::remove(tmp_path, ec);
        throw std::runtime_error("failed to finalize telemetry file: " + ec.message());
    }
}

} // namespace strikecem::telemetry

#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace strikecem::telemetry {

struct StageSpan {
    std::string name;
    double duration_seconds = 0.0;
    double start_timestamp_sec = 0.0;
    double end_timestamp_sec = 0.0;
};

struct MemoryMetrics {
    uint64_t vm_hwm_bytes = 0;   // Peak resident set size (High-Water Mark)
    uint64_t vm_rss_bytes = 0;   // Current resident set size
    uint64_t max_rss_rusage_bytes = 0; // ru_maxrss from getrusage
};

class TelemetryCollector {
public:
    TelemetryCollector();

    void start_stage(const std::string& name);
    void end_stage(const std::string& name);

    void record_metric(const std::string& key, double value);
    void record_counter(const std::string& key, uint64_t count);
    void record_metadata(const std::string& key, const std::string& value);
    void record_warning(const std::string& warning);

    // Queries OS-level memory metrics (/proc/self/status on Linux, getrusage fallback)
    MemoryMetrics query_memory_metrics() const;

    // Produces structured JSON report matching the diagnostics contract
    nlohmann::json export_json() const;

    // Writes report atomically to specified path
    void write_json_report(const std::string& filepath) const;

    const std::vector<StageSpan>& stages() const { return completed_stages_; }
    const std::map<std::string, double>& metrics() const { return metrics_; }
    const std::map<std::string, uint64_t>& counters() const { return counters_; }

private:
    std::chrono::steady_clock::time_point global_start_;
    std::map<std::string, std::chrono::steady_clock::time_point> active_stages_;
    std::vector<StageSpan> completed_stages_;
    std::map<std::string, double> metrics_;
    std::map<std::string, uint64_t> counters_;
    std::map<std::string, std::string> metadata_;
    std::vector<std::string> warnings_;
};

class ScopedStageTimer {
public:
    ScopedStageTimer(TelemetryCollector& tc, std::string name)
        : tc_(tc), name_(std::move(name)) {
        tc_.start_stage(name_);
    }
    ~ScopedStageTimer() {
        tc_.end_stage(name_);
    }
private:
    TelemetryCollector& tc_;
    std::string name_;
};

} // namespace strikecem::telemetry

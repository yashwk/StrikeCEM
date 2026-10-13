#pragma once
// Headless cluster worker daemon (Domain 4).
// Pulls shard tasks from coordinator, executes solver engine, sends heartbeats,
// and commits completed shards with SHA-256 verification.
#include <atomic>
#include <cstddef>
#include <string>

#include <nlohmann/json.hpp>

namespace strikecem {

struct WorkerOptions {
    std::string coordinator_url = "http://127.0.0.1:8080";
    std::string worker_id;
    std::string device = "cpu"; // "cpu", "cuda", "cuda:0", etc.
    std::string schema_path;
    int poll_interval_ms = 200;
    size_t max_tasks = 0; // 0 = run until campaign completes
};

struct WorkerResult {
    bool success = false;
    size_t tasks_completed = 0;
    std::string error_message;
};

class WorkerDaemon {
public:
    explicit WorkerDaemon(const WorkerOptions& options);
    ~WorkerDaemon();

    WorkerResult run();
    void stop();

    // Direct HTTP client helpers
    nlohmann::json send_request(const std::string& method, const std::string& path,
                               const nlohmann::json& body = nlohmann::json::object());

private:
    void parse_url(const std::string& url, std::string& out_host, int& out_port);
    std::string compute_sha256(const std::string& file_path);

    WorkerOptions options_;
    std::string host_ = "127.0.0.1";
    int port_ = 8080;
    std::string worker_id_;
    std::atomic<bool> running_{false};
};

} // namespace strikecem

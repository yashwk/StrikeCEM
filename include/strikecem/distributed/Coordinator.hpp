#pragma once
// Distributed cluster campaign coordinator broker (Domain 4).
// Manages shard leases, worker heartbeats, fault-tolerant retry, and automated merge reduction.
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/distributed/Shard.hpp"

namespace strikecem {

enum class TaskState {
    Pending = 0,
    Leased = 1,
    Completed = 2,
    Failed = 3
};

struct CoordinatorTask {
    size_t shard_id = 0;
    TaskState state = TaskState::Pending;
    std::string worker_id;
    std::string config_path;
    std::string output_path;
    std::string checksum;
    std::chrono::steady_clock::time_point lease_expires_at;
    int retry_count = 0;
};

struct CoordinatorOptions {
    std::string manifest_path;
    std::string output_path;
    std::string schema_path;
    std::string host = "127.0.0.1";
    int port = 0; // 0 = pick available ephemeral port
    int lease_duration_seconds = 30;
    bool auto_merge = true;
};

class CoordinatorServer {
public:
    explicit CoordinatorServer(const CoordinatorOptions& options);
    ~CoordinatorServer();

    // Starts HTTP REST broker. If background=true, runs asynchronously in separate thread.
    void start(bool background = true);
    void stop();

    bool is_running() const { return running_.load(); }
    int bound_port() const { return bound_port_; }
    const std::string& run_id() const { return run_id_; }

    // Direct programmatic API
    nlohmann::json acquire_task(const std::string& worker_id, const std::string& device_info);
    nlohmann::json heartbeat_task(size_t shard_id, const std::string& worker_id);
    nlohmann::json complete_task(size_t shard_id, const std::string& worker_id,
                                 const std::string& output_path, const std::string& checksum);
    nlohmann::json fail_task(size_t shard_id, const std::string& worker_id, const std::string& error);
    nlohmann::json get_status() const;

    bool is_campaign_complete() const;

private:
    void init_tasks_from_manifest();
    void server_loop();
    void handle_client(int client_fd);
    void check_expired_leases_locked();

    CoordinatorOptions options_;
    std::string run_id_;
    int bound_port_ = 0;
    int server_fd_ = -1;
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> server_thread_;

    mutable std::mutex mutex_;
    std::vector<CoordinatorTask> tasks_;
    size_t completed_count_ = 0;
    bool merge_executed_ = false;
};

} // namespace strikecem

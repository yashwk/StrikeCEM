#include "strikecem/distributed/Coordinator.hpp"

#include <arpa/inet.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

#include "strikecem/io/OutputError.hpp"

namespace fs = std::filesystem;

namespace strikecem {

CoordinatorServer::CoordinatorServer(const CoordinatorOptions& options)
    : options_(options) {
    init_tasks_from_manifest();
}

CoordinatorServer::~CoordinatorServer() {
    stop();
}

void CoordinatorServer::init_tasks_from_manifest() {
    if (!fs::exists(options_.manifest_path)) {
        throw OutputError("Manifest file does not exist: " + options_.manifest_path);
    }

    std::ifstream in(options_.manifest_path);
    if (!in) {
        throw OutputError("Cannot open manifest file: " + options_.manifest_path);
    }

    nlohmann::json j;
    in >> j;

    run_id_ = j.value("run_id", "anonymous_run");
    tasks_.clear();
    completed_count_ = 0;
    merge_executed_ = false;

    if (!j.contains("shards") || !j["shards"].is_array()) {
        throw OutputError("Manifest missing valid 'shards' array: " + options_.manifest_path);
    }

    const fs::path manifest_dir = fs::path(options_.manifest_path).parent_path();
    for (const auto& s : j["shards"]) {
        CoordinatorTask task;
        task.shard_id = s["shard_id"].get<size_t>();
        task.state = TaskState::Pending;
        task.config_path = s["config_path"].get<std::string>();
        task.output_path = s["output_path"].get<std::string>();

        // Make relative paths absolute to manifest directory
        if (fs::path(task.config_path).is_relative()) {
            task.config_path = (manifest_dir / task.config_path).string();
        }
        if (fs::path(task.output_path).is_relative()) {
            task.output_path = (manifest_dir / task.output_path).string();
        }

        tasks_.push_back(task);
    }
}

void CoordinatorServer::check_expired_leases_locked() {
    const auto now = std::chrono::steady_clock::now();
    for (auto& task : tasks_) {
        if (task.state == TaskState::Leased && now > task.lease_expires_at) {
            task.state = TaskState::Pending;
            task.worker_id.clear();
            task.retry_count++;
        }
    }
}

nlohmann::json CoordinatorServer::acquire_task(const std::string& worker_id,
                                              const std::string& device_info) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_expired_leases_locked();

    if (completed_count_ == tasks_.size() && !tasks_.empty()) {
        return {
            {"status", "campaign_complete"},
            {"all_complete", true},
            {"run_id", run_id_}
        };
    }

    for (auto& task : tasks_) {
        if (task.state == TaskState::Pending) {
            task.state = TaskState::Leased;
            task.worker_id = worker_id;
            task.lease_expires_at = std::chrono::steady_clock::now() +
                                    std::chrono::seconds(options_.lease_duration_seconds);

            return {
                {"status", "acquired"},
                {"all_complete", false},
                {"shard_id", task.shard_id},
                {"config_path", task.config_path},
                {"output_path", task.output_path},
                {"lease_duration_seconds", options_.lease_duration_seconds},
                {"device_info", device_info}
            };
        }
    }

    return {
        {"status", "none_available"},
        {"all_complete", false},
        {"run_id", run_id_}
    };
}

nlohmann::json CoordinatorServer::heartbeat_task(size_t shard_id, const std::string& worker_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_expired_leases_locked();

    for (auto& task : tasks_) {
        if (task.shard_id == shard_id) {
            if (task.state == TaskState::Leased && task.worker_id == worker_id) {
                task.lease_expires_at = std::chrono::steady_clock::now() +
                                        std::chrono::seconds(options_.lease_duration_seconds);
                return {{"status", "ok"}, {"lease_renewed", true}};
            } else {
                return {{"status", "error"}, {"message", "Task not leased to this worker"}};
            }
        }
    }
    return {{"status", "error"}, {"message", "Shard ID not found"}};
}

nlohmann::json CoordinatorServer::complete_task(size_t shard_id, const std::string& worker_id,
                                               const std::string& output_path,
                                               const std::string& checksum) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& task : tasks_) {
        if (task.shard_id == shard_id) {
            if (task.state == TaskState::Completed) {
                return {{"status", "ok"}, {"already_completed", true}};
            }
            task.state = TaskState::Completed;
            task.checksum = checksum;
            if (!output_path.empty()) {
                task.output_path = output_path;
            }
            completed_count_++;

            bool merged = false;
            if (completed_count_ == tasks_.size() && options_.auto_merge && !merge_executed_) {
                try {
                    merge_shards(options_.manifest_path, {}, options_.output_path, options_.schema_path);
                    merged = true;
                    merge_executed_ = true;
                } catch (const std::exception& e) {
                    return {
                        {"status", "error"},
                        {"message", "Merge failed: " + std::string(e.what())}
                    };
                }
            }

            return {
                {"status", "ok"},
                {"shard_id", shard_id},
                {"completed_count", completed_count_},
                {"total_shards", tasks_.size()},
                {"all_complete", completed_count_ == tasks_.size()},
                {"auto_merged", merged}
            };
        }
    }
    return {{"status", "error"}, {"message", "Shard ID not found"}};
}

nlohmann::json CoordinatorServer::fail_task(size_t shard_id, const std::string& worker_id,
                                           const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& task : tasks_) {
        if (task.shard_id == shard_id) {
            task.state = TaskState::Pending;
            task.worker_id.clear();
            task.retry_count++;
            return {
                {"status", "ok"},
                {"shard_id", shard_id},
                {"retry_count", task.retry_count},
                {"reported_error", error}
            };
        }
    }
    return {{"status", "error"}, {"message", "Shard ID not found"}};
}

nlohmann::json CoordinatorServer::get_status() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
    const_cast<CoordinatorServer*>(this)->check_expired_leases_locked();

    size_t pending = 0;
    size_t leased = 0;
    size_t completed = 0;

    for (const auto& t : tasks_) {
        if (t.state == TaskState::Pending) pending++;
        else if (t.state == TaskState::Leased) leased++;
        else if (t.state == TaskState::Completed) completed++;
    }

    const double progress = tasks_.empty() ? 100.0 : (100.0 * completed / tasks_.size());

    return {
        {"run_id", run_id_},
        {"total_shards", tasks_.size()},
        {"completed_shards", completed},
        {"leased_shards", leased},
        {"pending_shards", pending},
        {"progress_percent", progress},
        {"all_complete", completed == tasks_.size()},
        {"merged", merge_executed_}
    };
}

bool CoordinatorServer::is_campaign_complete() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !tasks_.empty() && completed_count_ == tasks_.size();
}

void CoordinatorServer::start(bool background) {
    if (running_.load()) return;

    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        throw OutputError("Failed to create coordinator server socket");
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(options_.port);
    inet_pton(AF_INET, options_.host.c_str(), &addr.sin_addr);

    if (bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(server_fd_);
        server_fd_ = -1;
        throw OutputError("Failed to bind coordinator socket to " + options_.host + ":" +
                          std::to_string(options_.port));
    }

    if (listen(server_fd_, 64) < 0) {
        ::close(server_fd_);
        server_fd_ = -1;
        throw OutputError("Failed to listen on coordinator socket");
    }

    // Retrieve actual assigned port if ephemeral (0) was requested
    sockaddr_in bound_addr{};
    socklen_t len = sizeof(bound_addr);
    if (getsockname(server_fd_, reinterpret_cast<sockaddr*>(&bound_addr), &len) == 0) {
        bound_port_ = ntohs(bound_addr.sin_port);
    } else {
        bound_port_ = options_.port;
    }

    running_.store(true);

    if (background) {
        server_thread_ = std::make_unique<std::thread>(&CoordinatorServer::server_loop, this);
    } else {
        server_loop();
    }
}

void CoordinatorServer::stop() {
    if (!running_.load()) return;
    running_.store(false);

    if (server_fd_ >= 0) {
        shutdown(server_fd_, SHUT_RDWR);
        ::close(server_fd_);
        server_fd_ = -1;
    }

    if (server_thread_ && server_thread_->joinable()) {
        server_thread_->join();
        server_thread_.reset();
    }
}

void CoordinatorServer::server_loop() {
    while (running_.load()) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (!running_.load()) break;
            continue;
        }

        std::thread([this, client_fd]() {
            handle_client(client_fd);
        }).detach();
    }
}

void CoordinatorServer::handle_client(int client_fd) {
    std::string raw_req;
    char buf[4096];
    ssize_t bytes_read = 0;

    while ((bytes_read = recv(client_fd, buf, sizeof(buf), 0)) > 0) {
        raw_req.append(buf, bytes_read);
        if (raw_req.find("\r\n\r\n") != std::string::npos) {
            // Check if Content-Length body has arrived
            size_t header_end = raw_req.find("\r\n\r\n");
            size_t cl_pos = raw_req.find("Content-Length:");
            if (cl_pos == std::string::npos) cl_pos = raw_req.find("content-length:");
            if (cl_pos != std::string::npos && cl_pos < header_end) {
                size_t num_start = raw_req.find_first_of("0123456789", cl_pos);
                size_t num_end = raw_req.find("\r\n", num_start);
                size_t content_len = std::stoull(raw_req.substr(num_start, num_end - num_start));
                size_t body_len = raw_req.size() - (header_end + 4);
                if (body_len < content_len) continue;
            }
            break;
        }
    }

    if (raw_req.empty()) {
        ::close(client_fd);
        return;
    }

    std::istringstream req_stream(raw_req);
    std::string method, path, http_ver;
    req_stream >> method >> path >> http_ver;

    std::string body;
    size_t body_pos = raw_req.find("\r\n\r\n");
    if (body_pos != std::string::npos) {
        body = raw_req.substr(body_pos + 4);
    }

    nlohmann::json req_json;
    if (!body.empty()) {
        try {
            req_json = nlohmann::json::parse(body);
        } catch (...) {
            // Ignore parse errors, body may be empty
        }
    }

    nlohmann::json res_json;
    int status_code = 200;

    if (path == "/api/v1/health") {
        res_json = {{"status", "ok"}, {"run_id", run_id_}, {"version", "1.0"}};
    } else if (path == "/api/v1/status") {
        res_json = get_status();
    } else if (path == "/api/v1/task/acquire" && method == "POST") {
        const std::string wid = req_json.value("worker_id", "anon_worker");
        const std::string dev = req_json.value("device", "cpu");
        res_json = acquire_task(wid, dev);
    } else if (path == "/api/v1/task/heartbeat" && method == "POST") {
        const size_t sid = req_json.value("shard_id", size_t(0));
        const std::string wid = req_json.value("worker_id", "");
        res_json = heartbeat_task(sid, wid);
    } else if (path == "/api/v1/task/complete" && method == "POST") {
        const size_t sid = req_json.value("shard_id", size_t(0));
        const std::string wid = req_json.value("worker_id", "");
        const std::string out_path = req_json.value("output_path", "");
        const std::string csum = req_json.value("checksum", "");
        res_json = complete_task(sid, wid, out_path, csum);
    } else if (path == "/api/v1/task/fail" && method == "POST") {
        const size_t sid = req_json.value("shard_id", size_t(0));
        const std::string wid = req_json.value("worker_id", "");
        const std::string err = req_json.value("error", "unknown error");
        res_json = fail_task(sid, wid, err);
    } else {
        status_code = 404;
        res_json = {{"error", "Not Found"}, {"path", path}};
    }

    const std::string res_body = res_json.dump();
    std::ostringstream res_stream;
    res_stream << "HTTP/1.1 " << status_code << " OK\r\n"
               << "Content-Type: application/json\r\n"
               << "Content-Length: " << res_body.size() << "\r\n"
               << "Connection: close\r\n\r\n"
               << res_body;

    const std::string res_str = res_stream.str();
    send(client_fd, res_str.data(), res_str.size(), 0);
    ::close(client_fd);
}

} // namespace strikecem

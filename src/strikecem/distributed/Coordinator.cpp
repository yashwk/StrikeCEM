#include "strikecem/distributed/Coordinator.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <netinet/in.h>
#include <openssl/evp.h>
#include <sstream>
#include <sys/socket.h>
#include <sys/time.h>
#include <system_error>
#include <unistd.h>

#include "strikecem/io/OutputError.hpp"

namespace fs = std::filesystem;

namespace strikecem {

namespace {

std::string sha256_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) return {};
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        return {};
    }

    char buffer[65536];
    while (true) {
        file.read(buffer, sizeof(buffer));
        const std::streamsize count = file.gcount();
        if (count > 0 && EVP_DigestUpdate(ctx, buffer, static_cast<size_t>(count)) != 1) {
            EVP_MD_CTX_free(ctx);
            return {};
        }
        if (file.eof()) break;
        if (!file) {
            EVP_MD_CTX_free(ctx);
            return {};
        }
    }

    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_size = 0;
    if (EVP_DigestFinal_ex(ctx, digest, &digest_size) != 1) {
        EVP_MD_CTX_free(ctx);
        return {};
    }
    EVP_MD_CTX_free(ctx);

    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_size; ++i) {
        result << std::setw(2) << static_cast<unsigned int>(digest[i]);
    }
    return result.str();
}

bool same_normalized_path(const std::string& a, const std::string& b) {
    std::error_code ec_a, ec_b;
    const fs::path absolute_a = fs::absolute(a, ec_a);
    const fs::path absolute_b = fs::absolute(b, ec_b);
    if (ec_a || ec_b) return false;
    const fs::path path_a = fs::weakly_canonical(absolute_a, ec_a);
    const fs::path path_b = fs::weakly_canonical(absolute_b, ec_b);
    return !ec_a && !ec_b && path_a == path_b;
}

bool has_authorization(const std::string& request, const std::string& token) {
    const size_t headers_end = request.find("\r\n\r\n");
    if (headers_end == std::string::npos) return false;
    size_t line_start = request.find("\r\n");
    if (line_start == std::string::npos || line_start >= headers_end) return false;
    line_start += 2;

    while (line_start < headers_end) {
        const size_t line_end = request.find("\r\n", line_start);
        if (line_end == std::string::npos || line_end > headers_end) break;
        const size_t colon = request.find(':', line_start);
        if (colon != std::string::npos && colon < line_end) {
            std::string name = request.substr(line_start, colon - line_start);
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });
            if (name == "authorization") {
                size_t value_start = colon + 1;
                while (value_start < line_end && request[value_start] == ' ') ++value_start;
                const std::string value = request.substr(value_start, line_end - value_start);
                return value == "Bearer " + token;
            }
        }
        line_start = line_end + 2;
    }
    return false;
}

} // namespace

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
    check_expired_leases_locked();
    for (auto& task : tasks_) {
        if (task.shard_id == shard_id) {
            if (task.state == TaskState::Completed) {
                return {{"status", "ok"}, {"already_completed", true}};
            }
            if (task.state != TaskState::Leased || task.worker_id != worker_id) {
                return {{"status", "error"}, {"message", "Task not leased to this worker"}};
            }
            if (output_path.empty() || !same_normalized_path(task.output_path, output_path)) {
                return {{"status", "error"}, {"message", "Reported output path does not match shard"}};
            }
            std::string reported_checksum = checksum;
            std::transform(reported_checksum.begin(), reported_checksum.end(),
                           reported_checksum.begin(), [](unsigned char ch) {
                               return static_cast<char>(std::tolower(ch));
                           });
            const std::string actual_checksum = sha256_file(task.output_path);
            if (actual_checksum.empty() || reported_checksum != actual_checksum) {
                return {{"status", "error"}, {"message", "Shard output checksum mismatch"}};
            }
            task.state = TaskState::Completed;
            task.checksum = actual_checksum;
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
    check_expired_leases_locked();
    for (auto& task : tasks_) {
        if (task.shard_id == shard_id) {
            if (task.state != TaskState::Leased || task.worker_id != worker_id) {
                return {{"status", "error"}, {"message", "Task not leased to this worker"}};
            }
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

    in_addr listen_addr{};
    if (inet_pton(AF_INET, options_.host.c_str(), &listen_addr) != 1) {
        throw OutputError("Coordinator host must be an IPv4 address: " + options_.host);
    }
    const bool loopback = (ntohl(listen_addr.s_addr) >> 24) == 127;
    if (!loopback && options_.auth_token.empty()) {
        throw OutputError("binding the coordinator beyond loopback requires "
                          "STRIKECEM_COORDINATOR_TOKEN");
    }

    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        throw OutputError("Failed to create coordinator server socket");
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(options_.port);
    addr.sin_addr = listen_addr;

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
    running_.store(false);

    if (server_fd_ >= 0) {
        shutdown(server_fd_, SHUT_RDWR);
        ::close(server_fd_);
    }

    if (server_thread_ && server_thread_->joinable()) {
        server_thread_->join();
        server_thread_.reset();
    }
    server_fd_ = -1;

    std::unique_lock<std::mutex> lock(clients_mutex_);
    clients_cv_.wait(lock, [this]() { return active_clients_ == 0; });
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
        const timeval timeout{5, 0};
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            ++active_clients_;
        }
        try {
            std::thread([this, client_fd]() {
                try {
                    handle_client(client_fd);
                } catch (...) {
                    ::close(client_fd);
                }
                {
                    std::lock_guard<std::mutex> lock(clients_mutex_);
                    --active_clients_;
                }
                clients_cv_.notify_all();
            }).detach();
        } catch (...) {
            ::close(client_fd);
            {
                std::lock_guard<std::mutex> lock(clients_mutex_);
                --active_clients_;
            }
            clients_cv_.notify_all();
        }
    }
}

void CoordinatorServer::handle_client(int client_fd) {
    std::string raw_req;
    char buf[4096];
    ssize_t bytes_read = 0;
    bool request_complete = false;
    bool has_content_length = false;
    size_t expected_body_len = 0;

    while ((bytes_read = recv(client_fd, buf, sizeof(buf), 0)) > 0) {
        raw_req.append(buf, bytes_read);
        constexpr size_t max_request_bytes = 1024 * 1024;
        if (raw_req.size() > max_request_bytes) {
            ::close(client_fd);
            return;
        }
        if (raw_req.find("\r\n\r\n") != std::string::npos) {
            // Check if Content-Length body has arrived
            size_t header_end = raw_req.find("\r\n\r\n");
            size_t cl_pos = raw_req.find("Content-Length:");
            if (cl_pos == std::string::npos) cl_pos = raw_req.find("content-length:");
            if (cl_pos != std::string::npos && cl_pos < header_end) {
                size_t num_start = raw_req.find(':', cl_pos);
                if (num_start != std::string::npos) ++num_start;
                while (num_start < header_end &&
                       (raw_req[num_start] == ' ' || raw_req[num_start] == '\t')) {
                    ++num_start;
                }
                if (num_start == std::string::npos || num_start >= header_end) {
                    ::close(client_fd);
                    return;
                }
                size_t num_end = raw_req.find("\r\n", num_start);
                if (num_end == std::string::npos || num_end > header_end) num_end = header_end;
                uint64_t content_len = 0;
                const char* begin = raw_req.data() + num_start;
                const char* end = raw_req.data() + num_end;
                const auto parsed = std::from_chars(begin, end, content_len);
                if (parsed.ec != std::errc{} || parsed.ptr != end ||
                    header_end + 4 > max_request_bytes ||
                    content_len > max_request_bytes - (header_end + 4)) {
                    ::close(client_fd);
                    return;
                }
                has_content_length = true;
                expected_body_len = static_cast<size_t>(content_len);
                size_t body_len = raw_req.size() - (header_end + 4);
                if (body_len < content_len) continue;
            }
            request_complete = true;
            break;
        }
    }

    if (raw_req.empty() || !request_complete) {
        ::close(client_fd);
        return;
    }

    std::istringstream req_stream(raw_req);
    std::string method, path, http_ver;
    req_stream >> method >> path >> http_ver;

    std::string body;
    size_t body_pos = raw_req.find("\r\n\r\n");
    if (body_pos != std::string::npos && has_content_length) {
        body = raw_req.substr(body_pos + 4, expected_body_len);
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

    if (!options_.auth_token.empty() && !has_authorization(raw_req, options_.auth_token)) {
        status_code = 401;
        res_json = {{"error", "Unauthorized"}};
    } else if (path == "/api/v1/health") {
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

    if (status_code == 200 && res_json.is_object() &&
        res_json.value("status", std::string()) == "error") {
        status_code = 400;
    }

    const std::string res_body = res_json.dump();
    std::ostringstream res_stream;
    const char* reason = status_code == 200 ? "OK" :
                         status_code == 401 ? "Unauthorized" :
                         status_code == 404 ? "Not Found" : "Bad Request";
    res_stream << "HTTP/1.1 " << status_code << " " << reason << "\r\n"
               << "Content-Type: application/json\r\n"
               << "Content-Length: " << res_body.size() << "\r\n"
               << "Connection: close\r\n\r\n"
               << res_body;

    const std::string res_str = res_stream.str();
    send(client_fd, res_str.data(), res_str.size(), 0);
    ::close(client_fd);
}

} // namespace strikecem

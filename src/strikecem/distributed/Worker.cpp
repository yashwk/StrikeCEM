#include "strikecem/distributed/Worker.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <netdb.h>
#include <netinet/in.h>
#include <openssl/evp.h>
#include <sstream>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include "strikecem/app/CLI.hpp"
#include "strikecem/io/OutputError.hpp"

namespace fs = std::filesystem;

namespace strikecem {

WorkerDaemon::WorkerDaemon(const WorkerOptions& options)
    : options_(options) {
    parse_url(options_.coordinator_url, host_, port_);
    if (options_.worker_id.empty()) {
        char hostname[256] = {0};
        gethostname(hostname, sizeof(hostname) - 1);
        std::ostringstream ss;
        ss << "worker_" << hostname << "_" << ::getpid() << "_"
           << (std::chrono::steady_clock::now().time_since_epoch().count() % 100000);
        worker_id_ = ss.str();
    } else {
        worker_id_ = options_.worker_id;
    }
}

WorkerDaemon::~WorkerDaemon() {
    stop();
}

void WorkerDaemon::stop() {
    running_.store(false);
}

void WorkerDaemon::parse_url(const std::string& url, std::string& out_host, int& out_port) {
    std::string s = url;
    if (s.rfind("http://", 0) == 0) {
        s = s.substr(7);
    }
    size_t slash_pos = s.find('/');
    if (slash_pos != std::string::npos) {
        s = s.substr(0, slash_pos);
    }
    size_t colon_pos = s.find(':');
    if (colon_pos != std::string::npos) {
        out_host = s.substr(0, colon_pos);
        out_port = std::stoi(s.substr(colon_pos + 1));
    } else {
        out_host = s;
        out_port = 8080;
    }
    if (out_host.empty()) out_host = "127.0.0.1";
}

std::string WorkerDaemon::compute_sha256(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) return "";

    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    if (!mdctx) return "";
    EVP_DigestInit_ex(mdctx, EVP_sha256(), nullptr);

    char buf[65536];
    while (file.read(buf, sizeof(buf)) || file.gcount() > 0) {
        EVP_DigestUpdate(mdctx, buf, file.gcount());
    }

    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len = 0;
    EVP_DigestFinal_ex(mdctx, hash, &hash_len);
    EVP_MD_CTX_free(mdctx);

    std::ostringstream ss;
    for (unsigned int i = 0; i < hash_len; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return ss.str();
}

nlohmann::json WorkerDaemon::send_request(const std::string& method, const std::string& path,
                                         const nlohmann::json& body) {
    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        throw OutputError("Worker socket creation failed");
    }

    struct timeval tv{};
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    setsockopt(sock_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port_);

    if (inet_pton(AF_INET, host_.c_str(), &server_addr.sin_addr) <= 0) {
        // Resolve hostname via gethostbyname
        hostent* he = gethostbyname(host_.c_str());
        if (!he) {
            ::close(sock_fd);
            throw OutputError("Failed to resolve coordinator host: " + host_);
        }
        std::memcpy(&server_addr.sin_addr, he->h_addr_list[0], he->h_length);
    }

    if (connect(sock_fd, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) < 0) {
        ::close(sock_fd);
        throw OutputError("Cannot connect to coordinator at " + host_ + ":" + std::to_string(port_));
    }

    const std::string body_str = body.is_null() ? "" : body.dump();
    std::ostringstream req_stream;
    req_stream << method << " " << path << " HTTP/1.1\r\n"
               << "Host: " << host_ << ":" << port_ << "\r\n"
               << "User-Agent: StrikeCEM-Worker/" << worker_id_ << "\r\n"
               << "Content-Type: application/json\r\n"
               << "Content-Length: " << body_str.size() << "\r\n"
               << "Connection: close\r\n\r\n"
               << body_str;

    const std::string req_str = req_stream.str();
    send(sock_fd, req_str.data(), req_str.size(), 0);

    std::string raw_res;
    char recv_buf[4096];
    ssize_t bytes = 0;
    while ((bytes = recv(sock_fd, recv_buf, sizeof(recv_buf), 0)) > 0) {
        raw_res.append(recv_buf, bytes);
    }
    ::close(sock_fd);

    size_t header_end = raw_res.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        throw OutputError("Invalid HTTP response from coordinator");
    }

    std::string res_body = raw_res.substr(header_end + 4);
    if (res_body.empty()) {
        return nlohmann::json::object();
    }

    try {
        return nlohmann::json::parse(res_body);
    } catch (const std::exception& e) {
        throw OutputError("Failed to parse coordinator JSON response: " + std::string(e.what()));
    }
}

WorkerResult WorkerDaemon::run() {
    running_.store(true);
    WorkerResult res;
    res.success = true;

    while (running_.load()) {
        nlohmann::json acquire_req = {
            {"worker_id", worker_id_},
            {"device", options_.device}
        };

        nlohmann::json task_res;
        try {
            task_res = send_request("POST", "/api/v1/task/acquire", acquire_req);
        } catch (const std::exception& e) {
            // Coordinator temporarily unreachable; back off and retry
            std::this_thread::sleep_for(std::chrono::milliseconds(options_.poll_interval_ms));
            continue;
        }

        const std::string status = task_res.value("status", "");
        if (status == "campaign_complete" || task_res.value("all_complete", false)) {
            // Entire distributed campaign is finished!
            break;
        }

        if (status == "none_available") {
            // All active tasks are leased or none pending; wait briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(options_.poll_interval_ms));
            continue;
        }

        if (status == "acquired") {
            const size_t shard_id = task_res["shard_id"].get<size_t>();
            const std::string cfg_path = task_res["config_path"].get<std::string>();
            const std::string out_path = task_res["output_path"].get<std::string>();
            const int lease_sec = task_res.value("lease_duration_seconds", 30);

            // Heartbeat thread to keep lease alive during execution
            std::atomic<bool> task_running{true};
            std::mutex cv_mtx;
            std::condition_variable cv;
            std::thread heartbeat_thread([this, shard_id, lease_sec, &task_running, &cv_mtx, &cv]() {
                const int hb_interval = std::max(1, lease_sec / 3);
                while (task_running.load()) {
                    std::unique_lock<std::mutex> lock(cv_mtx);
                    if (cv.wait_for(lock, std::chrono::seconds(hb_interval), [&]() { return !task_running.load(); })) {
                        break;
                    }
                    if (!task_running.load()) break;
                    try {
                        send_request("POST", "/api/v1/task/heartbeat", {
                            {"shard_id", shard_id},
                            {"worker_id", worker_id_}
                        });
                    } catch (...) {
                        // Suppress transient heartbeat network glitches
                    }
                }
            });

            // Execute the solver for this shard
            bool solve_ok = false;
            std::string solve_err;
            try {
                // If device override requested, adjust shard config
                if (!options_.device.empty() && options_.device != "default") {
                    std::ifstream in(cfg_path);
                    if (in) {
                        nlohmann::json j;
                        in >> j;
                        in.close();
                        if (options_.device.rfind("cuda:", 0) == 0) {
                            j["execution"]["accelerator"] = "cuda";
                            j["execution"]["cuda_device_id"] = std::stoi(options_.device.substr(5));
                        } else if (options_.device == "cuda") {
                            j["execution"]["accelerator"] = "cuda";
                            j["execution"]["cuda_device_id"] = 0;
                        } else {
                            j["execution"]["accelerator"] = "cpu";
                        }
                        std::ofstream out(cfg_path);
                        out << j.dump(2) << "\n";
                    }
                }

                std::vector<std::string> args = {"strikecem", "run", cfg_path};
                if (!options_.schema_path.empty()) {
                    args.push_back("--schema");
                    args.push_back(options_.schema_path);
                }
                std::vector<char*> c_argv;
                for (auto& a : args) c_argv.push_back(a.data());

                const int exit_code = cli::run(static_cast<int>(c_argv.size()), c_argv.data());
                if (exit_code != 0) {
                    throw OutputError("CLI run returned error code " + std::to_string(exit_code));
                }
                solve_ok = true;
            } catch (const std::exception& e) {
                solve_ok = false;
                solve_err = e.what();
            }

            // Stop heartbeat thread
            task_running.store(false);
            cv.notify_all();
            if (heartbeat_thread.joinable()) {
                heartbeat_thread.join();
            }

            if (solve_ok) {
                const std::string checksum = compute_sha256(out_path);
                try {
                    send_request("POST", "/api/v1/task/complete", {
                        {"shard_id", shard_id},
                        {"worker_id", worker_id_},
                        {"output_path", out_path},
                        {"checksum", checksum}
                    });
                    res.tasks_completed++;
                } catch (const std::exception& e) {
                    res.error_message = "Failed to report task completion: " + std::string(e.what());
                }
            } else {
                try {
                    send_request("POST", "/api/v1/task/fail", {
                        {"shard_id", shard_id},
                        {"worker_id", worker_id_},
                        {"error", solve_err}
                    });
                } catch (...) {}
            }

            if (options_.max_tasks > 0 && res.tasks_completed >= options_.max_tasks) {
                break;
            }
        }
    }

    return res;
}

} // namespace strikecem

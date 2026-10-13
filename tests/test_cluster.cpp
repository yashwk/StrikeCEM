// Unit and integration tests for Domain 4:
// Local Parallel Runner, Distributed Coordinator Broker, Headless Worker Daemon,
// and Cluster Scheduler (Slurm & Kubernetes) Generation.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <gtest/gtest.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "strikecem/app/CLI.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/distributed/ClusterScheduler.hpp"
#include "strikecem/distributed/Coordinator.hpp"
#include "strikecem/distributed/LocalParallelRunner.hpp"
#include "strikecem/distributed/Shard.hpp"
#include "strikecem/distributed/Worker.hpp"
#include "strikecem/io/Hdf5Reader.hpp"
#include "strikecem/io/OutputError.hpp"

namespace {

namespace fs = std::filesystem;

struct Workdir {
    fs::path dir = fs::temp_directory_path() / ("scem_cluster_test_" + std::to_string(::getpid()) + "_" +
                                                std::to_string(reinterpret_cast<uintptr_t>(this)));
    Workdir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
        fs::create_directories(dir, ec);
    }
    ~Workdir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

int call_cli(const std::vector<std::string>& arg_list) {
    std::vector<std::string> storage = {"strikecem"};
    for (const auto& a : arg_list) storage.push_back(a);
    std::vector<char*> argv;
    for (auto& s : storage) argv.push_back(s.data());
    return strikecem::cli::run(static_cast<int>(argv.size()), argv.data());
}

nlohmann::json make_test_config(const fs::path& out_path, size_t n_az = 6, size_t n_freq = 1) {
    nlohmann::json cfg;
    cfg["scem_schema_version"] = "1.0";
    cfg["model"] = {{"path", std::string(std::string(SCEM_FIXTURE_DIR) +
                                         "/../../examples/plate.stl")}};
    if (n_freq == 1) {
        cfg["frequency"] = {{"frequency_hz", 10e9}};
    } else {
        cfg["frequency"] = {{"sweep", {{"start", 8e9}, {"stop", 10e9}, {"step", 1e9}}}};
    }
    cfg["angles"]["azimuth"] = {{"start", 0}, {"stop", static_cast<double>((n_az - 1) * 30)}, {"step", 30}};
    cfg["angles"]["elevation"] = {{"start", 0}, {"stop", 0}, {"step", 1}};
    cfg["polarization"] = {"HH", "VV"};
    cfg["solver"] = {{"type", "PO"}};
    cfg["run"] = {{"checkpoint_every_n_samples", 2}};
    cfg["output"] = {{"path", out_path.string()}, {"format", "hdf5"}};
    return cfg;
}

fs::path write_json(const fs::path& path, const nlohmann::json& j) {
    std::ofstream out(path);
    out << j.dump(2);
    return path;
}

} // namespace

TEST(LocalParallelRunner, DirectVersusParallelExecutionMatch) {
    Workdir w;
    const fs::path mono_out = w.dir / "monolithic.h5";
    const fs::path parallel_out = w.dir / "parallel.h5";
    const fs::path mono_cfg = write_json(w.dir / "mono_config.json", make_test_config(mono_out, 6, 1));
    const fs::path par_cfg = write_json(w.dir / "par_config.json", make_test_config(parallel_out, 6, 1));

    // 1. Run monolithic execution
    ASSERT_EQ(call_cli({"run", mono_cfg.string()}), 0);
    ASSERT_TRUE(fs::exists(mono_out));

    // 2. Run local parallel execution with 3 workers
    strikecem::LocalParallelOptions opts;
    opts.config_path = par_cfg.string();
    opts.output_path = parallel_out.string();
    opts.num_workers = 3;
    opts.work_dir = (w.dir / "parallel_staging").string();
    opts.keep_shards = false;

    const auto par_res = strikecem::run_local_parallel(opts, SCEM_SCHEMA_PATH);
    ASSERT_TRUE(par_res.success) << par_res.error_message;
    EXPECT_EQ(par_res.num_shards, 3u);
    EXPECT_EQ(par_res.completed_shards, 3u);
    ASSERT_TRUE(fs::exists(parallel_out));

    // 3. Verify both outputs are bit-identical
    const auto mono_db = strikecem::read_hdf5_database(mono_out.string());
    const auto par_db = strikecem::read_hdf5_database(parallel_out.string());
    strikecem::validate_hdf5_complete(mono_db);
    strikecem::validate_hdf5_complete(par_db);

    ASSERT_EQ(mono_db.rows.size(), par_db.rows.size());
    EXPECT_EQ(mono_db.rows.size(), 12u); // 6 directions * 2 polarizations

    for (size_t i = 0; i < mono_db.rows.size(); ++i) {
        EXPECT_EQ(mono_db.rows[i].sample_id, par_db.rows[i].sample_id);
        EXPECT_EQ(mono_db.rows[i].direction_index, par_db.rows[i].direction_index);
        EXPECT_EQ(mono_db.rows[i].frequency_index, par_db.rows[i].frequency_index);
        EXPECT_EQ(mono_db.rows[i].polarization_index, par_db.rows[i].polarization_index);
        EXPECT_DOUBLE_EQ(mono_db.rows[i].rcs_sqm, par_db.rows[i].rcs_sqm);
        EXPECT_DOUBLE_EQ(mono_db.rows[i].scattering_real, par_db.rows[i].scattering_real);
        EXPECT_DOUBLE_EQ(mono_db.rows[i].scattering_imag, par_db.rows[i].scattering_imag);
    }
}

TEST(LocalParallelRunner, CliCommandDispatch) {
    Workdir w;
    const fs::path parallel_out = w.dir / "cli_parallel.h5";
    const fs::path par_cfg = write_json(w.dir / "par_config.json", make_test_config(parallel_out, 4, 1));

    const int rc = call_cli({"run-parallel", par_cfg.string(),
                            "--output", parallel_out.string(),
                            "--workers", "2"});
    ASSERT_EQ(rc, 0);
    ASSERT_TRUE(fs::exists(parallel_out));
    const auto db = strikecem::read_hdf5_database(parallel_out.string());
    EXPECT_EQ(db.rows.size(), 8u);
}

TEST(Coordinator, HttpRestEndpoints) {
    Workdir w;
    const fs::path dummy_out = w.dir / "final.h5";
    const fs::path cfg_file = write_json(w.dir / "config.json", make_test_config(dummy_out, 4, 1));
    const fs::path shards_dir = w.dir / "shards";
    ASSERT_EQ(call_cli({"shard", "--num-shards", "2", "--out-dir", shards_dir.string(), cfg_file.string()}), 0);
    const fs::path manifest_path = shards_dir / "shard_manifest.json";

    strikecem::CoordinatorOptions opts;
    opts.manifest_path = manifest_path.string();
    opts.output_path = dummy_out.string();
    opts.port = 0; // Ephemeral port
    opts.lease_duration_seconds = 60;
    opts.auto_merge = false;

    strikecem::CoordinatorServer server(opts);
    server.start(true);
    const int port = server.bound_port();
    ASSERT_GT(port, 0);

    strikecem::WorkerOptions wopts;
    wopts.coordinator_url = "http://127.0.0.1:" + std::to_string(port);
    wopts.worker_id = "test_worker_1";
    strikecem::WorkerDaemon client(wopts);

    // 1. GET /api/v1/health
    auto health_res = client.send_request("GET", "/api/v1/health");
    EXPECT_EQ(health_res.value("status", ""), "ok");
    EXPECT_EQ(health_res.value("version", ""), "1.0");

    // 2. GET /api/v1/status
    auto status_res = client.send_request("GET", "/api/v1/status");
    EXPECT_EQ(status_res.value("total_shards", 0), 2);
    EXPECT_EQ(status_res.value("pending_shards", 0), 2);
    EXPECT_EQ(status_res.value("completed_shards", 0), 0);

    // 3. POST /api/v1/task/acquire
    auto acq_res = client.send_request("POST", "/api/v1/task/acquire", {
        {"worker_id", "test_worker_1"},
        {"device", "cpu"}
    });
    EXPECT_EQ(acq_res.value("status", ""), "acquired");
    EXPECT_EQ(acq_res.value("shard_id", 999), 0);

    // 4. POST /api/v1/task/heartbeat
    auto hb_res = client.send_request("POST", "/api/v1/task/heartbeat", {
        {"shard_id", 0},
        {"worker_id", "test_worker_1"}
    });
    EXPECT_EQ(hb_res.value("status", ""), "ok");
    EXPECT_TRUE(hb_res.value("lease_renewed", false));

    // 5. POST /api/v1/task/complete
    auto comp_res = client.send_request("POST", "/api/v1/task/complete", {
        {"shard_id", 0},
        {"worker_id", "test_worker_1"},
        {"output_path", (shards_dir / "shard_0000.h5").string()},
        {"checksum", "deadbeef"}
    });
    EXPECT_EQ(comp_res.value("status", ""), "ok");
    EXPECT_EQ(comp_res.value("completed_count", 0), 1);

    // 6. Check updated status
    status_res = client.send_request("GET", "/api/v1/status");
    EXPECT_EQ(status_res.value("completed_shards", 0), 1);
    EXPECT_EQ(status_res.value("pending_shards", 0), 1);

    server.stop();
}

TEST(Coordinator, WorkerHeartbeatAndLeaseExpiry) {
    Workdir w;
    const fs::path dummy_out = w.dir / "final.h5";
    const fs::path cfg_file = write_json(w.dir / "config.json", make_test_config(dummy_out, 2, 1));
    const fs::path shards_dir = w.dir / "shards";
    ASSERT_EQ(call_cli({"shard", "--num-shards", "1", "--out-dir", shards_dir.string(), cfg_file.string()}), 0);
    const fs::path manifest_path = shards_dir / "shard_manifest.json";

    strikecem::CoordinatorOptions opts;
    opts.manifest_path = manifest_path.string();
    opts.port = 0;
    opts.lease_duration_seconds = 1; // 1-second lease for rapid expiration test
    opts.auto_merge = false;

    strikecem::CoordinatorServer server(opts);
    server.start(true);
    const int port = server.bound_port();

    strikecem::WorkerOptions wopts;
    wopts.coordinator_url = "http://127.0.0.1:" + std::to_string(port);
    strikecem::WorkerDaemon client(wopts);

    // Lease shard 0 to worker A
    auto res1 = client.send_request("POST", "/api/v1/task/acquire", {
        {"worker_id", "worker_A"},
        {"device", "cpu"}
    });
    EXPECT_EQ(res1.value("status", ""), "acquired");

    // Immediately try to lease again -> none available
    auto res2 = client.send_request("POST", "/api/v1/task/acquire", {
        {"worker_id", "worker_B"},
        {"device", "cpu"}
    });
    EXPECT_EQ(res2.value("status", ""), "none_available");

    // Sleep 1.2s so the lease expires
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));

    // Worker B should now successfully acquire the expired shard
    auto res3 = client.send_request("POST", "/api/v1/task/acquire", {
        {"worker_id", "worker_B"},
        {"device", "cpu"}
    });
    EXPECT_EQ(res3.value("status", ""), "acquired");
    EXPECT_EQ(res3.value("shard_id", 999), 0);

    server.stop();
}

TEST(Worker, AutonomousExecutionLoop) {
    Workdir w;
    const fs::path final_out = w.dir / "distributed_final.h5";
    const fs::path cfg_file = write_json(w.dir / "config.json", make_test_config(final_out, 6, 1));
    const fs::path shards_dir = w.dir / "shards_campaign";
    ASSERT_EQ(call_cli({"shard", "--num-shards", "3", "--out-dir", shards_dir.string(), cfg_file.string()}), 0);
    const fs::path manifest_path = shards_dir / "shard_manifest.json";

    strikecem::CoordinatorOptions copts;
    copts.manifest_path = manifest_path.string();
    copts.output_path = final_out.string();
    copts.port = 0;
    copts.lease_duration_seconds = 30;
    copts.auto_merge = true;
    copts.schema_path = SCEM_SCHEMA_PATH;

    strikecem::CoordinatorServer coordinator(copts);
    coordinator.start(true);
    const int port = coordinator.bound_port();
    const std::string coord_url = "http://127.0.0.1:" + std::to_string(port);

    // Launch 2 workers concurrently to consume shards
    std::vector<std::thread> workers;
    std::vector<strikecem::WorkerResult> results(2);

    for (size_t i = 0; i < 2; ++i) {
        workers.emplace_back([i, coord_url, &results]() {
            strikecem::WorkerOptions wopts;
            wopts.coordinator_url = coord_url;
            wopts.worker_id = "cluster_daemon_" + std::to_string(i);
            wopts.schema_path = SCEM_SCHEMA_PATH;
            wopts.poll_interval_ms = 50;

            strikecem::WorkerDaemon daemon(wopts);
            results[i] = daemon.run();
        });
    }

    for (auto& th : workers) {
        if (th.joinable()) th.join();
    }

    EXPECT_TRUE(results[0].success);
    EXPECT_TRUE(results[1].success);
    EXPECT_EQ(results[0].tasks_completed + results[1].tasks_completed, 3u);

    // Wait briefly for coordinator auto-merge to settle
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    EXPECT_TRUE(coordinator.is_campaign_complete());
    EXPECT_TRUE(fs::exists(final_out));

    const auto db = strikecem::read_hdf5_database(final_out.string());
    strikecem::validate_hdf5_complete(db);
    EXPECT_EQ(db.rows.size(), 12u);

    coordinator.stop();
}

TEST(ClusterScheduler, SlurmAndK8sGeneration) {
    Workdir w;
    const fs::path dummy_out = w.dir / "dummy.h5";
    const fs::path cfg_file = write_json(w.dir / "config.json", make_test_config(dummy_out, 8, 1));
    const fs::path shards_dir = w.dir / "shards_sched";
    ASSERT_EQ(call_cli({"shard", "--num-shards", "4", "--out-dir", shards_dir.string(), cfg_file.string()}), 0);
    const fs::path manifest_path = shards_dir / "shard_manifest.json";

    // 1. Test Slurm generation via API
    const fs::path slurm_out = w.dir / "job.slurm";
    strikecem::SlurmOptions sopts;
    sopts.manifest_path = manifest_path.string();
    sopts.output_path = slurm_out.string();
    sopts.partition = "accelerated";
    sopts.cpus_per_task = 8;
    sopts.time_limit = "02:30:00";
    sopts.job_name = "test_slurm_campaign";
    strikecem::generate_slurm_script(sopts);

    ASSERT_TRUE(fs::exists(slurm_out));
    std::ifstream s_in(slurm_out);
    std::string s_content((std::istreambuf_iterator<char>(s_in)), std::istreambuf_iterator<char>());
    EXPECT_NE(s_content.find("#SBATCH --job-name=test_slurm_campaign"), std::string::npos);
    EXPECT_NE(s_content.find("#SBATCH --array=0-3"), std::string::npos);
    EXPECT_NE(s_content.find("#SBATCH --partition=accelerated"), std::string::npos);
    EXPECT_NE(s_content.find("#SBATCH --cpus-per-task=8"), std::string::npos);
    EXPECT_NE(s_content.find("#SBATCH --time=02:30:00"), std::string::npos);

    // 2. Test Kubernetes generation via CLI
    const fs::path k8s_out = w.dir / "job.yaml";
    const int rc = call_cli({"cluster", "generate-k8s",
                            "--manifest", manifest_path.string(),
                            "--output", k8s_out.string(),
                            "--image", "custom/strikecem:v2",
                            "--coordinator", "http://coordinator.cluster.local:8080",
                            "--parallelism", "4"});
    ASSERT_EQ(rc, 0);
    ASSERT_TRUE(fs::exists(k8s_out));

    std::ifstream k_in(k8s_out);
    std::string k_content((std::istreambuf_iterator<char>(k_in)), std::istreambuf_iterator<char>());
    EXPECT_NE(k_content.find("kind: Job"), std::string::npos);
    EXPECT_NE(k_content.find("parallelism: 4"), std::string::npos);
    EXPECT_NE(k_content.find("completions: 4"), std::string::npos);
    EXPECT_NE(k_content.find("custom/strikecem:v2"), std::string::npos);
    EXPECT_NE(k_content.find("http://coordinator.cluster.local:8080"), std::string::npos);
}

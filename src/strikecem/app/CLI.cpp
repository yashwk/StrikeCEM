// v1 CLI: validate, estimate, and run the configured PO pipeline.
#include "strikecem/app/CLI.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/adaptive/AdaptiveSampling.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/diagnostics/VtkWriter.hpp"
#include "strikecem/distributed/Shard.hpp"
#include "strikecem/distributed/LocalParallelRunner.hpp"
#include "strikecem/distributed/Coordinator.hpp"
#include "strikecem/distributed/Worker.hpp"
#include "strikecem/distributed/ClusterScheduler.hpp"
#include "strikecem/solvers/Similitude.hpp"
#include "strikecem/analysis/VehiclePreScan.hpp"
#include "strikecem/analysis/IsarImaging.hpp"
#include "strikecem/rendering/Image.hpp"
#include "strikecem/rendering/ModelRenderer.hpp"
#include "strikecem/analysis/SignatureReport.hpp"
#include "strikecem/analysis/SarImaging.hpp"
#include "strikecem/io/AssetBundle.hpp"
#include "strikecem/io/BinaryTransport.hpp"
#include "strikecem/io/CsvWriter.hpp"
#include "strikecem/io/DesignerPackage.hpp"
#include "strikecem/io/Hdf5Writer.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/runtime/Estimate.hpp"
#include "strikecem/solvers/GpuPO.hpp"
#include "strikecem/solvers/Material.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"
#include "strikecem/telemetry/Telemetry.hpp"

namespace strikecem::cli {
namespace {

namespace fs = std::filesystem;

#ifndef SCEM_SCHEMA_INSTALL_RELATIVE_PATH
#define SCEM_SCHEMA_INSTALL_RELATIVE_PATH "../share/strikecem/schemas/config.schema.json"
#endif
#ifndef SCEM_SOURCE_SCHEMA_PATH
#define SCEM_SOURCE_SCHEMA_PATH ""
#endif

constexpr const char* kUsage =
    "usage: strikecem [--schema PATH] [--bench-profile PATH] <command> [options...] <args...>\n"
    "  validate     resolve config, inspect mesh, print hashes\n"
    "  estimate     validate plus plan counts and resource estimates\n"
    "  run          full pipeline (PO solver, CSV/HDF5 output) [--scale S]\n"
    "  shard        partition sample plan across N shards and write shard configs + manifest\n"
    "               options: --num-shards N (default 2), --out-dir DIR (default ./shards), --run-id ID\n"
    "  merge        merge completed shard outputs into a unified database\n"
    "               options: --output PATH <manifest.json | shard1 shard2 ...>\n"
    "  vtk          export 3D per-facet currents and lit/shadow diagnostics to VTK (.vtu)\n"
    "               options: --output PATH, --direction-id N (default 0)\n"
    "  export-mesh  export procedural or normalized mesh to standalone format\n"
    "               options: --output PATH (-o), --format <obj|stl|vtu>\n"
    "  bundle       package RCS database and geometry into StrikeEngine bundle (.scempack)\n"
    "               options: --rcs PATH, --mesh PATH, --output PATH (-o), --lod NAME=PATH\n"
    "  prescan      analyze 3D geometry features and recommend/tune hybrid solver settings\n"
    "               options: <config.json|mesh> [--report PATH] [--auto-tune-config PATH]\n"
    "  run-parallel execute local campaign across multiple CPU/GPU devices\n"
    "               options: <config.json> [-o PATH] [--workers N] [--devices d1,d2] [--work-dir DIR] [--keep-shards]\n"
    "  coordinate   run distributed cluster campaign coordinator HTTP REST broker\n"
    "               options: --manifest PATH [-o PATH] [--port P] [--host H] [--lease SEC] [--auto-exit]\n"
    "  worker       run headless cluster worker daemon connecting to coordinator\n"
    "               options: --coordinator URL [--device DEV] [--worker-id ID] [--poll-ms MS] [--max-tasks N]\n"
    "  cluster      generate HPC/cloud job scripts:\n"
    "               generate-slurm --manifest PATH --output PATH [--partition P] [--cpus N] [--time T]\n"
    "               generate-k8s   --manifest PATH --output PATH [--image IMG] [--coordinator URL] [--parallelism N]\n"
    "  isar         generate 2D ISAR radar image (range-Doppler) from frequency-azimuth sweep\n"
    "               options: <database.h5> [--output isar.json] [--image isar.png] [--ascii]\n"
    "                        [--pol POL] [--dyn-range DB] [--colormap turbo|jet|viridis]\n"
    "                        [--range-samples N] [--cross-range-samples M] [--no-window] [--elevation DEG]\n"
    "  render       render 3D vehicle snapshot (surface currents, lit/shadow, or shaded CAD)\n"
    "               options: <config.json|mesh> [--output PATH] [--mode currents|lit_shadow|shaded]\n"
    "                        [--az DEG] [--el DEG] [--freq HZ] [--pol HH|VV] [--camera radar|isometric]\n"
    "                        [--colormap turbo|jet|viridis] [--width W] [--height H]\n"
    "  report       generate standalone HTML radar signature diagnostic report\n"
    "               options: <database.h5|config.json> [-o report.html] [--config CFG] [--mesh MESH]\n"
    "                        [--title \"...\"] [--vehicle \"...\"] [--image img.png] [--isar isar.png]\n"
    "                        [--mode currents|lit_shadow|shaded] [--colormap turbo|jet|viridis]\n"
    "  sar          simulate spaceborne or airborne Synthetic Aperture Radar (SAR) 2D imaging\n"
    "               options: <config.json|mesh> [--output sar.png] [--preset satellite-ka|satellite-x|airborne-ka|airborne-x]\n"
    "                        [--freq HZ] [--bandwidth HZ] [--incidence DEG] [--altitude M] [--velocity M/S]\n"
    "                        [--ground tarmac|soil|none] [--no-shadow] [--colormap gray|turbo|jet] [--json sar.json]\n"
    "  --bench-profile PATH (run only) writes a benchmark profile for the run\n";

int fail(const std::exception& e, ExitCode code) {
    std::cerr << "strikecem: error: " << e.what() << "\n";
    return static_cast<int>(code);
}

void print_warnings(const SamplePlan& plan) {
    for (const auto& w : plan.warnings) std::cerr << "strikecem: warning: " << w << "\n";
}

fs::path config_dir_of(const std::string& config_path) {
    const fs::path dir = fs::path(config_path).parent_path();
    return dir.empty() ? fs::current_path() : dir;
}

std::string default_schema_path(const char* argv0) {
    std::error_code ec;
    fs::path executable = fs::read_symlink("/proc/self/exe", ec);
    if (ec) {
        ec.clear();
        const fs::path invoked(argv0 ? argv0 : "");
        if (invoked.has_parent_path()) executable = fs::absolute(invoked, ec);
    }
    if (!executable.empty()) {
        const fs::path installed = executable.parent_path() / SCEM_SCHEMA_INSTALL_RELATIVE_PATH;
        if (fs::is_regular_file(installed, ec)) return installed.string();
    }
    if (SCEM_SOURCE_SCHEMA_PATH[0] != '\0' && fs::is_regular_file(SCEM_SOURCE_SCHEMA_PATH, ec))
        return SCEM_SOURCE_SCHEMA_PATH;
    const fs::path from_cwd = fs::current_path(ec) / "schemas/config.schema.json";
    if (!ec && fs::is_regular_file(from_cwd, ec)) return from_cwd.string();
    return "schemas/config.schema.json";
}

void print_mesh_report(const MeshReport& r) {
    std::cout << "vertices: " << r.vertex_count << "\n";
    std::cout << "triangles: " << r.triangle_count << "\n";
    std::cout << "bounds_min_m: [" << r.bbox_min.x << ", " << r.bbox_min.y << ", "
              << r.bbox_min.z << "]\n";
    std::cout << "bounds_max_m: [" << r.bbox_max.x << ", " << r.bbox_max.y << ", "
              << r.bbox_max.z << "]\n";
    std::cout << "surface_area_m2: " << r.total_area_m2 << "\n";
    std::cout << "degenerate_triangles: " << r.degenerate_count << "\n";
    std::cout << "open_edges: " << r.open_edge_count << "\n";
    std::cout << "nonmanifold_edges: " << r.nonmanifold_edge_count << "\n";
    std::cout << "inconsistent_winding_edges: " << r.inconsistent_winding_count << "\n";
    std::cout << "aspect_ratio_max: " << r.aspect_max << "\n";
    std::cout << "aspect_over_limit: " << r.aspect_over_limit_count << "\n";
}

int cmd_validate(const ResolvedConfig& rc, const std::string& config_path) {
    const DesignerIdentity designer = validate_designer_package(rc.value, config_path);
    for (const auto& w : designer.warnings) std::cerr << "strikecem: warning: " << w << "\n";
    NormalizedMesh mesh =
        load_normalized_mesh(rc.value, config_dir_of(config_path), rc.schema_version);
    const ResolvedMaterials resolved_mat = resolve_materials(rc.value, mesh);
    for (const auto& w : resolved_mat.warnings) std::cerr << "strikecem: warning: " << w << "\n";
    std::cout << "scem_schema_version: " << rc.schema_version << "\n";
    std::cout << "resolved_config_hash: " << rc.hash << "\n";
    std::cout << "model: " << rc.value["model"]["path"].get<std::string>() << "\n";
    if (designer.packaged) {
        std::cout << "package: export_manifest.json ok\n";
        std::cout << "design_id: " << designer.design_id << "\n";
        std::cout << "design_revision: " << designer.revision << "\n";
        if (!designer.export_id.empty()) std::cout << "export_id: " << designer.export_id << "\n";
        std::cout << "export_manifest_hash: " << designer.manifest_hash << "\n";
    }
    print_mesh_report(mesh.report);
    if (mesh.repaired) {
        std::cout << "repaired: yes (triangles " << mesh.report_before.triangle_count << " -> "
                  << mesh.report.triangle_count << ")\n";
    } else {
        std::cout << "repaired: no\n";
    }
    std::cout << "geometry_hash: " << mesh.geometry_hash << "\n";
    std::cout << "normalized_mesh_hash: " << mesh.normalized_mesh_hash << "\n";
    std::cout << "cache: " << (mesh.cache_hit ? "hit" : "miss") << "\n";
    std::cout << "status: ok\n";
    return 0;
}

int cmd_estimate(const ResolvedConfig& rc, const std::string& config_path,
                 const std::vector<BenchmarkProfile>& profiles) {
    SamplePlan plan = build_sample_plan(rc.value);
    print_warnings(plan);
    const int rc_code = cmd_validate(rc, config_path);
    if (rc_code != 0) return rc_code;
    std::cout << "frequencies: " << plan.frequencies_hz.size() << "\n";
    std::cout << "directions: " << plan.directions.size() << "\n";
    std::cout << "polarizations: " << plan.polarizations.size() << "\n";
    std::cout << "samples: " << plan.sample_count() << "\n";
    NormalizedMesh mesh =
        load_normalized_mesh(rc.value, config_dir_of(config_path), rc.schema_version);
    const ResourceEstimate est = estimate_resources(rc.value, plan, mesh, profiles);
    for (const auto& w : est.warnings) std::cerr << "strikecem: warning: " << w << "\n";
    std::cout << "electrical_size_wavelengths: " << est.electrical_size_wavelengths << "\n";
    std::cout << "estimated_mesh_mb: " << est.mesh_bytes / 1048576.0 << "\n";
    std::cout << "estimated_output_mb: " << est.output_bytes / 1048576.0 << "\n";
    std::cout << "estimated_total_mb: " << est.total_bytes / 1048576.0 << "\n";
    std::cout << "memory_limit_mb: " << est.limit_bytes / 1048576.0 << "\n";
    std::cout << "operations: " << est.operation_count << "\n";
    std::cout << "backend: " << est.backend << "\n";
    if (est.backend == "cuda") {
        std::cout << "estimated_device_mb: " << est.device_bytes / 1048576.0 << "\n";
        const uint64_t override_units =
            rc.value["execution"].value("cuda_batch_units", 0u);
        if (override_units > 0)
            std::cout << "cuda_batch_units: " << override_units << " (configured)\n";
        else
            std::cout << "cuda_batch_units: auto (selected at run from device memory)\n";
    }
    if (est.calibrated)
        std::cout << "estimated_runtime_s: [" << est.runtime_lo_s << ", " << est.runtime_hi_s
                  << "] (profile " << est.profile_id << ")\n";
    else
        std::cout << "estimated_runtime_s: uncalibrated (" << est.operation_count
                  << " operations, no matching benchmark profile)\n";
    return 0;
}

int cmd_shard(const ResolvedConfig& rc, const SamplePlan& plan,
              const NormalizedMesh& mesh, const std::string& config_path,
              size_t num_shards, const std::string& out_dir,
              const std::string& run_id) {
    ShardManifest manifest =
        create_shards(rc, plan, mesh, config_path, num_shards, out_dir, run_id);
    std::cout << "shards: " << manifest.num_shards << "\n";
    std::cout << "total_units: " << manifest.total_units << "\n";
    std::cout << "total_samples: " << manifest.total_samples << "\n";
    std::cout << "run_id: " << manifest.run_id << "\n";
    std::cout << "manifest: " << (fs::path(out_dir) / "shard_manifest.json").string() << "\n";
    std::cout << "status: ok\n";
    return 0;
}

int cmd_merge(const std::string& first_path, const std::vector<std::string>& additional,
              std::string output_path, const std::string& schema_path) {
    if (output_path.empty()) {
        if (fs::exists(first_path) && fs::path(first_path).extension() == ".json") {
            try {
                std::ifstream in(first_path);
                const auto j = nlohmann::json::parse(in);
                if (j.contains("config_path") && fs::exists(j["config_path"].get<std::string>())) {
                    std::ifstream cfg_in(j["config_path"].get<std::string>());
                    const auto cfg_j = nlohmann::json::parse(cfg_in);
                    if (cfg_j.contains("output") && cfg_j["output"].contains("path")) {
                        output_path = cfg_j["output"]["path"].get<std::string>();
                    }
                }
            } catch (...) {}
        }
        if (output_path.empty()) {
            output_path = "merged.h5";
        }
    }
    merge_shards(first_path, additional, output_path, schema_path);
    std::cout << "merged_output: " << output_path << "\n";
    std::cout << "status: ok\n";
    return 0;
}

int cmd_vtk(const ResolvedConfig& rc, const SamplePlan& plan,
            const NormalizedMesh& mesh, const std::string& vtu_path,
            size_t direction_id, const FringeOptions& fringe,
            const GoOptions& go, const MaterialOptions& material) {
    if (plan.directions.empty()) throw OutputError("sample plan has no directions");
    const size_t dir_idx = std::min(direction_id, plan.directions.size() - 1);
    const Direction& dir = plan.directions[dir_idx];
    const double freq = plan.frequencies_hz.empty() ? 10e9 : plan.frequencies_hz.front();
    const std::string pol = plan.polarizations.empty() ? "HH" : plan.polarizations.front();

    const VtkMeshDiagnostics diag = compute_facet_diagnostics(mesh, dir, freq, pol, go, material);
    write_vtu(vtu_path, mesh, diag);
    std::cout << "vtk_output: " << vtu_path << "\n";
    std::cout << "direction_azimuth_deg: " << dir.azimuth_deg << "\n";
    std::cout << "direction_elevation_deg: " << dir.elevation_deg << "\n";
    std::cout << "status: ok\n";
    return 0;
}

int cmd_export_mesh(const ResolvedConfig& rc, const std::string& config_path,
                    const std::string& output_path, const std::string& format) {
    NormalizedMesh mesh =
        load_normalized_mesh(rc.value, config_dir_of(config_path), rc.schema_version);
    std::string out_mesh = output_path;
    if (out_mesh.empty()) {
        std::string ext = format.empty() ? "obj" : format;
        if (!ext.empty() && ext.front() == '.') ext = ext.substr(1);
        out_mesh = "exported_mesh." + ext;
    }
    export_mesh(mesh, out_mesh, format);
    std::cout << "exported_mesh: " << out_mesh << "\n";
    std::cout << "vertices: " << mesh.vertices.size() << "\n";
    std::cout << "triangles: " << mesh.triangles.size() << "\n";
    std::cout << "surface_area: " << mesh.report.total_area_m2 << "\n";
    std::cout << "status: ok\n";
    return 0;
}

int cmd_bundle(const std::string& rcs_path, const std::string& mesh_path,
               const std::string& output_path,
               const std::vector<std::pair<std::string, std::string>>& lod_meshes) {
    if (rcs_path.empty() || mesh_path.empty() || output_path.empty()) {
        std::cerr << "strikecem: error: bundle requires --rcs, --mesh, and --output\n";
        return static_cast<int>(ExitCode::Config);
    }
    create_strikeengine_package(output_path, rcs_path, mesh_path, lod_meshes);
    std::cout << "bundle: " << output_path << "\n";
    std::cout << "status: ok\n";
    return 0;
}

int cmd_run_parallel(const std::string& config_path, const std::string& output_path,
                     size_t workers, const std::vector<std::string>& devices,
                     const std::string& work_dir, bool keep_shards,
                     const std::string& schema_path) {
    LocalParallelOptions opts;
    opts.config_path = config_path;
    opts.output_path = output_path;
    opts.num_workers = workers;
    opts.devices = devices;
    opts.work_dir = work_dir;
    opts.keep_shards = keep_shards;

    const LocalParallelResult res = run_local_parallel(opts, schema_path);
    if (!res.success) {
        throw OutputError("run-parallel failed: " + res.error_message);
    }
    std::cout << "shards_completed: " << res.completed_shards << "/" << res.num_shards << "\n";
    std::cout << "wall_time_s: " << res.wall_time_seconds << "\n";
    std::cout << "output: " << res.final_output_path << "\n";
    std::cout << "status: ok\n";
    return 0;
}

int cmd_coordinate(const std::string& manifest_path, const std::string& output_path,
                   const std::string& host, int port, int lease_sec, bool auto_exit_on_complete,
                   const std::string& schema_path) {
    if (manifest_path.empty()) {
        std::cerr << "strikecem: error: coordinate requires --manifest PATH\n";
        return static_cast<int>(ExitCode::Config);
    }
    CoordinatorOptions opts;
    opts.manifest_path = manifest_path;
    opts.output_path = output_path;
    opts.host = host.empty() ? "127.0.0.1" : host;
    opts.port = port;
    opts.lease_duration_seconds = lease_sec > 0 ? lease_sec : 30;
    opts.schema_path = schema_path;
    opts.auto_merge = true;

    CoordinatorServer server(opts);
    server.start(true);
    std::cout << "coordinator_port: " << server.bound_port() << "\n";
    std::cout << "coordinator_url: http://" << opts.host << ":" << server.bound_port() << "\n";
    std::cout << "run_id: " << server.run_id() << "\n";
    std::cout << "status: running\n";
    std::cout.flush();

    while (server.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (auto_exit_on_complete && server.is_campaign_complete()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            std::cout << "status: ok\n";
            std::cout << "campaign: complete\n";
            server.stop();
            break;
        }
    }
    return 0;
}

int cmd_worker(const std::string& coord_url, const std::string& device,
               const std::string& worker_id, int poll_ms, size_t max_tasks,
               const std::string& schema_path) {
    if (coord_url.empty()) {
        std::cerr << "strikecem: error: worker requires --coordinator URL\n";
        return static_cast<int>(ExitCode::Config);
    }
    WorkerOptions opts;
    opts.coordinator_url = coord_url;
    opts.device = device.empty() ? "cpu" : device;
    opts.worker_id = worker_id;
    opts.poll_interval_ms = poll_ms > 0 ? poll_ms : 200;
    opts.max_tasks = max_tasks;
    opts.schema_path = schema_path;

    WorkerDaemon daemon(opts);
    const WorkerResult res = daemon.run();
    if (!res.success) {
        throw OutputError("Worker run failed: " + res.error_message);
    }
    std::cout << "tasks_completed: " << res.tasks_completed << "\n";
    std::cout << "status: ok\n";
    return 0;
}

int cmd_cluster_slurm(const std::string& manifest_path, const std::string& output_path,
                      const std::string& partition, int cpus, const std::string& gpus,
                      const std::string& time_limit, const std::string& job_name) {
    SlurmOptions opts;
    opts.manifest_path = manifest_path;
    opts.output_path = output_path;
    if (!partition.empty()) opts.partition = partition;
    if (cpus > 0) opts.cpus_per_task = cpus;
    opts.gpus_per_task = gpus;
    if (!time_limit.empty()) opts.time_limit = time_limit;
    if (!job_name.empty()) opts.job_name = job_name;

    generate_slurm_script(opts);
    std::cout << "slurm_script: " << output_path << "\n";
    std::cout << "status: ok\n";
    return 0;
}

int cmd_cluster_k8s(const std::string& manifest_path, const std::string& output_path,
                    const std::string& image, const std::string& coordinator_url,
                    int parallelism, int completions, const std::string& job_name) {
    K8sOptions opts;
    opts.manifest_path = manifest_path;
    opts.output_path = output_path;
    if (!image.empty()) opts.image_name = image;
    if (!coordinator_url.empty()) opts.coordinator_url = coordinator_url;
    if (parallelism > 0) opts.parallelism = parallelism;
    if (completions > 0) opts.completions = completions;
    if (!job_name.empty()) opts.job_name = job_name;

    generate_k8s_manifest(opts);
    std::cout << "k8s_manifest: " << output_path << "\n";
    std::cout << "status: ok\n";
    return 0;
}

int cmd_prescan(const std::string& input_path, const std::string& report_out,
                const std::string& tuned_out, const std::string& schema_path) {
    if (input_path.empty()) {
        std::cerr << "strikecem: error: prescan requires input config or mesh path\n";
        return static_cast<int>(ExitCode::Config);
    }

    NormalizedMesh mesh;
    nlohmann::json config_json;
    bool is_config = false;

    if (input_path.rfind(".json") != std::string::npos) {
        is_config = true;
        ResolvedConfig rc = load_config(input_path, schema_path);
        config_json = rc.value;
        mesh = load_normalized_mesh(rc.value, config_dir_of(input_path), rc.schema_version);
    } else {
        const fs::path abs_path = fs::absolute(input_path);
        nlohmann::json dummy_cfg = {
            {"model", {
                {"path", abs_path.string()},
                {"units", "m"},
                {"mesh_repair", "report"}
            }},
            {"mesh", {
                {"max_aspect_ratio", 100.0}
            }},
            {"scem_schema_version", "1.0"}
        };
        mesh = load_normalized_mesh(dummy_cfg, abs_path.parent_path(), "1.0");
    }

    const VehiclePreScanReport report = VehiclePreScan::analyze(mesh);
    std::cout << report.to_text_summary();

    if (!report_out.empty()) {
        std::ofstream out(report_out);
        if (!out) throw OutputError("Cannot write report file: " + report_out);
        out << report.to_json().dump(2) << "\n";
        std::cout << "prescan_report: " << report_out << "\n";
    }

    if (!tuned_out.empty()) {
        if (!is_config) {
            throw OutputError("--auto-tune-config requires a config JSON input file");
        }
        nlohmann::json tuned = VehiclePreScan::auto_tune_config(config_json, report);
        std::ofstream out(tuned_out);
        if (!out) throw OutputError("Cannot write tuned config file: " + tuned_out);
        out << tuned.dump(2) << "\n";
        std::cout << "tuned_config: " << tuned_out << "\n";
    }

    std::cout << "status: ok\n";
    return 0;
}

int cmd_isar(const std::string& input_path, const std::string& output_json,
             const std::string& image_out, const std::string& colormap_name,
             size_t img_width, size_t img_height,
             bool print_ascii, const std::string& pol, double dyn_range,
             size_t range_samples, size_t cross_range_samples, bool windowing,
             double elevation) {
    if (input_path.empty()) {
        std::cerr << "strikecem: error: isar requires an input HDF5 database path\n";
        return static_cast<int>(ExitCode::Config);
    }
    try {
        analysis::IsarConfig cfg;
        cfg.polarization = pol;
        cfg.dynamic_range_db = dyn_range;
        cfg.range_samples = range_samples;
        cfg.cross_range_samples = cross_range_samples;
        cfg.windowing = windowing;
        cfg.target_elevation_deg = elevation;

        const analysis::IsarImage img = analysis::IsarProcessor::compute_from_file(input_path, cfg);

        if (print_ascii || (output_json.empty() && image_out.empty())) {
            std::cout << img.to_ascii_heatmap();
        }

        if (!output_json.empty()) {
            img.write_json(output_json);
            std::cout << "isar_output: " << output_json << "\n";
        }

        if (!image_out.empty()) {
            analysis::IsarRenderOptions ropts;
            ropts.width = img_width > 0 ? img_width : 800;
            ropts.height = img_height > 0 ? img_height : 800;
            ropts.colormap = rendering::parse_colormap(colormap_name);
            img.write_image(image_out, ropts);
            std::cout << "isar_image: " << image_out << "\n";
        }

        std::cout << "status: ok\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "strikecem: error: ISAR processing failed: " << e.what() << "\n";
        return static_cast<int>(ExitCode::Output);
    }
}

int cmd_render(const std::string& input_path, const std::string& output_path,
               const std::string& mode_str, double az, double el,
               double freq_hz, const std::string& pol,
               const std::string& camera_type, const std::string& colormap_name,
               size_t width, size_t height, const std::string& schema_path) {
    if (input_path.empty()) {
        std::cerr << "strikecem: error: render requires input config or mesh path\n";
        return static_cast<int>(ExitCode::Config);
    }

    NormalizedMesh mesh;
    GoOptions go;
    MaterialOptions material;
    Direction dir;
    dir.azimuth_deg = az;
    dir.elevation_deg = el;
    dir.k_hat = direction_from_az_el(az, el);
    double freq = freq_hz > 0.0 ? freq_hz : 10.0e9;
    std::string polarization = pol.empty() ? "HH" : pol;

    if (input_path.rfind(".json") != std::string::npos) {
        ResolvedConfig rc = load_config(input_path, schema_path);
        mesh = load_normalized_mesh(rc.value, config_dir_of(input_path), rc.schema_version);
        SamplePlan plan = build_sample_plan(rc.value);
        if (!plan.directions.empty()) dir = plan.directions.front();
        if (!plan.frequencies_hz.empty()) freq = plan.frequencies_hz.front();
        if (!plan.polarizations.empty()) polarization = plan.polarizations.front();
        if (rc.value.contains("solver") && rc.value["solver"].contains("go") &&
            rc.value["solver"]["go"].contains("shadowing") && rc.value["solver"]["go"]["shadowing"].is_boolean()) {
            go.shadowing = rc.value["solver"]["go"]["shadowing"].get<bool>();
        } else if (rc.value.contains("solver") && rc.value["solver"].contains("shadowing") &&
                   rc.value["solver"]["shadowing"].is_boolean()) {
            go.shadowing = rc.value["solver"]["shadowing"].get<bool>();
        } else if (rc.value.contains("solver") && rc.value["solver"].contains("po_options") &&
                   rc.value["solver"]["po_options"].contains("shadowing") &&
                   rc.value["solver"]["po_options"]["shadowing"].is_boolean()) {
            go.shadowing = rc.value["solver"]["po_options"]["shadowing"].get<bool>();
        }
        if (rc.value.contains("materials")) {
            const ResolvedMaterials resolved_mat = resolve_materials(rc.value, mesh);
            material.enabled = resolved_mat.has_materials;
            material.models = resolved_mat.models;
            material.facet_material_indices = resolved_mat.facet_material_indices;
        }
    } else {
        const fs::path abs_path = fs::absolute(input_path);
        nlohmann::json dummy_cfg = {
            {"model", {
                {"path", abs_path.string()},
                {"units", "m"},
                {"mesh_repair", "report"}
            }},
            {"mesh", {{"max_aspect_ratio", 100.0}}},
            {"scem_schema_version", "1.0"}
        };
        mesh = load_normalized_mesh(dummy_cfg, abs_path.parent_path(), "1.0");
    }

    rendering::ModelRenderConfig rcfg;
    rcfg.width = width > 0 ? width : 1024;
    rcfg.height = height > 0 ? height : 768;
    rcfg.colormap = rendering::parse_colormap(colormap_name);

    if (mode_str == "lit_shadow") rcfg.mode = rendering::RenderShadingMode::LitShadow;
    else if (mode_str == "shaded") rcfg.mode = rendering::RenderShadingMode::ShadedCad;
    else rcfg.mode = rendering::RenderShadingMode::SurfaceCurrents;

    if (camera_type == "isometric") {
        rcfg.camera_radar_aligned = false;
        rcfg.camera = rendering::RenderCamera::isometric_view(mesh, 45.0, 25.0);
    } else {
        rcfg.camera_radar_aligned = true;
    }

    VtkMeshDiagnostics diag;
    if (rcfg.mode != rendering::RenderShadingMode::ShadedCad) {
        diag = compute_facet_diagnostics(mesh, dir, freq, polarization, go, material);
    } else {
        diag.facets.resize(mesh.triangles.size());
    }

    std::string out_file = output_path.empty() ? "model_render.png" : output_path;
    rendering::ModelRenderer::render_and_save(out_file, mesh, diag, dir, freq, polarization, rcfg);
    std::cout << "model_render: " << out_file << "\n";
    std::cout << "status: ok\n";
    return 0;
}

int cmd_report(const std::string& input_path, const std::string& output_html,
               const std::string& config_or_mesh, const std::string& title,
               const std::string& vehicle_name, const std::string& isar_img,
               const std::string& model_img, const std::string& mode_str,
               const std::string& colormap_str) {
    if (input_path.empty()) {
        std::cerr << "strikecem: error: report requires an input database (.h5) or config\n";
        return static_cast<int>(ExitCode::Config);
    }

    std::string hdf5_path = input_path;
    std::string companion = config_or_mesh;

    if (input_path.rfind(".json") != std::string::npos) {
        companion = input_path;
        try {
            std::ifstream f(input_path);
            nlohmann::json j = nlohmann::json::parse(f);
            if (j.contains("output") && j["output"].contains("path")) {
                std::string rel_p = j["output"]["path"].get<std::string>();
                fs::path p(rel_p);
                if (p.is_relative()) {
                    hdf5_path = (fs::path(input_path).parent_path() / p).string();
                } else {
                    hdf5_path = rel_p;
                }
            }
        } catch (...) {
            // fallback
        }
    }

    std::string out_file = output_html.empty() ? "signature_report.html" : output_html;
    analysis::SignatureReportConfig cfg;
    if (!title.empty()) cfg.title = title;
    if (!vehicle_name.empty()) cfg.vehicle_name = vehicle_name;
    if (!isar_img.empty()) cfg.custom_isar_path = isar_img;
    if (!model_img.empty()) cfg.custom_image_path = model_img;
    if (!mode_str.empty()) cfg.render_mode = mode_str;
    if (!colormap_str.empty()) cfg.colormap = colormap_str;

    try {
        analysis::SignatureReportGenerator::write_report(hdf5_path, out_file, cfg, companion);
        std::cout << "signature_report: " << out_file << "\n";
        std::cout << "status: ok\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "strikecem: error: report generation failed: " << e.what() << "\n";
        return static_cast<int>(ExitCode::Output);
    }
}

int cmd_sar(const std::string& input_path, const std::string& output_image,
            const std::string& output_json, const std::string& preset,
            double freq_hz, double bw_hz, double incidence_deg,
            double altitude_m, double velocity_mps, const std::string& ground_str,
            bool enable_shadow, bool overlay_wireframe, double radar_azimuth,
            const std::string& colormap_name,
            const std::string& sidelobe_window, double sidelobe_db, bool show_reticle,
            size_t width, size_t height, const std::string& schema_path) {
    if (input_path.empty()) {
        std::cerr << "strikecem: error: sar requires input mesh or config path\n";
        return static_cast<int>(ExitCode::Config);
    }

    NormalizedMesh mesh;
    std::string vehicle_name = "Target";
    ResolvedMaterials resolved_mats;

    if (input_path.rfind(".json") != std::string::npos) {
        ResolvedConfig rc = load_config(input_path, schema_path);
        mesh = load_normalized_mesh(rc.value, config_dir_of(input_path), rc.schema_version);
        if (rc.value.contains("materials")) {
            resolved_mats = resolve_materials(rc.value, mesh);
        }
        if (rc.value.contains("model") && rc.value["model"].contains("name")) {
            vehicle_name = rc.value["model"]["name"].get<std::string>();
        } else {
            vehicle_name = fs::path(input_path).stem().string();
        }
    } else {
        const fs::path abs_path = fs::absolute(input_path);
        nlohmann::json dummy_cfg = {
            {"model", {
                {"path", abs_path.string()},
                {"units", "m"},
                {"mesh_repair", "report"}
            }},
            {"mesh", {{"max_aspect_ratio", 100.0}}},
            {"scem_schema_version", "1.0"}
        };
        mesh = load_normalized_mesh(dummy_cfg, abs_path.parent_path(), "1.0");
        vehicle_name = abs_path.stem().string();
    }

    analysis::SarConfig cfg;
    if (!preset.empty()) {
        cfg = analysis::SarProcessor::make_preset(preset);
    } else {
        cfg = analysis::SarProcessor::make_preset("satellite-ka");
    }

    if (freq_hz > 0.0) cfg.center_frequency_hz = freq_hz;
    if (bw_hz > 0.0) cfg.bandwidth_hz = bw_hz;
    if (incidence_deg > 0.0) cfg.incidence_angle_deg = incidence_deg;
    if (altitude_m > 0.0) cfg.altitude_m = altitude_m;
    if (velocity_mps > 0.0) cfg.platform_velocity_mps = velocity_mps;
    if (!enable_shadow) cfg.enable_shadow = false;
    if (overlay_wireframe) cfg.overlay_wireframe = true;
    if (radar_azimuth != 0.0) cfg.radar_azimuth_deg = radar_azimuth;

    if (!ground_str.empty()) {
        if (ground_str == "none" || ground_str == "free_space") cfg.ground_type = analysis::SarGroundType::None;
        else if (ground_str == "soil") cfg.ground_type = analysis::SarGroundType::Soil;
        else if (ground_str == "water") cfg.ground_type = analysis::SarGroundType::Water;
        else if (ground_str == "pec") cfg.ground_type = analysis::SarGroundType::Pec;
        else cfg.ground_type = analysis::SarGroundType::Tarmac;
    }

    if (!colormap_name.empty()) cfg.colormap = colormap_name;
    if (!sidelobe_window.empty()) cfg.sidelobe_window = sidelobe_window;
    if (sidelobe_db > 0.0) cfg.sidelobe_suppression_db = sidelobe_db;
    if (show_reticle) cfg.show_reticle = true;
    if (width > 0) cfg.image_width = width;
    if (height > 0) cfg.image_height = height;

    try {
        analysis::SarImage sar_img = analysis::SarProcessor::compute(
            mesh, cfg, vehicle_name,
            resolved_mats.has_materials ? &resolved_mats : nullptr);

        std::string img_path = output_image.empty() ? "sar_image.png" : output_image;
        sar_img.write_image(img_path, cfg.image_width, cfg.image_height, cfg.colormap);
        std::cout << "sar_image: " << img_path << "\n";

        if (!output_json.empty()) {
            sar_img.write_json(output_json);
            std::cout << "sar_json: " << output_json << "\n";
        }

        std::cout << "status: ok\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "strikecem: error: SAR processing failed: " << e.what() << "\n";
        return static_cast<int>(ExitCode::Output);
    }
}

} // namespace

int run(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    std::string schema_path = default_schema_path(argc > 0 ? argv[0] : nullptr);
    std::string bench_profile;
    std::string command;
    std::string output_path;
    std::string mesh_format;
    std::string out_dir = "shards";
    std::string run_id;
    std::string bundle_rcs;
    std::string bundle_mesh;
    std::vector<std::pair<std::string, std::string>> lod_meshes;
    size_t num_shards = 2;
    size_t direction_id = 0;

    // Domain 4 parameters
    size_t workers = 0;
    std::vector<std::string> devices;
    std::string work_dir;
    bool keep_shards = false;
    std::string manifest_path;
    std::string host = "127.0.0.1";
    int port = 0;
    int lease_sec = 30;
    bool auto_exit = false;
    std::string coordinator_url;
    std::string worker_device = "cpu";
    std::string worker_id;
    int poll_ms = 200;
    size_t max_tasks = 0;
    std::string partition = "compute";
    int cpus_per_task = 4;
    std::string gpus_per_task;
    std::string time_limit = "04:00:00";
    std::string job_name;
    std::string image_name = "strikecem:latest";
    int parallelism = 4;
    int completions = 0;

    // Similitude & Pre-Scan parameters
    double scale_override = 0.0;
    std::string prescan_report;
    std::string auto_tune_config_out;

    // ISAR parameters
    bool isar_ascii = false;
    std::string isar_pol = "HH";
    double isar_dyn_range = 35.0;
    size_t isar_range_samples = 64;
    size_t isar_cross_range_samples = 64;
    bool isar_windowing = true;
    double isar_elevation = 0.0;

    // Rendering parameters
    std::string isar_image_out;
    std::string render_mode = "currents";
    std::string render_camera = "radar";
    std::string render_colormap = "turbo";
    size_t render_width = 1024;
    size_t render_height = 768;
    double render_az = 0.0;
    double render_el = 0.0;
    double render_freq = 10.0e9;

    // Report parameters
    std::string report_config;
    std::string report_title;
    std::string report_vehicle;
    std::string report_isar;

    // SAR parameters
    std::string sar_preset;
    double sar_freq = 0.0;
    double sar_bandwidth = 0.0;
    double sar_incidence = 0.0;
    double sar_altitude = 0.0;
    double sar_velocity = 0.0;
    std::string sar_ground;
    std::string sar_colormap;
    bool sar_shadow = true;
    bool sar_wireframe = false;
    double sar_azimuth = 0.0;
    std::string sar_sidelobe_window;
    double sar_sidelobe_db = 0.0;
    bool sar_reticle = false;
    std::string sar_json_out;

    std::vector<std::string> positional;

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema" && i + 1 < args.size()) {
            schema_path = args[++i];
        } else if (args[i] == "--bench-profile" && i + 1 < args.size()) {
            bench_profile = args[++i];
        } else if (args[i] == "--num-shards" && i + 1 < args.size()) {
            num_shards = static_cast<size_t>(std::stoul(args[++i]));
        } else if (args[i] == "--direction-id" && i + 1 < args.size()) {
            direction_id = static_cast<size_t>(std::stoul(args[++i]));
        } else if (args[i] == "--out-dir" && i + 1 < args.size()) {
            out_dir = args[++i];
        } else if (args[i] == "--run-id" && i + 1 < args.size()) {
            run_id = args[++i];
        } else if (args[i] == "--rcs" && i + 1 < args.size()) {
            bundle_rcs = args[++i];
        } else if (args[i] == "--mesh" && i + 1 < args.size()) {
            bundle_mesh = args[++i];
        } else if (args[i] == "--preset" && i + 1 < args.size()) {
            sar_preset = args[++i];
        } else if (args[i] == "--bandwidth" && i + 1 < args.size()) {
            sar_bandwidth = std::stod(args[++i]);
        } else if (args[i] == "--incidence" && i + 1 < args.size()) {
            sar_incidence = std::stod(args[++i]);
        } else if (args[i] == "--altitude" && i + 1 < args.size()) {
            sar_altitude = std::stod(args[++i]);
        } else if (args[i] == "--velocity" && i + 1 < args.size()) {
            sar_velocity = std::stod(args[++i]);
        } else if (args[i] == "--ground" && i + 1 < args.size()) {
            sar_ground = args[++i];
        } else if (args[i] == "--no-shadow") {
            sar_shadow = false;
        } else if (args[i] == "--wireframe") {
            sar_wireframe = true;
        } else if ((args[i] == "--sar-az" || args[i] == "--radar-az") && i + 1 < args.size()) {
            sar_azimuth = std::stod(args[++i]);
        } else if (args[i] == "--json" && i + 1 < args.size()) {
            sar_json_out = args[++i];
        } else if ((args[i] == "--sidelobe-window" || args[i] == "--window") && i + 1 < args.size()) {
            sar_sidelobe_window = args[++i];
        } else if (args[i] == "--sidelobe-db" && i + 1 < args.size()) {
            sar_sidelobe_db = std::stod(args[++i]);
        } else if (args[i] == "--reticle") {
            sar_reticle = true;
        } else if (args[i] == "--no-reticle") {
            sar_reticle = false;
        } else if (args[i] == "--lod" && i + 1 < args.size()) {
            std::string lod_spec = args[++i];
            size_t eq = lod_spec.find('=');
            if (eq != std::string::npos) {
                lod_meshes.emplace_back(lod_spec.substr(0, eq), lod_spec.substr(eq + 1));
            } else {
                lod_meshes.emplace_back("lod", lod_spec);
            }
        } else if (args[i] == "--workers" && i + 1 < args.size()) {
            workers = static_cast<size_t>(std::stoul(args[++i]));
        } else if (args[i] == "--devices" && i + 1 < args.size()) {
            std::string d_str = args[++i];
            std::stringstream ss(d_str);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) devices.push_back(item);
            }
        } else if (args[i] == "--work-dir" && i + 1 < args.size()) {
            work_dir = args[++i];
        } else if (args[i] == "--keep-shards") {
            keep_shards = true;
        } else if (args[i] == "--manifest" && i + 1 < args.size()) {
            manifest_path = args[++i];
        } else if (args[i] == "--port" && i + 1 < args.size()) {
            port = std::stoi(args[++i]);
        } else if (args[i] == "--host" && i + 1 < args.size()) {
            host = args[++i];
        } else if (args[i] == "--lease" && i + 1 < args.size()) {
            lease_sec = std::stoi(args[++i]);
        } else if (args[i] == "--auto-exit") {
            auto_exit = true;
        } else if (args[i] == "--coordinator" && i + 1 < args.size()) {
            coordinator_url = args[++i];
        } else if (args[i] == "--device" && i + 1 < args.size()) {
            worker_device = args[++i];
        } else if (args[i] == "--worker-id" && i + 1 < args.size()) {
            worker_id = args[++i];
        } else if (args[i] == "--poll-ms" && i + 1 < args.size()) {
            poll_ms = std::stoi(args[++i]);
        } else if (args[i] == "--max-tasks" && i + 1 < args.size()) {
            max_tasks = static_cast<size_t>(std::stoul(args[++i]));
        } else if (args[i] == "--partition" && i + 1 < args.size()) {
            partition = args[++i];
        } else if ((args[i] == "--cpus" || args[i] == "--cpus-per-task") && i + 1 < args.size()) {
            cpus_per_task = std::stoi(args[++i]);
        } else if ((args[i] == "--gpus" || args[i] == "--gpus-per-task") && i + 1 < args.size()) {
            gpus_per_task = args[++i];
        } else if (args[i] == "--time" && i + 1 < args.size()) {
            time_limit = args[++i];
        } else if (args[i] == "--job-name" && i + 1 < args.size()) {
            job_name = args[++i];
        } else if (args[i] == "--image" && i + 1 < args.size()) {
            ++i;
            if (command == "cluster" || command == "cluster-k8s" || command == "k8s") {
                image_name = args[i];
            } else if (command == "isar") {
                isar_image_out = args[i];
            } else {
                image_name = args[i];
                isar_image_out = args[i];
            }
        } else if (args[i] == "--parallelism" && i + 1 < args.size()) {
            parallelism = std::stoi(args[++i]);
        } else if (args[i] == "--completions" && i + 1 < args.size()) {
            completions = std::stoi(args[++i]);
        } else if (args[i] == "--scale" && i + 1 < args.size()) {
            scale_override = std::stod(args[++i]);
        } else if (args[i] == "--report" && i + 1 < args.size()) {
            prescan_report = args[++i];
        } else if (args[i] == "--auto-tune-config" && i + 1 < args.size()) {
            auto_tune_config_out = args[++i];
        } else if (args[i] == "--config" && i + 1 < args.size()) {
            report_config = args[++i];
        } else if (args[i] == "--title" && i + 1 < args.size()) {
            report_title = args[++i];
        } else if (args[i] == "--vehicle" && i + 1 < args.size()) {
            report_vehicle = args[++i];
        } else if (args[i] == "--isar" && i + 1 < args.size()) {
            report_isar = args[++i];
        } else if (args[i] == "--ascii") {
            isar_ascii = true;
        } else if (args[i] == "--pol" && i + 1 < args.size()) {
            isar_pol = args[++i];
        } else if (args[i] == "--dyn-range" && i + 1 < args.size()) {
            isar_dyn_range = std::stod(args[++i]);
        } else if (args[i] == "--range-samples" && i + 1 < args.size()) {
            isar_range_samples = static_cast<size_t>(std::stoul(args[++i]));
        } else if (args[i] == "--cross-range-samples" && i + 1 < args.size()) {
            isar_cross_range_samples = static_cast<size_t>(std::stoul(args[++i]));
        } else if (args[i] == "--no-window") {
            isar_windowing = false;
        } else if (args[i] == "--elevation" && i + 1 < args.size()) {
            isar_elevation = std::stod(args[++i]);
        } else if (args[i] == "-i" && i + 1 < args.size()) {
            isar_image_out = args[++i];
        } else if (args[i] == "--colormap" && i + 1 < args.size()) {
            render_colormap = args[++i];
            sar_colormap = render_colormap;
        } else if (args[i] == "--width" && i + 1 < args.size()) {
            render_width = static_cast<size_t>(std::stoul(args[++i]));
        } else if (args[i] == "--height" && i + 1 < args.size()) {
            render_height = static_cast<size_t>(std::stoul(args[++i]));
        } else if (args[i] == "--mode" && i + 1 < args.size()) {
            render_mode = args[++i];
        } else if (args[i] == "--camera" && i + 1 < args.size()) {
            render_camera = args[++i];
        } else if (args[i] == "--az" && i + 1 < args.size()) {
            render_az = std::stod(args[++i]);
        } else if (args[i] == "--el" && i + 1 < args.size()) {
            render_el = std::stod(args[++i]);
        } else if (args[i] == "--freq" && i + 1 < args.size()) {
            render_freq = std::stod(args[++i]);
            sar_freq = render_freq;
        } else if (args[i] == "--format" && i + 1 < args.size()) {
            mesh_format = args[++i];
        } else if ((args[i] == "--output" || args[i] == "-o") && i + 1 < args.size()) {
            output_path = args[++i];
        } else if (args[i] == "--help" || args[i] == "-h") {
            std::cout << kUsage;
            return 0;
        } else if (command.empty()) {
            command = args[i];
        } else {
            positional.push_back(args[i]);
        }
    }

    const bool allow_no_positional = (command == "bundle" || command == "worker" ||
                                      command == "coordinate" || command == "cluster" ||
                                      command == "cluster-slurm" || command == "cluster-k8s");
    if (command.empty() || (positional.empty() && !allow_no_positional)) {
        std::cerr << kUsage;
        return static_cast<int>(ExitCode::Config);
    }
    if (command != "validate" && command != "estimate" && command != "run" &&
        command != "shard" && command != "merge" && command != "vtk" &&
        command != "export-mesh" && command != "bundle" && command != "run-parallel" &&
        command != "coordinate" && command != "worker" && command != "cluster" &&
        command != "cluster-slurm" && command != "cluster-k8s" && command != "prescan" &&
        command != "isar" && command != "render" && command != "report" && command != "sar") {
        std::cerr << kUsage;
        return static_cast<int>(ExitCode::Config);
    }

    try {
        if (command == "bundle") {
            return cmd_bundle(bundle_rcs, bundle_mesh, output_path, lod_meshes);
        }
        if (command == "merge") {
            std::vector<std::string> additional(positional.begin() + 1, positional.end());
            return cmd_merge(positional[0], additional, output_path, schema_path);
        }
        if (command == "run-parallel") {
            const std::string& cfg = positional.empty() ? "" : positional[0];
            if (cfg.empty()) {
                std::cerr << "strikecem: error: run-parallel requires config path\n";
                return static_cast<int>(ExitCode::Config);
            }
            return cmd_run_parallel(cfg, output_path, workers, devices, work_dir, keep_shards, schema_path);
        }
        if (command == "coordinate") {
            std::string mf = manifest_path;
            if (mf.empty() && !positional.empty()) mf = positional[0];
            return cmd_coordinate(mf, output_path, host, port, lease_sec, auto_exit, schema_path);
        }
        if (command == "worker") {
            return cmd_worker(coordinator_url, worker_device, worker_id, poll_ms, max_tasks, schema_path);
        }
        if (command == "cluster" || command == "cluster-slurm" || command == "cluster-k8s") {
            std::string sub = command;
            if (command == "cluster" && !positional.empty()) {
                sub = positional[0];
            }
            std::string mf = manifest_path;
            if (mf.empty() && positional.size() > 1) {
                mf = positional[1];
            }
            if (sub == "generate-slurm" || sub == "slurm" || sub == "cluster-slurm") {
                return cmd_cluster_slurm(mf, output_path, partition, cpus_per_task, gpus_per_task, time_limit, job_name);
            }
            if (sub == "generate-k8s" || sub == "k8s" || sub == "cluster-k8s") {
                return cmd_cluster_k8s(mf, output_path, image_name, coordinator_url, parallelism, completions, job_name);
            }
            std::cerr << "strikecem: error: unknown cluster subcommand: " << sub << "\n";
            return static_cast<int>(ExitCode::Config);
        }
        if (command == "prescan") {
            const std::string& input = positional.empty() ? "" : positional[0];
            return cmd_prescan(input, prescan_report, auto_tune_config_out, schema_path);
        }
        if (command == "isar") {
            const std::string& input = positional.empty() ? "" : positional[0];
            return cmd_isar(input, output_path, isar_image_out, render_colormap,
                            render_width, render_height,
                            isar_ascii, isar_pol, isar_dyn_range,
                            isar_range_samples, isar_cross_range_samples, isar_windowing,
                            isar_elevation);
        }
        if (command == "render") {
            const std::string& input = positional.empty() ? "" : positional[0];
            return cmd_render(input, output_path, render_mode, render_az, render_el,
                              render_freq, isar_pol, render_camera, render_colormap,
                              render_width, render_height, schema_path);
        }
        if (command == "report") {
            const std::string& input = positional.empty() ? "" : positional[0];
            std::string companion = !report_config.empty() ? report_config : bundle_mesh;
            return cmd_report(input, output_path, companion, report_title,
                              report_vehicle, report_isar, isar_image_out,
                              render_mode, render_colormap);
        }
        if (command == "sar") {
            const std::string& input = positional.empty() ? "" : positional[0];
            return cmd_sar(input, output_path, sar_json_out, sar_preset,
                           sar_freq, sar_bandwidth, sar_incidence,
                           sar_altitude, sar_velocity, sar_ground,
                           sar_shadow, sar_wireframe, sar_azimuth,
                           sar_colormap, sar_sidelobe_window, sar_sidelobe_db,
                           sar_reticle, render_width,
                           render_height, schema_path);
        }

        if (positional.size() != 1) {
            std::cerr << kUsage;
            return static_cast<int>(ExitCode::Config);
        }
        const std::string& config_path = positional[0];
        ResolvedConfig rc = load_config(config_path, schema_path);
        if (!bench_profile.empty() && command != "run") {
            std::cerr << kUsage;
            return static_cast<int>(ExitCode::Config);
        }
        const std::vector<BenchmarkProfile> profiles = load_benchmark_profiles();
        if (command == "validate") return cmd_validate(rc, config_path);
        if (command == "estimate") return cmd_estimate(rc, config_path, profiles);
        if (command == "export-mesh") return cmd_export_mesh(rc, config_path, output_path, mesh_format);

        SamplePlan plan = build_sample_plan(rc.value);
        print_warnings(plan);
        const DesignerIdentity designer = validate_designer_package(rc.value, config_path);
        for (const auto& w : designer.warnings) std::cerr << "strikecem: warning: " << w << "\n";
        NormalizedMesh mesh =
            load_normalized_mesh(rc.value, config_dir_of(config_path), rc.schema_version);

        SimilitudeConfig sim_cfg = parse_similitude_config(rc.value);
        if (scale_override > 0.0) {
            sim_cfg.enabled = true;
            sim_cfg.geometric_scale = scale_override;
        }
        if (sim_cfg.enabled && sim_cfg.geometric_scale != 1.0) {
            SimilitudeSummary sim_sum = apply_similitude(rc, plan, mesh, sim_cfg);
            std::cout << "similitude_geometric_scale: " << sim_sum.geometric_scale << "\n";
            std::cout << "similitude_frequency_scale: " << sim_sum.frequency_scale << "\n";
            std::cout << "similitude_rcs_offset_dbsm: " << sim_sum.rcs_offset_dbsm << " dB\n";
        }

        if (command == "shard") {
            return cmd_shard(rc, plan, mesh, config_path, num_shards, out_dir, run_id);
        }

        std::cout << "scem_schema_version: " << rc.schema_version << "\n";
        std::cout << "resolved_config_hash: " << rc.hash << "\n";
        const ResolvedMaterials resolved_mat = resolve_materials(rc.value, mesh);
        for (const auto& w : resolved_mat.warnings) std::cerr << "strikecem: warning: " << w << "\n";
        MaterialOptions material;
        material.enabled = rc.value.contains("materials") || resolved_mat.has_materials;
        material.models = resolved_mat.models;
        material.facet_material_indices = resolved_mat.facet_material_indices;

        EdgeModel edge_model;
        FringeOptions fringe;
        const std::string edge_corr = rc.value["solver"]["po_options"].value("edge_correction", "none");
        if (edge_corr == "fringe" || edge_corr == "ptd_lite" || edge_corr == "ptd_full") {
            edge_model = extract_edges(mesh);
            fringe = FringeOptions{true, &edge_model, edge_corr};
            std::cout << "fringe_edges: " << edge_model.edges.size() << "\n";
        }
        GoOptions go{rc.value["solver"]["po_options"].value("shadowing", false),
                     rc.value["solver"]["po_options"].value("max_bounces", 1)};

        if (command == "vtk") {
            const std::string vtu_file = output_path.empty() ? "diag.vtu" : output_path;
            return cmd_vtk(rc, plan, mesh, vtu_file, direction_id, fringe, go, material);
        }

        const ResourceEstimate est = estimate_resources(rc.value, plan, mesh, profiles);
        for (const auto& w : est.warnings) std::cerr << "strikecem: warning: " << w << "\n";
        if (!est.fits) {
            std::cerr << "strikecem: error: estimated resource use exceeds limit; refusing run\n";
            return static_cast<int>(ExitCode::Resource);
        }
        const std::string format = rc.value["output"]["format"].get<std::string>();
        const fs::path out_path(rc.value["output"]["path"].get<std::string>());
        const bool is_shard = rc.value["run"].contains("shard");
        const bool resume = !rc.resume_from_checkpoint.empty();
        if (resume) {
            if (is_shard)
                throw ConfigError("resuming individual shards is not supported; rerun shard directly");
            if (format != "hdf5")
                throw ConfigError("run.resume_from_checkpoint requires HDF5 output");
            const std::string& checkpoint = rc.resume_from_checkpoint;
            if (!fs::exists(checkpoint))
                throw OutputError("checkpoint file not found: " + checkpoint);
            if (fs::absolute(checkpoint).lexically_normal() !=
                fs::absolute(out_path).lexically_normal())
                throw ConfigError("run.resume_from_checkpoint must equal output.path");
            const auto solve_start = std::chrono::steady_clock::now();
            const size_t solved = resume_hdf5_output(rc, plan, mesh, checkpoint, fringe, go, material);
            const double solve_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start)
                    .count();
            if (!bench_profile.empty())
                write_benchmark_profile(bench_profile, rc, mesh.report.triangle_count, solved,
                                        solve_seconds,
                                        rc.value["execution"].value("accelerator", "cpu"),
                                        cuda::active_device_name(rc.value));
            if (solved == 0) std::cout << "status: already complete\n";
            std::cout << "samples_resumed: " << solved << " / " << plan.sample_count() << "\n";
            std::cout << "status: ok\n";
            return 0;
        }
        if (fs::exists(out_path)) {
            if (is_shard) {
                fs::remove(out_path);
                if (format == "csv") {
                    fs::remove(out_path.string() + ".json");
                }
            } else {
                if (format == "hdf5") {
                    if (check_existing_hdf5(rc, plan, mesh, out_path.string())) {
                        std::cout << "status: already complete\n";
                        return 0;
                    }
                } else if (format == "scem") {
                    if (check_existing_scem(rc, plan, mesh, out_path.string())) {
                        std::cout << "status: already complete\n";
                        return 0;
                    }
                    throw OutputError("output file exists; remove it to regenerate: " +
                                      out_path.string());
                } else {
                    throw OutputError("output file exists; remove it to regenerate: " +
                                      out_path.string());
                }
            }
        }
        const bool is_adaptive = rc.value["angles"].contains("adaptive");
        const auto solve_start = std::chrono::steady_clock::now();
        PoResult result;
        SamplePlan execution_plan = plan;
        if (is_adaptive) {
            AdaptiveResult a_res = solve_adaptive(mesh, rc, fringe, go, material);
            result = a_res.merged_result;
            execution_plan = a_res.merged_plan;
            std::cout << "adaptive_coarse_samples: " << a_res.coarse_result.samples.size() << "\n";
            std::cout << "adaptive_peaks: " << a_res.peaks_detected << "\n";
            std::cout << "adaptive_windows: " << a_res.windows.size() << "\n";
            std::cout << "adaptive_refinement_samples: " << a_res.refinement_result.samples.size() << "\n";
        } else if (!is_shard) {
            result = solve_po(mesh, plan, rc.value, fringe, go, material);
        } else {
            const size_t shard_id = rc.value["run"]["shard"]["shard_id"].get<size_t>();
            const size_t shard_count = rc.value["run"]["shard"]["num_shards"].get<size_t>();
            const size_t total_units = plan.frequencies_hz.size() * plan.directions.size();
            const size_t base_units = total_units / shard_count;
            const size_t rem_units = total_units % shard_count;
            const size_t u_count = base_units + (shard_id < rem_units ? 1 : 0);
            const size_t u_start = shard_id * base_units + std::min(shard_id, rem_units);
            const size_t ndir = plan.directions.size();

            std::vector<std::pair<uint32_t, uint32_t>> shard_units;
            shard_units.reserve(u_count);
            for (size_t u = u_start; u < u_start + u_count; ++u) {
                const uint32_t fi = static_cast<uint32_t>(u / ndir);
                const uint32_t di = static_cast<uint32_t>(u % ndir);
                shard_units.emplace_back(fi, di);
            }
            result = solve_po_units(mesh, plan, rc.value, shard_units, fringe, go, material);
        }

        if (sim_cfg.enabled && sim_cfg.auto_rescale_rcs && sim_cfg.geometric_scale != 1.0) {
            rescale_po_result(result, sim_cfg);
        }

        const double solve_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
        for (const auto& w : result.warnings) std::cerr << "strikecem: warning: " << w << "\n";
        if (format == "csv")
            write_csv_and_sidecar(rc, execution_plan, mesh, result, designer, fringe, material);
        else if (format == "scem")
            write_scem_output(rc, execution_plan, mesh, result, designer, fringe, material);
        else
            write_hdf5_output(rc, execution_plan, mesh, result, designer, fringe, material);
        if (!bench_profile.empty())
            write_benchmark_profile(bench_profile, rc, mesh.report.triangle_count,
                                     execution_plan.sample_count(), solve_seconds,
                                     rc.value["execution"].value("accelerator", "cpu"),
                                     cuda::active_device_name(rc.value));
        if (rc.value["output"].contains("diagnostics") &&
            rc.value["output"]["diagnostics"].contains("vtk_export_path")) {
            const std::string vtu_file = rc.value["output"]["diagnostics"]["vtk_export_path"].get<std::string>();
            const size_t s_idx = rc.value["output"]["diagnostics"].value("sample_index", 0);
            const size_t dir_idx = std::min(s_idx, execution_plan.directions.size() - 1);
            const Direction& dir = execution_plan.directions[dir_idx];
            const double freq = execution_plan.frequencies_hz.front();
            const std::string pol = execution_plan.polarizations.front();
            const VtkMeshDiagnostics diag = compute_facet_diagnostics(mesh, dir, freq, pol, go, material);
            write_vtu(vtu_file, mesh, diag);
            std::cout << "vtk_diagnostics: " << vtu_file << "\n";
        }
        if (rc.value["output"].contains("render") ||
            (rc.value["output"].contains("diagnostics") &&
             rc.value["output"]["diagnostics"].contains("image_export_path"))) {
            std::string render_path = "model_render.png";
            std::string mode_str = "currents";
            std::string colormap_str = "turbo";
            size_t r_width = 1024, r_height = 768;
            size_t s_idx = 0;

            if (rc.value["output"].contains("render")) {
                const auto& r_cfg = rc.value["output"]["render"];
                render_path = r_cfg.value("path", "model_render.png");
                mode_str = r_cfg.value("mode", "currents");
                colormap_str = r_cfg.value("colormap", "turbo");
                r_width = r_cfg.value("width", 1024);
                r_height = r_cfg.value("height", 768);
                s_idx = r_cfg.value("sample_index", 0);
            } else {
                const auto& d_cfg = rc.value["output"]["diagnostics"];
                render_path = d_cfg["image_export_path"].get<std::string>();
                mode_str = d_cfg.value("image_mode", "currents");
                colormap_str = d_cfg.value("image_colormap", "turbo");
                s_idx = d_cfg.value("sample_index", 0);
            }

            const size_t dir_idx = std::min(s_idx, execution_plan.directions.size() - 1);
            const Direction& dir = execution_plan.directions[dir_idx];
            const double freq = execution_plan.frequencies_hz.front();
            const std::string pol = execution_plan.polarizations.front();

            const VtkMeshDiagnostics diag = compute_facet_diagnostics(mesh, dir, freq, pol, go, material);
            rendering::ModelRenderConfig mrc;
            mrc.width = r_width;
            mrc.height = r_height;
            mrc.colormap = rendering::parse_colormap(colormap_str);
            if (mode_str == "lit_shadow") mrc.mode = rendering::RenderShadingMode::LitShadow;
            else if (mode_str == "shaded") mrc.mode = rendering::RenderShadingMode::ShadedCad;
            else mrc.mode = rendering::RenderShadingMode::SurfaceCurrents;

            rendering::ModelRenderer::render_and_save(render_path, mesh, diag, dir, freq, pol, mrc);
            std::cout << "model_render: " << render_path << "\n";
        }
        if (rc.value["output"].contains("telemetry") &&
            rc.value["output"]["telemetry"].value("enabled", false)) {
            const std::string telem_path = rc.value["output"]["telemetry"].value("path", "telemetry.json");
            telemetry::TelemetryCollector tc;
            tc.record_metadata("config_hash", rc.hash);
            tc.record_metadata("solver", rc.value["solver"]["type"].get<std::string>());
            tc.record_metadata("format", format);
            tc.record_counter("samples_computed", result.samples.size());
            tc.record_counter("facets_count", mesh.report.triangle_count);
            tc.record_metric("solve_wall_seconds", solve_seconds);
            if (solve_seconds > 0.0) {
                tc.record_metric("samples_per_second", static_cast<double>(result.samples.size()) / solve_seconds);
                tc.record_metric("facet_evaluations_per_second",
                                 (static_cast<double>(result.samples.size()) * mesh.report.triangle_count) / solve_seconds);
            }
            tc.write_json_report(telem_path);
            std::cout << "telemetry: " << telem_path << "\n";
        }
        std::cout << "samples: " << result.samples.size() << "\n";
        std::cout << "status: ok\n";
        return 0;
    } catch (const ConfigError& e) {
        return fail(e, ExitCode::Config);
    } catch (const MeshError& e) {
        return fail(e, ExitCode::Mesh);
    } catch (const MeshLoadError& e) {
        return fail(e, ExitCode::Mesh);
    } catch (const OutputError& e) {
        return fail(e, ExitCode::Output);
    } catch (const cuda::CudaError& e) {
        return fail(e, ExitCode::Resource);
    } catch (const std::exception& e) {
        return fail(e, ExitCode::Solver);
    }
}

} // namespace strikecem::cli

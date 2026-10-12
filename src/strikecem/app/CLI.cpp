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
#include "strikecem/distributed/Shard.hpp"
#include "strikecem/io/CsvWriter.hpp"
#include "strikecem/io/DesignerPackage.hpp"
#include "strikecem/io/Hdf5Writer.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/runtime/Estimate.hpp"
#include "strikecem/solvers/GpuPO.hpp"
#include "strikecem/solvers/Material.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

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
    "usage: strikecem [--schema PATH] [--bench-profile PATH] <validate|estimate|run|shard|merge> [options...] <args...>\n"
    "  validate  resolve config, inspect mesh, print hashes\n"
    "  estimate  validate plus plan counts and resource estimates\n"
    "  run       full pipeline (PO solver, CSV/HDF5 output)\n"
    "  shard     partition sample plan across N shards and write shard configs + manifest\n"
    "            options: --num-shards N (default 2), --out-dir DIR (default ./shards), --run-id ID\n"
    "  merge     merge completed shard outputs into a unified database\n"
    "            options: --output PATH <manifest.json | shard1 shard2 ...>\n"
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

} // namespace

int run(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    std::string schema_path = default_schema_path(argc > 0 ? argv[0] : nullptr);
    std::string bench_profile;
    std::string command;
    std::string output_path;
    std::string out_dir = "shards";
    std::string run_id;
    size_t num_shards = 2;
    std::vector<std::string> positional;

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema" && i + 1 < args.size()) {
            schema_path = args[++i];
        } else if (args[i] == "--bench-profile" && i + 1 < args.size()) {
            bench_profile = args[++i];
        } else if (args[i] == "--num-shards" && i + 1 < args.size()) {
            num_shards = static_cast<size_t>(std::stoul(args[++i]));
        } else if (args[i] == "--out-dir" && i + 1 < args.size()) {
            out_dir = args[++i];
        } else if (args[i] == "--run-id" && i + 1 < args.size()) {
            run_id = args[++i];
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

    if (command.empty() || positional.empty()) {
        std::cerr << kUsage;
        return static_cast<int>(ExitCode::Config);
    }
    if (command != "validate" && command != "estimate" && command != "run" &&
        command != "shard" && command != "merge") {
        std::cerr << kUsage;
        return static_cast<int>(ExitCode::Config);
    }

    try {
        if (command == "merge") {
            std::vector<std::string> additional(positional.begin() + 1, positional.end());
            return cmd_merge(positional[0], additional, output_path, schema_path);
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

        SamplePlan plan = build_sample_plan(rc.value);
        print_warnings(plan);
        const DesignerIdentity designer = validate_designer_package(rc.value, config_path);
        for (const auto& w : designer.warnings) std::cerr << "strikecem: warning: " << w << "\n";
        NormalizedMesh mesh =
            load_normalized_mesh(rc.value, config_dir_of(config_path), rc.schema_version);

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
        if (rc.value["solver"]["po_options"].value("edge_correction", "none") == "fringe") {
            edge_model = extract_edges(mesh);
            fringe = FringeOptions{true, &edge_model};
            std::cout << "fringe_edges: " << edge_model.edges.size() << "\n";
        }
        GoOptions go{rc.value["solver"]["po_options"].value("shadowing", false),
                     rc.value["solver"]["po_options"].value("max_bounces", 1)};
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
                if (format != "hdf5")
                    throw OutputError("output file exists; remove it to regenerate: " +
                                      out_path.string());
                if (check_existing_hdf5(rc, plan, mesh, out_path.string())) {
                    std::cout << "status: already complete\n";
                    return 0;
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
        const double solve_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
        for (const auto& w : result.warnings) std::cerr << "strikecem: warning: " << w << "\n";
        if (format == "csv")
            write_csv_and_sidecar(rc, execution_plan, mesh, result, designer, fringe, material);
        else
            write_hdf5_output(rc, execution_plan, mesh, result, designer, fringe, material);
        if (!bench_profile.empty())
            write_benchmark_profile(bench_profile, rc, mesh.report.triangle_count,
                                     execution_plan.sample_count(), solve_seconds,
                                     rc.value["execution"].value("accelerator", "cpu"),
                                     cuda::active_device_name(rc.value));
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

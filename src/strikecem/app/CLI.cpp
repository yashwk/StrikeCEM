// v1 CLI: validate resolves, inspects the mesh, and reports; estimate adds
// plan counts; run validates then refuses until the Phase 1 solver lands.
#include "strikecem/app/CLI.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/CsvWriter.hpp"
#include "strikecem/io/DesignerPackage.hpp"
#include "strikecem/io/Hdf5Writer.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/runtime/Estimate.hpp"
#include "strikecem/solvers/GpuPO.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace strikecem::cli {
namespace {

namespace fs = std::filesystem;

constexpr const char* kUsage =
    "usage: strikecem [--schema PATH] [--bench-profile PATH] <validate|estimate|run> config.json\n"
    "  validate  resolve config, inspect mesh, print hashes\n"
    "  estimate  validate plus plan counts and resource estimates\n"
    "  run       full pipeline (PO solver, CSV/HDF5 output)\n"
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

} // namespace

int run(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    std::string schema_path = "schemas/config.schema.json";
    std::string bench_profile;
    std::string command, config_path;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema" && i + 1 < args.size()) {
            schema_path = args[++i];
        } else if (args[i] == "--bench-profile" && i + 1 < args.size()) {
            bench_profile = args[++i];
        } else if (args[i] == "--help" || args[i] == "-h") {
            std::cout << kUsage;
            return 0;
        } else if (command.empty()) {
            command = args[i];
        } else if (config_path.empty()) {
            config_path = args[i];
        } else {
            std::cerr << kUsage;
            return static_cast<int>(ExitCode::Config);
        }
    }
    if (command.empty() || config_path.empty() ||
        (command != "validate" && command != "estimate" && command != "run")) {
        std::cerr << kUsage;
        return static_cast<int>(ExitCode::Config);
    }
    try {
        ResolvedConfig rc = load_config(config_path, schema_path);
        if (!bench_profile.empty() && command != "run") {
            std::cerr << kUsage;
            return static_cast<int>(ExitCode::Config);
        }
        const std::vector<BenchmarkProfile> profiles = load_benchmark_profiles();
        if (command == "validate") return cmd_validate(rc, config_path);
        if (command == "estimate") return cmd_estimate(rc, config_path, profiles);
        std::cout << "scem_schema_version: " << rc.schema_version << "\n";
        std::cout << "resolved_config_hash: " << rc.hash << "\n";
        SamplePlan plan = build_sample_plan(rc.value);
        print_warnings(plan);
        const DesignerIdentity designer = validate_designer_package(rc.value, config_path);
        for (const auto& w : designer.warnings) std::cerr << "strikecem: warning: " << w << "\n";
        NormalizedMesh mesh =
            load_normalized_mesh(rc.value, config_dir_of(config_path), rc.schema_version);
        // Fringe correction is off unless the schema exposes edge_correction
        // "fringe" (slice D2d). The loader rejects non-manifold input, so
        // extraction cannot throw here.
        EdgeModel edge_model;
        FringeOptions fringe;
        if (rc.value["solver"]["po_options"].value("edge_correction", "none") == "fringe") {
            edge_model = extract_edges(mesh);
            fringe = FringeOptions{true, &edge_model};
            std::cout << "fringe_edges: " << edge_model.edges.size() << "\n";
        }
        const ResourceEstimate est = estimate_resources(rc.value, plan, mesh, profiles);
        for (const auto& w : est.warnings) std::cerr << "strikecem: warning: " << w << "\n";
        if (!est.fits) {
            std::cerr << "strikecem: error: estimated resource use exceeds limit; refusing run\n";
            return static_cast<int>(ExitCode::Resource);
        }
        const std::string format = rc.value["output"]["format"].get<std::string>();
        const fs::path out_path(rc.value["output"]["path"].get<std::string>());
        const bool resume = !rc.resume_from_checkpoint.empty();
        if (resume) {
            if (format != "hdf5")
                throw ConfigError("run.resume_from_checkpoint requires HDF5 output");
            const std::string& checkpoint = rc.resume_from_checkpoint;
            if (!fs::exists(checkpoint))
                throw OutputError("checkpoint file not found: " + checkpoint);
            if (fs::absolute(checkpoint).lexically_normal() !=
                fs::absolute(out_path).lexically_normal())
                throw ConfigError("run.resume_from_checkpoint must equal output.path");
            const auto solve_start = std::chrono::steady_clock::now();
            const size_t solved = resume_hdf5_output(rc, plan, mesh, checkpoint, fringe);
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
            if (format != "hdf5")
                throw OutputError("output file exists; remove it to regenerate: " +
                                  out_path.string());
            if (check_existing_hdf5(rc, plan, mesh, out_path.string())) {
                std::cout << "status: already complete\n";
                return 0;
            }
        }
        const auto solve_start = std::chrono::steady_clock::now();
        PoResult result = solve_po(mesh, plan, rc.value, fringe);
        const double solve_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
        for (const auto& w : result.warnings) std::cerr << "strikecem: warning: " << w << "\n";
        if (format == "csv")
            write_csv_and_sidecar(rc, plan, mesh, result, designer, fringe);
        else
            write_hdf5_output(rc, plan, mesh, result, designer, fringe);
        if (!bench_profile.empty())
            write_benchmark_profile(bench_profile, rc, mesh.report.triangle_count,
                                    plan.sample_count(), solve_seconds,
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

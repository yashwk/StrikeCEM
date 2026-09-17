// v1 CLI: validate resolves, inspects the mesh, and reports; estimate adds
// plan counts; run validates then refuses until the Phase 1 solver lands.
#include "strikecem/app/CLI.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/CsvWriter.hpp"
#include "strikecem/io/Hdf5Writer.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace strikecem::cli {
namespace {

namespace fs = std::filesystem;

constexpr const char* kUsage =
    "usage: strikecem [--schema PATH] <validate|estimate|run> config.json\n"
    "  validate  resolve config, inspect mesh, print hashes\n"
    "  estimate  validate plus frequency/direction/sample counts\n"
    "  run       full pipeline (PO solver: Phase 1)\n";

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
    NormalizedMesh mesh =
        load_normalized_mesh(rc.value, config_dir_of(config_path), rc.schema_version);
    std::cout << "scem_schema_version: " << rc.schema_version << "\n";
    std::cout << "resolved_config_hash: " << rc.hash << "\n";
    std::cout << "model: " << rc.value["model"]["path"].get<std::string>() << "\n";
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

int cmd_estimate(const ResolvedConfig& rc, const std::string& config_path) {
    SamplePlan plan = build_sample_plan(rc.value);
    print_warnings(plan);
    const int rc_code = cmd_validate(rc, config_path);
    if (rc_code != 0) return rc_code;
    std::cout << "frequencies: " << plan.frequencies_hz.size() << "\n";
    std::cout << "directions: " << plan.directions.size() << "\n";
    std::cout << "polarizations: " << plan.polarizations.size() << "\n";
    std::cout << "samples: " << plan.sample_count() << "\n";
    return 0;
}

} // namespace

int run(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    std::string schema_path = "schemas/config.schema.json";
    std::string command, config_path;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema" && i + 1 < args.size()) {
            schema_path = args[++i];
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
        if (command == "validate") return cmd_validate(rc, config_path);
        if (command == "estimate") return cmd_estimate(rc, config_path);
        std::cout << "scem_schema_version: " << rc.schema_version << "\n";
        std::cout << "resolved_config_hash: " << rc.hash << "\n";
        SamplePlan plan = build_sample_plan(rc.value);
        print_warnings(plan);
        NormalizedMesh mesh =
            load_normalized_mesh(rc.value, config_dir_of(config_path), rc.schema_version);
        PoResult result = solve_po(mesh, plan, rc.value);
        for (const auto& w : result.warnings) std::cerr << "strikecem: warning: " << w << "\n";
        if (rc.value["output"]["format"].get<std::string>() == "csv")
            write_csv_and_sidecar(rc, plan, mesh, result);
        else
            write_hdf5_output(rc, plan, mesh, result);
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
    } catch (const std::exception& e) {
        return fail(e, ExitCode::Solver);
    }
}

} // namespace strikecem::cli

// v1 CLI: validate resolves and reports; estimate adds plan counts;
// run validates then refuses until the Phase 1 solver lands.
#include "strikecem/app/CLI.hpp"

#include <iostream>
#include <string>
#include <vector>

#include "strikecem/core/Config.hpp"

namespace strikecem::cli {
namespace {

constexpr const char* kUsage =
    "usage: strikecem [--schema PATH] <validate|estimate|run> config.json\n"
    "  validate  resolve config, check mesh reference, print hash\n"
    "  estimate  validate plus frequency/direction/sample counts\n"
    "  run       full pipeline (PO solver: Phase 1)\n";

int fail(const std::exception& e, ExitCode code) {
    std::cerr << "strikecem: error: " << e.what() << "\n";
    return static_cast<int>(code);
}

void print_warnings(const SamplePlan& plan) {
    for (const auto& w : plan.warnings) std::cerr << "strikecem: warning: " << w << "\n";
}

int cmd_validate(const ResolvedConfig& rc) {
    std::cout << "scem_schema_version: " << rc.schema_version << "\n";
    std::cout << "resolved_config_hash: " << rc.hash << "\n";
    std::cout << "model: " << rc.value["model"]["path"].get<std::string>() << "\n";
    std::cout << "status: ok\n";
    return 0;
}

int cmd_estimate(const ResolvedConfig& rc) {
    SamplePlan plan = build_sample_plan(rc.value);
    print_warnings(plan);
    cmd_validate(rc);
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
        if (command == "validate") return cmd_validate(rc);
        if (command == "estimate") return cmd_estimate(rc);
        std::cerr << "strikecem: error: PO solver not implemented in this build (Phase 1)\n";
        return static_cast<int>(ExitCode::Solver);
    } catch (const ConfigError& e) {
        return fail(e, ExitCode::Config);
    } catch (const MeshError& e) {
        return fail(e, ExitCode::Mesh);
    }
}

} // namespace strikecem::cli

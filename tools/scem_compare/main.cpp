#include <iostream>
#include <string>

#include "strikecem/analysis/Comparison.hpp"
#include "strikecem/api/RcsQueryEngine.hpp"

using namespace strikecem;
using namespace strikecem::analysis;

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout << "Usage: scem_compare <database_a> <database_b> [options]\n"
                      << "Options:\n"
                      << "  --tolerance-db <val>   Maximum acceptable dBsm difference (default: 1.0)\n"
                      << "  --json                 Output comparison report in JSON format\n";
            return 0;
        }
    }

    if (argc < 3) {
        std::cerr << "Usage: scem_compare <database_a> <database_b> [options] (use --help for options)\n";
        return 1;
    }

    const std::string path_a = argv[1];
    const std::string path_b = argv[2];
    double tolerance_db = 1.0;
    bool output_json = false;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tolerance-db" && i + 1 < argc) {
            tolerance_db = std::stod(argv[++i]);
        } else if (arg == "--json") {
            output_json = true;
        }
    }

    try {
        auto engine_a = RcsQueryEngine::open_file(path_a);
        auto engine_b = RcsQueryEngine::open_file(path_b);

        ComparisonMetrics m = compare_databases(*engine_a, *engine_b, tolerance_db);

        if (output_json) {
            nlohmann::json report = comparison_to_json(m);
            report["database_a"] = path_a;
            report["database_b"] = path_b;
            report["tolerance_dbsm"] = tolerance_db;
            std::cout << report.dump(2) << "\n";
        } else {
            std::cout << "=== StrikeCEM Database Comparison ===\n";
            std::cout << "Baseline (A):          " << path_a << "\n";
            std::cout << "Target   (B):          " << path_b << "\n";
            std::cout << "Points compared:       " << m.sample_count << "\n";
            std::cout << "Tolerance threshold:   " << tolerance_db << " dBsm\n";
            std::cout << "-------------------------------------\n";
            std::cout << "Max Absolute Error:    " << m.max_absolute_error_dbsm << " dBsm\n";
            std::cout << "Mean Absolute Error:   " << m.mean_absolute_error_dbsm << " dBsm\n";
            std::cout << "RMSE:                  " << m.rmse_dbsm << " dBsm\n";
            std::cout << "Complex Correlation:   " << m.complex_correlation << "\n";
            std::cout << "Peak RCS Delta:        " << m.peak_rcs_delta_dbsm << " dBsm\n";
            std::cout << "-------------------------------------\n";
            std::cout << "Status:                " << (m.passed ? "PASSED" : "FAILED (exceeds tolerance)") << "\n";
        }

        return m.passed ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "scem_compare error: " << e.what() << "\n";
        return 1;
    }
}

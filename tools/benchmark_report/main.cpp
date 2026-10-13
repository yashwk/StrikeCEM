#include <iostream>
#include <string>
#include <vector>

#include "strikecem/analysis/BenchmarkReport.hpp"

using namespace strikecem::analysis;

int main(int argc, char** argv) {
    bool output_json = false;
    std::vector<std::string> search_paths;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: scem_benchreport [path_or_dir...] [options]\n"
                      << "Aggregates StrikeCEM benchmark profiles and reports scaling throughput.\n\n"
                      << "Options:\n"
                      << "  --json         Output report in JSON format\n"
                      << "  --ascii        Render ASCII throughput bar chart (default)\n"
                      << "  --dir <path>   Search directory for benchmark profiles\n"
                      << "  --help, -h     Show this help message\n";
            return 0;
        } else if (arg == "--json") {
            output_json = true;
        } else if (arg == "--ascii") {
            output_json = false;
        } else if (arg == "--dir" && i + 1 < argc) {
            search_paths.push_back(argv[++i]);
        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "Unknown option: " << arg << " (use --help for usage)\n";
            return 1;
        } else {
            search_paths.push_back(arg);
        }
    }

    try {
        const auto agg = load_and_aggregate(search_paths);
        if (output_json) {
            std::cout << benchmark_aggregation_to_json(agg).dump(2) << "\n";
        } else {
            std::cout << render_benchmark_ascii(agg) << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "scem_benchreport error: " << e.what() << "\n";
        return 1;
    }
}

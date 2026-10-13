#include <iostream>
#include <string>

#include "strikecem/analysis/Comparison.hpp"
#include "strikecem/api/RcsQueryEngine.hpp"

using namespace strikecem;
using namespace strikecem::analysis;

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout << "Usage: scem_quicklook <database_path> [options]\n"
                      << "Options:\n"
                      << "  --cut <azimuth|elevation>  Extract a 1D angle cut\n"
                      << "  --angle <deg>              Fixed angle for the cut (default: 0.0)\n"
                      << "  --freq <hz>                Frequency in Hz (default: first in DB)\n"
                      << "  --pol <channel>            Polarization channel (default: first in DB)\n"
                      << "  --step <deg>               Cut step in degrees (default: 1.0)\n"
                      << "  --ascii                    Render ASCII visual bar chart for cut\n"
                      << "  --json                     Output JSON format\n";
            return 0;
        }
    }

    if (argc < 2) {
        std::cerr << "Usage: scem_quicklook <database_path> [options] (use --help for options)\n";
        return 1;
    }

    const std::string db_path = argv[1];
    std::string cut_type;
    double fixed_angle = 0.0;
    double freq_hz = 0.0;
    std::string pol;
    double step_deg = 1.0;
    bool ascii_plot = false;
    bool output_json = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cut" && i + 1 < argc) cut_type = argv[++i];
        else if (arg == "--angle" && i + 1 < argc) fixed_angle = std::stod(argv[++i]);
        else if (arg == "--freq" && i + 1 < argc) freq_hz = std::stod(argv[++i]);
        else if (arg == "--pol" && i + 1 < argc) pol = argv[++i];
        else if (arg == "--step" && i + 1 < argc) step_deg = std::stod(argv[++i]);
        else if (arg == "--ascii") ascii_plot = true;
        else if (arg == "--json") output_json = true;
    }

    try {
        auto engine = RcsQueryEngine::open_file(db_path);

        if (freq_hz == 0.0 && !engine->frequencies().empty()) {
            freq_hz = engine->frequencies().front();
        }
        if (pol.empty() && !engine->polarizations().empty()) {
            pol = engine->polarizations().front();
        }

        if (cut_type.empty()) {
            DatabaseSummary sum = summarize_database(*engine);
            if (output_json) {
                std::cout << summary_to_json(sum).dump(2) << "\n";
            } else {
                std::cout << "=== StrikeCEM Database Summary: " << db_path << " ===\n";
                if (!sum.design_id.empty()) {
                    std::cout << "Design ID:       " << sum.design_id << " (rev: " << sum.design_revision << ")\n";
                }
                std::cout << "Directions:      " << sum.direction_count << "\n";
                std::cout << "Frequencies:     " << sum.frequencies_hz.size() << " (";
                for (size_t i = 0; i < sum.frequencies_hz.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << (sum.frequencies_hz[i] / 1e9) << " GHz";
                }
                std::cout << ")\n";
                std::cout << "Polarizations:   ";
                for (size_t i = 0; i < sum.polarizations.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << sum.polarizations[i];
                }
                std::cout << "\n";
                std::cout << "RCS Range:       [" << sum.min_rcs_dbsm << " dBsm, " << sum.max_rcs_dbsm << " dBsm]\n";
                std::cout << "Mean RCS:        " << sum.mean_rcs_dbsm << " dBsm\n";
                std::cout << "Peak RCS:        " << sum.max_rcs_dbsm << " dBsm at az=" << sum.peak_azimuth_deg
                          << " deg, el=" << sum.peak_elevation_deg << " deg, pol=" << sum.peak_polarization << "\n";
            }
        } else {
            DatabaseCut cut = extract_cut(*engine, cut_type, fixed_angle, freq_hz, pol, step_deg);
            if (output_json) {
                nlohmann::json j;
                j["cut_type"] = cut.cut_type;
                j["fixed_angle_deg"] = cut.fixed_angle_deg;
                j["frequency_hz"] = cut.frequency_hz;
                j["polarization"] = cut.polarization;
                nlohmann::json samples = nlohmann::json::array();
                for (const auto& s : cut.samples) {
                    samples.push_back({{"angle_deg", s.angle_deg}, {"rcs_sqm", s.rcs_sqm}, {"rcs_dbsm", s.rcs_dbsm}});
                }
                j["samples"] = samples;
                std::cout << j.dump(2) << "\n";
            } else if (ascii_plot) {
                std::cout << render_ascii_cut(cut);
            } else {
                std::cout << "Angle(deg)\tRCS(dBsm)\tRCS(m^2)\n";
                for (const auto& s : cut.samples) {
                    std::cout << s.angle_deg << "\t" << s.rcs_dbsm << "\t" << s.rcs_sqm << "\n";
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "scem_quicklook error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

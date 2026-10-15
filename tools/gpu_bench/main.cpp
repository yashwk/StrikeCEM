#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numbers>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/GpuPO.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

using namespace strikecem;

namespace {

NormalizedMesh subdivide_1to4(const NormalizedMesh& in) {
    NormalizedMesh out;
    out.vertices = in.vertices;

    std::map<std::pair<uint32_t, uint32_t>, uint32_t> edge_midpoints;
    auto get_mid = [&](uint32_t i1, uint32_t i2) -> uint32_t {
        const auto key = i1 < i2 ? std::make_pair(i1, i2) : std::make_pair(i2, i1);
        auto it = edge_midpoints.find(key);
        if (it != edge_midpoints.end()) return it->second;
        const geom::Vec3d mid = (in.vertices[i1] + in.vertices[i2]) * 0.5;
        const uint32_t new_idx = static_cast<uint32_t>(out.vertices.size());
        out.vertices.push_back(mid);
        edge_midpoints[key] = new_idx;
        return new_idx;
    };

    for (const auto& tri : in.triangles) {
        const uint32_t v0 = tri[0], v1 = tri[1], v2 = tri[2];
        const uint32_t m01 = get_mid(v0, v1);
        const uint32_t m12 = get_mid(v1, v2);
        const uint32_t m20 = get_mid(v2, v0);

        out.triangles.push_back({v0, m01, m20});
        out.triangles.push_back({v1, m12, m01});
        out.triangles.push_back({v2, m20, m12});
        out.triangles.push_back({m01, m12, m20});
    }

    // Compute normals and areas
    out.normals.resize(out.triangles.size());
    out.areas.resize(out.triangles.size());
    for (size_t t = 0; t < out.triangles.size(); ++t) {
        const auto& tri = out.triangles[t];
        const geom::Vec3d& a = out.vertices[tri[0]];
        const geom::Vec3d& b = out.vertices[tri[1]];
        const geom::Vec3d& c = out.vertices[tri[2]];
        const geom::Vec3d e1 = b - a;
        const geom::Vec3d e2 = c - a;
        geom::Vec3d cross = geom::cross(e1, e2);
        const double len = cross.length();
        out.areas[t] = len * 0.5;
        if (len > 1e-15) cross = cross * (1.0 / len);
        out.normals[t] = cross;
    }
    out.report = in.report;
    out.report.triangle_count = out.triangles.size();
    out.report.vertex_count = out.vertices.size();
    return out;
}

SamplePlan make_sweep_plan(double freq_hz, size_t num_azimuths) {
    SamplePlan plan;
    plan.frequencies_hz = {freq_hz};
    plan.polarizations = {"HH", "VV"};
    uint32_t id = 0;
    const double step = 360.0 / static_cast<double>(num_azimuths);
    for (size_t i = 0; i < num_azimuths; ++i) {
        const double az = static_cast<double>(i) * step;
        const double el = 0.0;
        plan.directions.push_back({id++, az, el, direction_from_az_el(az, el)});
    }
    return plan;
}

nlohmann::json make_solver_config(const std::string& precision, const std::string& accelerator,
                                 unsigned threads = 1, int device_id = 0,
                                 bool shadowing = false, int max_bounces = 1) {
    return nlohmann::json{
        {"solver", {
            {"type", "PO"},
            {"precision", precision},
            {"rcs_units", "dBsm"},
            {"po_options", {
                {"edge_correction", "none"},
                {"curvature_correction", false},
                {"shadowing", shadowing},
                {"max_bounces", max_bounces}
            }}
        }},
        {"physics", {
            {"monostatic", true},
            {"incident_amplitude", 1.0},
            {"polarization_basis", "linear"},
            {"phase_reference", {0.0, 0.0, 0.0}}
        }},
        {"frequency", {
            {"medium", {
                {"epsilon_r", 1.0},
                {"mu_r", 1.0},
                {"sigma", 0.0}
            }}
        }},
        {"execution", {
            {"accelerator", accelerator},
            {"cuda_device_id", device_id},
            {"cpu_threads", threads}
        }}
    };
}

} // namespace

int main(int argc, char** argv) {
    std::string mesh_path = "examples/f15/f15.obj";
    int subdiv_level = 0;
    size_t num_azimuths = 360;
    double freq_hz = 10.0e9; // 10 GHz X-band
    std::string json_output;
    int iters = 2;

    std::string material_choice = "pec";
    int max_bounces = 1;
    bool shadowing = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mesh" && i + 1 < argc) {
            mesh_path = argv[++i];
        } else if (arg == "--subdiv" && i + 1 < argc) {
            subdiv_level = std::stoi(argv[++i]);
        } else if (arg == "--dirs" && i + 1 < argc) {
            num_azimuths = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--freq" && i + 1 < argc) {
            freq_hz = std::stod(argv[++i]);
        } else if (arg == "--json" && i + 1 < argc) {
            json_output = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "--material" && i + 1 < argc) {
            material_choice = argv[++i];
        } else if (arg == "--bounces" && i + 1 < argc) {
            max_bounces = std::stoi(argv[++i]);
        } else if (arg == "--shadowing") {
            shadowing = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: scem_gpubench [options]\n"
                      << "Comprehensive native CUDA PO benchmark tool\n\n"
                      << "Options:\n"
                      << "  --mesh <path>     Input 3D model (default: examples/f15/f15.obj)\n"
                      << "  --subdiv <N>      Subdivide mesh 4^N times (0 to 4)\n"
                      << "  --dirs <N>        Azimuth directions (default: 360)\n"
                      << "  --freq <Hz>       Frequency in Hz (default: 10e9)\n"
                      << "  --material <type> Material wall: pec, ram (SIBC), coated (default: pec)\n"
                      << "  --bounces <N>     Max SBR ray bounces: 1, 2, or 3 (default: 1)\n"
                      << "  --shadowing       Enable GO line-of-sight self-shadowing\n"
                      << "  --iters <N>       Benchmark iterations (default: 2)\n"
                      << "  --json <path>     Write results as JSON\n"
                      << "  --help, -h        Show help\n";
            return 0;
        }
    }

    std::cout << "========================================================================\n"
              << "          StrikeCEM Native CUDA PO Solver Performance Benchmark         \n"
              << "========================================================================\n";

    if (!cuda::cuda_available()) {
        std::cerr << "error: no CUDA device available on this system!\n";
        return 1;
    }

    const auto dev_info = cuda::cuda_device_info(0);
    std::cout << "GPU Device:    " << dev_info.name << " (Compute " << dev_info.compute_major
              << "." << dev_info.compute_minor << ")\n"
              << "VRAM:          " << (dev_info.total_bytes / (1024 * 1024)) << " MiB total, "
              << (dev_info.free_bytes / (1024 * 1024)) << " MiB free\n"
              << "Frequency:     " << (freq_hz / 1.0e9) << " GHz (lambda = "
              << (299792458.0 / freq_hz * 100.0) << " cm)\n"
              << "Azimuth Dirs:  " << num_azimuths << " (HH + VV -> " << (num_azimuths * 2)
              << " samples)\n";

    // Load mesh
    nlohmann::json model_cfg = {
        {"model", {{"path", mesh_path}, {"units", "m"}, {"mesh_repair", "report"}}},
        {"mesh", {{"max_aspect_ratio", 100.0}}},
        {"frequency", {
            {"frequency_hz", freq_hz},
            {"medium", {{"epsilon_r", 1.0}, {"mu_r", 1.0}, {"sigma", 0.0}}}
        }}
    };
    NormalizedMesh mesh = load_normalized_mesh(model_cfg, ".", "1.0");
    std::cout << "Base Mesh:     " << mesh.report.triangle_count << " triangles, "
              << mesh.report.vertex_count << " vertices (" << mesh_path << ")\n";

    for (int l = 0; l < subdiv_level; ++l) {
        mesh = subdivide_1to4(mesh);
    }
    if (subdiv_level > 0) {
        std::cout << "Subdivided (" << subdiv_level << "): " << mesh.report.triangle_count
                  << " triangles, " << mesh.report.vertex_count << " vertices\n";
    }

    const SamplePlan plan = make_sweep_plan(freq_hz, num_azimuths);
    const size_t total_work_units = plan.frequencies_hz.size() * plan.directions.size();
    const double facet_directions = static_cast<double>(mesh.report.triangle_count) * total_work_units;

    std::cout << "Workload:      " << std::fixed << std::setprecision(2)
              << (facet_directions / 1.0e6) << " Million Facet-Directions per run\n\n";

    // Configure materials and multi-bounce options
    MaterialOptions mat_opts;
    if (material_choice == "ram" || material_choice == "sibc") {
        MaterialModel ram_model;
        ram_model.tag = "ram_sibc";
        ram_model.type = WallType::Impedance;
        ram_model.surface_impedance = {1.0, 0.0};
        mat_opts = MaterialOptions{true, WallType::Impedance, ram_model};
    } else if (material_choice == "coated") {
        MaterialModel coated_model;
        coated_model.tag = "ram_coated";
        coated_model.type = WallType::Coated;
        CoatingLayer layer;
        layer.thickness_m = 0.002;
        layer.medium.eps_r = {4.0, -0.8};
        layer.medium.mu_r = {1.0, 0.0};
        coated_model.layers.push_back(layer);
        mat_opts = MaterialOptions{true, WallType::Coated, coated_model};
    }

    GoOptions go_opts;
    go_opts.shadowing = shadowing;
    go_opts.max_bounces = max_bounces;

    std::cout << "Material:      " << material_choice << (mat_opts.enabled ? " (active)" : " (PEC)") << "\n"
              << "Multi-Bounce:  " << max_bounces << " bounce(s)\n"
              << "Shadowing:     " << (shadowing ? "true" : "false") << "\n";

    // 1. Warm up CUDA context
    std::cout << "Warming up CUDA device context... ";
    std::cout.flush();
    const auto t_warm0 = std::chrono::steady_clock::now();
    const auto cfg_warm = make_solver_config("float64", "cuda", 1, 0, shadowing, max_bounces);
    (void)solve_po(mesh, plan, cfg_warm, {}, go_opts, mat_opts);
    const double t_warm = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_warm0).count();
    std::cout << "Done (" << (t_warm * 1000.0) << " ms init time)\n\n";

    // 2. Run CPU Thread Scaling
    std::vector<unsigned> cpu_threads = {1, 4, 8, 16};
    std::map<unsigned, double> cpu_times;
    PoResult cpu_ref;

    for (unsigned th : cpu_threads) {
        const auto cfg_cpu = make_solver_config("float64", "cpu", th, 0, shadowing, max_bounces);
        double best_sec = 1e9;
        for (int it = 0; it < iters; ++it) {
            const auto t0 = std::chrono::steady_clock::now();
            PoResult res = solve_po(mesh, plan, cfg_cpu, {}, go_opts, mat_opts);
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (sec < best_sec) best_sec = sec;
            if (th == 16 && it == 0) cpu_ref = std::move(res);
        }
        cpu_times[th] = best_sec;
    }

    // 3. Run GPU Precision Modes
    std::vector<std::string> precisions = {"float64", "mixed", "float32"};
    std::map<std::string, double> gpu_times;
    std::map<std::string, PoResult> gpu_results;

    for (const auto& prec : precisions) {
        const auto cfg_gpu = make_solver_config(prec, "cuda", 1, 0, shadowing, max_bounces);
        double best_sec = 1e9;
        for (int it = 0; it < iters + 1; ++it) {
            const auto t0 = std::chrono::steady_clock::now();
            PoResult res = solve_po(mesh, plan, cfg_gpu, {}, go_opts, mat_opts);
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (sec < best_sec) best_sec = sec;
            if (it == 0) gpu_results[prec] = std::move(res);
        }
        gpu_times[prec] = best_sec;
    }

    // 4. Compute Accuracy Errors vs CPU Reference
    std::map<std::string, double> max_rcs_err;
    std::map<std::string, double> max_field_rel_err;
    for (const auto& prec : precisions) {
        const auto& gres = gpu_results[prec];
        double max_r_diff = 0.0;
        double max_s_diff = 0.0;
        for (size_t i = 0; i < cpu_ref.samples.size(); ++i) {
            const double r_diff = std::abs(cpu_ref.samples[i].rcs_sqm - gres.samples[i].rcs_sqm);
            if (r_diff > max_r_diff) max_r_diff = r_diff;

            const double cpu_mag = std::abs(cpu_ref.samples[i].scattering);
            const double s_diff = std::abs(cpu_ref.samples[i].scattering - gres.samples[i].scattering);
            const double rel = cpu_mag > 1e-12 ? s_diff / cpu_mag : s_diff;
            if (rel > max_s_diff) max_s_diff = rel;
        }
        max_rcs_err[prec] = max_r_diff;
        max_field_rel_err[prec] = max_s_diff;
    }

    // 5. Display Benchmark Results Table
    std::cout << "------------------------------------------------------------------------------------\n"
              << " Backend / Precision    | Time (ms)  | Speedup vs 1T | Speedup vs 16T | Throughput (M/s)\n"
              << "------------------------------------------------------------------------------------\n";

    for (unsigned th : cpu_threads) {
        const double sec = cpu_times[th];
        const double sp_1t = cpu_times[1] / sec;
        const double sp_16t = cpu_times[16] / sec;
        const double tput = (facet_directions / sec) / 1.0e6;
        std::cout << " CPU (" << std::setw(2) << th << " threads)       | "
                  << std::setw(8) << std::fixed << std::setprecision(2) << (sec * 1000.0) << " ms | "
                  << std::setw(11) << std::setprecision(1) << sp_1t << "x | "
                  << std::setw(12) << std::setprecision(1) << sp_16t << "x | "
                  << std::setw(14) << std::setprecision(1) << tput << "\n";
    }
    std::cout << "------------------------------------------------------------------------------------\n";

    for (const auto& prec : precisions) {
        const double sec = gpu_times[prec];
        const double sp_1t = cpu_times[1] / sec;
        const double sp_16t = cpu_times[16] / sec;
        const double tput = (facet_directions / sec) / 1.0e6;
        std::string label = " CUDA GPU (" + prec + ")";
        while (label.length() < 24) label += " ";
        std::cout << label << "| "
                  << std::setw(8) << std::fixed << std::setprecision(2) << (sec * 1000.0) << " ms | "
                  << std::setw(11) << std::setprecision(1) << sp_1t << "x | "
                  << std::setw(12) << std::setprecision(1) << sp_16t << "x | "
                  << std::setw(14) << std::setprecision(1) << tput << "\n";
    }
    std::cout << "------------------------------------------------------------------------------------\n\n";

    std::cout << "Numerical Precision & Accuracy Validation (vs CPU Double-Precision Reference):\n";
    for (const auto& prec : precisions) {
        std::cout << "  * GPU " << std::setw(7) << prec << " -> Max RCS Error: "
                  << std::scientific << std::setprecision(2) << max_rcs_err[prec]
                  << " sqm | Max Relative Field Diff: " << max_field_rel_err[prec] << "\n";
    }
    std::cout << "\n";

    // Save JSON if requested
    if (!json_output.empty()) {
        nlohmann::json j;
        j["gpu_device"] = dev_info.name;
        j["triangles"] = mesh.report.triangle_count;
        j["directions"] = num_azimuths;
        j["samples"] = num_azimuths * 2;
        j["frequency_hz"] = freq_hz;
        j["facet_directions"] = facet_directions;
        for (unsigned th : cpu_threads) {
            j["cpu_seconds"][std::to_string(th)] = cpu_times[th];
        }
        for (const auto& prec : precisions) {
            j["gpu_seconds"][prec] = gpu_times[prec];
            j["speedup_vs_1t"][prec] = cpu_times[1] / gpu_times[prec];
            j["speedup_vs_16t"][prec] = cpu_times[16] / gpu_times[prec];
            j["throughput_mfacets_sec"][prec] = (facet_directions / gpu_times[prec]) / 1.0e6;
            j["max_rcs_err_sqm"][prec] = max_rcs_err[prec];
            j["max_field_rel_err"][prec] = max_field_rel_err[prec];
        }
        std::ofstream out(json_output);
        out << j.dump(2) << "\n";
        std::cout << "Wrote detailed benchmark results to: " << json_output << "\n";
    }

    return 0;
}

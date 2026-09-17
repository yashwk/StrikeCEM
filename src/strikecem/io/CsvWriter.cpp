// Flat-row CSV writer plus JSON sidecar (OUTPUT_FORMAT.md).
#include "strikecem/io/CsvWriter.hpp"

#include <chrono>

#include "strikecem/solvers/GpuPO.hpp"
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#ifdef __linux__
#include <unistd.h>
#endif

namespace strikecem {
namespace {

namespace fs = std::filesystem;

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string hostname() {
#ifdef __linux__
    char buf[256] = {};
    if (::gethostname(buf, sizeof(buf) - 1) == 0) return buf;
#endif
    return "unknown";
}

} // namespace

void write_csv_and_sidecar(const ResolvedConfig& rc, const SamplePlan& plan,
                           const NormalizedMesh& mesh, const PoResult& result,
                           const DesignerIdentity& designer, const FringeOptions& fringe) {
    const auto& output = rc.value["output"];
    const fs::path csv_path(output["path"].get<std::string>());
    const int precision = output["precision"].get<int>();
    const std::string completed = utc_now();

    {
        std::ofstream out(csv_path, std::ios::trunc);
        if (!out) throw OutputError("cannot open output file: " + csv_path.string());
        out << "sample_id,azimuth_deg,elevation_deg,frequency_hz,polarization,"
               "scattering_real,scattering_imag,rcs_sqm,rcs_dBsm,valid,status\n";
        out << std::fixed << std::setprecision(precision) << std::boolalpha;
        for (const PoSampleResult& row : result.samples) {
            const Direction& dir = plan.directions[row.direction_id];
            const double freq = plan.frequencies_hz[row.frequency_id];
            const std::string& pol = plan.polarizations[row.pol_id];
            out << row.sample_id << ',' << dir.azimuth_deg << ',' << dir.elevation_deg << ','
                << freq << ',' << pol << ',' << row.scattering.real() << ','
                << row.scattering.imag() << ',' << row.rcs_sqm << ','
                << rcs_sqm_to_dbsm(row.rcs_sqm) << ',' << true << ',' << "ok\n";
        }
        out.flush();
        if (!out) throw OutputError("failed while writing output file: " + csv_path.string());
    }

    nlohmann::json sidecar;
    sidecar["scem_schema_version"] = rc.schema_version;
    sidecar["output_format_version"] = "1.0";
    sidecar["tool"] = {{"name", "strikecem"},
                       {"version", SCEM_VERSION},
                       {"git_revision", SCEM_GIT_REVISION},
                       {"build_type", SCEM_BUILD_TYPE},
                       {"compiler", std::string(SCEM_COMPILER_ID) + " " + SCEM_COMPILER_VERSION}};
    sidecar["host"] = {{"hostname", hostname()}, {"device", "cpu"}};
    sidecar["completed_utc"] = completed;
    sidecar["config_hash"] = rc.hash;
    sidecar["geometry_hash"] = mesh.geometry_hash;
    sidecar["normalized_mesh_hash"] = mesh.normalized_mesh_hash;
    sidecar["sample_plan"] = {{"frequencies", plan.frequencies_hz.size()},
                              {"directions", plan.directions.size()},
                              {"polarizations", plan.polarizations},
                              {"samples", plan.sample_count()},
                              {"hash", plan.hash}};
    sidecar["solver"] = {{"type", rc.value["solver"]["type"]},
                         {"precision", rc.value["solver"]["precision"]},
                         {"edge_correction",
                          rc.value["solver"]["po_options"].value("edge_correction", "none")},
                         {"fringe_edges", fringe.enabled && fringe.edges != nullptr
                                              ? fringe.edges->edges.size()
                                              : 0},
                         {"deterministic", rc.value["execution"]["deterministic"]},
                         {"backend", rc.value["execution"].value("accelerator", "cpu")},
                         {"device", cuda::active_device_name(rc.value)}};
    sidecar["mesh"] = {{"path", rc.value["model"]["path"]},
                       {"units", rc.value["model"]["units"]},
                       {"vertices", mesh.report.vertex_count},
                       {"triangles", mesh.report.triangle_count},
                       {"surface_area_m2", mesh.report.total_area_m2},
                       {"repaired", mesh.repaired}};
    if (!designer.design_id.empty()) {
        sidecar["designer"] = {{"design_id", designer.design_id},
                               {"revision", designer.revision},
                               {"export_id", designer.export_id},
                               {"export_manifest_hash",
                                designer.manifest_hash.empty()
                                    ? nlohmann::json(nullptr)
                                    : nlohmann::json(designer.manifest_hash)}};
    } else {
        sidecar["designer"] = nullptr;
    }
    sidecar["warnings"] = nlohmann::json::array();
    for (const auto& w : plan.warnings) sidecar["warnings"].push_back(w);
    for (const auto& w : result.warnings) sidecar["warnings"].push_back(w);
    sidecar["resolved_config"] = rc.value;

    const fs::path sidecar_path = csv_path.string() + ".json";
    {
        std::ofstream out(sidecar_path, std::ios::trunc);
        if (!out) throw OutputError("cannot open sidecar file: " + sidecar_path.string());
        out << sidecar.dump(2) << "\n";
        out.flush();
        if (!out) throw OutputError("failed while writing sidecar file: " + sidecar_path.string());
    }
}

} // namespace strikecem

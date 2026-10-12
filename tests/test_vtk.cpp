// Tests for Track 3: Diagnostic & Field Visualization Export (VTK .vtu).
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "strikecem/app/CLI.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/diagnostics/VtkWriter.hpp"
#include "strikecem/io/MeshLoader.hpp"

namespace {

namespace fs = std::filesystem;

struct Workdir {
    fs::path dir = fs::temp_directory_path() / ("scem_vtk_test_" + std::to_string(::getpid()) + "_" +
                                                std::to_string(reinterpret_cast<uintptr_t>(this)));
    Workdir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
        fs::create_directories(dir, ec);
    }
    ~Workdir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

int call_cli(const std::vector<std::string>& arg_list) {
    std::vector<std::string> storage = {"strikecem"};
    for (const auto& a : arg_list) storage.push_back(a);
    std::vector<char*> argv;
    for (auto& s : storage) argv.push_back(s.data());
    return strikecem::cli::run(static_cast<int>(argv.size()), argv.data());
}

nlohmann::json make_vtk_config(const fs::path& out_path, const fs::path& vtu_path) {
    nlohmann::json cfg;
    cfg["scem_schema_version"] = "1.0";
    cfg["model"] = {{"path", std::string(std::string(SCEM_FIXTURE_DIR) +
                                         "/../../examples/plate.stl")}};
    cfg["frequency"] = {{"frequency_hz", 10e9}};
    cfg["angles"]["azimuth"] = {{"start", 0}, {"stop", 0}, {"step", 1}};
    cfg["angles"]["elevation"] = {{"start", -90}, {"stop", -90}, {"step", 1}};
    cfg["polarization"] = {"HH", "VV"};
    cfg["solver"] = {{"type", "PO"}};
    cfg["output"] = {
        {"path", out_path.string()},
        {"format", "hdf5"},
        {"diagnostics", {
            {"vtk_export_path", vtu_path.string()},
            {"export_facet_currents", true},
            {"export_lit_status", true},
            {"export_material_ids", true},
            {"sample_index", 0}
        }}
    };
    return cfg;
}

fs::path write_json(const fs::path& path, const nlohmann::json& j) {
    std::ofstream out(path);
    out << j.dump(2);
    return path;
}

} // namespace

TEST(Vtk, BroadsideCurrentMagnitudeMatchesTheory) {
    Workdir w;
    const fs::path cfg_file = write_json(w.dir / "config.json", make_vtk_config(w.dir / "out.h5", w.dir / "diag.vtu"));
    const auto rc = strikecem::load_config(cfg_file.string(), SCEM_SCHEMA_PATH);
    const auto mesh = strikecem::load_normalized_mesh(rc.value, w.dir, rc.schema_version);

    // Direction at broadside: az = 0, el = -90 -> k_hat = (0, 0, -1)
    const strikecem::Direction dir{0, 0.0, -90.0, {0.0, 0.0, -1.0}};
    const auto diag = strikecem::compute_facet_diagnostics(mesh, dir, 10e9, "HH");

    ASSERT_EQ(diag.facets.size(), mesh.triangles.size());
    // Normal incidence on PEC plate: J_s = 2 * (n x H_inc) where H_inc = 1 / eta0
    // Expected current magnitude = 2.0 / 376.73031346... ~= 5.3088e-3 A/m
    const double expected_j = 2.0 / strikecem::kEta0;

    for (const auto& f : diag.facets) {
        EXPECT_EQ(f.lit, 1u);
        EXPECT_NEAR(f.current_magnitude, expected_j, 1e-6);
        EXPECT_GT(f.area_m2, 0.0);
    }
}

TEST(Vtk, BackfaceIlluminationIsShadowed) {
    Workdir w;
    const fs::path cfg_file = write_json(w.dir / "config.json", make_vtk_config(w.dir / "out.h5", w.dir / "diag.vtu"));
    const auto rc = strikecem::load_config(cfg_file.string(), SCEM_SCHEMA_PATH);
    const auto mesh = strikecem::load_normalized_mesh(rc.value, w.dir, rc.schema_version);

    // Direction from behind plate: az = 0, el = +90 -> k_hat = (0, 0, +1)
    const strikecem::Direction back_dir{0, 0.0, 90.0, {0.0, 0.0, 1.0}};
    const auto diag = strikecem::compute_facet_diagnostics(mesh, back_dir, 10e9, "HH");

    for (const auto& f : diag.facets) {
        EXPECT_EQ(f.lit, 0u);
        EXPECT_DOUBLE_EQ(f.current_magnitude, 0.0);
    }
}

TEST(Vtk, WriteVtuValidatesStructure) {
    Workdir w;
    const fs::path vtu_out = w.dir / "plate_diagnostics.vtu";
    const fs::path cfg_file = write_json(w.dir / "config.json", make_vtk_config(w.dir / "out.h5", vtu_out));
    const auto rc = strikecem::load_config(cfg_file.string(), SCEM_SCHEMA_PATH);
    const auto mesh = strikecem::load_normalized_mesh(rc.value, w.dir, rc.schema_version);

    const strikecem::Direction dir{0, 0.0, -90.0, {0.0, 0.0, -1.0}};
    const auto diag = strikecem::compute_facet_diagnostics(mesh, dir, 10e9, "HH");

    strikecem::write_vtu(vtu_out.string(), mesh, diag);
    ASSERT_TRUE(fs::exists(vtu_out));

    std::ifstream in(vtu_out);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    EXPECT_NE(content.find("<VTKFile type=\"UnstructuredGrid\""), std::string::npos);
    EXPECT_NE(content.find("<Points>"), std::string::npos);
    EXPECT_NE(content.find("<Cells>"), std::string::npos);
    EXPECT_NE(content.find("<CellData>"), std::string::npos);
    EXPECT_NE(content.find("Name=\"LitStatus\""), std::string::npos);
    EXPECT_NE(content.find("Name=\"CurrentMagnitude\""), std::string::npos);
    EXPECT_NE(content.find("Name=\"CurrentPhaseDeg\""), std::string::npos);
    EXPECT_NE(content.find("Name=\"SurfaceCurrentReal\""), std::string::npos);
    EXPECT_NE(content.find("Name=\"SurfaceCurrentImag\""), std::string::npos);
    EXPECT_NE(content.find("Name=\"MaterialId\""), std::string::npos);
    EXPECT_NE(content.find("Name=\"FacetArea\""), std::string::npos);
    EXPECT_NE(content.find("Name=\"FacetNormal\""), std::string::npos);
    EXPECT_NE(content.find("</VTKFile>"), std::string::npos);
}

TEST(Vtk, CliSubcommandExportsVtu) {
    Workdir w;
    const fs::path vtu_out = w.dir / "cli_diag.vtu";
    const fs::path cfg_file = write_json(w.dir / "config.json", make_vtk_config(w.dir / "out.h5", w.dir / "ignore.vtu"));

    ASSERT_EQ(call_cli({"vtk", "--output", vtu_out.string(), cfg_file.string()}), 0);
    ASSERT_TRUE(fs::exists(vtu_out));
    EXPECT_GT(fs::file_size(vtu_out), 100u);
}

TEST(Vtk, RunExportsVtuDiagnosticsAutomatically) {
    Workdir w;
    const fs::path out_h5 = w.dir / "run_out.h5";
    const fs::path vtu_out = w.dir / "auto_diag.vtu";
    const fs::path cfg_file = write_json(w.dir / "config.json", make_vtk_config(out_h5, vtu_out));

    ASSERT_EQ(call_cli({"run", cfg_file.string()}), 0);
    ASSERT_TRUE(fs::exists(out_h5));
    ASSERT_TRUE(fs::exists(vtu_out));
    EXPECT_GT(fs::file_size(vtu_out), 100u);
}

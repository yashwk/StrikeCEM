#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <gtest/gtest.h>
#include <H5Cpp.h>
#include <nlohmann/json.hpp>

#include "strikecem/app/CLI.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/core/Conventions.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/runtime/Estimate.hpp"
#include "strikecem/solvers/GpuPO.hpp"
#include "strikecem/solvers/Material.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace {

strikecem::MaterialModel dielectric_single() {
    strikecem::MaterialModel m;
    m.tag = "glass";
    m.type = strikecem::WallType::Dielectric;
    m.table.push_back({1e9, {{4.0, 0.0}, {1.0, 0.0}}, 0.0});
    return m;
}

TEST(Material, SingleEntryIsConstant) {
    const auto m = dielectric_single();
    EXPECT_NO_THROW(strikecem::validate_material(m));
    for (double f : {1e6, 1e9, 18e9}) {
        const auto e = strikecem::evaluate_material(m, f);
        EXPECT_DOUBLE_EQ(e.eps_r.real(), 4.0);
        EXPECT_DOUBLE_EQ(e.eps_r.imag(), 0.0);
    }
}

TEST(Material, LinearMidpointExact) {
    strikecem::MaterialModel m;
    m.tag = "interp";
    m.type = strikecem::WallType::Dielectric;
    m.table.push_back({1e9, {{2.0, -0.2}, {1.0, 0.0}}, 0.0});
    m.table.push_back({3e9, {{4.0, -0.6}, {1.0, 0.0}}, 2.0});
    const auto e = strikecem::evaluate_material(m, 2e9);
    EXPECT_DOUBLE_EQ(e.eps_r.real(), 3.0);
    EXPECT_DOUBLE_EQ(e.mu_r.real(), 1.0);
    const double eps0 = 1.0 / (376.73031346177066 * 299792458.0);
    const double folded = 1.0 / (2.0 * std::numbers::pi * 2e9 * eps0);
    EXPECT_DOUBLE_EQ(e.eps_r.imag(), -0.4 - folded);
}

TEST(Material, RangeAndOrderEnforced) {
    auto m = dielectric_single();
    m.table.push_back({0.5e9, {{4.0, 0.0}, {1.0, 0.0}}, 0.0});
    EXPECT_THROW(strikecem::validate_material(m), std::invalid_argument);
    auto ok = dielectric_single();
    ok.table.push_back({2e9, {{4.0, 0.0}, {1.0, 0.0}}, 0.0});
    EXPECT_THROW(strikecem::evaluate_material(ok, 0.5e9), std::invalid_argument);
    EXPECT_THROW(strikecem::evaluate_material(ok, 3e9), std::invalid_argument);
    EXPECT_THROW(strikecem::evaluate_material(ok, 0.0), std::invalid_argument);
    EXPECT_NO_THROW(strikecem::evaluate_material(ok, 1e9));
    EXPECT_NO_THROW(strikecem::evaluate_material(ok, 2e9));
}

TEST(Material, ShapePolicy) {
    EXPECT_THROW(strikecem::validate_material({}), std::invalid_argument);
    strikecem::MaterialModel pec;
    pec.tag = "metal";
    pec.type = strikecem::WallType::Pec;
    EXPECT_NO_THROW(strikecem::validate_material(pec));
    EXPECT_THROW(strikecem::evaluate_material(pec, 1e9), std::invalid_argument);
    auto pec_table = pec;
    pec_table.table.push_back({1e9, {{1.0, 0.0}, {1.0, 0.0}}, 0.0});
    EXPECT_THROW(strikecem::validate_material(pec_table), std::invalid_argument);
    strikecem::MaterialModel coated;
    coated.tag = "paint";
    coated.type = strikecem::WallType::Coated;
    EXPECT_THROW(strikecem::validate_material(coated), std::invalid_argument);
    coated.layers.push_back({0.001, {{3.0, -0.1}, {1.0, 0.0}}});
    EXPECT_NO_THROW(strikecem::validate_material(coated));
    coated.layers.push_back({0.0, {{3.0, 0.0}, {1.0, 0.0}}});
    EXPECT_THROW(strikecem::validate_material(coated), std::invalid_argument);
    auto mixed = dielectric_single();
    mixed.layers.push_back({0.001, {{3.0, 0.0}, {1.0, 0.0}}});
    EXPECT_THROW(strikecem::validate_material(mixed), std::invalid_argument);
}

TEST(Material, GainRejected) {
    auto m = dielectric_single();
    m.table[0].medium.eps_r = {4.0, 0.5};
    EXPECT_THROW(strikecem::validate_material(m), std::invalid_argument);
    auto neg = dielectric_single();
    neg.table[0].sigma = -1.0;
    EXPECT_THROW(strikecem::validate_material(neg), std::invalid_argument);
}

TEST(Material, ConductivityFoldsWithSign) {
    strikecem::MaterialEntry e{1e9, {{1.0, 0.0}, {1.0, 0.0}}, 1.0};
    const auto m = strikecem::effective_medium(e, 1e9);
    EXPECT_DOUBLE_EQ(m.eps_r.real(), 1.0);
    EXPECT_LT(m.eps_r.imag(), 0.0);
    const double eps0 = 1.0 / (376.73031346177066 * 299792458.0);
    EXPECT_DOUBLE_EQ(m.eps_r.imag(), -1.0 / (2.0 * std::numbers::pi * 1e9 * eps0));
}

TEST(Material, BarePecEmptyStack) {
    const auto r = strikecem::coated_pec_reflection({}, {}, 1.0, 1.0);
    EXPECT_DOUBLE_EQ(r.r_te.real(), -1.0);
    EXPECT_DOUBLE_EQ(r.r_te.imag(), 0.0);
    EXPECT_DOUBLE_EQ(r.r_tm.real(), 1.0);
    EXPECT_DOUBLE_EQ(r.r_tm.imag(), 0.0);
}

TEST(Material, HalfWaveAbsentee) {
    const strikecem::CoatingLayer layer{0.25, {{4.0, 0.0}, {1.0, 0.0}}};
    const auto r = strikecem::coated_pec_reflection({}, {layer}, 1.0, 1.0);
    EXPECT_NEAR(std::abs(r.r_te + 1.0), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(r.r_tm - 1.0), 0.0, 1e-12);
}

TEST(Material, ThinDegenerate) {
    const strikecem::CoatingLayer layer{1e-6, {{6.25, -0.09}, {1.0, 0.0}}};
    const auto r = strikecem::coated_pec_reflection({}, {layer}, std::cos(0.3), 1.0);
    EXPECT_LT(std::abs(r.r_te + 1.0), 1e-3);
    EXPECT_LT(std::abs(r.r_tm - 1.0), 1e-3);
}

TEST(Material, AsymmetricTwoLayer) {
    const strikecem::CoatingLayer outer{1.0 / 8.0, {{4.0, 0.0}, {1.0, 0.0}}};
    const strikecem::CoatingLayer inner{1.0 / 24.0, {{9.0, 0.0}, {1.0, 0.0}}};
    const auto r = strikecem::coated_pec_reflection({}, {outer, inner}, 1.0, 1.0);
    EXPECT_NEAR(r.r_te.real(), -0.28, 1e-9);
    EXPECT_NEAR(r.r_te.imag(), -0.96, 1e-9);
    EXPECT_NEAR(r.r_tm.real(), 0.28, 1e-9);
    EXPECT_NEAR(r.r_tm.imag(), 0.96, 1e-9);
    const auto swapped = strikecem::coated_pec_reflection({}, {inner, outer}, 1.0, 1.0);
    EXPECT_GT(std::abs(swapped.r_te - r.r_te), 0.1);
}

TEST(Material, LosslessUnitMagnitude) {
    const strikecem::CoatingLayer a{0.13, {{4.0, 0.0}, {1.0, 0.0}}};
    const strikecem::CoatingLayer b{0.07, {{2.25, 0.0}, {1.0, 0.0}}};
    for (double cos_i : {1.0, 0.8, 0.4}) {
        const auto r = strikecem::coated_pec_reflection({}, {a, b}, cos_i, 1.0);
        EXPECT_NEAR(std::abs(r.r_te), 1.0, 1e-9);
        EXPECT_NEAR(std::abs(r.r_tm), 1.0, 1e-9);
    }
    const strikecem::CoatingLayer lossy{0.13, {{4.0, -1.0}, {1.0, 0.0}}};
    const auto rl = strikecem::coated_pec_reflection({}, {lossy}, 0.9, 1.0);
    EXPECT_LE(std::abs(rl.r_te), 1.0 + 1e-12);
    EXPECT_LE(std::abs(rl.r_tm), 1.0 + 1e-12);
    EXPECT_TRUE(std::isfinite(rl.r_te.real() + rl.r_tm.imag()));
}

TEST(Material, CoatingBadInputsThrow) {
    const strikecem::CoatingLayer good{0.1, {{4.0, 0.0}, {1.0, 0.0}}};
    EXPECT_THROW(strikecem::coated_pec_reflection({}, {good}, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(strikecem::coated_pec_reflection({}, {good}, 2.0, 1.0), std::invalid_argument);
    const strikecem::CoatingLayer flat{0.0, {{4.0, 0.0}, {1.0, 0.0}}};
    EXPECT_THROW(strikecem::coated_pec_reflection({}, {flat}, 1.0, 1.0), std::invalid_argument);
    const strikecem::CoatingLayer gain{0.1, {{4.0, 0.5}, {1.0, 0.0}}};
    EXPECT_THROW(strikecem::coated_pec_reflection({}, {gain}, 1.0, 1.0), std::invalid_argument);
}

namespace fs = std::filesystem;

TEST(Material, MaterialFromJson) {
    // Dielectric single medium
    nlohmann::json j_diel = {
        {"tag", "glass"},
        {"type", "dielectric"},
        {"frequency_table", {
            {{"frequency_hz", 1e9}, {"epsilon_r", 4.0}}
        }}
    };
    auto m_diel = strikecem::material_from_json(j_diel);
    EXPECT_EQ(m_diel.tag, "glass");
    EXPECT_EQ(m_diel.type, strikecem::WallType::Dielectric);
    ASSERT_EQ(m_diel.table.size(), 1u);
    EXPECT_DOUBLE_EQ(m_diel.table[0].medium.eps_r.real(), 4.0);

    // Dielectric frequency table
    nlohmann::json j_table = {
        {"tag", "tab"},
        {"type", "dielectric"},
        {"frequency_table", {
            {{"frequency_hz", 1e9}, {"epsilon_r", {4.0, -0.1}}},
            {{"frequency_hz", 2e9}, {"epsilon_r", {4.5, -0.2}}}
        }}
    };
    auto m_tab = strikecem::material_from_json(j_table);
    ASSERT_EQ(m_tab.table.size(), 2u);
    EXPECT_DOUBLE_EQ(m_tab.table[1].frequency_hz, 2e9);

    // Coated
    nlohmann::json j_coat = {
        {"tag", "paint"},
        {"type", "coated"},
        {"layers", {
            {{"thickness_m", 0.001}, {"epsilon_r", 3.0}}
        }}
    };
    auto m_coat = strikecem::material_from_json(j_coat);
    EXPECT_EQ(m_coat.tag, "paint");
    EXPECT_EQ(m_coat.type, strikecem::WallType::Coated);
    ASSERT_EQ(m_coat.layers.size(), 1u);
    EXPECT_DOUBLE_EQ(m_coat.layers[0].thickness_m, 0.001);

    // PEC
    nlohmann::json j_pec = {{"tag", "metal"}, {"type", "pec"}};
    auto m_pec = strikecem::material_from_json(j_pec);
    EXPECT_EQ(m_pec.tag, "metal");
    EXPECT_EQ(m_pec.type, strikecem::WallType::Pec);

    // Invalid cases throw ConfigError
    EXPECT_THROW(strikecem::material_from_json({{"type", "pec"}}), strikecem::ConfigError);
    EXPECT_THROW(strikecem::material_from_json({{"tag", "x"}}), strikecem::ConfigError);
    EXPECT_THROW(strikecem::material_from_json({{"tag", "x"}, {"type", "unknown"}}), strikecem::ConfigError);
    EXPECT_THROW(strikecem::material_from_json({{"tag", "x"}, {"type", "coated"}}), strikecem::ConfigError);
    EXPECT_THROW(strikecem::material_from_json({{"tag", "x"}, {"type", "coated"}, {"layers", {{{"thickness_m", -1.0}}}}}), strikecem::ConfigError);
    EXPECT_THROW(strikecem::material_from_json({{"tag", "x"}, {"type", "dielectric"}}), strikecem::ConfigError);
}

TEST(Material, ResolveMaterialsBehavior) {
    strikecem::NormalizedMesh mesh;
    mesh.triangles.push_back({0, 1, 2});
    mesh.triangles.push_back({0, 2, 3});
    mesh.triangle_tags = {"pec_tag", "glass_tag"};

    nlohmann::json cfg = {
        {"materials", {
            {"tag_coverage", "strict"},
            {"default_material", "pec_tag"},
            {"definitions", {
                {{"tag", "pec_tag"}, {"type", "pec"}},
                {{"tag", "glass_tag"}, {"type", "dielectric"}, {"frequency_table", {{{"frequency_hz", 1e9}, {"epsilon_r", 4.0}}}}}
            }}
        }}
    };

    // Strict with exact tags matches
    auto res = strikecem::resolve_materials(cfg, mesh);
    EXPECT_TRUE(res.has_materials);
    EXPECT_EQ(res.non_pec_facets, 1u);
    EXPECT_EQ(res.facet_material_indices[0], 1u); // tag_to_model["pec_tag"]
    EXPECT_EQ(res.facet_material_indices[1], 2u); // tag_to_model["glass_tag"]

    // Strict with untagged facet throws ConfigError
    mesh.triangle_tags = {"", "glass_tag"};
    EXPECT_THROW(strikecem::resolve_materials(cfg, mesh), strikecem::ConfigError);

    // Strict with unmapped tag throws ConfigError
    mesh.triangle_tags = {"unknown_tag", "glass_tag"};
    EXPECT_THROW(strikecem::resolve_materials(cfg, mesh), strikecem::ConfigError);

    // Warn coverage with untagged facet logs warning
    cfg["materials"]["tag_coverage"] = "warn";
    mesh.triangle_tags = {"", "glass_tag"};
    res = strikecem::resolve_materials(cfg, mesh);
    EXPECT_FALSE(res.warnings.empty());

    // Warn coverage with unmapped tag logs warning
    mesh.triangle_tags = {"unknown_tag", "glass_tag"};
    res = strikecem::resolve_materials(cfg, mesh);
    EXPECT_FALSE(res.warnings.empty());

    // Allow untagged silently assigns default material
    cfg["materials"]["tag_coverage"] = "allow_untagged";
    mesh.triangle_tags = {"", "unknown_tag"};
    res = strikecem::resolve_materials(cfg, mesh);
    EXPECT_TRUE(res.warnings.empty());
    EXPECT_EQ(res.facet_material_indices[0], res.facet_material_indices[1]);

    // Missing default material throws ConfigError
    cfg["materials"]["default_material"] = "does_not_exist";
    EXPECT_THROW(strikecem::resolve_materials(cfg, mesh), strikecem::ConfigError);

    // Duplicate tag definition throws ConfigError
    nlohmann::json dup_cfg = {
        {"materials", {
            {"tag_coverage", "strict"},
            {"default_material", "pec"},
            {"definitions", {
                {{"tag", "glass"}, {"type", "dielectric"}, {"frequency_table", {{{"frequency_hz", 1e9}, {"epsilon_r", 4.0}}}}},
                {{"tag", "glass"}, {"type", "dielectric"}, {"frequency_table", {{{"frequency_hz", 1e9}, {"epsilon_r", 9.0}}}}}
            }}
        }}
    };
    EXPECT_THROW(strikecem::resolve_materials(dup_cfg, mesh), strikecem::ConfigError);
}

TEST(Material, MultiMaterialPlateSummation) {
    const auto rc = strikecem::load_config(std::string(SCEM_FIXTURE_DIR) + "/valid_minimal.json", SCEM_SCHEMA_PATH);

    // 2-triangle flat plate in z=0
    strikecem::NormalizedMesh mesh;
    mesh.vertices = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}};
    mesh.triangles = {{0, 1, 2}, {0, 2, 3}};
    mesh.normals = {{0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}};
    mesh.areas = {0.5, 0.5};

    strikecem::SamplePlan plan;
    plan.frequencies_hz = {10e9};
    plan.directions.push_back({0, 0.0, -90.0, strikecem::direction_from_az_el(0.0, -90.0)});
    plan.polarizations = {"HH"};

    strikecem::MaterialModel pec_model;
    pec_model.tag = "pec";
    pec_model.type = strikecem::WallType::Pec;

    strikecem::MaterialModel glass_model;
    glass_model.tag = "glass";
    glass_model.type = strikecem::WallType::Dielectric;
    glass_model.table.push_back({10e9, {{4.0, 0.0}, {1.0, 0.0}}, 0.0});

    // 1. Both PEC
    strikecem::MaterialOptions pec_opts;
    pec_opts.enabled = true;
    pec_opts.models = {pec_model, glass_model};
    pec_opts.facet_material_indices = {0, 0};
    auto res_pec = strikecem::solve_po(mesh, plan, rc.value, {}, {}, pec_opts);

    // 2. Both Glass (Fresnel reflection coeff at normal incidence is (1 - sqrt(4))/(1 + sqrt(4)) = -1/3, so 1/3 of PEC)
    strikecem::MaterialOptions glass_opts;
    glass_opts.enabled = true;
    glass_opts.models = {pec_model, glass_model};
    glass_opts.facet_material_indices = {1, 1};
    auto res_glass = strikecem::solve_po(mesh, plan, rc.value, {}, {}, glass_opts);

    EXPECT_NEAR(std::abs(res_glass.samples[0].scattering) / std::abs(res_pec.samples[0].scattering), 1.0 / 3.0, 1e-6);

    // 3. Mixed: 1 PEC facet + 1 Glass facet
    // By symmetry and equal area, field should be 0.5 * S_pec + 0.5 * S_glass = 0.5 * (1 + 1/3) * S_pec = (2/3) * S_pec
    strikecem::MaterialOptions mixed_opts;
    mixed_opts.enabled = true;
    mixed_opts.models = {pec_model, glass_model};
    mixed_opts.facet_material_indices = {0, 1};
    auto res_mixed = strikecem::solve_po(mesh, plan, rc.value, {}, {}, mixed_opts);

    EXPECT_NEAR(std::abs(res_mixed.samples[0].scattering) / std::abs(res_pec.samples[0].scattering), 2.0 / 3.0, 1e-6);
}

TEST(Material, MultiMaterialGuards) {
    const auto rc = strikecem::load_config(std::string(SCEM_FIXTURE_DIR) + "/valid_minimal.json", SCEM_SCHEMA_PATH);

    strikecem::NormalizedMesh mesh;
    mesh.vertices = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}};
    mesh.triangles = {{0, 1, 2}, {0, 2, 3}};
    mesh.normals = {{0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}};
    mesh.areas = {0.5, 0.5};

    strikecem::SamplePlan plan;
    plan.frequencies_hz = {10e9};
    plan.directions.push_back({0, 0.0, -90.0, strikecem::direction_from_az_el(0.0, -90.0)});
    plan.polarizations = {"HH"};

    auto resolved = rc.value;
    resolved["execution"]["accelerator"] = "cuda";

    strikecem::MaterialModel pec_model;
    pec_model.tag = "pec";
    pec_model.type = strikecem::WallType::Pec;

    strikecem::MaterialModel glass_model;
    glass_model.tag = "glass";
    glass_model.type = strikecem::WallType::Dielectric;
    glass_model.table.push_back({10e9, {{4.0, 0.0}, {1.0, 0.0}}, 0.0});

    strikecem::MaterialOptions mixed_opts;
    mixed_opts.enabled = true;
    mixed_opts.models = {pec_model, glass_model};
    mixed_opts.facet_material_indices = {0, 1};

    // CUDA with non-PEC facet throws CudaError
    EXPECT_THROW(strikecem::solve_po(mesh, plan, resolved, {}, {}, mixed_opts), strikecem::cuda::CudaError);

    // Multi-bounce GO with non-PEC facet throws invalid_argument
    resolved["execution"]["accelerator"] = "cpu";
    strikecem::GoOptions go{false, 2};
    EXPECT_THROW(strikecem::solve_po(mesh, plan, resolved, {}, go, mixed_opts), std::invalid_argument);
}

TEST(Material, OutputProvenanceCsvAndHdf5) {
    const fs::path dir = fs::temp_directory_path() / ("scem_mat_prov_" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    // Write a small obj with 2 groups
    const fs::path obj_path = dir / "mesh.obj";
    {
        std::ofstream out(obj_path);
        out << "v 0 0 0\n"
            << "v 1 0 0\n"
            << "v 1 1 0\n"
            << "v 0 1 0\n"
            << "g pec_part\n"
            << "f 1 2 3\n"
            << "g glass_part\n"
            << "f 1 3 4\n";
    }

    nlohmann::json cfg = {
        {"scem_schema_version", "1.0"},
        {"model", {{"path", obj_path.string()}}},
        {"frequency", {{"frequency_hz", 10e9}}},
        {"angles", {
            {"azimuth", {{"start", 0}, {"stop", 0}, {"step", 1}}},
            {"elevation", {{"start", -90}, {"stop", -90}, {"step", 1}}}
        }},
        {"polarization", {"HH"}},
        {"materials", {
            {"tag_coverage", "strict"},
            {"default_material", "pec"},
            {"definitions", {
                {{"tag", "pec_part"}, {"type", "pec"}},
                {{"tag", "glass_part"}, {"type", "dielectric"}, {"frequency_table", {{{"frequency_hz", 10e9}, {"epsilon_r", 4.0}}}}}
            }}
        }},
        {"solver", {{"type", "PO"}}},
        {"output", {{"path", (dir / "out.csv").string()}, {"format", "csv"}}}
    };

    const fs::path csv_cfg_path = dir / "cfg_csv.json";
    {
        std::ofstream out(csv_cfg_path);
        out << cfg.dump(2);
    }

    const std::string argv0 = "strikecem";
    char* argv_csv[] = {const_cast<char*>(argv0.c_str()), const_cast<char*>("run"),
                        const_cast<char*>(csv_cfg_path.c_str())};
    EXPECT_EQ(strikecem::cli::run(3, argv_csv), 0);

    // Verify CSV sidecar
    std::ifstream sidecar_in(dir / "out.csv.json");
    ASSERT_TRUE(sidecar_in.good());
    const auto sidecar = nlohmann::json::parse(sidecar_in);
    EXPECT_TRUE(sidecar["solver"]["materials"]["enabled"].get<bool>());
    EXPECT_EQ(sidecar["solver"]["materials"]["tag_coverage"], "strict");
    EXPECT_EQ(sidecar["solver"]["materials"]["models_count"], 3u);

    // Now run with HDF5
    cfg["output"] = {{"path", (dir / "out.h5").string()}, {"format", "hdf5"}};
    const fs::path h5_cfg_path = dir / "cfg_h5.json";
    {
        std::ofstream out(h5_cfg_path);
        out << cfg.dump(2);
    }
    char* argv_h5[] = {const_cast<char*>(argv0.c_str()), const_cast<char*>("run"),
                       const_cast<char*>(h5_cfg_path.c_str())};
    EXPECT_EQ(strikecem::cli::run(3, argv_h5), 0);

    // Verify HDF5 provenance attributes
    H5::Exception::dontPrint();
    H5::H5File h5file((dir / "out.h5").string(), H5F_ACC_RDONLY);
    H5::Attribute mat_en = h5file.openAttribute("materials_enabled");
    uint64_t enabled_val = 0;
    mat_en.read(H5::PredType::NATIVE_UINT64, &enabled_val);
    EXPECT_EQ(enabled_val, 1u);

    H5::Attribute mod_count = h5file.openAttribute("material_models_count");
    uint64_t count_val = 0;
    mod_count.read(H5::PredType::NATIVE_UINT64, &count_val);
    EXPECT_EQ(count_val, 3u);
    h5file.close();

    fs::remove_all(dir, ec);
}

TEST(Material, EstimatorWarnsOnNonPec) {
    const auto rc = strikecem::load_config(std::string(SCEM_FIXTURE_DIR) + "/valid_minimal.json", SCEM_SCHEMA_PATH);
    auto cfg = rc.value;
    cfg["materials"] = {
        {"default_material", "pec"},
        {"definitions", {
            {{"tag", "glass"}, {"type", "dielectric"}, {"frequency_table", {{{"frequency_hz", 10e9}, {"epsilon_r", 4.0}}}}}
        }}
    };
    strikecem::SamplePlan plan;
    plan.frequencies_hz = {10e9};
    strikecem::NormalizedMesh mesh;
    mesh.report.bbox_min = {0, 0, 0};
    mesh.report.bbox_max = {1, 1, 0};

    const auto est = strikecem::estimate_resources(cfg, plan, mesh, {});
    bool saw_warning = false;
    for (const auto& w : est.warnings) {
        if (w.find("material surfaces are excluded from benchmark profile calibrations") != std::string::npos) {
            saw_warning = true;
        }
    }
    EXPECT_TRUE(saw_warning);
}

} // namespace


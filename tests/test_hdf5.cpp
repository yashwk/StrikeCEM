// HDF5 writer/reader tests: write-read round trip, CSV parity, provenance,
// and reader rejection of incomplete or incompatible databases.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include <H5Cpp.h>
#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/CsvWriter.hpp"
#include "strikecem/io/Hdf5Reader.hpp"
#include "strikecem/io/Hdf5Writer.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace {

namespace fs = std::filesystem;

struct Workdir {
    fs::path dir = fs::temp_directory_path() / "scem_hdf5_test";
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

struct Pipeline {
    strikecem::ResolvedConfig rc;
    strikecem::SamplePlan plan;
    strikecem::NormalizedMesh mesh;
    strikecem::PoResult result;
};

Pipeline run_pipeline(const nlohmann::json& cfg, const fs::path& config_path) {
    {
        std::ofstream out(config_path);
        out << cfg.dump(2);
    }
    Pipeline p;
    p.rc = strikecem::load_config(config_path.string(), SCEM_SCHEMA_PATH);
    p.plan = strikecem::build_sample_plan(p.rc.value);
    p.mesh = strikecem::load_normalized_mesh(p.rc.value, config_path.parent_path(),
                                             p.rc.schema_version);
    p.result = strikecem::solve_po(p.mesh, p.plan, p.rc.value);
    return p;
}

nlohmann::json make_config(const std::string& output_path, const std::string& format) {
    nlohmann::json cfg;
    cfg["scem_schema_version"] = "1.0";
    cfg["model"] = {{"path", std::string(std::string(SCEM_FIXTURE_DIR) +
                                         "/../../examples/plate.stl")}};
    cfg["frequency"] = {{"frequency_hz", 10e9}};
    cfg["angles"]["azimuth"] = {{"start", 0}, {"stop", 90}, {"step", 90}};
    cfg["angles"]["elevation"] = {{"start", -90}, {"stop", -90}, {"step", 1}};
    cfg["polarization"] = {"HH", "VV"};
    cfg["solver"] = {{"type", "PO"}};
    cfg["output"] = {{"path", output_path}, {"format", format}};
    return cfg;
}

TEST(Hdf5, WriteReadRoundTripFloat64) {
    Workdir work;
    Pipeline p = run_pipeline(make_config((work.dir / "out.h5").string(), "hdf5"),
                              work.dir / "scem.json");
    ASSERT_EQ(p.result.samples.size(), 4u); // 1 freq x 2 dirs x 2 pols
    strikecem::write_hdf5_output(p.rc, p.plan, p.mesh, p.result);
    auto db = strikecem::read_hdf5_database((work.dir / "out.h5").string());
    EXPECT_NO_THROW(strikecem::validate_hdf5_complete(db));
    EXPECT_EQ(db.azimuth_deg, (std::vector<double>{0.0, 90.0}));
    EXPECT_EQ(db.frequencies_hz, (std::vector<double>{10e9}));
    EXPECT_EQ(db.polarizations, (std::vector<std::string>{"HH", "VV"}));
    ASSERT_EQ(db.rows.size(), p.result.samples.size());
    for (size_t i = 0; i < db.rows.size(); ++i) {
        const auto& row = db.rows[i];
        const auto& src = p.result.samples[i];
        EXPECT_EQ(row.sample_id, src.sample_id);
        EXPECT_EQ(row.direction_index, src.direction_id);
        EXPECT_EQ(row.frequency_index, src.frequency_id);
        EXPECT_EQ(row.polarization_index, src.pol_id);
        EXPECT_DOUBLE_EQ(row.scattering_real, src.scattering.real());
        EXPECT_DOUBLE_EQ(row.scattering_imag, src.scattering.imag());
        EXPECT_DOUBLE_EQ(row.rcs_sqm, src.rcs_sqm);
        EXPECT_TRUE(row.valid);
        EXPECT_EQ(row.status, "ok");
    }
    EXPECT_EQ(db.config_hash, p.rc.hash);
    EXPECT_EQ(db.output_format_version, "1.0");
    EXPECT_EQ(db.scem_schema_version, "1.0");
    EXPECT_EQ(db.solver_precision, "float64");
}

TEST(Hdf5, CsvParity) {
    // Same plan to CSV (6 decimals) and HDF5: values agree to print precision.
    Workdir work;
    Pipeline h5 = run_pipeline(make_config((work.dir / "out.h5").string(), "hdf5"),
                               work.dir / "h5.json");
    Pipeline csv = run_pipeline(make_config((work.dir / "out.csv").string(), "csv"),
                                work.dir / "csv.json");
    strikecem::write_hdf5_output(h5.rc, h5.plan, h5.mesh, h5.result);
    strikecem::write_csv_and_sidecar(csv.rc, csv.plan, csv.mesh, csv.result);
    const auto db = strikecem::read_hdf5_database((work.dir / "out.h5").string());
    std::ifstream in(work.dir / "out.csv");
    std::string header;
    std::getline(in, header);
    std::string line;
    size_t row = 0;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::vector<std::string> f;
        std::string tok;
        while (std::getline(ls, tok, ',')) f.push_back(tok);
        ASSERT_EQ(f.size(), 11u);
        EXPECT_NEAR(std::stod(f[5]), db.rows[row].scattering_real, 1e-6);
        EXPECT_NEAR(std::stod(f[6]), db.rows[row].scattering_imag, 1e-6);
        EXPECT_NEAR(std::stod(f[7]), db.rows[row].rcs_sqm,
                    1e-6 * std::max(db.rows[row].rcs_sqm, 1.0));
        ++row;
    }
    EXPECT_EQ(row, db.rows.size());
}

TEST(Hdf5, RejectsIncompleteProgress) {
    Workdir work;
    Pipeline p = run_pipeline(make_config((work.dir / "out.h5").string(), "hdf5"),
                              work.dir / "scem.json");
    strikecem::write_hdf5_output(p.rc, p.plan, p.mesh, p.result);
    {
        H5::H5File file((work.dir / "out.h5").string(), H5F_ACC_RDWR);
        H5::Group progress = file.openGroup("/progress");
        H5::DataSet flags = progress.openDataSet("completed_chunks");
        const uint8_t zero[1] = {0};
        const hsize_t count[1] = {1}, start[1] = {0};
        const H5::DataSpace mem(1, count);
        H5::DataSpace space = flags.getSpace();
        space.selectHyperslab(H5S_SELECT_SET, count, start);
        flags.write(zero, H5::PredType::NATIVE_UINT8, mem, space);
        file.flush(H5F_SCOPE_GLOBAL);
        file.close();
    }
    auto db = strikecem::read_hdf5_database((work.dir / "out.h5").string());
    EXPECT_THROW(strikecem::validate_hdf5_complete(db), strikecem::ReaderError);
}

TEST(Hdf5, RejectsInvalidRow) {
    Workdir work;
    Pipeline p = run_pipeline(make_config((work.dir / "out.h5").string(), "hdf5"),
                              work.dir / "scem.json");
    strikecem::write_hdf5_output(p.rc, p.plan, p.mesh, p.result);
    {
        H5::H5File file((work.dir / "out.h5").string(), H5F_ACC_RDWR);
        H5::Group samples = file.openGroup("/samples");
        H5::DataSet valid = samples.openDataSet("valid");
        const uint8_t zero[1] = {0};
        const hsize_t count[1] = {1}, start[1] = {0};
        const H5::DataSpace mem(1, count);
        H5::DataSpace space = valid.getSpace();
        space.selectHyperslab(H5S_SELECT_SET, count, start);
        valid.write(zero, H5::PredType::NATIVE_UINT8, mem, space);
        file.flush(H5F_SCOPE_GLOBAL);
        file.close();
    }
    auto db = strikecem::read_hdf5_database((work.dir / "out.h5").string());
    EXPECT_THROW(strikecem::validate_hdf5_complete(db), strikecem::ReaderError);
}

TEST(Hdf5, RejectsUnknownMajorVersion) {
    Workdir work;
    Pipeline p = run_pipeline(make_config((work.dir / "out.h5").string(), "hdf5"),
                              work.dir / "scem.json");
    strikecem::write_hdf5_output(p.rc, p.plan, p.mesh, p.result);
    {
        H5::H5File file((work.dir / "out.h5").string(), H5F_ACC_RDWR);
        const H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
        H5::Attribute attr = file.openAttribute("output_format_version");
        const std::string bumped = "2.0";
        attr.write(type, bumped);
        file.flush(H5F_SCOPE_GLOBAL);
        file.close();
    }
    auto db = strikecem::read_hdf5_database((work.dir / "out.h5").string());
    EXPECT_THROW(strikecem::validate_hdf5_complete(db), strikecem::ReaderError);
}

} // namespace

// Tests for compact binary transport format (.scem):
// Round-trip serialization (float64 and float32), HDF5 and CSV parity,
// header and payload corruption detection, truncated file rejection,
// CLI execution integration, and distributed shard merge with .scem.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "strikecem/app/CLI.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/distributed/Shard.hpp"
#include "strikecem/io/BinaryTransport.hpp"
#include "strikecem/io/CsvWriter.hpp"
#include "strikecem/io/Hdf5Reader.hpp"
#include "strikecem/io/Hdf5Writer.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace {

namespace fs = std::filesystem;

struct Workdir {
    fs::path dir = fs::temp_directory_path() / ("scem_binary_test_" + std::to_string(::getpid()));
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

nlohmann::json make_config(const std::string& output_path, const std::string& format,
                           const std::string& precision = "float64") {
    nlohmann::json cfg;
    cfg["scem_schema_version"] = "1.0";
    cfg["model"] = {{"path", std::string(std::string(SCEM_FIXTURE_DIR) +
                                         "/../../examples/plate.stl")}};
    cfg["frequency"] = {{"frequency_hz", 10e9}};
    cfg["angles"]["azimuth"] = {{"start", 0}, {"stop", 90}, {"step", 90}};
    cfg["angles"]["elevation"] = {{"start", -90}, {"stop", -90}, {"step", 1}};
    cfg["polarization"] = {"HH", "VV"};
    cfg["solver"] = {{"type", "PO"}, {"precision", precision}};
    cfg["output"] = {{"path", output_path}, {"format", format}};
    return cfg;
}

TEST(BinaryTransport, WriteReadRoundTripFloat64) {
    Workdir work;
    Pipeline p = run_pipeline(make_config((work.dir / "out.scem").string(), "scem"),
                              work.dir / "cfg.json");
    ASSERT_EQ(p.result.samples.size(), 4u); // 1 freq x 2 dirs x 2 pols

    strikecem::write_scem_output(p.rc, p.plan, p.mesh, p.result);
    auto db = strikecem::read_scem_database((work.dir / "out.scem").string());
    EXPECT_NO_THROW(strikecem::validate_scem_complete(db));

    EXPECT_EQ(db.version, 1);
    EXPECT_EQ(db.dtype, 1);
    EXPECT_EQ(db.frequencies_hz, (std::vector<double>{10e9}));
    ASSERT_EQ(db.directions.size(), 2u);
    EXPECT_DOUBLE_EQ(db.directions[0].azimuth_deg, 0.0);
    EXPECT_DOUBLE_EQ(db.directions[1].azimuth_deg, 90.0);
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

    EXPECT_EQ(db.metadata["config_hash"], p.rc.hash);
    EXPECT_EQ(db.metadata["geometry_hash"], p.mesh.geometry_hash);
    EXPECT_EQ(db.metadata["normalized_mesh_hash"], p.mesh.normalized_mesh_hash);
    EXPECT_EQ(db.metadata["output_format_version"], "1.0");
    EXPECT_EQ(db.metadata["scem_schema_version"], "1.0");
}

TEST(BinaryTransport, WriteReadRoundTripFloat32) {
    Workdir work;
    Pipeline p = run_pipeline(make_config((work.dir / "out32.scem").string(), "scem", "float32"),
                              work.dir / "cfg32.json");
    ASSERT_EQ(p.result.samples.size(), 4u);

    strikecem::write_scem_output(p.rc, p.plan, p.mesh, p.result);
    auto db = strikecem::read_scem_database((work.dir / "out32.scem").string());
    EXPECT_NO_THROW(strikecem::validate_scem_complete(db));

    EXPECT_EQ(db.version, 1);
    EXPECT_EQ(db.dtype, 0); // float32
    ASSERT_EQ(db.rows.size(), p.result.samples.size());
    for (size_t i = 0; i < db.rows.size(); ++i) {
        const auto& row = db.rows[i];
        const auto& src = p.result.samples[i];
        EXPECT_EQ(row.sample_id, src.sample_id);
        EXPECT_NEAR(row.scattering_real, src.scattering.real(), 1e-4);
        EXPECT_NEAR(row.scattering_imag, src.scattering.imag(), 1e-4);
        EXPECT_NEAR(row.rcs_sqm, src.rcs_sqm, 1e-1);
        EXPECT_NEAR(row.rcs_dbsm, strikecem::rcs_sqm_to_dbsm(src.rcs_sqm), 1e-3);
        EXPECT_TRUE(row.valid);
        EXPECT_EQ(row.status, "ok");
    }
}

TEST(BinaryTransport, Hdf5NumericalParity) {
    Workdir work;
    Pipeline h5 = run_pipeline(make_config((work.dir / "out.h5").string(), "hdf5"),
                               work.dir / "h5.json");
    Pipeline bin = run_pipeline(make_config((work.dir / "out.scem").string(), "scem"),
                                work.dir / "bin.json");

    strikecem::write_hdf5_output(h5.rc, h5.plan, h5.mesh, h5.result);
    strikecem::write_scem_output(bin.rc, bin.plan, bin.mesh, bin.result);

    const auto h5_db = strikecem::read_hdf5_database((work.dir / "out.h5").string());
    const auto bin_db = strikecem::read_scem_database((work.dir / "out.scem").string());

    ASSERT_EQ(h5_db.rows.size(), bin_db.rows.size());
    for (size_t i = 0; i < h5_db.rows.size(); ++i) {
        const auto& h_r = h5_db.rows[i];
        const auto& b_r = bin_db.rows[i];
        EXPECT_EQ(h_r.sample_id, b_r.sample_id);
        EXPECT_EQ(h_r.direction_index, b_r.direction_index);
        EXPECT_EQ(h_r.frequency_index, b_r.frequency_index);
        EXPECT_EQ(h_r.polarization_index, b_r.polarization_index);
        EXPECT_DOUBLE_EQ(h_r.scattering_real, b_r.scattering_real);
        EXPECT_DOUBLE_EQ(h_r.scattering_imag, b_r.scattering_imag);
        EXPECT_DOUBLE_EQ(h_r.rcs_sqm, b_r.rcs_sqm);
        EXPECT_DOUBLE_EQ(h_r.rcs_dbsm, b_r.rcs_dbsm);
        EXPECT_EQ(h_r.valid, b_r.valid);
        EXPECT_EQ(h_r.status, b_r.status);
    }
}

TEST(BinaryTransport, CsvNumericalParity) {
    Workdir work;
    Pipeline csv = run_pipeline(make_config((work.dir / "out.csv").string(), "csv"),
                                work.dir / "csv.json");
    Pipeline bin = run_pipeline(make_config((work.dir / "out.scem").string(), "scem"),
                                work.dir / "bin.json");

    strikecem::write_csv_and_sidecar(csv.rc, csv.plan, csv.mesh, csv.result);
    strikecem::write_scem_output(bin.rc, bin.plan, bin.mesh, bin.result);

    const auto bin_db = strikecem::read_scem_database((work.dir / "out.scem").string());

    std::ifstream in(work.dir / "out.csv");
    std::string header, line;
    std::getline(in, header);
    size_t row_idx = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        ASSERT_LT(row_idx, bin_db.rows.size());
        std::istringstream ss(line);
        std::string sid_s, az_s, el_s, fq_s, pol_s, re_s, im_s, sqm_s, db_s, val_s, st_s;
        std::getline(ss, sid_s, ',');
        std::getline(ss, az_s, ',');
        std::getline(ss, el_s, ',');
        std::getline(ss, fq_s, ',');
        std::getline(ss, pol_s, ',');
        std::getline(ss, re_s, ',');
        std::getline(ss, im_s, ',');
        std::getline(ss, sqm_s, ',');
        std::getline(ss, db_s, ',');
        std::getline(ss, val_s, ',');
        std::getline(ss, st_s, ',');

        const auto& b_r = bin_db.rows[row_idx];
        EXPECT_EQ(std::stoull(sid_s), b_r.sample_id);
        EXPECT_NEAR(std::stod(re_s), b_r.scattering_real, 1e-5);
        EXPECT_NEAR(std::stod(im_s), b_r.scattering_imag, 1e-5);
        EXPECT_NEAR(std::stod(sqm_s), b_r.rcs_sqm, 1e-5);
        EXPECT_NEAR(std::stod(db_s), b_r.rcs_dbsm, 1e-4);
        ++row_idx;
    }
    EXPECT_EQ(row_idx, bin_db.rows.size());
}

TEST(BinaryTransport, CorruptedHeaderDetection) {
    Workdir work;
    Pipeline p = run_pipeline(make_config((work.dir / "out.scem").string(), "scem"),
                              work.dir / "cfg.json");
    strikecem::write_scem_output(p.rc, p.plan, p.mesh, p.result);

    // Corrupt magic bytes
    {
        fs::path bad_file = work.dir / "bad_magic.scem";
        fs::copy_file(work.dir / "out.scem", bad_file);
        std::fstream f(bad_file, std::ios::in | std::ios::out | std::ios::binary);
        char bad_magic[4] = {'B', 'A', 'D', '!'};
        f.write(bad_magic, 4);
        f.close();
        EXPECT_THROW(strikecem::read_scem_database(bad_file.string()), strikecem::ReaderError);
    }

    // Corrupt version
    {
        fs::path bad_file = work.dir / "bad_ver.scem";
        fs::copy_file(work.dir / "out.scem", bad_file);
        std::fstream f(bad_file, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(4, std::ios::beg);
        uint16_t bad_version = 99;
        f.write(reinterpret_cast<const char*>(&bad_version), sizeof(bad_version));
        f.close();
        EXPECT_THROW(strikecem::read_scem_database(bad_file.string()), strikecem::ReaderError);
    }

    // Corrupt endianness marker
    {
        fs::path bad_file = work.dir / "bad_endian.scem";
        fs::copy_file(work.dir / "out.scem", bad_file);
        std::fstream f(bad_file, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(8, std::ios::beg);
        uint32_t bad_endian = 0xdeadbeef;
        f.write(reinterpret_cast<const char*>(&bad_endian), sizeof(bad_endian));
        f.close();
        EXPECT_THROW(strikecem::read_scem_database(bad_file.string()), strikecem::ReaderError);
    }
}

TEST(BinaryTransport, CorruptedPayloadDetection) {
    Workdir work;
    Pipeline p = run_pipeline(make_config((work.dir / "out.scem").string(), "scem"),
                              work.dir / "cfg.json");
    strikecem::write_scem_output(p.rc, p.plan, p.mesh, p.result);

    fs::path bad_file = work.dir / "bad_payload.scem";
    fs::copy_file(work.dir / "out.scem", bad_file);

    // Corrupt a byte in payload (beyond header at offset 300)
    std::fstream f(bad_file, std::ios::in | std::ios::out | std::ios::binary);
    f.seekg(300, std::ios::beg);
    char byte_val = 0;
    f.read(&byte_val, 1);
    byte_val ^= 0xFF; // flip bits
    f.seekp(300, std::ios::beg);
    f.write(&byte_val, 1);
    f.close();

    EXPECT_THROW(strikecem::read_scem_database(bad_file.string()), strikecem::ReaderError);
}

TEST(BinaryTransport, TruncatedFileDetection) {
    Workdir work;
    Pipeline p = run_pipeline(make_config((work.dir / "out.scem").string(), "scem"),
                              work.dir / "cfg.json");
    strikecem::write_scem_output(p.rc, p.plan, p.mesh, p.result);

    fs::path trunc_file = work.dir / "trunc.scem";
    // Only copy header portion
    std::ifstream in(work.dir / "out.scem", std::ios::binary);
    std::vector<char> partial(200);
    in.read(partial.data(), 200);
    in.close();

    std::ofstream out(trunc_file, std::ios::binary);
    out.write(partial.data(), 200);
    out.close();

    EXPECT_THROW(strikecem::read_scem_database(trunc_file.string()), strikecem::ReaderError);
}

TEST(BinaryTransport, CheckExistingScem) {
    Workdir work;
    Pipeline p = run_pipeline(make_config((work.dir / "out.scem").string(), "scem"),
                              work.dir / "cfg.json");
    EXPECT_FALSE(strikecem::check_existing_scem(p.rc, p.plan, p.mesh,
                                                (work.dir / "out.scem").string()));

    strikecem::write_scem_output(p.rc, p.plan, p.mesh, p.result);
    EXPECT_TRUE(strikecem::check_existing_scem(p.rc, p.plan, p.mesh,
                                               (work.dir / "out.scem").string()));
}

TEST(BinaryTransport, CLIExecutionRun) {
    Workdir work;
    const fs::path cfg_path = work.dir / "run_scem.json";
    const fs::path out_path = work.dir / "run_out.scem";
    {
        std::ofstream out(cfg_path);
        out << make_config(out_path.string(), "scem").dump(2);
    }
    const std::vector<std::string> args = {
        "--schema", SCEM_SCHEMA_PATH, "run", cfg_path.string()
    };
    EXPECT_EQ(call_cli(args), 0);
    EXPECT_TRUE(fs::exists(out_path));

    auto db = strikecem::read_scem_database(out_path.string());
    EXPECT_NO_THROW(strikecem::validate_scem_complete(db));
    EXPECT_EQ(db.rows.size(), 4u);
}

TEST(BinaryTransport, ShardAndMergeRoundTripScem) {
    Workdir work;
    const fs::path cfg_path = work.dir / "cfg_scem.json";
    const fs::path mono_out = work.dir / "mono.scem";
    {
        std::ofstream out(cfg_path);
        out << make_config(mono_out.string(), "scem").dump(2);
    }
    const auto rc = strikecem::load_config(cfg_path.string(), SCEM_SCHEMA_PATH);
    const auto plan = strikecem::build_sample_plan(rc.value);
    const auto mesh = strikecem::load_normalized_mesh(rc.value, cfg_path.parent_path(), rc.schema_version);
    const auto mono_res = strikecem::solve_po(mesh, plan, rc.value);
    strikecem::write_scem_output(rc, plan, mesh, mono_res);

    // Create 2 shards with format="scem"
    const fs::path shard_dir = work.dir / "shards";
    const auto manifest = strikecem::create_shards(rc, plan, mesh, cfg_path.string(), 2,
                                                   shard_dir.string(), "test-run");
    EXPECT_EQ(manifest.shards.size(), 2u);
    EXPECT_EQ(manifest.output_format, "scem");

    // Execute both shards
    for (const auto& s : manifest.shards) {
        const std::vector<std::string> args = {
            "--schema", SCEM_SCHEMA_PATH, "run", s.config_path
        };
        EXPECT_EQ(call_cli(args), 0);
        EXPECT_TRUE(fs::exists(s.output_path));
    }

    // Merge shards into merged.scem
    const fs::path merged_out = work.dir / "merged.scem";
    const fs::path manifest_path = shard_dir / "shard_manifest.json";
    strikecem::merge_shards(manifest_path.string(), {}, merged_out.string(), SCEM_SCHEMA_PATH);

    EXPECT_TRUE(fs::exists(merged_out));
    auto mono_db = strikecem::read_scem_database(mono_out.string());
    auto merged_db = strikecem::read_scem_database(merged_out.string());

    EXPECT_NO_THROW(strikecem::validate_scem_complete(merged_db));
    ASSERT_EQ(mono_db.rows.size(), merged_db.rows.size());
    for (size_t i = 0; i < mono_db.rows.size(); ++i) {
        EXPECT_EQ(mono_db.rows[i].sample_id, merged_db.rows[i].sample_id);
        EXPECT_DOUBLE_EQ(mono_db.rows[i].scattering_real, merged_db.rows[i].scattering_real);
        EXPECT_DOUBLE_EQ(mono_db.rows[i].scattering_imag, merged_db.rows[i].scattering_imag);
        EXPECT_DOUBLE_EQ(mono_db.rows[i].rcs_sqm, merged_db.rows[i].rcs_sqm);
    }
}

} // namespace

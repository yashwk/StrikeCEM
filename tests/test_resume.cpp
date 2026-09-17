// Checkpoint/resume tests: crash recovery with data loss, checksum tamper
// detection, resume rejection on changed plans, and overwrite protection.
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include <H5Cpp.h>
#include <nlohmann/json.hpp>

#include "strikecem/app/CLI.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/io/Hdf5Reader.hpp"

namespace {

namespace fs = std::filesystem;

struct Workdir {
    fs::path dir = fs::temp_directory_path() / "scem_resume_test";
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

nlohmann::json base_config(const fs::path& out_path) {
    nlohmann::json cfg;
    cfg["scem_schema_version"] = "1.0";
    cfg["model"] = {{"path", std::string(std::string(SCEM_FIXTURE_DIR) +
                                         "/../../examples/plate.stl")}};
    cfg["frequency"] = {{"frequency_hz", 10e9}};
    cfg["angles"]["azimuth"] = {{"start", 0}, {"stop", 270}, {"step", 90}};
    cfg["angles"]["elevation"] = {{"start", -90}, {"stop", -90}, {"step", 1}};
    cfg["polarization"] = {"HH", "VV"};
    cfg["solver"] = {{"type", "PO"}};
    cfg["run"] = {{"checkpoint_every_n_samples", 4}}; // 8 rows -> 2 chunks
    cfg["output"] = {{"path", out_path.string()}, {"format", "hdf5"}};
    return cfg;
}

fs::path write_config(const fs::path& dir, const nlohmann::json& cfg, const std::string& name) {
    const fs::path path = dir / name;
    std::ofstream out(path);
    out << cfg.dump(2);
    return path;
}

int cli_run(const fs::path& config_path) {
    const std::string argv0 = "strikecem";
    const std::string config = config_path.string();
    char* argv[] = {const_cast<char*>(argv0.c_str()), const_cast<char*>("run"),
                    const_cast<char*>(config.c_str())};
    return strikecem::cli::run(3, argv);
}

std::vector<double> read_scattering_real(const fs::path& h5) {
    auto db = strikecem::read_hdf5_database(h5.string());
    std::vector<double> values;
    for (const auto& row : db.rows) values.push_back(row.scattering_real);
    return values;
}

void zero_flag_and_rows(const fs::path& h5) {
    H5::H5File file(h5.string(), H5F_ACC_RDWR);
    H5::Group progress = file.openGroup("/progress");
    H5::DataSet flags = progress.openDataSet("completed_chunks");
    const uint8_t zero[1] = {0};
    const hsize_t count[1] = {1}, start[1] = {1};
    const H5::DataSpace mem(1, count);
    H5::DataSpace flag_space = flags.getSpace();
    flag_space.selectHyperslab(H5S_SELECT_SET, count, start);
    flags.write(zero, H5::PredType::NATIVE_UINT8, mem, flag_space);
    // Simulate crash data loss: zero the second chunk's field data too.
    H5::Group samples = file.openGroup("/samples");
    for (const char* name :
         {"scattering_real", "scattering_imag", "rcs_sqm", "rcs_dBsm"}) {
        H5::DataSet ds = samples.openDataSet(name);
        const hsize_t n[1] = {4}, off[1] = {4};
        const H5::DataSpace chunk_mem(1, n);
        H5::DataSpace chunk_space = ds.getSpace();
        chunk_space.selectHyperslab(H5S_SELECT_SET, n, off);
        std::vector<double> zeros(4, 0.0);
        ds.write(zeros.data(), H5::PredType::NATIVE_DOUBLE, chunk_mem, chunk_space);
    }
    file.flush(H5F_SCOPE_GLOBAL);
    file.close();
}

TEST(Resume, CompletesAfterDataLoss) {
    Workdir work;
    const fs::path out = work.dir / "out.h5";
    const fs::path config = write_config(work.dir, base_config(out), "scem.json");
    ASSERT_EQ(cli_run(config), 0);
    const auto reference = read_scattering_real(out);
    ASSERT_EQ(reference.size(), 8u);
    zero_flag_and_rows(out);
    auto crashed = strikecem::read_hdf5_database(out.string());
    EXPECT_THROW(strikecem::validate_hdf5_complete(crashed), strikecem::ReaderError);

    nlohmann::json resume_cfg;
    {
        std::ifstream in(config);
        resume_cfg = nlohmann::json::parse(in);
    }
    resume_cfg["run"]["resume_from_checkpoint"] = out.string();
    const fs::path resume_config = write_config(work.dir, resume_cfg, "resume.json");
    EXPECT_EQ(cli_run(resume_config), 0);
    auto db = strikecem::read_hdf5_database(out.string());
    EXPECT_NO_THROW(strikecem::validate_hdf5_complete(db));
    const auto restored = read_scattering_real(out);
    EXPECT_EQ(restored, reference); // deterministic recompute, bitwise
}

TEST(Resume, NoWorkWhenComplete) {
    Workdir work;
    const fs::path out = work.dir / "out.h5";
    const fs::path config = write_config(work.dir, base_config(out), "scem.json");
    ASSERT_EQ(cli_run(config), 0);
    nlohmann::json resume_cfg;
    {
        std::ifstream in(config);
        resume_cfg = nlohmann::json::parse(in);
    }
    resume_cfg["run"]["resume_from_checkpoint"] = out.string();
    EXPECT_EQ(cli_run(write_config(work.dir, resume_cfg, "resume.json")), 0);
}

TEST(Resume, RejectsChangedPlan) {
    Workdir work;
    const fs::path out = work.dir / "out.h5";
    const fs::path config = write_config(work.dir, base_config(out), "scem.json");
    ASSERT_EQ(cli_run(config), 0);
    zero_flag_and_rows(out);
    auto changed = base_config(out);
    changed["frequency"] = {{"frequency_hz", 11e9}};
    changed["run"]["resume_from_checkpoint"] = out.string();
    EXPECT_EQ(cli_run(write_config(work.dir, changed, "changed.json")), 6);
}

TEST(Resume, PartialWithoutFlagRefuses) {
    Workdir work;
    const fs::path out = work.dir / "out.h5";
    const fs::path config = write_config(work.dir, base_config(out), "scem.json");
    ASSERT_EQ(cli_run(config), 0);
    zero_flag_and_rows(out);
    EXPECT_EQ(cli_run(config), 6);
}

TEST(Resume, CompleteRerunIsIdempotent) {
    Workdir work;
    const fs::path out = work.dir / "out.h5";
    const fs::path config = write_config(work.dir, base_config(out), "scem.json");
    ASSERT_EQ(cli_run(config), 0);
    EXPECT_EQ(cli_run(config), 0);
}

TEST(Resume, CsvResumeIsRejected) {
    Workdir work;
    auto cfg = base_config(work.dir / "out.csv");
    cfg["output"]["format"] = "csv";
    cfg["run"]["resume_from_checkpoint"] = (work.dir / "out.csv").string();
    EXPECT_EQ(cli_run(write_config(work.dir, cfg, "scem.json")), 2);
}

TEST(Resume, ChecksumTamperIsRepaired) {
    Workdir work;
    const fs::path out = work.dir / "out.h5";
    const fs::path config = write_config(work.dir, base_config(out), "scem.json");
    ASSERT_EQ(cli_run(config), 0);
    const auto reference = read_scattering_real(out);
    {
        // Flip bits in a committed row without touching flags/checksums.
        H5::H5File file(out.string(), H5F_ACC_RDWR);
        H5::Group samples = file.openGroup("/samples");
        H5::DataSet ds = samples.openDataSet("scattering_real");
        std::vector<double> values(8);
        ds.read(values.data(), H5::PredType::NATIVE_DOUBLE);
        uint64_t bits = 0;
        std::memcpy(&bits, &values[0], sizeof(bits));
        bits ^= 1ULL << 32; // disturb the value, keep it finite
        std::memcpy(&values[0], &bits, sizeof(values[0]));
        ds.write(values.data(), H5::PredType::NATIVE_DOUBLE);
        file.flush(H5F_SCOPE_GLOBAL);
        file.close();
    }
    auto tampered = strikecem::read_hdf5_database(out.string());
    EXPECT_THROW(strikecem::validate_hdf5_complete(tampered), strikecem::ReaderError);
    nlohmann::json resume_cfg;
    {
        std::ifstream in(config);
        resume_cfg = nlohmann::json::parse(in);
    }
    resume_cfg["run"]["resume_from_checkpoint"] = out.string();
    EXPECT_EQ(cli_run(write_config(work.dir, resume_cfg, "resume.json")), 0);
    EXPECT_EQ(read_scattering_real(out), reference);
}

TEST(Resume, SamplePlanHashIsStable) {
    const auto load = [](const std::string& name) {
        const std::string path = std::string(SCEM_FIXTURE_DIR) + "/" + name;
        auto rc = strikecem::load_config(path, SCEM_SCHEMA_PATH);
        return strikecem::build_sample_plan(rc.value);
    };
    const auto a = load("valid_minimal.json");
    const auto b = load("valid_minimal.json");
    EXPECT_EQ(a.hash.size(), 64u);
    EXPECT_EQ(a.hash, b.hash);
    const auto c = load("valid_sweep_grid.json");
    EXPECT_NE(a.hash, c.hash);
}

} // namespace

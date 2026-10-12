// Tests for Track 1: Distributed batch execution and shard merger.
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <unistd.h>

#include <H5Cpp.h>
#include <nlohmann/json.hpp>

#include "strikecem/app/CLI.hpp"
#include "strikecem/core/Config.hpp"
#include "strikecem/distributed/Shard.hpp"
#include "strikecem/io/Hdf5Reader.hpp"
#include "strikecem/io/OutputError.hpp"

namespace {

namespace fs = std::filesystem;

struct Workdir {
    fs::path dir = fs::temp_directory_path() / ("scem_dist_test_" + std::to_string(::getpid()) + "_" +
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

nlohmann::json make_test_config(const fs::path& out_path, const std::string& fmt = "hdf5",
                                size_t n_az = 4, size_t n_freq = 1) {
    nlohmann::json cfg;
    cfg["scem_schema_version"] = "1.0";
    cfg["model"] = {{"path", std::string(std::string(SCEM_FIXTURE_DIR) +
                                         "/../../examples/plate.stl")}};
    if (n_freq == 1) {
        cfg["frequency"] = {{"frequency_hz", 10e9}};
    } else {
        cfg["frequency"] = {{"sweep", {{"start", 8e9}, {"stop", 10e9}, {"step", 1e9}}}};
    }
    cfg["angles"]["azimuth"] = {{"start", 0}, {"stop", static_cast<double>((n_az - 1) * 30)}, {"step", 30}};
    cfg["angles"]["elevation"] = {{"start", 0}, {"stop", 0}, {"step", 1}};
    cfg["polarization"] = {"HH", "VV"};
    cfg["solver"] = {{"type", "PO"}};
    cfg["run"] = {{"checkpoint_every_n_samples", 2}};
    cfg["output"] = {{"path", out_path.string()}, {"format", fmt}};
    return cfg;
}

fs::path write_json(const fs::path& path, const nlohmann::json& j) {
    std::ofstream out(path);
    out << j.dump(2);
    return path;
}

} // namespace

TEST(Distributed, PartitionMathAndManifestGeneration) {
    Workdir w;
    const fs::path cfg_file = write_json(w.dir / "config.json", make_test_config(w.dir / "full.h5", "hdf5", 5, 1));
    const auto rc = strikecem::load_config(cfg_file.string(), SCEM_SCHEMA_PATH);
    const auto plan = strikecem::build_sample_plan(rc.value);
    const auto mesh = strikecem::load_normalized_mesh(rc.value, w.dir, rc.schema_version);

    // 5 units across 3 shards -> shards should have 2, 2, 1 units
    const auto manifest = strikecem::create_shards(rc, plan, mesh, cfg_file.string(), 3, (w.dir / "shards").string(), "test-run");
    EXPECT_EQ(manifest.num_shards, 3u);
    EXPECT_EQ(manifest.total_units, 5u);
    EXPECT_EQ(manifest.total_samples, 10u);
    EXPECT_EQ(manifest.run_id, "test-run");
    ASSERT_EQ(manifest.shards.size(), 3u);

    EXPECT_EQ(manifest.shards[0].unit_start, 0u);
    EXPECT_EQ(manifest.shards[0].unit_count, 2u);
    EXPECT_EQ(manifest.shards[0].sample_id_start, 0u);
    EXPECT_EQ(manifest.shards[0].sample_id_count, 4u);

    EXPECT_EQ(manifest.shards[1].unit_start, 2u);
    EXPECT_EQ(manifest.shards[1].unit_count, 2u);
    EXPECT_EQ(manifest.shards[1].sample_id_start, 4u);
    EXPECT_EQ(manifest.shards[1].sample_id_count, 4u);

    EXPECT_EQ(manifest.shards[2].unit_start, 4u);
    EXPECT_EQ(manifest.shards[2].unit_count, 1u);
    EXPECT_EQ(manifest.shards[2].sample_id_start, 8u);
    EXPECT_EQ(manifest.shards[2].sample_id_count, 2u);

    // Verify manifest JSON and shard config files exist
    EXPECT_TRUE(fs::exists(w.dir / "shards" / "shard_manifest.json"));
    EXPECT_TRUE(fs::exists(manifest.shards[0].config_path));
    EXPECT_TRUE(fs::exists(manifest.shards[1].config_path));
    EXPECT_TRUE(fs::exists(manifest.shards[2].config_path));
}

TEST(Distributed, EndToEndHdf5DirectMatch) {
    Workdir w;
    const fs::path full_out = w.dir / "direct.h5";
    const fs::path cfg_file = write_json(w.dir / "config.json", make_test_config(full_out, "hdf5", 4, 1));

    // 1. Run direct unpartitioned
    ASSERT_EQ(call_cli({"run", cfg_file.string()}), 0);
    ASSERT_TRUE(fs::exists(full_out));

    // 2. Generate 2 shards
    const fs::path shards_dir = w.dir / "shards";
    ASSERT_EQ(call_cli({"shard", "--num-shards", "2", "--out-dir", shards_dir.string(), cfg_file.string()}), 0);

    // 3. Run each shard
    ASSERT_EQ(call_cli({"run", (shards_dir / "shard_0000.json").string()}), 0);
    ASSERT_EQ(call_cli({"run", (shards_dir / "shard_0001.json").string()}), 0);

    // 4. Merge shards
    const fs::path merged_out = w.dir / "merged.h5";
    ASSERT_EQ(call_cli({"merge", "--output", merged_out.string(), (shards_dir / "shard_manifest.json").string()}), 0);
    ASSERT_TRUE(fs::exists(merged_out));

    // 5. Assert direct and merged outputs are bit-identical and complete
    const auto direct_db = strikecem::read_hdf5_database(full_out.string());
    const auto merged_db = strikecem::read_hdf5_database(merged_out.string());
    strikecem::validate_hdf5_complete(direct_db);
    strikecem::validate_hdf5_complete(merged_db);

    ASSERT_EQ(direct_db.rows.size(), merged_db.rows.size());
    EXPECT_EQ(direct_db.config_hash, merged_db.config_hash);

    for (size_t i = 0; i < direct_db.rows.size(); ++i) {
        EXPECT_EQ(direct_db.rows[i].sample_id, merged_db.rows[i].sample_id);
        EXPECT_EQ(direct_db.rows[i].direction_index, merged_db.rows[i].direction_index);
        EXPECT_EQ(direct_db.rows[i].frequency_index, merged_db.rows[i].frequency_index);
        EXPECT_EQ(direct_db.rows[i].polarization_index, merged_db.rows[i].polarization_index);
        EXPECT_DOUBLE_EQ(direct_db.rows[i].rcs_sqm, merged_db.rows[i].rcs_sqm);
        EXPECT_DOUBLE_EQ(direct_db.rows[i].scattering_real, merged_db.rows[i].scattering_real);
        EXPECT_DOUBLE_EQ(direct_db.rows[i].scattering_imag, merged_db.rows[i].scattering_imag);
    }
}

TEST(Distributed, EndToEndCsvDirectMatch) {
    Workdir w;
    const fs::path full_out = w.dir / "direct.csv";
    const fs::path cfg_file = write_json(w.dir / "config.json", make_test_config(full_out, "csv", 4, 1));

    // 1. Run direct unpartitioned
    ASSERT_EQ(call_cli({"run", cfg_file.string()}), 0);
    ASSERT_TRUE(fs::exists(full_out));

    // 2. Generate 2 shards
    const fs::path shards_dir = w.dir / "shards_csv";
    ASSERT_EQ(call_cli({"shard", "--num-shards", "2", "--out-dir", shards_dir.string(), cfg_file.string()}), 0);

    // 3. Run each shard
    ASSERT_EQ(call_cli({"run", (shards_dir / "shard_0000.json").string()}), 0);
    ASSERT_EQ(call_cli({"run", (shards_dir / "shard_0001.json").string()}), 0);

    // 4. Merge shards
    const fs::path merged_out = w.dir / "merged.csv";
    ASSERT_EQ(call_cli({"merge", "--output", merged_out.string(), (shards_dir / "shard_manifest.json").string()}), 0);
    ASSERT_TRUE(fs::exists(merged_out));

    // 5. Compare lines of direct and merged CSV files
    std::ifstream f1(full_out), f2(merged_out);
    std::string l1, l2;
    while (std::getline(f1, l1) && std::getline(f2, l2)) {
        EXPECT_EQ(l1, l2);
    }
    EXPECT_FALSE(std::getline(f1, l1));
    EXPECT_FALSE(std::getline(f2, l2));
}

TEST(Distributed, UnevenPartitionThreeShards) {
    Workdir w;
    const fs::path full_out = w.dir / "direct.h5";
    // 5 azimuth angles, 1 frequency -> 5 units (10 samples)
    const fs::path cfg_file = write_json(w.dir / "config.json", make_test_config(full_out, "hdf5", 5, 1));

    ASSERT_EQ(call_cli({"run", cfg_file.string()}), 0);

    const fs::path shards_dir = w.dir / "shards_uneven";
    ASSERT_EQ(call_cli({"shard", "--num-shards", "3", "--out-dir", shards_dir.string(), cfg_file.string()}), 0);

    ASSERT_EQ(call_cli({"run", (shards_dir / "shard_0000.json").string()}), 0);
    ASSERT_EQ(call_cli({"run", (shards_dir / "shard_0001.json").string()}), 0);
    ASSERT_EQ(call_cli({"run", (shards_dir / "shard_0002.json").string()}), 0);

    const fs::path merged_out = w.dir / "merged.h5";
    ASSERT_EQ(call_cli({"merge", "--output", merged_out.string(), (shards_dir / "shard_manifest.json").string()}), 0);

    const auto direct_db = strikecem::read_hdf5_database(full_out.string());
    const auto merged_db = strikecem::read_hdf5_database(merged_out.string());
    ASSERT_EQ(direct_db.rows.size(), 10u);
    ASSERT_EQ(merged_db.rows.size(), 10u);
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(direct_db.rows[i].sample_id, merged_db.rows[i].sample_id);
        EXPECT_DOUBLE_EQ(direct_db.rows[i].rcs_sqm, merged_db.rows[i].rcs_sqm);
    }
}

TEST(Distributed, FailClosedOnMissingShard) {
    Workdir w;
    const fs::path full_out = w.dir / "direct.h5";
    const fs::path cfg_file = write_json(w.dir / "config.json", make_test_config(full_out, "hdf5", 4, 1));

    const fs::path shards_dir = w.dir / "shards_fail";
    ASSERT_EQ(call_cli({"shard", "--num-shards", "2", "--out-dir", shards_dir.string(), cfg_file.string()}), 0);
    ASSERT_EQ(call_cli({"run", (shards_dir / "shard_0000.json").string()}), 0);
    // Deliberately do not run shard_0001!

    // Merge must fail because shard_0001.h5 does not exist
    EXPECT_THROW(
        strikecem::merge_shards((shards_dir / "shard_manifest.json").string(), {}, (w.dir / "merged.h5").string()),
        strikecem::OutputError
    );
}

TEST(Distributed, FailClosedOnDuplicateOrGap) {
    Workdir w;
    const fs::path full_out = w.dir / "direct.h5";
    const fs::path cfg_file = write_json(w.dir / "config.json", make_test_config(full_out, "hdf5", 4, 1));

    const fs::path shards_dir = w.dir / "shards_dup";
    ASSERT_EQ(call_cli({"shard", "--num-shards", "2", "--out-dir", shards_dir.string(), cfg_file.string()}), 0);
    ASSERT_EQ(call_cli({"run", (shards_dir / "shard_0000.json").string()}), 0);

    const std::string shard0_path = (shards_dir / "shard_0000.h5").string();
    // Providing shard 0 twice should fail closed on duplicate sample_id
    EXPECT_THROW(
        strikecem::merge_shards(shard0_path, {shard0_path}, (w.dir / "merged.h5").string()),
        strikecem::OutputError
    );
}

TEST(Distributed, DirectFileListMerge) {
    Workdir w;
    const fs::path full_out = w.dir / "direct.h5";
    const fs::path cfg_file = write_json(w.dir / "config.json", make_test_config(full_out, "hdf5", 4, 1));

    const fs::path shards_dir = w.dir / "shards_list";
    ASSERT_EQ(call_cli({"shard", "--num-shards", "2", "--out-dir", shards_dir.string(), cfg_file.string()}), 0);
    ASSERT_EQ(call_cli({"run", (shards_dir / "shard_0000.json").string()}), 0);
    ASSERT_EQ(call_cli({"run", (shards_dir / "shard_0001.json").string()}), 0);

    const fs::path merged_out = w.dir / "merged.h5";
    // Merge via explicit file list on CLI instead of manifest
    ASSERT_EQ(call_cli({"merge", "--output", merged_out.string(),
                        (shards_dir / "shard_0000.h5").string(),
                        (shards_dir / "shard_0001.h5").string()}), 0);
    ASSERT_TRUE(fs::exists(merged_out));
    const auto merged_db = strikecem::read_hdf5_database(merged_out.string());
    EXPECT_EQ(merged_db.rows.size(), 8u);
}

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "strikecem/app/CLI.hpp"
#include "strikecem/io/AssetBundle.hpp"
#include "strikecem/io/OutputError.hpp"

namespace {

namespace fs = std::filesystem;

fs::path temp_dir(const std::string& sub) {
    fs::path p = fs::temp_directory_path() / ("scem_bundle_test_" + sub);
    fs::create_directories(p);
    return p;
}

TEST(AssetBundle, CreateAndReadRoundTrip) {
    const fs::path dir = temp_dir("roundtrip");
    const fs::path bundle_path = dir / "test.scempack";

    strikecem::AssetBundleWriter writer;
    nlohmann::json manifest = {
        {"design_id", "drone-x"},
        {"design_revision", "101"},
        {"units", "m"}
    };
    writer.set_manifest(manifest);

    writer.add_entry_text("database.scem", "application/x-scem", "SCEM_BINARY_DUMMY_DATA_12345");
    writer.add_entry_text("model.obj", "model/obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
    writer.add_entry_text("readme.txt", "text/plain", "Asset bundle for StrikeEngine runtime");

    writer.write_to_file(bundle_path.string());
    ASSERT_TRUE(fs::exists(bundle_path));
    EXPECT_GT(fs::file_size(bundle_path), 256u);

    // Read back
    const auto reader = strikecem::AssetBundleReader::open_file(bundle_path.string(), true);
    EXPECT_EQ(reader.manifest()["design_id"], "drone-x");
    EXPECT_EQ(reader.manifest()["design_revision"], "101");

    EXPECT_EQ(reader.entries().size(), 3u);
    EXPECT_TRUE(reader.has_entry("database.scem"));
    EXPECT_TRUE(reader.has_entry("model.obj"));
    EXPECT_TRUE(reader.has_entry("readme.txt"));
    EXPECT_FALSE(reader.has_entry("missing.dat"));

    EXPECT_EQ(reader.read_entry_text("database.scem"), "SCEM_BINARY_DUMMY_DATA_12345");
    EXPECT_EQ(reader.read_entry_text("readme.txt"), "Asset bundle for StrikeEngine runtime");

    fs::remove_all(dir);
}

TEST(AssetBundle, TamperedPayloadThrows) {
    const fs::path dir = temp_dir("tamper");
    const fs::path bundle_path = dir / "tamper.scempack";

    strikecem::AssetBundleWriter writer;
    writer.add_entry_text("data.bin", "application/octet-stream", "AUTHENTIC_CONTENT_SAFE");
    writer.write_to_file(bundle_path.string());

    // Corrupt one byte near the end of the file
    {
        std::fstream f(bundle_path.string(), std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(-5, std::ios::end);
        f.put('X');
    }

    // Opening with verify_sha256=true must throw
    EXPECT_THROW(strikecem::AssetBundleReader::open_file(bundle_path.string(), true),
                 strikecem::OutputError);

    fs::remove_all(dir);
}

TEST(AssetBundle, ExtractAllEntries) {
    const fs::path dir = temp_dir("extract");
    const fs::path bundle_path = dir / "archive.scempack";
    const fs::path extract_dir = dir / "extracted";

    strikecem::AssetBundleWriter writer;
    writer.add_entry_text("lod/mesh.obj", "model/obj", "v 1 2 3");
    writer.add_entry_text("data/table.bin", "application/octet-stream", "ROWDATA");
    writer.write_to_file(bundle_path.string());

    const auto reader = strikecem::AssetBundleReader::open_file(bundle_path.string());
    reader.extract_all(extract_dir.string());

    EXPECT_TRUE(fs::exists(extract_dir / "lod/mesh.obj"));
    EXPECT_TRUE(fs::exists(extract_dir / "data/table.bin"));

    std::ifstream in(extract_dir / "lod/mesh.obj");
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_EQ(content, "v 1 2 3");

    fs::remove_all(dir);
}

TEST(AssetBundle, CreateStrikeEnginePackageWithLods) {
    const fs::path dir = temp_dir("se_package");
    const fs::path rcs_file = dir / "test_out.scem";
    const fs::path mesh_file = dir / "main_body.obj";
    const fs::path lod1_file = dir / "body_lod1.obj";
    const fs::path bundle_file = dir / "package.scempack";

    {
        std::ofstream f(rcs_file);
        f << "SCEM_TABLE_DATA";
    }
    {
        std::ofstream f(mesh_file);
        f << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3";
    }
    {
        std::ofstream f(lod1_file);
        f << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3 (simplified)";
    }

    nlohmann::json meta = {
        {"target_class", "fighter"},
        {"radar_frequency_ghz", 10.0}
    };

    strikecem::create_strikeengine_package(
        bundle_file.string(),
        rcs_file.string(),
        mesh_file.string(),
        {{"lod1", lod1_file.string()}},
        meta);

    ASSERT_TRUE(fs::exists(bundle_file));
    const auto reader = strikecem::AssetBundleReader::open_file(bundle_file.string(), true);
    EXPECT_EQ(reader.manifest()["package_type"], "StrikeEngine_Asset_Bundle");
    EXPECT_EQ(reader.manifest()["target_class"], "fighter");
    EXPECT_TRUE(reader.has_entry("test_out.scem"));
    EXPECT_TRUE(reader.has_entry("main_body.obj"));
    EXPECT_TRUE(reader.has_entry("lod1_body_lod1.obj"));

    fs::remove_all(dir);
}

TEST(AssetBundle, CliExecution) {
    const fs::path dir = temp_dir("cli");
    const fs::path rcs_file = dir / "db.scem";
    const fs::path mesh_file = dir / "geom.obj";
    const fs::path bundle_file = dir / "asset.scempack";

    {
        std::ofstream f(rcs_file);
        f << "SCEM_PAYLOAD";
    }
    {
        std::ofstream f(mesh_file);
        f << "v 0 0 0";
    }

    std::vector<std::string> args = {
        "strikecem", "bundle",
        "--rcs", rcs_file.string(),
        "--mesh", mesh_file.string(),
        "--output", bundle_file.string()
    };
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());

    int rc = strikecem::cli::run(static_cast<int>(argv.size()), argv.data());
    EXPECT_EQ(rc, 0);
    ASSERT_TRUE(fs::exists(bundle_file));

    const auto reader = strikecem::AssetBundleReader::open_file(bundle_file.string());
    EXPECT_TRUE(reader.has_entry("db.scem"));
    EXPECT_TRUE(reader.has_entry("geom.obj"));

    fs::remove_all(dir);
}

} // namespace

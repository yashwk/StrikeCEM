#include "strikecem/io/AssetBundle.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/OutputError.hpp"

namespace strikecem {

namespace fs = std::filesystem;

void AssetBundleWriter::set_manifest(const nlohmann::json& manifest) {
    manifest_ = manifest;
}

void AssetBundleWriter::add_entry(const std::string& name, const std::string& mime_type,
                                 const std::vector<uint8_t>& data) {
    entries_.push_back({name, mime_type, data});
}

void AssetBundleWriter::add_entry_text(const std::string& name, const std::string& mime_type,
                                      const std::string& text) {
    std::vector<uint8_t> data(text.begin(), text.end());
    add_entry(name, mime_type, data);
}

void AssetBundleWriter::add_entry_file(const std::string& name, const std::string& mime_type,
                                       const std::string& file_path) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in) throw OutputError("cannot open file for bundling: " + file_path);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
    add_entry(name, mime_type, data);
}

void AssetBundleWriter::write_to_file(const std::string& output_path) {
    ScemPackHeader hdr{};
    hdr.entry_count = static_cast<uint32_t>(entries_.size());

    // Serialize manifest
    std::string manifest_str = manifest_.dump(2);
    hdr.manifest_offset = sizeof(ScemPackHeader);
    hdr.manifest_bytes = manifest_str.size();

    // Prepare table of entries
    nlohmann::json table_json = nlohmann::json::array();
    uint64_t current_payload_offset = hdr.manifest_offset + hdr.manifest_bytes;

    // First pass to determine entry offsets
    // We estimate table size or compute iteratively
    std::vector<BundleEntryRecord> records;
    records.reserve(entries_.size());

    std::string all_payloads;
    for (const auto& e : entries_) {
        BundleEntryRecord rec;
        rec.name = e.name;
        rec.mime_type = e.mime_type;
        rec.size = e.data.size();
        std::string bytes_str(reinterpret_cast<const char*>(e.data.data()), e.data.size());
        rec.sha256_hex = sha256_hex(bytes_str);
        all_payloads.append(bytes_str);
        records.push_back(rec);
    }

    std::string total_payload_hash = sha256_hex(all_payloads);
    std::memcpy(hdr.payload_sha256, total_payload_hash.data(), 64);

    // Build JSON table with placeholder offsets to measure size
    for (const auto& rec : records) {
        nlohmann::json j;
        j["name"] = rec.name;
        j["mime_type"] = rec.mime_type;
        j["offset"] = 0;
        j["size"] = rec.size;
        j["sha256"] = rec.sha256_hex;
        table_json.push_back(j);
    }

    std::string table_str = table_json.dump();
    hdr.table_offset = hdr.manifest_offset + hdr.manifest_bytes;
    hdr.table_bytes = table_str.size();

    // Now compute true payload offsets
    current_payload_offset = hdr.table_offset + hdr.table_bytes;
    for (size_t i = 0; i < records.size(); ++i) {
        records[i].offset = current_payload_offset;
        table_json[i]["offset"] = current_payload_offset;
        current_payload_offset += records[i].size;
    }

    // Re-serialize table with final offsets (string length is stable or close; pad if needed)
    table_str = table_json.dump();
    while (table_str.size() < hdr.table_bytes) {
        table_str.push_back(' '); // pad spaces within JSON padding
    }
    if (table_str.size() > hdr.table_bytes) {
        // Expand and recompute offsets
        hdr.table_bytes = table_str.size();
        current_payload_offset = hdr.table_offset + hdr.table_bytes;
        for (size_t i = 0; i < records.size(); ++i) {
            records[i].offset = current_payload_offset;
            table_json[i]["offset"] = current_payload_offset;
            current_payload_offset += records[i].size;
        }
        table_str = table_json.dump();
        while (table_str.size() < hdr.table_bytes) table_str.push_back(' ');
    }

    hdr.total_bundle_bytes = current_payload_offset;

    // Write file
    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out) throw OutputError("cannot create asset bundle file: " + output_path);

    // 1. Header
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(ScemPackHeader));
    // 2. Manifest
    out.write(manifest_str.data(), manifest_str.size());
    // 3. Table
    out.write(table_str.data(), table_str.size());
    // 4. Payloads
    for (const auto& e : entries_) {
        if (!e.data.empty()) {
            out.write(reinterpret_cast<const char*>(e.data.data()), e.data.size());
        }
    }
    out.flush();
    if (!out) throw OutputError("failed writing asset bundle data to: " + output_path);
}

AssetBundleReader AssetBundleReader::open_file(const std::string& bundle_path, bool verify_sha256) {
    std::ifstream in(bundle_path, std::ios::binary);
    if (!in) throw OutputError("cannot open asset bundle file: " + bundle_path);

    ScemPackHeader hdr{};
    in.read(reinterpret_cast<char*>(&hdr), sizeof(ScemPackHeader));
    if (in.gcount() != sizeof(ScemPackHeader)) {
        throw OutputError("invalid asset bundle: file smaller than header");
    }

    if (std::memcmp(hdr.magic, "SCEMPACK", 8) != 0) {
        throw OutputError("invalid asset bundle magic: not a .scempack file");
    }
    if (hdr.version != 1) {
        throw OutputError("unsupported .scempack version: " + std::to_string(hdr.version));
    }
    if (hdr.header_size != sizeof(ScemPackHeader)) {
        throw OutputError("invalid .scempack header size");
    }

    // Read manifest
    in.seekg(hdr.manifest_offset);
    std::string manifest_str(hdr.manifest_bytes, '\0');
    in.read(manifest_str.data(), hdr.manifest_bytes);
    nlohmann::json manifest;
    try {
        manifest = nlohmann::json::parse(manifest_str);
    } catch (const std::exception& e) {
        throw OutputError("corrupted bundle manifest JSON: " + std::string(e.what()));
    }

    // Read table
    in.seekg(hdr.table_offset);
    std::string table_str(hdr.table_bytes, '\0');
    in.read(table_str.data(), hdr.table_bytes);
    nlohmann::json table_json;
    try {
        table_json = nlohmann::json::parse(table_str);
    } catch (const std::exception& e) {
        throw OutputError("corrupted bundle table JSON: " + std::string(e.what()));
    }

    std::vector<BundleEntryRecord> entries;
    for (const auto& item : table_json) {
        BundleEntryRecord rec;
        rec.name = item["name"].get<std::string>();
        rec.mime_type = item["mime_type"].get<std::string>();
        rec.offset = item["offset"].get<uint64_t>();
        rec.size = item["size"].get<uint64_t>();
        rec.sha256_hex = item["sha256"].get<std::string>();
        entries.push_back(rec);
    }

    AssetBundleReader reader;
    reader.bundle_path_ = bundle_path;
    reader.header_ = hdr;
    reader.manifest_ = manifest;
    reader.entries_ = entries;

    if (verify_sha256) {
        reader.verify_integrity();
    }

    return reader;
}

bool AssetBundleReader::has_entry(const std::string& name) const {
    return get_entry_info(name) != nullptr;
}

const BundleEntryRecord* AssetBundleReader::get_entry_info(const std::string& name) const {
    for (const auto& e : entries_) {
        if (e.name == name) return &e;
    }
    return nullptr;
}

std::vector<uint8_t> AssetBundleReader::read_entry_bytes(const std::string& name) const {
    const auto* info = get_entry_info(name);
    if (!info) throw OutputError("entry not found in bundle: " + name);

    std::ifstream in(bundle_path_, std::ios::binary);
    if (!in) throw OutputError("cannot open bundle file: " + bundle_path_);

    in.seekg(info->offset);
    std::vector<uint8_t> buf(info->size);
    if (info->size > 0) {
        in.read(reinterpret_cast<char*>(buf.data()), info->size);
        if (static_cast<uint64_t>(in.gcount()) != info->size) {
            throw OutputError("truncated bundle entry payload for: " + name);
        }
    }
    return buf;
}

std::string AssetBundleReader::read_entry_text(const std::string& name) const {
    auto bytes = read_entry_bytes(name);
    return std::string(bytes.begin(), bytes.end());
}

void AssetBundleReader::extract_all(const std::string& dest_dir) const {
    fs::create_directories(dest_dir);
    for (const auto& e : entries_) {
        auto data = read_entry_bytes(e.name);
        fs::path target = fs::path(dest_dir) / e.name;
        fs::create_directories(target.parent_path());
        std::ofstream out(target, std::ios::binary | std::ios::trunc);
        if (!out) throw OutputError("cannot extract bundle entry to: " + target.string());
        if (!data.empty()) {
            out.write(reinterpret_cast<const char*>(data.data()), data.size());
        }
    }
}

bool AssetBundleReader::verify_integrity() const {
    std::ifstream in(bundle_path_, std::ios::binary);
    if (!in) throw OutputError("cannot open bundle file: " + bundle_path_);

    std::string all_payloads;
    for (const auto& e : entries_) {
        in.seekg(e.offset);
        std::string payload(e.size, '\0');
        if (e.size > 0) {
            in.read(payload.data(), e.size);
            if (static_cast<uint64_t>(in.gcount()) != e.size) {
                throw OutputError("bundle payload truncated for: " + e.name);
            }
        }
        if (sha256_hex(payload) != e.sha256_hex) {
            throw OutputError("SHA-256 verification failed for entry: " + e.name);
        }
        all_payloads.append(payload);
    }

    std::string total_hash = sha256_hex(all_payloads);
    if (std::memcmp(header_.payload_sha256, total_hash.data(), 64) != 0) {
        throw OutputError("total payload SHA-256 verification failed for bundle");
    }
    return true;
}

void create_strikeengine_package(
    const std::string& output_bundle_path,
    const std::string& rcs_database_path,
    const std::string& primary_mesh_path,
    const std::vector<std::pair<std::string, std::string>>& lod_meshes,
    const nlohmann::json& metadata) {
    AssetBundleWriter writer;

    nlohmann::json manifest = metadata;
    manifest["package_type"] = "StrikeEngine_Asset_Bundle";
    manifest["package_format_version"] = "1.0";
    manifest["created_by"] = "StrikeCEM";

    // Add RCS database
    fs::path rcs_p(rcs_database_path);
    std::string rcs_name = rcs_p.filename().string();
    std::string rcs_mime = (rcs_p.extension() == ".scem") ? "application/x-scem" : "application/x-hdf5";
    writer.add_entry_file(rcs_name, rcs_mime, rcs_database_path);
    manifest["rcs_database"] = rcs_name;

    // Add primary mesh
    fs::path mesh_p(primary_mesh_path);
    std::string mesh_name = mesh_p.filename().string();
    std::string mesh_mime = (mesh_p.extension() == ".obj") ? "model/obj" : "model/stl";
    writer.add_entry_file(mesh_name, mesh_mime, primary_mesh_path);
    manifest["primary_geometry"] = mesh_name;

    // Add LOD meshes
    nlohmann::json lod_json = nlohmann::json::object();
    for (const auto& [lod_name, lod_path] : lod_meshes) {
        fs::path p(lod_path);
        std::string name_in_bundle = lod_name + "_" + p.filename().string();
        std::string mime = (p.extension() == ".obj") ? "model/obj" : "model/stl";
        writer.add_entry_file(name_in_bundle, mime, lod_path);
        lod_json[lod_name] = name_in_bundle;
    }
    if (!lod_meshes.empty()) {
        manifest["lod_geometries"] = lod_json;
    }

    writer.set_manifest(manifest);
    writer.write_to_file(output_bundle_path);
}

} // namespace strikecem

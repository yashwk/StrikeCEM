#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

namespace strikecem {

#pragma pack(push, 1)
struct ScemPackHeader {
    char magic[8] = {'S', 'C', 'E', 'M', 'P', 'A', 'C', 'K'};
    uint16_t version = 1;
    uint16_t header_size = 256;
    uint32_t endianness_marker = 0x01020304;
    uint32_t entry_count = 0;
    uint32_t flags = 0;
    uint64_t manifest_offset = 0;
    uint64_t manifest_bytes = 0;
    uint64_t table_offset = 0;
    uint64_t table_bytes = 0;
    uint64_t total_bundle_bytes = 0;
    char payload_sha256[64] = {0}; // Hex-encoded SHA-256 over all concatenated entry payloads
    uint8_t reserved[128] = {0};
};
static_assert(sizeof(ScemPackHeader) == 256, "ScemPackHeader must be exactly 256 bytes");
#pragma pack(pop)

struct BundleEntryRecord {
    std::string name;          // e.g. "rcs.scem", "lod0.obj"
    std::string mime_type;     // e.g. "application/x-scem", "model/obj"
    uint64_t offset = 0;       // file offset
    uint64_t size = 0;         // payload size in bytes
    std::string sha256_hex;    // SHA-256 of entry payload
};

class AssetBundleWriter {
public:
    void set_manifest(const nlohmann::json& manifest);
    void add_entry(const std::string& name, const std::string& mime_type, const std::vector<uint8_t>& data);
    void add_entry_text(const std::string& name, const std::string& mime_type, const std::string& text);
    void add_entry_file(const std::string& name, const std::string& mime_type, const std::string& file_path);
    void write_to_file(const std::string& output_path);

private:
    struct PendingEntry {
        std::string name;
        std::string mime_type;
        std::vector<uint8_t> data;
    };
    nlohmann::json manifest_ = nlohmann::json::object();
    std::vector<PendingEntry> entries_;
};

class AssetBundleReader {
public:
    static AssetBundleReader open_file(const std::string& bundle_path, bool verify_sha256 = true);

    const ScemPackHeader& header() const { return header_; }
    const nlohmann::json& manifest() const { return manifest_; }
    const std::vector<BundleEntryRecord>& entries() const { return entries_; }

    bool has_entry(const std::string& name) const;
    const BundleEntryRecord* get_entry_info(const std::string& name) const;

    std::vector<uint8_t> read_entry_bytes(const std::string& name) const;
    std::string read_entry_text(const std::string& name) const;

    void extract_all(const std::string& dest_dir) const;
    bool verify_integrity() const;

private:
    std::string bundle_path_;
    ScemPackHeader header_{};
    nlohmann::json manifest_;
    std::vector<BundleEntryRecord> entries_;
};

// Convenience helper to package an RCS database, primary mesh, and optional LOD geometries into .scempack
void create_strikeengine_package(
    const std::string& output_bundle_path,
    const std::string& rcs_database_path,
    const std::string& primary_mesh_path,
    const std::vector<std::pair<std::string, std::string>>& lod_meshes = {},
    const nlohmann::json& metadata = nlohmann::json::object());

} // namespace strikecem

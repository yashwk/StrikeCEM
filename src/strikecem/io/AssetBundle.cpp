#include "strikecem/io/AssetBundle.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <openssl/evp.h>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/OutputError.hpp"

namespace strikecem {

namespace fs = std::filesystem;

namespace {

constexpr uint64_t kMaxBundleEntries = 100'000;
constexpr uint64_t kMaxManifestBytes = 16 * 1024 * 1024;
constexpr uint64_t kMaxTableBytes = 64 * 1024 * 1024;

bool range_within_file(uint64_t offset, uint64_t size, uint64_t file_size) {
    return offset <= file_size && size <= file_size - offset;
}

bool safe_entry_name(const std::string& name) {
    if (name.empty() || name.find('\0') != std::string::npos ||
        name.find('\\') != std::string::npos) {
        return false;
    }

    const fs::path path(name);
    if (path.is_absolute() || path.has_root_name() || path.has_root_directory()) return false;
    for (const auto& component : path) {
        if (component.empty() || component == "." || component == "..") return false;
    }
    return true;
}

bool is_within_directory(const fs::path& root, const fs::path& candidate) {
    const fs::path relative = candidate.lexically_relative(root);
    return !relative.empty() && *relative.begin() != fs::path("..") && !relative.is_absolute();
}

std::string sha256_range(std::ifstream& in, uint64_t offset, uint64_t size,
                         EVP_MD_CTX* aggregate) {
    EVP_MD_CTX* entry = EVP_MD_CTX_new();
    if (!entry || EVP_DigestInit_ex(entry, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(entry);
        throw OutputError("cannot initialize bundle SHA-256 verification");
    }

    in.clear();
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) {
        EVP_MD_CTX_free(entry);
        throw OutputError("invalid bundle entry offset");
    }

    char buffer[64 * 1024];
    uint64_t remaining = size;
    while (remaining > 0) {
        const auto count = static_cast<std::streamsize>(
            std::min<uint64_t>(remaining, sizeof(buffer)));
        in.read(buffer, count);
        if (in.gcount() != count ||
            EVP_DigestUpdate(entry, buffer, static_cast<size_t>(count)) != 1 ||
            EVP_DigestUpdate(aggregate, buffer, static_cast<size_t>(count)) != 1) {
            EVP_MD_CTX_free(entry);
            throw OutputError("failed reading bundle payload for SHA-256 verification");
        }
        remaining -= static_cast<uint64_t>(count);
    }

    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_size = 0;
    const bool finalized = EVP_DigestFinal_ex(entry, digest, &digest_size) == 1;
    EVP_MD_CTX_free(entry);
    if (!finalized) throw OutputError("cannot finalize bundle SHA-256 verification");

    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_size; ++i)
        result << std::setw(2) << static_cast<unsigned int>(digest[i]);
    return result.str();
}

} // namespace

void AssetBundleWriter::set_manifest(const nlohmann::json& manifest) {
    manifest_ = manifest;
}

void AssetBundleWriter::add_entry(const std::string& name, const std::string& mime_type,
                                 const std::vector<uint8_t>& data) {
    if (!safe_entry_name(name)) throw OutputError("invalid bundle entry path: " + name);
    if (std::any_of(entries_.begin(), entries_.end(), [&](const auto& entry) {
            return entry.name == name;
        })) {
        throw OutputError("duplicate bundle entry path: " + name);
    }
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
    if (entries_.size() > kMaxBundleEntries) {
        throw OutputError("too many entries for asset bundle format");
    }
    ScemPackHeader hdr{};
    hdr.entry_count = static_cast<uint32_t>(entries_.size());

    // Serialize manifest
    std::string manifest_str = manifest_.dump(2);
    if (manifest_str.size() > kMaxManifestBytes)
        throw OutputError("asset bundle manifest exceeds the format size limit");
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
        std::string bytes_str;
        if (!e.data.empty())
            bytes_str.assign(reinterpret_cast<const char*>(e.data.data()), e.data.size());
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

    hdr.table_offset = hdr.manifest_offset + hdr.manifest_bytes;
    hdr.table_bytes = table_json.dump().size();
    if (hdr.table_bytes > kMaxTableBytes)
        throw OutputError("asset bundle table exceeds the format size limit");

    // Offsets affect the serialized table size; iterate until the payload start stabilizes.
    std::string table_str;
    while (true) {
        if (hdr.table_offset > std::numeric_limits<uint64_t>::max() - hdr.table_bytes) {
            throw OutputError("asset bundle table size overflow");
        }
        current_payload_offset = hdr.table_offset + hdr.table_bytes;
        for (size_t i = 0; i < records.size(); ++i) {
            records[i].offset = current_payload_offset;
            table_json[i]["offset"] = current_payload_offset;
            if (records[i].size > std::numeric_limits<uint64_t>::max() - current_payload_offset) {
                throw OutputError("asset bundle payload size overflow");
            }
            current_payload_offset += records[i].size;
        }
        table_str = table_json.dump();
        if (table_str.size() > kMaxTableBytes)
            throw OutputError("asset bundle table exceeds the format size limit");
        if (table_str.size() == hdr.table_bytes) break;
        hdr.table_bytes = table_str.size();
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

    in.seekg(0, std::ios::end);
    const std::streamoff end_pos = in.tellg();
    if (end_pos < 0) throw OutputError("cannot determine asset bundle size: " + bundle_path);
    const uint64_t file_size = static_cast<uint64_t>(end_pos);
    in.seekg(0, std::ios::beg);

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
    if (hdr.endianness_marker != 0x01020304) {
        throw OutputError("unsupported .scempack endianness");
    }
    if (hdr.header_size != sizeof(ScemPackHeader)) {
        throw OutputError("invalid .scempack header size");
    }
    if (hdr.total_bundle_bytes != file_size) {
        throw OutputError("invalid .scempack total size");
    }
    if (hdr.entry_count > kMaxBundleEntries || hdr.manifest_bytes > kMaxManifestBytes ||
        hdr.table_bytes > kMaxTableBytes ||
        hdr.manifest_bytes > std::numeric_limits<size_t>::max() ||
        hdr.table_bytes > std::numeric_limits<size_t>::max()) {
        throw OutputError(".scempack metadata exceeds the supported size limits");
    }
    if (hdr.manifest_offset != sizeof(ScemPackHeader) ||
        !range_within_file(hdr.manifest_offset, hdr.manifest_bytes, file_size)) {
        throw OutputError("invalid .scempack manifest range");
    }
    const uint64_t manifest_end = hdr.manifest_offset + hdr.manifest_bytes;
    if (hdr.table_offset != manifest_end ||
        !range_within_file(hdr.table_offset, hdr.table_bytes, file_size)) {
        throw OutputError("invalid .scempack table range");
    }
    const uint64_t table_end = hdr.table_offset + hdr.table_bytes;

    // Read manifest
    in.seekg(hdr.manifest_offset);
    std::string manifest_str(hdr.manifest_bytes, '\0');
    in.read(manifest_str.data(), hdr.manifest_bytes);
    if (static_cast<uint64_t>(in.gcount()) != hdr.manifest_bytes) {
        throw OutputError("truncated .scempack manifest");
    }
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
    if (static_cast<uint64_t>(in.gcount()) != hdr.table_bytes) {
        throw OutputError("truncated .scempack entry table");
    }
    nlohmann::json table_json;
    try {
        table_json = nlohmann::json::parse(table_str);
    } catch (const std::exception& e) {
        throw OutputError("corrupted bundle table JSON: " + std::string(e.what()));
    }
    if (!table_json.is_array() || table_json.size() != hdr.entry_count) {
        throw OutputError("invalid .scempack entry table");
    }

    std::vector<BundleEntryRecord> entries;
    std::unordered_set<std::string> names;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (const auto& item : table_json) {
        BundleEntryRecord rec;
        rec.name = item["name"].get<std::string>();
        rec.mime_type = item["mime_type"].get<std::string>();
        rec.offset = item["offset"].get<uint64_t>();
        rec.size = item["size"].get<uint64_t>();
        rec.sha256_hex = item["sha256"].get<std::string>();
        if (!safe_entry_name(rec.name)) {
            throw OutputError("invalid .scempack entry path: " + rec.name);
        }
        if (!names.insert(rec.name).second) {
            throw OutputError("duplicate .scempack entry path: " + rec.name);
        }
        if (rec.offset < table_end || !range_within_file(rec.offset, rec.size, file_size)) {
            throw OutputError("invalid .scempack entry range: " + rec.name);
        }
        ranges.emplace_back(rec.offset, rec.offset + rec.size);
        entries.push_back(rec);
    }

    std::sort(ranges.begin(), ranges.end());
    uint64_t previous_end = table_end;
    for (const auto& range : ranges) {
        if (range.first != previous_end)
            throw OutputError("overlapping or non-contiguous .scempack entry ranges");
        previous_end = range.second;
    }
    if (previous_end != file_size) throw OutputError("invalid .scempack payload extent");

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

    if (info->size > std::vector<uint8_t>().max_size() ||
        info->size > std::numeric_limits<size_t>::max()) {
        throw OutputError("bundle entry is too large to read: " + name);
    }
    in.seekg(static_cast<std::streamoff>(info->offset));
    if (!in) throw OutputError("invalid bundle entry offset for: " + name);
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
    const fs::path root = fs::weakly_canonical(dest_dir);
    for (const auto& e : entries_) {
        if (!safe_entry_name(e.name)) {
            throw OutputError("invalid bundle entry path: " + e.name);
        }
        const fs::path relative_path(e.name);
        fs::path parent = root;
        for (const auto& component : relative_path.parent_path()) {
            parent /= component;
            if (fs::exists(parent)) {
                if (fs::is_symlink(parent) || !fs::is_directory(parent)) {
                    throw OutputError("bundle extraction path is not a safe directory: " + e.name);
                }
            } else if (!fs::create_directory(parent) && !fs::is_directory(parent)) {
                throw OutputError("cannot create bundle extraction directory: " + parent.string());
            }
            if (!is_within_directory(root, fs::weakly_canonical(parent))) {
                throw OutputError("bundle extraction path escapes destination: " + e.name);
            }
        }
        fs::path target = root / relative_path;
        if (fs::is_symlink(target)) {
            throw OutputError("bundle extraction path escapes destination: " + e.name);
        }
        if (fs::exists(target) &&
            !is_within_directory(root, fs::weakly_canonical(target))) {
            throw OutputError("bundle extraction path escapes destination: " + e.name);
        }
        if (fs::exists(target)) {
            if (!fs::is_regular_file(target) || !fs::remove(target)) {
                throw OutputError("cannot replace existing bundle path: " + e.name);
            }
        }
        auto data = read_entry_bytes(e.name);
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

    EVP_MD_CTX* aggregate = EVP_MD_CTX_new();
    if (!aggregate || EVP_DigestInit_ex(aggregate, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(aggregate);
        throw OutputError("cannot initialize bundle SHA-256 verification");
    }
    for (const auto& e : entries_) {
        std::string entry_hash;
        try {
            entry_hash = sha256_range(in, e.offset, e.size, aggregate);
        } catch (...) {
            EVP_MD_CTX_free(aggregate);
            throw;
        }
        if (entry_hash != e.sha256_hex) {
            EVP_MD_CTX_free(aggregate);
            throw OutputError("SHA-256 verification failed for entry: " + e.name);
        }
    }

    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_size = 0;
    const bool finalized = EVP_DigestFinal_ex(aggregate, digest, &digest_size) == 1;
    EVP_MD_CTX_free(aggregate);
    if (!finalized || digest_size != 32) {
        throw OutputError("cannot finalize bundle SHA-256 verification");
    }
    static constexpr char hex[] = "0123456789abcdef";
    char total_hash[64];
    for (size_t i = 0; i < 32; ++i) {
        total_hash[i * 2] = hex[digest[i] >> 4];
        total_hash[i * 2 + 1] = hex[digest[i] & 0x0f];
    }
    if (std::memcmp(header_.payload_sha256, total_hash, sizeof(total_hash)) != 0) {
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

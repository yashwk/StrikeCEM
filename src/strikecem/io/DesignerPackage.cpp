// StrikeDesigner export-package validation (SPEC FR-12).
#include "strikecem/io/DesignerPackage.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"

namespace strikecem {
namespace {

namespace fs = std::filesystem;
using nlohmann::json;

std::string read_text_file(const fs::path& path, const char* what) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw ConfigError(std::string("cannot open ") + what + ": " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

const char* kSupportedCoordinateSystem = "strike-right-handed-z-up";

} // namespace

DesignerIdentity validate_designer_package(const nlohmann::json& resolved,
                                           const std::string& config_path) {
    DesignerIdentity identity;
    if (resolved.contains("integration") && resolved["integration"].contains("designer")) {
        const auto& d = resolved["integration"]["designer"];
        identity.design_id = d["design_id"].get<std::string>();
        identity.revision = d["revision"].get<std::string>();
        if (d.contains("export_id")) identity.export_id = d["export_id"].get<std::string>();
    }
    const fs::path config_dir = fs::path(config_path).parent_path();
    const fs::path root = config_dir.empty() ? fs::current_path() : config_dir;
    const fs::path manifest_path = root / "export_manifest.json";
    std::error_code ec;
    if (!fs::exists(manifest_path, ec)) return identity; // direct config mode
    identity.packaged = true;

    const std::string manifest_bytes = read_text_file(manifest_path, "export manifest");
    identity.manifest_hash = sha256_hex(manifest_bytes);
    json manifest;
    try {
        manifest = json::parse(manifest_bytes);
    } catch (const std::exception& e) {
        throw ConfigError(std::string("invalid export manifest JSON: ") + e.what());
    }
    auto required_str = [&](const char* key) {
        if (!manifest.contains(key) || !manifest[key].is_string() ||
            manifest[key].get<std::string>().empty())
            throw ConfigError(std::string("export manifest lacks '") + key + "'");
        return manifest[key].get<std::string>();
    };
    if (required_str("export_format_version") != "1.0")
        throw ConfigError("unsupported export format version (supported: 1.0)");
    const std::string design_id = required_str("design_id");
    const std::string revision = required_str("design_revision");
    required_str("designer_version");
    required_str("units");
    const std::string coords = required_str("source_coordinate_system");
    if (coords != kSupportedCoordinateSystem)
        throw ConfigError(std::string("unsupported source coordinate system '") + coords +
                          "' (supported: " + kSupportedCoordinateSystem + ")");
    const std::string geometry_rel = required_str("geometry_path");
    const fs::path geometry_path =
        fs::path(geometry_rel).is_absolute() ? fs::path(geometry_rel) : root / geometry_rel;
    if (!fs::exists(geometry_path, ec))
        throw MeshError("export manifest geometry not found: " + geometry_path.string());

    // Geometry hash pins the exact bytes; fail closed on mismatch or stale export.
    std::ifstream geom(geometry_path, std::ios::binary);
    std::ostringstream ss;
    ss << geom.rdbuf();
    if (sha256_hex(ss.str()) != required_str("geometry_sha256"))
        throw MeshError("export manifest geometry hash mismatch: " + geometry_path.string());

    // The manifest geometry, units, and identity must agree with the config.
    const std::string model_rel = resolved["model"]["path"].get<std::string>();
    const fs::path model_path =
        fs::path(model_rel).is_absolute() ? fs::path(model_rel) : root / model_rel;
    std::error_code ec2;
    if (!fs::equivalent(model_path, geometry_path, ec2) || ec2)
        throw ConfigError("config model.path does not match manifest geometry_path");
    if (manifest["units"].get<std::string>() != resolved["model"]["units"].get<std::string>())
        throw ConfigError("config model.units does not match manifest units");
    if (!identity.design_id.empty() &&
        (identity.design_id != design_id || identity.revision != revision))
        throw ConfigError("config designer identity does not match export manifest");
    identity.design_id = design_id;
    identity.revision = revision;

    // Component groups are traceability metadata in v1 (all surfaces PEC):
    // verified against OBJ group names, warned — never silently dropped.
    if (manifest.contains("component_manifest")) {
        std::string mesh_ext = model_path.extension().string();
        std::transform(mesh_ext.begin(), mesh_ext.end(), mesh_ext.begin(),
                       [](char c) { return std::tolower(c); });
        if (mesh_ext != ".obj") {
            identity.warnings.push_back("component manifest cannot be verified against " +
                                        mesh_ext + " meshes");
        } else {
            std::vector<std::string> groups;
            try {
                groups = parse_obj(model_path).groups;
            } catch (const MeshLoadError& e) {
                throw MeshError(e.what());
            }
            for (const auto& component : manifest["component_manifest"]) {
                if (!component.contains("group")) continue;
                const std::string group = component["group"].get<std::string>();
                if (std::find(groups.begin(), groups.end(), group) == groups.end())
                    identity.warnings.push_back("manifest component group '" + group +
                                                "' not found in mesh");
            }
        }
    }
    return identity;
}

} // namespace strikecem

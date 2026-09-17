#pragma once
// StrikeDesigner export-package validation (SPEC FR-12): manifest checks,
// geometry hash verification, unit/coordinate agreement, and design
// identity resolution. Direct (non-packaged) configs pass through with
// identity taken from the config block when present.
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace strikecem {

struct DesignerIdentity {
    bool packaged = false;
    std::string design_id;
    std::string revision;
    std::string export_id;
    std::string manifest_hash; // SHA-256 of the manifest bytes
    std::vector<std::string> warnings; // unknown component groups, STL limits
};

// Validates the package in config_dir when export_manifest.json sits next
// to the config file; otherwise returns the direct-mode identity.
// Throws ConfigError (exit 2) or MeshError (exit 3); never warns-and-runs
// on identity or hash mismatches.
DesignerIdentity validate_designer_package(const nlohmann::json& resolved,
                                           const std::string& config_path);

} // namespace strikecem

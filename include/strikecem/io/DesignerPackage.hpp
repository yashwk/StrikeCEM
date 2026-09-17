#pragma once
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace strikecem {

struct DesignerIdentity {
    bool packaged = false;
    std::string design_id;
    std::string revision;
    std::string export_id;
    std::string manifest_hash;
    std::vector<std::string> warnings;
};

DesignerIdentity validate_designer_package(const nlohmann::json& resolved,
                                           const std::string& config_path);

}

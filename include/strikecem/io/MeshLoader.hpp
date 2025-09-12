#pragma once
#include <string>

namespace strikecem {

    struct MeshStats {
        size_t facets = 0;
        double bbox_min[3] = {0,0,0};
        double bbox_max[3] = {0,0,0};
    };

    MeshStats load_mesh_stats(const std::string &path);

} // namespace strikecem

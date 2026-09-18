#pragma once
#include <complex>
#include <string>
#include <vector>

#include "strikecem/solvers/Fresnel.hpp"

namespace strikecem {

enum class WallType { Pec, Dielectric, Coated };

struct MaterialEntry {
    double frequency_hz = 0.0;
    ComplexMedium medium;
    double sigma = 0.0;
};

struct CoatingLayer {
    double thickness_m = 0.0;
    ComplexMedium medium;
};

struct MaterialModel {
    std::string tag;
    WallType type = WallType::Dielectric;
    std::vector<MaterialEntry> table;
    std::vector<CoatingLayer> layers;
};

void validate_material(const MaterialModel& model);

ComplexMedium effective_medium(const MaterialEntry& entry, double frequency_hz);

ComplexMedium evaluate_material(const MaterialModel& model, double frequency_hz);

} // namespace strikecem

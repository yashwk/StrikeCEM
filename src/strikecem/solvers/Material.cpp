#include "strikecem/solvers/Material.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

#include "strikecem/core/Conventions.hpp"

namespace strikecem {
namespace {

constexpr double kPassiveTol = 1e-12;

bool finite_c(const std::complex<double>& z) { return std::isfinite(z.real() + z.imag()); }

void check_passive(const ComplexMedium& m) {
    if (!finite_c(m.eps_r) || !finite_c(m.mu_r))
        throw std::invalid_argument("material constants must be finite");
    if (m.eps_r.imag() > kPassiveTol || m.mu_r.imag() > kPassiveTol)
        throw std::invalid_argument("material must be passive");
}

} // namespace

void validate_material(const MaterialModel& model) {
    if (model.tag.empty()) throw std::invalid_argument("material tag must be non-empty");
    for (const auto& layer : model.layers) {
        if (!(layer.thickness_m > 0.0) || !std::isfinite(layer.thickness_m))
            throw std::invalid_argument("coating thickness must be positive and finite");
        check_passive(layer.medium);
    }
    if (model.type == WallType::Pec) {
        if (!model.table.empty() || !model.layers.empty())
            throw std::invalid_argument("PEC is bare by definition");
        return;
    }
    if (model.type == WallType::Dielectric && !model.layers.empty())
        throw std::invalid_argument("layered walls use the coated type");
    if (model.type == WallType::Coated && model.layers.empty())
        throw std::invalid_argument("coated walls need at least one layer");
    if (model.type == WallType::Dielectric && model.table.empty())
        throw std::invalid_argument("dielectric walls need a property table");
    for (size_t i = 0; i < model.table.size(); ++i) {
        const auto& e = model.table[i];
        if (!(e.frequency_hz > 0.0) || !std::isfinite(e.frequency_hz))
            throw std::invalid_argument("table frequencies must be positive and finite");
        if (i > 0 && !(e.frequency_hz > model.table[i - 1].frequency_hz))
            throw std::invalid_argument("table frequencies must be strictly ascending");
        check_passive(e.medium);
        if (!(e.sigma >= 0.0) || !std::isfinite(e.sigma))
            throw std::invalid_argument("conductivity must be non-negative and finite");
    }
}

ComplexMedium effective_medium(const MaterialEntry& entry, double frequency_hz) {
    if (!(frequency_hz > 0.0) || !std::isfinite(frequency_hz))
        throw std::invalid_argument("frequency must be positive and finite");
    check_passive(entry.medium);
    if (!(entry.sigma >= 0.0) || !std::isfinite(entry.sigma))
        throw std::invalid_argument("conductivity must be non-negative and finite");
    const double eps0 = 1.0 / (kEta0 * kSpeedOfLight);
    ComplexMedium out = entry.medium;
    out.eps_r -= std::complex<double>(0.0, entry.sigma / (2.0 * std::numbers::pi *
                                                           frequency_hz * eps0));
    return out;
}

ComplexMedium evaluate_material(const MaterialModel& model, double frequency_hz) {
    validate_material(model);
    if (!(frequency_hz > 0.0) || !std::isfinite(frequency_hz))
        throw std::invalid_argument("frequency must be positive and finite");
    if (model.table.empty())
        throw std::invalid_argument("no property table to evaluate");
    if (model.table.size() == 1) return effective_medium(model.table[0], frequency_hz);
    if (frequency_hz < model.table.front().frequency_hz ||
        frequency_hz > model.table.back().frequency_hz)
        throw std::invalid_argument("frequency outside material table range");
    size_t hi = 1;
    while (hi + 1 < model.table.size() && model.table[hi].frequency_hz < frequency_hz) ++hi;
    const MaterialEntry& lo = model.table[hi - 1];
    const MaterialEntry& up = model.table[hi];
    const double t = (frequency_hz - lo.frequency_hz) / (up.frequency_hz - lo.frequency_hz);
    MaterialEntry mix;
    mix.frequency_hz = frequency_hz;
    mix.medium.eps_r = lo.medium.eps_r + (up.medium.eps_r - lo.medium.eps_r) * t;
    mix.medium.mu_r = lo.medium.mu_r + (up.medium.mu_r - lo.medium.mu_r) * t;
    mix.sigma = lo.sigma + (up.sigma - lo.sigma) * t;
    return effective_medium(mix, frequency_hz);
}

} // namespace strikecem

#pragma once

#include <atomic>
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/api/strikecem_c_api.h"
#include "strikecem/core/Config.hpp"
#include "strikecem/io/BinaryTransport.hpp"
#include "strikecem/io/Hdf5Reader.hpp"

namespace strikecem {

enum class InterpolationMode {
    Nearest = SCEM_INTERP_NEAREST,
    Linear = SCEM_INTERP_LINEAR,
    Bicubic = SCEM_INTERP_BICUBIC,
    Spherical = SCEM_INTERP_SPHERICAL
};

class RcsQueryEngine {
public:
    static std::unique_ptr<RcsQueryEngine> open_file(const std::string& filepath);

    explicit RcsQueryEngine(ScemDatabase db);
    explicit RcsQueryEngine(Hdf5Database db);

    const std::string& design_id() const { return design_id_; }
    const std::string& design_revision() const { return design_revision_; }
    const std::vector<double>& frequencies() const { return frequencies_; }
    const std::vector<Direction>& directions() const { return directions_; }
    const std::vector<std::string>& polarizations() const { return polarizations_; }

    InterpolationMode interpolation_mode() const { return interp_mode_.load(std::memory_order_relaxed); }
    void set_interpolation_mode(InterpolationMode mode) { interp_mode_.store(mode, std::memory_order_relaxed); }

    bool verify_design(const std::string& expected_id, const std::string& expected_rev) const;

    scem_query_result_t query(double az_deg, double el_deg, double freq_hz,
                              const std::string& pol) const;
    scem_query_result_t query(double az_deg, double el_deg, double freq_hz,
                              const std::string& pol,
                              InterpolationMode mode) const;

private:
    void init_index();
    std::complex<double> query_direction(size_t fi, size_t pol_idx, double az_deg, double el_deg,
                                         InterpolationMode mode) const;
    std::complex<double> interpolate_regular_bicubic(size_t fi, size_t pol_idx, double az_deg, double el_deg) const;
    std::complex<double> interpolate_spherical(size_t fi, size_t pol_idx, double az_deg, double el_deg) const;

    inline const std::complex<double>& sample(size_t fi, size_t pol_idx, size_t dir_idx) const {
        return flat_samples_[(fi * num_pol_ + pol_idx) * num_dir_ + dir_idx];
    }
    inline std::complex<double>& sample(size_t fi, size_t pol_idx, size_t dir_idx) {
        return flat_samples_[(fi * num_pol_ + pol_idx) * num_dir_ + dir_idx];
    }
    inline size_t az_el_dir(size_t ai, size_t ei) const {
        return az_el_to_dir_[ai * unique_el_.size() + ei];
    }

    std::string design_id_;
    std::string design_revision_;
    std::string export_id_;

    std::vector<double> frequencies_;
    std::vector<Direction> directions_;
    std::vector<std::string> polarizations_;

    // Flat 1D sample storage: [ (freq_idx * npol + pol_idx) * ndir + dir_idx ]
    std::vector<std::complex<double>> flat_samples_;
    size_t num_freq_ = 0;
    size_t num_pol_ = 0;
    size_t num_dir_ = 0;

    // Grid properties if regular
    bool is_regular_grid_ = false;
    std::vector<double> unique_az_;
    std::vector<double> unique_el_;
    // Mapping from (az_idx, el_idx) -> dir_idx (flat 1D: az_idx * Nel + el_idx)
    std::vector<size_t> az_el_to_dir_;

    std::atomic<InterpolationMode> interp_mode_{InterpolationMode::Bicubic};
};

} // namespace strikecem

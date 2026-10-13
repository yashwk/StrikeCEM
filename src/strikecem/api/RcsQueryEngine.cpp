#include "strikecem/api/RcsQueryEngine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>

#include "strikecem/core/Conventions.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace fs = std::filesystem;

namespace strikecem {
namespace {

thread_local std::string g_last_error;

void set_last_error(const std::string& err) {
    g_last_error = err;
}

double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(v, hi));
}

} // namespace

std::unique_ptr<RcsQueryEngine> RcsQueryEngine::open_file(const std::string& filepath) {
    if (!fs::exists(filepath)) {
        throw ReaderError("database file does not exist: " + filepath);
    }
    const std::string ext = fs::path(filepath).extension().string();
    if (ext == ".scem") {
        ScemDatabase db = read_scem_database(filepath);
        validate_scem_complete(db);
        return std::make_unique<RcsQueryEngine>(std::move(db));
    } else {
        Hdf5Database db = read_hdf5_database(filepath);
        validate_hdf5_complete(db);
        return std::make_unique<RcsQueryEngine>(std::move(db));
    }
}

RcsQueryEngine::RcsQueryEngine(ScemDatabase db)
    : frequencies_(std::move(db.frequencies_hz)),
      directions_(std::move(db.directions)),
      polarizations_(std::move(db.polarizations)) {
    if (db.metadata.contains("design_id")) {
        design_id_ = db.metadata["design_id"].get<std::string>();
    }
    if (db.metadata.contains("design_revision")) {
        design_revision_ = db.metadata["design_revision"].get<std::string>();
    }
    if (db.metadata.contains("export_id")) {
        export_id_ = db.metadata["export_id"].get<std::string>();
    }
    if (design_id_.empty() && db.metadata.contains("resolved_config") &&
        db.metadata["resolved_config"].contains("integration") &&
        db.metadata["resolved_config"]["integration"].contains("designer")) {
        const auto& d = db.metadata["resolved_config"]["integration"]["designer"];
        design_id_ = d.value("design_id", "");
        design_revision_ = d.value("revision", "");
        export_id_ = d.value("export_id", "");
    }
    if (design_id_.empty() && db.metadata.contains("integration") &&
        db.metadata["integration"].contains("designer")) {
        const auto& d = db.metadata["integration"]["designer"];
        design_id_ = d.value("design_id", "");
        design_revision_ = d.value("revision", "");
        export_id_ = d.value("export_id", "");
    }

    const size_t nfreq = frequencies_.size();
    const size_t npol = polarizations_.size();
    const size_t ndir = directions_.size();

    grid_samples_.resize(nfreq, std::vector<std::vector<std::complex<double>>>(
                                    npol, std::vector<std::complex<double>>(ndir)));

    for (const auto& r : db.rows) {
        if (r.frequency_index < nfreq && r.polarization_index < npol && r.direction_index < ndir) {
            grid_samples_[r.frequency_index][r.polarization_index][r.direction_index] =
                {r.scattering_real, r.scattering_imag};
        }
    }

    init_index();
}

RcsQueryEngine::RcsQueryEngine(Hdf5Database db)
    : design_id_(std::move(db.design_id)),
      design_revision_(std::move(db.design_revision)),
      frequencies_(std::move(db.frequencies_hz)),
      polarizations_(std::move(db.polarizations)) {
    directions_.resize(db.azimuth_deg.size());
    for (size_t i = 0; i < db.azimuth_deg.size(); ++i) {
        directions_[i].id = static_cast<uint32_t>(i);
        directions_[i].azimuth_deg = db.azimuth_deg[i];
        directions_[i].elevation_deg = db.elevation_deg[i];
        directions_[i].k_hat = direction_from_az_el(db.azimuth_deg[i], db.elevation_deg[i]);
    }

    const size_t nfreq = frequencies_.size();
    const size_t npol = polarizations_.size();
    const size_t ndir = directions_.size();

    grid_samples_.resize(nfreq, std::vector<std::vector<std::complex<double>>>(
                                    npol, std::vector<std::complex<double>>(ndir)));

    for (const auto& r : db.rows) {
        if (r.frequency_index < nfreq && r.polarization_index < npol && r.direction_index < ndir) {
            grid_samples_[r.frequency_index][r.polarization_index][r.direction_index] =
                {r.scattering_real, r.scattering_imag};
        }
    }

    init_index();
}

void RcsQueryEngine::init_index() {
    if (directions_.empty()) return;

    std::vector<double> azs, els;
    azs.reserve(directions_.size());
    els.reserve(directions_.size());
    for (const auto& d : directions_) {
        azs.push_back(d.azimuth_deg);
        els.push_back(d.elevation_deg);
    }
    std::sort(azs.begin(), azs.end());
    azs.erase(std::unique(azs.begin(), azs.end(), [](double a, double b) {
        return std::abs(a - b) < 1e-7;
    }), azs.end());

    std::sort(els.begin(), els.end());
    els.erase(std::unique(els.begin(), els.end(), [](double a, double b) {
        return std::abs(a - b) < 1e-7;
    }), els.end());

    if (azs.size() * els.size() == directions_.size()) {
        is_regular_grid_ = true;
        unique_az_ = std::move(azs);
        unique_el_ = std::move(els);
        az_el_to_dir_.assign(unique_az_.size(), std::vector<size_t>(unique_el_.size(), 0));

        for (size_t di = 0; di < directions_.size(); ++di) {
            const auto& d = directions_[di];
            auto it_az = std::lower_bound(unique_az_.begin(), unique_az_.end(), d.azimuth_deg - 1e-7);
            auto it_el = std::lower_bound(unique_el_.begin(), unique_el_.end(), d.elevation_deg - 1e-7);
            if (it_az != unique_az_.end() && it_el != unique_el_.end()) {
                const size_t ai = std::distance(unique_az_.begin(), it_az);
                const size_t ei = std::distance(unique_el_.begin(), it_el);
                az_el_to_dir_[ai][ei] = di;
            }
        }
    } else {
        is_regular_grid_ = false;
    }
}

bool RcsQueryEngine::verify_design(const std::string& expected_id,
                                  const std::string& expected_rev) const {
    if (!expected_id.empty() && design_id_ != expected_id) {
        return false;
    }
    if (!expected_rev.empty() && design_revision_ != expected_rev) {
        return false;
    }
    return true;
}

scem_query_result_t RcsQueryEngine::query(double az_deg, double el_deg, double freq_hz,
                                         const std::string& pol) const {
    if (frequencies_.empty() || directions_.empty() || polarizations_.empty()) {
        throw ReaderError("empty database");
    }

    // Find polarization channel
    size_t pol_idx = 0;
    bool found_pol = false;
    for (size_t i = 0; i < polarizations_.size(); ++i) {
        if (polarizations_[i] == pol) {
            pol_idx = i;
            found_pol = true;
            break;
        }
    }
    if (!found_pol) {
        throw ReaderError("polarization channel not found: " + pol);
    }

    // Direction query helper for a specific frequency index
    auto query_dir = [&](size_t fi) -> std::complex<double> {
        if (is_regular_grid_) {
            if (unique_az_.size() > 1 && unique_el_.size() > 1) {
                // Find azimuth bracket
                const double az_clamped = clamp(az_deg, unique_az_.front(), unique_az_.back());
                const double el_clamped = clamp(el_deg, unique_el_.front(), unique_el_.back());

                auto it_az = std::upper_bound(unique_az_.begin(), unique_az_.end(), az_clamped);
                size_t a1 = (it_az == unique_az_.end()) ? unique_az_.size() - 1 : std::distance(unique_az_.begin(), it_az);
                size_t a0 = (a1 == 0) ? 0 : a1 - 1;

                auto it_el = std::upper_bound(unique_el_.begin(), unique_el_.end(), el_clamped);
                size_t e1 = (it_el == unique_el_.end()) ? unique_el_.size() - 1 : std::distance(unique_el_.begin(), it_el);
                size_t e0 = (e1 == 0) ? 0 : e1 - 1;

                const double daz = unique_az_[a1] - unique_az_[a0];
                const double u = (daz > 1e-9) ? (az_clamped - unique_az_[a0]) / daz : 0.0;

                const double del = unique_el_[e1] - unique_el_[e0];
                const double v = (del > 1e-9) ? (el_clamped - unique_el_[e0]) / del : 0.0;

                const size_t d00 = az_el_to_dir_[a0][e0];
                const size_t d10 = az_el_to_dir_[a1][e0];
                const size_t d01 = az_el_to_dir_[a0][e1];
                const size_t d11 = az_el_to_dir_[a1][e1];

                const auto& s00 = grid_samples_[fi][pol_idx][d00];
                const auto& s10 = grid_samples_[fi][pol_idx][d10];
                const auto& s01 = grid_samples_[fi][pol_idx][d01];
                const auto& s11 = grid_samples_[fi][pol_idx][d11];

                return (1.0 - u) * (1.0 - v) * s00 +
                       u * (1.0 - v) * s10 +
                       (1.0 - u) * v * s01 +
                       u * v * s11;
            } else if (unique_az_.size() > 1 && unique_el_.size() == 1) {
                // 1D azimuth interpolation
                const double az_clamped = clamp(az_deg, unique_az_.front(), unique_az_.back());
                auto it_az = std::upper_bound(unique_az_.begin(), unique_az_.end(), az_clamped);
                size_t a1 = (it_az == unique_az_.end()) ? unique_az_.size() - 1 : std::distance(unique_az_.begin(), it_az);
                size_t a0 = (a1 == 0) ? 0 : a1 - 1;
                const double daz = unique_az_[a1] - unique_az_[a0];
                const double u = (daz > 1e-9) ? (az_clamped - unique_az_[a0]) / daz : 0.0;
                const size_t d0 = az_el_to_dir_[a0][0];
                const size_t d1 = az_el_to_dir_[a1][0];
                return (1.0 - u) * grid_samples_[fi][pol_idx][d0] + u * grid_samples_[fi][pol_idx][d1];
            } else if (unique_az_.size() == 1 && unique_el_.size() > 1) {
                // 1D elevation interpolation
                const double el_clamped = clamp(el_deg, unique_el_.front(), unique_el_.back());
                auto it_el = std::upper_bound(unique_el_.begin(), unique_el_.end(), el_clamped);
                size_t e1 = (it_el == unique_el_.end()) ? unique_el_.size() - 1 : std::distance(unique_el_.begin(), it_el);
                size_t e0 = (e1 == 0) ? 0 : e1 - 1;
                const double del = unique_el_[e1] - unique_el_[e0];
                const double v = (del > 1e-9) ? (el_clamped - unique_el_[e0]) / del : 0.0;
                const size_t d0 = az_el_to_dir_[0][e0];
                const size_t d1 = az_el_to_dir_[0][e1];
                return (1.0 - v) * grid_samples_[fi][pol_idx][d0] + v * grid_samples_[fi][pol_idx][d1];
            } else {
                return grid_samples_[fi][pol_idx][az_el_to_dir_[0][0]];
            }
        } else {
            // Distance in (az, el) angle space
            size_t best_idx = 0;
            double min_dist_sq = 1e30;
            for (size_t di = 0; di < directions_.size(); ++di) {
                const double daz = directions_[di].azimuth_deg - az_deg;
                const double del = directions_[di].elevation_deg - el_deg;
                const double dsq = daz * daz + del * del;
                if (dsq < min_dist_sq) {
                    min_dist_sq = dsq;
                    best_idx = di;
                }
            }
            return grid_samples_[fi][pol_idx][best_idx];
        }
    };

    // Frequency interpolation
    std::complex<double> final_s;
    if (frequencies_.size() == 1 || freq_hz <= frequencies_.front()) {
        final_s = query_dir(0);
    } else if (freq_hz >= frequencies_.back()) {
        final_s = query_dir(frequencies_.size() - 1);
    } else {
        auto it_f = std::upper_bound(frequencies_.begin(), frequencies_.end(), freq_hz);
        const size_t f1 = std::distance(frequencies_.begin(), it_f);
        const size_t f0 = f1 - 1;
        const double w = (freq_hz - frequencies_[f0]) / (frequencies_[f1] - frequencies_[f0]);
        const auto s0 = query_dir(f0);
        const auto s1 = query_dir(f1);
        final_s = (1.0 - w) * s0 + w * s1;
    }

    scem_query_result_t res;
    res.scattering_real = final_s.real();
    res.scattering_imag = final_s.imag();
    res.rcs_sqm = 4.0 * std::numbers::pi * std::norm(final_s); // 4*pi*|S|^2
    res.rcs_dbsm = rcs_sqm_to_dbsm(res.rcs_sqm);
    return res;
}

} // namespace strikecem

// --- C ABI Implementation ---

struct scem_database {
    std::unique_ptr<strikecem::RcsQueryEngine> engine;
};

extern "C" {

SCEM_API scem_status_t scem_database_open(const char* path, scem_database_t** out_db) {
    if (!path || !out_db) {
        strikecem::set_last_error("null path or out_db pointer");
        return SCEM_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto engine = strikecem::RcsQueryEngine::open_file(path);
        auto* db = new scem_database{std::move(engine)};
        *out_db = db;
        return SCEM_OK;
    } catch (const strikecem::ReaderError& e) {
        strikecem::set_last_error(e.what());
        return SCEM_ERROR_CORRUPT_DATABASE;
    } catch (const std::exception& e) {
        strikecem::set_last_error(e.what());
        return SCEM_ERROR_INTERNAL;
    }
}

SCEM_API void scem_database_close(scem_database_t* db) {
    if (db) {
        delete db;
    }
}

SCEM_API scem_status_t scem_database_get_design_id(scem_database_t* db, char* buf, size_t buf_size) {
    if (!db || !buf || buf_size == 0) return SCEM_ERROR_INVALID_ARGUMENT;
    const auto& id = db->engine->design_id();
    std::strncpy(buf, id.c_str(), buf_size - 1);
    buf[buf_size - 1] = '\0';
    return SCEM_OK;
}

SCEM_API scem_status_t scem_database_get_design_revision(scem_database_t* db, char* buf, size_t buf_size) {
    if (!db || !buf || buf_size == 0) return SCEM_ERROR_INVALID_ARGUMENT;
    const auto& rev = db->engine->design_revision();
    std::strncpy(buf, rev.c_str(), buf_size - 1);
    buf[buf_size - 1] = '\0';
    return SCEM_OK;
}

SCEM_API scem_status_t scem_database_verify_design(scem_database_t* db, const char* expected_design_id, const char* expected_revision) {
    if (!db) return SCEM_ERROR_INVALID_ARGUMENT;
    const std::string id = expected_design_id ? expected_design_id : "";
    const std::string rev = expected_revision ? expected_revision : "";
    if (!db->engine->verify_design(id, rev)) {
        strikecem::set_last_error("design id or revision does not match database");
        return SCEM_ERROR_DESIGN_MISMATCH;
    }
    return SCEM_OK;
}

SCEM_API scem_status_t scem_database_get_frequencies(scem_database_t* db, double* out_freqs, size_t* inout_count) {
    if (!db || !inout_count) return SCEM_ERROR_INVALID_ARGUMENT;
    const auto& freqs = db->engine->frequencies();
    const size_t n = std::min(*inout_count, freqs.size());
    if (out_freqs) {
        std::copy(freqs.begin(), freqs.begin() + n, out_freqs);
    }
    *inout_count = freqs.size();
    return SCEM_OK;
}

SCEM_API scem_status_t scem_database_get_polarizations(scem_database_t* db, char out_pols[][8], size_t* inout_count) {
    if (!db || !inout_count) return SCEM_ERROR_INVALID_ARGUMENT;
    const auto& pols = db->engine->polarizations();
    const size_t n = std::min(*inout_count, pols.size());
    if (out_pols) {
        for (size_t i = 0; i < n; ++i) {
            std::strncpy(out_pols[i], pols[i].c_str(), 7);
            out_pols[i][7] = '\0';
        }
    }
    *inout_count = pols.size();
    return SCEM_OK;
}

SCEM_API scem_status_t scem_database_query(scem_database_t* db, double az_deg, double el_deg,
                                          double freq_hz, const char* pol,
                                          scem_query_result_t* out_result) {
    if (!db || !pol || !out_result) {
        strikecem::set_last_error("invalid argument to scem_database_query");
        return SCEM_ERROR_INVALID_ARGUMENT;
    }
    try {
        *out_result = db->engine->query(az_deg, el_deg, freq_hz, pol);
        return SCEM_OK;
    } catch (const strikecem::ReaderError& e) {
        strikecem::set_last_error(e.what());
        return SCEM_ERROR_CHANNEL_NOT_FOUND;
    } catch (const std::exception& e) {
        strikecem::set_last_error(e.what());
        return SCEM_ERROR_INTERNAL;
    }
}

SCEM_API scem_status_t scem_database_query_batch(scem_database_t* db,
                                                const scem_query_request_t* requests,
                                                size_t count,
                                                scem_query_result_t* out_results) {
    if (!db || !requests || !out_results) return SCEM_ERROR_INVALID_ARGUMENT;
    for (size_t i = 0; i < count; ++i) {
        const auto& req = requests[i];
        try {
            out_results[i] = db->engine->query(req.azimuth_deg, req.elevation_deg,
                                               req.frequency_hz, req.polarization);
        } catch (const std::exception& e) {
            strikecem::set_last_error(e.what());
            return SCEM_ERROR_INTERNAL;
        }
    }
    return SCEM_OK;
}

SCEM_API const char* scem_get_last_error(void) {
    return strikecem::g_last_error.c_str();
}

} // extern "C"

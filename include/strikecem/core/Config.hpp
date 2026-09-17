#pragma once
// v1 configuration loader: parse, schema-validate, resolve defaults,
// cross-field checks, canonical hash, and sample-plan expansion.
// SPEC FR-1. All execution uses the resolved configuration.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/core/Conventions.hpp"

namespace strikecem {

// SPEC FR-11 exit codes.
enum class ExitCode : int { Ok = 0, Config = 2, Mesh = 3, Resource = 4, Solver = 5, Output = 6 };

class ConfigError : public std::runtime_error {
  public:
    explicit ConfigError(const std::string& msg) : std::runtime_error(msg) {}
};

class MeshError : public std::runtime_error {
  public:
    explicit MeshError(const std::string& msg) : std::runtime_error(msg) {}
};

struct Direction {
    uint32_t id = 0;
    double azimuth_deg = 0.0;
    double elevation_deg = 0.0;
    geom::Vec3d k_hat;
};

struct SamplePlan {
    std::vector<double> frequencies_hz;
    std::vector<Direction> directions;
    std::vector<std::string> polarizations;
    std::vector<std::string> warnings;
    std::string hash; // SHA-256 over the canonical plan encoding
    uint64_t sample_count() const {
        return static_cast<uint64_t>(frequencies_hz.size()) * directions.size() *
               polarizations.size();
    }
};

struct ResolvedConfig {
    nlohmann::json value; // defaults materialized; the execution contract
    std::string hash;     // SHA-256 hex of the canonical serialization
    std::string schema_version = "1.0";
    // Run-control path, validated by the schema but stripped from value so
    // resuming the same database does not change its identity hash.
    std::string resume_from_checkpoint;
};

// Parse, schema-validate, resolve, and hash. Relative model paths resolve
// against the configuration file's directory. Throws ConfigError (exit 2)
// or MeshError (exit 3).
ResolvedConfig load_config(const std::string& config_path, const std::string& schema_path);

// Canonical bytes: nlohmann::json objects are std::map-ordered, so dump()
// has sorted keys and stable number formatting.
std::string canonical_json(const nlohmann::json& value);
std::string sha256_hex(const std::string& bytes);

SamplePlan build_sample_plan(const nlohmann::json& resolved);

} // namespace strikecem

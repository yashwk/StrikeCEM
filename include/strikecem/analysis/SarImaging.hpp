#pragma once
// General Synthetic Aperture Radar (SAR) Simulation & Image Formation Engine.
// Models spaceborne (LEO Ka/X-band satellite) and airborne (UAV/aircraft)
// spotlight, stripmap, and circular SAR imaging of ground entities and airframes.
// Simulates direct target scattering, geometric ground shadow casting,
// vehicle-ground dihedral multipath, and speckle ground clutter.
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/io/Hdf5Reader.hpp"
#include "strikecem/rendering/Colormap.hpp"
#include "strikecem/rendering/Image.hpp"
#include "strikecem/solvers/Material.hpp"

namespace strikecem::analysis {

enum class SarPlatform {
    Spaceborne, // LEO/MEO Satellite (altitude 300 - 800 km, velocity ~7.6 km/s)
    Airborne,   // UAV, reconnaissance aircraft (altitude 1 - 20 km, velocity 50 - 300 m/s)
    Terrestrial // Rail/gantry, tower, or stationary ground radar
};

enum class SarMode {
    Spotlight,  // Staring / steered beam: maximizes aperture angle and azimuth resolution
    Stripmap,   // Fixed side-looking broadside/squinted beam
    Circular    // CSAR: 360-degree circular orbit
};

enum class SarGroundType {
    None,       // Free-space target (in-flight entity, no ground plane)
    Tarmac,     // Concrete / asphalt apron or runway (clutter sigma0 ~ -20 dBsm/m^2)
    Soil,       // Dry soil / sand / gravel (clutter sigma0 ~ -15 dBsm/m^2)
    Water,      // Sea / lake surface (clutter sigma0 ~ -25 dBsm/m^2)
    Pec         // Perfect Electric Conductor ground plane
};

struct SarConfig {
    // Platform & trajectory
    SarPlatform platform = SarPlatform::Spaceborne;
    SarMode mode = SarMode::Spotlight;
    double altitude_m = 500.0e3;              // 500 km (LEO satellite)
    double platform_velocity_mps = 7600.0;    // 7.6 km/s orbital speed
    double incidence_angle_deg = 40.0;        // Angle from vertical (depression = 90 - inc = 50 deg)
    double radar_azimuth_deg = 45.0;          // Azimuth of radar illumination relative to vehicle (0=nose, 90=starboard)
    double heading_deg = 0.0;                 // Flight track azimuth (deg, 0 = North along +y)
    double standoff_ground_range_m = 0.0;     // Computed from altitude and incidence if 0
    bool overlay_wireframe = false;           // Overlay CAD wireframe as in simulation IF stage (Image 3)

    // Radar sensor parameters
    double center_frequency_hz = 35.0e9;      // 35.0 GHz (Ka-band default)
    double bandwidth_hz = 1.5e9;              // 1.5 GHz bandwidth -> delta_R = 10 cm
    std::string polarization = "VV";
    size_t num_pulses = 128;                  // Number of synthetic aperture pulses
    size_t num_frequencies = 32;              // Frequency steps across bandwidth

    // Synthetic aperture angle
    double aperture_angle_deg = 2.5;          // Angular synthetic aperture sweep span

    // Scene & Ground interaction
    SarGroundType ground_type = SarGroundType::Tarmac;
    double ground_clutter_sigma0_db = -20.0;  // Average clutter backscatter (dBsm/m^2)
    bool enable_shadow = true;                // Cast crisp geometric radar shadow onto ground
    bool enable_ground_multipath = true;      // Target-ground dihedral corner reflection
    bool enable_speckle = true;               // Coherent Rayleigh speckle texture on ground
    double ground_plane_z_m = 0.0;            // Ground z level (auto-aligned to bbox_min.z if 0.0)

    // Output grid & rendering
    double ground_range_extent_m = 0.0;       // Auto-sized to vehicle bbox + shadow if 0.0
    double azimuth_extent_m = 0.0;            // Auto-sized to vehicle bbox if 0.0
    size_t grid_samples_range = 512;
    size_t grid_samples_azimuth = 512;
    double dynamic_range_db = 40.0;           // dB dynamic range for display
    std::string colormap = "gray";            // "gray" (authentic satellite IMINT look), "turbo", "jet", "viridis"
    std::string sidelobe_window = "taylor";   // "taylor" (realistic -30dB apodization), "kaiser", "uniform", "clean"
    double sidelobe_suppression_db = 30.0;    // Peak sidelobe suppression level in dB (e.g. 30 dB for satellite SAR)
    bool show_reticle = false;                // Show red target reticle marker on peak scatterer (default false)
    bool show_hud = true;                     // Overlay satellite orbit / HUD annotations
    size_t image_width = 1024;
    size_t image_height = 768;
};

struct SarImage {
    // Physical resolutions
    double slant_range_resolution_m = 0.0;    // c / (2 * B)
    double ground_range_resolution_m = 0.0;   // slant_res / sin(incidence)
    double azimuth_resolution_m = 0.0;        // lambda / (2 * delta_theta)

    // Scene dimensions
    double ground_range_min_m = 0.0;
    double ground_range_max_m = 0.0;
    double azimuth_min_m = 0.0;
    double azimuth_max_m = 0.0;

    // Coordinate vectors
    std::vector<double> ground_range_axis_m;
    std::vector<double> azimuth_axis_m;

    // Calibrated 2D dB intensity grid: [range_idx][az_idx] (peak normalized to 0 dB)
    std::vector<std::vector<double>> intensity_db;

    // Peak scatterer metadata
    double peak_ground_range_m = 0.0;
    double peak_azimuth_m = 0.0;
    double peak_rcs_dbsm = 0.0;
    double shadow_area_fraction = 0.0;

    // Simulation provenance
    SarConfig config;
    std::string vehicle_name;
    size_t mesh_triangles = 0;
    std::vector<std::pair<geom::Vec3d, geom::Vec3d>> wireframe_edges;
    std::vector<geom::Vec3d> scattering_centers;

    // Export & Rendering
    nlohmann::json to_json() const;
    void write_json(const std::string& path) const;

    rendering::ImageBuffer render_image(size_t width = 1024, size_t height = 768,
                                        const std::string& colormap_name = "gray") const;
    void write_image(const std::string& path, size_t width = 1024, size_t height = 768,
                     const std::string& colormap_name = "gray") const;
};

class SarProcessor {
public:
    // Generate standard satellite and airborne mission presets
    static SarConfig make_preset(const std::string& preset_name);

    // Compute complete 2D SAR image from a normalized 3D CAD mesh
    static SarImage compute(const NormalizedMesh& mesh,
                            const SarConfig& config,
                            const std::string& vehicle_name = "Target",
                            const ResolvedMaterials* materials = nullptr);

    // Compute SAR image from an existing multi-angle/multi-frequency Hdf5Database
    static SarImage compute_from_database(const Hdf5Database& db,
                                          const SarConfig& config,
                                          const NormalizedMesh* mesh = nullptr,
                                          const std::string& vehicle_name = "Target",
                                          const ResolvedMaterials* materials = nullptr);

    // Geometric ground shadow projection along radar look vector k_hat
    static std::vector<std::vector<bool>> compute_ground_shadow_mask(
        const NormalizedMesh& mesh,
        const geom::Vec3d& k_hat,
        double ground_z,
        double x_min, double x_max, size_t nx,
        double y_min, double y_max, size_t ny);
};

} // namespace strikecem::analysis

# ADR-0035: General Synthetic Aperture Radar (SAR) Simulation and Image Formation Engine

## Status
Accepted

## Context
While Inverse Synthetic Aperture Radar (ISAR) operates on an idealized stationary turntable where the target rotates in place (ADR-0032), operational Earth observation and battlefield surveillance radars operate in Synthetic Aperture Radar (SAR) geometry. In SAR, the radar sensor moves along a flight trajectory (e.g., Low Earth Orbit [LEO] satellite or high-altitude reconnaissance UAV/aircraft) and illuminates entities on the Earth's surface at an oblique incidence angle $\theta_{\text{inc}}$.

Users and defense engineers required the capability to simulate high-resolution SAR images of ground vehicles and parked airframes as observed from orbit or airborne reconnaissance platforms—specifically high-frequency Ka-band (35 GHz) and X-band (9.6 GHz) spaceborne satellites and UAVs.

Simulating realistic SAR images requires physics beyond free-space target scattering:
1. **Geometric Radar Shadowing**: Radar energy incident at angle $\theta_{\text{inc}}$ cannot illuminate ground regions occluded by the target, casting distinct radar dark shadows onto the ground plane along the radar line-of-sight.
2. **Ground-Target Multipath**: Dihedral corner reflector interactions between the vehicle's vertical surfaces and the conductive or dielectric ground plane produce strong double-bounce backscatter.
3. **Ground Clutter and Rayleigh Speckle**: Natural and man-made terrain (tarmac, soil, water) exhibits diffuse surface scattering with coherent interference creating characteristic multiplicative Rayleigh amplitude speckle.
4. **Platform Dynamics and Orbit Mechanics**: Synthetic aperture integration length, orbital velocity ($v \approx 7.6\,\text{km/s}$ for LEO), slant-to-ground range projection, and finite RF bandwidth dictate spatial resolution in cross-track and along-track dimensions.

## Decision

1. **Architecture & Platform Modeling (`include/strikecem/analysis/SarImaging.hpp`, `src/strikecem/analysis/SarImaging.cpp`)**:
   - `SarPlatform`: `Spaceborne` (satellites), `Airborne` (aircraft / drones), `Terrestrial`.
   - `SarMode`: `Spotlight` (high-resolution steered beam), `Stripmap` (continuous strip scanning), `Circular`.
   - `SarGroundType`: `None`, `Tarmac`, `Soil`, `Water`, `Pec`.
   - `SarProcessor::make_preset`: Built-in operational presets:
     - `satellite-ka`: 35.0 GHz Ka-band, 500 km LEO orbit, 7.6 km/s velocity, 1.5 GHz bandwidth ($\approx 10\,\text{cm}$ resolution), $\theta_{\text{inc}} = 40^\circ$, synthetic aperture span $\Delta\theta = 2.5^\circ$, tarmac clutter.
     - `satellite-x`: 9.6 GHz X-band, 500 km LEO orbit, 7.6 km/s velocity, 600 MHz bandwidth ($\approx 25\,\text{cm}$ resolution), $\theta_{\text{inc}} = 35^\circ$, $\Delta\theta = 3.6^\circ$, soil clutter.
     - `airborne-ka`: 35.0 GHz Ka-band, 10 km altitude, 200 m/s velocity, 1.2 GHz bandwidth, $\theta_{\text{inc}} = 45^\circ$, $\Delta\theta = 2.5^\circ$, tarmac clutter.
     - `airborne-x`: 9.6 GHz X-band, 12 km altitude, 250 m/s velocity, 500 MHz bandwidth, $\theta_{\text{inc}} = 50^\circ$, $\Delta\theta = 3.6^\circ$, tarmac clutter.

2. **Diffraction-Limited Resolution Formulas**:
   - Slant-range resolution:
     $$\Delta R_{\text{slant}} = \frac{c}{2 B}$$
   - Ground cross-track resolution (accounting for grazing angle):
     $$\Delta R_{\text{ground}} = \frac{\Delta R_{\text{slant}}}{\sin \theta_{\text{inc}}} = \frac{c}{2 B \sin \theta_{\text{inc}}}$$
   - Azimuth along-track resolution:
     $$\Delta \text{Az} = \frac{\lambda}{2 \Delta \theta} = \frac{c}{2 f_c \Delta \theta}$$

3. **Ground Shadow Ray-Tracing (`compute_ground_shadow_mask`)**:
   - Evaluates the 3D radar look vector $\mathbf{\hat{k}} = (\sin \theta_{\text{inc}}, 0, -\cos \theta_{\text{inc}})$.
   - Projects all illuminated vehicle triangle vertices along $\mathbf{\hat{k}}$ onto the ground plane $z = z_{\text{ground}}$:
     $$\mathbf{p}_{\text{proj}} = \mathbf{v} - \frac{v_z - z_{\text{ground}}}{k_z} \mathbf{\hat{k}}$$
   - Rasterizes the shadow convex hulls onto the 2D ground grid, creating sharp geometric radar shadow silhouettes where ground clutter backscatter is completely suppressed ($I_{\text{clutter}} = 0$).

4. **Synthetic Visualization Synthesis**:
   - The default preview path draws a seeded, heuristic clutter and speckle scene, projected target footprint, approximate shadow, and display-shaped facet and edge returns.
   - Its point-spread shape and ground-multipath enhancement are visualization effects; this mode does not form a calibrated radar measurement or coherently simulate target and clutter phase history.

5. **Coherent Phase-History Backprojection**:
   - The opt-in `phase_history_backprojection` formation samples mesh facets as scalar-RCS point scatterers, computes coherent wideband returns along a constant-velocity synthetic aperture, range-compresses them, and backprojects onto the image grid.
   - It is checked against a point-target focus fixture. The image is normalized to its peak and the facet scattering model is approximate; vector physical-optics phase history, calibration, clutter, multipath, and platform-motion errors remain future work.

6. **Diagnostic Visual Image Generation (`write_image`)**:
   - Exports high-resolution 8-bit PNG and 24-bit BMP images without external graphics libraries.
   - Colormaps: Standard Military IMINT Radar Grayscale (`gray`) with configurable radar gamma ($0.65$ default for dynamic contrast enhancement), plus scientific false-color maps (`turbo`, `jet`, `viridis`).
   - Diagnostic HUD Overlay: Center frequency, bandwidth, ground/slant resolution, platform orbit/flight parameters, look angle, metric cross-track and along-track tick marks, flight path / radar look direction arrows, and peak scatterer crosshair reticle.

7. **CLI Subcommand (`strikecem sar`)**:
   - Integrated `strikecem sar <cad.obj>` command with flags:
     `--preset <satellite-ka|satellite-x|airborne-ka|airborne-x>`, `--formation <visualization|backprojection>`, `--heading <deg>`, `--aperture <deg>`, `--pulses <N>`, `--frequencies <N>`, `--max-scatterers <N>`, `--output <image.png>`, `--json <data.json>`, `--freq <Hz>`, `--bandwidth <Hz>`, `--incidence <deg>`, `--altitude <m>`, `--velocity <m/s>`, `--ground <tarmac|soil|water|pec|none>`, `--no-shadow`, `--colormap <gray|turbo|jet|viridis>`, `--width <N>`, `--height <N>`.

8. **Verification (`tests/test_sar.cpp`)**:
   - Tests cover presets, resolution calculations, shadow projection, preview rendering, JSON serialization, coherent point-target focusing, and invalid work estimates.

## Consequences
- StrikeCEM provides both a fast synthetic preview and a coherent scalar-facet backprojection mode for satellite and airborne geometry experiments.
- Output metadata identifies the image-formation mode, calibration limits, trajectory model, and ground-scene model so preview imagery is not mistaken for measurement-calibrated data.

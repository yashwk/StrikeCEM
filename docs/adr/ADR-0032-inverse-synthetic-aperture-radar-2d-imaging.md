# ADR-0032: 2D Inverse Synthetic Aperture Radar (ISAR) Imaging Engine

## Status
Accepted

## Context
Standard Radar Cross Section (RCS) analysis provides scalar intensity metrics (in $\text{m}^2$ or $\text{dBsm}$) across aspect angles and frequencies. While scalar RCS reveals *how much* power is reflected, it provides no spatial localization of *where* scattering originates on a vehicle.

To perform radar signature diagnostics, hot-spot mitigation, and stealth design validation, radar engineers rely on Inverse Synthetic Aperture Radar (ISAR) imaging. By coherently combining wideband frequency sweeps (providing down-range resolution via pulse compression / inverse Fourier transform) with turntable azimuth angular sweeps (providing cross-range resolution via Doppler / aperture synthesis), a 2D spatial reflectivity image of the vehicle is reconstructed.

A dedicated 2D ISAR imaging capability was required within StrikeCEM to transform frequency-aspect sweep databases into diagnostic radar images without external post-processing tools.

## Decision
1. **Physical Radar Resolution & Aperture Synthesis (`IsarImaging.hpp` / `IsarImaging.cpp`)**:
   - For stepped-frequency sweeps with bandwidth $B = f_{\max} - f_{\min}$ and step $\Delta f$, and azimuth sweeps with span $\Delta \theta$ and step $\Delta \theta_{\text{step}}$:
     $$\Delta R = \frac{c}{2 B}, \quad R_{\max} = \frac{c}{2 \Delta f}$$
     $$\Delta CR = \frac{\lambda_c}{2 \Delta \theta}, \quad CR_{\max} = \frac{\lambda_c}{2 \Delta \theta_{\text{step}}}$$
   - Azimuth phase unwrapping handles circular angle transitions across $[0^\circ, 360^\circ)$ smoothly.

2. **2D Range-Doppler Reconstruction Algorithm**:
   - Applies separable 2D inverse Fourier transformation directly mapping complex scattering matrix $S(f_m, \theta_n)$ to spatial coordinates $(y_p, x_q)$:
     $$F_R(p, n) = \sum_{m=0}^{M-1} w_{\text{freq}}(m) S(f_m, \theta_n) \exp\left( +j \frac{4\pi}{c}(f_m - f_c) y_p \right)$$
     $$I(p, q) = \frac{1}{M N} \sum_{n=0}^{N-1} w_{\text{az}}(n) F_R(p, n) \exp\left( +j \frac{4\pi f_c}{c}(\theta_n - \theta_c) x_q \right)$$
   - Windowing: 2D Hann windowing suppresses transform sidelobes by $>31\,\text{dB}$.
   - Intensity: calculates normalized decibels $20 \log_{10}(|I| / \max |I|)$ with user-configurable dynamic range clipping (default: $35\,\text{dB}$).

3. **ASCII Heatmap Terminal Renderer (`to_ascii_heatmap`)**:
   - Renders a terminal-friendly 2D character matrix with vertical down-range axis ($+y$ forward, $-y$ aft) and horizontal cross-range axis ($-x$ port, $+x$ starboard).
   - 10-level intensity scale: ` .:-=+*#%@`.
   - Annotates center frequency, bandwidth, resolution, and peak scatterer coordinates.

4. **JSON Serialization (`to_json`, `write_json`)**:
   - Exports complete frequency/azimuth axes, physical resolutions, peak metadata, coordinate vectors, and 2D intensity grid to structured JSON.

5. **CLI Subcommand (`strikecem isar`)**:
   - Added `strikecem isar <database.h5>` to the main CLI with options:
     `--output <json>`, `--ascii`, `--pol <HH|VV>`, `--dyn-range <dB>`, `--range-samples <N>`, `--cross-range-samples <M>`, `--no-window`, `--elevation <deg>`.

6. **Test Verification (`tests/test_isar.cpp`)**:
   - Validated point scatterer resolution limits and peak localization within resolution cells.
   - Tested multi-scatterer separation and relative amplitude preservation.
   - Tested ASCII heatmap rendering and JSON file export.
   - Tested input validation guards and Hdf5Database ingestion.

## Consequences
- Engineers can immediately localize RCS hot-spots and diagnostic scatterers on CAD geometries from the command line.
- Native C++ implementation has zero external dependencies and executes in milliseconds.
- Adds 6 new unit tests to the suite with 100% pass rate.

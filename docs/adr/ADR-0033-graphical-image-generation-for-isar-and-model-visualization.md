# ADR-0033: Graphical Image Generation for ISAR and 3D Model Visualization

## Status
Accepted

## Context
While ASCII heatmaps and JSON coordinate dumps provide lightweight terminal inspection of radar diagnostics, stealth vehicle design and RCS hot-spot mitigation demand publication-quality, high-resolution 2D and 3D graphical imagery:
1. **2D ISAR Radar Images**: Radar engineers require high-resolution false-color heatmaps (using perceptual and scientific colormaps such as Google Turbo, Jet, and Viridis) with calibrated metric axes, dynamic range colorbars ($-D\,\text{dB}$ to $0\,\text{dB}$), down-range and cross-range physical coordinates, and peak scatterer crosshair reticles.
2. **3D Vehicle Geometry and Current Visualizations**: During simulation runs or post-run analysis, visualizing illuminated lit/shadow facets and surface current density $|\mathbf{J}_s|$ directly on the 3D target geometry enables immediate physical diagnosis of edge diffraction, specular glint, and shadow boundaries without requiring third-party visualization software (e.g. ParaView) for quick checks.
3. **Automated Headless Snapshot Export**: In automated CI/CD pipelines, parametric sweeps, and distributed HPC clusters, headless image generation directly to PNG/BMP formats with embedded diagnostic telemetry overlays (HUD) is essential.

## Decision
1. **Lightweight Rasterization & Native Image Export (`rendering/Image.hpp`, `rendering/Image.cpp`)**:
   - Implemented an RGBA `ImageBuffer` supporting primitive 2D drawing (lines, stroked/filled rectangles, circles, alpha blending, and embedded 8x8 monospace bitmap font rendering for HUD and axis annotations).
   - Implemented native uncompressed Windows BMP (`.bmp`) and Netpbm binary PPM (`.ppm`) encoders with zero external dependencies.
   - Implemented standard Deflate-compressed PNG (`.png`) export using system `zlib` (`IHDR`, `IDAT`, `IEND` with CRC32 verification).
   - Unified `write_image(path, buffer)` automatically detects the output format based on file extension.

2. **Perceptually Calibrated Scientific Colormaps (`rendering/Colormap.hpp`)**:
   - `Turbo`: Google Turbo colormap (radar gold standard, smooth perceptual progression avoiding false boundary artifacts).
   - `Jet`: Classic rainbow colormap with high visual contrast.
   - `Viridis`: Perceptually uniform colormap accessible to colorblind viewers.
   - `Inferno`: High-contrast black-purple-orange-yellow thermal colormap.
   - `Grayscale`: Standard linear intensity mapping.

3. **Software 3D Triangle Rasterizer (`rendering/ModelRenderer.hpp`, `rendering/ModelRenderer.cpp`)**:
   - Implemented a headless software 3D rasterizer with hardware-free Z-buffer depth sorting.
   - Three dedicated visualization shading modes:
     - `ShadedCad`: Directional diffuse lighting + specular sheen + ambient fill for vehicle geometry inspection.
     - `LitShadow`: Radar line-of-sight illumination diagnostic (amber gold for illuminated lit facets, charcoal for shadowed facets).
     - `SurfaceCurrents`: Physical Physical-Optics surface current density $|\mathbf{J}_s|$ dynamically mapped through colormaps with logarithmic or normalized scaling.
   - Radar HUD Telemetry Overlay:
     - Aspect angle $(\text{Az}, \text{El})$, frequency, polarization, illuminated facet percentage, triangle count, and bounding box dimensions.
     - 3D orientation axis triad ($X$ red, $Y$ green, $Z$ blue).
     - Colorbar key and numeric value scale.
   - Dual Camera Modes: `radar` (view looking down the radar propagation vector $\mathbf{\hat{k}}$) and `isometric` (perspective oblique view).

4. **ISAR Heatmap Image Rendering (`IsarImaging.hpp`, `IsarImaging.cpp`)**:
   - Added `IsarImage::render_image` and `IsarImage::write_image`.
   - Bilinear interpolation resampling 2D $(y, x)$ dB reflectivity grids into high-resolution output buffers.
   - Down-Range ($+y$ bow, $-y$ stern) and Cross-Range ($-x$ port, $+x$ starboard) metric ticks and gridlines.
   - Dynamic range colorbar with decibel markings ($-D\,\text{dB}$ to $0\,\text{dB}$).
   - Peak scatterer reticle crosshairs annotating $(x_{\text{peak}}, y_{\text{peak}})$ coordinates.

5. **Configuration Schema & CLI Integration**:
   - Extended `schemas/config.schema.json` with:
     - `output.render`: `{enabled, path, mode, colormap, width, height, sample_index}`
     - `output.diagnostics.image_export_path`, `image_mode`, `image_colormap`
   - main solver (`cmd_run`) automatically generates snapshot images when configured.
   -main CLI added `strikecem render <config|mesh>` for standalone 3D rendering.
   - main CLI extended `strikecem isar <database.h5>` with `--image <path>`, `--colormap <name>`, `--width`, `--height`.

6. **Test Verification (`tests/test_render.cpp`)**:
   - Unit tests covering image buffer drawing primitives, PNG/BMP/PPM headers and encoding, colormap evaluation, 2D ISAR heatmap generation, and 3D model rendering across all three shading modes.
   - 100% test pass rate across 346 tests.

## Consequences
- Fast, dependency-light graphical image generation is available natively in StrikeCEM without requiring external GUI or Python dependencies.
- Radar engineers can visually analyze hot-spots, shadow regions, and surface currents in continuous integration pipelines and terminal workflows.

# ADR-0034: Interactive Standalone HTML Signature Report Generator

## Status
Accepted

## Context
Radar signature engineers, program managers, and stealth evaluation teams require comprehensive diagnostic reports summarizing simulation campaigns. Previously, analyzing a StrikeCEM campaign required inspecting raw CSV tables or HDF5 datasets, running external plotting scripts, and viewing 2D/3D images separately.

To make StrikeCEM simulation results immediately accessible, shareable, and actionable without requiring external Python environments, web servers, or third-party visualization tools, a native, single-file, self-contained HTML signature reporting engine was required.

## Decision
1. **Report Engine Architecture (`analysis/SignatureReport.hpp`, `analysis/SignatureReport.cpp`)**:
   - `SignatureReportGenerator` ingests an `Hdf5Database` and optional `NormalizedMesh` / configuration JSON.
   - Computes aerospace aspect sector statistics:
     - Nose sector ($\pm 15^\circ$ azimuth): linear power-mean, max, median.
     - Beam sectors ($75^\circ-105^\circ$ starboard, $255^\circ-285^\circ$ port): mean, max.
     - Stern sector ($165^\circ-195^\circ$ aft): mean, max.
     - Overall campaign peak RCS with exact $(\text{Az}, \text{El}, f)$ coordinates and median/mean dBsm levels.
   - Generates interactive, dependency-free inline SVG charts:
     - **Polar RCS Cut**: $360^\circ$ radar scope diagram with concentric decibel grid rings ($-40\,\text{dBsm}$ to $+20\,\text{dBsm}$), radial spokes, and angular markers (Nose, Stbd, Stern, Port).
     - **Rectangular Azimuth RCS Cut**: Cartesian plot of RCS vs Azimuth angle with labeled ticks and gridlines.
     - **Frequency Sweep Response**: Multi-frequency response curve (Frequency GHz vs RCS dBsm).

2. **Self-Contained Embedded Visualizations**:
   - Auto-computes and renders 2D ISAR radar images via `IsarProcessor` and 3D vehicle simulation snapshots via `ModelRenderer`.
   - Embeds PNG images directly into the HTML payload as `data:image/png;base64,...` using a native zero-dependency Base64 encoder (`ImageBuffer::encode_png`).
   - The generated HTML file is 100% offline-capable and requires no internet connection, external CSS/JS libraries, or CDN requests.

3. **CLI Subcommand (`strikecem report`)**:
   - Added command:
     ```bash
     strikecem report <database.h5|config.json> [-o report.html] [--config CFG] [--mesh MESH]
                      [--title "..."] [--vehicle "..."] [--image img.png] [--isar isar.png]
                      [--mode currents|lit_shadow|shaded] [--colormap turbo|jet|viridis]
     ```

4. **Verification (`tests/test_report.cpp`)**:
   - Added unit test suite covering Base64 encoding round-trips, sector statistics and linear power averaging, polar and rectangular SVG generation, full HTML structure validation, and end-to-end file generation using real UCAV simulation data.
   - All 351 / 351 tests passing in CTest.

## Consequences
- Single-command generation of executive and engineering signature reports directly from the StrikeCEM CLI.
- Reports can be archived, emailed, or opened in any modern browser on any operating system without additional dependencies.

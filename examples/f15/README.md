# StrikeCEM 4th-Gen F-15 Eagle Benchmark & SAR Simulation

This directory contains a complete, self-contained demonstration and benchmark campaign for the McDonnell Douglas **F-15 Eagle** 4th-Generation twin-engine air superiority fighter (twin upright vertical stabilizers at $0^\circ$ cant, variable-geometry rectangular 2D air intakes, Pratt & Whitney F100 afterburning turbofans, cropped-delta wings, and all-moving horizontal stabilators).

It demonstrates the entire StrikeCEM simulation lifecycle:
1. Procedural 3D mesh generation with full bilateral symmetry and 2-manifold closed topology (`f15.obj`, `f15_lod1.obj`, `f15_lod2.obj`).
2. Automated 3D electromagnetic feature pre-scan and hybrid solver tuning (`strikecem prescan`).
3. Full monostatic Physical Optics (PO) and SBR + PTD fringe edge diffraction calculations (`f15_baseline_po.h5`, `f15_hybrid.h5`).
4. Electrodynamic similitude scaling (1:5 geometric scale model at 50 GHz with automated $+13.98\,\text{dBsm}$ output renormalization).
5. Production `.scempack` asset packaging with multi-LOD geometry embedding (`f15.scempack`).
6. 3D diagnostic visualizations (lit/shadow ray tracing, surface currents, shaded CAD).
7. Range-Doppler 2D Inverse Synthetic Aperture Radar (ISAR) imaging (`f15_isar_turbo.png`).
8. Operational spaceborne Ka-band and airborne X-band Synthetic Aperture Radar (SAR) 2D imaging with tarmac ground clutter, prism shadow volumes, continuous Taylor apodized sidelobe flares, and simulation IF-stage wireframes.
9. Interactive standalone HTML Signature Report generation (`f15_signature_report.html`).
10. Sub-microsecond Python SDK evaluation throughput benchmarking (>200,000 queries/s).

---

## 1. The Target: F-15 Eagle Airframe

The model (`f15.obj`) represents the iconic McDonnell Douglas F-15 Eagle:
- **Length**: 19.40 meters ($x \in [-9.6, 9.8]$, true F-15: $19.43\,\text{m}$)
- **Wingspan**: 13.04 meters ($y \in [-6.52, 6.52]$, true F-15: $13.05\,\text{m}$)
- **Height**: 4.60 meters ($z \in [-0.40, 4.20]$, bottom of intakes to fin tips)
- **Surface Area**: $284.88\,\text{m}^2$
- **Electromagnetic Features**:
  - Twin upright vertical stabilizers mounted at $y = \pm 1.45\,\text{m}$ with **$0^\circ$ cant** (parallel vertical fins).
  - Variable-geometry rectangular 2D air intakes exposing internal compressor face cavities.
  - Rotating compressor bullet hubs and stator cascades.
  - $45^\circ$ leading-edge swept cropped-delta wings with wingtip missile rails.
  - Outboard tail booms with all-moving horizontal stabilators (tailerons) featuring aerodynamic dogtooth leading edges.
  - Twin Pratt & Whitney F100 afterburning turbofan nozzles with internal flameholders.
  - High bubble canopy and ogival nose radome.

---

## 2. Electromagnetic RCS Physics: Raw PO vs Operational Measurements

A key question in radar simulation is why theoretical flat-faceted computational electromagnetics (PO) calculates very large peak values compared to window-averaged operational flight test literature:

### A. The Classical Flat-Plate Specular Formula
Under high-frequency Physical Optics, every flat facet of area $A$ normal to radar wavelength $\lambda$ produces an instantaneous specular return:
$$\sigma_{\text{specular}} = \frac{4\pi A^2}{\lambda^2}$$

At $10\,\text{GHz}$ (X-band, $\lambda = 0.03\,\text{m}$):
$$\frac{4\pi}{\lambda^2} = \frac{4\pi}{0.0009} \approx 13,963\,\text{m}^{-2}$$

- For the F-15's large twin upright vertical stabilizers ($\approx 12\,\text{m}^2$ total vertical area) and flat slab fuselage sides ($\approx 18\,\text{m}^2$), if treated as a specular flat plate, PO calculates a theoretical specular peak of:
  $$\sigma_{\text{broadside}} \approx 478,183\,\text{m}^2 \approx +56.8\,\text{dBsm}$$

### B. Why Operational Radar Literature Values are $\approx 100 - 400\,\text{m}^2$ Broadside and $\approx 10 - 25\,\text{m}^2$ Frontal
Real operational radar measurements reported in literature (Knott, Shaeffer, Tuley; Skolnik; IEEE) quote:
- **Frontal Nose-On ($0^\circ \pm 10^\circ$)**: $\approx 10 - 25\,\text{m}^2$ ($+10$ to $+14\,\text{dBsm}$)
- **Broadside ($90^\circ / 270^\circ$)**: $\approx 100 - 400\,\text{m}^2$ ($+20$ to $+26\,\text{dBsm}$)
- **Aft ($180^\circ$)**: $\approx 15 - 50\,\text{m}^2$ ($+12$ to $+17\,\text{dBsm}$)

The physical reasons for this difference include:
1. **Airfoil Camber & Radius of Curvature**: Operational aircraft have curved NACA airfoils (e.g., NACA 64A-006). For a curved cylinder or airfoil with radius of curvature $R$, the specular return is limited to the narrow Fresnel zone ($\sigma = \pi R L$), which reduces peak returns by orders of magnitude compared to $A^2/\lambda^2$.
2. **Compressor Blade Dispersion**: A real jet engine compressor face consists of 36–48 spinning, twisted blades and conical center hubs that scatter radar energy obliquely across multiple cavity modes, rather than reflecting like a flat perpendicular mirror.
3. **Flight Jitter & Sector Window Averaging**: Literature values represent sliding-window medians (typically $5^\circ$ to $15^\circ$ azimuth windows) with dynamic flight micro-yawing ($\pm 0.2^\circ$), which smooths out sharp specular delta-peaks.

---

## 3. Reproduction Steps

### Step 1: Generate Clean 3D Mesh Geometry
```bash
python3 generate_f15.py
```
Generates 2-manifold models with 0 degenerate triangles and 0 non-manifold edges:
- `f15.obj` (fine benchmark resolution: 396 vertices, 752 triangles)
- `f15_lod1.obj` (medium LOD)
- `f15_lod2.obj` (coarse LOD: 100 vertices, 188 triangles)

### Step 2: Pre-Scan Vehicle Geometry
```bash
./build/release/strikecem prescan examples/f15/f15_baseline_po.json --report examples/f15/f15_prescan.json
```

### Step 3: Run Baseline PO Simulation
```bash
./build/release/strikecem --bench-profile examples/f15/bench_po.json run examples/f15/f15_baseline_po.json
```

### Step 4: Run Auto-Tuned Hybrid SBR+PTD Simulation
```bash
./build/release/strikecem --bench-profile examples/f15/bench_hybrid.json run examples/f15/f15_hybrid.json
```

### Step 5: Run 1:5 Similitude Scale Model
```bash
./build/release/strikecem --bench-profile examples/f15/bench_similitude.json run examples/f15/f15_similitude.json
```

### Step 6: Package into Production `.scempack` Bundle
```bash
./build/release/strikecem bundle --mesh examples/f15/f15.obj --rcs examples/f15/f15_hybrid.h5 -o examples/f15/f15.scempack --lod lod1=examples/f15/f15_lod1.obj --lod lod2=examples/f15/f15_lod2.obj
```

### Step 7: Generate 3D Visualizations & ISAR Imaging
```bash
./build/release/strikecem render examples/f15/f15_hybrid.json --mode currents -o examples/f15/images/f15_surface_currents.png
./build/release/strikecem render examples/f15/f15_hybrid.json --mode lit_shadow -o examples/f15/images/f15_lit_shadow.png
./build/release/strikecem render examples/f15/f15_hybrid.json --mode shaded -o examples/f15/images/f15_shaded_cad.png
./build/release/strikecem isar examples/f15/f15_isar.h5 --image examples/f15/images/f15_isar_turbo.png --colormap turbo
```

### Step 8: Generate High-Fidelity SAR Imagery
```bash
# Spaceborne Ka-band on tarmac with continuous horizontal flares
./build/release/strikecem sar examples/f15/f15_baseline_po.json --preset satellite-ka -o examples/f15/images/f15_sar_satellite_ka.png --json examples/f15/images/f15_sar_satellite_ka.json

# Spaceborne Ka-band simulation IF-stage CAD wireframe with '+' markers
./build/release/strikecem sar examples/f15/f15_baseline_po.json --preset satellite-ka --wireframe -o examples/f15/images/f15_sar_satellite_ka_wireframe.png

# Airborne X-band
./build/release/strikecem sar examples/f15/f15_baseline_po.json --preset airborne-x -o examples/f15/images/f15_sar_airborne_x.png
```

### Step 9: Generate Interactive Signature Report
```bash
./build/release/strikecem report examples/f15/f15_hybrid.h5 -o examples/f15/f15_signature_report.html --config examples/f15/f15_hybrid.json --title "F-15 Eagle Radar Signature Assessment" --vehicle "F-15 Eagle" --isar examples/f15/images/f15_isar_turbo.png --image examples/f15/images/f15_surface_currents.png
```

### Step 10: Run Verification & Sub-Microsecond Benchmark
```bash
python3 examples/f15/verify_f15.py
```
Demonstrates >200,000 queries per second across NEAREST, LINEAR, BICUBIC, and WENDLAND SPHERICAL interpolation modes.

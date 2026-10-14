# StrikeCEM 5th-Gen Stealth Fighter Benchmark & SAR Simulation

This directory contains a complete, self-contained demonstration and benchmark campaign for a 5th-Generation Twin-Engine Stealth Fighter airframe (twin canted vertical stabilizers, chined fuselage, dual re-entrant caret engine air intakes, 2D thrust-vectoring nozzle cavities, and trapezoidal wings — matching Reference Image 1).

It demonstrates the entire StrikeCEM simulation lifecycle:
1. Procedural 3D mesh generation with full bilateral symmetry and 2-manifold closed topology (`fighter.obj`, `fighter_lod1.obj`, `fighter_lod2.obj`).
2. Automated 3D electromagnetic feature pre-scan and hybrid solver tuning (`strikecem prescan`).
3. Full monostatic Physical Optics (PO) and SBR + PTD fringe edge diffraction calculations.
4. Electrodynamic similitude scaling (1:5 geometric scale model at 50 GHz with automated $+13.98\,\text{dBsm}$ output renormalization).
5. Production `.scempack` asset packaging with multi-LOD geometry embedding.
6. 3D diagnostic visualizations (lit/shadow ray tracing, surface currents, shaded CAD).
7. Range-Doppler 2D Inverse Synthetic Aperture Radar (ISAR) imaging.
8. Operational spaceborne Ka-band and airborne X-band Synthetic Aperture Radar (SAR) 2D imaging with tarmac ground clutter, prism shadow volumes, and continuous Taylor apodized sidelobe flares.
9. Interactive standalone HTML Signature Report generation (`fighter_signature_report.html`).
10. Sub-microsecond Python SDK evaluation throughput benchmarking (~267,000 queries/s).

---

## 1. The Target: 5th-Gen Stealth Fighter Airframe

The model (`fighter.obj`) represents an advanced twin-engine air superiority stealth fighter:
- **Length**: 18.8 meters ($x \in [-9.8, 9.0]$)
- **Wingspan**: 13.6 meters ($y \in [-6.8, 6.8]$)
- **Height**: 3.85 meters ($z \in [-0.25, 3.60]$)
- **Surface Area**: $293.16\,\text{m}^2$
- **Stealth & EM Features**:
  - $42^\circ$ planform leading-edge alignment across radome chine, strake (LERX), main wings, and all-moving horizontal tailerons.
  - Bubble cockpit canopy with smooth dorsal blending into central spine.
  - Twin canted vertical stabilizers ($28^\circ$ outward cant) to redirect broadside radar energy away from threat radars.
  - Twin re-entrant caret engine air intakes with deep internal duct cavities for SBR multi-bounce radar trapping.
  - Twin 2D thrust-vectoring exhaust nozzles with internal turbine cavity walls.
  - Planar faceted belly contours for internal weapons bay doors.
  - Perfect bilateral symmetry across the $xz$-plane ($y = 0$).

---

## 2. Reproduction Steps

### Step 1: Generate Clean 3D Mesh Geometry

```bash
python3 generate_fighter.py
```

Outputs watertight 2-manifold models with 0 degenerate triangles and 0 non-manifold edges:
- `fighter.obj` (fine benchmark resolution: 348 vertices, 656 triangles)
- `fighter_lod1.obj` (medium LOD)
- `fighter_lod2.obj` (coarse LOD: 86 vertices, 164 triangles)

### Step 2: Pre-Scan Vehicle Geometry

```bash
strikecem prescan fighter_baseline_po.json --report fighter_prescan.json
```

**Pre-Scan Findings**:
- Detects re-entrant intake duct and nozzle cavities ($984.4\,\text{m}^3$ estimated volume).
- Automatically recommends **SBR** solver type, enables **Shadowing**, and sets **Multi-Bounce Limit to 3**.

### Step 3: Run Baseline PO Simulation

Simulate full $360^\circ$ azimuth sweep at 10 GHz (X-band) using naive Physical Optics:

```bash
strikecem run fighter_baseline_po.json --bench-profile bench_po.json
```

### Step 4: Run Auto-Tuned Hybrid SBR+PTD Simulation

Simulate the same sweep using the auto-tuned hybrid configuration:

```bash
strikecem run fighter_hybrid.json --bench-profile bench_hybrid.json
```

### Step 5: Run 1:5 Similitude Scale Model

Simulate the target as a 1:5 scale model ($s_g = 0.2$) solving at 50 GHz with automated $+13.98\,\text{dBsm}$ output renormalization:

```bash
strikecem run fighter_similitude.json --bench-profile bench_similitude.json
```

### Step 6: Package into Production `.scempack` Bundle

```bash
strikecem bundle --rcs fighter_hybrid.h5 --mesh fighter.obj --lod lod1=fighter_lod1.obj --lod lod2=fighter_lod2.obj -o fighter.scempack
```

### Step 7: Generate 3D Visualizations & ISAR Imaging

```bash
# 1. Surface currents
strikecem render fighter_hybrid.json --mode currents -o images/fighter_surface_currents.png

# 2. Lit-shadow ray tracing
strikecem render fighter_hybrid.json --mode lit_shadow -o images/fighter_lit_shadow.png

# 3. Shaded CAD
strikecem render fighter_hybrid.json --mode shaded -o images/fighter_shaded_cad.png

# 4. Range-Doppler ISAR radar heatmap
strikecem run fighter_isar.json
strikecem isar fighter_isar.h5 --image images/fighter_isar_turbo.png --colormap turbo
```

### Step 8: Generate Spaceborne Ka-band & Airborne X-band SAR Imagery

```bash
# Operational Satellite Ka-band SAR (35 GHz, 0.25 m resolution, tarmac ground, shadow volume, Taylor apodized flares)
strikecem sar fighter_baseline_po.json --preset satellite-ka -o images/fighter_sar_satellite_ka.png

# IF-stage simulation CAD wireframe with '+' cross scatterer markers
strikecem sar fighter_baseline_po.json --preset satellite-ka --wireframe -o images/fighter_sar_satellite_ka_wireframe.png

# Airborne Tactical X-band SAR (10 GHz, 0.50 m resolution, soil ground)
strikecem sar fighter_baseline_po.json --preset airborne-x -o images/fighter_sar_airborne_x.png
```

### Step 9: Generate Standalone HTML Signature Report

```bash
strikecem report fighter_hybrid.h5 -o fighter_signature_report.html \
  --config fighter_hybrid.json \
  --title "5th-Gen Stealth Fighter Radar Signature Assessment" \
  --vehicle "Fighter-5G" \
  --isar images/fighter_isar_turbo.png \
  --image images/fighter_surface_currents.png
```

### Step 10: Run Runtime Verification Benchmark

```bash
python3 verify_fighter.py
```

---

## 3. Visual Gallery

### Spaceborne Ka-Band SAR (35 GHz Satellite Reconnaissance)
Operational satellite reconnaissance view of the 5th-Gen Stealth Fighter parked on concrete tarmac with calibrated Rayleigh/K-distribution speckle, full 3D shadow projection, and continuous horizontal sidelobe flares:

![Fighter Satellite Ka-Band SAR](images/fighter_sar_satellite_ka.png)

### Simulation IF-Stage Wireframe Overlay
Radar simulation intermediate frequency (IF) stage visualization showing the CAD wireframe in orthographic slant-range radar coordinates with yellow `+` cross markers at dominant scattering centers:

![Fighter SAR Wireframe Overlay](images/fighter_sar_satellite_ka_wireframe.png)

### Airborne Tactical X-Band SAR (10 GHz)
Tactical airborne standoff synthetic aperture radar snapshot:

![Fighter Airborne X-Band SAR](images/fighter_sar_airborne_x.png)

### 2D Range-Doppler ISAR Heatmap
Range-Doppler radar imaging via 2D FFT with Taylor windowing and Turbo colormap:

![Fighter ISAR Heatmap](images/fighter_isar_turbo.png)

### 3D Diagnostic Renderings
- **Surface Currents**: Induced surface current magnitude $|\vec{J}_s|$ under 10 GHz illumination.
- **Lit/Shadow Diagnostics**: Ray-traced geometric optical self-shadowing.
- **Shaded CAD**: Orthographic geometric representation.

| Surface Currents | Lit / Shadow | Shaded CAD |
| :---: | :---: | :---: |
| ![Surface Currents](images/fighter_surface_currents.png) | ![Lit Shadow](images/fighter_lit_shadow.png) | ![Shaded CAD](images/fighter_shaded_cad.png) |

---

## 4. Performance & Verification Metrics

```text
===========================================================================
StrikeCEM 5th-Gen Stealth Fighter Runtime Verification
===========================================================================
Loaded Database:    fighter_hybrid.h5
Frequencies (Hz):   [10000000000.0]
Polarizations:      ['HH', 'VV']
---------------------------------------------------------------------------
Aspect Name            | Azimuth  | Mode            | RCS (dBsm) | RCS (m^2) 
---------------------------------------------------------------------------
Nose-On Sector         |    0.0°  | Bicubic C^1     |    41.08   | 12,831.85
Starboard Chine        |   42.0°  | Bicubic C^1     |    51.58   | 143,801.83
Right Broadside        |   90.0°  | Bicubic C^1     |    49.95   | 98,875.47
Aft Oblique Taileron   |  138.0°  | Bicubic C^1     |    57.01   | 502,627.78
Aft 2D Nozzles         |  180.0°  | Bicubic C^1     |    54.49   | 281,352.12
Port Broadside         |  270.0°  | Bicubic C^1     |    49.95   | 98,838.30
---------------------------------------------------------------------------
Query Latency:         3.74 microseconds / query
Throughput:            267,561 queries / second
Status:                Verified & Operational
===========================================================================
```

# Tutorial: Automated Vehicle Feature Pre-Scan & Hybrid Solver Dispatch

This tutorial guides you through using StrikeCEM's automated vehicle feature pre-scan engine ([ADR-0030](file:///home/yashwanth/projects/StrikeCEM/docs/adr/ADR-0030-automated-vehicle-prescan-and-hybrid-solver-dispatch.md)). The pre-scan engine inspects 3D geometry, classifies surfaces into physical scattering mechanisms, and auto-tunes hybrid solver parameters for optimal accuracy along the Pareto frontier.

---

## 1. Why Pre-Scan Geometry?

Real-world aerospace, naval, and ground vehicles exhibit heterogeneous electromagnetic scattering mechanisms:
- **Broad planar plates** (wings, flat fuselage bulkheads) generate sharp specular flashes accurately captured by Physical Optics (PO).
- **Curved surfaces** (radomes, nacelles, leading-edge fillets) generate creeping waves and phase slip captured by Fock curvature corrections.
- **Sharp edges & wedges** (wing leading/trailing edges, control surface gaps) cause diffraction fringes modeled by Physical Theory of Diffraction (PTD) or Michaeli Equivalent Edge Currents (EEC).
- **Concave cavities & re-entrant ducts** (jet engine intakes, cockpit cavities, corner reflectors) trap radar energy in multiple reflections requiring Shooting and Bouncing Rays (SBR).

Manually configuring these solver switches across complex meshes is error-prone. The pre-scan engine eliminates guesswork by inspecting the mesh topology and auto-tuning the configuration.

---

## 2. Running the Pre-Scan

You can run `strikecem prescan` directly on a configuration JSON or on any standalone mesh (`.stl` or `.obj`):

```bash
strikecem prescan models/fighter_uav.obj
```

Or on a simulation config:

```bash
strikecem prescan configs/uav_xband.json
```

### Sample Terminal Output

```text
=== VEHICLE PRE-SCAN FEATURE ANALYSIS REPORT ===
Total Facets: 18450
Surface Area: 34.8210 m^2
Bounding Box: 8.4200 x 6.1500 x 1.8500 m
Estimated Cavity Volume: 0.4280 m^3

Feature Breakdown:
  - Planar Specular Surfaces: 11420 (61.8970%)
  - Smooth Curved Surfaces:   3890 (21.0840%)
  - Sharp Leading Edges/Wedges: 1820 (9.8645%)
  - Cavity / Duct Interiors:   1320 (7.1545%)

--- RECOMMENDED HYBRID SOLVER TUNING ---
  - Solver Type:             SBR (Shooting & Bouncing Rays)
  - Edge Correction Mode:    ptd_full
  - Shadowing Enabled:       true
  - Multi-Bounce Limit:      3
  - Fock Curvature Factor:   true
status: ok
```

---

## 3. Exporting Analysis Reports

To save the analysis metrics as machine-readable JSON for CI/CD pipelines or engineering archives, pass the `--report` option:

```bash
strikecem prescan models/fighter_uav.obj --report reports/uav_prescan.json
```

The exported JSON contains facet counts, surface percentages, bounding box dimensions, cavity volume, and recommended solver tuning:

```json
{
  "summary": {
    "total_facets": 18450,
    "surface_area_m2": 34.821,
    "planar_facets": 11420,
    "curved_facets": 3890,
    "edge_facets": 1820,
    "cavity_facets": 1320,
    "estimated_cavity_volume_m3": 0.428,
    "bounding_box_m": {
      "length_x": 8.42,
      "width_y": 6.15,
      "height_z": 1.85
    }
  },
  "feature_breakdown": {
    "planar_percent": 61.897,
    "curved_percent": 21.084,
    "edge_percent": 9.8645,
    "cavity_percent": 7.1545
  },
  "recommendations": {
    "recommended_solver": "SBR",
    "recommended_edge_correction": "ptd_full",
    "recommended_shadowing": true,
    "recommended_max_bounces": 3,
    "recommended_curvature_correction": true
  }
}
```

---

## 4. Automated Configuration Auto-Tuning

To automatically inject recommended solver settings into a base configuration file:

```bash
strikecem prescan configs/base_template.json --auto-tune-config configs/tuned_run.json
```

The output file `configs/tuned_run.json` will inherit all geometry, frequency, angle, and execution parameters from `base_template.json`, while updating the `"solver"` block:

```json
{
  "solver": {
    "type": "SBR",
    "edge_correction": "ptd_full",
    "shadowing": true,
    "po_options": {
      "edge_correction": "ptd_full",
      "shadowing": true,
      "max_bounces": 3,
      "curvature_correction": true
    },
    "sbr": {
      "max_bounces": 3
    }
  }
}
```

You can now run the optimized simulation immediately:

```bash
strikecem run configs/tuned_run.json
```

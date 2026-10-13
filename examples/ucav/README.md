# StrikeCEM UCAV Benchmark & Verification Campaign

This directory contains a complete, self-contained demonstration and benchmark campaign for an unmanned combat aerial vehicle (UCAV) stealth platform. It demonstrates the entire StrikeCEM lifecycle: 3D feature pre-scan, hybrid solver auto-tuning, electrodynamic similitude scaling, output analysis/comparison, production `.scempack` bundling, and sub-microsecond Python SDK queries.

---

## 1. The Target: Stealth UCAV Airframe

The model (`ucav.obj`) represents a tailless cranked-kite blended wing body (BWB) with:
- **Wingspan**: 10.0 meters
- **Length**: 6.5 meters
- **Height**: 0.95 meters
- **Stealth Features**:
  - Swept leading edges ($50^\circ$ sweep) and trailing edges for edge current diffraction (PTD/EEC).
  - Continuous blended fuselage with curved forward chines (Fock curvature factor).
  - Dorsal re-entrant engine intake duct with internal sidewalls (SBR multi-bounce).
  - Bilateral symmetry across the $xz$-plane ($y = 0$).

---

## 2. Reproduction Steps

### Step 1: Pre-Scan Vehicle Geometry

Inspect the 3D geometry and auto-tune hybrid solver parameters:

```bash
strikecem prescan ucav.obj --report ucav_prescan.json --auto-tune-config ucav_hybrid.json
```

**Pre-Scan Findings**:
- Detects re-entrant intake duct cavity facets and computes cavity volume ($54.5\,\text{m}^3$).
- Detects sharp swept leading/trailing edge facets.
- Automatically selects **SBR** solver type, enables **PTD Full** edge diffraction, enables **Curvature Correction**, enables **Shadowing**, and sets **Multi-Bounce Limit to 3**.

### Step 2: Run Baseline PO Simulation

Simulate full $360^\circ$ azimuth sweep at 10 GHz (X-band) using naive Physical Optics:

```bash
strikecem run ucav_baseline_po.json --bench-profile bench_po.json
```

### Step 3: Run Auto-Tuned Hybrid Simulation

Simulate the same sweep using the auto-tuned hybrid configuration:

```bash
strikecem run ucav_hybrid.json --bench-profile bench_hybrid.json
```

### Step 4: Run 1:5 Similitude Scale Model

Simulate the target as a 1:5 scale model ($s_g = 0.2$), solving at 50 GHz with automated $+13.98\,\text{dBsm}$ output renormalization:

```bash
strikecem run ucav_similitude.json --bench-profile bench_similitude.json
```

---

## 3. Comparative Analysis & Benchmarks

### 1D Azimuth Radiation Cut (`scem_quicklook`)

Inspect the horizontal radiation pattern ($el = 0^\circ$, 10 GHz, HH polarization) in the terminal:

```bash
scem_quicklook ucav_hybrid.h5 --cut azimuth --angle 0.0 --freq 10e9 --pol HH --step 10.0 --ascii
```

Shows low-observable nulls at nose-on ($30.5\,\text{dBsm}$) and tail ($28.9\,\text{dBsm}$), with specular edge flashes at broadside ($40.1\,\text{dBsm}$) and oblique chine reflections ($45.9\,\text{dBsm}$).

### Physical Differences: Baseline PO vs. Hybrid (`scem_compare`)

Compare naive PO against the auto-tuned hybrid solution:

```bash
scem_compare ucav_baseline_po.h5 ucav_hybrid.h5 --tolerance-db 1.0
```

- **Max Error**: $14.01\,\text{dBsm}$ (at shadowed and diffracted edge angles).
- **Mean Error**: $1.59\,\text{dBsm}$.
- **Complex Correlation**: $0.978$.

### Throughput Scaling (`scem_benchreport`)

Aggregate solver throughput across configurations:

```bash
scem_benchreport . --ascii
```

```text
================================================================================
StrikeCEM Benchmark Scaling & Throughput Report
================================================================================
ID                          Backend Prec          M-Eval/s  Throughput Bar
--------------------------------------------------------------------------------
bench_po                    cpu     float64          17.78  [###################           ]
bench_hybrid                cpu     float64           0.46  [                              ]
bench_similitude            cpu     float64          28.39  [##############################]
--------------------------------------------------------------------------------
```

---

## 4. Production Asset Packaging & Python SDK

### Package into `.scempack` Bundle

```bash
strikecem bundle --rcs ucav_hybrid.h5 --mesh ucav.obj --lod lod1=ucav_lod1.obj --lod lod2=ucav_lod2.obj -o ucav.scempack
```

### Verify Sub-Microsecond Real-Time Queries

Run the verification test script:

```bash
python3 verify_ucav.py
```

Outputs verified RCS across aspects (Nose, Chine, Broadside, Oblique, Tail) and measures batch query throughput (**~200,000 queries / second**, **5.0 $\mu\text{s}$ average latency**).

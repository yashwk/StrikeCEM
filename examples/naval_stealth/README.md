# StrikeCEM Naval Stealth Warship Benchmark & SAR Simulation

This directory contains a complete, self-contained demonstration and benchmark campaign for a modern naval stealth surface combatant—the **USS Zumwalt (DDG-1000) Guided-Missile Destroyer**—using **StrikeCEM**.

We explore how radical maritime stealth architecture (wave-piercing tumblehome hull, enclosed composite deckhouse, faceted AGS turrets, and reverse transom) combined with Radar Absorbent Material (RAM) coatings suppresses ship radar cross sections (RCS) against sea-skimming anti-ship missiles, maritime patrol aircraft, and spaceborne reconnaissance satellites.

---

## 1. Problem Overview: Maritime Radar Cross Section & Multipath

Conventional warships (e.g., Arleigh Burke DDG-51, Ticonderoga CG-47, or Sovremenny class) possess gargantuan radar cross sections:

- **Outward Hull Flare**: Standard hulls flare outward toward the main deck to shed waves. When radar waves hit the flared hull, they reflect downward onto the flat sea surface, which reflects them straight back to the radar—forming a massive **$90^\circ$ dihedral corner reflector**.
- **Cluttered Superstructures**: Exposed lattice masts, rotating parabolic antennas, open missile launchers, rigid-hull inflatable boats (RHIBs), life rafts, and deck winches act as hundreds of corner and cavity retro-reflectors.
- **Typical Conventional Warship RCS**: **$50 - 70\,\text{dBsm}$** ($100,000$ to $10,000,000\,\text{m}^2$), with beam specular spikes exceeding **$95 - 100\,\text{dBsm}$**. A conventional destroyer can be tracked by maritime search radars from over the radar horizon.

### The Zumwalt (DDG-1000) Solution
1. **Inverted Wave-Piercing Tumblehome Hull**: The hull sides slope *inward* by $>8^\circ$ from the waterline to the main deck. Instead of forming a dihedral corner with the ocean, horizontal radar waves bounce *upward* into space ($+19^\circ$ above the horizon).
2. **Inverted Wave-Piercing Stem**: The bow waterline knife edge extends furthest forward ($x = +95\,\text{m}$), while the forecastle terminates back at $x = +75\,\text{m}$.
3. **Integrated Composite Deckhouse & Mast**: All SPY-3 X-band / SPY-4 S-band planar AESA radar antennas, communications arrays, and engine exhaust stacks are flush-mounted inside a single sloped composite pyramid.
4. **Advanced Gun System (AGS) Stealth Turrets**: Low-profile faceted housings where 155mm gun barrels are recessed flush into the sloped turret roof when stowed.

---

## 2. 3D Airframe & Multi-Zone Semantic Topology

The warship model ([`zumwalt_stealth.obj`](file:///home/yashwanth/projects/StrikeCEM/examples/naval_stealth/zumwalt_stealth.obj)) was synthesized with clean **2-manifold closed geometry**:

- **True 1:1 Scale**: Length $190.0\,\text{m}$, Beam $24.6\,\text{m}$, Height $32.0\,\text{m}$ from waterline.
- **Topology Quality**: 128 vertices, 170 triangular facets, 0 degenerate triangles, 0 non-manifold edges.
- **Semantic Material Partitioning**:
  - `ram_hull` (46 facets): Inward tumblehome topside plating, wave-piercing bow stem, and reverse transom.
  - `ram_deckhouse` (16 facets): Sloped composite superstructure pyramid ($16.5\,\text{m}$ tall).
  - `ram_mast` (10 facets): Hexagonal faceted AESA radar aperture tower ($8.0\,\text{m}$ tall).
  - `ram_turret` (20 facets): AGS Mount 51 and Mount 52 low-RCS gun enclosures.
  - `pec_deck` (36 facets): Weather deck, perimeter walkways, and aft helicopter flight deck.
  - `pec_keel` (42 facets): Flat waterline closure plate at $z = 0.0\,\text{m}$.

---

## 3. Quantitative RCS Benchmark Results

Executing `python3 examples/naval_stealth/verify_naval.py` compares the solved HDF5 databases across full $360^\circ$ azimuth sweeps at $10.0\,\text{GHz}$ (X-Band Surface Search radar, HH polarization, $0.0^\circ$ elevation horizon look):

| Tactical Aspect Sector | Azimuth ($\phi$) | Bare PEC Baseline | Stealth RAM Suite | Delta ($\Delta \text{dB}$) | Power Absorbed |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Bow-On (Inverted Wave-Piercing Stem)** | $0.0^\circ$ | $68.14\,\text{dBsm}$ | **$59.41\,\text{dBsm}$** | **$-8.73\,\text{dB}$** | **$-86.6\%$** |
| **Starboard Bow Oblique** | $30.0^\circ$ | $84.09\,\text{dBsm}$ | **$71.61\,\text{dBsm}$** | **$-12.48\,\text{dB}$** | **$-94.3\%$** |
| **Forecastle Turrets Oblique** | $45.0^\circ$ | $68.24\,\text{dBsm}$ | **$59.38\,\text{dBsm}$** | **$-8.86\,\text{dB}$** | **$-87.0\%$** |
| **Starboard Beam (Tumblehome Hull)** | $90.0^\circ$ | $97.03\,\text{dBsm}$ | **$57.63\,\text{dBsm}$** | **$-39.41\,\text{dB}$** | **$-99.99\%$** |
| **Starboard Quarter (Hangar Transition)** | $135.0^\circ$ | $90.75\,\text{dBsm}$ | **$75.04\,\text{dBsm}$** | **$-15.71\,\text{dB}$** | **$-97.3\%$** |
| **Direct Stern (Reverse Slanted Transom)**| $180.0^\circ$ | $80.82\,\text{dBsm}$ | **$72.29\,\text{dBsm}$** | **$-8.53\,\text{dB}$** | **$-86.0\%$** |
| **Port Quartering Aspect** | $225.0^\circ$ | $90.75\,\text{dBsm}$ | **$75.04\,\text{dBsm}$** | **$-15.71\,\text{dB}$** | **$-97.3\%$** |
| **Port Beam (Tumblehome Hull)** | $270.0^\circ$ | $97.03\,\text{dBsm}$ | **$57.63\,\text{dBsm}$** | **$-39.41\,\text{dB}$** | **$-99.99\%$** |

### Key Takeaways
1. **Beam Flash Elimination**: On an untreated steel ship, broadside beam specular flash is catastrophic ($97.03\,\text{dBsm}$ or $\sim 5,000,000\,\text{m}^2$). The combination of the $9.5^\circ$ tumblehome slope and resonant magnetic RAM drops this spike by **$-39.41\,\text{dB}$ down to $57.63\,\text{dBsm}$**—an absorption of **$99.99\%$** of incident radar power.
2. **Bow-On Inversion**: The knifelike inverted stem and backward raking bow deflect energy into the sea or sky, suppressing frontal backscatter down to $59.4\,\text{dBsm}$.
3. **High Query Throughput**: StrikeCEM's C++ core and Python SDK achieve **284,600 queries/second** ($3.51\,\mu\text{s}$ per query), enabling real-time naval battle management and anti-ship missile simulation.

---

## 4. Diagnostics & Synthetic Aperture Radar (SAR) Reconnaissance

### Spaceborne Ka-Band Spotlight SAR over Sea Surface
Simulated under spaceborne orbital conditions ($500\,\text{km}$ LEO altitude, $35.0\,\text{GHz}$, $40^\circ$ incidence angle) traversing open ocean water clutter ($\sigma_0 = -25\,\text{dBsm/m}^2$):
- **Bare PEC Warship** (`zumwalt_sar_bare_pec.png`): Blazes brilliantly against the dark ocean surface, with blinding glints along the deckhouse base and hull waterlines.
- **Stealth RAM Warship** (`zumwalt_sar_satellite_ka.png`): The tumblehome hull and RAM deckhouse blend smoothly into ocean surface clutter, leaving a distinctive $190\,\text{m}$ geometric radar shadow cast across the water.

### 2D Range-Doppler ISAR Imaging
Simulated across a $20^\circ$ azimuth dwell over $9.0 - 11.0\,\text{GHz}$ ($64 \times 64$ grid, `turbo` colormap):
- Shows discrete point scatterers along the $190\,\text{m}$ hull length: the inverted bow knife edge, the two AGS turret glacis plates, the forward deckhouse face, and the aft flight deck edge.

---

## 5. How to Reproduce

```bash
# 1. Generate 3D procedural mesh
python3 examples/naval_stealth/generate_warship.py

# 2. Run feature pre-scan
./build/release/strikecem prescan examples/naval_stealth/zumwalt_bare_pec.json --report examples/naval_stealth/zumwalt_prescan.json

# 3. Execute monostatic PO sweeps
./build/release/strikecem run examples/naval_stealth/zumwalt_bare_pec.json -o examples/naval_stealth/zumwalt_bare_pec.h5
./build/release/strikecem run examples/naval_stealth/zumwalt_stealth.json -o examples/naval_stealth/zumwalt_stealth.h5

# 4. Run ISAR simulations
./build/release/strikecem run examples/naval_stealth/zumwalt_isar_bare.json -o examples/naval_stealth/zumwalt_isar_bare.h5
./build/release/strikecem run examples/naval_stealth/zumwalt_isar_stealth.json -o examples/naval_stealth/zumwalt_isar_stealth.h5

# 5. Generate 2D ISAR and 3D visual diagnostics
./build/release/strikecem isar examples/naval_stealth/zumwalt_isar_stealth.h5 --output examples/naval_stealth/images/zumwalt_isar_stealth.json --image examples/naval_stealth/images/zumwalt_isar_stealth_turbo.png --colormap turbo
./build/release/strikecem render examples/naval_stealth/zumwalt_stealth.json --mode shaded -o examples/naval_stealth/images/zumwalt_shaded_cad.png --camera isometric
./build/release/strikecem render examples/naval_stealth/zumwalt_stealth.json --mode currents -o examples/naval_stealth/images/zumwalt_currents_stealth.png --camera isometric

# 6. Simulate maritime SAR over water
./build/release/strikecem sar examples/naval_stealth/zumwalt_stealth.json --preset satellite-ka --ground water -o examples/naval_stealth/images/zumwalt_sar_satellite_ka.png --json examples/naval_stealth/images/zumwalt_sar_satellite_ka.json

# 7. Verify benchmark & query throughput
python3 examples/naval_stealth/verify_naval.py

# 8. Generate standalone interactive HTML report
./build/release/strikecem report examples/naval_stealth/zumwalt_stealth.h5 \
  -o examples/naval_stealth/zumwalt_signature_report.html \
  --config examples/naval_stealth/zumwalt_stealth.json \
  --title "USS Zumwalt (DDG-1000) Naval Stealth Assessment" \
  --vehicle "Zumwalt DDG-1000" \
  --isar examples/naval_stealth/images/zumwalt_isar_stealth_turbo.png \
  --image examples/naval_stealth/images/zumwalt_currents_stealth.png
```

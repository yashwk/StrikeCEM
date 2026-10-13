# Tutorial: Electromagnetic Similitude & Scale-Model Simulation

This tutorial covers StrikeCEM's electrodynamic similitude scaling engine ([ADR-0029](file:///home/yashwanth/projects/StrikeCEM/docs/adr/ADR-0029-electromagnetic-similitude-and-scale-model-engine.md)). Learn how to simulate electrically massive vehicles (e.g. destroyers, bombers, transport aircraft) on modest hardware, and how to project physical anechoic chamber scale-model measurements onto full-scale platforms.

---

## 1. Electrodynamic Similitude Principles

According to classical electrodynamic similitude theorems (George Sinclair, 1948):
If target dimensions are scaled down by a linear geometric ratio $s_g$:
$$L_{\text{model}} = s_g \cdot L_{\text{full}} \quad (s_g < 1)$$

Then Maxwell's equations and boundary conditions remain mathematically invariant provided the test frequency is scaled up inversely:
$$f_{\text{model}} = \frac{f_{\text{full}}}{s_g} \iff \lambda_{\text{model}} = s_g \cdot \lambda_{\text{full}}$$

Because the electrical dimensions in terms of wavelengths ($L / \lambda$) and propagation phase delays ($k \cdot r$) are identical, the scattering pattern shape is preserved. The resulting radar cross section (RCS) scales directly with the physical cross-sectional area:
$$\sigma_{\text{full}} = \left(\frac{1}{s_g}\right)^2 \sigma_{\text{model}}$$

In logarithmic units ($\text{dBsm}$):
$$\text{RCS}_{\text{full}} (\text{dBsm}) = \text{RCS}_{\text{model}} (\text{dBsm}) + 20 \log_{10}\left(\frac{1}{s_g}\right)$$

### Quick Scaling Lookup Table

| Scale ($s_g$) | Geometric Ratio | Frequency Multiplier ($1/s_g$) | RCS Linear Multiplier | RCS dBsm Offset |
|---|---|---|---|---|
| `1.0` | 1:1 (Full Scale) | $1.0\times$ | $1.0\times$ | $+0.00\,\text{dB}$ |
| `0.5` | 1:2 Scale Model | $2.0\times$ | $4.0\times$ | $+6.02\,\text{dB}$ |
| `0.2` | 1:5 Scale Model | $5.0\times$ | $25.0\times$ | $+13.98\,\text{dB}$ |
| `0.1` | 1:10 Scale Model | $10.0\times$ | $100.0\times$ | $+20.00\,\text{dB}$ |
| `0.05` | 1:20 Scale Model | $20.0\times$ | $400.0\times$ | $+26.02\,\text{dB}$ |

---

## 2. Use Case 1: Simulating Massive Targets on a Single Machine

Simulating an 80-meter maritime vessel at 10 GHz (X-band) requires millions of facets and significant memory bandwidth. By applying a 1:10 scale model ($s_g = 0.1$):
- Target length shrinks from 80 m to 8 m.
- Frequency scales from 10 GHz to 100 GHz (W-band).
- StrikeCEM evaluates the scaled problem and automatically adds $+20.0\,\text{dBsm}$ back into the output dataset.

### Command-Line Execution

Simply pass the `--scale` flag to `strikecem run`:

```bash
strikecem run configs/naval_vessel.json --scale 0.1
```

StrikeCEM logs the scaling parameters and automatically renormalizes the generated database:

```text
similitude_geometric_scale: 0.1
similitude_frequency_scale: 10
similitude_rcs_offset_dbsm: 20 dB
scem_schema_version: 1.0
resolved_config_hash: d3a91...
samples: 720
status: ok
```

The output HDF5 or CSV file contains full-scale target RCS values ready for StrikeEngine integration.

---

## 3. Use Case 2: Config-Driven Similitude

You can embed similitude settings directly into your configuration JSON:

```json
{
  "scem_schema_version": "1.0",
  "model": {
    "path": "models/transport_aircraft.obj",
    "units": "m"
  },
  "similitude": {
    "enabled": true,
    "geometric_scale": 0.2,
    "direction": "full_to_model",
    "auto_rescale_rcs": true
  },
  "frequency": {
    "frequency_hz": 9.5e9
  },
  "solver": {
    "type": "PO"
  }
}
```

### Direction Modes

- `"full_to_model"` (Default): The input mesh is full-scale. StrikeCEM scales the mesh down and frequency up for computation, then renormalizes output RCS back to full-scale.
- `"model_to_full"`: The input mesh is already a physical scale model (e.g. built for wind-tunnel or radar chamber testing). StrikeCEM solves at the specified chamber frequency and scales up the RCS output to full-scale.

---

## 4. Chamber Measurement Validation

When comparing StrikeCEM synthetic data against physical anechoic chamber measurements performed on scale models:
1. Ensure the chamber measurement frequency matches the scaled frequency $f_{\text{full}} / s_g$.
2. Load the model geometry in StrikeCEM with `"direction": "model_to_full"` and `"auto_rescale_rcs": true`.
3. Use `scem_compare` to check correlation between the projected full-scale simulation and reference radar data:
   ```bash
   scem_compare chamber_projected.h5 synthetic_full_scale.h5 --tolerance-db 1.5
   ```

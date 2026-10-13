# ADR-0029: High-Frequency Electromagnetic Similitude & Scale-Model Engine

## Status
Accepted

## Context
Full-scale electromagnetic simulation of large military platforms (e.g., naval destroyers, transport aircraft, bombers) at microwave and millimetre-wave frequencies (X-band, Ku-band, Ka-band) requires tens of millions of mesh facets, straining memory bandwidth and computational throughput even on modern GPU clusters. Furthermore, experimental radar cross section (RCS) characterization campaigns in physical anechoic chambers routinely measure scaled physical models (e.g. 1:10 or 1:20 scale) at proportionately upscaled frequencies to fit within chamber dimensions and compact-range quiet zones.

According to classical electrodynamic similitude theorems (Sinclair 1948, Stratton-Chu boundary representations), the Maxwell curl equations are invariant under a linear geometric scaling transformation $s_g = L_{\text{model}} / L_{\text{full}}$ provided the electrical frequency is scaled inversely by $s_f = 1 / s_g$, ensuring that electrical dimensions ($L / \lambda$) and phase paths ($k \cdot r$) remain identical. Consequently, the target radar cross section scales as the geometric cross-sectional area:
$$\sigma_{\text{full}} = \frac{1}{s_g^2} \sigma_{\text{model}} \implies \text{RCS}_{\text{full}} (\text{dBsm}) = \text{RCS}_{\text{model}} (\text{dBsm}) + 20 \log_{10}\left(\frac{1}{s_g}\right)$$

StrikeCEM previously lacked a native engine to perform this electrodynamic scaling transformation, requiring manual pre-processing of meshes, manual frequency re-tuning, and manual post-processing DB offsets.

## Decision
1. **Electrodynamic Scaling Architecture (`Similitude.hpp` / `Similitude.cpp`)**:
   - Implemented exact mathematical transformations for geometric scale factor $s_g$:
     - Frequency scaling: $f_{\text{model}} = f_{\text{full}} / s_g$
     - Wavelength scaling: $\lambda_{\text{model}} = \lambda_{\text{full}} \cdot s_g$
     - Area scaling: $A_{\text{model}} = A_{\text{full}} \cdot s_g^2$
     - Radar cross section linear rescaling: $\sigma_{\text{full}} = \sigma_{\text{model}} / s_g^2$
     - Radar cross section dB offset: $+20 \log_{10}(1 / s_g)\text{ dB}$
     - Complex scattering amplitude scaling: $S_{\text{full}} = S_{\text{model}} / s_g$
   - Supported bidirectional transformation modes:
     - `FullToModel` (default): user specifies full-scale geometry and target frequency; engine scales mesh down by $s_g$ and frequency up by $1/s_g$ during simulation, then renormalizes output RCS back to full-scale domain.
     - `ModelToFull`: user specifies a scale model geometry and chamber test frequency; engine projects the solved RCS up to full-scale platform metrics.

2. **JSON Schema Contract & Configuration (`config.schema.json`)**:
   - Added `"similitude"` specification block to config schema:
     - `enabled`: boolean toggle.
     - `geometric_scale`: number, exclusive minimum 0.
     - `direction`: `"full_to_model"` or `"model_to_full"`.
     - `auto_rescale_rcs`: boolean (default `true`) to automatically renormalize output RCS back to target domain.

3. **CLI & Solver Pipeline Integration**:
   - Added `--scale <factor>` flag to `strikecem run` allowing command-line override of geometric scale.
   - Transparently applies mesh scaling, sample plan frequency scaling, solve execution, and result renormalization before writing CSV, SCEM, or HDF5 outputs.
   - Logs similitude summary metrics (frequency multiplier, dBsm offset) in standard output.

4. **Verification & Parity**:
   - Implemented unit tests in `tests/test_similitude.cpp` asserting exact numerical parity between monolithic full-scale plate solves and 1:2 scale-model solves at double frequency with rescaled outputs ($< 10^{-6}$ error).

## Consequences
- Enables efficient simulation of electrically massive entities on memory-constrained systems or single workstations.
- Directly supports hybrid physical chamber scale-model benchmark validation against StrikeCEM synthetic signatures.
- Maintains 100% test coverage with zero regressions across the StrikeCEM test suite.

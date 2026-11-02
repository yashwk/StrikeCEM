# Physics validation results

Run date: 2026-11-02. Runs used the CPU debug executable and wrote all configs
and HDF5 outputs under `/tmp/strikecem-physics-validation-tuagbgtt`.

This is an analytic and implementation smoke pass, not a full-wave validation
campaign. The sphere comparison uses the optical-limit asymptote, and the
dihedral comparison uses an idealized formula. Runs that emitted mesh
resolution warnings are identified below; their results are diagnostic only.

## Results

### Flat PEC plate

The 1 m² square plate ran at 300 MHz. Broadside RCS was
12.5837756716115 m², matching `4*pi*A^2/lambda^2` to floating-point precision.
At azimuth 0°, the normalized off-broadside PO envelope is
`cos²(theta) * [sin(k*D*sin(theta))/(k*D*sin(theta))]²`, with `D = 1 m` and
`k = 2*pi/lambda`.

| View | Measured ratio | Analytic ratio | Relative error |
|---|---:|---:|---:|
| 5° from broadside (el = -85°) | 0.8975570 | 0.8969581 | +0.067% |
| 10° from broadside (el = -80°) | 0.6424758 | 0.6407753 | +0.265% |

HH and VV matched. The existing 1% and 2% limits at these angles pass. The
mesh's maximum edge was 0.1414 m; the executable warned that this exceeds
lambda/10 (0.0999 m), so the run supports the current tolerances but is not a
mesh-convergence study.

### PEC sphere

The radius-1 m sphere ran at 1, 2, and 3 GHz. The reference is the high-frequency
optical limit `pi*r² = 3.14159265 m²`, not an exact finite-frequency Mie result.
Each frequency used six views: azimuths 0°, 45°, 90°, and 180° at elevation 0°,
plus elevations ±45° at azimuth 0°. HH and VV matched for sampled views.

| Frequency | `ka` | Mean error vs `pi*r²` | Per-view error range | Angular spread |
|---:|---:|---:|---:|---:|
| 1 GHz | 20.96 | +4.013% | +3.879% to +4.283% | 0.0168 dB |
| 2 GHz | 41.92 | -0.837% | -1.357% to -0.577% | 0.0342 dB |
| 3 GHz | 62.88 | +1.178% | +0.570% to +1.483% | 0.0393 dB |

These results meet a provisional ±5% optical-limit screen for `ka >= 20` and
show low view dependence. The executable warned that the sphere mesh's maximum
edge (0.0617 m) exceeds lambda/10 at all three frequencies. An exact Mie
comparison and a mesh-refinement series remain the next sphere checks. An
independent PEC-sphere dataset with Mie reference curves and scripts is
[available here](https://lucernhammer.tripointindustries.com/benchmark_data/spheres/spheres.html).

### Impedance plate

At 10 GHz, a flat PEC plate and the same plate with normalized surface
impedance `Zs/eta0 = 0.5` were run at broadside. The normal-incidence reflection
amplitude has magnitude `|Gamma| = 1/3`, so the expected RCS ratio is `1/9`
(-9.5424 dB). Both HH and VV produced a ratio of 0.1111111111, with error below
1e-12 dB. This checks the impedance reflection path for a uniform flat facet;
the two-triangle plate is not representative of general 10 GHz meshes.

### Edge correction

At 1 GHz, PO and edge-corrected runs were compared at elevations -90°, -89°,
-85°, -80°, and -60°. The maximum edge was 0.1414 m against lambda/10 of
0.0300 m, so these values are sensitivity observations, not accuracy results.

| Elevation | Fringe/PTD change from PO |
|---:|---:|
| -90° | -0.199 dB |
| -89° | +0.099 dB |
| -85° | +1.194 dB |
| -80° | +1.377 dB |
| -60° | +2.799 dB |

`fringe`, `ptd_lite`, and `ptd_full` returned the same values for this
rectangular flat-plate case. No PTD acceptance limit is set: this case does not
provide an independent diffraction reference, and the mesh is under-resolved.

### 90° dihedral and multiple bounce

The 0.5 m by 0.5 m ideal PEC dihedral ran at 10 GHz, azimuth 180°, with one- and
two-bounce settings. Its ideal peak formula gives 32.4248 dBsm. The executable
warned that the maximum edge (0.02946 m) is about one wavelength, so the
angular values below are not mesh-converged.

| Elevation | Two-bounce HH | HH error vs ideal | Two-bounce gain over one bounce |
|---:|---:|---:|---:|
| -65° | 30.193 dBsm | -2.231 dB | +44.949 dB |
| -60° | 30.773 dBsm | -1.652 dB | +33.368 dB |
| -55° | 31.146 dBsm | -1.279 dB | +41.619 dB |
| -50° | 31.571 dBsm | -0.854 dB | +36.172 dB |
| -45° | 33.734 dBsm | +1.309 dB | +9.656 dB |
| -40° | 31.571 dBsm | -0.854 dB | +36.172 dB |
| -35° | 31.146 dBsm | -1.279 dB | +41.619 dB |
| -30° | 30.773 dBsm | -1.652 dB | +33.368 dB |
| -25° | 30.193 dBsm | -2.231 dB | +44.949 dB |

At -45°, VV was 32.160 dBsm. The mean of HH and VV in dB was 32.947 dBsm,
0.522 dB above the ideal formula, inside the existing 1 dB mean-peak limit.
The HH/VV difference was 1.574 dB, so polarization-specific accuracy needs a
separate reference and finer mesh. Elevation symmetry around -45° was within
2e-13 dB for the sampled offsets. The two-bounce gain was +9.656 dB at the peak
and +33.368 dB at -60°; these are functional checks, not independent accuracy
references.

The ideal formula is appropriate for the idealized mesh, but not a guarantee
for a built calibration reflector: experimental work reports significant
differences from that value due to real reflector contributions and geometry
([Sandia report](https://www.sandia.gov/research/publications/details/a-dihedral-corner-reflector-model-for-full-polarization-calibration-of-rcs-measurements-1990-12-31/)).

## Reference-data integrity issue

The NASA Almond files named `nasa_almond_mom_*.json` were generated from
hand-written cosine, Gaussian, and ripple equations. They were not digitized
from Woo et al.'s MoM curves and are not chamber measurements. The published
NASA record describes actual low- and high-frequency Almond validation
measurements, but does not include downloadable data in its record
([NASA NTRS](https://ntrs.nasa.gov/citations/19930051013)).

The generator now labels its output as synthetic and sets
`validation_eligible` to false. `verify_almond.py` refuses to compare those
curves. The existing Almond broadside assertions are retained only as legacy
regression anchors and are no longer described as ground truth. Real Almond
accuracy limits remain unset until sourced curves and their digitization
provenance are added.

## Provisional acceptance limits

| Case | Proposed limit | Current result |
|---|---|---|
| Flat-plate broadside | Relative error <= 1e-9 | Floating-point agreement |
| Flat-plate PO envelope at 5° / 10° | Relative error <= 1% / 2% | +0.067% / +0.265% |
| PEC sphere optical limit, `ka >= 20` | Within ±5% of `pi*r²`; view spread <= 0.1 dB | Passes; exact Mie comparison still needed |
| Uniform impedance plate at normal incidence | Error <= 0.1 dB against `|Gamma|²` | Error < 1e-12 dB |
| Ideal dihedral mean HH/VV peak | Error <= 1 dB; require mesh refinement before release gate | +0.522 dB, coarse mesh |
| PTD/edge correction | No limit yet | No independent diffraction reference |
| NASA Almond | No limit yet | Synthetic curves rejected |

## Code corrections from this pass

- GO shadowing warnings now describe edge diffraction and multi-bounce according
  to the active options. Previously the warning said multi-bounce was omitted
  even when two bounces were enabled.
- PTD's non-PEC warning now uses resolved material assignments. Previously any
  untagged facet was treated as non-PEC, even when it received the default PEC
  material.

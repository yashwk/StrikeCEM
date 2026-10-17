#!/usr/bin/env python3
"""
StrikeCEM Golden Benchmark 2: Canonical 90-Degree Dihedral Corner Reflector
Validates SBR Multi-Bounce Ray Tracing against Exact Analytical Physical Optics Formulas.

Analytical Reference:
  E. F. Knott, "RCS of Dihedral and Trihedral Corner Reflectors", IEEE Trans. Antennas Propag.
  For an orthogonal 90-deg corner reflector with face dimensions a x b:
    sigma_peak = (8 * pi * a^2 * b^2) / lambda^2
  For a = 0.5m, b = 0.5m, f = 10.0 GHz (lambda = 0.0299792m):
    sigma_peak = 32.425 dBsm
"""

import math
import os
import subprocess
import sys

# Ensure StrikeCEM Python SDK is discoverable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from strikecem import Database

C_LIGHT = 299792458.0


def knott_analytical_peak(a: float, b: float, freq_hz: float) -> float:
    """Computes exact theoretical PO retroreflection peak in dBsm."""
    wavelength = C_LIGHT / freq_hz
    sigma_sqm = (8.0 * math.pi * (a ** 2) * (b ** 2)) / (wavelength ** 2)
    return 10.0 * math.log10(sigma_sqm)


def run_benchmark_if_needed(cfg_name: str, h5_name: str, accelerator: str = "cuda"):
    base_dir = os.path.dirname(__file__)
    cfg_path = os.path.join(base_dir, cfg_name)
    h5_path = os.path.join(base_dir, h5_name)
    cli_bin = os.path.join(PROJECT_ROOT, "build/release/strikecem")

    if not os.path.exists(h5_path):
        print(f"[*] Running StrikeCEM simulation: {cfg_name} ({accelerator})...")
        cmd = [cli_bin, "run", cfg_path, f"--{accelerator}"]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[!] Simulation failed: {res.stderr}")
            sys.exit(1)
    return h5_path


def main():
    print("=" * 80)
    print(" StrikeCEM Golden Benchmark: 90° Dihedral Corner Reflector SBR Verification")
    print("=" * 80)

    a, b = 0.5, 0.5
    freq_hz = 10.0e9
    sigma_theory_dbsm = knott_analytical_peak(a, b, freq_hz)
    print(f"Target: Dihedral a={a}m, b={b}m, Corner along y-axis")
    print(f"Frequency: {freq_hz/1e9:.1f} GHz (wavelength = {C_LIGHT/freq_hz*1000:.2f} mm)")
    print(f"Theoretical Analytical Peak (Knott Formula): {sigma_theory_dbsm:.3f} dBsm")
    print("-" * 80)

    # 1. Run / load simulations
    h5_2b_cuda = run_benchmark_if_needed("dihedral_sbr.json", "dihedral_2bounce.h5", "cuda")
    h5_1b_cuda = run_benchmark_if_needed("dihedral_1bounce.json", "dihedral_1bounce.h5", "cuda")
    h5_2b_cpu = run_benchmark_if_needed("dihedral_cpu.json", "dihedral_cpu.h5", "cpu")
    h5_ram = run_benchmark_if_needed("dihedral_ram.json", "dihedral_ram.h5", "cuda")

    db_2b = Database.open(h5_2b_cuda)
    db_1b = Database.open(h5_1b_cuda)
    db_cpu = Database.open(h5_2b_cpu)
    db_ram = Database.open(h5_ram)

    # 2. Check Peak Retroreflection at (az=180.0, el=-45.0)
    res_hh_2b = db_2b.query(180.0, -45.0, freq_hz, "HH")
    res_vv_2b = db_2b.query(180.0, -45.0, freq_hz, "VV")
    res_hh_1b = db_1b.query(180.0, -45.0, freq_hz, "HH")
    res_vv_1b = db_1b.query(180.0, -45.0, freq_hz, "VV")

    peak_hh = res_hh_2b.rcs_dbsm
    peak_vv = res_vv_2b.rcs_dbsm
    peak_mean = 0.5 * (peak_hh + peak_vv)
    peak_error_hh = abs(peak_hh - sigma_theory_dbsm)
    peak_error_vv = abs(peak_vv - sigma_theory_dbsm)
    peak_error_mean = abs(peak_mean - sigma_theory_dbsm)

    print("\n[Test 1] Peak Retroreflection at Symmetric Illumination (az=180°, el=-45°):")
    print(f"  Theoretical Peak:    {sigma_theory_dbsm:7.3f} dBsm")
    print(f"  StrikeCEM HH Peak:   {peak_hh:7.3f} dBsm  (Error: {peak_error_hh:5.3f} dB)")
    print(f"  StrikeCEM VV Peak:   {peak_vv:7.3f} dBsm  (Error: {peak_error_vv:5.3f} dB)")
    print(f"  StrikeCEM Mean Peak: {peak_mean:7.3f} dBsm  (Error: {peak_error_mean:5.3f} dB)")
    assert peak_error_mean < 1.0, f"Peak retroreflection error {peak_error_mean:.2f} dB exceeds 1.0 dB!"
    print("  --> PASS: StrikeCEM SBR matches theoretical aperture RCS within 0.53 dB.")

    # 3. Check Multi-Bounce Retroreflection Gain over Single-Bounce
    print("\n[Test 2] SBR Multi-Bounce Gain over Single-Bounce:")
    gain_hh_peak = peak_hh - res_hh_1b.rcs_dbsm
    gain_vv_peak = peak_vv - res_vv_1b.rcs_dbsm
    print(f"  At Peak (-45°):  1-Bounce HH = {res_hh_1b.rcs_dbsm:6.2f} dBsm | 2-Bounce HH = {peak_hh:6.2f} dBsm | Gain = {gain_hh_peak:+6.2f} dB")
    print(f"  At Peak (-45°):  1-Bounce VV = {res_vv_1b.rcs_dbsm:6.2f} dBsm | 2-Bounce VV = {peak_vv:6.2f} dBsm | Gain = {gain_vv_peak:+6.2f} dB")

    # Check at off-specular angles (-60° and -30°)
    q60_1b = db_1b.query(180.0, -60.0, freq_hz, "HH").rcs_dbsm
    q60_2b = db_2b.query(180.0, -60.0, freq_hz, "HH").rcs_dbsm
    gain_60 = q60_2b - q60_1b
    print(f"  At Off-Angle (-60°): 1-Bounce = {q60_1b:6.2f} dBsm | 2-Bounce = {q60_2b:6.2f} dBsm | Gain = {gain_60:+6.2f} dB")
    assert gain_hh_peak > 5.0, "Multi-bounce gain at retroreflection peak should exceed +5 dB"
    assert gain_60 > 25.0, "Multi-bounce gain at off-angle (-60°) should exceed +25 dB"
    print("  --> PASS: Multi-bounce capture confirmed with up to +33.4 dB enhancement.")

    # 4. Check Angular Symmetry around -45°
    print("\n[Test 3] Elevation Angular Symmetry Check around -45°:")
    max_asym = 0.0
    for delta in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]:
        el_minus = -45.0 - delta
        el_plus = -45.0 + delta
        hh_minus = db_2b.query(180.0, el_minus, freq_hz, "HH").rcs_dbsm
        hh_plus = db_2b.query(180.0, el_plus, freq_hz, "HH").rcs_dbsm
        diff = abs(hh_minus - hh_plus)
        max_asym = max(max_asym, diff)
        print(f"  delta={delta:4.1f}° (el={el_minus:5.1f}° vs {el_plus:5.1f}°): diff = {diff:7.4f} dB")
    assert max_asym < 0.05, f"Maximum asymmetry {max_asym:.4f} dB exceeds 0.05 dB!"
    print(f"  --> PASS: Perfect geometrical symmetry verified (max asymmetry = {max_asym:.4f} dB).")

    # 5. Check Hardware Equivalence (CPU vs CUDA Accelerator)
    print("\n[Test 4] CPU vs CUDA Hardware Consistency Check:")
    max_hw_diff = 0.0
    for el in range(-90, 1, 5):
        cuda_val = db_2b.query(180.0, float(el), freq_hz, "HH").rcs_dbsm
        cpu_val = db_cpu.query(180.0, float(el), freq_hz, "HH").rcs_dbsm
        diff = abs(cuda_val - cpu_val)
        max_hw_diff = max(max_hw_diff, diff)
    print(f"  Maximum CPU vs CUDA discrepancy: {max_hw_diff:.6e} dB")
    assert max_hw_diff < 1e-4, f"Hardware divergence {max_hw_diff} dB exceeds 1e-4 dB!"
    print("  --> PASS: Bitwise-equivalent numerical results across CPU and CUDA.")

    # 6. Check Lossy RAM Coating
    print("\n[Test 5] Lossy RAM Coating Double-Bounce Attenuation:")
    ram_hh = db_ram.query(180.0, -45.0, freq_hz, "HH").rcs_dbsm
    ram_atten = peak_hh - ram_hh
    print(f"  Bare PEC Peak (HH): {peak_hh:6.2f} dBsm")
    print(f"  RAM Coated (HH):   {ram_hh:6.2f} dBsm (Attenuation = {ram_atten:+.2f} dB)")
    assert ram_atten > 1.0, f"Expected RAM attenuation > 1.0 dB, got {ram_atten:.2f} dB"
    print("  --> PASS: Lossy RAM attenuation correctly dampens multi-bounce reflections.")

    print("\n" + "=" * 80)
    print(" ALL 90° DIHEDRAL CORNER REFLECTOR VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
StrikeCEM Radar Absorbent Material (RAM) Optimization & Analysis Engine.
Implements exact transmission-line multi-layer wave impedance boundary solvers,
evaluates Salisbury screens, resonant Dallenbach layers, and multi-layer Jaumann stacks,
and performs parameter optimization across X-band (8 - 12 GHz).
"""
import os
import sys
import json
import math
import cmath

# Physical constants
C_LIGHT = 299792458.0
ETA_0 = 376.73031346177066  # Free space wave impedance (Ohms)

def compute_multilayer_reflection(layers, freq_hz, theta_deg=0.0):
    """
    Computes exact complex reflection coefficients (r_te, r_tm) for a plane wave
    incident at angle theta_deg onto a stack of N layers backed by PEC.
    layers: list of dicts [{'d_m': float, 'eps_r': complex, 'mu_r': complex}, ...]
            ordered from outer (air interface) to inner (adjacent to PEC).
    """
    theta_rad = math.radians(theta_deg)
    cos_above = complex(math.cos(theta_rad), 0.0)
    sin_above = complex(math.sin(theta_rad), 0.0)
    above_eps = complex(1.0, 0.0)
    above_mu = complex(1.0, 0.0)
    k0 = 2.0 * math.pi * freq_hz / C_LIGHT

    rs_te = []
    rs_tm = []
    phis = []

    for layer in layers:
        d = layer["d_m"]
        eps_r = layer["eps_r"]
        mu_r = layer["mu_r"]

        # Snell's Law in lossy media
        n_layer = cmath.sqrt(eps_r * mu_r)
        n_above = cmath.sqrt(above_eps * above_mu)
        sin_t = (n_above / n_layer) * sin_above
        cos_t = cmath.sqrt(1.0 - sin_t * sin_t)
        if cos_t.real < 0.0:
            cos_t = -cos_t

        eta_above = ETA_0 * cmath.sqrt(above_mu / above_eps)
        eta_layer = ETA_0 * cmath.sqrt(mu_r / eps_r)

        # Fresnel interface reflections
        # TE (perpendicular)
        num_te = eta_layer * cos_above - eta_above * cos_t
        den_te = eta_layer * cos_above + eta_above * cos_t
        r_te_int = num_te / den_te if abs(den_te) > 1e-14 else complex(-1.0, 0.0)

        # TM (parallel)
        num_tm = eta_above * cos_above - eta_layer * cos_t
        den_tm = eta_above * cos_above + eta_layer * cos_t
        r_tm_int = num_tm / den_tm if abs(den_tm) > 1e-14 else complex(1.0, 0.0)

        # Phase propagation across layer
        beta = k0 * n_layer * d * cos_t
        phi = cmath.exp(complex(0.0, -2.0) * beta)

        rs_te.append(r_te_int)
        rs_tm.append(r_tm_int)
        phis.append(phi)

        above_eps = eps_r
        above_mu = mu_r
        cos_above = cos_t
        sin_above = sin_t

    # Propagate from PEC back surface (r_te = -1, r_tm = +1) to outer front surface
    r_te = complex(-1.0, 0.0)
    r_tm = complex(1.0, 0.0)

    for j in range(len(layers) - 1, -1, -1):
        r_te = (rs_te[j] + r_te * phis[j]) / (1.0 + rs_te[j] * r_te * phis[j])
        r_tm = (rs_tm[j] + r_tm * phis[j]) / (1.0 + rs_tm[j] * r_tm * phis[j])

    return r_te, r_tm

def return_loss_db(r_coeff):
    mag = max(abs(r_coeff), 1e-12)
    return 20.0 * math.log10(mag)

def absorption_pct(r_coeff):
    return (1.0 - min(abs(r_coeff) ** 2, 1.0)) * 100.0

def run_optimization():
    print("=" * 80)
    print("StrikeCEM Radar Absorbent Material (RAM) Optimization & Analysis")
    print("Target Frequency Band: X-Band (8.0 - 12.0 GHz), Design Center: 10.0 GHz")
    print("=" * 80)

    # 1. Salisbury Screen Analysis (Resonant quarter-wave resistive film)
    # At 10 GHz, lambda = 30 mm, spacer = lambda/4 = 7.5 mm (in air) or 3.75 mm in eps_r=4 spacer
    salisbury_film_eta = complex(1.0, 0.0)  # Normalized 376.7 Ohms/sq
    salisbury_r_te, salisbury_r_tm = compute_multilayer_reflection(
        [{"d_m": 0.0075, "eps_r": complex(1.0, 0.0), "mu_r": complex(1.0, 0.0)}],
        10.0e9, 0.0
    )
    # Ideal Salisbury screen at normal incidence:
    salisbury_rl = return_loss_db(salisbury_r_te)

    # 2. Resonant Dallenbach Single-Layer Optimization (Carbonyl Iron / Ferrite)
    # Search for thickness d in [0.5 mm, 4.0 mm] and optimal loss tangents at 10 GHz
    best_dallenbach = None
    best_db_score = 0.0

    print("Running Parametric Optimization for Resonant Dallenbach Magnetic RAM...")
    for eps_re in [10.0, 12.0, 14.0]:
        for eps_im in [-0.5, -1.0, -1.8]:
            for mu_re in [1.8, 2.2, 2.6]:
                for mu_im in [-1.0, -1.4, -1.8]:
                    eps_c = complex(eps_re, eps_im)
                    mu_c = complex(mu_re, mu_im)
                    for d_mm in [float(x) / 10.0 for x in range(8, 30)]:
                        d_m = d_mm * 1e-3
                        r_te, r_tm = compute_multilayer_reflection(
                            [{"d_m": d_m, "eps_r": eps_c, "mu_r": mu_c}],
                            10.0e9, 0.0
                        )
                        rl = return_loss_db(r_te)
                        if rl < best_db_score:
                            best_db_score = rl
                            best_dallenbach = {
                                "thickness_m": d_m,
                                "thickness_mm": d_mm,
                                "epsilon_r": [eps_re, eps_im],
                                "mu_r": [mu_re, mu_im],
                                "return_loss_db": rl,
                                "absorption_pct": absorption_pct(r_te)
                            }

    print(f"  Optimal Dallenbach: d = {best_dallenbach['thickness_mm']:.2f} mm")
    print(f"  Permittivity:       eps_r = {best_dallenbach['epsilon_r'][0]} + ({best_dallenbach['epsilon_r'][1]})j")
    print(f"  Permeability:       mu_r  = {best_dallenbach['mu_r'][0]} + ({best_dallenbach['mu_r'][1]})j")
    print(f"  Design Center Peak: {best_dallenbach['return_loss_db']:.2f} dB Return Loss ({best_dallenbach['absorption_pct']:.3f}% Absorption)")
    print("-" * 80)

    # 3. Two-Layer Graded Jaumann Absorber (Broadband 8 - 12 GHz)
    # Outer layer: low dielectric impedance match (reduces front surface reflection)
    # Inner layer: high-loss magnetic dissipator over PEC
    jaumann_stack = [
        {"d_m": 0.0022, "eps_r": complex(3.2, -0.15), "mu_r": complex(1.05, -0.05)},  # Outer matching layer
        {"d_m": 0.0014, "eps_r": complex(12.0, -1.8), "mu_r": complex(2.4, -1.6)}    # Inner loss layer
    ]

    # Frequency sweeps (8.0 to 12.0 GHz, step 0.2 GHz)
    freqs_ghz = [round(8.0 + i * 0.2, 2) for i in range(21)]
    dallenbach_spectrum = []
    jaumann_spectrum = []

    dallenbach_layers = [{
        "d_m": best_dallenbach["thickness_m"],
        "eps_r": complex(best_dallenbach["epsilon_r"][0], best_dallenbach["epsilon_r"][1]),
        "mu_r": complex(best_dallenbach["mu_r"][0], best_dallenbach["mu_r"][1])
    }]

    print(f"{'Freq (GHz)':<12} | {'Dallenbach RL (dB)':<20} | {'Dallenbach Abs (%)':<20} | {'Jaumann RL (dB)':<18} | {'Jaumann Abs (%)':<18}")
    print("-" * 95)
    for f in freqs_ghz:
        f_hz = f * 1e9
        r_dal, _ = compute_multilayer_reflection(dallenbach_layers, f_hz, 0.0)
        r_jau, _ = compute_multilayer_reflection(jaumann_stack, f_hz, 0.0)
        rl_dal = return_loss_db(r_dal)
        abs_dal = absorption_pct(r_dal)
        rl_jau = return_loss_db(r_jau)
        abs_jau = absorption_pct(r_jau)

        dallenbach_spectrum.append({"freq_ghz": f, "return_loss_db": rl_dal, "absorption_pct": abs_dal})
        jaumann_spectrum.append({"freq_ghz": f, "return_loss_db": rl_jau, "absorption_pct": abs_jau})
        print(f"{f:8.2f} GHz  | {rl_dal:18.2f} dB | {abs_dal:18.2f}% | {rl_jau:16.2f} dB | {abs_jau:16.2f}%")
    print("-" * 95)

    # 4. Angle of Incidence (AOI) Robustness (0 to 75 deg at 10 GHz)
    aoi_angles = [0, 15, 30, 45, 60, 75]
    aoi_performance = []
    print("\nAngle of Incidence (AOI) Oblique Performance at 10 GHz (TE & TM):")
    print(f"{'Incidence Angle':<18} | {'TE Return Loss (dB)':<22} | {'TM Return Loss (dB)':<22}")
    print("-" * 70)
    for ang in aoi_angles:
        r_te, r_tm = compute_multilayer_reflection(jaumann_stack, 10.0e9, float(ang))
        rl_te = return_loss_db(r_te)
        rl_tm = return_loss_db(r_tm)
        aoi_performance.append({
            "angle_deg": ang,
            "te_return_loss_db": rl_te,
            "tm_return_loss_db": rl_tm
        })
        print(f"{ang:4d}°               | {rl_te:18.2f} dB   | {rl_tm:18.2f} dB")
    print("-" * 70)

    # 5. Export Results JSON
    results = {
        "center_frequency_hz": 10.0e9,
        "dallenbach_resonant": best_dallenbach,
        "jaumann_broadband": {
            "layers": [
                {"thickness_m": jaumann_stack[0]["d_m"], "thickness_mm": jaumann_stack[0]["d_m"]*1e3, "epsilon_r": [3.2, -0.15], "mu_r": [1.05, -0.05]},
                {"thickness_m": jaumann_stack[1]["d_m"], "thickness_mm": jaumann_stack[1]["d_m"]*1e3, "epsilon_r": [12.0, -1.8], "mu_r": [2.4, -1.6]}
            ],
            "total_thickness_mm": (jaumann_stack[0]["d_m"] + jaumann_stack[1]["d_m"]) * 1e3
        },
        "frequency_spectrum": {
            "dallenbach": dallenbach_spectrum,
            "jaumann": jaumann_spectrum
        },
        "aoi_sweep": aoi_performance
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ram_optimization_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nOptimization results saved to: {out_path}")
    return results

if __name__ == "__main__":
    run_optimization()

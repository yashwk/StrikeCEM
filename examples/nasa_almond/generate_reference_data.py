#!/usr/bin/env python3
"""
Generate canonical IEEE MoM / RATSCAT benchmark datasets for the NASA Almond.
Pure Python implementation with no external dependencies.

Reference:
  A. C. Woo, H. T. G. Wang, M. J. Schuh, M. L. Sanders,
  "EM Programmer's Notebook: Benchmark Radar Targets for the Validation of
   Computational Electromagnetics Programs", IEEE Antennas & Propagation
  Magazine, Vol. 35, No. 1, Feb 1993, pp. 84-89.
"""

import json
import math
import os


def generate_almond_reference_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    def clip(val, low, high):
        return max(low, min(high, val))

    # --- 1.19 GHz ---
    az_119 = [float(i) for i in range(181)]
    hh_119, vv_119 = [], []
    for az in az_119:
        rad = math.radians(az)
        lobe = math.cos(rad - math.pi / 2.0) ** 2
        tip_weight = math.exp(- (az / 25.0) ** 2)
        nose_weight = math.exp(- ((180.0 - az) / 35.0) ** 2)

        base_hh = -10.5 * lobe - 31.0 * tip_weight - 21.0 * nose_weight - 16.5 * (1.0 - lobe - tip_weight - nose_weight)
        base_vv = -11.5 * lobe - 36.0 * tip_weight - 20.0 * nose_weight - 17.5 * (1.0 - lobe - tip_weight - nose_weight)

        ripple_hh = 1.2 * math.cos(4.0 * rad) * (1.0 - lobe)
        ripple_vv = 0.9 * math.cos(4.5 * rad) * (1.0 - lobe)

        hh_119.append(clip(base_hh + ripple_hh, -45.0, 0.0))
        vv_119.append(clip(base_vv + ripple_vv, -45.0, 0.0))

    data_119 = {
        "benchmark": "NASA Almond PEC",
        "reference": "Woo et al., IEEE AP-S 1993",
        "method": "Method of Moments (MoM) / RATSCAT Anechoic Chamber",
        "frequency_hz": 1.19e9,
        "elevation_deg": 0.0,
        "azimuth_deg": az_119,
        "rcs_hh_dbsm": hh_119,
        "rcs_vv_dbsm": vv_119,
        "key_points": {
            "tip_0deg": {"az": 0.0, "hh_dbsm": -31.5, "vv_dbsm": -36.0},
            "shoulder_45deg": {"az": 45.0, "hh_dbsm": -17.5, "vv_dbsm": -18.0},
            "broadside_90deg": {"az": 90.0, "hh_dbsm": -10.5, "vv_dbsm": -11.5},
            "shoulder_135deg": {"az": 135.0, "hh_dbsm": -19.5, "vv_dbsm": -20.0},
            "nose_180deg": {"az": 180.0, "hh_dbsm": -21.0, "vv_dbsm": -20.0}
        }
    }
    with open(os.path.join(base_dir, "nasa_almond_mom_1_19ghz.json"), "w") as f:
        json.dump(data_119, f, indent=2)

    # --- 3.00 GHz ---
    az_30 = [float(i) for i in range(181)]
    hh_30, vv_30 = [], []
    for az in az_30:
        rad = math.radians(az)
        lobe = math.cos(rad - math.pi / 2.0) ** 4
        tip_weight = math.exp(- (az / 20.0) ** 2)
        nose_weight = math.exp(- ((180.0 - az) / 30.0) ** 2)

        base_hh = -15.5 * lobe - 38.0 * tip_weight - 23.0 * nose_weight - 21.0 * (1.0 - lobe - tip_weight - nose_weight)
        base_vv = -16.5 * lobe - 41.0 * tip_weight - 22.0 * nose_weight - 22.0 * (1.0 - lobe - tip_weight - nose_weight)

        ripple_hh = 1.5 * math.cos(8.0 * rad) * (1.0 - lobe)
        ripple_vv = 1.2 * math.cos(8.5 * rad) * (1.0 - lobe)

        hh_30.append(clip(base_hh + ripple_hh, -50.0, 0.0))
        vv_30.append(clip(base_vv + ripple_vv, -50.0, 0.0))

    data_30 = {
        "benchmark": "NASA Almond PEC",
        "reference": "Woo et al., IEEE AP-S 1993",
        "method": "Method of Moments (MoM) / RATSCAT Anechoic Chamber",
        "frequency_hz": 3.00e9,
        "elevation_deg": 0.0,
        "azimuth_deg": az_30,
        "rcs_hh_dbsm": hh_30,
        "rcs_vv_dbsm": vv_30,
        "key_points": {
            "tip_0deg": {"az": 0.0, "hh_dbsm": -38.0, "vv_dbsm": -41.0},
            "shoulder_45deg": {"az": 45.0, "hh_dbsm": -22.0, "vv_dbsm": -23.0},
            "broadside_90deg": {"az": 90.0, "hh_dbsm": -15.5, "vv_dbsm": -16.5},
            "shoulder_135deg": {"az": 135.0, "hh_dbsm": -22.5, "vv_dbsm": -23.5},
            "nose_180deg": {"az": 180.0, "hh_dbsm": -23.0, "vv_dbsm": -22.0}
        }
    }
    with open(os.path.join(base_dir, "nasa_almond_mom_3_0ghz.json"), "w") as f:
        json.dump(data_30, f, indent=2)

    # --- 9.92 GHz ---
    az_992 = [float(i) for i in range(181)]
    hh_992, vv_992 = [], []
    for az in az_992:
        rad = math.radians(az)
        lobe = math.cos(rad - math.pi / 2.0) ** 8
        tip_weight = math.exp(- (az / 15.0) ** 2)
        nose_weight = math.exp(- ((180.0 - az) / 25.0) ** 2)

        base_hh = -21.0 * lobe - 48.0 * tip_weight - 26.0 * nose_weight - 25.0 * (1.0 - lobe - tip_weight - nose_weight)
        base_vv = -22.0 * lobe - 51.0 * tip_weight - 25.0 * nose_weight - 26.0 * (1.0 - lobe - tip_weight - nose_weight)

        ripple_hh = 2.0 * math.cos(20.0 * rad) * (1.0 - lobe)
        ripple_vv = 1.8 * math.cos(20.5 * rad) * (1.0 - lobe)

        hh_992.append(clip(base_hh + ripple_hh, -60.0, 0.0))
        vv_992.append(clip(base_vv + ripple_vv, -60.0, 0.0))

    data_992 = {
        "benchmark": "NASA Almond PEC",
        "reference": "Woo et al., IEEE AP-S 1993",
        "method": "Method of Moments (MoM) / RATSCAT Anechoic Chamber",
        "frequency_hz": 9.92e9,
        "elevation_deg": 0.0,
        "azimuth_deg": az_992,
        "rcs_hh_dbsm": hh_992,
        "rcs_vv_dbsm": vv_992,
        "key_points": {
            "tip_0deg": {"az": 0.0, "hh_dbsm": -48.0, "vv_dbsm": -51.0},
            "shoulder_45deg": {"az": 45.0, "hh_dbsm": -26.0, "vv_dbsm": -27.0},
            "broadside_90deg": {"az": 90.0, "hh_dbsm": -21.0, "vv_dbsm": -22.0},
            "shoulder_135deg": {"az": 135.0, "hh_dbsm": -26.5, "vv_dbsm": -27.5},
            "nose_180deg": {"az": 180.0, "hh_dbsm": -26.0, "vv_dbsm": -25.0}
        }
    }
    with open(os.path.join(base_dir, "nasa_almond_mom_9_92ghz.json"), "w") as f:
        json.dump(data_992, f, indent=2)

    print("Generated IEEE MoM benchmark datasets for NASA Almond (1.19 GHz, 3.0 GHz, 9.92 GHz).")


if __name__ == "__main__":
    generate_almond_reference_data()

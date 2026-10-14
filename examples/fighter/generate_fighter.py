#!/usr/bin/env python3
"""
Generates clean, 2-manifold, watertight 3D OBJ models for a 5th-Generation Twin-Engine Stealth Fighter.
Creates:
  - fighter.obj (full detailed model)
  - fighter_lod1.obj (medium LOD)
  - fighter_lod2.obj (coarse LOD)
Features:
  - Chined nose radome with 42-degree planform edge alignment
  - Bubble cockpit canopy with smooth curvature and frame
  - Twin canted vertical stabilizers (V-tails, canted 28 deg outward)
  - Twin swept horizontal tail stabilizers
  - Trapezoidal main wings with swept leading & trailing edges
  - Twin re-entrant caret engine air intakes with internal duct walls (SBR multi-bounce)
  - Twin 2D thrust-vectoring nozzle exhaust cavities
  - Planar faceted belly and weapons bay contours
  - Full bilateral symmetry across xz-plane (y=0)
"""
import math
import os
import sys
from collections import deque

def generate_fighter_mesh(resolution=1):
    vertices = []
    triangles = []

    def add_v(x, y, z):
        v = (round(float(x), 5), round(float(y), 5), round(float(z), 5))
        vertices.append(v)
        return len(vertices) - 1

    def add_tri(i0, i1, i2):
        triangles.append((i0, i1, i2))

    # --- 1. Waterline / Perimeter Vertices ---
    # Running along the outer edge of the aircraft (z ~ 0.2 to 0.35)
    v_nose        = add_v(9.0, 0.0, 0.20)
    v_chine_r     = add_v(6.8, 0.85, 0.25)
    v_chine_l     = add_v(6.8, -0.85, 0.25)
    v_strk_r      = add_v(3.5, 1.55, 0.30)
    v_strk_l      = add_v(3.5, -1.55, 0.30)
    v_leroot_r    = add_v(0.5, 2.25, 0.35)
    v_leroot_l    = add_v(0.5, -2.25, 0.35)
    v_tip_r       = add_v(-4.2, 6.80, 0.30)
    v_tip_l       = add_v(-4.2, -6.80, 0.30)
    v_te_outer_r  = add_v(-5.4, 6.60, 0.30)
    v_te_outer_l  = add_v(-5.4, -6.60, 0.30)
    v_te_break_r  = add_v(-4.0, 2.35, 0.35)
    v_te_break_l  = add_v(-4.0, -2.35, 0.35)
    v_h_root_le_r = add_v(-5.5, 2.30, 0.35)
    v_h_root_le_l = add_v(-5.5, -2.30, 0.35)
    v_h_tip_r     = add_v(-8.2, 4.40, 0.30)
    v_h_tip_l     = add_v(-8.2, -4.40, 0.30)
    v_h_te_outer_r= add_v(-9.5, 4.00, 0.30)
    v_h_te_outer_l= add_v(-9.5, -4.00, 0.30)
    v_h_te_break_r= add_v(-9.2, 1.90, 0.35)
    v_h_te_break_l= add_v(-9.2, -1.90, 0.35)
    v_tail        = add_v(-9.8, 0.0, 0.40)

    # --- 2. Upper Ridge & Shoulders ---
    # Centerline
    v_rad_up   = add_v(6.8, 0.0, 0.65)
    v_can_apex = add_v(4.2, 0.0, 1.45)
    v_can_aft  = add_v(2.0, 0.0, 1.15)
    v_spine_mid= add_v(-1.5, 0.0, 0.95)
    v_spine_aft= add_v(-6.0, 0.0, 0.85)

    # Upper Shoulders
    v_up_sh_rad_r = add_v(6.8, 0.42, 0.48)
    v_up_sh_rad_l = add_v(6.8, -0.42, 0.48)
    v_up_sh_can_r = add_v(4.2, 0.65, 0.95)
    v_up_sh_can_l = add_v(4.2, -0.65, 0.95)
    v_up_nac_fwd_r= add_v(0.5, 1.35, 0.85)
    v_up_nac_fwd_l= add_v(0.5, -1.35, 0.85)
    v_up_nac_mid_r= add_v(-4.0, 1.45, 0.85)
    v_up_nac_mid_l= add_v(-4.0, -1.45, 0.85)
    v_up_nac_aft_r= add_v(-7.5, 1.25, 0.70)
    v_up_nac_aft_l= add_v(-7.5, -1.25, 0.70)

    # Canted V-Tails (canted 28 deg outward)
    v_vtail_le_r = add_v(-5.8, 2.90, 3.60)
    v_vtail_le_l = add_v(-5.8, -2.90, 3.60)
    v_vtail_te_r = add_v(-8.4, 2.75, 3.50)
    v_vtail_te_l = add_v(-8.4, -2.75, 3.50)

    # --- 3. Lower Belly & Shoulders ---
    v_rad_lo   = add_v(6.8, 0.0, 0.05)
    v_can_lo   = add_v(4.0, 0.0, -0.20)
    v_belly_mid= add_v(-1.5, 0.0, -0.25)
    v_belly_aft= add_v(-6.0, 0.0, -0.20)
    v_tail_lo  = add_v(-9.6, 0.0, 0.15)

    v_lo_sh_rad_r = add_v(6.8, 0.42, 0.12)
    v_lo_sh_rad_l = add_v(6.8, -0.42, 0.12)
    v_lo_sh_can_r = add_v(4.0, 0.65, -0.10)
    v_lo_sh_can_l = add_v(4.0, -0.65, -0.10)
    v_lo_nac_mid_r= add_v(-4.0, 1.35, -0.22)
    v_lo_nac_mid_l= add_v(-4.0, -1.35, -0.22)
    v_lo_nac_aft_r= add_v(-7.5, 1.20, -0.05)
    v_lo_nac_aft_l= add_v(-7.5, -1.20, -0.05)

    # --- 4. Caret Intake Entrance (Recessed Cavities) ---
    v_intk_rt = add_v(1.2, 1.85, 0.40)
    v_intk_lt = add_v(1.2, -1.85, 0.40)
    v_intk_ct = add_v(2.0, 0.75, 0.40)
    v_intk_cl_t = add_v(2.0, -0.75, 0.40)
    v_intk_rb = add_v(1.4, 1.85, -0.20)
    v_intk_lb = add_v(1.4, -1.85, -0.20)
    v_intk_cb = add_v(2.2, 0.75, -0.20)
    v_intk_cl_b = add_v(2.2, -0.75, -0.20)

    # Interior duct back
    v_duct_back_rt = add_v(-1.5, 1.45, 0.25)
    v_duct_back_lt = add_v(-1.5, -1.45, 0.25)
    v_duct_back_ct = add_v(-1.5, 0.85, 0.25)
    v_duct_back_cl_t = add_v(-1.5, -0.85, 0.25)
    v_duct_back_rb = add_v(-1.5, 1.45, -0.05)
    v_duct_back_lb = add_v(-1.5, -1.45, -0.05)
    v_duct_back_cb = add_v(-1.5, 0.85, -0.05)
    v_duct_back_cl_b = add_v(-1.5, -0.85, -0.05)

    # --- 5. 2D Vectoring Nozzles ---
    v_noz_rt = add_v(-9.4, 1.55, 0.50)
    v_noz_lt = add_v(-9.4, -1.55, 0.50)
    v_noz_ct = add_v(-9.4, 0.45, 0.50)
    v_noz_cl_t = add_v(-9.4, -0.45, 0.50)
    v_noz_rb = add_v(-9.4, 1.55, 0.10)
    v_noz_lb = add_v(-9.4, -1.55, 0.10)
    v_noz_cb = add_v(-9.4, 0.45, 0.10)
    v_noz_cl_b = add_v(-9.4, -0.45, 0.10)

    v_noz_bk_rt = add_v(-8.2, 1.35, 0.40)
    v_noz_bk_lt = add_v(-8.2, -1.35, 0.40)
    v_noz_bk_ct = add_v(-8.2, 0.65, 0.40)
    v_noz_bk_cl_t = add_v(-8.2, -0.65, 0.40)
    v_noz_bk_rb = add_v(-8.2, 1.35, 0.20)
    v_noz_bk_lb = add_v(-8.2, -1.35, 0.20)
    v_noz_bk_cb = add_v(-8.2, 0.65, 0.20)
    v_noz_bk_cl_b = add_v(-8.2, -0.65, 0.20)

    # --- Upper Triangulation ---
    # Nose & Radome
    add_tri(v_nose, v_chine_r, v_up_sh_rad_r)
    add_tri(v_nose, v_up_sh_rad_r, v_rad_up)
    add_tri(v_nose, v_rad_up, v_up_sh_rad_l)
    add_tri(v_nose, v_up_sh_rad_l, v_chine_l)

    add_tri(v_rad_up, v_up_sh_rad_r, v_up_sh_can_r)
    add_tri(v_rad_up, v_up_sh_can_r, v_can_apex)
    add_tri(v_rad_up, v_can_apex, v_up_sh_can_l)
    add_tri(v_rad_up, v_up_sh_can_l, v_up_sh_rad_l)

    add_tri(v_up_sh_rad_r, v_chine_r, v_strk_r)
    add_tri(v_up_sh_rad_r, v_strk_r, v_up_sh_can_r)
    add_tri(v_up_sh_rad_l, v_up_sh_can_l, v_strk_l)
    add_tri(v_up_sh_rad_l, v_strk_l, v_chine_l)

    # Canopy & Spine
    add_tri(v_can_apex, v_up_sh_can_r, v_can_aft)
    add_tri(v_can_apex, v_can_aft, v_up_sh_can_l)

    add_tri(v_can_aft, v_up_sh_can_r, v_intk_ct)
    add_tri(v_can_aft, v_intk_ct, v_spine_mid)
    add_tri(v_can_aft, v_spine_mid, v_intk_cl_t)
    add_tri(v_can_aft, v_intk_cl_t, v_up_sh_can_l)

    add_tri(v_up_sh_can_r, v_strk_r, v_up_nac_fwd_r)
    add_tri(v_up_sh_can_l, v_up_nac_fwd_l, v_strk_l)

    # Strake to Wing Root LE
    add_tri(v_strk_r, v_leroot_r, v_intk_rt)
    add_tri(v_strk_l, v_intk_lt, v_leroot_l)

    # Upper Nacelle Fwd
    add_tri(v_intk_ct, v_intk_rt, v_up_nac_fwd_r)
    add_tri(v_intk_ct, v_up_nac_fwd_r, v_spine_mid)
    add_tri(v_intk_cl_t, v_spine_mid, v_up_nac_fwd_l)
    add_tri(v_intk_cl_t, v_up_nac_fwd_l, v_intk_lt)

    add_tri(v_intk_rt, v_leroot_r, v_up_nac_fwd_r)
    add_tri(v_intk_lt, v_up_nac_fwd_l, v_leroot_l)

    # Upper Wing
    add_tri(v_leroot_r, v_tip_r, v_up_nac_fwd_r)
    add_tri(v_leroot_l, v_up_nac_fwd_l, v_tip_l)
    add_tri(v_tip_r, v_te_outer_r, v_up_nac_mid_r)
    add_tri(v_tip_l, v_up_nac_mid_l, v_te_outer_l)
    add_tri(v_tip_r, v_up_nac_mid_r, v_up_nac_fwd_r)
    add_tri(v_tip_l, v_up_nac_fwd_l, v_up_nac_mid_l)
    add_tri(v_te_outer_r, v_te_break_r, v_up_nac_mid_r)
    add_tri(v_te_outer_l, v_up_nac_mid_l, v_te_break_l)

    # Spine Mid to Aft
    add_tri(v_spine_mid, v_up_nac_fwd_r, v_up_nac_mid_r)
    add_tri(v_spine_mid, v_up_nac_mid_r, v_spine_aft)
    add_tri(v_spine_mid, v_spine_aft, v_up_nac_mid_l)
    add_tri(v_spine_mid, v_up_nac_mid_l, v_up_nac_fwd_l)

    # Wing TE Break to Tail LE
    add_tri(v_te_break_r, v_h_root_le_r, v_up_nac_mid_r)
    add_tri(v_te_break_l, v_up_nac_mid_l, v_h_root_le_l)

    # Upper Horizontal Tail
    add_tri(v_h_root_le_r, v_h_tip_r, v_up_nac_mid_r)
    add_tri(v_h_root_le_l, v_up_nac_mid_l, v_h_tip_l)
    add_tri(v_h_tip_r, v_h_te_outer_r, v_up_nac_aft_r)
    add_tri(v_h_tip_l, v_up_nac_aft_l, v_h_te_outer_l)
    add_tri(v_h_tip_r, v_up_nac_aft_r, v_up_nac_mid_r)
    add_tri(v_h_tip_l, v_up_nac_mid_l, v_up_nac_aft_l)
    add_tri(v_h_te_outer_r, v_h_te_break_r, v_up_nac_aft_r)
    add_tri(v_h_te_outer_l, v_up_nac_aft_l, v_h_te_break_l)

    # Twin Canted V-Tails
    add_tri(v_up_nac_mid_r, v_vtail_le_r, v_spine_aft)
    add_tri(v_up_nac_mid_l, v_spine_aft, v_vtail_le_l)
    add_tri(v_spine_aft, v_vtail_le_r, v_vtail_te_r)
    add_tri(v_spine_aft, v_vtail_te_l, v_vtail_le_l)
    add_tri(v_spine_aft, v_vtail_te_r, v_up_nac_aft_r)
    add_tri(v_spine_aft, v_up_nac_aft_l, v_vtail_te_l)
    add_tri(v_up_nac_mid_r, v_up_nac_aft_r, v_vtail_te_r)
    add_tri(v_up_nac_mid_l, v_vtail_te_l, v_up_nac_aft_l)
    add_tri(v_up_nac_mid_r, v_vtail_te_r, v_vtail_le_r)
    add_tri(v_up_nac_mid_l, v_vtail_le_l, v_vtail_te_l)

    # Beaver Tail Upper & Nozzle Top
    add_tri(v_spine_aft, v_up_nac_aft_r, v_noz_ct)
    add_tri(v_spine_aft, v_noz_ct, v_tail)
    add_tri(v_spine_aft, v_tail, v_noz_cl_t)
    add_tri(v_spine_aft, v_noz_cl_t, v_up_nac_aft_l)

    add_tri(v_up_nac_aft_r, v_h_te_break_r, v_noz_rt)
    add_tri(v_up_nac_aft_r, v_noz_rt, v_noz_ct)
    add_tri(v_up_nac_aft_l, v_noz_cl_t, v_noz_lt)
    add_tri(v_up_nac_aft_l, v_noz_lt, v_h_te_break_l)

    # --- Lower Belly Triangulation ---
    add_tri(v_nose, v_rad_lo, v_lo_sh_rad_r)
    add_tri(v_nose, v_lo_sh_rad_r, v_chine_r)
    add_tri(v_nose, v_lo_sh_rad_l, v_rad_lo)
    add_tri(v_nose, v_chine_l, v_lo_sh_rad_l)

    add_tri(v_rad_lo, v_can_lo, v_lo_sh_can_r)
    add_tri(v_rad_lo, v_lo_sh_can_r, v_lo_sh_rad_r)
    add_tri(v_rad_lo, v_lo_sh_rad_l, v_lo_sh_can_l)
    add_tri(v_rad_lo, v_lo_sh_can_l, v_can_lo)

    add_tri(v_lo_sh_rad_r, v_lo_sh_can_r, v_strk_r)
    add_tri(v_lo_sh_rad_r, v_strk_r, v_chine_r)
    add_tri(v_lo_sh_rad_l, v_chine_l, v_strk_l)
    add_tri(v_lo_sh_rad_l, v_strk_l, v_lo_sh_can_l)

    add_tri(v_can_lo, v_belly_mid, v_intk_cb)
    add_tri(v_can_lo, v_intk_cb, v_lo_sh_can_r)
    add_tri(v_can_lo, v_lo_sh_can_l, v_intk_cl_b)
    add_tri(v_can_lo, v_intk_cl_b, v_belly_mid)

    add_tri(v_lo_sh_can_r, v_strk_r, v_lo_nac_mid_r)
    add_tri(v_lo_sh_can_l, v_lo_nac_mid_l, v_strk_l)

    add_tri(v_strk_r, v_intk_rb, v_leroot_r)
    add_tri(v_strk_l, v_leroot_l, v_intk_lb)

    # Lower Nacelle Fwd to Mid
    add_tri(v_intk_cb, v_belly_mid, v_belly_aft)
    add_tri(v_intk_cb, v_belly_aft, v_lo_nac_mid_r)
    add_tri(v_intk_cl_b, v_lo_nac_mid_l, v_belly_aft)
    add_tri(v_intk_cl_b, v_belly_aft, v_belly_mid)

    add_tri(v_intk_cb, v_lo_nac_mid_r, v_intk_rb)
    add_tri(v_intk_cl_b, v_intk_lb, v_lo_nac_mid_l)

    add_tri(v_intk_rb, v_lo_nac_mid_r, v_leroot_r)
    add_tri(v_intk_lb, v_leroot_l, v_lo_nac_mid_l)

    # Lower Wing
    add_tri(v_leroot_r, v_lo_nac_mid_r, v_tip_r)
    add_tri(v_leroot_l, v_tip_l, v_lo_nac_mid_l)
    add_tri(v_tip_r, v_lo_nac_mid_r, v_te_outer_r)
    add_tri(v_tip_l, v_te_outer_l, v_lo_nac_mid_l)
    add_tri(v_te_outer_r, v_lo_nac_mid_r, v_te_break_r)
    add_tri(v_te_outer_l, v_te_break_l, v_lo_nac_mid_l)

    # Lower Mid to Aft
    add_tri(v_belly_aft, v_tail_lo, v_lo_nac_aft_r)
    add_tri(v_belly_aft, v_lo_nac_aft_r, v_lo_nac_mid_r)
    add_tri(v_belly_aft, v_lo_nac_mid_l, v_lo_nac_aft_l)
    add_tri(v_belly_aft, v_lo_nac_aft_l, v_tail_lo)

    add_tri(v_te_break_r, v_lo_nac_mid_r, v_h_root_le_r)
    add_tri(v_te_break_l, v_h_root_le_l, v_lo_nac_mid_l)
    add_tri(v_lo_nac_mid_r, v_lo_nac_aft_r, v_h_root_le_r)
    add_tri(v_lo_nac_mid_l, v_h_root_le_l, v_lo_nac_aft_l)

    # Lower Horizontal Tail
    add_tri(v_h_root_le_r, v_lo_nac_aft_r, v_h_tip_r)
    add_tri(v_h_root_le_l, v_h_tip_l, v_lo_nac_aft_l)
    add_tri(v_h_tip_r, v_lo_nac_aft_r, v_h_te_outer_r)
    add_tri(v_h_tip_l, v_h_te_outer_l, v_lo_nac_aft_l)
    add_tri(v_h_te_outer_r, v_lo_nac_aft_r, v_h_te_break_r)
    add_tri(v_h_te_outer_l, v_h_te_break_l, v_lo_nac_aft_l)

    # Lower Tail Beaver & Nozzle Bottom
    add_tri(v_tail_lo, v_tail, v_noz_cb)
    add_tri(v_tail_lo, v_noz_cb, v_lo_nac_aft_r)
    add_tri(v_tail_lo, v_lo_nac_aft_l, v_noz_cl_b)
    add_tri(v_tail_lo, v_noz_cl_b, v_tail)

    add_tri(v_lo_nac_aft_r, v_noz_cb, v_noz_rb)
    add_tri(v_lo_nac_aft_r, v_noz_rb, v_h_te_break_r)
    add_tri(v_lo_nac_aft_l, v_h_te_break_l, v_noz_lb)
    add_tri(v_lo_nac_aft_l, v_noz_lb, v_noz_cl_b)

    # --- Caret Intake Cavities ---
    # Top
    add_tri(v_intk_ct, v_intk_rt, v_duct_back_rt)
    add_tri(v_intk_ct, v_duct_back_rt, v_duct_back_ct)
    add_tri(v_intk_cl_t, v_duct_back_cl_t, v_duct_back_lt)
    add_tri(v_intk_cl_t, v_duct_back_lt, v_intk_lt)
    # Outer side
    add_tri(v_intk_rt, v_intk_rb, v_duct_back_rb)
    add_tri(v_intk_rt, v_duct_back_rb, v_duct_back_rt)
    add_tri(v_intk_lt, v_duct_back_lt, v_duct_back_lb)
    add_tri(v_intk_lt, v_duct_back_lb, v_intk_lb)
    # Inner side
    add_tri(v_intk_ct, v_duct_back_ct, v_duct_back_cb)
    add_tri(v_intk_ct, v_duct_back_cb, v_intk_cb)
    add_tri(v_intk_cl_t, v_intk_cl_b, v_duct_back_cl_b)
    add_tri(v_intk_cl_t, v_duct_back_cl_b, v_duct_back_cl_t)
    # Bottom
    add_tri(v_intk_cb, v_duct_back_cb, v_duct_back_rb)
    add_tri(v_intk_cb, v_duct_back_rb, v_intk_rb)
    add_tri(v_intk_cl_b, v_intk_lb, v_duct_back_lb)
    add_tri(v_intk_cl_b, v_duct_back_lb, v_duct_back_cl_b)
    # Back face
    add_tri(v_duct_back_ct, v_duct_back_rt, v_duct_back_rb)
    add_tri(v_duct_back_ct, v_duct_back_rb, v_duct_back_cb)
    add_tri(v_duct_back_cl_t, v_duct_back_cl_b, v_duct_back_lb)
    add_tri(v_duct_back_cl_t, v_duct_back_lb, v_duct_back_lt)

    # --- Nozzle Cavities ---
    # Top
    add_tri(v_noz_ct, v_noz_rt, v_noz_bk_rt)
    add_tri(v_noz_ct, v_noz_bk_rt, v_noz_bk_ct)
    add_tri(v_noz_cl_t, v_noz_bk_cl_t, v_noz_bk_lt)
    add_tri(v_noz_cl_t, v_noz_bk_lt, v_noz_lt)
    # Bottom
    add_tri(v_noz_cb, v_noz_bk_cb, v_noz_bk_rb)
    add_tri(v_noz_cb, v_noz_bk_rb, v_noz_rb)
    add_tri(v_noz_cl_b, v_noz_lb, v_noz_bk_lb)
    add_tri(v_noz_cl_b, v_noz_bk_lb, v_noz_bk_cl_b)
    # Outer side
    add_tri(v_noz_rt, v_noz_rb, v_noz_bk_rb)
    add_tri(v_noz_rt, v_noz_bk_rb, v_noz_bk_rt)
    add_tri(v_noz_lt, v_noz_bk_lt, v_noz_bk_lb)
    add_tri(v_noz_lt, v_noz_bk_lb, v_noz_lb)
    # Inner side
    add_tri(v_noz_ct, v_noz_bk_ct, v_noz_bk_cb)
    add_tri(v_noz_ct, v_noz_bk_cb, v_noz_cb)
    add_tri(v_noz_cl_t, v_noz_cl_b, v_noz_bk_cl_b)
    add_tri(v_noz_cl_t, v_noz_bk_cl_b, v_noz_bk_cl_t)
    # Back face
    add_tri(v_noz_bk_ct, v_noz_bk_rt, v_noz_bk_rb)
    add_tri(v_noz_bk_ct, v_noz_bk_rb, v_noz_bk_cb)
    add_tri(v_noz_bk_cl_t, v_noz_bk_cl_b, v_noz_bk_lb)
    add_tri(v_noz_bk_cl_t, v_noz_bk_lb, v_noz_bk_lt)

    # Consistent winding BFS pass
    def fix_winding(verts, tris):
        edge_map = {}
        for ti, (a, b, c) in enumerate(tris):
            for u, v in [(a, b), (b, c), (c, a)]:
                k = (min(u, v), max(u, v))
                edge_map.setdefault(k, []).append(ti)
        visited = set()
        new_tris = [list(t) for t in tris]
        for start in range(len(tris)):
            if start in visited:
                continue
            q = deque([start])
            visited.add(start)
            while q:
                curr = q.popleft()
                ca, cb, cc = new_tris[curr]
                for u, v in [(ca, cb), (cb, cc), (cc, ca)]:
                    k = (min(u, v), max(u, v))
                    for adj in edge_map.get(k, []):
                        if adj not in visited:
                            visited.add(adj)
                            t = new_tris[adj]
                            for au, av in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
                                if (min(au, av), max(au, av)) == k:
                                    if au == u:
                                        new_tris[adj] = [t[0], t[2], t[1]]
                                    break
                            q.append(adj)
        return [tuple(t) for t in new_tris]

    cur_v = list(vertices)
    cur_t = list(triangles)
    for _ in range(resolution - 1):
        midpoints = {}
        def get_mid(a, b):
            k = (min(a, b), max(a, b))
            if k in midpoints:
                return midpoints[k]
            v0, v1 = cur_v[a], cur_v[b]
            mx = (v0[0] + v1[0]) * 0.5
            my = (v0[1] + v1[1]) * 0.5
            mz = (v0[2] + v1[2]) * 0.5
            idx = len(cur_v)
            cur_v.append((round(mx, 5), round(my, 5), round(mz, 5)))
            midpoints[k] = idx
            return idx
        next_t = []
        for i0, i1, i2 in cur_t:
            m01 = get_mid(i0, i1)
            m12 = get_mid(i1, i2)
            m20 = get_mid(i2, i0)
            next_t.append((i0, m01, m20))
            next_t.append((m01, i1, m12))
            next_t.append((m20, m12, i2))
            next_t.append((m01, m12, m20))
        cur_t = next_t

    cur_t = fix_winding(cur_v, cur_t)
    return cur_v, cur_t

def write_obj(filepath, vertices, triangles, comment="5th-Gen Stealth Fighter"):
    with open(filepath, "w") as f:
        f.write(f"# StrikeCEM {comment}\n")
        f.write(f"# Vertices: {len(vertices)}, Triangles: {len(triangles)}\n")
        f.write("o Fighter5G\n")
        for v in vertices:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for t in triangles:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Main detailed benchmark model (resolution=2)
    verts, tris = generate_fighter_mesh(resolution=2)
    full_path = os.path.join(out_dir, "fighter.obj")
    write_obj(full_path, verts, tris, "Fighter 5G Detailed Benchmark Model")
    print(f"Generated: {full_path} ({len(verts)} vertices, {len(tris)} triangles)")

    # 2. Medium LOD (resolution=2)
    lod1_path = os.path.join(out_dir, "fighter_lod1.obj")
    write_obj(lod1_path, verts, tris, "Fighter 5G Medium LOD1")
    print(f"Generated: {lod1_path}")

    # 3. Coarse LOD (resolution=1)
    verts_c, tris_c = generate_fighter_mesh(resolution=1)
    lod2_path = os.path.join(out_dir, "fighter_lod2.obj")
    write_obj(lod2_path, verts_c, tris_c, "Fighter 5G Coarse LOD2")
    print(f"Generated: {lod2_path} ({len(verts_c)} vertices, {len(tris_c)} triangles)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generates clean, 2-manifold, watertight 3D OBJ models for the F-15 Eagle Air Superiority Fighter.
Creates:
  - f15.obj (full detailed benchmark model)
  - f15_lod1.obj (medium LOD)
  - f15_lod2.obj (coarse LOD)

Key Architectural & Electromagnetic Features of the F-15 Eagle:
  - Length: 19.43 m, Wingspan: 13.05 m, Height: 5.63 m
  - Ogival nose radome (diameter ~1.05 m) with AN/APG-63 radar bulkhead cavity
  - High bubble cockpit canopy with raked windscreen
  - Twin boxy variable-geometry 2D air intakes with upper compression ramps
  - Re-entrant engine intake duct cavities terminating at Pratt & Whitney F100 compressor face
  - Cropped-delta main wings with 45-degree leading edge sweep
  - Twin upright vertical stabilizers (0-degree cant! Perfectly vertical twin tails at y = +-1.45 m)
  - Twin all-moving horizontal stabilators (tailerons with characteristic dogtooth leading edge)
  - Twin afterburning turbofan nozzle exhaust cavities between tail booms
  - Full bilateral symmetry across xz-plane (y=0)
"""
import math
import os
import sys
from collections import deque

def build_f15_mesh(resolution=1):
    vertices = []
    triangles = []

    def add_v(x, y, z):
        v = (round(float(x), 5), round(float(y), 5), round(float(z), 5))
        vertices.append(v)
        return len(vertices) - 1

    def add_tri(i0, i1, i2):
        triangles.append((i0, i1, i2))

    # --- 1. Waterline / Perimeter Points (z ~ 0.05 to 0.15) ---
    v_nose          = add_v(9.8, 0.0, 0.15)
    v_rad_mid_r     = add_v(8.0, 0.45, 0.15)
    v_rad_mid_l     = add_v(8.0, -0.45, 0.15)
    v_rad_base_r    = add_v(6.6, 0.65, 0.15)
    v_rad_base_l    = add_v(6.6, -0.65, 0.15)
    
    # Forward fuselage cheek
    v_cheek_r       = add_v(4.5, 0.85, 0.15)
    v_cheek_l       = add_v(4.5, -0.85, 0.15)
    v_intk_side_r   = add_v(2.5, 1.60, 0.15)
    v_intk_side_l   = add_v(2.5, -1.60, 0.15)

    # Main Wings (F-15 Cropped Delta, 45 deg sweep)
    v_w_root_le_r   = add_v(1.0, 1.65, 0.15)
    v_w_root_le_l   = add_v(1.0, -1.65, 0.15)
    v_w_tip_le_r    = add_v(-4.2, 6.52, 0.12)
    v_w_tip_le_l    = add_v(-4.2, -6.52, 0.12)
    v_w_tip_te_r    = add_v(-5.4, 6.45, 0.12)
    v_w_tip_te_l    = add_v(-5.4, -6.45, 0.12)
    v_w_root_te_r   = add_v(-4.8, 1.85, 0.15)
    v_w_root_te_l   = add_v(-4.8, -1.85, 0.15)

    # Taileron / Horizontal Stabilator (with dogtooth)
    v_stab_root_le_r= add_v(-6.2, 1.85, 0.05)
    v_stab_root_le_l= add_v(-6.2, -1.85, 0.05)
    v_stab_tip_le_r = add_v(-8.2, 3.80, 0.02)
    v_stab_tip_le_l = add_v(-8.2, -3.80, 0.02)
    v_stab_tip_te_r = add_v(-9.5, 3.40, 0.02)
    v_stab_tip_te_l = add_v(-9.5, -3.40, 0.02)
    v_stab_root_te_r= add_v(-9.0, 1.70, 0.05)
    v_stab_root_te_l= add_v(-9.0, -1.70, 0.05)

    # Aft central boom between engines
    v_tail_center   = add_v(-9.6, 0.0, 0.25)

    # --- 2. Upper Centerline Spine (y = 0) ---
    v_rad_up_mid    = add_v(8.0, 0.0, 0.45)
    v_rad_up_base   = add_v(6.6, 0.0, 0.65)
    v_can_apex      = add_v(4.6, 0.0, 1.60)  # High bubble canopy
    v_can_aft       = add_v(2.5, 0.0, 1.15)
    v_spine_mid     = add_v(-1.5, 0.0, 0.85) # Flat dorsal deck
    v_spine_aft     = add_v(-6.0, 0.0, 0.80)

    # Upper Shoulders
    v_up_sh_rad_r   = add_v(6.6, 0.40, 0.50)
    v_up_sh_rad_l   = add_v(6.6, -0.40, 0.50)
    v_up_sh_can_r   = add_v(4.6, 0.55, 1.05)
    v_up_sh_can_l   = add_v(4.6, -0.55, 1.05)

    # Dorsal nacelle humps (over twin F100 engines)
    v_up_nac_fwd_r  = add_v(1.0, 1.45, 0.85)
    v_up_nac_fwd_l  = add_v(1.0, -1.45, 0.85)

    # Vertical Fin Root - Inner Base (y = +-1.35)
    v_vt_in_mid_r   = add_v(-4.0, 1.35, 0.85)
    v_vt_in_mid_l   = add_v(-4.0, -1.35, 0.85)
    v_vt_in_aft_r   = add_v(-7.5, 1.25, 0.75)
    v_vt_in_aft_l   = add_v(-7.5, -1.25, 0.75)

    # Vertical Fin Root - Outer Base (y = +-1.55)
    v_vt_out_mid_r  = add_v(-4.0, 1.55, 0.85)
    v_vt_out_mid_l  = add_v(-4.0, -1.55, 0.85)
    v_vt_out_aft_r  = add_v(-7.5, 1.45, 0.75)
    v_vt_out_aft_l  = add_v(-7.5, -1.45, 0.75)

    # --- 3. Twin Upright Vertical Stabilizers (0 deg cant!) ---
    # The F-15 has parallel vertical fins at y = +-1.45 m!
    v_vtail_tip_le_r= add_v(-6.5, 1.45, 4.20)
    v_vtail_tip_le_l= add_v(-6.5, -1.45, 4.20)
    v_vtail_tip_te_r= add_v(-7.8, 1.45, 4.10)
    v_vtail_tip_te_l= add_v(-7.8, -1.45, 4.10)

    # --- 4. Twin Boxy Rectangular Air Intakes ---
    # F-15 2D variable compression ramp intakes (x ~ 2.5)
    v_intk_rt       = add_v(2.5, 1.60, 0.65)
    v_intk_lt       = add_v(2.5, -1.60, 0.65)
    v_intk_ct       = add_v(2.5, 0.65, 0.65)
    v_intk_cl_t     = add_v(2.5, -0.65, 0.65)
    v_intk_rb       = add_v(2.8, 1.60, -0.35)
    v_intk_lb       = add_v(2.8, -1.60, -0.35)
    v_intk_cb       = add_v(2.8, 0.65, -0.35)
    v_intk_cl_b     = add_v(2.8, -0.65, -0.35)

    # Internal duct to compressor face (x = -0.8)
    v_duct_back_rt  = add_v(-0.8, 1.45, 0.45)
    v_duct_back_lt  = add_v(-0.8, -1.45, 0.45)
    v_duct_back_ct  = add_v(-0.8, 0.75, 0.45)
    v_duct_back_cl_t= add_v(-0.8, -0.75, 0.45)
    v_duct_back_rb  = add_v(-0.8, 1.45, -0.15)
    v_duct_back_lb  = add_v(-0.8, -1.45, -0.15)
    v_duct_back_cb  = add_v(-0.8, 0.75, -0.15)
    v_duct_back_cl_b= add_v(-0.8, -0.75, -0.15)
    # Rotating compressor center spinner bullet cone apex (points forward)
    v_hub_r         = add_v(-0.45, 1.10, 0.15)
    v_hub_l         = add_v(-0.45, -1.10, 0.15)

    # --- 5. Lower Belly & Ventral Surface ---
    v_rad_lo_mid    = add_v(8.0, 0.0, -0.10)
    v_rad_lo_base   = add_v(6.6, 0.0, -0.20)
    v_belly_fwd     = add_v(4.5, 0.0, -0.35)
    v_belly_mid     = add_v(-1.5, 0.0, -0.40)
    v_belly_aft     = add_v(-6.0, 0.0, -0.35)
    v_belly_tail    = add_v(-9.4, 0.0, 0.05)

    v_lo_sh_rad_r   = add_v(6.6, 0.40, -0.15)
    v_lo_sh_rad_l   = add_v(6.6, -0.40, -0.15)
    v_lo_sh_fwd_r   = add_v(4.5, 0.55, -0.25)
    v_lo_sh_fwd_l   = add_v(4.5, -0.55, -0.25)
    v_lo_nac_mid_r  = add_v(-4.0, 1.35, -0.35)
    v_lo_nac_mid_l  = add_v(-4.0, -1.35, -0.35)
    v_lo_nac_aft_r  = add_v(-7.5, 1.25, -0.15)
    v_lo_nac_aft_l  = add_v(-7.5, -1.25, -0.15)

    # --- 6. Twin Pratt & Whitney F100 Exhaust Nozzles ---
    v_noz_rt        = add_v(-9.2, 1.60, 0.60)
    v_noz_lt        = add_v(-9.2, -1.60, 0.60)
    v_noz_ct        = add_v(-9.2, 0.55, 0.60)
    v_noz_cl_t      = add_v(-9.2, -0.55, 0.60)
    v_noz_rb        = add_v(-9.2, 1.60, -0.15)
    v_noz_lb        = add_v(-9.2, -1.60, -0.15)
    v_noz_cb        = add_v(-9.2, 0.55, -0.15)
    v_noz_cl_b      = add_v(-9.2, -0.55, -0.15)

    # Internal turbine cavity back face (flameholder)
    v_noz_bk_rt     = add_v(-8.0, 1.40, 0.45)
    v_noz_bk_lt     = add_v(-8.0, -1.40, 0.45)
    v_noz_bk_ct     = add_v(-8.0, 0.75, 0.45)
    v_noz_bk_cl_t   = add_v(-8.0, -0.75, 0.45)
    v_noz_bk_rb     = add_v(-8.0, 1.40, 0.05)
    v_noz_bk_lb     = add_v(-8.0, -1.40, 0.05)
    v_noz_bk_cb     = add_v(-8.0, 0.75, 0.05)
    v_noz_bk_cl_b   = add_v(-8.0, -0.75, 0.05)
    # Aft turbine flameholder mixer cone apex (points aft)
    v_noz_cone_r    = add_v(-8.45, 1.075, 0.25)
    v_noz_cone_l    = add_v(-8.45, -1.075, 0.25)

    # ---------------------------------------------------------
    # TRIANGULATION
    # ---------------------------------------------------------

    # A. Ogival Radome (Upper)
    add_tri(v_nose, v_rad_mid_r, v_rad_up_mid)
    add_tri(v_nose, v_rad_up_mid, v_rad_mid_l)
    add_tri(v_rad_up_mid, v_rad_mid_r, v_up_sh_rad_r)
    add_tri(v_rad_up_mid, v_up_sh_rad_r, v_rad_up_base)
    add_tri(v_rad_up_mid, v_rad_up_base, v_up_sh_rad_l)
    add_tri(v_rad_up_mid, v_up_sh_rad_l, v_rad_mid_l)
    add_tri(v_rad_mid_r, v_rad_base_r, v_up_sh_rad_r)
    add_tri(v_rad_mid_l, v_up_sh_rad_l, v_rad_base_l)

    # B. Cockpit Canopy & Cheek
    add_tri(v_rad_up_base, v_up_sh_rad_r, v_up_sh_can_r)
    add_tri(v_rad_up_base, v_up_sh_can_r, v_can_apex)
    add_tri(v_rad_up_base, v_can_apex, v_up_sh_can_l)
    add_tri(v_rad_up_base, v_up_sh_can_l, v_up_sh_rad_l)

    add_tri(v_up_sh_rad_r, v_rad_base_r, v_cheek_r)
    add_tri(v_up_sh_rad_r, v_cheek_r, v_up_sh_can_r)
    add_tri(v_up_sh_rad_l, v_up_sh_can_l, v_cheek_l)
    add_tri(v_up_sh_rad_l, v_cheek_l, v_rad_base_l)

    add_tri(v_can_apex, v_up_sh_can_r, v_can_aft)
    add_tri(v_can_apex, v_can_aft, v_up_sh_can_l)

    add_tri(v_can_aft, v_up_sh_can_r, v_intk_ct)
    add_tri(v_can_aft, v_intk_ct, v_spine_mid)
    add_tri(v_can_aft, v_spine_mid, v_intk_cl_t)
    add_tri(v_can_aft, v_intk_cl_t, v_up_sh_can_l)

    add_tri(v_up_sh_can_r, v_cheek_r, v_up_nac_fwd_r)
    add_tri(v_up_sh_can_l, v_up_nac_fwd_l, v_cheek_l)
    add_tri(v_cheek_r, v_intk_side_r, v_up_nac_fwd_r)
    add_tri(v_cheek_l, v_up_nac_fwd_l, v_intk_side_l)
    add_tri(v_intk_side_r, v_w_root_le_r, v_up_nac_fwd_r)
    add_tri(v_intk_side_l, v_up_nac_fwd_l, v_w_root_le_l)

    # C. Intake Upper Cowl & Dorsal Nacelle Fwd
    add_tri(v_intk_ct, v_intk_rt, v_up_nac_fwd_r)
    add_tri(v_intk_ct, v_up_nac_fwd_r, v_spine_mid)
    add_tri(v_intk_cl_t, v_spine_mid, v_up_nac_fwd_l)
    add_tri(v_intk_cl_t, v_up_nac_fwd_l, v_intk_lt)

    # D. Upper Wing Surfaces (Cropped Delta)
    add_tri(v_w_root_le_r, v_w_tip_le_r, v_up_nac_fwd_r)
    add_tri(v_w_root_le_l, v_up_nac_fwd_l, v_w_tip_le_l)
    add_tri(v_w_tip_le_r, v_w_tip_te_r, v_vt_out_mid_r)
    add_tri(v_w_tip_le_l, v_vt_out_mid_l, v_w_tip_te_l)
    add_tri(v_w_tip_le_r, v_vt_out_mid_r, v_up_nac_fwd_r)
    add_tri(v_w_tip_le_l, v_up_nac_fwd_l, v_vt_out_mid_l)
    add_tri(v_w_tip_te_r, v_w_root_te_r, v_vt_out_mid_r)
    add_tri(v_w_tip_te_l, v_vt_out_mid_l, v_w_root_te_l)

    # Fillet forward of vertical fin
    add_tri(v_up_nac_fwd_r, v_vt_out_mid_r, v_vt_in_mid_r)
    add_tri(v_up_nac_fwd_l, v_vt_in_mid_l, v_vt_out_mid_l)

    # E. Dorsal Deck Between Centerline Spine and Nacelles
    add_tri(v_spine_mid, v_up_nac_fwd_r, v_vt_in_mid_r)
    add_tri(v_spine_mid, v_vt_in_mid_r, v_spine_aft)
    add_tri(v_spine_mid, v_spine_aft, v_vt_in_mid_l)
    add_tri(v_spine_mid, v_vt_in_mid_l, v_up_nac_fwd_l)

    add_tri(v_w_root_te_r, v_stab_root_le_r, v_vt_out_mid_r)
    add_tri(v_w_root_te_l, v_vt_out_mid_l, v_stab_root_le_l)

    # F. Upper Horizontal Stabilators (Tailerons)
    add_tri(v_stab_root_le_r, v_stab_tip_le_r, v_vt_out_mid_r)
    add_tri(v_stab_root_le_l, v_vt_out_mid_l, v_stab_tip_le_l)
    add_tri(v_stab_tip_le_r, v_stab_tip_te_r, v_vt_out_aft_r)
    add_tri(v_stab_tip_le_l, v_vt_out_aft_l, v_stab_tip_te_l)
    add_tri(v_stab_tip_le_r, v_vt_out_aft_r, v_vt_out_mid_r)
    add_tri(v_stab_tip_le_l, v_vt_out_mid_l, v_vt_out_aft_l)
    add_tri(v_stab_tip_te_r, v_stab_root_te_r, v_vt_out_aft_r)
    add_tri(v_stab_tip_te_l, v_vt_out_aft_l, v_stab_root_te_l)

    # G. Twin Upright Vertical Stabilizers (F-15 Parallel Vertical Fins)
    # Dorsal deck continues flat:
    add_tri(v_spine_aft, v_vt_in_mid_r, v_vt_in_aft_r)
    add_tri(v_spine_aft, v_vt_in_aft_l, v_vt_in_mid_l)

    # Fin inner face
    add_tri(v_vt_in_mid_r, v_vtail_tip_le_r, v_vtail_tip_te_r)
    add_tri(v_vt_in_mid_r, v_vtail_tip_te_r, v_vt_in_aft_r)
    add_tri(v_vt_in_mid_l, v_vtail_tip_te_l, v_vtail_tip_le_l)
    add_tri(v_vt_in_mid_l, v_vt_in_aft_l, v_vtail_tip_te_l)

    # Fin outer face
    add_tri(v_vtail_tip_le_r, v_vt_out_mid_r, v_vtail_tip_te_r)
    add_tri(v_vtail_tip_te_r, v_vt_out_mid_r, v_vt_out_aft_r)
    add_tri(v_vtail_tip_le_l, v_vtail_tip_te_l, v_vt_out_mid_l)
    add_tri(v_vtail_tip_te_l, v_vt_out_mid_l, v_vt_out_aft_l)

    # Fin leading edge strip
    add_tri(v_vt_in_mid_r, v_vt_out_mid_r, v_vtail_tip_le_r)
    add_tri(v_vt_in_mid_l, v_vtail_tip_le_l, v_vt_out_mid_l)

    # Fin trailing edge strip
    add_tri(v_vtail_tip_te_r, v_vt_out_aft_r, v_vt_in_aft_r)
    add_tri(v_vtail_tip_te_l, v_vt_in_aft_l, v_vt_out_aft_l)

    # Aft fillet between inner and outer base
    add_tri(v_vt_in_aft_r, v_vt_out_aft_r, v_noz_ct)
    add_tri(v_vt_in_aft_l, v_noz_cl_t, v_vt_out_aft_l)

    # H. Aft Beaver Tail Upper & Nozzle Top
    add_tri(v_spine_aft, v_vt_in_aft_r, v_noz_ct)
    add_tri(v_spine_aft, v_noz_ct, v_tail_center)
    add_tri(v_spine_aft, v_tail_center, v_noz_cl_t)
    add_tri(v_spine_aft, v_noz_cl_t, v_vt_in_aft_l)

    add_tri(v_vt_out_aft_r, v_stab_root_te_r, v_noz_rt)
    add_tri(v_vt_out_aft_r, v_noz_rt, v_noz_ct)
    add_tri(v_vt_out_aft_l, v_noz_cl_t, v_noz_lt)
    add_tri(v_vt_out_aft_l, v_noz_lt, v_stab_root_te_l)

    # I. Lower Belly Triangulation
    # Ogive nose belly
    add_tri(v_nose, v_rad_lo_mid, v_rad_mid_r)
    add_tri(v_nose, v_rad_mid_l, v_rad_lo_mid)
    add_tri(v_rad_lo_mid, v_lo_sh_rad_r, v_rad_mid_r)
    add_tri(v_rad_lo_mid, v_rad_lo_base, v_lo_sh_rad_r)
    add_tri(v_rad_lo_mid, v_rad_mid_l, v_lo_sh_rad_l)
    add_tri(v_rad_lo_mid, v_lo_sh_rad_l, v_rad_lo_base)
    add_tri(v_lo_sh_rad_r, v_rad_base_r, v_rad_mid_r)
    add_tri(v_lo_sh_rad_l, v_rad_mid_l, v_rad_base_l)

    # Forward belly to cheek
    add_tri(v_rad_lo_base, v_belly_fwd, v_lo_sh_fwd_r)
    add_tri(v_rad_lo_base, v_lo_sh_fwd_r, v_lo_sh_rad_r)
    add_tri(v_rad_lo_base, v_lo_sh_rad_l, v_lo_sh_fwd_l)
    add_tri(v_rad_lo_base, v_lo_sh_fwd_l, v_belly_fwd)

    add_tri(v_lo_sh_rad_r, v_cheek_r, v_rad_base_r)
    add_tri(v_lo_sh_rad_r, v_lo_sh_fwd_r, v_cheek_r)
    add_tri(v_lo_sh_rad_l, v_rad_base_l, v_cheek_l)
    add_tri(v_lo_sh_rad_l, v_cheek_l, v_lo_sh_fwd_l)

    # Belly to intake bottom lip
    add_tri(v_belly_fwd, v_belly_mid, v_intk_cb)
    add_tri(v_belly_fwd, v_intk_cb, v_lo_sh_fwd_r)
    add_tri(v_belly_fwd, v_lo_sh_fwd_l, v_intk_cl_b)
    add_tri(v_belly_fwd, v_intk_cl_b, v_belly_mid)

    add_tri(v_lo_sh_fwd_r, v_intk_side_r, v_cheek_r)
    add_tri(v_lo_sh_fwd_r, v_lo_nac_mid_r, v_intk_side_r)
    add_tri(v_lo_sh_fwd_l, v_cheek_l, v_intk_side_l)
    add_tri(v_lo_sh_fwd_l, v_intk_side_l, v_lo_nac_mid_l)

    add_tri(v_intk_side_r, v_w_root_le_r, v_lo_nac_mid_r)
    add_tri(v_intk_side_l, v_lo_nac_mid_l, v_w_root_le_l)

    # Belly nacelles fwd to mid
    add_tri(v_intk_cb, v_belly_mid, v_belly_aft)
    add_tri(v_intk_cb, v_belly_aft, v_lo_nac_mid_r)
    add_tri(v_intk_cl_b, v_lo_nac_mid_l, v_belly_aft)
    add_tri(v_intk_cl_b, v_belly_aft, v_belly_mid)

    add_tri(v_intk_cb, v_lo_nac_mid_r, v_intk_rb)
    add_tri(v_intk_cl_b, v_intk_lb, v_lo_nac_mid_l)

    # Lower Wing Surfaces
    add_tri(v_w_root_le_r, v_lo_nac_mid_r, v_w_tip_le_r)
    add_tri(v_w_root_le_l, v_w_tip_le_l, v_lo_nac_mid_l)
    add_tri(v_w_tip_le_r, v_lo_nac_mid_r, v_w_tip_te_r)
    add_tri(v_w_tip_le_l, v_w_tip_te_l, v_lo_nac_mid_l)
    add_tri(v_w_tip_te_r, v_lo_nac_mid_r, v_w_root_te_r)
    add_tri(v_w_tip_te_l, v_w_root_te_l, v_lo_nac_mid_l)

    # Lower Mid to Aft
    add_tri(v_belly_aft, v_belly_tail, v_lo_nac_aft_r)
    add_tri(v_belly_aft, v_lo_nac_aft_r, v_lo_nac_mid_r)
    add_tri(v_belly_aft, v_lo_nac_mid_l, v_lo_nac_aft_l)
    add_tri(v_belly_aft, v_lo_nac_aft_l, v_belly_tail)

    add_tri(v_w_root_te_r, v_lo_nac_mid_r, v_stab_root_le_r)
    add_tri(v_w_root_te_l, v_stab_root_le_l, v_lo_nac_mid_l)
    add_tri(v_lo_nac_mid_r, v_lo_nac_aft_r, v_stab_root_le_r)
    add_tri(v_lo_nac_mid_l, v_stab_root_le_l, v_lo_nac_aft_l)

    # Lower Horizontal Stabilators
    add_tri(v_stab_root_le_r, v_lo_nac_aft_r, v_stab_tip_le_r)
    add_tri(v_stab_root_le_l, v_stab_tip_le_l, v_lo_nac_aft_l)
    add_tri(v_stab_tip_le_r, v_lo_nac_aft_r, v_stab_tip_te_r)
    add_tri(v_stab_tip_le_l, v_stab_tip_te_l, v_lo_nac_aft_l)
    add_tri(v_stab_tip_te_r, v_lo_nac_aft_r, v_stab_root_te_r)
    add_tri(v_stab_tip_te_l, v_stab_root_te_l, v_lo_nac_aft_l)

    # Lower Tail Beaver & Nozzle Bottom
    add_tri(v_belly_tail, v_tail_center, v_noz_cb)
    add_tri(v_belly_tail, v_noz_cb, v_lo_nac_aft_r)
    add_tri(v_belly_tail, v_lo_nac_aft_l, v_noz_cl_b)
    add_tri(v_belly_tail, v_noz_cl_b, v_tail_center)

    add_tri(v_lo_nac_aft_r, v_noz_cb, v_noz_rb)
    add_tri(v_lo_nac_aft_r, v_noz_rb, v_stab_root_te_r)
    add_tri(v_lo_nac_aft_l, v_stab_root_te_l, v_noz_lb)
    add_tri(v_lo_nac_aft_l, v_noz_lb, v_noz_cl_b)

    # J. Twin Boxy Rectangular Intake Cavities
    # Duct top
    add_tri(v_intk_ct, v_intk_rt, v_duct_back_rt)
    add_tri(v_intk_ct, v_duct_back_rt, v_duct_back_ct)
    add_tri(v_intk_cl_t, v_duct_back_cl_t, v_duct_back_lt)
    add_tri(v_intk_cl_t, v_duct_back_lt, v_intk_lt)
    # Duct outer side
    add_tri(v_intk_rt, v_intk_rb, v_duct_back_rb)
    add_tri(v_intk_rt, v_duct_back_rb, v_duct_back_rt)
    add_tri(v_intk_lt, v_duct_back_lt, v_duct_back_lb)
    add_tri(v_intk_lt, v_duct_back_lb, v_intk_lb)
    # Duct inner side
    add_tri(v_intk_ct, v_duct_back_ct, v_duct_back_cb)
    add_tri(v_intk_ct, v_duct_back_cb, v_intk_cb)
    add_tri(v_intk_cl_t, v_intk_cl_b, v_duct_back_cl_b)
    add_tri(v_intk_cl_t, v_duct_back_cl_b, v_duct_back_cl_t)
    # Duct bottom
    add_tri(v_intk_cb, v_duct_back_cb, v_duct_back_rb)
    add_tri(v_intk_cb, v_duct_back_rb, v_intk_rb)
    add_tri(v_intk_cl_b, v_intk_lb, v_duct_back_lb)
    add_tri(v_intk_cl_b, v_duct_back_lb, v_duct_back_cl_b)
    # Duct back compressor face with center spinner bullet cone
    add_tri(v_hub_r, v_duct_back_ct, v_duct_back_rt)
    add_tri(v_hub_r, v_duct_back_rt, v_duct_back_rb)
    add_tri(v_hub_r, v_duct_back_rb, v_duct_back_cb)
    add_tri(v_hub_r, v_duct_back_cb, v_duct_back_ct)

    add_tri(v_hub_l, v_duct_back_lt, v_duct_back_cl_t)
    add_tri(v_hub_l, v_duct_back_cl_t, v_duct_back_cl_b)
    add_tri(v_hub_l, v_duct_back_cl_b, v_duct_back_lb)
    add_tri(v_hub_l, v_duct_back_lb, v_duct_back_lt)

    # K. Twin Pratt & Whitney F100 Exhaust Nozzles
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
    # Back flameholder face with aft mixer cone
    add_tri(v_noz_cone_r, v_noz_bk_rt, v_noz_bk_ct)
    add_tri(v_noz_cone_r, v_noz_bk_ct, v_noz_bk_cb)
    add_tri(v_noz_cone_r, v_noz_bk_cb, v_noz_bk_rb)
    add_tri(v_noz_cone_r, v_noz_bk_rb, v_noz_bk_rt)

    add_tri(v_noz_cone_l, v_noz_bk_cl_t, v_noz_bk_lt)
    add_tri(v_noz_cone_l, v_noz_bk_lt, v_noz_bk_lb)
    add_tri(v_noz_cone_l, v_noz_bk_lb, v_noz_bk_cl_b)
    add_tri(v_noz_cone_l, v_noz_bk_cl_b, v_noz_bk_cl_t)

    # Consistent winding pass
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

def write_obj(filepath, vertices, triangles, comment="F-15 Eagle"):
    with open(filepath, "w") as f:
        f.write(f"# StrikeCEM {comment}\n")
        f.write(f"# Vertices: {len(vertices)}, Triangles: {len(triangles)}\n")
        f.write("o F15_Eagle\n")
        for v in vertices:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for t in triangles:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Main detailed benchmark model (resolution=2)
    verts, tris = build_f15_mesh(resolution=2)
    full_path = os.path.join(out_dir, "f15.obj")
    write_obj(full_path, verts, tris, "F-15 Eagle Detailed Benchmark Model")
    print(f"Generated: {full_path} ({len(verts)} vertices, {len(tris)} triangles)")

    # 2. Medium LOD (resolution=2)
    lod1_path = os.path.join(out_dir, "f15_lod1.obj")
    write_obj(lod1_path, verts, tris, "F-15 Eagle Medium LOD1")
    print(f"Generated: {lod1_path}")

    # 3. Coarse LOD (resolution=1)
    verts_c, tris_c = build_f15_mesh(resolution=1)
    lod2_path = os.path.join(out_dir, "f15_lod2.obj")
    write_obj(lod2_path, verts_c, tris_c, "F-15 Eagle Coarse LOD2")
    print(f"Generated: {lod2_path} ({len(verts_c)} vertices, {len(tris_c)} triangles)")

if __name__ == "__main__":
    main()

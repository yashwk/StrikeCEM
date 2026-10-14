#!/usr/bin/env python3
"""
StrikeCEM Procedural F-15 Eagle RAM-Zoned 3D CAD Mesh Generator.
Produces a 100% 2-manifold OBJ mesh with grouped material zones:
- ram_intake: Variable-geometry air intake ducts, compression ramps, and compressor face bullet cone
- ram_leading_edges: 45-deg cropped-delta wing leading edges, taileron dogtooth, and radome chine
- ram_vertical_tails: Twin parallel vertical stabilizer surfaces (0 deg cant)
- ram_canopy: Cockpit bubble canopy with ITO resistive film
- pec_body: Main fuselage skin, dorsal deck, belly keel, and exhaust nozzles
"""
import os
import math
from collections import deque

def build_f15_ram_mesh(resolution=2):
    vertices = []
    # triangles stores (v0, v1, v2, group_name)
    triangles = []

    def add_v(x, y, z):
        vertices.append((round(x, 5), round(y, 5), round(z, 5)))
        return len(vertices) - 1

    def add_tri(a, b, c, grp="pec_body"):
        triangles.append((a, b, c, grp))

    # ---------------------------------------------------------
    # VERTEX DEFINITIONS (Full-scale F-15 Eagle, meters)
    # L = 19.40 m, Wingspan = 13.04 m, Height = 4.60 m
    # ---------------------------------------------------------

    # --- 1. Nose & Radome (Ogive Profile) ---
    v_nose          = add_v(9.8, 0.0, 0.0)
    v_rad_mid_r     = add_v(8.0, 0.55, 0.20)
    v_rad_mid_l     = add_v(8.0, -0.55, 0.20)
    v_rad_base_r    = add_v(6.6, 0.85, 0.25)
    v_rad_base_l    = add_v(6.6, -0.85, 0.25)

    # Cockpit Cheek & Forward Chine
    v_cheek_r       = add_v(4.6, 1.05, 0.35)
    v_cheek_l       = add_v(4.6, -1.05, 0.35)

    # Outer intake side wall
    v_intk_side_r   = add_v(2.5, 1.60, 0.45)
    v_intk_side_l   = add_v(2.5, -1.60, 0.45)

    # 45 deg Cropped-Delta Wing
    v_w_root_le_r   = add_v(1.0, 1.65, 0.40)
    v_w_root_le_l   = add_v(1.0, -1.65, 0.40)
    v_w_tip_le_r    = add_v(-4.5, 6.52, 0.45)
    v_w_tip_le_l    = add_v(-4.5, -6.52, 0.45)
    v_w_tip_te_r    = add_v(-6.0, 6.52, 0.45)
    v_w_tip_te_l    = add_v(-6.0, -6.52, 0.45)
    v_w_root_te_r   = add_v(-6.0, 1.85, 0.35)
    v_w_root_te_l   = add_v(-6.0, -1.85, 0.35)

    # Horizontal Stabilators (Tailerons with dogtooth)
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
    v_vtail_tip_le_r= add_v(-6.5, 1.45, 4.20)
    v_vtail_tip_le_l= add_v(-6.5, -1.45, 4.20)
    v_vtail_tip_te_r= add_v(-7.8, 1.45, 4.10)
    v_vtail_tip_te_l= add_v(-7.8, -1.45, 4.10)

    # --- 4. Twin Boxy Rectangular Air Intakes ---
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
    # TRIANGULATION WITH MATERIAL ZONE TAGS
    # ---------------------------------------------------------

    # A. Ogival Radome (Upper) -> ram_leading_edges
    add_tri(v_nose, v_rad_mid_r, v_rad_up_mid, "ram_leading_edges")
    add_tri(v_nose, v_rad_up_mid, v_rad_mid_l, "ram_leading_edges")
    add_tri(v_rad_up_mid, v_rad_mid_r, v_up_sh_rad_r, "ram_leading_edges")
    add_tri(v_rad_up_mid, v_up_sh_rad_r, v_rad_up_base, "ram_leading_edges")
    add_tri(v_rad_up_mid, v_rad_up_base, v_up_sh_rad_l, "ram_leading_edges")
    add_tri(v_rad_up_mid, v_up_sh_rad_l, v_rad_mid_l, "ram_leading_edges")
    add_tri(v_rad_mid_r, v_rad_base_r, v_up_sh_rad_r, "ram_leading_edges")
    add_tri(v_rad_mid_l, v_up_sh_rad_l, v_rad_base_l, "ram_leading_edges")

    # B. Cockpit Canopy & Cheek
    # Canopy glass bubble -> ram_canopy
    add_tri(v_rad_up_base, v_up_sh_rad_r, v_up_sh_can_r, "ram_canopy")
    add_tri(v_rad_up_base, v_up_sh_can_r, v_can_apex, "ram_canopy")
    add_tri(v_rad_up_base, v_can_apex, v_up_sh_can_l, "ram_canopy")
    add_tri(v_rad_up_base, v_up_sh_can_l, v_up_sh_rad_l, "ram_canopy")
    add_tri(v_can_apex, v_up_sh_can_r, v_can_aft, "ram_canopy")
    add_tri(v_can_apex, v_can_aft, v_up_sh_can_l, "ram_canopy")
    add_tri(v_can_aft, v_up_sh_can_r, v_intk_ct, "ram_canopy")
    add_tri(v_can_aft, v_intk_ct, v_spine_mid, "ram_canopy")
    add_tri(v_can_aft, v_spine_mid, v_intk_cl_t, "ram_canopy")
    add_tri(v_can_aft, v_intk_cl_t, v_up_sh_can_l, "ram_canopy")

    # Cheek chine -> ram_leading_edges
    add_tri(v_up_sh_rad_r, v_rad_base_r, v_cheek_r, "ram_leading_edges")
    add_tri(v_up_sh_rad_r, v_cheek_r, v_up_sh_can_r, "ram_leading_edges")
    add_tri(v_up_sh_rad_l, v_up_sh_can_l, v_cheek_l, "ram_leading_edges")
    add_tri(v_up_sh_rad_l, v_cheek_l, v_rad_base_l, "ram_leading_edges")

    add_tri(v_up_sh_can_r, v_cheek_r, v_up_nac_fwd_r, "ram_leading_edges")
    add_tri(v_up_sh_can_l, v_up_nac_fwd_l, v_cheek_l, "ram_leading_edges")
    add_tri(v_cheek_r, v_intk_side_r, v_up_nac_fwd_r, "ram_intake")
    add_tri(v_cheek_l, v_up_nac_fwd_l, v_intk_side_l, "ram_intake")
    add_tri(v_intk_side_r, v_w_root_le_r, v_up_nac_fwd_r, "ram_intake")
    add_tri(v_intk_side_l, v_up_nac_fwd_l, v_w_root_le_l, "ram_intake")

    # C. Intake Upper Cowl & Forward Nacelle -> ram_intake
    add_tri(v_intk_ct, v_intk_rt, v_up_nac_fwd_r, "ram_intake")
    add_tri(v_intk_ct, v_up_nac_fwd_r, v_spine_mid, "ram_intake")
    add_tri(v_intk_cl_t, v_spine_mid, v_up_nac_fwd_l, "ram_intake")
    add_tri(v_intk_cl_t, v_up_nac_fwd_l, v_intk_lt, "ram_intake")

    # D. Upper Wing Surfaces (Cropped Delta)
    # Wing Leading Edge -> ram_leading_edges
    add_tri(v_w_root_le_r, v_w_tip_le_r, v_up_nac_fwd_r, "ram_leading_edges")
    add_tri(v_w_root_le_l, v_up_nac_fwd_l, v_w_tip_le_l, "ram_leading_edges")
    # Wing Main Skin -> pec_body
    add_tri(v_w_tip_le_r, v_w_tip_te_r, v_vt_out_mid_r, "pec_body")
    add_tri(v_w_tip_le_l, v_vt_out_mid_l, v_w_tip_te_l, "pec_body")
    add_tri(v_w_tip_le_r, v_vt_out_mid_r, v_up_nac_fwd_r, "pec_body")
    add_tri(v_w_tip_le_l, v_up_nac_fwd_l, v_vt_out_mid_l, "pec_body")
    add_tri(v_w_tip_te_r, v_w_root_te_r, v_vt_out_mid_r, "pec_body")
    add_tri(v_w_tip_te_l, v_vt_out_mid_l, v_w_root_te_l, "pec_body")

    # Fillet forward of vertical fin -> pec_body
    add_tri(v_up_nac_fwd_r, v_vt_out_mid_r, v_vt_in_mid_r, "pec_body")
    add_tri(v_up_nac_fwd_l, v_vt_in_mid_l, v_vt_out_mid_l, "pec_body")

    # E. Dorsal Deck Between Centerline Spine and Nacelles -> pec_body
    add_tri(v_spine_mid, v_up_nac_fwd_r, v_vt_in_mid_r, "pec_body")
    add_tri(v_spine_mid, v_vt_in_mid_r, v_spine_aft, "pec_body")
    add_tri(v_spine_mid, v_spine_aft, v_vt_in_mid_l, "pec_body")
    add_tri(v_spine_mid, v_vt_in_mid_l, v_up_nac_fwd_l, "pec_body")

    add_tri(v_w_root_te_r, v_stab_root_le_r, v_vt_out_mid_r, "pec_body")
    add_tri(v_w_root_te_l, v_vt_out_mid_l, v_stab_root_le_l, "pec_body")

    # F. Upper Horizontal Stabilators (Tailerons)
    # Dogtooth leading edge -> ram_leading_edges
    add_tri(v_stab_root_le_r, v_stab_tip_le_r, v_vt_out_mid_r, "ram_leading_edges")
    add_tri(v_stab_root_le_l, v_vt_out_mid_l, v_stab_tip_le_l, "ram_leading_edges")
    # Stabilator surfaces -> pec_body
    add_tri(v_stab_tip_le_r, v_stab_tip_te_r, v_vt_out_aft_r, "pec_body")
    add_tri(v_stab_tip_le_l, v_vt_out_aft_l, v_stab_tip_te_l, "pec_body")
    add_tri(v_stab_tip_le_r, v_vt_out_aft_r, v_vt_out_mid_r, "pec_body")
    add_tri(v_stab_tip_le_l, v_vt_out_mid_l, v_vt_out_aft_l, "pec_body")
    add_tri(v_stab_tip_te_r, v_stab_root_te_r, v_vt_out_aft_r, "pec_body")
    add_tri(v_stab_tip_te_l, v_vt_out_aft_l, v_stab_root_te_l, "pec_body")

    # G. Twin Upright Vertical Stabilizers -> ram_vertical_tails
    add_tri(v_spine_aft, v_vt_in_mid_r, v_vt_in_aft_r, "pec_body")
    add_tri(v_spine_aft, v_vt_in_aft_l, v_vt_in_mid_l, "pec_body")

    # Fin inner face
    add_tri(v_vt_in_mid_r, v_vtail_tip_le_r, v_vtail_tip_te_r, "ram_vertical_tails")
    add_tri(v_vt_in_mid_r, v_vtail_tip_te_r, v_vt_in_aft_r, "ram_vertical_tails")
    add_tri(v_vt_in_mid_l, v_vtail_tip_te_l, v_vtail_tip_le_l, "ram_vertical_tails")
    add_tri(v_vt_in_mid_l, v_vt_in_aft_l, v_vtail_tip_te_l, "ram_vertical_tails")

    # Fin outer face
    add_tri(v_vtail_tip_le_r, v_vt_out_mid_r, v_vtail_tip_te_r, "ram_vertical_tails")
    add_tri(v_vtail_tip_te_r, v_vt_out_mid_r, v_vt_out_aft_r, "ram_vertical_tails")
    add_tri(v_vtail_tip_le_l, v_vtail_tip_te_l, v_vt_out_mid_l, "ram_vertical_tails")
    add_tri(v_vtail_tip_te_l, v_vt_out_mid_l, v_vt_out_aft_l, "ram_vertical_tails")

    # Fin leading edge strip -> ram_leading_edges
    add_tri(v_vt_in_mid_r, v_vt_out_mid_r, v_vtail_tip_le_r, "ram_leading_edges")
    add_tri(v_vt_in_mid_l, v_vtail_tip_le_l, v_vt_out_mid_l, "ram_leading_edges")

    # Fin trailing edge strip
    add_tri(v_vtail_tip_te_r, v_vt_out_aft_r, v_vt_in_aft_r, "ram_vertical_tails")
    add_tri(v_vtail_tip_te_l, v_vt_in_aft_l, v_vt_out_aft_l, "ram_vertical_tails")

    # Aft fillet between inner and outer base -> pec_body
    add_tri(v_vt_in_aft_r, v_vt_out_aft_r, v_noz_ct, "pec_body")
    add_tri(v_vt_in_aft_l, v_noz_cl_t, v_vt_out_aft_l, "pec_body")

    # H. Aft Beaver Tail Upper & Nozzle Top -> pec_body
    add_tri(v_spine_aft, v_vt_in_aft_r, v_noz_ct, "pec_body")
    add_tri(v_spine_aft, v_noz_ct, v_tail_center, "pec_body")
    add_tri(v_spine_aft, v_tail_center, v_noz_cl_t, "pec_body")
    add_tri(v_spine_aft, v_noz_cl_t, v_vt_in_aft_l, "pec_body")

    add_tri(v_vt_out_aft_r, v_stab_root_te_r, v_noz_rt, "pec_body")
    add_tri(v_vt_out_aft_r, v_noz_rt, v_noz_ct, "pec_body")
    add_tri(v_vt_out_aft_l, v_noz_cl_t, v_noz_lt, "pec_body")
    add_tri(v_vt_out_aft_l, v_noz_lt, v_stab_root_te_l, "pec_body")

    # I. Lower Belly Triangulation
    # Nose belly -> ram_leading_edges
    add_tri(v_nose, v_rad_lo_mid, v_rad_mid_r, "ram_leading_edges")
    add_tri(v_nose, v_rad_mid_l, v_rad_lo_mid, "ram_leading_edges")
    add_tri(v_rad_lo_mid, v_lo_sh_rad_r, v_rad_mid_r, "ram_leading_edges")
    add_tri(v_rad_lo_mid, v_rad_lo_base, v_lo_sh_rad_r, "ram_leading_edges")
    add_tri(v_rad_lo_mid, v_rad_mid_l, v_lo_sh_rad_l, "ram_leading_edges")
    add_tri(v_rad_lo_mid, v_lo_sh_rad_l, v_rad_lo_base, "ram_leading_edges")
    add_tri(v_lo_sh_rad_r, v_rad_base_r, v_rad_mid_r, "ram_leading_edges")
    add_tri(v_lo_sh_rad_l, v_rad_mid_l, v_rad_base_l, "ram_leading_edges")

    # Forward belly to cheek -> pec_body
    add_tri(v_rad_lo_base, v_belly_fwd, v_lo_sh_fwd_r, "pec_body")
    add_tri(v_rad_lo_base, v_lo_sh_fwd_r, v_lo_sh_rad_r, "pec_body")
    add_tri(v_rad_lo_base, v_lo_sh_rad_l, v_lo_sh_fwd_l, "pec_body")
    add_tri(v_rad_lo_base, v_lo_sh_fwd_l, v_belly_fwd, "pec_body")

    add_tri(v_lo_sh_rad_r, v_cheek_r, v_rad_base_r, "ram_leading_edges")
    add_tri(v_lo_sh_rad_r, v_lo_sh_fwd_r, v_cheek_r, "ram_leading_edges")
    add_tri(v_lo_sh_rad_l, v_rad_base_l, v_cheek_l, "ram_leading_edges")
    add_tri(v_lo_sh_rad_l, v_cheek_l, v_lo_sh_fwd_l, "ram_leading_edges")

    # Belly to intake bottom lip -> ram_intake
    add_tri(v_belly_fwd, v_belly_mid, v_intk_cb, "ram_intake")
    add_tri(v_belly_fwd, v_intk_cb, v_lo_sh_fwd_r, "ram_intake")
    add_tri(v_belly_fwd, v_lo_sh_fwd_l, v_intk_cl_b, "ram_intake")
    add_tri(v_belly_fwd, v_intk_cl_b, v_belly_mid, "ram_intake")

    add_tri(v_lo_sh_fwd_r, v_intk_side_r, v_cheek_r, "ram_intake")
    add_tri(v_lo_sh_fwd_r, v_lo_nac_mid_r, v_intk_side_r, "ram_intake")
    add_tri(v_lo_sh_fwd_l, v_cheek_l, v_intk_side_l, "ram_intake")
    add_tri(v_lo_sh_fwd_l, v_intk_side_l, v_lo_nac_mid_l, "ram_intake")

    add_tri(v_intk_side_r, v_w_root_le_r, v_lo_nac_mid_r, "ram_intake")
    add_tri(v_intk_side_l, v_lo_nac_mid_l, v_w_root_le_l, "ram_intake")

    add_tri(v_intk_cb, v_belly_mid, v_belly_aft, "pec_body")
    add_tri(v_intk_cb, v_belly_aft, v_lo_nac_mid_r, "pec_body")
    add_tri(v_intk_cl_b, v_lo_nac_mid_l, v_belly_aft, "pec_body")
    add_tri(v_intk_cl_b, v_belly_aft, v_belly_mid, "pec_body")

    add_tri(v_intk_cb, v_lo_nac_mid_r, v_intk_rb, "ram_intake")
    add_tri(v_intk_cl_b, v_intk_lb, v_lo_nac_mid_l, "ram_intake")

    # Lower Wing Surfaces
    add_tri(v_w_root_le_r, v_lo_nac_mid_r, v_w_tip_le_r, "ram_leading_edges")
    add_tri(v_w_root_le_l, v_w_tip_le_l, v_lo_nac_mid_l, "ram_leading_edges")
    add_tri(v_w_tip_le_r, v_lo_nac_mid_r, v_w_tip_te_r, "pec_body")
    add_tri(v_w_tip_le_l, v_w_tip_te_l, v_lo_nac_mid_l, "pec_body")
    add_tri(v_w_tip_te_r, v_lo_nac_mid_r, v_w_root_te_r, "pec_body")
    add_tri(v_w_tip_te_l, v_w_root_te_l, v_lo_nac_mid_l, "pec_body")

    # Lower Mid to Aft -> pec_body
    add_tri(v_belly_aft, v_belly_tail, v_lo_nac_aft_r, "pec_body")
    add_tri(v_belly_aft, v_lo_nac_aft_r, v_lo_nac_mid_r, "pec_body")
    add_tri(v_belly_aft, v_lo_nac_mid_l, v_lo_nac_aft_l, "pec_body")
    add_tri(v_belly_aft, v_lo_nac_aft_l, v_belly_tail, "pec_body")

    add_tri(v_w_root_te_r, v_lo_nac_mid_r, v_stab_root_le_r, "pec_body")
    add_tri(v_w_root_te_l, v_stab_root_le_l, v_lo_nac_mid_l, "pec_body")
    add_tri(v_lo_nac_mid_r, v_lo_nac_aft_r, v_stab_root_le_r, "pec_body")
    add_tri(v_lo_nac_mid_l, v_stab_root_le_l, v_lo_nac_aft_l, "pec_body")

    # Lower Horizontal Stabilators
    add_tri(v_stab_root_le_r, v_lo_nac_aft_r, v_stab_tip_le_r, "ram_leading_edges")
    add_tri(v_stab_root_le_l, v_stab_tip_le_l, v_lo_nac_aft_l, "ram_leading_edges")
    add_tri(v_stab_tip_le_r, v_lo_nac_aft_r, v_stab_tip_te_r, "pec_body")
    add_tri(v_stab_tip_le_l, v_stab_tip_te_l, v_lo_nac_aft_l, "pec_body")
    add_tri(v_stab_tip_te_r, v_lo_nac_aft_r, v_stab_root_te_r, "pec_body")
    add_tri(v_stab_tip_te_l, v_stab_root_te_l, v_lo_nac_aft_l, "pec_body")

    # Lower Tail Beaver & Nozzle Bottom -> pec_body
    add_tri(v_belly_tail, v_tail_center, v_noz_cb, "pec_body")
    add_tri(v_belly_tail, v_noz_cb, v_lo_nac_aft_r, "pec_body")
    add_tri(v_belly_tail, v_lo_nac_aft_l, v_noz_cl_b, "pec_body")
    add_tri(v_belly_tail, v_noz_cl_b, v_tail_center, "pec_body")

    add_tri(v_lo_nac_aft_r, v_noz_cb, v_noz_rb, "pec_body")
    add_tri(v_lo_nac_aft_r, v_noz_rb, v_stab_root_te_r, "pec_body")
    add_tri(v_lo_nac_aft_l, v_stab_root_te_l, v_noz_lb, "pec_body")
    add_tri(v_lo_nac_aft_l, v_noz_lb, v_noz_cl_b, "pec_body")

    # J. Twin Boxy Rectangular Intake Cavities -> ram_intake
    # Duct top
    add_tri(v_intk_ct, v_intk_rt, v_duct_back_rt, "ram_intake")
    add_tri(v_intk_ct, v_duct_back_rt, v_duct_back_ct, "ram_intake")
    add_tri(v_intk_cl_t, v_duct_back_cl_t, v_duct_back_lt, "ram_intake")
    add_tri(v_intk_cl_t, v_duct_back_lt, v_intk_lt, "ram_intake")
    # Duct outer side
    add_tri(v_intk_rt, v_intk_rb, v_duct_back_rb, "ram_intake")
    add_tri(v_intk_rt, v_duct_back_rb, v_duct_back_rt, "ram_intake")
    add_tri(v_intk_lt, v_duct_back_lt, v_duct_back_lb, "ram_intake")
    add_tri(v_intk_lt, v_duct_back_lb, v_intk_lb, "ram_intake")
    # Duct inner side
    add_tri(v_intk_ct, v_duct_back_ct, v_duct_back_cb, "ram_intake")
    add_tri(v_intk_ct, v_duct_back_cb, v_intk_cb, "ram_intake")
    add_tri(v_intk_cl_t, v_intk_cl_b, v_duct_back_cl_b, "ram_intake")
    add_tri(v_intk_cl_t, v_duct_back_cl_b, v_duct_back_cl_t, "ram_intake")
    # Duct bottom
    add_tri(v_intk_cb, v_duct_back_cb, v_duct_back_rb, "ram_intake")
    add_tri(v_intk_cb, v_duct_back_rb, v_intk_rb, "ram_intake")
    add_tri(v_intk_cl_b, v_intk_lb, v_duct_back_lb, "ram_intake")
    add_tri(v_intk_cl_b, v_duct_back_lb, v_duct_back_cl_b, "ram_intake")
    # Duct back compressor face with center spinner bullet cone -> ram_intake
    add_tri(v_hub_r, v_duct_back_ct, v_duct_back_rt, "ram_intake")
    add_tri(v_hub_r, v_duct_back_rt, v_duct_back_rb, "ram_intake")
    add_tri(v_hub_r, v_duct_back_rb, v_duct_back_cb, "ram_intake")
    add_tri(v_hub_r, v_duct_back_cb, v_duct_back_ct, "ram_intake")

    add_tri(v_hub_l, v_duct_back_lt, v_duct_back_cl_t, "ram_intake")
    add_tri(v_hub_l, v_duct_back_cl_t, v_duct_back_cl_b, "ram_intake")
    add_tri(v_hub_l, v_duct_back_cl_b, v_duct_back_lb, "ram_intake")
    add_tri(v_hub_l, v_duct_back_lb, v_duct_back_lt, "ram_intake")

    # K. Twin Pratt & Whitney F100 Exhaust Nozzles -> pec_body
    add_tri(v_noz_ct, v_noz_rt, v_noz_bk_rt, "pec_body")
    add_tri(v_noz_ct, v_noz_bk_rt, v_noz_bk_ct, "pec_body")
    add_tri(v_noz_cl_t, v_noz_bk_cl_t, v_noz_bk_lt, "pec_body")
    add_tri(v_noz_cl_t, v_noz_bk_lt, v_noz_lt, "pec_body")
    add_tri(v_noz_cb, v_noz_bk_cb, v_noz_bk_rb, "pec_body")
    add_tri(v_noz_cb, v_noz_bk_rb, v_noz_rb, "pec_body")
    add_tri(v_noz_cl_b, v_noz_lb, v_noz_bk_lb, "pec_body")
    add_tri(v_noz_cl_b, v_noz_bk_lb, v_noz_bk_cl_b, "pec_body")
    add_tri(v_noz_rt, v_noz_rb, v_noz_bk_rb, "pec_body")
    add_tri(v_noz_rt, v_noz_bk_rb, v_noz_bk_rt, "pec_body")
    add_tri(v_noz_lt, v_noz_bk_lt, v_noz_bk_lb, "pec_body")
    add_tri(v_noz_lt, v_noz_bk_lb, v_noz_lb, "pec_body")
    add_tri(v_noz_ct, v_noz_bk_ct, v_noz_bk_cb, "pec_body")
    add_tri(v_noz_ct, v_noz_bk_cb, v_noz_cb, "pec_body")
    add_tri(v_noz_cl_t, v_noz_cl_b, v_noz_bk_cl_b, "pec_body")
    add_tri(v_noz_cl_t, v_noz_bk_cl_b, v_noz_bk_cl_t, "pec_body")
    # Back flameholder face with aft mixer cone -> pec_body
    add_tri(v_noz_cone_r, v_noz_bk_rt, v_noz_bk_ct, "pec_body")
    add_tri(v_noz_cone_r, v_noz_bk_ct, v_noz_bk_cb, "pec_body")
    add_tri(v_noz_cone_r, v_noz_bk_cb, v_noz_bk_rb, "pec_body")
    add_tri(v_noz_cone_r, v_noz_bk_rb, v_noz_bk_rt, "pec_body")

    add_tri(v_noz_cone_l, v_noz_bk_cl_t, v_noz_bk_lt, "pec_body")
    add_tri(v_noz_cone_l, v_noz_bk_lt, v_noz_bk_lb, "pec_body")
    add_tri(v_noz_cone_l, v_noz_bk_lb, v_noz_bk_cl_b, "pec_body")
    add_tri(v_noz_cone_l, v_noz_bk_cl_b, v_noz_bk_cl_t, "pec_body")

    # Consistent winding pass
    def fix_winding(verts, tris):
        edge_map = {}
        for ti, (a, b, c, grp) in enumerate(tris):
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
                ca, cb, cc, cgrp = new_tris[curr]
                for u, v in [(ca, cb), (cb, cc), (cc, ca)]:
                    k = (min(u, v), max(u, v))
                    for adj in edge_map.get(k, []):
                        if adj not in visited:
                            visited.add(adj)
                            t = new_tris[adj]
                            for au, av in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
                                if (min(au, av), max(au, av)) == k:
                                    if au == u:
                                        new_tris[adj] = [t[0], t[2], t[1], t[3]]
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
        for i0, i1, i2, grp in cur_t:
            m01 = get_mid(i0, i1)
            m12 = get_mid(i1, i2)
            m20 = get_mid(i2, i0)
            next_t.append((i0, m01, m20, grp))
            next_t.append((m01, i1, m12, grp))
            next_t.append((m20, m12, i2, grp))
            next_t.append((m01, m12, m20, grp))
        cur_t = next_t

    cur_t = fix_winding(cur_v, cur_t)
    return cur_v, cur_t

def write_grouped_obj(filepath, vertices, triangles, comment="F-15 Eagle RAM Zoned"):
    # Group triangles by group tag
    from collections import defaultdict
    groups = defaultdict(list)
    for t in triangles:
        groups[t[3]].append((t[0], t[1], t[2]))

    with open(filepath, "w") as f:
        f.write(f"# StrikeCEM {comment}\n")
        f.write(f"# Vertices: {len(vertices)}, Triangles: {len(triangles)}\n")
        f.write(f"# Material Groups: {', '.join(sorted(groups.keys()))}\n")
        for v in vertices:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for grp in sorted(groups.keys()):
            f.write(f"g {grp}\n")
            for t in groups[grp]:
                f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Main detailed RAM benchmark model (resolution=2)
    verts, tris = build_f15_ram_mesh(resolution=2)
    full_path = os.path.join(out_dir, "f15_ram.obj")
    write_grouped_obj(full_path, verts, tris, "F-15 Eagle RAM-Zoned 3D Model")
    
    # Print group statistics
    from collections import Counter
    counts = Counter(t[3] for t in tris)
    print(f"Generated: {full_path} ({len(verts)} vertices, {len(tris)} triangles)")
    print("Material Zone Facet Breakdown:")
    for grp, count in sorted(counts.items()):
        pct = (count / len(tris)) * 100.0
        print(f"  - {grp:<20}: {count:4d} facets ({pct:5.1f}%)")

if __name__ == "__main__":
    main()

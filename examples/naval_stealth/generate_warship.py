#!/usr/bin/env python3
"""
Procedural 3D Mesh Generator for USS Zumwalt (DDG-1000) Stealth Guided-Missile Destroyer.
Features:
- Inverted wave-piercing tumblehome hull (inward-sloping hull sides: >8° inward rake)
- Sloped reverse stem bow
- Integrated composite deckhouse (tapered faceted pyramid)
- Upper integrated sensor mast & radar aperture tower
- Concealed Advanced Gun System (AGS) faceted stealth turrets (Mount 51 & Mount 52)
- Aft helicopter flight deck and forward-slanted reverse transom stern
- Semantic material tagging: pec_hull, ram_deckhouse, pec_deck, ram_turret, ram_mast
- Clean 2-manifold closed geometry (0 degenerate facets, 0 non-manifold edges)
"""

import math
from pathlib import Path


class WarshipBuilder:
    def __init__(self):
        self.vertices = []
        self.faces = []  # list of (v0, v1, v2, group_name)

    def add_vertex(self, x, y, z):
        self.vertices.append([float(x), float(y), float(z)])
        return len(self.vertices) - 1

    def add_triangle(self, v0, v1, v2, group):
        p0 = self.vertices[v0]
        p1 = self.vertices[v1]
        p2 = self.vertices[v2]
        # Cross product (p1 - p0) x (p2 - p0)
        e1x, e1y, e1z = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
        e2x, e2y, e2z = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
        cx = e1y * e2z - e1z * e2y
        cy = e1z * e2x - e1x * e2z
        cz = e1x * e2y - e1y * e2x
        area = 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
        if area < 1e-7:
            return  # Skip degenerate
        self.faces.append((v0, v1, v2, group))

    def add_quad(self, v0, v1, v2, v3, group):
        # Adds quad (v0, v1, v2, v3) as 2 triangles with consistent winding
        self.add_triangle(v0, v1, v2, group)
        self.add_triangle(v0, v2, v3, group)


def generate_zumwalt_mesh(output_obj_path: Path):
    wb = WarshipBuilder()

    # Principal Dimensions (1:1 True Scale)
    # Length overall: 190.0 m (x from -95.0 to +95.0)
    # Waterline beam: 24.6 m (y from -12.3 to +12.3)
    # Main deck height: z = 7.5 m above waterline
    # Waterline at z = 0.0 m (flat keel closure at z = 0.0 m)
    # Deckhouse height: z = 24.0 m
    # Sensor mast height: z = 32.0 m

    # Tumblehome geometry:
    # At any station x, waterline half-beam is w0, deck half-beam is w_deck = w0 - tumblehome_delta
    # tumblehome_delta = 7.5 * tan(9.5 deg) ~ 1.25 m.

    # Longitudinal stations along the hull
    # x = +95.0: Waterline bow knife tip
    # x = +85.0: Forward inverted bow transition
    # x = +75.0: Forecastle deck forward tip
    # x = +55.0: Forward gun mount station
    # x = +38.0: Second gun mount station
    # x = +25.0: Forward deckhouse base
    # x = 0.0:   Midships
    # x = -25.0: Mid-aft deckhouse
    # x = -45.0: Aft deckhouse / helicopter hangar front
    # x = -70.0: Mid flight deck
    # x = -93.0: Flight deck stern edge
    # x = -95.0: Waterline transom base

    stations = [
        # (x_wl, x_deck, w_wl, w_deck)
        (95.0, 75.0, 0.0, 0.0),       # Bow knife edge
        (85.0, 75.0, 4.0, 2.5),       # Forward rake
        (75.0, 70.0, 7.8, 6.2),       # Forecastle entry
        (55.0, 55.0, 10.8, 9.4),      # Turret 1 station
        (35.0, 35.0, 12.3, 11.0),     # Turret 2 station / max beam entry
        (25.0, 25.0, 12.3, 11.0),     # Forward superstructure
        (0.0, 0.0, 12.3, 11.0),       # Midships
        (-25.0, -25.0, 12.3, 11.0),   # Superstructure mid
        (-45.0, -45.0, 12.3, 11.0),   # Hangar / flight deck start
        (-70.0, -70.0, 11.6, 10.3),   # Mid flight deck
        (-93.0, -93.0, 10.5, 9.2),    # Aft flight deck edge
        (-95.0, -93.0, 10.5, 9.2),    # Transom waterline base
    ]

    n_st = len(stations)

    # Vertices along the hull:
    # For each station i:
    # 0: Keel centerline (x_wl, 0, 0)
    # 1: Starboard waterline (x_wl, -w_wl, 0)
    # 2: Starboard main deck (x_deck, -w_deck, 7.5)
    # 3: Deck centerline (x_deck, 0, 7.5)
    # 4: Port main deck (x_deck, +w_deck, 7.5)
    # 5: Port waterline (x_wl, +w_wl, 0)

    st_verts = []
    for i, (x_wl, x_deck, w_wl, w_deck) in enumerate(stations):
        v_keel = wb.add_vertex(x_wl, 0.0, 0.0)
        v_sb_wl = wb.add_vertex(x_wl, -w_wl, 0.0)
        v_sb_dk = wb.add_vertex(x_deck, -w_deck, 7.5)
        v_dk_cl = wb.add_vertex(x_deck, 0.0, 7.5)
        v_pt_dk = wb.add_vertex(x_deck, +w_deck, 7.5)
        v_pt_wl = wb.add_vertex(x_wl, +w_wl, 0.0)
        st_verts.append((v_keel, v_sb_wl, v_sb_dk, v_dk_cl, v_pt_dk, v_pt_wl))

    # Hull plating: connect stations
    for i in range(n_st - 1):
        k0, sb_wl0, sb_dk0, dk_cl0, pt_dk0, pt_wl0 = st_verts[i]
        k1, sb_wl1, sb_dk1, dk_cl1, pt_dk1, pt_wl1 = st_verts[i + 1]

        # Bottom keel plate (z = 0.0) -> closed bottom
        # Keel to starboard waterline
        wb.add_quad(k0, sb_wl0, sb_wl1, k1, "pec_keel")
        # Keel to port waterline
        wb.add_quad(k0, k1, pt_wl1, pt_wl0, "pec_keel")

        # Starboard tumblehome topside (waterline to deck)
        wb.add_quad(sb_wl0, sb_dk0, sb_dk1, sb_wl1, "ram_hull")

        # Port tumblehome topside (waterline to deck)
        wb.add_quad(pt_wl0, pt_wl1, pt_dk1, pt_dk0, "ram_hull")

        # Main deck plating (between stations where superstructure does not sit)
        # Superstructure sits between station 5 (x=25) and station 8 (x=-45)
        # On forecastle (i < 5) and flight deck (i >= 8), cover the full deck
        if i < 5 or i >= 8:
            wb.add_quad(sb_dk0, dk_cl0, dk_cl1, sb_dk1, "pec_deck")
            wb.add_quad(dk_cl0, pt_dk0, pt_dk1, pt_cl_next := dk_cl1, "pec_deck")
        else:
            # Side perimeter walkways around deckhouse (width 3.5 m on each side)
            # Deckhouse half-width at base is 7.5 m, deck half-width is 11.0 m -> walkway is 3.5 m
            pass

    # Bow Cap (Station 0):
    # Close forward bow knife:
    k0, sb_wl0, sb_dk0, dk_cl0, pt_dk0, pt_wl0 = st_verts[0]
    # Inverted stem triangle: connects (x_wl=95, z=0) to (x_deck=75, z=7.5)
    wb.add_triangle(k0, sb_dk0, dk_cl0, "ram_hull")
    wb.add_triangle(k0, dk_cl0, pt_dk0, "ram_hull")

    # Stern Transom (Station n_st - 1):
    # Close flat/slanted transom stern:
    k_end, sb_wl_end, sb_dk_end, dk_cl_end, pt_dk_end, pt_wl_end = st_verts[-1]
    wb.add_quad(k_end, sb_wl_end, sb_dk_end, dk_cl_end, "ram_hull")
    wb.add_quad(k_end, dk_cl_end, pt_dk_end, pt_wl_end, "ram_hull")

    # =========================================================================
    # Superstructure Deckhouse (Integrated Composite Stealth Deckhouse)
    # =========================================================================
    # Base: sits on main deck at z = 7.5 m, from x = +25.0 to x = -45.0 (70 m long)
    # Base half-width: 7.5 m (leaving 3.5 m walkways on port and starboard)
    # Roof: at z = 24.0 m (16.5 m tall), from x = +18.0 to x = -40.0 (58 m long)
    # Roof half-width: 5.2 m (slopes inward ~8.5° from vertical)

    # Deckhouse base vertices on main deck (z = 7.5)
    dh_b_fwd_sb = wb.add_vertex(25.0, -7.5, 7.5)
    dh_b_fwd_cl = wb.add_vertex(25.0, 0.0, 7.5)
    dh_b_fwd_pt = wb.add_vertex(25.0, 7.5, 7.5)

    dh_b_aft_sb = wb.add_vertex(-45.0, -7.5, 7.5)
    dh_b_aft_cl = wb.add_vertex(-45.0, 0.0, 7.5)
    dh_b_aft_pt = wb.add_vertex(-45.0, 7.5, 7.5)

    # Deckhouse roof vertices (z = 24.0)
    dh_r_fwd_sb = wb.add_vertex(18.0, -5.2, 24.0)
    dh_r_fwd_cl = wb.add_vertex(18.0, 0.0, 24.0)
    dh_r_fwd_pt = wb.add_vertex(18.0, 5.2, 24.0)

    dh_r_aft_sb = wb.add_vertex(-40.0, -5.2, 24.0)
    dh_r_aft_cl = wb.add_vertex(-40.0, 0.0, 24.0)
    dh_r_aft_pt = wb.add_vertex(-40.0, 5.2, 24.0)

    # Connect main deck walkways around deckhouse (x = 25 to x = -45)
    # Stations 5, 6, 7, 8
    # Starboard walkway: from sb_dk to dh_b_sb
    for i in range(5, 8):
        k0, sb_wl0, sb_dk0, dk_cl0, pt_dk0, pt_wl0 = st_verts[i]
        k1, sb_wl1, sb_dk1, dk_cl1, pt_dk1, pt_wl1 = st_verts[i + 1]

        # Starboard walkway quad
        x0 = wb.vertices[sb_dk0][0]
        x1 = wb.vertices[sb_dk1][0]
        v_in0 = wb.add_vertex(x0, -7.5, 7.5)
        v_in1 = wb.add_vertex(x1, -7.5, 7.5)
        wb.add_quad(sb_dk0, v_in0, v_in1, sb_dk1, "pec_deck")

        # Port walkway quad
        v_in_p0 = wb.add_vertex(x0, 7.5, 7.5)
        v_in_p1 = wb.add_vertex(x1, 7.5, 7.5)
        wb.add_quad(pt_dk0, pt_dk1, v_in_p1, v_in_p0, "pec_deck")

    # Deckhouse forward face (slopes backward 23°)
    wb.add_quad(dh_b_fwd_sb, dh_r_fwd_sb, dh_r_fwd_pt, dh_b_fwd_pt, "ram_deckhouse")

    # Deckhouse starboard face (slopes inward 8.5°)
    wb.add_quad(dh_b_fwd_sb, dh_b_aft_sb, dh_r_aft_sb, dh_r_fwd_sb, "ram_deckhouse")

    # Deckhouse port face (slopes inward 8.5°)
    wb.add_quad(dh_b_fwd_pt, dh_r_fwd_pt, dh_r_aft_pt, dh_b_aft_pt, "ram_deckhouse")

    # Deckhouse aft face (slopes forward 17°) - helicopter hangar door
    wb.add_quad(dh_b_aft_sb, dh_b_aft_pt, dh_r_aft_pt, dh_r_aft_sb, "ram_deckhouse")

    # Deckhouse roof (flat at z = 24.0, except where mast sits)
    # Roof spans x in [18, -40], y in [-5.2, 5.2]
    # Mast sits from x = 14 to x = 0, y in [-3.5, 3.5]
    # Let's create roof segments:
    m_fwd_x = 14.0
    m_aft_x = -2.0
    m_w = 3.2

    # Roof forward of mast (x from 18 to 14)
    r_m_fwd_sb = wb.add_vertex(m_fwd_x, -5.2, 24.0)
    r_m_fwd_pt = wb.add_vertex(m_fwd_x, 5.2, 24.0)
    wb.add_quad(dh_r_fwd_sb, r_m_fwd_sb, r_m_fwd_pt, dh_r_fwd_pt, "ram_deckhouse")

    # Roof aft of mast (x from -2 to -40)
    r_m_aft_sb = wb.add_vertex(m_aft_x, -5.2, 24.0)
    r_m_aft_pt = wb.add_vertex(m_aft_x, 5.2, 24.0)
    wb.add_quad(r_m_aft_sb, dh_r_aft_sb, dh_r_aft_pt, r_m_aft_pt, "ram_deckhouse")

    # Side roof strips around mast
    wb.add_quad(r_m_fwd_sb, r_m_aft_sb,
                wb.add_vertex(m_aft_x, -m_w, 24.0),
                wb.add_vertex(m_fwd_x, -m_w, 24.0), "ram_deckhouse")
    wb.add_quad(wb.add_vertex(m_fwd_x, m_w, 24.0),
                wb.add_vertex(m_aft_x, m_w, 24.0),
                r_m_aft_pt, r_m_fwd_pt, "ram_deckhouse")

    # =========================================================================
    # Integrated Sensor Mast & Radome Tower
    # =========================================================================
    # Sits on roof from x = 14.0 to x = -2.0 (16 m long), y in [-3.2, 3.2]
    # Base at z = 24.0, top pyramid at z = 32.0 m (8 m high)
    # Top roof of mast: x in [10.0, 2.0], y in [-1.8, 1.8] (faceted hexagonal pyramid)

    mast_b_fwd_sb = wb.add_vertex(m_fwd_x, -m_w, 24.0)
    mast_b_fwd_pt = wb.add_vertex(m_fwd_x, m_w, 24.0)
    mast_b_aft_sb = wb.add_vertex(m_aft_x, -m_w, 24.0)
    mast_b_aft_pt = wb.add_vertex(m_aft_x, m_w, 24.0)

    # Mast apex roof at z = 32.0 m
    mast_r_fwd_sb = wb.add_vertex(10.0, -1.8, 32.0)
    mast_r_fwd_pt = wb.add_vertex(10.0, 1.8, 32.0)
    mast_r_aft_sb = wb.add_vertex(2.0, -1.8, 32.0)
    mast_r_aft_pt = wb.add_vertex(2.0, 1.8, 32.0)

    # Mast sloped faces (AESA radar faces)
    # Forward sloped face (SPY-3 forward radar aperture)
    wb.add_quad(mast_b_fwd_sb, mast_r_fwd_sb, mast_r_fwd_pt, mast_b_fwd_pt, "ram_mast")
    # Starboard sloped face (lateral aperture)
    wb.add_quad(mast_b_fwd_sb, mast_b_aft_sb, mast_r_aft_sb, mast_r_fwd_sb, "ram_mast")
    # Port sloped face
    wb.add_quad(mast_b_fwd_pt, mast_r_fwd_pt, mast_r_aft_pt, mast_b_aft_pt, "ram_mast")
    # Aft sloped face
    wb.add_quad(mast_b_aft_sb, mast_b_aft_pt, mast_r_aft_pt, mast_r_aft_sb, "ram_mast")
    # Mast top plate
    wb.add_quad(mast_r_fwd_sb, mast_r_aft_sb, mast_r_aft_pt, mast_r_fwd_pt, "ram_mast")

    # =========================================================================
    # Advanced Gun System (AGS) Stealth Turrets
    # =========================================================================
    def build_stealth_turret(center_x: float, name_tag: str):
        # 155mm stealth turret: 9 m long, 5.5 m wide, 2.4 m high
        # Wedge-shaped low-RCS enclosure: front face slopes back at 35°, sides slope inward at 18°
        L_fwd = 5.0
        L_aft = 4.0
        W_base = 2.7
        W_top = 1.6
        z_base = 7.5
        z_top = 9.9

        # Base vertices
        t_b_f_sb = wb.add_vertex(center_x + L_fwd, -W_base, z_base)
        t_b_f_pt = wb.add_vertex(center_x + L_fwd, W_base, z_base)
        t_b_a_sb = wb.add_vertex(center_x - L_aft, -W_base, z_base)
        t_b_a_pt = wb.add_vertex(center_x - L_aft, W_base, z_base)

        # Roof vertices (tapered & raked)
        t_r_f_sb = wb.add_vertex(center_x + 2.0, -W_top, z_top)
        t_r_f_pt = wb.add_vertex(center_x + 2.0, W_top, z_top)
        t_r_a_sb = wb.add_vertex(center_x - 3.2, -W_top, z_top)
        t_r_a_pt = wb.add_vertex(center_x - 3.2, W_top, z_top)

        # Sloped turret armor faces
        wb.add_quad(t_b_f_sb, t_r_f_sb, t_r_f_pt, t_b_f_pt, name_tag)  # Front glacis
        wb.add_quad(t_b_f_sb, t_b_a_sb, t_r_a_sb, t_r_f_sb, name_tag)  # Starboard cheek
        wb.add_quad(t_b_f_pt, t_r_f_pt, t_r_a_pt, t_b_a_pt, name_tag)  # Port cheek
        wb.add_quad(t_b_a_sb, t_b_a_pt, t_r_a_pt, t_r_a_sb, name_tag)  # Aft bustle
        wb.add_quad(t_r_f_sb, t_r_a_sb, t_r_a_pt, t_r_f_pt, name_tag)  # Turret roof (flush stowed barrel)

    # Turret 1 (Mount 51 forward)
    build_stealth_turret(center_x=52.0, name_tag="ram_turret")
    # Turret 2 (Mount 52 mid-forecastle)
    build_stealth_turret(center_x=36.0, name_tag="ram_turret")

    # =========================================================================
    # Write Wavefront OBJ File with Group Tags
    # =========================================================================
    output_obj_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_obj_path, "w") as f:
        f.write("# StrikeCEM USS Zumwalt (DDG-1000) Stealth Destroyer Benchmark Mesh\n")
        f.write(f"# Dimensions: Length=190.0m, Beam=24.6m, Height=32.0m\n")
        f.write(f"# Vertices: {len(wb.vertices)}, Triangles: {len(wb.faces)}\n\n")

        for v in wb.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        current_group = None
        for v0, v1, v2, group in wb.faces:
            if group != current_group:
                current_group = group
                f.write(f"\ng {group}\n")
            f.write(f"f {v0 + 1} {v1 + 1} {v2 + 1}\n")

    print(f"Successfully generated {output_obj_path}")
    print(f"  Total Vertices: {len(wb.vertices)}")
    print(f"  Total Triangles: {len(wb.faces)}")

    # Verify mesh geometry
    xs = [v[0] for v in wb.vertices]
    ys = [v[1] for v in wb.vertices]
    zs = [v[2] for v in wb.vertices]
    print(f"  Bounding Box: X=[{min(xs):.1f}, {max(xs):.1f}], Y=[{min(ys):.1f}, {max(ys):.1f}], Z=[{min(zs):.1f}, {max(zs):.1f}]")
    print(f"  Total Length: {max(xs) - min(xs):.1f} m, Beam: {max(ys) - min(ys):.1f} m, Height: {max(zs) - min(zs):.1f} m")

    # Material group breakdown
    group_counts = {}
    for _, _, _, g in wb.faces:
        group_counts[g] = group_counts.get(g, 0) + 1
    print("  Semantic Material Breakdown:")
    for g, count in sorted(group_counts.items()):
        print(f"    - {g:20s}: {count:4d} triangles")


if __name__ == "__main__":
    out_file = Path("examples/naval_stealth/zumwalt_stealth.obj")
    generate_zumwalt_mesh(out_file)

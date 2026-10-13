#!/usr/bin/env python3
"""
Generates clean 3D OBJ models for a blended-wing-body stealth Unmanned Combat Aerial Vehicle (UCAV).
Creates:
  - ucav.obj (full detailed model)
  - ucav_lod1.obj (medium LOD)
  - ucav_lod2.obj (coarse LOD)
Features:
  - Swept sharp leading and trailing edges (PTD/EEC diffraction)
  - Blended curved upper fuselage (Fock curvature correction)
  - Flat planar wing/belly surfaces (Specular Physical Optics)
  - Re-entrant dorsal inlet cavity with internal sidewalls (SBR multi-bounce)
  - Full bilateral symmetry across xz-plane (y=0)
"""
import math
import os
import sys

def normalize(v):
    l = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if l < 1e-12:
        return (0.0, 0.0, 1.0)
    return (v[0]/l, v[1]/l, v[2]/l)

def cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    )

def sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def generate_mesh(resolution=1):
    vertices = []
    triangles = []

    def add_vertex(x, y, z):
        v = (round(float(x), 5), round(float(y), 5), round(float(z), 5))
        vertices.append(v)
        return len(vertices) - 1

    def add_tri(i0, i1, i2):
        # Ensure outward normal
        triangles.append((i0, i1, i2))

    def add_quad(i0, i1, i2, i3):
        add_tri(i0, i1, i2)
        add_tri(i0, i2, i3)

    # 1. Planform outer contour points (z=0 waterline)
    # Nose -> Outer Leading Edge -> Wingtip -> Outer Trailing Edge -> Inner Trailing Edge -> Tail beaver-tail
    # Coordinates in meters: +x is forward, +y is starboard (right), +z is up
    
    # Grid stations along length (x from 3.5 to -3.0)
    # Right half (y >= 0) and mirrored to left half (y <= 0)
    
    # Key landmark vertices (Right Side):
    # Nose tip
    v_nose = add_vertex(3.5, 0.0, 0.0)
    # Chine transition
    v_chine_r = add_vertex(2.0, 1.2, 0.0)
    v_chine_l = add_vertex(2.0, -1.2, 0.0)
    # Wing root / leading edge break
    v_leroot_r = add_vertex(0.8, 2.5, 0.0)
    v_leroot_l = add_vertex(0.8, -2.5, 0.0)
    # Wingtip
    v_tip_r = add_vertex(-1.2, 5.0, 0.0)
    v_tip_l = add_vertex(-1.2, -5.0, 0.0)
    # Trailing edge outer
    v_te_outer_r = add_vertex(-1.8, 4.8, 0.0)
    v_te_outer_l = add_vertex(-1.8, -4.8, 0.0)
    # Trailing edge break
    v_te_break_r = add_vertex(-2.2, 2.0, 0.0)
    v_te_break_l = add_vertex(-2.2, -2.0, 0.0)
    # Tail / beaver tail tip
    v_tail = add_vertex(-3.0, 0.0, 0.0)

    # 2. Upper Fuselage Ridge & Cockpit/Avionics Dome
    v_ridge_nose = add_vertex(3.0, 0.0, 0.15)
    v_ridge_fore = add_vertex(2.0, 0.0, 0.40)
    v_ridge_mid  = add_vertex(1.0, 0.0, 0.55)
    
    # Upper blended shoulders
    v_up_sh_r1 = add_vertex(2.0, 0.6, 0.28)
    v_up_sh_l1 = add_vertex(2.0, -0.6, 0.28)
    v_up_sh_r2 = add_vertex(0.8, 1.2, 0.35)
    v_up_sh_l2 = add_vertex(0.8, -1.2, 0.35)
    v_up_sh_r3 = add_vertex(-0.8, 1.8, 0.22)
    v_up_sh_l3 = add_vertex(-0.8, -1.8, 0.22)
    v_up_sh_r4 = add_vertex(-2.0, 1.0, 0.15)
    v_up_sh_l4 = add_vertex(-2.0, -1.0, 0.15)

    # 3. Lower Fuselage (Belly)
    v_belly_nose = add_vertex(3.0, 0.0, -0.10)
    v_belly_fore = add_vertex(2.0, 0.0, -0.28)
    v_belly_mid  = add_vertex(0.8, 0.0, -0.40)
    v_belly_aft  = add_vertex(-1.5, 0.0, -0.25)
    v_belly_tail = add_vertex(-2.6, 0.0, -0.10)

    # Lower blended shoulders
    v_lo_sh_r1 = add_vertex(2.0, 0.6, -0.20)
    v_lo_sh_l1 = add_vertex(2.0, -0.6, -0.20)
    v_lo_sh_r2 = add_vertex(0.8, 1.2, -0.25)
    v_lo_sh_l2 = add_vertex(0.8, -1.2, -0.25)
    v_lo_sh_r3 = add_vertex(-0.8, 1.8, -0.16)
    v_lo_sh_l3 = add_vertex(-0.8, -1.8, -0.16)
    v_lo_sh_r4 = add_vertex(-2.0, 1.0, -0.10)
    v_lo_sh_l4 = add_vertex(-2.0, -1.0, -0.10)

    # 4. Engine Inlet Cavity (Dorsal / Upper mounted, recessed into fuselage)
    # Lip / Opening entrance at x = 0.6 to 0.8
    v_inlet_lip_rt = add_vertex(0.8, 0.5, 0.45)
    v_inlet_lip_lt = add_vertex(0.8, -0.5, 0.45)
    v_inlet_lip_rb = add_vertex(0.8, 0.4, 0.15)
    v_inlet_lip_lb = add_vertex(0.8, -0.4, 0.15)

    # Duct Interior Cavity (Deep pocket extending forward/down to x = -0.6)
    v_duct_back_rt = add_vertex(-0.6, 0.45, 0.35)
    v_duct_back_lt = add_vertex(-0.6, -0.45, 0.35)
    v_duct_back_rb = add_vertex(-0.6, 0.35, 0.10)
    v_duct_back_lb = add_vertex(-0.6, -0.35, 0.10)

    # Upper fuselage aft of inlet
    v_aft_roof_r = add_vertex(-1.0, 0.6, 0.35)
    v_aft_roof_l = add_vertex(-1.0, -0.6, 0.35)
    v_aft_roof_c = add_vertex(-1.8, 0.0, 0.22)

    # --- Triangulation ---

    # A. Upper Forward Nose & Chines
    add_tri(v_nose, v_chine_r, v_ridge_nose)
    add_tri(v_nose, v_ridge_nose, v_chine_l)
    
    add_tri(v_ridge_nose, v_chine_r, v_up_sh_r1)
    add_tri(v_ridge_nose, v_up_sh_r1, v_ridge_fore)
    add_tri(v_ridge_nose, v_ridge_fore, v_up_sh_l1)
    add_tri(v_ridge_nose, v_up_sh_l1, v_chine_l)

    add_tri(v_ridge_fore, v_up_sh_r1, v_up_sh_r2)
    add_tri(v_ridge_fore, v_up_sh_r2, v_ridge_mid)
    add_tri(v_ridge_fore, v_ridge_mid, v_up_sh_l2)
    add_tri(v_ridge_fore, v_up_sh_l2, v_up_sh_l1)

    add_tri(v_chine_r, v_leroot_r, v_up_sh_r2)
    add_tri(v_chine_r, v_up_sh_r2, v_up_sh_r1)
    add_tri(v_chine_l, v_up_sh_l1, v_up_sh_l2)
    add_tri(v_chine_l, v_up_sh_l2, v_leroot_l)

    # B. Upper Wing Surfaces
    add_tri(v_leroot_r, v_tip_r, v_up_sh_r3)
    add_tri(v_up_sh_r2, v_leroot_r, v_up_sh_r3)
    add_tri(v_leroot_l, v_up_sh_l3, v_tip_l)
    add_tri(v_up_sh_l2, v_up_sh_l3, v_leroot_l)

    add_tri(v_tip_r, v_te_outer_r, v_up_sh_r3)
    add_tri(v_te_outer_r, v_te_break_r, v_up_sh_r3)
    add_tri(v_tip_l, v_up_sh_l3, v_te_outer_l)
    add_tri(v_te_outer_l, v_up_sh_l3, v_te_break_l)

    add_tri(v_te_break_r, v_up_sh_r4, v_up_sh_r3)
    add_tri(v_te_break_l, v_up_sh_l3, v_up_sh_l4)
    add_tri(v_te_break_r, v_tail, v_up_sh_r4)
    add_tri(v_te_break_l, v_up_sh_l4, v_tail)

    # C. Inlet Surround & Roof Fairing
    # Leading to lip
    add_tri(v_ridge_mid, v_up_sh_r2, v_inlet_lip_rt)
    add_tri(v_ridge_mid, v_inlet_lip_lt, v_up_sh_l2)
    add_tri(v_ridge_mid, v_inlet_lip_rt, v_inlet_lip_lt)

    add_tri(v_inlet_lip_rt, v_up_sh_r2, v_up_sh_r3)
    add_tri(v_inlet_lip_rt, v_up_sh_r3, v_aft_roof_r)
    add_tri(v_inlet_lip_lt, v_aft_roof_l, v_up_sh_l3)
    add_tri(v_inlet_lip_lt, v_up_sh_l3, v_up_sh_l2)

    # Aft upper deck
    add_tri(v_aft_roof_r, v_up_sh_r3, v_up_sh_r4)
    add_tri(v_aft_roof_l, v_up_sh_l4, v_up_sh_l3)
    add_tri(v_aft_roof_r, v_up_sh_r4, v_aft_roof_c)
    add_tri(v_aft_roof_l, v_aft_roof_c, v_up_sh_l4)
    add_tri(v_aft_roof_c, v_up_sh_r4, v_tail)
    add_tri(v_aft_roof_c, v_tail, v_up_sh_l4)

    # D. RE-ENTRANT ENGINE CAVITY (INTERIOR FACES)
    # The normals of these faces point INWARD towards the duct centerline!
    # Duct top ceiling
    add_tri(v_inlet_lip_rt, v_inlet_lip_lt, v_duct_back_lt)
    add_tri(v_inlet_lip_rt, v_duct_back_lt, v_duct_back_rt)
    # Duct right sidewall (facing left towards -y)
    add_tri(v_inlet_lip_rt, v_duct_back_rt, v_duct_back_rb)
    add_tri(v_inlet_lip_rt, v_duct_back_rb, v_inlet_lip_rb)
    # Duct left sidewall (facing right towards +y)
    add_tri(v_inlet_lip_lt, v_duct_back_lb, v_duct_back_lt)
    add_tri(v_inlet_lip_lt, v_inlet_lip_lb, v_duct_back_lb)
    # Duct floor
    add_tri(v_inlet_lip_rb, v_duct_back_rb, v_duct_back_lb)
    add_tri(v_inlet_lip_rb, v_duct_back_lb, v_inlet_lip_lb)
    # Duct back compressor face / cavity back wall (facing forward towards +x)
    add_tri(v_duct_back_rt, v_duct_back_lt, v_duct_back_lb)
    add_tri(v_duct_back_rt, v_duct_back_lb, v_duct_back_rb)

    # E. Lower Fuselage & Belly Surfaces
    add_tri(v_nose, v_belly_nose, v_chine_r)
    add_tri(v_nose, v_chine_l, v_belly_nose)

    add_tri(v_belly_nose, v_lo_sh_r1, v_chine_r)
    add_tri(v_belly_nose, v_belly_fore, v_lo_sh_r1)
    add_tri(v_belly_nose, v_chine_l, v_lo_sh_l1)
    add_tri(v_belly_nose, v_lo_sh_l1, v_belly_fore)

    add_tri(v_belly_fore, v_lo_sh_r1, v_lo_sh_r2)
    add_tri(v_belly_fore, v_lo_sh_r2, v_belly_mid)
    add_tri(v_belly_fore, v_lo_sh_l2, v_lo_sh_l1)
    add_tri(v_belly_fore, v_belly_mid, v_lo_sh_l2)

    add_tri(v_chine_r, v_lo_sh_r1, v_lo_sh_r2)
    add_tri(v_chine_r, v_lo_sh_r2, v_leroot_r)
    add_tri(v_chine_l, v_lo_sh_l2, v_lo_sh_l1)
    add_tri(v_chine_l, v_leroot_l, v_lo_sh_l2)

    # Lower wings
    add_tri(v_leroot_r, v_lo_sh_r2, v_lo_sh_r3)
    add_tri(v_leroot_r, v_lo_sh_r3, v_tip_r)
    add_tri(v_leroot_l, v_lo_sh_l3, v_lo_sh_l2)
    add_tri(v_leroot_l, v_tip_l, v_lo_sh_l3)

    add_tri(v_tip_r, v_lo_sh_r3, v_te_outer_r)
    add_tri(v_te_outer_r, v_lo_sh_r3, v_te_break_r)
    add_tri(v_tip_l, v_te_outer_l, v_lo_sh_l3)
    add_tri(v_te_outer_l, v_te_break_l, v_lo_sh_l3)

    add_tri(v_te_break_r, v_lo_sh_r3, v_lo_sh_r4)
    add_tri(v_te_break_l, v_lo_sh_l4, v_lo_sh_l3)
    add_tri(v_te_break_r, v_lo_sh_r4, v_tail)
    add_tri(v_te_break_l, v_tail, v_lo_sh_l4)

    # Lower mid belly
    add_tri(v_belly_mid, v_lo_sh_r2, v_lo_sh_r3)
    add_tri(v_belly_mid, v_lo_sh_r3, v_belly_aft)
    add_tri(v_belly_mid, v_lo_sh_l3, v_lo_sh_l2)
    add_tri(v_belly_mid, v_belly_aft, v_lo_sh_l3)

    add_tri(v_belly_aft, v_lo_sh_r3, v_lo_sh_r4)
    add_tri(v_belly_aft, v_lo_sh_r4, v_belly_tail)
    add_tri(v_belly_aft, v_lo_sh_l4, v_lo_sh_l3)
    add_tri(v_belly_aft, v_belly_tail, v_lo_sh_l4)

    add_tri(v_belly_tail, v_lo_sh_r4, v_tail)
    add_tri(v_belly_tail, v_tail, v_lo_sh_l4)

    # If sub-division is requested for detailed mesh (resolution > 1):
    cur_v = list(vertices)
    cur_t = list(triangles)
    
    for _ in range(resolution - 1):
        midpoints = {}
        def get_mid(a, b):
            k = (min(a, b), max(a, b))
            if k in midpoints:
                return midpoints[k]
            v0 = cur_v[a]
            v1 = cur_v[b]
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
    # Apply BFS winding consistency
    def fix_mesh_winding(verts, tris):
        edge_map = {}
        for ti, (a, b, c) in enumerate(tris):
            for u, v in [(a, b), (b, c), (c, a)]:
                k = (min(u, v), max(u, v))
                edge_map.setdefault(k, []).append(ti)
        
        visited = set()
        new_tris = [list(t) for t in tris]
        from collections import deque
        
        for start in range(len(tris)):
            if start in visited:
                continue
            queue = deque([start])
            visited.add(start)
            
            while queue:
                curr = queue.popleft()
                ca, cb, cc = new_tris[curr]
                for u, v in [(ca, cb), (cb, cc), (cc, ca)]:
                    k = (min(u, v), max(u, v))
                    for adj in edge_map.get(k, []):
                        if adj not in visited:
                            visited.add(adj)
                            t = new_tris[adj]
                            for au, av in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
                                if (min(au, av), max(au, av)) == k:
                                    if au == u: # Same forward direction; flip to reverse
                                        new_tris[adj] = [t[0], t[2], t[1]]
                                    break
                            queue.append(adj)
        return [tuple(t) for t in new_tris]

    cur_t = fix_mesh_winding(cur_v, cur_t)
    return cur_v, cur_t

def write_obj(filepath, vertices, triangles):
    with open(filepath, "w") as f:
        f.write("# StrikeCEM UCAV Benchmark Model\n")
        f.write(f"# Vertices: {len(vertices)}, Triangles: {len(triangles)}\n")
        f.write("o UCAV\n")
        for v in vertices:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for t in triangles:
            # 1-indexed in OBJ format
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Main detailed model (resolution=3 -> ~1000-2000 facets)
    v_main, t_main = generate_mesh(resolution=2)
    p_main = os.path.join(out_dir, "ucav.obj")
    write_obj(p_main, v_main, t_main)
    print(f"Generated {p_main}: {len(v_main)} vertices, {len(t_main)} triangles")

    # 2. Medium LOD (resolution=2)
    v_lod1, t_lod1 = generate_mesh(resolution=2)
    p_lod1 = os.path.join(out_dir, "ucav_lod1.obj")
    write_obj(p_lod1, v_lod1, t_lod1)
    print(f"Generated {p_lod1}: {len(v_lod1)} vertices, {len(t_lod1)} triangles")

    # 3. Coarse LOD (resolution=1)
    v_lod2, t_lod2 = generate_mesh(resolution=1)
    p_lod2 = os.path.join(out_dir, "ucav_lod2.obj")
    write_obj(p_lod2, v_lod2, t_lod2)
    print(f"Generated {p_lod2}: {len(v_lod2)} vertices, {len(t_lod2)} triangles")

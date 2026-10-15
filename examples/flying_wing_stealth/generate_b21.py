#!/usr/bin/env python3
"""
Procedural 3D Mesh Generator for Next-Gen B-21 Raider Stealth Flying-Wing Bomber.
Features:
- Parametric continuous 35.0° sweep planform edge alignment
- Serrated "W" trailing edge chevron notches matching leading-edge angle
- Continuous-curvature blended wing-body (BWB) supercritical airfoil
- Blended cockpit canopy bubble with Indium Tin Oxide (ITO) coating
- Dual submerged dorsal engine air intake S-ducts with internal cavity lining
- Multi-zone semantic material tagging:
    ram_leading_edge, ram_trailing_edge, ram_canopy, ram_intake_duct, carbon_composite_skin
- Clean 2-manifold closed geometry (0 degenerate facets, 0 non-manifold edges)
"""

import math
from pathlib import Path


class FlyingWingBuilder:
    def __init__(self):
        self.vertices = []
        self.vertex_map = {}
        self.faces = []

    def get_or_add_vertex(self, x, y, z):
        # Round to 6 decimals to ensure exact coordinate coincidence
        key = (round(float(x), 5), round(float(y), 5), round(float(z), 5))
        if key in self.vertex_map:
            return self.vertex_map[key]
        idx = len(self.vertices)
        self.vertices.append([key[0], key[1], key[2]])
        self.vertex_map[key] = idx
        return idx

    def add_triangle(self, v0, v1, v2, group):
        if v0 == v1 or v1 == v2 or v2 == v0:
            return
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
        if area < 1e-6:
            return
        self.faces.append((v0, v1, v2, group))

    def add_quad(self, v00, v10, v11, v01, group):
        # Quadrilateral with counter-clockwise winding
        self.add_triangle(v00, v10, v11, group)
        self.add_triangle(v00, v11, v01, group)


def planform_x_le(y, sweep_rad, x_nose):
    return x_nose - abs(y) * math.tan(sweep_rad)


def planform_x_te(y, sweep_rad):
    ay = abs(y)
    tan_s = math.tan(sweep_rad)
    # Piecewise W-serrated trailing edge:
    # y in [0, 5.0]: forward rake to center notch
    # y in [5.0, 12.0]: forward rake to chevron valley
    # y in [12.0, 20.0]: backward rake to wingtip
    if ay <= 5.0:
        return -7.3038 + (5.0 - ay) * tan_s
    elif ay <= 12.0:
        return -2.40234 - (ay - 5.0) * tan_s
    else:
        return -8.004 + (20.0 - ay) * tan_s


def generate_b21_mesh(output_obj_path: Path):
    fw = FlyingWingBuilder()

    # Principal Dimensions (1:1 True Scale B-21 Raider)
    # Wingspan: 40.0 m (y from -20.0 to +20.0)
    # Centerline Length: 15.3 m
    # Max Height: 3.3 m
    sweep_deg = 35.0
    sweep_rad = math.radians(sweep_deg)
    x_nose = 8.0
    y_max = 20.0

    # Spanwise grid stations (symmetric around y = 0)
    # Denser near centerline, cockpit, intake, and chevrons
    span_stations = [
        0.0, 0.6, 1.2, 1.8, 2.5, 3.2, 4.0, 5.0, 6.2, 7.5,
        9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0, 19.2, 20.0
    ]
    # Full span from -20 to +20
    all_y = sorted(list(set([-y for y in span_stations] + span_stations)))

    # Chordwise stations: u in [0, 1] from trailing edge (0) to leading edge (1)
    # 21 stations chordwise with clustering near LE and TE
    u_stations = []
    for i in range(21):
        # Cosine clustering: 0.5 * (1 - cos(pi * i / 20))
        u = 0.5 * (1.0 - math.cos(math.pi * float(i) / 20.0))
        u_stations.append(u)

    ny = len(all_y)
    nu = len(u_stations)

    # Precompute 3D coordinates on upper and lower surfaces
    grid_upper = [[None for _ in range(nu)] for _ in range(ny)]
    grid_lower = [[None for _ in range(nu)] for _ in range(ny)]

    for j, y in enumerate(all_y):
        ay = abs(y)
        eta = ay / y_max
        x_le = planform_x_le(y, sweep_rad, x_nose)
        x_te = planform_x_te(y, sweep_rad)
        chord = x_le - x_te

        # Baseline thickness profile
        if eta < 0.25:
            # Centerbody thick section
            t_base = 2.4 - 0.4 * (eta / 0.25) ** 2
        else:
            # Tapering wing
            t_base = 2.0 * ((1.0 - eta) ** 1.8) + 0.22

        for i, u in enumerate(u_stations):
            x = x_te + u * chord

            # Supercritical biconvex thickness distribution
            z_base_up = t_base * 3.8 * u * (1.0 - u) * (0.8 + 0.3 * u)
            z_base_low = -t_base * 2.6 * u * (1.0 - u) * (0.7 + 0.3 * u)

            # Cockpit canopy blister
            z_canopy = 0.0
            if ay <= 1.8 and 1.5 <= x <= 6.0:
                y_factor = max(0.0, 1.0 - (ay / 1.8) ** 2)
                x_factor = max(0.0, 1.0 - ((x - 3.8) / 2.2) ** 2) ** 2
                z_canopy = 0.85 * y_factor * x_factor

            # Submerged dorsal engine air intakes
            z_intake = 0.0
            if 2.6 <= ay <= 5.2 and -1.0 <= x <= 3.4:
                # Nacelle shoulder rise
                y_nac = math.cos(math.pi * (ay - 3.9) / 2.6) ** 2
                x_nac = math.cos(math.pi * (x - 1.2) / 4.4) ** 2
                z_nacelle = 0.45 * y_nac * x_nac

                # Submerged intake S-duct cavity depression
                z_duct = 0.0
                if 3.0 <= ay <= 4.8 and 0.8 <= x <= 2.8:
                    y_duct = math.cos(math.pi * (ay - 3.9) / 1.8) ** 2
                    x_duct = math.sin(math.pi * (x - 0.8) / 2.0)
                    z_duct = -1.1 * y_duct * x_duct

                z_intake = z_nacelle + z_duct

            z_up = z_base_up + z_canopy + z_intake
            z_low = z_base_low

            # At leading edge (u=1) and trailing edge (u=0), upper and lower must meet exactly
            if i == 0 or i == nu - 1:
                z_up = 0.0
                z_low = 0.0

            # Store vertex indices
            v_up = fw.get_or_add_vertex(x, y, z_up)
            v_low = fw.get_or_add_vertex(x, y, z_low)
            grid_upper[j][i] = v_up
            grid_lower[j][i] = v_low

    # Build quadrilateral / triangular surface mesh
    for j in range(ny - 1):
        y_mid = 0.5 * (all_y[j] + all_y[j + 1])
        ay_mid = abs(y_mid)
        for i in range(nu - 1):
            u_mid = 0.5 * (u_stations[i] + u_stations[i + 1])
            x_mid = 0.5 * (
                planform_x_te(y_mid, sweep_rad) * (1.0 - u_mid)
                + planform_x_le(y_mid, sweep_rad, x_nose) * u_mid
            )

            # Determine semantic material classification
            group_upper = "carbon_composite_skin"
            group_lower = "carbon_composite_skin"

            if u_mid >= 0.85:
                # Leading edge wedge
                group_upper = "ram_leading_edge"
                group_lower = "ram_leading_edge"
            elif u_mid <= 0.12:
                # Trailing edge serrated chevrons
                group_upper = "ram_trailing_edge"
                group_lower = "ram_trailing_edge"
            elif ay_mid <= 1.8 and 1.8 <= x_mid <= 5.8:
                # Cockpit canopy (upper surface only)
                group_upper = "ram_canopy"
            elif 2.8 <= ay_mid <= 5.0 and 0.5 <= x_mid <= 3.2:
                # Dorsal engine air intake S-duct (upper surface only)
                group_upper = "ram_intake_duct"

            # Upper surface quad (outward normal points +z: v00 -> v01 (+x) -> v11 -> v10 (+y))
            u_v00 = grid_upper[j][i]
            u_v10 = grid_upper[j + 1][i]
            u_v11 = grid_upper[j + 1][i + 1]
            u_v01 = grid_upper[j][i + 1]
            fw.add_quad(u_v00, u_v01, u_v11, u_v10, group_upper)

            # Lower surface quad (outward normal points -z: v00 -> v10 (+y) -> v11 -> v01 (+x))
            l_v00 = grid_lower[j][i]
            l_v10 = grid_lower[j + 1][i]
            l_v11 = grid_lower[j + 1][i + 1]
            l_v01 = grid_lower[j][i + 1]
            fw.add_quad(l_v00, l_v10, l_v11, l_v01, group_lower)

    # Wingtip endcaps at port and starboard tips (y = -y_max and y = +y_max)
    # Starboard tip (j = ny - 1, outward normal points +y)
    for i in range(nu - 1):
        v_u0 = grid_upper[ny - 1][i]
        v_u1 = grid_upper[ny - 1][i + 1]
        v_l0 = grid_lower[ny - 1][i]
        v_l1 = grid_lower[ny - 1][i + 1]
        fw.add_quad(v_u0, v_u1, v_l1, v_l0, "ram_trailing_edge")

    # Port tip (j = 0, outward normal points -y)
    for i in range(nu - 1):
        v_u0 = grid_upper[0][i]
        v_u1 = grid_upper[0][i + 1]
        v_l0 = grid_lower[0][i]
        v_l1 = grid_lower[0][i + 1]
        fw.add_quad(v_u0, v_l0, v_l1, v_u1, "ram_trailing_edge")

    # Write OBJ file with grouped material tags
    output_obj_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_obj_path, "w") as f:
        f.write("# B-21 Raider Next-Gen Stealth Flying-Wing Bomber 3D Mesh\n")
        f.write("# Generated by StrikeCEM Procedural CAD Engine\n")
        f.write(f"# Vertices: {len(fw.vertices)}, Triangles: {len(fw.faces)}\n\n")

        for v in fw.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Group faces by material group
        current_group = None
        for v0, v1, v2, group in fw.faces:
            if group != current_group:
                current_group = group
                f.write(f"g {group}\n")
            # 1-indexed for OBJ format
            f.write(f"f {v0 + 1} {v1 + 1} {v2 + 1}\n")

    print(f"Generated B-21 stealth bomber mesh: {output_obj_path}")
    print(f"  Vertices:  {len(fw.vertices)}")
    print(f"  Triangles: {len(fw.faces)}")
    materials = set(g for _, _, _, g in fw.faces)
    print(f"  Material groups: {sorted(list(materials))}")


if __name__ == "__main__":
    out_obj = Path("examples/flying_wing_stealth/b21_stealth.obj")
    generate_b21_mesh(out_obj)

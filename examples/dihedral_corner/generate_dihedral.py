#!/usr/bin/env python3
"""
Generate a 90-degree canonical Dihedral Corner Reflector mesh.
Plate 1: on z=0, x in [0, a], y in [0, b], normal = (0, 0, 1)
Plate 2: on x=0, z in [0, a], y in [0, b], normal = (1, 0, 0)
Corner line along the y-axis.
Symmetric retroreflection occurs at az=180 deg, el=-45 deg.
"""
import os

def generate_canonical_dihedral(a=0.5, b=0.5, n=24, output_path="dihedral_90deg.obj"):
    vertices = []
    triangles = []

    # Plate 1: z = 0, x in [0, a], y in [0, b], normal +z
    v_base1 = len(vertices)
    for j in range(n + 1):
        y = b * (j / n)
        for i in range(n + 1):
            x = a * (i / n)
            vertices.append((x, y, 0.0))

    for j in range(n):
        for i in range(n):
            p0 = v_base1 + j * (n + 1) + i
            p1 = v_base1 + j * (n + 1) + (i + 1)
            p2 = v_base1 + (j + 1) * (n + 1) + (i + 1)
            p3 = v_base1 + (j + 1) * (n + 1) + i
            # Normal: (p1-p0) x (p3-p0) -> dx x dy = +z
            triangles.append((p0 + 1, p1 + 1, p2 + 1))
            triangles.append((p0 + 1, p2 + 1, p3 + 1))

    # Plate 2: x = 0, z in [0, a], y in [0, b], normal +x
    v_base2 = len(vertices)
    for j in range(n + 1):
        y = b * (j / n)
        for k in range(n + 1):
            z = a * (k / n)
            vertices.append((0.0, y, z))

    for j in range(n):
        for k in range(n):
            p0 = v_base2 + j * (n + 1) + k
            p1 = v_base2 + (j + 1) * (n + 1) + k
            p2 = v_base2 + (j + 1) * (n + 1) + (k + 1)
            p3 = v_base2 + j * (n + 1) + (k + 1)
            # Normal: (p1-p0) x (p3-p0) -> dy x dz = +x
            triangles.append((p0 + 1, p1 + 1, p2 + 1))
            triangles.append((p0 + 1, p2 + 1, p3 + 1))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# StrikeCEM Golden Benchmark: Canonical 90-deg Dihedral Corner Reflector\n")
        f.write(f"# Dimensions: a={a}m, b={b}m, Vertices={len(vertices)}, Triangles={len(triangles)}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in triangles:
            f.write(f"f {t[0]} {t[1]} {t[2]}\n")

    print(f"Generated {output_path}: {len(vertices)} vertices, {len(triangles)} triangles")

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "dihedral_90deg.obj")
    generate_canonical_dihedral(a=0.5, b=0.5, n=24, output_path=out)

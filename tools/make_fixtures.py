#!/usr/bin/env python3
"""Regenerate analytic golden-test meshes (deterministic, no dependencies).

- plate_fine.stl : 1x1 m square in z=0, 10x10 quads, normal +z.
- sphere_1m.stl  : r=1 UV sphere, 144x72, outward winding.
"""
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"


def write_stl(path, facets):
    with open(path, "w") as f:
        f.write(f"solid {path.stem}\n")
        for (n, a, b, c) in facets:
            f.write(f"facet normal {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}\n outer loop\n")
            for v in (a, b, c):
                f.write(f"  vertex {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
            f.write(" endloop\nendfacet\n")
        f.write(f"endsolid {path.stem}\n")


def plate_fine(n=10):
    facets = []
    h = 1.0 / n
    for i in range(n):
        for j in range(n):
            x0, y0 = i * h, j * h
            x1, y1 = x0 + h, y0 + h
            facets.append(((0, 0, 1), (x0, y0, 0), (x1, y0, 0), (x1, y1, 0)))
            facets.append(((0, 0, 1), (x0, y0, 0), (x1, y1, 0), (x0, y1, 0)))
    return facets


def sphere_1m(nlon=144, nlat=72):
    def pt(theta, phi):
        return (math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta))
    rings = []
    for i in range(1, nlat):
        theta = math.pi * i / nlat
        rings.append([pt(theta, 2 * math.pi * j / nlon) for j in range(nlon)])
    north, south = (0, 0, 1), (0, 0, -1)
    facets = []
    r0 = rings[0]
    for j in range(nlon):
        facets.append(((0, 0, 1), north, r0[j], r0[(j + 1) % nlon]))
    for r_top, r_bot in zip(rings[:-1], rings[1:]):
        for j in range(nlon):
            j1 = (j + 1) % nlon
            facets.append(((0, 0, 0), r_top[j], r_bot[j], r_bot[j1]))
            facets.append(((0, 0, 0), r_top[j], r_bot[j1], r_top[j1]))
    rl = rings[-1]
    for j in range(nlon):
        facets.append(((0, 0, -1), south, rl[(j + 1) % nlon], rl[j]))
    return facets


if __name__ == "__main__":
    plate = plate_fine()
    write_stl(FIX / "plate_fine.stl", plate)
    print(f"plate_fine.stl: {len(plate)} facets")
    sphere = sphere_1m()
    write_stl(FIX / "sphere_1m.stl", sphere)
    print(f"sphere_1m.stl: {len(sphere)} facets")

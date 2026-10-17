#!/usr/bin/env python3
"""
Generate the canonical NASA Almond Benchmark Target mesh (Woo et al., 1993).

Reference:
  A. C. Woo, H. T. G. Wang, M. J. Schuh, and M. L. Sanders,
  "EM Programmer's Notebook: Benchmark Radar Targets for the Validation of
   Computational Electromagnetics Programs", IEEE Antennas and Propagation
  Magazine, Vol. 35, No. 1, February 1993, pp. 84-89.

Dimensions:
  Total length d = 9.936 inches = 0.2523744 m.
  Coordinate system:
    - x-axis: along longitudinal axis of symmetry.
      - Rounded nose at t = -0.41667 (x = -0.105156 m)
      - Join junction at t = 0.0 (x = 0.0 m)
      - Sharp tip at t = +0.58333 (x = +0.147218 m)
    - y-axis: major transverse axis (width).
    - z-axis: minor transverse axis (height).
    - Aspect angle: az = 0 deg is tip-on (+x), az = 90 deg is broadside (+y),
      az = 180 deg is nose-on (-x).
"""

import math
import os
import sys

# Standard benchmark length: 9.936 inches in meters
ALMOND_D = 0.2523744


def almond_semi_axes(t: float, d: float = ALMOND_D):
    """
    Returns (ey, ez) semi-axes at normalized coordinate t = x / d.
    Front section: -0.41667 <= t <= 0
    Rear section:   0.0 <= t <= 0.58333
    """
    if t < 0.0:
        val = 1.0 - (t / 0.41667) ** 2
        val = max(0.0, val)
        ey = 0.193333 * d * math.sqrt(val)
        ez = 0.064444 * d * math.sqrt(val)
    else:
        val = 1.0 - (t / 2.08335) ** 2
        val = max(0.0, val)
        ey = 4.83345 * d * (math.sqrt(val) - 0.96)
        ez = 1.61115 * d * (math.sqrt(val) - 0.96)
        ey = max(0.0, ey)
        ez = max(0.0, ez)
    return ey, ez


def generate_nasa_almond_mesh(d: float = ALMOND_D, n_x_front: int = 40,
                              n_x_rear: int = 60, n_psi: int = 64,
                              output_path: str = "nasa_almond.obj"):
    """
    Generates a high-quality watertight surface mesh of the NASA Almond.
    Uses cosine-spaced rings along x to properly cluster triangles near the
    sharp tip and rounded nose.
    """
    vertices = []
    triangles = []

    # Ring coordinates t:
    # Front: -0.41667 to 0 (cosine spacing towards nose)
    t_front = []
    for i in range(n_x_front):
        frac = i / n_x_front
        theta = (1.0 - frac) * (math.pi / 2.0)
        t_val = -0.41667 * math.sin(theta)
        t_front.append(t_val)

    # Rear: 0 to 0.58333 (cosine spacing towards tip)
    t_rear = []
    for i in range(1, n_x_rear + 1):
        frac = i / n_x_rear
        theta = frac * (math.pi / 2.0)
        t_val = 0.58333 * math.sin(theta)
        t_rear.append(t_val)

    t_all = t_front + t_rear

    # Add vertices ring by ring
    # Nose pole point at index 1 (OBJ is 1-indexed)
    vertices.append((-0.41667 * d, 0.0, 0.0))
    nose_idx = 1

    ring_start_indices = []
    for t in t_all[1:-1]:
        x = t * d
        ey, ez = almond_semi_axes(t, d)
        start_idx = len(vertices) + 1
        ring_start_indices.append(start_idx)
        for j in range(n_psi):
            psi = 2.0 * math.pi * j / n_psi
            y = ey * math.cos(psi)
            z = ez * math.sin(psi)
            vertices.append((x, y, z))

    # Tip pole point
    tip_idx = len(vertices) + 1
    vertices.append((0.58333 * d, 0.0, 0.0))

    # Connect nose pole to first ring
    first_ring = ring_start_indices[0]
    for j in range(n_psi):
        j_next = (j + 1) % n_psi
        p1 = first_ring + j
        p2 = first_ring + j_next
        # Outward normal for front cap: (p2 - nose) x (p1 - nose) points along -x
        triangles.append((nose_idx, p2, p1))

    # Connect intermediate rings
    for r in range(len(ring_start_indices) - 1):
        r1 = ring_start_indices[r]
        r2 = ring_start_indices[r + 1]
        for j in range(n_psi):
            j_next = (j + 1) % n_psi
            p1 = r1 + j
            p2 = r1 + j_next
            p3 = r2 + j
            p4 = r2 + j_next
            # Quadrilateral (p1, p2, p4, p3) -> 2 triangles
            triangles.append((p1, p2, p4))
            triangles.append((p1, p4, p3))

    # Connect last ring to tip pole
    last_ring = ring_start_indices[-1]
    for j in range(n_psi):
        j_next = (j + 1) % n_psi
        p1 = last_ring + j
        p2 = last_ring + j_next
        # Outward normal for rear cap: points towards +x
        triangles.append((tip_idx, p1, p2))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# NASA Almond Benchmark Target (Woo et al., 1993)\n")
        f.write(f"# Dimensions: Length d={d:.6f}m, Vertices={len(vertices)}, Triangles={len(triangles)}\n")
        f.write("g nasa_almond\n")
        for v in vertices:
            f.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")
        for t in triangles:
            f.write(f"f {t[0]} {t[1]} {t[2]}\n")

    print(f"Generated {output_path}: {len(vertices)} vertices, {len(triangles)} triangles")
    return output_path


if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)
    target_obj = os.path.join(out_dir, "nasa_almond.obj")
    generate_nasa_almond_mesh(d=ALMOND_D, n_x_front=40, n_x_rear=60, n_psi=64, output_path=target_obj)

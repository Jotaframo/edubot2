import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class TimedPose:
    t: float
    x: float
    y: float
    z: float


def plane_to_xyz(plane: str, a: float, b: float, center: Sequence[float]) -> Tuple[float, float, float]:
    cx, cy, cz = center
    if plane == "xy":
        return cx + a, cy + b, cz
    if plane == "xz":
        return cx + a, cy, cz + b
    if plane == "yz":
        return cx, cy + a, cz + b
    raise ValueError(f"Unsupported plane '{plane}'. Use 'xy', 'xz', or 'yz'.")


def shape_point(shape: str, u: float, size: float) -> Tuple[float, float]:
    h = 0.5 * size
    if shape == "circle":
        theta = 2.0 * math.pi * u
        return h * math.cos(theta), h * math.sin(theta)

    if shape == "square":
        if u < 0.25:
            s = u / 0.25
            return -h + s * size, -h
        if u < 0.50:
            s = (u - 0.25) / 0.25
            return h, -h + s * size
        if u < 0.75:
            s = (u - 0.50) / 0.25
            return h - s * size, h
        s = (u - 0.75) / 0.25
        return -h, h - s * size

    if shape == "triangle":
        # Equilateral triangle (centered around origin in 2D local plane)
        r = size / math.sqrt(3.0)
        p0 = (0.0, r)
        p1 = (-0.5 * size, -0.5 * r)
        p2 = (0.5 * size, -0.5 * r)
        if u < 1.0 / 3.0:
            s = u / (1.0 / 3.0)
            return p0[0] + s * (p1[0] - p0[0]), p0[1] + s * (p1[1] - p0[1])
        if u < 2.0 / 3.0:
            s = (u - 1.0 / 3.0) / (1.0 / 3.0)
            return p1[0] + s * (p2[0] - p1[0]), p1[1] + s * (p2[1] - p1[1])
        s = (u - 2.0 / 3.0) / (1.0 / 3.0)
        return p2[0] + s * (p0[0] - p2[0]), p2[1] + s * (p0[1] - p2[1])

    raise ValueError(f"Unsupported shape '{shape}'. Use 'circle', 'square', or 'triangle'.")


def generate_trajectory(
    shape_name: str,
    center_xyz: Sequence[float],
    plane: str,
    size: float,
    duration_s: float,
    rate_hz: float,
) -> List[TimedPose]:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be > 0")
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be > 0")
    if size <= 0.0:
        raise ValueError("size must be > 0")

    n = max(8, int(duration_s * rate_hz))
    dt = duration_s / n
    traj: List[TimedPose] = []

    for i in range(n):
        t = i * dt
        u = i / n  # normalized phase [0, 1)
        a, b = shape_point(shape_name.lower(), u, size)
        x, y, z = plane_to_xyz(plane.lower(), a, b, center_xyz)
        traj.append(TimedPose(t=t, x=x, y=y, z=z))

    return traj

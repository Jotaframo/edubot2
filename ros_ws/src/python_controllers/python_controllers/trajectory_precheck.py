import csv
from dataclasses import dataclass
import statistics
from typing import List, Optional, Sequence, Tuple

from Inverse_Kinematics import analytical_ik_closed_form
from trajectory_generator import generate_trajectory


@dataclass
class CandidateResult:
    center_x: float
    center_y: float
    center_z: float
    size: float
    feasible: bool
    fail_reason: str
    fail_t: float
    fail_xyz: Tuple[float, float, float]
    max_joint_step: float
    valid_points: int
    total_points: int


@dataclass
class PrecheckConfig:
    shape: str = "square"
    plane: str = "xz"
    duration_s: float = 12.0
    rate_hz: float = 20.0
    pitch: float = 0.0
    l2: float = 0.11167
    l3: float = 0.16000
    l4: float = 0.15000

    center_x_min: float = 0.10
    center_x_max: float = 0.26
    center_x_step: float = 0.02
    center_y_min: float = -0.08
    center_y_max: float = 0.08
    center_y_step: float = 0.02
    center_z_min: float = 0.06
    center_z_max: float = 0.24
    center_z_step: float = 0.02

    size_min: float = 0.07
    size_max: float = 0.12
    size_step: float = 0.01

    max_joint_step: float = 0.1  # radians per trajectory point, for feasibility filtering
    top_k: int = 10
    output_csv: str = ""


def _frange(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    vals = []
    x = start
    eps = step * 1e-6
    while x <= stop + eps:
        vals.append(round(x, 6))
        x += step
    return vals


def _closest_solution(solutions: Sequence[Tuple[float, float, float, float]], last_q: Optional[Sequence[float]]):
    if not solutions:
        return None
    if last_q is None:
        return solutions[0]
    best = None
    best_cost = None
    for q in solutions:
        cost = sum((q[i] - last_q[i]) ** 2 for i in range(4))
        if best is None or cost < best_cost:
            best = q
            best_cost = cost
    return best


def evaluate_candidate(cfg: PrecheckConfig, center_xyz: Tuple[float, float, float], size: float) -> CandidateResult:
    traj = generate_trajectory(
        shape_name=cfg.shape,
        center_xyz=center_xyz,
        plane=cfg.plane,
        size=size,
        duration_s=cfg.duration_s,
        rate_hz=cfg.rate_hz,
    )

    last_q = None
    max_step = 0.0
    valid_points = 0
    for p in traj:
        out = analytical_ik_closed_form(
            p.x, p.y, p.z, cfg.pitch, cfg.l2, cfg.l3, cfg.l4,
            track_invalid=True
        )
        valid = out["valid_solutions"]
        if not valid:
            reason = "geometry_unreachable"
            for r in out["invalid_reasons"]:
                if r.get("code") == "joint_limit_violation":
                    reason = "joint_limit_violation"
                    break
            return CandidateResult(
                center_x=center_xyz[0], center_y=center_xyz[1], center_z=center_xyz[2],
                size=size, feasible=False, fail_reason=reason,
                fail_t=p.t, fail_xyz=(p.x, p.y, p.z), max_joint_step=max_step,
                valid_points=valid_points, total_points=len(traj),
            )

        q = _closest_solution(valid, last_q)
        if last_q is not None:
            step = max(abs(q[i] - last_q[i]) for i in range(4))
            if step > max_step:
                max_step = step
            if step > cfg.max_joint_step:
                return CandidateResult(
                    center_x=center_xyz[0], center_y=center_xyz[1], center_z=center_xyz[2],
                    size=size, feasible=False, fail_reason="joint_step_limit",
                    fail_t=p.t, fail_xyz=(p.x, p.y, p.z), max_joint_step=max_step,
                    valid_points=valid_points, total_points=len(traj),
                )
        last_q = q
        valid_points += 1

    return CandidateResult(
        center_x=center_xyz[0], center_y=center_xyz[1], center_z=center_xyz[2],
        size=size, feasible=True, fail_reason="", fail_t=-1.0, fail_xyz=(0.0, 0.0, 0.0),
        max_joint_step=max_step, valid_points=valid_points, total_points=len(traj),
    )


def _write_csv(path: str, results: Sequence[CandidateResult]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "center_x", "center_y", "center_z", "size", "feasible",
            "fail_reason", "fail_t", "fail_x", "fail_y", "fail_z",
            "max_joint_step", "valid_points", "total_points"
        ])
        for r in results:
            w.writerow([
                r.center_x, r.center_y, r.center_z, r.size, int(r.feasible),
                r.fail_reason, r.fail_t, r.fail_xyz[0], r.fail_xyz[1], r.fail_xyz[2],
                r.max_joint_step, r.valid_points, r.total_points
            ])


def main():
    # Edit these values directly instead of using CLI flags.
    cfg = PrecheckConfig()

    xs = _frange(cfg.center_x_min, cfg.center_x_max, cfg.center_x_step)
    ys = _frange(cfg.center_y_min, cfg.center_y_max, cfg.center_y_step)
    zs = _frange(cfg.center_z_min, cfg.center_z_max, cfg.center_z_step)
    sizes = _frange(cfg.size_min, cfg.size_max, cfg.size_step)

    total = len(xs) * len(ys) * len(zs) * len(sizes)
    print(
        f"Precheck start: shape={cfg.shape}, plane={cfg.plane}, "
        f"candidates={total}, sample_rate={cfg.rate_hz:.1f} Hz"
    )

    results: List[CandidateResult] = []
    done = 0
    for cx in xs:
        for cy in ys:
            for cz in zs:
                for size in sizes:
                    done += 1
                    res = evaluate_candidate(cfg, (cx, cy, cz), size)
                    results.append(res)
                    if done % 100 == 0:
                        print(f"  checked {done}/{total}")

    feasible = [r for r in results if r.feasible]
    if feasible:
        median_size = statistics.median([r.size for r in feasible])
        feasible_sorted = sorted(
            feasible,
            key=lambda r: (abs(r.size - median_size), r.max_joint_step),
        )
    else:
        median_size = None
        feasible_sorted = []

    print(f"\nDone. feasible={len(feasible)} / {len(results)}")
    if feasible_sorted:
        print(f"Median feasible size: {median_size:.3f} m")
        print("\nTop feasible candidates:")
        for r in feasible_sorted[: cfg.top_k]:
            print(
                "  center=({:.3f},{:.3f},{:.3f}) size={:.3f} "
                "delta_to_median={:.3f} max_joint_step={:.4f}".format(
                    r.center_x, r.center_y, r.center_z, r.size,
                    abs(r.size - median_size), r.max_joint_step
                )
            )
            print(
                "    ros2 run python_controllers shape_follower --ros-args "
                f"-p shape:={cfg.shape} -p plane:={cfg.plane} "
                f"-p center_x:={r.center_x} -p center_y:={r.center_y} -p center_z:={r.center_z} "
                f"-p size:={r.size} -p duration_s:={cfg.duration_s} -p rate_hz:={cfg.rate_hz}"
            )
    else:
        print("No feasible candidates found with current search bounds.")

    if cfg.output_csv:
        _write_csv(cfg.output_csv, results)
        print(f"\nWrote report: {cfg.output_csv}")


if __name__ == "__main__":
    main()

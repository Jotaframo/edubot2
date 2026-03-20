import csv
from dataclasses import dataclass, field, replace
import time
import sys

import numpy as np

try:
    from python_controllers.Inverse_Kinematics_Numerical import ik_coordinate_descent
    from python_controllers.trajectory_generator import generate_trajectory
except ModuleNotFoundError:
    from Inverse_Kinematics_Numerical import ik_coordinate_descent
    from trajectory_generator import generate_trajectory


@dataclass
class OfflineTrajectoryConfig:
    shape: str = "square"
    plane: str = "xz"
    size: float = 0.008
    duration_s: float = 60.0
    rate_hz: float = 1
    center_x: float = -3.4033513892756203e-06
    center_y: float = 0.4485728271029048
    center_z: float = 0.1646111173500898
    pos_tol: float = 5e-3
    rot_tol: float = np.deg2rad(1.0)
    max_joint_jump: float = 0.20
    optimizer_max_iter: int = 200
    initial_joint_guess: list[float] = field(
        default_factory=lambda: [0.0, -1.2, 1.0, 0.6, 0.0]
    )
    
    output_csv: str = "/tmp/offline_joint_trajectory.csv"


@dataclass
class SweepConfig:
    center_x_min: float = 0.1 - 0.1
    center_x_max: float = 0.1 + 0.15
    center_x_step: float = 0.02
    center_y_min: float = 0.44 - 0.2
    center_y_max: float = 0.44 + 0.1
    center_y_step: float = 0.02
    center_z_min: float = 0.06
    center_z_max: float = 0.28
    center_z_step: float = 0.02
    size_max: float = 0.12
    size_min: float = 0.04
    size_step: float = 0.01
    report_csv: str = "/tmp/offline_trajectory_sweep.csv"


def _format_seconds(seconds):
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def _render_progress(completed, total, start_time, width=28):
    elapsed = time.perf_counter() - start_time
    ratio = 0.0 if total <= 0 else completed / total
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    percent = 100.0 * ratio
    if completed > 0 and total > 0:
        eta = elapsed * (total - completed) / completed
        eta_text = _format_seconds(eta)
    else:
        eta_text = "--"
    return (
        f"\r[{bar}] {completed}/{total} centers "
        f"({percent:5.1f}%) elapsed={_format_seconds(elapsed)} eta={eta_text}"
    )


def _frange(start, stop, step):
    if step <= 0:
        raise ValueError("step must be > 0")
    vals = []
    x = start
    eps = step * 1e-6
    while x <= stop + eps:
        vals.append(round(x, 6))
        x += step
    return vals


def _descending_sizes(size_max, size_min, size_step):
    return list(reversed(_frange(size_min, size_max, size_step)))


def _solve_pose(point, cfg, q_prev):
    result = ik_coordinate_descent(
        point.x,
        point.y,
        point.z,
        0.0,
        0.0,
        0.0,
        q_init=q_prev,
        max_iters=cfg.optimizer_max_iter,
        pos_tol=cfg.pos_tol,
        rot_tol=cfg.rot_tol,
        optimize_orientation=False,
    )
    return {
        "success": bool(result["success"]),
        "q_raw": np.asarray(result["q_raw"], dtype=float),
        "q": np.asarray(result["q"], dtype=float),
        "pos_error_raw": float(result["pos_error_raw"]),
        "rot_error_raw": float(result["rot_error_raw"]),
        "error_raw": float(result["error_raw"]),
    }


def _generate_points(cfg):
    return generate_trajectory(
        shape_name=cfg.shape,
        center_xyz=(cfg.center_x, cfg.center_y, cfg.center_z),
        plane=cfg.plane,
        size=cfg.size,
        duration_s=cfg.duration_s,
        rate_hz=cfg.rate_hz,
    )


def solve_offline_trajectory(cfg: OfflineTrajectoryConfig, log_every=50):
    trajectory = _generate_points(cfg)
    print(
        "Generated {} trajectory points to solve for shape='{}' plane='{}' size={:.3f}".format(
            len(trajectory), cfg.shape, cfg.plane, cfg.size
        )
    )

    solved_rows = []
    q_prev = np.asarray(cfg.initial_joint_guess, dtype=float)

    for idx, point in enumerate(trajectory):
        chosen = _solve_pose(point, cfg, q_prev)

        if idx == 0 or (idx + 1) % log_every == 0 or idx == len(trajectory) - 1:
            print(
                f"[{idx + 1:03d}/{len(trajectory)}] "
                f"xyz=({point.x:.3f}, {point.y:.3f}, {point.z:.3f}) "
                f"success={chosen['success']} pos={chosen['pos_error_raw']:.6f} "
                f"rot={chosen['rot_error_raw']:.6f}"
            )

        if not chosen["success"]:
            raise RuntimeError(
                "failed at idx={} t={:.3f} xyz=({:.3f}, {:.3f}, {:.3f}) "
                "pos_error={:.6f} rot_error={:.6f}".format(
                    idx,
                    point.t,
                    point.x,
                    point.y,
                    point.z,
                    chosen["pos_error_raw"],
                    chosen["rot_error_raw"],
                )
            )

        jump = float(np.max(np.abs(chosen["q_raw"] - q_prev)))
        if jump > cfg.max_joint_jump:
            raise RuntimeError(
                "joint jump {:.3f} exceeds max_joint_jump at idx={} t={:.3f}".format(
                    jump, idx, point.t
                )
            )

        q_prev = chosen["q_raw"].copy()
        solved_rows.append(
            {
                "t": point.t,
                "x": point.x,
                "y": point.y,
                "z": point.z,
                "q1": q_prev[0],
                "q2": q_prev[1],
                "q3": q_prev[2],
                "q4": q_prev[3],
                "q5": q_prev[4],
                "pos_error": chosen["pos_error_raw"],
                "rot_error": chosen["rot_error_raw"],
                "error": chosen["error_raw"],
            }
        )
    return solved_rows


def _evaluate_candidate(chain, cfg: OfflineTrajectoryConfig):
    trajectory = _generate_points(cfg)
    q_prev = np.asarray(cfg.initial_joint_guess, dtype=float)
    worst_pos = 0.0
    worst_rot = 0.0
    worst_total = 0.0
    max_jump = 0.0

    for idx, point in enumerate(trajectory):
        chosen = _solve_pose(point, cfg, q_prev)
        worst_pos = max(worst_pos, chosen["pos_error_raw"])
        worst_rot = max(worst_rot, chosen["rot_error_raw"])
        worst_total = max(worst_total, chosen["error_raw"])

        if not chosen["success"]:
            return {
                "success": False,
                "fail_reason": "ik_failed",
                "fail_idx": idx,
                "fail_t": point.t,
                "fail_x": point.x,
                "fail_y": point.y,
                "fail_z": point.z,
                "worst_pos_error": worst_pos,
                "worst_rot_error": worst_rot,
                "worst_total_error": worst_total,
                "max_joint_jump": max_jump,
                "solved_points": idx,
            }

        jump = float(np.max(np.abs(chosen["q_raw"] - q_prev)))
        max_jump = max(max_jump, jump)
        if jump > cfg.max_joint_jump:
            return {
                "success": False,
                "fail_reason": "joint_jump",
                "fail_idx": idx,
                "fail_t": point.t,
                "fail_x": point.x,
                "fail_y": point.y,
                "fail_z": point.z,
                "worst_pos_error": worst_pos,
                "worst_rot_error": worst_rot,
                "worst_total_error": worst_total,
                "max_joint_jump": max_jump,
                "solved_points": idx + 1,
            }

        q_prev = chosen["q_raw"].copy()

    return {
        "success": True,
        "fail_reason": "",
        "fail_idx": -1,
        "fail_t": -1.0,
        "fail_x": 0.0,
        "fail_y": 0.0,
        "fail_z": 0.0,
        "worst_pos_error": worst_pos,
        "worst_rot_error": worst_rot,
        "worst_total_error": worst_total,
        "max_joint_jump": max_jump,
        "solved_points": len(trajectory),
    }


def sweep_workspace(base_cfg: OfflineTrajectoryConfig, sweep_cfg: SweepConfig):
    sweep_start = time.perf_counter()
    xs = _frange(sweep_cfg.center_x_min, sweep_cfg.center_x_max, sweep_cfg.center_x_step)
    ys = _frange(sweep_cfg.center_y_min, sweep_cfg.center_y_max, sweep_cfg.center_y_step)
    zs = _frange(sweep_cfg.center_z_min, sweep_cfg.center_z_max, sweep_cfg.center_z_step)
    sizes = _descending_sizes(
        sweep_cfg.size_max, sweep_cfg.size_min, sweep_cfg.size_step
    )

    results = []
    total_centers = len(xs) * len(ys) * len(zs)
    center_idx = 0

    print(
        f"Sweep start: centers={total_centers}, sizes={len(sizes)}, "
        f"plane={base_cfg.plane}, shape={base_cfg.shape}"
    )
    sys.stdout.write(_render_progress(0, total_centers, sweep_start))
    sys.stdout.flush()

    for cx in xs:
        for cy in ys:
            for cz in zs:
                center_idx += 1
                best = None
                last_fail = None

                for size in sizes:
                    cfg = replace(
                        base_cfg,
                        center_x=cx,
                        center_y=cy,
                        center_z=cz,
                        size=size,
                    )
                    out = _evaluate_candidate(None, cfg)
                    if out["success"]:
                        best = out
                        break
                    last_fail = out

                row = {
                    "center_x": cx,
                    "center_y": cy,
                    "center_z": cz,
                    "feasible": int(best is not None),
                    "best_size": best and size or 0.0,
                    "worst_pos_error": (best or last_fail)["worst_pos_error"],
                    "worst_rot_error": (best or last_fail)["worst_rot_error"],
                    "worst_total_error": (best or last_fail)["worst_total_error"],
                    "max_joint_jump": (best or last_fail)["max_joint_jump"],
                    "fail_reason": "" if best is not None else last_fail["fail_reason"],
                    "fail_idx": -1 if best is not None else last_fail["fail_idx"],
                    "fail_t": -1.0 if best is not None else last_fail["fail_t"],
                    "fail_x": 0.0 if best is not None else last_fail["fail_x"],
                    "fail_y": 0.0 if best is not None else last_fail["fail_y"],
                    "fail_z": 0.0 if best is not None else last_fail["fail_z"],
                    "solved_points": (best or last_fail)["solved_points"],
                }
                results.append(row)
                sys.stdout.write(_render_progress(center_idx, total_centers, sweep_start))
                sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return results


def write_trajectory_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t",
                "x",
                "y",
                "z",
                "q1",
                "q2",
                "q3",
                "q4",
                "q5",
                "pos_error",
                "rot_error",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_sweep_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "center_x",
                "center_y",
                "center_z",
                "feasible",
                "best_size",
                "worst_pos_error",
                "worst_rot_error",
                "worst_total_error",
                "max_joint_jump",
                "fail_reason",
                "fail_idx",
                "fail_t",
                "fail_x",
                "fail_y",
                "fail_z",
                "solved_points",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    base_cfg = OfflineTrajectoryConfig()
    sweep_cfg = SweepConfig()
    rows = sweep_workspace(base_cfg, sweep_cfg)
    write_sweep_csv(sweep_cfg.report_csv, rows)
    feasible = sum(int(row["feasible"]) for row in rows)
    print(f"\nSweep complete. feasible_centers={feasible}/{len(rows)}")
    print(f"Wrote {sweep_cfg.report_csv}")


if __name__ == "__main__":
    main()

import csv
from dataclasses import replace
from datetime import datetime
from pathlib import Path

try:
    from python_controllers.offline_trajectory_solver import (
        OfflineTrajectoryConfig,
        solve_offline_trajectory,
        write_trajectory_csv,
    )
except ModuleNotFoundError:
    from offline_trajectory_solver import (
        OfflineTrajectoryConfig,
        solve_offline_trajectory,
        write_trajectory_csv,
    )


SWEEP_CSV = "/tmp/offline_trajectory_sweep.csv"


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_float(value):
    return float(value.strip())


def _load_feasible_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["feasible"]) != 1:
                continue
            if _safe_float(row["best_size"]) <= 0.0:
                continue
            rows.append(row)
    return rows


def _manifest_rows(export_dir, rows, base_cfg):
    feasible = _load_feasible_rows(rows)
    manifest = []
    total = len(feasible)
    if total == 0:
        print(f"No feasible rows found in {rows}")
        return manifest

    print(f"Exporting {total} feasible trajectories from {rows}")
    for idx, row in enumerate(feasible, start=1):
        cfg = replace(
            base_cfg,
            center_x=_safe_float(row["center_x"]),
            center_y=_safe_float(row["center_y"]),
            center_z=_safe_float(row["center_z"]),
            size=_safe_float(row["best_size"]),
        )
        print(
            f"[{idx}/{total}] center=({cfg.center_x:.3f}, {cfg.center_y:.3f}, {cfg.center_z:.3f}) "
            f"size={cfg.size:.3f}"
        )
        traj_rows = solve_offline_trajectory(cfg, log_every=200)
        name = (
            f"traj_{idx:04d}_cx_{cfg.center_x:+.3f}_cy_{cfg.center_y:+.3f}"
            f"_cz_{cfg.center_z:+.3f}_size_{cfg.size:.3f}.csv"
        ).replace("+", "p").replace("-", "m")
        out_path = export_dir / name
        write_trajectory_csv(out_path, traj_rows)
        manifest.append(
            {
                "index": idx,
                "center_x": cfg.center_x,
                "center_y": cfg.center_y,
                "center_z": cfg.center_z,
                "size": cfg.size,
                "num_points": len(traj_rows),
                "trajectory_csv": str(out_path),
            }
        )
    return manifest


def _write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "center_x",
                "center_y",
                "center_z",
                "size",
                "num_points",
                "trajectory_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    export_dir = Path(f"/tmp/feasible_trajectories_{_timestamp()}")
    export_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = OfflineTrajectoryConfig()
    manifest_rows = _manifest_rows(export_dir, SWEEP_CSV, base_cfg)
    manifest_path = export_dir / "manifest.csv"
    _write_manifest(manifest_path, manifest_rows)

    print(f"Wrote {len(manifest_rows)} trajectory CSVs to {export_dir}")
    print(f"Wrote manifest {manifest_path}")


if __name__ == "__main__":
    main()

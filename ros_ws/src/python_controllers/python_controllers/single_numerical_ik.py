from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ikpy.chain import Chain

try:
    from python_controllers.Forward_Kinematics_FINAL import create_tf_matrix
except ModuleNotFoundError:
    from Forward_Kinematics_FINAL import create_tf_matrix


ACTIVE_LINKS_MASK = [False, False, True, True, True, True, True, False]
JOINT_SLICE = slice(2, 7)
URDF_PATH = Path(__file__).resolve().parents[2] / "lerobot" / "urdf" / "lerobot.urdf"



@dataclass
class IKSolveResult:
    feasible: bool
    q_raw: np.ndarray
    q: np.ndarray
    pos_error_raw: float
    rot_error_raw: float
    error_raw: float


def _load_chain():
    return Chain.from_urdf_file(
        str(URDF_PATH),
        base_elements=["world"],
        active_links_mask=ACTIVE_LINKS_MASK,
    )


def _full_joint_state(chain, q_init):
    q_full = np.zeros(len(chain.links), dtype=float)
    if q_init is not None:
        q_full[JOINT_SLICE] = np.asarray(q_init, dtype=float)
    return q_full


def solve_single_pose_ikpy(
    x,
    y,
    z,
    roll,
    pitch,
    yaw,
    q_init=None,
    pos_tol=5e-3,
    rot_tol=np.deg2rad(1.0),
    max_iter=200,
):
    chain = _load_chain()
    target = create_tf_matrix(x, y, z, roll, pitch, yaw)
    q_full = chain.inverse_kinematics_frame(
        target,
        initial_position=_full_joint_state(chain, q_init),
        orientation_mode="all",
        max_iter=max_iter,
    )
    fk = chain.forward_kinematics(q_full)

    pos_error = float(np.linalg.norm(fk[:3, 3] - target[:3, 3]))
    rot_error = float(np.linalg.norm(fk[:3, :3] - target[:3, :3]))
    total_error = float(pos_error + 0.2 * rot_error)
    q_raw = np.asarray(q_full[JOINT_SLICE], dtype=float)

    return IKSolveResult(
        feasible=bool(pos_error <= pos_tol and rot_error <= rot_tol),
        q_raw=q_raw,
        q=np.round(q_raw, 4),
        pos_error_raw=pos_error,
        rot_error_raw=rot_error,
        error_raw=total_error,
    )


if __name__ == "__main__":
    # These are the test poses from Inverse_Kinematics_Numerical.py.
    test_poses = [
        ("I", [0.2000, 0.2000, 0.2000, 0.0000, 1.5700, 0.6500]),
        ("II", [0.2000, 0.1000, 0.4000, 0.0000, 0.0000, -1.5700]),
        ("III", [0.0000, 0.0000, 0.4000, 0.0000, -0.7850, 1.5700]),
        ("IV_a", [0.0000, 0.0000, 0.0700, 3.1410, 0.0000, 0.0000]),
        ("IV_b", [0.0000, 0.0452, 0.4500, -0.7850, 0.0000, 3.1410]),
    ]

    print("Single-pose numerical IK with ikpy\n")
    print("Using the 6D test poses from Inverse_Kinematics_Numerical.py.\n")

    for label, pose in test_poses:
        x, y, z, roll, pitch, yaw = pose
        result = solve_single_pose_ikpy(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
        print(f"Pose {label}: {pose}")
        print(f"  feasible={result.feasible}")
        print(
            f"  pos_error={result.pos_error_raw:.6f} "
            f"rot_error={result.rot_error_raw:.6f} total={result.error_raw:.6f}"
        )
        print(f"  q={result.q}")
        print()

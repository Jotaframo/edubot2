from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

try:
    from python_controllers.Forward_Kinematics_FINAL import create_tf_matrix, forward_kinematics_full
except ModuleNotFoundError:
    from Forward_Kinematics_FINAL import create_tf_matrix, forward_kinematics_full

NUM_ACTIVE_JOINTS = 5

JOINT_LIMITS = [
    (-2.0, 2.0),       # q1
    (-1.57, 1.57),     # q2
    (-1.58, 1.58),     # q3
    (-1.57, 1.57),     # q4
    (-np.pi, np.pi)    # q5
]

@dataclass
class IKSolveResult:
    feasible: bool
    q_raw: np.ndarray
    q: np.ndarray
    pos_error_raw: float
    rot_error_raw: float
    error_raw: float


def _ik_objective_function(q, target_matrix):
    current_tf = forward_kinematics_full(q[0], q[1], q[2], q[3], q[4])
    pos_error = np.linalg.norm(current_tf[:3, 3] - target_matrix[:3, 3])
    rot_error = np.linalg.norm(current_tf[:3, :3] - target_matrix[:3, :3])
    return pos_error + 0.2 * rot_error


def solve_multiple_guesses_scipy(
    x, y, z, roll, pitch, yaw,
    num_guesses=10,
    pos_tol=5e-3,
    rot_tol=np.deg2rad(1.0),
    max_iter=200,
):
    target_matrix = create_tf_matrix(x, y, z, roll, pitch, yaw)
    unique_solutions = []
    
    # Generate starting guesses: First one is always zero, the rest are random
    seeds = [np.zeros(NUM_ACTIVE_JOINTS)]
    for _ in range(num_guesses - 1):
        random_q = [np.random.uniform(low, high) for low, high in JOINT_LIMITS]
        seeds.append(np.array(random_q))

    for q_init in seeds:
        result = minimize(
            _ik_objective_function,
            q_init,
            args=(target_matrix,),
            method='L-BFGS-B',
            bounds=JOINT_LIMITS,
            options={'maxiter': max_iter, 'disp': False}
        )
        
        q_raw = result.x
        
        final_tf = forward_kinematics_full(q_raw[0], q_raw[1], q_raw[2], q_raw[3], q_raw[4])
        pos_error = float(np.linalg.norm(final_tf[:3, 3] - target_matrix[:3, 3]))
        rot_error = float(np.linalg.norm(final_tf[:3, :3] - target_matrix[:3, :3]))
        total_error = float(pos_error + 0.2 * rot_error)
        
        if pos_error <= pos_tol and rot_error <= rot_tol:
            # Check if this solution is already in our list (within a small tolerance)
            is_duplicate = False
            for existing_sol in unique_solutions:
                if np.allclose(existing_sol.q_raw, q_raw, atol=1e-2):
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_solutions.append(
                    IKSolveResult(
                        feasible=True,
                        q_raw=q_raw,
                        q=np.round(q_raw, 4),
                        pos_error_raw=pos_error,
                        rot_error_raw=rot_error,
                        error_raw=total_error,
                    )
                )
                
    return unique_solutions


if __name__ == "__main__":
    test_poses = [
        ("I", [0.2000, 0.2000, 0.2000, 0.0000, 1.5700, 0.6500]),
        ("II", [0.2000, 0.1000, 0.4000, 0.0000, 0.0000, -1.5700]),
        ("III", [0.0000, 0.0000, 0.4000, 0.0000, -0.7850, 1.5700]),
        ("IV_a", [0.0000, 0.0000, 0.0700, 3.1410, 0.0000, 0.0000]),
        ("IV_b", [0.0000, 0.0452, 0.4500, -0.7850, 0.0000, 3.1410]),
    ]

    print("Multi-Start IK with Scipy\n")

    print("| Pose | Sol # | q1 | q2 | q3 | q4 | q5 |")
    print("|---|---|---|---|---|---|---|")

    for label, pose in test_poses:
        x, y, z, roll, pitch, yaw = pose
        
        # Test 10 different starting configurations per pose
        results = solve_multiple_guesses_scipy(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, num_guesses=10
        )
        
        if not results:
            print(f"| {label} | None | N/A | N/A | N/A | N/A | N/A |")
            
        for idx, res in enumerate(results):
            q_f = [f"{val:.4f}" for val in res.q]
            print(f"| {label} | {idx + 1} | {q_f[0]} | {q_f[1]} | {q_f[2]} | {q_f[3]} | {q_f[4]} |")
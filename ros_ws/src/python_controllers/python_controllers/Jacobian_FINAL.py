"""
Numerical (finite-difference) Jacobian for the 5-DOF manipulator.

Used for online control because it is faster than the symbolic version
and only needs the FK function.
Also contains SVD analysis of assignment poses for report.

"""

import numpy as np

try:
    from python_controllers.Forward_Kinematics_FINAL import forward_kinematics_full
except ModuleNotFoundError:
    from Forward_Kinematics_FINAL import forward_kinematics_full


def fk_xyz_final(q, q5=0.0):
    """End-effector [x, y, z] for a 4-DOF joint vector q (q5 fixed)."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("q must have shape (4,)")

    T = forward_kinematics_full(q[0], q[1], q[2], q[3], float(q5))
    return np.asarray(T[:3, 3], dtype=float)


def jacobian_finite_difference_final(q, eps=1e-6, q5=0.0):
    """3x4 linear Jacobian via central finite differences."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("q must have shape (4,)")

    jac = np.zeros((3, 4), dtype=float)
    for i in range(4):
        dq = np.zeros(4, dtype=float)
        dq[i] = eps
        p_plus = fk_xyz_final(q + dq, q5=q5)
        p_minus = fk_xyz_final(q - dq, q5=q5)
        jac[:, i] = (p_plus - p_minus) / (2.0 * eps)
    return jac


def jacobian_svd_and_rank_final(q, eps=1e-6, rank_tol=1e-6, q5=0.0):
    """Return dict with Jacobian, singular values, and numerical rank."""
    jac = jacobian_finite_difference_final(q, eps=eps, q5=q5)
    singular_values = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(singular_values > rank_tol))
    return {
        "q": np.asarray(q, dtype=float),
        "jacobian": jac,
        "singular_values": singular_values,
        "rank": rank,
    }


def analyze_assignment_pose_jacobians_final(eps=1e-6, rank_tol=1e-6, q5=0.0):
    """Evaluate Jacobian SVD at each assignment pose via numerical IK."""
    try:
        from python_controllers.Inverse_Kinematics_Numerical import ik_coordinate_descent_multi_start
    except ModuleNotFoundError:
        from Inverse_Kinematics_Numerical import ik_coordinate_descent_multi_start

    # Assignment poses from AE4324: [x, y, z, rot_x, rot_y, rot_z]
    assignment_poses = [
        ("I", [0.2, 0.2, 0.2, 0.0, 1.57, 0.65]),
        ("II", [0.2, 0.1, 0.4, 0.0, 0.0, -1.57]),
        ("III", [0.0, 0.0, 0.4, 0.0, -0.785, 1.57]),
        ("IV", [0.0, 0.0, 0.07, 3.141, 0.0, 0.0]),
        ("V", [0.0, 0.0452, 0.45, -0.785, 0.0, 3.141]),
    ]

    results = []
    for label, pose in assignment_poses:
        x, y, z, rot_x, rot_y, rot_z = pose
        
        # Use numerical multi-start IK with 15 random initial guesses
        ik_solutions = ik_coordinate_descent_multi_start(
            x, y, z, rot_x, rot_y, rot_z,
            num_random=15,
            seed=42,
            unique_decimals=3,
            max_iters=5000,
            pos_tol=5e-3,
            rot_tol=np.deg2rad(1.0),
        )
        
        valid_solutions = [sol for sol in ik_solutions if sol["success"]]
        invalid_solutions = [sol for sol in ik_solutions if not sol["success"]]
        
        pose_result = {
            "label": label,
            "pose": pose,
            "num_valid_solutions": len(valid_solutions),
            "num_invalid_solutions": len(invalid_solutions),
            "solutions": [],
        }

        # Analyze valid solutions
        for i, ik_result in enumerate(valid_solutions):
            q_sol = ik_result["q_raw"][:4]  # Use first 4 DOFs for Jacobian
            analysis = jacobian_svd_and_rank_final(q_sol, eps=eps, rank_tol=rank_tol, q5=q5)
            pose_result["solutions"].append(
                {
                    "index": i,
                    "valid": True,
                    "q": ik_result["q"][:4],
                    "q_raw": q_sol,
                    "rank": analysis["rank"],
                    "singular_values": analysis["singular_values"],
                    "jacobian": analysis["jacobian"],
                    "pos_error": ik_result["pos_error"],
                    "rot_error": ik_result["rot_error"],
                }
            )

        # Analyze invalid solutions (best approximations)
        for i, ik_result in enumerate(invalid_solutions):
            q_sol = ik_result["q_raw"][:4]
            analysis = jacobian_svd_and_rank_final(q_sol, eps=eps, rank_tol=rank_tol, q5=q5)
            pose_result["solutions"].append(
                {
                    "index": len(valid_solutions) + i,
                    "valid": False,
                    "q": ik_result["q"][:4],
                    "q_raw": q_sol,
                    "rank": analysis["rank"],
                    "singular_values": analysis["singular_values"],
                    "jacobian": analysis["jacobian"],
                    "pos_error": ik_result["pos_error"],
                    "rot_error": ik_result["rot_error"],
                }
            )
        results.append(pose_result)
    return results


def print_assignment_pose_jacobian_analysis_final(eps=1e-6, rank_tol=1e-6, q5=0.0):
    """print SVD results for all assignment poses."""
    print("Assignment pose Jacobian analysis (FINAL FK chain):\n")
    results = analyze_assignment_pose_jacobians_final(
        eps=eps, rank_tol=rank_tol, q5=q5
    )
    for pose_result in results:
        label = pose_result["label"]
        pose = pose_result["pose"]
        num_valid = pose_result["num_valid_solutions"]
        num_invalid = pose_result["num_invalid_solutions"]
        
        print(f"{label}. pose={pose}")
        print(f"   Found {num_valid} valid solution(s) and {num_invalid} approximate solution(s)")
        
        if not pose_result["solutions"]:
            print("   No solutions found")
            continue
            
        for sol in pose_result["solutions"]:
            validity = "VALID" if sol["valid"] else "APPROX"
            sv = ", ".join(f"{float(v):.6f}" for v in sol["singular_values"])
            print(
                f"   Solution {sol['index']} [{validity}]: "
                f"rank={sol['rank']}, "
                f"pos_err={sol['pos_error']:.4f}, "
                f"rot_err={sol['rot_error']:.4f}, "
                f"SV=[{sv}]"
            )
        print()


if __name__ == "__main__":
    print_assignment_pose_jacobian_analysis_final()

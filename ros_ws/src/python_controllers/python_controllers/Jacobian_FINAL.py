import numpy as np

try:
    from python_controllers.Forward_Kinematics_FINAL import forward_kinematics_full
except ModuleNotFoundError:
    from Forward_Kinematics_FINAL import forward_kinematics_full


def fk_xyz_final(q, q5=0.0):
    """
    Return end-effector XYZ using the final 5-joint FK chain.

    `q` is the 4-DOF arm state [q1, q2, q3, q4]; q5 defaults to 0.0 because
    the constant velocity follower does not actively control wrist roll.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("q must have shape (4,)")

    T = forward_kinematics_full(q[0], q[1], q[2], q[3], float(q5))
    return np.asarray(T[:3, 3], dtype=float)


def jacobian_finite_difference_final(q, eps=1e-6, q5=0.0):
    """
    Compute the 3x4 linear Jacobian numerically via central differences,
    using the final FK chain for the first four joints and a fixed q5.
    """
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
    """
    Keep the same assignment-pose evaluation workflow as Forward_Kinematics.py,
    but evaluate Jacobians using the final FK chain.
    """
    try:
        from python_controllers.Inverse_Kinematics_Closed_Form import analytical_ik_closed_form
    except ModuleNotFoundError:
        from Inverse_Kinematics_Closed_Form import analytical_ik_closed_form

    l2, l3, l4 = 0.11167, 0.16000, 0.15000
    assignment_poses = [
        ("I", [0.2, 0.2, 0.2, 1.57, 0.0]),
        ("II", [0.2, 0.1, 0.4, 0.0, 1.57]),
        ("III", [0.0, 0.0, 0.45, 0.785, 0.785]),
        ("IV", [0.0, 0.0, 0.07, 3.141, 0.0]),
        ("V", [0.0, 0.0452, 0.45, 0.785, 3.141]),
    ]

    results = []
    for label, pose in assignment_poses:
        x, y, z, pitch, roll = pose
        out = analytical_ik_closed_form(x, y, z, pitch, l2, l3, l4, track_invalid=True)
        valid_solutions = out["valid_solutions"]
        invalid_solutions = [q for q in out["invalid_solutions"] if q is not None]
        pose_result = {
            "label": label,
            "pose": pose,
            "num_valid_solutions": len(valid_solutions),
            "num_invalid_solutions": len(invalid_solutions),
            "solutions": [],
        }

        for i, q_sol in enumerate(valid_solutions):
            analysis = jacobian_svd_and_rank_final(q_sol, eps=eps, rank_tol=rank_tol, q5=q5)
            pose_result["solutions"].append(
                {
                    "index": i,
                    "valid": True,
                    "q": analysis["q"],
                    "rank": analysis["rank"],
                    "singular_values": analysis["singular_values"],
                    "jacobian": analysis["jacobian"],
                }
            )

        for i, q_sol in enumerate(invalid_solutions):
            analysis = jacobian_svd_and_rank_final(q_sol, eps=eps, rank_tol=rank_tol, q5=q5)
            pose_result["solutions"].append(
                {
                    "index": len(valid_solutions) + i,
                    "valid": False,
                    "q": analysis["q"],
                    "rank": analysis["rank"],
                    "singular_values": analysis["singular_values"],
                    "jacobian": analysis["jacobian"],
                }
            )
        results.append(pose_result)
    return results


def print_assignment_pose_jacobian_analysis_final(eps=1e-6, rank_tol=1e-6, q5=0.0):
    print("Assignment pose Jacobian analysis (FINAL FK chain):\n")
    results = analyze_assignment_pose_jacobians_final(
        eps=eps, rank_tol=rank_tol, q5=q5
    )
    for pose_result in results:
        label = pose_result["label"]
        pose = pose_result["pose"]
        print(f"{label}. pose={pose}")
        if not pose_result["solutions"]:
            print("  no analytic IK branches available")
            continue
        for sol in pose_result["solutions"]:
            validity = "valid" if sol["valid"] else "invalid"
            sv = ", ".join(f"{float(v):.6f}" for v in sol["singular_values"])
            print(
                f"  branch {sol['index']} ({validity}): "
                f"rank={sol['rank']} singular_values=[{sv}]"
            )


if __name__ == "__main__":
    print_assignment_pose_jacobian_analysis_final()

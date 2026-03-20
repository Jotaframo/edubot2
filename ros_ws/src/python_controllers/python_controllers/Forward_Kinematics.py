import numpy as np


# Robot geometric parameters
L_SH_Y, L_SH_Z = -0.0452, 0.0165
L_UA_Y, L_UA_Z = -0.0306, 0.1025
L_LA_X, L_LA_Y = 0.11257, -0.028
L_WR_X, L_WR_Y = 0.0052, -0.1349
L_GR_X, L_GC_Z = -0.0601, 0.075


def rot_z_batch(angles):
    """Return (N, 3, 3) rotation matrices for an array of angles."""
    c, s = np.cos(angles), np.sin(angles)
    n = len(angles)
    rot = np.zeros((n, 3, 3))
    rot[:, 0, 0], rot[:, 0, 1] = c, -s
    rot[:, 1, 0], rot[:, 1, 1] = s, c
    rot[:, 2, 2] = 1.0
    return rot


def _as_batch(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1), True
    return arr, False


def _forward_kinematics_batch(q1, q2, q3, q4):
    """
    Compute end-effector XYZ position for batches of joint angles.
    Inputs are arrays of equal length.
    """
    r1 = rot_z_batch(q1)
    r2_local = rot_z_batch(q2)
    r3_local = rot_z_batch(q3)
    r4_local = rot_z_batch(1.57079 + q4)

    t_base_sh = np.array([0.0, L_SH_Y, L_SH_Z])
    t_sh_ua = np.array([0.0, L_UA_Y, L_UA_Z])
    t_ua_la = np.array([L_LA_X, L_LA_Y, 0.0])
    t_la_wr = np.array([L_WR_X, L_WR_Y, 0.0])
    t_wr_ee = np.array([L_GR_X, 0.0, L_GC_Z])

    p1 = np.einsum("nij,j->ni", r1, t_base_sh)

    ry_offset = np.array([[0, 0, -1],
                          [0, 1, 0],
                          [1, 0, 0]])
    r12 = np.einsum("nij,jk,nkl->nil", r1, ry_offset, r2_local)
    p2 = p1 + np.einsum("nij,j->ni", r1, t_sh_ua)

    r123 = np.einsum("nij,njk->nik", r12, r3_local)
    p3 = p2 + np.einsum("nij,j->ni", r12, t_ua_la)

    r1234 = np.einsum("nij,njk->nik", r123, r4_local)
    p4 = p3 + np.einsum("nij,j->ni", r123, t_la_wr)
    p_ee = p4 + np.einsum("nij,j->ni", r1234, t_wr_ee)

    r_world_base = np.array([[-1, 0, 0],
                             [0, -1, 0],
                             [0, 0, 1]])
    return np.einsum("ij,nj->ni", r_world_base, p_ee)


def forward_kinematics(q1, q2, q3, q4, batch_mode=None):
    """
    Compute end-effector XYZ position for either a single pose or a batch.

    If `batch_mode` is None, scalar inputs return shape `(3,)` and array inputs
    return shape `(N, 3)`. Set `batch_mode=True` to always return batched output.
    """
    q1_arr, q1_scalar = _as_batch(q1)
    q2_arr, q2_scalar = _as_batch(q2)
    q3_arr, q3_scalar = _as_batch(q3)
    q4_arr, q4_scalar = _as_batch(q4)

    if not (len(q1_arr) == len(q2_arr) == len(q3_arr) == len(q4_arr)):
        raise ValueError("All joint inputs must have equal length.")

    out = _forward_kinematics_batch(q1_arr, q2_arr, q3_arr, q4_arr)
    if batch_mode is True:
        return out
    if batch_mode is False:
        if len(out) != 1:
            raise ValueError("batch_mode=False requires single-pose inputs.")
        return out[0]

    all_scalar = q1_scalar and q2_scalar and q3_scalar and q4_scalar
    return out[0] if all_scalar else out


def jacobian_finite_difference(q, eps=1e-6):
    """
    Compute the 3x4 linear Jacobian numerically via central differences.
    `q` must be an iterable of four joint angles.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("q must have shape (4,)")

    jac = np.zeros((3, 4), dtype=float)
    for i in range(4):
        dq = np.zeros(4, dtype=float)
        dq[i] = eps
        p_plus = forward_kinematics(*(q + dq), batch_mode=False)
        p_minus = forward_kinematics(*(q - dq), batch_mode=False)
        jac[:, i] = (p_plus - p_minus) / (2.0 * eps)
    return jac


def jacobian_svd_and_rank(q, eps=1e-6, rank_tol=1e-6):
    """
    Compute the linear Jacobian plus its singular values and numerical rank.
    """
    jac = jacobian_finite_difference(q, eps=eps)
    singular_values = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(singular_values > rank_tol))
    return {
        "q": np.asarray(q, dtype=float),
        "jacobian": jac,
        "singular_values": singular_values,
        "rank": rank,
    }


def analyze_assignment_pose_jacobians(eps=1e-6, rank_tol=1e-6):
    """
    Solve the assignment pose set with IK and analyze Jacobian rank/SVD
    for each valid joint-space solution.
    """
    try:
        from Inverse_Kinematics import analytical_ik_closed_form
    except ModuleNotFoundError:
        from ros_ws.src.python_controllers.python_controllers.Inverse_Kinematics import analytical_ik_closed_form

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
        for i, q in enumerate(valid_solutions):
            analysis = jacobian_svd_and_rank(q, eps=eps, rank_tol=rank_tol)
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
        for i, q in enumerate(invalid_solutions):
            analysis = jacobian_svd_and_rank(q, eps=eps, rank_tol=rank_tol)
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


def partial_derivative_column(q, joint_index, eps=1e-6):
    """
    Return one Jacobian column, i.e. d p_ee / d q_i, via central differences.
    """
    if joint_index < 0 or joint_index > 3:
        raise ValueError("joint_index must be in [0, 3]")
    return jacobian_finite_difference(q, eps=eps)[:, joint_index]


def print_forward_kinematics():
    """Print the FK chain decomposition for the zero configuration."""
    q1 = np.array([0.0])
    q2 = np.array([0.0])
    q3 = np.array([0.0])
    q4 = np.array([0.0])

    t_base_sh = np.array([0.0, L_SH_Y, L_SH_Z])
    t_sh_ua = np.array([0.0, L_UA_Y, L_UA_Z])
    t_ua_la = np.array([L_LA_X, L_LA_Y, 0.0])
    t_la_wr = np.array([L_WR_X, L_WR_Y, 0.0])
    t_wr_ee = np.array([L_GR_X, 0.0, L_GC_Z])

    r1 = rot_z_batch(q1)
    r2_local = rot_z_batch(q2)
    r3_local = rot_z_batch(q3)
    r4_local = rot_z_batch(1.57079 + q4)

    ry_offset = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    r12 = np.einsum("nij,jk,nkl->nil", r1, ry_offset, r2_local)
    r123 = np.einsum("nij,njk->nik", r12, r3_local)
    r1234 = np.einsum("nij,njk->nik", r123, r4_local)

    print("FORWARD KINEMATICS AS LINEAR COMBINATION OF TRANSFORMATIONS:\n")
    print("p_ee = T_base_sh + R1*t_sh_ua + R12*t_ua_la + R123*t_la_wr + R1234*t_wr_ee\n")
    print(f"T_base_sh: {t_base_sh}")
    print(f"\nR1 (q1={q1[0]}):\n{r1[0]}")
    print(f"-> R1*t_sh_ua = {np.einsum('ij,j->i', r1[0], t_sh_ua)}")
    print(f"\nR12 (q1={q1[0]}, q2={q2[0]}):\n{r12[0]}")
    print(f"-> R12*t_ua_la = {np.einsum('ij,j->i', r12[0], t_ua_la)}")
    print(f"\nR123 (q1={q1[0]}, q2={q2[0]}, q3={q3[0]}):\n{r123[0]}")
    print(f"-> R123*t_la_wr = {np.einsum('ij,j->i', r123[0], t_la_wr)}")
    print(f"\nR1234 (q1={q1[0]}, q2={q2[0]}, q3={q3[0]}, q4={q4[0]}):\n{r1234[0]}")
    print(f"-> R1234*t_wr_ee = {np.einsum('ij,j->i', r1234[0], t_wr_ee)}")
    p_world = forward_kinematics(q1, q2, q3, q4, batch_mode=True)
    print(f"\nFINAL END EFFECTOR POSITION (World Frame):\n{p_world[0]}")


if __name__ == "__main__":
    print_forward_kinematics()
    print("\nAssignment pose Jacobian analysis:\n")
    results = analyze_assignment_pose_jacobians()
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
    analyze_assignment_pose_jacobians()

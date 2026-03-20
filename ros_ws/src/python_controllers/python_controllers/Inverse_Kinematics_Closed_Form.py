import math


def _default_joint_limits():
    return [
        (-2.0, 2.0),      # q1 Shoulder_Rotation
        (-1.57, 1.57),    # q2 Shoulder_Pitch
        (-1.58, 1.58),    # q3 Elbow
        (-1.57, 1.57),    # q4 Wrist_Pitch
    ]


def _limit_violations(q, joint_limits):
    violations = []
    for i, (lo, hi) in enumerate(joint_limits):
        if q[i] < lo or q[i] > hi:
            violations.append(
                {
                    "joint_index": i,
                    "value": q[i],
                    "min": lo,
                    "max": hi,
                }
            )
    return violations


def analytical_ik_closed_form(X, Y, Z, pitch, l2, l3, l4, joint_limits=None, track_invalid=False):
    """
    Analytically solves the closed-form IK equations.

    If track_invalid=False:
      Returns only valid (q1, q2, q3, q4) solutions (possibly empty list).

    If track_invalid=True:
      Returns a dict with:
        - valid_solutions
        - invalid_solutions
        - invalid_reasons
    """
    if joint_limits is None:
        joint_limits = _default_joint_limits()

    valid_solutions = []
    invalid_solutions = []
    invalid_reasons = []

    q1 = math.atan2(Y, X)

    R = math.hypot(X, Y)
    R_prime = R - l4 * math.cos(pitch)
    Z_prime = Z + l4 * math.sin(pitch)

    A = R_prime
    B = Z_prime
    C = (R_prime**2 + Z_prime**2 + l2**2 - l3**2) / (2 * l2)
    r = math.hypot(A, B)

    if abs(C) > r:
        if not track_invalid:
            return []
        invalid_solutions.append(None)
        invalid_reasons.append(
            {
                "code": "geometry_unreachable",
                "message": "wrist center is outside reachable annulus for links l2/l3",
            }
        )
        return {
            "valid_solutions": valid_solutions,
            "invalid_solutions": invalid_solutions,
            "invalid_reasons": invalid_reasons,
        }

    phi = math.atan2(B, A)
    s = C / r

    q2_sol1 = math.asin(s) - phi
    q2_sol2 = math.pi - math.asin(s) - phi

    for q2 in [q2_sol1, q2_sol2]:
        q2 = math.atan2(math.sin(q2), math.cos(q2))

        cos_alpha = (R_prime - l2 * math.sin(q2)) / l3
        sin_alpha = (Z_prime - l2 * math.cos(q2)) / l3
        alpha = math.atan2(sin_alpha, cos_alpha)

        q3 = -(alpha + q2)
        q4 = pitch + alpha

        q = (q1, q2, q3, q4)
        violations = _limit_violations(q, joint_limits)
        if len(violations) == 0:
            valid_solutions.append(q)
        elif track_invalid:
            invalid_solutions.append(q)
            invalid_reasons.append(
                {
                    "code": "joint_limit_violation",
                    "message": "one or more joints violate limits",
                    "violations": violations,
                }
            )

    if not track_invalid:
        return valid_solutions
    return {
        "valid_solutions": valid_solutions,
        "invalid_solutions": invalid_solutions,
        "invalid_reasons": invalid_reasons,
    }


def check_ik_solution_validity(X, Y, Z, pitch, l2, l3, l4, joint_limits=None):
    """
    Verify IK solutions using joint limits only.
    joint_limits format: [(q1_min,q1_max), (q2_min,q2_max), (q3_min,q3_max), (q4_min,q4_max)]
    """
    out = analytical_ik_closed_form(
        X, Y, Z, pitch, l2, l3, l4,
        joint_limits=joint_limits,
        track_invalid=True,
    )
    valid = out["valid_solutions"]
    invalid = [q for q in out["invalid_solutions"] if q is not None]
    reasons = out["invalid_reasons"]
    checks = []
    for i, q in enumerate(valid):
        checks.append({"index": i, "q": q, "violations": [], "valid": True})
    for i, q in enumerate(invalid):
        reason = reasons[i] if i < len(reasons) else {}
        checks.append(
            {
                "index": len(valid) + i,
                "q": q,
                "violations": reason.get("violations", []),
                "valid": False,
                "reason_code": reason.get("code", "unknown"),
            }
        )

    return {
        "target": (X, Y, Z, pitch),
        "num_solutions": len(valid) + len(invalid),
        "any_valid": len(valid) > 0,
        "checks": checks,
    }


def ik_feasibility_reason(X, Y, Z, pitch, l2, l3, l4, joint_limits=None):
    """Return a short reason based on geometry and joint-limit feasibility checks."""
    out = analytical_ik_closed_form(
        X, Y, Z, pitch, l2, l3, l4,
        joint_limits=joint_limits,
        track_invalid=True,
    )
    if len(out["valid_solutions"]) > 0:
        return "feasible (at least one IK branch satisfies all joint limits)"

    for reason in out["invalid_reasons"]:
        if reason.get("code") == "geometry_unreachable":
            return reason.get("message", "geometrically unreachable target")
    if len(out["invalid_solutions"]) > 0:
        return "IK branches exist, but all violate joint limits"

    R = math.hypot(X, Y)
    R_prime = R - l4 * math.cos(pitch)
    Z_prime = Z + l4 * math.sin(pitch)

    A = R_prime
    B = Z_prime
    C = (R_prime**2 + Z_prime**2 + l2**2 - l3**2) / (2 * l2)
    r = math.hypot(A, B)

    if r == 0.0:
        return "degenerate wrist-center geometry (r=0)"
    if abs(C) > r:
        return "wrist center is outside reachable annulus for links l2/l3"
    return "no solution for this model after branch evaluation"


if __name__ == "__main__":
    l2, l3, l4 = 0.11167, 0.16000, 0.15000

    print("Assignment Pose Feasibility Check")
    print("Note: This IK model solves [x, y, z, pitch]. Roll is not modeled.\n")

    assignment_poses = [
        ("I", [0.2, 0.2, 0.2, 1.57, 0.0]),
        ("II", [0.2, 0.1, 0.4, 0.0, 1.57]),
        ("III", [0.0, 0.0, 0.45, 0.785, 0.785]),
        ("IV", [0.0, 0.0, 0.07, 3.141, 0.0]),
        ("V", [0.0, 0.0452, 0.45, 0.785, 3.141]),
    ]

    print("a) IK solutions (YES/NO):")
    for label, pose in assignment_poses:
        x, y, z, pitch, roll = pose
        out = analytical_ik_closed_form(x, y, z, pitch, l2, l3, l4, track_invalid=True)
        total_branches = len(out["valid_solutions"]) + sum(
            1 for q in out["invalid_solutions"] if q is not None
        )
        yes_no = "YES" if len(out["valid_solutions"]) > 0 else "NO"
        print(f"  {label}. {pose} -> {yes_no} ({total_branches} IK branch(es))")

    print("\nb) Why NO for unsolved poses:")
    for label, pose in assignment_poses:
        x, y, z, pitch, roll = pose
        out = analytical_ik_closed_form(x, y, z, pitch, l2, l3, l4, track_invalid=True)
        if len(out["valid_solutions"]) > 0:
            continue
        reason = ik_feasibility_reason(x, y, z, pitch, l2, l3, l4)
        print(f"  {label}. {pose} -> {reason}")
        for i, q in enumerate(out["invalid_solutions"]):
            reason_i = out["invalid_reasons"][i] if i < len(out["invalid_reasons"]) else {}
            if reason_i.get("code") != "joint_limit_violation":
                continue
            violation_txt = ", ".join(
                f"q{v['joint_index']+1}={v['value']:.3f} not in [{v['min']:.3f}, {v['max']:.3f}]"
                for v in reason_i.get("violations", [])
            )
            print(f"     branch {i}: {violation_txt}")
        if abs(roll) > 1e-6:
            print("     Additional note: roll is ignored by this IK model.")

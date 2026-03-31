"""
Symbolic Jacobian for the 5-DOF  manipulator.

Rebuilds the FK chain symbolically (matching Forward_Kinematics_FINAL.py)
and derives the Jacobian:  J_i = [z_i x (p_ee - o_i); z_i].
"""

import numpy as np
import sympy as sp
from sympy import Matrix, cos, sin, pi, trigsimp, zeros

# Joint symbols

q1, q2, q3, q4, q5 = sp.symbols("q1 q2 q3 q4 q5", real=True)

# SymPy rotation / transform helpers (mirrors Forward_Kinematics_FINAL)

def _Rx(a):
    return Matrix([[1,0,0,0],[0,cos(a),-sin(a),0],[0,sin(a),cos(a),0],[0,0,0,1]])

def _Ry(a):
    return Matrix([[cos(a),0,sin(a),0],[0,1,0,0],[-sin(a),0,cos(a),0],[0,0,0,1]])

def _Rz(a):
    return Matrix([[cos(a),-sin(a),0,0],[sin(a),cos(a),0,0],[0,0,1,0],[0,0,0,1]])

def sym_create_tf(tx, ty, tz, roll, pitch, yaw):
    """SymPy equivalent of create_tf_matrix: T @ Rz @ Ry @ Rx."""
    T = Matrix([[1,0,0,tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]])
    return T * _Rz(yaw) * _Ry(pitch) * _Rx(roll)

def _R(x):
    """Float → exact SymPy Rational (mm precision)."""
    return sp.nsimplify(x, rational=True, tolerance=1e-6)


# FK chain (same URDF constants and ordering as Forward_Kinematics_FINAL.py)
T_world_base     = sym_create_tf(0, 0, 0, 0, 0, pi)
T_base_shoulder  = sym_create_tf(_R(0.0), _R(-0.0452), _R(0.0165), 0, 0, 0)
T_shoulder_upper = sym_create_tf(_R(0.0), _R(-0.0306), _R(0.1025), 0, -pi/2, 0)
T_upper_lower    = sym_create_tf(_R(0.11257), _R(-0.028), 0, 0, 0, 0)
T_lower_wrist    = sym_create_tf(_R(0.0052), _R(-0.1349), 0, 0, 0, pi/2)
T_wrist_gripper  = sym_create_tf(_R(-0.0601), 0, 0, 0, -pi/2, 0)
T_gripper_center = sym_create_tf(0, 0, _R(0.075), 0, 0, 0)

A1 = T_base_shoulder  * _Rz(q1)
A2 = T_shoulder_upper * _Rz(q2)
A3 = T_upper_lower    * _Rz(q3)
A4 = T_lower_wrist    * _Rz(q4)
A5 = T_wrist_gripper  * _Rz(q5)

T_total = T_world_base @ A1 @ A2 @ A3 @ A4 @ A5 @ T_gripper_center
p_ee = T_total[:3, 3]

#  Jacobian: z_i x (p_ee - o_i) for linear, z_i for angular
T_fixed = [T_base_shoulder, T_shoulder_upper, T_upper_lower,
           T_lower_wrist,   T_wrist_gripper]

# Cumulative transforms: cum[i] is the world-frame transform up to joint i
cum = [T_world_base]
for Ai in [A1, A2, A3, A4, A5]:
    cum.append(cum[-1] * Ai)

# Each axis_frames[i] gives the frame whose z-column is the rotation axis z_i
axis_frames = [cum[i] * T_fixed[i] for i in range(5)]

J_geo_linear  = zeros(3, 5)
J_geo_angular = zeros(3, 5)
for i in range(5):
    z_i = axis_frames[i][:3, 2]
    o_i = axis_frames[i][:3, 3]
    J_geo_linear[:, i]  = z_i.cross(p_ee - o_i)
    J_geo_angular[:, i] = z_i

# Fast numerical evaluators via lambdify
_J_gl_f = sp.lambdify((q1, q2, q3, q4, q5), J_geo_linear,  "numpy")
_J_ga_f = sp.lambdify((q1, q2, q3, q4, q5), J_geo_angular, "numpy")


def jacobian_numerical(q, q5_val=0.0):
    """6×4 Jacobian (linear + angular) at q (shape (4,))."""
    q = np.asarray(q, dtype=float)
    Jl = np.array(_J_gl_f(q[0], q[1], q[2], q[3], float(q5_val)), dtype=float)[:, :4]
    Ja = np.array(_J_ga_f(q[0], q[1], q[2], q[3], float(q5_val)), dtype=float)[:, :4]
    return np.vstack([Jl, Ja])


# Assignment-pose analysis
def analyze_assignment_poses(rank_tol=1e-5):
    """Print SVD and rank of J_v at every assignment-pose IK solution."""
    try:
        from ros_ws.src.python_controllers.python_controllers.t02_Inverse_Kinematics_Numerical import ik_coordinate_descent_multi_start
    except ModuleNotFoundError:
        from ros_ws.src.python_controllers.python_controllers.t02_Inverse_Kinematics_Numerical import ik_coordinate_descent_multi_start

    assignment_poses = [
        ("I",   [0.2, 0.2, 0.2, 0.0, 1.57, 0.65]),
        ("II",  [0.2, 0.1, 0.4, 0.0, 0.0, -1.57]),
        ("III", [0.0, 0.0, 0.4, 0.0, -0.785, 1.57]),
        ("IV",  [0.0, 0.0, 0.07, 3.141, 0.0, 0.0]),
        ("V",   [0.0, 0.0452, 0.45, -0.785, 0.0, 3.141]),
    ]

    print("Assignment pose Jacobian analysis (symbolic Jacobian):\n")
    for label, pose in assignment_poses:
        x, y, z, rx, ry, rz = pose
        ik_solutions = ik_coordinate_descent_multi_start(
            x, y, z, rx, ry, rz,
            num_random=15, seed=42, unique_decimals=3,
            max_iters=5000, pos_tol=5e-3, rot_tol=np.deg2rad(1.0),
        )

        valid   = [s for s in ik_solutions if s["success"]]
        approx  = [s for s in ik_solutions if not s["success"]]

        print(f"{label}. pose={pose}")
        print(f"   Found {len(valid)} valid solution(s) and {len(approx)} approximate solution(s)")

        for idx, sol in enumerate(valid + approx):
            tag = "VALID" if sol["success"] else "APPROX"
            q4 = sol["q_raw"][:4]
            jac = jacobian_numerical(q4)[:3, :]   # 3×4 linear part
            sv = np.linalg.svd(jac, compute_uv=False)
            rank = int(np.sum(sv > rank_tol))
            sv_str = ", ".join(f"{v:.6f}" for v in sv)
            print(
                f"   Solution {idx} [{tag}]: "
                f"rank={rank}, "
                f"pos_err={sol['pos_error']:.4f}, "
                f"rot_err={sol['rot_error']:.4f}, "
                f"SV=[{sv_str}]"
            )
        print()


# Standalone: symbolic printout, cross-check, assignment-pose analysis
if __name__ == "__main__":

    from ros_ws.src.python_controllers.python_controllers.t03_Jacobian_FINAL import jacobian_finite_difference_final

    # Joint axes in world frame
    print("Joint rotation axes (world frame, q5=0):")
    for i in range(5):
        print(f"  z_{i+1} = {trigsimp(axis_frames[i][:3, 2].subs(q5, 0)).T}")

    # Symbolic linear Jacobian entries (q5=0)
    J4 = J_geo_linear[:, :4].subs(q5, 0)
    coord = "xyz"
    print("\nLinear Jacobian J_v (3×4, q5=0):")
    for r in range(3):
        for c in range(4):
            print(f"  J_v[{coord[r]},q{c+1}] = {trigsimp(J4[r, c])}")

    # Cross-check: symbolic vs finite-difference numerical
    tq = np.array([0.3, -0.5, 0.8, -0.2])
    Jg = jacobian_numerical(tq)[:3, :]   # linear part only
    Jn = jacobian_finite_difference_final(tq)
    print(f"\nTest q = {tq}")
    print(f"  |Symbolic − Finite-diff|_max = {np.max(np.abs(Jg - Jn)):.2e}")

    # Assignment-pose analysis
    print("\n" + "="*70)
    analyze_assignment_poses()

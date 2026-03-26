import numpy as np
from scipy.optimize import least_squares

try:
    from python_controllers.Forward_Kinematics_FINAL import (
        forward_kinematics_full,
        create_tf_matrix,
    )
except ImportError:
    from Forward_Kinematics_FINAL import forward_kinematics_full, create_tf_matrix


BOUNDS_MIN = np.array([-2.0000, -1.5700, -1.5800, -1.5700, -3.1415], dtype=float)
BOUNDS_MAX = np.array([2.0000, 1.5700, 1.5800, 1.5700, 3.1415], dtype=float)
ORIENTATION_COEFF = 1.0


def calculate_errors(T_current, T_target):
    """Return position, rotation, and weighted total error."""
    pos_error = np.linalg.norm(T_target[0:3, 3] - T_current[0:3, 3])
    rot_error = np.linalg.norm(T_target[0:3, 0:3] - T_current[0:3, 0:3])
    total_error = pos_error + 0.2 * rot_error
    return float(pos_error), float(rot_error), float(total_error)


def calculate_scalar_error(T_current, T_target):
    """Return the combined weighted error."""
    _, _, total_error = calculate_errors(T_current, T_target)
    return total_error


def _residual_vector(
    q_active,
    target_frame,
    q_seed=None,
    regularization_parameter=None,
    optimize_orientation=True,
):
    fk = forward_kinematics_full(q_active[0], q_active[1], q_active[2], q_active[3], q_active[4])


    pos_error = fk[:3, 3] - target_frame[:3, 3]
    residual = pos_error
    
    if optimize_orientation:
        rot_error = (fk[:3, :3] - target_frame[:3, :3]).ravel()
        residual = np.concatenate([pos_error, ORIENTATION_COEFF * rot_error])

    if regularization_parameter is not None and q_seed is not None:
        
        reg = regularization_parameter * (q_active - np.asarray(q_seed, dtype=float))
        residual = np.concatenate([residual, reg])

    return residual


def _build_result(q, target_frame, iterations, pos_tol, rot_tol, scipy_result=None):
    q = np.clip(np.asarray(q, dtype=float), BOUNDS_MIN, BOUNDS_MAX)
    fk = forward_kinematics_full(*q)
    pos_error, rot_error, total_error = calculate_errors(fk, target_frame)
    success = bool(pos_error <= pos_tol and rot_error <= rot_tol)

    result = {
        "success": success,
        "q": np.round(q, 4),
        "q_raw": q.copy(),
        "pos_error": round(pos_error, 4),
        "pos_error_raw": pos_error,
        "rot_error": round(rot_error, 4),
        "rot_error_raw": rot_error,
        "error_raw": total_error,
        "iters": int(iterations),
    }

    if scipy_result is not None:
        result["optimizer_status"] = int(scipy_result.status)
        result["optimizer_message"] = str(scipy_result.message)
        result["optimizer_cost"] = float(scipy_result.cost)

    return result


def ik_coordinate_descent(
    x,
    y,
    z,
    rot_x,
    rot_y,
    rot_z,
    q_init=None,
    max_iters=5000,
    tolerance=3e-3,
    pos_tol=5e-3,
    rot_tol=np.deg2rad(1.0),
    regularization_parameter=None,
    optimize_orientation=True,
):
    """
    Legacy public entry point kept in place, but now backed by least-squares
    optimization similar to ikpy's inverse kinematics optimizer.
    """
    target_frame = create_tf_matrix(x, y, z, rot_x, rot_y, rot_z)
    q0 = np.zeros(5, dtype=float) if q_init is None else np.asarray(q_init, dtype=float)
    q0 = np.clip(q0, BOUNDS_MIN, BOUNDS_MAX)

    # Match ikpy more closely by relying on scipy's default least-squares
    # termination criteria. The public tolerance arguments are kept for
    # feasibility checks, not optimizer early-stop control.
    kwargs = {}
    if max_iters is not None:
        kwargs["max_nfev"] = max_iters

    result = least_squares(
        lambda q: _residual_vector(
            q,
            target_frame,
            q_seed=q0,
            regularization_parameter=regularization_parameter,
            optimize_orientation=optimize_orientation,
        ),
        q0,
        bounds=(BOUNDS_MIN, BOUNDS_MAX),
        **kwargs,
    )

    return _build_result(
        result.x,
        target_frame,
        iterations=result.nfev,
        pos_tol=pos_tol,
        rot_tol=(rot_tol if optimize_orientation else np.inf),
        scipy_result=result,
    )


def generate_initial_guesses(num_random=5, seed=42, q_prev=None):
    rng = np.random.default_rng(seed)
    guesses = []

    if q_prev is not None:
        guesses.append(np.clip(np.asarray(q_prev, dtype=float), BOUNDS_MIN, BOUNDS_MAX))

    guesses.extend(
        [
            np.zeros(5, dtype=float),
            BOUNDS_MIN.copy(),
            BOUNDS_MAX.copy(),
            0.5 * (BOUNDS_MIN + BOUNDS_MAX),
        ]
    )

    for _ in range(num_random):
        guesses.append(rng.uniform(BOUNDS_MIN, BOUNDS_MAX))

    return guesses


def ik_coordinate_descent_multi_start(
    x,
    y,
    z,
    rot_x,
    rot_y,
    rot_z,
    initial_guesses=None,
    q_prev=None,
    max_iters=5000,
    tolerance=3e-3,
    pos_tol=5e-3,
    rot_tol=np.deg2rad(1.0),
    unique_decimals=3,
    num_random=5,
    seed=42,
    regularization_parameter=None,
    optimize_orientation=True,
):
    if initial_guesses is None:
        initial_guesses = generate_initial_guesses(
            num_random=num_random, seed=seed, q_prev=q_prev
        )

    results = []
    seen = set()

    for q0 in initial_guesses:
        result = ik_coordinate_descent(
            x,
            y,
            z,
            rot_x,
            rot_y,
            rot_z,
            q_init=q0,
            max_iters=max_iters,
            tolerance=tolerance,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            regularization_parameter=regularization_parameter,
            optimize_orientation=optimize_orientation,
        )

        rounded_key = tuple(np.round(result["q_raw"], unique_decimals))
        if rounded_key in seen:
            continue
        seen.add(rounded_key)
        results.append(result)

    results.sort(key=lambda item: item["error_raw"])

    valid_results = [item for item in results if item["success"]]
    if valid_results:
        return valid_results
    return results[:1]


if __name__ == "__main__":
    poses = {
        "I": [0.2000, 0.2000, 0.2000, 0.0000, 1.5700, 0.6500],
        "II": [0.2000, 0.1000, 0.4000, 0.0000, 0.0000, -1.5700],
        "III": [0.0000, 0.0000, 0.4000, 0.0000, -0.7850, 1.5700],
        "IV_a": [0.0000, 0.0000, 0.0700, 3.1410, 0.0000, 0.0000],
        "IV_b": [0.0000, 0.0452, 0.4500, -0.7850, 0.0000, 3.1410],
    }

    print("Running least-squares numerical IK...\n")
    for name, pose in poses.items():
        # Use multi-start approach with 15 random initial guesses to find multiple solutions
        res_list = ik_coordinate_descent_multi_start(
            *pose,
            num_random=15,
            seed=42,
            unique_decimals=3
        )
        
        valid_results = [res for res in res_list if res["success"]]
        
        print(f"Pose {name}:")
        print(f"  Found {len(valid_results)} feasible solution(s) out of {len(res_list)} candidate(s)")
        
        for idx, res in enumerate(res_list, start=1):
            status = "FEASIBLE" if res["success"] else "approximate only"
            print(f"  Solution {idx} [{status}]")
            print(
                f"    Pos Error: {res['pos_error']:.4f}, "
                f"Rot Error: {res['rot_error']:.4f}, "
            )
            print(f"    Joint angles (rad): {res['q']}")
            print(f"    Iterations: {res['iters']}")
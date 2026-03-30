from dataclasses import dataclass
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from python_controllers.Inverse_Kinematics_Numerical import ik_coordinate_descent
from python_controllers.Forward_Kinematics_FINAL import forward_kinematics_full

@dataclass
class Stage:
    name: str
    xyz: tuple[float, float, float] | None = None
    gripper: float | None = None
    move_time: float = None
    hold_s: float = None



@dataclass(frozen=True)
class PhysicalPickPlaceTuning:
    # Encoder-tuned robot pose which will define location of the pick
    start_q: tuple[float, float, float, float, float] = (0.822, -0.3283, -0.4218, -1.1919, -1.0446)
    # If None, lock orientation to the FK orientation of start_q.
    # This keeps wrist pitch consistent with the calibrated start pose.
    fixed_world_rpy: tuple[float, float, float] | None = None
    x_offset_m: float = 0.10

    # Tuned values to ensure robot is gripping hard enough
    gripper_open: float = 0.5
    gripper_closed: float = 0.12

    # Values to be changed depending on the block that is picked
    pick_z_m: float = 0.01
    hover_z_m: float = 0.12
    travel_z_m: float = 0.07

    # Motion timing for the physical robot
    initial_approach_move_time_s: float = 3.0
    descend_move_time_s: float = 2.0
    grip_move_time_s: float = 1.2
    grip_hold_s: float = 1.0
    lift_move_time_s: float = 2.0
    transfer_move_time_s: float = 4.0


PHYSICAL_TUNING = PhysicalPickPlaceTuning()
LOCKED_J1_REGULARIZATION = 3.0
DESCENT_Z_TOL_M = 0.003

# Helper function to convert the calibrated pose to xyz location
def fk_xyz(q):
    T = forward_kinematics_full(q[0], q[1], q[2], q[3], q[4])
    return np.asarray(T[:3, 3], dtype=float)


def rpy_from_rotation_zyx(R: np.ndarray) -> tuple[float, float, float]:
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = float(np.arcsin(sy))
    cp = float(np.cos(pitch))

    if abs(cp) > 1e-8:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))

    return roll, pitch, yaw


def fk_rpy(q) -> tuple[float, float, float]:
    T = forward_kinematics_full(q[0], q[1], q[2], q[3], q[4])
    return rpy_from_rotation_zyx(T[:3, :3])

class PickPlaceOpenLoop(Node):
    def __init__(self):
        super().__init__("pick_place_world_locked")

        self.tuning = PHYSICAL_TUNING

        self.start_q = np.array(self.tuning.start_q, dtype=float)
        if self.tuning.fixed_world_rpy is None:
            self.locked_rpy = fk_rpy(self.start_q)
        else:
            self.locked_rpy = self.tuning.fixed_world_rpy
        
        self.gripper_open = self.tuning.gripper_open
        self.gripper_closed = self.tuning.gripper_closed
        
        self.pick_z_m = self.tuning.pick_z_m
        self.hover_z_m = self.tuning.hover_z_m
        self.travel_z_m = self.tuning.travel_z_m

        self.initial_approach_move_time_s = self.tuning.initial_approach_move_time_s
        self.descend_move_time_s = self.tuning.descend_move_time_s
        self.grip_move_time_s = self.tuning.grip_move_time_s
        self.grip_hold_s = self.tuning.grip_hold_s
        self.lift_move_time_s = self.tuning.lift_move_time_s
        self.transfer_move_time_s = self.tuning.transfer_move_time_s

        # Use the FK on the initial pose to obtain the (xyz) of the EE
        self.location_a = fk_xyz(self.start_q)
        # Add the offset to move in the x direction
        x_off = self.tuning.x_offset_m
        self.location_b = self.location_a + np.array([x_off, 0.0, 0.0])

        # Compute the target base angle for place by rotating around joint 1 only
        yaw_a = np.arctan2(self.location_a[1], self.location_a[0])
        yaw_b = np.arctan2(self.location_b[1], self.location_b[0])
        self.place_joint1_angle = float(self.start_q[0] + (yaw_b - yaw_a))
        # Publish joint commands
        self.pub = self.create_publisher(JointTrajectory, "/joint_cmds", 10)
        self.current_q = self.start_q.copy()
        self.current_gripper = self.gripper_open
        self.done = False
        # Build the stages using the class function
        self.stages = PickPlaceOpenLoop._build_stage_sequence(self)
        self.stage_idx = 0
        self.stage_active = False
        self.timer = self.create_timer(0.05, self._tick)

    def _pitch_only_descend_target(self, q_start, target_z):
        q_start = np.array(q_start, dtype=float)
        start_z = float(fk_xyz(q_start)[2])
        best_q = q_start.copy()
        best_err = abs(start_z - float(target_z))

        pitch_sum_ref = float(q_start[1] + q_start[2] + q_start[3])

        for d_shoulder in np.linspace(-0.8, 0.8, 81):
            for d_elbow in np.linspace(-0.8, 0.8, 81):
                q_candidate = q_start.copy()
                q_candidate[1] = float(q_start[1] + d_shoulder)
                q_candidate[2] = float(q_start[2] + d_elbow)
                q_candidate[3] = float(pitch_sum_ref - q_candidate[1] - q_candidate[2])

                z_candidate = float(fk_xyz(q_candidate)[2])
                if z_candidate > start_z:
                    continue

                err = abs(z_candidate - float(target_z))
                if err < best_err:
                    best_err = err
                    best_q = q_candidate

        best_q[0] = q_start[0]
        best_q[4] = q_start[4]
        return best_q

    def _solve_ik(self, target_xyz, q_init=None, lock_joint1=None):
        q_seed = self.current_q if q_init is None else q_init
        if lock_joint1 is None:
            res = ik_coordinate_descent(
                target_xyz[0], target_xyz[1], target_xyz[2],
                self.locked_rpy[0], self.locked_rpy[1], self.locked_rpy[2],
                q_init=q_seed,
                optimize_orientation=True,
            )
            return np.array(res["q_raw"], dtype=float)

        constrained_seed = np.array(q_seed, dtype=float)
        constrained_seed[0] = float(lock_joint1)

        res = ik_coordinate_descent(
            target_xyz[0], target_xyz[1], target_xyz[2],
            self.locked_rpy[0], self.locked_rpy[1], self.locked_rpy[2],
            q_init=constrained_seed,
            optimize_orientation=True,
            regularization_parameter=LOCKED_J1_REGULARIZATION,
        )
        q_sol = np.array(res["q_raw"], dtype=float)
        q_sol[0] = float(lock_joint1)

        ee_xyz = fk_xyz(q_sol)
        z_err = abs(float(ee_xyz[2]) - float(target_xyz[2]))
        if (not res["success"]) or (z_err > DESCENT_Z_TOL_M):
            res_retry = ik_coordinate_descent(
                target_xyz[0], target_xyz[1], target_xyz[2],
                self.locked_rpy[0], self.locked_rpy[1], self.locked_rpy[2],
                q_init=q_sol,
                optimize_orientation=True,
                regularization_parameter=0.5,
            )
            q_sol = np.array(res_retry["q_raw"], dtype=float)
            q_sol[0] = float(lock_joint1)

        res_refine = ik_coordinate_descent(
            target_xyz[0], target_xyz[1], target_xyz[2],
            self.locked_rpy[0], self.locked_rpy[1], self.locked_rpy[2],
            q_init=q_sol,
            optimize_orientation=True,
            regularization_parameter=LOCKED_J1_REGULARIZATION,
        )
        q_sol = np.array(res_refine["q_raw"], dtype=float)
        q_sol[0] = float(lock_joint1)
        return q_sol

    def _build_stage_sequence(self):
        
        # Cartesian coordinates of different positions in the pick and place pocedure

        a_hov = (self.location_a[0], self.location_a[1], self.hover_z_m)
        a_pk = (self.location_a[0], self.location_a[1], self.pick_z_m)
        a_tr = (self.location_a[0], self.location_a[1], self.travel_z_m)
        b_pk = (self.location_b[0], self.location_b[1], self.pick_z_m)
        b_tr = (self.location_b[0], self.location_b[1], self.travel_z_m)
        
        
        
        return [
            Stage("initial_approach", xyz=a_hov, gripper=self.gripper_open, move_time=self.initial_approach_move_time_s),
            Stage("descend_a", xyz=a_pk, gripper=self.gripper_open, move_time=self.descend_move_time_s),
            Stage("grasp_a", xyz=a_pk, gripper=self.gripper_closed, move_time=self.grip_move_time_s, hold_s=self.grip_hold_s),
            Stage("lift_a", xyz=a_tr, gripper=self.gripper_closed, move_time=self.lift_move_time_s),
            Stage("rotate_about_joint1", xyz=None, gripper=self.gripper_closed, move_time=self.transfer_move_time_s),
            Stage("descend_b", xyz=b_pk, gripper=self.gripper_closed, move_time=self.descend_move_time_s),
            Stage("release_b", xyz=b_pk, gripper=self.gripper_open, move_time=self.grip_move_time_s, hold_s=self.grip_hold_s),
            Stage("lift_from_b", xyz=b_tr, gripper=self.gripper_open, move_time=self.lift_move_time_s),
        ]

    def _tick(self):
        if self.done:
            return

        if self.stage_idx >= len(self.stages): return

        stage = self.stages[self.stage_idx]
        
        if not self.stage_active:
            self.stage_start_t = self.get_clock().now()
            self.stage_start_q = self.current_q.copy()
            self.stage_start_xyz = fk_xyz(self.stage_start_q)
            self.stage_target_xyz = None

            if stage.name == "rotate_about_joint1":
                self.stage_target_q = self.stage_start_q.copy()
                self.stage_target_q[0] = self.place_joint1_angle
            elif stage.name == "release_b":
                self.stage_target_q = self.stage_start_q.copy()
                self.stage_target_xyz = stage.xyz
            elif stage.name == "descend_b":
                self.stage_target_xyz = (
                    float(self.stage_start_xyz[0]),
                    float(self.stage_start_xyz[1]),
                    float(stage.xyz[2]),
                )
                self.stage_target_q = self._pitch_only_descend_target(
                    self.stage_start_q,
                    self.stage_target_xyz[2],
                )
            else:
                if stage.xyz is None:
                    raise ValueError(f"Stage '{stage.name}' is missing an xyz target")

                stage_target_xyz = stage.xyz

                lock_joint1 = None
                if stage.name in {"descend_b", "release_b", "lift_from_b"}:
                    lock_joint1 = self.stage_start_q[0]

                if stage.name in {"release_b", "lift_from_b"}:
                    current_xyz = self.stage_start_xyz
                    stage_target_xyz = (
                        float(current_xyz[0]),
                        float(current_xyz[1]),
                        float(stage.xyz[2]),
                    )

                self.stage_target_xyz = stage_target_xyz

                self.stage_target_q = self._solve_ik(
                    stage_target_xyz,
                    q_init=self.stage_start_q,
                    lock_joint1=lock_joint1,
                )

            self.stage_active = True

        elapsed = (self.get_clock().now() - self.stage_start_t).nanoseconds / 1e9
        alpha = np.clip(elapsed / stage.move_time, 0.0, 1.0)
        s_alpha = alpha**2 * (3 - 2*alpha) # Smoothstep

        q_cmd = self.stage_start_q + (self.stage_target_q - self.stage_start_q) * s_alpha
        if stage.name == "lift_from_b" and self.stage_target_xyz is not None:
            z_cmd = float(self.stage_start_xyz[2] + (self.stage_target_xyz[2] - self.stage_start_xyz[2]) * s_alpha)
            xyz_cmd = (
                float(self.stage_start_xyz[0]),
                float(self.stage_start_xyz[1]),
                z_cmd,
            )
            q_cmd = self._solve_ik(
                xyz_cmd,
                q_init=self.current_q,
                lock_joint1=float(self.stage_start_q[0]),
            )

        # Update tracking
        cmd_gripper = stage.gripper if stage.gripper is not None else self.current_gripper
        if stage.name == "release_b":
            ee_xyz_now = fk_xyz(q_cmd)
            release_z = self.pick_z_m if self.stage_target_xyz is None else float(self.stage_target_xyz[2])
            if ee_xyz_now[2] > release_z-0.005:
                cmd_gripper = self.gripper_closed
        
        # Publish
        msg = JointTrajectory()
        pt = JointTrajectoryPoint()
        pt.positions = q_cmd.tolist() + [cmd_gripper]
        pt.time_from_start = Duration(sec=0, nanosec=50000000)
        msg.points = [pt]
        self.pub.publish(msg)

        hold_s = stage.hold_s if stage.hold_s is not None else 0.0
        if elapsed >= stage.move_time + hold_s:
            self.current_q = q_cmd # Ensure continuity for next stage start
            self.current_gripper = cmd_gripper
            self.stage_idx += 1
            self.stage_active = False


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceOpenLoop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

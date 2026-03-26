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
    move_time: float = 5.0
    hold_s: float = 1.0
    use_ik: bool = True
    q_override: list[float] | None = None
    straight_line: bool = False


@dataclass(frozen=True)
class PhysicalPickPlaceTuning:
    # Encoder-tuned robot pose and orientation calibration
    home_q: tuple[float, float, float, float, float] = (0.4, 0.9, -0.8, -1.0, 0.0)
    fixed_world_rpy: tuple[float, float, float] = (3.14, 0.0, 0.0)
    use_cartesian_offset: bool = True
    x_offset_m: float = 0.10

    # Encoder-tuned gripper commands
    gripper_open: float = 0.8
    gripper_closed: float = 0.1

    # Encoder-tuned Cartesian heights
    pick_z_m: float = 0.03
    hover_z_m: float = 0.10
    travel_z_m: float = 0.12

    # Motion timing for the physical robot
    initial_approach_move_time_s: float = 3.0
    descend_move_time_s: float = 2.0
    grip_move_time_s: float = 1.2
    grip_hold_s: float = 1.0
    lift_move_time_s: float = 2.0
    transfer_move_time_s: float = 4.0


PHYSICAL_TUNING = PhysicalPickPlaceTuning()


def fk_xyz(q):
    T = forward_kinematics_full(q[0], q[1], q[2], q[3], q[4])
    return np.asarray(T[:3, 3], dtype=float)

class PickPlaceOpenLoop(Node):
    def __init__(self):
        super().__init__("pick_place_world_locked")

        self.tuning = PHYSICAL_TUNING

        self.declare_parameter("home_q", list(self.tuning.home_q))
        self.declare_parameter("fixed_world_rpy", list(self.tuning.fixed_world_rpy))
        self.declare_parameter("use_cartesian_offset", self.tuning.use_cartesian_offset)
        self.declare_parameter("x_offset_m", self.tuning.x_offset_m)

        self.home_q = np.array(self.get_parameter("home_q").value, dtype=float)
        self.locked_rpy = np.array(self.get_parameter("fixed_world_rpy").value, dtype=float)
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

        # Derived Locations
        self.location_a = fk_xyz(self.home_q)
        x_off = self.get_parameter("x_offset_m").value
        self.location_b = self.location_a + np.array([x_off, 0.0, 0.0])

        # State
        self.pub = self.create_publisher(JointTrajectory, "/joint_cmds", 10)
        self.current_q = self.home_q.copy()
        self.current_gripper = self.gripper_open
        
        self.stage_idx = 0
        self.stage_active = False
        self.timer = self.create_timer(0.05, self._tick)

    def _solve_ik(self, target_xyz):
        res = ik_coordinate_descent(
            target_xyz[0], target_xyz[1], target_xyz[2],
            self.locked_rpy[0], self.locked_rpy[1], self.locked_rpy[2],
            q_init=self.current_q, 
            optimize_orientation=True 
        )
        return np.array(res["q_raw"])

    def _tick(self):
        if self.stage_idx >= len(self.stages): return

        stage = self.stages[self.stage_idx]
        
        if not self.stage_active:
            self.stage_start_t = self.get_clock().now()
            self.stage_start_q = self.current_q.copy()
            
            if stage.use_ik and stage.xyz is not None:
                self.stage_target_q = self._solve_ik(stage.xyz)
                self.start_xyz = fk_xyz(self.stage_start_q)
            elif stage.q_override is not None:
                self.stage_target_q = np.array(stage.q_override)
            else:
                self.stage_target_q = self.stage_start_q

            self.stage_active = True

        elapsed = (self.get_clock().now() - self.stage_start_t).nanoseconds / 1e9
        alpha = np.clip(elapsed / stage.move_time, 0.0, 1.0)
        s_alpha = alpha**2 * (3 - 2*alpha) # Smoothstep

        if stage.straight_line and stage.xyz is not None:
            interp_xyz = self.start_xyz + (np.array(stage.xyz) - self.start_xyz) * s_alpha
            q_cmd = self._solve_ik(interp_xyz)
        else:
            q_cmd = self.stage_start_q + (self.stage_target_q - self.stage_start_q) * s_alpha

        # Update tracking
        cmd_gripper = stage.gripper if stage.gripper is not None else self.current_gripper
        
        # Publish
        msg = JointTrajectory()
        pt = JointTrajectoryPoint()
        pt.positions = q_cmd.tolist() + [cmd_gripper]
        pt.time_from_start = Duration(sec=0, nanosec=50000000)
        msg.points = [pt]
        self.pub.publish(msg)

        if elapsed >= stage.move_time + stage.hold_s:
            self.current_q = q_cmd # Ensure continuity for next stage start
            self.current_gripper = cmd_gripper
            self.stage_idx += 1
            self.stage_active = False

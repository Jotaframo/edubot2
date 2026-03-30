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
    start_q: tuple[float, float, float, float, float] = (0.4, 0.9, -0.8, -1.0, 0.0)
    # Common sense value since the gripper should point downwards this oritenatoin of
    # the EE should be fixed wrt to the world fram for the entire movement
    fixed_world_rpy: tuple[float, float, float] = (3.14, 0.0, 0.0)
    x_offset_m: float = 0.20

    # Tuned values to ensure robot is gripping hard enough
    gripper_open: float = 0.8
    gripper_closed: float = 0.1

    # Values to be changed depending on the block that is picked
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

# Helper function to convert the calibrated pose to xyz location
def fk_xyz(q):
    T = forward_kinematics_full(q[0], q[1], q[2], q[3], q[4])
    return np.asarray(T[:3, 3], dtype=float)

class PickPlaceOpenLoop(Node):
    def __init__(self):
        super().__init__("pick_place_world_locked")

        self.tuning = PHYSICAL_TUNING

        self.start_q = self.tuning.start_q
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

    def _solve_ik(self, target_xyz):
        res = ik_coordinate_descent(
            target_xyz[0], target_xyz[1], target_xyz[2],
            self.locked_rpy[0], self.locked_rpy[1], self.locked_rpy[2],
            q_init=self.current_q, 
            optimize_orientation=True 
        )
        return np.array(res["q_raw"])

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
            Stage("travel_to_b", xyz=b_tr, gripper=self.gripper_closed, move_time=self.transfer_move_time_s),
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

            if stage.xyz is None:
                raise ValueError(f"Stage '{stage.name}' is missing an xyz target")

            self.stage_target_q = self._solve_ik(stage.xyz)
            self.start_xyz = fk_xyz(self.stage_start_q)

            self.stage_active = True

        elapsed = (self.get_clock().now() - self.stage_start_t).nanoseconds / 1e9
        alpha = np.clip(elapsed / stage.move_time, 0.0, 1.0)
        s_alpha = alpha**2 * (3 - 2*alpha) # Smoothstep

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

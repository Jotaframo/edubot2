from dataclasses import dataclass

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


JOINT_COUNT = 6  # 5 arm joints + gripper


@dataclass
class Stage:
    name: str
    q: list[float]
    gripper: float
    move_time: float
    hold_s: float


def build_stages(q_hover, q_pick, q_hover_drop, q_drop, gripper_open, gripper_close):
    """
    All poses are full 5-joint arrays calibrated directly on the robot.
    q_hover_drop and q_drop are q_hover/q_pick with base joint rotated by base_delta.
    """
    return [
        Stage("hover_pick",  q_hover,      gripper_open,  move_time=4.0, hold_s=1.0),
        Stage("descend",     q_pick,       gripper_open,  move_time=2.5, hold_s=1.0),
        Stage("grasp",       q_pick,       gripper_close, move_time=1.2, hold_s=1.5),
        Stage("lift",        q_hover,      gripper_close, move_time=3.0, hold_s=1.0),
        Stage("hover_drop",  q_hover_drop, gripper_close, move_time=4.0, hold_s=1.0),
        Stage("descend_drop",q_drop,       gripper_close, move_time=2.5, hold_s=1.0),
        Stage("release",     q_drop,       gripper_open,  move_time=1.2, hold_s=1.5),
        Stage("lift_away",   q_hover_drop, gripper_open,  move_time=3.0, hold_s=1.0),
    ]


class PickPlace(Node):
    def __init__(self):
        super().__init__("pick_place")

        # --- parameters ---
        self.declare_parameter("topic", "/joint_cmds")
        self.declare_parameter("tick_s", 0.05)
        self.declare_parameter("gripper_open", 0.8)
        self.declare_parameter("gripper_close", 0.45)
        self.declare_parameter("base_rotation_delta_deg", 90.0)
        # Stack: added to the drop shoulder joint per iteration to rise with the stack
        self.declare_parameter("stack_joint_index", 1)
        self.declare_parameter("stack_joint_delta_per_layer", 0.05)
        self.declare_parameter("n_cycles", 1)

        # Calibrated poses — read these directly off the robot
        self.declare_parameter(
            "q_pick",
            [-1.41739824072, 0.64273794682, -1.4189322215, -0.98021371842, -1.54471864546],
        )
        self.declare_parameter("q_hover", 
            [-1.41739824072, 0.64273794682, -1.1, -0.98021371842, -1.54471864546]
        )

        self.topic        = str(self.get_parameter("topic").value)
        self.tick_s       = float(self.get_parameter("tick_s").value)
        self.gripper_open = float(self.get_parameter("gripper_open").value)
        self.gripper_close= float(self.get_parameter("gripper_close").value)
        self.base_delta   = np.deg2rad(float(self.get_parameter("base_rotation_delta_deg").value))
        self.stack_idx    = int(self.get_parameter("stack_joint_index").value)
        self.stack_delta  = float(self.get_parameter("stack_joint_delta_per_layer").value)
        self.n_cycles     = int(self.get_parameter("n_cycles").value)

        q_pick  = np.asarray(self.get_parameter("q_pick").value,  dtype=float)
        q_hover = np.asarray(self.get_parameter("q_hover").value, dtype=float)

        self.q_pick  = q_pick
        self.q_hover = q_hover

        # Drop poses: identical to pick poses but base joint rotated
        self.q_pick_drop  = q_pick.copy();  self.q_pick_drop[0]  += self.base_delta
        self.q_hover_drop = q_hover.copy(); self.q_hover_drop[0] += self.base_delta

        self.pub = self.create_publisher(JointTrajectory, self.topic, 10)

        # Execution state
        self.cycle        = 0
        self.stages       = []
        self.stage_idx    = 0
        self.stage_active = False
        self.done         = False

        self.current_q       = q_hover.copy()
        self.current_gripper = self.gripper_open

        self.stage_started      = None
        self.stage_start_q      = self.current_q.copy()
        self.stage_target_q     = self.current_q.copy()
        self.stage_start_grip   = self.current_gripper
        self.stage_target_grip  = self.current_gripper
        self.stage_move_time    = 1.0

        self._load_cycle()
        self.timer = self.create_timer(self.tick_s, self._tick)
        self.get_logger().info(
            f"pick_place ready: {self.n_cycles} cycle(s), "
            f"base_delta={np.rad2deg(self.base_delta):.1f} deg, "
            f"stack_delta={self.stack_delta:.3f} rad/layer on joint {self.stack_idx}"
        )

    # ------------------------------------------------------------------

    def _drop_poses_for_cycle(self, cycle):
        """Shift the drop poses upward (joint space) for stacking."""
        q_drop       = self.q_pick_drop.copy()
        q_hover_drop = self.q_hover_drop.copy()
        q_drop[self.stack_idx]       += cycle * self.stack_delta
        q_hover_drop[self.stack_idx] += cycle * self.stack_delta
        return q_hover_drop, q_drop

    def _load_cycle(self):
        q_hover_drop, q_drop = self._drop_poses_for_cycle(self.cycle)
        self.stages    = build_stages(
            q_hover      = self.q_hover,
            q_pick       = self.q_pick,
            q_hover_drop = q_hover_drop,
            q_drop       = q_drop,
            gripper_open = self.gripper_open,
            gripper_close= self.gripper_close,
        )
        self.stage_idx    = 0
        self.stage_active = False
        self.get_logger().info(f"starting cycle {self.cycle + 1} / {self.n_cycles}")

    # ------------------------------------------------------------------

    def _start_stage(self, stage):
        self.stage_start_q     = self.current_q.copy()
        self.stage_target_q    = np.asarray(stage.q, dtype=float)
        self.stage_start_grip  = self.current_gripper
        self.stage_target_grip = float(np.clip(stage.gripper, 0.0, 1.0))
        self.stage_move_time   = stage.move_time
        self.stage_started     = self.get_clock().now()
        self.stage_active      = True
        self.get_logger().info(
            f"  stage {self.stage_idx + 1}/{len(self.stages)}: {stage.name}"
        )

    def _interpolate(self, elapsed):
        alpha = float(np.clip(elapsed / max(self.stage_move_time, 1e-6), 0.0, 1.0))
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # cubic smoothstep
        q    = self.stage_start_q   + alpha * (self.stage_target_q    - self.stage_start_q)
        grip = self.stage_start_grip + alpha * (self.stage_target_grip - self.stage_start_grip)
        return q, float(np.clip(grip, 0.0, 1.0))

    def _publish(self, q, gripper, move_time):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        pt = JointTrajectoryPoint()
        pt.positions = [*q.tolist(), float(np.clip(gripper, 0.0, 1.0))]
        pt.velocities = [0.0] * JOINT_COUNT
        pt.time_from_start = Duration(
            sec=int(move_time), nanosec=int((move_time % 1.0) * 1e9)
        )
        msg.points = [pt]
        self.pub.publish(msg)

    # ------------------------------------------------------------------

    def _tick(self):
        if self.done:
            return

        if self.stage_idx >= len(self.stages):
            self.cycle += 1
            if self.cycle >= self.n_cycles:
                self.done = True
                self.timer.cancel()
                self.get_logger().info("all cycles complete")
                return
            self._load_cycle()
            return

        stage = self.stages[self.stage_idx]

        if not self.stage_active:
            self._start_stage(stage)
            return

        elapsed = (self.get_clock().now() - self.stage_started).nanoseconds * 1e-9
        q_cmd, grip_cmd = self._interpolate(elapsed)
        self._publish(q_cmd, grip_cmd, self.stage_move_time)

        if elapsed >= self.stage_move_time + stage.hold_s:
            self.current_q       = self.stage_target_q.copy()
            self.current_gripper = self.stage_target_grip
            self.stage_idx      += 1
            self.stage_active    = False


def main(args=None):
    rclpy.init(args=args)
    node = PickPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

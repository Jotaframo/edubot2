from dataclasses import dataclass

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from python_controllers.Inverse_Kinematics_Numerical import ik_coordinate_descent


ARM_JOINT_COUNT = 5


@dataclass
class Stage:
    name: str
    xyz: tuple[float, float, float] | None = None
    gripper: float | None = None
    hold_s: float = 1.0
    use_ik: bool = True
    q_override: list[float] | None = None


class PickPlaceOpenLoop(Node):
    def __init__(self):
        super().__init__("pick_place_open_loop")

        self.declare_parameter("topic", "/joint_cmds")
        self.declare_parameter("move_time", 1.5)
        self.declare_parameter("tick_s", 0.05)
        self.declare_parameter("hover_height", 0.08)
        self.declare_parameter("grasp_height", 0.025)
        self.declare_parameter("travel_height", 0.10)
        self.declare_parameter("orientation_rpy", [0.0, 0.0, 0.0])
        self.declare_parameter("home_q", [0.0, 0.9, -0.8, -1.0, 0.0])
        self.declare_parameter("location_a", [0.00, 0.44, 0.0])
        self.declare_parameter("location_b", [0.10, 0.44, 0.0])
        self.declare_parameter("gripper_open", 1.0)
        self.declare_parameter("gripper_gentle_close", 0.18)
        self.declare_parameter("ik_max_iters", 200)
        self.declare_parameter("ik_pos_tol", 0.005)
        self.declare_parameter("ik_rot_tol_deg", 1.0)

        self.topic = str(self.get_parameter("topic").value)
        self.move_time = float(self.get_parameter("move_time").value)
        self.tick_s = float(self.get_parameter("tick_s").value)
        self.hover_height = float(self.get_parameter("hover_height").value)
        self.grasp_height = float(self.get_parameter("grasp_height").value)
        self.travel_height = float(self.get_parameter("travel_height").value)
        self.orientation_rpy = np.asarray(
            self.get_parameter("orientation_rpy").value, dtype=float
        )
        self.home_q = np.asarray(self.get_parameter("home_q").value, dtype=float)
        self.location_a = np.asarray(self.get_parameter("location_a").value, dtype=float)
        self.location_b = np.asarray(self.get_parameter("location_b").value, dtype=float)
        self.gripper_open = float(self.get_parameter("gripper_open").value)
        self.gripper_gentle_close = float(
            self.get_parameter("gripper_gentle_close").value
        )
        self.ik_max_iters = int(self.get_parameter("ik_max_iters").value)
        self.ik_pos_tol = float(self.get_parameter("ik_pos_tol").value)
        self.ik_rot_tol = np.deg2rad(float(self.get_parameter("ik_rot_tol_deg").value))

        if self.orientation_rpy.shape != (3,):
            raise ValueError("orientation_rpy must contain exactly 3 values")
        if self.home_q.shape != (ARM_JOINT_COUNT,):
            raise ValueError("home_q must contain exactly 5 values")
        if self.location_a.shape != (3,) or self.location_b.shape != (3,):
            raise ValueError("location_a and location_b must contain exactly 3 values")

        self.pub = self.create_publisher(JointTrajectory, self.topic, 10)
        self.current_q = self.home_q.copy()
        self.current_gripper = self._clip_gripper(self.gripper_open)

        self.stages = self._build_stage_sequence()
        self.stage_index = 0
        self.stage_started = None
        self.stage_active = False
        self.done = False

        self.timer = self.create_timer(self.tick_s, self._tick)
        self.get_logger().info(
            "pick_place_open_loop publishing to {} with A={} B={} grasp_height={:.3f} m".format(
                self.topic,
                np.round(self.location_a, 3).tolist(),
                np.round(self.location_b, 3).tolist(),
                self.grasp_height,
            )
        )

    def _clip_gripper(self, value):
        return float(np.clip(value, 0.0, 1.0))

    def _pose_from_location(self, location, z):
        return (float(location[0]), float(location[1]), float(z))

    def _build_stage_sequence(self):
        a_hover = self._pose_from_location(self.location_a, self.hover_height)
        a_pick = self._pose_from_location(self.location_a, self.grasp_height)
        a_travel = self._pose_from_location(self.location_a, self.travel_height)
        b_hover = self._pose_from_location(self.location_b, self.hover_height)
        b_pick = self._pose_from_location(self.location_b, self.grasp_height)
        b_travel = self._pose_from_location(self.location_b, self.travel_height)

        return [
            Stage(
                name="home_open",
                gripper=self.gripper_open,
                hold_s=2.0,
                use_ik=False,
                q_override=self.home_q.tolist(),
            ),
            Stage(name="approach_a", xyz=a_hover, gripper=self.gripper_open, hold_s=1.5),
            Stage(name="descend_a", xyz=a_pick, gripper=self.gripper_open, hold_s=1.2),
            Stage(
                name="grasp_a",
                xyz=a_pick,
                gripper=self.gripper_gentle_close,
                hold_s=1.5,
            ),
            Stage(name="lift_a", xyz=a_travel, gripper=self.gripper_gentle_close, hold_s=1.5),
            Stage(
                name="move_above_b",
                xyz=b_travel,
                gripper=self.gripper_gentle_close,
                hold_s=1.5,
            ),
            Stage(name="descend_b", xyz=b_pick, gripper=self.gripper_gentle_close, hold_s=1.2),
            Stage(name="release_b", xyz=b_pick, gripper=self.gripper_open, hold_s=1.5),
            Stage(name="lift_from_b", xyz=b_travel, gripper=self.gripper_open, hold_s=1.5),
            Stage(
                name="home_after_drop",
                gripper=self.gripper_open,
                hold_s=2.0,
                use_ik=False,
                q_override=self.home_q.tolist(),
            ),
            Stage(name="approach_b", xyz=b_hover, gripper=self.gripper_open, hold_s=1.5),
            Stage(name="descend_b_pick", xyz=b_pick, gripper=self.gripper_open, hold_s=1.2),
            Stage(
                name="grasp_b",
                xyz=b_pick,
                gripper=self.gripper_gentle_close,
                hold_s=1.5,
            ),
            Stage(name="lift_b", xyz=b_travel, gripper=self.gripper_gentle_close, hold_s=1.5),
            Stage(
                name="move_above_a",
                xyz=a_travel,
                gripper=self.gripper_gentle_close,
                hold_s=1.5,
            ),
            Stage(name="descend_a_place", xyz=a_pick, gripper=self.gripper_gentle_close, hold_s=1.2),
            Stage(name="release_a", xyz=a_pick, gripper=self.gripper_open, hold_s=1.5),
            Stage(name="lift_from_a", xyz=a_travel, gripper=self.gripper_open, hold_s=1.5),
            Stage(
                name="home_done",
                gripper=self.gripper_open,
                hold_s=2.0,
                use_ik=False,
                q_override=self.home_q.tolist(),
            ),
        ]

    def _solve_stage_q(self, stage):
        if not stage.use_ik:
            if stage.q_override is None:
                raise RuntimeError(f"stage '{stage.name}' is missing q_override")
            q = np.asarray(stage.q_override, dtype=float)
            if q.shape != (ARM_JOINT_COUNT,):
                raise RuntimeError(
                    f"stage '{stage.name}' q_override must contain exactly 5 joints"
                )
            return q

        if stage.xyz is None:
            raise RuntimeError(f"stage '{stage.name}' is missing xyz pose")

        result = ik_coordinate_descent(
            stage.xyz[0],
            stage.xyz[1],
            stage.xyz[2],
            self.orientation_rpy[0],
            self.orientation_rpy[1],
            self.orientation_rpy[2],
            q_init=self.current_q,
            max_iters=self.ik_max_iters,
            pos_tol=self.ik_pos_tol,
            rot_tol=self.ik_rot_tol,
            optimize_orientation=False,
        )
        if not result["success"]:
            raise RuntimeError(
                "IK failed for stage '{}' at xyz=({}, {}, {}) pos_error={:.5f}".format(
                    stage.name,
                    stage.xyz[0],
                    stage.xyz[1],
                    stage.xyz[2],
                    result["pos_error_raw"],
                )
            )
        return np.asarray(result["q_raw"], dtype=float)

    def _build_msg(self, q, gripper):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()

        pt = JointTrajectoryPoint()
        pt.positions = [
            float(q[0]),
            float(q[1]),
            float(q[2]),
            float(q[3]),
            float(q[4]),
            self._clip_gripper(gripper),
        ]
        pt.velocities = [0.0] * len(pt.positions)
        pt.time_from_start = Duration(
            sec=int(self.move_time), nanosec=int((self.move_time % 1.0) * 1e9)
        )
        msg.points = [pt]
        return msg

    def _start_stage(self, stage):
        q_target = self._solve_stage_q(stage)
        gripper_target = (
            self.current_gripper if stage.gripper is None else self._clip_gripper(stage.gripper)
        )

        self.pub.publish(self._build_msg(q_target, gripper_target))
        self.current_q = q_target
        self.current_gripper = gripper_target
        self.stage_started = self.get_clock().now()
        self.stage_active = True

        pose_text = (
            "joint target"
            if stage.xyz is None
            else "xyz=({:.3f}, {:.3f}, {:.3f})".format(*stage.xyz)
        )
        self.get_logger().info(
            "stage {} / {}: {} {} gripper={:.2f}".format(
                self.stage_index + 1,
                len(self.stages),
                stage.name,
                pose_text,
                gripper_target,
            )
        )

    def _tick(self):
        if self.done:
            return

        if self.stage_index >= len(self.stages):
            self.done = True
            self.timer.cancel()
            self.get_logger().info("pick and place sequence complete")
            return

        stage = self.stages[self.stage_index]
        if not self.stage_active:
            self._start_stage(stage)
            return

        elapsed = (self.get_clock().now() - self.stage_started).nanoseconds * 1e-9
        if elapsed < stage.hold_s:
            return

        self.stage_index += 1
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

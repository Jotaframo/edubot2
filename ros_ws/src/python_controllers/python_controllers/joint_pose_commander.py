import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


POSE_TABLE = [
    ("I", [-0.9205, 0.6447, -0.8765, 0.2326, 1.5708]),
    ("II", [-1.3034, -0.1819, 0.4459, 1.3068, -0.2666]),
    ("III", [-0.0008, 0.9390, -0.1535, 1.5700, 1.5708]),
    ("IV", [-2.0000, 1.3089, -1.3107, -1.5700, 2.0000]),
    ("V_a", [-0.0005, 0.2271, 0.8200, 1.3087, 0.0000]),
    ("V_b", [-0.0005, -0.1821, 1.5800, 0.9510, 0.0000]),
]


class JointTableCommander(Node):
    def __init__(self):
        super().__init__("joint_table_commander")

        self.declare_parameter("topic", "/joint_cmds")
        self.declare_parameter("hold_s", 2.5)
        self.declare_parameter("loop", False)
        self.declare_parameter("mode", "hardware")
        self.declare_parameter("gripper_hardware", 0.12)
        self.declare_parameter("gripper_simulation", 0.5)
        self.declare_parameter("point_time_s", 0.25)
        self.declare_parameter("wait_for_subscriber", True)

        self.topic = str(self.get_parameter("topic").value)
        self.hold_s = float(self.get_parameter("hold_s").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.mode = str(self.get_parameter("mode").value).strip().lower()
        self.gripper_hardware = float(self.get_parameter("gripper_hardware").value)
        self.gripper_simulation = float(self.get_parameter("gripper_simulation").value)
        self.point_time_s = float(self.get_parameter("point_time_s").value)
        self.wait_for_subscriber = bool(self.get_parameter("wait_for_subscriber").value)

        if self.mode not in {"hardware", "simulation", "sim"}:
            self.get_logger().warn(
                f"Unknown mode '{self.mode}', defaulting to hardware"
            )
            self.mode = "hardware"

        if self.mode in {"simulation", "sim"}:
            self.gripper = self.gripper_simulation
        else:
            self.gripper = self.gripper_hardware

        self.pub = self.create_publisher(JointTrajectory, self.topic, 10)
        self.timer = self.create_timer(0.05, self._tick)

        self.pose_index = 0
        self.pose_start = self.get_clock().now()

        self.get_logger().info(
            f"Publishing {len(POSE_TABLE)} joint poses to {self.topic} "
            f"(mode={self.mode}, hold_s={self.hold_s}, loop={self.loop}, gripper={self.gripper})"
        )

    def _publish_pose(self, joint_positions):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()

        pt = JointTrajectoryPoint()
        pt.positions = [
            float(joint_positions[0]),
            float(joint_positions[1]),
            float(joint_positions[2]),
            float(joint_positions[3]),
            float(joint_positions[4]),
            float(self.gripper),
        ]
        pt.time_from_start = Duration(
            sec=int(self.point_time_s),
            nanosec=int((self.point_time_s - int(self.point_time_s)) * 1e9),
        )
        msg.points = [pt]
        self.pub.publish(msg)

    def _tick(self):
        if self.wait_for_subscriber and self.pub.get_subscription_count() == 0:
            self.get_logger().warn(
                f"Waiting for subscriber on {self.topic}...",
                throttle_duration_sec=2.0,
            )
            return

        if self.pose_index >= len(POSE_TABLE):
            if self.loop:
                self.pose_index = 0
                self.pose_start = self.get_clock().now()
            else:
                self.get_logger().info("Completed joint pose table.")
                self.timer.cancel()
                return

        pose_name, pose_joints = POSE_TABLE[self.pose_index]
        self._publish_pose(pose_joints)

        elapsed = (self.get_clock().now() - self.pose_start).nanoseconds * 1e-9
        if elapsed >= self.hold_s:
            self.get_logger().info(f"Pose {pose_name} done")
            self.pose_index += 1
            self.pose_start = self.get_clock().now()


def main(args=None):
    rclpy.init(args=args)
    node = JointTableCommander()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

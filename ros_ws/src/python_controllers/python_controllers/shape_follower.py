import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from python_controllers.Inverse_Kinematics import analytical_ik_closed_form
from python_controllers.trajectory_generator import generate_trajectory


class ShapeFollower(Node):
    def __init__(self):
        super().__init__("shape_follower")

        self.declare_parameter("topic", "/joint_cmds")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("move_time", 0.10)
        self.declare_parameter("shape", "square")
        self.declare_parameter("plane", "xz")
        self.declare_parameter("size", 0.08)
        self.declare_parameter("duration_s", 12.0)
        self.declare_parameter("center_x", 0.20)
        self.declare_parameter("center_y", 0.00)
        self.declare_parameter("center_z", 0.10)
        self.declare_parameter("pitch", 0.0)

        # Effective link lengths in meters.
        self.declare_parameter("l2", 0.11167)
        self.declare_parameter("l3", 0.16000)
        self.declare_parameter("l4", 0.15000)

        self.joint_limits = [
            (-2.0, 2.0),      # Shoulder_Rotation
            (-1.57, 1.57),    # Shoulder_Pitch
            (-1.58, 1.58),    # Elbow
            (-1.57, 1.57),    # Wrist_Pitch
        ]
        self.joint_names = [
            "Shoulder_Rotation",
            "Shoulder_Pitch",
            "Elbow",
            "Wrist_Pitch",
            "Wrist_Roll",
        ]

        topic = self.get_parameter("topic").get_parameter_value().string_value
        rate_hz = self.get_parameter("rate_hz").get_parameter_value().double_value

        self.traj_pub = self.create_publisher(JointTrajectory, topic, 10)
        self.start_time = self.get_clock().now()
        self.last_q = [0.0, 0.0, 0.0, 0.0]
        self.failed_count = 0
        self.trajectory = self._build_trajectory(rate_hz)

        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(f"shape_follower publishing to {topic} at {rate_hz:.1f} Hz")
        self.get_logger().info(
            f"shape={self.get_parameter('shape').value}, "
            f"plane={self.get_parameter('plane').value}, "
            f"size={self.get_parameter('size').value:.3f} m"
        )

    def _build_trajectory(self, rate_hz):
        shape = self.get_parameter("shape").get_parameter_value().string_value
        plane = self.get_parameter("plane").get_parameter_value().string_value
        size = self.get_parameter("size").get_parameter_value().double_value
        duration = self.get_parameter("duration_s").get_parameter_value().double_value
        cx = self.get_parameter("center_x").get_parameter_value().double_value
        cy = self.get_parameter("center_y").get_parameter_value().double_value
        cz = self.get_parameter("center_z").get_parameter_value().double_value
        return generate_trajectory(
            shape_name=shape,
            center_xyz=(cx, cy, cz),
            plane=plane,
            size=size,
            duration_s=duration,
            rate_hz=rate_hz,
        )

    def _target_point(self, t):
        duration = self.get_parameter("duration_s").get_parameter_value().double_value
        if not self.trajectory:
            return 0.0, 0.0, 0.0
        if duration <= 0.0:
            p = self.trajectory[0]
            return p.x, p.y, p.z

        t_mod = t % duration
        n = len(self.trajectory)
        dt = duration / n
        i0 = int(t_mod / dt) % n
        i1 = (i0 + 1) % n
        alpha = (t_mod - i0 * dt) / dt

        p0 = self.trajectory[i0]
        p1 = self.trajectory[i1]
        x = (1.0 - alpha) * p0.x + alpha * p1.x
        y = (1.0 - alpha) * p0.y + alpha * p1.y
        z = (1.0 - alpha) * p0.z + alpha * p1.z
        return x, y, z

    def _within_limits(self, q):
        for i, (lo, hi) in enumerate(self.joint_limits):
            if q[i] < lo or q[i] > hi:
                return False
        return True

    def _closest_solution(self, solutions):
        best = None
        best_cost = None
        for s in solutions:
            if not self._within_limits(s):
                continue
            cost = sum((s[i] - self.last_q[i]) ** 2 for i in range(4))
            if best is None or cost < best_cost:
                best = s
                best_cost = cost
        return best

    def _build_msg(self, q):
        t = self.get_parameter("move_time").get_parameter_value().double_value
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.joint_names

        pt = JointTrajectoryPoint()
        pt.positions = [float(q[0]), float(q[1]), float(q[2]), float(q[3]), 0.0]
        pt.velocities = [0.0] * len(self.joint_names)
        pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
        msg.points = [pt]
        return msg

    def _tick(self):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds * 1e-9
        x, y, z = self._target_point(t)

        pitch = self.get_parameter("pitch").get_parameter_value().double_value
        l2 = self.get_parameter("l2").get_parameter_value().double_value
        l3 = self.get_parameter("l3").get_parameter_value().double_value
        l4 = self.get_parameter("l4").get_parameter_value().double_value
        sols = analytical_ik_closed_form(x, y, z, pitch, l2, l3, l4)
        q = self._closest_solution(sols)

        if q is None:
            self.failed_count += 1
            if self.failed_count % 20 == 1:
                self.get_logger().warn(
                    f"No IK solution in limits for target ({x:.3f}, {y:.3f}, {z:.3f})"
                )
            return

        self.failed_count = 0
        self.last_q = [q[0], q[1], q[2], q[3]]
        self.traj_pub.publish(self._build_msg(q))


def main(args=None):
    rclpy.init(args=args)
    node = ShapeFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

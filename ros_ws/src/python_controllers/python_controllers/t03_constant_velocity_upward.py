"""
ROS 2 velocity controller. tracks a constant end-effector velocity.

At each timestep node:
  1. reads joint angles from /joint_states
  2. builds the 3x4 linear Jacobian via finite differences
  3. checks for singularity (SVD)
  4. computes joint velocities with the pseudoinverse
  5. clamps and publishes the velocity command

Usage:  ros2 run python_controllers constant_velocity_upward
"""

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

try:
    from python_controllers.t03_Jacobian_FINAL import jacobian_finite_difference_final
    from python_controllers.t01_Forward_Kinematics_FINAL import forward_kinematics_full
except ModuleNotFoundError:
    from t03_Jacobian_FINAL import jacobian_finite_difference_final
    from t01_Forward_Kinematics_FINAL import forward_kinematics_full


JOINT_NAMES = [
    "Shoulder_Rotation",
    "Shoulder_Pitch",
    "Elbow",
    "Wrist_Pitch",
]

ALL_JOINT_NAMES = JOINT_NAMES + ["Wrist_Roll"]

# Joint limits from URDF
JOINT_LIMITS = np.array(
    [
        [-2.0,  2.0 ],
        [-1.57, 1.57],
        [-1.58, 1.58],
        [-1.57, 1.57],
    ],
    dtype=float,
)

# Home position (rad)  matching robot_hw.yaml
HOME_Q = [0.0, 0.23, -1.2217, -1.0472]


class ConstantVelocityUpward(Node):
    """Track a constant EE velocity using J† = J^T (J J^T)^-1."""

    def __init__(self):
        super().__init__("constant_velocity_upward")

        self.declare_parameter("topic", "/joint_cmds")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("use_joint_state_feedback", True)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("duration_s", 30.0)
        self.declare_parameter("ee_velocity_xyz", [0.00, 0.0, 0.02])  # Move upward only
        self.declare_parameter("q_start", HOME_Q)
        self.declare_parameter("min_sigma", 0.01)      # Singularity threshold (pseudoinverse guard)
        self.declare_parameter("max_joint_velocity", 0.3)
        self.declare_parameter("return_home", True)
        self.declare_parameter("home_tolerance", 0.05)  # Tolerance for reaching home (rad)

        topic = self.get_parameter("topic").value
        joint_state_topic = self.get_parameter("joint_state_topic").value
        self.use_joint_state_feedback = bool(
            self.get_parameter("use_joint_state_feedback").value
        )
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.dt = 1.0 / self.rate_hz
        self.duration_s = float(self.get_parameter("duration_s").value)
        self.ee_velocity = np.asarray(
            self.get_parameter("ee_velocity_xyz").value, dtype=float
        )
        self.q_seed = np.asarray(self.get_parameter("q_start").value, dtype=float)
        self.min_sigma = float(self.get_parameter("min_sigma").value)
        self.max_joint_velocity = float(self.get_parameter("max_joint_velocity").value)
        self.return_home = bool(self.get_parameter("return_home").value)
        self.home_tolerance = float(self.get_parameter("home_tolerance").value)

        if self.ee_velocity.shape != (3,):
            raise ValueError("ee_velocity_xyz must contain exactly 3 values")
        if self.q_seed.shape != (4,):
            raise ValueError("q_start must contain exactly 4 values")

        self.current_q = self.q_seed.copy()
        self.start_time = None
        self.returning_home = False
        self.home_return_start = None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.state_sub = self.create_subscription(
            JointState, joint_state_topic, self._on_joint_state, qos
        )
        self.traj_pub = self.create_publisher(JointTrajectory, topic, 10)
        self.marker_pub = self.create_publisher(Marker, 'end_effector_trace', 10)
        self.done = False

        self._max_trace_points = 5000
        self._init_trace_marker()

        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"constant_velocity_upward publishing to {topic} at {self.rate_hz:.1f} Hz"
        )
        self.get_logger().info(
            "Moving UPWARD: ee_velocity_xyz=[{:.4f}, {:.4f}, {:.4f}] m/s, duration_s={:.2f}".format(
                self.ee_velocity[0],
                self.ee_velocity[1],
                self.ee_velocity[2],
                self.duration_s,
            )
        )

    def _on_joint_state(self, msg: JointState):
        """Callback: update current_q from encoder feedback."""
        if not self.use_joint_state_feedback:
            return

        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        if any(name not in name_to_idx for name in JOINT_NAMES):
            return

        q = np.zeros(4, dtype=float)
        for i, name in enumerate(JOINT_NAMES):
            q[i] = msg.position[name_to_idx[name]]
        self.current_q = q

        if self.start_time is None:
            self.start_time = self.get_clock().now()
            self.get_logger().info(
                "first joint state received — starting timer. q={}".format(
                    np.round(q, 3).tolist()
                )
            )

    def _build_vel_msg(self, qdot):
        """Pack joint velocities into a JointTrajectory message."""
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = ALL_JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.velocities = [
            float(qdot[0]),
            float(qdot[1]),
            float(qdot[2]),
            float(qdot[3]),
            0.0,
        ]
        pt.time_from_start = Duration(sec=0, nanosec=int(self.dt * 1e9))
        msg.points = [pt]
        return msg

    def _build_pos_msg(self, q, move_time=4.0):
        """Position command used for return-to-home."""
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = ALL_JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions  = [float(q[0]), float(q[1]), float(q[2]), float(q[3]), 0.0]
        pt.velocities = [0.0] * 5
        pt.time_from_start = Duration(sec=int(move_time), nanosec=0)
        msg.points = [pt]
        return msg

    def _effective_q(self):
        return self.current_q.copy()

    def _init_trace_marker(self):
        """Initialize the trace marker for RViz visualization"""
        self._trace_marker = Marker()
        self._trace_marker.header.frame_id = "world"
        self._trace_marker.ns = "velocity_trace"
        self._trace_marker.id = 0
        self._trace_marker.type = Marker.LINE_STRIP
        self._trace_marker.action = Marker.ADD
        self._trace_marker.scale.x = 0.005
        self._trace_marker.color.r = 1.0
        self._trace_marker.color.g = 0.0
        self._trace_marker.color.b = 1.0
        self._trace_marker.color.a = 1.0
        self._trace_marker.pose.orientation.w = 1.0

    def _publish_trace_point(self, x, y, z, stamp):
        """Add a point to the end-effector trace and publish"""
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)

        self._trace_marker.points.append(point)
        if len(self._trace_marker.points) > self._max_trace_points:
            self._trace_marker.points = self._trace_marker.points[-self._max_trace_points:]

        self._trace_marker.header.stamp = stamp.to_msg()
        self.marker_pub.publish(self._trace_marker)

    def _stop(self, reason):
        """Send zero velocity, optionally return to home, then shut down"""
        if self.done:
            return
        self.done = True
        self.timer.cancel()
        self.traj_pub.publish(self._build_vel_msg(np.zeros(4, dtype=float)))
        self.get_logger().warn(reason)

        if self.return_home:
            self.get_logger().info("initiating return to home...")
            self.returning_home = True
            self.home_return_start = self.get_clock().now()
            self.traj_pub.publish(self._build_pos_msg(HOME_Q, move_time=4.0))
        else:
            self.get_logger().info("execution complete, not returning to home")
            raise KeyboardInterrupt()

    def _tick(self):
        """Main control loop (called at 20 Hz by timer)."""
        if self.done and not self.returning_home:
            return

        # Handle return-to-home sequence
        if self.returning_home:
            if self.home_return_start is None:
                return
            
            elapsed = (self.get_clock().now() - self.home_return_start).nanoseconds * 1e-9
            q_error = np.linalg.norm(self.current_q - HOME_Q)
            
            # Check if home is reached (within tolerance or timeout after 6 seconds)
            if q_error < self.home_tolerance or elapsed > 6.0:
                if q_error < self.home_tolerance:
                    self.get_logger().info(
                        "successfully returned to home. q_error={:.6f}".format(q_error)
                    )
                else:
                    self.get_logger().warn(
                        "return-to-home timeout after {:.1f}s, q_error={:.6f}".format(
                            elapsed, q_error
                        )
                    )
                self.timer.cancel()
                raise KeyboardInterrupt()
            return

        if self.start_time is None:
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if elapsed >= self.duration_s:
            self._stop("constant velocity upward segment complete")
            return

        q = self._effective_q()
        jac = jacobian_finite_difference_final(q)
        singular_values = np.linalg.svd(jac, compute_uv=False)
        sigma_min = float(np.min(singular_values))
        rank = int(np.sum(singular_values > self.min_sigma))

        if rank < 3 or sigma_min < self.min_sigma:
            self._stop(
                "aborting near singularity: rank={} sigma_min={:.6f}".format(
                    rank, sigma_min
                )
            )
            return

        # Right pseudoinverse: qdot = J^T (J J^T)^-1 * x_dot
        # Gives minimum-norm joint velocity for the underdetermined 3x4 system
        jj_t = jac @ jac.T
        qdot = jac.T @ np.linalg.solve(jj_t, self.ee_velocity)

        # Clamp joint velocities for safety
        max_abs_vel = float(np.max(np.abs(qdot)))
        if max_abs_vel > self.max_joint_velocity:
            qdot *= self.max_joint_velocity / max_abs_vel

        # Check that the next step stays within URDF joint limits
        q_next = q + qdot * self.dt
        if np.any(q_next < JOINT_LIMITS[:, 0]) or np.any(q_next > JOINT_LIMITS[:, 1]):
            self._stop("aborting on joint limit guard")
            return

        if not self.use_joint_state_feedback:
            self.current_q = q_next

        self.traj_pub.publish(self._build_vel_msg(qdot))

        # Publish end-effector trace
        fk = forward_kinematics_full(q[0], q[1], q[2], q[3], 0.0)
        ee_x, ee_y, ee_z = fk[0, 3], fk[1, 3], fk[2, 3]
        self._publish_trace_point(ee_x, ee_y, ee_z, self.get_clock().now())


def main(args=None):
    rclpy.init(args=args)
    node = ConstantVelocityUpward()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

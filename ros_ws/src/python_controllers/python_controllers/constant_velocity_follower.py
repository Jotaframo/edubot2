import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from python_controllers.Forward_Kinematics import jacobian_finite_difference


JOINT_NAMES = [
    "Shoulder_Rotation",
    "Shoulder_Pitch",
    "Elbow",
    "Wrist_Pitch",
]

ALL_JOINT_NAMES = JOINT_NAMES + ["Wrist_Roll"]

# Widened Shoulder_Pitch upper limit — sim home is 1.833 rad, URDF says 1.57
JOINT_LIMITS = np.array(
    [
        [-2.0,  2.0 ],
        [-1.57, 1.57],
        [-1.58, 1.58],
        [-1.57, 1.57],
    ],
    dtype=float,
)

# Sim home from lerobot_sim.cpp: []degrees
HOME_Q = [0.0, 0.0, -0.0, -0.0] #edited 


class ConstantVelocityFollower(Node):
    def __init__(self):
        super().__init__("constant_velocity_follower")

        self.declare_parameter("topic", "/joint_cmds")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("use_joint_state_feedback", True)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("duration_s", 15.0)
        self.declare_parameter("ee_velocity_xyz", [0.0, 0.0, 0.01])
        self.declare_parameter("q_start", HOME_Q)   # matches sim home
        self.declare_parameter("damping_lambda", 0.03)
        self.declare_parameter("min_sigma", 0.02)
        self.declare_parameter("max_joint_velocity", 0.25)
        self.declare_parameter("return_home", True)

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
        self.damping_lambda = float(self.get_parameter("damping_lambda").value)
        self.min_sigma = float(self.get_parameter("min_sigma").value)
        self.max_joint_velocity = float(self.get_parameter("max_joint_velocity").value)
        self.return_home = bool(self.get_parameter("return_home").value)

        if self.ee_velocity.shape != (3,):
            raise ValueError("ee_velocity_xyz must contain exactly 3 values")
        if self.q_seed.shape != (4,):
            raise ValueError("q_start must contain exactly 4 values")

        self.current_q = self.q_seed.copy()
        self.start_time = None   # starts on first joint state, not node launch

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.state_sub = self.create_subscription(
            JointState, joint_state_topic, self._on_joint_state, qos
        )
        self.traj_pub = self.create_publisher(JointTrajectory, topic, 10)
        self.done = False

        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"constant_velocity_follower publishing to {topic} at {self.rate_hz:.1f} Hz"
        )
        self.get_logger().info(
            "ee_velocity_xyz=[{:.4f}, {:.4f}, {:.4f}] m/s, duration_s={:.2f}".format(
                self.ee_velocity[0],
                self.ee_velocity[1],
                self.ee_velocity[2],
                self.duration_s,
            )
        )

    def _on_joint_state(self, msg: JointState):
        if not self.use_joint_state_feedback:
            return

        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        if any(name not in name_to_idx for name in JOINT_NAMES):
            return

        q = np.zeros(4, dtype=float)
        for i, name in enumerate(JOINT_NAMES):
            q[i] = msg.position[name_to_idx[name]]
        self.current_q = q

        # Start the motion timer on first real feedback
        if self.start_time is None:
            self.start_time = self.get_clock().now()
            self.get_logger().info(
                "first joint state received — starting timer. q={}".format(
                    np.round(q, 3).tolist()
                )
            )

    def _build_vel_msg(self, qdot):
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

    def _stop(self, reason):
        if self.done:
            return
        self.done = True
        self.timer.cancel()
        self.traj_pub.publish(self._build_vel_msg(np.zeros(4, dtype=float)))
        self.get_logger().warn(reason)

        if self.return_home:
            self.get_logger().info("returning to home...")
            self.traj_pub.publish(self._build_pos_msg(HOME_Q, move_time=4.0))

    def _tick(self):
        if self.done:
            return

        # Wait until first joint state arrives before starting
        if self.start_time is None:
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if elapsed >= self.duration_s:
            self._stop("constant velocity segment complete")
            return

        q = self._effective_q()
        jac = jacobian_finite_difference(q)
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

        jj_t = jac @ jac.T
        damped = jj_t + (self.damping_lambda ** 2) * np.eye(3)
        qdot = jac.T @ np.linalg.solve(damped, self.ee_velocity)

        max_abs_vel = float(np.max(np.abs(qdot)))
        if max_abs_vel > self.max_joint_velocity:
            qdot *= self.max_joint_velocity / max_abs_vel

        q_next = q + qdot * self.dt
        if np.any(q_next < JOINT_LIMITS[:, 0]) or np.any(q_next > JOINT_LIMITS[:, 1]):
            self._stop("aborting on joint limit guard")
            return

        if not self.use_joint_state_feedback:
            self.current_q = q_next

        self.traj_pub.publish(self._build_vel_msg(qdot))


def main(args=None):
    rclpy.init(args=args)
    node = ConstantVelocityFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
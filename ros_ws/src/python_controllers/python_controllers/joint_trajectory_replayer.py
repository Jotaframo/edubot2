import csv

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class JointTrajectoryReplayer(Node):
    def __init__(self):
        super().__init__("joint_trajectory_replayer")
        self.declare_parameter("topic", "/joint_cmds")
        self.declare_parameter("trajectory_csv", "/tmp/offline_joint_trajectory.csv")
        self.declare_parameter("move_time", 0.10)

        self.topic = self.get_parameter("topic").value
        self.trajectory_csv = self.get_parameter("trajectory_csv").value
        self.move_time = float(self.get_parameter("move_time").value)
        self.rows = self._load_rows(self.trajectory_csv)
        if not self.rows:
            raise RuntimeError(f"No trajectory rows found in {self.trajectory_csv}")

        self.pub = self.create_publisher(JointTrajectory, self.topic, 10)
        self.joint_names = [
            "Shoulder_Rotation",
            "Shoulder_Pitch",
            "Elbow",
            "Wrist_Pitch",
            "Wrist_Roll",
        ]
        self.index = 0

        if len(self.rows) > 1:
            self.dt = max(1e-3, self.rows[1]["t"] - self.rows[0]["t"])
        else:
            self.dt = self.move_time
        self.timer = self.create_timer(self.dt, self._tick)

    def _load_rows(self, path):
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "t": float(row["t"]),
                        "q": [
                            float(row["q1"]),
                            float(row["q2"]),
                            float(row["q3"]),
                            float(row["q4"]),
                            float(row["q5"]),
                        ],
                    }
                )
        return rows

    def _build_msg(self, q):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.joint_names

        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.velocities = [0.0] * len(self.joint_names)
        pt.time_from_start = Duration(
            sec=int(self.move_time), nanosec=int((self.move_time % 1.0) * 1e9)
        )
        msg.points = [pt]
        return msg

    def _tick(self):
        if self.index >= len(self.rows):
            self.timer.cancel()
            self.get_logger().info("trajectory replay complete")
            return
        self.pub.publish(self._build_msg(self.rows[self.index]["q"]))
        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = JointTrajectoryReplayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

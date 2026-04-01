import rclpy
import numpy as np
import cv2
import numpy as np
from rclpy.node import Node
try:
    from python_controllers.t02_Inverse_Kinematics_Numerical import ik_coordinate_descent
    from python_controllers.t01_Forward_Kinematics_FINAL import forward_kinematics_full
except ModuleNotFoundError:
    from t02_Inverse_Kinematics_Numerical import ik_coordinate_descent
    from t01_Forward_Kinematics_FINAL import forward_kinematics_full
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class SilhouetteTraj(Node):

    def __init__(self, image_path='TU_Delft_Logo.png', plane='horizontal'):
        super().__init__('silhouette_trajectory')

        # Safe home pose for the 5-DOF arm
        self._HOME = np.array([
            np.deg2rad(0), np.deg2rad(0),
            np.deg2rad(0), np.deg2rad(0),
            np.deg2rad(0)
        ])

        self._last_q = self._HOME.copy()
        self._beginning = self.get_clock().now()
        self._plane = plane.lower()  # 'horizontal', 'xz', or 'yz'
        
        # Longer cycle for complex silhouettes
        self._cycle_time = 20.0 
        self._last_cycle_t = None

        # Image processing and waypoint generation
        self._waypoints = []
        self._cumulative_distances = []
        self._total_perimeter = 0.0
        self._load_and_scale_silhouette(image_path)

        # Publishers
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        self._marker_pub = self.create_publisher(Marker, 'end_effector_trace', 10)
        self._marker_pub_compat = self.create_publisher(Marker, 'fk_ee', 10)
        
        self._max_trace_points = 5000
        self._init_trace_marker()

        timer_period = 0.02  # 25 Hz
        self._timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info("Starting Silhouette Trajectory...")

    def _load_and_scale_silhouette(self, image_path):
        """Extracts the contour from the image and maps it to the robot's workspace."""


        # Read the image in grayscale with OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error(f"Could not load image at {image_path} :(")
            return
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV) # Threshold for a black shape on white background

        # Find contours and keep the largest one
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        if not contours:
            self.get_logger().error("No contours found in the image.")
            return
        

        
        main_contour = max(contours, key=cv2.contourArea).squeeze()
        # Normalize contour coordinates to [0, 1]
        x_coords = main_contour[:, 0]
        y_coords = main_contour[:, 1]
        
        x_min_img, x_max_img = x_coords.min(), x_coords.max()
        y_min_img, y_max_img = y_coords.min(), y_coords.max()

        ###SCALE AND MAP SHAPE COORDINATES###
        # Invert y because base and world origins are opposite
        norm_x = (x_coords - x_min_img) / (x_max_img - x_min_img)
        norm_y = 1.0 - ((y_coords - y_min_img) / (y_max_img - y_min_img))

        # Scale to a safe 20 cm by 20 cm workspace
        workspace_l = 0.20
        workspace_x_min = -0.10
        workspace_y_min = 0.20
        scaled_x = workspace_x_min + (norm_x * workspace_l) 
        scaled_y = workspace_y_min + (norm_y * workspace_l)

        self._waypoints = np.column_stack((scaled_x, scaled_y))

        # Build cumulative arc length for smooth timing
        self._cumulative_distances = [0.0]
        for i in range(1, len(self._waypoints)):
            dist = np.linalg.norm(self._waypoints[i] - self._waypoints[i-1])
            self._cumulative_distances.append(self._cumulative_distances[-1] + dist)
        
        # Close the loop back to the first waypoint
        dist = np.linalg.norm(self._waypoints[0] - self._waypoints[-1])
        self._cumulative_distances.append(self._cumulative_distances[-1] + dist)
        self._waypoints = np.vstack((self._waypoints, self._waypoints[0]))
        
        self._total_perimeter = self._cumulative_distances[-1]
        self.get_logger().info(f"Loaded silhouette with {len(self._waypoints)} waypoints.")

    def get_silhouette_pose(self, dt):
        """Interpolates the current target pose based on the elapsed cycle time."""
        t = dt % self._cycle_time
        fraction = t / self._cycle_time
        target_dist = fraction * self._total_perimeter

        # Find the current segment
        idx = np.searchsorted(self._cumulative_distances, target_dist) - 1
        
        # Guard edge cases at loop rollover
        if idx < 0: idx = 0
        if idx >= len(self._waypoints) - 1: idx = len(self._waypoints) - 2

        # Linearly interpolate within the segment
        p1 = self._waypoints[idx]
        p2 = self._waypoints[idx + 1]
        
        segment_length = self._cumulative_distances[idx + 1] - self._cumulative_distances[idx]
        if segment_length == 0:
            a, b = p1
        else:
            segment_fraction = (target_dist - self._cumulative_distances[idx]) / segment_length
            a = p1[0] + segment_fraction * (p2[0] - p1[0])
            b = p1[1] + segment_fraction * (p2[1] - p1[1])

        # Map to chosen plane
        if self._plane == 'xz':
            x, y, z = a, 0.20, b  # Vertical XZ plane at y=0.20
        elif self._plane == 'yz':
            x, y, z = -0.10, a, b  # Vertical YZ plane at x=-0.10
        else:  # horizontal
            x, y, z = a, b, 0.10  # Horizontal XY plane at z=0.10

        roll, pitch, yaw = 0.0, 0.0, 0.0  # Initial Orientation

        return x, y, z, roll, pitch, yaw
    
    def _init_trace_marker(self):
        self._trace_marker = Marker()
        self._trace_marker.header.frame_id = "world"
        self._trace_marker.ns = "silhouette_trace"
        self._trace_marker.id = 0
        self._trace_marker.type = Marker.LINE_STRIP
        self._trace_marker.action = Marker.ADD
        self._trace_marker.scale.x = 0.005 
        self._trace_marker.color.r = 0.0
        self._trace_marker.color.g = 1.0
        self._trace_marker.color.b = 0.0
        self._trace_marker.color.a = 1.0
        self._trace_marker.pose.orientation.w = -1.0

    @staticmethod
    def _world_to_base_coords(x, y, z):
        return -float(x), -float(y), float(z)

    def _publish_trace_point(self, x, y, z, stamp):
        point = Point()
        point.x = float(x)
        point.y = -float(y)
        point.z = float(z)

        self._trace_marker.points.append(point)
        if len(self._trace_marker.points) > self._max_trace_points:
            self._trace_marker.points = self._trace_marker.points[-self._max_trace_points:]

        self._trace_marker.header.stamp = stamp.to_msg()
        self._marker_pub.publish(self._trace_marker)
        self._marker_pub_compat.publish(self._trace_marker)

    def _clear_trace(self, stamp):
        self._trace_marker.header.stamp = stamp.to_msg()
        self._trace_marker.action = Marker.DELETE
        self._marker_pub.publish(self._trace_marker)
        self._marker_pub_compat.publish(self._trace_marker)

        self._trace_marker.points = []
        self._trace_marker.action = Marker.ADD
        self._trace_marker.header.stamp = stamp.to_msg()
        self._marker_pub.publish(self._trace_marker)
        self._marker_pub_compat.publish(self._trace_marker)

    def timer_callback(self):
        # Skip update if no waypoints were loaded
        if len(self._waypoints) == 0:
            return

        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()

        # elapsed time since start in seconds for trajectory phase
        dt = (now - self._beginning).nanoseconds * (1e-9)
        cycle_t = dt % self._cycle_time
        if self._last_cycle_t is not None and cycle_t < self._last_cycle_t:
            self._clear_trace(now)
        self._last_cycle_t = cycle_t
        
        target_x, target_y, target_z, r, p, y = self.get_silhouette_pose(dt)
        
        ik_result = ik_coordinate_descent(
            target_x, target_y, target_z, r, p, y,
            q_init=self._last_q,
            max_iters=100,
            optimize_orientation=False 
        )

        if ik_result["success"]:
            self._last_q = ik_result["q_raw"]
        else:
            self.get_logger().warn(
                f"IK Failed at X:{target_x:.3f} Y:{target_y:.3f} Z:{target_z:.3f}. Holding position.", 
                throttle_duration_sec=1.0
            )

        point = JointTrajectoryPoint()
        gripper_state = 0.0 
        
        point.positions = [
            float(self._last_q[0]),
            float(self._last_q[1]),
            float(self._last_q[2]),
            float(self._last_q[3]),
            float(self._last_q[4]),
            gripper_state
        ]
        
        msg.points = [point]
        self._publisher.publish(msg)
        fk = forward_kinematics_full(*self._last_q)
        ee_x, ee_y, ee_z = self._world_to_base_coords(fk[0, 3], fk[1, 3], fk[2, 3])
        self._publish_trace_point(ee_x, ee_y, ee_z, now)


class RectangleTraj(Node):

    def __init__(self, plane='horizontal'):
        super().__init__('rectangle_trajectory')

        # Defining home pose for the 5-DOF arm
        self._HOME = np.array([
            np.deg2rad(0), np.deg2rad(0),
            np.deg2rad(0), np.deg2rad(0),
            np.deg2rad(0)
        ])

        self._last_q = self._HOME.copy()
        self._beginning = self.get_clock().now()
        self._plane = plane.lower()  # 'horizontal', 'xz', or 'yz'
        self._cycle_time = 20.0
        self._last_cycle_t = None

        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)

        self._marker_pub = self.create_publisher(Marker, 'end_effector_trace', 10) # Marker topic for the 'pen'
        self._marker_pub_compat = self.create_publisher(Marker, 'fk_ee', 10)
        self._init_trace_marker()
        self._max_trace_points = 3000


        timer_period = 0.02  # 25 Hz
        self._timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info("Starting Horizontal Rectangle Trajectory...")

    def get_rectangle_pose(self, dt):
        t = dt % self._cycle_time
        fraction = t / self._cycle_time

        # Rectangle dimensions
        l=0.20
        perimeter = 0.60
        d = fraction * perimeter  

        # Map edges to plane coords (a, b)
        if d < l/2: 
            # First edge
            a = (d / (l/2)) * l
            b = 0.0
        elif d < l/2 + l: 
            # Second edge
            a = l
            b = ((d - l/2) / l) * l
        elif d < l/2 + l + l/2: 
            # Third edge
            a = l - ((d - l/2 - l) / (l/2)) * l
            b = l
        else: 
            # Fourth edge
            a = 0.0
            b = l - ((d - (l/2 + l + l/2)) / l) * l

        # Map to chosen plane and offset
        if self._plane == 'xz':
            x = -0.10 + a  # Horizontal range in X
            y = 0.20  # Fixed Y
            z = 0.10 + b  # Vertical range in Z
        elif self._plane == 'yz':
            x = -0.10  # Fixed X
            y = 0.20 + a  # Horizontal range in Y
            z = 0.10 + b  # Vertical range in Z
        else:  # horizontal
            x = -0.10 + a  # Horizontal range in X
            y = 0.20 + b  # Horizontal range in Y
            z = 0.1  # Fixed Z
        
        roll, pitch, yaw = 0.0, 0.0, 0.0  # Keep orientation fixed

        return x, y, z, roll, pitch, yaw
    
    def _init_trace_marker(self):
        """Sets up the visual properties of the trace line."""
        self._trace_marker = Marker()
        self._trace_marker.header.frame_id = "world"
        self._trace_marker.ns = "rectangle_trace"
        self._trace_marker.id = 0
        self._trace_marker.type = Marker.LINE_STRIP
        self._trace_marker.action = Marker.ADD
        
        # Trace line width
        self._trace_marker.scale.x = 0.005 
        
        # Bright green trace color
        self._trace_marker.color.r = 0.0
        self._trace_marker.color.g = 1.0
        self._trace_marker.color.b = 0.0
        self._trace_marker.color.a = 1.0
        self._trace_marker.pose.orientation.w = 1.0

    @staticmethod
    def _world_to_base_coords(x, y, z):
        return -float(x), -float(y), float(z)

    def _publish_trace_point(self, x, y, z, stamp):
        point = Point()
        point.x = float(x)
        point.y = -float(y)
        point.z = float(z)

        self._trace_marker.points.append(point)
        if len(self._trace_marker.points) > self._max_trace_points:
            self._trace_marker.points = self._trace_marker.points[-self._max_trace_points:]

        self._trace_marker.header.stamp = stamp.to_msg()
        self._marker_pub.publish(self._trace_marker)
        self._marker_pub_compat.publish(self._trace_marker)

    def _clear_trace(self, stamp):
        self._trace_marker.header.stamp = stamp.to_msg()
        self._trace_marker.action = Marker.DELETE
        self._marker_pub.publish(self._trace_marker)
        self._marker_pub_compat.publish(self._trace_marker)

        self._trace_marker.points = []
        self._trace_marker.action = Marker.ADD
        self._trace_marker.header.stamp = stamp.to_msg()
        self._marker_pub.publish(self._trace_marker)
        self._marker_pub_compat.publish(self._trace_marker)

    def timer_callback(self):
        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()

        dt = (now - self._beginning).nanoseconds * (1e-9)
        cycle_t = dt % self._cycle_time
        if self._last_cycle_t is not None and cycle_t < self._last_cycle_t:
            self._clear_trace(now)
        self._last_cycle_t = cycle_t
        
        target_x, target_y, target_z, r, p, y = self.get_rectangle_pose(dt)
        
        # Solve IK with fixed orientation for this 5-DOF arm
        ik_result = ik_coordinate_descent(
            target_x, target_y, target_z, r, p, y,
            q_init=self._last_q,
            max_iters=100,
            optimize_orientation=False 
        )

        if ik_result["success"]:
            self._last_q = ik_result["q_raw"]
        else:
            self.get_logger().warn(
                f"IK Failed at X:{target_x:.3f} Y:{target_y:.3f} Z:{target_z:.3f}. Holding position.", 
                throttle_duration_sec=1.0
            )

        point = JointTrajectoryPoint()
        gripper_state = 0.0
        
        point.positions = [
            float(self._last_q[0]),
            float(self._last_q[1]),
            float(self._last_q[2]),
            float(self._last_q[3]),
            float(self._last_q[4]),
            gripper_state,
        ]
        
        msg.points = [point]
        self._publisher.publish(msg)
        fk = forward_kinematics_full(*self._last_q)
        ee_x, ee_y, ee_z = self._world_to_base_coords(fk[0, 3], fk[1, 3], fk[2, 3])
        self._publish_trace_point(ee_x, ee_y, ee_z, now)






def main(args=None):
    rclpy.init(args=args)
    # Choose plane: 'horizontal', 'xz', or 'yz'
    plane = 'xz'
    solution_trajrectangle_traj = RectangleTraj(plane=plane)
    tudelft_sil= SilhouetteTraj(plane=plane) 
    # Select which trajectory to run
    solution_traj = tudelft_sil

    
    try:
        rclpy.spin(solution_traj)
    except KeyboardInterrupt:
        pass
    finally:
        solution_traj.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
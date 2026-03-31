import rclpy
import numpy as np
import cv2
import numpy as np
from rclpy.node import Node
from ros_ws.src.python_controllers.python_controllers.t02_Inverse_Kinematics_Numerical import ik_coordinate_descent
from ros_ws.src.python_controllers.python_controllers.t01_Forward_Kinematics_FINAL import forward_kinematics_full
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class SilhouetteTraj(Node):

    def __init__(self, image_path='TU_Delft_Logo.png'):
        super().__init__('silhouette_trajectory')

        # Initial safe home position for a 5-DOF arm
        self._HOME = np.array([
            np.deg2rad(0), np.deg2rad(0),
            np.deg2rad(0), np.deg2rad(0),
            np.deg2rad(0)
        ])

        self._last_q = self._HOME.copy()
        self._beginning = self.get_clock().now()
        
        # Increased cycle time since the perimeter of a complex shape is longer
        self._cycle_time = 20.0 
        self._last_cycle_t = None

        # --- Image Processing & Waypoint Generation ---
        self._waypoints = []
        self._cumulative_distances = []
        self._total_perimeter = 0.0
        self._load_and_scale_silhouette(image_path)

        # --- Publishers ---
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


        ###GETTING THE CONTOUR POINTS OF THE IMAGE###
        # 1. Read the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error(f"Could not load image at {image_path} :(")
            return
        # 2. Threshold the image (Assuming black silhouette on white background)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # 3. Find contours and grab the largest one
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        if not contours:
            self.get_logger().error("No contours found in the image.")
            return
        

        
        main_contour = max(contours, key=cv2.contourArea).squeeze()
        # 4. Normalize the contour to a 0.0 to 1.0 scale
        x_coords = main_contour[:, 0]
        y_coords = main_contour[:, 1]
        
        x_min_img, x_max_img = x_coords.min(), x_coords.max()
        y_min_img, y_max_img = y_coords.min(), y_coords.max()


        ###SCALING THE SILOUHETTE COORDINATES TO FIT IN THE VERIFIES SQUARE WORKSPACE###

        # Invert Y so the image doesn't draw upside down (image origin is top-left, world origin is bottom-left)
        norm_x = (x_coords - x_min_img) / (x_max_img - x_min_img)
        norm_y = 1.0 - ((y_coords - y_min_img) / (y_max_img - y_min_img))

        # 5. Scale to the robot's safe workspace (20cm x 20cm box)
        workspace_l = 0.20
        workspace_x_min = -0.10
        workspace_y_min = 0.20
        scaled_x = workspace_x_min + (norm_x * workspace_l) 
        scaled_y = workspace_y_min + (norm_y * workspace_l)

        self._waypoints = np.column_stack((scaled_x, scaled_y))

        # 6. Calculate cumulative distances for smooth time-based interpolation
        self._cumulative_distances = [0.0]
        for i in range(1, len(self._waypoints)):
            dist = np.linalg.norm(self._waypoints[i] - self._waypoints[i-1])
            self._cumulative_distances.append(self._cumulative_distances[-1] + dist)
        
        # Close the loop (distance from last point back to first point)
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

        # Find the segment we are currently on
        idx = np.searchsorted(self._cumulative_distances, target_dist) - 1
        
        # Protect against edge cases at exact loop rollover
        if idx < 0: idx = 0
        if idx >= len(self._waypoints) - 1: idx = len(self._waypoints) - 2

        # Linear interpolation between the two waypoints of the current segment
        p1 = self._waypoints[idx]
        p2 = self._waypoints[idx + 1]
        
        segment_length = self._cumulative_distances[idx + 1] - self._cumulative_distances[idx]
        if segment_length == 0:
            x, y = p1
        else:
            segment_fraction = (target_dist - self._cumulative_distances[idx]) / segment_length
            x = p1[0] + segment_fraction * (p2[0] - p1[0])
            y = p1[1] + segment_fraction * (p2[1] - p1[1])

        z = 0.10  # Raised slightly to safely clear the table
        roll, pitch, yaw = 0.0, 0.0, 0.0  # Fixed downward orientation

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
        # Safety check to ensure image loaded correctly
        if len(self._waypoints) == 0:
            return

        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()

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

    def __init__(self):
        super().__init__('rectangle_trajectory')

        # Initial safe home position for a 5-DOF arm
        self._HOME = np.array([
            np.deg2rad(0), np.deg2rad(0),
            np.deg2rad(0), np.deg2rad(0),
            np.deg2rad(0)
        ])

        self._last_q = self._HOME.copy()
        self._beginning = self.get_clock().now()
        self._cycle_time = 20.0
        self._last_cycle_t = None

        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)

        self._marker_pub = self.create_publisher(Marker, 'end_effector_trace', 10)
        self._marker_pub_compat = self.create_publisher(Marker, 'fk_ee', 10)
        self._init_trace_marker()
        self._max_trace_points = 3000


        timer_period = 0.2  # 25 Hz
        self._timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info("Starting Horizontal Rectangle Trajectory...")

    def get_rectangle_pose(self, dt):
        t = dt % self._cycle_time
        fraction = t / self._cycle_time

        # FIX 1 & 2: Moved into safe workspace and aligned dimensions with perimeter math
        # X delta is exactly 0.10, Y delta is exactly 0.20
        l=0.20
        x_min= -0.10
        x_max = x_min + l  # 10 cm forward from the center  
        y_min= 0.20
        y_max = y_min + l  # 20 cm wide rectangle centered on the Y-axis 
        z = 0.1  # Raised slightly to safely clear the table
        
        roll, pitch, yaw = 0.0, 0.0, 0.0  # Fixed orientation facing downwards 

        perimeter = 0.60
        d = fraction * perimeter  

        if d < l/2: 
            # Edge 1: Bottom (Moving forward in X)
            x = x_min + (d / (l/2)) * (x_max - x_min) #divide by l/2 because this edge is travelled from half the length of the rectangle
            y = y_min
        elif d < l/2 + l: 
            # Edge 2: Right (Moving left in Y)
            x = x_max
            y = y_min + ((d - l/2) / l) * (y_max - y_min)
        elif d < l/2 + l + l/2: 
            # Edge 3: Top (Moving backward in X)
            x = x_max - ((d - l/2 - l) / (l/2)) * (x_max - x_min) #divide by l/2 because the other half of this edge is already drawn in the first edge
            y = y_max
        else: 
            # Edge 4: Left (Moving right in Y)
            x = x_min
            y = y_max - ((d - (l/2 + l + l/2)) / l) * (y_max - y_min)

        return x, y, z, roll, pitch, yaw
    
    def _init_trace_marker(self):
        """Sets up the visual properties of the trace line."""
        self._trace_marker = Marker()
        self._trace_marker.header.frame_id = "world"
        self._trace_marker.ns = "rectangle_trace"
        self._trace_marker.id = 0
        self._trace_marker.type = Marker.LINE_STRIP
        self._trace_marker.action = Marker.ADD
        
        # Line width
        self._trace_marker.scale.x = 0.005 
        
        # Bright neon green color (R, G, B, Alpha)
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
        
        # FIX 3: Added optimize_orientation=False for the 5-DOF arm
        # Bumped max_iters slightly to ensure the first frame solves cleanly
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
    solution_trajrectangle_traj = RectangleTraj()
    tudelft_sil= SilhouetteTraj()
    solution_traj = solution_trajrectangle_traj

    
    try:
        #rclpy.spin(rectangle_traj)
        rclpy.spin(solution_traj)
    except KeyboardInterrupt:
        pass
    finally:
        solution_traj.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
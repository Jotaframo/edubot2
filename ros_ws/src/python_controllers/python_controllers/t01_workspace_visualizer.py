import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
try:
    from python_controllers.t01_Forward_Kinematics_FINAL import forward_kinematics_full
except ModuleNotFoundError:
    from t01_Forward_Kinematics_FINAL import forward_kinematics_full

# ─── Joint limits ────────────────────────────────────────────────────────────
LIMITS_UNCONSTRAINED = {
    'q1': (-3.14, 3.14),
    'q2': (-3.14, 3.14),
    'q3': (-3.14, 3.14),
    'q4': (-3.14, 3.14),
    'q5': (-3.14, 3.14),
}

LIMITS_CONSTRAINED = {
    'q1': (-2.0,    2.0   ),
    'q2': (-1.57,   1.57  ),
    'q3': (-1.58,   1.58  ),
    'q4': (-1.57,   1.57  ),
    'q5': (-1.58,   1.58  ),  # Wrist roll doesn't affect workspace boundary
}

STEP = 0.05  # radians between samples in joint space
MIN_SAMPLES = 8000
MAX_SAMPLES = 30000

#Boundary extraction 
N_AZ = 240   # azimuth bins  (0..2pi)
N_EL = 120   # elevation bins(-pi/2..pi/2)
ORIGIN = np.array([0.0, 0.0, 0.0]) # world center of the robot

def get_point_cloud(limits, step):
    """
    Computes FK position cloud.

    Parameters:
    - limits: dict of joint limits, e.g. {'q1': (-2,2), 'q2': (-1.57,1.57), ...}
    - step: radians between samples in joint space
    Returns: 
    - (N,3) array of XYZ points in the workspace
    
    """


    bins = [max(1, int((high - low) / step)) for (low, high) in limits.values()]
    est_grid_points = int(np.prod(bins, dtype=np.int64))
    num_samples = int(np.clip(est_grid_points // 2000, MIN_SAMPLES, MAX_SAMPLES))

    points = np.zeros((num_samples, 3), dtype=np.float64)
    for i in range(num_samples):
        q1 = np.random.uniform(*limits['q1'])
        q2 = np.random.uniform(*limits['q2'])
        q3 = np.random.uniform(*limits['q3'])
        q4 = np.random.uniform(*limits['q4'])
        q5 = np.random.uniform(*limits['q5'])
        tf = forward_kinematics_full(q1, q2, q3, q4, q5)
        points[i] = tf[:3, 3]

    return points

def boundary_by_spherical_binning(points_xyz: np.ndarray,
                                  n_az: int,
                                  n_el: int,
                                  origin: np.ndarray) -> np.ndarray:
    """
    Divide the sphere (directions from origin) into bins and keep only the farthest
    point in each (azimuth,elevation) bin.

    Parameters:
    - points_xyz: (N,3) array of points in 3D space
    - n_az: number of azimuth bins (0..2pi)
    - n_el: number of elevation bins (-pi/2..pi/2)
    - origin: the point from which to compute directions
    Returns:
    - (M,3) array of points that are the farthest in their spherical range
    """


    if points_xyz.size == 0:
        return points_xyz

    p = points_xyz - origin[None, :]
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.sqrt(x*x + y*y + z*z)

    eps = 1e-12
    r_safe = np.maximum(r, eps)

    phi = np.arctan2(y, x)  # azimuth: [-pi, pi]
    theta = np.arcsin(np.clip(z / r_safe, -1.0, 1.0))  # elevation: [-pi/2, pi/2]

    az = ((phi + np.pi) / (2*np.pi) * n_az).astype(np.int32)
    el = ((theta + np.pi/2) / (np.pi) * n_el).astype(np.int32)

    az = np.clip(az, 0, n_az - 1)
    el = np.clip(el, 0, n_el - 1)

    bin_id = el * n_az + az
    nbins = n_az * n_el

    best_r = -np.ones(nbins, dtype=np.float64)
    best_i = -np.ones(nbins, dtype=np.int32)

    for i in range(points_xyz.shape[0]):
        b = int(bin_id[i])
        if r[i] > best_r[b]:
            best_r[b] = r[i]
            best_i[b] = i

    sel = best_i[best_i >= 0]
    return points_xyz[sel]

#ROS 2 Visualization

def create_marker(xyz, marker_id, color, stamp, ns, size):
    """
    Creates a ROS Marker message for a point cloud.

    Parameters:
    - xyz: (N,3) array of XYZ points
    - marker_id: int, unique ID for the marker
    - color: (r,g,b,a) tuple with values in [0,1]
    - stamp: ROS time for the marker header
    - ns: string, namespace for the marker
    - size: float, scale for the marker points

    Returns:
    - Marker message with the given points and properties
    """
    m = Marker()
    m.header.frame_id, m.header.stamp = "world", stamp
    m.ns, m.id, m.type, m.action = ns, marker_id, Marker.POINTS, Marker.ADD
    m.scale.x = m.scale.y = size
    m.color.r, m.color.g, m.color.b, m.color.a = color
    for p_val in xyz:
        p = Point()
        p.x, p.y, p.z = float(p_val[0]), float(p_val[1]), float(p_val[2])
        m.points.append(p)
    return m

# ROS2 Node definition for workspace visualization
class WorkspaceVisualizer(Node):
    def __init__(self):
        super().__init__('workspace_visualizer')
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub = self.create_publisher(MarkerArray, 'workspace_points', qos)
        self.pub_rviz = self.create_publisher(MarkerArray, '/visualization_marker_array', qos)

        self.get_logger().info("Computing clouds (full)...")
        pts_full = get_point_cloud(LIMITS_UNCONSTRAINED, STEP)
        self.get_logger().info(f"Full cloud computed: {len(pts_full)} points. Extracting boundary...")
        self.pts_full = boundary_by_spherical_binning(pts_full, N_AZ, N_EL, ORIGIN)
        self.get_logger().info(f"Full boundary: {len(self.pts_full)} points.")

        self.get_logger().info("Computing clouds (constrained)...")
        pts_lim = get_point_cloud(LIMITS_CONSTRAINED, STEP)
        self.get_logger().info(f"Constrained cloud computed: {len(pts_lim)} points. Extracting boundary...")
        self.pts_lim = boundary_by_spherical_binning(pts_lim, N_AZ, N_EL, ORIGIN)
        self.get_logger().info(f"Constrained boundary: {len(self.pts_lim)} points.")

        self.timer = self.create_timer(1.0, self.publish)
        self.get_logger().info("Publishing markers on '/workspace_points' and '/visualization_marker_array' (frame='world').")
        self.get_logger().info("Done.")

    def publish(self):
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()
        # Red = Full boundary, Green = Constrained boundary
        ma.markers.append(create_marker(self.pts_full, 0, (1.0, 0.2, 0.2, 0.35), now, "full_boundary", 0.006))
        ma.markers.append(create_marker(self.pts_lim,  1, (0.2, 1.0, 0.4, 0.90), now, "lim_boundary",  0.007))
        self.pub.publish(ma)
        self.pub_rviz.publish(ma)

def main():
    rclpy.init()
    node = WorkspaceVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

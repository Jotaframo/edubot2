import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from Forward_Kinematics_FINAL import forward_kinematics_full

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

# ─── Boundary extraction (spherical bins) ─────────────────────────────────────
N_AZ = 240   # azimuth bins  (0..2pi)
N_EL = 120   # elevation bins(-pi/2..pi/2)
ORIGIN = np.array([0.0, 0.0, 0.0])  # bin directions around this point (world frame)

# ─── Workspace sampling ──────────────────────────────────────────────────────

def get_point_cloud(limits, step):
    """Generates the meshgrid and computes all FK positions."""
    q1 = np.arange(*limits['q1'], step)
    q2 = np.arange(*limits['q2'], step)
    q3 = np.arange(*limits['q3'], step)
    q4 = np.arange(*limits['q4'], step)

    Q1, Q2, Q3, Q4 = np.meshgrid(q1, q2, q3, q4, indexing='ij')
    return forward_kinematics_full(Q1.ravel(), Q2.ravel(), Q3.ravel(), Q4.ravel(), batch_mode=True)

def boundary_by_spherical_binning(points_xyz: np.ndarray,
                                  n_az: int,
                                  n_el: int,
                                  origin: np.ndarray) -> np.ndarray:
    """
    Divide the sphere (directions from origin) into bins and keep only the farthest
    point in each (azimuth,elevation) bin.
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

# ─── ROS 2 Visualization Logic ───────────────────────────────────────────────

def create_marker(xyz, marker_id, color, stamp, ns, size):
    """Creates a ROS Marker message for a point cloud."""
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

class WorkspaceVisualizer(Node):
    def __init__(self):
        super().__init__('workspace_visualizer')
        self.pub = self.create_publisher(MarkerArray, 'workspace_points', 10)

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
        self.get_logger().info("Done.")

    def publish(self):
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()
        # Red = Full boundary, Green = Constrained boundary
        ma.markers.append(create_marker(self.pts_full, 0, (1.0, 0.2, 0.2, 0.35), now, "full_boundary", 0.006))
        ma.markers.append(create_marker(self.pts_lim,  1, (0.2, 1.0, 0.4, 0.90), now, "lim_boundary",  0.007))
        self.pub.publish(ma)

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

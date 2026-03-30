#!/usr/bin/env python3
"""
Records joint states during a trajectory and plots the end-effector path.

Usage:
  Record new trajectory (in parallel with controller):
    python3 ee_trajectory_plotter.py --record
  
  Plot saved trajectory (no simulation needed):
    python3 ee_trajectory_plotter.py --load

  Record and auto-plot on Ctrl+C:
    python3 ee_trajectory_plotter.py
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys

try:
    from python_controllers.Forward_Kinematics_FINAL import forward_kinematics_full
except ModuleNotFoundError:
    from Forward_Kinematics_FINAL import forward_kinematics_full


JOINT_NAMES = [
    "Shoulder_Rotation",
    "Shoulder_Pitch",
    "Elbow",
    "Wrist_Pitch",
]


class EETrajectoryRecorder(Node):
    def __init__(self):
        super().__init__("ee_trajectory_recorder")
        
        self.ee_positions = []
        self.joint_states = []
        self.times = []
        self.start_time = None
        
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.state_sub = self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, qos
        )
        
        self.get_logger().info("Recording end-effector trajectory... press Ctrl+C to stop and plot")
    
    def _on_joint_state(self, msg: JointState):
        """Record joint state and compute EE position."""
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        
        # Check if all required joints are present
        if any(name not in name_to_idx for name in JOINT_NAMES):
            return
        
        # Extract joint angles in order
        q = np.zeros(4, dtype=float)
        for i, name in enumerate(JOINT_NAMES):
            q[i] = msg.position[name_to_idx[name]]
        
        # Get wrist roll (q5), default to 0
        q5 = msg.position[name_to_idx["Wrist_Roll"]] if "Wrist_Roll" in name_to_idx else 0.0
        
        # Compute forward kinematics
        try:
            T = forward_kinematics_full(q[0], q[1], q[2], q[3], q5)
            p_ee = T[:3, 3]
            self.ee_positions.append(p_ee)
            self.joint_states.append(q.copy())
            
            if self.start_time is None:
                self.start_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            
            elapsed = (msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9) - self.start_time
            self.times.append(elapsed)
            
            if len(self.ee_positions) % 20 == 0:
                self.get_logger().info(
                    f"Recorded {len(self.ee_positions)} points | "
                    f"EE position: [{p_ee[0]:.4f}, {p_ee[1]:.4f}, {p_ee[2]:.4f}]"
                )
        except Exception as e:
            self.get_logger().warn(f"FK computation failed: {e}")


def plot_trajectory(ee_positions, times):
    """Generate and display trajectory plots."""
    if len(ee_positions) == 0:
        print("No trajectory data to plot!")
        return
    
    ee_traj = np.array(ee_positions)
    times = np.array(times)
    
    # 3D trajectory plot
    fig = plt.figure(figsize=(15, 5))
    
    # 3D path
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2], 'b-', linewidth=2)
    ax1.scatter(ee_traj[0, 0], ee_traj[0, 1], ee_traj[0, 2], 
               c='g', s=100, label='Start', marker='o')
    ax1.scatter(ee_traj[-1, 0], ee_traj[-1, 1], ee_traj[-1, 2], 
               c='r', s=100, label='End', marker='s')
    
    # Set equal aspect ratio for all axes
    all_data = ee_traj.flatten()
    max_range = (np.max(ee_traj) - np.min(ee_traj)) / 2.0
    mid_x = (np.max(ee_traj[:, 0]) + np.min(ee_traj[:, 0])) * 0.5
    mid_y = (np.max(ee_traj[:, 1]) + np.min(ee_traj[:, 1])) * 0.5
    mid_z = (np.max(ee_traj[:, 2]) + np.min(ee_traj[:, 2])) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D End-Effector Trajectory')
    ax1.legend()
    ax1.grid()
    
    # X, Y, Z components over time
    ax2 = fig.add_subplot(132)
    ax2.plot(times, ee_traj[:, 0], 'r-', label='X', linewidth=2)
    ax2.plot(times, ee_traj[:, 1], 'g-', label='Y', linewidth=2)
    ax2.plot(times, ee_traj[:, 2], 'b-', label='Z', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('EE Position Components vs Time')
    ax2.legend()
    ax2.grid()
    
    # XY plane view (top-down)
    ax3 = fig.add_subplot(133)
    ax3.plot(ee_traj[:, 0], ee_traj[:, 1], 'b-', linewidth=2)
    ax3.scatter(ee_traj[0, 0], ee_traj[0, 1], c='g', s=100, label='Start', marker='o')
    ax3.scatter(ee_traj[-1, 0], ee_traj[-1, 1], c='r', s=100, label='End', marker='s')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Top-Down View (XY Plane)')
    ax3.legend()
    ax3.grid()
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("TRAJECTORY STATISTICS")
    print("="*60)
    print(f"Total points recorded: {len(ee_positions)}")
    print(f"Duration: {times[-1]:.2f} s")
    print(f"\nStart position: {ee_traj[0]}")
    print(f"End position:   {ee_traj[-1]}")
    print(f"Displacement:   {ee_traj[-1] - ee_traj[0]}")
    print(f"Total distance: {np.sum(np.linalg.norm(np.diff(ee_traj, axis=0), axis=1)):.4f} m")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Record or plot end-effector trajectories.')
    parser.add_argument('--load', action='store_true', help='Load and plot saved trajectory (no ROS needed)')
    parser.add_argument('--record', action='store_true', help='Record new trajectory from ROS (default)')
    args = parser.parse_args()
    
    # Default to record if no args
    if not args.load and not args.record:
        args.record = True
    
    if args.load:
        # Load and plot existing trajectory
        try:
            data = np.load('/tmp/ee_trajectory.npz')
            ee_positions = data['ee_positions']
            times = data['times']
            print(f"Loaded trajectory with {len(ee_positions)} points")
            plot_trajectory(ee_positions, times)
        except FileNotFoundError:
            print("Error: No saved trajectory found at /tmp/ee_trajectory.npz")
            print("Record a trajectory first with: python3 ee_trajectory_plotter.py")
        return
    
    # Record mode
    rclpy.init()
    recorder = EETrajectoryRecorder()
    
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        recorder.get_logger().info("\nRecording stopped. Saving and plotting trajectory...")
        recorder.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass
        
        # Save trajectory to file
        if len(recorder.ee_positions) > 0:
            np.savez('/tmp/ee_trajectory.npz',
                     ee_positions=np.array(recorder.ee_positions),
                     times=np.array(recorder.times))
            print("✓ Trajectory saved to /tmp/ee_trajectory.npz")
            print("  Next time, plot without simulation: python3 ee_trajectory_plotter.py --load")
        
        # Plot trajectory
        plot_trajectory(recorder.ee_positions, recorder.times)


if __name__ == "__main__":
    main()

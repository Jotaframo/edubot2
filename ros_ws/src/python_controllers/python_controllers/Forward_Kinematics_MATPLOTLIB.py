import numpy as np
import matplotlib.pyplot as plt

def create_tf_matrix(tx, ty, tz, roll, pitch, yaw):
    Rx = np.array([[1, 0, 0, 0], [0, np.cos(roll), -np.sin(roll), 0], [0, np.sin(roll), np.cos(roll), 0], [0, 0, 0, 1]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch), 0], [0, 1, 0, 0], [-np.sin(pitch), 0, np.cos(pitch), 0], [0, 0, 0, 1]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0], [np.sin(yaw), np.cos(yaw), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    return T @ Rz @ Ry @ Rx

def get_joint_rotation(q):
    return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def forward_kinematics(q1, q2, q3, q4, q5=0.0):
    T_world_base     = create_tf_matrix(0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  3.1416)
    T_base_shoulder  = create_tf_matrix(0.0000, -0.0452,  0.0165,  0.0000,  0.0000,  0.0000)
    T_shoulder_upper = create_tf_matrix(0.0000, -0.0306,  0.1025,  0.0000, -1.5708,  0.0000)
    T_upper_lower    = create_tf_matrix(0.1126, -0.0280,  0.0000,  0.0000,  0.0000,  0.0000)
    T_lower_wrist    = create_tf_matrix(0.0052, -0.1349,  0.0000,  0.0000,  0.0000,  1.5708)
    T_wrist_gripper  = create_tf_matrix(-0.0601, 0.0000,  0.0000,  0.0000, -1.5708,  0.0000)
    T_gripper_center = create_tf_matrix(0.0000,  0.0000,  0.0750,  0.0000,  0.0000,  0.0000)

    A1 = T_base_shoulder  @ get_joint_rotation(q1)
    A2 = T_shoulder_upper @ get_joint_rotation(q2)
    A3 = T_upper_lower    @ get_joint_rotation(q3)
    A4 = T_lower_wrist    @ get_joint_rotation(q4)
    A5 = T_wrist_gripper  @ get_joint_rotation(q5)
    
    T_final = T_world_base @ A1 @ A2 @ A3 @ A4 @ A5 @ T_gripper_center
    return T_final[0:3, 3]

# --- Workspace Generation ---

LIMITS_CONSTRAINED = {
    'q1': (-2.0,  2.0),
    'q2': (-1.57, 1.57),
    'q3': (-1.58, 1.58),
    'q4': (-1.57, 1.57),
    'q5': (-1.58, 1.58) 
}

num_samples = 20000  # Increased slightly for better density definition
points = np.zeros((num_samples, 3))

print(f"Generating {num_samples} constrained points. This may take a few seconds...")

for i in range(num_samples):
    q1 = np.random.uniform(*LIMITS_CONSTRAINED['q1'])
    q2 = np.random.uniform(*LIMITS_CONSTRAINED['q2'])
    q3 = np.random.uniform(*LIMITS_CONSTRAINED['q3'])
    q4 = np.random.uniform(*LIMITS_CONSTRAINED['q4'])
    
    pos = forward_kinematics(q1, q2, q3, q4)
    points[i] = pos

# --- Plotting ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(points[:,0], points[:,1], points[:,2], c=points[:,2], cmap='plasma', s=1, alpha=0.4)

ax.set_title("SO-ARM101 Constrained Workspace Point Cloud")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_zlabel("Z (meters)")

max_range = np.array([points[:,0].max()-points[:,0].min(), 
                      points[:,1].max()-points[:,1].min(), 
                      points[:,2].max()-points[:,2].min()]).max() / 2.0
mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
mid_z = (points[:,2].max()+points[:,2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Z Height (meters)')

ax.scatter([0], [0], [0], color='red', s=50, label='Base Origin')
ax.legend()

plt.show()
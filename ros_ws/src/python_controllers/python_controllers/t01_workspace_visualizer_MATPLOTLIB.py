import numpy as np
import matplotlib.pyplot as plt
try:
    from python_controllers.t01_Forward_Kinematics_FINAL import forward_kinematics_full
except ModuleNotFoundError:
    from t01_Forward_Kinematics_FINAL import forward_kinematics_full

LIMITS_CONSTRAINED = {
    'q1': (-2.0,  2.0),
    'q2': (-1.57, 1.57),
    'q3': (-1.58, 1.58),
    'q4': (-1.57, 1.57),
    'q5': (-1.58, 1.58) 
}

num_samples = 20000 
points = np.zeros((num_samples, 3))

print(f"Generating {num_samples} constrained points...")

for i in range(num_samples):
    q1 = np.random.uniform(*LIMITS_CONSTRAINED['q1'])
    q2 = np.random.uniform(*LIMITS_CONSTRAINED['q2'])
    q3 = np.random.uniform(*LIMITS_CONSTRAINED['q3'])
    q4 = np.random.uniform(*LIMITS_CONSTRAINED['q4'])
    q5 = np.random.uniform(*LIMITS_CONSTRAINED['q5'])
    
    tf = forward_kinematics_full(q1, q2, q3, q4, q5)
    points[i] = tf[:3, 3]
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
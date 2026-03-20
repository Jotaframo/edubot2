import numpy as np

def create_tf_matrix(tx, ty, tz, roll, pitch, yaw):
    Rx = np.array([[1, 0, 0, 0], [0, np.cos(roll), -np.sin(roll), 0], [0, np.sin(roll), np.cos(roll), 0], [0, 0, 0, 1]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch), 0], [0, 1, 0, 0], [-np.sin(pitch), 0, np.cos(pitch), 0], [0, 0, 0, 1]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0], [np.sin(yaw), np.cos(yaw), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    return T @ Rz @ Ry @ Rx

def get_joint_rotation(q):
    return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def forward_kinematics_full(q1, q2, q3, q4, q5):
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
    
    return T_world_base @ A1 @ A2 @ A3 @ A4 @ A5 @ T_gripper_center
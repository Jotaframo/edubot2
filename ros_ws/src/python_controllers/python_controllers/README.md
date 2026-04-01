# Python Controllers

Build and source before running any task:

```bash
cd /edubot2/ros_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select python_controllers
source install/setup.bash
```

## Task 1

`t01_Forward_Kinematics_FINAL.py` Implements the final forward-kinematics model of the robot and returns the full homogeneous transform from joint angles

`t01_workspace_visualizer.py` Samples joint configurations, maps them through FK, and visualizes the reachable workspace in 3D

```bash
ros2 run python_controllers workspace_visualizer
```

`t01_workspace_visualizer_MATPLOTLIB.py` Generates the same workspace-style visualization using Matplotlib instead of the ROS plotting workflow

## Task 2

`t02_Inverse_Kinematics_Numerical.py` Solves inverse kinematics numerically with coordinate descent, multistart search, and pose-error evaluation for assignment targets

`t02_Position_Trajectory_Final.py` Follows the assignment Cartesian trajectory by evaluating target positions over time and converting them to joint commands through IK

```bash
ros2 run python_controllers position_trajectory_follower
```

`t02_joint_pose_commander.py` Publishes a fixed table of assignment joint poses so each required pose can be checked directly on the robot or in simulation

```bash
ros2 run python_controllers joint_pose_commander
```

## Task 3

`t03_Jacobian_Symbolic.py` Derives the Jacobian symbolically using SymPy. Reconstructs the FK chain and computes each column as z_i × (p_ee − o_i). Also prints SVD/rank at all assignment poses.

`t03_Jacobian_FINAL.py` Computes the 3×4 linear Jacobian numerically via central finite differences (ε = 1e-6). Used for online control. Also prints SVD/rank at all assignment poses.

`t03_constant_velocity_upward.py` Runs a Jacobian-based velocity controller that commands a constant upward end-effector velocity while checking singularities and joint-limit safety

```bash
ros2 run python_controllers constant_velocity_upward
```

## Task 4

`t04_pick_place_oneway.py` Builds a one-way pick-and-place stage sequence, precomputes IK targets for each stage, and executes them with smooth joint-space interpolation

```bash
ros2 run python_controllers pick_place_oneway
```

`t04_pick_place_roundtrip.py` Extends the one-way controller by building a forward pick-and-place leg, returning home, and then planning the reverse leg back again

```bash
ros2 run python_controllers pick_place_roundtrip
```

## Task 5

`t05_block_stacking.py` Extends the pick-and-place controller to repeat the sequence over multiple cycles while increasing the placement height to stack blocks

```bash
ros2 run python_controllers block_stacking
```

## Examples

`example_pos_traj.py` Publishes a simple example joint-position trajectory to demonstrate the position-control command interface

```bash
ros2 run python_controllers example_pos_traj
```

`example_vel_traj.py` Publishes a simple example joint-velocity trajectory to demonstrate the velocity-control command interface

```bash
ros2 run python_controllers example_vel_traj
```

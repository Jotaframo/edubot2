# Python Controllers

Build and source before running any task:

```bash
cd /Users/grdc/edubot2/ros_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select python_controllers
source install/setup.bash
```

## Task 1

`t01_workspace_visualizer.py` visualizes the reachable workspace of the robot

```bash
ros2 run python_controllers workspace_visualizer
```

## Task 2

`t02_Position_Trajectory_Final.py` follows the assignment position trajectory in Cartesian space using IK

```bash
ros2 run python_controllers position_trajectory_follower
```

`t02_joint_pose_commander.py` sends the assignment joint poses one by one for quick pose checking

```bash
ros2 run python_controllers joint_pose_commander
```

## Task 3

`t03_constant_velocity_upward.py` drives the end effector upward with a Jacobian-based velocity controller

```bash
ros2 run python_controllers constant_velocity_upward
```

## Task 4

`t04_pick_place_oneway.py` performs a one-way pick-and-place sequence from the calibrated pick side to the place side

```bash
ros2 run python_controllers pick_place_oneway
```

`t04_pick_place_roundtrip.py` performs the pick-and-place sequence forward and then back again

```bash
ros2 run python_controllers pick_place_roundtrip
```

## Task 5

`t05_block_stacking.py` repeats the pick-and-place motion while increasing the place height to stack blocks

```bash
ros2 run python_controllers block_stacking
```

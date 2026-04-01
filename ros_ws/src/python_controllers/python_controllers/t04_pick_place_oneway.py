from dataclasses import dataclass
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    from python_controllers.t02_Inverse_Kinematics_Numerical import ik_coordinate_descent_multi_start
    from python_controllers.t01_Forward_Kinematics_FINAL import forward_kinematics_full
except ModuleNotFoundError:
    from t02_Inverse_Kinematics_Numerical import ik_coordinate_descent_multi_start
    from t01_Forward_Kinematics_FINAL import forward_kinematics_full

@dataclass
class Stage:
    name: str
    xyz: tuple[float, float, float] | None = None
    rpy: tuple[float, float, float] | None = None
    gripper: float | None = None
    move_time: float = None
    q_target: np.ndarray | None = None
    optimize_orientation: bool = True

class PhysicalPickPlaceTuning:
    start_q: tuple[float, float, float, float, float] = (0.9, 0.6, -1.195, -1.1198, 1.615)
    joint1_place_delta_rad: float = -1.5
    gripper_open: float = 0.5
    gripper_closed: float = 0.12
    pick_z_m: float = 0.02
    hover_z_m: float = 0.04
    travel_z_m: float = 0.05
    initial_approach_move_time_s: float = 3.0
    descend_move_time_s: float = 2.0
    grip_move_time_s: float = 2.2
    lift_move_time_s: float = 2.0
    transfer_move_time_s: float = 4.0


PHYSICAL_TUNING = PhysicalPickPlaceTuning()

def fk_xyz(q):
    """
    Compute the end-effector position for a joint configuration.

    Parameters:
    - q: joint vector [q1, q2, q3, q4, q5] in radians
    Returns:
    - (3,) array [x, y, z] of end-effector position in world coordinates
    """
    T = forward_kinematics_full(q[0], q[1], q[2], q[3], q[4])
    return np.asarray(T[:3, 3], dtype=float)


def fk_rpy(q) -> tuple[float, float, float]:
    """
    Compute roll, pitch, and yaw for a joint configuration.

    Parameters:
    - q: joint vector [q1, q2, q3, q4, q5] in radians
    Returns:
    - tuple (roll, pitch, yaw) in radians from the FK rotation matrix
    """
    T = forward_kinematics_full(q[0], q[1], q[2], q[3], q[4])
    R = T[:3, :3]

    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = float(np.arcsin(sy))
    cp = float(np.cos(pitch))

    if abs(cp) > 1e-8:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))

    return roll, pitch, yaw

class PickPlaceOneWay(Node):
    def __init__(self):
        """
        Initialize the one-way pick-and-place controller.

        Parameters:
        - none
        Returns:
        - none
        """
        super().__init__("pick_place_world_locked")

        self.tuning = PHYSICAL_TUNING
        
        self.gripper_open = self.tuning.gripper_open
        self.gripper_closed = self.tuning.gripper_closed
        
        self.pick_z_m = self.tuning.pick_z_m
        self.hover_z_m = self.tuning.hover_z_m
        self.travel_z_m = self.tuning.travel_z_m

        self.initial_approach_move_time_s = self.tuning.initial_approach_move_time_s
        self.descend_move_time_s = self.tuning.descend_move_time_s
        self.grip_move_time_s = self.tuning.grip_move_time_s
        self.lift_move_time_s = self.tuning.lift_move_time_s
        self.transfer_move_time_s = self.tuning.transfer_move_time_s

        self.start_q = np.array(self.tuning.start_q, dtype=float)
        # The orientation of the end-effector is locked to the FK orientation of the start_q at the start of the motion
        self.locked_rpy = fk_rpy(self.start_q)
        # Location A which is pick location is determined from the FK cartesian position of the start_q
        self.location_a = fk_xyz(self.start_q)

        #Create a publisher to publish joint trajectory commands to the robot, and a subscriber to listen to the current joint states. 
        self.pub = self.create_publisher(JointTrajectory, "/joint_cmds", 10)
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, qos_profile_sensor_data
        )

        # Initialize state variables for the control loop
        self.current_q = None
        self.current_gripper = self.gripper_open
        self.done = False
        self.motion_ready = False
        self.stages = []
        self.stage_idx = 0
        self.stage_active = False
        self.timer = self.create_timer(0.05, self._tick)

        # Log the initial configuration and tuning parameters
        self.get_logger().info(f"start_q: {np.round(self.start_q, 4)}")
        self.get_logger().info(f"location_a: {self.location_a}")
        self.get_logger().info(f"pick_rpy: {np.round(self.locked_rpy, 4)}")
        self.get_logger().info("Waiting for /joint_states before precomputing stages.")

    def _on_joint_state(self, msg: JointState):
        """
        Initialize motion planning from the first received joint state.

        Parameters:
        - msg: JointState message containing current joint positions
        Returns:
        - none
        """
        # Only start if we haven't already initialized and if the message contains enough joint positions
        if self.motion_ready:
            return
        if len(msg.position) < 5:
            return
        # Set the first received joint state as the current state
        self.current_q = np.array(msg.position[:5], dtype=float)

        # If the message contains a 6th position for the gripper, use it; otherwise, default to gripper_open
        if len(msg.position) >= 6:
            self.current_gripper = float(msg.position[5])

        # This calls an optional hook for subclasses to perform additional initialization after receiving the joint state 
        # which is needed for roundtrip and block stacking variants
        self._post_joint_state_init()
        # Precompute the stage sequence for the pick-and-place task based on the initial joint state
        self.stages = self._build_stage_sequence()
        self.stage_idx = 0
        self.stage_active = False
        self.done = False
        self.motion_ready = True
        self.get_logger().info(f"Initialized from /joint_states with q={np.round(self.current_q, 4)}")

    def _post_joint_state_init(self):
        """
        Hook for subclasses after the initial joint state is received.

        Parameters:
        - none
        Returns:
        - none
        """
        pass

    def _solve_ik(self, target_xyz, target_rpy, q_init=None, optimize_orientation=True, n_random=20):
        """
        Solve inverse kinematics for a target Cartesian pose.

        Parameters:
        - target_xyz: target position tuple (x, y, z)
        - target_rpy: target orientation tuple (roll, pitch, yaw)
        - q_init: optional initial joint seed
        - optimize_orientation: whether orientation error is included in optimization
        - n_random: number of random restarts for the numerical IK
        Returns:
        - array of joint values that best matches the target pose
        """
        x, y, z = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])
        q = q_init if q_init is not None else self.current_q

        results = ik_coordinate_descent_multi_start(
            x, y, z,
            target_rpy[0], target_rpy[1], target_rpy[2],
            q_prev=q,
            num_random=n_random,
            optimize_orientation=optimize_orientation,
            pos_tol=5e-3,
            rot_tol=np.deg2rad(1.0),
        )

        best = results[0]
        achieved = fk_xyz(np.array(best["q_raw"]))
        err = float(np.linalg.norm(achieved - np.array(target_xyz)))

        # Log a warning if the best IK solution has a large positional error, which may indicate that 
        # the target pose is unreachable or that the movement will be less accurate for that stage. 
        if err > 0.01:
            self.get_logger().warn(
                f"IK best residual {err:.4f}m for target {np.round(target_xyz, 4)} — may be unreachable"
            )
        return np.array(best["q_raw"])
    
    def _build_stage_sequence(self):
        """
        Build the stage list for the one-way pick-and-place sequence.

        Parameters:
        - none
        Returns:
        - list of Stage objects with precomputed joint targets
        """
        # Define key Cartsian positions for pick 
        # (x,y) are from the FK of the start_q, and z values are defined by tuning parameters.
        a_hov = (self.location_a[0], self.location_a[1], self.hover_z_m)
        a_pk  = (self.location_a[0], self.location_a[1], self.pick_z_m)
        a_tr  = (self.location_a[0], self.location_a[1], self.travel_z_m)
        stages = []

        # The initial approach stage moves from the current joint state to a hover pose above the pick location,
        #  with the end-effector orientation locked to the FK orientation of the start_q. 
        initial_approach = Stage(
            "initial_approach",
            xyz=a_hov,
            rpy=self.locked_rpy,
            gripper=self.gripper_open,
            move_time=self.initial_approach_move_time_s,
            optimize_orientation=True,
        )
        initial_approach.q_target = self._solve_ik(
            initial_approach.xyz,
            initial_approach.rpy,
            q_init=self.current_q,
            optimize_orientation=initial_approach.optimize_orientation,
        )
        stages.append(initial_approach)

        q = initial_approach.q_target.copy()

        for stage in [
            Stage("descend_a", xyz=a_pk, rpy=self.locked_rpy, gripper=self.gripper_open, move_time=self.descend_move_time_s, optimize_orientation=True),
            Stage("grasp_a", xyz=a_pk, rpy=self.locked_rpy, gripper=self.gripper_closed, move_time=self.grip_move_time_s, optimize_orientation=True),
            Stage("lift_a", xyz=a_tr, rpy=self.locked_rpy, gripper=self.gripper_closed, move_time=self.lift_move_time_s, optimize_orientation=True),
        ]:
            # For the grasp stage, we keep the same joint angles as the descend stage since we want to close the gripper at the same pose.
            # For the other stages, we solve IK to find the joint targets that achieve the desired Cartesian pose and orientation.
            if stage.name == "grasp_a":
                stage.q_target = q.copy()
            else:
                stage.q_target = self._solve_ik(
                    stage.xyz,
                    stage.rpy,
                    q_init=q,
                    optimize_orientation=stage.optimize_orientation,
                )
            stages.append(stage)
            q = stage.q_target

        # We store the computed joint angles from the target lift pose above the pick location 
        lift_a_q = stages[-1].q_target.copy()
        
        # The place location B pose is defined by taking the lift pose and applying a delta in the shoulder joint
        self.place_q_ref = lift_a_q.copy()
        self.place_q_ref[0] += float(self.tuning.joint1_place_delta_rad)

        # After computing this pose the Cartesian position and orientation of the place location B are determined by the 
        # FK of this place_q_ref, which is used for the remaining stages.
        self.location_b = fk_xyz(self.place_q_ref)
        self.place_rpy = fk_rpy(self.place_q_ref)


        # Define key Cartsian positions for pick 
        # (x,y) are from the FK of the start_q, and z values are defined by tuning parameters.
        b_pk = (self.location_b[0], self.location_b[1], self.pick_z_m)
        b_tr = (self.location_b[0], self.location_b[1], self.travel_z_m)

        for stage in [
            Stage("travel_to_b", xyz=tuple(self.location_b.tolist()), rpy=self.place_rpy, gripper=self.gripper_closed, move_time=self.transfer_move_time_s, q_target=self.place_q_ref.copy(), optimize_orientation=True),
            Stage("descend_b", xyz=b_pk, rpy=self.place_rpy, gripper=self.gripper_closed, move_time=self.descend_move_time_s, optimize_orientation=True),
            Stage("release_b", xyz=b_pk, rpy=self.place_rpy, gripper=self.gripper_open, move_time=self.grip_move_time_s, optimize_orientation=True),
            Stage("retreat_from_b", xyz=b_tr, rpy=self.place_rpy, gripper=self.gripper_open, move_time=self.lift_move_time_s, optimize_orientation=True),
        ]:
            # Same as before the release stage keeps the same joint angles as the descend stage just the grippert is opened
            if stage.name == "release_b":
                stage.q_target = q.copy()
            elif stage.q_target is None:
                stage.q_target = self._solve_ik(
                    stage.xyz,
                    stage.rpy,
                    q_init=q,
                    optimize_orientation=stage.optimize_orientation,
                )
            stages.append(stage)
            q = stage.q_target

        self.get_logger().info(f"location_b: {self.location_b}")
        self.get_logger().info(f"place_q_ref: {np.round(self.place_q_ref, 4)}")
        self.get_logger().info(f"place_rpy: {np.round(self.place_rpy, 4)}")

        for stage in stages:
            achieved = fk_xyz(stage.q_target)
            self.get_logger().info(
                f"[IK precompute] {stage.name} | target: {np.round(stage.xyz, 4)} | achieved: {np.round(achieved, 4)}"
            )

        return stages

def _tick(self):
    """
    Advance the active stage and publish the interpolated joint command.

    Parameters:
    - none
    Returns:
    - none
    """
    # Wait until startup state and stage plan are ready.
    if not self.motion_ready:
        return
    # `done` means the whole controller has finished permanently.
    if self.done:
        return
    # No active stage remains in the current sequence.
    if self.stage_idx >= len(self.stages):
        return

    # Select the current stage to execute.
    stage = self.stages[self.stage_idx]

    # On the first tick of a stage, store its start time and endpoints.
    if not self.stage_active:
        self.stage_start_t = self.get_clock().now()
        self.stage_start_q = self.current_q.copy()
        self.stage_target_q = stage.q_target
        self.stage_active = True

    # Compute normalized progress through the stage.
    elapsed = (self.get_clock().now() - self.stage_start_t).nanoseconds / 1e9
    alpha = np.clip(elapsed / stage.move_time, 0.0, 1.0)

    # Smooth the interpolation to avoid abrupt starts and stops.
    s_alpha = alpha**2 * (3 - 2*alpha)

    # Interpolate joint positions from stage start to stage target.
    q_cmd = self.stage_start_q + (self.stage_target_q - self.stage_start_q) * s_alpha

    # Use the stage gripper command if defined otherwise keep the current one.
    cmd_gripper = stage.gripper if stage.gripper is not None else self.current_gripper

    # Publish the commanded arm joints and gripper state.
    msg = JointTrajectory()
    pt = JointTrajectoryPoint()
    pt.positions = q_cmd.tolist() + [cmd_gripper]
    pt.time_from_start = Duration(sec=0, nanosec=50000000)
    msg.points = [pt]
    self.pub.publish(msg)

    # When the stage duration is complete, latch the final state and advance.
    if elapsed >= stage.move_time:
        self.current_q = q_cmd
        self.current_gripper = cmd_gripper
        self.stage_idx += 1
        self.stage_active = False


def main(args=None):

    rclpy.init(args=args)
    node = PickPlaceOneWay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

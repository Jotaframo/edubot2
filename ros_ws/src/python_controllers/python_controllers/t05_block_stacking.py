import numpy as np
import rclpy
try:
    from python_controllers.t04_pick_place_oneway import PickPlaceOneWay, Stage, fk_rpy, fk_xyz
except ModuleNotFoundError:
    from t04_pick_place_oneway import PickPlaceOneWay, Stage, fk_rpy, fk_xyz

class BlockStackingOpenLoop(PickPlaceOneWay):
    def __init__(self):
        """
        Initialize the block-stacking controller.

        Parameters:
        - none
        Returns:
        - none
        """
        super().__init__()
        self.declare_parameter("stack_count", 3)
        self.declare_parameter("stack_height_step", 0.025)

        self.stack_count = int(self.get_parameter("stack_count").value)
        self.stack_height_step = float(self.get_parameter("stack_height_step").value)
        
        self.stack_index = 0
        self.stage_idx = 0
        self.stage_active = False
        self.done = False

        self.get_logger().info(
            f"Block Stacking initialized: count={self.stack_count}, step={self.stack_height_step}"
        )

    def _build_stage_sequence(self):
        """
        Build the stage sequence for the current stacking cycle.

        Parameters:
        - none
        Returns:
        - list of Stage objects for the active stack level
        """
        z_offset = self.stack_index * self.stack_height_step
        # each cycle lifts the place height a bit more to build the stack
        b_pick_z = self.pick_z_m + z_offset
        a_travel_z = self.travel_z_m + z_offset
        b_travel_z = self.travel_z_m + z_offset

        a_hov = (self.location_a[0], self.location_a[1], self.hover_z_m)
        a_pk = (self.location_a[0], self.location_a[1], self.pick_z_m)
        a_tr = (self.location_a[0], self.location_a[1], a_travel_z)
        sequence = []

        if self.stack_index == 0:
            # only the first cycle needs to travel from the measured start pose into the task
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
            sequence.append(initial_approach)
            q = initial_approach.q_target.copy()
        else:
            # after the first cycle we keep going from where the arm alredy is
            q_source = self.current_q if self.current_q is not None else self.start_q
            q = np.array(q_source, dtype=float).copy()

        for stage in [
            Stage("descend_a", xyz=a_pk, rpy=self.locked_rpy, gripper=self.gripper_open, move_time=self.descend_move_time_s, optimize_orientation=True),
            Stage("grasp_a", xyz=a_pk, rpy=self.locked_rpy, gripper=self.gripper_closed, move_time=self.grip_move_time_s, optimize_orientation=True),
            Stage("lift_a", xyz=a_tr, rpy=self.locked_rpy, gripper=self.gripper_closed, move_time=self.lift_move_time_s, optimize_orientation=True),
        ]:
            if stage.name == "grasp_a":
                # grasp keeps the same arm pose and only changes the gripper state
                stage.q_target = q.copy()
            else:
                stage.q_target = self._solve_ik(
                    stage.xyz,
                    stage.rpy,
                    q_init=q,
                    optimize_orientation=stage.optimize_orientation,
                )
            sequence.append(stage)
            q = stage.q_target

        lift_a_q = sequence[-1].q_target.copy()
        place_q_ref = lift_a_q.copy()
        # like in oneway, the place side is made by rotating joint 1 from the lifted pose
        place_q_ref[0] += float(self.tuning.joint1_place_delta_rad)
        location_b = fk_xyz(place_q_ref)
        place_rpy = fk_rpy(place_q_ref)

        b_pk = (location_b[0], location_b[1], b_pick_z)
        b_tr = (location_b[0], location_b[1], b_travel_z)

        for stage in [
            Stage("travel_to_b", xyz=tuple(location_b.tolist()), rpy=place_rpy, gripper=self.gripper_closed, move_time=self.transfer_move_time_s, q_target=place_q_ref.copy(), optimize_orientation=True),
            Stage("descend_b", xyz=b_pk, rpy=place_rpy, gripper=self.gripper_closed, move_time=self.descend_move_time_s, optimize_orientation=True),
            Stage("release_b", xyz=b_pk, rpy=place_rpy, gripper=self.gripper_open, move_time=self.grip_move_time_s, optimize_orientation=True),
            Stage("retreat_from_b", xyz=b_tr, rpy=place_rpy, gripper=self.gripper_open, move_time=self.lift_move_time_s, optimize_orientation=True),
            Stage("return_to_pick_hover", xyz=a_hov, rpy=self.locked_rpy, gripper=self.gripper_open, move_time=self.transfer_move_time_s, optimize_orientation=True),
        ]:
            if stage.name == "release_b":
                # same idea as grasp_a, opening the gripper should not move the arm
                stage.q_target = q.copy()
            elif stage.q_target is None:
                stage.q_target = self._solve_ik(
                    stage.xyz,
                    stage.rpy,
                    q_init=q,
                    optimize_orientation=stage.optimize_orientation,
                )
            sequence.append(stage)
            q = stage.q_target

        self.get_logger().info(
            f"[stack {self.stack_index + 1}] location_b: {location_b} | place_q_ref: {place_q_ref.round(4)} | place_rpy: {place_rpy}"
        )
        for stage in sequence:
            achieved = fk_xyz(stage.q_target)
            self.get_logger().info(
                f"[stack {self.stack_index + 1}] [IK precompute] {stage.name} | target: {np.round(stage.xyz, 4)} | achieved: {np.round(achieved, 4)}"
        )

        return sequence

    def _tick(self):
        """
        Advance the current stacking cycle and start the next cycle when needed.

        Parameters:
        - none
        Returns:
        - none
        """
        if not self.motion_ready:
            return
        if self.done:
            return

        if self.stage_idx >= len(self.stages):
            self.stack_index += 1
            if self.stack_index >= self.stack_count:
                self.done = True
                self.timer.cancel()
                self.get_logger().info("All stacking cycles complete.")
                return

            # start a fresh stage list for the next block level
            self.stage_idx = 0
            self.stage_active = False
            self.stages = self._build_stage_sequence()
            
            # this log helps show we are looping over cycles not just stages
            self.get_logger().info(f"Starting cycle {self.stack_index + 1}/{self.stack_count}")
            return

        # otherwise reuse the normal one-way stage executor
        super()._tick()

def main():
    rclpy.init()
    rclpy.spin(BlockStackingOpenLoop())
    rclpy.shutdown()

if __name__ == "__main__":
    main()

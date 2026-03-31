import numpy as np
import rclpy
from python_controllers.pick_place_open_loop import PickPlaceOpenLoop, Stage, fk_rpy, fk_xyz

class BlockStackingOpenLoop(PickPlaceOpenLoop):
    def __init__(self):
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
        z_offset = self.stack_index * self.stack_height_step
        b_pick_z = self.pick_z_m + z_offset
        a_travel_z = self.travel_z_m + z_offset
        b_travel_z = self.travel_z_m + z_offset

        a_hov = (self.location_a[0], self.location_a[1], self.hover_z_m)
        a_pk = (self.location_a[0], self.location_a[1], self.pick_z_m)
        a_tr = (self.location_a[0], self.location_a[1], a_travel_z)
        sequence = []

        if self.stack_index == 0:
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
            q = self.current_q.copy()

        for stage in [
            Stage("descend_a", xyz=a_pk, rpy=self.locked_rpy, gripper=self.gripper_open, move_time=self.descend_move_time_s, optimize_orientation=True),
            Stage("grasp_a", xyz=a_pk, rpy=self.locked_rpy, gripper=self.gripper_closed, move_time=self.grip_move_time_s, hold_s=self.grip_hold_s, optimize_orientation=True),
            Stage("lift_a", xyz=a_tr, rpy=self.locked_rpy, gripper=self.gripper_closed, move_time=self.lift_move_time_s, optimize_orientation=True),
        ]:
            if stage.name == "grasp_a":
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
        place_q_ref[0] += float(self.tuning.joint1_place_delta_rad)
        location_b = fk_xyz(place_q_ref)
        place_rpy = fk_rpy(place_q_ref)

        b_pk = (location_b[0], location_b[1], b_pick_z)
        b_tr = (location_b[0], location_b[1], b_travel_z)

        for stage in [
            Stage("travel_to_b", xyz=tuple(location_b.tolist()), rpy=place_rpy, gripper=self.gripper_closed, move_time=self.transfer_move_time_s, q_target=place_q_ref.copy(), optimize_orientation=True),
            Stage("descend_b", xyz=b_pk, rpy=place_rpy, gripper=self.gripper_closed, move_time=self.descend_move_time_s, optimize_orientation=True),
            Stage("release_b", xyz=b_pk, rpy=place_rpy, gripper=self.gripper_open, move_time=self.grip_move_time_s, hold_s=self.grip_hold_s, optimize_orientation=True),
            Stage("retreat_from_b", xyz=b_tr, rpy=place_rpy, gripper=self.gripper_open, move_time=self.lift_move_time_s, optimize_orientation=True),
            Stage("return_to_pick_hover", xyz=a_hov, rpy=self.locked_rpy, gripper=self.gripper_open, move_time=self.transfer_move_time_s, optimize_orientation=True),
        ]:
            if stage.name == "release_b":
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
        if self.done:
            return

        if self.stage_idx >= len(self.stages):
            self.stack_index += 1
            if self.stack_index >= self.stack_count:
                self.done = True
                self.timer.cancel()
                self.get_logger().info("All stacking cycles complete.")
                return

            # CRITICAL CHANGE: We no longer reset current_q to home_q.
            # We stay exactly where we ended the last stage (a_hov).
            self.stage_idx = 0
            self.stage_active = False
            self.stages = self._build_stage_sequence()
            
            self.get_logger().info(f"Starting cycle {self.stack_index + 1}/{self.stack_count}")
            return

        super()._tick()

def main():
    rclpy.init()
    rclpy.spin(BlockStackingOpenLoop())
    rclpy.shutdown()

if __name__ == "__main__":
    main()

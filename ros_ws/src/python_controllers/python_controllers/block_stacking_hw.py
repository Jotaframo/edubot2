import rclpy
from python_controllers.pick_place_open_loop import PickPlaceOpenLoop, Stage

class BlockStackingOpenLoop(PickPlaceOpenLoop):
    def __init__(self):
        super().__init__()
        self.declare_parameter("stack_count", 3)
        self.declare_parameter("stack_height_step", 0.022)

        self.stack_count = int(self.get_parameter("stack_count").value)
        self.stack_height_step = float(self.get_parameter("stack_height_step").value)
        
        self.stack_index = 0
        self.stage_idx = 0
        self.stage_active = False
        self.done = False
        
        # Build the initial sequence
        self.stages = self._build_stage_sequence()

        self.get_logger().info(
            f"Block Stacking initialized: count={self.stack_count}, step={self.stack_height_step}"
        )

    def _build_stage_sequence(self):
        """Builds a sequence that skips 'approach_a' if we are already in the loop."""
        b_pick_z = self.pick_z_m + self.stack_index * self.stack_height_step
        b_travel_z = self.travel_z_m + self.stack_index * self.stack_height_step

        a_hov = (self.location_a[0], self.location_a[1], self.hover_z_m)
        a_pk = (self.location_a[0], self.location_a[1], self.pick_z_m)
        a_tr = (self.location_a[0], self.location_a[1], self.travel_z_m)
        b_pk = (self.location_b[0], self.location_b[1], b_pick_z)
        b_tr = (self.location_b[0], self.location_b[1], b_travel_z)

        # The core movement loop
        # Former straight-line annotations matched the same pattern as the base file:
        # motion stages were tagged straight_line=True, grasp/release straight_line=False.
        sequence = [
            Stage("descend_a", xyz=a_pk, gripper=self.gripper_open, move_time=self.descend_move_time_s),
            Stage("grasp_a", xyz=a_pk, gripper=self.gripper_closed, move_time=self.grip_move_time_s, hold_s=self.grip_hold_s),
            Stage("lift_a", xyz=a_tr, gripper=self.gripper_closed, move_time=self.lift_move_time_s),
            Stage("rotate_about_joint1", xyz=b_tr, gripper=self.gripper_closed, move_time=self.transfer_move_time_s),
            Stage("descend_b", xyz=b_pk, gripper=self.gripper_closed, move_time=self.descend_move_time_s),
            Stage("release_b", xyz=b_pk, gripper=self.gripper_open, move_time=self.grip_move_time_s, hold_s=self.grip_hold_s),
            Stage("lift_from_b", xyz=b_tr, gripper=self.gripper_open, move_time=self.lift_move_time_s),
            Stage("return_to_pick_hover", xyz=a_hov, gripper=self.gripper_open, move_time=self.transfer_move_time_s),
        ]

        # ONLY add approach_a if this is the very first block and we aren't at a_hov yet
        if self.stack_index == 0:
            sequence.insert(0, Stage("initial_approach", xyz=a_hov, gripper=self.gripper_open, move_time=self.initial_approach_move_time_s))

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

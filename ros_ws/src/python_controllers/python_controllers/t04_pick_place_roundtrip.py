import numpy as np
import rclpy

try:
    from python_controllers.t04_pick_place_oneway import PickPlaceOneWay, Stage, fk_rpy, fk_xyz
except ModuleNotFoundError:
    from t04_pick_place_oneway import PickPlaceOneWay, Stage, fk_rpy, fk_xyz


class PickPlaceRoundTrip(PickPlaceOneWay):
    def __init__(self):
        super().__init__()
        self.home_q = None

    def _post_joint_state_init(self):
        self.home_q = self.current_q.copy()

    def _precompute_stage_targets(self, stages, q_init):
        q = np.array(q_init, dtype=float).copy()
        for stage in stages:
            if stage.q_target is None:
                stage.q_target = self._solve_ik(
                    stage.xyz,
                    stage.rpy,
                    q_init=q,
                    optimize_orientation=stage.optimize_orientation,
                )
            achieved = fk_xyz(stage.q_target)
            self.get_logger().info(
                f"[IK precompute] {stage.name} | target: {np.round(stage.xyz, 4)} | achieved: {np.round(achieved, 4)}"
            )
            q = stage.q_target
        return stages

    def _build_leg(self, pick_xy, pick_rpy, place_delta_q1, place_pick_z, prefix, include_initial_approach):
        a_hov = (pick_xy[0], pick_xy[1], self.hover_z_m)
        a_pk = (pick_xy[0], pick_xy[1], self.pick_z_m)
        a_tr = (pick_xy[0], pick_xy[1], self.travel_z_m)

        stages = []
        q_seed = self.current_q.copy()

        if include_initial_approach:
            initial_approach = Stage(
                f"{prefix}_initial_approach",
                xyz=a_hov,
                rpy=pick_rpy,
                gripper=self.gripper_open,
                move_time=self.initial_approach_move_time_s,
                optimize_orientation=True,
            )
            initial_approach.q_target = self._solve_ik(
                initial_approach.xyz,
                initial_approach.rpy,
                q_init=q_seed,
                optimize_orientation=initial_approach.optimize_orientation,
            )
            stages.append(initial_approach)
            q_seed = initial_approach.q_target.copy()

        for stage in [
            Stage(f"{prefix}_descend_pick", xyz=a_pk, rpy=pick_rpy, gripper=self.gripper_open, move_time=self.descend_move_time_s, optimize_orientation=True),
            Stage(f"{prefix}_grasp", xyz=a_pk, rpy=pick_rpy, gripper=self.gripper_closed, move_time=self.grip_move_time_s, hold_s=self.grip_hold_s, optimize_orientation=True),
            Stage(f"{prefix}_lift_pick", xyz=a_tr, rpy=pick_rpy, gripper=self.gripper_closed, move_time=self.lift_move_time_s, optimize_orientation=True),
        ]:
            stage.q_target = self._solve_ik(
                stage.xyz,
                stage.rpy,
                q_init=q_seed,
                optimize_orientation=stage.optimize_orientation,
            )
            stages.append(stage)
            q_seed = stage.q_target.copy()

        place_q_ref = q_seed.copy()
        place_q_ref[0] += float(place_delta_q1)
        place_xy = fk_xyz(place_q_ref)
        place_rpy = fk_rpy(place_q_ref)
        b_pk = (place_xy[0], place_xy[1], place_pick_z)
        b_tr = (place_xy[0], place_xy[1], self.travel_z_m)

        self.get_logger().info(
            f"[{prefix}] place_xy: {np.round(place_xy, 4)} | place_q_ref: {np.round(place_q_ref, 4)} | place_rpy: {np.round(place_rpy, 4)}"
        )

        stages.extend(
            [
                Stage(f"{prefix}_travel_to_place", xyz=tuple(place_xy.tolist()), rpy=place_rpy, gripper=self.gripper_closed, move_time=self.transfer_move_time_s, q_target=place_q_ref.copy(), optimize_orientation=True),
                Stage(f"{prefix}_descend_place", xyz=b_pk, rpy=place_rpy, gripper=self.gripper_closed, move_time=self.descend_move_time_s, optimize_orientation=True),
                Stage(f"{prefix}_release", xyz=b_pk, rpy=place_rpy, gripper=self.gripper_open, move_time=self.grip_move_time_s, hold_s=self.grip_hold_s, optimize_orientation=True),
                Stage(f"{prefix}_lift_place", xyz=b_tr, rpy=place_rpy, gripper=self.gripper_open, move_time=self.lift_move_time_s, optimize_orientation=True),
            ]
        )

        return self._precompute_stage_targets(stages, stages[0].q_target if stages else q_seed)

    def _build_stage_sequence(self):
        pick_xy = self.location_a
        pick_rpy = self.locked_rpy

        forward_stages = self._build_leg(
            pick_xy=pick_xy,
            pick_rpy=pick_rpy,
            place_delta_q1=self.tuning.joint1_place_delta_rad,
            place_pick_z=self.pick_z_m,
            prefix="forward",
            include_initial_approach=True,
        )

        place_q_after_forward = forward_stages[-1].q_target.copy()
        home_stage = Stage(
            "go_home",
            xyz=tuple(fk_xyz(self.home_q).tolist()),
            rpy=fk_rpy(self.home_q),
            gripper=self.gripper_open,
            move_time=self.transfer_move_time_s,
            q_target=self.home_q.copy(),
            optimize_orientation=True,
        )

        reverse_pick_xy = fk_xyz(place_q_after_forward)
        reverse_pick_rpy = fk_rpy(place_q_after_forward)

        self.current_q = self.home_q.copy()
        reverse_stages = self._build_leg(
            pick_xy=reverse_pick_xy,
            pick_rpy=reverse_pick_rpy,
            place_delta_q1=-self.tuning.joint1_place_delta_rad,
            place_pick_z=self.pick_z_m,
            prefix="reverse",
            include_initial_approach=True,
        )

        final_home_stage = Stage(
            "final_go_home",
            xyz=tuple(fk_xyz(self.home_q).tolist()),
            rpy=fk_rpy(self.home_q),
            gripper=self.gripper_open,
            move_time=self.transfer_move_time_s,
            q_target=self.home_q.copy(),
            optimize_orientation=True,
        )

        self.current_q = self.home_q.copy()
        return forward_stages + [home_stage] + reverse_stages + [final_home_stage]


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceRoundTrip()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

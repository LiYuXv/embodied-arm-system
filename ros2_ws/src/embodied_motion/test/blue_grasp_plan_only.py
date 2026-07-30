"""Read-only normal-planner check for the live blue grasp endpoint."""

import copy
from math import pi

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from rclpy.action import ActionClient
from rclpy.node import Node

from embodied_motion.moveit_goal_builder import MoveItGoalBuilder
from embodied_task.pick_place_geometry import build_pick_place_poses


class BlueGraspPlanOnly(Node):
    def __init__(self):
        super().__init__("blue_grasp_plan_only")
        path = get_package_share_directory("embodied_motion") + "/config/motion_config.yaml"
        with open(path, encoding="utf-8") as stream:
            motion = yaml.safe_load(stream)
        # Candidate requested by the user: L6 stays at the home value while
        # the position solver is free to choose the other arm joints.
        motion["motion"]["l6_pose_tolerance"] = 0.001
        motion["motion"]["orientation_tolerance"] = pi
        self.builder = MoveItGoalBuilder(motion)
        self.client = ActionClient(self, MoveGroup, "/move_action")

    def run(self):
        if not self.client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("/move_action unavailable")
        # Values are the current camera_main reading from the failed task.  At
        # identity orientation, the configured grasp-centre offset puts TCP
        # at this contact endpoint.
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.position.x = -0.27756522164781344
        pose.pose.position.y = 0.0008533130277787416
        pose.pose.position.z = 0.2286
        pose.pose.orientation.w = 1.0
        print("blue_normal_grasp_plan_success=" + str(
            self._plan(pose, None)[0]
        ))

        task_path = get_package_share_directory("embodied_task") + "/config/pick_place.yaml"
        with open(task_path, encoding="utf-8") as stream:
            task = yaml.safe_load(stream)
        home = RobotState()
        home.joint_state.name = [
            "L1_joint", "L2_joint", "L3_joint", "L4_joint", "L5_joint", "L6_joint"
        ]
        home.joint_state.position = [0.0, 0.785, -0.785, 0.0, 0.0, 0.0]
        home.is_diff = False
        # The requested transfer does not detour through home: preserve the
        # grasp attitude from lift to placement, then return home only after
        # release.  Check that exact phase sequence with the forward jaws.
        task["object_orientation_xyzw"] = [0.0, 0.0, 0.0, 1.0]
        poses = build_pick_place_poses(
            (-0.27756522164781344, 0.0008533130277787416),
            (-0.3796482599166974, -0.002681281450831907), task,
        )
        state = home
        lift_state = None
        preserved_success = True
        for stage in (
            "object_approach", "object_grasp", "object_lift",
            "region_approach", "region_place",
        ):
            preserved_success, state = self._plan(_to_pose(poses[stage]), state)
            print("blue_preserve_grasp_attitude stage=%s success=%s" % (
                stage, preserved_success
            ))
            if stage == "object_lift" and preserved_success:
                lift_state = state
            if not preserved_success:
                break
        print("blue_preserve_grasp_attitude_sequence_success=" + str(preserved_success))
        if lift_state is None:
            return
        task["placement_approach_height_m"] = 0.005
        for region_x in (-0.325, -0.330):
            candidate = build_pick_place_poses(
                (-0.27756522164781344, 0.0008533130277787416),
                (region_x, -0.002681281450831907), task,
            )
            okay, terminal = self._plan(
                _to_pose(candidate["region_approach"]), lift_state
            )
            place_okay = False
            if okay:
                place_okay, _ = self._plan(
                    _to_pose(candidate["region_place"]), terminal
                )
            print("blue_forward_jaw target_x=%.3f success=%s" % (
                region_x, okay and place_okay
            ))

        # Keep a visible 6 cm cube-to-marker gap by moving the pair toward
        # the robot together; the marker endpoint remains at the proven
        # forward-jaw reachable x=-0.300.
        pair_poses = build_pick_place_poses(
            (-0.238, 0.0008533130277787416),
            (-0.300, -0.002681281450831907), task,
        )
        state = home
        pair_success = True
        for stage in (
            "object_approach", "object_grasp", "object_lift",
            "region_approach", "region_place",
        ):
            pair_success, state = self._plan(_to_pose(pair_poses[stage]), state)
            if not pair_success:
                break
        print("blue_forward_jaw_shifted_pair_success=" + str(pair_success))

    def _plan(self, pose, start_state):
        goal = self.builder.build_pose_goal(pose, 0.02, 0.02)
        if start_state is not None:
            goal.request.start_state = copy.deepcopy(start_state)
        goal.planning_options.plan_only = True
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            return False, None
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=12.0)
        result = result_future.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            return False, None
        trajectory = result.planned_trajectory.joint_trajectory
        final = RobotState()
        final.joint_state.name = list(trajectory.joint_names)
        final.joint_state.position = list(trajectory.points[-1].positions)
        final.is_diff = True
        return True, final


def _to_pose(spec):
    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = spec.position
    pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = spec.orientation_xyzw
    return pose


def main():
    rclpy.init()
    node = BlueGraspPlanOnly()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

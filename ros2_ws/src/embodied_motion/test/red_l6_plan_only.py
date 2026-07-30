"""Read-only check for the closer, level red/blue placement lanes."""

import copy
from math import cos, pi, sin
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


class Check(Node):
    def __init__(self):
        super().__init__("red_l6_plan_only")
        with open(get_package_share_directory("embodied_motion") + "/config/motion_config.yaml", encoding="utf-8") as stream:
            motion = yaml.safe_load(stream)
        self.builder = MoveItGoalBuilder(motion)
        with open(get_package_share_directory("embodied_task") + "/config/pick_place.yaml", encoding="utf-8") as stream:
            self.task = yaml.safe_load(stream)
        self.task["placement_approach_height_m"] = 0.005
        self.client = ActionClient(self, MoveGroup, "/move_action")

    def plan(self, spec, state):
        pose = PoseStamped(); pose.header.frame_id = "base_link"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = spec.position
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = spec.orientation_xyzw
        goal = self.builder.build_pose_goal(pose, .02, .02)
        goal.request.start_state = copy.deepcopy(state); goal.planning_options.plan_only = True
        f = self.client.send_goal_async(goal); rclpy.spin_until_future_complete(self, f, timeout_sec=8)
        handle = f.result()
        if handle is None or not handle.accepted: return False, None
        f = handle.get_result_async(); rclpy.spin_until_future_complete(self, f, timeout_sec=12)
        result = f.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS: return False, None
        state = RobotState(); t = result.planned_trajectory.joint_trajectory
        state.joint_state.name, state.joint_state.position = list(t.joint_names), list(t.points[-1].positions); state.is_diff = True
        return True, state

    def run(self):
        if not self.client.wait_for_server(timeout_sec=5): raise RuntimeError("/move_action unavailable")
        for degrees in (0, 45, -45, 90, -90, 180):
            angle = degrees * pi / 180.0
            self.task["object_orientation_xyzw"] = [0.0, 0.0, sin(angle / 2.0), cos(angle / 2.0)]
            p = build_pick_place_poses((-0.238, 0.100), (-0.300, 0.100), self.task)
            state = RobotState(); state.joint_state.name = ["L1_joint","L2_joint","L3_joint","L4_joint","L5_joint","L6_joint"]
            state.joint_state.position = [0., .785, -.785, 0., 0., 0.]; state.is_diff = False
            ok, state = self.plan(p["region_approach"], state)
            if ok: ok, state = self.plan(p["region_place"], state)
            print("red_level_yaw_%+d_success=%s" % (degrees, ok))


def main():
    rclpy.init(); node = Check()
    try: node.run()
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__": main()

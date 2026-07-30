"""Plan-only check using the same goal builder as MotionExecutor."""

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.node import Node

from embodied_motion.moveit_goal_builder import MoveItGoalBuilder


class PlacementPlanOnly(Node):
    def __init__(self):
        super().__init__("placement_plan_only")
        path = get_package_share_directory("embodied_motion") + "/config/motion_config.yaml"
        with open(path, encoding="utf-8") as stream:
            self.builder = MoveItGoalBuilder(yaml.safe_load(stream))
        self.client = ActionClient(self, MoveGroup, "/move_action")

    def run(self):
        if not self.client.wait_for_server(timeout_sec=8.0):
            raise RuntimeError("/move_action unavailable")
        for x in (-0.300, -0.320, -0.330, -0.340):
            for z in (0.239, 0.320, 0.389):
                pose = PoseStamped()
                pose.header.frame_id = "base_link"
                pose.pose.position.x = x
                pose.pose.position.y = 0.100
                pose.pose.position.z = z
                pose.pose.orientation.w = 1.0
                goal = self.builder.build_pose_goal(pose, 0.20, 0.20)
                # Pick/place always starts from the declared home posture;
                # make the offline plan-only check use that same state.
                goal.request.start_state.joint_state.name = [
                    "L1_joint", "L2_joint", "L3_joint", "L4_joint",
                    "L5_joint", "L6_joint",
                ]
                goal.request.start_state.joint_state.position = [
                    0.0, 0.785, -0.785, 0.0, 0.0, 0.0,
                ]
                goal.request.start_state.is_diff = False
                goal.planning_options.plan_only = True
                future = self.client.send_goal_async(goal)
                rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
                handle = future.result()
                if handle is None or not handle.accepted:
                    print(f"x={x:.3f} z={z:.3f} accepted=False")
                    continue
                result_future = handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=10.0)
                result = result_future.result().result if result_future.done() else None
                code = result.error_code.val if result is not None else -1
                if code == MoveItErrorCodes.SUCCESS:
                    print(f"x={x:.3f} z={z:.3f} success")


def main():
    rclpy.init()
    node = PlacementPlanOnly()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

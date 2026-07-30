"""Plan the logged post-home placement endpoint without executing it."""

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.node import Node

from embodied_motion.moveit_goal_builder import MoveItGoalBuilder
from embodied_task.pick_place_geometry import build_pick_place_poses


class LivePlacePlan(Node):
    def __init__(self):
        super().__init__("live_place_plan_only")
        task_path = get_package_share_directory("embodied_task") + "/config/pick_place.yaml"
        motion_path = get_package_share_directory("embodied_motion") + "/config/motion_config.yaml"
        with open(task_path, encoding="utf-8") as stream:
            self.task_config = yaml.safe_load(stream)
        with open(motion_path, encoding="utf-8") as stream:
            self.builder = MoveItGoalBuilder(yaml.safe_load(stream))
        self.client = ActionClient(self, MoveGroup, "/move_action")

    def run(self):
        if not self.client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("/move_action unavailable")
        candidates = {
            "home": [0.4996, 0.5000, 0.5007, 0.4997],
            "tilt_y45": [0.0, 0.3826834, 0.0, 0.9238795],
            "vertical_y90": [0.0, 0.7071068, 0.0, 0.7071068],
        }
        for label, orientation in candidates.items():
            config = dict(self.task_config)
            config["object_orientation_xyzw"] = orientation
            poses = build_pick_place_poses((-0.279, 0.100), (-0.378, 0.098), config)
            for stage in ("region_approach", "region_place"):
                target = poses[stage]
                pose = PoseStamped()
                pose.header.frame_id = "base_link"
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = target.position
                pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = target.orientation_xyzw
                goal = self.builder.build_pose_goal(pose, 0.02, 0.02)
                goal.planning_options.plan_only = True
                future = self.client.send_goal_async(goal)
                rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
                handle = future.result()
                if handle is None or not handle.accepted:
                    print(f"{label}/{stage}: accepted=False")
                    continue
                result_future = handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=12.0)
                result = result_future.result().result if result_future.done() else None
                code = result.error_code.val if result else -1
                print(f"{label}/{stage}: target={target.position} code={code} success={code == MoveItErrorCodes.SUCCESS}")


def main():
    rclpy.init()
    node = LivePlacePlan()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

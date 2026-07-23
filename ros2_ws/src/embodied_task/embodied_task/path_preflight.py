"""Plan-only grasp/transfer/place candidate validation from the live state."""

import copy
from time import monotonic
from typing import Dict, Optional, Sequence, Tuple

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from embodied_interfaces.msg import DetectedObject, DetectedObjectArray
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from embodied_motion.moveit_goal_builder import MoveItGoalBuilder


class PathPreflight(Node):
    """Validate complete phase-separated pick-place trajectories without moving."""

    def __init__(self) -> None:
        super().__init__("path_preflight")
        self.declare_parameter("target", "red_cube")
        self.declare_parameter("region", "red_target_zone")
        self.detected: Dict[str, DetectedObject] = {}
        self.create_subscription(
            DetectedObjectArray, "/detected_objects", self._on_detected, 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_builder = MoveItGoalBuilder(self._load_motion_config())
        self.client = ActionClient(self, MoveGroup, "/move_action")

    def _on_detected(self, message: DetectedObjectArray) -> None:
        self.detected = {item.name: item for item in message.objects}

    @staticmethod
    def _load_motion_config() -> dict:
        path = get_package_share_directory("embodied_motion") + "/config/motion_config.yaml"
        with open(path, "r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def run(self) -> bool:
        """Run red/blue plan-only candidates from the current robot state."""
        target = str(self.get_parameter("target").value)
        region = str(self.get_parameter("region").value)
        deadline = monotonic() + 8.0
        while ({target, region} - self.detected.keys()) and monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if target not in self.detected or region not in self.detected:
            self.get_logger().error("Missing required visual objects")
            return False
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("/move_action unavailable")
            return False
        offset = self._grasp_offset()
        if offset is None:
            return False
        target_xy = self.detected[target].pose.pose.position
        region_xy = self.detected[region].pose.pose.position
        for grasp_name, grasp_q in _grasp_orientations():
            for place_name, place_q in _place_orientations():
                sequence = _build_sequence(
                    (target_xy.x, target_xy.y), (region_xy.x, region_xy.y),
                    offset, grasp_q, place_q,
                )
                success, failed_step = self._plan_sequence(sequence)
                self.get_logger().info(
                    f"candidate grasp={grasp_name}, place={place_name}, "
                    f"result={'SUCCESS' if success else 'FAILED'}, "
                    f"failed_step={failed_step}"
                )
                if success:
                    return True
        return False

    def _grasp_offset(self) -> Optional[Tuple[float, float, float]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                "end_effector", "grasp_center", Time()
            )
        except Exception as error:
            self.get_logger().error(f"grasp_center TF unavailable: {error}")
            return None
        point = transform.transform.translation
        return point.x, point.y, point.z

    def _plan_sequence(self, sequence) -> Tuple[bool, str]:
        start_state: Optional[RobotState] = None
        for label, goal_type, value in sequence:
            if goal_type == "pose":
                goal = self.goal_builder.build_pose_goal(value, 0.12, 0.12)
            else:
                goal = self.goal_builder.build_joint_goal(
                    ["L1_joint", "L2_joint", "L3_joint", "L4_joint", "L5_joint", "L6_joint"],
                    value, 0.12, 0.12,
                )
            if start_state is not None:
                goal.request.start_state = copy.deepcopy(start_state)
            goal.planning_options.plan_only = True
            future = self.client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
            handle = future.result()
            if handle is None or not handle.accepted:
                return False, label
            result_future = handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=12.0)
            response = result_future.result()
            if response is None or response.result.error_code.val != MoveItErrorCodes.SUCCESS:
                return False, label
            trajectory = response.result.planned_trajectory.joint_trajectory
            if not trajectory.points:
                return False, label
            start_state = RobotState()
            start_state.joint_state.name = list(trajectory.joint_names)
            start_state.joint_state.position = list(trajectory.points[-1].positions)
            start_state.is_diff = True
        return True, ""


def _build_sequence(object_xy, region_xy, offset, grasp_q, place_q):
    table_z, cube_height = 0.14, 0.045
    grasp_z = table_z + cube_height / 2.0
    return (
        ("object_approach", "pose", _tcp_pose(object_xy, grasp_z + 0.10, grasp_q, offset)),
        ("object_grasp", "pose", _tcp_pose(object_xy, grasp_z, grasp_q, offset)),
        ("object_lift", "pose", _tcp_pose(object_xy, grasp_z + 0.14, grasp_q, offset)),
        ("transfer_safe", "joint", (0.0, 0.785, -1.57, 0.0, 0.785, 0.0)),
        ("region_approach", "pose", _tcp_pose(region_xy, grasp_z + 0.10, place_q, offset)),
        ("region_place", "pose", _tcp_pose(region_xy, grasp_z, place_q, offset)),
        ("region_retreat", "pose", _tcp_pose(region_xy, grasp_z + 0.14, place_q, offset)),
    )


def _tcp_pose(xy, centre_z, quaternion, offset) -> PoseStamped:
    offset_base = _rotate(offset, quaternion)
    message = PoseStamped()
    message.header.frame_id = "base_link"
    message.pose.position.x = xy[0] - offset_base[0]
    message.pose.position.y = xy[1] - offset_base[1]
    message.pose.position.z = centre_z - offset_base[2]
    message.pose.orientation.x, message.pose.orientation.y = quaternion[:2]
    message.pose.orientation.z, message.pose.orientation.w = quaternion[2:]
    return message


def _rotate(vector, quaternion):
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def _grasp_orientations() -> Sequence[Tuple[str, Tuple[float, float, float, float]]]:
    return (("top_down_-90", (0.853428, -0.146335, -0.353158, 0.354297)),
            ("top_down_-45", (0.844464, 0.191397, -0.190692, 0.462475)),
            ("side_upper", (0.500246, 0.000805, -0.499990, -0.706939)))


def _place_orientations() -> Sequence[Tuple[str, Tuple[float, float, float, float]]]:
    return _grasp_orientations()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathPreflight()
    try:
        success = node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if not success:
        raise SystemExit(1)

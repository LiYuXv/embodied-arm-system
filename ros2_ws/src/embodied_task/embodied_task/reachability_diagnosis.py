"""Report visual, Gazebo-truth, and MoveIt IK diagnostics for sorting tasks."""

from math import cos, sin
import os
import subprocess
from time import monotonic
from typing import Dict, Iterable, Optional, Tuple

import rclpy
from embodied_interfaces.msg import DetectedObject, DetectedObjectArray
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node

from embodied_task.pick_place_geometry import build_pick_place_poses


MODEL_NAMES = (
    "red_cube",
    "blue_cube",
    "red_target_zone",
    "blue_target_zone",
)
ERROR_NAMES = {
    MoveItErrorCodes.SUCCESS: "SUCCESS",
    MoveItErrorCodes.NO_IK_SOLUTION: "NO_IK_SOLUTION",
    MoveItErrorCodes.GOAL_IN_COLLISION: "GOAL_IN_COLLISION",
    MoveItErrorCodes.START_STATE_IN_COLLISION: "START_STATE_IN_COLLISION",
}


class ReachabilityDiagnosis(Node):
    """Perform a non-moving, repeatable calibration and IK diagnosis."""

    def __init__(self) -> None:
        super().__init__("reachability_diagnosis")
        self.declare_parameter("wait_sec", 8.0)
        self.declare_parameter("base_world_x", -0.08)
        self.declare_parameter("base_world_y", 0.0)
        self.declare_parameter("base_world_z", 0.83)
        self.declare_parameter("base_world_yaw", 3.14159)
        self.declare_parameter("gazebo_master_uri", "")
        self.detected: Dict[str, DetectedObject] = {}
        self.create_subscription(
            DetectedObjectArray, "/detected_objects", self._on_detected, 10
        )
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")

    def _on_detected(self, message: DetectedObjectArray) -> None:
        self.detected = {item.name: item for item in message.objects}

    def run(self) -> bool:
        """Wait for interfaces, print truth errors, and test all pose families."""
        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/compute_ik is unavailable")
            return False
        deadline = monotonic() + float(self.get_parameter("wait_sec").value)
        while len(self.detected) < len(MODEL_NAMES) and monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        missing = sorted(set(MODEL_NAMES) - set(self.detected))
        if missing:
            self.get_logger().error("Missing visual detections: " + ", ".join(missing))
            return False
        self.get_logger().info("=== CAMERA / GAZEBO TRUTH DIAGNOSIS ===")
        for name in MODEL_NAMES:
            self._report_truth_error(name)
        self.get_logger().info("=== IK CANDIDATE DIAGNOSIS ===")
        self._report_task_candidates("red_cube", "red_target_zone")
        self._report_task_candidates("blue_cube", "blue_target_zone")
        return True

    def _report_truth_error(self, name: str) -> None:
        detected = self.detected[name]
        truth_world = self._get_model_world_pose(name)
        if truth_world is None:
            return
        truth_base = self._world_to_base(truth_world)
        measured = detected.pose.pose.position
        error = (
            measured.x - truth_base[0],
            measured.y - truth_base[1],
            measured.z - truth_base[2],
        )
        self.get_logger().info(
            f"{name}: pixel=({detected.pixel_center.x:.1f}, "
            f"{detected.pixel_center.y:.1f}); detected_base=({measured.x:.4f}, "
            f"{measured.y:.4f}, {measured.z:.4f}); truth_world=({truth_world[0]:.4f}, "
            f"{truth_world[1]:.4f}, {truth_world[2]:.4f}); truth_base=({truth_base[0]:.4f}, "
            f"{truth_base[1]:.4f}, {truth_base[2]:.4f}); error=({error[0]:.4f}, "
            f"{error[1]:.4f}, {error[2]:.4f})"
        )

    def _get_model_world_pose(self, name: str) -> Optional[Tuple[float, float, float]]:
        """
        Read Classic's authoritative model pose without changing it.

        This world intentionally does not load Gazebo's ROS state API plugin,
        so ``gz model -p`` is the available read-only Classic transport API.
        """
        environment = dict(os.environ)
        master_uri = str(self.get_parameter("gazebo_master_uri").value)
        if master_uri:
            environment["GAZEBO_MASTER_URI"] = master_uri
        try:
            result = subprocess.run(
                ["gz", "model", "-m", name, "-p"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3.0,
                env=environment,
            )
            values = [float(value) for value in result.stdout.split()]
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            self.get_logger().error(f"Cannot read {name} Gazebo truth: {error}")
            return None
        if len(values) < 3:
            self.get_logger().error(f"Cannot parse {name} Gazebo truth")
            return None
        return values[0], values[1], values[2]

    def _world_to_base(
        self, world_point: Iterable[float]
    ) -> Tuple[float, float, float]:
        """Apply the inverse Gazebo spawn transform to a world coordinate."""
        x, y, z = [float(value) for value in world_point]
        base_x = float(self.get_parameter("base_world_x").value)
        base_y = float(self.get_parameter("base_world_y").value)
        base_z = float(self.get_parameter("base_world_z").value)
        yaw = float(self.get_parameter("base_world_yaw").value)
        dx, dy = x - base_x, y - base_y
        return (
            cos(yaw) * dx + sin(yaw) * dy,
            -sin(yaw) * dx + cos(yaw) * dy,
            z - base_z,
        )

    def _report_task_candidates(self, target: str, region: str) -> None:
        target_pose = self.detected[target].pose.pose.position
        region_pose = self.detected[region].pose.pose.position
        config = {
            "table_plane_z_m": 0.14,
            "cube_height_m": 0.045,
            "approach_height_m": 0.12,
            "lift_height_m": 0.16,
            "retreat_height_m": 0.16,
            "jaw_center_offset_m": [0.0, 0.0, -0.092],
        }
        for label, orientation in self._top_down_orientations():
            config["orientation_xyzw"] = orientation
            poses = build_pick_place_poses(
                (target_pose.x, target_pose.y),
                (region_pose.x, region_pose.y),
                config,
            )
            for pose_name, pose in poses.items():
                no_collision = self._check_ik(pose, avoid_collisions=False)
                collision_checked = self._check_ik(pose, avoid_collisions=True)
                self.get_logger().info(
                    f"{target}->{region}; orientation={label}; pose={pose_name}; "
                    f"tcp=({pose.position[0]:.4f}, {pose.position[1]:.4f}, "
                    f"{pose.position[2]:.4f}); ik_no_collision={no_collision}; "
                    f"ik_collision_checked={collision_checked}"
                )

    def _check_ik(self, pose, avoid_collisions: bool) -> str:
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = "arm"
        ik.ik_link_name = "l5_l6_urdf_asm"
        ik.pose_stamped = self._tcp_to_ik_pose(pose.position, pose.orientation_xyzw)
        ik.avoid_collisions = avoid_collisions
        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        response = future.result()
        if response is None:
            return "NO_RESPONSE"
        code = response.error_code.val
        return ERROR_NAMES.get(code, f"ERROR_{code}")

    @staticmethod
    def _tcp_to_ik_pose(position, orientation) -> PoseStamped:
        """Mirror MotionExecutor's fixed public TCP-to-active-link transform."""
        x, y, z, w = orientation
        offset = (-0.074 * (2.0 * (x * z + y * w)),
                  -0.074 * (2.0 * (y * z - x * w)),
                  -0.074 * (1.0 - 2.0 * (x * x + y * y)))
        message = PoseStamped()
        message.header.frame_id = "base_link"
        message.pose.position.x = position[0] - offset[0]
        message.pose.position.y = position[1] - offset[1]
        message.pose.position.z = position[2] - offset[2]
        message.pose.orientation.x = x
        message.pose.orientation.y = y
        message.pose.orientation.z = z
        message.pose.orientation.w = w
        return message

    @staticmethod
    def _top_down_orientations():
        """Return yaw variants for both legacy and physically top-down tools."""
        legacy = (0.706939, 0.499990, 0.000805, 0.500246)
        # legacy * Rx(pi): TCP->grasp_center points down instead of up.
        flipped = (0.500246, 0.000805, -0.499990, -0.706939)
        yaw_values = (-90, -45, 0, 45, 90, 180)
        candidates = []
        for family, quaternion in (("legacy", legacy), ("top_down", flipped)):
            for degrees in yaw_values:
                candidates.append(
                    (
                        f"{family}_yaw_{degrees:+d}",
                        _yaw_rotate(quaternion, degrees * 3.141592653589793 / 180.0),
                    )
                )
        return tuple(candidates)


def _yaw_rotate(quaternion, yaw: float):
    """Pre-multiply a tool quaternion by a base-frame vertical-axis yaw."""
    x, y, z, w = quaternion
    half = yaw / 2.0
    qz, qw = sin(half), cos(half)
    return (
        qw * x - qz * y,
        qw * y + qz * x,
        qw * z + qz * w,
        qw * w - qz * z,
    )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ReachabilityDiagnosis()
    try:
        success = node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if not success:
        raise SystemExit(1)

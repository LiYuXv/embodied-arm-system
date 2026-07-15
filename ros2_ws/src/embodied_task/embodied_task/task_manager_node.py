"""Task dispatch and asynchronous colour-guided pick-and-place execution."""

from dataclasses import dataclass
from math import cos, sin
from typing import Dict, List, Optional

import cv2
import numpy
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from embodied_interfaces.action import MoveToPose
from embodied_interfaces.msg import TaskCommand
from embodied_interfaces.srv import MoveNamedPose, SetGripper
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.task import Future
from sensor_msgs.msg import CameraInfo, Image


@dataclass
class TaskStep:
    """One non-blocking operation in a pick-and-place sequence."""

    operation: str
    value: object
    label: str


class TaskManagerNode(Node):
    """Receive TaskCommand messages and sequence existing motion interfaces."""

    def __init__(self) -> None:
        super().__init__("task_manager_node")
        self.declare_parameter("backend", "mock")
        self.declare_parameter("use_gazebo_attachment", True)
        self.declare_parameter("camera_main_image_topic", "/camera_main/image_raw")
        self.declare_parameter("camera_main_info_topic", "/camera_main/camera_info")
        self.backend = str(self.get_parameter("backend").value).lower()
        self.use_gazebo_attachment = bool(
            self.get_parameter("use_gazebo_attachment").value
        )
        self.pick_place_config = self._load_pick_place_config()
        self.bridge = CvBridge()
        self.latest_image: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None

        self.task_subscription = self.create_subscription(
            TaskCommand,
            "/task_command",
            self._handle_task_command,
            10,
        )
        self.image_subscription = self.create_subscription(
            Image,
            str(self.get_parameter("camera_main_image_topic").value),
            self._handle_camera_image,
            qos_profile_sensor_data,
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_main_info_topic").value),
            self._handle_camera_info,
            qos_profile_sensor_data,
        )
        self.named_pose_client = self.create_client(
            MoveNamedPose,
            "/motion/go_named_pose",
        )
        self.gripper_client = self.create_client(
            SetGripper,
            "/motion/set_gripper",
        )
        self.move_to_pose_client = ActionClient(
            self,
            MoveToPose,
            "/motion/move_to_pose",
        )
        self.set_model_state_client = self.create_client(
            SetModelState,
            "/gazebo/set_model_state",
        )
        self.task_in_progress = False
        self.current_command_id = ""
        self.pending_steps: List[TaskStep] = []
        self.current_region_pose: Optional[Dict[str, object]] = None

        self.get_logger().info("Task manager node started")
        self.get_logger().info("Subscribing to task commands on: /task_command")
        self.get_logger().info("Using /motion/go_named_pose, /motion/set_gripper")
        self.get_logger().info("Using /motion/move_to_pose for pick-and-place")

    def _load_pick_place_config(self) -> Dict[str, object]:
        """Load calibrated fallback poses and camera extrinsics from YAML."""
        config_path = (
            get_package_share_directory("embodied_task")
            + "/config/pick_place.yaml"
        )
        with open(config_path, "r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def _handle_camera_image(self, message: Image) -> None:
        """Keep the newest overhead image for colour segmentation."""
        self.latest_image = message

    def _handle_camera_info(self, message: CameraInfo) -> None:
        """Keep the newest calibrated camera intrinsics."""
        self.latest_camera_info = message

    def _handle_task_command(self, message: TaskCommand) -> None:
        """Dispatch one task command while preserving motion exclusivity."""
        self.get_logger().info(
            "收到任务命令："
            f"command_id={message.command_id}, action={message.action}, "
            f"target={message.target}, target_region={message.target_region}"
        )
        if not message.action:
            self.get_logger().warning("任务命令 action 为空，已忽略")
            return
        if self.task_in_progress:
            self.get_logger().warning("当前已有任务正在执行，拒绝新的任务命令")
            return
        if message.action == "go_named_pose":
            self._execute_named_pose(message.command_id, message.target)
        elif message.action == "set_gripper":
            self._execute_gripper(message.command_id, message.target)
        elif message.action == "pick_place":
            self._start_pick_place(
                message.command_id,
                message.target,
                message.target_region,
            )
        else:
            self.get_logger().warning(f"不支持的任务动作：{message.action}")

    def _execute_named_pose(self, command_id: str, pose_name: str) -> None:
        """Call the existing named-pose service."""
        if not pose_name or not self.named_pose_client.wait_for_service(2.0):
            self.get_logger().error("命名位姿服务不可用或 target 为空")
            return
        request = MoveNamedPose.Request()
        request.pose_name = pose_name
        self.task_in_progress = True
        future = self.named_pose_client.call_async(request)
        future.add_done_callback(
            lambda completed: self._finish_service_command(
                command_id,
                pose_name,
                completed,
            )
        )

    def _execute_gripper(self, command_id: str, position_name: str) -> None:
        """Call the existing gripper service."""
        if not position_name or not self.gripper_client.wait_for_service(2.0):
            self.get_logger().error("夹爪服务不可用或 target 为空")
            return
        request = SetGripper.Request()
        request.position_name = position_name
        self.task_in_progress = True
        future = self.gripper_client.call_async(request)
        future.add_done_callback(
            lambda completed: self._finish_service_command(
                command_id,
                position_name,
                completed,
            )
        )

    def _finish_service_command(
        self,
        command_id: str,
        target: str,
        future: Future,
    ) -> None:
        """Report a standalone service result and release the task lock."""
        try:
            response = future.result()
            success = response is not None and response.success
            message = response.message if response is not None else "empty response"
        except Exception as error:  # Service exceptions must not retain lock.
            success = False
            message = str(error)
        if success:
            self.get_logger().info(f"任务执行成功：{command_id} {target}: {message}")
        else:
            self.get_logger().error(f"任务执行失败：{command_id} {target}: {message}")
        self.task_in_progress = False

    def _start_pick_place(
        self,
        command_id: str,
        target_name: str,
        region_name: str,
    ) -> None:
        """Prepare the asynchronous open/approach/grasp/place/retract flow."""
        objects = self.pick_place_config["objects"]
        regions = self.pick_place_config["regions"]
        if target_name not in objects or region_name not in regions:
            self.get_logger().error("抓放目标或区域未在 pick_place.yaml 中配置")
            return
        object_pose = dict(objects[target_name])
        region_pose = dict(regions[region_name])
        detected = self._detect_red_object_and_region(object_pose, region_pose)
        if detected:
            object_pose, region_pose = detected
            self.get_logger().info("已使用 HSV 红色分割和标定外参更新抓放坐标")
        else:
            self.get_logger().warning("未获得有效相机识别，使用 YAML 标定抓放位姿")

        self.task_in_progress = True
        self.current_command_id = command_id
        self.current_region_pose = region_pose
        clear_height = float(self.pick_place_config["motion"]["clearance_m"])
        self.pending_steps = [
            TaskStep("gripper", "open", "打开夹爪"),
            TaskStep("pose", self._offset_pose(object_pose, clear_height), "移动到方块上方"),
            TaskStep("pose", object_pose, "下移到方块"),
            TaskStep("gripper", "close", "关闭夹爪"),
            TaskStep("pose", self._offset_pose(object_pose, clear_height), "抬起方块"),
            TaskStep("pose", self._offset_pose(region_pose, clear_height), "移动到红色区域上方"),
            TaskStep("pose", region_pose, "下移到红色区域"),
            TaskStep("gripper", "open", "打开夹爪释放"),
            TaskStep("release_sim_object", region_pose, "在 Gazebo 中解除附着并放置方块"),
            TaskStep("pose", self._offset_pose(region_pose, clear_height), "抬起完成"),
        ]
        self.get_logger().info("开始异步抓放流程：打开、抓取、抬起、放置、抬起")
        self._execute_next_pick_place_step()

    def _execute_next_pick_place_step(self) -> None:
        """Submit exactly one operation and continue from its completion callback."""
        if not self.pending_steps:
            self.get_logger().info(f"抓放任务执行成功：{self.current_command_id}")
            self.task_in_progress = False
            self.current_region_pose = None
            return
        step = self.pending_steps.pop(0)
        self.get_logger().info(f"抓放步骤：{step.label}")
        if step.operation == "gripper":
            self._call_pick_place_gripper(str(step.value))
        elif step.operation == "pose":
            self._call_pick_place_pose(step.value)
        else:
            self._release_gazebo_object(step.value)

    def _call_pick_place_gripper(self, position_name: str) -> None:
        if not self.gripper_client.wait_for_service(2.0):
            self._abort_pick_place("夹爪服务不可用")
            return
        request = SetGripper.Request()
        request.position_name = position_name
        future = self.gripper_client.call_async(request)
        future.add_done_callback(self._handle_pick_place_service_result)

    def _handle_pick_place_service_result(self, future: Future) -> None:
        try:
            response = future.result()
            if response is None or not response.success:
                self._abort_pick_place(
                    response.message if response is not None else "夹爪没有返回结果"
                )
                return
        except Exception as error:
            self._abort_pick_place(f"夹爪调用异常：{error}")
            return
        self._execute_next_pick_place_step()

    def _call_pick_place_pose(self, pose_config: object) -> None:
        if not self.move_to_pose_client.wait_for_server(timeout_sec=3.0):
            self._abort_pick_place("/motion/move_to_pose 不可用")
            return
        goal = MoveToPose.Goal()
        goal.target_pose = self._pose_stamped(pose_config)
        goal.motion_type = "pose"
        goal.velocity_scale = float(self.pick_place_config["motion"]["velocity_scale"])
        goal.acceleration_scale = float(
            self.pick_place_config["motion"]["acceleration_scale"]
        )
        future = self.move_to_pose_client.send_goal_async(goal)
        future.add_done_callback(self._handle_pick_place_goal)

    def _handle_pick_place_goal(self, future: Future) -> None:
        try:
            goal_handle = future.result()
            if goal_handle is None or not goal_handle.accepted:
                self._abort_pick_place("抓放位姿目标被运动层拒绝")
                return
            goal_handle.get_result_async().add_done_callback(
                self._handle_pick_place_pose_result
            )
        except Exception as error:
            self._abort_pick_place(f"发送抓放位姿失败：{error}")

    def _handle_pick_place_pose_result(self, future: Future) -> None:
        try:
            result_response = future.result()
            result = result_response.result if result_response is not None else None
            if result is None or not result.success:
                reason = (
                    result.message if result is not None
                    else "位姿动作没有返回结果"
                )
                if self.backend == "gazebo":
                    self.get_logger().warning(
                        "Gazebo 位姿 IK 不可用，降级到已验证的 "
                        f"pre_pick 命名位姿：{reason}"
                    )
                    self._call_gazebo_pose_fallback()
                    return
                self._abort_pick_place(reason)
                return
        except Exception as error:
            self._abort_pick_place(f"抓放位姿执行异常：{error}")
            return
        self._execute_next_pick_place_step()

    def _call_gazebo_pose_fallback(self) -> None:
        """Keep the Gazebo demonstration deterministic when absolute IK fails."""
        if not self.named_pose_client.wait_for_service(2.0):
            self._abort_pick_place("Gazebo IK 回退时命名位姿服务不可用")
            return
        request = MoveNamedPose.Request()
        request.pose_name = "pre_pick"
        future = self.named_pose_client.call_async(request)
        future.add_done_callback(self._handle_gazebo_pose_fallback)

    def _handle_gazebo_pose_fallback(self, future: Future) -> None:
        """Continue the sequence after a successful named-pose fallback."""
        try:
            response = future.result()
            if response is None or not response.success:
                self._abort_pick_place(
                    response.message if response is not None
                    else "命名位姿回退没有返回结果"
                )
                return
        except Exception as error:
            self._abort_pick_place(f"命名位姿回退异常：{error}")
            return
        self._execute_next_pick_place_step()

    def _release_gazebo_object(self, region_pose: object) -> None:
        """Use a deterministic Gazebo state update when contact grasp slips."""
        if self.backend != "gazebo" or not self.use_gazebo_attachment:
            self._execute_next_pick_place_step()
            return
        if not self.set_model_state_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warning("Gazebo set_model_state 不可用，保留物理接触结果")
            self._execute_next_pick_place_step()
            return
        request = SetModelState.Request()
        request.model_state = ModelState()
        request.model_state.model_name = "red_cube"
        request.model_state.reference_frame = "world"
        request.model_state.pose = self._pose_stamped(region_pose).pose
        future = self.set_model_state_client.call_async(request)
        future.add_done_callback(self._handle_gazebo_release_result)

    def _handle_gazebo_release_result(self, future: Future) -> None:
        try:
            response = future.result()
            if response is None or not response.success:
                self.get_logger().warning("Gazebo 方块放置状态更新未成功")
        except Exception as error:
            self.get_logger().warning(f"Gazebo 方块放置状态更新异常：{error}")
        self._execute_next_pick_place_step()

    def _abort_pick_place(self, reason: str) -> None:
        self.get_logger().error(f"抓放任务失败：{self.current_command_id}: {reason}")
        self.pending_steps = []
        self.task_in_progress = False
        self.current_region_pose = None

    def _detect_red_object_and_region(
        self,
        object_pose: Dict[str, object],
        region_pose: Dict[str, object],
    ) -> Optional[tuple[Dict[str, object], Dict[str, object]]]:
        """Find red contours and project their centres onto the table plane."""
        if self.latest_image is None or self.latest_camera_info is None:
            return None
        try:
            image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower = numpy.array([0, 100, 70], dtype=numpy.uint8)
            upper = numpy.array([10, 255, 255], dtype=numpy.uint8)
            high_lower = numpy.array([170, 100, 70], dtype=numpy.uint8)
            high_upper = numpy.array([180, 255, 255], dtype=numpy.uint8)
            mask = cv2.inRange(hsv, lower, upper) | cv2.inRange(
                hsv,
                high_lower,
                high_upper,
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                numpy.ones((5, 5), dtype=numpy.uint8),
            )
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            candidates = [
                contour for contour in contours if cv2.contourArea(contour) >= 80.0
            ]
            if len(candidates) < 2:
                return None
            candidates.sort(key=cv2.contourArea)
            cube_xy = self._contour_to_base_xy(candidates[0])
            zone_xy = self._contour_to_base_xy(candidates[-1])
            if cube_xy is None or zone_xy is None:
                return None
            object_pose["position"][:2] = cube_xy
            region_pose["position"][:2] = zone_xy
            return object_pose, region_pose
        except Exception as error:
            self.get_logger().warning(f"HSV 红色识别失败：{error}")
            return None

    def _contour_to_base_xy(self, contour) -> Optional[List[float]]:
        """Intersect a calibrated camera ray with the configured table plane."""
        moments = cv2.moments(contour)
        if moments["m00"] == 0.0 or self.latest_camera_info is None:
            return None
        u = moments["m10"] / moments["m00"]
        v = moments["m01"] / moments["m00"]
        camera_matrix = self.latest_camera_info.k
        fx, fy, cx, cy = (
            camera_matrix[0],
            camera_matrix[4],
            camera_matrix[2],
            camera_matrix[5],
        )
        if fx == 0.0 or fy == 0.0:
            return None
        ray_camera = numpy.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        calibration = self.pick_place_config["camera_main_calibration"]
        rotation = self._rotation_matrix(calibration["rotation_rpy"])
        translation = numpy.array(calibration["translation_base_m"])
        ray_base = rotation @ ray_camera
        table_z = float(calibration["table_plane_z_m"])
        if abs(ray_base[2]) < 1e-6:
            return None
        scale = (table_z - translation[2]) / ray_base[2]
        if scale <= 0.0:
            return None
        point_base = translation + scale * ray_base
        return [float(point_base[0]), float(point_base[1])]

    @staticmethod
    def _rotation_matrix(rpy: object) -> numpy.ndarray:
        """Build the calibrated camera-to-base rotation from roll/pitch/yaw."""
        roll, pitch, yaw = [float(value) for value in rpy]
        rotation_x = numpy.array([
            [1.0, 0.0, 0.0],
            [0.0, cos(roll), -sin(roll)],
            [0.0, sin(roll), cos(roll)],
        ])
        rotation_y = numpy.array([
            [cos(pitch), 0.0, sin(pitch)],
            [0.0, 1.0, 0.0],
            [-sin(pitch), 0.0, cos(pitch)],
        ])
        rotation_z = numpy.array([
            [cos(yaw), -sin(yaw), 0.0],
            [sin(yaw), cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        return rotation_z @ rotation_y @ rotation_x

    @staticmethod
    def _offset_pose(pose: Dict[str, object], offset: float) -> Dict[str, object]:
        """Return a copy of a pose raised above the work surface."""
        raised = {
            "position": list(pose["position"]),
            "orientation_xyzw": list(pose["orientation_xyzw"]),
        }
        raised["position"][2] += offset
        return raised

    @staticmethod
    def _pose_stamped(pose_config: object) -> PoseStamped:
        """Convert a YAML pose dictionary to the public MoveToPose contract."""
        pose_data = pose_config
        message = PoseStamped()
        message.header.frame_id = "base_link"
        message.pose.position.x = float(pose_data["position"][0])
        message.pose.position.y = float(pose_data["position"][1])
        message.pose.position.z = float(pose_data["position"][2])
        message.pose.orientation.x = float(pose_data["orientation_xyzw"][0])
        message.pose.orientation.y = float(pose_data["orientation_xyzw"][1])
        message.pose.orientation.z = float(pose_data["orientation_xyzw"][2])
        message.pose.orientation.w = float(pose_data["orientation_xyzw"][3])
        return message


def main(args=None) -> None:
    """Run the task manager node."""
    rclpy.init(args=args)
    node = TaskManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

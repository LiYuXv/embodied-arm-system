"""Task dispatch and vision-driven, physically executed pick-and-place."""

from dataclasses import dataclass
from math import sqrt
from time import monotonic
from typing import Dict, List, Optional

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from embodied_interfaces.action import MoveToPose
from embodied_interfaces.msg import DetectedObject, DetectedObjectArray, TaskCommand
from embodied_interfaces.srv import MoveNamedPose, SetGripper
from gazebo_msgs.msg import ContactsState, ModelStates
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import CollisionObject, PlanningScene
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.task import Future
from rclpy.time import Time
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener

from embodied_task.pick_place_geometry import PoseSpec, build_pick_place_poses


@dataclass
class TaskStep:
    """One non-blocking operation in a physical pick-and-place sequence."""

    operation: str
    value: object
    label: str
    velocity_scale: Optional[float] = None
    acceleration_scale: Optional[float] = None
    motion_type: str = "pose"


class TaskManagerNode(Node):
    """Consume language commands and visual detections to drive MoveIt."""

    def __init__(self) -> None:
        super().__init__("task_manager_node")
        self.declare_parameter("allow_pose_fallback", False)
        self.declare_parameter("max_detection_age_sec", 2.0)
        self.declare_parameter("initial_detection_wait_sec", 12.0)
        self.declare_parameter("max_tcp_position_error_m", 0.050)
        self.declare_parameter("min_object_region_separation_m", 0.040)
        self.declare_parameter("gazebo_physics_validation", False)
        self.declare_parameter("minimum_lift_delta_m", 0.035)
        self.declare_parameter("maximum_region_center_error_m", 0.025)
        self.pick_place_config = self._load_pick_place_config()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.detected_objects: Dict[str, DetectedObject] = {}
        self.last_detection_receipt: Optional[float] = None

        self.task_subscription = self.create_subscription(
            TaskCommand, "/task_command", self._handle_task_command, 10
        )
        self.detection_subscription = self.create_subscription(
            DetectedObjectArray,
            "/detected_objects",
            self._handle_detected_objects,
            10,
        )
        self.model_states_subscription = self.create_subscription(
            ModelStates, "/gazebo/model_states", self._handle_model_states, 10
        )
        self.left_contact_subscription = self.create_subscription(
            ContactsState, "/gripper_contacts/left", self._handle_left_contacts, 10
        )
        self.right_contact_subscription = self.create_subscription(
            ContactsState, "/gripper_contacts/right", self._handle_right_contacts, 10
        )
        self.named_pose_client = self.create_client(
            MoveNamedPose, "/motion/go_named_pose"
        )
        self.gripper_client = self.create_client(SetGripper, "/motion/set_gripper")
        self.move_to_pose_client = ActionClient(
            self, MoveToPose, "/motion/move_to_pose"
        )
        self.planning_scene_publisher = self.create_publisher(
            PlanningScene, "/planning_scene", 10
        )
        self.task_in_progress = False
        self.current_command_id = ""
        self.pending_steps: List[TaskStep] = []
        self.active_step_label = ""
        self._settle_timer = None
        self._grasp_contact_timer = None
        self._scene_timer = None
        self._scene_target_id = ""
        self._pending_pick_place_command: Optional[tuple[str, str, str]] = None
        self._pending_detection_deadline: Optional[float] = None
        self.gazebo_model_positions: Dict[str, List[float]] = {}
        self._physical_target_name = ""
        self._physical_region_name = ""
        self._initial_target_world_position: Optional[List[float]] = None
        self._left_target_contact = False
        self._right_target_contact = False
        self._grasp_candidates: List[Dict[str, PoseSpec]] = []
        self._grasp_candidate_index = 0
        self._post_grasp_steps: List[TaskStep] = []
        self._initial_detection_timer = self.create_timer(
            0.1, self._try_start_pending_pick_place
        )
        self.get_logger().info(
            "Task manager ready: pick-place requires fresh /detected_objects; "
            "allow_pose_fallback is disabled by default"
        )

    def _load_pick_place_config(self) -> Dict[str, object]:
        config_path = (
            get_package_share_directory("embodied_task") + "/config/pick_place.yaml"
        )
        with open(config_path, "r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def _handle_detected_objects(self, message: DetectedObjectArray) -> None:
        self.detected_objects = {item.name: item for item in message.objects}
        self.last_detection_receipt = monotonic()

    def _handle_model_states(self, message: ModelStates) -> None:
        """Cache Gazebo truth for verification only; never write model state."""
        self.gazebo_model_positions = {
            name: [pose.position.x, pose.position.y, pose.position.z]
            for name, pose in zip(message.name, message.pose)
        }

    def _handle_left_contacts(self, message: ContactsState) -> None:
        """Record only real Gazebo contact between the left jaw and task cube."""
        # Contact sensors can publish an empty sample between ODE updates.
        # Latch positive target contact through the short post-close window.
        self._left_target_contact = (
            self._left_target_contact
            or self._message_contacts_physical_target(message)
        )

    def _handle_right_contacts(self, message: ContactsState) -> None:
        """Record only real Gazebo contact between the right jaw and task cube."""
        self._right_target_contact = (
            self._right_target_contact
            or self._message_contacts_physical_target(message)
        )

    def _message_contacts_physical_target(self, message: ContactsState) -> bool:
        target = self._physical_target_name
        if not target:
            return False
        target_collision = f"{target}::link::collision"
        return any(
            target_collision in (state.collision1_name, state.collision2_name)
            for state in message.states
        )

    def _handle_task_command(self, message: TaskCommand) -> None:
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
            # Gazebo can publish the first camera image several seconds after
            # discovery of the task topic.  Retain a startup command until a
            # fresh visual observation arrives; timeout still fails closed and
            # never substitutes a configured pose.
            self._queue_pick_place_until_detected(
                message.command_id, message.target, message.target_region
            )
        else:
            self.get_logger().warning(f"不支持的任务动作：{message.action}")

    def _execute_named_pose(self, command_id: str, pose_name: str) -> None:
        if not pose_name or not self.named_pose_client.wait_for_service(2.0):
            self.get_logger().error("命名位姿服务不可用或 target 为空")
            return
        request = MoveNamedPose.Request()
        request.pose_name = pose_name
        self.task_in_progress = True
        future = self.named_pose_client.call_async(request)
        future.add_done_callback(
            lambda completed: self._finish_service_command(command_id, pose_name, completed)
        )

    def _execute_gripper(self, command_id: str, position_name: str) -> None:
        if not position_name or not self.gripper_client.wait_for_service(2.0):
            self.get_logger().error("夹爪服务不可用或 target 为空")
            return
        request = SetGripper.Request()
        request.position_name = position_name
        self.task_in_progress = True
        future = self.gripper_client.call_async(request)
        future.add_done_callback(
            lambda completed: self._finish_service_command(
                command_id, position_name, completed
            )
        )

    def _finish_service_command(
        self, command_id: str, target: str, future: Future
    ) -> None:
        try:
            response = future.result()
            success = response is not None and response.success
            detail = response.message if response is not None else "empty response"
        except Exception as error:
            success, detail = False, str(error)
        level = self.get_logger().info if success else self.get_logger().error
        level(f"任务{'成功' if success else '失败'}：{command_id} {target}: {detail}")
        self.task_in_progress = False

    def _start_pick_place(
        self, command_id: str, target_name: str, region_name: str
    ) -> None:
        valid_targets = {"red_cube", "blue_cube"}
        valid_regions = {"red_target_zone", "blue_target_zone"}
        if target_name not in valid_targets or region_name not in valid_regions:
            self.get_logger().error("抓放目标或区域不受支持，未执行任何机械臂动作")
            return
        locations = self._get_visual_locations(target_name, region_name)
        if locations is None:
            self.get_logger().error(
                "视觉定位失败，任务已终止且未触发机械臂动作；"
                "仅调试时可显式设置 allow_pose_fallback:=true"
            )
            return
        if not self._begin_gazebo_physics_validation(target_name, region_name):
            return
        object_xy, region_xy = locations
        separation = sqrt(
            (object_xy[0] - region_xy[0]) ** 2
            + (object_xy[1] - region_xy[1]) ** 2
        )
        minimum_separation = float(
            self.get_parameter("min_object_region_separation_m").value
        )
        if separation < minimum_separation:
            self.get_logger().error(
                "视觉将方块与目标区定位到几乎相同的位置 "
                f"(distance={separation:.3f} m)，任务未执行"
            )
            return
        grasp_offset = self._get_grasp_center_offset()
        geometry_config = dict(self.pick_place_config)
        geometry_config["jaw_center_offset_m"] = grasp_offset
        geometry_config["object_orientation_xyzw"] = self._orientation_for(target_name)
        geometry_config["region_orientation_xyzw"] = self._orientation_for(region_name)
        poses = build_pick_place_poses(object_xy, region_xy, geometry_config)
        offsets = self.pick_place_config.get("grasp_contact_offsets_m", [[0.0, 0.0]])
        self._grasp_candidates = []
        for offset in offsets:
            candidate_xy = [object_xy[0] + float(offset[0]), object_xy[1] + float(offset[1])]
            self._grasp_candidates.append(
                build_pick_place_poses(candidate_xy, region_xy, geometry_config)
            )
        self._grasp_candidate_index = 0
        self._log_dynamic_poses(target_name, region_name, object_xy, region_xy, poses)
        motion_config = self.pick_place_config["motion"]
        transit_velocity = float(motion_config["velocity_scale"])
        transit_acceleration = float(motion_config["acceleration_scale"])
        contact_velocity = float(
            motion_config.get("contact_velocity_scale", transit_velocity)
        )
        contact_acceleration = float(
            motion_config.get("contact_acceleration_scale", transit_acceleration)
        )
        lift_velocity = float(motion_config.get("lift_velocity_scale", transit_velocity))
        lift_acceleration = float(
            motion_config.get("lift_acceleration_scale", transit_acceleration)
        )
        self.task_in_progress = True
        self.current_command_id = command_id
        self._left_target_contact = False
        self._right_target_contact = False
        self._post_grasp_steps = [
            TaskStep(
                "pose", poses["region_approach"], "移动到目标区域上方",
                transit_velocity, transit_acceleration,
            ),
            TaskStep(
                "pose", poses["region_place"], "垂直下降到放置点",
                contact_velocity, contact_acceleration, "linear",
            ),
            TaskStep("gripper", "open", "打开夹爪释放"),
            TaskStep(
                "pose", poses["region_retreat"], "垂直撤离目标区域",
                transit_velocity, transit_acceleration, "linear",
            ),
        ]
        self.pending_steps = [
            TaskStep(
                "scene_add", (target_name, object_xy), "将目标方块加入 MoveIt 碰撞场景"
            ),
            TaskStep("gripper", "open", "打开夹爪"),
            TaskStep(
                "pose", self._grasp_candidates[0]["object_approach"], "移动到目标方块上方",
                transit_velocity, transit_acceleration,
            ),
            TaskStep(
                "scene_remove", target_name, "允许目标方块进入夹持接触区",
            ),
            TaskStep(
                "pose", self._grasp_candidates[0]["object_grasp"], "垂直下降到抓取点",
                contact_velocity, contact_acceleration, "linear",
            ),
            TaskStep("gripper", "close", "关闭夹爪抓取"),
            TaskStep(
                "pose", self._grasp_candidates[0]["object_lift"], "垂直抬升目标方块",
                lift_velocity, lift_acceleration, "linear",
            ),
        ] + self._post_grasp_steps
        self._execute_next_pick_place_step()

    def _queue_pick_place_until_detected(
        self, command_id: str, target_name: str, region_name: str
    ) -> None:
        valid_targets = {"red_cube", "blue_cube"}
        valid_regions = {"red_target_zone", "blue_target_zone"}
        if target_name not in valid_targets or region_name not in valid_regions:
            self.get_logger().error("抓放目标或区域不受支持，未执行任何机械臂动作")
            return
        if self._get_visual_locations(target_name, region_name) is not None:
            self._start_pick_place(command_id, target_name, region_name)
            return
        self.task_in_progress = True
        self.current_command_id = command_id
        self.active_step_label = "等待视觉定位"
        self._pending_pick_place_command = (command_id, target_name, region_name)
        self._pending_detection_deadline = (
            monotonic()
            + float(self.get_parameter("initial_detection_wait_sec").value)
        )
        self.get_logger().info(
            "等待 camera_main 的首个新鲜目标检测后开始抓放（最多 %.1f 秒）"
            % float(self.get_parameter("initial_detection_wait_sec").value)
        )

    def _try_start_pending_pick_place(self) -> None:
        pending = self._pending_pick_place_command
        if pending is None:
            return
        command_id, target_name, region_name = pending
        if self._get_visual_locations(
            target_name, region_name, log_missing=False
        ) is not None:
            self._pending_pick_place_command = None
            self._pending_detection_deadline = None
            # _start_pick_place sets task_in_progress for the physical steps.
            self.task_in_progress = False
            self._start_pick_place(command_id, target_name, region_name)
            return
        if (
            self._pending_detection_deadline is not None
            and monotonic() >= self._pending_detection_deadline
        ):
            self._pending_pick_place_command = None
            self._pending_detection_deadline = None
            self._abort_pick_place(
                "等待 camera_main 新鲜视觉目标超时，未触发机械臂动作"
            )

    def _orientation_for(self, object_name: str) -> List[float]:
        """Return the prevalidated wrist attitude for the requested visual item."""
        candidates = self.pick_place_config.get("orientation_by_object", {})
        value = candidates.get(object_name, self.pick_place_config["orientation_xyzw"])
        return [float(component) for component in value]

    def _get_grasp_center_offset(self) -> List[float]:
        """Read the original gripper geometry from TF, or its declared config."""
        try:
            transform = self.tf_buffer.lookup_transform(
                "end_effector", "grasp_center", Time()
            )
        except Exception as error:
            # The vendor robot description deliberately has no convenience
            # grasp_center link.  Keep that model untouched: this vector is
            # the documented transform derived from its existing fixed joints
            # (end_effector z=-0.074, gripper base z=0.045, jaw band z=-0.2111).
            offset = [
                float(value)
                for value in self.pick_place_config["jaw_center_offset_m"]
            ]
            self.get_logger().info(
                "grasp_center TF 不存在，使用原始夹爪固定几何推导的 "
                f"TCP 偏移 ({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}); "
                f"未修改机器人模型（TF 错误：{error}）"
            )
            return offset
        translation = transform.transform.translation
        offset = [translation.x, translation.y, translation.z]
        self.get_logger().info(
            "TCP->grasp_center (end_effector frame): "
            f"({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f})"
        )
        return offset

    def _get_visual_locations(
        self, target_name: str, region_name: str, log_missing: bool = True
    ) -> Optional[tuple[List[float], List[float]]]:
        is_fresh = (
            self.last_detection_receipt is not None
            and monotonic() - self.last_detection_receipt
            <= float(self.get_parameter("max_detection_age_sec").value)
        )
        target = self.detected_objects.get(target_name) if is_fresh else None
        region = self.detected_objects.get(region_name) if is_fresh else None
        if target is not None and region is not None:
            return (
                [target.pose.pose.position.x, target.pose.pose.position.y],
                [region.pose.pose.position.x, region.pose.pose.position.y],
            )
        if not bool(self.get_parameter("allow_pose_fallback").value):
            missing = [
                name for name, item in ((target_name, target), (region_name, region))
                if item is None
            ]
            if log_missing:
                self.get_logger().error(
                    "缺少新鲜的视觉目标：" + ", ".join(missing)
                )
            return None
        fallbacks = self.pick_place_config["debug_pose_fallbacks"]
        self.get_logger().warning("使用显式调试坐标回退，非正常 Gazebo 验收路径")
        return (list(fallbacks[target_name]), list(fallbacks[region_name]))

    def _log_dynamic_poses(self, target, region, object_xy, region_xy, poses) -> None:
        self.get_logger().info(
            f"视觉定位：{target}=({object_xy[0]:.3f}, {object_xy[1]:.3f}), "
            f"{region}=({region_xy[0]:.3f}, {region_xy[1]:.3f}) base_link"
        )
        for name, pose in poses.items():
            self.get_logger().info(
                f"{name}: ({pose.position[0]:.3f}, {pose.position[1]:.3f}, "
                f"{pose.position[2]:.3f}) base_link"
            )

    def _begin_gazebo_physics_validation(
        self, target_name: str, region_name: str
    ) -> bool:
        """Record the initial dynamic-cube state for a read-only Gazebo check."""
        self._physical_target_name = target_name
        self._physical_region_name = region_name
        self._initial_target_world_position = None
        if not bool(self.get_parameter("gazebo_physics_validation").value):
            return True
        target_position = self.gazebo_model_positions.get(target_name)
        region_position = self.gazebo_model_positions.get(region_name)
        if target_position is None or region_position is None:
            self.get_logger().error(
                "Gazebo 物理验证已启用，但尚未收到目标或区域的 /gazebo/model_states；"
                "未执行机械臂动作"
            )
            return False
        self._initial_target_world_position = list(target_position)
        self.get_logger().info(
            "Gazebo 真值基线（只读）：%s=(%.3f, %.3f, %.3f), %s=(%.3f, %.3f, %.3f)"
            % (
                target_name, *target_position, region_name, *region_position
            )
        )
        return True

    def _verify_gazebo_lift(self) -> bool:
        """Require the dynamic cube to have risen before beginning transport."""
        if not bool(self.get_parameter("gazebo_physics_validation").value):
            return True
        current = self.gazebo_model_positions.get(self._physical_target_name)
        baseline = self._initial_target_world_position
        if current is None or baseline is None:
            self._abort_pick_place("无法读取 Gazebo 方块状态以验证真实抬升")
            return False
        lift_delta = current[2] - baseline[2]
        self.get_logger().info(
            "Gazebo 真实抬升校验（只读）：%s initial_z=%.3f, current=(%.3f, %.3f, %.3f), delta_z=%.3f m"
            % (self._physical_target_name, baseline[2], *current, lift_delta)
        )
        minimum = float(self.get_parameter("minimum_lift_delta_m").value)
        if lift_delta < minimum:
            self.get_logger().warning(
                "方块未随本次候选抬升：delta_z=%.3f m，小于 %.3f m"
                % (lift_delta, minimum)
            )
            return False
        return True

    def _log_gazebo_cube_position(self, stage: str) -> None:
        """Log the dynamic cube around contact without influencing its state."""
        if not bool(self.get_parameter("gazebo_physics_validation").value):
            return
        cube = self.gazebo_model_positions.get(self._physical_target_name)
        if cube is not None:
            self.get_logger().info(
                "Gazebo 接触诊断（只读）：%s, %s=(%.3f, %.3f, %.3f)"
                % (stage, self._physical_target_name, *cube)
            )

    def _verify_gazebo_placement(self) -> bool:
        """Require the released dynamic cube centre to finish in its marker."""
        if not bool(self.get_parameter("gazebo_physics_validation").value):
            return True
        cube = self.gazebo_model_positions.get(self._physical_target_name)
        region = self.gazebo_model_positions.get(self._physical_region_name)
        if cube is None or region is None:
            self._abort_pick_place("无法读取 Gazebo 方块状态以验证真实放置")
            return False
        error_xy = sqrt((cube[0] - region[0]) ** 2 + (cube[1] - region[1]) ** 2)
        self.get_logger().info(
            "Gazebo 真实放置校验（只读）：%s=(%.3f, %.3f, %.3f), %s=(%.3f, %.3f, %.3f), xy_error=%.3f m"
            % (
                self._physical_target_name, *cube,
                self._physical_region_name, *region, error_xy,
            )
        )
        maximum = float(self.get_parameter("maximum_region_center_error_m").value)
        if error_xy > maximum:
            self._abort_pick_place(
                "方块未落入目标区域：中心 XY 误差 %.3f m，大于 %.3f m"
                % (error_xy, maximum)
            )
            return False
        return True

    def _publish_target_collision(
        self, target_name: str, object_xy: List[float]
    ) -> None:
        """Put the vision-derived cube into MoveIt's scene for the approach."""
        cube_height = float(self.pick_place_config["cube_height_m"])
        support_z = float(self.pick_place_config["cube_support_plane_z_m"])
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [cube_height, cube_height, cube_height]
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.position.x, pose.pose.position.y = object_xy
        pose.pose.position.z = support_z + cube_height / 2.0
        pose.pose.orientation.w = 1.0
        collision = CollisionObject()
        collision.id = f"task_{target_name}_obstacle"
        collision.header = pose.header
        collision.operation = CollisionObject.ADD
        collision.primitives = [primitive]
        collision.primitive_poses = [pose.pose]
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [collision]
        self.planning_scene_publisher.publish(scene)
        self._scene_target_id = collision.id
        self.get_logger().info(
            f"MoveIt 避障物已加入：{collision.id} at "
            f"({object_xy[0]:.3f}, {object_xy[1]:.3f}, {pose.pose.position.z:.3f})"
        )

    def _remove_target_collision(self, target_name: str) -> None:
        """Remove only the intended target before its real finger contact."""
        collision = CollisionObject()
        collision.id = self._scene_target_id or f"task_{target_name}_obstacle"
        collision.header.frame_id = "base_link"
        collision.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [collision]
        self.planning_scene_publisher.publish(scene)
        self.get_logger().info(f"MoveIt 避障物已移除以执行真实夹持：{collision.id}")
        self._scene_target_id = ""

    def _continue_after_scene_update(self) -> None:
        """Give MoveIt a brief chance to apply a planning-scene diff."""
        self._scene_timer = self.create_timer(0.35, self._after_scene_update)

    def _after_scene_update(self) -> None:
        if self._scene_timer is not None:
            self._scene_timer.cancel()
            self.destroy_timer(self._scene_timer)
            self._scene_timer = None
        self._execute_next_pick_place_step()

    def _execute_next_pick_place_step(self) -> None:
        if not self.pending_steps:
            self.get_logger().info(f"抓放任务执行成功：{self.current_command_id}")
            self.task_in_progress = False
            return
        step = self.pending_steps.pop(0)
        self.active_step_label = step.label
        self.get_logger().info(f"抓放步骤：{step.label}")
        if step.operation == "gripper":
            if step.value == "close":
                self._log_gazebo_cube_position("下降结束、闭合前")
            self._call_pick_place_gripper(str(step.value))
        elif step.operation == "scene_add":
            target_name, object_xy = step.value
            self._publish_target_collision(target_name, object_xy)
            self._continue_after_scene_update()
        elif step.operation == "scene_remove":
            self._remove_target_collision(str(step.value))
            self._continue_after_scene_update()
        elif step.operation == "pose":
            self._call_pick_place_pose(
                step.value, step.velocity_scale, step.acceleration_scale,
                step.motion_type,
            )
        else:
            self._abort_pick_place(f"不支持的抓放步骤：{step.operation}")

    def _call_pick_place_gripper(self, position_name: str) -> None:
        if not self.gripper_client.wait_for_service(2.0):
            self._abort_pick_place("夹爪服务不可用")
            return
        if position_name == "close":
            # A previous candidate must not satisfy this candidate's check.
            self._left_target_contact = False
            self._right_target_contact = False
        request = SetGripper.Request()
        request.position_name = position_name
        self.gripper_client.call_async(request).add_done_callback(
            self._handle_pick_place_service_result
        )

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
        if self.active_step_label == "关闭夹爪抓取":
            self._log_gazebo_cube_position("闭合结束、抬升前")
            # The trajectory action reports before the next Gazebo contact
            # sample.  Evaluate after a bounded physical settling interval,
            # never immediately from stale/empty sensor state.
            self.get_logger().info("夹爪闭合完成；等待 Gazebo 接触传感器刷新")
            self._grasp_contact_timer = self.create_timer(
                0.50, self._evaluate_grasp_contact
            )
            return
        # Gazebo publishes the final arm/gripper state a few cycles after the
        # gripper trajectory reports success.  Let MoveIt's state monitor see
        # that feedback before planning the following lift/place pose.
        self._settle_timer = self.create_timer(0.75, self._continue_after_gripper)

    def _evaluate_grasp_contact(self) -> None:
        """Allow lift only after real left and right jaw contact was observed."""
        if self._grasp_contact_timer is not None:
            self._grasp_contact_timer.cancel()
            self.destroy_timer(self._grasp_contact_timer)
            self._grasp_contact_timer = None
        self.get_logger().info(
            "夹持接触结果：left=%s right=%s"
            % (self._left_target_contact, self._right_target_contact)
        )
        if not (self._left_target_contact and self._right_target_contact):
            if self._schedule_next_grasp_candidate():
                return
            self._abort_pick_place(
                "夹爪闭合后未检测到目标方块与左右手指的真实 Gazebo 接触；"
                "未执行抬升"
            )
            return
        self._settle_timer = self.create_timer(0.30, self._continue_after_gripper)

    def _schedule_next_grasp_candidate(self) -> bool:
        """Retry a small visual-relative grasp offset before any lift occurs."""
        if self._grasp_candidate_index + 1 >= len(self._grasp_candidates):
            return False
        self._grasp_candidate_index += 1
        candidate = self._grasp_candidates[self._grasp_candidate_index]
        motion = self.pick_place_config["motion"]
        self.get_logger().warning(
            "候选抓取 %d/%d 未形成双侧接触；从安全上方尝试下一个视觉相对候选"
            % (self._grasp_candidate_index, len(self._grasp_candidates))
        )
        self.pending_steps = [
            TaskStep("gripper", "open", "打开夹爪重试"),
            TaskStep(
                "pose", candidate["object_approach"], "移动到候选抓取点上方",
                float(motion.get("contact_velocity_scale", motion["velocity_scale"])),
                float(motion.get("contact_acceleration_scale", motion["acceleration_scale"])),
                "linear",
            ),
            TaskStep(
                "pose", candidate["object_grasp"], "垂直下降到候选抓取点",
                float(motion.get("contact_velocity_scale", motion["velocity_scale"])),
                float(motion.get("contact_acceleration_scale", motion["acceleration_scale"])),
                "linear",
            ),
            TaskStep("gripper", "close", "关闭夹爪抓取"),
            TaskStep(
                "pose", candidate["object_lift"], "垂直抬升目标方块",
                float(motion.get("lift_velocity_scale", motion["velocity_scale"])),
                float(motion.get("lift_acceleration_scale", motion["acceleration_scale"])),
                "linear",
            ),
        ] + self._post_grasp_steps
        self._left_target_contact = False
        self._right_target_contact = False
        self._settle_timer = self.create_timer(0.75, self._continue_after_gripper)
        return True

    def _continue_after_gripper(self) -> None:
        if self._settle_timer is not None:
            self._settle_timer.cancel()
            self.destroy_timer(self._settle_timer)
            self._settle_timer = None
        self._execute_next_pick_place_step()

    def _call_pick_place_pose(
        self,
        pose: PoseSpec,
        velocity_scale: Optional[float] = None,
        acceleration_scale: Optional[float] = None,
        motion_type: str = "pose",
    ) -> None:
        if not self.move_to_pose_client.wait_for_server(timeout_sec=3.0):
            self._abort_pick_place("/motion/move_to_pose 不可用")
            return
        goal = MoveToPose.Goal()
        goal.target_pose = self._pose_stamped(pose)
        goal.motion_type = motion_type
        goal.velocity_scale = float(
            velocity_scale
            if velocity_scale is not None
            else self.pick_place_config["motion"]["velocity_scale"]
        )
        goal.acceleration_scale = float(
            acceleration_scale
            if acceleration_scale is not None
            else self.pick_place_config["motion"]["acceleration_scale"]
        )
        self.move_to_pose_client.send_goal_async(goal).add_done_callback(
            lambda future: self._handle_pick_place_goal(future, pose)
        )

    def _handle_pick_place_goal(self, future: Future, pose: PoseSpec) -> None:
        try:
            handle = future.result()
            if handle is None or not handle.accepted:
                self._abort_pick_place("抓放位姿目标被运动层拒绝")
                return
            handle.get_result_async().add_done_callback(
                lambda result_future: self._handle_pick_place_pose_result(
                    result_future, pose
                )
            )
        except Exception as error:
            self._abort_pick_place(f"发送抓放位姿失败：{error}")

    def _handle_pick_place_pose_result(self, future: Future, pose: PoseSpec) -> None:
        try:
            response = future.result()
            result = response.result if response is not None else None
            if result is None or not result.success:
                self._abort_pick_place(
                    result.message if result is not None else "位姿动作没有返回结果"
                )
                return
        except Exception as error:
            self._abort_pick_place(f"抓放位姿执行异常：{error}")
            return
        if not self._verify_actual_tcp(pose):
            return
        if self.active_step_label == "垂直抬升目标方块":
            if not self._verify_gazebo_lift():
                if self._schedule_next_grasp_candidate():
                    return
                self._abort_pick_place(
                    "所有视觉相对抓取候选均未通过真实抬升校验；"
                    "未进入搬运或放置"
                )
                return
        elif self.active_step_label == "垂直撤离目标区域":
            if not self._verify_gazebo_placement():
                return
        self._execute_next_pick_place_step()

    def _verify_actual_tcp(self, expected_pose: PoseSpec) -> bool:
        """
        Fail closed when the physical Gazebo TCP misses a commanded pose.

        The arm controller has a necessary joint-space tolerance for CAD
        inertia/gravity error.  A task is successful only when TF confirms the
        actual public TCP is close enough to the vision-derived target.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link", "end_effector", Time()
            )
        except Exception as error:
            self._abort_pick_place(f"无法读取执行后的 end_effector TF：{error}")
            return False
        translation = transform.transform.translation
        dx = translation.x - expected_pose.position[0]
        dy = translation.y - expected_pose.position[1]
        dz = translation.z - expected_pose.position[2]
        error_m = sqrt(dx * dx + dy * dy + dz * dz)
        self.get_logger().info(
            "实际 TCP 校验：target=(%.3f, %.3f, %.3f), actual=(%.3f, %.3f, %.3f), "
            "position_error=%.3f m"
            % (
                expected_pose.position[0], expected_pose.position[1],
                expected_pose.position[2], translation.x, translation.y,
                translation.z, error_m,
            )
        )
        maximum = float(self.get_parameter("max_tcp_position_error_m").value)
        if error_m > maximum:
            self._abort_pick_place(
                "实际 TCP 偏差 %.3f m 超过 %.3f m，未继续执行" % (error_m, maximum)
            )
            return False
        return True

    def _abort_pick_place(self, reason: str) -> None:
        self.get_logger().error(
            f"抓放任务失败：{self.current_command_id}; 步骤={self.active_step_label}; {reason}"
        )
        self.pending_steps = []
        self.task_in_progress = False

    @staticmethod
    def _pose_stamped(pose: PoseSpec) -> PoseStamped:
        message = PoseStamped()
        message.header.frame_id = "base_link"
        message.pose.position.x, message.pose.position.y, message.pose.position.z = pose.position
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = pose.orientation_xyzw
        return message


def main(args=None) -> None:
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

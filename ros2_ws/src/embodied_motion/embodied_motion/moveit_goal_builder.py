import copy
import math
from typing import Sequence

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    OrientationConstraint,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive


class MoveItGoalBuilder:
    """构造并校验 EDULITE_A3 的 MoveGroup 规划目标。"""

    def __init__(self, config: dict):
        robot_config = config["robot"]
        moveit_config = config["moveit"]
        motion_config = config["motion"]
        validation_config = config["validation"]

        # 机器人配置
        self.base_frame = robot_config["base_frame"]
        self.planning_group = robot_config["planning_group"]
        self.end_effector_link = robot_config["end_effector_link"]

        # MoveIt2 规划配置
        self.pipeline_id = str(moveit_config["pipeline_id"])
        self.planner_id = str(moveit_config["planner_id"])

        self.allowed_planning_time = float(
            moveit_config["allowed_planning_time"]
        )
        self.num_planning_attempts = int(
            moveit_config["num_planning_attempts"]
        )

        self.replan = bool(moveit_config["replan"])
        self.replan_attempts = int(
            moveit_config["replan_attempts"]
        )
        self.replan_delay = float(
            moveit_config["replan_delay"]
        )

        # 运动容差
        self.joint_tolerance = float(
            motion_config["joint_tolerance"]
        )
        self.position_tolerance = float(
            motion_config["position_tolerance"]
        )
        self.orientation_tolerance = float(
            motion_config["orientation_tolerance"]
        )

        # 速度和加速度限制
        self.min_velocity_scale = float(
            motion_config["min_velocity_scale"]
        )
        self.max_velocity_scale = float(
            motion_config["max_velocity_scale"]
        )
        self.min_acceleration_scale = float(
            motion_config["min_acceleration_scale"]
        )
        self.max_acceleration_scale = float(
            motion_config["max_acceleration_scale"]
        )

        # Pose 输入校验
        self.require_frame_id = bool(
            validation_config["require_frame_id"]
        )

        self.allowed_frames = {
            str(frame_id)
            for frame_id in validation_config["allowed_frames"]
        }

        self.reject_zero_quaternion = bool(
            validation_config["reject_zero_quaternion"]
        )

        self.quaternion_norm_tolerance = float(
            validation_config["quaternion_norm_tolerance"]
        )

    def build_joint_goal(
        self,
        joint_names: Sequence[str],
        positions: Sequence[float],
        velocity_scale: float,
        acceleration_scale: float,
    ) -> MoveGroup.Goal:
        """根据完整关节目标构造 MoveGroup Goal。"""

        if not joint_names:
            raise ValueError("joint_names must not be empty")

        if len(joint_names) != len(positions):
            raise ValueError(
                "Joint target length mismatch: "
                f"{len(joint_names)} names and "
                f"{len(positions)} positions"
            )

        joint_constraints = []

        for joint_name, position in zip(joint_names, positions):
            position = float(position)

            if not math.isfinite(position):
                raise ValueError(
                    f"Non-finite position for joint {joint_name}"
                )

            constraint = JointConstraint()
            constraint.joint_name = str(joint_name)
            constraint.position = position
            constraint.tolerance_above = self.joint_tolerance
            constraint.tolerance_below = self.joint_tolerance
            constraint.weight = 1.0

            joint_constraints.append(constraint)

        goal_constraints = Constraints()
        goal_constraints.name = "joint_target"
        goal_constraints.joint_constraints = joint_constraints

        goal = self._build_base_goal(
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
        )

        goal.request.goal_constraints = [goal_constraints]

        return goal

    def build_pose_goal(
        self,
        target_pose: PoseStamped,
        velocity_scale: float,
        acceleration_scale: float,
    ) -> MoveGroup.Goal:
        """根据末端位姿目标构造 MoveGroup Goal。"""

        target_pose = copy.deepcopy(target_pose)
        frame_id = target_pose.header.frame_id.strip()

        if not frame_id:
            if self.require_frame_id:
                raise ValueError(
                    "target_pose.header.frame_id is empty"
                )

            frame_id = self.base_frame
            target_pose.header.frame_id = frame_id

        if self.allowed_frames and frame_id not in self.allowed_frames:
            raise ValueError(
                f'Unsupported target frame "{frame_id}". '
                f"Allowed frames: {sorted(self.allowed_frames)}"
            )

        self._validate_pose(target_pose)

        goal_constraints = Constraints()
        goal_constraints.name = "end_effector_pose_target"

        # 位置约束
        position_constraint = PositionConstraint()
        position_constraint.header = copy.deepcopy(
            target_pose.header
        )
        position_constraint.link_name = self.end_effector_link
        position_constraint.weight = 1.0

        target_region = BoundingVolume()

        tolerance_sphere = SolidPrimitive()
        tolerance_sphere.type = SolidPrimitive.SPHERE
        tolerance_sphere.dimensions = [
            self.position_tolerance
        ]

        region_pose = Pose()
        region_pose.position = copy.deepcopy(
            target_pose.pose.position
        )
        region_pose.orientation.w = 1.0

        target_region.primitives = [tolerance_sphere]
        target_region.primitive_poses = [region_pose]

        position_constraint.constraint_region = target_region

        # 姿态约束
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header = copy.deepcopy(
            target_pose.header
        )
        orientation_constraint.link_name = self.end_effector_link
        orientation_constraint.orientation = copy.deepcopy(
            target_pose.pose.orientation
        )

        orientation_constraint.absolute_x_axis_tolerance = (
            self.orientation_tolerance
        )
        orientation_constraint.absolute_y_axis_tolerance = (
            self.orientation_tolerance
        )
        orientation_constraint.absolute_z_axis_tolerance = (
            self.orientation_tolerance
        )
        orientation_constraint.weight = 1.0

        goal_constraints.position_constraints = [
            position_constraint
        ]
        goal_constraints.orientation_constraints = [
            orientation_constraint
        ]

        goal = self._build_base_goal(
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
        )

        goal.request.goal_constraints = [goal_constraints]

        return goal

    def _build_base_goal(
        self,
        velocity_scale: float,
        acceleration_scale: float,
    ) -> MoveGroup.Goal:
        """构造关节目标和位姿目标共用的 MoveGroup Goal 字段。"""

        velocity_scale = self._validate_scale(
            name="velocity_scale",
            value=velocity_scale,
            minimum=self.min_velocity_scale,
            maximum=self.max_velocity_scale,
        )

        acceleration_scale = self._validate_scale(
            name="acceleration_scale",
            value=acceleration_scale,
            minimum=self.min_acceleration_scale,
            maximum=self.max_acceleration_scale,
        )

        goal = MoveGroup.Goal()

        goal.request.group_name = self.planning_group
        goal.request.num_planning_attempts = (
            self.num_planning_attempts
        )
        goal.request.allowed_planning_time = (
            self.allowed_planning_time
        )
        goal.request.max_velocity_scaling_factor = (
            velocity_scale
        )
        goal.request.max_acceleration_scaling_factor = (
            acceleration_scale
        )

        if self.pipeline_id:
            goal.request.pipeline_id = self.pipeline_id

        if self.planner_id:
            goal.request.planner_id = self.planner_id

        # 使用 MoveIt2 当前监控到的机器人状态作为规划起点
        goal.request.start_state.is_diff = True

        # 让 MoveGroup 完成规划并直接执行
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = self.replan
        goal.planning_options.replan_attempts = (
            self.replan_attempts
        )
        goal.planning_options.replan_delay = (
            self.replan_delay
        )

        # 使用当前规划场景
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = (
            True
        )

        return goal

    def _validate_pose(
        self,
        target_pose: PoseStamped,
    ) -> None:
        """检查 Pose 中的数值和四元数是否合法。"""

        position = target_pose.pose.position
        orientation = target_pose.pose.orientation

        values = [
            position.x,
            position.y,
            position.z,
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]

        if not all(
            math.isfinite(float(value))
            for value in values
        ):
            raise ValueError(
                "Target pose contains a non-finite number"
            )

        quaternion_norm = math.sqrt(
            float(orientation.x) ** 2
            + float(orientation.y) ** 2
            + float(orientation.z) ** 2
            + float(orientation.w) ** 2
        )

        if (
            self.reject_zero_quaternion
            and quaternion_norm
            <= self.quaternion_norm_tolerance
        ):
            raise ValueError(
                "Target orientation is a zero quaternion"
            )

        norm_error = abs(quaternion_norm - 1.0)

        if norm_error > self.quaternion_norm_tolerance:
            raise ValueError(
                "Target quaternion must be normalized: "
                f"norm={quaternion_norm:.6f}, "
                f"allowed error="
                f"{self.quaternion_norm_tolerance}"
            )

    @staticmethod
    def _validate_scale(
        name: str,
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """检查速度或加速度缩放是否合法。"""

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        if not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be in "
                f"[{minimum}, {maximum}], "
                f"received {value}"
            )

        return value
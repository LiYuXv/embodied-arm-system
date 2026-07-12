import os
import threading
from typing import Optional, Tuple

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from embodied_interfaces.action import MoveToPose
from embodied_interfaces.srv import MoveNamedPose, SetGripper
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MoveItErrorCodes
from rclpy.action import (
    ActionClient,
    ActionServer,
    CancelResponse,
    GoalResponse,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint

from embodied_motion.moveit_goal_builder import MoveItGoalBuilder


class MotionExecutorNode(Node):
    """EDULITE_A3 统一运动规划与执行节点。"""

    MOVEIT_ERROR_NAMES = {
        MoveItErrorCodes.SUCCESS: "SUCCESS",
        MoveItErrorCodes.FAILURE: "FAILURE",
        MoveItErrorCodes.PLANNING_FAILED: "PLANNING_FAILED",
        MoveItErrorCodes.INVALID_MOTION_PLAN: "INVALID_MOTION_PLAN",
        MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE:
            "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
        MoveItErrorCodes.CONTROL_FAILED: "CONTROL_FAILED",
        MoveItErrorCodes.TIMED_OUT: "TIMED_OUT",
        MoveItErrorCodes.PREEMPTED: "PREEMPTED",
        MoveItErrorCodes.START_STATE_IN_COLLISION:
            "START_STATE_IN_COLLISION",
        MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS:
            "START_STATE_VIOLATES_PATH_CONSTRAINTS",
        MoveItErrorCodes.GOAL_IN_COLLISION: "GOAL_IN_COLLISION",
        MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS:
            "GOAL_VIOLATES_PATH_CONSTRAINTS",
        MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED:
            "GOAL_CONSTRAINTS_VIOLATED",
        MoveItErrorCodes.INVALID_GROUP_NAME: "INVALID_GROUP_NAME",
        MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS:
            "INVALID_GOAL_CONSTRAINTS",
        MoveItErrorCodes.INVALID_ROBOT_STATE: "INVALID_ROBOT_STATE",
        MoveItErrorCodes.INVALID_LINK_NAME: "INVALID_LINK_NAME",
        MoveItErrorCodes.FRAME_TRANSFORM_FAILURE:
            "FRAME_TRANSFORM_FAILURE",
        MoveItErrorCodes.ROBOT_STATE_STALE: "ROBOT_STATE_STALE",
        MoveItErrorCodes.COMMUNICATION_FAILURE:
            "COMMUNICATION_FAILURE",
        MoveItErrorCodes.NO_IK_SOLUTION: "NO_IK_SOLUTION",
        MoveItErrorCodes.ABORT: "ABORT",
    }

    GOAL_STATUS_NAMES = {
        GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
        GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
        GoalStatus.STATUS_EXECUTING: "EXECUTING",
        GoalStatus.STATUS_CANCELING: "CANCELING",
        GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
        GoalStatus.STATUS_CANCELED: "CANCELED",
        GoalStatus.STATUS_ABORTED: "ABORTED",
    }

    GRIPPER_ERROR_NAMES = {
        FollowJointTrajectory.Result.SUCCESSFUL: "SUCCESSFUL",
        FollowJointTrajectory.Result.INVALID_GOAL: "INVALID_GOAL",
        FollowJointTrajectory.Result.INVALID_JOINTS: "INVALID_JOINTS",
        FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP:
            "OLD_HEADER_TIMESTAMP",
        FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
            "PATH_TOLERANCE_VIOLATED",
        FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED:
            "GOAL_TOLERANCE_VIOLATED",
    }

    SUPPORTED_MOTION_TYPES = {
        "",
        "pose",
        "ptp",
    }

    def __init__(self) -> None:
        super().__init__("motion_executor_node")

        self.callback_group = ReentrantCallbackGroup()

        self.config = self._load_motion_config()
        self._load_runtime_config()

        self.goal_builder = MoveItGoalBuilder(self.config)

        self._state_lock = threading.Lock()
        self._motion_busy = False
        self._active_motion_source = ""
        self._active_move_group_goal_handle = None

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            self.move_group_action_name,
            callback_group=self.callback_group,
        )

        self.gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.gripper_action_name,
            callback_group=self.callback_group,
        )

        self.go_named_pose_service = self.create_service(
            MoveNamedPose,
            "/motion/go_named_pose",
            self._handle_go_named_pose,
            callback_group=self.callback_group,
        )

        self.set_gripper_service = self.create_service(
            SetGripper,
            "/motion/set_gripper",
            self._handle_set_gripper,
            callback_group=self.callback_group,
        )

        self.move_to_pose_server = ActionServer(
            self,
            MoveToPose,
            "/motion/move_to_pose",
            execute_callback=self._execute_move_to_pose,
            goal_callback=self._handle_move_to_pose_goal,
            cancel_callback=self._handle_move_to_pose_cancel,
            callback_group=self.callback_group,
        )

        self._print_config_summary()
        self._check_move_group_server()
        self._check_gripper_server()

        self.get_logger().info(
            "Service ready: /motion/go_named_pose"
        )
        self.get_logger().info(
            "Service ready: /motion/set_gripper"
        )
        self.get_logger().info(
            "Action ready: /motion/move_to_pose"
        )

    def _load_motion_config(self) -> dict:
        package_share_dir = get_package_share_directory(
            "embodied_motion"
        )

        config_path = os.path.join(
            package_share_dir,
            "config",
            "motion_config.yaml",
        )

        self.get_logger().info(
            f"Loading motion config from: {config_path}"
        )

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Motion config file not found: {config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(
                "motion_config.yaml must contain a mapping"
            )

        return config

    def _load_runtime_config(self) -> None:
        robot_config = self.config["robot"]
        moveit_config = self.config["moveit"]
        controllers_config = self.config["controllers"]
        motion_config = self.config["motion"]

        self.robot_name = str(robot_config["name"])
        self.base_frame = str(robot_config["base_frame"])
        self.planning_group = str(
            robot_config["planning_group"]
        )
        self.end_effector_link = str(
            robot_config["end_effector_link"]
        )

        self.move_group_action_name = str(
            moveit_config["move_group_action"]
        )

        self.arm_joint_names = [
            str(joint_name)
            for joint_name
            in controllers_config["arm"]["joint_names"]
        ]

        gripper_config = controllers_config["gripper"]

        self.gripper_action_name = str(
            gripper_config["action_name"]
        )
        self.gripper_joint_names = [
            str(joint_name)
            for joint_name
            in gripper_config["joint_names"]
        ]

        self.named_poses = self.config["named_poses"]
        self.gripper_positions = self.config[
            "gripper_positions"
        ]

        self.default_velocity_scale = float(
            motion_config["default_velocity_scale"]
        )
        self.default_acceleration_scale = float(
            motion_config["default_acceleration_scale"]
        )

        self.action_server_timeout_sec = float(
            motion_config["action_server_timeout_sec"]
        )

    def _print_config_summary(self) -> None:
        self.get_logger().info(
            "========== Motion Executor Config =========="
        )
        self.get_logger().info(
            f"Robot: {self.robot_name}"
        )
        self.get_logger().info(
            f"Base frame: {self.base_frame}"
        )
        self.get_logger().info(
            f"Planning group: {self.planning_group}"
        )
        self.get_logger().info(
            f"End-effector link: {self.end_effector_link}"
        )
        self.get_logger().info(
            f"MoveGroup action: {self.move_group_action_name}"
        )
        self.get_logger().info(
            f"Gripper action: {self.gripper_action_name}"
        )
        self.get_logger().info(
            f"Arm joints: {self.arm_joint_names}"
        )
        self.get_logger().info(
            f"Gripper joints: {self.gripper_joint_names}"
        )
        self.get_logger().info(
            f"Named poses: {list(self.named_poses.keys())}"
        )
        self.get_logger().info(
            "Gripper positions: "
            f"{list(self.gripper_positions.keys())}"
        )
        self.get_logger().info(
            "============================================"
        )

    def _check_move_group_server(self) -> None:
        self.get_logger().info(
            f"Checking MoveGroup server: "
            f"{self.move_group_action_name}"
        )

        ready = self.move_group_client.wait_for_server(
            timeout_sec=self.action_server_timeout_sec
        )

        if ready:
            self.get_logger().info(
                f"MoveGroup server connected: "
                f"{self.move_group_action_name}"
            )
        else:
            self.get_logger().warning(
                f"MoveGroup server is not currently available: "
                f"{self.move_group_action_name}"
            )

    def _check_gripper_server(self) -> None:
        self.get_logger().info(
            f"Checking gripper server: "
            f"{self.gripper_action_name}"
        )

        ready = self.gripper_client.wait_for_server(
            timeout_sec=self.action_server_timeout_sec
        )

        if ready:
            self.get_logger().info(
                f"Gripper server connected: "
                f"{self.gripper_action_name}"
            )
        else:
            self.get_logger().warning(
                f"Gripper server is not currently available: "
                f"{self.gripper_action_name}"
            )

    def _reserve_motion(self, source: str) -> bool:
        with self._state_lock:
            if self._motion_busy:
                return False

            self._motion_busy = True
            self._active_motion_source = source

        self.get_logger().info(
            f"Motion slot reserved by: {source}"
        )
        return True

    def _release_motion(self) -> None:
        with self._state_lock:
            source = self._active_motion_source
            self._motion_busy = False
            self._active_motion_source = ""
            self._active_move_group_goal_handle = None

        self.get_logger().info(
            f"Motion slot released: {source}"
        )

    def _get_busy_source(self) -> str:
        with self._state_lock:
            return self._active_motion_source

    def _set_active_move_group_goal_handle(
        self,
        goal_handle,
    ) -> None:
        with self._state_lock:
            self._active_move_group_goal_handle = goal_handle

    def _get_active_move_group_goal_handle(self):
        with self._state_lock:
            return self._active_move_group_goal_handle

    def _resolve_scales(
        self,
        velocity_scale: float,
        acceleration_scale: float,
    ) -> Tuple[float, float]:
        velocity_scale = float(velocity_scale)
        acceleration_scale = float(acceleration_scale)

        if velocity_scale == 0.0:
            velocity_scale = self.default_velocity_scale

        if acceleration_scale == 0.0:
            acceleration_scale = (
                self.default_acceleration_scale
            )

        return velocity_scale, acceleration_scale

    async def _handle_go_named_pose(
        self,
        request: MoveNamedPose.Request,
        response: MoveNamedPose.Response,
    ) -> MoveNamedPose.Response:
        pose_name = request.pose_name.strip()

        self.get_logger().info(
            f"Received named-pose request: {pose_name}"
        )

        if not pose_name:
            response.success = False
            response.message = "pose_name is empty"
            return response

        if pose_name not in self.named_poses:
            response.success = False
            response.message = (
                f'Unknown named pose "{pose_name}". '
                f"Available poses: "
                f"{list(self.named_poses.keys())}"
            )
            return response

        pose_config = self.named_poses[pose_name]

        if pose_config.get("group", "arm") != "arm":
            response.success = False
            response.message = (
                f'Named pose "{pose_name}" does not belong '
                f'to planning group "arm"'
            )
            return response

        source = f"named_pose:{pose_name}"

        if not self._reserve_motion(source):
            response.success = False
            response.message = (
                "Motion executor is busy with: "
                f"{self._get_busy_source()}"
            )
            return response

        try:
            positions = pose_config["positions"]

            move_group_goal = self.goal_builder.build_joint_goal(
                joint_names=self.arm_joint_names,
                positions=positions,
                velocity_scale=self.default_velocity_scale,
                acceleration_scale=(
                    self.default_acceleration_scale
                ),
            )

            success, message, _, _ = (
                await self._send_move_group_goal(
                    move_group_goal
                )
            )

            response.success = success
            response.message = (
                f'Named pose "{pose_name}": {message}'
            )

            return response

        except (KeyError, TypeError, ValueError) as error:
            response.success = False
            response.message = (
                f'Invalid named pose "{pose_name}": {error}'
            )
            self.get_logger().error(response.message)
            return response

        except Exception as error:
            response.success = False
            response.message = (
                f"Unexpected named-pose error: {error}"
            )
            self.get_logger().exception(response.message)
            return response

        finally:
            self._release_motion()

    async def _handle_set_gripper(
        self,
        request: SetGripper.Request,
        response: SetGripper.Response,
    ) -> SetGripper.Response:
        position_name = request.position_name.strip().lower()

        self.get_logger().info(
            f"Received gripper request: {position_name}"
        )

        if not position_name:
            response.success = False
            response.message = "position_name is empty"
            return response

        if position_name not in self.gripper_positions:
            response.success = False
            response.message = (
                f'Unknown gripper position "{position_name}". '
                f"Available positions: "
                f"{list(self.gripper_positions.keys())}"
            )
            return response

        if len(self.gripper_joint_names) != 1:
            response.success = False
            response.message = (
                "Gripper controller must contain exactly "
                "one commanded joint"
            )
            return response

        source = f"gripper:{position_name}"

        if not self._reserve_motion(source):
            response.success = False
            response.message = (
                "Motion executor is busy with: "
                f"{self._get_busy_source()}"
            )
            return response

        try:
            position_config = self.gripper_positions[
                position_name
            ]

            position = float(position_config["position"])
            duration_sec = float(
                position_config["duration_sec"]
            )

            success, message = await self._send_gripper_goal(
                position=position,
                duration_sec=duration_sec,
            )

            response.success = success
            response.message = (
                f'Gripper "{position_name}": {message}'
            )
            return response

        except (KeyError, TypeError, ValueError) as error:
            response.success = False
            response.message = (
                f'Invalid gripper position '
                f'"{position_name}": {error}'
            )
            self.get_logger().error(response.message)
            return response

        except Exception as error:
            response.success = False
            response.message = (
                f"Unexpected gripper error: {error}"
            )
            self.get_logger().exception(response.message)
            return response

        finally:
            self._release_motion()

    def _handle_move_to_pose_goal(
        self,
        goal_request: MoveToPose.Goal,
    ) -> GoalResponse:
        motion_type = (
            goal_request.motion_type.strip().lower()
        )

        if motion_type not in self.SUPPORTED_MOTION_TYPES:
            self.get_logger().warning(
                f'Unsupported motion_type "{motion_type}". '
                f"Supported values: pose, ptp"
            )
            return GoalResponse.REJECT

        try:
            velocity_scale, acceleration_scale = (
                self._resolve_scales(
                    goal_request.velocity_scale,
                    goal_request.acceleration_scale,
                )
            )

            self.goal_builder.build_pose_goal(
                target_pose=goal_request.target_pose,
                velocity_scale=velocity_scale,
                acceleration_scale=acceleration_scale,
            )

        except (TypeError, ValueError) as error:
            self.get_logger().warning(
                f"Rejected move-to-pose goal: {error}"
            )
            return GoalResponse.REJECT

        if not self._reserve_motion("move_to_pose"):
            self.get_logger().warning(
                "Rejected move-to-pose goal because the "
                "executor is busy with: "
                f"{self._get_busy_source()}"
            )
            return GoalResponse.REJECT

        self.get_logger().info(
            "Accepted move-to-pose goal"
        )
        return GoalResponse.ACCEPT

    def _handle_move_to_pose_cancel(
        self,
        goal_handle,
    ) -> CancelResponse:
        self.get_logger().info(
            "Received cancellation request for move-to-pose"
        )

        internal_goal_handle = (
            self._get_active_move_group_goal_handle()
        )

        if internal_goal_handle is not None:
            internal_goal_handle.cancel_goal_async()

        return CancelResponse.ACCEPT

    async def _execute_move_to_pose(
        self,
        goal_handle,
    ) -> MoveToPose.Result:
        result = MoveToPose.Result()

        try:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = (
                    "Move-to-pose canceled before execution"
                )
                return result

            self._publish_move_to_pose_feedback(
                goal_handle,
                current_step="validating_target",
                progress=0.05,
            )

            request = goal_handle.request

            velocity_scale, acceleration_scale = (
                self._resolve_scales(
                    request.velocity_scale,
                    request.acceleration_scale,
                )
            )

            move_group_goal = (
                self.goal_builder.build_pose_goal(
                    target_pose=request.target_pose,
                    velocity_scale=velocity_scale,
                    acceleration_scale=acceleration_scale,
                )
            )

            self._publish_move_to_pose_feedback(
                goal_handle,
                current_step="submitting_to_move_group",
                progress=0.15,
            )

            def feedback_callback(feedback_message) -> None:
                state = feedback_message.feedback.state
                progress = self._progress_from_move_group_state(
                    state
                )

                self._publish_move_to_pose_feedback(
                    goal_handle,
                    current_step=state or "move_group_running",
                    progress=progress,
                )

            success, message, action_status, _ = (
                await self._send_move_group_goal(
                    move_group_goal,
                    feedback_callback=feedback_callback,
                )
            )

            if (
                goal_handle.is_cancel_requested
                or action_status
                == GoalStatus.STATUS_CANCELED
            ):
                goal_handle.canceled()
                result.success = False
                result.message = (
                    f"Move-to-pose canceled: {message}"
                )
                return result

            if success:
                self._publish_move_to_pose_feedback(
                    goal_handle,
                    current_step="completed",
                    progress=1.0,
                )

                goal_handle.succeed()
                result.success = True
                result.message = message
                return result

            goal_handle.abort()
            result.success = False
            result.message = message
            return result

        except (KeyError, TypeError, ValueError) as error:
            self.get_logger().error(
                f"Invalid move-to-pose request: {error}"
            )
            goal_handle.abort()
            result.success = False
            result.message = (
                f"Invalid move-to-pose request: {error}"
            )
            return result

        except Exception as error:
            self.get_logger().exception(
                f"Unexpected move-to-pose error: {error}"
            )
            goal_handle.abort()
            result.success = False
            result.message = (
                f"Unexpected move-to-pose error: {error}"
            )
            return result

        finally:
            self._release_motion()

    async def _send_gripper_goal(
        self,
        position: float,
        duration_sec: float,
    ) -> Tuple[bool, str]:
        if duration_sec <= 0.0:
            raise ValueError(
                "gripper duration_sec must be greater than zero"
            )

        if not self.gripper_client.server_is_ready():
            ready = self.gripper_client.wait_for_server(
                timeout_sec=self.action_server_timeout_sec
            )

            if not ready:
                message = (
                    "Gripper action server is unavailable: "
                    f"{self.gripper_action_name}"
                )
                self.get_logger().error(message)
                return False, message

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(
            self.gripper_joint_names
        )

        point = JointTrajectoryPoint()
        point.positions = [float(position)]

        seconds = int(duration_sec)
        nanoseconds = int(
            round((duration_sec - seconds) * 1_000_000_000)
        )

        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds -= 1_000_000_000

        point.time_from_start.sec = seconds
        point.time_from_start.nanosec = nanoseconds

        goal.trajectory.points = [point]

        self.get_logger().info(
            "Sending goal to gripper controller: "
            f"joint={self.gripper_joint_names[0]}, "
            f"position={position:.4f}"
        )

        goal_handle = await self.gripper_client.send_goal_async(
            goal
        )

        if not goal_handle.accepted:
            message = "Gripper controller rejected the goal"
            self.get_logger().error(message)
            return False, message

        self.get_logger().info(
            "Gripper controller accepted the goal"
        )

        result_response = await goal_handle.get_result_async()

        action_status = result_response.status
        result = result_response.result
        error_code = int(result.error_code)

        status_name = self.GOAL_STATUS_NAMES.get(
            action_status,
            str(action_status),
        )
        error_name = self.GRIPPER_ERROR_NAMES.get(
            error_code,
            f"UNKNOWN_ERROR_{error_code}",
        )

        success = (
            action_status == GoalStatus.STATUS_SUCCEEDED
            and error_code
            == FollowJointTrajectory.Result.SUCCESSFUL
        )

        message = (
            f"status={status_name}, "
            f"controller_error={error_name}"
        )

        error_string = str(result.error_string).strip()
        if error_string:
            message += f", detail={error_string}"

        if success:
            self.get_logger().info(
                f"Gripper motion succeeded: {message}"
            )
        else:
            self.get_logger().error(
                f"Gripper motion failed: {message}"
            )

        return success, message

    async def _send_move_group_goal(
        self,
        goal: MoveGroup.Goal,
        feedback_callback=None,
    ) -> Tuple[bool, str, int, int]:
        if not self.move_group_client.server_is_ready():
            ready = self.move_group_client.wait_for_server(
                timeout_sec=self.action_server_timeout_sec
            )

            if not ready:
                message = (
                    "MoveGroup action server is unavailable: "
                    f"{self.move_group_action_name}"
                )
                self.get_logger().error(message)

                return (
                    False,
                    message,
                    GoalStatus.STATUS_UNKNOWN,
                    MoveItErrorCodes.COMMUNICATION_FAILURE,
                )

        self.get_logger().info(
            "Sending goal to MoveGroup"
        )

        send_goal_future = (
            self.move_group_client.send_goal_async(
                goal,
                feedback_callback=feedback_callback,
            )
        )

        internal_goal_handle = await send_goal_future

        if not internal_goal_handle.accepted:
            message = "MoveGroup rejected the goal"
            self.get_logger().error(message)

            return (
                False,
                message,
                GoalStatus.STATUS_ABORTED,
                MoveItErrorCodes.FAILURE,
            )

        self._set_active_move_group_goal_handle(
            internal_goal_handle
        )

        self.get_logger().info(
            "MoveGroup accepted the goal"
        )

        try:
            result_response = await (
                internal_goal_handle.get_result_async()
            )
        finally:
            self._set_active_move_group_goal_handle(None)

        action_status = result_response.status
        move_group_result = result_response.result
        error_code = int(move_group_result.error_code.val)

        action_status_name = self.GOAL_STATUS_NAMES.get(
            action_status,
            str(action_status),
        )

        error_name = self.MOVEIT_ERROR_NAMES.get(
            error_code,
            f"UNKNOWN_ERROR_{error_code}",
        )

        planning_time = float(
            move_group_result.planning_time
        )

        success = (
            action_status == GoalStatus.STATUS_SUCCEEDED
            and error_code == MoveItErrorCodes.SUCCESS
        )

        message = (
            f"status={action_status_name}, "
            f"moveit_error={error_name}, "
            f"planning_time={planning_time:.3f}s"
        )

        if success:
            self.get_logger().info(
                f"Motion succeeded: {message}"
            )
        else:
            self.get_logger().error(
                f"Motion failed: {message}"
            )

        return (
            success,
            message,
            action_status,
            error_code,
        )

    @staticmethod
    def _publish_move_to_pose_feedback(
        goal_handle,
        current_step: str,
        progress: float,
    ) -> None:
        feedback = MoveToPose.Feedback()
        feedback.current_step = str(current_step)
        feedback.progress = float(
            max(0.0, min(1.0, progress))
        )

        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _progress_from_move_group_state(
        state: Optional[str],
    ) -> float:
        normalized_state = (state or "").strip().lower()

        if "plan" in normalized_state:
            return 0.35

        if (
            "execute" in normalized_state
            or "monitor" in normalized_state
        ):
            return 0.75

        if "idle" in normalized_state:
            return 0.90

        return 0.50


def main(args=None) -> None:
    rclpy.init(args=args)

    node = MotionExecutorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
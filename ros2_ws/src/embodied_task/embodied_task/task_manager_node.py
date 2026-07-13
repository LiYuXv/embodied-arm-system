"""机械臂具身操作系统任务调度节点."""

from typing import Optional

import rclpy
from embodied_interfaces.msg import TaskCommand
from embodied_interfaces.srv import MoveNamedPose, SetGripper
from rclpy.node import Node
from rclpy.task import Future


class TaskManagerNode(Node):
    """接收结构化任务命令并调用运动层完成任务."""

    def __init__(self) -> None:
        super().__init__("task_manager_node")

        self.task_subscription = self.create_subscription(
            TaskCommand,
            "/task_command",
            self._handle_task_command,
            10,
        )

        self.named_pose_client = self.create_client(
            MoveNamedPose,
            "/motion/go_named_pose",
        )

        self.gripper_client = self.create_client(
            SetGripper,
            "/motion/set_gripper",
        )

        self.task_in_progress = False

        self.get_logger().info("Task manager node started")
        self.get_logger().info(
            "Subscribing to task commands on: /task_command"
        )
        self.get_logger().info(
            "Using named-pose service: /motion/go_named_pose"
        )
        self.get_logger().info(
            "Using gripper service: /motion/set_gripper"
        )

    def _handle_task_command(
        self,
        message: TaskCommand,
    ) -> None:
        """接收并分发任务命令."""
        self.get_logger().info(
            "收到任务命令："
            f"command_id={message.command_id}, "
            f"action={message.action}, "
            f"target={message.target}, "
            f"source={message.source}"
        )

        if not message.action:
            self.get_logger().warning(
                "任务命令 action 为空，已忽略"
            )
            return

        if self.task_in_progress:
            self.get_logger().warning(
                "当前已有任务正在执行，拒绝新的任务命令"
            )
            return

        if message.action == "go_named_pose":
            self._execute_named_pose(
                command_id=message.command_id,
                pose_name=message.target,
            )
            return

        if message.action == "set_gripper":
            self._execute_gripper(
                command_id=message.command_id,
                position_name=message.target,
            )
            return

        self.get_logger().warning(
            f"不支持的任务动作：{message.action}"
        )

    def _execute_named_pose(
        self,
        command_id: str,
        pose_name: str,
    ) -> None:
        """调用运动层命名位姿服务."""
        if not pose_name:
            self.get_logger().warning(
                "命名位姿任务 target 为空，已忽略"
            )
            return

        if not self.named_pose_client.service_is_ready():
            self.get_logger().warning(
                "等待服务 /motion/go_named_pose"
            )

            service_ready = (
                self.named_pose_client.wait_for_service(
                    timeout_sec=2.0
                )
            )

            if not service_ready:
                self.get_logger().error(
                    "命名位姿服务不可用，任务执行失败"
                )
                return

        request = MoveNamedPose.Request()
        request.pose_name = pose_name

        self.task_in_progress = True

        self.get_logger().info(
            f"开始执行命名位姿任务：{pose_name}"
        )

        future = self.named_pose_client.call_async(request)

        future.add_done_callback(
            lambda completed_future: (
                self._handle_named_pose_response(
                    command_id=command_id,
                    pose_name=pose_name,
                    future=completed_future,
                )
            )
        )

    def _handle_named_pose_response(
        self,
        command_id: str,
        pose_name: str,
        future: Future,
    ) -> None:
        """处理命名位姿任务结果."""
        try:
            response: Optional[
                MoveNamedPose.Response
            ] = future.result()
        except Exception as error:
            self.get_logger().error(
                f"命名位姿服务调用异常：{error}"
            )
            self.task_in_progress = False
            return

        if response is None:
            self.get_logger().error(
                "命名位姿服务没有返回结果"
            )
            self.task_in_progress = False
            return

        if response.success:
            self.get_logger().info(
                "任务执行成功："
                f"command_id={command_id}, "
                f"pose={pose_name}"
            )
            self.get_logger().info(
                f"运动层返回：{response.message}"
            )
        else:
            self.get_logger().error(
                "任务执行失败："
                f"command_id={command_id}, "
                f"pose={pose_name}"
            )
            self.get_logger().error(
                f"运动层返回：{response.message}"
            )

        self.task_in_progress = False

    def _execute_gripper(
        self,
        command_id: str,
        position_name: str,
    ) -> None:
        """调用运动层夹爪服务."""
        if not position_name:
            self.get_logger().warning(
                "夹爪任务 target 为空，已忽略"
            )
            return

        if not self.gripper_client.service_is_ready():
            self.get_logger().warning(
                "等待服务 /motion/set_gripper"
            )

            service_ready = (
                self.gripper_client.wait_for_service(
                    timeout_sec=2.0
                )
            )

            if not service_ready:
                self.get_logger().error(
                    "夹爪服务不可用，任务执行失败"
                )
                return

        request = SetGripper.Request()
        request.position_name = position_name

        self.task_in_progress = True

        self.get_logger().info(
            f"开始执行夹爪任务：{position_name}"
        )

        future = self.gripper_client.call_async(request)

        future.add_done_callback(
            lambda completed_future: (
                self._handle_gripper_response(
                    command_id=command_id,
                    position_name=position_name,
                    future=completed_future,
                )
            )
        )

    def _handle_gripper_response(
        self,
        command_id: str,
        position_name: str,
        future: Future,
    ) -> None:
        """处理夹爪任务结果."""
        try:
            response: Optional[
                SetGripper.Response
            ] = future.result()
        except Exception as error:
            self.get_logger().error(
                f"夹爪服务调用异常：{error}"
            )
            self.task_in_progress = False
            return

        if response is None:
            self.get_logger().error(
                "夹爪服务没有返回结果"
            )
            self.task_in_progress = False
            return

        if response.success:
            self.get_logger().info(
                "任务执行成功："
                f"command_id={command_id}, "
                f"gripper={position_name}"
            )
            self.get_logger().info(
                f"运动层返回：{response.message}"
            )
        else:
            self.get_logger().error(
                "任务执行失败："
                f"command_id={command_id}, "
                f"gripper={position_name}"
            )
            self.get_logger().error(
                f"运动层返回：{response.message}"
            )

        self.task_in_progress = False


def main(args=None) -> None:
    """启动任务管理节点."""
    rclpy.init(args=args)

    node = TaskManagerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            "收到 Ctrl+C，正在关闭 task_manager_node"
        )
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

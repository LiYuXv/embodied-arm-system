"""终端中文指令输入与机械臂运动服务调用节点。"""

import queue
import threading
import uuid
from typing import Optional

import rclpy
from embodied_interfaces.msg import TaskCommand
from embodied_interfaces.srv import MoveNamedPose, SetGripper
from rclpy.node import Node
from rclpy.task import Future

from embodied_language.command_parser import CommandParser


class LanguageNode(Node):
    """接收终端中文指令并转换为机械臂任务命令。"""

    EXIT_COMMANDS = {"exit", "quit"}

    def __init__(self) -> None:
        super().__init__("language_node")

        self.parser = CommandParser()

        self.command_publisher = self.create_publisher(
            TaskCommand,
            "/task_command",
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

        self.input_queue: queue.Queue[str] = queue.Queue()
        self.running = True

        self.input_thread = threading.Thread(
            target=self._terminal_input_loop,
            daemon=True,
        )
        self.input_thread.start()

        self.input_timer = self.create_timer(
            0.1,
            self._process_input_queue,
        )

        self.get_logger().info("Language node started")
        self.get_logger().info(
            "Publishing task commands on: /task_command"
        )
        self.get_logger().info(
            "Using named-pose service: "
            "/motion/go_named_pose"
        )
        self.get_logger().info(
            "Using gripper service: /motion/set_gripper"
        )
        self.get_logger().info(
            "请输入中文指令；输入 exit 或 quit 退出"
        )

    def _terminal_input_loop(self) -> None:
        """在独立线程中读取终端输入，避免阻塞 ROS 回调。"""
        while self.running:
            try:
                text = input("\n请输入指令：")
            except EOFError:
                self.input_queue.put("exit")
                return
            except KeyboardInterrupt:
                return

            self.input_queue.put(text)

            if text.strip().lower() in self.EXIT_COMMANDS:
                return

    def _process_input_queue(self) -> None:
        """在 ROS 回调线程中处理终端输入。"""
        try:
            text = self.input_queue.get_nowait()
        except queue.Empty:
            return

        normalized_text = text.strip()

        if normalized_text.lower() in self.EXIT_COMMANDS:
            self.get_logger().info(
                "收到退出指令，正在关闭 language_node"
            )
            self.running = False
            return

        parsed_command = self.parser.parse(text)

        if parsed_command is None:
            self.get_logger().warning(
                f"无法识别指令：{text!r}"
            )
            self.get_logger().info(
                "当前支持：回家、观察位置、准备位置、"
                "准备抓取、打开夹爪、关闭夹爪"
            )
            return

        task_command = self._build_task_command(
            raw_text=parsed_command.raw_text,
            action=parsed_command.action,
            target=parsed_command.target,
        )

        self.command_publisher.publish(task_command)

        self.get_logger().info(
            "指令解析成功："
            f"action={task_command.action}, "
            f"target={task_command.target}, "
            f"command_id={task_command.command_id}"
        )

        if task_command.action == "go_named_pose":
            self._call_named_pose_service(
                task_command.target
            )
            return

        if task_command.action == "set_gripper":
            self._call_gripper_service(
                task_command.target
            )
            return

        self.get_logger().warning(
            f"当前不支持执行动作：{task_command.action}"
        )

    def _build_task_command(
        self,
        raw_text: str,
        action: str,
        target: str,
    ) -> TaskCommand:
        """构造统一任务命令消息。"""
        message = TaskCommand()

        message.header.stamp = (
            self.get_clock().now().to_msg()
        )
        message.header.frame_id = ""

        message.command_id = str(uuid.uuid4())
        message.raw_text = raw_text

        message.action = action
        message.target = target
        message.target_region = ""

        message.source = "terminal_rules"

        return message

    def _call_named_pose_service(
        self,
        pose_name: str,
    ) -> None:
        """调用运动层命名位姿服务。"""
        if not self.named_pose_client.service_is_ready():
            self.get_logger().warning(
                "命名位姿服务尚未就绪，等待 "
                "/motion/go_named_pose"
            )

            service_ready = (
                self.named_pose_client.wait_for_service(
                    timeout_sec=2.0
                )
            )

            if not service_ready:
                self.get_logger().error(
                    "命名位姿服务不可用，无法执行指令"
                )
                return

        request = MoveNamedPose.Request()
        request.pose_name = pose_name

        self.get_logger().info(
            f"正在请求运动到命名位姿：{pose_name}"
        )

        future = self.named_pose_client.call_async(
            request
        )
        future.add_done_callback(
            lambda completed_future: (
                self._handle_named_pose_response(
                    pose_name,
                    completed_future,
                )
            )
        )

    def _handle_named_pose_response(
        self,
        pose_name: str,
        future: Future,
    ) -> None:
        """处理命名位姿服务返回结果。"""
        try:
            response: Optional[
                MoveNamedPose.Response
            ] = future.result()
        except Exception as error:
            self.get_logger().error(
                f"调用命名位姿服务失败：{error}"
            )
            return

        if response is None:
            self.get_logger().error(
                "命名位姿服务没有返回结果"
            )
            return

        if response.success:
            self.get_logger().info(
                f"指令执行成功：{pose_name}"
            )
            self.get_logger().info(
                f"运动层返回：{response.message}"
            )
            return

        self.get_logger().error(
            f"指令执行失败：{pose_name}"
        )
        self.get_logger().error(
            f"运动层返回：{response.message}"
        )

    def _call_gripper_service(
        self,
        position_name: str,
    ) -> None:
        """调用运动层夹爪控制服务。"""
        if not self.gripper_client.service_is_ready():
            self.get_logger().warning(
                "夹爪服务尚未就绪，等待 "
                "/motion/set_gripper"
            )

            service_ready = (
                self.gripper_client.wait_for_service(
                    timeout_sec=2.0
                )
            )

            if not service_ready:
                self.get_logger().error(
                    "夹爪服务不可用，无法执行指令"
                )
                return

        request = SetGripper.Request()
        request.position_name = position_name

        self.get_logger().info(
            f"正在请求设置夹爪状态：{position_name}"
        )

        future = self.gripper_client.call_async(
            request
        )
        future.add_done_callback(
            lambda completed_future: (
                self._handle_gripper_response(
                    position_name,
                    completed_future,
                )
            )
        )

    def _handle_gripper_response(
        self,
        position_name: str,
        future: Future,
    ) -> None:
        """处理夹爪服务返回结果。"""
        try:
            response: Optional[
                SetGripper.Response
            ] = future.result()
        except Exception as error:
            self.get_logger().error(
                f"调用夹爪服务失败：{error}"
            )
            return

        if response is None:
            self.get_logger().error(
                "夹爪服务没有返回结果"
            )
            return

        if response.success:
            self.get_logger().info(
                f"夹爪指令执行成功：{position_name}"
            )
            self.get_logger().info(
                f"运动层返回：{response.message}"
            )
            return

        self.get_logger().error(
            f"夹爪指令执行失败：{position_name}"
        )
        self.get_logger().error(
            f"运动层返回：{response.message}"
        )

    def destroy_node(self) -> bool:
        """关闭节点时通知终端输入线程停止。"""
        self.running = False
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = LanguageNode()

    try:
        while rclpy.ok() and node.running:
            rclpy.spin_once(
                node,
                timeout_sec=0.1,
            )
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info(
                "收到 Ctrl+C，正在关闭 language_node"
            )
    finally:
        node.running = False
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
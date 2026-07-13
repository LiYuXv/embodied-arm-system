"""终端中文指令输入与任务命令发布节点."""

import queue
import threading
import uuid

import rclpy
from embodied_interfaces.msg import TaskCommand
from rclpy.node import Node

from embodied_language.command_parser import CommandParser


class LanguageNode(Node):
    """接收终端中文指令并发布结构化任务命令."""

    EXIT_COMMANDS = {"exit", "quit"}

    def __init__(self) -> None:
        super().__init__("language_node")

        self.parser = CommandParser()

        self.command_publisher = self.create_publisher(
            TaskCommand,
            "/task_command",
            10,
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
            "请输入中文指令；输入 exit 或 quit 退出"
        )

    def _terminal_input_loop(self) -> None:
        """在独立线程中读取终端输入."""
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
        """解析终端输入并发布任务命令."""
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
            "任务命令已发布："
            f"action={task_command.action}, "
            f"target={task_command.target}, "
            f"command_id={task_command.command_id}"
        )

    def _build_task_command(
        self,
        raw_text: str,
        action: str,
        target: str,
    ) -> TaskCommand:
        """构造统一任务命令消息."""
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

    def destroy_node(self) -> bool:
        """关闭节点时通知终端输入线程停止."""
        self.running = False
        return super().destroy_node()


def main(args=None) -> None:
    """启动语言交互节点."""
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

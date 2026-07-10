import argparse
import time

import rclpy
from embodied_interfaces.action import MoveToPose
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


class RelativePoseTestClient(Node):
    """读取当前末端位姿，并执行相对位移测试。"""

    def __init__(
        self,
        offset_x: float,
        offset_y: float,
        offset_z: float,
    ) -> None:
        super().__init__("relative_pose_test_client")

        self.base_frame = "base_link"
        self.end_effector_link = "end_effector"

        self.offset_x = float(offset_x)
        self.offset_y = float(offset_y)
        self.offset_z = float(offset_z)

        self._validate_offsets()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.action_client = ActionClient(
            self,
            MoveToPose,
            "/motion/move_to_pose",
        )

    def _validate_offsets(self) -> None:
        """限制单次测试位移，避免误输入过大的运动。"""

        offsets = {
            "dx": self.offset_x,
            "dy": self.offset_y,
            "dz": self.offset_z,
        }

        maximum_offset = 0.10

        for name, value in offsets.items():
            if abs(value) > maximum_offset:
                raise ValueError(
                    f"{name}={value} 超出测试限制。"
                    f"单轴最大允许位移为 {maximum_offset} m"
                )

        if all(abs(value) < 1e-9 for value in offsets.values()):
            raise ValueError(
                "dx、dy、dz 不能全部为 0"
            )

    def get_current_end_effector_pose(
        self,
        timeout_sec: float = 5.0,
    ) -> PoseStamped:
        """读取末端执行器在 base_link 坐标系下的当前位姿。"""

        deadline = time.monotonic() + timeout_sec

        while rclpy.ok():
            if self.tf_buffer.can_transform(
                self.base_frame,
                self.end_effector_link,
                Time(),
            ):
                break

            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "等待 TF 超时："
                    f"{self.base_frame} -> "
                    f"{self.end_effector_link}"
                )

            rclpy.spin_once(
                self,
                timeout_sec=0.1,
            )

        transform = self.tf_buffer.lookup_transform(
            self.base_frame,
            self.end_effector_link,
            Time(),
        )

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = (
            transform.transform.translation.x
        )
        pose.pose.position.y = (
            transform.transform.translation.y
        )
        pose.pose.position.z = (
            transform.transform.translation.z
        )

        pose.pose.orientation.x = (
            transform.transform.rotation.x
        )
        pose.pose.orientation.y = (
            transform.transform.rotation.y
        )
        pose.pose.orientation.z = (
            transform.transform.rotation.z
        )
        pose.pose.orientation.w = (
            transform.transform.rotation.w
        )

        return pose

    def build_target_pose(
        self,
        current_pose: PoseStamped,
    ) -> PoseStamped:
        """在当前位姿基础上增加相对位移，姿态保持不变。"""

        target_pose = PoseStamped()
        target_pose.header.frame_id = self.base_frame
        target_pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        target_pose.pose.position.x = (
            current_pose.pose.position.x
            + self.offset_x
        )
        target_pose.pose.position.y = (
            current_pose.pose.position.y
            + self.offset_y
        )
        target_pose.pose.position.z = (
            current_pose.pose.position.z
            + self.offset_z
        )

        target_pose.pose.orientation.x = (
            current_pose.pose.orientation.x
        )
        target_pose.pose.orientation.y = (
            current_pose.pose.orientation.y
        )
        target_pose.pose.orientation.z = (
            current_pose.pose.orientation.z
        )
        target_pose.pose.orientation.w = (
            current_pose.pose.orientation.w
        )

        return target_pose

    def feedback_callback(
        self,
        feedback_message,
    ) -> None:
        feedback = feedback_message.feedback

        self.get_logger().info(
            "反馈："
            f"step={feedback.current_step}, "
            f"progress={feedback.progress:.0%}"
        )

    def execute(self) -> bool:
        self.get_logger().info(
            "等待 /motion/move_to_pose Action..."
        )

        if not self.action_client.wait_for_server(
            timeout_sec=5.0
        ):
            self.get_logger().error(
                "/motion/move_to_pose 不可用"
            )
            return False

        current_pose = self.get_current_end_effector_pose()
        target_pose = self.build_target_pose(current_pose)

        self.get_logger().info(
            "相对位移："
            f"dx={self.offset_x:.4f}, "
            f"dy={self.offset_y:.4f}, "
            f"dz={self.offset_z:.4f}"
        )

        self.get_logger().info(
            "当前末端位置："
            f"x={current_pose.pose.position.x:.4f}, "
            f"y={current_pose.pose.position.y:.4f}, "
            f"z={current_pose.pose.position.z:.4f}"
        )

        self.get_logger().info(
            "目标末端位置："
            f"x={target_pose.pose.position.x:.4f}, "
            f"y={target_pose.pose.position.y:.4f}, "
            f"z={target_pose.pose.position.z:.4f}"
        )

        goal = MoveToPose.Goal()
        goal.target_pose = target_pose
        goal.motion_type = "pose"
        goal.velocity_scale = 0.1
        goal.acceleration_scale = 0.1

        send_goal_future = (
            self.action_client.send_goal_async(
                goal,
                feedback_callback=self.feedback_callback,
            )
        )

        rclpy.spin_until_future_complete(
            self,
            send_goal_future,
        )

        goal_handle = send_goal_future.result()

        if goal_handle is None:
            self.get_logger().error(
                "发送 Action 目标失败"
            )
            return False

        if not goal_handle.accepted:
            self.get_logger().error(
                "运动目标被拒绝"
            )
            return False

        self.get_logger().info(
            "运动目标已接受，等待执行结果..."
        )

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future,
        )

        result_response = result_future.result()

        if result_response is None:
            self.get_logger().error(
                "没有收到运动结果"
            )
            return False

        result = result_response.result

        if result.success:
            self.get_logger().info(
                f"任意位姿运动成功：{result.message}"
            )
            return True

        self.get_logger().error(
            f"任意位姿运动失败：{result.message}"
        )
        return False


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取当前末端位姿，并发送相对位移目标"
        )
    )

    parser.add_argument(
        "--dx",
        type=float,
        default=0.0,
        help="沿 base_link X 轴移动的距离，单位 m",
    )

    parser.add_argument(
        "--dy",
        type=float,
        default=0.0,
        help="沿 base_link Y 轴移动的距离，单位 m",
    )

    parser.add_argument(
        "--dz",
        type=float,
        default=-0.03,
        help="沿 base_link Z 轴移动的距离，单位 m",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    rclpy.init()

    node = RelativePoseTestClient(
        offset_x=arguments.dx,
        offset_y=arguments.dy,
        offset_z=arguments.dz,
    )

    try:
        success = node.execute()

        if success:
            print("任意末端位姿测试通过")
        else:
            print("任意末端位姿测试失败")

    except Exception as error:
        node.get_logger().exception(
            f"测试过程中发生异常：{error}"
        )

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
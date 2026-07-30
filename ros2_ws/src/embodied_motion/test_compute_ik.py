import copy
import time
import traceback
from typing import Dict, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener


class ComputeIkTestNode(Node):
    """使用真实关节状态作为种子，测试 EDULITE_A3 的 IK。"""

    ERROR_NAMES = {
        MoveItErrorCodes.SUCCESS: "SUCCESS",
        MoveItErrorCodes.FAILURE: "FAILURE",
        MoveItErrorCodes.PLANNING_FAILED: "PLANNING_FAILED",
        MoveItErrorCodes.INVALID_GROUP_NAME: "INVALID_GROUP_NAME",
        MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS:
            "INVALID_GOAL_CONSTRAINTS",
        MoveItErrorCodes.INVALID_ROBOT_STATE:
            "INVALID_ROBOT_STATE",
        MoveItErrorCodes.INVALID_LINK_NAME: "INVALID_LINK_NAME",
        MoveItErrorCodes.FRAME_TRANSFORM_FAILURE:
            "FRAME_TRANSFORM_FAILURE",
        MoveItErrorCodes.ROBOT_STATE_STALE:
            "ROBOT_STATE_STALE",
        MoveItErrorCodes.NO_IK_SOLUTION: "NO_IK_SOLUTION",
        MoveItErrorCodes.TIMED_OUT: "TIMED_OUT",
    }

    def __init__(self) -> None:
        super().__init__("compute_ik_test_node")

        self.base_frame = "base_link"
        self.end_effector_link = "end_effector"
        self.planning_group = "arm"

        self.arm_joint_names = [
            "L1_joint",
            "L2_joint",
            "L3_joint",
            "L4_joint",
            "L5_joint",
            "L6_joint",
        ]

        self.latest_joint_positions: Optional[
            Dict[str, float]
        ] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.joint_state_subscription = (
            self.create_subscription(
                JointState,
                "/joint_states",
                self.joint_state_callback,
                10,
            )
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            "/compute_ik",
        )

    def joint_state_callback(
        self,
        message: JointState,
    ) -> None:
        """保存最新的有效关节状态。"""

        if len(message.name) != len(message.position):
            return

        joint_map = {
            name: float(position)
            for name, position in zip(
                message.name,
                message.position,
            )
        }

        if not all(
            joint_name in joint_map
            for joint_name in self.arm_joint_names
        ):
            return

        self.latest_joint_positions = joint_map

    def wait_for_interfaces(self) -> None:
        """等待 IK 服务和关节状态。"""

        self.get_logger().info(
            "等待 /compute_ik 服务..."
        )

        if not self.ik_client.wait_for_service(
            timeout_sec=5.0
        ):
            raise RuntimeError(
                "/compute_ik 服务不可用"
            )

        self.get_logger().info(
            "等待 /joint_states..."
        )

        deadline = time.monotonic() + 5.0

        while (
            rclpy.ok()
            and self.latest_joint_positions is None
        ):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "等待有效 /joint_states 超时"
                )

            rclpy.spin_once(
                self,
                timeout_sec=0.1,
            )

        self.get_logger().info(
            "已获取当前关节状态"
        )

    def get_current_pose(
        self,
        timeout_sec: float = 5.0,
    ) -> PoseStamped:
        """读取当前末端在 base_link 下的位姿。"""

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
        pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

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

    def fill_seed_state(
        self,
        request: GetPositionIK.Request,
    ) -> None:
        """把当前 L1～L6 状态写入 IK 请求。"""

        if self.latest_joint_positions is None:
            raise RuntimeError(
                "当前关节状态尚未获取"
            )

        joint_state = (
            request.ik_request.robot_state.joint_state
        )

        joint_state.header.stamp = (
            self.get_clock().now().to_msg()
        )

        joint_state.name = list(
            self.arm_joint_names
        )

        joint_state.position = [
            self.latest_joint_positions[joint_name]
            for joint_name in self.arm_joint_names
        ]

        request.ik_request.robot_state.is_diff = False

    def compute_ik(
        self,
        label: str,
        target_pose: PoseStamped,
        ik_link_name: str,
        avoid_collisions: bool,
    ) -> bool:
        """调用 /compute_ik 并打印结果。"""

        request = GetPositionIK.Request()

        request.ik_request.group_name = (
            self.planning_group
        )

        request.ik_request.ik_link_name = (
            ik_link_name
        )

        request.ik_request.pose_stamped = (
            copy.deepcopy(target_pose)
        )

        request.ik_request.avoid_collisions = (
            avoid_collisions
        )

        request.ik_request.timeout.sec = 2
        request.ik_request.timeout.nanosec = 0

        self.fill_seed_state(request)

        display_link = (
            ik_link_name
            if ik_link_name
            else "<自动推断>"
        )

        self.get_logger().info(
            f"开始测试：{label}，"
            f"ik_link={display_link}，"
            f"avoid_collisions={avoid_collisions}"
        )

        future = self.ik_client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=5.0,
        )

        if not future.done():
            self.get_logger().error(
                f"{label}：调用超时"
            )
            return False

        response: Optional[
            GetPositionIK.Response
        ] = future.result()

        if response is None:
            self.get_logger().error(
                f"{label}：未收到响应"
            )
            return False

        error_code = int(response.error_code.val)

        error_name = self.ERROR_NAMES.get(
            error_code,
            f"UNKNOWN_ERROR_{error_code}",
        )

        if error_code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"{label}：IK 失败，"
                f"error={error_name} "
                f"({error_code})"
            )
            return False

        joint_state = response.solution.joint_state

        solution_map = dict(
            zip(
                joint_state.name,
                joint_state.position,
            )
        )

        self.get_logger().info(
            f"{label}：IK 成功"
        )

        for joint_name in self.arm_joint_names:
            if joint_name in solution_map:
                self.get_logger().info(
                    f"  {joint_name}: "
                    f"{solution_map[joint_name]:.6f}"
                )

        return True

    def print_seed_state(self) -> None:
        """打印当前提供给 IK 的种子状态。"""

        if self.latest_joint_positions is None:
            return

        self.get_logger().info(
            "IK 种子关节状态："
        )

        for joint_name in self.arm_joint_names:
            self.get_logger().info(
                f"  {joint_name}: "
                f"{self.latest_joint_positions[joint_name]:.6f}"
            )

    def execute(self) -> None:
        self.wait_for_interfaces()
        self.print_seed_state()

        current_pose = self.get_current_pose()

        lower_pose = copy.deepcopy(current_pose)
        lower_pose.header.stamp = (
            self.get_clock().now().to_msg()
        )
        lower_pose.pose.position.z -= 0.03

        self.get_logger().info(
            "当前位置："
            f"x={current_pose.pose.position.x:.4f}, "
            f"y={current_pose.pose.position.y:.4f}, "
            f"z={current_pose.pose.position.z:.4f}"
        )

        self.get_logger().info(
            "下移目标："
            f"x={lower_pose.pose.position.x:.4f}, "
            f"y={lower_pose.pose.position.y:.4f}, "
            f"z={lower_pose.pose.position.z:.4f}"
        )

        results = {
            "当前位姿，自动末端，无碰撞检查":
                self.compute_ik(
                    label=(
                        "当前位姿，自动末端，"
                        "无碰撞检查"
                    ),
                    target_pose=current_pose,
                    ik_link_name="",
                    avoid_collisions=False,
                ),

            "当前位姿，指定末端，无碰撞检查":
                self.compute_ik(
                    label=(
                        "当前位姿，指定末端，"
                        "无碰撞检查"
                    ),
                    target_pose=current_pose,
                    ik_link_name=self.end_effector_link,
                    avoid_collisions=False,
                ),

            "下移位姿，自动末端，无碰撞检查":
                self.compute_ik(
                    label=(
                        "下移位姿，自动末端，"
                        "无碰撞检查"
                    ),
                    target_pose=lower_pose,
                    ik_link_name="",
                    avoid_collisions=False,
                ),

            "下移位姿，指定末端，无碰撞检查":
                self.compute_ik(
                    label=(
                        "下移位姿，指定末端，"
                        "无碰撞检查"
                    ),
                    target_pose=lower_pose,
                    ik_link_name=self.end_effector_link,
                    avoid_collisions=False,
                ),

            "下移位姿，指定末端，启用碰撞检查":
                self.compute_ik(
                    label=(
                        "下移位姿，指定末端，"
                        "启用碰撞检查"
                    ),
                    target_pose=lower_pose,
                    ik_link_name=self.end_effector_link,
                    avoid_collisions=True,
                ),
        }

        print("\n========== IK 测试汇总 ==========")

        for label, success in results.items():
            state = "成功" if success else "失败"
            print(f"{label}: {state}")

        print("=================================")


def main() -> None:
    rclpy.init()

    node = ComputeIkTestNode()

    try:
        node.execute()

    except Exception as error:
        node.get_logger().error(
            f"IK 测试发生异常：{error}"
        )
        node.get_logger().error(
            traceback.format_exc()
        )

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

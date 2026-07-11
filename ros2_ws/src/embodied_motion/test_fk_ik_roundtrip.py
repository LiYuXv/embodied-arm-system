import copy
import math
import time
import traceback
from typing import Dict, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener


class FkIkRoundtripTestNode(Node):
    """使用相同 RobotState 完成 MoveIt FK → IK 闭环测试。"""

    ERROR_NAMES = {
        MoveItErrorCodes.SUCCESS: "SUCCESS",
        MoveItErrorCodes.FAILURE: "FAILURE",
        MoveItErrorCodes.PLANNING_FAILED: "PLANNING_FAILED",
        MoveItErrorCodes.INVALID_GROUP_NAME: "INVALID_GROUP_NAME",
        MoveItErrorCodes.INVALID_ROBOT_STATE:
            "INVALID_ROBOT_STATE",
        MoveItErrorCodes.INVALID_LINK_NAME: "INVALID_LINK_NAME",
        MoveItErrorCodes.FRAME_TRANSFORM_FAILURE:
            "FRAME_TRANSFORM_FAILURE",
        MoveItErrorCodes.NO_IK_SOLUTION: "NO_IK_SOLUTION",
        MoveItErrorCodes.TIMED_OUT: "TIMED_OUT",
    }

    def __init__(self) -> None:
        super().__init__("fk_ik_roundtrip_test_node")

        self.base_frame = "base_link"
        self.planning_group = "arm"
        self.end_effector_link = "end_effector"
        self.ik_link = "l5_l6_urdf_asm"

        self.arm_joint_names = [
            "L1_joint",
            "L2_joint",
            "L3_joint",
            "L4_joint",
            "L5_joint",
            "L6_joint",
        ]

        self.latest_joint_state: Optional[JointState] = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.joint_state_subscription = (
            self.create_subscription(
                JointState,
                "/joint_states",
                self.joint_state_callback,
                10,
            )
        )

        self.fk_client = self.create_client(
            GetPositionFK,
            "/compute_fk",
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            "/compute_ik",
        )

    def joint_state_callback(
        self,
        message: JointState,
    ) -> None:
        """保存包含机械臂六个关节的完整 JointState。"""

        if len(message.name) != len(message.position):
            return

        available_names = set(message.name)

        if not all(
            joint_name in available_names
            for joint_name in self.arm_joint_names
        ):
            return

        self.latest_joint_state = copy.deepcopy(message)

    def wait_for_interfaces(self) -> None:
        """等待 FK、IK 服务和完整关节状态。"""

        self.get_logger().info(
            "等待 /compute_fk 服务..."
        )

        if not self.fk_client.wait_for_service(
            timeout_sec=5.0
        ):
            raise RuntimeError(
                "/compute_fk 服务不可用"
            )

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

        start_time = self.get_clock().now()

        while (
            rclpy.ok()
            and self.latest_joint_state is None
        ):
            elapsed = (
                self.get_clock().now() - start_time
            ).nanoseconds / 1e9

            if elapsed > 5.0:
                raise RuntimeError(
                    "等待有效 /joint_states 超时"
                )

            rclpy.spin_once(
                self,
                timeout_sec=0.1,
            )

        self.get_logger().info(
            "FK、IK 和关节状态均已准备"
        )

    def build_robot_state(
        self,
        is_diff: bool,
    ) -> RobotState:
        """使用完整 /joint_states 构造 RobotState。"""

        if self.latest_joint_state is None:
            raise RuntimeError(
                "尚未获取 /joint_states"
            )

        robot_state = RobotState()
        robot_state.joint_state = copy.deepcopy(
            self.latest_joint_state
        )

        robot_state.joint_state.header.stamp = (
            self.get_clock().now().to_msg()
        )

        robot_state.is_diff = bool(is_diff)

        return robot_state

    def print_joint_state(self) -> None:
        """打印实际送入 MoveIt 的全部关节。"""

        if self.latest_joint_state is None:
            return

        self.get_logger().info(
            "完整 /joint_states："
        )

        for name, position in zip(
            self.latest_joint_state.name,
            self.latest_joint_state.position,
        ):
            self.get_logger().info(
                f"  {name}: {position:.6f}"
            )

    def compute_fk(self, link_name: str) -> PoseStamped:
        """由当前完整 RobotState 计算指定链接的 FK。"""

        request = GetPositionFK.Request()

        request.header.frame_id = self.base_frame
        request.header.stamp = (
            self.get_clock().now().to_msg()
        )

        request.fk_link_names = [link_name]

        request.robot_state = self.build_robot_state(
            is_diff=False
        )

        self.get_logger().info(
            "调用 /compute_fk..."
        )

        future = self.fk_client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=5.0,
        )

        if not future.done():
            raise RuntimeError(
                "/compute_fk 调用超时"
            )

        response = future.result()

        if response is None:
            raise RuntimeError(
                "/compute_fk 没有返回结果"
            )

        error_code = int(response.error_code.val)

        if error_code != MoveItErrorCodes.SUCCESS:
            error_name = self.ERROR_NAMES.get(
                error_code,
                f"UNKNOWN_ERROR_{error_code}",
            )

            raise RuntimeError(
                f"FK 失败：{error_name} "
                f"({error_code})"
            )

        if not response.pose_stamped:
            raise RuntimeError(
                "FK 成功但没有返回末端位姿"
            )

        pose = copy.deepcopy(
            response.pose_stamped[0]
        )

        # 使用零时间戳，避免历史 TF 时间问题。
        pose.header.stamp.sec = 0
        pose.header.stamp.nanosec = 0

        self.get_logger().info(
            f"MoveIt FK {link_name} 位置："
            f"x={pose.pose.position.x:.6f}, "
            f"y={pose.pose.position.y:.6f}, "
            f"z={pose.pose.position.z:.6f}"
        )

        self.get_logger().info(
            f"MoveIt FK {link_name} 四元数："
            f"x={pose.pose.orientation.x:.6f}, "
            f"y={pose.pose.orientation.y:.6f}, "
            f"z={pose.pose.orientation.z:.6f}, "
            f"w={pose.pose.orientation.w:.6f}"
        )

        return pose

    def compute_ik(
        self,
        label: str,
        target_pose: PoseStamped,
        is_diff: bool,
        ik_link_name: str,
    ) -> bool:
        """使用 FK 位姿作为目标调用 IK。"""

        request = GetPositionIK.Request()

        request.ik_request.group_name = (
            self.planning_group
        )

        request.ik_request.ik_link_name = ik_link_name

        request.ik_request.pose_stamped = (
            copy.deepcopy(target_pose)
        )

        request.ik_request.robot_state = (
            self.build_robot_state(
                is_diff=is_diff
            )
        )

        request.ik_request.avoid_collisions = False
        request.ik_request.timeout.sec = 2
        request.ik_request.timeout.nanosec = 0

        self.get_logger().info(
            f"开始 IK：{label}"
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

        response = future.result()

        if response is None:
            self.get_logger().error(
                f"{label}：没有返回结果"
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

        solution_map: Dict[str, float] = dict(
            zip(
                response.solution.joint_state.name,
                response.solution.joint_state.position,
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

    def verify_fk_matches_tf(self, fk_pose: PoseStamped) -> None:
        """Ensure MoveIt FK and robot_state_publisher agree on the TCP pose."""
        deadline = time.monotonic() + 5.0
        while not self.tf_buffer.can_transform(
            self.base_frame,
            self.end_effector_link,
            Time(),
        ):
            if time.monotonic() >= deadline:
                raise RuntimeError("等待 base_link -> end_effector TF 超时")
            rclpy.spin_once(self, timeout_sec=0.1)

        transform = self.tf_buffer.lookup_transform(
            self.base_frame,
            self.end_effector_link,
            Time(),
        )
        tf_translation = transform.transform.translation
        tf_rotation = transform.transform.rotation
        fk_position = fk_pose.pose.position
        fk_rotation = fk_pose.pose.orientation

        position_error = math.sqrt(
            (fk_position.x - tf_translation.x) ** 2
            + (fk_position.y - tf_translation.y) ** 2
            + (fk_position.z - tf_translation.z) ** 2
        )
        orientation_dot = abs(
            fk_rotation.x * tf_rotation.x
            + fk_rotation.y * tf_rotation.y
            + fk_rotation.z * tf_rotation.z
            + fk_rotation.w * tf_rotation.w
        )

        if position_error > 1e-5 or orientation_dot < 0.99999:
            raise RuntimeError(
                "MoveIt FK 与 TF 不一致："
                f"position_error={position_error:.8f}, "
                f"orientation_dot={orientation_dot:.8f}"
            )

        self.get_logger().info(
            "MoveIt FK 与 TF 一致："
            f"position_error={position_error:.8f}, "
            f"orientation_dot={orientation_dot:.8f}"
        )

    def execute(self) -> bool:
        self.wait_for_interfaces()
        self.print_joint_state()

        tcp_fk_pose = self.compute_fk(self.end_effector_link)
        self.verify_fk_matches_tf(tcp_fk_pose)
        ik_fk_pose = self.compute_fk(self.ik_link)

        lower_pose = copy.deepcopy(ik_fk_pose)
        lower_pose.pose.position.z -= 0.03

        results = {
            "TCP FK 原位姿，指定固定 TCP": (
                self.compute_ik(
                    label=(
                        "TCP FK 原位姿，完整状态，指定固定 TCP"
                    ),
                    target_pose=tcp_fk_pose,
                    is_diff=False,
                    ik_link_name=self.end_effector_link,
                ),
                True,
            ),

            "IK link FK 原位姿，自动 IK tip": (
                self.compute_ik(
                    label=(
                        "IK link FK 原位姿，完整状态，自动 IK tip"
                    ),
                    target_pose=ik_fk_pose,
                    is_diff=False,
                    ik_link_name="",
                ),
                True,
            ),

            "IK link FK 下移3厘米，自动 IK tip": (
                self.compute_ik(
                    label=(
                        "IK link FK 下移3厘米，完整状态，自动 IK tip"
                    ),
                    target_pose=lower_pose,
                    is_diff=False,
                    ik_link_name="",
                ),
                True,
            ),
        }

        print(
            "\n========== FK → IK 闭环测试汇总 =========="
        )

        all_expected = True
        for label, (success, expected) in results.items():
            state = "成功" if success else "失败"
            expected_state = "成功" if expected else "失败"
            passed = success == expected
            all_expected = all_expected and passed
            verdict = "通过" if passed else "不符合预期"
            print(f"{label}: {state}（预期{expected_state}，{verdict}）")

        print(
            "=========================================="
        )
        return all_expected


def main() -> None:
    rclpy.init()

    node = FkIkRoundtripTestNode()

    try:
        if not node.execute():
            raise RuntimeError("FK → IK 闭环测试结果不符合预期")

    except Exception as error:
        node.get_logger().error(
            f"FK → IK 测试异常：{error}"
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

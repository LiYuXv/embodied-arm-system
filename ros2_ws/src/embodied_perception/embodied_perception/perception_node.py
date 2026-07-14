"""机械臂具身操作系统的基础视觉感知节点."""

from typing import Optional

import rclpy
from embodied_interfaces.msg import DetectedObjectArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class PerceptionNode(Node):
    """接收相机数据并发布目标检测结果."""

    def __init__(self) -> None:
        """初始化感知参数、订阅器、发布器和状态监测."""
        super().__init__("perception_node")

        self.declare_parameter(
            "rgb_topic",
            "/camera/color/image_raw",
        )
        self.declare_parameter(
            "depth_topic",
            "/camera/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter("aux_rgb_topic", "")
        self.declare_parameter(
            "camera_info_topic",
            "/camera/color/camera_info",
        )
        self.declare_parameter(
            "target_color",
            "red",
        )
        self.declare_parameter(
            "camera_frame",
            "camera_color_optical_frame",
        )
        self.declare_parameter("require_depth", True)
        self.declare_parameter("require_aux_rgb", False)
        self.declare_parameter(
            "publish_rate_hz",
            1.0,
        )
        self.declare_parameter(
            "status_rate_hz",
            0.2,
        )

        self.rgb_topic = str(
            self.get_parameter("rgb_topic").value
        )
        self.depth_topic = str(
            self.get_parameter("depth_topic").value
        )
        self.aux_rgb_topic = str(
            self.get_parameter("aux_rgb_topic").value
        )
        self.camera_info_topic = str(
            self.get_parameter("camera_info_topic").value
        )
        self.target_color = str(
            self.get_parameter("target_color").value
        )
        self.camera_frame = str(
            self.get_parameter("camera_frame").value
        )
        self.require_depth = bool(
            self.get_parameter("require_depth").value
        )
        self.require_aux_rgb = bool(
            self.get_parameter("require_aux_rgb").value
        )

        publish_rate_hz = self._read_positive_rate(
            "publish_rate_hz",
            default_value=1.0,
        )
        status_rate_hz = self._read_positive_rate(
            "status_rate_hz",
            default_value=0.2,
        )

        self.latest_rgb: Optional[Image] = None
        self.latest_depth: Optional[Image] = None
        self.latest_aux_rgb: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None

        self.rgb_message_count = 0
        self.depth_message_count = 0
        self.aux_rgb_message_count = 0
        self.camera_info_message_count = 0

        self.rgb_subscription = self.create_subscription(
            Image,
            self.rgb_topic,
            self._handle_rgb_image,
            qos_profile_sensor_data,
        )
        self.depth_subscription = None
        if self.require_depth:
            self.depth_subscription = self.create_subscription(
                Image,
                self.depth_topic,
                self._handle_depth_image,
                qos_profile_sensor_data,
            )
        self.aux_rgb_subscription = None
        if self.require_aux_rgb:
            self.aux_rgb_subscription = self.create_subscription(
                Image,
                self.aux_rgb_topic,
                self._handle_aux_rgb_image,
                qos_profile_sensor_data,
            )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._handle_camera_info,
            qos_profile_sensor_data,
        )

        self.detected_objects_publisher = self.create_publisher(
            DetectedObjectArray,
            "/detected_objects",
            10,
        )

        self.publish_timer = self.create_timer(
            1.0 / publish_rate_hz,
            self._publish_placeholder_result,
        )
        self.status_timer = self.create_timer(
            1.0 / status_rate_hz,
            self._report_input_status,
        )

        self.get_logger().info("Perception node started")
        self.get_logger().info(
            f"RGB topic: {self.rgb_topic}"
        )
        self.get_logger().info(
            f"Depth topic: {self.depth_topic}"
        )
        if self.require_aux_rgb:
            self.get_logger().info(f"Aux RGB topic: {self.aux_rgb_topic}")
        self.get_logger().info(
            f"Camera info topic: {self.camera_info_topic}"
        )
        self.get_logger().info(
            f"Target color: {self.target_color}"
        )
        self.get_logger().info(
            "Publishing detected objects on: /detected_objects"
        )
        self.get_logger().info(
            "正在等待配置的相机数据"
        )

    def _read_positive_rate(
        self,
        parameter_name: str,
        default_value: float,
    ) -> float:
        """读取必须大于零的频率参数."""
        value = float(
            self.get_parameter(parameter_name).value
        )

        if value > 0.0:
            return value

        self.get_logger().warning(
            f"{parameter_name} 必须大于 0，"
            f"已使用默认值 {default_value}"
        )
        return default_value

    def _handle_rgb_image(self, message: Image) -> None:
        """保存最新 RGB 图像."""
        self.latest_rgb = message
        self.rgb_message_count += 1

    def _handle_depth_image(self, message: Image) -> None:
        """保存最新深度图像."""
        self.latest_depth = message
        self.depth_message_count += 1

    def _handle_aux_rgb_image(self, message: Image) -> None:
        """保存双 RGB 路线的辅助相机图像."""
        self.latest_aux_rgb = message
        self.aux_rgb_message_count += 1

    def _handle_camera_info(
        self,
        message: CameraInfo,
    ) -> None:
        """保存最新相机内参."""
        self.latest_camera_info = message
        self.camera_info_message_count += 1

    def _camera_inputs_ready(self) -> bool:
        """判断三路相机输入是否全部到达."""
        return (
            self.latest_rgb is not None
            and self.latest_camera_info is not None
            and (not self.require_depth or self.latest_depth is not None)
            and (not self.require_aux_rgb or self.latest_aux_rgb is not None)
        )

    def _report_input_status(self) -> None:
        """定期输出相机输入状态."""
        if self._camera_inputs_ready():
            self.get_logger().info(
                "相机输入已就绪："
                f"rgb={self.rgb_message_count}, "
                f"depth={self.depth_message_count}, "
                f"aux_rgb={self.aux_rgb_message_count}, "
                f"camera_info={self.camera_info_message_count}"
            )
            return

        missing_inputs = []

        if self.latest_rgb is None:
            missing_inputs.append("RGB")

        if self.require_depth and self.latest_depth is None:
            missing_inputs.append("Depth")

        if self.require_aux_rgb and self.latest_aux_rgb is None:
            missing_inputs.append("Aux RGB")

        if self.latest_camera_info is None:
            missing_inputs.append("CameraInfo")

        self.get_logger().warning(
            "等待相机输入："
            + ", ".join(missing_inputs)
        )

    def _publish_placeholder_result(self) -> None:
        """在目标检测实现前发布空目标列表."""
        message = DetectedObjectArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._get_output_frame()
        message.objects = []

        self.detected_objects_publisher.publish(message)

    def _get_output_frame(self) -> str:
        """优先使用实际 RGB 图像的坐标系."""
        if (
            self.latest_rgb is not None
            and self.latest_rgb.header.frame_id
        ):
            return self.latest_rgb.header.frame_id

        return self.camera_frame


def main(args=None) -> None:
    """启动视觉感知节点."""
    rclpy.init(args=args)

    node = PerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            "收到 Ctrl+C，正在关闭 perception_node"
        )
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

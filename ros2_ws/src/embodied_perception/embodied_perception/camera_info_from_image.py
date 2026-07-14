"""Publish usable CameraInfo when Gazebo Sim only exposes image frames."""

from math import tan

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class CameraInfoFromImage(Node):
    """Derive pinhole intrinsics from an image stream and configured FOV."""

    def __init__(self) -> None:
        super().__init__("camera_info_from_image")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("horizontal_fov", 1.047)
        self.declare_parameter("frame_id", "")
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.horizontal_fov = float(
            self.get_parameter("horizontal_fov").value
        )
        self.publisher = self.create_publisher(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            qos_profile_sensor_data,
        )
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self._publish_camera_info,
            qos_profile_sensor_data,
        )

    def _publish_camera_info(self, image: Image) -> None:
        """Mirror the image header and calculate a standard pinhole matrix."""
        if image.width == 0 or image.height == 0:
            return
        focal_length = image.width / (2.0 * tan(self.horizontal_fov / 2.0))
        message = CameraInfo()
        message.header = image.header
        if self.frame_id:
            message.header.frame_id = self.frame_id
        message.width = image.width
        message.height = image.height
        message.distortion_model = "plumb_bob"
        message.d = [0.0] * 5
        message.k = [
            focal_length, 0.0, image.width / 2.0,
            0.0, focal_length, image.height / 2.0,
            0.0, 0.0, 1.0,
        ]
        message.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        message.p = [
            focal_length, 0.0, image.width / 2.0, 0.0,
            0.0, focal_length, image.height / 2.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraInfoFromImage()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

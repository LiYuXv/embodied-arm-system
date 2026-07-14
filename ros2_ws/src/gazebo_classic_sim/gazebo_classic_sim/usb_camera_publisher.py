"""Minimal ROS 2 publisher for a physical V4L2 camera selected by by-path."""

from math import tan
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class UsbCameraPublisher(Node):
    """Capture one V4L2 device and publish image plus pinhole CameraInfo."""

    def __init__(self):
        super().__init__("usb_camera_publisher")
        self.declare_parameter("video_device", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("horizontal_fov", 1.047)
        self.device = str(self.get_parameter("video_device").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.horizontal_fov = float(self.get_parameter("horizontal_fov").value)
        self.bridge = CvBridge()
        self.image_publisher = self.create_publisher(
            Image, str(self.get_parameter("image_topic").value), qos_profile_sensor_data
        )
        self.camera_info_publisher = self.create_publisher(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            qos_profile_sensor_data,
        )
        self.capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        # Two 640x480 YUYV streams can exceed the USB controller budget.  Both
        # specified cameras advertise MJPEG at this size, so request it before
        # configuring resolution and frame rate.
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.get_parameter("width").value))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.get_parameter("height").value))
        self.capture.set(cv2.CAP_PROP_FPS, float(self.get_parameter("fps").value))
        fps = float(self.get_parameter("fps").value)
        self.timer = self.create_timer(1.0 / max(fps, 1.0), self.publish_frame)
        self.last_failure_log = 0.0
        if not self.capture.isOpened():
            self.get_logger().error(f"Unable to open USB camera: {self.device}")
        else:
            self.get_logger().info(f"Publishing USB camera from {self.device}")

    def publish_frame(self):
        """Read one frame and publish image and matching CameraInfo."""
        if not self.capture.isOpened():
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            now = time.monotonic()
            if now - self.last_failure_log >= 5.0:
                self.get_logger().warning("Failed to read a USB camera frame")
                self.last_failure_log = now
            return
        stamp = self.get_clock().now().to_msg()
        image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image.header.stamp = stamp
        image.header.frame_id = self.frame_id
        self.image_publisher.publish(image)
        height, width = frame.shape[:2]
        focal_length = width / (2.0 * tan(self.horizontal_fov / 2.0))
        info = CameraInfo()
        info.header = image.header
        info.width = width
        info.height = height
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [focal_length, 0.0, width / 2.0, 0.0, focal_length, height / 2.0, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [focal_length, 0.0, width / 2.0, 0.0, 0.0, focal_length, height / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.camera_info_publisher.publish(info)

    def destroy_node(self):
        if hasattr(self, "capture"):
            self.capture.release()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UsbCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

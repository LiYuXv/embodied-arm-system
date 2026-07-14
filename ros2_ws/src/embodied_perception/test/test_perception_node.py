"""Tests for the perception node input-state logic."""

import rclpy
from sensor_msgs.msg import CameraInfo, Image

from embodied_perception.perception_node import PerceptionNode


def test_camera_input_state_and_output_frame() -> None:
    """Camera callbacks should update readiness and output frame."""
    rclpy.init()

    node = PerceptionNode()

    try:
        assert node._camera_inputs_ready() is False
        assert node._get_output_frame() == (
            "camera_color_optical_frame"
        )

        rgb_message = Image()
        rgb_message.header.frame_id = "test_camera_frame"

        depth_message = Image()
        depth_message.header.frame_id = "test_camera_frame"

        camera_info_message = CameraInfo()
        camera_info_message.header.frame_id = (
            "test_camera_frame"
        )

        node._handle_rgb_image(rgb_message)

        assert node.rgb_message_count == 1
        assert node._camera_inputs_ready() is False
        assert node._get_output_frame() == (
            "test_camera_frame"
        )

        node._handle_depth_image(depth_message)

        assert node.depth_message_count == 1
        assert node._camera_inputs_ready() is False

        node._handle_camera_info(camera_info_message)

        assert node.camera_info_message_count == 1
        assert node._camera_inputs_ready() is True

        node.require_depth = False
        node.require_aux_rgb = True
        node.latest_aux_rgb = None
        assert node._camera_inputs_ready() is False
        node._handle_aux_rgb_image(Image())
        assert node.aux_rgb_message_count == 1
        assert node._camera_inputs_ready() is True

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

"""Run two physical USB cameras on the same contract as Classic dual RGB."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


MAIN_DEVICE = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6.1:1.0-video-index0"
AUX_DEVICE = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6.2:1.0-video-index0"


def camera_node(name, device, image_topic):
    """Create one by-path physical camera source using the local V4L2 bridge."""
    return Node(
        package="gazebo_classic_sim",
        executable="usb_camera_publisher",
        name=name,
        parameters=[{
            "video_device": device,
            "image_topic": image_topic,
            "camera_info_topic": image_topic.rsplit("/", 1)[0] + "/camera_info",
            "frame_id": name + "_optical_frame",
            "use_sim_time": False,
        }],
        output="screen",
    )


def generate_launch_description():
    config = get_package_share_directory("gazebo_classic_sim") + "/config/dual_rgb_perception.yaml"
    perception = Node(
        package="embodied_perception",
        executable="perception_node",
        name="perception_node",
        parameters=[config, {"use_sim_time": False}],
        output="screen",
    )
    return LaunchDescription([
        camera_node("camera_main", MAIN_DEVICE, "/camera_main/image_raw"),
        camera_node("camera_aux", AUX_DEVICE, "/camera_aux/image_raw"),
        perception,
    ])

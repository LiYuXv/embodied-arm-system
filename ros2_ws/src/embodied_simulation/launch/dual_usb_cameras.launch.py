"""Run the two physical USB cameras on the same topics as dual_rgb_sim."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


MAIN_DEVICE = (
    "/dev/v4l/by-path/"
    "pci-0000:00:14.0-usb-0:6.1:1.0-video-index0"
)
AUX_DEVICE = (
    "/dev/v4l/by-path/"
    "pci-0000:00:14.0-usb-0:6.2:1.0-video-index0"
)


def camera_node(name: str, device: str, image_topic: str) -> Node:
    """Create one V4L2 camera node with a stable by-path device name."""
    return Node(
        package="v4l2_camera",
        executable="v4l2_camera_node",
        name=name,
        parameters=[
            {
                "video_device": device,
                "use_sim_time": False,
            }
        ],
        remappings=[
            ("image_raw", image_topic),
            ("camera_info", image_topic.rsplit("/", 1)[0] + "/camera_info"),
        ],
        output="screen",
    )


def generate_launch_description() -> LaunchDescription:
    simulation_share = FindPackageShare("embodied_simulation")
    perception = Node(
        package="embodied_perception",
        executable="perception_node",
        name="perception_node",
        parameters=[PathJoinSubstitution([simulation_share, "config", "dual_rgb_perception.yaml"]), {"use_sim_time": False}],
        output="screen",
    )
    return LaunchDescription(
        [
            camera_node("camera_main", MAIN_DEVICE, "/camera_main/image_raw"),
            camera_node("camera_aux", AUX_DEVICE, "/camera_aux/image_raw"),
            perception,
        ]
    )

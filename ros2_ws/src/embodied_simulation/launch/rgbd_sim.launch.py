"""Launch the arm with its wrist-mounted RGB-D simulation route."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    simulation_share = FindPackageShare("embodied_simulation")
    spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([simulation_share, "launch", "spawn_el_a3.launch.py"])
        ),
        launch_arguments={
            "world_file": "sorting_workcell.sdf",
            "world_name": "sorting_workcell",
            "camera_mode": "rgbd",
        }.items(),
    )
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
        ],
        remappings=[
            ("/camera/image", "/camera/color/image_raw"),
            ("/camera/depth_image", "/camera/aligned_depth_to_color/image_raw"),
            ("/camera/camera_info", "/camera/color/camera_info"),
        ],
        output="screen",
    )
    perception = Node(
        package="embodied_perception",
        executable="perception_node",
        name="perception_node",
        parameters=[PathJoinSubstitution([simulation_share, "config", "rgbd_perception.yaml"])],
        output="screen",
    )
    camera_info = Node(
        package="embodied_perception",
        executable="camera_info_from_image",
        name="rgbd_camera_info",
        parameters=[
            {
                "use_sim_time": True,
                "image_topic": "/camera/color/image_raw",
                "camera_info_topic": "/camera/color/camera_info",
            }
        ],
        output="screen",
    )
    return LaunchDescription([spawn, bridge, camera_info, perception])

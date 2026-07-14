"""Launch the arm with two independent USB-style RGB simulation cameras."""

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
            "camera_mode": "dual_rgb",
        }.items(),
    )
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/camera_main@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera_aux@sensor_msgs/msg/Image@gz.msgs.Image",
        ],
        remappings=[
            ("/camera_main", "/camera_main/image_raw"),
            ("/camera_aux", "/camera_aux/image_raw"),
        ],
        output="screen",
    )
    perception = Node(
        package="embodied_perception",
        executable="perception_node",
        name="perception_node",
        parameters=[PathJoinSubstitution([simulation_share, "config", "dual_rgb_perception.yaml"])],
        output="screen",
    )
    camera_info_nodes = [
        Node(
            package="embodied_perception",
            executable="camera_info_from_image",
            name="camera_main_info",
            parameters=[
                {
                    "use_sim_time": True,
                    "image_topic": "/camera_main/image_raw",
                    "camera_info_topic": "/camera_main/camera_info",
                }
            ],
            output="screen",
        ),
        Node(
            package="embodied_perception",
            executable="camera_info_from_image",
            name="camera_aux_info",
            parameters=[
                {
                    "use_sim_time": True,
                    "image_topic": "/camera_aux/image_raw",
                    "camera_info_topic": "/camera_aux/camera_info",
                }
            ],
            output="screen",
        ),
    ]
    return LaunchDescription([spawn, bridge, *camera_info_nodes, perception])

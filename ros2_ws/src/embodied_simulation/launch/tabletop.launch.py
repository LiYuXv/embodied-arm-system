"""Launch the tabletop world in Gazebo Sim."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    world_path = PathJoinSubstitution(
        [
            FindPackageShare("embodied_simulation"),
            "worlds",
            "tabletop.sdf",
        ]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("ros_gz_sim"),
                    "launch",
                    "gz_sim.launch.py",
                ]
            )
        ),
        launch_arguments={
            "gz_args": ["-r -v 2 ", world_path],
        }.items(),
    )

    return LaunchDescription([gazebo])

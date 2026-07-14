"""Launch a minimal EDULITE A3 Gazebo Classic control validation."""

from launch import LaunchDescription
import os

from ament_index_python.packages import get_package_prefix
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("gazebo_classic_sim")
    controllers = PathJoinSubstitution([package_share, "config", "controllers.yaml"])
    world = PathJoinSubstitution([package_share, "worlds", "el_a3_workbench.world"])
    gazebo_plugin_path = os.environ.get("GAZEBO_PLUGIN_PATH", "")
    ros_plugin_dir = os.path.join(get_package_prefix("gazebo_ros2_control"), "lib")
    plugin_path = ros_plugin_dir if not gazebo_plugin_path else f"{ros_plugin_dir}:{gazebo_plugin_path}"
    description = ParameterValue(
        Command([
            FindExecutable(name="python3"), " ",
            PathJoinSubstitution([package_share, "scripts", "classic_robot_description.py"]),
            " --controllers ", controllers,
        ]),
        value_type=str,
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"])
        ),
        launch_arguments={"world": world, "verbose": LaunchConfiguration("verbose")}.items(),
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": description, "use_sim_time": True}],
        output="screen",
    )
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-topic", "robot_description", "-entity", "el_a3", "-x", "-0.08", "-z", "0.81"],
        output="screen",
    )
    spawn_controllers = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster", "arm_controller", "gripper_controller",
            "--activate-as-group", "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "60",
        ],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("verbose", default_value="false"),
        SetEnvironmentVariable("GAZEBO_PLUGIN_PATH", plugin_path),
        SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),
        gazebo,
        robot_state_publisher,
        spawn_robot,
        RegisterEventHandler(OnProcessExit(target_action=spawn_robot, on_exit=[spawn_controllers])),
    ])

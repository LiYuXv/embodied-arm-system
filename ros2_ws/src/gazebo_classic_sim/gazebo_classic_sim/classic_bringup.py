"""Shared launch construction for mutually exclusive Gazebo Classic routes."""

import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def build_classic_launch(world_name, camera_mode="none", perception_config=None):
    """Create one robot/world route; camera modes are intentionally exclusive."""
    simulation_share = get_package_share_directory("gazebo_classic_sim")
    controllers = os.path.join(simulation_share, "config", "controllers.yaml")
    world = os.path.join(simulation_share, "worlds", world_name)
    robot_adapter = os.path.join(simulation_share, "scripts", "classic_robot_description.py")
    plugin_dirs = [
        os.path.join(get_package_prefix("gazebo_ros2_control"), "lib"),
        os.path.join(get_package_prefix("gazebo_classic_gripper_mimic"), "lib"),
        os.path.join(get_package_prefix("gazebo_plugins"), "lib"),
        os.environ.get("GAZEBO_PLUGIN_PATH", ""),
    ]
    description_share = get_package_share_directory("el_a3_description")
    model_dirs = [
        description_share,
        os.path.dirname(description_share),
        os.environ.get("GAZEBO_MODEL_PATH", ""),
    ]
    description = ParameterValue(
        Command([
            FindExecutable(name="python3"), " ", robot_adapter,
            " --controllers ", controllers,
            " --camera-mode ", camera_mode,
        ]),
        value_type=str,
    )
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world": world,
            "verbose": LaunchConfiguration("verbose"),
            # Start the client explicitly after gzserver has loaded the SDF;
            # starting both simultaneously leaves the Classic GUI stuck on
            # "Preparing your world" on slower desktops.
            "gui": "false",
        }.items(),
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
        arguments=[
            "-topic", "robot_description", "-entity", "el_a3",
            "-x", "-0.08", "-z", "0.83", "-Y", "3.14159",
        ],
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
    actions = [
        DeclareLaunchArgument("verbose", default_value="false"),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument(
            "gazebo_master_uri",
            default_value="http://127.0.0.1:11346",
            description="Dedicated Gazebo Classic master URI for this system instance.",
        ),
        SetEnvironmentVariable("GAZEBO_PLUGIN_PATH", ":".join(filter(None, plugin_dirs))),
        SetEnvironmentVariable("GAZEBO_MODEL_PATH", ":".join(filter(None, model_dirs))),
        SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),
        SetEnvironmentVariable(
            "GAZEBO_MASTER_URI",
            LaunchConfiguration("gazebo_master_uri"),
        ),
        gazebo,
        TimerAction(
            period=4.0,
            actions=[ExecuteProcess(cmd=["gzclient"], output="screen")],
            condition=IfCondition(LaunchConfiguration("gazebo_gui")),
        ),
        robot_state_publisher,
        spawn_robot,
        RegisterEventHandler(OnProcessExit(target_action=spawn_robot, on_exit=[spawn_controllers])),
    ]
    if perception_config:
        actions.append(Node(
            package="embodied_perception",
            executable="perception_node",
            name="perception_node",
            parameters=[os.path.join(simulation_share, "config", perception_config), {"use_sim_time": True}],
            output="screen",
        ))
    return LaunchDescription(actions)

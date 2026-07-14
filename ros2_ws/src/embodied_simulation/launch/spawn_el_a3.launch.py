"""Spawn the EDULITE A3 robot in the tabletop Gazebo world."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.actions import RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def append_resource_path(variable_name: str, new_path: str) -> str:
    """Append a directory without discarding an existing resource path."""
    current_path = os.environ.get(variable_name, "")

    if current_path:
        return new_path + os.pathsep + current_path

    return new_path


def generate_launch_description() -> LaunchDescription:
    world_path = PathJoinSubstitution(
        [
            FindPackageShare("embodied_simulation"),
            "worlds",
            LaunchConfiguration("world_file"),
        ]
    )

    # Gazebo resolves model://el_a3_description/... by searching directories
    # containing the el_a3_description package directory.
    description_share = get_package_share_directory("el_a3_description")
    description_resource_root = os.path.dirname(description_share)

    set_ign_resource_path = SetEnvironmentVariable(
        name="IGN_GAZEBO_RESOURCE_PATH",
        value=append_resource_path(
            "IGN_GAZEBO_RESOURCE_PATH",
            description_resource_root,
        ),
    )

    # Keep the newer variable as well, so the launch file remains compatible
    # with later Gazebo Sim releases.
    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=append_resource_path(
            "GZ_SIM_RESOURCE_PATH",
            description_resource_root,
        ),
    )

    # Start paused so the robot cannot fall before Gazebo control is connected.
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
            # Gazebo starts paused unless -r is supplied. It is resumed only
            # after the position controllers are active below.
            "gz_args": ["-v 1 ", world_path],
        }.items(),
    )

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("embodied_simulation"),
                    "urdf",
                    "el_a3_sim.urdf.xacro",
                ]
            ),
            " camera_mode:=",
            LaunchConfiguration("camera_mode"),
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(
            robot_description_content,
            value_type=str,
        )
    }

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description],
        output="screen",
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic",
            "robot_description",
            "-name",
            "el_a3",
            "-x",
            "-0.08",
            "-y",
            "0.0",
            "-z",
            "0.83",
            "-Y",
            "3.14159",
        ],
        output="screen",
    )

    controllers_file = PathJoinSubstitution(
        [FindPackageShare("embodied_simulation"), "config", "gazebo_controllers.yaml"]
    )
    load_controllers = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "arm_controller",
            "gripper_controller",
            "--activate-as-group",
            "--inactive",
            "--param-file",
            controllers_file,
            "--controller-manager-timeout",
            "30",
        ],
        output="screen",
    )
    activate_controllers = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "arm_controller",
            "gripper_controller",
            "--activate-as-group",
            "--param-file",
            controllers_file,
            "--controller-manager-timeout",
            "30",
            "--switch-timeout",
            "30",
        ],
        output="screen",
    )
    resume_simulation = ExecuteProcess(
        cmd=[
            "ign",
            "service",
            "--service",
            ["/world/", LaunchConfiguration("world_name"), "/control"],
            "--reqtype",
            "ignition.msgs.WorldControl",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            "pause: false",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("world_file", default_value="sorting_workcell.sdf"),
            DeclareLaunchArgument("world_name", default_value="sorting_workcell"),
            DeclareLaunchArgument("camera_mode", default_value="none"),
            set_ign_resource_path,
            set_gz_resource_path,
            gazebo,
            robot_state_publisher,
            spawn_robot,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=spawn_robot,
                    on_exit=[load_controllers],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=load_controllers,
                    on_exit=[resume_simulation],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=resume_simulation,
                    on_exit=[activate_controllers],
                )
            ),
        ]
    )

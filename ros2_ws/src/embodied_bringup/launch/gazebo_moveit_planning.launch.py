"""Start only the MoveIt planning and optional RViz layers for Gazebo."""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _load_yaml(package_name, relative_path):
    """Load a MoveIt configuration YAML file."""
    path = os.path.join(
        get_package_share_directory(package_name),
        relative_path,
    )
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def generate_launch_description():
    """Build the planning-only launch without a controller or RSP node."""
    use_rviz = LaunchConfiguration("use_rviz")
    moveit_package = "el_a3_moveit_config"

    robot_description_content = Command([
        FindExecutable(name="xacro"),
        " ",
        os.path.join(
            get_package_share_directory("el_a3_description"),
            "urdf",
            "el_a3.urdf.xacro",
        ),
        " use_mock_hardware:=true",
    ])
    robot_description = {
        "robot_description": ParameterValue(
            robot_description_content,
            value_type=str,
        )
    }
    semantic_content = Command([
        "cat ",
        os.path.join(
            get_package_share_directory(moveit_package),
            "config",
            "el_a3.srdf",
        ),
    ])
    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(
            semantic_content,
            value_type=str,
        )
    }
    kinematics = _load_yaml(moveit_package, "config/kinematics.yaml")
    joint_limits = _load_yaml(moveit_package, "config/joint_limits.yaml")
    ompl = _load_yaml(moveit_package, "config/ompl_planning.yaml")
    controllers = _load_yaml(
        moveit_package,
        "config/moveit_controllers.yaml",
    )
    planning_scene_monitor = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "publish_planning_scene_hz": 4.0,
    }
    trajectory_execution = {
        # Gazebo owns /controller_manager and controller activation.
        "moveit_manage_controllers": False,
        # Gazebo's position controller settles a little after the trajectory
        # timestamp.  The former 1.2x + 0.5 s window canceled valid motions
        # while the controller was still reporting its final point.
        "trajectory_execution.allowed_execution_duration_scaling": 4.0,
        # Keep MoveIt's watchdog longer than arm_controller's physical
        # post-trajectory settling window.  Otherwise MoveIt cancels a valid
        # damped Gazebo trajectory before the controller can report its
        # collision-checked final state.
        "trajectory_execution.allowed_goal_duration_margin": 13.0,
        # Gazebo feedback is sampled while the previous point is settling;
        # retain a realistic tolerance so chained pick/place motions are not
        # rejected before they reach the controller.
        "trajectory_execution.allowed_start_tolerance": 0.60,
    }

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            {"robot_description_planning": joint_limits},
            kinematics,
            {"move_group": ompl},
            trajectory_execution,
            controllers,
            planning_scene_monitor,
            {"use_sim_time": True},
        ],
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=[
            "-d",
            os.path.join(
                get_package_share_directory(moveit_package),
                "config",
                "moveit.rviz",
            ),
        ],
        parameters=[
            robot_description,
            robot_description_semantic,
            {"robot_description_planning": joint_limits},
            {"robot_description_kinematics": kinematics},
            {"use_sim_time": True},
        ],
        condition=IfCondition(use_rviz),
    )
    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="true"),
        move_group,
        rviz,
    ])

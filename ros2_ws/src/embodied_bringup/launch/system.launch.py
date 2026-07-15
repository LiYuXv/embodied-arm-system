"""One-command launcher for mock and Gazebo Classic system backends."""

import os
import shutil

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


VALID_BACKENDS = {"mock", "gazebo"}
VALID_CAMERA_SOURCES = {
    "none",
    "rgbd_sim",
    "dual_rgb_sim",
    "dual_usb",
}


def _as_bool(context, name):
    """Read a ROS launch boolean argument."""
    return LaunchConfiguration(name).perform(context).strip().lower() in {
        "true",
        "1",
        "yes",
        "on",
    }


def _launch_language_node(context):
    """Optionally run the interactive language node in GNOME Terminal."""
    if not _as_bool(context, "open_language_terminal"):
        return [
            LogInfo(
                msg=(
                    "未自动启动 language_node；可在其他终端运行："
                    "ros2 run embodied_language language_node"
                )
            )
        ]

    terminal_command = shutil.which("gnome-terminal")
    if terminal_command is None:
        return [
            LogInfo(
                msg=(
                    "未找到 gnome-terminal，language_node 未自动启动；"
                    "请在新终端运行 ros2 run embodied_language language_node"
                )
            )
        ]
    return [
        LogInfo(msg="正在新终端中启动 language_node..."),
        ExecuteProcess(
            cmd=[
                terminal_command,
                "--wait",
                "--title=Embodied Language",
                "--",
                "bash",
                "-c",
                "ros2 run embodied_language language_node",
            ],
            output="screen",
        ),
    ]


def _include(package, filename, arguments=None):
    """Include a Python launch file from an installed ROS package."""
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(package),
                "launch",
                filename,
            )
        ),
        launch_arguments=(arguments or {}).items(),
    )


def _select_backend(context):
    """Select mutually exclusive mock or Gazebo infrastructure launches."""
    backend = LaunchConfiguration("backend").perform(context).strip().lower()
    camera_source = (
        LaunchConfiguration("camera_source").perform(context).strip().lower()
    )
    if backend not in VALID_BACKENDS:
        raise RuntimeError(
            "backend must be one of: " + ", ".join(sorted(VALID_BACKENDS))
        )
    if camera_source not in VALID_CAMERA_SOURCES:
        raise RuntimeError(
            "camera_source must be one of: "
            + ", ".join(sorted(VALID_CAMERA_SOURCES))
        )

    use_rviz = LaunchConfiguration("use_rviz")
    if backend == "mock":
        actions = [_include(
            "el_a3_moveit_config",
            "demo.launch.py",
            {"use_rviz": use_rviz},
        )]
        # The mock route retains the original perception node.  A USB source
        # owns its perception node so the node is not duplicated.
        if camera_source == "dual_usb":
            actions.append(_include(
                "gazebo_classic_sim",
                "classic_dual_usb_cameras.launch.py",
                _usb_camera_arguments(),
            ))
        else:
            actions.append(_perception_node(use_sim_time=False))
        return actions

    if camera_source == "rgbd_sim":
        gazebo_launch = "classic_rgbd_sim.launch.py"
    elif camera_source == "dual_rgb_sim":
        gazebo_launch = "classic_dual_rgb_sim.launch.py"
    else:
        gazebo_launch = "classic_el_a3.launch.py"

    actions = [_include(
        "gazebo_classic_sim",
        gazebo_launch,
        {"gazebo_gui": LaunchConfiguration("gazebo_gui")},
    )]
    if camera_source == "dual_usb":
        actions.append(_include(
            "gazebo_classic_sim",
            "classic_dual_usb_cameras.launch.py",
            _usb_camera_arguments(),
        ))
    actions.append(_include(
        "embodied_bringup",
        "gazebo_moveit_planning.launch.py",
        {"use_rviz": use_rviz},
    ))
    return actions


def _usb_camera_arguments():
    """Forward USB camera device and frame calibration launch arguments."""
    return {
        "camera_main_device": LaunchConfiguration("camera_main_device"),
        "camera_aux_device": LaunchConfiguration("camera_aux_device"),
        "camera_main_frame_id": LaunchConfiguration("camera_main_frame_id"),
        "camera_aux_frame_id": LaunchConfiguration("camera_aux_frame_id"),
    }


def _perception_node(use_sim_time):
    """Create the single mock perception node when no camera launch owns it."""
    return Node(
        package="embodied_perception",
        executable="perception_node",
        name="perception_node",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
        emulate_tty=True,
    )


def _launch_runtime_nodes(context):
    """Start motion and task layers with the selected clock source."""
    use_sim_time = (
        LaunchConfiguration("backend").perform(context).strip().lower()
        == "gazebo"
    )
    motion_executor = Node(
        package="embodied_motion",
        executable="motion_executor_node",
        name="motion_executor_node",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
        emulate_tty=True,
    )
    task_manager = Node(
        package="embodied_task",
        executable="task_manager_node",
        name="task_manager_node",
        parameters=[{
            "backend": LaunchConfiguration("backend"),
            "use_gazebo_attachment": True,
            "use_sim_time": use_sim_time,
        }],
        output="screen",
        emulate_tty=True,
    )
    return [
        TimerAction(period=6.0, actions=[motion_executor]),
        TimerAction(period=7.0, actions=[task_manager]),
    ]


def generate_launch_description():
    """Generate the full system launch description."""
    return LaunchDescription([
        DeclareLaunchArgument(
            "backend",
            default_value="mock",
            description="Execution backend: mock or gazebo.",
        ),
        DeclareLaunchArgument(
            "camera_source",
            default_value="none",
            description="Camera route: none, rgbd_sim, dual_rgb_sim, dual_usb.",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Start RViz.",
        ),
        DeclareLaunchArgument(
            "open_language_terminal",
            default_value="true",
            description="Start language_node in a GNOME Terminal.",
        ),
        DeclareLaunchArgument(
            "gazebo_gui",
            default_value="true",
            description="Show Gazebo Classic client when using backend=gazebo.",
        ),
        DeclareLaunchArgument(
            "camera_main_device",
            default_value=(
                "/dev/v4l/by-path/"
                "pci-0000:00:14.0-usb-0:6.1:1.0-video-index0"
            ),
            description="V4L2 device path for camera_main.",
        ),
        DeclareLaunchArgument(
            "camera_aux_device",
            default_value=(
                "/dev/v4l/by-path/"
                "pci-0000:00:14.0-usb-0:6.2:1.0-video-index0"
            ),
            description="V4L2 device path for camera_aux.",
        ),
        DeclareLaunchArgument(
            "camera_main_frame_id",
            default_value="camera_main_optical_frame",
            description="Published optical frame for camera_main USB images.",
        ),
        DeclareLaunchArgument(
            "camera_aux_frame_id",
            default_value="camera_aux_optical_frame",
            description="Published optical frame for camera_aux USB images.",
        ),
        LogInfo(msg="正在选择机械臂系统后端..."),
        OpaqueFunction(function=_select_backend),
        OpaqueFunction(function=_launch_runtime_nodes),
        TimerAction(
            period=8.0,
            actions=[OpaqueFunction(function=_launch_language_node)],
        ),
    ])

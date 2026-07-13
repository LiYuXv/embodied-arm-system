"""机械臂具身操作系统总启动文件."""

import os
import shutil

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_language_node(context):
    """在独立终端中启动交互式语言节点."""
    open_terminal = (
        LaunchConfiguration(
            "open_language_terminal"
        )
        .perform(context)
        .strip()
        .lower()
        in {"true", "1", "yes", "on"}
    )

    if not open_terminal:
        return [
            LogInfo(
                msg=(
                    "未自动启动 language_node。"
                    "可在其他终端手动运行："
                    "ros2 run embodied_language language_node"
                )
            )
        ]

    terminal_command = shutil.which("gnome-terminal")

    if terminal_command is None:
        return [
            LogInfo(
                msg=(
                    "未找到 gnome-terminal，"
                    "language_node 未自动启动。"
                    "请在新终端运行："
                    "ros2 run embodied_language language_node"
                )
            )
        ]

    return [
        LogInfo(
            msg="正在新终端中启动 language_node..."
        ),
        ExecuteProcess(
            cmd=[
                terminal_command,
                "--wait",
                "--title=Embodied Language",
                "--",
                "bash",
                "-c",
                (
                    "ros2 run "
                    "embodied_language "
                    "language_node"
                ),
            ],
            output="screen",
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    """生成完整系统启动描述."""
    moveit_share_directory = (
        get_package_share_directory(
            "el_a3_moveit_config"
        )
    )

    moveit_launch_file = os.path.join(
        moveit_share_directory,
        "launch",
        "demo.launch.py",
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            moveit_launch_file
        )
    )

    motion_executor_node = Node(
        package="embodied_motion",
        executable="motion_executor_node",
        name="motion_executor_node",
        output="screen",
        emulate_tty=True,
    )

    delayed_motion_executor = TimerAction(
        period=4.0,
        actions=[
            LogInfo(
                msg="正在启动 motion_executor_node..."
            ),
            motion_executor_node,
        ],
    )

    task_manager_node = Node(
        package="embodied_task",
        executable="task_manager_node",
        name="task_manager_node",
        output="screen",
        emulate_tty=True,
    )

    delayed_task_manager = TimerAction(
        period=5.0,
        actions=[
            LogInfo(
                msg="正在启动 task_manager_node..."
            ),
            task_manager_node,
        ],
    )

    perception_node = Node(
        package="embodied_perception",
        executable="perception_node",
        name="perception_node",
        output="screen",
        emulate_tty=True,
    )

    delayed_perception = TimerAction(
        period=6.0,
        actions=[
            LogInfo(
                msg="正在启动 perception_node..."
            ),
            perception_node,
        ],
    )

    delayed_language_node = TimerAction(
        period=7.0,
        actions=[
            OpaqueFunction(
                function=_launch_language_node
            )
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "open_language_terminal",
                default_value="true",
                description=(
                    "是否在独立 GNOME Terminal 中"
                    "启动交互式 language_node"
                ),
            ),
            LogInfo(
                msg=(
                    "正在启动机械臂具身操作系统："
                    "MoveIt、运动层、任务层、视觉感知层和语言交互层"
                )
            ),
            moveit_launch,
            delayed_motion_executor,
            delayed_task_manager,
            delayed_perception,
            delayed_language_node,
        ]
    )

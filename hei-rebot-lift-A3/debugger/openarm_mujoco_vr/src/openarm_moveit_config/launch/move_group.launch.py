from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # 1. 构建 MoveIt 配置
    moveit_config = MoveItConfigsBuilder("openarm", package_name="openarm_moveit_config").to_moveit_configs()

    # 2. 手动创建 move_group 节点，强行插入 use_sim_time
    # 这样写可以确保参数绝对生效，没有任何封装层能挡住它
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(), # 加载所有机器人配置
            {"use_sim_time": True},  # <--- 【核心修正】强制使用仿真时间
            {"monitor_dynamics": False}, # 关闭一些可能报错的动态监控
            {"publish_robot_description_semantic": True}
        ],
    )

    return LaunchDescription([
        run_move_group_node
    ])
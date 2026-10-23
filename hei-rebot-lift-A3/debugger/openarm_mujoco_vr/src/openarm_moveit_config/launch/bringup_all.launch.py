import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义 launch 参数
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # 1. 仿真与检测 Launch
    sim_launch_path = os.path.join(current_dir, 'run_simulation_with_detector.launch.py')
    
    # 2. MoveIt MoveGroup Launch
    move_group_launch_path = os.path.join(current_dir, 'move_group.launch.py')
    
    # 3. 你的控制脚本路径
    control_script_path = os.path.join(current_dir, 'pick_apple.py')

    # --- 动作定义 ---

    # 动作 1: 启动仿真环境 (它内部已经强制 set use_sim_time=True 了)
    launch_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch_path)
    )

    # 动作 2: 启动 MoveIt (关键修改：传递 use_sim_time 参数)
    launch_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(move_group_launch_path),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_table_node = ExecuteProcess(
        cmd=['python3',
            os.path.join(current_dir, 'spawn_table_scene.py'),  # 指向源码
            '--ros-args', '-p', 'use_sim_time:=true'],
        output='screen'
    )

    # 动作 3: 启动你的 Python 控制节点
    # 关键修改：添加 --ros-args -p use_sim_time:=true
    run_control_node = ExecuteProcess(
        cmd=['python3', control_script_path, '--ros-args', '-p', 'use_sim_time:=true'],
        output='screen'
    )


    delayed_control_node = TimerAction(
        period=8.0, 
        actions=[run_control_node]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo/Mujoco) clock if true'),
        
        launch_sim,
        launch_moveit,
        spawn_table_node,
        delayed_control_node
    ])
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
import os
os.environ['MUJOCO_GL'] = 'osmesa'

def generate_launch_description():
    scene_path = os.path.dirname(__file__)
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    
    mujoco_xml_file = os.path.join(model_dir, 'table_scene.xml')
    urdf_file = os.path.join(model_dir, 'table_scene_full.urdf')
    controller_config = os.path.join(scene_path, 'my_controllers.yaml')
    detector_script = os.path.join(scene_path, 'apple_detector.py')

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()
    robot_description = {'robot_description': robot_desc}

    # 1. 启动 Mujoco 仿真
    node_mujoco = Node(
        package='mujoco_ros2_control',
        executable='mujoco_ros2_control',
        output='screen',
        parameters=[
            robot_description,
            {'use_sim_time': True},
            {'mujoco_model_path': mujoco_xml_file},
            controller_config,
            {'headless': False},
            {'show_menus': True},
            {'render_every_frame': True},
        ]
    )

    # 2. 机器人状态发布器
    node_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': True
        }]
    )

    # --- 关键新增：发布相机的 TF 变换 ---
    
    # 变换 1: World -> MuJoCo Camera Body
    # 数据来源 scene.xml: pos="0.8 0 1.4" quat="0.659 0.159 0.159 0.659"
    # 注意: ROS args 顺序是 x y z qx qy qz qw
    # MuJoCo quat 是 (w, x, y, z)，所以这里要填 0.159 0.159 0.659 0.659
    tf_world_to_cam = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments = ['0.5', '0', '1.2', '0', '0', '0', '1', 'world', 'camera_base_link'],
        output='screen'
    )

    # 变换 2: MuJoCo Camera Body -> ROS Optical Frame (修正 OpenCV 坐标系差异)
    # MuJoCo 相机默认 -Z 朝前，+Y 朝上
    # OpenCV/ROS 相机默认 +Z 朝前，+Y 朝下
    # 需要绕 X 轴旋转 -180 度 (Roll=-3.14159)
    # 对应的四元数 (x=1, y=0, z=0, w=0)
    tf_cam_to_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments = ['0', '0', '0', '1', '0', '0', '0', 'camera_base_link', 'camera_optical_frame'],
        output='screen'
    )

    # -----------------------------------

    # 3. 控制器加载
    load_jsb = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'],
        output='screen'
    )

    load_right_arm = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'right_arm_controller'],
        output='screen'
    )

    load_left_arm = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'left_arm_controller'],
        output='screen'
    )
    
    load_right_hand = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'right_hand_controller'],
        output='screen'
    )
    
    load_left_hand = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'left_hand_controller'],
        output='screen'
    )

    load_cube_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'cube_controller'],
        output='screen'
    )

    # 4. 启动检测脚本 (关键修正：添加 use_sim_time，否则 TF 监听会失败)
    node_detector = ExecuteProcess(
        cmd=['python3', detector_script, '--ros-args', '-p', 'use_sim_time:=true'],
        output='screen'
    )

    return LaunchDescription([
        node_mujoco,
        node_rsp,
        tf_world_to_cam,    # 新增
        tf_cam_to_optical,  # 新增
        node_detector,
        load_jsb,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_jsb,
                on_exit=[load_right_arm, load_left_arm, load_right_hand, load_left_hand, load_cube_controller],
            )
        ),
    ])
ros2 launch embodied_bringup system.launch.py
  backend:**=**gazebo
  camera_source:**=**dual_rgb_sim
  use_rviz:**=**true
  gazebo_gui:**=**true
  show_camera_views:**=**true
  open_language_terminal:**=**true
  gazebo_master_uri:**=**http://127.0.0.1:11501

# 2026-07-15 Day 6：系统集成

## 目标

将 Gazebo Classic、双 RGB 相机、语言命令和抓放任务接入同一个启动入口，并保持原有 mock 流程可用。

## 实现

- `system.launch.py` 增加 `backend`、`camera_source`、`use_rviz`、`open_language_terminal`、`gazebo_gui`、`gazebo_master_uri` 和 `show_camera_views` 参数；Gazebo 路线由 Classic 启动拥有 `robot_state_publisher`、控制器和感知节点，新增的 MoveIt 启动文件仅运行规划层和可选 RViz。独立 master 消除了遗留 `gzserver` 导致的“waiting for service”冲突。
- 双 RGB 场景中的 `camera_main` 固定在工作台前上方；`camera_aux` 由机器人描述适配器固定到 `gripper_base_link`。两路话题保持 `/camera_main/*`、`/camera_aux/*`。
- 双 USB 启动将两个 V4L2 路径和两个 `frame_id` 改为 launch 参数。
- 语言解析器将“把红色方块抓到红色位置”映射为 `action=pick_place`、`target=red_cube`、`target_region=red_target_zone`。
- 任务管理器以 Action/Service 回调组成非阻塞状态机：打开夹爪、方块上方、下移、关闭、抬起、目标区上方、下移、打开、抬起。位姿、速度、相机外参和工作台高度存放在 `embodied_task/config/pick_place.yaml`。

## 颜色定位和变换

主相机图像先转换到 HSV；红色使用 `[0, 10]` 和 `[170, 180]` 两个 Hue 区间，之后做开运算去噪。面积较小和较大的红色外轮廓分别作为方块和目标区域，取它们的质心像素。利用 `CameraInfo.K` 反投影得到相机射线，使用 YAML 中的相机到 `base_link` 外参旋转/平移，并与已知工作台平面求交，得到机器人基坐标中的 XY。没有完整相机数据或检测不可靠时，任务安全回退到 YAML 标定位姿。

Gazebo 的红方块使用真实碰撞、摩擦和夹爪接触；任务管理器不使用 `/gazebo/set_model_state`、附着或任何位姿失败回退。必要位姿或执行失败时立即明确终止任务。

任一抓放位姿的 IK 或规划执行失败都会明确终止任务；不会以命名位姿替代失败段。

## 验证计划

```bash
cd ~/embodied-arm-system/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch embodied_bringup system.launch.py \
  backend:=gazebo camera_source:=dual_rgb_sim
```

在语言终端输入：

```text
把红色方块抓到红色位置
```

同时以 `backend:=mock` 验证“回家”和“打开夹爪”等既有规则指令。

## 本次实际检查

- 单独运行 `pre_pick`、红方块上方和抓取点均返回 MoveIt `SUCCEEDED`；将控制器 `goal_time` 从 0.5 s 增至 2.0 s，消除正常末端收敛期间的误报。
- 已验证两路图像及 CameraInfo 持续发布，语言节点实际发布 `pick_place/red_cube/red_target_zone`。
- 红色区域下移若由运动层返回失败，任务明确停止；该路径没有成功回退或状态注入。




```Shell
ros2 launch embodied_bringup system.launch.py \
  backend:=gazebo \
  camera_source:=dual_rgb_sim \
  use_rviz:=true \
  gazebo_gui:=true \
  show_camera_views:=true \
  open_language_terminal:=true \
  gazebo_master_uri:=http://127.0.0.1:11501
```




# 2026-07-15 Day 6：Gazebo 系统初步集成与实际验收

## 1. 今日目标

将 Gazebo Classic 仿真、双 RGB 相机、视觉识别、中文语言交互、任务调度和 MoveIt 机械臂控制接入同一个启动入口，并进行实际运行验证。

## 2. 今日完成内容

### 2.1 Gazebo 一键启动集成

在 embodied_bringup/system.launch.py 中集成以下模块：

- Gazebo Classic
- EDULITE_A3 机械臂模型
- ros2_control 控制器
- MoveIt 2
- RViz
- 运动执行节点
- 任务管理节点
- 视觉感知节点
- 中文语言交互节点
- 双 RGB 相机图像窗口

当前一键启动支持的主要参数包括：

- backend:=mock|gazebo
- camera_source:=none|rgbd_sim|dual_rgb_sim|dual_usb
- use_rviz:=true|false
- gazebo_gui:=true|false
- show_camera_views:=true|false
- open_language_terminal:=true|false
- gazebo_master_uri:=指定端口

Gazebo 模式下能够正常启动机械臂、控制器、MoveIt、任务节点、视觉节点和语言节点。

### 2.2 双 RGB 相机接入

当前系统包含两路 RGB 相机：

- camera_main：用于观察工作台和操作区域；
- camera_aux：安装在机械臂末端附近，用于观察夹爪和近距离抓取区域。

两路相机能够持续发布：

- /camera_main/image_raw
- /camera_main/camera_info
- /camera_aux/image_raw
- /camera_aux/camera_info

实际运行时，感知节点能够持续接收到两路 RGB 图像及相机内参，并输出“相机输入已就绪”。

### 2.3 中文语言节点接入

现有语言节点能够识别以下中文指令：

- 移动到观察位置
- 回家
- 打开夹爪
- 关闭夹爪
- 把红色方块抓到红色位置

实际输入：

    移动到观察位置

语言节点成功发布：

    action=go_named_pose
    target=observe

实际输入：

    把红色方块抓到红色位置

语言节点成功发布：

    action=pick_place
    target=red_cube
    target_region=red_target_zone

说明中文输入到 TaskCommand 消息的转换已经打通。

### 2.4 视觉识别接入

任务节点能够接收 camera_main 图像，并使用 HSV 颜色空间识别红色目标。

当前识别流程包括：

1. BGR 图像转换为 HSV；
2. 使用两个红色色相范围进行阈值分割；
3. 使用形态学开运算去噪；
4. 提取红色轮廓；
5. 根据轮廓面积区分红色方块与红色目标区域；
6. 计算轮廓中心像素。

实际测试中，系统已经能够识别红色方块和红色目标区域。

当前默认 use_camera_localization=false，因此视觉识别虽然已经运行，但抓放执行阶段仍主要使用 pick_place.yaml 中的标定位姿。

视觉坐标直接驱动机械臂运动尚未完成最终标定和验证。

### 2.5 任务管理与机械臂控制接入

任务管理器能够接收语言节点发布的 TaskCommand，并调用现有运动接口：

- /motion/go_named_pose
- /motion/set_gripper
- /motion/move_to_pose

“移动到观察位置”指令能够经过以下链路驱动 Gazebo 中的机械臂运动：

    中文输入
    → language_node
    → /task_command
    → task_manager_node
    → motion_executor_node
    → MoveIt
    → Gazebo 机械臂

抓放任务状态机目前包含：

1. 打开夹爪；
2. 移动到方块上方；
3. 下移到方块；
4. 关闭夹爪；
5. 抬起；
6. 移动到目标区域上方；
7. 下移到目标区域；
8. 打开夹爪；
9. 抬起完成。

### 2.6 实际抓放验证结果

实际输入：

    把红色方块抓到红色位置

语言节点成功发布抓放任务，任务管理器开始调用机械臂和夹爪接口。

视觉模块能够识别红色方块和红色目标区域，机械臂也能够开始执行抓放流程。

但完整抓取和放置尚未成功，运动层返回：

    status=ABORTED
    moveit_error=FAILURE
    planning_time=0.000s

任务管理器能够明确报告抓放失败，没有使用目标瞬移、状态注入或伪造成功结果。

## 3. 今日阶段成果

今天已经初步打通以下系统主链路：

    中文自然语言输入
    → 语言解析
    → TaskCommand 发布
    → 任务调度
    → 相机图像接收
    → 红色目标识别
    → MoveIt 运动规划
    → Gazebo 机械臂和夹爪执行

当前已经实现自然语言、视觉感知、任务调度、MoveIt 和 Gazebo 仿真控制的初步系统集成。

## 4. 当前存在的问题

1. 完整抓取和放置尚未成功；
2. 当前工作台和操作区域布局不够合理；
3. 红色方块和目标区域的位置与机械臂工作空间仍需重新设计；
4. camera_aux 的安装位置和视角仍需调整；
5. TCP、夹爪中心和物体中心之间的偏移尚未精确标定；
6. 视觉坐标目前尚未真正替代 YAML 标定位姿；
7. 抓取与放置位姿仍可能存在 IK 或规划失败；
8. Gazebo 中真实接触夹取的稳定性仍需验证。

## 5. 代码状态

- 仓库：LiYuXv/embodied-arm-system
- 分支：feat/day6-system-integration
- 最新提交：05b3550e6a3a368e78fa8d67169ae69b3cdf8fac
- Pull Request：Draft PR #8
- 目标分支：main
- 当前未合并 main

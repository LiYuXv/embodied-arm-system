# 基于自然语言交互的机械臂具身操作系统设计与实现

本科毕业设计项目，面向 EDULITE_A3 机械臂，基于 ROS 2 Humble、MoveIt 2 和 RViz 构建自然语言交互、任务调度、视觉感知与机械臂控制一体化的具身操作系统。

当前项目已经完成基础平台适配、机械臂运动执行、中文规则指令解析、任务层解耦、夹爪控制、Gazebo Classic 双相机仿真和系统一键启动。后续将在此基础上扩展更稳健的目标检测、真实相机标定和大语言模型交互。

## 1. 当前系统架构

当前已经打通的控制链路：

```text
中文终端输入
    ↓
embodied_language
规则解析并发布 TaskCommand
    ↓ /task_command
embodied_task
任务调度与并发保护
    ↓
embodied_motion
MoveIt 运动规划 / 夹爪控制
    ↓
EDULITE_A3 + ros2_control
    ↓
RViz 仿真执行
```

当前视觉感知链路：

```text
RGB 图像 ─────────────┐
深度图像 ─────────────┼→ embodied_perception
相机内参 ─────────────┘        ↓
                         输入状态监测
                               ↓
                       /detected_objects
```

`embodied_perception` 保持输入状态监测与 `/detected_objects` 接口；抓放任务同时在任务层使用主相机的 HSV 红色分割、`CameraInfo` 内参和 YAML 外参，将红色轮廓中心投影到工作台平面，作为抓取/放置位置。相机未就绪时会安全回退到 YAML 标定位姿。

## 2. 已实现功能

### 2.1 EDULITE_A3 基础平台

- 完成 EDULITE_A3 ROS 2 工程编译与依赖配置；
- 成功启动 RViz、MoveIt 2 和 ros2_control；
- 完成机械臂模型加载、运动规划和仿真执行验证；
- 验证 `arm_controller`、`gripper_controller` 和 `joint_state_broadcaster` 正常运行。

### 2.2 运动执行层

- 提供统一命名位姿服务 `/motion/go_named_pose`；
- 提供夹爪控制服务 `/motion/set_gripper`；
- 支持机械臂与夹爪执行互斥保护；
- 当前已验证命名位姿：
  - `home`
  - `observe`
  - `ready`
  - `pre_pick`
- 当前已验证夹爪状态：
  - `open`
  - `close`

### 2.3 中文语言交互

当前使用规则解析器将中文指令转换为结构化 `TaskCommand`，支持以下类型的表达：

```text
回家
复位
回到初始位置
移动到观察位置
移动到准备位置
准备抓取
移动到预抓取位置
打开夹爪
关闭夹爪
把红色方块抓到红色位置
```

解析器能够处理常见空格和中英文标点。无法识别的指令不会触发机械臂动作。

### 2.4 任务调度层

- 订阅 `/task_command`；
- 将语言理解与机械臂运动执行解耦；
- 根据任务动作调用对应运动服务；
- 当前支持：
  - `go_named_pose`
  - `set_gripper`
  - `pick_place`（打开夹爪 → 抓取 → 放置 → 抬起）
- 在任务执行期间拒绝新的并发任务，避免多个指令同时占用运动执行器。

### 2.5 视觉感知骨架

- 订阅 RGB 图像；
- 订阅对齐后的深度图像；
- 订阅相机内参 `CameraInfo`；
- 监测三路相机数据是否到达；
- 统计各输入话题的消息数量；
- 无相机时仍可正常启动；
- 发布 `/detected_objects`；
- 输出消息优先继承实际 RGB 图像的 `frame_id`；
- 已通过模拟 RGB-D 消息完成接口验证。

### 2.6 系统一键启动

`embodied_bringup` 可以统一启动：

```text
MoveIt / ros2_control / RViz
→ motion_executor_node
→ task_manager_node
→ perception_node
→ language_node
```

## 3. 开发环境

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- MoveIt 2
- RViz 2
- ros2_control
- EDULITE_A3
- VS Code

## 4. 仓库结构

```text
embodied-arm-system/
├── README.md
├── docs/
│   ├── dev_logs/                   # 按日期记录的开发日志
│   └── EDULITE_A3平台搭建记录.md
├── papers/                         # 论文、综述与阅读记录
├── third_party/
│   └── EDULITE_A3/                 # 机械臂基础平台子模块
└── ros2_ws/
    └── src/
        ├── embodied_interfaces/    # 自定义消息、服务和动作接口
        ├── embodied_language/      # 中文指令解析与 TaskCommand 发布
        ├── embodied_task/          # 任务调度与并发保护
        ├── embodied_motion/        # 机械臂和夹爪运动执行
        ├── embodied_perception/    # RGB-D 输入与感知结果发布
        ├── embodied_scene_sync/    # 仿真场景同步预留模块
        └── embodied_bringup/       # 完整系统启动入口
```

## 5. 主要 ROS 2 接口

| 类型 | 名称 | 作用 |
|---|---|---|
| Topic | `/task_command` | 发布结构化语言任务 |
| Topic | `/detected_objects` | 发布感知目标列表 |
| Service | `/motion/go_named_pose` | 执行机械臂命名位姿 |
| Service | `/motion/set_gripper` | 控制夹爪开合 |
| Action | `/arm_controller/follow_joint_trajectory` | 执行机械臂关节轨迹 |
| Action | `/gripper_controller/follow_joint_trajectory` | 执行夹爪轨迹 |

主要自定义接口包括：

```text
TaskCommand.msg
DetectedObject.msg
DetectedObjectArray.msg
MoveNamedPose.srv
SetGripper.srv
MoveToPose.action
ExecuteTask.action
```

## 6. 获取与编译

克隆仓库并初始化子模块：

```bash
git clone --recurse-submodules git@github.com:LiYuXv/embodied-arm-system.git
cd embodied-arm-system
git submodule update --init --recursive
```

确保 EDULITE_A3 基础平台已经完成编译，并且当前终端能够找到 `el_a3_moveit_config` 等基础功能包。

编译本项目工作空间：

```bash
cd ~/embodied-arm-system/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 7. 启动完整系统

默认 mock 后端的一键启动：

```bash
cd ~/embodied-arm-system/ros2_ws
source install/setup.bash
ros2 launch embodied_bringup system.launch.py
```

默认情况下，系统会尝试使用 GNOME Terminal 打开独立的中文语言交互窗口。

不自动打开语言终端：

```bash
ros2 launch embodied_bringup system.launch.py \
  open_language_terminal:=false
```

此时可在另一个终端手动启动：

```bash
cd ~/embodied-arm-system/ros2_ws
source install/setup.bash
ros2 run embodied_language language_node
```

Gazebo Classic 双 RGB 仿真（固定工作台主相机和夹爪基座辅助相机）：

```bash
ros2 launch embodied_bringup system.launch.py \
  backend:=gazebo \
  camera_source:=dual_rgb_sim \
  use_rviz:=true \
  gazebo_gui:=true
```

可用启动参数如下：

| 参数 | 可选值 | 说明 |
|---|---|---|
| `backend` | `mock`、`gazebo` | mock 使用原有 ros2_control；Gazebo 使用 Classic 控制器 |
| `camera_source` | `none`、`rgbd_sim`、`dual_rgb_sim`、`dual_usb` | 选择唯一的相机/感知路线 |
| `use_rviz` | `true`、`false` | 是否启动 RViz |
| `open_language_terminal` | `true`、`false` | 是否打开终端语言节点 |
| `gazebo_gui` | `true`、`false` | 是否显示 Gazebo Classic 客户端 |

`dual_usb` 路线还支持 `camera_main_device`、`camera_aux_device`、`camera_main_frame_id` 和 `camera_aux_frame_id`；默认设备使用稳定的 `/dev/v4l/by-path/...` 路径。Gazebo 后端的 Classic 启动负责 `robot_state_publisher`、`controller_manager` 和感知节点；一键启动仅额外启动 MoveIt 规划层，避免重复节点。

启动后可输入：

```text
移动到观察位置
准备抓取
打开夹爪
关闭夹爪
回家
把红色方块抓到红色位置
```

抓放位姿、工作台高度及主相机外参保存在 `embodied_task/config/pick_place.yaml`。算法先在 BGR 图像上转换 HSV，以两个红色阈值区间分割并形态学去噪；按轮廓面积区分方块和目标区域，再由像素中心、`CameraInfo` 内参和已标定的相机到 `base_link` 外参构建射线，与工作台平面求交得到机器人基坐标。Gazebo 中若接触抓取不稳定，释放步骤会调用 `/gazebo/set_model_state` 将 `red_cube` 放入红色目标区，保证演示结果可重复。

抓放任务将任一必需位姿的 IK 或规划执行失败明确报告为任务失败，不会以命名位姿替代失败段。

输入 `exit` 或 `quit` 可退出语言交互节点。

## 8. 单独验证感知节点

启动感知节点：

```bash
cd ~/embodied-arm-system/ros2_ws
source install/setup.bash
ros2 run embodied_perception perception_node
```

查看检测结果话题：

```bash
ros2 topic info /detected_objects
ros2 topic echo /detected_objects --once
```

默认相机话题：

```text
RGB:        /camera/color/image_raw
Depth:      /camera/aligned_depth_to_color/image_raw
CameraInfo: /camera/color/camera_info
```

在三路相机数据都到达后，节点会输出输入已就绪状态；没有相机数据时，节点会保持运行并提示当前缺失的输入。

## 9. 自动测试

运行核心功能包测试：

```bash
cd ~/embodied-arm-system/ros2_ws
source install/setup.bash

colcon test \
  --packages-select \
  embodied_interfaces \
  embodied_motion \
  embodied_language \
  embodied_task \
  embodied_perception \
  embodied_bringup

colcon test-result --verbose
```

截至 2026-07-13，已完成的阶段性验证包括：

- 语言、任务和启动模块：35 项测试，0 error，0 failure，3 skipped；
- 感知模块：3 passed，1 skipped，0 failed；
- 中文指令到仿真机械臂与夹爪执行的完整链路验证；
- RGB、Depth、CameraInfo 模拟输入和感知输出坐标系验证。

## 10. 当前限制

- 语言模块目前采用规则解析，尚未接入大语言模型；
- HSV 颜色分割仅覆盖当前红色方块/红色区域场景，仍需扩展多颜色和遮挡处理；
- 单目平面求交依赖固定相机标定，真实部署前需要完成手眼标定；
- `/detected_objects` 尚未作为任务层的通用多目标输入；
- VLA 模型训练与微调尚未开始；
- 真实机械臂迁移与安全验证尚未完成。

## 11. 后续计划

1. 确定 RGB-D 仿真环境与相机方案；
2. 接入 `cv_bridge`，完成 ROS 图像到 OpenCV 图像转换；
3. 实现基础颜色目标分割；
4. 结合深度图和相机内参计算目标三维位置；
5. 发布实际 `DetectedObjectArray`；
6. 将视觉目标接入任务调度层；
7. 实现接近、抓取、抬起和放置任务状态机；
8. 将规则解析器升级为大语言模型任务规划接口；
9. 评估并接入 VLA 模型；
10. 将仿真系统迁移到真实 EDULITE_A3 机械臂。

## 12. 毕设目标

最终实现用户通过自然语言下达桌面操作指令，系统完成语言理解、目标感知、任务规划、运动控制和执行反馈，并分别在仿真环境与真实 EDULITE_A3 机械臂平台上完成验证。

详细开发过程见 [`docs/dev_logs`](docs/dev_logs/)。

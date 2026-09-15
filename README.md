# 基于自然语言交互的机械臂具身操作系统设计与实现

本科毕业设计项目，面向 EDULITE_A3 机械臂，基于 Ubuntu 22.04、ROS 2 Humble、MoveIt 2、RViz 2 与 Gazebo Classic 构建自然语言交互、视觉感知、任务调度和机械臂操作一体化系统，并逐步扩展到真实机械臂与 LeRobot/VLA 数据采集、训练和推理链路。

当前已完成三个核心里程碑：

> 中文指令 → 红蓝目标视觉定位 → MoveIt 运动规划 → Gazebo 夹取、搬运与放置 → 回到安全位姿

> 真实 EDULITE_A3 → L1-L7 零力矩手动示教 → 双 USB RGB → LeRobotDataset v3.0

> 双 Joy-Con → 双臂末端 XYZ + 姿态 + 夹爪 → 两台真实 EDULITE_A3 同时遥操

红色和蓝色方块均已在同一 Gazebo 场景中完成顺序抓放验收；单臂真机已完成数据采集与云端 Pi0.5 环境验证；双臂侧已完成基础层、遥操、上位机、LeRobot 与 VLA 软件适配，并通过双 Joy-Con 控制两台真实机械臂的现场验证。当前阶段转入双臂真实任务试操作、测试数据采集与后续双臂 VLA 训练/真机联调。

## 1. 当前系统架构

```text
中文终端输入
    ↓
embodied_language
规则解析并发布 TaskCommand
    ↓ /task_command
embodied_task ← /detected_objects
任务状态机、视觉目标选择与并发保护
    ↓
embodied_motion
MoveIt 规划、笛卡尔运动与夹爪控制
    ↓
EDULITE_A3 + ros2_control
    ↓
Gazebo Classic / RViz
```

视觉链路：

```text
camera_main RGB 图像 + CameraInfo + 相机外参
    ↓
embodied_perception
HSV 分割、轮廓筛选、像素反投影与平面求交
    ↓
red_cube / blue_cube / red_target_zone / blue_target_zone
    ↓ /detected_objects
embodied_task
```

真机 VLA 数据采集链路：

```text
真实 EDULITE_A3
    ↓
L1-L7 零力矩手动示教
    ↓
双 USB RGB @ 20 Hz
    ↓
state[t] = [x, y, z, rx, ry, rz, gripper]
    ↓
action[t] = state[t+5]
    ↓
LeRobotDataset v3.0
```

正常 Gazebo 抓放任务使用实时视觉坐标计算目标位姿。视觉结果缺失、过期或必要规划段失败时，任务会明确失败；默认不会使用固定世界坐标伪造成功。

详细完成记录见：

- [2026-07-30 红蓝方块抓放闭环完成](docs/dev_logs/2026-07-30-pick-place-completion.md)
- [2026-09-03 真机零力矩示教与 LeRobot 数据录制链路打通](docs/dev_logs/2026-09-03-real-robot-lerobot-recording.md)
- [2026-09-06 手柄真机数据链路与云端 Pi0.5 推理环境打通](docs/dev_logs/2026-09-06-gamepad-cloud-pi05.md)
- [2026-09-15 双臂 Joy-Con 真机遥操链路完成](docs/dev_logs/2026-09-15-dual-arm-joycon-teleoperation.md)

## 2. 已实现功能

### 2.1 EDULITE_A3 基础平台

- 完成 EDULITE_A3 ROS 2 工程和 MoveIt 2 配置接入；
- 支持 RViz 规划验证与 Gazebo Classic 动力学仿真；
- 验证 `arm_controller`、`gripper_controller` 和 `joint_state_broadcaster`；
- 使用安全初始姿态，避免全零伸展构型作为抓放规划起点。

### 2.2 中文语言交互

当前规则解析器将中文指令转换为结构化 `TaskCommand`，支持：

```text
回家
复位
回到初始位置
移动到观察位置
移动到准备位置
准备抓取
打开夹爪
关闭夹爪
把红色方块放到红色位置
把红色方块抓到红色区域
把蓝色方块放到蓝色位置
把蓝色方块移动到蓝色区域
```

解析器可处理常见空格和中英文标点；无法识别的输入不会触发机械臂动作。

### 2.3 视觉感知

- 使用固定工作台主相机 `camera_main` 获取 RGB 图像和 `CameraInfo`；
- 通过 HSV 阈值、形态学处理和轮廓面积区分红蓝方块与红蓝目标区域；
- 将像素中心反投影为空间射线，并与对应物体表面平面求交；
- 在 `base_link` 坐标系发布 `DetectedObjectArray`；
- 支持检测结果时间有效性检查；
- 红、蓝方块与两个目标区域可同时稳定识别。

### 2.4 任务调度与抓放状态机

`embodied_task` 订阅 `/task_command` 与 `/detected_objects`，支持：

- `go_named_pose`；
- `set_gripper`；
- `pick_place`；
- 任务执行期间的并发保护；
- 规划或执行失败时的明确失败反馈。

已验证抓放流程：

```text
安全位姿
→ 打开夹爪
→ 目标上方
→ 抓取接触位姿
→ 闭合夹爪
→ 抬升
→ 安全搬运
→ 目标区域上方
→ 释放
→ 撤离
→ 回 home
```

### 2.5 运动执行

- `/motion/go_named_pose`：执行命名关节位姿；
- `/motion/set_gripper`：控制夹爪开合；
- `/motion/move_to_pose`：通过 MoveIt 执行末端位姿目标；
- 支持不同阶段的速度和加速度缩放；
- 接近、抓取、抬升、搬运和放置使用连续规划与执行结果；
- 抓取和放置姿态允许 L6 根据底座转角进行补偿，而不是固定为 0。

### 2.6 Gazebo Classic 工作场景

- 双 RGB 仿真路线：固定主相机和腕部辅助相机；
- 紧凑薄托盘、红蓝动态方块和红蓝静态目标区域；
- 夹爪左右指爪使用受力控制的 mimic 插件；
- 方块具有质量、惯性、碰撞和摩擦参数；
- 使用独立 `GAZEBO_MASTER_URI`，减少旧实例冲突。

为提高 Gazebo Classic 搬运阶段的稳定性，仿真中提供了受距离阈值约束的临时抓取附着插件：夹爪到达目标并闭合后创建临时 fixed joint，释放阶段解除。该机制不会修改方块世界坐标，也不调用 `/gazebo/set_model_state`，但属于仿真稳定化措施，真实机械臂迁移时需要由实际夹持力、摩擦和接触反馈替代。

### 2.7 一键启动

`embodied_bringup` 可统一启动：

```text
Gazebo Classic / MoveIt / ros2_control / RViz
→ perception_node
→ motion_executor_node
→ task_manager_node
→ language_node
```

### 2.8 真实机械臂、LeRobot 与双臂遥操

真实 EDULITE_A3 已完成以下验证：

- CANable2 通过 SLCAN 建立 `can0`，上位机使用 SocketCAN；
- L1-L7 反馈与小角度动作测试通过；
- L1-L7 可在零力矩示教状态下手动拖动，包括夹爪 L7；
- 两路外接 USB 摄像头使用 `/dev/v4l/by-path/` 固定物理端口，避免 `/dev/videoX` 重编号造成 cam1/cam2 混乱；
- 测试集 `ela3_test_0903` 成功生成 LeRobotDataset v3.0；
- 双路 RGB 为 `720×720`、`20 Hz`；
- `observation.state` 为 7 维 `[x, y, z, rx, ry, rz, gripper]`；
- `observation.gripper_torque` 为 1 维；
- `action` 为 7 维；
- 逐帧验证确认 `action[t] = state[t+5]`，对应 `20 Hz` 下 `0.25 s` lookahead；
- 单测试 episode 共 `783` 帧、约 `39.1 s`，双路 MP4、Parquet 和 Meta 文件均正常生成。

当前测试集只用于链路验证，正式 VLA 训练数据将重新按统一任务和采集规范录制。

双臂扩展已完成以下现场验证：

- 左右两台 EDULITE_A3 可通过独立 `can0/can1` 同时连接；
- Joy-Con L/R 可分别绑定左右机械臂；
- 双臂末端 XYZ 平移、Roll/Pitch/Yaw 姿态与夹爪均可独立遥操；
- 单臂 Joy-Con R 姿态最终使用 `log3 → rotation-vector sign → exp3`，真机滚转/俯仰/偏航方向均正确；
- 双臂统一数据接口为 `state[14] + torque[2] + 3 cameras → action[14]`；
- 双臂 LeRobot 与 VLA 软件适配已完成，下一步进入真实任务试操作与测试 episode 采集。

## 3. 开发环境

ROS 2 / 仿真环境：

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- MoveIt 2
- RViz 2
- Gazebo Classic 11
- ros2_control
- EDULITE_A3
- VS Code

LeRobot / VLA 环境与 ROS 2 Python 环境隔离：

- Miniforge / Conda `lerobot` 环境
- Python 3.12
- LeRobot 0.6.2
- PyTorch 2.11.0 + CUDA 12.8
- RTX 4060 Laptop GPU（8 GB，本地用于验证，不作为正式 SmolVLA 训练平台）

## 4. 仓库结构

```text
embodied-arm-system/
├── README.md
├── docs/
│   ├── dev_logs/                         # 按日期记录的开发日志
│   ├── gazebo_classic_spike.md
│   └── EDULITE_A3平台搭建记录.md
├── papers/                               # 论文、综述与阅读记录
├── third_party/
│   └── EDULITE_A3/                       # 基础平台子模块
└── ros2_ws/
    └── src/
        ├── embodied_interfaces/          # 自定义消息、服务和动作接口
        ├── embodied_language/            # 中文指令解析
        ├── embodied_perception/          # 红蓝目标视觉定位
        ├── embodied_task/                # 抓放任务状态机
        ├── embodied_motion/              # MoveIt 与夹爪运动执行
        ├── embodied_bringup/             # 完整系统启动入口
        ├── gazebo_classic_sim/           # Classic 场景、控制器与相机
        ├── gazebo_classic_gripper_mimic/ # 指爪受力跟随插件
        └── gazebo_classic_grasp_attachment/ # 仿真搬运稳定化插件
```

## 5. 主要 ROS 2 接口

| 类型 | 名称 | 作用 |
|---|---|---|
| Topic | `/task_command` | 发布结构化语言任务 |
| Topic | `/detected_objects` | 发布红蓝方块与目标区域坐标 |
| Service | `/motion/go_named_pose` | 执行机械臂命名位姿 |
| Service | `/motion/set_gripper` | 控制夹爪开合 |
| Action | `/motion/move_to_pose` | 执行末端位姿目标 |
| Action | `/arm_controller/follow_joint_trajectory` | 执行机械臂关节轨迹 |
| Action | `/gripper_controller/follow_joint_trajectory` | 执行夹爪轨迹 |
| Service | `/grasp_attachment/set_enabled` | Gazebo 搬运阶段临时附着/释放 |

主要自定义接口：

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

编译基础平台后，编译本项目工作空间：

```bash
cd ~/embodied-arm-system/ros2_ws
source ~/embodied-arm-system/third_party/EDULITE_A3/el_a3_ros/install/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 7. 启动完整抓放系统

推荐的 Gazebo Classic 双 RGB 路线：

```bash
cd ~/embodied-arm-system/ros2_ws
source ~/embodied-arm-system/third_party/EDULITE_A3/el_a3_ros/install/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=124

ros2 launch embodied_bringup system.launch.py \
  backend:=gazebo \
  camera_source:=dual_rgb_sim \
  use_rviz:=true \
  gazebo_gui:=true \
  show_camera_views:=true \
  gazebo_master_uri:=http://127.0.0.1:11421
```

启动前应确认没有遗留的同项目 `gzserver` 实例，避免多个相机发布者向同一话题发送图像。

默认情况下，系统会尝试在 GNOME Terminal 中启动语言节点。也可以设置：

```bash
ros2 launch embodied_bringup system.launch.py \
  backend:=gazebo \
  camera_source:=dual_rgb_sim \
  open_language_terminal:=false
```

随后在另一个终端运行：

```bash
cd ~/embodied-arm-system/ros2_ws
source ~/embodied-arm-system/third_party/EDULITE_A3/el_a3_ros/install/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=124
ros2 run embodied_language language_node
```

输入：

```text
把红色方块放到红色位置
把蓝色方块放到蓝色位置
```

输入 `exit` 或 `quit` 可退出语言交互节点。

### 常用启动参数

| 参数 | 可选值 | 说明 |
|---|---|---|
| `backend` | `mock`、`gazebo` | 选择运动后端 |
| `camera_source` | `none`、`rgbd_sim`、`dual_rgb_sim`、`dual_usb` | 选择唯一相机路线 |
| `use_rviz` | `true`、`false` | 是否启动 RViz |
| `open_language_terminal` | `true`、`false` | 是否自动打开语言终端 |
| `gazebo_gui` | `true`、`false` | 是否显示 Gazebo 客户端 |
| `gazebo_master_uri` | URI | 独立 Gazebo Classic master |
| `show_camera_views` | `true`、`false` | 是否打开双路图像窗口 |

## 8. 运行状态检查

检查主相机是否只有一个发布者：

```bash
ros2 topic info /camera_main/image_raw -v
```

检查视觉结果：

```bash
ros2 topic echo /detected_objects --once
```

检查 Gazebo 实例：

```bash
pgrep -af "gzserver|gzclient"
```

正常情况下应只有一个 `gzserver`、一个 `gzclient` 和一个 `/camera_main/image_raw` 发布者。

## 9. 自动测试

```bash
cd ~/embodied-arm-system/ros2_ws
source ~/embodied-arm-system/third_party/EDULITE_A3/el_a3_ros/install/setup.bash
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

PR #8 合并前完成的最终验收包括：

- `embodied_task` 构建通过；
- 单元测试 `7 passed`；
- Python 编译检查通过；
- `git diff --check` 通过；
- 同一 Gazebo 场景中红色、蓝色任务顺序执行成功；
- 两个任务均完成视觉定位、夹取、抬升、搬运、释放、撤离和回 home。

## 10. 当前限制

- 语言模块仍为规则解析，尚未接入大语言模型；
- Gazebo 视觉算法主要针对当前红蓝颜色、固定尺寸与固定工作台；
- 单目平面求交依赖已标定的固定主相机与已知表面高度；
- 抓取附着插件属于 Gazebo Classic 稳定化方案，不能直接代表真实夹持物理；
- 尚未形成通用多物体、多任务规划接口；
- 单臂与双臂 LeRobot 数据接口均已打通，但正式双臂任务数据尚未批量采集；
- 双臂 VLA 软件适配已完成，但双臂模型训练、checkpoint 验证和真机协作执行尚未完成；
- 当前真机零位为人工近似校准，正式 FK/IK 精度实验和手眼标定前需进一步校准；
- 真机部分姿态下的重力补偿稳定性仍需继续验证与调参；
- 真实机械臂的完整语言/VLA 闭环任务和安全性实验尚未完成。

## 11. 下一阶段

1. 选择一个简单、可重复的双臂协作任务，先进行多次 Joy-Con 真机试操作；
2. 录制少量双臂测试 episode，核验三相机、`state[14]`、`torque[2]`、`action[14]` 和 task；
3. 固定物体、目标区、三相机语义、双臂起始姿态、成功标准与 episode 时长；
4. 批量采集正式双臂数据并进行数据质量检查；
5. 上传服务器，完成双臂 VLA 微调与离线推理验证；
6. 回传 checkpoint，验证双臂真机连续推理与协作任务成功率；
7. 设计规则解析与大语言模型任务规划的统一接口，并完成安全验证和实验统计；
8. 将系统实现、实验结果和局限整理为论文内容。

## 12. 毕设目标

最终实现用户通过自然语言下达桌面操作指令，系统完成语言理解、目标感知、任务规划、运动控制和执行反馈，并分别在仿真环境与真实 EDULITE_A3 机械臂平台上完成验证。

详细开发过程见 [`docs/dev_logs`](docs/dev_logs/)。
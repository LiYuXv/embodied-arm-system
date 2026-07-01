# 基于自然语言交互的机械臂具身操作系统设计与实现

## 1. 项目简介

本项目为本科毕业设计项目，目标是设计并实现一个基于自然语言交互的机械臂具身操作系统。

系统计划通过自然语言指令理解、视觉感知、任务规划与机械臂控制等模块，实现用户输入语言指令后，机械臂能够在仿真环境及真实平台中完成简单具身操作任务。

当前阶段已完成 EDULITE_A3 基础平台搭建，成功编译 ROS2 工程，并在 RViz / MoveIt2 中完成机械臂模型加载与基础运动规划测试。

## 2. 当前进展

- 完成项目 GitHub 仓库搭建；
- 引入 EDULITE_A3 机械臂基础平台；
- 完成 EDULITE_A3 ROS2 工程编译；
- 解决依赖安装、Pinocchio 缺失和 Git LFS 模型文件问题；
- 成功启动 RViz / MoveIt2 仿真环境；
- 完成 MoveIt2 中 Plan 与 Plan & Execute 基础运动测试。

## 3. 项目结构

```text
embodied-arm-system/
├── README.md
├── .gitignore
├── .gitmodules
├── docs/                     # 毕设文档、任务书、进度计划、平台搭建记录
├── papers/                   # 文献资料与阅读笔记
├── third_party/
│   └── EDULITE_A3/            # 老师提供的机械臂基础平台
└── ros2_ws/
    └── src/                   # 后续存放本人编写的 ROS2 功能包
```

## 4. EDULITE_A3 基础平台

EDULITE_A3 平台代码位于：

```text
third_party/EDULITE_A3
```

其中主要使用：

```text
third_party/EDULITE_A3/el_a3_ros
```

该目录本身是一个 ROS2 工程，包含机械臂描述文件、ros2_control 硬件接口、MoveIt2 配置和遥操作相关功能包。

当前已完成基础平台链路：

```text
EDULITE_A3 ROS2 工程编译
→ RViz / MoveIt2 启动
→ RobotModel 正常显示
→ MoveIt2 基础规划与执行测试
```

## 5. 基础平台启动方式

进入 EDULITE_A3 的 ROS2 工程目录：

```bash
cd ~/embodied-arm-system/third_party/EDULITE_A3/el_a3_ros
```

加载 ROS2 Humble 环境：

```bash
source /opt/ros/humble/setup.bash
```

编译工程：

```bash
colcon build --symlink-install
```

加载当前工作空间：

```bash
source install/setup.bash
```

启动 MoveIt2 仿真环境：

```bash
ros2 launch el_a3_moveit_config demo.launch.py
```

## 6. 后续开发计划

下一阶段将在当前基础平台之上，创建本人自己的 ROS2 功能包，逐步实现：

- 机械臂控制封装；
- MoveIt2 运动规划接口调用；
- 简单目标位姿输入下的机械臂运动控制；
- 自然语言指令解析模块；
- 视觉感知与目标定位模块；
- 语言、视觉与机械臂控制的系统联调。

后续整体技术路线为：

```text
自然语言输入
→ 指令解析
→ 目标物体与目标区域识别
→ 任务规划
→ MoveIt2 运动规划
→ 仿真执行
→ 真实机械臂迁移
```

## 7. 毕设目标

最终目标是完成一个基于自然语言交互的机械臂具身操作系统，实现用户通过自然语言指令驱动机械臂完成简单桌面操作任务，并在仿真环境和真实机械臂平台上进行验证。
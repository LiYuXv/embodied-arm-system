# 2026-07-10 MoveIt2 位姿规划与 IK 修复开发记录

## 1. 今日目标

在前一日已完成 `/motion/go_named_pose` 关节空间运动接口的基础上，继续完善 `embodied_motion`，增加基于 MoveIt2 的末端位姿规划与执行能力。

本阶段目标是打通从本人系统层输入 `PoseStamped` 目标，到 MoveIt2 完成逆运动学求解、路径规划和控制器执行的完整链路，为后续视觉定位和任务规划模块提供统一的笛卡尔空间运动接口。

---

## 2. 已完成内容

### 2.1 完善 `motion_config.yaml`

扩展配置文件：

```text
ros2_ws/src/embodied_motion/embodied_motion/config/motion_config.yaml
```

新增配置内容包括：

- MoveIt2 规划组：`arm`
- 规划坐标系：`base_link`
- 对外末端执行器：`end_effector`
- MoveGroup Action：`/move_action`
- 轨迹执行 Action：`/execute_trajectory`
- 规划器：`RRTConnect`
- 规划时间和重规划参数
- 速度、加速度缩放范围
- 位置和姿态容差
- `home`、`observe`、`ready` 预设位姿
- 夹爪 `open`、`close` 位置
- 输入坐标系和四元数合法性检查参数

其中机械臂规划仍使用 6 个关节：

```text
L1_joint
L2_joint
L3_joint
L4_joint
L5_joint
L6_joint
```

夹爪继续由独立的 `L7_joint` 和 `gripper_controller` 控制。

---

### 2.2 增加 `/motion/move_to_pose` 动作接口

在 `motion_executor_node.py` 中增加：

```text
/motion/move_to_pose
```

该接口使用：

```text
embodied_interfaces/action/MoveToPose
```

实现功能包括：

- 接收 `PoseStamped` 目标位姿
- 检查目标坐标系是否合法
- 检查四元数是否有效并进行归一化
- 设置速度和加速度缩放
- 创建 MoveIt2 `MoveGroup` Action 目标
- 等待规划与执行结果
- 返回 MoveIt 错误码和执行状态
- 支持取消正在执行的动作
- 防止同一时刻重复提交多个运动任务

节点改为使用 `MultiThreadedExecutor` 和可重入回调组，保证服务、动作和 MoveGroup 回调可以正常并发处理。

---

### 2.3 封装 MoveIt2 目标构造逻辑

新增 MoveIt2 目标构造模块，用于统一生成：

- 规划组信息
- 起始状态
- 位置约束
- 姿态约束
- 规划管线参数
- 速度和加速度缩放参数

同时增加单元测试，检查目标位姿约束和 MoveGroup Goal 的构造结果，避免运动节点中直接拼装大量 MoveIt 消息。

---

### 2.4 增加 IK 与坐标变换诊断工具

为排查末端位姿规划失败问题，增加了以下诊断内容：

- 当前机器人状态的 FK 结果
- TF 中末端执行器位姿
- MoveIt `/compute_ik` 服务调用
- 当前 TCP 位姿的 IK 回代
- 指定 TCP 位姿的 IK 测试
- TCP 沿 Z 方向小距离偏移的 IK 测试
- 开启和关闭碰撞检查时的 IK 对比

通过 FK、TF 和 IK 交叉验证，可以区分目标位姿错误、坐标变换错误和运动学插件配置错误。

---

### 2.5 修复 MoveIt2 运动学配置

排查发现原配置中使用：

```text
pick_ik/PickIkPlugin
```

但当前 ROS2 Humble 环境中没有安装 `pick_ik` 软件包和插件库，导致 MoveIt 无法加载对应运动学求解器。

因此将机械臂运动学求解器改为系统已有的：

```text
kdl_kinematics_plugin/KDLKinematicsPlugin
```

同时发现 SRDF 中机械臂规划链的末端设置为固定 TCP：

```text
end_effector
```

KDL 需要将规划链末端设置为最后一个主动关节对应的 link，因此将 `arm` 规划组的 tip 修改为：

```text
l5_l6_urdf_asm
```

`end_effector` 仍保留为系统对外使用的 TCP。运动层在构造 MoveIt 目标时，将用户输入的 TCP 位姿换算为主动末端 link 的目标位姿，从而兼顾外部接口语义和 KDL 求解要求。

---

### 2.6 完成位姿规划与执行验证

完成以下测试：

```text
当前 TCP 位姿 IK
显式 TCP 位姿 IK
TCP 向下偏移 3 cm 的 IK
开启碰撞检查的 IK
关闭碰撞检查的 IK
```

各项 IK 测试均能得到有效解。

FK 与 TF 对比结果为：

```text
位置误差：0
四元数点积：1
```

说明 MoveIt FK 与 TF 坐标变换结果一致。

随后通过 `/motion/move_to_pose` 发送 TCP 相对移动约 5 mm 的目标，MoveIt 完成规划和执行，动作结果为：

```text
Goal status: SUCCEEDED
MoveIt error code: SUCCESS
```

单元测试和代码检查结果：

```text
pytest：3 passed
flake8：passed
```

---

## 3. 今日问题与解决

### 3.1 问题：MoveIt 无法加载 `pick_ik`

原 `kinematics.yaml` 配置了：

```text
pick_ik/PickIkPlugin
```

但当前系统中不存在该插件，因此 MoveIt 无法正常完成逆运动学求解。

解决方法：

- 使用 ROS2 Humble 已安装的 KDL 插件
- 将 KDL 作为当前机械臂的基础运动学求解器
- 保留后续安装并验证 `pick_ik` 后再切换的可能性

---

### 3.2 问题：KDL 规划链末端设置错误

原 SRDF 将固定的 `end_effector` 作为机械臂规划链 tip，导致 KDL 在 FK 到 IK 回代过程中不能稳定构造目标。

解决方法：

- 将 `arm` 规划组 tip 改为最后一个主动 link：`l5_l6_urdf_asm`
- 对外仍使用 `end_effector` 作为 TCP
- 在运动层中完成 TCP 目标到主动 link 目标的位姿转换

---

### 3.3 问题：安装文件诊断出现误报

原诊断脚本使用：

```bash
find -type f
```

检查安装目录中的配置文件，但项目使用：

```bash
colcon build --symlink-install
```

配置文件在安装目录中是符号链接，因此脚本将已存在的文件误报为缺失。

解决方法：

- 修改诊断逻辑，使其能够识别普通文件和符号链接
- 通过实际文件读取和运行时加载结果确认配置已正确安装

---

## 4. 当前阶段成果

当前已完成从系统运动接口到 MoveIt2 位姿规划和控制器执行的完整链路：

```text
/motion/move_to_pose
  ↓
motion_executor_node
  ↓
MoveItGoalBuilder
  ↓
/move_action
  ↓
move_group + KDL IK
  ↓
/arm_controller/follow_joint_trajectory
  ↓
RViz / 机械臂执行
```

运动执行层目前同时支持：

```text
/motion/go_named_pose   关节空间预设位姿
/motion/move_to_pose    笛卡尔空间末端位姿
```

这说明后续视觉模块可以直接提供目标位姿，任务规划模块也可以通过统一 Action 调用机械臂运动能力。

---

## 5. 下一步计划

### 5.1 稳定运动执行层

- 继续验证 `home`、`ready`、`zero` 等安全位姿
- 检查 MoveIt 规划和实际控制器执行的一致性
- 完善异常状态和执行结果日志

### 5.2 完成夹爪控制

- 验证 `L7_joint` 的实际开闭方向
- 增加夹爪打开和闭合接口
- 检查 RViz 中夹爪模型是否随 L7 状态变化

### 5.3 准备语言模块接入

- 将基础中文指令映射到命名位姿
- 将目标位姿任务映射到 `/motion/move_to_pose`
- 开始构建语言到运动执行的最小闭环

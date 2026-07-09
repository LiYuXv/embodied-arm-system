
# 2026-07-09 运动执行层开发记录

## 1. 今日目标

在已有 EDULITE_A3 MoveIt2 / ros2_control 基础平台之上，开始搭建本人毕设系统的正式 ROS2 功能包结构，并优先完成机械臂运动执行层的初步封装。

本阶段目标不是临时 demo，而是为后续语言模块、视觉模块和任务规划模块提供稳定的运动执行接口。

---

## 2. 已完成内容

### 2.1 创建正式 ROS2 功能包结构

在 `ros2_ws/src` 下创建了以下功能包：

- `embodied_interfaces`
- `embodied_language`
- `embodied_motion`
- `embodied_task`
- `embodied_perception`
- `embodied_scene_sync`
- `embodied_bringup`

其中今日重点完成：

- `embodied_interfaces`
- `embodied_motion`

---

### 2.2 完成 `embodied_interfaces` 接口包

创建了自定义消息、服务和动作接口。

#### 消息接口

- `TaskCommand.msg`
- `DetectedObject.msg`
- `DetectedObjectArray.msg`

#### 服务接口

- `MoveNamedPose.srv`

#### 动作接口

- `MoveToPose.action`
- `ExecuteTask.action`

接口已编译成功，并通过以下命令验证：

```bash
ros2 interface show embodied_interfaces/msg/TaskCommand
ros2 interface show embodied_interfaces/msg/DetectedObject
ros2 interface show embodied_interfaces/msg/DetectedObjectArray
ros2 interface show embodied_interfaces/srv/MoveNamedPose
ros2 interface show embodied_interfaces/action/MoveToPose
ros2 interface show embodied_interfaces/action/ExecuteTask
```

---

### 2.3 完成 `embodied_motion` 基础配置

创建配置文件：

```text
ros2_ws/src/embodied_motion/embodied_motion/config/motion_config.yaml
```

配置内容包括：

- 机器人名称：`edulite_a3`
- 基坐标系：`base_link`
- 机械臂控制器：`/arm_controller/follow_joint_trajectory`
- 夹爪控制器：`/gripper_controller/follow_joint_trajectory`
- 机械臂关节名
- 预设位姿：`home`、`observe`
- 默认运动时间和速度参数

---

### 2.4 确认 EDULITE_A3 控制器状态

通过以下命令确认控制器状态：

```bash
ros2 control list_controllers
```

结果显示：

```text
joint_state_broadcaster active
arm_controller active
gripper_controller active
```

通过以下命令确认 `arm_controller` 实际控制的关节：

```bash
ros2 param get /arm_controller joints
```

结果为：

```text
L1_joint
L2_joint
L3_joint
L4_joint
L5_joint
L6_joint
```

注意：`/joint_states` 中存在 `L7_joint`，但 `arm_controller` 实际只接收 6 个关节，因此 `motion_config.yaml` 中最终使用 6 个机械臂关节。

---

### 2.5 完成 `motion_executor_node`

创建节点：

```text
ros2_ws/src/embodied_motion/embodied_motion/motion_executor_node.py
```

实现功能：

- 读取 `motion_config.yaml`
- 打印机器人与控制器配置
- 创建 `FollowJointTrajectory` action client
- 连接 `/arm_controller/follow_joint_trajectory`
- 连接 `/gripper_controller/follow_joint_trajectory`
- 创建服务 `/motion/go_named_pose`
- 根据服务请求发送预设关节位姿

---

### 2.6 完成预设位姿执行测试

启动节点：

```bash
ros2 run embodied_motion motion_executor_node
```

调用 `observe` 位姿：

```bash
ros2 service call /motion/go_named_pose embodied_interfaces/srv/MoveNamedPose "{pose_name: 'observe'}"
```

执行成功，日志显示：

```text
Motion goal accepted by arm controller
Motion execution succeeded
```

调用 `home` 位姿：

```bash
ros2 service call /motion/go_named_pose embodied_interfaces/srv/MoveNamedPose "{pose_name: 'home'}"
```

执行成功，日志显示：

```text
Motion goal accepted by arm controller
Motion execution succeeded
```

---

## 3. 今日问题与解决

### 3.1 问题：轨迹目标被 `arm_controller` 拒绝

最初参考 `/joint_states` 中的关节列表，误将 7 个关节写入轨迹目标：

```text
L1_joint
L2_joint
L4_joint
L3_joint
L5_joint
L6_joint
L7_joint
```

发送轨迹后，控制器返回：

```text
Motion goal rejected by arm controller
```

排查后发现，`arm_controller` 实际只控制 6 个关节：

```text
L1_joint
L2_joint
L3_joint
L4_joint
L5_joint
L6_joint
```

因此修改 `motion_config.yaml`，将机械臂控制关节改为 6 个，并调整 `home`、`observe` 的 `positions` 长度与顺序，问题解决。

---

## 4. 当前阶段成果

当前已完成从本人系统层到 EDULITE_A3 底层控制器的运动执行链路：

```text
/motion/go_named_pose
  ↓
motion_executor_node
  ↓
/arm_controller/follow_joint_trajectory
  ↓
arm_controller
  ↓
RViz / MoveIt2 中机械臂执行
```

这说明机械臂运动执行层已具备基础可用能力，后续语言模块和任务规划模块可以通过统一接口调用该运动层。

---

## 5. 下一步计划

### 5.1 完善 `embodied_motion`

- 增加夹爪控制接口
- 增加更多安全位姿
- 后续封装笛卡尔空间目标位姿执行

### 5.2 开始 `embodied_language`

- 支持中文输入
- 解析“回到初始位置”“移动到观察位置”等基础指令
- 调用 `/motion/go_named_pose`

### 5.3 后续进入仿真视觉闭环

- 构建带桌面、目标物体和 RGB-D 相机的仿真环境
- 实现目标检测和三维定位
- 将语言、视觉、运动执行模块串联起来

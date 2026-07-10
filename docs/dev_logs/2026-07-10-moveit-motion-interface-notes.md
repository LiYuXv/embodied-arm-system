# 2026-07-10 MoveIt2 运动接口与配置确认记录

## 1. 记录目的

本记录用于保存 EDULITE_A3 当前 MoveIt2、控制器、规划组、末端执行器、命名位姿与规划参数的实际运行结果，避免后续重复执行命令排查。

---

## 2. MoveIt2 节点

执行：

```bash
ros2 node list | grep move
```

结果：

```text
/move_group
/move_group_private_100847066365344
/moveit_simple_controller_manager
```

主要规划节点为：

```text
/move_group
```

---

## 3. SRDF 语义配置

执行：

```bash
ros2 param get /move_group robot_description_semantic
```

### 3.1 机械臂规划组

```text
planning_group: arm
base_link: base_link
tip_link: end_effector
```

对应 SRDF：

```xml
<group name="arm">
    <chain base_link="base_link" tip_link="end_effector"/>
</group>
```

机械臂组由 `L1_joint` 至 `L6_joint` 构成。

### 3.2 夹爪规划组

```text
planning_group: gripper
joint: L7_joint
```

对应 SRDF：

```xml
<group name="gripper">
    <joint name="L7_joint"/>
</group>
```

### 3.3 末端执行器

```text
end_effector_name: ee
parent_link: end_effector
parent_group: arm
group: gripper
```

对应 SRDF：

```xml
<end_effector name="ee" parent_link="end_effector" group="gripper" parent_group="arm"/>
```

---

## 4. SRDF 中已有的命名位姿

### 4.1 home

```yaml
L1_joint: 0.0
L2_joint: 0.785
L3_joint: -0.785
L4_joint: 0.0
L5_joint: 0.0
L6_joint: 0.0
```

### 4.2 ready

```yaml
L1_joint: 0.0
L2_joint: 0.785
L3_joint: -1.57
L4_joint: 0.0
L5_joint: 0.785
L6_joint: 0.0
```

### 4.3 zero

```yaml
L1_joint: 0.0
L2_joint: 0.0
L3_joint: 0.0
L4_joint: 0.0
L5_joint: 0.0
L6_joint: 0.0
```

### 4.4 gripper open

```yaml
L7_joint: 1.5708
```

### 4.5 gripper close

```yaml
L7_joint: 0.0
```

---

## 5. 当前 Action 接口

执行：

```bash
ros2 action list -t
```

结果：

```text
/arm_controller/follow_joint_trajectory [control_msgs/action/FollowJointTrajectory]
/execute_trajectory [moveit_msgs/action/ExecuteTrajectory]
/gripper_controller/follow_joint_trajectory [control_msgs/action/FollowJointTrajectory]
/move_action [moveit_msgs/action/MoveGroup]
```

### 5.1 MoveIt2 规划与执行接口

```text
/move_action
类型：moveit_msgs/action/MoveGroup
服务端：/move_group
```

执行：

```bash
ros2 action info /move_action
```

结果：

```text
Action clients: /rviz2
Action servers: /move_group
```

`/move_action` 可接收规划请求，并通过 `planning_options.plan_only` 决定只规划或规划后执行。

### 5.2 已规划轨迹执行接口

```text
/execute_trajectory
类型：moveit_msgs/action/ExecuteTrajectory
```

该接口用于执行已经生成的 `RobotTrajectory`。

### 5.3 机械臂底层控制接口

```text
/arm_controller/follow_joint_trajectory
类型：control_msgs/action/FollowJointTrajectory
控制关节：L1_joint 至 L6_joint
```

### 5.4 夹爪底层控制接口

```text
/gripper_controller/follow_joint_trajectory
类型：control_msgs/action/FollowJointTrajectory
控制关节：L7_joint
action_ns: follow_joint_trajectory
```

确认命令：

```bash
ros2 param get /move_group moveit_simple_controller_manager.gripper_controller.joints
ros2 param get /move_group moveit_simple_controller_manager.gripper_controller.action_ns
```

结果：

```text
['L7_joint']
follow_joint_trajectory
```

---

## 6. 当前规划与逆运动学参数

执行：

```bash
ros2 param get /move_group default_planning_pipeline
ros2 param get /move_group move_group.arm.default_planner_config
ros2 param get /move_group arm.kinematics_solver
ros2 param get /move_group arm.position_threshold
ros2 param get /move_group arm.orientation_threshold
```

结果：

```yaml
default_planning_pipeline: move_group
default_planner_config: RRTConnect
kinematics_solver: pick_ik/PickIkPlugin
position_threshold: 0.001
orientation_threshold: 0.01
```

说明：

- `RRTConnect`：当前机械臂组的默认规划器配置。
- `pick_ik/PickIkPlugin`：当前逆运动学求解器。
- `position_threshold = 0.001`：位置求解阈值，单位为米，即 1 mm。
- `orientation_threshold = 0.01`：姿态求解阈值，单位按 Pick IK 配置解释。
- `default_planning_pipeline` 的运行值实际返回 `move_group`，后续实现时先保留原始结果，不凭经验改写；如需设置 `pipeline_id`，应继续核对 MoveIt2 配置文件中的实际 pipeline 名称。

---

## 7. 当前正式运动层设计结论

### 7.1 固定命名位姿

```text
/motion/go_named_pose
  -> 生成关节目标约束
  -> /move_action
  -> MoveIt2 规划、碰撞检测与执行
```

固定安全位姿至少保留：

```text
home
ready
zero
observe（项目自定义）
```

### 7.2 任意末端位姿

```text
/motion/move_to_pose
  -> geometry_msgs/PoseStamped
  -> 位置约束 + 姿态约束
  -> /move_action
  -> IK、碰撞检测、轨迹规划与执行
```

使用参数：

```yaml
planning_group: arm
planning_frame: base_link
end_effector_link: end_effector
move_group_action: /move_action
planner_id: RRTConnect
kinematics_solver: pick_ik/PickIkPlugin
position_tolerance: 0.001
orientation_tolerance: 0.01
```

### 7.3 夹爪控制

```text
/motion/set_gripper
  -> /gripper_controller/follow_joint_trajectory
  -> L7_joint
```

预设值：

```yaml
open: 1.5708
close: 0.0
```

---

## 8. 常用排查命令

```bash
# 查看 MoveIt2 节点
ros2 node list | grep move

# 查看 MoveIt2 参数
ros2 param list /move_group

# 查看 SRDF
ros2 param get /move_group robot_description_semantic

# 查看 Action 接口
ros2 action list -t

# 查看 MoveGroup 服务端
ros2 action info /move_action

# 查看夹爪关节
ros2 param get /move_group moveit_simple_controller_manager.gripper_controller.joints

# 查看规划器与 IK
ros2 param get /move_group move_group.arm.default_planner_config
ros2 param get /move_group arm.kinematics_solver

# 查看求解阈值
ros2 param get /move_group arm.position_threshold
ros2 param get /move_group arm.orientation_threshold
```

---

## 9. 下一步

1. 按本记录重构 `motion_config.yaml`。
2. 将 `/motion/go_named_pose` 改为通过 `/move_action` 完成规划和执行。
3. 实现 `/motion/move_to_pose` 自定义 Action 服务端。
4. 实现夹爪开合控制接口。
5. 增加忙碌状态、参数校验、错误码映射和取消处理。

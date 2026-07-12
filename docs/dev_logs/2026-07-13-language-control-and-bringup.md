
# 2026-07-13 规则语言控制与系统一键启动

## 1. 本次目标

完成机械臂具身操作系统的基础自然语言交互链路，使用户能够通过中文指令控制机械臂命名位姿和夹爪，并提供完整系统的一键启动入口。

本次实现的主要链路为：

```text
中文终端输入
→ 规则指令解析
→ TaskCommand 结构化消息
→ 运动层服务
→ MoveIt / gripper_controller
→ 仿真机械臂执行
```

## 2. 命名位姿扩展

在 `motion_config.yaml` 中新增 `pre_pick` 命名位姿，用于表示预抓取准备状态。

当前 `pre_pick` 暂时与 `ready` 使用相同的关节位置，待后续桌面、相机和目标物体场景确定后重新标定。

已验证：

```text
/motion/go_named_pose
pose_name: pre_pick
```

执行结果：

```text
status=SUCCEEDED
moveit_error=SUCCESS
```

## 3. 中文规则解析器

新增：

```text
embodied_language/command_parser.py
```

实现基础中文指令到标准任务命令的映射。

当前支持的机械臂命名位姿指令包括：

* 回家、复位、回到初始位置；
* 移动到观察位置；
* 移动到准备位置；
* 准备抓取、移动到预抓取位置。

解析结果示例：

```text
输入：准备抓取
action: go_named_pose
target: pre_pick
```

当前支持的夹爪指令包括：

* 打开夹爪、张开夹爪、松开夹爪；
* 关闭夹爪、闭合夹爪、合上夹爪。

解析结果示例：

```text
输入：打开夹爪
action: set_gripper
target: open
```

解析器会忽略空格和常见中英文标点。无法识别的指令返回 `None`，不会执行机械臂动作。

## 4. 解析器单元测试

新增：

```text
embodied_language/test/test_command_parser.py
```

测试内容包括：

* 命名位姿指令解析；
* 夹爪指令解析；
* 中文标点和空格归一化；
* 空指令处理；
* 未知指令拒绝；
* `action` 和 `target` 字段映射。

测试结果：

```text
26 passed
```

## 5. 语言交互节点

新增：

```text
embodied_language/language_node.py
```

节点名称：

```text
language_node
```

主要功能：

1. 在独立线程中读取终端中文输入；
2. 使用 `CommandParser` 解析指令；
3. 构造并发布 `TaskCommand`；
4. 根据 `action` 调用对应的运动层服务；
5. 输出执行成功或失败信息；
6. 支持通过 `exit` 或 `quit` 正常退出。

发布话题：

```text
/task_command
```

命名位姿服务：

```text
/motion/go_named_pose
```

夹爪服务：

```text
/motion/set_gripper
```

发布的消息示例：

```yaml
raw_text: 准备抓取
action: go_named_pose
target: pre_pick
target_region: ''
source: terminal_rules
```

目前语言节点会同时发布 `TaskCommand` 并直接调用运动层服务。这是当前基础闭环方案，后续增加任务调度模块后，将改为由任务层订阅 `TaskCommand` 并统一调度运动。

## 6. 夹爪运动接口

新增服务接口：

```text
embodied_interfaces/srv/SetGripper.srv
```

接口定义：

```text
string position_name
---
bool success
string message
```

运动层新增服务：

```text
/motion/set_gripper
```

当前支持：

```text
open
close
```

其中：

```text
open  → L7_joint = 1.5708
close → L7_joint = 0.0
```

运动层通过以下 Action 控制夹爪：

```text
/gripper_controller/follow_joint_trajectory
```

测试结果：

```text
open:
status=SUCCEEDED
controller_error=SUCCESSFUL

close:
status=SUCCEEDED
controller_error=SUCCESSFUL
```

非法参数 `half` 能够被正确拒绝，并返回当前可用状态列表。

夹爪动作与机械臂动作共用运动锁，能够避免多个动作同时占用执行器。

## 7. 中文控制联调

已完成以下中文指令联调：

```text
打开夹爪
准备抓取
关闭夹爪
回家
移动到观察位置
```

验证结果：

* 中文指令能够正确解析；
* `/task_command` 能够正常发布；
* 命名位姿服务能够正常执行；
* 夹爪服务能够正常执行；
* MoveIt 返回 `SUCCESS`；
* 夹爪控制器返回 `SUCCESSFUL`；
* 仿真机械臂和夹爪动作正常。

## 8. 系统一键启动

在 `embodied_bringup` 中新增：

```text
launch/system.launch.py
```

总启动文件统一启动：

* MoveIt；
* ros2_control 与机械臂控制器；
* RViz；
* `motion_executor_node`；
* `language_node`。

由于 `language_node` 需要使用终端标准输入，总启动文件会通过 GNOME Terminal 打开独立的语言交互窗口。

启动命令：

```bash
ros2 launch embodied_bringup system.launch.py
```

关闭自动语言终端：

```bash
ros2 launch embodied_bringup system.launch.py \
  open_language_terminal:=false
```

一键启动功能已完成实际验证。

## 9. 已知非致命启动日志

系统启动时可能出现以下日志：

1. `/get_planning_scene` 服务响应超时；
2. RViz `InteractiveMarkerDisplay` 插件工厂重复注册；
3. Octomap 未指定分辨率；
4. 未配置三维传感器 Octomap 更新插件。

当前这些日志不影响：

* RViz 显示；
* MoveIt 规划；
* 机械臂执行；
* 夹爪控制；
* 中文指令控制。

Octomap 和三维传感器插件将在后续视觉感知阶段，结合仿真深度相机或真实相机统一配置。

## 10. 当前成果

当前系统已具备以下基础能力：

```text
中文规则指令
→ 结构化 TaskCommand
→ 命名位姿控制
→ 夹爪开合控制
→ MoveIt 仿真执行
```

同时具备一键启动入口，为后续任务调度、视觉感知、目标抓取和大语言模型接入提供了基础。

## 11. 后续工作

下一阶段计划：

1. 增加 `embodied_task` 任务调度节点；
2. 将语言层与运动层解耦；
3. 由任务层订阅 `/task_command` 并统一调用运动服务；
4. 构建桌面、目标物体和相机仿真场景；
5. 接入目标检测与目标位姿估计；
6. 实现完整的抓取任务状态机。

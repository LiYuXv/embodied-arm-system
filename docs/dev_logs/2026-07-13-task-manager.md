
# 2026-07-13 任务调度模块与系统分层解耦

## 1. 本次目标

增加 `embodied_task` 任务调度模块，将语言解析与机械臂运动执行解耦，形成正式的分层系统结构。

本次完成后的执行链路为：

```text
中文终端输入
→ embodied_language
→ /task_command
→ embodied_task
→ embodied_motion
→ MoveIt / gripper_controller
→ 仿真机械臂执行
```

## 2. 任务管理节点

新增：

```text
embodied_task/task_manager_node.py
```

节点名称：

```text
task_manager_node
```

订阅话题：

```text
/task_command
```

任务节点接收 `TaskCommand`，并根据 `action` 字段调用对应的运动服务。

当前支持的任务动作：

```text
go_named_pose
set_gripper
```

对应服务：

```text
/motion/go_named_pose
/motion/set_gripper
```

## 3. 命名位姿任务调度

当任务命令为：

```yaml
action: go_named_pose
target: observe
```

任务管理节点调用：

```text
/motion/go_named_pose
```

已验证以下位姿任务能够正常执行：

```text
home
observe
ready
pre_pick
```

实际测试结果：

```text
status=SUCCEEDED
moveit_error=SUCCESS
```

## 4. 夹爪任务调度

当任务命令为：

```yaml
action: set_gripper
target: open
```

或：

```yaml
action: set_gripper
target: close
```

任务管理节点调用：

```text
/motion/set_gripper
```

已验证：

```text
open  → 执行成功
close → 执行成功
```

控制器返回：

```text
status=SUCCEEDED
controller_error=SUCCESSFUL
```

## 5. 语言层与运动层解耦

修改 `language_node.py`，删除语言节点中的运动服务客户端和运动结果处理逻辑。

语言模块现在只负责：

```text
接收中文输入
→ 规则解析
→ 构造 TaskCommand
→ 发布 /task_command
```

语言节点不再直接调用机械臂运动服务。

任务执行统一由 `task_manager_node` 负责。

## 6. 任务并发保护

任务管理节点增加：

```text
task_in_progress
```

当已有机械臂任务正在执行时，新的任务命令会被拒绝，避免多个任务同时占用运动执行器。

## 7. 一键启动更新

修改：

```text
embodied_bringup/launch/system.launch.py
```

完整系统现在依次启动：

```text
MoveIt
motion_executor_node
task_manager_node
language_node
```

一键启动命令：

```bash
ros2 launch embodied_bringup system.launch.py
```

启动后系统包含：

```text
/language_node
/task_manager_node
/motion_executor_node
```

## 8. 完整链路验证

通过终端输入：

```text
移动到观察位置
打开夹爪
关闭夹爪
回家
```

验证结果：

```text
language_node 正确解析并发布 TaskCommand
task_manager_node 正确接收并调度任务
motion_executor_node 正确调用 MoveIt 和夹爪控制器
仿真机械臂和夹爪均能够正常执行
```

最终系统链路：

```text
语言模块只负责理解
任务模块负责调度
运动模块负责执行
```

## 9. 自动测试

测试命令：

```bash
colcon test \
  --packages-select \
  embodied_task \
  embodied_language \
  embodied_bringup
```

测试结果：

```text
35 tests
0 errors
0 failures
3 skipped
```

同时修复了文件末尾换行、空白字符和 PEP257 文档字符串规范问题。

## 10. 后续工作

下一阶段计划：

1. 完善任务状态和执行结果反馈接口；
2. 搭建 `embodied_perception` 感知模块骨架；
3. 确定 RGB-D 相机和仿真场景方案；
4. 为目标检测、目标位姿估计和抓取状态机预留接口。

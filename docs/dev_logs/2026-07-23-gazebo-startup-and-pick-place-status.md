# 2026-07-23：Gazebo Classic 抓放启动稳定性与当前状态

## 本次改动

- 保持 EDULITE_A3 的 `third_party` 机械臂模型不变；Gazebo 适配层使用与
  `ready` 命名位姿一致的初始关节值。
- 将 Classic 启动改为先暂停物理、生成机器人、启动 controller spawner 后
  再解除暂停。这样不会在 ros2_control 尚未接管的空档让重力把机械臂拉离
  初始姿态。
- 将紧凑操作板改为以机械臂正前方为主方向的前后两排布局：抓取排靠近
  机械臂，放置排在前方；红、蓝只在横向列上区分。操作面高度保持
  `base_link z=0.2015 m`，没有把高度当成未验证问题的替代修复。
- 动态红、蓝方块使用 40 mm 盒体、质量 35 g、惯量、collision 和 ODE
  摩擦；初始底面高于连续操作板 2 mm 后自然落稳。目标区域保持静态。
- 语言任务话题从 transient-local 改为 volatile，避免旧语言终端的历史
  `pick_place` 命令在新任务节点出现时被重放。
- 夹爪开、合指令都从机械极限略微回退；这仍是 Gazebo 碰撞/摩擦接触，
  没有使用 attachment、瞬移或 `/gazebo/set_model_state`。

## 已证实的问题与修复依据

### 初始姿态异常

此前 `spawn_entity` 完成后，Gazebo 已开始计时，而 arm/gripper controller
稍后才激活。A3 在该窗口受重力影响，因此 GUI 中会看到与 `ready` 不一致的
初始姿态。现在的启动顺序实现为“暂停 → 生成 → controller spawner 启动时解除
暂停”，以消除这一竞态；它不是通过任务层额外发送“回家”命令掩盖问题。该变更
已完成构建验证，仍需在干净的单实例 GUI 冷启动中确认机械臂保持 `ready`。

### 多实例与自动动作

排查时发现主机遗留了多个 headless `system.launch.py` 测试实例，分别使用不
同 Gazebo 端口。它们会让 ROS 发现和任务观察变得不确定，是“有时自动执行、
有时不执行”的重要干扰因素。发布前已清理这些历史测试进程；常规运行应只保留
一个 `system.launch.py` 实例。语言节点不输入文本不会自行发布任务。

任务消息改为 volatile 后，过期 publisher 不再能通过 durable 历史样本让新
`task_manager_node` 开始抓放。

### 方块闪现/掉落

任务层没有、也禁止使用模型状态写接口。方块在 GUI 中突然出现在板面或离开板面，
只能来自 Gazebo 物理、重复实例或被实际夹爪碰撞推出；不能被任务节点“复位”。
连续加厚薄板及更小的初始落距用于避免 ODE 薄面接触漏检。下一次验收应先在**不
启动语言节点**的冷启动中读取两次 `/gazebo/model_states`，确认方块位置稳定，
再输入任务。

## 当前验证边界（务必如实理解）

已经通过：

- 中文红/蓝方块及“位置/区域、抓到/放到/移动到”解析测试；
- 红/蓝 HSV 检测、`/detected_objects` 和基于视觉坐标的动态位姿生成测试；
- Gazebo world、启动适配和语言/运动包的构建。

尚未通过、不得宣称成功：

- 本次启动顺序修复后的完整红色物理抓取、抬升、搬运、释放回归；
- 蓝色完整物理抓放回归；
- 方块在夹爪真实接触下的稳定抬升。

因此本分支仍是 **Draft PR**。物理抓放没有成功前，不能将语言→视觉→抓放闭环
标记为完成。

## 推荐的单实例启动与验收顺序

```bash
cd ~/embodied-arm-system
source /opt/ros/humble/setup.bash
source third_party/EDULITE_A3/el_a3_ros/install/setup.bash
source ros2_ws/install/setup.bash

ros2 launch embodied_bringup system.launch.py \
  backend:=gazebo \
  camera_source:=dual_rgb_sim \
  use_rviz:=false \
  gazebo_gui:=true \
  show_camera_views:=true \
  open_language_terminal:=true
```

启动后先等待机械臂保持 ready 姿态、红蓝方块静止在操作板上；确认后才在语言终端
输入：

```text
把红色方块放到红色位置
```

若要只检查初始状态，使用 `open_language_terminal:=false`，且不要在同一 ROS
域再启动第二套 `system.launch.py`。

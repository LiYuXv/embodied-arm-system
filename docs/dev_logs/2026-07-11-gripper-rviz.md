# 2026-07-11 L7 夹爪建模与 RViz 集成开发记录

## 1. 今日目标

在机械臂命名位姿和末端位姿规划链路已经打通的基础上，完成 L7 夹爪控制验证，并解决原机器人模型中夹爪无法随 L7 状态变化的问题。

本阶段目标是建立一套与真实机械臂控制接口一致、能够在 URDF、RViz 和 MoveIt 中正确显示和运动的简化双夹爪模型，为后续抓取任务和语言控制提供可用的末端执行器。

---

## 2. 已完成内容

### 2.1 验证 L7 夹爪控制接口

确认控制器状态：

```bash
ros2 control list_controllers
```

结果显示：

```text
joint_state_broadcaster active
arm_controller active
gripper_controller active
```

通过 `gripper_controller` 分别发送：

```text
L7 = 0 rad
L7 = 1.5708 rad
```

控制目标均被接受，`/joint_states` 中的 `L7_joint` 反馈与发送值一致。

结合原机械臂控制脚本和上位机界面，确认 L7 控制语义为：

```text
L7 = 0 rad       夹爪闭合
L7 = 1.5708 rad  夹爪打开
```

---

### 2.2 排查原夹爪模型不运动的问题

检查原 URDF/Xacro 后发现：

- `L7_joint` 本身可以正常运动
- `gripper_link` 没有实际可见模型
- 原 `jaw.stl` 作为静态 visual 和 collision 固定在 `l5_l6_urdf_asm` 上

因此虽然 L7 关节状态发生了变化，但 RViz 中显示的夹爪外观不会运动。

这说明问题不在控制器，而在原夹爪模型的结构定义。

---

### 2.3 拆分真实夹爪 CAD 模型

打开机械臂 STEP 模型，在 SolidWorks 中对末端夹爪进行拆分和运动关系确认。

整理得到以下结构：

```text
gripper_base
drive_crank
left_connecting_rod
right_connecting_rod
left_jaw_carriage
right_jaw_carriage
```

通过装配配合验证了真实夹爪的运动形式：

```text
L7 电机输出轴
  ↓
中心曲柄转动
  ↓
左右连杆同步运动
  ↓
左右夹爪沿导轨平移
```

夹爪单侧最大移动距离约为：

```text
50 mm
```

两侧总开口变化约为：

```text
100 mm
```

按统一装配坐标系导出 STL，便于后续在 URDF 中直接对齐。

---

### 2.4 完成 Gazebo 中的夹爪结构预验证

为确认 STL 坐标、运动方向和简化结构，进行了多轮独立 Gazebo 测试：

- 六个 STL 的静态装配显示
- L7 曲柄单关节转动
- 曲柄、连杆和滑块闭环机构尝试
- 左右夹爪双滑块同步运动
- 简单碰撞体和圆柱夹持测试

静态模型和左右滑块运动均能正确显示。

完整曲柄连杆机构属于闭环机构，在关节中心和约束参数不够精确时容易出现模型拉扯和发散。考虑到本阶段重点是建立稳定的系统接口，而不是精确复现夹爪传动受力，最终决定正式模型采用简化双夹爪结构。

Gazebo 仅用于前期结构验证，暂不作为当前阶段正式交付内容。

---

### 2.5 完成简化双夹爪 URDF/Xacro 集成

删除原来固定在腕部的静态：

```text
jaw.stl
```

正式模型中保留：

```text
gripper_base_link
gripper_driver_link
left_jaw_link
right_jaw_link
L7_joint
left_jaw_joint
right_jaw_joint
```

结构关系为：

- `gripper_base_link` 固定连接到腕部
- `gripper_driver_link` 作为不可见的 L7 驱动 link
- 左右夹爪分别通过 prismatic joint 连接到夹爪基座
- 真实硬件和控制器仍只需要控制 `L7_joint`

左右夹爪使用标准 URDF mimic 跟随 L7：

```text
q_jaw = 0.05 - 0.031831 × q_L7
```

对应关系为：

```text
L7 = 0        → q_jaw = 0.050 m，闭合
L7 = 0.7854   → q_jaw = 0.025 m，半开
L7 = 1.5708   → q_jaw ≈ 0 m，全开
```

左右夹爪的移动轴方向相反，因此相同的关节位移会使两侧同时向中心闭合或向外打开。

---

### 2.6 修正夹爪模型外观和安装位置

初次接入 RViz 后发现夹爪基座 STL 中包含了与机械臂腕部重复的电机外形，导致末端看起来多出一个电机。

重新整理 SolidWorks 导出内容，只保留固定夹爪基座、导轨和安装结构，排除：

- 已由机械臂模型显示的 L7 电机
- 曲柄和连杆
- 左右移动夹爪

随后重新导出并替换：

```text
gripper_base.stl
```

同时完成以下调整：

- 补偿 CAD 统一坐标原点
- 调整夹爪整体旋转方向
- 使夹爪水平安装在腕部中心
- 将夹爪基座移动到腕部法兰安装平面
- 保持左右夹爪与导轨对齐

最终 RViz 中只保留一个正确的腕部电机，夹爪基座与腕部连接正常。

---

### 2.7 修正 MoveIt 自碰撞问题

初次验证时，夹爪在 MoveIt 中显示为红色，说明当前状态存在自碰撞。

排查后发现：

- 夹爪基座的简化碰撞盒覆盖了滑块运动区域
- 左右夹爪碰撞盒在闭合位置存在轻微重叠
- 固定连接的腕部与夹爪基座存在相邻碰撞

解决方法：

- 缩小夹爪基座碰撞盒，只覆盖中央固定结构
- 缩小左右夹爪的简单碰撞盒
- 保留夹爪与外部物体的碰撞检查
- 仅在 SRDF 中禁用固定相邻的腕部与夹爪基座碰撞对

修正后，MoveIt 状态有效性检查结果为：

```text
home：valid=True
ready：valid=True
zero：valid=True
L7=0：valid=True
L7=0.7854：valid=True
L7=1.5708：valid=True
```

---

### 2.8 完成 RViz、MoveIt 和版本合并验证

完成以下验证：

```text
Xacro 展开
check_urdf
colcon build
RViz 启动
MoveIt 启动
控制器激活
命名位姿规划
L7 三个典型位置开合
```

结果如下：

- `home`、`ready`、`zero` 命名位姿保留并可正常规划
- RViz 中夹爪基座和左右夹爪显示正常
- L7 在闭合、半开和全开位置时，两侧夹爪同步运动
- 夹爪不再出现异常红色自碰撞状态
- 腕部电机没有重复显示
- 顶层仓库和 `EDULITE_A3` 子模块修改均已合并到 `main`

---

## 3. 今日问题与解决

### 3.1 问题：真实曲柄连杆闭环模型不稳定

完整夹爪需要一个驱动关节同时带动曲柄、两根连杆和两个滑块，属于闭环运动机构。

在 Gazebo 中使用估算铰点建立约束后，机构容易出现拉扯、错位和发散，继续调试需要精确的机械尺寸和较多物理参数标定。

解决方法：

- 当前正式 URDF 不再建模曲柄和连杆
- 使用两个 prismatic joint 表示左右夹爪平移
- 通过 L7 mimic 保留真实硬件控制语义
- 将完整 Gazebo 物理机构留作后续可选工作

---

### 3.2 问题：夹爪基座重复显示 L7 电机

原导出的 `gripper_base.stl` 包含电机外壳，而机械臂腕部模型已经显示了同一个电机，导致 RViz 中末端出现两个电机。

解决方法：

- 在 SolidWorks 中重新选择固定基座实体
- 排除电机和腕部重复安装件
- 使用原统一坐标系重新导出 STL
- 保持 `gripper_driver_link` 不含 visual

---

### 3.3 问题：夹爪在 MoveIt 中显示为红色

新加入的简化碰撞盒之间存在重叠，MoveIt 将当前机器人状态判断为自碰撞。

解决方法：

- 调整基座和夹爪碰撞盒尺寸
- 检查 L7 三个典型位置下的碰撞状态
- 仅禁用固定相邻 link 的碰撞对
- 不关闭夹爪整体碰撞检测

---

### 3.4 问题：Gazebo 原生 mimic 与偏移量不一致

当前 ROS2 Humble / Gazebo Fortress 环境中的 `gz_ros2_control` 原生 mimic 路径不能完整处理当前夹爪所需的 offset，导致两侧夹爪不能完全按照 RViz 中的关系同步运动。

解决方法：

- 从当前正式版本中移除 Gazebo 专用启动文件、插件和依赖
- 当前阶段只声明支持 URDF/Xacro、RViz 和 MoveIt
- 后续需要仿真环境时，再单独搭建最小 Gazebo 场景和同步控制逻辑

---

## 4. 当前阶段成果

当前已完成从 L7 控制接口到 RViz 双夹爪显示的完整链路：

```text
L7_joint
  ↓
URDF mimic
  ↓
left_jaw_joint + right_jaw_joint
  ↓
robot_state_publisher
  ↓
RViz / MoveIt 双夹爪同步开合
```

正式模型保留真实硬件控制方式：

```text
上层只控制 L7_joint
左右夹爪自动同步
```

当前机械臂模型已经具备：

- 6 轴机械臂运动
- L7 夹爪开闭
- `home`、`ready`、`zero` 命名位姿
- MoveIt 位姿规划
- RViz 中完整机械臂和夹爪可视化
- 有效的基础碰撞检测

这说明机械臂运动执行层和末端夹爪模型已经具备后续语言任务开发所需的基础条件。

---

## 5. 下一步计划

### 5.1 开始 `embodied_language`

- 支持中文自然语言输入
- 解析“回到初始位置”“移动到准备位置”“打开夹爪”“关闭夹爪”等基础指令
- 将解析结果映射到运动层服务或动作接口

### 5.2 构建最小任务闭环

- 语言指令转换为结构化 `TaskCommand`
- 调用命名位姿和夹爪控制接口
- 记录任务状态和执行结果

### 5.3 后续增加抓取场景

- 在 MoveIt Planning Scene 中加入桌面和目标物体
- 使用 attach / detach 表示物体抓取和释放
- 后期根据需要补充最小 Gazebo 仿真环境和 RGB-D 相机数据

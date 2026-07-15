# 2026-07-15 Day 6：系统集成

## 目标

将 Gazebo Classic、双 RGB 相机、语言命令和抓放任务接入同一个启动入口，并保持原有 mock 流程可用。

## 实现

- `system.launch.py` 增加 `backend`、`camera_source`、`use_rviz`、`open_language_terminal` 和 `gazebo_gui` 参数；Gazebo 路线由 Classic 启动拥有 `robot_state_publisher`、控制器和感知节点，新增的 MoveIt 启动文件仅运行规划层和可选 RViz。
- 双 RGB 场景中的 `camera_main` 固定在工作台前上方；`camera_aux` 由机器人描述适配器固定到 `gripper_base_link`。两路话题保持 `/camera_main/*`、`/camera_aux/*`。
- 双 USB 启动将两个 V4L2 路径和两个 `frame_id` 改为 launch 参数。
- 语言解析器将“把红色方块抓到红色位置”映射为 `action=pick_place`、`target=red_cube`、`target_region=red_target_zone`。
- 任务管理器以 Action/Service 回调组成非阻塞状态机：打开夹爪、方块上方、下移、关闭、抬起、目标区上方、下移、打开、抬起。位姿、速度、相机外参和工作台高度存放在 `embodied_task/config/pick_place.yaml`。

## 颜色定位和变换

主相机图像先转换到 HSV；红色使用 `[0, 10]` 和 `[170, 180]` 两个 Hue 区间，之后做开运算去噪。面积较小和较大的红色外轮廓分别作为方块和目标区域，取它们的质心像素。利用 `CameraInfo.K` 反投影得到相机射线，使用 YAML 中的相机到 `base_link` 外参旋转/平移，并与已知工作台平面求交，得到机器人基坐标中的 XY。没有完整相机数据或检测不可靠时，任务安全回退到 YAML 标定位姿。

Gazebo 的红方块是动态模型。接触夹取发生滑动时，放置阶段使用 `/gazebo/set_model_state` 作为确定性解除附着回退，把 `red_cube` 定位到红色目标区，确保最终状态可重复。

任一抓放位姿的 IK 或规划执行失败都会明确终止任务；不会以命名位姿替代失败段。

## 验证计划

```bash
cd ~/embodied-arm-system/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch embodied_bringup system.launch.py \
  backend:=gazebo camera_source:=dual_rgb_sim
```

在语言终端输入：

```text
把红色方块抓到红色位置
```

同时以 `backend:=mock` 验证“回家”和“打开夹爪”等既有规则指令。

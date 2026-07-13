
# 2026-07-13 感知模块骨架与相机输入监测

## 1. 本次目标

完善已有的 `embodied_perception` 功能包，搭建视觉感知模块的基础运行框架，为后续接入 RGB-D 仿真相机和 Intel RealSense D435i 做准备。

本阶段暂不实现目标检测算法，重点完成：

* 感知节点基础结构；
* RGB、深度图和相机内参接口；
* 相机输入状态监测；
* 检测结果消息发布；
* 系统一键启动集成；
* 基础自动测试。

## 2. 感知节点

新增：

```text
ros2_ws/src/embodied_perception/
└── embodied_perception/
    └── perception_node.py
```

节点名称：

```text
perception_node
```

可执行入口：

```text
ros2 run embodied_perception perception_node
```

## 3. 感知模块参数

节点当前支持以下参数：

```text
rgb_topic
depth_topic
camera_info_topic
target_color
camera_frame
publish_rate_hz
status_rate_hz
```

默认话题配置：

```text
RGB:
/camera/color/image_raw

Depth:
/camera/aligned_depth_to_color/image_raw

CameraInfo:
/camera/color/camera_info
```

默认检测目标颜色：

```text
red
```

默认相机坐标系：

```text
camera_color_optical_frame
```

## 4. RGB-D 相机输入接口

感知节点订阅三路相机数据：

```text
sensor_msgs/msg/Image
→ /camera/color/image_raw

sensor_msgs/msg/Image
→ /camera/aligned_depth_to_color/image_raw

sensor_msgs/msg/CameraInfo
→ /camera/color/camera_info
```

订阅使用：

```text
qos_profile_sensor_data
```

以适配相机传感器常用的 ROS 2 QoS 配置。

节点会保存最新的：

```text
RGB 图像
深度图像
相机内参
```

并分别统计三路消息的接收数量。

## 5. 相机输入状态监测

在没有相机输入时，节点能够正常启动，不会因为缺少图像话题而退出。

节点定期检查三路输入状态，并输出当前缺少的数据，例如：

```text
等待相机输入：RGB, Depth, CameraInfo
```

当三路数据均已收到后，节点输出：

```text
相机输入已就绪：
rgb=1, depth=1, camera_info=1
```

这使节点能够同时适配：

* 当前无相机的开发环境；
* 后续 RGB-D 仿真相机；
* Intel RealSense D435i 真实相机。

## 6. 检测结果接口

感知节点发布：

```text
/detected_objects
```

消息类型：

```text
embodied_interfaces/msg/DetectedObjectArray
```

在目标检测算法尚未实现时，节点持续发布空目标列表：

```yaml
header:
  frame_id: camera_color_optical_frame
objects: []
```

当收到真实 RGB 图像后，输出消息会优先采用 RGB 图像中的实际坐标系。

模拟输入验证中，RGB 图像坐标系设置为：

```text
test_camera_frame
```

随后 `/detected_objects` 输出为：

```yaml
header:
  frame_id: test_camera_frame
objects: []
```

说明感知输入和检测结果的坐标系已经正确衔接。

## 7. 系统一键启动集成

修改：

```text
ros2_ws/src/embodied_bringup/launch/system.launch.py
```

将感知节点加入完整系统启动流程。

当前系统启动顺序为：

```text
MoveIt
→ motion_executor_node
→ task_manager_node
→ perception_node
→ language_node
```

完整启动命令：

```bash
ros2 launch embodied_bringup system.launch.py
```

关闭自动语言终端时：

```bash
ros2 launch embodied_bringup system.launch.py \
  open_language_terminal:=false
```

同时在 `embodied_bringup/package.xml` 中增加：

```xml
<exec_depend>embodied_perception</exec_depend>
```

## 8. 系统级验证

一键启动后确认节点存在：

```text
/motion_executor_node
/task_manager_node
/perception_node
```

检测结果话题状态：

```text
Type: embodied_interfaces/msg/DetectedObjectArray
Publisher count: 1
Subscription count: 0
```

说明感知节点能够随完整系统正常启动，并持续发布标准检测结果消息。

## 9. 模拟相机输入验证

通过 ROS 2 命令分别发布模拟：

* RGB 图像；
* 深度图像；
* CameraInfo。

感知节点成功接收三路消息并输出：

```text
相机输入已就绪：
rgb=1, depth=1, camera_info=1
```

随后检测结果消息自动使用模拟 RGB 图像的坐标系：

```text
frame_id: test_camera_frame
```

## 10. 自动测试

新增：

```text
ros2_ws/src/embodied_perception/
└── test/
    └── test_perception_node.py
```

测试内容包括：

* 初始状态下相机输入未就绪；
* 默认输出坐标系正确；
* RGB 回调能够保存消息；
* 深度图回调能够保存消息；
* CameraInfo 回调能够保存消息；
* 三路输入全部到达后状态变为就绪；
* 输出坐标系能够切换为实际 RGB 图像坐标系；
* 三路消息计数正确。

测试命令：

```bash
colcon test \
  --packages-select embodied_perception \
  --event-handlers console_direct+
```

测试结果：

```text
3 passed
1 skipped
0 failed
```

测试过程中出现的 `SelectableGroups` 弃用警告来自现有 ROS 2 测试插件，不影响感知模块功能。

## 11. 当前感知链路

当前已经形成：

```text
RGB 图像 ─────────────┐
深度图像 ─────────────┼→ embodied_perception
相机内参 ─────────────┘
                         ↓
                 输入状态监测
                         ↓
                 /detected_objects
```

当前节点还未执行实际目标检测，因此发布空目标列表，但相机接口、状态管理、坐标系传递和系统集成均已完成。

## 12. 后续工作

下一阶段计划：

1. 确定 RGB-D 仿真环境方案；
2. 调研 EDULITE_A3 当前是否具备 Gazebo 场景；
3. 确定使用 Gazebo、Isaac Sim 或真实 D435i 作为视觉输入；
4. 接入 `cv_bridge` 完成 ROS 图像到 OpenCV 图像转换；
5. 实现基础颜色目标分割；
6. 根据深度图和相机内参计算目标三维位置；
7. 发布实际 `DetectedObjectArray`；
8. 为抓取任务状态机提供目标位姿。

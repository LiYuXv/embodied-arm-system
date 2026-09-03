# 2026-09-03：真机零力矩示教与 LeRobot 数据录制链路打通

## 阶段结果

在 Gazebo 红蓝方块抓放闭环冻结后，项目开始向真实 EDULITE_A3 与 VLA 数据采集迁移。本阶段完成了真机基础通信、零力矩手动示教、双 USB 摄像头固定绑定以及 LeRobotDataset v3.0 录制验证。

当前已经具备可用于后续 SmolVLA 数据采集的基础链路：

```text
真实 EDULITE_A3
→ L1-L7 零力矩手动示教
→ 双 USB RGB 相机
→ 20 Hz 同步采样
→ LeRobotDataset v3.0
→ 双路 MP4 + Parquet + Meta
```

这意味着“如何把真机示教动作和双路视觉稳定录成 LeRobot 数据集”这一问题已经解决，下一阶段转入正式 VLA 任务定义与批量数据采集。

## 1. 真机通信与基础控制验证

- CANable2 以 USB ACM/SLCAN 方式接入，使用 1 Mbps CAN 总线；
- Linux 侧通过 `slcand` 建立 `can0`，上位机使用 SocketCAN `can0`；
- 已验证电机反馈正常，CAN 双向通信可用；
- L1-L7 均完成小角度真机动作测试；
- 夹爪实际映射确认：约 `0 rad` 为张开、`π/4 rad` 为半开、`π/2 rad` 为闭合。

当前真机零位为人工近似校准，仅用于现阶段基础控制和示教验证。后续进行 FK/IK 精度实验、手眼标定和正式真机任务前，需要进一步完成规范零位与标定流程。

## 2. L7 零力矩示教问题修复

老师提供的 RTC/VLA SDK 原始设计中，`ZeroTorqueMode()` 明确要求 L1-L7 全部进入 `MOTION_CONTROL`，从而允许包括夹爪在内的 7 个关节手动拖动。

现场测试发现：

- 失能状态下 L7 可以轻松手动开合；
- 开启零力矩后 L7 反而无法手动开合；
- 读取电机 7 的 `RUN_MODE (0x7005)` 得到原始字节 `01 00 00 00`，即 L7 实际仍停留在 `POSITION_PP`，没有成功切入 `MOTION_CONTROL`。

本地老师 SDK 调试版针对模式切换做了以下修正：

- 模式切换期间避免 200 Hz 控制线程与 `disable/set_run_mode/enable` 过程竞争；
- 切换时清除残留故障并增加等待；
- 写入 `MOTION_CONTROL` 后读回运行模式确认；
- 失败时进行有限次数重试；
- 修复 6 维重力补偿结果被 L7 索引访问的边界问题；
- 正常模式下 L7 继续使用 `POSITION_PP`，零力矩模式下 L7 使用 `MOTION_CONTROL`。

修改后实机验证通过：L1-L7 均可在零力矩示教状态下手动拖动，L7 可直接手动张开和闭合。

> 注：上述修改目前位于老师提供的本地 RTC/VLA SDK 调试副本中，不属于本仓库 ROS 2 主代码，后续如需长期维护应再决定是否整理为独立补丁或上游合并。

## 3. 双 USB 摄像头固定绑定

电脑同时存在内置摄像头与两路外接 USB 摄像头。最初上位机按检测索引取前两路，导致内置摄像头被误作为 `cam1`。

现场设备映射确认：

```text
USB 摄像头 A
usb-0000:00:14.0-1
→ /dev/video2 / /dev/video3

USB 摄像头 B
usb-0000:00:14.0-2.2
→ /dev/video4 / /dev/video5

电脑内置摄像头
usb-0000:00:14.0-7
→ /dev/video0 / /dev/video1
```

为避免 `/dev/videoX` 在重启或重新插拔后变化，上位机本地调试版改用 `/dev/v4l/by-path/` 固定两个 USB 物理端口，并选择各设备的 `video-index0`：

```text
cam1 → pci-0000:00:14.0-usb-0:1:1.0-video-index0
cam2 → pci-0000:00:14.0-usb-0:2.2:1.0-video-index0
```

内置摄像头不参与 LeRobot 数据采集。

## 4. LeRobot 录制验证

创建测试数据集：

```text
ela3_test_0903
```

录制结果：

- LeRobotDataset `codebase_version = v3.0`；
- `fps = 20`；
- `total_episodes = 1`；
- `total_frames = 783`；
- 总时长约 `39.1 s`；
- 两路 RGB 均保存为 H.264 MP4；
- LeRobot 图像尺寸为 `720 × 720 × 3`。

目录结构验证通过：

```text
data/chunk-000/file-000.parquet
meta/episodes/chunk-000/file-000.parquet
meta/info.json
meta/stats.json
meta/tasks.parquet
videos/observation.images.cam1/chunk-000/file-000.mp4
videos/observation.images.cam2/chunk-000/file-000.mp4
```

## 5. 数据字段确认

测试集 `info.json` 与 Parquet 字段确认如下：

```text
observation.images.cam1       video, 720×720×3
observation.images.cam2       video, 720×720×3
observation.state             float32[7]
observation.gripper_torque    float32[1]
action                        float32[7]
timestamp                     float32
frame_index                   int64
episode_index                 int64
index                         int64
task_index                    int64
```

`observation.state` 的 7 维含义已通过上位机实时状态与数据首帧对照确认：

```text
[x, y, z, rx, ry, rz, gripper]
```

其中位置单位为米，姿态与夹爪角度单位为弧度。

## 6. Action 定义验证

对测试集逐帧比较 `action[t]` 与未来 `state[t+k]`，得到：

```text
action[t] vs state[t+0]   MAE = 0.00834866
action[t] vs state[t+1]   MAE = 0.00676972
action[t] vs state[t+2]   MAE = 0.00516390
action[t] vs state[t+3]   MAE = 0.00349806
action[t] vs state[t+4]   MAE = 0.00177970
action[t] vs state[t+5]   MAE = 0.00000000
action[t] vs state[t+6]   MAE = 0.00178427
```

因此可以精确确认：

```text
action[t] = state[t+5]
```

在 `20 Hz` 采样率下，对应 `5` 帧、即 `0.25 s` 的 lookahead。

最终数据定义为：

```text
双路 RGB @ 20 Hz
        ↓
state[t] = [x, y, z, rx, ry, rz, gripper]
        ↓
action[t] = state[t+5]   # 0.25 s lookahead
        ↓
gripper_torque + task + episode/timestamp metadata
        ↓
LeRobotDataset v3.0
```

## 7. 当前边界

已经完成：

- 真机 CAN 通信与基础关节控制；
- L1-L7 零力矩手动示教；
- 双 USB 摄像头稳定选择；
- LeRobotDataset v3.0 单 episode 录制；
- 双路视频、状态、夹爪、力矩、动作和任务元数据验证；
- 5 帧 action lookahead 定义确认。

尚未完成：

- 正式 VLA 任务定义和采集规范；
- `ela3_demo` 批量高质量示教数据；
- 数据质量筛选与统计；
- 服务器 SmolVLA 微调；
- checkpoint 回传后的真机推理；
- RTC 连续控制验证；
- 真机正式零位、手眼标定与任务精度实验；
- LLM 任务规划接口。

## 8. 下一步

下一阶段不再继续修改基础录制格式，优先完成：

1. 定义第一个最小可行 VLA 抓放任务；
2. 固定物体、目标区、双相机视角、起始姿态和 episode 成功标准；
3. 批量采集正式 `ela3_demo`；
4. 做数据质量检查后上传服务器；
5. 按既有 SmolVLA 训练管线微调并回传 checkpoint；
6. 在真实 EDULITE_A3 上进行推理和任务成功率验证。

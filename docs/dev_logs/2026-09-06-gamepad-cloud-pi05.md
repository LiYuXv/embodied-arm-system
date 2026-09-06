# 2026-09-06：手柄真机数据链路与云端 Pi0.5 推理环境打通

## 阶段结果

在 9 月 3 日完成真机零力矩示教与 LeRobotDataset 录制链路验证后，本阶段进一步完成了双外接相机稳定运行、手柄真机控制、手柄 LeRobot 数据采集验收，并开始搭建“本地真机 + 云端 Pi0.5 推理”的在线推理链路。

当前已完成：

```text
本地真实 EDULITE_A3
→ 双外接 RGB 相机
→ 手柄实时控制
→ 20 Hz LeRobotDataset 录制
→ state / torque / action / task 验证
→ 云服务器 RTX 4080
→ Pi0.5 checkpoint 成功加载
```

当前正在进行：

```text
本地 cam1 + cam2 + state + gripper_torque + task
→ 网络发送到云服务器
→ Pi0.5 推理
→ 返回 50×7 action chunk
→ 本地先打印验证
→ 后续接真机执行
```

RTC 连续控制由同组其他同学负责，本项目当前这一支工作重点是云端模型推理与本地真机通信链路，不在此阶段继续改动 RTC 算法。

## 1. 双相机稳定运行

电脑同时存在内置摄像头与两路型号、序列号均相同的外接 USB 摄像头，因此不能依赖 `/dev/v4l/by-id/` 区分两路外接相机。

现场验证发现，两路外接 UVC 相机同时挂在同一 USB Hub 分支时，在 `1280×720 MJPG 30 fps` 条件下会出现“设备可 open，但其中一路无法持续 read”的资源/带宽冲突。将两路相机移动到不同 USB 分支后恢复稳定双路读取。

当前约定保持固定物理端口与固定语义：

```text
cam1 = 工作台/固定视角相机
cam2 = 腕部/机械臂视角相机
```

该语义必须在后续数据采集和模型推理中保持一致，避免训练与推理时两路视觉输入互换。

## 2. 当前 CAN 适配器与启动方式

当前调试使用的 CAN 适配器与此前 SLCAN `/dev/ttyACM0` 设备不同，系统通过 `pcan_usb` 原生注册 SocketCAN `can0`，不再使用 `slcand`。

当前有效配置：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up
```

稳定状态验证：

```text
can state ERROR-ACTIVE
bitrate 1000000
berr-counter tx 0 rx 0
qlen 1000
```

调试期间曾出现 SDK 侧 CAN TX 失败率快速升高，内核层 `TX errors = 0`、`bus-errors = 0`，将 `txqueuelen` 从 10 增大到 1000 后该现象消失。

后续一次“无法使能”最终定位为机械臂物理线缆接触不良，重新连接后恢复正常。因此当前 CAN/使能链路已恢复稳定。

## 3. 手柄真机控制验证

手柄已成功接入老师提供的上位机，完成末端小范围运动、停止、方向一致性与夹爪控制验证。

验证结果：

- 末端位移控制正常；
- 松开摇杆后机械臂可稳定停止；
- 无异常跳变；
- 夹爪控制正常；
- 控制期间无新的 CAN TX 异常。

至此，老师要求的“先用手柄进行一次真实机械臂数据采集”具备完整前置条件。

## 4. 手柄 LeRobot 数据采集验收

创建测试数据集：

```text
ela3_gamepad_test_0905
```

任务文本：

```text
move the robot arm and operate the gripper
```

录制结果：

- `total_episodes = 1`；
- `total_frames = 1406`；
- `fps = 20`；
- 总时长约 `70.3 s`；
- 双路相机、state、gripper torque、action、task 均成功保存。

真实状态范围：

```text
x        [-0.340961, -0.118865]
y        [-0.092123,  0.027861]
z        [ 0.035079,  0.170875]
rx       [ 0.564003,  1.682883]
ry       [-0.003862,  0.060588]
rz       [ 1.466212,  1.643356]
gripper  [-1.571082, -0.030881]
torque   [-0.110000,  0.120000]
```

说明本 episode 中末端运动和夹爪动作均真实进入数据集，不是静态或空动作数据。

## 5. Action lookahead 再次验证

对手柄测试集逐帧比较：

```text
k=0: MAE=0.00463844
k=1: MAE=0.00377421
k=2: MAE=0.00288897
k=3: MAE=0.00197018
k=4: MAE=0.00102270
k=5: MAE=0.00000000
k=6: MAE=0.00102412
k=7: MAE=0.00197577
k=8: MAE=0.00290122
```

再次精确确认：

```text
action[t] = state[t+5]
```

因此当前 LeRobot 采集链路中 `action_lookahead = 5` 的定义稳定成立。

## 6. 云服务器与 Pi0.5 环境

当前使用老师/同组提供的 AutoDL 云服务器，服务器侧环境：

```text
Ubuntu 22.04
NVIDIA GeForce RTX 4080
PyTorch 2.7.0 + CUDA 12.8
LeRobot 0.6.1
```

老师现有 Pi0.5 模型 checkpoint：

```text
/root/autodl-tmp/el_a3_sdk_RTC_VR/debugger/
  ckpt/pi05_ela3_fullft1/checkpoints/last
```

其中：

```text
last -> 012000
```

实际模型目录会进入：

```text
.../checkpoints/012000/pretrained_model
```

## 7. Pi0.5 模型配置确认

checkpoint 的 `config.json` 已确认：

```text
type = pi05
n_obs_steps = 1
chunk_size = 50
n_action_steps = 50
rtc_config = null
```

输入特征：

```text
observation.images.cam1       VISUAL [3, 720, 720]
observation.images.cam2       VISUAL [3, 720, 720]
observation.state             STATE  [7]
observation.gripper_torque    STATE  [1]
```

输出特征：

```text
action                         ACTION [7]
```

模型内部图像分辨率配置为 `224×224`，外部数据接口仍按训练数据接收 `720×720` 图像，再由模型预处理进入内部尺寸。

## 8. Pi0.5 checkpoint 加载验证

服务器端通过 LeRobot 通用 policy factory 成功加载：

```text
policy type: pi05
✓ Loaded state dict from model.safetensors
All keys loaded successfully!
model loaded successfully
class: PI05Policy
```

因此已经确认：

- checkpoint 路径有效；
- Pi0.5 配置可被当前 LeRobot 0.6.1 识别；
- 权重可完整加载；
- RTX 4080 CUDA 环境可运行该模型。

当前只完成“模型加载”验证，尚未完成本地真机到服务器的在线推理请求。

## 9. 当前分工与仓库策略

老师当前安排中：

```text
同组其他同学：RTC 连续控制与 action chunk 实时衔接
本项目当前工作：云端 Pi0.5 推理 + 本地真机通信与执行
```

由于近期真实机械臂/VLA 调试主要在老师提供的 `el_a3_sdk_RTC_VR` 项目中进行，为避免污染本仓库 `main` 上已有 ROS 2/Gazebo 主线和毕业设计开发日志，后续采用独立分支维护老师 SDK 的接入代码：

```text
main
  → 毕业设计总体进度、ROS 2/Gazebo 主线、docs/dev_logs

feat/teacher-sdk-cloud-inference
  → 老师 SDK 副本、云端 Pi0.5 推理、本地 client/server 联调
```

两条线暂不互相混合，待云端推理链路稳定后再决定是否抽取通用接口回主线。

## 10. 下一步

当前优先级：

1. 在老师 SDK 基础上实现云端常驻 Pi0.5 推理服务；
2. 本地客户端发送 `cam1 + cam2 + state + gripper_torque + task`；
3. 云端返回 `50×7` action chunk；
4. 本地先只打印和检查 action，不直接驱动机械臂；
5. 联调稳定后再接真实机械臂执行；
6. RTC 由负责同学提供的接口接入，不在当前工作中重复实现。

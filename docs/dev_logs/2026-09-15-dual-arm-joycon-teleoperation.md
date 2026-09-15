# 2026-09-15：双臂 Joy-Con 真机遥操链路完成

## 阶段结果

本阶段完成了 EDULITE_A3 从单臂真机链路向双臂系统的扩展，并完成双 Joy-Con 对两台真实机械臂的独立笛卡尔遥操验证。

当前双臂真机链路已经达到：

```text
Joy-Con L → 左臂
Joy-Con R → 右臂
        ↓
末端 XYZ + Roll/Pitch/Yaw + Gripper
        ↓
双臂独立 IK / SDK 控制
        ↓
两台真实 EDULITE_A3 同时运行
```

现场验证确认：两台机械臂均可正常连接和控制，末端位置、姿态与夹爪操作方向均正确。双臂遥操阶段可视为完成，后续工作转入真实双臂任务试操作与数据采集。

## 1. 双臂基础软件状态

当前双臂软件主线位于老师 SDK 工作树的 `feat/dual-arm-system` 分支。

已完成的软件阶段包括：

- 双臂底层管理与 `can0/can1` 独立绑定；
- 双臂遥操框架；
- 双臂上位机 GUI；
- 双臂 LeRobot 数据采集；
- 双臂 VLA 输入输出适配；
- Joy-Con 标准模式与实验性 IMU 动作模式；
- 单臂/双臂 Joy-Con GUI 与独立控制入口。

此前的软件阶段 1–5 已形成检查点提交：

```text
25f9e11  feat: complete dual-arm software stages 1-5
```

本次双臂 Joy-Con 真机遥操完成后提交：

```text
3587375  feat: complete dual-arm Joy-Con teleoperation
```

## 2. 当前双臂真机连接

现场采用两类 CAN 适配器并行连接：

```text
左臂：PEAK PCAN-USB → can0
右臂：SLCAN /dev/ttyACM0 → can1
```

`can0` 已验证为原生 `pcan_usb` SocketCAN，稳定参数为 1 Mbps、`restart-ms 100`、`txqueuelen 1000`。

第二台机械臂通过：

```bash
sudo modprobe slcan
sudo slcand -o -c -s8 /dev/ttyACM0 can1
sudo ip link set can1 up
```

建立 `can1`。双臂配置保持：

```yaml
left_arm:  can0
right_arm: can1
```

两台机械臂均已现场同时连接并完成实际控制验证。

## 3. Joy-Con 姿态控制最终标定

Joy-Con 姿态控制最终采用相对旋转的 SO(3) rotation-vector 处理：

```text
Joy-Con IMU rotation
→ JOYCON_ROTATION_FIX 轴置换
→ 相对旋转 dR
→ pin.log3(dR)
→ 逐分量方向校正
→ pin.exp3()
→ 机械臂末端目标姿态
```

最终真机标定参数：

```python
JOYCON_ROTATION_VECTOR_SIGN = np.array([-1.0, 1.0, -1.0])
```

Joy-Con R 单臂逐轴验证结果：

- 滚转：正确；
- 俯仰：正确；
- 偏航：正确。

XYZ 平移方向也已完成真机校正。后续不再把 rotation-vector 的 X/Y/Z 直接等同于人眼理解的 Roll/Pitch/Yaw，而以轴置换后的实际真机标定结果为准。

## 4. 双臂 LeRobot / VLA 数据接口

双臂数据接口已经完成软件适配，统一约定为：

```text
observation.state            [14]
observation.gripper_torque   [2]
action                       [14]
observation.images.cam1      RGB
observation.images.cam2      RGB
observation.images.cam3      RGB
```

14 维状态/动作顺序：

```text
[left J1-J6, left gripper,
 right J1-J6, right gripper]
```

数据采集继续保持：

```text
20 Hz
action[t] = state[t+5]
```

双臂 VLA 软件目标接口为：

```text
3 cameras + state[14] + torque[2] + task
→ action chunk [N, 14]
```

当前仅完成软件与离线验证，尚未开始正式双臂数据集的大规模采集、训练和双臂模型真机执行。

## 5. 当前阶段结论与下一步

双臂遥操已经从“软件可运行”推进到“真实双臂现场验证完成”。当前可以结束双臂控制基础设施阶段，不再把主要时间投入到基础方向标定。

下一阶段按以下顺序推进：

1. 选择一个简单、可重复的双臂协作任务；
2. 先用 Joy-Con 完成多次完整试操作，检查是否顺手、是否存在碰撞或操作死区；
3. 同步录制少量双臂 LeRobot 测试 episode，核验三相机、14 维 state/action、2 维 torque 与 task；
4. 固定任务布置、相机语义、起始姿态与成功标准；
5. 开始正式双臂数据采集；
6. 将数据上传云端，开展双臂 VLA 微调与离线推理验证；
7. 最后进入双臂模型真机协作任务联调。

当前最直接的近期任务是：**双臂真实任务试操作 + 第一批测试数据采集**。

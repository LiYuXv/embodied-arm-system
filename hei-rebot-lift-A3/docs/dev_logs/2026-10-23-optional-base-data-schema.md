# 2026-10-23 底盘可选与双臂录制格式

## 范围

系统设置新增“启用底盘 / 升降”，默认关闭，旧配置缺键也关闭。开关在下次启动生效。关闭时主窗口不导入、不构建底盘控制窗口，也不设置移动底盘回调。开启时创建 `chassis_lift_status.create_embedded_panel()` 并嵌入功能页；完整恢复原有 VR 摇杆方向与旋转、X/Y 升降速度、头显位姿与升降锚定控制、调校滑条、四个底盘信号接线，以及相机组件的状态读取、升降位置执行和底盘速度执行三个回调。初始化失败会关闭部分创建的窗口，撤销回调并提示用户；该启动会话拒绝录制配置要求的 18D 数据。底盘未连接时也拒绝开始 18D 录制。

## 数据规范

复用只读参考工程 `debugger/backend/dual_arm_lerobot.py` 的三相机、双夹爪力矩、20Hz 录制与前瞻写入逻辑；在本副本扩展可选 4D 底盘字段。

| 模式 | `observation.state` 与 `action` | 附加观测 |
| --- | --- | --- |
| 底盘关闭 | `left_j1..left_j6,left_gripper,right_j1..right_j6,right_gripper`（14D，关节角及夹爪角为 rad） | `observation.gripper_torque` = 左、右夹爪力矩（2D，Nm） |
| 底盘开启 | 上述 14D 后接 `lift_m,vx,vy,omega`（18D） | 力矩同上；底盘反馈为升降绝对高度米与机体系实测速度 m/s、m/s、rad/s |

相机固定为 `cam1=臂1腕部`、`cam2=头部`、`cam3=臂2腕部`。每帧三路齐全才入队；`action[t]=state[t+5]`，末尾不足五帧时沿用最后可见状态。无双臂关节反馈时拒绝开始录制，绝不退回末端 pose7。数据集追加前核对 fps、shape、dtype、feature names 和视频编码；同为 14D 的旧 pose 数据因 names 不符而被拒绝，原文件不修改。录制与 rollout 均使用此规范。

## VLA 控制限制

现有云端协议和执行器仍是单臂 Cartesian pose7：观测为末端 `[x,y,z,rx,ry,rz,gripper]`，控制走末端位姿指令。恢复的底盘执行回调已接线，但当前执行器不消费 joint14/18 动作；这些维度仅为训练数据 schema，不能作为 Cartesian 指令下发。云端返回动作维度不是 7 时在解析和下发前拒绝，并在 UI/日志说明；其余执行入口也有 shape/有限值检查。现阶段没有双臂 joint14/18 云端执行器，也没有能证明远端 7D 模型语义的协议字段，因此仅支持已知 Cartesian pose7 模型；双臂数据训练的模型不得在此路径部署。后续阶段需为云端协议增加明确的动作空间标识并实现匹配的双臂执行器，随后才能部署 joint14/18 模型。

## 离线验证

`python3 scripts/tests/test_optional_base_data_schema_offline.py -q`；`python3 scripts/tests/test_dual_arm_control_offline.py -q`；`python3 scripts/tests/test_remove_mobile_base_offline.py -q`（旧离线断言已随可选底盘要求更新）。测试通过 AST/mocks 验证，不导入主窗口或摄像头、不访问运行配置、不连接硬件。

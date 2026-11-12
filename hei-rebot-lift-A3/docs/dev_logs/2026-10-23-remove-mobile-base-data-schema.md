# 2026-10-23 移除底盘/升降与双臂数据 schema

## 范围

- 主窗口不再导入、创建、连接或关闭底盘/升降面板；删去对应 Tab 和 VR 到底盘的联动。VR 控臂与录制快捷键仍保留。
- LeRobot 遥操作及 rollout/RECAP 录制共用固定 schema：`observation.state` 和 `action` 均为 `[arm1_pose7, arm2_pose7]`（14 维）；`observation.gripper_torque` 为两臂各一维；图像保留 cam1/cam2/cam3 三路。`action[t]` 仍取未来 lookahead 帧，力矩仍取当前帧。
- 录制线程不再接受底盘状态回调。已有目录只有 state/action 均为 14 维、双力矩与三路视频规格一致时才可追加。旧 18 维目录会记录明确错误并原样保留；使用新目录录制。没有迁移或修改历史数据。
- 当前 VLA 执行器是单臂兼容路径，观测保持 pose7；动作入口严格要求每步 7 维且数值有限。11/14/18 维输出均在动作下发前拒绝，避免历史 18 维模型被截取前 7 维执行。双臂 14 维模型的实际部署需要独立完成双臂观测、三相机输入和双臂动作执行适配后人工核验；不能在当前单臂执行器中使用。
- RECAP `SidecarDataset` 的默认 action 维数改为 14；默认夹爪豁免同时作用于下标 6、13。RECAP 维度缩放明确拒绝 18 维，单臂 7 维仍可显式指定。

## 离线验证

- `scripts/tests/test_remove_mobile_base_offline.py`：固定 schema、双臂 pose、历史目录不写入、动作维度拒绝、生命周期无底盘调用、RECAP 双夹爪。
- `scripts/tests/test_dual_arm_control_offline.py`：原 18 项安全回归。
- `py_compile` 与 `git diff --check`。

未启动 GUI、CAN、摄像头或真机，也未读取/改动模型权重与数据集。现场仍需人工检查主窗口启动、VR 控臂、双臂录制与 rollout 的实际数据帧，并用匹配 14 维格式的新模型验证完整双臂部署路径。历史 18 维权重/数据需要匹配原设备或重新训练，不能隐式裁剪。

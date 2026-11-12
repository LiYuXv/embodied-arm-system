# 2026-10-23 双 Joy-Con 上位机迁移

## 来源与范围

只读参考 `/home/liyu/el_a3_teacher/el_a3_sdk_RTC_VR（另一个复件）` 的 2026-09-15 已验收 Joy-Con 实现及开发日志。当前工作树原有未提交的双臂安全停机、Joint14 录制和可选底盘改动全部保留。本次只加入 Joy-Con 遥操；没有修改底盘开关、Joint14 格式、摄像头选择、横屏或 VLA 双臂执行，也没有提交或推送。

## 控制链路

- 固定绑定：Joy-Con L → `worker.arm` / `can0`，Joy-Con R → `worker.arm2` / `can1`。两个面板没有切换侧别控件，连接发现结果的 `side` 必须与面板一致。
- 保留参考 `JOYCON_PROFILES`、`JOYCON_ROTATION_FIX` 与 `JOYCON_ROTATION_VECTOR_SIGN=[-1,+1,-1]`、标准及实验 IMU 动作模式、夹爪、姿态归中、摇杆归零和 L/R 回工作位逻辑。evdev 仅在设备发现/读取时导入，pinocchio 仅在控制器实例化时导入。
- 已验收的 Joy-Con 控制器放在独立 `demo/joycon_arm_control.py`，避免改变现有 Quest VR 控制器。核对本仓库 SDK 具有 `_get_kinematics()`、`start_control_loop`、`control_loop_running`、`GetArmJointMsgs`、`GetArmJointEfforts`、`JointCtrl(..., velocities=...)`、`GripperCtrl` 及 `ArmEndPose`。Joy-Con 启动要求 worker 已为对应真臂使能且取得控制权；首次遥操时若该臂 SDK 控制循环尚未运行，面板按需以 200 Hz 启动并核查运行状态，已运行则不重复启动。未改变 worker 的通用电机使能流程。
- 每只手柄单独连接、模式、状态、遥操和日志。使能门控读取 `_connected/_enabled` 与 `_arm2_connected/_arm2_enabled`，拒绝模拟模式，不使用聚合 `worker.is_enabled` 判定某一臂。单臂失败不改变另一臂的面板状态。

## 停机与互锁

- 主窗口的安全退出和窗口关闭继续调用现有 `worker.stop(before_shutdown=...)`；回调先停 Joy-Con L/R 控制线程及输入线程，再清理 Xbox、VR、VLA。控制或输入线程 join 超时会留下运行对象并报告失败，`worker.stop()` 不会宣称安全退出。
- 工具栏失能、断开和急停先停两只 Joy-Con。失能/断开在停止未确认时拒绝继续；急停仍下发并明确报告可能滞后控制。worker 的原有 shutdown 顺序未改。
- Joy-Con 启动时检查同臂 Xbox、VR、VLA 占用；VLA 即使暂停推理控制仍保持占用，避免已下发轨迹与 Joy-Con 重叠。Xbox、VR、VLA 启动、VLA 恢复控制、单次动作请求以及单次 chunk 下发前检查 Joy-Con 占用，拒绝并发写 SDK。VLA 现有执行路径仅是左臂，VR 双臂，因此右臂 Joy-Con 不阻断左臂 Xbox/VLA。

## 离线验证

- 参考 `scripts/tests/test_joycon.py` 复制后只把控制器导入指向独立 Joy-Con 模块，23 项通过，涵盖手柄发现、profiles、轴置换和符号、标准/IMU 模式、夹爪与固定 L/R 绑定。
- `test_joycon_gui_offline.py` 扩至 10 项，覆盖对应侧 SDK 循环按需启动/不重复启动、启动异常或状态未运行时无面板控制线程、VLA 单次请求与 chunk 下发互锁，以及原有独立使能/模拟模式、错误侧别/无 evdev、join 超时、失能/断开/急停门控和懒加载检查。所有设备与线程控制以 mock 验证，不打开设备。
- 原有 `test_optional_base_data_schema_offline.py` 13 项、`test_dual_arm_control_offline.py` 18 项、`test_remove_mobile_base_offline.py` 6 项通过；`py_compile` 与 `git diff --check` 通过。原双臂清理测试增加了 Joy-Con L/R 两个源的调用检查。
- 系统 `python3` 运行 `test_joycon.py` 的 23 项通过。`lerobot` Python 3.12 环境运行同测试时，系统 ROS 的 `pinocchio` 扩展无法导入，且姿态矩阵零附近严格相对误差断言失败；该环境的其余四组离线测试通过。使用 `QT_QPA_PLATFORM=offscreen`、`lerobot` Python 和 `/tmp/hei-rebot-gui-deps-20261023` 的隔离 scipy，成功导入并分别构造 `JoyConPanel('left')`、`JoyConPanel('right')`，未运行主上位机或接入硬件。

## 尚存风险

本分支迁移后**未经过实机验证**，也未连接手柄、CAN 或摄像头。完整主界面渲染、evdev 设备线程退出时序、CAN 控制循环与急停的现场时序仍须在受控环境验证；控制或输入线程超时会阻止安全退出确认。

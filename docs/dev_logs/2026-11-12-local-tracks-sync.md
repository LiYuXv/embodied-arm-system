# 2026-11-12：本地双路线代码同步与毕设进度校正

## 目的与代码归属

此前 `main` 的 2026-09-21 双臂任务规划已经不能完整代表本地工作。按本地工作树、原有开发日志及本次核对结果，将两条路线明确分开：

| 路线 | 开发起点 | 功能分支 | 近期目标 |
| --- | --- | --- | --- |
| 单臂 π0.5 / RECAP | 老师的双臂上位机，另增单臂工作模式（本地 `/home/liyu/hei-rebot-lift-A3-codex`） | [`feat/hei-rebot-lift-a3`](https://github.com/LiYuXv/embodied-arm-system/tree/feat/hei-rebot-lift-a3/hei-rebot-lift-A3) | 先验证单臂云端 π0.5 推理及本地执行，再做单臂 RECAP |
| 双臂 VLA | 老师原单臂 RTC/VLA SDK，扩展双臂控制与 VLA（本地 `/home/liyu/el_a3_teacher/el_a3_sdk_RTC_VR（另一个复件）`） | [`feat/dual-arm-system`](https://github.com/LiYuXv/embodied-arm-system/tree/feat/dual-arm-system) | 双臂 14D VLA、模型与真机协作执行验证 |
| 基础仿真与项目总览 | `/home/liyu/embodied-arm-system` | `main` | ROS 2 / MoveIt 2 / Gazebo 历史成果与项目文档；不把不同 SDK 工作树直接混合 |

这三者是同一毕业设计下的不同工作空间。功能分支尚未合入 `main`，`main` 的说明和里程碑不等同于已将所有 SDK 源码合并。

## 已有里程碑与日期

- **7 月**：完成 ROS 2 + MoveIt 2 + Gazebo Classic 的中文指令、红蓝目标感知和抓放任务闭环；见 `2026-07-30-pick-place-completion.md`。
- **9 月 3 日**：真实 EDULITE_A3 零力矩示教、CAN、双 USB RGB 和 LeRobotDataset v3.0 单臂样例打通；见 `2026-09-03-real-robot-lerobot-recording.md`。
- **9 月 15 日**：双 Joy-Con 真机遥操完成现场验证，左右臂分别执行 XYZ、姿态及夹爪动作；见 `2026-09-15-dual-arm-joycon-teleoperation.md`。
- **9 月 23—28 日，双臂 VLA 路线**：在单臂 SDK 上扩展双臂 GUI、14D 状态/动作模型、推理线程、VLA 控制器和执行保护；记录 Joy-Con IMU 映射、控制和界面调整。曾使用真实 checkpoint 离线取得形状为 `(50, 14)` 的有限值动作结果；**不是双臂 VLA 真机自动执行验收**。本次复跑 VLA 控制器 12、执行器 2、运动交接 3，共 **17 项离线测试通过**。
- **10 月 23—28 日，单臂 RECAP 路线**：将既有双臂上位机中的控制、Joy-Con、相机语义绑定、LeRobot 录制和 RECAP 元数据逻辑补齐，并加入单/双臂模式；单臂规定 **cam1=臂1腕部、cam2=头部、state/action=末端 7D**，双臂仍为 **cam1=臂1腕部、cam2=头部、cam3=臂2腕部、joint/action=14D**。单臂 rollout 独立存放，避免误与双臂数据混录。更详细结果见该功能分支 `hei-rebot-lift-A3/docs/dev_logs/2026-10-28-single-dual-arm-mode.md`；本次复跑 **91 项离线测试通过**。
- **11 月机械结构**：现场已完成 EDULITE_A3 的 SolidWorks 改型；**尚未将新版 CAD 模型及相应 URDF / 运动学和碰撞参数整合进本仓库验证**。它属于机械结构工作，不等同于单臂 π0.5 或双臂 VLA 软件验收。

## 2026-11-12 GitHub 同步

- `feat/hei-rebot-lift-a3`：同步双臂上位机延续代码、单臂模式、相机角色、离线测试和 October 开发日志；提交 `dbe45c3`。
- `feat/dual-arm-system`：同步双臂 VLA runtime、控制器、UI、双臂遥操及离线测试、September 修改记录；提交 `03c0f7b`。
- `main`：更新项目首页、本记录与下一阶段计划；**不自动合并两条代码路线**。
- 没有上传本地相机录制视频、LeRobot 数据集、模型权重/checkpoint、个人设备设置、调试临时文件，也没有操作真实机械臂。

## 当前验证边界与风险

1. **单臂 π0.5 / RECAP**：两相机、7D 末端动作、旧云端协议及 RECAP 标注链路已经离线回归。云端模型是否对应相同末端动作语义（坐标系、单位、姿态和夹爪）、云端结果实际延迟、真实 USB/CAN、RTC 与单臂真机动作，仍需现场验证。RECAP 的真实 rollout/标注实验也未完成。
2. **双臂 VLA**：双臂控制器、14D 安全检查及 checkpoint 离线测试属于软件验证；真机自动协作执行、碰撞防护、正式训练后的成功率实验尚未验收。离线的数值限幅不能替代运动空间和自碰撞检查。
3. **格式严格区分**：单臂 `[x,y,z,rx,ry,rz,gripper]` 与双臂关节 `14D` 不可直接混用，RECAP 标注、录像和 checkpoint 需分别验证。
4. **测试范围**：单臂分支的 91 项 `test_*_offline.py` 回归通过；其开发日志另记录 Joy-Con IMU 专项测试在某一 Python 环境存在历史断言失败，此结果不能用 91 项覆盖或隐瞒。
5. **新版机械臂**：CAD 结构更改后仍需重新核对运动学、零位、碰撞和安全工作空间，不能沿用旧机械臂参数直接做危险动作。

## 下一步优先顺序

1. **P0 / 单臂**：确认现有 π0.5 checkpoint 语义、cam1/cam2 USB 绑定及 can0 实际反馈，先做云端 Dry-run（不使能动作），再在有急停保护下验证安全执行。
2. **P0 / 单臂**：π0.5 真机执行得到确认后，再开展单臂 RECAP rollout、success/source/policy/group 标注与统计。
3. **P1 / 双臂**：继续完成 14D checkpoint 语义、协作任务、运动安全和双臂 VLA 真机实验。
4. **工程并行**：新版 CAD 的 URDF/碰撞及标定更新；在真实实验验收后回填文档。

本记录主要依据本地 Git 代码、原始开发日志和 2026-11-12 实际执行的离线测试；未执行真机运动、远端模型推理或重新训练。
# 第五步：单臂 / 双臂工作模式（2026-10-28）

## 范围与安全边界

只修改当前 Codex 工作副本，保留进入本步前的未提交改动、第四步横屏布局和全部 10-23 / 10-28 既有日志。只读对照 `/home/liyu/el_a3_teacher/el_a3_sdk_RTC_VR（另一个复件）/debugger/widgets/camera_pyqt.py` 的单臂数据格式，没有复制或替换 SDK。

没有启动真实 CAN、连接/使能/运动机械臂，没有实际运行模型推理、SSH 或访问云服务器，没有终止既有上位机，没有提交、推送。当前实际配置仍为 `auto_connect_on_startup=false`、`mobile_base_enabled=false`，USB 绑定未被测试改写。Qt 测试在独立 offscreen 进程运行，设备入口全部 mock，配置与测试数据都在临时目录。

## 模式、互锁与界面

- 顶部新增明显的“单臂 / 双臂”选择框，窗口标题显示当前模式。`work_mode` 写入现有系统配置；旧配置缺键或值无效时默认单臂，配置缺失时也不再默认自动连接。
- 复用现有 `ArmWorker._do_connect(..., arm2=None)` 单臂接口。单臂工具栏不读取臂2配置、不查询 can1 状态；后端再次丢弃臂2参数，SocketCAN 只允许臂1 can0。模拟模式也不再伪造臂2连接、使能及反馈。
- 切换前检查连接/使能、SDK 实例、执行中与排队命令、轨迹和夹爪线程，以及录像、LeRobot/rollout 写盘线程、VLA/单次推理、示教零力矩/录制、标定、手柄/VR/Joy-Con 遥操。worker 用命令队列锁覆盖出队与执行中标记，避免连接命令出队后切换漏检。忙时提示先停止、停用并断开；配置保存失败回滚模式。
- 单臂隐藏臂2末端行、关节行、MoveJ 目标、夹爪控制/反馈、诊断/标定臂选择和 VR 臂2信息、右 Joy-Con；左 Joy-Con 继续操作臂1。切回双臂恢复原控件，清除旧臂2显示与示教缓存，单臂拒绝回放双臂示教轨迹。
- 保留横向左右分栏、纵向日志分隔条、滚动容器和原控制信号接线。末端状态独立横向滚动，避免缩窄窗口时挤出第二路相机；顶部工具栏可横向滚动，保证窄窗口仍能访问全部按钮。

## 相机与数据格式

| 工作模式 | state / action | 图像语义 | 夹爪力矩 |
| --- | --- | --- | --- |
| 单臂 | 7D `[x,y,z,rx,ry,rz,gripper_rad]`，与参考 Teacher 的末端 Euler 位姿一致 | cam1=臂1腕部，cam2=头部 | 1D |
| 双臂 | 原 joint14，左右各6关节+夹爪，单位rad | cam1=臂1腕部，cam2=头部，cam3=臂2腕部 | 2D |
| 原可选底盘 | 双臂原18D schema保留，当前配置关闭 | 原三路 | 原2D |

- 单臂只打开/预览已绑定的腕部与头部相机，不要求或打开臂2相机；第三列隐藏。USB 拓扑匹配、缺席不补位、重复角色/USB 拒绝、旧采集线程退出确认、录制/推理期间禁止重绑定与刷新等保护保留。
- 显式为单臂选中原来保留给臂2的 USB 时，可把该 USB 转为当前角色并记录提示；没有显式重选时保留臂2绑定，便于切回双臂。不会自动猜测实际腕部是哪台设备。
- 观测与录制都按角色表取帧，拒绝部分/重复/压缩后错位的角色表。VLA 固定 cam1 腕部、cam2 头部，与实际 USB 绑定一致，不按设备检测排序取图。
- 扩展现有队列录制器的 `single_arm` 参数，保留20Hz、`action[t]=state[t+5]` 和末尾使用最后可见反馈的规则；没有重新设计运动链路。单臂读取真实末端和夹爪反馈，无反馈不再补零，缺失夹爪力矩也拒绝开始录制。
- 单臂 feature names、robot_type 与旧 Teacher 一致：state=`["state"]`、action=`["action"]`、torque=`["torque"]`、robot_type=`ela3`；双臂 names 与 robot_type 不变。追加前继续核对 fps、robot_type、dtype、shape、names、图像键和编码，不允许7D/14D混录。
- teleop 与 rollout 共用同一模式数据通路、前瞻规则和 RECAP 标注逻辑。单臂 rollout 使用 `ela3_rollout_single`，双臂保留 `ela3_rollout`，避免已有14D rollout目录阻塞单臂录制；任务组候选随目录切换。旧数据不迁移、不覆盖。teleop 文件夹仍可在 UI 选择，旧目录不兼容时明确拒绝追加。

## VLA / RTC / 云端模型保护

- 新增不依赖硬件的 checkpoint 合约校验。启动本地推理前及模型线程加载时验证：当前单臂模式、两路 cam1/cam2 RGB 输入、state7/action7、单夹爪力矩；兼容原 ACT 的 pose7+torque8 输入约定。显式 joint-space、双臂模式/机器人类型、14/18D、多余相机、双夹爪模型均拒绝。
- 原本执行器就是单臂 Cartesian pose7，尚无双臂 joint14/18 执行器。本步明确禁止在双臂模式启动这条7D链路，不伪装为双臂执行器、不截断14D动作。双臂控制、示教与数据采集仍使用原 API。所有动作 chunk 仍经形状与有限值检查。
- 原本地/云端 VLA、RTC、trained RTC、advantage: positive、RECAP 接管标注功能保留在兼容单臂模式下；未加载模型或实际执行推理验证。
- **已撤销第五步引入的强制云端模型配置握手。** `PingReply`、pb2 与 `server.py` 均恢复到 Git 历史原协议（`fa7f2a0`），不再新增或回传 `model_config_json`。客户端不再要求 Ping/heartbeat 提供模型配置才允许 `InferChunk`；原 Ping 认证、token、图像尺寸能力通知、RTC 参数、reset/断链重连行为保留。
- **无需修改、部署或重启云服务器。** 原服务器可供客户端继续使用“先单臂 π0.5 云端推理、本地执行，随后单臂 RECAP 验证”的既定链路；本次只离线验证协议与客户端行为，没有连接服务器或执行模型推理。
- 原协议不能提供 checkpoint 的完整语义证明，客户端明确提示“checkpoint 动作语义未预先校验”。收到每次推理回复后必须先满足 `action_dim=7`、正数步数、长度精确匹配 `(num_steps, 7)` 且每帧所有数值有限，才发出可消费的动作 chunk；14D 绝不截断为 7D，执行入口继续做单臂模式、形状和有限值检查。本地 checkpoint 合约校验保留。
- **剩余风险：输出维度不能替代语义校验。** 合法 7D 也可能是关节、不同坐标系/单位或错误相机顺序，旧协议无法提前识别服务器重启后换成同维异义模型，也无法在请求前排除远端模型不兼容。实际执行前仍须人工确认 checkpoint 是 `[x,y,z,rx,ry,rz,gripper_rad]` 末端 Euler 位姿，以及相机顺序、坐标系、单位与训练约定；不能把逐回复形状检查视为已验证训练语义。

## 修改文件

- `debugger/main_window.py`：模式入口、互锁、持久化、隐藏/恢复与横屏适配。
- `debugger/backend/arm_worker.py`：模式互锁、命令执行标记、单臂CAN与模拟反馈保护。
- `debugger/backend/dual_arm_lerobot.py`：单臂schema和模式化队列写入；原双臂规则保留。
- `debugger/widgets/camera_pyqt.py`：双相机角色/观测、7D录制/rollout、真实反馈、VLA及云端保护。
- `debugger/widgets/toolbar_panel.py`、`joint_control_panel.py`、`gripper_panel.py`、`teaching_panel.py`、`settings_dialog.py`：最少的控件/布局引用、臂选择与不兼容轨迹保护。
- `debugger/utils/work_mode.py`、`mode_ui.py`、`system_config.py`：模式/模型合约、UI小工具与安全默认值。
- `debugger/cloud_inference/server.py`、`vla_inference.proto`、`vla_inference_pb2.py`：本次撤销元数据增量，逐字节恢复 Git 原协议版本，最终相对 HEAD 无差异。
- `scripts/tests/test_single_dual_arm_mode_offline.py`：第五步原24项保留其他模式回归，将两项强制握手测试改为旧协议兼容/逐回复拒绝测试，另补3项云客户端回归，当前共27项。
- `scripts/tests/test_camera_binding_offline.py`：原测试夹具增加角色就绪方法和显式映射，保持原64项回归有效。
- 本日志。其余进入本步前已有修改没有覆盖或撤销。

## 验证结果

1. 使用现有 lerobot Python：
   ```bash
   env -u PYTHONPATH QT_QPA_PLATFORM=offscreen \
     ~/miniforge3/envs/lerobot/bin/python -m unittest discover \
     -s scripts/tests -p 'test_*_offline.py' -q
   ```
   撤销握手前先运行原回归：**88 tests passed**，无跳过。修改后同一命令：**91 tests passed**，无跳过（原88项的回归范围保留，其中两项替换强制握手预期，另补3项）。覆盖互锁/待执行命令、can0且不触碰can1、控件隐藏恢复与持久化/失败回滚、腕部/头部USB语义与观测顺序、缺席拒绝与显式改绑、7D/14D及双/三相机、teleop/rollout启动与帧推入、未来动作标签、不兼容本地checkpoint、RTC/positive参数保留，以及旧云协议兼容和错误回复拒绝。
2. 实际在临时目录通过内置 LeRobot v3 写入器、CPU H.264编码生成7D/双视频与14D/三视频数据，读取parquet核对维度与前瞻动作，恢复已有数据集，验证错误模式追加被拒且原info.json未变；实际写入并读取RECAP success/source/policy/group标注。不是仅断言按钮或维度常量。
3. PyQt offscreen完整主窗口在1640×940、1180×760渲染单臂，1640×940渲染双臂；无设备入口执行，两路预览在窄窗口完整，左右与日志分隔条保留。截图仅留在`/tmp`。
4. 16个相关Python文件语法检查通过；`git -c core.whitespace=cr-at-eol diff --check`通过。项目原文件存在混合CRLF，本步按原有行尾保留未修改行。
5. 额外检查 `scripts/tests/test_joycon.py`：23项中，1个测试方法的15个旋转子例失败、0 errors（包括近零浮点旋转与IMU锚点断言）。该测试实际加载的业务模块与本步修改模块交集为空，属于进入本步前已有控制器/测试问题；没有擅自修改Joy-Con运动控制。上述88项离线回归中的Joy-Con GUI检查继续通过。真机使用IMU实验模式前应另行排查。

## 本次撤销握手的专项离线验证

- 单独运行云客户端相关测试：
  ```bash
  env -u PYTHONPATH QT_QPA_PLATFORM=offscreen \
    ~/miniforge3/envs/lerobot/bin/python -m unittest discover \
    -s scripts/tests -p test_single_dual_arm_mode_offline.py -k cloud -v
  ```
  **6 tests passed**，无跳过（包含在上述91项中）。使用原协议真实 protobuf 消息和完全 mock 的 grpc/channel/stub，直接同步调用客户端逻辑，不启动网络服务或推理线程；未运行会连接真实服务器的 `debugger/cloud_inference/test_client_thread.py` 或 `smoke_test.py`。
- 普通/单次请求都验证旧 PingReply 无扩展字段可认证、心跳和请求；合法7D回复可消费，14D、伪标7D但含14D数据、长短/空载荷、非正步数及后续帧的 NaN/±Inf 均不发出动作信号。模拟断链/重连后 reset 保留，再次拒绝14D、消费合法7D；token 错误、Ping/heartbeat error、RPC失败和过期回复不会放行动作。
- 训练RTC能力、延迟上下限钳制、`rtc_k`、`avg_n`/超时、positive前缀、暂停恢复reset、单次诊断参数/观测和图像尺寸通知均离线检查；没有使用真实CAN、摄像头或云端。
- 本次涉及的4个Python文件仅编译语法检查通过；`git -c core.whitespace=cr-at-eol diff --check`通过。协议/服务三文件与HEAD逐字节一致；开始时的文件哈希与最终差异核对表明，本次仅修改这三文件、`camera_pyqt.py`、该离线测试和本日志，其余已有改动及第四步布局保留。

## 真机待测

- 人工确认“臂1腕部 / 头部”的真实USB端口并从UI绑定；现存配置只保存了head/arm2，本步没有猜测或重写真实角色。单臂启动后确认完全没有can1错误或操作，再按现场安全流程人工连接can0并核对反馈。
- 在具备机械支撑及现场急停条件下，人工验证单臂使能/失能、夹爪、普通/重力示教、单臂Joy-Con/手柄/VR；切回双臂确认全部原控件与反馈恢复，活动时切换被拒绝。
- 验证真实两路USB带宽、断插恢复、刷新、预览及20Hz长录制；实际读取新数据集训练前确认7D末端位姿与14D关节语义。
- 不改云服务器，在后续现场授权后先人工核对现有单臂 π0.5 checkpoint 的末端动作语义、相机顺序/坐标系/单位，再验证云端推理本地执行、RTC/断链恢复/人工接管，随后单臂 RECAP rollout/标注；本次没有真实云端或硬件推理执行。双臂模型部署仍需另行实现并验证匹配执行器。

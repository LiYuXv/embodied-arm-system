# 第四步：上位机横屏布局优化（2026-10-28）

## 范围与目标
只修改 Codex 工作副本 `~/hei-rebot-lift-A3-codex/hei-rebot-lift-A3`。保留原有双臂控制、Joy-Con、RECAP、云端推理和三路摄像头角色绑定，暂不修改控制链路，也不触碰硬件。

## 实施
- `debugger/main_window.py`：窗口改为横屏默认 1640×940（最小 1120×700），顶部双臂连接/急停工具栏保持固定；中间水平 QSplitter 左为机械臂状态、三路摄像头、录制和 VLA，右为关节/轨迹/示教/诊断/夹爪/手柄/VR/双 Joy-Con 分页；下方垂直 QSplitter 为日志，可拖动调整高宽。左、右区域分别具备纵向滚动。
- `debugger/widgets/camera_pyqt.py`：现有“分辨率/采集录制”“VLA 控制”“RTC/云端”长控制行加独立横向滚动，避免缩窄时控件互相挤压；摄像头下拉框、USB 角色绑定及原有回调不变。
- `run.sh`：使用本机 `lerobot` 环境，去除 ROS Humble Python 3.10 路径对 Python 3.12 的污染，并优先使用环境的 cmeel Pinocchio 动态库；已赋予脚本执行权限。
- `debugger/settings/main_system_config.json`：当前仅用于布局调试，`auto_connect_on_startup` 置 false，防止打开界面时自动连接和使能机械臂；底盘/升降默认关闭。

## 验证
- `python -m py_compile` 通过；`./run.sh --help` 正常；`QT_QPA_PLATFORM=offscreen ./run.sh` 实际启动，无启动错误。
- PyQt 离线渲染 1640×940、缩小至 1180×760 均成功。横向分隔条从 [1045,587] 拖到 [880,752]，纵向从 [644,181] 拖到 [540,285]，左右滚动条工作正常。
- 8 个分页可切换；3 个摄像头角色下拉框与 VLA/推理录制按钮保留。
- `python -m unittest discover -s scripts/tests -p 'test_*_offline.py'`：64 tests passed。
- `git -c core.whitespace=cr-at-eol diff --check` 通过；文件原有换行符混用，普通 diff --check 会将新增 CRLF 行误报为尾随空格。

## 后续
第四步布局离线验证完成。下一步在真实桌面启动 Codex 版，只绑定摄像头与单臂 CAN 并检查状态反馈；先不启动云端推理或执行动作。注意该副本的三路摄像头绑定须根据真实 USB 连接重新核对。
未提交、未推送任何仓库。

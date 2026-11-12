# 三路摄像头主界面角色绑定（离线阶段）

## 范围

在当前老师上位机独立工作副本中修改 `CameraWidget` 和 `SettingsDialog`。保留已有底盘数据格式、双 JoyCon、安全退出以及 CameraWidget 的 LeRobot、RECAP、云 VLA 实现；没有修改真实 `settings/main_system_config.json`，没有真机探测、提交或推送。

## 行为

- 预览从左到右固定为臂2腕部、头部、臂1腕部。每路有「未分配」和已检测可用的 `/dev/videoN (usb 端口)` 候选；主界面提供刷新检测、应用绑定。
- 保存沿用 `main_system_config.json` 的 `camera_roles`，按 sysfs 中 USB 接口拓扑路径匹配。缺席 USB 的旧角色默认保留，并在下拉框标为缺席；可明确选择「未分配」解除绑定。拿不到 USB 身份时拒绝持久绑定并提示。未手动绑定的设备不会因索引顺序自动占据角色。
- 三个角色全部匹配时，内部录制/推理索引才按 `arm1, head, arm2` 建立。普通录像文件名按 `cam1, cam2, cam3`，与 `videoN` 索引无关。预览顺序独立为 `arm2, head, arm1`。
- 录像、LeRobot 写入、rollout 写入、VLA 和单次动作运行期间拒绝应用或刷新。设置对话框保存角色变化也走同一忙状态检查。重检测或角色切换时先等待采集线程退出，再释放 VideoCapture；线程未退出就取消重开，避免双开和销毁活跃 QThread。每路重开有独立令牌，丢弃旧线程排队到达的帧。
- 设置对话框与主界面读取、写入同一份 `camera_roles`；设置保存后主窗口调用 `apply_camera_roles()` 更新主界面。检测列表为空时设置对话框不再回退到独立 sysfs 列表。

## 离线验证

`scripts/tests/test_camera_binding_offline.py` 用 AST 提取与 mocks 测试无设备、三路选择、重复设备、USB 索引漂移、缺席 USB、保存失败、运行中拒绝、线程释放顺序、设置对话框互通、帧和录像名语义。测试不构造 CameraWidget，也不读取真实摄像头或运行配置。

使用 `/home/liyu/.venvs/el_a3_debugger/bin/python` 并保留现有 `PYTHONPATH` 运行全部指定测试：87 项通过（新增 17 项、JoyCon 23 项、GUI 10 项、双臂 18 项、底盘格式 13 项、移除底盘 6 项）。`py_compile` 与 `git diff --check` 通过。

## 真机待验证

三路 USB 摄像头实际 sysfs 端口标识、重插及重启后的索引漂移、检测耗时、采集线程在驱动 `read()` 阻塞时的停止行为、设置对话框改绑后的视频重开和录像/推理画面语义。真机验证时需确认各摄像头的实际 arm1/head/arm2 物理安装位置。

"""手柄控制面板：设备连接 + 输入数据监控 + 手柄控臂

本模块提供调试上位机中的“手柄”分页，支持：
1. 扫描并连接 Linux 下的 /dev/input/js* 游戏手柄设备。
2. 实时显示手柄各轴（摇杆、扳机）和按键状态。
3. 根据手柄配置文件（profile）将原始输入映射为语义量（LX/LY/RX/RY 等）。
4. 在连接机械臂后，启动后台线程通过 XboxArmController 控制机械臂运动。
"""

import glob
import threading
import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QGroupBox, QProgressBar,
    QScrollArea, QFrame,
)
from PyQt6.QtCore import pyqtSignal, QTimer, Qt

# 工具模块：多语言翻译、主题管理、颜色表
from debugger.utils.i18n import tr
from debugger.utils.theme_manager import ThemeManager
from debugger.utils.style import SCENE_COLORS

# SDK 底层手柄与手柄配置文件
from el_a3_sdk.joystick import LinuxJoystick
from el_a3_sdk.controller_profiles import (
    PROFILES, detect_controller, ControllerDetection,
)

logger = logging.getLogger("debugger.gamepad")


class _AxisBar(QProgressBar):
    """双向进度条，用于显示 -1.0 ~ 1.0 的轴值

    将 [-1, 1] 映射到 [0, 2000]，0 对应中间位置，用于直观展示摇杆/扳机偏移。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 2000)       # 0 表示 -1.0，2000 表示 +1.0，1000 表示 0
        self.setValue(1000)          # 初始置中
        self.setTextVisible(False)   # 不显示百分比文字
        self.setFixedHeight(12)
        self.apply_theme()           # 应用当前主题样式

    def apply_theme(self):
        """根据当前主题更新进度条颜色。"""
        sc = SCENE_COLORS[ThemeManager.instance().theme]
        self.setStyleSheet(
            f"QProgressBar {{ background: {sc['progress_bg']}; border: 1px solid {sc['progress_border']};"
            f"  border-radius: 4px; }}"
            f"QProgressBar::chunk {{ background: {sc['progress_chunk']}; border-radius: 3px; }}"
        )

    def set_value(self, v: float):
        """设置轴值 v（范围 [-1.0, 1.0]）。"""
        mapped = int((v + 1.0) * 1000)
        self.setValue(max(0, min(2000, mapped)))


class _ButtonIndicator(QLabel):
    """单个按钮指示灯

    以 QLabel 小方块形式显示某一手柄按键是否被按下。
    """

    def __init__(self, index: int, parent=None):
        super().__init__(str(index), parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pressed = False
        self.apply_theme()

    def apply_theme(self):
        """根据按键状态（按下/松开）应用主题色。"""
        sc = SCENE_COLORS[ThemeManager.instance().theme]
        if self._pressed:
            self.setStyleSheet(
                f"background: {sc['indicator_on_bg']}; border: 1px solid {sc['indicator_on_border']};"
                f"border-radius: 4px; min-width: 24px; min-height: 16px; max-height: 16px;"
                f"color: {sc['indicator_on_fg']}; font-size: 10px; font-weight: bold;"
                "qproperty-alignment: AlignCenter;"
            )
        else:
            self.setStyleSheet(
                f"background: {sc['indicator_off_bg']}; border: 1px solid {sc['indicator_off_border']};"
                f"border-radius: 4px; min-width: 24px; min-height: 16px; max-height: 16px;"
                f"color: {sc['indicator_off_fg']}; font-size: 10px;"
                "qproperty-alignment: AlignCenter;"
            )

    def set_pressed(self, pressed: bool):
        """更新按键状态并重绘样式。"""
        self._pressed = pressed
        self.apply_theme()


class GamepadPanel(QWidget):
    """手柄控制 Tab 面板

    包含三个区域：
    - 连接区：选择 /dev/input/js* 设备、选择 profile、连接/断开。
    - 监控区：显示所有轴和按钮的实时状态，以及映射后的语义值。
    - 控制区：启动/停止手柄控臂，显示当前速度档、控制模式和目标末端位姿。
    """

    # 向主窗口输出日志信息的信号
    gamepad_log = pyqtSignal(str)
    # 手柄连接/断开信号（用于与 VLA 推理控制互锁）
    gamepad_connected = pyqtSignal()
    gamepad_disconnected = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- 运行期状态 ---
        self._joystick: Optional[LinuxJoystick] = None   # 当前连接的手柄对象
        self._arm = None                                 # 机械臂接口实例（由主窗口设置）
        self._controller = None                          # XboxArmController 实例
        self._ctrl_thread: Optional[threading.Thread] = None  # 手柄控臂后台线程
        self._detection: Optional[ControllerDetection] = None  # 手柄型号识别结果

        # 默认速度档位（0~4，默认 2 即中速）
        self._speed_idx = 2
        # 手柄控臂后台线程是否正在运行
        self._ctrl_running = False

        # 夹爪力矩限制（Nm）：由夹爪面板输入框实时写入，启动控臂时同步给控制器
        self._gripper_torque_max = 0.5
        self._gripper_torque_min = -0.5

        self._init_ui()

        # 创建 20Hz 轮询定时器，用于刷新手柄输入显示
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)  # 50ms，即 20 Hz
        self._poll_timer.timeout.connect(self._poll_input)

    def _scene(self):
        """获取当前主题对应的颜色表。"""
        return SCENE_COLORS[ThemeManager.instance().theme]

    @staticmethod
    def _speed_level_specs():
        """定义 5 档速度：[(显示名称, 速度系数), ...]

        速度系数会传给 XboxArmController，用于缩放末端运动/关节运动速度。
        """
        return [
            (tr("gp.speed_very_slow"), 0.10),
            (tr("gp.speed_slow"), 0.25),
            (tr("gp.speed_medium"), 0.50),
            (tr("gp.speed_fast"), 0.75),
            (tr("gp.speed_max"), 1.00),
        ]

    def _apply_monitor_header_styles(self):
        """应用监控区标题与信息标签的样式（主题切换时调用）。"""
        sc = self._scene()
        self._axes_title.setStyleSheet(
            f"font-weight: bold; color: {sc['accent']}; margin-top: 2px;")
        self._buttons_title.setStyleSheet(
            f"font-weight: bold; color: {sc['accent']}; margin-top: 4px;")
        self._mapped_title.setStyleSheet(
            f"font-weight: bold; color: {sc['accent']}; margin-top: 4px;")
        self._mapped_display.setStyleSheet(
            f"font-family: monospace; font-size: 11px; color: {sc['mapped_fg']}; "
            f"background: {sc['mapped_bg']}; padding: 4px; border-radius: 3px;"
        )
        self._info_label.setStyleSheet(
            f"color: {sc['subtext']}; font-size: 11px;")

    # ---- UI 构建 ----

    def _init_ui(self):
        """初始化整体布局：使用 QScrollArea 包裹三个分组。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)            # 内部控件可随滚动区缩放
        scroll.setFrameShape(QFrame.Shape.NoFrame)  # 无边框
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(4)

        # 依次添加三个分组
        self._conn_group = self._build_connection_group()
        layout.addWidget(self._conn_group)
        self._monitor_group = self._build_monitor_group()
        layout.addWidget(self._monitor_group)
        self._ctrl_group = self._build_control_group()
        layout.addWidget(self._ctrl_group)
        layout.addStretch()

        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # 应用翻译文本与主题样式
        self.retranslate_ui()

    def _build_connection_group(self) -> QGroupBox:
        """构建设备连接分组（紧凑两行）。"""
        grp = QGroupBox()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 2, 6, 4)
        layout.setSpacing(2)

        # 第一行：设备 + 刷新 + Profile + 连接按钮
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        self._lbl_device = QLabel()
        row1.addWidget(self._lbl_device)
        self._dev_combo = QComboBox()
        self._dev_combo.setEditable(True)      # 允许手动输入路径
        self._dev_combo.setMinimumWidth(110)
        row1.addWidget(self._dev_combo, 1)

        self._refresh_btn = QPushButton()
        self._refresh_btn.setFixedWidth(40)
        self._refresh_btn.clicked.connect(self._scan_devices)
        row1.addWidget(self._refresh_btn)

        row1.addWidget(QLabel("Profile:"))
        self._profile_combo = QComboBox()
        self._profile_combo.addItem("auto", "auto")  # 自动识别
        for pid, prof in PROFILES.items():
            self._profile_combo.addItem(prof.display_name, pid)
        row1.addWidget(self._profile_combo, 1)

        self._connect_btn = QPushButton()
        self._connect_btn.setObjectName("connectBtn")  # 用于 QSS 主题区分
        self._connect_btn.clicked.connect(self._toggle_connection)
        row1.addWidget(self._connect_btn)
        layout.addLayout(row1)

        # 第二行：已连接设备信息
        self._info_label = QLabel()
        layout.addWidget(self._info_label)

        grp.setLayout(layout)
        self._scan_devices()  # 初始化时扫描一次设备
        return grp

    def _build_monitor_group(self) -> QGroupBox:
        """构建输入监控分组（轴两列排布、按钮一行 16 个、映射单行）。"""
        grp = QGroupBox()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 2, 6, 4)
        layout.setSpacing(2)

        # 轴显示网格：A0~A7 两列排布（左列 A0~A3，右列 A4~A7）
        self._axis_bars: list[_AxisBar] = []
        self._axis_labels: list[QLabel] = []
        axes_grid = QGridLayout()
        axes_grid.setHorizontalSpacing(4)
        axes_grid.setVerticalSpacing(1)
        half = (LinuxJoystick.MAX_AXES + 1) // 2
        for i in range(LinuxJoystick.MAX_AXES):
            col_base = 0 if i < half else 3
            row = i if i < half else i - half
            name_lbl = QLabel(f"A{i}:")
            name_lbl.setFixedWidth(24)
            name_lbl.setStyleSheet("font-size: 10px;")
            bar = _AxisBar()
            val_lbl = QLabel(" 0.000")
            val_lbl.setFixedWidth(46)
            val_lbl.setStyleSheet("font-family: monospace; font-size: 10px;")
            axes_grid.addWidget(name_lbl, row, col_base)
            axes_grid.addWidget(bar, row, col_base + 1)
            axes_grid.addWidget(val_lbl, row, col_base + 2)
            self._axis_bars.append(bar)
            self._axis_labels.append(val_lbl)
        axes_grid.setColumnStretch(1, 1)
        axes_grid.setColumnStretch(4, 1)
        layout.addLayout(axes_grid)

        # 按钮指示灯：标题 + 一行 16 个指示灯
        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)
        self._buttons_title = QLabel()
        btn_row.addWidget(self._buttons_title)
        self._btn_indicators: list[_ButtonIndicator] = []
        for i in range(LinuxJoystick.MAX_BUTTONS):
            ind = _ButtonIndicator(i)
            btn_row.addWidget(ind)
            self._btn_indicators.append(ind)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # 映射值：标题 + 单行显示
        mapped_row = QHBoxLayout()
        mapped_row.setSpacing(4)
        self._mapped_title = QLabel()
        mapped_row.addWidget(self._mapped_title)
        self._mapped_display = QLabel("—")
        mapped_row.addWidget(self._mapped_display, 1)
        layout.addLayout(mapped_row)

        # 占位，保持 retranslate_ui 兼容
        self._axes_title = QLabel()
        self._axes_title.hide()

        grp.setLayout(layout)
        return grp

    def _build_control_group(self) -> QGroupBox:
        """构建控制分组（紧凑两行）。"""
        grp = QGroupBox()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 2, 6, 4)
        layout.setSpacing(2)

        # 第一行：启停按钮 + 速度档 + 控制模式
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        self._ctrl_btn = QPushButton()
        self._ctrl_btn.setObjectName("enableBtn")
        self._ctrl_btn.setEnabled(False)  # 未连接手柄/机械臂时禁用
        self._ctrl_btn.clicked.connect(self._toggle_control)
        row1.addWidget(self._ctrl_btn)

        self._lbl_speed_level = QLabel()
        row1.addWidget(self._lbl_speed_level)
        self._speed_label = QLabel()
        self._speed_label.setStyleSheet("font-weight: bold;")
        row1.addWidget(self._speed_label)

        self._lbl_ctrl_mode = QLabel()
        row1.addWidget(self._lbl_ctrl_mode)
        self._mode_label = QLabel("—")
        row1.addWidget(self._mode_label)
        row1.addStretch()
        layout.addLayout(row1)

        # 第二行：目标末端位姿
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        self._lbl_end_pose = QLabel()
        row2.addWidget(self._lbl_end_pose)
        self._pose_label = QLabel("—")
        self._pose_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        row2.addWidget(self._pose_label, 1)
        layout.addLayout(row2)

        grp.setLayout(layout)
        return grp

    def retranslate_ui(self):
        """根据当前语言重新设置所有可翻译文本（语言切换时调用）。"""
        self._conn_group.setTitle(tr("gp.conn_group"))
        self._lbl_device.setText(tr("gp.device"))
        self._refresh_btn.setText(tr("gp.refresh"))
        if self._joystick and self._joystick.connected:
            self._connect_btn.setText(tr("gp.disconnect"))
        else:
            self._connect_btn.setText(tr("gp.connect"))
        if self._joystick and self._joystick.connected and self._detection:
            det = self._detection
            self._info_label.setText(
                tr(
                    "gp.connected_info",
                    name=det.name or "unknown",
                    vendor=det.vendor,
                    product=det.product,
                    pname=det.profile.display_name,
                    source=det.source,
                )
            )
        else:
            self._info_label.setText(tr("gp.not_connected"))

        self._monitor_group.setTitle(tr("gp.monitor_group"))
        self._axes_title.setText(tr("gp.axes"))
        self._buttons_title.setText(tr("gp.buttons"))
        self._mapped_title.setText(tr("gp.mapped"))

        self._ctrl_group.setTitle(tr("gp.ctrl_group"))
        self._update_ctrl_btn_state()
        self._lbl_speed_level.setText(tr("gp.speed_level"))
        self._lbl_ctrl_mode.setText(tr("gp.ctrl_mode"))
        self._lbl_end_pose.setText(tr("gp.end_pose"))

        specs = self._speed_level_specs()
        self._speed_label.setText(
            f"{self._speed_idx + 1}/5 [{specs[self._speed_idx][0]}]")
        self._apply_monitor_header_styles()
        for bar in self._axis_bars:
            bar.apply_theme()
        for ind in self._btn_indicators:
            ind.apply_theme()

    # ---- 设备扫描 ----

    def _scan_devices(self):
        """扫描 /dev/input/js* 设备并填充下拉框；若无设备则保留默认 js0。"""
        self._dev_combo.clear()
        devices = sorted(glob.glob("/dev/input/js*"))
        if devices:
            for d in devices:
                self._dev_combo.addItem(d)
        else:
            self._dev_combo.addItem("/dev/input/js0")

    # ---- 连接/断开 ----

    def _toggle_connection(self):
        """连接按钮槽函数：根据当前状态切换连接/断开。"""
        if self._joystick and self._joystick.connected:
            self._disconnect_gamepad()
        else:
            self._connect_gamepad()

    def _connect_gamepad(self):
        """连接手柄：识别型号、打开设备、启动轮询定时器。"""
        device = self._dev_combo.currentText().strip()
        if not device:
            return

        profile_id = self._profile_combo.currentData()

        # 1. 先根据设备识别手柄型号与 profile
        try:
            self._detection = detect_controller(device, requested_profile=profile_id)
        except Exception as e:
            es = str(e)
            self._info_label.setText(tr("gp.profile_fail", e=es))
            self.gamepad_log.emit(tr("gp.profile_fail_log", e=es))
            return

        # 2. 打开手柄设备
        joy = LinuxJoystick(device=device)
        if not joy.connect():
            self._info_label.setText(tr("gp.cannot_open", device=device))
            self.gamepad_log.emit(tr("gp.connect_fail_log", device=device))
            return

        # 3. 连接成功，更新 UI 与状态
        self._joystick = joy
        det = self._detection
        self._info_label.setText(
            tr(
                "gp.connected_info",
                name=det.name or "unknown",
                vendor=det.vendor,
                product=det.product,
                pname=det.profile.display_name,
                source=det.source,
            )
        )
        self._connect_btn.setText(tr("gp.disconnect"))
        self._connect_btn.setObjectName("disconnectBtn")
        self._connect_btn.setStyle(self._connect_btn.style())  # 触发样式重载
        self._poll_timer.start()       # 启动 20Hz 输入轮询
        self._update_ctrl_btn_state()  # 更新控制按钮可用状态

        # 连接成功后自动开启手柄控臂（需机械臂已就绪）
        if self._arm is not None:
            self._start_control()

        self.gamepad_connected.emit()

        self.gamepad_log.emit(
            tr(
                "gp.connected_log",
                device=device,
                profile=det.profile.display_name,
                source=det.source,
            )
        )

    def _disconnect_gamepad(self):
        """断开手柄：停止控臂、停止轮询、清空显示。"""
        if self._ctrl_running:
            self._stop_control()

        self._poll_timer.stop()
        if self._joystick:
            self._joystick.disconnect()
            self._joystick = None
        self._detection = None

        self._connect_btn.setText(tr("gp.connect"))
        self._connect_btn.setObjectName("connectBtn")
        self._connect_btn.setStyle(self._connect_btn.style())
        self._info_label.setText(tr("gp.not_connected"))
        self._update_ctrl_btn_state()

        # 清空监控区显示
        for bar in self._axis_bars:
            bar.set_value(0.0)
        for lbl in self._axis_labels:
            lbl.setText(" 0.000")
        for ind in self._btn_indicators:
            ind.set_pressed(False)
        self._mapped_display.setText("—")

        self.gamepad_log.emit(tr("gp.gamepad_disconnected"))
        self.gamepad_disconnected.emit()

    def is_connected(self) -> bool:
        """手柄当前是否处于连接状态。"""
        return self._joystick is not None and self._joystick.connected

    def disconnect_gamepad(self):
        """供外部调用：若手柄已连接则断开（与点击“断开手柄”等价）。"""
        if self.is_connected():
            self._disconnect_gamepad()

    # ---- 输入轮询（20 Hz） ----

    def _poll_input(self):
        """定时轮询手柄输入并刷新监控区。

        同时根据手柄控制线程的状态更新速度档、控制模式、目标末端位姿。
        """
        joy = self._joystick
        if not joy or not joy.connected:
            self._disconnect_gamepad()
            return

        # 刷新所有轴的进度条和数值
        for i in range(LinuxJoystick.MAX_AXES):
            v = joy.axes[i]
            self._axis_bars[i].set_value(v)
            self._axis_labels[i].setText(f"{v:+.3f}")

        # 刷新所有按钮指示灯
        for i in range(LinuxJoystick.MAX_BUTTONS):
            self._btn_indicators[i].set_pressed(bool(joy.buttons[i]))

        # 如果已识别 profile，显示映射后的语义值
        if self._detection:
            prof = self._detection.profile
            s = prof.sticks
            parts = []
            parts.append(f"LX={s.lx.read(joy.axes):+.2f}")
            parts.append(f"LY={s.ly.read(joy.axes):+.2f}")
            parts.append(f"RX={s.rx.read(joy.axes):+.2f}")
            parts.append(f"RY={s.ry.read(joy.axes):+.2f}")
            parts.append(f"LT={s.lt.read(joy.axes, joy.buttons):.2f}")
            parts.append(f"RT={s.rt.read(joy.axes, joy.buttons):.2f}")
            parts.append(f"DX={s.dpad_x.read(joy.axes):+.1f}")
            parts.append(f"DY={s.dpad_y.read(joy.axes):+.1f}")
            self._mapped_display.setText("  ".join(parts))

        # 如果控臂线程正在运行，刷新控制区状态
        if self._ctrl_running and self._controller:
            ctrl = self._controller
            idx = ctrl._speed_idx
            specs = self._speed_level_specs()
            name, _ = specs[idx]
            self._speed_label.setText(f"{idx+1}/5 [{name}]")

            sc = self._scene()
            if ctrl._zero_torque:
                self._mode_label.setText(tr("gp.mode_zt"))
                self._mode_label.setStyleSheet(
                    f"color: {sc['mode_zt']}; font-weight: bold;")
            elif ctrl._estop:
                self._mode_label.setText(tr("gp.mode_estop"))
                self._mode_label.setStyleSheet(
                    f"color: {sc['mode_estop']}; font-weight: bold;")
            else:
                self._mode_label.setText(tr("gp.mode_normal"))
                self._mode_label.setStyleSheet(
                    f"color: {sc['mode_normal']}; font-weight: bold;")

            if ctrl._target_pose:
                p = ctrl._target_pose
                self._pose_label.setText(
                    f"XYZ: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m\n"
                    f"RPY: ({p.rx:.2f}, {p.ry:.2f}, {p.rz:.2f}) rad"
                )

            if ctrl.exit_requested:
                self._stop_control()
                self.gamepad_log.emit(tr("gp.ctrl_exit"))

    # ---- 机械臂控制 ----

    def set_arm(self, arm):
        """由主窗口传入机械臂实例，控制按钮可用性会随之更新。"""
        self._arm = arm
        self._update_ctrl_btn_state()
        # 机械臂就绪时，若手柄已连接且尚未控臂，则自动开启
        if (
            arm is not None
            and self._joystick is not None
            and self._joystick.connected
            and not self._ctrl_running
        ):
            self._start_control()

    def _update_ctrl_btn_state(self):
        """根据手柄连接状态和机械臂实例，更新“启动控制”按钮的可用性与文本。"""
        can_start = (
            self._joystick is not None
            and self._joystick.connected
            and self._arm is not None
            and not self._ctrl_running
        )
        self._ctrl_btn.setEnabled(can_start or self._ctrl_running)
        if self._ctrl_running:
            self._ctrl_btn.setText(tr("gp.stop_ctrl"))
            self._ctrl_btn.setObjectName("disconnectBtn")
        else:
            self._ctrl_btn.setText(tr("gp.start_ctrl"))
            self._ctrl_btn.setObjectName("enableBtn")
        self._ctrl_btn.setStyle(self._ctrl_btn.style())

    def set_gripper_torque_limits(self, torque_max: float, torque_min: float):
        """设置夹爪力矩限制 (Nm)，由夹爪面板输入框实时写入。

        若手柄控臂已启动，同步透传给运行中的 XboxArmController。
        """
        self._gripper_torque_max = float(torque_max)
        self._gripper_torque_min = float(torque_min)
        if self._controller is not None:
            self._controller.set_gripper_torque_limits(
                self._gripper_torque_max, self._gripper_torque_min
            )

    def _toggle_control(self):
        """控制按钮槽函数：启动或停止手柄控臂。"""
        if self._ctrl_running:
            self._stop_control()
        else:
            self._start_control()

    def _start_control(self):
        """启动手柄控臂后台线程。"""
        if not self._joystick or not self._joystick.connected:
            self.gamepad_log.emit(tr("gp.connect_first"))
            return
        if not self._arm:
            self.gamepad_log.emit(tr("gp.connect_arm_first"))
            return
        if not self._detection:
            return

        # 导入并实例化 Xbox 手柄控制逻辑
        from demo.xbox_control import XboxArmController

        self._controller = XboxArmController(
            arm=self._arm,
            joystick=self._joystick,
            profile=self._detection.profile,
            update_rate=100.0,  # 控制循环 100Hz
        )
        # 同步当前夹爪力矩限值（夹爪面板输入框 → 手柄夹爪力矩保护）
        self._controller.set_gripper_torque_limits(
            self._gripper_torque_max, self._gripper_torque_min
        )

        self._ctrl_running = True
        self._ctrl_thread = threading.Thread(
            target=self._run_control, daemon=True, name="gamepad_arm_ctrl"
        )
        self._ctrl_thread.start()
        self._update_ctrl_btn_state()
        self.gamepad_log.emit(tr("gp.ctrl_started"))

    def _run_control(self):
        """后台线程入口：运行 XboxArmController。"""
        try:
            self._controller.start()
        except Exception as e:
            err_s = str(e)
            logger.error(tr("gp.ctrl_error", e=err_s))
            self.gamepad_log.emit(tr("gp.ctrl_error", e=err_s))
        finally:
            self._ctrl_running = False

    def _stop_control(self):
        """停止手柄控臂后台线程并重置控制区显示。"""
        if self._controller:
            self._controller.stop()
        if self._ctrl_thread:
            self._ctrl_thread.join(timeout=2.0)
            self._ctrl_thread = None
        self._controller = None
        self._ctrl_running = False
        self._update_ctrl_btn_state()

        self._mode_label.setText("—")
        self._mode_label.setStyleSheet("")
        self._pose_label.setText("—")
        specs = self._speed_level_specs()
        self._speed_label.setText(
            f"{self._speed_idx+1}/5 [{specs[self._speed_idx][0]}]")

        self.gamepad_log.emit(tr("gp.ctrl_stopped"))

    # ---- 资源清理 ----

    def cleanup(self):
        """程序退出时调用：停止控臂并断开局柄。"""
        if self._ctrl_running:
            self._stop_control()
        if self._joystick and self._joystick.connected:
            self._joystick.disconnect()
        self._poll_timer.stop()

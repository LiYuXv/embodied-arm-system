"""工具栏面板：双臂接口选择 (SocketCAN/SLCAN) + 连接/使能/急停/状态指示 + 主题/语言切换"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QMessageBox, QFrame,
)
from PyQt6.QtCore import pyqtSignal, Qt

from debugger.utils.style import SCENE_COLORS

from debugger.utils.can_utils import (
    get_can_state, get_can_bitrate,
    setup_can_interface, shutdown_can_interface,
    detect_serial_ports,
)
from debugger.utils.i18n import tr
from debugger.utils.theme_manager import ThemeManager

BITRATE_OPTIONS = [
    ("1M", 1000000),
    ("500K", 500000),
    ("250K", 250000),
]

SERIAL_BAUDRATE_OPTIONS = [
    ("2M", 2000000),
    ("1M", 1000000),
    ("921.6K", 921600),
    ("576K", 576000),
    ("500K", 500000),
    ("115.2K", 115200),
]

CAN_BITRATE_OPTIONS = [
    ("1M", 1000000),
    ("500K", 500000),
    ("250K", 250000),
    ("125K", 125000),
]


class InterfaceRow(QWidget):
    """单臂接口选择行：后端 + CAN/串口选择 + 开启/关闭 + 波特率 + 刷新"""

    def __init__(self, parent=None, default_can: str = "can0"):
        super().__init__(parent)
        self._default_can = default_can
        self._init_ui()
        self._on_backend_changed()

    def _init_ui(self):
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        # -- 后端选择 --
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("SocketCAN", "socketcan")
        self.backend_combo.addItem("SLCAN", "slcan")
        self.backend_combo.setFixedWidth(90)
        self.backend_combo.setCurrentIndex(0)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        row.addWidget(self.backend_combo)

        # -- SocketCAN 控件组 --
        self.socketcan_label = QLabel(tr("tb.can_label"))
        row.addWidget(self.socketcan_label)

        self.can_label_value = QLabel(self._default_can)
        self.can_label_value.setFixedWidth(120)
        row.addWidget(self.can_label_value)

        self.can_toggle_btn = QPushButton(tr("tb.open"))
        self.can_toggle_btn.setFixedWidth(46)
        self.can_toggle_btn.clicked.connect(self._on_can_toggle)
        row.addWidget(self.can_toggle_btn)

        self.bitrate_combo = QComboBox()
        self.bitrate_combo.setFixedWidth(60)
        for label, val in BITRATE_OPTIONS:
            self.bitrate_combo.addItem(label, val)
        self.bitrate_combo.setCurrentIndex(0)
        row.addWidget(self.bitrate_combo)

        # -- SLCAN 控件组 --
        self.slcan_label = QLabel(tr("tb.serial_label"))
        row.addWidget(self.slcan_label)

        self.serial_combo = QComboBox()
        self.serial_combo.setFixedWidth(140)
        self.serial_combo.setEditable(True)
        self.serial_combo.lineEdit().setPlaceholderText("COM3")
        row.addWidget(self.serial_combo)

        self.serial_baud_combo = QComboBox()
        self.serial_baud_combo.setFixedWidth(70)
        self.serial_baud_combo.setToolTip(tr("tb.serial_baud_tip"))
        for label, val in SERIAL_BAUDRATE_OPTIONS:
            self.serial_baud_combo.addItem(label, val)
        self.serial_baud_combo.setCurrentIndex(0)
        row.addWidget(self.serial_baud_combo)

        self.can_baud_label = QLabel(tr("tb.can_label"))
        row.addWidget(self.can_baud_label)

        self.can_baud_combo = QComboBox()
        self.can_baud_combo.setFixedWidth(60)
        self.can_baud_combo.setToolTip(tr("tb.can_baud_tip"))
        for label, val in CAN_BITRATE_OPTIONS:
            self.can_baud_combo.addItem(label, val)
        self.can_baud_combo.setCurrentIndex(0)
        row.addWidget(self.can_baud_combo)

        # -- 刷新 --
        self.refresh_btn = QPushButton("⟳")
        self.refresh_btn.setFixedSize(28, 28)
        self.refresh_btn.setToolTip(tr("tb.refresh_tip"))
        self.refresh_btn.clicked.connect(self._on_refresh_clicked)
        row.addWidget(self.refresh_btn)

    # ---- 后端切换 ----

    def is_slcan_mode(self) -> bool:
        return self.backend_combo.currentData() == "slcan"

    def _on_backend_changed(self):
        slcan = self.is_slcan_mode()

        self.socketcan_label.setVisible(not slcan)
        self.can_label_value.setVisible(not slcan)
        self.can_toggle_btn.setVisible(not slcan)
        self.bitrate_combo.setVisible(not slcan)

        self.slcan_label.setVisible(slcan)
        self.serial_combo.setVisible(slcan)
        self.serial_baud_combo.setVisible(slcan)
        self.can_baud_label.setVisible(slcan)
        self.can_baud_combo.setVisible(slcan)

        self._on_refresh_clicked()

    # ---- 刷新接口 ----

    def _on_refresh_clicked(self):
        if self.is_slcan_mode():
            self._refresh_serial_ports()
        else:
            self._refresh_can_interfaces()

    def _refresh_can_interfaces(self):
        name = self._default_can
        state = get_can_state(name)
        bitrate = get_can_bitrate(name)
        br_str = f" {bitrate // 1000}K" if bitrate else ""
        self.can_label_value.setText(f"{name} ({state}{br_str})")
        if state == "UP" and bitrate:
            self._sync_bitrate_combo(bitrate)
        self._update_can_toggle_label()

    def _refresh_serial_ports(self):
        prev_text = self.serial_combo.currentText().split(" ")[0].strip()
        self.serial_combo.clear()

        ports = detect_serial_ports()
        restore_idx = -1

        for i, info in enumerate(ports):
            port = info["port"]
            desc = info["desc"]
            label = f"{port} ({desc})" if desc and desc != port else port
            self.serial_combo.addItem(label, port)
            if port == prev_text:
                restore_idx = i

        if not ports:
            self.serial_combo.addItem(f"COM3 {tr('tb.no_can')}")
            self.serial_combo.setItemData(0, "COM3")

        if restore_idx >= 0:
            self.serial_combo.setCurrentIndex(restore_idx)

    def _sync_bitrate_combo(self, bitrate: int):
        for i in range(self.bitrate_combo.count()):
            if self.bitrate_combo.itemData(i) == bitrate:
                self.bitrate_combo.setCurrentIndex(i)
                return

    def set_bitrate(self, bitrate: int):
        """外部设置 CAN 波特率（SocketCAN 与 SLCAN 的 CAN 波特率下拉框同步）"""
        self._sync_bitrate_combo(bitrate)
        for i in range(self.can_baud_combo.count()):
            if self.can_baud_combo.itemData(i) == bitrate:
                self.can_baud_combo.setCurrentIndex(i)
                return

    def _get_selected_can_name(self) -> str:
        return self._default_can

    def _get_selected_serial_port(self) -> str:
        idx = self.serial_combo.currentIndex()
        if idx >= 0 and self.serial_combo.itemData(idx):
            return self.serial_combo.itemData(idx)
        raw = self.serial_combo.currentText().split(" ")[0].strip()
        return raw or "COM3"

    def _get_selected_state(self) -> str:
        return get_can_state(self._default_can)

    def _update_can_toggle_label(self):
        state = self._get_selected_state()
        if state == "UP":
            self.can_toggle_btn.setText(tr("tb.close"))
        else:
            self.can_toggle_btn.setText(tr("tb.open"))

    def _on_can_toggle(self):
        name = self._get_selected_can_name()
        state = self._get_selected_state()

        if state == "UP":
            ok, msg = shutdown_can_interface(name)
        else:
            bitrate = self.bitrate_combo.currentData() or 1000000
            ok, msg = setup_can_interface(name, bitrate)

        if ok:
            self._refresh_can_interfaces()
        else:
            QMessageBox.warning(self, tr("tb.can_fail"), msg)
            self._refresh_can_interfaces()

    # ---- 对外接口 ----

    def get_config(self) -> dict:
        """返回传给 worker 的连接参数 dict"""
        if self.is_slcan_mode():
            return {
                "backend": "slcan",
                "serial_port": self._get_selected_serial_port(),
                "serial_baudrate": self.serial_baud_combo.currentData() or 2000000,
                "can_bitrate": self.can_baud_combo.currentData() or 1000000,
            }
        return {
            "backend": "socketcan",
            "can_name": self._get_selected_can_name(),
        }

    def set_widgets_enabled(self, enabled: bool):
        self.backend_combo.setEnabled(enabled)
        self.refresh_btn.setEnabled(enabled)
        self.can_toggle_btn.setEnabled(enabled)
        self.bitrate_combo.setEnabled(enabled)
        self.serial_combo.setEnabled(enabled)
        self.serial_baud_combo.setEnabled(enabled)
        self.can_baud_combo.setEnabled(enabled)

    def retranslate_ui(self):
        self.socketcan_label.setText(tr("tb.can_label"))
        self.slcan_label.setText(tr("tb.serial_label"))
        self.can_baud_label.setText(tr("tb.can_label"))
        self.serial_baud_combo.setToolTip(tr("tb.serial_baud_tip"))
        self.can_baud_combo.setToolTip(tr("tb.can_baud_tip"))
        self.refresh_btn.setToolTip(tr("tb.refresh_tip"))
        self._update_can_toggle_label()


class ToolbarPanel(QWidget):
    """顶部工具栏（双臂两行接口选择 + 控制按钮）"""

    connect_requested = pyqtSignal(str, dict)
    disconnect_requested = pyqtSignal()
    enable_requested = pyqtSignal()
    disable_requested = pyqtSignal()
    emergency_stop_requested = pyqtSignal()
    safe_exit_requested = pyqtSignal()
    open_monitor_requested = pyqtSignal()
    open_settings_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._connected = False
        self._enabled = False
        self.setFixedHeight(80)
        self._separators: list[QFrame] = []
        self.setStyleSheet(
            "ToolbarPanel QPushButton { padding: 5px 10px; min-height: 20px; }"
            "ToolbarPanel QComboBox { min-height: 20px; padding: 2px 6px; }"
            "ToolbarPanel QCheckBox { min-height: 20px; }"
        )
        self._init_ui()

        tm = ThemeManager.instance()
        tm.theme_changed.connect(self._on_theme_changed)
        tm.language_changed.connect(lambda _: self.retranslate_ui())

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 2, 8, 2)
        root.setSpacing(2)

        # ---- 第一行：臂1 接口 + 控制按钮 ----
        row = QHBoxLayout()
        row.setSpacing(6)

        self.arm1_label = QLabel(tr("tb.arm1"))
        self.arm1_label.setStyleSheet("font-weight: bold;")
        row.addWidget(self.arm1_label)

        self.arm1_row = InterfaceRow(default_can="can0")
        row.addWidget(self.arm1_row)

        self._add_sep(row)

        self.connect_btn = QPushButton(tr("tb.connect"))
        self.connect_btn.setObjectName("connectBtn")
        self.connect_btn.setFixedWidth(60)
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        row.addWidget(self.connect_btn)

        self.enable_btn = QPushButton(tr("tb.enable"))
        self.enable_btn.setObjectName("enableBtn")
        self.enable_btn.setFixedWidth(60)
        self.enable_btn.setEnabled(False)
        self.enable_btn.clicked.connect(self._on_enable_clicked)
        row.addWidget(self.enable_btn)

        self._add_sep(row)

        self.estop_btn = QPushButton(tr("tb.estop"))
        self.estop_btn.setObjectName("emergencyStop")
        self.estop_btn.setFixedWidth(60)
        self.estop_btn.clicked.connect(self.emergency_stop_requested.emit)
        row.addWidget(self.estop_btn)

        self.safe_exit_btn = QPushButton(tr("tb.safe_exit"))
        self.safe_exit_btn.setObjectName("safeExitBtn")
        self.safe_exit_btn.setFixedWidth(76)
        self.safe_exit_btn.setToolTip(tr("tb.safe_exit_tip"))
        self.safe_exit_btn.clicked.connect(self.safe_exit_requested.emit)
        row.addWidget(self.safe_exit_btn)

        self._add_sep(row)

        self.monitor_btn = QPushButton(tr("tb.monitor"))
        self.monitor_btn.setFixedWidth(76)
        self.monitor_btn.setToolTip(tr("tb.monitor_tip"))
        self.monitor_btn.clicked.connect(self.open_monitor_requested.emit)
        row.addWidget(self.monitor_btn)

        self.settings_btn = QPushButton("设置")
        self.settings_btn.setFixedWidth(60)
        self.settings_btn.setToolTip("系统设置（启动自动连接 / CAN 波特率 / 摄像头）")
        self.settings_btn.clicked.connect(self.open_settings_requested.emit)
        row.addWidget(self.settings_btn)

        row.addStretch()

        # -- 主题 / 语言切换 --
        self.theme_btn = QPushButton("🌙")
        self.theme_btn.setFixedSize(36, 30)
        self.theme_btn.setToolTip("Dark / Light")
        self.theme_btn.clicked.connect(ThemeManager.instance().toggle_theme)
        row.addWidget(self.theme_btn)

        self.lang_btn = QPushButton("EN")
        self.lang_btn.setFixedSize(36, 30)
        self.lang_btn.setToolTip("中文 / English")
        self.lang_btn.clicked.connect(ThemeManager.instance().toggle_language)
        row.addWidget(self.lang_btn)

        self._add_sep(row)

        self.status_label = QLabel(tr("tb.not_connected"))
        self.status_label.setObjectName("statusLabel")
        row.addWidget(self.status_label)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("fpsLabel")
        self.fps_label.setFixedWidth(70)
        row.addWidget(self.fps_label)

        root.addLayout(row)

        # ---- 第二行：臂2 接口 ----
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        self.arm2_label = QLabel(tr("tb.arm2"))
        self.arm2_label.setStyleSheet("font-weight: bold;")
        row2.addWidget(self.arm2_label)

        self.arm2_row = InterfaceRow(default_can="can1")
        row2.addWidget(self.arm2_row)
        row2.addStretch()

        root.addLayout(row2)

    def _add_sep(self, layout):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedWidth(1)
        sep.setFixedHeight(24)
        sc = SCENE_COLORS[ThemeManager.instance().theme]
        sep.setStyleSheet(f"color: {sc['separator']}; background: {sc['separator']};")
        self._separators.append(sep)
        layout.addWidget(sep)

    # ---- 主题 / 语言 ----

    def _on_theme_changed(self, theme: str):
        icon = "☀️" if theme == "dark" else "🌙"
        self.theme_btn.setText(icon)
        sc = SCENE_COLORS[theme]
        for sep in self._separators:
            sep.setStyleSheet(f"color: {sc['separator']}; background: {sc['separator']};")

    def retranslate_ui(self):
        tm = ThemeManager.instance()
        self.lang_btn.setText("中" if tm.language == "en" else "EN")

        self.arm1_label.setText(tr("tb.arm1"))
        self.arm2_label.setText(tr("tb.arm2"))
        self.arm1_row.retranslate_ui()
        self.arm2_row.retranslate_ui()

        self.estop_btn.setText(tr("tb.estop"))
        self.safe_exit_btn.setText(tr("tb.safe_exit"))
        self.safe_exit_btn.setToolTip(tr("tb.safe_exit_tip"))
        self.monitor_btn.setText(tr("tb.monitor"))
        self.monitor_btn.setToolTip(tr("tb.monitor_tip"))

        if self._connected:
            self.connect_btn.setText(tr("tb.disconnect"))
        else:
            self.connect_btn.setText(tr("tb.connect"))

        if self._enabled:
            self.enable_btn.setText(tr("tb.disable"))
            self.status_label.setText(tr("tb.enabled"))
        elif self._connected:
            self.enable_btn.setText(tr("tb.enable"))
            self.status_label.setText(tr("tb.connected"))
        else:
            self.enable_btn.setText(tr("tb.enable"))
            self.status_label.setText(tr("tb.not_connected"))

    # ---- 连接 / 使能 ----

    def _on_connect_clicked(self):
        if not self._connected:
            cfg1 = self.arm1_row.get_config()
            cfg2 = self.arm2_row.get_config()
            kwargs = dict(cfg1)
            kwargs.pop("can_name", None)
            kwargs["arm2"] = cfg2
            can_name = cfg1.get("can_name") or cfg1.get("serial_port", "can0")
            self.connect_requested.emit(can_name, kwargs)
        else:
            self.disconnect_requested.emit()

    def _on_enable_clicked(self):
        if not self._enabled:
            self.enable_requested.emit()
        else:
            self.disable_requested.emit()

    # ---- 外部状态更新 ----

    def set_connected(self, connected: bool):
        self._connected = connected
        can_area_enabled = not connected
        if connected:
            self.connect_btn.setText(tr("tb.disconnect"))
            self.connect_btn.setObjectName("disconnectBtn")
            self.enable_btn.setEnabled(True)
            self.status_label.setText(tr("tb.connected"))
        else:
            self.connect_btn.setText(tr("tb.connect"))
            self.connect_btn.setObjectName("connectBtn")
            self.enable_btn.setEnabled(False)
            self._enabled = False
            self.status_label.setText(tr("tb.not_connected"))

        self.arm1_row.set_widgets_enabled(can_area_enabled)
        self.arm2_row.set_widgets_enabled(can_area_enabled)

        self.connect_btn.style().unpolish(self.connect_btn)
        self.connect_btn.style().polish(self.connect_btn)

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if enabled:
            self.enable_btn.setText(tr("tb.disable"))
            self.status_label.setText(tr("tb.enabled"))
            self.status_label.setObjectName("statusLabel")
        else:
            self.enable_btn.setText(tr("tb.enable"))
            if self._connected:
                self.status_label.setText(tr("tb.connected"))
            self.status_label.setObjectName("statusLabel")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def set_can_bitrate(self, bitrate: int):
        """设置双臂 CAN 波特率下拉框（由系统设置应用）"""
        self.arm1_row.set_bitrate(bitrate)
        self.arm2_row.set_bitrate(bitrate)

    def set_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.0f}")

    def set_error(self, msg: str):
        self.status_label.setText(tr("tb.error", msg=msg))
        self.status_label.setObjectName("errorLabel")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

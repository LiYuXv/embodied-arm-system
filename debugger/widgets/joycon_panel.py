"""Single-Joy-Con teleoperation panel for the legacy single-arm GUI."""

import threading
from typing import Optional

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from el_a3_sdk.joycon import (
    JoyConInput,
    JoyConPoseAdapter,
    discover_joycon,
)


class JoyConPanel(QWidget):
    """Connect one Joy-Con (L or R) and control the current single arm."""

    joycon_log = pyqtSignal(str)
    _control_finished = pyqtSignal()

    def __init__(self, parent=None, *, discovery_func=discover_joycon,
                 source_factory=JoyConInput, adapter_factory=JoyConPoseAdapter,
                 controller_factory=None):
        super().__init__(parent)
        self._discovery_func = discovery_func
        self._source_factory = source_factory
        self._adapter_factory = adapter_factory
        self._controller_factory = controller_factory
        self._arm = None
        self._arm_enabled = False
        self._source = None
        self._adapter = None
        self._controller = None
        self._controller_thread: Optional[threading.Thread] = None
        self._control_running = False

        self._init_ui()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._refresh_state)
        self._control_finished.connect(self._on_control_finished)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        connection = QGroupBox("Joy-Con 连接")
        row = QHBoxLayout(connection)
        row.addWidget(QLabel("手柄:"))
        self.side_combo = QComboBox()
        self.side_combo.addItem("Joy-Con (L)", "left")
        self.side_combo.addItem("Joy-Con (R)", "right")
        row.addWidget(self.side_combo)
        self.connect_btn = QPushButton("连接 Joy-Con")
        self.connect_btn.setObjectName("connectBtn")
        self.connect_btn.clicked.connect(self._toggle_connection)
        row.addWidget(self.connect_btn)
        self.recenter_btn = QPushButton("姿态归中")
        self.recenter_btn.setEnabled(False)
        self.recenter_btn.clicked.connect(self._recenter)
        row.addWidget(self.recenter_btn)
        row.addStretch()
        layout.addWidget(connection)

        control = QGroupBox("单臂遥操")
        control_layout = QVBoxLayout(control)
        control_row = QHBoxLayout()
        self.control_btn = QPushButton("开始遥操")
        self.control_btn.setObjectName("enableBtn")
        self.control_btn.setEnabled(False)
        self.control_btn.clicked.connect(self._toggle_control)
        control_row.addWidget(self.control_btn)
        control_row.addWidget(QLabel("运动缩放:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 2.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setValue(1.0)
        self.scale_spin.valueChanged.connect(self._set_scale)
        control_row.addWidget(self.scale_spin)
        control_row.addStretch()
        control_layout.addLayout(control_row)

        self.status_label = QLabel("未连接 Joy-Con")
        self.state_label = QLabel("姿态: --  |  离合: --  |  摇杆: --")
        self.state_label.setStyleSheet("font-family: monospace;")
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.state_label)
        layout.addWidget(control)

        help_label = QLabel(
            "ZL/ZR：按住启用 IMU 姿态旋转  |  摇杆：末端 X/Y（无需离合）  |  "
            "方向键或面键：Z/夹爪  |  "
            "摇杆按下：回零  |  L/R：回工作位"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        layout.addStretch()

    @property
    def is_connected(self):
        return self._adapter is not None and self._adapter.is_connected()

    @property
    def is_controlling(self):
        return self._control_running

    def set_arm(self, arm):
        if arm is None and self._control_running:
            self._stop_control()
        self._arm = arm
        self._update_buttons()

    def set_arm_enabled(self, enabled: bool):
        self._arm_enabled = bool(enabled)
        if not enabled and self._control_running:
            self._stop_control()
        self._update_buttons()

    def _toggle_connection(self):
        if self.is_connected:
            self._disconnect_input()
        else:
            self._connect_input()

    def _connect_input(self):
        side = self.side_combo.currentData()
        try:
            binding = self._discovery_func(side)
            self._source = self._source_factory(binding)
            self._adapter = self._adapter_factory(self._source)
            self._adapter.start()
        except Exception as exc:
            self._source = None
            self._adapter = None
            self.status_label.setText(f"连接失败: {exc}")
            self.joycon_log.emit(f"Joy-Con 连接失败: {exc}")
            self._update_buttons()
            return

        self.side_combo.setEnabled(False)
        self.connect_btn.setText("断开 Joy-Con")
        self.connect_btn.setObjectName("disconnectBtn")
        self.recenter_btn.setEnabled(True)
        self.status_label.setText(f"Joy-Con ({'L' if side == 'left' else 'R'}) 已连接")
        self._poll_timer.start()
        self._update_buttons()
        self.joycon_log.emit(f"Joy-Con ({side}) 已连接到单臂面板")

    def _disconnect_input(self):
        self._stop_control()
        self._poll_timer.stop()
        if self._adapter is not None:
            self._adapter.stop()
        self._source = None
        self._adapter = None
        self.side_combo.setEnabled(True)
        self.connect_btn.setText("连接 Joy-Con")
        self.connect_btn.setObjectName("connectBtn")
        self.recenter_btn.setEnabled(False)
        self.status_label.setText("未连接 Joy-Con")
        self.state_label.setText("姿态: --  |  离合: --  |  摇杆: --")
        self._update_buttons()
        self.joycon_log.emit("Joy-Con 已断开")

    def _recenter(self):
        if self._source is not None:
            self._source.recenter_orientation()
            self.joycon_log.emit("Joy-Con 姿态已归中")

    def _toggle_control(self):
        if self._control_running:
            self._stop_control()
        else:
            self._start_control()

    def _start_control(self):
        if not self.is_connected:
            self.joycon_log.emit("请先连接 Joy-Con")
            return
        if self._arm is None or not self._arm_enabled:
            self.joycon_log.emit("请先连接并使能机械臂")
            return
        try:
            if self._controller_factory is None:
                from demo.vr_control import VRArmController
                controller_factory = VRArmController
            else:
                controller_factory = self._controller_factory
            self._arm.start_control_loop(rate_hz=200.0)
            self._controller = controller_factory(
                arm=self._arm,
                receiver=self._adapter,
                state_getter=self._adapter.get_state,
                update_rate=100.0,
                scale=self.scale_spin.value(),
                position_fix=np.ones(3),
                rotation_fix=np.eye(3),
                remap_rotation_rpy=False,
                input_name=f"Joy-Con ({'L' if self.side_combo.currentData() == 'left' else 'R'})",
            )
        except Exception as exc:
            self._controller = None
            self.joycon_log.emit(f"Joy-Con 遥操启动失败: {exc}")
            return

        self._control_running = True
        self._controller_thread = threading.Thread(
            target=self._run_controller,
            daemon=True,
            name="single-arm-joycon-controller",
        )
        self._controller_thread.start()
        self.status_label.setText("Joy-Con 遥操中")
        self._update_buttons()
        self.joycon_log.emit("Joy-Con 单臂遥操已启动")

    def _run_controller(self):
        try:
            self._controller.start()
        except Exception as exc:
            self.joycon_log.emit(f"Joy-Con 遥操错误: {exc}")
        finally:
            self._control_finished.emit()

    def _stop_control(self):
        controller = self._controller
        if controller is not None:
            controller.stop()
        thread = self._controller_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._controller_thread = None
        self._controller = None
        was_running = self._control_running
        self._control_running = False
        self._update_buttons()
        if was_running:
            self.status_label.setText("Joy-Con 已连接，遥操已停止")
            self.joycon_log.emit("Joy-Con 单臂遥操已停止")

    def _on_control_finished(self):
        if self._control_running:
            self._stop_control()

    def _set_scale(self, value):
        if self._controller is not None:
            self._controller.scale = float(value)

    def _refresh_state(self):
        if self._source is None:
            return
        state = self._source.snapshot()
        self.state_label.setText(
            f"姿态: {'有效' if state.valid else '等待 IMU'}  |  "
            f"离合: {'接合' if state.clutch else '释放'}  |  "
            f"摇杆: ({state.stick_x:+.2f}, {state.stick_y:+.2f})"
        )
        if self._adapter is not None and self._adapter.exit_requested:
            self._stop_control()

    def _update_buttons(self):
        can_start = (
            self.is_connected and self._arm is not None and self._arm_enabled
        )
        self.control_btn.setEnabled(can_start or self._control_running)
        self.control_btn.setText("停止遥操" if self._control_running else "开始遥操")
        self.control_btn.setObjectName(
            "disconnectBtn" if self._control_running else "enableBtn"
        )

    def cleanup(self):
        if self._adapter is not None:
            self._disconnect_input()
        else:
            self._stop_control()
        self._poll_timer.stop()

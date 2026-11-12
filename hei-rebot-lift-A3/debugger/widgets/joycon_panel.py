"""Fixed-side Joy-Con teleoperation panel. No device is opened on import."""

import threading

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout,
                             QLabel, QPushButton, QVBoxLayout, QWidget)

from el_a3_sdk.joycon import (JOYCON_MODE_IMU_MOTION, JOYCON_MODE_STANDARD,
                             JOYCON_ROTATION_FIX, JOYCON_ROTATION_VECTOR_SIGN,
                             JoyConInput, JoyConPoseAdapter, discover_joycon)


class JoyConPanel(QWidget):
    joycon_log = pyqtSignal(str)
    _finished = pyqtSignal()

    def __init__(self, side, parent=None, *, discovery_func=discover_joycon,
                 source_factory=JoyConInput, adapter_factory=JoyConPoseAdapter,
                 controller_factory=None, ready_check=None, claim_check=None):
        if side not in ("left", "right"):
            raise ValueError("Joy-Con side must be left or right")
        super().__init__(parent)
        self.side = side
        self._discovery = discovery_func
        self._source_factory = source_factory
        self._adapter_factory = adapter_factory
        self._controller_factory = controller_factory
        self._ready_check = ready_check or (lambda _side: False)
        self._claim_check = claim_check or (lambda _side: True)
        self._arm = None
        self._arm_connected = False
        self._arm_enabled = False
        self._source = self._adapter = self._controller = self._thread = None
        self._running = False
        label = "L → 左臂 can0" if side == "left" else "R → 右臂 can1"
        layout = QVBoxLayout(self)
        group = QGroupBox(f"Joy-Con {label}")
        layout.addWidget(group)
        body = QVBoxLayout(group)
        row = QHBoxLayout()
        self.device_label = QLabel("设备: 未连接")
        row.addWidget(self.device_label)
        self.connect_btn = QPushButton("连接手柄")
        self.connect_btn.clicked.connect(self._toggle_connection)
        row.addWidget(self.connect_btn)
        self.recenter_btn = QPushButton("姿态归中")
        self.recenter_btn.clicked.connect(self._recenter)
        row.addWidget(self.recenter_btn)
        body.addLayout(row)
        self.arm_label = QLabel("机械臂: 未连接")
        body.addWidget(self.arm_label)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("标准模式", JOYCON_MODE_STANDARD)
        self.mode_combo.addItem("IMU 动作模式（实验）", JOYCON_MODE_IMU_MOTION)
        self.mode_combo.currentIndexChanged.connect(self._set_mode)
        mode_row.addWidget(self.mode_combo)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 2.0)
        self.scale_spin.setValue(1.0)
        mode_row.addWidget(QLabel("位置缩放"))
        mode_row.addWidget(self.scale_spin)
        self.rotation_scale_spin = QDoubleSpinBox()
        self.rotation_scale_spin.setRange(0.0, 2.0)
        self.rotation_scale_spin.setValue(1.0)
        mode_row.addWidget(QLabel("姿态缩放"))
        mode_row.addWidget(self.rotation_scale_spin)
        body.addLayout(mode_row)
        self.control_btn = QPushButton("开始遥操")
        self.control_btn.clicked.connect(self._toggle_control)
        body.addWidget(self.control_btn)
        self.status_label = QLabel("遥操已停止")
        self.state_label = QLabel("姿态: -- | 离合: -- | 摇杆: --")
        body.addWidget(self.status_label)
        body.addWidget(self.state_label)
        help_label = QLabel("标准：摇杆/按键 XYZ，ZL/ZR 姿态离合；IMU 实验：按住离合移动。方向键/面键夹爪；摇杆按下归零；L/R 回工作位。")
        help_label.setWordWrap(True)
        body.addWidget(help_label)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._refresh)
        self._finished.connect(self._on_finished)
        self._update_buttons()

    @property
    def is_connected(self):
        return self._adapter is not None and self._adapter.is_connected()

    @property
    def is_controlling(self):
        return self._running or (self._thread is not None and self._thread.is_alive())

    def set_arm_state(self, arm, connected, enabled):
        ready = bool(arm is not None and connected and enabled)
        if not ready and self.is_controlling:
            self.stop_control()
        self._arm = arm if connected else None
        self._arm_connected = bool(connected)
        self._arm_enabled = bool(enabled)
        self.arm_label.setText("机械臂: " + ("已使能" if ready else "已连接、未使能" if connected else "未连接"))
        self._update_buttons()

    def _toggle_connection(self):
        if self._adapter is None:
            self.connect_input()
        else:
            self.disconnect_input()

    def connect_input(self):
        if self._adapter is not None:
            return True
        binding = source = adapter = None
        try:
            binding = self._discovery(self.side)
            if binding.side != self.side:
                raise ValueError("手柄侧别与固定面板不符")
            source = self._source_factory(binding)
            adapter = self._adapter_factory(source)
            adapter.set_mode(self.mode_combo.currentData())
            self._source, self._adapter = source, adapter
            adapter.start()
        except Exception as exc:
            cleaned = True
            if adapter is not None:
                try:
                    adapter.stop()
                except Exception as stop_exc:
                    cleaned = False
                    self.joycon_log.emit(f"Joy-Con {self.side} 输入清理失败: {stop_exc}")
            elif source is not None:
                try:
                    source.stop()
                except Exception as stop_exc:
                    cleaned = False
                    self.joycon_log.emit(f"Joy-Con {self.side} 输入清理失败: {stop_exc}")
            elif binding is not None:
                for device in (getattr(binding, "control_device", None),
                               getattr(binding, "imu_device", None)):
                    try:
                        if device is not None:
                            device.close()
                    except (AttributeError, OSError):
                        pass
            if cleaned:
                self._source = self._adapter = None
            self.status_label.setText(f"连接失败: {exc}")
            self.joycon_log.emit(f"Joy-Con {self.side} 连接失败: {exc}")
            return False
        self.device_label.setText(f"设备: {binding.control_path} | IMU: {binding.imu_path}")
        self._timer.start()
        self._update_buttons()
        self.joycon_log.emit(f"Joy-Con {self.side} 已连接")
        return True

    def disconnect_input(self):
        if not self.stop_control():
            return False
        if self._adapter is not None:
            try:
                self._adapter.stop()
            except Exception as exc:
                self.joycon_log.emit(f"Joy-Con {self.side} 输入停止失败: {exc}")
                return False
        self._timer.stop()
        self._source = self._adapter = None
        self.device_label.setText("设备: 未连接")
        self.state_label.setText("姿态: -- | 离合: -- | 摇杆: --")
        self._update_buttons()
        self.joycon_log.emit(f"Joy-Con {self.side} 已断开")
        return True

    def _recenter(self):
        if self._source is not None:
            self._source.recenter_orientation()
            self.joycon_log.emit(f"Joy-Con {self.side} 姿态已归中")

    def _toggle_control(self):
        if self.is_controlling:
            self.stop_control()
        else:
            self.start_control()

    def start_control(self):
        if self.is_controlling:
            return False
        if not self.is_connected or self._arm is None or not self._arm_connected or not self._arm_enabled or not self._ready_check(self.side):
            self.joycon_log.emit(f"Joy-Con {self.side} 遥操拒绝：手柄或对应真机未连接使能")
            return False
        if not self._claim_check(self.side):
            self.joycon_log.emit(f"Joy-Con {self.side} 遥操拒绝：其他控制源占用该臂")
            return False
        try:
            factory = self._controller_factory
            if factory is None:
                from demo.joycon_arm_control import VRArmController
                factory = VRArmController
            if not self._arm.control_loop_running:
                self._arm.start_control_loop(rate_hz=200.0)
            if not self._arm.control_loop_running:
                raise RuntimeError("SDK 控制循环启动后仍未运行")
            self._controller = factory(
                arm=self._arm, receiver=self._adapter,
                state_getter=self._adapter.get_state, update_rate=100.0,
                scale=self.scale_spin.value(),
                rotation_scale=(self.rotation_scale_spin.value() if self.mode_combo.currentData() == JOYCON_MODE_IMU_MOTION else 1.0),
                position_fix=np.ones(3), rotation_fix=JOYCON_ROTATION_FIX,
                remap_rotation_rpy=False, rotation_vector_sign=JOYCON_ROTATION_VECTOR_SIGN,
                input_name=f"Joy-Con ({'L' if self.side == 'left' else 'R'})")
            self._running = True
            self._thread = threading.Thread(target=self._run_controller, daemon=True,
                                            name=f"joycon-{self.side}-controller")
            self._thread.start()
        except Exception as exc:
            self._running = False
            self._controller = self._thread = None
            self.joycon_log.emit(f"Joy-Con {self.side} 遥操启动失败: {exc}")
            return False
        self.status_label.setText("遥操中")
        self._update_buttons()
        self.joycon_log.emit(f"Joy-Con {self.side} 遥操已启动")
        return True

    def _run_controller(self):
        try:
            self._controller.start()
        except Exception as exc:
            self.joycon_log.emit(f"Joy-Con {self.side} 遥操错误: {exc}")
        finally:
            self._finished.emit()

    def stop_control(self):
        controller, thread = self._controller, self._thread
        if controller is not None:
            controller.stop()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():
                self.status_label.setText("停止未确认：控制线程仍在运行")
                self.joycon_log.emit(f"Joy-Con {self.side} 停止失败：控制线程超时")
                return False
        self._controller = self._thread = None
        self._running = False
        self.status_label.setText("遥操已停止")
        self._update_buttons()
        return True

    def _on_finished(self):
        if self._thread is not None and not self._thread.is_alive():
            self.stop_control()

    def _set_mode(self, _index):
        if self._adapter is not None and not self.is_controlling:
            self._adapter.set_mode(self.mode_combo.currentData())
        self._update_buttons()

    def _refresh(self):
        if self._source is None:
            return
        state = self._source.snapshot()
        self.state_label.setText(f"姿态: {'有效' if state.valid else '等待 IMU'} | 离合: {'接合' if state.clutch else '释放'} | 摇杆: ({state.stick_x:+.2f}, {state.stick_y:+.2f})")
        if self._adapter.exit_requested:
            self.stop_control()

    def _update_buttons(self):
        running = self.is_controlling
        self.connect_btn.setText("断开手柄" if self._adapter is not None else "连接手柄")
        self.recenter_btn.setEnabled(self._adapter is not None)
        self.control_btn.setEnabled(running or (self.is_connected and self._arm_connected and self._arm_enabled and self._ready_check(self.side)))
        self.control_btn.setText("停止遥操" if running else "开始遥操")
        self.mode_combo.setEnabled(not running)
        self.scale_spin.setEnabled(not running)
        self.rotation_scale_spin.setEnabled(not running and self.mode_combo.currentData() == JOYCON_MODE_IMU_MOTION)

    def cleanup(self):
        if not self.disconnect_input():
            raise RuntimeError(f"Joy-Con {self.side} 停止未确认")

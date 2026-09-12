"""Dual-arm debugger main window without the legacy 3D viewer."""

import os
import time

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from el_a3_sdk import initialize_dual_arms
from debugger.backend.arm_worker import ArmWorker
from debugger.utils.theme_manager import ThemeManager
from debugger.vla_features import DualArmActionDispatcher
from debugger.widgets.dual_arm_panel import ArmControlPanel
from debugger.widgets.dual_lerobot_panel import DualArmLeRobotPanel
from debugger.widgets.triple_camera_panel import TripleCameraPanel

ARM_CAN_MAP = {
    "left_arm": "can0",
    "right_arm": "can1",
}


class MainWindow(QMainWindow):
    """Operate two independent arms and three independent camera streams."""

    mode_switch_requested = pyqtSignal(str)

    def __init__(self, urdf_path=None, mesh_dir=None, *, arm_manager=None,
                 sim_mode=False, start_workers=True, dataset_root=None):
        # Keep obsolete arguments so existing launch commands remain valid.
        # The 3D viewer and its model are intentionally not created.
        del mesh_dir
        super().__init__()
        self.setWindowTitle("EL-A3 双臂调试上位机")
        self.setMinimumSize(1500, 860)
        self.resize(1900, 1040)
        self._urdf_path = urdf_path
        self._dataset_root = dataset_root or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "lerobot_data"
        )
        self._sim_mode = bool(sim_mode)
        self._closing = False
        self._connected = {name: False for name in ARM_CAN_MAP}
        self._enabled = {name: False for name in ARM_CAN_MAP}

        self.arm_manager = arm_manager or initialize_dual_arms(auto_connect=False)
        self._validate_arm_manager()
        self._init_workers(start_workers)
        self._init_ui()
        self.vla_action_dispatcher = DualArmActionDispatcher(
            self.left_worker,
            self.right_worker,
            execute=False,
            print_fn=self._append_log,
        )
        self._connect_signals()
        self.statusBar().showMessage(
            "就绪（双臂模拟模式）" if self._sim_mode else "就绪"
        )

    def _validate_arm_manager(self):
        for arm_name, expected_can in ARM_CAN_MAP.items():
            if not self.arm_manager.has_arm(arm_name):
                raise ValueError(f"双臂管理器缺少 {arm_name}")
            arm = self.arm_manager[arm_name]
            actual_can = getattr(arm, "can_name", None) or getattr(arm, "_can_name", None)
            if actual_can is not None and actual_can != expected_can:
                raise ValueError(
                    f"{arm_name} 必须绑定 {expected_can}，当前为 {actual_can}"
                )
        if self.arm_manager["left_arm"] is self.arm_manager["right_arm"]:
            raise ValueError("左右臂必须绑定不同的 SDK 实例")

    def _init_workers(self, start_workers):
        self.workers = {}
        for arm_name, can_name in ARM_CAN_MAP.items():
            worker = ArmWorker(
                arm=self.arm_manager[arm_name],
                arm_name=arm_name,
                can_name=can_name,
                parent=self,
            )
            self.workers[arm_name] = worker
            if start_workers:
                worker.start()

        self.left_worker = self.workers["left_arm"]
        self.right_worker = self.workers["right_arm"]
        # Compatibility alias for integrations that read the old attribute.
        self.worker = self.left_worker

    def _init_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addWidget(self._build_system_controls())
        self.lerobot_panel = DualArmLeRobotPanel(
            self._dataset_root,
            frames_callback=self._get_camera_frames,
            state_callback=self._get_dual_arm_state,
            torque_callback=self._get_dual_gripper_torque,
        )
        root.addWidget(self.lerobot_panel)

        vertical = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(vertical, 1)

        arm_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.arm_panels = {
            name: ArmControlPanel(
                name, can_name, urdf_path=self._urdf_path
            )
            for name, can_name in ARM_CAN_MAP.items()
        }
        self.left_arm_panel = self.arm_panels["left_arm"]
        self.right_arm_panel = self.arm_panels["right_arm"]
        for panel in self.arm_panels.values():
            arm_splitter.addWidget(panel)
        arm_splitter.setSizes([800, 800])
        vertical.addWidget(arm_splitter)

        self.camera_panel = TripleCameraPanel()
        self.camera_widget = self.camera_panel
        vertical.addWidget(self.camera_panel)

        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(6, 6, 6, 6)
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(130)
        log_layout.addWidget(self.log_console)
        vertical.addWidget(log_group)
        vertical.setStretchFactor(0, 7)
        vertical.setStretchFactor(1, 3)
        vertical.setStretchFactor(2, 1)
        vertical.setSizes([590, 260, 100])

        # Compatibility alias for code that only read the former joint panel.
        self.joint_panel = self.left_arm_panel.joint_panel

    def _build_system_controls(self):
        group = QGroupBox("系统控制")
        row = QHBoxLayout(group)
        row.setContentsMargins(8, 6, 8, 6)
        self.system_status_label = QLabel("左臂: 未连接  |  右臂: 未连接")
        row.addWidget(self.system_status_label)
        row.addStretch()

        self.connect_all_btn = QPushButton("连接双臂")
        self.connect_all_btn.clicked.connect(self._connect_all)
        row.addWidget(self.connect_all_btn)

        self.disconnect_all_btn = QPushButton("断开双臂")
        self.disconnect_all_btn.clicked.connect(self._disconnect_all)
        row.addWidget(self.disconnect_all_btn)

        self.enable_all_btn = QPushButton("使能双臂")
        self.enable_all_btn.setObjectName("enableBtn")
        self.enable_all_btn.clicked.connect(self._enable_all)
        row.addWidget(self.enable_all_btn)

        self.disable_all_btn = QPushButton("失能双臂")
        self.disable_all_btn.clicked.connect(self._disable_all)
        row.addWidget(self.disable_all_btn)

        self.stop_all_btn = QPushButton("双臂停止")
        self.stop_all_btn.clicked.connect(self._stop_all)
        row.addWidget(self.stop_all_btn)

        self.estop_all_btn = QPushButton("双臂急停")
        self.estop_all_btn.setObjectName("emergencyStop")
        self.estop_all_btn.clicked.connect(self._emergency_stop_all)
        row.addWidget(self.estop_all_btn)

        self.theme_btn = QPushButton("主题")
        self.theme_btn.clicked.connect(ThemeManager.instance().toggle_theme)
        row.addWidget(self.theme_btn)

        self.mode_switch_btn = QPushButton("切换单臂")
        self.mode_switch_btn.setToolTip("安全关闭当前双臂会话并切换到原单臂界面")
        self.mode_switch_btn.clicked.connect(
            lambda: self.mode_switch_requested.emit("single")
        )
        row.addWidget(self.mode_switch_btn)
        return group

    def _connect_signals(self):
        for arm_name, panel in self.arm_panels.items():
            worker = self.workers[arm_name]
            panel.bind_worker(worker, sim_mode=self._sim_mode)
            panel.connected_changed.connect(
                lambda connected, name=arm_name: self._on_connected(name, connected)
            )
            panel.enabled_changed.connect(
                lambda enabled, name=arm_name: self._on_enabled(name, enabled)
            )
            panel.error_occurred.connect(
                lambda message, name=arm_name: self._on_arm_error(name, message)
            )
            panel.enable_rejected.connect(
                lambda message, name=arm_name: self._on_enable_rejected(name, message)
            )
            panel.log_message.connect(
                lambda message, name=arm_name: self._append_log(f"[{name}] {message}")
            )

        self.camera_panel.log_message.connect(
            lambda message: self._append_log(f"[摄像头] {message}")
        )
        self.lerobot_panel.log_message.connect(
            lambda message: self._append_log(f"[LeRobot] {message}")
        )

    def _get_camera_frames(self):
        return self.camera_panel.latest_frames()

    def _get_dual_arm_state(self):
        left = self.left_arm_panel.get_joint_state()
        right = self.right_arm_panel.get_joint_state()
        if left is None or right is None:
            return None
        return np.asarray(left + right, dtype=np.float32)

    def _get_dual_gripper_torque(self):
        values = (
            self.left_arm_panel.get_gripper_torque(),
            self.right_arm_panel.get_gripper_torque(),
        )
        return np.asarray(
            [0.0 if value is None else value for value in values],
            dtype=np.float32,
        )

    def preview_vla_action_chunk(self, action_chunk, step=0):
        """Split/print one dual VLA step; hardware execution stays disabled."""
        return self.vla_action_dispatcher.dispatch(action_chunk, step=step)

    def _connect_arm(self, arm_name):
        self.arm_panels[arm_name].request_connect()

    def _connect_all(self):
        for arm_name in ARM_CAN_MAP:
            if not self._connected[arm_name]:
                self._connect_arm(arm_name)

    def _disconnect_all(self):
        for worker in self.workers.values():
            worker.submit_command("disconnect")

    def _enable_all(self):
        for arm_name, worker in self.workers.items():
            if self._connected[arm_name]:
                worker.submit_command("enable")

    def _disable_all(self):
        for worker in self.workers.values():
            worker.submit_command("disable")

    def _stop_all(self):
        for worker in self.workers.values():
            worker.submit_command("cancel_motion")
        self._append_log("已下发双臂停止运动命令")

    def _emergency_stop_all(self):
        for worker in self.workers.values():
            worker.submit_command("emergency_stop")
        self._append_log("已下发双臂急停命令")

    def _on_connected(self, arm_name, connected):
        self._connected[arm_name] = connected
        if not connected:
            self._enabled[arm_name] = False
        self._update_system_status()

    def _on_enabled(self, arm_name, enabled):
        self._enabled[arm_name] = enabled
        self._update_system_status()

    def _update_system_status(self):
        def state(name):
            if self._enabled[name]:
                return "已使能"
            if self._connected[name]:
                return "已连接"
            return "未连接"

        self.system_status_label.setText(
            f"左臂: {state('left_arm')}  |  右臂: {state('right_arm')}"
        )

    def _on_arm_error(self, arm_name, message):
        self._append_log(f"[{arm_name}] 错误: {message}")

    def _on_enable_rejected(self, arm_name, message):
        self._append_log(f"[{arm_name}] {message.splitlines()[0]}")
        QMessageBox.warning(self, f"{arm_name} 使能失败", message)

    def _append_log(self, message):
        self.log_console.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return
        self._closing = True
        self._append_log("正在停止双臂与摄像头…")
        self.lerobot_panel.cleanup()
        self.camera_panel.cleanup()
        for panel in self.arm_panels.values():
            panel.cleanup()

        connected_workers = [
            worker for name, worker in self.workers.items() if self._connected[name]
        ]
        for worker in connected_workers:
            worker.submit_command("emergency_stop")
        if connected_workers:
            time.sleep(0.25)
        for worker in connected_workers:
            worker.submit_command("disconnect")
        if connected_workers:
            time.sleep(0.20)
        for worker in self.workers.values():
            worker.stop()
        event.accept()

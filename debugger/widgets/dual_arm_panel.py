"""Full single-arm control workspace reused by each side of the dual GUI."""

import logging
import os
import time

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

try:
    from el_a3_sdk.kinematics import ELA3Kinematics
except Exception:
    ELA3Kinematics = None

from debugger.utils.i18n import tr
from debugger.utils.theme_manager import ThemeManager
from debugger.widgets.diagnostics_panel import DiagnosticsPanel
from debugger.widgets.gamepad_panel import GamepadPanel
from debugger.widgets.gripper_panel import GripperPanel
from debugger.widgets.joint_control_panel import JointControlPanel
from debugger.widgets.joycon_panel import JoyConPanel
from debugger.widgets.monitoring_window import MonitoringWindow
from debugger.widgets.real_time_kinematics_display import RealTimeKinematicsDisplay
from debugger.widgets.teaching_panel import TeachingPanel
from debugger.widgets.toolbar_panel import ToolbarPanel
from debugger.widgets.trajectory_panel import TrajectoryPanel
from debugger.widgets.vr_panel import VrPanel

logger = logging.getLogger("debugger")


class ArmControlPanel(QWidget):
    """One complete arm debugger, intentionally excluding only the 3D viewer."""

    connect_requested = pyqtSignal()
    disconnect_requested = pyqtSignal()
    enable_requested = pyqtSignal()
    disable_requested = pyqtSignal()
    refresh_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    emergency_stop_requested = pyqtSignal()
    connected_changed = pyqtSignal(bool)
    enabled_changed = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    enable_rejected = pyqtSignal(str)
    log_message = pyqtSignal(str)

    UI_UPDATE_INTERVAL_S = 0.05

    def __init__(self, arm_name: str, can_name: str, parent=None,
                 *, urdf_path=None):
        super().__init__(parent)
        self.arm_name = arm_name
        self.can_name = can_name
        self.worker = None
        self._sim_mode = False
        self._connected = False
        self._enabled = False
        self._last_joint_states = None
        self._last_effort_states = None
        self._last_end_pose = None
        self._last_gripper_angle_rad = None
        self._last_ui_update_time = 0.0
        self._auto_sync_pending = False
        self._auto_sync_tries = 0
        self._auto_sync_timer = None
        self._cleaned = False
        self._kin = self._load_kinematics(urdf_path)

        self.setObjectName(f"{arm_name}Panel")
        self._init_ui()

        self._theme_manager = ThemeManager.instance()
        self._theme_manager.language_changed.connect(self._retranslate_ui)
        self._theme_manager.theme_changed.connect(self._on_theme_changed)
        self._theme_connections_active = True

    @staticmethod
    def _load_kinematics(urdf_path):
        if ELA3Kinematics is None or not urdf_path or not os.path.isfile(urdf_path):
            return None
        try:
            return ELA3Kinematics(urdf_path=urdf_path)
        except Exception as exc:
            logger.warning("[ArmControlPanel] 加载运动学模型失败: %s", exc)
            return None

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(4)

        self.group = QGroupBox(f"{self.arm_name}  ·  {self.can_name}")
        outer.addWidget(self.group)
        layout = QVBoxLayout(self.group)
        layout.setContentsMargins(5, 6, 5, 5)
        layout.setSpacing(4)

        self.toolbar = ToolbarPanel(preferred_can_name=self.can_name)
        layout.addWidget(self.toolbar)

        quick_row = QHBoxLayout()
        self.mapping_label = QLabel(
            f"固定映射: {self.arm_name} → {self.can_name}"
        )
        quick_row.addWidget(self.mapping_label)
        quick_row.addStretch()
        self.refresh_feedback_btn = QPushButton("刷新状态")
        self.refresh_feedback_btn.clicked.connect(self.refresh_requested.emit)
        quick_row.addWidget(self.refresh_feedback_btn)
        self.stop_btn = QPushButton("停止运动")
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        quick_row.addWidget(self.stop_btn)
        layout.addLayout(quick_row)

        self.tabs = QTabWidget()
        self.joint_panel = JointControlPanel()
        self.joint_panel.set_joint_names(
            ["J1", "J2", "J3", "J4", "J5", "J6", "夹爪"]
        )
        self.trajectory_panel = TrajectoryPanel()
        self.trajectory_panel.drag_btn.hide()
        self.trajectory_panel.sync_btn.hide()
        self.teaching_panel = TeachingPanel()
        self.diagnostics_panel = DiagnosticsPanel()
        self.gripper_panel = GripperPanel()
        self.gamepad_panel = GamepadPanel()
        self.joycon_panel = JoyConPanel()
        self.vr_panel = VrPanel()
        self.kinematics_display = RealTimeKinematicsDisplay()

        panels = (
            (self.joint_panel, tr("tab.joint")),
            (self.trajectory_panel, tr("tab.trajectory")),
            (self.teaching_panel, tr("tab.teaching")),
            (self.diagnostics_panel, tr("tab.diagnostics")),
            (self.gripper_panel, tr("tab.gripper")),
            (self.gamepad_panel, tr("tab.gamepad")),
            (self.joycon_panel, "Joy-Con"),
            (self.vr_panel, tr("tab.vr")),
            (self.kinematics_display, "末端状态"),
        )
        for panel, title in panels:
            self.tabs.addTab(panel, title)
        layout.addWidget(self.tabs, 1)

        self.joint_panel.set_enabled(False)
        self.refresh_feedback_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        tb = self.toolbar
        tb.connect_requested.connect(self._on_toolbar_connect)
        tb.disconnect_requested.connect(self.disconnect_requested.emit)
        tb.enable_requested.connect(self.enable_requested.emit)
        tb.disable_requested.connect(self.disable_requested.emit)
        tb.emergency_stop_requested.connect(self.emergency_stop_requested.emit)
        tb.safe_exit_requested.connect(self._safe_exit)
        tb.open_monitor_requested.connect(self._open_monitoring)

    def bind_worker(self, worker, *, sim_mode=False):
        """Bind this complete workspace to exactly one ArmWorker."""
        if self.worker is not None:
            raise RuntimeError(f"{self.arm_name} 已绑定 Worker")
        self.worker = worker
        self._sim_mode = bool(sim_mode)
        self.monitoring_window = MonitoringWindow(worker.data_buffer, parent=self)

        self.connect_requested.connect(self._connect_arm)
        self.disconnect_requested.connect(
            lambda: worker.submit_command("disconnect")
        )
        self.enable_requested.connect(lambda: worker.submit_command("enable"))
        self.disable_requested.connect(lambda: worker.submit_command("disable"))
        self.refresh_requested.connect(
            lambda: worker.submit_command("refresh_feedback")
        )
        self.stop_requested.connect(
            lambda: worker.submit_command("cancel_motion")
        )
        self.emergency_stop_requested.connect(
            lambda: worker.submit_command("emergency_stop")
        )

        worker.connected_changed.connect(self._on_connected)
        worker.enabled_changed.connect(self._on_enabled)
        worker.error_occurred.connect(self._on_error)
        worker.enable_rejected.connect(self.enable_rejected.emit)
        worker.log_message.connect(self.log_message.emit)
        worker.can_fps_updated.connect(self.toolbar.set_fps)
        worker.can_fps_updated.connect(self.diagnostics_panel.update_can_stats)
        worker.joints_updated.connect(self._on_joints_updated)
        worker.efforts_updated.connect(self._on_efforts_updated)
        worker.end_pose_updated.connect(self._on_end_pose_updated)
        worker.motor_feedback_updated.connect(
            self.diagnostics_panel.update_motor_feedback
        )
        worker.motor_feedback_updated.connect(
            self.joint_panel.update_motor_feedback
        )
        worker.zero_sta_verified.connect(
            self.diagnostics_panel.update_zero_sta_result
        )
        worker.motor_scan_result.connect(self.diagnostics_panel.update_scan_result)

        self._connect_motion_panels(worker)

    def _connect_motion_panels(self, worker):
        self.joint_panel.joint_command.connect(
            lambda positions: worker.submit_command("joint_ctrl", positions)
        )

        tp = self.trajectory_panel
        tp.move_j_requested.connect(
            lambda positions, duration: worker.submit_command(
                "move_j", positions, duration
            )
        )
        tp.end_pose_requested.connect(
            lambda x, y, z, rx, ry, rz, duration: worker.submit_command(
                "end_pose_ctrl", x, y, z, rx, ry, rz, duration
            )
        )
        tp.end_pose_quat_requested.connect(
            lambda x, y, z, w, qx, qy, qz, duration: worker.submit_command(
                "end_pose_ctrl_quat", x, y, z, w, qx, qy, qz, duration
            )
        )
        tp.waypoints_batch_requested.connect(
            lambda waypoints, durations: worker.submit_command(
                "move_waypoints", waypoints, durations
            )
        )
        tp.cancel_requested.connect(self.stop_requested.emit)

        teach = self.teaching_panel
        teach.zero_torque_requested.connect(
            lambda enabled: worker.submit_command("zero_torque", enabled)
        )
        teach.zero_torque_gravity_requested.connect(
            lambda enabled: worker.submit_command("zero_torque_gravity", enabled)
        )
        teach.joint_ctrl_requested.connect(
            lambda positions: worker.submit_command("joint_ctrl", positions)
        )
        teach.move_j_requested.connect(
            lambda positions, duration: worker.submit_command(
                "move_j", positions, duration
            )
        )
        teach.trajectory_playback_requested.connect(
            lambda points: worker.submit_command("playback_trajectory", points)
        )

        gripper = self.gripper_panel
        gripper.gripper_command.connect(
            lambda angle, torque_max, torque_min: worker.submit_command(
                "gripper_ctrl", angle, torque_max, torque_min
            )
        )
        gripper.set_zero_requested.connect(
            lambda: worker.submit_command("set_zero_position", 7)
        )
        gripper.torque_limit_changed.connect(
            self.vr_panel.set_gripper_torque_limits
        )
        gripper.torque_limit_changed.connect(
            self.gamepad_panel.set_gripper_torque_limits
        )

        diagnostics = self.diagnostics_panel
        diagnostics.read_param_requested.connect(
            lambda motor_id, param: worker.submit_command(
                "read_motor_param", motor_id, param
            )
        )
        diagnostics.write_param_requested.connect(
            lambda motor_id, param, value: worker.submit_command(
                "write_motor_param", motor_id, param, value
            )
        )
        diagnostics.set_zero_requested.connect(
            lambda motor: worker.submit_command("set_zero_position", motor)
        )
        diagnostics.verify_zero_sta_requested.connect(
            lambda: worker.submit_command("verify_zero_sta")
        )
        diagnostics.scan_motors_requested.connect(
            lambda: worker.submit_command("scan_motors")
        )

        self.gamepad_panel.gamepad_log.connect(self.log_message.emit)
        self.joycon_panel.joycon_log.connect(self.log_message.emit)
        self.vr_panel.vr_log.connect(self.log_message.emit)
        self.vr_panel.record_toggle_requested.connect(
            lambda: self.log_message.emit(
                "双臂模式未启用 LeRobot 录制快捷键"
            )
        )

    def _on_toolbar_connect(self, selected_can, connect_kwargs):
        if connect_kwargs.get("backend") == "slcan":
            self._on_error("双臂模式使用预绑定 SocketCAN 机械臂")
            return
        if selected_can != self.can_name:
            self._on_error(
                f"{self.arm_name} 固定绑定 {self.can_name}，"
                f"不能连接 {selected_can}"
            )
            return
        self.connect_requested.emit()

    def request_connect(self):
        self.connect_requested.emit()

    def _connect_arm(self):
        if self.worker is not None:
            self.worker.submit_command(
                "connect", self.can_name, sim_mode=self._sim_mode
            )

    def _on_connected(self, connected):
        self._connected = bool(connected)
        if not connected:
            self._enabled = False
        self.toolbar.set_connected(connected)
        self.refresh_feedback_btn.setEnabled(connected)
        self.stop_btn.setEnabled(connected)
        self._update_movej_pd(connected)
        self._schedule_auto_sync(connected)

        arm = (
            self.worker.arm
            if connected and not self._sim_mode and self.worker is not None
            else None
        )
        self.gamepad_panel.set_arm(arm)
        self.vr_panel.set_arm(arm)
        self.joycon_panel.set_arm(arm)
        if not connected:
            self.joycon_panel.set_arm_enabled(False)
        self.connected_changed.emit(connected)

    def _on_enabled(self, enabled):
        self._enabled = bool(enabled)
        self.toolbar.set_enabled(enabled)
        self.joint_panel.set_enabled(enabled)
        self.joycon_panel.set_arm_enabled(enabled)
        self.enabled_changed.emit(enabled)

    def _on_error(self, message):
        self.toolbar.set_error(message)
        self.error_occurred.emit(message)

    def _update_movej_pd(self, connected):
        kp = kd = None
        if (connected and self.worker is not None and not self.worker._sim_mode
                and self.worker.arm is not None):
            try:
                kp, kd = self.worker.arm.GetPositionPD()
            except Exception:
                kp = kd = None
        self.trajectory_panel.update_pd_info(kp, kd)

    def _schedule_auto_sync(self, connected):
        if connected:
            self._auto_sync_pending = True
            self._auto_sync_tries = 0
            if self._auto_sync_timer is None:
                self._auto_sync_timer = QTimer(self)
                self._auto_sync_timer.setInterval(200)
                self._auto_sync_timer.timeout.connect(self._auto_sync_tick)
            self._auto_sync_timer.start()
        else:
            self._auto_sync_pending = False
            self._last_joint_states = None
            if self._auto_sync_timer is not None:
                self._auto_sync_timer.stop()

    def _auto_sync_tick(self):
        if not self._auto_sync_pending:
            if self._auto_sync_timer is not None:
                self._auto_sync_timer.stop()
            return
        if self._last_joint_states is None:
            self._auto_sync_tries += 1
            if self._auto_sync_tries >= 30:
                self._auto_sync_pending = False
                self._auto_sync_timer.stop()
                self.log_message.emit("等待关节反馈超时，自动同步失败")
            return
        self._auto_sync_pending = False
        self._auto_sync_timer.stop()
        self.joint_panel.update_feedback(self._last_joint_states)
        self.joint_panel.sync_to_feedback()
        self.log_message.emit("已自动同步当前位置")

    def _on_joints_updated(self, joint_states):
        self._last_joint_states = joint_states
        self.teaching_panel.update_positions(joint_states)
        self.trajectory_panel.update_current_positions(joint_states)

        positions = joint_states.to_list(include_gripper=True)
        self._last_gripper_angle_rad = (
            float(positions[6]) if len(positions) >= 7 else None
        )
        if self._kin is not None:
            try:
                self._last_end_pose = self._kin.forward_kinematics(positions[:6])
            except Exception as exc:
                logger.warning("[%s] 正运动学计算失败: %s", self.arm_name, exc)

        now = time.monotonic()
        if now - self._last_ui_update_time < self.UI_UPDATE_INTERVAL_S:
            return
        self._last_ui_update_time = now
        self.joint_panel.update_feedback(joint_states)
        self.gripper_panel.update_feedback(joint_states, self._last_effort_states)
        self.diagnostics_panel.update_motor_states(
            joint_states, None, self._last_effort_states
        )
        self.kinematics_display.update_end_pose(
            self._last_end_pose,
            self._last_gripper_angle_rad,
            self._get_gripper_torque(),
        )

    def _on_efforts_updated(self, effort_states):
        self._last_effort_states = effort_states

    def _on_end_pose_updated(self, end_pose):
        if self._kin is None:
            self._last_end_pose = end_pose
            self.kinematics_display.update_end_pose(
                end_pose,
                self._last_gripper_angle_rad,
                self._get_gripper_torque(),
            )

    def _get_gripper_torque(self):
        if not self._last_effort_states:
            return None
        try:
            torques = self._last_effort_states.to_list(include_gripper=True)
            return float(torques[6]) if len(torques) >= 7 else None
        except Exception:
            return None

    def get_joint_state(self):
        """Return J1-J6 plus gripper in radians, or None before feedback."""
        if self._last_joint_states is None:
            return None
        try:
            values = self._last_joint_states.to_list(include_gripper=True)
            if len(values) < 7:
                return None
            return [float(value) for value in values[:7]]
        except Exception:
            return None

    def get_gripper_torque(self):
        return self._get_gripper_torque()

    def _safe_exit(self):
        if not self._connected:
            self.log_message.emit("未连接，无需安全退出")
            return
        self.emergency_stop_requested.emit()
        self.disconnect_requested.emit()
        self.log_message.emit("已下发急停和断开命令")

    def _open_monitoring(self):
        if self.worker is None:
            return
        self.monitoring_window.show()
        self.monitoring_window.raise_()
        self.monitoring_window.activateWindow()

    def _on_theme_changed(self, _theme):
        if self.worker is not None:
            self.monitoring_window.panel.apply_theme()
        self.diagnostics_panel.retranslate_ui()
        self.gripper_panel.retranslate_ui()
        self.gamepad_panel.retranslate_ui()
        self.vr_panel.retranslate_ui()

    def _retranslate_ui(self, _language=None):
        titles = (
            tr("tab.joint"), tr("tab.trajectory"), tr("tab.teaching"),
            tr("tab.diagnostics"), tr("tab.gripper"), tr("tab.gamepad"),
            "Joy-Con", tr("tab.vr"), "末端状态",
        )
        for index, title in enumerate(titles):
            self.tabs.setTabText(index, title)
        for panel in (
            self.toolbar, self.joint_panel, self.trajectory_panel,
            self.teaching_panel, self.diagnostics_panel, self.gripper_panel,
            self.gamepad_panel, self.vr_panel, self.kinematics_display,
        ):
            if hasattr(panel, "retranslate_ui"):
                panel.retranslate_ui()
        if self.worker is not None:
            self.monitoring_window.retranslate_ui()

    def cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
        if self._theme_connections_active:
            try:
                self._theme_manager.language_changed.disconnect(
                    self._retranslate_ui
                )
                self._theme_manager.theme_changed.disconnect(
                    self._on_theme_changed
                )
            except (TypeError, RuntimeError):
                pass
            self._theme_connections_active = False
        if self._auto_sync_timer is not None:
            self._auto_sync_timer.stop()
        self.gamepad_panel.cleanup()
        self.joycon_panel.cleanup()
        self.vr_panel.cleanup()
        if self.worker is not None:
            monitor = self.monitoring_window.panel
            if hasattr(monitor, "_timer"):
                monitor._timer.stop()
            self.monitoring_window.hide()

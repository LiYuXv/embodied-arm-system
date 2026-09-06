"""主窗口：QDockWidget 布局 + 信号连接"""

import os
import time
import logging
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QTabWidget,
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QApplication,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, QEvent

try:
    from el_a3_sdk.kinematics import ELA3Kinematics
except Exception:
    ELA3Kinematics = None

from debugger.backend.arm_worker import ArmWorker
from debugger.widgets.toolbar_panel import ToolbarPanel
from debugger.widgets.joint_control_panel import JointControlPanel
from debugger.widgets.monitoring_window import MonitoringWindow
from debugger.widgets.trajectory_panel import TrajectoryPanel
from debugger.widgets.teaching_panel import TeachingPanel
from debugger.widgets.diagnostics_panel import DiagnosticsPanel
from debugger.widgets.gripper_panel import GripperPanel
from debugger.widgets.gamepad_panel import GamepadPanel
from debugger.widgets.vr_panel import VrPanel
from debugger.widgets.viewer_3d import Viewer3DPanel
from debugger.utils.i18n import tr
from debugger.utils.theme_manager import ThemeManager

# 摄像头帧信号 - 用于后续lerobot数据集收集
from debugger.widgets.camera_pyqt import CameraWidget

logger = logging.getLogger("debugger")


class MainWindow(QMainWindow):
    """EL-A3 调试上位机主窗口"""

    UI_UPDATE_INTERVAL_S = 0.05  # 20 Hz UI refresh cap

    def __init__(self, urdf_path=None, mesh_dir=None):
        super().__init__()
        self.setWindowTitle(tr("win.title"))
        self.setMinimumSize(1280, 800)
        self.resize(1600, 960)

        self._urdf_path = urdf_path
        self._mesh_dir = mesh_dir
        self._last_joint_states = None
        self._last_effort_states = None
        self._last_ui_update_time = 0.0
        self._last_end_pose = None
        self._last_gripper_angle_rad = None
        # VLA ↔ 手柄控制互锁重入守卫（防止“继续控制→断开手柄→恢复控制”递归）
        self._vla_interlock_guard = False
        # 连接成功后待执行“同步当前位置”状态（重试定时器 + 标记）
        self._auto_sync_pending = False
        self._auto_sync_tries = 0
        self._auto_sync_timer = None

        # 初始化运动学（用于从关节角实时计算末端位姿）
        self._kin = None
        if ELA3Kinematics is not None and urdf_path and os.path.isfile(urdf_path):
            try:
                self._kin = ELA3Kinematics(urdf_path=urdf_path)
                logger.info(f"[MainWindow] 运动学模型已加载: {urdf_path}")
            except Exception as e:
                logger.warning(f"[MainWindow] 加载运动学模型失败: {e}")
        else:
            logger.warning("[MainWindow] 未提供 URDF，无法从关节角计算末端位姿")

        self._init_worker()
        self._init_ui()
        self._connect_signals()

        tm = ThemeManager.instance()
        tm.language_changed.connect(lambda _: self._retranslate_ui())
        tm.theme_changed.connect(lambda _: self.viewer_3d.apply_theme())
        tm.theme_changed.connect(
            lambda _: self.monitoring_window.panel.apply_theme()
        )
        tm.theme_changed.connect(lambda _: self.diagnostics_panel.retranslate_ui())
        tm.theme_changed.connect(lambda _: self.gripper_panel.retranslate_ui())
        tm.theme_changed.connect(lambda _: self.gamepad_panel.retranslate_ui())
        tm.theme_changed.connect(lambda _: self.vr_panel.retranslate_ui())

        QTimer.singleShot(500, self._init_3d_model)

    def _init_worker(self):
        self.worker = ArmWorker()
        self.worker.start()

    def _init_ui(self):
        # --- 顶部工具栏（单行固定高度） ---
        toolbar_widget = ToolbarPanel()
        self.toolbar = toolbar_widget
        self.toolbar_dock = QDockWidget(tr("win.toolbar"), self)
        self.toolbar_dock.setWidget(toolbar_widget)
        self.toolbar_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        empty_title = QWidget()
        empty_title.setFixedHeight(0)
        self.toolbar_dock.setTitleBarWidget(empty_title)
        self.toolbar_dock.setStyleSheet("QDockWidget { border: none; }")
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.toolbar_dock)

        # --- 左侧：3D 可视化 ---
        self.viewer_3d = Viewer3DPanel(
            urdf_path=self._urdf_path,
            mesh_dir=self._mesh_dir,
        )
        self.viewer_dock = QDockWidget(tr("win.viewer"), self)
        self.viewer_dock.setWidget(self.viewer_3d)
        self.viewer_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        _hide = QWidget(); _hide.setFixedHeight(0)
        self.viewer_dock.setTitleBarWidget(_hide)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.viewer_dock)

        # --- 右侧：功能面板 Tab ---
        self.tabs = QTabWidget()

        self.joint_panel = JointControlPanel()
        self.tabs.addTab(self.joint_panel, tr("tab.joint"))

        self.trajectory_panel = TrajectoryPanel()
        self.tabs.addTab(self.trajectory_panel, tr("tab.trajectory"))

        self.teaching_panel = TeachingPanel()
        self.tabs.addTab(self.teaching_panel, tr("tab.teaching"))

        self.diagnostics_panel = DiagnosticsPanel()
        self.tabs.addTab(self.diagnostics_panel, tr("tab.diagnostics"))

        self.gripper_panel = GripperPanel()
        self.tabs.addTab(self.gripper_panel, tr("tab.gripper"))

        self.gamepad_panel = GamepadPanel()
        self.tabs.addTab(self.gamepad_panel, tr("tab.gamepad"))

        self.vr_panel = VrPanel()
        self.tabs.addTab(self.vr_panel, tr("tab.vr"))

        self.tabs_dock = QDockWidget(tr("win.panels"), self)
        self.tabs_dock.setWidget(self.tabs)
        self.tabs_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        _hide2 = QWidget(); _hide2.setFixedHeight(0)
        self.tabs_dock.setTitleBarWidget(_hide2)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.tabs_dock)

        self.log_console = QTextEdit()
        self.log_console.setObjectName("logConsole")
        self.log_console.setReadOnly(True)
        # 高度不再固定，由 _sync_log_console_height 跟随右侧摄像头面板高度

        # --- 底部：日志 + 末端位姿/摄像头显示 ---
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        # 左侧：日志（底部对齐，与右侧摄像头面板顶/底边对齐）
        bottom_layout.addWidget(
            self.log_console, stretch=1, alignment=Qt.AlignmentFlag.AlignBottom
        )

        # 右侧：末端位姿 + 摄像头面板（上下排列）
        from debugger.widgets.real_time_kinematics_display import RealTimeKinematicsDisplay
        self.kinematics_display = RealTimeKinematicsDisplay()
        self.camera_widget = CameraWidget(self, dataset_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lerobot_data"))
        self.camera_widget.setMinimumWidth(320)
        self.camera_widget.set_end_pose_callback(self._get_current_end_pose)
        self.camera_widget.set_torque_callback(self._get_gripper_torque)
        self.camera_widget.set_motion_callback(self.worker.submit_command)
        # 摄像头面板日志输出到主界面日志文本框（如 LeRobot 录制保存结果）
        self.camera_widget.camera_log.connect(self._append_log)
        # 日志框高度跟随摄像头面板：监听其尺寸变化实时同步
        self.camera_widget.installEventFilter(self)
        QTimer.singleShot(0, self._sync_log_console_height)

        right_bottom_widget = QWidget()
        right_bottom_layout = QVBoxLayout(right_bottom_widget)
        right_bottom_layout.setContentsMargins(0, 0, 0, 0)
        right_bottom_layout.setSpacing(4)
        right_bottom_layout.addWidget(self.kinematics_display)
        right_bottom_layout.addWidget(self.camera_widget, stretch=1)

        bottom_layout.addWidget(right_bottom_widget)

        self.log_dock = QDockWidget(tr("win.log"), self)
        self.log_dock.setWidget(bottom_widget)
        self.log_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        _hide3 = QWidget(); _hide3.setFixedHeight(0)
        self.log_dock.setTitleBarWidget(_hide3)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

        # --- 实时监控弹出窗口（按需打开） ---
        self.monitoring_window = MonitoringWindow(self.worker.data_buffer, parent=self)

        self.statusBar().showMessage(tr("win.ready"))

        QTimer.singleShot(0, lambda: self._adjust_dock_sizes(self.viewer_dock, self.tabs_dock))

    def _connect_signals(self):
        tb = self.toolbar
        tb.connect_requested.connect(self._on_connect)
        tb.disconnect_requested.connect(lambda: self.worker.submit_command("disconnect"))
        tb.enable_requested.connect(lambda: self.worker.submit_command("enable"))
        tb.disable_requested.connect(lambda: self.worker.submit_command("disable"))
        tb.emergency_stop_requested.connect(
            lambda: self.worker.submit_command("emergency_stop")
        )
        tb.safe_exit_requested.connect(self._on_safe_exit)
        tb.open_monitor_requested.connect(self._open_monitoring)

        self.worker.connected_changed.connect(tb.set_connected)
        # 连接后刷新轨迹控制页 MoveJ 的 PD 增益显示
        self.worker.connected_changed.connect(self._on_connected_update_movej_pd)
        # 连接成功后自动执行一次关节控制页的“同步当前位置”
        self.worker.connected_changed.connect(self._on_auto_sync_after_connect)
        self.worker.enabled_changed.connect(tb.set_enabled)
        self.worker.enabled_changed.connect(self.joint_panel.set_enabled)
        self.worker.enabled_changed.connect(self.viewer_3d.set_enabled)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.enable_rejected.connect(self._on_enable_rejected)
        self.worker.log_message.connect(self._append_log)
        self.worker.can_fps_updated.connect(tb.set_fps)
        self.worker.can_fps_updated.connect(self.diagnostics_panel.update_can_stats)

        # ---- 摄像头帧信号连接（为后续lerobot数据集收集做准备） ----
        # 当主界面摄像头启动后，摄像头帧信号可用于同步记录电机状态+视觉数据
        # 示例用法（后续lerobot集成时取消注释）：
        # self.camera_widget.frame_captured.connect(
        #     lambda idx, frame: self._on_camera_frame(idx, frame)
        # )

        self.worker.joints_updated.connect(self._on_joints_updated)
        self.worker.efforts_updated.connect(self._on_efforts_updated)
        self.worker.end_pose_updated.connect(self._on_end_pose_updated)
        self.worker.motor_feedback_updated.connect(
            self.diagnostics_panel.update_motor_feedback
        )
        self.worker.motor_feedback_updated.connect(
            self.joint_panel.update_motor_feedback
        )

        self.joint_panel.joint_command.connect(
            lambda pos: self.worker.submit_command("joint_ctrl", pos)
        )

        tp = self.trajectory_panel
        tp.move_j_requested.connect(
            lambda pos, dur: self.worker.submit_command("move_j", pos, dur)
        )
        tp.end_pose_requested.connect(
            lambda x, y, z, rx, ry, rz, d: self.worker.submit_command(
                "end_pose_ctrl", x, y, z, rx, ry, rz, d
            )
        )
        tp.end_pose_quat_requested.connect(
            lambda x, y, z, w, qx, qy, qz, d: self.worker.submit_command(
                "end_pose_ctrl_quat", x, y, z, w, qx, qy, qz, d
            )
        )
        tp.waypoints_batch_requested.connect(
            lambda wp, dur: self.worker.submit_command("move_waypoints", wp, dur)
        )
        tp.cancel_requested.connect(
            lambda: self.worker.submit_command("cancel_motion")
        )

        teach = self.teaching_panel
        teach.zero_torque_requested.connect(
            lambda en: self.worker.submit_command("zero_torque", en)
        )
        teach.zero_torque_gravity_requested.connect(
            lambda en: self.worker.submit_command("zero_torque_gravity", en)
        )
        teach.joint_ctrl_requested.connect(
            lambda pos: self.worker.submit_command("joint_ctrl", pos)
        )
        teach.move_j_requested.connect(
            lambda pos, dur: self.worker.submit_command("move_j", pos, dur)
        )
        teach.trajectory_playback_requested.connect(
            lambda pts: self.worker.submit_command("playback_trajectory", pts)
        )

        gp = self.gripper_panel
        gp.gripper_command.connect(
            lambda angle, t_max, t_min: self.worker.submit_command(
                "gripper_ctrl", angle, t_max, t_min
            )
        )
        gp.set_zero_requested.connect(
            lambda: self.worker.submit_command("set_zero_position", 7)
        )
        # 夹爪力矩限制 → VR 摇杆夹爪控制（闭合/张开受力矩上限/下限约束）
        gp.torque_limit_changed.connect(
            lambda t_max, t_min: self.vr_panel.set_gripper_torque_limits(t_max, t_min)
        )
        # 夹爪力矩限制 → 手柄 D-pad 夹爪控制（同一安全策略）
        gp.torque_limit_changed.connect(
            lambda t_max, t_min: self.gamepad_panel.set_gripper_torque_limits(t_max, t_min)
        )

        diag = self.diagnostics_panel
        diag.read_param_requested.connect(
            lambda mid, pidx: self.worker.submit_command("read_motor_param", mid, pidx)
        )
        diag.write_param_requested.connect(
            lambda mid, pidx, val: self.worker.submit_command(
                "write_motor_param", mid, pidx, val
            )
        )
        diag.set_zero_requested.connect(
            lambda m: self.worker.submit_command("set_zero_position", m)
        )
        diag.verify_zero_sta_requested.connect(
            lambda: self.worker.submit_command("verify_zero_sta")
        )
        self.worker.zero_sta_verified.connect(diag.update_zero_sta_result)
        diag.scan_motors_requested.connect(
            lambda: self.worker.submit_command("scan_motors")
        )
        self.worker.motor_scan_result.connect(diag.update_scan_result)

        tp.drag_mode_toggled.connect(self.viewer_3d.set_drag_mode)
        tp.sync_feedback_requested.connect(self.viewer_3d.sync_to_feedback)
        self.viewer_3d.drag_angles_changed.connect(tp.update_drag_angles)

        self.gamepad_panel.gamepad_log.connect(self._append_log)
        self.worker.connected_changed.connect(self._on_connected_for_gamepad)

        self.vr_panel.vr_log.connect(self._append_log)
        # VR 右手柄扳机键：快捷切换 LeRobot 录制（开始/停止）
        self.vr_panel.record_toggle_requested.connect(
            self._on_vr_record_toggle)

        # ---- VLA 推理 ↔ 手柄控制互锁 ----
        # 连接手柄 → 自动暂停 VLA 控制；断开手柄 → 自动恢复 VLA 控制；
        # VLA 请求恢复控制时 → 自动断开手柄。
        self.gamepad_panel.gamepad_connected.connect(self._on_gamepad_connected_for_vla)
        self.gamepad_panel.gamepad_disconnected.connect(self._on_gamepad_disconnected_for_vla)
        self.camera_widget.set_vla_takeover_callback(self._on_vla_takeover_requested)

    # ---- VLA ↔ 手柄互锁 ----

    def _on_gamepad_connected_for_vla(self):
        """手柄连接时：若 VLA 正在推理，自动触发“暂停控制”，让手柄安全接管。"""
        if self._vla_interlock_guard:
            return
        try:
            self.camera_widget.pause_vla_control()
        except Exception as e:
            logger.warning(f"[MainWindow] 联动暂停 VLA 控制失败: {e}")

    def _on_gamepad_disconnected_for_vla(self):
        """手柄断开时：若 VLA 仍在运行且处于暂停，自动触发“继续控制”，VLA 重新接管。"""
        if self._vla_interlock_guard:
            return
        try:
            self.camera_widget.resume_vla_control()
        except Exception as e:
            logger.warning(f"[MainWindow] 联动恢复 VLA 控制失败: {e}")

    def _on_vla_takeover_requested(self):
        """VLA 请求恢复控制权（点击“继续控制”）：若手柄仍连接，自动断开手柄。"""
        if self._vla_interlock_guard:
            return
        self._vla_interlock_guard = True
        try:
            if self.gamepad_panel.is_connected():
                logger.info("[MainWindow] VLA 恢复控制，自动断开手柄")
                self.gamepad_panel.disconnect_gamepad()
        except Exception as e:
            logger.warning(f"[MainWindow] 联动断开手柄失败: {e}")
        finally:
            self._vla_interlock_guard = False

    def _on_vr_record_toggle(self):
        """VR 右手柄扳机键：快捷切换 LeRobot 录制（开始/停止）"""
        try:
            was_recording = self.camera_widget._lerobot_recording
            self.camera_widget.toggle_lerobot_record()
            self._append_log(
                "[VR] 扳机键: 停止 LeRobot 录制" if was_recording
                else "[VR] 扳机键: 开始 LeRobot 录制")
        except Exception as e:
            logger.warning(f"[MainWindow] VR 扳机键切换录制失败: {e}")

    # ---- retranslate ----

    def _retranslate_ui(self):
        self.setWindowTitle(tr("win.title"))
        self.toolbar_dock.setWindowTitle(tr("win.toolbar"))
        self.viewer_dock.setWindowTitle(tr("win.viewer"))
        self.tabs_dock.setWindowTitle(tr("win.panels"))
        self.log_dock.setWindowTitle(tr("win.log"))
        self.statusBar().showMessage(tr("win.ready"))

        self.tabs.setTabText(0, tr("tab.joint"))
        self.tabs.setTabText(1, tr("tab.trajectory"))
        self.tabs.setTabText(2, tr("tab.teaching"))
        self.tabs.setTabText(3, tr("tab.diagnostics"))
        self.tabs.setTabText(4, tr("tab.gripper"))
        self.tabs.setTabText(5, tr("tab.gamepad"))
        self.tabs.setTabText(6, tr("tab.vr"))

        for panel in (self.toolbar, self.joint_panel, self.trajectory_panel,
                      self.teaching_panel, self.diagnostics_panel,
                      self.gripper_panel, self.gamepad_panel, self.vr_panel,
                      self.viewer_3d):
            if hasattr(panel, "retranslate_ui"):
                panel.retranslate_ui()
        self.monitoring_window.retranslate_ui()
        if hasattr(self, 'kinematics_display') and self.kinematics_display:
            self.kinematics_display.retranslate_ui()

    # ---- helpers ----

    def _adjust_dock_sizes(self, viewer_dock, tabs_dock):
        w = self.width()
        left_w = int(w * 0.50)
        right_w = w - left_w
        self.resizeDocks(
            [viewer_dock, tabs_dock], [left_w, right_w], Qt.Orientation.Horizontal
        )

    def _sync_log_console_height(self):
        """把左下角日志框高度同步为摄像头面板高度（与 CameraWidget 顶/底边对齐）"""
        h = self.camera_widget.height()
        if h > 0 and self.log_console.height() != h:
            self.log_console.setFixedHeight(h)

    def eventFilter(self, obj, event):
        # 摄像头面板尺寸变化（窗口缩放、dock 调整等）时同步日志框高度
        if obj is self.camera_widget and event.type() == QEvent.Type.Resize:
            self._sync_log_console_height()
        return super().eventFilter(obj, event)

    def _open_monitoring(self):
        mw = self.monitoring_window
        if mw.isVisible():
            mw.raise_()
            mw.activateWindow()
        else:
            mw.show()
            mw.raise_()

    # ---- 摄像头帧处理回调（预留接口，供后续lerobot数据集收集使用） ----
    def _on_camera_frame(self, camera_idx: int, frame):
        """摄像头帧回调：可在此同步记录电机状态与视觉帧，用于lerobot数据集收集。

        示例集成思路（后续实现）：
        1. 获取当前电机关节状态: self._last_joint_states
        2. 获取末端执行器状态: self._last_effort_states
        3. 将 (frame, joint_states, timestamp) 写入lerobot格式的数据集
        """
        pass

    def _on_connected_update_movej_pd(self, connected: bool):
        """连接/断开后刷新轨迹控制页 MoveJ 的 PD 增益信息框。

        真实连接时从 SDK 读取全局默认 PD（MoveJ 走控制循环，
        实际增益为 _position_kp/_position_kd，debugger 未做逐关节覆盖）；
        模拟模式或断开时显示 SDK 默认值。
        """
        kp = kd = None
        if connected and not self.worker._sim_mode and self.worker.arm is not None:
            try:
                kp, kd = self.worker.arm.GetPositionPD()
            except Exception:
                kp = kd = None
        self.trajectory_panel.update_pd_info(kp, kd)

    def _on_auto_sync_after_connect(self, connected: bool):
        """连接成功后：轮询等待关节反馈到达，自动执行关节控制页的“同步当前位置”。

        不能收到 connected_changed 就立刻同步——连接瞬间 SDK 缓存的关节消息
        可能还是全 0 或尚未到达，必须等 50Hz 轮询把真实电机位置送达后再同步，
        否则滑块会被置 0；反馈未就绪时定时重试（最多约 6 秒）。
        """
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
            # 清空旧反馈，确保下次连接同步到新反馈而不是上次会话的旧值
            self._last_joint_states = None
            if self._auto_sync_timer is not None:
                self._auto_sync_timer.stop()

    def _auto_sync_tick(self):
        """自动同步重试：反馈到达后用最新关节位置刷新显示并点击“同步当前位置”。"""
        if not self._auto_sync_pending:
            if self._auto_sync_timer is not None:
                self._auto_sync_timer.stop()
            return
        if self._last_joint_states is None:
            self._auto_sync_tries += 1
            if self._auto_sync_tries >= 30:  # ~6 秒超时
                self._auto_sync_pending = False
                if self._auto_sync_timer is not None:
                    self._auto_sync_timer.stop()
                logger.warning("[MainWindow] 等待关节反馈超时，自动同步当前位置失败")
            return
        self._auto_sync_pending = False
        if self._auto_sync_timer is not None:
            self._auto_sync_timer.stop()
        try:
            # 先刷新位置显示，再点击“同步当前位置”（更新滑块/spinbox 到真实位置）
            self.joint_panel.update_feedback(self._last_joint_states)
            self.joint_panel.sync_btn.click()
            self._append_log("已自动同步当前位置")
        except Exception as e:
            logger.warning(f"[MainWindow] 自动同步当前位置失败: {e}")

    def _on_connected_for_gamepad(self, connected: bool):
        if connected and self.worker.arm is not None:
            self.gamepad_panel.set_arm(self.worker.arm)
            self.vr_panel.set_arm(self.worker.arm)
        elif not connected:
            self.gamepad_panel.set_arm(None)
            self.vr_panel.set_arm(None)

    def _on_connect(self, can_name: str, connect_kwargs: dict):
        self.worker.submit_command("connect", can_name, **connect_kwargs)

    def _on_joints_updated(self, joint_states):
        self._last_joint_states = joint_states

        self.teaching_panel.update_positions(joint_states)
        self.trajectory_panel.update_current_positions(joint_states)

        # 提取夹爪角度（弧度，与机械臂 GripperCtrl 接口、数据采集/推理保持一致）
        positions = joint_states.to_list(include_gripper=True)
        if len(positions) >= 7:
            self._last_gripper_angle_rad = float(positions[6])
        else:
            self._last_gripper_angle_rad = None

        # 关键修复：从关节角实时正运动学计算末端位姿，避免依赖 worker 的 5Hz 慢速 polling
        if self._kin is not None:
            try:
                arm_positions = positions[:6]
                self._last_end_pose = self._kin.forward_kinematics(arm_positions)
            except Exception as e:
                logger.warning(f"[MainWindow] 正运动学计算失败: {e}")

        now = time.monotonic()
        if now - self._last_ui_update_time < self.UI_UPDATE_INTERVAL_S:
            return
        self._last_ui_update_time = now

        self.joint_panel.update_feedback(joint_states)
        self.viewer_3d.update_joint_angles(joint_states)
        self.gripper_panel.update_feedback(joint_states, self._last_effort_states)
        self.diagnostics_panel.update_motor_states(
            joint_states, None, self._last_effort_states
        )
        # 同步更新末端位姿显示（使用 FK 计算结果）
        if hasattr(self, 'kinematics_display') and self.kinematics_display:
            self.kinematics_display.update_end_pose(
                self._last_end_pose, self._last_gripper_angle_rad,
                self._get_gripper_torque()
            )

    def _on_efforts_updated(self, effort_states):
        self._last_effort_states = effort_states

    def _get_current_end_pose(self):
        """供摄像头组件获取最新末端位姿和夹爪角度（夹爪为弧度）"""
        end_pose = getattr(self, "_last_end_pose", None)
        gripper_rad = getattr(self, "_last_gripper_angle_rad", None)
        return end_pose, gripper_rad

    def _get_gripper_torque(self):
        """获取最新夹爪力矩 (Nm)，读不到时返回 None"""
        if not self._last_effort_states:
            return None
        try:
            torques = self._last_effort_states.to_list(include_gripper=True)
            if len(torques) >= 7:
                return float(torques[6])
        except Exception:
            return None
        return None

    def _on_end_pose_updated(self, end_pose):
        """末端位姿更新回调（备用：当无法从关节角计算 FK 时使用）。"""
        # 只有当运动学模型不可用时，才使用 worker 传来的末端位姿
        if self._kin is None:
            self._last_end_pose = end_pose
            if hasattr(self, 'kinematics_display') and self.kinematics_display:
                self.kinematics_display.update_end_pose(
                    end_pose, getattr(self, "_last_gripper_angle_rad", None),
                    self._get_gripper_torque()
                )

    def _on_error(self, msg: str):
        self.toolbar.set_error(msg)
        self._append_log(tr("win.error", msg=msg))

    def _on_enable_rejected(self, msg: str):
        """使能前检查未通过：状态栏 + 日志 + 弹窗提示用户"""
        self.toolbar.set_error(msg.split("\n")[0])
        QMessageBox.warning(self, tr("tb.enable"), msg)

    def _append_log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_console.append(f"[{timestamp}] {msg}")
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _init_3d_model(self):
        success = self.viewer_3d.initialize_model()
        if success:
            self._append_log(tr("win.model_ok"))
        else:
            self._append_log(tr("win.model_fail"))

    def _on_safe_exit(self):
        """安全退出：急停电机并断开连接，最后关闭关节控制界面"""
        if not self.worker.is_connected:
            self._append_log("未连接，无需安全退出")
            return

        self._append_log("=== 开始安全退出 ===")

        # 1. 急停（停止控制循环 + 失能电机 + 清除故障）
        self._append_log("  [1/3] 执行急停...")
        self.worker.submit_command("emergency_stop")
        import time
        time.sleep(0.5)

        # 2. 断开连接
        self._append_log("  [2/3] 断开连接...")
        self.worker.submit_command("disconnect")
        time.sleep(0.3)

        # 3. 停止后台线程
        self._append_log("  [3/3] 停止后台线程...")
        self.worker.stop()

        # 4. 关闭关节控制界面和示教界面
        self._append_log("  [4/4] 关闭关节控制界面...")
        if hasattr(self, 'joint_panel') and self.joint_panel.isVisible():
            self.joint_panel.close()
        if hasattr(self, 'teaching_panel') and self.teaching_panel.isVisible():
            self.teaching_panel.close()

        self._append_log("=== 安全退出完成 ===")

    def closeEvent(self, event):
        self._append_log(tr("win.closing"))
        # 直接执行急停，确保电机失能
        if self.worker.is_connected:
            self._append_log("正在执行急停...")
            self.worker.submit_command("emergency_stop")
            import time
            time.sleep(0.5)  # 等待急停命令执行完成
        self._cleanup_before_close()
        super().closeEvent(event)

    def _cleanup_before_close(self):
        """关闭程序前的资源清理"""
        self._append_log("正在清理资源...")

        # 停止所有定时器
        if hasattr(self.monitoring_window, 'panel') and hasattr(self.monitoring_window.panel, '_timer'):
            self.monitoring_window.panel._timer.stop()

        # 停止手柄控制
        self.gamepad_panel.cleanup()

        # 停止 VR 控制
        self.vr_panel.cleanup()

        # 断开连接
        if self.worker.is_connected:
            self.worker.submit_command("disconnect")
            import time
            time.sleep(0.2)

        # 停止后台线程
        self.worker.stop()

        # 关闭监控窗口
        if self.monitoring_window.isVisible():
            self.monitoring_window.close()

        # 清理摄像头资源
        if hasattr(self, 'camera_widget') and self.camera_widget:
            self.camera_widget.cleanup()

        # 释放3D视图
        if hasattr(self.viewer_3d, '_plotter') and self.viewer_3d._plotter:
            try:
                self.viewer_3d._plotter.close()
                self.viewer_3d._plotter = None
            except Exception:
                pass

        # 清理变量
        self._last_joint_states = None
        self._last_effort_states = None
        self._last_velocity_states = None
        self._last_end_pose = None
        self._last_status = None
        self._last_motor_feedback = None

        self._append_log("已清理资源")

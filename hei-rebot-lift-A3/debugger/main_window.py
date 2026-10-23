"""主窗口：QDockWidget 布局 + 信号连接"""

import os
import time
import math
import logging
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QTabWidget,
    QWidget, QVBoxLayout, QTextEdit, QApplication,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer

try:
    from el_a3_sdk.kinematics import ELA3Kinematics
except Exception:
    ELA3Kinematics = None

from debugger.backend.arm_worker import ArmWorker
from debugger.widgets.toolbar_panel import ToolbarPanel
from debugger.widgets.joint_control_panel import JointControlPanel
from debugger.widgets.monitoring_window import MonitoringWindow

try:
    from debugger.widgets.performance_monitor import PerformanceMonitorDialog
except Exception as _perf_err:  # noqa: F841
    PerformanceMonitorDialog = None
from debugger.widgets.trajectory_panel import TrajectoryPanel
from debugger.widgets.teaching_panel import TeachingPanel
from debugger.widgets.diagnostics_panel import DiagnosticsPanel
from debugger.widgets.gripper_panel import GripperPanel
from debugger.widgets.gamepad_panel import GamepadPanel
from debugger.widgets.vr_panel import VrPanel
from debugger.utils.i18n import tr
from debugger.utils.theme_manager import ThemeManager
from debugger.utils.system_config import load_config
from debugger.widgets.settings_dialog import SettingsDialog

# 底盘 + 升降控制面板（工控机本地 CAN 电机，复用 chassis_lift_status）
from chassis_lift_status import create_embedded_panel

# 摄像头帧信号 - 用于后续lerobot数据集收集
from debugger.widgets.camera_pyqt import CameraWidget

logger = logging.getLogger("debugger")


class MainWindow(QMainWindow):
    """EL-A3 调试上位机主窗口"""

    UI_UPDATE_INTERVAL_S = 0.05  # 20 Hz UI refresh cap

    def __init__(self, urdf_path=None, mesh_dir=None, auto_connect=None):
        super().__init__()
        self.setWindowTitle(tr("win.title"))
        self.setMinimumSize(800, 1100)
        self.resize(900, 1400)

        # 加载系统设置（settings/main_system_config.json）
        self._sys_config = load_config()
        if auto_connect is None:
            auto_connect = bool(
                self._sys_config.get("auto_connect_on_startup", True))
        # 自动连接成功后自动使能（跟随自动连接开关）
        self._auto_enable_pending = False

        self._urdf_path = urdf_path
        self._mesh_dir = mesh_dir
        self._last_joint_states = None
        self._last_joint_states2 = None
        self._last_effort_states = None
        self._last_effort_states2 = None
        self._last_ui_update_time = 0.0
        self._last_ui_update_time2 = 0.0  # 臂2 UI 刷新节流（独立时间戳）
        self._last_end_pose = None
        self._last_end_pose2 = None
        self._last_gripper_angle_rad = None
        self._last_gripper_angle_rad2 = None
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
        tm.theme_changed.connect(
            lambda _: self.monitoring_window.panel.apply_theme()
        )
        tm.theme_changed.connect(lambda _: self.diagnostics_panel.retranslate_ui())
        tm.theme_changed.connect(lambda _: self.gripper_panel.retranslate_ui())
        tm.theme_changed.connect(lambda _: self.gamepad_panel.retranslate_ui())
        tm.theme_changed.connect(lambda _: self.vr_panel.retranslate_ui())

        # 应用配置中的 CAN 波特率到工具栏下拉框
        try:
            self.toolbar.set_can_bitrate(
                int(self._sys_config.get("can_bitrate", 500000)))
        except Exception as e:
            logger.warning(f"[MainWindow] 应用 CAN 波特率配置失败: {e}")

        QTimer.singleShot(500, self._on_chassis_ready)

        if auto_connect:
            QTimer.singleShot(1000, self._auto_connect_arms)

    def _auto_connect_arms(self):
        if not self.toolbar._connected:
            self._append_log("[Main] 启动自动连接机械臂（臂1=can0，臂2=can1）")
            self._auto_enable_pending = True
            self.toolbar.connect_btn.click()

    def _on_auto_enable_after_connect(self, connected: bool):
        """自动连接成功后自动使能双臂（仅启动时自动连接触发一次）"""
        if not self._auto_enable_pending:
            return
        if not connected:
            self._auto_enable_pending = False
            return
        self._auto_enable_pending = False
        self._append_log("[Main] 自动连接成功，自动使能双臂")
        QTimer.singleShot(
            300, lambda: self.worker.submit_command("enable"))

    def _open_settings(self):
        """打开系统设置对话框，保存后应用 CAN 波特率到工具栏"""
        try:
            camera_ports = self.camera_widget.camera_usb_ports()
        except Exception:
            camera_ports = None
        dlg = SettingsDialog(self, camera_ports=camera_ports)
        dlg.settings_saved.connect(self._on_settings_saved)
        dlg.exec()

    def _on_settings_saved(self, cfg: dict):
        self._sys_config.update(cfg)
        try:
            self.toolbar.set_can_bitrate(int(cfg.get("can_bitrate", 500000)))
        except Exception as e:
            logger.warning(f"[MainWindow] 应用 CAN 波特率配置失败: {e}")
        # 摄像头角色绑定已写入 main_system_config.json，立即重新分配生效
        try:
            if hasattr(self, "camera_widget") and self.camera_widget is not None:
                self.camera_widget.apply_camera_roles()
        except Exception as e:
            logger.warning(f"[MainWindow] 应用摄像头角色设置失败: {e}")
        # VR 面板默认设置即时应用
        try:
            self.vr_panel.apply_defaults(cfg.get("vr_defaults", {}))
        except Exception as e:
            logger.warning(f"[MainWindow] 应用 VR 默认设置失败: {e}")
        self._append_log(
            f"[设置] 已保存: 自动连接="
            f"{'开' if cfg.get('auto_connect_on_startup') else '关'}, "
            f"CAN 波特率={int(cfg.get('can_bitrate', 500000)) // 1000}K")

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

        # --- 底盘 + 升降控制面板（作为底部 Tab 的第一分页） ---
        self._chassis_win, self.chassis_panel = create_embedded_panel()

        # --- 底部：功能面板 Tab（竖屏布局，横跨整个窗口宽度） ---
        self.tabs = QTabWidget()

        self.tabs.addTab(self.chassis_panel, "底盘 / 升降")

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
        self.vr_panel.apply_defaults(self._sys_config.get("vr_defaults", {}))
        self.tabs.addTab(self.vr_panel, tr("tab.vr"))

        self.tabs_dock = QDockWidget(tr("win.panels"), self)
        self.tabs_dock.setWidget(self.tabs)
        self.tabs_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        _hide2 = QWidget(); _hide2.setFixedHeight(0)
        self.tabs_dock.setTitleBarWidget(_hide2)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.tabs_dock)

        self.log_console = QTextEdit()
        self.log_console.setObjectName("logConsole")
        self.log_console.setReadOnly(True)
        self.log_console.setFixedHeight(160)

        # --- 中央：末端位姿 + 摄像头 + 日志（竖屏上下排列） ---
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(4)

        from debugger.widgets.real_time_kinematics_display import RealTimeKinematicsDisplay
        self.kinematics_display = RealTimeKinematicsDisplay()
        self.camera_widget = CameraWidget(self, dataset_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lerobot_data"))
        self.camera_widget.setMinimumWidth(320)
        self.camera_widget.set_end_pose_callback(self._get_current_end_pose)
        self.camera_widget.set_end_pose2_callback(self._get_current_end_pose2)
        self.camera_widget.set_torque_callback(self._get_gripper_torque)
        self.camera_widget.set_torque2_callback(self._get_gripper_torque2)
        self.camera_widget.set_motion_callback(self.worker.submit_command)
        # 摄像头面板日志输出到主界面日志文本框（如 LeRobot 录制保存结果）
        self.camera_widget.camera_log.connect(self._append_log)

        central_layout.addWidget(self.kinematics_display)
        central_layout.addWidget(self.camera_widget, stretch=1)
        central_layout.addWidget(self.log_console)

        self.setCentralWidget(central_widget)

        # --- 实时监控弹出窗口（按需打开） ---
        self.monitoring_window = MonitoringWindow(self.worker.data_buffer, parent=self)
        # --- 系统性能监测对话框（「监控」按钮弹出，按需创建） ---
        self._perf_dialog = None

        self.statusBar().showMessage(tr("win.ready"))

        QTimer.singleShot(0, self._adjust_dock_sizes)

    def _connect_signals(self):
        tb = self.toolbar
        tb.connect_requested.connect(self._on_connect)
        tb.disconnect_requested.connect(lambda: self.worker.submit_command("disconnect"))
        tb.enable_requested.connect(lambda: self.worker.submit_command("enable"))
        tb.disable_requested.connect(self._on_disable_requested)
        tb.emergency_stop_requested.connect(
            lambda: self.worker.submit_command("emergency_stop")
        )
        tb.safe_exit_requested.connect(self._on_safe_exit)
        tb.open_monitor_requested.connect(self._open_monitoring)
        tb.open_settings_requested.connect(self._open_settings)

        self.worker.connected_changed.connect(tb.set_connected)
        # 连接后刷新轨迹控制页 MoveJ 的 PD 增益显示
        self.worker.connected_changed.connect(self._on_connected_update_movej_pd)
        # 连接成功后自动执行一次关节控制页的“同步当前位置”
        self.worker.connected_changed.connect(self._on_auto_sync_after_connect)
        # 启动自动连接成功后自动使能
        self.worker.connected_changed.connect(self._on_auto_enable_after_connect)
        self.worker.enabled_changed.connect(tb.set_enabled)
        self.worker.enabled_changed.connect(self.joint_panel.set_enabled)
        # 失能/急停时自动断开 VR 连接（停止双臂 VR 控制）
        self.worker.enabled_changed.connect(self._on_disabled_disconnect_vr)
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
        self.worker.joints_updated2.connect(self._on_joints_updated2)
        self.worker.efforts_updated.connect(self._on_efforts_updated)
        self.worker.efforts_updated2.connect(self._on_efforts_updated2)
        self.worker.end_pose_updated.connect(self._on_end_pose_updated)
        self.worker.motor_feedback_updated.connect(
            self.diagnostics_panel.update_motor_feedback
        )
        self.worker.motor_feedback_updated.connect(
            self.joint_panel.update_motor_feedback
        )
        self.worker.motor_feedback_updated2.connect(
            self.joint_panel.update_motor_feedback_arm2
        )
        self.worker.motor_feedback_updated2.connect(
            self.diagnostics_panel.update_motor_feedback_arm2
        )

        self.joint_panel.joint_command.connect(
            lambda pos1, pos2: self.worker.submit_command("joint_ctrl", pos1, pos2)
        )
        # 关节软力矩限制 → SDK 控制循环（双臂同时生效）
        self.joint_panel.torque_limit_changed.connect(
            lambda l13, l46, en: self.worker.submit_command(
                "set_torque_limits", l13, l46, en)
        )

        tp = self.trajectory_panel
        tp.move_j_requested.connect(
            lambda pos1, pos2, dur: self.worker.submit_command(
                "move_j_dual", pos1, pos2, dur)
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
            lambda pos1, pos2: self.worker.submit_command("joint_ctrl", pos1, pos2)
        )
        teach.move_j_requested.connect(
            lambda pos, dur: self.worker.submit_command("move_j", pos, dur)
        )
        teach.trajectory_playback_requested.connect(
            lambda pts1, pts2: self.worker.submit_command(
                "playback_trajectory", pts1, pts2)
        )

        # --- 标定面板（示教 Tab 内子分页，臂1/臂2 可选） ---
        calib = self.teaching_panel.calibration_panel
        calib.move_j_requested.connect(
            lambda arm_sel, pos, dur: self.worker.submit_command(
                "move_j_arm_block", arm_sel, pos, dur)
        )
        self.worker.move_j_done.connect(calib.notify_move_done)

        gp = self.gripper_panel
        gp.gripper_command.connect(
            lambda a1, a2, t_max, t_min: self.worker.submit_command(
                "gripper_ctrl", a1, t_max, t_min, a2
            )
        )
        # 夹爪设零：双臂同时进行（arm_sel=2）
        gp.set_zero_requested.connect(
            lambda: self.worker.submit_command("set_zero_position", 7, 2)
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
            lambda arm, mid, pidx: self.worker.submit_command(
                "read_motor_param", arm, mid, pidx)
        )
        diag.write_param_requested.connect(
            lambda arm, mid, pidx, val: self.worker.submit_command(
                "write_motor_param", arm, mid, pidx, val
            )
        )
        diag.set_zero_requested.connect(
            lambda arm, m: self.worker.submit_command("set_zero_position", m, arm)
        )
        diag.verify_zero_sta_requested.connect(
            lambda arm: self.worker.submit_command("verify_zero_sta", arm)
        )
        diag.set_all_zero_sta_requested.connect(
            lambda arm: self.worker.submit_command("set_all_zero_sta", arm)
        )
        self.worker.zero_sta_verified.connect(diag.update_zero_sta_result)
        diag.scan_motors_requested.connect(
            lambda arm: self.worker.submit_command("scan_motors", arm)
        )
        self.worker.motor_scan_result.connect(diag.update_scan_result)

        self.gamepad_panel.gamepad_log.connect(self._append_log)
        self.worker.connected_changed.connect(self._on_connected_for_gamepad)

        self.vr_panel.vr_log.connect(self._append_log)
        # VR 左摇杆 → 底盘平移（右摇杆 X → 底盘旋转）
        self.vr_panel.chassis_cmd.connect(self._on_vr_chassis_cmd)
        # VR 左手柄 Y/X 键连续升降速度
        self.vr_panel.lift_vel_cmd.connect(self._on_vr_lift_vel_cmd)
        # VR 左手柄 X 按住 → 头显位姿控制底盘/升降
        self.vr_panel.hmd_cmd.connect(self._on_vr_hmd_cmd)
        self.vr_panel.hmd_released.connect(self._on_vr_hmd_released)
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

    def _on_vr_chassis_cmd(self, fb: float, lr: float, ld: float, rd: float,
                           rot: float, lift_mm: float):
        """VR 摇杆底盘方向指令：fb/lr/ld/rd 语义与主界面四个方向滑条完全一致
        （fb +前进，lr +右移，ld +前左，rd +前右），rot 为旋转；VR 为独立于底盘
        内部实现的上层控制，全向运动合成与加减速斜坡由底盘面板统一执行。
        lift_mm 为右摇杆 Y 升降位置目标（nan=不控升降，升降保持）。"""
        try:
            self._chassis_win.set_external_direction_pads(fb, lr, ld, rd, rot)
            if not math.isnan(lift_mm):
                self._chassis_win.set_lift_position_mm(lift_mm)
        except Exception as e:
            logger.warning(f"[MainWindow] VR 底盘指令失败: {e}")

    def _on_vr_lift_vel_cmd(self, direction: float, speed_rpm: float):
        """VR 左手柄 Y/X 键连续升降速度控制（direction -1=升 +1=降 0=停）。"""
        try:
            self._chassis_win.set_lift_velocity(direction, speed_rpm=speed_rpm)
        except Exception as e:
            logger.warning(f"[MainWindow] VR 升降速度控制失败: {e}")

    def _on_vr_hmd_cmd(self, target_x: float, target_y: float, target_yaw: float,
                       max_lin: float, lift_mm: float):
        """VR 头显底盘/升降指令（左手柄 X 按住期间 20Hz 下发）：
        target_x/y/yaw 为世界系底盘目标位姿 (m, m, rad)，max_lin 为位置伺服
        线速度上限 (m/s)，lift_mm 为升降位置目标 (nan=不控升降)。底盘走里程计
        位置伺服 + 加速度斜坡，升降走位置伺服通道。"""
        try:
            self._chassis_win.set_external_pose_target(
                target_x, target_y, target_yaw, max_lin=max_lin)
            if not math.isnan(lift_mm):
                self._chassis_win.set_lift_position_mm(lift_mm)
        except Exception as e:
            logger.warning(f"[MainWindow] VR 头显底盘指令失败: {e}")

    def _on_vr_hmd_released(self):
        """VR 左手柄 X 松开：底盘平滑停车（清位姿目标沿斜坡减速），升降冻结。"""
        try:
            self._chassis_win.clear_external_pose_target()
            self._chassis_win.clear_lift_target()
        except Exception as e:
            logger.warning(f"[MainWindow] VR 头显控制释放失败: {e}")

    def _on_chassis_ready(self):
        """底盘面板就绪后的初始化日志 + 云端推理底盘回调接线。"""
        try:
            self.camera_widget.set_mobile_base_callbacks(
                get_state=self._chassis_win.get_mobile_state,
                lift_position=self._chassis_win.set_lift_position_mm,
                chassis_velocity=self._chassis_win.set_external_velocity_si,
            )
            # VR 头显升降锚定需要读取当前升降高度，位置跟随需要读取底盘位姿
            # （位姿走里程计 get_odom_pose；get_mobile_state 为训练用速度语义）
            self.vr_panel.set_lift_provider(self._chassis_win.get_lift_mm)
            self.vr_panel.set_chassis_pose_provider(
                self._chassis_win.get_odom_pose)
            self._append_log("底盘/升降面板已就绪（已接入 VR 摇杆与云端推理）")
        except Exception as e:
            logger.warning(f"[MainWindow] 底盘回调接线失败: {e}")

    # ---- retranslate ----

    def _retranslate_ui(self):
        self.setWindowTitle(tr("win.title"))
        self.toolbar_dock.setWindowTitle(tr("win.toolbar"))
        self.tabs_dock.setWindowTitle(tr("win.panels"))
        self.statusBar().showMessage(tr("win.ready"))

        self.tabs.setTabText(0, "底盘 / 升降")
        self.tabs.setTabText(1, tr("tab.joint"))
        self.tabs.setTabText(2, tr("tab.trajectory"))
        self.tabs.setTabText(3, tr("tab.teaching"))
        self.tabs.setTabText(4, tr("tab.diagnostics"))
        self.tabs.setTabText(5, tr("tab.gripper"))
        self.tabs.setTabText(6, tr("tab.gamepad"))
        self.tabs.setTabText(7, tr("tab.vr"))

        for panel in (self.toolbar, self.joint_panel, self.trajectory_panel,
                      self.teaching_panel, self.diagnostics_panel,
                      self.gripper_panel, self.gamepad_panel, self.vr_panel):
            if hasattr(panel, "retranslate_ui"):
                panel.retranslate_ui()
        self.monitoring_window.retranslate_ui()
        if hasattr(self, 'kinematics_display') and self.kinematics_display:
            self.kinematics_display.retranslate_ui()

    # ---- helpers ----

    def _adjust_dock_sizes(self):
        # 竖屏布局：底部 Tab 面板约占 45% 高度
        h = self.height()
        self.resizeDocks(
            [self.tabs_dock], [int(h * 0.45)], Qt.Orientation.Vertical
        )

    def _on_disable_requested(self):
        """失能：若零力矩已开启先自动关闭，再询问是否先回安全位置（第4关节26°，其余关节0）"""
        # 若示教模式处于"开启零力矩"状态，先自动切换为"关闭零力矩"
        # （恢复位置保持，避免零力矩状态下直接失能/回安全位置异常）
        if self.teaching_panel._zero_torque_active:
            self._append_log("[Main] 失能前检测到零力矩已开启，自动关闭零力矩")
            self.teaching_panel._toggle_zero_torque()
        ret = QMessageBox.question(
            self, "失能确认",
            "失能前是否让双臂回到安全位置？\n（第 4 关节回到 26°，其余关节回到 0°）",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if ret == QMessageBox.StandardButton.Cancel:
            return
        if ret == QMessageBox.StandardButton.Yes:
            self._append_log("[Main] 失能前双臂回安全位置（J4=26°，其余0）")
            self.worker.submit_command("move_safe_then_disable")
        else:
            self.worker.submit_command("disable")

    def _open_monitoring(self):
        if PerformanceMonitorDialog is None:
            self._append_log("[Main] 性能监测模块不可用（需安装 psutil）")
            return
        if self._perf_dialog is None:
            self._perf_dialog = PerformanceMonitorDialog(self)
        self._perf_dialog.show()
        self._perf_dialog.raise_()
        self._perf_dialog.activateWindow()

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
            self._last_joint_states2 = None
            if self._auto_sync_timer is not None:
                self._auto_sync_timer.stop()

    def _auto_sync_tick(self):
        """自动同步重试：双臂反馈都到达后刷新显示并点击“同步当前位置”。"""
        if not self._auto_sync_pending:
            if self._auto_sync_timer is not None:
                self._auto_sync_timer.stop()
            return
        # 已连接的臂需等到反馈到达；未连接的臂跳过等待
        arm1_ready = (not self.worker.arm1_connected) or self._last_joint_states is not None
        arm2_ready = (not self.worker.arm2_connected) or self._last_joint_states2 is not None
        if not (arm1_ready and arm2_ready):
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
            if self._last_joint_states is not None:
                self.joint_panel.update_feedback(self._last_joint_states)
            if self._last_joint_states2 is not None:
                self.joint_panel.update_feedback_arm2(self._last_joint_states2)
            self.joint_panel.sync_btn.click()
            self._append_log("已自动同步当前位置")
        except Exception as e:
            logger.warning(f"[MainWindow] 自动同步当前位置失败: {e}")

    def _on_disabled_disconnect_vr(self, enabled: bool):
        """失能（含急停、断开连接）时：若 VR 已连接则自动断开 VR 链路。"""
        if enabled:
            return
        if self.vr_panel._receiver is not None:
            self._append_log("[VR] 电机失能，自动断开 VR 连接")
            try:
                self.vr_panel._stop_link()
            except Exception as e:
                logger.warning(f"[MainWindow] 失能联动断开 VR 失败: {e}")

    def _on_connected_for_gamepad(self, connected: bool):
        if connected and self.worker.arm is not None:
            self.gamepad_panel.set_arm(self.worker.arm)
            self.vr_panel.set_arm(self.worker.arm)
        elif not connected:
            self.gamepad_panel.set_arm(None)
            self.vr_panel.set_arm(None)
        # 臂2（can1）：左手柄控制；未连接时传 None
        self.vr_panel.set_arm2(self.worker.arm2 if connected else None)
        # 连接后把 GUI 当前力矩限制值同步到 SDK（双臂）
        if connected:
            self.joint_panel._emit_torque_limit()

    def _on_connect(self, can_name: str, connect_kwargs: dict):
        self.worker.submit_command("connect", can_name, **connect_kwargs)

    def _on_joints_updated2(self, joint_states):
        """臂2 关节反馈：更新关节控制/轨迹/示教/诊断/夹爪的臂2显示"""
        self._last_joint_states2 = joint_states

        self.trajectory_panel.update_current_positions_arm2(joint_states)
        self.teaching_panel.update_positions_arm2(joint_states)

        # 标定面板数据喂入（臂2，不受 UI 节流限制）
        positions6 = joint_states.to_list(include_gripper=False)
        self.teaching_panel.calibration_panel.feed_positions(2, positions6)

        # 提取臂2夹爪角度（弧度）并从关节角实时 FK 计算臂2末端位姿
        positions = joint_states.to_list(include_gripper=True)
        if len(positions) >= 7:
            self._last_gripper_angle_rad2 = float(positions[6])
        else:
            self._last_gripper_angle_rad2 = None

        if self._kin is not None:
            try:
                self._last_end_pose2 = self._kin.forward_kinematics(positions[:6])
            except Exception as e:
                logger.warning(f"[MainWindow] 臂2 正运动学计算失败: {e}")

        now = time.monotonic()
        if now - self._last_ui_update_time2 < self.UI_UPDATE_INTERVAL_S:
            return
        self._last_ui_update_time2 = now

        self.joint_panel.update_feedback_arm2(joint_states)
        self.gripper_panel.update_feedback_arm2(
            joint_states, self._last_effort_states2)
        self.diagnostics_panel.update_motor_states_arm2(
            joint_states, None, self._last_effort_states2
        )
        # 同步更新臂2 末端位姿显示（使用 FK 计算结果）
        if hasattr(self, 'kinematics_display') and self.kinematics_display:
            self.kinematics_display.update_end_pose_arm2(
                self._last_end_pose2, self._last_gripper_angle_rad2,
                self._get_gripper_torque2()
            )

    def _on_joints_updated(self, joint_states):
        self._last_joint_states = joint_states

        self.teaching_panel.update_positions(joint_states)
        self.trajectory_panel.update_current_positions(joint_states)

        # 标定面板数据喂入（臂1，不受 UI 节流限制）
        positions6 = joint_states.to_list(include_gripper=False)
        self.teaching_panel.calibration_panel.feed_positions(1, positions6)

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
        efforts = effort_states.to_list(include_gripper=False)
        self.teaching_panel.calibration_panel.feed_efforts(1, efforts)

    def _on_efforts_updated2(self, effort_states):
        self._last_effort_states2 = effort_states
        efforts = effort_states.to_list(include_gripper=False)
        self.teaching_panel.calibration_panel.feed_efforts(2, efforts)

    def _get_current_end_pose(self):
        """供摄像头组件获取最新末端位姿和夹爪角度（夹爪为弧度）"""
        end_pose = getattr(self, "_last_end_pose", None)
        gripper_rad = getattr(self, "_last_gripper_angle_rad", None)
        return end_pose, gripper_rad

    def _get_current_end_pose2(self):
        """供摄像头组件获取臂2 最新末端位姿和夹爪角度（夹爪为弧度）"""
        end_pose = getattr(self, "_last_end_pose2", None)
        gripper_rad = getattr(self, "_last_gripper_angle_rad2", None)
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

    def _get_gripper_torque2(self):
        """获取臂2 最新夹爪力矩 (Nm)，读不到时返回 None"""
        if not self._last_effort_states2:
            return None
        try:
            torques = self._last_effort_states2.to_list(include_gripper=True)
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
        if self._perf_dialog is not None:
            self._perf_dialog.close()

        # 清理摄像头资源
        if hasattr(self, 'camera_widget') and self.camera_widget:
            self.camera_widget.cleanup()

        # 关闭底盘/升降面板（安全停机 + 断开串口）
        if hasattr(self, '_chassis_win') and self._chassis_win is not None:
            try:
                self._chassis_win.close()
            except Exception:
                pass


        # 清理变量
        self._last_joint_states = None
        self._last_joint_states2 = None
        self._last_effort_states = None
        self._last_velocity_states = None
        self._last_end_pose = None
        self._last_end_pose2 = None
        self._last_gripper_angle_rad2 = None
        self._last_status = None
        self._last_motor_feedback = None

        self._append_log("已清理资源")

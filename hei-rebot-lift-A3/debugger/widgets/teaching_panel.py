"""示教面板：零力矩模式 + 轨迹录制/回放"""

import math
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QPushButton, QGroupBox,
    QListWidget, QFileDialog, QCheckBox, QTabWidget,
)
from PyQt6.QtCore import pyqtSignal, QTimer

from debugger.backend.trajectory_recorder import TrajectoryRecorder
from debugger.widgets.calibration_panel import CalibrationPanel
from debugger.utils.i18n import tr
from debugger.utils.theme_manager import ThemeManager
from debugger.utils.style import SCENE_COLORS

logger = logging.getLogger("debugger.teaching")


class TeachingPanel(QWidget):
    """示教模式：零力矩拖动 + 轨迹录制回放"""

    zero_torque_requested = pyqtSignal(bool)
    zero_torque_gravity_requested = pyqtSignal(bool)
    move_j_requested = pyqtSignal(list, float)  # positions, duration
    joint_ctrl_requested = pyqtSignal(list, object)  # (臂1 positions, 臂2 positions or None)
    trajectory_playback_requested = pyqtSignal(list, object)  # (臂1 SDK点列, 臂2 SDK点列 or None)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.recorder = TrajectoryRecorder(sample_rate_hz=10.0)
        self._zero_torque_active = False
        self._current_positions = [0.0] * 7
        self._current_positions2 = [0.0] * 7
        self._has_arm2 = False  # 是否收到过臂2反馈
        self._rec_complete_info = None  # (n, t) after recording stops, else None
        self._init_ui()

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        teach_page = QWidget()
        layout = QVBoxLayout(teach_page)
        layout.setContentsMargins(4, 4, 4, 4)

        self.zt_group = QGroupBox(tr("teach.zt_group"))
        zt_layout = QVBoxLayout()

        self.gravity_check = QCheckBox(tr("teach.gravity"))
        self.gravity_check.setChecked(True)
        zt_layout.addWidget(self.gravity_check)

        self.zt_toggle_btn = QPushButton(tr("teach.zt_on"))
        self.zt_toggle_btn.setObjectName("enableBtn")
        self.zt_toggle_btn.clicked.connect(self._toggle_zero_torque)
        zt_layout.addWidget(self.zt_toggle_btn)

        self.zt_group.setLayout(zt_layout)
        layout.addWidget(self.zt_group)

        self.rec_group = QGroupBox(tr("teach.rec_group"))
        rec_layout = QVBoxLayout()

        rate_layout = QHBoxLayout()
        self.rate_label = QLabel(tr("teach.sample_rate"))
        rate_layout.addWidget(self.rate_label)
        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(1.0, 50.0)
        self.rate_spin.setValue(10.0)
        self.rate_spin.setSuffix(" Hz")
        rate_layout.addWidget(self.rate_spin)
        rate_layout.addStretch()
        rec_layout.addLayout(rate_layout)

        btn_layout = QHBoxLayout()
        self.rec_btn = QPushButton(tr("teach.start_rec"))
        self.rec_btn.clicked.connect(self._toggle_recording)
        btn_layout.addWidget(self.rec_btn)

        self.rec_status = QLabel(tr("teach.not_rec"))
        btn_layout.addWidget(self.rec_status)
        btn_layout.addStretch()
        rec_layout.addLayout(btn_layout)

        self.rec_group.setLayout(rec_layout)
        layout.addWidget(self.rec_group)

        self.traj_group = QGroupBox(tr("teach.traj_group"))
        traj_layout = QVBoxLayout()

        self.traj_list = QListWidget()
        traj_layout.addWidget(self.traj_list)

        traj_btn_layout = QHBoxLayout()
        self.play_btn = QPushButton(tr("teach.play"))
        self.play_btn.clicked.connect(self._playback)
        traj_btn_layout.addWidget(self.play_btn)

        self.save_btn = QPushButton(tr("teach.save"))
        self.save_btn.clicked.connect(self._save_trajectory)
        traj_btn_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton(tr("teach.load"))
        self.load_btn.clicked.connect(self._load_trajectory)
        traj_btn_layout.addWidget(self.load_btn)

        self.del_btn = QPushButton(tr("teach.delete"))
        self.del_btn.clicked.connect(self._delete_trajectory)
        traj_btn_layout.addWidget(self.del_btn)

        traj_layout.addLayout(traj_btn_layout)
        self.traj_group.setLayout(traj_layout)
        layout.addWidget(self.traj_group)

        self._rec_timer = QTimer(self)
        self._rec_timer.timeout.connect(self._sample_position)

        self._tabs.addTab(teach_page, tr("teach.tab_teaching"))
        self.calibration_panel = CalibrationPanel()
        self._tabs.addTab(self.calibration_panel, tr("teach.tab_calibration"))
        outer.addWidget(self._tabs)

    def retranslate_ui(self):
        self._tabs.setTabText(0, tr("teach.tab_teaching"))
        self._tabs.setTabText(1, tr("teach.tab_calibration"))
        self.calibration_panel.retranslate_ui()
        self.zt_group.setTitle(tr("teach.zt_group"))
        self.gravity_check.setText(tr("teach.gravity"))
        if self._zero_torque_active:
            self.zt_toggle_btn.setText(tr("teach.zt_off"))
        else:
            self.zt_toggle_btn.setText(tr("teach.zt_on"))
        self.rec_group.setTitle(tr("teach.rec_group"))
        self.rate_label.setText(tr("teach.sample_rate"))
        if self.recorder.is_recording:
            self.rec_btn.setText(tr("teach.stop_rec"))
        else:
            self.rec_btn.setText(tr("teach.start_rec"))
        self._refresh_rec_status_text()
        self.traj_group.setTitle(tr("teach.traj_group"))
        self.play_btn.setText(tr("teach.play"))
        self.save_btn.setText(tr("teach.save"))
        self.load_btn.setText(tr("teach.load"))
        self.del_btn.setText(tr("teach.delete"))

    def _refresh_rec_status_text(self):
        if self.recorder.is_recording:
            if self.recorder.current_trajectory:
                n = self.recorder.current_trajectory.num_points
                if n > 0:
                    self.rec_status.setText(tr("teach.recording_n", n=n))
                else:
                    self.rec_status.setText(tr("teach.recording"))
            else:
                self.rec_status.setText(tr("teach.recording"))
        elif self._rec_complete_info is not None:
            n, t = self._rec_complete_info
            self.rec_status.setText(tr("teach.rec_done", n=n, t=t))
        else:
            self.rec_status.setText(tr("teach.not_rec"))

    def _toggle_zero_torque(self):
        self._zero_torque_active = not self._zero_torque_active

        if self._zero_torque_active:
            self.zt_toggle_btn.setText(tr("teach.zt_off"))
            sc = SCENE_COLORS[ThemeManager.instance().theme]
            self.zt_toggle_btn.setStyleSheet(
                f"background-color: {sc['zt_active_bg']}; color: white; border-radius: 6px;"
            )
            if self.gravity_check.isChecked():
                self.zero_torque_gravity_requested.emit(True)
            else:
                self.zero_torque_requested.emit(True)
        else:
            self.zt_toggle_btn.setText(tr("teach.zt_on"))
            self.zt_toggle_btn.setStyleSheet("")
            self.zt_toggle_btn.setObjectName("enableBtn")
            if self.gravity_check.isChecked():
                self.zero_torque_gravity_requested.emit(False)
            else:
                self.zero_torque_requested.emit(False)
                # 双臂发送完整7个关节位置（包含L7夹爪），避免L7回到旧目标位置
                pos2 = self._current_positions2[:7] if self._has_arm2 else None
                logger.info(f"退出零力矩: 发送joint_ctrl, positions={self._current_positions[:7]}")
                self.joint_ctrl_requested.emit(self._current_positions[:7], pos2)

    def _toggle_recording(self):
        if self.recorder.is_recording:
            traj = self.recorder.stop_recording()
            self._rec_timer.stop()
            self.rec_btn.setText(tr("teach.start_rec"))
            self.rec_btn.setStyleSheet("")
            if traj:
                self._rec_complete_info = (traj.num_points, traj.duration)
                self.rec_status.setText(
                    tr("teach.rec_done", n=traj.num_points, t=traj.duration)
                )
                self.traj_list.addItem(
                    f"{traj.name} ({traj.num_points}点, {traj.duration:.1f}s)"
                )
        else:
            self._rec_complete_info = None
            self.recorder.sample_rate_hz = self.rate_spin.value()
            self.recorder.start_recording()
            interval_ms = int(1000.0 / self.rate_spin.value())
            self._rec_timer.start(interval_ms)
            self.rec_btn.setText(tr("teach.stop_rec"))
            sc = SCENE_COLORS[ThemeManager.instance().theme]
            self.rec_btn.setStyleSheet(
                f"background-color: {sc['recording_bg']}; color: white; border-radius: 6px;"
            )
            self.rec_status.setText(tr("teach.recording"))

    def _sample_position(self):
        """双臂同时采样（臂2无反馈时 positions2 为 None）"""
        pos2 = self._current_positions2 if self._has_arm2 else None
        self.recorder.add_sample(self._current_positions, pos2)
        if self.recorder.current_trajectory:
            n = self.recorder.current_trajectory.num_points
            self.rec_status.setText(tr("teach.recording_n", n=n))

    def update_positions(self, joint_states):
        self._current_positions = joint_states.to_list(include_gripper=True)

    def update_positions_arm2(self, joint_states):
        self._current_positions2 = joint_states.to_list(include_gripper=True)
        self._has_arm2 = True

    @staticmethod
    def _plan_arm_spline(traj_points, position_getter):
        """用三次样条把 10Hz 录制点平滑成 200Hz 轨迹，时间严格对齐录制间隔。

        position_getter: 从录制点取该臂位置列表的函数。
        """
        from el_a3_sdk.trajectory import (
            TrajectoryPoint as SDKTrajectoryPoint,
            CubicSplinePlanner,
        )

        control_dt = 0.005  # 200Hz
        has_gripper = len(position_getter(traj_points[0])) >= 7
        num_joints = 7 if has_gripper else 6

        n_points = len(traj_points)
        waypoints = [position_getter(pt)[:num_joints] for pt in traj_points]
        durations = []
        for i in range(n_points - 1):
            dt = traj_points[i + 1].timestamp - traj_points[i].timestamp
            durations.append(max(1e-6, dt))

        spline_points = CubicSplinePlanner.plan_waypoints(
            waypoints, durations, dt=control_dt,
            start_velocity=[0.0] * num_joints,
            end_velocity=[0.0] * num_joints,
        )

        sdk_points = [
            SDKTrajectoryPoint(
                time=pt.time,
                positions=list(pt.positions),
                velocities=list(pt.velocities),
                accelerations=list(pt.accelerations),
            )
            for pt in spline_points
        ]

        # 保证最后一个点速度为 0，避免结束后继续微漂
        if sdk_points:
            last = sdk_points[-1]
            last.velocities = [0.0] * num_joints
            last.accelerations = [0.0] * num_joints
        else:
            sdk_points.append(SDKTrajectoryPoint(
                time=0.0,
                positions=position_getter(traj_points[-1])[:num_joints],
                velocities=[0.0] * num_joints,
                accelerations=[0.0] * num_joints,
            ))
        return sdk_points

    def _playback(self):
        idx = self.traj_list.currentRow()
        if idx < 0 or idx >= len(self.recorder.trajectories):
            return
        traj = self.recorder.trajectories[idx]
        if not traj.points:
            return

        # 臂1 轨迹（必有）
        sdk_points = self._plan_arm_spline(traj.points, lambda pt: pt.positions)

        # 臂2 轨迹（录制时含臂2数据才回放臂2；两臂独立规划、同时开始）
        sdk_points2 = None
        if traj.points[0].positions2 is not None:
            sdk_points2 = self._plan_arm_spline(traj.points, lambda pt: pt.positions2)

        self.trajectory_playback_requested.emit(sdk_points, sdk_points2)

    def _save_trajectory(self):
        idx = self.traj_list.currentRow()
        if idx < 0 or idx >= len(self.recorder.trajectories):
            return
        traj = self.recorder.trajectories[idx]
        filepath, _ = QFileDialog.getSaveFileName(
            self, tr("teach.save_title"), f"{traj.name}.json", "JSON Files (*.json)"
        )
        if filepath:
            self.recorder.save_trajectory(traj, filepath)

    def _load_trajectory(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, tr("teach.load_title"), "", "JSON Files (*.json)"
        )
        if filepath:
            traj = self.recorder.load_trajectory(filepath)
            self.traj_list.addItem(
                f"{traj.name} ({traj.num_points}点, {traj.duration:.1f}s)"
            )

    def _delete_trajectory(self):
        idx = self.traj_list.currentRow()
        if idx < 0 or idx >= len(self.recorder.trajectories):
            return
        self.recorder.trajectories.pop(idx)
        self.traj_list.takeItem(idx)

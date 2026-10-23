"""夹爪控制面板（双臂）"""

import math
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QDoubleSpinBox, QPushButton, QGroupBox,
)
from PyQt6.QtCore import pyqtSignal, Qt

from debugger.utils.i18n import tr
from debugger.utils.theme_manager import ThemeManager
from debugger.utils.style import SCENE_COLORS


class GripperPanel(QWidget):
    """夹爪控制：双臂角度滑块、全开/全关、设零、力矩限制（设置对双臂同时生效）"""

    gripper_command = pyqtSignal(float, float, float, float)  # (angle1 rad, angle2 rad, torque_max, torque_min)
    set_zero_requested = pyqtSignal()
    torque_limit_changed = pyqtSignal(float, float)  # (max, min) in Nm

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._init_ui()

    def _scene(self):
        return SCENE_COLORS[ThemeManager.instance().theme]

    def _apply_feedback_label_styles(self):
        sc = self._scene()
        for fb_label in self._fb_labels:
            fb_label.setStyleSheet(
                f"color: {sc['accent']}; font-weight: bold;")
        for torque_label in self._torque_labels:
            torque_label.setStyleSheet(f"color: {sc['warning']};")

    def _build_arm_row(self, arm_idx):
        """构建一臂的夹爪角度控制行，返回 (slider, spin)"""
        row = QHBoxLayout()
        arm_label = QLabel(tr("jc.arm1") if arm_idx == 0 else tr("jc.arm2"))
        arm_label.setStyleSheet("font-weight: bold;")
        arm_label.setFixedWidth(32)
        row.addWidget(arm_label)
        self._arm_labels.append(arm_label)

        lbl_angle = QLabel(tr("grip.angle"))
        row.addWidget(lbl_angle)
        self._lbl_angles.append(lbl_angle)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(-900, 900)
        slider.setValue(0)
        slider.valueChanged.connect(
            lambda v, a=arm_idx: self._on_slider_changed(a, v))
        row.addWidget(slider)

        spin = QDoubleSpinBox()
        spin.setRange(-90.0, 90.0)
        spin.setDecimals(1)
        spin.setSuffix("°")
        spin.setFixedWidth(100)
        spin.valueChanged.connect(
            lambda v, a=arm_idx: self._on_spin_changed(a, v))
        row.addWidget(spin)
        return row, slider, spin

    def _build_feedback_row(self, arm_idx):
        """构建一臂的夹爪反馈行（实际角度 + 力矩）"""
        row = QHBoxLayout()
        arm_label = QLabel(tr("jc.arm1") if arm_idx == 0 else tr("jc.arm2"))
        arm_label.setStyleSheet("font-weight: bold;")
        arm_label.setFixedWidth(32)
        row.addWidget(arm_label)
        self._fb_arm_labels.append(arm_label)

        lbl_actual = QLabel(tr("grip.actual"))
        row.addWidget(lbl_actual)
        self._lbl_actuals.append(lbl_actual)
        fb_label = QLabel("0.0°")
        row.addWidget(fb_label)
        self._fb_labels.append(fb_label)

        row.addSpacing(20)
        lbl_torque = QLabel(tr("grip.torque"))
        row.addWidget(lbl_torque)
        self._lbl_torques.append(lbl_torque)
        torque_label = QLabel("0.00 Nm")
        row.addWidget(torque_label)
        self._torque_labels.append(torque_label)
        row.addStretch()
        return row

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._arm_labels = []
        self._fb_arm_labels = []
        self._lbl_angles = []
        self._lbl_actuals = []
        self._lbl_torques = []
        self._fb_labels = []
        self._torque_labels = []

        self.group = QGroupBox()
        g_layout = QVBoxLayout()

        # 双臂角度控制行
        row1, self.angle_slider, self.angle_spin = self._build_arm_row(0)
        g_layout.addLayout(row1)
        row2, self.angle_slider2, self.angle_spin2 = self._build_arm_row(1)
        g_layout.addLayout(row2)

        # 双臂反馈行
        g_layout.addLayout(self._build_feedback_row(0))
        g_layout.addLayout(self._build_feedback_row(1))

        # 力矩限制输入（双臂共用，默认最大 0.5 Nm / 最小 -0.5 Nm）
        limit_layout = QHBoxLayout()
        self._lbl_limit = QLabel()
        limit_layout.addWidget(self._lbl_limit)
        self.torque_max_spin = QDoubleSpinBox()
        self.torque_max_spin.setRange(-10.0, 10.0)
        self.torque_max_spin.setDecimals(2)
        self.torque_max_spin.setSingleStep(0.05)
        self.torque_max_spin.setValue(0.5)
        self.torque_max_spin.setSuffix(" Nm")
        self.torque_max_spin.setFixedWidth(90)
        self.torque_max_spin.valueChanged.connect(self._emit_torque_limit)
        limit_layout.addWidget(self.torque_max_spin)
        limit_layout.addSpacing(8)
        self._lbl_limit_min = QLabel()
        limit_layout.addWidget(self._lbl_limit_min)
        self.torque_min_spin = QDoubleSpinBox()
        self.torque_min_spin.setRange(-10.0, 10.0)
        self.torque_min_spin.setDecimals(2)
        self.torque_min_spin.setSingleStep(0.05)
        self.torque_min_spin.setValue(-0.5)
        self.torque_min_spin.setSuffix(" Nm")
        self.torque_min_spin.setFixedWidth(90)
        self.torque_min_spin.valueChanged.connect(self._emit_torque_limit)
        limit_layout.addWidget(self.torque_min_spin)
        limit_layout.addStretch()
        g_layout.addLayout(limit_layout)

        btn_layout = QHBoxLayout()
        self.send_btn = QPushButton()
        self.send_btn.setObjectName("enableBtn")
        self.send_btn.clicked.connect(self._send_command)
        btn_layout.addWidget(self.send_btn)

        self.open_btn = QPushButton()
        self.open_btn.clicked.connect(lambda: self._set_angle(-90.0))
        btn_layout.addWidget(self.open_btn)

        self.close_btn = QPushButton()
        self.close_btn.clicked.connect(lambda: self._set_angle(0.0))
        btn_layout.addWidget(self.close_btn)

        self.zero_btn = QPushButton()
        self.zero_btn.clicked.connect(self.set_zero_requested.emit)
        btn_layout.addWidget(self.zero_btn)

        btn_layout.addStretch()
        g_layout.addLayout(btn_layout)

        self.group.setLayout(g_layout)
        layout.addWidget(self.group)
        layout.addStretch()

        self.retranslate_ui()

    def retranslate_ui(self):
        self.group.setTitle(tr("grip.group"))
        self._arm_labels[0].setText(tr("jc.arm1"))
        self._arm_labels[1].setText(tr("jc.arm2"))
        self._fb_arm_labels[0].setText(tr("jc.arm1"))
        self._fb_arm_labels[1].setText(tr("jc.arm2"))
        for lbl in self._lbl_angles:
            lbl.setText(tr("grip.angle"))
        for lbl in self._lbl_actuals:
            lbl.setText(tr("grip.actual"))
        for lbl in self._lbl_torques:
            lbl.setText(tr("grip.torque"))
        self._lbl_limit.setText(tr("grip.torque_limit"))
        self._lbl_limit_min.setText(tr("grip.torque_min"))
        self.send_btn.setText(tr("grip.send"))
        self.open_btn.setText(tr("grip.open"))
        self.close_btn.setText(tr("grip.close"))
        self.zero_btn.setText(tr("grip.set_zero"))
        self._apply_feedback_label_styles()

    def _emit_torque_limit(self):
        self.torque_limit_changed.emit(
            self.torque_max_spin.value(), self.torque_min_spin.value()
        )

    def _widgets(self, arm_idx):
        if arm_idx == 0:
            return self.angle_slider, self.angle_spin
        return self.angle_slider2, self.angle_spin2

    def _on_slider_changed(self, arm_idx, val):
        if self._updating:
            return
        _, spin = self._widgets(arm_idx)
        self._updating = True
        spin.setValue(val / 10.0)
        self._updating = False

    def _on_spin_changed(self, arm_idx, val):
        if self._updating:
            return
        slider, _ = self._widgets(arm_idx)
        self._updating = True
        slider.setValue(int(val * 10))
        self._updating = False

    def _set_angle(self, deg):
        """全开/全关：双臂同时设置并发送"""
        self._updating = True
        for slider, spin in (self._widgets(0), self._widgets(1)):
            spin.setValue(deg)
            slider.setValue(int(deg * 10))
        self._updating = False
        self._send_command()

    def _send_command(self):
        angle1_rad = math.radians(self.angle_spin.value())
        angle2_rad = math.radians(self.angle_spin2.value())
        self.gripper_command.emit(
            angle1_rad, angle2_rad,
            self.torque_max_spin.value(), self.torque_min_spin.value()
        )

    def _update_feedback_arm(self, arm_idx, joint_states, effort_states=None):
        positions = joint_states.to_list(include_gripper=True)
        if len(positions) >= 7:
            deg = math.degrees(positions[6])
            self._fb_labels[arm_idx].setText(f"{deg:.1f}°")
        if effort_states:
            torques = effort_states.to_list(include_gripper=True)
            if len(torques) >= 7:
                self._torque_labels[arm_idx].setText(f"{torques[6]:.2f} Nm")

    def update_feedback(self, joint_states, effort_states=None):
        """臂1 夹爪反馈"""
        self._update_feedback_arm(0, joint_states, effort_states)

    def update_feedback_arm2(self, joint_states, effort_states=None):
        """臂2 夹爪反馈"""
        self._update_feedback_arm(1, joint_states, effort_states)

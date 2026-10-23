"""关节控制面板：双臂 7 关节滑块 + SpinBox 控制"""

import math
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QDoubleSpinBox, QPushButton, QGroupBox,
    QGridLayout, QCheckBox,
)
from PyQt6.QtCore import pyqtSignal, Qt

from debugger.utils.style import JOINT_COLORS, SCENE_COLORS
from debugger.utils.theme_manager import ThemeManager
from debugger.utils.i18n import tr

JOINT_LIMITS_DEG = {
    1: (-160.0, 160.0),
    2: (0.0, 210.0),
    3: (-230.0, 0.0),
    4: (-90.0, 90.0),
    5: (-90.0, 90.0),
    6: (-90.0, 90.0),
    7: (-90.0, 90.0),
}

JOINT_NAMES = ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]

SLIDER_RESOLUTION = 1000


class JointControlPanel(QWidget):
    """双臂 7 关节手动控制面板"""

    joint_command = pyqtSignal(list, list)  # (臂1 7个弧度值, 臂2 7个弧度值)
    go_zero_requested = pyqtSignal()
    torque_limit_changed = pyqtSignal(float, float, bool)  # (L1-L3 限值, L4-L6 限值, 启用)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating_from_feedback = False
        self._enabled = False
        # 每臂一组控件: {sliders, spinboxes, pos/vel/torque/temp labels, name_labels}
        self._arms = []
        self._header_labels = []
        self._arm_title_labels = []
        self._init_ui()

    # ==================================================================
    # UI 构建
    # ==================================================================

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(1)

        for arm_idx in range(2):
            title = QLabel(tr("jc.arm1" if arm_idx == 0 else "jc.arm2"))
            sc = SCENE_COLORS[ThemeManager.instance().theme]
            title.setStyleSheet(
                f"font-weight: bold; color: {sc['header_text']};")
            title.setFixedHeight(18)
            layout.addWidget(title)
            self._arm_title_labels.append(title)

            grid = QGridLayout()
            grid.setSpacing(2)
            grid.setContentsMargins(0, 0, 0, 0)

            headers = [tr("jc.header_joint"), tr("jc.header_ctrl"), tr("jc.header_target"),
                     tr("jc.header_pos"), tr("jc.header_vel"), tr("jc.header_torque"),
                     "", tr("jc.header_temp")]
            arm_headers = []
            for col, h in enumerate(headers):
                lbl = QLabel(h)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setStyleSheet(f"font-weight: bold; color: {sc['header_text']};")
                lbl.setFixedHeight(20)
                grid.addWidget(lbl, 0, col)
                arm_headers.append(lbl)
            self._header_labels.append(arm_headers)

            widgets = self._build_joint_rows(grid, arm_idx)
            self._arms.append(widgets)

            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 5)
            grid.setColumnStretch(2, 0)
            grid.setColumnStretch(3, 0)
            grid.setColumnStretch(4, 0)
            grid.setColumnStretch(5, 0)
            layout.addLayout(grid)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 2, 0, 0)
        self.zero_btn = QPushButton(tr("jc.go_zero"))
        self.zero_btn.setFixedHeight(26)
        self.zero_btn.clicked.connect(self._on_go_zero)
        btn_layout.addWidget(self.zero_btn)

        self.sync_btn = QPushButton(tr("jc.sync_pos"))
        self.sync_btn.setFixedHeight(26)
        self.sync_btn.clicked.connect(self._sync_to_feedback)
        btn_layout.addWidget(self.sync_btn)

        btn_layout.addStretch()
        self.send_btn = QPushButton(tr("jc.send"))
        self.send_btn.setObjectName("enableBtn")
        self.send_btn.setFixedHeight(26)
        self.send_btn.clicked.connect(self._emit_command)
        btn_layout.addWidget(self.send_btn)

        layout.addLayout(btn_layout)

        # 关节软力矩限制（双臂共用，触限冻结 + 滞回解锁）
        torque_layout = QHBoxLayout()
        torque_layout.setContentsMargins(0, 2, 0, 0)
        self.torque_enable_chk = QCheckBox(tr("jc.torque_limit"))
        self.torque_enable_chk.setChecked(True)
        self.torque_enable_chk.toggled.connect(self._emit_torque_limit)
        torque_layout.addWidget(self.torque_enable_chk)

        self.torque_l13_label = QLabel(tr("jc.torque_l13"))
        torque_layout.addWidget(self.torque_l13_label)
        self.torque_l13_spin = QDoubleSpinBox()
        self.torque_l13_spin.setRange(0.0, 14.0)
        self.torque_l13_spin.setDecimals(1)
        self.torque_l13_spin.setSingleStep(0.5)
        self.torque_l13_spin.setValue(13.0)
        self.torque_l13_spin.valueChanged.connect(self._emit_torque_limit)
        torque_layout.addWidget(self.torque_l13_spin)

        self.torque_l46_label = QLabel(tr("jc.torque_l46"))
        torque_layout.addWidget(self.torque_l46_label)
        self.torque_l46_spin = QDoubleSpinBox()
        self.torque_l46_spin.setRange(0.0, 6.0)
        self.torque_l46_spin.setDecimals(1)
        self.torque_l46_spin.setSingleStep(0.5)
        self.torque_l46_spin.setValue(5.0)
        self.torque_l46_spin.valueChanged.connect(self._emit_torque_limit)
        torque_layout.addWidget(self.torque_l46_spin)
        torque_layout.addStretch()

        layout.addLayout(torque_layout)
        layout.addStretch()

    def _emit_torque_limit(self, *args):
        self.torque_limit_changed.emit(
            self.torque_l13_spin.value(),
            self.torque_l46_spin.value(),
            self.torque_enable_chk.isChecked())

    def _build_joint_rows(self, grid: QGridLayout, arm_idx: int) -> dict:
        """在 grid 中为一臂构建 7 行关节控件，返回控件字典"""
        sliders = []
        spinboxes = []
        pos_labels = []
        vel_labels = []
        torque_labels = []
        temp_labels = []
        name_labels = []

        for i in range(7):
            row = i + 1
            joint_id = i + 1
            lo, hi = JOINT_LIMITS_DEG[joint_id]

            name_label = QLabel(JOINT_NAMES[i])
            name_label.setStyleSheet(f"color: {JOINT_COLORS[i]}; font-weight: bold;")
            name_label.setFixedWidth(28)
            name_label.setFixedHeight(22)
            name_label.setToolTip(tr(f"jc.tooltip_L{i+1}"))
            grid.addWidget(name_label, row, 0)
            name_labels.append(name_label)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, SLIDER_RESOLUTION)
            mid = int((0.0 - lo) / (hi - lo) * SLIDER_RESOLUTION) if hi != lo else SLIDER_RESOLUTION // 2
            slider.setValue(max(0, min(SLIDER_RESOLUTION, mid)))
            slider.setFixedHeight(20)
            slider.valueChanged.connect(
                lambda v, a=arm_idx, idx=i: self._on_slider_changed(a, idx, v))
            sliders.append(slider)
            grid.addWidget(slider, row, 1)

            spinbox = QDoubleSpinBox()
            spinbox.setRange(lo, hi)
            spinbox.setDecimals(1)
            spinbox.setSingleStep(1.0)
            spinbox.setSuffix("°")
            spinbox.setFixedWidth(85)
            spinbox.setFixedHeight(24)
            spinbox.setValue(0.0 if lo <= 0.0 <= hi else lo)
            spinbox.valueChanged.connect(
                lambda v, a=arm_idx, idx=i: self._on_spinbox_changed(a, idx, v))
            spinboxes.append(spinbox)
            grid.addWidget(spinbox, row, 2)

            pos_label = QLabel("0.0°")
            pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pos_label.setFixedWidth(60)
            pos_label.setFixedHeight(24)
            pos_label.setStyleSheet(f"color: {JOINT_COLORS[i]};")
            pos_labels.append(pos_label)
            grid.addWidget(pos_label, row, 3)

            vel_label = QLabel("0.0")
            vel_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vel_label.setFixedWidth(55)
            vel_label.setFixedHeight(24)
            vel_label.setStyleSheet("color: #888;")
            vel_labels.append(vel_label)
            grid.addWidget(vel_label, row, 4)

            torque_label = QLabel("0.0")
            torque_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            torque_label.setFixedWidth(55)
            torque_label.setFixedHeight(24)
            torque_label.setStyleSheet("color: #888;")
            torque_labels.append(torque_label)
            grid.addWidget(torque_label, row, 5)

            grid.addWidget(QLabel("Nm"), row, 6)

            temp_label = QLabel("--")
            temp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            temp_label.setFixedWidth(60)
            temp_label.setFixedHeight(24)
            temp_label.setStyleSheet("color: #888;")
            temp_labels.append(temp_label)
            grid.addWidget(temp_label, row, 7)

        return {
            "sliders": sliders,
            "spinboxes": spinboxes,
            "pos_labels": pos_labels,
            "vel_labels": vel_labels,
            "torque_labels": torque_labels,
            "temp_labels": temp_labels,
            "name_labels": name_labels,
        }

    # ==================================================================
    # retranslate
    # ==================================================================

    def retranslate_ui(self):
        sc = SCENE_COLORS[ThemeManager.instance().theme]
        headers = [tr("jc.header_joint"), tr("jc.header_ctrl"), tr("jc.header_target"),
                   tr("jc.header_pos"), tr("jc.header_vel"), tr("jc.header_torque"),
                   "", tr("jc.header_temp")]
        for arm_headers in self._header_labels:
            for i, lbl in enumerate(arm_headers):
                lbl.setText(headers[i])
                lbl.setStyleSheet(f"font-weight: bold; color: {sc['header_text']};")
        self._arm_title_labels[0].setText(tr("jc.arm1"))
        self._arm_title_labels[1].setText(tr("jc.arm2"))
        for lbl in self._arm_title_labels:
            lbl.setStyleSheet(f"font-weight: bold; color: {sc['header_text']};")
        self.zero_btn.setText(tr("jc.go_zero"))
        self.sync_btn.setText(tr("jc.sync_pos"))
        self.send_btn.setText(tr("jc.send"))
        self.torque_enable_chk.setText(tr("jc.torque_limit"))
        self.torque_l13_label.setText(tr("jc.torque_l13"))
        self.torque_l46_label.setText(tr("jc.torque_l46"))
        tooltips = [tr(f"jc.tooltip_L{i+1}") for i in range(7)]
        for widgets in self._arms:
            for i, lbl in enumerate(widgets["name_labels"]):
                lbl.setToolTip(tooltips[i])

    # ==================================================================
    # 滑块 <-> SpinBox 联动
    # ==================================================================

    def _slider_to_deg(self, joint_idx, slider_val):
        lo, hi = JOINT_LIMITS_DEG[joint_idx + 1]
        return lo + (hi - lo) * slider_val / SLIDER_RESOLUTION

    def _deg_to_slider(self, joint_idx, deg_val):
        lo, hi = JOINT_LIMITS_DEG[joint_idx + 1]
        if hi == lo:
            return SLIDER_RESOLUTION // 2
        return int((deg_val - lo) / (hi - lo) * SLIDER_RESOLUTION)

    def _on_slider_changed(self, arm_idx, idx, val):
        if self._updating_from_feedback:
            return
        deg = self._slider_to_deg(idx, val)
        self._updating_from_feedback = True
        self._arms[arm_idx]["spinboxes"][idx].setValue(deg)
        self._updating_from_feedback = False

    def _on_spinbox_changed(self, arm_idx, idx, val):
        if self._updating_from_feedback:
            return
        self._updating_from_feedback = True
        self._arms[arm_idx]["sliders"][idx].setValue(self._deg_to_slider(idx, val))
        self._updating_from_feedback = False

    # ==================================================================
    # 命令
    # ==================================================================

    def _get_arm_positions_rad(self, arm_idx):
        return [math.radians(self._arms[arm_idx]["spinboxes"][i].value())
                for i in range(7)]

    def _emit_command(self):
        self.joint_command.emit(self._get_arm_positions_rad(0),
                                self._get_arm_positions_rad(1))

    def _on_go_zero(self):
        """归零：双臂目标同时置零并发送"""
        self._updating_from_feedback = True
        for widgets in self._arms:
            for i in range(7):
                lo, hi = JOINT_LIMITS_DEG[i + 1]
                zero_val = 0.0 if lo <= 0.0 <= hi else lo
                widgets["spinboxes"][i].setValue(zero_val)
                widgets["sliders"][i].setValue(self._deg_to_slider(i, zero_val))
        self._updating_from_feedback = False
        self._emit_command()

    def _sync_to_feedback(self):
        """将双臂控制滑块分别同步到各自当前反馈位置"""
        self._updating_from_feedback = True
        for widgets in self._arms:
            for i in range(7):
                text = widgets["pos_labels"][i].text().replace("°", "")
                try:
                    deg = float(text)
                    widgets["spinboxes"][i].setValue(deg)
                    widgets["sliders"][i].setValue(self._deg_to_slider(i, deg))
                except ValueError:
                    pass
        self._updating_from_feedback = False

    def sync_to_feedback(self):
        """公开接口：将控制滑块/输入框同步到当前反馈位置"""
        self._sync_to_feedback()

    # ==================================================================
    # 反馈更新
    # ==================================================================

    def _update_feedback_arm(self, arm_idx, joint_states):
        positions = joint_states.to_list(include_gripper=True)
        self._updating_from_feedback = True
        pos_labels = self._arms[arm_idx]["pos_labels"]
        for i in range(min(7, len(positions))):
            deg = math.degrees(positions[i])
            pos_labels[i].setText(f"{deg:.1f}°")
        self._updating_from_feedback = False

    def update_feedback(self, joint_states):
        """从 ArmJointStates 更新臂1位置显示"""
        self._update_feedback_arm(0, joint_states)

    def update_feedback_arm2(self, joint_states):
        """从 ArmJointStates 更新臂2位置显示"""
        self._update_feedback_arm(1, joint_states)

    def _update_motor_feedback_arm(self, arm_idx, motor_feedback_list):
        """从 MotorFeedback 列表更新速度、力矩和温度显示。

        温度着色：正常灰色；≥70°C 淡红；≥80°C 深红。
        """
        widgets = self._arms[arm_idx]
        self._updating_from_feedback = True
        for i, fb in enumerate(motor_feedback_list[:7]):
            widgets["vel_labels"][i].setText(f"{fb.velocity:.2f}")
            widgets["torque_labels"][i].setText(f"{fb.torque:.2f}")
            if hasattr(fb, 'temperature') and fb.temperature is not None:
                t = float(fb.temperature)
                widgets["temp_labels"][i].setText(f"{t:.1f}")
                if t >= 80:
                    color = "#d32f2f"
                elif t >= 70:
                    color = "#ff6b6b"
                else:
                    color = "#888"
                widgets["temp_labels"][i].setStyleSheet(
                    f"color: {color}; font-weight: bold;")
            else:
                widgets["temp_labels"][i].setText("--")
                widgets["temp_labels"][i].setStyleSheet("color: #888;")
        self._updating_from_feedback = False

    def update_motor_feedback(self, motor_feedback_list):
        self._update_motor_feedback_arm(0, motor_feedback_list)

    def update_motor_feedback_arm2(self, motor_feedback_list):
        self._update_motor_feedback_arm(1, motor_feedback_list)

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        for widgets in self._arms:
            for s in widgets["sliders"]:
                s.setEnabled(enabled)
            for s in widgets["spinboxes"]:
                s.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        self.zero_btn.setEnabled(enabled)

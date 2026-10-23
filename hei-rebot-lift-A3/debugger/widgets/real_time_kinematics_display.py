"""末端位姿实时显示面板：双臂 XYZ 位置 + 四元数姿态

布局：臂1 一行（位置 | 四元数 | 欧拉角 | 夹爪），臂2 一行，竖直排列；
各组无标题，仅在位置组内以“臂1/臂2”标记区分
"""

import math

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
)
from PyQt6.QtCore import Qt

from debugger.utils.i18n import tr


class RealTimeKinematicsDisplay(QWidget):
    """实时显示双臂末端位姿：位置 + 四元数 + 欧拉角 + 夹爪"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_end_pose = None
        self._last_end_pose2 = None
        self._arm_rows = []
        self._init_ui()
        self.retranslate_ui()

    def _build_arm_row(self):
        """构建单臂显示行：位置 | 四元数 | 欧拉角 | 夹爪，返回控件字典"""
        row = QHBoxLayout()
        row.setSpacing(6)
        w = {}

        # 位置显示组
        pos_group = QGroupBox()
        w["pos_group"] = pos_group
        pos_layout = QHBoxLayout(pos_group)
        pos_layout.setSpacing(6)
        pos_layout.setContentsMargins(4, 4, 4, 4)

        w["arm_label"] = QLabel()
        w["arm_label"].setStyleSheet("font-weight: bold; font-size: 11px;")
        pos_layout.addWidget(w["arm_label"])

        pos_labels = {}
        for axis in ["X", "Y", "Z"]:
            lbl_name = QLabel(f"{axis}:")
            lbl_name.setStyleSheet("font-weight: bold; font-size: 10px;")
            pos_layout.addWidget(lbl_name)

            lbl_val = QLabel("0.000")
            lbl_val.setStyleSheet("font-family: monospace; font-size: 11px; color: #2196F3;")
            lbl_val.setMinimumWidth(50)
            pos_layout.addWidget(lbl_val)

            lbl_unit = QLabel("m")
            lbl_unit.setStyleSheet("font-size: 9px; color: gray;")
            pos_layout.addWidget(lbl_unit)

            pos_labels[axis] = lbl_val
        w["pos_labels"] = pos_labels
        row.addWidget(pos_group)

        # 四元数显示组
        quat_group = QGroupBox()
        w["quat_group"] = quat_group
        quat_layout = QHBoxLayout(quat_group)
        quat_layout.setSpacing(4)
        quat_layout.setContentsMargins(4, 4, 4, 4)

        quat_labels = {}
        for comp in ["w", "x", "y", "z"]:
            lbl_name = QLabel(f"{comp}:")
            lbl_name.setStyleSheet("font-weight: bold; font-size: 10px;")
            quat_layout.addWidget(lbl_name)

            lbl_val = QLabel("0.00")
            lbl_val.setStyleSheet("font-family: monospace; font-size: 11px; color: #4CAF50;")
            lbl_val.setMinimumWidth(45)
            quat_layout.addWidget(lbl_val)

            quat_labels[comp] = lbl_val
        w["quat_labels"] = quat_labels
        row.addWidget(quat_group)

        # 欧拉角显示组
        euler_group = QGroupBox()
        w["euler_group"] = euler_group
        euler_layout = QHBoxLayout(euler_group)
        euler_layout.setSpacing(4)
        euler_layout.setContentsMargins(4, 4, 4, 4)

        euler_labels = {}
        for name, key in [("R", "rx"), ("P", "ry"), ("Y", "rz")]:
            lbl_name = QLabel(f"{name}:")
            lbl_name.setStyleSheet("font-weight: bold; font-size: 10px;")
            euler_layout.addWidget(lbl_name)

            lbl_val = QLabel("0.0")
            lbl_val.setStyleSheet("font-family: monospace; font-size: 11px; color: #FF9800;")
            lbl_val.setMinimumWidth(40)
            euler_layout.addWidget(lbl_val)

            lbl_unit = QLabel("°")
            lbl_unit.setStyleSheet("font-size: 9px; color: gray;")
            euler_layout.addWidget(lbl_unit)

            euler_labels[key] = lbl_val
        w["euler_labels"] = euler_labels
        row.addWidget(euler_group)

        # 夹爪角度/力矩显示组
        gripper_group = QGroupBox()
        w["gripper_group"] = gripper_group
        gripper_layout = QHBoxLayout(gripper_group)
        gripper_layout.setSpacing(4)
        gripper_layout.setContentsMargins(4, 4, 4, 4)

        lbl_grip_name = QLabel("G:")
        lbl_grip_name.setStyleSheet("font-weight: bold; font-size: 10px;")
        gripper_layout.addWidget(lbl_grip_name)

        w["gripper_angle_label"] = QLabel("0.0")
        w["gripper_angle_label"].setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #E91E63;")
        w["gripper_angle_label"].setMinimumWidth(40)
        gripper_layout.addWidget(w["gripper_angle_label"])

        lbl_grip_unit = QLabel("rad")
        lbl_grip_unit.setStyleSheet("font-size: 9px; color: gray;")
        gripper_layout.addWidget(lbl_grip_unit)

        lbl_torque_name = QLabel("T:")
        lbl_torque_name.setStyleSheet("font-weight: bold; font-size: 10px;")
        gripper_layout.addWidget(lbl_torque_name)

        w["gripper_torque_label"] = QLabel("0.00")
        w["gripper_torque_label"].setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #9C27B0;")
        w["gripper_torque_label"].setMinimumWidth(40)
        gripper_layout.addWidget(w["gripper_torque_label"])

        lbl_torque_unit = QLabel("Nm")
        lbl_torque_unit.setStyleSheet("font-size: 9px; color: gray;")
        gripper_layout.addWidget(lbl_torque_unit)

        row.addWidget(gripper_group)
        w["row"] = row
        return w

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- 两行：臂1 | 臂2（每行：位置 | 四元数 | 欧拉角 | 夹爪） ---
        for _ in range(2):
            w = self._build_arm_row()
            self._arm_rows.append(w)
            layout.addLayout(w["row"])

        layout.addStretch()

    def retranslate_ui(self):
        for i, w in enumerate(self._arm_rows):
            w["arm_label"].setText(tr(f"jc.arm{i + 1}"))
            w["pos_group"].setTitle("")
            w["quat_group"].setTitle("")
            w["euler_group"].setTitle("")
            w["gripper_group"].setTitle("")

    def _update_row(self, arm_idx, end_pose, gripper_angle_rad, gripper_torque):
        if end_pose is None:
            return
        w = self._arm_rows[arm_idx]

        # 更新位置 (m)
        w["pos_labels"]["X"].setText(f"{end_pose.x:.3f}")
        w["pos_labels"]["Y"].setText(f"{end_pose.y:.3f}")
        w["pos_labels"]["Z"].setText(f"{end_pose.z:.3f}")

        # 欧拉角转四元数（ZYX 约定，等价 pin.rpy.rpyToMatrix 的 R=Rz@Ry@Rx）。
        # 不依赖 pinocchio：VLA 环境的 pinocchio 包无 rpy 模块（伪包/版本差异）。
        cr = math.cos(end_pose.rx * 0.5); sr = math.sin(end_pose.rx * 0.5)
        cp = math.cos(end_pose.ry * 0.5); sp = math.sin(end_pose.ry * 0.5)
        cy = math.cos(end_pose.rz * 0.5); sy = math.sin(end_pose.rz * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        # 更新四元数
        w["quat_labels"]["w"].setText(f"{qw:.4f}")
        w["quat_labels"]["x"].setText(f"{qx:.4f}")
        w["quat_labels"]["y"].setText(f"{qy:.4f}")
        w["quat_labels"]["z"].setText(f"{qz:.4f}")

        # 更新欧拉角显示 (rad -> deg)
        w["euler_labels"]["rx"].setText(f"{math.degrees(end_pose.rx):.2f}")
        w["euler_labels"]["ry"].setText(f"{math.degrees(end_pose.ry):.2f}")
        w["euler_labels"]["rz"].setText(f"{math.degrees(end_pose.rz):.2f}")

        # 更新夹爪角度（弧度）
        if gripper_angle_rad is not None:
            w["gripper_angle_label"].setText(f"{gripper_angle_rad:.3f}")

        # 更新夹爪力矩 (Nm)
        if gripper_torque is not None:
            w["gripper_torque_label"].setText(f"{gripper_torque:.2f}")

    def update_end_pose(self, end_pose, gripper_angle_rad=None, gripper_torque=None):
        """
        更新臂1 末端位姿显示

        Args:
            end_pose: ArmEndPose 对象 (x, y, z, rx, ry, rz)
            gripper_angle_rad: 夹爪角度（弧度），可选
            gripper_torque: 夹爪力矩（Nm），可选
        """
        if end_pose is not None:
            self._last_end_pose = end_pose
        self._update_row(0, end_pose, gripper_angle_rad, gripper_torque)

    def update_end_pose_arm2(self, end_pose, gripper_angle_rad=None, gripper_torque=None):
        """更新臂2 末端位姿显示，参数同 update_end_pose"""
        if end_pose is not None:
            self._last_end_pose2 = end_pose
        self._update_row(1, end_pose, gripper_angle_rad, gripper_torque)

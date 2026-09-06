"""末端位姿实时显示面板：显示 XYZ 位置 + 四元数姿态

布局：底部信息反馈框右侧，显示末端执行器的实时位姿
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QGridLayout, QFrame,
)
from PyQt6.QtCore import Qt

from debugger.utils.i18n import tr


class RealTimeKinematicsDisplay(QWidget):
    """实时显示机械臂末端位姿：位置 + 四元数"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_end_pose = None
        self._init_ui()
        self.retranslate_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- 水平排列：位置 | 四元数 | 欧拉角 ---
        main_row = QHBoxLayout()
        main_row.setSpacing(6)

        # 位置显示组
        pos_group = QGroupBox()
        self.pos_group = pos_group
        pos_layout = QHBoxLayout(pos_group)
        pos_layout.setSpacing(6)
        pos_layout.setContentsMargins(4, 4, 4, 4)

        self.pos_labels = {}
        pos_axes = ["X", "Y", "Z"]
        for axis in pos_axes:
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

            self.pos_labels[axis] = lbl_val

        main_row.addWidget(pos_group)

        # 四元数显示组
        quat_group = QGroupBox()
        self.quat_group = quat_group
        quat_layout = QHBoxLayout(quat_group)
        quat_layout.setSpacing(4)
        quat_layout.setContentsMargins(4, 4, 4, 4)

        self.quat_labels = {}
        quat_comps = ["w", "x", "y", "z"]
        for comp in quat_comps:
            lbl_name = QLabel(f"{comp}:")
            lbl_name.setStyleSheet("font-weight: bold; font-size: 10px;")
            quat_layout.addWidget(lbl_name)

            lbl_val = QLabel("0.00")
            lbl_val.setStyleSheet("font-family: monospace; font-size: 11px; color: #4CAF50;")
            lbl_val.setMinimumWidth(45)
            quat_layout.addWidget(lbl_val)

            self.quat_labels[comp] = lbl_val

        main_row.addWidget(quat_group)

        # 欧拉角显示组
        euler_group = QGroupBox()
        self.euler_group = euler_group
        euler_layout = QHBoxLayout(euler_group)
        euler_layout.setSpacing(4)
        euler_layout.setContentsMargins(4, 4, 4, 4)

        self.euler_labels = {}
        euler_axes = [("R", "rx"), ("P", "ry"), ("Y", "rz")]
        for name, key in euler_axes:
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

            self.euler_labels[key] = lbl_val

        main_row.addWidget(euler_group)

        # 夹爪角度显示组
        gripper_group = QGroupBox()
        self.gripper_group = gripper_group
        gripper_layout = QHBoxLayout(gripper_group)
        gripper_layout.setSpacing(4)
        gripper_layout.setContentsMargins(4, 4, 4, 4)

        lbl_grip_name = QLabel("G:")
        lbl_grip_name.setStyleSheet("font-weight: bold; font-size: 10px;")
        gripper_layout.addWidget(lbl_grip_name)

        self.gripper_angle_label = QLabel("0.0")
        self.gripper_angle_label.setStyleSheet("font-family: monospace; font-size: 11px; color: #E91E63;")
        self.gripper_angle_label.setMinimumWidth(40)
        gripper_layout.addWidget(self.gripper_angle_label)

        lbl_grip_unit = QLabel("rad")
        lbl_grip_unit.setStyleSheet("font-size: 9px; color: gray;")
        gripper_layout.addWidget(lbl_grip_unit)

        lbl_torque_name = QLabel("T:")
        lbl_torque_name.setStyleSheet("font-weight: bold; font-size: 10px;")
        gripper_layout.addWidget(lbl_torque_name)

        self.gripper_torque_label = QLabel("0.00")
        self.gripper_torque_label.setStyleSheet("font-family: monospace; font-size: 11px; color: #9C27B0;")
        self.gripper_torque_label.setMinimumWidth(40)
        gripper_layout.addWidget(self.gripper_torque_label)

        lbl_torque_unit = QLabel("Nm")
        lbl_torque_unit.setStyleSheet("font-size: 9px; color: gray;")
        gripper_layout.addWidget(lbl_torque_unit)

        main_row.addWidget(gripper_group)
        layout.addLayout(main_row)
        layout.addStretch()

    def retranslate_ui(self):
        self.pos_group.setTitle(tr("kin.pos_title"))
        self.quat_group.setTitle(tr("kin.quat_title"))
        self.euler_group.setTitle(tr("kin.euler_title"))
        self.gripper_group.setTitle(tr("kin.gripper_title") if hasattr(self, 'gripper_group') else "Gripper")

    def update_end_pose(self, end_pose, gripper_angle_rad=None, gripper_torque=None):
        """
        更新末端位姿显示

        Args:
            end_pose: ArmEndPose 对象 (x, y, z, rx, ry, rz)
            gripper_angle_rad: 夹爪角度（弧度），可选
            gripper_torque: 夹爪力矩（Nm），可选
        """
        if end_pose is None:
            return

        self._last_end_pose = end_pose

        # 更新位置 (m)
        self.pos_labels["X"].setText(f"{end_pose.x:.3f}")
        self.pos_labels["Y"].setText(f"{end_pose.y:.3f}")
        self.pos_labels["Z"].setText(f"{end_pose.z:.3f}")

        # 欧拉角转四元数（ZYX 约定，等价 pin.rpy.rpyToMatrix 的 R=Rz@Ry@Rx）。
        # 不依赖 pinocchio：VLA 环境的 pinocchio 包无 rpy 模块（伪包/版本差异）。
        import math
        cr = math.cos(end_pose.rx * 0.5); sr = math.sin(end_pose.rx * 0.5)
        cp = math.cos(end_pose.ry * 0.5); sp = math.sin(end_pose.ry * 0.5)
        cy = math.cos(end_pose.rz * 0.5); sy = math.sin(end_pose.rz * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        # 更新四元数
        self.quat_labels["w"].setText(f"{qw:.4f}")
        self.quat_labels["x"].setText(f"{qx:.4f}")
        self.quat_labels["y"].setText(f"{qy:.4f}")
        self.quat_labels["z"].setText(f"{qz:.4f}")

        # 更新欧拉角显示 (rad -> deg)
        import math
        self.euler_labels["rx"].setText(f"{math.degrees(end_pose.rx):.2f}")
        self.euler_labels["ry"].setText(f"{math.degrees(end_pose.ry):.2f}")
        self.euler_labels["rz"].setText(f"{math.degrees(end_pose.rz):.2f}")

        # 更新夹爪角度（弧度）
        if gripper_angle_rad is not None:
            self.gripper_angle_label.setText(f"{gripper_angle_rad:.3f}")

        # 更新夹爪力矩 (Nm)
        if gripper_torque is not None:
            self.gripper_torque_label.setText(f"{gripper_torque:.2f}")

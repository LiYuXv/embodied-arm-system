"""
RobStride 电机控制界面 - 仅包含UI和交互逻辑
用于后续适配其他关节电机的界面基础
"""

import sys
import math
import time

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QTextEdit,
    QGridLayout,
    QSlider,
    QTabWidget,
    QFrame,
    QDialog,
    QLineEdit,
    QCheckBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont


class SystemSettingsDialog(QDialog):
    """系统设置对话框"""

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config or {}
        self.setWindowTitle("系统设置")
        self.setGeometry(200, 200, 500, 450)
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_comm_tab(), "通信设置")
        self.tabs.addTab(self.create_safety_tab(), "安全限制")

        layout.addWidget(self.tabs)

        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("保存设置")
        self.save_btn.setObjectName("success")
        self.save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        self.reset_btn = QPushButton("恢复默认")
        self.reset_btn.clicked.connect(self.reset_defaults)
        btn_layout.addWidget(self.reset_btn)

        layout.addLayout(btn_layout)

        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton#success {
                background-color: #27ae60;
            }
            QPushButton#success:hover {
                background-color: #219a52;
            }
            QDoubleSpinBox, QSpinBox, QComboBox {
                padding: 4px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
            }
            QLabel {
                padding: 2px;
            }
        """)

    def create_comm_tab(self):
        """通信设置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        serial_group = QGroupBox("通信设置")
        serial_layout = QGridLayout()

        serial_layout.addWidget(QLabel("接口类型:"), 0, 0)
        self.interface_type_combo = QComboBox()
        self.interface_type_combo.addItem("串口 (Serial) - 推荐", "serial")
        self.interface_type_combo.addItem("CAN总线", "can")
        self.interface_type_combo.currentIndexChanged.connect(self.on_interface_type_changed)
        serial_layout.addWidget(self.interface_type_combo, 0, 1)

        serial_layout.addWidget(QLabel("端口:"), 1, 0)
        self.default_port_combo = QComboBox()
        self.default_port_combo.setEditable(True)
        self.default_port_combo.addItem("can0")
        serial_layout.addWidget(self.default_port_combo, 1, 1)

        serial_layout.addWidget(QLabel("波特率:"), 2, 0)
        self.default_baud_combo = QComboBox()
        self.default_baud_combo.addItems(["921600", "115200", "460800", "1000000"])
        self.default_baud_combo.setCurrentText("921600")
        serial_layout.addWidget(self.default_baud_combo, 2, 1)

        serial_group.setLayout(serial_layout)
        layout.addWidget(serial_group)

        freq_group = QGroupBox("更新频率设置")
        freq_layout = QGridLayout()

        freq_layout.addWidget(QLabel("状态更新频率 (Hz):"), 0, 0)
        self.status_freq_spin = QSpinBox()
        self.status_freq_spin.setRange(10, 1000)
        self.status_freq_spin.setValue(100)
        self.status_freq_spin.setSuffix(" Hz")
        freq_layout.addWidget(self.status_freq_spin, 0, 1)

        freq_layout.addWidget(QLabel("运控控制频率 (Hz):"), 1, 0)
        self.control_freq_spin = QSpinBox()
        self.control_freq_spin.setRange(10, 1000)
        self.control_freq_spin.setValue(500)
        self.control_freq_spin.setSuffix(" Hz")
        freq_layout.addWidget(self.control_freq_spin, 1, 1)

        hint = QLabel("提示: 状态更新频率建议10-200Hz，控制频率建议100-1000Hz")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        freq_layout.addWidget(hint, 2, 0, 1, 2)

        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group)

        slider_group = QGroupBox("滑条控制设置")
        slider_layout = QGridLayout()

        slider_layout.addWidget(QLabel("步长 (°):"), 0, 0)
        self.slider_step_spin = QDoubleSpinBox()
        self.slider_step_spin.setRange(0.1, 90.0)
        self.slider_step_spin.setValue(5.0)
        self.slider_step_spin.setDecimals(1)
        self.slider_step_spin.setSuffix(" °")
        slider_layout.addWidget(self.slider_step_spin, 0, 1)

        slider_layout.addWidget(QLabel("步长时间 (s):"), 1, 0)
        self.slider_step_time_spin = QDoubleSpinBox()
        self.slider_step_time_spin.setRange(0.1, 10.0)
        self.slider_step_time_spin.setValue(1.0)
        self.slider_step_time_spin.setDecimals(1)
        self.slider_step_time_spin.setSuffix(" s")
        slider_layout.addWidget(self.slider_step_time_spin, 1, 1)

        slider_layout.addWidget(QLabel("控制频率 (Hz):"), 2, 0)
        self.slider_freq_spin = QSpinBox()
        self.slider_freq_spin.setRange(10, 200)
        self.slider_freq_spin.setValue(100)
        self.slider_freq_spin.setSuffix(" Hz")
        slider_layout.addWidget(self.slider_freq_spin, 2, 1)

        hint = QLabel("提示: 步长为每次点击±按钮的角度，步长时间为运动时间，控制频率为滑条实时更新频率")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        slider_layout.addWidget(hint, 3, 0, 1, 2)

        slider_group.setLayout(slider_layout)
        layout.addWidget(slider_group)

        layout.addStretch()
        return tab

    def create_safety_tab(self):
        """安全限制标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.safety_tabs = QTabWidget()

        for i in range(7):
            joint_tab = self.create_joint_safety_tab(i)
            self.safety_tabs.addTab(joint_tab, f"关节{i + 1}")

        layout.addWidget(self.safety_tabs)

        layout.addStretch()
        return tab

    def create_joint_safety_tab(self, joint_idx):
        """创建单个关节的安全限制标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        enable_check = QCheckBox(f"启用关节{joint_idx + 1}安全限制")
        enable_check.setChecked(True)
        enable_check.setStyleSheet("font-weight: bold; color: #e74c3c;")
        layout.addWidget(enable_check)
        setattr(self, f"joint{joint_idx + 1}_enable_check", enable_check)

        limit_group = QGroupBox(f"关节{joint_idx + 1}安全限制设置")
        limit_layout = QGridLayout()

        limit_layout.addWidget(QLabel("位置限制 (rad):"), 0, 0)
        pos_layout = QHBoxLayout()
        pos_min_spin = QDoubleSpinBox()
        pos_min_spin.setRange(-50, 50)
        pos_min_spin.setValue(-3.0)
        pos_min_spin.setDecimals(2)
        pos_layout.addWidget(QLabel("最小:"))
        pos_layout.addWidget(pos_min_spin)
        pos_layout.addWidget(QLabel("最大:"))
        pos_max_spin = QDoubleSpinBox()
        pos_max_spin.setRange(-50, 50)
        pos_max_spin.setValue(3.0)
        pos_max_spin.setDecimals(2)
        pos_layout.addWidget(pos_max_spin)
        limit_layout.addLayout(pos_layout, 0, 1)
        setattr(self, f"joint{joint_idx + 1}_pos_min", pos_min_spin)
        setattr(self, f"joint{joint_idx + 1}_pos_max", pos_max_spin)

        limit_layout.addWidget(QLabel("速度限制 (rad/s):"), 1, 0)
        vel_layout = QHBoxLayout()
        vel_min_spin = QDoubleSpinBox()
        vel_min_spin.setRange(-100, 100)
        vel_min_spin.setValue(-6.0)
        vel_min_spin.setDecimals(1)
        vel_layout.addWidget(QLabel("最小:"))
        vel_layout.addWidget(vel_min_spin)
        vel_layout.addWidget(QLabel("最大:"))
        vel_max_spin = QDoubleSpinBox()
        vel_max_spin.setRange(-100, 100)
        vel_max_spin.setValue(6.0)
        vel_max_spin.setDecimals(1)
        vel_layout.addWidget(vel_max_spin)
        limit_layout.addLayout(vel_layout, 1, 1)
        setattr(self, f"joint{joint_idx + 1}_vel_min", vel_min_spin)
        setattr(self, f"joint{joint_idx + 1}_vel_max", vel_max_spin)

        limit_layout.addWidget(QLabel("力矩限制 (Nm):"), 2, 0)
        torque_layout = QHBoxLayout()
        torque_min_spin = QDoubleSpinBox()
        torque_min_spin.setRange(-50, 50)
        torque_min_spin.setValue(-17.0)
        torque_min_spin.setDecimals(1)
        torque_layout.addWidget(QLabel("最小:"))
        torque_layout.addWidget(torque_min_spin)
        torque_layout.addWidget(QLabel("最大:"))
        torque_max_spin = QDoubleSpinBox()
        torque_max_spin.setRange(-50, 50)
        torque_max_spin.setValue(17.0)
        torque_max_spin.setDecimals(1)
        torque_layout.addWidget(torque_max_spin)
        limit_layout.addLayout(torque_layout, 2, 1)
        setattr(self, f"joint{joint_idx + 1}_torque_min", torque_min_spin)
        setattr(self, f"joint{joint_idx + 1}_torque_max", torque_max_spin)

        limit_group.setLayout(limit_layout)
        layout.addWidget(limit_group)

        warning = QLabel(
            f"⚠️ 警告: 当关节{joint_idx + 1}的位置、速度或力矩超出限制时，\n"
            "系统将自动失能电机以确保安全！"
        )
        warning.setStyleSheet(
            "color: #e74c3c; font-weight: bold; background-color: #fadbd8; "
            "padding: 10px; border-radius: 5px; margin-top: 10px;"
        )
        warning.setAlignment(Qt.AlignCenter)
        layout.addWidget(warning)

        layout.addStretch()
        return tab

    def on_interface_type_changed(self, index):
        """处理接口类型切换"""
        interface_type = self.interface_type_combo.itemData(index)

        if interface_type == "can":
            self.default_port_combo.clear()
            self.default_port_combo.setEditable(True)
            self.default_port_combo.addItem("0")
            self.default_baud_combo.setEnabled(False)
        else:
            self.default_port_combo.clear()
            self.default_port_combo.setEditable(True)
            self.default_port_combo.addItem("can0")
            self.default_baud_combo.setEnabled(True)

    def load_settings(self):
        """加载设置到界面"""
        if not self.config:
            return

        interface_type = self.config.get("interface_type", "serial")
        for i in range(self.interface_type_combo.count()):
            if self.interface_type_combo.itemData(i) == interface_type:
                self.interface_type_combo.setCurrentIndex(i)
                break

        self.default_port_combo.setCurrentText(self.config.get("default_port", "can0"))
        self.default_baud_combo.setCurrentText(
            str(self.config.get("default_baudrate", 921600))
        )
        self.status_freq_spin.setValue(self.config.get("status_update_frequency", 100))
        self.control_freq_spin.setValue(
            self.config.get("motion_control_frequency", 500)
        )

        self.slider_step_spin.setValue(self.config.get("slider_step_deg", 5.0))
        self.slider_step_time_spin.setValue(self.config.get("slider_step_time", 1.0))
        self.slider_freq_spin.setValue(self.config.get("slider_freq", 100))

        safety_config = self.config.get("safety", {})
        for i in range(7):
            joint_safety = safety_config.get(
                f"joint{i + 1}",
                {
                    "enabled": True,
                    "pos_min": -3.0,
                    "pos_max": 3.0,
                    "vel_min": -6.0,
                    "vel_max": 6.0,
                    "torque_min": -17.0,
                    "torque_max": 17.0,
                },
            )
            enable_check = getattr(self, f"joint{i + 1}_enable_check", None)
            if enable_check:
                enable_check.setChecked(joint_safety.get("enabled", True))
            pos_min = getattr(self, f"joint{i + 1}_pos_min", None)
            pos_max = getattr(self, f"joint{i + 1}_pos_max", None)
            if pos_min:
                pos_min.setValue(joint_safety.get("pos_min", -3.0))
            if pos_max:
                pos_max.setValue(joint_safety.get("pos_max", 3.0))
            vel_min = getattr(self, f"joint{i + 1}_vel_min", None)
            vel_max = getattr(self, f"joint{i + 1}_vel_max", None)
            if vel_min:
                vel_min.setValue(joint_safety.get("vel_min", -6.0))
            if vel_max:
                vel_max.setValue(joint_safety.get("vel_max", 6.0))
            torque_min = getattr(self, f"joint{i + 1}_torque_min", None)
            torque_max = getattr(self, f"joint{i + 1}_torque_max", None)
            if torque_min:
                torque_min.setValue(joint_safety.get("torque_min", -17.0))
            if torque_max:
                torque_max.setValue(joint_safety.get("torque_max", 17.0))

    def get_settings(self):
        """获取当前设置"""
        safety = {}
        for i in range(7):
            enable_check = getattr(self, f"joint{i + 1}_enable_check", None)
            pos_min = getattr(self, f"joint{i + 1}_pos_min", None)
            pos_max = getattr(self, f"joint{i + 1}_pos_max", None)
            vel_min = getattr(self, f"joint{i + 1}_vel_min", None)
            vel_max = getattr(self, f"joint{i + 1}_vel_max", None)
            torque_min = getattr(self, f"joint{i + 1}_torque_min", None)
            torque_max = getattr(self, f"joint{i + 1}_torque_max", None)

            safety[f"joint{i + 1}"] = {
                "enabled": enable_check.isChecked() if enable_check else True,
                "pos_min": pos_min.value() if pos_min else -3.0,
                "pos_max": pos_max.value() if pos_max else 3.0,
                "vel_min": vel_min.value() if vel_min else -6.0,
                "vel_max": vel_max.value() if vel_max else 6.0,
                "torque_min": torque_min.value() if torque_min else -17.0,
                "torque_max": torque_max.value() if torque_max else 17.0,
            }

        return {
            "interface_type": self.interface_type_combo.itemData(
                self.interface_type_combo.currentIndex()
            ),
            "default_port": self.default_port_combo.currentText(),
            "default_baudrate": int(self.default_baud_combo.currentText()),
            "status_update_frequency": self.status_freq_spin.value(),
            "motion_control_frequency": self.control_freq_spin.value(),
            "slider_step_deg": self.slider_step_spin.value(),
            "slider_step_time": self.slider_step_time_spin.value(),
            "slider_freq": self.slider_freq_spin.value(),
            "safety": safety,
        }

    def reset_defaults(self):
        """恢复默认设置"""
        self.interface_type_combo.setCurrentIndex(1)
        self.default_port_combo.setCurrentText("can0")
        self.default_baud_combo.setCurrentText("921600")
        self.status_freq_spin.setValue(100)
        self.control_freq_spin.setValue(500)
        self.slider_step_spin.setValue(5.0)
        self.slider_step_time_spin.setValue(1.0)
        self.slider_freq_spin.setValue(100)

        for i in range(7):
            enable_check = getattr(self, f"joint{i + 1}_enable_check", None)
            if enable_check:
                enable_check.setChecked(True)
            pos_min = getattr(self, f"joint{i + 1}_pos_min", None)
            pos_max = getattr(self, f"joint{i + 1}_pos_max", None)
            if pos_min:
                pos_min.setValue(-3.0)
            if pos_max:
                pos_max.setValue(3.0)
            vel_min = getattr(self, f"joint{i + 1}_vel_min", None)
            vel_max = getattr(self, f"joint{i + 1}_vel_max", None)
            if vel_min:
                vel_min.setValue(-6.0)
            if vel_max:
                vel_max.setValue(6.0)
            torque_min = getattr(self, f"joint{i + 1}_torque_min", None)
            torque_max = getattr(self, f"joint{i + 1}_torque_max", None)
            if torque_min:
                torque_min.setValue(-17.0)
            if torque_max:
                torque_max.setValue(17.0)


class MotorControlGUI(QMainWindow):
    """电机控制主界面 - 仅包含UI和交互逻辑"""

    NUM_MOTORS = 7

    def __init__(self):
        super().__init__()

        self.system_config = {
            "interface_type": "can",
            "default_port": "can0",
            "default_baudrate": 921600,
            "status_update_frequency": 100,
            "motion_control_frequency": 500,
            "slider_step_deg": 5.0,
            "slider_step_time": 1.0,
            "slider_freq": 100,
            "safety": {
                "enabled": True,
                "pos_min": -3.0,
                "pos_max": 3.0,
                "vel_min": -6.0,
                "vel_max": 6.0,
                "torque_min": -17.0,
                "torque_max": 17.0,
            },
        }

        self.status_update_frequency = 100
        self.motion_control_frequency = 500

        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("RobStride 电机控制 - PyQt5")
        self.setGeometry(100, 100, 950, 750)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)

        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)

        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QPushButton#danger {
                background-color: #e74c3c;
            }
            QPushButton#danger:hover {
                background-color: #c0392b;
            }
            QPushButton#success {
                background-color: #27ae60;
            }
            QPushButton#success:hover {
                background-color: #219a52;
            }
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: Consolas, Monaco, monospace;
            }
        """)

    def create_left_panel(self):
        """创建左侧面板（控制区）"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # ===== 连接控制组 =====
        conn_group = QGroupBox("连接控制")
        conn_layout = QHBoxLayout()

        self.connect_btn = QPushButton("🔌 连接")
        self.connect_btn.setObjectName("success")
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        conn_layout.addWidget(self.connect_btn)

        self.system_settings_btn = QPushButton("⚙ 设置")
        self.system_settings_btn.setToolTip("配置串口、频率和安全限制")
        self.system_settings_btn.clicked.connect(self.on_settings_clicked)
        conn_layout.addWidget(self.system_settings_btn)

        self.safety_status_label = QLabel("安全限制: 关闭")
        self.safety_status_label.setStyleSheet("color: gray;")
        conn_layout.addWidget(self.safety_status_label)

        # 隐藏的控件
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.addItem("can0")

        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["921600", "115200", "460800", "1000000"])
        self.baud_combo.setCurrentText("921600")

        self.motor_id_spin = QSpinBox()
        self.motor_id_spin.setRange(1, 127)
        self.motor_id_spin.setValue(1)

        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)

        # ===== 电机控制组 =====
        motor_group = QGroupBox("电机控制")
        motor_layout = QHBoxLayout()

        self.enable_btn = QPushButton("▶ 使能")
        self.enable_btn.setObjectName("success")
        self.enable_btn.clicked.connect(self.on_enable_clicked)
        self.enable_btn.setEnabled(False)
        motor_layout.addWidget(self.enable_btn)

        self.disable_btn = QPushButton("⏹ 失能")
        self.disable_btn.setObjectName("danger")
        self.disable_btn.clicked.connect(self.on_disable_clicked)
        self.disable_btn.setEnabled(False)
        motor_layout.addWidget(self.disable_btn)

        self.zero_btn = QPushButton("⭕ 置零")
        self.zero_btn.clicked.connect(self.on_zero_clicked)
        self.zero_btn.setEnabled(False)
        motor_layout.addWidget(self.zero_btn)

        self.clear_error_btn = QPushButton("🔄 清错")
        self.clear_error_btn.clicked.connect(self.on_clear_error_clicked)
        self.clear_error_btn.setEnabled(False)
        motor_layout.addWidget(self.clear_error_btn)

        self.home_btn = QPushButton("🏠 初始位置")
        self.home_btn.clicked.connect(self.on_home_clicked)
        self.home_btn.setEnabled(False)
        motor_layout.addWidget(self.home_btn)

        self.pp_home_btn = QPushButton("🔄 PP回零")
        self.pp_home_btn.setObjectName("success")
        self.pp_home_btn.clicked.connect(self.on_pp_home_clicked)
        self.pp_home_btn.setEnabled(False)
        motor_layout.addWidget(self.pp_home_btn)

        motor_group.setLayout(motor_layout)
        layout.addWidget(motor_group)

        # ===== 模式选择组 =====
        mode_group = QGroupBox("控制模式选择")
        mode_layout = QVBoxLayout()

        mode_select_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "运控模式 (Motion Control)",
            "位置模式 - PP (Profile Position)",
            "位置模式 - CSP (Cyclic Synchronous Position)",
            "速度模式 (Speed)",
            "电流模式 (Current)",
            "JOG 模式",
        ])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_select_layout.addWidget(self.mode_combo, 3)

        self.switch_mode_btn = QPushButton("🔄 切换模式")
        self.switch_mode_btn.setObjectName("success")
        self.switch_mode_btn.clicked.connect(self.on_switch_mode_clicked)
        self.switch_mode_btn.setEnabled(False)
        mode_select_layout.addWidget(self.switch_mode_btn, 1)

        mode_layout.addLayout(mode_select_layout)

        self.mode_tabs = QTabWidget()
        self.mode_tabs.setEnabled(False)

        self.mode_tabs.addTab(self.create_motion_tab(), "运控参数")
        self.mode_tabs.addTab(self.create_position_tab(), "位置PP参数")
        self.mode_tabs.addTab(self.create_csp_tab(), "位置CSP参数")
        self.mode_tabs.addTab(self.create_speed_tab(), "速度参数")
        self.mode_tabs.addTab(self.create_current_tab(), "电流参数")
        self.mode_tabs.addTab(self.create_jog_tab(), "JOG参数")

        mode_layout.addWidget(self.mode_tabs)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ 启动运动")
        self.start_btn.setObjectName("success")
        self.start_btn.clicked.connect(self.on_start_clicked)
        self.start_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹ 停止运动")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        mode_layout.addLayout(btn_layout)

        self.mode_hint_label = QLabel("提示: 选择模式后点击'切换模式'按钮进行模式切换")
        self.mode_hint_label.setStyleSheet("color: gray; font-size: 12px;")
        mode_layout.addWidget(self.mode_hint_label)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # ===== 日志显示组 =====
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        log_layout.addWidget(self.log_text)

        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(clear_log_btn)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()
        return panel

    def create_motion_tab(self):
        """运控模式参数"""
        tab = QWidget()
        layout = QGridLayout(tab)

        layout.addWidget(QLabel("目标位置 (rad):"), 0, 0)
        self.motion_pos_spin = QDoubleSpinBox()
        self.motion_pos_spin.setRange(-12.57, 12.57)
        self.motion_pos_spin.setValue(0.0)
        self.motion_pos_spin.setDecimals(2)
        self.motion_pos_spin.setSingleStep(0.1)
        layout.addWidget(self.motion_pos_spin, 0, 1)

        self.get_pos_btn = QPushButton("获取当前位置")
        self.get_pos_btn.clicked.connect(lambda: self.on_get_position_clicked(self.motion_pos_spin))
        layout.addWidget(self.get_pos_btn, 0, 2)

        layout.addWidget(QLabel("目标速度 (rad/s):"), 1, 0)
        self.motion_vel_spin = QDoubleSpinBox()
        self.motion_vel_spin.setRange(-33.0, 33.0)
        self.motion_vel_spin.setValue(0.0)
        self.motion_vel_spin.setDecimals(1)
        layout.addWidget(self.motion_vel_spin, 1, 1)

        layout.addWidget(QLabel("力矩前馈 (Nm):"), 2, 0)
        self.motion_torque_spin = QDoubleSpinBox()
        self.motion_torque_spin.setRange(-14.0, 14.0)
        self.motion_torque_spin.setValue(0.0)
        self.motion_torque_spin.setDecimals(2)
        layout.addWidget(self.motion_torque_spin, 2, 1)

        layout.addWidget(QLabel("位置增益 Kp:"), 3, 0)
        self.motion_kp_spin = QDoubleSpinBox()
        self.motion_kp_spin.setRange(0, 500)
        self.motion_kp_spin.setValue(0.0)
        self.motion_kp_spin.setDecimals(1)
        layout.addWidget(self.motion_kp_spin, 3, 1)

        layout.addWidget(QLabel("速度增益 Kd:"), 4, 0)
        self.motion_kd_spin = QDoubleSpinBox()
        self.motion_kd_spin.setRange(0, 5)
        self.motion_kd_spin.setValue(0.0)
        self.motion_kd_spin.setDecimals(2)
        layout.addWidget(self.motion_kd_spin, 4, 1)

        layout.addWidget(QLabel("控制频率 (Hz):"), 5, 0)
        self.motion_freq_spin = QSpinBox()
        self.motion_freq_spin.setRange(10, 1000)
        self.motion_freq_spin.setValue(500)
        layout.addWidget(self.motion_freq_spin, 5, 1)

        hint = QLabel("提示: Kp=0,Kd=0 为纯力矩模式\nKp>0 为位置跟踪模式")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(hint, 6, 0, 1, 3)

        return tab

    def create_position_tab(self):
        """位置PP模式参数"""
        tab = QWidget()
        layout = QGridLayout(tab)

        layout.addWidget(QLabel("目标位置 (rad):"), 0, 0)
        self.pos_target_spin = QDoubleSpinBox()
        self.pos_target_spin.setRange(-12.57, 12.57)
        self.pos_target_spin.setValue(0.0)
        self.pos_target_spin.setDecimals(2)
        self.pos_target_spin.setSingleStep(0.1)
        layout.addWidget(self.pos_target_spin, 0, 1)

        self.get_pp_pos_btn = QPushButton("获取当前位置")
        self.get_pp_pos_btn.clicked.connect(lambda: self.on_get_position_clicked(self.pos_target_spin))
        layout.addWidget(self.get_pp_pos_btn, 0, 2)

        layout.addWidget(QLabel("目标速度 (rad/s):"), 1, 0)
        self.pos_speed_spin = QDoubleSpinBox()
        self.pos_speed_spin.setRange(0.1, 33.0)
        self.pos_speed_spin.setValue(5.0)
        self.pos_speed_spin.setDecimals(1)
        layout.addWidget(self.pos_speed_spin, 1, 1)

        layout.addWidget(QLabel("加速度 (rad/s²):"), 2, 0)
        self.pos_acc_spin = QDoubleSpinBox()
        self.pos_acc_spin.setRange(0.1, 100)
        self.pos_acc_spin.setValue(10.0)
        self.pos_acc_spin.setDecimals(1)
        layout.addWidget(self.pos_acc_spin, 2, 1)

        hint = QLabel("提示: PP模式适合点位运动，会规划轨迹")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(hint, 3, 0, 1, 3)

        return tab

    def create_csp_tab(self):
        """位置CSP模式参数"""
        tab = QWidget()
        layout = QGridLayout(tab)

        layout.addWidget(QLabel("目标位置 (rad):"), 0, 0)
        self.csp_target_spin = QDoubleSpinBox()
        self.csp_target_spin.setRange(-12.57, 12.57)
        self.csp_target_spin.setValue(0.0)
        self.csp_target_spin.setDecimals(2)
        self.csp_target_spin.setSingleStep(0.1)
        layout.addWidget(self.csp_target_spin, 0, 1)

        self.get_csp_pos_btn = QPushButton("获取当前位置")
        self.get_csp_pos_btn.clicked.connect(lambda: self.on_get_position_clicked(self.csp_target_spin))
        layout.addWidget(self.get_csp_pos_btn, 0, 2)

        layout.addWidget(QLabel("速度限制 (rad/s):"), 1, 0)
        self.csp_speed_spin = QDoubleSpinBox()
        self.csp_speed_spin.setRange(0.1, 33.0)
        self.csp_speed_spin.setValue(5.0)
        self.csp_speed_spin.setDecimals(1)
        layout.addWidget(self.csp_speed_spin, 1, 1)

        info = QLabel(
            "CSP模式特点:\n"
            "• 周期性同步位置控制\n"
            "• 目标位置立即生效，无轨迹规划\n"
            "• 适合多轴同步运动\n"
            "• 切换模式时自动继承当前位置"
        )
        info.setStyleSheet(
            "color: #2980b9; font-size: 11px; background-color: #ecf0f1; padding: 8px; border-radius: 4px;"
        )
        layout.addWidget(info, 2, 0, 1, 3)

        return tab

    def create_speed_tab(self):
        """速度模式参数"""
        tab = QWidget()
        layout = QGridLayout(tab)

        layout.addWidget(QLabel("目标速度 (rad/s):"), 0, 0)
        speed_input_layout = QHBoxLayout()

        self.speed_target_spin = QDoubleSpinBox()
        self.speed_target_spin.setRange(-33.0, 33.0)
        self.speed_target_spin.setValue(0.0)
        self.speed_target_spin.setDecimals(1)
        speed_input_layout.addWidget(self.speed_target_spin)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(-330, 330)
        self.speed_slider.setValue(0)
        self.speed_slider.valueChanged.connect(self.on_speed_slider_changed)
        speed_input_layout.addWidget(self.speed_slider)

        layout.addLayout(speed_input_layout, 0, 1)
        self.speed_target_spin.valueChanged.connect(self.update_speed_slider)

        layout.addWidget(QLabel("加速度 (rad/s²):"), 1, 0)
        self.speed_acc_spin = QDoubleSpinBox()
        self.speed_acc_spin.setRange(0.1, 100)
        self.speed_acc_spin.setValue(10.0)
        self.speed_acc_spin.setDecimals(1)
        layout.addWidget(self.speed_acc_spin, 1, 1)

        layout.addWidget(QLabel("电流限制 (A):"), 2, 0)
        self.speed_cur_spin = QDoubleSpinBox()
        self.speed_cur_spin.setRange(0.1, 16.0)
        self.speed_cur_spin.setValue(5.0)
        self.speed_cur_spin.setDecimals(1)
        layout.addWidget(self.speed_cur_spin, 2, 1)

        return tab

    def create_current_tab(self):
        """电流模式参数"""
        tab = QWidget()
        layout = QGridLayout(tab)

        layout.addWidget(QLabel("目标电流 (A):"), 0, 0)
        current_input_layout = QHBoxLayout()

        self.current_target_spin = QDoubleSpinBox()
        self.current_target_spin.setRange(-16.0, 16.0)
        self.current_target_spin.setValue(0.0)
        self.current_target_spin.setDecimals(2)
        current_input_layout.addWidget(self.current_target_spin)

        self.current_slider = QSlider(Qt.Horizontal)
        self.current_slider.setRange(-1600, 1600)
        self.current_slider.setValue(0)
        self.current_slider.valueChanged.connect(self.on_current_slider_changed)
        current_input_layout.addWidget(self.current_slider)

        layout.addLayout(current_input_layout, 0, 1)
        self.current_target_spin.valueChanged.connect(self.update_current_slider)

        note = QLabel("正值正转，负值反转\n提示: 切换模式前会自动获取当前位置")
        note.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(note, 1, 0, 1, 2)

        return tab

    def create_jog_tab(self):
        """JOG模式参数"""
        tab = QWidget()
        layout = QGridLayout(tab)

        layout.addWidget(QLabel("JOG 速度 (%):"), 0, 0)
        self.jog_speed_slider = QSlider(Qt.Horizontal)
        self.jog_speed_slider.setRange(0, 100)
        self.jog_speed_slider.setValue(20)
        layout.addWidget(self.jog_speed_slider, 0, 1)
        self.jog_speed_label = QLabel("20%")
        layout.addWidget(self.jog_speed_label, 0, 2)

        self.jog_speed_slider.valueChanged.connect(
            lambda v: self.jog_speed_label.setText(f"{v}%")
        )

        jog_btn_layout = QHBoxLayout()
        self.jog_cw_btn = QPushButton("JOG + (正转)")
        self.jog_cw_btn.pressed.connect(lambda: self.on_jog_clicked("CW"))
        self.jog_cw_btn.released.connect(self.on_jog_released)
        jog_btn_layout.addWidget(self.jog_cw_btn)

        self.jog_ccw_btn = QPushButton("JOG - (反转)")
        self.jog_ccw_btn.pressed.connect(lambda: self.on_jog_clicked("CCW"))
        self.jog_ccw_btn.released.connect(self.on_jog_released)
        jog_btn_layout.addWidget(self.jog_ccw_btn)

        layout.addLayout(jog_btn_layout, 1, 0, 1, 3)

        note = QLabel("按住按钮运行，松开停止\n使用前请确保已使能电机")
        note.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(note, 2, 0, 1, 3)

        return tab

    def create_right_panel(self):
        """创建右侧面板（状态显示+滑条控制）"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # ===== 7个电机状态显示组 =====
        status_group = QGroupBox("7个关节状态")
        status_layout = QVBoxLayout()

        overall_layout = QHBoxLayout()
        overall_layout.addWidget(QLabel("连接状态:"))
        self.conn_status_label = QLabel("● 未连接")
        self.conn_status_label.setStyleSheet("color: red; font-weight: bold;")
        overall_layout.addWidget(self.conn_status_label)
        overall_layout.addWidget(QLabel("当前模式:"))
        self.current_mode_label = QLabel("无")
        overall_layout.addWidget(self.current_mode_label)
        overall_layout.addStretch()
        status_layout.addLayout(overall_layout)

        self.motor_status_frames = []
        self.motor_status_labels = []

        for i in range(self.NUM_MOTORS):
            frame = QFrame()
            frame.setFrameStyle(QFrame.StyledPanel)
            frame_layout = QHBoxLayout(frame)
            frame_layout.setContentsMargins(5, 2, 5, 2)

            num_label = QLabel(f"关节{i + 1}:")
            num_label.setFixedWidth(50)
            num_label.setStyleSheet("font-weight: bold;")
            frame_layout.addWidget(num_label)

            enabled_label = QLabel("●")
            enabled_label.setStyleSheet("color: gray;")
            enabled_label.setFixedWidth(20)
            frame_layout.addWidget(enabled_label)

            pos_label = QLabel("0.000")
            pos_label.setStyleSheet("color: #3498db; font-family: Consolas; font-size: 11px;")
            pos_label.setFixedWidth(55)
            frame_layout.addWidget(pos_label)

            vel_label = QLabel("0.00")
            vel_label.setStyleSheet("color: #27ae60; font-family: Consolas; font-size: 11px;")
            vel_label.setFixedWidth(45)
            frame_layout.addWidget(vel_label)

            torque_label = QLabel("0.00")
            torque_label.setStyleSheet("color: #e74c3c; font-family: Consolas; font-size: 11px;")
            torque_label.setFixedWidth(45)
            frame_layout.addWidget(torque_label)

            frame_layout.addStretch()
            status_layout.addWidget(frame)

            self.motor_status_frames.append(frame)
            self.motor_status_labels.append({
                "enabled": enabled_label,
                "pos": pos_label,
                "vel": vel_label,
                "torque": torque_label,
            })

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # ===== 7个电机位置控制组 =====
        position_group = QGroupBox("7个关节位置控制")
        position_layout = QVBoxLayout()

        control_mode_layout = QHBoxLayout()
        control_mode_layout.addWidget(QLabel("控制方式:"))
        self.control_mode_combo = QComboBox()
        self.control_mode_combo.addItems(["直接控制", "直线插值", "三次多项式"])
        self.control_mode_combo.setCurrentIndex(0)
        control_mode_layout.addWidget(self.control_mode_combo, 1)
        position_layout.addLayout(control_mode_layout)

        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("步长 (°):"))
        self.step_deg_label = QLabel("5.0")
        step_layout.addWidget(self.step_deg_label)
        step_layout.addWidget(QLabel("步长时间 (s):"))
        self.step_time_label = QLabel("1.0")
        step_layout.addWidget(self.step_time_label)
        step_layout.addStretch()
        position_layout.addLayout(step_layout)

        self.motor_sliders = []
        self.motor_minus_btns = []
        self.motor_plus_btns = []
        self.motor_pos_labels = []
        self.motor_slider_targets = [0.0] * self.NUM_MOTORS

        for i in range(self.NUM_MOTORS):
            motor_frame = QFrame()
            motor_frame.setFrameStyle(QFrame.StyledPanel)
            motor_layout = QHBoxLayout(motor_frame)
            motor_layout.setContentsMargins(5, 2, 5, 2)

            motor_label = QLabel(f"关节{i + 1}:")
            motor_label.setStyleSheet("font-weight: bold;")
            motor_label.setFixedWidth(50)
            motor_layout.addWidget(motor_label)

            minus_btn = QPushButton("−")
            minus_btn.setFixedSize(25, 25)
            minus_btn.setStyleSheet("""
                QPushButton { background-color: #e74c3c; color: white; border-radius: 12px; font-weight: bold; font-size: 14px; }
                QPushButton:hover { background-color: #c0392b; }
                QPushButton:pressed { background-color: #a93226; }
                QPushButton:disabled { background-color: #bdc3c7; }
            """)
            minus_btn.setEnabled(False)
            minus_btn.clicked.connect(lambda checked, idx=i: self.on_motor_step_minus(idx))
            motor_layout.addWidget(minus_btn)
            self.motor_minus_btns.append(minus_btn)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(-3000, 3000)
            slider.setValue(0)
            slider.setEnabled(False)
            slider.valueChanged.connect(lambda value, idx=i: self.on_motor_slider_changed(idx, value))
            motor_layout.addWidget(slider, 1)
            self.motor_sliders.append(slider)

            plus_btn = QPushButton("+")
            plus_btn.setFixedSize(25, 25)
            plus_btn.setStyleSheet("""
                QPushButton { background-color: #27ae60; color: white; border-radius: 12px; font-weight: bold; font-size: 14px; }
                QPushButton:hover { background-color: #219a52; }
                QPushButton:pressed { background-color: #1e8449; }
                QPushButton:disabled { background-color: #bdc3c7; }
            """)
            plus_btn.setEnabled(False)
            plus_btn.clicked.connect(lambda checked, idx=i: self.on_motor_step_plus(idx))
            motor_layout.addWidget(plus_btn)
            self.motor_plus_btns.append(plus_btn)

            pos_label = QLabel("0.000")
            pos_label.setFixedWidth(60)
            pos_label.setStyleSheet("font-family: Consolas; font-size: 11px;")
            motor_layout.addWidget(pos_label)
            self.motor_pos_labels.append(pos_label)

            position_layout.addWidget(motor_frame)

        self.slider_control_check = QCheckBox("开启滑条实时控制 (仅PP/CSP模式, 50Hz)")
        self.slider_control_check.setEnabled(False)
        self.slider_control_check.stateChanged.connect(self.on_slider_control_changed)
        position_layout.addWidget(self.slider_control_check)

        tracking_layout = QHBoxLayout()
        tracking_layout.addWidget(QLabel("跟踪速度 (rad/s):"))
        self.tracking_speed_spin = QDoubleSpinBox()
        self.tracking_speed_spin.setRange(0.1, 50.0)
        self.tracking_speed_spin.setValue(5.0)
        self.tracking_speed_spin.setDecimals(1)
        tracking_layout.addWidget(self.tracking_speed_spin)
        tracking_layout.addStretch()
        position_layout.addLayout(tracking_layout)

        pp_acc_layout = QHBoxLayout()
        pp_acc_layout.addWidget(QLabel("PP加速度 (rad/s²):"))
        self.pos_acc_spin = QDoubleSpinBox()
        self.pos_acc_spin.setRange(0.1, 100.0)
        self.pos_acc_spin.setValue(10.0)
        self.pos_acc_spin.setDecimals(1)
        pp_acc_layout.addWidget(self.pos_acc_spin)
        pp_acc_layout.addStretch()
        position_layout.addLayout(pp_acc_layout)

        position_group.setLayout(position_layout)
        layout.addWidget(position_group)

        return panel

    # ===== 事件处理方法 =====

    def on_connect_clicked(self):
        """连接按钮点击"""
        self.log("点击: 连接按钮 (需连接实际电机控制逻辑)")
        pass

    def on_settings_clicked(self):
        """设置按钮点击"""
        self.log("点击: 设置按钮")
        dialog = SystemSettingsDialog(self, self.system_config)

        if dialog.exec_() == QDialog.Accepted:
            new_config = dialog.get_settings()
            self.system_config.update(new_config)
            self.apply_system_settings()
            self.log("✓ 系统设置已更新")

    def apply_system_settings(self):
        """应用系统设置"""
        self.status_update_frequency = self.system_config.get(
            "status_update_frequency", 100
        )
        self.motion_control_frequency = self.system_config.get(
            "motion_control_frequency", 500
        )

        self.step_deg_label.setText(str(self.system_config.get("slider_step_deg", 5.0)))
        self.step_time_label.setText(str(self.system_config.get("slider_step_time", 1.0)))

        safety = self.system_config.get("safety", {})
        any_enabled = any(
            safety.get(f"joint{i + 1}", {}).get("enabled", False) for i in range(7)
        )
        if any_enabled:
            self.safety_status_label.setText("安全限制: 开启")
            self.safety_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.safety_status_label.setText("安全限制: 关闭")
            self.safety_status_label.setStyleSheet("color: gray;")

    def on_enable_clicked(self):
        """使能按钮点击"""
        self.log("点击: 使能按钮")
        self.update_connection_status(True)
        pass

    def on_disable_clicked(self):
        """失能按钮点击"""
        self.log("点击: 失能按钮")
        self.update_connection_status(False)
        pass

    def on_zero_clicked(self):
        """置零按钮点击"""
        self.log("点击: 置零按钮")
        pass

    def on_pp_home_clicked(self):
        """PP模式下所有关节回零"""
        self.log("点击: PP回零按钮")
        pass

    def on_clear_error_clicked(self):
        """清错按钮点击"""
        self.log("点击: 清错按钮")
        pass

    def on_mode_changed(self, index):
        """模式选择改变"""
        self.mode_tabs.setCurrentIndex(index)
        hints = [
            "提示: 运控模式需要持续发送指令，建议 Kp>0 用于位置跟踪",
            "提示: PP模式设置位置后会自动规划轨迹运行，可开启滑条控制",
            "提示: CSP模式设置位置后会立即跟踪，适合滑条控制",
            "提示: 速度模式启动后可用滑块实时调整速度",
            "提示: 电流模式直接控制力矩，滑块可实时调整",
            "提示: JOG模式按住按钮运行，松开停止",
        ]
        if 0 <= index < len(hints):
            self.mode_hint_label.setText(hints[index])

        if index in [1, 2]:
            self.slider_control_check.setEnabled(True)
            self.control_mode_combo.setEnabled(True)
            for i in range(self.NUM_MOTORS):
                self.motor_minus_btns[i].setEnabled(True)
                self.motor_plus_btns[i].setEnabled(True)
        else:
            self.slider_control_check.setChecked(False)
            self.slider_control_check.setEnabled(False)
            self.control_mode_combo.setEnabled(False)
            for i in range(self.NUM_MOTORS):
                self.motor_minus_btns[i].setEnabled(False)
                self.motor_plus_btns[i].setEnabled(False)

    def on_switch_mode_clicked(self):
        """切换模式按钮点击"""
        mode_index = self.mode_combo.currentIndex()
        mode_names = [
            "运控模式", "位置PP模式", "位置CSP模式",
            "速度模式", "电流模式", "JOG模式"
        ]
        self.log(f"点击: 切换模式按钮 → {mode_names[mode_index]}")
        pass

    def on_start_clicked(self):
        """启动运动按钮点击"""
        mode_index = self.mode_combo.currentIndex()
        mode_names = [
            "运控模式", "位置PP模式", "位置CSP模式",
            "速度模式", "电流模式", "JOG模式"
        ]
        self.log(f"点击: 启动运动 → {mode_names[mode_index]}")
        pass

    def on_stop_clicked(self):
        """停止运动按钮点击"""
        self.log("点击: 停止运动")
        pass

    def on_get_position_clicked(self, spin_box):
        """获取当前位置按钮点击"""
        self.log("点击: 获取当前位置")
        pass

    def on_speed_slider_changed(self):
        """速度滑块改变"""
        value = self.speed_slider.value() / 10.0
        self.speed_target_spin.setValue(value)
        self.log(f"速度滑块值改变: {value:.1f}")

    def update_speed_slider(self):
        """速度数值改变时更新滑块"""
        value = self.speed_target_spin.value()
        self.speed_slider.setValue(int(value * 10))

    def on_current_slider_changed(self):
        """电流滑块改变"""
        value = self.current_slider.value() / 100.0
        self.current_target_spin.setValue(value)
        self.log(f"电流滑块值改变: {value:.2f}")

    def update_current_slider(self):
        """电流数值改变时更新滑块"""
        value = self.current_target_spin.value()
        self.current_slider.setValue(int(value * 100))

    def on_jog_clicked(self, direction):
        """JOG按钮点击"""
        speed = self.jog_speed_slider.value()
        self.log(f"点击: JOG {direction} (速度 {speed}%)")

    def on_jog_released(self):
        """JOG按钮释放"""
        self.log("松开: JOG按钮")

    def on_slider_control_changed(self, state):
        """滑条控制开关改变"""
        is_enabled = (state == Qt.Checked)
        if is_enabled:
            current_mode_index = self.mode_combo.currentIndex()
            if current_mode_index not in [1, 2]:
                self.slider_control_check.setChecked(False)
                self.log("滑条控制仅在PP或CSP模式下可用")
                return
            self.log("开启: 滑条实时控制 (50Hz)")
            for i in range(self.NUM_MOTORS):
                self.motor_sliders[i].setEnabled(True)
                self.motor_minus_btns[i].setEnabled(True)
                self.motor_plus_btns[i].setEnabled(True)
        else:
            self.log("关闭: 滑条实时控制")
            for i in range(self.NUM_MOTORS):
                self.motor_sliders[i].setEnabled(False)

    def on_motor_slider_changed(self, motor_idx, value):
        """单个电机滑条值改变"""
        self.motor_slider_targets[motor_idx] = value / 1000.0
        self.motor_pos_labels[motor_idx].setText(f"{self.motor_slider_targets[motor_idx]:.3f}")

    def on_motor_step_plus(self, motor_idx):
        """单个电机加按钮点击"""
        step_deg = 5.0
        step_rad = math.radians(step_deg)
        current_value = self.motor_sliders[motor_idx].value()
        new_value = current_value + int(step_rad * 1000)
        new_value = max(-3000, min(3000, new_value))
        self.motor_sliders[motor_idx].setValue(new_value)
        self.motor_slider_targets[motor_idx] = new_value / 1000.0

    def on_motor_step_minus(self, motor_idx):
        """单个电机减按钮点击"""
        step_deg = 5.0
        step_rad = math.radians(step_deg)
        current_value = self.motor_sliders[motor_idx].value()
        new_value = current_value - int(step_rad * 1000)
        new_value = max(-3000, min(3000, new_value))
        self.motor_sliders[motor_idx].setValue(new_value)
        self.motor_slider_targets[motor_idx] = new_value / 1000.0

    def update_connection_status(self, connected):
        """更新连接状态显示"""
        if connected:
            self.conn_status_label.setText("● 已连接")
            self.conn_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.connect_btn.setText("断开")
            self.connect_btn.setObjectName("danger")

            self.enable_btn.setEnabled(True)
            self.disable_btn.setEnabled(True)
            self.zero_btn.setEnabled(True)
            self.clear_error_btn.setEnabled(True)
            self.pp_home_btn.setEnabled(True)
            self.mode_tabs.setEnabled(True)
            self.switch_mode_btn.setEnabled(True)

            current_mode_index = self.mode_combo.currentIndex()
            if current_mode_index in [1, 2]:
                self.slider_control_check.setEnabled(True)
            self.control_mode_combo.setEnabled(True)

            for i in range(self.NUM_MOTORS):
                self.motor_minus_btns[i].setEnabled(True)
                self.motor_plus_btns[i].setEnabled(True)
        else:
            self.conn_status_label.setText("● 未连接")
            self.conn_status_label.setStyleSheet("color: red; font-weight: bold;")
            self.connect_btn.setText("连接")
            self.connect_btn.setObjectName("success")

            self.enable_btn.setEnabled(False)
            self.disable_btn.setEnabled(False)
            self.zero_btn.setEnabled(False)
            self.clear_error_btn.setEnabled(False)
            self.pp_home_btn.setEnabled(False)
            self.mode_tabs.setEnabled(False)
            self.switch_mode_btn.setEnabled(False)
            self.slider_control_check.setEnabled(False)
            self.slider_control_check.setChecked(False)
            self.control_mode_combo.setEnabled(False)

            self.current_mode_label.setText("无")

            for i in range(self.NUM_MOTORS):
                self.motor_minus_btns[i].setEnabled(False)
                self.motor_plus_btns[i].setEnabled(False)
                self.motor_sliders[i].setEnabled(False)
                self.motor_status_labels[i]["enabled"].setText("●")
                self.motor_status_labels[i]["enabled"].setStyleSheet("color: gray;")
                self.motor_status_labels[i]["pos"].setText("0.000")
                self.motor_status_labels[i]["vel"].setText("0.00")
                self.motor_status_labels[i]["torque"].setText("0.00")

            self.enable_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    def log(self, message):
        """添加日志"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    window = MotorControlGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
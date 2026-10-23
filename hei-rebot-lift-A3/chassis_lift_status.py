# -*- coding: utf-8 -*-
"""
Qt5 底盘 + 升降 5 电机状态监控工具

- 底盘 4 轮 + 升降电机，共用一路 U2CAN 串口（默认 /dev/ttyACM1）
- 电机 CAN ID（SlaveID，接收ID）与 MasterID（反馈ID）：
    右前轮  CAN ID 0x11  MasterID 0x01
    右后轮  CAN ID 0x12  MasterID 0x02
    左后轮  CAN ID 0x13  MasterID 0x03
    左前轮  CAN ID 0x14  MasterID 0x04
    升降电机 CAN ID 0x15  MasterID 0x05

功能：
1. 串口选择（自动扫描）+ 波特率
2. 连接 / 断开
3. 使能 / 失能
4. 显示电机 ID、MasterID、控制模式、使能状态、位置、速度、力矩、错误码、温度、电机状态
5. 设置 / 读取每个电机的控制模式
"""

import sys
import time
import math
from contextlib import suppress

import serial
import serial.tools.list_ports
from PyQt6 import QtCore, QtWidgets

from control_widgets import JoystickWidget, HJoystickWidget
from lift_debug_dialog import (
    LimitSwitchReader,
    LIFT_UP_SIGN,
    LIFT_LEAD_MM_PER_REV,
    LIFT_POSITION_WRAP_RAD,
)

try:
    from DM_CAN import (
        MotorControl,
        Motor,
        DM_Motor_Type,
        Control_Type,
        DM_variable,
    )
    try:
        from DM_CAN import Motor_Status  # noqa: F401
    except ImportError:
        Motor_Status = None
    DM_CAN_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    DM_CAN_AVAILABLE = False
    DM_CAN_IMPORT_ERROR = str(_e)

from system_analyzer.settings_manager import load_settings
from system_analyzer.settings_dialog import SystemSettingsDialog

# 底盘 4 轮 + 升降电机。字段与 Motor(MotorType, SlaveID, MasterID) 对应：
# SlaveID = 电机 CAN ID（接收ID，使能/控制命令发往该 ID）；MasterID = 反馈ID（主机ID）。
MOTOR_DEFS = [
    {"name": "右前轮", "slave_id": 0x11, "master_id": 0x01},
    {"name": "右后轮", "slave_id": 0x12, "master_id": 0x02},
    {"name": "左后轮", "slave_id": 0x13, "master_id": 0x03},
    {"name": "左前轮", "slave_id": 0x14, "master_id": 0x04},
    {"name": "升降电机", "slave_id": 0x15, "master_id": 0x05},
]

# 控制模式下拉项 (value, 显示名)
MODE_OPTIONS = [
    ("MIT", "MIT (力矩)"),
    ("POS_VEL", "POS_VEL (位置-速度)"),
    ("VEL", "VEL (速度)"),
    ("Torque_Pos", "Torque_Pos (力位)"),
]
DEFAULT_MODE_NAME = "VEL"  # 底盘与升降默认速度模式

# 底盘麦轮运动学（参考 HEI ReBot Lift 底盘驱动，轮序：右前/右后/左后/左前）
WHEEL_KINEMATICS = [
    [-1.0, -1.0, -1.0],  # 右前轮 RF
    [-1.0, 1.0, -1.0],   # 右后轮 RR
    [1.0, 1.0, -1.0],    # 左后轮 LR
    [1.0, -1.0, -1.0],   # 左前轮 LF
]
LINEAR_SPEED_SCALE = 9.0   # 线速度(m/s) -> 轮速(rad/s)
YAW_SPEED_SCALE = 2.0      # 角速度(rad/s) -> 轮速(rad/s)
MAX_LINEAR_SPEED = 1.0     # 最大平移速度(m/s)
MAX_WHEEL_SPEED = 9.0      # 单轮最大速度(rad/s)
LIFT_MAX_RPM = 1000.0      # 新升降电机最高转速(转/分)，升降速度上限独立可调
# ── VR/云端外部指令底盘平滑（缓启缓停）──
# 轮速指令突变会使电机反电动势回充母线，电压冲击可导致工控机重启；
# 对外部速度指令施加加速度斜坡，限制加/减速度。
EXT_CMD_DT = 1.0 / 30.0        # 底盘命令循环周期 (s)，与 cmd_timer(33ms) 对应
CHASSIS_MAX_LIN_ACCEL = 0.3    # m/s² 外部指令线加速度上限
CHASSIS_MAX_ANG_ACCEL = 1.5    # rad/s² 外部指令角加速度上限
# 减速度单独放宽：起步缓（防回充冲击），松手/反向时快速停车（缩短停车延迟）
CHASSIS_MAX_LIN_DECEL = 1.0    # m/s² 外部指令线减速度上限
CHASSIS_MAX_ANG_DECEL = 3.0    # rad/s² 外部指令角减速度上限
# 位姿伺服（VR 头显位置跟随）专用斜坡：P 伺服随误差收敛自行限速、
# 目标随头部运动连续变化，不会像摇杆速度指令那样阶跃突变，可用更高的
# 加/减速度上限以缩短响应延迟（仍保留斜坡以防轮速突变回充母线）
CHASSIS_POSE_LIN_ACCEL = 0.8   # m/s² 位姿伺服线加速度上限
CHASSIS_POSE_LIN_DECEL = 2.0   # m/s² 位姿伺服线减速度上限
CHASSIS_POSE_ANG_ACCEL = 3.0   # rad/s² 位姿伺服角加速度上限
CHASSIS_POSE_ANG_DECEL = 5.0   # rad/s² 位姿伺服角减速度上限

COLS = [
    "名称",
    "CAN ID(SlaveID)",
    "MasterID",
    "控制模式",
    "使能",
    "位置(rad)",
    "速度(rad/s)",
    "力矩",
    "错误码",
    "MOS温度(℃)",
    "线圈温度(℃)",
    "电机状态",
]

# 苹果(macOS)风格界面样式
APPLE_QSS = """
* {
    font-family: "Helvetica Neue", "SF Pro Text", "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog {
    background-color: #F5F5F7;
}
QWidget {
    color: #1D1D1F;
}
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #D2D2D7;
    border-radius: 10px;
    margin-top: 14px;
    font-weight: 600;
    color: #1D1D1F;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 2px;
    padding: 0 6px;
    background-color: transparent;
    color: #6E6E73;
}
QPushButton {
    background-color: #FFFFFF;
    color: #1D1D1F;
    border: 1px solid #D2D2D7;
    border-radius: 7px;
    padding: 6px 14px;
    min-height: 22px;
}
QPushButton:hover {
    background-color: #EDEDF0;
}
QPushButton:pressed {
    background-color: #E3E3E6;
}
QPushButton:disabled {
    background-color: #F5F5F7;
    color: #AEAEB2;
    border-color: #E5E5EA;
}
QPushButton#mitBtn:checked {
    background-color: #34C759;
    color: #FFFFFF;
    border-color: #34C759;
}
QComboBox, QLineEdit, QPlainTextEdit {
    background-color: #FFFFFF;
    border: 1px solid #D2D2D7;
    border-radius: 6px;
    padding: 4px 8px;
}
QComboBox:focus, QLineEdit:focus {
    border-color: #0071E3;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QTableWidget {
    background-color: #FFFFFF;
    alternate-background-color: #FAFAFA;
    border: 1px solid #D2D2D7;
    border-radius: 8px;
    gridline-color: #E5E5EA;
}
QHeaderView::section {
    background-color: #F5F5F7;
    color: #6E6E73;
    border: none;
    border-bottom: 1px solid #D2D2D7;
    padding: 6px;
    font-weight: 600;
}
QPlainTextEdit {
    background-color: #FFFFFF;
    border: 1px solid #D2D2D7;
    border-radius: 8px;
    font-family: "Menlo", "Consolas", "Courier New", monospace;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #E5E5EA;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #0071E3;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 1px solid #D2D2D7;
    width: 18px;
    height: 18px;
    margin: -6px 0;
    border-radius: 9px;
}
QStatusBar {
    background-color: #F5F5F7;
    color: #6E6E73;
}
"""


def _mode_value(mode_name):
    """把 Control_Type 属性名转成整数值，若库不可用则回退到本地映射。"""
    if DM_CAN_AVAILABLE:
        attr = getattr(Control_Type, mode_name, None)
        if attr is not None:
            return int(attr)
    fallback = {"MIT": 1, "POS_VEL": 2, "VEL": 3, "Torque_Pos": 4}
    return fallback.get(mode_name, 3)


def _mode_name(value):
    """把整数值转成 Control_Type 属性名。"""
    if DM_CAN_AVAILABLE:
        for name in ("MIT", "POS_VEL", "VEL", "Torque_Pos"):
            if int(getattr(Control_Type, name)) == int(value):
                return name
    mapping = {1: "MIT", 2: "POS_VEL", 3: "VEL", 4: "Torque_Pos"}
    return mapping.get(int(value), "VEL")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("底盘 + 升降 5 电机状态监控")
        self.resize(760, 720)

        self.settings = load_settings()
        self.refresh_ms = int(self.settings["refresh_ms"])
        self.mit_interval_ms = int(self.settings["mit_interval_ms"])
        self.serial_timeout_s = float(self.settings["serial_timeout_s"])
        self.default_port = str(self.settings["default_port"])
        self.default_io_port = str(self.settings.get("default_io_port", ""))
        self.default_baud = int(self.settings["baud"])
        self.kp_default = float(self.settings["kp_default"])
        self.kd_default = float(self.settings["kd_default"])

        self.serial_dev = None
        self.mc = None
        self.motors = []  # list[Motor]
        self.connected = False
        self.enabled = False
        self._lift_debug_active = False
        self.lift_reader = None
        self._lift_last_raw_rad = None
        self._lift_turn_offset = 0.0

        # 底盘里程计（根据轮速积分，无 IMU）
        self.odom_x_mm = 0.0
        self.odom_y_mm = 0.0
        self.odom_yaw = 0.0
        self._last_odom_time = None
        # 机体系实测速度 (vx m/s 前, vy m/s 左, omega rad/s)，
        # 由 _update_odometry 轮速逆解更新，供 VLA 观测/数据采集
        # （里程计位姿无 IMU 漂移大且原点任意，不适合做训练标签）
        self._last_body_vel = (0.0, 0.0, 0.0)
        self._chassis_returning = False
        self._last_return_vx = 0.0
        self._last_return_vy = 0.0
        self._last_return_omega = 0.0
        self._return_accel_lin = 0.2   # m/s² 缓启缓停
        self._return_accel_ang = 0.8   # rad/s²

        # 外部控制（VR 摇杆 / 云端 VLA 推理）
        self._ext_vel = None           # (jx, jy, rot) 归一化 -1..1，body: x前 y左
        self._ext_vel_time = 0.0
        self._ext_vel_timeout_s = 1.0  # 外部速度指令保鲜期，超时自动失效
        self._ext_filt = [0.0, 0.0, 0.0]  # 斜坡滤波后的底盘速度 (vx, vy, omega)
        # 外部底盘位姿目标（VR 头显位置跟随）：里程计闭环 P 伺服 + 限速
        self._ext_pose = None          # (x_m, y_m, yaw_rad) 世界系目标位姿
        self._ext_pose_limits = (0.3, 1.0)  # (max_lin m/s, max_ang rad/s) 伺服限速
        self._ext_pose_filt = [0.0, 0.0, 0.0]  # 斜坡滤波后的下发速度 (vx, vy, omega)
        self._lift_target_mm = None    # 升降位置目标（外部伺服），None=跟随界面滑条
        self._lift_mm = None           # refresh_status 更新的最新升降位置
        self._lift_jog_dir = 0         # 全速点动方向：-1=上升 +1=下降 0=无
        self._lift_jog_until = 0.0     # 点动截止时间(time.monotonic)
        self._lift_jog_speed_rad = 0.0  # 点动转速 (rad/s)，_jog_lift 时写入
        self._ext_lift_vel_rad = None  # VR 外部升降速度 (rad/s)，None=未控制
        self._ext_lift_vel_time = 0.0  # 最近一次外部升降速度指令时间
        self._ext_lift_vel_timeout_s = 0.5  # 外部升降速度保鲜期，超时自动停车

        self._auto_retry_count = 0

        self.mit_mode_enabled = False
        self.mit_targets = []
        self.kp = self.kp_default
        self.kd = self.kd_default
        self.mit_timer = QtCore.QTimer()
        self.mit_timer.timeout.connect(self.mit_control_loop)
        self.mit_timer.setInterval(self.mit_interval_ms)

        self._build_ui()
        self.refresh_ports()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_status)
        self.timer.start(self.refresh_ms)

        # 底盘/升降控制命令循环（约 30Hz）
        self.cmd_timer = QtCore.QTimer()
        self.cmd_timer.timeout.connect(self._send_control_commands)
        self.cmd_timer.start(33)

        # 升降限位 IO 轮询（更新灯泡）
        self.lift_io_timer = QtCore.QTimer()
        self.lift_io_timer.timeout.connect(self._poll_lift_io)
        self.lift_io_timer.start(100)

        if not DM_CAN_AVAILABLE:
            self.log(f"警告：DM_CAN 导入失败：{DM_CAN_IMPORT_ERROR}", error=True)

        # 启动后自动连接并使能（设置中可开关，默认开启）
        if self.settings.get("auto_connect_enable", False):
            QtCore.QTimer.singleShot(800, self._auto_connect_and_enable)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(left)

        # 顶部工具栏
        top_bar = QtWidgets.QWidget()
        top_layout = QtWidgets.QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        title_lbl = QtWidgets.QLabel("底盘 + 升降 5 电机状态监控")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: 600; color: #1D1D1F;")
        top_layout.addWidget(title_lbl)
        top_layout.addStretch()
        self.settingsBtn = QtWidgets.QPushButton("设置")
        self.settingsBtn.clicked.connect(self.open_settings)
        top_layout.addWidget(self.settingsBtn)
        layout.addWidget(top_bar)

        # 串口连接区
        conn_group = QtWidgets.QGroupBox("串口连接")
        conn_layout = QtWidgets.QVBoxLayout()

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("串口:"))
        self.portCombo = QtWidgets.QComboBox()
        self.portCombo.setEditable(True)
        self.portCombo.setMinimumWidth(140)
        row1.addWidget(self.portCombo)

        self.refreshPortsBtn = QtWidgets.QPushButton("刷新")
        self.refreshPortsBtn.clicked.connect(self.refresh_ports)
        row1.addWidget(self.refreshPortsBtn)

        row1.addWidget(QtWidgets.QLabel("波特率:"))
        self.baudCombo = QtWidgets.QComboBox()
        self.baudCombo.addItems(["921600", "115200", "57600", "38400"])
        self.baudCombo.setCurrentText(str(self.default_baud))
        row1.addWidget(self.baudCombo)

        self.connectBtn = QtWidgets.QPushButton("连接")
        self.connectBtn.clicked.connect(self.connect_serial)
        row1.addWidget(self.connectBtn)

        self.disconnectBtn = QtWidgets.QPushButton("断开")
        self.disconnectBtn.setEnabled(False)
        self.disconnectBtn.clicked.connect(self.disconnect_serial)
        row1.addWidget(self.disconnectBtn)
        row1.addStretch()
        conn_layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("升降IO串口:"))
        self.ioPortCombo = QtWidgets.QComboBox()
        self.ioPortCombo.setEditable(True)
        self.ioPortCombo.setMinimumWidth(120)
        row2.addWidget(self.ioPortCombo)

        row2.addWidget(QtWidgets.QLabel("波特率:"))
        self.ioBaudCombo = QtWidgets.QComboBox()
        self.ioBaudCombo.addItems(["115200", "57600", "38400", "19200", "9600"])
        self.ioBaudCombo.setCurrentText("115200")
        row2.addWidget(self.ioBaudCombo)

        row2.addWidget(QtWidgets.QLabel("上限:"))
        self.upperBulb = QtWidgets.QLabel("上")
        self.upperBulb.setFixedSize(30, 30)
        self.upperBulb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        row2.addWidget(self.upperBulb)

        row2.addWidget(QtWidgets.QLabel("下限:"))
        self.lowerBulb = QtWidgets.QLabel("下")
        self.lowerBulb.setFixedSize(30, 30)
        self.lowerBulb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        row2.addWidget(self.lowerBulb)

        self._set_bulb_color(self.upperBulb, False)
        self._set_bulb_color(self.lowerBulb, False)

        row2.addStretch()
        conn_layout.addLayout(row2)

        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)

        # 电机控制区
        ctrl_group = QtWidgets.QGroupBox("电机控制")
        ctrl_layout = QtWidgets.QHBoxLayout()

        self.enableBtn = QtWidgets.QPushButton("使能")
        self.enableBtn.setEnabled(False)
        self.enableBtn.clicked.connect(self.enable_motors)
        ctrl_layout.addWidget(self.enableBtn)

        self.disableBtn = QtWidgets.QPushButton("失能")
        self.disableBtn.setEnabled(False)
        self.disableBtn.clicked.connect(self.disable_motors)
        ctrl_layout.addWidget(self.disableBtn)

        self.setModeBtn = QtWidgets.QPushButton("设置控制模式")
        self.setModeBtn.setEnabled(False)
        self.setModeBtn.clicked.connect(self.apply_control_modes)
        ctrl_layout.addWidget(self.setModeBtn)

        self.readModeBtn = QtWidgets.QPushButton("读取控制模式")
        self.readModeBtn.setEnabled(False)
        self.readModeBtn.clicked.connect(self.read_control_modes)
        ctrl_layout.addWidget(self.readModeBtn)

        self.stopBtn = QtWidgets.QPushButton("零速度停止")
        self.stopBtn.setEnabled(False)
        self.stopBtn.clicked.connect(self.stop_motors)
        ctrl_layout.addWidget(self.stopBtn)

        self.liftDebugBtn = QtWidgets.QPushButton("升降调试")
        self.liftDebugBtn.clicked.connect(self.open_lift_debug)
        ctrl_layout.addWidget(self.liftDebugBtn)

        self.connStateLbl = QtWidgets.QLabel("未连接")
        self.connStateLbl.setStyleSheet("color: gray; font-weight: bold;")
        ctrl_layout.addWidget(self.connStateLbl)

        ctrl_layout.addStretch()
        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        # MIT 模式区
        mit_group = QtWidgets.QGroupBox("MIT 模式")
        mit_layout = QtWidgets.QHBoxLayout()

        self.mitBtn = QtWidgets.QPushButton("开启MIT模式")
        self.mitBtn.setObjectName("mitBtn")
        self.mitBtn.setCheckable(True)
        self.mitBtn.setEnabled(False)
        self.mitBtn.toggled.connect(self.toggle_mit_mode)
        mit_layout.addWidget(self.mitBtn)

        mit_layout.addWidget(QtWidgets.QLabel("kp:"))
        self.kpSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.kpSlider.setRange(0, 500)  # 0.0 ~ 50.0 (×10)
        self.kpSlider.setValue(int(round(self.kp_default * 10)))
        self.kpSlider.setMinimumWidth(150)
        self.kpSlider.valueChanged.connect(self._on_kp_changed)
        mit_layout.addWidget(self.kpSlider)
        self.kpValueLbl = QtWidgets.QLabel(f"{self.kp_default:.1f}")
        self.kpValueLbl.setMinimumWidth(40)
        mit_layout.addWidget(self.kpValueLbl)

        mit_layout.addWidget(QtWidgets.QLabel("kd:"))
        self.kdSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.kdSlider.setRange(0, 150)  # 0.00 ~ 1.50 (×100)
        self.kdSlider.setValue(int(round(self.kd_default * 100)))
        self.kdSlider.setMinimumWidth(150)
        self.kdSlider.valueChanged.connect(self._on_kd_changed)
        mit_layout.addWidget(self.kdSlider)
        self.kdValueLbl = QtWidgets.QLabel(f"{self.kd_default:.2f}")
        self.kdValueLbl.setMinimumWidth(40)
        mit_layout.addWidget(self.kdValueLbl)

        self.mitDragCheck = QtWidgets.QCheckBox("拖拽模式")
        self.mitDragCheck.setToolTip(
            "勾选后 MIT 目标位置每周期跟随当前反馈位置，底盘可被外力拖动；"
            "取消勾选则回到位置保持。升降电机始终位置保持。"
        )
        mit_layout.addWidget(self.mitDragCheck)

        mit_layout.addStretch()
        mit_group.setLayout(mit_layout)
        layout.addWidget(mit_group)

        # 零点设置区
        zero_group = QtWidgets.QGroupBox("零点设置")
        zero_layout = QtWidgets.QHBoxLayout()
        zero_layout.addWidget(QtWidgets.QLabel("电机ID:"))
        self.motorIdCombo = QtWidgets.QComboBox()
        self.motorIdCombo.setEditable(True)
        self.motorIdCombo.setMinimumWidth(140)
        zero_layout.addWidget(self.motorIdCombo)

        self.scanMotorsBtn = QtWidgets.QPushButton("扫描电机")
        self.scanMotorsBtn.setEnabled(False)
        self.scanMotorsBtn.clicked.connect(self.scan_motors)
        zero_layout.addWidget(self.scanMotorsBtn)

        self.setZeroBtn = QtWidgets.QPushButton("设置零点(选中ID)")
        self.setZeroBtn.setEnabled(False)
        self.setZeroBtn.clicked.connect(self.set_zero_selected)
        zero_layout.addWidget(self.setZeroBtn)

        self.setZeroAllBtn = QtWidgets.QPushButton("所有电机设置零点")
        self.setZeroAllBtn.setEnabled(False)
        self.setZeroAllBtn.clicked.connect(self.set_zero_all)
        zero_layout.addWidget(self.setZeroAllBtn)

        zero_layout.addStretch()
        zero_group.setLayout(zero_layout)
        layout.addWidget(zero_group)

        # 状态表格
        table_group = QtWidgets.QGroupBox("电机状态")
        table_layout = QtWidgets.QVBoxLayout()
        self.table = QtWidgets.QTableWidget(len(MOTOR_DEFS), len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self.table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        # 日志区
        log_group = QtWidgets.QGroupBox("日志")
        log_layout = QtWidgets.QVBoxLayout()
        self.logEdit = QtWidgets.QPlainTextEdit()
        self.logEdit.setReadOnly(True)
        self.logEdit.setMaximumHeight(120)
        log_layout.addWidget(self.logEdit)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # 右侧控制面板（底盘摇杆 + 旋转 + 升降）
        right = self._build_control_panel()
        main_layout.addWidget(left, 1)
        main_layout.addWidget(right, 0)

        self.setCentralWidget(central)

        # 填充静态列并初始化控制模式下拉
        for row, mdef in enumerate(MOTOR_DEFS):
            name_item = QtWidgets.QTableWidgetItem(mdef["name"])
            self.table.setItem(row, 0, name_item)

            id_item = QtWidgets.QTableWidgetItem(f"0x{mdef['slave_id']:02X} ({mdef['slave_id']})")
            id_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, id_item)

            master_item = QtWidgets.QTableWidgetItem(f"0x{mdef['master_id']:02X} ({mdef['master_id']})")
            master_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, master_item)

            combo = QtWidgets.QComboBox()
            for name, label in MODE_OPTIONS:
                combo.addItem(label, _mode_value(name))
            idx = combo.findData(_mode_value(DEFAULT_MODE_NAME))
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.table.setCellWidget(row, 3, combo)

            for col in (4, 5, 6, 7, 8, 9, 10, 11):
                item = QtWidgets.QTableWidgetItem("-")
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)

    # --------------------------------------------------------------- 控制面板
    def _build_control_panel(self):
        panel = QtWidgets.QGroupBox("底盘 / 升降控制")
        v = QtWidgets.QVBoxLayout(panel)
        v.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # 二维摇杆（平移）
        self.joystick = JoystickWidget(180)
        joy_row = QtWidgets.QHBoxLayout()
        joy_row.addStretch()
        joy_row.addWidget(self.joystick)
        joy_row.addStretch()
        v.addLayout(joy_row)

        # 四个方向速度滑条（前后 / 左右 / 左斜角 / 右斜角）
        v.addWidget(QtWidgets.QLabel("方向速度"))
        self.fbSlider = self._add_dir_slider(v, "前后")
        self.lrSlider = self._add_dir_slider(v, "左右")
        self.ldSlider = self._add_dir_slider(v, "左斜角")
        self.rdSlider = self._add_dir_slider(v, "右斜角")

        # 旋转摇杆（顺/逆时针）
        v.addWidget(QtWidgets.QLabel("旋转（顺 / 逆时针）"))
        self.rotJoystick = HJoystickWidget()
        v.addWidget(self.rotJoystick)

        # 升降速度滑条 + 位置显示滑条
        lift_row = QtWidgets.QHBoxLayout()
        lift_row.addStretch()

        speed_col = QtWidgets.QVBoxLayout()
        speed_col.addWidget(QtWidgets.QLabel("速度"), 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.liftSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.liftSlider.setRange(-100, 100)
        self.liftSlider.setValue(0)
        self.liftSlider.setFixedHeight(160)
        self.liftSlider.sliderReleased.connect(self._lift_slider_released)
        speed_col.addWidget(self.liftSlider, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        lift_row.addLayout(speed_col)

        pos_col = QtWidgets.QVBoxLayout()
        pos_col.addWidget(QtWidgets.QLabel("位置"), 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.liftPosSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.liftPosSlider.setRange(-700, 0)  # 上=0mm，下=-700mm
        self.liftPosSlider.setValue(0)
        self.liftPosSlider.setFixedHeight(160)
        self.liftPosSlider.setEnabled(False)  # 仅显示位置，不可拖动
        pos_col.addWidget(self.liftPosSlider, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        lift_row.addLayout(pos_col)

        lift_row.addStretch()
        v.addLayout(lift_row)

        self.liftPosLbl = QtWidgets.QLabel("位置: 0.0 mm")
        self.liftPosLbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.liftPosLbl)

        # 升降归零按钮
        self.homeLiftBtn = QtWidgets.QPushButton("升降归零")
        self.homeLiftBtn.clicked.connect(self._home_lift)
        v.addWidget(self.homeLiftBtn)

        # 升降 XYZ 位置 / 笛卡尔姿态
        self.liftPoseLbl = QtWidgets.QLabel(
            "XYZ: (0.0, 0.0, 0.0) mm\n姿态: R 0.0°  P 0.0°  Y 0.0°"
        )
        self.liftPoseLbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.liftPoseLbl)

        # 里程计复位 / 底盘回零
        odom_row = QtWidgets.QHBoxLayout()
        self.resetOdomBtn = QtWidgets.QPushButton("里程计复位")
        self.resetOdomBtn.clicked.connect(self._reset_odometry)
        odom_row.addWidget(self.resetOdomBtn)
        self.returnZeroBtn = QtWidgets.QPushButton("底盘回零")
        self.returnZeroBtn.clicked.connect(self._start_chassis_return)
        odom_row.addWidget(self.returnZeroBtn)
        v.addLayout(odom_row)

        # 回零速度设置
        ret_row = QtWidgets.QHBoxLayout()
        ret_row.addWidget(QtWidgets.QLabel("回零速度:"))
        self.returnSpeedSpin = QtWidgets.QDoubleSpinBox()
        self.returnSpeedSpin.setRange(0.05, 1.0)
        self.returnSpeedSpin.setDecimals(2)
        self.returnSpeedSpin.setValue(0.1)
        self.returnSpeedSpin.setSuffix(" m/s")
        ret_row.addWidget(self.returnSpeedSpin)
        ret_row.addStretch()
        v.addLayout(ret_row)

        # 最大速度阈值（同时作用于旋转与平移）
        speed_row = QtWidgets.QHBoxLayout()
        speed_row.addWidget(QtWidgets.QLabel("最大速度:"))
        self.maxAngularSpin = QtWidgets.QDoubleSpinBox()
        self.maxAngularSpin.setRange(10.0, 720.0)
        self.maxAngularSpin.setDecimals(0)
        self.maxAngularSpin.setValue(180.0)
        self.maxAngularSpin.setSuffix(" °/s")
        speed_row.addWidget(self.maxAngularSpin)
        speed_row.addStretch()
        v.addLayout(speed_row)

        # 升降速度上限（独立限制，新电机最高 1000 转/分）
        lift_speed_row = QtWidgets.QHBoxLayout()
        lift_speed_row.addWidget(QtWidgets.QLabel("升降速度上限:"))
        self.liftMaxSpin = QtWidgets.QDoubleSpinBox()
        self.liftMaxSpin.setRange(1.0, LIFT_MAX_RPM)
        self.liftMaxSpin.setDecimals(0)
        self.liftMaxSpin.setValue(LIFT_MAX_RPM)
        self.liftMaxSpin.setSuffix(" 转/分")
        lift_speed_row.addWidget(self.liftMaxSpin)
        lift_speed_row.addStretch()
        v.addLayout(lift_speed_row)

        # 全速点动：以最大速度上/下运动 1 秒
        jog_row = QtWidgets.QHBoxLayout()
        self.jogUpBtn = QtWidgets.QPushButton("全速上升 1s")
        self.jogUpBtn.clicked.connect(lambda: self._jog_lift(-1))
        jog_row.addWidget(self.jogUpBtn)
        self.jogDownBtn = QtWidgets.QPushButton("全速下降 1s")
        self.jogDownBtn.clicked.connect(lambda: self._jog_lift(1))
        jog_row.addWidget(self.jogDownBtn)
        v.addLayout(jog_row)

        tip = QtWidgets.QLabel("需先「使能」电机，摇杆/滑条才生效。最大速度同时限制旋转与平移。")
        tip.setStyleSheet("color: #6E6E73;")
        tip.setWordWrap(True)
        v.addWidget(tip)

        v.addStretch()
        return panel

    def _add_dir_slider(self, v, label):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label)
        lbl.setMinimumWidth(48)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(-100, 100)
        slider.setValue(0)
        slider.sliderReleased.connect(lambda s=slider: s.setValue(0))
        row.addWidget(lbl)
        row.addWidget(slider)
        v.addLayout(row)
        return slider

    def _lift_slider_released(self):
        self.liftSlider.setValue(0)

    def _lift_max_speed_rad(self):
        """升降速度上限（rad/s），由界面独立的转/分设置换算。"""
        return float(self.liftMaxSpin.value()) * 2.0 * math.pi / 60.0

    def _jog_lift(self, direction):
        """以最大速度点动升降 1 秒：direction -1=上升 +1=下降（电机负速度=向上）。"""
        if not self.connected or self.mc is None or not self.enabled:
            QtWidgets.QMessageBox.warning(self, "警告", "请先连接串口并使能电机")
            return
        self._lift_jog_dir = int(direction)
        self._lift_jog_until = time.monotonic() + 1.0
        self._lift_jog_speed_rad = self._lift_max_speed_rad()
        self.log("升降全速点动：%s 1s" % ("上升" if direction < 0 else "下降"))

    def _motor_rad_to_mm(self, rad):
        return LIFT_UP_SIGN * rad / (2.0 * math.pi) * LIFT_LEAD_MM_PER_REV

    def _lift_total_rad(self, raw_rad):
        """多圈累计，避免电机位置在 ±wrap/2 处跳变（换算关系与测试模块一致）。"""
        raw_rad = float(raw_rad)
        if self._lift_last_raw_rad is None:
            self._lift_last_raw_rad = raw_rad
            return raw_rad
        delta = raw_rad - self._lift_last_raw_rad
        half = LIFT_POSITION_WRAP_RAD * 0.5
        if delta > half:
            self._lift_turn_offset -= LIFT_POSITION_WRAP_RAD
        elif delta < -half:
            self._lift_turn_offset += LIFT_POSITION_WRAP_RAD
        self._lift_last_raw_rad = raw_rad
        return raw_rad + self._lift_turn_offset

    def _update_odometry(self):
        """根据四轮实测速度做里程计积分（无 IMU）。"""
        if len(self.motors) < 4:
            return
        now = time.monotonic()
        if self._last_odom_time is None:
            self._last_odom_time = now
            return
        dt = now - self._last_odom_time
        self._last_odom_time = now
        if dt <= 0.0 or dt > 1.0:
            return
        try:
            w_rf = float(self.motors[0].getVelocity())
            w_rr = float(self.motors[1].getVelocity())
            w_lr = float(self.motors[2].getVelocity())
            w_lf = float(self.motors[3].getVelocity())
        except Exception:
            return
        # 轮速 -> 缩放后的机体速度（O 型麦轮逆解）
        sx = (w_lr - w_rf - w_rr + w_lf) / 4.0
        sy = (w_lr - w_rf + w_rr - w_lf) / 4.0
        st = -(w_rf + w_rr + w_lr + w_lf) / 4.0
        vx = sx / LINEAR_SPEED_SCALE   # m/s（机体系前进方向）
        vy = sy / LINEAR_SPEED_SCALE   # m/s（机体系左方向）
        omega = st / YAW_SPEED_SCALE   # rad/s
        self._last_body_vel = (vx, vy, omega)
        yaw = self.odom_yaw
        self.odom_yaw += omega * dt
        # 转到世界系并积分（mm）
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        vx_mm_s = vx * 1000.0
        vy_mm_s = vy * 1000.0
        self.odom_x_mm += (vx_mm_s * cos_y - vy_mm_s * sin_y) * dt
        self.odom_y_mm += (vx_mm_s * sin_y + vy_mm_s * cos_y) * dt

    def _reset_odometry(self):
        self.odom_x_mm = 0.0
        self.odom_y_mm = 0.0
        self.odom_yaw = 0.0
        self._last_odom_time = None
        self.log("里程计已复位")

    def _start_chassis_return(self):
        if not self.connected or self.mc is None or not self.enabled:
            QtWidgets.QMessageBox.warning(self, "警告", "请先连接串口并使能电机")
            return
        # 复位速度平滑状态，从 0 开始缓启动
        self._last_return_vx = 0.0
        self._last_return_vy = 0.0
        self._last_return_omega = 0.0
        self._chassis_returning = True
        self.log("底盘回零中...")

    def _ramp(self, current, target, max_delta):
        if target - current > max_delta:
            return current + max_delta
        if current - target > max_delta:
            return current - max_delta
        return target

    def _normalize_angle(self, a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _chassis_return_control(self):
        """返回 (vx, vy, omega, done)，单位 m/s、m/s、rad/s。"""
        x_m = self.odom_x_mm / 1000.0
        y_m = self.odom_y_mm / 1000.0
        yaw = self.odom_yaw

        ex_w = -x_m
        ey_w = -y_m
        dist = math.hypot(ex_w, ey_w)
        yaw_err = self._normalize_angle(-yaw)

        dist_tol = 0.02      # 2 cm
        yaw_tol = 0.05       # ~2.9°
        if dist < dist_tol and abs(yaw_err) < yaw_tol:
            return 0.0, 0.0, 0.0, True

        return_speed = float(self.returnSpeedSpin.value())

        # 位置控制：比例 + 限速；误差很小时只调姿态，避免来回摆动
        vx = 0.0
        vy = 0.0
        if dist >= dist_tol:
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            ex_b = ex_w * cos_y + ey_w * sin_y
            ey_b = -ex_w * sin_y + ey_w * cos_y
            kp = 3.0
            vx = kp * ex_b
            vy = kp * ey_b
            mag = math.hypot(vx, vy)
            if mag > return_speed:
                vx = vx / mag * return_speed
                vy = vy / mag * return_speed

        # 姿态控制：比例 + 限幅，死区内不转
        if abs(yaw_err) < yaw_tol:
            omega = 0.0
        else:
            omega = 1.5 * yaw_err
            omega = max(-0.8, min(0.8, omega))

        return vx, vy, omega, False

    def _body_to_wheel_vel(self, vx, vy, omega):
        sx = vx * LINEAR_SPEED_SCALE
        sy = vy * LINEAR_SPEED_SCALE
        st = omega * YAW_SPEED_SCALE
        wheel = []
        for row in WHEEL_KINEMATICS:
            wheel.append(row[0] * sx + row[1] * sy + row[2] * st)
        max_abs = max(abs(w) for w in wheel)
        if max_abs > MAX_WHEEL_SPEED:
            wheel = [w * MAX_WHEEL_SPEED / max_abs for w in wheel]
        return wheel

    # --------------------------------------------------------------- 底盘/升降命令
    def _send_control_commands(self):
        if not self.connected or self.mc is None or not self.enabled:
            return
        if self.mit_mode_enabled:
            return  # MIT 模式下由 MIT 循环接管，不发送底盘速度

        # 底盘回零模式
        if self._chassis_returning:
            vx_t, vy_t, omega_t, done = self._chassis_return_control()
            if done:
                # 已到位：目标速度设为 0，缓慢减速停止
                vx_t = 0.0
                vy_t = 0.0
                omega_t = 0.0

            dt = 1.0 / 30.0
            max_dlin = self._return_accel_lin * dt
            max_dang = self._return_accel_ang * dt
            self._last_return_vx = self._ramp(self._last_return_vx, vx_t, max_dlin)
            self._last_return_vy = self._ramp(self._last_return_vy, vy_t, max_dlin)
            self._last_return_omega = self._ramp(self._last_return_omega, omega_t, max_dang)

            wheel = self._body_to_wheel_vel(
                self._last_return_vx, self._last_return_vy, self._last_return_omega
            )
            for motor, spd in zip(self.motors[:4], wheel):
                with suppress(Exception):
                    self.mc.control_Vel(motor, float(spd))

            # 到位且速度已降到接近 0 才算真正完成
            if (
                done
                and abs(self._last_return_vx) < 1e-3
                and abs(self._last_return_vy) < 1e-3
                and abs(self._last_return_omega) < 1e-3
            ):
                self._chassis_returning = False
                for motor in self.motors[:4]:
                    with suppress(Exception):
                        self.mc.control_Vel(motor, 0.0)
                self.odom_x_mm = 0.0
                self.odom_y_mm = 0.0
                self.odom_yaw = 0.0
                self._last_odom_time = None
                self.log("底盘回零完成")
            # 回零期间升降保持不动
            if len(self.motors) > 4 and not self._lift_debug_active:
                with suppress(Exception):
                    self.mc.control_Vel(self.motors[4], 0.0)
            return

        # 外部底盘位姿目标（VR 头显位置跟随）：里程计闭环 P 伺服 + 限速 + 斜坡。
        # 优先于外部速度指令（摇杆），头显位置跟随期间接管底盘平移/旋转。
        if self._ext_pose is not None:
            x_m = self.odom_x_mm / 1000.0
            y_m = self.odom_y_mm / 1000.0
            yaw = self.odom_yaw
            tx, ty, tyaw = self._ext_pose
            max_lin, max_ang = self._ext_pose_limits

            ex_w = tx - x_m
            ey_w = ty - y_m
            yaw_err = self._normalize_angle(tyaw - yaw)
            dist = math.hypot(ex_w, ey_w)

            pos_tol = 0.01     # 1 cm
            yaw_tol = 0.03     # ~1.7°
            if dist < pos_tol and abs(yaw_err) < yaw_tol:
                vx_t = vy_t = omega_t = 0.0
            else:
                # 世界系误差 → 机体系速度（P 控制 + 限速）
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                kp = 3.0
                vx_t = kp * (ex_w * cos_y + ey_w * sin_y)
                vy_t = kp * (-ex_w * sin_y + ey_w * cos_y)
                mag = math.hypot(vx_t, vy_t)
                if mag > max_lin:
                    vx_t = vx_t / mag * max_lin
                    vy_t = vy_t / mag * max_lin
                omega_t = 1.5 * yaw_err
                omega_t = max(-max_ang, min(max_ang, omega_t))

            dt = 1.0 / 30.0
            dlin_acc = CHASSIS_POSE_LIN_ACCEL * dt
            dlin_dec = CHASSIS_POSE_LIN_DECEL * dt
            dang_acc = CHASSIS_POSE_ANG_ACCEL * dt
            dang_dec = CHASSIS_POSE_ANG_DECEL * dt
            targets = (vx_t, vy_t, omega_t)
            rates = ((dlin_acc, dlin_dec), (dlin_acc, dlin_dec), (dang_acc, dang_dec))
            for i in range(3):
                cur, tgt = self._ext_pose_filt[i], targets[i]
                acc_d, dec_d = rates[i]
                step = dec_d if (abs(tgt) < abs(cur) or tgt * cur < 0.0) else acc_d
                self._ext_pose_filt[i] = self._ramp(cur, tgt, step)
            wheel_speeds = self._body_to_wheel_vel(*self._ext_pose_filt)
            for motor, spd in zip(self.motors[:4], wheel_speeds):
                with suppress(Exception):
                    self.mc.control_Vel(motor, float(spd))
            if len(self.motors) > 4 and not self._lift_debug_active:
                with suppress(Exception):
                    self.mc.control_Vel(self.motors[4],
                                        self._lift_command_velocity(0.0))
            return

        # 外部速度指令（VR 摇杆 / 云端推理）优先：保鲜期内接管底盘。
        # 目标速度先经加速度斜坡滤波再下发（缓启缓停），避免轮速突变导致
        # 电机反电动势回充母线、电压冲击使工控机重启；指令失效后沿斜坡平滑停车。
        ext_fresh = (self._ext_vel is not None
                     and (time.monotonic() - self._ext_vel_time) < self._ext_vel_timeout_s)
        if ext_fresh or any(abs(v) > 1e-3 for v in self._ext_filt):
            if ext_fresh:
                jx_e, jy_e, rot_e = self._ext_vel
            else:
                jx_e = jy_e = rot_e = 0.0
            max_angular = math.radians(self.maxAngularSpin.value())
            max_linear = MAX_LINEAR_SPEED * (self.maxAngularSpin.value() / 180.0)
            dlin_acc = CHASSIS_MAX_LIN_ACCEL * EXT_CMD_DT
            dlin_dec = CHASSIS_MAX_LIN_DECEL * EXT_CMD_DT
            dang_acc = CHASSIS_MAX_ANG_ACCEL * EXT_CMD_DT
            dang_dec = CHASSIS_MAX_ANG_DECEL * EXT_CMD_DT
            targets = (jx_e * max_linear, jy_e * max_linear, rot_e * max_angular)
            rates = ((dlin_acc, dlin_dec), (dlin_acc, dlin_dec), (dang_acc, dang_dec))
            for i in range(3):
                cur, tgt = self._ext_filt[i], targets[i]
                acc_d, dec_d = rates[i]
                # 目标使速度绝对值减小（或反向）时用减速度斜坡，松手快速停车
                step = dec_d if (abs(tgt) < abs(cur) or tgt * cur < 0.0) else acc_d
                self._ext_filt[i] = self._ramp(cur, tgt, step)
            wheel_speeds = self._body_to_wheel_vel(*self._ext_filt)
            for motor, spd in zip(self.motors[:4], wheel_speeds):
                with suppress(Exception):
                    self.mc.control_Vel(motor, float(spd))
            if len(self.motors) > 4 and not self._lift_debug_active:
                with suppress(Exception):
                    self.mc.control_Vel(self.motors[4],
                                        self._lift_command_velocity(0.0))
            return

        # 摇杆：上=前进，右=右移（body: x=前后, y 左为正）
        jx_body = self.joystick.y()
        jy_body = -self.joystick.x()

        # 四个方向速度滑条（-1..1）
        fb = self.fbSlider.value() / 100.0   # 前后：+前进
        lr = self.lrSlider.value() / 100.0   # 左右：+右移
        ld = self.ldSlider.value() / 100.0   # 左斜角：+前左
        rd = self.rdSlider.value() / 100.0   # 右斜角：+前右
        inv2 = 1.0 / math.sqrt(2.0)
        jx_body += fb + (ld + rd) * inv2
        jy_body += -lr + (ld - rd) * inv2

        # 平移幅值钳制到 [-1, 1]
        mag = math.hypot(jx_body, jy_body)
        if mag > 1.0:
            jx_body /= mag
            jy_body /= mag

        rot = self.rotJoystick.value()
        lift = self.liftSlider.value() / 100.0

        max_angular = math.radians(self.maxAngularSpin.value())
        # 最大速度阈值同样作用于平移：按相对 180°/s 的比例缩放线速度
        max_linear = MAX_LINEAR_SPEED * (self.maxAngularSpin.value() / 180.0)

        wheel_speeds = self._body_to_wheel(jx_body, jy_body, rot, max_angular, max_linear)
        for motor, spd in zip(self.motors[:4], wheel_speeds):
            with suppress(Exception):
                self.mc.control_Vel(motor, float(spd))
        # 升降电机（升降调试对话框打开时，升降电机由对话框接管）
        if len(self.motors) > 4 and not self._lift_debug_active:
            gui_vel = -float(lift * self._lift_max_speed_rad())  # 滑块向上=上升（负速度=向上）
            lift_vel = self._lift_command_velocity(gui_vel)
            with suppress(Exception):
                self.mc.control_Vel(self.motors[4], lift_vel)

    def _body_to_wheel(self, jx, jy, rot, max_angular, max_linear=None):
        if max_linear is None:
            max_linear = MAX_LINEAR_SPEED
        sx = jx * max_linear * LINEAR_SPEED_SCALE
        sy = jy * max_linear * LINEAR_SPEED_SCALE
        st = rot * max_angular * YAW_SPEED_SCALE
        wheel = []
        for row in WHEEL_KINEMATICS:
            wheel.append(row[0] * sx + row[1] * sy + row[2] * st)
        max_abs = max(abs(w) for w in wheel)
        if max_abs > MAX_WHEEL_SPEED:
            wheel = [w * MAX_WHEEL_SPEED / max_abs for w in wheel]
        return wheel

    # ------------------------------------------------- 外部控制接口（VR / 云端推理）
    def set_external_direction_pads(self, fb, lr, ld=0.0, rd=0.0, rot=0.0):
        """外部方向指令（VR 摇杆）：语义与界面四个方向滑条完全一致。
        fb 前后(+前进)、lr 左右(+右移)、ld 左斜角(+前左)、rd 右斜角(+前右)，
        均 -1..1；rot 与旋转摇杆同向。内部按滑条合成公式换算为底盘速度，
        走与 set_external_velocity 相同的加速度斜坡通道。"""
        inv2 = 1.0 / math.sqrt(2.0)
        jx = float(fb) + (float(ld) + float(rd)) * inv2
        jy = -float(lr) + (float(ld) - float(rd)) * inv2
        mag = math.hypot(jx, jy)
        if mag > 1.0:
            jx /= mag
            jy /= mag
        self.set_external_velocity(jx, jy, rot)

    def set_external_velocity(self, jx, jy, rot=0.0):
        """归一化底盘速度指令 (-1..1)：jx 前进+，jy 左移+，rot 与旋转摇杆同向。"""
        self._ext_vel = (float(jx), float(jy), float(rot))
        self._ext_vel_time = time.monotonic()

    def set_external_velocity_si(self, vx, vy, omega=0.0):
        """SI 单位底盘速度指令：vx/vy (m/s，body: x前 y左)，omega (rad/s)。"""
        max_angular = math.radians(float(self.maxAngularSpin.value()))
        max_linear = MAX_LINEAR_SPEED * (float(self.maxAngularSpin.value()) / 180.0)

        def _norm(v, m):
            return max(-1.0, min(1.0, v / m)) if m > 1e-9 else 0.0

        self.set_external_velocity(
            _norm(float(vx), max_linear),
            _norm(float(vy), max_linear),
            _norm(float(omega), max_angular),
        )

    def clear_external_velocity(self):
        self._ext_vel = None

    def set_external_pose_target(self, x_m, y_m, yaw_rad, max_lin=0.3, max_ang=1.0):
        """外部底盘位姿目标（VR 头显位置跟随）：世界系 (x, y, yaw)，由里程计闭环
        P 伺服追踪，max_lin/max_ang 为伺服线/角速度上限。"""
        self._ext_pose = (float(x_m), float(y_m), float(yaw_rad))
        self._ext_pose_limits = (float(max_lin), float(max_ang))

    def clear_external_pose_target(self):
        self._ext_pose = None

    def set_lift_position_mm(self, mm):
        """云端推理：升降位置伺服目标（mm，0=上限位，负值向下）。"""
        self._lift_target_mm = max(-700.0, min(0.0, float(mm)))

    def clear_lift_target(self):
        self._lift_target_mm = None

    def set_lift_velocity(self, direction, speed_rpm=200.0):
        """VR 升降连续速度控制：direction -1=上升 +1=下降，0=停止；
        speed_rpm 转速（转/分）。按住期间由 VR 面板持续下发以续期，
        松开/断连后指令保鲜期超时自动停车；内部限位开关保护不变。"""
        if not self.connected or self.mc is None or not self.enabled:
            return
        self._lift_target_mm = None            # 清除位置伺服目标，避免与速度控制冲突
        if int(direction) == 0 or float(speed_rpm) <= 0.0:
            self._ext_lift_vel_rad = 0.0
        else:
            self._ext_lift_vel_rad = int(direction) * float(speed_rpm) * 2.0 * math.pi / 60.0
        self._ext_lift_vel_time = time.monotonic()

    def get_lift_mm(self):
        return float(self._lift_mm) if self._lift_mm is not None else 0.0

    def get_odom_pose(self):
        """返回里程计世界系位姿 (x_m, y_m, yaw_rad)，供 VR 头显位置跟随
        伺服使用（训练数据采集请用 get_mobile_state，语义为速度）。"""
        return (
            self.odom_x_mm / 1000.0,
            self.odom_y_mm / 1000.0,
            float(self.odom_yaw),
        )

    def get_mobile_state(self):
        """返回 (lift_m, vx, vy, omega)，供 VLA 观测/数据采集。

        底盘为机体系实测速度（m/s, m/s, rad/s，由轮速逆解）：
        里程计位姿无 IMU 漂移大、原点随连接重置任意，不能跨 episode
        泛化，故训练标签用速度；升降为绝对位置（米，多圈跟踪无漂移）。
        """
        vx, vy, omega = self._last_body_vel
        return (
            self.get_lift_mm() / 1000.0,
            float(vx),
            float(vy),
            float(omega),
        )

    def _lift_command_velocity(self, gui_vel):
        """升降电机速度指令：VR 外部速度 > 点动 > 外部位置伺服 > 界面速度。"""
        if (self._ext_lift_vel_rad is not None
                and (time.monotonic() - self._ext_lift_vel_time) < self._ext_lift_vel_timeout_s):
            vel = self._ext_lift_vel_rad
        elif self._lift_jog_dir != 0 and time.monotonic() < self._lift_jog_until:
            vel = self._lift_jog_dir * self._lift_jog_speed_rad
        elif self._lift_target_mm is not None:
            if self._lift_mm is None:
                vel = 0.0
            else:
                err = float(self._lift_target_mm) - float(self._lift_mm)
                if abs(err) <= 1.5:
                    vel = 0.0
                else:
                    v_mm_s = max(-80.0, min(80.0, 2.5 * err))
                    vel = v_mm_s * (2.0 * math.pi) / (LIFT_UP_SIGN * LIFT_LEAD_MM_PER_REV)
                    max_rad = self._lift_max_speed_rad()
                    vel = max(-max_rad, min(max_rad, vel))
        else:
            vel = gui_vel
        # 限位保护：到限位后同方向不能动，反方向可离开
        if self.lift_reader is not None:
            if self.lift_reader.both_active:
                return 0.0
            if self.lift_reader.upper_active and vel * LIFT_UP_SIGN > 0:
                return 0.0
            if self.lift_reader.lower_active and vel * LIFT_UP_SIGN < 0:
                return 0.0
        return vel

    # --------------------------------------------------------------- 端口
    def refresh_ports(self):
        ports = sorted({p.device for p in serial.tools.list_ports.comports()})
        current = self.portCombo.currentText()
        self.portCombo.clear()
        self.portCombo.addItems(ports)
        if self.default_port in ports:
            self.portCombo.setCurrentText(self.default_port)
        elif current:
            self.portCombo.setCurrentText(current)

        # 升降限位 IO 串口下拉框（默认 default_io_port）
        io_current = self.ioPortCombo.currentText()
        self.ioPortCombo.clear()
        self.ioPortCombo.addItems(ports)
        if self.default_io_port in ports:
            self.ioPortCombo.setCurrentText(self.default_io_port)
        elif io_current and io_current in ports:
            self.ioPortCombo.setCurrentText(io_current)
        self.log(f"扫描到串口: {ports if ports else '无'}")

    # ------------------------------------------------------------- 连接
    def connect_serial(self, show_error=True):
        if self.connected:
            return
        self.motors = []
        self.enabled = False
        self.mit_mode_enabled = False
        port = self.portCombo.currentText().strip()
        if not port:
            if show_error:
                QtWidgets.QMessageBox.warning(self, "警告", "请选择串口")
            return
        baud = int(self.baudCombo.currentText())
        try:
            # 1) 连接底盘主串口
            self.serial_dev = serial.Serial(port, baud, timeout=self.serial_timeout_s)
            self.mc = MotorControl(self.serial_dev)

            for mdef in MOTOR_DEFS:
                motor = Motor(
                    MotorType=DM_Motor_Type.DM4310,
                    SlaveID=mdef["slave_id"],
                    MasterID=mdef["master_id"],
                )
                self.motors.append(motor)
                self.mc.addMotor(motor)

            # 2) 同时连接升降限位 IO 串口
            self._connect_lift_io()

            self.connected = True
            self.connectBtn.setEnabled(False)
            self.disconnectBtn.setEnabled(True)
            self.enableBtn.setEnabled(True)
            self.setModeBtn.setEnabled(True)
            self.readModeBtn.setEnabled(True)
            self.stopBtn.setEnabled(True)
            self.scanMotorsBtn.setEnabled(True)
            self.setZeroBtn.setEnabled(True)
            self.setZeroAllBtn.setEnabled(True)
            self.mitBtn.setEnabled(True)

            # 先用已知电机 ID 填充下拉框，再自动扫描串口上的电机
            self._populate_motor_id_combo([mdef["slave_id"] for mdef in MOTOR_DEFS])
            self.scan_motors()
            self.auto_ensure_vel_mode()

            # 复位升降位置多圈跟踪与底盘里程计
            self._lift_last_raw_rad = None
            self._lift_turn_offset = 0.0
            self.odom_x_mm = 0.0
            self.odom_y_mm = 0.0
            self.odom_yaw = 0.0
            self._last_odom_time = None
            self._last_body_vel = (0.0, 0.0, 0.0)

            # 串口文本框背景变绿
            self._set_port_combo_style(self.portCombo, True)
            self._set_port_combo_style(self.ioPortCombo, True)

            self.connStateLbl.setText(f"已连接 {port}")
            self.connStateLbl.setStyleSheet("color: green; font-weight: bold;")
            self.log(
                f"连接成功: 主串口 {port} @ {baud}，升降IO {self.ioPortCombo.currentText()} "
                f"@ {self.ioBaudCombo.currentText()}，已注册 {len(self.motors)} 个电机"
            )
        except Exception as e:
            self._cleanup_serial()
            self.disconnect_lift_io()
            self._set_port_combo_style(self.portCombo, False)
            self._set_port_combo_style(self.ioPortCombo, False)
            self.log(f"连接失败: {e}", error=True)
            if show_error:
                QtWidgets.QMessageBox.critical(self, "错误", f"连接失败：{e}")

    def disconnect_serial(self):
        if not self.connected:
            return
        self._stop_mit_mode()
        if self.enabled:
            self._send_zero_velocity()
            self.log("断开前已发送零速度命令")
            time.sleep(0.05)
            self._disable_all_motors()
            self.enabled = False
        self._cleanup_serial()
        self.disconnect_lift_io()
        self._set_port_combo_style(self.portCombo, False)
        self._set_port_combo_style(self.ioPortCombo, False)
        self.connected = False
        self.enabled = False
        self.motors = []

        self.connectBtn.setEnabled(True)
        self.disconnectBtn.setEnabled(False)
        self.enableBtn.setEnabled(False)
        self.disableBtn.setEnabled(False)
        self.setModeBtn.setEnabled(False)
        self.readModeBtn.setEnabled(False)
        self.stopBtn.setEnabled(False)
        self.scanMotorsBtn.setEnabled(False)
        self.setZeroBtn.setEnabled(False)
        self.setZeroAllBtn.setEnabled(False)
        self.mitBtn.setEnabled(False)

        self.motorIdCombo.clear()
        self.connStateLbl.setText("未连接")
        self.connStateLbl.setStyleSheet("color: gray; font-weight: bold;")
        self._reset_table()
        self.log("已断开连接")

    def _send_zero_velocity(self):
        if self.mc is None:
            return
        for motor in self.motors:
            with suppress(Exception):
                self.mc.control_Vel(motor, 0.0)

    def _disable_all_motors(self):
        if self.mc is None:
            return
        for motor in self.motors:
            with suppress(Exception):
                self.mc.disable(motor)

    def _cleanup_serial(self):
        self._disable_all_motors()
        with suppress(Exception):
            if self.serial_dev is not None and self.serial_dev.is_open:
                self.serial_dev.close()
        self.mc = None
        self.serial_dev = None

    # ------------------------------------------------------------ 使能
    def enable_motors(self):
        if not self.connected:
            return
        vel_value = int(Control_Type.VEL)
        # 读取各电机当前控制模式
        modes = []
        for i, motor in enumerate(self.motors):
            try:
                m = self.mc.read_motor_param(motor, DM_variable.CTRL_MODE)
            except Exception:
                m = None
            modes.append((motor, MOTOR_DEFS[i]["name"], m))

        if all(m is not None and int(m) == vel_value for _, _, m in modes):
            self._do_enable()
            return

        # 非速度模式：弹窗选择使能方式
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("非速度模式")
        box.setText("检测到电机处于非速度模式，请选择使能方式：")
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        keep_btn = box.addButton("保持MIT模式", QtWidgets.QMessageBox.ButtonRole.NoRole)
        switch_btn = box.addButton("切换到速度模式", QtWidgets.QMessageBox.ButtonRole.YesRole)
        box.addButton("取消", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(keep_btn)
        box.exec()
        clicked = box.clickedButton()

        if clicked == switch_btn:
            # 切换到速度模式 -> 发送0速度 -> 使能
            for motor, name, m in modes:
                if m is None or int(m) != vel_value:
                    if self._switch_mode_with_retry(motor, Control_Type.VEL):
                        self.log(f"{name}: 已切换到速度模式")
                    else:
                        self.log(f"{name}: 切换到速度模式失败", error=True)
            self._send_zero_velocity()
            self.log("已发送零速度命令")
            self._do_enable()
        elif clicked == keep_btn:
            # 保持 MIT 模式 -> 发送当前位置/0速度/0力矩 -> 使能
            targets = []
            for motor, name, m in modes:
                q = 0.0
                try:
                    self.mc.refresh_motor_status(motor)
                    q = float(motor.getPosition())
                except Exception:
                    pass
                targets.append(q)
            for (motor, name, m), q in zip(modes, targets):
                try:
                    self.mc.controlMIT(motor, self._kp_value(), self._kd_value(), q, 0.0, 0.0)
                except Exception as e:
                    self.log(f"{name}: 发送 MIT 指令失败 {e}", error=True)
            self._do_enable()
            self.mit_targets = targets
            self.mit_mode_enabled = True
            self.mit_timer.start()
            self.mitBtn.blockSignals(True)
            self.mitBtn.setChecked(True)
            self.mitBtn.blockSignals(False)
            self.log("已在 MIT 模式下使能")
        else:
            self.log("已取消使能")

    def _do_enable(self):
        try:
            for i, motor in enumerate(self.motors):
                self.mc.enable(motor)
                self.log(f"使能 {MOTOR_DEFS[i]['name']} (ID {MOTOR_DEFS[i]['slave_id']})")
                time.sleep(0.02)
            self.enabled = True
            self.enableBtn.setEnabled(False)
            self.disableBtn.setEnabled(True)
            self.log("所有电机已使能")
        except Exception as e:
            self.log(f"使能失败: {e}", error=True)
            QtWidgets.QMessageBox.critical(self, "错误", f"使能失败：{e}")

    def _auto_connect_and_enable(self):
        """启动后自动连接并使能（无人工弹窗干预，带重试）。"""
        if self.connected:
            return
        if self._auto_retry_count == 0:
            self.log("自动连接（启动后自动连接已开启）...")
        self.connect_serial(show_error=False)
        if self.connected:
            self._auto_retry_count = 0
            try:
                self._send_zero_velocity()
                self._do_enable()
                self.log("自动使能完成")
            except Exception as e:
                self.log(f"自动使能失败: {e}", error=True)
            return
        self._auto_retry_count += 1
        if self._auto_retry_count < 5:
            self.log(
                f"自动连接失败，2 秒后重试 ({self._auto_retry_count}/5)...", error=True
            )
            QtCore.QTimer.singleShot(2000, self._auto_connect_and_enable)
        else:
            self._auto_retry_count = 0
            self.log("自动连接多次失败，请检查串口/电机", error=True)

    def disable_motors(self):
        if not self.connected:
            return
        self._stop_mit_mode()
        try:
            self._send_zero_velocity()
            self.log("失能前已发送零速度命令")
            time.sleep(0.05)
            for i, motor in enumerate(self.motors):
                self.mc.disable(motor)
                self.log(f"失能 {MOTOR_DEFS[i]['name']}")
                time.sleep(0.01)
            self.enabled = False
            self.enableBtn.setEnabled(True)
            self.disableBtn.setEnabled(False)
            self.log("所有电机已失能")
        except Exception as e:
            self.log(f"失能失败: {e}", error=True)

    # ------------------------------------------------------- 控制模式
    def _selected_mode_value(self, row):
        combo = self.table.cellWidget(row, 3)
        if combo is None:
            return _mode_value(DEFAULT_MODE_NAME)
        return int(combo.currentData())

    def apply_control_modes(self):
        if not self.connected:
            return
        for row, motor in enumerate(self.motors):
            mode = self._selected_mode_value(row)
            name = MOTOR_DEFS[row]["name"]
            try:
                ok = self.mc.switchControlMode(motor, mode)
                self.log(f"{name}: 设置控制模式 {_mode_name(mode)} -> {'成功' if ok else '失败'}")
            except Exception as e:
                self.log(f"{name}: 设置控制模式失败 {e}", error=True)

    def read_control_modes(self):
        if not self.connected:
            return
        for row, motor in enumerate(self.motors):
            name = MOTOR_DEFS[row]["name"]
            try:
                value = self.mc.read_motor_param(motor, DM_variable.CTRL_MODE)
                if value is None:
                    self.log(f"{name}: 未读取到控制模式")
                    continue
                combo = self.table.cellWidget(row, 3)
                idx = combo.findData(int(value))
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                self.log(f"{name}: 实际控制模式 = {_mode_name(value)} ({value})")
            except Exception as e:
                self.log(f"{name}: 读取控制模式失败 {e}", error=True)

    # ------------------------------------------------------- MIT 模式
    def _switch_mode_with_retry(self, motor, target_mode, max_attempts=3):
        """切换控制模式并校验，失败自动重试，最多 max_attempts 次。返回是否成功。"""
        target = int(target_mode)
        for _ in range(max_attempts):
            try:
                self.mc.switchControlMode(motor, target)
            except Exception:
                pass
            try:
                actual = self.mc.read_motor_param(motor, DM_variable.CTRL_MODE)
            except Exception:
                actual = None
            if actual is not None and int(actual) == target:
                return True
            time.sleep(0.05)
        return False

    def auto_ensure_vel_mode(self):
        """连接成功后自动读取各电机控制模式，非速度模式则切换到速度模式。"""
        if not self.connected or self.mc is None:
            return
        vel_value = int(Control_Type.VEL)
        for row, motor in enumerate(self.motors):
            name = MOTOR_DEFS[row]["name"]
            try:
                value = self.mc.read_motor_param(motor, DM_variable.CTRL_MODE)
            except Exception as e:
                self.log(f"{name}: 读取控制模式失败 {e}", error=True)
                value = None
            if value is not None and int(value) == vel_value:
                self.log(f"{name}: 已为速度模式")
            else:
                if self._switch_mode_with_retry(motor, Control_Type.VEL):
                    self.log(f"{name}: 已切换到速度模式")
                else:
                    self.log(f"{name}: 切换到速度模式失败(已重试3次)", error=True)
            combo = self.table.cellWidget(row, 3)
            idx = combo.findData(vel_value)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def toggle_mit_mode(self, checked):
        if not self.connected:
            return
        if checked:
            if not self.enabled:
                QtWidgets.QMessageBox.warning(self, "警告", "请先使能电机再开启 MIT 模式")
                self._reset_mit_btn()
                return
            self.enter_mit_mode()
        else:
            self.exit_mit_mode()

    def _kp_value(self):
        return self.kpSlider.value() / 10.0

    def _kd_value(self):
        return self.kdSlider.value() / 100.0

    def _on_kp_changed(self, v):
        self.kpValueLbl.setText(f"{v / 10.0:.1f}")

    def _on_kd_changed(self, v):
        self.kdValueLbl.setText(f"{v / 100.0:.2f}")

    def enter_mit_mode(self):
        if not self.connected or self.mc is None:
            return
        self.kp = self._kp_value()
        self.kd = self._kd_value()
        self.mit_targets = []
        for i, motor in enumerate(self.motors):
            q = 0.0
            try:
                self.mc.refresh_motor_status(motor)
                q = float(motor.getPosition())
            except Exception as e:
                self.log(f"{MOTOR_DEFS[i]['name']}: 读取当前位置失败 {e}", error=True)
            self.mit_targets.append(q)
        # 切换到 MIT 并校验，失败自动重试（最多3次）
        all_ok = True
        for i, motor in enumerate(self.motors):
            if self._switch_mode_with_retry(motor, Control_Type.MIT):
                self.log(f"{MOTOR_DEFS[i]['name']}: 已切换到 MIT 模式")
            else:
                self.log(f"{MOTOR_DEFS[i]['name']}: 切换到 MIT 失败(已重试3次)", error=True)
                all_ok = False
        if not all_ok:
            QtWidgets.QMessageBox.warning(self, "警告", "部分电机切换 MIT 失败，已中止开启")
            self._reset_mit_btn()
            return
        # 全部确认 MIT 后，再下发目标位置 + 0 速度 + 0 力矩
        for i, motor in enumerate(self.motors):
            try:
                self.mc.controlMIT(motor, self.kp, self.kd, self.mit_targets[i], 0.0, 0.0)
            except Exception as e:
                self.log(f"{MOTOR_DEFS[i]['name']}: 发送 MIT 指令失败 {e}", error=True)
        self.mit_mode_enabled = True
        self.mit_timer.start()
        self.log(f"MIT 模式已开启 (kp={self.kp}, kd={self.kd})")

    def exit_mit_mode(self):
        self.mit_timer.stop()
        self.mit_mode_enabled = False
        if self.connected and self.mc is not None:
            # 切换到速度模式并校验，失败自动重试（最多3次）
            all_ok = True
            for i, motor in enumerate(self.motors):
                if self._switch_mode_with_retry(motor, Control_Type.VEL):
                    self.log(f"{MOTOR_DEFS[i]['name']}: 已切回速度模式")
                else:
                    self.log(f"{MOTOR_DEFS[i]['name']}: 切回速度模式失败(已重试3次)", error=True)
                    all_ok = False
            if all_ok:
                self._send_zero_velocity()
                self.log("MIT 模式已关闭，已切换到速度模式并发送零速度")
            else:
                self.log("部分电机切回速度模式失败，未发送零速度", error=True)
        else:
            self.log("MIT 模式已关闭")

    def mit_control_loop(self):
        if not self.mit_mode_enabled or not self.connected or self.mc is None:
            return
        # 实时读取 kp/kd，MIT 模式下可随时修改并即时生效
        self.kp = self._kp_value()
        self.kd = self._kd_value()
        drag = self.mitDragCheck.isChecked()
        for i, motor in enumerate(self.motors):
            with suppress(Exception):
                if drag and i < 4:
                    self.mc.refresh_motor_status(motor)
                    self.mit_targets[i] = float(motor.getPosition())
                self.mc.controlMIT(motor, self.kp, self.kd, self.mit_targets[i], 0.0, 0.0)

    def _reset_mit_btn(self):
        if hasattr(self, "mitBtn"):
            self.mitBtn.blockSignals(True)
            self.mitBtn.setChecked(False)
            self.mitBtn.blockSignals(False)

    def _stop_mit_mode(self):
        self.mit_timer.stop()
        self.mit_mode_enabled = False
        self._reset_mit_btn()

    # ------------------------------------------------------- 零点设置
    def _populate_motor_id_combo(self, ids):
        self.motorIdCombo.clear()
        for sid in sorted(set(ids)):
            self.motorIdCombo.addItem(f"0x{sid:02X} ({sid})", int(sid))

    def scan_motors(self):
        if not self.connected or self.mc is None:
            return
        online = []
        offline = []
        for i, motor in enumerate(self.motors):
            name = MOTOR_DEFS[i]["name"]
            try:
                val = self.mc.read_motor_param(motor, DM_variable.ESC_ID)
            except Exception:
                val = None
            if val is not None:
                online.append(int(motor.SlaveID))
            else:
                offline.append(name)
        # 下拉框显示所有已注册电机 ID（自动遍历），供“设置零点”选择
        self._populate_motor_id_combo([mdef["slave_id"] for mdef in MOTOR_DEFS])
        if online:
            self.log(f"扫描完成，在线电机ID: {[hex(i) for i in online]}")
        else:
            self.log("扫描完成，未检测到在线电机", error=True)
        if offline:
            self.log(f"未响应电机: {', '.join(offline)}", error=True)

    def _motor_by_slave_id(self, slave_id):
        for motor in self.motors:
            if int(motor.SlaveID) == int(slave_id):
                return motor
        return None

    def set_zero_selected(self):
        if not self.connected or self.mc is None:
            return
        sid = self.motorIdCombo.currentData()
        if sid is None:
            text = self.motorIdCombo.currentText().strip()
            try:
                sid = int(text, 16) if text.lower().startswith("0x") else int(text)
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "警告", "请先选择或输入有效的电机ID")
                return
        sid = int(sid)
        motor = self._motor_by_slave_id(sid)
        if motor is None:
            motor = Motor(DM_Motor_Type.DM4310, sid, 0)
            self.mc.addMotor(motor)
        try:
            self.mc.set_zero_position(motor)
            self.log(f"电机 0x{sid:02X} 已设置当前位置为 0 点")
        except Exception as e:
            self.log(f"设置零点失败: {e}", error=True)
            QtWidgets.QMessageBox.critical(self, "错误", f"设置零点失败：{e}")

    def set_zero_all(self):
        if not self.connected or self.mc is None:
            return
        try:
            for i, motor in enumerate(self.motors):
                self.mc.set_zero_position(motor)
                self.log(f"设置零点 {MOTOR_DEFS[i]['name']} (ID 0x{motor.SlaveID:02X})")
                time.sleep(0.02)
            self.log("所有电机已设置当前位置为 0 点")
        except Exception as e:
            self.log(f"设置零点失败: {e}", error=True)
            QtWidgets.QMessageBox.critical(self, "错误", f"设置零点失败：{e}")

    # ----------------------------------------------------------- 停止
    def stop_motors(self):
        if not self.connected:
            return
        self._chassis_returning = False
        self.joystick.reset()
        self.rotJoystick.reset()
        self.liftSlider.setValue(0)
        self._send_zero_velocity()
        self.log("已发送零速度命令")

    # --------------------------------------------------------- 刷新
    def refresh_status(self):
        if not self.connected:
            return
        for row, motor in enumerate(self.motors):
            try:
                self.mc.refresh_motor_status(motor)
                online = True
            except Exception:
                online = False

            pos = self._safe(lambda: float(motor.getPosition()))
            vel = self._safe(lambda: float(motor.getVelocity()))
            tau = self._safe(lambda: float(motor.getTorque()))
            err = self._safe(lambda: int(motor.getMotorStatus()))
            t_mos = self._safe(lambda: motor.getT_MOS())
            t_rot = self._safe(lambda: motor.getT_Rotor())
            state_txt = self._safe(lambda: motor.getMotorStatusText())

            self.table.item(row, 4).setText("已使能" if (online and self.enabled) else "未使能")
            self.table.item(row, 5).setText(f"{pos:.3f}" if pos is not None else "-")
            self.table.item(row, 6).setText(f"{vel:.3f}" if vel is not None else "-")
            self.table.item(row, 7).setText(f"{tau:.3f}" if tau is not None else "-")
            self.table.item(row, 8).setText(str(err) if err is not None else "-")
            self.table.item(row, 9).setText(str(t_mos) if t_mos is not None else "-")
            self.table.item(row, 10).setText(str(t_rot) if t_rot is not None else "-")
            status = state_txt if state_txt else ("在线" if online else "离线")
            self.table.item(row, 11).setText(status)

        # 底盘里程计（根据轮速积分，无 IMU）
        self._update_odometry()

        # 升降位置显示（多圈跟踪 + 角度→mm 换算，与测试模块一致）
        if len(self.motors) > 4:
            with suppress(Exception):
                raw_rad = float(self.motors[4].getPosition())
                total_rad = self._lift_total_rad(raw_rad)
                mm = self._motor_rad_to_mm(total_rad)
                self._lift_mm = mm
                self.liftPosSlider.setValue(int(round(mm)))
                self.liftPosLbl.setText(f"位置: {mm:.1f} mm")
                self.liftPoseLbl.setText(
                    f"XYZ: ({self.odom_x_mm:.1f}, {self.odom_y_mm:.1f}, {mm:.1f}) mm\n"
                    f"姿态: R 0.0°  P 0.0°  Y {math.degrees(self.odom_yaw):.1f}°"
                )

    @staticmethod
    def _safe(fn):
        with suppress(Exception):
            return fn()
        return None

    def _reset_table(self):
        for row in range(len(MOTOR_DEFS)):
            for col in (4, 5, 6, 7, 8, 9, 10, 11):
                self.table.item(row, col).setText("-")

    # ----------------------------------------------------------- 日志
    def log(self, message, error=False):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        self.logEdit.appendPlainText(line)
        if error:
            self.statusBar().showMessage(message, 5000)

    # ------------------------------------------------------- 升降调试
    def _connect_lift_io(self):
        io_port = self.ioPortCombo.currentText().strip()
        if not io_port:
            raise RuntimeError("未选择升降限位IO串口")
        baud = int(self.ioBaudCombo.currentText())
        from lift_debug_dialog import LIFT_UPPER_BIT, LIFT_LOWER_BIT

        self.lift_reader = LimitSwitchReader(io_port, baud, LIFT_UPPER_BIT, LIFT_LOWER_BIT)

    def disconnect_lift_io(self):
        if self.lift_reader is not None:
            with suppress(Exception):
                self.lift_reader.close()
            self.lift_reader = None
        self._set_bulb_color(self.upperBulb, False)
        self._set_bulb_color(self.lowerBulb, False)

    def _set_port_combo_style(self, combo, connected):
        style = "color: #2E8B57;" if connected else ""
        combo.setStyleSheet(style)
        line_edit = combo.lineEdit()
        if line_edit is not None:
            line_edit.setStyleSheet(style)

    def _home_lift(self):
        if not self.connected or self.mc is None or len(self.motors) <= 4:
            QtWidgets.QMessageBox.warning(self, "警告", "请先连接串口")
            return
        if self.lift_reader is None:
            QtWidgets.QMessageBox.warning(self, "警告", "升降限位IO未连接")
            return
        from lift_debug_dialog import LiftController

        self.log("升降归零中（向上到上限位）...")
        QtWidgets.QApplication.processEvents()
        try:
            lift = LiftController(self.mc, self.motors[4], self.lift_reader)
            lift.home()
            # 归零后复位多圈跟踪，位置滑条显示 0
            self._lift_last_raw_rad = None
            self._lift_turn_offset = 0.0
            self.liftPosSlider.setValue(0)
            self.liftPosLbl.setText("位置: 0.0 mm")
            self.liftPoseLbl.setText(
                f"XYZ: ({self.odom_x_mm:.1f}, {self.odom_y_mm:.1f}, 0.0) mm\n"
                f"姿态: R 0.0°  P 0.0°  Y {math.degrees(self.odom_yaw):.1f}°"
            )
            self.log("升降归零完成")
        except Exception as e:
            self.log(f"升降归零失败: {e}", error=True)
            QtWidgets.QMessageBox.critical(self, "错误", f"升降归零失败：{e}")

    def _poll_lift_io(self):
        if self.lift_reader is None:
            return
        with suppress(Exception):
            self.lift_reader.poll()
        self._update_lift_bulbs()

    def _update_lift_bulbs(self):
        if self.lift_reader is None:
            return
        self._set_bulb_color(self.upperBulb, bool(self.lift_reader.upper_active))
        self._set_bulb_color(self.lowerBulb, bool(self.lift_reader.lower_active))

    def _set_bulb_color(self, bulb, active):
        color = "#FF3B30" if active else "#A9A9A9"
        bulb.setStyleSheet(
            f"background-color: {color}; border-radius: 15px; "
            f"color: white; font-weight: bold;"
        )

    def open_lift_debug(self):
        from lift_debug_dialog import LiftDebugDialog

        if not self.connected or self.mc is None or len(self.motors) <= 4:
            QtWidgets.QMessageBox.warning(self, "警告", "请先连接串口（升降电机在主串口上）")
            return
        if self.lift_reader is None:
            QtWidgets.QMessageBox.warning(self, "警告", "升降限位IO未连接，请重新连接串口")
            return
        self._lift_debug_active = True
        try:
            dlg = LiftDebugDialog(self.mc, self.motors[4], self.lift_reader, self)
            dlg.exec()
        finally:
            self._lift_debug_active = False

    # ------------------------------------------------------- 系统设置
    def open_settings(self):
        dlg = SystemSettingsDialog(self)
        dlg.exec()

    def apply_settings(self, settings):
        self.settings = settings
        self.refresh_ms = int(settings["refresh_ms"])
        self.mit_interval_ms = int(settings["mit_interval_ms"])
        self.serial_timeout_s = float(settings["serial_timeout_s"])
        self.default_port = str(settings["default_port"])
        self.default_io_port = str(settings.get("default_io_port", ""))
        self.default_baud = int(settings["baud"])
        self.kp_default = float(settings["kp_default"])
        self.kd_default = float(settings["kd_default"])
        # 频率立即生效
        self.timer.setInterval(self.refresh_ms)
        self.mit_timer.setInterval(self.mit_interval_ms)
        if self.serial_dev is not None:
            with suppress(Exception):
                self.serial_dev.timeout = self.serial_timeout_s
        self.log("设置已应用：频率已实时生效；串口/波特率/kp/kd 默认值重启后生效")

    # --------------------------------------------------------- 关闭
    def closeEvent(self, event):
        self.timer.stop()
        self.cmd_timer.stop()
        self.lift_io_timer.stop()
        self._stop_mit_mode()
        self.disconnect_lift_io()
        if self.connected:
            self.log("程序退出：执行安全停机")
            self.disconnect_serial()
        event.accept()


def create_embedded_panel():
    """供其他 Qt 程序嵌入使用：创建底盘+升降控制面板。

    返回 (window, widget)：window 为隐藏的 MainWindow（承载全部逻辑与定时器，
    调用方必须保留引用并在退出时调用 window.close()）；widget 为其中央部件，
    可直接放入 QDockWidget / 布局中。
    """
    win = MainWindow()
    widget = win.takeCentralWidget()
    return win, widget


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APPLE_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

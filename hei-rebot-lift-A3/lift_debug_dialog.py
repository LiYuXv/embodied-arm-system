# -*- coding: utf-8 -*-
"""
升降模块单独调试对话框（不依赖 lerobot，使用本地 DM_CAN + 自实现限位 IO 解析）。

功能：串口连接/断开、归零(home)、上移/下移位置控制、限位 IO 与电机状态显示。
"""

import math
import time
from contextlib import suppress

import serial
from PyQt6 import QtCore, QtWidgets

from DM_CAN import Control_Type

TWO_PI = 2.0 * math.pi

# 升降参数（参考 HEI ReBot Lift 配置；升降电机复用主串口，SlaveID 0x15）
LIFT_IO_BAUD = 115200
LIFT_UPPER_BIT = 1
LIFT_LOWER_BIT = 2
LIFT_UP_SIGN = -1.0
LIFT_LEAD_MM_PER_REV = 10.0
LIFT_POSITION_WRAP_RAD = 25.0
LIFT_HOME_SPEED_RAD_S = 30.0
LIFT_IO_WAIT_TIMEOUT_S = 3.0
LIFT_HOME_TIMEOUT_S = 30.0
LIFT_MAX_SPEED_RAD_S = 18.0
LIFT_MAX_ACCEL_RAD_S2 = 30.0
LIFT_POSITION_KP = 0.45
LIFT_POSITION_TOLERANCE_MM = 1.0
LIFT_MIN_HEIGHT_MM = -800.0
LIFT_MAX_HEIGHT_MM = 0.0


def _crc16_modbus(data):
    crc = 0xFFFF
    for value in data:
        crc ^= value
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


class LimitSwitchReader:
    """限位开关 IO 读取（Modbus 帧：01 04 02 00 state crc_lo crc_hi）。"""

    def __init__(self, port, baudrate, upper_bit, lower_bit):
        self.serial = serial.Serial(port, baudrate, timeout=0.0)
        self.upper_mask = 1 << (upper_bit - 1)
        self.lower_mask = 1 << (lower_bit - 1)
        self.buffer = bytearray()
        self.state = 0
        self.last_update_s = 0.0

    def close(self):
        self.serial.close()

    def poll(self):
        data = self.serial.read(256)
        if data:
            self.buffer.extend(data)
        self._parse_buffer()

    def _parse_buffer(self):
        while len(self.buffer) >= 7:
            start = self._find_frame_start()
            if start is None:
                self.buffer.clear()
                return
            if start > 0:
                del self.buffer[:start]
            if len(self.buffer) < 7:
                return
            frame = bytes(self.buffer[:7])
            del self.buffer[:7]
            if frame[0] != 0x01 or frame[1] != 0x04 or frame[2] != 0x02 or frame[3] != 0x00:
                continue
            frame_crc = frame[5] | (frame[6] << 8)
            if _crc16_modbus(frame[:5]) != frame_crc:
                continue
            if frame[4] not in (0x00, 0x01, 0x02, 0x03):
                continue
            self.state = frame[4]
            self.last_update_s = time.monotonic()

    def _find_frame_start(self):
        for index in range(len(self.buffer) - 3):
            if (
                self.buffer[index] == 0x01
                and self.buffer[index + 1] == 0x04
                and self.buffer[index + 2] == 0x02
                and self.buffer[index + 3] == 0x00
            ):
                return index
        return None

    @property
    def upper_active(self):
        return bool(self.state & self.upper_mask)

    @property
    def lower_active(self):
        return bool(self.state & self.lower_mask)

    @property
    def both_active(self):
        return self.upper_active and self.lower_active

    @property
    def online(self):
        return self.last_update_s > 0.0


class MultiTurnPositionTracker:
    """多圈位置跟踪，避免电机位置在 wrap_rad 处跳变。"""

    def __init__(self, wrap_rad):
        self.wrap_rad = float(wrap_rad)
        self.half_wrap_rad = self.wrap_rad * 0.5
        self.last_raw_rad = None
        self.turn_offset_rad = 0.0

    def reset(self, raw_rad=0.0):
        self.last_raw_rad = float(raw_rad)
        self.turn_offset_rad = -float(raw_rad)

    def update(self, raw_rad):
        raw_rad = float(raw_rad)
        if self.last_raw_rad is None:
            self.reset(raw_rad)
            return raw_rad + self.turn_offset_rad
        delta = raw_rad - self.last_raw_rad
        if delta > self.half_wrap_rad:
            self.turn_offset_rad -= self.wrap_rad
        elif delta < -self.half_wrap_rad:
            self.turn_offset_rad += self.wrap_rad
        self.last_raw_rad = raw_rad
        return raw_rad + self.turn_offset_rad


class LiftController:
    """升降控制器（本地 DM_CAN + 限位 IO），接口对齐 lerobot._LiftRuntime 的核心能力。"""

    def __init__(self, motor_control, motor, reader):
        self.motor_control = motor_control  # 主界面的 MotorControl
        self.motor = motor                  # 升降电机（主界面 motors[4]，SlaveID 0x15）
        self.reader = reader                # 主界面已连接的限位 IO reader
        self.tracker = MultiTurnPositionTracker(LIFT_POSITION_WRAP_RAD)
        self.target_height_mm = 0.0
        self.height_mm = 0.0
        self.last_velocity_rad_s = 0.0
        self.last_command_time_s = None

        # 确保升降电机处于 VEL 模式并已使能（限位 IO 由主界面连接）
        self.motor_control.switchControlMode(self.motor, Control_Type.VEL)
        self.motor_control.enable(self.motor)
        with suppress(Exception):
            self.motor_control.refresh_motor_status(self.motor)
        self.tracker.reset(float(self.motor.getPosition()))
        self.height_mm = self._motor_rad_to_mm(float(self.motor.getPosition()))
        self.target_height_mm = self.height_mm

    def _motor_rad_to_mm(self, rad):
        return LIFT_UP_SIGN * rad / TWO_PI * LIFT_LEAD_MM_PER_REV

    def _poll_height(self):
        # 限位 IO 由主界面周期轮询，这里只读取电机高度
        if self.motor_control is not None and self.motor is not None:
            with suppress(Exception):
                self.motor_control.refresh_motor_status(self.motor)
            raw_rad = float(self.motor.getPosition())
            self.height_mm = self._motor_rad_to_mm(self.tracker.update(raw_rad))
        return self.height_mm

    def home(self):
        if self.reader is None or self.motor_control is None or self.motor is None:
            return
        home_vel = LIFT_UP_SIGN * abs(LIFT_HOME_SPEED_RAD_S)
        io_wait_start = time.monotonic()
        while not self.reader.online and time.monotonic() - io_wait_start < LIFT_IO_WAIT_TIMEOUT_S:
            self.reader.poll()
            time.sleep(0.01)
        if not self.reader.online:
            self.motor_control.control_Vel(self.motor, 0.0)
            raise RuntimeError("未收到限位IO数据，无法归零。")

        homed = False
        upper_since = None  # 上限位连续触发的起始时间（去抖）
        start = time.monotonic()
        while time.monotonic() - start < LIFT_HOME_TIMEOUT_S:
            self.reader.poll()
            if not self.reader.online:
                self.motor_control.control_Vel(self.motor, 0.0)
                raise RuntimeError("归零过程中限位IO超时。")
            if self.reader.both_active:
                self.motor_control.control_Vel(self.motor, 0.0)
                upper_since = None
                time.sleep(0.01)
                continue
            if self.reader.upper_active:
                if upper_since is None:
                    upper_since = time.monotonic()
                elif time.monotonic() - upper_since >= 0.05:
                    # 上限位连续稳定 50ms 才判定到达，避免抖动/误触发导致过早停止
                    self.motor_control.control_Vel(self.motor, 0.0)
                    homed = True
                    break
            else:
                upper_since = None
            self.motor_control.control_Vel(self.motor, float(home_vel))
            time.sleep(0.01)
        if not homed:
            self.motor_control.control_Vel(self.motor, 0.0)
            raise RuntimeError("归零超时，未到达上限位。")

        self.motor_control.disable(self.motor)
        self.motor_control.set_zero_position(self.motor)
        self.motor_control.refresh_motor_status(self.motor)
        self.motor_control.enable(self.motor)
        self.tracker.reset(float(self.motor.getPosition()))
        self.height_mm = 0.0
        self.target_height_mm = 0.0
        self.last_velocity_rad_s = 0.0
        self.last_command_time_s = None

    def command_position(self, target_height_mm):
        if self.motor_control is None or self.motor is None:
            return self.height_mm
        self.target_height_mm = max(
            LIFT_MIN_HEIGHT_MM, min(LIFT_MAX_HEIGHT_MM, float(target_height_mm))
        )
        current = self._poll_height()
        error_mm = self.target_height_mm - current
        velocity_rad_s = 0.0
        if abs(error_mm) >= LIFT_POSITION_TOLERANCE_MM:
            velocity_rad_s = max(
                -LIFT_MAX_SPEED_RAD_S,
                min(LIFT_MAX_SPEED_RAD_S, LIFT_POSITION_KP * error_mm / LIFT_UP_SIGN),
            )
        if self.reader is not None:
            if self.reader.both_active:
                velocity_rad_s = 0.0
            elif self.reader.upper_active and velocity_rad_s * LIFT_UP_SIGN > 0:
                velocity_rad_s = 0.0
            elif self.reader.lower_active and velocity_rad_s * LIFT_UP_SIGN < 0:
                velocity_rad_s = 0.0
        now = time.monotonic()
        if self.last_command_time_s is None:
            limited_velocity_rad_s = velocity_rad_s
        else:
            dt = max(now - self.last_command_time_s, 0.0)
            max_delta = LIFT_MAX_ACCEL_RAD_S2 * dt
            limited_velocity_rad_s = self.last_velocity_rad_s + max(
                -max_delta, min(max_delta, velocity_rad_s - self.last_velocity_rad_s)
            )
        self.last_command_time_s = now
        self.last_velocity_rad_s = limited_velocity_rad_s
        self.motor_control.control_Vel(self.motor, float(limited_velocity_rad_s))
        return self.target_height_mm

    def read_height(self):
        return self._poll_height()

    def stop(self):
        if self.motor_control is not None and self.motor is not None:
            with suppress(Exception):
                self.motor_control.control_Vel(self.motor, 0.0)
        self.last_velocity_rad_s = 0.0
        self.last_command_time_s = None


class LiftDebugDialog(QtWidgets.QDialog):
    def __init__(self, motor_control, motor, reader, parent=None):
        super().__init__(parent)
        self.setWindowTitle("升降模块调试")
        self.resize(480, 500)

        self.motor_control = motor_control
        self.motor = motor
        self.reader = reader
        self.lift = LiftController(motor_control, motor, reader)
        self.target_height_mm = self.lift.target_height_mm

        self._build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "升降电机：复用主串口 (SlaveID 0x15)\n限位IO：由主界面「升降连接」提供"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #6E6E73;")
        layout.addWidget(info)

        ctrl_group = QtWidgets.QGroupBox("控制")
        ctrl = QtWidgets.QHBoxLayout(ctrl_group)
        self.homeBtn = QtWidgets.QPushButton("归零")
        self.homeBtn.clicked.connect(self._home)
        self.upBtn = QtWidgets.QPushButton("上移 ▲")
        self.upBtn.clicked.connect(self._up)
        self.downBtn = QtWidgets.QPushButton("下移 ▼")
        self.downBtn.clicked.connect(self._down)
        self.stopBtn = QtWidgets.QPushButton("停止")
        self.stopBtn.clicked.connect(self._stop)
        ctrl.addWidget(self.homeBtn)
        ctrl.addWidget(self.upBtn)
        ctrl.addWidget(self.downBtn)
        ctrl.addWidget(self.stopBtn)
        layout.addWidget(ctrl_group)

        step_row = QtWidgets.QHBoxLayout()
        step_row.addWidget(QtWidgets.QLabel("步进:"))
        self.stepSpin = QtWidgets.QDoubleSpinBox()
        self.stepSpin.setRange(0.5, 100.0)
        self.stepSpin.setValue(2.0)
        self.stepSpin.setSuffix(" mm")
        step_row.addWidget(self.stepSpin)
        step_row.addStretch()
        layout.addLayout(step_row)

        status_group = QtWidgets.QGroupBox("状态")
        status_form = QtWidgets.QFormLayout(status_group)
        self.heightLbl = QtWidgets.QLabel("-")
        self.targetLbl = QtWidgets.QLabel("-")
        self.errorLbl = QtWidgets.QLabel("-")
        self.ioLbl = QtWidgets.QLabel("-")
        self.motorLbl = QtWidgets.QLabel("-")
        status_form.addRow("当前高度(mm):", self.heightLbl)
        status_form.addRow("目标高度(mm):", self.targetLbl)
        status_form.addRow("位置误差(mm):", self.errorLbl)
        status_form.addRow("限位IO:", self.ioLbl)
        status_form.addRow("电机状态:", self.motorLbl)
        layout.addWidget(status_group)

        self.statusBarLbl = QtWidgets.QLabel("就绪")
        self.statusBarLbl.setStyleSheet("color: #6E6E73;")
        layout.addWidget(self.statusBarLbl)

    def _home(self):
        if self.lift is None:
            return
        self.statusBarLbl.setText("归零中（向上到上限位）...")
        QtWidgets.QApplication.processEvents()
        try:
            self.lift.home()
            with suppress(Exception):
                self.target_height_mm = float(self.lift.height_mm)
            self.statusBarLbl.setText("归零完成")
        except Exception as e:
            self.statusBarLbl.setText(f"归零失败: {e}")

    def _up(self):
        if self.lift is None:
            return
        self.target_height_mm = min(self.target_height_mm + self.stepSpin.value(), LIFT_MAX_HEIGHT_MM)

    def _down(self):
        if self.lift is None:
            return
        self.target_height_mm = max(self.target_height_mm - self.stepSpin.value(), LIFT_MIN_HEIGHT_MM)

    def _stop(self):
        if self.lift is None:
            return
        with suppress(Exception):
            self.lift.stop()
        with suppress(Exception):
            self.target_height_mm = float(self.lift.read_height())

    # ---------------------------------------------------------------- 周期刷新
    def _tick(self):
        if self.lift is None:
            return
        with suppress(Exception):
            self.lift.command_position(self.target_height_mm)
        self._update_status()

    def _update_status(self):
        lift = self.lift
        if lift is None:
            return
        height = float(lift.height_mm)
        self.heightLbl.setText(f"{height:.2f}")
        self.targetLbl.setText(f"{self.target_height_mm:.2f}")
        self.errorLbl.setText(f"{self.target_height_mm - height:.2f}")

        reader = lift.reader
        if reader is not None:
            io_online = bool(reader.online)
            upper = bool(reader.upper_active)
            lower = bool(reader.lower_active)
            both = bool(reader.both_active)
            io_text = (
                f"{'在线' if io_online else '离线'} | "
                f"上限:{'触发' if upper else '释放'} | "
                f"下限:{'触发' if lower else '释放'}"
            )
            if both:
                io_text += " | ⚠双限位"
        else:
            io_text = "无IO"
        self.ioLbl.setText(io_text)

        motor = lift.motor
        if motor is not None:
            pos = float(motor.getPosition())
            vel = float(motor.getVelocity())
            tau = float(motor.getTorque())
            self.motorLbl.setText(
                f"位置:{pos:.3f} rad | 速度:{vel:.3f} rad/s | 力矩:{tau:.3f} | "
                f"状态:{motor.getMotorStatusText()}"
            )
        else:
            self.motorLbl.setText("-")

    def closeEvent(self, event):
        self.timer.stop()
        if self.lift is not None:
            with suppress(Exception):
                self.lift.stop()
        event.accept()

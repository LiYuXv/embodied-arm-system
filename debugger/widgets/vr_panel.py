"""VR 控制面板：Quest 3 手柄连接 + 位姿监控 + 相对笛卡尔控臂

功能：
1. 通过 UDP (默认 5006) 接收 Quest 3 VR 手柄位姿/按键数据（vr_receiver）。
2. 实时显示 R/L 手柄末端位置和姿态（四元数）、扳机/握持值。
3. 相对笛卡尔控制：按住右手柄侧键 Grip 离合接合，手柄相对锚点的
   位移/旋转增量映射到机械臂末端；松开 Grip 冻结。缩放系数滑条默认 1.0。
   摇杆 Y 控制夹爪开合（上推=闭合，下拉=张开）。

控制逻辑与 openarm_mujoco_vr/mujoco_cartesian_control_fk.py 的 Relative 模式一致：
    pos_target = ee_anchor_pos + (vr_pos - vr_anchor_pos) * scale
    rot_target = ee_anchor_rot @ (vr_anchor_rot^-1 @ vr_rot)
"""

import sys
import time
import math
import threading
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QScrollArea, QFrame, QSpinBox, QSlider,
)
from PyQt6.QtCore import pyqtSignal, QTimer, Qt

from debugger.utils.i18n import tr
from debugger.utils.theme_manager import ThemeManager
from debugger.utils.style import SCENE_COLORS

from el_a3_sdk.data_types import ArmEndPose

# vr_receiver 位于 debugger/openarm_mujoco_vr/ 目录（非 Python 包），加入搜索路径
_VR_DIR = Path(__file__).resolve().parent.parent / "openarm_mujoco_vr"
if _VR_DIR.is_dir() and str(_VR_DIR) not in sys.path:
    sys.path.insert(0, str(_VR_DIR))

try:
    from vr_receiver import VRControllerReceiver, get_local_ips
    VR_AVAILABLE = True
except Exception as _vr_import_err:  # noqa: F841
    VR_AVAILABLE = False

logger = logging.getLogger("debugger.vr")

GRIP_CLUTCH_THRESHOLD = 0.5  # Grip 模拟量超过此值判定为按住（离合接合）
_TRIGGER_RECORD_THRESHOLD = 0.6  # 右手柄扳机超过此值判定为按下（切换 LeRobot 录制）
# ── 摇杆 → 夹爪开合（无需按住 Grip 离合）──
# 摇杆 Y 上推(>死区) = 闭合(→0)，下拉(<-死区) = 张开(→GRIPPER_OPEN_RAD)；
# 速度累积并钳位在 [GRIPPER_OPEN_RAD, 0]；摇杆回中时夹爪保持当前角度不动。
GRIPPER_OPEN_RAD = -1.835    # 夹爪全开角度（负方向为张开，SDK 限位 ±1.5708 内的实测全开）
GRIPPER_CLOSE_RAD = 0.0      # 夹爪全闭角度
_JOY_GRIPPER_GAIN = 1.0      # 摇杆 Y → 夹爪角速度 (rad/s)，方向反了改 -1.0
_JOY_GRIPPER_DEADZONE = 0.10  # 摇杆激活死区，低于此值夹爪保持不动
GRIPPER_TORQUE_MAX = 0.5     # 夹爪力矩上限 (Nm)：闭合时力矩≥此值停止继续闭合
GRIPPER_TORQUE_MIN = -0.5    # 夹爪力矩下限 (Nm)：张开时力矩≤此值停止继续张开
# ── 夹爪力矩限制增强（消除快速闭合时的力矩超调）──
# 问题：纯反应式钳位存在反馈延迟+电机惯性，快速闭合时力矩会冲到 3Nm。
# 对策：力矩接近上限时按比例预减速，触限后锁定方向保位（不再继续推进）。
_GRIPPER_TORQUE_SOFT_RATIO = 0.6    # 软阈值比例：力矩达上限的此比例时开始预减速（0.6→0.3Nm）

# ── EL-A3 方向修正（vr_receiver 默认映射针对 OpenArm，与 EL-A3 基座系不符）──
# 位置：VR 参考系与机械臂基座系相差 180° 偏航 → x/y 取反（修正前后/左右反向）
_POS_FIX = np.array([-1.0, -1.0, 1.0])
# 姿态：右乘 Rx(180) 抵消接收端 r_fix 中的 Rx(180) 项
# （修正横滚反向、俯仰未映射问题；抵消后：俯仰→y轴、偏航→x轴、横滚→+z轴）
_ROT_FIX = np.diag([1.0, -1.0, -1.0])

# 与 xbox_control.py 手柄 X/B 键一致的目标位置
ZERO_POSITIONS = [0.0] * 6                    # 零位（A 键）
HOME_POSITIONS = [0.0, 0.785, -0.785, 1, 0.0, 0.0]  # 工作位（B 键）


class VrArmController:
    """VR 相对笛卡尔控制器（后台线程运行）

    100Hz 控制循环：右手柄 Grip 离合 → 相对位姿增量 → ik_step →
    2 阶临界阻尼滤波 → JointCtrl；摇杆 Y → GripperCtrl。
    """

    def __init__(
        self,
        arm,
        receiver: "VRControllerReceiver",
        update_rate: float = 100.0,
        filter_omega: float = 14.0,
        max_ik_jump: float = 0.5,
    ):
        self._arm = arm
        self._receiver = receiver
        self._rate = update_rate
        self._dt = 1.0 / update_rate
        self._filter_omega = filter_omega
        self._max_ik_jump = max_ik_jump

        self._kin = arm._get_kinematics()
        if self._kin is None:
            raise RuntimeError("Pinocchio kinematics unavailable")

        import pinocchio as pin
        self._pin = pin

        # 运动缩放（UI 滑条实时写入，控制线程读取）
        self.scale = 1.0

        # 离合与锚点状态
        self._clutched = False
        self._vr_anchor_pos = np.zeros(3)
        self._vr_anchor_rot = np.eye(3)
        self._ee_anchor_pos = np.zeros(3)
        self._ee_anchor_rot = np.eye(3)

        # 目标位姿（世界系）
        self._target_pos = np.zeros(3)
        self._target_rot = np.eye(3)

        # IK / 滤波状态
        self._ik_seed: Optional[list] = None
        self._ik_filter_pos: Optional[list] = None
        self._ik_filter_vel: Optional[list] = None
        self._ik_raw: Optional[list] = None
        self._consecutive_rejects = 0
        self._consecutive_ik_fails = 0

        self._gripper_angle: Optional[float] = None

        # 夹爪力矩限制（Nm）：由夹爪面板输入框写入
        self.gripper_torque_max = GRIPPER_TORQUE_MAX
        self.gripper_torque_min = GRIPPER_TORQUE_MIN
        # 夹爪力矩锁定：触限后锁定该方向，摇杆回中或反向才解锁（防止反复撞击）
        self._gripper_close_locked = False
        self._gripper_open_locked = False

        # A/B 按键边沿检测与异步移动状态
        self._prev_btn_a = False
        self._prev_btn_b = False
        self._is_moving = False
        self.log_callback = None  # 由面板设置，用于向主窗口日志输出
        # 右手柄扳机键（边沿触发）：切换 LeRobot 录制 开始/停止
        self._prev_trigger_pressed = False
        self.record_toggle_callback = None  # 由面板设置，转发到主窗口

        self._running = False
        self.last_ik_err = 0.0

    # ---- 生命周期 ----

    def start(self):
        self._running = True
        self._initialize()
        period = 1.0 / self._rate
        next_tick = time.monotonic()
        while self._running:
            next_tick += period
            try:
                self._tick()
            except Exception as e:
                logger.error("VR 控制循环异常: %s", e)
            now = time.monotonic()
            sleep_s = next_tick - now
            if sleep_s < -period:
                next_tick = now + period
            elif sleep_s > 0:
                time.sleep(sleep_s)

    def stop(self):
        self._running = False

    def _initialize(self):
        try:
            q_current = self._arm.GetArmJointMsgs().to_list()[:6]
        except Exception as e:
            logger.warning("读取当前关节位置失败: %s, 回退到零位", e)
            q_current = [0.0] * 6

        self._ik_seed = list(q_current)
        self._ik_filter_pos = list(q_current)
        self._ik_filter_vel = [0.0] * 6
        self._ik_raw = None
        self._consecutive_rejects = 0
        self._consecutive_ik_fails = 0

        # 以当前夹爪反馈角度为基准，避免启动时夹爪突动
        try:
            positions = self._arm.GetArmJointMsgs().to_list(include_gripper=True)
            if len(positions) >= 7:
                self._gripper_angle = float(positions[6])
        except Exception:
            pass

        se3 = self._kin.forward_kinematics_se3(q_current)
        self._target_pos = np.asarray(se3.translation, dtype=float).copy()
        self._target_rot = np.asarray(se3.rotation, dtype=float).copy()
        self._clutched = False
        logger.info(
            "VR 控臂初始化完成, 当前末端: (%.3f, %.3f, %.3f) m",
            self._target_pos[0], self._target_pos[1], self._target_pos[2],
        )

    def _read_gripper_torque(self) -> Optional[float]:
        """读取夹爪(L7)当前力矩 (Nm)，失败返回 None"""
        try:
            efforts = self._arm.GetArmJointEfforts().to_list(include_gripper=True)
            if len(efforts) >= 7:
                return float(efforts[6])
        except Exception:
            pass
        return None

    def _read_gripper_feedback(self) -> float:
        """读取夹爪(L7)当前反馈角度 (rad)，失败返回当前目标角度"""
        try:
            positions = self._arm.GetArmJointMsgs().to_list(include_gripper=True)
            if len(positions) >= 7:
                return float(positions[6])
        except Exception:
            pass
        return float(self._gripper_angle or 0.0)

    # ---- 主循环 ----

    def _log(self, msg: str):
        logger.info("%s", msg)
        if self.log_callback:
            try:
                self.log_callback(msg)
            except Exception:
                pass

    def _tick(self):
        ctrl = self._receiver.get_right_state()

        # A/B 按键（边沿触发，与手柄 X/B 键逻辑一致）
        if ctrl.button_a and not self._prev_btn_a:
            self._async_move(ZERO_POSITIONS, "零位", duration=4.0,
                             v_max=0.6, a_max=1.0)
        if ctrl.button_b and not self._prev_btn_b:
            self._async_move(HOME_POSITIONS, "工作位")
        self._prev_btn_a = ctrl.button_a
        self._prev_btn_b = ctrl.button_b

        # 扳机键（上升沿触发）：切换 LeRobot 录制 开始/停止
        trig_pressed = ctrl.valid and ctrl.trigger > _TRIGGER_RECORD_THRESHOLD
        if trig_pressed and not self._prev_trigger_pressed \
                and self.record_toggle_callback is not None:
            try:
                self.record_toggle_callback()
            except Exception as e:
                logger.warning("扳机键切换录制回调失败: %s", e)
        self._prev_trigger_pressed = trig_pressed

        if self._is_moving:
            return

        # 摇杆 Y → 夹爪开合（无需按住 Grip 离合，位置速度累积式）
        # 上推(joy_y>死区)=闭合(→0)，下拉(joy_y<-死区)=张开(→GRIPPER_OPEN_RAD)
        joy_y = float(np.clip(ctrl.joystick_y, -1.0, 1.0))
        # 摇杆回中时解锁力矩锁定，允许用户松开后重新尝试闭合/张开
        if abs(joy_y) <= _JOY_GRIPPER_DEADZONE:
            self._gripper_close_locked = False
            self._gripper_open_locked = False
        else:
            if self._gripper_angle is None:
                self._gripper_angle = 0.0
            closing = joy_y > 0.0
            increment = _JOY_GRIPPER_GAIN * joy_y * self._dt  # 带符号：闭合>0，张开<0
            # 反向操作时解锁对应方向的锁定
            if closing:
                self._gripper_open_locked = False
            else:
                self._gripper_close_locked = False

            # 力矩受限推进：预减速 + 触限锁定回退，消除快速闭合时的力矩超调
            effective_inc = increment
            gripper_torque = self._read_gripper_torque()
            if gripper_torque is not None:
                if closing:
                    soft = self.gripper_torque_max * _GRIPPER_TORQUE_SOFT_RATIO
                    if gripper_torque >= self.gripper_torque_max or self._gripper_close_locked:
                        # 触限：锁定闭合方向，保位在当前反馈位置（不再继续推进）
                        self._gripper_close_locked = True
                        effective_inc = self._read_gripper_feedback() - self._gripper_angle
                    elif gripper_torque > soft and self.gripper_torque_max > soft:
                        # 预减速：力矩从软阈值→上限时，推进量线性衰减到 0
                        k = (self.gripper_torque_max - gripper_torque) / (self.gripper_torque_max - soft)
                        effective_inc = increment * max(0.0, min(1.0, k))
                else:  # 张开方向（对称处理）
                    soft = self.gripper_torque_min * _GRIPPER_TORQUE_SOFT_RATIO
                    if gripper_torque <= self.gripper_torque_min or self._gripper_open_locked:
                        self._gripper_open_locked = True
                        effective_inc = self._read_gripper_feedback() - self._gripper_angle
                    elif gripper_torque < soft and self.gripper_torque_min < soft:
                        k = (self.gripper_torque_min - gripper_torque) / (self.gripper_torque_min - soft)
                        effective_inc = increment * max(0.0, min(1.0, k))

            angle = float(np.clip(
                self._gripper_angle + effective_inc,
                GRIPPER_OPEN_RAD, GRIPPER_CLOSE_RAD))
            self._gripper_angle = angle
            try:
                self._arm.GripperCtrl(gripper_angle=angle)
            except Exception as e:
                logger.warning("夹爪控制失败: %s", e)

        gripped = ctrl.valid and ctrl.grip > GRIP_CLUTCH_THRESHOLD

        if not gripped:
            # 离合释放：冻结机械臂（保持当前滤波位置）
            self._clutched = False
            self._send_filtered()
            return

        if not self._clutched:
            # Grip 刚按下：以当前手柄位姿和末端目标位姿重新锚定
            self._vr_anchor_pos = ctrl.position * _POS_FIX
            self._vr_anchor_rot = ctrl.rotation @ _ROT_FIX
            self._ee_anchor_pos = self._target_pos.copy()
            self._ee_anchor_rot = self._target_rot.copy()
            self._clutched = True

        scale = float(self.scale)

        vr_pos = ctrl.position * _POS_FIX
        vr_rot = ctrl.rotation @ _ROT_FIX

        # 相对平移：ee_anchor + (vr_current - vr_anchor) * scale（世界系增量）
        target_pos = self._ee_anchor_pos + (vr_pos - self._vr_anchor_pos) * scale

        # 相对旋转 1:1 跟随末端姿态；修正偏航/俯仰接反：
        # 接收端处理后 dR 的 RPY 中 c[0]=手柄偏航、c[1]=手柄俯仰、c[2]=手柄横滚，
        # 交换 c[0]与c[1] 使 手柄偏航→末端偏航、手柄俯仰→末端俯仰、横滚→横滚；
        # c[1] 取负以纠正俯仰方向。
        dR = self._vr_anchor_rot.T @ vr_rot
        c = self._pin.rpy.matrixToRpy(dR)
        target_rot = self._ee_anchor_rot @ self._pin.rpy.rpyToMatrix(
            float(-c[1]), float(c[0]), float(c[2]))

        self._target_pos = target_pos
        self._target_rot = target_rot

        # IK
        rpy = self._pin.rpy.matrixToRpy(target_rot)
        target_pose = ArmEndPose(
            x=float(target_pos[0]), y=float(target_pos[1]), z=float(target_pos[2]),
            rx=float(rpy[0]), ry=float(rpy[1]), rz=float(rpy[2]),
        )
        try:
            q_sol, ik_err = self._kin.ik_step(
                target_pose, self._ik_seed,
                damping=5e-3, max_step=self._max_ik_jump,
            )
            self.last_ik_err = ik_err
            if q_sol is not None and self._accept_ik(q_sol):
                self._ik_raw = q_sol
                self._ik_seed = list(q_sol)
                self._consecutive_ik_fails = 0
            else:
                self._consecutive_ik_fails += 1
                if self._consecutive_ik_fails >= 50:
                    logger.warning("IK 连续失败 50+ 次，自动重新同步")
                    self._resync_ik()
        except Exception as e:
            logger.error("IK 异常: %s", e)

        self._send_filtered()

    # ---- IK 跳变保护 ----

    def _accept_ik(self, q_new) -> bool:
        ref = self._ik_seed
        if ref is None:
            return True
        max_diff = max(abs(q_new[i] - ref[i]) for i in range(6))
        if max_diff <= self._max_ik_jump:
            self._consecutive_rejects = 0
            return True
        self._consecutive_rejects += 1
        if self._consecutive_rejects >= 50:
            logger.warning("IK 连续拒绝 50+ 帧，自动重新同步")
            self._resync_ik()
        return False

    def _resync_ik(self):
        try:
            q = self._arm.GetArmJointMsgs().to_list()[:6]
        except Exception:
            return
        self._ik_seed = list(q)
        self._ik_filter_pos = list(q)
        self._ik_raw = list(q)
        for i in range(6):
            self._ik_filter_vel[i] *= 0.2
        self._consecutive_rejects = 0
        self._consecutive_ik_fails = 0
        se3 = self._kin.forward_kinematics_se3(q)
        self._target_pos = np.asarray(se3.translation, dtype=float).copy()
        self._target_rot = np.asarray(se3.rotation, dtype=float).copy()
        self._clutched = False

    # ---- A/B 按键：异步移动到指定位形（与 xbox_control 逻辑一致） ----

    def _async_move(self, positions, name: str, duration: float = 5.0,
                    v_max=None, a_max=None):
        if self._is_moving:
            return
        self._is_moving = True
        self._log(f"正在移动到 {name}...")
        threading.Thread(
            target=self._do_move, args=(positions, name, duration, v_max, a_max),
            daemon=True).start()

    def _do_move(self, positions, name: str, duration: float, v_max, a_max):
        try:
            self._arm.MoveJ(positions, duration=duration,
                            v_max=v_max, a_max=a_max, block=True)
            self._ik_seed = list(positions)
            self._ik_filter_pos = list(positions)
            self._ik_filter_vel = [0.0] * 6
            self._ik_raw = list(positions)
            self._consecutive_rejects = 0
            self._consecutive_ik_fails = 0
            se3 = self._kin.forward_kinematics_se3(positions)
            self._target_pos = np.asarray(se3.translation, dtype=float).copy()
            self._target_rot = np.asarray(se3.rotation, dtype=float).copy()
            self._clutched = False
            self._log(f"已到达 {name}")
        except Exception as e:
            logger.error("移动异常: %s", e)
            self._log(f"移动到 {name} 失败: {e}")
        finally:
            self._is_moving = False

    # ---- 2 阶滤波 + 关节指令 ----

    def _send_filtered(self):
        if self._ik_raw is None and self._ik_filter_pos is None:
            return
        if self._ik_filter_pos is None and self._ik_raw is not None:
            self._ik_filter_pos = list(self._ik_raw)
            self._ik_filter_vel = [0.0] * 6

        if self._ik_raw is not None:
            omega = self._filter_omega
            dt = self._dt
            a = omega * dt
            ea = math.exp(-a)
            for i in range(6):
                err = self._ik_raw[i] - self._ik_filter_pos[i]
                vel = self._ik_filter_vel[i]
                err_new = ea * ((1.0 + a) * err - dt * vel)
                vel_new = ea * (omega * omega * dt * err + (1.0 - a) * vel)
                self._ik_filter_pos[i] = self._ik_raw[i] - err_new
                self._ik_filter_vel[i] = vel_new

        self._arm.JointCtrl(*self._ik_filter_pos,
                            velocities=list(self._ik_filter_vel))

    # ---- UI 状态读取 ----

    @property
    def clutched(self) -> bool:
        return self._clutched

    def target_pose_text(self) -> str:
        rpy = self._pin.rpy.matrixToRpy(self._target_rot)
        return (
            f"XYZ: ({self._target_pos[0]:.3f}, {self._target_pos[1]:.3f}, {self._target_pos[2]:.3f}) m  "
            f"RPY: ({rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}) rad"
        )


class VrPanel(QWidget):
    """VR 控制 Tab 面板（连接区 + 位姿监控区 + 控臂区）"""

    vr_log = pyqtSignal(str)
    # 右手柄扳机键按下：请求切换 LeRobot 录制（开始/停止）
    record_toggle_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._receiver: Optional["VRControllerReceiver"] = None
        self._arm = None
        self._controller: Optional[VrArmController] = None
        self._ctrl_thread: Optional[threading.Thread] = None
        self._ctrl_running = False

        # 夹爪力矩限制（Nm）：夹爪面板输入框写入，控制启动时同步
        self._gripper_torque_max = GRIPPER_TORQUE_MAX
        self._gripper_torque_min = GRIPPER_TORQUE_MIN

        self._init_ui()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)  # 20 Hz 刷新
        self._poll_timer.timeout.connect(self._poll_status)

    # ---- UI 构建 ----

    def _init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(4)

        self._conn_group = self._build_connection_group()
        layout.addWidget(self._conn_group)
        self._monitor_group = self._build_monitor_group()
        layout.addWidget(self._monitor_group)
        self._ctrl_group = self._build_control_group()
        layout.addWidget(self._ctrl_group)
        layout.addStretch()

        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self.retranslate_ui()

    def _build_connection_group(self) -> QGroupBox:
        grp = QGroupBox()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 2, 6, 4)
        layout.setSpacing(2)

        row = QHBoxLayout()
        row.setSpacing(4)
        self._lbl_port = QLabel()
        row.addWidget(self._lbl_port)
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1024, 65535)
        self._port_spin.setValue(5006)
        row.addWidget(self._port_spin)

        self._connect_btn = QPushButton()
        self._connect_btn.setObjectName("connectBtn")
        self._connect_btn.clicked.connect(self._toggle_connection)
        row.addWidget(self._connect_btn, 1)
        layout.addLayout(row)

        self._status_label = QLabel()
        self._status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._status_label)

        self._ip_label = QLabel()
        self._ip_label.setStyleSheet("font-size: 10px;")
        self._ip_label.setWordWrap(True)
        layout.addWidget(self._ip_label)

        grp.setLayout(layout)
        return grp

    def _build_monitor_group(self) -> QGroupBox:
        grp = QGroupBox()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 2, 6, 4)
        layout.setSpacing(2)

        self._right_label = QLabel("R: ---")
        self._right_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self._right_label)
        self._left_label = QLabel("L: ---")
        self._left_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self._left_label)

        grp.setLayout(layout)
        return grp

    def _build_control_group(self) -> QGroupBox:
        grp = QGroupBox()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 2, 6, 4)
        layout.setSpacing(2)

        row1 = QHBoxLayout()
        row1.setSpacing(6)
        self._ctrl_btn = QPushButton()
        self._ctrl_btn.setObjectName("enableBtn")
        self._ctrl_btn.setEnabled(False)
        self._ctrl_btn.clicked.connect(self._toggle_control)
        row1.addWidget(self._ctrl_btn)

        self._lbl_clutch = QLabel()
        row1.addWidget(self._lbl_clutch)
        self._clutch_label = QLabel("—")
        row1.addWidget(self._clutch_label)
        row1.addStretch()
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(4)
        self._lbl_scale = QLabel()
        row2.addWidget(self._lbl_scale)
        self._scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._scale_slider.setRange(1, 30)      # 0.1x ~ 3.0x
        self._scale_slider.setValue(10)         # 默认 1.0x
        self._scale_slider.valueChanged.connect(self._on_scale_changed)
        row2.addWidget(self._scale_slider, 1)
        self._scale_label = QLabel("1.0x")
        self._scale_label.setFixedWidth(36)
        row2.addWidget(self._scale_label)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(6)
        self._lbl_target = QLabel()
        row3.addWidget(self._lbl_target)
        self._target_label = QLabel("—")
        self._target_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        row3.addWidget(self._target_label, 1)
        layout.addLayout(row3)

        grp.setLayout(layout)
        return grp

    def retranslate_ui(self):
        self._conn_group.setTitle(tr("vr.conn_group"))
        self._lbl_port.setText(tr("vr.port"))
        if self._receiver is not None:
            self._connect_btn.setText(tr("vr.disconnect"))
        else:
            self._connect_btn.setText(tr("vr.connect"))
        if not VR_AVAILABLE:
            self._status_label.setText(tr("vr.no_module"))
        else:
            self._update_status_text()
        self._update_ip_label()

        self._monitor_group.setTitle(tr("vr.monitor_group"))
        self._ctrl_group.setTitle(tr("vr.ctrl_group"))
        self._update_ctrl_btn_state()
        self._lbl_clutch.setText(tr("vr.clutch_state"))
        self._lbl_scale.setText(tr("vr.scale"))
        self._lbl_target.setText(tr("vr.target_pose"))

    # ---- VR 连接 ----

    def _toggle_connection(self):
        if self._receiver is not None:
            self._stop_link()
        else:
            self._start_link()

    def _start_link(self):
        if not VR_AVAILABLE:
            self.vr_log.emit(tr("vr.no_module"))
            return
        port = self._port_spin.value()
        self._receiver = VRControllerReceiver(port=port, smoothing_enabled=True)
        self._receiver.start()
        self._poll_timer.start()
        self._connect_btn.setText(tr("vr.disconnect"))
        self._connect_btn.setObjectName("disconnectBtn")
        self._connect_btn.setStyle(self._connect_btn.style())
        self._port_spin.setEnabled(False)
        self._update_ctrl_btn_state()
        self.vr_log.emit(tr("vr.link_started", port=port))

        # VR 连接后机械臂自动回零位
        if self._arm is not None:
            threading.Thread(target=self._auto_move_zero, daemon=True,
                             name="vr_auto_zero").start()

    def _auto_move_zero(self):
        try:
            self.vr_log.emit(tr("vr.auto_zero"))
            self._arm.MoveJ(list(ZERO_POSITIONS), duration=4.0,
                            v_max=0.6, a_max=1.0, block=True)
            self.vr_log.emit(tr("vr.auto_zero_done"))
        except Exception as e:
            logger.error("VR 自动回零位失败: %s", e)
            self.vr_log.emit(tr("vr.ctrl_error", e=e))

    def _stop_link(self):
        if self._ctrl_running:
            self._stop_control()
        self._poll_timer.stop()
        if self._receiver is not None:
            self._receiver.stop()
            self._receiver = None
        self._connect_btn.setText(tr("vr.connect"))
        self._connect_btn.setObjectName("connectBtn")
        self._connect_btn.setStyle(self._connect_btn.style())
        self._port_spin.setEnabled(True)
        self._update_status_text()
        self._update_ctrl_btn_state()
        self._right_label.setText("R: ---")
        self._left_label.setText("L: ---")
        self.vr_log.emit(tr("vr.link_stopped"))

    def is_connected(self) -> bool:
        return self._receiver is not None and self._receiver.is_connected()

    def _update_status_text(self):
        if self._receiver is None:
            self._status_label.setText(tr("vr.status_disconnected"))
        elif self._receiver.is_connected():
            self._status_label.setText(tr("vr.status_connected"))
        else:
            self._status_label.setText(tr("vr.status_waiting"))

    def _update_ip_label(self):
        if VR_AVAILABLE:
            ips = get_local_ips()
            if ips:
                self._ip_label.setText(tr("vr.local_ips", ips=", ".join(ips)))
            else:
                self._ip_label.setText("")

    # ---- 状态轮询 (20 Hz) ----

    def _poll_status(self):
        if self._receiver is None:
            return
        self._update_status_text()

        right = self._receiver.get_right_state()
        left = self._receiver.get_left_state()
        self._right_label.setText(self._fmt_ctrl("R", right))
        self._left_label.setText(self._fmt_ctrl("L", left))

        if self._ctrl_running and self._controller:
            sc = SCENE_COLORS[ThemeManager.instance().theme]
            if self._controller.clutched:
                self._clutch_label.setText(tr("vr.clutch_engaged"))
                self._clutch_label.setStyleSheet(
                    f"color: {sc['mode_normal']}; font-weight: bold;")
            else:
                self._clutch_label.setText(tr("vr.clutch_released"))
                self._clutch_label.setStyleSheet(
                    f"color: {sc['mode_estop']}; font-weight: bold;")
            self._target_label.setText(self._controller.target_pose_text())

    @staticmethod
    def _fmt_ctrl(tag: str, ctrl) -> str:
        if not ctrl.valid:
            return f"{tag}: [no signal]"
        q = ctrl.quaternion  # w,x,y,z
        return (
            f"{tag}: pos=({ctrl.position[0]:+.3f},{ctrl.position[1]:+.3f},{ctrl.position[2]:+.3f}) m  "
            f"quat=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f})  "
            f"trig={ctrl.trigger:.2f} grip={ctrl.grip:.2f}"
        )

    # ---- VR 控臂 ----

    def set_arm(self, arm):
        self._arm = arm
        self._update_ctrl_btn_state()

    def set_gripper_torque_limits(self, torque_max: float, torque_min: float):
        """设置夹爪力矩限制 (Nm)，由夹爪面板输入框实时写入"""
        self._gripper_torque_max = float(torque_max)
        self._gripper_torque_min = float(torque_min)
        if self._controller is not None:
            self._controller.gripper_torque_max = self._gripper_torque_max
            self._controller.gripper_torque_min = self._gripper_torque_min

    def _update_ctrl_btn_state(self):
        can_start = (
            self._receiver is not None
            and self._arm is not None
            and not self._ctrl_running
        )
        self._ctrl_btn.setEnabled(can_start or self._ctrl_running)
        if self._ctrl_running:
            self._ctrl_btn.setText(tr("vr.stop_ctrl"))
            self._ctrl_btn.setObjectName("disconnectBtn")
        else:
            self._ctrl_btn.setText(tr("vr.start_ctrl"))
            self._ctrl_btn.setObjectName("enableBtn")
        self._ctrl_btn.setStyle(self._ctrl_btn.style())

    def _on_scale_changed(self, value: int):
        scale = value / 10.0
        self._scale_label.setText(f"{scale:.1f}x")
        if self._controller is not None:
            self._controller.scale = scale

    def _toggle_control(self):
        if self._ctrl_running:
            self._stop_control()
        else:
            self._start_control()

    def _start_control(self):
        if self._receiver is None:
            self.vr_log.emit(tr("vr.connect_first"))
            return
        if self._arm is None:
            self.vr_log.emit(tr("vr.connect_arm_first"))
            return
        try:
            self._controller = VrArmController(
                arm=self._arm,
                receiver=self._receiver,
                update_rate=100.0,
            )
        except Exception as e:
            self.vr_log.emit(tr("vr.need_kin"))
            logger.error("VR 控制器初始化失败: %s", e)
            return
        self._controller.scale = self._scale_slider.value() / 10.0
        self._controller.log_callback = self.vr_log.emit
        # 扳机键切换 LeRobot 录制：控制线程回调 → Qt 信号（自动排队到主线程）
        self._controller.record_toggle_callback = self.record_toggle_requested.emit
        self._controller.gripper_torque_max = self._gripper_torque_max
        self._controller.gripper_torque_min = self._gripper_torque_min
        self._receiver.reset_smoothing()

        self._ctrl_running = True
        self._ctrl_thread = threading.Thread(
            target=self._run_control, daemon=True, name="vr_arm_ctrl"
        )
        self._ctrl_thread.start()
        self._update_ctrl_btn_state()
        self.vr_log.emit(tr("vr.ctrl_started"))

    def _run_control(self):
        try:
            self._controller.start()
        except Exception as e:
            logger.error(tr("vr.ctrl_error", e=e))
            self.vr_log.emit(tr("vr.ctrl_error", e=e))
        finally:
            self._ctrl_running = False

    def _stop_control(self):
        if self._controller:
            self._controller.stop()
        if self._ctrl_thread:
            self._ctrl_thread.join(timeout=2.0)
            self._ctrl_thread = None
        self._controller = None
        self._ctrl_running = False
        self._update_ctrl_btn_state()
        self._clutch_label.setText("—")
        self._clutch_label.setStyleSheet("")
        self._target_label.setText("—")
        self.vr_log.emit(tr("vr.ctrl_stopped"))

    # ---- 资源清理 ----

    def cleanup(self):
        if self._ctrl_running:
            self._stop_control()
        if self._receiver is not None:
            self._receiver.stop()
            self._receiver = None
        self._poll_timer.stop()

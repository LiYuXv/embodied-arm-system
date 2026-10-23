"""VR 控制面板：Quest 3 手柄连接 + 位姿监控 + 相对笛卡尔控臂

功能：
1. 通过 UDP (默认 5006) 接收 Quest 3 VR 手柄位姿/按键数据（vr_receiver）。
2. 实时显示 R/L 手柄末端位置和姿态（四元数）、扳机/握持值。
3. 相对笛卡尔控制：按住手柄侧键 Grip 离合接合，手柄相对锚点的
   位移/旋转增量映射到机械臂末端；松开 Grip 冻结。缩放系数滑条默认 1.0。
   摇杆 Y 控制对应机械臂夹爪开合（上推=闭合，下拉=张开）。
4. 底盘与机械臂互斥：任一手柄 Grip 按住 → 摇杆归机械臂/夹爪，底盘屏蔽；
   双手 Grip 都松开 → 左摇杆控制底盘 8 方向平移，右摇杆 X 反向旋转；
    左手柄 Y 键按住升高、X 键按住降低（连续速度，转速界面可调）；
    左手柄前方扳机按住 → 头显位姿控制底盘/升降。

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
from scipy.spatial.transform import Rotation

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QScrollArea, QFrame, QSpinBox, QDoubleSpinBox, QSlider,
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
# ── 摇杆 → 夹爪开合（仅按住本手柄 Grip 时生效）──
# 摇杆 Y 下拉(<-死区) = 闭合(→0)，上推(>死区) = 张开(→GRIPPER_OPEN_RAD)；
# 速度累积并钳位在 [GRIPPER_OPEN_RAD, 0]；摇杆回中时夹爪保持当前角度不动。
# 与底盘控制互斥：按住 Grip = 摇杆归夹爪，松开 Grip = 摇杆归底盘。
GRIPPER_OPEN_RAD = -1.835    # 夹爪全开角度（负方向为张开，SDK 限位 ±1.5708 内的实测全开）
GRIPPER_CLOSE_RAD = 0.0      # 夹爪全闭角度
_JOY_GRIPPER_GAIN = 2.0      # 摇杆 Y → 夹爪角速度 (rad/s)，默认 2 倍速
_JOY_GRIPPER_SIGN = -1.0     # 下拉=闭合、上推=张开；反向改 +1.0
_JOY_GRIPPER_DEADZONE = 0.10  # 摇杆激活死区，低于此值夹爪保持不动
GRIPPER_TORQUE_MAX = 0.5     # 夹爪力矩上限 (Nm)：闭合时力矩≥此值停止继续闭合
GRIPPER_TORQUE_MIN = -0.5    # 夹爪力矩下限 (Nm)：张开时力矩≤此值停止继续张开
# ── 摇杆 → 底盘：上层离散方向控制（独立于底盘内部实现）──
# VR 层不做任何坐标系换算/轮速映射，只把左摇杆量化为 8 个方向扇区，输出与
# 主界面 前后(fb)/左右(lr)/左斜(ld)/右斜(rd) 四个方向滑条完全一致的指令；
# 全向运动合成（4 电机运动学）与加减速斜坡由底盘面板
# set_external_direction_pads 统一执行（该路径方向已验证正确）。
_CHASSIS_JOY_DEADZONE = 0.12   # 摇杆径向死区（低于此幅值视为回中）
_CHASSIS_MOVE_SPEED = 0.25     # 平移满量程速度比例（1.0 = 面板最大线速度）
_CHASSIS_TURN_SPEED = 0.4      # 旋转满量程速度比例（1.0 = 面板最大角速度）
_CHASSIS_TURN_SIGN = -1.0      # 右摇杆左右方向调换（恢复默认改 +1.0）
# 摇杆 raw 系角度偏置（校准用）：物理前推时终端打印的 ang 若 ≠ 90°，
# 令本值 = 90° - 打印的 ang 即可（标准摇杆为 0）。
_CHASSIS_JOY_ANGLE_OFFSET = 0.0
# ── 左手柄 Y/X 键 → 底盘升降（连续速度控制，仅双手 Grip 都松开时生效）──
# Y 按住 = 上升，X 按住 = 下降；按住期间持续下发速度，松开发送 0 停止。
# 转速由界面"升降转速"设定（默认 _CHASSIS_LIFT_SPEED_RPM 转/分）。
# 上下限位由底盘面板限位开关保护（升降行程约 -700~0 mm）。
_CHASSIS_LIFT_SPEED_RPM = 200.0    # 升降转速默认值 (转/分)，界面"升降转速"可调
_CHASSIS_LIFT_MIN_MM = -700.0      # 升降下限位 (mm，仅参考)
_CHASSIS_LIFT_MAX_MM = 0.0         # 升降上限位 (mm，0=最高，仅参考)
# 8 扇区（扇区角 = idx×45°，0°=raw X 正轴，逆时针）→ (fb, lr, ld, rd) 滑条组合；
# fb +前进，lr +右移，ld +前左，rd +前右（与主界面滑条语义一致）。
_CHASSIS_SECTOR_PADS = (
    (0.0, 1.0, 0.0, 0.0),    # 0°   右
    (0.0, 0.0, 0.0, 1.0),    # 45°  前右
    (1.0, 0.0, 0.0, 0.0),    # 90°  前
    (0.0, 0.0, 1.0, 0.0),    # 135° 前左
    (0.0, -1.0, 0.0, 0.0),   # 180° 左
    (-1.0, -1.0, 0.0, 0.0),  # 225° 后左
    (-1.0, 0.0, 0.0, 0.0),   # 270° 后
    (-1.0, 1.0, 0.0, 0.0),   # 315° 后右
)
# ── 左手柄前方扳机按住 → 头显位姿控制底盘/升降（位置跟随，与仿真
#    mujoco_cartesian_control_fk.py --mobile4 的 HMD 绝对模式映射一致）──
# 扣下扳机瞬间：以当前头显位姿、升降高度、底盘位姿为锚点（归零）；按住期间：
# 头部前/后平移 × 位置缩放 → 底盘目标 x/y，头部左/右平移 → 目标 y（先映射到
# 锚定时刻的底盘机体系——头前移=底盘前进、头右移=底盘右移，与左摇杆语义一致——
# 再按锚定 yaw 旋转到里程计世界系），转头 → 目标 yaw，蹲下/站起 → 升降降/升
# （1:1 位置伺服，不受位置缩放影响，头回锚点升降回锚点）。底盘目标由底盘面板
# 里程计 P 伺服追踪，线速度上限由“速度跟随”滑条给出。松开扳机：底盘停车、
# 升降保持；再次扣下重新锚定归零。
_HMD_TRIGGER_HOLD_THRESHOLD = 0.5  # 左扳机模拟量超过此值判定为按住（头显底盘控制）
_HMD_DEADZONE_XY = 0.03    # 头部平移死区 (m)
_HMD_DEADZONE_YAW = 5.0    # 转头死区 (deg)
_HMD_DEADZONE_Z = 0.03     # 头部高度死区 (m)
_HMD_LIFT_GAIN = 1.5       # 升降增益（升降 m 每 m 头高变化）
_HMD_YAW_SIGN = +1.0       # 转头方向与底盘旋转反了改 -1.0
_HMD_STRAFE_SIGN = -1.0    # 头部右移对应底盘 y 的方向，反了改 +1.0
_HMD_FWD_SIGN = +1.0       # 头部前移对应底盘 x 的方向，反了改 -1.0
_HMD_LIFT_SIGN = +1.0      # 站起应对应升降上升，反了改 -1.0


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _beyond_deadzone(v: float, dz: float) -> float:
    """去掉死区后保留符号的剩余量"""
    return math.copysign(max(0.0, abs(v) - dz), v)


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

# 目标位置：零位第 4 关节保持 26°，其余关节为 0
ZERO_POSITIONS = [0.0, 0.0, 0.0, math.radians(26.0), 0.0, 0.0]  # 零位（A 键）
HOME_POSITIONS = [0.0, 0.785, -0.785, 1, 0.0, 0.0]  # 工作位（B 键）


class VrArmController:
    """VR 相对笛卡尔控制器（后台线程运行）

    100Hz 控制循环：手柄 Grip 离合 → 相对位姿增量 → ik_step →
    2 阶临界阻尼滤波 → JointCtrl；摇杆 Y → GripperCtrl。

    side="right"：右手柄控臂1（含 A/B 按键回零/工作位、扳机切录制）；
    side="left"：左手柄控臂2（左摇杆上下 → 臂2夹爪，无 A/B/扳机功能）。
    """

    def __init__(
        self,
        arm,
        receiver: "VRControllerReceiver",
        side: str = "right",
        update_rate: float = 100.0,
        filter_omega: float = 14.0,
        max_ik_jump: float = 0.5,
    ):
        self._arm = arm
        self._receiver = receiver
        self._side = side
        # 另一臂控制器：右手柄 A/B 按键时双臂联动（由面板注入）
        self._linked_controller: Optional["VrArmController"] = None
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
        # 摇杆 Y → 夹爪角速度 (rad/s)，由面板输入框写入（默认 2 倍速）
        self.gripper_joy_gain = _JOY_GRIPPER_GAIN
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
        tag = "VR-臂2" if self._side == "left" else "VR-臂1"
        logger.info("[%s] %s", tag, msg)
        if self.log_callback:
            try:
                self.log_callback(f"[{tag}] {msg}")
            except Exception:
                pass

    def _get_ctrl_state(self):
        if self._side == "left":
            return self._receiver.get_left_state()
        return self._receiver.get_right_state()

    def _tick(self):
        ctrl = self._get_ctrl_state()

        # A/B 按键与扳机仅右手柄（臂1控制器）处理
        if self._side == "right":
            # A/B 按键（边沿触发，与手柄 X/B 键逻辑一致）：双臂同时回零/工作位
            if ctrl.button_a and not self._prev_btn_a:
                self._async_move_both(ZERO_POSITIONS, "零位", duration=4.0,
                                      v_max=0.6, a_max=1.0)
            if ctrl.button_b and not self._prev_btn_b:
                self._async_move_both(HOME_POSITIONS, "工作位")
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

        # Grip 离合状态提前判定：摇杆→夹爪仅在按住本手柄 Grip 时生效，
        # 松开 Grip 后摇杆归底盘控制（见 VrPanel._poll_status 互斥逻辑）
        gripped = ctrl.valid and ctrl.grip > GRIP_CLUTCH_THRESHOLD

        # 摇杆 Y → 夹爪开合（仅按住 Grip 时，位置速度累积式）
        # 下拉(joy_y<-死区)=闭合(→0)，上推(joy_y>死区)=张开(→GRIPPER_OPEN_RAD)
        joy_y = float(np.clip(ctrl.joystick_y, -1.0, 1.0)) if gripped else 0.0
        # 摇杆回中时解锁力矩锁定，允许用户松开后重新尝试闭合/张开
        if abs(joy_y) <= _JOY_GRIPPER_DEADZONE:
            self._gripper_close_locked = False
            self._gripper_open_locked = False
        else:
            if self._gripper_angle is None:
                self._gripper_angle = 0.0
            increment = _JOY_GRIPPER_SIGN * self.gripper_joy_gain * joy_y * self._dt  # 带符号：闭合>0，张开<0
            closing = increment > 0.0
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

    def _async_move_both(self, positions, name: str, duration: float = 5.0,
                         v_max=None, a_max=None):
        """A/B 键双臂联动：本臂与联动臂同时异步移动到同一指定位形"""
        self._async_move(positions, name, duration=duration,
                         v_max=v_max, a_max=a_max)
        other = self._linked_controller
        if other is not None and not other._is_moving:
            other._async_move(positions, name, duration=duration,
                              v_max=v_max, a_max=a_max)

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
    # 左摇杆 → 底盘方向滑条指令 (fb, lr, ld, rd，语义与主界面四滑条一致)，
    # 右摇杆 X → 底盘旋转 (rot)，lift_mm 恒为 nan（X/Y 键升降改走 lift_vel_cmd）；
    # 仅双手 Grip 都松开时输出（任一 Grip 按住期间摇杆归机械臂/夹爪）
    chassis_cmd = pyqtSignal(float, float, float, float, float, float)
    # 左手柄扳机按住 → 头显底盘/升降指令：(target_x, target_y, target_yaw,
    # max_lin, lift_mm)；target 为世界系底盘目标位姿 (m, m, rad)，max_lin 为
    # 位置伺服线速度上限 (m/s)，lift_mm 升降位置目标 (nan=不控升降)
    hmd_cmd = pyqtSignal(float, float, float, float, float)
    # 左手柄 X 松开：停止底盘、冻结升降
    hmd_released = pyqtSignal()
    # 左手柄 Y/X 键连续升降速度：(direction, speed_rpm)；
    # direction=-1 上升 / +1 下降 / 0 停止（与 chassis._jog_lift 约定一致）
    lift_vel_cmd = pyqtSignal(float, float)
    # 双臂自动回零位完成后触发（后台线程发出，经队列回主线程自动启动控臂）
    auto_zero_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._receiver: Optional["VRControllerReceiver"] = None
        self._arm = None
        self._arm2 = None  # 第二臂（can1，左手柄控制）
        self._controller: Optional[VrArmController] = None
        self._controller2: Optional[VrArmController] = None
        self._ctrl_thread: Optional[threading.Thread] = None
        self._ctrl_thread2: Optional[threading.Thread] = None
        self._ctrl_running = False

        # 夹爪力矩限制（Nm）：夹爪面板输入框写入，控制启动时同步
        self._gripper_torque_max = GRIPPER_TORQUE_MAX
        self._gripper_torque_min = GRIPPER_TORQUE_MIN
        # 摇杆 Y → 夹爪速度 (rad/s)：界面输入框写入，控制启动时同步（默认 2 倍速）
        self._gripper_joy_gain = _JOY_GRIPPER_GAIN

        # 头显底盘/升降控制锚点状态（左手柄前方扳机按住时生效）
        self._hmd_anchor_set = False
        self._hmd_anchor = None      # (pos, fwd2, yaw, lift_mm, base_x, base_y, base_yaw)
        self._lift_provider = None   # 由主窗口注入 chassis_win.get_lift_mm
        self._pose_provider = None   # 由主窗口注入 (x_m, y_m, yaw_rad)
        self._hmd_pos_scale = 1.0    # 位置缩放比例（滑条）
        self._hmd_max_lin = 0.30     # 位置伺服线速度上限 m/s（滑条）

        # 左手柄 Y/X 键连续升降速度控制：当前方向状态 + 转速 (RPM，界面可调)
        self._last_lift_dir = 0.0
        self._lift_rpm = _CHASSIS_LIFT_SPEED_RPM

        self._init_ui()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)  # 20 Hz 刷新
        self._poll_timer.timeout.connect(self._poll_status)

        # 双臂自动回零位完成后，自动启动 VR 控臂（无需再手动点“启动控制”）
        self.auto_zero_finished.connect(self._start_control)

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

        # 夹爪速度设置（rad/s）
        row2b = QHBoxLayout()
        row2b.setSpacing(4)
        self._lbl_gripper_speed = QLabel("夹爪速度")
        row2b.addWidget(self._lbl_gripper_speed)
        self._gripper_speed_spin = QDoubleSpinBox()
        self._gripper_speed_spin.setRange(0.1, 5.0)
        self._gripper_speed_spin.setSingleStep(0.1)
        self._gripper_speed_spin.setDecimals(1)
        self._gripper_speed_spin.setSuffix(" rad/s")
        self._gripper_speed_spin.setValue(_JOY_GRIPPER_GAIN)
        self._gripper_speed_spin.setToolTip(
            "摇杆 Y 控制夹爪开合速度 (rad/s)，默认 2 倍速")
        self._gripper_speed_spin.valueChanged.connect(self._on_gripper_speed_changed)
        row2b.addWidget(self._gripper_speed_spin)
        row2b.addStretch()
        layout.addLayout(row2b)

        row3 = QHBoxLayout()
        row3.setSpacing(6)
        self._lbl_target = QLabel()
        row3.addWidget(self._lbl_target)
        self._target_label = QLabel("—")
        self._target_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        row3.addWidget(self._target_label, 1)
        layout.addLayout(row3)

        row3b = QHBoxLayout()
        row3b.setSpacing(6)
        self._lbl_target2 = QLabel()
        row3b.addWidget(self._lbl_target2)
        self._target_label2 = QLabel("—")
        self._target_label2.setStyleSheet("font-family: monospace; font-size: 10px;")
        row3b.addWidget(self._target_label2, 1)
        layout.addLayout(row3b)

        # 头显位置跟随：位置缩放比例 + 速度跟随上限
        row4 = QHBoxLayout()
        row4.setSpacing(4)
        self._lbl_hpos_scale = QLabel("位置缩放")
        row4.addWidget(self._lbl_hpos_scale)
        self._hpos_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._hpos_scale_slider.setRange(2, 50)   # 0.2x ~ 5.0x
        self._hpos_scale_slider.setValue(10)      # 默认 1.0x
        self._hpos_scale_slider.valueChanged.connect(self._on_hpos_scale_changed)
        row4.addWidget(self._hpos_scale_slider, 1)
        self._hpos_scale_label = QLabel("1.0x")
        self._hpos_scale_label.setFixedWidth(36)
        row4.addWidget(self._hpos_scale_label)
        layout.addLayout(row4)

        row5 = QHBoxLayout()
        row5.setSpacing(4)
        self._lbl_hspeed = QLabel("速度跟随")
        row5.addWidget(self._lbl_hspeed)
        self._hspeed_slider = QSlider(Qt.Orientation.Horizontal)
        self._hspeed_slider.setRange(5, 100)      # 0.05 ~ 1.00 m/s
        self._hspeed_slider.setValue(30)          # 默认 0.30 m/s
        self._hspeed_slider.valueChanged.connect(self._on_hspeed_changed)
        row5.addWidget(self._hspeed_slider, 1)
        self._hspeed_label = QLabel("0.30")
        self._hspeed_label.setFixedWidth(36)
        row5.addWidget(self._hspeed_label)
        layout.addLayout(row5)

        # 左手柄 X/Y 键连续升降速度转速设置（RPM）
        row6 = QHBoxLayout()
        row6.setSpacing(4)
        self._lbl_lift_speed = QLabel("升降转速")
        row6.addWidget(self._lbl_lift_speed)
        self._lift_speed_spin = QDoubleSpinBox()
        self._lift_speed_spin.setRange(10.0, 1000.0)
        self._lift_speed_spin.setSingleStep(10.0)
        self._lift_speed_spin.setDecimals(0)
        self._lift_speed_spin.setSuffix(" RPM")
        self._lift_speed_spin.setValue(_CHASSIS_LIFT_SPEED_RPM)
        self._lift_speed_spin.setToolTip(
            "左手柄 Y/X 键按住时的升降电机转速 (RPM)")
        self._lift_speed_spin.valueChanged.connect(self._on_lift_rpm_changed)
        row6.addWidget(self._lift_speed_spin)
        row6.addStretch()
        layout.addLayout(row6)

        grp.setLayout(layout)
        return grp

    def _on_lift_rpm_changed(self, val: float):
        self._lift_rpm = float(val)

    def _on_gripper_speed_changed(self, val: float):
        self._gripper_joy_gain = float(val)
        for ctrl in (self._controller, self._controller2):
            if ctrl is not None:
                ctrl.gripper_joy_gain = self._gripper_joy_gain

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
        self._lbl_target2.setText(f"{tr('jc.arm2')} {tr('vr.target_pose')}")

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

        # VR 连接后机械臂自动回零位，回零完成后自动启动 VR 控臂
        if self._arm is not None:
            threading.Thread(target=self._auto_move_zero, daemon=True,
                             name="vr_auto_zero").start()

    def _auto_move_zero(self):
        """VR 连接后双臂自动回零位（两臂各自独立规划、并行执行）"""
        def _move(arm, label):
            try:
                arm.MoveJ(list(ZERO_POSITIONS), duration=4.0,
                          v_max=0.6, a_max=1.0, block=True)
            except Exception as e:
                logger.error("%s VR 自动回零位失败: %s", label, e)
                self.vr_log.emit(tr("vr.ctrl_error", e=e))

        self.vr_log.emit(tr("vr.auto_zero"))
        threads = []
        if self._arm is not None:
            t = threading.Thread(target=_move, args=(self._arm, "臂1"), daemon=True)
            t.start()
            threads.append(t)
        if self._arm2 is not None:
            t = threading.Thread(target=_move, args=(self._arm2, "臂2"), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        self.vr_log.emit(tr("vr.auto_zero_done"))
        # 回零位完成后自动启动控臂（信号经队列回主线程，避免后台线程直接改 UI）
        self.auto_zero_finished.emit()

    def _stop_link(self):
        if self._ctrl_running:
            self._stop_control()
        if self._hmd_anchor_set:
            self._hmd_anchor_set = False
            self._hmd_anchor = None
            self.hmd_released.emit()
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

        # ── 左手柄前方扳机按住 → 头显位姿控制底盘/升降（优先于摇杆底盘指令）──
        hmd_active = self._poll_hmd_chassis(left)
        if hmd_active:
            # 头显控制期间屏蔽摇杆底盘指令，避免指令通道互相覆盖
            self._chassis_was_active = False

        # 底盘/机械臂互斥：任一手柄 Grip（侧方扳机）按住 → 摇杆归机械臂/夹爪，
        # 底盘指令屏蔽；双手 Grip 都松开 → 摇杆归底盘。
        # 左摇杆 → 底盘方向滑条指令 (fb/lr/ld/rd)，右摇杆 X → 底盘旋转 (rot)，
        # 左手柄 Y 键 → 底盘升高、X 键 → 底盘降低 (lift_mm)。
        # 上层离散方向控制：摇杆量化为 8 扇区 → 查表得方向滑条组合，不做任何
        # 坐标系换算/轮速映射；全向运动合成与加减速斜坡由底盘面板统一执行。
        # 只在摇杆非零（或刚回中补发一次零）时发送，避免覆盖云端推理的底盘指令。
        try:
            grip_held = (
                (right.valid and right.grip > GRIP_CLUTCH_THRESHOLD)
                or (left.valid and left.grip > GRIP_CLUTCH_THRESHOLD)
            )
            if grip_held:
                lsx = lsy = rsx = 0.0
                # Grip 按住期间 X/Y 键不控升降（摇杆归夹爪）
            else:
                lsx = left.joystick_x if left.valid else 0.0
                lsy = left.joystick_y if left.valid else 0.0
                rsx = (_CHASSIS_TURN_SIGN * right.joystick_x
                       if right.valid else 0.0)
            fb = lr = ld = rd = 0.0
            mag = math.hypot(lsx, lsy)
            if mag >= _CHASSIS_JOY_DEADZONE:
                ang = math.degrees(math.atan2(lsy, lsx)) + _CHASSIS_JOY_ANGLE_OFFSET
                sector = int(round(ang / 45.0)) % 8
                k = min(1.0, (mag - _CHASSIS_JOY_DEADZONE)
                        / (1.0 - _CHASSIS_JOY_DEADZONE)) * _CHASSIS_MOVE_SPEED
                fb, lr, ld, rd = (p * k for p in _CHASSIS_SECTOR_PADS[sector])
                # 校准用调试输出：每秒打印一次 raw 值、角度、扇区与滑条指令
                self._chassis_dbg_tick = getattr(self, "_chassis_dbg_tick", 0) + 1
                if self._chassis_dbg_tick >= 20:
                    self._chassis_dbg_tick = 0
                    print(f"[VR-CHASSIS] raw=({lsx:+.2f},{lsy:+.2f}) "
                          f"ang={ang:+.0f} sector={sector} "
                          f"fb={fb:+.2f} lr={lr:+.2f} ld={ld:+.2f} rd={rd:+.2f}")
            rot = 0.0
            if abs(rsx) >= _CHASSIS_JOY_DEADZONE:
                rot = math.copysign(
                    (abs(rsx) - _CHASSIS_JOY_DEADZONE) / (1.0 - _CHASSIS_JOY_DEADZONE)
                    * _CHASSIS_TURN_SPEED, rsx)
            # 左手柄 Y/X 键 → 底盘升降连续速度控制：
            # Y 按住=上升，X 按住=下降（转速由界面"升降转速"设定）；松开发送 0 停止。
            # 按住期间每周期下发以续期（底盘端指令保鲜期），松开/进入 HMD/握住 Grip 自动发 0。
            lift_mm = float("nan")
            lift_dir = 0.0
            if left.valid and not hmd_active and not grip_held:
                if left.button_b and not left.button_a:
                    lift_dir = -1.0   # Y = 上升
                elif left.button_a and not left.button_b:
                    lift_dir = +1.0   # X = 下降
            if lift_dir != self._last_lift_dir or lift_dir != 0.0:
                self.lift_vel_cmd.emit(lift_dir, self._lift_rpm)
            self._last_lift_dir = lift_dir
            joy_active = (fb, lr, ld, rd, rot) != (0.0,) * 5 \
                or not math.isnan(lift_mm)
            if not hmd_active and (joy_active
                                   or getattr(self, "_chassis_was_active", False)):
                self.chassis_cmd.emit(fb, lr, ld, rd, rot, lift_mm)
            if not hmd_active:
                self._chassis_was_active = joy_active
        except Exception:
            pass

        if self._ctrl_running and self._controller:
            sc = SCENE_COLORS[ThemeManager.instance().theme]
            clutched_any = self._controller.clutched or (
                self._controller2 is not None and self._controller2.clutched)
            if clutched_any:
                self._clutch_label.setText(tr("vr.clutch_engaged"))
                self._clutch_label.setStyleSheet(
                    f"color: {sc['mode_normal']}; font-weight: bold;")
            else:
                self._clutch_label.setText(tr("vr.clutch_released"))
                self._clutch_label.setStyleSheet(
                    f"color: {sc['mode_estop']}; font-weight: bold;")
            self._target_label.setText(self._controller.target_pose_text())
            if self._controller2 is not None:
                self._target_label2.setText(self._controller2.target_pose_text())

    # ---- 头显底盘/升降控制（左手柄前方扳机按住） ----

    def set_lift_provider(self, fn):
        """由主窗口注入底盘面板的 get_lift_mm，用于扳机按下时锚定升降高度"""
        self._lift_provider = fn

    def set_chassis_pose_provider(self, fn):
        """由主窗口注入 (x_m, y_m, yaw_rad) 提供函数，扳机按下时锚定底盘位姿"""
        self._pose_provider = fn

    def _on_hpos_scale_changed(self, value: int):
        self._hmd_pos_scale = value / 10.0
        self._hpos_scale_label.setText(f"{self._hmd_pos_scale:.1f}x")

    def _on_hspeed_changed(self, value: int):
        self._hmd_max_lin = value / 100.0
        self._hspeed_label.setText(f"{self._hmd_max_lin:.2f}")

    def _poll_hmd_chassis(self, left) -> bool:
        """左手柄扳机按住：头显位姿相对锚点的增量 → 底盘目标位姿（位置跟随）+ 升降位置。

        扣下瞬间锚定当前头显位置/朝向/升降高度/底盘位姿为零点；按住期间头部
        相对位移 × 位置缩放 → 底盘目标 (x,y)，转头 → 目标 yaw，升降按 1:1 位置
        跟随（不受位置缩放影响）；位置伺服线速度上限由“速度跟随”滑条给出。
        松开扳机：补发一次释放信号（底盘停车、升降保持）；再次扣下重新锚定归零。
        """
        trig_held = bool(left.valid
                         and left.trigger > _HMD_TRIGGER_HOLD_THRESHOLD)
        hmd = self._receiver.get_headset_state() if self._receiver else None

        if not (trig_held and hmd is not None and hmd.valid):
            if self._hmd_anchor_set:
                self._hmd_anchor_set = False
                self._hmd_anchor = None
                self.hmd_released.emit()
                self.vr_log.emit("[VR] 已松开左扳机：头显底盘/升降控制断开（底盘停车，升降保持）")
            return False

        if not self._hmd_anchor_set:
            # 按下瞬间锚定：当前头显位置/朝向/升降高度/底盘位姿 = 零点
            q = hmd.quaternion  # w,x,y,z
            rot_m = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            gaze = rot_m @ np.array([0.0, 0.0, -1.0])  # 头显前向（VR 房间系，Y 向上）
            fwd2 = np.array([gaze[0], gaze[2]])
            n = float(np.linalg.norm(fwd2))
            fwd2 = fwd2 / n if n > 1e-9 else np.array([1.0, 0.0])
            lift_mm = None
            if self._lift_provider is not None:
                try:
                    lift_mm = float(self._lift_provider())
                except Exception:
                    lift_mm = None
            bx = by = byaw = 0.0
            if self._pose_provider is not None:
                try:
                    px, py, pyaw = self._pose_provider()
                    bx, by, byaw = float(px), float(py), float(pyaw)
                except Exception:
                    bx = by = byaw = 0.0
            self._hmd_anchor = (hmd.position.copy(), fwd2, float(hmd.yaw),
                                lift_mm, bx, by, byaw)
            self._hmd_anchor_set = True
            self.vr_log.emit("[VR] 左扳机按住：头显底盘/升降控制已归零锚定"
                             "（头部平移=底盘位移，转头=旋转，蹲起=升降）")

        pos_a, fwd, yaw_a, lift_a, bx, by, byaw = self._hmd_anchor
        delta = hmd.position - pos_a
        right2 = np.array([-fwd[1], fwd[0]])
        d_fwd = float(delta[0] * fwd[0] + delta[2] * fwd[1])
        d_lat = float(delta[0] * right2[0] + delta[2] * right2[1])
        dyaw = math.radians(float(hmd.yaw) - yaw_a)
        dyaw = (dyaw + math.pi) % (2.0 * math.pi) - math.pi
        dz = float(hmd.position[1] - pos_a[1])

        scale = float(self._hmd_pos_scale)
        efwd = _beyond_deadzone(d_fwd, _HMD_DEADZONE_XY)
        elat = _beyond_deadzone(d_lat, _HMD_DEADZONE_XY)
        eyaw = _beyond_deadzone(dyaw, math.radians(_HMD_DEADZONE_YAW))

        # 头部平移先映射到锚定时刻的底盘机体系（与左摇杆底盘控制一致：
        # 头前移=底盘前进、头右移=底盘右移，均为机体方向），再按锚定 yaw
        # 旋转到里程计世界系；否则锚定时底盘 yaw≠0 会导致运动方向整体
        # 偏转（如头前移/左移时底盘走 45° 斜向）
        dx_b = _HMD_FWD_SIGN * efwd * scale            # 机体系：+前进
        dy_b = _HMD_STRAFE_SIGN * elat * scale         # 机体系：+左移
        cos_a, sin_a = math.cos(byaw), math.sin(byaw)
        target_x = bx + dx_b * cos_a - dy_b * sin_a
        target_y = by + dx_b * sin_a + dy_b * cos_a
        target_yaw = _wrap_pi(byaw + _HMD_YAW_SIGN * eyaw)

        lift_mm = float("nan")
        if lift_a is not None:
            lift_mm = float(np.clip(
                lift_a + _HMD_LIFT_SIGN * _beyond_deadzone(dz, _HMD_DEADZONE_Z)
                * _HMD_LIFT_GAIN * 1000.0,
                -700.0, 0.0))

        self.hmd_cmd.emit(target_x, target_y, target_yaw,
                          float(self._hmd_max_lin), lift_mm)
        return True

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

    def set_arm2(self, arm):
        """设置第二臂（can1），由左手柄控制"""
        self._arm2 = arm
        self._update_ctrl_btn_state()

    def set_gripper_torque_limits(self, torque_max: float, torque_min: float):
        """设置夹爪力矩限制 (Nm)，由夹爪面板输入框实时写入（双臂共用）"""
        self._gripper_torque_max = float(torque_max)
        self._gripper_torque_min = float(torque_min)
        for ctrl in (self._controller, self._controller2):
            if ctrl is not None:
                ctrl.gripper_torque_max = self._gripper_torque_max
                ctrl.gripper_torque_min = self._gripper_torque_min

    def apply_defaults(self, defaults: dict):
        """应用 VR 面板默认设置（来自系统设置「VR 控制」分页）。

        defaults 键：lift_speed_rpm / gripper_speed_rad_s / scale /
        hmd_pos_scale / hmd_max_lin；缺省保持当前值。控件 setValue 会
        自动触发 valueChanged 处理器，把值传播到运行中的控制器。
        """
        if not defaults:
            return
        try:
            if "lift_speed_rpm" in defaults:
                self._lift_speed_spin.setValue(float(defaults["lift_speed_rpm"]))
            if "gripper_speed_rad_s" in defaults:
                self._gripper_speed_spin.setValue(float(defaults["gripper_speed_rad_s"]))
            if "scale" in defaults:
                self._scale_slider.setValue(int(round(float(defaults["scale"]) * 10.0)))
            if "hmd_pos_scale" in defaults:
                self._hpos_scale_slider.setValue(int(round(float(defaults["hmd_pos_scale"]) * 10.0)))
            if "hmd_max_lin" in defaults:
                self._hspeed_slider.setValue(int(round(float(defaults["hmd_max_lin"]) * 100.0)))
        except Exception as e:
            logger.warning("应用 VR 默认设置失败: %s", e)

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
        for ctrl in (self._controller, self._controller2):
            if ctrl is not None:
                ctrl.scale = scale

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
        if self._ctrl_running:
            return
        # 必须确保 SDK 控制循环在跑：关节软力矩限制/关节限位都在控制循环内执行，
        # 循环未启动时 JointCtrl 直发运控帧会完全绕过这些保护（导致过流失能）。
        for label, arm in (("臂1", self._arm), ("臂2", self._arm2)):
            if arm is not None and not arm.control_loop_running:
                arm.start_control_loop(rate_hz=200.0)
                self.vr_log.emit(f"[VR] {label}控制循环已自动启动 (200Hz)")
        try:
            # 右手柄 → 臂1
            self._controller = VrArmController(
                arm=self._arm,
                receiver=self._receiver,
                side="right",
                update_rate=100.0,
            )
            # 左手柄 → 臂2（可选，臂2 未连接时仅右臂）
            self._controller2 = None
            if self._arm2 is not None:
                self._controller2 = VrArmController(
                    arm=self._arm2,
                    receiver=self._receiver,
                    side="left",
                    update_rate=100.0,
                )
                # 右手柄 A/B 键双臂联动（回零位 / 工作位）
                self._controller._linked_controller = self._controller2
        except Exception as e:
            self.vr_log.emit(tr("vr.need_kin"))
            logger.error("VR 控制器初始化失败: %s", e)
            self._controller = None
            self._controller2 = None
            return
        scale = self._scale_slider.value() / 10.0
        self._controller.scale = scale
        self._controller.log_callback = self.vr_log.emit
        # 扳机键切换 LeRobot 录制：控制线程回调 → Qt 信号（自动排队到主线程）
        self._controller.record_toggle_callback = self.record_toggle_requested.emit
        self._controller.gripper_torque_max = self._gripper_torque_max
        self._controller.gripper_torque_min = self._gripper_torque_min
        self._controller.gripper_joy_gain = self._gripper_joy_gain
        if self._controller2 is not None:
            self._controller2.scale = scale
            self._controller2.log_callback = self.vr_log.emit
            self._controller2.gripper_torque_max = self._gripper_torque_max
            self._controller2.gripper_torque_min = self._gripper_torque_min
            self._controller2.gripper_joy_gain = self._gripper_joy_gain
        self._receiver.reset_smoothing()

        self._ctrl_running = True
        self._ctrl_thread = threading.Thread(
            target=self._run_control, daemon=True, name="vr_arm_ctrl"
        )
        self._ctrl_thread.start()
        if self._controller2 is not None:
            self._ctrl_thread2 = threading.Thread(
                target=self._run_control2, daemon=True, name="vr_arm2_ctrl"
            )
            self._ctrl_thread2.start()
        self._update_ctrl_btn_state()
        self.vr_log.emit(tr("vr.ctrl_started"))
        if self._controller2 is not None:
            self.vr_log.emit("[VR] 双臂控制：右手柄→臂1，左手柄→臂2"
                             "（按住 Grip 时摇杆=夹爪，松开 Grip 后摇杆=底盘；"
                             "A/B 键双臂回零/工作位）")

    def _run_control(self):
        try:
            self._controller.start()
        except Exception as e:
            logger.error(tr("vr.ctrl_error", e=e))
            self.vr_log.emit(tr("vr.ctrl_error", e=e))
        finally:
            self._ctrl_running = False

    def _run_control2(self):
        try:
            self._controller2.start()
        except Exception as e:
            logger.error("[VR-臂2] %s", tr("vr.ctrl_error", e=e))
            self.vr_log.emit(tr("vr.ctrl_error", e=e))

    def _stop_control(self):
        if self._controller:
            self._controller.stop()
        if self._controller2:
            self._controller2.stop()
        if self._ctrl_thread:
            self._ctrl_thread.join(timeout=2.0)
            self._ctrl_thread = None
        if self._ctrl_thread2:
            self._ctrl_thread2.join(timeout=2.0)
            self._ctrl_thread2 = None
        self._controller = None
        self._controller2 = None
        self._ctrl_running = False
        self._update_ctrl_btn_state()
        self._clutch_label.setText("—")
        self._clutch_label.setStyleSheet("")
        self._target_label.setText("—")
        self._target_label2.setText("—")
        self.vr_log.emit(tr("vr.ctrl_stopped"))

    # ---- 资源清理 ----

    def cleanup(self):
        if self._ctrl_running:
            self._stop_control()
        if self._receiver is not None:
            self._receiver.stop()
            self._receiver = None
        self._poll_timer.stop()

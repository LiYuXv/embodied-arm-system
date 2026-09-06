#!/usr/bin/env python3
"""
EL-A3 VR (Meta Quest 3) 控制程序（纯 SDK，无 ROS 依赖）

通过 UDP 接收 Quest 3 手柄位姿数据（vr_receiver），采用相对模式离合控制：
按住右手柄侧键 Grip 时，手柄相对锚点的位移/旋转增量映射到机械臂末端，
经 Pinocchio IK 求解关节角，2 阶临界阻尼滤波后下发 JointCtrl。

控制映射（右手柄）:
  Grip (侧键)     → 离合：按住跟随，松开冻结（每次按下重新锚定）
  摇杆 Y (上/下)  → 夹爪开合：上推=闭合(→0)，下拉=张开(→-1.835)，无需按住 Grip
  A               → 回零位
  B               → 回工作位 (Home)
  Ctrl+C          → 退出程序

使用前确保:
  1. CAN 接口已激活: sudo ip link set can0 up type can bitrate 1000000
  2. 机械臂已上电
  3. Quest 3 APK 已运行且 IP 指向本机 (UDP 5006)
  4. 安装 pinocchio: pip install pin

用法:
  python3 vr_control.py                    # 默认参数
  python3 vr_control.py --can can1         # 指定 CAN 接口
  python3 vr_control.py --port 5006        # 指定 UDP 端口
  python3 vr_control.py --scale 1.5        # 运动放大比例
  python3 vr_control.py --debug            # 调试模式
"""

import os
import sys
import time
import math
import threading
import signal
import argparse
import logging
from typing import Optional, List

import numpy as np

from el_a3_sdk import ELA3Interface, ArmEndPose, LogLevel

# vr_receiver 位于 debugger/openarm_mujoco_vr/ 目录（非 Python 包），加入搜索路径
_VR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "debugger", "openarm_mujoco_vr")
if os.path.isdir(_VR_DIR) and _VR_DIR not in sys.path:
    sys.path.insert(0, _VR_DIR)

try:
    from vr_receiver import VRControllerReceiver, get_local_ips
    VR_AVAILABLE = True
except Exception as _vr_import_err:  # noqa: F841
    VR_AVAILABLE = False

logger = logging.getLogger("vr_control")

GRIP_CLUTCH_THRESHOLD = 0.5  # Grip 模拟量超过此值判定为按住（离合接合）
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

ZERO_POSITIONS = [0.0] * 6
HOME_POSITIONS = [0.0, 0.785, -0.785, 1, 0.0, 0.0]


# ================================================================
# VR Arm Controller
# ================================================================

class VRArmController:
    """VR 相对笛卡尔控制器

    核心控制流程:
      100Hz 控制循环 → Grip 离合 → 相对位姿增量 → ik_step →
      2 阶临界阻尼滤波 → JointCtrl；摇杆 Y → GripperCtrl
      SDK 200Hz 控制循环 → 运控模式 Type 1 帧（PD + 速度前馈 + 重力补偿）
    """

    def __init__(
        self,
        arm: ELA3Interface,
        receiver: "VRControllerReceiver",
        update_rate: float = 100.0,
        scale: float = 1.0,
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
            raise RuntimeError("Pinocchio 运动学不可用")

        import pinocchio as pin
        self._pin = pin

        # 运动缩放
        self.scale = scale

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
        self._ik_seed: Optional[List[float]] = None
        self._ik_filter_pos: Optional[List[float]] = None
        self._ik_filter_vel: Optional[List[float]] = None
        self._ik_raw: Optional[List[float]] = None
        self._consecutive_rejects = 0
        self._consecutive_ik_fails = 0

        self._gripper_angle: Optional[float] = None

        # 夹爪力矩锁定：触限后锁定该方向，摇杆回中或反向才解锁（防止反复撞击）
        self._gripper_close_locked = False
        self._gripper_open_locked = False

        # A/B 按键边沿检测与异步移动状态
        self._prev_btn_a = False
        self._prev_btn_b = False
        self._is_moving = False

        self._running = False
        self._exit_requested = False
        self.last_ik_err = 0.0

        # 诊断
        self._diag_tick = 0

    @property
    def exit_requested(self) -> bool:
        return self._exit_requested

    # ---- 生命周期 ----

    def start(self):
        """启动控制循环（阻塞，直到 stop() 或 exit_requested）"""
        self._running = True
        self._initialize()
        self._print_banner()

        period = 1.0 / self._rate
        next_tick = time.monotonic()

        while self._running and not self._exit_requested:
            next_tick += period
            try:
                self._tick()
            except Exception as e:
                logger.error("控制循环异常: %s", e)

            now = time.monotonic()
            sleep_s = next_tick - now
            if sleep_s < -period:
                next_tick = now + period
            elif sleep_s > 0:
                time.sleep(sleep_s)

    def stop(self):
        self._running = False

    # ---- 初始化 ----

    def _initialize(self):
        logger.info("从当前位置启动控制...")
        try:
            q_current = self._arm.GetArmJointMsgs().to_list()[:6]
        except Exception as e:
            logger.warning("读取当前关节位置失败: %s, 回退到零位", e)
            q_current = list(ZERO_POSITIONS)

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
            "初始化完成, 当前末端: (%.3f, %.3f, %.3f) m",
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

    def _print_banner(self):
        print("\n" + "=" * 52)
        print("     EL-A3 VR 控制（Quest 3，纯 SDK 模式）")
        print("=" * 52)
        print("  Grip (侧键)  →  离合：按住跟随 / 松开冻结")
        print("  摇杆 Y        →  夹爪（上推=闭合, 下拉=张开, 无需按住 Grip）")
        print("  A            →  回零位")
        print("  B            →  回工作位 (Home)")
        print("  Ctrl+C       →  退出")
        print("=" * 52)
        print(f"  运动缩放:    {self.scale:.1f}x")
        print(f"  末端目标:    ({self._target_pos[0]:.3f}, "
              f"{self._target_pos[1]:.3f}, {self._target_pos[2]:.3f}) m")
        print("  等待 VR 数据... (确认 APK 中 IP 为本机地址)")

    # ---- 主循环 ----

    def _tick(self):
        ctrl = self._receiver.get_right_state()

        # A/B 按键（边沿触发）
        if ctrl.button_a and not self._prev_btn_a:
            self._async_move(ZERO_POSITIONS, "零位", duration=4.0,
                             v_max=0.6, a_max=1.0)
        if ctrl.button_b and not self._prev_btn_b:
            self._async_move(HOME_POSITIONS, "工作位")
        self._prev_btn_a = ctrl.button_a
        self._prev_btn_b = ctrl.button_b

        if self._is_moving:
            self._periodic_status(ctrl)
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
                    soft = GRIPPER_TORQUE_MAX * _GRIPPER_TORQUE_SOFT_RATIO
                    if gripper_torque >= GRIPPER_TORQUE_MAX or self._gripper_close_locked:
                        # 触限：锁定闭合方向，保位在当前反馈位置（不再继续推进）
                        self._gripper_close_locked = True
                        effective_inc = self._read_gripper_feedback() - self._gripper_angle
                    elif gripper_torque > soft and GRIPPER_TORQUE_MAX > soft:
                        # 预减速：力矩从软阈值→上限时，推进量线性衰减到 0
                        k = (GRIPPER_TORQUE_MAX - gripper_torque) / (GRIPPER_TORQUE_MAX - soft)
                        effective_inc = increment * max(0.0, min(1.0, k))
                else:  # 张开方向（对称处理）
                    soft = GRIPPER_TORQUE_MIN * _GRIPPER_TORQUE_SOFT_RATIO
                    if gripper_torque <= GRIPPER_TORQUE_MIN or self._gripper_open_locked:
                        self._gripper_open_locked = True
                        effective_inc = self._read_gripper_feedback() - self._gripper_angle
                    elif gripper_torque < soft and GRIPPER_TORQUE_MIN < soft:
                        k = (GRIPPER_TORQUE_MIN - gripper_torque) / (GRIPPER_TORQUE_MIN - soft)
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
            self._periodic_status(ctrl)
            return

        if not self._clutched:
            # Grip 刚按下：以当前手柄位姿和末端目标位姿重新锚定
            self._vr_anchor_pos = ctrl.position * _POS_FIX
            self._vr_anchor_rot = ctrl.rotation @ _ROT_FIX
            self._ee_anchor_pos = self._target_pos.copy()
            self._ee_anchor_rot = self._target_rot.copy()
            self._clutched = True
            logger.info("离合接合，已锚定")

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
        self._periodic_status(ctrl)

    # ---- IK 跳变保护 ----

    def _accept_ik(self, q_new: List[float]) -> bool:
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

    # ---- A/B 按键：异步移动到指定位形 ----

    def _async_move(self, positions: List[float], name: str, duration: float = 5.0,
                    v_max=None, a_max=None):
        if self._is_moving:
            logger.warning("正在执行其他动作，请稍后再试")
            return
        logger.info("正在移动到 %s...", name)
        self._is_moving = True
        threading.Thread(
            target=self._do_move, args=(positions, name, duration, v_max, a_max),
            daemon=True).start()

    def _do_move(self, positions: List[float], name: str, duration: float,
                 v_max, a_max):
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
            logger.info("已到达 %s", name)
        except Exception as e:
            logger.error("移动异常: %s", e)
        finally:
            self._is_moving = False

    # ---- 2 阶滤波 + 关节指令 ----

    def _send_filtered(self):
        """2 阶临界阻尼滤波 (exact matrix-exponential) → SDK JointCtrl"""
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

    # ---- 诊断 ----

    def _periodic_status(self, ctrl):
        self._diag_tick += 1
        if self._diag_tick < int(self._rate * 2):
            return
        self._diag_tick = 0

        connected = self._receiver.is_connected()
        clutch = "接合" if self._clutched else "释放"
        vr = "OK" if ctrl.valid else "INVALID"
        conn = "已连接" if connected else "等待数据"
        p = self._target_pos
        rpy = self._pin.rpy.matrixToRpy(self._target_rot)
        print(f"  [VR:{conn}/{vr} 离合:{clutch}] "
              f"目标: ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) m  "
              f"({rpy[0]:+.2f},{rpy[1]:+.2f},{rpy[2]:+.2f}) rad  "
              f"ik_err={self.last_ik_err:.4f}")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EL-A3 VR (Quest 3) 控制（纯 SDK 模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          # 默认参数
  %(prog)s --can can1 --port 5006
  %(prog)s --scale 1.5
  %(prog)s --debug
""",
    )
    parser.add_argument("--can", default="can0",
                        help="CAN 接口名 (默认: can0)")
    parser.add_argument("--port", type=int, default=5006,
                        help="VR UDP 接收端口 (默认: 5006)")
    parser.add_argument("--rate", type=float, default=100.0,
                        help="控制频率 Hz (默认: 100)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="运动放大比例 robot=VR*scale (默认: 1.0)")
    parser.add_argument("--kp", type=float, default=80.0,
                        help="位置增益 Kp (默认: 80)")
    parser.add_argument("--kd", type=float, default=4.0,
                        help="速度增益 Kd (默认: 4)")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(name)s][%(levelname)s] %(message)s",
    )
    logging.getLogger("el_a3_sdk").propagate = False

    if not VR_AVAILABLE:
        print("\n无法导入 vr_receiver 模块")
        print(f"请确认文件存在: {os.path.abspath(_VR_DIR)}/vr_receiver.py")
        print("以及依赖已安装: pip install numpy scipy")
        return 1

    # ---- 启动 VR 接收 ----
    receiver = VRControllerReceiver(port=args.port, smoothing_enabled=True)
    receiver.start()
    ips = get_local_ips()
    if ips:
        print(f"本机 IP (在 Quest APK 中填入): {', '.join(ips)}")

    # ---- 连接机械臂 ----
    arm = ELA3Interface(
        can_name=args.can,
        default_kp=args.kp,
        default_kd=args.kd,
        logger_level=LogLevel.INFO,
        gravity_feedforward_ratio=1.0,
    )

    if not arm.ConnectPort():
        receiver.stop()
        print(f"\nCAN 接口 {args.can} 连接失败")
        print("请确认:")
        print(f"  sudo ip link set {args.can} up type can bitrate 1000000")
        return 1

    # ---- CAN 总线预检 ----
    try:
        with open(f"/sys/class/net/{args.can}/tx_queue_len") as f:
            qlen = int(f.read().strip())
        if qlen < 64:
            print(f"\n[WARNING] CAN TX 队列过小 (qlen={qlen})，可能导致丢帧，建议执行:")
            print(f"  sudo ip link set {args.can} txqueuelen 128")
    except Exception:
        pass
    bus_state = arm.GetCanBusState()
    if bus_state not in ("ERROR-ACTIVE", "UNKNOWN"):
        print(f"\n[WARNING] CAN 总线状态异常: {bus_state}，建议重置总线:")
        print(f"  sudo ip link set {args.can} down && "
              f"sudo ip link set {args.can} up type can bitrate 1000000")

    arm.EnableArm()
    time.sleep(0.5)
    arm.start_control_loop(rate_hz=200.0)

    # ---- 信号处理 ----
    shutdown = threading.Event()

    def on_signal(_sig, _frame):
        shutdown.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        controller = VRArmController(
            arm=arm,
            receiver=receiver,
            update_rate=args.rate,
            scale=args.scale,
        )
    except Exception as e:
        receiver.stop()
        arm.stop_control_loop()
        arm.DisableArm()
        arm.DisconnectPort()
        print(f"\nVR 控制器初始化失败: {e}")
        print("笛卡尔控制需要 pinocchio: pip install pin")
        return 1

    ctrl_thread = threading.Thread(target=controller.start, daemon=True)
    ctrl_thread.start()

    try:
        while not shutdown.is_set() and not controller.exit_requested:
            shutdown.wait(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n正在清理...")
        controller.stop()
        receiver.stop()
        try:
            arm.ZeroTorqueMode(False)
        except Exception:
            pass
        arm.stop_control_loop()
        arm.DisableArm()
        arm.DisconnectPort()
        print("已退出")

    return 0


if __name__ == "__main__":
    exit(main())

"""
EL-A3 机械臂 SDK 主接口（纯 Python，无 ROS 依赖）

提供：
  - 连接管理 (ConnectPort / DisconnectPort)
  - 电机使能/失能 (EnableArm / DisableArm)
  - 后台高频控制循环 (200Hz)：EMA 平滑、速度前馈、重力补偿、关节限位保护
  - 运动控制 (JointCtrl / MoveJ / MoveL / EndPoseCtrl / GripperCtrl)
  - 零力矩模式 (ZeroTorqueMode / ZeroTorqueModeWithGravity)：自适应 Kd
  - 状态反馈 (GetArmJointMsgs / GetArmStatus / GetMotorStates / ...)
  - 安全控制 (EmergencyStop / ResetArm)
  - 动力学 (ComputeGravityTorques / GetJacobian / GetMassMatrix / ...)

底层使用 Robstride 私有协议（CAN 2.0 扩展帧），单位全部采用 SI（rad, m, Nm）。
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from el_a3_sdk.can_driver import RobstrideCanDriver, _busy_wait_us
from el_a3_sdk.protocol import (
    MotorType, RunMode, ControlMode, MoveMode, ModeState, ArmState,
    MotorParams, ParamIndex, LogLevel,
    MOTOR_PARAMS, DEFAULT_MOTOR_TYPE_MAP,
    DEFAULT_JOINT_DIRECTIONS, DEFAULT_JOINT_OFFSETS, DEFAULT_JOINT_LIMITS,
)
from el_a3_sdk.data_types import (
    MotorFeedback, ArmJointStates, ArmEndPose, ArmStatus,
    MotorHighSpdInfo, MotorLowSpdInfo,
    MotorAngleLimitMaxVel, MotorMaxAccLimit,
    ParamReadResult, FirmwareVersion,
    DynamicsInfo, TrajectoryResult,
)
from el_a3_sdk.utils import clamp, slerp_euler


logger = logging.getLogger("el_a3_sdk")

class ELA3Interface:
    """
    EL-A3 机械臂主接口（纯 Python SDK）

    内置 200Hz 后台控制循环，移植自 C++ el_a3_hardware 的关键特性：
    - EMA 位置平滑
    - 速度前馈（4-sample MA + 2阶 EMA + 加速度限制）
    - Pinocchio RNEA 重力补偿
    - 关节限位保护（减速 + 硬停止）
    - 自适应 Kd 零力矩模式

    用法示例::

        from el_a3_sdk import ELA3Interface

        arm = ELA3Interface(can_name="can0")
        arm.ConnectPort()
        arm.EnableArm()
        arm.JointCtrl(0.0, 1.57, -0.78, 0.0, 0.0, 0.0)
        print(arm.GetArmJointMsgs())
        arm.DisableArm()
        arm.DisconnectPort()
    """

    NUM_JOINTS = 7
    NUM_ARM_JOINTS = 6

    def __init__(
        self,
        can_name: str = "can0",
        host_can_id: int = 0xFD,
        motor_type_map: Optional[Dict[int, MotorType]] = None,
        joint_directions: Optional[Dict[int, float]] = None,
        joint_offsets: Optional[Dict[int, float]] = None,
        joint_limits: Optional[Dict[int, tuple]] = None,
        start_sdk_joint_limit: bool = True,
        torque_limits: Optional[Dict[int, float]] = None,
        torque_limit_enabled: bool = True,
        torque_soft_ratio: float = 0.6,
        torque_unlock_ratio: float = 0.8,
        default_kp: float = 100,
        default_kd: float = 4.0,
        urdf_path: Optional[str] = None,
        inertia_config_path: Optional[str] = None,
        logger_level: LogLevel = LogLevel.WARNING,
        control_rate_hz: float = 200.0,
        smoothing_alpha: float = 0.8,
        max_velocity: float = 3.0,
        max_acceleration: float = 15.0,
        velocity_limit: float = 10.0,
        gravity_feedforward_ratio: float = 1.0,
        limit_margin: float = 0.15,
        limit_stop_margin: float = 0.02,
        limit_decel_factor: float = 0.3,
        adaptive_kd_enabled: bool = True,
        zero_torque_kd_min: float = 0.001,
        zero_torque_kd_max: float = 0.15,
        per_joint_kd_min: Optional[Dict[int, float]] = None,
        per_joint_kd_max: Optional[Dict[int, float]] = None,
        kd_velocity_ref: float = 1.0,
        kd_smoothing_alpha: float = 0.15,
        pp_velocity: float = 6.0,
        pp_acceleration: float = 15.0,
        backend: str = "socketcan",
        serial_port: Optional[str] = None,
        serial_baudrate: int = 2000000,
        can_bitrate: int = 1000000,
    ):
        """
        Args:
            can_name: CAN 接口名（如 "can0"）
            host_can_id: 主机 CAN ID（默认 0xFD）
            motor_type_map: 电机 ID -> 型号映射
            joint_directions: 关节方向映射（1.0 或 -1.0）
            joint_offsets: 关节偏移映射（rad）
            joint_limits: 关节限位映射 {id: (lower, upper)} (rad)
            start_sdk_joint_limit: 是否启用 SDK 关节限位检查
            torque_limits: 关节软力矩限制映射 {motor_id: 限值 Nm}（正负对称），
                           默认 {1:13, 2:13, 3:13, 4:5, 5:5, 6:5}；L7 夹爪不纳入
            torque_limit_enabled: 是否启用软力矩限制
            torque_soft_ratio: 软阈值比例，力矩达 限值*此比例 时开始预减速
            torque_unlock_ratio: 滞回解锁比例，|tau| 降到 限值*此比例 以下时解锁
            default_kp: 默认位置增益
            default_kd: 默认速度增益
            urdf_path: URDF 路径（Pinocchio 运动学/动力学）
            inertia_config_path: 标定惯量参数 YAML 路径
            logger_level: 日志级别
            control_rate_hz: 控制循环频率 (Hz)
            smoothing_alpha: EMA 平滑系数 (0=保持, 1=直通)
            max_velocity: 最大关节速度 (rad/s)
            max_acceleration: 最大关节加速度 (rad/s²)
            velocity_limit: 速度前馈上限 (rad/s)
            gravity_feedforward_ratio: 重力补偿前馈比例 (0~1)
            limit_margin: 限位减速区宽度 (rad)
            limit_stop_margin: 限位硬停止区宽度 (rad)
            limit_decel_factor: 减速区最低增益比例 (0~1)
            adaptive_kd_enabled: 是否启用自适应 Kd
            zero_torque_kd_min: 自适应 Kd 全局下限
            zero_torque_kd_max: 自适应 Kd 全局上限
            per_joint_kd_min: 每关节 Kd 下限 {motor_id: value}，覆盖全局值
            per_joint_kd_max: 每关节 Kd 上限 {motor_id: value}，覆盖全局值
            kd_velocity_ref: 自适应 Kd 速度参考值 (rad/s)
            kd_smoothing_alpha: 自适应 Kd EMA 平滑系数
            pp_velocity: PP 模式最大速度 (rad/s)
            pp_acceleration: PP 模式加速度 (rad/s²)
            backend: CAN 驱动后端 ("socketcan" 或 "slcan")，默认 "socketcan"
            serial_port: SLCAN 串口名（如 "COM3"），backend="slcan" 时使用；
                         为 None 时使用 can_name 作为串口名
            serial_baudrate: SLCAN 串口通信波特率 (bps)，默认 2000000
            can_bitrate: CAN 总线波特率 (bps)，默认 1000000
        """
        _sdk_logger = logging.getLogger("el_a3_sdk")
        _sdk_logger.setLevel(int(logger_level))
        if not _sdk_logger.handlers:
            _handler = logging.StreamHandler()
            _handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
            _sdk_logger.addHandler(_handler)

        self._can_name = can_name
        if backend == "slcan":
            from el_a3_sdk.slcan_can_driver import SlcanCanDriver
            port = serial_port or can_name
            self._driver = SlcanCanDriver(
                serial_port=port,
                host_can_id=host_can_id,
                motor_type_map=motor_type_map,
                serial_baudrate=serial_baudrate,
                can_bitrate=can_bitrate,
            )
        else:
            self._driver = RobstrideCanDriver(
                can_name=can_name,
                host_can_id=host_can_id,
                motor_type_map=motor_type_map,
            )
        self._joint_directions = joint_directions or dict(DEFAULT_JOINT_DIRECTIONS)
        self._joint_offsets = joint_offsets or dict(DEFAULT_JOINT_OFFSETS)
        self._joint_limits = joint_limits or dict(DEFAULT_JOINT_LIMITS)
        self._joint_limit_enabled = start_sdk_joint_limit

        # 关节软力矩限制（触限冻结 + 滞回解锁）
        self._torque_limits: Dict[int, float] = (
            dict(torque_limits) if torque_limits
            else {1: 13.0, 2: 13.0, 3: 13.0, 4: 5.0, 5: 5.0, 6: 5.0}
        )
        self._torque_limit_enabled = torque_limit_enabled
        self._torque_soft_ratio = clamp(torque_soft_ratio, 0.0, 1.0)
        self._torque_unlock_ratio = clamp(torque_unlock_ratio, 0.0, 1.0)
        self._torque_locked = [0] * self.NUM_JOINTS  # 0=未锁定, +1/-1=锁定方向

        self._connected = False
        self._state = ArmState.DISCONNECTED
        self._ctrl_mode = ControlMode.STANDBY
        self._move_mode = MoveMode.MOVE_J
        self._move_spd_rate = 50

        self._position_kp = default_kp
        self._position_kd = default_kd
        self._joint_kp: Dict[int, float] = {}
        self._joint_kd: Dict[int, float] = {}

        # Pinocchio（延迟初始化）
        self._kin = None
        self._urdf_path = urdf_path
        self._inertia_config_path = inertia_config_path

        # ============ 控制循环参数 ============
        self._control_rate_hz = control_rate_hz
        self._control_period = 1.0 / control_rate_hz
        self._smoothing_alpha = clamp(smoothing_alpha, 0.01, 1.0)
        self._max_velocity = max_velocity
        self._max_acceleration = max_acceleration
        self._velocity_limit = velocity_limit
        self._gravity_feedforward_ratio = clamp(gravity_feedforward_ratio, 0.0, 1.0)

        # 关节限位保护
        self._limit_margin = limit_margin
        self._limit_stop_margin = limit_stop_margin
        self._limit_decel_factor = clamp(limit_decel_factor, 0.0, 1.0)

        # PP 模式参数
        self._pp_velocity = pp_velocity
        self._pp_acceleration = pp_acceleration

        # 零力矩 / 自适应 Kd
        self._zero_torque_mode = False
        self._zero_torque_kd = 1.0
        self._adaptive_kd_enabled = adaptive_kd_enabled
        self._zero_torque_kd_min = zero_torque_kd_min
        self._zero_torque_kd_max = zero_torque_kd_max
        self._per_joint_kd_min = per_joint_kd_min or {}
        self._per_joint_kd_max = per_joint_kd_max or {}
        self._kd_velocity_ref = kd_velocity_ref
        self._kd_smoothing_alpha = kd_smoothing_alpha

        # ============ 控制循环内部状态 ============
        self._control_running = False
        self._control_thread: Optional[threading.Thread] = None
        self._cmd_lock = threading.Lock()
        self._state_lock = threading.Lock()  # guards _state, _zero_torque_mode, _first_command, _position_kp/kd

        n = self.NUM_ARM_JOINTS
        self._target_positions = [0.0] * n
        self._target_gripper = 0.0
        self._gripper_dirty = False
        self._smoothed_positions = [0.0] * n
        self._smoothed_velocities = [0.0] * n
        self._last_cmd_positions = [0.0] * n
        self._filtered_cmd_velocities = [0.0] * n
        self._velocity_ff_stage2 = [0.0] * n
        self._vel_ma_buffer = [[0.0] * 4 for _ in range(n)]
        self._vel_ma_idx = [0] * n
        self._gravity_input_positions = [0.0] * n
        self._gravity_smooth_alpha = 0.1
        self._target_velocities = [0.0] * n
        self._adaptive_kd_values = [zero_torque_kd_max] * self.NUM_JOINTS
        self._first_command = True

        # ============ 轨迹队列 ============
        self._trajectory: Optional[List] = None
        self._traj_index = 0
        self._traj_start_time = 0.0
        self._traj_vel_reset = False
        self._traj_lock = threading.Lock()
        self._motion_done = threading.Event()
        self._motion_done.set()

        # 伺服流插值状态（VLA/RTC 流式控制）
        self._servo_last_time = 0.0    # 上一次 servo 调用时间
        self._servo_hist: List[List[float]] = []       # 最近目标历史

    # ================================================================
    # 连接管理
    # ================================================================

    def ConnectPort(self) -> bool:
        """连接机械臂并启动收发线程"""
        if self._connected:
            logger.warning("已经连接，无需重复调用 ConnectPort")
            return True

        if not self._driver.connect():
            return False

        self._driver.start_receive_thread()
        self._connected = True
        with self._state_lock:
            self._state = ArmState.IDLE
        logger.info("机械臂已连接: %s", self._can_name)
        return True

    def DisconnectPort(self):
        """断开连接并释放资源"""
        if not self._connected:
            return
        self.stop_control_loop()
        self.DisableArm()
        time.sleep(0.1)
        self._driver.disconnect()
        self._connected = False
        with self._state_lock:
            self._state = ArmState.DISCONNECTED
        logger.info("机械臂已断开: %s", self._can_name)

    def get_connect_status(self) -> bool:
        return self._connected and self._driver.is_connected

    @property
    def arm_state(self) -> ArmState:
        with self._state_lock:
            return self._state

    # ================================================================
    # 后台控制循环
    # ================================================================

    def start_control_loop(self, rate_hz: Optional[float] = None):
        """
        启动后台控制循环（类似 ros2_control 的 200Hz update）

        Args:
            rate_hz: 控制频率，None 则使用构造时的 control_rate_hz
        """
        if self._control_running:
            return
        if rate_hz is not None:
            self._control_rate_hz = rate_hz
            self._control_period = 1.0 / rate_hz

        time.sleep(0.1)

        fb_ok = False
        for attempt in range(3):
            js = self.GetArmJointMsgs()
            if js is not None and js.timestamp > 0:
                fb = js.to_list()
                with self._cmd_lock:
                    for i in range(min(len(fb), self.NUM_ARM_JOINTS)):
                        self._target_positions[i] = fb[i]
                        self._gravity_input_positions[i] = fb[i]
                logger.info("控制循环初始位置 (反馈): %s",
                            [f"{v:.3f}" for v in fb[:self.NUM_ARM_JOINTS]])
                fb_ok = True
                break
            time.sleep(0.05)
        if not fb_ok:
            logger.warning("无法获取电机反馈，_target_positions 使用默认值 [0]*6")

        self._control_running = True
        self._control_thread = threading.Thread(
            target=self._control_loop_main,
            daemon=True,
            name="el_a3_control",
        )
        self._control_thread.start()
        logger.info("控制循环已启动 (%.0f Hz)", self._control_rate_hz)

    def stop_control_loop(self):
        """停止后台控制循环"""
        if not self._control_running:
            return
        self._control_running = False
        self.cancel_motion()
        if self._control_thread:
            self._control_thread.join(timeout=2.0)
            self._control_thread = None
        logger.info("控制循环已停止")

    @property
    def control_loop_running(self) -> bool:
        return self._control_running

    def _control_loop_main(self):
        """控制循环主线程"""
        tick_count = 0
        while self._control_running and self._connected:
            t_start = time.perf_counter()

            with self._state_lock:
                state = self._state
            if state in (ArmState.RUNNING, ArmState.ZERO_TORQUE):
                self._control_loop_tick(self._control_period)

            tick_count += 1
            if tick_count % int(self._control_rate_hz) == 0:
                if not self._driver.is_bus_healthy:
                    logger.error("CAN 总线异常: %s", self._driver.check_bus_health())

            elapsed = time.perf_counter() - t_start
            sleep_time = self._control_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _control_loop_tick(self, dt: float):
        """
        单次控制周期：
        运控模式：关节限位 → 速度前馈 + 重力补偿 → send_motion_control (Type 1)
        零力矩模式：Kp=0 + 自适应 Kd + 重力补偿（Type 1）
        """
        # --- 1. 从轨迹队列取目标 ---
        with self._traj_lock:
            if self._trajectory is not None:
                elapsed = time.perf_counter() - self._traj_start_time
                while (self._traj_index < len(self._trajectory) - 1 and
                       self._trajectory[self._traj_index + 1].time <= elapsed):
                    self._traj_index += 1

                pt = self._trajectory[self._traj_index]
                with self._cmd_lock:
                    # L1-L6 从轨迹获取
                    n_arm = min(len(pt.positions), self.NUM_ARM_JOINTS)
                    for i in range(n_arm):
                        self._target_positions[i] = pt.positions[i]
                    # L7 (夹爪) 从轨迹获取（如果轨迹包含7个关节）
                    if len(pt.positions) > self.NUM_ARM_JOINTS:
                        self._target_gripper = pt.positions[self.NUM_ARM_JOINTS]
                        self._gripper_dirty = True
                    if pt.velocities:
                        n_vel = min(len(pt.velocities), self.NUM_ARM_JOINTS)
                        for i in range(n_vel):
                            self._target_velocities[i] = pt.velocities[i]

                if self._traj_index >= len(self._trajectory) - 1:
                    self._trajectory = None
                    self._traj_index = 0
                    self._motion_done.set()
                    with self._cmd_lock:
                        for i in range(self.NUM_ARM_JOINTS):
                            self._target_velocities[i] = 0.0

        with self._cmd_lock:
            target_positions = list(self._target_positions)
            gripper_dirty = self._gripper_dirty
            gripper_target = self._target_gripper
            if gripper_dirty:
                self._gripper_dirty = False

        with self._state_lock:
            zero_torque = self._zero_torque_mode
            first_cmd = self._first_command

        if first_cmd:
            with self._state_lock:
                self._first_command = False
            # 使用反馈位置初始化_last_cmd_positions，避免使用旧目标位置
            # 这确保电机不会向旧目标（如0）运动
            fb_positions = self._read_feedback_positions()
            gripper_fb = self._driver.get_feedback(self.NUM_JOINTS)
            if gripper_fb and gripper_fb.is_valid:
                direction = self._joint_directions.get(self.NUM_JOINTS, 1.0)
                offset = self._joint_offsets.get(self.NUM_JOINTS, 0.0)
                gripper_pos = (gripper_fb.position - offset) * direction
            else:
                gripper_pos = gripper_target
                logger.warning("L7反馈无效，使用gripper_target=%.3f", gripper_target)
            full_target = list(fb_positions)
            full_target.append(gripper_pos)
            self._last_cmd_positions = full_target
            self._gravity_input_positions = list(fb_positions)
            logger.info("first_cmd: _last_cmd_positions=%s, gripper_target=%.3f", 
                       [f"{v:.3f}" for v in full_target], gripper_target)

        # --- Gravity computation (EMA-smoothed, every tick) ---
        if zero_torque:
            gravity_raw_positions = self._read_feedback_positions()
        else:
            gravity_raw_positions = target_positions
        alpha = self._gravity_smooth_alpha
        for i in range(self.NUM_ARM_JOINTS):
            self._gravity_input_positions[i] = (
                alpha * gravity_raw_positions[i]
                + (1.0 - alpha) * self._gravity_input_positions[i]
            )
        gravity_torques = self._compute_gravity_vector(self._gravity_input_positions)

        # --- Read target velocities ---
        with self._cmd_lock:
            target_velocities = list(self._target_velocities)

        # --- Per-joint processing (L1-L6 + L7) ---
        for i in range(self.NUM_JOINTS):
            mid = i + 1
            direction = self._joint_directions.get(mid, 1.0)
            offset = self._joint_offsets.get(mid, 0.0)

            # L7 (夹爪) 的目标位置处理
            if i < self.NUM_ARM_JOINTS:
                pos = target_positions[i]
            else:
                # L7 使用 gripper_target
                pos = gripper_target

            if self._joint_limit_enabled:
                limits = self._joint_limits.get(mid)
                if limits:
                    lo_stop = limits[0] + self._limit_stop_margin
                    hi_stop = limits[1] - self._limit_stop_margin
                    pos = clamp(pos, lo_stop, hi_stop)

            motor_pos = pos * direction + offset

            if zero_torque:
                fb = self._driver.get_feedback(mid)
                current_motor_pos = fb.position if (fb and fb.is_valid) else motor_pos
                motor_kp = 0.0
                if self._adaptive_kd_enabled:
                    vel = fb.velocity * direction if (fb and fb.is_valid) else 0.0
                    self._adaptive_kd_values[i] = self._compute_adaptive_kd(
                        i, vel, self._adaptive_kd_values[i])
                    motor_kd = self._adaptive_kd_values[i]
                else:
                    motor_kd = clamp(self._zero_torque_kd, 0.0, 5.0)

                grav_torque = gravity_torques[i] * direction if gravity_torques else 0.0
                self._driver.send_motion_control(
                    mid, current_motor_pos, 0.0, motor_kp, motor_kd, grav_torque)
            else:
                if i < self.NUM_ARM_JOINTS:
                    vel_from_caller = target_velocities[i]
                else:
                    vel_from_caller = 0.0

                if abs(vel_from_caller) > 1e-6:
                    vel_ff = vel_from_caller
                else:
                    if i < self.NUM_ARM_JOINTS:
                        diff = pos - self._last_cmd_positions[i]
                    else:
                        diff = pos - self._last_cmd_positions[i] if i < len(self._last_cmd_positions) else 0.0
                    vel_ff = diff / dt if dt > 0 else 0.0
                    if abs(diff) > 0.05:
                        vel_ff = 0.0
                vel_ff = clamp(vel_ff, -self._velocity_limit, self._velocity_limit)

                # --- 关节软力矩限制（触限冻结 + 滞回解锁，仅 L1-L6）---
                if (i < self.NUM_ARM_JOINTS and self._torque_limit_enabled
                        and mid in self._torque_limits):
                    t_limit = abs(self._torque_limits[mid])
                    if t_limit > 0:
                        fb = self._driver.get_feedback(mid)
                        if fb and fb.is_valid:
                            tau = fb.torque * direction  # 关节系力矩
                            fb_joint_pos = (fb.position - offset) * direction
                            if abs(pos - fb_joint_pos) > 1e-4:
                                cmd_dir = 1 if pos > fb_joint_pos else -1
                            elif abs(vel_ff) > 1e-3:
                                cmd_dir = 1 if vel_ff > 0 else -1
                            else:
                                cmd_dir = 0

                            lock = self._torque_locked[i]
                            if lock != 0 and (abs(tau) < t_limit * self._torque_unlock_ratio
                                              or (cmd_dir != 0 and cmd_dir != lock)):
                                self._torque_locked[i] = 0
                                lock = 0
                                logger.info("电机 %d 力矩解锁 (tau=%.2f Nm)", mid, tau)
                            if lock == 0:
                                if tau >= t_limit and cmd_dir >= 0:
                                    lock = 1
                                elif tau <= -t_limit and cmd_dir <= 0:
                                    lock = -1
                                if lock != 0:
                                    self._torque_locked[i] = lock
                                    logger.warning(
                                        "电机 %d 关节力矩 %.2f Nm 触限 %.1f Nm，冻结%s方向",
                                        mid, tau, t_limit, "正向" if lock > 0 else "反向")
                            if lock != 0:
                                # 锁定中（反向命令已在上方解锁）：冻结在反馈位置，
                                # 保留 kp/kd 位置保持 + 重力前馈，防止卸力下坠
                                pos = fb_joint_pos
                                motor_pos = pos * direction + offset
                                vel_ff = 0.0
                            elif lock == 0 and cmd_dir != 0:
                                # 软阈值区预减速（与夹爪力矩限制策略一致）
                                soft = t_limit * self._torque_soft_ratio
                                if t_limit > soft:
                                    if cmd_dir > 0 and tau > soft and vel_ff > 0:
                                        vel_ff *= clamp((t_limit - tau) / (t_limit - soft), 0.0, 1.0)
                                    elif cmd_dir < 0 and tau < -soft and vel_ff < 0:
                                        vel_ff *= clamp((t_limit + tau) / (t_limit - soft), 0.0, 1.0)

                motor_vel = vel_ff * direction

                motor_kp = self._joint_kp.get(mid, self._position_kp)
                motor_kd = self._joint_kd.get(mid, self._position_kd)
                if abs(vel_ff) < 0.01:
                    motor_kd = min(motor_kd * 1.25, 5.0)

                grav_torque = gravity_torques[i] * direction * self._gravity_feedforward_ratio if gravity_torques else 0.0
                self._driver.send_motion_control(
                    mid, motor_pos, motor_vel, motor_kp, motor_kd, grav_torque)

            _busy_wait_us(150)

        # 更新 last_cmd_positions 包含 L7
        full_target = list(target_positions[:self.NUM_ARM_JOINTS])
        full_target.append(gripper_target)
        self._last_cmd_positions = full_target

        # --- Gripper (仅在非零力矩模式下发送PP指令，零力矩时已在上面处理) ---
        if gripper_dirty and not zero_torque:
            gripper_motor_id = 7
            self._driver.set_position_pp(gripper_motor_id, gripper_target)

    def _read_feedback_positions(self) -> List[float]:
        """读取所有臂关节的反馈位置（URDF 坐标系）"""
        positions = [0.0] * self.NUM_ARM_JOINTS
        for i in range(self.NUM_ARM_JOINTS):
            mid = i + 1
            fb = self._driver.get_feedback(mid)
            if fb and fb.is_valid:
                direction = self._joint_directions.get(mid, 1.0)
                offset = self._joint_offsets.get(mid, 0.0)
                positions[i] = (fb.position - offset) * direction
        return positions

    def _sync_command_targets_from_feedback(self, retries: int = 5, delay_s: float = 0.02) -> List[float]:
        """将控制缓存重同步到最新反馈，避免旧目标在设零后继续生效。"""
        feedback = None
        for _ in range(max(1, retries)):
            js = self.GetArmJointMsgs()
            if js is not None and js.timestamp > 0:
                feedback = js.to_list(include_gripper=True)
                break
            time.sleep(delay_s)

        if feedback is None:
            feedback = self._read_feedback_positions()
            gripper_fb = self._driver.get_feedback(self.NUM_JOINTS)
            if gripper_fb and gripper_fb.is_valid:
                direction = self._joint_directions.get(self.NUM_JOINTS, 1.0)
                offset = self._joint_offsets.get(self.NUM_JOINTS, 0.0)
                feedback.append((gripper_fb.position - offset) * direction)
            else:
                feedback.append(self._target_gripper)

        arm_feedback = list(feedback[:self.NUM_ARM_JOINTS])
        gripper_feedback = feedback[self.NUM_ARM_JOINTS] if len(feedback) > self.NUM_ARM_JOINTS else self._target_gripper

        with self._cmd_lock:
            self._target_positions = list(arm_feedback)
            self._last_cmd_positions = list(arm_feedback)
            self._gravity_input_positions = list(arm_feedback)
            self._target_velocities = [0.0] * self.NUM_ARM_JOINTS
            self._target_gripper = gripper_feedback
            self._gripper_dirty = False

        with self._state_lock:
            self._first_command = True

        return arm_feedback

    def _compute_gravity_vector(self, positions: List[float]) -> Optional[List[float]]:
        """计算 6 自由度重力补偿力矩"""
        kin = self._get_kinematics()
        if kin is None:
            return None
        try:
            return kin.compute_gravity(positions)
        except Exception:
            return None

    def _compute_adaptive_kd(self, joint_idx: int, velocity: float, prev_kd: float) -> float:
        """
        自适应 Kd 计算（洛伦兹衰减 + EMA 平滑）
        支持 per-joint kd_min/kd_max 覆盖全局值（与 ROS computeAdaptiveKd 一致）

        kd_raw = kd_min + (kd_max - kd_min) / (1 + (|v| / v_ref)^2)
        kd = alpha * kd_raw + (1 - alpha) * prev_kd
        """
        mid = joint_idx + 1
        kd_min = self._per_joint_kd_min.get(mid, self._zero_torque_kd_min)
        kd_max = self._per_joint_kd_max.get(mid, self._zero_torque_kd_max)
        v = abs(velocity)
        ratio = v / max(self._kd_velocity_ref, 1e-6)
        kd_raw = kd_min + (kd_max - kd_min) / (1.0 + ratio * ratio)
        kd = self._kd_smoothing_alpha * kd_raw + (1.0 - self._kd_smoothing_alpha) * prev_kd
        return clamp(kd, 0.0, 5.0)

    # ================================================================
    # 电机使能 / 失能
    # ================================================================

    def EnableArm(self, motor_num: int = 0xFF, run_mode: RunMode = RunMode.MOTION_CONTROL,
                  startup_kd: float = 4.0) -> bool:
        """
        使能电机

        Args:
            motor_num: 电机编号（1-7 单个，0xFF=全部）
            run_mode: 运行模式（默认运控模式）
            startup_kd: 软启动阻尼系数（运控模式）
        """
        if not self._connected:
            logger.error("未连接，请先调用 ConnectPort()")
            return False

        motor_ids = self._resolve_motor_ids(motor_num)
        success = True

        for mid in motor_ids:
            is_gripper = (mid == self.NUM_JOINTS)
            actual_mode = RunMode.POSITION_PP if is_gripper else run_mode

            self._driver.disable_motor(mid, clear_fault=True)
            time.sleep(0.03)

            if not self._driver.set_run_mode(mid, actual_mode):
                logger.error("电机 %d 设置运行模式失败", mid)
                success = False
                continue
            time.sleep(0.03)

            if not self._driver.enable_motor(mid):
                logger.error("电机 %d 使能失败", mid)
                success = False
                continue

            if actual_mode == RunMode.POSITION_PP:
                self._driver.set_pp_velocity(mid, self._pp_velocity)
                time.sleep(0.005)
                self._driver.set_pp_acceleration(mid, self._pp_acceleration)
                time.sleep(0.005)
            elif actual_mode == RunMode.MOTION_CONTROL:
                self._driver.write_parameter(mid, ParamIndex.CAN_TIMEOUT, 0.0)
                time.sleep(0.005)
                fb = self._driver.get_feedback(mid)
                init_pos = fb.position if (fb and fb.is_valid) else 0.0
                self._driver.send_motion_control(mid, init_pos, 0.0, 5, startup_kd, 0.0)

            logger.info("电机 %d 已使能 (mode=%s)", mid, actual_mode.name)
            time.sleep(0.03)

        if success:
            with self._state_lock:
                self._state = ArmState.ENABLED
                self._first_command = True
        return success

    def DisableArm(self, motor_num: int = 0xFF) -> bool:
        """失能电机"""
        if not self._connected:
            return False

        self.cancel_motion()
        motor_ids = self._resolve_motor_ids(motor_num)
        success = True

        for mid in motor_ids:
            if not self._driver.disable_motor(mid):
                success = False
            time.sleep(0.005)

        if success and motor_num == 0xFF:
            with self._state_lock:
                self._state = ArmState.IDLE
                self._zero_torque_mode = False
        return success

    # ================================================================
    # 安全控制
    # ================================================================

    def EmergencyStop(self) -> bool:
        """急停：立即失能所有电机"""
        if not self._connected:
            return False

        self.cancel_motion()
        success = True
        for mid in range(1, self.NUM_JOINTS + 1):
            if not self._driver.disable_motor(mid, clear_fault=True):
                success = False

        with self._state_lock:
            self._zero_torque_mode = False
            self._state = ArmState.IDLE
        logger.warning("急停已执行！所有电机已失能")
        return success

    def ResetArm(self) -> bool:
        """复位机械臂"""
        success = self.EmergencyStop()
        self._ctrl_mode = ControlMode.STANDBY
        self._move_mode = MoveMode.MOVE_J
        with self._state_lock:
            self._state = ArmState.IDLE
        logger.info("机械臂已复位")
        return success

    # ================================================================
    # 模式控制
    # ================================================================

    def ModeCtrl(
        self,
        ctrl_mode: "ControlMode | int" = ControlMode.CAN_COMMAND,
        move_mode: "MoveMode | int" = MoveMode.MOVE_J,
        move_spd_rate_ctrl: int = 50,
    ):
        """设置控制模式和运动模式"""
        self._ctrl_mode = ControlMode(int(ctrl_mode))
        self._move_mode = MoveMode(int(move_mode))
        self._move_spd_rate = clamp(move_spd_rate_ctrl, 0, 100)

        if self._ctrl_mode == ControlMode.STANDBY:
            logger.info("切换到待机模式")
            return

        run_mode_map = {
            MoveMode.MOVE_J: RunMode.MOTION_CONTROL,
            MoveMode.MOVE_CSP: RunMode.POSITION_CSP,
            MoveMode.MOVE_VELOCITY: RunMode.VELOCITY,
            MoveMode.MOVE_CURRENT: RunMode.CURRENT,
        }
        target_run_mode = run_mode_map.get(self._move_mode, RunMode.MOTION_CONTROL)

        for mid in range(1, self.NUM_JOINTS + 1):
            self._driver.disable_motor(mid, clear_fault=False)
            time.sleep(0.03)
            self._driver.set_run_mode(mid, target_run_mode)
            time.sleep(0.03)
            self._driver.enable_motor(mid)
            time.sleep(0.03)

        logger.info("模式已设置: ctrl=%s, move=%s, spd_rate=%d%%",
                     self._ctrl_mode.name, self._move_mode.name, self._move_spd_rate)

    # ================================================================
    # 关节控制
    # ================================================================

    def JointCtrl(
        self,
        joint_1: float, joint_2: float, joint_3: float,
        joint_4: float, joint_5: float, joint_6: float,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
        velocity: float = 0.0,
        torque_ff: Optional[List[float]] = None,
        velocities: Optional[List[float]] = None,
    ) -> bool:
        """
        关节角度控制（运控模式 Type 1）

        如果控制循环正在运行，设置目标位置和速度前馈（由控制循环发送运控帧）。
        否则直接发送运控帧。

        Args:
            joint_1~6: 目标关节角度 (rad)
            kp, kd: PD 增益覆盖（None 则使用默认值）
            velocity: 统一速度前馈 (rad/s)，仅非控制循环模式
            torque_ff: 各关节力矩前馈 (Nm)，长度 6
            velocities: 各关节速度前馈 (rad/s)，长度 6，控制循环优先使用
        """
        if not self._connected:
            logger.error("未连接")
            return False
        with self._state_lock:
            if self._state not in (ArmState.ENABLED, ArmState.RUNNING, ArmState.IDLE):
                logger.error("当前状态 %s 不允许关节控制，请先 EnableArm()", self._state.name)
                return False
            self._state = ArmState.RUNNING

        positions = [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]

        if self._control_running:
            with self._cmd_lock:
                for i in range(self.NUM_ARM_JOINTS):
                    self._target_positions[i] = positions[i]
                if velocities and len(velocities) >= self.NUM_ARM_JOINTS:
                    for i in range(self.NUM_ARM_JOINTS):
                        self._target_velocities[i] = velocities[i]
                else:
                    for i in range(self.NUM_ARM_JOINTS):
                        self._target_velocities[i] = 0.0
            return True

        motor_kp = kp if kp is not None else self._position_kp
        motor_kd = kd if kd is not None else self._position_kd

        success = True
        for i, mid in enumerate(range(1, self.NUM_ARM_JOINTS + 1)):
            target_pos = positions[i]
            if self._joint_limit_enabled:
                limits = self._joint_limits.get(mid)
                if limits:
                    target_pos = clamp(target_pos, limits[0], limits[1])

            direction = self._joint_directions.get(mid, 1.0)
            offset = self._joint_offsets.get(mid, 0.0)
            motor_pos = target_pos * direction + offset
            motor_vel = velocity * direction
            torque = torque_ff[i] * direction if torque_ff and i < len(torque_ff) else 0.0

            if not self._driver.send_motion_control(
                    mid, motor_pos, motor_vel, motor_kp, motor_kd, torque):
                success = False
            _busy_wait_us(150)

        return success

    def JointCtrlList(self, positions: List[float], **kwargs) -> bool:
        """便捷接口：列表形式的关节控制（6=仅手臂，7=手臂+夹爪）"""
        n = len(positions)
        if n < self.NUM_ARM_JOINTS or n > self.NUM_JOINTS:
            logger.error("positions 长度必须为 %d 或 %d", self.NUM_ARM_JOINTS, self.NUM_JOINTS)
            return False
        result = self.JointCtrl(*positions[:6], **kwargs)
        if n >= 7:
            result = self.GripperCtrl(gripper_angle=positions[6]) and result
        return result

    # ================================================================
    # 夹爪控制
    # ================================================================

    def GripperCtrl(
        self,
        gripper_angle: float = 0.0,
        gripper_effort: float = 0.0,
        gripper_enable: bool = True,
        set_zero: bool = False,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
    ) -> bool:
        """夹爪控制"""
        gripper_motor_id = 7

        if set_zero:
            return self._driver.set_zero_position(gripper_motor_id)
        if not gripper_enable:
            return self._driver.disable_motor(gripper_motor_id)

        if self._control_running:
            with self._cmd_lock:
                self._target_gripper = gripper_angle
                self._gripper_dirty = True
            return True

        # 如果零力矩模式已启用（L7处于运控模式），发送运控帧而非PP帧
        if self._zero_torque_mode:
            fb = self._driver.get_feedback(gripper_motor_id)
            current_pos = fb.position if (fb and fb.is_valid) else 0.0
            direction = self._joint_directions.get(gripper_motor_id, 1.0)
            motor_torque = gripper_effort * direction
            return self._driver.send_motion_control(
                gripper_motor_id, current_pos, 0.0, 0.0,
                self._zero_torque_kd, motor_torque)

        return self._driver.set_position_pp(gripper_motor_id, gripper_angle)

    # ================================================================
    # 零力矩模式
    # ================================================================

    def ZeroTorqueMode(self, enable: bool, kd: float = 1.0,
                       gravity_torques: Optional[List[float]] = None) -> bool:
        """
        零力矩模式（Kp=0，可手动拖动机械臂，包括L7夹爪关节）

        进入零力矩时所有关节（L1-L7）都切换到运控模式（Type 1 帧）。
        退出时 L1-L6 保持运控模式，使用当前位置作为目标恢复阻抗控制；
        L7 夹爪恢复 PP 位置模式（与 EnableArm 的约定一致）——否则夹爪
        滞留在运控模式，GripperCtrl 的 PP 指令被电机拒绝，实际由 200Hz
        Type-1 帧以臂级增益 kp=80 驱动，控制严重顿挫。
        如果控制循环正在运行，由控制循环以自适应 Kd 持续发送。
        否则发送单次指令。
        """
        if not self._connected:
            return False
        with self._state_lock:
            if enable and self._state not in (ArmState.ENABLED, ArmState.RUNNING, ArmState.ZERO_TORQUE):
                logger.error("当前状态 %s 不允许零力矩，请先 EnableArm()", self._state.name)
                return False

            self._zero_torque_mode = enable
            self._zero_torque_kd = kd

            if enable:
                self._state = ArmState.ZERO_TORQUE

        if enable:
            # 零力矩模式下所有关节（包括L7夹爪）都切换到运控模式
            # 以便L7也能被拖动
            for mid in range(1, self.NUM_JOINTS + 1):
                target_mode = RunMode.MOTION_CONTROL
                current_mode = self._driver.get_run_mode(mid)
                fb = self._driver.get_feedback(mid)
                is_enabled = fb and fb.is_valid and fb.mode_state == 2

                if is_enabled and current_mode is not None and current_mode == int(target_mode):
                    continue

                self._driver.disable_motor(mid, clear_fault=False)
                time.sleep(0.01)
                self._driver.set_run_mode(mid, target_mode)
                time.sleep(0.01)
                self._driver.enable_motor(mid)
                time.sleep(0.01)

            for i in range(self.NUM_JOINTS):
                self._adaptive_kd_values[i] = self._zero_torque_kd_max

            if not self._control_running:
                grav = list(gravity_torques or [0.0] * self.NUM_JOINTS)
                for i, mid in enumerate(range(1, self.NUM_JOINTS + 1)):
                    fb = self._driver.get_feedback(mid)
                    current_pos = fb.position if (fb and fb.is_valid) else 0.0
                    direction = self._joint_directions.get(mid, 1.0)
                    motor_torque = grav[i] * direction if i < len(grav) else 0.0
                    self._driver.send_motion_control(mid, current_pos, 0.0, 0.0, kd, motor_torque)
                    _busy_wait_us(150)
            logger.warning("零力矩模式已启用 (Kd=%.3f, adaptive=%s)",
                           kd, "ON" if self._adaptive_kd_enabled else "OFF")
        else:
            # 获取当前反馈位置，用于同步目标位置
            current_positions = self._read_feedback_positions()
            # 同时读取L7夹爪的当前位置
            gripper_fb = self._driver.get_feedback(self.NUM_JOINTS)
            if gripper_fb and gripper_fb.is_valid:
                direction = self._joint_directions.get(self.NUM_JOINTS, 1.0)
                offset = self._joint_offsets.get(self.NUM_JOINTS, 0.0)
                current_gripper_pos = (gripper_fb.position - offset) * direction
            else:
                current_gripper_pos = self._target_gripper

            # 同步目标位置到当前位置，避免回到旧目标（如0位置）
            with self._cmd_lock:
                for i in range(min(len(current_positions), self.NUM_ARM_JOINTS)):
                    self._target_positions[i] = current_positions[i]
                    self._last_cmd_positions[i] = current_positions[i]
                    self._target_velocities[i] = 0.0
                # 同步L7夹爪目标位置到当前位置
                self._target_gripper = current_gripper_pos
                self._gripper_dirty = False
            logger.info("零力矩退出: L1-L6目标=%s, L7目标=%.3f", 
                       [f"{v:.3f}" for v in current_positions], current_gripper_pos)

            # 退出零力矩时，L1-L6 保持运控模式，使用当前位置作为目标，
            # 开启关节空间阻抗控制（力矩 = Kp*(当前位置-目标位置) + Kd*(速度-0) + 重力补偿）
            for mid in range(1, self.NUM_ARM_JOINTS + 1):
                fb = self._driver.get_feedback(mid)
                motor_pos = fb.position if (fb and fb.is_valid) else 0.0

                target_mode = RunMode.MOTION_CONTROL
                current_mode = self._driver.get_run_mode(mid)
                is_enabled = fb and fb.is_valid and fb.mode_state == 2

                if is_enabled and current_mode is not None and current_mode == int(target_mode):
                    # 保持运控模式，使用当前位置作为目标，开启关节空间阻抗控制
                    self._driver.write_parameter(mid, ParamIndex.CAN_TIMEOUT, 0.0)
                    time.sleep(0.005)
                    self._driver.send_motion_control(
                        mid, motor_pos, 0.0, self._position_kp, self._position_kd, 0.0)
                    time.sleep(0.01)
                    continue

                # 如果模式不对，才需要重新配置电机
                self._driver.disable_motor(mid, clear_fault=False)
                time.sleep(0.01)
                self._driver.set_run_mode(mid, target_mode)
                time.sleep(0.01)
                self._driver.enable_motor(mid)
                time.sleep(0.005)
                self._driver.write_parameter(mid, ParamIndex.CAN_TIMEOUT, 0.0)
                time.sleep(0.005)
                self._driver.send_motion_control(
                    mid, motor_pos, 0.0, self._position_kp, self._position_kd, 0.0)
                time.sleep(0.01)

            # L7 夹爪恢复 PP 位置模式（与 EnableArm 的约定一致）：
            # 进入零力矩时 L7 被切到运控模式以便拖动，若退出后滞留在运控模式，
            # GripperCtrl 的 PP 指令（set_position_pp）会被电机拒绝，夹爪实际
            # 由 200Hz Type-1 帧以臂级增益 kp=80 刚性驱动，手柄控制严重顿挫。
            # 恢复 PP 后固件重新按 vel/acc 轮廓规划，夹爪控制恢复平滑。
            gripper_id = self.NUM_JOINTS
            self._driver.disable_motor(gripper_id, clear_fault=False)
            time.sleep(0.01)
            self._driver.set_run_mode(gripper_id, RunMode.POSITION_PP)
            time.sleep(0.01)
            self._driver.enable_motor(gripper_id)
            time.sleep(0.005)
            self._driver.set_pp_velocity(gripper_id, self._pp_velocity)
            time.sleep(0.005)
            self._driver.set_pp_acceleration(gripper_id, self._pp_acceleration)
            time.sleep(0.005)
            # 以当前位置为 PP 目标保位，避免恢复后夹爪跳动
            self._driver.set_position_pp(gripper_id, current_gripper_pos)

            with self._state_lock:
                self._state = ArmState.ENABLED
                self._first_command = True
            logger.info(
                "零力矩模式已关闭：L1-L6 关节空间阻抗模式（目标锁定当前位置），"
                "L7 恢复 PP 位置模式（目标=%.3f rad）", current_gripper_pos)

        return True

    def ZeroTorqueModeWithGravity(
        self, enable: bool, kd=1.0, update_rate: float = 100.0,
    ) -> bool:
        """
        带重力补偿的零力矩模式

        推荐使用控制循环模式（start_control_loop），控制循环会自动计算
        重力补偿并以自适应 Kd 发送。

        如果控制循环未运行，则启动后台线程以指定频率更新。
        """
        if not self._connected:
            return False

        if enable:
            if isinstance(kd, (list, tuple)):
                kd_list = list(kd)
                if len(kd_list) < self.NUM_JOINTS:
                    kd_list += [kd_list[-1]] * (self.NUM_JOINTS - len(kd_list))
            else:
                kd_list = [float(kd)] * self.NUM_JOINTS

            if self._control_running:
                self._zero_torque_kd = kd_list[0]
                return self.ZeroTorqueMode(True, kd=kd_list[0])

            kin = self._get_kinematics()
            if kin is None:
                logger.warning("Pinocchio 不可用，使用无重力补偿零力矩模式")
                return self.ZeroTorqueMode(True, kd=kd_list[0])

            self.ZeroTorqueMode(True, kd=kd_list[0])
            self._zt_gravity_running = True
            self._zt_gravity_thread = threading.Thread(
                target=self._zero_torque_gravity_loop,
                args=(kd_list, 1.0 / update_rate),
                daemon=True,
                name="zero_torque_gravity",
            )
            self._zt_gravity_thread.start()
            return True
        else:
            self._zt_gravity_running = False
            if hasattr(self, '_zt_gravity_thread') and self._zt_gravity_thread:
                self._zt_gravity_thread.join(timeout=1.0)
                self._zt_gravity_thread = None
            return self.ZeroTorqueMode(False)

    def _zero_torque_gravity_loop(self, kd_list: list, dt: float):
        """零力矩 + 重力补偿后台循环（仅在控制循环未运行时使用）"""
        kin = self._get_kinematics()
        smoothed = list(self._gravity_input_positions)
        q_init = self.GetArmJointMsgs().to_list()
        for i in range(min(len(q_init), len(smoothed))):
            smoothed[i] = q_init[i]

        alpha = self._gravity_smooth_alpha
        while self._zt_gravity_running and self._connected:
            q = self.GetArmJointMsgs().to_list()
            for i in range(min(len(q), self.NUM_ARM_JOINTS)):
                smoothed[i] = alpha * q[i] + (1.0 - alpha) * smoothed[i]
            arm_grav = kin.compute_gravity(smoothed[:self.NUM_ARM_JOINTS]) if kin else [0.0] * self.NUM_ARM_JOINTS
            grav = list(arm_grav) + [0.0] * (self.NUM_JOINTS - len(arm_grav))

            for i, mid in enumerate(range(1, self.NUM_JOINTS + 1)):
                fb = self._driver.get_feedback(mid)
                current_pos = fb.position if (fb and fb.is_valid) else 0.0
                direction = self._joint_directions.get(mid, 1.0)
                motor_torque = grav[i] * direction

                vel = fb.velocity * direction if (fb and fb.is_valid) else 0.0
                if self._adaptive_kd_enabled:
                    self._adaptive_kd_values[i] = self._compute_adaptive_kd(
                        i, vel, self._adaptive_kd_values[i])
                    use_kd = self._adaptive_kd_values[i]
                else:
                    use_kd = kd_list[i]

                self._driver.send_motion_control(
                    mid, current_pos, 0.0, 0.0, use_kd, motor_torque)

            time.sleep(dt)

    # ================================================================
    # 主从配置
    # ================================================================

    def MasterSlaveConfig(self, mode: int = 0xFC, *args) -> bool:
        if mode == 0xFD:
            return self.ZeroTorqueMode(True, kd=self._zero_torque_kd)
        else:
            return self.ZeroTorqueMode(False)

    # ================================================================
    # 反馈函数
    # ================================================================

    def GetArmJointMsgs(self) -> ArmJointStates:
        """获取关节角度反馈 (rad)"""
        positions = []
        latest_ts = 0.0
        for mid in range(1, self.NUM_JOINTS + 1):
            fb = self._driver.get_feedback(mid)
            if fb and fb.is_valid:
                direction = self._joint_directions.get(mid, 1.0)
                offset = self._joint_offsets.get(mid, 0.0)
                joint_pos = (fb.position - offset) * direction
                positions.append(joint_pos)
                latest_ts = max(latest_ts, fb.timestamp)
            else:
                positions.append(0.0)
        return ArmJointStates.from_list(positions, timestamp=latest_ts)

    def GetArmJointVelocities(self) -> ArmJointStates:
        """获取关节速度反馈 (rad/s)"""
        velocities = []
        latest_ts = 0.0
        for mid in range(1, self.NUM_JOINTS + 1):
            fb = self._driver.get_feedback(mid)
            if fb and fb.is_valid:
                direction = self._joint_directions.get(mid, 1.0)
                velocities.append(fb.velocity * direction)
                latest_ts = max(latest_ts, fb.timestamp)
            else:
                velocities.append(0.0)
        return ArmJointStates.from_list(velocities, timestamp=latest_ts)

    def GetArmJointEfforts(self) -> ArmJointStates:
        """获取关节力矩反馈 (Nm)"""
        efforts = []
        latest_ts = 0.0
        for mid in range(1, self.NUM_JOINTS + 1):
            fb = self._driver.get_feedback(mid)
            if fb and fb.is_valid:
                direction = self._joint_directions.get(mid, 1.0)
                efforts.append(fb.torque * direction)
                latest_ts = max(latest_ts, fb.timestamp)
            else:
                efforts.append(0.0)
        return ArmJointStates.from_list(efforts, timestamp=latest_ts)

    def GetArmStatus(self) -> ArmStatus:
        """获取机械臂综合状态"""
        status = ArmStatus()
        status.ctrl_mode = int(self._ctrl_mode)
        status.move_mode = int(self._move_mode)
        for i, mid in enumerate(range(1, self.NUM_JOINTS + 1)):
            fb = self._driver.get_feedback(mid)
            if fb and fb.is_valid:
                status.joint_enabled[i] = (fb.mode_state == ModeState.MOTOR)
                status.joint_faults[i] = fb.fault_code
                status.joint_mode_states[i] = fb.mode_state
                status.timestamp = max(status.timestamp, fb.timestamp)
        if status.has_fault:
            status.arm_status = 0x05
        elif not status.all_enabled:
            status.arm_status = 0x01
        else:
            status.arm_status = 0x00
        return status

    def GetArmEnableStatus(self) -> List[bool]:
        result = []
        for mid in range(1, self.NUM_JOINTS + 1):
            fb = self._driver.get_feedback(mid)
            result.append(fb.mode_state == ModeState.MOTOR if fb and fb.is_valid else False)
        return result

    def GetArmHighSpdInfoMsgs(self) -> List[MotorHighSpdInfo]:
        result = []
        for mid in range(1, self.NUM_JOINTS + 1):
            fb = self._driver.get_feedback(mid)
            info = MotorHighSpdInfo(motor_id=mid)
            if fb and fb.is_valid:
                direction = self._joint_directions.get(mid, 1.0)
                info.speed = fb.velocity * direction
                info.position = (fb.position - self._joint_offsets.get(mid, 0.0)) * direction
                info.torque = fb.torque * direction
                info.timestamp = fb.timestamp
            result.append(info)
        return result

    def GetArmLowSpdInfoMsgs(self) -> List[MotorLowSpdInfo]:
        result = []
        for mid in range(1, self.NUM_JOINTS + 1):
            fb = self._driver.get_feedback(mid)
            info = MotorLowSpdInfo(motor_id=mid)
            if fb and fb.is_valid:
                info.motor_temp = fb.temperature
                info.fault_code = fb.fault_code
                info.timestamp = fb.timestamp
            result.append(info)
        return result

    def GetMotorStates(self) -> Dict[int, MotorFeedback]:
        return self._driver.get_all_feedbacks()

    # ================================================================
    # 参数查询
    # ================================================================

    def SearchMotorMaxAngleSpdAccLimit(
        self, motor_num: int = 1, search_content: int = 0x01
    ) -> Optional[ParamReadResult]:
        if search_content == 0x01:
            return self._driver.read_parameter(motor_num, ParamIndex.LIMIT_SPD)
        elif search_content == 0x02:
            return self._driver.read_parameter(motor_num, ParamIndex.ACC_RAD)
        return None

    def GetCurrentMotorAngleLimitMaxVel(self) -> List[MotorAngleLimitMaxVel]:
        result = []
        for mid in range(1, self.NUM_JOINTS + 1):
            limits = self._joint_limits.get(mid, (-6.28, 6.28))
            motor_type = self._driver.motor_type_map.get(mid, MotorType.RS00)
            params = MOTOR_PARAMS[motor_type]
            result.append(MotorAngleLimitMaxVel(
                motor_num=mid,
                min_angle_limit=limits[0],
                max_angle_limit=limits[1],
                max_joint_spd=params.v_max,
            ))
        return result

    def GetAllMotorMaxAccLimit(self) -> List[MotorMaxAccLimit]:
        result = []
        for mid in range(1, self.NUM_JOINTS + 1):
            pr = self._driver.read_parameter(mid, ParamIndex.ACC_RAD, timeout=0.3)
            result.append(MotorMaxAccLimit(
                motor_num=mid,
                max_joint_acc=pr.value if (pr and pr.success) else 20.0,
            ))
        return result

    def ReadMotorParameter(self, motor_id: int, param_index: int) -> Optional[ParamReadResult]:
        return self._driver.read_parameter(motor_id, param_index)

    def WriteMotorParameter(self, motor_id: int, param_index: int, value: float) -> bool:
        return self._driver.write_parameter(motor_id, param_index, value)

    def WriteMotorParameterInt(self, motor_id: int, param_index: int, value: int) -> bool:
        """写入整数类型参数（uint8/uint16/uint32）"""
        return self._driver.write_parameter_int(motor_id, param_index, value)

    def GetMotorVoltage(self, motor_id: int) -> Optional[float]:
        result = self._driver.read_parameter(motor_id, ParamIndex.VBUS, timeout=0.3)
        return result.value if (result and result.success) else None

    def GetFirmwareVersion(self, motor_id: int = 1) -> Optional[FirmwareVersion]:
        return self._driver.query_firmware_version(motor_id)

    def GetAllFirmwareVersions(self) -> Dict[int, FirmwareVersion]:
        result = {}
        for mid in range(1, self.NUM_JOINTS + 1):
            ver = self._driver.query_firmware_version(mid, timeout=0.3)
            if ver:
                result[mid] = ver
        return result

    # ================================================================
    # 零位设置
    # ================================================================

    def SetZeroPosition(self, motor_num: int = 0xFF) -> bool:
        was_control_running = self._control_running
        if was_control_running:
            self.stop_control_loop()

        motor_ids = self._resolve_motor_ids(motor_num)
        success = True
        try:
            for mid in motor_ids:
                if not self._driver.set_zero_position(mid):
                    success = False
                time.sleep(0.05)

            # 给反馈一点时间刷新，再将控制目标对齐到新的零点坐标系。
            time.sleep(0.05)
            synced = self._sync_command_targets_from_feedback()
            logger.info("设零后目标已重同步: %s", [f"{v:.3f}" for v in synced])
        finally:
            if was_control_running:
                self.start_control_loop(rate_hz=self._control_rate_hz)
        return success

    # ================================================================
    # SDK 信息
    # ================================================================

    def GetCanFps(self) -> float:
        return self._driver.get_can_fps()

    def GetCanTxStats(self):
        """返回 (成功数, 失败数, 最近 1 秒失败率)"""
        return self._driver.get_tx_stats()

    def GetCanBusState(self) -> str:
        """返回 CAN 总线状态字符串"""
        return self._driver.check_bus_health()

    def GetCanName(self) -> str:
        return self._can_name

    def GetCurrentSDKVersion(self) -> str:
        from el_a3_sdk import __version__
        return __version__

    def GetCurrentProtocolVersion(self) -> str:
        return "Robstride Private Protocol v1.0 (CAN 2.0 Extended Frame)"

    # ================================================================
    # 控制参数设置
    # ================================================================

    def SetPositionPD(self, kp: float, kd: float):
        """设置全局默认 PD 增益（未配置逐关节增益的关节使用此值）"""
        with self._state_lock:
            self._position_kp = clamp(kp, 0.0, 500.0)
            self._position_kd = clamp(kd, 0.0, 5.0)
            _kp, _kd = self._position_kp, self._position_kd
        logger.info("全局 PD 增益已设置: Kp=%.1f, Kd=%.2f", _kp, _kd)

    def GetPositionPD(self) -> Tuple[float, float]:
        """获取全局默认 PD 增益，返回 (kp, kd)"""
        with self._state_lock:
            return self._position_kp, self._position_kd

    def GetJointPD(self, motor_id: int) -> Tuple[float, float]:
        """获取指定关节实际生效的 PD 增益（逐关节覆盖优先，否则用全局默认）"""
        with self._state_lock:
            kp = self._joint_kp.get(motor_id, self._position_kp)
            kd = self._joint_kd.get(motor_id, self._position_kd)
        return kp, kd

    def SetJointPD(self, motor_id: int, kp: float, kd: float):
        """设置单个关节的 PD 增益（覆盖全局默认值）"""
        with self._state_lock:
            self._joint_kp[motor_id] = clamp(kp, 0.0, 500.0)
            self._joint_kd[motor_id] = clamp(kd, 0.0, 5.0)
        logger.info("关节 %d PD 增益已设置: Kp=%.1f, Kd=%.2f",
                     motor_id, self._joint_kp[motor_id], self._joint_kd[motor_id])

    def SetAllJointPD(self, kp_map: Dict[int, float], kd_map: Dict[int, float]):
        """批量设置逐关节 PD 增益"""
        with self._state_lock:
            for mid, kp_val in kp_map.items():
                self._joint_kp[mid] = clamp(kp_val, 0.0, 500.0)
            for mid, kd_val in kd_map.items():
                self._joint_kd[mid] = clamp(kd_val, 0.0, 5.0)
        logger.info("逐关节 PD 增益已设置: Kp=%s, Kd=%s",
                     self._joint_kp, self._joint_kd)

    def ClearJointPD(self):
        """清除所有逐关节 PD 覆盖，恢复使用全局默认值"""
        with self._state_lock:
            self._joint_kp.clear()
            self._joint_kd.clear()
        logger.info("逐关节 PD 增益已清除，使用全局默认值")

    def SetJointLimitEnabled(self, enabled: bool):
        self._joint_limit_enabled = enabled
        logger.info("关节限位检查: %s", "启用" if enabled else "禁用")

    def SetTorqueLimitEnabled(self, enabled: bool):
        """开关关节软力矩限制（触限冻结 + 滞回解锁）"""
        self._torque_limit_enabled = enabled
        if not enabled:
            self._torque_locked = [0] * self.NUM_JOINTS
        logger.info("关节软力矩限制: %s", "启用" if enabled else "禁用")

    def SetTorqueLimits(self, limits: Dict[int, float]):
        """运行时更新关节软力矩限制 {motor_id: 限值 Nm}（正负对称）"""
        for mid, val in limits.items():
            self._torque_limits[mid] = abs(float(val))
        logger.info("关节软力矩限制已更新: %s", self._torque_limits)

    def SetSmoothingAlpha(self, alpha: float):
        """设置 EMA 平滑系数 (0=保持, 1=直通)"""
        self._smoothing_alpha = clamp(alpha, 0.01, 1.0)

    def SetGravityFeedforwardRatio(self, ratio: float):
        """设置重力补偿前馈比例 (0~1)"""
        self._gravity_feedforward_ratio = clamp(ratio, 0.0, 1.0)

    # ================================================================
    # 笛卡尔控制
    # ================================================================

    def EndPoseCtrl(
        self, x: float, y: float, z: float,
        rx: float, ry: float, rz: float,
        duration: float = 2.0,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
        block: bool = True,
        servo: bool = False,
        interp: str = "linear",
    ) -> bool:
        """笛卡尔位姿控制（Pinocchio IK -> MoveJ）

        servo=True：伺服流模式，供 ≥20Hz 高频连续下发（VLA/RTC 流式控制），
        IK 后按 interp 方式把离散关节目标上采样为控制频率平滑设定点：
        - "none"：不插值，直接写关节目标
        - "linear"：关节空间线性插值，段内匀速、段间位置连续
        """
        if not self._connected:
            logger.error("未连接")
            return False

        kin = self._get_kinematics()
        if kin is None:
            return False

        target = ArmEndPose(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)
        current_q = self.GetArmJointMsgs().to_list()
        q_target = kin.inverse_kinematics(target, q_init=current_q)
        if q_target is None:
            logger.error("IK 求解失败，目标位姿可能不可达")
            return False

        if servo and self._control_running:
            return self._servo_stream(q_target, duration, interp)

        return self.MoveJ(q_target, duration=duration, kp=kp, kd=kd, block=block)

    def _servo_stream(self, q_target, duration: float, interp: str = "linear") -> bool:
        """伺服流式微轨迹：把离散关节目标上采样为控制频率（200Hz）平滑设定点。

        每次调用生成一段跨越 duration 的微轨迹并推入控制循环；下一段从
        当前指令设定点接续，段间位置连续，轨迹被高频替换时运动不中断。
        模型只输出离散位置（无速度/加速度），速度前馈由插值解析导数给出。
        """
        if interp == "none":
            return self.JointCtrl(*q_target[:self.NUM_ARM_JOINTS])

        from el_a3_sdk.trajectory import TrajectoryPoint
        with self._cmd_lock:
            q0 = list(self._target_positions[:self.NUM_ARM_JOINTS])
        dt = self._control_period
        dur = max(float(duration), dt)
        n = max(2, int(round(dur / dt)) + 1)
        nj = self.NUM_ARM_JOINTS
        q1 = list(q_target[:nj])
        dq = [q1[i] - q0[i] for i in range(nj)]

        traj = []
        for k in range(n):
            s = k / (n - 1)
            pos = [q0[i] + s * dq[i] for i in range(nj)]
            vel = [dq[i] / dur for i in range(nj)]
            traj.append(TrajectoryPoint(time=k * dt, positions=pos, velocities=vel))

        return self._execute_trajectory_async(traj, block=False)

    def CartesianVelocityCtrl(
        self, vx: float, vy: float, vz: float,
        wx: float, wy: float, wz: float,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
    ) -> bool:
        """笛卡尔速度控制（Jacobian 伪逆 -> 关节增量）"""
        if not self._connected:
            return False

        kin = self._get_kinematics()
        if kin is None:
            return False

        current_q = self.GetArmJointMsgs().to_list()
        J = kin.compute_jacobian(current_q)
        v_des = np.array([vx, vy, vz, wx, wy, wz])
        damping = 1e-4
        JtJ = J.T @ J + damping * np.eye(J.shape[1])
        dq = np.linalg.solve(JtJ, J.T @ v_des)

        dt = self._control_period if self._control_running else 0.02
        target_q = [current_q[i] + dq[i] * dt for i in range(self.NUM_ARM_JOINTS)]
        return self.JointCtrl(*target_q, kp=kp, kd=kd)

    def GetArmEndPoseMsgs(self) -> ArmEndPose:
        """获取末端位姿（Pinocchio FK）"""
        kin = self._get_kinematics()
        if kin is None:
            return ArmEndPose()
        q = self.GetArmJointMsgs().to_list()
        return kin.forward_kinematics(q)

    # ================================================================
    # 轨迹运动（支持控制循环异步执行）
    # ================================================================

    def MoveJ(
        self, positions: List[float], duration: float = 2.0,
        v_max: Optional[float] = None, a_max: Optional[float] = None,
        kp: Optional[float] = None, kd: Optional[float] = None,
        block: bool = True,
    ) -> bool:
        """
        关节空间运动（S-curve 规划）

        如果控制循环正在运行，轨迹推入控制循环队列异步执行。
        否则在调用线程逐点同步发送。

        Args:
            positions: 目标关节角度 (rad), 长度 6
            duration: 期望运动持续时间 (s)
            v_max: 最大关节速度 (rad/s)
            a_max: 最大关节加速度 (rad/s²)
            kp, kd: PD 增益覆盖（仅同步模式）
            block: 是否阻塞等待完成（仅控制循环模式）
        """
        if not self._connected:
            logger.error("未连接")
            return False

        from el_a3_sdk.trajectory import MultiJointPlanner, TrajectoryPoint

        if self._control_running:
            fb = self._read_feedback_positions()
            start_q = list(fb)
            with self._cmd_lock:
                for i in range(self.NUM_ARM_JOINTS):
                    self._target_positions[i] = fb[i]
                    self._target_velocities[i] = 0.0
            self._last_cmd_positions = list(fb)
        else:
            start_q = self.GetArmJointMsgs().to_list()[:self.NUM_ARM_JOINTS]

        vm = v_max or self._max_velocity
        am = a_max or self._max_acceleration

        planner = MultiJointPlanner(
            n_joints=self.NUM_ARM_JOINTS, v_max=vm, a_max=am, j_max=50.0)
        profiles = planner.plan_sync(start_q, positions[:self.NUM_ARM_JOINTS])
        dt = self._control_period if self._control_running else 0.005
        traj = planner.generate_trajectory(profiles, dt=dt)

        if self._control_running:
            n_hold = max(1, int(0.05 / dt))
            hold_dur = n_hold * dt
            hold_pts = [
                TrajectoryPoint(
                    time=k * dt,
                    positions=list(start_q),
                    velocities=[0.0] * self.NUM_ARM_JOINTS,
                    accelerations=[0.0] * self.NUM_ARM_JOINTS,
                )
                for k in range(n_hold)
            ]
            for pt in traj:
                pt.time += hold_dur
            traj = hold_pts + traj
            return self._execute_trajectory_async(traj, block=block)

        # 同步模式
        with self._state_lock:
            self._state = ArmState.RUNNING
        gravity_torques = None
        kin = self._get_kinematics()

        start_time = time.time()
        for pt in traj:
            if kin:
                gravity_torques = kin.compute_gravity(pt.positions)
            self.JointCtrl(*pt.positions, kp=kp, kd=kd, torque_ff=gravity_torques)
            elapsed = time.time() - start_time
            sleep_time = pt.time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        return True

    def MoveL(
        self, target_pose: ArmEndPose, duration: float = 2.0,
        n_waypoints: int = 50,
        kp: Optional[float] = None, kd: Optional[float] = None,
        block: bool = True,
    ) -> bool:
        """
        直线运动（笛卡尔空间线性插值 + IK）

        Args:
            target_pose: 目标末端位姿
            duration: 运动持续时间 (s)
            n_waypoints: 笛卡尔空间插值点数
            block: 是否阻塞等待完成
        """
        if not self._connected:
            logger.error("未连接")
            return False

        kin = self._get_kinematics()
        if kin is None:
            logger.error("Pinocchio 未初始化，MoveL 不可用")
            return False

        current_q = self.GetArmJointMsgs().to_list()
        start_pose = kin.forward_kinematics(current_q)

        from el_a3_sdk.trajectory import TrajectoryPoint

        traj_points = []
        dt = duration / n_waypoints
        q_prev = current_q

        for i in range(1, n_waypoints + 1):
            s = i / n_waypoints
            interp_rx, interp_ry, interp_rz = slerp_euler(
                start_pose.rx, start_pose.ry, start_pose.rz,
                target_pose.rx, target_pose.ry, target_pose.rz, s)
            wp = ArmEndPose(
                x=start_pose.x + s * (target_pose.x - start_pose.x),
                y=start_pose.y + s * (target_pose.y - start_pose.y),
                z=start_pose.z + s * (target_pose.z - start_pose.z),
                rx=interp_rx, ry=interp_ry, rz=interp_rz,
            )
            q_sol = kin.inverse_kinematics(wp, q_init=q_prev)
            if q_sol is None:
                logger.error("MoveL IK 失败 at waypoint %d/%d", i, n_waypoints)
                return False
            traj_points.append(TrajectoryPoint(
                time=i * dt,
                positions=q_sol,
            ))
            q_prev = q_sol

        if self._control_running:
            return self._execute_trajectory_async(traj_points, block=block)

        # 同步模式
        with self._state_lock:
            self._state = ArmState.RUNNING
        start_time = time.time()
        for pt in traj_points:
            gravity_torques = kin.compute_gravity(pt.positions)
            self.JointCtrl(*pt.positions, kp=kp, kd=kd, torque_ff=gravity_torques)
            elapsed = time.time() - start_time
            sleep_time = pt.time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        return True

    def _execute_trajectory_async(self, traj_points: List, block: bool = True) -> bool:
        """将轨迹推入控制循环队列"""
        with self._traj_lock:
            self._trajectory = traj_points
            self._traj_index = 0
            self._traj_start_time = time.perf_counter()
            self._motion_done.clear()
            self._traj_vel_reset = True

        with self._state_lock:
            self._state = ArmState.RUNNING

        if block:
            return self.wait_for_motion()
        return True

    def is_moving(self) -> bool:
        """查询是否有正在执行的轨迹"""
        return not self._motion_done.is_set()

    def wait_for_motion(self, timeout: Optional[float] = None) -> bool:
        """等待当前轨迹执行完成"""
        return self._motion_done.wait(timeout=timeout)

    def cancel_motion(self):
        """取消当前轨迹"""
        with self._traj_lock:
            self._trajectory = None
            self._traj_index = 0
        self._motion_done.set()

    # ================================================================
    # 动力学接口
    # ================================================================

    def ComputeGravityTorques(self, positions: Optional[List[float]] = None) -> List[float]:
        kin = self._get_kinematics()
        if kin is None:
            return [0.0] * self.NUM_ARM_JOINTS
        q = positions or self.GetArmJointMsgs().to_list()
        return kin.compute_gravity(q)

    def GetJacobian(self, positions: Optional[List[float]] = None) -> np.ndarray:
        kin = self._get_kinematics()
        if kin is None:
            return np.zeros((6, self.NUM_ARM_JOINTS))
        q = positions or self.GetArmJointMsgs().to_list()
        return kin.compute_jacobian(q)

    def GetMassMatrix(self, positions: Optional[List[float]] = None) -> np.ndarray:
        kin = self._get_kinematics()
        if kin is None:
            return np.eye(self.NUM_ARM_JOINTS)
        q = positions or self.GetArmJointMsgs().to_list()
        return kin.mass_matrix(q)

    def InverseDynamics(self, q: List[float], v: List[float], a: List[float]) -> List[float]:
        kin = self._get_kinematics()
        if kin is None:
            return [0.0] * self.NUM_ARM_JOINTS
        return kin.inverse_dynamics(q, v, a)

    def ForwardDynamics(self, q: List[float], v: List[float], tau: List[float]) -> List[float]:
        kin = self._get_kinematics()
        if kin is None:
            return [0.0] * self.NUM_ARM_JOINTS
        return kin.forward_dynamics(q, v, tau)

    def GetDynamicsInfo(self, positions: Optional[List[float]] = None) -> DynamicsInfo:
        q = positions or self.GetArmJointMsgs().to_list()
        kin = self._get_kinematics()
        if kin is None:
            return DynamicsInfo()
        return DynamicsInfo(
            gravity_torques=kin.compute_gravity(q),
            mass_matrix=kin.mass_matrix(q),
            jacobian=kin.compute_jacobian(q),
            timestamp=time.time(),
        )

    # ================================================================
    # 多路点轨迹
    # ================================================================

    def MoveWaypoints(
        self, waypoints: List[List[float]], durations: List[float],
        kp: Optional[float] = None, kd: Optional[float] = None,
        block: bool = True,
    ) -> bool:
        """
        多路点轨迹运动（三次样条插值）

        Args:
            waypoints: 路点列表，每个路点为 6 个关节角度 (rad)
            durations: 各段持续时间 (s)，长度 = len(waypoints) - 1
            block: 是否阻塞等待完成
        """
        if not self._connected:
            logger.error("未连接")
            return False
        if len(waypoints) < 2:
            logger.error("至少需要 2 个路点")
            return False
        if len(durations) != len(waypoints) - 1:
            logger.error("durations 长度必须为 len(waypoints) - 1")
            return False

        from el_a3_sdk.trajectory import CubicSplinePlanner

        dt = self._control_period if self._control_running else 0.005
        traj = CubicSplinePlanner.plan_waypoints(waypoints, durations, dt=dt)

        if self._control_running:
            return self._execute_trajectory_async(traj, block=block)

        with self._state_lock:
            self._state = ArmState.RUNNING
        kin = self._get_kinematics()
        start_time = time.time()
        for pt in traj:
            gravity_torques = kin.compute_gravity(pt.positions) if kin else None
            self.JointCtrl(*pt.positions[:self.NUM_ARM_JOINTS],
                           kp=kp, kd=kd, torque_ff=gravity_torques)
            elapsed = time.time() - start_time
            sleep_time = pt.time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        return True

    # ================================================================
    # 参数保存
    # ================================================================

    def SaveParameters(self, motor_num: int = 0xFF) -> bool:
        """将电机参数保存到 Flash（掉电不丢失）"""
        if not self._connected:
            logger.error("未连接")
            return False
        motor_ids = self._resolve_motor_ids(motor_num)
        success = True
        for mid in motor_ids:
            if not self._driver.save_parameters(mid):
                success = False
            time.sleep(0.05)
        if success:
            logger.info("参数已保存到 Flash")
        return success

    # ================================================================
    # 反馈回调
    # ================================================================

    def SetFeedbackCallback(self, callback):
        """
        注册电机反馈回调函数

        Args:
            callback: 回调函数签名 (MotorFeedback) -> None，传 None 清除回调
        """
        self._driver.set_feedback_callback(callback)

    # ================================================================
    # 上下文管理器
    # ================================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.stop_control_loop()
        except Exception:
            pass
        try:
            self.DisableArm()
        except Exception:
            pass
        try:
            self.DisconnectPort()
        except Exception:
            pass
        return False

    # ================================================================
    # 内部辅助
    # ================================================================

    def _get_kinematics(self):
        if self._kin is not None:
            return self._kin
        try:
            from el_a3_sdk.kinematics import ELA3Kinematics
            self._kin = ELA3Kinematics(
                urdf_path=self._urdf_path,
                inertia_config_path=self._inertia_config_path,
                joint_directions=self._joint_directions,
            )
            return self._kin
        except Exception as e:
            logger.debug("Pinocchio 初始化失败 (可选功能): %s", e)
            return None

    def _resolve_motor_ids(self, motor_num: int) -> List[int]:
        if motor_num == 0xFF:
            return list(range(1, self.NUM_JOINTS + 1))
        return [motor_num]

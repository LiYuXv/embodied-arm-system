"""后台工作线程：封装 ELA3Interface，50Hz 数据采集 + 线程安全命令队列"""

import os
import time
import math
import logging
import traceback
import threading
from pathlib import Path
from typing import List, Optional
from queue import Queue, Empty
from collections import deque

from PyQt6.QtCore import QThread, pyqtSignal

from debugger.backend.data_buffer import DataBuffer

logger = logging.getLogger("debugger.worker")

# Sentinel for typing
try:
    from el_a3_sdk import ELA3Interface, ArmJointStates, ArmEndPose, ArmStatus
    from el_a3_sdk.protocol import DEFAULT_JOINT_LIMITS, DEFAULT_MOTOR_TYPE_MAP, ParamIndex
except ImportError:
    ELA3Interface = None


class ArmWorker(QThread):
    """
    后台线程：
    - 50Hz 轮询 SDK 反馈
    - 通过 Signal 将数据发送到 UI 线程
    - 通过命令队列接收 UI 的控制指令
    """

    joints_updated = pyqtSignal(object)
    velocities_updated = pyqtSignal(object)
    efforts_updated = pyqtSignal(object)
    status_updated = pyqtSignal(object)
    end_pose_updated = pyqtSignal(object)
    motor_feedback_updated = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    connected_changed = pyqtSignal(bool)
    enabled_changed = pyqtSignal(bool)
    control_loop_changed = pyqtSignal(bool)
    log_message = pyqtSignal(str)
    can_fps_updated = pyqtSignal(float)
    zero_sta_verified = pyqtSignal(list)
    motor_scan_result = pyqtSignal(list)
    # 使能前检查未通过（携带给用户的中文提示），由 UI 弹窗提示
    enable_rejected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.arm: Optional[object] = None
        self.data_buffer = DataBuffer(max_samples=500, num_channels=7)
        self._running = False
        self._connected = False
        self._enabled = False
        self._cmd_queue: Queue = Queue()
        self._poll_rate_hz = 50.0
        self._sim_mode = False

        self._slow_poll_counter = 0
        self._trajectory_playing = False

        # Simulation state
        self._sim_positions = [0.0] * 7
        self._sim_velocities = [0.0] * 7
        self._sim_torques = [0.0] * 7
        self._sim_target = [0.0] * 7

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_enabled(self):
        return self._enabled

    # 高频流式运动指令：同类只保留最新一条，防止队列无限积压
    _STREAMING_CMDS = {"end_pose_ctrl", "end_pose_ctrl_quat", "joint_ctrl", "gripper_ctrl"}
    # 安全/状态指令：入队前清除积压的流式运动指令，保证立即生效
    _PRIORITY_CMDS = {"disable", "emergency_stop", "cancel_motion", "disconnect", "enable"}

    def submit_command(self, cmd: str, *args, **kwargs):
        with self._cmd_queue.mutex:
            if cmd in self._PRIORITY_CMDS:
                self._cmd_queue.queue = deque(
                    item for item in self._cmd_queue.queue
                    if item[0] not in self._STREAMING_CMDS
                )
            elif cmd in self._STREAMING_CMDS:
                self._cmd_queue.queue = deque(
                    item for item in self._cmd_queue.queue
                    if item[0] != cmd
                )
            self._cmd_queue.queue.append((cmd, args, kwargs))
            self._cmd_queue.unfinished_tasks += 1
            self._cmd_queue.not_empty.notify()

    def run(self):
        self._running = True
        interval = 1.0 / self._poll_rate_hz
        while self._running:
            t0 = time.time()
            try:
                self._process_commands()
                if self._connected:
                    self._poll_feedback()
                # 检查非阻塞轨迹是否完成
                self._check_trajectory_done()
            except Exception as e:
                logger.error(f"Worker error: {e}\n{traceback.format_exc()}")
                self.error_occurred.emit(str(e))
            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        self._running = False
        self._cancel_gripper_move()
        self.wait(3000)

    def _process_commands(self):
        while True:
            try:
                cmd, args, kwargs = self._cmd_queue.get_nowait()
            except Empty:
                break
            try:
                self._execute_command(cmd, args, kwargs)
            except Exception as e:
                logger.error(f"Command '{cmd}' failed: {e}")
                self.error_occurred.emit(f"命令 {cmd} 失败: {e}")

    def _execute_command(self, cmd: str, args, kwargs):
        if cmd == "connect":
            self._do_connect(*args, **kwargs)
        elif cmd == "disconnect":
            self._do_disconnect()
        elif cmd == "enable":
            self._do_enable()
        elif cmd == "disable":
            self._do_disable()
        elif cmd == "emergency_stop":
            self._do_emergency_stop()

        elif cmd == "joint_ctrl":
            self._do_joint_ctrl(*args)
        elif cmd == "move_j":
            self._do_move_j(*args, **kwargs)
        elif cmd == "move_l":
            self._do_move_l(*args, **kwargs)
        elif cmd == "end_pose_ctrl":
            self._do_end_pose_ctrl(*args, **kwargs)
        elif cmd == "end_pose_ctrl_quat":
            self._do_end_pose_ctrl_quat(*args, **kwargs)
        elif cmd == "move_waypoints":
            self._do_move_waypoints(*args, **kwargs)
        elif cmd == "cancel_motion":
            self._do_cancel_motion()
        elif cmd == "zero_torque":
            self._do_zero_torque(*args, **kwargs)
        elif cmd == "zero_torque_gravity":
            self._do_zero_torque_gravity(*args, **kwargs)
        elif cmd == "gripper_ctrl":
            self._do_gripper_ctrl(*args)
        elif cmd == "set_zero_position":
            self._do_set_zero(*args)
        elif cmd == "verify_zero_sta":
            self._do_verify_zero_sta()
        elif cmd == "scan_motors":
            self._do_scan_motors()
        elif cmd == "read_motor_param":
            self._do_read_param(*args)
        elif cmd == "write_motor_param":
            self._do_write_param(*args)
        elif cmd == "start_control_loop":
            self._do_start_control_loop(*args)
        elif cmd == "stop_control_loop":
            self._do_stop_control_loop()
        elif cmd == "playback_trajectory":
            self._do_playback_trajectory(*args)
        elif cmd == "set_smoothing_alpha":
            if self.arm and not self._sim_mode:
                self.arm.SetSmoothingAlpha(args[0])

    def _do_connect(self, can_name="can0", sim_mode=False,
                    backend="socketcan", serial_port=None,
                    serial_baudrate=2000000, can_bitrate=1000000):
        if self._connected:
            return
        self._sim_mode = sim_mode
        if sim_mode:
            self._connected = True
            self.connected_changed.emit(True)
            self.log_message.emit("已连接（模拟模式）")
            return
        if ELA3Interface is None:
            self.error_occurred.emit("el_a3_sdk 未安装")
            return

        if backend != "slcan":
            from debugger.utils.can_utils import get_can_state
            state = get_can_state(can_name)
            if state != "UP":
                self.error_occurred.emit(
                    f"CAN 接口 {can_name} 未开启（当前状态: {state}），请先在工具栏中开启"
                )
                return

        try:
            sdk_root = Path(__file__).parent.parent.parent
            inertia_path = sdk_root / "resources" / "config" / "inertia_params.yaml"
            kwargs = dict(can_name=can_name)
            if inertia_path.exists():
                kwargs["inertia_config_path"] = str(inertia_path)
            kwargs["per_joint_kd_min"] = {4: 0.005, 5: 0.005, 6: 0.005, 7: 0.02}
            kwargs["per_joint_kd_max"] = {4: 0.10, 5: 0.05, 6: 0.05, 7: 0.10}
            if backend == "slcan":
                kwargs["backend"] = "slcan"
                kwargs["serial_port"] = serial_port or can_name
                kwargs["serial_baudrate"] = serial_baudrate
                kwargs["can_bitrate"] = can_bitrate
            self.arm = ELA3Interface(**kwargs)
            self.arm.ConnectPort()
            self._connected = True
            self.connected_changed.emit(True)
            display_name = f"{serial_port or can_name} (SLCAN)" if backend == "slcan" else can_name
            self.log_message.emit(f"已连接到 {display_name}")
            self._request_initial_feedback()
        except Exception as e:
            self.error_occurred.emit(f"连接失败: {e}")

    def _do_disconnect(self):
        if not self._connected:
            return
        if self._sim_mode:
            self._connected = False
            self._enabled = False
            self.connected_changed.emit(False)
            self.enabled_changed.emit(False)
            self.log_message.emit("已断开（模拟模式）")
            return
        try:
            if self.arm:
                if self._enabled:
                    self.arm.DisableArm()
                self.arm.DisconnectPort()
                self.arm = None
            self._connected = False
            self._enabled = False
            self.connected_changed.emit(False)
            self.enabled_changed.emit(False)
            self.log_message.emit("已断开连接")
        except Exception as e:
            self.error_occurred.emit(f"断开失败: {e}")
        finally:
            self._cancel_gripper_move()

    # ---- 使能前安全检查（角度单位：度） ----
    _ENABLE_JOINT_ABS_LIMIT_DEG = 50.0    # J1~J6 必须在 ±50° 以内
    _ENABLE_GRIPPER_MIN_DEG = -100.0      # J7(夹爪) 必须在 -100° ~ +20° 以内
    _ENABLE_GRIPPER_MAX_DEG = 20.0
    _ENABLE_FEEDBACK_TIMEOUT_S = 1.0      # 等待全部电机反馈的最长时间
    _ENABLE_FEEDBACK_FRESH_S = 1.0        # 反馈时间戳新鲜度阈值

    def _do_enable(self):
        if not self._connected:
            return
        if self._sim_mode:
            self._enabled = True
            self.enabled_changed.emit(True)
            self.log_message.emit("电机已使能（模拟模式）")
            return
        try:
            # 1) 先成功读取全部 7 个关节的位置反馈，否则不允许使能
            positions = self._read_joint_positions_for_enable()
            if positions is None:
                msg = ("使能失败：无法读取各关节位置信息。\n"
                       "请检查电源是否打开、串口/CAN 连接是否异常。")
                self.log_message.emit("使能失败：无法读取各关节位置（电源未开或串口/CAN 异常）")
                self.enable_rejected.emit(msg)
                return

            # 2) 位置范围检查：J1~J6 在 ±50° 以内，J7(夹爪) 在 -100° ~ +20° 以内
            degs = [math.degrees(p) for p in positions[:7]]
            out_of_range = []
            for i in range(6):
                if abs(degs[i]) > self._ENABLE_JOINT_ABS_LIMIT_DEG:
                    out_of_range.append(
                        f"J{i+1} = {degs[i]:.1f}°（要求 ±{self._ENABLE_JOINT_ABS_LIMIT_DEG:.0f}°）")
            if not (self._ENABLE_GRIPPER_MIN_DEG <= degs[6] <= self._ENABLE_GRIPPER_MAX_DEG):
                out_of_range.append(
                    f"J7(夹爪) = {degs[6]:.1f}°（要求 "
                    f"{self._ENABLE_GRIPPER_MIN_DEG:.0f}° ~ {self._ENABLE_GRIPPER_MAX_DEG:.0f}°）")
            if out_of_range:
                msg = ("使能失败：关节位置超出允许范围：\n  " + "\n  ".join(out_of_range) +
                       "\n\n请让机械臂回到0位置附近，或重新设置各关节0点位置。")
                self.log_message.emit(
                    "使能失败：关节位置超差 [" + ", ".join(out_of_range) + "]")
                self.enable_rejected.emit(msg)
                return

            # 3) 检查通过，执行使能
            self.arm.EnableArm()
            self._enabled = True
            self.enabled_changed.emit(True)
            self.log_message.emit("电机已使能")
        except Exception as e:
            self.error_occurred.emit(f"使能失败: {e}")

    def _read_joint_positions_for_enable(self) -> Optional[List[float]]:
        """使能前读取 7 个电机的位置反馈。

        主动发送运控帧触发电机回传，等待全部 7 个电机都有有效且新鲜的反馈。
        成功返回 7 关节位置列表 (rad，含夹爪)，失败返回 None。
        """
        if not self.arm:
            return None
        deadline = time.monotonic() + self._ENABLE_FEEDBACK_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                self._trigger_feedback_poll()
            except Exception:
                pass
            time.sleep(0.05)
            try:
                states = self.arm.GetMotorStates()
            except Exception:
                states = None
            if not states:
                continue
            now = time.time()
            all_ok = True
            for mid in range(1, 8):
                fb = states.get(mid)
                if (fb is None or not fb.is_valid
                        or (now - fb.timestamp) > self._ENABLE_FEEDBACK_FRESH_S):
                    all_ok = False
                    break
            if all_ok:
                return self.arm.GetArmJointMsgs().to_list(include_gripper=True)
        return None

    def _do_disable(self):
        if self._sim_mode:
            self._enabled = False
            self.enabled_changed.emit(False)
            self.log_message.emit("电机已失能（模拟模式）")
            return
        try:
            if self.arm:
                self.arm.DisableArm()
            self._enabled = False
            self.enabled_changed.emit(False)
            self.log_message.emit("电机已失能")
        except Exception as e:
            self.error_occurred.emit(f"失能失败: {e}")

    def _do_emergency_stop(self):
        if self._sim_mode:
            self._enabled = False
            self.enabled_changed.emit(False)
            self.log_message.emit("急停已触发（模拟模式）")
            return
        try:
            if self.arm:
                self.arm.EmergencyStop()
            self._enabled = False
            self.enabled_changed.emit(False)
            self.log_message.emit("急停已触发")
        except Exception as e:
            self.error_occurred.emit(f"急停失败: {e}")

    def _ensure_control_loop(self):
        """运动命令前自动启动控制循环（200Hz EMA 平滑 + 速度前馈 + 重力补偿）"""
        if self.arm and not self.arm.control_loop_running:
            self.arm.start_control_loop(rate_hz=200.0)
            self.control_loop_changed.emit(True)
            self.log_message.emit("控制循环已自动启动 (200Hz)")

    def _request_initial_feedback(self):
        """连接成功后获取初始电机状态并同步到UI"""
        if not self.arm:
            return
        try:
            joint_states = self.arm.GetArmJointMsgs()
            if joint_states:
                self.joints_updated.emit(joint_states)
            states = self.arm.GetMotorStates()
            if states:
                motor_fb_list = []
                for mid in range(1, 8):
                    fb = states.get(mid)
                    if fb is not None:
                        motor_fb_list.append(fb)
                if motor_fb_list:
                    self.motor_feedback_updated.emit(motor_fb_list)
            self.log_message.emit("已同步初始关节位置")
        except Exception as e:
            self.log_message.emit(f"同步初始位置失败: {e}")

    def _do_joint_ctrl(self, positions: List[float]):
        if self._sim_mode:
            self._sim_target = list(positions[:7])
            return
        if self.arm and self._enabled:
            self._ensure_control_loop()
            self.arm.JointCtrlList(positions[:7])

    def _do_move_j(self, positions, duration=2.0, block=False):
        if self._sim_mode:
            self._sim_target = list(positions[:7]) + [0.0] * (7 - len(positions))
            self.log_message.emit(f"MoveJ 执行（模拟）duration={duration}s")
            return
        if self.arm and self._enabled:
            self._ensure_control_loop()
            # 确保传入7个关节位置（包含夹爪L7）
            full_positions = list(positions[:7]) + [0.0] * (7 - len(positions))
            self.arm.MoveJ(full_positions[:6], duration=duration, block=block)
            # 单独控制夹爪L7
            if len(full_positions) >= 7:
                self.arm.GripperCtrl(gripper_angle=full_positions[6])
            self.log_message.emit(f"MoveJ 执行中 duration={duration}s")

    def _do_move_l(self, target_pose, duration=2.0, block=False):
        if self._sim_mode:
            self.log_message.emit(f"MoveL 执行（模拟）duration={duration}s")
            return
        if self.arm and self._enabled:
            self._ensure_control_loop()
            self.arm.MoveL(target_pose, duration=duration, block=block)
            self.log_message.emit(f"MoveL 执行中 duration={duration}s")

    def _do_end_pose_ctrl(self, x, y, z, rx, ry, rz, duration=2.0, block=True,
                          servo=False, interp="linear"):
        if self._sim_mode:
            self.log_message.emit(f"EndPoseCtrl（模拟）[{x:.3f},{y:.3f},{z:.3f}]")
            return
        if self.arm and self._enabled:
            self._ensure_control_loop()
            # 强制转 Python float：调用方可能传入 np.float32（如 VLA 推理限幅结果），
            # 而 IK 里 pin.rpy.rpyToMatrix 的 C++ 绑定只接受 double
            self.arm.EndPoseCtrl(float(x), float(y), float(z),
                                 float(rx), float(ry), float(rz),
                                 duration=duration, block=block,
                                 servo=servo, interp=interp)

    def _do_end_pose_ctrl_quat(self, x, y, z, w, qx, qy, qz, duration=2.0, block=True):
        import math, pinocchio as pin
        if self._sim_mode:
            self.log_message.emit(f"EndPoseCtrlQuat（模拟）[{x:.3f},{y:.3f},{z:.3f}] q=[{w:.3f},{qx:.3f},{qy:.3f},{qz:.3f}]")
            return
        if not (self.arm and self._enabled):
            return
        # 归一化四元数
        norm = math.sqrt(w * w + qx * qx + qy * qy + qz * qz)
        if norm < 1e-10:
            self.error_occurred.emit("四元数模长为零，无法转换姿态")
            return
        w, qx, qy, qz = w / norm, qx / norm, qy / norm, qz / norm
        # 将四元数转换为 Pinocchio 的 RPY（与实时显示面板约定一致）
        R = pin.Quaternion(w, qx, qy, qz).matrix()
        rpy = pin.rpy.matrixToRpy(R)
        self._ensure_control_loop()
        # 全部强制转 Python float：调用方可能传入 np.float32（如 VLA 推理输出），
        # IK 里 pin.rpy.rpyToMatrix 的 C++ 绑定只接受 double
        self.arm.EndPoseCtrl(float(x), float(y), float(z),
                             float(rpy[0]), float(rpy[1]), float(rpy[2]),
                             duration=duration, block=block)
        self.log_message.emit(f"EndPoseCtrlQuaternion 执行 [{x:.3f},{y:.3f},{z:.3f}] duration={duration}s")

    def _do_move_waypoints(self, waypoints, durations):
        """批量路点运动：依次串行执行 MoveJ，每个路点独立完整运动"""
        if self._sim_mode:
            self.log_message.emit(f"路点运动（模拟）{len(waypoints)} 个路点")
            return
        if not self.arm or not self._enabled:
            return
        if len(waypoints) < 1:
            return

        self._ensure_control_loop()
        n_points = len(waypoints)

        for i, (positions, duration) in enumerate(zip(waypoints, durations)):
            self.log_message.emit(f"路点 {i+1}/{n_points} 执行中...")
            try:
                # 确保传入7个关节位置（包含夹爪L7）
                full_positions = list(positions[:7]) + [0.0] * (7 - len(positions))
                # block=True 阻塞等待当前 MoveJ 完成后再执行下一个
                self.arm.MoveJ(full_positions[:6], duration=duration, block=True)
                # 单独控制夹爪L7
                if len(full_positions) >= 7:
                    self.arm.GripperCtrl(gripper_angle=full_positions[6])
                self.log_message.emit(f"路点 {i+1}/{n_points} 完成")
            except Exception as e:
                self.log_message.emit(f"路点 {i+1}/{n_points} 失败: {e}")
                self.error_occurred.emit(f"路点 {i+1} 运动失败: {e}")
                return

        self.log_message.emit(f"路点运动全部完成: {n_points} 个路点")

    def _do_cancel_motion(self):
        if self._sim_mode:
            return
        if self.arm:
            self.arm.cancel_motion()
            self.log_message.emit("运动已取消")

    _ZERO_TORQUE_KD = [0.05, 0.05, 0.05, 0.0125, 0.0125, 0.0125, 0.05]

    def _do_zero_torque(self, enable):
        if self._sim_mode:
            state = "开启" if enable else "关闭"
            self.log_message.emit(f"零力矩模式{state}（模拟）")
            return
        if self.arm:
            kd = self._ZERO_TORQUE_KD[0] if enable else 1.0
            self.arm.ZeroTorqueMode(enable, kd=kd)
            state = "开启" if enable else "关闭"
            self.log_message.emit(f"零力矩模式{state}")

    def _do_zero_torque_gravity(self, enable):
        if self._sim_mode:
            state = "开启" if enable else "关闭"
            self.log_message.emit(f"重力补偿零力矩{state}（模拟）")
            return
        if self.arm:
            self.arm.ZeroTorqueModeWithGravity(
                enable, kd=self._ZERO_TORQUE_KD, update_rate=100.0)
            state = "开启" if enable else "关闭"
            self.log_message.emit(f"重力补偿零力矩{state}")

    def _do_gripper_ctrl(self, angle, torque_max=None, torque_min=None):
        if self._sim_mode:
            self._sim_target[6] = angle
            return
        if self.arm and self._enabled:
            if torque_max is None or torque_min is None:
                self.arm.GripperCtrl(angle)
                return
            # 力矩受限移动：后台线程逐步逼近目标角度，
            # 力矩达到上限/下限时停止推进（位置控制保位），防止夹爪继续夹闭/张开。
            self._cancel_gripper_move()
            self._gripper_move_stop = threading.Event()
            self._gripper_move_thread = threading.Thread(
                target=self._gripper_torque_limited_move,
                args=(angle, torque_max, torque_min),
                daemon=True, name="gripper_torque_move")
            self._gripper_move_thread.start()

    # ---- 夹爪力矩受限移动（后台线程） ----

    _GRIPPER_MOVE_STEP_RAD = 0.02   # 单步推进量 (rad)
    _GRIPPER_MOVE_INTERVAL_S = 0.05  # 步进间隔 (s) → 约 0.4 rad/s
    _GRIPPER_MOVE_TIMEOUT_S = 30.0   # 总超时保护

    def _cancel_gripper_move(self):
        if getattr(self, "_gripper_move_stop", None) is not None:
            self._gripper_move_stop.set()
        thread = getattr(self, "_gripper_move_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._gripper_move_thread = None

    def _gripper_torque_limited_move(self, target, torque_max, torque_min):
        """逐步推进夹爪目标角度；闭合力矩≥上限 / 张开口力矩≤下限 时停止并保位。"""
        if self.arm is None:
            return
        stop = getattr(self, "_gripper_move_stop", None) or threading.Event()
        t0 = time.monotonic()
        try:
            while not stop.is_set():
                if time.monotonic() - t0 > self._GRIPPER_MOVE_TIMEOUT_S:
                    self.log_message.emit("夹爪力矩受限移动超时")
                    break
                try:
                    positions = self.arm.GetArmJointMsgs().to_list(include_gripper=True)
                    efforts = self.arm.GetArmJointEfforts().to_list(include_gripper=True)
                except Exception as e:
                    logger.warning("读取夹爪反馈失败: %s", e)
                    break
                if len(positions) < 7 or len(efforts) < 7:
                    break
                cur = float(positions[6])
                torque = float(efforts[6])
                diff = target - cur

                if abs(diff) <= self._GRIPPER_MOVE_STEP_RAD:
                    self.arm.GripperCtrl(target)
                    break

                # 闭合方向（diff>0，角度向0增大）：力矩≥上限 → 保位停止
                if diff > 0.0 and torque >= torque_max:
                    self.arm.GripperCtrl(cur)
                    self.log_message.emit(
                        f"夹爪力矩达上限 {torque_max:.2f} Nm，停止闭合，保位 {math.degrees(cur):.1f}°")
                    break
                # 张开方向（diff<0，角度向全开减小）：力矩≤下限 → 保位停止
                if diff < 0.0 and torque <= torque_min:
                    self.arm.GripperCtrl(cur)
                    self.log_message.emit(
                        f"夹爪力矩达下限 {torque_min:.2f} Nm，停止张开，保位 {math.degrees(cur):.1f}°")
                    break

                next_target = cur + self._GRIPPER_MOVE_STEP_RAD * (1.0 if diff > 0 else -1.0)
                self.arm.GripperCtrl(next_target)
                time.sleep(self._GRIPPER_MOVE_INTERVAL_S)
        except Exception as e:
            logger.error("夹爪力矩受限移动异常: %s", e)
        finally:
            self._gripper_move_thread = None

    def _do_set_zero(self, motor_num=0xFF):
        if self._sim_mode:
            self._sim_positions = [0.0] * 7
            self.log_message.emit("零位已设置（模拟）")
            return
        if self.arm:
            self.arm.SetZeroPosition(motor_num)
            self.log_message.emit(f"电机{motor_num}零位已设置")

    def _do_verify_zero_sta(self):
        from el_a3_sdk.protocol import ParamIndex
        results = []
        if not self.arm:
            return
        self.log_message.emit("开始校验 ZERO_STA 参数...")
        all_ok = True
        for mid in range(1, 8):
            result = self.arm.ReadMotorParameter(mid, ParamIndex.ZERO_STA)
            if result and result.success:
                val = result.value_uint8
                ok = (val == 1)
                results.append((mid, val, True))
                status = "✓" if ok else "✗"
                raw_hex = result.raw_bytes.hex() if result.raw_bytes else ""
                self.log_message.emit(
                    f"  电机{mid} ZERO_STA = {val} {status}  (raw: {raw_hex})")
                if not ok:
                    all_ok = False
            else:
                results.append((mid, 0, False))
                self.log_message.emit(f"  电机{mid} ZERO_STA 读取失败")
                all_ok = False
        self.zero_sta_verified.emit(results)
        if all_ok:
            self.log_message.emit("ZERO_STA 校验完成: 全部通过")
        else:
            self.log_message.emit("ZERO_STA 校验完成: 存在异常")

    def _do_scan_motors(self):
        results = []
        if self._sim_mode:
            results = [(mid, True, "v1.0.0-sim", 24.0) for mid in range(1, 8)]
            self.motor_scan_result.emit(results)
            self.log_message.emit("电机扫描完成（模拟）: 7/7 在线")
            return
        if not self.arm:
            return
        self.log_message.emit("开始扫描电机...")
        online_count = 0
        for mid in range(1, 8):
            fw = self.arm.GetFirmwareVersion(mid)
            voltage = self.arm.GetMotorVoltage(mid)
            online = fw is not None or voltage is not None
            fw_str = fw.version_str if fw else ""
            if online:
                online_count += 1
                v_str = f" {voltage:.1f}V" if voltage is not None else ""
                self.log_message.emit(
                    f"  电机{mid}: 在线  固件={fw_str or '—'}{v_str}")
            else:
                self.log_message.emit(f"  电机{mid}: 离线")
            results.append((mid, online, fw_str, voltage))
        self.motor_scan_result.emit(results)
        self.log_message.emit(f"电机扫描完成: {online_count}/7 在线")

    def _do_read_param(self, motor_id, param_index):
        if self._sim_mode:
            self.log_message.emit(f"读取参数（模拟）motor={motor_id} param=0x{param_index:04X}")
            return
        if self.arm:
            result = self.arm.ReadMotorParameter(motor_id, param_index)
            self.log_message.emit(
                f"电机{motor_id} 参数0x{param_index:04X} = {result}"
            )

    def _do_write_param(self, motor_id, param_index, value):
        if self._sim_mode:
            self.log_message.emit(
                f"写入参数（模拟）motor={motor_id} param=0x{param_index:04X} val={value}"
            )
            return
        if self.arm:
            self.arm.WriteMotorParameter(motor_id, param_index, value)
            self.log_message.emit(
                f"电机{motor_id} 参数0x{param_index:04X} 已写入 {value}"
            )

    def _do_start_control_loop(self, rate_hz=200.0):
        if self._sim_mode:
            self.control_loop_changed.emit(True)
            self.log_message.emit(f"控制循环已启动（模拟）{rate_hz}Hz")
            return
        if self.arm:
            self.arm.start_control_loop(rate_hz=rate_hz)
            self.control_loop_changed.emit(True)
            self.log_message.emit(f"控制循环已启动 {rate_hz}Hz")

    def _do_stop_control_loop(self):
        if self._sim_mode:
            self.control_loop_changed.emit(False)
            self.log_message.emit("控制循环已停止（模拟）")
            return
        if self.arm:
            self.arm.stop_control_loop()
            self.control_loop_changed.emit(False)
            self.log_message.emit("控制循环已停止")

    def _do_playback_trajectory(self, traj_points):
        """示教轨迹回放：推入控制循环队列，200Hz Kp跟踪。

        使用 block=False 防止阻塞 worker 线程，确保反馈轮询继续运行，
        这样摄像头录制和界面显示才能实时更新末端位姿。
        """
        if self._sim_mode:
            self.log_message.emit(f"轨迹回放（模拟）{len(traj_points)} 个点")
            return
        if not self.arm or not self._enabled:
            self.log_message.emit("轨迹回放失败：未连接或未使能")
            return
        if not traj_points:
            return

        self._ensure_control_loop()
        n_points = len(traj_points)
        self.log_message.emit(f"轨迹回放开始：{n_points} 个点，推入控制循环队列")

        try:
            # 非阻塞推入轨迹；worker 主循环继续轮询反馈
            self.arm._execute_trajectory_async(traj_points, block=False)
            self._trajectory_playing = True
        except Exception as e:
            self.log_message.emit(f"轨迹回放失败: {e}")
            self.error_occurred.emit(f"轨迹回放失败: {e}")

    def _check_trajectory_done(self):
        """检查非阻塞轨迹是否执行完成。"""
        if not self._trajectory_playing:
            return
        if self.arm is None:
            self._trajectory_playing = False
            return
        try:
            if not self.arm.is_moving():
                self._trajectory_playing = False
                self.log_message.emit("轨迹回放完成")
        except Exception as e:
            logger.debug(f"检查轨迹状态失败: {e}")
            self._trajectory_playing = False

    def _trigger_feedback_poll(self):
        """发送运控帧触发电机反馈（连接后使能前使用）"""
        driver = self.arm._driver
        for motor_id in range(1, 8):
            driver.send_motion_control(motor_id, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _poll_feedback(self):
        now = time.time()
        self._slow_poll_counter += 1
        do_slow_poll = (self._slow_poll_counter % 10 == 0)  # ~5Hz for heavy queries

        if self._sim_mode:
            self._update_sim_state()
            positions = list(self._sim_positions)
            velocities = list(self._sim_velocities)
            torques = list(self._sim_torques)
            temperatures = [25.0 + i * 0.5 for i in range(7)]
        else:
            try:
                if not self._enabled:
                    self._trigger_feedback_poll()

                joint_msg = self.arm.GetArmJointMsgs()
                vel_msg = self.arm.GetArmJointVelocities()
                eff_msg = self.arm.GetArmJointEfforts()

                positions = joint_msg.to_list()
                velocities = vel_msg.to_list()
                torques = eff_msg.to_list()
                temperatures = [0.0] * 7

                self.joints_updated.emit(joint_msg)
                self.velocities_updated.emit(vel_msg)
                self.efforts_updated.emit(eff_msg)

                # 关键：末端位姿每次轮询都更新（与视频/录制同步）
                try:
                    end_pose = self.arm.GetArmEndPoseMsgs()
                    self.end_pose_updated.emit(end_pose)
                except Exception:
                    pass

                if do_slow_poll:
                    try:
                        status = self.arm.GetArmStatus()
                        self.status_updated.emit(status)
                    except Exception:
                        pass

                    try:
                        fps = self.arm.GetCanFps()
                        self.can_fps_updated.emit(fps)
                    except Exception:
                        pass

                    motor_fb_list = []
                    try:
                        states = self.arm.GetMotorStates()
                        if states:
                            for mid in range(1, 8):
                                fb = states.get(mid)
                                if fb is not None:
                                    motor_fb_list.append(fb)
                                    if hasattr(fb, 'temperature'):
                                        temperatures[mid - 1] = fb.temperature
                    except Exception:
                        pass

                    if motor_fb_list:
                        self.motor_feedback_updated.emit(motor_fb_list)

            except Exception as e:
                logger.debug(f"Poll error: {e}")
                return

        self.data_buffer.append(
            now, positions, velocities, torques, temperatures
        )

        if self._sim_mode:
            from el_a3_sdk.data_types import ArmJointStates
            js = ArmJointStates.from_list(positions, timestamp=now)
            js.hz = self._poll_rate_hz
            self.joints_updated.emit(js)
            vs = ArmJointStates.from_list(velocities, timestamp=now)
            self.velocities_updated.emit(vs)
            es = ArmJointStates.from_list(torques, timestamp=now)
            self.efforts_updated.emit(es)
            self.can_fps_updated.emit(200.0)

    def _update_sim_state(self):
        """简单的一阶位置模拟"""
        alpha = 0.05
        for i in range(7):
            diff = self._sim_target[i] - self._sim_positions[i]
            self._sim_velocities[i] = diff * 2.0
            self._sim_positions[i] += diff * alpha
            self._sim_torques[i] = diff * 10.0
            if abs(diff) < 0.001:
                self._sim_velocities[i] = 0.0
                self._sim_torques[i] = 0.0

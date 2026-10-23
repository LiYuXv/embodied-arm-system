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

SAFE_POSITIONS = [0.0, 0.0, 0.0, math.radians(26.0), 0.0, 0.0]

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
    joints_updated2 = pyqtSignal(object)  # 臂2 关节反馈
    efforts_updated2 = pyqtSignal(object)  # 臂2 力矩反馈
    motor_feedback_updated2 = pyqtSignal(object)  # 臂2 电机反馈
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
    # 阻塞式 MoveJ 完成（供标定流程使用）
    move_j_done = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.arm: Optional[object] = None
        self.arm2: Optional[object] = None  # 第二臂（独立 CAN 接口）
        self.data_buffer = DataBuffer(max_samples=500, num_channels=7)
        self._running = False
        self._connected = False
        self._enabled = False
        self._arm2_connected = False
        self._arm2_enabled = False
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
        # 臂2 模拟状态
        self._sim_positions2 = [0.0] * 7
        self._sim_velocities2 = [0.0] * 7
        self._sim_torques2 = [0.0] * 7
        self._sim_target2 = [0.0] * 7

        # 夹爪力矩受限移动线程（每臂一个）
        self._gripper_move_stops = {}
        self._gripper_move_threads = {}

    @property
    def is_connected(self):
        return self._connected or self._arm2_connected

    @property
    def is_enabled(self):
        return self._enabled or self._arm2_enabled

    @property
    def arm1_connected(self):
        return self._connected

    @property
    def arm2_connected(self):
        return self._arm2_connected

    # 高频流式运动指令：同类只保留最新一条，防止队列无限积压
    _STREAMING_CMDS = {"end_pose_ctrl", "end_pose_ctrl_quat", "joint_ctrl", "gripper_ctrl"}
    # 安全/状态指令：入队前清除积压的流式运动指令，保证立即生效
    _PRIORITY_CMDS = {"disable", "emergency_stop", "cancel_motion", "disconnect", "enable",
                      "move_safe_then_disable"}

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
                if self._connected or self._arm2_connected:
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
        elif cmd == "move_safe_then_disable":
            self._do_move_safe_then_disable()
        elif cmd == "emergency_stop":
            self._do_emergency_stop()

        elif cmd == "joint_ctrl":
            self._do_joint_ctrl(*args)
        elif cmd == "move_j":
            self._do_move_j(*args, **kwargs)
        elif cmd == "move_j_arm_block":
            self._do_move_j_arm_block(*args, **kwargs)
        elif cmd == "move_j_dual":
            self._do_move_j_dual(*args, **kwargs)
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
            self._do_verify_zero_sta(*args)
        elif cmd == "set_all_zero_sta":
            self._do_set_all_zero_sta(*args)
        elif cmd == "scan_motors":
            self._do_scan_motors(*args)
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
        elif cmd == "set_torque_limits":
            self._do_set_torque_limits(*args)

    # 连接后通信验证：主动读电机参数，确认总线真的通（ConnectPort 成功不代表能收发）
    _CONNECT_MAX_RETRIES = 3
    _CONNECT_RETRY_DELAY_S = 0.8

    def _verify_arm_comm(self, arm) -> bool:
        """连接后验证：读电机1运行模式参数，有应答才认为总线可用"""
        try:
            result = arm.ReadMotorParameter(1, ParamIndex.RUN_MODE)
            return result is not None
        except Exception:
            return False

    def _create_arm(self, can_name="can0", backend="socketcan", serial_port=None,
                    serial_baudrate=2000000, can_bitrate=1000000, arm_id=1):
        """创建并连接一个 ELA3Interface 实例（含通信验证+自动重连），失败抛异常"""
        if ELA3Interface is None:
            raise RuntimeError("el_a3_sdk 未安装")
        if backend != "slcan":
            from debugger.utils.can_utils import get_can_state
            state = get_can_state(can_name)
            if state != "UP":
                raise RuntimeError(
                    f"CAN 接口 {can_name} 未开启（当前状态: {state}），请先在工具栏中开启"
                )
        sdk_root = Path(__file__).parent.parent.parent
        # 优先加载该臂独立的标定文件（标定面板按臂保存），缺失时回退到共用文件
        per_arm_path = sdk_root / "resources" / "config" / f"inertia_params_arm{arm_id}.yaml"
        default_path = sdk_root / "resources" / "config" / "inertia_params.yaml"
        kwargs = dict(can_name=can_name)
        if per_arm_path.exists():
            kwargs["inertia_config_path"] = str(per_arm_path)
            self.log_message.emit(f"臂{arm_id} 加载标定参数: {per_arm_path.name}")
        elif default_path.exists():
            kwargs["inertia_config_path"] = str(default_path)
        kwargs["per_joint_kd_min"] = {4: 0.005, 5: 0.005, 6: 0.005, 7: 0.02}
        kwargs["per_joint_kd_max"] = {4: 0.10, 5: 0.05, 6: 0.05, 7: 0.10}
        if backend == "slcan":
            kwargs["backend"] = "slcan"
            kwargs["serial_port"] = serial_port or can_name
            kwargs["serial_baudrate"] = serial_baudrate
            kwargs["can_bitrate"] = can_bitrate

        last_err = None
        for attempt in range(1, self._CONNECT_MAX_RETRIES + 1):
            arm = ELA3Interface(**kwargs)
            try:
                if not arm.ConnectPort():
                    raise RuntimeError("ConnectPort 返回失败")
                if self._verify_arm_comm(arm):
                    if attempt > 1:
                        self.log_message.emit(
                            f"{can_name} 第 {attempt} 次连接成功（总线已恢复）")
                    return arm
                raise RuntimeError("已连接但电机无应答（总线不通）")
            except Exception as e:
                last_err = e
                try:
                    arm.DisconnectPort()
                except Exception:
                    pass
                if attempt < self._CONNECT_MAX_RETRIES:
                    self.log_message.emit(
                        f"{can_name} 连接后通信验证失败（{e}），"
                        f"{self._CONNECT_RETRY_DELAY_S}s 后重连 "
                        f"({attempt}/{self._CONNECT_MAX_RETRIES})"
                    )
                    time.sleep(self._CONNECT_RETRY_DELAY_S)
        raise RuntimeError(
            f"连接失败（已重试 {self._CONNECT_MAX_RETRIES} 次）: {last_err}"
        )

    def _do_connect(self, can_name="can0", sim_mode=False,
                    backend="socketcan", serial_port=None,
                    serial_baudrate=2000000, can_bitrate=1000000,
                    arm2=None):
        """连接双臂：arm2 为第二臂连接参数 dict（None 表示只连臂1）。
        任一臂失败不影响另一臂（保留成功臂）。"""
        if self._connected or self._arm2_connected:
            return
        self._sim_mode = sim_mode
        if sim_mode:
            self._connected = True
            self._arm2_connected = True
            self.connected_changed.emit(True)
            self.log_message.emit("已连接（模拟模式，双臂）")
            return

        # ---- 臂1 ----
        try:
            self.arm = self._create_arm(
                can_name=can_name, backend=backend, serial_port=serial_port,
                serial_baudrate=serial_baudrate, can_bitrate=can_bitrate,
                arm_id=1)
            self._connected = True
            display_name = f"{serial_port or can_name} (SLCAN)" if backend == "slcan" else can_name
            self.log_message.emit(f"臂1 已连接到 {display_name}")
        except Exception as e:
            self.arm = None
            self.error_occurred.emit(f"臂1 连接失败: {e}")

        # ---- 臂2 ----
        if arm2:
            try:
                self.arm2 = self._create_arm(arm_id=2, **arm2)
                self._arm2_connected = True
                a2_backend = arm2.get("backend", "socketcan")
                a2_name = arm2.get("can_name", "can1")
                display2 = (f"{arm2.get('serial_port') or a2_name} (SLCAN)"
                            if a2_backend == "slcan" else a2_name)
                self.log_message.emit(f"臂2 已连接到 {display2}")
            except Exception as e:
                self.arm2 = None
                self.error_occurred.emit(f"臂2 连接失败: {e}")

        if self._connected or self._arm2_connected:
            self.connected_changed.emit(True)
            self._request_initial_feedback()

    def _do_disconnect(self):
        if not (self._connected or self._arm2_connected):
            return
        if self._sim_mode:
            self._connected = False
            self._enabled = False
            self._arm2_connected = False
            self._arm2_enabled = False
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
            if self.arm2:
                if self._arm2_enabled:
                    self.arm2.DisableArm()
                self.arm2.DisconnectPort()
                self.arm2 = None
            self._connected = False
            self._enabled = False
            self._arm2_connected = False
            self._arm2_enabled = False
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

    def _check_positions_in_range(self, positions: List[float]) -> List[str]:
        """位置范围检查：J1~J6 在 ±50° 以内，J7(夹爪) 在 -100° ~ +20° 以内。
        返回超差描述列表（空列表表示通过）。"""
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
        return out_of_range

    def _do_enable(self):
        if not (self._connected or self._arm2_connected):
            return
        if self._sim_mode:
            self._enabled = True
            self._arm2_enabled = True
            self.enabled_changed.emit(True)
            self.log_message.emit("电机已使能（模拟模式，双臂）")
            return

        # 逐臂独立检查 + 使能：失败的臂不使能，成功的臂正常使能（保留成功臂）
        failures = []
        enabled_any = False
        for label, arm in (("臂1", self.arm), ("臂2", self.arm2)):
            if arm is None:
                continue
            try:
                # 1) 先成功读取全部 7 个关节的位置反馈，否则不允许使能
                positions = self._read_joint_positions_for_enable(arm)
                if positions is None:
                    failures.append(
                        f"{label}：无法读取各关节位置信息，请检查电源是否打开、串口/CAN 连接是否异常")
                    self.log_message.emit(f"{label} 使能失败：无法读取各关节位置（电源未开或串口/CAN 异常）")
                    continue

                # 2) 位置范围检查
                out_of_range = self._check_positions_in_range(positions)
                if out_of_range:
                    failures.append(
                        f"{label} 关节位置超出允许范围：\n  " + "\n  ".join(out_of_range))
                    self.log_message.emit(
                        f"{label} 使能失败：关节位置超差 [" + ", ".join(out_of_range) + "]")
                    continue

                # 3) 检查通过，执行使能
                arm.EnableArm()
                if arm is self.arm:
                    self._enabled = True
                else:
                    self._arm2_enabled = True
                enabled_any = True
                self.log_message.emit(f"{label}电机已使能")
            except Exception as e:
                failures.append(f"{label} 使能失败: {e}")

        if enabled_any:
            self.enabled_changed.emit(True)
        if failures:
            msg = ("部分机械臂使能失败：\n" + "\n".join(failures) +
                   "\n\n请让机械臂回到0位置附近，或重新设置各关节0点位置。")
            self.enable_rejected.emit(msg)

    def _read_joint_positions_for_enable(self, arm=None) -> Optional[List[float]]:
        """使能前读取 7 个电机的位置反馈。

        主动发送运控帧触发电机回传，等待全部 7 个电机都有有效且新鲜的反馈。
        成功返回 7 关节位置列表 (rad，含夹爪)，失败返回 None。
        """
        arm = arm or self.arm
        if not arm:
            return None
        deadline = time.monotonic() + self._ENABLE_FEEDBACK_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                self._trigger_feedback_poll(arm)
            except Exception:
                pass
            time.sleep(0.05)
            try:
                states = arm.GetMotorStates()
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
                return arm.GetArmJointMsgs().to_list(include_gripper=True)
        return None

    def _do_disable(self):
        if self._sim_mode:
            self._enabled = False
            self._arm2_enabled = False
            self.enabled_changed.emit(False)
            self.log_message.emit("电机已失能（模拟模式）")
            return
        try:
            if self.arm:
                self.arm.DisableArm()
            if self.arm2:
                self.arm2.DisableArm()
            self._enabled = False
            self._arm2_enabled = False
            self.enabled_changed.emit(False)
            self.log_message.emit("电机已失能")
        except Exception as e:
            self.error_occurred.emit(f"失能失败: {e}")

    def _do_move_safe_then_disable(self):
        """失能前：双臂先 MoveJ 回安全位置（第4关节26°，其余关节0），到位后再失能"""
        duration = 4.0
        if self._sim_mode:
            self._sim_target = list(SAFE_POSITIONS)
            self._sim_target2 = list(SAFE_POSITIONS)
            self.log_message.emit("双臂回安全位置（模拟）")
            self._do_disable()
            return
        moved_arms = []
        for label, arm, enabled in (
                ("臂1", self.arm, self._enabled),
                ("臂2", self.arm2, self._arm2_enabled)):
            if not (arm and enabled):
                continue
            try:
                self._ensure_control_loop(arm)
                arm.MoveJ(list(SAFE_POSITIONS), duration=duration,
                          v_max=0.6, a_max=1.0, block=False)
                moved_arms.append((label, arm))
                self.log_message.emit(f"{label} 回安全位置中 duration={duration}s")
            except Exception as e:
                self.error_occurred.emit(f"{label} 回安全位置失败: {e}")
        # 双臂并行运动（block=False），分别等待各自 MoveJ 结束后立即失能
        for label, arm in moved_arms:
            try:
                if not arm.wait_for_motion(timeout=duration + 2.0):
                    self.log_message.emit(f"{label} 回安全位置等待超时，仍执行失能")
            except Exception as e:
                self.error_occurred.emit(f"{label} 等待运动结束失败: {e}")
        self._do_disable()

    def _do_emergency_stop(self):
        if self._sim_mode:
            self._enabled = False
            self._arm2_enabled = False
            self.enabled_changed.emit(False)
            self.log_message.emit("急停已触发（模拟模式）")
            return
        try:
            if self.arm:
                self.arm.EmergencyStop()
            if self.arm2:
                self.arm2.EmergencyStop()
            self._enabled = False
            self._arm2_enabled = False
            self.enabled_changed.emit(False)
            self.log_message.emit("急停已触发")
        except Exception as e:
            self.error_occurred.emit(f"急停失败: {e}")

    def _ensure_control_loop(self, arm=None):
        """运动命令前自动启动控制循环（200Hz EMA 平滑 + 速度前馈 + 重力补偿）"""
        arm = arm or self.arm
        if arm and not arm.control_loop_running:
            arm.start_control_loop(rate_hz=200.0)
            self.control_loop_changed.emit(True)
            label = "臂2" if arm is self.arm2 else "臂1"
            self.log_message.emit(f"{label}控制循环已自动启动 (200Hz)")

    def _request_initial_feedback(self):
        """连接成功后获取初始电机状态并同步到UI"""
        for arm, joint_sig, fb_sig in (
                (self.arm, self.joints_updated, self.motor_feedback_updated),
                (self.arm2, self.joints_updated2, self.motor_feedback_updated2)):
            if not arm:
                continue
            try:
                joint_states = arm.GetArmJointMsgs()
                if joint_states:
                    joint_sig.emit(joint_states)
                states = arm.GetMotorStates()
                if states:
                    motor_fb_list = []
                    for mid in range(1, 8):
                        fb = states.get(mid)
                        if fb is not None:
                            motor_fb_list.append(fb)
                    if motor_fb_list:
                        fb_sig.emit(motor_fb_list)
            except Exception as e:
                self.log_message.emit(f"同步初始位置失败: {e}")
        self.log_message.emit("已同步初始关节位置")

    def _do_joint_ctrl(self, positions: List[float], positions2: Optional[List[float]] = None):
        """关节控制：positions → 臂1，positions2 → 臂2（None 表示不控臂2）"""
        if self._sim_mode:
            self._sim_target = list(positions[:7])
            if positions2 is not None:
                self._sim_target2 = list(positions2[:7])
            return
        if self.arm and self._enabled:
            self._ensure_control_loop(self.arm)
            self.arm.JointCtrlList(positions[:7])
        if positions2 is not None and self.arm2 and self._arm2_enabled:
            self._ensure_control_loop(self.arm2)
            self.arm2.JointCtrlList(positions2[:7])

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

    def _do_move_j_arm_block(self, arm_sel, positions, duration=2.0):
        """阻塞式单臂 MoveJ，完成后发出 move_j_done 信号（供标定流程使用）。

        arm_sel: 1=臂1, 2=臂2。
        """
        if self._sim_mode:
            if arm_sel == 2:
                self._sim_target2 = list(positions[:7]) + [0.0] * (7 - len(positions))
            else:
                self._sim_target = list(positions[:7]) + [0.0] * (7 - len(positions))
            time.sleep(min(duration, 2.0))
            self.move_j_done.emit()
            return
        arm = self.arm2 if arm_sel == 2 else self.arm
        enabled = self._arm2_enabled if arm_sel == 2 else self._enabled
        if arm and enabled:
            self._ensure_control_loop(arm)
            # 确保传入7个关节位置（包含夹爪L7）
            full_positions = list(positions[:7]) + [0.0] * (7 - len(positions))
            arm.MoveJ(full_positions[:6], duration=duration, block=True)
            # 单独控制夹爪L7
            if len(full_positions) >= 7:
                arm.GripperCtrl(gripper_angle=full_positions[6])
            label = "臂2" if arm_sel == 2 else "臂1"
            self.log_message.emit(f"{label} MoveJ 完成 duration={duration}s")
        self.move_j_done.emit()

    def _do_move_j_dual(self, positions1, positions2, duration=2.0):
        """双臂 MoveJ：两个臂共用同一时长与 PD 增益，但各自独立做运动规划，
        均使用 block=False 使两臂同时开始运动。"""
        if self._sim_mode:
            self._sim_target = list(positions1[:7]) + [0.0] * (7 - len(positions1))
            self._sim_target2 = list(positions2[:7]) + [0.0] * (7 - len(positions2))
            self.log_message.emit(f"双臂 MoveJ 执行（模拟）duration={duration}s")
            return
        for label, arm, enabled, positions in (
                ("臂1", self.arm, self._enabled, positions1),
                ("臂2", self.arm2, self._arm2_enabled, positions2)):
            if not (arm and enabled) or positions is None:
                continue
            self._ensure_control_loop(arm)
            # 确保传入7个关节位置（包含夹爪L7）
            full_positions = list(positions[:7]) + [0.0] * (7 - len(positions))
            arm.MoveJ(full_positions[:6], duration=duration, block=False)
            # 单独控制夹爪L7
            if len(full_positions) >= 7:
                arm.GripperCtrl(gripper_angle=full_positions[6])
            self.log_message.emit(f"{label} MoveJ 执行中 duration={duration}s")

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
            self.log_message.emit(f"零力矩模式{state}（模拟，双臂）")
            return
        kd = self._ZERO_TORQUE_KD[0] if enable else 1.0
        state = "开启" if enable else "关闭"
        for label, arm in (("臂1", self.arm), ("臂2", self.arm2)):
            if arm:
                arm.ZeroTorqueMode(enable, kd=kd)
                self.log_message.emit(f"{label}零力矩模式{state}")

    def _do_zero_torque_gravity(self, enable):
        if self._sim_mode:
            state = "开启" if enable else "关闭"
            self.log_message.emit(f"重力补偿零力矩{state}（模拟，双臂）")
            return
        state = "开启" if enable else "关闭"
        for label, arm in (("臂1", self.arm), ("臂2", self.arm2)):
            if arm:
                arm.ZeroTorqueModeWithGravity(
                    enable, kd=self._ZERO_TORQUE_KD, update_rate=100.0)
                self.log_message.emit(f"{label}重力补偿零力矩{state}")

    def _do_set_torque_limits(self, l13_limit, l46_limit, enabled):
        """设置双臂关节软力矩限制：L1-L3 与 L4-L6 分组限值 (Nm)，enabled 开关"""
        limits = {
            1: l13_limit, 2: l13_limit, 3: l13_limit,
            4: l46_limit, 5: l46_limit, 6: l46_limit,
        }
        for label, arm in (("臂1", self.arm), ("臂2", self.arm2)):
            if arm and not self._sim_mode:
                arm.SetTorqueLimits(limits)
                arm.SetTorqueLimitEnabled(enabled)
                self.log_message.emit(
                    f"{label}软力矩限制: L1-L3=±{l13_limit:.1f} Nm, "
                    f"L4-L6=±{l46_limit:.1f} Nm ({'启用' if enabled else '禁用'})")

    def _do_gripper_ctrl(self, angle, torque_max=None, torque_min=None, angle2=None):
        """夹爪控制：angle → 臂1，angle2 → 臂2（None 表示不控臂2）"""
        if self._sim_mode:
            self._sim_target[6] = angle
            if angle2 is not None:
                self._sim_target2[6] = angle2
            return
        targets = []
        if self.arm and self._enabled:
            targets.append(("臂1", self.arm, angle))
        if angle2 is not None and self.arm2 and self._arm2_enabled:
            targets.append(("臂2", self.arm2, angle2))
        if not targets:
            return
        if torque_max is None or torque_min is None:
            for _, arm, ang in targets:
                arm.GripperCtrl(ang)
            return
        # 力矩受限移动：每臂一个后台线程逐步逼近目标角度，
        # 力矩达到上限/下限时停止推进（位置控制保位），防止夹爪继续夹闭/张开。
        self._cancel_gripper_move()
        for label, arm, ang in targets:
            stop = threading.Event()
            self._gripper_move_stops[label] = stop
            t = threading.Thread(
                target=self._gripper_torque_limited_move,
                args=(arm, label, ang, torque_max, torque_min, stop),
                daemon=True, name=f"gripper_torque_move_{label}")
            self._gripper_move_threads[label] = t
            t.start()

    # ---- 夹爪力矩受限移动（后台线程，每臂一个） ----

    _GRIPPER_MOVE_STEP_RAD = 0.02   # 单步推进量 (rad)
    _GRIPPER_MOVE_INTERVAL_S = 0.05  # 步进间隔 (s) → 约 0.4 rad/s
    _GRIPPER_MOVE_TIMEOUT_S = 30.0   # 总超时保护

    def _cancel_gripper_move(self):
        for stop in self._gripper_move_stops.values():
            stop.set()
        for t in self._gripper_move_threads.values():
            if t is not None and t.is_alive():
                t.join(timeout=1.0)
        self._gripper_move_stops = {}
        self._gripper_move_threads = {}

    def _gripper_torque_limited_move(self, arm, label, target, torque_max,
                                     torque_min, stop):
        """逐步推进夹爪目标角度；闭合力矩≥上限 / 张开口力矩≤下限 时停止并保位。"""
        if arm is None:
            return
        t0 = time.monotonic()
        try:
            while not stop.is_set():
                if time.monotonic() - t0 > self._GRIPPER_MOVE_TIMEOUT_S:
                    self.log_message.emit(f"[{label}] 夹爪力矩受限移动超时")
                    break
                try:
                    positions = arm.GetArmJointMsgs().to_list(include_gripper=True)
                    efforts = arm.GetArmJointEfforts().to_list(include_gripper=True)
                except Exception as e:
                    logger.warning("[%s] 读取夹爪反馈失败: %s", label, e)
                    break
                if len(positions) < 7 or len(efforts) < 7:
                    break
                cur = float(positions[6])
                torque = float(efforts[6])
                diff = target - cur

                if abs(diff) <= self._GRIPPER_MOVE_STEP_RAD:
                    arm.GripperCtrl(target)
                    break

                # 闭合方向（diff>0，角度向0增大）：力矩≥上限 → 保位停止
                if diff > 0.0 and torque >= torque_max:
                    arm.GripperCtrl(cur)
                    self.log_message.emit(
                        f"[{label}] 夹爪力矩达上限 {torque_max:.2f} Nm，停止闭合，保位 {math.degrees(cur):.1f}°")
                    break
                # 张开方向（diff<0，角度向全开减小）：力矩≤下限 → 保位停止
                if diff < 0.0 and torque <= torque_min:
                    arm.GripperCtrl(cur)
                    self.log_message.emit(
                        f"[{label}] 夹爪力矩达下限 {torque_min:.2f} Nm，停止张开，保位 {math.degrees(cur):.1f}°")
                    break

                next_target = cur + self._GRIPPER_MOVE_STEP_RAD * (1.0 if diff > 0 else -1.0)
                arm.GripperCtrl(next_target)
                time.sleep(self._GRIPPER_MOVE_INTERVAL_S)
        except Exception as e:
            logger.error("[%s] 夹爪力矩受限移动异常: %s", label, e)
        finally:
            self._gripper_move_threads.pop(label, None)
            self._gripper_move_stops.pop(label, None)

    def _resolve_arms(self, arm_sel=0):
        """按选择解析目标臂: 0=臂1, 1=臂2, 2=双臂。返回 [(label, arm), ...]"""
        if arm_sel == 1:
            return [("臂2", self.arm2)] if self.arm2 else []
        if arm_sel == 2:
            return [(label, arm) for label, arm in
                    (("臂1", self.arm), ("臂2", self.arm2)) if arm]
        return [("臂1", self.arm)] if self.arm else []

    def _do_set_zero(self, motor_num=0xFF, arm_sel=0):
        if self._sim_mode:
            self._sim_positions = [0.0] * 7
            self._sim_positions2 = [0.0] * 7
            self.log_message.emit("零位已设置（模拟）")
            return
        for label, arm in self._resolve_arms(arm_sel):
            arm.SetZeroPosition(motor_num)
            self.log_message.emit(f"{label}电机{motor_num}零位已设置")

    def _do_verify_zero_sta(self, arm_sel=0):
        from el_a3_sdk.protocol import ParamIndex
        results = []
        arms = self._resolve_arms(arm_sel)
        if not arms:
            return
        label, arm = arms[0]
        self.log_message.emit(f"开始校验{label} ZERO_STA 参数...")
        all_ok = True
        for mid in range(1, 8):
            result = arm.ReadMotorParameter(mid, ParamIndex.ZERO_STA)
            if result and result.success:
                val = result.value_uint8
                ok = (val == 1)
                results.append((mid, val, True))
                status = "✓" if ok else "✗"
                raw_hex = result.raw_bytes.hex() if result.raw_bytes else ""
                self.log_message.emit(
                    f"  {label}电机{mid} ZERO_STA = {val} {status}  (raw: {raw_hex})")
                if not ok:
                    all_ok = False
            else:
                results.append((mid, 0, False))
                self.log_message.emit(f"  {label}电机{mid} ZERO_STA 读取失败")
                all_ok = False
        self.zero_sta_verified.emit(results)
        if all_ok:
            self.log_message.emit(f"{label} ZERO_STA 校验完成: 全部通过")
        else:
            self.log_message.emit(f"{label} ZERO_STA 校验完成: 存在异常")

    def _do_set_all_zero_sta(self, arm_sel=0):
        """一键设置选中臂全部 7 个电机 ZERO_STA=1，成功后保存到 Flash 并重新校验"""
        from el_a3_sdk.protocol import ParamIndex
        if self._sim_mode:
            self.log_message.emit("一键设置 ZERO_STA=1（模拟）")
            return
        arms = self._resolve_arms(arm_sel)
        if not arms:
            self.log_message.emit("未连接，无法设置 ZERO_STA")
            return
        label, arm = arms[0]
        self.log_message.emit(f"开始设置{label}全部电机 ZERO_STA=1 ...")
        fail_count = 0
        for mid in range(1, 8):
            ok = arm.WriteMotorParameterInt(mid, ParamIndex.ZERO_STA, 1)
            if ok:
                self.log_message.emit(f"  {label}电机{mid} ZERO_STA 已设置为 1")
            else:
                self.log_message.emit(f"  {label}电机{mid} ZERO_STA 设置失败")
                fail_count += 1
        if fail_count == 0:
            self.log_message.emit(f"{label}全部电机 ZERO_STA 设置完成，保存参数到 Flash ...")
            time.sleep(0.05)
            arm.SaveParameters(0xFF)
            self.log_message.emit(f"{label}参数已保存")
        else:
            self.log_message.emit(
                f"{label} ZERO_STA 设置完成: {fail_count} 个电机失败，跳过保存")
        self._do_verify_zero_sta(arm_sel)

    def _do_scan_motors(self, arm_sel=0):
        results = []
        if self._sim_mode:
            results = [(mid, True, "v1.0.0-sim", 24.0) for mid in range(1, 8)]
            self.motor_scan_result.emit(results)
            self.log_message.emit("电机扫描完成（模拟）: 7/7 在线")
            return
        arms = self._resolve_arms(arm_sel)
        if not arms:
            return
        label, arm = arms[0]
        self.log_message.emit(f"开始扫描{label}电机...")
        online_count = 0
        for mid in range(1, 8):
            fw = arm.GetFirmwareVersion(mid)
            voltage = arm.GetMotorVoltage(mid)
            online = fw is not None or voltage is not None
            fw_str = fw.version_str if fw else ""
            if online:
                online_count += 1
                v_str = f" {voltage:.1f}V" if voltage is not None else ""
                self.log_message.emit(
                    f"  {label}电机{mid}: 在线  固件={fw_str or '—'}{v_str}")
            else:
                self.log_message.emit(f"  {label}电机{mid}: 离线")
            results.append((mid, online, fw_str, voltage))
        self.motor_scan_result.emit(results)
        self.log_message.emit(f"{label}电机扫描完成: {online_count}/7 在线")

    def _do_read_param(self, arm_sel, motor_id, param_index):
        if self._sim_mode:
            self.log_message.emit(f"读取参数（模拟）motor={motor_id} param=0x{param_index:04X}")
            return
        for label, arm in self._resolve_arms(arm_sel):
            result = arm.ReadMotorParameter(motor_id, param_index)
            self.log_message.emit(
                f"{label}电机{motor_id} 参数0x{param_index:04X} = {result}"
            )

    def _do_write_param(self, arm_sel, motor_id, param_index, value):
        if self._sim_mode:
            self.log_message.emit(
                f"写入参数（模拟）motor={motor_id} param=0x{param_index:04X} val={value}"
            )
            return
        for label, arm in self._resolve_arms(arm_sel):
            arm.WriteMotorParameter(motor_id, param_index, value)
            self.log_message.emit(
                f"{label}电机{motor_id} 参数0x{param_index:04X} 已写入 {value}"
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

    def _do_playback_trajectory(self, traj_points, traj_points2=None):
        """示教轨迹回放（双臂）：各自推入本臂控制循环队列，200Hz Kp跟踪，
        两臂同时开始、独立规划。traj_points2 为 None 时仅回放臂1。

        使用 block=False 防止阻塞 worker 线程，确保反馈轮询继续运行，
        这样摄像头录制和界面显示才能实时更新末端位姿。
        """
        if self._sim_mode:
            n2 = f"，臂2 {len(traj_points2)} 个点" if traj_points2 else ""
            self.log_message.emit(f"轨迹回放（模拟）臂1 {len(traj_points)} 个点{n2}")
            return
        started = []
        for label, arm, enabled, pts in (
                ("臂1", self.arm, self._enabled, traj_points),
                ("臂2", self.arm2, self._arm2_enabled, traj_points2)):
            if not pts:
                continue
            if not arm or not enabled:
                self.log_message.emit(f"{label}轨迹回放跳过：未连接或未使能")
                continue
            self._ensure_control_loop(arm)
            try:
                # 非阻塞推入轨迹；worker 主循环继续轮询反馈
                arm._execute_trajectory_async(pts, block=False)
                started.append(f"{label} {len(pts)} 个点")
            except Exception as e:
                self.log_message.emit(f"{label}轨迹回放失败: {e}")
                self.error_occurred.emit(f"{label}轨迹回放失败: {e}")
        if started:
            self._trajectory_playing = True
            self.log_message.emit("轨迹回放开始：" + "，".join(started))
        else:
            self.log_message.emit("轨迹回放失败：未连接或未使能")

    def _check_trajectory_done(self):
        """检查非阻塞轨迹是否执行完成（双臂全部停止才算完成）。"""
        if not self._trajectory_playing:
            return
        if self.arm is None and self.arm2 is None:
            self._trajectory_playing = False
            return
        try:
            moving = any(
                arm.is_moving() for arm in (self.arm, self.arm2) if arm is not None
            )
            if not moving:
                self._trajectory_playing = False
                self.log_message.emit("轨迹回放完成")
        except Exception as e:
            logger.debug(f"检查轨迹状态失败: {e}")
            self._trajectory_playing = False

    def _trigger_feedback_poll(self, arm=None):
        """发送运控帧触发电机反馈（连接后使能前使用）"""
        arm = arm or self.arm
        driver = arm._driver
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
            # ---- 臂2 反馈轮询（独立 try，不影响臂1） ----
            if self.arm2 is not None:
                try:
                    if not self._arm2_enabled:
                        self._trigger_feedback_poll(self.arm2)
                    joint_msg2 = self.arm2.GetArmJointMsgs()
                    self.joints_updated2.emit(joint_msg2)
                    self.efforts_updated2.emit(self.arm2.GetArmJointEfforts())
                    if do_slow_poll:
                        motor_fb_list2 = []
                        states2 = self.arm2.GetMotorStates()
                        if states2:
                            for mid in range(1, 8):
                                fb = states2.get(mid)
                                if fb is not None:
                                    motor_fb_list2.append(fb)
                        if motor_fb_list2:
                            self.motor_feedback_updated2.emit(motor_fb_list2)
                except Exception as e:
                    logger.debug(f"Arm2 poll error: {e}")
            try:
                if not self._enabled:
                    self._trigger_feedback_poll(self.arm)

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
            js2 = ArmJointStates.from_list(list(self._sim_positions2), timestamp=now)
            js2.hz = self._poll_rate_hz
            self.joints_updated2.emit(js2)
            es2 = ArmJointStates.from_list(list(self._sim_torques2), timestamp=now)
            self.efforts_updated2.emit(es2)
            self.can_fps_updated.emit(200.0)

    def _update_sim_state(self):
        """简单的一阶位置模拟（双臂）"""
        alpha = 0.05
        for i in range(7):
            diff = self._sim_target[i] - self._sim_positions[i]
            self._sim_velocities[i] = diff * 2.0
            self._sim_positions[i] += diff * alpha
            self._sim_torques[i] = diff * 10.0
            if abs(diff) < 0.001:
                self._sim_velocities[i] = 0.0
                self._sim_torques[i] = 0.0
            diff2 = self._sim_target2[i] - self._sim_positions2[i]
            self._sim_velocities2[i] = diff2 * 2.0
            self._sim_positions2[i] += diff2 * alpha
            self._sim_torques2[i] = diff2 * 10.0
            if abs(diff2) < 0.001:
                self._sim_velocities2[i] = 0.0
                self._sim_torques2[i] = 0.0

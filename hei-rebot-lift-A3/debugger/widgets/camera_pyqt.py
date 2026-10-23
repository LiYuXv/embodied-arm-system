"""摄像头监控组件：支持双路USB摄像头预览、拍照、录像，可嵌入主界面。

优化要点：
- 后台采集线程读取帧，主线程只负责显示
- 默认640x480降低带宽，使用MJPEG格式
- 图像缩放使用FastTransformation
- 帧信号降频发射（10Hz），避免主线程过载
"""

import os
import sys
import logging
import time
import json
import shutil
import socket
import signal
import subprocess
import threading
import math
from pathlib import Path
from queue import Queue, Empty, Full
from collections import deque

import numpy as np

logger = logging.getLogger("debugger")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# ===== 在导入 cv2 之前，把 C 层 stdout/stderr 重定向到 nul，屏蔽 OpenCV C++ 垃圾日志 =====
if sys.platform == "win32":
    _saved_stdout_fd = os.dup(1)
    _saved_stderr_fd = os.dup(2)
    _devnull = open("nul", "w")
    os.dup2(_devnull.fileno(), 1)
    os.dup2(_devnull.fileno(), 2)
    sys.stdout = os.fdopen(_saved_stdout_fd, "w", closefd=True)
    sys.stderr = os.fdopen(_saved_stderr_fd, "w", closefd=True)

import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QComboBox, QMessageBox, QGridLayout, QLineEdit, QFileDialog,
    QCheckBox, QDoubleSpinBox, QTextEdit, QApplication
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap


# ============================================================================
# 后台采集线程：每路摄像头一个线程，避免阻塞主线程
# ============================================================================
class CameraCaptureThread(QThread):
    """后台线程：持续读取摄像头帧"""
    frame_ready = pyqtSignal(int, object)  # (camera_idx, frame)
    error_occurred = pyqtSignal(int, str)   # (camera_idx, error_msg)

    def __init__(self, camera_idx, cap, parent=None):
        super().__init__(parent)
        self.camera_idx = camera_idx
        self.cap = cap
        self._running = True
        self._last_frame = None

    def run(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                self._last_frame = frame
                self.frame_ready.emit(self.camera_idx, frame)
            else:
                time.sleep(0.001)  # 1ms避免CPU空转

    def stop(self):
        self._running = False
        self.wait(500)

    def get_last_frame(self):
        return self._last_frame


class CameraDetectThread(QThread):
    """后台线程：检测可用摄像头索引"""
    indices_found = pyqtSignal(list)

    def run(self):
        indices = []
        for idx in range(8):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                if not (ret and frame is not None and frame.size > 0):
                    # 部分摄像头（如 Sonix）默认 YUYV 无法出流，需 MJPG
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    indices.append(idx)
                    logger.info(f"[Camera] 摄像头 [{idx}] 可用")
                else:
                    logger.info(f"[Camera] 摄像头 [{idx}] 无法读取")
            else:
                logger.info(f"[Camera] 摄像头 [{idx}] 无法打开")
            cap.release()
        self.indices_found.emit(indices)


# ============================================================================
# LeRobot 后台录制线程：在独立线程中执行 add_frame，避免阻塞主线程
# ============================================================================
class LeRobotRecordThread(QThread):
    """后台线程：持续从队列取帧并写入 LeRobotDataset。

    设计要点：
    - 主线程只把 (rgb0, rgb1, rgb2, pose14, task) 推入队列（<0.1ms）
    - 后台线程负责调用 dataset.add_frame（即使慢也不影响 UI）
    - 队列满时自动丢弃最旧帧，保证实时性
    - 停止时把队列剩余帧全部写入，再调用 save_episode
    """
    frame_written = pyqtSignal(int)   # 已写入帧数
    status = pyqtSignal(str)          # 状态文本
    finished_ok = pyqtSignal(str)     # 保存完成结果
    error = pyqtSignal(str)           # 错误信息

    def __init__(self, dataset, pose_callback, fps=20, max_queue=60,
                 torque_callback=None, action_lookahead=1, parent=None,
                 extra_state_callback=None, pose2_callback=None,
                 torque2_callback=None):
        super().__init__(parent)
        self.dataset = dataset
        self.pose_callback = pose_callback
        self.torque_callback = torque_callback
        # 臂2 回调：位姿/夹爪力矩（None 时以 0 填充，保持 14 维状态维度一致）
        self.pose2_callback = pose2_callback
        self.torque2_callback = torque2_callback
        # 底盘/升降附加状态回调：() -> array(4) [lift_m, vx, vy, omega]，
        # （升降绝对位置米 + 底盘机体系实测速度 m/s, m/s, rad/s）
        # 设置后 state/action 从 14 维扩展为 18 维
        self.extra_state_callback = extra_state_callback
        self.fps = fps
        self.max_queue = max_queue
        # action 标签前瞻帧数：action[t] = pose[t+lookahead]。
        # 1 = 下一帧位姿（修复问题5的最低要求）；5~10 可增大每步目标增量，
        # 提高信噪比、抑制模型输出摆动（20fps 下 lookahead=5 即领先 0.25s）。
        self.action_lookahead = max(1, int(action_lookahead))
        self._queue = Queue(maxsize=max_queue)
        self._running = True
        self._frame_count = 0
        self._push_count = 0
        self._task_name = ""

    def set_task_name(self, name):
        self._task_name = name

    def push_frame(self, rgb0, rgb1, rgb2=None):
        """主线程调用：快速把帧和位姿推入队列。如果队列满，丢弃最旧帧。

        rgb0=臂1腕部(cam1)，rgb1=头部(cam2)，rgb2=臂2腕部(cam3)。
        rgb2 为 None 时丢帧（数据集固定 3 路摄像头，缺路不录）。
        """
        if rgb2 is None:
            self._push_count += 1
            if self._push_count % 100 == 1:
                logger.warning("[LeRobotThread] 第3路摄像头无帧，丢弃该帧（需3路齐全）")
            return
        pose = self._get_pose_dual()
        torque = self._get_torque()
        item = {"cam1": rgb0, "cam2": rgb1, "cam3": rgb2, "pose": pose,
                "torque": torque, "task": self._task_name}
        try:
            self._queue.put_nowait(item)
        except Full:
            try:
                self._queue.get_nowait()  # 丢弃最旧帧
                self._queue.put_nowait(item)
            except Empty:
                pass

        # 调试日志：每 20 帧输出一次当前推入的位姿，便于排查 state/action 恒定问题
        self._push_count += 1
        if self._push_count % 20 == 0:
            logger.info(
                f"[LeRobotThread] 已推入第 {self._push_count} 帧，"
                f"pose_dim={len(pose)}, "
                f"arm1_xyz=[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}], "
                f"arm2_xyz=[{pose[7]:.3f}, {pose[8]:.3f}, {pose[9]:.3f}], "
                f"gripper_torque=[{float(torque[0]):.4f}, {float(torque[1]):.4f}] Nm"
            )

    def _get_torque(self):
        """获取双臂夹爪当前力矩 (Nm)，2 维 float32 [臂1, 臂2]，失败填 0。

        力矩是观测信号（传感器读数），与 observation.state 同属当前时刻，
        不做一帧延迟（延迟仅用于 action）。
        """
        out = np.zeros(2, dtype=np.float32)
        for i, cb in enumerate((self.torque_callback, self.torque2_callback)):
            if cb is None:
                continue
            try:
                val = cb()
                if val is not None:
                    out[i] = round(float(val), 2)
            except Exception as e:
                logger.warning(f"[LeRobotThread] 获取臂{i + 1}夹爪力矩失败: {e}")
        return out

    def _get_pose_dual(self):
        """获取双臂位姿 [arm1_pose7, arm2_pose7] (+底盘 4 维)，失败对应臂填 0。

        每臂 7 维：[x,y,z,rx,ry,rz,gripper_rad]，欧拉角为弧度，与 MoveL/EndPoseCtrl
        约定一致，推理端可直接用于逆解控制。接入底盘后追加
        [lift_m, vx, vy, omega]（升降绝对位置 + 底盘机体系实测速度），
        与推理观测维度一致。
        """
        pose = np.concatenate([self._base_pose7(self.pose_callback),
                               self._base_pose7(self.pose2_callback)])
        if self.extra_state_callback is not None:
            try:
                extra = np.asarray(self.extra_state_callback(), dtype=np.float32)
            except Exception as e:
                logger.warning(f"[LeRobotThread] 获取底盘状态失败: {e}")
                extra = np.zeros(4, dtype=np.float32)
            pose = np.concatenate([pose, extra])
        return pose

    def _base_pose7(self, callback):
        if callback is None:
            return np.zeros(7, dtype=np.float32)
        try:
            result = callback()
        except Exception as e:
            logger.warning(f"[LeRobotThread] 获取末端位姿失败: {e}")
            return np.zeros(7, dtype=np.float32)

        if isinstance(result, tuple) and len(result) >= 2:
            end_pose, gripper_rad = result[0], result[1]
        else:
            end_pose = result
            gripper_rad = 0.0

        if end_pose is None:
            return np.zeros(7, dtype=np.float32)

        x = getattr(end_pose, "x", 0.0)
        y = getattr(end_pose, "y", 0.0)
        z = getattr(end_pose, "z", 0.0)
        rx = getattr(end_pose, "rx", 0.0)
        ry = getattr(end_pose, "ry", 0.0)
        rz = getattr(end_pose, "rz", 0.0)

        # 夹爪角度统一使用弧度（与位姿 rx/ry/rz、机械臂 GripperCtrl 接口一致）
        if gripper_rad is None:
            gripper_rad = 0.0

        return np.array([x, y, z, rx, ry, rz, float(gripper_rad)], dtype=np.float32)

    def stop(self, flush=False, max_remaining=0):
        """停止入队并触发保存。

        Args:
            flush: 是否把队列中剩余帧继续写入。False 表示丢弃未写入帧，立即保存，
                   这样点击停止后不会继续录制数秒。
            max_remaining: 即使 flush=True，也最多保留这么多帧写入，防止队列积压。
        """
        self._running = False
        if not flush:
            # 丢弃队列中未写入的帧，确保点击停止后不再继续录制
            dropped = 0
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    dropped += 1
                except Empty:
                    break
            if dropped:
                logger.info(f"[LeRobotThread] 停止时丢弃 {dropped} 帧未写入队列")
        else:
            # 限制 flush 帧数，避免队列过大导致停止后等待太久
            while self._queue.qsize() > max_remaining:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break

    def run(self):
        try:
            interval = 1.0 / max(self.fps, 1)
            # 关键：action 必须是“未来目标位姿”，不能等于当前 observation.state。
            # 采用 lookahead 帧延迟写入——缓冲 N+1 帧，收到第 t+N 帧时才写第 t 帧，
            # 令 action[t]=pose7[t+N]。N=1 避免 action==state 恒等退化（问题5）；
            # N=5~10 增大目标增量、提高信噪比（详见 数据问题分析.md 问题 5 与摆动分析）。
            lookahead = self.action_lookahead
            buf = deque()

            def _write_frame(obs_item, action_pose):
                # action 用 np.array(..., copy=True) 独立拷贝，避免与某个 item 的
                # pose 共享内存导致后续被改写；state 同理。
                # torque 是观测信号，取 obs_item（当前帧），与 state 同时刻；
                # 只有 action 用未来第 N 帧位姿，避免 action==state 退化。
                frame_dict = {
                    "observation.images.cam1": obs_item["cam1"],
                    "observation.images.cam2": obs_item["cam2"],
                    "observation.images.cam3": obs_item["cam3"],
                    "observation.state": np.array(obs_item["pose"], dtype=np.float32, copy=True),
                    "action": np.array(action_pose, dtype=np.float32, copy=True),
                    "task": obs_item["task"],
                }
                if "torque" in obs_item:
                    frame_dict["observation.gripper_torque"] = np.array(
                        obs_item["torque"], dtype=np.float32, copy=True)
                self.dataset.add_frame(frame_dict)

            # 恒等映射守卫：统计连续多帧 action≈state 的情况。正常运动时 action（未来
            # 位姿）应与 state（当前位姿）有明显差异；若长时间近乎相同，说明位姿回调
            # 返回了缓存/陈旧值，会重新触发“action==state 退化”（数据问题分析.md 问题 5）。
            consec_near_const = 0
            warned_const = False

            def _after_write(obs_item, action_pose):
                nonlocal consec_near_const, warned_const
                diff = float(np.max(np.abs(
                    np.asarray(action_pose, dtype=np.float64)
                    - np.asarray(obs_item["pose"], dtype=np.float64))))
                if diff < 1e-6:
                    consec_near_const += 1
                    if consec_near_const >= 20 and not warned_const:
                        logger.warning(
                            f"[LeRobotThread] 连续 {consec_near_const} 帧 "
                            f"action≈state(最大差 {diff:.2e})，位姿回调可能返回陈旧值，"
                            f"存在 action==state 退化风险！"
                        )
                        warned_const = True
                else:
                    consec_near_const = 0
                    warned_const = False

            while self._running or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.1)
                except Empty:
                    continue

                t0 = time.perf_counter()
                buf.append(item)
                # 缓冲满 N+1 帧时写出最旧帧：observation=buf[0]，action=未来第 N 帧位姿
                if len(buf) > lookahead:
                    obs_item = buf.popleft()
                    _write_frame(obs_item, buf[lookahead - 1]["pose"])
                    self._frame_count += 1
                    _after_write(obs_item, buf[lookahead - 1]["pose"])
                    if self._frame_count % 20 == 0:
                        logger.info(
                            f"[LeRobotThread] 已写入 {self._frame_count} 帧，"
                            f"add_frame 耗时 {(time.perf_counter() - t0) * 1000:.1f}ms"
                        )
                    self.frame_written.emit(self._frame_count)
                dt = (time.perf_counter() - t0) * 1000
                time.sleep(max(0, interval - dt / 1000.0))

            # 队列耗尽后冲刷缓冲：尾部不足 N 帧的，action 用可及的最远帧位姿
            # （末帧目标=保持最终位姿静止）
            for j, obs_item in enumerate(buf):
                k = min(j + lookahead, len(buf) - 1)
                _write_frame(obs_item, buf[k]["pose"])
                self._frame_count += 1
                _after_write(obs_item, buf[k]["pose"])
                self.frame_written.emit(self._frame_count)
            if buf:
                logger.info(f"[LeRobotThread] 冲刷尾部 {len(buf)} 帧，共 {self._frame_count} 帧")

            # 队列已空，在后台线程中执行 save_episode，避免阻塞主线程
            if self._frame_count > 0:
                t0 = time.perf_counter()
                self.dataset.save_episode(parallel_encoding=True)
                # lerobot 0.6+ 的 episode 元数据为缓冲写入，finalize() 后才落盘
                # （否则下次加载该数据集会因缺少 meta/episodes 而失败）；
                # 0.4.x 的 finalize() 是整数据集收尾语义，不能每次调用，
                # 用 resume 方法是否存在作为 0.6+ 的版本判据
                if hasattr(type(self.dataset), "resume"):
                    self.dataset.finalize()
                dt = (time.perf_counter() - t0) * 1000
                logger.info(f"[LeRobotThread] save_episode 完成，耗时 {dt:.1f}ms")
                self.finished_ok.emit(f"已保存 {self._frame_count} 帧")
            else:
                self.finished_ok.emit("没有录制到帧")

        except Exception as e:
            logger.error(f"[LeRobotThread] 写入或保存失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.error.emit(str(e))
            return

    def save_and_close(self):
        """已废弃：保存逻辑已移到 run() 中自动执行，保留空实现兼容旧调用。"""
        logger.warning("[LeRobotThread] save_and_close 已废弃，无需调用")
        return None


# ============================================================================
# 主界面组件
# ============================================================================
def _test_encoder_usable(vcodec):
    """实际试编码一帧，判断编码器在当前机器上是否真正可用。

    av.codec.Codec 只检查 ffmpeg 编译时是否包含该编码器——无 NVIDIA GPU
    的工控机上 h264_nvenc 也会"注册成功"，必须真正打开编码器编码一帧
    才能判定（NVENC 在首个 encode 时才初始化硬件，此刻才暴露失败）。
    全程在内存 BytesIO 中进行，不落盘。
    """
    try:
        import av
        import io
        from fractions import Fraction
        buf = io.BytesIO()
        container = av.open(buf, "w", format="mp4")
        options = {"g": "30"} if "nvenc" in vcodec else {}
        stream = container.add_stream(vcodec, rate=20, options=options)
        stream.pix_fmt = "yuv420p"
        stream.width = 64
        stream.height = 64
        stream.time_base = Fraction(1, 20)
        frame = av.VideoFrame.from_ndarray(
            np.zeros((64, 64, 3), dtype=np.uint8), format="rgb24")
        frame.pts = 0
        frame.time_base = Fraction(1, 20)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return True
    except Exception as e:
        logger.info(f"[LeRobot] 编码器 {vcodec} 不可用: {e}")
        return False


def _resolve_vcodec_for_recording():
    """选择最优视频编码器：有可用 NVIDIA GPU 时用 NVENC (H.264)，否则回退软件 libx264。

    两者在 info.json 中的 canonical 编码都是 'h264'，因此可互相兼容。
    """
    for codec in ["h264_nvenc", "hevc_nvenc"]:
        if _test_encoder_usable(codec):
            logger.info(f"[LeRobot] 使用硬件编码器: {codec}")
            return codec
    logger.info("[LeRobot] 使用软件编码器: h264 (libx264)")
    return "h264"


def _make_rgb_encoder_config(vcodec):
    """按编码器类型构造 RGBEncoderConfig（lerobot 0.6+ 的编码器参数载体）。

    NVENC 要求 GOP 长度 > B帧数+1，lerobot 默认 g=2 会导致 h264_nvenc
    初始化失败（InitializeEncoder failed: Gop Length should be greater
    than number of B frames + 1），因此 NVENC 显式使用 g=30；
    软件编码（libx264）保持 lerobot 默认参数。
    """
    from lerobot.configs import RGBEncoderConfig
    if "nvenc" in vcodec:
        return RGBEncoderConfig(vcodec=vcodec, g=30)
    return RGBEncoderConfig(vcodec=vcodec)


# ============================================================================
# VLA 推理线程：加载 SmolVLA 模型，按固定频率输出 7 维动作
# ============================================================================
class VLAInferenceThread(QThread):
    """后台加载模型并循环推理，输出 [x,y,z,rx,ry,rz,gripper_rad]

    按 checkpoint config.json 的 type 字段自动识别策略类型：
    - SmolVLA：走外部 policy_preprocessor/postprocessor 做归一化，带 task 文本
      条件，力矩为独立输入键（若模型声明）。
    - ACT（如 ckpt/act_ela3_real）：无外部 pre/post processor（归一化约定由
      训练侧保证）；图像需缩放到训练尺寸 512x512；力矩并入 observation.state
      （pose7 + torque = 8 维，与 train_act_real.py 一致）；无 task 条件；
      select_action 内部按 n_action_steps 管理动作队列。

    支持两种执行模式（均为按需 chunk 推理，推理与执行解耦）：
    - 普通模式（顺序块执行）：主线程收到整段 chunk 后一次性平滑执行完
      （MoveJ/S 曲线插值到末端位姿），执行窗口结束后再 request_chunk() 请求
      下一段；
    - RTC 模式（管道平滑块执行）：当前段未执行完时就提前 request_chunk()，
      新段到达后以 servo 插值从当前设定点接续，段间无停顿。
    """

    # ACT 训练时的图像输入尺寸（见 train_act.log “Resizing images to 512x512”）。
    # checkpoint config.json 记录的是数据集原始分辨率 (720x720) 而非训练输入
    # 尺寸；ACT 没有外部 preprocessor 负责缩放，推理侧必须显式缩到该尺寸，
    # 否则输入分布与训练不一致。
    ACT_IMAGE_SIZE = 512

    chunk_ready = pyqtSignal(object, float)  # (np.ndarray (H,7), 推理耗时秒)
    single_chunk_ready = pyqtSignal(object, object)  # (np.ndarray (H,7), 观测信息 dict)
    status = pyqtSignal(str)
    model_loaded = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_path, task, get_obs, fps=20, rtc_mode=False,
                 get_torque=None, rtc_trained=False, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.task = task
        self._task_lock = threading.Lock()  # 保护 self.task：推理期间可在主线程切换任务
        self.get_obs = get_obs      # callable -> (rgb0, rgb1, state8) 或 None
        self.get_torque = get_torque  # callable -> float (夹爪力矩 Nm) 或 None
        self.fps = fps
        self.rtc_mode = rtc_mode
        # 训练时 RTC（arXiv:2512.05964）界面开关：仅对 rtc_training_max_delay>0
        # 训练的 pi05 checkpoint 生效；加载时自动检测，不支持则退回 d 步丢弃
        self.rtc_trained = rtc_trained
        self._trained_rtc_active = False  # checkpoint 支持且开关打开时为 True
        self._trained_max_delay = 0       # checkpoint 训练时的 rtc_training_max_delay
        self._last_raw_chunk = None  # 上次推理的归一化空间 chunk (H,A) tensor（prefix 条件来源）
        self._last_used_delay = 0    # 上次推理实际使用的 inference_delay（主线程读取）
        self._pending_rtc = None     # 主线程请求时写入的 (已执行步索引 k, 预测延迟 d)
        # 多次采样平均：>1 时同一观测采 N 次 chunk 取均值（每次抽不同噪声），
        # 抑制 flow-matching 采样摆动；界面开关实时修改此值
        self.avg_n = 1
        self._running = True
        self._chunk_request = threading.Event()
        self._single_request = threading.Event()  # “推理一次”请求（两种模式通用）
        self._torque_key = None      # 模型声明力矩输入时为 "observation.gripper_torque"
        self._torque_warned = False
        self._is_act = False         # ACT 策略：无外部 processor，力矩并入 state
        self._act_state_dim = 0      # ACT 的 observation.state 维度（7 或 8=pose7+torque）
        self._last_obs_none_note = 0.0  # “等待有效观测”提示的节流时间戳

    def request_chunk(self):
        """RTC 模式：请求一次 chunk 推理（主线程调用，立即返回）"""
        self._chunk_request.set()

    def request_single_chunk(self):
        """“推理一次”：请求一次完整 chunk 推理并回传观测快照（主线程调用，立即返回）"""
        self._single_request.set()

    def set_task(self, task):
        """推理运行期间切换任务指令（线程安全，主线程调用）"""
        with self._task_lock:
            self.task = task

    def _current_task(self):
        # [RECAP R1c/G5] 优势条件化部署：advantage_prefix 由主界面开关设置
        # （"advantage: positive, "），与训练侧 factory.py patch 的 task 前缀
        # 格式严格一致——训练开了条件化的模型，部署恒命令正优势
        with self._task_lock:
            task = self.task
        prefix = getattr(self, "advantage_prefix", "")
        return f"{prefix}{task}" if prefix else task

    def _torque_tensor(self, device):
        """读取当前夹爪力矩，返回 (1,1) float32 tensor；读不到以 0 填充并告警一次。"""
        import torch
        tq = None
        if self.get_torque is not None:
            try:
                tq = self.get_torque()
            except Exception:
                tq = None
        if tq is None:
            if not self._torque_warned:
                self._torque_warned = True
                self.status.emit("警告: 读取夹爪力矩失败，力矩输入将以 0 填充")
            tq = 0.0
        return torch.tensor([[float(tq)]], dtype=torch.float32, device=device)

    def _read_torque_value(self):
        """读取当前夹爪力矩 (Nm float)；读不到返回 0.0 并告警一次。"""
        tq = None
        if self.get_torque is not None:
            try:
                tq = self.get_torque()
            except Exception:
                tq = None
        if tq is None:
            if not self._torque_warned:
                self._torque_warned = True
                self.status.emit("警告: 读取夹爪力矩失败，力矩输入将以 0 填充")
            tq = 0.0
        return float(tq)

    def _build_batch(self, rgb0, rgb1, state, device):
        """构建模型输入 batch，ACT 与 SmolVLA 的差异集中在此处理。

        - ACT：图像缩放到训练尺寸 (ACT_IMAGE_SIZE)；state 为 pose7+torque 拼接
          （与 train_act_real.py 的 merge_torque 一致）；不带 task 文本键；
          归一化在策略内部，无需外部 pre/post processor。
        - SmolVLA：原样透传（task 文本 + 可选独立力矩键），由外部
          policy_preprocessor / policy_postprocessor 处理。
        """
        import torch
        state = np.asarray(state, dtype=np.float32)
        if self._is_act:
            s = self.ACT_IMAGE_SIZE
            if rgb0.shape[0] != s or rgb0.shape[1] != s:
                rgb0 = cv2.resize(rgb0, (s, s), interpolation=cv2.INTER_LINEAR)
                rgb1 = cv2.resize(rgb1, (s, s), interpolation=cv2.INTER_LINEAR)
            if self._act_state_dim == state.shape[0] + 1:
                # 模型 state 比位姿多 1 维：按训练约定把夹爪力矩拼到末尾
                state = np.concatenate(
                    [state, [self._read_torque_value()]]).astype(np.float32)
            return {
                "observation.images.cam1": torch.from_numpy(
                    np.ascontiguousarray(rgb0)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device),
                "observation.images.cam2": torch.from_numpy(
                    np.ascontiguousarray(rgb1)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device),
                "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
            }
        batch = {
            "observation.images.cam1": torch.from_numpy(
                np.ascontiguousarray(rgb0)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device),
            "observation.images.cam2": torch.from_numpy(
                np.ascontiguousarray(rgb1)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device),
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
            "task": [self._current_task()],
        }
        if self._torque_key is not None:
            batch[self._torque_key] = self._torque_tensor(device)
        return batch

    def _run_single_shot(self, policy, preprocessor, postprocessor, device):
        """“推理一次”：完整 chunk 推理并回传观测快照（图像/位姿/力矩/任务），
        供主线程保存诊断数据与下发执行。不影响 select_action 的动作队列。"""
        import torch
        try:
            obs = self.get_obs()
        except Exception:
            obs = None
        if obs is None:
            self.status.emit("推理一次: 无有效观测（摄像头/位姿未就绪）")
            return
        try:
            rgb0, rgb1, state = obs
            torque = None
            if self.get_torque is not None:
                try:
                    torque = self.get_torque()
                except Exception:
                    torque = None
            batch = self._build_batch(rgb0, rgb1, state, device)
            policy.reset()  # 单次诊断使用干净观测队列（n_obs_steps>1 时尤为关键）
            with torch.no_grad():
                batch_in = preprocessor(batch) if preprocessor is not None else batch
                if hasattr(policy, "predict_action_chunk"):
                    action = policy.predict_action_chunk(batch_in)
                else:
                    action = policy.select_action(batch_in)
                if postprocessor is not None:
                    out = postprocessor({"action": action})
                    action = out["action"] if isinstance(out, dict) else out
            c = action.detach().cpu().numpy().astype(np.float64)
            c = np.reshape(c, (-1, c.shape[-1]))  # (H, 7)
            info = {
                "rgb0": rgb0, "rgb1": rgb1,
                "state": np.asarray(state, dtype=np.float64).copy(),
                "torque": torque, "task": self._current_task(),
                "time": time.time(),
            }
            self.single_chunk_ready.emit(c, info)
        except Exception as e:
            self.status.emit(f"推理一次失败: {e}")

    def stop(self):
        self._running = False
        self._chunk_request.set()  # 唤醒等待中的循环以便退出

    def run(self):
        try:
            import torch
            # 限制 torch CPU 线程数，避免推理占满所有核导致 GUI 卡死
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
            from pathlib import Path as _Path
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class
            from lerobot.processor import DataProcessorPipeline
            try:
                import configuration_smolvla_torque  # noqa: F401 注册 smolvla_torque 策略
            except Exception:
                import traceback
                logger.warning(f"[VLA] smolvla_torque 注册失败:\n{traceback.format_exc()}")

            self.status.emit("正在加载模型...")
            ckpt = _Path(self.model_path).resolve()
            if (ckpt / "pretrained_model").is_dir():
                ckpt = ckpt / "pretrained_model"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                config = PreTrainedConfig.from_pretrained(str(ckpt))
                policy_type = str(getattr(config, "type", "") or "").lower()
            except Exception:
                # 个别 lerobot 版本的 draccus 严格校验 config.json 多余字段，
                # 退回直接解析 json 拿策略类型
                config = None
                with open(ckpt / "config.json") as f:
                    policy_type = str(json.load(f).get("type", "")).lower()
            self._is_act = (policy_type == "act")
            policy = get_policy_class(policy_type).from_pretrained(str(ckpt))
            policy.to(device)
            policy.eval()
            cfg = config if config is not None else policy.config

            # 多次采样平均包装（实例级，对 select_action / predict_action_chunk
            # 均生效——两版 lerobot 二者内部都汇聚到 _get_action_chunk）：
            # 同一观测采 avg_n 次取均值，逼近动作分布中心，实测 chunk 摆动 -30%。
            # ACT 没有 _get_action_chunk 且 eval 时为确定性输出（latent=0），
            # 该包装对 ACT 无意义，跳过。
            if hasattr(policy, "_get_action_chunk"):
                orig_get_chunk = policy._get_action_chunk
                def _averaged_get_chunk(batch, noise=None, **kwargs):
                    n = max(1, int(self.avg_n))
                    if n == 1:
                        return orig_get_chunk(batch, noise, **kwargs)
                    outs = [orig_get_chunk(batch, None, **kwargs) for _ in range(n)]
                    return torch.stack(outs, dim=0).mean(dim=0)
                policy._get_action_chunk = _averaged_get_chunk
            elif int(self.avg_n) > 1:
                self.status.emit("提示: 该模型为确定性输出，多次采样平均不生效")
            policy.reset()  # 清空观测/动作队列（n_obs_steps>1 时避免陈旧历史）

            # 训练时 RTC（arXiv:2512.05964）：checkpoint 带 rtc_training_max_delay>0
            # 且界面开启"训练RTC"时启用 trained 模式——推理时把上一段剩余动作
            # hard clamp 进新 chunk 前缀（零额外去噪开销），时滞由模型吸收，
            # 替代应用层的 d 步丢弃补偿
            self._trained_rtc_active = False
            if self.rtc_trained and not self._is_act and hasattr(policy, "init_rtc_processor"):
                self._trained_max_delay = int(getattr(cfg, "rtc_training_max_delay", 0) or 0)
                if self._trained_max_delay > 0:
                    from lerobot.policies.rtc.configuration_rtc import RTCConfig
                    policy.config.rtc_config = RTCConfig(enabled=True, mode="trained")
                    policy.init_rtc_processor()
                    self._trained_rtc_active = True
                else:
                    self.status.emit(
                        "提示: 该 checkpoint 未按训练时 RTC 训练"
                        "（rtc_training_max_delay=0），训练RTC 不生效，退回 d 步丢弃补偿")

            if self._is_act:
                # ACT：无外部 pre/post processor（checkpoint 中也不含
                # policy_preprocessor.json）；力矩并入 state，不走独立输入键
                preprocessor = None
                postprocessor = None
                self._torque_key = None
                state_ft = (getattr(cfg, "input_features", None) or {}
                            ).get("observation.state")
                try:
                    self._act_state_dim = int(state_ft.shape[0])
                except Exception:
                    self._act_state_dim = 7
            else:
                preprocessor = DataProcessorPipeline.from_pretrained(
                    str(ckpt), config_filename="policy_preprocessor.json")
                postprocessor = DataProcessorPipeline.from_pretrained(
                    str(ckpt), config_filename="policy_postprocessor.json")
                # 力矩输入：仅当模型（checkpoint config）声明了该输入特征时才传入，
                # 与 test_inference_episode.py 判定逻辑一致；旧版 smolvla 行为不变
                torque_key = getattr(cfg, "torque_key", "observation.gripper_torque")
                input_features = getattr(cfg, "input_features", None) or {}
                self._torque_key = torque_key if torque_key in input_features else None
                if self._torque_key and self.get_torque is None:
                    self.status.emit("警告: 模型需要力矩输入，但未设置力矩回调，将以 0 填充")
            self.model_loaded.emit()
            if self._is_act:
                self.status.emit(
                    f"模型加载完成 (ACT, {device})：图像缩放到 "
                    f"{self.ACT_IMAGE_SIZE}x{self.ACT_IMAGE_SIZE}，"
                    f"state {self._act_state_dim} 维（pose7+力矩），推理中...")
            else:
                torque_note = f"，力矩输入: {self._torque_key}" if self._torque_key else ""
                rtc_note = (f"，trained RTC 已启用(max_delay={self._trained_max_delay})"
                            if self._trained_rtc_active else "")
                self.status.emit(f"模型加载完成 ({device}){torque_note}{rtc_note}，推理中...")
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())
            return

        # 普通/RTC 两种模式统一走“按需 chunk 推理”循环：
        # - RTC：主线程边执行边异步请求下一段，新旧 chunk 软融合（连续运动）
        # - 普通：主线程顺序块执行——收到整段 chunk 后一次性平滑执行完
        #   （MoveJ/S 曲线插值到末端位姿），执行窗口结束后再请求下一段
        self._rtc_inference_loop(policy, preprocessor, postprocessor, device)

    def _rtc_inference_loop(self, policy, preprocessor, postprocessor, device):
        """按需推理完整动作 chunk（普通/RTC 模式共用），推理期间不阻塞动作执行。

        每次 request_chunk() 触发一次推理，输出整个 action chunk (H,7) 及实测
        推理延迟。RTC 模式下主线程执行器根据该延迟决定提前多少步触发下一次
        推理，保证动作队列始终不为空（异步执行 + 延迟补偿 d）；普通模式下
        主线程在上一段执行窗口结束后才请求，顺序执行。
        """
        import torch
        while self._running:
            if self._single_request.is_set():
                self._single_request.clear()
                self._run_single_shot(policy, preprocessor, postprocessor, device)
                continue
            if not self._chunk_request.wait(timeout=0.1):
                continue
            self._chunk_request.clear()
            if not self._running:
                break
            try:
                obs = self.get_obs()
            except Exception:
                obs = None
            if obs is None:
                # 同普通模式：观测缺失时给出可见提示（节流到 2s 一次）
                now = time.time()
                if now - self._last_obs_none_note >= 2.0:
                    self._last_obs_none_note = now
                    self.status.emit("等待有效观测（需 2 路摄像头画面与位姿反馈）...")
                continue
            try:
                t0 = time.perf_counter()
                rgb0, rgb1, state = obs
                batch = self._build_batch(rgb0, rgb1, state, device)
                with torch.no_grad():
                    if preprocessor is not None:
                        batch = preprocessor(batch)
                    # 直接取完整动作 chunk（RTC 需要整段轨迹做新旧融合）；
                    # 不支持的策略退化为单步动作
                    rtc_kwargs = {}
                    if self._trained_rtc_active and self._pending_rtc is not None:
                        # trained RTC：取上一段归一化 chunk 自 k 步起的剩余动作
                        # （= 本次推理期间机械臂将执行的动作）作 prefix 条件
                        k, d = self._pending_rtc
                        raw = self._last_raw_chunk
                        if d > 0 and raw is not None and raw.shape[0] > k:
                            leftover = raw[k:, :]
                            if leftover.shape[0] < d:
                                pad = leftover[-1:].expand(d - leftover.shape[0], -1)
                                leftover = torch.cat([leftover, pad], dim=0)
                            rtc_kwargs = {"inference_delay": d,
                                          "prev_chunk_left_over": leftover}
                    self._pending_rtc = None
                    if hasattr(policy, "predict_action_chunk"):
                        action = policy.predict_action_chunk(batch, **rtc_kwargs)
                    else:
                        action = policy.select_action(batch)
                    self._last_used_delay = int(rtc_kwargs.get("inference_delay", 0))
                    if self._trained_rtc_active:
                        # 存归一化空间输出（postprocessor 之前），供下一段作 prefix
                        self._last_raw_chunk = action[0].detach()
                    if postprocessor is not None:
                        out = postprocessor({"action": action})
                        action = out["action"] if isinstance(out, dict) else out
                c = action.detach().cpu().numpy().astype(np.float64)
                c = np.reshape(c, (-1, c.shape[-1]))  # (B,H,D) 或 (B,D) -> (H,D)
                latency = time.perf_counter() - t0
                if self._running:
                    self.chunk_ready.emit(c, latency)
            except Exception as e:
                self.status.emit(f"RTC chunk 推理失败: {e}")


# ============================================================================
# 云端 VLA 推理线程：与 VLAInferenceThread 接口完全对齐，推理在云端 gRPC 服务
# ============================================================================
class CloudInferenceThread(QThread):
    """把观测（2×JPEG + 位姿 + 力矩 + 任务）发往云端 gRPC 服务，接收动作 chunk。

    与 VLAInferenceThread 接口逐点对齐（主线程执行层零改动）：
    - 信号：chunk_ready / single_chunk_ready / status / model_loaded / error
    - 方法：request_chunk / request_single_chunk / set_task / stop
    - 属性：avg_n / _pending_rtc / _last_used_delay / _trained_rtc_active /
      _trained_max_delay / _last_raw_chunk（置 None → 下次请求带 reset）

    安全机制：
    - RPC 超时（看门狗）：单次推理超过 rpc_timeout×avg_n 未返回 → link_lost，
      主线程安全停（cancel_motion + 保位 + 暂停控制），恢复需人工确认；
    - 心跳：每 1s 一次 Ping，连续 2 次失败提示链路劣化；断链后 gRPC 通道
      指数退避自动重连，恢复时发 link_restored（不自动恢复运动）；
    - 过期响应丢弃：request_id 单调递增，不匹配即丢弃；
    - token 错误（UNAUTHENTICATED）属配置错误 → error 信号终止，不误判为断链。
    """

    chunk_ready = pyqtSignal(object, float)      # (np.ndarray (H,7), 端到端往返耗时秒)
    single_chunk_ready = pyqtSignal(object, object)  # (np.ndarray (H,7), 观测信息 dict)
    status = pyqtSignal(str)
    model_loaded = pyqtSignal()
    error = pyqtSignal(str)
    link_lost = pyqtSignal(str)     # 链路中断/超时 → 主线程安全停
    link_restored = pyqtSignal()    # 重连成功（仅提示，不自动恢复运动）
    server_info = pyqtSignal(dict)  # 握手能力信息（policy_type/device/图像尺寸等）

    DEFAULT_ADDR = "127.0.0.1:50051"   # SSH 隧道映射地址
    DEFAULT_TOKEN = "ela3-vla-2026"

    def __init__(self, server_addr, token, task, get_obs, fps=20,
                 get_torque=None, rtc_trained=False, parent=None):
        super().__init__(parent)
        self.server_addr = server_addr or self.DEFAULT_ADDR
        self.token = token or self.DEFAULT_TOKEN
        self.task = task
        self._task_lock = threading.Lock()
        self.get_obs = get_obs      # callable -> (rgb0, rgb1, state7) 或 None
        self.get_torque = get_torque
        self.fps = fps
        self.rtc_trained = rtc_trained
        # ---- 与 VLAInferenceThread 对齐的对外属性 ----
        self.avg_n = 1
        self._trained_rtc_active = False
        self._trained_max_delay = 0
        self._last_used_delay = 0
        self._pending_rtc = None       # 主线程写入的 (已执行步索引 k, 预测延迟 d)
        # ---- 内部状态 ----
        self._running = True
        self._chunk_request = threading.Event()
        self._single_request = threading.Event()
        self._need_reset = True        # 首次请求/重连/暂停恢复后带 reset
        self._request_id = 0
        self.rpc_timeout = 1.5         # 看门狗：单次推理 RPC 超时基准 (s)
        self.jpeg_quality = 90
        self._link_down = False
        self._torque_warned = False
        self._last_obs_none_note = 0.0

    # _last_raw_chunk 兼容属性：归一化 chunk 保存在云端会话内，本地只持有
    # "重置语义"——主线程暂停恢复时将其置 None，这里转换为下次请求带 reset
    @property
    def _last_raw_chunk(self):
        return None

    @_last_raw_chunk.setter
    def _last_raw_chunk(self, value):
        if value is None:
            self._need_reset = True

    def request_chunk(self):
        self._chunk_request.set()

    def request_single_chunk(self):
        self._single_request.set()

    def set_task(self, task):
        with self._task_lock:
            self.task = task

    def _current_task(self):
        # 优势条件化前缀（与本地 VLAInferenceThread 行为一致）：
        # 主界面"advantage条件"开关设置 advantage_prefix，部署恒命令正优势
        with self._task_lock:
            task = self.task
        prefix = getattr(self, "advantage_prefix", "")
        return f"{prefix}{task}" if prefix else task

    def stop(self):
        self._running = False
        self._chunk_request.set()

    def _read_torque_value(self):
        tq = None
        if self.get_torque is not None:
            try:
                tq = self.get_torque()
            except Exception:
                tq = None
        if tq is None:
            if not self._torque_warned:
                self._torque_warned = True
                self.status.emit("警告: 读取夹爪力矩失败，力矩输入将以 0 填充")
            tq = 0.0
        return float(tq)

    def _encode_jpeg(self, rgb):
        """RGB (H,W,3) uint8 -> JPEG 字节（转 BGR 编码，服务端解码后转回 RGB）"""
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return buf.tobytes()

    def run(self):
        try:
            import grpc
            from cloud_inference import vla_inference_pb2 as pb2
            from cloud_inference import vla_inference_pb2_grpc as pb2_grpc
        except Exception:
            import traceback
            self.error.emit(
                "云端推理依赖缺失，请在本地执行: pip install grpcio protobuf\n"
                + traceback.format_exc())
            return

        self._pb2 = pb2
        channel = grpc.insecure_channel(
            self.server_addr,
            options=[
                ("grpc.max_send_message_length", 32 * 1024 * 1024),
                ("grpc.max_receive_message_length", 32 * 1024 * 1024),
                # 通道 keepalive：配合心跳快速感知断链，触发指数退避重连
                ("grpc.keepalive_time_ms", 2000),
                ("grpc.keepalive_timeout_ms", 1000),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.http2.max_pings_without_data", 0),
            ])
        stub = pb2_grpc.VLAInferenceStub(channel)

        # ---- 握手：获取服务端模型能力（trained RTC / 图像尺寸 / 设备）----
        try:
            reply = stub.Ping(pb2.PingRequest(token=self.token), timeout=10)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAUTHENTICATED:
                self.error.emit(f"云端 token 校验失败，请检查访问令牌: {e.details()}")
            else:
                self.error.emit(
                    f"无法连接云端推理服务 {self.server_addr}: "
                    f"{e.code().name} {e.details()}\n"
                    f"请确认 SSH 隧道已建立：\n"
                    f"  autossh -M 0 -N -L 50051:localhost:50051 "
                    f"-p <ssh端口> root@<云端地址>")
            return
        self._trained_max_delay = int(reply.trained_max_delay)
        self._trained_rtc_active = bool(self.rtc_trained and self._trained_max_delay > 0)
        self.server_info.emit({
            "policy_type": reply.policy_type, "model_path": reply.model_path,
            "device": reply.device, "trained_max_delay": reply.trained_max_delay,
            "image_width": reply.image_width, "image_height": reply.image_height,
        })
        rtc_note = (f"，trained RTC 可用(max_delay={self._trained_max_delay})"
                    if self._trained_rtc_active else "")
        self.status.emit(
            f"云端模型就绪 ({reply.policy_type} @ {reply.device}){rtc_note}，推理中...")
        self.model_loaded.emit()

        next_hb = 0.0
        hb_fail = 0
        while self._running:
            # ---- 心跳：1s 一次，连续 2 次失败提示链路劣化 ----
            now = time.monotonic()
            if now >= next_hb:
                next_hb = now + 1.0
                try:
                    stub.Ping(pb2.PingRequest(token=self.token), timeout=2)
                    hb_fail = 0
                    if self._link_down:
                        self._link_down = False
                        self._need_reset = True  # 重连=新会话语义：清云端 prefix
                        self.link_restored.emit()
                except grpc.RpcError:
                    hb_fail += 1
                    if hb_fail == 2 and not self._link_down:
                        self.status.emit("警告: 云端链路心跳失败，链路劣化")

            if self._single_request.is_set():
                self._single_request.clear()
                self._do_infer(stub, single=True)
                continue
            if not self._chunk_request.wait(timeout=0.1):
                continue
            self._chunk_request.clear()
            if not self._running:
                break
            self._do_infer(stub, single=False)

        try:
            channel.close()
        except Exception:
            pass

    def _do_infer(self, stub, single):
        """执行一次云端 chunk 推理（普通/单次诊断共用）。"""
        import grpc
        pb2 = self._pb2
        try:
            obs = self.get_obs()
        except Exception:
            obs = None
        if obs is None:
            now = time.time()
            if now - self._last_obs_none_note >= 2.0:
                self._last_obs_none_note = now
                self.status.emit("等待有效观测（需 2 路摄像头画面与位姿反馈）...")
            return
        rgb0, rgb1, state = obs
        torque = self._read_torque_value()
        try:
            t_enc = time.perf_counter()
            j1 = self._encode_jpeg(rgb0)
            j2 = self._encode_jpeg(rgb1)
            enc_ms = (time.perf_counter() - t_enc) * 1000
        except Exception as e:
            self.status.emit(f"图像编码失败: {e}")
            return

        self._request_id += 1
        rid = self._request_id
        # trained RTC：取主线程推算的 (k, d)，d 钳制到训练上限
        k, d = 0, 0
        if self._trained_rtc_active and self._pending_rtc is not None and not single:
            k, d = self._pending_rtc
            d = max(0, min(int(d), self._trained_max_delay))
        self._pending_rtc = None

        req = pb2.InferenceRequest(
            request_id=rid, token=self.token,
            cam1_jpeg=j1, cam2_jpeg=j2,
            state=[float(v) for v in np.asarray(state, dtype=np.float32)],
            torque=torque, task=self._current_task(),
            rtc_k=int(k), rtc_d=int(d), avg_n=int(self.avg_n),
            reset=self._need_reset or single, is_single=single)
        self._need_reset = False

        # 看门狗：avg_n>1 时推理耗时成倍，超时按比例放宽；单次诊断固定 30s
        timeout = 30.0 if single else self.rpc_timeout * max(1, int(self.avg_n))
        t0 = time.perf_counter()
        try:
            reply = stub.InferChunk(req, timeout=timeout)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAUTHENTICATED:
                self.error.emit(f"云端 token 校验失败: {e.details()}")
                self._running = False
                return
            self._need_reset = True
            if not self._link_down:
                self._link_down = True
                self.link_lost.emit(f"{e.code().name}: {e.details() or '推理超时/断链'}")
            return
        dt = time.perf_counter() - t0

        if reply.error:
            self.status.emit(f"云端推理失败: {reply.error}")
            return
        # 过期响应丢弃：request_id 必须是最新发出的
        if reply.request_id != self._request_id:
            return
        try:
            chunk = np.asarray(reply.actions, dtype=np.float64).reshape(
                int(reply.num_steps), int(reply.action_dim))
        except Exception:
            self.status.emit("云端响应解析失败（actions 形状异常）")
            return
        if not np.isfinite(chunk).all():
            self.status.emit("云端响应含 NaN/Inf，已丢弃本次 chunk")
            return
        self._last_used_delay = int(reply.used_delay)
        if not self._running:
            return
        if single:
            info = {
                "rgb0": rgb0, "rgb1": rgb1,
                "state": np.asarray(state, dtype=np.float64).copy(),
                "torque": torque, "task": self._current_task(),
                "time": time.time(),
            }
            self.single_chunk_ready.emit(chunk, info)
        else:
            self.chunk_ready.emit(chunk, dt)


# ============================================================================
# 实时价值判断：recap_value.py 训练的 201-bin 价值头的部署侧封装
# ============================================================================
class ValueJudge:
    """推理中实时评估"照这样下去还有多久完成"（RECAP §4.7 部署侧）。

    - V(s)∈[-1,0] = 预期剩余代价（→0 接近成功完成）；
      预计剩余时间 ≈ −V × denominator / fps 秒；
    - 回顾式 TD 优势（实时因果可用，无需未来帧）：
      A = −elapsed×fps/denominator + (V_t − V_ref)，
      即"过去窗口实际步数代价 + V 变化"；A>0 正在拉近完成（正向动作），
      A<0 在耽误/恶化（负分动作）——与训练侧 N 步前瞻 TD 同式的时间镜像。

    特征与训练侧严格对齐（recap_critic 同款）：
    - state 后端：直接用 7 维位姿；
    - siglip 后端：pi0.5 冻结视觉塔双相机嵌入(均值) ⊕ state
      （224x224 INTER_AREA / BGR→RGB / [-1,1] / patch 均值池化）。
    """

    def __init__(self, model_dir, pi05_path="./pretrained_models/pi05_base"):
        self.model_dir = Path(model_dir)
        self.pi05_path = pi05_path
        self.backend = "state"
        self.denominator = 1.0
        self.fps = 20
        self._p = {}          # 价值头权重
        self._atoms = None    # 201 bin 中心
        self._tower = None    # siglip 视觉塔（仅 siglip 后端）
        self._device = "cpu"

    def load(self):
        """阻塞加载（放后台线程调用）。价值头 + （可选）SigLIP 视觉塔。"""
        import torch
        ckpt_path = self.model_dir / "value_head.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"缺少 {ckpt_path}——先运行 recap_value.py train")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        self.backend = cfg.get("backend", "state")
        self.denominator = float(cfg.get("denominator", 1.0))
        self.fps = int(cfg.get("fps", 20))
        num_bins = int(cfg.get("num_bins", 201))
        self._atoms = np.linspace(float(cfg.get("v_min", -1.0)),
                                  float(cfg.get("v_max", 0.0)), num_bins,
                                  dtype=np.float32)
        self._p = {n: ck[n].detach().float() for n in
                   ["ln_g", "ln_b", "W1", "b1", "W2", "b2", "W3", "b3"]}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        for n in self._p:
            self._p[n] = self._p[n].to(self._device)

        if self.backend == "siglip":
            self._load_siglip_tower()
        logger.info(
            f"[ValueJudge] 加载完成: backend={self.backend}, "
            f"denominator={self.denominator:.1f}, fps={self.fps}, "
            f"dim_in={self._p['W1'].shape[0]}, device={self._device}")

    def _load_siglip_tower(self):
        """pi05 整机 CPU 载入 → 仅取视觉塔 → del 整机 → 塔上 GPU
        （与 recap_critic.extract_features_siglip 同流程，防 16GB 卡 OOM）。"""
        import torch
        import gc
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.configs.policies import PreTrainedConfig
        config = PreTrainedConfig.from_pretrained(self.pi05_path, local_files_only=True)
        config.device = "cpu"
        policy = PI05Policy.from_pretrained(self.pi05_path, config=config)
        tower = policy.model.paligemma_with_expert.paligemma.model.vision_tower
        del policy
        gc.collect()
        if self._device == "cuda":
            torch.cuda.empty_cache()
        self._tower = tower.to(self._device).eval()
        for p in self._tower.parameters():
            p.requires_grad_(False)

    def _value_from_feats(self, feats):
        import torch
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).to(self._device)
        with torch.no_grad():
            h = torch.nn.functional.layer_norm(
                x, (x.shape[-1],), self._p["ln_g"], self._p["ln_b"])
            h = torch.relu(h @ self._p["W1"] + self._p["b1"])
            h = torch.relu(h @ self._p["W2"] + self._p["b2"])
            logits = h @ self._p["W3"] + self._p["b3"]
            probs = torch.softmax(logits, dim=-1)
            a = torch.from_numpy(self._atoms).to(self._device)
            v = (probs * a).sum(dim=-1)
        return float(v[0].item())

    def judge(self, frame0_bgr, frame1_bgr, state7):
        """单帧判断：返回 V(s)∈[-1,0]。输入与录制/推理相同的 BGR 帧 + 7 维位姿。"""
        state7 = np.asarray(state7, dtype=np.float32)
        if self.backend == "siglip" and self._tower is not None:
            import torch
            embs = []
            for frame in (frame0_bgr, frame1_bgr):
                if frame is None:
                    continue
                x = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
                x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    out = self._tower(x)
                e = getattr(out, "pooler_output", None)
                if e is None:
                    e = out.last_hidden_state.mean(dim=1)
                embs.append(e.float().cpu().numpy()[0])
            if embs:
                emb = np.mean(np.stack(embs), axis=0).astype(np.float32)
                feats = np.concatenate([emb, state7])[None, :]
            else:
                feats = state7[None, -self._p["W1"].shape[0]:]
        else:
            feats = state7[None, :]
        return self._value_from_feats(feats)

    def release(self):
        """释放视觉塔显存（cleanup 调用）。"""
        if self._tower is not None:
            del self._tower
            self._tower = None
            try:
                import gc
                import torch
                gc.collect()
                if self._device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass


class ValueJudgeLoadThread(QThread):
    """后台加载 ValueJudge（siglip 后端需 CPU 载入 pi05 整机取塔，~10-20s，不卡 UI）"""
    loaded = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, judge, parent=None):
        super().__init__(parent)
        self.judge = judge

    def run(self):
        try:
            self.judge.load()
            self.loaded.emit()
        except Exception:
            import traceback
            logger.error(f"[ValueJudge] 加载失败:\n{traceback.format_exc()}")
            self.failed.emit(traceback.format_exc().strip().splitlines()[-1])


class CameraWidget(QWidget):
    """可嵌入的摄像头监控组件（双路摄像头）—— 优化版，支持 LeRobot v0.3 数据采集"""

    # 摄像头角色（录制/推理语义）：cam1=臂1腕部, cam2=头部, cam3=臂2腕部。
    # 角色与物理 USB 端口（sysfs 拓扑路径，重启不变）绑定，持久化到
    # settings/main_system_config.json（camera_roles 字段）；/dev/videoN 索引号
    # 重启后会漂移，仅作显示。
    CAMERA_ROLES = ["arm1", "head", "arm2"]
    CAMERA_ROLE_NAMES = {"arm1": "臂1", "head": "头部", "arm2": "臂2"}
    # 界面显示顺序（左→右，对应物理布局）：臂2腕部 / 头部 / 臂1腕部
    CAMERA_DISPLAY_ROLES = ["arm2", "head", "arm1"]
    # 旧版摄像头配置文件路径（仅用于首次迁移，之后统一走 main_system_config.json）
    _LEGACY_CAMERA_SETTINGS_FILES = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "camera_settings.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "settings", "camera_settings.json"),
    ]

    # 信号：降频到10Hz发射，用于后续lerobot数据集收集
    frame_captured = pyqtSignal(int, object)  # (camera_idx, frame_ndarray)
    # 日志文本（供主窗口显示到日志文本框）
    camera_log = pyqtSignal(str)

    def __init__(self, parent=None, save_dir=None, dataset_root=None):
        super().__init__(parent)

        # 分辨率选项 (名称, 宽度, 高度)：只保留 LeRobot 录制所需的采集分辨率
        # （720x720 中心裁切同样来自 1280x720 采集）
        self.resolutions = [
            ("640 x 480 (双摄推荐)", 640, 480),
            ("848 x 480", 848, 480),
            ("1280 x 720 (单摄)", 1280, 720),
        ]
        # 默认采集分辨率：两路摄像头共用 USB 带宽，1280x720 双摄会超出
        # USB 2.0 带宽导致第二路读不到帧，故默认 640x480（所有摄像头都支持，
        # 848x480 部分摄像头不支持，Sonix 设置后会卡死）。
        self.capture_res = (640, 480)

        # 保存目录
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"[Camera] 保存目录: {self.save_dir}")

        # LeRobot 数据集根目录
        if dataset_root is None:
            dataset_root = os.path.join(self.save_dir, "lerobot_dataset")
        self.dataset_root = dataset_root

        self.latest_frames = {}
        self.caps = {}
        self.capture_threads = {}  # 后台采集线程
        self.recorders = {}
        self.is_recording = False
        self.camera_indices = []
        # 角色 -> 摄像头索引（{"arm1": idx, "head": idx, "arm2": idx}），
        # 检测后按持久化的 USB 端口绑定分配，也可用画面下方下拉框手动调整
        self._role_to_idx = {}

        # 对焦配置：默认关闭自动对焦，固定焦距 0.5（归一化值，0.0~1.0）
        self._autofocus_enabled = False
        self._focus_value = 0.5

        # 画面显示配置：默认不显示；摄像头照常采集/录制/推理，仅不刷新界面画面，
        # 勾选后才将帧绘制到界面标签（降低 CPU 占用）
        self._show_video = False

        # 曝光锁定配置：默认不勾选；勾选瞬间读取摄像头当前曝光值并全程锁定，
        # 避免录制过程中自动曝光漂移导致画面明暗变化，影响数据集一致性
        self._exposure_locked = False
        self._locked_exposure = {}  # {camera_idx: 锁定的曝光值}

        # 显示降频控制：限制UI刷新到15fps
        self._display_timer = None
        self._pending_frames = {}
        self._last_emit_time = {}

        # LeRobot 采集状态
        self._lerobot_recording = False
        self._lerobot_dataset = None
        self._lerobot_frame_count = 0
        self._lerobot_get_end_pose = None  # callback() -> (ArmEndPose-like, gripper_rad) 臂1
        self._lerobot_get_torque = None    # callback() -> float (臂1夹爪力矩 Nm) 或 None
        self._lerobot_get_end_pose2 = None  # callback() -> (ArmEndPose-like, gripper_rad) 臂2
        self._lerobot_get_torque2 = None    # callback() -> float (臂2夹爪力矩 Nm) 或 None
        self._lerobot_task_name = ""
        self._lerobot_repo_name = "ela3_demo"
        self._lerobot_fps = 20
        self._lerobot_image_size = (720, 720)
        # 录制前中心正方形裁切边长（None=按目标长宽比自适应裁切）；
        # 用于"720视野缩放到480"类选项：先裁 720x720 视野再降采样，
        # 保留视野、只减像素
        self._lerobot_crop_side = None
        # action 标签前瞻帧数（action[t]=pose[t+N]），界面“前瞻帧”可调；
        # 5~10 增大每步目标增量、提高信噪比，抑制模型输出摆动
        self._action_lookahead = 5
        self._lerobot_record_thread = None  # 后台录制线程

        # VLA 推理状态
        self._vla_thread = None
        self._vla_running = False
        self._vla_control_paused = False  # True: 模型仍在运行，但不把动作下发给机械臂
        self._motion_cb = None       # fn(cmd, *args)，由主窗口转发到机械臂 worker
        self._vla_lowpass_enabled = True  # 动作输出 EMA 低通开关（界面复选框，默认开启）
        self._vla_lowpass_alpha = 0.8     # 低通 EMA 系数 α（复选框旁文本框实时可调：
                                          #  越小滤波越强、响应越慢；1.0=不滤波）
        self._vla_avg_n = 1          # 多次采样平均次数（界面复选框，默认 1=关闭）

        # 实时价值判断（recap_value 201-bin 价值头，RECAP §4.7 部署侧）：
        # 推理中 2Hz 计算 V(s)（预期剩余代价 → 预计剩余时间）与回顾式 TD 优势
        # （A = −窗口步数代价 + ΔV；正=拉近完成，负=耽误/恶化）
        self._value_judge = None          # ValueJudge 实例（加载完成后）
        self._value_load_thread = None    # 后台加载线程
        self._value_timer = None          # 判断定时器（2Hz）
        self._value_history = deque()     # (t_monotonic, V) 回顾窗口
        self._value_window_sec = 2.0      # TD 回顾窗口（秒）
        self._value_tick_count = 0
        self._value_last_sign = 0         # 上次判定符号（翻转时写日志）

        # 云端 SSH 隧道托管状态（界面填 SSH 地址后自动建/复用隧道）
        self._tunnel_proc = None     # 本程序拉起的隧道进程（sshpass+autossh/ssh）
        self._tunnel_target = None   # 该进程对应的 "user@host:port"

        # “推理一次”单次执行诊断状态
        self._single_active = False      # 单次 chunk 正在下发执行
        self._single_queue = deque()     # 待执行的单次 chunk 动作
        self._single_timer = None        # 按 fps 消费单次队列的定时器
        self._single_exec_log = []       # 执行期间 (目标, 反馈位姿) 记录，用于分析往复运动
        self._single_save_dir = None     # 本次诊断数据保存目录

        # RTC (Real-Time Chunking) 管道平滑块执行状态：
        # 当前段未执行完时就提前推理下一段，新段到达后立即以 servo 插值从当前
        # 设定点接续（位置连续、无保位段、无停顿）——相当于两个“推理一次”
        # 之间做插值拼接
        self._rtc_mode = False       # 本次推理是否启用 RTC（启动时读取复选框）
        self._rtc_inflight = False   # 是否有一次 chunk 推理正在进行
        self._rtc_chunk_count = 0    # 已接收 chunk 数（状态显示用）
        self._rtc_last_sent = None   # 上一段的 6 维端点目标（段间 EMA 基准）
        self._rtc_last_dir = None    # 上一段实际下发的运动方向单位向量（往复抑制用）
        self._rtc_last_rot_dir = None  # 上一段实际下发的姿态变化方向单位向量
        self._rtc_next_timer = None  # 下一次推理的提前调度定时器（单次触发）
        # 方向单调性约束（往复抑制升级：投影剥离）。新 chunk 端点可能落在
        # 当前位置"身后"（观测时刻早于下发时刻 + 采样噪声），直接执行会让
        # 所有关节反向来回。把位移/姿态增量沿已建立方向分解，剥掉反向分量、
        # 保留侧向修正 → 主方向运动单调；大幅反向（真实意图）或连续 6 段
        # （~3s）坚持反向（慢速真实撤退）则放行完整反向段
        self._rtc_deadband = 0.006     # 方向基准更新的最小位移 (m)，兼作噪声死区
        self._rtc_genuine_rev = 0.04   # 位置真实反向判定阈值 (m)：大于则放行
        self._rtc_genuine_rev_rot = 0.25  # 姿态真实反向判定阈值 (rad)
        self._rtc_proj_streak = 0      # 连续被投影剥离的段计数（≥6 放行一次）
        self._rtc_target_step = 10   # 段目标取 chunk 第 N 步（窗口固定 0.5s 不变，
                                     # N 越大瞄准越远点 → 成比例提速；N=10 为原速）
        # RTC 夹爪控制（v4，学习手柄 xbox_control 的摇杆夹爪策略）：
        # 角度增大=闭合，减小=张开。每段新 chunk 只取窗口末值（最新意图），
        # 从当前位置限速斜坡推进（不重放序列 → 消除段间回拉往复）；
        # 力矩保护：闭合方向 soft 预减速 → 触限锁定保位 → 出现张开意图解锁
        self._grip_close_speed = 1.0   # 闭合斜坡速度 (rad/s)，与手柄 GRIPPER_SPEED 一致
        self._grip_open_speed = 1.5    # 张开斜坡速度 (rad/s)
        self._grip_tq_max = 0.5        # 闭合力矩上限 (Nm)：≥此值停止推进（保位）
        self._grip_tq_soft_ratio = 0.6  # 预减速软阈值比例（同手柄 _GRIPPER_TORQUE_SOFT_RATIO）
        self._grip_tq_min = -0.5       # 张开力矩下限 (Nm)
        self._grip_target = None       # RTC 夹爪当前目标值（None=无目标停止）
        self._rtc_grip_timer = None    # RTC 夹爪斜坡定时器（20Hz）
        self._grip_close_locked = False  # 闭合触限锁定标志
        self._grip_open_locked = False   # 张开触限锁定标志
        self._grip_last_sent = None    # 上一次下发的夹爪值（仅记录用）
        self._rtc_grip_timer = None    # RTC 夹爪限速斜坡定时器（50ms）

        # trained RTC（训练时 RTC, arXiv:2512.05964）执行跟踪：记录本段在执行
        # chunk 中的起始步索引与下发时刻；请求下一段时推算推理期间将被执行的
        # 步索引 k 与预测延迟 d，交给推理线程构造 prev_chunk_left_over 条件
        self._rtc_exec_offset = 0      # 本段 rows 在原始 chunk 中的起始步索引（=上一段的 d）
        self._rtc_exec_t0 = 0.0        # 本段下发时刻（perf_counter），0=无在执行段
        self._rtc_last_latency = 0.0   # 上一次实测推理延迟（秒），用于预测下一段 d

        # RTC 可调参数（界面 RTC 参数行，实时生效）
        # 步长mm 即段执行速度旋钮：段时长 = 段位移 ÷ (步长mm × fps)，
        # 即执行速度 = 步长mm × 20（5mm→0.10m/s，8mm→0.16m/s）；
        # 同时限制单段最大位移 = 步长mm × 10 步（8mm→80mm 安全限幅）
        self._rtc_max_step_mm = 8.0    # 每步末端位置限幅 (mm/步)，折算段窗口限速
        self._rtc_max_rot_step = 0.05  # 每步姿态限幅 (rad/步)
        # （原段端点 EMA 低通系数 _rtc_smooth_alpha 已合并到低通滤波文本框
        #   _vla_lowpass_alpha：同一参数单一控件，避免两处编辑不同步）
        self._rtc_interp = "linear"    # 伺服插值方式：none/linear

        # 外部控制互锁回调：当 VLA 与手柄/其他控制源需要切换时，通知外部模块
        self._on_vla_control_pause_changed = None  # callable(paused: bool)
        self._on_vla_takeover_requested = None     # callable() -> 请求外部释放控制权

        # 普通模式顺序块执行状态：一次推理一整段 chunk，以 MoveJ（S 曲线插值）
        # 方式平滑执行到末端位姿，执行窗口结束后再推理下一段——与“推理一次”
        # 同一控制逻辑，避免 20Hz 伺服滴灌因 chunk 边界跳变造成的猛冲/抖动
        self._seq_inflight = False   # 已请求、等待 chunk 返回
        self._seq_chunk_count = 0    # 已执行 chunk 数（状态显示用）
        self._seq_timer = None       # 执行窗口定时器（单次触发，到期请求下一段）
        self._seq_grip_timer = None  # 夹爪 20Hz 滴灌定时器
        self._seq_grip_values = deque()  # 当前 chunk 的夹爪序列
        self._seq_max_jump = 0.10    # 每段末端位置最大位移 (m)，安全限幅
        self._seq_max_rot = 0.6      # 每段末端姿态最大变化 (rad)

        # LeRobot 录制定时器：控制采集频率，主线程只负责把帧推入队列
        self._lerobot_timer = QTimer(self)
        self._lerobot_timer.timeout.connect(self._lerobot_push_frame)
        self._lerobot_timer.setTimerType(Qt.TimerType.PreciseTimer)

        # 推理录制（部署 rollout 采集，RECAP P2）状态
        # 与遥操作录制互斥；数据集固定 ela3_rollout；接管自动标注干预段（无确认制）
        self._rollout_recording = False
        self._rollout_record_thread = None
        self._rollout_dataset = None
        self._rollout_repo_name = "ela3_rollout"
        self._rollout_frame_count = 0
        self._rollout_staged_intervals = []   # 暂存接管区间（保存时统一写入标注）
        self._rollout_takeover_start = None   # 当前接管起始推帧计数（None=未在接管中）
        self._rollout_takeover_count = 0      # 本次会话接管次数（弹窗展示/日志用）
        # 接管前自动标 bad 的窗口帧数（APO 规则；0=关闭；20 帧@20fps=1s，
        # APO 原文为 10 个动作，此处放宽可按效果调小）
        self._rollout_pre_bad_frames = 20

        # π*0.6 在线 RL 元数据（PI06_RL_迁移实施方案.md R1a）：策略版本 + 任务组
        self._rollout_policy_tag = None        # 本次录制开始时的策略标识（模型路径）
        self._rollout_active_group_id = None   # 本次录制所属任务组（None=未分组）

        self._setup_ui()

        # 任务组下拉框：启动即加载历史组名（首项留空=不分组；开录时再刷新）
        self.rollout_group_input.addItem("")
        self._refresh_rollout_group_labels()

        # UI 立即显示，后台线程只检测索引
        self.status_label.setText("正在连接摄像头...")
        self.detect_thread = CameraDetectThread()
        self.detect_thread.indices_found.connect(self.on_indices_found)
        self.detect_thread.start()

        # 显示刷新定时器：统一15fps，避免每帧都刷新UI
        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._refresh_display)
        self._display_timer.start(66)  # 15fps

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 三个视频显示标签，横向排列（第3路仅显示，不参与录制/推理）
        video_layout = QHBoxLayout()
        self.video_labels = []
        self.port_labels = []
        for i in range(3):
            col = QVBoxLayout()
            col.setSpacing(2)
            label = QLabel(f"摄像头 {i + 1}\n未连接")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background-color: #222; color: #fff; font-size: 12px;")
            label.setMinimumSize(240, 180)
            label.setScaledContents(False)
            self.video_labels.append(label)
            col.addWidget(label)
            port_label = QLabel("USB: —")
            port_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            port_label.setStyleSheet("color: #666; font-size: 10px;")
            self.port_labels.append(port_label)
            col.addWidget(port_label)
            video_layout.addLayout(col)
        layout.addLayout(video_layout)

        # 状态标签
        self.status_label = QLabel("准备就绪")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #0078d7; font-size: 11px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # 控制栏
        control_layout = QHBoxLayout()

        self.res_combo = QComboBox()
        for name, *_ in self.resolutions:
            self.res_combo.addItem(name)
        self.res_combo.setCurrentIndex(0)
        self.res_combo.currentIndexChanged.connect(self.change_resolution)

        # LeRobot 图像分辨率设置
        self.lerobot_res_combo = QComboBox()
        self.lerobot_res_options = [
            ("1280 x 720", 1280, 720),
            ("720 x 720 (中心裁切)", 720, 720),
            ("640 x 480", 640, 480),
            ("480 x 480 (中心裁切)", 480, 480),
            # 第4元素 crop_side：先裁该边长的中心正方形（保留 720 视野），
            # 再缩放到 480x480 存储，编码压力约为 720x720 的 44%
            ("480 x 480 (720视野缩放, 低性能推荐)", 480, 480, 720),
        ]
        for name, *_ in self.lerobot_res_options:
            self.lerobot_res_combo.addItem(name)
        self.lerobot_res_combo.setCurrentIndex(1)  # 默认 720 x 720 中心裁切（与训练数据一致）
        self.lerobot_res_combo.setToolTip(
            "录制/推理共用的图像尺寸：\n"
            "- 1280x720：完整 16:9 视野（letterbox 后有效 512x288）\n"
            "- 720x720 中心裁切：方形满幅利用 512x512 画布（有效像素 +78%），"
            "水平视野变窄，左右两侧拍不到\n"
            "- 640x480：4:3 视野（有效 512x384，+33%）\n"
            "- 480x480 中心裁切：以画面中心 480x480 区域录制（视野变窄）\n"
            "- 480x480 (720视野缩放)：保留 720x720 中心视野，仅降采样到 "
            "480x480 存储，编码/存储压力约为 720x720 的 44%，"
            "适合无 GPU 的弱性能工控机软编码\n"
            "长宽比不一致时自动中心裁切（无拉伸失真）。\n"
            "注意：改动后需重录重训，新旧数据不可混训"
        )
        self.lerobot_res_combo.currentIndexChanged.connect(self._change_lerobot_resolution)

        self.btn_snap = QPushButton("拍照 (Snap)")
        self.btn_snap.setStyleSheet("font-size: 12px; padding: 4px 12px;")
        self.btn_snap.clicked.connect(self.take_snapshot)

        self.btn_record = QPushButton("开始录像")
        self.btn_record.setStyleSheet("font-size: 12px; padding: 4px 12px;")
        self.btn_record.clicked.connect(self.toggle_record)

        self.btn_lerobot = QPushButton("开始LeRobot录制")
        self.btn_lerobot.setStyleSheet(
            "font-size: 12px; padding: 4px 12px; background-color: #4CAF50; color: white;"
        )
        self.btn_lerobot.clicked.connect(self.toggle_lerobot_record)

        # 推理录制（部署 rollout 采集）：仅 VLA 推理状态可启动；录制 ela3_rollout，
        # 接管自动标注（前 20 帧 bad + 干预段恒正），停止时弹窗评价成败（RECAP P2）
        self.btn_rollout_record = QPushButton("推理录制")
        self.btn_rollout_record.setStyleSheet(
            "font-size: 12px; padding: 4px 12px; background-color: #00695C; color: white;"
        )
        self.btn_rollout_record.setToolTip(
            "部署 rollout 数据采集（RECAP）：\n"
            "- 仅 VLA 推理运行时可启动，与遥操作录制互斥\n"
            "- 录制到独立数据集 ela3_rollout（含失败片段，成败都保留）\n"
            "- 录制中手柄接管自动生成标注：接管前 20 帧=bad（策略失败动作，\n"
            "  P1 降权 / P3 critic 监督），接管期间=干预段(恒正优势)\n"
            "- 停止时弹窗选择任务最终评价（成功/失败/跳过），支持多次接管")
        self.btn_rollout_record.clicked.connect(self.toggle_rollout_record)

        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("Task description in English, e.g., 'pick up the red cube'")
        self.task_input.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        # 限制只能输入英文、数字、空格和常见标点（禁止中文）
        from PyQt6.QtCore import QRegularExpression
        from PyQt6.QtGui import QRegularExpressionValidator
        self.task_input.setValidator(QRegularExpressionValidator(
            QRegularExpression("^[a-zA-Z0-9_\\s\\-',.()]+$")
        ))

        # LeRobot 录制文件夹名输入框：默认 ela3_demo，可修改为其他名称。
        # 开始录制时：文件夹存在且格式兼容 -> 追加录制；存在但格式不兼容 ->
        # 弹窗提示修改文件夹名；不存在 -> 新建数据集。
        self.repo_input = QLineEdit(self._lerobot_repo_name)
        self.repo_input.setPlaceholderText("ela3_demo")
        self.repo_input.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        self.repo_input.setFixedWidth(140)
        self.repo_input.setToolTip(
            "LeRobot 录制文件夹名（默认 ela3_demo）：\n"
            "- 文件夹存在且格式兼容：追加录制到该数据集\n"
            "- 文件夹存在但格式不兼容：弹窗提示，请修改文件夹名\n"
            "- 文件夹不存在：自动新建数据集\n"
            "仅允许字母、数字、下划线、连字符")
        self.repo_input.setValidator(QRegularExpressionValidator(
            QRegularExpression("^[a-zA-Z0-9_\\-]+$")
        ))

        control_layout.addWidget(QLabel("分辨率:"))
        control_layout.addWidget(self.res_combo)
        control_layout.addWidget(QLabel("LeRobot图像:"))
        control_layout.addWidget(self.lerobot_res_combo)

        # 画面显示开关（默认不显示）：关闭时摄像头照常采集/录制/推理，
        # 仅不刷新界面画面；勾选后实时显示摄像头画面
        self.chk_show_video = QCheckBox("显示画面")
        self.chk_show_video.setStyleSheet("font-size: 11px;")
        self.chk_show_video.setChecked(self._show_video)
        self.chk_show_video.setToolTip(
            "显示画面（默认不勾选）：\n"
            "- 勾选后实时显示摄像头画面\n"
            "- 不勾选时摄像头照常采集/录制/推理，仅不刷新界面，降低 CPU 占用")
        self.chk_show_video.toggled.connect(self._on_show_video_toggled)
        control_layout.addWidget(self.chk_show_video)

        # 对焦控制：自动对焦开关 + 固定焦距值
        self.chk_autofocus = QCheckBox("自动对焦")
        self.chk_autofocus.setStyleSheet("font-size: 11px;")
        self.chk_autofocus.setChecked(self._autofocus_enabled)
        self.chk_autofocus.toggled.connect(self._on_autofocus_toggled)
        control_layout.addWidget(self.chk_autofocus)

        control_layout.addWidget(QLabel("焦距:"))
        self.focus_spin = QDoubleSpinBox()
        self.focus_spin.setRange(0.0, 1.0)
        self.focus_spin.setSingleStep(0.05)
        self.focus_spin.setDecimals(2)
        self.focus_spin.setValue(self._focus_value)
        self.focus_spin.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        self.focus_spin.setFixedWidth(70)
        self.focus_spin.setEnabled(not self._autofocus_enabled)
        self.focus_spin.valueChanged.connect(self._on_focus_changed)
        control_layout.addWidget(self.focus_spin)

        # 锁定曝光开关（默认不勾选）：勾选瞬间读取各摄像头当前曝光值并全程锁定；
        # 取消勾选则恢复自动曝光，并清空已记录的锁定值
        self.chk_exposure_lock = QCheckBox("锁定曝光")
        self.chk_exposure_lock.setStyleSheet("font-size: 11px;")
        self.chk_exposure_lock.setChecked(self._exposure_locked)
        self.chk_exposure_lock.setToolTip(
            "锁定曝光（默认不勾选）：\n"
            "- 勾选瞬间读取摄像头当前曝光值，并切换为手动曝光全程锁定该值\n"
            "- 防止录制过程中自动曝光漂移导致画面明暗变化\n"
            "- 取消勾选恢复自动曝光；重新勾选将以当前曝光为准重新锁定")
        self.chk_exposure_lock.toggled.connect(self._on_exposure_lock_toggled)
        control_layout.addWidget(self.chk_exposure_lock)

        control_layout.addWidget(QLabel("前瞻帧:"))
        self.spin_lookahead = QDoubleSpinBox()
        self.spin_lookahead.setRange(1, 20)
        self.spin_lookahead.setSingleStep(1)
        self.spin_lookahead.setDecimals(0)
        self.spin_lookahead.setValue(self._action_lookahead)
        self.spin_lookahead.setFixedWidth(58)
        self.spin_lookahead.setStyleSheet("font-size: 11px; padding: 1px 2px;")
        self.spin_lookahead.setToolTip(
            "action 标签前瞻帧数 N：录制时 action[t] = pose[t+N]。\n"
            "- 1 = 下一帧位姿（最低要求，避免 action==state 退化）\n"
            "- 5~10 = 每步目标增量 ×N，信噪比更高，模型输出更平滑\n"
            "注意：与旧数据（N=1）混训前需用 fix_action_labels.py --lookahead N 统一；"
            "录制期间锁定不可改")
        self.spin_lookahead.valueChanged.connect(
            lambda v: setattr(self, "_action_lookahead", int(v)))
        control_layout.addWidget(self.spin_lookahead)

        control_layout.addStretch()
        control_layout.addWidget(QLabel("文件夹:"))
        control_layout.addWidget(self.repo_input)
        control_layout.addWidget(QLabel("任务:"))
        control_layout.addWidget(self.task_input, stretch=1)
        control_layout.addWidget(self.btn_snap)
        control_layout.addWidget(self.btn_record)
        # 任务组下拉框（π*0.6 组相对优势，可编辑）：下拉可选历史组名继续该组，
        # 手动输入新组名 = 新建组（开录时生效）；留空 = 不分组。
        # group_id = 组名 | 策略短名：同组名+同策略的所有录制归属同一组
        # （跨会话延续是显式选择同名组，视为有意合并，无需"新组"按钮）
        self.rollout_group_input = QComboBox()
        self.rollout_group_input.setEditable(True)
        self.rollout_group_input.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.rollout_group_input.setFixedWidth(170)
        self.rollout_group_input.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        line_edit = self.rollout_group_input.lineEdit()
        line_edit.setPlaceholderText("任务组(可空), 如 做饭-开火")
        self.rollout_group_input.setToolTip(
            "任务组标记（π*0.6 组相对优势，纯元数据，不进模型输入）：\n"
            "- 下拉选择历史组名 = 继续该组；输入新名称 = 新建组\n"
            "- 同一场景摆放 + 同一任务 + 同一策略，连录 4~10 轮 = 一组\n"
            "- 组名用自然语言短描述（如 做饭-开火 / 搬运-左上到右下）\n"
            "- 完整 group_id = 组名 | 策略短名（同组名+同策略自动归并）\n"
            "- 留空 = 不分组（普通数据，走现有标注规则）\n"
            "⚠ 勿把组名写进上方\"任务\"框（那是给模型看的任务指令）")

        control_layout.addWidget(self.btn_lerobot)
        control_layout.addWidget(QLabel("组:"))
        control_layout.addWidget(self.rollout_group_input)
        control_layout.addWidget(self.btn_rollout_record)
        layout.addLayout(control_layout)

        # VLA 推理控制栏
        vla_layout = QHBoxLayout()
        vla_layout.addWidget(QLabel("VLA推理 任务:"))
        self.vla_task_input = QComboBox()
        self.vla_task_input.setEditable(True)
        self.vla_task_input.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.vla_task_input.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        self.vla_task_input.setToolTip(
            "推理任务描述：选择模型后自动列出其训练数据集中的任务，也可手动输入")
        self.vla_task_input.lineEdit().setPlaceholderText("选择模型后自动列出训练任务，可手输")
        self.vla_task_input.currentTextChanged.connect(self._on_vla_task_changed)
        vla_layout.addWidget(self.vla_task_input, stretch=1)
        vla_layout.addWidget(QLabel("模型:"))
        self.vla_model_edit = QLineEdit("./ckpt/smolvla_ela3_torque_la5/checkpoints/last")
        self.vla_model_edit.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        self.vla_model_edit.editingFinished.connect(self._refresh_task_candidates)
        vla_layout.addWidget(self.vla_model_edit, stretch=2)
        self.btn_vla_browse = QPushButton("浏览")
        self.btn_vla_browse.setStyleSheet("font-size: 12px; padding: 4px 8px;")
        self.btn_vla_browse.clicked.connect(self._browse_vla_model)
        vla_layout.addWidget(self.btn_vla_browse)
        self.btn_vla = QPushButton("开始推理")
        self.btn_vla.setStyleSheet(
            "font-size: 12px; padding: 4px 12px; background-color: #9C27B0; color: white;"
        )
        self.btn_vla.clicked.connect(self.toggle_vla_inference)
        vla_layout.addWidget(self.btn_vla)

        # RTC (Real-Time Chunking) 管道平滑块执行：当前段未执行完时就提前
        # 推理下一段（按实测推理延迟提前触发），新段到达后立即以 servo 插值
        # 从当前设定点接续——相当于两个“推理一次”之间做插值拼接，
        # 位置连续、无保位段、无停顿、无 chunk 边界猛冲
        self.chk_rtc = QCheckBox("RTC异步")
        self.chk_rtc.setStyleSheet("font-size: 11px; color: #9C27B0; font-weight: bold;")
        self.chk_rtc.setToolTip(
            "Real-Time Chunking 管道平滑块执行：\n"
            "- 当前段（0.5s）未执行完时，提前采集图像/力矩推理下一段\n"
            "- 新段推理完成后立即下发，从当前设定点 200Hz 插值接续\n"
            "- 位置连续、无停顿；每段一次平滑插值，无边界猛冲\n"
            "（对比普通模式：顺序块执行，段间有微小停顿）"
        )
        vla_layout.addWidget(self.chk_rtc)

        # 训练时 RTC（arXiv:2512.05964）：仅对 rtc_training_max_delay>0 训练的
        # pi05 checkpoint 生效，需配合"RTC异步"。推理时把上一段剩余动作 hard
        # clamp 进新 chunk 前缀（prefix 条件化），时滞由模型吸收——替代应用层
        # 的 d 步丢弃补偿；模型不支持时自动退回原补偿方式
        self.chk_rtc_trained = QCheckBox("训练RTC")
        self.chk_rtc_trained.setChecked(True)
        self.chk_rtc_trained.setStyleSheet("font-size: 11px; color: #AD1457; font-weight: bold;")
        self.chk_rtc_trained.setToolTip(
            "训练时 RTC（Training-Time Action Conditioning, arXiv:2512.05964）：\n"
            "- 需配合\"RTC异步\"，且模型须带 rtc_training_max_delay>0 训练\n"
            "- 推理时以上一段剩余动作为条件（hard clamp 前缀），零额外推理开销\n"
            "- 时滞由模型吸收，替代应用层 d 步丢弃；论文表明高延迟下优于丢弃/inpainting\n"
            "- 模型不支持时自动退回原 d 步丢弃补偿（仅提示，不报错）"
        )
        vla_layout.addWidget(self.chk_rtc_trained)

        # 动作输出 EMA 低通开关（默认开启）：抑制 flow-matching 采样噪声与
        # chunk 边界跳变造成的真机抖动；普通/RTC 两种模式均生效
        self.chk_lowpass = QCheckBox("低通滤波")
        self.chk_lowpass.setChecked(True)
        self.chk_lowpass.setStyleSheet("font-size: 11px; color: #2E7D32; font-weight: bold;")
        self.chk_lowpass.setToolTip(
            "对模型输出动作做 EMA 低通（out = α*新动作 + (1-α)*上次目标）：\n"
            "- 抑制高频采样噪声与 chunk 边界跳变，真机更平稳\n"
            "- 平滑系数 α 用右侧文本框调节（默认 0.8），运行中实时生效\n"
            "- 关闭后动作直发，仅保留安全限幅"
        )
        self.chk_lowpass.toggled.connect(
            lambda on: setattr(self, "_vla_lowpass_enabled", bool(on)))
        vla_layout.addWidget(self.chk_lowpass)

        # 低通滤波阈值 α 文本框：推理运行中实时修改（回车/失焦生效），
        # 取代原 RTC 参数行的"平滑α"旋钮（同一参数单一控件）
        self.edit_lowpass_alpha = QLineEdit(f"{self._vla_lowpass_alpha:.2f}")
        self.edit_lowpass_alpha.setFixedWidth(42)
        self.edit_lowpass_alpha.setStyleSheet("font-size: 11px; padding: 1px 4px;")
        self.edit_lowpass_alpha.setToolTip(
            "低通滤波系数 α（0.05~1.0，回车或失焦生效，推理中实时调整）：\n"
            "out = α*新动作 + (1-α)*上次目标\n"
            "- 0.8（默认）：兼顾平滑与速度，推荐\n"
            "- 调小（0.5/0.3）：滤波更强、动作更平稳，但响应变慢（拖影）\n"
            "- 1.0：不滤波，最快，采样噪声直接体现为段间抖动\n"
            "- ≤0.5 会明显拖慢闭环速度，可配合增大 RTC\"目标步N\"补偿"
        )
        self.edit_lowpass_alpha.editingFinished.connect(self._on_lowpass_alpha_changed)
        self.chk_lowpass.toggled.connect(self.edit_lowpass_alpha.setEnabled)
        vla_layout.addWidget(self.edit_lowpass_alpha)

        # 多次采样平均开关（默认关闭）：同一观测采 3 次 chunk 取均值，
        # 抑制采样摆动（实测 chunk 路径/净位移 -30%）；代价是推理耗时 ×3
        self.chk_avg3 = QCheckBox("3次采样平均")
        self.chk_avg3.setChecked(False)
        self.chk_avg3.setStyleSheet("font-size: 11px; color: #00695C; font-weight: bold;")
        self.chk_avg3.setToolTip(
            "对同一观测以不同随机噪声推理 3 次，动作 chunk 逐帧取平均：\n"
            "- 显著抑制模型输出摆动（实测 -30%）\n"
            "- 推理耗时 ×3（普通模式可能跟不上 20fps，建议配合 RTC 异步）\n"
            "- 运行中可实时切换"
        )
        self.chk_avg3.toggled.connect(self._on_avg3_toggled)
        vla_layout.addWidget(self.chk_avg3)

        # [RECAP R1c/G5] 优势条件化部署开关：给任务文本恒加 "advantage: positive, "
        # 前缀（与训练侧 factory.py 条件化注入格式严格一致）。
        # ⚠ 仅对开过 advantage_conditioning 训练的模型开启；普通 SFT 模型开启
        # 会造成训练/部署分布不一致
        self.chk_advantage_condition = QCheckBox("advantage条件")
        self.chk_advantage_condition.setChecked(False)
        self.chk_advantage_condition.setStyleSheet(
            "font-size: 11px; color: #5E35B1; font-weight: bold;")
        self.chk_advantage_condition.setToolTip(
            "优势条件化部署（pi*0.6 preconditioning）：\n"
            "- 给任务文本恒加前缀 'advantage: positive, '（命令模型出高优势动作）\n"
            "- 前缀格式与训练侧 advantage_conditioning 注入严格一致\n"
            "- 仅当模型用 recap 条件化训练过才可开启；普通模型开启=分布错配")
        self.chk_advantage_condition.toggled.connect(self._on_advantage_condition_toggled)
        vla_layout.addWidget(self.chk_advantage_condition)

        # 实时价值判断（recap_value 价值头，RECAP §4.7）：推理中实时显示
        # V(s)（预期剩余代价 → 预计剩余时间）与回顾式 TD 优势正负
        # （当前动作在拉近还是拉远完成）。仅作指示，不影响控制
        self.chk_value_judge = QCheckBox("价值判断")
        self.chk_value_judge.setChecked(False)
        self.chk_value_judge.setStyleSheet(
            "font-size: 11px; color: #00838F; font-weight: bold;")
        self.chk_value_judge.setToolTip(
            "实时价值判断（recap_value 201-bin 价值头）：\n"
            "- 推理中以 2Hz 计算 V(s)∈[-1,0]（预期剩余代价→预计剩余秒数）\n"
            "- 回顾式 TD 优势 A=−窗口步数代价+ΔV：▲正=在拉近完成，▼负=耽误/恶化\n"
            "- siglip 后端首次开启需后台加载视觉塔 ~10-20s（不卡界面）\n"
            "- 仅作指示/日志，不影响推理与控制")
        self.chk_value_judge.toggled.connect(self._on_value_judge_toggled)
        vla_layout.addWidget(self.chk_value_judge)

        self.value_model_edit = QLineEdit("./lerobot_data/recap_value_cfail100")
        self.value_model_edit.setFixedWidth(170)
        self.value_model_edit.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        self.value_model_edit.setToolTip(
            "价值模型目录（含 value_head.pt，recap_value.py train 的输出目录）\n"
            "siglip 后端的视觉塔固定取 ./pretrained_models/pi05_base")
        vla_layout.addWidget(self.value_model_edit)

        self.value_status_label = QLabel("V: —")
        self.value_status_label.setStyleSheet("font-size: 11px; color: #888;")
        self.value_status_label.setToolTip("V=预期剩余代价(→0 接近完成) | A=回顾式TD优势 | 预计剩余时间")
        vla_layout.addWidget(self.value_status_label)

        self.btn_vla_pause = QPushButton("暂停控制")
        self.btn_vla_pause.setStyleSheet(
            "font-size: 12px; padding: 4px 12px; background-color: #FF9800; color: white;"
        )
        self.btn_vla_pause.setEnabled(False)
        self.btn_vla_pause.clicked.connect(self.toggle_vla_control_pause)
        vla_layout.addWidget(self.btn_vla_pause)

        # “推理一次”：暂停控制状态下点击，执行单次推理的完整动作块后自动回到暂停，
        # 并把观测/结果保存到独立文件夹，用于诊断往复运动等故障
        self.btn_vla_once = QPushButton("推理一次")
        self.btn_vla_once.setStyleSheet(
            "font-size: 12px; padding: 4px 12px; background-color: #00695C; color: white;"
        )
        self.btn_vla_once.setEnabled(False)
        self.btn_vla_once.setToolTip(
            "在“暂停控制”状态下执行一次完整 chunk 推理并下发：\n"
            "- 信息框显示推理出的全部帧位置姿态\n"
            "- 执行完最后一步后自动回到“暂停控制”，机械臂停止\n"
            "- 摄像头画面/力矩/任务文本/动作块/实际执行轨迹保存到\n"
            "  inference_once_logs/<时间戳>/，用于分析是否来回运动"
        )
        self.btn_vla_once.clicked.connect(self._on_vla_once_clicked)
        vla_layout.addWidget(self.btn_vla_once)
        layout.addLayout(vla_layout)

        # “推理一次”结果信息框
        self.vla_once_info = QTextEdit()
        self.vla_once_info.setReadOnly(True)
        self.vla_once_info.setMaximumHeight(130)
        self.vla_once_info.setStyleSheet("font-family: monospace; font-size: 10px;")
        self.vla_once_info.setPlaceholderText(
            "点击“推理一次”后，这里显示模型输出的整段动作块（每帧: x y z rx ry rz gripper）")
        layout.addWidget(self.vla_once_info)

        # RTC 参数行：实时生效，仅 RTC 模式使用
        rtc_layout = QHBoxLayout()
        rtc_layout.addWidget(QLabel("RTC参数:"))

        rtc_layout.addWidget(QLabel("插值:"))
        self.combo_rtc_interp = QComboBox()
        for _name, _mode in [("线性", "linear"), ("不插值", "none")]:
            self.combo_rtc_interp.addItem(_name, _mode)
        self.combo_rtc_interp.setStyleSheet("font-size: 11px;")
        self.combo_rtc_interp.setToolTip(
            "伺服插值方式（200Hz 上采样，IK 后在关节空间执行）：\n"
            "- 线性：段内匀速，段间位置连续\n"
            "- 不插值：直接写关节目标，仅靠控制循环平滑")
        self.combo_rtc_interp.currentIndexChanged.connect(
            lambda i: setattr(self, "_rtc_interp",
                              self.combo_rtc_interp.itemData(i)))
        rtc_layout.addWidget(self.combo_rtc_interp)

        def _rtc_spin(minv, maxv, step, decimals, value, tooltip, setter, width=62):
            sp = QDoubleSpinBox()
            sp.setRange(minv, maxv)
            sp.setSingleStep(step)
            sp.setDecimals(decimals)
            sp.setValue(value)
            sp.setFixedWidth(width)
            sp.setStyleSheet("font-size: 11px; padding: 1px 2px;")
            sp.setToolTip(tooltip)
            sp.valueChanged.connect(setter)
            return sp

        rtc_layout.addWidget(QLabel("步长mm:"))
        self.spin_rtc_step = _rtc_spin(
            0.5, 50.0, 0.5, 1, self._rtc_max_step_mm,
            "段速度上限：单段最大位移 = 步长mm × 10步，段时长固定 0.5s，\n"
            "即最大速度 = 步长mm × 20（8mm → 0.16m/s）。调大运动更快。\n"
            "（段时长固定：短窗口会造成段间停顿/边界晃动）",
            lambda v: setattr(self, "_rtc_max_step_mm", float(v)))
        rtc_layout.addWidget(self.spin_rtc_step)

        rtc_layout.addWidget(QLabel("姿态rad:"))
        self.spin_rtc_rot = _rtc_spin(
            0.005, 0.5, 0.005, 3, self._rtc_max_rot_step,
            "每步姿态限幅 (rad/步)：折算每段窗口最大姿态变化，\n"
            "限制姿态变化速度，抑制姿态抖动",
            lambda v: setattr(self, "_rtc_max_rot_step", float(v)))
        rtc_layout.addWidget(self.spin_rtc_rot)

        rtc_layout.addWidget(QLabel("目标步N:"))
        self.spin_rtc_tstep = _rtc_spin(
            5, 40, 1, 0, self._rtc_target_step,
            "段目标取相对\"当前时刻\"领先 N 步（自动补偿推理时滞 d 步丢弃，\n"
            "段窗口固定 0.5s 不变）：\n"
            "- N=10（默认）：0.5s 新鲜前瞻，忠实演示原速\n"
            "- N=14~20：瞄准更远点 → 约 1.4~2 倍演示速度（提速旋钮）\n"
            "前瞻越远预测误差越大（由限幅/投影剥离兜底）；\n"
            "ACT chunk=10 时有效 N 上限≈10-时滞步数，SmolVLA chunk=40 可用",
            lambda v: setattr(self, "_rtc_target_step", int(v)),
            width=52)
        rtc_layout.addWidget(self.spin_rtc_tstep)

        # 云端推理（方案 A：全量云端推理）：模型部署在云服务器，本地只采集
        # 观测经 gRPC 上行，接收动作 chunk 执行。默认走 SSH 隧道（127.0.0.1:50051）。
        self.chk_cloud = QCheckBox("云端推理")
        self.chk_cloud.setStyleSheet("font-size: 11px; color: #1565C0; font-weight: bold;")
        self.chk_cloud.setToolTip(
            "云端推理（模型部署在云服务器）：\n"
            "- 本地只采集 2 路 720x720 JPEG + 位姿 + 力矩，经 gRPC 发送到云端\n"
            "- 云端返回完整动作 chunk，执行层（限幅/夹爪/RTC 调度）不变\n"
            "- 断链/超时自动安全停（保位），恢复需人工点'继续控制'\n"
            "- 填写 SSH 地址+密码后自动建立隧道，无需手动命令；\n"
            "  SSH 留空 = 公网直连或复用已存在的隧道"
        )
        rtc_layout.addWidget(self.chk_cloud)

        self.cloud_addr_edit = QLineEdit(CloudInferenceThread.DEFAULT_ADDR)
        self.cloud_addr_edit.setStyleSheet("font-size: 11px; padding: 1px 2px;")
        self.cloud_addr_edit.setFixedWidth(120)
        self.cloud_addr_edit.setToolTip(
            "云端 gRPC 地址。SSH 隧道方式固定为 127.0.0.1:50051；\n"
            "公网直连则填 <云端IP>:<端口>")
        rtc_layout.addWidget(self.cloud_addr_edit)

        self.cloud_token_edit = QLineEdit(CloudInferenceThread.DEFAULT_TOKEN)
        self.cloud_token_edit.setStyleSheet("font-size: 11px; padding: 1px 2px;")
        self.cloud_token_edit.setFixedWidth(110)
        self.cloud_token_edit.setPlaceholderText("访问令牌")
        self.cloud_token_edit.setToolTip("云端访问令牌（与服务端 --token 一致）")
        rtc_layout.addWidget(self.cloud_token_edit)

        self.cloud_ssh_edit = QLineEdit()
        self.cloud_ssh_edit.setStyleSheet("font-size: 11px; padding: 1px 2px;")
        self.cloud_ssh_edit.setFixedWidth(170)
        self.cloud_ssh_edit.setPlaceholderText("SSH user@主机:端口")
        self.cloud_ssh_edit.setToolTip(
            "云服务器 SSH 地址（如 root@connect.xxx.seetacloud.com:43394）。\n"
            "点'开始推理'时自动建立 SSH 隧道（首次连接自动接受主机密钥），\n"
            "且 cloud_inference/ 文件缺失时自动从服务器拉取；\n"
            "留空 = 公网直连（gRPC 填云端IP:端口）或复用已有隧道")
        rtc_layout.addWidget(self.cloud_ssh_edit)

        self.cloud_ssh_pass_edit = QLineEdit()
        self.cloud_ssh_pass_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.cloud_ssh_pass_edit.setStyleSheet("font-size: 11px; padding: 1px 2px;")
        self.cloud_ssh_pass_edit.setFixedWidth(90)
        self.cloud_ssh_pass_edit.setPlaceholderText("SSH 密码")
        self.cloud_ssh_pass_edit.setToolTip("云服务器 SSH 密码（仅用于建立隧道/拉取文件）")
        rtc_layout.addWidget(self.cloud_ssh_pass_edit)

        self.chk_cloud_save = QCheckBox("记住")
        self.chk_cloud_save.setStyleSheet("font-size: 11px;")
        self.chk_cloud_save.setToolTip(
            "勾'记住'后，SSH 地址/密码/gRPC 地址/令牌保存到\n"
            "debugger/cloud_settings.json（明文，仅本机可读权限 600），\n"
            "下次启动自动填入——多机器人部署时每台只需填一次")
        rtc_layout.addWidget(self.chk_cloud_save)

        self._load_cloud_settings()

        rtc_layout.addStretch()
        layout.addLayout(rtc_layout)

        # 保存路径提示标签
        self.lbl_save_dir = QLabel(f"保存目录: {self.save_dir}")
        self.lbl_save_dir.setStyleSheet("color: #666; font-size: 10px;")
        self.lbl_save_dir.setWordWrap(True)
        self.lbl_save_dir.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_save_dir)

        # 启动时按默认模型路径预填推理任务候选
        self._refresh_task_candidates()

    def on_indices_found(self, indices):
        """后台线程检测到可用索引后，在主线程中打开摄像头并启动采集线程。

        角色（臂1腕部/头部/臂2腕部）按持久化的 USB 端口绑定自动分配，
        camera_indices 重排为 [臂1腕部, 头部, 臂2腕部]，保证录制/推理的
        cam1/cam2/cam3 与摄像头硬件的对应关系不随 video 索引漂移。"""
        self._assign_camera_roles(indices)
        ordered = [self._role_to_idx[r] for r in self.CAMERA_ROLES
                   if self._role_to_idx.get(r) is not None]
        self.camera_indices = ordered + [i for i in indices if i not in ordered]
        cam_count = len(self.camera_indices)
        logger.info(f"[Camera] 检测到 {cam_count} 个摄像头: {indices}，"
                    f"角色映射: {self._role_map_text()}")

        for idx in self.camera_indices[:3]:
            self.open_camera(idx, *self.capture_res)

        if cam_count == 0:
            self.status_label.setText("未检测到摄像头")
            self.video_labels[0].setText("摄像头 1\n未连接")
            self.video_labels[1].setText("摄像头 2\n未连接")
            self.video_labels[2].setText("摄像头 3\n未连接")
        else:
            shown = self.camera_indices[:3]
            self.status_label.setText(
                f"已连接 {cam_count} 个摄像头：{self._role_map_text()}"
            )
            for i in range(len(shown), 3):
                self.video_labels[i].setText(f"摄像头 {i + 1}\n未连接")
            # 画面显示关闭时：已连接的摄像头提示“画面已隐藏”（采集/录制正常进行）
            if not self._show_video:
                for i in range(len(shown)):
                    self.video_labels[i].setText(f"摄像头 {i + 1}\n画面已隐藏")
        self._update_port_labels()

    # ------------------------------------------------- 角色 ↔ USB 端口绑定
    def _role_map_text(self):
        return "，".join(
            f"{self.CAMERA_ROLE_NAMES[r]}=video{self._role_to_idx[r]}"
            for r in self.CAMERA_ROLES if self._role_to_idx.get(r) is not None
        ) or "无"

    @staticmethod
    def _usb_id_of_index(idx):
        """摄像头索引对应的 USB 端口拓扑路径（如 3-2.1，重启不变）；读不到返回 None"""
        try:
            sys_link = f"/sys/class/video4linux/video{idx}/device"
            if os.path.islink(sys_link):
                return os.path.basename(os.readlink(sys_link))
        except Exception:
            pass
        return None

    def camera_usb_ports(self):
        """返回本组件检测到的摄像头端口 [(video_idx, usb_id)]，供设置对话框显示。

        与主界面 on_indices_found / _update_port_labels 使用同一检测结果和
        _usb_id_of_index，保证设置界面与主界面显示的摄像头串口号一致。
        检测未完成（camera_indices 为空）时返回空列表，由调用方兜底。
        """
        return [(idx, self._usb_id_of_index(idx))
                for idx in self.camera_indices[:3]]

    def _load_camera_settings(self):
        from debugger.utils.system_config import load_config, save_config
        cfg = load_config()
        roles = cfg.get("camera_roles")
        if not isinstance(roles, dict):
            roles = {}
        if not roles:
            roles = self._migrate_legacy_camera_roles()
            if roles:
                cfg["camera_roles"] = roles
                save_config(cfg)
        return {"roles": roles}

    def _migrate_legacy_camera_roles(self):
        """把旧版 debugger/camera_settings.json 迁移到 main_system_config.json。"""
        for path in self._LEGACY_CAMERA_SETTINGS_FILES:
            try:
                if not os.path.isfile(path):
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                roles = (data or {}).get("roles") if isinstance(data, dict) else None
                if isinstance(roles, dict) and roles:
                    logger.info(
                        f"[Camera] 已迁移旧摄像头配置 {path} -> main_system_config.json")
                    return dict(roles)
            except Exception as e:
                logger.warning(f"[Camera] 迁移旧摄像头配置失败 {path}: {e}")
        return {}

    def _save_camera_settings(self):
        from debugger.utils.system_config import load_config, save_config
        cfg = load_config()
        roles = dict(cfg.get("camera_roles") or {})
        for role, idx in self._role_to_idx.items():
            roles[role] = {"usb": self._usb_id_of_index(idx), "index": idx}
        cfg["camera_roles"] = roles
        try:
            save_config(cfg)
        except Exception as e:
            logger.warning(f"[Camera] 保存摄像头角色设置失败: {e}")

    def _assign_camera_roles(self, indices):
        """按持久化的 USB 端口绑定分配角色；未匹配的角色按索引顺序兜底并保存。"""
        shown = list(indices[:3])
        usb_map = {idx: self._usb_id_of_index(idx) for idx in shown}
        saved = self._load_camera_settings().get("roles", {})
        role_to_idx = {}
        used = set()
        # 1) USB 端口匹配（重启后 video 索引变化也能对上）
        for role in self.CAMERA_ROLES:
            usb = (saved.get(role) or {}).get("usb")
            if usb:
                for idx in shown:
                    if idx not in used and usb_map.get(idx) == usb:
                        role_to_idx[role] = idx
                        used.add(idx)
                        break
        # 2) 仅对从未绑定过的角色按剩余索引顺序兜底（首次使用时）；
        #    已绑定但本次缺席的角色保留原绑定，不抢占其他摄像头，
        #    避免某次检测不全把持久化映射打乱
        remaining = [idx for idx in shown if idx not in used]
        for role in self.CAMERA_ROLES:
            if (role not in role_to_idx and remaining
                    and not (saved.get(role) or {}).get("usb")):
                role_to_idx[role] = remaining.pop(0)
        self._role_to_idx = role_to_idx
        self._save_camera_settings()

    def apply_camera_roles(self):
        """设置对话框保存摄像头角色后重新按 USB 端口绑定分配，并刷新显示。

        主窗口在设置对话框保存后调用，使新的角色↔USB 端口映射立即生效，
        无需重启（重启后由 _assign_camera_roles 自动加载，二者同源）。
        """
        if not self.camera_indices:
            return
        self._assign_camera_roles(self.camera_indices)
        ordered = [self._role_to_idx[r] for r in self.CAMERA_ROLES
                   if self._role_to_idx.get(r) is not None]
        self.camera_indices = ordered + [i for i in self.camera_indices
                                         if i not in ordered]
        self._update_port_labels()
        self.status_label.setText(
            f"已连接 {len(self.camera_indices)} 个摄像头：{self._role_map_text()}")

    def _update_port_labels(self):
        """在各路画面下方显示 角色名 + /dev/videoN + USB 端口号"""
        ordered = self._display_order_indices()
        idx_to_role = {v: k for k, v in self._role_to_idx.items()}
        for i in range(3):
            if i < len(ordered):
                idx = ordered[i]
                role = idx_to_role.get(idx)
                prefix = f"{self.CAMERA_ROLE_NAMES[role]} " if role else ""
                self.port_labels[i].setText(prefix + self._usb_port_text(idx))
            else:
                self.port_labels[i].setText("USB: —")

    def _usb_port_text(self, idx):
        dev = f"/dev/video{idx}"
        usb_id = self._usb_id_of_index(idx)
        return f"{dev} (usb {usb_id})" if usb_id else dev

    def open_camera(self, idx, width, height):
        """在主线程中打开指定摄像头并启动后台采集线程"""
        # 关闭已有的
        if idx in self.capture_threads:
            self.capture_threads[idx].stop()
            del self.capture_threads[idx]
        if idx in self.caps and self.caps[idx] is not None:
            self.caps[idx].release()
            del self.caps[idx]

        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            logger.error(f"[Camera] 无法打开摄像头 [{idx}]")
            return

        # 使用MJPEG格式降低CPU占用
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲降低延迟

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != width or actual_h != height:
            logger.warning(
                f"[Camera] 摄像头 [{idx}] 实际分辨率 ({actual_w}x{actual_h}) "
                f"与请求分辨率 ({width}x{height}) 不一致，请检查摄像头是否支持该分辨率"
            )
        logger.info(f"[Camera] 摄像头 [{idx}] 已就绪，实际分辨率: {actual_w} x {actual_h}")

        # 应用对焦设置（自动对焦开关 / 固定焦距）
        self._apply_focus(cap)
        # 应用曝光锁定（若勾选）。新打开的摄像头需先让自动曝光收敛再冻结，
        # 否则会锁定到刚打开时未收敛的偏暗状态；已有锁定值的重开摄像头立即恢复。
        if self._exposure_locked:
            if idx in self._locked_exposure:
                self._apply_exposure_lock(cap, idx)
            else:
                QTimer.singleShot(
                    800, lambda c=cap, i=idx: self._apply_exposure_lock_deferred(c, i)
                )

        self.caps[idx] = cap

        # 启动后台采集线程
        thread = CameraCaptureThread(idx, cap)
        thread.frame_ready.connect(self._on_frame_received)
        thread.start()
        self.capture_threads[idx] = thread

    def _apply_focus(self, cap):
        """对单个 VideoCapture 应用当前对焦设置。

        AUTOFOCUS: 0=手动模式, 1=自动对焦。
        FOCUS: 归一化焦距值 0.0~1.0（仅手动模式生效），不同摄像头/驱动对该值
        的物理含义可能不同，0.5 通常对应中近焦段。
        """
        if cap is None:
            return
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self._autofocus_enabled else 0)
            if not self._autofocus_enabled:
                cap.set(cv2.CAP_PROP_FOCUS, float(self._focus_value))
                actual = cap.get(cv2.CAP_PROP_FOCUS)
                logger.info(
                    f"[Camera] 对焦=手动, 目标焦距={self._focus_value:.2f}, 实际={actual}"
                )
            else:
                logger.info("[Camera] 对焦=自动")
        except Exception as e:
            logger.warning(f"[Camera] 设置对焦失败: {e}")

    def _on_autofocus_toggled(self, checked):
        """自动对焦开关切换，应用到已打开的摄像头"""
        self._autofocus_enabled = checked
        self.focus_spin.setEnabled(not checked)
        for cap in self.caps.values():
            self._apply_focus(cap)

    def _on_show_video_toggled(self, checked):
        """画面显示开关：关闭时清空界面画面（采集/录制/推理不受影响）"""
        self._show_video = checked
        if not checked:
            for i, label in enumerate(self.video_labels):
                label.clear()
                label.setText(f"摄像头 {i + 1}\n画面已隐藏")

    def _on_focus_changed(self, value):
        """焦距值变化，应用到已打开的摄像头（仅手动模式生效）"""
        self._focus_value = float(value)
        if self._autofocus_enabled:
            return
        for cap in self.caps.values():
            self._apply_focus(cap)

    def _set_exposure_mode(self, cap, auto):
        """切换曝光模式，成功返回 True。

        OpenCV 各版本 V4L2 后端对 CAP_PROP_AUTO_EXPOSURE 的取值约定不一致
        （0.25/0.75 缩放约定，或整数直接透传 V4L2 枚举值），且部分摄像头
        auto_exposure 菜单仅支持 Manual(1) / AperturePriority(3)（实测本机
        OpenCV 4.13 直接透传整数值，0.75 会被取整为 1=手动），
        因此逐个候选值尝试并回读验证，确保模式真正生效。
        """
        # (目标 V4L2 枚举值, 候选 OpenCV 写入值)
        if auto:
            plans = [(3, (3.0, 0.75)), (0, (0.0,))]  # 自动：光圈优先(3)，退化为 AUTO(0)
        else:
            plans = [(1, (1.0, 0.25))]               # 手动：MANUAL(1)
        for target, values in plans:
            for v in values:
                try:
                    if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v):
                        continue
                    mode = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                    # 回读可能是原始枚举值(1/3)或归一化值(0.25/0.75)
                    if int(round(mode)) == target or abs(mode - target * 0.25) < 1e-3:
                        logger.info(
                            f"[Camera] 曝光模式={'自动' if auto else '手动'} "
                            f"(写入 {v}, 回读 {mode})"
                        )
                        return True
                except Exception:
                    continue
        logger.warning(f"[Camera] 曝光模式切换失败 (auto={auto})")
        return False

    def _apply_exposure_lock(self, cap, idx=None):
        """对单个 VideoCapture 应用曝光锁定/恢复设置。

        锁定流程（勾选时）：先切换到手动曝光——UVC 摄像头从自动切手动时会
        冻结当前自动曝光状态（画面亮度不变），稍候读取该真实曝光值存档，
        供摄像头重开后复用（全程锁定）。
        注意：不能在自动模式下先读曝光值再写回——实测本机摄像头自动模式下
        exposure_time_absolute 恒为默认值(83)，并非当前实际曝光，
        写回会导致画面突然变暗。
        取消勾选：恢复自动曝光。
        """
        if cap is None:
            return
        try:
            if self._exposure_locked:
                # 已有锁定值则复用（全程锁定同一曝光）；否则冻结当前自动曝光
                locked = self._locked_exposure.get(idx) if idx is not None else None
                if not self._set_exposure_mode(cap, auto=False):
                    return
                if locked is None:
                    # 切手动后曝光已冻结在当前自动值，稍候读取真实曝光存档
                    time.sleep(0.05)
                    locked = cap.get(cv2.CAP_PROP_EXPOSURE)
                    if locked is not None and locked > 0 and idx is not None:
                        self._locked_exposure[idx] = locked
                else:
                    # 摄像头重开：写回原锁定值
                    cap.set(cv2.CAP_PROP_EXPOSURE, float(locked))
                actual = cap.get(cv2.CAP_PROP_EXPOSURE)
                logger.info(
                    f"[Camera] 曝光=手动锁定, 锁定值={locked}, 实际={actual}"
                )
            else:
                # 恢复自动曝光：回到未锁定前的摄像头默认自动模式
                self._set_exposure_mode(cap, auto=True)
        except Exception as e:
            logger.warning(f"[Camera] 设置曝光失败: {e}")

    def _apply_exposure_lock_deferred(self, cap, idx):
        """摄像头打开后延迟执行的曝光锁定：等自动曝光收敛后再冻结当前曝光，
        避免锁定到刚打开时未收敛的偏暗状态。
        若期间摄像头已被关闭/替换，或用户取消了锁定，则跳过。"""
        if not self._exposure_locked:
            return
        if self.caps.get(idx) is not cap:
            return
        self._apply_exposure_lock(cap, idx)

    def _on_exposure_lock_toggled(self, checked):
        """锁定曝光开关：勾选时读取各摄像头当前曝光并全程锁定；取消则恢复自动曝光"""
        self._exposure_locked = checked
        if not checked:
            self._locked_exposure.clear()  # 解除锁定，清空记录值
        for idx, cap in self.caps.items():
            self._apply_exposure_lock(cap, idx)

    def _on_frame_received(self, camera_idx, frame):
        """后台线程收到新帧的回调——只缓存，不处理图像"""
        self.latest_frames[camera_idx] = frame
        self._pending_frames[camera_idx] = frame

        # 帧信号降频到10Hz发射，避免主线程过载
        now = time.time()
        last_emit = self._last_emit_time.get(camera_idx, 0)
        if now - last_emit >= 0.1:  # 100ms = 10Hz
            self._last_emit_time[camera_idx] = now
            self.frame_captured.emit(camera_idx, frame)

        # 录像直接写入（不经过UI线程）
        if self.is_recording and camera_idx in self.recorders:
            writer = self.recorders[camera_idx]
            if writer is not None:
                writer.write(frame)

    def _display_order_indices(self):
        """按界面布局（左=臂2腕部，中=头部，右=臂1腕部）返回显示顺序的摄像头索引；
        未分配角色的索引保持原相对顺序附后"""
        shown = list(self.camera_indices[:3])
        ordered = []
        for role in self.CAMERA_DISPLAY_ROLES:
            idx = self._role_to_idx.get(role)
            if idx is not None and idx in shown:
                ordered.append(idx)
        ordered += [i for i in shown if i not in ordered]
        return ordered

    def _refresh_display(self):
        """统一显示刷新：15fps，将缓存的帧转换为QPixmap显示。

        显示前按 LeRobot 目标分辨率做与录制一致的中心裁切，实现
        “所见即所录”：例如 LeRobot 选择 720x720 时，界面也显示
        从 1280x720 画面中心裁出的方形区域。
        """
        # 画面显示关闭时：仅丢弃待显示帧，不做任何转换/绘制（采集与录制不受影响）
        if not self._show_video:
            self._pending_frames.clear()
            return
        for i, idx in enumerate(self._display_order_indices()):
            if idx not in self._pending_frames:
                continue

            frame = self._pending_frames[idx]
            # 与 _resize_frame 一致的中心裁切：界面显示 = 实际录制视野
            frame = self._center_crop_to_lerobot(frame)
            # 显示时缩小尺寸减少计算量
            label = self.video_labels[i]
            target_w = min(label.width() - 4, 640)
            target_h = min(label.height() - 4, 480)
            if target_w <= 0 or target_h <= 0:
                continue

            # 如果帧比目标大，先用OpenCV快速缩放（比QPixmap缩放快）
            h, w = frame.shape[:2]
            if w > target_w or h > target_h:
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            label.setPixmap(pixmap)

        # 清空待处理帧（只保留最新）
        self._pending_frames.clear()

    def _first_camera_idx(self):
        """返回第一个可用摄像头索引，不存在则返回 None"""
        return self.camera_indices[0] if self.camera_indices else None

    def change_resolution(self, index):
        """切换采集分辨率并重新打开摄像头"""
        name, w, h = self.resolutions[index]
        logger.info(f"[Camera] 切换分辨率: {name}")
        self.capture_res = (w, h)

        if self.is_recording:
            self.toggle_record()

        for idx in self.camera_indices[:3]:
            if idx in self.caps:
                self.open_camera(idx, w, h)

    def take_snapshot(self):
        saved = []
        for idx in self.camera_indices[:3]:
            if idx not in self.latest_frames:
                logger.warning(f"[Camera] 摄像头 [{idx}] 没有可用帧，跳过拍照")
                continue
            frame = self.latest_frames[idx].copy()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_dir, f"SnapShot_cam{idx}_{timestamp}.jpg")
            try:
                ret, buf = cv2.imencode('.jpg', frame)
                if ret:
                    with open(filename, 'wb') as f:
                        f.write(buf.tobytes())
                    if os.path.exists(filename) and os.path.getsize(filename) > 0:
                        saved.append(filename)
                        logger.info(f"[Camera] 照片已保存: {filename}")
            except Exception as e:
                logger.error(f"[Camera] 保存异常: {e}")
        if saved:
            self.status_label.setText(f"已保存 {len(saved)} 张照片")
            QTimer.singleShot(100, lambda: QMessageBox.information(
                self, "拍照成功",
                f"已保存 {len(saved)} 张照片到:\n{self.save_dir}"
            ))
        else:
            QMessageBox.warning(self, "拍照失败", "没有可保存的图像，请检查摄像头连接。")

    def toggle_record(self):
        """开始/停止录像"""
        if self.is_recording:
            for idx in list(self.recorders.keys()):
                writer = self.recorders[idx]
                if writer is not None:
                    writer.release()
            self.recorders.clear()
            self.is_recording = False
            self.btn_record.setText("开始录像")
            self.btn_record.setStyleSheet("font-size: 12px; padding: 4px 12px;")
            self.status_label.setText("录像已结束")
            self.btn_record.setEnabled(True)
            QTimer.singleShot(100, lambda: QMessageBox.information(
                self, "录像结束",
                f"视频已保存到:\n{self.save_dir}"
            ))
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for idx in self.camera_indices[:3]:
                if idx not in self.caps:
                    continue
                cap = self.caps[idx]
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = 20.0
                filename = os.path.join(self.save_dir, f"Video_cam{idx}_{timestamp}.mp4")
                writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
                self.recorders[idx] = writer
                logger.info(f"[Camera] 开始录像: {filename} @ {w}x{h}")

            if self.recorders:
                self.is_recording = True
                self.btn_record.setText("停止录像")
                self.btn_record.setStyleSheet(
                    "font-size: 12px; padding: 4px 12px; background-color: #ff4444; color: white;"
                )
                self.status_label.setText("正在录像...")
            else:
                QMessageBox.warning(self, "提示", "没有可用的摄像头进行录像")

    def get_latest_frames(self):
        """获取最新帧，供外部lerobot数据集收集使用"""
        return self.latest_frames.copy()

    def set_end_pose_callback(self, callback):
        """设置获取末端位姿的回调函数。

        callback() 应返回 (ArmEndPose, gripper_angle_rad) 或仅 ArmEndPose。
        ArmEndPose 包含 x,y,z,rx,ry,rz (m/rad)。
        gripper_angle_rad 是夹爪角度（弧度）。
        """
        self._lerobot_get_end_pose = callback

    def set_end_pose2_callback(self, callback):
        """设置臂2 末端位姿回调，签名同 set_end_pose_callback（用于双臂录制）"""
        self._lerobot_get_end_pose2 = callback

    def set_torque2_callback(self, callback):
        """设置臂2 夹爪力矩回调，签名同 set_torque_callback（用于双臂录制）"""
        self._lerobot_get_torque2 = callback

    def set_torque_callback(self, callback):
        """设置获取夹爪力矩的回调函数。

        callback() 应返回夹爪当前力矩 (Nm, float) 或 None（读取失败时）。
        该力矩作为 observation.gripper_torque 写入 LeRobot 数据集。
        """
        self._lerobot_get_torque = callback

    def set_motion_callback(self, callback):
        """设置运动指令回调 fn(cmd, *args)，由主窗口转发到机械臂 worker。

        使用的指令：end_pose_ctrl(x,y,z,rx,ry,rz,duration,block)（欧拉角逆解）、
        gripper_ctrl(angle_rad)、cancel_motion()。
        """
        self._motion_cb = callback

    def set_vla_control_pause_callback(self, callback):
        """设置 VLA 控制暂停状态变化回调 fn(paused: bool)。"""
        self._on_vla_control_pause_changed = callback

    def set_vla_takeover_callback(self, callback):
        """设置 VLA 请求接管控制权时的回调 fn()，外部应断连其他控制源。"""
        self._on_vla_takeover_requested = callback

    def set_mobile_base_callbacks(self, get_state=None, lift_position=None,
                                  chassis_velocity=None):
        """设置移动底盘回调（底盘+升降，可选；不设置则保持 7 维纯机械臂模式）。

        - get_state() -> (lift_m, vx, vy, omega)：并入
          observation.state / action，维度从 7 扩展为 11：
          [x,y,z,rx,ry,rz,gripper, lift_m, vx, vy, omega]
          （底盘为机体系实测速度 m/s, m/s, rad/s——里程计位姿无 IMU
          漂移大且原点任意，不能作训练标签；升降为绝对位置米）
        - lift_position(mm)：云端推理动作升降维（lift_m → mm）位置伺服目标
        - chassis_velocity(vx, vy, omega)：云端推理动作底盘 3 维
          (m/s, m/s, rad/s) 底盘速度指令
        """
        self._mobile_get_state = get_state
        self._mobile_lift_cmd = lift_position
        self._mobile_chassis_vel = chassis_velocity

    def _get_mobile_state(self):
        """返回 [lift_m, vx, vy, omega]（升降绝对位置米 + 底盘机体系
        实测速度 m/s, m/s, rad/s）；未接底盘时返回 None。"""
        cb = getattr(self, "_mobile_get_state", None)
        if cb is None:
            return None
        try:
            lift_m, vx, vy, omega = cb()
            return np.array([lift_m, vx, vy, omega], dtype=np.float32)
        except Exception as e:
            logger.warning(f"[VLA] 获取底盘状态失败: {e}")
            return np.zeros(4, dtype=np.float32)

    def _pose_dim(self):
        """录制用 state/action 维度：双臂 14（arm1_pose7 + arm2_pose7），
        接入底盘后 +4（lift/vx/vy/omega）= 18"""
        return 14 + (4 if getattr(self, "_mobile_get_state", None) is not None else 0)

    def _zero_pose(self):
        return np.zeros(self._pose_dim(), dtype=np.float32)

    def _dispatch_mobile_action(self, rows):
        """把 action 末尾 4 维分发给升降/底盘：[-4]=lift_m（绝对位置伺服），
        [-3:]=vx,vy,omega（机体系速度指令，与录制标签同源——实测速度经
        lookahead 写入，部署按速度下发，语义闭环）。

        底盘维位置随臂数变化：单臂 11 维在 [7:11]，双臂 18 维在 [14:18]，
        故从末尾倒推，不能硬编码 [7]/[8:11]。
        """
        try:
            dim = int(rows[0].shape[0]) if hasattr(rows[0], "shape") else len(rows[0])
        except Exception:
            return
        # 仅当接入了底盘回调且维度确实是含底盘的合法布局时分发
        # （11=单臂+底盘，18=双臂+底盘；14=双臂无底盘等维度不分发，
        # 防止旧模型输出被错当底盘指令）
        if getattr(self, "_mobile_get_state", None) is None or dim not in (11, 18):
            return
        base = dim - 4
        last = rows[-1]
        lift_cb = getattr(self, "_mobile_lift_cmd", None)
        if lift_cb is not None:
            try:
                lift_cb(float(last[base]) * 1000.0)  # m → mm
            except Exception as e:
                logger.warning(f"[VLA] 升降目标下发失败: {e}")
        vel_cb = getattr(self, "_mobile_chassis_vel", None)
        if vel_cb is not None:
            try:
                vel_cb(float(last[base + 1]), float(last[base + 2]),
                       float(last[base + 3]))
            except Exception as e:
                logger.warning(f"[VLA] 底盘速度下发失败: {e}")


    # ==================== VLA 推理 ====================

    def _browse_vla_model(self):
        path = QFileDialog.getExistingDirectory(self, "选择模型 checkpoint 目录", ".")
        if path:
            self.vla_model_edit.setText(path)
            self._sync_image_size_from_model(path)
            self._refresh_task_candidates()
            self._show_model_type(path)

    def _show_model_type(self, model_path):
        """读取 checkpoint config.json 的策略类型并显示在状态栏，
        让用户在启动推理前确认识别结果（如 ACT / SmolVLA）。"""
        try:
            ckpt = Path(model_path).resolve()
            if (ckpt / "pretrained_model").is_dir():
                ckpt = ckpt / "pretrained_model"
            with open(ckpt / "config.json") as f:
                ptype = str(json.load(f).get("type", "") or "")
            if not ptype:
                return
            if ptype == "act":
                note = "（无 task 条件，力矩并入 state，图像自动缩放到 512x512）"
            else:
                note = ""
            self.status_label.setText(f"[VLA] 已选择模型: {ptype} {note}")
            logger.info(f"[VLA] 选择模型: {ckpt}，策略类型={ptype}")
        except Exception as e:
            logger.warning(f"[VLA] 读取模型类型失败: {e}")

    def _on_vla_task_changed(self, text):
        """推理运行期间实时切换任务指令：直接把新任务文本同步给推理线程，
        无需停止/重启推理；普通模式与 RTC 异步模式均在下一帧/下一 chunk 生效。"""
        task = text.strip() or "default_task"
        if self._vla_running and self._vla_thread is not None:
            try:
                self._vla_thread.set_task(task)
            except Exception as e:
                logger.warning(f"[VLA] 切换任务失败: {e}")

    def _refresh_task_candidates(self):
        """根据所选模型 checkpoint 的 train_config.json 找到训练数据集，
        读取其 meta/tasks.parquet，把任务名自动填充到推理任务下拉框。"""
        model_path = self.vla_model_edit.text().strip()
        if not model_path:
            return
        try:
            ckpt = Path(model_path).resolve()
            if (ckpt / "pretrained_model").is_dir():
                ckpt = ckpt / "pretrained_model"
            cfg_path = ckpt / "train_config.json"
            if not cfg_path.is_file():
                return
            with open(cfg_path) as f:
                train_cfg = json.load(f)
            ds_cfg = train_cfg.get("dataset") or {}
            root, repo_id = ds_cfg.get("root"), ds_cfg.get("repo_id")
            debugger_dir = Path(__file__).resolve().parent.parent
            candidates = []
            if root:
                candidates += [Path(root), debugger_dir / root]
            if repo_id:
                candidates.append(debugger_dir / "lerobot_data" / repo_id)
            tasks_path = next(
                (c / "meta" / "tasks.parquet" for c in candidates
                 if (c / "meta" / "tasks.parquet").is_file()), None)
            if tasks_path is None:
                logger.warning(f"[VLA] 未找到训练数据集 tasks.parquet（root={root}）")
                return
            import pandas as pd
            tasks = [str(t) for t in pd.read_parquet(tasks_path).index.tolist()]
            if not tasks:
                return
            cur = self.vla_task_input.currentText().strip()
            self.vla_task_input.blockSignals(True)
            self.vla_task_input.clear()
            self.vla_task_input.addItems(tasks)
            if cur and cur not in tasks:
                self.vla_task_input.setEditText(cur)
            self.vla_task_input.blockSignals(False)
            logger.info(f"[VLA] 任务候选已从 {tasks_path} 加载: {tasks}")
        except Exception as e:
            logger.warning(f"[VLA] 加载任务候选失败: {e}")

    def _vla_get_obs(self):
        """供推理线程调用：返回 (rgb0, rgb1, state7) 或 None。与录制数据格式一致。"""
        if len(self.camera_indices) < 2:
            return None
        f0 = self.latest_frames.get(self.camera_indices[0])
        f1 = self.latest_frames.get(self.camera_indices[1])
        if f0 is None or f1 is None:
            return None
        rgb0 = self._resize_frame(f0, *self._lerobot_image_size,
                                  crop_side=self._lerobot_crop_side)
        rgb1 = self._resize_frame(f1, *self._lerobot_image_size,
                                  crop_side=self._lerobot_crop_side)
        state = self._get_pose_7d()
        return rgb0, rgb1, state

    def _sync_image_size_from_model(self, model_path):
        """从模型 checkpoint 的 config.json 读取图像输入分辨率，同步到
        _lerobot_image_size 与 UI 下拉框，防止推理分辨率与训练数据不一致
        （如训练 720x720 而推理喂 1280x720 会被 letterbox 成 512x288，分布偏移）。"""
        try:
            ckpt = Path(model_path).resolve()
            if (ckpt / "pretrained_model").is_dir():
                ckpt = ckpt / "pretrained_model"
            cfg_path = ckpt / "config.json"
            if not cfg_path.is_file():
                return
            with open(cfg_path) as f:
                cfg = json.load(f)
            shape = (cfg.get("input_features", {})
                       .get("observation.images.cam1", {}).get("shape"))
            if not shape or len(shape) != 3:
                return
            _, h, w = int(shape[0]), int(shape[1]), int(shape[2])
            if (h, w) == tuple(self._lerobot_image_size):
                return
            self._lerobot_image_size = (h, w)
            # 模型配置不含裁切语义，切换分辨率时退回自适应长宽比裁切
            self._lerobot_crop_side = None
            logger.info(f"[VLA] 从模型配置同步图像分辨率: {w}x{h}")
            for i, (name, ow, oh, *_) in enumerate(self.lerobot_res_options):
                if ow == w and oh == h:
                    self.lerobot_res_combo.blockSignals(True)
                    self.lerobot_res_combo.setCurrentIndex(i)
                    self.lerobot_res_combo.blockSignals(False)
                    break
            else:
                logger.warning(f"[VLA] 模型分辨率 {w}x{h} 不在下拉选项中，仅内部生效")
        except Exception as e:
            logger.warning(f"[VLA] 读取模型图像分辨率失败: {e}")

    def toggle_vla_inference(self):
        if self._vla_running:
            self._stop_vla_inference()
        else:
            self._start_vla_inference()

    def _start_vla_inference(self):
        if self._motion_cb is None:
            QMessageBox.warning(self, "提示", "运动控制未连接（motion callback 未设置）")
            return
        model_path = self.vla_model_edit.text().strip()
        task = self.vla_task_input.currentText().strip() or "default_task"
        use_cloud = self.chk_cloud.isChecked()
        if not use_cloud:
            if not model_path:
                QMessageBox.warning(self, "提示", "请填写模型路径")
                return
            # 推理分辨率必须与训练数据一致：从模型 config.json 自动同步（先于采集分辨率检查）
            self._sync_image_size_from_model(model_path)
        else:
            # 云端模式启动前置：cloud_inference/ 缺失自动拉取 + 无隧道自动建立
            # （失败直接返回，界面字段保持可编辑，便于改密码/地址后重试）
            ok, msg = self._prepare_cloud_connection()
            if not ok:
                QMessageBox.warning(self, "云端推理连接失败", msg)
                self.status_label.setText("[VLA-云] 云端连接失败，详见提示")
                return
        # 保证摄像头采集分辨率不低于训练分辨率（云端模式由握手后
        # _on_cloud_server_info 按服务端模型配置再同步一次）
        self._ensure_capture_resolution()
        self._rtc_mode = self.chk_rtc.isChecked()
        self.chk_rtc.setEnabled(False)  # 运行期间锁定模式，停止后可重新选择
        self.chk_rtc_trained.setEnabled(False)
        self.chk_cloud.setEnabled(False)
        self.cloud_addr_edit.setEnabled(False)
        self.cloud_token_edit.setEnabled(False)
        self.cloud_ssh_edit.setEnabled(False)
        self.cloud_ssh_pass_edit.setEnabled(False)
        self.chk_cloud_save.setEnabled(False)
        if use_cloud:
            self._vla_thread = CloudInferenceThread(
                self.cloud_addr_edit.text().strip(), self.cloud_token_edit.text().strip(),
                task, self._vla_get_obs, fps=self._lerobot_fps,
                get_torque=self._lerobot_get_torque,
                rtc_trained=self.chk_rtc_trained.isChecked(), parent=self)
            self._vla_thread.link_lost.connect(self._on_cloud_link_lost)
            self._vla_thread.link_restored.connect(self._on_cloud_link_restored)
            self._vla_thread.server_info.connect(self._on_cloud_server_info)
        else:
            self._vla_thread = VLAInferenceThread(
                model_path, task, self._vla_get_obs, fps=self._lerobot_fps,
                rtc_mode=self._rtc_mode, get_torque=self._lerobot_get_torque,
                rtc_trained=self.chk_rtc_trained.isChecked(), parent=self)
        self._vla_thread.avg_n = self._vla_avg_n  # 同步界面预设的采样平均次数
        # 优势条件化前缀（部署恒 positive；未勾选=空前缀，行为不变）
        self._vla_thread.advantage_prefix = (
            "advantage: positive, " if self.chk_advantage_condition.isChecked() else "")
        if self._rtc_mode:
            self._vla_thread.chunk_ready.connect(self._on_rtc_chunk)
        else:
            self._vla_thread.chunk_ready.connect(self._on_seq_chunk)
        self._vla_thread.single_chunk_ready.connect(self._on_single_chunk)
        self._vla_thread.status.connect(lambda s: self.status_label.setText(f"[VLA] {s}"))
        self._vla_thread.error.connect(self._on_vla_error)
        self._vla_thread.finished.connect(self._on_vla_finished)
        self._vla_running = True
        self.btn_vla_once.setEnabled(True)
        self.btn_vla.setText("停止推理")
        self.btn_vla.setStyleSheet(
            "font-size: 12px; padding: 4px 12px; background-color: #ff4444; color: white;"
        )
        self._vla_control_paused = False
        self.btn_vla_pause.setText("暂停控制")
        self.btn_vla_pause.setStyleSheet(
            "font-size: 12px; padding: 4px 12px; background-color: #FF9800; color: white;"
        )
        self.btn_vla_pause.setEnabled(True)
        self.status_label.setText(
            "[VLA-云] 正在连接云端推理服务..." if use_cloud else "[VLA] 正在加载模型...")
        self._vla_thread.start()

        # VLA 启动即请求独占控制权：若手柄仍在连接/控臂，自动断开其 100Hz
        # 保位控制循环。否则手柄与 VLA 的动作流互相覆盖，机械臂保持不动
        # （症状：连续推理不下发可见运动，需反复点“继续控制”断开手柄后才动）
        if self._on_vla_takeover_requested is not None:
            try:
                self._on_vla_takeover_requested()
            except Exception as e:
                logger.warning(f"[VLA] 启动时请求接管控制权失败: {e}")

        # 低通 EMA 基准复位（普通/RTC 模式共用）：从当前位姿起步，避免首帧跳变
        self._rtc_last_sent = None
        self._rtc_last_dir = None
        self._rtc_last_rot_dir = None
        self._rtc_proj_streak = 0
        # 夹爪状态复位（直控模式仅记录值）
        self._grip_last_sent = None

        if self._rtc_mode:
            # RTC（管道平滑块执行）：立即请求首个 chunk；之后每段下发时按
            # 实测推理延迟提前调度下一段推理，保证段间无缝接续
            self._rtc_inflight = True
            self._rtc_chunk_count = 0
            self._rtc_last_sent = None
            self._rtc_last_dir = None
            self._rtc_last_rot_dir = None
            self._rtc_proj_streak = 0
            self._rtc_exec_offset = 0
            self._rtc_exec_t0 = 0.0
            self._rtc_last_latency = 0.0
            self._seq_grip_values.clear()
            self._grip_target = None
            self._grip_close_locked = False
            self._grip_open_locked = False
            if self._rtc_next_timer is None:
                self._rtc_next_timer = QTimer(self)
                self._rtc_next_timer.setSingleShot(True)
                self._rtc_next_timer.setTimerType(Qt.TimerType.PreciseTimer)
                self._rtc_next_timer.timeout.connect(self._request_next_rtc_chunk)
            if self._seq_grip_timer is None:
                self._seq_grip_timer = QTimer(self)
                self._seq_grip_timer.setTimerType(Qt.TimerType.PreciseTimer)
                self._seq_grip_timer.timeout.connect(self._on_seq_grip_tick)
            if self._rtc_grip_timer is None:
                self._rtc_grip_timer = QTimer(self)
                self._rtc_grip_timer.setTimerType(Qt.TimerType.PreciseTimer)
                self._rtc_grip_timer.timeout.connect(self._on_rtc_grip_tick)
            self._vla_thread.request_chunk()
            logger.info("[VLA-RTC] 管道平滑块执行模式已启动（段间插值拼接）")
        else:
            # 普通模式（顺序块执行）：创建执行定时器并请求首个 chunk。
            # 收到整段 chunk 后一次性平滑执行完（MoveJ/S 曲线插值到末端位姿），
            # 执行窗口结束后再请求下一段
            self._seq_inflight = True
            self._seq_chunk_count = 0
            self._seq_grip_values.clear()
            if self._seq_timer is None:
                self._seq_timer = QTimer(self)
                self._seq_timer.setSingleShot(True)
                self._seq_timer.setTimerType(Qt.TimerType.PreciseTimer)
                self._seq_timer.timeout.connect(self._request_next_seq_chunk)
            if self._seq_grip_timer is None:
                self._seq_grip_timer = QTimer(self)
                self._seq_grip_timer.setTimerType(Qt.TimerType.PreciseTimer)
                self._seq_grip_timer.timeout.connect(self._on_seq_grip_tick)
            self._vla_thread.request_chunk()
            logger.info("[VLA] 顺序块执行模式已启动（整段 chunk 平滑执行）")

    def _on_avg3_toggled(self, on):
        """3次采样平均开关：实时修改运行中推理线程的采样次数。"""
        self._vla_avg_n = 3 if on else 1
        if self._vla_thread is not None:
            self._vla_thread.avg_n = self._vla_avg_n
        self.status_label.setText(
            f"[VLA] 多次采样平均: {'开启 (3次)' if on else '关闭'}")

    def _on_advantage_condition_toggled(self, on):
        """优势条件化开关：实时设置推理线程的任务前缀（模型未条件化训练时勿开）。"""
        prefix = "advantage: positive, " if on else ""
        if self._vla_thread is not None:
            self._vla_thread.advantage_prefix = prefix
        self.status_label.setText(
            f"[VLA] 优势条件化: {'开启 (advantage: positive)' if on else '关闭'}")

    # ==================== 实时价值判断（RECAP §4.7 部署侧） ====================

    def _on_value_judge_toggled(self, on):
        """价值判断开关：开启时后台加载价值头（siglip 后端 ~10-20s），完成后启动 2Hz 判断。"""
        if not on:
            if self._value_timer is not None:
                self._value_timer.stop()
            self._value_history.clear()
            self._value_last_sign = 0
            self.value_status_label.setText("V: —")
            self.value_status_label.setStyleSheet("font-size: 11px; color: #888;")
            return
        model_dir = self.value_model_edit.text().strip()
        if not model_dir:
            self.chk_value_judge.setChecked(False)
            QMessageBox.warning(self, "价值判断", "请填写价值模型目录")
            return
        if self._value_judge is None:
            if self._value_load_thread is not None and self._value_load_thread.isRunning():
                return  # 正在加载，避免重复
            self._value_judge = ValueJudge(model_dir)
            self._value_load_thread = ValueJudgeLoadThread(self._value_judge, parent=self)
            self._value_load_thread.loaded.connect(self._on_value_judge_loaded)
            self._value_load_thread.failed.connect(self._on_value_judge_error)
            self.value_status_label.setText("V: 加载中...")
            self._value_load_thread.start()
        else:
            self._on_value_judge_loaded()  # 已加载过（如换目录前），直接启动

    def _on_value_judge_loaded(self):
        j = self._value_judge
        self.camera_log.emit(
            f"[价值判断] 模型就绪: backend={j.backend}, denominator={j.denominator:.0f}, "
            f"fps={j.fps}（{j.model_dir}）")
        self._value_history.clear()
        self._value_tick_count = 0
        if self._value_timer is None:
            self._value_timer = QTimer(self)
            self._value_timer.timeout.connect(self._on_value_tick)
        self._value_timer.start(500)  # 2Hz
        self.value_status_label.setText("V: 等待推理...")

    def _on_value_judge_error(self, msg):
        logger.error(f"[ValueJudge] {msg}")
        self.value_status_label.setText("V: 加载失败")
        self.chk_value_judge.setChecked(False)
        QMessageBox.warning(self, "价值判断加载失败",
                            f"{msg}\n\n请先运行 recap_value.py train 生成价值头。")
        if self._value_judge is not None:
            self._value_judge.release()
            self._value_judge = None

    def _on_value_tick(self):
        """2Hz 判断：V(s) + 回顾式 TD 优势。仅 VLA 推理运行中生效。"""
        j = self._value_judge
        if j is None:
            return
        if not self._vla_running:
            self.value_status_label.setText("V: 等待推理...")
            self.value_status_label.setStyleSheet("font-size: 11px; color: #888;")
            self._value_history.clear()
            self._value_last_sign = 0
            return
        if len(self.camera_indices) < 2:
            return
        f0 = self.latest_frames.get(self.camera_indices[0])
        f1 = self.latest_frames.get(self.camera_indices[1])
        if f0 is None or f1 is None:
            return
        try:
            v = j.judge(f0, f1, self._get_pose_7d())
        except Exception as e:
            logger.warning(f"[ValueJudge] 判断失败: {e}")
            return

        now = time.monotonic()
        self._value_history.append((now, v))
        # 回顾窗口：找 ~window_sec 前的 V 作基准（不足窗口则跳过优势显示）
        ref = None
        for t_ref, v_ref in self._value_history:
            if now - t_ref >= self._value_window_sec:
                ref = (t_ref, v_ref)
                break
        while self._value_history and now - self._value_history[0][0] > 2 * self._value_window_sec:
            self._value_history.popleft()

        # 预计剩余时间：V = -剩余代价/denominator → 剩余秒 ≈ -V×denominator/fps
        remain_sec = max(0.0, -v * j.denominator / max(j.fps, 1))

        adv, sign = None, self._value_last_sign
        if ref is not None:
            elapsed_frames = (now - ref[0]) * j.fps
            cost = -elapsed_frames / j.denominator
            adv = cost + (v - ref[1])
            # 死区 = 窗口代价的一半（V 噪声低于该量级时判中性，防符号闪烁）
            dead = 0.5 * abs(cost)
            sign = 1 if adv > dead else (-1 if adv < -dead else 0)

        # 显示：V=… 剩~Ns ▲+0.012 正向（绿/红/灰）
        if adv is None:
            txt, color = f"V={v:+.3f} 剩~{remain_sec:.0f}s（建窗中）", "#888"
        elif sign > 0:
            txt, color = f"V={v:+.3f} 剩~{remain_sec:.0f}s ▲{adv:+.3f} 正向", "#2E7D32"
        elif sign < 0:
            txt, color = f"V={v:+.3f} 剩~{remain_sec:.0f}s ▼{adv:+.3f} 负向", "#C62828"
        else:
            txt, color = f"V={v:+.3f} 剩~{remain_sec:.0f}s ◆{adv:+.3f} 中性", "#888"
        self.value_status_label.setText(txt)
        self.value_status_label.setStyleSheet(f"font-size: 11px; color: {color};")

        # 日志：符号翻转即记；同号每 20 tick（10s）节流一次
        self._value_tick_count += 1
        if sign != self._value_last_sign or self._value_tick_count % 20 == 0:
            sign_txt = {1: "正向", -1: "负向", 0: "中性"}[sign]
            self.camera_log.emit(
                f"[价值判断] V={v:+.3f} 预计剩~{remain_sec:.0f}s A={adv if adv is not None else float('nan'):+.4f} → {sign_txt}")
        self._value_last_sign = sign

    def _on_lowpass_alpha_changed(self):
        """低通滤波阈值 α 文本框提交（回车/失焦）：钳位到 [0.05, 1.0]。

        解析失败或越界不报错——回写当前有效值，推理循环下次取 chunk 端点
        时即用新 α（无锁读取，主线程独占写、推理下发同样在主线程定时器，
        不存在竞态）。
        """
        try:
            v = float(self.edit_lowpass_alpha.text().strip())
        except ValueError:
            self.edit_lowpass_alpha.setText(f"{self._vla_lowpass_alpha:.2f}")
            return
        v = min(1.0, max(0.05, v))
        self._vla_lowpass_alpha = v
        self.edit_lowpass_alpha.setText(f"{v:.2f}")  # 归一化显示（含钳位回写）
        self.status_label.setText(
            f"[VLA] 低通滤波 α={v:.2f}（{'已开启' if self._vla_lowpass_enabled else '已关闭'}）"
            f"{'，滤波增强、响应变慢' if v < 0.8 else ''}")

    def toggle_vla_control_pause(self):
        """切换 VLA 控制暂停/继续。

        暂停时：模型线程继续运行，但推理结果不再下发给机械臂运动控制程序，
        这样手柄可以接入控制。继续后：VLA 重新接管机械臂运动控制。
        """
        # 如果当前准备恢复 VLA 控制，先请求外部控制源释放（如手柄）
        if self._vla_control_paused and self._on_vla_takeover_requested is not None:
            try:
                self._on_vla_takeover_requested()
            except Exception as e:
                logger.warning(f"[VLA] 请求接管控制权回调失败: {e}")

        self._vla_control_paused = not self._vla_control_paused
        # 推理录制接管自动标注（RECAP P2，无确认制；未在推理录制时内部自动跳过）
        self._rollout_on_control_boundary(self._vla_control_paused)
        if self._vla_control_paused:
            self.btn_vla_pause.setText("继续控制")
            self.btn_vla_pause.setStyleSheet(
                "font-size: 12px; padding: 4px 12px; background-color: #4CAF50; color: white;"
            )
            self.status_label.setText("[VLA] 控制已暂停（手柄可接管）")
            logger.info("[VLA] 控制已暂停，模型仍在运行但不向机械臂下发动作")
            # 顺序块/管道块执行：暂停调度与夹爪滴灌（已下发的轨迹会走完，≤0.6s）
            if self._seq_timer is not None:
                self._seq_timer.stop()
            if self._rtc_next_timer is not None:
                self._rtc_next_timer.stop()
            if self._rtc_grip_timer is not None:
                self._rtc_grip_timer.stop()
            if self._seq_grip_timer is not None:
                self._seq_grip_timer.stop()
        else:
            self.btn_vla_pause.setText("暂停控制")
            self.btn_vla_pause.setStyleSheet(
                "font-size: 12px; padding: 4px 12px; background-color: #FF9800; color: white;"
            )
            self.status_label.setText("[VLA] 控制已继续，VLA 接管机械臂")
            logger.info("[VLA] 控制继续，VLA 重新接管机械臂")
            # 暂停期间机械臂可能被手柄移动，重置 RTC 平滑基准为当前位姿，
            # 避免恢复瞬间从旧目标位置回拉
            self._rtc_last_sent = None
            self._rtc_last_dir = None
            self._rtc_last_rot_dir = None
            self._rtc_proj_streak = 0
            # trained RTC：上一段剩余动作已失效，恢复后首个 chunk 不带 prefix 条件
            self._rtc_exec_t0 = 0.0
            if self._vla_thread is not None:
                self._vla_thread._last_raw_chunk = None
                self._vla_thread._pending_rtc = None
            # 块执行链恢复：基于最新观测立即请求新 chunk
            if not self._rtc_mode:
                self._request_next_seq_chunk()
            else:
                self._request_next_rtc_chunk()
        if self._on_vla_control_pause_changed is not None:
            try:
                self._on_vla_control_pause_changed(self._vla_control_paused)
            except Exception as e:
                logger.warning(f"[VLA] 控制暂停状态回调失败: {e}")

    def pause_vla_control(self):
        """外部请求暂停 VLA 控制（如手柄连接接管）。仅在推理运行且未暂停时生效。"""
        if self._vla_running and not self._vla_control_paused:
            self.toggle_vla_control_pause()

    def resume_vla_control(self):
        """外部请求恢复 VLA 控制（如手柄断开释放）。仅在推理运行且已暂停时生效。"""
        if self._vla_running and self._vla_control_paused:
            self.toggle_vla_control_pause()

    def _stop_vla_inference(self):
        # 推理录制联动：推理停止时自动结束录制（保存 + 弹窗评价）
        if self._rollout_recording:
            self._stop_rollout_record()
        # 立即标记停止，块执行链不再下发新动作
        self._vla_running = False
        self.btn_vla.setEnabled(False)
        self.btn_vla_once.setEnabled(False)
        self.status_label.setText("[VLA] 正在停止...")
        self._stop_rtc_executor()
        if self._seq_timer is not None:
            self._seq_timer.stop()
        if self._seq_grip_timer is not None:
            self._seq_grip_timer.stop()
        if self._rtc_grip_timer is not None:
            self._rtc_grip_timer.stop()
        self._grip_target = None
        self._seq_inflight = False
        self._seq_grip_values.clear()
        if self._single_timer is not None:
            self._single_timer.stop()
        self._single_active = False
        if self._motion_cb is not None:
            try:
                self._motion_cb("cancel_motion")
            except Exception:
                pass
        if self._vla_thread is not None:
            self._vla_thread.stop()

    def _stop_rtc_executor(self):
        """停止 RTC 管道调度定时器并复位状态"""
        if self._rtc_next_timer is not None:
            self._rtc_next_timer.stop()
        self._rtc_inflight = False

    def _on_vla_finished(self):
        self._vla_running = False
        self._stop_rtc_executor()
        self.chk_rtc.setEnabled(True)
        self.chk_rtc_trained.setEnabled(True)
        self.chk_cloud.setEnabled(True)
        self.cloud_addr_edit.setEnabled(True)
        self.cloud_token_edit.setEnabled(True)
        self.cloud_ssh_edit.setEnabled(True)
        self.cloud_ssh_pass_edit.setEnabled(True)
        self.chk_cloud_save.setEnabled(True)
        if self._vla_thread is not None:
            self._vla_thread.deleteLater()
            self._vla_thread = None
        self._vla_control_paused = False
        self.btn_vla_pause.setText("暂停控制")
        self.btn_vla_pause.setStyleSheet(
            "font-size: 12px; padding: 4px 12px; background-color: #FF9800; color: white;"
        )
        self.btn_vla_pause.setEnabled(False)
        self.btn_vla.setText("开始推理")
        self.btn_vla.setStyleSheet(
            "font-size: 12px; padding: 4px 12px; background-color: #9C27B0; color: white;"
        )
        self.btn_vla.setEnabled(True)

    def _on_vla_error(self, tb):
        logger.error(f"[VLA] 推理出错:\n{tb}")
        self.status_label.setText("[VLA] 推理出错，详见日志")

    # ==================== 云端推理：链路安全处理 ====================

    def _on_cloud_server_info(self, info):
        """云端握手成功：按服务端模型配置同步图像分辨率，防止输入分布偏移。"""
        w, h = int(info.get("image_width", 0)), int(info.get("image_height", 0))
        if w > 0 and h > 0 and (h, w) != tuple(self._lerobot_image_size):
            self._lerobot_image_size = (h, w)
            # 云端模型配置不含裁切语义，退回自适应长宽比裁切
            self._lerobot_crop_side = None
            logger.info(f"[VLA-云] 从云端模型配置同步图像分辨率: {w}x{h}")
            self._ensure_capture_resolution()
        logger.info(f"[VLA-云] 服务端信息: {info}")

    def _on_cloud_link_lost(self, msg):
        """云端链路中断/推理超时 → 立即安全停（保位），恢复需人工确认。

        动作：cancel_motion 停臂 → 复位 inflight 标志（防止恢复后死锁）→
        进入"暂停控制"（停止块调度与夹爪斜坡定时器，机械臂保位）。
        后台心跳自动重连，重连成功后由人工点"继续控制"恢复运动。
        """
        if not self._vla_running:
            return
        logger.warning(f"[VLA-云] 链路中断，执行安全停（保位）: {msg}")
        self._rtc_inflight = False
        self._seq_inflight = False
        if self._motion_cb is not None:
            try:
                self._motion_cb("cancel_motion")
            except Exception:
                pass
        if not self._vla_control_paused:
            self.toggle_vla_control_pause()
        self.status_label.setText(
            f"[VLA-云] 链路中断已安全停（保位）: {msg}；重连中，恢复后请人工点'继续控制'")

    def _on_cloud_link_restored(self):
        """云端链路重连成功：仅提示，不自动恢复运动（需人工确认）。"""
        logger.info("[VLA-云] 链路已恢复，等待人工确认后继续控制")
        self.status_label.setText("[VLA-云] 云端链路已恢复，请确认安全后点'继续控制'")

    # ==================== 云端推理：SSH 隧道自动托管 + 配置持久化 ====================
    # 界面填 SSH 地址+密码后，"开始推理"自动完成：cloud_inference/ 缺失则 rsync
    # 拉取 → 本地无隧道则 sshpass+autossh/ssh 建立 → 保存配置到 cloud_settings.json。
    # 多机器人部署：拷贝 debugger 文件夹 → 界面填一次服务器信息（勾"记住"）即可。
    CLOUD_REMOTE_DIR = "/root/autodl-tmp/debugger/cloud_inference"  # 云端代码目录

    def _cloud_settings_path(self):
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "cloud_settings.json")

    def _load_cloud_settings(self):
        """启动时回填上次保存的云端连接配置（文件不存在/损坏则忽略）。"""
        try:
            with open(self._cloud_settings_path()) as f:
                d = json.load(f)
        except Exception:
            return
        self.cloud_ssh_edit.setText(str(d.get("ssh", "") or ""))
        if d.get("grpc_addr"):
            self.cloud_addr_edit.setText(str(d["grpc_addr"]))
        if d.get("token"):
            self.cloud_token_edit.setText(str(d["token"]))
        if d.get("save_pwd"):
            self.cloud_ssh_pass_edit.setText(str(d.get("password", "") or ""))
            self.chk_cloud_save.setChecked(True)

    def _save_cloud_settings(self):
        """保存云端连接配置（密码仅在勾'记住'时落盘；文件权限 600）。"""
        d = {
            "ssh": self.cloud_ssh_edit.text().strip(),
            "grpc_addr": self.cloud_addr_edit.text().strip(),
            "token": self.cloud_token_edit.text().strip(),
            "save_pwd": self.chk_cloud_save.isChecked(),
            "password": (self.cloud_ssh_pass_edit.text()
                         if self.chk_cloud_save.isChecked() else ""),
        }
        try:
            p = self._cloud_settings_path()
            with open(p, "w") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            os.chmod(p, 0o600)
        except Exception as e:
            logger.warning(f"[VLA-云] 保存云端配置失败: {e}")

    @staticmethod
    def _parse_ssh_target(text):
        """解析 'user@host:port'（user 默认 root，port 默认 22）。
        返回 dict / None（未填写） / 'invalid'（格式错误）。"""
        text = (text or "").strip()
        if not text:
            return None
        user = "root"
        if "@" in text:
            user, text = text.split("@", 1)
            user = user.strip() or "root"
        port = 22
        if ":" in text:
            text, p = text.rsplit(":", 1)
            try:
                port = int(p.strip())
            except ValueError:
                return "invalid"
        host = text.strip()
        if not host:
            return "invalid"
        return {"user": user, "host": host, "port": port}

    def _grpc_host_port(self):
        addr = (self.cloud_addr_edit.text().strip()
                or CloudInferenceThread.DEFAULT_ADDR)
        host, _, p = addr.rpartition(":")
        try:
            port = int(p)
        except ValueError:
            host, port = addr, 50051
        return (host.strip() or "127.0.0.1"), port

    @staticmethod
    def _port_open(host, port, timeout=1.0):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _ensure_cloud_files(self, ssh):
        """cloud_inference/vla_inference_pb2.py 缺失时按 SSH 信息自动 rsync 拉取。"""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_dir = os.path.join(base, "cloud_inference")
        pb2 = os.path.join(local_dir, "vla_inference_pb2.py")
        if os.path.isfile(pb2):
            return True, ""
        if ssh is None or ssh == "invalid":
            return False, (
                "本地缺少 cloud_inference/vla_inference_pb2.py，且未填写 SSH 地址，"
                "无法自动拉取。\n请填写云服务器 SSH 地址+密码后重试。")
        if shutil.which("rsync") is None or shutil.which("sshpass") is None:
            return False, "自动拉取需要 rsync 与 sshpass：sudo apt install -y rsync sshpass"
        env = dict(os.environ)
        env["SSHPASS"] = self.cloud_ssh_pass_edit.text()
        cmd = ["sshpass", "-e", "rsync", "-az", "-e",
               f"ssh -p {ssh['port']} -o ConnectTimeout=15 "
               f"-o StrictHostKeyChecking=accept-new",
               f"{ssh['user']}@{ssh['host']}:{self.CLOUD_REMOTE_DIR}/",
               local_dir + "/", "--exclude", "__pycache__", "--exclude", "*.log"]
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
        except Exception as e:
            return False, f"从服务器拉取 cloud_inference/ 失败: {e}"
        if r.returncode != 0 or not os.path.isfile(pb2):
            tail = (r.stderr or r.stdout or "").strip()[-400:]
            return False, (f"从服务器拉取 cloud_inference/ 失败：\n{tail}\n"
                           f"请检查 SSH 地址/密码是否正确。")
        logger.info(f"[VLA-云] cloud_inference/ 已从 {ssh['host']} 自动同步")
        return True, ""

    def _kill_cloud_tunnel(self):
        """终止本程序拉起的隧道进程组（外部建立的隧道不受影响）。"""
        if self._tunnel_proc is None:
            return
        try:
            if self._tunnel_proc.poll() is None:
                os.killpg(os.getpgid(self._tunnel_proc.pid), signal.SIGTERM)
                try:
                    self._tunnel_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self._tunnel_proc.pid), signal.SIGKILL)
        except Exception:
            pass
        self._tunnel_proc = None

    def _ensure_cloud_tunnel(self, ssh):
        """确保 gRPC 地址可达：公网直连/已有隧道直接通过；否则按 SSH 信息自动建隧道。"""
        host, port = self._grpc_host_port()
        if host not in ("127.0.0.1", "localhost", "::1"):
            return True, ""  # 公网直连，无需隧道
        if self._port_open(host, port):
            return True, ""  # 已有隧道（外部建立或本程序上次拉起）在监听
        if ssh == "invalid":
            return False, "SSH 地址格式应为 user@主机:端口（如 root@1.2.3.4:43394）"
        if ssh is None:
            return False, (
                f"本地 {host}:{port} 无隧道监听，且未填写 SSH 地址。\n"
                f"请填写云服务器 SSH 地址+密码（自动建立隧道），"
                f"或手动执行 autossh -M 0 -N -L {port}:localhost:{port} ...")
        if shutil.which("sshpass") is None:
            return False, "自动建隧道需要 sshpass：sudo apt install -y sshpass"
        target = f"{ssh['user']}@{ssh['host']}:{ssh['port']}"
        alive_same = (self._tunnel_proc is not None
                      and self._tunnel_proc.poll() is None
                      and self._tunnel_target == target)
        if not alive_same:
            self._kill_cloud_tunnel()  # 目标变了或旧进程已死，先清理
            use_autossh = shutil.which("autossh") is not None
            fwd = f"{port}:localhost:{port}"
            ssh_opts = (f"-p {ssh['port']} -o StrictHostKeyChecking=accept-new "
                        f"-o ServerAliveInterval=30 -o ServerAliveCountMax=3 "
                        f"-o ExitOnForwardFailure=yes").split()
            tool = "autossh" if use_autossh else "ssh"
            cmd = (["sshpass", "-e", tool] +
                   (["-M", "0"] if use_autossh else []) +
                   ["-N", "-L", fwd] + ssh_opts +
                   [f"{ssh['user']}@{ssh['host']}"])
            env = dict(os.environ)
            env["SSHPASS"] = self.cloud_ssh_pass_edit.text()
            log_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "cloud_tunnel.log")
            env["AUTOSSH_LOGFILE"] = log_path
            env["AUTOSSH_LOGLEVEL"] = "7"
            try:
                logf = open(log_path, "ab")
                self._tunnel_proc = subprocess.Popen(
                    cmd, env=env, stdin=subprocess.DEVNULL,
                    stdout=logf, stderr=logf, start_new_session=True)
                logf.close()  # 子进程已持有 fd，父进程句柄可关
                self._tunnel_target = target
            except Exception as e:
                return False, f"启动隧道进程失败: {e}"
        # 等待端口就绪（首次连接含主机密钥接受+密码认证，通常 <8s）
        for _ in range(30):
            if self._port_open(host, port):
                logger.info(f"[VLA-云] SSH 隧道已就绪: {target} -> 127.0.0.1:{port}")
                return True, ""
            if self._tunnel_proc is not None and self._tunnel_proc.poll() is not None:
                break
            time.sleep(0.5)
        return False, (f"SSH 隧道建立失败（{target}）。\n"
                       f"请检查：SSH 地址/密码是否正确、云端 50051 服务是否已启动。\n"
                       f"日志见 debugger/cloud_tunnel.log")

    def _prepare_cloud_connection(self):
        """云端模式启动前置：文件检查 → 隧道建立 → 保存配置。返回 (成功与否, 错误说明)。"""
        self.status_label.setText("[VLA-云] 正在准备云端连接（文件检查/隧道）...")
        QApplication.processEvents()  # 隧道建立需数秒，先让状态可见
        ssh = self._parse_ssh_target(self.cloud_ssh_edit.text())
        ok, msg = self._ensure_cloud_files(ssh)
        if not ok:
            return False, msg
        ok, msg = self._ensure_cloud_tunnel(ssh)
        if not ok:
            return False, msg
        self._save_cloud_settings()
        return True, ""

    # ==================== 普通模式：顺序块执行（平滑，类 MoveJ 插值） ====================

    def _clamp_pose_jump(self, x, y, z, rx, ry, rz, max_jump, max_rot):
        """以当前位姿为基准，限制单段运动的最大位置/姿态变化（安全限幅）。

        返回 (x,y,z,rx,ry,rz)——若目标跳变超过限制，沿当前位姿方向等比缩放。
        """
        cur = self._get_pose_7d()
        d = np.array([x - cur[0], y - cur[1], z - cur[2]])
        n = float(np.linalg.norm(d))
        if n > max_jump:
            s = max_jump / n
            x, y, z = cur[0] + d[0] * s, cur[1] + d[1] * s, cur[2] + d[2] * s
        dr = np.array([rx - cur[3], ry - cur[4], rz - cur[5]])
        rn = float(np.linalg.norm(dr))
        if rn > max_rot:
            s = max_rot / rn
            rx, ry, rz = cur[3] + dr[0] * s, cur[4] + dr[1] * s, cur[5] + dr[2] * s
        # cur 来自 _get_pose_7d()（np.float32），限幅分支算术会污染出 np.float32，
        # 必须转回 python float，否则 EndPoseCtrl -> rpyToMatrix 的 C++ 绑定拒绝
        return (float(x), float(y), float(z), float(rx), float(ry), float(rz))

    def _on_seq_chunk(self, chunk, latency):
        """普通模式（顺序块执行）：收到一整段 chunk 后一次性平滑执行。

        与“推理一次”同一控制逻辑：取 chunk 末端位姿作为目标，通过
        end_pose_ctrl(servo=False) 让 SDK 做 MoveJ/S 曲线插值一次到位；
        夹爪按 chunk 内序列以 20Hz 滴灌跟随；执行窗口（chunk 时长）结束后
        再请求下一段推理。相比 20Hz 逐点伺服滴灌，无 chunk 边界猛冲/抖动。
        """
        self._seq_inflight = False
        if not self._vla_running or self._motion_cb is None:
            return
        if self._vla_control_paused or self._single_active:
            return
        try:
            rows = [np.asarray(r, dtype=np.float64) for r in chunk]
        except Exception:
            return
        if not rows:
            return

        # 每段执行窗口最长 0.5s（fps//2 步）：截断过长的 chunk（如 SmolVLA 50 步）
        max_steps = max(1, int(self._lerobot_fps) // 2)
        rows = rows[:max_steps]
        fps = max(self._lerobot_fps, 1)

        x, y, z, rx, ry, rz, _g = [float(v) for v in rows[-1][:7]]
        x, y, z, rx, ry, rz = self._clamp_pose_jump(
            x, y, z, rx, ry, rz, self._seq_max_jump, self._seq_max_rot)

        # 主运动：servo=False → SDK MoveJ，S 曲线插值平滑运动到 chunk 末端位姿；
        # 整段只发一次，不会像 20Hz 滴灌那样被新轨迹打断保位段
        self._motion_cb("end_pose_ctrl", x, y, z, rx, ry, rz,
                        len(rows) / fps, False, False, self._rtc_interp)
        # 夹爪按 chunk 内序列 20Hz 滴灌跟随（夹爪命令与臂部轨迹互不影响）
        self._seq_grip_values = deque(float(r[6]) for r in rows)
        if self._seq_grip_timer is not None:
            self._seq_grip_timer.start(int(1000 / fps))
        # 底盘/升降动作分量（action 第 8~11 维，未接底盘时为空操作）
        self._dispatch_mobile_action(rows)
        # 执行窗口结束 → 请求下一段推理
        chunk_ms = int(1000.0 * len(rows) / fps)
        if self._seq_timer is not None:
            self._seq_timer.start(chunk_ms)

        self._seq_chunk_count += 1
        self.status_label.setText(
            f"[VLA] 顺序块执行 #{self._seq_chunk_count}：{len(rows)} 步/"
            f"{chunk_ms}ms，推理 {latency * 1000:.0f}ms")
        logger.info(
            f"[VLA] 顺序块 #{self._seq_chunk_count}: steps={len(rows)}, "
            f"window={chunk_ms}ms, latency={latency * 1000:.0f}ms, "
            f"target=({x:.3f},{y:.3f},{z:.3f})")

    def _request_next_seq_chunk(self):
        """顺序块执行：执行窗口结束（或恢复控制）后请求下一段 chunk 推理。"""
        if not self._vla_running or self._vla_thread is None:
            return
        if self._vla_control_paused or self._single_active or self._seq_inflight:
            return
        self._seq_inflight = True
        self._vla_thread.request_chunk()

    def _on_seq_grip_tick(self):
        """夹爪滴灌（顺序块/RTC/推理一次共用）：按当前 chunk 的序列 20Hz
        逐步直发模型目标值——不滤波、无死区、无换向驻留、无力矩闭锁，
        回到最初的直接控制模式。

        角度约定（数据实测）：角度增大=闭合，减小=张开。
        仅保留物理量程钳位 [-π/2, π/60]。
        """
        if not self._seq_grip_values:
            if self._seq_grip_timer is not None:
                self._seq_grip_timer.stop()
            return
        g = float(np.clip(self._seq_grip_values.popleft(),
                          -math.pi / 2.0, math.pi / 60.0))
        if self._motion_cb is None:
            return
        self._grip_last_sent = g
        self._motion_cb("gripper_ctrl", g)

    def _read_grip_torque(self):
        """读取当前夹爪力矩 (Nm)；回调不可用返回 None。"""
        if self._lerobot_get_torque is None:
            return None
        try:
            return self._lerobot_get_torque()
        except Exception:
            return None

    def _on_rtc_grip_tick(self):
        """RTC 夹爪斜坡定时器（20Hz）：从当前值向目标限速推进（学习手柄
        xbox_control 的 D-pad 夹爪策略）。

        - 终点覆盖：目标始终是最新 chunk 窗口末值，从当前位置连续斜坡过去，
          不重放序列——新段起点再"过期"也不会回拉（往复根源消除）；
        - 限速：闭合 1.0 rad/s / 张开 1.5 rad/s（每 tick 步长 = 速度 × dt），
          闭合过快会造成力矩瞬间冲到 1Nm+（远超 0.5Nm 限幅），位置模式
          大幅跳变还会引发推力超调；
        - 力矩保护（与手柄一致）：闭合方向 soft 预减速（力矩从软阈值到上限
          线性衰减推进量）→ ≥上限锁定保位（目标钳到当前值，不再推进）→
          新段目标出现反向意图时解锁；张开方向对称。
        """
        if self._grip_target is None or self._motion_cb is None:
            if self._rtc_grip_timer is not None:
                self._rtc_grip_timer.stop()
            return
        if self._grip_last_sent is None:
            # 首个目标：以模型目标为起点（无历史基准，直接接管）
            self._grip_last_sent = self._grip_target
            self._motion_cb("gripper_ctrl", self._grip_last_sent)
            return

        cur = self._grip_last_sent
        tgt = self._grip_target
        dt = 1.0 / max(self._lerobot_fps, 1)
        delta = tgt - cur
        if abs(delta) < 1e-4:
            if self._rtc_grip_timer is not None:
                self._rtc_grip_timer.stop()
            return
        closing = delta > 0  # 角度增大 = 闭合
        speed = self._grip_close_speed if closing else self._grip_open_speed
        step = speed * dt

        # 力矩保护：预减速 + 触限锁定保位（策略同手柄 D-pad 夹爪）
        tq = self._read_grip_torque()
        effective = step
        if closing:
            soft = self._grip_tq_max * self._grip_tq_soft_ratio
            if tq is not None:
                if tq >= self._grip_tq_max or self._grip_close_locked:
                    if not self._grip_close_locked:
                        logger.warning(f"[VLA] RTC 夹爪闭合力矩 {tq:.2f}Nm ≥ "
                                       f"{self._grip_tq_max}Nm，锁定保位")
                    self._grip_close_locked = True
                    return  # 保位：不推进
                if tq > soft and self._grip_tq_max > soft:
                    k = (self._grip_tq_max - tq) / (self._grip_tq_max - soft)
                    effective = step * max(0.0, min(1.0, k))
        else:
            soft = self._grip_tq_min * self._grip_tq_soft_ratio
            if tq is not None:
                if tq <= self._grip_tq_min or self._grip_open_locked:
                    if not self._grip_open_locked:
                        logger.warning(f"[VLA] RTC 夹爪张开力矩 {tq:.2f}Nm ≤ "
                                       f"{self._grip_tq_min}Nm，锁定保位")
                    self._grip_open_locked = True
                    return
                if tq < soft and self._grip_tq_min < soft:
                    k = (self._grip_tq_min - tq) / (self._grip_tq_min - soft)
                    effective = step * max(0.0, min(1.0, k))

        # 限速推进（不超过目标）
        adv = math.copysign(min(effective, abs(delta)), delta)
        new_val = float(np.clip(cur + adv, -math.pi / 2.0, math.pi / 60.0))
        self._grip_last_sent = new_val
        self._motion_cb("gripper_ctrl", new_val)

    def _on_rtc_grip_tick(self):
        """RTC 夹爪限速斜坡（50ms tick）：从上次下发位置向当前目标推进。

        策略学习手柄 xbox_control 的 D-pad 夹爪控制：
        - 每步限速：闭合 1.0 rad/s、张开 1.5 rad/s（每 tick 最多 0.05/0.075 rad）
          ——限制闭合速度，防止力矩瞬间冲过上限（力矩观测突变会让模型分布外）；
        - 力矩软阈值预减速：闭合方向力矩从 0.6×上限 起推进量线性衰减到 0；
        - 触限锁定保位：力矩 ≥ 上限（闭合）/ ≤ 下限（张开）→ 停在当前反馈位置，
          锁定该方向；新目标出现反向/保持意图时在 _on_rtc_chunk 中解锁；
        - 无目标（None）或已到位 → 停表。
        连续性（治反复）：目标更新时斜坡总是从【上次实际下发位置】出发推进，
        永不回绕——旧段走到 -20、新段目标 -5 → 直接 -20→-5；新段目标 -60
        → 直接 -20→-60，不会先回 -32 再走。
        """
        if self._motion_cb is None:
            return
        target = self._grip_target
        if target is None:
            if self._rtc_grip_timer is not None:
                self._rtc_grip_timer.stop()
            return
        cur = self._grip_last_sent
        if cur is None:
            # 首次下发：以模型目标为起点直接下发（后续从该位置斜坡）
            self._grip_last_sent = target
            self._motion_cb("gripper_ctrl", target)
            return
        diff = target - cur
        step_close = self._grip_close_speed * 0.05   # 50ms tick
        step_open = self._grip_open_speed * 0.05
        if abs(diff) <= max(step_close, step_open):
            self._grip_last_sent = target
            self._motion_cb("gripper_ctrl", target)
            if self._rtc_grip_timer is not None:
                self._rtc_grip_timer.stop()
            return

        closing = diff > 0  # 角度增大 = 闭合
        step = step_close if closing else step_open

        # 力矩保护（同手柄策略）：读实时力矩
        tq = None
        if self._lerobot_get_torque is not None:
            try:
                tq = self._lerobot_get_torque()
            except Exception:
                tq = None
        if tq is not None:
            tq = float(tq)
            if closing:
                if (self._grip_close_locked or tq >= self._grip_tq_max):
                    # 触限：锁定闭合方向，保位在当前位置（不再推进）
                    if not self._grip_close_locked:
                        self._grip_close_locked = True
                        logger.warning(f"[VLA] RTC夹爪闭合力矩 {tq:.2f}Nm ≥ "
                                       f"{self._grip_tq_max}Nm，锁定保位 {cur:.2f}rad")
                    if self._rtc_grip_timer is not None:
                        self._rtc_grip_timer.stop()
                    return
                soft = self._grip_tq_max * self._grip_tq_soft_ratio
                if tq > soft:
                    # 预减速：软阈值→上限之间推进量线性衰减
                    k = (self._grip_tq_max - tq) / max(self._grip_tq_max - soft, 1e-6)
                    step *= max(0.0, min(1.0, k))
            else:
                if (self._grip_open_locked or tq <= self._grip_tq_min):
                    if not self._grip_open_locked:
                        self._grip_open_locked = True
                        logger.warning(f"[VLA] RTC夹爪张开力矩 {tq:.2f}Nm ≤ "
                                       f"{self._grip_tq_min}Nm，锁定保位 {cur:.2f}rad")
                    if self._rtc_grip_timer is not None:
                        self._rtc_grip_timer.stop()
                    return
                soft = self._grip_tq_min * self._grip_tq_soft_ratio
                if tq < soft:
                    k = (self._grip_tq_min - tq) / min(self._grip_tq_min - soft, -1e-6)
                    step *= max(0.0, min(1.0, k))

        nxt = float(np.clip(cur + (step if closing else -step),
                            -math.pi / 2.0, math.pi / 60.0))
        self._grip_last_sent = nxt
        self._motion_cb("gripper_ctrl", nxt)

    # ==================== “推理一次”单次执行诊断 ====================

    def _on_vla_once_clicked(self):
        """点击“推理一次”：确保处于暂停控制状态，请求一次完整 chunk 推理。"""
        if not self._vla_running or self._vla_thread is None:
            QMessageBox.warning(self, "提示", "请先启动模型推理")
            return
        if self._motion_cb is None:
            QMessageBox.warning(self, "提示", "运动控制未连接（motion callback 未设置）")
            return
        if self._single_active:
            self.status_label.setText("[VLA] 单次执行进行中，请等待完成")
            return
        # 确保处于“暂停控制”状态：普通/RTC 的连续动作不再下发，机械臂归单次执行独占
        if not self._vla_control_paused:
            self.toggle_vla_control_pause()
        self.status_label.setText("[VLA] 推理一次: 正在推理...")
        self._vla_thread.request_single_chunk()

    def _on_single_chunk(self, chunk, info):
        """收到单次推理 chunk：显示到信息框、保存诊断数据、装入队列开始执行。"""
        chunk = np.asarray(chunk, dtype=np.float64)
        lines = [f"chunk {chunk.shape[0]} 帧 @ {self._lerobot_fps}fps "
                 f"(列: x y z rx ry rz gripper_rad)"]
        for k, a in enumerate(chunk):
            lines.append("k=%02d " % k + " ".join(f"{float(v):+.4f}" for v in a[:7]))
        self.vla_once_info.setPlainText("\n".join(lines))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("inference_once_logs") / ts
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            cv2.imwrite(str(save_dir / "cam1.png"),
                        cv2.cvtColor(info["rgb0"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_dir / "cam2.png"),
                        cv2.cvtColor(info["rgb1"], cv2.COLOR_RGB2BGR))
        except Exception as e:
            logger.warning(f"[VLA] 保存相机图像失败: {e}")
        meta = {
            "time": ts,
            "model_path": self.vla_model_edit.text().strip(),
            "task": info["task"],
            "torque": info["torque"],
            "state": [float(v) for v in info["state"]],
            "fps": self._lerobot_fps,
            "rtc_mode": self._rtc_mode,
            "lowpass_enabled": self._vla_lowpass_enabled,
            "smooth_alpha": self._vla_lowpass_alpha,
            "avg_n": self._vla_avg_n,
            "n_frames": int(chunk.shape[0]),
            "columns": ["x", "y", "z", "rx", "ry", "rz", "gripper_rad"],
        }
        with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        np.savetxt(save_dir / "action_chunk.csv", chunk, delimiter=",",
                   header="x,y,z,rx,ry,rz,gripper_rad", comments="")
        self._single_save_dir = save_dir

        # 平滑执行：整段 chunk 以 MoveJ（S 曲线插值）一次性运动到末端位姿，
        # 不再 20Hz 逐点伺服滴灌（滴灌会把 chunk 内目标抖动直接放大成猛冲）。
        # 机械臂主运动在此下发一次；_on_single_tick 只负责夹爪跟随与日志记录。
        x, y, z, rx, ry, rz, _g = [float(v) for v in chunk[-1][:7]]
        x, y, z, rx, ry, rz = self._clamp_pose_jump(
            x, y, z, rx, ry, rz, self._seq_max_jump, self._seq_max_rot)
        try:
            self._motion_cb("end_pose_ctrl", x, y, z, rx, ry, rz,
                            float(chunk.shape[0]) / max(self._lerobot_fps, 1),
                            False, False, self._rtc_interp)
        except Exception as e:
            logger.warning(f"[VLA] 推理一次主运动下发失败: {e}")

        # 底盘/升降动作分量（action 第 8~11 维，未接底盘时为空操作）
        self._dispatch_mobile_action([np.asarray(r) for r in chunk])

        # EMA 基准从当前位姿起步，避免与上一段执行的残留状态叠加
        self._rtc_last_sent = None
        self._single_queue = deque(chunk)
        self._single_exec_log = []
        if self._single_timer is None:
            self._single_timer = QTimer(self)
            self._single_timer.setTimerType(Qt.TimerType.PreciseTimer)
            self._single_timer.timeout.connect(self._on_single_tick)
        self._single_active = True
        self._single_timer.start(int(1000 / self._lerobot_fps))
        self.status_label.setText(
            f"[VLA] 推理一次: 执行中 ({chunk.shape[0]} 帧)，日志 -> {save_dir}")

    def _on_single_tick(self):
        """单次执行定时器：夹爪按 chunk 序列跟随，并逐帧记录目标/反馈位姿。

        机械臂主运动已在 _on_single_chunk 中以一次 MoveJ 平滑下发；
        这里不再逐步下发臂部目标（避免 20Hz 滴灌猛冲）。
        """
        if not self._single_queue:
            self._finish_single()
            return
        a = self._single_queue.popleft()
        if self._motion_cb is not None:
            try:
                grip = float(np.clip(float(a[6]), -math.pi / 2.0, math.pi / 60.0))
                self._motion_cb("gripper_ctrl", grip)
            except Exception as e:
                logger.warning(f"[VLA] 推理一次夹爪下发失败: {e}")
        try:
            fb = np.asarray(self._get_pose_7d(), dtype=np.float64)
        except Exception:
            fb = np.zeros(7)
        self._single_exec_log.append(
            np.concatenate([np.asarray(a[:7], dtype=np.float64), fb]))

    def _finish_single(self):
        """单次执行完成：停表、保存执行轨迹、回到“暂停控制”状态。"""
        if self._single_timer is not None:
            self._single_timer.stop()
        self._single_active = False
        if self._single_save_dir is not None and self._single_exec_log:
            arr = np.asarray(self._single_exec_log)
            np.savetxt(self._single_save_dir / "executed.csv", arr, delimiter=",",
                       header="tgt_x,tgt_y,tgt_z,tgt_rx,tgt_ry,tgt_rz,tgt_grip,"
                              "fb_x,fb_y,fb_z,fb_rx,fb_ry,fb_rz,fb_grip", comments="")
        # 再次进入“暂停控制”状态，机械臂不再运动
        if not self._vla_control_paused:
            self.toggle_vla_control_pause()
        self.status_label.setText(
            f"[VLA] 推理一次执行完成，已回到暂停控制。日志: {self._single_save_dir}")

    # ==================== RTC：管道平滑块执行（段间插值拼接） ====================

    def _on_rtc_chunk(self, chunk, latency):
        """RTC 管道平滑块执行：当前段未执行完时提前推理，新段到达即插值接续。

        相当于两个“推理一次”之间做插值拼接：
        - 观测时滞补偿（两种互斥方案）：trained RTC 时新 chunk 前缀是上一段
          剩余动作的 clamp 重放，跳过后时滞已被模型吸收；否则按 d 步丢弃
          （d=latency×fps），臂部目标与夹爪序列统一跳过前 d 步，目标取
          相对"现在"领先 N 步（目标步N，末 3 步均值）——消除段间回拉；
          整段一次 end_pose_ctrl(servo=True) 下发，SDK 伺服流从当前设定点
          200Hz 线性插值到新端点（即"从当前位置插补到新目标、不再到旧目标
          E"），位置连续、无保位段；N>10 可按比例提速；
        - 段端点做 EMA 低通（平滑α）与窗口限速（步长mm/姿态rad × 窗口步数）；
          方向单调性约束（投影剥离）：新目标落在当前位置"身后"时剥掉反向
          分量保留侧向修正（位置/姿态独立处理），大幅反向或连续 6 段坚持
          反向则放行——既消除"所有关节来回动"，又不误杀真实撤退；
        - 本段下发后按实测推理延迟提前调度下一段推理，保证段间无缝衔接；
          夹爪按 chunk 窗口内序列 20Hz 滴灌跟随，带死区+换向驻留防抖。
        """
        self._rtc_inflight = False
        if not self._vla_running or self._motion_cb is None:
            return
        if self._vla_control_paused or self._single_active:
            return
        try:
            rows = [np.asarray(r, dtype=np.float64) for r in chunk]
        except Exception:
            return
        if not rows:
            return

        # 每段执行窗口固定 0.5s（fps//2 步）。段时长不做自适应缩短——短窗口
        # （<推理延迟 ~0.3s）会让机械臂走完一段后空等下一段（卡顿），且段边界
        # 过密（速度阶梯）引发晃动；速度由段目标远近体现（见"目标步N"）
        max_steps = max(1, int(self._lerobot_fps) // 2)
        fps = max(self._lerobot_fps, 1)
        window_s = max_steps / fps

        # 观测时滞补偿（两种互斥方案）：
        # - 训练时 RTC（trained）：新 chunk 前 trained_d 步是上一段剩余动作的
        #   clamp 重放（推理期间机械臂已在执行它们），直接跳过；时滞已被
        #   prefix 条件化吸收，不再做 d 步丢弃
        # - 应用层 d 步丢弃（RTC 论文的 discard 基线）：chunk 的观测采集于
        #   请求时刻，推理完成到达时机械臂已走了 d = latency×fps 步。若目标取
        #   chunk 第 N 步（=过期观测+N 步），相对当前位置只剩 N-d 步前瞻，甚至
        #   落在身后——这就是"段间回拉/各关节来回动"的时滞根源。目标改为
        #   第 d+N 步（=相对"现在"领先 N 步），夹爪滴灌同步跳过前 d 步过期序列
        trained_d = 0
        if self._rtc_trained_active():
            trained_d = min(int(getattr(self._vla_thread, "_last_used_delay", 0)),
                            max(0, len(rows) - 1))
            if trained_d > 0:
                rows = rows[trained_d:]
        d_stale = 0 if self._rtc_trained_active() else min(
            int(round(latency * fps)), max(0, len(rows) - 1))

        # 段端点：取 chunk 第 d+N 步的末 3 步均值（目标步N = 相对当前时刻的
        # 前瞻；N=10=0.5s 原速跟踪，N=14~20 瞄准更远点 → 成比例提速）
        tgt_idx = min(d_stale + max(1, int(self._rtc_target_step)), len(rows)) - 1
        rows = rows[:max(max_steps + d_stale, tgt_idx + 1)]
        tail = np.mean([r[:6] for r in rows[max(0, tgt_idx - 2):tgt_idx + 1]], axis=0)
        if self._vla_lowpass_enabled and self._rtc_last_sent is not None:
            alpha = self._vla_lowpass_alpha  # 界面文本框实时可调（0.05~1.0）
            tail = alpha * tail + (1.0 - alpha) * self._rtc_last_sent
        x, y, z, rx, ry, rz = [float(v) for v in tail]

        # 单段位移/姿态上限（安全限幅 + 速度封顶 = 步长mm×10步/0.5s，与 N 无关）
        x, y, z, rx, ry, rz = self._clamp_pose_jump(
            x, y, z, rx, ry, rz,
            self._rtc_max_step_mm / 1000.0 * max_steps,
            self._rtc_max_rot_step * max_steps)

        # 方向单调性约束（往复抑制升级：投影剥离）
        # 新 chunk 端点可能落在当前位置"身后"（观测时刻早于下发时刻、
        # flow-matching 采样噪声），直接执行 = 所有关节反向来回。把位移/姿态
        # 增量沿已建立的运动方向分解，剥掉反向分量、保留侧向修正——主方向
        # 运动单调；大幅反向（真实意图：放置后撤退/大姿态调整）或连续 6 段
        # （~3s）坚持反向（慢速真实撤退）则放行完整反向段
        cur = self._get_pose_7d()
        cur_p = np.asarray(cur[:3], dtype=np.float64)
        cur_r = np.asarray(cur[3:6], dtype=np.float64)
        v = np.array([x - cur_p[0], y - cur_p[1], z - cur_p[2]])
        dist = float(np.linalg.norm(v))
        rv = np.array([rx - cur_r[0], ry - cur_r[1], rz - cur_r[2]])
        rot_n = float(np.linalg.norm(rv))

        allow_reverse = self._rtc_proj_streak >= 6
        projected = False
        if not allow_reverse:
            # 位置：小幅反向 → 剥掉沿主方向的反向分量，保留侧向修正
            if (self._rtc_last_dir is not None and dist < self._rtc_genuine_rev
                    and dist > 1e-9):
                fwd = float(np.dot(v, self._rtc_last_dir))
                if fwd < 0:
                    v = v - fwd * self._rtc_last_dir
                    projected = True
            # 姿态：位置不回退但姿态回退，经 IK 映射同样是"各关节来回动"，
            # 独立做同样的投影剥离
            if (self._rtc_last_rot_dir is not None
                    and rot_n < self._rtc_genuine_rev_rot and rot_n > 1e-9):
                rfwd = float(np.dot(rv, self._rtc_last_rot_dir))
                if rfwd < 0:
                    rv = rv - rfwd * self._rtc_last_rot_dir
                    projected = True

        if projected:
            self._rtc_proj_streak += 1
            x, y, z = (cur_p + v).tolist()
            rx, ry, rz = (cur_r + rv).tolist()
            # EMA/拼接基准记录投影后的实际下发目标；方向基准不更新（不记忆噪声）
            self._rtc_last_sent = np.array([x, y, z, rx, ry, rz])
        else:
            self._rtc_proj_streak = 0
            # 方向基准从"实际下发的运动"（目标-当前位姿）更新，而不是从目标
            # 序列差更新——EMA 后目标可能落在当前位置附近，目标序列差会把
            # 基准污染成反向，导致后续前向段被误剥（首版 bug，单元测试逮住）
            if dist > self._rtc_deadband:
                self._rtc_last_dir = v / dist
            if rot_n > 0.01:
                self._rtc_last_rot_dir = rv / rot_n
            self._rtc_last_sent = tail

        # 下发本段：servo 流从当前设定点线性插值到目标端点（段间位置连续拼接）
        self._motion_cb("end_pose_ctrl", x, y, z, rx, ry, rz,
                        window_s, False, True, self._rtc_interp)
        # trained RTC 执行跟踪：本段 rows 已从原始 chunk 跳过 trained_d 步，
        # 记录起始索引与下发时刻，供下一段请求时推算 prev_chunk_left_over
        self._rtc_exec_offset = trained_d
        self._rtc_exec_t0 = time.perf_counter()
        self._rtc_last_latency = latency
        # 夹爪（v4 终点覆盖）：不重放序列——只取时滞补偿窗口的末值作为目标
        # （= 模型对"现在起 0.5s 后"的最新开合意图），由斜坡定时器从当前
        # 位置限速推进过去。消除"新段起点落后当前实际位置"造成的回拉往复：
        # 第一轮执行到 -20 时新段到达，直接从 -20 连续推进到新段终点。
        grip_rows = rows[d_stale:d_stale + max_steps]
        if grip_rows:
            self._grip_target = float(np.clip(
                grip_rows[-1][6], -math.pi / 2.0, math.pi / 60.0))
            ref = self._grip_last_sent if self._grip_last_sent is not None else self._grip_target
            if self._grip_target > ref + 0.02:
                # 明确的进一步闭合意图（目标明显超过已到达位置）：
                # 解锁闭合锁定重试（保持意图不解锁，避免触限后反复撞击）
                self._grip_close_locked = False
            if self._grip_target < ref + 1e-9:
                # 张开/保持意图：解锁张开锁定（张开方向永不被闭合锁定阻塞）
                self._grip_open_locked = False
            if self._rtc_grip_timer is not None:
                self._rtc_grip_timer.start(int(1000 / fps))

        # 底盘/升降动作分量（action 第 8~11 维，未接底盘时为空操作）
        self._dispatch_mobile_action(rows)

        # 提前调度下一段推理：在本段窗口结束前 (实测延迟+40ms 裕量) 触发，
        # 新段到达时本段已走完 ≥90%（少吞段尾），且留 40ms 抖动余量防停顿；
        # 延迟超过窗口则立即请求（退化为段间短停顿）
        self._rtc_chunk_count += 1
        lead_ms = min(window_s * 900.0, latency * 1000.0 + 40.0)
        next_ms = max(50, int(window_s * 1000.0 - lead_ms))
        if self._rtc_next_timer is not None:
            self._rtc_next_timer.start(next_ms)

        mode_note = (f"trained prefix={trained_d}" if trained_d > 0
                     else f"d={d_stale}+N")
        self.status_label.setText(
            f"[VLA-RTC] 段 #{self._rtc_chunk_count}：目标步{tgt_idx + 1}"
            f"({mode_note})/{int(window_s * 1000)}ms"
            f"（位移 {dist * 1000:.0f}mm"
            f"{'，反向放行' if allow_reverse else ('，投影剥离' if projected else '')}"
            f"），推理 {latency * 1000:.0f}ms")
        logger.info(
            f"[VLA-RTC] 段 #{self._rtc_chunk_count}: steps={len(rows)}, "
            f"window={window_s * 1000:.0f}ms, dist={dist * 1000:.0f}mm, "
            f"latency={latency * 1000:.0f}ms, next_in={next_ms}ms, "
            f"target=({x:.3f},{y:.3f},{z:.3f})")

    def _rtc_trained_active(self):
        """训练时 RTC 是否实际生效（界面开关 + RTC异步 + checkpoint 支持）。"""
        return (self._rtc_mode and self._vla_thread is not None
                and bool(getattr(self._vla_thread, "_trained_rtc_active", False)))

    def _request_next_rtc_chunk(self):
        """RTC 管道调度：本段窗口即将结束，请求下一段 chunk 推理。"""
        if not self._vla_running or self._vla_thread is None:
            return
        if self._vla_control_paused or self._single_active or self._rtc_inflight:
            return
        self._rtc_inflight = True
        # trained RTC：推算本段已执行到的步索引 k 与下一段推理的预测延迟 d
        # （d = 实测延迟×fps，钳制到 [1, 训练 max_delay]），交给推理线程
        # 从上一段归一化 chunk 的 k 处取剩余动作作 prefix 条件
        if self._rtc_trained_active() and self._rtc_exec_t0 > 0:
            fps = max(self._lerobot_fps, 1)
            elapsed = int((time.perf_counter() - self._rtc_exec_t0) * fps)
            k = self._rtc_exec_offset + elapsed
            d = int(round(self._rtc_last_latency * fps))
            d = max(1, min(d, int(getattr(self._vla_thread, "_trained_max_delay", 1))))
            self._vla_thread._pending_rtc = (k, d)
        self._vla_thread.request_chunk()

    def _ensure_lerobot_dataset(self):
        """初始化或加载 LeRobot v3.0 数据集（双臂 14/18 维 + 3 路摄像头）

        优先使用完整 lerobot 库；工控机未安装时自动回退到内置轻量写入器
        （debugger/utils/lerobot_lite_dataset.py，磁盘格式与 lerobot 0.6.1 一致）。

        策略：
        1. 文件夹不存在：直接创建新数据集
        2. 文件夹存在且格式兼容：加载已有数据集，追加 episode
        3. 文件夹存在但格式不兼容：返回字符串 "incompatible"，
           由调用方弹窗提示用户修改文件夹名
        """
        # 优先使用完整 lerobot；工控机无 lerobot 时回退到内置轻量写入器
        # （格式严格对齐 lerobot 0.6.1 v3.0，训练侧可直接加载）
        LeRobotDataset = None
        use_lite = False
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except Exception as e:
            logger.warning(f"[LeRobot] 无法导入 lerobot ({e})，使用内置轻量写入器")
            from debugger.utils.lerobot_lite_dataset import (
                LiteLeRobotDataset as LeRobotDataset)
            use_lite = True

        dataset_dir = os.path.join(self.dataset_root, self._lerobot_repo_name)

        # 当前目标配置
        target_h, target_w = self._lerobot_image_size
        target_cam_dtype = "video"  # 使用视频编码
        target_video_codec = "h264"  # canonical 名称，对应 h264_nvenc / libx264
        state_dim = self._pose_dim()
        target_state_shape = (state_dim,)
        target_action_shape = (state_dim,)
        target_torque_shape = (2,)  # 双臂夹爪力矩 [臂1, 臂2]

        # 编码器（流式编码用）：硬件优先 NVENC，否则软件 libx264。
        # 流式编码把 add_frame 的帧直接喂给 PyAV 编码器（后台线程），
        # 不再落盘 PNG，从根本上消除 PNG 编/解码这一主要瓶颈。
        vcodec = _resolve_vcodec_for_recording()
        encoder_threads = None if vcodec in ("h264_nvenc", "hevc_nvenc") else (
            2 if use_lite else 4)

        # 如果数据集已存在，检查是否兼容
        info_path = os.path.join(dataset_dir, "meta", "info.json")
        # 文件夹已存在但不是 LeRobot 数据集（缺少 meta/info.json）：视为格式不兼容
        if os.path.exists(dataset_dir) and not os.path.exists(info_path):
            logger.warning(
                f"[LeRobot] 文件夹已存在但不是有效的 LeRobot 数据集: {dataset_dir}")
            return "incompatible"

        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    import json
                    info = json.load(f)

                existing_features = info.get("features", {})
                state_shape = existing_features.get("observation.state", {}).get("shape", [7])
                action_shape = existing_features.get("action", {}).get("shape", [7])
                torque_shape = existing_features.get(
                    "observation.gripper_torque", {}).get("shape", [1])

                cam_feats = {}
                for ci in (1, 2, 3):
                    cf = existing_features.get(f"observation.images.cam{ci}", {})
                    cam_feats[ci] = (
                        cf.get("shape", [0, 0, 0]),
                        cf.get("dtype", "image"),
                        cf.get("info", {}).get("video.codec", ""),
                    )

                # 夹爪力矩特征必须存在且为双臂 2 维，否则 add_frame 会失败
                has_torque = "observation.gripper_torque" in existing_features

                compatible = (
                    tuple(state_shape) == target_state_shape and
                    tuple(action_shape) == target_action_shape and
                    tuple(torque_shape) == target_torque_shape and
                    has_torque and
                    all(cam_feats[ci][1] == target_cam_dtype and
                        cam_feats[ci][2] == target_video_codec and
                        tuple(cam_feats[ci][0]) == (target_h, target_w, 3)
                        for ci in (1, 2, 3))
                )

                if not compatible:
                    logger.warning(
                        f"[LeRobot] 数据集 '{self._lerobot_repo_name}' 格式不兼容 "
                        f"(state={state_shape}, action={action_shape}, "
                        f"torque={torque_shape}, cam_feats={cam_feats})"
                    )
                    return "incompatible"
                else:
                    # 兼容，加载已有数据集
                    import os as os_module
                    old_hf_offline = os_module.environ.get("HF_HUB_OFFLINE")
                    os_module.environ["HF_HUB_OFFLINE"] = "1"
                    if use_lite:
                        dataset = LeRobotDataset.resume(
                            self._lerobot_repo_name, root=dataset_dir,
                            vcodec=vcodec, encoder_threads=encoder_threads,
                        )
                    elif hasattr(LeRobotDataset, "resume"):
                        # lerobot 0.6+：用 resume() 追加录制；必须显式指定与已有
                        # 数据一致的编码器，否则默认 libsvtav1 会与 h264 旧数据混码
                        dataset = LeRobotDataset.resume(
                            self._lerobot_repo_name, root=dataset_dir,
                            streaming_encoding=True, encoder_threads=encoder_threads,
                            rgb_encoder=_make_rgb_encoder_config(vcodec),
                        )
                    else:
                        # lerobot 0.4.x：直接构造并传入 vcodec
                        dataset = LeRobotDataset(
                            self._lerobot_repo_name, root=dataset_dir,
                            streaming_encoding=True, vcodec=vcodec,
                            encoder_threads=encoder_threads,
                        )
                    if old_hf_offline is not None:
                        os_module.environ["HF_HUB_OFFLINE"] = old_hf_offline
                    else:
                        os_module.environ.pop("HF_HUB_OFFLINE", None)
                    logger.info(f"[LeRobot] 加载已有数据集: {dataset_dir} "
                                f"(episodes={dataset.num_episodes}, frames={dataset.num_frames})")
                    return dataset
            except Exception as e:
                logger.warning(
                    f"[LeRobot] 加载已有数据集 '{self._lerobot_repo_name}' 失败: {e}")
                return "incompatible"

        # LeRobot v3.0 features 格式 - 摄像头使用 video dtype 才能配合 use_videos=True
        # 双臂：state/action = arm1_pose7 + arm2_pose7（+底盘 4 维 = 18）；
        # 三路摄像头：cam1=臂1腕部，cam2=头部，cam3=臂2腕部
        features = {
            "observation.images.cam1": {
                "dtype": "video",
                "shape": (*self._lerobot_image_size, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.cam2": {
                "dtype": "video",
                "shape": (*self._lerobot_image_size, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.cam3": {
                "dtype": "video",
                "shape": (*self._lerobot_image_size, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "observation.gripper_torque": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["torque_arm1", "torque_arm2"],
            },
            "action": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["action"],
            },
        }

        try:
            # 设置离线模式避免联网
            import os as os_module
            old_hf_offline = os_module.environ.get("HF_HUB_OFFLINE")
            os_module.environ["HF_HUB_OFFLINE"] = "1"

            # 编码参数：完整 lerobot 按版本自适应（0.6+ 用 rgb_encoder，
            # 0.4.x 用 vcodec）；轻量写入器直接传 vcodec
            if use_lite:
                dataset = LeRobotDataset.create(
                    repo_id=self._lerobot_repo_name,
                    fps=self._lerobot_fps,
                    features=features,
                    root=dataset_dir,
                    robot_type="ela3",
                    use_videos=True,
                    vcodec=vcodec,
                    encoder_threads=encoder_threads,
                )
            else:
                import inspect as _inspect
                codec_kwargs = {}
                if "rgb_encoder" in _inspect.signature(LeRobotDataset.create).parameters:
                    codec_kwargs["rgb_encoder"] = _make_rgb_encoder_config(vcodec)
                else:
                    codec_kwargs["vcodec"] = vcodec

                dataset = LeRobotDataset.create(
                    repo_id=self._lerobot_repo_name,
                    fps=self._lerobot_fps,
                    features=features,
                    root=dataset_dir,
                    robot_type="ela3",
                    use_videos=True,  # 视频编码，大幅降低 I/O 和 save_episode 耗时
                    image_writer_processes=0,
                    image_writer_threads=4,
                    encoder_threads=encoder_threads,
                    # 流式编码：add_frame 直接喂给 PyAV(H264/NVENC) 后台线程，
                    # 跳过"每帧写 PNG → save_episode 再读回 PNG 编码"的慢路径。
                    # 实测 add_frame 68ms→8ms/帧、save_episode 1765ms→497ms。
                    streaming_encoding=True,
                    **codec_kwargs,
                )
            
            # 恢复环境变量
            if old_hf_offline is not None:
                os_module.environ["HF_HUB_OFFLINE"] = old_hf_offline
            else:
                os_module.environ.pop("HF_HUB_OFFLINE", None)
                
            logger.info(f"[LeRobot] 创建新数据集: {dataset_dir}")
            return dataset
        except Exception as e:
            logger.error(f"[LeRobot] 创建数据集失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _lerobot_push_frame(self):
        """主线程定时器回调：快速把当前双路帧推入后台队列，不执行 add_frame。

        遥操作录制（ela3_demo）与推理录制（ela3_rollout）共用本定时器，
        两路互斥，推给当前活动的录制线程。
        """
        if self._lerobot_recording and self._lerobot_record_thread is not None:
            thread = self._lerobot_record_thread
        elif self._rollout_recording and self._rollout_record_thread is not None:
            thread = self._rollout_record_thread
        else:
            return

        if len(self.camera_indices) < 3:
            return

        # cam1=臂1腕部(idx0), cam2=头部(idx1), cam3=臂2腕部(idx2)
        idx0, idx1, idx2 = (self.camera_indices[0], self.camera_indices[1],
                            self.camera_indices[2])
        frame0 = self.latest_frames.get(idx0)
        frame1 = self.latest_frames.get(idx1)
        frame2 = self.latest_frames.get(idx2)
        if frame0 is None or frame1 is None or frame2 is None:
            return

        try:
            t0 = time.perf_counter()
            rgb0 = self._resize_frame(frame0, *self._lerobot_image_size,
                                      crop_side=self._lerobot_crop_side)
            rgb1 = self._resize_frame(frame1, *self._lerobot_image_size,
                                      crop_side=self._lerobot_crop_side)
            rgb2 = self._resize_frame(frame2, *self._lerobot_image_size,
                                      crop_side=self._lerobot_crop_side)
            thread.push_frame(rgb0, rgb1, rgb2)
            dt = (time.perf_counter() - t0) * 1000

            if thread is self._rollout_record_thread:
                self._rollout_frame_count += 1
            else:
                self._lerobot_frame_count += 1
            if (self._lerobot_frame_count + self._rollout_frame_count) % 20 == 0:
                logger.info(
                    f"[LeRobot] 已采集 lerobot={self._lerobot_frame_count} / "
                    f"rollout={self._rollout_frame_count} 帧，主线程入队耗时 {dt:.1f}ms"
                )
        except Exception as e:
            logger.warning(f"[LeRobot] 图像缩放或入队失败: {e}")

    def _lerobot_on_timer(self):
        """已废弃：旧主线程 add_frame 路径，保留兼容"""
        pass

    def _zero_pose7(self):
        """VLA 单臂路径的零位姿：7 + 底盘 4 维（与 _get_pose_7d 一致）"""
        return np.zeros(
            7 + (4 if getattr(self, "_mobile_get_state", None) is not None else 0),
            dtype=np.float32)

    def _get_pose_7d(self):
        """返回 [x, y, z, rx, ry, rz, gripper_rad] (+底盘 4 维) 或回调不可用时全0

        基础 7 维：位置(3) + 欧拉角(3，弧度) + 夹爪角度(1，弧度)。
        接入底盘后追加 [lift_m, vx, vy, omega]（升降绝对位置 +
        底盘机体系实测速度）共 11 维。
        """
        if self._lerobot_get_end_pose is None:
            return self._zero_pose7()

        try:
            result = self._lerobot_get_end_pose()
        except Exception as e:
            logger.warning(f"[LeRobot] 获取末端位姿失败: {e}")
            return self._zero_pose7()

        # 兼容旧接口：可能只返回 ArmEndPose
        if isinstance(result, tuple) and len(result) >= 2:
            end_pose, gripper_rad = result[0], result[1]
        else:
            end_pose = result
            gripper_rad = 0.0

        if end_pose is None:
            return self._zero_pose7()

        x = getattr(end_pose, "x", 0.0)
        y = getattr(end_pose, "y", 0.0)
        z = getattr(end_pose, "z", 0.0)
        rx = getattr(end_pose, "rx", 0.0)
        ry = getattr(end_pose, "ry", 0.0)
        rz = getattr(end_pose, "rz", 0.0)

        # 夹爪角度：统一弧度（与机械臂 GripperCtrl 接口、末端姿态 rx/ry/rz 单位一致）
        if gripper_rad is None:
            gripper_rad = 0.0

        pose = np.array([x, y, z, rx, ry, rz, float(gripper_rad)], dtype=np.float32)
        mobile = self._get_mobile_state()
        if mobile is not None:
            pose = np.concatenate([pose, mobile])
        return pose

    def _center_crop_to_lerobot(self, frame):
        """按 LeRobot 目标分辨率的长宽比对 BGR 帧做中心裁切（与 _resize_frame 一致）。

        用于界面预览同步裁切：如 LeRobot 选 720x720，则把 1280x720 画面
        中心裁出 720x720 方形区域再显示，所见即所录。
        设置了 _lerobot_crop_side 时（720视野缩放到480类选项），
        裁切该边长的中心正方形，与录制/推理的预处理保持一致。
        """
        target_h, target_w = self._lerobot_image_size
        h, w = frame.shape[:2]
        if self._lerobot_crop_side:
            side = min(h, w, self._lerobot_crop_side)
            x0 = (w - side) // 2
            y0 = (h - side) // 2
            return frame[y0:y0 + side, x0:x0 + side]
        src_ar = w / h
        dst_ar = target_w / target_h
        if abs(src_ar - dst_ar) > 1e-3:
            if src_ar > dst_ar:   # 源更宽：左右中心裁切
                new_w = int(h * dst_ar)
                x0 = (w - new_w) // 2
                frame = frame[:, x0:x0 + new_w]
            else:                 # 源更高：上下中心裁切
                new_h = int(w / dst_ar)
                y0 = (h - new_h) // 2
                frame = frame[y0:y0 + new_h, :]
        return frame

    @staticmethod
    def _resize_frame(frame, target_h, target_w, crop_side=None):
        """将 BGR 帧缩放为 RGB numpy 数组 (H, W, 3)。

        crop_side 为 None 时：目标长宽比与源不一致先中心裁切再缩放
        （如 1280x720 -> 720x720 取画面中央方形区域），避免拉伸失真。
        crop_side 指定时：先裁该边长的中心正方形（保留更大视野，如 720），
        再降采样到目标分辨率（如 480x480）——视野不变，只减像素。
        源不足 crop_side 时退化为源内最大正方形裁切。
        """
        h, w = frame.shape[:2]
        if crop_side:
            side = min(h, w, crop_side)
            x0 = (w - side) // 2
            y0 = (h - side) // 2
            frame = frame[y0:y0 + side, x0:x0 + side]
        else:
            src_ar = w / h
            dst_ar = target_w / target_h
            if abs(src_ar - dst_ar) > 1e-3:
                if src_ar > dst_ar:   # 源更宽：左右中心裁切
                    new_w = int(h * dst_ar)
                    x0 = (w - new_w) // 2
                    frame = frame[:, x0:x0 + new_w]
                else:                 # 源更高：上下中心裁切
                    new_h = int(w / dst_ar)
                    y0 = (h - new_h) // 2
                    frame = frame[y0:y0 + new_h, :]
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb

    def toggle_lerobot_record(self):
        """开始/停止 LeRobot v0.3 数据采集"""
        if self._lerobot_recording:
            self._stop_lerobot_record()
        else:
            self._start_lerobot_record()

    def _change_lerobot_resolution(self, index):
        """切换 LeRobot 数据集的图像分辨率"""
        name, w, h, *rest = self.lerobot_res_options[index]
        self._lerobot_image_size = (h, w)
        # 可选第4元素：固定中心正方形裁切边长（如 720视野缩放到480）
        self._lerobot_crop_side = rest[0] if rest else None
        logger.info(f"[Camera] LeRobot 图像分辨率切换为: {name}")

    def _ensure_capture_resolution(self):
        """确保摄像头采集分辨率不低于 LeRobot 录制目标分辨率。

        如果当前采集分辨率低于目标分辨率，自动切换到目标分辨率并重新打开摄像头。
        这是为了防止“录制设置 1280x720，但实际是从 320x240 放大存储”的画质问题。
        """
        target_h, target_w = self._lerobot_image_size
        need_w, need_h = target_w, target_h
        # "720视野缩放到480"类选项：裁切区域为 crop_side 正方形，采集模式
        # 必须能覆盖该正方形（摄像头无 720x720 原生模式，选最小的
        # 宽/高均 >= crop_side 的支持模式，如 1280x720），
        # 否则视野来自放大而非原生
        if self._lerobot_crop_side:
            side = self._lerobot_crop_side
            candidates = sorted(
                ((w, h) for _, w, h in self.resolutions if w >= side and h >= side),
                key=lambda wh: wh[0] * wh[1])
            if candidates:
                need_w, need_h = candidates[0]
            else:
                need_w = max(need_w, side)
                need_h = max(need_h, side)
        cap_w, cap_h = self.capture_res

        if cap_w < need_w or cap_h < need_h:
            logger.warning(
                f"[LeRobot] 当前采集分辨率 {cap_w}x{cap_h} 低于录制目标 {need_w}x{need_h}，"
                f"自动切换到 {need_w}x{need_h}"
            )
            self.capture_res = (need_w, need_h)
            # 同步更新 UI 下拉框
            for i, (name, w, h) in enumerate(self.resolutions):
                if w == need_w and h == need_h:
                    self.res_combo.blockSignals(True)
                    self.res_combo.setCurrentIndex(i)
                    self.res_combo.blockSignals(False)
                    break
            # 重新打开摄像头
            for idx in self.camera_indices[:3]:
                if idx in self.caps:
                    self.open_camera(idx, need_w, need_h)

    def _start_lerobot_record(self):
        if self._rollout_recording:
            QMessageBox.information(self, "LeRobot 录制", "推理录制进行中，请先停止。")
            return
        if len(self.camera_indices) < 3:
            QMessageBox.warning(self, "LeRobot 录制",
                                "需要 3 个摄像头才能录制（臂1腕部/头部/臂2腕部）")
            return

        # 关键：录制前确保采集分辨率不低于目标分辨率，避免低分辨率放大存储
        self._ensure_capture_resolution()

        # 检查任务描述：禁止中文
        task_text = self.task_input.text().strip()
        if any('\u4e00' <= c <= '\u9fff' for c in task_text):
            QMessageBox.warning(
                self, "Task Format Error",
                "任务描述不能包含中文，请使用英文。\n"
                "例如: 'pick up the red cube'"
            )
            return

        # 读取录制文件夹名：默认 ela3_demo，留空时回退默认
        repo_name = self.repo_input.text().strip()
        if not repo_name:
            repo_name = "ela3_demo"
            self.repo_input.setText(repo_name)
        self._lerobot_repo_name = repo_name

        # 记录实际采集分辨率，便于排查画质问题
        for idx in self.camera_indices[:3]:
            if idx in self.caps:
                cap = self.caps[idx]
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"[LeRobot] 摄像头 [{idx}] 实际采集分辨率: {actual_w}x{actual_h}")

        dataset = self._ensure_lerobot_dataset()
        if isinstance(dataset, str) and dataset == "incompatible":
            QMessageBox.warning(
                self, "LeRobot 录制",
                f"文件夹 '{self._lerobot_repo_name}' 已存在，但格式不兼容，"
                f"请修改文件夹名。"
            )
            return
        if dataset is None:
            QMessageBox.critical(self, "LeRobot 录制", "无法初始化 LeRobotDataset，请检查日志")
            return

        self._lerobot_dataset = dataset
        self._lerobot_task_name = task_text or "default_task"
        self._lerobot_frame_count = 0
        self._lerobot_recording = True

        # 启动后台录制线程
        self._lerobot_record_thread = LeRobotRecordThread(
            dataset=self._lerobot_dataset,
            pose_callback=self._lerobot_get_end_pose,
            pose2_callback=self._lerobot_get_end_pose2,
            fps=self._lerobot_fps,
            max_queue=5,
            torque_callback=self._lerobot_get_torque,
            torque2_callback=self._lerobot_get_torque2,
            action_lookahead=self._action_lookahead,
            extra_state_callback=self._get_mobile_state,
            parent=self,
        )
        self._lerobot_record_thread.set_task_name(self._lerobot_task_name)
        self._lerobot_record_thread.frame_written.connect(self._on_lerobot_frame_written)
        self._lerobot_record_thread.error.connect(self._on_lerobot_thread_error)
        self._lerobot_record_thread.start()

        # 启动定时器，以固定频率采集并推入队列（20Hz = 50ms）
        interval_ms = int(1000 / self._lerobot_fps)
        self._lerobot_timer.start(interval_ms)

        self.spin_lookahead.setEnabled(False)  # 录制期间锁定前瞻帧数
        self.repo_input.setEnabled(False)      # 录制期间锁定文件夹名
        self.btn_rollout_record.setEnabled(False)  # 互斥：录制期间禁用推理录制
        self.btn_lerobot.setText("停止LeRobot录制")
        self.btn_lerobot.setStyleSheet(
            "font-size: 12px; padding: 4px 12px; background-color: #ff4444; color: white;"
        )
        self.status_label.setText(f"LeRobot 录制中: {self._lerobot_task_name}")
        logger.info(f"[LeRobot] 开始录制任务: {self._lerobot_task_name}，频率 {self._lerobot_fps}Hz")
        if getattr(self, "_mobile_get_state", None) is not None:
            logger.info(
                "[LeRobot] 底盘 4 维 = [lift_m(绝对位置), vx, vy, omega(机体系实测速度)]。"
                "注意：与旧的 [lift, odom_x, odom_y, yaw] 位姿数据语义不同，不可混训")

    def _on_lerobot_frame_written(self, count):
        """后台线程写入帧数回调（可选）"""
        pass

    def _on_lerobot_thread_error(self, msg):
        """后台线程出错回调（线程已退出，finished_ok 不会再发，
        不能走 _stop_lerobot_record 的等待路径，否则 UI 卡在"正在保存"）"""
        logger.error(f"[LeRobot] 录制线程错误: {msg}")
        self._lerobot_recording = False
        self._lerobot_timer.stop()
        QMessageBox.critical(self, "LeRobot 录制错误", msg)
        self._finalize_lerobot_stop("录制失败，本段数据未保存")

    def _stop_lerobot_record(self):
        """停止录制并保存 episode，确保文件正确关闭"""
        logger.info("[LeRobot] 停止录制")
        self._lerobot_recording = False
        self._lerobot_timer.stop()

        # 先停止入队，等待后台线程把队列里剩余帧写完并自动保存
        if self._lerobot_record_thread is not None:
            self._lerobot_record_thread.finished_ok.connect(self._on_lerobot_finished)
            self._lerobot_record_thread.stop(flush=False)
            self.btn_lerobot.setEnabled(False)
            self.status_label.setText("LeRobot 正在保存，请稍候...")
            # 不能在这里 wait，因为 save_episode 在 run() 里执行会阻塞
            # 等 finished_ok 信号后再做 UI 清理
        else:
            self._finalize_lerobot_stop()

    def _on_lerobot_finished(self, result):
        """后台线程保存完成后回调（主线程）"""
        logger.info(f"[LeRobot] {result}")
        self._finalize_lerobot_stop(result)

    def _finalize_lerobot_stop(self, result=""):
        """完成 UI 清理和 parquet 验证（非阻塞）"""
        self.spin_lookahead.setEnabled(True)  # 解除前瞻帧数锁定
        self.repo_input.setEnabled(True)      # 解除文件夹名锁定
        # 先记录 dataset 路径，并在日志文本框打印保存结果（不再弹窗确认）。
        dataset_root = str(self._lerobot_dataset.root) if self._lerobot_dataset else ""
        if result:
            self.camera_log.emit(f"[LeRobot] 保存成功: {result} 到 {dataset_root}")

        # 遥操作示范：保存默认成功标签（RECAP P2.1，静默不弹窗；
        # 录砸的后续在 visualize_data 用三态控件改判，人工 > 派生）
        if result and "已保存" in result and dataset_root:
            try:
                ep_idx = max(0, self._meta_episode_count(dataset_root) - 1)
                self._write_recap_annotations(dataset_root, ep_idx, [], True,
                                              source="teleop")
            except Exception as e:
                logger.warning(f"[LeRobot] 示范默认成功标注失败: {e}")

        # 验证当前 data 目录下存在的 parquet 文件（注意不要在备份目录里验证）。
        # save_episode 刚结束时可能还有临时文件，失败时短暂重试避免误报。
        try:
            import pandas as pd
            import time as _time
            if dataset_root and os.path.exists(dataset_root):
                data_dir = os.path.join(dataset_root, "data")
                if os.path.exists(data_dir):
                    for root, dirs, files in os.walk(data_dir):
                        # 跳过备份目录
                        dirs[:] = [d for d in dirs if not d.endswith(".bak")]
                        for f in files:
                            if f.endswith('.parquet'):
                                fpath = os.path.join(root, f)
                                df = None
                                last_err = None
                                for attempt in range(3):
                                    try:
                                        df = pd.read_parquet(fpath)
                                        break
                                    except Exception as e:
                                        last_err = e
                                        # 第一次失败等 300ms，让 LeRobot 完成内部文件迁移
                                        if attempt < 2:
                                            _time.sleep(0.3)
                                if df is not None:
                                    logger.info(f"[LeRobot] 验证 parquet OK: {os.path.relpath(fpath, dataset_root)} ({len(df)} rows)")
                                else:
                                    logger.error(f"[LeRobot] 验证 parquet 失败: {os.path.relpath(fpath, dataset_root)}: {last_err}")
        except Exception as e:
            logger.warning(f"[LeRobot] 验证 parquet 时出错: {e}")

        # 清理 dataset 引用，避免再次触发析构写文件
        self._lerobot_dataset = None

        self._lerobot_frame_count = 0
        if self._lerobot_record_thread is not None:
            self._lerobot_record_thread.wait(2000)
            self._lerobot_record_thread.deleteLater()
            self._lerobot_record_thread = None

        self.btn_lerobot.setText("开始LeRobot录制")
        self.btn_lerobot.setStyleSheet(
            "font-size: 12px; padding: 4px 12px; background-color: #4CAF50; color: white;"
        )
        self.status_label.setText("LeRobot 录制已停止")
        self.btn_lerobot.setEnabled(True)
        self.btn_rollout_record.setEnabled(True)  # 互斥解除：恢复推理录制按钮

    # ==================== 推理录制（部署 rollout 采集，RECAP P2） ====================

    def toggle_rollout_record(self):
        """开始/停止推理录制（部署 rollout 采集）。

        与遥操作 LeRobot 录制互斥；仅 VLA 推理状态可启动；
        录制到 ela3_rollout；接管自动标注直接生效（无确认制）。
        """
        if self._rollout_recording:
            self._stop_rollout_record()
        else:
            self._start_rollout_record()

    def _start_rollout_record(self):
        # 推理状态判断：仅 VLA 推理运行中才可开始推理录制
        if not self._vla_running:
            QMessageBox.information(self, "推理录制", "请先启动 VLA 推理，再开始推理录制。")
            return
        if self._lerobot_recording:
            QMessageBox.information(self, "推理录制", "遥操作录制进行中，请先停止。")
            return
        if len(self.camera_indices) < 3:
            QMessageBox.warning(self, "推理录制",
                                "需要 3 个摄像头才能录制（臂1腕部/头部/臂2腕部）")
            return

        self._ensure_capture_resolution()

        # 复用数据集创建逻辑：临时切换到固定 rollout 仓库名（默认 ela3_rollout）
        prev_repo = self._lerobot_repo_name
        self._lerobot_repo_name = self._rollout_repo_name
        dataset = self._ensure_lerobot_dataset()
        self._lerobot_repo_name = prev_repo
        if isinstance(dataset, str) and dataset == "incompatible":
            QMessageBox.warning(self, "推理录制", "ela3_rollout 已存在但格式不兼容，请检查。")
            return
        if dataset is None:
            QMessageBox.critical(self, "推理录制", "无法初始化 LeRobotDataset，请检查日志")
            return

        self._rollout_dataset = dataset
        task_text = self.vla_task_input.currentText().strip() or "default_task"
        self._rollout_frame_count = 0
        self._rollout_staged_intervals = []
        self._rollout_takeover_count = 0
        # π*0.6 元数据：录制开始即固化策略标识与组归属（下拉选历史组=继续该组，
        # 输入新组名=新建组；同组名+同策略的录制自动归并为一组）
        self._refresh_rollout_group_labels()
        self._rollout_policy_tag = self._current_policy_tag()
        self._rollout_active_group_id = self._rollout_resolve_group_id()
        # 录制开始时若已处于接管（手柄已连接），干预从第 0 帧起算
        self._rollout_takeover_start = 0 if self._vla_control_paused else None
        self._rollout_recording = True

        self._rollout_record_thread = LeRobotRecordThread(
            dataset=self._rollout_dataset,
            pose_callback=self._lerobot_get_end_pose,
            pose2_callback=self._lerobot_get_end_pose2,
            fps=self._lerobot_fps,
            max_queue=5,
            torque_callback=self._lerobot_get_torque,
            torque2_callback=self._lerobot_get_torque2,
            action_lookahead=self._action_lookahead,
            extra_state_callback=self._get_mobile_state,
            parent=self,
        )
        self._rollout_record_thread.set_task_name(task_text)
        self._rollout_record_thread.error.connect(self._on_rollout_thread_error)
        self._rollout_record_thread.start()

        self._lerobot_timer.start(int(1000 / self._lerobot_fps))

        self.btn_lerobot.setEnabled(False)  # 互斥：录制期间禁用遥操作录制
        self.btn_rollout_record.setText("停止推理录制")
        self.btn_rollout_record.setStyleSheet(
            "font-size: 12px; padding: 4px 12px; background-color: #ff4444; color: white;"
        )
        grp = f", 组={self._rollout_active_group_id}" if self._rollout_active_group_id else ""
        self.status_label.setText(f"推理录制中: {task_text} → ela3_rollout{grp}")
        logger.info(
            f"[RolloutRec] 开始推理录制: task={task_text}, repo=ela3_rollout, "
            f"policy={self._rollout_policy_tag}, group={self._rollout_active_group_id}")

    def _on_rollout_thread_error(self, msg):
        logger.error(f"[RolloutRec] 录制线程错误: {msg}")
        QMessageBox.critical(self, "推理录制错误", msg)
        self._stop_rollout_record()

    def _rollout_push_count(self):
        """当前已推入的帧数（与数据集 frame_index 同序，作接管标注的帧时钟）。"""
        t = self._rollout_record_thread
        return int(getattr(t, "_push_count", 0) or 0) if t is not None else 0

    # ---------- π*0.6 元数据：策略版本 + 任务组（PI06_RL_迁移实施方案.md R1a） ----------

    def _current_policy_tag(self):
        """当前推理策略标识：优先推理线程实际加载的 model_path，回退界面输入框。"""
        try:
            if self._vla_thread is not None and getattr(self._vla_thread, "model_path", None):
                return str(self._vla_thread.model_path).strip()
        except Exception:
            pass
        return self.vla_model_edit.text().strip() or None

    @staticmethod
    def _policy_short(path):
        """模型路径压缩为策略短名（如 ckpt/pi05_ela3_fullft20260830/checkpoints/012000
        → pi05_ela3_fullft20260830/012000）。

        规则：路径含 checkpoints 段时取「上一级目录（run 名）/ checkpoint 段名」。
        同 run 不同 checkpoint 是不同策略版本（008000 与 012000 性能不同），
        组相对优势要求同组同策略，编号必须保留（PI06 §4.1 设计示例
        "做饭-开火 | pi05_xxx/030000"）。否则取最后一个非通用目录段。
        """
        if not path:
            return "unknown"
        raw = [s for s in str(path).replace("\\", "/").split("/") if s]
        if "checkpoints" in raw:
            i = raw.index("checkpoints")
            if i >= 1:
                run = raw[i - 1]
                step = raw[i + 1] if i + 1 < len(raw) else ""
                return f"{run}/{step}" if step else run
        generic = {"", ".", "ckpt", "checkpoints", "pretrained_models", "debugger"}
        segs = [s for s in raw if s not in generic]
        return segs[-1] if segs else str(path)

    def _rollout_resolve_group_id(self):
        """解析本次录制的 group_id（录制开始时调用一次并缓存到 active 字段）。

        group_id = "组名 | 策略短名"（如 "做饭-开火 | pi05_xxx/030000"）。
        下拉选择历史组名 = 继续该组；输入新名称 = 新建组；留空 = None（不分组）。
        同组名+同策略的所有录制自动归并为一组（跨会话选同名组是有意延续）。
        """
        label = self.rollout_group_input.currentText().strip()
        if not label:
            return None
        # 复用开录时已捕获的 _rollout_policy_tag（与写入标注的血缘严格一致）
        policy = self._policy_short(self._rollout_policy_tag or self._current_policy_tag())
        self._remember_group_label(label)
        return f"{label} | {policy}"

    def _refresh_rollout_group_labels(self):
        """从 ela3_rollout 标注加载历史组名到下拉框（保留当前输入不覆盖）。

        组名 = group_id 按 " | " 拆分的首段；无数据集/无标注时静默跳过。
        """
        try:
            import json as _json
            ann_path = Path(self.dataset_root) / self._rollout_repo_name / "meta" / "reward_annotations.json"
            if not ann_path.is_file():
                return
            with open(ann_path, "r", encoding="utf-8") as f:
                ann = _json.load(f)
            for ep in (ann.get("episodes", {}) or {}).values():
                gid = ep.get("group_id")
                if gid:
                    self._remember_group_label(str(gid).split(" | ")[0].strip())
        except Exception as e:
            logger.debug(f"[RolloutRec] 加载历史组名失败（忽略）: {e}")

    def _remember_group_label(self, label):
        """把组名加入下拉框候选（去重、不改变当前输入）。"""
        if not label:
            return
        items = [self.rollout_group_input.itemText(i)
                 for i in range(self.rollout_group_input.count())]
        if label not in items:
            self.rollout_group_input.addItem(label)

    def _rollout_on_control_boundary(self, takeover_started):
        """控制暂停翻转钩子：推理录制中由 toggle_vla_control_pause 调用。

        接管开始记起始帧；接管结束生成干预区间（人工纠正，恒正优势）。
        """
        if not self._rollout_recording:
            return
        frame = self._rollout_push_count()
        if takeover_started:
            if self._rollout_takeover_start is None:
                self._rollout_takeover_start = frame
                logger.info(f"[RolloutRec] 接管开始 @帧{frame}")
        else:
            T = self._rollout_takeover_start
            if T is None:
                return
            self._rollout_takeover_start = None
            self._rollout_stage_takeover_intervals(T, frame)
            logger.info(f"[RolloutRec] 接管结束 @帧{frame}（T={T}）")

    def _rollout_stage_takeover_intervals(self, T, R):
        """生成一次接管的标注区间对（决策修订 2026-08-29：恢复接管前 bad 窗口）。

        - 干预区间 [T, R-1]：恒正优势（RECAP 规则：干预段 I_t=True）；
        - bad 窗口 [T-N, T-1]（N=_rollout_pre_bad_frames，默认 20）：
          接管前策略失败动作标 bad——P1 阶段负优势的唯一自动来源（否则失败帧
          按 0 权重被正常模仿）；P3 阶段作为 critic 的监督样本（"接管前状态
          V 应低"）。出处 APO 论文 last-10-actions 规则（我们默认 20 帧@20fps=1s）。
          边界处理：起点 <0 截断；与上一段已暂存区间重叠时裁剪（快速连续接管）。

        T=接管开始帧（推帧计数），R=接管结束帧。多次接管各自追加，互不重叠。
        """
        # bad 窗口（防重叠：不与已暂存区间的最后一帧重叠）
        n = int(getattr(self, "_rollout_pre_bad_frames", 20))
        if n > 0 and T >= 1:
            s = max(0, T - n)
            if self._rollout_staged_intervals:
                prev_end = int(self._rollout_staged_intervals[-1]["end"])
                s = max(s, prev_end + 1)
            if s <= T - 1:
                self._rollout_staged_intervals.append({
                    "start": s, "end": T - 1, "label": "bad",
                    "note": f"接管前{n}帧内（策略失败动作，APO规则）",
                    "is_human_intervention": False,
                })
        # 干预段
        if R - 1 >= T:
            self._rollout_staged_intervals.append({
                "start": T, "end": R - 1, "label": "good",
                "note": "人工接管纠正", "is_human_intervention": True,
            })
        self._rollout_takeover_count += 1

    def _stop_rollout_record(self):
        """停止推理录制：保存 episode → 弹窗评价成败 → 写入标注与部署日志。"""
        logger.info("[RolloutRec] 停止推理录制")
        # 停止时仍处于接管：按当前帧收尾干预段
        if self._rollout_takeover_start is not None:
            T = self._rollout_takeover_start
            self._rollout_takeover_start = None
            self._rollout_stage_takeover_intervals(T, self._rollout_push_count())
        self._rollout_recording = False
        if not self._lerobot_recording:
            self._lerobot_timer.stop()
        if self._rollout_record_thread is not None:
            self._rollout_record_thread.finished_ok.connect(self._on_rollout_saved)
            self._rollout_record_thread.stop(flush=False)
            self.btn_rollout_record.setEnabled(False)
            self.status_label.setText("推理录制正在保存，请稍候...")
        else:
            self._finalize_rollout_stop(0, "")

    def _on_rollout_saved(self, result):
        frames = int(getattr(self._rollout_record_thread, "_frame_count", 0) or 0)
        logger.info(f"[RolloutRec] {result}")
        success = None
        if frames > 0:
            success = self._ask_rollout_episode_success(frames)
        self._finalize_rollout_stop(frames, result, success)

    def _ask_rollout_episode_success(self, frames):
        """弹窗询问本段推理录制的最终成败；返回 True/False/None(跳过)。"""
        box = QMessageBox(self)
        box.setWindowTitle("推理录制评价")
        box.setText(
            f"本段推理录制已保存（{frames} 帧，接管 {self._rollout_takeover_count} 次）。\n"
            f"任务最终成功了吗？\n\n"
            f"成功/失败将写入 episode_success（critic 的 r_T 监督信号）；\n"
            f"跳过则不评，后续由片段标注派生。")
        btn_ok = box.addButton("✓ 成功", QMessageBox.ButtonRole.YesRole)
        btn_bad = box.addButton("✗ 失败", QMessageBox.ButtonRole.NoRole)
        box.addButton("跳过(不评)", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is btn_ok:
            return True
        if clicked is btn_bad:
            return False
        return None

    def _finalize_rollout_stop(self, frames, result, success=None):
        """写入标注（区间+成败）与部署日志，恢复 UI。"""
        dataset_root = str(self._rollout_dataset.root) if self._rollout_dataset else ""
        if frames > 0 and dataset_root:
            try:
                # 接管区间以"推帧计数"为时钟，但停止录制 stop(flush=False) 会丢弃
                # 未写入队列（1~3 帧）、队列满时 push_frame 也会丢旧帧 → 实际保存
                # 帧数(frames) < 推帧计数。仍在接管中停止的区间 end 会越界
                # （超出部分本就未保存）。写入前按实际保存帧数钳位，杜绝越界
                # 干预区间（visualize_data 数据集检查问题8 的已知伪影来源）。
                n_clamped = 0
                clamped_ivs = []
                for iv in self._rollout_staged_intervals:
                    s, e = max(0, int(iv["start"])), int(iv["end"])
                    e2 = min(e, frames - 1)
                    if e2 < s:
                        n_clamped += 1  # 整段落在未保存区间 → 丢弃
                        logger.warning(
                            f"[RolloutRec] 暂存区间 [{s},{e}] 完全超出保存帧数 "
                            f"({frames})，已丢弃（帧从未写入数据集）")
                        continue
                    if e2 != e:
                        n_clamped += 1
                        logger.info(
                            f"[RolloutRec] 暂存区间 [{s},{e}] end 钳位到 {e2} "
                            f"（推帧计数领先实际保存 {e - e2} 帧，flush 丢帧）")
                        iv = dict(iv, end=e2)
                    clamped_ivs.append(iv)
                if n_clamped:
                    self._rollout_staged_intervals = clamped_ivs
                ep_idx = max(0, self._meta_episode_count(dataset_root) - 1)
                self._write_recap_annotations(
                    dataset_root, ep_idx, self._rollout_staged_intervals, success,
                    source="rollout",
                    policy_checkpoint=self._rollout_policy_tag,
                    group_id=self._rollout_active_group_id)
                self._append_deployment_log(
                    dataset_root, ep_idx, frames, self._rollout_staged_intervals, success,
                    policy_checkpoint=self._rollout_policy_tag,
                    group_id=self._rollout_active_group_id)
            except Exception as e:
                logger.warning(f"[RolloutRec] 标注/日志写入失败（数据已保存）: {e}")
        self.btn_lerobot.setEnabled(True)  # 互斥解除
        if self._rollout_record_thread is not None:
            self._rollout_record_thread.wait(2000)
            self._rollout_record_thread.deleteLater()
            self._rollout_record_thread = None
        self._rollout_dataset = None
        self._rollout_staged_intervals = []
        self._rollout_takeover_count = 0
        # 单集元数据随会话结束清空；组会话（epoch/label_key）保留——
        # 连续多次录制共享同组，直到文本/模型变化或点"新组"
        self._rollout_policy_tag = None
        self._rollout_active_group_id = None
        self.btn_rollout_record.setText("推理录制")
        self.btn_rollout_record.setStyleSheet(
            "font-size: 12px; padding: 4px 12px; background-color: #00695C; color: white;"
        )
        self.btn_rollout_record.setEnabled(True)
        succ_txt = {True: "成功", False: "失败", None: "未评"}[success]
        self.status_label.setText(f"推理录制已停止（{frames} 帧，评价: {succ_txt}）")
        if result:
            self.camera_log.emit(
                f"[RolloutRec] 保存成功: {result} 到 {dataset_root}（评价: {succ_txt}）")

    # ---------- RECAP 标注/部署日志写入（遥操作默认成功与推理录制共用） ----------

    def _load_annotation_store(self, dataset_root):
        """加载 visualize_data 的 AnnotationStore（懒导入，失败返回 None 不阻断录制）。"""
        try:
            vd_dir = Path(__file__).resolve().parent.parent / "visualize_data"
            if vd_dir.is_dir() and str(vd_dir) not in sys.path:
                sys.path.insert(0, str(vd_dir))
            from annotation_store import AnnotationStore
            store = AnnotationStore(str(dataset_root))
            store.load()
            return store
        except Exception as e:
            logger.warning(f"[LeRobot] AnnotationStore 加载失败（跳过标注写入）: {e}")
            return None

    @staticmethod
    def _meta_episode_count(dataset_root):
        """从 meta/episodes 统计 episode 数（保存完成后调用；失败返回 0）。"""
        try:
            import pyarrow.parquet as pq
            ep_dir = Path(dataset_root) / "meta" / "episodes"
            files = sorted(ep_dir.rglob("*.parquet"))
            return sum(pq.read_metadata(f).num_rows for f in files)
        except Exception as e:
            logger.warning(f"[LeRobot] 统计 episode 数失败: {e}")
            return 0

    def _write_recap_annotations(self, dataset_root, episode_index, intervals, success,
                                 source=None, policy_checkpoint=None, group_id=None):
        """把区间标注与 episode 成败写入 reward_annotations.json（直接生效，无确认制）。

        source：数据来源溯源（"teleop" 遥操作示范 / "rollout" 推理部署录制），
        合并后训练按来源配权重（防遗忘混训）用。
        policy_checkpoint / group_id：π*0.6 在线 RL 元数据（纯标注字段，不进
        parquet/模型输入；teleop 调用不传 → 保持 None=专家数据）。
        """
        store = self._load_annotation_store(dataset_root)
        if store is None:
            return
        for iv in intervals:
            try:
                store.add_interval(
                    episode_index, iv["start"], iv["end"], iv["label"],
                    iv.get("note", ""),
                    is_human_intervention=iv.get("is_human_intervention", False))
            except ValueError as e:
                logger.warning(f"[LeRobot] 标注区间被跳过（重叠）: {e}")
        if success is not None:
            store.set_episode_success(episode_index, success)
        if source is not None:
            store.set_episode_source(episode_index, source)
        # π*0.6 元数据：无条件写入（含 None=显式空），保证遥操作/推理两类录制
        # schema 统一——后续数据集融合、按字段筛选、编辑删除重映射零差异
        store.set_episode_policy(episode_index, policy_checkpoint or None)
        store.set_episode_group(episode_index, group_id or None)
        store.save()
        logger.info(
            f"[LeRobot] RECAP 标注已写入: ep{episode_index}, "
            f"区间 {len(intervals)} 个, episode_success={success}, source={source}, "
            f"policy={policy_checkpoint or '-'}, group={group_id or '-'}")

    def _append_deployment_log(self, dataset_root, episode_index, frames, intervals, success,
                               policy_checkpoint=None, group_id=None):
        """追加部署日志 CSV（北极星指标：干预率随训练轮数应下降）。

        π*0.6 R1a 新列：policy_checkpoint / group_id——按策略版本分组统计干预率、
        按组计算组相对优势的数据来源。旧日志文件自动迁移表头（旧行新列留空）。
        """
        import csv
        log_path = Path(dataset_root) / "meta" / "deployment_log.csv"
        itv = [iv for iv in intervals if iv.get("is_human_intervention")]
        itv_frames = sum(iv["end"] - iv["start"] + 1 for iv in itv)
        fields = ["timestamp", "episode_index", "frames", "takeover_count",
                  "intervention_frames", "intervention_ratio", "episode_success",
                  "policy_checkpoint", "group_id"]
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "episode_index": episode_index,
            "frames": frames,
            "takeover_count": len(itv),
            "intervention_frames": itv_frames,
            "intervention_ratio": f"{(itv_frames / frames) if frames else 0.0:.4f}",
            "episode_success": {True: 1, False: 0, None: -1}[success],
            "policy_checkpoint": policy_checkpoint or "",
            "group_id": group_id or "",
        }
        # 旧表头迁移：已存在文件缺新列时整表重写（旧行新列补空），防列错位
        write_header = not log_path.exists()
        if not write_header:
            try:
                with open(log_path, newline="", encoding="utf-8") as f:
                    existing_header = next(csv.reader(f), [])
                if existing_header and any(c not in existing_header for c in
                                           ("policy_checkpoint", "group_id")):
                    with open(log_path, newline="", encoding="utf-8") as f:
                        old_rows = list(csv.DictReader(f))
                    with open(log_path, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        w.writeheader()
                        for r in old_rows:
                            r.setdefault("policy_checkpoint", "")
                            r.setdefault("group_id", "")
                            w.writerow(r)
                    logger.info("[RolloutRec] deployment_log.csv 表头已升级"
                                "（新增 policy_checkpoint/group_id 列，旧行留空）")
                    write_header = False
                elif not existing_header:
                    write_header = True
            except Exception as e:
                logger.warning(f"[RolloutRec] 读取旧部署日志失败，直接追加: {e}")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerow(row)

    def cleanup(self):
        """清理资源，在关闭窗口时调用"""
        self._stop_rtc_executor()
        # 实时价值判断：停定时器 + 释放视觉塔显存
        if self._value_timer is not None:
            self._value_timer.stop()
        if self._value_load_thread is not None:
            self._value_load_thread.wait(2000)
            self._value_load_thread = None
        if self._value_judge is not None:
            self._value_judge.release()
            self._value_judge = None
        if self._vla_thread is not None:
            self._vla_thread.stop()
            self._vla_thread.wait(2000)
            self._vla_thread = None
            self._vla_running = False
        if self._lerobot_recording:
            self._stop_lerobot_record()
        else:
            if self._lerobot_timer is not None:
                self._lerobot_timer.stop()
            if self._lerobot_record_thread is not None:
                self._lerobot_record_thread.stop()
                self._lerobot_record_thread.wait(2000)
                self._lerobot_record_thread.deleteLater()
                self._lerobot_record_thread = None
            # 清理 dataset 引用，避免在关闭窗口时触发再次写入
            if self._lerobot_dataset is not None:
                self._lerobot_dataset = None
        # 推理录制清理（未在录制时也兜底线程引用）
        if self._rollout_recording:
            self._stop_rollout_record()
        elif self._rollout_record_thread is not None:
            self._rollout_record_thread.stop()
            self._rollout_record_thread.wait(2000)
            self._rollout_record_thread.deleteLater()
            self._rollout_record_thread = None
            self._rollout_dataset = None

        if self._display_timer is not None:
            self._display_timer.stop()

        # 停止所有采集线程
        for thread in self.capture_threads.values():
            thread.stop()
        self.capture_threads.clear()

        if self.is_recording:
            for writer in self.recorders.values():
                if writer is not None:
                    writer.release()
            self.recorders.clear()
            self.is_recording = False
        for cap in self.caps.values():
            if cap is not None:
                cap.release()
        self.caps.clear()
        self.latest_frames.clear()
        self._pending_frames.clear()
        logger.info("[Camera] 摄像头资源已清理")

    def closeEvent(self, event):
        """退出时释放资源"""
        self.cleanup()
        event.accept()


# 兼容旧接口：CameraWindow 作为独立窗口（基于 CameraWidget）
class CameraWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("USB 摄像头 - 双摄像头预览")
        self.resize(640, 480)

        layout = QVBoxLayout(self)
        self.camera_widget = CameraWidget(self)
        layout.addWidget(self.camera_widget)

    def cleanup(self):
        self.camera_widget.cleanup()

    def closeEvent(self, event):
        self.cleanup()
        event.accept()


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = CameraWindow()
    window.show()
    sys.exit(app.exec())

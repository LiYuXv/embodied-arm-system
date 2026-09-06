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
import threading
import math
from pathlib import Path
from queue import Queue, Empty, Full
from collections import deque

import numpy as np

logger = logging.getLogger("debugger")

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
    QCheckBox, QDoubleSpinBox, QTextEdit
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
    - 主线程只把 (rgb0, rgb1, pose7, task) 推入队列（<0.1ms）
    - 后台线程负责调用 dataset.add_frame（即使慢也不影响 UI）
    - 队列满时自动丢弃最旧帧，保证实时性
    - 停止时把队列剩余帧全部写入，再调用 save_episode
    """
    frame_written = pyqtSignal(int)   # 已写入帧数
    status = pyqtSignal(str)          # 状态文本
    finished_ok = pyqtSignal(str)     # 保存完成结果
    error = pyqtSignal(str)           # 错误信息

    def __init__(self, dataset, pose_callback, fps=20, max_queue=60,
                 torque_callback=None, action_lookahead=1, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.pose_callback = pose_callback
        self.torque_callback = torque_callback
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

    def push_frame(self, rgb0, rgb1):
        """主线程调用：快速把帧和位姿推入队列。如果队列满，丢弃最旧帧。"""
        pose7 = self._get_pose7()
        torque = self._get_torque()
        item = {"cam1": rgb0, "cam2": rgb1, "pose7": pose7,
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
                f"pose7=[{pose7[0]:.3f}, {pose7[1]:.3f}, {pose7[2]:.3f}, "
                f"{pose7[3]:.3f}, {pose7[4]:.3f}, {pose7[5]:.3f}, {pose7[6]:.3f}], "
                f"gripper_torque={float(torque[0]) if torque is not None and len(torque) else 0.0:.4f} Nm"
            )

    def _get_torque(self):
        """获取夹爪当前力矩 (Nm)，作为 1 维 float32 数组返回，失败返回 [0.0]。

        力矩是观测信号（传感器读数），与 observation.state 同属当前时刻，
        不做一帧延迟（延迟仅用于 action）。
        """
        if self.torque_callback is None:
            return np.zeros(1, dtype=np.float32)
        try:
            val = self.torque_callback()
        except Exception as e:
            logger.warning(f"[LeRobotThread] 获取夹爪力矩失败: {e}")
            return np.zeros(1, dtype=np.float32)
        if val is None:
            return np.zeros(1, dtype=np.float32)
        return np.array([round(float(val), 2)], dtype=np.float32)

    def _get_pose7(self):
        """获取 7 维位姿 [x,y,z,rx,ry,rz,gripper_rad]，失败返回 0 数组。

        直接使用末端位姿的欧拉角（弧度），不再转四元数；与 MoveL/EndPoseCtrl
        的欧拉角约定一致，推理端可直接用于逆解控制。
        """
        if self.pose_callback is None:
            return np.zeros(7, dtype=np.float32)
        try:
            result = self.pose_callback()
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

            def _write_frame(obs_item, action_pose7):
                # action 用 np.array(..., copy=True) 独立拷贝，避免与某个 item 的
                # pose7 共享内存导致后续被改写；state 同理。
                # torque 是观测信号，取 obs_item（当前帧），与 state 同时刻；
                # 只有 action 用未来第 N 帧位姿，避免 action==state 退化。
                frame_dict = {
                    "observation.images.cam1": obs_item["cam1"],
                    "observation.images.cam2": obs_item["cam2"],
                    "observation.state": np.array(obs_item["pose7"], dtype=np.float32, copy=True),
                    "action": np.array(action_pose7, dtype=np.float32, copy=True),
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

            def _after_write(obs_item, action_pose7):
                nonlocal consec_near_const, warned_const
                diff = float(np.max(np.abs(
                    np.asarray(action_pose7, dtype=np.float64)
                    - np.asarray(obs_item["pose7"], dtype=np.float64))))
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
                    _write_frame(obs_item, buf[lookahead - 1]["pose7"])
                    self._frame_count += 1
                    _after_write(obs_item, buf[lookahead - 1]["pose7"])
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
                _write_frame(obs_item, buf[k]["pose7"])
                self._frame_count += 1
                _after_write(obs_item, buf[k]["pose7"])
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
def _resolve_vcodec_for_recording():
    """选择最优视频编码器：优先 NVIDIA NVENC (H.264)，否则回退到软件 libx264。

    两者在 info.json 中的 canonical 编码都是 'h264'，因此可互相兼容。
    """
    try:
        import av
        for codec in ["h264_nvenc", "hevc_nvenc"]:
            try:
                av.codec.Codec(codec, "w")
                logger.info(f"[LeRobot] 使用硬件编码器: {codec}")
                return codec
            except Exception:
                pass
    except Exception:
        pass
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

    支持两种模式：
    - 普通模式：按固定频率逐帧调用 select_action，逐帧 emit action_ready
    - RTC 模式 (Real-Time Chunking)：被动等待 request_chunk() 请求，每次推理
      输出完整动作 chunk (H,7) 并 emit chunk_ready(chunk, latency_s)，由主线程
      RTC 执行器按控制频率异步消费，实现"边执行当前 chunk、边后台推理下一段"
      的连续运动，推理延迟不会打断动作流。
    """

    action_ready = pyqtSignal(object)        # np.ndarray (7,)
    chunk_ready = pyqtSignal(object, float)  # (np.ndarray (H,7), 推理耗时秒)
    single_chunk_ready = pyqtSignal(object, object)  # (np.ndarray (H,7), 观测信息 dict)
    status = pyqtSignal(str)
    model_loaded = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_path, task, get_obs, fps=20, rtc_mode=False,
                 get_torque=None, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.task = task
        self.get_obs = get_obs      # callable -> (rgb0, rgb1, state8) 或 None
        self.get_torque = get_torque  # callable -> float (夹爪力矩 Nm) 或 None
        self.fps = fps
        self.rtc_mode = rtc_mode
        # 多次采样平均：>1 时同一观测采 N 次 chunk 取均值（每次抽不同噪声），
        # 抑制 flow-matching 采样摆动；界面开关实时修改此值
        self.avg_n = 1
        self._running = True
        self._chunk_request = threading.Event()
        self._single_request = threading.Event()  # “推理一次”请求（两种模式通用）
        self._torque_key = None      # 模型声明力矩输入时为 "observation.gripper_torque"
        self._torque_warned = False

    def request_chunk(self):
        """RTC 模式：请求一次 chunk 推理（主线程调用，立即返回）"""
        self._chunk_request.set()

    def request_single_chunk(self):
        """“推理一次”：请求一次完整 chunk 推理并回传观测快照（主线程调用，立即返回）"""
        self._single_request.set()

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
            batch = {
                "observation.images.cam1": torch.from_numpy(
                    np.ascontiguousarray(rgb0)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device),
                "observation.images.cam2": torch.from_numpy(
                    np.ascontiguousarray(rgb1)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device),
                "observation.state": torch.from_numpy(
                    state.astype(np.float32)).unsqueeze(0).to(device),
                "task": [self.task],
            }
            if self._torque_key is not None:
                batch[self._torque_key] = self._torque_tensor(device)
            policy.reset()  # 单次诊断使用干净观测队列（n_obs_steps>1 时尤为关键）
            with torch.no_grad():
                batch_in = preprocessor(batch)
                if hasattr(policy, "predict_action_chunk"):
                    action = policy.predict_action_chunk(batch_in)
                else:
                    action = policy.select_action(batch_in)
                out = postprocessor({"action": action})
            c = out["action"] if isinstance(out, dict) else out
            c = c.detach().cpu().numpy().astype(np.float64)
            c = np.reshape(c, (-1, c.shape[-1]))  # (H, 7)
            info = {
                "rgb0": rgb0, "rgb1": rgb1,
                "state": np.asarray(state, dtype=np.float64).copy(),
                "torque": torque, "task": self.task,
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
            config = PreTrainedConfig.from_pretrained(str(ckpt))
            policy = get_policy_class(config.type).from_pretrained(str(ckpt))
            policy.to(device)
            policy.eval()

            # 多次采样平均包装（实例级，对 select_action / predict_action_chunk
            # 均生效——两版 lerobot 二者内部都汇聚到 _get_action_chunk）：
            # 同一观测采 avg_n 次取均值，逼近动作分布中心，实测 chunk 摆动 -30%
            orig_get_chunk = policy._get_action_chunk
            def _averaged_get_chunk(batch, noise=None, **kwargs):
                n = max(1, int(self.avg_n))
                if n == 1:
                    return orig_get_chunk(batch, noise, **kwargs)
                outs = [orig_get_chunk(batch, None, **kwargs) for _ in range(n)]
                return torch.stack(outs, dim=0).mean(dim=0)
            policy._get_action_chunk = _averaged_get_chunk
            policy.reset()  # 清空观测/动作队列（n_obs_steps>1 时避免陈旧历史）
            preprocessor = DataProcessorPipeline.from_pretrained(
                str(ckpt), config_filename="policy_preprocessor.json")
            postprocessor = DataProcessorPipeline.from_pretrained(
                str(ckpt), config_filename="policy_postprocessor.json")
            # 力矩输入：仅当模型（checkpoint config）声明了该输入特征时才传入，
            # 与 test_inference_episode.py 判定逻辑一致；旧版 smolvla 行为不变
            torque_key = getattr(config, "torque_key", "observation.gripper_torque")
            input_features = getattr(config, "input_features", None) or {}
            self._torque_key = torque_key if torque_key in input_features else None
            if self._torque_key and self.get_torque is None:
                self.status.emit("警告: 模型需要力矩输入，但未设置力矩回调，将以 0 填充")
            self.model_loaded.emit()
            torque_note = f"，力矩输入: {self._torque_key}" if self._torque_key else ""
            self.status.emit(f"模型加载完成 ({device}){torque_note}，推理中...")
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())
            return

        if self.rtc_mode:
            self._rtc_inference_loop(policy, preprocessor, postprocessor, device)
            return

        period = 1.0 / max(self.fps, 1)
        while self._running:
            if self._single_request.is_set():
                self._single_request.clear()
                self._run_single_shot(policy, preprocessor, postprocessor, device)
                continue
            t0 = time.perf_counter()
            try:
                obs = self.get_obs()
            except Exception:
                obs = None
            if obs is not None and self._running:
                try:
                    rgb0, rgb1, state = obs
                    batch = {
                        "observation.images.cam1": torch.from_numpy(
                            np.ascontiguousarray(rgb0)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device),
                        "observation.images.cam2": torch.from_numpy(
                            np.ascontiguousarray(rgb1)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device),
                        "observation.state": torch.from_numpy(
                            state.astype(np.float32)).unsqueeze(0).to(device),
                        "task": [self.task],
                    }
                    if self._torque_key is not None:
                        batch[self._torque_key] = self._torque_tensor(device)
                    with torch.no_grad():
                        batch = preprocessor(batch)
                        action = policy.select_action(batch)
                        out = postprocessor({"action": action})
                    a = out["action"] if isinstance(out, dict) else out
                    a = a.squeeze(0).detach().cpu().numpy().astype(np.float64)
                    if self._running:
                        self.action_ready.emit(a)
                except Exception as e:
                    self.status.emit(f"推理帧失败: {e}")
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)

    def _rtc_inference_loop(self, policy, preprocessor, postprocessor, device):
        """RTC 模式：按需推理完整动作 chunk，推理期间不阻塞动作执行。

        每次 request_chunk() 触发一次推理，输出整个 action chunk (H,7) 及实测
        推理延迟。主线程 RTC 执行器根据该延迟决定提前多少步触发下一次推理，
        从而保证动作队列始终不为空（对应 RTC 论文中的异步执行 + 延迟补偿 d）。
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
                continue
            try:
                t0 = time.perf_counter()
                rgb0, rgb1, state = obs
                batch = {
                    "observation.images.cam1": torch.from_numpy(
                        np.ascontiguousarray(rgb0)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device),
                    "observation.images.cam2": torch.from_numpy(
                        np.ascontiguousarray(rgb1)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device),
                    "observation.state": torch.from_numpy(
                        state.astype(np.float32)).unsqueeze(0).to(device),
                    "task": [self.task],
                }
                if self._torque_key is not None:
                    batch[self._torque_key] = self._torque_tensor(device)
                with torch.no_grad():
                    batch = preprocessor(batch)
                    # 直接取完整动作 chunk（RTC 需要整段轨迹做新旧融合）；
                    # 不支持的策略退化为单步动作
                    if hasattr(policy, "predict_action_chunk"):
                        action = policy.predict_action_chunk(batch)
                    else:
                        action = policy.select_action(batch)
                    out = postprocessor({"action": action})
                c = out["action"] if isinstance(out, dict) else out
                c = c.detach().cpu().numpy().astype(np.float64)
                c = np.reshape(c, (-1, c.shape[-1]))  # (B,H,D) 或 (B,D) -> (H,D)
                latency = time.perf_counter() - t0
                if self._running:
                    self.chunk_ready.emit(c, latency)
            except Exception as e:
                self.status.emit(f"RTC chunk 推理失败: {e}")


class CameraWidget(QWidget):
    """可嵌入的摄像头监控组件（双路摄像头）—— 优化版，支持 LeRobot v0.3 数据采集"""

    # 信号：降频到10Hz发射，用于后续lerobot数据集收集
    frame_captured = pyqtSignal(int, object)  # (camera_idx, frame_ndarray)

    def __init__(self, parent=None, save_dir=None, dataset_root=None):
        super().__init__(parent)

        # 分辨率选项 (名称, 宽度, 高度)：只保留 LeRobot 录制所需的采集分辨率
        # （720x720 中心裁切同样来自 1280x720 采集）
        self.resolutions = [
            ("1280 x 720 (推荐)", 1280, 720),
            ("640 x 480", 640, 480),
        ]
        # 默认采集分辨率：必须与 LeRobot 默认分辨率保持一致，否则会出现
        # “设置 1280x720，实际录制的是从 320x240 放大上来的模糊画面”的问题。
        self.capture_res = (1280, 720)

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

        # 对焦配置：默认关闭自动对焦，固定焦距 0.5（归一化值，0.0~1.0）
        self._autofocus_enabled = False
        self._focus_value = 0.5

        # 曝光锁定配置：默认勾选；勾选瞬间读取摄像头当前曝光值并全程锁定，
        # 避免录制过程中自动曝光漂移导致画面明暗变化，影响数据集一致性
        self._exposure_locked = True
        self._locked_exposure = {}  # {camera_idx: 锁定的曝光值}

        # 显示降频控制：限制UI刷新到15fps
        self._display_timer = None
        self._pending_frames = {}
        self._last_emit_time = {}

        # LeRobot 采集状态
        self._lerobot_recording = False
        self._lerobot_dataset = None
        self._lerobot_frame_count = 0
        self._lerobot_get_end_pose = None  # callback(end_pose_dict) -> ArmEndPose-like
        self._lerobot_get_torque = None    # callback() -> float (夹爪力矩 Nm) 或 None
        self._lerobot_task_name = ""
        self._lerobot_repo_name = "ela3_demo"
        self._lerobot_fps = 20
        self._lerobot_image_size = (720, 720)
        # action 标签前瞻帧数（action[t]=pose[t+N]），界面“前瞻帧”可调；
        # 5~10 增大每步目标增量、提高信噪比，抑制模型输出摆动
        self._action_lookahead = 5
        self._lerobot_record_thread = None  # 后台录制线程

        # VLA 推理状态
        self._vla_thread = None
        self._vla_running = False
        self._vla_control_paused = False  # True: 模型仍在运行，但不把动作下发给机械臂
        self._motion_cb = None       # fn(cmd, *args)，由主窗口转发到机械臂 worker
        self._vla_max_step = 0.05    # 每步末端位移限幅 (m)，防止模型异常输出导致猛冲
        self._vla_lowpass_enabled = True  # 动作输出 EMA 低通开关（界面复选框，默认开启）
        self._vla_avg_n = 1          # 多次采样平均次数（界面复选框，默认 1=关闭）

        # “推理一次”单次执行诊断状态
        self._single_active = False      # 单次 chunk 正在下发执行
        self._single_queue = deque()     # 待执行的单次 chunk 动作
        self._single_timer = None        # 按 fps 消费单次队列的定时器
        self._single_exec_log = []       # 执行期间 (目标, 反馈位姿) 记录，用于分析往复运动
        self._single_save_dir = None     # 本次诊断数据保存目录

        # RTC (Real-Time Chunking) 异步推理状态
        self._rtc_mode = False       # 本次推理是否启用 RTC（启动时读取复选框）
        self._rtc_timer = None       # 按 fps 消费动作队列的执行定时器
        self._rtc_queue = deque()    # 待执行动作队列（绝对位姿 7 维）
        self._rtc_inflight = False   # 是否有一次 chunk 推理正在进行
        self._rtc_delay_steps = 6    # 延迟补偿步数 d（按实测推理延迟自适应更新）
        self._rtc_chunk_count = 0    # 已接收 chunk 数（状态显示用）
        self._rtc_last_sent = None   # 上一次 RTC 下发的 6 维位姿目标（EMA/限幅基准）

        # RTC 可调参数（界面 RTC 参数行，实时生效）
        self._rtc_max_step_mm = 5.0    # 每步末端位置限幅 (mm/步)，限制最大速度
        self._rtc_max_rot_step = 0.05  # 每步姿态限幅 (rad/步)
        self._rtc_smooth_alpha = 0.5   # 下发目标 EMA 低通系数 (0~1，越小越平滑但滞后越大)
        self._rtc_blend_k = 3.0        # 新旧 chunk 软融合强度 k（w=1-exp(-k(j+1)/N)）
        self._rtc_delay_margin = 1     # 延迟裕量（步）：d = ceil(延迟*fps) + 裕量
        self._rtc_overlap_steps = 10   # 目标重叠步数 O：队列剩 d+O 步即触发推理，
                                       # 保证新 chunk 到达时旧队列仍有 O 步可软融合
        self._rtc_interp = "linear"    # 伺服插值方式：none/linear

        # 外部控制互锁回调：当 VLA 与手柄/其他控制源需要切换时，通知外部模块
        self._on_vla_control_pause_changed = None  # callable(paused: bool)
        self._on_vla_takeover_requested = None     # callable() -> 请求外部释放控制权

        # LeRobot 录制定时器：控制采集频率，主线程只负责把帧推入队列
        self._lerobot_timer = QTimer(self)
        self._lerobot_timer.timeout.connect(self._lerobot_push_frame)
        self._lerobot_timer.setTimerType(Qt.TimerType.PreciseTimer)

        self._setup_ui()

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

        # 两个视频显示标签，横向排列
        video_layout = QHBoxLayout()
        self.video_labels = []
        for i in range(2):
            label = QLabel(f"摄像头 {i + 1}\n未连接")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background-color: #222; color: #fff; font-size: 12px;")
            label.setMinimumSize(240, 180)
            label.setScaledContents(False)
            self.video_labels.append(label)
            video_layout.addWidget(label)
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

        # 锁定曝光开关（默认勾选）：勾选瞬间读取各摄像头当前曝光值并全程锁定；
        # 取消勾选则恢复自动曝光，并清空已记录的锁定值
        self.chk_exposure_lock = QCheckBox("锁定曝光")
        self.chk_exposure_lock.setStyleSheet("font-size: 11px;")
        self.chk_exposure_lock.setChecked(self._exposure_locked)
        self.chk_exposure_lock.setToolTip(
            "锁定曝光（默认勾选）：\n"
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
        control_layout.addWidget(self.btn_lerobot)
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

        # RTC (Real-Time Chunking) 异步推理开关：勾选后启动推理时，模型按需输出
        # 完整动作 chunk，执行器边执行边在后台推理下一段，新旧 chunk 以指数衰减
        # 权重软融合（Soft Masking），消除 chunk 切换处的停顿/跳变，实现连续平滑运动
        self.chk_rtc = QCheckBox("RTC异步")
        self.chk_rtc.setStyleSheet("font-size: 11px; color: #9C27B0; font-weight: bold;")
        self.chk_rtc.setToolTip(
            "Real-Time Chunking 异步推理：\n"
            "- 边执行当前动作 chunk，边后台推理下一段\n"
            "- 按实测推理延迟提前触发下一次推理，动作流不中断\n"
            "- 新旧 chunk 重叠区以指数衰减权重软融合，消除切换跳变"
        )
        vla_layout.addWidget(self.chk_rtc)

        # 动作输出 EMA 低通开关（默认开启）：抑制 flow-matching 采样噪声与
        # chunk 边界跳变造成的真机抖动；普通/RTC 两种模式均生效
        self.chk_lowpass = QCheckBox("低通滤波")
        self.chk_lowpass.setChecked(True)
        self.chk_lowpass.setStyleSheet("font-size: 11px; color: #2E7D32; font-weight: bold;")
        self.chk_lowpass.setToolTip(
            "对模型输出动作做 EMA 低通（out = α*新动作 + (1-α)*上次目标）：\n"
            "- 抑制高频采样噪声与 chunk 边界跳变，真机更平稳\n"
            "- 平滑系数 α 用下方 RTC 参数行的\"平滑α\"调节（默认 0.5）\n"
            "- 关闭后动作直发，仅保留安全限幅"
        )
        self.chk_lowpass.toggled.connect(
            lambda on: setattr(self, "_vla_lowpass_enabled", bool(on)))
        vla_layout.addWidget(self.chk_lowpass)

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
            "每步末端位置限幅 (mm/步)：限制最大运动速度。\n"
            "值越小越稳但跟踪越慢；5mm@20Hz ≈ 0.1m/s",
            lambda v: setattr(self, "_rtc_max_step_mm", float(v)))
        rtc_layout.addWidget(self.spin_rtc_step)

        rtc_layout.addWidget(QLabel("姿态rad:"))
        self.spin_rtc_rot = _rtc_spin(
            0.005, 0.5, 0.005, 3, self._rtc_max_rot_step,
            "每步姿态限幅 (rad/步)：限制姿态变化速度，抑制姿态抖动",
            lambda v: setattr(self, "_rtc_max_rot_step", float(v)))
        rtc_layout.addWidget(self.spin_rtc_rot)

        rtc_layout.addWidget(QLabel("平滑α:"))
        self.spin_rtc_alpha = _rtc_spin(
            0.05, 1.0, 0.05, 2, self._rtc_smooth_alpha,
            "下发目标 EMA 低通系数：out = α*新目标 + (1-α)*上次目标。\n"
            "越小越平滑但滞后越大；1.0 = 不平滑",
            lambda v: setattr(self, "_rtc_smooth_alpha", float(v)))
        rtc_layout.addWidget(self.spin_rtc_alpha)

        rtc_layout.addWidget(QLabel("融合k:"))
        self.spin_rtc_blend = _rtc_spin(
            0.5, 10.0, 0.5, 1, self._rtc_blend_k,
            "新旧 chunk 软融合强度 k：w = 1-exp(-k*(j+1)/N)。\n"
            "越大越快信任新 chunk，越小过渡越平缓",
            lambda v: setattr(self, "_rtc_blend_k", float(v)))
        rtc_layout.addWidget(self.spin_rtc_blend)

        rtc_layout.addWidget(QLabel("延迟裕量:"))
        self.spin_rtc_margin = QDoubleSpinBox()
        self.spin_rtc_margin.setRange(0, 20)
        self.spin_rtc_margin.setSingleStep(1)
        self.spin_rtc_margin.setDecimals(0)
        self.spin_rtc_margin.setValue(self._rtc_delay_margin)
        self.spin_rtc_margin.setFixedWidth(52)
        self.spin_rtc_margin.setStyleSheet("font-size: 11px; padding: 1px 2px;")
        self.spin_rtc_margin.setToolTip(
            "延迟裕量（步）：d = ceil(推理延迟*fps) + 裕量。\n"
            "队列剩余 ≤ d 步时触发下一次推理；裕量越大触发越早")
        self.spin_rtc_margin.valueChanged.connect(
            lambda v: setattr(self, "_rtc_delay_margin", int(v)))
        rtc_layout.addWidget(self.spin_rtc_margin)

        rtc_layout.addWidget(QLabel("重叠步:"))
        self.spin_rtc_overlap = QDoubleSpinBox()
        self.spin_rtc_overlap.setRange(0, 60)
        self.spin_rtc_overlap.setSingleStep(1)
        self.spin_rtc_overlap.setDecimals(0)
        self.spin_rtc_overlap.setValue(self._rtc_overlap_steps)
        self.spin_rtc_overlap.setFixedWidth(52)
        self.spin_rtc_overlap.setStyleSheet("font-size: 11px; padding: 1px 2px;")
        self.spin_rtc_overlap.setToolTip(
            "目标重叠步数 O：队列剩余 ≤ d+O 步时即触发下一次推理。\n"
            "保证新 chunk 到达时旧队列仍剩约 O 步用于软融合，\n"
            "消除 chunk 交接处的硬切换抖动。过小→交接抖动，过大→推理过频")
        self.spin_rtc_overlap.valueChanged.connect(
            lambda v: setattr(self, "_rtc_overlap_steps", int(v)))
        rtc_layout.addWidget(self.spin_rtc_overlap)

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
        """后台线程检测到可用索引后，在主线程中打开摄像头并启动采集线程"""
        self.camera_indices = indices
        cam_count = len(indices)
        logger.info(f"[Camera] 检测到 {cam_count} 个摄像头: {indices}")

        for idx in indices[:2]:
            self.open_camera(idx, *self.capture_res)

        if cam_count == 0:
            self.status_label.setText("未检测到摄像头")
            self.video_labels[0].setText("摄像头 1\n未连接")
            self.video_labels[1].setText("摄像头 2\n未连接")
        elif cam_count == 1:
            idx = indices[0]
            self.status_label.setText(f"已连接 1 个摄像头 (索引 {idx})")
            self.video_labels[1].setText("摄像头 2\n未连接")
        else:
            idx1, idx2 = indices[0], indices[1]
            self.status_label.setText(f"已连接 2 个摄像头 (索引 {idx1}, {idx2})")

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

    def _refresh_display(self):
        """统一显示刷新：15fps，将缓存的帧转换为QPixmap显示。

        显示前按 LeRobot 目标分辨率做与录制一致的中心裁切，实现
        “所见即所录”：例如 LeRobot 选择 720x720 时，界面也显示
        从 1280x720 画面中心裁出的方形区域。
        """
        for i, idx in enumerate(self.camera_indices[:2]):
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

        for idx in self.camera_indices[:2]:
            if idx in self.caps:
                self.open_camera(idx, w, h)

    def take_snapshot(self):
        saved = []
        for idx in self.camera_indices[:2]:
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

            for idx in self.camera_indices[:2]:
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

    # ==================== VLA 推理 ====================

    def _browse_vla_model(self):
        path = QFileDialog.getExistingDirectory(self, "选择模型 checkpoint 目录", ".")
        if path:
            self.vla_model_edit.setText(path)
            self._sync_image_size_from_model(path)
            self._refresh_task_candidates()

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
        rgb0 = self._resize_frame(f0, *self._lerobot_image_size)
        rgb1 = self._resize_frame(f1, *self._lerobot_image_size)
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
            logger.info(f"[VLA] 从模型配置同步图像分辨率: {w}x{h}")
            for i, (name, ow, oh) in enumerate(self.lerobot_res_options):
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
        if not model_path:
            QMessageBox.warning(self, "提示", "请填写模型路径")
            return
        # 推理分辨率必须与训练数据一致：从模型 config.json 自动同步（先于采集分辨率检查）
        self._sync_image_size_from_model(model_path)
        # 保证摄像头采集分辨率不低于训练分辨率
        self._ensure_capture_resolution()
        self._rtc_mode = self.chk_rtc.isChecked()
        self.chk_rtc.setEnabled(False)  # 运行期间锁定模式，停止后可重新选择
        self._vla_thread = VLAInferenceThread(
            model_path, task, self._vla_get_obs, fps=self._lerobot_fps,
            rtc_mode=self._rtc_mode, get_torque=self._lerobot_get_torque,
            parent=self)
        self._vla_thread.avg_n = self._vla_avg_n  # 同步界面预设的采样平均次数
        self._vla_thread.action_ready.connect(self._on_vla_action)
        self._vla_thread.chunk_ready.connect(self._on_rtc_chunk)
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
        self.status_label.setText("[VLA] 正在加载模型...")
        self._vla_thread.start()

        # 低通 EMA 基准复位（普通/RTC 模式共用）：从当前位姿起步，避免首帧跳变
        self._rtc_last_sent = None

        if self._rtc_mode:
            # 初始化 RTC 执行器：清空队列，立即请求首个 chunk，
            # 并按控制频率启动消费定时器（队列空时定时器空转等待）
            self._rtc_queue.clear()
            self._rtc_inflight = True
            self._rtc_chunk_count = 0
            self._vla_thread.request_chunk()
            if self._rtc_timer is None:
                self._rtc_timer = QTimer(self)
                self._rtc_timer.setTimerType(Qt.TimerType.PreciseTimer)
                self._rtc_timer.timeout.connect(self._on_rtc_tick)
            self._rtc_timer.start(int(1000 / self._lerobot_fps))
            logger.info(f"[VLA-RTC] 异步推理模式已启动，执行频率 {self._lerobot_fps}Hz")

    def _on_avg3_toggled(self, on):
        """3次采样平均开关：实时修改运行中推理线程的采样次数。"""
        self._vla_avg_n = 3 if on else 1
        if self._vla_thread is not None:
            self._vla_thread.avg_n = self._vla_avg_n
        self.status_label.setText(
            f"[VLA] 多次采样平均: {'开启 (3次)' if on else '关闭'}")

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
        if self._vla_control_paused:
            self.btn_vla_pause.setText("继续控制")
            self.btn_vla_pause.setStyleSheet(
                "font-size: 12px; padding: 4px 12px; background-color: #4CAF50; color: white;"
            )
            self.status_label.setText("[VLA] 控制已暂停（手柄可接管）")
            logger.info("[VLA] 控制已暂停，模型仍在运行但不向机械臂下发动作")
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
        # 立即标记停止，_on_vla_action 不再下发新动作
        self._vla_running = False
        self.btn_vla.setEnabled(False)
        self.btn_vla_once.setEnabled(False)
        self.status_label.setText("[VLA] 正在停止...")
        self._stop_rtc_executor()
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
        """停止 RTC 执行定时器并清空动作队列"""
        if self._rtc_timer is not None:
            self._rtc_timer.stop()
        self._rtc_queue.clear()
        self._rtc_inflight = False

    def _on_vla_finished(self):
        self._vla_running = False
        self._stop_rtc_executor()
        self.chk_rtc.setEnabled(True)
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

    def _on_vla_action(self, a):
        """收到模型动作后下发到机械臂（主线程槽函数）。

        模型输出 7 维 [x,y,z,rx,ry,rz,gripper_rad]，与训练数据一致（欧拉角+夹爪，
        均为弧度）。姿态直接用欧拉角，借 end_pose_ctrl→EndPoseCtrl 做逆解控制，
        无需任何四元数转换。
        """
        if not self._vla_running or self._motion_cb is None:
            return
        if self._single_active:
            # 单次执行独占机械臂，忽略连续推理的动作
            return
        if self._vla_control_paused:
            # 模型仍在运行，但暂时不把动作下发给机械臂，以便手柄等其他控制源接管
            logger.debug(f"[VLA] 控制已暂停，忽略动作: {a[:3]}...")
            return
        self._dispatch_vla_action(a)

    def _dispatch_vla_action(self, a):
        """把单步 7 维动作限幅后通过 EndPoseCtrl / GripperCtrl 下发。

        普通模式由 _on_vla_action 直接调用；RTC 模式由执行定时器逐步调用。
        """
        try:
            x, y, z, rx, ry, rz, grip_rad = [float(v) for v in a[:7]]

            # 每步位移限幅：以当前位姿为基准，防止模型异常输出导致猛冲
            cur = self._get_pose_7d()

            if self._vla_lowpass_enabled or self._rtc_mode:
                # 平滑链路（借鉴旧模式 S-curve 的平滑效果）：
                # 1) 低通开启时对下发目标做 EMA，抑制采样噪声/台阶抖动
                #    （普通/RTC 模式均生效；关闭后仅保留安全限幅）
                # 2) RTC 模式相对上一次下发目标做位置/姿态步长限幅，严格限速
                tgt = np.array([x, y, z, rx, ry, rz], dtype=np.float64)
                if self._rtc_last_sent is None:
                    self._rtc_last_sent = np.asarray(cur[:6], dtype=np.float64)
                if self._vla_lowpass_enabled:
                    alpha = self._rtc_smooth_alpha
                    tgt = alpha * tgt + (1.0 - alpha) * self._rtc_last_sent

                if self._rtc_mode:
                    step = tgt[:3] - self._rtc_last_sent[:3]
                    n = float(np.linalg.norm(step))
                    max_step = self._rtc_max_step_mm / 1000.0
                    if n > max_step:
                        tgt[:3] = self._rtc_last_sent[:3] + step * (max_step / n)
                    rot = tgt[3:] - self._rtc_last_sent[3:]
                    rn = float(np.linalg.norm(rot))
                    if rn > self._rtc_max_rot_step:
                        tgt[3:] = self._rtc_last_sent[3:] + rot * (self._rtc_max_rot_step / rn)
                self._rtc_last_sent = tgt
                x, y, z, rx, ry, rz = [float(v) for v in tgt]

            d = np.array([x - cur[0], y - cur[1], z - cur[2]])
            n = float(np.linalg.norm(d))
            if n > self._vla_max_step:
                s = self._vla_max_step / n
                x, y, z = cur[0] + d[0] * s, cur[1] + d[1] * s, cur[2] + d[2] * s

            # 夹爪弧度限幅：[-π/2, π/60]
            grip_rad = float(np.clip(grip_rad, -math.pi / 2.0, math.pi / 60.0))

            # block=False：伺服模式异步下发目标轨迹，由控制循环执行，
            # 避免 worker 阻塞等待导致命令队列积压、失能/停止指令失效。
            # end_pose_ctrl 接收 (x,y,z,rx,ry,rz,duration,block)，内部由
            # arm.EndPoseCtrl 做笛卡尔逆解（与 MoveL 页面同一接口）。
            # RTC 模式下 servo=True：SDK 按选定插值方式（线性/B样条/S曲线/不插值）
            # 把 20Hz 离散目标在关节空间上采样为 200Hz 平滑设定点流，
            # 段间连续，消除台阶式抖动。
            self._motion_cb("end_pose_ctrl", x, y, z, rx, ry, rz,
                            1.0 / self._lerobot_fps, False, self._rtc_mode,
                            self._rtc_interp)
            # 夹爪已是弧度，GripperCtrl 接口接收弧度，无需转换
            self._motion_cb("gripper_ctrl", grip_rad)
        except Exception as e:
            logger.warning(f"[VLA] 动作下发失败: {e}")

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
            "smooth_alpha": self._rtc_smooth_alpha,
            "avg_n": self._vla_avg_n,
            "n_frames": int(chunk.shape[0]),
            "columns": ["x", "y", "z", "rx", "ry", "rz", "gripper_rad"],
        }
        with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        np.savetxt(save_dir / "action_chunk.csv", chunk, delimiter=",",
                   header="x,y,z,rx,ry,rz,gripper_rad", comments="")
        self._single_save_dir = save_dir

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
        """单次执行定时器：逐帧下发 chunk 动作并记录目标/反馈位姿。"""
        if not self._single_queue:
            self._finish_single()
            return
        a = self._single_queue.popleft()
        self._dispatch_vla_action(a)
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

    # ==================== RTC (Real-Time Chunking) 异步执行器 ====================

    def _on_rtc_tick(self):
        """RTC 执行定时器（主线程，fps Hz）：从动作队列取出下一步下发。

        关键点（对应 RTC 论文）：
        - 动作执行与模型推理解耦：只要队列非空就连续下发，推理在后台并行进行，
          推理延迟不会打断动作流。
        - 延迟补偿：队列剩余步数 ≤ d（实测推理延迟换算的步数）时提前触发
          下一次 chunk 推理，保证旧 chunk 耗尽前新 chunk 已到达。
        - 队列为空时保持不动（不弹出、不下发），等待新 chunk，由底层 200Hz
          控制循环保持当前目标位姿。
        """
        if not self._vla_running or self._motion_cb is None:
            return
        if self._vla_control_paused:
            # 暂停期间不消耗队列也不触发推理，恢复后从原队列继续执行
            return

        # 提前触发：队列剩余 ≤ d+O 步时发起推理。推理耗时约 d 步，
        # 新 chunk 到达时旧队列恰好还剩约 O 步，保证软融合有足够的
        # 重叠区，避免 chunk 交接处硬切换造成的一小段一小段卡顿
        if (not self._rtc_inflight and self._vla_thread is not None
                and len(self._rtc_queue) <= self._rtc_delay_steps + self._rtc_overlap_steps):
            self._rtc_inflight = True
            self._vla_thread.request_chunk()

        if not self._rtc_queue:
            return
        a = self._rtc_queue.popleft()
        self._dispatch_vla_action(a)

    def _on_rtc_chunk(self, chunk, latency):
        """收到新动作 chunk（主线程槽函数）：与旧队列余量做软融合后替换队列。

        RTC 三区域近似的推理时实现（无需重训练）：
        - Frozen 区：新 chunk 前 d 步对应推理期间已流逝的时间，直接丢弃，
          这些时刻已由旧 chunk 覆盖执行。
        - Intermediate 区：新 chunk 第 d 步起与旧队列余量的重叠段，用指数
          递增权重 w=1-exp(-3(j+1)/overlap) 软融合（Soft Masking），仅对位置/
          姿态前 6 维插值，夹爪维度不插值、直接采用新 chunk 值，
          避免多模态分布下硬切换导致的跳变/保护停。
        - New 区：重叠段之后的新 chunk 尾部直接接管。
        """
        self._rtc_inflight = False
        if not self._vla_running:
            return
        try:
            rows = [np.asarray(r, dtype=np.float64) for r in chunk]
        except Exception:
            return
        if not rows:
            return

        fps = max(self._lerobot_fps, 1)
        # 按实测推理延迟自适应更新 d（+安全裕量），并限制在 chunk 长度内
        d = int(math.ceil(latency * fps)) + self._rtc_delay_margin
        d = max(1, min(d, max(1, len(rows) - 1)))
        self._rtc_delay_steps = d

        if not self._rtc_queue:
            # 首个 chunk（或队列已耗尽）：直接从 d 之后开始执行，
            # 前 d 步基于请求时刻的旧观测，已过期不可信
            self._rtc_queue.extend(rows[d:] if len(rows) > d else rows)
        else:
            old = list(self._rtc_queue)
            head = rows[d:] if len(rows) > d else rows[-1:]
            overlap = min(len(old), len(head))
            k = self._rtc_blend_k
            merged = []
            for j in range(overlap):
                # Soft Masking：指数递增权重，越靠后越信任新 chunk。
                # 只对位置/姿态（前6维）融合；夹爪（第7维）不参与插值，
                # 直接取新 chunk 的阶跃值，避免开合动作被新旧融合拖慢
                w = 1.0 - math.exp(-k * (j + 1) / max(overlap, 1))
                m = w * head[j] + (1.0 - w) * old[j]
                m[6] = head[j][6]
                merged.append(m)
            merged.extend(head[overlap:])
            self._rtc_queue.clear()
            self._rtc_queue.extend(merged)

        self._rtc_chunk_count += 1
        self.status_label.setText(
            f"[VLA-RTC] chunk #{self._rtc_chunk_count}，"
            f"推理 {latency * 1000:.0f}ms，d={d}，队列 {len(self._rtc_queue)} 步"
        )
        logger.info(
            f"[VLA-RTC] chunk #{self._rtc_chunk_count} 到达: "
            f"len={len(rows)}, latency={latency * 1000:.0f}ms, d={d}, "
            f"queue={len(self._rtc_queue)}"
        )

    def _ensure_lerobot_dataset(self):
        """初始化或加载 LeRobot v3.0 数据集 (适配 lerobot 0.4.4)

        策略：
        1. 文件夹不存在：直接创建新数据集
        2. 文件夹存在且格式兼容：加载已有数据集，追加 episode
        3. 文件夹存在但格式不兼容：返回字符串 "incompatible"，
           由调用方弹窗提示用户修改文件夹名
        """
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except Exception as e:
            logger.error(f"[LeRobot] 无法导入 LeRobotDataset: {e}")
            return None

        dataset_dir = os.path.join(self.dataset_root, self._lerobot_repo_name)

        # 当前目标配置
        target_h, target_w = self._lerobot_image_size
        target_cam_dtype = "video"  # 使用视频编码
        target_video_codec = "h264"  # canonical 名称，对应 h264_nvenc / libx264
        target_state_shape = (7,)
        target_action_shape = (7,)

        # 编码器（流式编码用）：硬件优先 NVENC，否则软件 libx264。
        # 流式编码把 add_frame 的帧直接喂给 PyAV 编码器（后台线程），
        # 不再落盘 PNG，从根本上消除 PNG 编/解码这一主要瓶颈。
        vcodec = _resolve_vcodec_for_recording()
        encoder_threads = None if vcodec in ("h264_nvenc", "hevc_nvenc") else 4

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

                cam1_feature = existing_features.get("observation.images.cam1", {})
                cam2_feature = existing_features.get("observation.images.cam2", {})
                cam1_shape = cam1_feature.get("shape", [0, 0, 0])
                cam2_shape = cam2_feature.get("shape", [0, 0, 0])
                cam1_dtype = cam1_feature.get("dtype", "image")
                cam2_dtype = cam2_feature.get("dtype", "image")
                cam1_codec = cam1_feature.get("info", {}).get("video.codec", "")
                cam2_codec = cam2_feature.get("info", {}).get("video.codec", "")

                # 夹爪力矩特征必须存在，否则 add_frame 写入 observation.gripper_torque 会失败
                has_torque = "observation.gripper_torque" in existing_features

                compatible = (
                    tuple(state_shape) == target_state_shape and
                    tuple(action_shape) == target_action_shape and
                    cam1_dtype == target_cam_dtype and
                    cam2_dtype == target_cam_dtype and
                    cam1_codec == target_video_codec and
                    cam2_codec == target_video_codec and
                    tuple(cam1_shape) == (target_h, target_w, 3) and
                    tuple(cam2_shape) == (target_h, target_w, 3) and
                    has_torque
                )

                if not compatible:
                    logger.warning(
                        f"[LeRobot] 数据集 '{self._lerobot_repo_name}' 格式不兼容 "
                        f"(state={state_shape}, action={action_shape}, "
                        f"cam_dtype={cam1_dtype}/{cam2_dtype}, "
                        f"cam_codec={cam1_codec}/{cam2_codec}, "
                        f"cam_shape={cam1_shape}/{cam2_shape}, torque={has_torque})"
                    )
                    return "incompatible"
                else:
                    # 兼容，加载已有数据集
                    import os as os_module
                    old_hf_offline = os_module.environ.get("HF_HUB_OFFLINE")
                    os_module.environ["HF_HUB_OFFLINE"] = "1"
                    if hasattr(LeRobotDataset, "resume"):
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

        # LeRobot 0.4.4 (v3.0) features 格式 - 摄像头使用 video dtype 才能配合 use_videos=True
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
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "observation.gripper_torque": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["torque"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],
            },
        }

        try:
            # 设置离线模式避免联网
            import os as os_module
            old_hf_offline = os_module.environ.get("HF_HUB_OFFLINE")
            os_module.environ["HF_HUB_OFFLINE"] = "1"

            # 编码参数按已安装 lerobot 版本自适应：
            # - lerobot 0.6+：create() 通过 rgb_encoder=RGBEncoderConfig(vcodec=...)
            #   指定编码器（已无 vcodec 关键字参数）
            # - lerobot 0.4.x：直接传 vcodec 关键字参数
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
        """主线程定时器回调：快速把当前双路帧推入后台队列，不执行 add_frame"""
        if not self._lerobot_recording or self._lerobot_record_thread is None:
            return

        if len(self.camera_indices) < 2:
            return

        idx0, idx1 = self.camera_indices[0], self.camera_indices[1]
        frame0 = self.latest_frames.get(idx0)
        frame1 = self.latest_frames.get(idx1)
        if frame0 is None or frame1 is None:
            return

        try:
            t0 = time.perf_counter()
            rgb0 = self._resize_frame(frame0, *self._lerobot_image_size)
            rgb1 = self._resize_frame(frame1, *self._lerobot_image_size)
            self._lerobot_record_thread.push_frame(rgb0, rgb1)
            dt = (time.perf_counter() - t0) * 1000

            self._lerobot_frame_count += 1
            if self._lerobot_frame_count % 20 == 0:
                logger.info(
                    f"[LeRobot] 已采集 {self._lerobot_frame_count} 帧，"
                    f"主线程入队耗时 {dt:.1f}ms"
                )
        except Exception as e:
            logger.warning(f"[LeRobot] 图像缩放或入队失败: {e}")

    def _lerobot_on_timer(self):
        """已废弃：旧主线程 add_frame 路径，保留兼容"""
        pass

    def _get_pose_7d(self):
        """返回 [x, y, z, rx, ry, rz, gripper_rad] 或回调不可用时全0

        7维数据：位置(3) + 欧拉角(3，弧度) + 夹爪角度(1，弧度)。
        与 MoveL/EndPoseCtrl 的欧拉角约定一致，可直接用于逆解控制。
        """
        if self._lerobot_get_end_pose is None:
            return np.zeros(7, dtype=np.float32)

        try:
            result = self._lerobot_get_end_pose()
        except Exception as e:
            logger.warning(f"[LeRobot] 获取末端位姿失败: {e}")
            return np.zeros(7, dtype=np.float32)

        # 兼容旧接口：可能只返回 ArmEndPose
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

        # 夹爪角度：统一弧度（与机械臂 GripperCtrl 接口、末端姿态 rx/ry/rz 单位一致）
        if gripper_rad is None:
            gripper_rad = 0.0

        return np.array([x, y, z, rx, ry, rz, float(gripper_rad)], dtype=np.float32)

    def _center_crop_to_lerobot(self, frame):
        """按 LeRobot 目标分辨率的长宽比对 BGR 帧做中心裁切（与 _resize_frame 一致）。

        用于界面预览同步裁切：如 LeRobot 选 720x720，则把 1280x720 画面
        中心裁出 720x720 方形区域再显示，所见即所录。
        """
        target_h, target_w = self._lerobot_image_size
        h, w = frame.shape[:2]
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
    def _resize_frame(frame, target_h, target_w):
        """将 BGR 帧缩放为 RGB numpy 数组 (H, W, 3)。

        目标长宽比与源不一致时先中心裁切再缩放（如 1280x720 -> 720x720
        取画面中央方形区域），避免直接 resize 造成的拉伸失真。
        """
        h, w = frame.shape[:2]
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
        name, w, h = self.lerobot_res_options[index]
        self._lerobot_image_size = (h, w)
        logger.info(f"[Camera] LeRobot 图像分辨率切换为: {name}")

    def _ensure_capture_resolution(self):
        """确保摄像头采集分辨率不低于 LeRobot 录制目标分辨率。

        如果当前采集分辨率低于目标分辨率，自动切换到目标分辨率并重新打开摄像头。
        这是为了防止“录制设置 1280x720，但实际是从 320x240 放大存储”的画质问题。
        """
        target_h, target_w = self._lerobot_image_size
        cap_w, cap_h = self.capture_res

        if cap_w < target_w or cap_h < target_h:
            logger.warning(
                f"[LeRobot] 当前采集分辨率 {cap_w}x{cap_h} 低于录制目标 {target_w}x{target_h}，"
                f"自动切换到 {target_w}x{target_h}"
            )
            self.capture_res = (target_w, target_h)
            # 同步更新 UI 下拉框
            for i, (name, w, h) in enumerate(self.resolutions):
                if w == target_w and h == target_h:
                    self.res_combo.blockSignals(True)
                    self.res_combo.setCurrentIndex(i)
                    self.res_combo.blockSignals(False)
                    break
            # 重新打开摄像头
            for idx in self.camera_indices[:2]:
                if idx in self.caps:
                    self.open_camera(idx, target_w, target_h)

    def _start_lerobot_record(self):
        if len(self.camera_indices) < 2:
            QMessageBox.warning(self, "LeRobot 录制", "需要至少 2 个摄像头才能录制 LeRobot 数据集")
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
        for idx in self.camera_indices[:2]:
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
            fps=self._lerobot_fps,
            max_queue=5,
            torque_callback=self._lerobot_get_torque,
            action_lookahead=self._action_lookahead,
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
        self.btn_lerobot.setText("停止LeRobot录制")
        self.btn_lerobot.setStyleSheet(
            "font-size: 12px; padding: 4px 12px; background-color: #ff4444; color: white;"
        )
        self.status_label.setText(f"LeRobot 录制中: {self._lerobot_task_name}")
        logger.info(f"[LeRobot] 开始录制任务: {self._lerobot_task_name}，频率 {self._lerobot_fps}Hz")

    def _on_lerobot_frame_written(self, count):
        """后台线程写入帧数回调（可选）"""
        pass

    def _on_lerobot_thread_error(self, msg):
        """后台线程出错回调"""
        logger.error(f"[LeRobot] 录制线程错误: {msg}")
        QMessageBox.critical(self, "LeRobot 录制错误", msg)
        self._stop_lerobot_record()

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
        # 先记录 dataset 路径，并立即弹完成对话框（同步），确保用户一定能看到结果，
        # 不被后续 parquet 验证或线程 wait() 阻塞/异常影响。
        dataset_root = str(self._lerobot_dataset.root) if self._lerobot_dataset else ""
        if result:
            QMessageBox.information(
                self, "LeRobot 录制完成",
                f"{result}到:\n{dataset_root}"
            )

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

    def cleanup(self):
        """清理资源，在关闭窗口时调用"""
        self._stop_rtc_executor()
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

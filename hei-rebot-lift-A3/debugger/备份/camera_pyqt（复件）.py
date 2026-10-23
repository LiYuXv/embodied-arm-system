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
from pathlib import Path
from queue import Queue, Empty, Full

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
    QComboBox, QMessageBox, QGridLayout, QLineEdit
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
    - 主线程只把 (rgb0, rgb1, pose8, task) 推入队列（<0.1ms）
    - 后台线程负责调用 dataset.add_frame（即使慢也不影响 UI）
    - 队列满时自动丢弃最旧帧，保证实时性
    - 停止时把队列剩余帧全部写入，再调用 save_episode
    """
    frame_written = pyqtSignal(int)   # 已写入帧数
    status = pyqtSignal(str)          # 状态文本
    finished_ok = pyqtSignal(str)     # 保存完成结果
    error = pyqtSignal(str)           # 错误信息

    def __init__(self, dataset, pose_callback, fps=20, max_queue=60, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.pose_callback = pose_callback
        self.fps = fps
        self.max_queue = max_queue
        self._queue = Queue(maxsize=max_queue)
        self._running = True
        self._frame_count = 0
        self._task_name = ""

    def set_task_name(self, name):
        self._task_name = name

    def push_frame(self, rgb0, rgb1):
        """主线程调用：快速把帧和位姿推入队列。如果队列满，丢弃最旧帧。"""
        pose8 = self._get_pose8()
        item = {"cam1": rgb0, "cam2": rgb1, "pose8": pose8, "task": self._task_name}
        try:
            self._queue.put_nowait(item)
        except Full:
            try:
                self._queue.get_nowait()  # 丢弃最旧帧
                self._queue.put_nowait(item)
            except Empty:
                pass

    def _get_pose8(self):
        """获取 8 维位姿，失败返回 0 数组"""
        if self.pose_callback is None:
            return np.zeros(8, dtype=np.float32)
        try:
            result = self.pose_callback()
        except Exception as e:
            logger.warning(f"[LeRobotThread] 获取末端位姿失败: {e}")
            return np.zeros(8, dtype=np.float32)

        if isinstance(result, tuple) and len(result) >= 2:
            end_pose, gripper_deg = result[0], result[1]
        else:
            end_pose = result
            gripper_deg = 0.0

        if end_pose is None:
            return np.zeros(8, dtype=np.float32)

        x = getattr(end_pose, "x", 0.0)
        y = getattr(end_pose, "y", 0.0)
        z = getattr(end_pose, "z", 0.0)
        rx = getattr(end_pose, "rx", 0.0)
        ry = getattr(end_pose, "ry", 0.0)
        rz = getattr(end_pose, "rz", 0.0)

        try:
            from el_a3_sdk.utils import euler_to_quat
            qw, qx, qy, qz = euler_to_quat(rx, ry, rz)
        except Exception:
            qx = qy = qz = qw = 0.0

        if gripper_deg is None:
            gripper_deg = 0.0

        return np.array([x, y, z, qx, qy, qz, qw, float(gripper_deg)], dtype=np.float32)

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
            while self._running or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.1)
                except Empty:
                    continue

                t0 = time.perf_counter()
                self.dataset.add_frame({
                    "observation.images.cam1": item["cam1"],
                    "observation.images.cam2": item["cam2"],
                    "observation.state": item["pose8"].astype(np.float32),
                    "action": item["pose8"].astype(np.float32),
                    "task": item["task"],
                })
                self._frame_count += 1
                dt = (time.perf_counter() - t0) * 1000

                if self._frame_count % 20 == 0:
                    logger.info(f"[LeRobotThread] 已写入 {self._frame_count} 帧，add_frame 耗时 {dt:.1f}ms")

                self.frame_written.emit(self._frame_count)
                time.sleep(max(0, interval - dt / 1000.0))

            # 队列已空，在后台线程中执行 save_episode，避免阻塞主线程
            if self._frame_count > 0:
                t0 = time.perf_counter()
                self.dataset.save_episode(parallel_encoding=True)
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


class CameraWidget(QWidget):
    """可嵌入的摄像头监控组件（双路摄像头）—— 优化版，支持 LeRobot v0.3 数据采集"""

    # 信号：降频到10Hz发射，用于后续lerobot数据集收集
    frame_captured = pyqtSignal(int, object)  # (camera_idx, frame_ndarray)

    def __init__(self, parent=None, save_dir=None, dataset_root=None):
        super().__init__(parent)

        # 分辨率选项 (名称, 宽度, 高度, 推荐用于预览)
        # 默认640x480平衡清晰度和性能
        self.resolutions = [
            ("1280 x 720 (推荐)", 1280, 720),
            ("640 x 480", 640, 480),
            ("320 x 240", 320, 240),
            ("800 x 600", 800, 600),
            ("1024 x 768", 1024, 768),
            ("1280 x 800", 1280, 800),
            ("1920 x 1080", 1920, 1080),
            ("2592 x 1944", 2592, 1944),
        ]
        self.capture_res = (320, 240)  # 默认摄像头预览分辨率 320x240

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

        # 显示降频控制：限制UI刷新到15fps
        self._display_timer = None
        self._pending_frames = {}
        self._last_emit_time = {}

        # LeRobot 采集状态
        self._lerobot_recording = False
        self._lerobot_dataset = None
        self._lerobot_frame_count = 0
        self._lerobot_get_end_pose = None  # callback(end_pose_dict) -> ArmEndPose-like
        self._lerobot_task_name = ""
        self._lerobot_repo_name = "ela3_demo"
        self._lerobot_fps = 20
        self._lerobot_image_size = (720, 1280)
        self._lerobot_record_thread = None  # 后台录制线程

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
            ("640 x 480", 640, 480),
            ("480 x 480", 480, 480),
            ("256 x 256", 256, 256),
        ]
        for name, *_ in self.lerobot_res_options:
            self.lerobot_res_combo.addItem(name)
        self.lerobot_res_combo.setCurrentIndex(0)  # 默认 1280 x 720
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

        control_layout.addWidget(QLabel("分辨率:"))
        control_layout.addWidget(self.res_combo)
        control_layout.addWidget(QLabel("LeRobot图像:"))
        control_layout.addWidget(self.lerobot_res_combo)
        control_layout.addStretch()
        control_layout.addWidget(QLabel("任务:"))
        control_layout.addWidget(self.task_input, stretch=1)
        control_layout.addWidget(self.btn_snap)
        control_layout.addWidget(self.btn_record)
        control_layout.addWidget(self.btn_lerobot)
        layout.addLayout(control_layout)

        # 保存路径提示标签
        self.lbl_save_dir = QLabel(f"保存目录: {self.save_dir}")
        self.lbl_save_dir.setStyleSheet("color: #666; font-size: 10px;")
        self.lbl_save_dir.setWordWrap(True)
        self.lbl_save_dir.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_save_dir)

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
        logger.info(f"[Camera] 摄像头 [{idx}] 已就绪，实际分辨率: {actual_w} x {actual_h}")

        self.caps[idx] = cap

        # 启动后台采集线程
        thread = CameraCaptureThread(idx, cap)
        thread.frame_ready.connect(self._on_frame_received)
        thread.start()
        self.capture_threads[idx] = thread

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
        """统一显示刷新：15fps，将缓存的帧转换为QPixmap显示"""
        for i, idx in enumerate(self.camera_indices[:2]):
            if idx not in self._pending_frames:
                continue

            frame = self._pending_frames[idx]
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

        callback() 应返回 (ArmEndPose, gripper_angle_deg) 或仅 ArmEndPose。
        ArmEndPose 包含 x,y,z,rx,ry,rz (m/rad)。
        gripper_angle_deg 是夹爪角度（度）。
        """
        self._lerobot_get_end_pose = callback

    def _ensure_lerobot_dataset(self):
        """初始化或加载 LeRobot v3.0 数据集 (适配 lerobot 0.4.4) - 修复联网卡住问题

        策略：
        1. 如果已有同维度、同视频类型、同分辨率数据集，直接加载追加 episode
        2. 如果不匹配，自动创建带时间戳的新数据集目录（不删除旧的）
        3. 如果是全新数据集，直接创建
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
        target_state_shape = (8,)
        target_action_shape = (8,)

        # 如果数据集已存在，检查是否兼容
        info_path = os.path.join(dataset_dir, "meta", "info.json")
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

                compatible = (
                    tuple(state_shape) == target_state_shape and
                    tuple(action_shape) == target_action_shape and
                    cam1_dtype == target_cam_dtype and
                    cam2_dtype == target_cam_dtype and
                    cam1_codec == target_video_codec and
                    cam2_codec == target_video_codec and
                    tuple(cam1_shape) == (target_h, target_w, 3) and
                    tuple(cam2_shape) == (target_h, target_w, 3)
                )

                if not compatible:
                    logger.warning(
                        f"[LeRobot] 已有数据集不兼容 "
                        f"(state={state_shape}, action={action_shape}, "
                        f"cam_dtype={cam1_dtype}/{cam2_dtype}, "
                        f"cam_codec={cam1_codec}/{cam2_codec}, "
                        f"cam_shape={cam1_shape}/{cam2_shape})，将创建新数据集"
                    )
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_repo_name = f"{self._lerobot_repo_name}_{timestamp}"
                    self._lerobot_repo_name = new_repo_name
                    dataset_dir = os.path.join(self.dataset_root, self._lerobot_repo_name)
                    logger.info(f"[LeRobot] 新数据集目录: {dataset_dir}")
                else:
                    # 兼容，加载已有数据集
                    import os as os_module
                    old_hf_offline = os_module.environ.get("HF_HUB_OFFLINE")
                    os_module.environ["HF_HUB_OFFLINE"] = "1"
                    dataset = LeRobotDataset(self._lerobot_repo_name, root=dataset_dir)
                    if old_hf_offline is not None:
                        os_module.environ["HF_HUB_OFFLINE"] = old_hf_offline
                    else:
                        os_module.environ.pop("HF_HUB_OFFLINE", None)
                    logger.info(f"[LeRobot] 加载已有数据集: {dataset_dir} "
                                f"(episodes={dataset.num_episodes}, frames={dataset.num_frames})")
                    return dataset
            except Exception as e:
                logger.warning(f"[LeRobot] 加载已有数据集失败: {e}")
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_repo_name = f"{self._lerobot_repo_name}_{timestamp}"
                self._lerobot_repo_name = new_repo_name
                dataset_dir = os.path.join(self.dataset_root, self._lerobot_repo_name)
                logger.info(f"[LeRobot] 加载失败，创建新数据集: {dataset_dir}")

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
                "shape": (8,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["action"],
            },
        }

        try:
            # 设置离线模式避免联网
            import os as os_module
            old_hf_offline = os_module.environ.get("HF_HUB_OFFLINE")
            os_module.environ["HF_HUB_OFFLINE"] = "1"

            vcodec = _resolve_vcodec_for_recording()
            # 硬件编码器本身已充分利用 GPU，不需要额外线程；软件编码用多线程加速
            encoder_threads = None if vcodec in ("h264_nvenc", "hevc_nvenc") else 4

            dataset = LeRobotDataset.create(
                repo_id=self._lerobot_repo_name,
                fps=self._lerobot_fps,
                features=features,
                root=dataset_dir,
                robot_type="ela3",
                use_videos=True,  # 视频编码，大幅降低 I/O 和 save_episode 耗时
                image_writer_processes=0,
                image_writer_threads=4,
                vcodec=vcodec,
                encoder_threads=encoder_threads,
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

    def _get_end_pose_8d(self):
        """返回 [x, y, z, qx, qy, qz, qw, gripper_deg] 或回调不可用时全0

        8维数据：位置(3) + 四元数(4) + 夹爪角度(1，单位度)
        """
        if self._lerobot_get_end_pose is None:
            return np.zeros(8, dtype=np.float32)

        try:
            result = self._lerobot_get_end_pose()
        except Exception as e:
            logger.warning(f"[LeRobot] 获取末端位姿失败: {e}")
            return np.zeros(8, dtype=np.float32)

        # 兼容旧接口：可能只返回 ArmEndPose
        if isinstance(result, tuple) and len(result) >= 2:
            end_pose, gripper_deg = result[0], result[1]
        else:
            end_pose = result
            gripper_deg = 0.0

        if end_pose is None:
            return np.zeros(8, dtype=np.float32)

        x = getattr(end_pose, "x", 0.0)
        y = getattr(end_pose, "y", 0.0)
        z = getattr(end_pose, "z", 0.0)
        rx = getattr(end_pose, "rx", 0.0)
        ry = getattr(end_pose, "ry", 0.0)
        rz = getattr(end_pose, "rz", 0.0)

        try:
            from el_a3_sdk.utils import euler_to_quat
            qw, qx, qy, qz = euler_to_quat(rx, ry, rz)
        except Exception:
            qx = qy = qz = qw = 0.0

        # 夹爪角度：如果为None则默认0
        if gripper_deg is None:
            gripper_deg = 0.0

        return np.array([x, y, z, qx, qy, qz, qw, float(gripper_deg)], dtype=np.float32)

    @staticmethod
    def _resize_frame(frame, target_h, target_w):
        """将 BGR 帧缩放为 RGB numpy 数组 (H, W, 3)"""
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

    def _start_lerobot_record(self):
        if len(self.camera_indices) < 2:
            QMessageBox.warning(self, "LeRobot 录制", "需要至少 2 个摄像头才能录制 LeRobot 数据集")
            return

        # 检查任务描述：禁止中文
        task_text = self.task_input.text().strip()
        if any('\u4e00' <= c <= '\u9fff' for c in task_text):
            QMessageBox.warning(
                self, "Task Format Error",
                "任务描述不能包含中文，请使用英文。\n"
                "例如: 'pick up the red cube'"
            )
            return

        dataset = self._ensure_lerobot_dataset()
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
            parent=self,
        )
        self._lerobot_record_thread.set_task_name(self._lerobot_task_name)
        self._lerobot_record_thread.frame_written.connect(self._on_lerobot_frame_written)
        self._lerobot_record_thread.error.connect(self._on_lerobot_thread_error)
        self._lerobot_record_thread.start()

        # 启动定时器，以固定频率采集并推入队列（20Hz = 50ms）
        interval_ms = int(1000 / self._lerobot_fps)
        self._lerobot_timer.start(interval_ms)

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
        # 验证当前 data 目录下存在的 parquet 文件（注意不要在备份目录里验证）。
        # save_episode 刚结束时可能还有临时文件，失败时短暂重试避免误报。
        try:
            import pandas as pd
            import time as _time
            dataset_root = str(self._lerobot_dataset.root) if self._lerobot_dataset else None
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

        if result:
            QTimer.singleShot(100, lambda: QMessageBox.information(
                self, "LeRobot 录制完成",
                f"{result}到:\n{dataset_root}"
            ))

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

"""摄像头监控组件：支持双路USB摄像头预览、拍照、录像，可嵌入主界面。

优化要点：
- 后台采集线程读取帧，主线程只负责显示
- 默认640x480降低带宽，使用MJPEG格式
- 图像缩放使用FastTransformation
- 帧信号降频发射（10Hz），避免主线程过载
- 可选YOLOv5目标检测（后台线程推理）
"""

import os
import sys
import logging
import time

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
    QComboBox, QMessageBox, QGridLayout, QCheckBox, QDoubleSpinBox,
    QGroupBox, QFileDialog, QSpinBox
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
# 主界面组件
# ============================================================================
class CameraWidget(QWidget):
    """可嵌入的摄像头监控组件（双路摄像头）—— 优化版，支持YOLO检测"""

    # 信号：降频到10Hz发射，用于后续lerobot数据集收集
    frame_captured = pyqtSignal(int, object)  # (camera_idx, frame_ndarray)

    def __init__(self, parent=None, save_dir=None):
        super().__init__(parent)

        # 分辨率选项 (名称, 宽度, 高度, 推荐用于预览)
        # 默认640x480平衡清晰度和性能
        self.resolutions = [
            ("640 x 480 (推荐)", 640, 480),
            ("320 x 240", 320, 240),
            ("800 x 600", 800, 600),
            ("1024 x 768", 1024, 768),
            ("1280 x 720", 1280, 720),
            ("1280 x 800", 1280, 800),
            ("1920 x 1080", 1920, 1080),
            ("2592 x 1944", 2592, 1944),
        ]
        self.capture_res = (640, 480)  # 默认降低分辨率减少卡顿

        # 保存目录
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"[Camera] 保存目录: {self.save_dir}")

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

        # ========== YOLO 检测相关 ==========
        self._yolo_detector = None      # YOLO检测器实例
        self._yolo_thread = None        # YOLO后台检测线程
        self._yolo_enabled = False      # 检测开关状态
        self._yolo_results = {}         # 检测结果缓存 {camera_idx: result_frame}
        # 默认配置文件路径
        default_yolo_config = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "yolo", "config", "yolov5s.yaml"
        )
        self._yolo_config_path = default_yolo_config

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

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
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

        # ---- 摄像头控制栏 ----
        control_layout = QHBoxLayout()

        self.res_combo = QComboBox()
        for name, *_ in self.resolutions:
            self.res_combo.addItem(name)
        self.res_combo.setCurrentIndex(0)
        self.res_combo.currentIndexChanged.connect(self.change_resolution)

        self.btn_snap = QPushButton("拍照 (Snap)")
        self.btn_snap.setStyleSheet("font-size: 12px; padding: 4px 12px;")
        self.btn_snap.clicked.connect(self.take_snapshot)

        self.btn_record = QPushButton("开始录像")
        self.btn_record.setStyleSheet("font-size: 12px; padding: 4px 12px;")
        self.btn_record.clicked.connect(self.toggle_record)

        control_layout.addWidget(QLabel("分辨率:"))
        control_layout.addWidget(self.res_combo)
        control_layout.addStretch()
        control_layout.addWidget(self.btn_snap)
        control_layout.addWidget(self.btn_record)
        layout.addLayout(control_layout)

        # ---- YOLO 检测控制面板 ----
        yolo_group = QGroupBox("YOLO 目标检测")
        yolo_group.setCheckable(True)
        yolo_group.setChecked(False)
        yolo_group.toggled.connect(self._on_yolo_group_toggled)
        self._yolo_group = yolo_group

        yolo_layout = QGridLayout(yolo_group)
        yolo_layout.setSpacing(4)

        # 模型配置
        yolo_layout.addWidget(QLabel("配置文件:"), 0, 0)
        self._yolo_config_label = QLabel("默认: yolov5s.yaml")
        self._yolo_config_label.setStyleSheet("color: #4fc3f7;")
        yolo_layout.addWidget(self._yolo_config_label, 0, 1)

        self._btn_yolo_config = QPushButton("选择配置...")
        self._btn_yolo_config.clicked.connect(self._select_yolo_config)
        yolo_layout.addWidget(self._btn_yolo_config, 0, 2)

        # 检测参数
        yolo_layout.addWidget(QLabel("置信度:"), 1, 0)
        self._spin_conf = QDoubleSpinBox()
        self._spin_conf.setRange(0.01, 1.0)
        self._spin_conf.setSingleStep(0.05)
        self._spin_conf.setValue(0.25)
        self._spin_conf.valueChanged.connect(self._on_yolo_param_changed)
        yolo_layout.addWidget(self._spin_conf, 1, 1)

        yolo_layout.addWidget(QLabel("IOU阈值:"), 2, 0)
        self._spin_iou = QDoubleSpinBox()
        self._spin_iou.setRange(0.01, 1.0)
        self._spin_iou.setSingleStep(0.05)
        self._spin_iou.setValue(0.45)
        self._spin_iou.valueChanged.connect(self._on_yolo_param_changed)
        yolo_layout.addWidget(self._spin_iou, 2, 1)

        # 检测开关
        self._chk_yolo_detect = QCheckBox("开启检测")
        self._chk_yolo_detect.stateChanged.connect(self._on_yolo_detect_toggled)
        yolo_layout.addWidget(self._chk_yolo_detect, 3, 0, 1, 3)

        # 检测统计
        self._lbl_yolo_status = QLabel("YOLO: 未加载")
        self._lbl_yolo_status.setStyleSheet("color: #f44336; font-size: 10px;")
        yolo_layout.addWidget(self._lbl_yolo_status, 4, 0, 1, 3)

        layout.addWidget(yolo_group)

        # 保存路径提示标签
        self.lbl_save_dir = QLabel(f"保存目录: {self.save_dir}")
        self.lbl_save_dir.setStyleSheet("color: #666; font-size: 10px;")
        self.lbl_save_dir.setWordWrap(True)
        self.lbl_save_dir.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_save_dir)

    # ------------------------------------------------------------------
    # YOLO 检测控制
    # ------------------------------------------------------------------
    def _on_yolo_group_toggled(self, checked):
        """YOLO面板展开/收起"""
        if not checked:
            # 收起时自动关闭检测
            if self._yolo_enabled:
                self._chk_yolo_detect.setChecked(False)

    def _select_yolo_config(self):
        """选择YOLO配置文件"""
        # 默认打开yolo/config目录
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo", "config")
        if not os.path.exists(default_dir):
            default_dir = os.path.dirname(os.path.abspath(__file__))
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择YOLOv5配置文件",
            default_dir,
            "YAML文件 (*.yaml *.yml)"
        )
        if file_path:
            self._yolo_config_path = file_path
            self._yolo_config_label.setText(os.path.basename(file_path))
            self._yolo_config_label.setStyleSheet("color: #4fc3f7;")
            logger.info(f"[YOLO] 已选择配置: {file_path}")
            # 如果检测已开启，自动重新加载
            if self._yolo_enabled:
                self._reload_yolo()

    def _load_yolo_model(self):
        """加载YOLO模型"""
        if not self._yolo_config_path or not os.path.exists(self._yolo_config_path):
            QMessageBox.warning(self, "提示", f"配置文件不存在:\n{self._yolo_config_path}")
            return False

        try:
            from widgets.yolo_detector import YoloDetector

            self._yolo_detector = YoloDetector(self._yolo_config_path)
            self._yolo_detector.load_model()

            # 同步阈值到UI
            self._spin_conf.setValue(self._yolo_detector.conf_threshold)
            self._spin_iou.setValue(self._yolo_detector.iou_threshold)

            self._lbl_yolo_status.setText(f"YOLO: 已加载 ({len(self._yolo_detector.names)}类)")
            self._lbl_yolo_status.setStyleSheet("color: #4caf50; font-size: 10px;")
            logger.info("[YOLO] 模型加载成功")
            return True

        except Exception as e:
            self._lbl_yolo_status.setText(f"YOLO: 加载失败")
            self._lbl_yolo_status.setStyleSheet("color: #f44336; font-size: 10px;")
            logger.error(f"[YOLO] 模型加载失败: {e}")
            QMessageBox.critical(self, "错误", f"模型加载失败:\n{str(e)}")
            return False

    def _reload_yolo(self):
        """重新加载YOLO模型（配置变更后）"""
        if self._yolo_thread:
            self._yolo_thread.stop()
            self._yolo_thread = None
        if self._yolo_detector:
            self._yolo_detector.unload()
            self._yolo_detector = None
        self._yolo_results.clear()
        self._load_yolo_model()
        if self._yolo_enabled and self._yolo_detector and self._yolo_detector._loaded:
            from widgets.yolo_detector import YoloDetectionThread
            self._yolo_thread = YoloDetectionThread(self._yolo_detector)
            self._yolo_thread.result_ready.connect(self._on_yolo_result)
            self._yolo_thread.error_occurred.connect(self._on_yolo_error)
            self._yolo_thread.start()

    def _on_yolo_param_changed(self):
        """YOLO参数变化"""
        if self._yolo_detector and self._yolo_detector._loaded:
            self._yolo_detector.set_thresholds(
                conf=self._spin_conf.value(),
                iou=self._spin_iou.value()
            )

    def _on_yolo_detect_toggled(self, state):
        """开启/关闭目标检测"""
        self._yolo_enabled = (state == Qt.CheckState.Checked.value)

        if self._yolo_enabled:
            # 自动加载模型（如果未加载）
            if self._yolo_detector is None or not self._yolo_detector._loaded:
                if not self._load_yolo_model():
                    self._chk_yolo_detect.setChecked(False)
                    return

            # 启动YOLO后台检测线程
            from widgets.yolo_detector import YoloDetectionThread
            self._yolo_thread = YoloDetectionThread(self._yolo_detector)
            self._yolo_thread.result_ready.connect(self._on_yolo_result)
            self._yolo_thread.error_occurred.connect(self._on_yolo_error)
            self._yolo_thread.start()

            self._lbl_yolo_status.setText("YOLO: 检测中...")
            self._lbl_yolo_status.setStyleSheet("color: #ff9800; font-size: 10px;")
            logger.info("[YOLO] 检测已开启")
        else:
            # 停止检测线程
            if self._yolo_thread:
                self._yolo_thread.stop()
                self._yolo_thread = None
            self._yolo_results.clear()
            self._lbl_yolo_status.setText(f"YOLO: 已加载 ({len(self._yolo_detector.names)}类)")
            self._lbl_yolo_status.setStyleSheet("color: #4caf50; font-size: 10px;")
            logger.info("[YOLO] 检测已关闭")

    def _on_yolo_result(self, camera_idx, result_frame, detections):
        """YOLO检测结果回调"""
        self._yolo_results[camera_idx] = result_frame
        # 更新状态显示检测到的目标数
        if detections:
            self._lbl_yolo_status.setText(f"YOLO: 检测到 {len(detections)} 个目标")
        else:
            self._lbl_yolo_status.setText("YOLO: 检测中...")

    def _on_yolo_error(self, error_msg):
        """YOLO错误回调"""
        logger.error(f"[YOLO] {error_msg}")
        self._lbl_yolo_status.setText(f"YOLO: 错误")
        self._lbl_yolo_status.setStyleSheet("color: #f44336; font-size: 10px;")

    # ------------------------------------------------------------------
    # 摄像头控制
    # ------------------------------------------------------------------
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

        # YOLO检测：降频送入检测线程（每100ms一帧），避免GPU队列堆积
        if self._yolo_enabled and self._yolo_thread:
            now = time.time()
            last_yolo = getattr(self, '_last_yolo_time', {}).get(camera_idx, 0)
            if now - last_yolo >= 0.1:
                if not hasattr(self, '_last_yolo_time'):
                    self._last_yolo_time = {}
                self._last_yolo_time[camera_idx] = now
                self._yolo_thread.set_frame(camera_idx, frame.copy())

    def _refresh_display(self):
        """统一显示刷新：15fps，将缓存的帧转换为QPixmap显示"""
        for i, idx in enumerate(self.camera_indices[:2]):
            if idx not in self._pending_frames:
                continue

            # 如果有YOLO检测结果，优先显示检测结果
            if self._yolo_enabled and idx in self._yolo_results:
                frame = self._yolo_results[idx]
                # 消费掉，避免重复显示同一帧
                del self._yolo_results[idx]
            else:
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
        """拍照（保存原始帧或检测结果）"""
        saved = []
        for idx in self.camera_indices[:2]:
            if idx not in self.latest_frames:
                logger.warning(f"[Camera] 摄像头 [{idx}] 没有可用帧，跳过拍照")
                continue

            # 如果检测开启且有结果，保存检测结果图
            if self._yolo_enabled and self._yolo_detector and self._yolo_detector._loaded:
                frame, _ = self._yolo_detector.detect(self.latest_frames[idx].copy())
            else:
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

    def cleanup(self):
        """清理资源，在关闭窗口时调用"""
        if self._display_timer is not None:
            self._display_timer.stop()

        # 停止YOLO检测
        if self._yolo_thread:
            self._yolo_thread.stop()
            self._yolo_thread = None
        if self._yolo_detector:
            self._yolo_detector.unload()
            self._yolo_detector = None

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
        self._yolo_results.clear()
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

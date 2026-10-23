"""YOLOv5 检测器封装：为 camera_pyqt.py 提供可选的目标检测功能。

支持：
- 动态加载/卸载模型
- 置信度和IOU阈值调节
- 后台线程推理不阻塞UI
- 检测框绘制
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path

logger = logging.getLogger("debugger")

# YOLOv5 路径设置
YOLO_DIR = Path(__file__).parent / "yolo"
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal

# 延迟导入 torch，避免未安装时崩溃
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    logger.warning("[YOLO] torch 未安装，YOLO检测功能不可用")


class YoloDetector:
    """YOLOv5 检测器封装"""

    def __init__(self, config_path=None):
        self.config_path = config_path
        self.device = None
        self.model = None
        self.names = []
        self.stride = 0
        self.is_half = False
        self.colors = []
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.input_size = 640
        self._loaded = False

    def is_available(self):
        """检查是否可以加载模型"""
        return _HAS_TORCH and self.config_path and Path(self.config_path).exists()

    def load_model(self, config_path=None):
        """加载YOLOv5模型"""
        if not _HAS_TORCH:
            raise RuntimeError("torch 未安装，无法加载YOLO模型")

        if config_path:
            self.config_path = config_path

        if not self.config_path or not Path(self.config_path).exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        import yaml
        from models.experimental import attempt_load
        from utils.torch_utils import select_device

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.input_size = config.get('input_size', 640)
        device_str = config.get('device', 'cpu')
        weight_path = config.get('weight', 'weights/yolov5s.pt')

        # 如果是相对路径，基于配置文件目录解析
        if not Path(weight_path).is_absolute():
            weight_path = Path(self.config_path).parent / weight_path

        self.conf_threshold = config.get('threshold', {}).get('confidence', 0.25)
        self.iou_threshold = config.get('threshold', {}).get('iou', 0.45)
        class_num = config.get('class_num', 80)
        self.names = config.get('names', [f'class_{i}' for i in range(class_num)])

        self.device = select_device(device_str)
        self.is_half = self.device.type != 'cpu'

        logger.info(f"[YOLO] 正在加载模型: {weight_path} ...")
        self.model = attempt_load(str(weight_path), map_location=self.device)
        self.stride = int(self.model.stride.max())

        if hasattr(self.model, 'module'):
            self.names = self.model.module.names
        else:
            self.names = self.model.names

        if self.is_half:
            self.model.half()

        # Warmup
        img = torch.zeros((1, 3, self.input_size, self.input_size), device=self.device)
        _ = self.model(img.half() if self.is_half else img)

        # 生成颜色
        np.random.seed(42)
        self.colors = [[np.random.randint(0, 255) for _ in range(3)]
                        for _ in range(len(self.names))]

        self._loaded = True
        logger.info(f"[YOLO] 模型加载完成，类别数: {len(self.names)}")
        return True

    def set_thresholds(self, conf=None, iou=None):
        """设置检测阈值"""
        if conf is not None:
            self.conf_threshold = conf
        if iou is not None:
            self.iou_threshold = iou

    def preprocess(self, img):
        """图像预处理"""
        import torch

        # 调整尺寸
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        # BGR -> RGB，HWC -> CHW
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)
        img_resized = np.ascontiguousarray(img_resized)

        img_tensor = torch.from_numpy(img_resized).to(self.device)
        img_tensor = img_tensor.half() if self.is_half else img_tensor.float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def detect(self, img):
        """执行检测，返回绘制了检测框的图像和检测结果列表"""
        if not self._loaded or self.model is None:
            return img, []

        from utils.general import non_max_suppression, scale_coords
        import torch

        # 预处理
        img_processed = self.preprocess(img)

        # 推理
        with torch.no_grad():
            pred = self.model(img_processed, augment=False)[0]

        # NMS
        pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold)

        detections = []
        canvas = img.copy()

        if pred[0] is not None and len(pred[0]) > 0:
            # 将坐标映射回原图尺寸
            pred[0][:, :4] = scale_coords(
                img_processed.shape[2:], pred[0][:, :4], img.shape
            ).round()

            for *xyxy, conf_val, cls in pred[0]:
                cls_id = int(cls)
                label = f'{self.names[cls_id]} {conf_val:.2f}'
                self._plot_one_box(xyxy, canvas, label, self.colors[cls_id])
                detections.append({
                    'class': self.names[cls_id],
                    'class_id': cls_id,
                    'confidence': float(conf_val),
                    'bbox': [int(x) for x in xyxy]
                })

        return canvas, detections

    def _plot_one_box(self, x, img, label=None, color=None, line_thickness=2):
        """绘制检测框"""
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        color = color or [np.random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                       [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def unload(self):
        """卸载模型释放内存"""
        self.model = None
        self._loaded = False
        if self.device and self.device.type == 'cuda':
            import torch
            torch.cuda.empty_cache()
        logger.info("[YOLO] 模型已卸载")


class YoloDetectionThread(QThread):
    """YOLO 后台检测线程：避免阻塞UI"""
    result_ready = pyqtSignal(int, object, list)  # (camera_idx, result_frame, detections)
    error_occurred = pyqtSignal(str)

    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self.detector = detector
        self._running = False
        self._frame_queue = {}  # camera_idx -> frame
        self._queue_lock = threading.Lock()

    def set_frame(self, camera_idx, frame):
        """设置待检测帧（线程安全，直接替换旧帧）"""
        with self._queue_lock:
            self._frame_queue[camera_idx] = frame

    def run(self):
        self._running = True
        while self._running:
            with self._queue_lock:
                if not self._frame_queue:
                    has_work = False
                else:
                    has_work = True
                    queue_copy = dict(self._frame_queue)
                    self._frame_queue.clear()
            if not has_work:
                self.msleep(5)
                continue

            for camera_idx, frame in queue_copy.items():
                if frame is None:
                    continue
                try:
                    result, detections = self.detector.detect(frame)
                    self.result_ready.emit(camera_idx, result, detections)
                except Exception as e:
                    self.error_occurred.emit(f"检测失败: {e}")

    def stop(self):
        self._running = False
        self.wait(500)

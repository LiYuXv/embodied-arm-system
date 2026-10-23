"""Sonix 摄像头 (0c45:6360) PyQt6 画面测试程序

自动检测所有可用摄像头（含 MJPG 回退，兼容 Sonix），
横向显示全部画面，可切换分辨率。

用法: python3 test_sonix_pyqt6.py
"""

import sys
import cv2
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget
)

RESOLUTIONS = [
    ("640 x 480", 640, 480),
    ("1280 x 720", 1280, 720),
    ("1920 x 1080", 1920, 1080),
]


class CameraDetectThread(QThread):
    """后台检测摄像头索引，默认格式失败时回退 MJPG 重试（Sonix 必须）"""
    indices_found = pyqtSignal(list)

    def run(self):
        indices = []
        for idx in range(8):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                if not (ret and frame is not None and frame.size > 0):
                    cap.set(cv2.CAP_PROP_FOURCC,
                            cv2.VideoWriter_fourcc(*'MJPG'))
                    ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    indices.append(idx)
            cap.release()
        self.indices_found.emit(indices)


class CameraCaptureThread(QThread):
    frame_ready = pyqtSignal(int, object)

    def __init__(self, idx, cap):
        super().__init__()
        self.idx = idx
        self.cap = cap
        self._running = True

    def run(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.frame_ready.emit(self.idx, frame)
            self.msleep(10)

    def stop(self):
        self._running = False
        self.wait(500)


class SonixTestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sonix 摄像头测试")
        self.caps = {}
        self.threads = {}
        self.pending = {}
        self.capture_res = (640, 480)

        layout = QVBoxLayout(self)
        video_layout = QHBoxLayout()
        self.labels = []
        for i in range(3):
            label = QLabel(f"摄像头 {i + 1}\n未连接")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(
                "background-color: #222; color: #fff; font-size: 14px;")
            label.setMinimumSize(320, 240)
            self.labels.append(label)
            video_layout.addWidget(label)
        layout.addLayout(video_layout)

        bar = QHBoxLayout()
        self.res_combo = QComboBox()
        for name, *_ in RESOLUTIONS:
            self.res_combo.addItem(name)
        self.res_combo.currentIndexChanged.connect(self.change_resolution)
        bar.addWidget(QLabel("分辨率:"))
        bar.addWidget(self.res_combo)
        self.status_label = QLabel("正在检测摄像头...")
        self.status_label.setStyleSheet("color: #0078d7;")
        bar.addWidget(self.status_label, stretch=1)
        layout.addLayout(bar)

        self.detect = CameraDetectThread()
        self.detect.indices_found.connect(self.on_indices_found)
        self.detect.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_display)
        self.timer.start(66)

    def on_indices_found(self, indices):
        self.indices = indices
        if not indices:
            self.status_label.setText(
                "未检测到摄像头。Sonix 请确认: "
                "cat /sys/module/uvcvideo/parameters/quirks 输出 64，"
                "并已拔插一次 USB")
            return
        for idx in indices[:3]:
            self.open_camera(idx)
        self.status_label.setText(
            f"检测到 {len(indices)} 个摄像头: {indices}，显示前 3 路")

    def open_camera(self, idx):
        self.close_camera(idx)
        w, h = self.capture_res
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.status_label.setText(f"摄像头 [{idx}] 打开失败")
            return
        self.caps[idx] = cap
        t = CameraCaptureThread(idx, cap)
        t.frame_ready.connect(self.on_frame)
        t.start()
        self.threads[idx] = t

    def close_camera(self, idx):
        if idx in self.threads:
            self.threads[idx].stop()
            del self.threads[idx]
        if idx in self.caps:
            self.caps[idx].release()
            del self.caps[idx]

    def change_resolution(self, i):
        _, w, h = RESOLUTIONS[i]
        self.capture_res = (w, h)
        for idx in list(self.caps.keys()):
            self.open_camera(idx)

    def on_frame(self, idx, frame):
        self.pending[idx] = frame

    def refresh_display(self):
        for i, idx in enumerate(getattr(self, 'indices', [])[:3]):
            if idx not in self.pending:
                continue
            frame = self.pending[idx]
            label = self.labels[i]
            h, w = frame.shape[:2]
            scale = min((label.width() - 4) / w, (label.height() - 4) / h, 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))
        self.pending.clear()

    def closeEvent(self, event):
        for idx in list(self.threads.keys()):
            self.close_camera(idx)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SonixTestWindow()
    win.resize(1000, 420)
    win.show()
    sys.exit(app.exec())

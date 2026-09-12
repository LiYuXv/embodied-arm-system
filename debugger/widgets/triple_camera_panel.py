"""Three independent USB camera previews for the dual-arm debugger."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Reuse the existing capture and detection workers; this module deliberately
# does not import or instantiate CameraWidget's LeRobot/VLA features.
from debugger.widgets.camera_pyqt import (
    CameraCaptureThread,
    CameraDetectThread,
    cv2,
)

class CameraStreamWidget(QGroupBox):
    """One named, independently controlled video stream."""

    frame_captured = pyqtSignal(str, int, object)
    log_message = pyqtSignal(str)

    def __init__(self, stream_name: str, default_index: int = 0, parent=None,
                 capture_factory=None, thread_factory=None):
        super().__init__(stream_name, parent)
        self.stream_name = stream_name
        self.setObjectName(f"{stream_name}Camera")
        self._capture_factory = capture_factory or self._default_capture_factory
        self._thread_factory = thread_factory or CameraCaptureThread
        self._cap = None
        self._thread = None
        self._last_pixmap = None
        self._last_frame = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        self.video_label = QLabel("未启动")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(240, 160)
        self.video_label.setStyleSheet(
            "background: #10111d; color: #777b91; border: 1px solid #363850;"
        )
        layout.addWidget(self.video_label, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel("设备:"))
        self.index_combo = QComboBox()
        self.index_combo.setEditable(True)
        self.set_available_indices(list(range(8)), default_index)
        row.addWidget(self.index_combo)
        self.toggle_btn = QPushButton("启动")
        self.toggle_btn.clicked.connect(self.toggle_stream)
        row.addWidget(self.toggle_btn)
        self.status_label = QLabel("已停止")
        row.addWidget(self.status_label)
        row.addStretch()
        layout.addLayout(row)

    @staticmethod
    def _default_capture_factory(index):
        return cv2.VideoCapture(index, cv2.CAP_V4L2)

    @property
    def is_running(self):
        return self._thread is not None

    @property
    def camera_index(self):
        data = self.index_combo.currentData()
        if data is not None:
            return int(data)
        try:
            return int(self.index_combo.currentText().strip())
        except ValueError:
            return 0

    def set_available_indices(self, indices, preferred=None):
        current = self.camera_index if self.index_combo.count() else preferred
        selected = current if current in indices else preferred
        self.index_combo.clear()
        for index in indices:
            self.index_combo.addItem(f"/dev/video{index}", int(index))
        if not indices:
            self.index_combo.addItem("0", 0)
        target = selected if selected is not None else indices[0] if indices else 0
        for row in range(self.index_combo.count()):
            if self.index_combo.itemData(row) == target:
                self.index_combo.setCurrentIndex(row)
                break

    def toggle_stream(self):
        if self.is_running:
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self):
        if self.is_running:
            return True
        index = self.camera_index
        cap = self._capture_factory(index)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            self.status_label.setText("打开失败")
            self.log_message.emit(f"{self.stream_name}: 无法打开 /dev/video{index}")
            return False

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        thread = self._thread_factory(index, cap)
        thread.frame_ready.connect(self._on_frame)
        if hasattr(thread, "error_occurred"):
            thread.error_occurred.connect(self._on_capture_error)
        self._cap = cap
        self._thread = thread
        thread.start()
        self.index_combo.setEnabled(False)
        self.toggle_btn.setText("停止")
        self.status_label.setText(f"运行中 · video{index}")
        self.log_message.emit(f"{self.stream_name}: 已启动 /dev/video{index}")
        return True

    def stop_stream(self):
        thread, cap = self._thread, self._cap
        self._thread = None
        self._cap = None
        if thread is not None:
            thread.stop()
        if cap is not None:
            cap.release()
        self.index_combo.setEnabled(True)
        self.toggle_btn.setText("启动")
        self.status_label.setText("已停止")
        self.video_label.setText("未启动")
        self.video_label.setPixmap(QPixmap())
        self._last_pixmap = None
        self._last_frame = None
        if thread is not None or cap is not None:
            self.log_message.emit(f"{self.stream_name}: 已停止")

    def _on_frame(self, camera_index, frame):
        if frame is None or getattr(frame, "size", 0) == 0:
            return
        self._last_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        ).copy()
        self._last_pixmap = QPixmap.fromImage(image)
        self._render_last_frame()
        self.frame_captured.emit(self.stream_name, int(camera_index), frame)

    def _render_last_frame(self):
        if self._last_pixmap is None:
            return
        self.video_label.setPixmap(self._last_pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        ))

    def _on_capture_error(self, _camera_index, message):
        self.status_label.setText("采集错误")
        self.log_message.emit(f"{self.stream_name}: {message}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_last_frame()

    def cleanup(self):
        self.stop_stream()


class TripleCameraPanel(QWidget):
    """Container exposing exactly cam1, cam2 and cam3."""

    frame_captured = pyqtSignal(str, int, object)
    log_message = pyqtSignal(str)

    def __init__(self, parent=None, stream_factory=None):
        super().__init__(parent)
        self.setObjectName("tripleCameraPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        header = QHBoxLayout()
        header.addWidget(QLabel("三路摄像头"))
        self.detect_btn = QPushButton("检测设备")
        self.detect_btn.clicked.connect(self.detect_cameras)
        header.addWidget(self.detect_btn)
        header.addStretch()
        layout.addLayout(header)

        grid = QGridLayout()
        factory = stream_factory or CameraStreamWidget
        self.streams = {}
        for column, name in enumerate(("cam1", "cam2", "cam3")):
            stream = factory(name, column)
            stream.frame_captured.connect(self.frame_captured.emit)
            stream.log_message.connect(self.log_message.emit)
            self.streams[name] = stream
            grid.addWidget(stream, 0, column)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        layout.addLayout(grid, 1)
        self._detector = None

    def detect_cameras(self):
        if self._detector is not None and self._detector.isRunning():
            return
        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("检测中…")
        self._detector = CameraDetectThread(self)
        self._detector.indices_found.connect(self._on_indices_found)
        self._detector.finished.connect(lambda: self.detect_btn.setEnabled(True))
        self._detector.start()

    def _on_indices_found(self, indices):
        for offset, stream in enumerate(self.streams.values()):
            if not stream.is_running:
                preferred = indices[offset] if offset < len(indices) else offset
                stream.set_available_indices(indices or list(range(8)), preferred)
        self.detect_btn.setText("检测设备")
        self.log_message.emit(f"已检测到 {len(indices)} 路可用摄像头")

    def latest_frames(self):
        """Return the latest BGR frame for each canonical camera name."""
        return {
            name: stream._last_frame
            for name, stream in self.streams.items()
        }

    def cleanup(self):
        for stream in self.streams.values():
            stream.cleanup()
        if self._detector is not None and self._detector.isRunning():
            self._detector.requestInterruption()
            self._detector.wait(2000)

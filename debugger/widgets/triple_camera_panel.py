"""Three independent USB camera previews for the dual-arm debugger."""

import time

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
        self._autofocus_enabled = False
        self._focus_value = 0.5
        self._exposure_locked = True
        self._locked_exposure = None

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

        camera_controls = QHBoxLayout()
        self.chk_autofocus = QCheckBox("自动对焦")
        self.chk_autofocus.setChecked(self._autofocus_enabled)
        self.chk_autofocus.toggled.connect(self._on_autofocus_toggled)
        camera_controls.addWidget(self.chk_autofocus)

        camera_controls.addWidget(QLabel("焦距:"))
        self.focus_spin = QDoubleSpinBox()
        self.focus_spin.setRange(0.0, 1.0)
        self.focus_spin.setSingleStep(0.05)
        self.focus_spin.setDecimals(2)
        self.focus_spin.setValue(self._focus_value)
        self.focus_spin.setFixedWidth(70)
        self.focus_spin.setEnabled(not self._autofocus_enabled)
        self.focus_spin.valueChanged.connect(self._on_focus_changed)
        camera_controls.addWidget(self.focus_spin)

        self.chk_exposure_lock = QCheckBox("锁定曝光")
        self.chk_exposure_lock.setChecked(self._exposure_locked)
        self.chk_exposure_lock.setToolTip(
            "勾选：先让自动曝光收敛，再冻结当前曝光；\n"
            "取消：恢复自动曝光。"
        )
        self.chk_exposure_lock.toggled.connect(self._on_exposure_lock_toggled)
        camera_controls.addWidget(self.chk_exposure_lock)
        camera_controls.addStretch()
        layout.addLayout(camera_controls)

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

        self._apply_focus(cap)
        if self._exposure_locked:
            if self._locked_exposure is not None:
                self._apply_exposure_lock(cap)
            else:
                QTimer.singleShot(
                    800, lambda c=cap: self._apply_exposure_lock_deferred(c)
                )

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

    def _apply_focus(self, cap):
        if cap is None:
            return
        try:
            cap.set(
                cv2.CAP_PROP_AUTOFOCUS,
                1 if self._autofocus_enabled else 0,
            )
            if not self._autofocus_enabled:
                cap.set(cv2.CAP_PROP_FOCUS, float(self._focus_value))
        except Exception as exc:
            self.log_message.emit(f"{self.stream_name}: 设置对焦失败: {exc}")

    def _on_autofocus_toggled(self, checked):
        self._autofocus_enabled = bool(checked)
        self.focus_spin.setEnabled(not checked)
        self._apply_focus(self._cap)

    def _on_focus_changed(self, value):
        self._focus_value = float(value)
        if not self._autofocus_enabled:
            self._apply_focus(self._cap)

    def _set_exposure_mode(self, cap, auto):
        if cap is None:
            return False
        plans = (
            [(3, (3.0, 0.75)), (0, (0.0,))]
            if auto
            else [(1, (1.0, 0.25))]
        )
        for target, values in plans:
            for value in values:
                try:
                    if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value):
                        continue
                    mode = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                    if (
                        int(round(mode)) == target
                        or abs(mode - target * 0.25) < 1e-3
                    ):
                        return True
                except Exception:
                    continue
        self.log_message.emit(
            f"{self.stream_name}: "
            f"{'自动' if auto else '手动'}曝光模式切换失败"
        )
        return False

    def _apply_exposure_lock(self, cap):
        if cap is None:
            return
        try:
            if self._exposure_locked:
                if not self._set_exposure_mode(cap, auto=False):
                    return
                if self._locked_exposure is None:
                    time.sleep(0.05)
                    locked = cap.get(cv2.CAP_PROP_EXPOSURE)
                    if locked is not None and locked > 0:
                        self._locked_exposure = float(locked)
                else:
                    cap.set(
                        cv2.CAP_PROP_EXPOSURE,
                        float(self._locked_exposure),
                    )
                actual = cap.get(cv2.CAP_PROP_EXPOSURE)
                self.log_message.emit(
                    f"{self.stream_name}: 曝光已锁定 "
                    f"(value={actual:.1f})"
                )
            else:
                self._set_exposure_mode(cap, auto=True)
                self.log_message.emit(f"{self.stream_name}: 自动曝光已恢复")
        except Exception as exc:
            self.log_message.emit(f"{self.stream_name}: 设置曝光失败: {exc}")

    def _apply_exposure_lock_deferred(self, cap):
        if not self._exposure_locked or self._cap is not cap:
            return
        self._apply_exposure_lock(cap)

    def _on_exposure_lock_toggled(self, checked):
        self._exposure_locked = bool(checked)
        if not checked:
            self._locked_exposure = None
        self._apply_exposure_lock(self._cap)

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

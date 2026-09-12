"""Controls for 20 Hz, three-camera dual-arm LeRobot recording."""

import logging
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from debugger.backend.dual_arm_lerobot import (
    DUAL_ARM_CAMERA_KEYS,
    DUAL_ARM_FPS,
    DualArmLeRobotRecordThread,
    create_or_resume_dual_arm_dataset,
)
from debugger.widgets.camera_pyqt import cv2

logger = logging.getLogger("debugger")


class DualArmLeRobotPanel(QWidget):
    """Small GUI facade around the dual-arm dataset writer."""

    log_message = pyqtSignal(str)

    def __init__(self, dataset_root, *, frames_callback, state_callback,
                 torque_callback, parent=None, dataset_factory=None,
                 thread_factory=None, image_size=(480, 640)):
        super().__init__(parent)
        self.dataset_root = dataset_root
        self.frames_callback = frames_callback
        self.state_callback = state_callback
        self.torque_callback = torque_callback
        self.dataset_factory = (
            dataset_factory or create_or_resume_dual_arm_dataset
        )
        self.thread_factory = thread_factory or DualArmLeRobotRecordThread
        self.image_size = tuple(image_size)
        self.fps = DUAL_ARM_FPS
        self._dataset = None
        self._record_thread = None
        self._recording = False
        self._frame_count = 0
        self._last_missing_log = 0.0

        self._init_ui()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._push_frame)

    def _init_ui(self):
        group = QGroupBox("LeRobot 双臂数据采集")
        row = QHBoxLayout(group)
        row.setContentsMargins(7, 5, 7, 5)

        row.addWidget(QLabel("数据集:"))
        self.repo_input = QLineEdit("ela3_dual_demo")
        self.repo_input.setMaximumWidth(180)
        row.addWidget(self.repo_input)
        row.addWidget(QLabel("任务:"))
        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("pick up the object")
        row.addWidget(self.task_input, 1)
        row.addWidget(QLabel("前瞻帧:"))
        self.lookahead_spin = QSpinBox()
        self.lookahead_spin.setRange(1, 20)
        self.lookahead_spin.setValue(5)
        self.lookahead_spin.setToolTip(
            "action[t] = observation.state[t + N]"
        )
        row.addWidget(self.lookahead_spin)
        self.record_btn = QPushButton("开始双臂录制")
        self.record_btn.setObjectName("enableBtn")
        self.record_btn.clicked.connect(self.toggle_recording)
        row.addWidget(self.record_btn)
        self.status_label = QLabel(
            "20 Hz | cam1/cam2/cam3 | state/action 14 | torque 2"
        )
        row.addWidget(self.status_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(group)

    @property
    def is_recording(self):
        return self._recording

    def toggle_recording(self):
        if self._recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if self._recording:
            return True
        task = self.task_input.text().strip() or "default_task"
        if any("\u4e00" <= char <= "\u9fff" for char in task):
            QMessageBox.warning(
                self, "Task Format Error", "任务描述请使用英文"
            )
            return False
        repo_name = self.repo_input.text().strip() or "ela3_dual_demo"
        self.repo_input.setText(repo_name)

        try:
            images = self.frames_callback()
            missing = [
                name for name in DUAL_ARM_CAMERA_KEYS
                if images.get(name) is None
            ]
            if missing:
                raise ValueError(f"请先启动摄像头: {', '.join(missing)}")
            state = self.state_callback()
            if state is None or np.asarray(state).reshape(-1).shape != (14,):
                raise ValueError("请先连接双臂并等待 14 维关节反馈")
            dataset = self.dataset_factory(
                self.dataset_root,
                repo_name,
                image_size=self.image_size,
                fps=self.fps,
            )
        except Exception as exc:
            self.status_label.setText(f"无法开始: {exc}")
            self.log_message.emit(f"LeRobot 双臂录制启动失败: {exc}")
            return False

        self._dataset = dataset
        self._frame_count = 0
        self._record_thread = self.thread_factory(
            dataset,
            state_callback=self.state_callback,
            torque_callback=self.torque_callback,
            fps=self.fps,
            max_queue=5,
            action_lookahead=self.lookahead_spin.value(),
            parent=self,
        )
        self._record_thread.set_task_name(task)
        self._record_thread.frame_written.connect(self._on_frame_written)
        self._record_thread.finished_ok.connect(self._on_finished)
        self._record_thread.error.connect(self._on_error)
        self._record_thread.start()

        self._recording = True
        self._timer.start(int(1000 / self.fps))
        self.repo_input.setEnabled(False)
        self.task_input.setEnabled(False)
        self.lookahead_spin.setEnabled(False)
        self.record_btn.setText("停止双臂录制")
        self.record_btn.setObjectName("disconnectBtn")
        self.status_label.setText(f"录制中 · 20 Hz · {task}")
        self.log_message.emit(
            f"LeRobot 双臂录制已启动: {repo_name}, task={task}"
        )
        return True

    def _push_frame(self):
        if not self._recording or self._record_thread is None:
            return
        try:
            source_frames = self.frames_callback()
            images = {
                name: self._prepare_rgb(source_frames.get(name))
                for name in DUAL_ARM_CAMERA_KEYS
            }
            self._record_thread.push_frame(images)
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_missing_log >= 1.0:
                self._last_missing_log = now
                self.log_message.emit(f"LeRobot 等待同步帧: {exc}")

    def _prepare_rgb(self, frame):
        if frame is None:
            return None
        target_h, target_w = self.image_size
        height, width = frame.shape[:2]
        source_ratio = width / height
        target_ratio = target_w / target_h
        if abs(source_ratio - target_ratio) > 1e-3:
            if source_ratio > target_ratio:
                crop_width = int(height * target_ratio)
                x0 = (width - crop_width) // 2
                frame = frame[:, x0:x0 + crop_width]
            else:
                crop_height = int(width / target_ratio)
                y0 = (height - crop_height) // 2
                frame = frame[y0:y0 + crop_height, :]
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(
                frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _on_frame_written(self, count):
        self._frame_count = int(count)
        self.status_label.setText(f"录制中 · 20 Hz · {count} 帧")

    def stop_recording(self):
        if not self._recording:
            return
        self._recording = False
        self._timer.stop()
        self.record_btn.setEnabled(False)
        self.status_label.setText("正在保存 episode…")
        if self._record_thread is not None:
            self._record_thread.stop(flush=False)
        else:
            self._reset_ui("没有录制到帧")

    def _on_finished(self, result):
        self.log_message.emit(f"LeRobot 双臂录制: {result}")
        self._reset_ui(result)

    def _on_error(self, message):
        self.log_message.emit(f"LeRobot 双臂录制错误: {message}")
        self._recording = False
        self._timer.stop()
        self._reset_ui(f"录制失败: {message}")

    def _reset_ui(self, status):
        thread = self._record_thread
        self._record_thread = None
        self._dataset = None
        if thread is not None and not thread.isRunning():
            thread.deleteLater()
        self.repo_input.setEnabled(True)
        self.task_input.setEnabled(True)
        self.lookahead_spin.setEnabled(True)
        self.record_btn.setEnabled(True)
        self.record_btn.setText("开始双臂录制")
        self.record_btn.setObjectName("enableBtn")
        self.status_label.setText(status)

    def cleanup(self):
        self._timer.stop()
        thread = self._record_thread
        if thread is not None:
            if self._recording:
                self._recording = False
                thread.stop(flush=False)
            thread.wait(10000)
            self._record_thread = None
        self._dataset = None

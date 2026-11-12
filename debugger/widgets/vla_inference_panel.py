"""Controls and status display for dual-arm VLA inference."""

from pathlib import Path

from PyQt6.QtCore import QSignalBlocker, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


DEFAULT_DUAL_VLA_CHECKPOINT = str(
    Path(__file__).resolve().parents[1]
    / "ckpt"
    / "smolvla_dual_smoke_0923_manual"
    / "checkpoints"
    / "000100"
    / "pretrained_model"
)


class VLAInferencePanel(QWidget):
    """View-only VLA controls; inference/execution lives in the controller."""

    load_requested = pyqtSignal()
    single_requested = pyqtSignal()
    continuous_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    mode_changed = pyqtSignal(bool)
    checkpoint_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        group = QGroupBox("VLA 本地推理")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(7, 5, 7, 5)
        layout.setSpacing(4)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("检查点:"))
        self.checkpoint_input = QLineEdit(DEFAULT_DUAL_VLA_CHECKPOINT)
        self.checkpoint_input.setToolTip("本地 LeRobot pretrained_model 目录")
        self.checkpoint_input.textEdited.connect(self.checkpoint_changed)
        model_row.addWidget(self.checkpoint_input, 1)
        self.load_btn = QPushButton("加载模型")
        self.load_btn.clicked.connect(self.load_requested)
        model_row.addWidget(self.load_btn)
        layout.addLayout(model_row)

        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("任务:"))
        self.task_input = QLineEdit("pick up the object one by one")
        action_row.addWidget(self.task_input, 1)

        self.execute_checkbox = QCheckBox("真机执行")
        self.execute_checkbox.setToolTip(
            "默认为 Dry-run；勾选后会要求二次确认并向双臂发送绝对关节命令"
        )
        self.execute_checkbox.toggled.connect(self._confirm_execute_mode)
        action_row.addWidget(self.execute_checkbox)

        action_row.addWidget(QLabel("单步最大变化:"))
        self.max_delta_spin = QDoubleSpinBox()
        self.max_delta_spin.setDecimals(3)
        self.max_delta_spin.setRange(0.001, 0.500)
        self.max_delta_spin.setSingleStep(0.005)
        self.max_delta_spin.setValue(0.050)
        self.max_delta_spin.setSuffix(" rad")
        self.max_delta_spin.setToolTip(
            "相对于最新关节反馈的每条 20 Hz 指令最大变化；默认 0.05 rad"
        )
        action_row.addWidget(self.max_delta_spin)

        self.single_btn = QPushButton("单次推理")
        self.single_btn.clicked.connect(self.single_requested)
        action_row.addWidget(self.single_btn)
        self.start_btn = QPushButton("连续推理 / Start")
        self.start_btn.setObjectName("enableBtn")
        self.start_btn.clicked.connect(self.continuous_requested)
        action_row.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止 VLA")
        self.stop_btn.setObjectName("disconnectBtn")
        self.stop_btn.clicked.connect(self.stop_requested)
        action_row.addWidget(self.stop_btn)
        layout.addLayout(action_row)

        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("模式:"))
        self.mode_label = QLabel("DRY-RUN（不发送机械臂命令）")
        self.mode_label.setStyleSheet("color: #d18f00; font-weight: bold;")
        status_row.addWidget(self.mode_label)
        status_row.addSpacing(16)
        status_row.addWidget(QLabel("状态:"))
        self.status_label = QLabel("模型未加载")
        self.status_label.setStyleSheet("font-weight: bold;")
        status_row.addWidget(self.status_label, 1)
        self.contract_label = QLabel(
            "cam1/cam2/cam3 | state 14 | torque 2 | action [N,14] | 20 Hz"
        )
        status_row.addWidget(self.contract_label)
        layout.addLayout(status_row)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(group)
        self.set_runtime_state(model_loaded=False, busy=False, active=False)

    @property
    def checkpoint_path(self):
        return self.checkpoint_input.text().strip()

    @property
    def task(self):
        return self.task_input.text().strip()

    @property
    def execute_enabled(self):
        return self.execute_checkbox.isChecked()

    @property
    def max_delta_rad(self):
        return float(self.max_delta_spin.value())

    def _confirm_execute_mode(self, checked):
        if checked:
            answer = QMessageBox.warning(
                self,
                "确认 VLA 真机执行",
                "VLA 将以 20 Hz 向已连接、已使能的左右机械臂发送关节位置命令。\n"
                "请确认工作区无人且急停可用。是否启用真机执行？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                blocker = QSignalBlocker(self.execute_checkbox)
                self.execute_checkbox.setChecked(False)
                del blocker
                checked = False
        self.mode_label.setText(
            "EXECUTE（真机动作）"
            if checked
            else "DRY-RUN（不发送机械臂命令）"
        )
        self.mode_label.setStyleSheet(
            "color: #d33; font-weight: bold;"
            if checked
            else "color: #d18f00; font-weight: bold;"
        )
        self.mode_changed.emit(bool(checked))

    def set_status(self, text, *, error=False):
        self.status_label.setText(str(text))
        self.status_label.setStyleSheet(
            "color: #d33; font-weight: bold;" if error else "font-weight: bold;"
        )

    def set_runtime_state(self, *, model_loaded, busy, active, loading=False):
        locked = bool(busy or active or loading)
        self.load_btn.setEnabled(not locked)
        self.checkpoint_input.setEnabled(not locked)
        self.single_btn.setEnabled(bool(model_loaded) and not locked)
        self.start_btn.setEnabled(bool(model_loaded) and not locked)
        self.stop_btn.setEnabled(bool(busy or active))
        self.task_input.setEnabled(not locked)
        self.execute_checkbox.setEnabled(not locked)
        self.max_delta_spin.setEnabled(not locked)

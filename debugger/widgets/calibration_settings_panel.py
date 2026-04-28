"""标定设置面板：加载/编辑/保存惯性参数 + 启动标定程序"""

import os
import sys
import yaml
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QDoubleSpinBox, QCheckBox,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QTextEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal, QProcess
from PyQt6.QtGui import QTextCursor

from debugger.utils.i18n import tr
from debugger.utils.theme_manager import ThemeManager

logger = logging.getLogger("debugger")

SDK_ROOT = Path(__file__).parent.parent.parent

DEFAULT_INERTIA_PATH = SDK_ROOT / "resources" / "config" / "inertia_params.yaml"

LINK_NAMES = ["L2", "L3", "L4", "L5", "L6"]

DEFAULT_INERTIA_PARAMS = {
    "L2": {"mass": 0.465144, "com": [0.1016, 0.0, 0.0]},
    "L3": {"mass": 0.146738, "com": [-0.0864, -0.0544, 0.003]},
    "L4": {"mass": 0.946534, "com": [-0.0199, 0.0111, 0.0]},
    "L5": {"mass": 0.031, "com": [0.004, 0.0, 0.0196]},
    "L6": {"mass": 0.6317, "com": [0.0, -0.001, -0.0433]},
}


class CalibrationSettingsPanel(QWidget):
    """标定设置面板"""

    params_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._inertia_params: dict = {}
        self._current_file: Optional[Path] = None
        self._use_calibrated = False
        self._init_ui()
        self._auto_load_default()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        file_group = QGroupBox()
        file_layout = QHBoxLayout()

        self.load_btn = QPushButton(tr("set.load_file"))
        self.load_btn.clicked.connect(self._on_load)
        file_layout.addWidget(self.load_btn)

        self.save_btn = QPushButton(tr("set.save_file"))
        self.save_btn.clicked.connect(self._on_save)
        file_layout.addWidget(self.save_btn)

        self.use_calib_check = QCheckBox(tr("set.use_calib"))
        self.use_calib_check.setChecked(True)
        self.use_calib_check.stateChanged.connect(self._on_use_calib_changed)
        file_layout.addWidget(self.use_calib_check)

        self.file_label = QLabel(tr("set.not_loaded"))
        self.file_label.setObjectName("fileLabel")
        file_layout.addWidget(self.file_label)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        param_group = QGroupBox()
        param_layout = QVBoxLayout()

        self.param_table = QTableWidget(5, 5)
        self.param_table.setHorizontalHeaderLabels([
            tr("set.link"), tr("set.label_mass"), tr("set.x"), tr("set.y"), tr("set.z")
        ])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.param_table.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)

        for row, link in enumerate(LINK_NAMES):
            self.param_table.setItem(row, 0, QTableWidgetItem(link))
            self.param_table.item(row, 0).setFlags(Qt.ItemFlag.NoItemFlags)

            mass_spin = QDoubleSpinBox()
            mass_spin.setRange(0.0, 10.0)
            mass_spin.setDecimals(4)
            mass_spin.setSingleStep(0.001)
            self.param_table.setCellWidget(row, 1, mass_spin)

            for col in range(3):
                com_spin = QDoubleSpinBox()
                com_spin.setRange(-1.0, 1.0)
                com_spin.setDecimals(4)
                com_spin.setSingleStep(0.001)
                self.param_table.setCellWidget(row, 2 + col, com_spin)

        self.param_table.setMaximumHeight(180)
        param_layout.addWidget(self.param_table)

        self.calib_info_label = QLabel("")
        self.calib_info_label.setObjectName("calibInfo")
        param_layout.addWidget(self.calib_info_label)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        calib_group = QGroupBox()
        calib_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.start_calib_btn = QPushButton(tr("set.start_calib"))
        self.start_calib_btn.setEnabled(False)
        self.start_calib_btn.setToolTip("暂未实现")
        btn_layout.addWidget(self.start_calib_btn)

        self.stop_calib_btn = QPushButton(tr("set.stop_calib"))
        self.stop_calib_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_calib_btn)

        calib_layout.addLayout(btn_layout)

        self.calib_console = QTextEdit()
        self.calib_console.setReadOnly(True)
        self.calib_console.setMaximumHeight(120)
        calib_layout.addWidget(self.calib_console)

        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)

        layout.addStretch()

    def _auto_load_default(self):
        if DEFAULT_INERTIA_PATH.exists():
            self._load_from_file(DEFAULT_INERTIA_PATH)
        else:
            self._load_default_params()

    def _load_default_params(self):
        self._inertia_params = {link: dict(data) for link, data in DEFAULT_INERTIA_PARAMS.items()}
        self._current_file = None
        self._use_calibrated = False
        self._update_ui_from_params()
        self.file_label.setText(tr("set.not_loaded"))
        self.use_calib_check.setChecked(False)

    def _load_from_file(self, path: Path) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data and "inertia_params" in data:
                self._inertia_params = data.get("inertia_params", {})
                self._current_file = path
                self._use_calibrated = data.get("use_calibrated_params", False)

                self._update_ui_from_params()
                self.use_calib_check.setChecked(self._use_calibrated)
                self.file_label.setText(tr("set.loaded", path=path.name))

                cal_info = data.get("calibration_info", {})
                if cal_info:
                    date = cal_info.get("date", "N/A")
                    rmse = cal_info.get("rmse", 0.0)
                    r2 = cal_info.get("r_squared", 0.0)
                    samples = cal_info.get("num_samples", 0)
                    self.calib_info_label.setText(
                        f"{tr('set.calib_info')}: {date} | "
                        f"{tr('set.rmse', val=f'{rmse:.4f}')} | "
                        f"{tr('set.r2', val=f'{r2:.4f}')} | "
                        f"{tr('set.samples')}: {samples}"
                    )
                else:
                    self.calib_info_label.setText("")

                logger.info(f"Loaded inertia params from {path}")
                return True
            else:
                logger.warning(f"Invalid inertia params file: {path}")
                return False

        except Exception as e:
            logger.error(f"Failed to load inertia params: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load file: {e}")
            return False

    def _update_ui_from_params(self):
        for row, link in enumerate(LINK_NAMES):
            params = self._inertia_params.get(link, DEFAULT_INERTIA_PARAMS.get(link, {}))
            mass = params.get("mass", 0.0)
            com = params.get("com", [0.0, 0.0, 0.0])

            self.param_table.cellWidget(row, 1).setValue(mass)
            self.param_table.cellWidget(row, 2).setValue(com[0] if len(com) > 0 else 0.0)
            self.param_table.cellWidget(row, 3).setValue(com[1] if len(com) > 1 else 0.0)
            self.param_table.cellWidget(row, 4).setValue(com[2] if len(com) > 2 else 0.0)

    def _get_params_from_ui(self) -> dict:
        params = {}
        for row, link in enumerate(LINK_NAMES):
            mass = self.param_table.cellWidget(row, 1).value()
            com_x = self.param_table.cellWidget(row, 2).value()
            com_y = self.param_table.cellWidget(row, 3).value()
            com_z = self.param_table.cellWidget(row, 4).value()
            params[link] = {
                "mass": mass,
                "com": [com_x, com_y, com_z],
            }
        return params

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("set.load_file"),
            str(SDK_ROOT / "resources" / "config"),
            "YAML (*.yaml *.yml)",
        )
        if path:
            self._load_from_file(Path(path))

    def _on_save(self):
        self._inertia_params = self._get_params_from_ui()
        self._use_calibrated = self.use_calib_check.isChecked()

        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("set.save_file"),
            str(SDK_ROOT / "resources" / "config" / "inertia_params.yaml"),
            "YAML (*.yaml *.yml)",
        )
        if path:
            path = Path(path)
            data = {
                "use_calibrated_params": self._use_calibrated,
                "inertia_params": self._inertia_params,
            }
            try:
                with open(path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                self._current_file = path
                self.file_label.setText(tr("set.saved", path=path.name))
                logger.info(f"Saved inertia params to {path}")
                self.params_changed.emit(self._get_params_for_export())
            except Exception as e:
                logger.error(f"Failed to save inertia params: {e}")
                QMessageBox.warning(self, "Error", f"Failed to save: {e}")

    def _on_use_calib_changed(self, state):
        self._use_calibrated = state == Qt.CheckState.Checked.value
        self.params_changed.emit(self._get_params_for_export())

    def _get_params_for_export(self) -> dict:
        return {
            "use_calibrated": self._use_calibrated,
            "params": self._inertia_params,
            "file": str(self._current_file) if self._current_file else None,
        }

    def get_inertia_params(self) -> dict:
        if self._use_calibrated and self._inertia_params:
            return self._inertia_params
        return DEFAULT_INERTIA_PARAMS

    def is_use_calibrated(self) -> bool:
        return self._use_calibrated

    def _on_start_calib(self):
        calib_script = Path(__file__).parent.parent / "gravity_calibration.py"

        if not calib_script.exists():
            QMessageBox.warning(self, "Error", "重力标定程序不存在")
            return

        self.start_calib_btn.setEnabled(False)
        self.stop_calib_btn.setEnabled(True)
        self.calib_console.clear()
        self._append_console("="*50 + "\n")
        self._append_console("启动重力补偿标定程序\n")
        self._append_console("="*50 + "\n")

        self._calib_process = QProcess(self)
        self._calib_process.readyReadStandardOutput.connect(self._on_calib_output)
        self._calib_process.readyReadStandardError.connect(self._on_calib_error)
        self._calib_process.finished.connect(self._on_calib_finished)

        self._calib_process.setProgram(sys.executable)
        self._calib_process.setArguments([str(calib_script), "--quick"])
        self._calib_process.setWorkingDirectory(str(SDK_ROOT))

        self._calib_process.start()

    def _on_stop_calib(self):
        if hasattr(self, "_calib_process") and self._calib_process.state() == QProcess.ProcessState.Running:
            self._calib_process.terminate()
            self._append_console("\n正在停止标定程序...\n")

    def _on_calib_output(self):
        data = self._calib_process.readAllStandardOutput()
        text = bytes(data).decode("utf-8", errors="replace")
        self._append_console(text)

    def _on_calib_error(self):
        data = self._calib_process.readAllStandardError()
        text = bytes(data).decode("utf-8", errors="replace")
        if text.strip():
            self._append_console(text)

    def _on_calib_finished(self, exitCode, exitStatus):
        self.start_calib_btn.setEnabled(True)
        self.stop_calib_btn.setEnabled(False)
        self._append_console(f"\n标定完成 (退出码: {exitCode})\n")

        if exitCode == 0:
            self._load_from_file(DEFAULT_INERTIA_PATH)

    def _append_console(self, text: str):
        self.calib_console.moveCursor(QTextCursor.MoveOperation.End)
        self.calib_console.insertPlainText(text)
        self.calib_console.moveCursor(QTextCursor.MoveOperation.End)
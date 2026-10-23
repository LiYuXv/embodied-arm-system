# -*- coding: utf-8 -*-
"""
系统设置对话框（分页：频率设置 / 系统设置）。
"""

from PyQt6 import QtWidgets

from .settings_manager import load_settings, save_settings


class SystemSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统设置")
        self.setModal(True)
        self.resize(480, 420)
        self._settings = load_settings()
        self._build_ui()
        self._load_to_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_freq_tab(), "频率设置")
        self.tabs.addTab(self._build_sys_tab(), "系统设置")
        layout.addWidget(self.tabs)

        btn_row = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("加载")
        self.save_btn = QtWidgets.QPushButton("保存")
        self.apply_btn = QtWidgets.QPushButton("应用")
        self.ok_btn = QtWidgets.QPushButton("确定")
        self.cancel_btn = QtWidgets.QPushButton("取消")
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.save_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.ok_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.load_btn.clicked.connect(self._on_load)
        self.save_btn.clicked.connect(self._on_save)
        self.apply_btn.clicked.connect(self._on_apply)
        self.ok_btn.clicked.connect(self._on_ok)
        self.cancel_btn.clicked.connect(self.reject)

    def _build_freq_tab(self):
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)

        self.refresh_spin = QtWidgets.QSpinBox()
        self.refresh_spin.setRange(20, 5000)
        self.refresh_spin.setSuffix(" ms")
        form.addRow("状态刷新周期:", self.refresh_spin)

        self.mit_spin = QtWidgets.QSpinBox()
        self.mit_spin.setRange(1, 500)
        self.mit_spin.setSuffix(" ms")
        form.addRow("MIT 循环周期:", self.mit_spin)

        self.analyzer_spin = QtWidgets.QDoubleSpinBox()
        self.analyzer_spin.setRange(0.2, 60.0)
        self.analyzer_spin.setDecimals(1)
        self.analyzer_spin.setSuffix(" s")
        form.addRow("系统监测间隔:", self.analyzer_spin)

        self.timeout_spin = QtWidgets.QDoubleSpinBox()
        self.timeout_spin.setRange(0.05, 2.0)
        self.timeout_spin.setDecimals(2)
        self.timeout_spin.setSuffix(" s")
        form.addRow("串口超时:", self.timeout_spin)

        tip = QtWidgets.QLabel("频率修改点击「应用」立即生效。")
        tip.setStyleSheet("color: #6E6E73;")
        form.addRow(tip)
        return w

    def _build_sys_tab(self):
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)

        self.port_edit = QtWidgets.QLineEdit()
        form.addRow("默认串口:", self.port_edit)

        self.io_port_edit = QtWidgets.QLineEdit()
        form.addRow("升降IO串口:", self.io_port_edit)

        self.baud_combo = QtWidgets.QComboBox()
        self.baud_combo.addItems(["921600", "115200", "57600", "38400"])
        form.addRow("波特率:", self.baud_combo)

        self.kp_spin = QtWidgets.QDoubleSpinBox()
        self.kp_spin.setRange(0.0, 50.0)
        self.kp_spin.setDecimals(1)
        form.addRow("MIT kp 默认:", self.kp_spin)

        self.kd_spin = QtWidgets.QDoubleSpinBox()
        self.kd_spin.setRange(0.0, 1.5)
        self.kd_spin.setDecimals(2)
        form.addRow("MIT kd 默认:", self.kd_spin)

        self.auto_connect_check = QtWidgets.QCheckBox("启动后自动连接并使能电机")
        self.auto_connect_check.setChecked(True)
        form.addRow(self.auto_connect_check)

        tip = QtWidgets.QLabel("本页修改保存后建议重启程序生效。")
        tip.setStyleSheet("color: #6E6E73;")
        form.addRow(tip)
        return w

    def _load_to_ui(self):
        s = self._settings
        self.refresh_spin.setValue(int(s["refresh_ms"]))
        self.mit_spin.setValue(int(s["mit_interval_ms"]))
        self.analyzer_spin.setValue(float(s["analyzer_interval_s"]))
        self.timeout_spin.setValue(float(s["serial_timeout_s"]))
        self.port_edit.setText(str(s["default_port"]))
        self.io_port_edit.setText(str(s.get("default_io_port", "")))
        idx = self.baud_combo.findText(str(s["baud"]))
        self.baud_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.kp_spin.setValue(float(s["kp_default"]))
        self.kd_spin.setValue(float(s["kd_default"]))
        self.auto_connect_check.setChecked(bool(s.get("auto_connect_enable", True)))

    def _collect(self):
        return {
            "refresh_ms": self.refresh_spin.value(),
            "mit_interval_ms": self.mit_spin.value(),
            "analyzer_interval_s": self.analyzer_spin.value(),
            "serial_timeout_s": self.timeout_spin.value(),
            "default_port": self.port_edit.text().strip(),
            "default_io_port": self.io_port_edit.text().strip(),
            "baud": int(self.baud_combo.currentText()),
            "kp_default": self.kp_spin.value(),
            "kd_default": self.kd_spin.value(),
            "auto_connect_enable": self.auto_connect_check.isChecked(),
        }

    def _apply_to_parent(self):
        settings = self._collect()
        self._settings = settings
        save_settings(settings)
        parent = self.parent()
        if parent is not None and hasattr(parent, "apply_settings"):
            parent.apply_settings(settings)

    def _on_load(self):
        self._settings = load_settings()
        self._load_to_ui()

    def _on_save(self):
        settings = self._collect()
        self._settings = settings
        save_settings(settings)
        QtWidgets.QMessageBox.information(
            self,
            "提示",
            "设置已保存。\n频率设置点击「应用」立即生效；\n系统设置（串口/波特率/kp/kd）建议重启程序后生效。",
        )

    def _on_apply(self):
        self._apply_to_parent()
        QtWidgets.QMessageBox.information(
            self,
            "提示",
            "设置已应用。\n频率已实时生效；系统设置（串口/波特率/kp/kd）建议重启程序后生效。",
        )

    def _on_ok(self):
        self._apply_to_parent()
        self.accept()

"""实时监控面板：单窗口 7 关节温度实时曲线。"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel,
)
from PyQt6.QtCore import QTimer

import pyqtgraph as pg

from debugger.utils.i18n import tr
from debugger.utils.theme_manager import ThemeManager
from debugger.utils.style import SCENE_COLORS, JOINT_COLORS
from debugger.backend.data_buffer import DataBuffer

pg.setConfigOptions(antialias=True)

CHANNEL_NAMES = ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]


class MonitoringPanel(QWidget):
    """实时温度曲线监控面板（7 个关节温度绘制在同一曲线上）"""

    def __init__(self, data_buffer: DataBuffer, parent=None):
        super().__init__(parent)
        self.data_buffer = data_buffer
        self._paused = False
        self._plot = None
        self._curves = []
        self._init_ui()
        self.apply_theme()
        self.retranslate_ui()
        self._start_timer()

    def apply_theme(self):
        sc = SCENE_COLORS[ThemeManager.instance().theme]
        if self._plot is not None:
            self._plot.setBackground(sc["pg_bg"])

    def retranslate_ui(self):
        if self._plot is not None:
            self._plot.setTitle(tr("mon.title_temp"))
            self._plot.setLabel("bottom", tr("mon.time"), units="s")
        self.pause_btn.setText(tr("mon.resume") if self._paused else tr("mon.pause"))
        self.clear_btn.setText(tr("mon.clear"))
        self.export_btn.setText(tr("mon.export"))

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # ---- 顶部控制栏 ----
        top_bar = QHBoxLayout()
        self.pause_btn = QPushButton()
        self.pause_btn.setFixedWidth(60)
        self.pause_btn.clicked.connect(self._toggle_pause)
        top_bar.addWidget(self.pause_btn)

        self.clear_btn = QPushButton()
        self.clear_btn.setFixedWidth(60)
        self.clear_btn.clicked.connect(self._clear_data)
        top_bar.addWidget(self.clear_btn)

        self.export_btn = QPushButton()
        self.export_btn.setFixedWidth(80)
        self.export_btn.clicked.connect(self._export_csv)
        top_bar.addWidget(self.export_btn)

        top_bar.addSpacing(20)
        self._channel_swatches: list[QLabel] = []
        for ch in range(7):
            swatch = QLabel()
            swatch.setStyleSheet(
                f"color: {JOINT_COLORS[ch]}; font-weight: bold; font-size: 11px;"
            )
            top_bar.addWidget(swatch)
            self._channel_swatches.append(swatch)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        # ---- 主体：单一温度曲线（7 关节同图） ----
        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLabel("left", tr("mon.title_temp"))

        for ch in range(7):
            pen = pg.mkPen(color=JOINT_COLORS[ch], width=1.5)
            curve = self._plot.plot([], [], pen=pen)
            self._curves.append(curve)

        layout.addWidget(self._plot, stretch=1)

        for ch, lbl in enumerate(self._channel_swatches):
            lbl.setText(f"■ {CHANNEL_NAMES[ch]}")

    def _start_timer(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_plots)
        self._timer.start(50)  # 20Hz display refresh

    def _update_plots(self):
        if self._paused or self.data_buffer.count == 0:
            return

        ts, _pos, _vel, _torq, temp = self.data_buffer.get_data()
        if len(ts) == 0:
            return

        t_rel = ts - ts[0]
        for ch in range(7):
            self._curves[ch].setData(t_rel, temp[:, ch])

    def _toggle_pause(self):
        self._paused = not self._paused
        self.pause_btn.setText(tr("mon.resume") if self._paused else tr("mon.pause"))

    def _clear_data(self):
        self.data_buffer.clear()
        for c in self._curves:
            c.setData([], [])

    def _export_csv(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, tr("mon.export_title"), "arm_data.csv", "CSV Files (*.csv)"
        )
        if filepath:
            self.data_buffer.export_csv(filepath)
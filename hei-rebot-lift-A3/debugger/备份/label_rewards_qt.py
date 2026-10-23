#!/usr/bin/env python3
"""
片段级奖励标注工具 (PyQt6)

功能:
  1. 选择 lerobot_data 下的数据集与 episode
  2. 双摄 (cam1/cam2) 同步回放
  3. 用快捷键/按钮标记区间起始与结束，分配 good/bad/neutral 标签
  4. 时间轴可视化已标注区间 (彩色色带)
  5. 标注保存到 meta/reward_annotations.json
  6. 区间列表可编辑/删除

RECAP 映射:
    good -> advantage_indicator = 1  (I_t=True)
    bad  -> advantage_indicator = -1 (I_t=False)

用法:
    python3 label_rewards_qt.py --root ./lerobot_data
    python3 label_rewards_qt.py --root ./lerobot_data --dataset ela3_demo2

快捷键:
    Space   播放/暂停
    左/右   后退/前进一帧
    G/B/N   设当前标签为 good/bad/neutral
    [       标记区间起点
    ]       标记区间终点
    Enter   添加区间 (用当前 [start,end] + 当前标签)
    Ctrl+S  保存
    Ctrl+Z  撤销最后一个区间
"""

import sys
import os
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np

# ── 路径修正，确保能导入同目录模块 ──
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from annotation_store import (
    AnnotationStore,
    LABEL_COLORS,
    LABEL_ADVANTAGE,
    ALL_LABELS,
)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QMessageBox, QFrame, QHeaderView,
    QTableWidget, QTableWidgetItem, QGroupBox, QComboBox, QCheckBox,
    QProgressBar, QStatusBar, QSlider, QLineEdit, QSpinBox, QSplitter,
    QSizePolicy, QDialog, QTextEdit, QListWidget, QListWidgetItem,
    QAbstractItemView,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF, QSize
from PyQt6.QtGui import (
    QFont, QImage, QPixmap, QPainter, QBrush, QPen, QColor, QPolygonF,
    QAction, QKeySequence, QIcon, QShortcut,
)


# ═══════════════════════════════════════════════════════════════
#  数据集发现
# ═══════════════════════════════════════════════════════════════

def discover_datasets(root: Path) -> list:
    """扫描 root 下所有含 meta/info.json 的子目录。"""
    datasets = []
    if not root.exists():
        return datasets
    for child in sorted(root.iterdir()):
        if (child / "meta" / "info.json").exists():
            datasets.append(child.name)
    return datasets


# ═══════════════════════════════════════════════════════════════
#  轻量数据集封装 (延迟导入 OfflineDataset，避免循环依赖)
# ═══════════════════════════════════════════════════════════════

def load_offline_dataset(root_path: str):
    """从 visualize_data_qt 导入 OfflineDataset。"""
    try:
        from visualize_data_qt import OfflineDataset
        return OfflineDataset(root_path)
    except Exception as e:
        print(f"[label_rewards] 无法导入 OfflineDataset: {e}")
        print("  请确保 visualize_data_qt.py 在同一目录。")
        return None


# ═══════════════════════════════════════════════════════════════
#  回放线程
# ═══════════════════════════════════════════════════════════════

class PlaybackThread(QThread):
    """顺序回放 episode，发出帧索引信号。"""
    frame_changed = pyqtSignal(int)  # 当前帧索引 (episode 内)

    def __init__(self, fps=20.0, parent=None):
        super().__init__(parent)
        self.fps = fps
        self._running = False
        self._paused = False
        self._current = 0
        self._total = 0
        self._exit = False

    def configure(self, total_frames, start=0):
        self._total = total_frames
        self._current = start

    def run(self):
        self._running = True
        self._paused = False
        self._exit = False
        interval = 1.0 / self.fps
        while not self._exit:
            if self._paused:
                self.msleep(20)
                continue
            if self._total <= 0:
                self.msleep(20)
                continue
            if self._current >= self._total:
                self._current = self._total - 1
                self._paused = True
                continue
            self.frame_changed.emit(self._current)
            self._current += 1
            self.msleep(int(interval * 1000))

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def toggle(self):
        self._paused = not self._paused

    def is_paused(self):
        return self._paused

    def seek(self, frame):
        self._current = max(0, min(frame, self._total - 1))
        self.frame_changed.emit(self._current)

    def step(self, delta):
        self._current = max(0, min(self._current + delta, self._total - 1))
        self.frame_changed.emit(self._current)

    def current_frame(self):
        return self._current

    def stop(self):
        self._exit = True
        self.wait(2000)


# ═══════════════════════════════════════════════════════════════
#  时间轴 Widget (显示进度 + 已标注区间)
# ═══════════════════════════════════════════════════════════════

class TimelineWidget(QWidget):
    """时间轴：显示已标注区间色带 + 当前位置 + 待确认区间。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(90)
        self.setMaximumHeight(90)
        self._total = 0
        self._current = 0
        self._intervals = []          # 已保存的区间
        self._pending = None          # (start, end, label) 或 None
        self._start_mark = None       # 标记的起点

    def set_total(self, total):
        self._total = max(0, total)
        self.update()

    def set_current(self, frame):
        self._current = frame
        self.update()

    def set_intervals(self, intervals):
        self._intervals = intervals
        self.update()

    def set_start_mark(self, frame):
        self._start_mark = frame
        self.update()

    def clear_start_mark(self):
        self._start_mark = None
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()

        # 背景
        p.fillRect(self.rect(), QColor("#1e1e2e"))

        if self._total <= 0:
            p.setPen(QColor("#888888"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "无数据")
            return

        # 时间轴轨道
        track_y = h - 28
        track_h = 18
        p.setBrush(QColor("#2d2d44"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(4, track_y, w - 8, track_h, 4, 4)

        # 已标注区间
        for iv in self._intervals:
            x1 = 4 + int((iv["start"] / self._total) * (w - 8))
            x2 = 4 + int((iv["end"] / self._total) * (w - 8))
            color = QColor(LABEL_COLORS.get(iv["label"], "#888888"))
            p.setBrush(color)
            p.drawRoundedRect(x1, track_y, max(2, x2 - x1), track_h, 3, 3)

        # 待确认区间 (start_mark 到 current)
        if self._start_mark is not None:
            x1 = 4 + int((self._start_mark / self._total) * (w - 8))
            x2 = 4 + int((self._current / self._total) * (w - 8))
            p.setBrush(QColor(255, 255, 255, 80))
            p.drawRect(min(x1, x2), track_y, abs(x2 - x1), track_h)
            p.setPen(QColor("#ffffff"))
            p.drawText(min(x1, x2), track_y - 4,
                       f"[{self._start_mark} → {self._current}]")

        # 当前位置指示线
        cx = 4 + int((self._current / max(1, self._total - 1)) * (w - 8))
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawLine(cx, track_y - 6, cx, track_y + track_h + 6)

        # 刻度文字
        p.setPen(QColor("#aaaaaa"))
        font = p.font()
        font.setPointSize(8)
        p.setFont(font)
        p.drawText(4, h - 4, f"frame {self._current} / {self._total - 1}")


# ═══════════════════════════════════════════════════════════════
#  主窗口
# ═══════════════════════════════════════════════════════════════

class LabelingWindow(QMainWindow):
    def __init__(self, root="./lerobot_data", dataset_name=None):
        super().__init__()
        self.setWindowTitle("片段级奖励标注工具 (RECAP-style)")
        self.resize(1500, 950)

        self.root = Path(root)
        self.datasets = discover_datasets(self.root)
        self.dataset = None          # OfflineDataset 实例
        self.store = None            # AnnotationStore 实例
        self.current_ep = 0
        self.ep_length = 0
        self.cameras = []            # 可用相机名列表

        # 标注会话状态
        self.current_label = "good"
        self.start_mark = None       # 区间起点 (帧号)
        self.playback = PlaybackThread(parent=self)
        self.playback.frame_changed.connect(self._on_frame_changed)

        self._build_ui()
        self._build_shortcuts()

        # 默认选择数据集
        if dataset_name and dataset_name in self.datasets:
            self.dataset_combo.setCurrentText(dataset_name)
        elif self.datasets:
            self.dataset_combo.setCurrentText(self.datasets[0])

    # ─────────── UI 构建 ───────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QVBoxLayout(central)

        # ── 顶部：数据集/episode 选择 ──
        top = QHBoxLayout()
        top.addWidget(QLabel("数据集:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(self.datasets)
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        top.addWidget(self.dataset_combo, 1)

        top.addWidget(QLabel("Episode:"))
        self.episode_combo = QComboBox()
        self.episode_combo.currentIndexChanged.connect(self._on_episode_changed)
        top.addWidget(self.episode_combo)

        self.info_label = QLabel("—")
        top.addWidget(self.info_label, 1)
        main_lay.addLayout(top)

        # ── 中部：视频 + 右侧控制 ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 视频区
        video_box = QGroupBox("回放")
        video_lay = QVBoxLayout(video_box)
        # 动态创建两个相机标签
        self.cam_labels = {}   # camera_name -> QLabel
        for name in ["observation.images.cam1", "observation.images.cam2"]:
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(480, 270)
            lbl.setStyleSheet("background:#000;")
            self.cam_labels[name] = lbl
        # 横排放两个相机
        cam_row = QHBoxLayout()
        for name in ["observation.images.cam1", "observation.images.cam2"]:
            cam_row.addWidget(self.cam_labels[name])
        video_lay.addLayout(cam_row)

        # 时间轴
        self.timeline = TimelineWidget()
        video_lay.addWidget(self.timeline)

        # 播放控制
        ctrl_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_prev = QPushButton("◀ 前一帧")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next = QPushButton("后一帧 ▶")
        self.btn_next.clicked.connect(lambda: self._step(1))
        self.btn_start = QPushButton("|◀ 到头")
        self.btn_start.clicked.connect(lambda: self._seek(0))
        self.btn_end = QPushButton("▶| 到尾")
        self.btn_end.clicked.connect(lambda: self._seek(self.ep_length - 1))

        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(0)
        self.frame_spin.valueChanged.connect(self._seek)

        ctrl_row.addWidget(self.btn_start)
        ctrl_row.addWidget(self.btn_prev)
        ctrl_row.addWidget(self.btn_play)
        ctrl_row.addWidget(self.btn_next)
        ctrl_row.addWidget(self.btn_end)
        ctrl_row.addWidget(QLabel("跳转:"))
        ctrl_row.addWidget(self.frame_spin)
        ctrl_row.addStretch()
        video_lay.addLayout(ctrl_row)

        splitter.addWidget(video_box)

        # 右侧控制面板
        right_box = QGroupBox("标注")
        right_lay = QVBoxLayout(right_box)

        # 当前标签选择
        lbl_grp = QFormLayout()
        self.label_combo = QComboBox()
        self.label_combo.addItems(ALL_LABELS)
        self.label_combo.setCurrentText("good")
        self.label_combo.currentTextChanged.connect(self._on_label_changed)
        lbl_grp.addRow("当前标签:", self.label_combo)

        self.start_spin = QSpinBox()
        self.start_spin.setMaximum(999999)
        self.end_spin = QSpinBox()
        self.end_spin.setMaximum(999999)
        lbl_grp.addRow("区间起点:", self.start_spin)
        lbl_grp.addRow("区间终点:", self.end_spin)

        right_lay.addLayout(lbl_grp)

        # 标记按钮
        mark_row = QHBoxLayout()
        self.btn_mark_start = QPushButton("[ 设起点")
        self.btn_mark_start.clicked.connect(self._mark_start)
        self.btn_mark_end = QPushButton("设终点 ]")
        self.btn_mark_end.clicked.connect(self._mark_end)
        mark_row.addWidget(self.btn_mark_start)
        mark_row.addWidget(self.btn_mark_end)
        right_lay.addLayout(mark_row)

        self.btn_add = QPushButton("✚ 添加区间")
        self.btn_add.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn_add.clicked.connect(self._add_interval_from_inputs)
        right_lay.addWidget(self.btn_add)

        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("备注 (可选)")
        right_lay.addWidget(QLabel("备注:"))
        right_lay.addWidget(self.note_edit)

        right_lay.addWidget(QLabel("快捷操作:"))
        quick_row = QHBoxLayout()
        self.btn_quick_good = QPushButton("✓ 好")
        self.btn_quick_good.setStyleSheet(f"background:{LABEL_COLORS['good']}; color:#fff; font-weight:bold;")
        self.btn_quick_good.clicked.connect(lambda: self._quick_mark("good"))
        self.btn_quick_bad = QPushButton("✗ 坏")
        self.btn_quick_bad.setStyleSheet(f"background:{LABEL_COLORS['bad']}; color:#fff; font-weight:bold;")
        self.btn_quick_bad.clicked.connect(lambda: self._quick_mark("bad"))
        self.btn_quick_neutral = QPushButton("〜 中")
        self.btn_quick_neutral.setStyleSheet(f"background:{LABEL_COLORS['neutral']}; color:#fff;")
        self.btn_quick_neutral.clicked.connect(lambda: self._quick_mark("neutral"))
        quick_row.addWidget(self.btn_quick_good)
        quick_row.addWidget(self.btn_quick_bad)
        quick_row.addWidget(self.btn_quick_neutral)
        right_lay.addLayout(quick_row)

        # 整 episode 标注为成功/失败
        right_lay.addWidget(QLabel("整 episode 快标:"))
        ep_row = QHBoxLayout()
        self.btn_ep_good = QPushButton("全部=好")
        self.btn_ep_good.clicked.connect(lambda: self._mark_whole_episode("good"))
        self.btn_ep_bad = QPushButton("全部=坏")
        self.btn_ep_bad.clicked.connect(lambda: self._mark_whole_episode("bad"))
        ep_row.addWidget(self.btn_ep_good)
        ep_row.addWidget(self.btn_ep_bad)
        right_lay.addLayout(ep_row)

        # 区间列表
        right_lay.addWidget(QLabel("已标注区间:"))
        self.interval_table = QTableWidget(0, 5)
        self.interval_table.setHorizontalHeaderLabels(
            ["#", "起点", "终点", "标签", "备注"])
        self.interval_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.interval_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.interval_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        right_lay.addWidget(self.interval_table)

        del_row = QHBoxLayout()
        self.btn_del = QPushButton("✖ 删除选中")
        self.btn_del.clicked.connect(self._delete_selected_interval)
        self.btn_undo = QPushButton("↶ 撤销最后")
        self.btn_undo.clicked.connect(self._undo_last)
        self.btn_clear_ep = QPushButton("清空本 episode")
        self.btn_clear_ep.clicked.connect(self._clear_episode)
        del_row.addWidget(self.btn_del)
        del_row.addWidget(self.btn_undo)
        del_row.addWidget(self.btn_clear_ep)
        right_lay.addLayout(del_row)

        # 保存
        self.btn_save = QPushButton("💾 保存标注 (Ctrl+S)")
        self.btn_save.setStyleSheet("padding:8px; font-weight:bold;")
        self.btn_save.clicked.connect(self._save)
        right_lay.addWidget(self.btn_save)

        right_box.setMaximumWidth(460)
        splitter.addWidget(right_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_lay.addWidget(splitter, 1)

        # 底部状态栏
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._update_status()

    def _build_shortcuts(self):
        def mk(seq, fn):
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(fn)
        mk("Space", self._toggle_play)
        mk("Left", lambda: self._step(-1))
        mk("Right", lambda: self._step(1))
        mk("G", lambda: self.label_combo.setCurrentText("good"))
        mk("B", lambda: self.label_combo.setCurrentText("bad"))
        mk("N", lambda: self.label_combo.setCurrentText("neutral"))
        mk("[", self._mark_start)
        mk("]", self._mark_end)
        mk("Return", self._add_interval_from_inputs)
        mk("Ctrl+S", self._save)
        mk("Ctrl+Z", self._undo_last)
        mk("Ctrl+D", self._delete_selected_interval)

    # ─────────── 数据集 / episode 切换 ───────────

    def _on_dataset_changed(self, name):
        if not name:
            return
        ds_root = self.root / name
        self.dataset = load_offline_dataset(str(ds_root))
        if self.dataset is None:
            return
        self.store = AnnotationStore(str(ds_root))
        self.store.load()

        # 找可用相机
        self.cameras = [
            k for k, v in self.dataset.features.items()
            if v.get("dtype") in ("video", "image") or "image" in k.lower()
        ]
        self._refresh_episode_combo()
        self.status.showMessage(f"已加载数据集 {name}: {len(self.dataset.episodes)} episodes, 相机={self.cameras}")

    def _refresh_episode_combo(self):
        self.episode_combo.blockSignals(True)
        self.episode_combo.clear()
        for i in range(len(self.dataset.episodes)):
            length = self._ep_length(i)
            task = ""
            try:
                ti = self.dataset.episodes[i].get("task_index", None)
                if ti is not None:
                    for t in self.dataset.tasks:
                        if t.get("task_index") == int(ti):
                            task = t.get("task", "")
                            break
            except Exception:
                pass
            label = f"ep{i} [{length}f] {task}"
            self.episode_combo.addItem(label)
        self.episode_combo.blockSignals(False)
        if self.episode_combo.count() > 0:
            self.episode_combo.setCurrentIndex(0)
            self._on_episode_changed(0)

    def _ep_length(self, ep_idx):
        ep = self.dataset.episodes[ep_idx]
        if "length" in ep:
            return int(ep["length"])
        if "dataset_from_index" in ep and "dataset_to_index" in ep:
            return int(ep["dataset_to_index"]) - int(ep["dataset_from_index"])
        try:
            return len(self.dataset.get_episode_frames(ep_idx))
        except Exception:
            return 0

    def _on_episode_changed(self, idx):
        if idx < 0:
            return
        self.current_ep = idx
        self.ep_length = self._ep_length(idx)

        self.frame_spin.blockSignals(True)
        self.frame_spin.setMaximum(max(0, self.ep_length - 1))
        self.frame_spin.setValue(0)
        self.start_spin.setMaximum(max(0, self.ep_length - 1))
        self.end_spin.setMaximum(max(0, self.ep_length - 1))
        self.frame_spin.blockSignals(False)

        self.timeline.set_total(self.ep_length)
        self.start_mark = None
        self.timeline.clear_start_mark()

        # 配置回放
        self.playback.configure(self.ep_length, 0)
        self._refresh_interval_table()
        self._refresh_timeline()
        self._show_frame(0)
        self._update_status()

    # ─────────── 回放 ───────────

    def _toggle_play(self):
        if not self.playback.isRunning():
            self.playback.start()
        self.playback.toggle()
        self.btn_play.setText("⏸ 暂停" if not self.playback.is_paused() else "▶ 播放")

    def _step(self, delta):
        if not self.playback.isRunning():
            self.playback.start()
            self.playback.pause()
        self.playback.step(delta)
        self.btn_play.setText("▶ 播放")

    def _seek(self, frame):
        if frame is None:
            return
        if not self.playback.isRunning():
            self.playback.start()
            self.playback.pause()
        self.playback.seek(int(frame))
        self.btn_play.setText("▶ 播放")

    def _on_frame_changed(self, frame):
        self._show_frame(frame)
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(frame)
        self.frame_spin.blockSignals(False)
        self.timeline.set_current(frame)

    def _show_frame(self, frame):
        """显示指定帧的所有相机图像。"""
        if self.dataset is None:
            return
        for cam_name, lbl in self.cam_labels.items():
            img = self.dataset.get_video_frame(self.current_ep, frame, cam_name)
            if img is not None:
                h, w = img.shape[:2]
                qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(
                    lbl.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                lbl.setPixmap(pix)
            else:
                lbl.setText(f"{cam_name}\n(无视频帧)")

    # ─────────── 标注操作 ───────────

    def _on_label_changed(self, label):
        self.current_label = label

    def _mark_start(self):
        self.start_mark = self.playback.current_frame()
        self.start_spin.setValue(self.start_mark)
        self.timeline.set_start_mark(self.start_mark)
        self.status.showMessage(f"区间起点设为 {self.start_mark}")

    def _mark_end(self):
        self.end_spin.setValue(self.playback.current_frame())
        self.status.showMessage(f"区间终点设为 {self.end_spin.value()}")

    def _add_interval_from_inputs(self):
        start = self.start_spin.value()
        end = self.end_spin.value()
        if end < start:
            start, end = end, start
        label = self.label_combo.currentText()
        note = self.note_edit.text().strip()
        self.store.add_interval(self.current_ep, start, end, label, note)
        self.note_edit.clear()
        self.start_mark = None
        self.timeline.clear_start_mark()
        self._refresh_interval_table()
        self._refresh_timeline()
        self._update_status()
        self.status.showMessage(f"已添加区间 [{start},{end}] = {label}")

    def _quick_mark(self, label):
        """快速标记：以当前帧为起点，前推 N 帧或用 start_mark。"""
        cur = self.playback.current_frame()
        if self.start_mark is not None and self.start_mark != cur:
            start, end = min(self.start_mark, cur), max(self.start_mark, cur)
        else:
            start = cur
            end = cur
        self.store.add_interval(self.current_ep, start, end, label, "")
        self.start_mark = None
        self.timeline.clear_start_mark()
        self._refresh_interval_table()
        self._refresh_timeline()
        self._update_status()
        self.status.showMessage(f"快标 [{start},{end}] = {label}")

    def _mark_whole_episode(self, label):
        if self.ep_length <= 0:
            return
        self.store.clear_episode(self.current_ep)
        self.store.add_interval(self.current_ep, 0, self.ep_length - 1, label, "whole_episode")
        self._refresh_interval_table()
        self._refresh_timeline()
        self._update_status()

    def _delete_selected_interval(self):
        rows = sorted(
            {idx.row() for idx in self.interval_table.selectedIndexes()},
            reverse=True)
        for r in rows:
            self.store.remove_interval(self.current_ep, r)
        self._refresh_interval_table()
        self._refresh_timeline()
        self._update_status()

    def _undo_last(self):
        intervals = self.store.get_intervals(self.current_ep)
        if intervals:
            self.store.remove_interval(self.current_ep, len(intervals) - 1)
            self._refresh_interval_table()
            self._refresh_timeline()
            self._update_status()

    def _clear_episode(self):
        ret = QMessageBox.question(
            self, "确认", f"清空 episode {self.current_ep} 的所有标注？")
        if ret == QMessageBox.StandardButton.Yes:
            self.store.clear_episode(self.current_ep)
            self._refresh_interval_table()
            self._refresh_timeline()
            self._update_status()

    # ─────────── 刷新 UI ───────────

    def _refresh_interval_table(self):
        intervals = self.store.get_intervals(self.current_ep)
        self.interval_table.setRowCount(len(intervals))
        for i, iv in enumerate(intervals):
            items = [
                str(i),
                str(iv["start"]),
                str(iv["end"]),
                iv["label"],
                iv.get("note", ""),
            ]
            for col, txt in enumerate(items):
                item = QTableWidgetItem(txt)
                if col == 3:
                    item.setForeground(QBrush(QColor(LABEL_COLORS.get(txt, "#888888"))))
                self.interval_table.setItem(i, col, item)

    def _refresh_timeline(self):
        intervals = self.store.get_intervals(self.current_ep)
        self.timeline.set_intervals(intervals)

    def _update_status(self):
        if self.store is None:
            return
        s = self.store.stats()
        n_eps = self.store.labeled_episode_count()
        total_eps = len(self.dataset.episodes) if self.dataset else 0
        msg = (f"已标注 episodes: {n_eps}/{total_eps} | "
               f"区间数: {s['_intervals']} | "
               f"good帧: {s['good']} bad帧: {s['bad']} neutral帧: {s['neutral']}")
        self.status.showMessage(msg)

    # ─────────── 保存 ───────────

    def _save(self):
        if self.store is None:
            return
        self.store.save()
        self.status.showMessage(f"✓ 已保存到 {self.store.filepath}", 5000)

    # ─────────── 关闭 ───────────

    def closeEvent(self, event):
        if self.store is not None and self.store.stats()["_intervals"] > 0:
            ret = QMessageBox.question(
                self, "保存", "退出前保存标注？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel)
            if ret == QMessageBox.StandardButton.Yes:
                self._save()
            elif ret == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
        self.playback.stop()
        if self.dataset is not None:
            self.dataset.release_video_caps()
        event.accept()


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="片段级奖励标注工具")
    parser.add_argument("--root", default="./lerobot_data",
                        help="lerobot_data 根目录")
    parser.add_argument("--dataset", default=None,
                        help="直接指定数据集名")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = LabelingWindow(root=args.root, dataset_name=args.dataset)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

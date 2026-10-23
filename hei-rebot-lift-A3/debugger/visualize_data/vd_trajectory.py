# -*- coding: utf-8 -*-
"""轨迹曲线面板：Episode XYZ 位置轨迹曲线、子任务/语言/RECAP 标注、数据集合并分页。"""
import vd_common as _vd_common  # noqa: F401  确保 sys.path / matplotlib 后端已初始化

import os
import json
import math
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QMessageBox, QFrame, QHeaderView, QTableWidget, QTableWidgetItem, QGroupBox, QComboBox, QDoubleSpinBox, QCheckBox, QProgressBar, QSlider, QLineEdit, QSizePolicy, QTextEdit, QSpinBox, QTabWidget, QAbstractItemView, QListWidget, QListWidgetItem, QPlainTextEdit
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from auto_language_annotate import color_for_index, segment_fixed_count
from annotation_store import (
    LABEL_COLORS as RECAP_LABEL_COLORS,
    ALL_LABELS as RECAP_ALL_LABELS,
    has_annotations as recap_has_annotations,
    backup_annotations as recap_backup_annotations,
)
from merge_lerobot_datasets import MergeWorker, preflight_check


# ==================== Episode XYZ 位置轨迹曲线 ====================
def detect_stationary_intervals(frames_data, window_size, threshold_pct):
    """检测 7 个指标同时变化较小的连续帧区间（纯数据计算，无任何 UI 操作）。

    与 TrajectoryPlotWidget._run_analysis 同一算法（向量化实现，批量剪辑时
    直接调用，避免逐集加载视频/绘图/IK 造成的卡顿）：
    每个指标以整段 Episode 的最大绝对值作为变化基准，滑窗内所有指标的
    相对极差均小于阈值时，窗内帧判定为静止。

    Args:
        frames_data: dict，含 'x','y','z','rx','ry','rz','gripper' 七个序列
            （_load_episode_position_data 的返回结构）
        window_size: 判定静止的最少连续帧数
        threshold_pct: 相对变化阈值（百分数，如 0.05 表示 0.05%）

    Returns:
        [(start, end), ...] 闭区间列表，按起点升序；无静止区间返回 []
    """
    if frames_data is None:
        return []
    x = frames_data.get('x')
    if x is None or len(x) == 0:
        return []
    n = len(x)
    window = max(1, int(window_size))
    if n < window:
        return []

    keys = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
    arrays = []
    scales = []
    for key in keys:
        arr = np.asarray(frames_data.get(key, []), dtype=np.float64)
        if len(arr) < n:
            arr = np.zeros(n)  # 缺列按零处理（整段不变即静止，与原逻辑一致）
        scale = float(np.max(np.abs(arr))) if n > 0 else 0.0
        arrays.append(arr)
        scales.append(max(scale, 1e-9))

    # 向量化滑窗极差：(n-window+1, 7) —— 每行一个窗口、每列一个指标
    from numpy.lib.stride_tricks import sliding_window_view
    ranges = np.stack([
        np.ptp(sliding_window_view(a, window), axis=1) for a in arrays
    ], axis=1)
    threshold = float(threshold_pct) / 100.0
    all_stable = (ranges < (np.asarray(scales) * threshold)).all(axis=1)

    # 窗 s 静止 → 帧 [s, s+window-1] 全部标记（滑动叠加等价于原循环赋值）
    stationary = np.convolve(
        all_stable.astype(np.int32), np.ones(window, dtype=np.int32))[:n] > 0

    # 合并连续区间为 (start, end) 闭区间列表
    intervals = []
    start = None
    for i in range(n):
        if stationary[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                intervals.append((start, i - 1))
                start = None
    if start is not None:
        intervals.append((start, n - 1))
    return intervals


def _zigzag_reversal_frames(arr, dead):
    """带迟滞死区的方向反转帧号序列（ZigZag 指标，纯 numpy/python）。

    只有逆当前方向走过 ≥dead 的运动才计一次反转（过滤数值噪声的符号抖动）。
    返回反转发生的帧号数组（用于滑窗计数）。
    """
    a = np.asarray(arr, dtype=np.float64)
    n = len(a)
    if n < 3:
        return np.zeros(0, dtype=np.int64)
    events = []
    direction = 0
    last_extreme = a[0]
    for i in range(1, n):
        v = a[i]
        if direction >= 0 and (last_extreme - v) >= dead:
            if direction == 1:
                events.append(i)
            direction = -1
            last_extreme = v
        elif direction <= 0 and (v - last_extreme) >= dead:
            if direction == -1:
                events.append(i)
            direction = 1
            last_extreme = v
        elif direction == 1 and v > last_extreme:
            last_extreme = v
        elif direction == -1 and v < last_extreme:
            last_extreme = v
    return np.asarray(events, dtype=np.int64)


def detect_jitter_intervals(frames_data, window_size, pos_amp_mm, rot_amp_deg,
                            min_reversals=2):
    """滑窗检测机械臂往复抖动区间（方向反转计数 + 幅度阈值，纯数据计算）。

    判定：窗口内某轴（x/y/z 位置、rx/ry/rz 姿态）的带迟滞方向反转次数
    ≥ min_reversals 且窗口内该轴极差 ≥ 对应阈值（位置 pos_amp_mm 毫米 /
    姿态 rot_amp_deg 度）→ 该窗判抖动；任一轴命中即标记（帧级 OR）。
    迟滞死区 = 幅度阈值/2（每次反转隐含 ≥阈值/2 的真实行程，抗数值噪声）。

    min_reversals=2 = 完整一个来回（正-反-正型振荡，典型抖动形态）；
    =1 可捕捉单次大幅折返——谨慎使用，可能是正常任务语义（先升后降）。

    Args:
        frames_data: dict，含 'x','y','z','rx','ry','rz','gripper' 序列
        window_size: 滑窗帧数（40 帧 @20fps = 2s）
        pos_amp_mm: 位置轴抖动幅度阈值（毫米）
        rot_amp_deg: 姿态轴抖动幅度阈值（度）
        min_reversals: 窗口内最少方向反转次数

    Returns:
        [(start, end), ...] 闭区间列表，按起点升序；无抖动返回 []
    """
    if frames_data is None:
        return []
    x = frames_data.get('x')
    if x is None or len(x) == 0:
        return []
    n = len(x)
    window = max(3, int(window_size))
    if n < window:
        return []

    axis_specs = [
        ('x', float(pos_amp_mm) * 1e-3), ('y', float(pos_amp_mm) * 1e-3),
        ('z', float(pos_amp_mm) * 1e-3),
        ('rx', math.radians(float(rot_amp_deg))), ('ry', math.radians(float(rot_amp_deg))),
        ('rz', math.radians(float(rot_amp_deg))),
    ]

    from numpy.lib.stride_tricks import sliding_window_view
    jitter_any = np.zeros(max(0, n - window + 1), dtype=bool)
    for key, amp in axis_specs:
        arr = np.asarray(frames_data.get(key, []), dtype=np.float64)
        if len(arr) < n or amp <= 0:
            continue
        dead = amp / 2.0
        events = _zigzag_reversal_frames(arr, dead)
        if len(events) < min_reversals:
            continue
        # 滑窗内反转计数（前缀和）
        cnt = np.zeros(n + 1, dtype=np.int64)
        np.add.at(cnt, events + 1, 1)
        cnt = np.cumsum(cnt)  # cnt[i] = 前 i 帧内的反转数
        win_cnt = cnt[window:] - cnt[:-window] if n >= window else np.zeros(0, dtype=np.int64)
        # 滑窗极差
        win_rng = np.ptp(sliding_window_view(arr, window), axis=1)
        hit = (win_cnt >= min_reversals) & (win_rng >= amp)
        jitter_any |= hit[:len(jitter_any)]

    # 窗 s 抖动 → 帧 [s, s+window-1] 全部标记，合并连续区间
    jitter = np.convolve(
        jitter_any.astype(np.int32), np.ones(window, dtype=np.int32))[:n] > 0
    intervals = []
    start = None
    for i in range(n):
        if jitter[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                intervals.append((start, i - 1))
                start = None
    if start is not None:
        intervals.append((start, n - 1))
    return intervals

PRESET_LABELS = [
    "抓取第一个黑块",
    "放置第一个黑块",
    "抓取第二个黑块",
    "放置第二个黑块",
    "抓取小熊",
    "放置小熊",
    "机械臂归位",
]

class TrajectoryPlotWidget(QWidget):
    """显示单个 Episode 内末端位姿随帧序号变化的曲线。

    位置、欧拉角、夹爪分别绘制在三幅子图中，均显示原始值（不归一化）。
    右侧提供智能分析模块，可检测机械臂静止状态并在位置子图中用灰色条标注。

    支持：
    - 整个 Episode 以绝对帧序号显示
    - 选中片段时以相对帧序号（0 ~ 片段长度-1）显示
    - X 轴（帧范围）同步缩放三幅子图
    - Y 轴缩放仅作用于位置子图（欧拉角和夹爪保持自适应）
    - 当前帧竖线在三幅子图中同步，顶部子图带帧号标签
    - 使用 blit 局部重绘，保证播放流畅
    """

    # 信号：请求主界面剪辑检测到的静止区间，参数为 [(start, end), ...]
    trim_stationary_requested = pyqtSignal(list)
    # 信号：请求主界面剪辑检测到的抖动区间，参数为 [(start, end), ...]
    trim_jitter_requested = pyqtSignal(list)
    # 信号：请求主界面依次剪辑所有 Episode 的静止区间
    trim_all_stationary_requested = pyqtSignal()
    # 信号：请求主界面依次剪辑选中 Episode 的抖动区间（后期处理分页参数）
    trim_all_jitter_requested = pyqtSignal()
    # 信号：请求对当前 Episode 进行降噪（params dict）
    denoise_requested = pyqtSignal(dict)
    # 信号：请求对所有 Episode 批量降噪
    denoise_all_requested = pyqtSignal(dict)
    # 信号：请求预览当前 Episode 的降噪效果（不修改数据）
    preview_denoise_requested = pyqtSignal(dict)
    # 信号：请求把自动语言标注结果写回数据集（参数为 [{start,end,task,color,note}, ...]）
    subtask_apply_requested = pyqtSignal(list)
    # 信号：请求主窗口用 VLM 给当前分段生成标签（参数为 段数 N）
    vlm_label_requested = pyqtSignal(int)
    # 信号：数据集合并完成（参数为目标数据集名称），请求主窗口刷新数据集列表
    merge_completed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._current_frame = 0
        self._total_frames = 0

        # 片段模式：选中片段后，x 轴变为片段内相对帧序号
        self._segment_mode = False
        self._segment_start = 0
        self._segment_end = 0

        # 视图状态（均以 "zoom=1 时的基准值" 为参考）
        self._view_center = 0.0
        self._view_half_width = 0.0
        self._x_base_half_width = 0.0
        self._x_zoom_level = 1

        # 位置 Y 轴（受 Y 缩放滑块控制）
        self._y_center = 0.0
        self._y_half_range = 0.0
        self._y_base_half_range = 0.0
        self._y_zoom_level = 1

        # 欧拉角 Y 轴（固定自适应，不受 Y 缩放滑块控制）
        self._euler_y_center = 0.0
        self._euler_y_half_range = 0.0

        # 夹爪 Y 轴（固定自适应）
        self._grip_y_center = 0.0
        self._grip_y_half_range = 0.0

        # 夹爪力矩 Y 轴（右侧 twin axis，固定自适应）
        self._torque_y_center = 0.0
        self._torque_y_half_range = 0.0
        self._has_torque = False  # 当前 episode 是否含力矩数据

        # 智能分析结果
        self._stationary_intervals = []
        self._window_size = 10
        self._threshold_pct = 0.05
        self._stationary_bars = []

        # 抖动分析结果（后期处理分页）
        self._jitter_intervals = []
        self._jitter_bars = []

        # 自动语言标注（子任务分段）结果
        self._subtask_segments = []      # [{start,end,task,color,note}, ...]
        self._subtask_bars = []          # axvspan 句柄
        self._subtask_texts = []         # 文本标注句柄
        self._subtask_vlines = []        # 段边界竖线句柄

        # 降噪预览叠加数据（与 _data 同结构、同长度），None 表示不叠加
        self._smoothed = None

        # 数据集合并状态
        self._merge_root = None
        self._merge_worker = None
        self._merge_target_name = ""

        # 主布局：左侧绘图 + 右侧分析面板
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(5, 8), dpi=100, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(520)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        left_layout.addWidget(self.canvas, stretch=1)

        # 三幅子图共享 X 轴：位置 / 欧拉角 / 夹爪
        self.ax_pos = self.figure.add_subplot(311)
        self.ax_euler = self.figure.add_subplot(312, sharex=self.ax_pos)
        self.ax_grip = self.figure.add_subplot(313, sharex=self.ax_pos)
        self.axes = [self.ax_pos, self.ax_euler, self.ax_grip]

        # 夹爪力矩 twin axis（共享夹爪子图 X 轴，独立右侧 Y 轴）
        self.ax_torque = self.ax_grip.twinx()

        self.ax_pos.set_ylabel("位置 (m)")
        self.ax_euler.set_ylabel("欧拉角 (rad)")
        self.ax_grip.set_ylabel("夹爪 (rad)")
        self.ax_torque.set_ylabel("力矩 (Nm)")
        self.ax_grip.set_xlabel("帧序号")

        for ax in self.axes:
            ax.grid(True, linestyle='--', alpha=0.5)

        self.lines = {}
        self.vlines = []  # 三幅子图各一条竖线
        self.cursor_label = None  # 播放标签显示在顶部子图

        # 每幅子图的 blit 背景缓存
        self._bg_pos = None
        self._bg_euler = None
        self._bg_grip = None

        # 缩放控制
        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel("X缩放:"))
        self.x_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.x_zoom_slider.setRange(1, 100)
        self.x_zoom_slider.setValue(1)
        self.x_zoom_slider.setToolTip("水平放大：1=当前基准，100=最大放大（同步三幅子图）")
        self.x_zoom_slider.valueChanged.connect(self._on_x_zoom_changed)
        controls.addWidget(self.x_zoom_slider, stretch=2)

        controls.addWidget(QLabel("Y缩放:"))
        self.y_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.y_zoom_slider.setRange(1, 100)
        self.y_zoom_slider.setValue(1)
        self.y_zoom_slider.setToolTip("垂直放大：仅作用于位置曲线")
        self.y_zoom_slider.valueChanged.connect(self._on_y_zoom_changed)
        controls.addWidget(self.y_zoom_slider, stretch=2)

        self.reset_zoom_btn = QPushButton("重置")
        self.reset_zoom_btn.setToolTip("重置为显示整个 Episode")
        self.reset_zoom_btn.clicked.connect(self._reset_zoom)
        controls.addWidget(self.reset_zoom_btn)

        left_layout.addLayout(controls)

        # 右侧智能分析面板
        analysis_group = QGroupBox("智能分析")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setSpacing(8)

        form_layout = QFormLayout()
        form_layout.setSpacing(6)

        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 1000)
        self.window_spin.setValue(self._window_size)
        self.window_spin.setToolTip("判定静止状态的最少连续帧数")
        self.window_spin.valueChanged.connect(self._on_window_size_changed)
        form_layout.addRow("静止窗口 (帧):", self.window_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 2.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(self._threshold_pct)
        self.threshold_spin.setSuffix(" %")
        self.threshold_spin.setToolTip("窗口内 7 个指标相对变化均小于此阈值时判定为静止（可直接输入）")
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        form_layout.addRow("变化阈值:", self.threshold_spin)

        analysis_layout.addLayout(form_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self.analyze_btn = QPushButton("🔍 智能分析")
        self.analyze_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 6px; }"
        )
        self.analyze_btn.clicked.connect(self._run_analysis)
        btn_layout.addWidget(self.analyze_btn)

        self.trim_stationary_btn = QPushButton("✂ 剪辑静止")
        self.trim_stationary_btn.setStyleSheet(
            "QPushButton { background-color: #FF5722; color: white; font-weight: bold; padding: 6px; }"
        )
        self.trim_stationary_btn.setToolTip("删除当前检测到的所有静止帧")
        self.trim_stationary_btn.clicked.connect(self._on_trim_stationary_clicked)
        btn_layout.addWidget(self.trim_stationary_btn)

        self.trim_all_stationary_btn = QPushButton("✂ 剪辑所有静止")
        self.trim_all_stationary_btn.setStyleSheet(
            "QPushButton { background-color: #D84315; color: white; font-weight: bold; padding: 6px; }"
        )
        self.trim_all_stationary_btn.setToolTip("依次分析并删除所有 Episode 的静止帧（逐个处理，避免文件损坏）")
        self.trim_all_stationary_btn.clicked.connect(self._on_trim_all_stationary_clicked)
        btn_layout.addWidget(self.trim_all_stationary_btn)

        analysis_layout.addLayout(btn_layout)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("点击“智能分析”查看结果")
        self.result_text.setMaximumWidth(260)
        self.result_text.setMinimumHeight(120)
        analysis_layout.addWidget(self.result_text, stretch=1)

        # ---------- 自动语言标注（子任务分段） ----------
        subtask_sep = QFrame()
        subtask_sep.setFrameShape(QFrame.Shape.HLine)
        subtask_sep.setFrameShadow(QFrame.Shadow.Sunken)
        analysis_layout.addWidget(subtask_sep)

        analysis_layout.addWidget(QLabel("🗣 自动语言标注（智能分段 + VLM 标签）"))

        # 段数 + 含力矩
        seg_row = QHBoxLayout()
        seg_row.setSpacing(4)
        seg_row.addWidget(QLabel("段数:"))
        self.subtask_count_spin = QSpinBox()
        self.subtask_count_spin.setRange(1, 30)
        self.subtask_count_spin.setValue(7)
        self.subtask_count_spin.setToolTip("指定要切分的段数（如 3/5/8），将强制切出恰好这么多段")
        seg_row.addWidget(self.subtask_count_spin)
        self.subtask_torque_chk = QCheckBox("用力矩")
        self.subtask_torque_chk.setChecked(True)
        self.subtask_torque_chk.setToolTip("勾选后边界评分加入夹爪力矩（接触/释放物体事件），切分更准")
        seg_row.addWidget(self.subtask_torque_chk)
        self.subtask_angle_chk = QCheckBox("用夹角")
        self.subtask_angle_chk.setChecked(False)
        self.subtask_angle_chk.setToolTip("勾选后边界评分加入夹爪角度变化（开/合事件）。默认关：仅靠位姿停顿+力矩分段")
        seg_row.addWidget(self.subtask_angle_chk)
        analysis_layout.addLayout(seg_row)

        # 智能分段按钮：按力矩+位姿切出 N 段（占位标签 步骤1/2/..）
        self.subtask_segment_btn = QPushButton("🤖 智能分段")
        self.subtask_segment_btn.setStyleSheet(
            "QPushButton { background-color: #7B1FA2; color: white; font-weight: bold; padding: 5px; }"
        )
        self.subtask_segment_btn.setToolTip("根据夹爪力矩+位姿变化切出指定段数的真实边界，并在曲线上着色预览")
        self.subtask_segment_btn.clicked.connect(self._run_smart_segment)
        analysis_layout.addWidget(self.subtask_segment_btn)

        self.subtask_preset_btn = QPushButton("📋 套用预设名称")
        self.subtask_preset_btn.setStyleSheet(
            "QPushButton { background-color: #455A64; color: white; font-weight: bold; padding: 5px; }"
        )
        self.subtask_preset_btn.setToolTip(
            "按当前段数套用预设名称（抓取/放置黑块·小熊·机械臂归位）。段数超出预设时多余段保持 \"步骤N\"")
        self.subtask_preset_btn.clicked.connect(self._apply_preset_labels)
        analysis_layout.addWidget(self.subtask_preset_btn)

        # 可编辑标签表：步 | 标签
        self.subtask_label_table = QTableWidget(0, 2)
        self.subtask_label_table.setHorizontalHeaderLabels(["步", "标签（可编辑）"])
        self.subtask_label_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.subtask_label_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.subtask_label_table.setMaximumHeight(120)
        self.subtask_label_table.setMaximumWidth(260)
        self.subtask_label_table.setToolTip("双击“标签”列可手动修改名称；修改后会同步更新曲线着色")
        self.subtask_label_table.itemChanged.connect(self._on_label_table_edited)
        analysis_layout.addWidget(self.subtask_label_table)

        # 摄像头路数选择（生成标签 VLM 推理用）
        vlm_cam_row = QHBoxLayout()
        vlm_cam_row.addWidget(QLabel("摄像头:"))
        self.subtask_cam_combo = QComboBox()
        self.subtask_cam_combo.addItems(["摄像头1 (cam1)", "摄像头2 (cam2)", "双摄 (cam1+cam2)"])
        self.subtask_cam_combo.setCurrentIndex(0)
        self.subtask_cam_combo.setToolTip(
            "选择传入 Qwen 模型的摄像头画面（用于“生成标签(VLM)”）。\n"
            " · 摄像头1 / 摄像头2：只用对应单路画面\n"
            " · 双摄：同一消息里同时喂两路画面")
        vlm_cam_row.addWidget(self.subtask_cam_combo, 1)
        analysis_layout.addLayout(vlm_cam_row)

        # 生成标签(VLM) + 清除
        sub_btn_row = QHBoxLayout()
        sub_btn_row.setSpacing(4)
        self.subtask_vlm_btn = QPushButton("🏷 生成标签(VLM)")
        self.subtask_vlm_btn.setStyleSheet(
            "QPushButton { background-color: #1565C0; color: white; font-weight: bold; padding: 5px; }"
        )
        self.subtask_vlm_btn.setToolTip("调用 Qwen3-VL 分析每段关键帧，自动生成“动作+物体”标签（后台运行）")
        self.subtask_vlm_btn.setEnabled(False)
        self.subtask_vlm_btn.clicked.connect(self._on_vlm_label_clicked)
        sub_btn_row.addWidget(self.subtask_vlm_btn)

        self.subtask_clear_btn = QPushButton("🗑 清除")
        self.subtask_clear_btn.setToolTip("清除曲线上的子任务标注")
        self.subtask_clear_btn.clicked.connect(self._clear_subtask_annotate)
        sub_btn_row.addWidget(self.subtask_clear_btn)
        analysis_layout.addLayout(sub_btn_row)

        self.subtask_apply_btn = QPushButton("💾 应用到数据集")
        self.subtask_apply_btn.setStyleSheet(
            "QPushButton { background-color: #00695C; color: white; font-weight: bold; padding: 5px; }"
        )
        self.subtask_apply_btn.setToolTip("把当前标签写回 LeRobot 数据集（逐帧 language_persistent 列）并记入标签库")
        self.subtask_apply_btn.setEnabled(False)
        self.subtask_apply_btn.clicked.connect(self._on_apply_subtask_clicked)
        analysis_layout.addWidget(self.subtask_apply_btn)

        self.subtask_result_label = QLabel("（设段数 → 智能分段 → 生成标签 → 应用）")
        self.subtask_result_label.setWordWrap(True)
        self.subtask_result_label.setMaximumWidth(260)
        self.subtask_result_label.setStyleSheet("QLabel { color: #555; font-size: 11px; }")
        analysis_layout.addWidget(self.subtask_result_label)


        # 右侧容器：分页（智能分析 + RECAP）
        right_container = QTabWidget()
        # Tab 1: 智能分析（原 analysis_group + 智能降噪）
        analysis_tab = QWidget()
        analysis_tab_lay = QVBoxLayout(analysis_tab)
        analysis_tab_lay.setContentsMargins(2, 2, 2, 2)
        analysis_tab_lay.setSpacing(8)
        analysis_tab_lay.addWidget(analysis_group, stretch=2)
        analysis_tab_lay.addWidget(self._build_denoise_group(), stretch=1)
        right_container.addTab(analysis_tab, "智能分析")
        # Tab 1.5: 后期处理（抖动检测 + 剪辑晃动）
        right_container.addTab(self._build_postprocess_tab(), "后期处理")
        # Tab 2: RECAP 奖励标注
        right_container.addTab(self._build_recap_tab(), "RECAP")
        # Tab 3: 数据集合并
        right_container.addTab(self._build_merge_tab(), "数据集合并")

        main_layout.addWidget(left_widget, stretch=3)
        main_layout.addWidget(right_container, stretch=1)

        self._empty_plot()

    # ==================== 后期处理分页（抖动检测 + 剪辑晃动） ====================
    def _build_postprocess_tab(self):
        """构建“后期处理”分页：检测推理采集数据中的机械臂往复抖动并剪辑。

        仅构建控件；分析/剪辑逻辑在本类与 MainWindow 中（信号连接由
        MainWindow 统一完成）。
        """
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(6)

        hint = QLabel(
            "检测推理录制数据中的机械臂往复抖动（策略输出振荡/边界跳变），\n"
            "紫色半透明标记在曲线上，确认后可剪辑（自动重算 action 标签）"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        lay.addWidget(hint)

        form = QFormLayout()
        self.jitter_window_spin = QSpinBox()
        self.jitter_window_spin.setRange(10, 300)
        self.jitter_window_spin.setValue(15)
        self.jitter_window_spin.setToolTip(
            "滑窗帧数（15 帧 @20fps ≈ 0.75s）：窗口内统计方向反转次数与幅度")
        form.addRow("窗口 (帧):", self.jitter_window_spin)

        self.jitter_amp_spin = QDoubleSpinBox()
        self.jitter_amp_spin.setRange(0.5, 100.0)
        self.jitter_amp_spin.setDecimals(1)
        self.jitter_amp_spin.setSingleStep(0.5)
        self.jitter_amp_spin.setValue(7.0)
        self.jitter_amp_spin.setSuffix(" mm")
        self.jitter_amp_spin.setToolTip(
            "位置轴（x/y/z）抖动幅度阈值：窗口内极差 ≥ 此值才判抖动\n"
            "（推理抖动典型 3~20mm，按实测调整）")
        form.addRow("位置幅度:", self.jitter_amp_spin)

        self.jitter_rot_spin = QDoubleSpinBox()
        self.jitter_rot_spin.setRange(0.1, 30.0)
        self.jitter_rot_spin.setDecimals(1)
        self.jitter_rot_spin.setSingleStep(0.5)
        self.jitter_rot_spin.setValue(5.0)
        self.jitter_rot_spin.setSuffix(" °")
        self.jitter_rot_spin.setToolTip("姿态轴（rx/ry/rz）抖动幅度阈值（度）")
        form.addRow("姿态幅度:", self.jitter_rot_spin)

        self.jitter_rev_spin = QSpinBox()
        self.jitter_rev_spin.setRange(1, 10)
        self.jitter_rev_spin.setValue(2)
        self.jitter_rev_spin.setToolTip(
            "窗口内最少方向反转次数（带迟滞死区=幅度/2，抗数值噪声）：\n"
            "· 2（默认）= 完整一个来回（正-反-正振荡，典型抖动）\n"
            "· 1 = 单次大幅折返也算——谨慎！可能是正常任务语义（先升后降）")
        form.addRow("反转次数:", self.jitter_rev_spin)
        lay.addLayout(form)

        self.jitter_analyze_btn = QPushButton("🔍 分析抖动")
        self.jitter_analyze_btn.setStyleSheet(
            "QPushButton { background-color: #7B1FA2; color: white; font-weight: bold; padding: 6px; }")
        self.jitter_analyze_btn.clicked.connect(self._run_jitter_analysis)
        lay.addWidget(self.jitter_analyze_btn)

        self.jitter_result_label = QLabel("（设参数 → 分析抖动 → 核对紫色区域 → 剪辑晃动）")
        self.jitter_result_label.setWordWrap(True)
        self.jitter_result_label.setStyleSheet("QLabel { color: #555; font-size: 11px; }")
        lay.addWidget(self.jitter_result_label)

        self.jitter_trim_btn = QPushButton("✂ 剪辑晃动")
        self.jitter_trim_btn.setStyleSheet(
            "QPushButton { background-color: #D84315; color: white; font-weight: bold; padding: 6px; }")
        self.jitter_trim_btn.setToolTip(
            "删除检测到的全部抖动区间（与剪辑静止同一安全管线）：\n"
            "· RECAP 标注区间自动随帧重映射\n"
            "· 剪辑后自动重算 action 标签（action[t]=state[t+前瞻]，运动段剪辑必需）\n"
            "· 建议先“剪辑前自动备份”勾选")
        self.jitter_trim_btn.clicked.connect(self._on_trim_jitter_clicked)
        lay.addWidget(self.jitter_trim_btn)

        self.trim_all_jitter_btn = QPushButton("✂✂ 剪辑所有晃动")
        self.trim_all_jitter_btn.setStyleSheet(
            "QPushButton { background-color: #8D6E63; color: white; font-weight: bold; padding: 6px; }")
        self.trim_all_jitter_btn.setToolTip(
            "对选中的 Episode 依次分析并剪辑抖动区间（与本页参数一致）：\n"
            "· 弹窗选择 Episode（同\"剪辑所有静止\"）\n"
            "· 无抖动的 Episode 直接跳过（纯数据计算，毫秒级）\n"
            "· 全部剪辑完成后统一重算一次 action 标签，\n"
            "  并自动重新生成训练 sidecar（若已生成过）")
        self.trim_all_jitter_btn.clicked.connect(self._on_trim_all_jitter_clicked)
        lay.addWidget(self.trim_all_jitter_btn)

        lay.addStretch()
        return tab

    def _run_jitter_analysis(self):
        """抖动分析：检测往复区间，紫色半透明标记到三幅子图。"""
        if self._data is None or self._total_frames == 0:
            self.jitter_result_label.setText("请先选择 Episode。")
            return
        self._jitter_intervals = detect_jitter_intervals(
            self._data, self.jitter_window_spin.value(),
            self.jitter_amp_spin.value(), self.jitter_rot_spin.value(),
            self.jitter_rev_spin.value())
        n = len(self._jitter_intervals)
        total = sum(e - s + 1 for s, e in self._jitter_intervals)
        pct = total / max(self._total_frames, 1) * 100
        if n == 0:
            self.jitter_result_label.setText(
                f"未检测到抖动（窗口{self.jitter_window_spin.value()}帧 / "
                f"幅度{self.jitter_amp_spin.value()}mm / "
                f"反转≥{self.jitter_rev_spin.value()}）。可放宽参数重试")
        else:
            detail = "；".join(f"{s}~{e}" for s, e in self._jitter_intervals[:6])
            if n > 6:
                detail += f" … 共{n}段"
            self.jitter_result_label.setText(
                f"检测到 {n} 段抖动，共 {total}/{self._total_frames} 帧 "
                f"({pct:.1f}%)\n{detail}")
        # 重绘显示紫色条
        self._plot_data(
            self._segment_start if self._segment_mode else 0,
            self._segment_end if self._segment_mode else self._total_frames - 1,
            self._get_current_rel_frame(),
            x_label="片段内帧序号" if self._segment_mode else "帧序号")
        self._apply_view()

    def _on_trim_jitter_clicked(self):
        """请求主窗口剪辑检测到的抖动区间（确认对话框在主窗口）。"""
        if not self._jitter_intervals:
            self.jitter_result_label.setText("请先点击“分析抖动”")
            return
        self.trim_jitter_requested.emit(list(self._jitter_intervals))

    def _draw_jitter_bars(self):
        """根据抖动分析结果，在三幅子图中同步绘制紫色半透明条。"""
        for bar in self._jitter_bars:
            try:
                bar.remove()
            except Exception:
                pass
        self._jitter_bars = []

        if not self._jitter_intervals or self._data is None:
            return

        if self._segment_mode:
            seg_s, seg_e = self._segment_start, self._segment_end
        else:
            seg_s, seg_e = 0, self._total_frames - 1

        first = True
        for s_ep, e_ep in self._jitter_intervals:
            vs = max(s_ep, seg_s)
            ve = min(e_ep, seg_e)
            if vs > ve:
                continue
            if self._segment_mode:
                x0 = vs - seg_s - 0.5
                x1 = ve - seg_s + 0.5
            else:
                x0 = vs - 0.5
                x1 = ve + 0.5
            for ax in self.axes:
                bar = ax.axvspan(
                    x0, x1, color='#9C27B0', alpha=0.22, zorder=0,
                    label='抖动区间' if (first and ax is self.ax_pos) else '')
                self._jitter_bars.append(bar)
            first = False

    # ==================== RECAP 奖励标注面板 ====================
    def _build_recap_tab(self):
        """构建 RECAP 奖励标注分页（参照 label_rewards_qt.py 的标注方式）。

        仅构建控件；信号连接由 MainWindow._connect_recap_signals() 统一完成。
        """
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(6)

        hint = QLabel(
            "RECAP 片段级奖励标注：标记 good/bad/neutral/干预 区间 + episode 级成败 →\n"
            "make_recap_sidecar.py 展开为逐帧 sidecar（不改 parquet）→ 训练\n"
            "快捷键：播放中按 G=好 / B=坏 / N=中（以最近 2 秒为区间）\n"
            "修改：双击下方区间表行或轨道上的色条进入编辑，改完点\"保存修改\""
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        lay.addWidget(hint)

        # ===== episode 级信息（成败按标注粒度决策由片段标注派生，只读不人工标注）=====
        lay.addWidget(QLabel("本 episode:"))
        self.recap_ep_derived_label = QLabel("成败（派生）: —")
        self.recap_ep_derived_label.setStyleSheet("font-weight:bold;")
        self.recap_ep_derived_label.setToolTip(
            "由片段标注自动派生（人工只需标片段）：\n"
            "good 帧占标注帧 >50% 且不存在未接恢复段的 bad 段 → 成功；\n"
            "干预段（人工接管纠正）即恢复行为。")
        lay.addWidget(self.recap_ep_derived_label)
        self.recap_reviewed_check = QCheckBox("已审核（确认标注完整，区分\"未标注\"）")
        lay.addWidget(self.recap_reviewed_check)

        # 当前标签 + 起止帧
        form = QFormLayout()
        self.recap_label_combo = QComboBox()
        self.recap_label_combo.addItems(RECAP_ALL_LABELS)
        self.recap_label_combo.setCurrentText("good")
        form.addRow("当前标签:", self.recap_label_combo)

        self.recap_start_spin = QSpinBox()
        self.recap_start_spin.setMaximum(999999)
        self.recap_end_spin = QSpinBox()
        self.recap_end_spin.setMaximum(999999)
        form.addRow("区间起点:", self.recap_start_spin)
        form.addRow("区间终点:", self.recap_end_spin)
        lay.addLayout(form)

        # 标记按钮
        mark_row = QHBoxLayout()
        self.recap_mark_start_btn = QPushButton("[ 设起点")
        self.recap_mark_end_btn = QPushButton("设终点 ]")
        mark_row.addWidget(self.recap_mark_start_btn)
        mark_row.addWidget(self.recap_mark_end_btn)
        lay.addLayout(mark_row)

        # 干预段勾选（人工接管纠正：RECAP/APO 约定恒为正优势，轨道显示蓝色）
        self.recap_intervention_check = QCheckBox(
            "干预段（人工接管纠正，恒记正优势）")
        self.recap_intervention_check.setToolTip(
            "勾选后新添加的区间标记为人工接管纠正段：\n"
            "- 展开时 advantage 恒为正（与 RECAP/APO 约定一致）\n"
            "- 轨道上以蓝色显示，与普通 good 段区分")
        lay.addWidget(self.recap_intervention_check)

        # 添加区间（编辑模式下变为"保存修改"）+ 取消编辑（编辑模式下显示）
        add_row = QHBoxLayout()
        self.recap_add_btn = QPushButton("✚ 添加区间")
        self.recap_add_btn.setStyleSheet("font-weight:bold; padding:5px;")
        add_row.addWidget(self.recap_add_btn)
        self.recap_cancel_edit_btn = QPushButton("取消编辑")
        self.recap_cancel_edit_btn.setVisible(False)
        add_row.addWidget(self.recap_cancel_edit_btn)
        lay.addLayout(add_row)

        # 备注
        self.recap_note_edit = QLineEdit()
        self.recap_note_edit.setPlaceholderText("备注 (可选)")
        lay.addWidget(QLabel("备注:"))
        lay.addWidget(self.recap_note_edit)

        # 快捷操作
        lay.addWidget(QLabel("快捷操作:"))
        quick_row = QHBoxLayout()
        self.recap_quick_good = QPushButton("✓ 好")
        self.recap_quick_good.setStyleSheet(
            f"background:{RECAP_LABEL_COLORS['good']}; color:#fff; font-weight:bold;")
        self.recap_quick_bad = QPushButton("✗ 坏")
        self.recap_quick_bad.setStyleSheet(
            f"background:{RECAP_LABEL_COLORS['bad']}; color:#fff; font-weight:bold;")
        self.recap_quick_neutral = QPushButton("〜 中")
        self.recap_quick_neutral.setStyleSheet(
            f"background:{RECAP_LABEL_COLORS['neutral']}; color:#fff;")
        quick_row.addWidget(self.recap_quick_good)
        quick_row.addWidget(self.recap_quick_bad)
        quick_row.addWidget(self.recap_quick_neutral)
        lay.addLayout(quick_row)

        # 整 episode 快标
        lay.addWidget(QLabel("整 episode 快标:"))
        ep_row = QHBoxLayout()
        self.recap_ep_good = QPushButton("全部=好")
        self.recap_ep_bad = QPushButton("全部=坏")
        ep_row.addWidget(self.recap_ep_good)
        ep_row.addWidget(self.recap_ep_bad)
        lay.addLayout(ep_row)

        # 区间列表
        lay.addWidget(QLabel("已标注区间:"))
        self.recap_interval_table = QTableWidget(0, 6)
        self.recap_interval_table.setHorizontalHeaderLabels(
            ["#", "起点", "终点", "标签", "干预", "备注"])
        self.recap_interval_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.recap_interval_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.recap_interval_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.recap_interval_table.setMaximumWidth(300)
        self.recap_interval_table.setToolTip("双击某行进入编辑：修改起止帧/标签/干预后点\"保存修改\"")
        lay.addWidget(self.recap_interval_table)

        # 删除/撤销/清空
        del_row = QHBoxLayout()
        self.recap_del_btn = QPushButton("✖ 删除选中")
        self.recap_undo_btn = QPushButton("↶ 撤销最后")
        self.recap_clear_btn = QPushButton("清空本 episode")
        del_row.addWidget(self.recap_del_btn)
        del_row.addWidget(self.recap_undo_btn)
        del_row.addWidget(self.recap_clear_btn)
        lay.addLayout(del_row)

        # 保存
        self.recap_save_btn = QPushButton("💾 保存标注")
        self.recap_save_btn.setStyleSheet("padding:7px; font-weight:bold;")
        lay.addWidget(self.recap_save_btn)

        # ── 训练导出：把区间级标注稿展开为训练程序消费的逐帧 sidecar ──
        export_group = QGroupBox("训练导出（逐帧 sidecar）")
        export_lay = QVBoxLayout(export_group)
        export_lay.setSpacing(4)
        export_hint = QLabel(
            "“保存标注”存的是区间稿（人读）；训练程序需要逐帧标签，\n"
            "点下方按钮展开为 meta/recap_sidecar.jsonl（不改 parquet）")
        export_hint.setWordWrap(True)
        export_hint.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        export_lay.addWidget(export_hint)
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("回传N帧:"))
        self.recap_bp_spin = QSpinBox()
        self.recap_bp_spin.setRange(0, 200)
        self.recap_bp_spin.setValue(0)
        self.recap_bp_spin.setToolTip(
            "RECAP 负优势回传帧数（0=关闭）：\n"
            "bad 段向前回传 N 帧、线性衰减（离失败点越近负得越强），\n"
            "对应 π*0.6 n-step 优势前瞻（论文 N=50）的无 critic 近似\n"
            "只填充未被人工标注覆盖的帧")
        opt_row.addWidget(self.recap_bp_spin)
        opt_row.addWidget(QLabel("失败末K帧:"))
        self.recap_tf_spin = QSpinBox()
        self.recap_tf_spin.setRange(0, 100)
        self.recap_tf_spin.setValue(0)
        self.recap_tf_spin.setToolTip(
            "终端失败惩罚帧数（0=关闭）：\n"
            "派生为失败的 episode 末 K 帧自动标 bad，\n"
            "对应 RECAP 稀疏奖励 r_T=-C_fail / APO 接管前 10 动作规则（建议 10）\n"
            "只填充未被人工标注覆盖的帧")
        opt_row.addWidget(self.recap_tf_spin)
        export_lay.addLayout(opt_row)
        self.recap_export_btn = QPushButton("🚀 生成训练 sidecar")
        self.recap_export_btn.setStyleSheet(
            "QPushButton { background-color: #00695C; color: white; font-weight: bold; padding: 7px; }")
        self.recap_export_btn.setToolTip(
            "先保存当前标注，再后台展开为逐帧 sidecar（make_recap_sidecar.expand_dataset）。\n"
            "注意：剪辑/合并/删除 episode 后索引重排，必须重新展开")
        export_lay.addWidget(self.recap_export_btn)
        lay.addWidget(export_group)

        # 状态提示
        self.recap_status_label = QLabel("（选择数据集后自动加载 reward_annotations.json）")
        self.recap_status_label.setWordWrap(True)
        self.recap_status_label.setStyleSheet("QLabel { color: #555; font-size: 11px; }")
        lay.addWidget(self.recap_status_label)

        # ── 价值优势显示（⑤⑥ 产物可视化）──
        self.recap_adv_btn = QPushButton("📈 显示优势")
        self.recap_adv_btn.setStyleSheet(
            "QPushButton { background-color: #5E35B1; color: white; font-weight: bold; padding: 7px; }")
        self.recap_adv_btn.setToolTip(
            "显示当前 episode 的价值/优势变化（读 meta/recap_sidecar.jsonl）：\n"
            "- 上图：V(s) 预期剩余代价（⑤ td_value，→0 接近成功完成）\n"
            "- 下图：TD 优势（⑤ td_advantage）/ 训练用合并优势（⑥ pg_advantage）\n"
            "- 背景色带为人标区间（good 绿 / 干预蓝 / bad 红）\n"
            "sidecar 不存在时会提示先生成（①）")
        lay.addWidget(self.recap_adv_btn)

        lay.addStretch()
        return tab

    # ==================== 数据集合并面板 ====================
    def _build_merge_tab(self):
        """构建数据集合并分页（复用 merge_lerobot_datasets.MergeWorker）。

        勾选源数据集（按列表顺序合并）→ 指定目标名称 → 后台线程合并，
        完成后通过 merge_completed 信号通知主窗口刷新数据集列表。
        """
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(6)

        hint = QLabel(
            "将多个 LeRobot v3.0 数据集合并为一个：重建 data/meta 指针、"
            "规整 frame_index/timestamp、重算 stats，避免 数据问题分析.md 中的 4 类元数据问题"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        lay.addWidget(hint)

        head_row = QHBoxLayout()
        head_row.addWidget(QLabel("源数据集 (按顺序合并):"))
        self.merge_refresh_btn = QPushButton("🔄")
        self.merge_refresh_btn.setToolTip("重新扫描数据根目录")
        self.merge_refresh_btn.setMaximumWidth(36)
        self.merge_refresh_btn.clicked.connect(self._merge_refresh_list)
        head_row.addWidget(self.merge_refresh_btn)
        lay.addLayout(head_row)

        self.merge_list = QListWidget()
        self.merge_list.setMaximumHeight(180)
        lay.addWidget(self.merge_list)

        op_row = QHBoxLayout()
        op_row.setSpacing(4)
        self.merge_all_btn = QPushButton("全选")
        self.merge_all_btn.clicked.connect(lambda: self._merge_set_all(True))
        self.merge_none_btn = QPushButton("全不选")
        self.merge_none_btn.clicked.connect(lambda: self._merge_set_all(False))
        self.merge_up_btn = QPushButton("上移")
        self.merge_up_btn.clicked.connect(lambda: self._merge_move_item(-1))
        self.merge_down_btn = QPushButton("下移")
        self.merge_down_btn.clicked.connect(lambda: self._merge_move_item(1))
        for b in (self.merge_all_btn, self.merge_none_btn, self.merge_up_btn, self.merge_down_btn):
            op_row.addWidget(b)
        lay.addLayout(op_row)

        lay.addWidget(QLabel("目标数据集 (可填已有名称覆盖, 或新名称):"))
        self.merge_target_combo = QComboBox()
        self.merge_target_combo.setEditable(True)
        lay.addWidget(self.merge_target_combo)

        self.merge_start_btn = QPushButton("🔀 开始合并")
        self.merge_start_btn.setStyleSheet(
            "QPushButton { background-color: #6A1B9A; color: white; font-weight: bold; padding: 7px; }"
        )
        self.merge_start_btn.clicked.connect(self._on_merge_start)
        lay.addWidget(self.merge_start_btn)

        self.merge_progress = QProgressBar()
        self.merge_progress.setMaximumHeight(16)
        lay.addWidget(self.merge_progress)

        self.merge_log = QPlainTextEdit()
        self.merge_log.setReadOnly(True)
        self.merge_log.setMaximumHeight(140)
        self.merge_log.setPlaceholderText("合并日志")
        lay.addWidget(self.merge_log)

        lay.addStretch()
        return tab

    def set_merge_root(self, root):
        """由主窗口设置数据根目录并刷新源列表。"""
        self._merge_root = root
        self._merge_refresh_list()

    def _merge_find_datasets(self):
        if not self._merge_root or not os.path.isdir(self._merge_root):
            return []
        out = []
        for name in sorted(os.listdir(self._merge_root)):
            path = os.path.join(self._merge_root, name)
            if not os.path.isdir(path) or name.startswith("."):
                continue
            info_path = os.path.join(path, "meta", "info.json")
            if not os.path.exists(info_path):
                continue
            try:
                with open(info_path, encoding="utf-8") as f:
                    info = json.load(f)
                out.append((name, info.get("total_episodes", "?"), info.get("total_frames", "?")))
            except Exception:
                out.append((name, "?", "?"))
        return out

    def _merge_refresh_list(self):
        checked = {
            self.merge_list.item(i).text().split("  (")[0]
            for i in range(self.merge_list.count())
            if self.merge_list.item(i).checkState() == Qt.CheckState.Checked
        }
        self.merge_list.clear()
        self.merge_target_combo.clear()
        for name, eps, frames in self._merge_find_datasets():
            item = QListWidgetItem(f"{name}  ({eps} eps / {frames} 帧)")
            item.setCheckState(
                Qt.CheckState.Checked if name in checked else Qt.CheckState.Unchecked
            )
            self.merge_list.addItem(item)
            self.merge_target_combo.addItem(name)

    def _merge_set_all(self, checked):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.merge_list.count()):
            self.merge_list.item(i).setCheckState(state)

    def _merge_move_item(self, delta):
        row = self.merge_list.currentRow()
        if row < 0:
            return
        new_row = row + delta
        if not (0 <= new_row < self.merge_list.count()):
            return
        item = self.merge_list.takeItem(row)
        self.merge_list.insertItem(new_row, item)
        self.merge_list.setCurrentRow(new_row)

    def _on_merge_start(self):
        if not self._merge_root:
            QMessageBox.warning(self, "提示", "数据根目录未设置")
            return
        sources = []
        for i in range(self.merge_list.count()):
            it = self.merge_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                sources.append(it.text().split("  (")[0])
        target = self.merge_target_combo.currentText().strip()
        if not sources:
            QMessageBox.warning(self, "提示", "请至少勾选一个源数据集")
            return
        if not target:
            QMessageBox.warning(self, "提示", "请填写目标数据集名称")
            return
        existing = {name for name, _, _ in self._merge_find_datasets()}
        # P0.3：合并会重排目标数据集的 episode_index —— 若目标已有 RECAP 标注，
        # 先警告；确认后备份并移除（源数据集不被修改，其标注保持有效）
        if target in existing:
            _target_path = os.path.join(self._merge_root, target)
            try:
                _target_annotated = recap_has_annotations(_target_path)
            except Exception:
                _target_annotated = False
            if _target_annotated:
                ret = QMessageBox.question(
                    self, "目标数据集已有 RECAP 标注",
                    f"目标数据集 {target} 已存在 RECAP 标注（meta/reward_annotations.json）。\n"
                    "合并将重建该数据集：目标原有标注会自动备份为\n"
                    "reward_annotations.stale.<时间戳>.bak 并移除（源数据集的标注\n"
                    "会按合并后的 episode_index 自动重映射保留）。\n\n"
                    "若想保留目标标注，请换一个新目标名。是否继续？",
                )
                if ret != QMessageBox.StandardButton.Yes:
                    self.merge_log.appendPlainText("用户取消合并（目标数据集含 RECAP 标注）")
                    return
                try:
                    _bak = recap_backup_annotations(_target_path, remove_original=True)
                    if _bak is not None:
                        self.merge_log.appendPlainText(f"已备份并移除目标标注: {_bak}")
                except Exception as _e:
                    self.merge_log.appendPlainText(f"⚠️ 标注备份失败: {_e}")
        try:
            global_issues, drop_plan = preflight_check(self._merge_root, sources)
        except Exception as e:
            QMessageBox.critical(self, "检查失败", f"读取源数据集信息失败:\n{e}")
            return
        drop_features = []
        allow_overrides = False
        if global_issues or drop_plan:
            parts = []
            if global_issues:
                parts.append("全局元数据不一致 (继续将采用首个数据集的值):\n"
                             + "\n".join(f"  · {i}" for i in global_issues))
            if drop_plan:
                parts.append("以下特征不兼容 (继续将舍弃这些特征):\n"
                             + "\n".join(f"  · {f} — {r}" for f, r in drop_plan))
            ret = QMessageBox.question(
                self,
                "格式不兼容",
                "检测到格式不兼容:\n\n" + "\n\n".join(parts) + "\n\n是否继续合并?",
            )
            if ret != QMessageBox.StandardButton.Yes:
                self.merge_log.appendPlainText("用户取消合并")
                return
            drop_features = [f for f, _ in drop_plan]
            allow_overrides = bool(global_issues)
            if drop_features:
                self.merge_log.appendPlainText("用户确认继续, 将舍弃特征: " + ", ".join(drop_features))
            if allow_overrides:
                self.merge_log.appendPlainText("用户确认继续, 全局元数据采用首个数据集的值")
        if target in existing:
            if target in sources:
                text = f"目标 {target} 同时是源数据集, 合并后原数据将被替换 (自动备份)。继续?"
            else:
                text = f"目标 {target} 已存在且不在源列表中, 其内容将被合并结果替换 (自动备份)。继续?"
            ret = QMessageBox.question(self, "确认覆盖", text)
            if ret != QMessageBox.StandardButton.Yes:
                return
        self.merge_start_btn.setEnabled(False)
        self.merge_progress.setValue(0)
        self.merge_log.appendPlainText(f"开始合并: {sources} -> {target}")
        self._merge_target_name = target
        self._merge_worker = MergeWorker(
            self._merge_root, sources, target,
            drop_features=drop_features, allow_overrides=allow_overrides,
        )
        self._merge_worker.sig_log.connect(self.merge_log.appendPlainText)
        self._merge_worker.sig_progress.connect(self._on_merge_progress)
        self._merge_worker.sig_done.connect(self._on_merge_done)
        self._merge_worker.start()

    def _on_merge_progress(self, cur, total, msg):
        self.merge_progress.setMaximum(total)
        self.merge_progress.setValue(cur)

    def _on_merge_done(self, ok, msg):
        self.merge_start_btn.setEnabled(True)
        self.merge_log.appendPlainText(msg)
        target = self._merge_target_name
        self._merge_refresh_list()
        if ok:
            self.merge_completed.emit(target)
        else:
            QMessageBox.critical(self, "合并失败", msg)

    # ==================== 智能降噪面板 ====================
    def _build_denoise_group(self):
        """构建“智能降噪”面板：滤波方法、平滑参数、预览/单个/批量降噪按钮。"""
        group = QGroupBox("智能降噪")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setSpacing(6)

        self.denoise_method_combo = QComboBox()
        self.denoise_method_combo.addItems([
            "Savitzky-Golay（推荐）",
            "移动平均",
            "高斯滤波",
        ])
        self.denoise_method_combo.setToolTip(
            "Savitzky-Golay：零相位、不引入时间滞后，最适合离线轨迹平滑\n"
            "（lerobot 官方仅在遥操作端使用 EMA 在线平滑，离线数据不推荐，"
            "EMA 滞后会导致 action 与图像帧错位）"
        )
        self.denoise_method_combo.currentIndexChanged.connect(self._on_denoise_method_changed)
        form.addRow("滤波方法:", self.denoise_method_combo)

        self.denoise_window_spin = QSpinBox()
        self.denoise_window_spin.setRange(3, 101)
        self.denoise_window_spin.setSingleStep(2)
        self.denoise_window_spin.setValue(11)
        self.denoise_window_spin.setToolTip("平滑窗口长度（帧），越大越平滑；20fps 下 11 帧≈0.55s")
        form.addRow("平滑窗口 (帧):", self.denoise_window_spin)

        self.denoise_poly_spin = QSpinBox()
        self.denoise_poly_spin.setRange(1, 5)
        self.denoise_poly_spin.setValue(2)
        self.denoise_poly_spin.setToolTip("Savitzky-Golay 多项式阶数，需小于窗口长度")
        form.addRow("多项式阶数:", self.denoise_poly_spin)

        self.denoise_sigma_spin = QDoubleSpinBox()
        self.denoise_sigma_spin.setRange(0.3, 20.0)
        self.denoise_sigma_spin.setDecimals(1)
        self.denoise_sigma_spin.setSingleStep(0.5)
        self.denoise_sigma_spin.setValue(2.0)
        self.denoise_sigma_spin.setEnabled(False)
        self.denoise_sigma_spin.setToolTip("高斯滤波标准差（帧），越大越平滑")
        form.addRow("高斯 σ:", self.denoise_sigma_spin)

        layout.addLayout(form)

        target_layout = QHBoxLayout()
        target_layout.setSpacing(6)
        target_layout.addWidget(QLabel("作用对象:"))
        self.denoise_action_check = QCheckBox("action")
        self.denoise_action_check.setChecked(True)
        self.denoise_action_check.setToolTip("平滑 action 轨迹（训练目标，推荐勾选）")
        self.denoise_state_check = QCheckBox("state")
        self.denoise_state_check.setChecked(True)
        self.denoise_state_check.setToolTip("平滑 observation.state 轨迹")
        target_layout.addWidget(self.denoise_action_check)
        target_layout.addWidget(self.denoise_state_check)
        target_layout.addStretch()
        layout.addLayout(target_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self.preview_denoise_btn = QPushButton("🧪 预览降噪")
        self.preview_denoise_btn.setToolTip("在当前曲线上叠加虚线显示平滑效果（不修改数据）")
        self.preview_denoise_btn.setStyleSheet(
            "QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 6px; }"
        )
        self.preview_denoise_btn.clicked.connect(self._on_preview_denoise_clicked)
        btn_layout.addWidget(self.preview_denoise_btn)

        self.denoise_btn = QPushButton("✨ 智能降噪")
        self.denoise_btn.setToolTip("对当前 Episode 应用平滑降噪（只改 state/action 数值，帧数不变）")
        self.denoise_btn.setStyleSheet(
            "QPushButton { background-color: #009688; color: white; font-weight: bold; padding: 6px; }"
        )
        self.denoise_btn.clicked.connect(self._on_denoise_clicked)
        btn_layout.addWidget(self.denoise_btn)

        self.denoise_all_btn = QPushButton("✨ 批量智能降噪")
        self.denoise_all_btn.setToolTip("对所有 Episode 依次应用平滑降噪（逐个处理，避免文件损坏）")
        self.denoise_all_btn.setStyleSheet(
            "QPushButton { background-color: #00796B; color: white; font-weight: bold; padding: 6px; }"
        )
        self.denoise_all_btn.clicked.connect(self._on_denoise_all_clicked)
        btn_layout.addWidget(self.denoise_all_btn)

        layout.addLayout(btn_layout)

        self.denoise_status_label = QLabel("仅修改 state/action 数值，帧数与元数据不变")
        self.denoise_status_label.setWordWrap(True)
        self.denoise_status_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        layout.addWidget(self.denoise_status_label)
        return group

    def _on_denoise_method_changed(self, idx):
        """根据滤波方法启用对应参数。"""
        self.denoise_poly_spin.setEnabled(idx == 0)
        self.denoise_window_spin.setEnabled(idx in (0, 1))
        self.denoise_sigma_spin.setEnabled(idx == 2)

    def _get_denoise_params(self):
        """收集当前降噪参数。"""
        method = ["savgol", "moving_avg", "gaussian"][self.denoise_method_combo.currentIndex()]
        return {
            "method": method,
            "window": int(self.denoise_window_spin.value()),
            "polyorder": int(self.denoise_poly_spin.value()),
            "sigma": float(self.denoise_sigma_spin.value()),
            "smooth_action": self.denoise_action_check.isChecked(),
            "smooth_state": self.denoise_state_check.isChecked(),
        }

    def _on_preview_denoise_clicked(self):
        if self._data is None or self._total_frames == 0:
            self.denoise_status_label.setText("请先选择 Episode")
            return
        self.preview_denoise_requested.emit(self._get_denoise_params())

    def _on_denoise_clicked(self):
        params = self._get_denoise_params()
        if not params["smooth_action"] and not params["smooth_state"]:
            self.denoise_status_label.setText("请至少勾选一个作用对象")
            return
        self.denoise_requested.emit(params)

    def _on_denoise_all_clicked(self):
        params = self._get_denoise_params()
        if not params["smooth_action"] and not params["smooth_state"]:
            self.denoise_status_label.setText("请至少勾选一个作用对象")
            return
        self.denoise_all_requested.emit(params)

    def set_denoise_buttons_enabled(self, enabled):
        self.preview_denoise_btn.setEnabled(enabled)
        self.denoise_btn.setEnabled(enabled)
        self.denoise_all_btn.setEnabled(enabled)

    def set_smoothed_overlay(self, smoothed_data):
        """设置平滑预览数据（与 _data 同结构、同长度），以虚线叠加显示。"""
        self._smoothed = smoothed_data
        if self._data is None:
            return
        self._plot_data(
            self._segment_start if self._segment_mode else 0,
            self._segment_end if self._segment_mode else self._total_frames - 1,
            self._get_current_rel_frame(),
            x_label="片段内帧序号" if self._segment_mode else "帧序号"
        )
        self._apply_view()

    def _on_window_size_changed(self, value):
        self._window_size = max(1, int(value))

    def _on_threshold_changed(self, value):
        self._threshold_pct = float(value)

    def _on_trim_stationary_clicked(self):
        """点击“剪辑静止”按钮：将当前检测到的静止区间提交给主界面处理。"""
        if not self._stationary_intervals:
            QMessageBox.information(
                self, "提示", "当前没有检测到静止区间，请先点击“智能分析”。"
            )
            return
        self.trim_stationary_requested.emit(list(self._stationary_intervals))

    def _on_trim_all_stationary_clicked(self):
        """点击“剪辑所有静止”按钮：请求主界面依次处理所有 Episode。"""
        self.trim_all_stationary_requested.emit()

    def _on_trim_all_jitter_clicked(self):
        """点击“剪辑所有晃动”按钮：请求主界面弹窗选集后依次处理。"""
        self.trim_all_jitter_requested.emit()

    def _compute_y_range(self, keys, start=0, end=None):
        """计算指定帧范围 [start, end] 内给定数据键的 Y 轴范围。"""
        if self._data is None:
            return 0.0, 1.0
        if end is None:
            end = self._total_frames - 1
        s = max(0, min(start, end, self._total_frames - 1))
        e = min(self._total_frames, max(start, end, 0) + 1)
        if e <= s:
            e = s + 1
        arrays = [self._data.get(k, [])[s:e] for k in keys if k in self._data]
        if not arrays:
            return 0.0, 1.0
        all_vals = np.concatenate(arrays)
        if len(all_vals) == 0:
            return 0.0, 1.0
        ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
        if abs(ymax - ymin) < 1e-9:
            ymax = ymin + 1.0
        return ymin, ymax

    def _capture_backgrounds(self):
        """捕获三幅子图的背景，用于 blit 快速更新播放竖线。"""
        try:
            self.canvas.draw()
            self._bg_pos = self.canvas.copy_from_bbox(self.ax_pos.bbox)
            self._bg_euler = self.canvas.copy_from_bbox(self.ax_euler.bbox)
            self._bg_grip = self.canvas.copy_from_bbox(self.ax_grip.bbox)
        except Exception:
            self._bg_pos = None
            self._bg_euler = None
            self._bg_grip = None

    def _empty_plot(self):
        """清空为占位状态。"""
        for ax in self.axes:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.5)
        self.ax_torque.clear()
        self.ax_torque.grid(False)

        # ax.clear() 已经移除所有旧 artist
        self._stationary_bars = []

        self.ax_pos.set_ylabel("位置 (m)")
        self.ax_euler.set_ylabel("欧拉角 (rad)")
        self.ax_grip.set_ylabel("夹爪 (rad)")
        self.ax_torque.set_ylabel("力矩 (Nm)")
        self.ax_grip.set_xlabel("帧序号")

        self.ax_pos.set_title("Episode 末端位姿变化曲线")
        self.ax_pos.text(
            0.5, 0.5, "请选择 Episode",
            horizontalalignment='center', verticalalignment='center',
            transform=self.ax_pos.transAxes, fontsize=12, color='gray'
        )

        self.lines = {}
        self.vlines = []
        self.cursor_label = None
        self._data = None
        self._smoothed = None
        self._total_frames = 0
        self._segment_mode = False
        self._segment_start = 0
        self._segment_end = 0
        self._bg_pos = None
        self._bg_euler = None
        self._bg_grip = None
        self._stationary_intervals = []
        self.result_text.setPlaceholderText("点击“智能分析”查看结果")
        self.canvas.draw_idle()

    def _draw_stationary_bars(self):
        """根据当前分析结果，在三幅子图中同步绘制静止区间灰色条。"""
        # 清除旧条（容错：可能已被 ax.clear() 移除）
        for bar in self._stationary_bars:
            try:
                bar.remove()
            except Exception:
                pass
        self._stationary_bars = []

        if not self._stationary_intervals or self._data is None:
            return

        if self._segment_mode:
            seg_s, seg_e = self._segment_start, self._segment_end
        else:
            seg_s, seg_e = 0, self._total_frames - 1

        for s_ep, e_ep in self._stationary_intervals:
            vs = max(s_ep, seg_s)
            ve = min(e_ep, seg_e)
            if vs > ve:
                continue
            if self._segment_mode:
                x0 = vs - seg_s - 0.5
                x1 = ve - seg_s + 0.5
            else:
                x0 = vs - 0.5
                x1 = ve + 0.5

            # 在位置、欧拉角、夹爪子图中同步标注
            first = not self._stationary_bars
            for ax in self.axes:
                bar = ax.axvspan(
                    x0, x1, color='#AAAAAA', alpha=0.35, zorder=0,
                    label='静止区间' if (first and ax is self.ax_pos) else ''
                )
                self._stationary_bars.append(bar)
                first = False

    # ==================== 自动语言标注（子任务分段） ====================
    def _draw_subtask_bars(self):
        """在夹爪曲线（及其它子图）横坐标上用不同颜色标注各子任务片段。"""
        if not self._subtask_segments or self._data is None:
            return

        if self._segment_mode:
            seg_s, seg_e = self._segment_start, self._segment_end
        else:
            seg_s, seg_e = 0, self._total_frames - 1

        first = True
        for seg in self._subtask_segments:
            vs = max(seg["start"], seg_s)
            ve = min(seg["end"], seg_e)
            if vs > ve:
                continue
            if self._segment_mode:
                x0 = vs - seg_s - 0.5
                x1 = ve - seg_s + 0.5
            else:
                x0 = vs - 0.5
                x1 = ve + 0.5

            color = seg.get("color", "#4CAF50")
            task = seg.get("task", "")
            # 在三幅子图同步着色（夹爪子图最深，便于辨识）
            for ax in self.axes:
                alpha = 0.30 if ax is self.ax_grip else 0.16
                bar = ax.axvspan(
                    x0, x1, color=color, alpha=alpha, zorder=0,
                    label=task if (first and ax is self.ax_grip) else "",
                )
                self._subtask_bars.append(bar)
            # 段边界竖线（仅在夹爪子图）
            vl = self.ax_grip.axvline(x0, color=color, linewidth=1.0,
                                      linestyle=':', alpha=0.7, zorder=1)
            self._subtask_vlines.append(vl)
            # 子任务名标签（夹爪子图顶部，旋转避免重叠）
            xc = (x0 + x1) / 2.0
            txt = self.ax_grip.text(
                xc, 1.02, task, transform=self.ax_grip.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=8, color=color,
                rotation=0, clip_on=False,
            )
            self._subtask_texts.append(txt)
            first = False

    def _run_smart_segment(self):
        """按力矩+位姿切出指定段数，在曲线上着色，标签表用占位名。"""
        if self._data is None or self._total_frames == 0:
            self.subtask_result_label.setText("请先选择 Episode。")
            return
        n = self.subtask_count_spin.value()
        use_torque = self.subtask_torque_chk.isChecked()
        use_gripper = self.subtask_angle_chk.isChecked()
        try:
            from auto_language_annotate import segment_fixed_count
            intervals = segment_fixed_count(self._data, n, use_torque=use_torque, use_gripper=use_gripper)
        except Exception as ex:
            self.subtask_result_label.setText(f"分段失败: {ex}")
            return
        self._subtask_segments = [{
            "start": int(s), "end": int(e),
            "task": PRESET_LABELS[i] if i < len(PRESET_LABELS) else f"步骤{i + 1}",
            "color": color_for_index(i), "note": "auto-fixed",
        } for i, (s, e) in enumerate(intervals)]
        self._populate_label_table()
        self._redraw_subtask()
        self.subtask_vlm_btn.setEnabled(bool(self._subtask_segments))
        self.subtask_apply_btn.setEnabled(bool(self._subtask_segments))
        self.subtask_result_label.setText(
            f"已切 {len(self._subtask_segments)} 段（力矩={use_torque}, 夹角={use_gripper}）。可点“生成标签(VLM)”自动命名，或双击表格手动改。")

    def _apply_preset_labels(self):
        """把预设名称按当前段数套用到分段（不足取前 N，超出保持 "步骤N"）。"""
        if not self._subtask_segments:
            self.subtask_result_label.setText("请先点“智能分段”切出段落后再套用预设。")
            return
        for i, seg in enumerate(self._subtask_segments):
            seg["task"] = PRESET_LABELS[i] if i < len(PRESET_LABELS) else f"步骤{i + 1}"
        self._populate_label_table()
        self._redraw_subtask()
        self.subtask_result_label.setText(
            f"已套用预设名称（{len(self._subtask_segments)} 段）。可双击表格微调，或点“生成标签(VLM)”用模型重命名。")

    def _populate_label_table(self):
        """用当前分段刷新可编辑标签表（阻塞 itemChanged 避免回环）。"""
        self.subtask_label_table.blockSignals(True)
        self.subtask_label_table.setRowCount(len(self._subtask_segments))
        for i, seg in enumerate(self._subtask_segments):
            it0 = QTableWidgetItem(str(i + 1))
            it0.setFlags(Qt.ItemFlag.ItemIsEnabled)
            it1 = QTableWidgetItem(seg["task"])
            it1.setToolTip(f"帧 {seg['start']}~{seg['end']}（{seg['end']-seg['start']+1} 帧）")
            self.subtask_label_table.setItem(i, 0, it0)
            self.subtask_label_table.setItem(i, 1, it1)
        self.subtask_label_table.blockSignals(False)

    def _on_label_table_edited(self, item):
        """用户在标签表里手动改名 → 同步到分段并重绘着色。"""
        if item.column() != 1:
            return
        row = item.row()
        if 0 <= row < len(self._subtask_segments):
            self._subtask_segments[row]["task"] = item.text().strip() or f"步骤{row + 1}"
            self._redraw_subtask()

    def _on_vlm_label_clicked(self):
        """请求主窗口用 VLM 给当前分段生成标签。"""
        if not self._subtask_segments:
            return
        self.subtask_vlm_btn.setEnabled(False)
        self.subtask_result_label.setText("VLM 分析中（后台），请稍候…")
        self.vlm_label_requested.emit(len(self._subtask_segments))

    def set_segment_labels(self, labels, note_msg=""):
        """主窗口 VLM 线程完成后回调：更新标签表与着色。"""
        if not self._subtask_segments:
            return
        for i, lb in enumerate(labels):
            if i < len(self._subtask_segments):
                self._subtask_segments[i]["task"] = str(lb)
        self._populate_label_table()
        self._redraw_subtask()
        self.subtask_vlm_btn.setEnabled(True)
        self.subtask_result_label.setText(note_msg or f"VLM 已为 {len(labels)} 段生成标签，可手动微调后应用。")

    def _clear_subtask_annotate(self):
        """清除曲线上的子任务颜色标注。"""
        self._subtask_segments = []
        self.subtask_label_table.setRowCount(0)
        self.subtask_vlm_btn.setEnabled(False)
        self.subtask_apply_btn.setEnabled(False)
        self.subtask_result_label.setText("（已清除）设段数 → 智能分段 → 生成标签 → 应用")
        self._plot_data(
            self._segment_start if self._segment_mode else 0,
            self._segment_end if self._segment_mode else self._total_frames - 1,
            self._get_current_rel_frame(),
            x_label="片段内帧序号" if self._segment_mode else "帧序号",
        )
        self._apply_view()

    def _redraw_subtask(self):
        """重绘曲线以反映当前分段着色。"""
        self._plot_data(
            self._segment_start if self._segment_mode else 0,
            self._segment_end if self._segment_mode else self._total_frames - 1,
            self._get_current_rel_frame(),
            x_label="片段内帧序号" if self._segment_mode else "帧序号",
        )
        self._apply_view()

    def _on_apply_subtask_clicked(self):
        """请求主窗口把预览结果写回数据集。"""
        if not self._subtask_segments:
            return
        self.subtask_apply_requested.emit(list(self._subtask_segments))

    def _plot_data(self, slice_start, slice_end, rel_frame, x_label="帧序号"):
        """内部绘图：传入已经切好的数据切片和当前帧（均已是当前模式的坐标）。"""
        for ax in self.axes:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.5)
        self.ax_torque.clear()
        self.ax_torque.grid(False)

        # ax.clear() 已移除所有旧 artist，清空本地缓存避免重复 remove
        self._stationary_bars = []
        self._jitter_bars = []
        self._subtask_bars = []
        self._subtask_texts = []
        self._subtask_vlines = []

        self.ax_grip.set_xlabel(x_label)

        title = "Episode 末端位姿变化曲线"
        if self._segment_mode:
            title += f"  [片段 {self._segment_start}~{self._segment_end}]"
        self.ax_pos.set_title(title)

        if self._data is None:
            self._empty_plot()
            return

        n = len(self._data.get('x', []))
        if n == 0:
            self._empty_plot()
            return

        s = max(0, min(slice_start, slice_end, n - 1))
        e = min(n, max(slice_start, slice_end, 0) + 1)
        if e <= s:
            e = s + 1
        indices = np.arange(e - s)

        # 绘制静止区间灰色条（在曲线下层）
        self._draw_stationary_bars()

        # 绘制抖动区间紫色半透明条（后期处理分页，在静止条之上）
        self._draw_jitter_bars()

        # 绘制子任务语言标注彩色条（在曲线下层、静止条之上）
        self._draw_subtask_bars()

        # 位置曲线（原始值）
        sx = self._data.get('x', [])[s:e]
        sy = self._data.get('y', [])[s:e]
        sz = self._data.get('z', [])[s:e]
        self.ax_pos.set_ylabel("位置 (m)")
        has_arm2 = 'x2' in self._data
        self.lines['x'], = self.ax_pos.plot(indices, sx, 'r-', label='X1' if has_arm2 else 'X', linewidth=1.2, zorder=2)
        self.lines['y'], = self.ax_pos.plot(indices, sy, 'g-', label='Y1' if has_arm2 else 'Y', linewidth=1.2, zorder=2)
        self.lines['z'], = self.ax_pos.plot(indices, sz, 'b-', label='Z1' if has_arm2 else 'Z', linewidth=1.2, zorder=2)
        # 臂2 位置曲线：同色虚线叠加
        if has_arm2:
            self.lines['x2'], = self.ax_pos.plot(indices, self._data['x2'][s:e], 'r--', label='X2', linewidth=1.2, alpha=0.8, zorder=2)
            self.lines['y2'], = self.ax_pos.plot(indices, self._data['y2'][s:e], 'g--', label='Y2', linewidth=1.2, alpha=0.8, zorder=2)
            self.lines['z2'], = self.ax_pos.plot(indices, self._data['z2'][s:e], 'b--', label='Z2', linewidth=1.2, alpha=0.8, zorder=2)
        self.ax_pos.legend(loc='upper right', fontsize=8)

        # 欧拉角曲线（rx/ry/rz，弧度）
        rx = self._data.get('rx', [])[s:e]
        ry = self._data.get('ry', [])[s:e]
        rz = self._data.get('rz', [])[s:e]
        self.ax_euler.set_ylabel("欧拉角 (rad)")
        self.lines['rx'], = self.ax_euler.plot(indices, rx, 'c-', label='rx1' if has_arm2 else 'rx', linewidth=1.2, zorder=2)
        self.lines['ry'], = self.ax_euler.plot(indices, ry, 'm-', label='ry1' if has_arm2 else 'ry', linewidth=1.2, zorder=2)
        self.lines['rz'], = self.ax_euler.plot(indices, rz, 'y-', label='rz1' if has_arm2 else 'rz', linewidth=1.2, zorder=2)
        # 臂2 欧拉角曲线：同色虚线叠加
        if has_arm2:
            self.lines['rx2'], = self.ax_euler.plot(indices, self._data['rx2'][s:e], 'c--', label='rx2', linewidth=1.2, alpha=0.8, zorder=2)
            self.lines['ry2'], = self.ax_euler.plot(indices, self._data['ry2'][s:e], 'm--', label='ry2', linewidth=1.2, alpha=0.8, zorder=2)
            self.lines['rz2'], = self.ax_euler.plot(indices, self._data['rz2'][s:e], 'y--', label='rz2', linewidth=1.2, alpha=0.8, zorder=2)
        self.ax_euler.legend(loc='upper right', fontsize=8)

        # 夹爪曲线（弧度）
        grip = self._data.get('gripper', [])[s:e]
        self.ax_grip.set_ylabel("夹爪 (rad)")
        self.lines['gripper'], = self.ax_grip.plot(
            indices, grip, '#FF5722', label='夹爪1' if has_arm2 else '夹爪', linewidth=1.5, zorder=2
        )
        if has_arm2 and 'gripper2' in self._data:
            self.lines['gripper2'], = self.ax_grip.plot(
                indices, self._data['gripper2'][s:e], '#FF5722', linestyle='--',
                label='夹爪2', linewidth=1.5, alpha=0.8, zorder=2
            )
        self.ax_grip.legend(loc='upper left', fontsize=8)

        # 夹爪力矩曲线（Nm，右侧 twin axis；仅有数据时绘制；臂2 用虚线）
        if self._has_torque and 'gripper_torque' in self._data:
            torque = self._data.get('gripper_torque', [])[s:e]
            self.ax_torque.set_ylabel("力矩 (Nm)")
            self.lines['gripper_torque'], = self.ax_torque.plot(
                indices, torque, '#9C27B0', label='力矩1' if has_arm2 else '力矩', linewidth=1.2,
                alpha=0.8, zorder=2
            )
            if has_arm2 and 'gripper_torque2' in self._data:
                self.lines['gripper_torque2'], = self.ax_torque.plot(
                    indices, self._data['gripper_torque2'][s:e], '#9C27B0',
                    linestyle='--', label='力矩2', linewidth=1.2, alpha=0.8, zorder=2
                )
            self.ax_torque.legend(loc='upper right', fontsize=8)
        else:
            self.ax_torque.set_yticks([])

        # 降噪预览：以虚线叠加平滑后的位置/欧拉角/夹爪曲线
        if self._smoothed is not None:
            sm = self._smoothed
            labeled = False
            for key, color in [('x', 'r'), ('y', 'g'), ('z', 'b')]:
                if key in sm and len(sm[key]) >= e:
                    self.ax_pos.plot(
                        indices, sm[key][s:e], color + '--', alpha=0.85, linewidth=1.0,
                        zorder=2, label='平滑预览' if not labeled else None
                    )
                    labeled = True
            for key, color in [('rx', 'c'), ('ry', 'm'), ('rz', 'y')]:
                if key in sm and len(sm[key]) >= e:
                    self.ax_euler.plot(indices, sm[key][s:e], color + '--',
                                       alpha=0.85, linewidth=1.0, zorder=2)
            if 'gripper' in sm and len(sm['gripper']) >= e:
                self.ax_grip.plot(indices, sm['gripper'][s:e], '#FF5720',
                                  linestyle='--', alpha=0.85, linewidth=1.0, zorder=2)
            if labeled:
                self.ax_pos.legend(loc='upper right', fontsize=8)

        frame = max(0, min(rel_frame, e - s - 1))
        self.vlines = [
            self.ax_pos.axvline(x=frame, color='k', linestyle='--', linewidth=1.5, zorder=3),
            self.ax_euler.axvline(x=frame, color='k', linestyle='--', linewidth=1.5, zorder=3),
            self.ax_grip.axvline(x=frame, color='k', linestyle='--', linewidth=1.5, zorder=3),
        ]

        # 播放标签：显示当前帧号，放在顶部子图 Y 轴可视范围顶部
        y_top = self._y_center + self._y_half_range if self._y_half_range > 0 else 1.0
        self.cursor_label = self.ax_pos.annotate(
            f"帧{frame}",
            xy=(frame, y_top),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=9,
            color='black',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8, edgecolor='black'),
            arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
            zorder=4,
        )

    def _update_analysis_result_text(self):
        """更新分析结果文本。"""
        if not self._stationary_intervals:
            self.result_text.setPlainText(
                "未检测到机械臂静止状态。\n\n"
                "提示：增大“静止窗口”或放宽“变化阈值”可增加检测概率。"
            )
            return
        lines = []
        total = 0
        for s, e in self._stationary_intervals:
            length = e - s + 1
            total += length
            lines.append(f"帧 {s}~{e} （{length} 帧）")
        pct = total / max(self._total_frames, 1) * 100
        result = (
            f"检测到 {len(self._stationary_intervals)} 段静止，"
            f"共 {total}/{self._total_frames} 帧 ({pct:.1f}%)\n\n"
        )
        result += "\n".join(lines)
        self.result_text.setPlainText(result)

    def _run_analysis(self):
        """智能分析：检测 7 个指标同时变化较小的连续帧区间。"""
        if self._data is None or self._total_frames == 0:
            self.result_text.setPlainText("请先选择 Episode。")
            return

        # 检测算法已提取为模块级 detect_stationary_intervals（向量化实现），
        # 供“剪辑所有静止”批量流程在不加载 UI 的情况下直接调用，保证两处结果一致
        self._stationary_intervals = detect_stationary_intervals(
            self._data, self._window_size, self._threshold_pct)
        self._update_analysis_result_text()

        # 重绘以显示灰色条
        self._plot_data(
            self._segment_start if self._segment_mode else 0,
            self._segment_end if self._segment_mode else self._total_frames - 1,
            self._get_current_rel_frame(),
            x_label="片段内帧序号" if self._segment_mode else "帧序号"
        )
        self._apply_view()

    def set_data(self, frames_data, current_frame=0, clear_annotations=True):
        """设置并绘制整个 Episode 的位姿曲线（绝对帧序号）。

        clear_annotations=True（默认，切换 Episode 时）会清空子任务分段与标签、
        静止/抖动分析结果；False（如点播放重绘同一 episode）则全部保留在曲线上
        （播放视频时智能分析/抖动的半透明标记不消失），仅随 _plot_data 重画。
        """
        self._data = frames_data
        self._current_frame = current_frame
        self._segment_mode = False
        self._smoothed = None
        if clear_annotations:
            self._stationary_intervals = []
            for bar in self._stationary_bars:
                bar.remove()
            self._stationary_bars = []
            # 切换 Episode：清空抖动分析结果
            self._jitter_intervals = []
            self._jitter_bars = []
            if hasattr(self, "jitter_result_label"):
                self.jitter_result_label.setText(
                    "（设参数 → 分析抖动 → 核对紫色区域 → 剪辑晃动）")
        # _plot_data 内 ax.clear() 已移除旧 artist，这里只需清空本地句柄缓存；
        # 仅在 clear_annotations=True（切换 Episode）时才丢弃分段数据与标签。
        self._subtask_bars = []
        self._subtask_texts = []
        self._subtask_vlines = []
        if clear_annotations:
            self._subtask_segments = []
            if hasattr(self, "subtask_apply_btn"):
                self.subtask_apply_btn.setEnabled(False)
            if hasattr(self, "subtask_vlm_btn"):
                self.subtask_vlm_btn.setEnabled(False)
            if hasattr(self, "subtask_label_table"):
                self.subtask_label_table.setRowCount(0)
            if hasattr(self, "subtask_result_label"):
                self.subtask_result_label.setText("（设段数 → 智能分段 → 生成标签 → 应用）")
        else:
            # 保留分段：重绘后按是否有分段恢复按钮可用状态
            has_seg = bool(self._subtask_segments)
            if hasattr(self, "subtask_apply_btn"):
                self.subtask_apply_btn.setEnabled(has_seg)
            if hasattr(self, "subtask_vlm_btn"):
                self.subtask_vlm_btn.setEnabled(has_seg)

        if frames_data is None:
            self._empty_plot()
            return

        x = frames_data.get('x')
        if x is None or len(x) == 0:
            self._empty_plot()
            return

        self._total_frames = len(x)
        self._segment_start = 0
        self._segment_end = self._total_frames - 1

        # 先计算 Y 范围，_plot_data 中的 cursor_label 会用到（双臂时并入臂2 键，缺失自动跳过）
        ymin, ymax = self._compute_y_range(['x', 'y', 'z', 'x2', 'y2', 'z2'], 0, self._total_frames - 1)
        self._y_center = (ymin + ymax) / 2.0
        self._y_base_half_range = (ymax - ymin) / 2.0
        self._y_half_range = self._y_base_half_range

        qmin, qmax = self._compute_y_range(['rx', 'ry', 'rz', 'rx2', 'ry2', 'rz2'], 0, self._total_frames - 1)
        self._euler_y_center = (qmin + qmax) / 2.0
        self._euler_y_half_range = max((qmax - qmin) / 2.0, 1e-6)

        gmin, gmax = self._compute_y_range(['gripper', 'gripper2'], 0, self._total_frames - 1)
        self._grip_y_center = (gmin + gmax) / 2.0
        self._grip_y_half_range = max((gmax - gmin) / 2.0, 1e-6)

        # 夹爪力矩 Y 轴范围（仅有数据时计算）
        self._has_torque = 'gripper_torque' in (frames_data or {})
        if self._has_torque:
            tmin, tmax = self._compute_y_range(['gripper_torque', 'gripper_torque2'], 0, self._total_frames - 1)
            self._torque_y_center = (tmin + tmax) / 2.0
            self._torque_y_half_range = max((tmax - tmin) / 2.0, 1e-6)
        else:
            self._torque_y_center = 0.0
            self._torque_y_half_range = 1.0

        # 先按默认参数完成静止分析（纯数据向量化计算，约 1ms），再一次性绘图——
        # 避免旧流程“先绘一次、_run_analysis 后再绘一次”的两次全量重绘，
        # 切换 Episode 的曲线绘制耗时减半（实测 ~400ms → ~200ms）
        self._stationary_intervals = detect_stationary_intervals(
            frames_data, self._window_size, self._threshold_pct)
        self._update_analysis_result_text()

        self._plot_data(0, self._total_frames - 1, current_frame, x_label="帧序号")

        # 视图中心和基准按整个 Episode 设置；视图中心默认跟随当前播放标签
        self._x_base_half_width = self._total_frames / 2.0
        self._view_half_width = self._x_base_half_width
        self._update_view_center_to_current_frame()

        self._x_zoom_level = 1
        self._y_zoom_level = 1
        self.x_zoom_slider.blockSignals(True)
        self.x_zoom_slider.setValue(1)
        self.x_zoom_slider.blockSignals(False)
        self.y_zoom_slider.blockSignals(True)
        self.y_zoom_slider.setValue(1)
        self.y_zoom_slider.blockSignals(False)

        self._apply_view()
        # 静止分析与灰色条已在上面单次绘图中完成，无需再调 _run_analysis 重绘

    def set_segment_range(self, start, end):
        """切换到片段模式：以相对帧序号 0~N-1 显示 [start, end] 内所有帧的曲线。"""
        if self._data is None or self._total_frames <= 0:
            return
        start = max(0, min(start, self._total_frames - 1))
        end = max(0, min(end, self._total_frames - 1))
        if start > end:
            start, end = end, start
        if start == end:
            return

        self._segment_mode = True
        self._segment_start = start
        self._segment_end = end
        segment_len = end - start + 1

        # 先计算片段 Y 范围（双臂时并入臂2 键）
        ymin, ymax = self._compute_y_range(['x', 'y', 'z', 'x2', 'y2', 'z2'], start, end)
        self._y_center = (ymin + ymax) / 2.0
        self._y_base_half_range = (ymax - ymin) / 2.0
        self._y_half_range = self._y_base_half_range

        qmin, qmax = self._compute_y_range(['rx', 'ry', 'rz', 'rx2', 'ry2', 'rz2'], start, end)
        self._euler_y_center = (qmin + qmax) / 2.0
        self._euler_y_half_range = max((qmax - qmin) / 2.0, 1e-6)

        gmin, gmax = self._compute_y_range(['gripper', 'gripper2'], start, end)
        self._grip_y_center = (gmin + gmax) / 2.0
        self._grip_y_half_range = max((gmax - gmin) / 2.0, 1e-6)

        if self._has_torque:
            tmin, tmax = self._compute_y_range(['gripper_torque', 'gripper_torque2'], start, end)
            self._torque_y_center = (tmin + tmax) / 2.0
            self._torque_y_half_range = max((tmax - tmin) / 2.0, 1e-6)

        rel_frame = max(0, min(self._current_frame - start, segment_len - 1))
        self._plot_data(start, end, rel_frame, x_label="片段内帧序号")

        # 视图中心和基准按片段设置
        self._view_center = segment_len / 2.0
        self._x_base_half_width = segment_len / 2.0
        self._view_half_width = self._x_base_half_width

        self._x_zoom_level = 1
        self._y_zoom_level = 1
        self.x_zoom_slider.blockSignals(True)
        self.x_zoom_slider.setValue(1)
        self.x_zoom_slider.blockSignals(False)
        self.y_zoom_slider.blockSignals(True)
        self.y_zoom_slider.setValue(1)
        self.y_zoom_slider.blockSignals(False)

        self._apply_view()

    def _on_x_zoom_changed(self, value):
        """X 轴缩放：始终以当前播放标签为中心放大两侧曲线（同步三幅子图）。"""
        if self._data is None:
            return
        self._x_zoom_level = max(1, value)
        self._view_half_width = max(1.0, self._x_base_half_width / self._x_zoom_level)
        self._update_view_center_to_current_frame()
        self._apply_view()

    def _on_y_zoom_changed(self, value):
        """Y 轴缩放：仅作用于位置曲线。"""
        if self._data is None:
            return
        self._y_zoom_level = max(1, value)
        self._y_half_range = max(1e-6, self._y_base_half_range / self._y_zoom_level)
        self._apply_view()

    def _reset_zoom(self):
        """重置为显示整个 Episode（绝对帧序号）。"""
        if self._data is None or self._total_frames <= 0:
            return
        # 直接切回全 Episode 视图（同一 episode 重绘，保留子任务标注）
        self.set_data(self._data, self._current_frame, clear_annotations=False)

    def _get_current_rel_frame(self):
        """返回当前帧在当前坐标系（全 Episode 或片段）中的相对位置。"""
        if self._data is None:
            return 0
        if self._segment_mode:
            total_n = self._segment_end - self._segment_start + 1
            return max(0, min(self._current_frame - self._segment_start, total_n - 1))
        else:
            return max(0, min(self._current_frame, self._total_frames - 1))

    def _update_view_center_to_current_frame(self):
        """将视图中心设为当前播放标签所在帧。"""
        self._view_center = float(self._get_current_rel_frame())

    def _update_cursor_artist(self):
        """更新播放竖线和标签的位置/文本。"""
        if not self.vlines:
            return
        rel_frame = self._get_current_rel_frame()
        for vline in self.vlines:
            vline.set_xdata([rel_frame, rel_frame])

        if self.cursor_label is not None:
            # 标签 y 坐标放在位置子图 Y 轴可视范围顶部
            y_top = self._y_center + self._y_half_range
            self.cursor_label.xy = (rel_frame, y_top)
            self.cursor_label.set_text(f"帧{self._current_frame}")

    def set_current_frame(self, frame):
        """更新当前帧：竖线、标签同步滑动；若处于 X 放大状态则视图跟随标签平移。"""
        if not self.vlines or self._data is None:
            return
        new_frame = max(0, min(int(frame), self._total_frames - 1))
        if new_frame == self._current_frame:
            return
        self._current_frame = new_frame
        self._update_view_center_to_current_frame()
        self._update_cursor_artist()

        # 若 X 方向有放大，则整体视图以播放标签为中心平移
        if self._x_zoom_level > 1:
            self._apply_view()
            return

        # 无 X 放大时，使用 blit 只重绘竖线和标签
        if self._bg_pos is not None and self._bg_euler is not None and self._bg_grip is not None:
            try:
                self.canvas.restore_region(self._bg_pos)
                self.canvas.restore_region(self._bg_euler)
                self.canvas.restore_region(self._bg_grip)

                self.ax_pos.draw_artist(self.vlines[0])
                self.ax_euler.draw_artist(self.vlines[1])
                self.ax_grip.draw_artist(self.vlines[2])
                if self.cursor_label is not None:
                    self.ax_pos.draw_artist(self.cursor_label)

                self.canvas.blit(self.ax_pos.bbox)
                self.canvas.blit(self.ax_euler.bbox)
                self.canvas.blit(self.ax_grip.bbox)
                return
            except Exception:
                self._bg_pos = None
                self._bg_euler = None
                self._bg_grip = None
        self.canvas.draw_idle()

    def _apply_view(self):
        """应用当前视图范围并重绘。

        zoom=1 时始终显示整个 Episode/片段；zoom>1 时以播放标签为中心放大两侧。
        """
        if self._data is None:
            return

        # 当前坐标系下的总点数
        if self._segment_mode:
            total_n = self._segment_end - self._segment_start + 1
        else:
            total_n = self._total_frames

        # X 轴范围（同步三幅子图）
        if self._x_zoom_level <= 1:
            # 无放大：显示全部帧
            x_min = 0.0
            x_max = max(0.0, total_n - 1)
        else:
            # 放大：以播放标签为中心
            x_min = max(0.0, self._view_center - self._view_half_width)
            x_max = min(total_n - 1, self._view_center + self._view_half_width)
        if x_max <= x_min:
            x_max = x_min + 1
        for ax in self.axes:
            ax.set_xlim(x_min, x_max)

        # 位置 Y 轴范围（受 Y 缩放控制）
        self.ax_pos.set_ylim(
            self._y_center - self._y_half_range,
            self._y_center + self._y_half_range
        )

        # 欧拉角 Y 轴范围（固定自适应）
        self.ax_euler.set_ylim(
            self._euler_y_center - self._euler_y_half_range,
            self._euler_y_center + self._euler_y_half_range
        )

        # 夹爪 Y 轴范围（固定自适应）
        self.ax_grip.set_ylim(
            self._grip_y_center - self._grip_y_half_range,
            self._grip_y_center + self._grip_y_half_range
        )

        # 夹爪力矩 Y 轴范围（右侧 twin axis，固定自适应）
        self.ax_torque.set_ylim(
            self._torque_y_center - self._torque_y_half_range,
            self._torque_y_center + self._torque_y_half_range
        )

        # 竖线和标签
        self._update_cursor_artist()

        self.canvas.draw_idle()
        # 绘制完成后缓存三幅子图背景
        self._capture_backgrounds()

    def clear(self):
        """清空曲线。"""
        self._data = None
        self._smoothed = None
        self._current_frame = 0
        self._total_frames = 0
        self._segment_mode = False
        self._segment_start = 0
        self._segment_end = 0
        self._view_center = 0.0
        self._view_half_width = 0.0
        self._x_base_half_width = 0.0
        self._y_center = 0.0
        self._y_half_range = 0.0
        self._y_base_half_range = 0.0
        self._x_zoom_level = 1
        self._y_zoom_level = 1
        self._euler_y_center = 0.0
        self._euler_y_half_range = 0.0
        self._grip_y_center = 0.0
        self._grip_y_half_range = 0.0
        self._torque_y_center = 0.0
        self._torque_y_half_range = 1.0
        self._has_torque = False
        self._bg_pos = None
        self._bg_euler = None
        self._bg_grip = None
        self._stationary_intervals = []
        for bar in self._stationary_bars:
            bar.remove()
        self._stationary_bars = []
        self._jitter_intervals = []
        self._jitter_bars = []
        self.x_zoom_slider.blockSignals(True)
        self.x_zoom_slider.setValue(1)
        self.x_zoom_slider.blockSignals(False)
        self.y_zoom_slider.blockSignals(True)
        self.y_zoom_slider.setValue(1)
        self.y_zoom_slider.blockSignals(False)
        self._empty_plot()


# ==================== 底盘 / 升降台曲线面板 ====================
class ChassisPlotWidget(QWidget):
    """底盘速度（vx/vy/omega）与升降台位置（lift）双子图曲线。

    数据键（由 MainWindow._load_episode_position_data 提供，缺失则不绘制）：
        lift  升降台绝对位置 (m)
        vx    底盘机体系前向速度 (m/s)
        vy    底盘机体系横向速度 (m/s)
        omega 底盘旋转角速度 (rad/s)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._total_frames = 0
        self._current_frame = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(5, 3.2), dpi=100, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(220)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self.canvas, stretch=1)

        self.ax_lift = self.figure.add_subplot(211)
        self.ax_vel = self.figure.add_subplot(212, sharex=self.ax_lift)
        self.axes = [self.ax_lift, self.ax_vel]
        self.ax_lift.set_ylabel("升降 (m)")
        self.ax_vel.set_ylabel("速度")
        self.ax_vel.set_xlabel("帧序号")
        self.vlines = []
        self._empty_plot()

    def _empty_plot(self):
        for ax in self.axes:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.5)
        self.ax_lift.set_ylabel("升降 (m)")
        self.ax_vel.set_ylabel("速度 (m/s, rad/s)")
        self.ax_vel.set_xlabel("帧序号")
        self.vlines = []
        self.canvas.draw_idle()

    def set_data(self, chassis_data, current_frame=0):
        """设置并绘制整个 Episode 的底盘/升降曲线；None 或空则显示占位。"""
        self._data = chassis_data
        self._current_frame = current_frame
        if not chassis_data:
            self._total_frames = 0
            self._empty_plot()
            return
        lift = chassis_data.get('lift')
        vx = chassis_data.get('vx')
        if lift is None and vx is None:
            self._total_frames = 0
            self._empty_plot()
            return
        ref = lift if lift is not None else vx
        n = len(ref)
        if n == 0:
            self._total_frames = 0
            self._empty_plot()
            return
        self._total_frames = n

        for ax in self.axes:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.5)
        indices = np.arange(n)

        if lift is not None:
            self.ax_lift.plot(indices, lift, '#3F51B5', label='升降', linewidth=1.2)
        self.ax_lift.set_ylabel("升降 (m)")
        self.ax_lift.legend(loc='upper right', fontsize=8)

        for key, color, label in [('vx', 'r', 'vx'), ('vy', 'g', 'vy'), ('omega', 'y', 'omega')]:
            arr = chassis_data.get(key)
            if arr is not None:
                self.ax_vel.plot(indices, arr, color + '-', label=label, linewidth=1.2)
        self.ax_vel.set_ylabel("速度 (m/s, rad/s)")
        self.ax_vel.set_xlabel("帧序号")
        self.ax_vel.legend(loc='upper right', fontsize=8)

        for ax in self.axes:
            ax.set_xlim(0, max(0, n - 1))
        self.ax_lift.relim()
        self.ax_lift.autoscale_view(scalex=False)
        self.ax_vel.relim()
        self.ax_vel.autoscale_view(scalex=False)

        frame = max(0, min(int(current_frame), n - 1))
        self.vlines = [
            self.ax_lift.axvline(x=frame, color='k', linestyle='--', linewidth=1.5, zorder=3),
            self.ax_vel.axvline(x=frame, color='k', linestyle='--', linewidth=1.5, zorder=3),
        ]
        self.canvas.draw_idle()

    def set_current_frame(self, frame):
        """同步播放竖线位置。"""
        if not self.vlines or self._data is None or self._total_frames <= 0:
            return
        new_frame = max(0, min(int(frame), self._total_frames - 1))
        if new_frame == self._current_frame:
            return
        self._current_frame = new_frame
        for vline in self.vlines:
            vline.set_xdata([new_frame, new_frame])
        self.canvas.draw_idle()

    def clear(self):
        """清空曲线。"""
        self._data = None
        self._total_frames = 0
        self._current_frame = 0
        self._empty_plot()

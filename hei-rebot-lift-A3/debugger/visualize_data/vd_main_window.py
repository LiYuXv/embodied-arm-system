# -*- coding: utf-8 -*-
"""主窗口：数据集列表 / Episode 列表 / 3路图像回放 / 双臂轨迹与底盘曲线 / 标注 的整合界面。"""
from vd_common import _DEFAULT_DATA_ROOT, _SDK_ROOT, logger, quat_to_rpy_recording

import os
import json
import time
import math
import shutil
from collections import OrderedDict
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QSplitter, QMessageBox, QHeaderView, QTableWidget, QTableWidgetItem, QGroupBox, QComboBox, QDoubleSpinBox, QCheckBox, QProgressBar, QStatusBar, QScrollArea, QLineEdit, QSizePolicy, QDialog
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QBrush, QColor, QKeySequence, QShortcut

from auto_language_annotate import LanguageAnnotationStore, LabelLibrary, write_language_to_dataset
from annotation_store import (
    AnnotationStore as RewardAnnotationStore,
    LABEL_COLORS as RECAP_LABEL_COLORS,
    INTERVENTION_COLOR as RECAP_INTERVENTION_COLOR,
    has_annotations as recap_has_annotations,
    backup_annotations as recap_backup_annotations,
    remap_intervals_after_trim,
)

from vd_dataset import OfflineDataset
from vd_widgets import FullscreenImageDialog, ImageLabel, RecapTrackWidget, TrimSlider
from vd_trajectory import TrajectoryPlotWidget, ChassisPlotWidget, detect_jitter_intervals, detect_stationary_intervals
from vd_data_ops import _assert_dataset_consistent, _atomic_write_parquet, _episode_task_display_name, _gc_orphan_tasks_df, _recompute_global_stats, _remove_files, _smooth_pose_matrix, _sync_episode_data_pointers, _write_parquet_chunks, refix_action_labels
from vd_threads import BatchDeleteEpisodesThread, DeleteEpisodeThread, DenoiseThread, PlaybackThread, SidecarExportThread, TrimEpisodeThread, TrimStationaryThread, _VlmLabelThread
from vd_dialogs import BatchDeleteDialog, DatasetCheckDialog, DatasetStatsDialog, AdvantagePlotDialog


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self, root=_DEFAULT_DATA_ROOT):
        super().__init__()
        self.root = root
        self.dataset = None
        self.playback_thread = None
        self._current_total_frames = 0
        self._current_frame_idx = 0

        # 任务编辑状态
        self._current_episode_idx = -1
        self._original_task_name = ""
        self._task_modified = False
        self._task_edit_connected = False
        self._task_edit_in_progress = False  # 防止 editingFinished 重复触发导致对话框多次弹出
        self._original_group_label = ""      # 组名编辑框的当前已保存值（π*0.6 任务组）

        # 批量剪辑静止状态
        self._trim_all_in_progress = False
        self._trim_all_queue = []
        self._trim_all_current_ep = -1
        self._trim_all_total = 0
        self._trim_all_trimmed = 0
        self._trim_all_skipped = 0
        self._trim_all_failed = 0
        self._trim_all_backup_enabled = False
        self._trim_all_mode = "stationary"  # "stationary" | "jitter"（批量剪辑类型）

        # RECAP 奖励标注状态
        self._reward_store = None            # RewardAnnotationStore 实例
        self._recap_current_label = "good"   # 当前选中的标签
        self._recap_start_mark = None        # 区间起点（帧号）
        self._recap_edit_index = None        # 正在编辑的区间下标（None=添加模式）
        self._recap_export_thread = None     # sidecar 展开后台线程
        self._recap_trim_pending_ranges = {}   # 剪辑成功后待重映射的 {episode_idx: 删除区间或None}（P0.3 方案乙）
        self._pending_action_refix = False     # 抖动剪辑（运动段）成功后需重算 action 标签

        # 批量删除状态
        self.batch_delete_thread = None      # BatchDeleteEpisodesThread 实例
        self._pending_episode_indices = []   # 正在批量删除的 episode 编号

        # 首帧预览图 LRU 缓存：(episode_idx, camera_name) -> np.ndarray
        # 切换 episode 时免去重复的视频打开/解码（~30ms/路）；数据集变更时清空
        self._preview_frame_cache = OrderedDict()
        self._PREVIEW_CACHE_MAX = 16

        # 滑条拖动节流：用 QTimer 合并连续的 position_changed 事件，
        # 拖动时最多 ~33fps 解码，避免每个像素移动都触发视频解码
        self._scrub_target_frame = 0
        self._scrub_timer = QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(30)   # 30ms ≈ 33fps
        self._scrub_timer.timeout.connect(self._on_scrub_timeout)

        self.setWindowTitle("LeRobot 数据可视化工具")
        self.setGeometry(80, 80, 1700, 950)
        
        self._init_ui()
        self._scan_datasets()
        
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # ===== 左侧面板 =====
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 数据集选择
        dataset_group = QGroupBox("数据集选择")
        dataset_layout = QVBoxLayout(dataset_group)
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_selected)
        dataset_layout.addWidget(self.dataset_combo)
        
        refresh_btn = QPushButton("🔄 刷新列表")
        refresh_btn.clicked.connect(self._scan_datasets)
        dataset_layout.addWidget(refresh_btn)
        
        left_layout.addWidget(dataset_group)
        
        # 数据集总信息
        info_group = QGroupBox("数据集总信息")
        info_layout = QVBoxLayout(info_group)
        
        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setMaximumHeight(480)
        
        self.info_label = QLabel("未加载数据集")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }")
        info_scroll.setWidget(self.info_label)
        info_layout.addWidget(info_scroll)
        
        left_layout.addWidget(info_group)
        
        # Episode 列表
        episode_group = QGroupBox("Episode 列表")
        episode_layout = QVBoxLayout(episode_group)
        
        self.episode_table = QTableWidget()
        self.episode_table.setColumnCount(7)
        self.episode_table.setHorizontalHeaderLabels(
            ["编号", "帧数", "时长(秒)", "分辨率", "任务", "策略", "组"])
        # 交互式列宽（可拖动）：列总宽超出可视区时自动出现横向滚动条，
        # 不再整体 Stretch 挤压——策略/组等长文本列可完整显示
        _ep_header = self.episode_table.horizontalHeader()
        _ep_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        _ep_header.setStretchLastSection(False)
        for _col, _w in enumerate([52, 62, 74, 92, 170, 150, 130]):
            self.episode_table.setColumnWidth(_col, _w)
        self.episode_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.episode_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.episode_table.itemSelectionChanged.connect(self._on_episode_selected)
        self.episode_table.setMinimumHeight(150)
        self.episode_table.verticalHeader().setDefaultSectionSize(28)
        episode_layout.addWidget(self.episode_table)
        
        self.episode_stats_label = QLabel("请选择数据集")
        self.episode_stats_label.setStyleSheet("QLabel { color: #666; font-size: 12px; }")
        episode_layout.addWidget(self.episode_stats_label)

        # 数据集统计 / 检查 按钮（水平并排）
        ds_btn_layout = QHBoxLayout()
        ds_btn_layout.setSpacing(6)

        # 数据集统计分析按钮
        self.stats_btn = QPushButton("📊 数据集统计分析")
        self.stats_btn.setToolTip("分析所有 Episode 的位置/欧拉角/夹爪角度分布，发现异常值与极值（横轴为 Episode 编号）")
        self.stats_btn.setStyleSheet(
            "QPushButton { background-color: #00838F; color: white; font-weight: bold; padding: 7px; }"
            "QPushButton:hover { background-color: #006064; }"
            "QPushButton:disabled { background-color: #ccc; color: #888; }"
        )
        self.stats_btn.clicked.connect(self._show_dataset_stats)
        self.stats_btn.setEnabled(False)
        ds_btn_layout.addWidget(self.stats_btn)

        # 数据集检查按钮（检查 7 类问题，判断能否训练）
        self.check_btn = QPushButton("🔍 数据集检查")
        self.check_btn.setToolTip("检查数据集是否存在 数据问题分析.md 的 7 类问题（data映射/视频/帧索引/stats一致性/恒等映射/meta与data行一致性/task_index连续性），给出能否训练结论")
        self.check_btn.setStyleSheet(
            "QPushButton { background-color: #558B2F; color: white; font-weight: bold; padding: 7px; }"
            "QPushButton:hover { background-color: #33691E; }"
            "QPushButton:disabled { background-color: #ccc; color: #888; }"
        )
        self.check_btn.clicked.connect(self._show_dataset_check)
        self.check_btn.setEnabled(False)
        ds_btn_layout.addWidget(self.check_btn)

        episode_layout.addLayout(ds_btn_layout)

        left_layout.addWidget(episode_group, stretch=4)

        # 任务名称编辑
        task_group = QGroupBox("Episode 任务")
        task_layout = QVBoxLayout(task_group)

        self.task_name_edit = QLineEdit()
        self.task_name_edit.setPlaceholderText("选择 Episode 后显示任务名称")
        self.task_name_edit.setEnabled(False)
        # 允许中文、英文、数字、下划线、空格、连字符等常见任务名称字符
        from PyQt6.QtGui import QRegularExpressionValidator
        from PyQt6.QtCore import QRegularExpression
        validator = QRegularExpressionValidator(
            QRegularExpression(r"^[\w\x{4e00}-\x{9fa5}][\w\x{4e00}-\x{9fa5}\s\-]*$"), self.task_name_edit
        )
        self.task_name_edit.setValidator(validator)
        task_row = QHBoxLayout()
        task_row.addWidget(self.task_name_edit, 1)
        self.task_batch_btn = QPushButton("批量修改")
        self.task_batch_btn.setToolTip("批量替换任务名称：将所有使用该任务名的 Episode 一次性改为新名称")
        self.task_batch_btn.clicked.connect(self._show_task_batch_dialog)
        task_row.addWidget(self.task_batch_btn)
        task_layout.addLayout(task_row)

        # 组名编辑（π*0.6 任务组元数据）：显示/修改当前 Episode 的组名标签段，
        # 置空 = 取消分组；保存时自动保留 group_id 的策略段
        group_row = QHBoxLayout()
        self.group_name_edit = QLineEdit()
        self.group_name_edit.setPlaceholderText("选择 Episode 后显示组名（可改；置空=取消分组）")
        self.group_name_edit.setEnabled(False)
        self.group_name_edit.setToolTip(
            "任务组名（π*0.6 组相对优势元数据，不进模型）：\n"
            "- 显示 group_id 的组名段（策略段自动保留）\n"
            "- 回车确认保存；置空 = 取消该集分组\n"
            "- 批量改组名用右侧按钮（同组 Episode 一次性替换）")
        self.group_name_edit.editingFinished.connect(self._on_group_editing_finished)
        group_row.addWidget(self.group_name_edit, 1)
        self.group_batch_btn = QPushButton("批量修改组名")
        self.group_batch_btn.setToolTip("把某个组名的所有 Episode 一次性改为新组名（保留各自策略段）")
        self.group_batch_btn.clicked.connect(self._show_group_batch_dialog)
        group_row.addWidget(self.group_batch_btn)
        task_layout.addLayout(group_row)

        self.task_status_label = QLabel("")
        self.task_status_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        task_layout.addWidget(self.task_status_label)

        left_layout.addWidget(task_group)

        # 播放控制
        control_group = QGroupBox("播放控制")
        control_layout = QVBoxLayout(control_group)
        
        self.current_episode_label = QLabel("当前未选择 Episode")
        self.current_episode_label.setStyleSheet("QLabel { font-weight: bold; color: #2196F3; }")
        control_layout.addWidget(self.current_episode_label)
        
        # 可剪辑进度条（三标记：起止 + 播放头）
        self.trim_slider = TrimSlider()
        self.trim_slider.setEnabled(False)
        self.trim_slider.setToolTip(
            "拖动蓝色/紫色三角选择片段；双击滑条可将右侧位置曲线缩放到所选片段"
        )
        self.trim_slider.markers_changed.connect(self._on_trim_markers_changed)
        self.trim_slider.position_changed.connect(self._on_trim_position_changed)
        self.trim_slider.position_clicked.connect(self._on_trim_position_clicked)
        self.trim_slider.segment_selected.connect(self._on_trim_segment_selected)
        control_layout.addWidget(self.trim_slider)

        # RECAP 奖励标注轨道（显示 good/bad/neutral 彩色带）+ 显示开关
        recap_track_row = QHBoxLayout()
        self.recap_track = RecapTrackWidget()
        self.recap_track.setVisible(True)
        self.recap_show_check = QCheckBox("显示 RECAP 标记")
        self.recap_show_check.setChecked(True)
        self.recap_show_check.setToolTip("在播放时间线下方显示 good(绿)/bad(红)/neutral(橙) 彩色标注带")
        self.recap_show_check.toggled.connect(self._on_recap_track_toggled)
        recap_track_row.addWidget(self.recap_show_check)
        control_layout.addLayout(recap_track_row)
        control_layout.addWidget(self.recap_track)
        
        # 进度标签
        self.progress_label = QLabel("0 / 0")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("QLabel { font-size: 12px; color: #666; }")
        control_layout.addWidget(self.progress_label)

        # 剪辑区域信息
        self.trim_info_label = QLabel("剪辑区域: 未选择")
        self.trim_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trim_info_label.setStyleSheet("QLabel { font-size: 12px; color: #FF5722; font-weight: bold; }")
        self.trim_info_label.setToolTip("拖动下方蓝色/紫色三角标记选择要删除的片段")
        control_layout.addWidget(self.trim_info_label)

        # 通用进度条（用于删除/剪辑等耗时操作）
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setEnabled(False)
        control_layout.addWidget(self.progress_bar)
        
        btn_layout = QHBoxLayout()

        # 三独立按键：重新播放(从头) / 继续(从滑条当前位置) / 暂停。
        # 继续与暂停不受"重新播放"影响——滑条拖/点到任意位置后点"继续"即从该帧播放
        self.play_btn = QPushButton("🔄 重新播放")
        self.play_btn.setToolTip("从当前 Episode 的第 0 帧重新播放（自动连播时从头连播）")
        self.play_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )
        self.play_btn.clicked.connect(self._on_replay_clicked)
        btn_layout.addWidget(self.play_btn)

        self.resume_btn = QPushButton("▶ 继续")
        self.resume_btn.setToolTip(
            "从播放条当前位置继续播放：\n"
            "- 播放中点=从当前滑条位置继续\n"
            "- 暂停/停止后点=从滑条所在帧起播\n"
            "- RECAP 区间表选中行会自动定位到片段开头，点本键即播放该片段")
        self.resume_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }"
        )
        self.resume_btn.clicked.connect(self._on_continue_clicked)
        btn_layout.addWidget(self.resume_btn)

        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.setToolTip("暂停当前播放（画面停在当前帧；之后可拖滑条再点'继续'）")
        self.pause_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }"
        )
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        btn_layout.addWidget(self.pause_btn)

        # 删除按钮
        self.del_btn = QPushButton("🗑 删除片段")
        self.del_btn.setToolTip("删除选中的 Episode（永久删除，不可恢复）")
        self.del_btn.setStyleSheet("""
            QPushButton { 
                background-color: #9C27B0; 
                color: white; 
                font-weight: bold; 
                padding: 8px;
                border: 2px solid #7B1FA2;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #7B1FA2; }
            QPushButton:disabled { background-color: #ccc; color: #888; border: 2px solid #bbb; }
        """)
        self.del_btn.clicked.connect(self._delete_selected_episode)
        self.del_btn.setEnabled(False)
        btn_layout.addWidget(self.del_btn)

        # 批量删除按钮
        self.batch_del_btn = QPushButton("🗑 批量删除")
        self.batch_del_btn.setToolTip(
            "弹出对话框批量删除 Episode：可手动多选，也可按任务名筛选后一键勾选（永久删除，不可恢复）")
        self.batch_del_btn.setStyleSheet("""
            QPushButton {
                background-color: #6A1B9A;
                color: white;
                font-weight: bold;
                padding: 8px;
                border: 2px solid #4A148C;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #4A148C; }
            QPushButton:disabled { background-color: #ccc; color: #888; border: 2px solid #bbb; }
        """)
        self.batch_del_btn.clicked.connect(self._batch_delete_episodes)
        self.batch_del_btn.setEnabled(False)
        btn_layout.addWidget(self.batch_del_btn)

        # 剪辑按钮
        self.trim_btn = QPushButton("✂ 剪辑")
        self.trim_btn.setToolTip("删除两个标记之间的片段")
        self.trim_btn.setStyleSheet("""
            QPushButton { 
                background-color: #FF5722; 
                color: white; 
                font-weight: bold; 
                padding: 8px;
                border: 2px solid #E64A19;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #E64A19; }
            QPushButton:disabled { background-color: #ccc; color: #888; border: 2px solid #bbb; }
        """)
        self.trim_btn.clicked.connect(self._trim_selected_segment)
        self.trim_btn.setEnabled(False)
        btn_layout.addWidget(self.trim_btn)
        
        control_layout.addLayout(btn_layout)
        
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("播放速度:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 5.0)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setDecimals(1)
        speed_layout.addWidget(self.speed_spin)
        speed_layout.addWidget(QLabel("x"))
        speed_layout.addStretch()
        control_layout.addLayout(speed_layout)
        
        self.auto_play_check = QCheckBox("自动播放所有 Episode")
        control_layout.addWidget(self.auto_play_check)

        self.backup_before_trim_check = QCheckBox("剪辑前自动备份原数据")
        self.backup_before_trim_check.setChecked(False)
        self.backup_before_trim_check.setToolTip(
            "勾选后会在剪辑前生成 data.bak_*/meta.bak_*/videos.bak_* 备份目录；"
            "取消勾选则不再生成备份，请确保已自行备份重要数据。"
        )
        control_layout.addWidget(self.backup_before_trim_check)
        
        left_layout.addWidget(control_group)
        left_layout.addStretch()
        
        splitter.addWidget(left_panel)
        
        # ===== 中间面板: 图像回放（3 路摄像头） =====
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.setContentsMargins(0, 0, 0, 0)

        # 图像回放
        image_group = QGroupBox("图像回放")
        image_layout = QVBoxLayout(image_group)

        self.cam1_label = ImageLabel("摄像头 1 (臂1腕部)")
        self.cam1_label.setStyleSheet("background-color: #222; color: #fff; font-size: 12px; border: 1px solid #444;")
        self.cam1_label.setMinimumSize(320, 180)
        self.cam1_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        image_layout.addWidget(self.cam1_label)

        self.cam2_label = ImageLabel("摄像头 2 (头部)")
        self.cam2_label.setStyleSheet("background-color: #222; color: #fff; font-size: 12px; border: 1px solid #444;")
        self.cam2_label.setMinimumSize(320, 180)
        self.cam2_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        image_layout.addWidget(self.cam2_label)

        self.cam3_label = ImageLabel("摄像头 3 (臂2腕部)")
        self.cam3_label.setStyleSheet("background-color: #222; color: #fff; font-size: 12px; border: 1px solid #444;")
        self.cam3_label.setMinimumSize(320, 180)
        self.cam3_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        image_layout.addWidget(self.cam3_label)

        # 双击图像弹出全分辨率查看窗口
        self.cam1_label.doubleClicked.connect(
            lambda: self._on_image_double_clicked("observation.images.cam1")
        )
        self.cam2_label.doubleClicked.connect(
            lambda: self._on_image_double_clicked("observation.images.cam2")
        )
        self.cam3_label.doubleClicked.connect(
            lambda: self._on_image_double_clicked("observation.images.cam3")
        )

        self.frame_info_label = QLabel("帧信息: 未开始播放")
        self.frame_info_label.setStyleSheet("QLabel { color: #666; font-size: 11px; padding: 5px; }")
        self.frame_info_label.setWordWrap(True)
        self.frame_info_label.setMinimumHeight(80)
        image_layout.addWidget(self.frame_info_label)
        middle_layout.addWidget(image_group, stretch=1)

        splitter.addWidget(middle_panel)

        # ===== 右侧面板: 位置曲线 =====
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Episode 末端位姿变化曲线（移至右侧）
        trajectory_group = QGroupBox("位置曲线")
        trajectory_layout = QVBoxLayout(trajectory_group)
        self.trajectory_plot = TrajectoryPlotWidget()
        self.trajectory_plot.trim_stationary_requested.connect(self._trim_stationary_segments)
        # 后期处理分页：剪辑抖动（运动段剪辑 → 剪辑后重算 action 标签）
        self.trajectory_plot.trim_jitter_requested.connect(self._trim_jitter_segments)
        self.trajectory_plot.trim_all_stationary_requested.connect(self._trim_all_stationary_segments)
        # 后期处理分页：批量剪辑抖动（选集 → 逐集剪辑 → 最后统一 refix + sidecar）
        self.trajectory_plot.trim_all_jitter_requested.connect(self._trim_all_jitter_segments)
        self.trajectory_plot.denoise_requested.connect(self._denoise_current_episode)
        self.trajectory_plot.denoise_all_requested.connect(self._denoise_all_episodes)
        self.trajectory_plot.preview_denoise_requested.connect(self._preview_denoise)
        self.trajectory_plot.subtask_apply_requested.connect(self._apply_subtask_annotations)
        self.trajectory_plot.vlm_label_requested.connect(self._on_vlm_label_requested)
        # 数据集合并分页：设置数据根目录，合并完成后刷新数据集列表
        self.trajectory_plot.set_merge_root(self.root)
        self.trajectory_plot.merge_completed.connect(self._on_merge_completed)
        # RECAP 标注：信号连接（控件在 trajectory_plot 上，逻辑在 MainWindow）
        tp = self.trajectory_plot
        tp.recap_label_combo.currentTextChanged.connect(self._on_recap_label_changed)
        tp.recap_mark_start_btn.clicked.connect(self._recap_mark_start)
        tp.recap_mark_end_btn.clicked.connect(self._recap_mark_end)
        tp.recap_add_btn.clicked.connect(self._recap_add_interval)
        tp.recap_quick_good.clicked.connect(lambda: self._recap_quick_mark("good"))
        tp.recap_quick_bad.clicked.connect(lambda: self._recap_quick_mark("bad"))
        tp.recap_quick_neutral.clicked.connect(lambda: self._recap_quick_mark("neutral"))
        tp.recap_ep_good.clicked.connect(lambda: self._recap_mark_whole_episode("good"))
        tp.recap_ep_bad.clicked.connect(lambda: self._recap_mark_whole_episode("bad"))
        tp.recap_del_btn.clicked.connect(self._recap_delete_selected)
        tp.recap_undo_btn.clicked.connect(self._recap_undo_last)
        tp.recap_clear_btn.clicked.connect(self._recap_clear_episode)
        tp.recap_save_btn.clicked.connect(self._recap_save)
        tp.recap_export_btn.clicked.connect(self._recap_export_sidecar)
        tp.recap_adv_btn.clicked.connect(self._recap_show_advantage)
        # 双击编辑：区间表行 / 轨道色条 → 进入编辑模式；取消按钮退出
        tp.recap_cancel_edit_btn.clicked.connect(self._recap_cancel_edit)
        tp.recap_interval_table.cellDoubleClicked.connect(self._on_recap_table_double_clicked)
        # 选中区间行 → 播放条跳到该片段起始帧并进入暂停待续播（点"继续"播放）
        tp.recap_interval_table.itemSelectionChanged.connect(
            self._on_recap_interval_selected_playback)
        self.recap_track.interval_double_clicked.connect(self._recap_begin_edit)
        # RECAP episode 级：成败为派生只读（标注粒度决策，不人工标注）；已审核可勾选
        tp.recap_reviewed_check.toggled.connect(self._recap_set_reviewed)
        # RECAP 键盘快捷键：播放中 G=好 / B=坏 / N=中（以最近 2 秒为区间）
        for _key, _label in (("G", "good"), ("B", "bad"), ("N", "neutral")):
            _sc = QShortcut(QKeySequence(_key), self)
            _sc.setContext(Qt.ShortcutContext.WindowShortcut)
            _sc.activated.connect(lambda l=_label: self._recap_quick_mark_recent(l))
        # VLM 标签后台线程（模型只加载一次，之后复用）
        _vlm_model = str(_SDK_ROOT / "models" / "Qwen" / "Qwen3-VL-4B-Instruct")
        self._vlm_thread = _VlmLabelThread(_vlm_model, self)
        self._vlm_thread.labels_ready.connect(self._on_vlm_labels_ready)
        self._vlm_thread.load_failed.connect(self._on_vlm_load_failed)
        trajectory_layout.addWidget(self.trajectory_plot)
        right_layout.addWidget(trajectory_group, stretch=5)

        # 底盘速度 / 升降台位置曲线（新增，双臂+底盘数据集）
        chassis_group = QGroupBox("底盘 / 升降曲线")
        chassis_layout = QVBoxLayout(chassis_group)
        self.chassis_plot = ChassisPlotWidget()
        chassis_layout.addWidget(self.chassis_plot)
        right_layout.addWidget(chassis_group, stretch=2)

        # 操作说明按钮（点击弹出对话框，避免占用右侧空间）
        self.help_btn = QPushButton("📖 操作说明")
        self.help_btn.setStyleSheet(
            "QPushButton { font-size: 12px; padding: 6px; }"
        )
        self.help_btn.clicked.connect(self._show_help_dialog)
        right_layout.addWidget(self.help_btn)
        
        right_layout.addStretch()
        
        splitter.addWidget(right_panel)
        splitter.setSizes([460, 720, 520])
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请选择数据集")
        
    def _scan_datasets(self):
        self.dataset_combo.clear()
        self.dataset_combo.addItem("-- 请选择数据集 --", None)
        
        if not os.path.exists(self.root):
            self.status_bar.showMessage(f"数据目录不存在: {self.root}")
            return
            
        found = False
        for name in sorted(os.listdir(self.root)):
            path = os.path.join(self.root, name)
            if os.path.isdir(path):
                has_meta = os.path.exists(os.path.join(path, "meta", "info.json"))
                if has_meta:
                    self.dataset_combo.addItem(name, path)
                    found = True
                    
        if not found:
            self.status_bar.showMessage("未找到任何数据集")
        else:
            count = self.dataset_combo.count() - 1
            self.status_bar.showMessage(f"找到 {count} 个数据集")

    def _on_merge_completed(self, target_name):
        """数据集合并完成：刷新数据集列表，并加载合并目标数据集。

        若目标正是当前已加载的数据集（覆盖合并），下拉框索引不变不会触发
        currentIndexChanged，需要手动调用 _on_dataset_selected 强制重新加载。
        """
        current_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        self._scan_datasets()
        for i in range(1, self.dataset_combo.count()):
            if self.dataset_combo.itemText(i) == target_name:
                same_as_current = self.dataset_combo.itemData(i) == current_path
                self.dataset_combo.setCurrentIndex(i)
                if same_as_current:
                    self._on_dataset_selected(i)
                break
        self.status_bar.showMessage(f"数据集合并完成: {target_name}")

    def _show_help_dialog(self):
        """弹出操作说明对话框。"""
        dialog = QDialog(self)
        dialog.setWindowTitle("操作说明")
        dialog.resize(500, 600)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)

        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)

        help_text = QLabel(
            "<b>基本操作:</b><br>"
            "1. 从下拉菜单选择数据集<br>"
            "2. 在 Episode 列表中单击选择<br>"
            "3. 点击 <b>▶ 播放</b> 开始回放<br><br>"
            "<b>播放控制:</b><br>"
            "• <b>▶ 播放</b> - 播放选中的 Episode<br>"
            "• <b>⏸ 暂停</b> - 暂停当前播放（画面停在当前帧）<br>"
            "• <b>▶ 继续</b> - 从播放条当前位置继续播放（拖滑条到任意帧后点此键）<br>"
            "• <b>🔄 重新播放</b> - 从头重新播放<br>"
            "<br>"
            "<b>图像检查:</b><br>"
            "• <b>双击</b>右侧任意摄像头图像，可弹出该摄像头当前帧的<b>全分辨率图像</b><br><br>"
            "<b>位置曲线:</b><br>"
            "• 选择 Episode 后，右侧会显示该 Episode 末端位置 XYZ 随帧序号变化的曲线<br>"
            "• 播放或拖动进度条时，黑色竖线会同步指示当前帧位置<br>"
            "• <b>双击</b>进度条（剪辑滑条），可将曲线缩放到两个三角标记之间的片段<br>"
            "• 使用曲线下方的 X缩放 / Y缩放 滑条进一步放大检查细节，点击 重置 恢复全段视图<br><br>"
            "<b>其他功能:</b><br>"
            "• 双击 Episode 行可直接播放<br>"
            "• 勾选'自动播放所有'可连续播放<br>"
            "• 调整速度滑块改变回放速度<br><br>"
            "<b>视频剪辑:</b><br>"
            "• 拖动进度条下方蓝色/紫色三角选择要删除的片段<br>"
            "• 绿色播放头可拖动，实时显示对应帧画面与数据<br>"
            "• 点击轨道空白处可跳转当前帧<br>"
            "• 播放中拖动/点击播放头会自动暂停并定位到该帧<br>"
            "• 点击 ✂ 剪辑按钮删除两个三角标记之间的片段<br><br>"
            "<b>智能降噪:</b><br>"
            "• 在右侧'智能降噪'面板选择滤波方法（推荐 Savitzky-Golay）与平滑窗口<br>"
            "• 点击 🧪 预览降噪 可在曲线上叠加虚线对比效果（不修改数据）<br>"
            "• 点击 ✨ 智能降噪 平滑当前 Episode，✨ 批量智能降噪 处理全部 Episode<br>"
            "• 只修改 state/action 数值，帧数与元数据不变，处理后自动重算 stats.json"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("QLabel { line-height: 1.5; }")
        help_text.setTextFormat(Qt.TextFormat.RichText)
        help_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        help_text.setOpenExternalLinks(True)
        content_layout.addWidget(help_text)
        content_layout.addStretch()

        scroll.setWidget(content)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dialog.exec()
            
    def _show_dataset_stats(self):
        """弹出数据集统计分布分析对话框，分析所有 Episode 的位姿分布。"""
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return
        if len(self.dataset.episodes) == 0:
            QMessageBox.warning(self, "警告", "当前数据集没有 Episode")
            return
        try:
            self.status_bar.showMessage("正在分析数据集分布...")
            QApplication.processEvents()
            dialog = DatasetStatsDialog(self, self.dataset)
            self.status_bar.showMessage("统计分析完成")
            dialog.exec()
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "错误", f"统计分析失败:\n{str(e)}\n\n{traceback.format_exc()}")
            self.status_bar.showMessage("统计分析失败")

    def _show_dataset_check(self):
        """弹出数据集 7 类问题检查报告，判断能否用于 VLA 训练。"""
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return
        if len(self.dataset.episodes) == 0:
            QMessageBox.warning(self, "警告", "当前数据集没有 Episode")
            return
        # 防误触：按钮变"检查中"并禁用，检查完成/异常后恢复
        self.check_btn.setEnabled(False)
        self.check_btn.setText("⏳ 检查中...")
        QApplication.processEvents()
        try:
            self.status_bar.showMessage("正在检查数据集...")
            QApplication.processEvents()
            dialog = DatasetCheckDialog(self, self.dataset)
            self.status_bar.showMessage("数据集检查完成")
            dialog.exec()
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "错误", f"数据集检查失败:\n{str(e)}\n\n{traceback.format_exc()}")
            self.status_bar.showMessage("数据集检查失败")
        finally:
            self.check_btn.setText("🔍 数据集检查")
            self.check_btn.setEnabled(len(self.dataset.episodes) > 0)

    def _on_dataset_selected(self, index):
        if index <= 0:
            self.episode_table.setRowCount(0)
            self.info_label.setText("未加载数据集")
            self.episode_stats_label.setText("请选择数据集")
            self.current_episode_label.setText("当前未选择 Episode")
            self.del_btn.setEnabled(False)
            self.batch_del_btn.setEnabled(False)
            self.trim_btn.setEnabled(False)
            self.stats_btn.setEnabled(False)
            self.check_btn.setEnabled(False)
            self.trim_slider.set_total_frames(0)
            self.dataset = None
            self._preview_frame_cache.clear()

            # 清空任务编辑和位置曲线
            self._reset_task_editor()
            self.trajectory_plot.clear()
            return
            
        dataset_path = self.dataset_combo.itemData(index)
        if dataset_path is None:
            return
            
        try:
            self.status_bar.showMessage(f"正在加载数据集...")
            QApplication.processEvents()
            
            self.dataset = OfflineDataset(dataset_path)
            self._preview_frame_cache.clear()  # 新数据集：旧缓存全部失效

            # 初始化 RECAP 奖励标注存储（读取/创建 reward_annotations.json）
            self._recap_exit_edit_mode()  # 切换数据集：退出编辑模式并恢复按钮状态
            try:
                self._reward_store = RewardAnnotationStore(str(dataset_path))
                self._reward_store.load()
            except Exception:
                self._reward_store = None
            self._update_recap_status()

            self._update_episode_table()
            self._update_info()
            self.stats_btn.setEnabled(len(self.dataset.episodes) > 0)
            self.check_btn.setEnabled(len(self.dataset.episodes) > 0)
            self.batch_del_btn.setEnabled(len(self.dataset.episodes) > 0)
            
            self.status_bar.showMessage(
                f"数据集加载完成: {len(self.dataset.episodes)} 个 episode, "
                f"{self.dataset.num_frames} 帧"
            )
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "错误", f"加载数据集失败:\n{str(e)}\n\n{traceback.format_exc()}")
            self.status_bar.showMessage("数据集加载失败")
            
    def _update_episode_table(self):
        if self.dataset is None:
            return
            
        episodes = self.dataset.episodes
        self.episode_table.setRowCount(len(episodes))
        
        # 获取所有图像分辨率，按 camera 名称排序
        resolutions = self.dataset.get_all_image_resolutions()
        resolution_text = ""
        if resolutions:
            resolution_text = ", ".join(
                f"{w}x{h}" for _, (w, h) in sorted(resolutions.items())
            )
        else:
            # 如果 info.json 中没有 shape，尝试读取第一个 episode 的第一帧图像或视频帧
            try:
                camera_names = [name for name, f in self.dataset.features.items()
                                if f.get("dtype") in ["image", "video"] or f.get("type") in ["image", "video"] or "image" in name.lower()]
                for cam in camera_names:
                    if self.dataset.is_video_feature(cam):
                        frame = self.dataset.get_video_frame(
                            self.dataset.episodes[0].get("episode_index", 0), 0, cam
                        )
                        if frame is not None and isinstance(frame, np.ndarray):
                            h, w = frame.shape[:2]
                            resolution_text = f"{w}x{h}"
                            break
                    else:
                        img_path = self.dataset.get_image_path(
                            self.dataset.episodes[0].get("episode_index", 0), 0, cam
                        )
                        if img_path:
                            img = QImage(img_path)
                            if not img.isNull():
                                resolution_text = f"{img.width()}x{img.height()}"
                                break
            except Exception:
                resolution_text = ""
        
        for i, ep in enumerate(episodes):
            length = ep.get("length", ep.get("dataset_to_index", 0) - ep.get("dataset_from_index", 0))
            duration = length / self.dataset.fps if self.dataset.fps > 0 else 0
            
            self.episode_table.setItem(i, 0, QTableWidgetItem(str(ep.get("episode_index", i))))
            self.episode_table.setItem(i, 1, QTableWidgetItem(str(length)))
            self.episode_table.setItem(i, 2, QTableWidgetItem(f"{duration:.1f}"))
            self.episode_table.setItem(i, 3, QTableWidgetItem(resolution_text))
            
            # LeRobot 0.4.4 episodes 中存储的是 tasks 字段（任务名称数组）
            task_name = ""
            if "tasks" in ep and ep["tasks"] is not None:
                # tasks 可能是 numpy array 或 list
                tasks = ep["tasks"]
                if hasattr(tasks, '__iter__') and not isinstance(tasks, str):
                    task_list = list(tasks)
                    if task_list:
                        task_name = str(task_list[0])
            elif "task_index" in ep:
                # 旧格式：从 task_index 查找（按值查找，不能用列表位置索引）
                task_name = str(self.dataset.task_name_for_index(ep["task_index"]))
            
            self.episode_table.setItem(i, 4, QTableWidgetItem(task_name))

            # 策略/组列（π*0.6 在线 RL 元数据，无标注或未标记时留空）
            policy_txt, group_txt = self._episode_policy_group_text(i)
            self.episode_table.setItem(i, 5, QTableWidgetItem(policy_txt))
            self.episode_table.setItem(i, 6, QTableWidgetItem(group_txt))

        self.episode_stats_label.setText(f"共 {len(episodes)} 个 episode")

    @staticmethod
    def _policy_run_name(path):
        """checkpoint 路径 → run 名（ckpt/xxx/checkpoints/012000 → xxx）。

        与采集端 camera_pyqt._policy_short 同一规则：含 checkpoints 段取其
        上一级目录（训练 run 名）；否则取末个非通用目录段。
        """
        if not path:
            return ""
        raw = [s for s in str(path).replace("\\", "/").split("/") if s]
        if "checkpoints" in raw:
            i = raw.index("checkpoints")
            if i >= 1:
                return raw[i - 1]
        generic = {"", ".", "ckpt", "checkpoints", "pretrained_models", "debugger"}
        segs = [s for s in raw if s not in generic]
        return segs[-1] if segs else str(path)

    def _episode_policy_group_text(self, ep_idx):
        """读取该集的策略 run 名与组名标签段（无标注/无字段/越界返回空串）。

        组名取 group_id 的首段（"做饭-开火 | pi05_xxx" → "做饭-开火"）。
        直读 data dict（不经过 _get_episode）——后者会为未标注集创建空条目，
        表格每刷新一遍就给 store 塞 N 个空 episode，save 时全部落盘污染标注。
        """
        if self._reward_store is None or self.dataset is None:
            return "", ""
        try:
            if ep_idx < 0 or ep_idx >= len(self.dataset.episodes):
                return "", ""
            ep_data = (self._reward_store.data.get("episodes", {}) or {}).get(
                str(int(ep_idx)))
            if not ep_data:
                return "", ""
            pol = ep_data.get("policy_checkpoint")
            grp = ep_data.get("group_id")
            return (self._policy_run_name(pol) if pol else "",
                    str(grp).split(" | ")[0].strip() if grp else "")
        except Exception:
            return "", ""

    def _policy_group_stats_html(self):
        """总信息面板：策略版本与任务组的 episode 统计（无标注/异常返回空串）。"""
        if self._reward_store is None or self.dataset is None:
            return ""
        try:
            from collections import Counter
            n_eps = len(self.dataset.episodes)
            if n_eps == 0:
                return ""
            pol_cnt, grp_cnt = Counter(), Counter()
            for i in range(n_eps):
                p, g = self._episode_policy_group_text(i)
                pol_cnt[p or "专家/未标"] += 1
                grp_cnt[g or "未分组"] += 1
            pol_line = " | ".join(f"{k} ×{v}" for k, v in pol_cnt.most_common(6))
            grp_line = " | ".join(f"{k} ×{v}" for k, v in grp_cnt.most_common(8))
            return (f"<br><b>策略版本:</b> {pol_line}<br>"
                    f"<b>任务组:</b> {grp_line}")
        except Exception:
            return ""
        
    def _update_info(self):
        if self.dataset is None:
            return
            
        info = self.dataset.info
        features = info.get("features", {})
        
        # 统计任务分布
        task_stats = self._compute_task_stats()
        
        info_text = (
            f"<b>数据集:</b> {info.get('repo_id', 'N/A')}<br>"
            f"<b>版本:</b> {info.get('codebase_version', 'N/A')}<br>"
            f"<b>总帧数:</b> {info.get('total_frames', 'N/A')}<br>"
            f"<b>Episode 数:</b> {info.get('total_episodes', 'N/A')}<br>"
            f"<b>FPS:</b> {info.get('fps', 'N/A')}<br>"
            f"<b>特征:</b> {', '.join(features.keys())}"
        )

        # action 标签前瞻（lookahead）估计：扫描 k∈[0,20] 使 mean|action[t]-state[t+k]|
        # 最小的 k 即为观测首目标距离。k=0 意味着 action==state 退化（数据问题分析.md 问题5）
        la = self._estimate_action_lookahead()
        if la is not None:
            best_k, best_s, s0, s1 = la
            if best_k == 0:
                info_text += (f"<br><b>action 前瞻:</b> <span style='color:#c62828'>"
                              f"0 帧（action==state 退化，需修复后训练！）</span>")
            else:
                fps = info.get('fps', 20) or 20
                info_text += (f"<br><b>action 前瞻:</b> {best_k} 帧"
                              f"（首目标领先 {best_k / fps * 1000:.0f}ms；"
                              f"残差 {best_s:.4f}，k=0 时 {s0:.4f}）")
        else:
            info_text += "<br><b>action 前瞻:</b> 无法估计（缺 state/action 列）"

        # 添加任务统计
        if task_stats:
            info_text += "<br><b>任务分布:</b><br>"
            for task_name, frame_count, frame_ratio, ep_count, ep_ratio in task_stats:
                info_text += f"  • {task_name}: {frame_count} 帧 ({frame_ratio:.1%}), {ep_count} 个 episode ({ep_ratio:.1%})<br>"

        # tasks.parquet 健康检测（孤儿任务/编号空洞/total_tasks 不一致等）
        task_problems, task_summary = self._check_tasks_health()
        if task_summary:
            n_tasks = len(task_summary)
            info_text += f"<br><b>tasks.parquet:</b> {n_tasks} 个任务"
            if task_problems:
                info_text += (f" <span style='color:#c62828'>⚠ "
                              + "；".join(task_problems) + "</span>")
            else:
                info_text += " <span style='color:#2e7d32'>✓ 正常</span>"
            # 孤儿任务（0 帧引用）单独标红列出
            orphan_names = [name for name, _ti, nf in task_summary if nf == 0]
            if orphan_names:
                info_text += (f"<br>  <span style='color:#c62828'>孤儿任务: "
                              f"{', '.join(orphan_names)}"
                              f"（删除引用它们的 episode 后残留，可用 "
                              f"check_tasks_parquet.py --fix 清理）</span>")

        # 策略版本/任务组统计（π*0.6 在线 RL 元数据，R1a；无标注时跳过）
        pg_html = self._policy_group_stats_html()
        if pg_html:
            info_text += pg_html

        # RECAP 标注概览（与 make_recap_sidecar.py --dry-run 平坦模式统计同格式；
        # 直接用内存 data_df + 标注 store 计算，毫秒级，不重读 parquet）
        recap_html = self._recap_summary_html()
        if recap_html:
            info_text += recap_html

        self.info_label.setText(info_text)

    def _recap_summary_html(self):
        """RECAP 标注概览 HTML 段（总信息面板用）；无标注/失败时返回空串。

        缓存键 = (数据集路径, 总帧数, 标注文件 mtime)：标注保存/剪辑/删除后
        自动失效重算；episode 切换等高频刷新直接命中缓存。
        """
        if self.dataset is None or self._reward_store is None:
            return ""
        try:
            if not self._reward_store.filepath.is_file():
                return ""
            ann_mtime = self._reward_store.filepath.stat().st_mtime
            sc_path = self._reward_store.meta_dir / "recap_sidecar.jsonl"
            sc_mtime = sc_path.stat().st_mtime if sc_path.is_file() else -1.0
            key = (str(self._reward_store.dataset_root),
                   int(self.dataset.num_frames), ann_mtime, sc_mtime)
            if getattr(self, "_recap_summary_cache_key", None) == key:
                return self._recap_summary_cache_html
            from make_recap_sidecar import summarize_from_frames
            lines = summarize_from_frames(
                self._reward_store, self.dataset.data_df,
                dataset_name=self._reward_store.dataset_root.name)
            if not lines:
                return ""
            body = "<br>".join(f"&nbsp;&nbsp;{l}" for l in lines)

            # sidecar 状态：未生成 → dry-run 提示；已生成 → 行数+时间+时效
            if sc_path.is_file():
                n_rows = 0
                with open(sc_path, "rb") as f:
                    for _ in f:
                        n_rows += 1
                ts = datetime.fromtimestamp(sc_mtime).strftime("%m-%d %H:%M")
                if sc_mtime < ann_mtime:
                    sc_line = (f"<span style='color:#c62828'>已生成训练 sidecar：{n_rows} 行"
                               f"（{ts}）⚠ 早于最近标注保存，需重新生成</span>")
                else:
                    sc_line = f"已生成训练 sidecar：{n_rows} 行（{ts}）"
            else:
                sc_line = ("[sidecar] dry-run，未写文件（未生成；"
                           "请在 RECAP 面板点\u201c生成训练 sidecar\u201d）")

            html = (f"<br><b>RECAP 标注概览</b>"
                    f"（同 make_recap_sidecar.py --dry-run）:<br>{body}<br>"
                    f"&nbsp;&nbsp;{sc_line}")
            self._recap_summary_cache_key = key
            self._recap_summary_cache_html = html
            return html
        except Exception as e:
            logger.warning(f"[RECAP] 概览计算失败: {e}")
            return ""
    
    def _estimate_action_lookahead(self, max_k=20, max_samples=3000):
        """估计 action 标签的前瞻帧数（观测首目标距离）。

        对每个候选 k∈[0,max_k]，在 episode 内计算 mean|action[t]−state[t+k]|，
        取最小值对应的 k。录制约定 action[t]=pose[t+N]，故残差最小的 k≈N；
        末帧沿用末位姿会带来少量噪声，采样均值下不影响 argmin。

        返回 (best_k, 残差_min, k=0 残差, k=1 残差)，数据不足返回 None。
        """
        df = getattr(self.dataset, "data_df", None)
        if df is None or "action" not in df.columns or "observation.state" not in df.columns:
            return None
        scores = np.zeros(max_k + 1)
        counts = 0
        for _, g in df.groupby("episode_index"):
            if len(g) < max_k + 10:
                continue
            s = np.stack(g["observation.state"].to_numpy())
            a = np.stack(g["action"].to_numpy())
            idx = np.arange(0, len(g) - max_k, 5)  # 隔 5 帧采样，控制计算量
            if len(idx) == 0:
                continue
            for k in range(max_k + 1):
                scores[k] += np.abs(a[idx] - s[idx + k]).sum()
            counts += len(idx)
            if counts >= max_samples:
                break
        if counts == 0:
            return None
        scores /= counts
        best_k = int(np.argmin(scores))
        return best_k, float(scores[best_k]), float(scores[0]), float(scores[1])

    def _compute_task_stats(self):
        """计算各任务的帧数、比例，以及 Episode 个数和占比"""
        if self.dataset is None or not self.dataset.tasks:
            return []
        
        # 从 data_df 的 task_index 列统计帧数
        if 'task_index' not in self.dataset.data_df.columns:
            return []
        
        task_frame_counts = self.dataset.data_df['task_index'].value_counts().sort_index()
        total_frames = len(self.dataset.data_df)
        
        # 统计每个任务的 Episode 个数
        task_episode_counts = {}
        total_episodes = len(self.dataset.episodes)
        for ep in self.dataset.episodes:
            ep_tasks = ep.get('tasks', [])
            if ep_tasks is not None and hasattr(ep_tasks, '__iter__') and not isinstance(ep_tasks, str):
                task_list = list(ep_tasks)
                if task_list:
                    task_name = str(task_list[0])
                    task_episode_counts[task_name] = task_episode_counts.get(task_name, 0) + 1
        
        stats = []
        for task_idx, frame_count in task_frame_counts.items():
            task_name = self.dataset.task_name_for_index(task_idx, f'task_{task_idx}')
            
            frame_ratio = frame_count / total_frames if total_frames > 0 else 0
            ep_count = task_episode_counts.get(task_name, 0)
            ep_ratio = ep_count / total_episodes if total_episodes > 0 else 0
            
            stats.append((task_name, frame_count, frame_ratio, ep_count, ep_ratio))

        return stats

    def _check_tasks_health(self):
        """检测 tasks.parquet 健康状况。

        返回 (problems, summary)：
        - problems: 问题描述列表（空 = 健康）
        - summary:  [(任务名, task_index, 被引用帧数), ...]，按 task_index 排序

        检查项：存在性 / info.json total_tasks 一致 / task_index 0 基连续 /
        data 引用越界 / 孤儿任务（0 帧引用）/ episodes tasks 名在 tasks.parquet 中。
        """
        if self.dataset is None:
            return [], []
        tasks_path = self.dataset.root / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            return ["tasks.parquet 不存在"], []
        try:
            tdf = pd.read_parquet(tasks_path)
        except Exception as e:
            return [f"tasks.parquet 读取失败: {e}"], []

        problems = []
        n_tasks = len(tdf)
        ti_vals = [int(v) for v in tdf["task_index"]]

        info_tt = self.dataset.info.get("total_tasks")
        if info_tt is not None and int(info_tt) != n_tasks:
            problems.append(f"info.json total_tasks={info_tt} ≠ 实际 {n_tasks}")

        if sorted(ti_vals) != list(range(n_tasks)):
            problems.append(f"task_index 非 0 基连续: {sorted(ti_vals)}")

        # data 帧引用统计
        referenced = {}
        df = getattr(self.dataset, "data_df", None)
        if df is not None and "task_index" in df.columns:
            referenced = {int(k): int(v)
                          for k, v in df["task_index"].value_counts().items()}
            out_of_range = sorted(v for v in referenced if v < 0 or v >= n_tasks)
            if out_of_range:
                problems.append(f"data 引用了不存在的 task_index: {out_of_range}")

        # episodes 的 tasks 名必须都在 tasks.parquet 中
        task_names = {str(name) for name in tdf.index}
        bad_ep_names = set()
        for ep in self.dataset.episodes:
            ep_tasks = ep.get("tasks")
            if ep_tasks is not None and hasattr(ep_tasks, "__iter__") \
                    and not isinstance(ep_tasks, str):
                for name in ep_tasks:
                    if str(name) not in task_names:
                        bad_ep_names.add(str(name))
        if bad_ep_names:
            problems.append(f"episodes 引用了 tasks.parquet 中不存在的任务名: "
                            f"{sorted(bad_ep_names)}")

        summary = [(str(name), int(row["task_index"]), referenced.get(int(row["task_index"]), 0))
                   for name, row in tdf.iterrows()]
        summary.sort(key=lambda x: x[1])
        orphan_count = sum(1 for _, _, nf in summary if nf == 0)
        if orphan_count:
            problems.append(f"{orphan_count} 个孤儿任务（0 帧引用）")

        return problems, summary
        
    def _on_trim_markers_changed(self, start, end):
        """当双标记位置改变时更新剪辑信息。"""
        self._update_trim_info()

    def _on_trim_position_changed(self, frame):
        """播放头拖动时：先暂停播放，滑条位置/进度标签即时更新，
        视频帧解码通过 QTimer 节流（合并连续 mouseMove，避免逐像素解码卡顿）。"""
        if self.playback_thread is not None:
            if not self.playback_thread.is_paused():
                self.playback_thread.pause()
                import time
                for _ in range(20):
                    if self.playback_thread.is_paused():
                        break
                    time.sleep(0.01)
                    QApplication.processEvents()
        # 记录目标帧（滑条位置 = "继续"按键的起播帧，同步 _current_frame_idx）
        self._current_frame_idx = max(0, int(frame))
        self._scrub_target_frame = frame
        self.trim_slider.set_current_frame(frame)
        if self._current_total_frames > 0:
            self.progress_label.setText(f"{frame + 1} / {self._current_total_frames}")
        # 启动/重启节流定时器：30ms 内若有新事件则覆盖，到期才真正解码
        self._scrub_timer.start()

    def _on_scrub_timeout(self):
        """节流定时器到期：加载目标帧画面（最多 ~33fps）。"""
        self._preview_frame_at(self._scrub_target_frame)

    def _on_trim_position_clicked(self, frame):
        """点击轨道或释放播放头时跳转到指定帧；播放中拖动后会保持暂停。"""
        if self.playback_thread is not None:
            if not self.playback_thread.is_paused():
                self.playback_thread.pause()
                # 等待线程真正进入暂停
                import time
                for _ in range(30):
                    if self.playback_thread.is_paused():
                        break
                    time.sleep(0.01)
                    QApplication.processEvents()
            self.playback_thread.jump_to_frame(frame, resume_after=False)
        self._current_frame_idx = max(0, int(frame))
        # 同步预览到释放位置
        self._preview_frame_at(frame)
        self.trim_slider.set_current_frame(frame)
        if self._current_total_frames > 0:
            self.progress_label.setText(f"{frame + 1} / {self._current_total_frames}")

    def _on_trim_segment_selected(self, start, end):
        """双击剪辑滑条时，将右侧位置曲线缩放到所选片段范围。"""
        if self.dataset is None or self._current_episode_idx < 0:
            return
        if start == end:
            return
        logger.info(f"[Trajectory] 双击选中片段 {start}~{end}，曲线缩放至该范围")
        self.trajectory_plot.set_segment_range(start, end)
        self.status_bar.showMessage(f"位置曲线已缩放到片段: {start} ~ {end}")

    def _update_trim_info(self):
        """更新剪辑信息标签显示。"""
        total = self._current_total_frames
        if total <= 0:
            self.trim_info_label.setText("剪辑区域: 未选择")
            return
        start, end = self.trim_slider.get_range()
        if start == end:
            self.trim_info_label.setText("剪辑区域: 未选择")
        else:
            fps = self.dataset.fps if self.dataset else 20
            start_t = start / fps
            end_t = end / fps
            self.trim_info_label.setText(
                f"剪辑区域: {start}~{end} 帧 ({start_t:.2f}s ~ {end_t:.2f}s) 共 {end - start + 1} 帧"
            )

    def _preview_frame_at(self, frame_idx):
        """在预览区显示指定帧。"""
        if self.dataset is None or self._current_episode_idx < 0:
            return
        try:
            ep_frames = self.dataset.get_episode_frames(self._current_episode_idx)
            total = len(ep_frames)
            if total == 0 or frame_idx < 0 or frame_idx >= total:
                return
            row = ep_frames.iloc[frame_idx]
            cam1 = self._load_image_for_preview(self._current_episode_idx, frame_idx, "observation.images.cam1")
            cam2 = self._load_image_for_preview(self._current_episode_idx, frame_idx, "observation.images.cam2")
            cam3 = self._load_image_for_preview(self._current_episode_idx, frame_idx, "observation.images.cam3")
            state_dim = self._state_dim()
            action = np.array(row.get("action", [0] * state_dim))
            state = np.array(row.get("observation.state", [0] * state_dim))
            timestamp = row.get("timestamp", frame_idx / self.dataset.fps if self.dataset.fps else 0)
            task_index = int(row.get("task_index", 0))
            task_name = self.dataset.task_name_for_index(task_index)
            self._display_frame(cam1, cam2, cam3, action, state, frame_idx, timestamp, task_name)
            self._current_frame_idx = frame_idx
            self.progress_label.setText(f"{frame_idx + 1} / {total}")
            # 同步位置曲线与底盘曲线竖线
            self.trajectory_plot.set_current_frame(frame_idx)
            self.chassis_plot.set_current_frame(frame_idx)
        except Exception as e:
            print(f"[Preview] 跳转帧失败: {e}")

    def _state_dim(self):
        """当前数据集 observation.state 维度（默认 18：双臂7+7+底盘4）。"""
        try:
            shape = self.dataset.info.get("features", {}).get("observation.state", {}).get("shape")
            if shape:
                return int(shape[0])
        except Exception:
            pass
        return 18

    def _preview_denoise(self, params):
        """在当前位置曲线上叠加平滑后的虚线预览（不修改数据）。"""
        if self.dataset is None or self._current_episode_idx < 0:
            return
        try:
            frames_data = self._load_episode_position_data(self._current_episode_idx)
            if frames_data is None:
                return
            keys = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
            mat = np.stack([np.asarray(frames_data[k], dtype=np.float64) for k in keys], axis=1)
            smoothed = _smooth_pose_matrix(
                mat, params.get("method", "savgol"), params.get("window", 11),
                params.get("polyorder", 2), params.get("sigma", 2.0),
            )
            overlay = {k: smoothed[:, i] for i, k in enumerate(keys)}
            self.trajectory_plot.set_smoothed_overlay(overlay)
            self.status_bar.showMessage("已叠加降噪预览（虚线），未修改数据")
        except Exception as e:
            QMessageBox.warning(self, "预览失败", f"降噪预览失败: {e}")

    # ==================== RECAP 奖励标注逻辑 ====================

    def _on_recap_label_changed(self, label):
        self._recap_current_label = label

    def _recap_mark_start(self):
        if self._reward_store is None:
            return
        tp = self.trajectory_plot
        self._recap_start_mark = self._current_frame_idx
        tp.recap_start_spin.setValue(self._recap_start_mark)
        self.status_bar.showMessage(f"RECAP 区间起点设为 {self._recap_start_mark}")

    def _recap_mark_end(self):
        self.trajectory_plot.recap_end_spin.setValue(self._current_frame_idx)
        self.status_bar.showMessage(f"RECAP 区间终点设为 {self._current_frame_idx}")

    def _recap_add_interval(self):
        if self._reward_store is None or self._current_episode_idx < 0:
            QMessageBox.information(self, "RECAP", "请先选择数据集与 episode。")
            return
        tp = self.trajectory_plot
        start = tp.recap_start_spin.value()
        end = tp.recap_end_spin.value()
        if end < start:
            start, end = end, start
        label = tp.recap_label_combo.currentText()
        note = tp.recap_note_edit.text().strip()
        intervention = tp.recap_intervention_check.isChecked()
        # 编辑模式：双击已有区间后保存修改（重叠校验排除自身，重排序后选中新区位）
        if self._recap_edit_index is not None:
            try:
                new_index = self._reward_store.edit_interval(
                    self._current_episode_idx, self._recap_edit_index,
                    start, end, label, note, is_human_intervention=intervention)
            except ValueError as e:
                QMessageBox.warning(self, "RECAP 区间重叠", str(e))
                return
            self._recap_exit_edit_mode()
            tp.recap_note_edit.clear()
            tp.recap_intervention_check.setChecked(False)
            self._recap_start_mark = None
            self._refresh_recap_table()
            self._update_recap_status()
            if 0 <= new_index < tp.recap_interval_table.rowCount():
                tp.recap_interval_table.selectRow(new_index)
            msg = f"RECAP 已修改区间 → [{start},{end}] = {label}"
            if intervention:
                msg += "（干预段）"
            self.status_bar.showMessage(msg, 5000)
            return
        try:
            self._reward_store.add_interval(
                self._current_episode_idx, start, end, label, note,
                is_human_intervention=intervention)
        except ValueError as e:
            QMessageBox.warning(self, "RECAP 区间重叠", str(e))
            return
        tp.recap_note_edit.clear()
        tp.recap_intervention_check.setChecked(False)
        self._recap_start_mark = None
        self._refresh_recap_table()
        self._update_recap_status()
        msg = f"RECAP 已添加区间 [{start},{end}] = {label}"
        if intervention:
            msg += "（干预段）"
        self.status_bar.showMessage(msg)

    # ── 区间双击编辑（P0.2 增强）──

    def _on_recap_interval_selected_playback(self):
        """RECAP 区间表选中行：播放条定位到该片段起始帧，进入暂停待续播。

        定位后处于暂停态，用户点独立"▶ 继续"键即从片段开头播放。
        表格重建（切集/刷新/增删区间）触发的空选择自动忽略；
        编辑模式内键盘导航的行切换同样定位（边改边预览）。
        """
        if self.dataset is None or self._current_episode_idx < 0:
            return
        sel = self.trajectory_plot.recap_interval_table.selectedItems()
        if not sel:
            return
        row = sel[0].row()
        if self._reward_store is None:
            return
        intervals = self._reward_store.get_intervals(self._current_episode_idx)
        if row < 0 or row >= len(intervals):
            return
        start = int(intervals[row].get("start", 0))
        end = int(intervals[row].get("end", start))
        self._seek_frame_paused(start)
        self.status_bar.showMessage(
            f"已定位到区间 [{start},{end}] 开头（帧 {start}，暂停中）——"
            f"点\"继续\"播放该片段")

    def _seek_frame_paused(self, frame_idx):
        """定位到指定帧并进入"暂停待续播"状态（播放键=停止、暂停键=继续）。

        无回放会话时先建会话（等价 _start_playback 的按钮态），随后暂停并
        jump_to_frame(resume_after=False)——线程发出该帧数据后自动暂停，
        播放条/进度随 frame_updated 同步。
        """
        ep = self._current_episode_idx
        if ep < 0 or self.dataset is None:
            return
        if (self.playback_thread is None
                or getattr(self.playback_thread, "episode_index", ep) != ep):
            self._start_playback(ep, auto_play=False)  # 建会话后立即暂停定位
        t = self.playback_thread
        if t is None:
            return
        if not t.is_paused():
            t.pause()
        t.jump_to_frame(max(0, int(frame_idx)), resume_after=False)
        self._current_frame_idx = max(0, int(frame_idx))  # "继续"从片段开头起播

    def _on_recap_table_double_clicked(self, row, _col):
        """双击区间表行：进入编辑模式。"""
        self._recap_begin_edit(row)

    def _recap_begin_edit(self, index):
        """双击区间（表格行或轨道色条）：把区间装入编辑器，进入编辑模式。"""
        if self._reward_store is None or self._current_episode_idx < 0:
            return
        intervals = self._reward_store.get_intervals(self._current_episode_idx)
        if not (0 <= index < len(intervals)):
            return
        iv = intervals[index]
        tp = self.trajectory_plot
        tp.recap_start_spin.setValue(iv["start"])
        tp.recap_end_spin.setValue(iv["end"])
        tp.recap_label_combo.setCurrentText(iv["label"])
        tp.recap_intervention_check.setChecked(bool(iv.get("is_human_intervention")))
        tp.recap_note_edit.setText(iv.get("note", ""))
        self._recap_edit_index = index
        tp.recap_add_btn.setText("💾 保存修改")
        tp.recap_cancel_edit_btn.setVisible(True)
        tp.recap_interval_table.selectRow(index)
        self.status_bar.showMessage(
            f"RECAP 编辑区间 #{index} [{iv['start']},{iv['end']}]："
            "修改起止/标签/干预后点“保存修改”，点“取消编辑”放弃", 8000)

    def _recap_exit_edit_mode(self):
        """退出区间编辑模式，恢复“添加区间”状态。"""
        if self._recap_edit_index is None:
            return
        self._recap_edit_index = None
        tp = self.trajectory_plot
        tp.recap_add_btn.setText("✚ 添加区间")
        tp.recap_cancel_edit_btn.setVisible(False)

    def _recap_cancel_edit(self):
        self._recap_exit_edit_mode()
        self.status_bar.showMessage("RECAP 已取消编辑，恢复添加模式")

    def _recap_quick_mark(self, label):
        """快捷标记：用 start_mark 到当前帧，或单帧。"""
        if self._reward_store is None or self._current_episode_idx < 0:
            QMessageBox.information(self, "RECAP", "请先选择数据集与 episode。")
            return
        cur = self._current_frame_idx
        if self._recap_start_mark is not None and self._recap_start_mark != cur:
            start, end = min(self._recap_start_mark, cur), max(self._recap_start_mark, cur)
        else:
            start, end = cur, cur
        intervention = self.trajectory_plot.recap_intervention_check.isChecked()
        try:
            self._reward_store.add_interval(
                self._current_episode_idx, start, end, label, "",
                is_human_intervention=intervention)
        except ValueError as e:
            QMessageBox.warning(self, "RECAP 区间重叠", str(e))
            return
        self.trajectory_plot.recap_intervention_check.setChecked(False)
        self._recap_start_mark = None
        self._recap_exit_edit_mode()  # 快标直接新增，退出任何进行中的编辑
        self._refresh_recap_table()
        self._update_recap_status()
        self.status_bar.showMessage(f"RECAP 快标 [{start},{end}] = {label}")

    def _recap_quick_mark_recent(self, label, seconds=2.0):
        """键盘快捷键（G/B/N）：以最近 seconds 秒为区间打标。

        焦点在文本/数字输入框时不触发，避免与正常输入冲突。
        """
        fw = QApplication.focusWidget()
        if isinstance(fw, (QLineEdit, QComboBox, QDoubleSpinBox)) or \
                fw.__class__.__name__ in ("QTextEdit", "QPlainTextEdit", "QSpinBox"):
            return
        if self._reward_store is None or self._current_episode_idx < 0:
            return
        fps = self.dataset.fps if self.dataset and self.dataset.fps else 20
        cur = self._current_frame_idx
        start = max(0, cur - int(seconds * fps))
        intervention = self.trajectory_plot.recap_intervention_check.isChecked()
        try:
            self._reward_store.add_interval(
                self._current_episode_idx, start, cur, label, "",
                is_human_intervention=intervention)
        except ValueError as e:
            self.status_bar.showMessage(f"RECAP 快标被拒绝: {e}", 5000)
            return
        self.trajectory_plot.recap_intervention_check.setChecked(False)
        self._recap_exit_edit_mode()  # 快标直接新增，退出任何进行中的编辑
        self._refresh_recap_table()
        self._update_recap_status()
        self.status_bar.showMessage(
            f"RECAP 快标 [{start},{cur}] = {label}（快捷键）")

    def _recap_mark_whole_episode(self, label):
        if self._reward_store is None or self._current_total_frames <= 0:
            return
        self._recap_exit_edit_mode()
        self._reward_store.clear_episode(self._current_episode_idx)
        self._reward_store.add_interval(
            self._current_episode_idx, 0, self._current_total_frames - 1, label, "whole_episode")
        self._refresh_recap_table()
        self._update_recap_status()

    def _recap_set_reviewed(self, checked):
        """设置本 episode 的已审核标志。"""
        if self._reward_store is None or self._current_episode_idx < 0:
            return
        self._reward_store.set_reviewed(self._current_episode_idx, bool(checked))
        self._update_recap_status()

    def _refresh_recap_meta(self):
        """刷新 episode 级控件：成败（人工直标优先，派生兜底）+ 已审核勾选。

        成败由部署侧保存时一键标记（人工直标，critic 的 r_T 主监督信号）；
        未直标时由片段标注派生（规则见 AnnotationStore.derive_episode_success）。
        """
        tp = self.trajectory_plot
        if self._reward_store is None or self._current_episode_idx < 0:
            tp.recap_ep_derived_label.setText("成败（派生）: —")
            tp.recap_ep_derived_label.setStyleSheet("font-weight:bold; color:#888888;")
            tp.recap_reviewed_check.blockSignals(True)
            tp.recap_reviewed_check.setChecked(False)
            tp.recap_reviewed_check.blockSignals(False)
            return
        manual = self._reward_store.get_episode_success(self._current_episode_idx)
        derived = self._reward_store.derive_episode_success(self._current_episode_idx)
        if manual is not None:
            text = "✓ 成功" if manual else "✗ 失败"
            color = "#2ecc71" if manual else "#e74c3c"
            tp.recap_ep_derived_label.setText(f"成败（人工标记）: {text}")
            tp.recap_ep_derived_label.setToolTip(
                "部署/保存时人工直标的 episode 成败（优先于片段派生值）。\n"
                f"（片段派生参考值: {derived}）")
        else:
            text = {True: "✓ 成功", False: "✗ 失败", None: "未标注"}[derived]
            color = {True: "#2ecc71", False: "#e74c3c", None: "#888888"}[derived]
            tp.recap_ep_derived_label.setText(f"成败（派生）: {text}")
        tp.recap_ep_derived_label.setStyleSheet(f"font-weight:bold; color:{color};")
        tp.recap_reviewed_check.blockSignals(True)
        tp.recap_reviewed_check.setChecked(
            self._reward_store.get_reviewed(self._current_episode_idx))
        tp.recap_reviewed_check.blockSignals(False)

    def _recap_delete_selected(self):
        if self._reward_store is None:
            return
        self._recap_exit_edit_mode()
        tp = self.trajectory_plot
        rows = sorted(
            {idx.row() for idx in tp.recap_interval_table.selectedIndexes()},
            reverse=True)
        for r in rows:
            self._reward_store.remove_interval(self._current_episode_idx, r)
        self._refresh_recap_table()
        self._update_recap_status()

    def _recap_undo_last(self):
        if self._reward_store is None:
            return
        self._recap_exit_edit_mode()
        intervals = self._reward_store.get_intervals(self._current_episode_idx)
        if intervals:
            self._reward_store.remove_interval(self._current_episode_idx, len(intervals) - 1)
            self._refresh_recap_table()
            self._update_recap_status()

    def _recap_clear_episode(self):
        if self._reward_store is None:
            return
        ret = QMessageBox.question(
            self, "确认", f"清空 episode {self._current_episode_idx} 的所有 RECAP 标注？")
        if ret == QMessageBox.StandardButton.Yes:
            self._recap_exit_edit_mode()
            self._reward_store.clear_episode(self._current_episode_idx)
            self._refresh_recap_table()
            self._update_recap_status()

    def _recap_save(self):
        if self._reward_store is None:
            return
        self._reward_store.save()
        self._update_info()  # RECAP 概览随标注保存刷新
        self.status_bar.showMessage(f"✓ RECAP 已保存到 {self._reward_store.filepath}", 5000)

    # ── 训练导出：展开为逐帧 sidecar（P1.1，内部调 make_recap_sidecar.expand_dataset）──

    def _recap_export_sidecar(self):
        """一键展开 RECAP 标注为逐帧 sidecar：先落盘标注（保证展开以最新稿为准），
        再后台线程展开（读 parquet 列，ela3_demo 量级数秒），完成弹窗显示分布摘要。"""
        if self._reward_store is None or self.dataset is None:
            QMessageBox.information(self, "RECAP", "请先选择数据集。")
            return
        if self._recap_export_thread is not None and self._recap_export_thread.isRunning():
            self.status_bar.showMessage("RECAP sidecar 正在展开中，请稍候...")
            return
        try:
            self._reward_store.save()  # 展开前强制落盘，sidecar 与标注稿严格一致
        except Exception as e:
            QMessageBox.warning(self, "RECAP", f"标注保存失败，已取消展开:\n{e}")
            return
        tp = self.trajectory_plot
        bp = tp.recap_bp_spin.value()
        tf = tp.recap_tf_spin.value()
        tp.recap_export_btn.setEnabled(False)
        self.status_bar.showMessage(
            f"RECAP sidecar 展开中（回传N={bp}, 失败末K={tf}）...")
        self._recap_export_thread = SidecarExportThread(
            str(self._reward_store.dataset_root), bad_backprop=bp, terminal_fail=tf,
            parent=self)
        self._recap_export_thread.finished_ok.connect(self._on_recap_export_done)
        self._recap_export_thread.error.connect(self._on_recap_export_error)
        self._recap_export_thread.start()

    def _on_recap_export_done(self, result):
        self.trajectory_plot.recap_export_btn.setEnabled(True)
        out = result.get("out_path")
        summary = "\n".join(result.get("summary_lines", []))
        self.status_bar.showMessage(f"✓ RECAP sidecar 已生成: {out}", 8000)
        QMessageBox.information(
            self, "RECAP sidecar 展开完成",
            f"{summary}\n\n原 parquet 未修改；剪辑/合并/删除 episode 后需重新展开。")

    def _on_recap_export_error(self, err):
        self.trajectory_plot.recap_export_btn.setEnabled(True)
        self.status_bar.showMessage("RECAP sidecar 展开失败", 5000)
        QMessageBox.warning(self, "RECAP sidecar 展开失败", err)

    # ── 价值优势显示（⑤⑥ 产物可视化，读 meta/recap_sidecar.jsonl）──

    def _recap_show_advantage(self):
        """📈 显示优势：先检测当前数据集是否有 meta/recap_sidecar.jsonl，
        有则弹窗绘制当前 episode 的价值/优势变化曲线（人标区间作背景色带）。"""
        if self.dataset is None or self._current_episode_idx < 0:
            QMessageBox.information(self, "显示优势", "请先选择数据集与 Episode。")
            return
        sc = Path(self.dataset.root) / "meta" / "recap_sidecar.jsonl"
        if not sc.is_file():
            QMessageBox.information(
                self, "显示优势",
                f"缺少 {sc}\n\n请先在下方点「🚀 生成训练 sidecar」（①）。")
            return
        ep = self._current_episode_idx
        rows = []
        try:
            with open(sc, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if int(r.get("episode_index", -1)) == ep:
                        rows.append(r)
        except Exception as e:
            QMessageBox.warning(self, "显示优势", f"sidecar 读取失败: {e}")
            return
        if not rows:
            QMessageBox.information(
                self, "显示优势",
                f"sidecar 中没有 episode {ep} 的行——sidecar 可能已过期，\n"
                "请重新点「🚀 生成训练 sidecar」（剪辑/合并/删除后索引会重排）。")
            return
        dlg = AdvantagePlotDialog(self, ep, rows)
        dlg.exec()

    def _refresh_sidecar_after_edit(self, dataset_root=None):
        """剪辑/删除等改变总帧数的操作完成后，自动重新展开训练 sidecar。

        仅当已生成过 sidecar（meta/recap_sidecar.jsonl 存在）才刷新：未生成过的
        数据集保持原样（展开是显式训练准备动作）。刷新后台线程完成，不打弹窗，
        避免"数据集检查"问题8 报 sidecar 行数≠总帧数 / 早于标注保存
        （RECAP_MEM_实施路线.md P0.4：剪辑/删除/合并后必须重新展开）。

        调用前提：标注重映射已落盘（_recap_after_trim_annotations /
        _recap_remap_after_delete 在本函数之前执行），data parquet 已写完。
        """
        try:
            ds_root = None
            if dataset_root:
                ds_root = Path(dataset_root)
            elif getattr(self, "_pending_dataset_path", None):
                ds_root = Path(self._pending_dataset_path)
            elif self.dataset is not None:
                ds_root = Path(self.dataset.root)
            if not ds_root or not (ds_root / "meta" / "recap_sidecar.jsonl").is_file():
                return  # 从未生成过 sidecar：不自动创建
            if (self._recap_export_thread is not None
                    and self._recap_export_thread.isRunning()):
                self.status_bar.showMessage(
                    "训练 sidecar 正在展开中；完成后请在 RECAP 面板手动重新生成", 6000)
                return
            tp = self.trajectory_plot
            tp.recap_export_btn.setEnabled(False)
            self.status_bar.showMessage("正在自动重新生成训练 sidecar（剪辑/删除后同步）...")
            self._recap_export_thread = SidecarExportThread(
                str(ds_root), bad_backprop=tp.recap_bp_spin.value(),
                terminal_fail=tp.recap_tf_spin.value(), parent=self)
            self._recap_export_thread.finished_ok.connect(self._on_recap_export_auto_done)
            self._recap_export_thread.error.connect(self._on_recap_export_auto_error)
            self._recap_export_thread.start()
        except Exception as e:
            logger.warning(f"[SidecarAuto] 剪辑/删除后自动重展开 sidecar 失败: {e}")

    def _on_recap_export_auto_done(self, result):
        """自动重展开完成（静默：状态栏提示，不弹窗）。"""
        self.trajectory_plot.recap_export_btn.setEnabled(True)
        out = result.get("out_path")
        n = result.get("total")
        self._update_info()  # RECAP 概览的 sidecar 行数/时间同步刷新
        self.status_bar.showMessage(f"✓ 训练 sidecar 已自动重新生成（{n} 行 -> {out}）", 8000)
        logger.info(f"[SidecarAuto] sidecar 已随剪辑/删除自动重展开: {n} 行 -> {out}")

    def _on_recap_export_auto_error(self, err):
        """自动重展开失败（静默降级：提示手动重新生成，不影响剪辑结果）。"""
        self.trajectory_plot.recap_export_btn.setEnabled(True)
        self.status_bar.showMessage(
            "训练 sidecar 自动重生成失败，请在 RECAP 面板手动\"生成训练 sidecar\"", 8000)
        logger.warning(f"[SidecarAuto] 自动重展开失败（需手动重新生成）: {err}")

    def _refresh_recap_table(self):
        """刷新 RECAP 区间列表。"""
        tp = self.trajectory_plot
        if self._reward_store is None:
            tp.recap_interval_table.setRowCount(0)
            self._refresh_recap_track()
            self._refresh_recap_meta()
            return
        intervals = self._reward_store.get_intervals(self._current_episode_idx)
        tp.recap_interval_table.setRowCount(len(intervals))
        for i, iv in enumerate(intervals):
            items = [
                str(i), str(iv["start"]), str(iv["end"]),
                iv["label"],
                "✓" if iv.get("is_human_intervention") else "",
                iv.get("note", ""),
            ]
            for col, txt in enumerate(items):
                item = QTableWidgetItem(txt)
                if col == 3:
                    item.setForeground(QBrush(QColor(RECAP_LABEL_COLORS.get(txt, "#888888"))))
                elif col == 4 and txt:
                    item.setForeground(QBrush(QColor(RECAP_INTERVENTION_COLOR)))
                tp.recap_interval_table.setItem(i, col, item)
        # 同步刷新播放时间线下方的 RECAP 彩色轨道
        self._refresh_recap_track()
        # 同步刷新 episode 级控件（结论高亮 / 已审核勾选）
        self._refresh_recap_meta()

    def _update_recap_status(self):
        tp = self.trajectory_plot
        if self._reward_store is None:
            tp.recap_status_label.setText("（未加载数据集）")
            return
        s = self._reward_store.stats()
        n_eps = self._reward_store.labeled_episode_count()
        n_rev = self._reward_store.reviewed_episode_count()
        total_eps = len(self.dataset.episodes) if self.dataset else 0
        # 成败统计：人工直标（部署侧）优先，片段派生兜底
        n_suc = n_fail = 0
        if self.dataset is not None:
            for i in range(total_eps):
                d = self._reward_store.get_episode_success(i)
                if d is None:
                    d = self._reward_store.derive_episode_success(i)
                if d is True:
                    n_suc += 1
                elif d is False:
                    n_fail += 1
        tp.recap_status_label.setText(
            f"已标注 episodes: {n_eps}/{total_eps}（已审核 {n_rev}） | "
            f"成败(人工/派生): ✓{n_suc} ✗{n_fail} | "
            f"区间数: {s['_intervals']} | "
            f"good帧: {s['good']} bad帧: {s['bad']} neutral帧: {s['neutral']} "
            f"干预帧: {s['intervention']}")

    # ── 剪辑时的标注防护（P0.3，RECAP_MEM_实施路线.md）──

    def _recap_annotation_guard_text(self, dataset_path):
        """数据集存在 RECAP 标注时，返回追加到剪辑确认框的提示文本；否则返回空串。"""
        try:
            if dataset_path and recap_has_annotations(dataset_path):
                return ("\n\nℹ️ 该数据集存在 RECAP 标注。剪辑后标注将按保留帧自动重映射：\n"
                        "· 原标注文件自动备份为 reward_annotations.stale.<时间戳>.bak\n"
                        "· 被剪辑 episode 的标注区间随帧号平移/截断/拆分"
                        "（完全落在删除段内的丢弃），标签与干预标志保留\n"
                        "· episode 级成败/已审核/来源字段保留")
        except Exception:
            pass
        return ""

    def _recap_pre_trim_backup(self, dataset_path, ep_indices):
        """剪辑确认后：备份标注文件，并登记剪辑成功后待处理的 episode。

        此时删除区间尚未最终确定（批量剪辑逐集检测），登记为 None 兜底；
        实际删除区间在剪辑线程启动时由 _recap_register_trim_ranges 覆盖。
        """
        try:
            bak = recap_backup_annotations(dataset_path)
            if bak is not None:
                for ep in ep_indices:
                    self._recap_trim_pending_ranges.setdefault(ep, None)
                self.status_bar.showMessage(f"RECAP 标注已备份: {bak}", 8000)
        except Exception as e:
            logger.warning(f"[RECAP] 标注备份失败: {e}")

    def _recap_register_trim_ranges(self, episode_idx, removed_ranges):
        """登记本次剪辑实际删除的帧区间（剪辑成功后据此重映射标注区间）。"""
        self._recap_trim_pending_ranges[episode_idx] = [
            (int(s), int(e)) for s, e in removed_ranges]

    def _recap_after_trim_annotations(self, episode_idx):
        """剪辑成功后：按删除区间重映射标注（P0.3 方案乙，剪辑不丢标注）。

        区间平移/截断/拆分（删除段内整段丢弃），episode 级字段
        （成败/已审核/source）原样保留；删除范围未知（None）时退回整段清除。
        """
        if (self._reward_store is None or episode_idx < 0
                or episode_idx not in self._recap_trim_pending_ranges):
            return
        ranges = self._recap_trim_pending_ranges.pop(episode_idx)
        try:
            if ranges is None:
                self._reward_store.clear_episode(episode_idx)
                self._reward_store.save()
                logger.info(f"[RECAP] episode {episode_idx} 的片段级标注已随剪辑清除（原文件已备份）")
            else:
                old = self._reward_store.get_intervals(episode_idx)
                new_ivs, stats = remap_intervals_after_trim(old, ranges)
                self._reward_store._get_episode(episode_idx)["intervals"] = new_ivs
                self._reward_store.save()
                logger.info(f"[RECAP] episode {episode_idx} 标注已随剪辑重映射: {stats}")
                if stats["dropped"] or stats["split"] or stats["truncated"]:
                    self.status_bar.showMessage(
                        f"RECAP 标注重映射: 原样保留{stats['kept']} 拆分{stats['split']} "
                        f"截断{stats['truncated']} 丢弃{stats['dropped']}", 8000)
        except Exception as e:
            logger.warning(f"[RECAP] 剪辑后标注重映射失败: {e}")
        self._refresh_recap_table()
        self._update_recap_status()
        self._update_info()  # RECAP 概览随剪辑重映射刷新

    def _recap_delete_guard_text(self, dataset_path):
        """删除 episode 时的标注提示（删除会使其后 episode_index 前移，标注键自动重映射）。"""
        try:
            if dataset_path and recap_has_annotations(dataset_path):
                return ("\n\nℹ️ 该数据集存在 RECAP 标注：删除后其后 episode_index 前移，\n"
                        "标注键将自动重映射（被删 episode 的标注一并移除），原文件自动备份。")
        except Exception:
            pass
        return ""

    def _recap_remap_after_delete(self, deleted_indices):
        """删除成功后：重映射标注键并保存（被删 episode 的标注一并移除）。"""
        if self._reward_store is None or not deleted_indices:
            return
        try:
            self._reward_store.remap_after_delete(deleted_indices)
            self._reward_store.save()
            self._refresh_recap_table()
            self._update_recap_status()
            logger.info(f"[RECAP] 标注键已随删除重映射: 删除 {sorted(deleted_indices)}")
        except Exception as e:
            logger.warning(f"[RECAP] 删除后标注重映射失败: {e}")

    # ── RECAP 轨道（播放时间线下方的彩色带）──

    def _on_recap_track_toggled(self, checked):
        """显示/隐藏 RECAP 标记轨道。"""
        self.recap_track.setVisible(checked)
        if checked:
            self._refresh_recap_track()

    def _refresh_recap_track(self):
        """刷新 RECAP 轨道：同步总帧数、区间、当前位置。"""
        if self._reward_store is None or self._current_episode_idx < 0:
            self.recap_track.set_total(0)
            self.recap_track.set_intervals([])
            return
        self.recap_track.set_total(self._current_total_frames)
        intervals = self._reward_store.get_intervals(self._current_episode_idx)
        self.recap_track.set_intervals(intervals)
        self.recap_track.set_current(self._current_frame_idx)

    def _on_vlm_label_requested(self, n):
        """用户点了“生成标签(VLM)”：按所选摄像头路数取视频，交给后台线程。"""
        if self.dataset is None or self._current_episode_idx < 0:
            self.trajectory_plot.subtask_result_label.setText("请先选择 Episode。")
            return
        vkeys = [name for name, f in self.dataset.features.items()
                 if self.dataset.is_video_feature(name)]
        if not vkeys:
            self.trajectory_plot.subtask_result_label.setText("数据集无视频，无法用 VLM 标签。")
            self.trajectory_plot.subtask_vlm_btn.setEnabled(True)
            return
        # 摄像头路数选择（默认 cam1）
        sel = 0
        if hasattr(self.trajectory_plot, "subtask_cam_combo"):
            sel = self.trajectory_plot.subtask_cam_combo.currentIndex()
        cam1_key = vkeys[0] if len(vkeys) >= 1 else None
        cam2_key = vkeys[1] if len(vkeys) >= 2 else None
        if sel == 1:        # 仅 cam2
            primary_key, secondary_key = cam2_key, None
        elif sel == 2:      # 双摄 cam1+cam2
            primary_key, secondary_key = cam1_key, cam2_key
        else:               # cam1（默认）
            primary_key, secondary_key = cam1_key, None
        if not primary_key:
            self.trajectory_plot.subtask_result_label.setText("所选摄像头不可用。")
            self.trajectory_plot.subtask_vlm_btn.setEnabled(True)
            return
        try:
            vpath = self.dataset.get_video_path(self._current_episode_idx, primary_key)
        except Exception:
            vpath = None
        if not vpath or not Path(vpath).exists():
            self.trajectory_plot.subtask_result_label.setText(
                f"视频不存在: {primary_key} ep{self._current_episode_idx}")
            self.trajectory_plot.subtask_vlm_btn.setEnabled(True)
            return
        # 双摄时取另一路
        cam2_path = None
        if secondary_key:
            try:
                p2 = self.dataset.get_video_path(self._current_episode_idx, secondary_key)
            except Exception:
                p2 = None
            if p2 and Path(p2).exists():
                cam2_path = p2
            else:
                self.status_bar.showMessage(f"提示：{secondary_key} 视频不存在，回退到单摄", 4000)
        segs = list(self.trajectory_plot._subtask_segments)
        if not segs:
            self.trajectory_plot.subtask_vlm_btn.setEnabled(True)
            return
        # 从标签库取已知标签作参考（提升跨 episode 一致性）
        try:
            lib = LabelLibrary(str(self.dataset.root))
            lib.load()
            known = lib.suggest(20)
        except Exception:
            known = []
        self._vlm_thread.set_work(str(vpath), segs, known_labels=known, cam2_path=cam2_path)
        if not self._vlm_thread.isRunning():
            self._vlm_thread.start()

    def _on_vlm_labels_ready(self, labels, note):
        """VLM 线程完成：把标签回填到曲线组件。"""
        self.trajectory_plot.set_segment_labels(labels, note)
        self.status_bar.showMessage(f"VLM 标签生成完成：{', '.join(labels)}")

    def _on_vlm_load_failed(self, err):
        self.trajectory_plot.subtask_vlm_btn.setEnabled(True)
        self.trajectory_plot.subtask_result_label.setText(f"VLM 失败: {err}")
        QMessageBox.warning(self, "VLM 标签失败", str(err))

    def _apply_subtask_annotations(self, assignments):
        """把预览的子任务语言标注写回 LeRobot 数据集。

        1. 保存到 meta/language_annotations.json（LanguageAnnotationStore）
        2. 写入逐帧 language_persistent 列到 data/*.parquet（write_language_to_dataset）
        3. 更新 info.json 特征注册
        """
        if self.dataset is None or self._current_episode_idx < 0:
            QMessageBox.warning(self, "无法应用", "请先选择数据集与 Episode。")
            return
        if not assignments:
            return

        ep_idx = self._current_episode_idx
        ds_root = self.dataset.root
        names = [a.get("task", "") for a in assignments]
        summary = "\n".join(
            f"  · [{a['start']}~{a['end']}] ({a['end']-a['start']+1}帧) {a['task']}"
            for a in assignments
        )
        # 二次确认（写入会修改原始 parquet）
        ret = QMessageBox.question(
            self, "确认写回数据集",
            f"将向数据集写回 Episode {ep_idx} 的子任务语言标注：\n\n{summary}\n\n"
            f"数据集：{ds_root}\n\n这会修改原始 parquet 文件（language_persistent 列），\n"
            f"建议先备份。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return

        try:
            # 1. 保存分段 JSON
            store = LanguageAnnotationStore(str(ds_root))
            store.load()
            store.set_episode(ep_idx, assignments)
            store.save()

            # 2. 写回逐帧列
            stat = write_language_to_dataset(ds_root, assignments, ep_idx, dry_run=False)

            # 3. 记入标签库（跨 episode 复用）
            lib = LabelLibrary(str(ds_root))
            lib.load()
            lib.set_episode_labels(ep_idx, names)
            lib.save()

            self.status_bar.showMessage(
                f"已写回语言标注：Episode {ep_idx}，"
                f"覆盖 {stat['covered']}/{stat['total']} 帧，"
                f"修改 {stat['files']} 个文件；标签已入库"
            )
            QMessageBox.information(
                self, "写回完成",
                f"已写回 Episode {ep_idx} 的子任务语言标注。\n"
                f"覆盖 {stat['covered']}/{stat['total']} 帧，"
                f"修改 {stat['files']} 个 parquet 文件。\n\n"
                f"子任务：{', '.join(names)}\n"
                f"标注文件：{Path(ds_root) / 'meta' / 'language_annotations.json'}\n"
                f"标签库：{Path(ds_root) / 'meta' / 'label_library.json'}",
            )
        except Exception as e:
            QMessageBox.critical(self, "写回失败", f"写回语言标注失败: {e}")

    def _denoise_current_episode(self, params):
        """对当前 Episode 应用平滑降噪。"""
        if self.dataset is None or self._current_episode_idx < 0:
            QMessageBox.warning(self, "警告", "请先选择 Episode")
            return
        reply = QMessageBox.question(
            self, "确认智能降噪",
            "确定对 Episode {} 应用平滑降噪吗？\n\n"
            "方法: {}\n窗口: {} 帧\n\n"
            "只修改 state/action 数值，帧数、元数据与视频保持不变；\n"
            "处理后自动重算 stats.json。\n"
            "是否备份取决于“剪辑前自动备份”复选框。".format(
                self._current_episode_idx,
                self.trajectory_plot.denoise_method_combo.currentText(),
                params.get("window", 11),
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._start_denoise_thread(self._current_episode_idx, params)

    def _denoise_all_episodes(self, params):
        """对所有 Episode 依次应用平滑降噪（单个线程顺序处理，避免文件并发写入）。"""
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return
        n_eps = len(self.dataset.episodes)
        if n_eps == 0:
            QMessageBox.warning(self, "警告", "当前数据集没有 Episode")
            return
        reply = QMessageBox.question(
            self, "确认批量智能降噪",
            "确定对全部 {} 个 Episode 依次应用平滑降噪吗？\n\n"
            "方法: {}\n窗口: {} 帧\n\n"
            "只修改 state/action 数值，帧数、元数据与视频保持不变；\n"
            "处理后自动重算 stats.json。\n"
            "是否备份取决于“剪辑前自动备份”复选框。".format(
                n_eps,
                self.trajectory_plot.denoise_method_combo.currentText(),
                params.get("window", 11),
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._start_denoise_thread(-1, params)

    def _start_denoise_thread(self, episode_idx, params):
        """启动降噪后台线程。episode_idx=-1 表示批量处理全部 Episode。"""
        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        if dataset_path is None:
            return
        self._pending_episode_idx = episode_idx
        self._set_controls_enabled(False)
        self.trajectory_plot.set_denoise_buttons_enabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("正在降噪...")
        self.status_bar.showMessage("正在平滑降噪，请稍候...")

        self.denoise_thread = DenoiseThread(
            dataset_path, episode_idx, params,
            backup_enabled=self.backup_before_trim_check.isChecked()
        )
        self.denoise_thread.progress.connect(self._on_denoise_progress)
        self.denoise_thread.finished_ok.connect(self._on_denoise_finished)
        self.denoise_thread.error.connect(self._on_denoise_error)
        self.denoise_thread.start()

    def _on_denoise_progress(self, current, total, msg):
        if total > 1:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        self.status_bar.showMessage(msg)

    def _on_denoise_finished(self, result, df_new):
        """降噪成功回调：刷新内存数据与当前曲线。"""
        try:
            if df_new is not None:
                self.dataset.data_df = df_new.reset_index(drop=True)
                self.dataset.num_frames = len(self.dataset.data_df)
            self._update_episode_table()
            self._update_info()
            if 0 <= self._current_episode_idx < len(self.dataset.episodes):
                self._load_and_show_trajectory(self._current_episode_idx, self._current_frame_idx)
        except Exception as e:
            import traceback
            print(f"[Denoise] 刷新失败: {e}\n{traceback.format_exc()}")

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_controls_enabled(True)
        self.trajectory_plot.set_denoise_buttons_enabled(True)
        self.status_bar.showMessage("降噪完成")
        QMessageBox.information(self, "智能降噪完成", result)

    def _on_denoise_error(self, error_msg):
        """降噪失败回调。"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0 / 0")
        self._set_controls_enabled(True)
        self.trajectory_plot.set_denoise_buttons_enabled(True)
        self.status_bar.showMessage("降噪失败")
        QMessageBox.critical(self, "降噪失败", error_msg)

    def _trim_stationary_segments(self, intervals):
        """根据智能分析检测到的静止区间，剪辑掉这些片段。（用户手动点击，当前 episode）"""
        if self.dataset is None or self._current_episode_idx < 0:
            QMessageBox.warning(self, "警告", "请先选择 Episode")
            return

        if not intervals:
            QMessageBox.information(self, "提示", "当前没有检测到静止区间")
            return

        merged = self._merge_stationary_intervals(intervals)
        if not merged:
            QMessageBox.information(self, "提示", "当前没有检测到静止区间")
            return

        ep_frames = self.dataset.get_episode_frames(self._current_episode_idx)
        total = len(ep_frames)
        n_removed = sum(e - s + 1 for s, e in merged)

        fps = self.dataset.fps if self.dataset.fps else 20
        detail_lines = []
        for s, e in merged:
            detail_lines.append(f"帧 {s}~{e} ({e - s + 1} 帧, {s / fps:.2f}s ~ {e / fps:.2f}s)")
        detail = "\n".join(detail_lines)

        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        reply = QMessageBox.question(
            self, "确认剪辑静止",
            "确定删除 Episode {} 中以下 {} 段静止片段吗？\n\n"
            "{}\n\n"
            "删除帧数: {}\n"
            "保留帧数: {}\n\n"
            "删除后当前 episode 会被更新，原始数据是否备份取决于“剪辑前自动备份”复选框。".format(
                self._current_episode_idx, len(merged), detail,
                n_removed, total - n_removed
            ) + self._recap_annotation_guard_text(dataset_path),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._recap_pre_trim_backup(dataset_path, [self._current_episode_idx])
        self._start_trim_stationary_thread(self._current_episode_idx, merged, show_progress=True)

    def _trim_jitter_segments(self, intervals):
        """后期处理分页：剪辑检测到的抖动区间（运动段剪辑，含 action 重算）。

        与剪辑静止同一安全管线（备份/RECAP 标注重映射/一致性校验），
        区别是抖动段位姿在变化，剪辑后必须重算 action[t]=state[t+k]
        （_on_trim_finished 中按 _pending_action_refix 标志执行 refix_action_labels）。
        """
        if self.dataset is None or self._current_episode_idx < 0:
            QMessageBox.warning(self, "警告", "请先选择 Episode")
            return
        if not intervals:
            QMessageBox.information(self, "提示", "没有可剪辑的抖动区间，请先在后期处理分页“分析抖动”")
            return
        merged = self._merge_stationary_intervals(intervals)
        if not merged:
            QMessageBox.information(self, "提示", "没有有效的抖动区间")
            return

        ep_frames = self.dataset.get_episode_frames(self._current_episode_idx)
        total = len(ep_frames)
        n_removed = sum(e - s + 1 for s, e in merged)
        fps = self.dataset.fps if self.dataset.fps else 20
        detail = "\n".join(
            f"帧 {s}~{e} （{e - s + 1} 帧, {s / fps:.2f}s ~ {e / fps:.2f}s）"
            for s, e in merged[:8])
        if len(merged) > 8:
            detail += f"\n... 共 {len(merged)} 段"

        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        # ⚠ 干预段保护提示：抖动窗若覆盖干预段，剪辑会连带删除（RECAP 最珍贵数据）
        itv_warning = ""
        try:
            if self._reward_store is not None:
                itvs = self._reward_store.get_intervals(self._current_episode_idx)
                overlap = [(iv["start"], iv["end"]) for iv in itvs
                           if iv.get("is_human_intervention")
                           and any(not (e < s or iv["start"] > e) for s, e in merged)]
                if overlap:
                    itv_warning = ("\n\n⚠️ 警告：以下人工干预段与抖动区间重叠，剪辑将被删除：\n"
                                   + "；".join(f"[{s},{e}]" for s, e in overlap)
                                   + "\n（干预段是最珍贵的训练数据，请确认是否真的要剪）")
        except Exception:
            pass
        reply = QMessageBox.question(
            self, "确认剪辑晃动",
            "确定删除 Episode {} 中以下 {} 段抖动区间吗？\n\n{}\n\n"
            "删除帧数: {}\n保留帧数: {}\n\n"
            "⚠ 抖动段位姿在变化，剪辑后将自动重算 action 标签"
            "（action[t]=state[t+前瞻]）；\nRECAP 标注区间自动随帧重映射。"
            "原始数据是否备份取决于“剪辑前自动备份”复选框。".format(
                self._current_episode_idx, len(merged), detail,
                n_removed, total - n_removed
            ) + itv_warning + self._recap_annotation_guard_text(dataset_path),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._recap_pre_trim_backup(dataset_path, [self._current_episode_idx])
        self._pending_action_refix = True  # 剪辑成功后重算 action（运动段剪辑必需）
        self._start_trim_stationary_thread(self._current_episode_idx, merged, show_progress=True)

    def _merge_stationary_intervals(self, intervals):
        """排序并合并相邻的静止区间。"""
        intervals = sorted([(int(s), int(e)) for s, e in intervals if s <= e])
        if not intervals:
            return []
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def _start_trim_stationary_thread(self, episode_idx, intervals, show_progress=True):
        """启动剪辑静止线程（不弹确认框）。"""
        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        self._pending_dataset_path = dataset_path
        self._pending_episode_idx = episode_idx
        self._recap_register_trim_ranges(episode_idx, intervals)  # P0.3 乙：登记删除区间

        self._set_controls_enabled(False)
        if show_progress:
            self.progress_bar.setRange(0, 0)
            self.progress_label.setText("正在剪辑静止片段...")
            self.status_bar.showMessage(f"正在剪辑 Episode {episode_idx} 的静止片段...")

        self.trim_thread = TrimStationaryThread(
            dataset_path, episode_idx, intervals,
            backup_enabled=self.backup_before_trim_check.isChecked()
        )
        self.trim_thread.finished_ok.connect(self._on_trim_finished)
        self.trim_thread.error.connect(self._on_trim_error)
        self.trim_thread.start()

    def _trim_all_stationary_segments(self):
        """对筛选出的 Episode 依次分析并剪辑静止区间（逐个处理，避免文件并发写入）。

        先弹出 Episode 筛选对话框（与批量删除同一交互：手动多选 / 任务名筛选），
        只处理勾选的 Episode。
        """
        if self._trim_all_in_progress:
            QMessageBox.information(self, "提示", "批量剪辑正在进行中，请等待完成。")
            return
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return

        n_eps = len(self.dataset.episodes)
        if n_eps == 0:
            QMessageBox.warning(self, "警告", "当前数据集没有 Episode")
            return

        # 弹出筛选对话框：手动多选 / 按任务名筛选（复用批量删除对话框的交互）
        rows = []
        for i, ep in enumerate(self.dataset.episodes):
            ep_idx = int(ep.get("episode_index", i))
            length = int(ep.get(
                "length", ep.get("dataset_to_index", 0) - ep.get("dataset_from_index", 0)))
            rows.append((ep_idx, length, _episode_task_display_name(self.dataset, ep)))
        dlg = BatchDeleteDialog(
            self, rows, self.dataset.fps,
            title="剪辑所有静止 - 选择 Episode",
            ok_text="✂ 剪辑选中", ok_color="#D84315")
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        indices = dlg.selected_indices()
        if not indices:
            return

        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        reply = QMessageBox.question(
            self, "确认剪辑所有静止",
            "确定对选中的 {} 个 Episode 依次检测并删除静止片段吗？\n\n"
            "处理规则：\n"
            "1. 逐个 Episode 分析并剪辑；\n"
            "2. 没有检测到静止的 Episode 会直接跳过（纯数据分析，不做视频解码/绘图/IK，毫秒级跳过）；\n"
            "3. 每次只写入一个 Episode，避免文件损坏；\n"
            "4. 原始数据是否备份取决于“剪辑前自动备份”复选框。\n\n"
            "该过程可能耗时较长，请耐心等待。".format(len(indices))
            + self._recap_annotation_guard_text(dataset_path),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._recap_pre_trim_backup(dataset_path, indices)

        # 初始化批量状态（队列只含筛选出的 Episode）
        self._trim_all_in_progress = True
        self._trim_all_mode = "stationary"
        self._trim_all_queue = list(indices)
        self._trim_all_total = len(indices)
        self._trim_all_trimmed = 0
        self._trim_all_skipped = 0
        self._trim_all_failed = 0
        self._trim_all_backup_enabled = self.backup_before_trim_check.isChecked()

        # 禁用备份复选框切换，避免批量过程中改变备份策略；
        # 静止/晃动批量入口共用同一条写入管线，必须全部互斥
        self.backup_before_trim_check.setEnabled(False)
        self.trajectory_plot.trim_stationary_btn.setEnabled(False)
        self.trajectory_plot.trim_all_stationary_btn.setEnabled(False)
        self.trajectory_plot.jitter_trim_btn.setEnabled(False)
        self.trajectory_plot.trim_all_jitter_btn.setEnabled(False)

        self.progress_bar.setRange(0, self._trim_all_total)
        self.progress_bar.setValue(0)
        self._set_controls_enabled(False)

        self._trim_all_process_next()

    def _trim_all_jitter_segments(self):
        """后期处理分页"剪辑所有晃动"：对选中的 Episode 依次检测并剪辑抖动区间。

        与"剪辑所有静止"同一批量管线（选集对话框 → 逐集分析/剪辑 → 完成），
        区别：抖动是运动段剪辑——
        · 检测用后期处理分页当前参数（窗口/幅度/反转次数）；
        · 全部剪辑完成后才统一重算一次 action 标签（action[t]=state[t+k]），
          避免逐集重算反复全量重写 parquet（N 集代价 O(N×总帧)）；
        · 训练 sidecar 也在最后统一自动重新生成（若已生成过）。
        """
        if self._trim_all_in_progress:
            QMessageBox.information(self, "提示", "批量剪辑正在进行中，请等待完成。")
            return
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return
        n_eps = len(self.dataset.episodes)
        if n_eps == 0:
            QMessageBox.warning(self, "警告", "当前数据集没有 Episode")
            return

        # 弹出筛选对话框：手动多选 / 按任务名筛选（与剪辑所有静止同一交互）
        rows = []
        for i, ep in enumerate(self.dataset.episodes):
            ep_idx = int(ep.get("episode_index", i))
            length = int(ep.get(
                "length", ep.get("dataset_to_index", 0) - ep.get("dataset_from_index", 0)))
            rows.append((ep_idx, length, _episode_task_display_name(self.dataset, ep)))
        dlg = BatchDeleteDialog(
            self, rows, self.dataset.fps,
            title="剪辑所有晃动 - 选择 Episode",
            ok_text="✂ 剪辑选中", ok_color="#8D6E63")
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        indices = dlg.selected_indices()
        if not indices:
            return

        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())

        # 干预段保护预检（纯数据计算，毫秒级）：抖动区间若与人工干预段重叠，
        # 剪辑会连带删除 RECAP 最珍贵数据，确认框中逐集列出
        itv_warning = ""
        try:
            tp = self.trajectory_plot
            overlap_eps = []
            if self._reward_store is not None:
                for ep_i in indices:
                    itvs = self._reward_store.get_intervals(ep_i)
                    if not any(iv.get("is_human_intervention") for iv in itvs):
                        continue
                    fd = self._load_episode_position_data(ep_i)
                    jit = detect_jitter_intervals(
                        fd, tp.jitter_window_spin.value(), tp.jitter_amp_spin.value(),
                        tp.jitter_rot_spin.value(), tp.jitter_rev_spin.value())
                    if jit and any(
                            not (e < iv["start"] or iv["end"] < s)
                            for iv in itvs if iv.get("is_human_intervention")
                            for s, e in jit):
                        overlap_eps.append(ep_i)
            if overlap_eps:
                itv_warning = ("\n\n⚠️ 警告：以下 Episode 的人工干预段与抖动区间重叠，"
                               "剪辑将被删除：\nEpisode " +
                               ", ".join(str(i) for i in overlap_eps) +
                               "\n（干预段是最珍贵的训练数据，请确认是否真的要剪）")
        except Exception:
            pass

        reply = QMessageBox.question(
            self, "确认剪辑所有晃动",
            "确定对选中的 {} 个 Episode 依次检测并删除抖动区间吗？\n\n"
            "处理规则：\n"
            "1. 逐个 Episode 分析并剪辑（参数=后期处理分页当前值：窗口{}帧 / "
            "位置{}mm / 姿态{}° / 反转≥{}）；\n"
            "2. 没有检测到抖动的 Episode 直接跳过（纯数据分析，毫秒级）；\n"
            "3. 每次只写入一个 Episode，避免文件损坏；\n"
            "4. 全部剪辑完成后统一重算一次 action 标签"
            "（action[t]=state[t+前瞻]，运动段剪辑必需），\n"
            "   并自动重新生成训练 sidecar（若已生成过）；\n"
            "5. 原始数据是否备份取决于“剪辑前自动备份”复选框。".format(
                len(indices),
                self.trajectory_plot.jitter_window_spin.value(),
                self.trajectory_plot.jitter_amp_spin.value(),
                self.trajectory_plot.jitter_rot_spin.value(),
                self.trajectory_plot.jitter_rev_spin.value())
            + itv_warning + self._recap_annotation_guard_text(dataset_path),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._recap_pre_trim_backup(dataset_path, indices)

        # 初始化批量状态（与剪辑所有静止共用队列/回调，仅 mode 不同）
        self._trim_all_in_progress = True
        self._trim_all_mode = "jitter"
        self._trim_all_queue = list(indices)
        self._trim_all_total = len(indices)
        self._trim_all_trimmed = 0
        self._trim_all_skipped = 0
        self._trim_all_failed = 0
        self._trim_all_backup_enabled = self.backup_before_trim_check.isChecked()

        self.backup_before_trim_check.setEnabled(False)
        self.trajectory_plot.trim_stationary_btn.setEnabled(False)
        self.trajectory_plot.trim_all_stationary_btn.setEnabled(False)
        self.trajectory_plot.jitter_trim_btn.setEnabled(False)
        self.trajectory_plot.trim_all_jitter_btn.setEnabled(False)

        self.progress_bar.setRange(0, self._trim_all_total)
        self.progress_bar.setValue(0)
        self._set_controls_enabled(False)

        self._trim_all_process_next()

    def _trim_all_process_next(self):
        """处理批量队列中的下一个 Episode。"""
        if not self._trim_all_in_progress:
            return

        if not self._trim_all_queue:
            self._trim_all_finish()
            return

        self._trim_all_current_ep = self._trim_all_queue.pop(0)
        episode_idx = self._trim_all_current_ep

        self.progress_bar.setValue(self._trim_all_total - len(self._trim_all_queue) - 1)
        self.progress_label.setText(
            f"批量剪辑: {self._trim_all_total - len(self._trim_all_queue)} / {self._trim_all_total}"
        )
        self.status_bar.showMessage(
            f"正在分析 Episode {episode_idx} ({self._trim_all_total - len(self._trim_all_queue)}/{self._trim_all_total})"
        )

        try:
            # 纯数据分析：不做整集 UI 加载（无视频解码/绘图/IK，毫秒级），
            # 检测算法与对应分页按钮共用同一函数，结果一致。
            frames_data = self._load_episode_position_data(episode_idx)
            if self._trim_all_mode == "jitter":
                tp = self.trajectory_plot
                intervals = detect_jitter_intervals(
                    frames_data, tp.jitter_window_spin.value(),
                    tp.jitter_amp_spin.value(), tp.jitter_rot_spin.value(),
                    tp.jitter_rev_spin.value())
                seg_name = "抖动"
            else:
                intervals = detect_stationary_intervals(
                    frames_data,
                    self.trajectory_plot._window_size,
                    self.trajectory_plot._threshold_pct)
                seg_name = "静止"
            if not intervals:
                self._trim_all_skipped += 1
                self.status_bar.showMessage(f"Episode {episode_idx} 无{seg_name}片段，跳过")
                # 无任何 UI 加载，立即处理下一个
                QTimer.singleShot(0, self._trim_all_process_next)
                return

            merged = self._merge_stationary_intervals(intervals)
            if not merged:
                self._trim_all_skipped += 1
                QTimer.singleShot(0, self._trim_all_process_next)
                return

            # 校验不能剪辑整个 episode
            ep_frames = self.dataset.get_episode_frames(episode_idx)
            total = len(ep_frames)
            n_removed = sum(e - s + 1 for s, e in merged)
            if n_removed >= total:
                self._trim_all_skipped += 1
                self.status_bar.showMessage(
                    f"Episode {episode_idx} {seg_name}区间覆盖全部帧，跳过"
                )
                QTimer.singleShot(0, self._trim_all_process_next)
                return

            self._trim_all_trimmed += 1
            self.status_bar.showMessage(
                f"正在剪辑 Episode {episode_idx} ({self._trim_all_total - len(self._trim_all_queue)}/{self._trim_all_total})"
            )
            # 逐个剪辑，线程完成后再继续下一个
            self._start_trim_stationary_thread(episode_idx, merged, show_progress=False)
        except Exception as e:
            import traceback
            self._trim_all_failed += 1
            print(f"[TrimAll] Episode {episode_idx} 处理异常: {e}\n{traceback.format_exc()}")
            self.status_bar.showMessage(f"Episode {episode_idx} 处理失败: {e}")
            QTimer.singleShot(0, self._trim_all_process_next)

    def _trim_all_finish(self):
        """批量剪辑结束，恢复 UI 并展示结果。"""
        self._trim_all_in_progress = False

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0 / 0")
        self._set_controls_enabled(True)
        self.backup_before_trim_check.setEnabled(True)
        self.trajectory_plot.trim_stationary_btn.setEnabled(True)
        self.trajectory_plot.trim_all_stationary_btn.setEnabled(True)
        self.trajectory_plot.jitter_trim_btn.setEnabled(True)
        self.trajectory_plot.trim_all_jitter_btn.setEnabled(True)

        # 批量过程不再逐集加载视图；若当前有选中的 Episode，其数据可能已被
        # 剪辑更新，最后统一重载一次视图保持一致（一次性开销，不在循环内）
        if self._current_episode_idx >= 0 and self.dataset is not None:
            try:
                self._load_first_frame(self._current_episode_idx)
            except Exception:
                pass

        seg_name = "晃动" if self._trim_all_mode == "jitter" else "静止"
        msg = (
            f"批量剪辑完成（{seg_name}）。\n\n"
            f"选中 Episode 数: {self._trim_all_total}\n"
            f"已剪辑: {self._trim_all_trimmed}\n"
            f"无{seg_name}跳过: {self._trim_all_skipped}\n"
            f"失败: {self._trim_all_failed}"
        )

        # 晃动批量收尾：全部剪辑完成后统一重算一次 action 标签
        # （action[t]=state[t+前瞻]；逐集重算需反复全量重写 parquet，
        #   N 集代价 O(N×总帧)，推迟到最后一次完成）
        if self._trim_all_mode == "jitter" and self._trim_all_trimmed > 0:
            self.status_bar.showMessage("正在统一重算 action 标签（晃动批量剪辑后）...")
            QApplication.processEvents()
            self._pending_action_refix = True
            self._run_action_refix_if_pending()
            msg += "\n\naction 标签已统一重算（action[t]=state[t+前瞻]）"

        self.status_bar.showMessage("批量剪辑完成")

        # 批量剪辑改变了总帧数且标注重映射已落盘：全部剪辑完成后才自动同步
        # 训练 sidecar（若有）——顺序保证 sidecar 行数/mtime 与最终数据一致
        self._refresh_sidecar_after_edit()

        QMessageBox.information(self, "批量剪辑完成", msg)

    def _trim_selected_segment(self):
        """删除两个标记之间的片段（包含两端），保留其余部分。"""
        if self.dataset is None or self._current_episode_idx < 0:
            QMessageBox.warning(self, "警告", "请先选择 Episode")
            return

        start, end = self.trim_slider.get_range()
        if start == end:
            QMessageBox.warning(self, "警告", "请先用下方三角标记选择要删除的片段")
            return

        ep_frames = self.dataset.get_episode_frames(self._current_episode_idx)
        total = len(ep_frames)
        if start >= end or end >= total or start < 0:
            QMessageBox.warning(self, "警告", "选择的剪辑范围无效")
            return

        fps = self.dataset.fps if self.dataset.fps else 20
        start_t = start / fps
        end_t = end / fps
        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        reply = QMessageBox.question(
            self, "确认剪辑",
            "确定删除 Episode {} 中以下片段吗？\\n\\n"
            "帧范围: {} ~ {}\\n"
            "时间段: {:.2f}s ~ {:.2f}s\\n"
            "删除帧数: {}\\n"
            "保留帧数: {}\\n\\n"
            "删除后当前 episode 会被更新，原始数据是否备份取决于“剪辑前自动备份”复选框。".format(
                self._current_episode_idx, start, end, start_t, end_t,
                end - start + 1, total - (end - start + 1)
            ) + self._recap_annotation_guard_text(dataset_path),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._recap_pre_trim_backup(dataset_path, [self._current_episode_idx])

        # 执行剪辑
        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        self._pending_dataset_path = dataset_path
        self._pending_episode_idx = self._current_episode_idx
        self._recap_register_trim_ranges(self._current_episode_idx,
                                         [(start, end)])  # P0.3 乙：登记删除区间

        self._set_controls_enabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("正在剪辑...")
        self.status_bar.showMessage(f"正在剪辑 Episode {self._current_episode_idx}...")

        self.trim_thread = TrimEpisodeThread(
            dataset_path, self._current_episode_idx, start, end,
            backup_enabled=self.backup_before_trim_check.isChecked()
        )
        self.trim_thread.finished_ok.connect(self._on_trim_finished)
        self.trim_thread.error.connect(self._on_trim_error)
        self.trim_thread.start()

    def _set_controls_enabled(self, enabled):
        """启用/禁用操作控件。"""
        self.play_btn.setEnabled(enabled)
        self.pause_btn.setEnabled(enabled)
        self.resume_btn.setEnabled(enabled)
        self.del_btn.setEnabled(enabled)
        self.trim_btn.setEnabled(enabled)
        self.episode_table.setEnabled(enabled)
        self.dataset_combo.setEnabled(enabled)
        # 批量删除与批量剪辑共用数据集文件，必须互斥，防止并发写入损坏数据
        self.batch_del_btn.setEnabled(
            enabled and self.dataset is not None and len(self.dataset.episodes) > 0)

    def _reload_stats_in_memory(self):
        """从磁盘重载 meta/stats.json 到内存 dataset.stats。

        trim/denoise 等操作会重算 stats.json 并更新内存 data_df/info，
        但 OfflineDataset.stats 是打开时加载的快照；不重载会让
        "数据集检查"读到旧 count 误报问题4。删除类操作整体重建 dataset，
        无需本函数。
        """
        try:
            ds_path = getattr(self, "_pending_dataset_path", None)
            if not ds_path and self.dataset is not None:
                ds_path = getattr(self.dataset, "root", None)
            if not ds_path:
                return
            stats_fp = Path(ds_path) / "meta" / "stats.json"
            if stats_fp.is_file():
                with open(stats_fp, "r", encoding="utf-8") as f:
                    self.dataset.stats = json.load(f)
        except Exception as e:
            logger.warning(f"[DataOps] 刷新内存 stats 失败: {e}")

    def _run_action_refix_if_pending(self):
        """抖动剪辑（运动段）成功后：重算 action 标签并刷新内存数据。

        磁盘上 refix_action_labels 已重写 data 与 stats；内存需重读 data_df
        （线程返回的 df_new 中 action 仍是剪辑前的旧值）。
        """
        if not getattr(self, "_pending_action_refix", False):
            return
        self._pending_action_refix = False
        try:
            ds_path = getattr(self, "_pending_dataset_path", None) or (
                str(self.dataset.root) if self.dataset else None)
            if not ds_path:
                return
            self.status_bar.showMessage("正在重算 action 标签（抖动剪辑后）...")
            QApplication.processEvents()
            k, n = refix_action_labels(ds_path)
            # 重新读入内存（action 已变）
            import pandas as pd
            files = sorted((Path(ds_path) / "data").rglob("*.parquet"))
            self.dataset.data_df = pd.concat(
                [pd.read_parquet(p) for p in files], ignore_index=True)
            self.dataset.num_frames = len(self.dataset.data_df)
            self._reload_stats_in_memory()
            self.status_bar.showMessage(
                f"action 标签已重算（前瞻 {k} 帧 / {n} 帧，action[t]=state[t+k]）", 8000)
            logger.info(f"[TrimJitter] action 标签重算完成: k={k}, frames={n}")
        except Exception as e:
            logger.warning(f"[TrimJitter] action 标签重算/刷新失败（磁盘可能已正确）: {e}")

    def _on_trim_finished(self, result, df_new, new_ep_df, info):
        """剪辑成功回调。"""
        episode_idx = self._pending_episode_idx
        # 剪辑重写了视频文件，首帧缓存全部失效
        self._preview_frame_cache.clear()

        # 批量模式：更新数据后继续下一个，不弹成功框
        if self._trim_all_in_progress:
            self.status_bar.showMessage(f"Episode {episode_idx} 剪辑完成，继续下一个...")
            try:
                if df_new is not None:
                    self.dataset.data_df = df_new.reset_index(drop=True)
                    self.dataset.num_frames = len(self.dataset.data_df)
                if new_ep_df is not None:
                    self.dataset.episodes = new_ep_df.to_dict("records")
                if info is not None:
                    self.dataset.info = info
                self._reload_stats_in_memory()  # trim 已重算 stats.json，同步内存防检查误报
                self.dataset.release_video_caps()
                # P0.3 乙：按删除区间重映射标注（不丢失），episode 级字段保留
                self._recap_after_trim_annotations(episode_idx)
                self._run_action_refix_if_pending()  # 抖动剪辑：重算 action 标签
                self._update_episode_table()
                self._update_info()
            except Exception as e:
                import traceback
                self._trim_all_failed += 1
                print(f"[TrimAll] Episode {episode_idx} 刷新失败: {e}\n{traceback.format_exc()}")
            # 继续处理下一个 episode
            QTimer.singleShot(50, self._trim_all_process_next)
            return

        self.status_bar.showMessage("正在刷新数据集...")
        try:
            # 直接用线程返回的数据更新内存，避免重新读取整个数据集
            if df_new is not None:
                self.dataset.data_df = df_new.reset_index(drop=True)
                self.dataset.num_frames = len(self.dataset.data_df)
            if new_ep_df is not None:
                self.dataset.episodes = new_ep_df.to_dict("records")
            if info is not None:
                self.dataset.info = info
            # 刷新内存 stats：trim/denoise 已重算 meta/stats.json，不同步会让
            # "数据集检查"读旧值误报 count 不一致（问题4）
            self._reload_stats_in_memory()
            # 释放旧视频缓存，因为视频文件已被修改
            self.dataset.release_video_caps()

            self._update_episode_table()
            self._update_info()
            # P0.3 乙：按删除区间重映射标注（不丢失），episode 级字段保留
            self._recap_after_trim_annotations(episode_idx)
            # 抖动剪辑（运动段）：重算 action 标签并刷新内存
            self._run_action_refix_if_pending()
            # 重新选择同一 episode
            if episode_idx < len(self.dataset.episodes):
                self.episode_table.selectRow(episode_idx)
                self._on_episode_selected()
        except Exception as e:
            import traceback
            QMessageBox.warning(self, "刷新警告", f"数据集刷新失败: {e}\n\n{traceback.format_exc()}\n请手动刷新。")
            self._set_controls_enabled(True)
            return

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self._set_controls_enabled(True)
        self.trim_btn.setEnabled(True)
        self.del_btn.setEnabled(True)

        self.status_bar.showMessage(f"已剪辑 Episode {episode_idx}")

        # 剪辑改变了总帧数且标注重映射已落盘：若已生成过训练 sidecar，
        # 后台自动重新展开，保持"数据集检查"问题8 通过
        self._refresh_sidecar_after_edit()

        QMessageBox.information(self, "剪辑成功", result)

    def _on_trim_error(self, error_msg):
        """剪辑失败回调。"""
        # 批量模式：记录失败并终止批量流程，避免在可能已损坏的数据上继续处理
        if self._trim_all_in_progress:
            self._trim_all_failed += 1
            self._trim_all_in_progress = False
            print(f"[TrimAll] Episode {self._pending_episode_idx} 剪辑失败: {error_msg}")

            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_label.setText("0 / 0")
            self._set_controls_enabled(True)
            self.backup_before_trim_check.setEnabled(True)
            self.trajectory_plot.trim_stationary_btn.setEnabled(True)
            self.trajectory_plot.trim_all_stationary_btn.setEnabled(True)
            self.trajectory_plot.jitter_trim_btn.setEnabled(True)
            self.trajectory_plot.trim_all_jitter_btn.setEnabled(True)

            # 晃动批量中途失败：已剪辑的 Episode action 仍是旧标签，
            # 立即统一重算一次并同步 sidecar，保证磁盘数据一致可检查
            if self._trim_all_mode == "jitter" and self._trim_all_trimmed > 0:
                self._pending_action_refix = True
                self._run_action_refix_if_pending()
                self._refresh_sidecar_after_edit()

            self.status_bar.showMessage("批量剪辑已中止")
            QMessageBox.critical(
                self, "批量剪辑失败",
                f"Episode {self._pending_episode_idx} 剪辑失败，批量流程已中止，"
                f"请检查该 episode 数据或日志后重试。\n\n错误信息:\n{error_msg}"
            )
            return

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0 / 0")
        self.status_bar.showMessage("剪辑失败")
        QMessageBox.critical(self, "剪辑失败", error_msg)
        self._set_controls_enabled(True)
        self.trim_btn.setEnabled(False)
        self.del_btn.setEnabled(False)

    def _on_episode_selected(self):
        selected = self.episode_table.selectedItems()
        if not selected or self.dataset is None:
            return

        episode_idx = int(selected[0].text())
        self._current_episode_idx = episode_idx

        if episode_idx < len(self.dataset.episodes):
            ep = self.dataset.episodes[episode_idx]
            length = ep.get("length", 0)
            duration = length / self.dataset.fps if self.dataset.fps > 0 else 0
            self.current_episode_label.setText(
                f"已选择 Episode {episode_idx} ({length} 帧, {duration:.1f}秒)"
            )
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(True)
            self.del_btn.setEnabled(True)
            self.trim_btn.setEnabled(True)

            # 自动加载并显示第一帧的图像和数据
            self._load_first_frame(episode_idx)

            # 设置任务编辑器
            self._setup_task_editor(episode_idx)

            # 刷新 RECAP 标注区间表（加载该 episode 的已标注区间）
            self._recap_exit_edit_mode()  # 切换 episode：退出编辑模式，防止误改旧 episode 的区间
            self._refresh_recap_table()
            self._update_recap_status()
            self.trajectory_plot.recap_start_spin.setMaximum(max(0, length - 1))
            self.trajectory_plot.recap_end_spin.setMaximum(max(0, length - 1))

    def _setup_task_editor(self, episode_idx):
        """根据选中的 episode 初始化任务名称编辑器。"""
        if not self.dataset or episode_idx < 0 or episode_idx >= len(self.dataset.episodes):
            self._reset_task_editor()
            return

        ep = self.dataset.episodes[episode_idx]
        task_name = ""

        # 优先从 episode 的 tasks 数组读取
        if "tasks" in ep and ep["tasks"] is not None:
            tasks = ep["tasks"]
            if hasattr(tasks, '__iter__') and not isinstance(tasks, str):
                task_list = list(tasks)
                if task_list:
                    task_name = str(task_list[0]).strip()

        # 如果为空，再尝试从 task_index 查找（按值查找，不能用列表位置索引）
        if not task_name and "task_index" in ep:
            task_name = str(self.dataset.task_name_for_index(ep["task_index"])).strip()

        self._original_task_name = task_name
        self._task_modified = False

        # 断开之前的信号，避免设置文本时触发保存提示
        if self._task_edit_connected:
            try:
                self.task_name_edit.editingFinished.disconnect(self._on_task_editing_finished)
            except Exception:
                pass
            self._task_edit_connected = False

        self.task_name_edit.blockSignals(True)
        self.task_name_edit.setText(task_name)
        self.task_name_edit.setEnabled(True)
        self.task_name_edit.blockSignals(False)

        self.task_status_label.setText("已加载任务名称")
        self.task_status_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")

        # 连接编辑完成信号
        self.task_name_edit.editingFinished.connect(self._on_task_editing_finished)
        self._task_edit_connected = True

        # 组名编辑框：加载组名标签段（策略段不显示、保存时自动保留）
        _, group_label = self._episode_policy_group_text(episode_idx)
        self._original_group_label = group_label
        self.group_name_edit.blockSignals(True)
        self.group_name_edit.setText(group_label)
        self.group_name_edit.setEnabled(True)
        self.group_name_edit.blockSignals(False)

    def _reset_task_editor(self):
        """重置任务编辑器为未选择状态。"""
        if self._task_edit_connected:
            try:
                self.task_name_edit.editingFinished.disconnect(self._on_task_editing_finished)
            except Exception:
                pass
            self._task_edit_connected = False

        self.task_name_edit.blockSignals(True)
        self.task_name_edit.clear()
        self.task_name_edit.setPlaceholderText("选择 Episode 后显示任务名称")
        self.task_name_edit.setEnabled(False)
        self.task_name_edit.blockSignals(False)

        # 组名编辑框同步复位
        self.group_name_edit.blockSignals(True)
        self.group_name_edit.clear()
        self.group_name_edit.setEnabled(False)
        self.group_name_edit.blockSignals(False)
        self._original_group_label = ""

        self.task_status_label.setText("")
        self._original_task_name = ""
        self._task_modified = False
        self._task_edit_in_progress = False
        self._current_episode_idx = -1

    def _on_task_editing_finished(self):
        """任务名称编辑完成（按 Enter 或失去焦点）时触发。"""
        if self._current_episode_idx < 0 or self.dataset is None:
            return

        # 防止 editingFinished 重复触发（如按回车后再失去焦点）导致对话框多次弹出
        if self._task_edit_in_progress:
            return

        new_name = self.task_name_edit.text().strip()

        # 空名称检查
        if not new_name:
            QMessageBox.warning(self, "无效任务名称", "任务名称不能为空。")
            self.task_name_edit.blockSignals(True)
            self.task_name_edit.setText(self._original_task_name)
            self.task_name_edit.blockSignals(False)
            return

        # 未变化则忽略
        if new_name == self._original_task_name:
            self._task_modified = False
            self.task_status_label.setText("已加载任务名称")
            self.task_status_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
            return

        self._task_modified = True
        self.task_status_label.setText("任务名称已修改，请保存")
        self.task_status_label.setStyleSheet("QLabel { color: #FF9800; font-size: 11px; }")

        self._task_edit_in_progress = True
        try:
            # 提示是否保存
            reply = QMessageBox.question(
                self, "保存任务名称",
                f"是否将 Episode {self._current_episode_idx} 的任务名称\n"
                f"从 \"{self._original_task_name}\" 保存为 \"{new_name}\"？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._save_task_name(self._current_episode_idx, new_name)
            else:
                # 恢复原始名称
                self.task_name_edit.blockSignals(True)
                self.task_name_edit.setText(self._original_task_name)
                self.task_name_edit.blockSignals(False)
                self._task_modified = False
                self.task_status_label.setText("已恢复原始名称")
                self.task_status_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        finally:
            self._task_edit_in_progress = False

    # ── 组名编辑/批量修改（π*0.6 任务组元数据，R1a 配套） ──

    def _on_group_editing_finished(self):
        """组名编辑完成（回车/失焦）：确认后写回标注；置空 = 取消分组。"""
        if self._current_episode_idx < 0 or self._reward_store is None:
            return
        new_label = self.group_name_edit.text().strip()
        old_label = getattr(self, "_original_group_label", "")
        if new_label == old_label:
            return
        reply = QMessageBox.question(
            self, "保存组名",
            f"是否将 Episode {self._current_episode_idx} 的组名\n"
            f"从 \"{old_label or '(未分组)'}\" 改为 \"{new_label or '(未分组)'}\"？\n\n"
            f"（group_id 的策略段自动保留；剪辑/合并不受影响）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._save_group_name(self._current_episode_idx, new_label)
        else:
            self.group_name_edit.blockSignals(True)
            self.group_name_edit.setText(old_label)
            self.group_name_edit.blockSignals(False)

    def _save_group_name(self, ep_idx, label):
        """写回组名：group_id = 组名 | 策略run名（无策略时仅组名）；空 = 取消分组。"""
        try:
            pol = self._reward_store.get_episode_policy(ep_idx)
            if label and pol:
                gid = f"{label} | {self._policy_run_name(pol)}"
            else:
                gid = label or None
            self._reward_store.set_episode_group(ep_idx, gid)
            self._reward_store.save()  # 自动备份
            self._original_group_label = label
            self.task_status_label.setText(
                f"✓ Episode {ep_idx} 组名已保存: {gid or '(未分组)'}")
            self.task_status_label.setStyleSheet("QLabel { color: #2e7d32; font-size: 11px; }")
            self._update_episode_table()  # 表格"组"列刷新
            self._update_info()           # 总信息统计刷新
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"组名保存失败: {e}")

    def _show_group_batch_dialog(self):
        """批量修改组名：原组名（下拉，自动带出当前集）→ 新组名（可空=全部取消分组）。"""
        if self.dataset is None or self._reward_store is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return
        # 收集现有组名标签（按首次出现顺序去重）
        labels = []
        for i in range(len(self.dataset.episodes)):
            g = self._reward_store.get_episode_group(i)
            if g:
                lab = str(g).split(" | ")[0].strip()
                if lab and lab not in labels:
                    labels.append(lab)
        if not labels:
            QMessageBox.information(self, "批量修改组名", "当前数据集没有已分组的 Episode")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("批量修改组名")
        dlg.setMinimumWidth(440)
        layout = QVBoxLayout(dlg)

        form = QFormLayout()
        old_combo = QComboBox()
        old_combo.addItems(labels)
        if getattr(self, "_original_group_label", "") in labels:
            old_combo.setCurrentText(self._original_group_label)
        form.addRow("原组名:", old_combo)

        new_edit = QLineEdit()
        new_edit.setPlaceholderText("输入新组名（留空 = 该组全部取消分组）")
        form.addRow("新组名:", new_edit)
        layout.addLayout(form)

        count_label = QLabel()
        count_label.setStyleSheet("QLabel { color: #666; }")

        def _affected_ep_indices(name):
            out = []
            for i in range(len(self.dataset.episodes)):
                g = self._reward_store.get_episode_group(i)
                if g and str(g).split(" | ")[0].strip() == name:
                    out.append(i)
            return out

        def update_count():
            n = len(_affected_ep_indices(old_combo.currentText()))
            count_label.setText(f"将影响 {n} 个 Episode（同组全部替换，各自策略段保留）")

        old_combo.currentTextChanged.connect(lambda _: update_count())
        update_count()
        layout.addWidget(count_label)

        from PyQt6.QtWidgets import QDialogButtonBox
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        old_label = old_combo.currentText().strip()
        new_label = new_edit.text().strip()
        if not old_label or new_label == old_label:
            return

        targets = _affected_ep_indices(old_label)
        try:
            for i in targets:
                pol = self._reward_store.get_episode_policy(i)
                gid = f"{new_label} | {self._policy_run_name(pol)}" if (new_label and pol) else (new_label or None)
                self._reward_store.set_episode_group(i, gid)
            self._reward_store.save()  # 自动备份
        except Exception as e:
            QMessageBox.warning(self, "批量修改组名", f"保存失败: {e}")
            return

        if self._current_episode_idx in targets:
            self._original_group_label = new_label
            self.group_name_edit.blockSignals(True)
            self.group_name_edit.setText(new_label)
            self.group_name_edit.blockSignals(False)
        self._update_episode_table()
        self._update_info()
        action = "改为" if new_label else "取消分组"
        name_part = f' "{new_label}"' if new_label else ""
        QMessageBox.information(
            self, "批量修改组名完成",
            f"{len(targets)} 个 Episode 的组名已{action}{name_part}。")

    def _save_task_name(self, episode_idx, new_name):
        """保存任务名称。每个 episode 独立命名：如果新名称已存在则复用其 task_index，否则创建新的 task_index。"""
        if not self.dataset or episode_idx < 0 or episode_idx >= len(self.dataset.episodes):
            return

        ep = self.dataset.episodes[episode_idx]
        old_name = self._original_task_name

        # 空名称/未变化不处理
        if not new_name or new_name == old_name:
            self._original_task_name = new_name
            self._task_modified = False
            self._update_episode_table()
            self._update_info()
            return

        # 查找新名称是否已存在，确定目标 task_index
        new_task_index = None
        for t in self.dataset.tasks:
            if t.get("task", "") == new_name:
                new_task_index = t.get("task_index")
                break

        # 如果不存在，先确定新的 task_index（内存追加延后到写盘成功后，
        # 避免写盘失败时内存 tasks 与磁盘 tasks.parquet 不一致）
        is_new_task = new_task_index is None
        if is_new_task:
            new_task_index = max(
                [t.get("task_index", 0) for t in self.dataset.tasks], default=-1
            ) + 1

        try:
            root = self.dataset.root
            meta_dir = root / "meta"
            data_dir = root / "data"

            # 0. 编辑前置闸门：meta/data/info 不一致时拒绝动笔
            _assert_dataset_consistent(root)

            # 1. 内存中准备 tasks.parquet 新内容（先不写盘）
            tasks_path = meta_dir / "tasks.parquet"
            tasks_df = None
            if tasks_path.exists():
                tasks_df = pd.read_parquet(tasks_path)
                if new_name not in tasks_df.index:
                    new_row = pd.DataFrame(
                        {"task_index": [new_task_index]}, index=[new_name]
                    )
                    tasks_df = pd.concat([tasks_df, new_row])

            # 2. 内存中准备 episodes 新内容：只改当前 episode 的 task_index 和 tasks
            ep_parquet_dir = meta_dir / "episodes"
            ep_jsonl_path = meta_dir / "episodes.jsonl"
            ep_fmt = None
            ep_files = []
            ep_df = None
            ep_rows = None

            if ep_parquet_dir.exists():
                ep_files = sorted(ep_parquet_dir.rglob("*.parquet"))
                if ep_files:
                    ep_fmt = "parquet"
                    ep_df = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
                    if "task_index" in ep_df.columns:
                        ep_df.loc[episode_idx, "task_index"] = new_task_index
                    if "tasks" in ep_df.columns:
                        ep_df.at[episode_idx, "tasks"] = [new_name]
            elif ep_jsonl_path.exists():
                ep_fmt = "jsonl"
                ep_files = [ep_jsonl_path]
                with open(ep_jsonl_path) as f:
                    ep_rows = [json.loads(l) for l in f if l.strip()]
                if 0 <= episode_idx < len(ep_rows):
                    ep_rows[episode_idx]["task_index"] = new_task_index
                    ep_rows[episode_idx]["tasks"] = [new_name]

            # 3. 内存中准备 data 改动：只改写包含该 episode 行的文件，
            #    每文件行数/行序不变 → data/chunk_index、data/file_index 指针保持有效，
            #    无需整体重写数据集（原实现对单个 task_index 改动重写全部 data，无备份）
            data_files = sorted(data_dir.rglob("*.parquet"))
            changed_data = []   # [(path, new_df), ...]
            df_all_parts = []
            file_lens = []
            for p in data_files:
                df = pd.read_parquet(p)
                file_lens.append(len(df))
                new_df = df
                if "episode_index" in df.columns and "task_index" in df.columns:
                    mask = df["episode_index"] == episode_idx
                    if bool(mask.any()):
                        new_df = df.copy()
                        new_df.loc[mask, "task_index"] = new_task_index
                        changed_data.append((p, new_df))
                df_all_parts.append(new_df)
            df_all = pd.concat(df_all_parts, ignore_index=True) if df_all_parts else None

            # 3.5 回收孤儿任务：本次改名使旧任务名失去最后一处引用时，删除该任务行
            #     并把剩余 task_index 压实为 0 基连续（lerobot 按 iloc 取任务名，
            #     孤儿/空洞是"数据集检查"的孤儿告警源，见 数据问题分析.md 问题7）。
            #     压实会平移其他任务的 task_index → 涉及的 data 文件全部重写
            #     （行数/行序不变，episode 数据指针不受影响）。
            gc_removed = []
            if tasks_df is not None and df_all is not None and ep_df is not None:
                gc_tasks, gc_removed = _gc_orphan_tasks_df(tasks_df, df_all, ep_df)
                if gc_tasks is not None:
                    tasks_df = gc_tasks
                    off = 0
                    changed_data = []
                    for p, n in zip(data_files, file_lens):
                        changed_data.append((p, df_all.iloc[off:off + n].copy()))
                        off += n
                    if new_name in tasks_df.index:
                        new_task_index = int(tasks_df.loc[new_name, "task_index"])

            # 4. 全部内容就绪后校验（parquet/jsonl 两种布局都校验；
            #    布局未变应恒通过，不一致说明数据集已带病，抛错且此时未动任何文件）
            if ep_df is not None and df_all is not None:
                _sync_episode_data_pointers(ep_df, df_all)

            # 5. 备份将被修改的文件到 .bak_task_save_<时间戳>/
            backup_dir = root / f".bak_task_save_{datetime.now():%Y%m%d_%H%M%S}"

            def backup_file(p):
                p = Path(p)
                dst = backup_dir / p.relative_to(root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)

            for p, _ in changed_data:
                backup_file(p)
            for p in ep_files:
                backup_file(p)
            if tasks_df is not None:
                backup_file(tasks_path)

            # 6. 原子写盘（data → episodes → tasks → stats）
            for p, new_df in changed_data:
                _atomic_write_parquet(new_df, p)

            if ep_fmt == "parquet" and ep_df is not None:
                _remove_files([str(p) for p in ep_files])
                _write_parquet_chunks(ep_df, ep_parquet_dir / "chunk-000")
            elif ep_fmt == "jsonl" and ep_rows is not None:
                with open(ep_jsonl_path, "w") as f:
                    for row in ep_rows:
                        f.write(json.dumps(row) + "\n")

            if tasks_df is not None:
                # tasks.parquet 以任务名为索引（index=True），不能用 _atomic_write_parquet；
                # 同样采用 tmp + os.replace 原子写
                tmp_tasks = tasks_path.with_suffix(".tmp.parquet")
                tasks_df.to_parquet(tmp_tasks)
                os.replace(tmp_tasks, tasks_path)

                # info.json 的 total_tasks 同步
                info_path = meta_dir / "info.json"
                if info_path.exists():
                    with open(info_path, encoding="utf-8") as f:
                        info = json.load(f)
                    info["total_tasks"] = len(tasks_df)
                    with open(info_path, "w", encoding="utf-8") as f:
                        json.dump(info, f, indent=2, ensure_ascii=False)
                    self.dataset.info = info

            # task_index 已变化，重算全局统计
            if df_all is not None and changed_data:
                _recompute_global_stats(str(meta_dir), df_all)
                self.dataset.data_df = df_all

            # 写盘全部成功后再更新内存中的 tasks 列表（含孤儿回收后的压实重排）
            if tasks_df is not None:
                self.dataset.tasks = [
                    {"task_index": int(v), "task": str(n)}
                    for n, v in zip(tasks_df.index, tasks_df["task_index"])
                ]

            # 7. 更新内存中的 episode
            ep["task_index"] = new_task_index
            if "tasks" in ep and ep["tasks"] is not None:
                ep["tasks"] = [new_name]

            # 8. 刷新显示
            self._original_task_name = new_name
            self._task_modified = False
            self._update_episode_table()
            self._update_info()

            # 确保输入框显示的是最终保存的名称
            self.task_name_edit.blockSignals(True)
            self.task_name_edit.setText(new_name)
            self.task_name_edit.blockSignals(False)

            self.task_status_label.setText("任务名称已保存")
            self.task_status_label.setStyleSheet("QLabel { color: #4CAF50; font-size: 11px; }")
            _gc_note = f"，回收孤儿任务: {gc_removed}" if gc_removed else ""
            self.status_bar.showMessage(
                f"Episode {episode_idx} 任务名称已保存为 \"{new_name}\" "
                f"(task_index={new_task_index}){_gc_note}"
            )

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "保存失败", f"保存任务名称失败:\n{str(e)}\n\n{traceback.format_exc()}")
            # 恢复原始名称
            self.task_name_edit.blockSignals(True)
            self.task_name_edit.setText(self._original_task_name)
            self.task_name_edit.blockSignals(False)

    # ==================== 批量修改任务名称 ====================
    def _episode_task_name(self, ep):
        tasks = ep.get("tasks")
        if tasks is None:
            return ""
        try:
            tasks = list(tasks)
        except TypeError:
            return str(tasks)
        return str(tasks[0]) if tasks else ""

    def _show_task_batch_dialog(self):
        """批量修改任务名对话框：自动带出当前 Episode 的任务名，输入新名后全部替换。"""
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return
        task_names = [t.get("task", "") for t in self.dataset.tasks if t.get("task")]
        if not task_names:
            QMessageBox.warning(self, "警告", "当前数据集没有任务")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("批量修改任务名称")
        dlg.setMinimumWidth(420)
        layout = QVBoxLayout(dlg)

        form = QFormLayout()
        old_combo = QComboBox()
        old_combo.addItems(task_names)
        current = self._original_task_name
        if current in task_names:
            old_combo.setCurrentText(current)
        form.addRow("原始任务名:", old_combo)

        new_edit = QLineEdit()
        new_edit.setPlaceholderText("输入新任务名称")
        from PyQt6.QtGui import QRegularExpressionValidator
        from PyQt6.QtCore import QRegularExpression
        new_edit.setValidator(QRegularExpressionValidator(
            QRegularExpression(r"^[\w\x{4e00}-\x{9fa5}][\w\x{4e00}-\x{9fa5}\s\-]*$"), new_edit
        ))
        form.addRow("新任务名:", new_edit)
        layout.addLayout(form)

        count_label = QLabel()
        count_label.setStyleSheet("QLabel { color: #666; }")

        def update_count():
            name = old_combo.currentText()
            n = sum(1 for ep in self.dataset.episodes
                    if self._episode_task_name(ep) == name)
            count_label.setText(f"将影响 {n} 个 Episode（同名任务全部替换）")

        old_combo.currentTextChanged.connect(lambda _: update_count())
        update_count()
        layout.addWidget(count_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        apply_btn = QPushButton("全部替换")
        apply_btn.setStyleSheet(
            "QPushButton { background-color: #E65100; color: white; font-weight: bold; padding: 6px 18px; }"
        )
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        def do_apply():
            old_name = old_combo.currentText()
            new_name = new_edit.text().strip()
            if not new_name:
                QMessageBox.warning(dlg, "提示", "请输入新任务名称")
                return
            if new_name == old_name:
                QMessageBox.information(dlg, "提示", "新名称与原名称相同，无需修改")
                return
            n = sum(1 for ep in self.dataset.episodes
                    if self._episode_task_name(ep) == old_name)
            if n == 0:
                QMessageBox.information(dlg, "提示", f"没有使用任务 \"{old_name}\" 的 Episode")
                return
            ret = QMessageBox.question(
                dlg, "确认批量替换",
                f"将把 {n} 个 Episode 的任务\n  \"{old_name}\"\n全部替换为\n  \"{new_name}\"\n\n是否继续？"
            )
            if ret != QMessageBox.StandardButton.Yes:
                return
            ok, msg = self._batch_rename_task(old_name, new_name)
            if ok:
                dlg.accept()
                QMessageBox.information(self, "完成", msg)
            else:
                QMessageBox.critical(dlg, "失败", msg)

        apply_btn.clicked.connect(do_apply)
        dlg.exec()

    def _batch_rename_task(self, old_name, new_name):
        """批量重命名任务：所有任务名为 old_name 的 Episode 改为 new_name。

        安全性（对照 数据问题分析.md）：
        - data parquet 逐文件原地改写 task_index 列，行数/行序/文件布局不变，
          data/chunk_index、data/file_index 指针保持有效（避免问题1）；
        - frame_index/timestamp 不触碰（避免问题3）；
        - task_index 变化后重算全局 stats.json 与受影响 episode 的 per-episode
          stats/task_index 列（避免问题4）；
        - 所有被修改的文件先备份到 <root>/.bak_task_rename_<时间戳>/。
        """
        if self.dataset is None:
            return False, "未加载数据集"
        root = self.dataset.root
        meta_dir = root / "meta"
        tasks_path = meta_dir / "tasks.parquet"

        # 编辑前置闸门：meta/data/info 不一致时拒绝动笔
        try:
            _assert_dataset_consistent(root)
        except ValueError as e:
            return False, str(e)

        try:
            tasks_df = pd.read_parquet(tasks_path)
        except Exception as e:
            return False, f"读取 tasks.parquet 失败: {e}"
        if old_name not in tasks_df.index:
            return False, f"任务 \"{old_name}\" 不存在"

        old_idx = int(tasks_df.loc[old_name, "task_index"])
        merge_existing = new_name in tasks_df.index
        new_idx = int(tasks_df.loc[new_name, "task_index"]) if merge_existing else old_idx

        backup_dir = root / f".bak_task_rename_{datetime.now():%Y%m%d_%H%M%S}"
        backed_up = []

        def backup_file(p):
            p = Path(p)
            rel = p.relative_to(root)
            dst = backup_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            backed_up.append(p)

        try:
            # 1. tasks.parquet：纯改名只换名称（task_index 不变，data 无需改动）；
            #    并入已有任务则删除旧任务行，并把剩余 task_index 压实为 0 基连续
            #    （lerobot dataset_reader 用 iloc 按位置取任务，留空洞会越界，
            #    见 数据问题分析.md 问题7）
            backup_file(tasks_path)
            compact = None
            frame_remap = None
            if merge_existing:
                tasks_df = tasks_df.drop(index=old_name)
                tasks_df = tasks_df.sort_values("task_index")
                compact = {int(v): i for i, v in enumerate(tasks_df["task_index"])}
                tasks_df["task_index"] = [compact[int(v)] for v in tasks_df["task_index"]]
                frame_remap = dict(compact)
                frame_remap[old_idx] = compact[new_idx]
                new_idx = compact[new_idx]
            else:
                tasks_df = tasks_df.rename(index={old_name: new_name})
            # tasks.parquet 以任务名为索引（index=True），tmp + os.replace 原子写
            _tmp = tasks_path.with_suffix(".tmp.parquet")
            tasks_df.to_parquet(_tmp, index=True)
            os.replace(_tmp, tasks_path)

            # 2. data parquet：仅并入已有任务时需要改写 task_index（旧值→新值，
            #    其余值随压实同步平移），逐文件原地改写，保持文件布局与行序不变
            if merge_existing:
                data_files = sorted((root / "data").rglob("*.parquet"))
                for p in data_files:
                    df = pd.read_parquet(p)
                    if "task_index" not in df.columns:
                        continue
                    mapped = df["task_index"].map(
                        lambda v: frame_remap.get(int(v), int(v))
                    )
                    if bool((mapped == df["task_index"]).all()):
                        continue
                    backup_file(p)
                    df["task_index"] = mapped.astype(df["task_index"].dtype)
                    _atomic_write_parquet(df, p)

            # 3. meta/episodes：更新 tasks 字符串；并入已有任务时 task_index 列
            #    与 per-episode stats/task_index 统计列需随压实整体平移（不只改
            #    被并任务的行），逐文件原地改写
            ep_files = sorted((meta_dir / "episodes").rglob("*.parquet"))
            affected_eps = set()
            for p in ep_files:
                df = pd.read_parquet(p)
                if "tasks" not in df.columns:
                    continue
                tasks_col = df["tasks"].to_numpy(copy=True)
                row_hit = []
                for pos in range(len(df)):
                    tv = tasks_col[pos]
                    try:
                        tl = list(tv) if tv is not None else []
                    except TypeError:
                        tl = [tv]
                    if old_name in tl:
                        row_hit.append((pos, tl))
                ti_mapped = None
                if merge_existing and "task_index" in df.columns:
                    ti_mapped = df["task_index"].map(
                        lambda v: frame_remap.get(int(v), int(v))
                    )
                ti_changed = ti_mapped is not None and bool(
                    (ti_mapped != df["task_index"]).any()
                )
                if not row_hit and not ti_changed:
                    continue
                backup_file(p)
                for pos, tl in row_hit:
                    new_tasks = [new_name if t == old_name else t for t in tl]
                    tasks_col[pos] = np.array(new_tasks, dtype=object)
                    if "episode_index" in df.columns:
                        affected_eps.add(int(df["episode_index"].iloc[pos]))
                    else:
                        affected_eps.add(pos)
                df["tasks"] = tasks_col
                if ti_mapped is not None:
                    df["task_index"] = ti_mapped.astype(df["task_index"].dtype)
                    # 每个 episode 只有一个任务：min=max=mean=分位数=task_index，std=0
                    ti_vals = df["task_index"].to_numpy(dtype=np.float64)
                    prefix = "stats/task_index/"
                    for suffix, arr in [
                        ("min", ti_vals), ("max", ti_vals), ("mean", ti_vals),
                        ("std", np.zeros_like(ti_vals)), ("q01", ti_vals),
                        ("q10", ti_vals), ("q50", ti_vals), ("q90", ti_vals),
                        ("q99", ti_vals),
                    ]:
                        col = prefix + suffix
                        if col not in df.columns:
                            continue
                        col_vals = np.empty(len(arr), dtype=object)
                        for i, v in enumerate(arr):
                            col_vals[i] = np.array([float(v)], dtype=np.float64)
                        df[col] = col_vals
                _atomic_write_parquet(df, p)

            if not affected_eps:
                # 数据层面没有引用该任务的 episode，只有 tasks.parquet 变了
                pass

            # 4. task_index 值发生变化时，重算全局 stats.json（含图像 count 同步）
            if merge_existing:
                stats_path = meta_dir / "stats.json"
                if stats_path.exists():
                    backup_file(stats_path)
                df_all = self.dataset.data_df.copy()
                if "task_index" in df_all.columns:
                    df_all["task_index"] = df_all["task_index"].map(
                        lambda v: frame_remap.get(int(v), int(v))
                    )
                _recompute_global_stats(str(meta_dir), df_all)
                self.dataset.data_df = df_all

                # info.json 的 total_tasks 同步
                info_path = meta_dir / "info.json"
                if info_path.exists():
                    backup_file(info_path)
                    with open(info_path, encoding="utf-8") as f:
                        info = json.load(f)
                    info["total_tasks"] = len(tasks_df)
                    with open(info_path, "w", encoding="utf-8") as f:
                        json.dump(info, f, indent=2, ensure_ascii=False)
                    self.dataset.info = info

            # 5. 内存同步（并入时保留任务的 task_index 随压实同步平移）
            new_tasks_mem = []
            for t in self.dataset.tasks:
                if t.get("task") == old_name:
                    if merge_existing:
                        continue
                    new_tasks_mem.append({"task_index": new_idx, "task": new_name})
                else:
                    if merge_existing and compact is not None:
                        t = dict(t)
                        t["task_index"] = compact.get(
                            int(t.get("task_index", -1)), t.get("task_index")
                        )
                    new_tasks_mem.append(t)
            self.dataset.tasks = new_tasks_mem
            for ep in self.dataset.episodes:
                if self._episode_task_name(ep) == old_name:
                    ep["tasks"] = [new_name]
                if merge_existing and "task_index" in ep:
                    ep["task_index"] = frame_remap.get(
                        int(ep["task_index"]), ep["task_index"]
                    )

            # 6. 界面刷新；若当前编辑的 Episode 受影响，同步任务编辑框
            if self._original_task_name == old_name:
                self._original_task_name = new_name
                self.task_name_edit.blockSignals(True)
                self.task_name_edit.setText(new_name)
                self.task_name_edit.blockSignals(False)
            self._update_episode_table()
            self._update_info()

            n_eps = len(affected_eps)
            mode = "并入已有任务" if merge_existing else "重命名"
            self.status_bar.showMessage(
                f"批量替换完成: \"{old_name}\" -> \"{new_name}\" ({mode}, {n_eps} 个 Episode)"
            )
            return True, (
                f"已将 {n_eps} 个 Episode 的任务\n\"{old_name}\" → \"{new_name}\"\n"
                f"方式: {mode}\n原文件备份: {backup_dir.name}"
            )
        except Exception as e:
            import traceback
            return False, f"批量替换失败:\n{e}\n\n{traceback.format_exc()}\n\n已修改文件的原始版本备份在: {backup_dir}"

    def _on_replay_clicked(self):
        """重新播放：无论当前状态，停止并从第 0 帧重头播放。"""
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return
        selected = self.episode_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "警告", "请先选择一个 Episode")
            return
        episode_idx = int(selected[0].text())
        auto_play = self.auto_play_check.isChecked()
        self._start_playback(episode_idx, auto_play)

    def _on_continue_clicked(self):
        """继续播放：从播放条（滑条）当前位置起播。

        三态统一处理：播放中=跳到滑条帧继续；暂停中=从滑条帧续播；
        无会话=以滑条帧为起始帧新建回放。滑条位置是唯一事实来源。
        """
        if self.dataset is None or self._current_episode_idx < 0:
            QMessageBox.warning(self, "警告", "请先选择一个 Episode")
            return
        total = max(1, self._current_total_frames)
        frame = min(max(0, int(getattr(self, "_current_frame_idx", 0) or 0)), total - 1)
        t = self.playback_thread
        if (t is not None and t.isRunning()
                and getattr(t, "episode_index", -1) == self._current_episode_idx):
            # 先 jump 后 resume：线程唤醒后先处理跳转请求，不吐旧帧
            t.jump_to_frame(frame, resume_after=True)
            if t.is_paused():
                t.resume()
            self.status_bar.showMessage(f"继续播放（帧 {frame + 1}/{total}）")
        else:
            self._start_playback(self._current_episode_idx, start_frame=frame)

    def _on_pause_clicked(self):
        """暂停：仅暂停正在进行的播放（独立按键，非暂停/继续切换）。"""
        t = self.playback_thread
        if t is None or not t.isRunning():
            return
        if not t.is_paused():
            t.pause()
            
    def _start_playback(self, episode_index, auto_play=False, start_frame=0):
        self._stop_playback()

        # 确保 trim_slider 与当前 episode 帧数同步
        ep_frames = self.dataset.get_episode_frames(episode_index)
        total_frames = len(ep_frames)
        self._current_total_frames = total_frames
        self.trim_slider.set_total_frames(total_frames)
        self.trim_slider.set_current_frame(0)
        self.trim_slider.setEnabled(True)
        self.progress_label.setText(f"1 / {total_frames}")

        # 重新加载该 episode 的位置曲线（_stop_playback 会清空它）；
        # 保留曲线上的子任务分段色块与 VLM 标签（不随播放清除）
        self._load_and_show_trajectory(episode_index, current_frame=0,
                                       clear_annotations=False)

        speed = self.speed_spin.value()
        
        self.playback_thread = PlaybackThread(
            self.dataset, episode_index, speed, auto_play, start_frame=start_frame
        )
        self.playback_thread.frame_data.connect(self._on_frame_data)
        self.playback_thread.frame_updated.connect(self._on_frame_updated)
        self.playback_thread.episode_finished.connect(self._on_episode_finished)
        self.playback_thread.playback_finished.connect(self._on_playback_finished)
        self.playback_thread.error_occurred.connect(self._on_error)
        self.playback_thread.status_changed.connect(self._on_status_changed)
        
        self.playback_thread.start()

        # 三独立按键：文本恒定，不再切换"停止/继续"状态
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(True)
        self.auto_play_check.setEnabled(False)
        self.status_bar.showMessage("回放中")
        
    def _load_first_frame_cached(self, episode_idx, camera_name):
        """带 LRU 缓存的首帧预览加载：回访同一 episode 时免去视频打开/解码。"""
        key = (episode_idx, camera_name)
        img = self._preview_frame_cache.get(key)
        if img is not None:
            self._preview_frame_cache.move_to_end(key)
            return img
        img = self._load_image_for_preview(episode_idx, 0, camera_name)
        if img is not None and isinstance(img, np.ndarray) and img.size > 0:
            self._preview_frame_cache[key] = img
            self._preview_frame_cache.move_to_end(key)
            while len(self._preview_frame_cache) > self._PREVIEW_CACHE_MAX:
                self._preview_frame_cache.popitem(last=False)
        return img

    def _load_first_frame(self, episode_idx):
        """加载并显示指定 episode 的第一帧图像和数据（不启动播放线程）"""
        try:
            ep_frames = self.dataset.get_episode_frames(episode_idx)
            total_frames = len(ep_frames)
            if total_frames == 0:
                return
            
            # 获取第一帧数据
            row = ep_frames.iloc[0]
            
            # 加载图像（LRU 缓存：回访的 episode 不再重复解码视频）
            cam1_img = self._load_first_frame_cached(episode_idx, "observation.images.cam1")
            cam2_img = self._load_first_frame_cached(episode_idx, "observation.images.cam2")
            cam3_img = self._load_first_frame_cached(episode_idx, "observation.images.cam3")

            # 获取 action 和 state
            state_dim = self._state_dim()
            action = np.array(row.get("action", [0] * state_dim))
            state = np.array(row.get("observation.state", [0] * state_dim))
            timestamp = row.get("timestamp", 0.0)
            task_index = int(row.get("task_index", 0))

            # 获取任务名称（按 task_index 值查找，不能用列表位置索引）
            task_name = self.dataset.task_name_for_index(task_index)

            # 更新显示
            self._display_frame(cam1_img, cam2_img, cam3_img, action, state, 0, timestamp, task_name)
            
            # 更新进度条为初始状态
            self._current_total_frames = total_frames
            self._current_frame_idx = 0
            self.trim_slider.set_total_frames(total_frames)
            self.trim_slider.set_current_frame(0)
            self.trim_slider.setEnabled(True)
            self.progress_label.setText(f"1 / {total_frames}")
            self._update_trim_info()

            # 加载并显示该 episode 的 xyz 位置曲线
            self._load_and_show_trajectory(episode_idx, current_frame=0)
            
        except Exception as e:
            print(f"[Preview] 加载第一帧失败: {e}")

    def _load_and_show_trajectory(self, episode_idx, current_frame=0, clear_annotations=True):
        """加载指定 episode 的 xyz 位置数据并刷新右侧曲线图。

        clear_annotations=False 时保留曲线上的子任务分段色块与 VLM 标签
        （用于点播放重绘同一 episode）。
        """
        try:
            frames_data = self._load_episode_position_data(episode_idx)
            self.trajectory_plot.set_data(frames_data, current_frame=current_frame,
                                          clear_annotations=clear_annotations)
            # 底盘/升降曲线（无底盘数据时自动显示占位）
            self.chassis_plot.set_data(frames_data, current_frame=current_frame)
        except Exception as e:
            print(f"[Trajectory] 加载位置曲线失败: {e}")
            self.trajectory_plot.clear()
            self.chassis_plot.clear()

    # ------------------------------------------------------------------
    # Episode 位姿 / 底盘数据提取
    # ------------------------------------------------------------------
    def _load_episode_position_data(self, episode_idx):
        """从 data_df 中提取指定 episode 的位姿序列。

        返回 dict: {
            'x': array, 'y': array, 'z': array,
            'rx': array, 'ry': array, 'rz': array,  # 欧拉角 (rad)
            'gripper': array,  # 夹爪 (rad)
            'timestamps': array
        }
        自动兼容两种格式：
          - 7 维新格式 [x,y,z,rx,ry,rz,gripper_rad]：直接读取
          - 8 维旧格式 [x,y,z,qx,qy,qz,qw,gripper_deg]：四元数→欧拉角，度→弧度
        如果数据不存在则返回 None。
        """
        if self.dataset is None or episode_idx < 0:
            return None
        try:
            ep_frames = self.dataset.get_episode_frames(episode_idx)
            if ep_frames is None or len(ep_frames) == 0:
                return None

            if "observation.state" not in ep_frames.columns:
                return None

            states = ep_frames["observation.state"].to_numpy()
            states = np.vstack(states)
            if states.shape[1] < 3:
                return None
            n = states.shape[0]

            result = {
                'x': states[:, 0],
                'y': states[:, 1],
                'z': states[:, 2],
            }

            if states.shape[1] >= 8:
                # 旧格式：[x,y,z,qx,qy,qz,qw,gripper_deg] → 转欧拉角(弧度)
                qx = states[:, 3].astype(np.float64)
                qy = states[:, 4].astype(np.float64)
                qz = states[:, 5].astype(np.float64)
                qw = states[:, 6].astype(np.float64)
                rx_arr = np.empty(n); ry_arr = np.empty(n); rz_arr = np.empty(n)
                for i in range(n):
                    rx_arr[i], ry_arr[i], rz_arr[i] = quat_to_rpy_recording(
                        float(qw[i]), float(qx[i]), float(qy[i]), float(qz[i]))
                result['rx'] = rx_arr
                result['ry'] = ry_arr
                result['rz'] = rz_arr
                # 夹爪度→弧度
                result['gripper'] = np.radians(states[:, 7])
            elif states.shape[1] >= 7:
                # 新格式：[x,y,z,rx,ry,rz,gripper_rad] 直接读取
                result['rx'] = states[:, 3]
                result['ry'] = states[:, 4]
                result['rz'] = states[:, 5]
                result['gripper'] = states[:, 6]
            else:
                # 不足 7 列：欧拉角补零
                result['rx'] = np.zeros(n)
                result['ry'] = np.zeros(n)
                result['rz'] = np.zeros(n)
                result['gripper'] = np.zeros(n)

            # 双臂+底盘 18 维：[臂1(7) | 臂2(7) | lift_m, vx, vy, omega]
            if states.shape[1] >= 14:
                result['x2'] = states[:, 7]
                result['y2'] = states[:, 8]
                result['z2'] = states[:, 9]
                result['rx2'] = states[:, 10]
                result['ry2'] = states[:, 11]
                result['rz2'] = states[:, 12]
                result['gripper2'] = states[:, 13]
            if states.shape[1] >= 18:
                result['lift'] = states[:, 14]
                result['vx'] = states[:, 15]
                result['vy'] = states[:, 16]
                result['omega'] = states[:, 17]

            if "timestamp" in ep_frames.columns:
                result['timestamps'] = ep_frames["timestamp"].to_numpy()

            # 夹爪力矩 (Nm)：新录制的数据集含 observation.gripper_torque 列
            # 1 维=单臂；2 维=[臂1, 臂2]
            if "observation.gripper_torque" in ep_frames.columns:
                try:
                    tq = np.vstack(ep_frames["observation.gripper_torque"].to_numpy())
                    if tq.ndim == 1:
                        tq = tq.reshape(-1, 1)
                    result['gripper_torque'] = tq[:, 0].astype(np.float64)
                    if tq.shape[1] >= 2:
                        result['gripper_torque2'] = tq[:, 1].astype(np.float64)
                except Exception:
                    pass
            return result
        except Exception as e:
            print(f"[Trajectory] 提取位姿数据失败: {e}")
            return None
            
    def _load_image_for_preview(self, episode_index, frame_index, camera_name):
        """为预览加载图像，支持三种存储方式：
        1. videos/ 目录下的视频文件
        2. images/ 目录下的 PNG 文件
        3. parquet 中的 bytes 字段
        """
        # 方式1: 优先从 videos/ 目录加载
        if self.dataset.is_video_feature(camera_name):
            try:
                frame = self.dataset.get_video_frame(episode_index, frame_index, camera_name)
                if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                    return frame
            except Exception as e:
                print(f"[Preview] 从视频加载失败: {e}")
        
        # 方式2: 尝试从 images/ 目录加载
        img_path = self.dataset.get_image_path(episode_index, frame_index, camera_name)
        if img_path and os.path.exists(img_path):
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            return np.array(img)
        
        # 方式3: 从 parquet 的 bytes 字段解码
        try:
            ep_frames = self.dataset.get_episode_frames(episode_index)
            if frame_index < len(ep_frames):
                row = ep_frames.iloc[frame_index]
                if camera_name in row:
                    img_data = row[camera_name]
                    if isinstance(img_data, dict):
                        if 'bytes' in img_data and img_data['bytes']:
                            from PIL import Image
                            import io
                            img = Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
                            return np.array(img)
                        elif 'path' in img_data and img_data['path']:
                            from PIL import Image
                            img = Image.open(img_data['path']).convert("RGB")
                            return np.array(img)
        except Exception as e:
            print(f"[Preview] 从 parquet 加载图像失败: {e}")
        
        # 返回黑色图像作为占位
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def _on_image_double_clicked(self, camera_name):
        """双击图像回放框时，弹出对应摄像头当前帧的全分辨率图像。"""
        if self.dataset is None or self._current_episode_idx < 0:
            return

        try:
            img = self._load_image_for_preview(
                self._current_episode_idx, self._current_frame_idx, camera_name
            )
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                QMessageBox.warning(self, "警告", "无法加载当前帧图像")
                return

            cam_display_name = f"摄像头 {camera_name[-1]}" if camera_name[-1].isdigit() else camera_name
            title = (
                f"{cam_display_name} 全分辨率图像 - "
                f"Episode {self._current_episode_idx} / 帧 {self._current_frame_idx}"
            )
            dialog = FullscreenImageDialog(self, img, title)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开全分辨率图像失败:\n{str(e)}")
        
    @staticmethod
    def _format_state_text(state, step, timestamp, task_name):
        """把单帧 state 数组格式化为信息文本。

        兼容格式：7 维单臂欧拉角 / 8 维单臂四元数 / 18 维双臂+底盘
        [臂1(7) | 臂2(7) | lift_m, vx, vy, omega]。
        """
        state = np.asarray(state, dtype=float)
        dim = state.shape[0]
        if dim >= 18:
            # 双臂+底盘格式
            a1, a2 = state[:7], state[7:14]
            lift, vx, vy, omega = state[14], state[15], state[16], state[17]
            return (
                f"帧: {step} | 时间: {timestamp:.3f}s | 任务: {task_name}\n"
                f"臂1 位置: [{a1[0]:.3f}, {a1[1]:.3f}, {a1[2]:.3f}] "
                f"欧拉角: [{a1[3]:.3f}, {a1[4]:.3f}, {a1[5]:.3f}] 夹爪: {a1[6]:.3f}\n"
                f"臂2 位置: [{a2[0]:.3f}, {a2[1]:.3f}, {a2[2]:.3f}] "
                f"欧拉角: [{a2[3]:.3f}, {a2[4]:.3f}, {a2[5]:.3f}] 夹爪: {a2[6]:.3f}\n"
                f"升降(m): {lift:.3f} | 底盘速度 vx/vy/ω: [{vx:.3f}, {vy:.3f}, {omega:.3f}]"
            )
        pos = state[:3]
        if dim >= 8:
            # 旧四元数格式：转欧拉角显示，夹爪度→弧度
            rx, ry, rz = quat_to_rpy_recording(
                float(state[6]), float(state[3]), float(state[4]), float(state[5]))
            grip = math.radians(float(state[7])) if dim > 7 else 0.0
        elif dim >= 7:
            # 新欧拉角格式：直接读取
            rx, ry, rz = float(state[3]), float(state[4]), float(state[5])
            grip = float(state[6])
        else:
            rx = ry = rz = grip = 0.0
        return (
            f"帧: {step} | 时间: {timestamp:.3f}s | 任务: {task_name}\n"
            f"位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n"
            f"欧拉角(rad): [{rx:.3f}, {ry:.3f}, {rz:.3f}]\n"
            f"夹爪(rad): {grip:.3f}"
        )

    def _display_frame(self, rgb_cam1, rgb_cam2, rgb_cam3, action, state, step, timestamp, task_name):
        """显示单帧图像和数据（不依赖播放线程）"""
        self.cam1_label.setImage(rgb_cam1)
        self.cam2_label.setImage(rgb_cam2)
        self.cam3_label.setImage(rgb_cam3)
        self._current_frame_idx = step
        self.frame_info_label.setText(
            self._format_state_text(state, step, timestamp, task_name))

    def _on_frame_data(self, rgb_cam1, rgb_cam2, rgb_cam3, action, state, step, timestamp, task_name):
        self.cam1_label.setImage(rgb_cam1)
        self.cam2_label.setImage(rgb_cam2)
        self.cam3_label.setImage(rgb_cam3)
        # 播放线程推送新帧时同步播放头位置
        if self._current_total_frames > 0:
            self.trim_slider.set_current_frame(step)
        # 自动兼容 7 维欧拉角 / 8 维四元数 / 18 维双臂+底盘格式
        self.frame_info_label.setText(
            self._format_state_text(state, step, timestamp, task_name))
        
    def _update_data_panel(self, step, timestamp, pos, quat, action, task_name=""):
        """更新机械臂数据面板显示（已删除面板，方法保留兼容）"""
        pass
        
    def _on_frame_updated(self, current, total):
        self._current_total_frames = total
        self._current_frame_idx = max(0, current - 1)
        if total > 0:
            self.trim_slider.set_current_frame(current - 1)
            self.progress_label.setText(f"{current} / {total}")
            # 同步更新位置曲线与底盘曲线竖线
            self.trajectory_plot.set_current_frame(current - 1)
            self.chassis_plot.set_current_frame(current - 1)
            # 同步 RECAP 轨道播放头
            self.recap_track.set_current(current - 1)
            
    def _on_episode_finished(self):
        self.trim_slider.set_current_frame(0)
        
    def _on_playback_finished(self):
        """播放自然结束时，恢复到 episode 仍被选中的状态，不用重新去列表里点。"""
        self._restore_after_playback()
        self.status_bar.showMessage("播放完成")

    def _restore_after_playback(self):
        """恢复到当前 Episode 已选择、可立即重新播放/剪辑的状态。"""
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(True)
        self.auto_play_check.setEnabled(True)

        if self._current_episode_idx >= 0 and self.dataset is not None:
            # 保持 episode 选择状态，并启用对应操作按钮
            self.del_btn.setEnabled(True)
            self.trim_btn.setEnabled(True)
            if self._current_total_frames > 0:
                self.trim_slider.setEnabled(True)
                self.progress_label.setText(f"1 / {self._current_total_frames}")
        else:
            self.del_btn.setEnabled(False)
            self.trim_btn.setEnabled(False)
            self.trim_slider.setEnabled(False)
            self.progress_label.setText("0 / 0")

    def _on_status_changed(self, status):
        self.status_bar.showMessage(status)

    def _on_error(self, msg):
        QMessageBox.critical(self, "回放错误", msg)
        self._reset_ui_state()

    def _stop_playback(self):
        if self.playback_thread is not None:
            self.playback_thread.stop()
            self.playback_thread.wait(3000)
            self.playback_thread = None

        self._restore_after_playback()

    def _reset_ui_state(self):
        self.play_btn.setEnabled(True)
        self.pause_btn.setText("⏸ 暂停")
        self.del_btn.setEnabled(False)
        self.auto_play_check.setEnabled(True)
        
        self.trim_slider.set_total_frames(0)
        self.trim_slider.setEnabled(False)
        self.progress_label.setText("0 / 0")
        self.trim_info_label.setText("剪辑区域: 未选择")
        self._current_total_frames = 0
        
        self.cam1_label.setText("摄像头 1 (臂1腕部)")
        self.cam1_label.clearImage()
        self.cam2_label.setText("摄像头 2 (头部)")
        self.cam2_label.clearImage()
        self.cam3_label.setText("摄像头 3 (臂2腕部)")
        self.cam3_label.clearImage()
        self.frame_info_label.setText("帧信息: 未开始播放")

        # 清空位置曲线与底盘曲线
        self.trajectory_plot.clear()
        self.chassis_plot.clear()

        self.status_bar.showMessage("就绪 - 请选择数据集")
        
    def _delete_selected_episode(self):
        """删除选中的 episode（使用后台线程，避免 UI 卡顿）。"""
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return

        selected = self.episode_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "警告", "请先选择一个 Episode")
            return

        episode_idx = int(selected[0].text())
        ep = self.dataset.episodes[episode_idx]
        length = ep.get("length", ep.get("dataset_to_index", 0) - ep.get("dataset_from_index", 0))
        duration = length / self.dataset.fps if self.dataset.fps > 0 else 0

        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要永久删除 Episode {episode_idx} 吗？\n\n"
            f"帧数: {length}\n"
            f"时长: {duration:.1f}秒\n"
            f"是否备份原数据取决于“剪辑前自动备份”复选框；不备份则不可恢复！"
            + self._recap_delete_guard_text(dataset_path),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # 获取数据集路径
        if dataset_path is None:
            return
        recap_backup_annotations(dataset_path)  # P0.3：删除前备份标注（重映射在成功后进行）
        self._pending_dataset_path = dataset_path
        self._pending_episode_idx = episode_idx

        # 禁用所有操作按钮
        self.del_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.episode_table.setEnabled(False)
        self.dataset_combo.setEnabled(False)

        # 显示忙碌进度条
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("正在删除...")
        self.status_bar.showMessage(f"正在删除 Episode {episode_idx}，请稍候...")

        # 启动后台线程
        self.delete_thread = DeleteEpisodeThread(
            dataset_path, episode_idx,
            backup_enabled=self.backup_before_trim_check.isChecked())
        self.delete_thread.finished_ok.connect(self._on_delete_finished)
        self.delete_thread.error.connect(self._on_delete_error)
        self.delete_thread.start()

    def _batch_delete_episodes(self):
        """批量删除 episode：对话框多选/任务筛选，后台线程单次重写数据集。"""
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return

        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        if dataset_path is None:
            return

        # 组装对话框行数据：(episode_index, length, task_name)
        rows = []
        for i, ep in enumerate(self.dataset.episodes):
            ep_idx = int(ep.get("episode_index", i))
            length = int(ep.get(
                "length", ep.get("dataset_to_index", 0) - ep.get("dataset_from_index", 0)))
            rows.append((ep_idx, length, _episode_task_display_name(self.dataset, ep)))

        dlg = BatchDeleteDialog(self, rows, self.dataset.fps)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        indices = dlg.selected_indices()
        if not indices:
            return
        if len(indices) >= len(self.dataset.episodes):
            QMessageBox.warning(self, "警告", "不能删除全部 episode（至少保留 1 个）")
            return

        length_map = {r[0]: r[1] for r in rows}
        total_frames = sum(length_map[i] for i in indices)
        duration = total_frames / self.dataset.fps if self.dataset.fps > 0 else 0
        shown = indices if len(indices) <= 20 else indices[:20]
        idx_text = ", ".join(str(i) for i in shown)
        if len(indices) > 20:
            idx_text += f", ... 共 {len(indices)} 个"

        reply = QMessageBox.question(
            self, "确认批量删除",
            f"确定要永久删除以下 {len(indices)} 个 Episode 吗？\n\n"
            f"编号: {idx_text}\n"
            f"总帧数: {total_frames}\n"
            f"总时长: {duration:.1f}秒\n"
            f"是否备份原数据取决于“剪辑前自动备份”复选框；不备份则不可恢复！"
            + self._recap_delete_guard_text(dataset_path),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        recap_backup_annotations(dataset_path)  # P0.3：删除前备份标注（重映射在成功后进行）

        self._pending_dataset_path = dataset_path
        self._pending_episode_indices = indices

        # 禁用所有操作按钮（剪辑类按钮与删除共用数据集文件，必须互斥）
        self.del_btn.setEnabled(False)
        self.batch_del_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.trim_btn.setEnabled(False)
        self.trajectory_plot.trim_stationary_btn.setEnabled(False)
        self.trajectory_plot.trim_all_stationary_btn.setEnabled(False)
        self.trajectory_plot.jitter_trim_btn.setEnabled(False)
        self.trajectory_plot.trim_all_jitter_btn.setEnabled(False)
        self.episode_table.setEnabled(False)
        self.dataset_combo.setEnabled(False)

        # 显示忙碌进度条
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("正在批量删除...")
        self.status_bar.showMessage(f"正在批量删除 {len(indices)} 个 Episode，请稍候...")

        # 启动后台线程
        self.batch_delete_thread = BatchDeleteEpisodesThread(
            dataset_path, indices,
            backup_enabled=self.backup_before_trim_check.isChecked())
        self.batch_delete_thread.finished_ok.connect(self._on_batch_delete_finished)
        self.batch_delete_thread.error.connect(self._on_delete_error)
        self.batch_delete_thread.start()

    def _on_batch_delete_finished(self, result):
        """批量删除成功回调"""
        indices = self._pending_episode_indices
        dataset_path = self._pending_dataset_path

        # 重新加载数据集
        self.status_bar.showMessage("正在重新加载数据集...")
        try:
            self.dataset = OfflineDataset(dataset_path)
            self._preview_frame_cache.clear()  # 删除后索引重排，缓存全部失效
            self._update_episode_table()
            self._update_info()
        except Exception as e:
            QMessageBox.warning(self, "重新加载警告", f"数据集重新加载失败: {e}\n请手动刷新。")

        # P0.3：标注键随删除自动重映射（被删 episode 的标注一并移除）
        self._recap_remap_after_delete(indices)

        # 恢复进度条
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0 / 0")

        # 恢复状态显示
        self.status_bar.showMessage(f"已批量删除 {len(indices)} 个 Episode")

        # 删除改变了总帧数且标注键已重映射落盘：自动同步训练 sidecar（若有）
        self._refresh_sidecar_after_edit(dataset_path)

        QMessageBox.information(self, "批量删除成功", result)

        # 重置选择
        self.current_episode_label.setText("当前未选择 Episode")

        # 恢复控件
        self._restore_ui_after_delete()

    def _on_delete_finished(self, result):
        """删除成功回调"""
        episode_idx = self._pending_episode_idx
        dataset_path = self._pending_dataset_path

        # 重新加载数据集
        self.status_bar.showMessage("正在重新加载数据集...")
        try:
            self.dataset = OfflineDataset(dataset_path)
            self._preview_frame_cache.clear()  # 删除后索引重排，缓存全部失效
            self._update_episode_table()
            self._update_info()
        except Exception as e:
            QMessageBox.warning(self, "重新加载警告", f"数据集重新加载失败: {e}\n请手动刷新。")

        # P0.3：标注键随删除自动重映射（被删 episode 的标注一并移除）
        self._recap_remap_after_delete([episode_idx])

        # 恢复进度条
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0 / 0")

        # 恢复状态显示
        self.status_bar.showMessage(f"已删除 Episode {episode_idx}")

        # 删除改变了总帧数且标注键已重映射落盘：自动同步训练 sidecar（若有）
        self._refresh_sidecar_after_edit(dataset_path)

        QMessageBox.information(self, "删除成功", result)

        # 重置选择
        self.current_episode_label.setText("当前未选择 Episode")

        # 恢复控件
        self._restore_ui_after_delete()

    def _on_delete_error(self, error_msg):
        """删除失败回调"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0 / 0")
        self.status_bar.showMessage("删除失败")
        QMessageBox.critical(self, "删除失败", error_msg)

        self._restore_ui_after_delete()

    def _restore_ui_after_delete(self):
        """删除操作完成后恢复 UI 控件状态。"""
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.episode_table.setEnabled(True)
        self.dataset_combo.setEnabled(True)
        self.trim_btn.setEnabled(False)
        self.trim_slider.set_total_frames(0)
        self.trim_slider.setEnabled(False)
        # 批量删除按钮只依赖"数据集已加载"，删除完成重新加载后恢复
        self.batch_del_btn.setEnabled(
            self.dataset is not None and len(self.dataset.episodes) > 0)
        # 剪辑静止按钮常驻于分析面板，删除完成后恢复
        self.trajectory_plot.trim_stationary_btn.setEnabled(True)
        self.trajectory_plot.trim_all_stationary_btn.setEnabled(True)
        self.trajectory_plot.jitter_trim_btn.setEnabled(True)
        self.trajectory_plot.trim_all_jitter_btn.setEnabled(True)
        # del_btn, pause_btn 保持禁用，因为没有选中 episode 或没有播放

    def closeEvent(self, event):
        # 如果任务名称有未保存修改，提示用户
        if self._check_unsaved_task_change():
            event.ignore()
            return
        self._stop_playback()
        if self.dataset is not None:
            try:
                self.dataset.release_video_caps()
            except Exception:
                pass
        event.accept()

    def _check_unsaved_task_change(self):
        """检查是否有未保存的任务名称修改，若有则提示并返回 True。"""
        if self._current_episode_idx < 0 or self.dataset is None:
            return False

        current_text = self.task_name_edit.text().strip()
        if current_text != self._original_task_name:
            reply = QMessageBox.question(
                self, "未保存的修改",
                f"任务名称已被修改，是否保存为 \"{current_text}\"？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._save_task_name(self._current_episode_idx, current_text)
            elif reply == QMessageBox.StandardButton.Cancel:
                return True
            # No: 放弃修改，继续
        return False

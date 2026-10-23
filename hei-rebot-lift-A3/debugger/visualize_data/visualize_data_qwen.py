#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot 数据可视化工具 (PyQt6 界面版) - 离线本地数据加载

功能:
  1. 启动后显示已有数据集列表
  2. 选择数据集后显示 Episode 列表
  3. 显示数据集总信息
  4. 选择 Episode 后播放/暂停/重新播放
  5. 支持自动播放所有 episode

模块结构（2026-08-29 按功能拆分，原单文件逻辑不变）:
  vd_common.py       路径初始化 / 日志 / matplotlib 中文字体 / IK 逆运动学
  vd_dataset.py      OfflineDataset 离线数据集加载
  vd_widgets.py      通用 UI 控件（剪辑滑条 / RECAP 轨道 / 图像标签 / 弹窗）
  vd_trajectory.py   轨迹曲线面板（标注 / 合并分页）
  vd_data_ops.py     数据集磁盘操作（删除 / 剪辑 / 降噪 / 一致性检查）
  vd_threads.py      后台工作线程
  vd_dialogs.py      统计 / 检查 / 批量删除对话框
  vd_robot3d.py      机械臂 3D 回放面板（已不在主窗口使用，保留模块）
  vd_main_window.py  主窗口（双臂+底盘+3 路相机；臂2 曲线以虚线与臂1 同图显示）

用法:
  python3 visualize_data_qwen.py --root ../lerobot_data
  （不传 --root 时默认使用 debugger/lerobot_data）
"""

import sys
from PyQt6.QtWidgets import QApplication

from vd_common import _DEFAULT_DATA_ROOT
from vd_main_window import MainWindow

# ---- 向后兼容：原单文件的所有顶层定义仍可从本模块直接访问 ----
from vd_common import ARM_JOINT_NAMES, _DBG_DIR, _DEBUGGER_DIR, _DEFAULT_DATA_ROOT, _KIN, _KIN_LOCK, _SDK_ROOT, _setup_matplotlib_chinese_font, logger, quat_to_rpy_recording, solve_episode_joint_angles
from vd_dataset import OfflineDataset, EpisodeSampler
from vd_widgets import TrimSlider, RecapTrackWidget, ImageLabel, FullscreenImageDialog
from vd_trajectory import detect_stationary_intervals, PRESET_LABELS, TrajectoryPlotWidget
from vd_data_ops import _read_episodes, _write_parquet_chunks, _get_dataset_fps, _reset_frame_columns, _sync_episode_data_pointers, _atomic_write_parquet, check_dataset_consistency, _assert_dataset_consistent, _gc_orphan_tasks, _recompute_global_stats, _sync_episode_parquet_stats_counts, _update_video_timestamps_after_trim, _remove_files, _reindex_video_files, _update_stats_json, delete_episode, _update_stats_json_batch, _reindex_video_files_batch, delete_episodes, _episode_task_display_name, _get_video_files_for_episode, _find_video_files_for_trim, _backup_for_trim, trim_episode, _update_episode_stats_for_trim, _trim_video_files, _trim_video_files_multiple, trim_episode_ranges, _smooth_1d, _smooth_pose_matrix, denoise_dataset
from vd_threads import DeleteEpisodeThread, BatchDeleteEpisodesThread, TrimEpisodeThread, TrimStationaryThread, DenoiseThread, IKSolveThread, PlaybackThread, _VlmLabelThread
from vd_dialogs import BatchDeleteDialog, DatasetStatsDialog, DatasetCheckDialog
from vd_robot3d import RobotArmReplayWidget
from vd_main_window import MainWindow


# ==================== 入口 ====================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="LeRobot 数据可视化工具")
    parser.add_argument("--root", type=str, default=_DEFAULT_DATA_ROOT, help="数据集根目录")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = MainWindow(root=args.root)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

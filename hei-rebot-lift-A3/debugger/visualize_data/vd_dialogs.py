# -*- coding: utf-8 -*-
"""对话框：批量删除 Episode 对话框、数据集统计分布分析对话框、数据集 8 类问题检查对话框。"""
from vd_common import quat_to_rpy_recording

import glob
import json
import os
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QHeaderView, QTableWidget, QTableWidgetItem, QComboBox, QLineEdit, QSizePolicy, QDialog, QTextEdit, QAbstractItemView
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ==================== 批量删除 Episode 对话框 ====================
class BatchDeleteDialog(QDialog):
    """批量 Episode 选择对话框：任务名筛选 + 复选框多选 + 汇总确认。

    默认用作“批量删除片段”的删除确认对话框；通过 title/ok_text/ok_color
    参数复用于“剪辑所有静止”的 Episode 筛选（同一套筛选交互）。

    通过 selected_indices() 返回选中的 episode_index 列表（已按升序排列）。
    """

    def __init__(self, parent, episode_rows, fps, title="批量删除片段",
                 ok_text="🗑 删除选中", ok_color="#9C27B0"):
        """
        Args:
            episode_rows: [(episode_index, length, task_name), ...]
            fps: 数据集帧率（时长列显示用）
            title: 对话框标题
            ok_text: 确认按钮文字（不含数量后缀）
            ok_color: 确认按钮背景色
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 540)
        self._rows = list(episode_rows)
        self._fps = fps if fps and fps > 0 else 20.0
        self._ok_text = ok_text

        layout = QVBoxLayout(self)

        # ---- 筛选行：任务名下拉 + 关键词 ----
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("任务筛选:"))
        self.task_filter_combo = QComboBox()
        self.task_filter_combo.addItem("全部任务")
        for name in sorted({r[2] for r in self._rows if r[2]}):
            self.task_filter_combo.addItem(name)
        self.task_filter_combo.currentTextChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.task_filter_combo, stretch=1)

        filter_layout.addWidget(QLabel("关键词:"))
        self.keyword_edit = QLineEdit()
        self.keyword_edit.setPlaceholderText("按编号或任务名过滤…")
        self.keyword_edit.textChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.keyword_edit, stretch=1)
        layout.addLayout(filter_layout)

        # ---- 快捷选择按钮 ----
        sel_layout = QHBoxLayout()
        self.select_filtered_btn = QPushButton("☑ 勾选筛选结果")
        self.select_filtered_btn.setToolTip("勾选当前可见（未被过滤）的全部 Episode")
        self.select_filtered_btn.clicked.connect(lambda: self._set_visible_checked(True))
        sel_layout.addWidget(self.select_filtered_btn)

        self.unselect_filtered_btn = QPushButton("☐ 取消勾选筛选结果")
        self.unselect_filtered_btn.setToolTip("取消勾选当前可见的全部 Episode")
        self.unselect_filtered_btn.clicked.connect(lambda: self._set_visible_checked(False))
        sel_layout.addWidget(self.unselect_filtered_btn)

        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        sel_layout.addWidget(self.select_all_btn)

        self.select_none_btn = QPushButton("全不选")
        self.select_none_btn.clicked.connect(lambda: self._set_all_checked(False))
        sel_layout.addWidget(self.select_none_btn)
        sel_layout.addStretch()
        layout.addLayout(sel_layout)

        # ---- Episode 复选表格 ----
        self.table = QTableWidget(len(self._rows), 5)
        self.table.setHorizontalHeaderLabels(["勾选", "Episode", "帧数", "时长(s)", "任务名"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)

        for row_i, (ep_idx, length, task_name) in enumerate(self._rows):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            chk.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row_i, 0, chk)

            idx_item = QTableWidgetItem(str(ep_idx))
            idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_i, 1, idx_item)

            len_item = QTableWidgetItem(str(length))
            len_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_i, 2, len_item)

            dur_item = QTableWidgetItem(f"{length / self._fps:.1f}")
            dur_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_i, 3, dur_item)

            self.table.setItem(row_i, 4, QTableWidgetItem(task_name))

        self.table.itemChanged.connect(self._update_summary)
        layout.addWidget(self.table, stretch=1)

        # ---- 汇总信息 + 确认/取消 ----
        bottom_layout = QHBoxLayout()
        self.summary_label = QLabel("已选 0 个 Episode（0 帧，0.0 秒）")
        self.summary_label.setStyleSheet("QLabel { color: #c62828; font-weight: bold; }")
        bottom_layout.addWidget(self.summary_label)
        bottom_layout.addStretch()

        self.ok_btn = QPushButton(f"{self._ok_text} (0)")
        self.ok_btn.setEnabled(False)
        self.ok_btn.setStyleSheet(
            f"QPushButton {{ background-color: {ok_color}; color: white; font-weight: bold; padding: 6px 16px; }}"
            "QPushButton:disabled { background-color: #ccc; color: #888; }"
        )
        self.ok_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(self.ok_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet("QPushButton { padding: 6px 16px; }")
        cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(cancel_btn)
        layout.addLayout(bottom_layout)

        self._update_summary()

    def _apply_filter(self):
        """按任务名下拉 + 关键词过滤表格行（隐藏不匹配行，不改动勾选状态）。"""
        task = self.task_filter_combo.currentText()
        kw = self.keyword_edit.text().strip().lower()
        for row_i, (ep_idx, _length, task_name) in enumerate(self._rows):
            visible = True
            if task != "全部任务" and task_name != task:
                visible = False
            if kw and kw not in str(ep_idx) and kw not in task_name.lower():
                visible = False
            self.table.setRowHidden(row_i, not visible)

    def _set_visible_checked(self, checked):
        """勾选/取消勾选当前可见（未被过滤）的行。"""
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row_i in range(len(self._rows)):
            if self.table.isRowHidden(row_i):
                continue
            item = self.table.item(row_i, 0)
            if item is not None:
                item.setCheckState(state)

    def _set_all_checked(self, checked):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row_i in range(len(self._rows)):
            item = self.table.item(row_i, 0)
            if item is not None:
                item.setCheckState(state)

    def _update_summary(self):
        indices = set(self.selected_indices())
        n_frames = sum(length for ep_idx, length, _t in self._rows
                       if ep_idx in indices)
        duration = n_frames / self._fps
        self.summary_label.setText(
            f"已选 {len(indices)} 个 Episode（{n_frames} 帧，{duration:.1f} 秒）")
        self.ok_btn.setText(f"{self._ok_text} ({len(indices)})")
        self.ok_btn.setEnabled(len(indices) > 0)

    def selected_indices(self):
        """返回勾选的 episode_index 列表（升序）。"""
        result = []
        for row_i, (ep_idx, _length, _task) in enumerate(self._rows):
            item = self.table.item(row_i, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                result.append(ep_idx)
        return sorted(result)


# ==================== 数据集统计分布分析对话框 ====================
class DatasetStatsDialog(QDialog):
    """分析所选数据集所有 Episode 的位置 xyz、欧拉角、夹爪角度分布。

    以 Episode 编号为横轴，每个指标绘制 min~max 范围带与均值曲线，
    超出全局 均值±3σ 的 Episode 标记为异常值，便于发现极值与异常轨迹。
    自动兼容 7 维欧拉角新格式与 8 维四元数旧格式（旧格式自动转欧拉角）。
    """

    def __init__(self, parent, dataset):
        super().__init__(parent)
        self.dataset = dataset
        self.setWindowTitle("数据集统计分布分析")
        self.resize(1200, 900)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hint = QLabel(
            "横轴为 Episode 编号；每幅子图灰色带表示该指标在该 Episode 内的最小值~最大值范围，"
            "蓝色实线为均值；红色点为异常值（该 Episode 均值偏离全局均值超过 3σ）。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #555; font-size: 11px; padding: 2px; }")
        layout.addWidget(hint)

        self.figure = Figure(figsize=(11, 9), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas, stretch=5)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(180)
        layout.addWidget(self.summary_text, stretch=1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet("QPushButton { padding: 6px 28px; }")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self._compute_and_plot()

    def _compute_and_plot(self):
        n_eps = len(self.dataset.episodes)
        if n_eps == 0:
            self.summary_text.setText("数据集没有 Episode。")
            return

        df = self.dataset.data_df
        if "observation.state" not in df.columns:
            self.summary_text.setText("数据中没有 observation.state 列，无法分析。")
            return

        try:
            states = np.vstack(df["observation.state"].to_numpy()).astype(np.float64)
        except Exception as e:
            self.summary_text.setText(f"解析 observation.state 失败: {e}")
            return
        if states.ndim != 2 or states.shape[1] < 3:
            self.summary_text.setText("observation.state 列数不足，无法分析。")
            return

        n_rows = states.shape[0]
        zeros = np.zeros(n_rows)
        # 自动识别格式：>=8 列为旧四元数格式，==7 为新欧拉角格式
        if states.shape[1] >= 8:
            qx = states[:, 3].astype(np.float64)
            qy = states[:, 4].astype(np.float64)
            qz = states[:, 5].astype(np.float64)
            qw = states[:, 6].astype(np.float64)
            rx_c = np.empty(n_rows); ry_c = np.empty(n_rows); rz_c = np.empty(n_rows)
            for i in range(n_rows):
                rx_c[i], ry_c[i], rz_c[i] = quat_to_rpy_recording(
                    float(qw[i]), float(qx[i]), float(qy[i]), float(qz[i]))
            metric_cols = {
                'x': states[:, 0],
                'y': states[:, 1],
                'z': states[:, 2],
                'rx': rx_c,
                'ry': ry_c,
                'rz': rz_c,
                'gripper': np.radians(states[:, 7]),
            }
        else:
            metric_cols = {
                'x': states[:, 0],
                'y': states[:, 1],
                'z': states[:, 2],
                'rx': states[:, 3] if states.shape[1] > 3 else zeros,
                'ry': states[:, 4] if states.shape[1] > 4 else zeros,
                'rz': states[:, 5] if states.shape[1] > 5 else zeros,
                'gripper': states[:, 6] if states.shape[1] > 6 else zeros,
            }

        if "episode_index" in df.columns:
            ep_col = df["episode_index"].to_numpy()
        else:
            ep_col = np.zeros(n_rows, dtype=int)

        # 按 episode 排序后分组，避免逐 episode 全表扫描
        sort_idx = np.argsort(ep_col, kind='stable')
        ep_sorted = ep_col[sort_idx]
        unique_eps, start_idx = np.unique(ep_sorted, return_index=True)
        split_points = start_idx[1:]
        x = np.arange(len(unique_eps))

        metrics = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
        titles = ['位置 X (m)', '位置 Y (m)', '位置 Z (m)', '欧拉角 rx (rad)',
                  '欧拉角 ry (rad)', '欧拉角 rz (rad)', '夹爪角度 (rad)']

        stats = {}
        for m in metrics:
            vals_sorted = metric_cols[m][sort_idx]
            groups = np.split(vals_sorted, split_points)
            stats[m] = {
                'min': np.array([g.min() for g in groups], dtype=float),
                'max': np.array([g.max() for g in groups], dtype=float),
                'mean': np.array([g.mean() for g in groups], dtype=float),
                'std': np.array([g.std() for g in groups], dtype=float),
            }

        # 绘图
        self.figure.clear()
        color_band = '#CFD8DC'
        color_line = '#1565C0'
        color_outlier = '#E53935'

        outlier_eps = set()
        summary_lines = [f"全局统计（共 {len(unique_eps)} 个 Episode，{n_rows} 帧）：\n"]

        for idx, m in enumerate(metrics):
            ax = self.figure.add_subplot(4, 2, idx + 1)
            s = stats[m]
            ax.fill_between(x, s['min'], s['max'], color=color_band, alpha=0.7, label='min~max 范围')
            ax.plot(x, s['mean'], color=color_line, linewidth=1.2, marker='.', markersize=3, label='均值')

            gmean = float(np.mean(s['mean']))
            gstd = float(np.std(s['mean']))
            thr = 3.0 * max(gstd, 1e-12)
            out_mask = np.abs(s['mean'] - gmean) > thr
            if np.any(out_mask):
                ax.scatter(x[out_mask], s['mean'][out_mask], color=color_outlier,
                           s=35, zorder=5, label='异常值')
                for oi in np.where(out_mask)[0]:
                    outlier_eps.add(int(unique_eps[oi]))

            ax.set_title(titles[idx], fontsize=10, fontweight='bold')
            ax.set_xlabel('Episode 编号', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='best', fontsize=7)
            ax.tick_params(labelsize=8)

            # 汇总：全局最小/最大及其所在 episode
            gmin = float(np.min(s['min']))
            gmax = float(np.max(s['max']))
            gmin_ep = int(unique_eps[int(np.argmin(s['min']))])
            gmax_ep = int(unique_eps[int(np.argmax(s['max']))])
            out_list = [str(int(unique_eps[oi])) for oi in np.where(out_mask)[0]]
            out_str = ','.join(out_list)
            line = (f"• {titles[idx]}: 全局最小 {gmin:.4f} (EP#{gmin_ep}), "
                    f"全局最大 {gmax:.4f} (EP#{gmax_ep}), "
                    f"均值 {gmean:.4f}, 标准差 {gstd:.4f}")
            if out_str:
                line += f"  → 异常EP: {out_str}"
            summary_lines.append(line)

        self.figure.tight_layout()
        self.canvas.draw_idle()

        summary_lines.append("")
        if outlier_eps:
            summary_lines.append(f"异常 Episode 汇总: {sorted(outlier_eps)}")
            summary_lines.append("提示：以上 Episode 的某个指标均值显著偏离整体分布，建议重点核查。")
        else:
            summary_lines.append("未检测到异常 Episode（各指标均值均在 3σ 范围内）。")
        self.summary_text.setText('\n'.join(summary_lines))


# ==================== 数据集 7 类问题检查对话框 ====================
class DatasetCheckDialog(QDialog):
    """检查数据集是否存在 数据问题分析.md 中记录的 7 类问题 + RECAP 标注一致性，给出能否训练的结论。

    问题1: meta/episodes 的 data 文件映射失效
    问题2: 视频 to_timestamp 与实际视频时长不符
    问题3: data parquet 中 frame_index / timestamp 列错乱
    问题4: meta/stats.json 统计量不一致（count/维度）
    问题5: action 与 observation.state 完全相同（恒等映射）
    问题6: meta/episodes 与 data 数据行不一致（编辑中断残留）
    问题7: tasks.parquet 的 task_index 非 0 基连续（任务合并后未压实）
    问题8: RECAP 标注一致性与错位（标注键越界/区间越界/重叠/字段非法/
           sidecar 过期或行数不符——剪辑、删除、合并后未重映射/未重新展开的特征）
    """

    def __init__(self, parent, dataset):
        super().__init__(parent)
        self.dataset = dataset
        self.root = str(dataset.root)
        self.fps = dataset.fps
        self.setWindowTitle("数据集检查报告")
        self.resize(720, 600)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hint = QLabel("对照 数据问题分析.md 的 7 类问题 + RECAP 标注一致性逐项检查，判断数据集是否可用于 VLA 训练。")
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #555; font-size: 11px; padding: 2px; }")
        layout.addWidget(hint)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setStyleSheet(
            "QTextEdit { font-family: monospace; font-size: 12px; padding: 6px; }"
        )
        layout.addWidget(self.report_text, stretch=1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet("QPushButton { padding: 6px 28px; }")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self._run_checks()

    def _run_checks(self):
        import subprocess
        ds = self.dataset
        df = ds.data_df
        N = len(df)
        fps = self.fps
        lines = []
        lines.append(f"数据集: {self.root}")
        lines.append(f"总帧数: {N}  |  Episodes: {len(ds.episodes)}  |  FPS: {fps}")
        lines.append(f"state 维度: {ds.features.get('observation.state', {}).get('shape', '?')}")
        lines.append("=" * 64)

        states = np.vstack(df["observation.state"].to_numpy()) if "observation.state" in df.columns else np.zeros((0, 1))
        actions = np.vstack(df["action"].to_numpy()) if "action" in df.columns else np.zeros((0, 1))
        ep_col = df["episode_index"].to_numpy() if "episode_index" in df.columns else np.zeros(N, dtype=int)
        unique_eps = sorted(np.unique(ep_col))

        results = []

        # ---- 问题1: data 文件映射 ----
        lines.append("\n【问题1】meta/episodes 的 data 文件映射")
        ep_meta_files = sorted(glob.glob(str(ds.meta_dir / "episodes" / "**" / "*.parquet"), recursive=True))
        ok1 = True; bad1 = []
        if ep_meta_files:
            ep_df = pd.concat([pd.read_parquet(f) for f in ep_meta_files], ignore_index=True)
            for _, r in ep_df.iterrows():
                ci = int(r["data/chunk_index"]); fi = int(r["data/file_index"])
                p = ds.data_dir / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
                if not p.is_file():
                    ok1 = False; bad1.append(f"ep{int(r['episode_index'])}→file-{fi:03d}")
        if ok1:
            lines.append("  ✓ 全部 episode 的 data/chunk_index+file_index 指向的文件均存在")
        else:
            lines.append(f"  ✗ 失效映射: {', '.join(bad1[:5])}")
        results.append(ok1)

        # ---- 问题2: 视频帧数 / to_timestamp ----
        lines.append("\n【问题2】视频帧数 vs meta length + to_timestamp")
        ok2 = True; vid_details = []
        def _ffprobe_nb(path):
            try:
                r = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                     "-show_entries", "stream=nb_read_frames",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                    capture_output=True, text=True, timeout=120)
                return int(r.stdout.strip())
            except Exception:
                return None
        if ep_meta_files:
            for _, r in ep_df.iterrows():
                ep = int(r["episode_index"]); length = int(r["length"])
                for cam in ["cam1", "cam2"]:
                    col_ci = f"videos/observation.images.{cam}/chunk_index"
                    col_fi = f"videos/observation.images.{cam}/file_index"
                    col_ts = f"videos/observation.images.{cam}/to_timestamp"
                    if col_ci not in r.index:
                        continue
                    ci = int(r[col_ci]); fi = int(r[col_fi])
                    vp = ds.videos_dir / f"observation.images.{cam}" / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
                    if not vp.is_file():
                        ok2 = False; vid_details.append(f"  ✗ ep{ep} {cam}: {vp.name} 不存在")
                        continue
                    nb = _ffprobe_nb(vp)
                    to_ts = float(r[col_ts]) if col_ts in r.index else None
                    frame_ok = (nb == length) if nb is not None else True
                    ts_ok = (to_ts is not None) and (abs(to_ts - length / fps) < 0.05)
                    if not (frame_ok and ts_ok):
                        ok2 = False
                        vid_details.append(
                            f"  ✗ ep{ep} {cam}: frames={nb} vs length={length} "
                            f"{'✓' if frame_ok else '✗'} | to_ts={to_ts} vs {length/fps:.2f} {'✓' if ts_ok else '✗'}")
                    # 只打印前 3 个正常的 + 全部异常的
                    elif len([d for d in vid_details if '✗' in d]) == 0 and ep in unique_eps[:3] and cam == "cam1":
                        vid_details.append(f"  ✓ ep{ep} {cam}: frames={nb}={length}, to_ts={to_ts:.2f}")
        if ok2:
            lines.append("  ✓ 全部视频帧数与 meta length 一致，to_timestamp=length/fps")
        else:
            lines.append("  ✗ 存在不一致（详见下方）")
        lines.extend(vid_details[:8])
        if len(vid_details) > 8:
            lines.append(f"  ... 共 {len(vid_details)} 条详情")
        results.append(ok2)

        # ---- 问题3: frame_index / timestamp ----
        lines.append("\n【问题3】frame_index / timestamp 列规整")
        ok3 = True; bad3 = []
        for ep in unique_eps:
            sub = df[ep_col == ep]
            fi = sub["frame_index"].to_numpy() if "frame_index" in sub.columns else None
            ts = sub["timestamp"].to_numpy() if "timestamp" in sub.columns else None
            length = len(sub)
            if fi is not None and not np.array_equal(fi, np.arange(length)):
                ok3 = False; bad3.append(f"ep{ep}: frame_index 错乱 [{fi.min()}~{fi.max()}] 应[0~{length-1}]")
            if ts is not None and not np.allclose(ts, np.arange(length) / fps, atol=1e-6):
                ok3 = False; bad3.append(f"ep{ep}: timestamp 错乱")
        if ok3:
            lines.append(f"  ✓ 全部 {len(unique_eps)} 个 episode 的 frame_index/timestamp 规整")
        else:
            lines.append(f"  ✗ {len(bad3)} 个 episode 错乱:")
            for b in bad3[:5]:
                lines.append(f"    {b}")
        results.append(ok3)

        # ---- 问题4: stats 一致性 ----
        lines.append("\n【问题4】stats.json count 一致性")
        ok4 = True; bad4 = []
        # 注意：image/video 特征的 count 是“像素采样数”（LeRobot 对下采样后的像素做统计），
        # 不等于帧数 N；只有标量特征（state/action/timestamp/frame_index 等）的 count 才应 == N。
        # 因此对 image/video 特征只校验统计存在(>0)，不与 N 比较，避免误报。
        for key in ["observation.state", "action", "observation.images.cam1",
                     "observation.images.cam2", "timestamp", "frame_index"]:
            c = ds.stats.get(key, {}).get("count", [0])
            c = c[0] if isinstance(c, list) else c
            dtype = ds.features.get(key, {}).get("dtype", "")
            if dtype in ("image", "video"):
                if not c or int(c) <= 0:
                    ok4 = False; bad4.append(f"{key}: count={c}（图像统计缺失）")
            else:
                if int(c) != N:
                    ok4 = False; bad4.append(f"{key}: count={c} ≠ N={N}")
        dim_info = ds.features.get("observation.state", {}).get("shape", [0])[0]
        dim_actual = states.shape[1] if states.size else 0
        if int(dim_info) != dim_actual:
            ok4 = False; bad4.append(f"state维度: info={dim_info} ≠ 实际={dim_actual}")
        if ok4:
            lines.append(f"  ✓ 标量特征 count={N}；图像特征 count 为像素采样数（正常）；维度一致(dim={dim_actual})")
        else:
            for b in bad4:
                lines.append(f"  ✗ {b}")
        results.append(ok4)

        # ---- 问题5: 恒等映射 ----
        lines.append("\n【问题5】action == observation.state 恒等映射")
        diff = np.abs(actions - states).max(axis=1) if states.size else np.array([0])
        match = 0; tot = 0
        for ep in unique_eps:
            mask = ep_col == ep
            s = states[mask]; a = actions[mask]
            if len(s) > 1:
                d = np.abs(a[:-1] - s[1:]).max(axis=1)
                match += int((d < 1e-6).sum()); tot += len(d)
        identity = diff.max() < 1e-6 if len(diff) else True
        pct = 100 * match / max(tot, 1)
        lines.append(f"  max|a-s|={diff.max():.4f}, mean|a-s|={diff.mean():.6f}")
        lines.append(f"  action[t]==state[t+1]: {match}/{tot} ({pct:.1f}%)")
        if identity:
            lines.append("  ✗ 存在 action==state 恒等映射！模型会退化为忽略视觉")
        else:
            lines.append("  ✓ 无恒等映射问题")
        results.append(not identity)
        # 静止帧统计
        static_pct = 100 * (diff < 1e-6).mean()
        lines.append(f"  静止帧占比: {static_pct:.1f}% {'(偏高，建议裁剪)' if static_pct > 30 else ''}")

        # ---- 问题6: meta/episodes 与 data 行一致性（编辑中断残留）----
        lines.append("\n【问题6】meta/episodes 与 data 行一致性")
        ok6 = True
        bad6 = []
        if ep_meta_files and "episode_index" in df.columns:
            data_counts = pd.Series(ep_col).value_counts().to_dict()
            data_counts = {int(k): int(v) for k, v in data_counts.items()}
            if "episode_index" in ep_df.columns:
                meta_eps = [int(r["episode_index"]) for _, r in ep_df.iterrows()]
            else:
                meta_eps = list(range(len(ep_df)))
            meta_set = set(meta_eps)

            phantom = [e for e in meta_eps if e not in data_counts]
            if phantom:
                ok6 = False
                bad6.append(f"meta 有但 data 无数据行的 episode（幻影）: "
                            f"{phantom[:8]}{' ...' if len(phantom) > 8 else ''}")
            orphan = [e for e in unique_eps if e not in meta_set]
            if orphan:
                ok6 = False
                bad6.append(f"data 有但 meta 缺失的 episode_index（孤儿行）: "
                            f"{[int(e) for e in orphan[:8]]}{' ...' if len(orphan) > 8 else ''}")

            len_bad = []
            for _, r in ep_df.iterrows():
                e = int(r["episode_index"]) if "episode_index" in ep_df.columns else None
                if e in data_counts and int(r["length"]) != data_counts[e]:
                    len_bad.append(f"ep{e}: meta={int(r['length'])} vs data={data_counts[e]}")
            if len_bad:
                ok6 = False
                bad6.append(f"length 与 data 行数不符（{len(len_bad)} 个）: "
                            + "; ".join(len_bad[:5]) + (" ..." if len(len_bad) > 5 else ""))

            # dataset_from/to_index 连续性：from[0]==0、首尾相接、长度匹配、末尾==N
            if "dataset_from_index" in ep_df.columns and "dataset_to_index" in ep_df.columns:
                ptr = ep_df[["dataset_from_index", "dataset_to_index", "length"]].to_numpy(dtype=np.int64)
                ptr = ptr[np.argsort(ptr[:, 0])]
                contig = (int(ptr[0, 0]) == 0
                          and bool((ptr[1:, 0] == ptr[:-1, 1]).all())
                          and bool((ptr[:, 1] - ptr[:, 0] == ptr[:, 2]).all()))
                if not contig or int(ptr[-1, 1]) != N:
                    ok6 = False
                    bad6.append(f"dataset_from/to_index 不连续或与实际帧数不符"
                                f"（末尾 to={int(ptr[-1, 1])}，data 实际 {N} 行）")
        elif not ep_meta_files:
            lines.append("  （无 meta/episodes parquet，跳过指针校验）")

        info_tf = ds.info.get("total_frames")
        info_te = ds.info.get("total_episodes")
        if info_tf is not None and int(info_tf) != N:
            ok6 = False
            bad6.append(f"info.json total_frames={info_tf} ≠ data 实际 {N}")
        if ep_meta_files and info_te is not None and int(info_te) != len(ep_df):
            ok6 = False
            bad6.append(f"info.json total_episodes={info_te} ≠ meta 实际 {len(ep_df)}")

        if ok6:
            lines.append("  ✓ meta/data/info 三方一致（无编辑中断残留）")
        else:
            for b in bad6:
                lines.append(f"  ✗ {b}")
            lines.append("  → 该不一致会导致修剪/删除报错（_sync_episode_data_pointers 拦截），")
            lines.append("    继续编辑会加重错位，请先修复（见 数据问题分析.md 问题6）")
        results.append(ok6)

        # ---- 问题7: task_index 0基连续性（任务合并删除行后未压实）----
        lines.append("\n【问题7】task_index 0基连续性")
        ok7 = True; bad7 = []
        n_tasks = 0; tasks_idx_vals = []; task_names = []
        tasks_path = ds.meta_dir / "tasks.parquet"
        if tasks_path.is_file():
            tdf = pd.read_parquet(tasks_path)
            n_tasks = len(tdf)
            task_names = [str(i) for i in tdf.index.tolist()]
            if "task_index" in tdf.columns:
                tasks_idx_vals = sorted(int(v) for v in tdf["task_index"].tolist())
                expect = list(range(n_tasks))
                if tasks_idx_vals != expect:
                    ok7 = False
                    bad7.append(f"tasks.parquet 的 task_index={tasks_idx_vals} 非 0 基连续"
                                f"（应为 {expect}），lerobot 按位置 iloc 取任务名将越界/错位")
            else:
                ok7 = False
                bad7.append("tasks.parquet 缺少 task_index 列")
        else:
            ok7 = False
            bad7.append("meta/tasks.parquet 不存在")
        if "task_index" in df.columns and N > 0:
            ti = df["task_index"].to_numpy(dtype=np.int64)
            ti_min, ti_max = int(ti.min()), int(ti.max())
            if ti_min < 0 or ti_max >= max(n_tasks, 1):
                ok7 = False
                bad7.append(f"data 帧 task_index∈[{ti_min},{ti_max}] 超出 tasks 位置范围 "
                            f"[0,{n_tasks - 1}] → 训练取任务名时 IndexError")
            if tasks_idx_vals:
                missing = sorted(set(int(v) for v in np.unique(ti)) - set(tasks_idx_vals))
                if missing:
                    ok7 = False
                    bad7.append(f"data 引用了 tasks.parquet 中不存在的 task_index: {missing}")
        if ep_meta_files and "tasks" in ep_df.columns and task_names:
            name_set = set(task_names); unknown = set()
            for tv in ep_df["tasks"].to_numpy():
                try:
                    tl = list(tv) if tv is not None else []
                except TypeError:
                    tl = [tv]
                for t in tl:
                    if str(t) not in name_set:
                        unknown.add(str(t))
            if unknown:
                ok7 = False
                bad7.append(f"episodes 引用了 tasks.parquet 中不存在的任务名: "
                            f"{sorted(unknown)[:5]}")
        if ok7:
            lines.append(f"  ✓ {n_tasks} 个任务 task_index 为 0 基连续，data/episodes 引用一致")
        else:
            for b in bad7:
                lines.append(f"  ✗ {b}")
            lines.append("  → 修复：压实 tasks.parquet 的 task_index 为 0..n-1，同步改写 data 帧")
            lines.append("    task_index 与 stats.json 统计（见 数据问题分析.md 问题7）")
        results.append(ok7)

        # ---- 问题8: RECAP 标注一致性与错位 ----
        lines.append("\n【问题8】RECAP 标注一致性（区间错位/越界/sidecar 过期）")
        ok8 = True; bad8 = []; info8 = []
        ann_path = ds.meta_dir / "reward_annotations.json"
        sidecar_path = ds.meta_dir / "recap_sidecar.jsonl"

        ep_len = {}
        for r in (ds.episodes or []):
            try:
                ep_len[int(r.get("episode_index", -1))] = int(r.get(
                    "length",
                    int(r.get("dataset_to_index", 0)) - int(r.get("dataset_from_index", 0))))
            except Exception:
                pass
        n_eps_total = len(ds.episodes) if ds.episodes else 0

        ann_loaded = None
        if ann_path.is_file():
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann_raw = json.load(f)
                from annotation_store import AnnotationStore as _AS
                ann_loaded = {
                    str(k): _AS._migrate_episode(v)
                    for k, v in (ann_raw.get("episodes", {}) or {}).items()}
            except Exception as e:
                ok8 = False
                bad8.append(f"reward_annotations.json 解析失败: {e}")

        if ann_loaded is not None:
            n_pending = 0
            n_manual_suc = n_manual_fail = 0
            for k, ep in sorted(ann_loaded.items(),
                                key=lambda x: int(x[0]) if str(x[0]).lstrip("-").isdigit() else 1 << 30):
                if not str(k).lstrip("-").isdigit():
                    ok8 = False
                    bad8.append(f"标注 episode 键非数字: {k!r}")
                    continue
                ek = int(k)
                if ek < 0 or ek >= n_eps_total:
                    ok8 = False
                    bad8.append(f"标注键 ep{ek} 超出数据集范围（共 {n_eps_total} 集）"
                                f"——疑似删除/合并后未重映射")
                    continue
                L = ep_len.get(ek)
                ivs = sorted(ep.get("intervals", []),
                             key=lambda x: int(x.get("start", 0)))
                prev_end = None
                for iv in ivs:
                    s, e = int(iv.get("start", -1)), int(iv.get("end", -1))
                    lbl = iv.get("label", "")
                    if e < s:
                        ok8 = False
                        bad8.append(f"ep{ek} 区间 [{s},{e}] start>end")
                    if lbl not in ("good", "bad", "neutral"):
                        ok8 = False
                        bad8.append(f"ep{ek} 区间 [{s},{e}] 标签非法: {lbl!r}")
                    if L is not None and (e >= L or s < 0):
                        ok8 = False
                        if iv.get("is_human_intervention") and 0 <= s < L and e >= L:
                            # 录制侧已知伪影：接管区间以"推帧计数"为时钟，停止录制
                            # 时 stop(flush=False) 丢弃未写入队列（1~3 帧）→ 保存的
                            # 实际帧数 < 推帧计数，仍在接管中收尾的区间 end 越界。
                            # 剪辑重映射数学上保界（剪辑前备份即已越界可证），
                            # 非剪辑/合并错位，非数据损坏。
                            bad8.append(
                                f"ep{ek} 干预区间 [{s},{e}] 越界（该集现长 {L} 帧，"
                                f"超出 {e - L + 1} 帧）——录制停止时队列 flush 丢帧所致"
                                f"（推帧计数>实际保存帧数，接管在停止时收尾），"
                                f"非数据损坏；end 截断到 {L - 1} 即可")
                        else:
                            bad8.append(f"ep{ek} 区间 [{s},{e}] 越界（该集现长 {L} 帧）"
                                        f"——疑似剪辑后未重映射")
                    if prev_end is not None and s <= prev_end:
                        ok8 = False
                        bad8.append(f"ep{ek} 区间 [{s},{e}] 与前一区间重叠（帧 {max(s, 0)}~{prev_end}）")
                    prev_end = e if prev_end is None else max(prev_end, e)
                    if iv.get("pending"):
                        n_pending += 1
                succ = ep.get("episode_success", None)
                if succ not in (True, False, None):
                    ok8 = False
                    bad8.append(f"ep{ek} episode_success 值非法: {succ!r}")
                elif succ is True:
                    n_manual_suc += 1
                elif succ is False:
                    n_manual_fail += 1
                src = ep.get("source", None)
                if src not in ("teleop", "rollout", None):
                    ok8 = False
                    bad8.append(f"ep{ek} source 值非法: {src!r}")
            n_ann_eps = sum(1 for ep in ann_loaded.values()
                            if ep.get("intervals") or ep.get("episode_success") is not None)
            info8.append(f"标注 {n_ann_eps}/{n_eps_total} 集；人工直标成败 ✓{n_manual_suc} ✗{n_manual_fail}")
            if n_pending:
                info8.append(f"待确认预标区间 {n_pending} 个（pending，展开时跳过）")

            # π*0.6 元数据一致性（策略版本/任务组，PI06_RL_迁移实施方案.md R1a）：
            # ① 同一 group_id 的所有 episode 必须同 policy——组相对优势的前提
            #   （GRPO 式基线只对"同任务实例+同策略"成立）；同组混策略会静默
            #   污染优势计算。合并数据集后同名组按 group_id（含策略段）自然
            #   归并，此检查兜住手工编辑 JSON 造成的污染；
            # ② group_id 的策略段与该集 policy 字段不符 → 标注内部自相矛盾。
            # 两者均为标注元数据问题（非数据损坏），改组名/补 policy 即可修复。
            group_policies = {}   # group_id -> {policy_key: [ep, ...]}
            n_grouped = 0
            for k, ep in ann_loaded.items():
                ek = int(k) if str(k).lstrip("-").isdigit() else -1
                gid = ep.get("group_id")
                pol = ep.get("policy_checkpoint") or "无"
                if gid:
                    n_grouped += 1
                    group_policies.setdefault(str(gid), {}).setdefault(str(pol), []).append(ek)
            if group_policies:
                for gid, pols in group_policies.items():
                    if len(pols) > 1:
                        ok8 = False
                        detail = "; ".join(
                            f"{p}: ep{sorted(eps)[:4]}{'…' if len(eps) > 4 else ''}"
                            for p, eps in pols.items())
                        bad8.append(
                            f"组 \"{gid.split(' | ')[0]}\" 混入 {len(pols)} 个不同策略"
                            f"（{detail}）——组相对优势要求同组同策略，非数据损坏；"
                            f"在 Episode 列表改组名拆分即可修复")
                # group_id 策略段与 policy 字段一致性
                for k, ep in ann_loaded.items():
                    gid = str(ep.get("group_id") or "")
                    pol = ep.get("policy_checkpoint")
                    if gid and pol and " | " in gid and gid.rsplit(" | ", 1)[1] not in str(pol):
                        ok8 = False
                        bad8.append(
                            f"ep{int(k)} group_id 策略段({gid.rsplit(' | ', 1)[1]}) "
                            f"与 policy 字段({pol}) 不符——标注元数据自相矛盾，"
                            f"非数据损坏；重新设置该集组名即可")
                info8.append(
                    f"任务组 {len(group_policies)} 个（{n_grouped} 集）；"
                    f"策略版本 {len({str(ep.get('policy_checkpoint') or '专家/未标') for ep in ann_loaded.values()})} 种")

            # sidecar 一致性（存在才检查）
            if sidecar_path.is_file():
                try:
                    with open(sidecar_path, "r", encoding="utf-8") as f:
                        sc_rows = [json.loads(l) for l in f.read().splitlines() if l.strip()]
                    if len(sc_rows) != N:
                        ok8 = False
                        bad8.append(f"sidecar 行数 {len(sc_rows)} ≠ 数据总帧数 {N}"
                                    f"——数据集已变更（剪辑/删除后属正常，非数据损坏），"
                                    f"重新生成即可")
                    ep_max = max((int(r.get("episode_index", -1)) for r in sc_rows), default=-1)
                    if ep_max >= n_eps_total:
                        ok8 = False
                        bad8.append(f"sidecar 引用 ep{ep_max} 超出数据集范围（共 {n_eps_total} 集）")
                    if os.path.getmtime(sidecar_path) < os.path.getmtime(ann_path):
                        ok8 = False
                        bad8.append("sidecar 早于最近一次标注保存（标注已更新，非数据损坏）"
                                    "——重新点\"生成训练 sidecar\"")
                    info8.append(f"sidecar {len(sc_rows)} 行已生成")
                except Exception as e:
                    ok8 = False
                    bad8.append(f"sidecar 读取/解析失败: {e}")
            else:
                info8.append("sidecar 未生成（训练前需在 RECAP 面板\"生成训练 sidecar\"）")

        if not ann_path.is_file():
            lines.append("  （无 reward_annotations.json，未做 RECAP 标注，跳过）")
        elif ok8:
            lines.append("  ✓ 标注键/区间边界/标签/字段全部合法，sidecar 同步")
            for i in info8:
                lines.append(f"  · {i}")
        else:
            for b in bad8[:10]:
                lines.append(f"  ✗ {b}")
            if len(bad8) > 10:
                lines.append(f"  ... 共 {len(bad8)} 条")
            for i in info8:
                lines.append(f"  · {i}")
            lines.append("  → 修复：越界/重叠/错位区间在 RECAP 面板双击修改或删除；")
            lines.append("    剪辑/删除/合并数据集后必须重新展开 sidecar（RECAP_MEM_实施路线.md P0.4）")
        results.append(ok8)

        # ---- 总结 ----
        all_ok = all(results)
        lines.append("\n" + "=" * 64)
        names = ["问题1 data文件映射", "问题2 视频帧数/to_ts", "问题3 frame_index/ts",
                 "问题4 stats一致性", "问题5 恒等映射", "问题6 meta/data一致性",
                 "问题7 task_index连续性", "问题8 RECAP标注一致性"]
        for name, ok in zip(names, results):
            lines.append(f"  {'✓ 通过' if ok else '✗ 存在问题'}  {name}")
        lines.append("")
        if all_ok:
            lines.append(f"  ✅ 结论：全部 {len(names)} 类问题通过，可用于 VLA 模型训练。")
        else:
            n_fail = sum(1 for ok in results if not ok)
            # 良性问题集合（剪辑/删除/标注更新后的正常现象，非数据损坏）：
            # - sidecar 过期（行数≠总帧数 / 早于标注保存）
            # - 干预区间越界（录制停止队列 flush 丢帧，截断即修复）
            # → 给出专门的温和结论，避免误读为结构损坏
            benign_only = (n_fail == 1 and not ok8 and bad8
                           and all("非数据损坏" in b for b in bad8))
            if benign_only:
                has_iv_oob = any("干预区间" in b for b in bad8)
                has_sc_stale = any("sidecar" in b for b in bad8)
                if has_iv_oob and not has_sc_stale:
                    lines.append("  ⚠ 结论：数据结构完好（仅录制期干预区间轻微越界——")
                    lines.append("    停止录制时队列 flush 丢帧的正常伪影，非数据损坏）。")
                    lines.append("    修复：在 RECAP 面板双击该区间把 end 改到集长-1，")
                    lines.append("    再重新\"生成训练 sidecar\"。")
                else:
                    lines.append("  ⚠ 结论：数据结构完好（仅训练 sidecar 过期）——剪辑/删除/")
                    lines.append("    标注更新后的正常现象，非数据损坏。请在 RECAP 面板重新")
                    lines.append("    \"生成训练 sidecar\" 后再训练（一键，秒级完成）。")
                    if has_iv_oob:
                        lines.append("    另有个别干预区间轻微越界（flush 丢帧伪影），")
                        lines.append("    顺手在 RECAP 面板双击把 end 改到集长-1 即可。")
            else:
                lines.append(f"  ❌ 结论：{n_fail} 项存在问题，需修复后再训练。")

        # 用 HTML 渲染，绿色✓ / 红色✗
        html = self._colorize("\n".join(lines))
        self.report_text.setHtml(html)

    @staticmethod
    def _colorize(text):
        """把报告文本转为简单 HTML，✓ 绿色、✅ 绿色、✗ 红色、❌ 红色。"""
        import html as _html
        out = _html.escape(text)
        out = out.replace("✓", '<span style="color:#2e7d32;font-weight:bold">✓</span>')
        out = out.replace("✗", '<span style="color:#c62828;font-weight:bold">✗</span>')
        out = out.replace("✅", '<span style="color:#2e7d32;font-weight:bold">✅</span>')
        out = out.replace("❌", '<span style="color:#c62828;font-weight:bold">❌</span>')
        out = out.replace("\n", "<br>")
        out = out.replace("  ", "&nbsp;&nbsp;")
        return f"<div style='white-space:pre-wrap'>{out}</div>"


# ==================== 价值优势显示对话框 ====================
class AdvantagePlotDialog(QDialog):
    """显示单个 episode 的价值/优势变化（数据源：meta/recap_sidecar.jsonl 逐帧列）。

    上下两个子图（共享 X=帧号）：
    - 上：V(s) 预期剩余代价（⑤ 的 td_value 列，∈[-1,0]，→0 接近成功完成）；
      无 td 列时显示提示文字；
    - 下：优势曲线——pg_advantage（⑥ 训练用合并优势，蓝）/ td_advantage
      （⑤ TD 优势，橙）/ advantage_score（① 平坦兜底，灰），零线基准；
    - 背景色带：人标区间（good 绿 / 干预蓝 / bad 红，从逐帧列直接提取）。
    """

    def __init__(self, parent, ep_idx, rows):
        super().__init__(parent)
        self.setWindowTitle(f"Episode {ep_idx} 价值/优势变化")
        self.resize(880, 720)

        layout = QVBoxLayout(self)
        fig = Figure(figsize=(8.6, 5.2), dpi=100)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        # ---- 图表说明（HTML 图文，颜色与曲线一一对应）----
        explain = QTextEdit()
        explain.setReadOnly(True)
        explain.setMaximumHeight(150)
        explain.setHtml(
            "<div style='font-size:11px; line-height:1.5'>"
            "<b>上图｜V(s) 预期剩余代价</b>（∈[-1,0]）：价值函数对“照这样下去还有多久完成”的估计，"
            "越接近 0 = 越接近成功完成，越负 = 剩余代价越大。"
            "<b>上升=在接近完成（好），下降/停滞=在恶化或卡住（坏）</b>。<br>"
            "<b>下图｜优势 A</b>（正=正向动作/多模仿，负=负分动作/降权）：<br>"
            "&nbsp;&nbsp;<span style='color:#1565C0'><b>■ 蓝粗线 pg_advantage</b></span>："
            "⑥ 训练实际使用的合并优势（人标 ±1 / 干预 +1 / 组优势调制），"
            "AWR 权重 exp(A/τ) 的输入——决定该帧在训练中被模仿的力度。<br>"
            "&nbsp;&nbsp;<span style='color:#EF6C00'><b>■ 橙线 td_advantage</b></span>："
            "⑤ TD 优势 = 未来 N 步实际代价 + V'−V（价值模型预测），正=此刻动作在拉近完成。"
            "<br>"
            "<b>背景色带（人标区间）</b>："
            "<span style='color:#1976D2'><b>■ 蓝=干预段</b></span>（人工接管纠正，恒正）、"
            "<span style='color:#2e7d32'><b>■ 绿=good 段</b></span>、"
            "<span style='color:#c62828'><b>■ 红=bad 段</b></span>（接管前窗口/人标坏）。"
            "健康形态：红/蓝带交界处 V 应止跌回升；bad 段内 td_advantage 应偏负。"
            "</div>")
        layout.addWidget(explain)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        rows = sorted(rows, key=lambda r: int(r.get("frame_index", 0)))
        x = np.array([int(r.get("frame_index", i)) for i, r in enumerate(rows)])

        def _col(key):
            if key not in rows[0]:
                return None
            return np.array([float(r.get(key, 0) or 0) for r in rows])

        ind = np.array([int(r.get("advantage_indicator", 0) or 0) for r in rows])
        itv = np.array([int(r.get("is_human_intervention", 0) or 0) for r in rows])
        succ = rows[0].get("episode_success", None)
        grp = rows[0].get("group_advantage", None)
        td_value = _col("td_value")
        td_adv = _col("td_advantage")
        pg_adv = _col("pg_advantage")
        score = _col("advantage_score")

        ax_v = fig.add_subplot(211)
        ax_a = fig.add_subplot(212, sharex=ax_v)

        # ---- 背景色带：人标区间（good 绿 / 干预蓝 / bad 红）----
        def _runs(mask):
            out, s = [], None
            for i, v in enumerate(mask):
                if v and s is None:
                    s = i
                elif not v and s is not None:
                    out.append((s, i - 1))
                    s = None
            if s is not None:
                out.append((s, len(mask) - 1))
            return out

        for ax in (ax_v, ax_a):
            for s, e in _runs(itv > 0):
                ax.axvspan(x[s], x[e], color="#1976D2", alpha=0.18, lw=0)
            for s, e in _runs((ind > 0) & (itv == 0)):
                ax.axvspan(x[s], x[e], color="#2e7d32", alpha=0.15, lw=0)
            for s, e in _runs(ind < 0):
                ax.axvspan(x[s], x[e], color="#c62828", alpha=0.15, lw=0)

        # ---- 上：V(s) ----
        if td_value is not None:
            ax_v.plot(x, td_value, color="#5E35B1", lw=1.6,
                      label="V(s) 预期剩余代价（td_value）")
            ax_v.set_ylim(-1.05, 0.05)
            ax_v.axhline(0, color="#888", lw=0.8, ls="--")
            ax_v.legend(loc="lower left", fontsize=8)
            ax_v.set_ylabel("V(s) ∈ [-1,0]")
        else:
            ax_v.text(0.5, 0.5, "无 td_value 列（⑤ TD 价值未跑）",
                      ha="center", va="center", transform=ax_v.transAxes,
                      color="#888")
        ax_v.grid(alpha=0.3)
        succ_txt = {1: "成功", 0: "失败", -1: "未评", True: "成功", False: "失败",
                    None: "未评"}.get(succ, str(succ))
        title = f"ep{ep_idx}（成败: {succ_txt}）"
        if grp is not None:
            title += f" | 组优势: {float(grp):+.3f}"
        ax_v.set_title(title, fontsize=11)

        # ---- 下：优势 ----
        plotted = False
        if pg_adv is not None:
            ax_a.plot(x, pg_adv, color="#1565C0", lw=1.8,
                      label="pg_advantage（⑥ 训练用）")
            plotted = True
        if td_adv is not None:
            ax_a.plot(x, td_adv, color="#EF6C00", lw=1.2, alpha=0.85,
                      label="td_advantage（⑤ TD）")
            plotted = True
        if not plotted and score is not None:
            ax_a.plot(x, score, color="#666", lw=1.2,
                      label="advantage_score（① 平坦）")
            plotted = True
        ax_a.axhline(0, color="#333", lw=0.9)
        ax_a.set_xlabel("帧号")
        ax_a.set_ylabel("优势 A")
        ax_a.grid(alpha=0.3)
        if plotted:
            ax_a.legend(loc="upper right", fontsize=8)
        # 色带图例（手工代理）
        from matplotlib.patches import Patch
        handles = [Patch(fc="#1976D2", alpha=0.25, label="干预段"),
                   Patch(fc="#2e7d32", alpha=0.22, label="good"),
                   Patch(fc="#c62828", alpha=0.22, label="bad")]
        ax_v.legend(handles=handles, loc="lower right", fontsize=8)

        fig.tight_layout()
        canvas.draw()

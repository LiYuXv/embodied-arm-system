#!/usr/bin/env python
"""训练日志分析 + 离线推理一体化 GUI（PyQt6）。

左栏：解析 train.log，同图绘制 train loss / eval loss 曲线，自动标注 eval
最低点并映射到实际存在的最近 checkpoint（最佳模型），可一键应用到右栏。

右栏：对选定 checkpoint 跑 episode 离线推理（复用 test_inference_episode.py 的
load_policy/run_episode），pred/gt 逐维曲线嵌入显示并保存 PNG。

用法：
    conda run -n VLA --no-capture-output python test_inference_gui.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # 绘图线程安全；画布用 FigureCanvasQTAgg 嵌入
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QTextEdit,
    QFileDialog, QSplitter, QGroupBox,
)

from test_inference_episode import load_policy, run_episode  # 复用现有推理逻辑


# ==================== train.log 解析 ====================

_TRAIN_RE = re.compile(r"step:(\d+)(K?)\s.*?loss:([\d.eE+-]+)")
_EVAL_RE = re.compile(r"step\s+(\d+):\s*eval_loss=([\d.eE+-]+)")


def parse_train_log(path):
    """解析训练日志。返回 (train_steps, train_losses), (eval_steps, eval_losses)。

    train 行格式: "step:22K smpl:... loss:0.044 ..."（<1000 时为 "step:50"）
    eval 行格式:  "step 500: eval_loss=0.3629"
    """
    train_steps, train_losses, eval_steps, eval_losses = [], [], [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _EVAL_RE.search(line)
            if m:
                eval_steps.append(int(m.group(1)))
                eval_losses.append(float(m.group(2)))
                continue
            m = _TRAIN_RE.search(line)
            if m:
                step = int(m.group(1)) * (1000 if m.group(2) == "K" else 1)
                train_steps.append(step)
                train_losses.append(float(m.group(3)))
    return (np.array(train_steps), np.array(train_losses)), \
           (np.array(eval_steps), np.array(eval_losses))


def find_best_checkpoint(eval_steps, eval_losses, ckpt_root):
    """eval_loss 最低点 -> 映射到 checkpoints/ 下实际存在的最近存档目录。

    优先取不超过最优 step 的最大存档；若无则取最近的存档。
    返回 (best_step, best_eval, ckpt_path 或 None, 过拟合涨幅)。
    """
    i = int(np.argmin(eval_losses))
    best_step, best_eval = int(eval_steps[i]), float(eval_losses[i])
    overfit = float(eval_losses[-1] - best_eval) if len(eval_losses) else 0.0

    ckpt_path = None
    ckpt_dir = Path(ckpt_root) / "checkpoints"
    if ckpt_dir.is_dir():
        saved = sorted(int(p.name) for p in ckpt_dir.iterdir()
                       if p.is_dir() and p.name.isdigit())
        if saved:
            le = [s for s in saved if s <= best_step]
            ckpt_step = le[-1] if le else saved[0]
            ckpt_path = str(ckpt_dir / f"{ckpt_step:06d}")
    return best_step, best_eval, ckpt_path, overfit


def default_log_for_checkpoint(ckpt_path):
    """按 checkpoint 路径推导 train.log：ckpt/<run>/checkpoints/... -> ckpt/<run>/train.log"""
    p = Path(ckpt_path)
    parts = list(p.parts)
    if "checkpoints" in parts:
        run_dir = Path(*parts[: parts.index("checkpoints")])
        cand = run_dir / "train.log"
        if cand.is_file():
            return str(cand)
    return "./train.log"


# ==================== 离线推理后台线程 ====================

class InferenceWorker(QThread):
    status = pyqtSignal(str)
    episode_done = pyqtSignal(int, int, float, object, object)  # ep, frames, err, pred, gt
    finished_all = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, ckpt, dataset_root, repo_id, episodes, cam_mode, seed, parent=None):
        super().__init__(parent)
        self.ckpt = ckpt
        self.dataset_root = dataset_root
        self.repo_id = repo_id
        self.episodes = episodes
        self.cam_mode = cam_mode
        self.seed = seed

    def run(self):
        try:
            self.status.emit("加载模型与数据集（约 1 分钟）...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            policy, preprocessor, postprocessor, config = load_policy(self.ckpt, "cuda" if device.type == "cuda" else "cpu")

            torque_key = getattr(config, "torque_key", "observation.gripper_torque")
            input_features = getattr(config, "input_features", None) or {}
            torque_key = torque_key if torque_key in input_features else None

            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset = LeRobotDataset(self.repo_id, root=self.dataset_root)
            self.status.emit(
                f"数据集 {dataset.num_episodes} episodes / {dataset.num_frames} 帧 | "
                f"力矩: {'启用' if torque_key else '无'} | cam={self.cam_mode} seed={self.seed}")

            total_err, total_frames = 0.0, 0
            for ep in self.episodes:
                if ep >= dataset.num_episodes:
                    self.status.emit(f"episode {ep} 超出范围，跳过")
                    continue
                # 每个 episode 重置种子：cam 模式之间可公平对比
                torch.manual_seed(self.seed)
                actions, gt = run_episode(
                    policy, preprocessor, postprocessor, dataset, ep, device,
                    torque_key=torque_key, cam_mode=self.cam_mode)
                err = torch.mean(torch.abs(actions - gt.to(actions.device))).item()
                n = actions.shape[0]
                total_err += err * n
                total_frames += n
                self.episode_done.emit(ep, n, err, actions.cpu(), gt.cpu())
            if total_frames:
                self.finished_all.emit(
                    f"总体 mean abs err = {total_err / total_frames:.4f}（{total_frames} 帧）")
            else:
                self.finished_all.emit("没有有效 episode")
        except Exception:
            import traceback
            self.failed.emit(traceback.format_exc())


# ==================== 主窗口 ====================

class InferenceGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SmolVLA 训练分析 + 离线推理")
        self.resize(1500, 900)
        self.worker = None
        self._best_ckpt = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setSizes([750, 750])

        self._auto_load_log()

    # ---------- 左栏：训练曲线 ----------
    def _build_left(self):
        box = QGroupBox("训练日志分析（train/eval loss）")
        lay = QVBoxLayout(box)

        row = QHBoxLayout()
        row.addWidget(QLabel("train.log:"))
        self.log_edit = QLineEdit("./ckpt/smolvla_ela3_torque_la5/train.log")
        row.addWidget(self.log_edit, stretch=1)
        btn = QPushButton("浏览")
        btn.clicked.connect(self._browse_log)
        row.addWidget(btn)
        btn2 = QPushButton("分析")
        btn2.clicked.connect(self._analyze_log)
        row.addWidget(btn2)
        lay.addLayout(row)

        self.loss_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        lay.addWidget(self.loss_canvas, stretch=1)

        self.best_label = QLabel("尚未分析")
        self.best_label.setWordWrap(True)
        self.best_label.setStyleSheet("font-size: 12px; padding: 4px;")
        lay.addWidget(self.best_label)

        self.btn_use_best = QPushButton("使用最佳模型 →")
        self.btn_use_best.setEnabled(False)
        self.btn_use_best.clicked.connect(self._apply_best)
        lay.addWidget(self.btn_use_best)
        return box

    # ---------- 右栏：离线推理 ----------
    def _build_right(self):
        box = QGroupBox("离线推理（pred vs gt）")
        lay = QVBoxLayout(box)

        def add_row(label, widget):
            r = QHBoxLayout()
            r.addWidget(QLabel(label))
            r.addWidget(widget, stretch=1)
            lay.addLayout(r)

        self.ckpt_edit = QLineEdit("./ckpt/smolvla_ela3_torque_la5/checkpoints/last")
        add_row("checkpoint:", self.ckpt_edit)
        self.dataset_edit = QLineEdit("./lerobot_data/ela3_demo")
        add_row("数据集 root:", self.dataset_edit)
        self.repo_edit = QLineEdit("ela3_demo")
        add_row("repo_id:", self.repo_edit)

        row = QHBoxLayout()
        row.addWidget(QLabel("episode:"))
        self.ep_edit = QLineEdit("5")
        self.ep_edit.setPlaceholderText("如 5 或 5,10,30 或 all")
        row.addWidget(self.ep_edit, stretch=1)
        row.addWidget(QLabel("cam:"))
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(["both", "cam1", "cam2", "cam0"])
        row.addWidget(self.cam_combo)
        row.addWidget(QLabel("seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(1000)
        row.addWidget(self.seed_spin)
        lay.addLayout(row)

        self.btn_run = QPushButton("运行推理")
        self.btn_run.setStyleSheet(
            "background-color: #00695C; color: white; font-weight: bold; padding: 6px;")
        self.btn_run.clicked.connect(self._run_inference)
        lay.addWidget(self.btn_run)

        self.infer_canvas = FigureCanvas(Figure(figsize=(6, 5)))
        lay.addWidget(self.infer_canvas, stretch=1)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        self.result_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        lay.addWidget(self.result_text)
        return box

    # ---------- 日志分析逻辑 ----------
    def _auto_load_log(self):
        p = default_log_for_checkpoint(self.ckpt_edit.text().strip())
        if Path(p).is_file():
            self.log_edit.setText(p)
            self._analyze_log()

    def _browse_log(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择 train.log", ".", "日志 (*.log *.txt);;所有文件 (*)")
        if f:
            self.log_edit.setText(f)
            self._analyze_log()

    def _analyze_log(self):
        path = self.log_edit.text().strip()
        if not Path(path).is_file():
            self.best_label.setText(f"文件不存在: {path}")
            return
        (ts, tl), (es, el) = parse_train_log(path)
        fig = self.loss_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        if len(ts):
            ax.plot(ts, tl, lw=1, alpha=0.7, label=f"train loss ({len(ts)} 点)")
        info_lines = [f"train {len(ts)} 点 / eval {len(el)} 点"]
        self._best_ckpt = None
        if len(es):
            ax.plot(es, el, "o-", color="#c62828", lw=1.5, label="eval loss")
            ckpt_root = str(Path(self.ckpt_edit.text().strip()).parent.parent)
            best_step, best_eval, ckpt_path, overfit = find_best_checkpoint(es, el, ckpt_root)
            ax.axvline(best_step, color="gray", ls="--", alpha=0.6)
            ax.plot([best_step], [best_eval], "r*", ms=15)
            ax.annotate(f"best {best_eval:.4f} @ {best_step}",
                        (best_step, best_eval), textcoords="offset points",
                        xytext=(10, 12), color="#c62828")
            info_lines.append(
                f"<b style='color:#c62828'>eval 最低点: {best_eval:.4f} @ step {best_step}</b>")
            if ckpt_path:
                self._best_ckpt = ckpt_path
                info_lines.append(f"最佳模型（最近存档）: <b>{ckpt_path}</b>")
                self.btn_use_best.setEnabled(True)
            else:
                info_lines.append("checkpoints/ 下无存档目录")
                self.btn_use_best.setEnabled(False)
            if overfit > 0.02:
                info_lines.append(
                    f"<span style='color:#e65100'>⚠ 过拟合：eval 已从最低点上升 {overfit:.4f}"
                    f"（最新 {el[-1]:.4f}），建议用最佳档而非 last</span>")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        self.loss_canvas.draw_idle()
        self.best_label.setText("<br>".join(info_lines))

    def _apply_best(self):
        if self._best_ckpt:
            self.ckpt_edit.setText(self._best_ckpt)

    # ---------- 推理逻辑 ----------
    def _run_inference(self):
        if self.worker is not None and self.worker.isRunning():
            self.result_text.append("推理进行中，请等待完成")
            return
        ep_text = self.ep_edit.text().strip()
        if ep_text.lower() == "all":
            # all 需要先加载数据集才知道数量；用占位让 worker 展开
            episodes = None
        else:
            try:
                episodes = [int(e) for e in ep_text.split(",")]
            except ValueError:
                self.result_text.append("episode 格式错误（如 5 或 5,10,30 或 all）")
                return
        self.btn_run.setEnabled(False)
        self.result_text.clear()
        self.worker = InferenceWorker(
            self.ckpt_edit.text().strip(), self.dataset_edit.text().strip(),
            self.repo_edit.text().strip(), episodes,
            self.cam_combo.currentText(), self.seed_spin.value())
        if episodes is None:  # all：先探测 episode 数
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
                meta = LeRobotDatasetMetadata(self.repo_edit.text().strip(),
                                              root=self.dataset_edit.text().strip())
                self.worker.episodes = list(range(meta.total_episodes))
            except Exception as e:
                self.result_text.append(f"读取数据集元数据失败: {e}")
                self.btn_run.setEnabled(True)
                return
        self.worker.status.connect(lambda s: self.result_text.append(s))
        self.worker.episode_done.connect(self._on_episode_done)
        self.worker.finished_all.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_episode_done(self, ep, n, err, pred, gt):
        self.result_text.append(f"episode {ep:03d}: frames={n}, mean abs err={err:.4f}")
        self._plot_pred_gt(pred, gt, f"episode {ep} [{self.cam_combo.currentText()}] err={err:.4f}")
        # 同步保存 PNG（与命令行脚本一致）
        from test_inference_episode import plot_actions
        save_dir = Path("./inference_plots")
        save_dir.mkdir(exist_ok=True)
        suffix = "" if self.cam_combo.currentText() == "both" else f"_{self.cam_combo.currentText()}"
        plot_actions(pred, gt, pred.shape[1],
                     f"episode {ep}  (mean abs err={err:.4f})",
                     save_dir / f"episode_{ep:03d}{suffix}.png")

    def _plot_pred_gt(self, pred, gt, title):
        fig = self.infer_canvas.figure
        fig.clear()
        labels = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
        dim = pred.shape[1]
        axs = fig.subplots(dim, 1, sharex=True)
        if dim == 1:
            axs = [axs]
        for i in range(dim):
            axs[i].plot(pred[:, i].numpy(), lw=0.8, label="pred")
            axs[i].plot(gt[:, i].numpy(), lw=0.8, label="gt")
            axs[i].set_ylabel(labels[i] if i < len(labels) else f"dim{i}", fontsize=8)
            if i == 0:
                axs[i].legend(fontsize=8, loc="upper right")
            axs[i].tick_params(labelsize=7)
        axs[-1].set_xlabel("frame", fontsize=8)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        self.infer_canvas.draw_idle()

    def _on_finished(self, msg):
        self.result_text.append(f"\n{msg}")
        self.btn_run.setEnabled(True)

    def _on_failed(self, tb):
        self.result_text.append(f"\n推理失败:\n{tb}")
        self.btn_run.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    win = InferenceGui()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

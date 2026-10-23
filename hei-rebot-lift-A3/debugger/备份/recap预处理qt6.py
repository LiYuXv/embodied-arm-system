#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recap预处理qt6.py — RECAP 预处理 Qt6 界面（π*0.6 R1b/R3 + RLinf 价值函数数据准备链）。

训练数据准备链（PI06_RL_迁移实施方案.md §4.6/§4.7，单数据集口径）：
  ① 生成 sidecar    visualize_data/make_recap_sidecar.py     （标注/剪辑后必须重新生成）
  ② 组相对优势      recap_group_advantage.py                 （R1b，无组全 0 无害）
  ③ 计算回报        recap_value.py returns                   （r=-1/步、终止 0/−C_fail、
                     [-1,0] 归一化 → returns_<tag>.parquet）
  ④ 训练价值头      recap_value.py train                     （201-bin two-hot 交叉熵，
                     siglip 后端 → recap_value_<tag>/value_head.pt）
  ⑤ TD 优势         recap_value.py advantage                 （N 步 TD + 分位数二值化，
                     写 sidecar td_value/td_advantage/td_label 列）
  ⑥ pg 优势合并     recap_pg.py --write                      （TD∪人标∪组优势 →
                     pg_advantage 列；RecapPGWeighter 实际读取，不可省）
  ⑦ 权重预览        recap_sidecar.py                         （P1.2，验证用，不进链）
  ⑧ 训练前门禁      recap_check.py                           （检查 ①~⑦ 产物完整性，
                     通过后输出 yaml 修改建议与训练启动命令；不进链）

各步骤用"VLA 环境 python"作为子进程运行（依赖 lerobot 0.6.1 + torch），输出实时
显示在右侧日志面板；支持"一键全流程"（①→⑥ 顺序执行）与"停止"。
两轮数据的物理合并在 visualize_data 的合并工具中完成（本界面不含合并功能）；
合并/剪辑/删除后帧索引重排，必须对目标数据集从 ① 重新走完整链。

用法：
    # 在 VLA 环境里直接跑（GUI 自身只需 PyQt6）
    conda run -n VLA --no-capture-output python recap预处理qt6.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt
from PyQt6.QtGui import QColor, QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# 本文件位于 备份/ 子目录：子进程工作目录与 recap 脚本/数据集的相对路径
# 均以其上一级（debugger 项目目录）为基准
DEBUGGER_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = "/home/chen/anaconda3/envs/VLA/bin/python"

# ---------------------------------------------------------------------------
# 小工具：带标签的参数行
# ---------------------------------------------------------------------------
class ParamRow(QWidget):
    """一个 label + widget 的水平行。"""

    def __init__(self, label: str, widget: QWidget, tip: str = ""):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(label)
        lab.setMinimumWidth(150)
        lay.addWidget(lab)
        lay.addWidget(widget, 1)
        if tip:
            widget.setToolTip(tip)
            lab.setToolTip(tip)


def make_int_spin(v: int, lo: int, hi: int, tip=""):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(v)
    if tip:
        w.setToolTip(tip)
    return w


def make_float_spin(v: float, lo: float, hi: float, dec: int = 2, tip=""):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(0.05)
    w.setValue(v)
    if tip:
        w.setToolTip(tip)
    return w


def make_combo(items: list[str], cur: str, tip=""):
    w = QComboBox()
    w.addItems(items)
    if cur in items:
        w.setCurrentText(cur)
    if tip:
        w.setToolTip(tip)
    return w


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class RecapPreprocessWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RECAP 预处理（π*0.6 数据准备链）")
        self.resize(1460, 860)  # 高度不变，加宽以容纳右侧日志栏

        self.process: QProcess | None = None
        self.chain_queue: list[callable] = []
        self._steps: dict[str, QGroupBox] = {}

        self._build_ui()
        self._refresh_datasets()

    # ---------- UI 构建 ----------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # --- 左列：全局设置 + 步骤 + 链按钮 ---
        left = QVBoxLayout()

        # --- 顶部：数据集 / python 解释器 ---
        top = QGroupBox("全局设置")
        top_lay = QVBoxLayout(top)
        ds_row = QHBoxLayout()
        ds_row.addWidget(QLabel("数据集"))
        self.ds_combo = QComboBox()
        self.ds_combo.setEditable(True)
        self.ds_combo.setMinimumWidth(320)
        ds_row.addWidget(self.ds_combo, 1)
        btn_browse = QPushButton("浏览…")
        btn_browse.clicked.connect(self._browse_dataset)
        ds_row.addWidget(btn_browse)
        btn_refresh = QPushButton("刷新列表")
        btn_refresh.clicked.connect(self._refresh_datasets)
        ds_row.addWidget(btn_refresh)
        top_lay.addLayout(ds_row)

        py_row = QHBoxLayout()
        py_row.addWidget(QLabel("VLA Python"))
        self.py_edit = QLineEdit(DEFAULT_PYTHON)
        self.py_edit.setToolTip("运行 recap 脚本的解释器（需 lerobot 0.6.1 + torch，VLA 环境）")
        py_row.addWidget(self.py_edit, 1)
        top_lay.addLayout(py_row)

        tag_row = QHBoxLayout()
        tag_row.addWidget(QLabel("实验标签 tag"))
        self.tag_edit = QLineEdit("cfail500")
        self.tag_edit.setToolTip(
            "recap_value 三阶段（③回报/④价值头/⑤TD优势）共用的实验标签：\n"
            "回报文件 returns_<tag>.parquet、价值头目录 recap_value_<tag>/、\n"
            "TD 产物 td_advantage_<tag>_N<>_q<>.parquet 均以此命名。\n"
            "③ 的 cfail 改动时若 tag 形如 cfail<数字> 会自动同步。")
        tag_row.addWidget(self.tag_edit, 1)
        top_lay.addLayout(tag_row)
        left.addWidget(top)

        # --- 中部：步骤（滚动区）---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        steps_lay = QVBoxLayout(scroll_widget)
        steps_lay.addWidget(self._build_sidecar_step())
        steps_lay.addWidget(self._build_group_step())
        steps_lay.addWidget(self._build_returns_step())
        steps_lay.addWidget(self._build_vtrain_step())
        steps_lay.addWidget(self._build_adv_step())
        steps_lay.addWidget(self._build_pg_step())
        steps_lay.addWidget(self._build_preview_step())
        steps_lay.addWidget(self._build_check_step())
        steps_lay.addStretch(1)
        scroll.setWidget(scroll_widget)
        left.addWidget(scroll, 1)

        # --- 底部：全流程 / 停止 ---
        chain_row = QHBoxLayout()
        self.btn_chain = QPushButton("一键全流程（①→⑥）")
        self.btn_chain.clicked.connect(self._run_chain)
        chain_row.addWidget(self.btn_chain)
        self.btn_stop = QPushButton("停止当前步骤")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        chain_row.addWidget(self.btn_stop)
        chain_row.addStretch(1)
        left.addLayout(chain_row)

        root.addLayout(left, 1)

        # --- 右列：日志面板（等高，约占一半宽度）---
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        f = QFont("Monospace")
        f.setStyleHint(QFont.StyleHint.Monospace)
        f.setPointSize(9)
        self.log.setFont(f)
        self.log.setMinimumWidth(460)
        root.addWidget(self.log, 1)

    def _add_step_title(self, box: QGroupBox, title: str, desc: str):
        box.setTitle(title)
        lay = QVBoxLayout(box)
        d = QLabel(desc)
        d.setWordWrap(True)
        d.setStyleSheet("color:#666;")
        lay.addWidget(d)
        return lay

    # ---------- 步骤 ① 生成 sidecar ----------
    def _build_sidecar_step(self) -> QGroupBox:
        box = QGroupBox()
        lay = self._add_step_title(
            box, "① 生成训练 sidecar（make_recap_sidecar.py）",
            "把 reward_annotations.json 展开为逐帧 sidecar（不改 parquet）。前置：已完成标注。"
            "合并/剪辑/删除后必须重跑（帧索引重排）。R3 链建议 bad-backprop=0、"
            "terminal-fail=0——中性帧的密集化交给 ④⑤ 的 TD 价值优势。")
        self.s_bb = make_int_spin(0, 0, 200, "bad 段负优势向前回传帧数（R3 建议 0；R2 老链用 50）")
        self.s_tf = make_int_spin(0, 0, 100, "派生失败 episode 末 K 帧标 bad（R3 建议 0；R2 老链用 10）")
        self.s_dry = QCheckBox("只预览不写文件（dry-run）")
        lay.addWidget(ParamRow("bad-backprop", self.s_bb))
        lay.addWidget(ParamRow("terminal-fail", self.s_tf))
        lay.addWidget(self.s_dry)
        btn = QPushButton("运行 ①")
        btn.clicked.connect(lambda: self._run_step("sidecar"))
        lay.addWidget(btn)
        self._steps["sidecar"] = box
        return box

    # ---------- 步骤 ② 组优势 ----------
    def _build_group_step(self) -> QGroupBox:
        box = QGroupBox()
        lay = self._add_step_title(
            box, "② 组相对优势（recap_group_advantage.py）",
            "同组内 GRPO 式标准化，写 sidecar 的 group_advantage 列。需采集端已标 group_id。")
        self.g_pen = make_float_spin(0.25, 0.0, 2.0, 2, "每次人工接管的回报惩罚")
        self.g_min = make_int_spin(4, 1, 50, "成组最少 episode 数（小于则 A=0）")
        self.g_dry = QCheckBox("只预览不写文件（dry-run）")
        lay.addWidget(ParamRow("takeover-penalty", self.g_pen))
        lay.addWidget(ParamRow("min-group", self.g_min))
        lay.addWidget(self.g_dry)
        btn = QPushButton("运行 ②")
        btn.clicked.connect(lambda: self._run_step("group"))
        lay.addWidget(btn)
        self._steps["group"] = box
        return box

    # ---------- 步骤 ③ 计算回报（recap_value.py returns） ----------
    def _build_returns_step(self) -> QGroupBox:
        box = QGroupBox()
        lay = self._add_step_title(
            box, "③ 计算回报（recap_value.py returns）",
            "DynamicReturn：每步 r=-1（多耽误一步扣一分），成功终止 0 / 失败终止 −C_fail，"
            "G_t 按 |全局最小回报| 归一化到 [-1,0]，写 meta/returns_<tag>.parquet。"
            "C_fail 建议与典型集长同量级（500~1000，ela3 集长 ~200-1800 帧）。")
        self.r_cfail = make_int_spin(500, 1, 5000, "失败集终止惩罚 C_fail（建议 500~1000；100 对千帧集偏小）")
        self.r_cfail.valueChanged.connect(self._sync_tag_from_cfail)
        self.r_gamma = make_float_spin(1.0, 0.9, 1.0, 2, "折扣因子 γ（1.0=无折扣）")
        self.r_unrated = make_combo(["derive", "fail", "skip"], "derive",
                                    "未评集（保存时跳过成败评价）处理：derive=按片段标注派生/fail=按失败/skip=不进入训练")
        lay.addWidget(ParamRow("cfail", self.r_cfail))
        lay.addWidget(ParamRow("gamma", self.r_gamma))
        lay.addWidget(ParamRow("unrated", self.r_unrated))
        btn = QPushButton("运行 ③")
        btn.clicked.connect(lambda: self._run_step("returns"))
        lay.addWidget(btn)
        self._steps["returns"] = box
        return box

    # ---------- 步骤 ④ 训练价值头（recap_value.py train） ----------
    def _build_vtrain_step(self) -> QGroupBox:
        box = QGroupBox()
        lay = self._add_step_title(
            box, "④ 训练价值头（recap_value.py train，201-bin）",
            "201-bin 分布式价值头（two-hot 软标签交叉熵，V=E[atoms]∈[-1,0]）。"
            "特征复用 recap_critic 缓存；siglip 后端首轮提特征 ~10-15 分钟。"
            "看日志 Spearman（排序一致性，越高越好）与 bad/good 分离度。"
            "输出 lerobot_data/recap_value_<tag>/value_head.pt（含 denominator/fps，部署侧用）。")
        self.v_backend = make_combo(["siglip", "state"], "siglip", "视觉后端（siglip 更准；state 仅冒烟）")
        self.v_pi05 = QLineEdit("./pretrained_models/pi05_base")
        self.v_epochs = make_int_spin(30, 1, 500, "价值头训练 epoch 数")
        self.v_holdout = make_float_spin(0.1, 0.0, 0.5, 2, "留出集比例（按 episode 划分）")
        lay.addWidget(ParamRow("backend", self.v_backend))
        lay.addWidget(ParamRow("pi05-path", self.v_pi05))
        lay.addWidget(ParamRow("epochs", self.v_epochs))
        lay.addWidget(ParamRow("holdout", self.v_holdout))
        btn = QPushButton("运行 ④")
        btn.clicked.connect(lambda: self._run_step("vtrain"))
        lay.addWidget(btn)
        self._steps["vtrain"] = box
        return box

    # ---------- 步骤 ⑤ TD 优势（recap_value.py advantage） ----------
    def _build_adv_step(self) -> QGroupBox:
        box = QGroupBox()
        lay = self._add_step_title(
            box, "⑤ TD 优势 + 二值化（recap_value.py advantage）",
            "A_t = norm(r_{t:t+N}) + γ^N·V(o_{t+N}) − V(o_t)，分位数二值化（top-q% 为正），"
            "写 sidecar td_value/td_advantage/td_label 列 + 独立 parquet。"
            "⚠ fill-unlabeled 是 CFG 式标注（供条件化 prompt 取符号）："
            "与 RecapWeighter 降权同开会把 (1-q)% 中性帧标负，慎用——"
            "加权训练走 ⑥ 的连续 td_advantage 通道即可，无需勾选。")
        self.a_lookahead = make_int_spin(10, 1, 50, "N 步前瞻（RLinf 默认 10）")
        self.a_quantile = make_float_spin(0.3, 0.05, 0.9, 2, "正样本分位数（0.3=top 30%）")
        self.a_fill = QCheckBox("--fill-unlabeled：未标注帧按 td_label 补 indicator=±1（人标/干预帧不动）")
        self.a_fill.setChecked(False)
        lay.addWidget(ParamRow("lookahead", self.a_lookahead))
        lay.addWidget(ParamRow("quantile", self.a_quantile))
        lay.addWidget(self.a_fill)
        btn = QPushButton("运行 ⑤")
        btn.clicked.connect(lambda: self._run_step("adv"))
        lay.addWidget(btn)
        self._steps["adv"] = box
        return box

    # ---------- 步骤 ⑥ pg 优势合并 ----------
    def _build_pg_step(self) -> QGroupBox:
        box = QGroupBox()
        lay = self._add_step_title(
            box, "⑥ 优势合并 + 权重（recap_pg.py --write，R3）",
            "合并 TD 优势（⑤，优先）∪ 人标 ∪ 组优势 → pg_advantage 列，经算子映射权重。"
            "★ 此步生成 RecapPGWeighter 实际读取的列，训练前不可省；"
            "先不勾 --write 预览权重分布，确认合理再勾选写出。"
            "critic/TD 校验（bad 帧显著低于 good 帧）通过才启用，否则退回 indicator。")
        self.p_op = make_combo(["awr", "crr", "linear"], "awr", "策略改进算子")
        self.p_tau = make_float_spin(1.0, 0.05, 20.0, 2, "AWR 温度")
        self.p_beta = make_float_spin(1.0, 0.0, 10.0, 2, "linear 算子斜率")
        self.p_neg = make_float_spin(0.1, 0.0, 1.0, 2, "负优势帧权重下限")
        self.p_itv = make_float_spin(2.0, 1.0, 10.0, 2, "干预帧乘子")
        self.p_guard = make_float_spin(1.0, 0.0, 5.0, 2, "critic 分离度守卫阈值（×σ）")
        self.p_gbeta = make_float_spin(0.5, 0.0, 3.0, 2, "组相对优势调制权重 β_g（A += β_g×group_adv）")
        self.p_noguard = QCheckBox("关闭 critic 守卫（--no-critic-guard，慎用）")
        self.p_write = QCheckBox("写回 pg_advantage 列（--write）")
        self.p_write.setChecked(True)
        lay.addWidget(ParamRow("operator", self.p_op))
        lay.addWidget(ParamRow("tau", self.p_tau))
        lay.addWidget(ParamRow("beta", self.p_beta))
        lay.addWidget(ParamRow("negative", self.p_neg))
        lay.addWidget(ParamRow("intervention", self.p_itv))
        lay.addWidget(ParamRow("guard-frac", self.p_guard))
        lay.addWidget(ParamRow("group-beta", self.p_gbeta))
        lay.addWidget(self.p_noguard)
        lay.addWidget(self.p_write)
        btn = QPushButton("运行 ⑥")
        btn.clicked.connect(lambda: self._run_step("pg"))
        lay.addWidget(btn)
        self._steps["pg"] = box
        return box

    # ---------- 步骤 ⑦ 权重预览 ----------
    def _build_preview_step(self) -> QGroupBox:
        box = QGroupBox()
        lay = self._add_step_title(
            box, "⑦ 权重分布预览（recap_sidecar.py，P1.2）",
            "训练前查看 R2 加权（filtered BC）权重分布是否合理，不写文件。")
        self.pv_pos = make_float_spin(1.0, 0.0, 5.0, 2, "good 帧权重")
        self.pv_neg = make_float_spin(0.1, 0.0, 2.0, 2, "bad 帧权重")
        self.pv_itv = make_float_spin(2.0, 0.0, 10.0, 2, "干预帧乘子")
        lay.addWidget(ParamRow("positive-weight", self.pv_pos))
        lay.addWidget(ParamRow("negative-weight", self.pv_neg))
        lay.addWidget(ParamRow("intervention-weight", self.pv_itv))
        btn = QPushButton("运行 ⑦")
        btn.clicked.connect(lambda: self._run_step("preview"))
        lay.addWidget(btn)
        self._steps["preview"] = box
        return box

    # ---------- 步骤 ⑧ 训练前门禁 ----------
    def _build_check_step(self) -> QGroupBox:
        box = QGroupBox()
        lay = self._add_step_title(
            box, "⑧ 训练前门禁 + 启动建议（recap_check.py）",
            "检查 ①~⑦ 产物完整性：sidecar 新鲜度 / pg_advantage 列（硬）/"
            "action≠state 抽查（硬，问题5 防线）/ returns / 价值头 / TD / 组优势。"
            "通过后给出 yaml 修改建议与训练启动命令。"
            "不进一键链，链跑完后手动执行；❌ 硬失败会指出缺失步骤。")
        btn = QPushButton("运行 ⑧")
        btn.clicked.connect(lambda: self._run_step("check"))
        lay.addWidget(btn)
        self._steps["check"] = box
        return box

    # ---------- 数据集选择 ----------
    def _refresh_datasets(self):
        cur = self.ds_combo.currentText()
        self.ds_combo.clear()
        found = []
        data_dir = DEBUGGER_DIR / "lerobot_data"
        if data_dir.is_dir():
            for d in sorted(data_dir.iterdir()):
                if d.is_dir() and (d / "meta" / "info.json").is_file():
                    found.append(d.name)
        self.ds_combo.addItems(found)
        if cur in found:
            self.ds_combo.setCurrentText(cur)
        elif found:
            self.ds_combo.setCurrentText("ela3_demo6" if "ela3_demo6" in found else found[0])

    def _sync_tag_from_cfail(self, v: int):
        """cfail 改动时，tag 形如 cfail<数字> 则自动同步（③④⑤ 共用同一 tag）。"""
        import re
        if re.fullmatch(r"cfail\d+", self.tag_edit.text().strip()):
            self.tag_edit.setText(f"cfail{v}")

    def _browse_dataset(self):
        d = QFileDialog.getExistingDirectory(self, "选择数据集目录",
                                             str(DEBUGGER_DIR / "lerobot_data"))
        if d:
            self.ds_combo.setCurrentText(d)

    def _dataset(self) -> str:
        t = self.ds_combo.currentText().strip()
        if not t:
            return ""
        p = Path(t)
        if p.is_absolute() or p.exists():
            return str(p)
        return str(DEBUGGER_DIR / "lerobot_data" / t)

    def _python(self) -> str:
        t = self.py_edit.text().strip()
        return t or DEFAULT_PYTHON

    # ---------- 命令构建 ----------
    def _build_cmd(self, step: str) -> list[str] | None:
        ds = self._dataset()
        if not ds:
            self._append("⚠️ 请先选择数据集\n")
            return None
        tag = self.tag_edit.text().strip() or f"cfail{self.r_cfail.value()}"
        if step == "sidecar":
            args = ["visualize_data/make_recap_sidecar.py", "--dataset", ds,
                    "--bad-backprop", str(self.s_bb.value()),
                    "--terminal-fail", str(self.s_tf.value())]
            if self.s_dry.isChecked():
                args.append("--dry-run")
            return args
        if step == "group":
            args = ["recap_group_advantage.py", "--dataset", ds,
                    "--takeover-penalty", str(self.g_pen.value()),
                    "--min-group", str(self.g_min.value())]
            if self.g_dry.isChecked():
                args.append("--dry-run")
            return args
        if step == "returns":
            return ["recap_value.py", "returns", "--dataset", ds,
                    "--cfail", str(self.r_cfail.value()),
                    "--gamma", str(self.r_gamma.value()),
                    "--unrated", self.r_unrated.currentText(),
                    "--tag", tag]
        if step == "vtrain":
            return ["recap_value.py", "train", "--dataset", ds,
                    "--backend", self.v_backend.currentText(),
                    "--pi05-path", self.v_pi05.text().strip() or "./pretrained_models/pi05_base",
                    "--epochs", str(self.v_epochs.value()),
                    "--holdout", str(self.v_holdout.value()),
                    "--tag", tag]
        if step == "adv":
            args = ["recap_value.py", "advantage", "--dataset", ds,
                    "--tag", tag,
                    "--lookahead", str(self.a_lookahead.value()),
                    "--quantile", str(self.a_quantile.value())]
            if self.a_fill.isChecked():
                args.append("--fill-unlabeled")
            return args
        if step == "pg":
            args = ["recap_pg.py", "--dataset", ds,
                    "--operator", self.p_op.currentText(),
                    "--tau", str(self.p_tau.value()),
                    "--beta", str(self.p_beta.value()),
                    "--negative", str(self.p_neg.value()),
                    "--intervention", str(self.p_itv.value()),
                    "--guard-frac", str(self.p_guard.value()),
                    "--group-beta", str(self.p_gbeta.value())]
            if self.p_noguard.isChecked():
                args.append("--no-critic-guard")
            if self.p_write.isChecked():
                args.append("--write")
            return args
        if step == "preview":
            return ["recap_sidecar.py", "--dataset", ds,
                    "--positive-weight", str(self.pv_pos.value()),
                    "--negative-weight", str(self.pv_neg.value()),
                    "--intervention-weight", str(self.pv_itv.value())]
        if step == "check":
            return ["recap_check.py", "--dataset", ds, "--tag", tag]
        return None

    # ---------- 运行 ----------
    def _critic_records(self, ds: str) -> list[str]:
        """列出该数据集已有的特征缓存/价值头记录（④ vtrain 重算/复用判断用）。"""
        meta = Path(ds) / "meta"
        found = []
        for name in ["critic_feats_siglip.npz", "critic_feats_state.npz", "critic_head.pt"]:
            p = meta / name
            if p.is_file():
                found.append(str(p))
        return found

    def _ask_recompute(self, records: list[str]) -> str:
        """弹窗询问如何处理旧 critic 记录。返回 recompute / reuse / cancel。"""
        msg = QMessageBox(self)
        msg.setWindowTitle("发现已有 critic 记录")
        msg.setText(
            "该数据集已存在 critic 记录：\n\n" + "\n".join(records) +
            "\n\n是否删除旧记录并重新计算特征？\n"
            "（改过 recap_critic.py 代码后建议选「重新计算」）"
        )
        btn_recompute = msg.addButton("重新计算", QMessageBox.ButtonRole.AcceptRole)
        btn_reuse = msg.addButton("复用缓存", QMessageBox.ButtonRole.DestructiveRole)
        btn_cancel = msg.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_reuse)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked is btn_recompute:
            return "recompute"
        if clicked is btn_reuse:
            return "reuse"
        return "cancel"

    def _pre_step_check(self, step: str) -> bool:
        """运行前的额外检查。返回 False 表示该步骤被取消，不应运行。"""
        if step == "vtrain":
            ds = self._dataset()
            if ds:
                records = self._critic_records(ds)
                if records:
                    ret = self._ask_recompute(records)
                    if ret == "recompute":
                        for f in records:
                            try:
                                Path(f).unlink()
                            except OSError:
                                pass
                        self._append("[已删除旧特征缓存，将重新提取特征]\n", color="#a60")
                        return True
                    if ret == "reuse":
                        self._append("[复用已有特征缓存（命中后不再重提，直接训价值头）]\n")
                        return True
                    self._append("[已取消 ④ 价值头训练]\n", color="#888")
                    return False
        return True

    def _run_step(self, step: str):
        if not self._pre_step_check(step):
            return
        args = self._build_cmd(step)
        if args is None:
            return
        self._append(f"\n$ python -u {' '.join(args)}\n", color="#0a6")
        self._start_process(args)

    def _run_chain(self):
        self.chain_queue = [
            ("sidecar", "①"),
            ("group", "②"),
            ("returns", "③"),
            ("vtrain", "④"),
            ("adv", "⑤"),
            ("pg", "⑥"),
        ]
        self._append("\n========== 一键全流程 ①→⑥ ==========\n", color="#06c")
        self._run_next_chain()

    def _run_next_chain(self):
        if not self.chain_queue:
            self._append("========== 全流程结束 ==========\n", color="#06c")
            return
        step, label = self.chain_queue.pop(0)
        if not self._pre_step_check(step):
            self._append(f"⚠️ 步骤 {label} 被取消，全流程中断\n", color="#c00")
            self.chain_queue = []
            return
        args = self._build_cmd(step)
        if args is None:
            self._append(f"⚠️ 步骤 {label} 跳过（无数据集）\n")
            self._run_next_chain()
            return
        self._append(f"\n---------- 步骤 {label} ----------\n", color="#06c")
        self._append(f"$ python -u {' '.join(args)}\n", color="#0a6")
        self._start_process(args)

    def _start_process(self, args: list[str]):
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self._append("⚠️ 已有步骤在运行，请先停止\n")
            return
        self.process = QProcess(self)
        self.process.setProgram(self._python())
        self.process.setArguments(["-u", *args])
        self.process.setWorkingDirectory(str(DEBUGGER_DIR))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("HF_HUB_OFFLINE", "1")
        env.insert("TOKENIZERS_PARALLELISM", "false")
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)
        self.btn_stop.setEnabled(True)
        self.process.start()

    def _stop(self):
        if self.process is not None:
            self.process.kill()
            self._append("\n[已请求停止]\n", color="#c00")

    def _on_stdout(self):
        self._append(bytes(self.process.readAllStandardOutput()).decode(errors="replace"))

    def _on_stderr(self):
        self._append(bytes(self.process.readAllStandardError()).decode(errors="replace"),
                    color="#a60")

    def _on_finished(self, code, status):
        self.btn_stop.setEnabled(False)
        self._append(f"\n[进程结束 exit={code}]\n", color="#888")
        self.process = None
        if self.chain_queue:
            if code == 0:
                self._run_next_chain()
            else:
                self._append("⚠️ 上一步失败，全流程中断（可手动修正后重跑）\n", color="#c00")
                self.chain_queue = []

    def _append(self, text: str, color: str | None = None):
        cur = self.log.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        if color:
            self.log.setTextColor(QColor(color))
        cur.insertText(text)
        self.log.setTextCursor(cur)
        self.log.ensureCursorVisible()


def main():
    app = QApplication(sys.argv)
    win = RecapPreprocessWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

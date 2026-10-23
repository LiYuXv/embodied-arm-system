import json
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

DEFAULT_BASE = str(Path(__file__).resolve().parent / "lerobot_data")

NUMERIC_FEATURES = [
    "observation.state",
    "observation.gripper_torque",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]
DATA_COLUMN_ORDER = [
    "observation.state",
    "observation.gripper_torque",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "language_persistent",
    "language_events",
]
QUANTILES = [("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)]

AUX_FEATURES = {"language_persistent", "language_events"}
VIDEO_PROPS = [
    "video.codec",
    "video.pix_fmt",
    "video.height",
    "video.width",
    "video.fps",
    "video.channels",
]


def check_format_compatibility(named_infos):
    """检查多个数据集的格式是否兼容。

    Args:
        named_infos: [(名称, info.json dict), ...]，至少 2 个

    Returns:
        (global_issues, drop_plan)：
        - global_issues: 全局元数据不一致列表（fps/版本/robot_type），
          继续合并时将采用第一个数据集的值
        - drop_plan: [(特征名, 原因)]，继续合并时将舍弃这些特征
    """
    global_issues = []
    drop_plan = []
    if len(named_infos) < 2:
        return global_issues, drop_plan
    base_name, base_info = named_infos[0]

    def add_drop(fk, reason):
        for existing, _ in drop_plan:
            if existing == fk:
                return
        drop_plan.append((fk, reason))

    for name, info in named_infos[1:]:
        tag = f"[{name}] vs [{base_name}]"
        for key, label in [
            ("codebase_version", "数据格式版本"),
            ("fps", "fps"),
            ("robot_type", "robot_type"),
        ]:
            bv, sv = base_info.get(key), info.get(key)
            if bv != sv:
                global_issues.append(
                    f"{tag} {label} 不一致: {sv} != {bv} (继续将采用 [{base_name}] 的值 {bv})"
                )

        base_feats = base_info.get("features", {})
        feats = info.get("features", {})
        all_keys = list(dict.fromkeys(list(base_feats.keys()) + list(feats.keys())))
        for fk in all_keys:
            bv = base_feats.get(fk)
            sv = feats.get(fk)
            if bv is None or sv is None:
                missing = name if bv is not None else base_name
                add_drop(fk, f"在 [{missing}] 中缺失")
                continue
            if sv.get("dtype") != bv.get("dtype"):
                add_drop(fk, f"dtype 不一致: {sv.get('dtype')} != {bv.get('dtype')} ({tag})")
                continue
            if list(sv.get("shape") or []) != list(bv.get("shape") or []):
                add_drop(fk, f"维度不一致: {sv.get('shape')} != {bv.get('shape')} ({tag})")
                continue
            if bv.get("dtype") == "video":
                binfo = bv.get("info", {}) or {}
                sinfo = sv.get("info", {}) or {}
                bad_props = [
                    f"{prop}: {sinfo.get(prop)} != {binfo.get(prop)}"
                    for prop in VIDEO_PROPS
                    if sinfo.get(prop) != binfo.get(prop)
                ]
                if bad_props:
                    add_drop(fk, f"视频属性不一致 ({'; '.join(bad_props)}) ({tag})")
    return global_issues, drop_plan


def preflight_check(base_dir, source_names):
    """只读取各源数据集的 meta/info.json 做快速兼容性预检（不加载数据）。"""
    named_infos = []
    for name in source_names:
        info_path = Path(base_dir) / name / "meta" / "info.json"
        named_infos.append((name, json.loads(info_path.read_text(encoding="utf-8"))))
    return check_format_compatibility(named_infos)


def feature_stats(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr2 = arr.reshape(-1, 1)
    else:
        arr2 = arr
    out = {
        "min": arr2.min(axis=0),
        "max": arr2.max(axis=0),
        "mean": arr2.mean(axis=0),
        "std": arr2.std(axis=0),
        "count": np.array([len(arr2)]),
    }
    for name, q in QUANTILES:
        out[name] = np.quantile(arr2, q, axis=0)
    return out


def episode_frame(df_ep):
    return df_ep.sort_values("index").reset_index(drop=True)


def compute_episode_stats(df_ep, numeric_cols):
    stats = {}
    for col in numeric_cols:
        if col in ("observation.state", "action"):
            arr = np.stack(df_ep[col].values).astype(np.float64)
        else:
            arr = df_ep[col].to_numpy(dtype=np.float64)
        stats[col] = feature_stats(arr)
    return stats


def pool_image_stats(parts):
    """池化各 episode 的图像统计量。

    注意：lerobot 录制时 per-episode 图像统计存为 (C,)，而全局 stats.json
    要求 (C,1,1)（save_episode 的 aggregate_stats 会校验该形状）。
    此处统一先 reshape 为 (C,1,1) 再池化。
    """

    def norm(v):
        a = np.asarray(v, dtype=np.float64)
        return a.reshape(-1, 1, 1) if a.ndim == 1 else a

    counts = np.array([float(np.asarray(p["count"]).reshape(-1)[0]) for p in parts])
    total = counts.sum()
    means = np.stack([norm(p["mean"]) for p in parts])
    stds = np.stack([norm(p["std"]) for p in parts])
    mean = (means * counts.reshape((-1,) + (1,) * (means.ndim - 1))).sum(axis=0) / total
    var = (
        counts.reshape((-1,) + (1,) * (means.ndim - 1))
        * (stds**2 + (means - mean) ** 2)
    ).sum(axis=0) / total
    out = {
        "min": np.min(np.stack([norm(p["min"]) for p in parts]), axis=0),
        "max": np.max(np.stack([norm(p["max"]) for p in parts]), axis=0),
        "mean": mean,
        "std": np.sqrt(var),
        "count": np.array([int(total)]),
    }
    for name, _ in QUANTILES:
        qs = np.stack([norm(p[name]) for p in parts])
        out[name] = (qs * counts.reshape((-1,) + (1,) * (qs.ndim - 1))).sum(axis=0) / total
    return out


def normalize_image_stats_shape(stats, channels):
    """lerobot 的 aggregate_stats 要求图像统计量形状为 (C,1,1)；
    录制管线写入 meta/episodes 的 per-episode 图像是扁平 (C,)，聚合前需规整。"""
    for k, v in stats.items():
        if k == "count":
            continue
        arr = np.asarray(v, dtype=np.float64)
        if arr.size == channels and arr.shape != (channels, 1, 1):
            stats[k] = arr.reshape(channels, 1, 1)
    return stats


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


class SourceDataset:
    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.name
        self.info = json.loads((self.path / "meta" / "info.json").read_text(encoding="utf-8"))
        ep_files = sorted((self.path / "meta" / "episodes").rglob("*.parquet"))
        self.episodes = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
        self.episodes = self.episodes.sort_values("episode_index").reset_index(drop=True)
        self.tasks = pd.read_parquet(self.path / "meta" / "tasks.parquet")
        data_files = sorted((self.path / "data").rglob("*.parquet"))
        frames = []
        for f in data_files:
            try:
                frames.append(pd.read_parquet(f))
            except Exception:
                continue
        self.data = pd.concat(frames, ignore_index=True)
        self.data_by_ep = {
            int(k): episode_frame(g) for k, g in self.data.groupby("episode_index")
        }
        self.task_by_index = {
            int(row["task_index"]): str(idx) for idx, row in self.tasks.iterrows()
        }

    @property
    def video_keys(self):
        return [k for k, v in self.info["features"].items() if v.get("dtype") == "video"]


def build_merged_dataset(sources, target_path, log, progress, drop_features=None, allow_overrides=False):
    global_issues, drop_plan = check_format_compatibility(
        [(s.name, s.info) for s in sources]
    )
    detected = {f for f, _ in drop_plan}
    allowed = set(drop_features or [])
    if detected - allowed:
        raise ValueError(
            "存在未确认舍弃的不兼容特征, 已中止合并: "
            + ", ".join(sorted(detected - allowed))
        )
    if global_issues and not allow_overrides:
        raise ValueError(
            "存在未确认的全局元数据不一致, 已中止合并:\n" + "\n".join(global_issues)
        )
    drop = detected
    for f, reason in drop_plan:
        log(f"舍弃特征 {f}: {reason}")
    for issue in global_issues:
        log(f"全局元数据不一致: {issue}")

    fps = sources[0].info["fps"]
    chunks_size = int(sources[0].info.get("chunks_size", 1000))
    video_keys = [k for k in sources[0].video_keys if k not in drop]
    if not video_keys:
        raise ValueError("所有相机特征均被舍弃, 无法合并")

    all_data_cols = []
    for s in sources:
        for c in s.data.columns:
            if c not in all_data_cols:
                all_data_cols.append(c)
    common_data_cols = [
        c
        for c in all_data_cols
        if c not in drop and all(c in s.data.columns for s in sources)
    ]
    dropped_cols = [c for c in all_data_cols if c not in common_data_cols]
    if dropped_cols:
        log(f"以下列不进入合并结果: {dropped_cols}")

    merged_tasks = []
    for s in sources:
        for t in s.task_by_index.values():
            if t not in merged_tasks:
                merged_tasks.append(t)
    task_to_new = {t: i for i, t in enumerate(merged_tasks)}
    task_remaps = {
        s.name: {old: task_to_new[t] for old, t in s.task_by_index.items()} for s in sources
    }

    ep_columns = list(sources[0].episodes.columns)
    for s in sources[1:]:
        for c in s.episodes.columns:
            if c not in ep_columns:
                ep_columns.append(c)
    ep_columns = [
        c
        for c in ep_columns
        if not any(
            c.startswith(f"stats/{f}/") or c.startswith(f"videos/{f}/") for f in drop
        )
    ]

    target_path = Path(target_path)
    if target_path.exists() and any(target_path.iterdir()):
        raise FileExistsError(f"目标路径已存在且非空: {target_path}")
    (target_path / "data").mkdir(parents=True)
    (target_path / "meta" / "episodes").mkdir(parents=True)
    for vk in video_keys:
        (target_path / "videos" / vk).mkdir(parents=True)

    numeric_cols = [c for c in NUMERIC_FEATURES if c in common_data_cols]
    data_cols = [c for c in DATA_COLUMN_ORDER if c in common_data_cols] + [
        c for c in common_data_cols if c not in DATA_COLUMN_ORDER
    ]

    plan = []
    for s in sources:
        for _, meta_row in s.episodes.iterrows():
            ep_idx = int(meta_row["episode_index"])
            df_ep = s.data_by_ep.get(ep_idx)
            if df_ep is None or len(df_ep) == 0:
                log(f"跳过: {s.name} episode {ep_idx} 没有数据行")
                continue
            missing_video = False
            for vk in video_keys:
                cidx = int(meta_row[f"videos/{vk}/chunk_index"])
                fidx = int(meta_row[f"videos/{vk}/file_index"])
                vpath = s.path / "videos" / vk / f"chunk-{cidx:03d}" / f"file-{fidx:03d}.mp4"
                if not vpath.exists():
                    log(f"跳过: {s.name} episode {ep_idx} 缺少视频 {vpath}")
                    missing_video = True
                    break
                if float(meta_row[f"videos/{vk}/from_timestamp"]) != 0.0:
                    log(f"跳过: {s.name} episode {ep_idx} 的 {vk} from_timestamp != 0, 不支持切片视频")
                    missing_video = True
                    break
            if missing_video:
                continue
            plan.append((s, meta_row, df_ep))

    total = len(plan)
    if total == 0:
        raise ValueError("没有可合并的 episode")

    meta_rows = []
    global_numeric = {c: [] for c in numeric_cols}
    image_stats_by_key = {vk: [] for vk in video_keys}
    global_index = 0

    for g, (s, meta_row, df_ep) in enumerate(plan):
        chunk = g // chunks_size
        fidx = g % chunks_size
        df_ep = df_ep.copy()
        n = len(df_ep)
        df_ep["frame_index"] = np.arange(n, dtype=np.int64)
        df_ep["timestamp"] = (np.arange(n, dtype=np.float64) / fps).astype(np.float32)
        df_ep["episode_index"] = g
        df_ep["index"] = np.arange(global_index, global_index + n, dtype=np.int64)
        remap = task_remaps[s.name]
        unknown = sorted({int(v) for v in df_ep["task_index"].unique()} - set(remap))
        if unknown:
            raise ValueError(
                f"{s.name} episode {int(meta_row['episode_index'])} 的 data 含 tasks 元数据中"
                f"不存在的 task_index={unknown}，已中止合并。"
                "请先修复该源数据集的 task_index（见 数据问题分析.md 问题7）"
            )
        df_ep["task_index"] = df_ep["task_index"].map(
            lambda v: remap[int(v)]
        ).astype(np.int64)

        data_dir = target_path / "data" / f"chunk-{chunk:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        df_ep[data_cols].to_parquet(data_dir / f"file-{fidx:03d}.parquet", index=False)

        row = {}
        for c in ep_columns:
            row[c] = meta_row[c] if c in meta_row.index else None
        row["episode_index"] = g
        row["length"] = n
        row["data/chunk_index"] = chunk
        row["data/file_index"] = fidx
        row["dataset_from_index"] = global_index
        row["dataset_to_index"] = global_index + n
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0

        for vk in video_keys:
            sc = int(meta_row[f"videos/{vk}/chunk_index"])
            sf = int(meta_row[f"videos/{vk}/file_index"])
            src_video = s.path / "videos" / vk / f"chunk-{sc:03d}" / f"file-{sf:03d}.mp4"
            dst_dir = target_path / "videos" / vk / f"chunk-{chunk:03d}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_video, dst_dir / f"file-{fidx:03d}.mp4")
            row[f"videos/{vk}/chunk_index"] = chunk
            row[f"videos/{vk}/file_index"] = fidx
            row[f"videos/{vk}/from_timestamp"] = 0.0
            row[f"videos/{vk}/to_timestamp"] = n / fps

        ep_stats = compute_episode_stats(df_ep, numeric_cols)
        for feat, st in ep_stats.items():
            for stat_name, val in st.items():
                row[f"stats/{feat}/{stat_name}"] = np.asarray(val)
        for vk in video_keys:
            for stat_name in ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]:
                key = f"stats/{vk}/{stat_name}"
                if key in meta_row.index and meta_row[key] is not None:
                    row[key] = np.asarray(meta_row[key])
            src_stat = {
                stat_name: row[f"stats/{vk}/{stat_name}"]
                for stat_name in ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
                if f"stats/{vk}/{stat_name}" in row and row[f"stats/{vk}/{stat_name}"] is not None
            }
            if src_stat:
                image_stats_by_key[vk].append(src_stat)

        for c in numeric_cols:
            if c in ("observation.state", "action"):
                global_numeric[c].append(np.stack(df_ep[c].values).astype(np.float64))
            else:
                global_numeric[c].append(df_ep[c].to_numpy(dtype=np.float64))

        meta_rows.append(row)
        global_index += n
        progress(g + 1, total, f"{s.name} ep{int(meta_row['episode_index'])} -> 新 ep{g} ({n} 帧)")

    meta_df = pd.DataFrame(meta_rows, columns=ep_columns)
    ep_dir = target_path / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    meta_df.to_parquet(ep_dir / "file-000.parquet", index=False)

    tasks_df = pd.DataFrame({"task_index": list(range(len(merged_tasks)))}, index=merged_tasks)
    tasks_df.to_parquet(target_path / "meta" / "tasks.parquet", index=True)

    stats_out = {}
    for c, parts in global_numeric.items():
        arr = np.concatenate(parts)
        stats_out[c] = feature_stats(arr)
    for vk, parts in image_stats_by_key.items():
        if parts:
            pooled = pool_image_stats(parts)
            shape = sources[0].info["features"].get(vk, {}).get("shape") or []
            if len(shape) >= 1:
                pooled = normalize_image_stats_shape(pooled, int(shape[-1]))
            stats_out[vk] = pooled
    (target_path / "meta" / "stats.json").write_text(
        json.dumps(to_jsonable(stats_out), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    info = dict(sources[0].info)
    info["total_episodes"] = total
    info["total_frames"] = int(global_index)
    info["total_tasks"] = len(merged_tasks)
    info["splits"] = {"train": f"0:{global_index}"}
    drop_feats = [
        k
        for k, v in info["features"].items()
        if k in drop or (v.get("dtype") not in ("video",) and k not in common_data_cols)
    ]
    for k in drop_feats:
        info["features"].pop(k, None)
        log(f"info.json 移除特征: {k} (并非所有源都有该数据列)")
    (target_path / "meta" / "info.json").write_text(
        json.dumps(to_jsonable(info), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return total, int(global_index)


def merge_into(base_dir, source_names, target_name, log, progress, drop_features=None, allow_overrides=False):
    base_dir = Path(base_dir)
    sources = []
    for name in source_names:
        log(f"加载源数据集: {name}")
        sources.append(SourceDataset(base_dir / name))
    tmp_dir = Path(tempfile.mkdtemp(prefix=".merge_tmp_", dir=base_dir))
    try:
        total_eps, total_frames = build_merged_dataset(
            sources, tmp_dir, log, progress,
            drop_features=drop_features, allow_overrides=allow_overrides,
        )
        target = base_dir / target_name
        backup = None
        if target.exists():
            backup = base_dir / f"{target_name}.bak_{datetime.now():%Y%m%d_%H%M%S}"
            target.rename(backup)
            log(f"原目标已备份为: {backup.name}")
        tmp_dir.rename(target)
        log(f"完成: {total_eps} episodes / {total_frames} 帧 -> {target}")
        return target, backup, total_eps, total_frames
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


class MergeWorker(QThread):
    sig_log = pyqtSignal(str)
    sig_progress = pyqtSignal(int, int, str)
    sig_done = pyqtSignal(bool, str)

    def __init__(self, base_dir, source_names, target_name, drop_features=None, allow_overrides=False):
        super().__init__()
        self.base_dir = base_dir
        self.source_names = source_names
        self.target_name = target_name
        self.drop_features = drop_features
        self.allow_overrides = allow_overrides

    def run(self):
        try:
            target, backup, eps, frames = merge_into(
                self.base_dir,
                self.source_names,
                self.target_name,
                log=self.sig_log.emit,
                progress=lambda a, b, m: self.sig_progress.emit(a, b, m),
                drop_features=self.drop_features,
                allow_overrides=self.allow_overrides,
            )
            msg = f"合并成功: {eps} episodes / {frames} 帧"
            if backup:
                msg += f"\n原目标备份: {backup.name}"
            self.sig_done.emit(True, msg)
        except Exception as e:
            self.sig_log.emit(traceback.format_exc())
            self.sig_done.emit(False, f"合并失败: {e}")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LeRobot 数据集合并工具 (v3.0)")
        self.resize(860, 640)
        self.worker = None

        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("数据根目录:"))
        self.base_edit = QLineEdit(DEFAULT_BASE)
        row.addWidget(self.base_edit)
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self.on_browse)
        row.addWidget(btn_browse)
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self.refresh_list)
        row.addWidget(btn_refresh)
        layout.addLayout(row)

        layout.addWidget(QLabel("源数据集 (勾选要合并的, 按列表顺序合并):"))
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        row2 = QHBoxLayout()
        btn_all = QPushButton("全选")
        btn_all.clicked.connect(self.select_all)
        btn_none = QPushButton("全不选")
        btn_none.clicked.connect(self.select_none)
        btn_up = QPushButton("上移")
        btn_up.clicked.connect(lambda: self.move_item(-1))
        btn_down = QPushButton("下移")
        btn_down.clicked.connect(lambda: self.move_item(1))
        row2.addWidget(btn_all)
        row2.addWidget(btn_none)
        row2.addWidget(btn_up)
        row2.addWidget(btn_down)
        row2.addStretch()
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("目标数据集名称 (可填已有名称覆盖合并, 或新名称):"))
        self.target_combo = QComboBox()
        self.target_combo.setEditable(True)
        row3.addWidget(self.target_combo)
        self.btn_merge = QPushButton("开始合并")
        self.btn_merge.clicked.connect(self.on_merge)
        row3.addWidget(self.btn_merge)
        layout.addLayout(row3)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit)

        self.refresh_list()

    def log(self, msg):
        self.log_edit.appendPlainText(msg)

    def find_datasets(self):
        base = Path(self.base_edit.text())
        if not base.is_dir():
            return []
        out = []
        for p in sorted(base.iterdir()):
            if p.is_dir() and not p.name.startswith(".") and (p / "meta" / "info.json").exists():
                try:
                    info = json.loads((p / "meta" / "info.json").read_text(encoding="utf-8"))
                    out.append((p.name, info.get("total_episodes", "?"), info.get("total_frames", "?")))
                except Exception:
                    out.append((p.name, "?", "?"))
        return out

    def refresh_list(self):
        checked = {self.list_widget.item(i).text().split("  (")[0]
                   for i in range(self.list_widget.count())
                   if self.list_widget.item(i).checkState() == Qt.CheckState.Checked}
        self.list_widget.clear()
        self.target_combo.clear()
        for name, eps, frames in self.find_datasets():
            item = QListWidgetItem(f"{name}  ({eps} eps / {frames} 帧)")
            item.setCheckState(Qt.CheckState.Checked if name in checked else Qt.CheckState.Unchecked)
            self.list_widget.addItem(item)
            self.target_combo.addItem(name)

    def on_browse(self):
        d = QFileDialog.getExistingDirectory(self, "选择数据根目录", self.base_edit.text())
        if d:
            self.base_edit.setText(d)
            self.refresh_list()

    def select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Checked)

    def select_none(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Unchecked)

    def move_item(self, delta):
        row = self.list_widget.currentRow()
        if row < 0:
            return
        new_row = row + delta
        if not (0 <= new_row < self.list_widget.count()):
            return
        item = self.list_widget.takeItem(row)
        self.list_widget.insertItem(new_row, item)
        self.list_widget.setCurrentRow(new_row)

    def on_merge(self):
        sources = []
        for i in range(self.list_widget.count()):
            it = self.list_widget.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                sources.append(it.text().split("  (")[0])
        target = self.target_combo.currentText().strip()
        if not sources:
            QMessageBox.warning(self, "提示", "请至少勾选一个源数据集")
            return
        if not target:
            QMessageBox.warning(self, "提示", "请填写目标数据集名称")
            return
        existing = {name for name, _, _ in self.find_datasets()}
        try:
            global_issues, drop_plan = preflight_check(self.base_edit.text(), sources)
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
                self.log("用户取消合并")
                return
            drop_features = [f for f, _ in drop_plan]
            allow_overrides = bool(global_issues)
            if drop_features:
                self.log("用户确认继续, 将舍弃特征: " + ", ".join(drop_features))
            if allow_overrides:
                self.log("用户确认继续, 全局元数据采用首个数据集的值")
        if target in existing:
            if target in sources:
                text = f"目标 {target} 同时是源数据集, 合并后原数据将被替换 (自动备份)。继续?"
            else:
                text = f"目标 {target} 已存在且不在源列表中, 其内容将被合并结果替换 (自动备份)。继续?"
            ret = QMessageBox.question(self, "确认覆盖", text)
            if ret != QMessageBox.StandardButton.Yes:
                return
        self.btn_merge.setEnabled(False)
        self.progress.setValue(0)
        self.worker = MergeWorker(
            self.base_edit.text(), sources, target,
            drop_features=drop_features, allow_overrides=allow_overrides,
        )
        self.worker.sig_log.connect(self.log)
        self.worker.sig_progress.connect(self.on_progress)
        self.worker.sig_done.connect(self.on_done)
        self.worker.start()

    def on_progress(self, cur, total, msg):
        self.progress.setMaximum(total)
        self.progress.setValue(cur)
        self.setWindowTitle(f"合并中 {cur}/{total} - {msg}")

    def on_done(self, ok, msg):
        self.btn_merge.setEnabled(True)
        self.setWindowTitle("LeRobot 数据集合并工具 (v3.0)")
        self.log(msg)
        self.refresh_list()
        if ok:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.critical(self, "失败", msg)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

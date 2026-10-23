#!/usr/bin/env python3
"""
自动语言标注模块（子任务级）。

受 LeRobot 0.6 的 `lerobot-annotate`（VLM 子任务标注）启发，提供一个
**不依赖 VLM** 的轻量实现：基于运动学信号（末端位姿 + 夹爪）自动将单个
Episode 切分为连续的“原子子任务”片段，并按用户给定的预设子任务名称顺序
匹配，生成带时间戳的语言描述。

核心流程：
    输入位姿序列  →  运动能量分段  →  段数对齐预设名  →  语言标注区间

输出（LanguageAnnotationStore, JSON）：
{
    "version": "1.0",
    "label_type": "subtask_language",
    "labels": {"抓取": {"color": "#4CAF50"}, "移动": {"color": "#FF9800"}, ...},
    "episodes": {
        "0": [
            {"start": 0,  "end": 45, "task": "抓取", "note": "auto"},
            {"start": 46, "end": 80, "task": "移动", "note": "auto"}
        ]
    }
}

与 LeRobot 数据集的对接：
    write_language_to_dataset() 将逐帧 language 字符串写入 data/*.parquet，
    并在 meta/info.json 注册特征（参照 expand_annotations.py 的范式）。

作为独立脚本运行时，可对指定数据集/episode 做命令行预览。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------- 配色板（子任务区分色，尽量可区分） ----------
DEFAULT_PALETTE = [
    "#4CAF50",  # 绿
    "#FF9800",  # 橙
    "#2196F3",  # 蓝
    "#E91E63",  # 粉
    "#9C27B0",  # 紫
    "#00BCD4",  # 青
    "#FFC107",  # 黄
    "#795548",  # 棕
    "#3F51B5",  # 靛
    "#009688",  # 蓝绿
]


def color_for_index(i: int) -> str:
    """按索引循环取色；超过板长则在 HSV 上均匀生成。"""
    if i < len(DEFAULT_PALETTE):
        return DEFAULT_PALETTE[i]
    import colorsys
    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.85)
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


def parse_subtask_names(text: str) -> List[str]:
    """把多行/逗号/顿号分隔的输入解析为有序子任务名列表（去空、去重保序）。"""
    if not text:
        return []
    raw: List[str] = []
    for line in text.splitlines():
        for seg in line.replace("、", ",").replace(";", ",").replace("；", ",").split(","):
            seg = seg.strip()
            if seg:
                raw.append(seg)
    seen = set()
    names: List[str] = []
    for n in raw:
        if n not in seen:
            seen.add(n)
            names.append(n)
    return names


# ==================== 分段器 ====================

class SubtaskSegmenter:
    """基于运动能量的子任务分段器。

    思路：机器人操作任务通常是“运动—停顿—运动”交替的。把每一帧的归一化
    运动速度求和得到“能量信号”，经平滑后阈值化得到“活动”掩码；连续的
    活动帧合并为一个子任务片段，片段间的停顿即为子任务边界。
    """

    def __init__(
        self,
        active_threshold: float = 0.02,
        smooth_window: int = 7,
        min_segment_frames: int = 5,
        max_gap_frames: int = 8,
    ):
        self.active_threshold = active_threshold
        self.smooth_window = max(3, int(smooth_window))
        self.min_segment_frames = max(1, int(min_segment_frames))
        self.max_gap_frames = max(0, int(max_gap_frames))

    @staticmethod
    def _energy(data: Dict[str, np.ndarray]) -> np.ndarray:
        keys = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
        n = len(data.get('x', []))
        if n == 0:
            return np.zeros(0)
        # 每个轴以其全程幅度归一化，避免量纲差异
        diffs = []
        for k in keys:
            arr = np.asarray(data.get(k, []), dtype=np.float64)
            if arr.size == 0:
                arr = np.zeros(n)
            scale = float(np.ptp(arr)) if np.ptp(arr) > 1e-9 else 1.0
            d = np.zeros(n)
            d[1:] = np.abs(np.diff(arr)) / scale
            diffs.append(d)
        energy = np.sum(diffs, axis=0)
        energy[0] = energy[1] if n > 1 else 0.0
        return energy

    def _smooth(self, x: np.ndarray) -> np.ndarray:
        w = self.smooth_window
        if w < 3 or x.size < w:
            return x
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="same")

    def segment(self, data: Dict[str, np.ndarray]) -> List[Tuple[int, int]]:
        """返回活动片段区间列表 [(start, end), ...]（闭区间，帧序号）。"""
        n = len(data.get('x', []))
        if n < 2:
            return [(0, n - 1)] if n == 1 else []

        energy = self._smooth(self._energy(data))
        active = energy > self.active_threshold

        # 1) 把不超过 max_gap 的停顿桥接为活动（避免一次动作被误切成两段）
        active = self._bridge_short_gaps(active, self.max_gap_frames)

        # 2) 提取连续活动区间
        raw = self._mask_to_intervals(active)

        # 3) 过滤过短片段
        raw = [(s, e) for (s, e) in raw if (e - s + 1) >= self.min_segment_frames]
        if not raw:
            return [(0, n - 1)]

        # 4) 保证区间不重叠且覆盖合理（端点对齐到 0..n-1）
        merged = self._merge_touching(raw)
        return merged

    @staticmethod
    def _bridge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
        out = mask.copy()
        n = len(mask)
        i = 0
        while i < n:
            if not out[i]:
                j = i
                while j < n and not out[j]:
                    j += 1
                gap_len = j - i
                if 0 < gap_len <= max_gap and i > 0 and j < n:
                    out[i:j] = True
                i = j
            else:
                i += 1
        return out

    @staticmethod
    def _mask_to_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
        intervals = []
        start = None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                intervals.append((start, i - 1))
                start = None
        if start is not None:
            intervals.append((start, len(mask) - 1))
        return intervals

    @staticmethod
    def _merge_touching(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not intervals:
            return []
        ivs = sorted(intervals)
        merged = [ivs[0]]
        for s, e in ivs[1:]:
            ls, le = merged[-1]
            if s <= le + 1:
                merged[-1] = (ls, max(le, e))
            else:
                merged.append((s, e))
        return merged


# ==================== 名称匹配 ====================

def assign_names(
    segments: Sequence[Tuple[int, int]],
    names: Sequence[str],
    total_frames: int,
    repeat: bool = False,
) -> List[Dict]:
    """把预设子任务名按顺序匹配到分段上。

    repeat=False（默认，单一序列）：
    - 段数 == 名字数：1:1 顺序匹配
    - 段数 > 名字数：贪心地把“最短相邻段”并入其邻居，直到数量相等
    - 段数 < 名字数：顺序匹配前 len(segments) 个名字，多余名字忽略

    repeat=True（循环模板，如 抓取→移动→放置→归位 反复出现）：
    - 不合并任何段，全部保留
    - 名字按 names[i % len(names)] 循环贴到每一段
    - 颜色按名字固定，便于同一子任务跨循环用相同色对比
    """
    segs = [tuple(s) for s in segments]
    names = list(names)

    if not segs:
        if total_frames > 0:
            segs = [(0, total_frames - 1)]
        else:
            return []

    result: List[Dict] = []
    if repeat and names:
        # 循环模式：保留全部段，名字循环贴
        for i, (s, e) in enumerate(segs):
            name = names[i % len(names)]
            result.append({
                "start": int(s), "end": int(e), "task": name,
                "color": color_for_index(names.index(name) % len(DEFAULT_PALETTE)),
                "note": "auto-cycle",
            })
        return result

    # 单一序列模式：对齐段数与名字数
    if names:
        # 段过多：合并最短相邻段
        while len(segs) > len(names) and len(segs) > 1:
            lengths = [e - s + 1 for (s, e) in segs]
            idx = int(np.argmin(lengths))
            if idx < len(segs) - 1:
                ps, _ = segs[idx]
                _, ne = segs[idx + 1]
                segs[idx] = (ps, ne)
                del segs[idx + 1]
            else:
                ps, _ = segs[idx - 1]
                _, ne = segs[idx]
                segs[idx - 1] = (ps, ne)
                del segs[idx]
        used = names[: len(segs)]
    else:
        used = []

    for i, (s, e) in enumerate(segs):
        task = used[i] if i < len(used) else f"子任务{i + 1}"
        result.append({
            "start": int(s),
            "end": int(e),
            "task": task,
            "color": color_for_index(i),
            "note": "auto",
        })
    return result


# ==================== 强制 N 段分段（力矩 + 位姿边界评分） ====================

def compute_boundary_scores(data: Dict[str, np.ndarray], use_torque: bool = True, use_gripper: bool = False) -> np.ndarray:
    """计算每帧作为“子任务边界”的得分。

    边界 = 机械臂停顿（能量低谷）+ 夹爪开合切换 + 力矩突变（接触/释放物体）。
    力矩和夹爪状态变化权重更高，因为它们对应抓取/释放物体的真实事件。
    """
    n = len(data.get('x', []))
    if n == 0:
        return np.zeros(0)
    pose_keys = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    # 位姿能量（归一化速度和）
    pose_diffs = []
    for k in pose_keys:
        arr = np.asarray(data.get(k, []), dtype=np.float64)
        if arr.size == 0:
            arr = np.zeros(n)
        scale = float(np.ptp(arr)) if np.ptp(arr) > 1e-9 else 1.0
        d = np.zeros(n); d[1:] = np.abs(np.diff(arr)) / scale
        pose_diffs.append(d)
    pose_energy = np.sum(pose_diffs, axis=0)

    # 停顿得分：能量越低越可能是边界（平滑后的逆能量）
    sm = SubtaskSegmenter._smooth  # 复用滑动平均
    pose_smooth = sm.__func__(None, pose_energy, 9) if False else _smooth1d(pose_energy, 9)
    pause = 1.0 / (1.0 + pose_smooth * 50.0)

    # 夹爪开合切换：夹爪速度的局部峰值（抓/放事件）
    grip = np.asarray(data.get('gripper', []), dtype=np.float64)
    if grip.size == 0:
        grip = np.zeros(n)
    grip_vel = np.zeros(n); grip_vel[1:] = np.abs(np.diff(grip))
    grip_scale = float(np.ptp(grip)) if np.ptp(grip) > 1e-9 else 1.0
    grip_trans = _smooth1d(grip_vel / max(grip_scale, 1e-9), 5)

    score = pause
    if use_gripper:
        score = score + 1.5 * grip_trans

    # 力矩突变（接触/释放物体，权重大）+ 力矩状态机强边界
    if use_torque and 'gripper_torque' in data:
            tq = np.asarray(data.get('gripper_torque', []), dtype=np.float64)
            if tq.size == n:
                # 只关注正向力矩：负向值（反向电流/噪声）截断为 0，避免误触发边界
                tq = np.clip(tq, 0.0, None)
                tq_vel = np.zeros(n); tq_vel[1:] = np.abs(np.diff(tq))
                tq_scale = float(np.ptp(tq_vel)) if np.ptp(tq_vel) > 1e-9 else 1.0
                tq_trans = _smooth1d(tq_vel / tq_scale, 5)
                score = score + 2.0 * tq_trans

                # 力矩状态机：抓取(低→高上升沿)/释放(高→低下降沿)跃迁点 → 强边界
                # 力矩≈0=未夹取；上升沿=正在抓取；稳定非零=持物移动；下降沿=释放
                tq_peak = float(np.percentile(tq, 95)) if tq.size > 0 else 0.0
                if tq_peak > 1e-6:
                    low_th = 0.15 * tq_peak
                    high_th = 0.40 * tq_peak
                    state = 0
                    spike = np.zeros(n)
                    for i in range(n):
                        lvl = 1 if tq[i] >= high_th else (0 if tq[i] <= low_th else state)
                        if lvl != state:
                            spike[i] = 1.0
                            state = lvl
                    state_trans = _smooth1d(spike, 5)
                    score = score + 2.5 * state_trans

    # 抑制首尾（边界不应在开头/结尾）
    edge = max(3, n // 20)
    score[:edge] *= np.linspace(0, 1, edge)
    score[-edge:] *= np.linspace(1, 0, edge)
    return score


def _smooth1d(x: np.ndarray, w: int) -> np.ndarray:
    if w < 3 or x.size < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def segment_fixed_count(
    data: Dict[str, np.ndarray],
    n_segments: int,
    use_torque: bool = True,
    use_gripper: bool = False,
    min_seg_frac: float = 0.04,
) -> List[Tuple[int, int]]:
    """强制切出恰好 n_segments 段。

    在边界得分最高的 (n_segments-1) 个位置切分（要求最小间距），保证段数精确。
    """
    n = len(data.get('x', []))
    n_segments = max(1, int(n_segments))
    if n == 0:
        return []
    if n_segments == 1:
        return [(0, n - 1)]

    min_seg = max(3, int(n * min_seg_frac))
    scores = compute_boundary_scores(data, use_torque=use_torque, use_gripper=use_gripper).copy()

    boundaries: List[int] = []
    for _ in range(n_segments - 1):
        # 屏蔽已选边界附近的得分
        for b in boundaries:
            lo = max(0, b - min_seg)
            hi = min(n, b + min_seg + 1)
            scores[lo:hi] = -1.0
        cand = int(np.argmax(scores))
        if scores[cand] <= 0:
            break  # 没有更多有效边界，改用均分补齐
        boundaries.append(cand)

    # 若有效边界不足，用均分补齐剩余切点
    if len(boundaries) < n_segments - 1:
        existing = sorted(boundaries)
        need = n_segments - 1 - len(existing)
        cands = set(np.linspace(min_seg, n - min_seg, n_segments + need + 2, dtype=int))
        cands -= set(range(min_seg)) | set(range(n - min_seg, n))
        for b in existing:
            cands -= {bb for bb in cands if abs(bb - b) < min_seg}
        for b in sorted(cands, key=lambda x: -abs(scores[x] if x < n else 0))[:need]:
            existing.append(int(b))
        boundaries = sorted(existing)
    else:
        boundaries = sorted(boundaries)

    # 由切点构造闭区间
    edges = [0] + boundaries + [n - 1]
    segs = []
    for i in range(len(edges) - 1):
        s = max(0, edges[i])
        e = min(n - 1, edges[i + 1])
        if e >= s:
            segs.append((s, e if i < len(edges) - 2 else n - 1))
    if not segs:
        segs = [(0, n - 1)]
    return segs


def annotate_fixed_count(
    data: Dict[str, np.ndarray],
    n_segments: int,
    use_torque: bool = True,
    use_gripper: bool = False,
    labels: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """端到端：强制 N 段切分，并贴上标签（labels 可为 None，则用 步骤1/2/.. 占位）。"""
    n = len(data.get('x', []))
    segs = segment_fixed_count(data, n_segments, use_torque=use_torque, use_gripper=use_gripper)
    out = []
    for i, (s, e) in enumerate(segs):
        task = labels[i] if (labels and i < len(labels)) else f"步骤{i + 1}"
        out.append({
            "start": int(s), "end": int(e), "task": task,
            "color": color_for_index(i), "note": "auto-fixed",
        })
    return out


# ==================== 标签库（跨 episode 复用） ====================

class LabelLibrary:
    """累积各 episode 生成过的标签，供后续 episode 复用/建议。

    存储格式 (meta/label_library.json):
    {
        "version": "1.0",
        "all_labels": ["抓取方块", "移动玩偶", ...],   # 去重累积池
        "episodes": {"0": ["抓取方块","移动玩偶",...]} # 每 episode 的有序标签
    }
    """

    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.meta_dir = self.dataset_root / "meta"
        self.filepath = self.meta_dir / "label_library.json"
        self.data: Dict = {"version": "1.0", "all_labels": [], "episodes": {}}

    def load(self) -> Dict:
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                self.data.setdefault("all_labels", [])
                self.data.setdefault("episodes", {})
            except (json.JSONDecodeError, OSError):
                self.data = {"version": "1.0", "all_labels": [], "episodes": {}}
        return self.data

    def save(self) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.filepath.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.filepath)

    def set_episode_labels(self, episode_index: int, labels: Sequence[str]) -> None:
        key = str(int(episode_index))
        clean = [str(x).strip() for x in labels if str(x).strip()]
        self.data["episodes"][key] = clean
        pool = list(self.data.get("all_labels", []))
        for lb in clean:
            if lb not in pool:
                pool.append(lb)
        self.data["all_labels"] = pool

    def get_episode_labels(self, episode_index: int) -> List[str]:
        return list(self.data.get("episodes", {}).get(str(int(episode_index)), []))

    def suggest(self, limit: int = 30) -> List[str]:
        """返回累积标签池（按出现顺序），供下拉建议。"""
        return list(self.data.get("all_labels", []))[:limit]


def annotate_episode(
    data: Dict[str, np.ndarray],
    names: Sequence[str],
    sensitivity: float = 1.0,
    repeat: bool = False,
) -> List[Dict]:
    """端到端：对单个 Episode 的位姿数据做自动语言标注。

    sensitivity > 1 更敏感（切得更细），< 1 更粗。
    repeat=True 时按循环模板贴名（同子任务跨循环用相同色）。
    返回 [{start,end,task,color,note}, ...]
    """
    n = len(data.get('x', []))
    if n == 0:
        return []
    threshold = max(0.002, 0.02 / max(sensitivity, 0.1))
    min_seg = max(2, int(round(5 / max(sensitivity, 0.1))))
    seg = SubtaskSegmenter(
        active_threshold=threshold,
        min_segment_frames=min_seg,
        max_gap_frames=max(4, int(round(8 / max(sensitivity, 0.1)))),
    )
    intervals = seg.segment(data)
    return assign_names(intervals, names, n, repeat=repeat)


# ==================== 存储 ====================

class LanguageAnnotationStore:
    """管理单个 LeRobot 数据集的子任务语言标注读写。"""

    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.meta_dir = self.dataset_root / "meta"
        self.filepath = self.meta_dir / "language_annotations.json"
        self.data: Dict = self._default_structure()

    @staticmethod
    def _default_structure() -> Dict:
        return {"version": "1.0", "label_type": "subtask_language", "labels": {}, "episodes": {}}

    def load(self) -> Dict:
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                base = self._default_structure()
                base["labels"] = loaded.get("labels", {})
                base["episodes"] = loaded.get("episodes", {})
                self.data = base
            except (json.JSONDecodeError, OSError):
                self.data = self._default_structure()
        else:
            self.data = self._default_structure()
        return self.data

    def save(self) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.filepath.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.filepath)

    def set_episode(self, episode_index: int, assignments: List[Dict]) -> None:
        key = str(int(episode_index))
        cleaned = []
        for a in assignments:
            cleaned.append({
                "start": int(a["start"]),
                "end": int(a["end"]),
                "task": str(a["task"]),
                "note": str(a.get("note", "auto")),
            })
        self.data.setdefault("episodes", {})[key] = cleaned
        # 维护 labels（颜色）
        labels = self.data.setdefault("labels", {})
        for i, a in enumerate(assignments):
            t = str(a["task"])
            if t not in labels:
                labels[t] = {"color": a.get("color") or color_for_index(i)}

    def get_episode(self, episode_index: int) -> List[Dict]:
        return self.data.get("episodes", {}).get(str(int(episode_index)), [])


# ==================== 写回 LeRobot 数据集 ====================

LANGUAGE_FEATURE_NAME = "language_persistent"


def build_per_frame_language(
    dataset_root: Path,
    assignments: List[Dict],
    episode_index: int,
) -> "object":
    """构建逐帧 language 字符串（仅供预览统计）。

    返回 dict: {"covered": int, "total": int, "per_frame": np.ndarray(dtype=object)}
    """
    import pandas as pd
    meta_dir = dataset_root / "meta"
    with open(meta_dir / "info.json") as f:
        info = json.load(f)
    data_dir = dataset_root / "data"
    data_files = sorted(data_dir.rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"找不到数据文件: {data_dir}")

    frames_df = pd.concat(
        [pd.read_parquet(p, columns=["episode_index", "frame_index"]) for p in data_files],
        ignore_index=True,
    )
    total = len(frames_df)
    per_frame = np.empty(total, dtype=object)
    per_frame[:] = ""

    ep_mask = frames_df["episode_index"].values == int(episode_index)
    for a in assignments:
        s, e = int(a["start"]), int(a["end"])
        task = str(a["task"])
        local = ep_mask & (
            (frames_df["frame_index"].values >= s) &
            (frames_df["frame_index"].values <= e)
        )
        per_frame[local] = task
    covered = int((per_frame != "").sum())
    return {"covered": covered, "total": total, "per_frame": per_frame}


def write_language_to_dataset(
    dataset_root: Path,
    assignments: List[Dict],
    episode_index: int,
    dry_run: bool = False,
) -> Dict:
    """将语言标注写回 LeRobot 数据集（逐帧 language_persistent 列）。

    返回统计 dict。dry_run=True 时只预览不写入。
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset_root = Path(dataset_root).resolve()
    info_path = dataset_root / "meta" / "info.json"
    data_dir = dataset_root / "data"
    data_files = sorted(data_dir.rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"找不到数据文件: {data_dir}")

    frames_df = pd.concat(
        [pd.read_parquet(p, columns=["episode_index", "frame_index", "index"]) for p in data_files],
        ignore_index=True,
    )

    lang = np.empty(len(frames_df), dtype=object)
    lang[:] = ""
    ep_mask = frames_df["episode_index"].values == int(episode_index)
    for a in assignments:
        s, e = int(a["start"]), int(a["end"])
        task = str(a["task"])
        local = ep_mask & (
            (frames_df["frame_index"].values >= s) &
            (frames_df["frame_index"].values <= e)
        )
        lang[local] = task
    covered = int((lang != "").sum())

    if dry_run:
        return {"covered": covered, "total": len(frames_df), "files": len(data_files), "dry_run": True}

    # 写盘前备份全部 data 文件到 .bak_language_<时间戳>/（该操作会覆盖所有
    # data parquet；逐文件 tmp+os.replace 原子写，防止中途崩溃留下截断文件）
    backup_dir = dataset_root / f".bak_language_{datetime.now():%Y%m%d_%H%M%S}"
    for pf in data_files:
        rel = pf.relative_to(dataset_root)
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pf, dst)

    # 逐文件注入列（原子写：临时文件 + os.replace）
    lookup = pd.Series(lang, index=frames_df["index"].values)
    for pf in data_files:
        table = pq.read_table(pf)
        df = table.to_pandas()
        if "index" not in df.columns:
            continue
        col = lookup.reindex(df["index"].values).fillna("").astype(object).values
        df[LANGUAGE_FEATURE_NAME] = col
        out = pa.Table.from_pandas(df, preserve_index=False)
        tmp = pf.with_suffix(".tmp.parquet")
        pq.write_table(out, tmp)
        os.replace(tmp, pf)

    # 注册特征到 info.json
    with open(info_path) as f:
        info = json.load(f)
    features = info.setdefault("features", {})
    if LANGUAGE_FEATURE_NAME not in features:
        features[LANGUAGE_FEATURE_NAME] = {"dtype": "string", "shape": [1], "names": None}
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

    return {"covered": covered, "total": len(frames_df), "files": len(data_files), "dry_run": False}


# ==================== 命令行入口（预览） ====================

def _cli_preview(dataset_root: str, episode_index: int, names: Sequence[str]) -> None:
    """命令行预览：读取 episode 位姿并打印自动标注结果。"""
    # 复用 visualize_data_qt 的离线加载器会引入 PyQt；这里独立做个最小读取
    import pandas as pd
    root = Path(dataset_root).resolve()
    data_dir = root / "data"
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        print(f"找不到数据: {data_dir}")
        sys.exit(1)
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    ep = df[df["episode_index"] == episode_index]
    if len(ep) == 0:
        print(f"Episode {episode_index} 不存在")
        sys.exit(1)
    states = np.vstack(ep["observation.state"].to_numpy())
    data = {
        'x': states[:, 0], 'y': states[:, 1], 'z': states[:, 2],
        'rx': states[:, 3], 'ry': states[:, 4], 'rz': states[:, 5],
        'gripper': states[:, 6] if states.shape[1] >= 7 else np.zeros(len(states)),
    }
    assignments = annotate_episode(data, names)
    print(f"Episode {episode_index}: 共 {len(states)} 帧 → {len(assignments)} 个子任务")
    for a in assignments:
        print(f"  [{a['start']:>4} ~ {a['end']:<4}] ({a['end']-a['start']+1:>3} 帧) "
              f"{a['color']}  {a['task']}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="自动语言标注（子任务分段）")
    ap.add_argument("--root", required=True, help="数据集根目录")
    ap.add_argument("--episode", type=int, default=0, help="episode 索引")
    ap.add_argument("--names", default="抓取,移动,放置", help="预设子任务名（逗号分隔）")
    args = ap.parse_args()
    names = parse_subtask_names(args.names)
    _cli_preview(args.root, args.episode, names)


if __name__ == "__main__":
    main()

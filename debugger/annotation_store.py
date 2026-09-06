#!/usr/bin/env python3
"""
片段级奖励标注存储模块。

存储格式 (reward_annotations.json):
{
    "version": "1.0",
    "label_type": "fragment_interval",
    "labels": {
        "good":    {"color": "#2ecc71", "advantage": 1,  "desc": "好的操作/正确动作"},
        "bad":     {"color": "#e74c3c", "advantage": -1, "desc": "错误操作/失败动作"},
        "neutral": {"color": "#f39c12", "advantage": 0,  "desc": "中性/可接受但不理想"}
    },
    "episodes": {
        "0": [
            {"start": 10, "end": 50, "label": "good", "note": "抓取成功"},
            {"start": 60, "end": 90, "label": "bad",  "note": "掉落物体"}
        ],
        "1": [...]
    }
}

与 RECAP 的映射:
    good    -> advantage_indicator = 1   (I_t = True,  正优势)
    bad     -> advantage_indicator = -1  (I_t = False, 负优势)
    neutral -> advantage_indicator = 0
    未标注   -> advantage_indicator = 0   (留给价值函数推断)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


LABEL_COLORS = {
    "good":    "#2ecc71",
    "bad":     "#e74c3c",
    "neutral": "#f39c12",
}

LABEL_ADVANTAGE = {
    "good":    1,
    "bad":    -1,
    "neutral": 0,
}

LABEL_DESC = {
    "good":    "好的操作/正确动作",
    "bad":     "错误操作/失败动作",
    "neutral": "中性/可接受但不理想",
}

ALL_LABELS = ["good", "bad", "neutral"]


class AnnotationStore:
    """管理单个 LeRobot 数据集的奖励标注读写。"""

    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.meta_dir = self.dataset_root / "meta"
        self.filepath = self.meta_dir / "reward_annotations.json"
        self.data: Dict = self._default_structure()

    @staticmethod
    def _default_structure() -> Dict:
        return {
            "version": "1.0",
            "label_type": "fragment_interval",
            "labels": {
                name: {
                    "color": LABEL_COLORS[name],
                    "advantage": LABEL_ADVANTAGE[name],
                    "desc": LABEL_DESC[name],
                }
                for name in ALL_LABELS
            },
            "episodes": {},
        }

    def load(self) -> Dict:
        """从磁盘加载标注；不存在则返回空结构。"""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # 合并，确保 labels 定义始终完整
                base = self._default_structure()
                base["episodes"] = loaded.get("episodes", {})
                self.data = base
            except (json.JSONDecodeError, OSError) as e:
                print(f"[AnnotationStore] 加载失败，使用空标注: {e}")
                self.data = self._default_structure()
        else:
            self.data = self._default_structure()
        return self.data

    def save(self) -> None:
        """保存标注到磁盘。"""
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.filepath.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.filepath)

    # ---------- 区间操作 ----------

    def get_episodes(self) -> Dict[str, List[Dict]]:
        return self.data.setdefault("episodes", {})

    def get_intervals(self, episode_index: int) -> List[Dict]:
        key = str(episode_index)
        return self.get_episodes().setdefault(key, [])

    def add_interval(
        self,
        episode_index: int,
        start: int,
        end: int,
        label: str,
        note: str = "",
    ) -> Dict:
        """添加一个标注区间，自动按 start 排序。"""
        if label not in ALL_LABELS:
            raise ValueError(f"未知标签 '{label}'，可选: {ALL_LABELS}")
        if end < start:
            start, end = end, start
        interval = {"start": int(start), "end": int(end), "label": label, "note": note}
        intervals = self.get_intervals(episode_index)
        intervals.append(interval)
        intervals.sort(key=lambda x: x["start"])
        return interval

    def remove_interval(self, episode_index: int, index: int) -> None:
        intervals = self.get_intervals(episode_index)
        if 0 <= index < len(intervals):
            intervals.pop(index)

    def update_interval(self, episode_index: int, index: int, **fields) -> None:
        intervals = self.get_intervals(episode_index)
        if 0 <= index < len(intervals):
            intervals[index].update(fields)

    def clear_episode(self, episode_index: int) -> None:
        key = str(episode_index)
        if key in self.data["episodes"]:
            self.data["episodes"][key] = []

    # ---------- 统计 ----------

    def stats(self) -> Dict[str, int]:
        """返回各标签的帧数统计。"""
        counts = {name: 0 for name in ALL_LABELS}
        total_intervals = 0
        for ep_intervals in self.data["episodes"].values():
            for iv in ep_intervals:
                total_intervals += 1
                lbl = iv.get("label", "neutral")
                if lbl in counts:
                    counts[lbl] += (iv["end"] - iv["start"] + 1)
        counts["_intervals"] = total_intervals
        return counts

    def labeled_episode_count(self) -> int:
        return sum(
            1 for ivs in self.data["episodes"].values() if len(ivs) > 0
        )

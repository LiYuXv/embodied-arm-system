#!/usr/bin/env python3
"""
片段级奖励标注存储模块（schema v2.0）。

存储格式 (reward_annotations.json):
{
    "version": "2.0",
    "label_type": "fragment_interval",
    "labels": {
        "good":    {"color": "#2ecc71", "advantage": 1,  "desc": "好的操作/正确动作"},
        "bad":     {"color": "#e74c3c", "advantage": -1, "desc": "错误操作/失败动作"},
        "neutral": {"color": "#f39c12", "advantage": 0,  "desc": "中性/可接受但不理想"}
    },
    "episodes": {
        "0": {
            "episode_success": true,          // episode 级结论：true=成功 / false=失败 / null=未评
            "reviewed": true,                 // 是否已完整审核（区分"未标注"与"确认中性"）
            "source": "teleop",               // 数据来源溯源："teleop"(遥操作示范) /
                                              //   "rollout"(推理部署录制) / null(未知，老数据)
                                              //   合并后训练按来源配权重（防遗忘混训）用
            "intervals": [
                {"start": 10, "end": 50, "label": "good", "note": "抓取成功",
                 "is_human_intervention": false},   // 干预段（人工接管纠正）标志
                {"start": 60, "end": 90, "label": "bad",  "note": "掉落物体",
                 "is_human_intervention": false}
            ]
        }
    }
}

与 RECAP 的映射:
    good    -> advantage_indicator = 1   (I_t = True,  正优势)
    bad     -> advantage_indicator = -1  (I_t = False, 负优势)
    neutral -> advantage_indicator = 0
    未标注   -> advantage_indicator = 0   (留给价值函数推断)
    is_human_intervention=True -> 干预段恒为正优势（RECAP/APO 约定）

v1.0 兼容：旧格式 episodes 值为区间列表，load() 自动迁移为 v2.0 字典形态。

注意：标注以 episode_index 为键，剪辑/合并会重排索引使标注错位。
操作顺序纪律见 RECAP_MEM_实施路线.md P0.3；保存时自动留时间戳备份。
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


LABEL_COLORS = {
    "good":    "#2ecc71",
    "bad":     "#e74c3c",
    "neutral": "#f39c12",
}

LABEL_ADVANTAGE = {
    "good":    1,
    "bad":     -1,
    "neutral": 0,
}

LABEL_DESC = {
    "good":    "好的操作/正确动作",
    "bad":     "错误操作/失败动作",
    "neutral": "中性/可接受但不理想",
}

ALL_LABELS = ["good", "bad", "neutral"]

# 干预段（人工接管纠正）在轨道上的显示颜色（蓝色，覆盖标签色）
INTERVENTION_COLOR = "#3498db"

# 保存时保留的自动备份份数（超出后删最旧）
_MAX_BACKUPS = 10


class AnnotationStore:
    """管理单个 LeRobot 数据集的奖励标注读写（schema v2.0）。"""

    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.meta_dir = self.dataset_root / "meta"
        self.filepath = self.meta_dir / "reward_annotations.json"
        self.data: Dict = self._default_structure()

    @staticmethod
    def _default_structure() -> Dict:
        return {
            "version": "2.0",
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

    @staticmethod
    def _migrate_episode(value) -> Dict:
        """把单个 episode 的标注统一迁移到 v2.0 字典形态。

        v1.0: value 为区间列表 [{"start","end","label","note"}, ...]
        v2.0: value 为 {"episode_success": ..., "reviewed": ..., "intervals": [...]}
        """
        if isinstance(value, list):
            ep = {"episode_success": None, "reviewed": False, "source": None, "intervals": value}
        elif isinstance(value, dict):
            ep = value
        else:
            ep = {"episode_success": None, "reviewed": False, "source": None, "intervals": []}
        ep.setdefault("episode_success", None)
        ep.setdefault("reviewed", False)
        ep.setdefault("source", None)
        intervals = ep.setdefault("intervals", [])
        for iv in intervals:
            iv.setdefault("note", "")
            iv.setdefault("is_human_intervention", False)
        return ep

    def load(self) -> Dict:
        """从磁盘加载标注（v1.0 自动迁移 v2.0）；不存在则返回空结构。"""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                base = self._default_structure()
                base["episodes"] = {
                    str(k): self._migrate_episode(v)
                    for k, v in (loaded.get("episodes", {}) or {}).items()
                }
                self.data = base
            except (json.JSONDecodeError, OSError) as e:
                print(f"[AnnotationStore] 加载失败，使用空标注: {e}")
                self.data = self._default_structure()
        else:
            self.data = self._default_structure()
        return self.data

    def save(self) -> None:
        """保存标注到磁盘。覆盖前先留时间戳备份（最多保留最近 10 份）。"""
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        if self.filepath.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            bak = self.meta_dir / f"reward_annotations.{ts}.bak"
            try:
                shutil.copy2(self.filepath, bak)
                # 清理超出数量的旧备份（按文件名时间戳排序，删最旧）
                baks = sorted(self.meta_dir.glob("reward_annotations.*.bak"))
                for old in baks[: max(0, len(baks) - _MAX_BACKUPS)]:
                    try:
                        old.unlink()
                    except OSError:
                        pass
            except OSError as e:
                print(f"[AnnotationStore] 备份失败（仍继续保存）: {e}")
        tmp = self.filepath.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.filepath)

    # ---------- episode 级操作（v2.0 新增） ----------

    def _get_episode(self, episode_index: int) -> Dict:
        key = str(episode_index)
        episodes = self.data.setdefault("episodes", {})
        if key not in episodes:
            episodes[key] = self._migrate_episode([])
        else:
            episodes[key] = self._migrate_episode(episodes[key])
        return episodes[key]

    def get_episode_success(self, episode_index: int) -> Optional[bool]:
        """返回 True=成功 / False=失败 / None=未评。"""
        val = self._get_episode(episode_index).get("episode_success")
        return val if isinstance(val, bool) else None

    def set_episode_success(self, episode_index: int, value: Optional[bool]) -> None:
        """设置 episode 级结论；value=None 表示清除。"""
        if value is not None:
            value = bool(value)
        self._get_episode(episode_index)["episode_success"] = value

    def get_reviewed(self, episode_index: int) -> bool:
        return bool(self._get_episode(episode_index).get("reviewed", False))

    def set_reviewed(self, episode_index: int, reviewed: bool) -> None:
        self._get_episode(episode_index)["reviewed"] = bool(reviewed)

    def get_episode_source(self, episode_index: int) -> Optional[str]:
        """数据来源："teleop"(遥操作示范) / "rollout"(推理部署录制) / None(未知)。"""
        val = self._get_episode(episode_index).get("source")
        return str(val) if val else None

    def set_episode_source(self, episode_index: int, source: Optional[str]) -> None:
        """设置数据来源溯源字段；None 表示未知（老数据）。"""
        self._get_episode(episode_index)["source"] = source

    def set_episode_policy(self, episode_index: int, policy_checkpoint: Optional[str]) -> None:
        """设置产生本集数据的策略 checkpoint 标识（π*0.6 在线 RL 迭代前提）。

        None/空 = 专家示范（遥操作）或未知（老数据）。字段为纯元数据，
        永不进入模型输入（见 PI06_RL_迁移实施方案.md §4.1 三条文本红线）。
        """
        self._get_episode(episode_index)["policy_checkpoint"] = policy_checkpoint or None

    def get_episode_policy(self, episode_index: int) -> Optional[str]:
        """返回产生本集数据的策略标识；None = 专家/未知。"""
        val = self._get_episode(episode_index).get("policy_checkpoint")
        return str(val) if val else None

    def set_episode_group(self, episode_index: int, group_id: Optional[str]) -> None:
        """设置任务组标识（同任务同场景同策略的 N 轮成组，组相对优势用）。

        None = 未分组（普通数据）。字段为纯元数据。
        """
        self._get_episode(episode_index)["group_id"] = group_id or None

    def get_episode_group(self, episode_index: int) -> Optional[str]:
        """返回本集所属任务组；None = 未分组。"""
        val = self._get_episode(episode_index).get("group_id")
        return str(val) if val else None

    def derive_episode_success(self, episode_index: int) -> Optional[bool]:
        """从片段标注派生 episode 成败（标注粒度决策：不做人工整段标注）。

        规则（RECAP_MEM_实施路线.md P0.1）：
        - 无任何标注区间 → None（未评）；
        - 正优势帧（good 或干预段）占标注帧比例 > 50% 且不存在"未接恢复段的
          bad 段" → True（成功）；
        - 否则 → False（失败）。
        "未接恢复段"：bad 区间之后（按 start 比较）没有任何 good/干预区间，
        即失败后没有纠正/恢复行为。干预段（人工接管纠正）按 RECAP/APO 约定
        恒为正优势，既计入占比分子、也计入恢复依据。

        该函数供标注面板只读显示与 make_recap_sidecar.py 展开时使用；
        派生结果在展开时写入 sidecar（供 P3 critic 训练），非人工标注字段。
        """
        intervals = self.get_intervals(episode_index)
        if not intervals:
            return None
        good_frames = 0
        labeled = 0
        good_starts, bad_starts = [], []
        for iv in intervals:
            n = iv["end"] - iv["start"] + 1
            lbl = iv.get("label", "neutral")
            labeled += n
            # 干预段恒为正优势（无论所标标签）
            if lbl == "good" or iv.get("is_human_intervention"):
                good_frames += n
                good_starts.append(iv["start"])
            elif lbl == "bad":
                bad_starts.append(iv["start"])
        if labeled <= 0:
            return None
        unrecovered_bad = any(
            not any(g > b for g in good_starts) for b in bad_starts)
        return (good_frames / labeled > 0.5) and not unrecovered_bad

    def remap_after_delete(self, deleted_indices) -> None:
        """删除 episode 后重映射标注键：被删 episode 的标注一并移除，
        其后 episode 的标注键前移（ep>E → ep-删除数），保证标注继续对齐。
        """
        deleted = sorted({int(i) for i in deleted_indices})
        if not deleted:
            return
        new_eps = {}
        for k, v in self.data.get("episodes", {}).items():
            idx = int(k)
            if idx in deleted:
                continue
            shift = sum(1 for d in deleted if d < idx)
            new_eps[str(idx - shift)] = self._migrate_episode(v)
        self.data["episodes"] = new_eps

    # ---------- 区间操作 ----------

    @staticmethod
    def _check_overlap(intervals, start, end, exclude_index=None) -> None:
        """闭区间重叠校验：有公共帧即报错。exclude_index 用于编辑时排除自身。"""
        for i, iv in enumerate(intervals):
            if exclude_index is not None and i == exclude_index:
                continue
            if start <= iv["end"] and iv["start"] <= end:
                raise ValueError(
                    f"区间 [{start},{end}] 与已有区间 "
                    f"[{iv['start']},{iv['end']}] ({iv['label']}) 帧重叠；"
                    f"请先删除/调整已有区间再操作"
                )

    def get_episodes(self) -> Dict[str, List[Dict]]:
        return self.data.setdefault("episodes", {})

    def get_intervals(self, episode_index: int) -> List[Dict]:
        return self._get_episode(episode_index)["intervals"]

    def add_interval(
        self,
        episode_index: int,
        start: int,
        end: int,
        label: str,
        note: str = "",
        is_human_intervention: bool = False,
    ) -> Dict:
        """添加一个标注区间，自动按 start 排序。

        Raises:
            ValueError: 标签未知，或与已有区间存在帧重叠（矛盾标签会破坏
                展开结果，必须人工拆分/合并后再添加）。
        """
        if label not in ALL_LABELS:
            raise ValueError(f"未知标签 '{label}'，可选: {ALL_LABELS}")
        if end < start:
            start, end = end, start
        intervals = self.get_intervals(episode_index)
        self._check_overlap(intervals, start, end)
        interval = {
            "start": int(start), "end": int(end), "label": label,
            "note": note, "is_human_intervention": bool(is_human_intervention),
        }
        intervals.append(interval)
        intervals.sort(key=lambda x: x["start"])
        return interval

    def edit_interval(
        self,
        episode_index: int,
        index: int,
        start: int,
        end: int,
        label: str,
        note: str = "",
        is_human_intervention: bool = False,
    ) -> int:
        """编辑已有区间（GUI 双击修改）：校验标签/重叠（排除自身）后更新并重排序。

        Returns:
            更新后区间在列表中的新下标（重排序后可能变化），供 GUI 重新选中。

        Raises:
            ValueError: 标签未知，或与其他区间帧重叠。
            IndexError: 区间下标越界。
        """
        if label not in ALL_LABELS:
            raise ValueError(f"未知标签 '{label}'，可选: {ALL_LABELS}")
        intervals = self.get_intervals(episode_index)
        if not (0 <= index < len(intervals)):
            raise IndexError(f"区间下标越界: {index}（共 {len(intervals)} 个）")
        if end < start:
            start, end = end, start
        self._check_overlap(intervals, start, end, exclude_index=index)
        intervals[index].update({
            "start": int(start), "end": int(end), "label": label,
            "note": note, "is_human_intervention": bool(is_human_intervention),
        })
        iv = intervals.pop(index)
        intervals.append(iv)
        intervals.sort(key=lambda x: x["start"])
        return intervals.index(iv)

    def remove_interval(self, episode_index: int, index: int) -> None:
        intervals = self.get_intervals(episode_index)
        if 0 <= index < len(intervals):
            intervals.pop(index)

    def update_interval(self, episode_index: int, index: int, **fields) -> None:
        intervals = self.get_intervals(episode_index)
        if 0 <= index < len(intervals):
            intervals[index].update(fields)

    def clear_episode(self, episode_index: int) -> None:
        """清空该 episode 的全部区间（保留 episode 级结论与已审核标志）。"""
        self._get_episode(episode_index)["intervals"] = []

    # ---------- 统计 ----------

    def stats(self) -> Dict[str, int]:
        """返回各标签的帧数统计（另含干预段帧数与 episode 级统计）。"""
        counts = {name: 0 for name in ALL_LABELS}
        counts["intervention"] = 0
        total_intervals = 0
        n_success = n_fail = 0
        for ep in self.data["episodes"].values():
            ep = self._migrate_episode(ep)
            if ep.get("episode_success") is True:
                n_success += 1
            elif ep.get("episode_success") is False:
                n_fail += 1
            for iv in ep["intervals"]:
                total_intervals += 1
                lbl = iv.get("label", "neutral")
                if lbl in counts:
                    counts[lbl] += (iv["end"] - iv["start"] + 1)
                if iv.get("is_human_intervention"):
                    counts["intervention"] += (iv["end"] - iv["start"] + 1)
        counts["_intervals"] = total_intervals
        counts["_success_eps"] = n_success
        counts["_fail_eps"] = n_fail
        return counts

    def labeled_episode_count(self) -> int:
        return sum(
            1 for ep in self.data["episodes"].values()
            if len(self._migrate_episode(ep)["intervals"]) > 0
        )

    def reviewed_episode_count(self) -> int:
        return sum(
            1 for ep in self.data["episodes"].values()
            if self._migrate_episode(ep).get("reviewed")
        )


# ---------- 模块级工具：标注存在性检查与手动备份（P0.3 剪辑/合并防护用） ----------

def remap_intervals_after_trim(intervals, removed_ranges):
    """剪辑后把标注区间按保留帧重映射（P0.3 方案乙，剪辑不丢标注）。

    Args:
        intervals: [{"start","end","label","note","is_human_intervention"}, ...]
        removed_ranges: 被删除的闭区间 [(s,e),...]（episode 内帧索引，可乱序/相邻）

    Returns:
        (new_intervals, stats)。stats = {"kept": 原样前移的区间数,
        "split": 被拆分为多段的区间数, "truncated": 端点被截的区间数,
        "dropped": 整段落在删除区内被丢弃的区间数}

    规则：区间映射为其与保留帧的交集——交集为空则丢弃；跨删除段的区间
    截断/拆分为多个保留子段（label/note/干预标志原样继承），帧号前移
    （new = old − 该帧之前被删除的帧数）。
    """
    # 归并删除区间（排序 + 相邻合并）
    rs = sorted((int(s), int(e)) for s, e in removed_ranges if int(e) >= int(s))
    merged = []
    for s, e in rs:
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    def removed_before(f):
        """帧号 f 之前（不含 f）被删除的帧数（f 为保留帧时即其前移量）。"""
        total = 0
        for s, e in merged:
            if e < f:
                total += e - s + 1
            elif s < f:  # f 落在删除区内（调用方只对保留帧使用，防御性分支）
                total += f - s
            else:
                break
        return total

    def kept_subranges(s, e):
        """[s,e] 与保留帧的交集子段列表（按序）。"""
        out, cur = [], s
        for rs_, re_ in merged:
            if re_ < cur:
                continue
            if rs_ > e:
                break
            if rs_ > cur:
                out.append((cur, min(rs_ - 1, e)))
            cur = max(cur, re_ + 1)
            if cur > e:
                break
        if cur <= e:
            out.append((cur, e))
        return out

    stats = {"kept": 0, "split": 0, "truncated": 0, "dropped": 0}
    out = []
    for iv in intervals:
        s, e = int(iv["start"]), int(iv["end"])
        subs = kept_subranges(s, e)
        if not subs:
            stats["dropped"] += 1
            continue
        for ss, se in subs:
            niv = dict(iv)
            niv["start"] = ss - removed_before(ss)
            niv["end"] = se - removed_before(se)
            out.append(niv)
        if len(subs) == 1 and subs[0] == (s, e):
            stats["kept"] += 1
        elif len(subs) > 1:
            stats["split"] += 1
        else:
            stats["truncated"] += 1
    out.sort(key=lambda x: x["start"])
    return out, stats

def has_annotations(dataset_root) -> bool:
    """数据集是否存在任何 RECAP 标注（区间或 episode 级结论）。

    文件存在但损坏（JSON 解析失败）时按"有标注"处理，宁拦勿放——
    防止在标注状态未知的情况下执行剪辑/合并。
    """
    fp = Path(dataset_root) / "meta" / "reward_annotations.json"
    if not fp.is_file():
        return False
    try:
        with open(fp, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return True
    try:
        for v in (raw.get("episodes", {}) or {}).values():
            ep = AnnotationStore._migrate_episode(v)
            if ep["intervals"] or ep.get("episode_success") is not None:
                return True
    except Exception:
        return True
    return False


def backup_annotations(dataset_root, remove_original: bool = False) -> Optional[Path]:
    """把 reward_annotations.json 备份为带时间戳的 .bak，返回备份路径。

    用于剪辑/合并等会使标注（部分）失效的操作之前：
    - 剪辑（remove_original=False）：仅复制备份；调用方随后按需清除失效 episode
      的区间（episode 级结论仍有效，保留）；
    - 合并目标数据集（remove_original=True）：episode_index 全部重排，
      标注整体失效，备份后移除原文件，防止后续被误展开进训练数据。
    文件不存在时返回 None。
    """
    fp = Path(dataset_root) / "meta" / "reward_annotations.json"
    if not fp.is_file():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = fp.parent / f"reward_annotations.stale.{ts}.bak"
    shutil.copy2(fp, bak)
    if remove_original:
        fp.unlink()
    return bak

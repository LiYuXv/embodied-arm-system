#!/usr/bin/env python3
"""
make_recap_sidecar.py — 把 reward_annotations.json 片段标注展开为逐帧 sidecar JSONL。

路线文档：RECAP_MEM_实施路线.md P1.1。**不修改 parquet、不碰 info.json**——
sidecar 是外挂文件，训练侧用 SidecarDataset 包装合并（P1.2）。

两个层次的区别（常见问题）：
- "保存标注"按钮保存的是**区间级标注稿**（reward_annotations.json，几十个区间，
  供人编辑/回放显示）；
- 本脚本把它**展开编译为逐帧标签**（recap_sidecar.jsonl，每帧一行），
  训练 dataloader 逐帧取样本时才能消费。
GUI 中可直接用 RECAP 面板的"生成训练 sidecar"按钮（内部调用 expand_dataset）。

输出（默认 <dataset>/meta/recap_sidecar.jsonl），逐帧一行：
{
  "episode_index": 0, "frame_index": 10,
  "advantage_indicator": 1,        # int 符号路由：1=正优势 / -1=负优势 / 0=未标注或中性
  "advantage_score": 1.0,          # float 强度 [-1,1]（回传衰减/终端规则时为分数幅值，
                                   # 平坦模式下等于 indicator；供训练侧调制 loss 权重）
  "is_human_intervention": 0,      # 干预段=1（人工接管纠正）
  "episode_success": 1,            # 1=成功 / 0=失败 / -1=未评；人工直标（部署时
                                   # 保存标记）优先，无直标时由片段标注派生兜底，逐帧平铺
  "source": "rollout",             # 数据来源溯源："teleop"(遥操作示范) /
                                   # "rollout"(推理部署录制) / null(老数据)；
                                   # 训练按来源配权重（防遗忘混训）用
  "rl_token_weight": 1.0           # 训练侧覆盖点（CBC schema 兼容），此处恒 1.0
}

借鉴 π\*0.6（RECAP, arXiv:2511.14759）的打分规则：
1. **干预段恒 I_t=True**（论文强制规则）：is_human_intervention 区间无论所标标签，
   advantage 恒为正；
2. **负优势回传（bad_backprop>0，默认关）**：n-step 优势前瞻（论文 N=50）会把负优势
   回传到失败前的因果帧。无 critic 阶段的启发式近似：bad 段向前回传 N 帧，
   线性衰减——离失败点越近负得越强（frame(s-1)≈-1，frame(s-N)→0）。
   只填充未被人标覆盖的帧（人工标注优先）；
3. **终端失败惩罚（terminal_fail>0，默认关）**：论文稀疏奖励 r_T=-C_fail。
   派生为失败的 episode 末 K 帧自动标 bad（建议 K=10，对齐 APO "接管前最后
   10 个动作"规则）。只填充未被人标覆盖的帧；
4. 二值化阈值 ε（30%/40% 正例）属 critic 阶段（P3）规则，人工标注展开不适用，
   本脚本不实现；I_t 的 30% dropout 在训练侧（P1.2/P4）实现，不在此处理。

pending=true 的预标区间（部署侧 APO 自动预标，待人工确认）**跳过不展开**。

用法：
  # 预览（不写文件，打印标签分布）
  python3 make_recap_sidecar.py --dataset ../lerobot_data/ela3_demo --dry-run
  # 平坦展开（路线文档 P1.1 默认）
  python3 make_recap_sidecar.py --dataset ../lerobot_data/ela3_demo
  # RECAP 打分规则：bad 回传 50 帧 + 失败 episode 末 10 帧自动标 bad
  python3 make_recap_sidecar.py --dataset ../lerobot_data/ela3_demo \
      --bad-backprop 50 --terminal-fail 10

注意：标注以 episode_index 为键，剪辑/合并/删除 episode 后索引重排，
旧 sidecar 作废，必须重跑（P0.4 数据质量门禁第 3 条）。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# 同目录导入（schema v2.0，含 derive_episode_success）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotation_store import AnnotationStore  # noqa: E402

_LABEL_TO_ADV = {"good": 1, "bad": -1, "neutral": 0}


def _load_frame_table(dataset_root: Path) -> pd.DataFrame:
    """读 data/**/*.parquet 的 (episode_index, frame_index, index) 三列。

    只读三个整型列，ela3_demo 量级（~8.6 万帧 / 151 文件）秒级完成。
    """
    data_dir = dataset_root / "data"
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"找不到数据 parquet: {data_dir}")
    cols = ["episode_index", "frame_index", "index"]
    dfs = []
    for f in files:
        try:
            dfs.append(pq.read_table(f, columns=cols).to_pandas())
        except Exception as e:
            print(f"[sidecar] 跳过损坏的 parquet {f.name}: {e}")
    if not dfs:
        raise FileNotFoundError(f"全部 parquet 读取失败: {data_dir}")
    return pd.concat(dfs, ignore_index=True)


def expand_episode(store: AnnotationStore, ep_idx: int, ep_len: int,
                   bad_backprop: int, terminal_fail: int,
                   stats: dict) -> tuple:
    """展开单个 episode，返回 (indicator, score, intervention, ep_success)。

    规则优先级：人工标注（含 neutral）> bad 回传 / 终端失败（只填未覆盖帧）。
    """
    indicator = np.zeros(ep_len, dtype=np.int8)
    score = np.zeros(ep_len, dtype=np.float64)
    intervention = np.zeros(ep_len, dtype=np.int8)
    covered = np.zeros(ep_len, dtype=bool)  # 人工标注覆盖（含 neutral）

    intervals = store.get_intervals(ep_idx)
    for iv in intervals:
        if iv.get("pending"):
            stats["pending_skipped"] += 1
            continue
        s = max(0, int(iv["start"]))
        e = min(ep_len - 1, int(iv["end"]))
        if e < s:
            stats["clipped_intervals"] += 1
            continue
        if s != int(iv["start"]) or e != int(iv["end"]):
            stats["clipped_intervals"] += 1  # 区间越界（剪辑后旧标注），截断
        covered[s:e + 1] = True
        lbl = iv.get("label", "neutral")
        val = _LABEL_TO_ADV.get(lbl, 0)
        if iv.get("is_human_intervention"):
            intervention[s:e + 1] = 1
            val = 1  # RECAP 规则 1：干预段恒正优势
            stats["intervention_frames"] += e - s + 1
        indicator[s:e + 1] = val
        score[s:e + 1] = float(val)
        stats[f"human_{lbl}_frames"] += e - s + 1

        # RECAP 规则 2：负优势向失败点回传（线性衰减，只填未覆盖帧）
        if lbl == "bad" and bad_backprop > 0:
            for d in range(1, bad_backprop + 1):
                f = s - d
                if f < 0:
                    break
                mag = -(1.0 - d / bad_backprop)  # d=1 最强 ≈-1；d→N 趋 0
                if mag >= 0:
                    break
                if covered[f]:
                    continue
                if mag < score[f]:  # 多 bad 段回传重叠时取最强负
                    score[f] = mag
                    indicator[f] = -1
                    stats["backprop_frames"] += 1

    # RECAP 规则 3：终端失败惩罚（r_T=-C_fail 的近似，只填未覆盖帧）
    # episode 成败优先级：人工直标（部署侧保存时一键标记）> 片段派生（兜底）
    ep_success = store.get_episode_success(ep_idx)
    if ep_success is None:
        ep_success = store.derive_episode_success(ep_idx)
    if terminal_fail > 0 and ep_success is False and ep_len > 0:
        for f in range(max(0, ep_len - terminal_fail), ep_len):
            if covered[f]:
                continue
            indicator[f] = -1
            score[f] = -1.0
            stats["terminal_fail_frames"] += 1

    return indicator, score, intervention, ep_success


def expand_dataset(dataset_root, bad_backprop: int = 0, terminal_fail: int = 0,
                   output=None, write: bool = True) -> dict:
    """展开单个数据集为逐帧 sidecar JSONL，返回结果 dict（含人类可读摘要行）。

    不写 parquet、不改 info.json；sidecar 原子写。CLI（main）与 GUI
    （RECAP 面板"生成训练 sidecar"按钮）共用本函数。

    Args:
        dataset_root: 数据集根目录（含 meta/ 与 data/）
        bad_backprop: bad 段负优势回传帧数（0=关闭；RECAP n-step 近似，建议 50）
        terminal_fail: 派生失败 episode 末 K 帧自动标 bad（0=关闭；APO 规则，建议 10）
        output: 输出路径（默认 <dataset>/meta/recap_sidecar.jsonl）
        write: False 时只计算统计不写文件（dry-run）

    Returns:
        dict: out_path / total / dist / coverage_pct / n_success / n_fail /
              n_unknown / stats / summary_lines（人类可读摘要，逐行）
    """
    dataset_root = Path(dataset_root).resolve()
    store = AnnotationStore(str(dataset_root))
    store.load()
    has_ann = (dataset_root / "meta" / "reward_annotations.json").is_file()

    frames = _load_frame_table(dataset_root)
    out_path = Path(output) if output else (dataset_root / "meta" / "recap_sidecar.jsonl")

    stats = {"pending_skipped": 0, "clipped_intervals": 0,
             "intervention_frames": 0, "backprop_frames": 0,
             "terminal_fail_frames": 0,
             "human_good_frames": 0, "human_bad_frames": 0, "human_neutral_frames": 0}

    records = []
    n_success = n_fail = n_unknown = 0
    for ep_idx, g in frames.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index")
        ep_idx = int(ep_idx)
        indicator, score, intervention, ep_success = expand_episode(
            store, ep_idx, len(g), bad_backprop, terminal_fail, stats)
        ep_source = store.get_episode_source(ep_idx)  # 来源溯源（teleop/rollout/None）
        # π*0.6 元数据透传（PI06_RL_迁移实施方案.md §4.1）：策略版本/任务组随帧
        # 平铺；group_advantage 由 recap_group_advantage.py 训练前写入（缺省 0）
        ep_policy = store.get_episode_policy(ep_idx)
        ep_group = store.get_episode_group(ep_idx)
        if ep_success is True:
            succ_val, n_success = 1, n_success + 1
        elif ep_success is False:
            succ_val, n_fail = 0, n_fail + 1
        else:
            succ_val, n_unknown = -1, n_unknown + 1
        for fi, ind, sc, itv in zip(
                g["frame_index"].to_numpy(), indicator, score, intervention):
            records.append({
                "episode_index": ep_idx,
                "frame_index": int(fi),
                "advantage_indicator": int(ind),
                "advantage_score": round(float(sc), 4),
                "is_human_intervention": int(itv),
                "episode_success": succ_val,
                "source": ep_source,
                "policy_checkpoint": ep_policy,
                "group_id": ep_group,
                "group_advantage": 0.0,
                "rl_token_weight": 1.0,
            })

    # ---------- 统计 ----------
    inds = np.array([r["advantage_indicator"] for r in records], dtype=np.int8) \
        if records else np.zeros(0, dtype=np.int8)
    total = len(records)
    dist = {int(k): int(v) for k, v in zip(*np.unique(inds, return_counts=True))} \
        if total else {}
    coverage_pct = 100.0 * int((inds != 0).sum()) / total if total else 0.0

    summary = [
        f"数据集: {dataset_root.name}  总帧数: {total}  "
        f"episode: 成功 {n_success} / 失败 {n_fail} / 未评 {n_unknown}",
        f"advantage_indicator 分布: {dist}  (标注覆盖率 {coverage_pct:.1f}%)",
        f"人工标注帧: good={stats['human_good_frames']} "
        f"bad={stats['human_bad_frames']} neutral={stats['human_neutral_frames']} "
        f"干预={stats['intervention_frames']}",
    ]
    if bad_backprop > 0:
        summary.append(f"RECAP 回传帧: {stats['backprop_frames']} (bad-backprop={bad_backprop})")
    if terminal_fail > 0:
        summary.append(f"终端失败标记帧: {stats['terminal_fail_frames']} (terminal-fail={terminal_fail})")
    if stats["pending_skipped"]:
        summary.append(f"跳过待确认预标区间: {stats['pending_skipped']} 个")
    if stats["clipped_intervals"]:
        summary.append(f"警告: {stats['clipped_intervals']} 个区间越界被截断/丢弃"
                       f"（疑似剪辑后的旧标注，建议复核）")
    if not has_ann:
        summary.insert(0, "警告: meta/reward_annotations.json 不存在，展开为全 0"
                          "（请先在 visualize_data 中标注）")

    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(out_path)  # 原子替换
        summary.append(f"已写出 {total} 行 -> {out_path}")
    else:
        summary.append("dry-run，未写文件")

    return {
        "out_path": out_path, "total": total, "dist": dist,
        "coverage_pct": coverage_pct, "stats": stats,
        "n_success": n_success, "n_fail": n_fail, "n_unknown": n_unknown,
        "summary_lines": summary,
    }


def summarize_from_frames(store, frames, dataset_name=None):
    """从内存 store + 帧 DataFrame 快速生成 RECAP 标注概览。

    与 `make_recap_sidecar.py --dry-run`（平坦模式）的统计输出同格式，
    供可视化"数据集总信息"面板调用：直接复用已加载的 data_df 与
    reward_annotations，不重读 parquet、不构建逐帧记录，毫秒级完成。

    Returns:
        摘要行列表（3 行核心统计 + 可选的截断警告行）；无帧时返回空列表。
    """
    dataset_name = dataset_name or getattr(store.dataset_root, "name", "") if store else ""
    stats = {"pending_skipped": 0, "clipped_intervals": 0,
             "intervention_frames": 0, "backprop_frames": 0,
             "terminal_fail_frames": 0,
             "human_good_frames": 0, "human_bad_frames": 0, "human_neutral_frames": 0}
    n_success = n_fail = n_unknown = 0
    ind_parts = []
    total = 0
    for ep_idx, g in frames.groupby("episode_index", sort=True):
        ep_idx = int(ep_idx)
        indicator, _score, _itv, ep_success = expand_episode(
            store, ep_idx, len(g), 0, 0, stats)
        ind_parts.append(indicator)
        total += len(g)
        if ep_success is True:
            n_success += 1
        elif ep_success is False:
            n_fail += 1
        else:
            n_unknown += 1
    if total <= 0:
        return []
    inds = np.concatenate(ind_parts) if ind_parts else np.zeros(0, dtype=np.int8)
    dist = {int(k): int(v) for k, v in zip(*np.unique(inds, return_counts=True))}
    coverage = 100.0 * int((inds != 0).sum()) / total
    lines = [
        f"{dataset_name}: 总帧数 {total}  episode: 成功 {n_success} / 失败 {n_fail} / 未评 {n_unknown}",
        f"advantage_indicator 分布: {dist}  (标注覆盖率 {coverage:.1f}%)",
        f"人工标注帧: good={stats['human_good_frames']} "
        f"bad={stats['human_bad_frames']} neutral={stats['human_neutral_frames']} "
        f"干预={stats['intervention_frames']}",
    ]
    if stats["clipped_intervals"]:
        lines.append(f"警告: {stats['clipped_intervals']} 个区间越界被截断/丢弃（建议复核）")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path,
                    help="数据集根目录（含 meta/ 与 data/），如 ../lerobot_data/ela3_demo")
    ap.add_argument("--output", type=Path, default=None,
                    help="输出路径（默认 <dataset>/meta/recap_sidecar.jsonl）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印标签分布，不写文件")
    ap.add_argument("--bad-backprop", type=int, default=0,
                    help="bad 段负优势向前回传帧数（0=关闭；RECAP n-step 近似，建议 50）")
    ap.add_argument("--terminal-fail", type=int, default=0,
                    help="派生失败 episode 末 K 帧自动标 bad（0=关闭；APO 规则，建议 10）")
    args = ap.parse_args()

    result = expand_dataset(args.dataset, bad_backprop=args.bad_backprop,
                            terminal_fail=args.terminal_fail,
                            output=args.output, write=not args.dry_run)
    for line in result["summary_lines"]:
        print(f"[sidecar] {line}")
    if not args.dry_run:
        print("[sidecar] 原 parquet 未做任何修改；训练侧经 SidecarDataset 合并使用（P1.2）")


if __name__ == "__main__":
    main()

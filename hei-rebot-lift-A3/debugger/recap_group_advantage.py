#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recap_group_advantage.py — 组相对优势计算（π*0.6 R1b，PI06_RL_迁移实施方案.md §4.4 Step-1）。

把 episode 级回报在同任务组内做 GRPO 式标准化，得到每集优势 A_i，写回
sidecar 的逐帧列 `group_advantage`（同集所有帧取该集 A）。

回报定义（系数可配）：
    R_i = success(1/0) − takeover_penalty × 接管次数      截断 [-1, 1]
    success 取 episode_success（人工直标优先）；未评时由片段派生兜底
    接管次数 = 干预段（is_human_intervention）计数

优势定义：
    A_i = (R_i − mean(R_group)) / (std(R_group) + ε)     仅 n≥min_group 的组
    未分组 / 小组 / std=0（全同结局）→ A=0（不参与组调制，走现有 P1.2 规则）

用法：
    python3 recap_group_advantage.py --dataset ../lerobot_data/ela3_demo5
    python3 recap_group_advantage.py --dataset ... --takeover-penalty 0.25 --min-group 4 --dry-run

顺序纪律：本脚本必须在"生成训练 sidecar"之后运行（剪辑/标注/分组变更后
两步都要重跑）；本脚本只改 sidecar 的 group_advantage 列，原子写。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "visualize_data"))
from annotation_store import AnnotationStore  # noqa: E402


def load_sidecar_rows(dataset_root: str | Path) -> tuple[list[dict], Path]:
    path = Path(dataset_root) / "meta" / "recap_sidecar.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 {path}——请先在 visualize_data 生成训练 sidecar（P1.1）")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows, path


def compute_group_advantages(
    store: AnnotationStore,
    takeover_penalty: float = 0.25,
    min_group: int = 4,
    eps: float = 1e-6,
) -> tuple[dict[int, float], dict[str, dict]]:
    """计算每集组优势。Returns: ({episode_index: A}, 每组摘要供展示)。"""
    # 1) episode → group 与回报
    ep_return: dict[int, float] = {}
    ep_group: dict[int, str] = {}
    for k, ep in store.data.get("episodes", {}).items():
        try:
            ek = int(k)
        except ValueError:
            continue
        gid = ep.get("group_id")
        if not gid:
            continue  # 未分组不参与
        succ = ep.get("episode_success")
        if succ is None:
            succ = store.derive_episode_success(ek)  # 片段派生兜底
        if succ is None:
            continue  # 连派生都没有 → 无回报信号，跳过
        takeovers = sum(1 for iv in ep.get("intervals", [])
                        if iv.get("is_human_intervention"))
        r = (1.0 if succ else 0.0) - takeover_penalty * takeovers
        ep_return[ek] = float(np.clip(r, -1.0, 1.0))
        ep_group[ek] = str(gid)

    # 2) 分桶 → 组内标准化
    buckets: dict[str, list[int]] = defaultdict(list)
    for ek, gid in ep_group.items():
        buckets[gid].append(ek)
    advantages: dict[int, float] = {}
    summaries: dict[str, dict] = {}
    for gid, eps_list in buckets.items():
        rs = np.array([ep_return[e] for e in eps_list], dtype=np.float64)
        n = len(eps_list)
        mean, std = float(rs.mean()), float(rs.std())
        if n < min_group:
            for e in eps_list:
                advantages[e] = 0.0
            summaries[gid] = {"n": n, "mean": mean, "std": std,
                              "status": f"n={n}<{min_group} → A=0"}
            continue
        if std < eps:
            for e in eps_list:
                advantages[e] = 0.0
            summaries[gid] = {"n": n, "mean": mean, "std": std,
                              "status": "全同结局 → A=0"}
            continue
        for e, r in zip(eps_list, rs):
            advantages[e] = float((r - mean) / (std + eps))
        summaries[gid] = {"n": n, "mean": round(mean, 4), "std": round(std, 4),
                          "status": "ok"}
    return advantages, summaries


def write_group_advantages(rows: list[dict], advantages: dict[int, float],
                           path: Path) -> int:
    """把 A 写回 sidecar 的 group_advantage 列（原子替换），返回更新帧数。"""
    tmp = path.with_suffix(".jsonl.tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            a = advantages.get(int(r.get("episode_index", -1)), 0.0)
            r = dict(r)
            r["group_advantage"] = round(float(a), 4)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def main():
    ap = argparse.ArgumentParser(description="π*0.6 R1b 组相对优势计算（训练前运行）")
    ap.add_argument("--dataset", required=True, help="数据集根目录")
    ap.add_argument("--takeover-penalty", type=float, default=0.25,
                    help="每次人工接管的回报惩罚（默认 0.25）")
    ap.add_argument("--min-group", type=int, default=4,
                    help="成组最少 episode 数（默认 4，小于则 A=0）")
    ap.add_argument("--dry-run", action="store_true", help="只打印不写文件")
    args = ap.parse_args()

    root = Path(args.dataset)
    rows, sc_path = load_sidecar_rows(root)
    store = AnnotationStore(str(root))
    store.load()
    adv, summaries = compute_group_advantages(
        store, takeover_penalty=args.takeover_penalty, min_group=args.min_group)

    print(f"数据集: {root.name}  sidecar {len(rows)} 行")
    if not summaries:
        print("无分组数据（group_id 全空）——全部帧 A=0，走现有 P1.2 规则。")
    for gid, s in sorted(summaries.items(), key=lambda kv: -kv[1]["n"]):
        label = gid.split(" | ")[0]
        members = sorted(e for e, g in
                         {k: str(v.get("group_id") or "")
                          for k, v in store.data.get("episodes", {}).items()}.items()
                         if g == gid)
        a_str = ", ".join(f"ep{e}:A={adv.get(int(e), 0.0):+.3f}" for e in members[:8])
        if len(members) > 8:
            a_str += f" … 共{len(members)}集"
        print(f"  组[{label}] n={s['n']} R̄={s['mean']} σ={s['std']} ({s['status']})")
        if s["status"] == "ok":
            print(f"    {a_str}")

    n_pos = sum(1 for a in adv.values() if a > 0)
    n_neg = sum(1 for a in adv.values() if a < 0)
    print(f"正优势 {n_pos} 集 / 负优势 {n_neg} 集 / 其余 A=0")

    if args.dry_run:
        print("dry-run，未写文件")
        return
    n = write_group_advantages(rows, adv, sc_path)
    print(f"已写出 group_advantage 列（{n} 行，原子替换）-> {sc_path}")


if __name__ == "__main__":
    main()

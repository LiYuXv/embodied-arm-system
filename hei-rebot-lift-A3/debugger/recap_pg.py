#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recap_pg.py — π*0.6 R3：优势加权的策略改进（pi0.5 flow-matching 上的 offline RL）。

背景（PI06_RL_迁移实施方案.md §5 R3 / §8 诚实声明）：
  pi0.5 是 flow-matching 策略，无可解析动作对数似然 → PPO/DPO/KTO/REINFORCE
  似然比不可用。离线数据 + 无仿真器时，标准且可实现的 RL 策略改进算子是
  **优势加权回归（AWR / CRR）**——这正是 flow-QL / Diffusion-QL / SARM(RA-BC)
  的"策略提取"步骤。本模块把 R1b 的组优势 + P3 critic 的 V 水平优势 + 人标
  indicator 合并成**逐帧密集优势**，再经 AWR/CRR/linear 算子映射为逐样本权重，
  供训练侧加权 flow-matching 回归使用。

与 R2（RecapWeighter，filtered BC）的区别：
  R2 权重 = 启发式（good×1 / bad×0.1 / 干预×2 / 组优势线性调制）；
  R3 权重 = 以 critic 密集优势为信号、经 RL 算子（exp(A/τ) 等）得出，
  是真正"按优势重加权"的策略改进（AWR 论文的目标：max E[A·log π]）。

优势来源（逐帧，优先级从高到低，见 compute_pg_advantage）：
  1. 人工干预帧（is_human_intervention）→ A=+1（RECAP 规则：干预恒正）
  2. 人标 advantage_indicator ≠ 0 → A=±1（人标优先，最高可信）
  3. **TD 优势（recap_value.py advantage 产物 td_advantage 列，存在时优先）**：
     A_t = norm(r_{t:t+N}) + γ^N·V(o_{t+N}) − V(o_t)，真奖励前瞻 + 价值自举，
     信用分配比 V 水平口径更准（RLinf RECAP Step 3 同式）；z-score 归一化
  4. critic V 水平优势 = V(s) − 本集 mean(V)，全局 z-score 归一化（recap_critic.py
     写出的 critic_value 列；bad 帧 V 低→负优势，good 帧 V 高→正优势）
  5. group_advantage（R1b 组相对优势，episode 级，未标注帧兜底）
  6. 其余 → A=0（不调制）

权重算子（policy improvement operator）：
  awr    w = exp(A / τ)             （AWR，τ=温度；τ→∞ 退化为均匀）
  crr    w = 1[A>0] ? 1.0 : negative （CRR-max，负优势降权不滤除）
  linear w = clamp(1 + β·A)          （与 R1b 连续，用于消融）
  干预帧再 ×intervention_weight；负优势帧夹爪维豁免（APO，见 RecapWeighter）。

用法（cd debugger；sidecar 需先生成，建议先生成 group_advantage 与 critic）：
  python3 recap_pg.py --dataset lerobot_data/ela3_demo5 --operator awr --preview
  python3 recap_pg.py --dataset lerobot_data/ela3_demo5 --operator awr --write  # 写 pg_advantage 列
  python3 recap_pg.py --dataset lerobot_data/ela3_demo5 --operator crr --tau 0.5 --preview

顺序纪律：本脚本必须在"生成训练 sidecar"之后运行；若 critic / 组优势尚未生成，
缺失来源自动降级（无 critic_value → 用 indicator+group；无 group_advantage → A=0）。
训练侧读取 pg_advantage 的是 site-packages 的 RecapPGWeighter（type="recap_pg"）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "visualize_data"))

# 默认算子参数（与 yaml sample_weighting.extra_params 一一对应）
DEFAULT_OPERATOR = "awr"
DEFAULT_TAU = 1.0
DEFAULT_BETA = 1.0
DEFAULT_NEGATIVE = 0.1
DEFAULT_INTERVENTION = 2.0
DEFAULT_MIN_WEIGHT = 0.1       # 与 R2 的 bad×0.1 一致（降权而非过滤）
DEFAULT_MAX_WEIGHT = 5.0
DEFAULT_CLIP_ADV = 1.0         # 最终优势裁剪到 ±1（统一人标/组优势/critic 的尺度）
DEFAULT_GUARD_FRAC = 1.0       # critic 校验：V(good)−V(bad) 分离度须 > frac×σ(V)（弱 critic 判废）
DEFAULT_GROUP_BETA = 0.5       # π*0.6 组相对优势 episode 级调制权重（A += β_g×group_adv）


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


def _col(rows: list[dict], key: str, dtype=np.float32, default=0.0):
    out = np.full(len(rows), default, dtype=dtype)
    for i, r in enumerate(rows):
        v = r.get(key)
        if v is None:
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def compute_pg_advantage(
    rows: list[dict],
    clip_adv: float = DEFAULT_CLIP_ADV,
    critic_guard: bool = True,
    guard_frac: float = DEFAULT_GUARD_FRAC,
    group_beta: float = DEFAULT_GROUP_BETA,
) -> tuple[np.ndarray, dict]:
    """合并多来源为逐帧密集优势 A(t)。

    critic 优势来源用「V 水平相对基线」（非原 ΔV）：A = V(s) − 本集 mean(V)，再全局
    z-score 到单位方差、裁剪 ±clip_adv。守卫用 V 水平分离度：mean V(bad) 应显著低于
    mean V(good)（分离度 > guard_frac×σ(V)），否则丢弃 critic 来源退回 indicator。
    原 ΔV 口径方向测反（测坏段结束回升而非进入下降，2026-08-30 实测误判），已弃用。

    组相对优势（R1b，GRPO 组内标准化，来自 episode_success 独立标注）作为 episode 级
    加性调制项（π*0.6 方式）：A(t) += group_beta × group_adv(episode)，作用于该集
    所有帧，再裁剪 ±clip_adv。组优势免疫任务难度偏差，只有成组(≥4轮)的集才有非零值。

    Returns:
        (adv (N,) float32, stats dict)
    """
    n = len(rows)
    indicator = _col(rows, "advantage_indicator", np.int8, 0)
    itv = _col(rows, "is_human_intervention", np.uint8, 0)
    group_adv = _col(rows, "group_advantage", np.float32, 0.0)
    ep_idx = _col(rows, "episode_index", np.int64, -1)

    # critic_value：缺失帧用 NaN 标记（读原始 dict，区分"无 critic"与"0.0"）
    critic_v = np.full(n, np.nan, dtype=np.float32)
    # td_advantage（recap_value.py advantage 产物）：已含真奖励前瞻 + V 自举，
    # 存在时优先于 critic_value 的 V 水平口径
    td_adv = np.full(n, np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        if "critic_value" in r and r["critic_value"] is not None:
            critic_v[i] = float(r["critic_value"])
        if "td_advantage" in r and r["td_advantage"] is not None:
            td_adv[i] = float(r["td_advantage"])
    td_available = not np.isnan(td_adv).all()

    critic_used = False
    adv_critic = np.zeros(n, dtype=np.float32)
    v_bad = v_good = v_std = float("nan")
    if not np.isnan(critic_v).all():
        valid = ~np.isnan(critic_v)
        bad_mask = (indicator < 0)
        good_mask = (indicator > 0)
        v_bad = float(np.nanmean(critic_v[bad_mask])) if bad_mask.any() else float("nan")
        v_good = float(np.nanmean(critic_v[good_mask])) if good_mask.any() else float("nan")
        v_std = float(np.nanstd(critic_v[valid]))

        # critic 有效性校验：V(bad) 应显著低于 V(good)
        critic_ok = True
        if critic_guard and bad_mask.any() and good_mask.any():
            separation = v_good - v_bad
            critic_ok = (separation > 0) and (v_std > 0) and (separation > guard_frac * v_std)
            if not critic_ok:
                print(f"[recap_pg] critic 校验未通过（V(bad)={v_bad:.4f} V(good)={v_good:.4f} "
                      f"分离度={separation:.4f} ≤ {guard_frac}×σ={guard_frac * v_std:.4f}）"
                      f"——critic 未学到好坏区分，丢弃，退回 indicator+组优势来源")

        if critic_ok:
            # V 水平相对基线：V − 本集 mean(V)，再全局 z-score 到单位方差
            v_dm = critic_v.copy()
            for ep in np.unique(ep_idx[valid]):
                m = (ep_idx == ep) & valid
                v_dm[m] = critic_v[m] - float(np.nanmean(critic_v[m]))
            mu = float(np.nanmean(v_dm[valid]))
            std = float(np.nanstd(v_dm[valid]))
            if std < 1e-6:
                adv_critic = np.zeros(n, dtype=np.float32)
            else:
                adv_critic = np.clip((v_dm - mu) / std, -clip_adv, clip_adv).astype(np.float32)
            critic_used = True

    # TD 优势来源（recap_value.py advantage 产物）：优先于 V 水平口径。
    # 校验口径同 critic：bad 帧的 TD 优势应显著低于 good 帧（分离度 > frac×σ(td)）。
    if td_available:
        valid_td = ~np.isnan(td_adv)
        bad_mask = (indicator < 0) & valid_td
        good_mask = (indicator > 0) & valid_td
        td_ok = True
        td_bad = td_good = td_std = float("nan")
        if bad_mask.any() and good_mask.any():
            td_bad = float(np.nanmean(td_adv[bad_mask]))
            td_good = float(np.nanmean(td_adv[good_mask]))
            td_std = float(np.nanstd(td_adv[valid_td]))
            separation = td_good - td_bad
            td_ok = (not critic_guard) or (
                separation > 0 and td_std > 0 and separation > guard_frac * td_std)
            if not td_ok:
                print(f"[recap_pg] TD 优势校验未通过（A(bad)={td_bad:.4f} A(good)={td_good:.4f} "
                      f"分离度={separation:.4f} ≤ {guard_frac}×σ={guard_frac * td_std:.4f}）"
                      f"——丢弃 TD 来源，退回 critic V 水平/indicator")
        if td_ok:
            # TD 优势已是优势空间（零均值附近），直接全局 z-score，不再逐集去均值
            mu = float(np.nanmean(td_adv[valid_td]))
            std = float(np.nanstd(td_adv[valid_td]))
            if std > 1e-6:
                adv_td = np.clip((td_adv - mu) / std, -clip_adv, clip_adv).astype(np.float32)
                adv_critic = np.where(valid_td, adv_td, 0.0).astype(np.float32)
                critic_used = True
                td_used = True
            else:
                td_used = False
        else:
            td_used = False
    else:
        td_used = False

    # 合并逐帧优势（优先级：干预 > 人标 > critic(TD 优先) > 0）
    adv = np.zeros(n, dtype=np.float32)
    adv[itv > 0] = 1.0
    labeled = (indicator != 0)
    adv[labeled] = indicator[labeled].astype(np.float32)
    dense_valid = (~np.isnan(td_adv)) if td_used else (~np.isnan(critic_v))
    critic_fill = (indicator == 0) & (itv == 0) & dense_valid & critic_used
    adv[critic_fill] = adv_critic[critic_fill]

    # π*0.6 组相对优势：episode 级加性调制（GRPO 组内标准化），作用于该集所有帧
    group_modulated = (group_adv != 0)
    if group_beta != 0 and group_modulated.any():
        adv = adv + (group_beta * group_adv).astype(np.float32)

    # 最终裁剪：统一各来源尺度到 [-clip_adv, clip_adv]（人标 ±1 / critic V 水平 / 组优势调制）
    adv = np.clip(adv, -clip_adv, clip_adv).astype(np.float32)

    stats = {
        "n": n,
        "n_human_labeled": int(labeled.sum()),
        "n_intervention": int((itv > 0).sum()),
        "n_critic_filled": int(critic_fill.sum()),
        "n_group_modulated": int(group_modulated.sum()),
        "n_neutral": int((adv == 0).sum()),
        "n_pos": int((adv > 0).sum()),
        "n_neg": int((adv < 0).sum()),
        "critic_used": bool(critic_used),
        "td_used": bool(td_used),
        "critic_v_bad": round(v_bad, 4) if np.isfinite(v_bad) else None,
        "critic_v_good": round(v_good, 4) if np.isfinite(v_good) else None,
        "critic_v_std": round(v_std, 4) if np.isfinite(v_std) else None,
        "group_beta": group_beta,
    }
    return adv, stats


def apply_operator(
    adv: np.ndarray,
    itv: np.ndarray,
    operator: str = DEFAULT_OPERATOR,
    tau: float = DEFAULT_TAU,
    beta: float = DEFAULT_BETA,
    negative: float = DEFAULT_NEGATIVE,
    intervention: float = DEFAULT_INTERVENTION,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> np.ndarray:
    """把优势映射为逐样本权重（策略改进算子）。返回 (N,) float32 原始权重（未 batch 归一化）。"""
    a = np.asarray(adv, dtype=np.float32)
    if operator == "awr":
        w = np.exp(a / max(tau, 1e-6)).astype(np.float32)
    elif operator == "crr":
        # CRR-max：正优势→1.0，负优势→negative，中性(A=0)→1.0（正常模仿，防遗忘）
        w = np.where(a > 0, 1.0, np.where(a < 0, negative, 1.0)).astype(np.float32)
    elif operator == "linear":
        w = (1.0 + beta * a).astype(np.float32)
    else:
        raise ValueError(f"未知算子: {operator}（可选 awr/crr/linear）")
    w = np.clip(w, min_weight, max_weight)
    itv_mask = np.asarray(itv, dtype=bool)
    w[itv_mask] *= intervention
    return np.clip(w, min_weight, max_weight).astype(np.float32)


def preview_text(dataset_name: str, adv: np.ndarray, itv: np.ndarray,
                 stats: dict, operator: str, tau: float, beta: float,
                 negative: float, intervention: float) -> str:
    w = apply_operator(adv, itv, operator, tau, beta, negative, intervention)
    n = len(adv)
    lines = [
        f"数据集: {dataset_name}  总帧数: {n}",
        f"优势分布: 正 A={stats['n_pos']} 负 A={stats['n_neg']} 中性/未标 A=0={stats['n_neutral']}",
        f"来源构成: 人标 {stats['n_human_labeled']} 干预 {stats['n_intervention']} "
        f"critic {stats['n_critic_filled']} 组优势调制 {stats['n_group_modulated']} 帧",
    ]
    if stats["critic_v_std"] is not None:
        lines.append(f"critic V: bad={stats['critic_v_bad']} good={stats['critic_v_good']} "
                     f"σ={stats['critic_v_std']}（分离度 good−bad={stats['critic_v_good'] - stats['critic_v_bad'] if stats['critic_v_good'] is not None and stats['critic_v_bad'] is not None else None}）")
    lines.append(f"critic 来源: {'TD 优势（recap_value）' if stats.get('td_used') else ('V 水平（recap_critic）' if stats['critic_used'] else '已禁用/校验未通过')}")
    lines.append(f"组优势调制: β_g={stats['group_beta']}（A += β_g×group_adv，作用于成组集的全部帧）")
    lines.append(f"算子[{operator}]: tau={tau} beta={beta} negative={negative} "
                 f"intervention={intervention} clamp=[{DEFAULT_MIN_WEIGHT},{DEFAULT_MAX_WEIGHT}]")
    lines.append(f"权重分布: mean={w.mean():.3f} min={w.min():.3f} max={w.max():.3f}"
                 f"（有效样本量≈{w.sum() / n * 100:.1f}% of {n}）")
    pos = adv > 0
    neg = adv < 0
    if pos.any():
        lines.append(f"  正优势帧权重: mean={w[pos].mean():.3f}（{int(pos.sum())} 帧）")
    if neg.any():
        lines.append(f"  负优势帧权重: mean={w[neg].mean():.3f}（{int(neg.sum())} 帧）")
    lines.append(f"  中性帧权重: 1.0（{int((adv == 0).sum())} 帧，正常模仿）")
    return "\n".join(lines)


def write_pg_advantage(rows: list[dict], adv: np.ndarray, path: Path) -> int:
    tmp = path.with_suffix(".jsonl.tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r, a in zip(rows, adv, strict=True):
            r = dict(r)
            r["pg_advantage"] = round(float(a), 4)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def main():
    ap = argparse.ArgumentParser(
        description="π*0.6 R3 优势加权的策略改进（pi0.5 flow-matching offline RL）")
    ap.add_argument("--dataset", required=True, help="数据集根目录")
    ap.add_argument("--operator", default=DEFAULT_OPERATOR, choices=["awr", "crr", "linear"],
                    help="策略改进算子（默认 awr）")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU, help="AWR 温度 τ（默认 1.0）")
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA, help="linear 算子斜率 β（默认 1.0）")
    ap.add_argument("--negative", type=float, default=DEFAULT_NEGATIVE,
                    help="负优势帧权重下限（crr 用，默认 0.1）")
    ap.add_argument("--intervention", type=float, default=DEFAULT_INTERVENTION,
                    help="干预帧乘子（默认 2.0）")
    ap.add_argument("--clip-adv", type=float, default=DEFAULT_CLIP_ADV,
                    help="最终优势裁剪到 ±N（默认 1.0，统一各来源尺度）")
    ap.add_argument("--no-critic-guard", action="store_true",
                    help="关闭 critic 有效性校验（强制使用 critic V 水平优势，即使校验未通过）")
    ap.add_argument("--guard-frac", type=float, default=DEFAULT_GUARD_FRAC,
                    help="critic 校验分离度阈值（默认 1.0：V(good)−V(bad) 须 > 1.0×σ）")
    ap.add_argument("--group-beta", type=float, default=DEFAULT_GROUP_BETA,
                    help="组相对优势 episode 级调制权重 β_g（默认 0.5：A += β_g×group_adv）")
    ap.add_argument("--preview", action="store_true", help="打印权重分布预览（默认行为）")
    ap.add_argument("--write", action="store_true", help="把 pg_advantage 列写回 sidecar")
    args = ap.parse_args()

    root = Path(args.dataset)
    rows, sc_path = load_sidecar_rows(root)
    adv, stats = compute_pg_advantage(rows, clip_adv=args.clip_adv,
                                      critic_guard=not args.no_critic_guard,
                                      guard_frac=args.guard_frac,
                                      group_beta=args.group_beta)
    itv = _col(rows, "is_human_intervention", np.uint8, 0)

    if not args.write or args.preview:
        print(preview_text(root.name, adv, itv, stats, args.operator, args.tau,
                           args.beta, args.negative, args.intervention))

    if args.write:
        n = write_pg_advantage(rows, adv, sc_path)
        print(f"已写出 pg_advantage 列（{n} 行，原子替换）-> {sc_path}")


if __name__ == "__main__":
    main()

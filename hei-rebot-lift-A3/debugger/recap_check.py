#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recap_check.py — RECAP 训练前门禁（数据准备链第 ⑧ 步）。

对指定数据集逐项检测 ①~⑦ 的产物，全部硬性检查通过后输出 yaml 修改建议与
训练启动命令（GUI「RECAP 预处理」第 ⑧ 步调用本脚本；也可命令行单独使用）。

硬性检查（✗ 不通过则拒绝给出训练建议）：
  1. sidecar 新鲜度：行数 == info.json total_frames（剪辑/合并后未重跑会不等）
  2. action≠state 抽查：mean|action−state| > 1e-4（P0.4 问题5 防线，
     action==state 退化曾致真机全失败）
  3. returns_<tag>.parquet 存在且覆盖全部 episode（③ 产物）
  4. 成败集统计：成功与失败集均需存在（无失败集则优势/价值监督无意义）
  5. sidecar 含 pg_advantage 列（⑥ 已写出；RecapPGWeighter 实际读取）
  6. pg_advantage 分布合理性：干预帧全正 / bad 帧全负 / AWR 权重在
     [0.1,5] 且有效样本量 90%~140%
  7. 训练环境补丁：lerobot sample_weighting.py 含 recap_pg（RecapPGWeighter；
     重装 lerobot 后须按 TRAIN_README §4 重打，否则训练静默退化为普通 SFT）

软性检查（⚠ 只提示不阻断）：
  8. 价值头 recap_value_<tag>/value_head.pt 是否存在；存在且有特征缓存时
     计算 V/TD 分离度（弱则提示 ⑥ 将退回人标+组优势——这是守卫正常履职）
  9. sidecar 是否含 td_* 列（⑤ 产物）与 group_advantage 是否有非零（② 产物）

用法（cd debugger）：
  python3 recap_check.py --dataset lerobot_data/ela3_demo6 --tag cfail500

exit code：0=硬性全过（附 yaml/命令建议）；1=有硬性失败（逐项给出修复指引）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEBUGGER_DIR = Path(__file__).resolve().parent


class Reporter:
    def __init__(self):
        self.n_fail = 0
        self.n_warn = 0

    def ok(self, msg):
        print(f"✓ {msg}")

    def warn(self, msg):
        print(f"⚠ {msg}")
        self.n_warn += 1

    def fail(self, msg, fix=""):
        print(f"✗ {msg}")
        if fix:
            print(f"    修复: {fix}")
        self.n_fail += 1


# ---------------------------------------------------------------------------
# 硬性检查
# ---------------------------------------------------------------------------
def check_sidecar_fresh(root: Path, rep: Reporter) -> list[dict] | None:
    sc = root / "meta" / "recap_sidecar.jsonl"
    if not sc.is_file():
        rep.fail("缺少 meta/recap_sidecar.jsonl", "先运行 ① 生成训练 sidecar")
        return None
    rows = [json.loads(l) for l in open(sc, encoding="utf-8") if l.strip()]
    try:
        info = json.load(open(root / "meta" / "info.json", encoding="utf-8"))
        total = int(info.get("total_frames", -1))
    except Exception as e:
        rep.fail(f"info.json 读取失败: {e}")
        return None
    if len(rows) == total:
        rep.ok(f"1. sidecar 新鲜度: {len(rows)} 行 == total_frames（未过期）")
    else:
        rep.fail(f"1. sidecar 过期: {len(rows)} 行 ≠ total_frames {total}",
                 "剪辑/合并/删除后必须重跑 ① 生成 sidecar")
    return rows


def check_action_not_state(root: Path, rep: Reporter, max_shards: int = 4):
    """P0.4 问题5 防线：mean|action−state| > 1e-4（逐集抽查前几个 shard）。"""
    import pandas as pd
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        rep.fail("2. 未找到 data parquet", "数据集结构异常")
        return
    diffs = []
    try:
        dfs = [pd.read_parquet(f, columns=["action", "observation.state", "episode_index"])
               for f in files[:max_shards]]
        df = pd.concat(dfs)
        for _ep, g in df.groupby("episode_index"):
            a = np.vstack(g["action"].to_numpy())
            s = np.vstack(g["observation.state"].to_numpy())
            diffs.append(float(np.abs(a - s).mean()))
    except Exception as e:
        rep.fail(f"2. action≠state 抽查异常: {e}")
        return
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    if mean_diff > 1e-4:
        rep.ok(f"2. action≠state: mean|a−s|={mean_diff:.4f}（抽查 {len(diffs)} 集）")
    else:
        rep.fail(f"2. action==state 退化（mean={mean_diff:.2e}）——整集报废不得训练",
                 "检查录制链路 lookahead 写入（action[t]=pose[t+N]），重录数据")


def check_returns(root: Path, tag: str, rep: Reporter) -> tuple[float, int, int]:
    """returns parquet 覆盖性 + 成败集统计。返回 (denominator, n_success, n_failure)。"""
    import pandas as pd
    pq = root / "meta" / f"returns_{tag}.parquet"
    js = root / "meta" / f"returns_{tag}.json"
    if not pq.is_file():
        rep.fail(f"3. 缺少 meta/returns_{tag}.parquet", "先运行 ③ 计算回报（tag 需一致）")
        return 1.0, 0, 0
    df = pd.read_parquet(pq)
    denom = 1.0
    if js.is_file():
        denom = float(json.load(open(js, encoding="utf-8")).get("denominator", 1.0))
    # 覆盖性：returns 集数 == meta/episodes 集数（跳过集除外，给出提示）
    n_eps = len(set(int(e) for e in df["episode_index"]))
    import pyarrow.parquet as pqm
    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    total_eps = sum(pqm.read_metadata(f).num_rows for f in ep_files)
    skipped = total_eps - n_eps
    if skipped == 0:
        rep.ok(f"3. returns_{tag}.parquet: 覆盖全部 {n_eps} 集，分母 {denom:.1f}")
    else:
        rep.warn(f"3. returns_{tag}.parquet: {n_eps}/{total_eps} 集"
                 f"（{skipped} 集未评被跳过——属正常，未评集不进训练）")
    # 成败集统计（hard：需要同时有成功与失败）
    ep_succ = df.groupby("episode_index")["episode_success"].first()
    n_succ, n_fail = int((ep_succ == 1).sum()), int((ep_succ == 0).sum())
    if n_succ > 0 and n_fail > 0:
        rep.ok(f"4. 成败集统计: 成功 {n_succ} / 失败 {n_fail}（成败区分度来源齐备）")
    else:
        rep.fail(f"4. 成败集统计: 成功 {n_succ} / 失败 {n_fail}",
                 "失败集为 0 时价值/优势无监督——用推理录制采集含失败的 rollout 后重走 ①→⑥")
    return denom, n_succ, n_fail


def check_pg_column(rows: list[dict], rep: Reporter) -> bool:
    if rows is None:
        rep.fail("5. sidecar 不可用，无法检查 pg_advantage")
        return False
    has = "pg_advantage" in rows[0]
    if has:
        rep.ok("5. sidecar 含 pg_advantage 列（⑥ 已写出）")
    else:
        rep.fail("5. sidecar 缺少 pg_advantage 列",
                 "运行 ⑥ 优势合并写出（recap_pg.py --write）——RecapPGWeighter 读此列")
    return has


def check_pg_sanity(rows: list[dict], rep: Reporter):
    """pg_advantage 分布合理性（hard）：方向正确性 + AWR 权重区间。"""
    if not rows or "pg_advantage" not in rows[0]:
        return
    adv = np.array([float(r.get("pg_advantage", 0) or 0) for r in rows])
    itv = np.array([int(r.get("is_human_intervention", 0) or 0) for r in rows])
    ind = np.array([int(r.get("advantage_indicator", 0) or 0) for r in rows])

    problems = []
    n_itv = int((itv > 0).sum())
    if n_itv and not (adv[itv > 0] > 0).all():
        problems.append(f"干预帧含非正优势（{int((adv[itv > 0] <= 0).sum())}/{n_itv}）")
    n_bad = int((ind < 0).sum())
    if n_bad and not (adv[ind < 0] < 0).all():
        problems.append(f"bad 帧含非负优势（{int((adv[ind < 0] >= 0).sum())}/{n_bad}）")

    # AWR 权重（与 RecapPGWeighter 同口径：exp(A/τ) ×干预 → clamp[0.1,5]）
    w = np.exp(np.clip(adv, -1.0, 1.0) / 1.0)
    w = np.clip(w, 0.1, 5.0)
    if n_itv:
        w[itv > 0] = np.clip(w[itv > 0] * 2.0, 0.1, 5.0)
    eff = float(w.sum() / len(w) * 100)
    wmin, wmax = float(w.min()), float(w.max())
    if not (0.09 <= wmin and wmax <= 5.01):
        problems.append(f"权重越界 [{wmin:.3f}, {wmax:.3f}]")
    if not (90.0 <= eff <= 140.0):
        problems.append(f"有效样本量 {eff:.1f}% 偏离 [90%,140%] 合理带")

    if problems:
        rep.fail("6. pg_advantage 分布异常: " + "；".join(problems),
                 "检查 ⑥ 的参数（--operator/--intervention/--group-beta）后重跑 ⑥")
    else:
        rep.ok(f"6. pg_advantage 分布: 正 {int((adv > 0).sum())} / 负 {int((adv < 0).sum())} "
               f"/ 中性 {int((adv == 0).sum())}；AWR 权重 mean={w.mean():.3f} "
               f"[{wmin:.3f},{wmax:.3f}]，有效样本量 {eff:.1f}%")


def check_training_env(rep: Reporter):
    """训练环境补丁（hard）：lerobot sample_weighting.py 须含 recap_pg。"""
    try:
        import inspect
        import lerobot
        sw = Path(inspect.getfile(lerobot)).parent / "utils" / "sample_weighting.py"
        if sw.is_file() and "recap_pg" in sw.read_text(encoding="utf-8"):
            rep.ok(f"7. 训练环境补丁: RecapPGWeighter 已就位（{sw.name}）")
        else:
            rep.fail("7. lerobot sample_weighting.py 缺少 recap_pg（RecapPGWeighter）",
                     "重装 lerobot 后须按 TRAIN_README §4 重打补丁，否则训练静默退化为普通 SFT")
    except ImportError:
        rep.fail("7. 当前 python 环境无 lerobot",
                 "本脚本须在 VLA 环境运行（GUI 已默认如此）；命令行请用 VLA 环境 python")


# ---------------------------------------------------------------------------
# 软性检查
# ---------------------------------------------------------------------------
def check_value_head(tag: str, root: Path, denom: float, rows: list[dict] | None,
                     rep: Reporter, eval_separation: bool = True):
    head_dir = DEBUGGER_DIR / "lerobot_data" / f"recap_value_{tag}"
    head_path = head_dir / "value_head.pt"
    if not head_path.is_file():
        rep.warn(f"8. 价值头不存在（{head_path.relative_to(DEBUGGER_DIR)}）——"
                 f"④⑤ 未跑，⑥ 将仅用人标+组优势（可训练，建议补跑④⑤增强 TD 通道）")
        return
    if not eval_separation:
        rep.ok("8. 价值头存在（--no-value-eval：跳过分力度评估）")
        return
    try:
        import torch
        ck = torch.load(head_path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        backend = cfg.get("backend", "state")
        rep.ok(f"8. 价值头存在（backend={backend}, denominator={cfg.get('denominator', '?')}, "
               f"fps={cfg.get('fps', '?')}）")
        hdenom = float(cfg.get("denominator", denom))
        if abs(hdenom - denom) > 1e-6:
            rep.warn(f"  价值头分母 {hdenom:.1f} ≠ returns 分母 {denom:.1f}——"
                     f"tag 混用（不同 cfail 的 returns/train 不一致），建议同 tag 重跑 ③④⑤")
        _separation_report(root, backend, ck, denom, rows, rep)
    except Exception as e:
        rep.warn(f"8. 价值头读取/分离度计算失败（不阻断）: {e}")


def _separation_report(root: Path, backend: str, ck: dict, denom: float,
                       rows: list[dict] | None, rep: Reporter):
    """V/TD 分离度复算（需要特征缓存与 torch；任何失败降级为提示）。"""
    import torch
    cache = root / "meta" / f"critic_feats_{backend}.npz"
    if rows is None or not cache.is_file():
        rep.warn(f"  分离度复算跳过（缺 sidecar 或特征缓存 critic_feats_{backend}.npz）")
        return
    p = {n: ck[n].float() for n in ["ln_g", "ln_b", "W1", "b1", "W2", "b2", "W3", "b3"]}
    n_bins = p["W3"].shape[1]
    atoms = np.linspace(-1.0, 0.0, n_bins, dtype=np.float32)
    z = np.load(cache)
    feats, keys = z["feats"], z["keys"]
    x = torch.from_numpy(feats)
    V = np.empty(len(x), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(x), 8192):
            xb = x[i:i + 8192]
            h = torch.nn.functional.layer_norm(xb, (xb.shape[-1],), p["ln_g"], p["ln_b"])
            h = torch.relu(h @ p["W1"] + p["b1"])
            h = torch.relu(h @ p["W2"] + p["b2"])
            probs = torch.softmax(h @ p["W3"] + p["b3"], dim=-1)
            V[i:i + 8192] = (probs * torch.from_numpy(atoms)).sum(-1).numpy()
    vmap = {(int(k[0]), int(k[1])): v for k, v in zip(keys, V)}
    ind = np.array([int(r.get("advantage_indicator", 0) or 0) for r in rows])
    vk = np.array([vmap.get((int(r["episode_index"]), int(r["frame_index"])), np.nan)
                   for r in rows])
    if not (ind < 0).any() or not (ind > 0).any():
        rep.warn("  分离度复算跳过（无 bad/good 标注帧）")
        return
    sep_v = float(vk[ind > 0].mean() - vk[ind < 0].mean())
    std_v = float(np.nanstd(vk))
    ratio_v = sep_v / std_v if std_v > 0 else 0.0

    # TD 优势（N=10, γ=1，与 ⑤ 同式）
    import pandas as pd
    n_step = 10
    a_map = {}
    tag_parquets = sorted((root / "meta").glob("returns_*.parquet"))
    if not tag_parquets:
        rep.warn("  TD 分离度跳过（无 returns parquet）")
        return
    ret = pd.read_parquet(tag_parquets[0])
    rew = {int(ep): g.sort_values("frame_index")["reward"].to_numpy()
           for ep, g in ret.groupby("episode_index")}
    for i, r in enumerate(rows):
        ep, f = int(r["episode_index"]), int(r["frame_index"])
        rr = rew.get(ep)
        if rr is None or f >= len(rr):
            continue
        r_sum = float(rr[f:f + n_step].sum())
        v_next = vmap.get((ep, f + n_step), 0.0) if f + n_step < len(rr) else 0.0
        a_map[i] = r_sum / denom + v_next - vmap.get((ep, f), 0.0)
    A = np.array([a_map.get(i, np.nan) for i in range(len(rows))])
    valid = ~np.isnan(A)
    sep_a = float(np.nanmean(A[(ind > 0) & valid]) - np.nanmean(A[(ind < 0) & valid]))
    std_a = float(np.nanstd(A[valid]))
    ratio_a = sep_a / std_a if std_a > 0 else 0.0

    def _verdict(ratio):
        return "强（守卫将启用）" if ratio > 1.0 else ("弱（⑥ 守卫将退回人标+组优势，不阻断）" if ratio > 0.2 else "极弱（价值头主要学到进度，⑥ 将退回）")
    print(f"  · V 分离度 {sep_v:+.4f}（{ratio_v:.2f}σ）→ {_verdict(ratio_v)}")
    print(f"  · TD 分离度 {sep_a:+.4f}（{ratio_a:.2f}σ）→ {_verdict(ratio_a)}")
    if ratio_a <= 0.2:
        print("    增强建议：失败/bad 样本量积累后自然增强；或给特征加入 gripper_torque"
              "（抓空/滑脱的力矩前兆是最强失败先兆）")


def check_td_group_columns(rows: list[dict] | None, rep: Reporter):
    if not rows:
        return
    r0 = rows[0]
    if "td_advantage" in r0:
        rep.ok("9. sidecar 含 td_value/td_advantage/td_label 列（⑤ 已跑）")
    else:
        rep.warn("9. sidecar 无 td_* 列（⑤ 未跑——TD 通道未启用，可训练但建议补跑④⑤）")
    grp = np.array([float(r.get("group_advantage", 0) or 0) for r in rows])
    nz = int((grp != 0).sum())
    if nz:
        rep.ok(f"  组优势: {nz} 帧非零（② 已生效，成组集参与调制）")
    else:
        rep.warn("  组优势全 0（② 未跑或无成组数据——未分组数据按无组处理，可接受）")


# ---------------------------------------------------------------------------
# 训练建议输出
# ---------------------------------------------------------------------------
def print_training_advice(ds_name: str, yaml_name: str):
    # 按目标 yaml 推断策略前缀（pi05_omy_*.yaml → pi05；smolvla_omy_*.yaml → smolvla）
    prefix = "smolvla" if "smolvla" in yaml_name else "pi05"
    job = f"{prefix}_{ds_name}_recap_pg_fullft"
    hw = "16GB 4080S 本机可跑" if prefix == "smolvla" else "⚠ 需 48GB 服务器；16GB 会立即 OOM"
    print()
    print("=" * 60)
    print("✅ 硬性检查全部通过，可以开始训练")
    print("=" * 60)
    print()
    print(f"yaml 修改（{yaml_name}，共 4 处）：")
    print(f"  dataset:")
    print(f"    repo_id: {ds_name}")
    print(f"    root: ./lerobot_data/{ds_name}")
    print(f"  output_dir: ./ckpt/{job}")
    print(f"  job_name: {job}")
    print()
    print(f"训练命令（{hw}）：")
    print("  HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True \\")
    print(f"    python train_model.py --config_path {yaml_name}"
          f" 2>&1 | tee train_recap_pg_{prefix}.log")
    print()
    print("对照实验纪律（判断 R3 是否起效，同 seed 三组）：")
    print("  ① recap_pg（本配置）② type 改 recap（R2 加权）③ 删 sample_weighting 段（纯 SFT）")
    print("  评估：test_inference_episode.py 留出集 mean action error + 真机干预率对比")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="RECAP 训练前门禁（第 ⑧ 步）：检查 ①~⑦ 产物，通过后给出训练建议")
    ap.add_argument("--dataset", required=True, help="数据集根目录")
    ap.add_argument("--tag", default="cfail500", help="实验标签（与 ③④⑤ 一致）")
    ap.add_argument("--yaml", default="pi05_omy_fullft_recap_pg.yaml",
                    help="目标训练 yaml 文件名（建议输出据此生成；pi05 需 48GB，smolvla 本机可跑）")
    ap.add_argument("--no-value-eval", action="store_true",
                    help="跳过价值头分离度评估（默认评估；加载特征缓存 ~30s）")
    args = ap.parse_args()

    root = Path(args.dataset)
    if not root.is_dir():
        print(f"✗ 数据集不存在: {root}")
        return 1
    ds_name = root.name

    print("=" * 60)
    print(f"RECAP 训练前门禁 | 数据集: {ds_name} | tag: {args.tag}")
    print("=" * 60)

    rep = Reporter()
    rows = check_sidecar_fresh(root, rep)
    check_action_not_state(root, rep)
    denom, _ns, _nf = check_returns(root, args.tag, rep)
    has_pg = check_pg_column(rows, rep)
    if has_pg:
        check_pg_sanity(rows, rep)
    check_training_env(rep)
    check_value_head(args.tag, root, denom, rows, rep,
                     eval_separation=not args.no_value_eval)
    check_td_group_columns(rows, rep)

    print()
    print("-" * 60)
    print(f"检测汇总: 硬性失败 {rep.n_fail} 项 / 提示 {rep.n_warn} 项")
    if rep.n_fail > 0:
        print("❌ 存在硬性失败项，请按上方『修复』指引处理后重跑本检测")
        return 1
    print_training_advice(ds_name, args.yaml)
    return 0


if __name__ == "__main__":
    sys.exit(main())

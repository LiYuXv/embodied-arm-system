#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recap_value.py — RECAP 价值函数（V 模型）训练管线（RLinf RECAP 四阶段的 Step 1/2/3
ela3 真机复刻，见 RLinf-main/docs/.../recap.rst 与 visualize_data/RLinf与我的项目对比.md）。

奖励结构（DynamicReturn，与用户规格一致）：
  - 每个非终止步 r_t = -1（多耽误一步扣一分 → "尽快做完分数越高"）
  - 终止步：成功 r_T = 0；失败 r_T = -C_fail（--cfail，默认 100）
  - 回报 G_t = Σ γ^k·r_{t+k}（γ 默认 1.0），按全局最小回报归一化到 [-1, 0]
    （分母 = |min G|，即最长失败集的 G_0 ≈ -(max_len-1) - C_fail；与 RLinf 的
    return_min 归一化逐位一致）
  - V(s) 离散为 201 个 bin（atoms = linspace(-1, 0, 201)），two-hot 软标签
    交叉熵训练（RLinf 的 Dirac delta 投影），解码 V = Σ softmax(logits)·atoms

三阶段（顺序执行）：
  returns    逐集计算回报 → meta/returns_<tag>.parquet + returns_<tag>.json（分母等）
  train      201-bin 分布式价值头训练（特征复用 recap_critic 的 state/siglip 后端，
             缓存 npz 共享）；留出集监控 MAE / Spearman / bad-good 分离度
  advantage  N 步 TD 优势 A_t = norm(r_{t:t+N}) + γ^N·V(o_{t+N}) − V(o_t)
             （越界 V=0：回报已结清），分位数二值化（top-q% 为正），
             写回 sidecar 列 td_value/td_advantage/td_label，
             --fill-unlabeled 时未标注帧补 advantage_indicator=±1（人标/干预帧不动）

用法（cd debugger）：
  # Step 1：两轮数据分别算回报（rollout 含失败集才有成败区分度）
  python3 recap_value.py returns --dataset lerobot_data/ela3_demo \
      --dataset lerobot_data/ela3_rollout --cfail 100
  # Step 2：合并两轮数据训价值头（siglip 后端需 pi05_base 视觉塔，state 后端秒级冒烟）
  python3 recap_value.py train --dataset lerobot_data/ela3_demo \
      --dataset lerobot_data/ela3_rollout --backend siglip --tag cfail100
  # Step 3：TD 优势 + 二值化写回（对要参与训练的每个数据集各跑一次）
  python3 recap_value.py advantage --dataset lerobot_data/ela3_rollout \
      --tag cfail100 --lookahead 10 --quantile 0.3 --fill-unlabeled

未评集（episode_success=null，"跳过"）处理：--unrated derive（默认：按片段标注
派生——good 帧占比>50% 且无未恢复 bad 段记成功，否则失败；无任何标注则回退
--unrated-fallback，默认 skip 不进入回报/训练）。

⚠ --fill-unlabeled 语义注意：分位数二值化是 CFG 式标注（top-q% 为正、其余为负，
供 P4 优势条件化 prompt 取符号，与 RLinf Step3→4 一致）；若同时用 R2 的
RecapWeighter 样本降权（bad×0.1），(1-q)% 帧被标负会大面积降权中性数据——
两者同开时建议不加 --fill-unlabeled，改由 recap_pg.py 直接消费 td_advantage
连续列（AWR 加权，中性帧自然落 1.0 附近）。

部署侧衔接（"照这样下去还有多久完成"）：V(s)∈[-1,0] 即"预期剩余代价"，
V→0 表示接近成功完成；推理时解码当前帧 V，A 估计 = norm(r 前瞻) + V' − V 的正负
即"当前动作在拉近还是拉远完成"（正=正向动作，负=负分动作）。模型加载：
value_head.pt + norm.json（atoms/denominator/backend/dim_in）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEBUGGER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEBUGGER_DIR / "visualize_data"))

NUM_BINS = 201
V_MIN, V_MAX = -1.0, 0.0


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _episode_lengths(root: Path) -> dict[int, int]:
    """从 meta/episodes/**/*.parquet 读 {episode_index: length}。"""
    import pyarrow.parquet as pq
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{root} 缺少 meta/episodes parquet")
    out: dict[int, int] = {}
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "length"])
        for ep, ln in zip(t.column("episode_index").to_pylist(),
                          t.column("length").to_pylist()):
            out[int(ep)] = int(ln)
    return out


def _load_annotations(root: Path) -> dict:
    path = root / "meta" / "reward_annotations.json"
    if not path.is_file():
        return {"episodes": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _derive_success(ep_ann: dict, length: int) -> bool | None:
    """未直标集的派生规则（与 derive_episode_success 同口径）：
    good 帧占比 > 50% 且无"未恢复 bad 段"（最后一段 bad 之后没有干预段）→ 成功。
    无任何区间标注 → None（交 --unrated-fallback 决定）。
    """
    itvs = ep_ann.get("intervals") or []
    if not itvs:
        return None
    good = sum(int(i["end"]) - int(i["start"]) + 1
               for i in itvs if i.get("label") == "good")
    bads = [i for i in itvs if i.get("label") == "bad"]
    unrecovered_bad = False
    if bads:
        last_bad_end = max(int(i["end"]) for i in bads)
        unrecovered_bad = not any(
            i.get("is_human_intervention") and int(i["start"]) >= last_bad_end
            for i in itvs)
    if length <= 0:
        return None
    return bool(good / length > 0.5 and not unrecovered_bad)


def _resolve_success(ep_ann: dict, length: int, unrated: str,
                     fallback: str) -> bool | None:
    """人工直标 > 派生 > fallback。返回 True/False/None(跳过)。"""
    v = ep_ann.get("episode_success", None)
    if v is True or v == 1:
        return True
    if v is False or v == 0:
        return False
    if unrated == "fail":
        return False
    if unrated == "skip":
        return None
    # derive
    d = _derive_success(ep_ann, length)
    if d is not None:
        return d
    return None if fallback == "skip" else False


# ---------------------------------------------------------------------------
# Step 1: 计算回报
# ---------------------------------------------------------------------------
def cmd_returns(args):
    for ds_path in args.dataset:
        root = Path(ds_path)
        lengths = _episode_lengths(root)
        ann = _load_annotations(root)
        eps_ann = ann.get("episodes", {})

        # 第一遍：解析成败 + 算 G_0 找分母
        resolved: dict[int, bool] = {}
        n_skip = 0
        g_min = 0.0
        for ep, ln in sorted(lengths.items()):
            succ = _resolve_success(eps_ann.get(str(ep), {}), ln,
                                    args.unrated, args.unrated_fallback)
            if succ is None:
                n_skip += 1
                continue
            resolved[ep] = succ
            g0 = -(args.gamma ** 0) * 0.0  # 占位，下面直接算
            # G_0 = -(L-1) + 终止奖励（γ=1 时；一般式见下循环，这里用闭式）
            g0 = -(ln - 1) + (0.0 if succ else -args.cfail)
            g_min = min(g_min, g0)
        if not resolved:
            raise SystemExit(f"{root.name}: 没有可用集（全部未评且 fallback=skip）")
        denom = abs(g_min) if g_min != 0 else 1.0

        # 第二遍：逐帧回报 + 归一化，写 parquet
        import pandas as pd
        recs = []
        for ep, ln in sorted(lengths.items()):
            if ep not in resolved:
                continue
            succ = resolved[ep]
            rewards = np.full(ln, -1.0, dtype=np.float64)
            rewards[-1] = 0.0 if succ else -args.cfail
            g = np.zeros(ln, dtype=np.float64)
            g[-1] = rewards[-1]
            for t in range(ln - 2, -1, -1):
                g[t] = rewards[t] + args.gamma * g[t + 1]
            src = eps_ann.get(str(ep), {}).get("source", "")
            for t in range(ln):
                recs.append((ep, t, float(rewards[t]), float(g[t]),
                             float(g[t] / denom), int(succ), src))
        df = pd.DataFrame(recs, columns=["episode_index", "frame_index", "reward",
                                         "return", "return_norm",
                                         "episode_success", "source"])
        tag = args.tag or f"cfail{int(args.cfail)}"
        out_pq = root / "meta" / f"returns_{tag}.parquet"
        df.to_parquet(out_pq, index=False)
        meta = {"tag": tag, "cfail": args.cfail, "gamma": args.gamma,
                "denominator": denom, "num_bins": NUM_BINS,
                "v_min": V_MIN, "v_max": V_MAX,
                "episodes_used": len(resolved), "episodes_skipped": n_skip,
                "frames": len(df),
                "success_eps": int(sum(resolved.values())),
                "failure_eps": int(len(resolved) - sum(resolved.values()))}
        with open(root / "meta" / f"returns_{tag}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[returns] {root.name}: {len(resolved)} 集/{len(df)} 帧 → {out_pq.name} "
              f"（成功 {meta['success_eps']} / 失败 {meta['failure_eps']} / 跳过 {n_skip}；"
              f"分母={denom:.1f}，G∈[{g_min:.1f}, 0]）")


# ---------------------------------------------------------------------------
# Step 2: 201-bin 分布式价值头
# ---------------------------------------------------------------------------
class BinValueHead:
    """LayerNorm → 256 → 256 → 201（logits）。atoms=linspace(-1,0,201)，V=E[atoms]。"""

    def __init__(self, dim_in: int, dim_hidden: int = 256, seed: int = 0):
        import torch
        rng = np.random.default_rng(seed)
        def init(*shape):
            return torch.tensor(
                rng.normal(0, (2 / shape[0]) ** 0.5, shape).astype(np.float32),
                requires_grad=True)
        self.W1, self.b1 = init(dim_in, dim_hidden), torch.zeros(dim_hidden, requires_grad=True)
        self.W2, self.b2 = init(dim_hidden, dim_hidden), torch.zeros(dim_hidden, requires_grad=True)
        self.W3, self.b3 = init(dim_hidden, NUM_BINS), torch.zeros(NUM_BINS, requires_grad=True)
        self.ln_g = torch.ones(dim_in, requires_grad=True)
        self.ln_b = torch.zeros(dim_in, requires_grad=True)
        self.atoms = np.linspace(V_MIN, V_MAX, NUM_BINS, dtype=np.float64)
        self.delta_z = (V_MAX - V_MIN) / (NUM_BINS - 1)

    def params(self):
        return [self.ln_g, self.ln_b, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def logits(self, x):
        import torch
        h = torch.nn.functional.layer_norm(x, (x.shape[-1],), self.ln_g, self.ln_b)
        h = torch.relu(h @ self.W1 + self.b1)
        h = torch.relu(h @ self.W2 + self.b2)
        return h @ self.W3 + self.b3  # (B, 201)

    def decode(self, x):
        """V(s) = Σ softmax(logits)·atoms。"""
        import torch
        p = torch.softmax(self.logits(x), dim=-1)
        a = torch.from_numpy(self.atoms).float()
        return (p * a).sum(dim=-1)

    def state(self):
        return {n: v.clone() for n, v in zip(
            ["ln_g", "ln_b", "W1", "b1", "W2", "b2", "W3", "b3"], self.params())}


def _two_hot_ce_loss(logits, targets):
    """two-hot（Dirac delta 投影）软标签交叉熵，与 RLinf _compute_categorical_loss 一致。"""
    import torch
    b = (targets - V_MIN) / ((V_MAX - V_MIN) / (NUM_BINS - 1))
    b = b.clamp(0, NUM_BINS - 1)
    lo = b.floor().long().clamp(0, NUM_BINS - 1)
    hi = b.ceil().long().clamp(0, NUM_BINS - 1)
    d_lo = b - lo.float()
    d_hi = hi.float() - b
    same = lo == hi
    d_lo = torch.where(same, torch.zeros_like(d_lo), d_lo)
    d_hi = torch.where(same, torch.ones_like(d_hi), d_hi)
    B = targets.shape[0]
    probs = torch.zeros(B, NUM_BINS, device=logits.device)
    idx = torch.arange(B, device=logits.device)
    probs[idx, lo] += d_hi
    probs[idx, hi] += d_lo
    return -(probs * torch.nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _load_returns(root: Path, tag: str):
    import pandas as pd
    pq_path = root / "meta" / f"returns_{tag}.parquet"
    js_path = root / "meta" / f"returns_{tag}.json"
    if not pq_path.is_file():
        raise SystemExit(f"缺少 {pq_path}——先运行 recap_value.py returns")
    meta = json.load(open(js_path, encoding="utf-8")) if js_path.is_file() else {}
    return pd.read_parquet(pq_path), meta


def _get_feats_multi(datasets: list[str], backend: str, pi05_path: str):
    """多数据集特征拼接。复用 recap_critic 的提取/缓存；keys=(N,3)=[ds,ep,frame]。"""
    import recap_critic as rc
    feats_all, keys_all, names = [], [], []
    for di, ds_path in enumerate(datasets):
        root = Path(ds_path)
        from vd_dataset import OfflineDataset
        ds = OfflineDataset(str(root))
        feats, keys = rc.get_features(ds, root, backend, pi05_path)
        feats_all.append(feats)
        keys_all.append(np.concatenate(
            [np.full((len(keys), 1), di), keys], axis=1))
        names.append(root.name)
    return np.concatenate(feats_all), np.concatenate(keys_all), names


def cmd_train(args):
    import torch

    # 回报目标按 (ds, ep, frame) 对齐
    y_map: dict[tuple[int, int, int], float] = {}
    for di, ds_path in enumerate(args.dataset):
        df, _meta = _load_returns(Path(ds_path), args.tag)
        for r in df.itertuples(index=False):
            y_map[(di, int(r.episode_index), int(r.frame_index))] = float(r.return_norm)

    feats, keys, names = _get_feats_multi(args.dataset, args.backend, args.pi05_path)
    y = np.array([y_map.get((int(k[0]), int(k[1]), int(k[2])), np.nan)
                  for k in keys], dtype=np.float64)
    valid = ~np.isnan(y)
    feats, keys, y = feats[valid], keys[valid], y[valid]
    print(f"[train] 数据集 {names}，有效帧 {len(y)}（{int((~valid).sum())} 帧无回报被丢弃，"
          f"多为未评跳过集）")

    # 留出集按 (ds, ep) 划分
    ep_set = sorted({(int(k[0]), int(k[1])) for k in keys})
    rng = np.random.default_rng(args.seed)
    n_hold = max(1, int(len(ep_set) * args.holdout))
    hold_eps = set(map(tuple, rng.choice(np.array(ep_set, dtype=object),
                                         n_hold, replace=False).tolist()))
    ep_of = [(int(k[0]), int(k[1])) for k in keys]
    tr = np.array([e not in hold_eps for e in ep_of])
    ho = ~tr
    if ho.sum() == 0:
        raise SystemExit("留出集为空——调低 --holdout")

    head = BinValueHead(feats.shape[1], seed=args.seed)
    opt = torch.optim.AdamW(head.params(), lr=args.lr, weight_decay=1e-4)
    x_all = torch.from_numpy(feats.astype(np.float32))
    y_all = torch.from_numpy(y.astype(np.float32))
    idx_tr = torch.from_numpy(np.where(tr)[0])
    idx_ho = torch.from_numpy(np.where(ho)[0])

    best_ho, best_state, bad_ep = 1e18, None, 0
    for ep in range(args.epochs):
        perm = idx_tr[torch.randperm(len(idx_tr))]
        for i0 in range(0, len(perm), args.batch):
            b = perm[i0:i0 + args.batch]
            loss = _two_hot_ce_loss(head.logits(x_all[b]), y_all[b])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ho_loss = _two_hot_ce_loss(head.logits(x_all[idx_ho]), y_all[idx_ho]).item()
        if ho_loss < best_ho - 1e-4:
            best_ho, best_state, bad_ep = ho_loss, head.state(), 0
        else:
            bad_ep += 1
            if bad_ep >= 5:
                break
    if best_state:
        for p, s in zip(head.params(), [best_state[n] for n in
                                        ["ln_g", "ln_b", "W1", "b1", "W2", "b2", "W3", "b3"]]):
            p.data.copy_(s)

    with torch.no_grad():
        v_ho = head.decode(x_all[idx_ho]).numpy()
    mae = float(np.mean(np.abs(v_ho - y[ho])))
    sp = _spearman(v_ho, y[ho])
    print(f"[train] 留出集: {int(ho.sum())} 帧（{len(hold_eps)} 集） "
          f"CE={best_ho:.4f} MAE(V)={mae:.4f} Spearman={sp:.4f}（{ep + 1} epochs）")

    # 分离度验收（有标注的 rollout 集）：V(bad) 应显著低于 V(good)
    sep_note = "无标注帧，跳过分离度验收"
    try:
        sc = Path(args.dataset[-1]) / "meta" / "recap_sidecar.jsonl"
        if sc.is_file():
            rows = [json.loads(l) for l in open(sc, encoding="utf-8") if l.strip()]
            ind = {(int(r["episode_index"]), int(r["frame_index"])):
                   int(r.get("advantage_indicator", 0)) for r in rows}
            di = len(args.dataset) - 1
            with torch.no_grad():
                v_all = head.decode(x_all).numpy()
            bads = [v_all[i] for i, k in enumerate(keys)
                    if int(k[0]) == di and ind.get((int(k[1]), int(k[2])), 0) < 0]
            goods = [v_all[i] for i, k in enumerate(keys)
                     if int(k[0]) == di and ind.get((int(k[1]), int(k[2])), 0) > 0]
            if bads and goods:
                sep_note = (f"V(bad)={np.mean(bads):.4f}({len(bads)}帧) "
                            f"V(good)={np.mean(goods):.4f}({len(goods)}帧) "
                            f"分离度={np.mean(goods) - np.mean(bads):+.4f}（应显著为正）")
    except Exception as e:
        sep_note = f"分离度验收跳过（{e}）"
    print(f"[train] 验收: {sep_note}")

    out_dir = Path(args.out or f"lerobot_data/recap_value_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # 部署侧（camera_pyqt 实时价值判断）所需常数随模型保存：
    # denominator = 各数据集归一化分母的最大值（保守：|A| 估计偏小方向的误差更安全）；
    # fps = 首个数据集的采集帧率（回报 -1/步 的"步"=数据集帧）
    denoms, fps = [], 20
    for ds_path in args.dataset:
        _df, _m = _load_returns(Path(ds_path), args.tag)
        if _m.get("denominator"):
            denoms.append(float(_m["denominator"]))
        try:
            _info = json.load(open(Path(ds_path) / "meta" / "info.json", encoding="utf-8"))
            fps = int(_info.get("fps", fps))
        except Exception:
            pass
    torch.save({"config": {"dim_in": int(feats.shape[1]), "backend": args.backend,
                           "num_bins": NUM_BINS, "v_min": V_MIN, "v_max": V_MAX,
                           "tag": args.tag, "datasets": [str(d) for d in args.dataset],
                           "denominator": max(denoms) if denoms else 1.0,
                           "fps": fps},
                **head.state()}, out_dir / "value_head.pt")
    print(f"[train] 价值头已保存: {out_dir / 'value_head.pt'} "
          f"（denominator={max(denoms) if denoms else 1.0:.1f}, fps={fps}）")


# ---------------------------------------------------------------------------
# Step 3: N 步 TD 优势 + 分位数二值化
# ---------------------------------------------------------------------------
def _load_head(out_dir: Path):
    import torch
    ck = torch.load(out_dir / "value_head.pt", map_location="cpu", weights_only=False)
    head = BinValueHead(ck["config"]["dim_in"])
    for p, n in zip(head.params(), ["ln_g", "ln_b", "W1", "b1", "W2", "b2", "W3", "b3"]):
        p.data.copy_(ck[n])
    return head, ck["config"]


def cmd_advantage(args):
    import torch
    out_dir = Path(args.out or f"lerobot_data/recap_value_{args.tag}")
    head, cfg = _load_head(out_dir)
    backend = cfg.get("backend", "state")

    for ds_path in args.dataset:
        root = Path(ds_path)
        df, rmeta = _load_returns(root, args.tag)
        denom = float(rmeta.get("denominator", 1.0))
        gamma = float(rmeta.get("gamma", 1.0))

        import recap_critic as rc
        from vd_dataset import OfflineDataset
        ds = OfflineDataset(str(root))
        feats, keys = rc.get_features(ds, root, backend, args.pi05_path)
        with torch.no_grad():
            v = head.decode(torch.from_numpy(feats.astype(np.float32))).numpy()
        v_map = {(int(k[0]), int(k[1])): float(vv) for k, vv in zip(keys, v)}

        # 每集 reward 序列
        adv_map: dict[tuple[int, int], tuple[float, float]] = {}
        for ep, gdf in df.groupby("episode_index"):
            ep = int(ep)
            r = gdf.sort_values("frame_index")["reward"].to_numpy(dtype=np.float64)
            L = len(r)
            for t in range(L):
                k_end = min(t + args.lookahead, L)
                r_sum = 0.0
                for k, ti in enumerate(range(t, k_end)):
                    r_sum += (gamma ** k) * r[ti]
                v_next = v_map.get((ep, t + args.lookahead), 0.0) \
                    if t + args.lookahead < L else 0.0
                v_t = v_map.get((ep, t), 0.0)
                a = r_sum / denom + (gamma ** args.lookahead) * v_next - v_t
                adv_map[(ep, t)] = (float(a), v_t)

        # 分位数二值化（top-q% 为正）。
        # π*0.6/RLinf 口径（--expert-positive 默认开）：专家（遥操作 source==teleop）
        # 帧恒为正样本，分位数阈值只在 rollout 帧上计算——专家数据不进分位池
        # （RLinf compute_advantages.py："sft" 数据集强制 advantage 全 True）。
        a_all = np.array([x[0] for x in adv_map.values()], dtype=np.float64)
        expert_keys = set()
        if args.expert_positive and "source" in df.columns:
            em = df["source"].astype(str) == "teleop"
            expert_keys = set(zip(df.loc[em, "episode_index"].astype(int),
                                  df.loc[em, "frame_index"].astype(int)))
        pool = (np.array([av[0] for k, av in adv_map.items() if k not in expert_keys])
                if expert_keys else a_all)
        if len(pool) == 0:
            pool = a_all
        thr = float(np.percentile(pool, (1.0 - args.quantile) * 100.0))

        def _label(key, a):
            if key in expert_keys:
                return True
            return bool(a >= thr)

        n_pos = sum(1 for k, av in adv_map.items() if _label(k, av[0]))

        # 写 parquet 产物
        import pandas as pd
        recs = [(ep, t, round(av[1], 5), round(av[0], 5), _label((ep, t), av[0]))
                for (ep, t), av in sorted(adv_map.items())]
        out_df = pd.DataFrame(recs, columns=["episode_index", "frame_index",
                                             "td_value", "td_advantage", "td_label"])
        out_pq = root / "meta" / f"td_advantage_{args.tag}_N{args.lookahead}_q{int(args.quantile * 100)}.parquet"
        out_df.to_parquet(out_pq, index=False)
        print(f"[adv] {root.name}: N={args.lookahead} q={args.quantile} 阈值={thr:+.5f}，"
              f"正样本 {n_pos}/{len(a_all)}（{n_pos / len(a_all) * 100:.1f}%）"
              f"{'，专家帧恒正 ' + str(len(expert_keys)) + ' 帧，分位池 ' + str(len(pool)) + ' 帧' if expert_keys else ''}"
              f"→ {out_pq.name}")

        # 写回 sidecar（存在时）
        sc_path = root / "meta" / "recap_sidecar.jsonl"
        if sc_path.is_file():
            rows = [json.loads(l) for l in open(sc_path, encoding="utf-8") if l.strip()]
            n_fill = 0
            out_rows = []
            for r in rows:
                r = dict(r)
                key = (int(r["episode_index"]), int(r["frame_index"]))
                av = adv_map.get(key)
                if av is None:
                    out_rows.append(r)
                    continue
                r["td_value"] = round(av[1], 4)
                r["td_advantage"] = round(av[0], 4)
                r["td_label"] = _label(key, av[0])
                if args.fill_unlabeled:
                    labeled = (int(r.get("advantage_indicator", 0)) != 0
                               or int(r.get("is_human_intervention", 0)) != 0)
                    if not labeled:
                        r["advantage_indicator"] = 1 if _label(key, av[0]) else -1
                        r["advantage_score"] = round(float(np.clip(
                            av[0] / (abs(thr) + 1e-9), -1, 1)), 4)
                        n_fill += 1
                out_rows.append(r)
            tmp = sc_path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for r in out_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(sc_path)
            print(f"[adv] {root.name}: sidecar 已更新（td_value/td_advantage/td_label"
                  f"{'，未标注帧补标 ' + str(n_fill) + ' 帧' if args.fill_unlabeled else ''}）")
        else:
            print(f"[adv] {root.name}: 无 sidecar（{sc_path.name}），仅写 parquet；"
                  f"如需接入 RecapWeighter 请先生成训练 sidecar")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ret = sub.add_parser("returns", help="Step 1：逐集计算折扣回报并归一化")
    p_ret.add_argument("--dataset", action="append", required=True,
                       help="数据集根目录（可多次传入）")
    p_ret.add_argument("--cfail", type=float, default=100.0,
                       help="失败集终止惩罚 C_fail（默认 100）")
    p_ret.add_argument("--gamma", type=float, default=1.0, help="折扣因子（默认 1.0）")
    p_ret.add_argument("--tag", default=None, help="输出标签（默认 cfail<C>）")
    p_ret.add_argument("--unrated", default="derive",
                       choices=["derive", "fail", "skip"],
                       help="未评集处理：derive=片段派生(默认)/fail=按失败/skip=跳过")
    p_ret.add_argument("--unrated-fallback", default="skip", choices=["skip", "fail"],
                       help="derive 后仍无法判定时的回退（默认 skip）")
    p_ret.set_defaults(func=cmd_returns)

    p_tr = sub.add_parser("train", help="Step 2：训练 201-bin 分布式价值头")
    p_tr.add_argument("--dataset", action="append", required=True)
    p_tr.add_argument("--tag", default="cfail100", help="returns 的标签")
    p_tr.add_argument("--backend", default="state", choices=["state", "siglip"])
    p_tr.add_argument("--pi05-path", default="./pretrained_models/pi05_base")
    p_tr.add_argument("--epochs", type=int, default=30)
    p_tr.add_argument("--batch", type=int, default=4096)
    p_tr.add_argument("--lr", type=float, default=1e-3)
    p_tr.add_argument("--holdout", type=float, default=0.1)
    p_tr.add_argument("--seed", type=int, default=42)
    p_tr.add_argument("--out", default=None, help="输出目录（默认 lerobot_data/recap_value_<tag>）")
    p_tr.set_defaults(func=cmd_train)

    p_adv = sub.add_parser("advantage", help="Step 3：N 步 TD 优势 + 分位数二值化")
    p_adv.add_argument("--dataset", action="append", required=True)
    p_adv.add_argument("--tag", default="cfail100")
    p_adv.add_argument("--out", default=None, help="价值头目录（同 train --out）")
    p_adv.add_argument("--pi05-path", default="./pretrained_models/pi05_base")
    p_adv.add_argument("--lookahead", type=int, default=10, help="N 步前瞻（默认 10）")
    p_adv.add_argument("--quantile", type=float, default=0.3,
                       help="正样本分位数（默认 0.3=top 30%%，只在 rollout 帧上计算）")
    p_adv.add_argument("--no-expert-positive", dest="expert_positive",
                       action="store_false",
                       help="关闭专家帧恒正（默认开，π*0.6/RLinf 口径：遥操作专家数据全部为正样本）")
    p_adv.add_argument("--fill-unlabeled", action="store_true",
                       help="把未标注帧按 td_label 补 advantage_indicator=±1（人标/干预帧不动）")
    p_adv.set_defaults(func=cmd_advantage)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

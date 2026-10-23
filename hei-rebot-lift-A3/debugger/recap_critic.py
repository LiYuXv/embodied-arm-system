#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recap_critic.py — 最简版价值 critic（π*0.6 P3 路线 A，RECAP_MEM 实施路线.md P3）。

把稀疏人标（episode 成败 + 片段 good/bad/干预）密集化为逐帧价值 V(s)∈[0,1]，
再以 A(t) = V(s_{t+1}) − V(s_t) 自动标注 sidecar 未标注帧（替代 bad-backprop /
terminal-fail 启发式的信用分配；命令 1 的两个开关在 critic 流程下无需开启）。

双特征后端：
  state  仅位姿/时间特征（CPU 秒级，小数据集/冒烟/基线用）
  siglip pi0.5 冻结视觉塔图像嵌入 + state（GPU，需要 pretrained_models/pi05_base；
         特征一次性提取并缓存 npz，重训头/重应用不再重算）

用法（cd debugger；sidecar 必须先生成）：
  python3 recap_critic.py --dataset lerobot_data/ela3_rollout train --backend state
  python3 recap_critic.py --dataset lerobot_data/ela3_rollout train --backend siglip
  python3 recap_critic.py --dataset lerobot_data/ela3_rollout apply
  python3 recap_critic.py --dataset lerobot_data/ela3_rollout train --backend siglip --apply

产物（均在 <dataset>/meta/）：
  critic_feats_<backend>.npz  特征缓存（ep/frame 对齐）
  critic_head.pt              价值头权重 + 配置
  sidecar 追加列：critic_value（全帧 V，审查用）；未标注帧的
  advantage_indicator/advantage_score 按 V 水平（相对本集均值）补写（人标/干预帧不动）

验收口径（P3）：留出集 V(bad) 显著低于 V(good)（V 水平分离度 > 0）；自动标注抽 10% 人工复核。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEBUGGER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEBUGGER_DIR / "visualize_data"))

FEATURE_DIM = {"state": None, "siglip": None}  # 运行时确定


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------
def _iter_episodes_frames(ds):
    """产出 (ep_idx, frames_df)。"""
    for ep in ds.episodes:
        yield int(ep.get("episode_index", 0)), ds.get_episode_frames(
            int(ep.get("episode_index", 0)))


def extract_features_state(ds) -> tuple[np.ndarray, np.ndarray]:
    """state 后端：[x,y,z,rx,ry,rz,gripper] （N, 7）。

    注意：不拼"帧进度"特征——它会令 V(s)≈单调递增，ΔV 恒为正，
    淹没 bad 段价值下降的信号（2026-08-30 实测 bad ΔV=+0.0034 判废）。
    """
    feats, keys = [], []
    for ep_idx, gdf in _iter_episodes_frames(ds):
        n = len(gdf)
        if n == 0:
            continue
        s = np.vstack(gdf["observation.state"].to_numpy()).astype(np.float32)
        feats.append(s)
        keys.append(np.stack([np.full(n, ep_idx), np.arange(n)] , axis=1))
    return np.concatenate(feats), np.concatenate(keys)


def extract_features_siglip(ds, pi05_path: str, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """siglip 后端：pi0.5 冻结视觉塔双相机嵌入(均值) ⊕ state。

    嵌入按 episode 顺序解码视频（顺序读远快于逐帧 seek）。
    整机(3.3B)直接上 GPU 会 OOM（16GB 卡实测），故整机 CPU 载入后只留视觉塔：
    config.device='cpu' → from_pretrained → 取 vision_tower → del 整机 → 塔上 GPU。
    """
    import torch
    import cv2
    import gc
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.configs.policies import PreTrainedConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[critic] 加载 pi0.5 视觉塔（{pi05_path}，整机 CPU 载入→仅保留 SigLIP）...")
    config = PreTrainedConfig.from_pretrained(pi05_path, local_files_only=True)
    config.device = "cpu"  # 强制 CPU 载入，避免整机 3.3B 上 GPU OOM
    policy = PI05Policy.from_pretrained(pi05_path, config=config)
    tower = policy.model.paligemma_with_expert.paligemma.model.vision_tower
    del policy  # 先释放整机（语言模型+动作专家），仅留视觉塔引用
    # rtc_processor↔model 存在循环引用，del 后需 gc 才真正释放整机；否则整机会
    # 被 .to(cuda) 连带搬上 GPU（16G 卡 OOM）
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    tower.to(device).eval()
    emb_dim = getattr(tower.config, "hidden_size", 1152)

    cam_keys = [k for k in ds.features if ds.is_video_feature(k)]
    cam_keys = sorted(cam_keys)[:2] or ["observation.images.cam1"]

    def _emb(frame_bgr):
        x = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return x  # (224,224,3) in [-1,1]

    feats, keys = [], []
    with torch.no_grad():
        for ep_idx, gdf in _iter_episodes_frames(ds):
            n = len(gdf)
            frames = [[] for _ in cam_keys]
            for ci, ck in enumerate(cam_keys):
                path = ds.get_video_path(ep_idx, ck) if hasattr(ds, "get_video_path") else None
                if path is None:
                    # 兜底：逐帧取（慢路径）
                    for fi in range(n):
                        img = ds.get_video_frame(ep_idx, fi, ck)
                        frames[ci].append(_emb(img) if img is not None else np.zeros((224, 224, 3), np.float32))
                    continue
                cap = cv2.VideoCapture(str(path))
                for fi in range(n):
                    ok, img = cap.read()
                    frames[ci].append(_emb(img) if ok else np.zeros((224, 224, 3), np.float32))
                cap.release()
            emb = np.zeros((n, emb_dim), dtype=np.float32)
            for ci in range(len(cam_keys)):
                arr = np.stack(frames[ci])  # (n,224,224,3)
                for i0 in range(0, n, batch):
                    x = torch.from_numpy(arr[i0:i0 + batch]).permute(0, 3, 1, 2).to(device)
                    out = tower(x)
                    # pi05 视觉塔 use_head=False → pooler_output 为 None；对 patch 序列均值池化
                    if getattr(out, "pooler_output", None) is not None:
                        e = out.pooler_output
                    else:
                        e = out.last_hidden_state.mean(dim=1)
                    emb[i0:i0 + batch] += e.float().cpu().numpy()
            emb /= len(cam_keys)
            s = np.vstack(gdf["observation.state"].to_numpy()).astype(np.float32)
            # 不拼帧进度特征（见 extract_features_state 说明）
            feats.append(np.concatenate([emb, s], axis=1))
            keys.append(np.stack([np.full(n, ep_idx), np.arange(n)], axis=1))
            print(f"[critic] ep{ep_idx} 特征完成（{n} 帧）")
    return np.concatenate(feats), np.concatenate(keys)


def get_features(ds, root: Path, backend: str, pi05_path: str | None):
    cache = root / "meta" / f"critic_feats_{backend}.npz"
    if cache.is_file():
        z = np.load(cache)
        print(f"[critic] 命中特征缓存: {cache}（{z['feats'].shape[0]} 帧）")
        return z["feats"], z["keys"]
    if backend == "state":
        feats, keys = extract_features_state(ds)
    else:
        feats, keys = extract_features_siglip(ds, pi05_path or "./pretrained_models/pi05_base")
    np.savez_compressed(cache, feats=feats, keys=keys)
    print(f"[critic] 特征已缓存: {cache}（{feats.shape}）")
    return feats, keys


# ---------------------------------------------------------------------------
# 价值头
# ---------------------------------------------------------------------------
class TinyValueHead:
    """LayerNorm → 256 → 256 → 1（sigmoid）。torch 实现，CPU 可训。"""

    def __init__(self, dim_in: int, dim_hidden: int = 256, seed: int = 0):
        import torch
        rng = np.random.default_rng(seed)
        def init(*shape):
            t = torch.tensor(rng.normal(0, (2 / shape[0]) ** 0.5, shape).astype(np.float32),
                             requires_grad=True)
            return t
        self.W1, self.b1 = init(dim_in, dim_hidden), torch.zeros(dim_hidden, requires_grad=True)
        self.W2, self.b2 = init(dim_hidden, dim_hidden), torch.zeros(dim_hidden, requires_grad=True)
        self.W3, self.b3 = init(dim_hidden, 1), torch.zeros(1, requires_grad=True)
        self.ln_g, self.ln_b = torch.ones(dim_in, requires_grad=True), torch.zeros(dim_in, requires_grad=True)

    def params(self):
        return [self.ln_g, self.ln_b, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def __call__(self, x):
        import torch
        h = torch.nn.functional.layer_norm(x, (x.shape[-1],), self.ln_g, self.ln_b)
        h = torch.torch.relu(h @ self.W1 + self.b1)
        h = torch.relu(h @ self.W2 + self.b2)
        return (h @ self.W3 + self.b3).squeeze(-1)  # logits

    def state(self):
        import torch
        return {n: v.clone() for n, v in zip(
            ["ln_g", "ln_b", "W1", "b1", "W2", "b2", "W3", "b3"], self.params())}


def train_head(feats, keys, sidecar_rows, epochs=30, holdout=0.1, seed=42):
    """监督：片段/干预=强标签(w=1)，episode 成败=弱标签(w=0.3)，其余 mask。

    Returns: (head, holdout_metrics dict)
    """
    import torch
    eps_set = sorted({int(k[0]) for k in keys})
    rng = np.random.default_rng(seed)
    n_hold = max(1, int(len(eps_set) * holdout))
    hold_eps = set(rng.choice(eps_set, n_hold, replace=False).tolist())
    ep_of_frame = keys[:, 0]

    # 逐帧目标（来自 sidecar 平铺标注）
    itv = np.array([r.get("is_human_intervention", 0) for r in sidecar_rows])
    ind = np.array([r.get("advantage_indicator", 0) for r in sidecar_rows])
    succ = np.array([r.get("episode_success", -1) for r in sidecar_rows])
    assert len(ind) == len(feats), f"sidecar {len(ind)} 帧 ≠ 特征 {len(feats)} 帧（先重新生成 sidecar）"

    target = np.zeros(len(feats), dtype=np.float32)
    weight = np.zeros(len(feats), dtype=np.float32)
    strong = (ind > 0) | (itv > 0)
    bad = ind < 0
    target[strong], weight[strong] = 1.0, 1.0
    target[bad], weight[bad] = 0.0, 1.0
    weak_ok = (weight == 0) & (succ >= 0)
    target[weak_ok], weight[weak_ok] = succ[weak_ok].astype(np.float32), 0.3

    tr = ~np.isin(ep_of_frame, list(hold_eps))
    ho = ~tr
    hold_mask = ho & (weight > 0)
    if hold_mask.sum() == 0:
        raise RuntimeError("留出集无任何监督帧（数据太少）——调低 --holdout")

    head = TinyValueHead(feats.shape[1], seed=seed)
    opt = torch.optim.AdamW(head.params(), lr=1e-3, weight_decay=1e-4)
    x_all = torch.from_numpy(feats)
    t_all = torch.from_numpy(target)
    w_all = torch.from_numpy(weight)
    tr_w = torch.from_numpy(weight > 0) & torch.from_numpy(tr)

    best_ho, best_state, bad_epochs = 1e9, None, 0
    idx_tr = torch.where(tr_w)[0]
    for ep in range(epochs):
        perm = idx_tr[torch.randperm(len(idx_tr))]
        for i0 in range(0, len(perm), 4096):
            b = perm[i0:i0 + 4096]
            logits = head(x_all[b])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, t_all[b], weight=w_all[b])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            hl = torch.nn.functional.binary_cross_entropy_with_logits(
                head(x_all[hold_mask]), t_all[hold_mask]).item()
        if hl < best_ho - 1e-4:
            best_ho, best_state, bad_epochs = hl, head.state(), 0
        else:
            bad_epochs += 1
            if bad_epochs >= 5:
                break
    if best_state:
        for p, s in zip(head.params(), [best_state[n] for n in
                                        ["ln_g","ln_b","W1","b1","W2","b2","W3","b3"]]):
            p.data.copy_(s)

    with torch.no_grad():
        v = torch.sigmoid(head(x_all[hold_mask])).numpy()
        y = t_all[hold_mask].numpy()
    auc = _auc(v, y)
    return head, {"holdout_eps": sorted(int(e) for e in hold_eps),
                  "holdout_frames": int(hold_mask.sum()),
                  "holdout_loss": round(best_ho, 4), "holdout_auc": round(auc, 4),
                  "epochs_trained": ep + 1}


def _auc(scores, labels):
    pos = scores[labels > 0.5]; neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# 应用：V 水平写回 sidecar
# ---------------------------------------------------------------------------
def apply_critic(head, feats, keys, sidecar_path: Path, threshold: float = 0.02):
    """写回 critic_value + 未标注帧按 V 水平相对本集基线补标。

    原 ΔV 口径（V(t+1)−V(t)）方向测反（见 bad_segment_dv 说明），改用 V 水平：
    未标注帧的 V 高于本集均值 → 补正(+1)，低于本集均值超 threshold → 补负(−1)。
    """
    import torch
    with torch.no_grad():
        v = torch.sigmoid(head(torch.from_numpy(feats))).numpy()

    rows = []
    with open(sidecar_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    n_new_pos = n_new_neg = n_untouched = 0
    out = []
    v_by_key = {(int(k[0]), int(k[1])): float(vv) for k, vv in zip(keys, v)}
    # 每集 V 均值作为基线
    ep_vals: dict[int, list[float]] = {}
    for (ek, _), vv in v_by_key.items():
        ep_vals.setdefault(ek, []).append(vv)
    ep_mean = {ek: float(np.mean(vs)) for ek, vs in ep_vals.items()}

    for r in rows:
        r = dict(r)
        ek, fk = int(r["episode_index"]), int(r["frame_index"])
        vv = v_by_key.get((ek, fk))
        if vv is None:
            out.append(r); continue
        r["critic_value"] = round(vv, 4)
        base = ep_mean.get(ek, vv)
        adv = float(vv - base)  # V 水平优势：高于本集均值=好，低于=坏
        labeled = (int(r.get("advantage_indicator", 0)) != 0
                   or int(r.get("is_human_intervention", 0)) != 0)
        if labeled:
            n_untouched += 1
        elif abs(adv) >= threshold:
            r["advantage_indicator"] = 1 if adv > 0 else -1
            r["advantage_score"] = round(float(np.clip(adv, -1, 1)), 4)
            if adv > 0: n_new_pos += 1
            else: n_new_neg += 1
        out.append(r)

    tmp = sidecar_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(sidecar_path)
    return {"new_positive": n_new_pos, "new_negative": n_new_neg,
            "labeled_untouched": n_untouched, "total": len(rows)}


def bad_segment_dv(head, feats, keys, sidecar_rows):
    """验收口径：V 水平分离度——bad 帧 V 应显著低于 good 帧。

    原 ΔV 口径（对 bad 帧求 V(t+1)−V(t) 均值）方向测反：捕获的是坏段结束的回升(+)
    而非进入坏段的下降(−)，导致合格 critic 被误判为失败。改用 V 水平：
    mean V(bad) 应 < mean V(good)（分离度 > 0）。
    Returns: (bad_mean, bad_n, good_mean, good_n, separation)
    """
    import torch
    with torch.no_grad():
        v = torch.sigmoid(head(torch.from_numpy(feats))).numpy().astype(np.float64)
    vmap = {(int(k[0]), int(k[1])): vv for k, vv in zip(keys, v)}
    bad_v, good_v = [], []
    for r in sidecar_rows:
        ek, fk = int(r["episode_index"]), int(r["frame_index"])
        vv = vmap.get((ek, fk))
        if vv is None:
            continue
        ind = int(r.get("advantage_indicator", 0))
        if ind < 0:
            bad_v.append(vv)
        elif ind > 0:
            good_v.append(vv)
    bad_mean = float(np.mean(bad_v)) if bad_v else float("nan")
    good_mean = float(np.mean(good_v)) if good_v else float("nan")
    separation = good_mean - bad_mean if bad_v and good_v else float("nan")
    return bad_mean, len(bad_v), good_mean, len(good_v), separation


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="最简版价值 critic（π*0.6 P3 路线 A）")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("action", choices=["train", "apply"], help="train=训价值头；apply=写回 sidecar")
    ap.add_argument("--backend", default="state", choices=["state", "siglip"])
    ap.add_argument("--pi05-path", default="./pretrained_models/pi05_base")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.02, help="V 水平补标阈值（|V−本集均值| 超此值才补标）")
    ap.add_argument("--apply", action="store_true", help="train 完成后立即应用")
    args = ap.parse_args()

    root = Path(args.dataset)
    sc_path = root / "meta" / "recap_sidecar.jsonl"
    if not sc_path.is_file():
        raise SystemExit(f"缺少 {sc_path}——先运行 make_recap_sidecar.py 生成")
    rows = [json.loads(l) for l in open(sc_path, encoding="utf-8") if l.strip()]

    head_path = root / "meta" / "critic_head.pt"
    if args.action == "apply" and not head_path.is_file():
        raise SystemExit("缺少 critic_head.pt——先 train")

    from vd_dataset import OfflineDataset
    ds = OfflineDataset(str(root))

    if args.action == "train":
        feats, keys = get_features(ds, root, args.backend, args.pi05_path)
        head, metrics = train_head(feats, keys, rows, epochs=args.epochs,
                                   holdout=args.holdout)
        (bad_m, bad_n, good_m, good_n, sep) = bad_segment_dv(head, feats, keys, rows)
        print(f"[critic] 留出集: {metrics['holdout_frames']} 帧 AUC={metrics['holdout_auc']} "
              f"loss={metrics['holdout_loss']}（{metrics['epochs_trained']} epochs，"
              f"留出集 {metrics['holdout_eps'][:6]}{'…' if len(metrics['holdout_eps'])>6 else ''}）")
        print(f"[critic] 验收: V(bad)={bad_m:.4f}（{bad_n} 帧） V(good)={good_m:.4f}（{good_n} 帧） "
              f"分离度={sep:+.4f}（应显著为正：bad 帧价值更低）")
        import torch
        torch.save({"config": {"dim_in": int(feats.shape[1]), "backend": args.backend},
                    **head.state()}, head_path)
        print(f"[critic] 价值头已保存: {head_path}")
        if not args.apply:
            return
        # fallthrough 到 apply
    else:
        feats, keys = get_features(ds, root, args.backend, args.pi05_path)

    import torch
    ck = torch.load(head_path, map_location="cpu", weights_only=False)
    head = TinyValueHead(ck["config"]["dim_in"])
    for p, n in zip(head.params(), ["ln_g", "ln_b", "W1", "b1", "W2", "b2", "W3", "b3"]):
        p.data.copy_(ck[n])
    stats = apply_critic(head, feats, keys, sc_path, threshold=args.threshold)
    print(f"[critic] sidecar 已更新（原子替换）: 补正优势 {stats['new_positive']} 帧 / "
          f"补负优势 {stats['new_negative']} 帧 / 人标与干预帧保留 {stats['labeled_untouched']} / "
          f"共 {stats['total']} 帧")


if __name__ == "__main__":
    main()

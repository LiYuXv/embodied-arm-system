#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recap_sidecar.py — RECAP filtered-BC（P1.2）本地支持模块。

配合 lerobot 0.6.1 内置的 sample_weighting 训练钩子（lerobot_train.py 的
update_policy 已支持逐样本加权：policy.forward(batch, reduction="none") 后
按权重加权求均值）。本模块提供：

  1. load_sidecar()            读取 meta/recap_sidecar.jsonl 为 numpy 数组
  2. compute_frame_weights()   权重映射：good×positive / bad×negative /
                               中性=1.0 / 干预×intervention，clamp [min,max]
  3. gripper_dim_scale()       APO 夹爪维豁免：负样本的夹爪维有效权重恒 1.0
                               （dim_scale=1/negative_weight，抵消样本级降权）
  4. SidecarDataset            CBC 风格包装类：按 (episode_index, frame_index)
                               把 sidecar 字段合并进样本（调试/未来扩展用；
                               lerobot_train 路径不需要——RecapWeighter 直接按
                               batch["index"] 全局帧号查 sidecar，零包装）
  5. CLI 权重分布预览          训练前一键查看加权重分布是否合理

site-packages 侧（重装 lerobot 后需重打，见 TRAIN_README §4）：
  - lerobot/utils/sample_weighting.py   RecapWeighter（type: "recap"）
  - lerobot/policies/pi05/modeling_pi05.py          夹爪维豁免 patch
  - lerobot/policies/smolvla/modeling_smolvla.py    夹爪维豁免 patch

训练 yaml 用法（默认关闭；不开则训练行为与现状逐位一致）：

  sample_weighting:
    type: recap
    extra_params:
      positive_weight: 1.0      # good 帧
      negative_weight: 0.1      # bad 帧（降权而非过滤，APO 建议 0.1~0.3）
      intervention_weight: 2.0  # 人工干预帧（APO 建议 ≤2~3）
      max_weight: 3.0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 权重默认值（RECAP_MEM_实施路线.md P1.2 起步建议）
DEFAULT_POSITIVE_WEIGHT = 1.0
DEFAULT_NEGATIVE_WEIGHT = 0.1
DEFAULT_INTERVENTION_WEIGHT = 2.0
DEFAULT_MIN_WEIGHT = 0.0
DEFAULT_MAX_WEIGHT = 3.0


@dataclass
class RecapWeightConfig:
    """filtered BC 权重配置（与 yaml sample_weighting.extra_params 一一对应）。"""

    positive_weight: float = DEFAULT_POSITIVE_WEIGHT
    negative_weight: float = DEFAULT_NEGATIVE_WEIGHT
    intervention_weight: float = DEFAULT_INTERVENTION_WEIGHT
    min_weight: float = DEFAULT_MIN_WEIGHT
    max_weight: float = DEFAULT_MAX_WEIGHT
    gripper_index: int = -1     # 动作维中的夹爪下标（默认最后一维）

    @classmethod
    def from_extra_params(cls, extra: dict | None) -> "RecapWeightConfig":
        extra = dict(extra or {})
        return cls(
            positive_weight=float(extra.get("positive_weight", DEFAULT_POSITIVE_WEIGHT)),
            negative_weight=float(extra.get("negative_weight", DEFAULT_NEGATIVE_WEIGHT)),
            intervention_weight=float(extra.get("intervention_weight", DEFAULT_INTERVENTION_WEIGHT)),
            min_weight=float(extra.get("min_weight", DEFAULT_MIN_WEIGHT)),
            max_weight=float(extra.get("max_weight", DEFAULT_MAX_WEIGHT)),
            gripper_index=int(extra.get("gripper_index", -1)),
        )


def resolve_sidecar_path(dataset_root_or_path: str | Path) -> Path:
    """接受数据集根目录或 sidecar 文件路径，返回 jsonl 路径。"""
    p = Path(dataset_root_or_path)
    if p.suffix == ".jsonl":
        return p
    return p / "meta" / "recap_sidecar.jsonl"


def load_sidecar(dataset_root_or_path: str | Path) -> dict:
    """读取 sidecar 为 numpy 数组字典。

    Returns:
        {advantage (int8 N), intervention (uint8 N), episode_index (int64 N),
         frame_index (int64 N), n_rows, path}
        行序即数据集全局帧序（make_recap_sidecar 按 episode 升序、帧升序写出，
        与 LeRobotDataset 的 index 列一致），可直接按全局帧号索引。
    """
    path = resolve_sidecar_path(dataset_root_or_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 sidecar: {path}\n"
            f"请先在 visualize_data RECAP 面板点\"生成训练 sidecar\"，或运行 "
            f"python3 make_recap_sidecar.py --dataset <数据集根目录>")
    adv, itv, ep, fr = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            adv.append(int(r.get("advantage_indicator", 0)))
            itv.append(int(r.get("is_human_intervention", 0)))
            ep.append(int(r.get("episode_index", 0)))
            fr.append(int(r.get("frame_index", 0)))
    return {
        "advantage": np.asarray(adv, dtype=np.int8),
        "intervention": np.asarray(itv, dtype=np.uint8),
        "episode_index": np.asarray(ep, dtype=np.int64),
        "frame_index": np.asarray(fr, dtype=np.int64),
        "n_rows": len(adv),
        "path": path,
    }


def compute_frame_weights(
    advantage: np.ndarray,
    intervention: np.ndarray,
    cfg: RecapWeightConfig,
) -> np.ndarray:
    """逐帧权重映射（filtered BC 核心）。

    - advantage>0 → positive_weight（good 帧）
    - advantage<0 → negative_weight（bad/接管前/回传/终端失败帧）
    - advantage==0 → 1.0（中性：未标注帧正常模仿，不被降权——与 CBC 的
      "非正即负"映射不同，本项目以中性为默认，防遗忘混训）
    - is_human_intervention → 再乘 intervention_weight（干预段恒正优势）
    - clamp [min_weight, max_weight]
    """
    adv = np.asarray(advantage)
    w = np.where(adv > 0, cfg.positive_weight,
                 np.where(adv < 0, cfg.negative_weight, 1.0)).astype(np.float64)
    itv = np.asarray(intervention) > 0
    w[itv] *= cfg.intervention_weight
    return np.clip(w, cfg.min_weight, cfg.max_weight).astype(np.float32)


def gripper_dim_scale(
    is_negative: np.ndarray,
    action_dim: int,
    cfg: RecapWeightConfig,
) -> np.ndarray:
    """APO 夹爪维豁免的逐维缩放 (N, action_dim)，全 1 除负样本夹爪维。

    负样本夹爪维 scale = 1/negative_weight：与样本级标量权重相乘后有效权重
    恒为 1.0（接管前后夹爪状态往往相同，惩罚它会教出错误夹爪行为——APO 论文
    明确豁免 gripper token；本项目 filtered BC 用"抵消降权"实现同等效果）。
    调用方（模型 forward patch / ACT 自定义循环）把 losses 乘上本缩放后再做
    样本级加权。
    """
    gi = cfg.gripper_index if cfg.gripper_index >= 0 else action_dim + cfg.gripper_index
    scale = np.ones((len(is_negative), action_dim), dtype=np.float32)
    neg = np.asarray(is_negative, dtype=bool)
    if neg.any():
        scale[neg, gi] = 1.0 / max(cfg.negative_weight, 1e-6)
    return scale


class SidecarDataset:
    """CBC 风格 sidecar 包装类：按 (episode_index, frame_index) 合并字段进样本。

    lerobot_train（pi0.5/SmolVLA）路径无需本类——site-packages 的 RecapWeighter
    直接按 batch["index"] 全局帧号查 sidecar。保留本类用于：
    - 自定义训练循环（把 recap_sample_weight / recap_dim_scale 合并进 batch）
    - 离线调试（检查某几帧的 sidecar 字段与权重）
    """

    def __init__(self, dataset, dataset_root, cfg: RecapWeightConfig | None = None,
                 action_dim: int = 7):
        self._dataset = dataset
        self._cfg = cfg or RecapWeightConfig()
        sc = load_sidecar(dataset_root)
        self._lookup = {
            (int(e), int(f)): i
            for i, (e, f) in enumerate(zip(sc["episode_index"], sc["frame_index"]))
        }
        self._adv = sc["advantage"]
        self._itv = sc["intervention"]
        self._weights = compute_frame_weights(sc["advantage"], sc["intervention"], self._cfg)
        self._dim_scale = gripper_dim_scale(sc["advantage"] < 0, action_dim, self._cfg)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        item = dict(self._dataset[index])
        ep = int(item.get("episode_index", -1))
        fr = int(item.get("frame_index", item.get("index", index)))
        i = self._lookup.get((ep, fr))
        if i is not None:
            item["advantage_indicator"] = int(self._adv[i])
            item["is_human_intervention"] = int(self._itv[i])
            item["recap_sample_weight"] = float(self._weights[i])
            item["recap_dim_scale"] = self._dim_scale[i].copy()
        else:
            item["advantage_indicator"] = 0
            item["is_human_intervention"] = 0
            item["recap_sample_weight"] = 1.0
            item["recap_dim_scale"] = np.ones(self._dim_scale.shape[1], dtype=np.float32)
        return item


def preview(dataset_root: str | Path, cfg: RecapWeightConfig) -> str:
    """生成权重分布预览文本（CLI 与训练前检查共用）。"""
    sc = load_sidecar(dataset_root)
    w = compute_frame_weights(sc["advantage"], sc["intervention"], cfg)
    adv, itv = sc["advantage"], sc["intervention"]
    n = sc["n_rows"]
    lines = [
        f"数据集: {sc['path'].parent.parent.name}  总帧数: {n}",
        f"标注分布: good(adv>0)={int((adv > 0).sum())}  bad(adv<0)={int((adv < 0).sum())}"
        f"  中性/未标注={int((adv == 0).sum())}  干预帧={int((itv > 0).sum())}",
        f"权重配置: positive={cfg.positive_weight} negative={cfg.negative_weight} "
        f"intervention={cfg.intervention_weight} clamp=[{cfg.min_weight},{cfg.max_weight}]",
        f"权重分布: mean={w.mean():.3f} min={w.min():.3f} max={w.max():.3f}"
        f"（均值≠1 表示有效样本量≈{w.sum() / n * 100:.1f}% of {n}）",
        f"  w={cfg.negative_weight}: {int((adv < 0).sum() + 0)} 帧（bad 段+回传+终端失败）",
        f"  w=1.0: {int(((adv == 0) & ~itv.astype(bool)).sum())} 帧（中性/未标注正常模仿）",
        f"  w={cfg.positive_weight}: {int(((adv > 0) & ~(itv.astype(bool))).sum())} 帧（good 非干预）",
        f"  w×intervention: {int((itv > 0).sum())} 帧（干预段，有效权重 {cfg.intervention_weight:.1f}）",
        f"夹爪维豁免: 负样本夹爪维有效权重恒 1.0（scale=1/{cfg.negative_weight}），"
        f"其余维 {cfg.negative_weight}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="RECAP filtered-BC 权重分布预览（P1.2，训练前检查）")
    ap.add_argument("--dataset", required=True, help="数据集根目录（含 meta/recap_sidecar.jsonl）")
    ap.add_argument("--positive-weight", type=float, default=DEFAULT_POSITIVE_WEIGHT)
    ap.add_argument("--negative-weight", type=float, default=DEFAULT_NEGATIVE_WEIGHT)
    ap.add_argument("--intervention-weight", type=float, default=DEFAULT_INTERVENTION_WEIGHT)
    ap.add_argument("--min-weight", type=float, default=DEFAULT_MIN_WEIGHT)
    ap.add_argument("--max-weight", type=float, default=DEFAULT_MAX_WEIGHT)
    args = ap.parse_args()
    cfg = RecapWeightConfig(
        positive_weight=args.positive_weight,
        negative_weight=args.negative_weight,
        intervention_weight=args.intervention_weight,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )
    print(preview(args.dataset, cfg))


if __name__ == "__main__":
    main()

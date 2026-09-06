#!/usr/bin/env python3
"""
将区间标注展开为逐帧 advantage_indicator 列，写入 LeRobot 数据集。

RECAP 映射:
    good    -> 1  (I_t = True,  正优势: 这步动作比平均好)
    bad     -> -1 (I_t = False, 负优势: 这步动作比平均差)
    neutral -> 0
    未标注   -> 0  (留给价值函数推断)

两种模式:
    --mode inject  : 直接在原数据集 parquet 文件中增加 advantage_indicator 列
                     (会修改原始 parquet，建议先备份)
    --mode copy    : 复制数据集到新目录，在副本上添加列 (安全)

同时更新 meta/info.json 的 features，注册 advantage_indicator 特征，
使 LeRobot / SmolVLA 训练管线能自动加载该列。

用法:
    # 预览（不写入）
    python3 expand_annotations.py --root ./lerobot_data/ela3_demo2 --dry-run

    # 安全模式：复制到 ela3_demo2_annotated
    python3 expand_annotations.py --root ./lerobot_data/ela3_demo2 --mode copy

    # 直接注入（修改原文件）
    python3 expand_annotations.py --root ./lerobot_data/ela3_demo2 --mode inject
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# 确保能导入 annotation_store
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from annotation_store import AnnotationStore, LABEL_ADVANTAGE, ALL_LABELS


ADVANTAGE_FEATURE_NAME = "advantage_indicator"


def build_per_frame_array(dataset_root: Path, store: AnnotationStore) -> pd.DataFrame:
    """根据标注，构建 episode_index -> per-frame indicator 的映射。

    返回 DataFrame: columns=[index(全局), advantage_indicator]
    """
    meta_dir = dataset_root / "meta"
    with open(meta_dir / "info.json") as f:
        info = json.load(f)

    data_dir = dataset_root / "data"
    data_files = sorted(data_dir.rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"找不到数据文件: {data_dir}")

    # 读取全部数据获取 episode_index / frame_index / 全局 index
    frames_df = pd.concat(
        [pd.read_parquet(p, columns=["episode_index", "frame_index", "index"])
         for p in data_files],
        ignore_index=True,
    )

    # 初始化为 0
    indicators = np.zeros(len(frames_df), dtype=np.int8)

    # 遍历标注的区间
    episodes_anno = store.data.get("episodes", {})
    filled = 0
    for ep_key, intervals in episodes_anno.items():
        try:
            ep_idx = int(ep_key)
        except ValueError:
            continue
        mask_ep = frames_df["episode_index"].values == ep_idx
        for iv in intervals:
            start = int(iv["start"])
            end = int(iv["end"])
            label = iv.get("label", "neutral")
            adv = LABEL_ADVANTAGE.get(label, 0)
            # frame_index 在 episode 内
            local_mask = mask_ep & (
                (frames_df["frame_index"].values >= start) &
                (frames_df["frame_index"].values <= end))
            indicators[local_mask] = adv
            filled += int(local_mask.sum())

    frames_df[ADVANTAGE_FEATURE_NAME] = indicators
    print(f"[expand] 已填充 {filled}/{len(frames_df)} 帧 "
          f"({100*filled/len(frames_df):.1f}%)")
    print(f"[expand] 标签分布: "
          f"{dict(zip(*np.unique(indicators, return_counts=True)))}")
    return frames_df[["index", ADVANTAGE_FEATURE_NAME]]


def update_info_json(meta_dir: Path) -> bool:
    """在 info.json 的 features 中注册 advantage_indicator 特征。"""
    info_path = meta_dir / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    features = info.setdefault("features", {})
    if ADVANTAGE_FEATURE_NAME in features:
        print(f"[expand] info.json 已含 {ADVANTAGE_FEATURE_NAME}，跳过")
        return False

    features[ADVANTAGE_FEATURE_NAME] = {
        "dtype": "int8",
        "shape": [1],
        "names": None,
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    print(f"[expand] 已在 info.json 注册 {ADVANTAGE_FEATURE_NAME}")
    return True


def inject_column(dataset_root: Path, indicator_df: pd.DataFrame) -> None:
    """逐个 parquet 文件注入 advantage_indicator 列。"""
    data_dir = dataset_root / "data"
    data_files = sorted(data_dir.rglob("*.parquet"))

    # 按全局 index 建立快速查找
    lookup = indicator_df.set_index("index")[ADVANTAGE_FEATURE_NAME]

    for pf in data_files:
        table = pq.read_table(pf)
        df = table.to_pandas()
        global_indices = df["index"].values
        col = lookup.reindex(global_indices).fillna(0).astype(np.int8).values
        df[ADVANTAGE_FEATURE_NAME] = col
        # 写回
        out_table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(out_table, pf)
    print(f"[expand] 已注入列到 {len(data_files)} 个 parquet 文件")


def copy_dataset(src: Path, dst: Path) -> None:
    """复制整个数据集目录。"""
    if dst.exists():
        raise FileExistsError(f"目标已存在: {dst}")
    shutil.copytree(src, dst)
    print(f"[expand] 已复制 {src.name} -> {dst.name}")


def main():
    ap = argparse.ArgumentParser(description="展开区间标注为逐帧 advantage_indicator")
    ap.add_argument("--root", required=True, help="数据集根目录")
    ap.add_argument("--mode", choices=["inject", "copy"], default="copy",
                    help="inject=修改原文件; copy=复制到新目录 (默认)")
    ap.add_argument("--dry-run", action="store_true",
                    help="只预览，不写入")
    args = ap.parse_args()

    src_root = Path(args.root).resolve()
    if not (src_root / "meta" / "info.json").exists():
        print(f"错误: 不是有效数据集: {src_root}")
        sys.exit(1)

    store = AnnotationStore(str(src_root))
    store.load()

    anno_count = store.stats()["_intervals"]
    if anno_count == 0:
        print(f"警告: {src_root}/meta/reward_annotations.json 中没有任何标注区间。")
        print("       将全部填充为 0。建议先用 label_rewards_qt.py 标注。")

    target_root = src_root
    if args.mode == "copy":
        target_root = src_root.parent / f"{src_root.name}_annotated"
        if not args.dry_run:
            copy_dataset(src_root, target_root)
        else:
            print(f"[dry-run] 将复制到 {target_root}")

    if args.dry_run:
        print("\n=== 预览模式 ===")
        indicator_df = build_per_frame_array(src_root, store)
        print(f"\n[dry-run] 模式={args.mode} 不会写入任何文件")
        return

    # 构建逐帧 indicator
    indicator_df = build_per_frame_array(target_root if args.mode == "copy" else src_root, store)

    # 更新 info.json
    update_info_json(target_root / "meta")

    # 注入列
    inject_column(target_root, indicator_df)

    print(f"\n✓ 完成! 数据集: {target_root}")
    print(f"  现在可以用该数据集训练，advantage_indicator 作为额外输入特征。")


if __name__ == "__main__":
    main()

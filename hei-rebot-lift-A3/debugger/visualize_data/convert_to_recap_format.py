#!/usr/bin/env python3
"""
convert_to_recap_format.py — 把旧数据集转换为带 RECAP 标注的新格式
（对齐 ela3_demo3/ela3_rollout 等 2026-08-29 后录制的数据集形态）。

旧数据集只缺 RECAP 元数据，**parquet 数据本身格式完全相同**，本脚本
不改任何数据文件，只做两件事：
  1. 补建 meta/reward_annotations.json：为每个 episode 填默认标注
     （source / episode_success / reviewed，intervals 留空 = 无干预）；
  2. 可选重新生成 meta/recap_sidecar.jsonl（--generate-sidecar），
     替换旧的全零 sidecar。

填充规则（只补缺、不覆盖已有标注）：
  - source          ：--source（默认 teleop；旧遥操作数据集）
  - episode_success ：--success（默认 true=成功；none=不评，留给片段派生）
  - reviewed        ：--reviewed 开关（默认不勾）
  - 已有 intervals / 成败 / source 的 episode 完全不动（如部分已人工标注）。

用法：
  # 预览（不写文件）
  python3 convert_to_recap_format.py --dataset ../lerobot_data/ela3_demo --dry-run
  # 遥操作示范数据集：全部默认成功 + source=teleop + 生成 sidecar
  python3 convert_to_recap_format.py --dataset ../lerobot_data/ela3_demo --generate-sidecar
  # 只补 source 不评成败（成败留给片段派生）
  python3 convert_to_recap_format.py --dataset ../lerobot_data/ela3_demo \
      --success none --generate-sidecar

说明：
  - reward_annotations.json 写入前自动留时间戳备份（AnnotationStore.save）；
  - 旧 sidecar 会被 --generate-sidecar 覆盖（原子替换），数据文件零改动；
  - 转换后可用 visualize_data 打开逐集精标（片段级 good/bad/干预），
    也可用"数据集检查"问题8 验证一致性。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotation_store import AnnotationStore  # noqa: E402


def _episode_count(dataset_root: Path) -> int:
    """从 meta/episodes 统计 episode 数（v3 parquet；失败时报错退出）。"""
    import pyarrow.parquet as pq
    ep_dir = dataset_root / "meta" / "episodes"
    files = sorted(ep_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"找不到 meta/episodes parquet: {ep_dir}")
    return sum(pq.read_metadata(f).num_rows for f in files)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path,
                    help="数据集根目录（如 ../lerobot_data/ela3_demo）")
    ap.add_argument("--source", default="teleop", choices=["teleop", "rollout"],
                    help="数据来源溯源字段（默认 teleop=遥操作示范）")
    ap.add_argument("--success", default="true", choices=["true", "false", "none"],
                    help="默认 episode 成败（默认 true=示范 presumed 成功；none=不评）")
    ap.add_argument("--reviewed", action="store_true",
                    help="同时标记 reviewed=True（默认不标，供人工复核流程）")
    ap.add_argument("--generate-sidecar", action="store_true",
                    help="补标注后重新生成 meta/recap_sidecar.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将填充的内容，不写文件")
    args = ap.parse_args()

    root = args.dataset.resolve()
    n_eps = _episode_count(root)
    print(f"[convert] 数据集: {root.name}  episodes={n_eps}")

    store = AnnotationStore(str(root))
    store.load()
    succ = {"true": True, "false": False, "none": None}[args.success]

    n_new = n_fill_source = n_fill_succ = n_kept = 0
    for ep in range(n_eps):
        entry = store._get_episode(ep)  # 迁移到 v2.0 并取引用
        has_existing = bool(entry["intervals"]) or entry.get("episode_success") is not None
        if has_existing:
            n_kept += 1
            # 只补缺失字段（不覆盖已有值）
            if entry.get("source") is None:
                entry["source"] = args.source
                n_fill_source += 1
            continue
        n_new += 1
        entry["source"] = args.source
        entry["episode_success"] = succ
        if args.reviewed:
            entry["reviewed"] = True

    print(f"[convert] 新建标注: {n_new} 集（source={args.source}, "
          f"episode_success={succ}, reviewed={args.reviewed}）")
    if n_kept:
        print(f"[convert] 已有标注保留: {n_kept} 集"
              + (f"（其中 {n_fill_source} 集补填缺失的 source）" if n_fill_source else ""))
    if not args.dry_run:
        store.save()
        print(f"[convert] 已写入 {store.filepath}（旧文件自动备份为 .bak）")
    else:
        print("[convert] dry-run 完成，未写任何文件")

    # 概览（与总信息面板同源）
    try:
        import pandas as pd
        frames = pd.concat(
            [pd.read_parquet(p, columns=["episode_index", "frame_index"])
             for p in sorted((root / "data").rglob("*.parquet"))], ignore_index=True)
        from make_recap_sidecar import summarize_from_frames
        for line in summarize_from_frames(store, frames, dataset_name=root.name):
            print(f"[convert] {line}")
    except Exception as e:
        print(f"[convert] 概览统计跳过: {e}")

    if args.generate_sidecar:
        if args.dry_run:
            print("[convert] dry-run：跳过 sidecar 生成")
        else:
            from make_recap_sidecar import expand_dataset
            result = expand_dataset(root)
            for line in result["summary_lines"]:
                print(f"[sidecar] {line}")
    elif not args.dry_run:
        stale = (root / "meta" / "recap_sidecar.jsonl").exists()
        if stale:
            print("[convert] 注意: 已存在旧 sidecar（本次未重新生成）；"
                  "标注已更新，训练前请加 --generate-sidecar 或在 RECAP 面板重新生成")


if __name__ == "__main__":
    main()

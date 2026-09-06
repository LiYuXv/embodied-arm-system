#!/usr/bin/env python
"""tasks.parquet 健康检查与孤儿任务清理工具（LeRobot v3 数据集，纯 pandas 依赖）。

检查项（对照 数据问题分析.md 问题7 与孤儿任务问题）：
  1. tasks.parquet 存在且可读
  2. info.json total_tasks 与实际任务数一致
  3. task_index 0 基连续（== range(n)，留空洞会导致 lerobot iloc 越界）
  4. data/meta 引用的 task_index 均存在（无越界引用）
  5. 孤儿任务：删除引用它们的 episode 后残留、0 帧引用的任务行
  6. episodes 的 tasks 名都存在于 tasks.parquet

用法:
    python check_tasks_parquet.py                          # 扫描 ./lerobot_data 下所有数据集
    python check_tasks_parquet.py lerobot_data/ela3_demo   # 只检查指定数据集
    python check_tasks_parquet.py lerobot_data/ela3_demo --fix   # 清理孤儿任务并压实编号（自动备份）

退出码: 0 = 全部健康；1 = 存在问题（或出错）。
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def _is_dataset_dir(p: Path) -> bool:
    return p.is_dir() and (p / "meta" / "info.json").exists()


def _iter_datasets(root: Path):
    if _is_dataset_dir(root):
        yield root
        return
    for sub in sorted(root.iterdir()):
        if _is_dataset_dir(sub):
            yield sub


def _load_tasks(root: Path):
    """返回 (tasks_df 或 None, error_msg 或 None)。"""
    tp = root / "meta" / "tasks.parquet"
    if not tp.exists():
        return None, "tasks.parquet 不存在"
    try:
        return pd.read_parquet(tp), None
    except Exception as e:
        return None, f"tasks.parquet 读取失败: {e}"


def _frame_task_counts(root: Path):
    """data 中每个 task_index 的帧数 {task_index: n_frames}。"""
    import pyarrow.parquet as pq
    counts = {}
    for pf in sorted((root / "data").rglob("*.parquet")):
        if "task_index" not in pq.read_schema(pf).names:
            continue
        col = pd.read_parquet(pf, columns=["task_index"])["task_index"]
        for v, n in col.value_counts().items():
            counts[int(v)] = counts.get(int(v), 0) + int(n)
    return counts


def _episode_task_counts(root: Path, tdf: pd.DataFrame):
    """meta/episodes 中每个 task_index 的 episode 数。

    v3.0 的 meta/episodes 无 task_index 列，episode→任务关联通过 tasks 名称列，
    需经 tasks.parquet 的 名称→task_index 映射换算。
    """
    counts = {}
    name2idx = {str(n): int(v) for n, v in zip(tdf.index, tdf["task_index"])}
    ep_dir = root / "meta" / "episodes"
    if not ep_dir.exists():
        return counts
    for pf in sorted(ep_dir.rglob("*.parquet")):
        df = pd.read_parquet(pf)
        if "tasks" not in df.columns:
            continue
        for v in df["tasks"]:
            try:
                names = list(v) if v is not None else []
            except TypeError:
                names = [v]
            for name in names:
                ti = name2idx.get(str(name))
                if ti is not None:
                    counts[ti] = counts.get(ti, 0) + 1
    return counts


def scan_dataset(root: Path) -> dict:
    root = Path(root)
    rep = {"root": root, "ok": True, "problems": [], "tasks": [],
           "n_tasks": 0, "info_total_tasks": None}

    tdf, err = _load_tasks(root)
    if err:
        rep["problems"].append(err)
        rep["ok"] = False
        return rep

    rep["n_tasks"] = len(tdf)
    ti_vals = [int(v) for v in tdf["task_index"]]

    # info.json total_tasks
    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text())
        rep["info_total_tasks"] = info.get("total_tasks")
        if info.get("total_tasks") is not None and int(info["total_tasks"]) != len(tdf):
            rep["problems"].append(
                f"info.json total_tasks={info['total_tasks']} ≠ 实际 {len(tdf)}")
    except Exception as e:
        rep["problems"].append(f"info.json 读取失败: {e}")

    # task_index 0 基连续
    if sorted(ti_vals) != list(range(len(tdf))):
        rep["problems"].append(f"task_index 非 0 基连续: {sorted(ti_vals)}")

    # 引用统计
    frame_counts = _frame_task_counts(root)
    ep_counts = _episode_task_counts(root, tdf)

    out_of_range = sorted(v for v in frame_counts if v < 0 or v >= len(tdf))
    if out_of_range:
        rep["problems"].append(f"data 引用了不存在的 task_index: {out_of_range}")

    # episodes tasks 名存在性
    task_names = {str(n) for n in tdf.index}
    ep_dir = root / "meta" / "episodes"
    bad_names = set()
    if ep_dir.exists():
        for pf in sorted(ep_dir.rglob("*.parquet")):
            df = pd.read_parquet(pf)
            if "tasks" not in df.columns:
                continue
            for v in df["tasks"]:
                try:
                    names = list(v) if v is not None else []
                except TypeError:
                    names = [v]
                for name in names:
                    if str(name) not in task_names:
                        bad_names.add(str(name))
    if bad_names:
        rep["problems"].append(f"episodes 引用了不存在的任务名: {sorted(bad_names)}")

    for name, row in tdf.iterrows():
        ti = int(row["task_index"])
        rep["tasks"].append({
            "name": str(name), "task_index": ti,
            "frames": frame_counts.get(ti, 0), "episodes": ep_counts.get(ti, 0),
        })
    rep["tasks"].sort(key=lambda d: d["task_index"])

    orphans = [t["name"] for t in rep["tasks"] if t["frames"] == 0]
    if orphans:
        rep["problems"].append(f"孤儿任务（0 帧引用）: {', '.join(orphans)}")

    rep["ok"] = not rep["problems"]
    return rep


def print_report(rep: dict):
    status = "✓ 健康" if rep["ok"] else "✗ 存在问题"
    print(f"\n{'='*64}\n数据集: {rep['root']}\ntasks.parquet: {rep['n_tasks']} 个任务 | "
          f"info.json total_tasks={rep['info_total_tasks']} | {status}")
    for t in rep["tasks"]:
        mark = " ⚠ 孤儿" if t["frames"] == 0 else ""
        print(f"  [{t['task_index']}] {t['name']:<40} {t['frames']:>6} 帧, "
              f"{t['episodes']:>3} episodes{mark}")
    for p in rep["problems"]:
        print(f"  ✗ {p}")


def _atomic_write_df(df: pd.DataFrame, path: Path, index: bool):
    tmp = Path(str(path) + ".tmp.parquet")
    df.to_parquet(tmp, index=index)
    os.replace(tmp, path)


def fix_dataset(root: Path, backup: bool = True) -> bool:
    """清理孤儿任务并压实 task_index（0 基连续），同步平移 data/meta/episodes/
    stats.json 与 info.json total_tasks。返回是否执行了修改。

    仅自动修复"孤儿任务/编号空洞"；data 引用不存在编号时拒绝修复（语义不明）。
    """
    root = Path(root)
    rep = scan_dataset(root)
    has_orphan = any(t["frames"] == 0 for t in rep["tasks"])
    has_holes = sorted(t["task_index"] for t in rep["tasks"]) != list(range(rep["n_tasks"]))
    if not has_orphan and not has_holes:
        if rep["ok"]:
            print("  无需修复")
            return False
        print("  ✗ 存在无法自动修复的问题（如引用越界/名称缺失），请人工处理")
        return False

    tdf, _ = _load_tasks(root)
    referenced = set(_frame_task_counts(root))
    keep_mask = [int(v) in referenced for v in tdf["task_index"]]
    if not any(keep_mask):
        print("  ✗ 全部任务均无引用，拒绝自动清理（疑似数据损坏），请人工检查")
        return False

    removed = [str(n) for n, k in zip(tdf.index, keep_mask) if not k]
    tasks_new = tdf[keep_mask].sort_values("task_index")
    remap = {int(old): new for new, old in enumerate(tasks_new["task_index"])}
    if all(old == new for old, new in remap.items()) and not removed:
        print("  无需修复")
        return False
    tasks_new = tasks_new.copy()
    tasks_new["task_index"] = [remap[int(v)] for v in tasks_new["task_index"]]

    # ---- 备份将被修改的文件 ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = root / f".bak_tasks_fix_{ts}"

    def _bak(p: Path):
        if not backup:
            return
        dst = bak / p.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)

    print(f"  修复: 删除孤儿任务 {removed}，task_index 压实映射 {remap}"
          + (f"（备份到 {bak.name}/）" if backup else "（未备份！）"))

    # ---- data：逐文件平移 task_index（只重写有变化的文件，原子写）----
    stats_vals = []
    for pf in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(pf)
        if "task_index" not in df.columns:
            continue
        stats_vals.append(df["task_index"].map(lambda v: remap[int(v)]))
        mapped = df["task_index"].map(lambda v: remap[int(v)])
        if bool((mapped == df["task_index"]).all()):
            continue
        _bak(pf)
        df["task_index"] = mapped.astype(df["task_index"].dtype)
        _atomic_write_df(df, pf, index=False)

    # ---- meta/episodes：平移 task_index 列与 per-episode stats/task_index/* ----
    # 注意：v3.0 的 meta/episodes 通常没有 task_index 列，episode→任务关联在
    # tasks 名称列；此时按 名称→新 task_index 映射重算 per-episode 统计列。
    import numpy as np
    name2idx_new = {str(n): int(v) for n, v in zip(tasks_new.index, tasks_new["task_index"])}
    ep_dir = root / "meta" / "episodes"
    if ep_dir.exists():
        for pf in sorted(ep_dir.rglob("*.parquet")):
            df = pd.read_parquet(pf)
            has_ti = "task_index" in df.columns
            has_stats = any(c.startswith("stats/task_index/") for c in df.columns)
            if not has_ti and not has_stats:
                continue
            _bak(pf)
            if has_ti:
                df["task_index"] = df["task_index"].map(lambda v: remap[int(v)])
                ti = df["task_index"].to_numpy(dtype="float64")
            elif "tasks" in df.columns:
                def _row_ti(v):
                    try:
                        names = list(v) if v is not None else []
                    except TypeError:
                        names = [v]
                    return float(name2idx_new.get(str(names[0]), 0)) if names else 0.0
                ti = df["tasks"].map(_row_ti).to_numpy(dtype="float64")
            else:
                ti = None
            if ti is not None:
                for suffix, arr in [("min", ti), ("max", ti), ("mean", ti),
                                    ("std", np.zeros_like(ti)), ("q01", ti),
                                    ("q10", ti), ("q50", ti), ("q90", ti), ("q99", ti)]:
                    col = f"stats/task_index/{suffix}"
                    if col in df.columns:
                        vals = np.empty(len(arr), dtype=object)
                        for i, v in enumerate(arr):
                            vals[i] = np.array([float(v)], dtype=np.float64)
                        df[col] = vals
            _atomic_write_df(df, pf, index=False)

    # ---- tasks.parquet ----
    tp = root / "meta" / "tasks.parquet"
    _bak(tp)
    _atomic_write_df(tasks_new, tp, index=True)

    # ---- info.json total_tasks ----
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        _bak(info_path)
        info = json.loads(info_path.read_text())
        info["total_tasks"] = len(tasks_new)
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))

    # ---- stats.json 的 task_index 项重算 ----
    stats_path = root / "meta" / "stats.json"
    if stats_path.exists() and stats_vals:
        import numpy as np
        _bak(stats_path)
        arr = pd.concat(stats_vals).to_numpy(dtype="float64").reshape(-1, 1)
        stats = json.loads(stats_path.read_text())
        stats["task_index"] = {
            "min": arr.min(axis=0).tolist(), "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q10": np.quantile(arr, 0.10, axis=0).tolist(),
            "q50": np.quantile(arr, 0.50, axis=0).tolist(),
            "q90": np.quantile(arr, 0.90, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
        stats_path.write_text(json.dumps(stats, indent=2))

    # ---- 修复后复检 ----
    rep2 = scan_dataset(root)
    if rep2["ok"]:
        print("  ✓ 修复后复检通过")
    else:
        print("  ✗ 修复后仍有问题:")
        for p in rep2["problems"]:
            print(f"    - {p}")
    return True


def main():
    ap = argparse.ArgumentParser(description="tasks.parquet 健康检查与孤儿任务清理")
    ap.add_argument("root", nargs="?", default="./lerobot_data",
                    help="数据集路径或包含多个数据集的目录（默认 ./lerobot_data）")
    ap.add_argument("--fix", action="store_true",
                    help="清理孤儿任务并压实 task_index（自动备份到 .bak_tasks_fix_<时间戳>/）")
    ap.add_argument("--no-backup", action="store_true", help="--fix 时不备份（不推荐）")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"路径不存在: {root}")
        return 1

    datasets = list(_iter_datasets(root))
    if not datasets:
        print(f"未找到数据集（需含 meta/info.json）: {root}")
        return 1

    all_ok = True
    for ds in datasets:
        if args.fix:
            print(f"\n{'='*64}\n数据集: {ds}")
            fix_dataset(ds, backup=not args.no_backup)
        rep = scan_dataset(ds)
        print_report(rep)
        all_ok = all_ok and rep["ok"]

    print(f"\n{'='*64}\n总计: {len(datasets)} 个数据集，"
          f"{'全部健康 ✓' if all_ok else '存在异常 ✗'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

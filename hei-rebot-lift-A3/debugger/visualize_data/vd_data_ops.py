# -*- coding: utf-8 -*-
"""数据集磁盘操作（纯函数，不依赖 Qt）：episode 删除/批量删除、剪辑、多段剪辑、
轨迹平滑降噪、数据集一致性检查。所有函数直接读写 parquet/jsonl/视频文件。"""
import os
import json
import glob
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path


# ==================== Episode 删除/剪辑/降噪 数据工具函数 ====================
def _read_episodes(meta_dir):
    """读取 episodes 元数据，返回 (DataFrame, format, file_paths)。"""
    ep_parquet_dir = os.path.join(meta_dir, "episodes")
    ep_jsonl_path  = os.path.join(meta_dir, "episodes.jsonl")

    if os.path.exists(ep_parquet_dir):
        ep_files = sorted(glob.glob(os.path.join(ep_parquet_dir, "**", "*.parquet"), recursive=True))
        if ep_files:
            ep_df = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
            return ep_df, "parquet", ep_files
    if os.path.exists(ep_jsonl_path):
        with open(ep_jsonl_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        return pd.DataFrame(rows), "jsonl", [ep_jsonl_path]
    raise FileNotFoundError(f"找不到 episodes 元数据：{meta_dir}")

def _write_parquet_chunks(df, out_dir, chunk_size=1000):
    """将 DataFrame 按 chunk_size 分块写入 parquet（原子写，防中途崩溃留下截断文件）。"""
    os.makedirs(out_dir, exist_ok=True)
    for i, start in enumerate(range(0, max(len(df), 1), chunk_size)):
        _atomic_write_parquet(
            df.iloc[start:start + chunk_size],
            os.path.join(out_dir, f"file-{i:03d}.parquet"))

def _get_dataset_fps(meta_dir):
    """从 info.json 读取数据集 fps，失败时返回 20。"""
    try:
        with open(os.path.join(str(meta_dir), "info.json")) as f:
            return float(json.load(f).get("fps", 20))
    except Exception:
        return 20.0

def _reset_frame_columns(df, fps):
    """按 episode 重置 frame_index/timestamp，使其与 episode 内行序一致。"""
    if "episode_index" not in df.columns:
        return df
    df = df.copy()
    df["frame_index"] = df.groupby("episode_index", sort=False).cumcount()
    if "timestamp" in df.columns:
        df["timestamp"] = (df["frame_index"] / fps).astype(np.float32)
    return df

def _sync_episode_data_pointers(ep_df, df_data, chunk_size=1000):
    """data 重写为 chunk 文件后，重算每个 episode 的 data/chunk_index 与 data/file_index。"""
    if "episode_index" not in df_data.columns:
        return ep_df
    if "data/chunk_index" not in ep_df.columns or "data/file_index" not in ep_df.columns:
        return ep_df
    ep_df = ep_df.copy()
    ep_col = df_data["episode_index"].to_numpy()
    starts = {}
    for i, ep in enumerate(ep_col):
        ep = int(ep)
        if ep not in starts:
            starts[ep] = i
    for i in range(len(ep_df)):
        ep = int(ep_df.loc[i, "episode_index"]) if "episode_index" in ep_df.columns else i
        if ep not in starts:
            raise ValueError(f"data 中找不到 episode {ep} 的数据行")
        ep_df.loc[i, "data/chunk_index"] = 0
        ep_df.loc[i, "data/file_index"] = starts[ep] // chunk_size
    return ep_df

def _atomic_write_parquet(df, path):
    """原子写 parquet：先写同目录临时文件再 os.replace，
    避免写盘中途崩溃/断电留下被截断的 parquet（列读取直接报错）。"""
    import tempfile
    path = str(path)
    dir_name = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".parquet", dir=dir_name)
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise

def check_dataset_consistency(dataset_root):
    """编辑前置闸门：快速校验 meta/episodes ↔ data ↔ info.json 三方一致。

    只读 parquet 的 episode_index 列，秒级完成（不做 ffprobe 逐帧数视频）。
    返回 (ok, problems)：ok=False 时 problems 为问题描述列表。
    """
    problems = []
    root = Path(dataset_root)
    meta_dir = root / "meta"
    data_dir = root / "data"

    data_files = sorted(data_dir.rglob("*.parquet"))
    if not data_files:
        return False, [f"找不到数据 parquet：{data_dir}"]
    counts = {}
    total_rows = 0
    for p in data_files:
        col = pd.read_parquet(p, columns=["episode_index"])["episode_index"]
        total_rows += len(col)
        for eid, n in col.value_counts().items():
            counts[int(eid)] = counts.get(int(eid), 0) + int(n)

    try:
        ep_df, _, _ = _read_episodes(str(meta_dir))
    except Exception as e:
        return False, [f"读取 meta/episodes 失败: {e}"]

    n_meta = len(ep_df)
    if "episode_index" in ep_df.columns:
        meta_eps = [int(e) for e in ep_df["episode_index"].tolist()]
    else:
        meta_eps = list(range(n_meta))
    meta_set = set(meta_eps)

    phantom = [e for e in meta_eps if e not in counts]
    if phantom:
        problems.append(f"meta 有但 data 无数据行的 episode: "
                        f"{phantom[:8]}{' ...' if len(phantom) > 8 else ''}")
    orphan = [e for e in counts if e not in meta_set]
    if orphan:
        problems.append(f"data 有但 meta 缺失的 episode_index: "
                        f"{[int(e) for e in orphan[:8]]}{' ...' if len(orphan) > 8 else ''}")

    if "length" in ep_df.columns:
        len_bad = []
        for pos in range(n_meta):
            e = meta_eps[pos]
            mlen = int(ep_df["length"].iloc[pos])
            if e in counts and mlen != counts[e]:
                len_bad.append(f"ep{e}: meta={mlen} vs data={counts[e]}")
        if len_bad:
            problems.append(f"length 与 data 行数不符（{len(len_bad)} 个）: "
                            + "; ".join(len_bad[:5]) + (" ..." if len(len_bad) > 5 else ""))

    if "dataset_from_index" in ep_df.columns and "dataset_to_index" in ep_df.columns \
            and "length" in ep_df.columns:
        ptr = ep_df[["dataset_from_index", "dataset_to_index", "length"]].to_numpy(dtype=np.int64)
        ptr = ptr[np.argsort(ptr[:, 0])]
        contig = (int(ptr[0, 0]) == 0
                  and bool((ptr[1:, 0] == ptr[:-1, 1]).all())
                  and bool((ptr[:, 1] - ptr[:, 0] == ptr[:, 2]).all()))
        if not contig or int(ptr[-1, 1]) != total_rows:
            problems.append(f"dataset_from/to_index 不连续或与实际帧数不符"
                            f"（末尾 to={int(ptr[-1, 1])}，data 实际 {total_rows} 行）")

    info_path = meta_dir / "info.json"
    if info_path.exists():
        try:
            with open(info_path) as f:
                info = json.load(f)
            if info.get("total_frames") is not None and int(info["total_frames"]) != total_rows:
                problems.append(f"info.json total_frames={info['total_frames']} ≠ data 实际 {total_rows}")
            if info.get("total_episodes") is not None and int(info["total_episodes"]) != n_meta:
                problems.append(f"info.json total_episodes={info['total_episodes']} ≠ meta 实际 {n_meta}")
        except Exception as e:
            problems.append(f"读取 info.json 失败: {e}")

    return (len(problems) == 0), problems

def _assert_dataset_consistent(dataset_root):
    """编辑函数动笔前必须调用：数据集不一致时直接抛错拒绝编辑。

    防止在已损坏的数据集上继续编辑（_sync_episode_data_pointers 只校验
    episode 是否有数据行，不校验 length/区间，带病数据会被静默切错行）。
    """
    ok, problems = check_dataset_consistency(dataset_root)
    if not ok:
        raise ValueError(
            "数据集 meta/data/info 不一致，已拒绝编辑（继续编辑会加重错位，"
            "请先修复，或用\"数据集检查\"查看详情）：\n  - " + "\n  - ".join(problems))

def _gc_orphan_tasks(tasks_path, df_data, ep_df):
    """回收 tasks.parquet 中不再被任何数据行引用的任务（孤儿任务）。

    删除 episode 后，被带走最后一处引用的任务（如 default_task）会残留为
    孤儿行。清理时必须同步把剩余 task_index 压实为 0 基连续，并平移
    data 帧、meta/episodes 的 task_index 列与 per-episode stats/task_index/*
    统计列（lerobot dataset_reader 按 iloc 位置取任务行，留空洞会越界，
    见 数据问题分析.md 问题7）。

    约定：调用方传入的 df_data / ep_df 均为新建副本，本函数原地修改。
    返回 (new_tasks_df 或 None, removed_names)；None 表示无孤儿任务，
    tasks.parquet 无需改写。
    """
    if not os.path.exists(tasks_path) or "task_index" not in df_data.columns:
        return None, []
    tasks_df = pd.read_parquet(tasks_path)
    return _gc_orphan_tasks_df(tasks_df, df_data, ep_df)


def _gc_orphan_tasks_df(tasks_df, df_data, ep_df):
    """_gc_orphan_tasks 的核心：接收内存中的 tasks_df（可含尚未落盘的新任务行）。

    供"单条任务改名"等已持有准备好的 tasks_df 的调用方复用——
    改名产生的孤儿旧任务同样要回收压实，否则留 0 引用孤儿行
    （lerobot iloc 不越界但污染任务表，且总信息/检查报孤儿告警）。
    """
    if tasks_df is None or "task_index" not in df_data.columns:
        return None, []
    referenced = {int(v) for v in df_data["task_index"].unique()}
    keep_mask = [int(v) in referenced for v in tasks_df["task_index"]]
    if all(keep_mask) or not any(keep_mask):
        # 无孤儿；或全部成为孤儿（异常情况，保留现状由上层报错）
        return None, []

    removed = [str(name) for name, keep in zip(tasks_df.index, keep_mask) if not keep]
    tasks_df = tasks_df[keep_mask].sort_values("task_index")
    remap = {int(old): new for new, old in enumerate(tasks_df["task_index"])}
    tasks_df["task_index"] = [remap[int(v)] for v in tasks_df["task_index"]]

    df_data["task_index"] = df_data["task_index"].map(
        lambda v: remap[int(v)]).astype(df_data["task_index"].dtype)

    # meta/episodes 平移：v3.0 通常无 task_index 列（关联在 tasks 名称列），
    # 此时按 名称→新 task_index 换算 per-episode stats/task_index/* 统计列
    if "task_index" in ep_df.columns:
        ep_df["task_index"] = ep_df["task_index"].map(
            lambda v: remap[int(v)]).astype(ep_df["task_index"].dtype)
        ti_vals = ep_df["task_index"].to_numpy(dtype=np.float64)
    elif "tasks" in ep_df.columns:
        name2idx = {str(n): int(v) for n, v in zip(tasks_df.index, tasks_df["task_index"])}

        def _row_ti(v):
            try:
                names = list(v) if v is not None else []
            except TypeError:
                names = [v]
            return float(name2idx.get(str(names[0]), 0)) if names else 0.0

        ti_vals = ep_df["tasks"].map(_row_ti).to_numpy(dtype=np.float64)
    else:
        ti_vals = None

    if ti_vals is not None:
        # per-episode 统计：每个 episode 只有一个任务 → min=max=mean=分位数=值, std=0
        for suffix, arr in [("min", ti_vals), ("max", ti_vals), ("mean", ti_vals),
                            ("std", np.zeros_like(ti_vals)), ("q01", ti_vals),
                            ("q10", ti_vals), ("q50", ti_vals), ("q90", ti_vals),
                            ("q99", ti_vals)]:
            col = f"stats/task_index/{suffix}"
            if col in ep_df.columns:
                vals = np.empty(len(arr), dtype=object)
                for i, v in enumerate(arr):
                    vals[i] = np.array([float(v)], dtype=np.float64)
                ep_df[col] = vals
    return tasks_df, removed

def _recompute_global_stats(meta_dir, df):
    """根据当前 data 内容重算 stats.json 中各特征的聚合统计。

    - parquet 数值列（observation.state/action/timestamp/frame_index 等）：
      全量重算 min/max/mean/std/count/分位数。
    - 视频图像特征（observation.images.cam*，不在 parquet 中）：
      仅把 count 同步为当前实际帧数，像素统计（min/max/mean/std/分位数）保持不变。
      原因：图像特征使用 VISUAL/IDENTITY 归一化（不依赖像素统计），全量重算需解码
      所有视频帧代价过高；但 count 必须与实际帧数一致，否则 stats 一致性校验失败
      （即 数据问题分析.md 问题4）。
    """
    stats_path = os.path.join(str(meta_dir), "stats.json")
    if not os.path.exists(stats_path):
        return
    with open(stats_path) as f:
        stats = json.load(f)
    n_frames = len(df)
    for key in df.columns:
        if key not in stats:
            continue
        try:
            col = df[key]
            sample = col.iloc[0]
            if np.isscalar(sample):
                arr = col.to_numpy(dtype=np.float64).reshape(-1, 1)
            else:
                arr = np.stack(col.to_numpy()).astype(np.float64)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
        except Exception:
            continue
        stats[key] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q10": np.quantile(arr, 0.10, axis=0).tolist(),
            "q50": np.quantile(arr, 0.50, axis=0).tolist(),
            "q90": np.quantile(arr, 0.90, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
    # 视频图像特征不在 parquet 列中：仅同步 count，像素统计保持不变
    for key, entry in stats.items():
        if key in df.columns or not isinstance(entry, dict) or "count" not in entry:
            continue
        old_count = entry.get("count")
        new_count = [int(n_frames)]
        if old_count != new_count:
            entry["count"] = new_count
            print(f"[stats] {key}: count {old_count} → {new_count}（图像特征仅同步计数）")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # 同步 meta/episodes 中每个 episode 的 stats/*/count 为各自当前 length
    _sync_episode_parquet_stats_counts(str(meta_dir))

def _sync_episode_parquet_stats_counts(meta_dir):
    """同步 meta/episodes parquet 中每个 episode 的 stats/*/count 列为各自 length。

    编辑（删除/裁剪/降噪）后，per-episode 的 count 字段会过期（如 state_count 仍为
    旧帧数）。此处统一刷新为当前 length，保持一致性（数据问题分析.md 问题4）。
    """
    ep_files = sorted(glob.glob(os.path.join(meta_dir, "episodes", "**", "*.parquet"), recursive=True))
    if not ep_files:
        return
    for f in ep_files:
        ep_df = pd.read_parquet(f)
        count_cols = [c for c in ep_df.columns if c.startswith("stats/") and c.endswith("/count")]
        if not count_cols or "length" not in ep_df.columns:
            continue
        changed = False
        for idx in ep_df.index:
            length = int(ep_df.at[idx, "length"])
            for col in count_cols:
                if ep_df.at[idx, col] != [length]:
                    ep_df.at[idx, col] = [length]
                    changed = True
        if changed:
            ep_df.to_parquet(f, index=False)
            print(f"[stats] 已同步 {os.path.basename(f)} 中各 episode 的 stats/*/count")

def _update_video_timestamps_after_trim(ep_df, episode_idx, new_ep_len, fps):
    """裁剪 episode 后，将其所有摄像头的 to_timestamp 更新为 from_timestamp + new_len/fps。"""
    ep_df = ep_df.copy()
    for col in [c for c in ep_df.columns if c.startswith("videos/") and c.endswith("/to_timestamp")]:
        from_col = col.replace("to_timestamp", "from_timestamp")
        from_ts = float(ep_df.loc[episode_idx, from_col]) if from_col in ep_df.columns else 0.0
        ep_df.loc[episode_idx, col] = from_ts + new_ep_len / fps
    return ep_df

def _remove_files(file_list):
    """安全删除文件列表。"""
    for p in file_list:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

def _reindex_video_files(video_dir, deleted_idx, num_episodes_after):
    """删除 episode 后，重命名后续视频文件使索引连续。"""
    if not os.path.exists(video_dir):
        return [], []

    vid_pattern = os.path.join(video_dir, "**", "episode_*.mp4")
    vid_files = glob.glob(vid_pattern, recursive=True)

    entries = []
    for vf in vid_files:
        basename = os.path.basename(vf)
        if basename.startswith("episode_") and basename.endswith(".mp4"):
            try:
                ep_num = int(basename[8:14])
                entries.append((vf, os.path.dirname(vf), ep_num, basename))
            except ValueError:
                continue

    from collections import defaultdict
    cam_groups = defaultdict(list)
    for vf, cam_path, ep_num, basename in entries:
        cam_groups[cam_path].append((vf, ep_num, basename))

    deleted = []
    renamed = []

    for cam_path, items in cam_groups.items():
        to_delete = [(vf, ep, bn) for vf, ep, bn in items if ep == deleted_idx]
        to_reindex = [(vf, ep, bn) for vf, ep, bn in items if ep > deleted_idx]

        for vf, _, _ in to_delete:
            try:
                os.remove(vf)
                deleted.append(vf)
            except Exception:
                pass

        # 升序重命名避免覆盖级联：所有文件都向下移动（new = old - 1 < old），
        # 处理到 ep_num 时目标位置必然空闲（更小编号的文件已移走/删除）。
        # （旧代码用降序：ep5→4 会覆盖尚未移动的 ep4 文件，连锁破坏保留集视频）
        to_reindex_sorted = sorted(to_reindex, key=lambda x: x[1])
        for vf, ep_num, bn in to_reindex_sorted:
            new_ep = ep_num - 1
            new_bn = f"episode_{new_ep:06d}.mp4"
            new_vf = os.path.join(cam_path, new_bn)
            try:
                os.rename(vf, new_vf)
                renamed.append((vf, new_vf))
            except Exception:
                pass

    return deleted, renamed

def _update_stats_json(meta_dir, deleted_idx, num_episodes_after):
    """更新 stats.json / stats.jsonl，删除对应 episode 的统计并重新索引。"""
    stats_jsonl = os.path.join(meta_dir, "stats.jsonl")
    if os.path.exists(stats_jsonl):
        with open(stats_jsonl) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        new_rows = []
        for row in rows:
            if row.get("episode_index") == deleted_idx:
                continue
            if row.get("episode_index", -1) > deleted_idx:
                row = dict(row)
                row["episode_index"] -= 1
            new_rows.append(row)
        with open(stats_jsonl, "w") as f:
            for row in new_rows:
                f.write(json.dumps(row) + "\n")

    stats_json = os.path.join(meta_dir, "stats.json")
    if os.path.exists(stats_json):
        with open(stats_json) as f:
            stats = json.load(f)
        if "episodes" in stats:
            ep_stats = stats["episodes"]
            if isinstance(ep_stats, list):
                new_ep_stats = []
                for s in ep_stats:
                    if s.get("episode_index") == deleted_idx:
                        continue
                    if s.get("episode_index", -1) > deleted_idx:
                        s = dict(s)
                        s["episode_index"] -= 1
                    new_ep_stats.append(s)
                stats["episodes"] = new_ep_stats
            elif isinstance(ep_stats, dict):
                new_ep_stats = {}
                for k, v in ep_stats.items():
                    idx = int(k)
                    if idx == deleted_idx:
                        continue
                    if idx > deleted_idx:
                        new_ep_stats[str(idx - 1)] = v
                    else:
                        new_ep_stats[k] = v
                stats["episodes"] = new_ep_stats
        with open(stats_json, "w") as f:
            json.dump(stats, f, indent=2)

def delete_episode(dataset_root, episode_idx, backup_enabled=True):
    """安全删除指定 episode，重新编排所有索引，保证数据一致性。

    动笔前先跑三方一致性闸门（不一致直接抛错拒绝）；
    backup_enabled=True 时生成 data.bak_*/meta.bak_*/videos.bak_* 备份目录。
    """
    meta_dir  = os.path.join(dataset_root, "meta")
    data_dir  = os.path.join(dataset_root, "data")
    video_dir = os.path.join(dataset_root, "videos")

    # 编辑前置闸门：meta/data/info 不一致时拒绝动笔（带病数据会被静默切错行）
    _assert_dataset_consistent(dataset_root)

    ep_df, ep_fmt, ep_files = _read_episodes(meta_dir)
    if episode_idx < 0 or episode_idx >= len(ep_df):
        raise ValueError(f"episode_idx {episode_idx} 超出范围（共 {len(ep_df)} 个）")

    if "dataset_from_index" in ep_df.columns and "dataset_to_index" in ep_df.columns:
        from_idx = int(ep_df.loc[episode_idx, "dataset_from_index"])
        to_idx   = int(ep_df.loc[episode_idx, "dataset_to_index"])
    else:
        from_idx = 0
        for i in range(episode_idx):
            from_idx += int(ep_df.loc[i, "length"])
        to_idx = from_idx + int(ep_df.loc[episode_idx, "length"])
    n_del = to_idx - from_idx

    # 在重写 meta 之前，先记录被删 episode 的视频文件（v3.0 布局：每集独立 mp4）
    video_files_to_delete = _get_video_files_for_episode(video_dir, meta_dir, episode_idx)

    data_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    if not data_files:
        raise FileNotFoundError(f"找不到数据 parquet：{data_dir}")
    df_all = pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)
    df_new = pd.concat([df_all.iloc[:from_idx], df_all.iloc[to_idx:]], ignore_index=True)

    if "episode_index" in df_new.columns:
        df_new.loc[df_new["episode_index"] > episode_idx, "episode_index"] -= 1
    if "index" in df_new.columns:
        df_new["index"] = range(len(df_new))
    fps = _get_dataset_fps(meta_dir)
    df_new = _reset_frame_columns(df_new, fps)

    # 先在内存中完成全部校验（_sync_episode_data_pointers 发现不一致会抛错），
    # 通过后再删改 data 文件，避免中途异常导致 data 已改、meta 未改的损坏
    new_ep_df = ep_df.drop(index=episode_idx).reset_index(drop=True)
    if "dataset_from_index" in new_ep_df.columns and "dataset_to_index" in new_ep_df.columns:
        mask = new_ep_df["dataset_from_index"] >= to_idx
        new_ep_df.loc[mask, "dataset_from_index"] -= n_del
        new_ep_df.loc[mask, "dataset_to_index"]   -= n_del
    new_ep_df["episode_index"] = range(len(new_ep_df))

    # 回收因删除而成为孤儿的任务（tasks.parquet 中不再被引用的行），
    # 并压实 task_index 为 0 基连续（lerobot 按 iloc 取任务，留空洞会越界）
    new_tasks_df, removed_tasks = _gc_orphan_tasks(
        os.path.join(meta_dir, "tasks.parquet"), df_new, new_ep_df)

    new_ep_df = _sync_episode_data_pointers(new_ep_df, df_new)

    # 校验通过、动笔前备份（data 会被整体重写，被删集视频将被删除）
    _backup_for_trim(Path(dataset_root), Path(meta_dir), Path(data_dir),
                     Path(video_dir), ep_files, video_files_to_delete,
                     backup_enabled=backup_enabled)

    _remove_files(data_files)
    chunk_dir = os.path.join(data_dir, "chunk-000")
    _write_parquet_chunks(df_new, chunk_dir)

    _remove_files(ep_files)
    if ep_fmt == "parquet":
        chunk_ep = os.path.join(meta_dir, "episodes", "chunk-000")
        _write_parquet_chunks(new_ep_df, chunk_ep)
    else:
        with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
            for row in new_ep_df.to_dict(orient="records"):
                f.write(json.dumps(row) + "\n")

    # 孤儿任务已回收：原子写 tasks.parquet（以任务名为索引，tmp+replace）
    if new_tasks_df is not None:
        tasks_path = os.path.join(meta_dir, "tasks.parquet")
        tmp_tasks = tasks_path + ".tmp.parquet"
        new_tasks_df.to_parquet(tmp_tasks)
        os.replace(tmp_tasks, tasks_path)

    info_path = os.path.join(meta_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        if "total_frames"   in info: info["total_frames"]   -= n_del
        if "total_episodes" in info: info["total_episodes"] -= 1
        if new_tasks_df is not None and "total_tasks" in info:
            info["total_tasks"] = len(new_tasks_df)
        if "splits" in info and "train" in info["splits"]:
            info["splits"]["train"] = f"0:{len(df_new)}"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    num_after = len(new_ep_df)
    _update_stats_json(meta_dir, episode_idx, num_after)
    _recompute_global_stats(meta_dir, df_new)

    deleted_vids, renamed_vids = _reindex_video_files(video_dir, episode_idx, num_after)
    for vf in video_files_to_delete:
        bn = os.path.basename(vf)
        if bn.startswith("episode_") and bn.endswith(".mp4"):
            # 旧布局文件已由 _reindex_video_files 统一删除/压实；
            # 此处再按旧路径删会误伤刚重命名到该名字的保留集视频
            continue
        try:
            os.remove(vf)
            deleted_vids.append(vf)
        except OSError:
            pass

    result = (f"已删除 Episode {episode_idx}（{n_del} 帧）\n"
              f"剩余 {num_after} 个 episode，{len(df_new)} 帧")
    if removed_tasks:
        result += f"\n已清理未被引用的任务: {', '.join(removed_tasks)}"
    if deleted_vids:
        result += f"\n已删除视频: {len(deleted_vids)} 个"
    if renamed_vids:
        result += f"\n已重命名视频: {len(renamed_vids)} 个"
    return result

def _update_stats_json_batch(meta_dir, deleted_indices):
    """批量版 _update_stats_json：删除多个 episode 的统计并重新索引。"""
    del_sorted = sorted(int(i) for i in deleted_indices)

    def _shift(idx):
        # 新索引 = 旧索引 - 其之前被删 episode 的个数
        return idx - int(np.searchsorted(del_sorted, idx, side="right"))

    stats_jsonl = os.path.join(meta_dir, "stats.jsonl")
    if os.path.exists(stats_jsonl):
        with open(stats_jsonl) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        new_rows = []
        for row in rows:
            ep = row.get("episode_index", -1)
            if ep in del_sorted:
                continue
            row = dict(row)
            row["episode_index"] = _shift(ep)
            new_rows.append(row)
        with open(stats_jsonl, "w") as f:
            for row in new_rows:
                f.write(json.dumps(row) + "\n")

    stats_json = os.path.join(meta_dir, "stats.json")
    if os.path.exists(stats_json):
        with open(stats_json) as f:
            stats = json.load(f)
        if "episodes" in stats:
            ep_stats = stats["episodes"]
            if isinstance(ep_stats, list):
                new_ep_stats = []
                for s in ep_stats:
                    ep = s.get("episode_index", -1)
                    if ep in del_sorted:
                        continue
                    s = dict(s)
                    s["episode_index"] = _shift(ep)
                    new_ep_stats.append(s)
                stats["episodes"] = new_ep_stats
            elif isinstance(ep_stats, dict):
                new_ep_stats = {}
                for k, v in ep_stats.items():
                    idx = int(k)
                    if idx in del_sorted:
                        continue
                    new_ep_stats[str(_shift(idx))] = v
                stats["episodes"] = new_ep_stats
        with open(stats_json, "w") as f:
            json.dump(stats, f, indent=2)

def _reindex_video_files_batch(video_dir, deleted_indices):
    """批量版 _reindex_video_files：删除被删集的旧式视频文件，后续文件按下移量重命名。"""
    if not os.path.exists(video_dir):
        return [], []
    del_sorted = sorted(int(i) for i in deleted_indices)

    vid_pattern = os.path.join(video_dir, "**", "episode_*.mp4")
    vid_files = glob.glob(vid_pattern, recursive=True)

    entries = []
    for vf in vid_files:
        basename = os.path.basename(vf)
        if basename.startswith("episode_") and basename.endswith(".mp4"):
            try:
                ep_num = int(basename[8:14])
                entries.append((vf, os.path.dirname(vf), ep_num, basename))
            except ValueError:
                continue

    from collections import defaultdict
    cam_groups = defaultdict(list)
    for vf, cam_path, ep_num, basename in entries:
        cam_groups[cam_path].append((vf, ep_num, basename))

    deleted = []
    renamed = []

    for cam_path, items in cam_groups.items():
        to_delete = [(vf, ep, bn) for vf, ep, bn in items if ep in del_sorted]
        to_reindex = [(vf, ep, bn) for vf, ep, bn in items if ep not in del_sorted]

        for vf, _, _ in to_delete:
            try:
                os.remove(vf)
                deleted.append(vf)
            except Exception:
                pass

        # 升序重命名避免相互覆盖：所有文件都向下移动（new < old），
        # 处理到 ep_num 时目标位置必然空闲（更小的文件已移到更低位置，
        # 被删位置已清空）；shift=0 的文件无需变动
        to_reindex_sorted = sorted(to_reindex, key=lambda x: x[1])
        for vf, ep_num, bn in to_reindex_sorted:
            shift = int(np.searchsorted(del_sorted, ep_num, side="right"))
            if shift == 0:
                continue
            new_bn = f"episode_{ep_num - shift:06d}.mp4"
            new_vf = os.path.join(cam_path, new_bn)
            try:
                os.rename(vf, new_vf)
                renamed.append((vf, new_vf))
            except Exception:
                pass

    return deleted, renamed

def delete_episodes(dataset_root, episode_indices, backup_enabled=True):
    """批量安全删除多个 episode：单次重写 data/meta/stats 并重排索引，保证数据一致性。

    动笔前先跑三方一致性闸门（不一致直接抛错拒绝）；
    backup_enabled=True 时生成 data.bak_*/meta.bak_*/videos.bak_* 备份目录。

    与 delete_episode 维持相同的结构不变量（见 数据问题分析.md 问题 1~4）：
    - data parquet 按删除区间一次性过滤，episode_index 按下移量重排；
    - index 列重置为全局连续，frame_index/timestamp 按 episode 重生成；
    - meta episodes 删除对应行，dataset_from/to_index 按保留集 length 累重重算，
      data/chunk_index、data/file_index 与实际 parquet 布局同步；
    - info.json 的 total_frames/total_episodes/splits 同步更新；
    - stats.json(l) 的 per-episode 统计删除并重索引，全量聚合统计重写；
    - v3.0 布局删除被删集的独立视频文件；旧布局（episode_XXXXXX.mp4）重命名压实索引。

    整个过程 data/meta 只读写一遍（逐集删除会全量重写 N 次，且中途易损坏）。
    """
    del_sorted = sorted({int(i) for i in episode_indices})
    if not del_sorted:
        raise ValueError("未指定要删除的 episode")

    # 编辑前置闸门：meta/data/info 不一致时拒绝动笔（带病数据会被静默切错行）
    _assert_dataset_consistent(dataset_root)

    meta_dir  = os.path.join(dataset_root, "meta")
    data_dir  = os.path.join(dataset_root, "data")
    video_dir = os.path.join(dataset_root, "videos")

    ep_df, ep_fmt, ep_files = _read_episodes(meta_dir)
    n_eps = len(ep_df)
    if del_sorted[0] < 0 or del_sorted[-1] >= n_eps:
        raise ValueError(f"episode_idx 超出范围（共 {n_eps} 个）：{del_sorted}")
    if len(del_sorted) >= n_eps:
        raise ValueError("不能删除全部 episode（至少保留 1 个）")

    # 每个 episode 的 [from, to) 行区间
    if "dataset_from_index" in ep_df.columns and "dataset_to_index" in ep_df.columns:
        ranges = [(int(ep_df.loc[i, "dataset_from_index"]),
                   int(ep_df.loc[i, "dataset_to_index"])) for i in range(n_eps)]
    else:
        ranges = []
        pos = 0
        for i in range(n_eps):
            ln = int(ep_df.loc[i, "length"])
            ranges.append((pos, pos + ln))
            pos += ln
    n_del_total = sum(ranges[i][1] - ranges[i][0] for i in del_sorted)

    # 在重写 meta 之前，先记录被删 episode 的视频文件（v3.0 布局：每集独立 mp4）
    video_files_to_delete = []
    for idx in del_sorted:
        video_files_to_delete.extend(
            _get_video_files_for_episode(video_dir, meta_dir, idx))

    data_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    if not data_files:
        raise FileNotFoundError(f"找不到数据 parquet：{data_dir}")
    df_all = pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)

    # 一次性过滤所有被删 episode 的数据行（保持其余行的相对顺序）
    keep_mask = np.ones(len(df_all), dtype=bool)
    for i in del_sorted:
        keep_mask[ranges[i][0]:ranges[i][1]] = False
    df_new = df_all.iloc[np.nonzero(keep_mask)[0]].reset_index(drop=True)

    if "episode_index" in df_new.columns:
        ep_col = df_new["episode_index"].to_numpy()
        df_new["episode_index"] = ep_col - np.searchsorted(
            del_sorted, ep_col, side="right")
    if "index" in df_new.columns:
        df_new["index"] = range(len(df_new))
    fps = _get_dataset_fps(meta_dir)
    df_new = _reset_frame_columns(df_new, fps)

    # 先在内存中完成全部校验（_sync_episode_data_pointers 发现不一致会抛错），
    # 通过后再删改 data 文件，避免中途异常导致 data 已改、meta 未改的损坏
    new_ep_df = ep_df.drop(index=del_sorted).reset_index(drop=True)
    if "dataset_from_index" in new_ep_df.columns and "dataset_to_index" in new_ep_df.columns:
        pos = 0
        for i in range(len(new_ep_df)):
            if "length" in new_ep_df.columns:
                ln = int(new_ep_df.loc[i, "length"])
            else:
                ln = int(new_ep_df.loc[i, "dataset_to_index"]) - \
                     int(new_ep_df.loc[i, "dataset_from_index"])
            new_ep_df.loc[i, "dataset_from_index"] = pos
            new_ep_df.loc[i, "dataset_to_index"] = pos + ln
            pos += ln
    new_ep_df["episode_index"] = range(len(new_ep_df))

    # 回收因删除而成为孤儿的任务（tasks.parquet 中不再被引用的行），
    # 并压实 task_index 为 0 基连续（lerobot 按 iloc 取任务，留空洞会越界）
    new_tasks_df, removed_tasks = _gc_orphan_tasks(
        os.path.join(meta_dir, "tasks.parquet"), df_new, new_ep_df)

    new_ep_df = _sync_episode_data_pointers(new_ep_df, df_new)

    # 校验通过、动笔前备份（data 会被整体重写，被删集视频将被删除）
    _backup_for_trim(Path(dataset_root), Path(meta_dir), Path(data_dir),
                     Path(video_dir), ep_files, video_files_to_delete,
                     backup_enabled=backup_enabled)

    _remove_files(data_files)
    chunk_dir = os.path.join(data_dir, "chunk-000")
    _write_parquet_chunks(df_new, chunk_dir)

    _remove_files(ep_files)
    if ep_fmt == "parquet":
        chunk_ep = os.path.join(meta_dir, "episodes", "chunk-000")
        _write_parquet_chunks(new_ep_df, chunk_ep)
    else:
        with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
            for row in new_ep_df.to_dict(orient="records"):
                f.write(json.dumps(row) + "\n")

    # 孤儿任务已回收：原子写 tasks.parquet（以任务名为索引，tmp+replace）
    if new_tasks_df is not None:
        tasks_path = os.path.join(meta_dir, "tasks.parquet")
        tmp_tasks = tasks_path + ".tmp.parquet"
        new_tasks_df.to_parquet(tmp_tasks)
        os.replace(tmp_tasks, tasks_path)

    info_path = os.path.join(meta_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        if "total_frames"   in info: info["total_frames"]   -= n_del_total
        if "total_episodes" in info: info["total_episodes"] -= len(del_sorted)
        if new_tasks_df is not None and "total_tasks" in info:
            info["total_tasks"] = len(new_tasks_df)
        if "splits" in info and "train" in info["splits"]:
            info["splits"]["train"] = f"0:{len(df_new)}"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    _update_stats_json_batch(meta_dir, del_sorted)
    _recompute_global_stats(meta_dir, df_new)

    deleted_vids, renamed_vids = _reindex_video_files_batch(video_dir, del_sorted)
    for vf in video_files_to_delete:
        bn = os.path.basename(vf)
        if bn.startswith("episode_") and bn.endswith(".mp4"):
            # 旧布局文件由 _reindex_video_files_batch 统一删除/压实；
            # 此处再删会误伤刚重命名到该名字的保留集视频
            continue
        try:
            os.remove(vf)
            deleted_vids.append(vf)
        except OSError:
            pass

    result = (f"已批量删除 {len(del_sorted)} 个 Episode（共 {n_del_total} 帧）\n"
              f"编号: {del_sorted}\n"
              f"剩余 {len(new_ep_df)} 个 episode，{len(df_new)} 帧")
    if removed_tasks:
        result += f"\n已清理未被引用的任务: {', '.join(removed_tasks)}"
    if deleted_vids:
        result += f"\n已删除视频: {len(deleted_vids)} 个"
    if renamed_vids:
        result += f"\n已重命名视频: {len(renamed_vids)} 个"
    return result

def _episode_task_display_name(dataset, ep):
    """从 episode 元数据提取任务显示名（与 _update_episode_table 第 4 列逻辑一致）。"""
    if "tasks" in ep and ep["tasks"] is not None:
        tasks = ep["tasks"]
        if hasattr(tasks, '__iter__') and not isinstance(tasks, str):
            task_list = list(tasks)
            if task_list:
                return str(task_list[0])
    elif "task_index" in ep:
        # 旧格式：按 task_index 值查找（不能用列表位置索引，见 数据问题分析.md 问题 7）
        return str(dataset.task_name_for_index(ep["task_index"]))
    return ""


# ==================== Episode 剪辑工具 ====================
def _get_video_files_for_episode(video_dir, meta_dir, episode_idx):
    """根据 episodes 元数据查找指定 episode 对应的视频文件路径列表。

    支持 LeRobot 0.4.4 格式（videos/{cam}/chunk-{idx}/file-{idx}.mp4）
    以及兜底格式（videos/{cam}/episode_{episode_idx:06d}.mp4）。
    """
    if not os.path.exists(video_dir) or not os.path.exists(meta_dir):
        return []

    try:
        ep_df, _, _ = _read_episodes(str(meta_dir))
    except Exception:
        return []

    if episode_idx < 0 or episode_idx >= len(ep_df):
        return []

    ep = ep_df.iloc[episode_idx]
    files = []

    for cam in os.listdir(video_dir):
        cam_path = os.path.join(video_dir, cam)
        if not os.path.isdir(cam_path):
            continue

        # LeRobot 0.4.4 格式：从 metadata 读取 chunk_index / file_index
        chunk_key = f"videos/{cam}/chunk_index"
        file_key = f"videos/{cam}/file_index"
        if chunk_key in ep and file_key in ep:
            chunk_idx = int(ep[chunk_key])
            file_idx = int(ep[file_key])
            vf = os.path.join(cam_path, f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.mp4")
            if os.path.exists(vf):
                files.append(vf)
                continue

        # 兜底格式：episode_{episode_idx:06d}.mp4
        for name in [f"episode_{episode_idx:06d}.mp4", f"episode-{episode_idx:06d}.mp4"]:
            vf = os.path.join(cam_path, name)
            if os.path.exists(vf):
                files.append(vf)
                break

    return files

def _find_video_files_for_trim(video_dir):
    """兼容旧名，实际行为改为返回空列表，避免误伤其他 episode 视频。"""
    return []

def _backup_for_trim(root, meta_dir, data_dir, video_dir, ep_files, video_files_to_modify, backup_enabled=True):
    """仅备份会被修改的文件/目录，避免复制整个数据集导致的长等待。

    当 backup_enabled 为 False 时直接返回，不生成任何备份目录。
    """
    if not backup_enabled:
        return
    from datetime import datetime
    import shutil

    bak_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 备份 data parquet 文件（data 会被整体重写）
    if data_dir.exists():
        bak_data = root / f"data.bak_{bak_suffix}"
        bak_data.mkdir(parents=True, exist_ok=True)
        for src in sorted(data_dir.rglob("*.parquet")):
            rel = src.relative_to(data_dir)
            dst = bak_data / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # 备份 meta 文件
    if meta_dir.exists():
        bak_meta = root / f"meta.bak_{bak_suffix}"
        bak_meta.mkdir(parents=True, exist_ok=True)
        for src in ep_files:
            shutil.copy2(src, bak_meta / Path(src).name)
        for name in ["stats.json", "stats.jsonl", "info.json", "tasks.parquet"]:
            src = meta_dir / name
            if src.exists():
                shutil.copy2(src, bak_meta / name)

    # 只备份会被修改的视频文件
    if video_files_to_modify:
        bak_videos = root / f"videos.bak_{bak_suffix}"
        bak_videos.mkdir(parents=True, exist_ok=True)
        for src in video_files_to_modify:
            rel = Path(src).relative_to(video_dir)
            dst = bak_videos / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

def trim_episode(dataset_root, episode_idx, start_frame, end_frame, backup_enabled=True):
    """删除 episode 内 [start_frame, end_frame] 闭区间的片段，保留其余部分，并重新编排索引。

    当 backup_enabled=True 时生成备份目录 meta.bak/、data.bak/、videos.bak/。
    """
    from datetime import datetime
    import shutil

    root = Path(dataset_root)
    meta_dir = root / "meta"
    data_dir = root / "data"
    video_dir = root / "videos"

    # 编辑前置闸门：meta/data/info 不一致时拒绝动笔（带病数据会被静默切错行）
    _assert_dataset_consistent(dataset_root)

    ep_df, ep_fmt, ep_files = _read_episodes(str(meta_dir))
    if episode_idx < 0 or episode_idx >= len(ep_df):
        raise ValueError(f"episode_idx {episode_idx} 超出范围（共 {len(ep_df)} 个）")

    if "dataset_from_index" in ep_df.columns and "dataset_to_index" in ep_df.columns:
        from_idx = int(ep_df.loc[episode_idx, "dataset_from_index"])
        to_idx   = int(ep_df.loc[episode_idx, "dataset_to_index"])
    else:
        from_idx = 0
        for i in range(episode_idx):
            from_idx += int(ep_df.loc[i, "length"])
        to_idx = from_idx + int(ep_df.loc[episode_idx, "length"])
    ep_data_len = to_idx - from_idx

    if start_frame < 0 or end_frame >= ep_data_len or start_frame > end_frame:
        raise ValueError(f"剪辑范围 {start_frame}~{end_frame} 超出 episode 帧数 {ep_data_len}")

    video_files_to_modify = _get_video_files_for_episode(str(video_dir), str(meta_dir), episode_idx)

    # 仅备份会被修改的文件/目录，避免复制整个数据集导致的长等待
    _backup_for_trim(root, meta_dir, data_dir, video_dir, ep_files, video_files_to_modify, backup_enabled=backup_enabled)

    n_removed = end_frame - start_frame + 1

    # 读取并裁剪 data
    data_files = sorted(data_dir.rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"找不到数据 parquet：{data_dir}")
    df_all = pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)
    remove_start = from_idx + start_frame
    remove_end = from_idx + end_frame + 1
    df_new = pd.concat([df_all.iloc[:remove_start], df_all.iloc[remove_end:]], ignore_index=True)

    if "episode_index" in df_new.columns:
        # 仅剪辑当前 episode 的片段，不删除 episode，因此 episode_index 保持不变
        pass
    if "index" in df_new.columns:
        df_new["index"] = range(len(df_new))
    fps = _get_dataset_fps(meta_dir)
    df_new = _reset_frame_columns(df_new, fps)

    # 先在内存中完成全部校验（_sync_episode_data_pointers 发现不一致会抛错），
    # 通过后再删改 data 文件，避免中途异常导致 data 已改、meta 未改的损坏
    new_ep_len = ep_data_len - n_removed
    new_ep_df = ep_df.copy()
    if "dataset_from_index" in new_ep_df.columns and "dataset_to_index" in new_ep_df.columns:
        new_ep_df.loc[episode_idx, "dataset_to_index"] = from_idx + new_ep_len
        new_ep_df.loc[episode_idx, "length"] = new_ep_len
        mask = new_ep_df["dataset_from_index"] >= to_idx
        new_ep_df.loc[mask, "dataset_from_index"] -= n_removed
        new_ep_df.loc[mask, "dataset_to_index"] -= n_removed
    else:
        new_ep_df.loc[episode_idx, "length"] = new_ep_len
    new_ep_df = _sync_episode_data_pointers(new_ep_df, df_new)
    new_ep_df = _update_video_timestamps_after_trim(new_ep_df, episode_idx, new_ep_len, fps)

    _remove_files([str(p) for p in data_files])
    chunk_dir = data_dir / "chunk-000"
    _write_parquet_chunks(df_new, chunk_dir)

    # 如果有 episode_stats 也更新
    _update_episode_stats_for_trim(str(meta_dir), episode_idx, new_ep_len)
    _recompute_global_stats(str(meta_dir), df_new)

    # 重写 episodes
    _remove_files(ep_files)
    if ep_fmt == "parquet":
        chunk_ep = meta_dir / "episodes" / "chunk-000"
        _write_parquet_chunks(new_ep_df, chunk_ep)
    else:
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for row in new_ep_df.to_dict(orient="records"):
                f.write(json.dumps(row) + "\n")

    # 更新 info.json
    info = None
    info_path = meta_dir / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        if "total_frames" in info:
            info["total_frames"] -= n_removed
        if "splits" in info and "train" in info["splits"]:
            info["splits"]["train"] = f"0:{len(df_new)}"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    # 剪辑视频：重新编码当前 episode 的视频，并删除被选中片段
    _trim_video_files(str(video_dir), episode_idx, start_frame, end_frame, video_files=video_files_to_modify)

    result = f"已剪辑 Episode {episode_idx}\n删除帧: {start_frame}~{end_frame} ({n_removed} 帧)\n保留帧: {new_ep_len}"
    return result, df_new, new_ep_df, info

def _update_episode_stats_for_trim(meta_dir, episode_idx, new_length):
    """更新 stats.json / stats.jsonl 中当前 episode 的长度字段。"""
    stats_jsonl = os.path.join(meta_dir, "stats.jsonl")
    if os.path.exists(stats_jsonl):
        with open(stats_jsonl) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        for row in rows:
            if row.get("episode_index") == episode_idx:
                row["length"] = new_length
        with open(stats_jsonl, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    stats_json = os.path.join(meta_dir, "stats.json")
    if os.path.exists(stats_json):
        with open(stats_json) as f:
            stats = json.load(f)
        if "episodes" in stats:
            ep_stats = stats["episodes"]
            if isinstance(ep_stats, list):
                for s in ep_stats:
                    if s.get("episode_index") == episode_idx:
                        s["length"] = new_length
            elif isinstance(ep_stats, dict):
                if str(episode_idx) in ep_stats:
                    ep_stats[str(episode_idx)]["length"] = new_length
        # 同时更新全局 total_frames / stats 汇总
        if "stats" in stats and "episode_index" in stats["stats"]:
            se = stats["stats"]["episode_index"]
            if isinstance(se, dict) and "length" in se and str(episode_idx) in se["length"]:
                se["length"][str(episode_idx)] = new_length
        with open(stats_json, "w") as f:
            json.dump(stats, f, indent=2)

def _trim_video_files(video_dir, episode_idx, start_frame, end_frame, video_files=None):
    """对指定 episode 的视频进行片段删除，保留前后两段。

    优先使用 ffmpeg 直接处理，避免把整个视频解码到内存；未安装 ffmpeg 时回退到 imageio。
    删除范围为 [start_frame, end_frame] 闭区间（与 data 裁剪一致）。
    video_files 为可选的具体文件路径列表；若提供则只处理这些文件，否则根据元数据查找。
    """
    import shutil
    import subprocess

    if not os.path.exists(video_dir):
        return

    if video_files is None:
        # 根据调用位置，meta_dir 应该是 video_dir 的兄弟目录
        meta_dir = os.path.join(os.path.dirname(video_dir), "meta")
        video_files = _get_video_files_for_episode(video_dir, meta_dir, episode_idx)

    if not video_files:
        return

    has_ffmpeg = shutil.which("ffmpeg") is not None

    for vf in video_files:
        if not os.path.exists(vf):
            continue
        tmp = vf + ".tmp.mp4"
        try:
            if has_ffmpeg:
                # 删除 [start, end] 闭区间：between(n, start, end) 选中该区间的帧
                select_expr = f"select='not(between(n\\,{start_frame}\\,{end_frame}))',setpts=N/FRAME_RATE/TB"
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", vf,
                    "-vf", select_expr,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-preset", "medium",
                    tmp
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg 剪辑失败: {result.stderr}")
                os.replace(tmp, vf)
            else:
                import imageio.v3 as iio
                frames = list(iio.imiter(vf))
                if not frames:
                    continue
                # 删除 [start_frame, end_frame] 闭区间，保留前后两段
                keep = list(frames[:start_frame]) + list(frames[end_frame + 1:])
                if not keep:
                    os.remove(vf)
                    continue
                iio.imwrite(tmp, keep, codec="libx264", fps=20, quality=8)
                os.replace(tmp, vf)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

def _trim_video_files_multiple(video_dir, episode_idx, ranges, video_files=None):
    """对指定 episode 的视频进行多段删除，保留其余部分。

    优先使用 ffmpeg 直接处理；未安装 ffmpeg 时回退到 imageio。
    ranges 为已合并排序的 [(start, end), ...] 闭区间列表。
    video_files 为可选的具体文件路径列表；若提供则只处理这些文件，否则根据元数据查找。
    """
    import shutil
    import subprocess

    if not os.path.exists(video_dir):
        return

    if video_files is None:
        meta_dir = os.path.join(os.path.dirname(video_dir), "meta")
        video_files = _get_video_files_for_episode(video_dir, meta_dir, episode_idx)

    if not video_files:
        return

    has_ffmpeg = shutil.which("ffmpeg") is not None

    for vf in video_files:
        if not os.path.exists(vf):
            continue
        tmp = vf + ".tmp.mp4"
        try:
            if has_ffmpeg:
                terms = [f"between(n\\,{s}\\,{e})" for s, e in ranges]
                select_expr = "select='not(" + "+".join(terms) + ")',setpts=N/FRAME_RATE/TB"
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", vf,
                    "-vf", select_expr,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-preset", "medium",
                    tmp
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg 剪辑失败: {result.stderr}")
                os.replace(tmp, vf)
            else:
                import imageio.v3 as iio
                frames = list(iio.imiter(vf))
                if not frames:
                    continue
                keep = []
                prev_end = 0
                for s, e in ranges:
                    keep.extend(frames[prev_end:s])
                    prev_end = e + 1
                keep.extend(frames[prev_end:])
                if not keep:
                    os.remove(vf)
                    continue
                iio.imwrite(tmp, keep, codec="libx264", fps=20, quality=8)
                os.replace(tmp, vf)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

def trim_episode_ranges(dataset_root, episode_idx, ranges, backup_enabled=True):
    """删除 episode 内多个 [start, end] 闭区间片段，保留其余部分，并重新编排索引。

    当 backup_enabled=True 时生成备份目录 meta.bak/、data.bak/、videos.bak/。
    ranges 为 episode 内相对帧索引的 [(start, end), ...] 列表。
    """
    from datetime import datetime
    import shutil

    root = Path(dataset_root)
    meta_dir = root / "meta"
    data_dir = root / "data"
    video_dir = root / "videos"

    # 编辑前置闸门：meta/data/info 不一致时拒绝动笔（带病数据会被静默切错行）
    _assert_dataset_consistent(dataset_root)

    ep_df, ep_fmt, ep_files = _read_episodes(str(meta_dir))
    if episode_idx < 0 or episode_idx >= len(ep_df):
        raise ValueError(f"episode_idx {episode_idx} 超出范围（共 {len(ep_df)} 个）")

    if "dataset_from_index" in ep_df.columns and "dataset_to_index" in ep_df.columns:
        from_idx = int(ep_df.loc[episode_idx, "dataset_from_index"])
        to_idx   = int(ep_df.loc[episode_idx, "dataset_to_index"])
    else:
        from_idx = 0
        for i in range(episode_idx):
            from_idx += int(ep_df.loc[i, "length"])
        to_idx = from_idx + int(ep_df.loc[episode_idx, "length"])
    ep_data_len = to_idx - from_idx

    # 校验并合并区间
    normalized = []
    for s, e in ranges:
        s = int(s)
        e = int(e)
        if s < 0 or e >= ep_data_len or s > e:
            continue
        normalized.append((s, e))
    if not normalized:
        raise ValueError("没有有效的静止区间可剪辑")
    normalized.sort()

    merged = [normalized[0]]
    for s, e in normalized[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    n_removed = sum(e - s + 1 for s, e in merged)
    if n_removed >= ep_data_len:
        raise ValueError("静止区间覆盖了整个 Episode，无法全部剪辑，请使用删除 Episode")

    video_files_to_modify = _get_video_files_for_episode(str(video_dir), str(meta_dir), episode_idx)
    _backup_for_trim(root, meta_dir, data_dir, video_dir, ep_files, video_files_to_modify, backup_enabled=backup_enabled)

    # 读取完整 data
    data_files = sorted(data_dir.rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"找不到数据 parquet：{data_dir}")
    df_all = pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)

    # 保留区间 = 当前 episode 之前的数据 + 当前 episode 去掉删除段后的数据 + 当前 episode 之后的数据
    keep_parts = []
    if from_idx > 0:
        keep_parts.append(df_all.iloc[:from_idx])

    prev_end = from_idx
    for s, e in merged:
        abs_s = from_idx + s
        abs_e = from_idx + e + 1
        if prev_end < abs_s:
            keep_parts.append(df_all.iloc[prev_end:abs_s])
        prev_end = abs_e
    if prev_end < to_idx:
        keep_parts.append(df_all.iloc[prev_end:to_idx])

    if to_idx < len(df_all):
        keep_parts.append(df_all.iloc[to_idx:])

    df_new = pd.concat(keep_parts, ignore_index=True)

    if "episode_index" in df_new.columns:
        # 仅剪辑当前 episode 的片段，不删除 episode，因此 episode_index 保持不变
        pass
    if "index" in df_new.columns:
        df_new["index"] = range(len(df_new))
    fps = _get_dataset_fps(meta_dir)
    df_new = _reset_frame_columns(df_new, fps)

    # 先在内存中完成全部校验（_sync_episode_data_pointers 发现不一致会抛错），
    # 通过后再删改 data 文件，避免中途异常导致 data 已改、meta 未改的损坏
    new_ep_len = ep_data_len - n_removed
    new_ep_df = ep_df.copy()
    if "dataset_from_index" in new_ep_df.columns and "dataset_to_index" in new_ep_df.columns:
        new_ep_df.loc[episode_idx, "dataset_to_index"] = from_idx + new_ep_len
        new_ep_df.loc[episode_idx, "length"] = new_ep_len
        mask = new_ep_df["dataset_from_index"] >= to_idx
        new_ep_df.loc[mask, "dataset_from_index"] -= n_removed
        new_ep_df.loc[mask, "dataset_to_index"] -= n_removed
    else:
        new_ep_df.loc[episode_idx, "length"] = new_ep_len
    new_ep_df = _sync_episode_data_pointers(new_ep_df, df_new)
    new_ep_df = _update_video_timestamps_after_trim(new_ep_df, episode_idx, new_ep_len, fps)

    _remove_files([str(p) for p in data_files])
    chunk_dir = data_dir / "chunk-000"
    _write_parquet_chunks(df_new, chunk_dir)

    _update_episode_stats_for_trim(str(meta_dir), episode_idx, new_ep_len)
    _recompute_global_stats(str(meta_dir), df_new)

    _remove_files(ep_files)
    if ep_fmt == "parquet":
        chunk_ep = meta_dir / "episodes" / "chunk-000"
        _write_parquet_chunks(new_ep_df, chunk_ep)
    else:
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for row in new_ep_df.to_dict(orient="records"):
                f.write(json.dumps(row) + "\n")

    # 更新 info.json
    info = None
    info_path = meta_dir / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        if "total_frames" in info:
            info["total_frames"] -= n_removed
        if "splits" in info and "train" in info["splits"]:
            info["splits"]["train"] = f"0:{len(df_new)}"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    # 剪辑视频
    _trim_video_files_multiple(str(video_dir), episode_idx, merged, video_files=video_files_to_modify)

    result = (f"已剪辑 Episode {episode_idx}\n"
              f"删除段数: {len(merged)}\n"
              f"删除帧: {n_removed}\n"
              f"保留帧: {new_ep_len}")
    return result, df_new, new_ep_df, info


# ==================== 轨迹平滑降噪 ====================
def _smooth_1d(arr, method, window, polyorder, sigma):
    """对一维序列做零相位平滑（不引入时间滞后，避免 action 与图像帧错位）。"""
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n < 3:
        return arr.copy()
    if method == "gaussian":
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(arr, sigma, mode="nearest")
    # 窗口类滤波：窗口取奇数且不超过序列长度
    window = int(window)
    if window % 2 == 0:
        window += 1
    max_odd = n if n % 2 == 1 else n - 1
    window = max(3, min(window, max_odd))
    if method == "moving_avg":
        pad = window // 2
        kernel = np.ones(window) / window
        padded = np.pad(arr, pad, mode="edge")
        return np.convolve(padded, kernel, mode="valid")
    # 默认 Savitzky-Golay（零相位、保形状，离线轨迹平滑推荐）
    from scipy.signal import savgol_filter
    polyorder = max(1, min(int(polyorder), window - 1))
    return savgol_filter(arr, window, polyorder, mode="interp")

def _smooth_pose_matrix(mat, method, window, polyorder, sigma):
    """对 (N, D) 位姿矩阵做平滑降噪，返回 float32 矩阵。

    - 旧 8 维四元数格式（第 4~7 维 qx,qy,qz,qw）：先做符号连续化（q 与 -q 等价），
      滤波后重新单位化，保证输出仍是合法旋转。
    - 新 7 维欧拉角格式：所有维度直接滤波（欧拉角无需符号连续化/单位化）。
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] < 3:
        return mat.astype(np.float32)
    out = np.empty_like(mat)
    for d in range(mat.shape[1]):
        out[:, d] = _smooth_1d(mat[:, d], method, window, polyorder, sigma)
    # 仅旧 8 维四元数格式才需要四元数符号连续化 + 单位化
    if mat.shape[1] >= 8:
        q = mat[:, 3:7].copy()
        for i in range(1, len(q)):
            if float(np.dot(q[i], q[i - 1])) < 0.0:
                q[i] = -q[i]
        for d in range(4):
            out[:, 3 + d] = _smooth_1d(q[:, d], method, window, polyorder, sigma)
        norms = np.linalg.norm(out[:, 3:7], axis=1)
        norms = np.where(norms < 1e-9, 1.0, norms)
        out[:, 3:7] = out[:, 3:7] / norms[:, None]
    return out.astype(np.float32)

def denoise_dataset(dataset_root, episode_idx, params, backup_enabled=True, progress_cb=None):
    """对指定 episode（episode_idx>=0）或全部 episode（episode_idx=-1）的
    observation.state / action 轨迹进行平滑降噪。

    安全性说明（避免破坏 LeRobot v3 数据一致性）：
      - 只修改 data parquet 中 observation.state / action 两列的数值；
      - 帧数、行序、frame_index/timestamp/episode_index/index/task_index 完全不变；
      - 视频文件与 meta/episodes 不触碰（episode 长度不变，to_timestamp 仍有效）；
      - 写回后调用 _recompute_global_stats 重算 meta/stats.json，避免统计量过期。
    """
    root = Path(dataset_root)
    meta_dir = root / "meta"
    data_dir = root / "data"
    video_dir = root / "videos"

    # 编辑前置闸门：meta/data/info 不一致时拒绝动笔
    _assert_dataset_consistent(dataset_root)

    method = params.get("method", "savgol")
    window = int(params.get("window", 11))
    polyorder = int(params.get("polyorder", 2))
    sigma = float(params.get("sigma", 2.0))
    smooth_action = bool(params.get("smooth_action", True))
    smooth_state = bool(params.get("smooth_state", True))
    if not smooth_action and not smooth_state:
        raise ValueError("请至少选择一个作用对象（action 或 state）")

    ep_df, _, ep_files = _read_episodes(str(meta_dir))
    n_eps = len(ep_df)
    if episode_idx >= n_eps:
        raise ValueError(f"episode_idx {episode_idx} 超出范围（共 {n_eps} 个）")
    targets = list(range(n_eps)) if episode_idx < 0 else [int(episode_idx)]

    data_files = sorted(data_dir.rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"找不到数据 parquet：{data_dir}")
    # 逐文件读取并记录原始行数，写回时保持每个文件的行数不变，
    # 使 meta/episodes 中 data/chunk_index、data/file_index 指针保持有效
    dfs = [pd.read_parquet(p) for p in data_files]
    file_rows = [len(d) for d in dfs]
    df_all = pd.concat(dfs, ignore_index=True)

    cols = [c for c, flag in [("action", smooth_action), ("observation.state", smooth_state)]
            if flag and c in df_all.columns]
    if not cols:
        raise ValueError("数据中没有可降噪的 action / observation.state 列")

    _backup_for_trim(root, meta_dir, data_dir, video_dir, ep_files, [],
                     backup_enabled=backup_enabled)

    processed = 0
    for i, ep_i in enumerate(targets):
        if progress_cb is not None:
            progress_cb(i, len(targets), f"正在降噪 Episode {ep_i}")
        ep = ep_df.iloc[ep_i]
        if "dataset_from_index" in ep_df.columns and "dataset_to_index" in ep_df.columns:
            from_idx = int(ep["dataset_from_index"])
            to_idx = int(ep["dataset_to_index"])
        else:
            from_idx = 0
            for j in range(ep_i):
                from_idx += int(ep_df.iloc[j]["length"])
            to_idx = from_idx + int(ep["length"])
        if to_idx - from_idx < 3:
            continue
        idx = df_all.index[from_idx:to_idx]
        for col in cols:
            mat = np.stack(df_all[col].iloc[from_idx:to_idx].to_numpy())
            smoothed = _smooth_pose_matrix(mat, method, window, polyorder, sigma)
            df_all.loc[idx, col] = pd.Series(list(smoothed), index=idx)
        processed += 1

    # 按原始分块布局写回：每个文件行数不变，data 文件指针保持有效（原子写）
    pos = 0
    for p, n in zip(data_files, file_rows):
        _atomic_write_parquet(df_all.iloc[pos:pos + n], p)
        pos += n
    _recompute_global_stats(str(meta_dir), df_all)

    method_names = {"savgol": "Savitzky-Golay", "moving_avg": "移动平均", "gaussian": "高斯滤波"}
    result = (f"降噪完成\n"
              f"方法: {method_names.get(method, method)} (窗口={window}, 阶数={polyorder}, σ={sigma})\n"
              f"作用对象: {', '.join(cols)}\n"
              f"处理 Episode 数: {processed}\n"
              f"总帧数不变: {len(df_all)}（元数据与视频未改动，stats.json 已重算）")
    return result, df_all


def refix_action_labels(dataset_root):
    """剪辑运动段后重算 action 标签：action[t] = state[t + k]（k 自动估计）。

    录制约定 action[t]=state[t+lookahead]（数据问题分析.md 问题5），剪辑静止段
    时位姿恒定、边界误差可忽略；但剪辑抖动/运动段后，剪点前的 action 仍指向
    已被删除的未来状态（目标跳变），必须重算。

    做法：
    1. 估计 lookahead k：扫描 k∈[0,20]，取 mean|action[t]−state[t+k]| 最小
       （与总信息面板"action 前瞻"同一判据，逐 episode 采样）；
    2. 逐 episode 重写 action[t] = state[min(t+k, L−1)]（末尾夹持）；
    3. 原子重写全部 data parquet（分块布局不变）+ 重算 stats.json。

    Returns:
        (k, n_frames)：估计的前瞻帧数与重写的总帧数；数据集无 state/action 时
        抛 ValueError。
    """
    root = Path(dataset_root)
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"找不到数据 parquet: {root / 'data'}")
    df_all = pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)
    if "action" not in df_all.columns or "observation.state" not in df_all.columns:
        raise ValueError("数据集缺少 action / observation.state 列，无法重算")

    # 1. 估计 lookahead（隔帧采样控制计算量）
    max_k = 20
    scores = np.zeros(max_k + 1)
    counts = 0
    for _, g in df_all.groupby("episode_index"):
        if len(g) < max_k + 10:
            continue
        s = np.stack(g["observation.state"].to_numpy())
        a = np.stack(g["action"].to_numpy())
        idx = np.arange(0, len(g) - max_k, 5)
        if len(idx) == 0:
            continue
        for k in range(max_k + 1):
            scores[k] += np.abs(a[idx] - s[idx + k]).sum()
        counts += len(idx)
        if counts >= 3000:
            break
    if counts == 0:
        raise ValueError("没有足够长的 episode 估计 lookahead")
    scores /= counts
    k = int(np.argmin(scores))

    # 2. 逐 episode 重写 action（list-of-arrays 赋值，保持每行 (7,) 结构）
    new_cols = []
    for _, g in df_all.groupby("episode_index", sort=True):
        s = np.stack(g["observation.state"].to_numpy())
        L = len(s)
        shifted = s[np.clip(np.arange(L) + k, 0, L - 1)]
        new_cols.append(shifted)
    df_all = df_all.copy()
    shifted_all = np.concatenate(new_cols, axis=0)
    df_all["action"] = [shifted_all[i] for i in range(len(shifted_all))]

    # 3. 原子重写 + stats 重算（保持原文件边界）
    offsets = []
    off = 0
    for p in data_files:
        n = len(pd.read_parquet(p, columns=["index"]))
        offsets.append((p, off, n))
        off += n
    for p, s0, n in offsets:
        _atomic_write_parquet(df_all.iloc[s0:s0 + n], p)
    _recompute_global_stats(str(root / "meta"), df_all)
    return k, len(df_all)

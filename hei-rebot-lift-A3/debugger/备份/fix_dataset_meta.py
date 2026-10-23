#!/usr/bin/env python
"""Fix stale LeRobot v3.0 metadata in lerobot_data/ela3_demo.

Problems fixed:
1. meta/episodes data/chunk_index & data/file_index pointed to per-episode
   files (file-001..file-095) while data was consolidated into
   file-000..file-011 (~1000 rows each).
2. videos/*/to_timestamp held wall-clock durations that do not match the
   actual mp4 durations (length / fps), breaking timestamp-based decoding.
3. meta/stats.json aggregates were computed on a larger, pre-trim dataset.
"""
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).parent / "lerobot_data" / "ela3_demo"
FPS = 20

ep_path = ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
table = pq.read_table(ep_path)
data = table.to_pydict()
N = len(data["episode_index"])

# 1. locate each episode's rows in the consolidated data files
ep_to_file = {}
for f in sorted(glob.glob(str(ROOT / "data" / "chunk-000" / "file-*.parquet"))):
    t = pq.read_table(f, columns=["episode_index"])
    file_idx = int(Path(f).stem.split("-")[1])
    for ep in set(t.column("episode_index").to_pylist()):
        ep_to_file[ep] = file_idx

assert len(ep_to_file) == N, f"found {len(ep_to_file)} episodes in data, meta has {N}"

for i in range(N):
    ep = data["episode_index"][i]
    data["data/chunk_index"][i] = 0
    data["data/file_index"][i] = ep_to_file[ep]
    to_ts = data["length"][i] / FPS
    data["videos/observation.images.cam1/to_timestamp"][i] = to_ts
    data["videos/observation.images.cam2/to_timestamp"][i] = to_ts

new_table = pq.read_table(ep_path).schema.empty_table()
import pyarrow as pa

new_table = pa.Table.from_pydict(data, schema=table.schema)
shutil.copy2(ep_path, str(ep_path) + ".bak")
pq.write_table(new_table, ep_path)
print(f"fixed {N} episodes in {ep_path}")

# 2. recompute aggregate stats from actual data (non-image keys)
frames = []
for f in sorted(glob.glob(str(ROOT / "data" / "chunk-000" / "file-*.parquet"))):
    frames.append(pq.read_table(f))
data_table = pa.concat_tables(frames)


def agg_stats(col):
    arr = np.stack(data_table.column(col).to_pylist()).astype(np.float64)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [arr.shape[0]],
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


stats_path = ROOT / "meta" / "stats.json"
with open(stats_path) as f:
    stats = json.load(f)

for key in ["observation.state", "action", "timestamp"]:
    stats[key] = agg_stats(key)
    print(f"recomputed stats for {key}, count={stats[key]['count']}")

shutil.copy2(stats_path, str(stats_path) + ".bak")
with open(stats_path, "w") as f:
    json.dump(stats, f)
print("done")

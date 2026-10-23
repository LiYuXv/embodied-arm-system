"""轻量 LeRobot v3.0 数据集写入器（不依赖 lerobot/torch）

仅实现录制所需的最小 API，接口与 lerobot 0.6.1 的 LeRobotDataset 对齐：
    create / resume / add_frame / save_episode / finalize
    属性 num_episodes / num_frames

磁盘格式严格对齐 lerobot 0.6.1 (codebase_version="v3.0")：
    meta/info.json            — DatasetInfo 全字段
    meta/tasks.parquet        — index="task", column="task_index"
    meta/stats.json           — 聚合统计（mean/std/min/max/count/q01..q99）
    meta/episodes/chunk-000/file-000.parquet — 每 episode 一行
    data/chunk-XXX/file-YYY.parquet          — 帧数据（每 episode 一个文件）
    videos/{video_key}/chunk-XXX/file-YYY.mp4 — 每 episode 一个视频文件
      （h264/yuv420p，PyAV 实时流式编码，录制期间不落地 PNG）

与官方写入器的差异（对读取端透明）：
- 官方把多个 episode 打包进同一个 parquet/mp4（按大小滚动），本实现每 episode
  独立文件，chunk/file 索引按 episode_index 分配（chunks_size=1000）。
  v3.0 读取端按 episode 元数据逐段寻址，完全兼容。
- 视频 quantile 统计为蓄水池采样近似值；数值特征为精确值。

依赖：numpy / pyarrow / pandas / av（无 torch/transformers）。
"""

import json
import logging
import math
import queue
import threading
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

logger = logging.getLogger(__name__)

CODEBASE_VERSION = "v3.0"
CHUNKS_SIZE = 1000
QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]

# lerobot 0.6.1 DEFAULT_FEATURES（录制管道自动填充的元特征）
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}

DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-000/file-000.parquet"
TASKS_PATH = "meta/tasks.parquet"
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"


class _CameraEncoderThread(threading.Thread):
    """单摄像头编码线程：队列取 RGB 帧 → PyAV 编码 → MP4（每 episode 一个文件）。

    顺带做降采样统计（sum/sumsq/min/max/count + 蓄水池采样用于分位数），
    归一化到 [0,1] 在汇总时进行。
    """

    STATS_STEP = 8          # 统计降采样步长（720→90px）
    RESERVOIR_CAP = 20000   # 每通道蓄水池样本上限

    def __init__(self, video_path, fps, vcodec="h264", g=2, crf=30,
                 encoder_threads=2, parent_dataset=None):
        super().__init__(daemon=True)
        self.video_path = Path(video_path)
        self.fps = fps
        self.vcodec = vcodec
        self.options = {"g": str(g)}
        if crf is not None and "nvenc" not in vcodec:
            self.options["crf"] = str(crf)
        if encoder_threads and "nvenc" not in vcodec:
            self.options["threads"] = str(encoder_threads)
        # 弱性能工控机：软编码必须加速预设，否则 720p 多路无法实时
        if "nvenc" not in vcodec and vcodec in ("h264", "libx264"):
            self.options.setdefault("preset", "veryfast")
        self._queue = queue.Queue(maxsize=60)
        self.error = None
        self.frame_count = 0
        # 统计状态（uint8 域）
        self._sum = None
        self._sumsq = None
        self._min = None
        self._max = None
        self._count = 0
        self._reservoir = None  # (cap, 3) uint8
        self._seen = 0

    def feed(self, frame_rgb):
        """送入一帧 RGB uint8 (H,W,3)。队列满时丢帧（与官方行为一致）。

        编码线程已崩溃时立即抛错（向上冒泡到录制线程 error 信号），
        避免静默丢帧直到停止保存时才在 finish() 死锁。
        """
        if self.error is not None:
            raise RuntimeError(f"视频编码失败: {self.error}")
        if not self.is_alive():
            raise RuntimeError(f"视频编码线程已退出: {self.video_path}")
        try:
            self._queue.put_nowait(frame_rgb.copy())
        except queue.Full:
            pass

    def _update_stats(self, frame):
        img = frame[::self.STATS_STEP, ::self.STATS_STEP, :].astype(np.float64)
        flat = img.reshape(-1, 3)
        if self._sum is None:
            self._sum = flat.sum(axis=0)
            self._sumsq = (flat ** 2).sum(axis=0)
            self._min = flat.min(axis=0)
            self._max = flat.max(axis=0)
            self._reservoir = np.zeros((self.RESERVOIR_CAP, 3), dtype=np.uint8)
        else:
            self._sum += flat.sum(axis=0)
            self._sumsq += (flat ** 2).sum(axis=0)
            self._min = np.minimum(self._min, flat.min(axis=0))
            self._max = np.maximum(self._max, flat.max(axis=0))
        self._count += flat.shape[0]
        # 蓄水池近似：每帧随机抽 512 行写入循环缓冲（向量化，避免逐行 Python 循环）
        k = min(512, flat.shape[0])
        idx = np.random.randint(0, flat.shape[0], size=k)
        rows = flat[idx].astype(np.uint8)
        start = self._seen % self.RESERVOIR_CAP
        end = start + rows.shape[0]
        if end <= self.RESERVOIR_CAP:
            self._reservoir[start:end] = rows
        else:
            first = self.RESERVOIR_CAP - start
            self._reservoir[start:] = rows[:first]
            self._reservoir[:end - self.RESERVOIR_CAP] = rows[first:]
        self._seen += rows.shape[0]

    def get_stats(self):
        """返回 [0,1] 归一化的每通道统计 {mean,std,min,max,count,q01..q99}，形状 (3,1,1)"""
        if self._count == 0 or self._sum is None:
            return None
        mean = self._sum / self._count
        var = np.maximum(self._sumsq / self._count - mean ** 2, 0.0)
        std = np.sqrt(var)
        n = min(self._seen, self.RESERVOIR_CAP)
        sample = self._reservoir[:n].astype(np.float64) / 255.0
        qs = np.quantile(sample, QUANTILES, axis=0)  # (5, 3)

        def _c(x):
            return np.asarray(x, dtype=np.float64).reshape(3, 1, 1)

        return {
            "mean": _c(mean / 255.0), "std": _c(std / 255.0),
            "min": _c(self._min / 255.0), "max": _c(self._max / 255.0),
            "count": np.array([self._count]),
            "q01": _c(qs[0]), "q10": _c(qs[1]), "q50": _c(qs[2]),
            "q90": _c(qs[3]), "q99": _c(qs[4]),
        }

    def run(self):
        container = None
        stream = None
        try:
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            container = av.open(str(self.video_path), "w")
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                if stream is None:
                    h, w = frame.shape[:2]
                    stream = container.add_stream(self.vcodec, rate=self.fps,
                                                  options=self.options)
                    stream.pix_fmt = "yuv420p"
                    stream.width = w
                    stream.height = h
                    stream.time_base = Fraction(1, self.fps)
                vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
                vf.pts = self.frame_count
                vf.time_base = Fraction(1, self.fps)
                for packet in stream.encode(vf):
                    container.mux(packet)
                self._update_stats(frame)
                self.frame_count += 1
            if stream is not None:
                for packet in stream.encode():
                    container.mux(packet)
        except Exception as e:
            self.error = str(e)
            logger.error(f"[LiteDataset] 编码线程错误 ({self.video_path}): {e}")
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass

    def finish(self):
        """收尾编码线程。编码线程崩溃且队列满时裸 put(None) 会永久阻塞
        （消费者已死，队列永远不清空），因此投递哨兵用带超时的循环，
        线程死亡立即报错而非死锁界面。"""
        if self.error is not None:
            raise RuntimeError(f"视频编码失败: {self.error}")
        deadline = time.monotonic() + 35
        while True:
            if not self.is_alive():
                raise RuntimeError(
                    f"视频编码线程已退出: {self.error or self.video_path}")
            try:
                self._queue.put(None, timeout=0.5)
                break
            except queue.Full:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "视频编码线程无响应（队列持续满），保存中止")
        self.join(timeout=30)
        if self.error:
            raise RuntimeError(f"视频编码失败: {self.error}")


class LiteLeRobotDataset:
    """LeRobot v3.0 数据集轻量写入器（API 对齐 lerobot 0.6.1 LeRobotDataset）"""

    def __init__(self):
        self.root = None
        self.repo_id = ""
        self.fps = 20
        self.features = {}          # 用户特征 + DEFAULT_FEATURES
        self.robot_type = None
        self._video_keys = []
        self._vcodec = "h264"
        self._encoder_g = 2
        self._encoder_crf = 30
        self._encoder_threads = 4
        self._tasks = []            # task 字符串列表，索引即 task_index
        self._episodes = []         # 每 episode 的元数据 dict
        self._total_frames = 0
        self._stats = None          # 聚合统计
        self._buf = None            # 当前 episode 缓冲
        self._encoders = {}
        self._video_info_written = False
        self._finalized = False

    # ------------------------------------------------------------------ API

    @classmethod
    def create(cls, repo_id, fps, features, root=None, robot_type=None,
               use_videos=True, streaming_encoding=True, vcodec="h264",
               rgb_encoder=None, encoder_threads=4, g=None, **kwargs):
        """创建新数据集（root 必须不存在）。多余 kwargs 静默忽略以兼容官方签名。"""
        obj = cls()
        obj.root = Path(root)
        if obj.root.exists():
            raise FileExistsError(f"数据集目录已存在: {obj.root}")
        obj.root.mkdir(parents=True)
        obj.repo_id = repo_id
        obj.fps = int(fps)
        obj.robot_type = robot_type
        obj.features = {**{k: dict(v) for k, v in features.items()},
                        **{k: dict(v) for k, v in DEFAULT_FEATURES.items()}}
        obj._video_keys = [k for k, v in obj.features.items()
                           if v.get("dtype") == "video"]
        if obj._video_keys and not use_videos:
            raise ValueError("features 含 video 键但 use_videos=False")
        # 编码参数：兼容 rgb_encoder=RGBEncoderConfig(vcodec=..., g=...) 与 vcodec=...
        if rgb_encoder is not None:
            obj._vcodec = getattr(rgb_encoder, "vcodec", vcodec)
            obj._encoder_g = getattr(rgb_encoder, "g", 2) or 2
            obj._encoder_crf = getattr(rgb_encoder, "crf", 30)
        else:
            obj._vcodec = vcodec
            # NVENC 要求 GOP 长度 > B帧数+1，默认 g=2 会初始化失败，须 g=30
            obj._encoder_g = g or (30 if "nvenc" in vcodec else 2)
        obj._encoder_threads = encoder_threads or 4
        obj._buf = obj._new_buffer()
        obj._write_info()
        logger.info(f"[LiteDataset] 创建数据集 {repo_id} @ {obj.root}")
        return obj

    @classmethod
    def resume(cls, repo_id, root=None, streaming_encoding=True, vcodec=None,
               rgb_encoder=None, encoder_threads=4, g=None, **kwargs):
        """加载已有数据集继续追加录制。"""
        obj = cls()
        obj.root = Path(root)
        with open(obj.root / INFO_PATH, "r") as f:
            info = json.load(f)
        if info.get("codebase_version") != CODEBASE_VERSION:
            raise ValueError(f"不支持的数据集版本: {info.get('codebase_version')}")
        obj.repo_id = repo_id
        obj.fps = info["fps"]
        obj.robot_type = info.get("robot_type")
        obj.features = {k: dict(v) for k, v in info["features"].items()}
        for v in obj.features.values():
            if isinstance(v.get("shape"), list):
                v["shape"] = tuple(v["shape"])
        obj._video_keys = [k for k, v in obj.features.items()
                           if v.get("dtype") == "video"]
        obj._total_frames = info.get("total_frames", 0)
        # 编码器需与已有数据一致（调用方传入）
        if rgb_encoder is not None:
            obj._vcodec = getattr(rgb_encoder, "vcodec", "h264")
            obj._encoder_g = getattr(rgb_encoder, "g", 2) or 2
            obj._encoder_crf = getattr(rgb_encoder, "crf", 30)
        else:
            obj._vcodec = vcodec or "h264"
            obj._encoder_g = g or (30 if "nvenc" in obj._vcodec else 2)
        obj._encoder_threads = encoder_threads or 4
        obj._video_info_written = obj._total_frames > 0
        # 加载 tasks
        tasks_file = obj.root / TASKS_PATH
        if tasks_file.exists():
            import pandas as pd
            df = pd.read_parquet(tasks_file)
            df = df.sort_values("task_index")
            obj._tasks = list(df.index.astype(str))
        # 加载 episodes 元数据
        ep_file = obj.root / EPISODES_PATH
        if ep_file.exists():
            import pandas as pd
            df = pd.read_parquet(ep_file)
            obj._episodes = df.to_dict("records")
        obj._buf = obj._new_buffer()
        logger.info(f"[LiteDataset] 加载数据集 {repo_id}: episodes={obj.num_episodes}, "
                    f"frames={obj.num_frames}")
        return obj

    @property
    def num_episodes(self):
        return len(self._episodes)

    @property
    def num_frames(self):
        return self._total_frames

    def add_frame(self, frame: dict):
        """添加一帧：图像键送编码线程，数值键入缓冲。task 必填。"""
        buf = self._buf
        frame_index = buf["size"]
        timestamp = frame_index / self.fps
        task = frame.pop("task", self._tasks[-1] if self._tasks else "default")

        # 首帧启动编码线程
        if frame_index == 0 and not self._encoders:
            ep_idx = self.num_episodes
            chunk_idx, file_idx = ep_idx // CHUNKS_SIZE, ep_idx % CHUNKS_SIZE
            for key in self._video_keys:
                vpath = self.root / VIDEO_PATH_TEMPLATE.format(
                    video_key=key, chunk_index=chunk_idx, file_index=file_idx)
                enc = _CameraEncoderThread(
                    vpath, self.fps, vcodec=self._vcodec, g=self._encoder_g,
                    crf=self._encoder_crf, encoder_threads=self._encoder_threads)
                enc.start()
                self._encoders[key] = enc

        for key, value in frame.items():
            if key in self._video_keys:
                img = np.asarray(value)
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                if img.shape[0] in (1, 3) and img.ndim == 3:  # CHW -> HWC
                    img = img.transpose(1, 2, 0)
                if key in self._encoders:
                    self._encoders[key].feed(img)
            elif key in self.features:
                buf[key].append(np.asarray(value, dtype=np.float32))

        buf["frame_index"].append(frame_index)
        buf["timestamp"].append(timestamp)
        buf["task"].append(task)
        buf["size"] += 1

    def save_episode(self, episode_data=None, parallel_encoding=True):
        """保存当前 episode：收尾视频、写 parquet、更新元数据与统计。"""
        buf = episode_data if episode_data is not None else self._buf
        length = buf.get("size", 0)
        if length == 0:
            logger.warning("[LiteDataset] 空 episode，跳过保存")
            return
        ep_idx = self.num_episodes
        chunk_idx, file_idx = ep_idx // CHUNKS_SIZE, ep_idx % CHUNKS_SIZE

        # 1. 收尾编码线程
        video_meta = {}
        ep_stats = {}
        for key, enc in self._encoders.items():
            enc.finish()
            vpath = enc.video_path
            duration = self._video_duration(vpath)
            video_meta[f"videos/{key}/chunk_index"] = chunk_idx
            video_meta[f"videos/{key}/file_index"] = file_idx
            video_meta[f"videos/{key}/from_timestamp"] = 0.0
            video_meta[f"videos/{key}/to_timestamp"] = duration
            st = enc.get_stats()
            if st is not None:
                ep_stats[key] = st
        self._encoders = {}

        # 2. 数值特征统计 + 数据 parquet
        tasks = buf["task"]
        episode_tasks = sorted(set(tasks))
        columns = {}
        import pyarrow as pa
        schema_fields = []
        for key, ft in self.features.items():
            if ft.get("dtype") == "video":
                continue
            if key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
                continue  # 元列后面统一加
            arr = np.stack(buf[key])
            if tuple(ft["shape"]) == (1,):
                arr = arr.reshape(length)
            columns[key] = arr
            if ft["dtype"] in ("float32", "float64", "int64", "int32"):
                ep_stats[key] = self._numeric_stats(arr.reshape(length, -1))
            if arr.ndim == 1:
                schema_fields.append((key, self._pa_scalar(ft["dtype"])))
            else:
                schema_fields.append((key, pa.list_(self._pa_scalar(ft["dtype"]))))

        # 3. tasks 注册（需在 task_index 计算之前）
        for t in episode_tasks:
            if t not in self._tasks:
                self._tasks.append(t)

        task_indices = np.array([self._task_index(t) for t in tasks], dtype=np.int64)
        meta_cols = {
            "timestamp": np.asarray(buf["timestamp"], dtype=np.float32),
            "frame_index": np.asarray(buf["frame_index"], dtype=np.int64),
            "episode_index": np.full(length, ep_idx, dtype=np.int64),
            "index": np.arange(self._total_frames, self._total_frames + length,
                               dtype=np.int64),
            "task_index": task_indices,
        }
        meta_schema = [("timestamp", pa.float32()), ("frame_index", pa.int64()),
                       ("episode_index", pa.int64()), ("index", pa.int64()),
                       ("task_index", pa.int64())]

        all_cols = {**columns, **meta_cols}
        schema = pa.schema(schema_fields + meta_schema)
        arrays = []
        for name, _ in schema_fields + meta_schema:
            arr = all_cols[name]
            if arr.ndim == 1:
                arrays.append(pa.array(arr))
            else:
                arrays.append(pa.array([row for row in arr]))
        table = pa.Table.from_arrays(arrays, schema=schema)
        data_path = self.root / DATA_PATH_TEMPLATE.format(
            chunk_index=chunk_idx, file_index=file_idx)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        import pyarrow.parquet as pq
        pq.write_table(table, data_path, compression="snappy")

        # 3. tasks 落盘
        self._write_tasks()

        # 4. episodes 元数据
        ep_meta = {
            "episode_index": ep_idx,
            "tasks": episode_tasks,
            "length": length,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": self._total_frames,
            "dataset_to_index": self._total_frames + length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        ep_meta.update(video_meta)
        self._episodes.append(ep_meta)
        self._write_episodes()

        # 5. info.json：计数 + 首个 episode 后补视频 info
        self._total_frames += length
        if not self._video_info_written:
            self._fill_video_info()
            self._video_info_written = True
        self._write_info()

        # 6. 聚合统计
        self._stats = self._aggregate_stats(self._stats, ep_stats)
        self._write_stats()

        logger.info(f"[LiteDataset] episode {ep_idx} 已保存: {length} 帧")
        self._buf = self._new_buffer()

    def finalize(self):
        """收尾（本实现即时落盘，finalize 仅作标记，幂等）"""
        self._finalized = True
        self._write_info()

    # ------------------------------------------------------------- internal

    @staticmethod
    def _pa_scalar(dtype):
        import pyarrow as pa
        return {"float32": pa.float32(), "float64": pa.float64(),
                "int64": pa.int64(), "int32": pa.int32()}[dtype]

    def _new_buffer(self):
        buf = {"size": 0, "task": [], "frame_index": [], "timestamp": []}
        for key, ft in self.features.items():
            if ft.get("dtype") != "video":
                buf[key] = []
        return buf

    def _task_index(self, task):
        return self._tasks.index(task)

    @staticmethod
    def _video_duration(path):
        try:
            import av
            with av.open(str(path)) as c:
                s = c.streams.video[0]
                if s.duration is not None and s.time_base is not None:
                    return float(s.duration * s.time_base)
                return float(s.frames / s.base_rate) if s.base_rate else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _numeric_stats(arr):
        """arr: (N, D) → mean/std/min/max/count/q* 每维 (D,)；与官方 shape 对齐"""
        def _v(x):
            return np.asarray(x, dtype=np.float64)
        qs = np.quantile(arr, QUANTILES, axis=0)
        return {
            "mean": _v(arr.mean(axis=0)), "std": _v(arr.std(axis=0)),
            "min": _v(arr.min(axis=0)), "max": _v(arr.max(axis=0)),
            "count": np.array([arr.shape[0]]),
            "q01": _v(qs[0]), "q10": _v(qs[1]), "q50": _v(qs[2]),
            "q90": _v(qs[3]), "q99": _v(qs[4]),
        }

    def _aggregate_stats(self, old, new):
        if old is None:
            return new
        merged = {}
        for key in new:
            if key not in old:
                merged[key] = new[key]
                continue
            o, n = old[key], new[key]
            co, cn = int(o["count"][0]), int(n["count"][0])
            total = co + cn
            mean = (o["mean"] * co + n["mean"] * cn) / total
            var = ((o["std"] ** 2 + o["mean"] ** 2) * co
                   + (n["std"] ** 2 + n["mean"] ** 2) * cn) / total - mean ** 2
            merged[key] = {
                "mean": mean, "std": np.sqrt(np.maximum(var, 0.0)),
                "min": np.minimum(o["min"], n["min"]),
                "max": np.maximum(o["max"], n["max"]),
                "count": np.array([total]),
                # 分位数按帧数加权近似
                **{q: (o[q] * co + n[q] * cn) / total
                   for q in ("q01", "q10", "q50", "q90", "q99")},
            }
        return merged

    def _fill_video_info(self):
        """首 episode 编码完成后回填视频参数到 features（对齐 update_video_info）"""
        for key in self._video_keys:
            ft = self.features[key]
            h, w = ft["shape"][0], ft["shape"][1]
            codec = {"h264_nvenc": "h264", "hevc_nvenc": "hevc"}.get(
                self._vcodec, self._vcodec)
            ft["info"] = {
                "video.height": h, "video.width": w,
                "video.codec": codec, "video.pix_fmt": "yuv420p",
                "video.fps": self.fps, "video.channels": 3,
                "has_audio": False,
                "video.g": self._encoder_g, "video.crf": self._encoder_crf,
                "video.preset": None,
                "video.fast_decode": 0, "video.video_backend": "pyav",
                "video.extra_options": {},
                "is_depth_map": False,
            }

    def _write_info(self):
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": self.num_episodes,
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": self.num_episodes * len(self._video_keys),
            "total_chunks": math.ceil(max(self.num_episodes, 1) / CHUNKS_SIZE),
            "chunks_size": CHUNKS_SIZE,
            "fps": self.fps,
            "splits": {"train": f"0:{self.num_episodes}"},
            "data_path": DATA_PATH_TEMPLATE,
            "video_path": VIDEO_PATH_TEMPLATE,
            "features": {
                k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                    for kk, vv in v.items()}
                for k, v in self.features.items()
            },
        }
        path = self.root / INFO_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

    def _write_tasks(self):
        import pandas as pd
        df = pd.DataFrame({"task_index": range(len(self._tasks))},
                          index=pd.Index(self._tasks, name="task"))
        path = self.root / TASKS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def _write_episodes(self):
        import pandas as pd
        df = pd.DataFrame(self._episodes)
        path = self.root / EPISODES_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def _write_stats(self):
        if self._stats is None:
            return
        serialized = {
            key: {sk: np.asarray(sv).tolist() for sk, sv in stats.items()}
            for key, stats in self._stats.items()
        }
        with open(self.root / STATS_PATH, "w") as f:
            json.dump(serialized, f)

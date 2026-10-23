# -*- coding: utf-8 -*-
"""离线数据集加载模块：OfflineDataset（LeRobot v3 数据集读取）与 EpisodeSampler。"""
import json
import threading
import pandas as pd
try:
    import torch
    _SamplerBase = torch.utils.data.Sampler
except ImportError:
    # torch 仅用于 EpisodeSampler 基类，纯可视化环境（无 torch）下退化为 object
    _SamplerBase = object
import cv2
from pathlib import Path


# ==================== 离线数据集加载器 ====================
class OfflineDataset:
    """直接从本地文件加载 LeRobot 数据集，无需联网，不依赖 LeRobotDataset"""
    
    def __init__(self, root_path):
        self.root = Path(root_path)
        self.meta_dir = self.root / "meta"
        self.data_dir = self.root / "data"
        self.images_dir = self.root / "images"
        self.videos_dir = self.root / "videos"
        
        # 加载元数据
        with open(self.meta_dir / "info.json") as f:
            self.info = json.load(f)
        
        with open(self.meta_dir / "stats.json") as f:
            self.stats = json.load(f)
        
        # 加载 episodes
        self.episodes = self._load_episodes()
        
        # 加载任务
        self.tasks = self._load_tasks()
        
        # 加载所有数据
        self.data_df = self._load_data()
        
        # 提取特征信息
        self.features = self.info.get("features", {})
        self.fps = self.info.get("fps", 20)
        self.num_frames = len(self.data_df)
        
        # 缓存视频读取器: {(episode_index, camera_name): (cap, current_idx)}
        self._video_caps = {}
        # FFmpeg 解码器非线程安全：播放线程与主线程(滑条预览)会并发访问
        # 同一个缓存 VideoCapture，必须用锁序列化，否则 libavcodec 断言闪退
        self._video_lock = threading.RLock()
        
    def get_video_path(self, episode_index, camera_name):
        """根据 episodes 元数据查找视频文件路径"""
        if episode_index < 0 or episode_index >= len(self.episodes):
            return None
        ep = self.episodes[episode_index]
        
        # LeRobot 0.4.4 video 路径格式: videos/{camera_name}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
        chunk_key = f"videos/{camera_name}/chunk_index"
        file_key = f"videos/{camera_name}/file_index"
        if chunk_key in ep and file_key in ep:
            chunk_idx = int(ep[chunk_key])
            file_idx = int(ep[file_key])
            video_path = self.videos_dir / camera_name / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
            if video_path.exists():
                return str(video_path)
        
        # 兜底：尝试直接查找 videos/{camera_name}/episode_{episode_index:06d}.mp4
        patterns = [
            self.videos_dir / camera_name / f"episode_{episode_index:06d}.mp4",
            self.videos_dir / camera_name / f"episode-{episode_index:06d}.mp4",
        ]
        for p in patterns:
            if p.exists():
                return str(p)
        
        return None
    
    def get_video_frame(self, episode_index, frame_index, camera_name):
        """从视频中读取指定帧，返回 RGB numpy 数组。

        缓存一个持久的 cv2.VideoCapture：
        顺序读取（N → N+1）直接 cap.read() 解码下一帧，最快；
        随机访问（拖动滑条/跳帧）在同一个 cap 上 set(CAP_PROP_POS_FRAMES)
        做快速 seek，避免每帧重新打开视频文件的巨大开销。
        """
        video_path = self.get_video_path(episode_index, camera_name)
        if video_path is None:
            return None

        key = (episode_index, camera_name)
        with self._video_lock:
            return self._read_video_frame_locked(key, video_path, episode_index, frame_index)

    def _read_video_frame_locked(self, key, video_path, episode_index, frame_index):
        """在 _video_lock 保护下完成实际读帧（调用方必须已持有锁）。"""
        cached = self._video_caps.get(key)

        try:
            if cached is not None:
                cap, current_idx = cached
                if frame_index == current_idx + 1:
                    # 顺序读取：直接解码下一帧，最快路径
                    ret, frame_bgr = cap.read()
                    if ret and frame_bgr is not None:
                        self._video_caps[key] = (cap, frame_index)
                        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    # 读失败（可能到末尾）：关闭后走重新打开逻辑
                    self._close_video_reader(key)
                else:
                    # 跳帧/倒退/重复同一帧：同一 cap 上快速 seek
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame_bgr = cap.read()
                    if ret and frame_bgr is not None:
                        self._video_caps[key] = (cap, frame_index)
                        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    self._close_video_reader(key)
                    return None

            # 首次访问：打开视频并缓存 cap，后续帧复用
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                return None
            if frame_index > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                cap.release()
                return None
            self._video_caps[key] = (cap, frame_index)
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[OfflineDataset] 读取视频帧失败 ep={episode_index}, frame={frame_index}: {e}")
            self._close_video_reader(key)
            return None

    def _close_video_reader(self, key):
        """关闭指定视频 reader（须在 _video_lock 保护下调用）"""
        cached = self._video_caps.pop(key, None)
        if cached is not None:
            reader, _ = cached
            for closer in (getattr(reader, "close", None), getattr(reader, "release", None)):
                if closer is not None:
                    try:
                        closer()
                    except Exception:
                        pass
                    break

    def release_video_caps(self):
        """释放所有视频读取器"""
        with self._video_lock:
            for key in list(self._video_caps.keys()):
                self._close_video_reader(key)
            self._video_caps.clear()
    
    def is_video_feature(self, camera_name):
        """判断指定 camera 是否为 video 类型"""
        feature = self.features.get(camera_name, {})
        return feature.get("dtype") == "video" or feature.get("type") == "video"

    def task_name_for_index(self, task_index, default=""):
        """按 task_index 值查找任务名。

        严禁用列表位置索引 self.tasks：批量改名“并入已有任务”会删除中间任务行，
        剩余任务的 task_index 与列表位置错位（如 default_task(0) 被并入后，
        剩余 task_index 从 1 开始，列表位置却从 0 开始）。
        """
        try:
            task_index = int(task_index)
        except (TypeError, ValueError):
            return default
        for t in self.tasks:
            if int(t.get("task_index", -1)) == task_index:
                return t.get("task", default)
        return default

    def _load_episodes(self):
        """加载 episodes 元数据"""
        # 尝试 parquet 格式
        ep_parquet_dir = self.meta_dir / "episodes"
        if ep_parquet_dir.exists():
            ep_files = sorted(ep_parquet_dir.rglob("*.parquet"))
            if ep_files:
                return pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True).to_dict("records")
        
        # 尝试 jsonl 格式
        ep_jsonl_path = self.meta_dir / "episodes.jsonl"
        if ep_jsonl_path.exists():
            with open(ep_jsonl_path) as f:
                return [json.loads(line) for line in f if line.strip()]
        
        # 从 data_df 推断 episodes
        return self._infer_episodes_from_data()

    def get_image_path(self, episode_index, frame_index, camera_name):
        """获取图像文件路径"""
        # 尝试 images/ 目录结构
        img_dir = self.images_dir / camera_name / f"episode_{episode_index:06d}"
        if img_dir.exists():
            img_path = img_dir / f"frame_{frame_index:06d}.png"
            if img_path.exists():
                return str(img_path)
        
        # 尝试其他命名格式
        patterns = [
            self.images_dir / camera_name / f"episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png",
            self.images_dir / camera_name / f"episode_{episode_index:06d}" / f"{frame_index:06d}.png",
            self.images_dir / f"{camera_name}_episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png",
        ]
        for p in patterns:
            if p.exists():
                return str(p)
        
        return None
    
    def _infer_episodes_from_data(self):
        """从数据文件推断 episode 信息"""
        df = self._load_data()
        episodes = []
        for ep_idx in sorted(df["episode_index"].unique()):
            ep_df = df[df["episode_index"] == ep_idx]
            episodes.append({
                "episode_index": int(ep_idx),
                "length": len(ep_df),
            })
        return episodes
    
    def _load_tasks(self):
        """加载任务数据 - LeRobot 0.4.4 tasks.parquet 格式: index=task_name, task_index=col"""
        tasks_path = self.meta_dir / "tasks.parquet"
        if tasks_path.exists():
            try:
                df = pd.read_parquet(tasks_path)
                # LeRobot 0.4.4 格式: task_name 是 index, task_index 是列值
                # 转换为 [{"task_index": 0, "task": "open"}, ...]
                result = []
                for task_name, row in df.iterrows():
                    result.append({
                        "task_index": int(row["task_index"]),
                        "task": str(task_name),
                    })
                return result
            except Exception as e:
                print(f"[OfflineDataset] 加载 tasks 失败: {e}")
        return []
    
    def _load_data(self):
        """加载所有帧数据，跳过损坏的 parquet 文件"""
        data_files = sorted(self.data_dir.rglob("*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"找不到数据 parquet 文件: {self.data_dir}")

        dfs = []
        for p in data_files:
            try:
                dfs.append(pd.read_parquet(p))
            except Exception as e:
                print(f"[OfflineDataset] 跳过损坏的数据文件 {p.name}: {e}")
                # 将损坏文件重命名为 .corrupted 备份，避免下次再次尝试读取
                corrupted_path = p.with_suffix(p.suffix + ".corrupted")
                if not corrupted_path.exists():
                    try:
                        p.rename(corrupted_path)
                        print(f"[OfflineDataset] 已备份为 {corrupted_path.name}")
                    except Exception as rename_err:
                        print(f"[OfflineDataset] 备份失败: {rename_err}")
        if not dfs:
            raise FileNotFoundError(f"所有数据 parquet 文件均损坏: {self.data_dir}")
        return pd.concat(dfs, ignore_index=True)
    
    def get_episode_frames(self, episode_index):
        """获取指定 episode 的所有帧数据"""
        ep = self.episodes[episode_index]
        
        # 新格式：有 dataset_from_index 和 dataset_to_index
        if "dataset_from_index" in ep and "dataset_to_index" in ep:
            from_idx = int(ep["dataset_from_index"])
            to_idx = int(ep["dataset_to_index"])
            return self.data_df.iloc[from_idx:to_idx].reset_index(drop=True)
        
        # 旧格式：只有 length，需要累加计算
        if "length" in ep:
            from_idx = 0
            for i in range(episode_index):
                from_idx += self.episodes[i]["length"]
            to_idx = from_idx + ep["length"]
            return self.data_df.iloc[from_idx:to_idx].reset_index(drop=True)
        
        # 直接从 episode_index 列筛选
        return self.data_df[self.data_df["episode_index"] == episode_index].reset_index(drop=True)
    
    def get_image_path(self, episode_index, frame_index, camera_name):
        """获取图像文件路径"""
        # 尝试 images/ 目录结构
        img_dir = self.images_dir / camera_name / f"episode_{episode_index:06d}"
        if img_dir.exists():
            img_path = img_dir / f"frame_{frame_index:06d}.png"
            if img_path.exists():
                return str(img_path)
        
        # 尝试其他命名格式
        patterns = [
            self.images_dir / camera_name / f"episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png",
            self.images_dir / camera_name / f"episode_{episode_index:06d}" / f"{frame_index:06d}.png",
            self.images_dir / f"{camera_name}_episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png",
        ]
        for p in patterns:
            if p.exists():
                return str(p)
        
        return None
    
    def get_image_resolution(self, camera_name):
        """从 features 元数据获取指定 camera 的分辨率，返回 (width, height)"""
        feature = self.features.get(camera_name)
        if feature is None:
            return None
        shape = feature.get("shape")
        names = feature.get("names")
        if shape is None or len(shape) < 2:
            return None
        
        # 优先使用 names 字段定位 height/width
        if isinstance(names, (list, tuple)) and len(names) == len(shape):
            names = [str(n).lower() if n else "" for n in names]
            if "height" in names and "width" in names:
                h = int(shape[names.index("height")])
                w = int(shape[names.index("width")])
                return (w, h)
            if "h" in names and "w" in names:
                h = int(shape[names.index("h")])
                w = int(shape[names.index("w")])
                return (w, h)
        
        # 推断：若最后一个维度很小（1-4），视为 channels，shape 为 [H, W, C]
        if len(shape) == 3 and 1 <= int(shape[-1]) <= 4:
            h, w = int(shape[0]), int(shape[1])
            return (w, h)
        
        # 若第一个维度很小，视为 [C, H, W]
        if len(shape) == 3 and 1 <= int(shape[0]) <= 4:
            h, w = int(shape[1]), int(shape[2])
            return (w, h)
        
        # 兜底：最后两个维度作为 H, W
        h, w = int(shape[-2]), int(shape[-1])
        return (w, h)
    
    def get_all_image_resolutions(self):
        """获取所有图像/视频特征的分辨率，返回 {camera_name: (width, height)}"""
        res = {}
        for name, feature in self.features.items():
            if feature.get("dtype") in ["image", "video"] or feature.get("type") in ["image", "video"] or "image" in name.lower():
                shape = feature.get("shape")
                if shape is None or len(shape) < 2:
                    continue
                try:
                    resolution = self.get_image_resolution(name)
                    if resolution is not None:
                        res[name] = resolution
                except (TypeError, ValueError):
                    pass
        return res
    
    def __len__(self):
        return self.num_frames


# ==================== EpisodeSampler ====================
class EpisodeSampler(_SamplerBase):
    """单个 episode 的采样器"""
    def __init__(self, dataset: OfflineDataset, episode_index: int):
        ep = dataset.episodes[episode_index]
        if "dataset_from_index" in ep and "dataset_to_index" in ep:
            from_idx = ep["dataset_from_index"]
            to_idx = ep["dataset_to_index"]
        else:
            from_idx = 0
            for i in range(episode_index):
                from_idx += dataset.episodes[i]["length"]
            to_idx = from_idx + ep["length"]
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

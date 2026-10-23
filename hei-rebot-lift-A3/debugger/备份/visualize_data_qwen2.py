#!/usr/bin/env python3
"""
LeRobot 数据可视化工具 (PyQt6 界面版) - 离线本地数据加载

功能:
  1. 启动后显示已有数据集列表
  2. 选择数据集后显示 Episode 列表
  3. 显示数据集总信息
  4. 选择 Episode 后播放/暂停/重新播放
  5. 支持自动播放所有 episode

用法:
  python3 visualize_data_qt.py --root ./lerobot_data
"""

import sys
import os
import json
import glob
import time
import math
import shutil
import logging
import threading
import warnings
from collections import OrderedDict
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path

# 确保可导入 el_a3_sdk 与 debugger 包（直接运行本脚本时需要）
_SDK_ROOT = Path(__file__).resolve().parent.parent
if str(_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SDK_ROOT))
if str(_SDK_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_SDK_ROOT.parent))
_DBG_DIR = Path(__file__).resolve().parent
if str(_DBG_DIR) not in sys.path:
    sys.path.insert(0, str(_DBG_DIR))

# 自动语言标注模块（子任务分段 + 颜色 + 写回数据集）
from auto_language_annotate import (
    color_for_index, segment_fixed_count,
    LanguageAnnotationStore, LabelLibrary, write_language_to_dataset,
)

# RECAP 奖励标注模块
from annotation_store import (
    AnnotationStore as RewardAnnotationStore,
    LABEL_COLORS as RECAP_LABEL_COLORS,
    LABEL_ADVANTAGE as RECAP_LABEL_ADVANTAGE,
    ALL_LABELS as RECAP_ALL_LABELS,
)

# 数据集合并模块（第3分页复用其后台合并线程与格式兼容性预检）
from merge_lerobot_datasets import MergeWorker, preflight_check

logger = logging.getLogger("debugger")
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QSplitter, QMessageBox, QFrame, QHeaderView,
    QTableWidget, QTableWidgetItem, QGroupBox, QComboBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QStatusBar, QScrollArea, QSlider, QLineEdit,
    QSizePolicy, QDialog, QTextEdit, QSpinBox, QTabWidget, QAbstractItemView,
    QListWidget, QListWidgetItem, QPlainTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition, QPoint, QPointF, QTimer
from PyQt6.QtGui import QFont, QImage, QPixmap, QPainter, QBrush, QPen, QColor, QPolygonF

# Matplotlib 嵌入 PyQt6
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def _setup_matplotlib_chinese_font():
    """挑选系统中支持中文的字体，避免图例/标题中文显示为方框。"""
    candidates = [
        'WenQuanYi Micro Hei',
        'Noto Sans CJK SC',
        'Source Han Sans SC',
        'SimHei',
        'Microsoft YaHei',
        'PingFang SC',
        'Heiti SC',
        'Noto Sans CJK JP',
        'Noto Serif CJK JP',
        'AR PL UMing CN',
        'AR PL UKai CN',
        'Droid Sans Fallback',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break

    # 设置 sans-serif 字体列表：优先中文候选，其次 DejaVu Sans 兜底
    if chosen:
        matplotlib.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 抑制字体缺字警告（使用 font fallback 时仍可能产生大量噪音）
    warnings.filterwarnings(
        'ignore',
        message='missing from font',
        category=UserWarning,
    )


# 模块加载时即设置好中文字体
_setup_matplotlib_chinese_font()


# ==================== 逆运动学 (IK) 与 3D 机械臂可视化 ====================
# 尝试加载 pinocchio 运动学，失败则禁用机械臂反解功能
_KIN = None
try:
    from el_a3_sdk.kinematics import ELA3Kinematics
    from el_a3_sdk.data_types import ArmEndPose
    _KIN = ELA3Kinematics()
    logger.info("ELA3Kinematics 加载成功，机械臂关节反解可用")
except Exception as _ik_err:
    logger.warning(f"未加载逆运动学模块，机械臂反解不可用: {_ik_err}")

# pinocchio Model/Data 非线程安全：后台 IK 线程复用同一个 _KIN 单例，
# 切换 episode 时被取消的旧线程可能尚未退出，需加锁防止并发访问数据竞争
_KIN_LOCK = threading.Lock()

# 7 个关节名（与 viewer_3d 保持一致：L1~L6 + L7 夹爪）
ARM_JOINT_NAMES = [
    "L1_joint", "L2_joint", "L3_joint",
    "L4_joint", "L5_joint", "L6_joint", "L7_joint",
]


def quat_to_rpy_recording(qw, qx, qy, qz):
    """与录制端 euler_to_quat（XYZ 内旋，R=Rx·Ry·Rz）配对的四元数→欧拉角解算。

    训练数据中的四元数 [qx, qy, qz, qw] 由 FK 的 rpy 经 XYZ 内旋编码得到，
    必须用同一约定解出 rpy，才能正确反求关节角度（与 camera_pyqt 中一致）。
    返回 (rx, ry, rz)，单位 rad。
    """
    m = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    ry = math.asin(float(np.clip(m[0, 2], -1.0, 1.0)))
    rx = math.atan2(float(-m[1, 2]), float(m[2, 2]))
    rz = math.atan2(float(-m[0, 1]), float(m[0, 0]))
    return rx, ry, rz


def solve_episode_joint_angles(states, cancel_event=None):
    """对整段 episode 的末端位姿序列批量反求关节角度。

    Args:
        states: (N, M) 数组，自动识别两种格式：
          - M>=8 旧格式 [x,y,z,qx,qy,qz,qw,gripper_deg...]
          - M==7 新格式 [x,y,z,rx,ry,rz,gripper_rad]
        cancel_event: 可选 threading.Event；置位时每帧检查并立即放弃
          （切换 episode 时旧反解任务作废，避免 UI 等待线程收尾）

    Returns:
        (N, 7) float 数组，每行 = [L1..L6 rad, L7 夹爪 rad]；IK 失败的帧沿用上一帧解；
        被取消时返回 None
    """
    if _KIN is None or states is None or len(states) == 0:
        return None
    states = np.asarray(states, dtype=float)
    n = len(states)
    result = np.zeros((n, 7), dtype=float)
    q_prev = [0.0] * 6
    is_quat = states.shape[1] >= 8  # 旧四元数格式
    # 批量反解时降低 kinematics 日志级别，避免个别近奇异帧刷屏（仍沿用上一帧解）
    _kin_logger = logging.getLogger("el_a3_sdk.kinematics")
    _orig_level = _kin_logger.level
    _kin_logger.setLevel(logging.ERROR)
    try:
        for i in range(n):
            # 显式让出 GIL（每帧一次，总开销 <1ms）：pinocchio C++ 绑定调用不
            # 释放 GIL，IK 循环会近乎独占 GIL（实测 1.3s 内其他线程仅被调度
            # 2 次），导致 UI 线程饿死、切换 episode 卡死；让出后 UI 保持响应，
            # 取消标志也能及时被对端置位/被本循环看到
            time.sleep(0)
            if cancel_event is not None and cancel_event.is_set():
                return None
            row = states[i]
            x, y, z = float(row[0]), float(row[1]), float(row[2])
            if is_quat:
                qx, qy, qz, qw = float(row[3]), float(row[4]), float(row[5]), float(row[6])
                grip_deg = float(row[7]) if row.shape[0] > 7 else 0.0
                rx, ry, rz = quat_to_rpy_recording(qw, qx, qy, qz)
                grip_rad = math.radians(max(-90.0, min(90.0, grip_deg)))
            else:
                rx, ry, rz = float(row[3]), float(row[4]), float(row[5])
                grip_rad = float(row[6]) if row.shape[0] > 6 else 0.0
            try:
                pose = ArmEndPose(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)
                # pinocchio 的 Model/Data 非线程安全：切换 episode 时被取消的
                # 旧线程可能尚未退出，新线程并发调用会数据竞争，需加锁串行化
                with _KIN_LOCK:
                    # 热启动快速路径：ik_step（实时控制级 Jacobian 迭代，1~3 步
                    # 收敛），实测比全量 inverse_kinematics 快约 9 倍（逐帧链路
                    # 残差 ~0.01 rad，3D 回放显示足够）；残差过大（快速运动/
                    # 近奇异帧）时回退全量 IK 保证正确性
                    q, ik_err = _KIN.ik_step(pose, q_prev, damping=5e-3, max_step=0.5)
                    if q is None or ik_err > 0.05:
                        q_full = _KIN.inverse_kinematics(pose, q_init=q_prev)
                        if q_full is not None:
                            q = q_full
                if q is None:
                    q = q_prev
                else:
                    q_prev = list(q)
            except Exception:
                q = q_prev
            for j in range(6):
                result[i, j] = q[j]
            result[i, 6] = grip_rad
    finally:
        _kin_logger.setLevel(_orig_level)
    return result


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




# ==================== 可拖动剪辑标记滑条 ====================
class TrimSlider(QWidget):
    """自定义三标记滑条：起止标记 + 可拖动播放头。

    双击滑条时发出 segment_selected 信号，用于将位置曲线缩放到所选片段。
    """
    markers_changed = pyqtSignal(int, int)
    position_changed = pyqtSignal(int)  # 播放头位置实时变化（拖动/点击）
    position_clicked = pyqtSignal(int)    # 点击轨道/释放播放头时触发
    segment_selected = pyqtSignal(int, int)  # 双击选中起止片段

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(58)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._total = 0
        self._current = 0
        self._start = 0
        self._end = 0
        self._dragging = None  # 'start' / 'end' / 'playhead' / None

    def mouseDoubleClickEvent(self, event):
        """双击选中当前起止标记之间的片段。"""
        if self._total <= 0:
            return
        start, end = self.get_range()
        self.segment_selected.emit(start, end)
        super().mouseDoubleClickEvent(event)

    def set_total_frames(self, total):
        self._total = max(total, 0)
        self._start = 0
        self._end = max(0, self._total - 1)
        self._current = 0
        self.update()
        if self._total > 0:
            self.markers_changed.emit(self._start, self._end)

    def set_current_frame(self, frame):
        self._current = max(0, min(frame, max(0, self._total - 1)))
        self.update()

    def set_range(self, start, end):
        max_frame = max(0, self._total - 1)
        self._start = max(0, min(start, max_frame))
        self._end = max(0, min(end, max_frame))
        self.update()
        self.markers_changed.emit(self._start, self._end)

    def get_range(self):
        return (min(self._start, self._end), max(self._start, self._end))

    def _frame_from_x(self, x):
        rect = self.rect()
        if rect.width() <= 0 or self._total <= 0:
            return 0
        margin = 12
        w = rect.width() - 2 * margin
        ratio = max(0.0, min(1.0, (x - margin) / w))
        return int(round(ratio * self._total))

    def _x_from_frame(self, frame):
        rect = self.rect()
        if self._total <= 0:
            return rect.left() + 12
        margin = 12
        w = rect.width() - 2 * margin
        ratio = max(0.0, min(1.0, frame / self._total))
        return rect.left() + margin + int(ratio * w)

    def mousePressEvent(self, event):
        if self._total <= 0:
            return
        x = int(event.position().x())
        start_x = self._x_from_frame(self._start)
        end_x = self._x_from_frame(self._end)
        cur_x = self._x_from_frame(self._current)
        if abs(x - start_x) < 10:
            self._dragging = 'start'
        elif abs(x - end_x) < 10:
            self._dragging = 'end'
        elif abs(x - cur_x) < 12:
            self._dragging = 'playhead'
        else:
            # 点击轨道跳转当前位置
            self._dragging = None
            frame = self._frame_from_x(x)
            self._current = frame
            self.update()
            self.position_clicked.emit(frame)

    def mouseMoveEvent(self, event):
        if self._dragging is None or self._total <= 0:
            return
        frame = self._frame_from_x(int(event.position().x()))
        if self._dragging == 'start':
            self._start = max(0, min(frame, self._total - 1))
            self.update()
            self.markers_changed.emit(self._start, self._end)
        elif self._dragging == 'end':
            self._end = max(0, min(frame, self._total - 1))
            self.update()
            self.markers_changed.emit(self._start, self._end)
        elif self._dragging == 'playhead':
            self._current = max(0, min(frame, self._total - 1))
            self.update()
            self.position_changed.emit(self._current)

    def mouseReleaseEvent(self, event):
        if self._dragging == 'playhead':
            self.position_clicked.emit(self._current)
        self._dragging = None

    def leaveEvent(self, event):
        self._dragging = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        margin = 12
        groove = rect.adjusted(margin, 10, -margin, -18)

        # 背景 groove
        painter.setBrush(QBrush(QColor("#555555")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(groove, 4, 4)

        if self._total <= 0:
            painter.end()
            return

        # 选中区域（两个标记之间）
        start_x = self._x_from_frame(self._start)
        end_x = self._x_from_frame(self._end)
        sel_rect = groove.adjusted(0, 0, 0, 0)
        sel_rect.setLeft(min(start_x, end_x))
        sel_rect.setRight(max(start_x, end_x))
        painter.setBrush(QBrush(QColor("#FF5722")))
        painter.drawRoundedRect(sel_rect, 4, 4)

        # 播放头（可拖动）
        cur_x = self._x_from_frame(self._current)
        ph_top = groove.top() - 10
        ph_bottom = groove.bottom() + 10
        painter.setPen(QPen(QColor("#4CAF50"), 2))
        painter.drawLine(cur_x, ph_top, cur_x, ph_bottom)
        # 播放头小三角
        painter.setBrush(QBrush(QColor("#4CAF50")))
        painter.setPen(Qt.PenStyle.NoPen)
        ph_tri = QPolygonF([
            QPointF(cur_x, ph_top - 2),
            QPointF(cur_x - 6, ph_top - 12),
            QPointF(cur_x + 6, ph_top - 12),
        ])
        painter.drawPolygon(ph_tri)

        # 拖动标记三角形 + 文字（起止剪辑标记）
        painter.setPen(QPen(QColor("#333333"), 1))
        for fx, kind in [(start_x, 'start'), (end_x, 'end')]:
            color = QColor("#2196F3") if kind == 'start' else QColor("#9C27B0")
            painter.setBrush(QBrush(color))
            tri = QPolygonF([
                QPointF(fx, groove.bottom() + 2),
                QPointF(fx - 7, groove.bottom() + 16),
                QPointF(fx + 7, groove.bottom() + 16),
            ])
            painter.drawPolygon(tri)
            # 标记文字
            frame_val = self._start if kind == 'start' else self._end
            text = str(frame_val)
            painter.setPen(QPen(QColor("#FFFFFF")))
            painter.drawText(int(fx) - 12, groove.bottom() + 30, 24, 14, Qt.AlignmentFlag.AlignCenter, text)
        # 播放头文字
        painter.setPen(QPen(QColor("#4CAF50")))
        painter.drawText(int(cur_x) - 16, ph_top - 18, 32, 14, Qt.AlignmentFlag.AlignCenter, str(self._current))
        painter.end()

# ==================== RECAP 奖励标注轨道 ====================
class RecapTrackWidget(QWidget):
    """RECAP 奖励标注轨道：在播放时间线下方用彩色细带显示 good/bad/neutral 区间。

    灰色=未标记，绿色=good，红色=bad，橙色=neutral。宽度和边距与 TrimSlider 对齐。
    """

    LABEL_COLORS = {"good": "#2ecc71", "bad": "#e74c3c", "neutral": "#f39c12"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(12)
        self.setMaximumHeight(12)
        self._total = 0
        self._current = 0
        self._intervals = []

    def set_total(self, total):
        self._total = max(0, total)
        self.update()

    def set_intervals(self, intervals):
        self._intervals = list(intervals or [])
        self.update()

    def set_current(self, frame):
        self._current = max(0, frame)
        self.update()

    def _x_from_frame(self, frame):
        """与 TrimSlider 相同的坐标映射（margin=12）。"""
        if self._total <= 0:
            return 12
        margin = 12
        w = self.width() - 2 * margin
        ratio = max(0.0, min(1.0, frame / self._total))
        return margin + int(ratio * w)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        margin = 12
        track_w = w - 2 * margin
        track_y = 1
        track_h = h - 2

        if track_w <= 0:
            return

        if self._total <= 0:
            p.setBrush(QColor("#3a3a3a"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(margin, track_y, track_w, track_h, 2, 2)
            return

        # 灰色底（未标记区域）
        p.setBrush(QColor("#888888"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(margin, track_y, track_w, track_h, 2, 2)

        # 彩色区间段（覆盖在灰色底之上）
        for iv in self._intervals:
            x1 = self._x_from_frame(iv["start"])
            x2 = self._x_from_frame(iv["end"] + 1)
            color = QColor(self.LABEL_COLORS.get(iv.get("label"), "#888888"))
            p.setBrush(color)
            p.drawRoundedRect(x1, track_y, max(2, x2 - x1), track_h, 2, 2)

        # 当前播放位置线
        cx = margin + int((self._current / max(1, self._total - 1)) * track_w)
        p.setPen(QPen(QColor("#ffffff"), 1))
        p.drawLine(cx, 0, cx, h)


# ==================== 等比例缩放图像标签 ====================
class ImageLabel(QLabel):
    """固定 16:9（128:72）显示区域，保持原图比例缩放，避免画面被压缩或拉伸。
    
    支持鼠标双击事件，用于弹出全分辨率图像。
    """
    
    ASPECT_W = 128
    ASPECT_H = 72
    doubleClicked = pyqtSignal()
    
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._original = None       # 原始 numpy 图像 (H, W, C)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 固定 16:9 显示区域，高度根据宽度自动计算
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
    
    def mouseDoubleClickEvent(self, event):
        """双击时发出信号，通知主窗口弹出全分辨率图像。"""
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)
    
    def setImage(self, image):
        """设置原始图像，可传入 numpy 数组 (H,W,C) 或 QPixmap"""
        if isinstance(image, QPixmap):
            if image.isNull():
                self._original = None
            else:
                qimg = image.toImage().convertToFormat(QImage.Format.Format_RGB888)
                w, h = qimg.width(), qimg.height()
                ptr = qimg.bits()
                if hasattr(ptr, 'asarray'):
                    arr = np.array(ptr.asarray(h * w * 3)).reshape((h, w, 3))
                else:
                    arr = np.array(ptr).reshape((h, w, 3))
                self._original = arr
        elif isinstance(image, np.ndarray):
            self._original = image.copy() if image is not None else None
        else:
            self._original = None
        
        self.updateGeometry()
        self._apply_scaling()
    
    def setPixmap(self, pixmap):
        """兼容 QLabel.setPixmap"""
        self.setImage(pixmap)
    
    def clearImage(self):
        self._original = None
        self.updateGeometry()
        super().setPixmap(QPixmap())
    
    def hasHeightForWidth(self):
        return True
    
    def heightForWidth(self, width):
        """固定 16:9 比例"""
        return int(width * ImageLabel.ASPECT_H / ImageLabel.ASPECT_W)
    
    def sizeHint(self):
        from PyQt6.QtCore import QSize
        return QSize(self.ASPECT_W * 3, self.ASPECT_H * 3)  # 384 x 216 默认
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scaling()
    
    def _apply_scaling(self):
        if self._original is None:
            super().setPixmap(QPixmap())
            return
        
        label_w = max(self.width(), 1)
        label_h = max(self.height(), 1)
        img_h, img_w = self._original.shape[:2]
        
        # 在 16:9 的显示框内，按原图比例等比例缩放（不拉伸），可能会有黑边
        scale = min(label_w / img_w, label_h / img_h)
        if scale > 1.0:
            new_w, new_h = img_w, img_h
        else:
            new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        if new_w <= 0 or new_h <= 0:
            super().setPixmap(QPixmap())
            return
        
        try:
            import cv2
            resized = cv2.resize(self._original, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w, ch = resized.shape
            bytes_per_line = ch * w
            q_image = QImage(resized.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
        except Exception:
            pixmap = QPixmap.fromImage(
                QImage(self._original.data, img_w, img_h, img_w * 3, QImage.Format.Format_RGB888)
            ).scaled(new_w, new_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        super().setPixmap(pixmap)


# ==================== 全分辨率图像弹窗 ====================
class FullscreenImageDialog(QDialog):
    """双击图像回放框后弹出的全分辨率图像查看窗口。

    使用 QScrollArea 承载原始分辨率图像，支持滚动查看细节。
    """

    def __init__(self, parent, image, title="全分辨率图像"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 800)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 滚动区域，方便查看大分辨率图像
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.scroll_area)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

        # 将 numpy 图像转换为 QPixmap 并显示
        if isinstance(image, np.ndarray) and image.size > 0:
            rgb = np.ascontiguousarray(image[..., :3])
            h, w, ch = rgb.shape
            q_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image.copy())
            self.image_label.setPixmap(pixmap)
            self.image_label.setFixedSize(w, h)

        # 关闭按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet("QPushButton { padding: 6px 20px; }")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)


# ==================== Episode XYZ 位置轨迹曲线 ====================

def detect_stationary_intervals(frames_data, window_size, threshold_pct):
    """检测 7 个指标同时变化较小的连续帧区间（纯数据计算，无任何 UI 操作）。

    与 TrajectoryPlotWidget._run_analysis 同一算法（向量化实现，批量剪辑时
    直接调用，避免逐集加载视频/绘图/IK 造成的卡顿）：
    每个指标以整段 Episode 的最大绝对值作为变化基准，滑窗内所有指标的
    相对极差均小于阈值时，窗内帧判定为静止。

    Args:
        frames_data: dict，含 'x','y','z','rx','ry','rz','gripper' 七个序列
            （_load_episode_position_data 的返回结构）
        window_size: 判定静止的最少连续帧数
        threshold_pct: 相对变化阈值（百分数，如 0.05 表示 0.05%）

    Returns:
        [(start, end), ...] 闭区间列表，按起点升序；无静止区间返回 []
    """
    if frames_data is None:
        return []
    x = frames_data.get('x')
    if x is None or len(x) == 0:
        return []
    n = len(x)
    window = max(1, int(window_size))
    if n < window:
        return []

    keys = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
    arrays = []
    scales = []
    for key in keys:
        arr = np.asarray(frames_data.get(key, []), dtype=np.float64)
        if len(arr) < n:
            arr = np.zeros(n)  # 缺列按零处理（整段不变即静止，与原逻辑一致）
        scale = float(np.max(np.abs(arr))) if n > 0 else 0.0
        arrays.append(arr)
        scales.append(max(scale, 1e-9))

    # 向量化滑窗极差：(n-window+1, 7) —— 每行一个窗口、每列一个指标
    from numpy.lib.stride_tricks import sliding_window_view
    ranges = np.stack([
        np.ptp(sliding_window_view(a, window), axis=1) for a in arrays
    ], axis=1)
    threshold = float(threshold_pct) / 100.0
    all_stable = (ranges < (np.asarray(scales) * threshold)).all(axis=1)

    # 窗 s 静止 → 帧 [s, s+window-1] 全部标记（滑动叠加等价于原循环赋值）
    stationary = np.convolve(
        all_stable.astype(np.int32), np.ones(window, dtype=np.int32))[:n] > 0

    # 合并连续区间为 (start, end) 闭区间列表
    intervals = []
    start = None
    for i in range(n):
        if stationary[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                intervals.append((start, i - 1))
                start = None
    if start is not None:
        intervals.append((start, n - 1))
    return intervals


PRESET_LABELS = [
    "抓取第一个黑块",
    "放置第一个黑块",
    "抓取第二个黑块",
    "放置第二个黑块",
    "抓取小熊",
    "放置小熊",
    "机械臂归位",
]


class TrajectoryPlotWidget(QWidget):
    """显示单个 Episode 内末端位姿随帧序号变化的曲线。

    位置、欧拉角、夹爪分别绘制在三幅子图中，均显示原始值（不归一化）。
    右侧提供智能分析模块，可检测机械臂静止状态并在位置子图中用灰色条标注。

    支持：
    - 整个 Episode 以绝对帧序号显示
    - 选中片段时以相对帧序号（0 ~ 片段长度-1）显示
    - X 轴（帧范围）同步缩放三幅子图
    - Y 轴缩放仅作用于位置子图（欧拉角和夹爪保持自适应）
    - 当前帧竖线在三幅子图中同步，顶部子图带帧号标签
    - 使用 blit 局部重绘，保证播放流畅
    """

    # 信号：请求主界面剪辑检测到的静止区间，参数为 [(start, end), ...]
    trim_stationary_requested = pyqtSignal(list)
    # 信号：请求主界面依次剪辑所有 Episode 的静止区间
    trim_all_stationary_requested = pyqtSignal()
    # 信号：请求对当前 Episode 进行降噪（params dict）
    denoise_requested = pyqtSignal(dict)
    # 信号：请求对所有 Episode 批量降噪
    denoise_all_requested = pyqtSignal(dict)
    # 信号：请求预览当前 Episode 的降噪效果（不修改数据）
    preview_denoise_requested = pyqtSignal(dict)
    # 信号：请求把自动语言标注结果写回数据集（参数为 [{start,end,task,color,note}, ...]）
    subtask_apply_requested = pyqtSignal(list)
    # 信号：请求主窗口用 VLM 给当前分段生成标签（参数为 段数 N）
    vlm_label_requested = pyqtSignal(int)
    # 信号：数据集合并完成（参数为目标数据集名称），请求主窗口刷新数据集列表
    merge_completed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._current_frame = 0
        self._total_frames = 0

        # 片段模式：选中片段后，x 轴变为片段内相对帧序号
        self._segment_mode = False
        self._segment_start = 0
        self._segment_end = 0

        # 视图状态（均以 "zoom=1 时的基准值" 为参考）
        self._view_center = 0.0
        self._view_half_width = 0.0
        self._x_base_half_width = 0.0
        self._x_zoom_level = 1

        # 位置 Y 轴（受 Y 缩放滑块控制）
        self._y_center = 0.0
        self._y_half_range = 0.0
        self._y_base_half_range = 0.0
        self._y_zoom_level = 1

        # 欧拉角 Y 轴（固定自适应，不受 Y 缩放滑块控制）
        self._euler_y_center = 0.0
        self._euler_y_half_range = 0.0

        # 夹爪 Y 轴（固定自适应）
        self._grip_y_center = 0.0
        self._grip_y_half_range = 0.0

        # 夹爪力矩 Y 轴（右侧 twin axis，固定自适应）
        self._torque_y_center = 0.0
        self._torque_y_half_range = 0.0
        self._has_torque = False  # 当前 episode 是否含力矩数据

        # 智能分析结果
        self._stationary_intervals = []
        self._window_size = 10
        self._threshold_pct = 0.05
        self._stationary_bars = []

        # 自动语言标注（子任务分段）结果
        self._subtask_segments = []      # [{start,end,task,color,note}, ...]
        self._subtask_bars = []          # axvspan 句柄
        self._subtask_texts = []         # 文本标注句柄
        self._subtask_vlines = []        # 段边界竖线句柄

        # 降噪预览叠加数据（与 _data 同结构、同长度），None 表示不叠加
        self._smoothed = None

        # 数据集合并状态
        self._merge_root = None
        self._merge_worker = None
        self._merge_target_name = ""

        # 主布局：左侧绘图 + 右侧分析面板
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(5, 8), dpi=100, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(520)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        left_layout.addWidget(self.canvas, stretch=1)

        # 三幅子图共享 X 轴：位置 / 欧拉角 / 夹爪
        self.ax_pos = self.figure.add_subplot(311)
        self.ax_euler = self.figure.add_subplot(312, sharex=self.ax_pos)
        self.ax_grip = self.figure.add_subplot(313, sharex=self.ax_pos)
        self.axes = [self.ax_pos, self.ax_euler, self.ax_grip]

        # 夹爪力矩 twin axis（共享夹爪子图 X 轴，独立右侧 Y 轴）
        self.ax_torque = self.ax_grip.twinx()

        self.ax_pos.set_ylabel("位置 (m)")
        self.ax_euler.set_ylabel("欧拉角 (rad)")
        self.ax_grip.set_ylabel("夹爪 (rad)")
        self.ax_torque.set_ylabel("力矩 (Nm)")
        self.ax_grip.set_xlabel("帧序号")

        for ax in self.axes:
            ax.grid(True, linestyle='--', alpha=0.5)

        self.lines = {}
        self.vlines = []  # 三幅子图各一条竖线
        self.cursor_label = None  # 播放标签显示在顶部子图

        # 每幅子图的 blit 背景缓存
        self._bg_pos = None
        self._bg_euler = None
        self._bg_grip = None

        # 缩放控制
        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel("X缩放:"))
        self.x_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.x_zoom_slider.setRange(1, 100)
        self.x_zoom_slider.setValue(1)
        self.x_zoom_slider.setToolTip("水平放大：1=当前基准，100=最大放大（同步三幅子图）")
        self.x_zoom_slider.valueChanged.connect(self._on_x_zoom_changed)
        controls.addWidget(self.x_zoom_slider, stretch=2)

        controls.addWidget(QLabel("Y缩放:"))
        self.y_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.y_zoom_slider.setRange(1, 100)
        self.y_zoom_slider.setValue(1)
        self.y_zoom_slider.setToolTip("垂直放大：仅作用于位置曲线")
        self.y_zoom_slider.valueChanged.connect(self._on_y_zoom_changed)
        controls.addWidget(self.y_zoom_slider, stretch=2)

        self.reset_zoom_btn = QPushButton("重置")
        self.reset_zoom_btn.setToolTip("重置为显示整个 Episode")
        self.reset_zoom_btn.clicked.connect(self._reset_zoom)
        controls.addWidget(self.reset_zoom_btn)

        left_layout.addLayout(controls)

        # 右侧智能分析面板
        analysis_group = QGroupBox("智能分析")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setSpacing(8)

        form_layout = QFormLayout()
        form_layout.setSpacing(6)

        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 1000)
        self.window_spin.setValue(self._window_size)
        self.window_spin.setToolTip("判定静止状态的最少连续帧数")
        self.window_spin.valueChanged.connect(self._on_window_size_changed)
        form_layout.addRow("静止窗口 (帧):", self.window_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 2.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(self._threshold_pct)
        self.threshold_spin.setSuffix(" %")
        self.threshold_spin.setToolTip("窗口内 7 个指标相对变化均小于此阈值时判定为静止（可直接输入）")
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        form_layout.addRow("变化阈值:", self.threshold_spin)

        analysis_layout.addLayout(form_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self.analyze_btn = QPushButton("🔍 智能分析")
        self.analyze_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 6px; }"
        )
        self.analyze_btn.clicked.connect(self._run_analysis)
        btn_layout.addWidget(self.analyze_btn)

        self.trim_stationary_btn = QPushButton("✂ 剪辑静止")
        self.trim_stationary_btn.setStyleSheet(
            "QPushButton { background-color: #FF5722; color: white; font-weight: bold; padding: 6px; }"
        )
        self.trim_stationary_btn.setToolTip("删除当前检测到的所有静止帧")
        self.trim_stationary_btn.clicked.connect(self._on_trim_stationary_clicked)
        btn_layout.addWidget(self.trim_stationary_btn)

        self.trim_all_stationary_btn = QPushButton("✂ 剪辑所有静止")
        self.trim_all_stationary_btn.setStyleSheet(
            "QPushButton { background-color: #D84315; color: white; font-weight: bold; padding: 6px; }"
        )
        self.trim_all_stationary_btn.setToolTip("依次分析并删除所有 Episode 的静止帧（逐个处理，避免文件损坏）")
        self.trim_all_stationary_btn.clicked.connect(self._on_trim_all_stationary_clicked)
        btn_layout.addWidget(self.trim_all_stationary_btn)

        analysis_layout.addLayout(btn_layout)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("点击“智能分析”查看结果")
        self.result_text.setMaximumWidth(260)
        self.result_text.setMinimumHeight(120)
        analysis_layout.addWidget(self.result_text, stretch=1)

        # ---------- 自动语言标注（子任务分段） ----------
        subtask_sep = QFrame()
        subtask_sep.setFrameShape(QFrame.Shape.HLine)
        subtask_sep.setFrameShadow(QFrame.Shadow.Sunken)
        analysis_layout.addWidget(subtask_sep)

        analysis_layout.addWidget(QLabel("🗣 自动语言标注（智能分段 + VLM 标签）"))

        # 段数 + 含力矩
        seg_row = QHBoxLayout()
        seg_row.setSpacing(4)
        seg_row.addWidget(QLabel("段数:"))
        self.subtask_count_spin = QSpinBox()
        self.subtask_count_spin.setRange(1, 30)
        self.subtask_count_spin.setValue(7)
        self.subtask_count_spin.setToolTip("指定要切分的段数（如 3/5/8），将强制切出恰好这么多段")
        seg_row.addWidget(self.subtask_count_spin)
        self.subtask_torque_chk = QCheckBox("用力矩")
        self.subtask_torque_chk.setChecked(True)
        self.subtask_torque_chk.setToolTip("勾选后边界评分加入夹爪力矩（接触/释放物体事件），切分更准")
        seg_row.addWidget(self.subtask_torque_chk)
        self.subtask_angle_chk = QCheckBox("用夹角")
        self.subtask_angle_chk.setChecked(False)
        self.subtask_angle_chk.setToolTip("勾选后边界评分加入夹爪角度变化（开/合事件）。默认关：仅靠位姿停顿+力矩分段")
        seg_row.addWidget(self.subtask_angle_chk)
        analysis_layout.addLayout(seg_row)

        # 智能分段按钮：按力矩+位姿切出 N 段（占位标签 步骤1/2/..）
        self.subtask_segment_btn = QPushButton("🤖 智能分段")
        self.subtask_segment_btn.setStyleSheet(
            "QPushButton { background-color: #7B1FA2; color: white; font-weight: bold; padding: 5px; }"
        )
        self.subtask_segment_btn.setToolTip("根据夹爪力矩+位姿变化切出指定段数的真实边界，并在曲线上着色预览")
        self.subtask_segment_btn.clicked.connect(self._run_smart_segment)
        analysis_layout.addWidget(self.subtask_segment_btn)

        self.subtask_preset_btn = QPushButton("📋 套用预设名称")
        self.subtask_preset_btn.setStyleSheet(
            "QPushButton { background-color: #455A64; color: white; font-weight: bold; padding: 5px; }"
        )
        self.subtask_preset_btn.setToolTip(
            "按当前段数套用预设名称（抓取/放置黑块·小熊·机械臂归位）。段数超出预设时多余段保持 \"步骤N\"")
        self.subtask_preset_btn.clicked.connect(self._apply_preset_labels)
        analysis_layout.addWidget(self.subtask_preset_btn)

        # 可编辑标签表：步 | 标签
        self.subtask_label_table = QTableWidget(0, 2)
        self.subtask_label_table.setHorizontalHeaderLabels(["步", "标签（可编辑）"])
        self.subtask_label_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.subtask_label_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.subtask_label_table.setMaximumHeight(120)
        self.subtask_label_table.setMaximumWidth(260)
        self.subtask_label_table.setToolTip("双击“标签”列可手动修改名称；修改后会同步更新曲线着色")
        self.subtask_label_table.itemChanged.connect(self._on_label_table_edited)
        analysis_layout.addWidget(self.subtask_label_table)

        # 摄像头路数选择（生成标签 VLM 推理用）
        vlm_cam_row = QHBoxLayout()
        vlm_cam_row.addWidget(QLabel("摄像头:"))
        self.subtask_cam_combo = QComboBox()
        self.subtask_cam_combo.addItems(["摄像头1 (cam1)", "摄像头2 (cam2)", "双摄 (cam1+cam2)"])
        self.subtask_cam_combo.setCurrentIndex(0)
        self.subtask_cam_combo.setToolTip(
            "选择传入 Qwen 模型的摄像头画面（用于“生成标签(VLM)”）。\n"
            " · 摄像头1 / 摄像头2：只用对应单路画面\n"
            " · 双摄：同一消息里同时喂两路画面")
        vlm_cam_row.addWidget(self.subtask_cam_combo, 1)
        analysis_layout.addLayout(vlm_cam_row)

        # 生成标签(VLM) + 清除
        sub_btn_row = QHBoxLayout()
        sub_btn_row.setSpacing(4)
        self.subtask_vlm_btn = QPushButton("🏷 生成标签(VLM)")
        self.subtask_vlm_btn.setStyleSheet(
            "QPushButton { background-color: #1565C0; color: white; font-weight: bold; padding: 5px; }"
        )
        self.subtask_vlm_btn.setToolTip("调用 Qwen3-VL 分析每段关键帧，自动生成“动作+物体”标签（后台运行）")
        self.subtask_vlm_btn.setEnabled(False)
        self.subtask_vlm_btn.clicked.connect(self._on_vlm_label_clicked)
        sub_btn_row.addWidget(self.subtask_vlm_btn)

        self.subtask_clear_btn = QPushButton("🗑 清除")
        self.subtask_clear_btn.setToolTip("清除曲线上的子任务标注")
        self.subtask_clear_btn.clicked.connect(self._clear_subtask_annotate)
        sub_btn_row.addWidget(self.subtask_clear_btn)
        analysis_layout.addLayout(sub_btn_row)

        self.subtask_apply_btn = QPushButton("💾 应用到数据集")
        self.subtask_apply_btn.setStyleSheet(
            "QPushButton { background-color: #00695C; color: white; font-weight: bold; padding: 5px; }"
        )
        self.subtask_apply_btn.setToolTip("把当前标签写回 LeRobot 数据集（逐帧 language_persistent 列）并记入标签库")
        self.subtask_apply_btn.setEnabled(False)
        self.subtask_apply_btn.clicked.connect(self._on_apply_subtask_clicked)
        analysis_layout.addWidget(self.subtask_apply_btn)

        self.subtask_result_label = QLabel("（设段数 → 智能分段 → 生成标签 → 应用）")
        self.subtask_result_label.setWordWrap(True)
        self.subtask_result_label.setMaximumWidth(260)
        self.subtask_result_label.setStyleSheet("QLabel { color: #555; font-size: 11px; }")
        analysis_layout.addWidget(self.subtask_result_label)


        # 右侧容器：分页（智能分析 + RECAP）
        right_container = QTabWidget()
        # Tab 1: 智能分析（原 analysis_group + 智能降噪）
        analysis_tab = QWidget()
        analysis_tab_lay = QVBoxLayout(analysis_tab)
        analysis_tab_lay.setContentsMargins(2, 2, 2, 2)
        analysis_tab_lay.setSpacing(8)
        analysis_tab_lay.addWidget(analysis_group, stretch=2)
        analysis_tab_lay.addWidget(self._build_denoise_group(), stretch=1)
        right_container.addTab(analysis_tab, "智能分析")
        # Tab 2: RECAP 奖励标注
        right_container.addTab(self._build_recap_tab(), "RECAP")
        # Tab 3: 数据集合并
        right_container.addTab(self._build_merge_tab(), "数据集合并")

        main_layout.addWidget(left_widget, stretch=3)
        main_layout.addWidget(right_container, stretch=1)

        self._empty_plot()

    # ==================== RECAP 奖励标注面板 ====================
    def _build_recap_tab(self):
        """构建 RECAP 奖励标注分页（参照 label_rewards_qt.py 的标注方式）。

        仅构建控件；信号连接由 MainWindow._connect_recap_signals() 统一完成。
        """
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(6)

        hint = QLabel(
            "RECAP 片段级奖励标注：标记 good/bad/neutral 区间 →\n"
            "expand_annotations.py 展开为逐帧 advantage_indicator → 训练"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        lay.addWidget(hint)

        # 当前标签 + 起止帧
        form = QFormLayout()
        self.recap_label_combo = QComboBox()
        self.recap_label_combo.addItems(RECAP_ALL_LABELS)
        self.recap_label_combo.setCurrentText("good")
        form.addRow("当前标签:", self.recap_label_combo)

        self.recap_start_spin = QSpinBox()
        self.recap_start_spin.setMaximum(999999)
        self.recap_end_spin = QSpinBox()
        self.recap_end_spin.setMaximum(999999)
        form.addRow("区间起点:", self.recap_start_spin)
        form.addRow("区间终点:", self.recap_end_spin)
        lay.addLayout(form)

        # 标记按钮
        mark_row = QHBoxLayout()
        self.recap_mark_start_btn = QPushButton("[ 设起点")
        self.recap_mark_end_btn = QPushButton("设终点 ]")
        mark_row.addWidget(self.recap_mark_start_btn)
        mark_row.addWidget(self.recap_mark_end_btn)
        lay.addLayout(mark_row)

        # 添加区间
        self.recap_add_btn = QPushButton("✚ 添加区间")
        self.recap_add_btn.setStyleSheet("font-weight:bold; padding:5px;")
        lay.addWidget(self.recap_add_btn)

        # 备注
        self.recap_note_edit = QLineEdit()
        self.recap_note_edit.setPlaceholderText("备注 (可选)")
        lay.addWidget(QLabel("备注:"))
        lay.addWidget(self.recap_note_edit)

        # 快捷操作
        lay.addWidget(QLabel("快捷操作:"))
        quick_row = QHBoxLayout()
        self.recap_quick_good = QPushButton("✓ 好")
        self.recap_quick_good.setStyleSheet(
            f"background:{RECAP_LABEL_COLORS['good']}; color:#fff; font-weight:bold;")
        self.recap_quick_bad = QPushButton("✗ 坏")
        self.recap_quick_bad.setStyleSheet(
            f"background:{RECAP_LABEL_COLORS['bad']}; color:#fff; font-weight:bold;")
        self.recap_quick_neutral = QPushButton("〜 中")
        self.recap_quick_neutral.setStyleSheet(
            f"background:{RECAP_LABEL_COLORS['neutral']}; color:#fff;")
        quick_row.addWidget(self.recap_quick_good)
        quick_row.addWidget(self.recap_quick_bad)
        quick_row.addWidget(self.recap_quick_neutral)
        lay.addLayout(quick_row)

        # 整 episode 快标
        lay.addWidget(QLabel("整 episode 快标:"))
        ep_row = QHBoxLayout()
        self.recap_ep_good = QPushButton("全部=好")
        self.recap_ep_bad = QPushButton("全部=坏")
        ep_row.addWidget(self.recap_ep_good)
        ep_row.addWidget(self.recap_ep_bad)
        lay.addLayout(ep_row)

        # 区间列表
        lay.addWidget(QLabel("已标注区间:"))
        self.recap_interval_table = QTableWidget(0, 5)
        self.recap_interval_table.setHorizontalHeaderLabels(["#", "起点", "终点", "标签", "备注"])
        self.recap_interval_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.recap_interval_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.recap_interval_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.recap_interval_table.setMaximumWidth(300)
        lay.addWidget(self.recap_interval_table)

        # 删除/撤销/清空
        del_row = QHBoxLayout()
        self.recap_del_btn = QPushButton("✖ 删除选中")
        self.recap_undo_btn = QPushButton("↶ 撤销最后")
        self.recap_clear_btn = QPushButton("清空本 episode")
        del_row.addWidget(self.recap_del_btn)
        del_row.addWidget(self.recap_undo_btn)
        del_row.addWidget(self.recap_clear_btn)
        lay.addLayout(del_row)

        # 保存
        self.recap_save_btn = QPushButton("💾 保存标注")
        self.recap_save_btn.setStyleSheet("padding:7px; font-weight:bold;")
        lay.addWidget(self.recap_save_btn)

        # 状态提示
        self.recap_status_label = QLabel("（选择数据集后自动加载 reward_annotations.json）")
        self.recap_status_label.setWordWrap(True)
        self.recap_status_label.setStyleSheet("QLabel { color: #555; font-size: 11px; }")
        lay.addWidget(self.recap_status_label)

        lay.addStretch()
        return tab

    # ==================== 数据集合并面板 ====================
    def _build_merge_tab(self):
        """构建数据集合并分页（复用 merge_lerobot_datasets.MergeWorker）。

        勾选源数据集（按列表顺序合并）→ 指定目标名称 → 后台线程合并，
        完成后通过 merge_completed 信号通知主窗口刷新数据集列表。
        """
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(6)

        hint = QLabel(
            "将多个 LeRobot v3.0 数据集合并为一个：重建 data/meta 指针、"
            "规整 frame_index/timestamp、重算 stats，避免 数据问题分析.md 中的 4 类元数据问题"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        lay.addWidget(hint)

        head_row = QHBoxLayout()
        head_row.addWidget(QLabel("源数据集 (按顺序合并):"))
        self.merge_refresh_btn = QPushButton("🔄")
        self.merge_refresh_btn.setToolTip("重新扫描数据根目录")
        self.merge_refresh_btn.setMaximumWidth(36)
        self.merge_refresh_btn.clicked.connect(self._merge_refresh_list)
        head_row.addWidget(self.merge_refresh_btn)
        lay.addLayout(head_row)

        self.merge_list = QListWidget()
        self.merge_list.setMaximumHeight(180)
        lay.addWidget(self.merge_list)

        op_row = QHBoxLayout()
        op_row.setSpacing(4)
        self.merge_all_btn = QPushButton("全选")
        self.merge_all_btn.clicked.connect(lambda: self._merge_set_all(True))
        self.merge_none_btn = QPushButton("全不选")
        self.merge_none_btn.clicked.connect(lambda: self._merge_set_all(False))
        self.merge_up_btn = QPushButton("上移")
        self.merge_up_btn.clicked.connect(lambda: self._merge_move_item(-1))
        self.merge_down_btn = QPushButton("下移")
        self.merge_down_btn.clicked.connect(lambda: self._merge_move_item(1))
        for b in (self.merge_all_btn, self.merge_none_btn, self.merge_up_btn, self.merge_down_btn):
            op_row.addWidget(b)
        lay.addLayout(op_row)

        lay.addWidget(QLabel("目标数据集 (可填已有名称覆盖, 或新名称):"))
        self.merge_target_combo = QComboBox()
        self.merge_target_combo.setEditable(True)
        lay.addWidget(self.merge_target_combo)

        self.merge_start_btn = QPushButton("🔀 开始合并")
        self.merge_start_btn.setStyleSheet(
            "QPushButton { background-color: #6A1B9A; color: white; font-weight: bold; padding: 7px; }"
        )
        self.merge_start_btn.clicked.connect(self._on_merge_start)
        lay.addWidget(self.merge_start_btn)

        self.merge_progress = QProgressBar()
        self.merge_progress.setMaximumHeight(16)
        lay.addWidget(self.merge_progress)

        self.merge_log = QPlainTextEdit()
        self.merge_log.setReadOnly(True)
        self.merge_log.setMaximumHeight(140)
        self.merge_log.setPlaceholderText("合并日志")
        lay.addWidget(self.merge_log)

        lay.addStretch()
        return tab

    def set_merge_root(self, root):
        """由主窗口设置数据根目录并刷新源列表。"""
        self._merge_root = root
        self._merge_refresh_list()

    def _merge_find_datasets(self):
        if not self._merge_root or not os.path.isdir(self._merge_root):
            return []
        out = []
        for name in sorted(os.listdir(self._merge_root)):
            path = os.path.join(self._merge_root, name)
            if not os.path.isdir(path) or name.startswith("."):
                continue
            info_path = os.path.join(path, "meta", "info.json")
            if not os.path.exists(info_path):
                continue
            try:
                with open(info_path, encoding="utf-8") as f:
                    info = json.load(f)
                out.append((name, info.get("total_episodes", "?"), info.get("total_frames", "?")))
            except Exception:
                out.append((name, "?", "?"))
        return out

    def _merge_refresh_list(self):
        checked = {
            self.merge_list.item(i).text().split("  (")[0]
            for i in range(self.merge_list.count())
            if self.merge_list.item(i).checkState() == Qt.CheckState.Checked
        }
        self.merge_list.clear()
        self.merge_target_combo.clear()
        for name, eps, frames in self._merge_find_datasets():
            item = QListWidgetItem(f"{name}  ({eps} eps / {frames} 帧)")
            item.setCheckState(
                Qt.CheckState.Checked if name in checked else Qt.CheckState.Unchecked
            )
            self.merge_list.addItem(item)
            self.merge_target_combo.addItem(name)

    def _merge_set_all(self, checked):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.merge_list.count()):
            self.merge_list.item(i).setCheckState(state)

    def _merge_move_item(self, delta):
        row = self.merge_list.currentRow()
        if row < 0:
            return
        new_row = row + delta
        if not (0 <= new_row < self.merge_list.count()):
            return
        item = self.merge_list.takeItem(row)
        self.merge_list.insertItem(new_row, item)
        self.merge_list.setCurrentRow(new_row)

    def _on_merge_start(self):
        if not self._merge_root:
            QMessageBox.warning(self, "提示", "数据根目录未设置")
            return
        sources = []
        for i in range(self.merge_list.count()):
            it = self.merge_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                sources.append(it.text().split("  (")[0])
        target = self.merge_target_combo.currentText().strip()
        if not sources:
            QMessageBox.warning(self, "提示", "请至少勾选一个源数据集")
            return
        if not target:
            QMessageBox.warning(self, "提示", "请填写目标数据集名称")
            return
        existing = {name for name, _, _ in self._merge_find_datasets()}
        try:
            global_issues, drop_plan = preflight_check(self._merge_root, sources)
        except Exception as e:
            QMessageBox.critical(self, "检查失败", f"读取源数据集信息失败:\n{e}")
            return
        drop_features = []
        allow_overrides = False
        if global_issues or drop_plan:
            parts = []
            if global_issues:
                parts.append("全局元数据不一致 (继续将采用首个数据集的值):\n"
                             + "\n".join(f"  · {i}" for i in global_issues))
            if drop_plan:
                parts.append("以下特征不兼容 (继续将舍弃这些特征):\n"
                             + "\n".join(f"  · {f} — {r}" for f, r in drop_plan))
            ret = QMessageBox.question(
                self,
                "格式不兼容",
                "检测到格式不兼容:\n\n" + "\n\n".join(parts) + "\n\n是否继续合并?",
            )
            if ret != QMessageBox.StandardButton.Yes:
                self.merge_log.appendPlainText("用户取消合并")
                return
            drop_features = [f for f, _ in drop_plan]
            allow_overrides = bool(global_issues)
            if drop_features:
                self.merge_log.appendPlainText("用户确认继续, 将舍弃特征: " + ", ".join(drop_features))
            if allow_overrides:
                self.merge_log.appendPlainText("用户确认继续, 全局元数据采用首个数据集的值")
        if target in existing:
            if target in sources:
                text = f"目标 {target} 同时是源数据集, 合并后原数据将被替换 (自动备份)。继续?"
            else:
                text = f"目标 {target} 已存在且不在源列表中, 其内容将被合并结果替换 (自动备份)。继续?"
            ret = QMessageBox.question(self, "确认覆盖", text)
            if ret != QMessageBox.StandardButton.Yes:
                return
        self.merge_start_btn.setEnabled(False)
        self.merge_progress.setValue(0)
        self.merge_log.appendPlainText(f"开始合并: {sources} -> {target}")
        self._merge_target_name = target
        self._merge_worker = MergeWorker(
            self._merge_root, sources, target,
            drop_features=drop_features, allow_overrides=allow_overrides,
        )
        self._merge_worker.sig_log.connect(self.merge_log.appendPlainText)
        self._merge_worker.sig_progress.connect(self._on_merge_progress)
        self._merge_worker.sig_done.connect(self._on_merge_done)
        self._merge_worker.start()

    def _on_merge_progress(self, cur, total, msg):
        self.merge_progress.setMaximum(total)
        self.merge_progress.setValue(cur)

    def _on_merge_done(self, ok, msg):
        self.merge_start_btn.setEnabled(True)
        self.merge_log.appendPlainText(msg)
        target = self._merge_target_name
        self._merge_refresh_list()
        if ok:
            self.merge_completed.emit(target)
        else:
            QMessageBox.critical(self, "合并失败", msg)

    # ==================== 智能降噪面板 ====================
    def _build_denoise_group(self):
        """构建“智能降噪”面板：滤波方法、平滑参数、预览/单个/批量降噪按钮。"""
        group = QGroupBox("智能降噪")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setSpacing(6)

        self.denoise_method_combo = QComboBox()
        self.denoise_method_combo.addItems([
            "Savitzky-Golay（推荐）",
            "移动平均",
            "高斯滤波",
        ])
        self.denoise_method_combo.setToolTip(
            "Savitzky-Golay：零相位、不引入时间滞后，最适合离线轨迹平滑\n"
            "（lerobot 官方仅在遥操作端使用 EMA 在线平滑，离线数据不推荐，"
            "EMA 滞后会导致 action 与图像帧错位）"
        )
        self.denoise_method_combo.currentIndexChanged.connect(self._on_denoise_method_changed)
        form.addRow("滤波方法:", self.denoise_method_combo)

        self.denoise_window_spin = QSpinBox()
        self.denoise_window_spin.setRange(3, 101)
        self.denoise_window_spin.setSingleStep(2)
        self.denoise_window_spin.setValue(11)
        self.denoise_window_spin.setToolTip("平滑窗口长度（帧），越大越平滑；20fps 下 11 帧≈0.55s")
        form.addRow("平滑窗口 (帧):", self.denoise_window_spin)

        self.denoise_poly_spin = QSpinBox()
        self.denoise_poly_spin.setRange(1, 5)
        self.denoise_poly_spin.setValue(2)
        self.denoise_poly_spin.setToolTip("Savitzky-Golay 多项式阶数，需小于窗口长度")
        form.addRow("多项式阶数:", self.denoise_poly_spin)

        self.denoise_sigma_spin = QDoubleSpinBox()
        self.denoise_sigma_spin.setRange(0.3, 20.0)
        self.denoise_sigma_spin.setDecimals(1)
        self.denoise_sigma_spin.setSingleStep(0.5)
        self.denoise_sigma_spin.setValue(2.0)
        self.denoise_sigma_spin.setEnabled(False)
        self.denoise_sigma_spin.setToolTip("高斯滤波标准差（帧），越大越平滑")
        form.addRow("高斯 σ:", self.denoise_sigma_spin)

        layout.addLayout(form)

        target_layout = QHBoxLayout()
        target_layout.setSpacing(6)
        target_layout.addWidget(QLabel("作用对象:"))
        self.denoise_action_check = QCheckBox("action")
        self.denoise_action_check.setChecked(True)
        self.denoise_action_check.setToolTip("平滑 action 轨迹（训练目标，推荐勾选）")
        self.denoise_state_check = QCheckBox("state")
        self.denoise_state_check.setChecked(True)
        self.denoise_state_check.setToolTip("平滑 observation.state 轨迹")
        target_layout.addWidget(self.denoise_action_check)
        target_layout.addWidget(self.denoise_state_check)
        target_layout.addStretch()
        layout.addLayout(target_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self.preview_denoise_btn = QPushButton("🧪 预览降噪")
        self.preview_denoise_btn.setToolTip("在当前曲线上叠加虚线显示平滑效果（不修改数据）")
        self.preview_denoise_btn.setStyleSheet(
            "QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 6px; }"
        )
        self.preview_denoise_btn.clicked.connect(self._on_preview_denoise_clicked)
        btn_layout.addWidget(self.preview_denoise_btn)

        self.denoise_btn = QPushButton("✨ 智能降噪")
        self.denoise_btn.setToolTip("对当前 Episode 应用平滑降噪（只改 state/action 数值，帧数不变）")
        self.denoise_btn.setStyleSheet(
            "QPushButton { background-color: #009688; color: white; font-weight: bold; padding: 6px; }"
        )
        self.denoise_btn.clicked.connect(self._on_denoise_clicked)
        btn_layout.addWidget(self.denoise_btn)

        self.denoise_all_btn = QPushButton("✨ 批量智能降噪")
        self.denoise_all_btn.setToolTip("对所有 Episode 依次应用平滑降噪（逐个处理，避免文件损坏）")
        self.denoise_all_btn.setStyleSheet(
            "QPushButton { background-color: #00796B; color: white; font-weight: bold; padding: 6px; }"
        )
        self.denoise_all_btn.clicked.connect(self._on_denoise_all_clicked)
        btn_layout.addWidget(self.denoise_all_btn)

        layout.addLayout(btn_layout)

        self.denoise_status_label = QLabel("仅修改 state/action 数值，帧数与元数据不变")
        self.denoise_status_label.setWordWrap(True)
        self.denoise_status_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        layout.addWidget(self.denoise_status_label)
        return group

    def _on_denoise_method_changed(self, idx):
        """根据滤波方法启用对应参数。"""
        self.denoise_poly_spin.setEnabled(idx == 0)
        self.denoise_window_spin.setEnabled(idx in (0, 1))
        self.denoise_sigma_spin.setEnabled(idx == 2)

    def _get_denoise_params(self):
        """收集当前降噪参数。"""
        method = ["savgol", "moving_avg", "gaussian"][self.denoise_method_combo.currentIndex()]
        return {
            "method": method,
            "window": int(self.denoise_window_spin.value()),
            "polyorder": int(self.denoise_poly_spin.value()),
            "sigma": float(self.denoise_sigma_spin.value()),
            "smooth_action": self.denoise_action_check.isChecked(),
            "smooth_state": self.denoise_state_check.isChecked(),
        }

    def _on_preview_denoise_clicked(self):
        if self._data is None or self._total_frames == 0:
            self.denoise_status_label.setText("请先选择 Episode")
            return
        self.preview_denoise_requested.emit(self._get_denoise_params())

    def _on_denoise_clicked(self):
        params = self._get_denoise_params()
        if not params["smooth_action"] and not params["smooth_state"]:
            self.denoise_status_label.setText("请至少勾选一个作用对象")
            return
        self.denoise_requested.emit(params)

    def _on_denoise_all_clicked(self):
        params = self._get_denoise_params()
        if not params["smooth_action"] and not params["smooth_state"]:
            self.denoise_status_label.setText("请至少勾选一个作用对象")
            return
        self.denoise_all_requested.emit(params)

    def set_denoise_buttons_enabled(self, enabled):
        self.preview_denoise_btn.setEnabled(enabled)
        self.denoise_btn.setEnabled(enabled)
        self.denoise_all_btn.setEnabled(enabled)

    def set_smoothed_overlay(self, smoothed_data):
        """设置平滑预览数据（与 _data 同结构、同长度），以虚线叠加显示。"""
        self._smoothed = smoothed_data
        if self._data is None:
            return
        self._plot_data(
            self._segment_start if self._segment_mode else 0,
            self._segment_end if self._segment_mode else self._total_frames - 1,
            self._get_current_rel_frame(),
            x_label="片段内帧序号" if self._segment_mode else "帧序号"
        )
        self._apply_view()

    def _on_window_size_changed(self, value):
        self._window_size = max(1, int(value))

    def _on_threshold_changed(self, value):
        self._threshold_pct = float(value)

    def _on_trim_stationary_clicked(self):
        """点击“剪辑静止”按钮：将当前检测到的静止区间提交给主界面处理。"""
        if not self._stationary_intervals:
            QMessageBox.information(
                self, "提示", "当前没有检测到静止区间，请先点击“智能分析”。"
            )
            return
        self.trim_stationary_requested.emit(list(self._stationary_intervals))

    def _on_trim_all_stationary_clicked(self):
        """点击“剪辑所有静止”按钮：请求主界面依次处理所有 Episode。"""
        self.trim_all_stationary_requested.emit()

    def _compute_y_range(self, keys, start=0, end=None):
        """计算指定帧范围 [start, end] 内给定数据键的 Y 轴范围。"""
        if self._data is None:
            return 0.0, 1.0
        if end is None:
            end = self._total_frames - 1
        s = max(0, min(start, end, self._total_frames - 1))
        e = min(self._total_frames, max(start, end, 0) + 1)
        if e <= s:
            e = s + 1
        arrays = [self._data.get(k, [])[s:e] for k in keys if k in self._data]
        if not arrays:
            return 0.0, 1.0
        all_vals = np.concatenate(arrays)
        if len(all_vals) == 0:
            return 0.0, 1.0
        ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
        if abs(ymax - ymin) < 1e-9:
            ymax = ymin + 1.0
        return ymin, ymax

    def _capture_backgrounds(self):
        """捕获三幅子图的背景，用于 blit 快速更新播放竖线。"""
        try:
            self.canvas.draw()
            self._bg_pos = self.canvas.copy_from_bbox(self.ax_pos.bbox)
            self._bg_euler = self.canvas.copy_from_bbox(self.ax_euler.bbox)
            self._bg_grip = self.canvas.copy_from_bbox(self.ax_grip.bbox)
        except Exception:
            self._bg_pos = None
            self._bg_euler = None
            self._bg_grip = None

    def _empty_plot(self):
        """清空为占位状态。"""
        for ax in self.axes:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.5)
        self.ax_torque.clear()
        self.ax_torque.grid(False)

        # ax.clear() 已经移除所有旧 artist
        self._stationary_bars = []

        self.ax_pos.set_ylabel("位置 (m)")
        self.ax_euler.set_ylabel("欧拉角 (rad)")
        self.ax_grip.set_ylabel("夹爪 (rad)")
        self.ax_torque.set_ylabel("力矩 (Nm)")
        self.ax_grip.set_xlabel("帧序号")

        self.ax_pos.set_title("Episode 末端位姿变化曲线")
        self.ax_pos.text(
            0.5, 0.5, "请选择 Episode",
            horizontalalignment='center', verticalalignment='center',
            transform=self.ax_pos.transAxes, fontsize=12, color='gray'
        )

        self.lines = {}
        self.vlines = []
        self.cursor_label = None
        self._data = None
        self._smoothed = None
        self._total_frames = 0
        self._segment_mode = False
        self._segment_start = 0
        self._segment_end = 0
        self._bg_pos = None
        self._bg_euler = None
        self._bg_grip = None
        self._stationary_intervals = []
        self.result_text.setPlaceholderText("点击“智能分析”查看结果")
        self.canvas.draw_idle()

    def _draw_stationary_bars(self):
        """根据当前分析结果，在三幅子图中同步绘制静止区间灰色条。"""
        # 清除旧条（容错：可能已被 ax.clear() 移除）
        for bar in self._stationary_bars:
            try:
                bar.remove()
            except Exception:
                pass
        self._stationary_bars = []

        if not self._stationary_intervals or self._data is None:
            return

        if self._segment_mode:
            seg_s, seg_e = self._segment_start, self._segment_end
        else:
            seg_s, seg_e = 0, self._total_frames - 1

        for s_ep, e_ep in self._stationary_intervals:
            vs = max(s_ep, seg_s)
            ve = min(e_ep, seg_e)
            if vs > ve:
                continue
            if self._segment_mode:
                x0 = vs - seg_s - 0.5
                x1 = ve - seg_s + 0.5
            else:
                x0 = vs - 0.5
                x1 = ve + 0.5

            # 在位置、欧拉角、夹爪子图中同步标注
            first = not self._stationary_bars
            for ax in self.axes:
                bar = ax.axvspan(
                    x0, x1, color='#AAAAAA', alpha=0.35, zorder=0,
                    label='静止区间' if (first and ax is self.ax_pos) else ''
                )
                self._stationary_bars.append(bar)
                first = False

    # ==================== 自动语言标注（子任务分段） ====================
    def _draw_subtask_bars(self):
        """在夹爪曲线（及其它子图）横坐标上用不同颜色标注各子任务片段。"""
        if not self._subtask_segments or self._data is None:
            return

        if self._segment_mode:
            seg_s, seg_e = self._segment_start, self._segment_end
        else:
            seg_s, seg_e = 0, self._total_frames - 1

        first = True
        for seg in self._subtask_segments:
            vs = max(seg["start"], seg_s)
            ve = min(seg["end"], seg_e)
            if vs > ve:
                continue
            if self._segment_mode:
                x0 = vs - seg_s - 0.5
                x1 = ve - seg_s + 0.5
            else:
                x0 = vs - 0.5
                x1 = ve + 0.5

            color = seg.get("color", "#4CAF50")
            task = seg.get("task", "")
            # 在三幅子图同步着色（夹爪子图最深，便于辨识）
            for ax in self.axes:
                alpha = 0.30 if ax is self.ax_grip else 0.16
                bar = ax.axvspan(
                    x0, x1, color=color, alpha=alpha, zorder=0,
                    label=task if (first and ax is self.ax_grip) else "",
                )
                self._subtask_bars.append(bar)
            # 段边界竖线（仅在夹爪子图）
            vl = self.ax_grip.axvline(x0, color=color, linewidth=1.0,
                                      linestyle=':', alpha=0.7, zorder=1)
            self._subtask_vlines.append(vl)
            # 子任务名标签（夹爪子图顶部，旋转避免重叠）
            xc = (x0 + x1) / 2.0
            txt = self.ax_grip.text(
                xc, 1.02, task, transform=self.ax_grip.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=8, color=color,
                rotation=0, clip_on=False,
            )
            self._subtask_texts.append(txt)
            first = False

    def _run_smart_segment(self):
        """按力矩+位姿切出指定段数，在曲线上着色，标签表用占位名。"""
        if self._data is None or self._total_frames == 0:
            self.subtask_result_label.setText("请先选择 Episode。")
            return
        n = self.subtask_count_spin.value()
        use_torque = self.subtask_torque_chk.isChecked()
        use_gripper = self.subtask_angle_chk.isChecked()
        try:
            from auto_language_annotate import segment_fixed_count
            intervals = segment_fixed_count(self._data, n, use_torque=use_torque, use_gripper=use_gripper)
        except Exception as ex:
            self.subtask_result_label.setText(f"分段失败: {ex}")
            return
        self._subtask_segments = [{
            "start": int(s), "end": int(e),
            "task": PRESET_LABELS[i] if i < len(PRESET_LABELS) else f"步骤{i + 1}",
            "color": color_for_index(i), "note": "auto-fixed",
        } for i, (s, e) in enumerate(intervals)]
        self._populate_label_table()
        self._redraw_subtask()
        self.subtask_vlm_btn.setEnabled(bool(self._subtask_segments))
        self.subtask_apply_btn.setEnabled(bool(self._subtask_segments))
        self.subtask_result_label.setText(
            f"已切 {len(self._subtask_segments)} 段（力矩={use_torque}, 夹角={use_gripper}）。可点“生成标签(VLM)”自动命名，或双击表格手动改。")

    def _apply_preset_labels(self):
        """把预设名称按当前段数套用到分段（不足取前 N，超出保持 "步骤N"）。"""
        if not self._subtask_segments:
            self.subtask_result_label.setText("请先点“智能分段”切出段落后再套用预设。")
            return
        for i, seg in enumerate(self._subtask_segments):
            seg["task"] = PRESET_LABELS[i] if i < len(PRESET_LABELS) else f"步骤{i + 1}"
        self._populate_label_table()
        self._redraw_subtask()
        self.subtask_result_label.setText(
            f"已套用预设名称（{len(self._subtask_segments)} 段）。可双击表格微调，或点“生成标签(VLM)”用模型重命名。")

    def _populate_label_table(self):
        """用当前分段刷新可编辑标签表（阻塞 itemChanged 避免回环）。"""
        self.subtask_label_table.blockSignals(True)
        self.subtask_label_table.setRowCount(len(self._subtask_segments))
        for i, seg in enumerate(self._subtask_segments):
            it0 = QTableWidgetItem(str(i + 1))
            it0.setFlags(Qt.ItemFlag.ItemIsEnabled)
            it1 = QTableWidgetItem(seg["task"])
            it1.setToolTip(f"帧 {seg['start']}~{seg['end']}（{seg['end']-seg['start']+1} 帧）")
            self.subtask_label_table.setItem(i, 0, it0)
            self.subtask_label_table.setItem(i, 1, it1)
        self.subtask_label_table.blockSignals(False)

    def _on_label_table_edited(self, item):
        """用户在标签表里手动改名 → 同步到分段并重绘着色。"""
        if item.column() != 1:
            return
        row = item.row()
        if 0 <= row < len(self._subtask_segments):
            self._subtask_segments[row]["task"] = item.text().strip() or f"步骤{row + 1}"
            self._redraw_subtask()

    def _on_vlm_label_clicked(self):
        """请求主窗口用 VLM 给当前分段生成标签。"""
        if not self._subtask_segments:
            return
        self.subtask_vlm_btn.setEnabled(False)
        self.subtask_result_label.setText("VLM 分析中（后台），请稍候…")
        self.vlm_label_requested.emit(len(self._subtask_segments))

    def set_segment_labels(self, labels, note_msg=""):
        """主窗口 VLM 线程完成后回调：更新标签表与着色。"""
        if not self._subtask_segments:
            return
        for i, lb in enumerate(labels):
            if i < len(self._subtask_segments):
                self._subtask_segments[i]["task"] = str(lb)
        self._populate_label_table()
        self._redraw_subtask()
        self.subtask_vlm_btn.setEnabled(True)
        self.subtask_result_label.setText(note_msg or f"VLM 已为 {len(labels)} 段生成标签，可手动微调后应用。")

    def _clear_subtask_annotate(self):
        """清除曲线上的子任务颜色标注。"""
        self._subtask_segments = []
        self.subtask_label_table.setRowCount(0)
        self.subtask_vlm_btn.setEnabled(False)
        self.subtask_apply_btn.setEnabled(False)
        self.subtask_result_label.setText("（已清除）设段数 → 智能分段 → 生成标签 → 应用")
        self._plot_data(
            self._segment_start if self._segment_mode else 0,
            self._segment_end if self._segment_mode else self._total_frames - 1,
            self._get_current_rel_frame(),
            x_label="片段内帧序号" if self._segment_mode else "帧序号",
        )
        self._apply_view()

    def _redraw_subtask(self):
        """重绘曲线以反映当前分段着色。"""
        self._plot_data(
            self._segment_start if self._segment_mode else 0,
            self._segment_end if self._segment_mode else self._total_frames - 1,
            self._get_current_rel_frame(),
            x_label="片段内帧序号" if self._segment_mode else "帧序号",
        )
        self._apply_view()

    def _on_apply_subtask_clicked(self):
        """请求主窗口把预览结果写回数据集。"""
        if not self._subtask_segments:
            return
        self.subtask_apply_requested.emit(list(self._subtask_segments))

    def _plot_data(self, slice_start, slice_end, rel_frame, x_label="帧序号"):
        """内部绘图：传入已经切好的数据切片和当前帧（均已是当前模式的坐标）。"""
        for ax in self.axes:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.5)
        self.ax_torque.clear()
        self.ax_torque.grid(False)

        # ax.clear() 已移除所有旧 artist，清空本地缓存避免重复 remove
        self._stationary_bars = []
        self._subtask_bars = []
        self._subtask_texts = []
        self._subtask_vlines = []

        self.ax_grip.set_xlabel(x_label)

        title = "Episode 末端位姿变化曲线"
        if self._segment_mode:
            title += f"  [片段 {self._segment_start}~{self._segment_end}]"
        self.ax_pos.set_title(title)

        if self._data is None:
            self._empty_plot()
            return

        n = len(self._data.get('x', []))
        if n == 0:
            self._empty_plot()
            return

        s = max(0, min(slice_start, slice_end, n - 1))
        e = min(n, max(slice_start, slice_end, 0) + 1)
        if e <= s:
            e = s + 1
        indices = np.arange(e - s)

        # 绘制静止区间灰色条（在曲线下层）
        self._draw_stationary_bars()

        # 绘制子任务语言标注彩色条（在曲线下层、静止条之上）
        self._draw_subtask_bars()

        # 位置曲线（原始值）
        sx = self._data.get('x', [])[s:e]
        sy = self._data.get('y', [])[s:e]
        sz = self._data.get('z', [])[s:e]
        self.ax_pos.set_ylabel("位置 (m)")
        self.lines['x'], = self.ax_pos.plot(indices, sx, 'r-', label='X', linewidth=1.2, zorder=2)
        self.lines['y'], = self.ax_pos.plot(indices, sy, 'g-', label='Y', linewidth=1.2, zorder=2)
        self.lines['z'], = self.ax_pos.plot(indices, sz, 'b-', label='Z', linewidth=1.2, zorder=2)
        self.ax_pos.legend(loc='upper right', fontsize=8)

        # 欧拉角曲线（rx/ry/rz，弧度）
        rx = self._data.get('rx', [])[s:e]
        ry = self._data.get('ry', [])[s:e]
        rz = self._data.get('rz', [])[s:e]
        self.ax_euler.set_ylabel("欧拉角 (rad)")
        self.lines['rx'], = self.ax_euler.plot(indices, rx, 'c-', label='rx', linewidth=1.2, zorder=2)
        self.lines['ry'], = self.ax_euler.plot(indices, ry, 'm-', label='ry', linewidth=1.2, zorder=2)
        self.lines['rz'], = self.ax_euler.plot(indices, rz, 'y-', label='rz', linewidth=1.2, zorder=2)
        self.ax_euler.legend(loc='upper right', fontsize=8)

        # 夹爪曲线（弧度）
        grip = self._data.get('gripper', [])[s:e]
        self.ax_grip.set_ylabel("夹爪 (rad)")
        self.lines['gripper'], = self.ax_grip.plot(
            indices, grip, '#FF5722', label='夹爪', linewidth=1.5, zorder=2
        )
        self.ax_grip.legend(loc='upper left', fontsize=8)

        # 夹爪力矩曲线（Nm，右侧 twin axis；仅有数据时绘制）
        if self._has_torque and 'gripper_torque' in self._data:
            torque = self._data.get('gripper_torque', [])[s:e]
            self.ax_torque.set_ylabel("力矩 (Nm)")
            self.lines['gripper_torque'], = self.ax_torque.plot(
                indices, torque, '#9C27B0', label='力矩', linewidth=1.2,
                alpha=0.8, zorder=2
            )
            self.ax_torque.legend(loc='upper right', fontsize=8)
        else:
            self.ax_torque.set_yticks([])

        # 降噪预览：以虚线叠加平滑后的位置/欧拉角/夹爪曲线
        if self._smoothed is not None:
            sm = self._smoothed
            labeled = False
            for key, color in [('x', 'r'), ('y', 'g'), ('z', 'b')]:
                if key in sm and len(sm[key]) >= e:
                    self.ax_pos.plot(
                        indices, sm[key][s:e], color + '--', alpha=0.85, linewidth=1.0,
                        zorder=2, label='平滑预览' if not labeled else None
                    )
                    labeled = True
            for key, color in [('rx', 'c'), ('ry', 'm'), ('rz', 'y')]:
                if key in sm and len(sm[key]) >= e:
                    self.ax_euler.plot(indices, sm[key][s:e], color + '--',
                                       alpha=0.85, linewidth=1.0, zorder=2)
            if 'gripper' in sm and len(sm['gripper']) >= e:
                self.ax_grip.plot(indices, sm['gripper'][s:e], '#FF5720',
                                  linestyle='--', alpha=0.85, linewidth=1.0, zorder=2)
            if labeled:
                self.ax_pos.legend(loc='upper right', fontsize=8)

        frame = max(0, min(rel_frame, e - s - 1))
        self.vlines = [
            self.ax_pos.axvline(x=frame, color='k', linestyle='--', linewidth=1.5, zorder=3),
            self.ax_euler.axvline(x=frame, color='k', linestyle='--', linewidth=1.5, zorder=3),
            self.ax_grip.axvline(x=frame, color='k', linestyle='--', linewidth=1.5, zorder=3),
        ]

        # 播放标签：显示当前帧号，放在顶部子图 Y 轴可视范围顶部
        y_top = self._y_center + self._y_half_range if self._y_half_range > 0 else 1.0
        self.cursor_label = self.ax_pos.annotate(
            f"帧{frame}",
            xy=(frame, y_top),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=9,
            color='black',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8, edgecolor='black'),
            arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
            zorder=4,
        )

    def _update_analysis_result_text(self):
        """更新分析结果文本。"""
        if not self._stationary_intervals:
            self.result_text.setPlainText(
                "未检测到机械臂静止状态。\n\n"
                "提示：增大“静止窗口”或放宽“变化阈值”可增加检测概率。"
            )
            return
        lines = []
        total = 0
        for s, e in self._stationary_intervals:
            length = e - s + 1
            total += length
            lines.append(f"帧 {s}~{e} （{length} 帧）")
        pct = total / max(self._total_frames, 1) * 100
        result = (
            f"检测到 {len(self._stationary_intervals)} 段静止，"
            f"共 {total}/{self._total_frames} 帧 ({pct:.1f}%)\n\n"
        )
        result += "\n".join(lines)
        self.result_text.setPlainText(result)

    def _run_analysis(self):
        """智能分析：检测 7 个指标同时变化较小的连续帧区间。"""
        if self._data is None or self._total_frames == 0:
            self.result_text.setPlainText("请先选择 Episode。")
            return

        # 检测算法已提取为模块级 detect_stationary_intervals（向量化实现），
        # 供“剪辑所有静止”批量流程在不加载 UI 的情况下直接调用，保证两处结果一致
        self._stationary_intervals = detect_stationary_intervals(
            self._data, self._window_size, self._threshold_pct)
        self._update_analysis_result_text()

        # 重绘以显示灰色条
        self._plot_data(
            self._segment_start if self._segment_mode else 0,
            self._segment_end if self._segment_mode else self._total_frames - 1,
            self._get_current_rel_frame(),
            x_label="片段内帧序号" if self._segment_mode else "帧序号"
        )
        self._apply_view()

    def set_data(self, frames_data, current_frame=0, clear_annotations=True):
        """设置并绘制整个 Episode 的位姿曲线（绝对帧序号）。

        clear_annotations=True（默认，切换 Episode 时）会清空子任务分段与标签；
        False（如点播放重绘同一 episode）则保留分段色块与 VLM 标签，
        仅随 _plot_data 内部的 _draw_subtask_bars 重画色块。
        """
        self._data = frames_data
        self._current_frame = current_frame
        self._segment_mode = False
        self._smoothed = None
        self._stationary_intervals = []
        for bar in self._stationary_bars:
            bar.remove()
        self._stationary_bars = []
        # _plot_data 内 ax.clear() 已移除旧 artist，这里只需清空本地句柄缓存；
        # 仅在 clear_annotations=True（切换 Episode）时才丢弃分段数据与标签。
        self._subtask_bars = []
        self._subtask_texts = []
        self._subtask_vlines = []
        if clear_annotations:
            self._subtask_segments = []
            if hasattr(self, "subtask_apply_btn"):
                self.subtask_apply_btn.setEnabled(False)
            if hasattr(self, "subtask_vlm_btn"):
                self.subtask_vlm_btn.setEnabled(False)
            if hasattr(self, "subtask_label_table"):
                self.subtask_label_table.setRowCount(0)
            if hasattr(self, "subtask_result_label"):
                self.subtask_result_label.setText("（设段数 → 智能分段 → 生成标签 → 应用）")
        else:
            # 保留分段：重绘后按是否有分段恢复按钮可用状态
            has_seg = bool(self._subtask_segments)
            if hasattr(self, "subtask_apply_btn"):
                self.subtask_apply_btn.setEnabled(has_seg)
            if hasattr(self, "subtask_vlm_btn"):
                self.subtask_vlm_btn.setEnabled(has_seg)

        if frames_data is None:
            self._empty_plot()
            return

        x = frames_data.get('x')
        if x is None or len(x) == 0:
            self._empty_plot()
            return

        self._total_frames = len(x)
        self._segment_start = 0
        self._segment_end = self._total_frames - 1

        # 先计算 Y 范围，_plot_data 中的 cursor_label 会用到
        ymin, ymax = self._compute_y_range(['x', 'y', 'z'], 0, self._total_frames - 1)
        self._y_center = (ymin + ymax) / 2.0
        self._y_base_half_range = (ymax - ymin) / 2.0
        self._y_half_range = self._y_base_half_range

        qmin, qmax = self._compute_y_range(['rx', 'ry', 'rz'], 0, self._total_frames - 1)
        self._euler_y_center = (qmin + qmax) / 2.0
        self._euler_y_half_range = max((qmax - qmin) / 2.0, 1e-6)

        gmin, gmax = self._compute_y_range(['gripper'], 0, self._total_frames - 1)
        self._grip_y_center = (gmin + gmax) / 2.0
        self._grip_y_half_range = max((gmax - gmin) / 2.0, 1e-6)

        # 夹爪力矩 Y 轴范围（仅有数据时计算）
        self._has_torque = 'gripper_torque' in (frames_data or {})
        if self._has_torque:
            tmin, tmax = self._compute_y_range(['gripper_torque'], 0, self._total_frames - 1)
            self._torque_y_center = (tmin + tmax) / 2.0
            self._torque_y_half_range = max((tmax - tmin) / 2.0, 1e-6)
        else:
            self._torque_y_center = 0.0
            self._torque_y_half_range = 1.0

        # 先按默认参数完成静止分析（纯数据向量化计算，约 1ms），再一次性绘图——
        # 避免旧流程“先绘一次、_run_analysis 后再绘一次”的两次全量重绘，
        # 切换 Episode 的曲线绘制耗时减半（实测 ~400ms → ~200ms）
        self._stationary_intervals = detect_stationary_intervals(
            frames_data, self._window_size, self._threshold_pct)
        self._update_analysis_result_text()

        self._plot_data(0, self._total_frames - 1, current_frame, x_label="帧序号")

        # 视图中心和基准按整个 Episode 设置；视图中心默认跟随当前播放标签
        self._x_base_half_width = self._total_frames / 2.0
        self._view_half_width = self._x_base_half_width
        self._update_view_center_to_current_frame()

        self._x_zoom_level = 1
        self._y_zoom_level = 1
        self.x_zoom_slider.blockSignals(True)
        self.x_zoom_slider.setValue(1)
        self.x_zoom_slider.blockSignals(False)
        self.y_zoom_slider.blockSignals(True)
        self.y_zoom_slider.setValue(1)
        self.y_zoom_slider.blockSignals(False)

        self._apply_view()
        # 静止分析与灰色条已在上面单次绘图中完成，无需再调 _run_analysis 重绘

    def set_segment_range(self, start, end):
        """切换到片段模式：以相对帧序号 0~N-1 显示 [start, end] 内所有帧的曲线。"""
        if self._data is None or self._total_frames <= 0:
            return
        start = max(0, min(start, self._total_frames - 1))
        end = max(0, min(end, self._total_frames - 1))
        if start > end:
            start, end = end, start
        if start == end:
            return

        self._segment_mode = True
        self._segment_start = start
        self._segment_end = end
        segment_len = end - start + 1

        # 先计算片段 Y 范围
        ymin, ymax = self._compute_y_range(['x', 'y', 'z'], start, end)
        self._y_center = (ymin + ymax) / 2.0
        self._y_base_half_range = (ymax - ymin) / 2.0
        self._y_half_range = self._y_base_half_range

        qmin, qmax = self._compute_y_range(['rx', 'ry', 'rz'], start, end)
        self._euler_y_center = (qmin + qmax) / 2.0
        self._euler_y_half_range = max((qmax - qmin) / 2.0, 1e-6)

        gmin, gmax = self._compute_y_range(['gripper'], start, end)
        self._grip_y_center = (gmin + gmax) / 2.0
        self._grip_y_half_range = max((gmax - gmin) / 2.0, 1e-6)

        if self._has_torque:
            tmin, tmax = self._compute_y_range(['gripper_torque'], start, end)
            self._torque_y_center = (tmin + tmax) / 2.0
            self._torque_y_half_range = max((tmax - tmin) / 2.0, 1e-6)

        rel_frame = max(0, min(self._current_frame - start, segment_len - 1))
        self._plot_data(start, end, rel_frame, x_label="片段内帧序号")

        # 视图中心和基准按片段设置
        self._view_center = segment_len / 2.0
        self._x_base_half_width = segment_len / 2.0
        self._view_half_width = self._x_base_half_width

        self._x_zoom_level = 1
        self._y_zoom_level = 1
        self.x_zoom_slider.blockSignals(True)
        self.x_zoom_slider.setValue(1)
        self.x_zoom_slider.blockSignals(False)
        self.y_zoom_slider.blockSignals(True)
        self.y_zoom_slider.setValue(1)
        self.y_zoom_slider.blockSignals(False)

        self._apply_view()

    def _on_x_zoom_changed(self, value):
        """X 轴缩放：始终以当前播放标签为中心放大两侧曲线（同步三幅子图）。"""
        if self._data is None:
            return
        self._x_zoom_level = max(1, value)
        self._view_half_width = max(1.0, self._x_base_half_width / self._x_zoom_level)
        self._update_view_center_to_current_frame()
        self._apply_view()

    def _on_y_zoom_changed(self, value):
        """Y 轴缩放：仅作用于位置曲线。"""
        if self._data is None:
            return
        self._y_zoom_level = max(1, value)
        self._y_half_range = max(1e-6, self._y_base_half_range / self._y_zoom_level)
        self._apply_view()

    def _reset_zoom(self):
        """重置为显示整个 Episode（绝对帧序号）。"""
        if self._data is None or self._total_frames <= 0:
            return
        # 直接切回全 Episode 视图（同一 episode 重绘，保留子任务标注）
        self.set_data(self._data, self._current_frame, clear_annotations=False)

    def _get_current_rel_frame(self):
        """返回当前帧在当前坐标系（全 Episode 或片段）中的相对位置。"""
        if self._data is None:
            return 0
        if self._segment_mode:
            total_n = self._segment_end - self._segment_start + 1
            return max(0, min(self._current_frame - self._segment_start, total_n - 1))
        else:
            return max(0, min(self._current_frame, self._total_frames - 1))

    def _update_view_center_to_current_frame(self):
        """将视图中心设为当前播放标签所在帧。"""
        self._view_center = float(self._get_current_rel_frame())

    def _update_cursor_artist(self):
        """更新播放竖线和标签的位置/文本。"""
        if not self.vlines:
            return
        rel_frame = self._get_current_rel_frame()
        for vline in self.vlines:
            vline.set_xdata([rel_frame, rel_frame])

        if self.cursor_label is not None:
            # 标签 y 坐标放在位置子图 Y 轴可视范围顶部
            y_top = self._y_center + self._y_half_range
            self.cursor_label.xy = (rel_frame, y_top)
            self.cursor_label.set_text(f"帧{self._current_frame}")

    def set_current_frame(self, frame):
        """更新当前帧：竖线、标签同步滑动；若处于 X 放大状态则视图跟随标签平移。"""
        if not self.vlines or self._data is None:
            return
        new_frame = max(0, min(int(frame), self._total_frames - 1))
        if new_frame == self._current_frame:
            return
        self._current_frame = new_frame
        self._update_view_center_to_current_frame()
        self._update_cursor_artist()

        # 若 X 方向有放大，则整体视图以播放标签为中心平移
        if self._x_zoom_level > 1:
            self._apply_view()
            return

        # 无 X 放大时，使用 blit 只重绘竖线和标签
        if self._bg_pos is not None and self._bg_euler is not None and self._bg_grip is not None:
            try:
                self.canvas.restore_region(self._bg_pos)
                self.canvas.restore_region(self._bg_euler)
                self.canvas.restore_region(self._bg_grip)

                self.ax_pos.draw_artist(self.vlines[0])
                self.ax_euler.draw_artist(self.vlines[1])
                self.ax_grip.draw_artist(self.vlines[2])
                if self.cursor_label is not None:
                    self.ax_pos.draw_artist(self.cursor_label)

                self.canvas.blit(self.ax_pos.bbox)
                self.canvas.blit(self.ax_euler.bbox)
                self.canvas.blit(self.ax_grip.bbox)
                return
            except Exception:
                self._bg_pos = None
                self._bg_euler = None
                self._bg_grip = None
        self.canvas.draw_idle()

    def _apply_view(self):
        """应用当前视图范围并重绘。

        zoom=1 时始终显示整个 Episode/片段；zoom>1 时以播放标签为中心放大两侧。
        """
        if self._data is None:
            return

        # 当前坐标系下的总点数
        if self._segment_mode:
            total_n = self._segment_end - self._segment_start + 1
        else:
            total_n = self._total_frames

        # X 轴范围（同步三幅子图）
        if self._x_zoom_level <= 1:
            # 无放大：显示全部帧
            x_min = 0.0
            x_max = max(0.0, total_n - 1)
        else:
            # 放大：以播放标签为中心
            x_min = max(0.0, self._view_center - self._view_half_width)
            x_max = min(total_n - 1, self._view_center + self._view_half_width)
        if x_max <= x_min:
            x_max = x_min + 1
        for ax in self.axes:
            ax.set_xlim(x_min, x_max)

        # 位置 Y 轴范围（受 Y 缩放控制）
        self.ax_pos.set_ylim(
            self._y_center - self._y_half_range,
            self._y_center + self._y_half_range
        )

        # 欧拉角 Y 轴范围（固定自适应）
        self.ax_euler.set_ylim(
            self._euler_y_center - self._euler_y_half_range,
            self._euler_y_center + self._euler_y_half_range
        )

        # 夹爪 Y 轴范围（固定自适应）
        self.ax_grip.set_ylim(
            self._grip_y_center - self._grip_y_half_range,
            self._grip_y_center + self._grip_y_half_range
        )

        # 夹爪力矩 Y 轴范围（右侧 twin axis，固定自适应）
        self.ax_torque.set_ylim(
            self._torque_y_center - self._torque_y_half_range,
            self._torque_y_center + self._torque_y_half_range
        )

        # 竖线和标签
        self._update_cursor_artist()

        self.canvas.draw_idle()
        # 绘制完成后缓存三幅子图背景
        self._capture_backgrounds()

    def clear(self):
        """清空曲线。"""
        self._data = None
        self._smoothed = None
        self._current_frame = 0
        self._total_frames = 0
        self._segment_mode = False
        self._segment_start = 0
        self._segment_end = 0
        self._view_center = 0.0
        self._view_half_width = 0.0
        self._x_base_half_width = 0.0
        self._y_center = 0.0
        self._y_half_range = 0.0
        self._y_base_half_range = 0.0
        self._x_zoom_level = 1
        self._y_zoom_level = 1
        self._euler_y_center = 0.0
        self._euler_y_half_range = 0.0
        self._grip_y_center = 0.0
        self._grip_y_half_range = 0.0
        self._torque_y_center = 0.0
        self._torque_y_half_range = 1.0
        self._has_torque = False
        self._bg_pos = None
        self._bg_euler = None
        self._bg_grip = None
        self._stationary_intervals = []
        for bar in self._stationary_bars:
            bar.remove()
        self._stationary_bars = []
        self.x_zoom_slider.blockSignals(True)
        self.x_zoom_slider.setValue(1)
        self.x_zoom_slider.blockSignals(False)
        self.y_zoom_slider.blockSignals(True)
        self.y_zoom_slider.setValue(1)
        self.y_zoom_slider.blockSignals(False)
        self._empty_plot()

# ==================== Episode 删除工具 ====================

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
    """将 DataFrame 按 chunk_size 分块写入 parquet。"""
    os.makedirs(out_dir, exist_ok=True)
    for i, start in enumerate(range(0, max(len(df), 1), chunk_size)):
        df.iloc[start:start + chunk_size].to_parquet(
            os.path.join(out_dir, f"file-{i:03d}.parquet"), index=False)


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


def delete_episode(dataset_root, episode_idx):
    """安全删除指定 episode，重新编排所有索引，保证数据一致性。"""
    meta_dir  = os.path.join(dataset_root, "meta")
    data_dir  = os.path.join(dataset_root, "data")
    video_dir = os.path.join(dataset_root, "videos")

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
    new_ep_df = _sync_episode_data_pointers(new_ep_df, df_new)

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

    info_path = os.path.join(meta_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        if "total_frames"   in info: info["total_frames"]   -= n_del
        if "total_episodes" in info: info["total_episodes"] -= 1
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


def delete_episodes(dataset_root, episode_indices):
    """批量安全删除多个 episode：单次重写 data/meta/stats 并重排索引，保证数据一致性。

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
    new_ep_df = _sync_episode_data_pointers(new_ep_df, df_new)

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

    info_path = os.path.join(meta_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        if "total_frames"   in info: info["total_frames"]   -= n_del_total
        if "total_episodes" in info: info["total_episodes"] -= len(del_sorted)
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


# ==================== 删除后台线程 ====================
class DeleteEpisodeThread(QThread):
    """在独立线程中执行 episode 删除，避免 UI 卡顿。"""
    finished_ok = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, dataset_root, episode_idx):
        super().__init__()
        self.dataset_root = dataset_root
        self.episode_idx = episode_idx

    def run(self):
        try:
            result = delete_episode(self.dataset_root, self.episode_idx)
            self.finished_ok.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class BatchDeleteEpisodesThread(QThread):
    """在独立线程中批量删除 episode（单次重写），避免 UI 卡顿。"""
    finished_ok = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, dataset_root, episode_indices):
        super().__init__()
        self.dataset_root = dataset_root
        self.episode_indices = list(episode_indices)

    def run(self):
        try:
            result = delete_episodes(self.dataset_root, self.episode_indices)
            self.finished_ok.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


# ==================== 批量删除 Episode 对话框 ====================
class BatchDeleteDialog(QDialog):
    """批量 Episode 选择对话框：任务名筛选 + 复选框多选 + 汇总确认。

    默认用作“批量删除片段”的删除确认对话框；通过 title/ok_text/ok_color
    参数复用于“剪辑所有静止”的 Episode 筛选（同一套筛选交互）。

    通过 selected_indices() 返回选中的 episode_index 列表（已按升序排列）。
    """

    def __init__(self, parent, episode_rows, fps, title="批量删除片段",
                 ok_text="🗑 删除选中", ok_color="#9C27B0"):
        """
        Args:
            episode_rows: [(episode_index, length, task_name), ...]
            fps: 数据集帧率（时长列显示用）
            title: 对话框标题
            ok_text: 确认按钮文字（不含数量后缀）
            ok_color: 确认按钮背景色
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 540)
        self._rows = list(episode_rows)
        self._fps = fps if fps and fps > 0 else 20.0
        self._ok_text = ok_text

        layout = QVBoxLayout(self)

        # ---- 筛选行：任务名下拉 + 关键词 ----
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("任务筛选:"))
        self.task_filter_combo = QComboBox()
        self.task_filter_combo.addItem("全部任务")
        for name in sorted({r[2] for r in self._rows if r[2]}):
            self.task_filter_combo.addItem(name)
        self.task_filter_combo.currentTextChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.task_filter_combo, stretch=1)

        filter_layout.addWidget(QLabel("关键词:"))
        self.keyword_edit = QLineEdit()
        self.keyword_edit.setPlaceholderText("按编号或任务名过滤…")
        self.keyword_edit.textChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.keyword_edit, stretch=1)
        layout.addLayout(filter_layout)

        # ---- 快捷选择按钮 ----
        sel_layout = QHBoxLayout()
        self.select_filtered_btn = QPushButton("☑ 勾选筛选结果")
        self.select_filtered_btn.setToolTip("勾选当前可见（未被过滤）的全部 Episode")
        self.select_filtered_btn.clicked.connect(lambda: self._set_visible_checked(True))
        sel_layout.addWidget(self.select_filtered_btn)

        self.unselect_filtered_btn = QPushButton("☐ 取消勾选筛选结果")
        self.unselect_filtered_btn.setToolTip("取消勾选当前可见的全部 Episode")
        self.unselect_filtered_btn.clicked.connect(lambda: self._set_visible_checked(False))
        sel_layout.addWidget(self.unselect_filtered_btn)

        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        sel_layout.addWidget(self.select_all_btn)

        self.select_none_btn = QPushButton("全不选")
        self.select_none_btn.clicked.connect(lambda: self._set_all_checked(False))
        sel_layout.addWidget(self.select_none_btn)
        sel_layout.addStretch()
        layout.addLayout(sel_layout)

        # ---- Episode 复选表格 ----
        self.table = QTableWidget(len(self._rows), 5)
        self.table.setHorizontalHeaderLabels(["勾选", "Episode", "帧数", "时长(s)", "任务名"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)

        for row_i, (ep_idx, length, task_name) in enumerate(self._rows):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            chk.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row_i, 0, chk)

            idx_item = QTableWidgetItem(str(ep_idx))
            idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_i, 1, idx_item)

            len_item = QTableWidgetItem(str(length))
            len_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_i, 2, len_item)

            dur_item = QTableWidgetItem(f"{length / self._fps:.1f}")
            dur_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_i, 3, dur_item)

            self.table.setItem(row_i, 4, QTableWidgetItem(task_name))

        self.table.itemChanged.connect(self._update_summary)
        layout.addWidget(self.table, stretch=1)

        # ---- 汇总信息 + 确认/取消 ----
        bottom_layout = QHBoxLayout()
        self.summary_label = QLabel("已选 0 个 Episode（0 帧，0.0 秒）")
        self.summary_label.setStyleSheet("QLabel { color: #c62828; font-weight: bold; }")
        bottom_layout.addWidget(self.summary_label)
        bottom_layout.addStretch()

        self.ok_btn = QPushButton(f"{self._ok_text} (0)")
        self.ok_btn.setEnabled(False)
        self.ok_btn.setStyleSheet(
            f"QPushButton {{ background-color: {ok_color}; color: white; font-weight: bold; padding: 6px 16px; }}"
            "QPushButton:disabled { background-color: #ccc; color: #888; }"
        )
        self.ok_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(self.ok_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet("QPushButton { padding: 6px 16px; }")
        cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(cancel_btn)
        layout.addLayout(bottom_layout)

        self._update_summary()

    def _apply_filter(self):
        """按任务名下拉 + 关键词过滤表格行（隐藏不匹配行，不改动勾选状态）。"""
        task = self.task_filter_combo.currentText()
        kw = self.keyword_edit.text().strip().lower()
        for row_i, (ep_idx, _length, task_name) in enumerate(self._rows):
            visible = True
            if task != "全部任务" and task_name != task:
                visible = False
            if kw and kw not in str(ep_idx) and kw not in task_name.lower():
                visible = False
            self.table.setRowHidden(row_i, not visible)

    def _set_visible_checked(self, checked):
        """勾选/取消勾选当前可见（未被过滤）的行。"""
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row_i in range(len(self._rows)):
            if self.table.isRowHidden(row_i):
                continue
            item = self.table.item(row_i, 0)
            if item is not None:
                item.setCheckState(state)

    def _set_all_checked(self, checked):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row_i in range(len(self._rows)):
            item = self.table.item(row_i, 0)
            if item is not None:
                item.setCheckState(state)

    def _update_summary(self):
        indices = set(self.selected_indices())
        n_frames = sum(length for ep_idx, length, _t in self._rows
                       if ep_idx in indices)
        duration = n_frames / self._fps
        self.summary_label.setText(
            f"已选 {len(indices)} 个 Episode（{n_frames} 帧，{duration:.1f} 秒）")
        self.ok_btn.setText(f"{self._ok_text} ({len(indices)})")
        self.ok_btn.setEnabled(len(indices) > 0)

    def selected_indices(self):
        """返回勾选的 episode_index 列表（升序）。"""
        result = []
        for row_i, (ep_idx, _length, _task) in enumerate(self._rows):
            item = self.table.item(row_i, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                result.append(ep_idx)
        return sorted(result)


class TrimEpisodeThread(QThread):
    """在独立线程中执行 episode 片段剪辑。"""
    finished_ok = pyqtSignal(str, object, object, object)
    error = pyqtSignal(str)

    def __init__(self, dataset_root, episode_idx, start_frame, end_frame, backup_enabled=True):
        super().__init__()
        self.dataset_root = dataset_root
        self.episode_idx = episode_idx
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.backup_enabled = backup_enabled

    def run(self):
        try:
            result, df_new, new_ep_df, info = trim_episode(
                self.dataset_root, self.episode_idx, self.start_frame, self.end_frame,
                backup_enabled=self.backup_enabled
            )
            self.finished_ok.emit(result, df_new, new_ep_df, info)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class TrimStationaryThread(QThread):
    """在独立线程中执行“剪辑静止”：删除 episode 内多个静止区间。"""
    finished_ok = pyqtSignal(str, object, object, object)
    error = pyqtSignal(str)

    def __init__(self, dataset_root, episode_idx, intervals, backup_enabled=True):
        super().__init__()
        self.dataset_root = dataset_root
        self.episode_idx = episode_idx
        self.intervals = intervals
        self.backup_enabled = backup_enabled

    def run(self):
        try:
            result, df_new, new_ep_df, info = trim_episode_ranges(
                self.dataset_root, self.episode_idx, self.intervals,
                backup_enabled=self.backup_enabled
            )
            self.finished_ok.emit(result, df_new, new_ep_df, info)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


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


# 保留旧函数名作为兼容别名（不再被新代码使用）
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
        for name in ["stats.json", "stats.jsonl", "info.json"]:
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

    ep_df, ep_fmt, ep_files = _read_episodes(str(meta_dir))
    if episode_idx < 0 or episode_idx >= len(ep_df):
        raise ValueError(f"episode_idx {episode_idx} 超出范围（共 {len(ep_df)} 个）")

    if "dataset_from_index" in ep_df.columns and "dataset_to_index" in ep_df.columns:
        from_idx = int(ep_df.loc[episode_idx, "dataset_from_index"])
        to_idx = int(ep_df.loc[episode_idx, "dataset_to_index"])
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

    ep_df, ep_fmt, ep_files = _read_episodes(str(meta_dir))
    if episode_idx < 0 or episode_idx >= len(ep_df):
        raise ValueError(f"episode_idx {episode_idx} 超出范围（共 {len(ep_df)} 个）")

    if "dataset_from_index" in ep_df.columns and "dataset_to_index" in ep_df.columns:
        from_idx = int(ep_df.loc[episode_idx, "dataset_from_index"])
        to_idx = int(ep_df.loc[episode_idx, "dataset_to_index"])
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

    # 按原始分块布局写回：每个文件行数不变，data 文件指针保持有效
    pos = 0
    for p, n in zip(data_files, file_rows):
        df_all.iloc[pos:pos + n].to_parquet(str(p), index=False)
        pos += n
    _recompute_global_stats(str(meta_dir), df_all)

    method_names = {"savgol": "Savitzky-Golay", "moving_avg": "移动平均", "gaussian": "高斯滤波"}
    result = (f"降噪完成\n"
              f"方法: {method_names.get(method, method)} (窗口={window}, 阶数={polyorder}, σ={sigma})\n"
              f"作用对象: {', '.join(cols)}\n"
              f"处理 Episode 数: {processed}\n"
              f"总帧数不变: {len(df_all)}（元数据与视频未改动，stats.json 已重算）")
    return result, df_all


class DenoiseThread(QThread):
    """在独立线程中执行轨迹平滑降噪（单个或全部 Episode）。"""
    finished_ok = pyqtSignal(str, object)   # result, df_new
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)    # 当前进度, 总数, 描述

    def __init__(self, dataset_root, episode_idx, params, backup_enabled=True):
        super().__init__()
        self.dataset_root = dataset_root
        self.episode_idx = episode_idx
        self.params = params
        self.backup_enabled = backup_enabled

    def run(self):
        try:
            result, df_new = denoise_dataset(
                self.dataset_root, self.episode_idx, self.params,
                backup_enabled=self.backup_enabled,
                progress_cb=lambda c, t, m: self.progress.emit(c, t, m),
            )
            self.finished_ok.emit(result, df_new)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


# ==================== EpisodeSampler ====================
class EpisodeSampler(torch.utils.data.Sampler):
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


# ==================== 关节角度反解线程 ====================
class IKSolveThread(QThread):
    """后台批量反解整段 Episode 的关节角度，避免阻塞 UI。

    支持 cancel()：切换 episode 时旧任务立即作废（逐帧检查取消标志），
    调用方无需 wait() 阻塞 UI 线程。
    """
    finished_ok = pyqtSignal(object, object)  # (joint_angles (N,7) ndarray | None, episode_idx)
    failed = pyqtSignal(str, int)             # (error_msg, episode_idx)

    def __init__(self, states, episode_idx):
        super().__init__()
        self._states = states
        self._episode_idx = episode_idx
        self._cancel = threading.Event()

    def cancel(self):
        """请求取消：求解循环在下一帧边界退出，不再发出结果。"""
        self._cancel.set()

    def run(self):
        try:
            angles = solve_episode_joint_angles(self._states, cancel_event=self._cancel)
            if self._cancel.is_set():
                return  # 已取消：结果必然过期（调用方已切换到别的 episode）
            self.finished_ok.emit(angles, self._episode_idx)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}", self._episode_idx)


# ==================== 机械臂 3D 可视化面板 ====================
class RobotArmReplayWidget(QWidget):
    """数据回放用的机械臂 3D 可视化面板。

    复用 debugger.widgets.viewer_3d.Viewer3DPanel（含“重置视角/线框/关闭视图”按键），
    在其基础上叠加：根据预反解的关节角度数组，按帧刷新机械臂姿态，并显示各关节角度。
    """

    def __init__(self, parent=None, urdf_path=None, mesh_dir=None):
        super().__init__(parent)
        self._joint_angles = None      # (N, 7) 预反解结果，None 表示未就绪
        self._current_frame = 0
        self._viewer = None
        self._enabled = _KIN is not None
        self._init_ui(urdf_path, mesh_dir)

    def _init_ui(self, urdf_path, mesh_dir):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        if not self._enabled:
            label = QLabel("逆运动学模块 (pinocchio) 未加载，机械臂反解不可用")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #b00; font-size: 12px; padding: 24px;")
            layout.addWidget(label)
            return

        try:
            from debugger.widgets.viewer_3d import Viewer3DPanel
        except Exception as e:
            label = QLabel(f"无法加载 3D 可视化模块:\n{e}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            label.setStyleSheet("color: #b00; font-size: 12px; padding: 20px;")
            layout.addWidget(label)
            self._enabled = False
            return

        self._viewer = Viewer3DPanel(self, urdf_path=urdf_path, mesh_dir=mesh_dir)
        # 回放场景无需拖拽，关闭拖拽相关高亮/事件过滤
        try:
            self._viewer.set_drag_mode(False)
        except Exception:
            pass
        layout.addWidget(self._viewer, stretch=1)

        # 关节角度信息标签
        self._info_label = QLabel("机械臂关节角度: 未加载 Episode")
        self._info_label.setStyleSheet(
            "QLabel { color: #333; font-size: 11px; padding: 3px 6px;"
            " background-color: #f0f0f0; border-radius: 3px; }"
        )
        self._info_label.setWordWrap(True)
        self._info_label.setMinimumHeight(20)
        layout.addWidget(self._info_label)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def ensure_initialized(self):
        """惰性初始化 3D 模型（首次显示时调用，避免启动卡顿）。"""
        if not self._enabled or self._viewer is None:
            return False
        try:
            if not getattr(self._viewer, "_initialized", False):
                return self._viewer.initialize_model()
        except Exception as e:
            print(f"[RobotArm] 3D 模型初始化失败: {e}")
            return False
        return True

    def clear(self):
        """清空预反解数据并复位姿态。"""
        self._joint_angles = None
        self._current_frame = 0
        if self._enabled and self._viewer is not None:
            try:
                self._render_zero()
            except Exception:
                pass
        if hasattr(self, "_info_label"):
            self._info_label.setText("机械臂关节角度: 未加载 Episode")

    def _render_zero(self):
        if self._viewer is None:
            return
        angles = {name: 0.0 for name in ARM_JOINT_NAMES}
        if self._viewer._model is not None:
            self._viewer._render_pose(angles)

    # ------------------------------------------------------------------
    # 数据与帧更新
    # ------------------------------------------------------------------
    def set_joint_angles(self, angles):
        """设置预反解的整段关节角度数组 (N, 7)。"""
        self._joint_angles = np.asarray(angles) if angles is not None else None
        self._current_frame = 0
        self.show_frame(0)

    def show_frame(self, frame_idx):
        """刷新到指定帧的机械臂姿态。"""
        if not self._enabled or self._viewer is None:
            return
        if self._joint_angles is None or len(self._joint_angles) == 0:
            return
        frame_idx = max(0, min(int(frame_idx), len(self._joint_angles) - 1))
        self._current_frame = frame_idx

        if not self.ensure_initialized():
            return

        row = self._joint_angles[frame_idx]
        angles = {name: float(row[i]) for i, name in enumerate(ARM_JOINT_NAMES)}
        try:
            self._viewer._render_pose(angles)
        except Exception as e:
            print(f"[RobotArm] 渲染失败: {e}")
            return

        # 更新关节角度文本（度）
        degs = [math.degrees(a) for a in row]
        text = "机械臂关节角度 (°):  " + "  ".join(
            f"L{i+1}={d:+.1f}" for i, d in enumerate(degs[:6])
        )
        text += f"  |  夹爪={degs[6]:+.1f}°  |  帧 {frame_idx}"
        self._info_label.setText(text)

    def current_frame(self):
        return self._current_frame

    def has_data(self):
        return self._joint_angles is not None and len(self._joint_angles) > 0


# ==================== 回放线程 ====================
class PlaybackThread(QThread):
    """真机数据纯图像回放线程，支持暂停/继续"""
    frame_data = pyqtSignal(object, object, object, object, int, object, object)  # cam1_rgb, cam2_rgb, action, state, step, timestamp, task_name
    frame_updated = pyqtSignal(int, int)  # current_frame, total_frames
    episode_finished = pyqtSignal()
    playback_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    
    def __init__(self, dataset: OfflineDataset, episode_index: int, speed=1.0, auto_play=False):
        super().__init__()
        self.dataset = dataset
        self.episode_index = episode_index
        self.speed = speed
        self.auto_play = auto_play
        self._is_running = True
        self._is_paused = False
        self._jump_to_frame = None
        self._jump_resume_after = False
        self._mutex = QMutex()
        self._pause_condition = QWaitCondition()
        
    def stop(self):
        self._is_running = False
        self._is_paused = False
        self._jump_to_frame = None
        self._pause_condition.wakeAll()
                
    def pause(self):
        self._is_paused = True
        self.status_changed.emit("已暂停")
        
    def resume(self):
        self._mutex.lock()
        self._is_paused = False
        self._pause_condition.wakeAll()
        self._mutex.unlock()
        self.status_changed.emit("播放中")
        
    def is_paused(self):
        return self._is_paused
        
    def jump_to_frame(self, frame_idx, resume_after=False):
        """跳转到指定帧
        
        Args:
            frame_idx: 目标帧索引
            resume_after: 跳转后是否继续播放。False=保持暂停，True=继续播放
        """
        self._mutex.lock()
        self._jump_to_frame = frame_idx
        self._jump_resume_after = resume_after
        self._pause_condition.wakeAll()
        self._mutex.unlock()
        
    def _check_pause(self):
        self._mutex.lock()
        if self._is_paused:
            self.status_changed.emit("已暂停")
            self._pause_condition.wait(self._mutex)
            self.status_changed.emit("播放中")
        self._mutex.unlock()
        
    def _emit_frame(self, ep_idx, step, ep_frames, total_frames):
        """发送指定帧的数据"""
        row = ep_frames.iloc[step]
        
        # 获取图像
        cam1_img = self._load_image(ep_idx, step, "observation.images.cam1")
        cam2_img = self._load_image(ep_idx, step, "observation.images.cam2")
        
        # 获取 action 和 state
        action = np.array(row.get("action", [0] * 7))
        state = np.array(row.get("observation.state", [0] * 7))
        timestamp = row.get("timestamp", step / self.dataset.fps)
        task_index = int(row.get("task_index", 0))
        
        # 获取任务名称（按 task_index 值查找，不能用列表位置索引）
        task_name = self.dataset.task_name_for_index(task_index)
        
        # 发送帧数据
        self.frame_data.emit(cam1_img, cam2_img, action, state, step, timestamp, task_name)
        self.frame_updated.emit(step + 1, total_frames)
        
    def run(self):
        try:
            episodes_to_play = range(len(self.dataset.episodes)) if self.auto_play else [self.episode_index]
            
            for ep_idx in episodes_to_play:
                if not self._is_running:
                    break
                    
                self.status_changed.emit(f"正在播放 Episode {ep_idx}")
                
                # 获取 episode 数据
                ep_frames = self.dataset.get_episode_frames(ep_idx)
                total_frames = len(ep_frames)
                
                hz = int(self.dataset.fps * self.speed)
                frame_interval = 1.0 / hz
                last_frame_time = time.perf_counter()
                
                step = 0
                while step < total_frames:
                    if not self._is_running:
                        break
                    
                    # 检查是否有帧跳转请求
                    self._mutex.lock()
                    if self._jump_to_frame is not None:
                        step = max(0, min(self._jump_to_frame, total_frames - 1))
                        resume_after_jump = self._jump_resume_after
                        self._jump_to_frame = None
                        self._mutex.unlock()
                        # 发送跳转后的帧数据
                        self._emit_frame(ep_idx, step, ep_frames, total_frames)
                        # 如果不需要继续播放，则暂停，等待下一次循环被 _check_pause 阻塞
                        if not resume_after_jump:
                            self._is_paused = True
                            self.status_changed.emit("已暂停")
                        else:
                            step += 1
                        continue
                    self._mutex.unlock()
                    
                    self._check_pause()
                    
                    if not self._is_running:
                        break
                    
                    # 控制帧率
                    current_time = time.perf_counter()
                    elapsed = current_time - last_frame_time
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
                    last_frame_time = time.perf_counter()
                    
                    # 发送当前帧
                    self._emit_frame(ep_idx, step, ep_frames, total_frames)
                    
                    step += 1
                    
                if not self._is_running:
                    break
                    
                self.episode_finished.emit()
                
            self.playback_finished.emit()
                    
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
    
    def _load_image(self, episode_index, frame_index, camera_name):
        """加载图像，支持三种存储方式：
        1. videos/ 目录下的视频文件（LeRobot 0.4.4 默认格式）
        2. images/ 目录下的 PNG 文件
        3. parquet 中的 bytes 字段（LeRobot 0.4.4 use_videos=False 模式）
        """
        # 方式1: 优先从 videos/ 目录加载（当前默认格式）
        if self.dataset.is_video_feature(camera_name):
            try:
                frame = self.dataset.get_video_frame(episode_index, frame_index, camera_name)
                if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                    return frame
            except Exception as e:
                print(f"[Playback] 从视频加载失败: {e}")
        
        # 方式2: 尝试从 images/ 目录加载
        img_path = self.dataset.get_image_path(episode_index, frame_index, camera_name)
        if img_path and os.path.exists(img_path):
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            return np.array(img)
        
        # 方式3: 从 parquet 的 bytes 字段解码（LeRobot 0.4.4 格式）
        try:
            ep_frames = self.dataset.get_episode_frames(episode_index)
            if frame_index < len(ep_frames):
                row = ep_frames.iloc[frame_index]
                if camera_name in row:
                    img_data = row[camera_name]
                    # LeRobot 0.4.4 存储为 {'bytes': b'PNG...'} 或 {'path': '...'}
                    if isinstance(img_data, dict):
                        if 'bytes' in img_data and img_data['bytes']:
                            from PIL import Image
                            import io
                            img = Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
                            return np.array(img)
                        elif 'path' in img_data and img_data['path']:
                            from PIL import Image
                            img = Image.open(img_data['path']).convert("RGB")
                            return np.array(img)
        except Exception as e:
            print(f"[Playback] 从 parquet 加载图像失败: {e}")
        
        # 返回黑色图像作为占位
        return np.zeros((240, 320, 3), dtype=np.uint8)


# ==================== 数据集统计分布分析对话框 ====================
class DatasetStatsDialog(QDialog):
    """分析所选数据集所有 Episode 的位置 xyz、欧拉角、夹爪角度分布。

    以 Episode 编号为横轴，每个指标绘制 min~max 范围带与均值曲线，
    超出全局 均值±3σ 的 Episode 标记为异常值，便于发现极值与异常轨迹。
    自动兼容 7 维欧拉角新格式与 8 维四元数旧格式（旧格式自动转欧拉角）。
    """

    def __init__(self, parent, dataset):
        super().__init__(parent)
        self.dataset = dataset
        self.setWindowTitle("数据集统计分布分析")
        self.resize(1200, 900)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hint = QLabel(
            "横轴为 Episode 编号；每幅子图灰色带表示该指标在该 Episode 内的最小值~最大值范围，"
            "蓝色实线为均值；红色点为异常值（该 Episode 均值偏离全局均值超过 3σ）。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #555; font-size: 11px; padding: 2px; }")
        layout.addWidget(hint)

        self.figure = Figure(figsize=(11, 9), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas, stretch=5)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(180)
        layout.addWidget(self.summary_text, stretch=1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet("QPushButton { padding: 6px 28px; }")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self._compute_and_plot()

    def _compute_and_plot(self):
        n_eps = len(self.dataset.episodes)
        if n_eps == 0:
            self.summary_text.setText("数据集没有 Episode。")
            return

        df = self.dataset.data_df
        if "observation.state" not in df.columns:
            self.summary_text.setText("数据中没有 observation.state 列，无法分析。")
            return

        try:
            states = np.vstack(df["observation.state"].to_numpy()).astype(np.float64)
        except Exception as e:
            self.summary_text.setText(f"解析 observation.state 失败: {e}")
            return
        if states.ndim != 2 or states.shape[1] < 3:
            self.summary_text.setText("observation.state 列数不足，无法分析。")
            return

        n_rows = states.shape[0]
        zeros = np.zeros(n_rows)
        # 自动识别格式：>=8 列为旧四元数格式，==7 为新欧拉角格式
        if states.shape[1] >= 8:
            qx = states[:, 3].astype(np.float64)
            qy = states[:, 4].astype(np.float64)
            qz = states[:, 5].astype(np.float64)
            qw = states[:, 6].astype(np.float64)
            rx_c = np.empty(n_rows); ry_c = np.empty(n_rows); rz_c = np.empty(n_rows)
            for i in range(n_rows):
                rx_c[i], ry_c[i], rz_c[i] = quat_to_rpy_recording(
                    float(qw[i]), float(qx[i]), float(qy[i]), float(qz[i]))
            metric_cols = {
                'x': states[:, 0],
                'y': states[:, 1],
                'z': states[:, 2],
                'rx': rx_c,
                'ry': ry_c,
                'rz': rz_c,
                'gripper': np.radians(states[:, 7]),
            }
        else:
            metric_cols = {
                'x': states[:, 0],
                'y': states[:, 1],
                'z': states[:, 2],
                'rx': states[:, 3] if states.shape[1] > 3 else zeros,
                'ry': states[:, 4] if states.shape[1] > 4 else zeros,
                'rz': states[:, 5] if states.shape[1] > 5 else zeros,
                'gripper': states[:, 6] if states.shape[1] > 6 else zeros,
            }

        if "episode_index" in df.columns:
            ep_col = df["episode_index"].to_numpy()
        else:
            ep_col = np.zeros(n_rows, dtype=int)

        # 按 episode 排序后分组，避免逐 episode 全表扫描
        sort_idx = np.argsort(ep_col, kind='stable')
        ep_sorted = ep_col[sort_idx]
        unique_eps, start_idx = np.unique(ep_sorted, return_index=True)
        split_points = start_idx[1:]
        x = np.arange(len(unique_eps))

        metrics = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
        titles = ['位置 X (m)', '位置 Y (m)', '位置 Z (m)', '欧拉角 rx (rad)',
                  '欧拉角 ry (rad)', '欧拉角 rz (rad)', '夹爪角度 (rad)']

        stats = {}
        for m in metrics:
            vals_sorted = metric_cols[m][sort_idx]
            groups = np.split(vals_sorted, split_points)
            stats[m] = {
                'min': np.array([g.min() for g in groups], dtype=float),
                'max': np.array([g.max() for g in groups], dtype=float),
                'mean': np.array([g.mean() for g in groups], dtype=float),
                'std': np.array([g.std() for g in groups], dtype=float),
            }

        # 绘图
        self.figure.clear()
        color_band = '#CFD8DC'
        color_line = '#1565C0'
        color_outlier = '#E53935'

        outlier_eps = set()
        summary_lines = [f"全局统计（共 {len(unique_eps)} 个 Episode，{n_rows} 帧）：\n"]

        for idx, m in enumerate(metrics):
            ax = self.figure.add_subplot(4, 2, idx + 1)
            s = stats[m]
            ax.fill_between(x, s['min'], s['max'], color=color_band, alpha=0.7, label='min~max 范围')
            ax.plot(x, s['mean'], color=color_line, linewidth=1.2, marker='.', markersize=3, label='均值')

            gmean = float(np.mean(s['mean']))
            gstd = float(np.std(s['mean']))
            thr = 3.0 * max(gstd, 1e-12)
            out_mask = np.abs(s['mean'] - gmean) > thr
            if np.any(out_mask):
                ax.scatter(x[out_mask], s['mean'][out_mask], color=color_outlier,
                           s=35, zorder=5, label='异常值')
                for oi in np.where(out_mask)[0]:
                    outlier_eps.add(int(unique_eps[oi]))

            ax.set_title(titles[idx], fontsize=10, fontweight='bold')
            ax.set_xlabel('Episode 编号', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='best', fontsize=7)
            ax.tick_params(labelsize=8)

            # 汇总：全局最小/最大及其所在 episode
            gmin = float(np.min(s['min']))
            gmax = float(np.max(s['max']))
            gmin_ep = int(unique_eps[int(np.argmin(s['min']))])
            gmax_ep = int(unique_eps[int(np.argmax(s['max']))])
            out_list = [str(int(unique_eps[oi])) for oi in np.where(out_mask)[0]]
            out_str = ','.join(out_list)
            line = (f"• {titles[idx]}: 全局最小 {gmin:.4f} (EP#{gmin_ep}), "
                    f"全局最大 {gmax:.4f} (EP#{gmax_ep}), "
                    f"均值 {gmean:.4f}, 标准差 {gstd:.4f}")
            if out_str:
                line += f"  → 异常EP: {out_str}"
            summary_lines.append(line)

        self.figure.tight_layout()
        self.canvas.draw_idle()

        summary_lines.append("")
        if outlier_eps:
            summary_lines.append(f"异常 Episode 汇总: {sorted(outlier_eps)}")
            summary_lines.append("提示：以上 Episode 的某个指标均值显著偏离整体分布，建议重点核查。")
        else:
            summary_lines.append("未检测到异常 Episode（各指标均值均在 3σ 范围内）。")
        self.summary_text.setText('\n'.join(summary_lines))


# ==================== 数据集 7 类问题检查对话框 ====================
class DatasetCheckDialog(QDialog):
    """检查数据集是否存在 数据问题分析.md 中记录的 7 类问题，给出能否训练的结论。

    问题1: meta/episodes 的 data 文件映射失效
    问题2: 视频 to_timestamp 与实际视频时长不符
    问题3: data parquet 中 frame_index / timestamp 列错乱
    问题4: meta/stats.json 统计量不一致（count/维度）
    问题5: action 与 observation.state 完全相同（恒等映射）
    问题6: meta/episodes 与 data 数据行不一致（编辑中断残留）
    问题7: tasks.parquet 的 task_index 非 0 基连续（任务合并后未压实）
    """

    def __init__(self, parent, dataset):
        super().__init__(parent)
        self.dataset = dataset
        self.root = str(dataset.root)
        self.fps = dataset.fps
        self.setWindowTitle("数据集检查报告")
        self.resize(720, 600)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hint = QLabel("对照 数据问题分析.md 的 7 类问题逐项检查，判断数据集是否可用于 VLA 训练。")
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #555; font-size: 11px; padding: 2px; }")
        layout.addWidget(hint)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setStyleSheet(
            "QTextEdit { font-family: monospace; font-size: 12px; padding: 6px; }"
        )
        layout.addWidget(self.report_text, stretch=1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet("QPushButton { padding: 6px 28px; }")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self._run_checks()

    def _run_checks(self):
        import subprocess
        ds = self.dataset
        df = ds.data_df
        N = len(df)
        fps = self.fps
        lines = []
        lines.append(f"数据集: {self.root}")
        lines.append(f"总帧数: {N}  |  Episodes: {len(ds.episodes)}  |  FPS: {fps}")
        lines.append(f"state 维度: {ds.features.get('observation.state', {}).get('shape', '?')}")
        lines.append("=" * 64)

        states = np.vstack(df["observation.state"].to_numpy()) if "observation.state" in df.columns else np.zeros((0, 1))
        actions = np.vstack(df["action"].to_numpy()) if "action" in df.columns else np.zeros((0, 1))
        ep_col = df["episode_index"].to_numpy() if "episode_index" in df.columns else np.zeros(N, dtype=int)
        unique_eps = sorted(np.unique(ep_col))

        results = []

        # ---- 问题1: data 文件映射 ----
        lines.append("\n【问题1】meta/episodes 的 data 文件映射")
        ep_meta_files = sorted(glob.glob(str(ds.meta_dir / "episodes" / "**" / "*.parquet"), recursive=True))
        ok1 = True; bad1 = []
        if ep_meta_files:
            ep_df = pd.concat([pd.read_parquet(f) for f in ep_meta_files], ignore_index=True)
            for _, r in ep_df.iterrows():
                ci = int(r["data/chunk_index"]); fi = int(r["data/file_index"])
                p = ds.data_dir / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
                if not p.is_file():
                    ok1 = False; bad1.append(f"ep{int(r['episode_index'])}→file-{fi:03d}")
        if ok1:
            lines.append("  ✓ 全部 episode 的 data/chunk_index+file_index 指向的文件均存在")
        else:
            lines.append(f"  ✗ 失效映射: {', '.join(bad1[:5])}")
        results.append(ok1)

        # ---- 问题2: 视频帧数 / to_timestamp ----
        lines.append("\n【问题2】视频帧数 vs meta length + to_timestamp")
        ok2 = True; vid_details = []
        def _ffprobe_nb(path):
            try:
                r = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                     "-show_entries", "stream=nb_read_frames",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                    capture_output=True, text=True, timeout=120)
                return int(r.stdout.strip())
            except Exception:
                return None
        if ep_meta_files:
            for _, r in ep_df.iterrows():
                ep = int(r["episode_index"]); length = int(r["length"])
                for cam in ["cam1", "cam2"]:
                    col_ci = f"videos/observation.images.{cam}/chunk_index"
                    col_fi = f"videos/observation.images.{cam}/file_index"
                    col_ts = f"videos/observation.images.{cam}/to_timestamp"
                    if col_ci not in r.index:
                        continue
                    ci = int(r[col_ci]); fi = int(r[col_fi])
                    vp = ds.videos_dir / f"observation.images.{cam}" / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
                    if not vp.is_file():
                        ok2 = False; vid_details.append(f"  ✗ ep{ep} {cam}: {vp.name} 不存在")
                        continue
                    nb = _ffprobe_nb(vp)
                    to_ts = float(r[col_ts]) if col_ts in r.index else None
                    frame_ok = (nb == length) if nb is not None else True
                    ts_ok = (to_ts is not None) and (abs(to_ts - length / fps) < 0.05)
                    if not (frame_ok and ts_ok):
                        ok2 = False
                        vid_details.append(
                            f"  ✗ ep{ep} {cam}: frames={nb} vs length={length} "
                            f"{'✓' if frame_ok else '✗'} | to_ts={to_ts} vs {length/fps:.2f} {'✓' if ts_ok else '✗'}")
                    # 只打印前 3 个正常的 + 全部异常的
                    elif len([d for d in vid_details if '✗' in d]) == 0 and ep in unique_eps[:3] and cam == "cam1":
                        vid_details.append(f"  ✓ ep{ep} {cam}: frames={nb}={length}, to_ts={to_ts:.2f}")
        if ok2:
            lines.append("  ✓ 全部视频帧数与 meta length 一致，to_timestamp=length/fps")
        else:
            lines.append("  ✗ 存在不一致（详见下方）")
        lines.extend(vid_details[:8])
        if len(vid_details) > 8:
            lines.append(f"  ... 共 {len(vid_details)} 条详情")
        results.append(ok2)

        # ---- 问题3: frame_index / timestamp ----
        lines.append("\n【问题3】frame_index / timestamp 列规整")
        ok3 = True; bad3 = []
        for ep in unique_eps:
            sub = df[ep_col == ep]
            fi = sub["frame_index"].to_numpy() if "frame_index" in sub.columns else None
            ts = sub["timestamp"].to_numpy() if "timestamp" in sub.columns else None
            length = len(sub)
            if fi is not None and not np.array_equal(fi, np.arange(length)):
                ok3 = False; bad3.append(f"ep{ep}: frame_index 错乱 [{fi.min()}~{fi.max()}] 应[0~{length-1}]")
            if ts is not None and not np.allclose(ts, np.arange(length) / fps, atol=1e-6):
                ok3 = False; bad3.append(f"ep{ep}: timestamp 错乱")
        if ok3:
            lines.append(f"  ✓ 全部 {len(unique_eps)} 个 episode 的 frame_index/timestamp 规整")
        else:
            lines.append(f"  ✗ {len(bad3)} 个 episode 错乱:")
            for b in bad3[:5]:
                lines.append(f"    {b}")
        results.append(ok3)

        # ---- 问题4: stats 一致性 ----
        lines.append("\n【问题4】stats.json count 一致性")
        ok4 = True; bad4 = []
        # 注意：image/video 特征的 count 是“像素采样数”（LeRobot 对下采样后的像素做统计），
        # 不等于帧数 N；只有标量特征（state/action/timestamp/frame_index 等）的 count 才应 == N。
        # 因此对 image/video 特征只校验统计存在(>0)，不与 N 比较，避免误报。
        for key in ["observation.state", "action", "observation.images.cam1",
                     "observation.images.cam2", "timestamp", "frame_index"]:
            c = ds.stats.get(key, {}).get("count", [0])
            c = c[0] if isinstance(c, list) else c
            dtype = ds.features.get(key, {}).get("dtype", "")
            if dtype in ("image", "video"):
                if not c or int(c) <= 0:
                    ok4 = False; bad4.append(f"{key}: count={c}（图像统计缺失）")
            else:
                if int(c) != N:
                    ok4 = False; bad4.append(f"{key}: count={c} ≠ N={N}")
        dim_info = ds.features.get("observation.state", {}).get("shape", [0])[0]
        dim_actual = states.shape[1] if states.size else 0
        if int(dim_info) != dim_actual:
            ok4 = False; bad4.append(f"state维度: info={dim_info} ≠ 实际={dim_actual}")
        if ok4:
            lines.append(f"  ✓ 标量特征 count={N}；图像特征 count 为像素采样数（正常）；维度一致(dim={dim_actual})")
        else:
            for b in bad4:
                lines.append(f"  ✗ {b}")
        results.append(ok4)

        # ---- 问题5: 恒等映射 ----
        lines.append("\n【问题5】action == observation.state 恒等映射")
        diff = np.abs(actions - states).max(axis=1) if states.size else np.array([0])
        match = 0; tot = 0
        for ep in unique_eps:
            mask = ep_col == ep
            s = states[mask]; a = actions[mask]
            if len(s) > 1:
                d = np.abs(a[:-1] - s[1:]).max(axis=1)
                match += int((d < 1e-6).sum()); tot += len(d)
        identity = diff.max() < 1e-6 if len(diff) else True
        pct = 100 * match / max(tot, 1)
        lines.append(f"  max|a-s|={diff.max():.4f}, mean|a-s|={diff.mean():.6f}")
        lines.append(f"  action[t]==state[t+1]: {match}/{tot} ({pct:.1f}%)")
        if identity:
            lines.append("  ✗ 存在 action==state 恒等映射！模型会退化为忽略视觉")
        else:
            lines.append("  ✓ 无恒等映射问题")
        results.append(not identity)
        # 静止帧统计
        static_pct = 100 * (diff < 1e-6).mean()
        lines.append(f"  静止帧占比: {static_pct:.1f}% {'(偏高，建议裁剪)' if static_pct > 30 else ''}")

        # ---- 问题6: meta/episodes 与 data 行一致性（编辑中断残留）----
        lines.append("\n【问题6】meta/episodes 与 data 行一致性")
        ok6 = True
        bad6 = []
        if ep_meta_files and "episode_index" in df.columns:
            data_counts = pd.Series(ep_col).value_counts().to_dict()
            data_counts = {int(k): int(v) for k, v in data_counts.items()}
            if "episode_index" in ep_df.columns:
                meta_eps = [int(r["episode_index"]) for _, r in ep_df.iterrows()]
            else:
                meta_eps = list(range(len(ep_df)))
            meta_set = set(meta_eps)

            phantom = [e for e in meta_eps if e not in data_counts]
            if phantom:
                ok6 = False
                bad6.append(f"meta 有但 data 无数据行的 episode（幻影）: "
                            f"{phantom[:8]}{' ...' if len(phantom) > 8 else ''}")
            orphan = [e for e in unique_eps if e not in meta_set]
            if orphan:
                ok6 = False
                bad6.append(f"data 有但 meta 缺失的 episode_index（孤儿行）: "
                            f"{[int(e) for e in orphan[:8]]}{' ...' if len(orphan) > 8 else ''}")

            len_bad = []
            for _, r in ep_df.iterrows():
                e = int(r["episode_index"]) if "episode_index" in ep_df.columns else None
                if e in data_counts and int(r["length"]) != data_counts[e]:
                    len_bad.append(f"ep{e}: meta={int(r['length'])} vs data={data_counts[e]}")
            if len_bad:
                ok6 = False
                bad6.append(f"length 与 data 行数不符（{len(len_bad)} 个）: "
                            + "; ".join(len_bad[:5]) + (" ..." if len(len_bad) > 5 else ""))

            # dataset_from/to_index 连续性：from[0]==0、首尾相接、长度匹配、末尾==N
            if "dataset_from_index" in ep_df.columns and "dataset_to_index" in ep_df.columns:
                ptr = ep_df[["dataset_from_index", "dataset_to_index", "length"]].to_numpy(dtype=np.int64)
                ptr = ptr[np.argsort(ptr[:, 0])]
                contig = (int(ptr[0, 0]) == 0
                          and bool((ptr[1:, 0] == ptr[:-1, 1]).all())
                          and bool((ptr[:, 1] - ptr[:, 0] == ptr[:, 2]).all()))
                if not contig or int(ptr[-1, 1]) != N:
                    ok6 = False
                    bad6.append(f"dataset_from/to_index 不连续或与实际帧数不符"
                                f"（末尾 to={int(ptr[-1, 1])}，data 实际 {N} 行）")
        elif not ep_meta_files:
            lines.append("  （无 meta/episodes parquet，跳过指针校验）")

        info_tf = ds.info.get("total_frames")
        info_te = ds.info.get("total_episodes")
        if info_tf is not None and int(info_tf) != N:
            ok6 = False
            bad6.append(f"info.json total_frames={info_tf} ≠ data 实际 {N}")
        if ep_meta_files and info_te is not None and int(info_te) != len(ep_df):
            ok6 = False
            bad6.append(f"info.json total_episodes={info_te} ≠ meta 实际 {len(ep_df)}")

        if ok6:
            lines.append("  ✓ meta/data/info 三方一致（无编辑中断残留）")
        else:
            for b in bad6:
                lines.append(f"  ✗ {b}")
            lines.append("  → 该不一致会导致修剪/删除报错（_sync_episode_data_pointers 拦截），")
            lines.append("    继续编辑会加重错位，请先修复（见 数据问题分析.md 问题6）")
        results.append(ok6)

        # ---- 问题7: task_index 0基连续性（任务合并删除行后未压实）----
        lines.append("\n【问题7】task_index 0基连续性")
        ok7 = True; bad7 = []
        n_tasks = 0; tasks_idx_vals = []; task_names = []
        tasks_path = ds.meta_dir / "tasks.parquet"
        if tasks_path.is_file():
            tdf = pd.read_parquet(tasks_path)
            n_tasks = len(tdf)
            task_names = [str(i) for i in tdf.index.tolist()]
            if "task_index" in tdf.columns:
                tasks_idx_vals = sorted(int(v) for v in tdf["task_index"].tolist())
                expect = list(range(n_tasks))
                if tasks_idx_vals != expect:
                    ok7 = False
                    bad7.append(f"tasks.parquet 的 task_index={tasks_idx_vals} 非 0 基连续"
                                f"（应为 {expect}），lerobot 按位置 iloc 取任务名将越界/错位")
            else:
                ok7 = False
                bad7.append("tasks.parquet 缺少 task_index 列")
        else:
            ok7 = False
            bad7.append("meta/tasks.parquet 不存在")
        if "task_index" in df.columns and N > 0:
            ti = df["task_index"].to_numpy(dtype=np.int64)
            ti_min, ti_max = int(ti.min()), int(ti.max())
            if ti_min < 0 or ti_max >= max(n_tasks, 1):
                ok7 = False
                bad7.append(f"data 帧 task_index∈[{ti_min},{ti_max}] 超出 tasks 位置范围 "
                            f"[0,{n_tasks - 1}] → 训练取任务名时 IndexError")
            if tasks_idx_vals:
                missing = sorted(set(int(v) for v in np.unique(ti)) - set(tasks_idx_vals))
                if missing:
                    ok7 = False
                    bad7.append(f"data 引用了 tasks.parquet 中不存在的 task_index: {missing}")
        if ep_meta_files and "tasks" in ep_df.columns and task_names:
            name_set = set(task_names); unknown = set()
            for tv in ep_df["tasks"].to_numpy():
                try:
                    tl = list(tv) if tv is not None else []
                except TypeError:
                    tl = [tv]
                for t in tl:
                    if str(t) not in name_set:
                        unknown.add(str(t))
            if unknown:
                ok7 = False
                bad7.append(f"episodes 引用了 tasks.parquet 中不存在的任务名: "
                            f"{sorted(unknown)[:5]}")
        if ok7:
            lines.append(f"  ✓ {n_tasks} 个任务 task_index 为 0 基连续，data/episodes 引用一致")
        else:
            for b in bad7:
                lines.append(f"  ✗ {b}")
            lines.append("  → 修复：压实 tasks.parquet 的 task_index 为 0..n-1，同步改写 data 帧")
            lines.append("    task_index 与 stats.json 统计（见 数据问题分析.md 问题7）")
        results.append(ok7)

        # ---- 总结 ----
        all_ok = all(results)
        lines.append("\n" + "=" * 64)
        names = ["问题1 data文件映射", "问题2 视频帧数/to_ts", "问题3 frame_index/ts",
                 "问题4 stats一致性", "问题5 恒等映射", "问题6 meta/data一致性",
                 "问题7 task_index连续性"]
        for name, ok in zip(names, results):
            lines.append(f"  {'✓ 通过' if ok else '✗ 存在问题'}  {name}")
        lines.append("")
        if all_ok:
            lines.append("  ✅ 结论：全部 7 类问题通过，可用于 VLA 模型训练。")
        else:
            n_fail = sum(1 for ok in results if not ok)
            lines.append(f"  ❌ 结论：{n_fail} 项存在问题，需修复后再训练。")

        # 用 HTML 渲染，绿色✓ / 红色✗
        html = self._colorize("\n".join(lines))
        self.report_text.setHtml(html)

    @staticmethod
    def _colorize(text):
        """把报告文本转为简单 HTML，✓ 绿色、✅ 绿色、✗ 红色、❌ 红色。"""
        import html as _html
        out = _html.escape(text)
        out = out.replace("✓", '<span style="color:#2e7d32;font-weight:bold">✓</span>')
        out = out.replace("✗", '<span style="color:#c62828;font-weight:bold">✗</span>')
        out = out.replace("✅", '<span style="color:#2e7d32;font-weight:bold">✅</span>')
        out = out.replace("❌", '<span style="color:#c62828;font-weight:bold">❌</span>')
        out = out.replace("\n", "<br>")
        out = out.replace("  ", "&nbsp;&nbsp;")
        return f"<div style='white-space:pre-wrap'>{out}</div>"


# ==================== VLM 标签后台线程 ====================
class _VlmLabelThread(QThread):
    """后台加载 Qwen3-VL 并为各分段生成“动作+物体”标签。

    模型只加载一次（缓存在 self._model/self._processor），后续调用复用。
    """
    labels_ready = pyqtSignal(list, str)   # (labels, note)
    load_failed = pyqtSignal(str)

    def __init__(self, model_path: str, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self._model = None
        self._processor = None
        self._video_path = None
        self._cam2_path = None
        self._segments = []
        self._known_labels = []

    def set_work(self, video_path, segments, known_labels=None, cam2_path=None):
        self._video_path = video_path
        self._cam2_path = cam2_path
        self._segments = segments
        self._known_labels = known_labels or []

    def run(self):
        try:
            if self._model is None:
                from qwen3vl_subtask_annotate import load_model
                self._model, self._processor = load_model(self.model_path)
            from qwen3vl_subtask_annotate import generate_labels
            labels = generate_labels(
                self._model, self._processor,
                Path(self._video_path), self._segments,
                known_labels=self._known_labels,
                cam2_path=self._cam2_path,
            )
            self.labels_ready.emit(labels, f"VLM 已为 {len(labels)} 段生成标签，可手动微调后应用。")
        except Exception as e:
            self.load_failed.emit(str(e))


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self, root="./lerobot_data"):
        super().__init__()
        self.root = root
        self.dataset = None
        self.playback_thread = None
        self._current_total_frames = 0
        self._current_frame_idx = 0

        # 任务编辑状态
        self._current_episode_idx = -1
        self._original_task_name = ""
        self._task_modified = False
        self._task_edit_connected = False
        self._task_edit_in_progress = False  # 防止 editingFinished 重复触发导致对话框多次弹出

        # 批量剪辑静止状态
        self._trim_all_in_progress = False
        self._trim_all_queue = []
        self._trim_all_current_ep = -1
        self._trim_all_total = 0
        self._trim_all_trimmed = 0
        self._trim_all_skipped = 0
        self._trim_all_failed = 0
        self._trim_all_backup_enabled = False

        # 机械臂关节反解状态
        self._ik_thread = None
        self._ik_episode_idx = -1          # 当前正在反解 / 已反解完成的 episode
        self._ik_solving = False
        self._ik_retiring = set()          # 已取消、等待自行退出的旧反解线程

        # RECAP 奖励标注状态
        self._reward_store = None            # RewardAnnotationStore 实例
        self._recap_current_label = "good"   # 当前选中的标签
        self._recap_start_mark = None        # 区间起点（帧号）

        # 批量删除状态
        self.batch_delete_thread = None      # BatchDeleteEpisodesThread 实例
        self._pending_episode_indices = []   # 正在批量删除的 episode 编号

        # 首帧预览图 LRU 缓存：(episode_idx, camera_name) -> np.ndarray
        # 切换 episode 时免去重复的视频打开/解码（~30ms/路）；数据集变更时清空
        self._preview_frame_cache = OrderedDict()
        self._PREVIEW_CACHE_MAX = 16

        # 滑条拖动节流：用 QTimer 合并连续的 position_changed 事件，
        # 拖动时最多 ~33fps 解码，避免每个像素移动都触发视频解码
        self._scrub_target_frame = 0
        self._scrub_timer = QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(30)   # 30ms ≈ 33fps
        self._scrub_timer.timeout.connect(self._on_scrub_timeout)

        self.setWindowTitle("LeRobot 数据可视化工具")
        self.setGeometry(80, 80, 1700, 950)
        
        self._init_ui()
        self._scan_datasets()
        
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # ===== 左侧面板 =====
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 数据集选择
        dataset_group = QGroupBox("数据集选择")
        dataset_layout = QVBoxLayout(dataset_group)
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_selected)
        dataset_layout.addWidget(self.dataset_combo)
        
        refresh_btn = QPushButton("🔄 刷新列表")
        refresh_btn.clicked.connect(self._scan_datasets)
        dataset_layout.addWidget(refresh_btn)
        
        left_layout.addWidget(dataset_group)
        
        # 数据集总信息
        info_group = QGroupBox("数据集总信息")
        info_layout = QVBoxLayout(info_group)
        
        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setMaximumHeight(480)
        
        self.info_label = QLabel("未加载数据集")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }")
        info_scroll.setWidget(self.info_label)
        info_layout.addWidget(info_scroll)
        
        left_layout.addWidget(info_group)
        
        # Episode 列表
        episode_group = QGroupBox("Episode 列表")
        episode_layout = QVBoxLayout(episode_group)
        
        self.episode_table = QTableWidget()
        self.episode_table.setColumnCount(5)
        self.episode_table.setHorizontalHeaderLabels(["编号", "帧数", "时长(秒)", "分辨率", "任务"])
        self.episode_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.episode_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.episode_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.episode_table.itemSelectionChanged.connect(self._on_episode_selected)
        self.episode_table.setMinimumHeight(150)
        self.episode_table.verticalHeader().setDefaultSectionSize(28)
        episode_layout.addWidget(self.episode_table)
        
        self.episode_stats_label = QLabel("请选择数据集")
        self.episode_stats_label.setStyleSheet("QLabel { color: #666; font-size: 12px; }")
        episode_layout.addWidget(self.episode_stats_label)

        # 数据集统计 / 检查 按钮（水平并排）
        ds_btn_layout = QHBoxLayout()
        ds_btn_layout.setSpacing(6)

        # 数据集统计分析按钮
        self.stats_btn = QPushButton("📊 数据集统计分析")
        self.stats_btn.setToolTip("分析所有 Episode 的位置/欧拉角/夹爪角度分布，发现异常值与极值（横轴为 Episode 编号）")
        self.stats_btn.setStyleSheet(
            "QPushButton { background-color: #00838F; color: white; font-weight: bold; padding: 7px; }"
            "QPushButton:hover { background-color: #006064; }"
            "QPushButton:disabled { background-color: #ccc; color: #888; }"
        )
        self.stats_btn.clicked.connect(self._show_dataset_stats)
        self.stats_btn.setEnabled(False)
        ds_btn_layout.addWidget(self.stats_btn)

        # 数据集检查按钮（检查 7 类问题，判断能否训练）
        self.check_btn = QPushButton("🔍 数据集检查")
        self.check_btn.setToolTip("检查数据集是否存在 数据问题分析.md 的 7 类问题（data映射/视频/帧索引/stats一致性/恒等映射/meta与data行一致性/task_index连续性），给出能否训练结论")
        self.check_btn.setStyleSheet(
            "QPushButton { background-color: #558B2F; color: white; font-weight: bold; padding: 7px; }"
            "QPushButton:hover { background-color: #33691E; }"
            "QPushButton:disabled { background-color: #ccc; color: #888; }"
        )
        self.check_btn.clicked.connect(self._show_dataset_check)
        self.check_btn.setEnabled(False)
        ds_btn_layout.addWidget(self.check_btn)

        episode_layout.addLayout(ds_btn_layout)

        left_layout.addWidget(episode_group, stretch=4)

        # 任务名称编辑
        task_group = QGroupBox("Episode 任务")
        task_layout = QVBoxLayout(task_group)

        self.task_name_edit = QLineEdit()
        self.task_name_edit.setPlaceholderText("选择 Episode 后显示任务名称")
        self.task_name_edit.setEnabled(False)
        # 允许中文、英文、数字、下划线、空格、连字符等常见任务名称字符
        from PyQt6.QtGui import QRegularExpressionValidator
        from PyQt6.QtCore import QRegularExpression
        validator = QRegularExpressionValidator(
            QRegularExpression(r"^[\w\x{4e00}-\x{9fa5}][\w\x{4e00}-\x{9fa5}\s\-]*$"), self.task_name_edit
        )
        self.task_name_edit.setValidator(validator)
        task_row = QHBoxLayout()
        task_row.addWidget(self.task_name_edit, 1)
        self.task_batch_btn = QPushButton("批量修改")
        self.task_batch_btn.setToolTip("批量替换任务名称：将所有使用该任务名的 Episode 一次性改为新名称")
        self.task_batch_btn.clicked.connect(self._show_task_batch_dialog)
        task_row.addWidget(self.task_batch_btn)
        task_layout.addLayout(task_row)

        self.task_status_label = QLabel("")
        self.task_status_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        task_layout.addWidget(self.task_status_label)

        left_layout.addWidget(task_group)

        # 播放控制
        control_group = QGroupBox("播放控制")
        control_layout = QVBoxLayout(control_group)
        
        self.current_episode_label = QLabel("当前未选择 Episode")
        self.current_episode_label.setStyleSheet("QLabel { font-weight: bold; color: #2196F3; }")
        control_layout.addWidget(self.current_episode_label)
        
        # 可剪辑进度条（三标记：起止 + 播放头）
        self.trim_slider = TrimSlider()
        self.trim_slider.setEnabled(False)
        self.trim_slider.setToolTip(
            "拖动蓝色/紫色三角选择片段；双击滑条可将右侧位置曲线缩放到所选片段"
        )
        self.trim_slider.markers_changed.connect(self._on_trim_markers_changed)
        self.trim_slider.position_changed.connect(self._on_trim_position_changed)
        self.trim_slider.position_clicked.connect(self._on_trim_position_clicked)
        self.trim_slider.segment_selected.connect(self._on_trim_segment_selected)
        control_layout.addWidget(self.trim_slider)

        # RECAP 奖励标注轨道（显示 good/bad/neutral 彩色带）+ 显示开关
        recap_track_row = QHBoxLayout()
        self.recap_track = RecapTrackWidget()
        self.recap_track.setVisible(True)
        self.recap_show_check = QCheckBox("显示 RECAP 标记")
        self.recap_show_check.setChecked(True)
        self.recap_show_check.setToolTip("在播放时间线下方显示 good(绿)/bad(红)/neutral(橙) 彩色标注带")
        self.recap_show_check.toggled.connect(self._on_recap_track_toggled)
        recap_track_row.addWidget(self.recap_show_check)
        control_layout.addLayout(recap_track_row)
        control_layout.addWidget(self.recap_track)
        
        # 进度标签
        self.progress_label = QLabel("0 / 0")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("QLabel { font-size: 12px; color: #666; }")
        control_layout.addWidget(self.progress_label)

        # 剪辑区域信息
        self.trim_info_label = QLabel("剪辑区域: 未选择")
        self.trim_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trim_info_label.setStyleSheet("QLabel { font-size: 12px; color: #FF5722; font-weight: bold; }")
        self.trim_info_label.setToolTip("拖动下方蓝色/紫色三角标记选择要删除的片段")
        control_layout.addWidget(self.trim_info_label)

        # 通用进度条（用于删除/剪辑等耗时操作）
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setEnabled(False)
        control_layout.addWidget(self.progress_bar)
        
        btn_layout = QHBoxLayout()

        self.play_btn = QPushButton("▶ 播放")
        self.play_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )
        self.play_btn.clicked.connect(self._toggle_playback)
        btn_layout.addWidget(self.play_btn)

        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }"
        )
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.pause_btn.setEnabled(False)
        btn_layout.addWidget(self.pause_btn)

        # 删除按钮
        self.del_btn = QPushButton("🗑 删除片段")
        self.del_btn.setToolTip("删除选中的 Episode（永久删除，不可恢复）")
        self.del_btn.setStyleSheet("""
            QPushButton { 
                background-color: #9C27B0; 
                color: white; 
                font-weight: bold; 
                padding: 8px;
                border: 2px solid #7B1FA2;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #7B1FA2; }
            QPushButton:disabled { background-color: #ccc; color: #888; border: 2px solid #bbb; }
        """)
        self.del_btn.clicked.connect(self._delete_selected_episode)
        self.del_btn.setEnabled(False)
        btn_layout.addWidget(self.del_btn)

        # 批量删除按钮
        self.batch_del_btn = QPushButton("🗑 批量删除")
        self.batch_del_btn.setToolTip(
            "弹出对话框批量删除 Episode：可手动多选，也可按任务名筛选后一键勾选（永久删除，不可恢复）")
        self.batch_del_btn.setStyleSheet("""
            QPushButton {
                background-color: #6A1B9A;
                color: white;
                font-weight: bold;
                padding: 8px;
                border: 2px solid #4A148C;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #4A148C; }
            QPushButton:disabled { background-color: #ccc; color: #888; border: 2px solid #bbb; }
        """)
        self.batch_del_btn.clicked.connect(self._batch_delete_episodes)
        self.batch_del_btn.setEnabled(False)
        btn_layout.addWidget(self.batch_del_btn)

        # 剪辑按钮
        self.trim_btn = QPushButton("✂ 剪辑")
        self.trim_btn.setToolTip("删除两个标记之间的片段")
        self.trim_btn.setStyleSheet("""
            QPushButton { 
                background-color: #FF5722; 
                color: white; 
                font-weight: bold; 
                padding: 8px;
                border: 2px solid #E64A19;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #E64A19; }
            QPushButton:disabled { background-color: #ccc; color: #888; border: 2px solid #bbb; }
        """)
        self.trim_btn.clicked.connect(self._trim_selected_segment)
        self.trim_btn.setEnabled(False)
        btn_layout.addWidget(self.trim_btn)
        
        control_layout.addLayout(btn_layout)
        
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("播放速度:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 5.0)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setDecimals(1)
        speed_layout.addWidget(self.speed_spin)
        speed_layout.addWidget(QLabel("x"))
        speed_layout.addStretch()
        control_layout.addLayout(speed_layout)
        
        self.auto_play_check = QCheckBox("自动播放所有 Episode")
        control_layout.addWidget(self.auto_play_check)

        self.backup_before_trim_check = QCheckBox("剪辑前自动备份原数据")
        self.backup_before_trim_check.setChecked(False)
        self.backup_before_trim_check.setToolTip(
            "勾选后会在剪辑前生成 data.bak_*/meta.bak_*/videos.bak_* 备份目录；"
            "取消勾选则不再生成备份，请确保已自行备份重要数据。"
        )
        control_layout.addWidget(self.backup_before_trim_check)
        
        left_layout.addWidget(control_group)
        left_layout.addStretch()
        
        splitter.addWidget(left_panel)
        
        # ===== 中间面板: 图像回放 + 机械臂 3D 可视化 =====
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.setContentsMargins(0, 0, 0, 0)

        # 图像回放（上方）
        image_group = QGroupBox("图像回放")
        image_layout = QVBoxLayout(image_group)
        
        self.cam1_label = ImageLabel("摄像头 1 (Cam1)")
        self.cam1_label.setStyleSheet("background-color: #222; color: #fff; font-size: 12px; border: 1px solid #444;")
        self.cam1_label.setMinimumSize(320, 180)
        self.cam1_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        image_layout.addWidget(self.cam1_label)

        self.cam2_label = ImageLabel("摄像头 2 (Cam2)")
        self.cam2_label.setStyleSheet("background-color: #222; color: #fff; font-size: 12px; border: 1px solid #444;")
        self.cam2_label.setMinimumSize(320, 180)
        self.cam2_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        image_layout.addWidget(self.cam2_label)

        # 双击图像弹出全分辨率查看窗口
        self.cam1_label.doubleClicked.connect(
            lambda: self._on_image_double_clicked("observation.images.cam1")
        )
        self.cam2_label.doubleClicked.connect(
            lambda: self._on_image_double_clicked("observation.images.cam2")
        )
        
        self.frame_info_label = QLabel("帧信息: 未开始播放")
        self.frame_info_label.setStyleSheet("QLabel { color: #666; font-size: 11px; padding: 5px; }")
        self.frame_info_label.setWordWrap(True)
        self.frame_info_label.setMinimumHeight(80)
        image_layout.addWidget(self.frame_info_label)
        middle_layout.addWidget(image_group, stretch=4)

        # 机械臂 3D 可视化（下方：根据末端位姿反解关节角度并实时显示）
        arm_group = QGroupBox("机械臂可视化 (末端位姿反解关节角度)")
        arm_layout = QVBoxLayout(arm_group)
        arm_layout.setContentsMargins(2, 2, 2, 2)
        self.robot_arm_widget = RobotArmReplayWidget()
        arm_layout.addWidget(self.robot_arm_widget)
        middle_layout.addWidget(arm_group, stretch=5)

        splitter.addWidget(middle_panel)

        # ===== 右侧面板: 位置曲线 =====
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Episode 末端位姿变化曲线（移至右侧）
        trajectory_group = QGroupBox("位置曲线")
        trajectory_layout = QVBoxLayout(trajectory_group)
        self.trajectory_plot = TrajectoryPlotWidget()
        self.trajectory_plot.trim_stationary_requested.connect(self._trim_stationary_segments)
        self.trajectory_plot.trim_all_stationary_requested.connect(self._trim_all_stationary_segments)
        self.trajectory_plot.denoise_requested.connect(self._denoise_current_episode)
        self.trajectory_plot.denoise_all_requested.connect(self._denoise_all_episodes)
        self.trajectory_plot.preview_denoise_requested.connect(self._preview_denoise)
        self.trajectory_plot.subtask_apply_requested.connect(self._apply_subtask_annotations)
        self.trajectory_plot.vlm_label_requested.connect(self._on_vlm_label_requested)
        # 数据集合并分页：设置数据根目录，合并完成后刷新数据集列表
        self.trajectory_plot.set_merge_root(self.root)
        self.trajectory_plot.merge_completed.connect(self._on_merge_completed)
        # RECAP 标注：信号连接（控件在 trajectory_plot 上，逻辑在 MainWindow）
        tp = self.trajectory_plot
        tp.recap_label_combo.currentTextChanged.connect(self._on_recap_label_changed)
        tp.recap_mark_start_btn.clicked.connect(self._recap_mark_start)
        tp.recap_mark_end_btn.clicked.connect(self._recap_mark_end)
        tp.recap_add_btn.clicked.connect(self._recap_add_interval)
        tp.recap_quick_good.clicked.connect(lambda: self._recap_quick_mark("good"))
        tp.recap_quick_bad.clicked.connect(lambda: self._recap_quick_mark("bad"))
        tp.recap_quick_neutral.clicked.connect(lambda: self._recap_quick_mark("neutral"))
        tp.recap_ep_good.clicked.connect(lambda: self._recap_mark_whole_episode("good"))
        tp.recap_ep_bad.clicked.connect(lambda: self._recap_mark_whole_episode("bad"))
        tp.recap_del_btn.clicked.connect(self._recap_delete_selected)
        tp.recap_undo_btn.clicked.connect(self._recap_undo_last)
        tp.recap_clear_btn.clicked.connect(self._recap_clear_episode)
        tp.recap_save_btn.clicked.connect(self._recap_save)
        # VLM 标签后台线程（模型只加载一次，之后复用）
        _vlm_model = str(_DBG_DIR.parent / "models" / "Qwen" / "Qwen3-VL-4B-Instruct")
        self._vlm_thread = _VlmLabelThread(_vlm_model, self)
        self._vlm_thread.labels_ready.connect(self._on_vlm_labels_ready)
        self._vlm_thread.load_failed.connect(self._on_vlm_load_failed)
        trajectory_layout.addWidget(self.trajectory_plot)
        right_layout.addWidget(trajectory_group, stretch=6)

        # 操作说明按钮（点击弹出对话框，避免占用右侧空间）
        self.help_btn = QPushButton("📖 操作说明")
        self.help_btn.setStyleSheet(
            "QPushButton { font-size: 12px; padding: 6px; }"
        )
        self.help_btn.clicked.connect(self._show_help_dialog)
        right_layout.addWidget(self.help_btn)
        
        right_layout.addStretch()
        
        splitter.addWidget(right_panel)
        splitter.setSizes([460, 720, 520])
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请选择数据集")
        
    def _scan_datasets(self):
        self.dataset_combo.clear()
        self.dataset_combo.addItem("-- 请选择数据集 --", None)
        
        if not os.path.exists(self.root):
            self.status_bar.showMessage(f"数据目录不存在: {self.root}")
            return
            
        found = False
        for name in sorted(os.listdir(self.root)):
            path = os.path.join(self.root, name)
            if os.path.isdir(path):
                has_meta = os.path.exists(os.path.join(path, "meta", "info.json"))
                if has_meta:
                    self.dataset_combo.addItem(name, path)
                    found = True
                    
        if not found:
            self.status_bar.showMessage("未找到任何数据集")
        else:
            count = self.dataset_combo.count() - 1
            self.status_bar.showMessage(f"找到 {count} 个数据集")

    def _on_merge_completed(self, target_name):
        """数据集合并完成：刷新数据集列表，并加载合并目标数据集。

        若目标正是当前已加载的数据集（覆盖合并），下拉框索引不变不会触发
        currentIndexChanged，需要手动调用 _on_dataset_selected 强制重新加载。
        """
        current_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        self._scan_datasets()
        for i in range(1, self.dataset_combo.count()):
            if self.dataset_combo.itemText(i) == target_name:
                same_as_current = self.dataset_combo.itemData(i) == current_path
                self.dataset_combo.setCurrentIndex(i)
                if same_as_current:
                    self._on_dataset_selected(i)
                break
        self.status_bar.showMessage(f"数据集合并完成: {target_name}")

    def _show_help_dialog(self):
        """弹出操作说明对话框。"""
        dialog = QDialog(self)
        dialog.setWindowTitle("操作说明")
        dialog.resize(500, 600)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)

        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)

        help_text = QLabel(
            "<b>基本操作:</b><br>"
            "1. 从下拉菜单选择数据集<br>"
            "2. 在 Episode 列表中单击选择<br>"
            "3. 点击 <b>▶ 播放</b> 开始回放<br><br>"
            "<b>播放控制:</b><br>"
            "• <b>▶ 播放</b> - 播放选中的 Episode<br>"
            "• <b>⏸ 暂停</b> - 暂停/继续当前播放<br>"
            "• <b>🔄 重新播放</b> - 从头重新播放<br>"
            "• <b>⏹ 停止</b> - 停止播放<br><br>"
            "<b>图像检查:</b><br>"
            "• <b>双击</b>右侧任意摄像头图像，可弹出该摄像头当前帧的<b>全分辨率图像</b><br><br>"
            "<b>位置曲线:</b><br>"
            "• 选择 Episode 后，右侧会显示该 Episode 末端位置 XYZ 随帧序号变化的曲线<br>"
            "• 播放或拖动进度条时，黑色竖线会同步指示当前帧位置<br>"
            "• <b>双击</b>进度条（剪辑滑条），可将曲线缩放到两个三角标记之间的片段<br>"
            "• 使用曲线下方的 X缩放 / Y缩放 滑条进一步放大检查细节，点击 重置 恢复全段视图<br><br>"
            "<b>其他功能:</b><br>"
            "• 双击 Episode 行可直接播放<br>"
            "• 勾选'自动播放所有'可连续播放<br>"
            "• 调整速度滑块改变回放速度<br><br>"
            "<b>视频剪辑:</b><br>"
            "• 拖动进度条下方蓝色/紫色三角选择要删除的片段<br>"
            "• 绿色播放头可拖动，实时显示对应帧画面与数据<br>"
            "• 点击轨道空白处可跳转当前帧<br>"
            "• 播放中拖动/点击播放头会自动暂停并定位到该帧<br>"
            "• 点击 ✂ 剪辑按钮删除两个三角标记之间的片段<br><br>"
            "<b>智能降噪:</b><br>"
            "• 在右侧'智能降噪'面板选择滤波方法（推荐 Savitzky-Golay）与平滑窗口<br>"
            "• 点击 🧪 预览降噪 可在曲线上叠加虚线对比效果（不修改数据）<br>"
            "• 点击 ✨ 智能降噪 平滑当前 Episode，✨ 批量智能降噪 处理全部 Episode<br>"
            "• 只修改 state/action 数值，帧数与元数据不变，处理后自动重算 stats.json"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("QLabel { line-height: 1.5; }")
        help_text.setTextFormat(Qt.TextFormat.RichText)
        help_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        help_text.setOpenExternalLinks(True)
        content_layout.addWidget(help_text)
        content_layout.addStretch()

        scroll.setWidget(content)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dialog.exec()
            
    def _show_dataset_stats(self):
        """弹出数据集统计分布分析对话框，分析所有 Episode 的位姿分布。"""
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return
        if len(self.dataset.episodes) == 0:
            QMessageBox.warning(self, "警告", "当前数据集没有 Episode")
            return
        try:
            self.status_bar.showMessage("正在分析数据集分布...")
            QApplication.processEvents()
            dialog = DatasetStatsDialog(self, self.dataset)
            self.status_bar.showMessage("统计分析完成")
            dialog.exec()
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "错误", f"统计分析失败:\n{str(e)}\n\n{traceback.format_exc()}")
            self.status_bar.showMessage("统计分析失败")

    def _show_dataset_check(self):
        """弹出数据集 7 类问题检查报告，判断能否用于 VLA 训练。"""
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return
        if len(self.dataset.episodes) == 0:
            QMessageBox.warning(self, "警告", "当前数据集没有 Episode")
            return
        # 防误触：按钮变"检查中"并禁用，检查完成/异常后恢复
        self.check_btn.setEnabled(False)
        self.check_btn.setText("⏳ 检查中...")
        QApplication.processEvents()
        try:
            self.status_bar.showMessage("正在检查数据集...")
            QApplication.processEvents()
            dialog = DatasetCheckDialog(self, self.dataset)
            self.status_bar.showMessage("数据集检查完成")
            dialog.exec()
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "错误", f"数据集检查失败:\n{str(e)}\n\n{traceback.format_exc()}")
            self.status_bar.showMessage("数据集检查失败")
        finally:
            self.check_btn.setText("🔍 数据集检查")
            self.check_btn.setEnabled(len(self.dataset.episodes) > 0)

    def _on_dataset_selected(self, index):
        if index <= 0:
            self.episode_table.setRowCount(0)
            self.info_label.setText("未加载数据集")
            self.episode_stats_label.setText("请选择数据集")
            self.current_episode_label.setText("当前未选择 Episode")
            self.del_btn.setEnabled(False)
            self.batch_del_btn.setEnabled(False)
            self.trim_btn.setEnabled(False)
            self.stats_btn.setEnabled(False)
            self.check_btn.setEnabled(False)
            self.trim_slider.set_total_frames(0)
            self.dataset = None
            self._preview_frame_cache.clear()

            # 清空任务编辑和位置曲线
            self._reset_task_editor()
            self.trajectory_plot.clear()
            return
            
        dataset_path = self.dataset_combo.itemData(index)
        if dataset_path is None:
            return
            
        try:
            self.status_bar.showMessage(f"正在加载数据集...")
            QApplication.processEvents()
            
            self.dataset = OfflineDataset(dataset_path)
            self._preview_frame_cache.clear()  # 新数据集：旧缓存全部失效

            # 初始化 RECAP 奖励标注存储（读取/创建 reward_annotations.json）
            try:
                self._reward_store = RewardAnnotationStore(str(dataset_path))
                self._reward_store.load()
            except Exception:
                self._reward_store = None
            self._update_recap_status()

            self._update_episode_table()
            self._update_info()
            self.stats_btn.setEnabled(len(self.dataset.episodes) > 0)
            self.check_btn.setEnabled(len(self.dataset.episodes) > 0)
            self.batch_del_btn.setEnabled(len(self.dataset.episodes) > 0)
            
            self.status_bar.showMessage(
                f"数据集加载完成: {len(self.dataset.episodes)} 个 episode, "
                f"{self.dataset.num_frames} 帧"
            )
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "错误", f"加载数据集失败:\n{str(e)}\n\n{traceback.format_exc()}")
            self.status_bar.showMessage("数据集加载失败")
            
    def _update_episode_table(self):
        if self.dataset is None:
            return
            
        episodes = self.dataset.episodes
        self.episode_table.setRowCount(len(episodes))
        
        # 获取所有图像分辨率，按 camera 名称排序
        resolutions = self.dataset.get_all_image_resolutions()
        resolution_text = ""
        if resolutions:
            resolution_text = ", ".join(
                f"{w}x{h}" for _, (w, h) in sorted(resolutions.items())
            )
        else:
            # 如果 info.json 中没有 shape，尝试读取第一个 episode 的第一帧图像或视频帧
            try:
                camera_names = [name for name, f in self.dataset.features.items()
                                if f.get("dtype") in ["image", "video"] or f.get("type") in ["image", "video"] or "image" in name.lower()]
                for cam in camera_names:
                    if self.dataset.is_video_feature(cam):
                        frame = self.dataset.get_video_frame(
                            self.dataset.episodes[0].get("episode_index", 0), 0, cam
                        )
                        if frame is not None and isinstance(frame, np.ndarray):
                            h, w = frame.shape[:2]
                            resolution_text = f"{w}x{h}"
                            break
                    else:
                        img_path = self.dataset.get_image_path(
                            self.dataset.episodes[0].get("episode_index", 0), 0, cam
                        )
                        if img_path:
                            img = QImage(img_path)
                            if not img.isNull():
                                resolution_text = f"{img.width()}x{img.height()}"
                                break
            except Exception:
                resolution_text = ""
        
        for i, ep in enumerate(episodes):
            length = ep.get("length", ep.get("dataset_to_index", 0) - ep.get("dataset_from_index", 0))
            duration = length / self.dataset.fps if self.dataset.fps > 0 else 0
            
            self.episode_table.setItem(i, 0, QTableWidgetItem(str(ep.get("episode_index", i))))
            self.episode_table.setItem(i, 1, QTableWidgetItem(str(length)))
            self.episode_table.setItem(i, 2, QTableWidgetItem(f"{duration:.1f}"))
            self.episode_table.setItem(i, 3, QTableWidgetItem(resolution_text))
            
            # LeRobot 0.4.4 episodes 中存储的是 tasks 字段（任务名称数组）
            task_name = ""
            if "tasks" in ep and ep["tasks"] is not None:
                # tasks 可能是 numpy array 或 list
                tasks = ep["tasks"]
                if hasattr(tasks, '__iter__') and not isinstance(tasks, str):
                    task_list = list(tasks)
                    if task_list:
                        task_name = str(task_list[0])
            elif "task_index" in ep:
                # 旧格式：从 task_index 查找（按值查找，不能用列表位置索引）
                task_name = str(self.dataset.task_name_for_index(ep["task_index"]))
            
            self.episode_table.setItem(i, 4, QTableWidgetItem(task_name))
            
        self.episode_stats_label.setText(f"共 {len(episodes)} 个 episode")
        
    def _update_info(self):
        if self.dataset is None:
            return
            
        info = self.dataset.info
        features = info.get("features", {})
        
        # 统计任务分布
        task_stats = self._compute_task_stats()
        
        info_text = (
            f"<b>数据集:</b> {info.get('repo_id', 'N/A')}<br>"
            f"<b>版本:</b> {info.get('codebase_version', 'N/A')}<br>"
            f"<b>总帧数:</b> {info.get('total_frames', 'N/A')}<br>"
            f"<b>Episode 数:</b> {info.get('total_episodes', 'N/A')}<br>"
            f"<b>FPS:</b> {info.get('fps', 'N/A')}<br>"
            f"<b>特征:</b> {', '.join(features.keys())}"
        )

        # action 标签前瞻（lookahead）估计：扫描 k∈[0,20] 使 mean|action[t]-state[t+k]|
        # 最小的 k 即为观测首目标距离。k=0 意味着 action==state 退化（数据问题分析.md 问题5）
        la = self._estimate_action_lookahead()
        if la is not None:
            best_k, best_s, s0, s1 = la
            if best_k == 0:
                info_text += (f"<br><b>action 前瞻:</b> <span style='color:#c62828'>"
                              f"0 帧（action==state 退化，需修复后训练！）</span>")
            else:
                fps = info.get('fps', 20) or 20
                info_text += (f"<br><b>action 前瞻:</b> {best_k} 帧"
                              f"（首目标领先 {best_k / fps * 1000:.0f}ms；"
                              f"残差 {best_s:.4f}，k=0 时 {s0:.4f}）")
        else:
            info_text += "<br><b>action 前瞻:</b> 无法估计（缺 state/action 列）"

        # 添加任务统计
        if task_stats:
            info_text += "<br><b>任务分布:</b><br>"
            for task_name, frame_count, frame_ratio, ep_count, ep_ratio in task_stats:
                info_text += f"  • {task_name}: {frame_count} 帧 ({frame_ratio:.1%}), {ep_count} 个 episode ({ep_ratio:.1%})<br>"
        
        self.info_label.setText(info_text)
    
    def _estimate_action_lookahead(self, max_k=20, max_samples=3000):
        """估计 action 标签的前瞻帧数（观测首目标距离）。

        对每个候选 k∈[0,max_k]，在 episode 内计算 mean|action[t]−state[t+k]|，
        取最小值对应的 k。录制约定 action[t]=pose[t+N]，故残差最小的 k≈N；
        末帧沿用末位姿会带来少量噪声，采样均值下不影响 argmin。

        返回 (best_k, 残差_min, k=0 残差, k=1 残差)，数据不足返回 None。
        """
        df = getattr(self.dataset, "data_df", None)
        if df is None or "action" not in df.columns or "observation.state" not in df.columns:
            return None
        scores = np.zeros(max_k + 1)
        counts = 0
        for _, g in df.groupby("episode_index"):
            if len(g) < max_k + 10:
                continue
            s = np.stack(g["observation.state"].to_numpy())
            a = np.stack(g["action"].to_numpy())
            idx = np.arange(0, len(g) - max_k, 5)  # 隔 5 帧采样，控制计算量
            if len(idx) == 0:
                continue
            for k in range(max_k + 1):
                scores[k] += np.abs(a[idx] - s[idx + k]).sum()
            counts += len(idx)
            if counts >= max_samples:
                break
        if counts == 0:
            return None
        scores /= counts
        best_k = int(np.argmin(scores))
        return best_k, float(scores[best_k]), float(scores[0]), float(scores[1])

    def _compute_task_stats(self):
        """计算各任务的帧数、比例，以及 Episode 个数和占比"""
        if self.dataset is None or not self.dataset.tasks:
            return []
        
        # 从 data_df 的 task_index 列统计帧数
        if 'task_index' not in self.dataset.data_df.columns:
            return []
        
        task_frame_counts = self.dataset.data_df['task_index'].value_counts().sort_index()
        total_frames = len(self.dataset.data_df)
        
        # 统计每个任务的 Episode 个数
        task_episode_counts = {}
        total_episodes = len(self.dataset.episodes)
        for ep in self.dataset.episodes:
            ep_tasks = ep.get('tasks', [])
            if ep_tasks is not None and hasattr(ep_tasks, '__iter__') and not isinstance(ep_tasks, str):
                task_list = list(ep_tasks)
                if task_list:
                    task_name = str(task_list[0])
                    task_episode_counts[task_name] = task_episode_counts.get(task_name, 0) + 1
        
        stats = []
        for task_idx, frame_count in task_frame_counts.items():
            task_name = self.dataset.task_name_for_index(task_idx, f'task_{task_idx}')
            
            frame_ratio = frame_count / total_frames if total_frames > 0 else 0
            ep_count = task_episode_counts.get(task_name, 0)
            ep_ratio = ep_count / total_episodes if total_episodes > 0 else 0
            
            stats.append((task_name, frame_count, frame_ratio, ep_count, ep_ratio))
        
        return stats
        
    def _on_trim_markers_changed(self, start, end):
        """当双标记位置改变时更新剪辑信息。"""
        self._update_trim_info()

    def _on_trim_position_changed(self, frame):
        """播放头拖动时：先暂停播放，滑条位置/进度标签即时更新，
        视频帧解码通过 QTimer 节流（合并连续 mouseMove，避免逐像素解码卡顿）。"""
        if self.playback_thread is not None:
            if not self.playback_thread.is_paused():
                self.playback_thread.pause()
                self.pause_btn.setText("▶ 继续")
                import time
                for _ in range(20):
                    if self.playback_thread.is_paused():
                        break
                    time.sleep(0.01)
                    QApplication.processEvents()
        # 记录目标帧，滑条与标签即时更新（开销极小）
        self._scrub_target_frame = frame
        self.trim_slider.set_current_frame(frame)
        if self._current_total_frames > 0:
            self.progress_label.setText(f"{frame + 1} / {self._current_total_frames}")
        # 启动/重启节流定时器：30ms 内若有新事件则覆盖，到期才真正解码
        self._scrub_timer.start()

    def _on_scrub_timeout(self):
        """节流定时器到期：加载目标帧画面（最多 ~33fps）。"""
        self._preview_frame_at(self._scrub_target_frame)

    def _on_trim_position_clicked(self, frame):
        """点击轨道或释放播放头时跳转到指定帧；播放中拖动后会保持暂停。"""
        if self.playback_thread is not None:
            if not self.playback_thread.is_paused():
                self.playback_thread.pause()
                self.pause_btn.setText("▶ 继续")
                # 等待线程真正进入暂停
                import time
                for _ in range(30):
                    if self.playback_thread.is_paused():
                        break
                    time.sleep(0.01)
                    QApplication.processEvents()
            self.playback_thread.jump_to_frame(frame, resume_after=False)
        # 同步预览到释放位置
        self._preview_frame_at(frame)
        self.trim_slider.set_current_frame(frame)
        if self._current_total_frames > 0:
            self.progress_label.setText(f"{frame + 1} / {self._current_total_frames}")

    def _on_trim_segment_selected(self, start, end):
        """双击剪辑滑条时，将右侧位置曲线缩放到所选片段范围。"""
        if self.dataset is None or self._current_episode_idx < 0:
            return
        if start == end:
            return
        logger.info(f"[Trajectory] 双击选中片段 {start}~{end}，曲线缩放至该范围")
        self.trajectory_plot.set_segment_range(start, end)
        self.status_bar.showMessage(f"位置曲线已缩放到片段: {start} ~ {end}")

    def _update_trim_info(self):
        """更新剪辑信息标签显示。"""
        total = self._current_total_frames
        if total <= 0:
            self.trim_info_label.setText("剪辑区域: 未选择")
            return
        start, end = self.trim_slider.get_range()
        if start == end:
            self.trim_info_label.setText("剪辑区域: 未选择")
        else:
            fps = self.dataset.fps if self.dataset else 20
            start_t = start / fps
            end_t = end / fps
            self.trim_info_label.setText(
                f"剪辑区域: {start}~{end} 帧 ({start_t:.2f}s ~ {end_t:.2f}s) 共 {end - start + 1} 帧"
            )

    def _preview_frame_at(self, frame_idx):
        """在预览区显示指定帧。"""
        if self.dataset is None or self._current_episode_idx < 0:
            return
        try:
            ep_frames = self.dataset.get_episode_frames(self._current_episode_idx)
            total = len(ep_frames)
            if total == 0 or frame_idx < 0 or frame_idx >= total:
                return
            row = ep_frames.iloc[frame_idx]
            cam1 = self._load_image_for_preview(self._current_episode_idx, frame_idx, "observation.images.cam1")
            cam2 = self._load_image_for_preview(self._current_episode_idx, frame_idx, "observation.images.cam2")
            action = np.array(row.get("action", [0] * 7))
            state = np.array(row.get("observation.state", [0] * 7))
            timestamp = row.get("timestamp", frame_idx / self.dataset.fps if self.dataset.fps else 0)
            task_index = int(row.get("task_index", 0))
            task_name = self.dataset.task_name_for_index(task_index)
            self._display_frame(cam1, cam2, action, state, frame_idx, timestamp, task_name)
            self._current_frame_idx = frame_idx
            self.progress_label.setText(f"{frame_idx + 1} / {total}")
            # 同步位置曲线竖线
            self.trajectory_plot.set_current_frame(frame_idx)
            # 同步机械臂 3D 视图到该帧
            self.robot_arm_widget.show_frame(frame_idx)
        except Exception as e:
            print(f"[Preview] 跳转帧失败: {e}")

    def _preview_denoise(self, params):
        """在当前位置曲线上叠加平滑后的虚线预览（不修改数据）。"""
        if self.dataset is None or self._current_episode_idx < 0:
            return
        try:
            frames_data = self._load_episode_position_data(self._current_episode_idx)
            if frames_data is None:
                return
            keys = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
            mat = np.stack([np.asarray(frames_data[k], dtype=np.float64) for k in keys], axis=1)
            smoothed = _smooth_pose_matrix(
                mat, params.get("method", "savgol"), params.get("window", 11),
                params.get("polyorder", 2), params.get("sigma", 2.0),
            )
            overlay = {k: smoothed[:, i] for i, k in enumerate(keys)}
            self.trajectory_plot.set_smoothed_overlay(overlay)
            self.status_bar.showMessage("已叠加降噪预览（虚线），未修改数据")
        except Exception as e:
            QMessageBox.warning(self, "预览失败", f"降噪预览失败: {e}")

    # ==================== RECAP 奖励标注逻辑 ====================

    def _on_recap_label_changed(self, label):
        self._recap_current_label = label

    def _recap_mark_start(self):
        if self._reward_store is None:
            return
        tp = self.trajectory_plot
        self._recap_start_mark = self._current_frame_idx
        tp.recap_start_spin.setValue(self._recap_start_mark)
        self.status_bar.showMessage(f"RECAP 区间起点设为 {self._recap_start_mark}")

    def _recap_mark_end(self):
        self.trajectory_plot.recap_end_spin.setValue(self._current_frame_idx)
        self.status_bar.showMessage(f"RECAP 区间终点设为 {self._current_frame_idx}")

    def _recap_add_interval(self):
        if self._reward_store is None or self._current_episode_idx < 0:
            QMessageBox.information(self, "RECAP", "请先选择数据集与 episode。")
            return
        tp = self.trajectory_plot
        start = tp.recap_start_spin.value()
        end = tp.recap_end_spin.value()
        if end < start:
            start, end = end, start
        label = tp.recap_label_combo.currentText()
        note = tp.recap_note_edit.text().strip()
        self._reward_store.add_interval(self._current_episode_idx, start, end, label, note)
        tp.recap_note_edit.clear()
        self._recap_start_mark = None
        self._refresh_recap_table()
        self._update_recap_status()
        self.status_bar.showMessage(f"RECAP 已添加区间 [{start},{end}] = {label}")

    def _recap_quick_mark(self, label):
        """快捷标记：用 start_mark 到当前帧，或单帧。"""
        if self._reward_store is None or self._current_episode_idx < 0:
            QMessageBox.information(self, "RECAP", "请先选择数据集与 episode。")
            return
        cur = self._current_frame_idx
        if self._recap_start_mark is not None and self._recap_start_mark != cur:
            start, end = min(self._recap_start_mark, cur), max(self._recap_start_mark, cur)
        else:
            start, end = cur, cur
        self._reward_store.add_interval(self._current_episode_idx, start, end, label, "")
        self._recap_start_mark = None
        self._refresh_recap_table()
        self._update_recap_status()
        self.status_bar.showMessage(f"RECAP 快标 [{start},{end}] = {label}")

    def _recap_mark_whole_episode(self, label):
        if self._reward_store is None or self._current_total_frames <= 0:
            return
        self._reward_store.clear_episode(self._current_episode_idx)
        self._reward_store.add_interval(
            self._current_episode_idx, 0, self._current_total_frames - 1, label, "whole_episode")
        self._refresh_recap_table()
        self._update_recap_status()

    def _recap_delete_selected(self):
        if self._reward_store is None:
            return
        tp = self.trajectory_plot
        rows = sorted(
            {idx.row() for idx in tp.recap_interval_table.selectedIndexes()},
            reverse=True)
        for r in rows:
            self._reward_store.remove_interval(self._current_episode_idx, r)
        self._refresh_recap_table()
        self._update_recap_status()

    def _recap_undo_last(self):
        if self._reward_store is None:
            return
        intervals = self._reward_store.get_intervals(self._current_episode_idx)
        if intervals:
            self._reward_store.remove_interval(self._current_episode_idx, len(intervals) - 1)
            self._refresh_recap_table()
            self._update_recap_status()

    def _recap_clear_episode(self):
        if self._reward_store is None:
            return
        ret = QMessageBox.question(
            self, "确认", f"清空 episode {self._current_episode_idx} 的所有 RECAP 标注？")
        if ret == QMessageBox.StandardButton.Yes:
            self._reward_store.clear_episode(self._current_episode_idx)
            self._refresh_recap_table()
            self._update_recap_status()

    def _recap_save(self):
        if self._reward_store is None:
            return
        self._reward_store.save()
        self.status_bar.showMessage(f"✓ RECAP 已保存到 {self._reward_store.filepath}", 5000)

    def _refresh_recap_table(self):
        """刷新 RECAP 区间列表。"""
        tp = self.trajectory_plot
        if self._reward_store is None:
            tp.recap_interval_table.setRowCount(0)
            self._refresh_recap_track()
            return
        intervals = self._reward_store.get_intervals(self._current_episode_idx)
        tp.recap_interval_table.setRowCount(len(intervals))
        for i, iv in enumerate(intervals):
            items = [
                str(i), str(iv["start"]), str(iv["end"]),
                iv["label"], iv.get("note", ""),
            ]
            for col, txt in enumerate(items):
                item = QTableWidgetItem(txt)
                if col == 3:
                    item.setForeground(QBrush(QColor(RECAP_LABEL_COLORS.get(txt, "#888888"))))
                tp.recap_interval_table.setItem(i, col, item)
        # 同步刷新播放时间线下方的 RECAP 彩色轨道
        self._refresh_recap_track()

    def _update_recap_status(self):
        tp = self.trajectory_plot
        if self._reward_store is None:
            tp.recap_status_label.setText("（未加载数据集）")
            return
        s = self._reward_store.stats()
        n_eps = self._reward_store.labeled_episode_count()
        total_eps = len(self.dataset.episodes) if self.dataset else 0
        tp.recap_status_label.setText(
            f"已标注 episodes: {n_eps}/{total_eps} | "
            f"区间数: {s['_intervals']} | "
            f"good帧: {s['good']} bad帧: {s['bad']} neutral帧: {s['neutral']}")

    # ── RECAP 轨道（播放时间线下方的彩色带）──

    def _on_recap_track_toggled(self, checked):
        """显示/隐藏 RECAP 标记轨道。"""
        self.recap_track.setVisible(checked)
        if checked:
            self._refresh_recap_track()

    def _refresh_recap_track(self):
        """刷新 RECAP 轨道：同步总帧数、区间、当前位置。"""
        if self._reward_store is None or self._current_episode_idx < 0:
            self.recap_track.set_total(0)
            self.recap_track.set_intervals([])
            return
        self.recap_track.set_total(self._current_total_frames)
        intervals = self._reward_store.get_intervals(self._current_episode_idx)
        self.recap_track.set_intervals(intervals)
        self.recap_track.set_current(self._current_frame_idx)

    def _on_vlm_label_requested(self, n):
        """用户点了“生成标签(VLM)”：按所选摄像头路数取视频，交给后台线程。"""
        if self.dataset is None or self._current_episode_idx < 0:
            self.trajectory_plot.subtask_result_label.setText("请先选择 Episode。")
            return
        vkeys = [name for name, f in self.dataset.features.items()
                 if self.dataset.is_video_feature(name)]
        if not vkeys:
            self.trajectory_plot.subtask_result_label.setText("数据集无视频，无法用 VLM 标签。")
            self.trajectory_plot.subtask_vlm_btn.setEnabled(True)
            return
        # 摄像头路数选择（默认 cam1）
        sel = 0
        if hasattr(self.trajectory_plot, "subtask_cam_combo"):
            sel = self.trajectory_plot.subtask_cam_combo.currentIndex()
        cam1_key = vkeys[0] if len(vkeys) >= 1 else None
        cam2_key = vkeys[1] if len(vkeys) >= 2 else None
        if sel == 1:        # 仅 cam2
            primary_key, secondary_key = cam2_key, None
        elif sel == 2:      # 双摄 cam1+cam2
            primary_key, secondary_key = cam1_key, cam2_key
        else:               # cam1（默认）
            primary_key, secondary_key = cam1_key, None
        if not primary_key:
            self.trajectory_plot.subtask_result_label.setText("所选摄像头不可用。")
            self.trajectory_plot.subtask_vlm_btn.setEnabled(True)
            return
        try:
            vpath = self.dataset.get_video_path(self._current_episode_idx, primary_key)
        except Exception:
            vpath = None
        if not vpath or not Path(vpath).exists():
            self.trajectory_plot.subtask_result_label.setText(
                f"视频不存在: {primary_key} ep{self._current_episode_idx}")
            self.trajectory_plot.subtask_vlm_btn.setEnabled(True)
            return
        # 双摄时取另一路
        cam2_path = None
        if secondary_key:
            try:
                p2 = self.dataset.get_video_path(self._current_episode_idx, secondary_key)
            except Exception:
                p2 = None
            if p2 and Path(p2).exists():
                cam2_path = p2
            else:
                self.status_bar.showMessage(f"提示：{secondary_key} 视频不存在，回退到单摄", 4000)
        segs = list(self.trajectory_plot._subtask_segments)
        if not segs:
            self.trajectory_plot.subtask_vlm_btn.setEnabled(True)
            return
        # 从标签库取已知标签作参考（提升跨 episode 一致性）
        try:
            lib = LabelLibrary(str(self.dataset.root))
            lib.load()
            known = lib.suggest(20)
        except Exception:
            known = []
        self._vlm_thread.set_work(str(vpath), segs, known_labels=known, cam2_path=cam2_path)
        if not self._vlm_thread.isRunning():
            self._vlm_thread.start()

    def _on_vlm_labels_ready(self, labels, note):
        """VLM 线程完成：把标签回填到曲线组件。"""
        self.trajectory_plot.set_segment_labels(labels, note)
        self.status_bar.showMessage(f"VLM 标签生成完成：{', '.join(labels)}")

    def _on_vlm_load_failed(self, err):
        self.trajectory_plot.subtask_vlm_btn.setEnabled(True)
        self.trajectory_plot.subtask_result_label.setText(f"VLM 失败: {err}")
        QMessageBox.warning(self, "VLM 标签失败", str(err))

    def _apply_subtask_annotations(self, assignments):
        """把预览的子任务语言标注写回 LeRobot 数据集。

        1. 保存到 meta/language_annotations.json（LanguageAnnotationStore）
        2. 写入逐帧 language_persistent 列到 data/*.parquet（write_language_to_dataset）
        3. 更新 info.json 特征注册
        """
        if self.dataset is None or self._current_episode_idx < 0:
            QMessageBox.warning(self, "无法应用", "请先选择数据集与 Episode。")
            return
        if not assignments:
            return

        ep_idx = self._current_episode_idx
        ds_root = self.dataset.root
        names = [a.get("task", "") for a in assignments]
        summary = "\n".join(
            f"  · [{a['start']}~{a['end']}] ({a['end']-a['start']+1}帧) {a['task']}"
            for a in assignments
        )
        # 二次确认（写入会修改原始 parquet）
        ret = QMessageBox.question(
            self, "确认写回数据集",
            f"将向数据集写回 Episode {ep_idx} 的子任务语言标注：\n\n{summary}\n\n"
            f"数据集：{ds_root}\n\n这会修改原始 parquet 文件（language_persistent 列），\n"
            f"建议先备份。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return

        try:
            # 1. 保存分段 JSON
            store = LanguageAnnotationStore(str(ds_root))
            store.load()
            store.set_episode(ep_idx, assignments)
            store.save()

            # 2. 写回逐帧列
            stat = write_language_to_dataset(ds_root, assignments, ep_idx, dry_run=False)

            # 3. 记入标签库（跨 episode 复用）
            lib = LabelLibrary(str(ds_root))
            lib.load()
            lib.set_episode_labels(ep_idx, names)
            lib.save()

            self.status_bar.showMessage(
                f"已写回语言标注：Episode {ep_idx}，"
                f"覆盖 {stat['covered']}/{stat['total']} 帧，"
                f"修改 {stat['files']} 个文件；标签已入库"
            )
            QMessageBox.information(
                self, "写回完成",
                f"已写回 Episode {ep_idx} 的子任务语言标注。\n"
                f"覆盖 {stat['covered']}/{stat['total']} 帧，"
                f"修改 {stat['files']} 个 parquet 文件。\n\n"
                f"子任务：{', '.join(names)}\n"
                f"标注文件：{Path(ds_root) / 'meta' / 'language_annotations.json'}\n"
                f"标签库：{Path(ds_root) / 'meta' / 'label_library.json'}",
            )
        except Exception as e:
            QMessageBox.critical(self, "写回失败", f"写回语言标注失败: {e}")

    def _denoise_current_episode(self, params):
        """对当前 Episode 应用平滑降噪。"""
        if self.dataset is None or self._current_episode_idx < 0:
            QMessageBox.warning(self, "警告", "请先选择 Episode")
            return
        reply = QMessageBox.question(
            self, "确认智能降噪",
            "确定对 Episode {} 应用平滑降噪吗？\n\n"
            "方法: {}\n窗口: {} 帧\n\n"
            "只修改 state/action 数值，帧数、元数据与视频保持不变；\n"
            "处理后自动重算 stats.json。\n"
            "是否备份取决于“剪辑前自动备份”复选框。".format(
                self._current_episode_idx,
                self.trajectory_plot.denoise_method_combo.currentText(),
                params.get("window", 11),
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._start_denoise_thread(self._current_episode_idx, params)

    def _denoise_all_episodes(self, params):
        """对所有 Episode 依次应用平滑降噪（单个线程顺序处理，避免文件并发写入）。"""
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return
        n_eps = len(self.dataset.episodes)
        if n_eps == 0:
            QMessageBox.warning(self, "警告", "当前数据集没有 Episode")
            return
        reply = QMessageBox.question(
            self, "确认批量智能降噪",
            "确定对全部 {} 个 Episode 依次应用平滑降噪吗？\n\n"
            "方法: {}\n窗口: {} 帧\n\n"
            "只修改 state/action 数值，帧数、元数据与视频保持不变；\n"
            "处理后自动重算 stats.json。\n"
            "是否备份取决于“剪辑前自动备份”复选框。".format(
                n_eps,
                self.trajectory_plot.denoise_method_combo.currentText(),
                params.get("window", 11),
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._start_denoise_thread(-1, params)

    def _start_denoise_thread(self, episode_idx, params):
        """启动降噪后台线程。episode_idx=-1 表示批量处理全部 Episode。"""
        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        if dataset_path is None:
            return
        self._pending_episode_idx = episode_idx
        self._set_controls_enabled(False)
        self.trajectory_plot.set_denoise_buttons_enabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("正在降噪...")
        self.status_bar.showMessage("正在平滑降噪，请稍候...")

        self.denoise_thread = DenoiseThread(
            dataset_path, episode_idx, params,
            backup_enabled=self.backup_before_trim_check.isChecked()
        )
        self.denoise_thread.progress.connect(self._on_denoise_progress)
        self.denoise_thread.finished_ok.connect(self._on_denoise_finished)
        self.denoise_thread.error.connect(self._on_denoise_error)
        self.denoise_thread.start()

    def _on_denoise_progress(self, current, total, msg):
        if total > 1:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        self.status_bar.showMessage(msg)

    def _on_denoise_finished(self, result, df_new):
        """降噪成功回调：刷新内存数据与当前曲线。"""
        try:
            if df_new is not None:
                self.dataset.data_df = df_new.reset_index(drop=True)
                self.dataset.num_frames = len(self.dataset.data_df)
            # 数据已变化，使关节反解缓存失效以便重新反解
            self._ik_episode_idx = -1
            self._update_episode_table()
            self._update_info()
            if 0 <= self._current_episode_idx < len(self.dataset.episodes):
                self._load_and_show_trajectory(self._current_episode_idx, self._current_frame_idx)
        except Exception as e:
            import traceback
            print(f"[Denoise] 刷新失败: {e}\n{traceback.format_exc()}")

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_controls_enabled(True)
        self.trajectory_plot.set_denoise_buttons_enabled(True)
        self.status_bar.showMessage("降噪完成")
        QMessageBox.information(self, "智能降噪完成", result)

    def _on_denoise_error(self, error_msg):
        """降噪失败回调。"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0 / 0")
        self._set_controls_enabled(True)
        self.trajectory_plot.set_denoise_buttons_enabled(True)
        self.status_bar.showMessage("降噪失败")
        QMessageBox.critical(self, "降噪失败", error_msg)

    def _trim_stationary_segments(self, intervals):
        """根据智能分析检测到的静止区间，剪辑掉这些片段。（用户手动点击，当前 episode）"""
        if self.dataset is None or self._current_episode_idx < 0:
            QMessageBox.warning(self, "警告", "请先选择 Episode")
            return

        if not intervals:
            QMessageBox.information(self, "提示", "当前没有检测到静止区间")
            return

        merged = self._merge_stationary_intervals(intervals)
        if not merged:
            QMessageBox.information(self, "提示", "当前没有检测到静止区间")
            return

        ep_frames = self.dataset.get_episode_frames(self._current_episode_idx)
        total = len(ep_frames)
        n_removed = sum(e - s + 1 for s, e in merged)

        fps = self.dataset.fps if self.dataset.fps else 20
        detail_lines = []
        for s, e in merged:
            detail_lines.append(f"帧 {s}~{e} ({e - s + 1} 帧, {s / fps:.2f}s ~ {e / fps:.2f}s)")
        detail = "\n".join(detail_lines)

        reply = QMessageBox.question(
            self, "确认剪辑静止",
            "确定删除 Episode {} 中以下 {} 段静止片段吗？\n\n"
            "{}\n\n"
            "删除帧数: {}\n"
            "保留帧数: {}\n\n"
            "删除后当前 episode 会被更新，原始数据是否备份取决于“剪辑前自动备份”复选框。".format(
                self._current_episode_idx, len(merged), detail,
                n_removed, total - n_removed
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._start_trim_stationary_thread(self._current_episode_idx, merged, show_progress=True)

    def _merge_stationary_intervals(self, intervals):
        """排序并合并相邻的静止区间。"""
        intervals = sorted([(int(s), int(e)) for s, e in intervals if s <= e])
        if not intervals:
            return []
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def _start_trim_stationary_thread(self, episode_idx, intervals, show_progress=True):
        """启动剪辑静止线程（不弹确认框）。"""
        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        self._pending_dataset_path = dataset_path
        self._pending_episode_idx = episode_idx

        self._set_controls_enabled(False)
        if show_progress:
            self.progress_bar.setRange(0, 0)
            self.progress_label.setText("正在剪辑静止片段...")
            self.status_bar.showMessage(f"正在剪辑 Episode {episode_idx} 的静止片段...")

        self.trim_thread = TrimStationaryThread(
            dataset_path, episode_idx, intervals,
            backup_enabled=self.backup_before_trim_check.isChecked()
        )
        self.trim_thread.finished_ok.connect(self._on_trim_finished)
        self.trim_thread.error.connect(self._on_trim_error)
        self.trim_thread.start()

    def _trim_all_stationary_segments(self):
        """对筛选出的 Episode 依次分析并剪辑静止区间（逐个处理，避免文件并发写入）。

        先弹出 Episode 筛选对话框（与批量删除同一交互：手动多选 / 任务名筛选），
        只处理勾选的 Episode。
        """
        if self._trim_all_in_progress:
            QMessageBox.information(self, "提示", "批量剪辑正在进行中，请等待完成。")
            return
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return

        n_eps = len(self.dataset.episodes)
        if n_eps == 0:
            QMessageBox.warning(self, "警告", "当前数据集没有 Episode")
            return

        # 弹出筛选对话框：手动多选 / 按任务名筛选（复用批量删除对话框的交互）
        rows = []
        for i, ep in enumerate(self.dataset.episodes):
            ep_idx = int(ep.get("episode_index", i))
            length = int(ep.get(
                "length", ep.get("dataset_to_index", 0) - ep.get("dataset_from_index", 0)))
            rows.append((ep_idx, length, _episode_task_display_name(self.dataset, ep)))
        dlg = BatchDeleteDialog(
            self, rows, self.dataset.fps,
            title="剪辑所有静止 - 选择 Episode",
            ok_text="✂ 剪辑选中", ok_color="#D84315")
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        indices = dlg.selected_indices()
        if not indices:
            return

        reply = QMessageBox.question(
            self, "确认剪辑所有静止",
            "确定对选中的 {} 个 Episode 依次检测并删除静止片段吗？\n\n"
            "处理规则：\n"
            "1. 逐个 Episode 分析并剪辑；\n"
            "2. 没有检测到静止的 Episode 会直接跳过（纯数据分析，不做视频解码/绘图/IK，毫秒级跳过）；\n"
            "3. 每次只写入一个 Episode，避免文件损坏；\n"
            "4. 原始数据是否备份取决于“剪辑前自动备份”复选框。\n\n"
            "该过程可能耗时较长，请耐心等待。".format(len(indices)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # 初始化批量状态（队列只含筛选出的 Episode）
        self._trim_all_in_progress = True
        self._trim_all_queue = list(indices)
        self._trim_all_total = len(indices)
        self._trim_all_trimmed = 0
        self._trim_all_skipped = 0
        self._trim_all_failed = 0
        self._trim_all_backup_enabled = self.backup_before_trim_check.isChecked()

        # 禁用备份复选框切换，避免批量过程中改变备份策略
        self.backup_before_trim_check.setEnabled(False)
        self.trajectory_plot.trim_stationary_btn.setEnabled(False)
        self.trajectory_plot.trim_all_stationary_btn.setEnabled(False)

        self.progress_bar.setRange(0, self._trim_all_total)
        self.progress_bar.setValue(0)
        self._set_controls_enabled(False)

        self._trim_all_process_next()

    def _trim_all_process_next(self):
        """处理批量队列中的下一个 Episode。"""
        if not self._trim_all_in_progress:
            return

        if not self._trim_all_queue:
            self._trim_all_finish()
            return

        self._trim_all_current_ep = self._trim_all_queue.pop(0)
        episode_idx = self._trim_all_current_ep

        self.progress_bar.setValue(self._trim_all_total - len(self._trim_all_queue) - 1)
        self.progress_label.setText(
            f"批量剪辑: {self._trim_all_total - len(self._trim_all_queue)} / {self._trim_all_total}"
        )
        self.status_bar.showMessage(
            f"正在分析 Episode {episode_idx} ({self._trim_all_total - len(self._trim_all_queue)}/{self._trim_all_total})"
        )

        try:
            # 纯数据静止分析：不再 selectRow/_on_episode_selected 做整集 UI 加载
            # （旧逻辑每集都做视频首帧解码 + matplotlib 绘图 + IK 全帧反解，
            #   无静止的 episode 也要白白加载一遍，是批量剪辑慢的根因）。
            # 检测算法与“智能分析”按钮共用 detect_stationary_intervals，结果一致。
            frames_data = self._load_episode_position_data(episode_idx)
            intervals = detect_stationary_intervals(
                frames_data,
                self.trajectory_plot._window_size,
                self.trajectory_plot._threshold_pct)
            if not intervals:
                self._trim_all_skipped += 1
                self.status_bar.showMessage(f"Episode {episode_idx} 无静止片段，跳过")
                # 无任何 UI 加载，立即处理下一个
                QTimer.singleShot(0, self._trim_all_process_next)
                return

            merged = self._merge_stationary_intervals(intervals)
            if not merged:
                self._trim_all_skipped += 1
                QTimer.singleShot(0, self._trim_all_process_next)
                return

            # 校验不能剪辑整个 episode
            ep_frames = self.dataset.get_episode_frames(episode_idx)
            total = len(ep_frames)
            n_removed = sum(e - s + 1 for s, e in merged)
            if n_removed >= total:
                self._trim_all_skipped += 1
                self.status_bar.showMessage(
                    f"Episode {episode_idx} 静止区间覆盖全部帧，跳过"
                )
                QTimer.singleShot(0, self._trim_all_process_next)
                return

            self._trim_all_trimmed += 1
            self.status_bar.showMessage(
                f"正在剪辑 Episode {episode_idx} ({self._trim_all_total - len(self._trim_all_queue)}/{self._trim_all_total})"
            )
            # 逐个剪辑，线程完成后再继续下一个
            self._start_trim_stationary_thread(episode_idx, merged, show_progress=False)
        except Exception as e:
            import traceback
            self._trim_all_failed += 1
            print(f"[TrimAll] Episode {episode_idx} 处理异常: {e}\n{traceback.format_exc()}")
            self.status_bar.showMessage(f"Episode {episode_idx} 处理失败: {e}")
            QTimer.singleShot(0, self._trim_all_process_next)

    def _trim_all_finish(self):
        """批量剪辑结束，恢复 UI 并展示结果。"""
        self._trim_all_in_progress = False

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0 / 0")
        self._set_controls_enabled(True)
        self.backup_before_trim_check.setEnabled(True)
        self.trajectory_plot.trim_stationary_btn.setEnabled(True)
        self.trajectory_plot.trim_all_stationary_btn.setEnabled(True)

        # 批量过程不再逐集加载视图；若当前有选中的 Episode，其数据可能已被
        # 剪辑更新，最后统一重载一次视图保持一致（一次性开销，不在循环内）
        if self._current_episode_idx >= 0 and self.dataset is not None:
            try:
                self._load_first_frame(self._current_episode_idx)
            except Exception:
                pass

        msg = (
            f"批量剪辑完成。\n\n"
            f"选中 Episode 数: {self._trim_all_total}\n"
            f"已剪辑: {self._trim_all_trimmed}\n"
            f"无静止跳过: {self._trim_all_skipped}\n"
            f"失败: {self._trim_all_failed}"
        )
        self.status_bar.showMessage("批量剪辑完成")
        QMessageBox.information(self, "批量剪辑完成", msg)

    def _trim_selected_segment(self):
        """删除两个标记之间的片段（包含两端），保留其余部分。"""
        if self.dataset is None or self._current_episode_idx < 0:
            QMessageBox.warning(self, "警告", "请先选择 Episode")
            return

        start, end = self.trim_slider.get_range()
        if start == end:
            QMessageBox.warning(self, "警告", "请先用下方三角标记选择要删除的片段")
            return

        ep_frames = self.dataset.get_episode_frames(self._current_episode_idx)
        total = len(ep_frames)
        if start >= end or end >= total or start < 0:
            QMessageBox.warning(self, "警告", "选择的剪辑范围无效")
            return

        fps = self.dataset.fps if self.dataset.fps else 20
        start_t = start / fps
        end_t = end / fps
        reply = QMessageBox.question(
            self, "确认剪辑",
            "确定删除 Episode {} 中以下片段吗？\\n\\n"
            "帧范围: {} ~ {}\\n"
            "时间段: {:.2f}s ~ {:.2f}s\\n"
            "删除帧数: {}\\n"
            "保留帧数: {}\\n\\n"
            "删除后当前 episode 会被更新，原始数据是否备份取决于“剪辑前自动备份”复选框。".format(
                self._current_episode_idx, start, end, start_t, end_t,
                end - start + 1, total - (end - start + 1)
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # 执行剪辑
        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        self._pending_dataset_path = dataset_path
        self._pending_episode_idx = self._current_episode_idx

        self._set_controls_enabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("正在剪辑...")
        self.status_bar.showMessage(f"正在剪辑 Episode {self._current_episode_idx}...")

        self.trim_thread = TrimEpisodeThread(
            dataset_path, self._current_episode_idx, start, end,
            backup_enabled=self.backup_before_trim_check.isChecked()
        )
        self.trim_thread.finished_ok.connect(self._on_trim_finished)
        self.trim_thread.error.connect(self._on_trim_error)
        self.trim_thread.start()

    def _set_controls_enabled(self, enabled):
        """启用/禁用操作控件。"""
        self.play_btn.setEnabled(enabled)
        self.pause_btn.setEnabled(enabled)
        self.del_btn.setEnabled(enabled)
        self.trim_btn.setEnabled(enabled)
        self.episode_table.setEnabled(enabled)
        self.dataset_combo.setEnabled(enabled)
        # 批量删除与批量剪辑共用数据集文件，必须互斥，防止并发写入损坏数据
        self.batch_del_btn.setEnabled(
            enabled and self.dataset is not None and len(self.dataset.episodes) > 0)

    def _on_trim_finished(self, result, df_new, new_ep_df, info):
        """剪辑成功回调。"""
        episode_idx = self._pending_episode_idx
        # 剪辑重写了视频文件，首帧缓存全部失效
        self._preview_frame_cache.clear()

        # 批量模式：更新数据后继续下一个，不弹成功框
        if self._trim_all_in_progress:
            self.status_bar.showMessage(f"Episode {episode_idx} 剪辑完成，继续下一个...")
            try:
                if df_new is not None:
                    self.dataset.data_df = df_new.reset_index(drop=True)
                    self.dataset.num_frames = len(self.dataset.data_df)
                if new_ep_df is not None:
                    self.dataset.episodes = new_ep_df.to_dict("records")
                if info is not None:
                    self.dataset.info = info
                self.dataset.release_video_caps()
                self._update_episode_table()
                self._update_info()
            except Exception as e:
                import traceback
                self._trim_all_failed += 1
                print(f"[TrimAll] Episode {episode_idx} 刷新失败: {e}\n{traceback.format_exc()}")
            # 继续处理下一个 episode
            QTimer.singleShot(50, self._trim_all_process_next)
            return

        self.status_bar.showMessage("正在刷新数据集...")
        try:
            # 直接用线程返回的数据更新内存，避免重新读取整个数据集
            if df_new is not None:
                self.dataset.data_df = df_new.reset_index(drop=True)
                self.dataset.num_frames = len(self.dataset.data_df)
            if new_ep_df is not None:
                self.dataset.episodes = new_ep_df.to_dict("records")
            if info is not None:
                self.dataset.info = info
            # 释放旧视频缓存，因为视频文件已被修改
            self.dataset.release_video_caps()

            self._update_episode_table()
            self._update_info()
            # 数据已变化，使关节反解缓存失效以便重新反解
            self._ik_episode_idx = -1
            # 重新选择同一 episode
            if episode_idx < len(self.dataset.episodes):
                self.episode_table.selectRow(episode_idx)
                self._on_episode_selected()
        except Exception as e:
            import traceback
            QMessageBox.warning(self, "刷新警告", f"数据集刷新失败: {e}\n\n{traceback.format_exc()}\n请手动刷新。")
            self._set_controls_enabled(True)
            return

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self._set_controls_enabled(True)
        self.trim_btn.setEnabled(True)
        self.del_btn.setEnabled(True)

        self.status_bar.showMessage(f"已剪辑 Episode {episode_idx}")
        QMessageBox.information(self, "剪辑成功", result)

    def _on_trim_error(self, error_msg):
        """剪辑失败回调。"""
        # 批量模式：记录失败并终止批量流程，避免在可能已损坏的数据上继续处理
        if self._trim_all_in_progress:
            self._trim_all_failed += 1
            self._trim_all_in_progress = False
            print(f"[TrimAll] Episode {self._pending_episode_idx} 剪辑失败: {error_msg}")

            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_label.setText("0 / 0")
            self._set_controls_enabled(True)
            self.backup_before_trim_check.setEnabled(True)
            self.trajectory_plot.trim_stationary_btn.setEnabled(True)
            self.trajectory_plot.trim_all_stationary_btn.setEnabled(True)

            self.status_bar.showMessage("批量剪辑已中止")
            QMessageBox.critical(
                self, "批量剪辑失败",
                f"Episode {self._pending_episode_idx} 剪辑失败，批量流程已中止，"
                f"请检查该 episode 数据或日志后重试。\n\n错误信息:\n{error_msg}"
            )
            return

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0 / 0")
        self.status_bar.showMessage("剪辑失败")
        QMessageBox.critical(self, "剪辑失败", error_msg)
        self._set_controls_enabled(True)
        self.trim_btn.setEnabled(False)
        self.del_btn.setEnabled(False)

    def _on_episode_selected(self):
        selected = self.episode_table.selectedItems()
        if not selected or self.dataset is None:
            return

        episode_idx = int(selected[0].text())
        self._current_episode_idx = episode_idx

        if episode_idx < len(self.dataset.episodes):
            ep = self.dataset.episodes[episode_idx]
            length = ep.get("length", 0)
            duration = length / self.dataset.fps if self.dataset.fps > 0 else 0
            self.current_episode_label.setText(
                f"已选择 Episode {episode_idx} ({length} 帧, {duration:.1f}秒)"
            )
            self.play_btn.setEnabled(True)
            self.play_btn.setText("▶ 播放")
            self.del_btn.setEnabled(True)
            self.trim_btn.setEnabled(True)

            # 自动加载并显示第一帧的图像和数据
            self._load_first_frame(episode_idx)

            # 设置任务编辑器
            self._setup_task_editor(episode_idx)

            # 刷新 RECAP 标注区间表（加载该 episode 的已标注区间）
            self._refresh_recap_table()
            self._update_recap_status()
            self.trajectory_plot.recap_start_spin.setMaximum(max(0, length - 1))
            self.trajectory_plot.recap_end_spin.setMaximum(max(0, length - 1))

    def _setup_task_editor(self, episode_idx):
        """根据选中的 episode 初始化任务名称编辑器。"""
        if not self.dataset or episode_idx < 0 or episode_idx >= len(self.dataset.episodes):
            self._reset_task_editor()
            return

        ep = self.dataset.episodes[episode_idx]
        task_name = ""

        # 优先从 episode 的 tasks 数组读取
        if "tasks" in ep and ep["tasks"] is not None:
            tasks = ep["tasks"]
            if hasattr(tasks, '__iter__') and not isinstance(tasks, str):
                task_list = list(tasks)
                if task_list:
                    task_name = str(task_list[0]).strip()

        # 如果为空，再尝试从 task_index 查找（按值查找，不能用列表位置索引）
        if not task_name and "task_index" in ep:
            task_name = str(self.dataset.task_name_for_index(ep["task_index"])).strip()

        self._original_task_name = task_name
        self._task_modified = False

        # 断开之前的信号，避免设置文本时触发保存提示
        if self._task_edit_connected:
            try:
                self.task_name_edit.editingFinished.disconnect(self._on_task_editing_finished)
            except Exception:
                pass
            self._task_edit_connected = False

        self.task_name_edit.blockSignals(True)
        self.task_name_edit.setText(task_name)
        self.task_name_edit.setEnabled(True)
        self.task_name_edit.blockSignals(False)

        self.task_status_label.setText("已加载任务名称")
        self.task_status_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")

        # 连接编辑完成信号
        self.task_name_edit.editingFinished.connect(self._on_task_editing_finished)
        self._task_edit_connected = True

    def _reset_task_editor(self):
        """重置任务编辑器为未选择状态。"""
        if self._task_edit_connected:
            try:
                self.task_name_edit.editingFinished.disconnect(self._on_task_editing_finished)
            except Exception:
                pass
            self._task_edit_connected = False

        self.task_name_edit.blockSignals(True)
        self.task_name_edit.clear()
        self.task_name_edit.setPlaceholderText("选择 Episode 后显示任务名称")
        self.task_name_edit.setEnabled(False)
        self.task_name_edit.blockSignals(False)

        self.task_status_label.setText("")
        self._original_task_name = ""
        self._task_modified = False
        self._task_edit_in_progress = False
        self._current_episode_idx = -1

    def _on_task_editing_finished(self):
        """任务名称编辑完成（按 Enter 或失去焦点）时触发。"""
        if self._current_episode_idx < 0 or self.dataset is None:
            return

        # 防止 editingFinished 重复触发（如按回车后再失去焦点）导致对话框多次弹出
        if self._task_edit_in_progress:
            return

        new_name = self.task_name_edit.text().strip()

        # 空名称检查
        if not new_name:
            QMessageBox.warning(self, "无效任务名称", "任务名称不能为空。")
            self.task_name_edit.blockSignals(True)
            self.task_name_edit.setText(self._original_task_name)
            self.task_name_edit.blockSignals(False)
            return

        # 未变化则忽略
        if new_name == self._original_task_name:
            self._task_modified = False
            self.task_status_label.setText("已加载任务名称")
            self.task_status_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
            return

        self._task_modified = True
        self.task_status_label.setText("任务名称已修改，请保存")
        self.task_status_label.setStyleSheet("QLabel { color: #FF9800; font-size: 11px; }")

        self._task_edit_in_progress = True
        try:
            # 提示是否保存
            reply = QMessageBox.question(
                self, "保存任务名称",
                f"是否将 Episode {self._current_episode_idx} 的任务名称\n"
                f"从 \"{self._original_task_name}\" 保存为 \"{new_name}\"？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._save_task_name(self._current_episode_idx, new_name)
            else:
                # 恢复原始名称
                self.task_name_edit.blockSignals(True)
                self.task_name_edit.setText(self._original_task_name)
                self.task_name_edit.blockSignals(False)
                self._task_modified = False
                self.task_status_label.setText("已恢复原始名称")
                self.task_status_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        finally:
            self._task_edit_in_progress = False

    def _save_task_name(self, episode_idx, new_name):
        """保存任务名称。每个 episode 独立命名：如果新名称已存在则复用其 task_index，否则创建新的 task_index。"""
        if not self.dataset or episode_idx < 0 or episode_idx >= len(self.dataset.episodes):
            return

        ep = self.dataset.episodes[episode_idx]
        old_name = self._original_task_name

        # 空名称/未变化不处理
        if not new_name or new_name == old_name:
            self._original_task_name = new_name
            self._task_modified = False
            self._update_episode_table()
            self._update_info()
            return

        # 查找新名称是否已存在，确定目标 task_index
        new_task_index = None
        for t in self.dataset.tasks:
            if t.get("task", "") == new_name:
                new_task_index = t.get("task_index")
                break

        # 如果不存在，创建新的 task_index
        if new_task_index is None:
            new_task_index = max(
                [t.get("task_index", 0) for t in self.dataset.tasks], default=-1
            ) + 1
            self.dataset.tasks.append({"task_index": new_task_index, "task": new_name})

        try:
            meta_dir = self.dataset.root / "meta"
            data_dir = self.dataset.root / "data"

            # 1. 更新 tasks.parquet：添加新任务（如果新名称不存在）
            tasks_path = meta_dir / "tasks.parquet"
            if tasks_path.exists():
                df = pd.read_parquet(tasks_path)
                if new_name not in df.index:
                    new_row = pd.DataFrame(
                        {"task_index": [new_task_index]}, index=[new_name]
                    )
                    df = pd.concat([df, new_row])
                    df.to_parquet(tasks_path)

            # 2. 更新 episodes 文件：只改当前 episode 的 task_index 和 tasks
            ep_parquet_dir = meta_dir / "episodes"
            ep_jsonl_path = meta_dir / "episodes.jsonl"

            if ep_parquet_dir.exists():
                ep_files = sorted(ep_parquet_dir.rglob("*.parquet"))
                if ep_files:
                    ep_df = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
                    if "task_index" in ep_df.columns:
                        ep_df.loc[episode_idx, "task_index"] = new_task_index
                    if "tasks" in ep_df.columns:
                        ep_df.at[episode_idx, "tasks"] = [new_name]

                    for p in ep_files:
                        os.remove(p)
                    chunk_dir = ep_parquet_dir / "chunk-000"
                    os.makedirs(chunk_dir, exist_ok=True)
                    for i, start in enumerate(range(0, max(len(ep_df), 1), 1000)):
                        ep_df.iloc[start:start + 1000].to_parquet(
                            chunk_dir / f"file-{i:03d}.parquet", index=False)

            elif ep_jsonl_path.exists():
                rows = []
                with open(ep_jsonl_path) as f:
                    for i, line in enumerate(f):
                        if line.strip():
                            row = json.loads(line)
                            if i == episode_idx:
                                row["task_index"] = new_task_index
                                row["tasks"] = [new_name]
                            rows.append(row)
                with open(ep_jsonl_path, "w") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")

            # 3. 更新 data.parquet：只改当前 episode 所有帧的 task_index
            data_files = sorted(data_dir.rglob("*.parquet"))
            if data_files:
                df_all = pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)
                if "episode_index" in df_all.columns and "task_index" in df_all.columns:
                    mask = df_all["episode_index"] == episode_idx
                    df_all.loc[mask, "task_index"] = new_task_index

                # 先完成 meta/episodes 指针校验（发现不一致会抛错），再删改 data 文件
                ep_df2, ep_fmt2, ep_files2 = _read_episodes(str(meta_dir))
                if ep_fmt2 == "parquet":
                    ep_df2 = _sync_episode_data_pointers(ep_df2, df_all)

                for p in data_files:
                    os.remove(p)
                chunk_dir = data_dir / "chunk-000"
                os.makedirs(chunk_dir, exist_ok=True)
                for i, start in enumerate(range(0, max(len(df_all), 1), 1000)):
                    df_all.iloc[start:start + 1000].to_parquet(
                        chunk_dir / f"file-{i:03d}.parquet", index=False)

                # data 布局已变为 1000 行/文件的合并块，同步 meta/episodes 中的 data 文件指针
                if ep_fmt2 == "parquet":
                    _remove_files([str(p) for p in ep_files2])
                    _write_parquet_chunks(ep_df2, meta_dir / "episodes" / "chunk-000")
                # task_index 已变化，重算全局统计
                _recompute_global_stats(str(meta_dir), df_all)

                # 同步内存中的 data_df
                self.dataset.data_df = df_all

            # 4. 更新内存中的 episode
            ep["task_index"] = new_task_index
            if "tasks" in ep and ep["tasks"] is not None:
                ep["tasks"] = [new_name]

            # 5. 刷新显示
            self._original_task_name = new_name
            self._task_modified = False
            self._update_episode_table()
            self._update_info()

            # 确保输入框显示的是最终保存的名称
            self.task_name_edit.blockSignals(True)
            self.task_name_edit.setText(new_name)
            self.task_name_edit.blockSignals(False)

            self.task_status_label.setText("任务名称已保存")
            self.task_status_label.setStyleSheet("QLabel { color: #4CAF50; font-size: 11px; }")
            self.status_bar.showMessage(
                f"Episode {episode_idx} 任务名称已保存为 \"{new_name}\" (task_index={new_task_index})"
            )

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "保存失败", f"保存任务名称失败:\n{str(e)}\n\n{traceback.format_exc()}")
            # 恢复原始名称
            self.task_name_edit.blockSignals(True)
            self.task_name_edit.setText(self._original_task_name)
            self.task_name_edit.blockSignals(False)

    # ==================== 批量修改任务名称 ====================
    def _episode_task_name(self, ep):
        tasks = ep.get("tasks")
        if tasks is None:
            return ""
        try:
            tasks = list(tasks)
        except TypeError:
            return str(tasks)
        return str(tasks[0]) if tasks else ""

    def _show_task_batch_dialog(self):
        """批量修改任务名对话框：自动带出当前 Episode 的任务名，输入新名后全部替换。"""
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return
        task_names = [t.get("task", "") for t in self.dataset.tasks if t.get("task")]
        if not task_names:
            QMessageBox.warning(self, "警告", "当前数据集没有任务")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("批量修改任务名称")
        dlg.setMinimumWidth(420)
        layout = QVBoxLayout(dlg)

        form = QFormLayout()
        old_combo = QComboBox()
        old_combo.addItems(task_names)
        current = self._original_task_name
        if current in task_names:
            old_combo.setCurrentText(current)
        form.addRow("原始任务名:", old_combo)

        new_edit = QLineEdit()
        new_edit.setPlaceholderText("输入新任务名称")
        from PyQt6.QtGui import QRegularExpressionValidator
        from PyQt6.QtCore import QRegularExpression
        new_edit.setValidator(QRegularExpressionValidator(
            QRegularExpression(r"^[\w\x{4e00}-\x{9fa5}][\w\x{4e00}-\x{9fa5}\s\-]*$"), new_edit
        ))
        form.addRow("新任务名:", new_edit)
        layout.addLayout(form)

        count_label = QLabel()
        count_label.setStyleSheet("QLabel { color: #666; }")

        def update_count():
            name = old_combo.currentText()
            n = sum(1 for ep in self.dataset.episodes
                    if self._episode_task_name(ep) == name)
            count_label.setText(f"将影响 {n} 个 Episode（同名任务全部替换）")

        old_combo.currentTextChanged.connect(lambda _: update_count())
        update_count()
        layout.addWidget(count_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        apply_btn = QPushButton("全部替换")
        apply_btn.setStyleSheet(
            "QPushButton { background-color: #E65100; color: white; font-weight: bold; padding: 6px 18px; }"
        )
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        def do_apply():
            old_name = old_combo.currentText()
            new_name = new_edit.text().strip()
            if not new_name:
                QMessageBox.warning(dlg, "提示", "请输入新任务名称")
                return
            if new_name == old_name:
                QMessageBox.information(dlg, "提示", "新名称与原名称相同，无需修改")
                return
            n = sum(1 for ep in self.dataset.episodes
                    if self._episode_task_name(ep) == old_name)
            if n == 0:
                QMessageBox.information(dlg, "提示", f"没有使用任务 \"{old_name}\" 的 Episode")
                return
            ret = QMessageBox.question(
                dlg, "确认批量替换",
                f"将把 {n} 个 Episode 的任务\n  \"{old_name}\"\n全部替换为\n  \"{new_name}\"\n\n是否继续？"
            )
            if ret != QMessageBox.StandardButton.Yes:
                return
            ok, msg = self._batch_rename_task(old_name, new_name)
            if ok:
                dlg.accept()
                QMessageBox.information(self, "完成", msg)
            else:
                QMessageBox.critical(dlg, "失败", msg)

        apply_btn.clicked.connect(do_apply)
        dlg.exec()

    def _batch_rename_task(self, old_name, new_name):
        """批量重命名任务：所有任务名为 old_name 的 Episode 改为 new_name。

        安全性（对照 数据问题分析.md）：
        - data parquet 逐文件原地改写 task_index 列，行数/行序/文件布局不变，
          data/chunk_index、data/file_index 指针保持有效（避免问题1）；
        - frame_index/timestamp 不触碰（避免问题3）；
        - task_index 变化后重算全局 stats.json 与受影响 episode 的 per-episode
          stats/task_index 列（避免问题4）；
        - 所有被修改的文件先备份到 <root>/.bak_task_rename_<时间戳>/。
        """
        if self.dataset is None:
            return False, "未加载数据集"
        root = self.dataset.root
        meta_dir = root / "meta"
        tasks_path = meta_dir / "tasks.parquet"
        try:
            tasks_df = pd.read_parquet(tasks_path)
        except Exception as e:
            return False, f"读取 tasks.parquet 失败: {e}"
        if old_name not in tasks_df.index:
            return False, f"任务 \"{old_name}\" 不存在"

        old_idx = int(tasks_df.loc[old_name, "task_index"])
        merge_existing = new_name in tasks_df.index
        new_idx = int(tasks_df.loc[new_name, "task_index"]) if merge_existing else old_idx

        backup_dir = root / f".bak_task_rename_{datetime.now():%Y%m%d_%H%M%S}"
        backed_up = []

        def backup_file(p):
            p = Path(p)
            rel = p.relative_to(root)
            dst = backup_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            backed_up.append(p)

        try:
            # 1. tasks.parquet：纯改名只换名称（task_index 不变，data 无需改动）；
            #    并入已有任务则删除旧任务行，并把剩余 task_index 压实为 0 基连续
            #    （lerobot dataset_reader 用 iloc 按位置取任务，留空洞会越界，
            #    见 数据问题分析.md 问题7）
            backup_file(tasks_path)
            compact = None
            frame_remap = None
            if merge_existing:
                tasks_df = tasks_df.drop(index=old_name)
                tasks_df = tasks_df.sort_values("task_index")
                compact = {int(v): i for i, v in enumerate(tasks_df["task_index"])}
                tasks_df["task_index"] = [compact[int(v)] for v in tasks_df["task_index"]]
                frame_remap = dict(compact)
                frame_remap[old_idx] = compact[new_idx]
                new_idx = compact[new_idx]
            else:
                tasks_df = tasks_df.rename(index={old_name: new_name})
            tasks_df.to_parquet(tasks_path, index=True)

            # 2. data parquet：仅并入已有任务时需要改写 task_index（旧值→新值，
            #    其余值随压实同步平移），逐文件原地改写，保持文件布局与行序不变
            if merge_existing:
                data_files = sorted((root / "data").rglob("*.parquet"))
                for p in data_files:
                    df = pd.read_parquet(p)
                    if "task_index" not in df.columns:
                        continue
                    mapped = df["task_index"].map(
                        lambda v: frame_remap.get(int(v), int(v))
                    )
                    if bool((mapped == df["task_index"]).all()):
                        continue
                    backup_file(p)
                    df["task_index"] = mapped.astype(df["task_index"].dtype)
                    df.to_parquet(p, index=False)

            # 3. meta/episodes：更新 tasks 字符串；并入已有任务时 task_index 列
            #    与 per-episode stats/task_index 统计列需随压实整体平移（不只改
            #    被并任务的行），逐文件原地改写
            ep_files = sorted((meta_dir / "episodes").rglob("*.parquet"))
            affected_eps = set()
            for p in ep_files:
                df = pd.read_parquet(p)
                if "tasks" not in df.columns:
                    continue
                tasks_col = df["tasks"].to_numpy(copy=True)
                row_hit = []
                for pos in range(len(df)):
                    tv = tasks_col[pos]
                    try:
                        tl = list(tv) if tv is not None else []
                    except TypeError:
                        tl = [tv]
                    if old_name in tl:
                        row_hit.append((pos, tl))
                ti_mapped = None
                if merge_existing and "task_index" in df.columns:
                    ti_mapped = df["task_index"].map(
                        lambda v: frame_remap.get(int(v), int(v))
                    )
                ti_changed = ti_mapped is not None and bool(
                    (ti_mapped != df["task_index"]).any()
                )
                if not row_hit and not ti_changed:
                    continue
                backup_file(p)
                for pos, tl in row_hit:
                    new_tasks = [new_name if t == old_name else t for t in tl]
                    tasks_col[pos] = np.array(new_tasks, dtype=object)
                    if "episode_index" in df.columns:
                        affected_eps.add(int(df["episode_index"].iloc[pos]))
                    else:
                        affected_eps.add(pos)
                df["tasks"] = tasks_col
                if ti_mapped is not None:
                    df["task_index"] = ti_mapped.astype(df["task_index"].dtype)
                    # 每个 episode 只有一个任务：min=max=mean=分位数=task_index，std=0
                    ti_vals = df["task_index"].to_numpy(dtype=np.float64)
                    prefix = "stats/task_index/"
                    for suffix, arr in [
                        ("min", ti_vals), ("max", ti_vals), ("mean", ti_vals),
                        ("std", np.zeros_like(ti_vals)), ("q01", ti_vals),
                        ("q10", ti_vals), ("q50", ti_vals), ("q90", ti_vals),
                        ("q99", ti_vals),
                    ]:
                        col = prefix + suffix
                        if col not in df.columns:
                            continue
                        col_vals = np.empty(len(arr), dtype=object)
                        for i, v in enumerate(arr):
                            col_vals[i] = np.array([float(v)], dtype=np.float64)
                        df[col] = col_vals
                df.to_parquet(p, index=False)

            if not affected_eps:
                # 数据层面没有引用该任务的 episode，只有 tasks.parquet 变了
                pass

            # 4. task_index 值发生变化时，重算全局 stats.json（含图像 count 同步）
            if merge_existing:
                stats_path = meta_dir / "stats.json"
                if stats_path.exists():
                    backup_file(stats_path)
                df_all = self.dataset.data_df.copy()
                if "task_index" in df_all.columns:
                    df_all["task_index"] = df_all["task_index"].map(
                        lambda v: frame_remap.get(int(v), int(v))
                    )
                _recompute_global_stats(str(meta_dir), df_all)
                self.dataset.data_df = df_all

                # info.json 的 total_tasks 同步
                info_path = meta_dir / "info.json"
                if info_path.exists():
                    backup_file(info_path)
                    with open(info_path, encoding="utf-8") as f:
                        info = json.load(f)
                    info["total_tasks"] = len(tasks_df)
                    with open(info_path, "w", encoding="utf-8") as f:
                        json.dump(info, f, indent=2, ensure_ascii=False)
                    self.dataset.info = info

            # 5. 内存同步（并入时保留任务的 task_index 随压实同步平移）
            new_tasks_mem = []
            for t in self.dataset.tasks:
                if t.get("task") == old_name:
                    if merge_existing:
                        continue
                    new_tasks_mem.append({"task_index": new_idx, "task": new_name})
                else:
                    if merge_existing and compact is not None:
                        t = dict(t)
                        t["task_index"] = compact.get(
                            int(t.get("task_index", -1)), t.get("task_index")
                        )
                    new_tasks_mem.append(t)
            self.dataset.tasks = new_tasks_mem
            for ep in self.dataset.episodes:
                if self._episode_task_name(ep) == old_name:
                    ep["tasks"] = [new_name]
                if merge_existing and "task_index" in ep:
                    ep["task_index"] = frame_remap.get(
                        int(ep["task_index"]), ep["task_index"]
                    )

            # 6. 界面刷新；若当前编辑的 Episode 受影响，同步任务编辑框
            if self._original_task_name == old_name:
                self._original_task_name = new_name
                self.task_name_edit.blockSignals(True)
                self.task_name_edit.setText(new_name)
                self.task_name_edit.blockSignals(False)
            self._update_episode_table()
            self._update_info()

            n_eps = len(affected_eps)
            mode = "并入已有任务" if merge_existing else "重命名"
            self.status_bar.showMessage(
                f"批量替换完成: \"{old_name}\" -> \"{new_name}\" ({mode}, {n_eps} 个 Episode)"
            )
            return True, (
                f"已将 {n_eps} 个 Episode 的任务\n\"{old_name}\" → \"{new_name}\"\n"
                f"方式: {mode}\n原文件备份: {backup_dir.name}"
            )
        except Exception as e:
            import traceback
            return False, f"批量替换失败:\n{e}\n\n{traceback.format_exc()}\n\n已修改文件的原始版本备份在: {backup_dir}"

    def _toggle_playback(self):
        if self.playback_thread is not None and self.playback_thread.isRunning():
            self._stop_playback()
            return

        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return

        selected = self.episode_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "警告", "请先选择一个 Episode")
            return

        episode_idx = int(selected[0].text())
        auto_play = self.auto_play_check.isChecked()
        self._start_playback(episode_idx, auto_play)

    def _toggle_pause(self):
        if self.playback_thread is None:
            return
        if self.playback_thread.is_paused():
            self.playback_thread.resume()
            self.pause_btn.setText("⏸ 暂停")
        else:
            self.playback_thread.pause()
            self.pause_btn.setText("▶ 继续")
            
    def _start_playback(self, episode_index, auto_play=False):
        self._stop_playback()

        # 确保 trim_slider 与当前 episode 帧数同步
        ep_frames = self.dataset.get_episode_frames(episode_index)
        total_frames = len(ep_frames)
        self._current_total_frames = total_frames
        self.trim_slider.set_total_frames(total_frames)
        self.trim_slider.set_current_frame(0)
        self.trim_slider.setEnabled(True)
        self.progress_label.setText(f"1 / {total_frames}")

        # 重新加载该 episode 的位置曲线（_stop_playback 会清空它）；
        # 保留曲线上的子任务分段色块与 VLM 标签（不随播放清除）
        self._load_and_show_trajectory(episode_index, current_frame=0,
                                       clear_annotations=False)

        speed = self.speed_spin.value()
        
        self.playback_thread = PlaybackThread(
            self.dataset, episode_index, speed, auto_play
        )
        self.playback_thread.frame_data.connect(self._on_frame_data)
        self.playback_thread.frame_updated.connect(self._on_frame_updated)
        self.playback_thread.episode_finished.connect(self._on_episode_finished)
        self.playback_thread.playback_finished.connect(self._on_playback_finished)
        self.playback_thread.error_occurred.connect(self._on_error)
        self.playback_thread.status_changed.connect(self._on_status_changed)
        
        self.playback_thread.start()

        self.play_btn.setEnabled(True)
        self.play_btn.setText("⏹ 停止")
        self.play_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px; }"
        )
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("⏸ 暂停")
        self.auto_play_check.setEnabled(False)
        self.status_bar.showMessage("回放中")
        
    def _load_first_frame_cached(self, episode_idx, camera_name):
        """带 LRU 缓存的首帧预览加载：回访同一 episode 时免去视频打开/解码。"""
        key = (episode_idx, camera_name)
        img = self._preview_frame_cache.get(key)
        if img is not None:
            self._preview_frame_cache.move_to_end(key)
            return img
        img = self._load_image_for_preview(episode_idx, 0, camera_name)
        if img is not None and isinstance(img, np.ndarray) and img.size > 0:
            self._preview_frame_cache[key] = img
            self._preview_frame_cache.move_to_end(key)
            while len(self._preview_frame_cache) > self._PREVIEW_CACHE_MAX:
                self._preview_frame_cache.popitem(last=False)
        return img

    def _load_first_frame(self, episode_idx):
        """加载并显示指定 episode 的第一帧图像和数据（不启动播放线程）"""
        try:
            ep_frames = self.dataset.get_episode_frames(episode_idx)
            total_frames = len(ep_frames)
            if total_frames == 0:
                return
            
            # 获取第一帧数据
            row = ep_frames.iloc[0]
            
            # 加载图像（LRU 缓存：回访的 episode 不再重复解码视频）
            cam1_img = self._load_first_frame_cached(episode_idx, "observation.images.cam1")
            cam2_img = self._load_first_frame_cached(episode_idx, "observation.images.cam2")
            
            # 获取 action 和 state
            action = np.array(row.get("action", [0] * 7))
            state = np.array(row.get("observation.state", [0] * 7))
            timestamp = row.get("timestamp", 0.0)
            task_index = int(row.get("task_index", 0))
            
            # 获取任务名称（按 task_index 值查找，不能用列表位置索引）
            task_name = self.dataset.task_name_for_index(task_index)
            
            # 更新显示
            self._display_frame(cam1_img, cam2_img, action, state, 0, timestamp, task_name)
            
            # 更新进度条为初始状态
            self._current_total_frames = total_frames
            self._current_frame_idx = 0
            self.trim_slider.set_total_frames(total_frames)
            self.trim_slider.set_current_frame(0)
            self.trim_slider.setEnabled(True)
            self.progress_label.setText(f"1 / {total_frames}")
            self._update_trim_info()

            # 加载并显示该 episode 的 xyz 位置曲线
            self._load_and_show_trajectory(episode_idx, current_frame=0)
            
        except Exception as e:
            print(f"[Preview] 加载第一帧失败: {e}")

    def _load_and_show_trajectory(self, episode_idx, current_frame=0, clear_annotations=True):
        """加载指定 episode 的 xyz 位置数据并刷新右侧曲线图。

        clear_annotations=False 时保留曲线上的子任务分段色块与 VLM 标签
        （用于点播放重绘同一 episode）。
        """
        try:
            frames_data = self._load_episode_position_data(episode_idx)
            self.trajectory_plot.set_data(frames_data, current_frame=current_frame,
                                          clear_annotations=clear_annotations)
        except Exception as e:
            print(f"[Trajectory] 加载位置曲线失败: {e}")
            self.trajectory_plot.clear()

        # 同步启动机械臂关节角度批量反解（后台线程，完成后刷新 3D 视图）
        self._start_ik_solve(episode_idx, current_frame=current_frame)

    # ------------------------------------------------------------------
    # 机械臂关节反解（IK）
    # ------------------------------------------------------------------
    def _start_ik_solve(self, episode_idx, current_frame=0):
        """后台批量反解整段 episode 的关节角度，避免阻塞 UI。"""
        if _KIN is None:
            return
        if self.dataset is None or episode_idx < 0:
            return
        # 同一 episode 已反解完成，直接刷新当前帧
        if self._ik_episode_idx == episode_idx and self.robot_arm_widget.has_data():
            self.robot_arm_widget.show_frame(current_frame)
            return

        # 清空旧数据，复位机械臂视图
        self.robot_arm_widget.clear()
        self._ik_episode_idx = episode_idx
        self._ik_solving = True
        self.status_bar.showMessage(f"正在反解 Episode {episode_idx} 关节角度...")

        try:
            ep_frames = self.dataset.get_episode_frames(episode_idx)
            if ep_frames is None or len(ep_frames) == 0 or "observation.state" not in ep_frames.columns:
                self._ik_solving = False
                return
            states = np.vstack(ep_frames["observation.state"].to_numpy())
        except Exception as e:
            print(f"[IK] 提取位姿数据失败: {e}")
            self._ik_solving = False
            return

        # 取消上一次未完成的反解线程（不 wait() 阻塞 UI——旧线程逐帧检查取消
        # 标志后自行退出，其结果因 episode_idx 过期会被 _on_ik_finished 丢弃；
        # 保留引用直到线程真正结束，避免 QThread 运行中被销毁）
        old = self._ik_thread
        if old is not None:
            if old.isRunning():
                old.cancel()
                self._ik_retiring.add(old)
                old.finished.connect(lambda: self._ik_retiring.discard(old))
            else:
                old.deleteLater()

        self._ik_thread = IKSolveThread(states, episode_idx)
        self._ik_thread.finished_ok.connect(self._on_ik_finished)
        self._ik_thread.failed.connect(self._on_ik_failed)
        self._ik_thread.start()

    def _on_ik_finished(self, angles, episode_idx):
        """关节角度反解完成回调：注入 3D 视图并刷新当前帧。"""
        self._ik_solving = False
        if episode_idx != self._current_episode_idx:
            # 用户已切换到其它 episode，丢弃过期结果
            return
        if angles is None or len(angles) == 0:
            self.status_bar.showMessage("关节角度反解失败：无有效数据")
            return
        self.robot_arm_widget.set_joint_angles(angles)
        # 刷新到当前播放/预览帧
        frame = self._current_frame_idx if self._current_frame_idx < len(angles) else 0
        self.robot_arm_widget.show_frame(frame)
        self.status_bar.showMessage(
            f"已反解 Episode {episode_idx} 共 {len(angles)} 帧关节角度"
        )

    def _on_ik_failed(self, error_msg, episode_idx):
        self._ik_solving = False
        if episode_idx != self._current_episode_idx:
            return
        self.status_bar.showMessage("关节角度反解失败")
        print(f"[IK] 反解失败: {error_msg}")

    def _load_episode_position_data(self, episode_idx):
        """从 data_df 中提取指定 episode 的位姿序列。

        返回 dict: {
            'x': array, 'y': array, 'z': array,
            'rx': array, 'ry': array, 'rz': array,  # 欧拉角 (rad)
            'gripper': array,  # 夹爪 (rad)
            'timestamps': array
        }
        自动兼容两种格式：
          - 7 维新格式 [x,y,z,rx,ry,rz,gripper_rad]：直接读取
          - 8 维旧格式 [x,y,z,qx,qy,qz,qw,gripper_deg]：四元数→欧拉角，度→弧度
        如果数据不存在则返回 None。
        """
        if self.dataset is None or episode_idx < 0:
            return None
        try:
            ep_frames = self.dataset.get_episode_frames(episode_idx)
            if ep_frames is None or len(ep_frames) == 0:
                return None

            if "observation.state" not in ep_frames.columns:
                return None

            states = ep_frames["observation.state"].to_numpy()
            states = np.vstack(states)
            if states.shape[1] < 3:
                return None
            n = states.shape[0]

            result = {
                'x': states[:, 0],
                'y': states[:, 1],
                'z': states[:, 2],
            }

            if states.shape[1] >= 8:
                # 旧格式：[x,y,z,qx,qy,qz,qw,gripper_deg] → 转欧拉角(弧度)
                qx = states[:, 3].astype(np.float64)
                qy = states[:, 4].astype(np.float64)
                qz = states[:, 5].astype(np.float64)
                qw = states[:, 6].astype(np.float64)
                rx_arr = np.empty(n); ry_arr = np.empty(n); rz_arr = np.empty(n)
                for i in range(n):
                    rx_arr[i], ry_arr[i], rz_arr[i] = quat_to_rpy_recording(
                        float(qw[i]), float(qx[i]), float(qy[i]), float(qz[i]))
                result['rx'] = rx_arr
                result['ry'] = ry_arr
                result['rz'] = rz_arr
                # 夹爪度→弧度
                result['gripper'] = np.radians(states[:, 7])
            elif states.shape[1] >= 7:
                # 新格式：[x,y,z,rx,ry,rz,gripper_rad] 直接读取
                result['rx'] = states[:, 3]
                result['ry'] = states[:, 4]
                result['rz'] = states[:, 5]
                result['gripper'] = states[:, 6]
            else:
                # 不足 7 列：欧拉角补零
                result['rx'] = np.zeros(n)
                result['ry'] = np.zeros(n)
                result['rz'] = np.zeros(n)
                result['gripper'] = np.zeros(n)

            if "timestamp" in ep_frames.columns:
                result['timestamps'] = ep_frames["timestamp"].to_numpy()

            # 夹爪力矩 (Nm)：新录制的数据集含 observation.gripper_torque 列
            if "observation.gripper_torque" in ep_frames.columns:
                try:
                    tq = ep_frames["observation.gripper_torque"].to_numpy()
                    tq = np.vstack(tq).reshape(-1)
                    result['gripper_torque'] = tq.astype(np.float64)
                except Exception:
                    pass
            return result
        except Exception as e:
            print(f"[Trajectory] 提取位姿数据失败: {e}")
            return None
            
    def _load_image_for_preview(self, episode_index, frame_index, camera_name):
        """为预览加载图像，支持三种存储方式：
        1. videos/ 目录下的视频文件
        2. images/ 目录下的 PNG 文件
        3. parquet 中的 bytes 字段
        """
        # 方式1: 优先从 videos/ 目录加载
        if self.dataset.is_video_feature(camera_name):
            try:
                frame = self.dataset.get_video_frame(episode_index, frame_index, camera_name)
                if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                    return frame
            except Exception as e:
                print(f"[Preview] 从视频加载失败: {e}")
        
        # 方式2: 尝试从 images/ 目录加载
        img_path = self.dataset.get_image_path(episode_index, frame_index, camera_name)
        if img_path and os.path.exists(img_path):
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            return np.array(img)
        
        # 方式3: 从 parquet 的 bytes 字段解码
        try:
            ep_frames = self.dataset.get_episode_frames(episode_index)
            if frame_index < len(ep_frames):
                row = ep_frames.iloc[frame_index]
                if camera_name in row:
                    img_data = row[camera_name]
                    if isinstance(img_data, dict):
                        if 'bytes' in img_data and img_data['bytes']:
                            from PIL import Image
                            import io
                            img = Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
                            return np.array(img)
                        elif 'path' in img_data and img_data['path']:
                            from PIL import Image
                            img = Image.open(img_data['path']).convert("RGB")
                            return np.array(img)
        except Exception as e:
            print(f"[Preview] 从 parquet 加载图像失败: {e}")
        
        # 返回黑色图像作为占位
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def _on_image_double_clicked(self, camera_name):
        """双击图像回放框时，弹出对应摄像头当前帧的全分辨率图像。"""
        if self.dataset is None or self._current_episode_idx < 0:
            return

        try:
            img = self._load_image_for_preview(
                self._current_episode_idx, self._current_frame_idx, camera_name
            )
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                QMessageBox.warning(self, "警告", "无法加载当前帧图像")
                return

            cam_display_name = "摄像头 1" if "cam1" in camera_name else "摄像头 2"
            title = (
                f"{cam_display_name} 全分辨率图像 - "
                f"Episode {self._current_episode_idx} / 帧 {self._current_frame_idx}"
            )
            dialog = FullscreenImageDialog(self, img, title)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开全分辨率图像失败:\n{str(e)}")
        
    @staticmethod
    def _format_state_text(state, step, timestamp, task_name):
        """把单帧 state 数组格式化为信息文本，自动兼容 7 维欧拉角 / 8 维四元数格式。"""
        state = np.asarray(state, dtype=float)
        pos = state[:3]
        if state.shape[0] >= 8:
            # 旧四元数格式：转欧拉角显示，夹爪度→弧度
            rx, ry, rz = quat_to_rpy_recording(
                float(state[6]), float(state[3]), float(state[4]), float(state[5]))
            grip = math.radians(float(state[7])) if state.shape[0] > 7 else 0.0
        elif state.shape[0] >= 7:
            # 新欧拉角格式：直接读取
            rx, ry, rz = float(state[3]), float(state[4]), float(state[5])
            grip = float(state[6])
        else:
            rx = ry = rz = grip = 0.0
        return (
            f"帧: {step} | 时间: {timestamp:.3f}s | 任务: {task_name}\n"
            f"位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n"
            f"欧拉角(rad): [{rx:.3f}, {ry:.3f}, {rz:.3f}]\n"
            f"夹爪(rad): {grip:.3f}"
        )

    def _display_frame(self, rgb_cam1, rgb_cam2, action, state, step, timestamp, task_name):
        """显示单帧图像和数据（不依赖播放线程）"""
        self.cam1_label.setImage(rgb_cam1)
        self.cam2_label.setImage(rgb_cam2)
        self._current_frame_idx = step
        self.frame_info_label.setText(
            self._format_state_text(state, step, timestamp, task_name))

    def _on_frame_data(self, rgb_cam1, rgb_cam2, action, state, step, timestamp, task_name):
        self.cam1_label.setImage(rgb_cam1)
        self.cam2_label.setImage(rgb_cam2)
        # 播放线程推送新帧时同步播放头位置
        if self._current_total_frames > 0:
            self.trim_slider.set_current_frame(step)
        # 自动兼容 7 维欧拉角 / 8 维四元数格式
        self.frame_info_label.setText(
            self._format_state_text(state, step, timestamp, task_name))
        
    def _update_data_panel(self, step, timestamp, pos, quat, action, task_name=""):
        """更新机械臂数据面板显示（已删除面板，方法保留兼容）"""
        pass
        
    def _on_frame_updated(self, current, total):
        self._current_total_frames = total
        self._current_frame_idx = max(0, current - 1)
        if total > 0:
            self.trim_slider.set_current_frame(current - 1)
            self.progress_label.setText(f"{current} / {total}")
            # 同步更新位置曲线竖线
            self.trajectory_plot.set_current_frame(current - 1)
            # 同步 RECAP 轨道播放头
            self.recap_track.set_current(current - 1)
            # 同步机械臂 3D 视图到当前播放帧
            self.robot_arm_widget.show_frame(current - 1)
            
    def _on_episode_finished(self):
        self.trim_slider.set_current_frame(0)
        
    def _on_playback_finished(self):
        """播放自然结束时，恢复到 episode 仍被选中的状态，不用重新去列表里点。"""
        self._restore_after_playback()
        self.status_bar.showMessage("播放完成")

    def _restore_after_playback(self):
        """恢复到当前 Episode 已选择、可立即重新播放/剪辑的状态。"""
        self.play_btn.setEnabled(True)
        self.play_btn.setText("▶ 播放")
        self.play_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ 暂停")
        self.auto_play_check.setEnabled(True)

        if self._current_episode_idx >= 0 and self.dataset is not None:
            # 保持 episode 选择状态，并启用对应操作按钮
            self.del_btn.setEnabled(True)
            self.trim_btn.setEnabled(True)
            if self._current_total_frames > 0:
                self.trim_slider.setEnabled(True)
                self.progress_label.setText(f"1 / {self._current_total_frames}")
        else:
            self.del_btn.setEnabled(False)
            self.trim_btn.setEnabled(False)
            self.trim_slider.setEnabled(False)
            self.progress_label.setText("0 / 0")

    def _on_status_changed(self, status):
        self.status_bar.showMessage(status)

    def _on_error(self, msg):
        QMessageBox.critical(self, "回放错误", msg)
        self._reset_ui_state()

    def _stop_playback(self):
        if self.playback_thread is not None:
            self.playback_thread.stop()
            self.playback_thread.wait(3000)
            self.playback_thread = None

        self._restore_after_playback()

    def _reset_ui_state(self):
        self.play_btn.setEnabled(True)
        self.play_btn.setText("▶ 播放")
        self.play_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ 暂停")
        self.del_btn.setEnabled(False)
        self.auto_play_check.setEnabled(True)
        
        self.trim_slider.set_total_frames(0)
        self.trim_slider.setEnabled(False)
        self.progress_label.setText("0 / 0")
        self.trim_info_label.setText("剪辑区域: 未选择")
        self._current_total_frames = 0
        
        self.cam1_label.setText("摄像头 1 (Cam1)")
        self.cam1_label.clearImage()
        self.cam2_label.setText("摄像头 2 (Cam2)")
        self.cam2_label.clearImage()
        self.frame_info_label.setText("帧信息: 未开始播放")

        # 清空位置曲线
        self.trajectory_plot.clear()

        # 清空机械臂 3D 视图与反解缓存
        self.robot_arm_widget.clear()
        self._ik_episode_idx = -1
        self._ik_solving = False

        self.status_bar.showMessage("就绪 - 请选择数据集")
        
    def _delete_selected_episode(self):
        """删除选中的 episode（使用后台线程，避免 UI 卡顿）。"""
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return

        selected = self.episode_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "警告", "请先选择一个 Episode")
            return

        episode_idx = int(selected[0].text())
        ep = self.dataset.episodes[episode_idx]
        length = ep.get("length", ep.get("dataset_to_index", 0) - ep.get("dataset_from_index", 0))
        duration = length / self.dataset.fps if self.dataset.fps > 0 else 0

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要永久删除 Episode {episode_idx} 吗？\n\n"
            f"帧数: {length}\n"
            f"时长: {duration:.1f}秒\n"
            f"此操作不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # 获取数据集路径
        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        if dataset_path is None:
            return
        self._pending_dataset_path = dataset_path
        self._pending_episode_idx = episode_idx

        # 禁用所有操作按钮
        self.del_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.episode_table.setEnabled(False)
        self.dataset_combo.setEnabled(False)

        # 显示忙碌进度条
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("正在删除...")
        self.status_bar.showMessage(f"正在删除 Episode {episode_idx}，请稍候...")

        # 启动后台线程
        self.delete_thread = DeleteEpisodeThread(dataset_path, episode_idx)
        self.delete_thread.finished_ok.connect(self._on_delete_finished)
        self.delete_thread.error.connect(self._on_delete_error)
        self.delete_thread.start()

    def _batch_delete_episodes(self):
        """批量删除 episode：对话框多选/任务筛选，后台线程单次重写数据集。"""
        if self.dataset is None:
            QMessageBox.warning(self, "警告", "请先选择数据集")
            return

        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        if dataset_path is None:
            return

        # 组装对话框行数据：(episode_index, length, task_name)
        rows = []
        for i, ep in enumerate(self.dataset.episodes):
            ep_idx = int(ep.get("episode_index", i))
            length = int(ep.get(
                "length", ep.get("dataset_to_index", 0) - ep.get("dataset_from_index", 0)))
            rows.append((ep_idx, length, _episode_task_display_name(self.dataset, ep)))

        dlg = BatchDeleteDialog(self, rows, self.dataset.fps)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        indices = dlg.selected_indices()
        if not indices:
            return
        if len(indices) >= len(self.dataset.episodes):
            QMessageBox.warning(self, "警告", "不能删除全部 episode（至少保留 1 个）")
            return

        length_map = {r[0]: r[1] for r in rows}
        total_frames = sum(length_map[i] for i in indices)
        duration = total_frames / self.dataset.fps if self.dataset.fps > 0 else 0
        shown = indices if len(indices) <= 20 else indices[:20]
        idx_text = ", ".join(str(i) for i in shown)
        if len(indices) > 20:
            idx_text += f", ... 共 {len(indices)} 个"

        reply = QMessageBox.question(
            self, "确认批量删除",
            f"确定要永久删除以下 {len(indices)} 个 Episode 吗？\n\n"
            f"编号: {idx_text}\n"
            f"总帧数: {total_frames}\n"
            f"总时长: {duration:.1f}秒\n"
            f"此操作不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._pending_dataset_path = dataset_path
        self._pending_episode_indices = indices

        # 禁用所有操作按钮（剪辑类按钮与删除共用数据集文件，必须互斥）
        self.del_btn.setEnabled(False)
        self.batch_del_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.trim_btn.setEnabled(False)
        self.trajectory_plot.trim_stationary_btn.setEnabled(False)
        self.trajectory_plot.trim_all_stationary_btn.setEnabled(False)
        self.episode_table.setEnabled(False)
        self.dataset_combo.setEnabled(False)

        # 显示忙碌进度条
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("正在批量删除...")
        self.status_bar.showMessage(f"正在批量删除 {len(indices)} 个 Episode，请稍候...")

        # 启动后台线程
        self.batch_delete_thread = BatchDeleteEpisodesThread(dataset_path, indices)
        self.batch_delete_thread.finished_ok.connect(self._on_batch_delete_finished)
        self.batch_delete_thread.error.connect(self._on_delete_error)
        self.batch_delete_thread.start()

    def _on_batch_delete_finished(self, result):
        """批量删除成功回调"""
        indices = self._pending_episode_indices
        dataset_path = self._pending_dataset_path

        # 重新加载数据集
        self.status_bar.showMessage("正在重新加载数据集...")
        try:
            self.dataset = OfflineDataset(dataset_path)
            self._preview_frame_cache.clear()  # 删除后索引重排，缓存全部失效
            self._update_episode_table()
            self._update_info()
        except Exception as e:
            QMessageBox.warning(self, "重新加载警告", f"数据集重新加载失败: {e}\n请手动刷新。")

        # 恢复进度条
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0 / 0")

        # 恢复状态显示
        self.status_bar.showMessage(f"已批量删除 {len(indices)} 个 Episode")

        QMessageBox.information(self, "批量删除成功", result)

        # 重置选择
        self.current_episode_label.setText("当前未选择 Episode")

        # 恢复控件
        self._restore_ui_after_delete()

    def _on_delete_finished(self, result):
        """删除成功回调"""
        episode_idx = self._pending_episode_idx
        dataset_path = self._pending_dataset_path

        # 重新加载数据集
        self.status_bar.showMessage("正在重新加载数据集...")
        try:
            self.dataset = OfflineDataset(dataset_path)
            self._preview_frame_cache.clear()  # 删除后索引重排，缓存全部失效
            self._update_episode_table()
            self._update_info()
        except Exception as e:
            QMessageBox.warning(self, "重新加载警告", f"数据集重新加载失败: {e}\n请手动刷新。")

        # 恢复进度条
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0 / 0")

        # 恢复状态显示
        self.status_bar.showMessage(f"已删除 Episode {episode_idx}")

        QMessageBox.information(self, "删除成功", result)

        # 重置选择
        self.current_episode_label.setText("当前未选择 Episode")

        # 恢复控件
        self._restore_ui_after_delete()

    def _on_delete_error(self, error_msg):
        """删除失败回调"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0 / 0")
        self.status_bar.showMessage("删除失败")
        QMessageBox.critical(self, "删除失败", error_msg)

        self._restore_ui_after_delete()

    def _restore_ui_after_delete(self):
        """删除操作完成后恢复 UI 控件状态。"""
        self.play_btn.setEnabled(True)
        self.play_btn.setText("▶ 播放")
        self.play_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )
        self.episode_table.setEnabled(True)
        self.dataset_combo.setEnabled(True)
        self.trim_btn.setEnabled(False)
        self.trim_slider.set_total_frames(0)
        self.trim_slider.setEnabled(False)
        # 批量删除按钮只依赖"数据集已加载"，删除完成重新加载后恢复
        self.batch_del_btn.setEnabled(
            self.dataset is not None and len(self.dataset.episodes) > 0)
        # 剪辑静止按钮常驻于分析面板，删除完成后恢复
        self.trajectory_plot.trim_stationary_btn.setEnabled(True)
        self.trajectory_plot.trim_all_stationary_btn.setEnabled(True)
        # del_btn, pause_btn 保持禁用，因为没有选中 episode 或没有播放

    def closeEvent(self, event):
        # 如果任务名称有未保存修改，提示用户
        if self._check_unsaved_task_change():
            event.ignore()
            return
        self._stop_playback()
        # 停止关节反解后台线程并释放视频/3D 资源（先 cancel 再 wait，
        # 反解循环逐帧检查取消标志，通常毫秒级退出；退休线程一并清理）
        try:
            if self._ik_thread is not None and self._ik_thread.isRunning():
                self._ik_thread.cancel()
            for t in list(self._ik_retiring):
                t.cancel()
            if self._ik_thread is not None and self._ik_thread.isRunning():
                self._ik_thread.wait(2000)
            for t in list(self._ik_retiring):
                t.wait(2000)
            self._ik_retiring.clear()
        except Exception:
            pass
        if self.dataset is not None:
            try:
                self.dataset.release_video_caps()
            except Exception:
                pass
        event.accept()

    def _check_unsaved_task_change(self):
        """检查是否有未保存的任务名称修改，若有则提示并返回 True。"""
        if self._current_episode_idx < 0 or self.dataset is None:
            return False

        current_text = self.task_name_edit.text().strip()
        if current_text != self._original_task_name:
            reply = QMessageBox.question(
                self, "未保存的修改",
                f"任务名称已被修改，是否保存为 \"{current_text}\"？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._save_task_name(self._current_episode_idx, current_text)
            elif reply == QMessageBox.StandardButton.Cancel:
                return True
            # No: 放弃修改，继续
        return False


# ==================== 入口 ====================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="LeRobot 数据可视化工具")
    parser.add_argument("--root", type=str, default="./lerobot_data", help="数据集根目录")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = MainWindow(root=args.root)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

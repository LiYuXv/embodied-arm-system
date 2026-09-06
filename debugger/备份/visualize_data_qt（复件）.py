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
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSplitter, QMessageBox, QFrame, QHeaderView,
    QTableWidget, QTableWidgetItem, QGroupBox, QComboBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QStatusBar, QScrollArea, QSlider, QLineEdit,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition, QPoint, QPointF
from PyQt6.QtGui import QFont, QImage, QPixmap, QPainter, QBrush, QPen, QColor, QPolygonF


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
        
        # 缓存视频读取器: {(episode_index, camera_name): cap}
        self._video_caps = {}
        
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

        使用 imageio.v3 的 imiter 顺序读取。playback 通常是按帧顺序 0,1,2...
        调用，因此缓存 generator 可大幅提升速度。随机访问时重新打开并跳过。
        """
        import imageio.v3 as iio

        video_path = self.get_video_path(episode_index, camera_name)
        if video_path is None:
            return None

        key = (episode_index, camera_name)
        cached = self._video_caps.get(key)

        try:
            if cached is not None:
                gen, current_idx = cached
                if frame_index == current_idx + 1:
                    # 顺序读取：最快路径
                    frame = next(gen)
                    self._video_caps[key] = (gen, frame_index)
                    return frame
                elif frame_index == current_idx:
                    # 重复请求同一帧：不支持回退，重新打开
                    self._close_video_reader(key)
                # 否则跳帧或倒退，需要重新定位

            # 新建 generator 并跳到目标帧
            gen = iio.imiter(video_path)
            # 跳过前 frame_index 帧
            for _ in range(frame_index):
                next(gen)
            frame = next(gen)
            self._video_caps[key] = (gen, frame_index)
            return frame
        except StopIteration:
            # 帧索引超出范围
            return None
        except Exception as e:
            print(f"[OfflineDataset] 读取视频帧失败 ep={episode_index}, frame={frame_index}: {e}")
            # 失败后清理缓存，下次重试
            self._close_video_reader(key)
            return None

    def _close_video_reader(self, key):
        """关闭指定视频 reader"""
        cached = self._video_caps.pop(key, None)
        if cached is not None:
            gen, _ = cached
            try:
                gen.close()
            except Exception:
                pass

    def release_video_caps(self):
        """释放所有视频读取器"""
        for key in list(self._video_caps.keys()):
            self._close_video_reader(key)
        self._video_caps.clear()
    
    def is_video_feature(self, camera_name):
        """判断指定 camera 是否为 video 类型"""
        feature = self.features.get(camera_name, {})
        return feature.get("dtype") == "video" or feature.get("type") == "video"

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
        """加载所有帧数据"""
        data_files = sorted(self.data_dir.rglob("*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"找不到数据 parquet 文件: {self.data_dir}")
        return pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)
    
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
    """自定义三标记滑条：起止标记 + 可拖动播放头。"""
    markers_changed = pyqtSignal(int, int)
    position_changed = pyqtSignal(int)  # 播放头位置实时变化（拖动/点击）
    position_clicked = pyqtSignal(int)    # 点击轨道/释放播放头时触发

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(58)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._total = 0
        self._current = 0
        self._start = 0
        self._end = 0
        self._dragging = None  # 'start' / 'end' / 'playhead' / None

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

# ==================== 等比例缩放图像标签 ====================
class ImageLabel(QLabel):
    """固定 16:9（128:72）显示区域，保持原图比例缩放，避免画面被压缩或拉伸。"""
    
    ASPECT_W = 128
    ASPECT_H = 72
    
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._original = None       # 原始 numpy 图像 (H, W, C)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 固定 16:9 显示区域，高度根据宽度自动计算
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
    
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

        to_reindex_sorted = sorted(to_reindex, key=lambda x: x[1], reverse=True)
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

    data_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    if not data_files:
        raise FileNotFoundError(f"找不到数据 parquet：{data_dir}")
    df_all = pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)
    df_new = pd.concat([df_all.iloc[:from_idx], df_all.iloc[to_idx:]], ignore_index=True)

    if "episode_index" in df_new.columns:
        df_new.loc[df_new["episode_index"] > episode_idx, "episode_index"] -= 1
    if "index" in df_new.columns:
        df_new["index"] = range(len(df_new))

    _remove_files(data_files)
    chunk_dir = os.path.join(data_dir, "chunk-000")
    _write_parquet_chunks(df_new, chunk_dir)

    new_ep_df = ep_df.drop(index=episode_idx).reset_index(drop=True)
    if "dataset_from_index" in new_ep_df.columns and "dataset_to_index" in new_ep_df.columns:
        mask = new_ep_df["dataset_from_index"] >= to_idx
        new_ep_df.loc[mask, "dataset_from_index"] -= n_del
        new_ep_df.loc[mask, "dataset_to_index"]   -= n_del
    new_ep_df["episode_index"] = range(len(new_ep_df))

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

    deleted_vids, renamed_vids = _reindex_video_files(video_dir, episode_idx, num_after)

    result = (f"已删除 Episode {episode_idx}（{n_del} 帧）\n"
              f"剩余 {num_after} 个 episode，{len(df_new)} 帧")
    if deleted_vids:
        result += f"\n已删除视频: {len(deleted_vids)} 个"
    if renamed_vids:
        result += f"\n已重命名视频: {len(renamed_vids)} 个"
    return result


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


class TrimEpisodeThread(QThread):
    """在独立线程中执行 episode 片段剪辑。"""
    finished_ok = pyqtSignal(str, object, object, object)
    error = pyqtSignal(str)

    def __init__(self, dataset_root, episode_idx, start_frame, end_frame):
        super().__init__()
        self.dataset_root = dataset_root
        self.episode_idx = episode_idx
        self.start_frame = start_frame
        self.end_frame = end_frame

    def run(self):
        try:
            result, df_new, new_ep_df, info = trim_episode(
                self.dataset_root, self.episode_idx, self.start_frame, self.end_frame
            )
            self.finished_ok.emit(result, df_new, new_ep_df, info)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


def _find_video_files_for_trim(video_dir):
    """查找会被剪辑的视频文件路径列表（当前 episode 对应的 chunk-*/file-*.mp4）。"""
    if not os.path.exists(video_dir):
        return []
    files = []
    for cam in os.listdir(video_dir):
        cam_path = os.path.join(video_dir, cam)
        if not os.path.isdir(cam_path):
            continue
        candidates = glob.glob(os.path.join(cam_path, "**", "*.mp4"), recursive=True)
        candidates = [c for c in candidates if "chunk-" in c and "file-" in c]
        files.extend(candidates)
    return files


def _backup_for_trim(root, meta_dir, data_dir, video_dir, ep_files, video_files_to_modify):
    """仅备份会被修改的文件/目录，避免复制整个数据集导致的长等待。"""
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


def trim_episode(dataset_root, episode_idx, start_frame, end_frame):
    """删除 episode 内 [start_frame, end_frame] 闭区间的片段，保留其余部分，并重新编排索引。

    同时生成备份目录 meta.bak/、data.bak/、videos.bak/。
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

    video_files_to_modify = _find_video_files_for_trim(str(video_dir))

    # 仅备份会被修改的文件/目录，避免复制整个数据集
    _backup_for_trim(root, meta_dir, data_dir, video_dir, ep_files, video_files_to_modify)

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
        df_new.loc[df_new["episode_index"] > episode_idx, "episode_index"] -= 1
    if "index" in df_new.columns:
        df_new["index"] = range(len(df_new))

    _remove_files([str(p) for p in data_files])
    chunk_dir = data_dir / "chunk-000"
    _write_parquet_chunks(df_new, chunk_dir)

    # 更新当前 episode：长度变短
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

    # 如果有 episode_stats 也更新
    _update_episode_stats_for_trim(str(meta_dir), episode_idx, new_ep_len)

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
    _trim_video_files(str(video_dir), episode_idx, start_frame, end_frame)

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


def _trim_video_files(video_dir, episode_idx, start_frame, end_frame):
    """对 videos 目录下当前 episode 的视频进行片段删除，保留前后两段。

    优先使用 ffmpeg 直接处理，避免把整个视频解码到内存；未安装 ffmpeg 时回退到 imageio。
    删除范围为 [start_frame, end_frame] 闭区间（与 data 裁剪一致）。
    """
    import shutil
    import subprocess

    if not os.path.exists(video_dir):
        return

    has_ffmpeg = shutil.which("ffmpeg") is not None

    for cam in os.listdir(video_dir):
        cam_path = os.path.join(video_dir, cam)
        if not os.path.isdir(cam_path):
            continue
        # 支持 LeRobot 0.4.4 chunk-000/file-000.mp4 结构
        candidates = glob.glob(os.path.join(cam_path, "**", "*.mp4"), recursive=True)
        candidates = [c for c in candidates if "chunk-" in c and "file-" in c]

        for vf in candidates:
            tmp = vf + ".tmp.mp4"
            try:
                if has_ffmpeg:
                    # 删除 [start, end] 闭区间：between(n, start, end+1) 选中该区间的帧
                    select_expr = f"select='not(between(n\,{start_frame}\,{end_frame + 1}))',setpts=N/FRAME_RATE/TB"
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
        
        # 获取任务名称
        task_name = ""
        if self.dataset.tasks and task_index < len(self.dataset.tasks):
            task_name = self.dataset.tasks[task_index].get("task", "")
        
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


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self, root="./lerobot_data"):
        super().__init__()
        self.root = root
        self.dataset = None
        self.playback_thread = None
        self._current_total_frames = 0

        # 任务编辑状态
        self._current_episode_idx = -1
        self._original_task_name = ""
        self._task_modified = False
        self._task_edit_connected = False

        self.setWindowTitle("LeRobot 数据可视化工具")
        self.setGeometry(100, 100, 1400, 900)
        
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

        left_layout.addWidget(episode_group, stretch=4)

        # 任务名称编辑
        task_group = QGroupBox("Episode 任务")
        task_layout = QVBoxLayout(task_group)

        self.task_name_edit = QLineEdit()
        self.task_name_edit.setPlaceholderText("选择 Episode 后显示任务名称")
        self.task_name_edit.setEnabled(False)
        # 限制只能输入英文、数字、下划线、空格
        from PyQt6.QtGui import QRegularExpressionValidator
        from PyQt6.QtCore import QRegularExpression
        validator = QRegularExpressionValidator(
            QRegularExpression("^[a-zA-Z0-9_][a-zA-Z0-9_ ]*$"), self.task_name_edit
        )
        self.task_name_edit.setValidator(validator)
        task_layout.addWidget(self.task_name_edit)

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
        self.trim_slider.markers_changed.connect(self._on_trim_markers_changed)
        self.trim_slider.position_changed.connect(self._on_trim_position_changed)
        self.trim_slider.position_clicked.connect(self._on_trim_position_clicked)
        control_layout.addWidget(self.trim_slider)
        
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
        self.play_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.play_btn.clicked.connect(self._play_selected_episode)
        btn_layout.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }")
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.pause_btn.setEnabled(False)
        btn_layout.addWidget(self.pause_btn)
        
        self.replay_btn = QPushButton("🔄 重新播放")
        self.replay_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }")
        self.replay_btn.clicked.connect(self._replay_episode)
        self.replay_btn.setEnabled(False)
        btn_layout.addWidget(self.replay_btn)
        
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px; }")
        self.stop_btn.clicked.connect(self._stop_playback)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
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
        
        left_layout.addWidget(control_group)
        left_layout.addStretch()
        
        splitter.addWidget(left_panel)
        
        # ===== 右侧面板: 图像显示 =====
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
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
        
        self.frame_info_label = QLabel("帧信息: 未开始播放")
        self.frame_info_label.setStyleSheet("QLabel { color: #666; font-size: 11px; padding: 5px; }")
        self.frame_info_label.setWordWrap(True)
        self.frame_info_label.setMinimumHeight(80)
        image_layout.addWidget(self.frame_info_label)
        
        right_layout.addWidget(image_group, stretch=3)
        
        # 操作说明
        help_group = QGroupBox("操作说明")
        help_layout = QVBoxLayout(help_group)
        
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
            "<b>其他功能:</b><br>"
            "• 双击 Episode 行可直接播放<br>"
            "• 勾选'自动播放所有'可连续播放<br>"
            "• 调整速度滑块改变回放速度<br><br>"
            "<b>视频剪辑:</b><br>"
            "• 拖动进度条下方蓝色/紫色三角选择要删除的片段<br>"
            "• 绿色播放头可拖动，实时显示对应帧画面与数据<br>"
            "• 点击轨道空白处可跳转当前帧<br>"
            "• 播放中拖动/点击播放头会自动暂停并定位到该帧<br>"
            "• 点击 ✂ 剪辑按钮删除两个三角标记之间的片段"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("QLabel { line-height: 1.5; }")
        help_layout.addWidget(help_text)
        
        right_layout.addWidget(help_group, stretch=1)
        right_layout.addStretch()
        
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 500])
        
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
            
    def _on_dataset_selected(self, index):
        if index <= 0:
            self.episode_table.setRowCount(0)
            self.info_label.setText("未加载数据集")
            self.episode_stats_label.setText("请选择数据集")
            self.current_episode_label.setText("当前未选择 Episode")
            self.del_btn.setEnabled(False)
            self.trim_btn.setEnabled(False)
            self.trim_slider.set_total_frames(0)
            self.dataset = None

            # 清空任务编辑
            self._reset_task_editor()
            return
            
        dataset_path = self.dataset_combo.itemData(index)
        if dataset_path is None:
            return
            
        try:
            self.status_bar.showMessage(f"正在加载数据集...")
            QApplication.processEvents()
            
            self.dataset = OfflineDataset(dataset_path)
            
            self._update_episode_table()
            self._update_info()
            
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
                # 旧格式：从 task_index 查找
                task_idx = ep["task_index"]
                if self.dataset.tasks and task_idx < len(self.dataset.tasks):
                    task_name = str(self.dataset.tasks[task_idx].get("task", ""))
            
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
        
        # 添加任务统计
        if task_stats:
            info_text += "<br><b>任务分布:</b><br>"
            for task_name, frame_count, frame_ratio, ep_count, ep_ratio in task_stats:
                info_text += f"  • {task_name}: {frame_count} 帧 ({frame_ratio:.1%}), {ep_count} 个 episode ({ep_ratio:.1%})<br>"
        
        self.info_label.setText(info_text)
    
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
            task_name = ""
            if task_idx < len(self.dataset.tasks):
                task_name = self.dataset.tasks[task_idx].get('task', f'task_{task_idx}')
            
            frame_ratio = frame_count / total_frames if total_frames > 0 else 0
            ep_count = task_episode_counts.get(task_name, 0)
            ep_ratio = ep_count / total_episodes if total_episodes > 0 else 0
            
            stats.append((task_name, frame_count, frame_ratio, ep_count, ep_ratio))
        
        return stats
        
    def _on_trim_markers_changed(self, start, end):
        """当双标记位置改变时更新剪辑信息。"""
        self._update_trim_info()

    def _on_trim_position_changed(self, frame):
        """播放头拖动时实时显示对应帧画面和信息；播放中拖动则自动暂停。"""
        if self.playback_thread is not None:
            if not self.playback_thread.is_paused():
                self.playback_thread.pause()
                self.pause_btn.setText("▶ 继续")
                # 等待线程真正进入暂停
                import time
                for _ in range(20):
                    if self.playback_thread.is_paused():
                        break
                    time.sleep(0.01)
                    QApplication.processEvents()
        # 未播放/已暂停时：通过预览方式实时显示对应帧
        self._preview_frame_at(frame)
        # 同步 trim_slider 的当前位置与进度标签
        self.trim_slider.set_current_frame(frame)
        if self._current_total_frames > 0:
            self.progress_label.setText(f"{frame + 1} / {self._current_total_frames}")

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
            task_name = ""
            if self.dataset.tasks and task_index < len(self.dataset.tasks):
                task_name = self.dataset.tasks[task_index].get("task", "")
            self._display_frame(cam1, cam2, action, state, frame_idx, timestamp, task_name)
            self.progress_label.setText(f"{frame_idx + 1} / {total}")
        except Exception as e:
            print(f"[Preview] 跳转帧失败: {e}")

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
            "删除后当前 episode 会被更新，原始数据备份到 .bak 目录。".format(
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
            dataset_path, self._current_episode_idx, start, end
        )
        self.trim_thread.finished_ok.connect(self._on_trim_finished)
        self.trim_thread.error.connect(self._on_trim_error)
        self.trim_thread.start()

    def _set_controls_enabled(self, enabled):
        """启用/禁用操作控件。"""
        self.play_btn.setEnabled(enabled)
        self.replay_btn.setEnabled(enabled)
        self.pause_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.del_btn.setEnabled(enabled)
        self.trim_btn.setEnabled(enabled)
        self.episode_table.setEnabled(enabled)
        self.dataset_combo.setEnabled(enabled)

    def _on_trim_finished(self, result, df_new, new_ep_df, info):
        """剪辑成功回调。"""
        episode_idx = self._pending_episode_idx

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
        self.replay_btn.setEnabled(True)

        self.status_bar.showMessage(f"已剪辑 Episode {episode_idx}")
        QMessageBox.information(self, "剪辑成功", result)

    def _on_trim_error(self, error_msg):
        """剪辑失败回调。"""
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
            self.replay_btn.setEnabled(True)
            self.del_btn.setEnabled(True)
            self.trim_btn.setEnabled(True)

            # 自动加载并显示第一帧的图像和数据
            self._load_first_frame(episode_idx)

            # 设置任务编辑器
            self._setup_task_editor(episode_idx)

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

        # 如果为空，再尝试从 task_index 查找
        if not task_name and "task_index" in ep:
            task_idx = ep["task_index"]
            if self.dataset.tasks and task_idx < len(self.dataset.tasks):
                task_name = str(self.dataset.tasks[task_idx].get("task", "")).strip()

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
        self._current_episode_idx = -1

    def _on_task_editing_finished(self):
        """任务名称编辑完成（按 Enter 或失去焦点）时触发。"""
        if self._current_episode_idx < 0 or self.dataset is None:
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

        # 提示是否保存
        reply = QMessageBox.question(
            self, "保存任务名称",
            f"是否将任务名称从 \"{self._original_task_name}\" 保存为 \"{new_name}\"？\n\n"
            f"注意：这会更新所有共享该任务索引的 episode。",
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

    def _save_task_name(self, episode_idx, new_name):
        """保存任务名称。每个 episode 独立命名：如果新名称已存在则复用其 task_index，否则创建新的 task_index。"""
        if not self.dataset or episode_idx < 0 or episode_idx >= len(self.dataset.episodes):
            return

        ep = self.dataset.episodes[episode_idx]
        old_task_index = ep.get("task_index", 0)
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

        # 如果 task_index 没变，说明改回了原名称（理论上不会走到这里）
        if new_task_index == old_task_index:
            self._original_task_name = new_name
            self._task_modified = False
            self._update_episode_table()
            self._update_info()
            return

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

                for p in data_files:
                    os.remove(p)
                chunk_dir = data_dir / "chunk-000"
                os.makedirs(chunk_dir, exist_ok=True)
                for i, start in enumerate(range(0, max(len(df_all), 1), 1000)):
                    df_all.iloc[start:start + 1000].to_parquet(
                        chunk_dir / f"file-{i:03d}.parquet", index=False)

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

    def _play_selected_episode(self):
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
        
    def _replay_episode(self):
        if self.dataset is None:
            return
        selected = self.episode_table.selectedItems()
        if not selected:
            return
        episode_idx = int(selected[0].text())
        self._start_playback(episode_idx, self.auto_play_check.isChecked())
        
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
        
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("⏸ 暂停")
        self.replay_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.auto_play_check.setEnabled(False)
        self.status_bar.showMessage("回放中")
        
    def _load_first_frame(self, episode_idx):
        """加载并显示指定 episode 的第一帧图像和数据（不启动播放线程）"""
        try:
            ep_frames = self.dataset.get_episode_frames(episode_idx)
            total_frames = len(ep_frames)
            if total_frames == 0:
                return
            
            # 获取第一帧数据
            row = ep_frames.iloc[0]
            
            # 加载图像
            cam1_img = self._load_image_for_preview(episode_idx, 0, "observation.images.cam1")
            cam2_img = self._load_image_for_preview(episode_idx, 0, "observation.images.cam2")
            
            # 获取 action 和 state
            action = np.array(row.get("action", [0] * 7))
            state = np.array(row.get("observation.state", [0] * 7))
            timestamp = row.get("timestamp", 0.0)
            task_index = int(row.get("task_index", 0))
            
            # 获取任务名称
            task_name = ""
            if self.dataset.tasks and task_index < len(self.dataset.tasks):
                task_name = self.dataset.tasks[task_index].get("task", "")
            
            # 更新显示
            self._display_frame(cam1_img, cam2_img, action, state, 0, timestamp, task_name)
            
            # 更新进度条为初始状态
            self._current_total_frames = total_frames
            self.trim_slider.set_total_frames(total_frames)
            self.trim_slider.set_current_frame(0)
            self.trim_slider.setEnabled(True)
            self.progress_label.setText(f"1 / {total_frames}")
            self._update_trim_info()
            
        except Exception as e:
            print(f"[Preview] 加载第一帧失败: {e}")
            
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
        
    def _display_frame(self, rgb_cam1, rgb_cam2, action, state, step, timestamp, task_name):
        """显示单帧图像和数据（不依赖播放线程）"""
        self.cam1_label.setImage(rgb_cam1)
        self.cam2_label.setImage(rgb_cam2)
        
        # 提取位置和四元数
        pos = state[:3]
        quat = state[3:7]
        
        # 更新帧信息标签（包含机械臂数据）
        info_text = (
            f"帧: {step} | 时间: {timestamp:.3f}s | 任务: {task_name}\n"
            f"位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n"
            f"四元数: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]"
        )
        self.frame_info_label.setText(info_text)
        
    def _on_frame_data(self, rgb_cam1, rgb_cam2, action, state, step, timestamp, task_name):
        self.cam1_label.setImage(rgb_cam1)
        self.cam2_label.setImage(rgb_cam2)
        # 播放线程推送新帧时同步播放头位置
        if self._current_total_frames > 0:
            self.trim_slider.set_current_frame(step)
        
        # 提取位置、四元数和夹爪角度 (state 和 action 都是 [x, y, z, qx, qy, qz, qw, gripper_deg])
        pos = state[:3]
        quat = state[3:7]
        # 第8维是夹爪角度（度），如果数据长度不足则默认为0
        gripper_deg = float(state[7]) if len(state) > 7 else 0.0
        
        # 更新帧信息标签（包含机械臂数据）
        info_text = (
            f"帧: {step} | 时间: {timestamp:.3f}s | 任务: {task_name}\n"
            f"位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n"
            f"四元数: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]\n"
            f"夹爪: {gripper_deg:.1f}°"
        )
        self.frame_info_label.setText(info_text)
        
    def _update_data_panel(self, step, timestamp, pos, quat, action, task_name=""):
        """更新机械臂数据面板显示（已删除面板，方法保留兼容）"""
        pass
        
    def _on_frame_updated(self, current, total):
        self._current_total_frames = total
        if total > 0:
            self.trim_slider.set_current_frame(current - 1)
            self.progress_label.setText(f"{current} / {total}")
            
    def _on_episode_finished(self):
        self.trim_slider.set_current_frame(0)
        
    def _on_playback_finished(self):
        self._reset_ui_state()
        self.status_bar.showMessage("播放完成")
        
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
            
        self._reset_ui_state()
        
    def _reset_ui_state(self):
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ 暂停")
        self.replay_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
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
        self.replay_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
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

    def _on_delete_finished(self, result):
        """删除成功回调"""
        episode_idx = self._pending_episode_idx
        dataset_path = self._pending_dataset_path

        # 重新加载数据集
        self.status_bar.showMessage("正在重新加载数据集...")
        try:
            self.dataset = OfflineDataset(dataset_path)
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
        self.replay_btn.setEnabled(False)

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
        self.episode_table.setEnabled(True)
        self.dataset_combo.setEnabled(True)
        self.trim_btn.setEnabled(False)
        self.trim_slider.set_total_frames(0)
        self.trim_slider.setEnabled(False)
        # del_btn, replay_btn, pause_btn, stop_btn 保持禁用，因为没有选中 episode 或没有播放

    def closeEvent(self, event):
        # 如果任务名称有未保存修改，提示用户
        if self._check_unsaved_task_change():
            event.ignore()
            return
        self._stop_playback()
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

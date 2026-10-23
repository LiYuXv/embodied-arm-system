# -*- coding: utf-8 -*-
"""后台工作线程：删除/剪辑/降噪/IK 反解/回放/VLM 标签等 QThread 工作线程。"""
from vd_common import solve_episode_joint_angles

import os
import time
import threading
import numpy as np
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

from vd_dataset import OfflineDataset
from vd_data_ops import delete_episode, delete_episodes, denoise_dataset, trim_episode, trim_episode_ranges


# ==================== 删除后台线程 ====================
class DeleteEpisodeThread(QThread):
    """在独立线程中执行 episode 删除，避免 UI 卡顿。"""
    finished_ok = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, dataset_root, episode_idx, backup_enabled=True):
        super().__init__()
        self.dataset_root = dataset_root
        self.episode_idx = episode_idx
        self.backup_enabled = backup_enabled

    def run(self):
        try:
            result = delete_episode(self.dataset_root, self.episode_idx,
                                    backup_enabled=self.backup_enabled)
            self.finished_ok.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")

class BatchDeleteEpisodesThread(QThread):
    """在独立线程中批量删除 episode（单次重写），避免 UI 卡顿。"""
    finished_ok = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, dataset_root, episode_indices, backup_enabled=True):
        super().__init__()
        self.dataset_root = dataset_root
        self.episode_indices = list(episode_indices)
        self.backup_enabled = backup_enabled

    def run(self):
        try:
            result = delete_episodes(self.dataset_root, self.episode_indices,
                                     backup_enabled=self.backup_enabled)
            self.finished_ok.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")

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


# ==================== 回放线程 ====================
class PlaybackThread(QThread):
    """真机数据纯图像回放线程，支持暂停/继续"""
    frame_data = pyqtSignal(object, object, object, object, object, int, object, object)  # cam1_rgb, cam2_rgb, cam3_rgb, action, state, step, timestamp, task_name
    frame_updated = pyqtSignal(int, int)  # current_frame, total_frames
    episode_finished = pyqtSignal()
    playback_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    
    def __init__(self, dataset: OfflineDataset, episode_index: int, speed=1.0,
                 auto_play=False, start_frame=0):
        super().__init__()
        self.dataset = dataset
        self.episode_index = episode_index
        self.speed = speed
        self.auto_play = auto_play
        # 起始帧（"继续"按键从滑条当前位置起播；0=从头，clamp 到集长内）
        self.start_frame = max(0, int(start_frame))
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
        
        # 获取图像（无 cam3 的旧数据集返回 None，UI 侧显示占位）
        cam1_img = self._load_image(ep_idx, step, "observation.images.cam1")
        cam2_img = self._load_image(ep_idx, step, "observation.images.cam2")
        cam3_img = self._load_image(ep_idx, step, "observation.images.cam3")

        # 获取 action 和 state（默认维度取数据集声明的 observation.state 维度）
        state_dim = self.dataset.features.get("observation.state", {}).get("shape", [18])[0]
        action = np.array(row.get("action", [0] * state_dim))
        state = np.array(row.get("observation.state", [0] * state_dim))
        timestamp = row.get("timestamp", step / self.dataset.fps)
        task_index = int(row.get("task_index", 0))

        # 获取任务名称（按 task_index 值查找，不能用列表位置索引）
        task_name = self.dataset.task_name_for_index(task_index)

        # 发送帧数据
        self.frame_data.emit(cam1_img, cam2_img, cam3_img, action, state, step, timestamp, task_name)
        self.frame_updated.emit(step + 1, total_frames)
        
    def run(self):
        try:
            episodes_to_play = range(len(self.dataset.episodes)) if self.auto_play else [self.episode_index]
            
            pending_start = self.start_frame  # 仅第一集从 start_frame 起播（连播后续集从头）
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
                
                step = min(pending_start, total_frames - 1) if total_frames > 0 else 0
                pending_start = 0
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
                    # 唤醒后若已有待处理跳转（"继续"键：先 jump 后 resume），
                    # 先回到循环顶处理跳转，避免先吐出旧序列的下一帧
                    if self._jump_to_frame is not None:
                        continue
                    
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


# ==================== RECAP sidecar 展开线程 ====================
class SidecarExportThread(QThread):
    """后台展开 RECAP 标注为逐帧 sidecar（make_recap_sidecar.expand_dataset），
    避免读取 parquet 列时卡住 UI。"""
    finished_ok = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, dataset_root, bad_backprop=0, terminal_fail=0, parent=None):
        super().__init__(parent)
        self.dataset_root = dataset_root
        self.bad_backprop = bad_backprop
        self.terminal_fail = terminal_fail

    def run(self):
        try:
            from make_recap_sidecar import expand_dataset
            result = expand_dataset(
                self.dataset_root,
                bad_backprop=self.bad_backprop,
                terminal_fail=self.terminal_fail,
            )
            self.finished_ok.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")

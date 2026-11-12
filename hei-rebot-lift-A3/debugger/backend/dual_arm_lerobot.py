"""Dual-arm LeRobot recording primitives.

The legacy single-arm recorder remains in ``camera_pyqt.py``.  This module
adds the explicit joint-space dual-arm schema and keeps the same lookahead
label rule: ``action[t] = observation.state[t + N]``.
"""

import inspect
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger("debugger")

DUAL_ARM_FPS = 20
DUAL_ARM_CAMERA_KEYS = ("cam1", "cam2", "cam3")
DUAL_ARM_IMAGE_KEYS = tuple(
    f"observation.images.{name}" for name in DUAL_ARM_CAMERA_KEYS
)
DUAL_ARM_STATE_NAMES = tuple(
    [f"left_j{i}" for i in range(1, 7)]
    + ["left_gripper"]
    + [f"right_j{i}" for i in range(1, 7)]
    + ["right_gripper"]
)
DUAL_ARM_TORQUE_NAMES = ("left_gripper", "right_gripper")
MOBILE_STATE_NAMES = ("lift_m", "vx", "vy", "omega")


def make_dual_arm_features(image_size=(480, 640), *, include_mobile=False):
    """Return the canonical three-camera, joint14 dataset feature schema."""
    height, width = (int(image_size[0]), int(image_size[1]))
    image_feature = {
        "dtype": "video",
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }
    state_names = DUAL_ARM_STATE_NAMES + (MOBILE_STATE_NAMES if include_mobile else ())
    return {
        key: dict(image_feature) for key in DUAL_ARM_IMAGE_KEYS
    } | {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": list(state_names),
        },
        "observation.gripper_torque": {
            "dtype": "float32",
            "shape": (2,),
            "names": list(DUAL_ARM_TORQUE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": list(state_names),
        },
    }


def make_single_arm_features(image_size=(480, 640)):
    """Legacy ELA3 Teacher Cartesian pose7 + one torque + wrist/head videos."""
    features = make_dual_arm_features(image_size)
    del features["observation.images.cam3"]
    features["observation.state"].update(shape=(7,), names=["state"])
    features["action"].update(shape=(7,), names=["action"])
    features["observation.gripper_torque"].update(shape=(1,), names=["torque"])
    return features


def _float_vector(value, size, field):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{field} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must contain finite values")
    return np.array(array, dtype=np.float32, copy=True)


def build_dual_arm_frame(images, state, action, gripper_torque, task, *, include_mobile=False, single_arm=False):
    """Build one validated frame using the fixed public key ordering."""
    camera_keys = DUAL_ARM_CAMERA_KEYS[:2] if single_arm else DUAL_ARM_CAMERA_KEYS
    missing = [name for name in camera_keys if images.get(name) is None]
    if missing:
        raise ValueError(f"missing camera frames: {', '.join(missing)}")
    dim = 7 if single_arm else (18 if include_mobile else 14)
    return {
        **{
            f"observation.images.{name}": images[name]
            for name in camera_keys
        },
        "observation.state": _float_vector(state, dim, "observation.state"),
        "observation.gripper_torque": _float_vector(
            gripper_torque, 1 if single_arm else 2, "observation.gripper_torque"
        ),
        "action": _float_vector(action, dim, "action"),
        "task": str(task),
    }


def features_are_compatible(existing, expected):
    """Check only storage-defining fields needed for safe episode append."""
    observation_keys = {key for key in existing if key.startswith("observation.") or key == "action"}
    if observation_keys != set(expected):
        return False
    for key, target in expected.items():
        current = existing.get(key)
        if not isinstance(current, dict):
            return False
        if current.get("dtype") != target["dtype"]:
            return False
        if not isinstance(current.get("shape"), (list, tuple)) or tuple(current["shape"]) != tuple(target["shape"]):
            return False
        if not isinstance(current.get("names"), (list, tuple)) or list(current["names"]) != list(target["names"]):
            return False
        if target["dtype"] == "video":
            codec = current.get("info", {}).get("video.codec")
            if codec not in (None, "", "h264"):
                return False
    return True


def _resolve_vcodec():
    try:
        import av
        for codec in ("h264_nvenc", "hevc_nvenc"):
            try:
                av.codec.Codec(codec, "w")
                return codec
            except Exception:
                continue
    except Exception:
        pass
    return "h264"


def _codec_kwargs(method, vcodec):
    params = inspect.signature(method).parameters
    if "rgb_encoder" in params:
        from lerobot.configs import RGBEncoderConfig
        config = (
            RGBEncoderConfig(vcodec=vcodec, g=30)
            if "nvenc" in vcodec
            else RGBEncoderConfig(vcodec=vcodec)
        )
        return {"rgb_encoder": config}
    return {"vcodec": vcodec}


def create_or_resume_dual_arm_dataset(
    dataset_root,
    repo_name,
    *,
    image_size=(480, 640),
    fps=DUAL_ARM_FPS,
    include_mobile=False,
    dataset_cls=None,
):
    """Create a joint14 dataset or resume an exactly compatible one."""
    if float(fps) != DUAL_ARM_FPS:
        raise ValueError(f"dual-arm recording fps must remain {DUAL_ARM_FPS}")
    if not repo_name or Path(repo_name).name != repo_name:
        raise ValueError("repo_name must be one non-empty folder name")
    if dataset_cls is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset_cls = LeRobotDataset

    root = Path(dataset_root).expanduser().resolve()
    dataset_dir = root / repo_name
    info_path = dataset_dir / "meta" / "info.json"
    expected = make_dual_arm_features(image_size, include_mobile=include_mobile)
    robot_type = "ela3_dual_arm_mobile" if include_mobile else "ela3_dual_arm"
    if dataset_dir.exists() and not info_path.exists():
        raise ValueError(f"existing folder is not a LeRobot dataset: {dataset_dir}")

    vcodec = _resolve_vcodec()
    encoder_threads = None if "nvenc" in vcodec else 4
    old_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        if info_path.exists():
            with info_path.open("r", encoding="utf-8") as handle:
                info = json.load(handle)
            if int(info.get("fps", -1)) != DUAL_ARM_FPS:
                raise ValueError(
                    f"existing dataset fps must be {DUAL_ARM_FPS}: {dataset_dir}"
                )
            existing = info.get("features", {})
            if info.get("robot_type") != robot_type or not features_are_compatible(existing, expected):
                raise ValueError(
                    f"existing dataset schema is incompatible: {dataset_dir}"
                )
            if hasattr(dataset_cls, "resume"):
                return dataset_cls.resume(
                    repo_name,
                    root=dataset_dir,
                    streaming_encoding=True,
                    encoder_threads=encoder_threads,
                    **_codec_kwargs(dataset_cls.resume, vcodec),
                )
            return dataset_cls(
                repo_name,
                root=dataset_dir,
                streaming_encoding=True,
                encoder_threads=encoder_threads,
                **_codec_kwargs(dataset_cls, vcodec),
            )

        return dataset_cls.create(
            repo_id=repo_name,
            fps=DUAL_ARM_FPS,
            features=expected,
            root=dataset_dir,
            robot_type=robot_type,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4,
            encoder_threads=encoder_threads,
            streaming_encoding=True,
            **_codec_kwargs(dataset_cls.create, vcodec),
        )
    finally:
        if old_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_offline


class DualArmLeRobotRecordThread(QThread):
    """Queue-backed 20 Hz writer preserving the legacy lookahead semantics."""

    frame_written = pyqtSignal(int)
    status = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, dataset, *, state_callback=None, torque_callback=None,
                 fps=DUAL_ARM_FPS, max_queue=5, action_lookahead=5,
                 include_mobile=False, single_arm=False,
                 parent=None):
        super().__init__(parent)
        if float(fps) != DUAL_ARM_FPS:
            raise ValueError(f"dual-arm recording fps must remain {DUAL_ARM_FPS}")
        self.dataset = dataset
        self.single_arm = bool(single_arm)
        self.include_mobile = bool(include_mobile)
        if self.single_arm and self.include_mobile:
            raise ValueError("single-arm recording does not include mobile base")
        self.state_callback = state_callback
        self.torque_callback = torque_callback
        self.fps = DUAL_ARM_FPS
        self.action_lookahead = max(1, int(action_lookahead))
        self._queue = Queue(maxsize=max(1, int(max_queue)))
        self._running = True
        self._frame_count = 0
        self._task_name = ""

    @property
    def frame_count(self):
        return self._frame_count

    def set_task_name(self, name):
        self._task_name = str(name)

    def push_frame(self, images, state=None, gripper_torque=None):
        """Capture images and both arms' current observations atomically."""
        if state is None and self.state_callback is not None:
            state = self.state_callback()
        if gripper_torque is None and self.torque_callback is not None:
            gripper_torque = self.torque_callback()
        if state is None:
            raise ValueError("dual-arm joint feedback is not ready")
        single = getattr(self, "single_arm", False)
        if gripper_torque is None:
            gripper_torque = np.zeros(1 if single else 2, dtype=np.float32)

        item = {
            "images": {name: images.get(name) for name in
                       (DUAL_ARM_CAMERA_KEYS[:2] if single else DUAL_ARM_CAMERA_KEYS)},
            "state": _float_vector(state, 7 if single else (18 if self.include_mobile else 14), "observation.state"),
            "gripper_torque": _float_vector(
                gripper_torque, 1 if single else 2, "gripper_torque"
            ),
            "task": self._task_name,
        }
        missing = [name for name, frame in item["images"].items() if frame is None]
        if missing:
            raise ValueError(f"missing camera frames: {', '.join(missing)}")
        try:
            self._queue.put_nowait(item)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            self._queue.put_nowait(item)
        return True

    def stop(self, flush=False, max_remaining=0):
        self._running = False
        if not flush:
            while True:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
        else:
            while self._queue.qsize() > max(0, int(max_remaining)):
                try:
                    self._queue.get_nowait()
                except Empty:
                    break

    def run(self):
        try:
            interval = 1.0 / DUAL_ARM_FPS
            lookahead = self.action_lookahead
            buffer = deque()

            def write(obs_item, action_state):
                frame = build_dual_arm_frame(
                    obs_item["images"],
                    obs_item["state"],
                    action_state,
                    obs_item["gripper_torque"],
                    obs_item["task"],
                    include_mobile=self.include_mobile,
                    single_arm=getattr(self, "single_arm", False),
                )
                self.dataset.add_frame(frame)
                self._frame_count += 1
                self.frame_written.emit(self._frame_count)

            while self._running or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                started = time.perf_counter()
                buffer.append(item)
                if len(buffer) > lookahead:
                    observation = buffer.popleft()
                    write(observation, buffer[lookahead - 1]["state"])
                elapsed = time.perf_counter() - started
                time.sleep(max(0.0, interval - elapsed))

            for index, observation in enumerate(buffer):
                action_index = min(index + lookahead, len(buffer) - 1)
                write(observation, buffer[action_index]["state"])

            if self._frame_count:
                self.dataset.save_episode(parallel_encoding=True)
                if hasattr(type(self.dataset), "resume"):
                    self.dataset.finalize()
                self.finished_ok.emit(f"已保存 {self._frame_count} 帧")
            else:
                self.finished_ok.emit("没有录制到帧")
        except Exception as exc:
            logger.exception("[DualArmLeRobot] 写入或保存失败")
            self.error.emit(str(exc))

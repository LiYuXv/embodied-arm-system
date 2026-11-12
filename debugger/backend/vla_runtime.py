"""Backend-neutral VLA runtime contracts and the local SmolVLA backend."""

from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from debugger.vla_features import (
    ACTION_KEY,
    DUAL_ARM_SPEC,
    as_action_chunk,
    build_vla_observation,
    prepare_observation_for_policy,
    validate_policy_checkpoint,
)


@dataclass(frozen=True)
class VLAObservationSnapshot:
    """Immutable-by-contract copy of one GUI-thread observation."""

    frames: tuple[np.ndarray, np.ndarray, np.ndarray]
    state: np.ndarray
    gripper_torque: np.ndarray
    task: str

    @classmethod
    def capture(cls, images, state, gripper_torque, task):
        """Validate and copy live values before they cross into the worker thread."""
        observation = build_vla_observation(
            images, state, gripper_torque, task, mode="dual"
        )
        frames = tuple(
            np.array(observation[key], copy=True)
            for key in DUAL_ARM_SPEC.camera_keys
        )
        state_copy = np.array(observation["observation.state"], copy=True)
        torque_copy = np.array(
            observation["observation.gripper_torque"], copy=True
        )
        for value in (*frames, state_copy, torque_copy):
            value.setflags(write=False)
        return cls(
            frames=frames,
            state=state_copy,
            gripper_torque=torque_copy,
            task=str(observation["task"]),
        )

    @property
    def frame_shapes(self):
        return tuple(tuple(frame.shape) for frame in self.frames)

    def to_observation(self):
        """Return worker-owned copies so preprocessors may safely mutate inputs."""
        images = {
            name: np.array(frame, copy=True)
            for name, frame in zip(DUAL_ARM_SPEC.camera_names, self.frames)
        }
        return build_vla_observation(
            images,
            np.array(self.state, copy=True),
            np.array(self.gripper_torque, copy=True),
            self.task,
            mode="dual",
        )


@dataclass(frozen=True)
class VLABackendInfo:
    checkpoint: str
    policy_type: str
    device: str
    chunk_size: int
    n_action_steps: int


@dataclass(frozen=True)
class VLAInferenceResult:
    action_chunk: np.ndarray
    latency_ms: float


class VLAInferenceBackend(ABC):
    """Interface shared by local inference and a future cloud backend."""

    @abstractmethod
    def load(self, checkpoint_path: str) -> VLABackendInfo:
        raise NotImplementedError

    @abstractmethod
    def infer(self, snapshot: VLAObservationSnapshot) -> VLAInferenceResult:
        raise NotImplementedError

    def close(self):
        """Release backend resources."""


class LocalSmolVLABackend(VLAInferenceBackend):
    """Offline-only local LeRobot/SmolVLA checkpoint runtime."""

    def __init__(self, *, device: str | None = None, bgr_images: bool = True):
        self.requested_device = device
        self.bgr_images = bool(bgr_images)
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.config = None
        self.info = None
        self._torch = None

    @staticmethod
    def _resolve_checkpoint(checkpoint_path):
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if (checkpoint / "pretrained_model").is_dir():
            checkpoint = checkpoint / "pretrained_model"
        required = (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"invalid local VLA checkpoint {checkpoint}: missing {missing}"
            )
        return checkpoint

    def load(self, checkpoint_path: str) -> VLABackendInfo:
        """Load weights and processors strictly from a local checkpoint."""
        self.close()
        checkpoint = self._resolve_checkpoint(checkpoint_path)
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class
            from lerobot.processor import DataProcessorPipeline

            # Importing the configuration registers the custom policy type;
            # the factory then discovers debugger.modeling_smolvla_torque.
            from debugger import configuration_smolvla_torque  # noqa: F401
            from debugger import processor_smolvla_torque  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "local VLA runtime requires the VLA environment with torch and lerobot"
            ) from exc

        device = self.requested_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        config = PreTrainedConfig.from_pretrained(
            checkpoint, local_files_only=True
        )
        validate_policy_checkpoint(config, "dual")
        policy_type = str(getattr(config, "type", "") or "").lower()
        if policy_type != "smolvla_torque":
            raise ValueError(
                "local dual-arm backend currently requires policy type "
                f"'smolvla_torque', got '{policy_type}'"
            )
        config.device = device

        policy_class = get_policy_class(policy_type)
        policy = policy_class.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
        )
        policy.to(device)
        policy.eval()

        device_override = {"device_processor": {"device": device}}
        preprocessor = DataProcessorPipeline.from_pretrained(
            checkpoint,
            config_filename="policy_preprocessor.json",
            local_files_only=True,
            overrides=device_override,
        )
        postprocessor = DataProcessorPipeline.from_pretrained(
            checkpoint,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
        )

        self._torch = torch
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.config = config
        self.info = VLABackendInfo(
            checkpoint=str(checkpoint),
            policy_type=policy_type,
            device=str(device),
            chunk_size=int(getattr(config, "chunk_size", 0) or 0),
            n_action_steps=int(getattr(config, "n_action_steps", 0) or 0),
        )
        if self.info.n_action_steps <= 0:
            self.close()
            raise ValueError("checkpoint config.n_action_steps must be positive")
        return self.info

    def infer(self, snapshot: VLAObservationSnapshot) -> VLAInferenceResult:
        if self.policy is None or self.config is None or self._torch is None:
            raise RuntimeError("VLA model is not loaded")
        if not isinstance(snapshot, VLAObservationSnapshot):
            raise TypeError("inference requires a VLAObservationSnapshot")

        torch = self._torch
        observation = snapshot.to_observation()
        batch = prepare_observation_for_policy(
            observation,
            self.config,
            mode="dual",
            device="cpu",
            bgr_images=self.bgr_images,
        )
        if str(self.info.device).startswith("cuda"):
            torch.cuda.synchronize(self.info.device)
        started = time.perf_counter()
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        with torch.inference_mode():
            processed = self.preprocessor(batch)
            action = self.policy.predict_action_chunk(processed)
            processed_action = self.postprocessor(
                action if isinstance(action, Mapping) else {ACTION_KEY: action}
            )
        chunk = as_action_chunk(processed_action, mode="dual")
        if str(self.info.device).startswith("cuda"):
            torch.cuda.synchronize(self.info.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return VLAInferenceResult(action_chunk=chunk, latency_ms=latency_ms)

    def close(self):
        torch = self._torch
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.config = None
        self.info = None
        self._torch = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

"""Shared single/dual-arm VLA observation and action contracts.

The public dual-arm contract always keeps torque as an independent observation.
Policy-specific adapters may merge it into ``observation.state`` internally,
but callers and datasets continue to use ``observation.gripper_torque``.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from el_a3_sdk.protocol import DEFAULT_JOINT_LIMITS


STATE_KEY = "observation.state"
TORQUE_KEY = "observation.gripper_torque"
ACTION_KEY = "action"
TASK_KEY = "task"
CAMERA_PREFIX = "observation.images."


@dataclass(frozen=True)
class VLAModeSpec:
    name: str
    camera_names: tuple[str, ...]
    state_dim: int
    torque_dim: int
    action_dim: int

    @property
    def camera_keys(self):
        return tuple(f"{CAMERA_PREFIX}{name}" for name in self.camera_names)


SINGLE_ARM_SPEC = VLAModeSpec("single", ("cam1", "cam2"), 7, 1, 7)
DUAL_ARM_SPEC = VLAModeSpec("dual", ("cam1", "cam2", "cam3"), 14, 2, 14)


def get_mode_spec(mode: str | VLAModeSpec) -> VLAModeSpec:
    if isinstance(mode, VLAModeSpec):
        return mode
    if mode == "single":
        return SINGLE_ARM_SPEC
    if mode == "dual":
        return DUAL_ARM_SPEC
    raise ValueError("mode must be 'single' or 'dual'")


def _shape(feature: Any) -> tuple[int, ...] | None:
    if feature is None:
        return None
    value = feature.get("shape") if isinstance(feature, Mapping) else getattr(feature, "shape", None)
    return tuple(int(item) for item in value) if value is not None else None


def _feature_maps(config_or_features):
    if isinstance(config_or_features, Mapping):
        features = dict(config_or_features)
        return features, features
    inputs = dict(getattr(config_or_features, "input_features", None) or {})
    outputs = dict(getattr(config_or_features, "output_features", None) or {})
    return inputs, outputs


def validate_vla_features(config_or_features, mode="dual", *, allow_merged_torque=False):
    """Validate a dataset or policy feature map and return the mode spec.

    ``allow_merged_torque`` is used by ACT checkpoints whose model-facing state
    is 8D/16D after the independent torque feature has been normalized and
    concatenated by the data adapter.
    """
    spec = get_mode_spec(mode)
    inputs, outputs = _feature_maps(config_or_features)
    missing_cameras = [key for key in spec.camera_keys if key not in inputs]
    if missing_cameras:
        raise ValueError(f"{spec.name}-arm model is missing cameras: {missing_cameras}")

    state_shape = _shape(inputs.get(STATE_KEY))
    torque_shape = _shape(inputs.get(TORQUE_KEY))
    raw_pair = state_shape == (spec.state_dim,) and torque_shape == (spec.torque_dim,)
    legacy_single = (
        spec.name == "single"
        and state_shape == (spec.state_dim,)
        and torque_shape is None
    )
    merged = (
        allow_merged_torque
        and state_shape == (spec.state_dim + spec.torque_dim,)
        and torque_shape is None
    )
    if not (raw_pair or legacy_single or merged):
        expected = (
            f"{STATE_KEY}[{spec.state_dim}] + {TORQUE_KEY}[{spec.torque_dim}]"
        )
        if allow_merged_torque:
            expected += f" (or internal merged state[{spec.state_dim + spec.torque_dim}])"
        raise ValueError(
            f"{spec.name}-arm model input mismatch: expected {expected}, "
            f"got state={state_shape}, torque={torque_shape}"
        )

    action_shape = _shape(outputs.get(ACTION_KEY))
    if action_shape != (spec.action_dim,):
        raise ValueError(
            f"{spec.name}-arm action must be [{spec.action_dim}], got {action_shape}. "
            "A 7D checkpoint must not be used as a dual-arm model."
        )
    return spec


def validate_policy_checkpoint(config, mode="dual"):
    """Reject checkpoints whose camera/state/torque/action contract is unsafe."""
    policy_type = str(getattr(config, "type", "") or "").lower()
    allow_merged = policy_type == "act"
    spec = validate_vla_features(config, mode, allow_merged_torque=allow_merged)
    if spec.name == "dual" and policy_type in {"smolvla", "pi05"}:
        raise ValueError(
            f"policy type '{policy_type}' does not consume the independent 2D torque input; "
            f"use '{policy_type}_torque' for dual-arm training/inference"
        )
    return spec


def _vector(value, dim, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return np.array(result, dtype=np.float32, copy=True)


def build_vla_observation(images, state, gripper_torque, task, mode="dual"):
    """Build the canonical raw observation accepted by local or cloud VLA clients."""
    spec = get_mode_spec(mode)
    observation = {}
    for name, key in zip(spec.camera_names, spec.camera_keys):
        frame = images.get(name) if isinstance(images, Mapping) else None
        if frame is None:
            raise ValueError(f"missing camera frame: {name}")
        array = np.asarray(frame)
        if (
            array.ndim != 3
            or array.shape[-1] != 3
            or array.shape[0] <= 0
            or array.shape[1] <= 0
        ):
            raise ValueError(f"{name} must have shape [H,W,3], got {array.shape}")
        if not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
        ):
            raise ValueError(f"{name} must contain numeric pixels, got {array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
        observation[key] = np.array(array, copy=True)
    observation[STATE_KEY] = _vector(state, spec.state_dim, STATE_KEY)
    observation[TORQUE_KEY] = _vector(gripper_torque, spec.torque_dim, TORQUE_KEY)
    observation[TASK_KEY] = str(task)
    return observation


def observation_to_torch(observation, *, device="cpu", bgr_images=False):
    """Convert one canonical raw observation to a batched PyTorch observation."""
    import torch

    result = {}
    for key, value in observation.items():
        if key.startswith(CAMERA_PREFIX):
            image = np.asarray(value)
            if bgr_images:
                image = image[..., ::-1]
            image = np.ascontiguousarray(np.transpose(image, (2, 0, 1))[None, ...])
            result[key] = torch.tensor(image, dtype=torch.float32, device=device).div_(255.0)
        elif key in {STATE_KEY, TORQUE_KEY}:
            vector = np.ascontiguousarray(np.asarray(value, dtype=np.float32)[None, ...])
            result[key] = torch.tensor(vector, dtype=torch.float32, device=device)
        elif key == TASK_KEY:
            result[key] = [str(value)]
    return result


def prepare_observation_for_policy(
    observation, config, *, mode="dual", device="cpu", bgr_images=False
):
    """Build the model-facing batch while keeping the external schema stable."""
    spec = validate_policy_checkpoint(config, mode)
    batch = observation_to_torch(observation, device=device, bgr_images=bgr_images)
    policy_type = str(getattr(config, "type", "") or "").lower()
    input_features = dict(getattr(config, "input_features", None) or {})

    # Old/manual ACT checkpoints store the already merged state feature.
    state_shape = _shape(input_features.get(STATE_KEY))
    if policy_type == "act" and state_shape == (spec.state_dim + spec.torque_dim,):
        import torch
        batch[STATE_KEY] = torch.cat([batch[STATE_KEY], batch.pop(TORQUE_KEY)], dim=-1)

    # Only pass visual keys declared by the checkpoint. Dual validation above
    # guarantees all three canonical cameras are present for dual mode.
    for key in tuple(batch):
        if key.startswith(CAMERA_PREFIX) and key not in input_features:
            batch.pop(key)
    return batch


def as_action_chunk(action, mode="dual"):
    """Normalize a policy/cloud response to one unbatched chunk ``[N,D]``."""
    spec = get_mode_spec(mode)
    if isinstance(action, Mapping):
        if ACTION_KEY not in action:
            raise ValueError("inference response does not contain 'action'")
        action = action[ACTION_KEY]
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    chunk = np.asarray(action, dtype=np.float32)
    if chunk.ndim == 3:
        if chunk.shape[0] != 1:
            raise ValueError(f"only batch size 1 is supported, got {chunk.shape}")
        chunk = chunk[0]
    elif chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2 or chunk.shape[-1] != spec.action_dim:
        raise ValueError(
            f"{spec.name}-arm action chunk must have shape [N,{spec.action_dim}], got {chunk.shape}"
        )
    if chunk.shape[0] == 0:
        raise ValueError("action chunk must contain at least one step")
    if not np.isfinite(chunk).all():
        raise ValueError("action chunk contains NaN or Inf")
    return np.array(chunk, dtype=np.float32, copy=True)


def split_dual_action(action):
    """Return ``left[...,0:7]`` and ``right[...,7:14]`` from a 14D chunk."""
    chunk = as_action_chunk(action, DUAL_ARM_SPEC)
    return np.array(chunk[..., :7], copy=True), np.array(chunk[..., 7:14], copy=True)


DUAL_JOINT_NAMES = tuple(
    [f"left_J{i}" for i in range(1, 7)]
    + ["left_gripper"]
    + [f"right_J{i}" for i in range(1, 7)]
    + ["right_gripper"]
)


@dataclass(frozen=True)
class DualArmSafetyResult:
    """One safe absolute dual-arm command plus its clipping diagnostics."""

    action: np.ndarray
    joint_limit_indices: tuple[int, ...]
    delta_limit_indices: tuple[int, ...]

    @property
    def was_modified(self):
        return bool(self.joint_limit_indices or self.delta_limit_indices)


def dual_joint_limit_arrays():
    """Return SDK limits in left J1..J7, right J1..J7 ordering."""
    per_arm = [DEFAULT_JOINT_LIMITS[index] for index in range(1, 8)]
    lower = np.asarray([item[0] for item in per_arm] * 2, dtype=np.float32)
    upper = np.asarray([item[1] for item in per_arm] * 2, dtype=np.float32)
    return lower, upper


def sanitize_dual_arm_action(action, current_state, max_delta_rad):
    """Clamp one absolute 14D action to SDK and per-command delta limits.

    ``current_state`` and ``action`` use the public dual-arm ordering:
    left J1..J6 + gripper, then right J1..J6 + gripper.
    """
    command = np.asarray(action, dtype=np.float32)
    current = np.asarray(current_state, dtype=np.float32)
    if command.shape != (14,):
        raise ValueError(f"dual-arm command must have shape (14,), got {command.shape}")
    if current.shape != (14,):
        raise ValueError(f"current dual-arm state must have shape (14,), got {current.shape}")
    if not np.isfinite(command).all():
        raise ValueError("dual-arm command contains NaN or Inf")
    if not np.isfinite(current).all():
        raise ValueError("current dual-arm state contains NaN or Inf")
    try:
        max_delta = float(max_delta_rad)
    except (TypeError, ValueError) as exc:
        raise ValueError("maximum per-command delta must be a finite positive number") from exc
    if not np.isfinite(max_delta) or max_delta <= 0:
        raise ValueError("maximum per-command delta must be a finite positive number")

    lower, upper = dual_joint_limit_arrays()
    tolerance = 1e-4
    if np.any(current < lower - tolerance) or np.any(current > upper + tolerance):
        bad = np.flatnonzero((current < lower - tolerance) | (current > upper + tolerance))
        names = ", ".join(DUAL_JOINT_NAMES[index] for index in bad)
        raise ValueError(f"current arm state is outside SDK joint limits: {names}")

    absolute_safe = np.clip(command, lower, upper)
    joint_indices = tuple(
        int(index) for index in np.flatnonzero(~np.isclose(command, absolute_safe, atol=1e-7))
    )
    delta_safe = np.clip(absolute_safe, current - max_delta, current + max_delta)
    # The intersection is non-empty because the measured state was verified to
    # be inside the SDK limits. Clamp once more for floating-point edge cases.
    safe = np.clip(delta_safe, lower, upper).astype(np.float32, copy=False)
    delta_indices = tuple(
        int(index)
        for index in np.flatnonzero(~np.isclose(absolute_safe, safe, atol=1e-7))
    )
    return DualArmSafetyResult(
        action=np.array(safe, dtype=np.float32, copy=True),
        joint_limit_indices=joint_indices,
        delta_limit_indices=delta_indices,
    )


class DualArmActionDispatcher:
    """Bind dual action output to two workers, with dry-run as the safe default."""

    def __init__(self, left_worker, right_worker, *, execute=False, print_fn=print):
        self.left_worker = left_worker
        self.right_worker = right_worker
        self.execute = bool(execute)
        self.print_fn = print_fn

    def require_ready(self):
        """Refuse real execution unless both workers are connected and enabled."""
        workers = (self.left_worker, self.right_worker)
        if not all(getattr(worker, "is_connected", False) for worker in workers):
            raise RuntimeError("both arms must be connected before VLA execution")
        if not all(getattr(worker, "is_enabled", False) for worker in workers):
            raise RuntimeError("both arms must be enabled before VLA execution")

    def dispatch(self, action, *, step=0, log=True):
        left, right = split_dual_action(action)
        if step < 0 or step >= len(left):
            raise IndexError(f"step {step} outside action chunk of length {len(left)}")
        left_step = left[step]
        right_step = right[step]
        if log:
            self.print_fn(
                f"[VLA] dry-run={not self.execute} step={step} "
                f"left={left_step.tolist()} right={right_step.tolist()}"
            )
        if self.execute:
            self.require_ready()
            self.left_worker.submit_command("joint_ctrl", left_step.tolist())
            self.right_worker.submit_command("joint_ctrl", right_step.tolist())
        return {"left_action": left, "right_action": right, "executed": self.execute}


class Pi05DualArmAdapter:
    """Transport-neutral adapter for local or future cloud Pi0.5 clients."""

    mode = "dual"

    @staticmethod
    def build_request(images, state, gripper_torque, task):
        return build_vla_observation(images, state, gripper_torque, task, mode="dual")

    @staticmethod
    def validate_checkpoint(config):
        spec = validate_policy_checkpoint(config, mode="dual")
        if str(getattr(config, "type", "") or "").lower() != "pi05_torque":
            raise ValueError("dual-arm Pi0.5 requires a pi05_torque 14D checkpoint")
        return spec

    @staticmethod
    def parse_response(response):
        chunk = as_action_chunk(response, mode="dual")
        left, right = split_dual_action(chunk)
        return {"action": chunk, "left_action": left, "right_action": right}

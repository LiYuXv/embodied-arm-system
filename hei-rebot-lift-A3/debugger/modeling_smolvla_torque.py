#!/usr/bin/env python
"""SmolVLA + 夹爪力矩输入的策略模型。

继承 SmolVLAPolicy，仅重写 `prepare_state`：将力矩观测（已由 preprocessor
按 STATE 规则 MEAN_STD 归一化）拼接到 state 向量末尾，再 pad 到
max_state_dim，经原有 state_proj 投影为 state token 进入 VLM prefix。

形状约定（与基类一致）：
    训练：batch[key] 为 (B, n_obs_steps, dim)，取最后一帧 -> (B, dim)
    推理：batch[key] 为 (B, dim)，直接使用

力矩缺失（旧数据集/旧观测流程）时退化为基类行为，保持向后兼容。
"""

import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
from lerobot.utils.constants import OBS_STATE

try:
    from .configuration_smolvla_torque import SmolVLATorqueConfig
except ImportError:  # 以顶层模块方式导入（python train_model.py 运行于 debugger/）
    from configuration_smolvla_torque import SmolVLATorqueConfig


class SmolVLATorquePolicy(SmolVLAPolicy):
    """SmolVLA 策略，state 向量末尾拼接夹爪力矩作为额外本体感知输入。"""

    config_class = SmolVLATorqueConfig
    name = "smolvla_torque"

    @staticmethod
    def _latest_step(vec: torch.Tensor) -> torch.Tensor:
        """(B, T, dim) 取最后一帧 -> (B, dim)；(B, dim) 原样返回；(B,) -> (B, 1)。"""
        if vec.ndim > 2:
            vec = vec[:, -1, :]
        if vec.ndim == 1:
            vec = vec[:, None]
        return vec

    def prepare_state(self, batch):
        """拼接 state 与力矩后 pad 到 max_state_dim。"""
        state = self._latest_step(batch[OBS_STATE])
        torque_key = getattr(self.config, "torque_key", "observation.gripper_torque")
        if torque_key in batch:
            torque = self._latest_step(batch[torque_key])
            state = torch.cat([state, torque.to(dtype=state.dtype, device=state.device)], dim=-1)
        return pad_vector(state, self.config.max_state_dim)

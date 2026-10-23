#!/usr/bin/env python
"""SmolVLA + 夹爪力矩策略的预/后处理器工厂。

力矩特征（observation.gripper_torque）已被识别为 STATE 类型并随
input_features 进入 NormalizerProcessorStep 做 MEAN_STD 归一化，
因此处理流程与原版 SmolVLA 完全一致，直接复用其工厂函数。
"""

from typing import Any

import torch

from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

try:
    from .configuration_smolvla_torque import SmolVLATorqueConfig
except ImportError:  # 以顶层模块方式导入（python train_model.py 运行于 debugger/）
    from configuration_smolvla_torque import SmolVLATorqueConfig


def make_smolvla_torque_pre_post_processors(
    config: SmolVLATorqueConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """构造 smolvla_torque 的 pre/post processor，逻辑同 SmolVLA。"""
    return make_smolvla_pre_post_processors(config, dataset_stats)

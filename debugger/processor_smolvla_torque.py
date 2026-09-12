#!/usr/bin/env python
"""Pre/post processors for SmolVLA with an independent torque feature."""

from typing import Any

import torch

from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

try:
    from .configuration_smolvla_torque import SmolVLATorqueConfig
except ImportError:
    from configuration_smolvla_torque import SmolVLATorqueConfig


def make_smolvla_torque_pre_post_processors(
    config: SmolVLATorqueConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Normalize state, torque and actions using the standard SmolVLA pipeline."""
    return make_smolvla_pre_post_processors(config, dataset_stats)

#!/usr/bin/env python
"""Pi0.5 processor that incorporates an independent torque observation."""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from lerobot.configs import PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

try:
    from .act_data_utils import merge_torque_stats
    from .configuration_pi05_torque import PI05TorqueConfig
    from .vla_features import STATE_KEY, TORQUE_KEY
except ImportError:
    from act_data_utils import merge_torque_stats
    from configuration_pi05_torque import PI05TorqueConfig
    from vla_features import STATE_KEY, TORQUE_KEY


@ProcessorStepRegistry.register(name="merge_torque_into_state_processor")
@dataclass
class MergeTorqueIntoStateProcessorStep(ProcessorStep):
    """Concatenate state and torque on the last dimension and remove torque."""

    state_key: str = STATE_KEY
    torque_key: str = TORQUE_KEY

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        result = transition.copy()
        observation = dict(result.get(TransitionKey.OBSERVATION) or {})
        if self.state_key not in observation or self.torque_key not in observation:
            raise ValueError(
                f"Pi0.5 torque adapter requires {self.state_key} and {self.torque_key}"
            )
        state, torque = observation[self.state_key], observation[self.torque_key]
        if isinstance(state, torch.Tensor):
            torque = torch.as_tensor(torque, dtype=state.dtype, device=state.device)
            observation[self.state_key] = torch.cat([state, torque], dim=-1)
        else:
            observation[self.state_key] = np.concatenate(
                [np.asarray(state), np.asarray(torque)], axis=-1
            )
        observation.pop(self.torque_key)
        result[TransitionKey.OBSERVATION] = observation
        return result

    def transform_features(self, features):
        result = deepcopy(features)
        for group in result.values():
            if self.state_key not in group or self.torque_key not in group:
                continue
            state_ft = group[self.state_key]
            torque_ft = group.pop(self.torque_key)
            group[self.state_key] = replace(
                state_ft, shape=(state_ft.shape[0] + torque_ft.shape[0],)
            )
        return result


def _processor_config(config: PI05TorqueConfig):
    merged = deepcopy(config)
    inputs = dict(merged.input_features)
    state_ft = inputs[STATE_KEY]
    torque_ft = inputs.pop(config.torque_key)
    inputs[STATE_KEY] = PolicyFeature(
        type=state_ft.type,
        shape=(state_ft.shape[0] + torque_ft.shape[0],),
    )
    merged.input_features = inputs
    return merged


def make_pi05_torque_pre_post_processors(
    config: PI05TorqueConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Normalize the merged 8D/16D proprioception and standard Pi0.5 actions."""
    config.validate_features()
    processor_config = _processor_config(config)
    merged_stats = merge_torque_stats(dataset_stats) if dataset_stats else dataset_stats
    steps = make_default_policy_processor_steps(processor_config, merged_stats)
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )
    merge_step = MergeTorqueIntoStateProcessorStep(torque_key=config.torque_key)

    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        relative_step,
        merge_step,
        steps.normalize,
        Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            include_state_in_prompt=not config.use_proprioceptive_memory,
        ),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        steps.to_device,
    ]
    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(
            enabled=config.use_relative_actions, relative_step=relative_step
        ),
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(
        input_steps=input_steps, output_steps=output_steps
    )

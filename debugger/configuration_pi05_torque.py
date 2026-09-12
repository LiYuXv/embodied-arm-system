#!/usr/bin/env python
"""Pi0.5 configuration that consumes independent gripper torque observations."""

from dataclasses import dataclass

try:
    from lerobot.configs import PreTrainedConfig
except ImportError:
    from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi05_torque")
@dataclass
class PI05TorqueConfig(PI05Config):
    """Pi0.5 with state/torque merged by its processor after normalization."""

    torque_key: str = "observation.gripper_torque"
    robot_mode: str = "single"

    def validate_features(self) -> None:
        super().validate_features()
        if not self.input_features or not self.output_features:
            return
        try:
            from .vla_features import validate_vla_features
        except ImportError:
            from vla_features import validate_vla_features
        spec = validate_vla_features(self, self.robot_mode)
        if spec.state_dim + spec.torque_dim > self.max_state_dim:
            raise ValueError(
                f"state+torque dimension {spec.state_dim + spec.torque_dim} "
                f"exceeds max_state_dim={self.max_state_dim}"
            )
        if spec.action_dim > self.max_action_dim:
            raise ValueError(
                f"action dimension {spec.action_dim} exceeds "
                f"max_action_dim={self.max_action_dim}"
            )

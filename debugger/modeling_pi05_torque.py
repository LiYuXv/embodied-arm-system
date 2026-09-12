#!/usr/bin/env python
"""Pi0.5 policy class for the torque-aware processor contract."""

from lerobot.policies.pi05.modeling_pi05 import PI05Policy

try:
    from .configuration_pi05_torque import PI05TorqueConfig
except ImportError:
    from configuration_pi05_torque import PI05TorqueConfig


class PI05TorquePolicy(PI05Policy):
    """The processor supplies normalized ``state + torque`` to stock Pi0.5."""

    config_class = PI05TorqueConfig
    name = "pi05_torque"

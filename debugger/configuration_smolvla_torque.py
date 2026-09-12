#!/usr/bin/env python
"""SmolVLA + 夹爪力矩输入的策略配置。

继承 SmolVLAConfig，新增 `torque_key` 指定力矩观测在 batch 中的键名。
数据集中以 `observation.` 开头的特征（如 observation.gripper_torque）会被
lerobot 自动归类为 FeatureType.STATE，进入 input_features 并按
normalization_mapping 的 STATE -> MEAN_STD 规则归一化（使用数据集
stats.json 中该特征的 mean/std），无需额外配置。

注册名 "smolvla_torque"：训练/推理时按 lerobot 插件约定解析
    configuration_smolvla_torque.SmolVLATorqueConfig
    -> modeling_smolvla_torque.SmolVLATorquePolicy
    -> processor_smolvla_torque.make_smolvla_torque_pre_post_processors
"""

from dataclasses import dataclass

try:  # lerobot >= 0.5 顶层导出；0.4.x 需从子模块导入
    from lerobot.configs import PreTrainedConfig
except ImportError:
    from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolvla_torque")
@dataclass
class SmolVLATorqueConfig(SmolVLAConfig):
    """SmolVLA 配置，额外将夹爪力矩观测拼入 state 向量输入模型。"""

    # 力矩观测在 batch / 数据集 features 中的键名。
    # 该特征须以 "observation." 开头才会被识别为 STATE 类型并归一化。
    torque_key: str = "observation.gripper_torque"

    # Checkpoint contract. Existing checkpoints omit this field and therefore
    # remain single-arm by default; new dual checkpoints must set it explicitly.
    robot_mode: str = "single"

    # 兼容新版 lerobot（>=0.6）训练保存的 config.json 中多出的字段；
    # 旧版（0.4.x）的 draccus 严格校验，遇到未知字段会直接拒绝加载。
    pretrained_revision: str | None = None

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

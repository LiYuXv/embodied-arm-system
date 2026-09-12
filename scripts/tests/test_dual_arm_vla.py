#!/usr/bin/env python3
"""Dual/single-arm VLA contract tests without cameras, robot hardware, or weights."""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEBUGGER = os.path.join(ROOT, "debugger")
for path in (ROOT, DEBUGGER):
    if path not in sys.path:
        sys.path.insert(0, path)

from act_data_utils import (  # noqa: E402
    make_collate_fn,
    merge_torque_into_state_features,
    merge_torque_stats,
)
from processor_pi05_torque import MergeTorqueIntoStateProcessorStep  # noqa: E402
from configuration_pi05_torque import PI05TorqueConfig  # noqa: E402
from modeling_smolvla_torque import SmolVLATorquePolicy  # noqa: E402
from vla_features import (  # noqa: E402
    ACTION_KEY,
    DUAL_ARM_SPEC,
    SINGLE_ARM_SPEC,
    STATE_KEY,
    TASK_KEY,
    TORQUE_KEY,
    DualArmActionDispatcher,
    Pi05DualArmAdapter,
    as_action_chunk,
    build_vla_observation,
    observation_to_torch,
    split_dual_action,
    validate_policy_checkpoint,
    validate_vla_features,
)


def feature(shape):
    return {"shape": list(shape)}


def policy_config(policy_type, mode, *, merged=False, include_torque=True):
    spec = DUAL_ARM_SPEC if mode == "dual" else SINGLE_ARM_SPEC
    state_dim = spec.state_dim + (spec.torque_dim if merged else 0)
    inputs = {key: feature((24, 32, 3)) for key in spec.camera_keys}
    inputs[STATE_KEY] = feature((state_dim,))
    if include_torque and not merged:
        inputs[TORQUE_KEY] = feature((spec.torque_dim,))
    return SimpleNamespace(
        type=policy_type,
        robot_mode=mode,
        input_features=inputs,
        output_features={ACTION_KEY: feature((spec.action_dim,))},
    )


class TestVLAFeatures(unittest.TestCase):
    def test_dual_observation_has_three_cameras_state_torque_and_task(self):
        images = {
            name: np.full((24, 32, 3), index, dtype=np.uint8)
            for index, name in enumerate(DUAL_ARM_SPEC.camera_names, start=1)
        }
        observation = build_vla_observation(
            images,
            np.arange(14),
            [0.25, -0.25],
            "pick the block",
            mode="dual",
        )
        self.assertEqual(set(DUAL_ARM_SPEC.camera_keys).issubset(observation), True)
        self.assertEqual(observation[STATE_KEY].shape, (14,))
        self.assertEqual(observation[TORQUE_KEY].shape, (2,))
        self.assertEqual(observation[TASK_KEY], "pick the block")

        batch = observation_to_torch(observation)
        self.assertEqual(batch[DUAL_ARM_SPEC.camera_keys[0]].shape, (1, 3, 24, 32))
        self.assertEqual(batch[STATE_KEY].shape, (1, 14))
        self.assertEqual(batch[TORQUE_KEY].shape, (1, 2))
        self.assertEqual(batch[TASK_KEY], ["pick the block"])

    def test_dual_action_chunk_splits_left_then_right(self):
        chunk = np.arange(42, dtype=np.float32).reshape(3, 14)
        left, right = split_dual_action(chunk)
        np.testing.assert_array_equal(left, chunk[:, 0:7])
        np.testing.assert_array_equal(right, chunk[:, 7:14])
        self.assertEqual(as_action_chunk(chunk, "dual").shape, (3, 14))

    def test_single_arm_and_legacy_checkpoint_remain_compatible(self):
        raw = policy_config("smolvla_torque", "single")
        legacy = policy_config("smolvla", "single", include_torque=False)
        self.assertEqual(validate_policy_checkpoint(raw, "single").action_dim, 7)
        self.assertEqual(validate_policy_checkpoint(legacy, "single").action_dim, 7)
        self.assertEqual(as_action_chunk(np.zeros(7), "single").shape, (1, 7))

    def test_seven_dimensional_checkpoint_is_rejected_for_dual_mode(self):
        config = policy_config("pi05_torque", "dual")
        config.output_features[ACTION_KEY] = feature((7,))
        with self.assertRaisesRegex(ValueError, "14"):
            validate_policy_checkpoint(config, "dual")

    def test_standard_pi05_cannot_silently_ignore_dual_torque(self):
        with self.assertRaisesRegex(ValueError, "pi05_torque"):
            validate_policy_checkpoint(policy_config("pi05", "dual"), "dual")
        self.assertEqual(
            Pi05DualArmAdapter.validate_checkpoint(
                policy_config("pi05_torque", "dual")
            ).action_dim,
            14,
        )

    def test_pi05_transport_adapter_uses_canonical_request_and_chunk_response(self):
        images = {
            name: np.zeros((8, 12, 3), dtype=np.uint8)
            for name in DUAL_ARM_SPEC.camera_names
        }
        request = Pi05DualArmAdapter.build_request(
            images, np.zeros(14), np.zeros(2), "stack blocks"
        )
        self.assertEqual(request[STATE_KEY].shape, (14,))
        self.assertEqual(request[TORQUE_KEY].shape, (2,))
        parsed = Pi05DualArmAdapter.parse_response(
            {ACTION_KEY: np.arange(28).reshape(2, 14)}
        )
        self.assertEqual(parsed[ACTION_KEY].shape, (2, 14))
        self.assertEqual(parsed["left_action"].shape, (2, 7))
        self.assertEqual(parsed["right_action"].shape, (2, 7))


class TestTorqueAdapters(unittest.TestCase):
    def test_smolvla_model_prepares_fourteen_plus_two_dimensional_state(self):
        policy = SmolVLATorquePolicy.__new__(SmolVLATorquePolicy)
        torch.nn.Module.__init__(policy)
        policy.config = SimpleNamespace(max_state_dim=32, torque_key=TORQUE_KEY)
        prepared = policy.prepare_state({
            STATE_KEY: torch.arange(14, dtype=torch.float32).unsqueeze(0),
            TORQUE_KEY: torch.tensor([[0.4, -0.2]]),
        })
        self.assertEqual(prepared.shape, (1, 32))
        torch.testing.assert_close(
            prepared[0, :16],
            torch.tensor(list(range(14)) + [0.4, -0.2]),
        )

    def test_pi05_dual_config_validates_three_camera_raw_contract(self):
        from lerobot.configs.types import FeatureType, PolicyFeature

        config = PI05TorqueConfig(robot_mode="dual", device="cpu")
        config.input_features = {
            **{
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 24, 32))
                for key in DUAL_ARM_SPEC.camera_keys
            },
            STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            TORQUE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
        config.output_features = {
            ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(14,))
        }
        config.validate_features()

    def test_pi05_processor_merges_two_dimensional_torque(self):
        transition = {
            "observation": {
                STATE_KEY: torch.arange(14, dtype=torch.float32).unsqueeze(0),
                TORQUE_KEY: torch.tensor([[0.4, -0.2]]),
                TASK_KEY: ["task"],
            }
        }
        result = MergeTorqueIntoStateProcessorStep()(transition)
        self.assertEqual(result["observation"][STATE_KEY].shape, (1, 16))
        self.assertNotIn(TORQUE_KEY, result["observation"])
        torch.testing.assert_close(
            result["observation"][STATE_KEY][0, -2:],
            torch.tensor([0.4, -0.2]),
        )

    def test_act_features_stats_and_batch_merge_to_sixteen_dimensions(self):
        # The helper rebuilds a PolicyFeature, so use real compatible feature types.
        from lerobot.configs.types import FeatureType, PolicyFeature

        features = {
            STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            TORQUE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
        merge_torque_into_state_features(features)
        self.assertEqual(features[STATE_KEY].shape, (16,))

        stats = {
            STATE_KEY: {"mean": np.arange(14), "std": np.ones(14)},
            TORQUE_KEY: {"mean": np.array([14, 15]), "std": np.ones(2)},
        }
        merged_stats = merge_torque_stats(stats)
        self.assertEqual(merged_stats[STATE_KEY]["mean"].shape, (16,))

        collate = make_collate_fn()
        batch = collate([
            {STATE_KEY: torch.arange(14), TORQUE_KEY: torch.tensor([0.1, 0.2])}
        ])
        self.assertEqual(batch[STATE_KEY].shape, (1, 16))
        self.assertNotIn(TORQUE_KEY, batch)


class FakeWorker:
    def __init__(self, *, connected=True, enabled=True):
        self.is_connected = connected
        self.is_enabled = enabled
        self.commands = []

    def submit_command(self, name, value):
        self.commands.append((name, value))


class TestDualActionDispatcher(unittest.TestCase):
    def test_dry_run_is_default_and_does_not_touch_workers(self):
        left, right = FakeWorker(), FakeWorker()
        messages = []
        result = DualArmActionDispatcher(left, right, print_fn=messages.append).dispatch(
            np.arange(28).reshape(2, 14), step=1
        )
        self.assertFalse(result["executed"])
        self.assertEqual(left.commands, [])
        self.assertEqual(right.commands, [])
        self.assertIn("dry-run=True", messages[0])

    def test_explicit_execution_dispatches_matching_seven_dimensional_halves(self):
        left, right = FakeWorker(), FakeWorker()
        DualArmActionDispatcher(
            left, right, execute=True, print_fn=lambda _message: None
        ).dispatch(np.arange(14), step=0)
        self.assertEqual(left.commands, [("joint_ctrl", list(range(7)))])
        self.assertEqual(right.commands, [("joint_ctrl", list(range(7, 14)))])


if __name__ == "__main__":
    unittest.main(verbosity=2)

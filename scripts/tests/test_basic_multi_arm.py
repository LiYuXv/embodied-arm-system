#!/usr/bin/env python3
"""双臂基础层测试；使用替身接口，不连接 CAN 或真实机械臂。"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from el_a3_sdk import ArmManager, initialize_dual_arms
from el_a3_sdk.arm_manager import DEFAULT_MULTI_ARM_CONFIG


class FakeArm:
    """只记录初始化和连接行为的 ELA3Interface 替身。"""

    def __init__(self, can_name="can0", **kwargs):
        self.can_name = can_name
        self.kwargs = kwargs
        self.connect_calls = 0
        self.disconnect_calls = 0

    def ConnectPort(self):
        self.connect_calls += 1
        return True

    def DisconnectPort(self):
        self.disconnect_calls += 1


class TestBasicMultiArm(unittest.TestCase):
    def setUp(self):
        self.interface_patch = patch("el_a3_sdk.interface.ELA3Interface", FakeArm)
        self.interface_patch.start()
        ArmManager.reset()

    def tearDown(self):
        ArmManager.reset()
        self.interface_patch.stop()

    def test_default_config_declares_left_and_right_can_mapping(self):
        self.assertTrue(DEFAULT_MULTI_ARM_CONFIG.is_file())

        manager = initialize_dual_arms()
        left_arm = manager["left_arm"]
        right_arm = manager["right_arm"]

        self.assertEqual(manager.arm_names, ["left_arm", "right_arm"])
        self.assertEqual(left_arm.can_name, "can0")
        self.assertEqual(right_arm.can_name, "can1")
        self.assertEqual(left_arm.kwargs["host_can_id"], 253)
        self.assertEqual(right_arm.kwargs["host_can_id"], 253)
        for arm in (left_arm, right_arm):
            self.assertNotIn("gravity_feedforward_ratio", arm.kwargs)
            self.assertNotIn("per_joint_zero_torque_gravity_ratio", arm.kwargs)
            self.assertEqual(
                arm.kwargs["inertia_config_path"],
                str(DEFAULT_MULTI_ARM_CONFIG.parent / "inertia_params.yaml"),
            )
            self.assertEqual(
                arm.kwargs["per_joint_kd_min"],
                {4: 0.005, 5: 0.005, 6: 0.005, 7: 0.02},
            )
            self.assertEqual(
                arm.kwargs["per_joint_kd_max"],
                {4: 0.10, 5: 0.05, 6: 0.05, 7: 0.10},
            )

    def test_left_and_right_are_independent_instances(self):
        manager = initialize_dual_arms()

        self.assertIsNot(manager["left_arm"], manager["right_arm"])
        self.assertIs(manager.get_arm("left_arm"), manager["left_arm"])
        self.assertIs(manager.get_arm("right_arm"), manager["right_arm"])

    def test_gravity_ratio_can_differ_between_arms(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "multi_arm.yaml"
            config_path.write_text(
                "arms:\n"
                "  left_arm:\n"
                "    can_interface: can0\n"
                "    gravity_feedforward_ratio: 0.7\n"
                "  right_arm:\n"
                "    can_interface: can1\n"
                "    gravity_feedforward_ratio: 0.8\n",
                encoding="utf-8",
            )
            manager = ArmManager.from_config(config_path)

        self.assertEqual(manager["left_arm"].kwargs["gravity_feedforward_ratio"], 0.7)
        self.assertEqual(manager["right_arm"].kwargs["gravity_feedforward_ratio"], 0.8)

    def test_auto_connect_connects_each_arm_once(self):
        manager = initialize_dual_arms(auto_connect=True)

        self.assertEqual(manager["left_arm"].connect_calls, 1)
        self.assertEqual(manager["right_arm"].connect_calls, 1)

    def test_existing_single_arm_registration_api_is_preserved(self):
        manager = ArmManager()
        arm = manager.register_can_arm("legacy_arm", can_name="can9")

        self.assertIs(manager.register_can_arm("legacy_arm", can_name="can9"), arm)
        self.assertIs(manager.get_arm("legacy_arm"), arm)

    def test_relative_resource_paths_are_resolved_from_config_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "multi_arm.yaml"
            config_path.write_text(
                "arms:\n"
                "  test_arm:\n"
                "    can_interface: vcan0\n"
                "    inertia_config_path: inertia.yaml\n",
                encoding="utf-8",
            )

            manager = ArmManager.from_config(config_path)

        self.assertEqual(
            manager["test_arm"].kwargs["inertia_config_path"],
            str((Path(temp_dir) / "inertia.yaml").resolve()),
        )

    def test_same_name_with_different_config_is_rejected(self):
        manager = ArmManager()
        manager.register_can_arm("arm", can_name="can0")

        with self.assertRaisesRegex(ValueError, "不同配置"):
            manager.register_can_arm("arm", can_name="can1")


if __name__ == "__main__":
    unittest.main(verbosity=2)

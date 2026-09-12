#!/usr/bin/env python3
"""Joy-Con 输入、左右绑定和双臂遥操离线测试。"""

import os
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from demo.joycon_dual_arm_control import build_dual_joycon_controllers
from demo.vr_control import VRArmController
from el_a3_sdk.controller_profiles import JOYCON_PROFILES
from el_a3_sdk.joycon import (
    ImuOrientationFilter,
    JoyConDeviceBinding,
    JoyConInput,
    JoyConPoseAdapter,
    JoyConSnapshot,
    LinuxEventCodes,
    RelativePoseInput,
    discover_joycon,
    discover_joycons,
)


@dataclass
class FakeDeviceInfo:
    vendor: int
    product: int


@dataclass
class FakeAbsInfo:
    min: int = -32767
    max: int = 32767
    flat: int = 500
    resolution: int = 0


class FakeDevice:
    def __init__(self, path, name, product, uniq):
        self.path = path
        self.name = name
        self.info = FakeDeviceInfo(0x057E, product)
        self.uniq = uniq
        self.phys = f"bluetooth/{uniq}"

    def input_props(self):
        return [LinuxEventCodes.INPUT_PROP_ACCELEROMETER] if "IMU" in self.name else []

    def absinfo(self, code):
        if "IMU" not in self.name:
            return FakeAbsInfo()
        if code in (LinuxEventCodes.ABS_X, LinuxEventCodes.ABS_Y, LinuxEventCodes.ABS_Z):
            return FakeAbsInfo(resolution=4096)
        return FakeAbsInfo(resolution=14247)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_binding(side="left"):
    product = JOYCON_PROFILES[side].product_id
    label = "Left" if side == "left" else "Right"
    return JoyConDeviceBinding(
        side=side,
        control_device=FakeDevice(f"/dev/input/{side}-control", f"Nintendo Switch {label} Joy-Con", product, side),
        imu_device=FakeDevice(f"/dev/input/{side}-imu", f"Nintendo Switch {label} Joy-Con (IMU)", product, side),
    )


class TestJoyConDiscovery(unittest.TestCase):
    def test_discovers_one_side_without_requiring_a_pair(self):
        devices = [
            FakeDevice("/dev/input/event2", "Nintendo Switch Left Joy-Con", 0x2006, "L-MAC"),
            FakeDevice("/dev/input/event31", "Nintendo Switch Left Joy-Con (IMU)", 0x2006, "L-MAC"),
        ]

        binding = discover_joycon(
            "left", devices=devices, ecodes=LinuxEventCodes
        )

        self.assertEqual(binding.side, "left")
        self.assertEqual(binding.control_path, "/dev/input/event2")
        self.assertEqual(binding.imu_path, "/dev/input/event31")

    def test_discovers_and_pairs_shuffled_event_nodes_without_event_numbers(self):
        devices = [
            FakeDevice("/dev/input/event19", "Nintendo Switch Right Joy-Con (IMU)", 0x2007, "R-MAC"),
            FakeDevice("/dev/input/event2", "Nintendo Switch Left Joy-Con", 0x2006, "L-MAC"),
            FakeDevice("/dev/input/event8", "Nintendo Switch Right Joy-Con", 0x2007, "R-MAC"),
            FakeDevice("/dev/input/event31", "Nintendo Switch Left Joy-Con (IMU)", 0x2006, "L-MAC"),
        ]

        pair = discover_joycons(devices=devices, ecodes=LinuxEventCodes)

        self.assertEqual(pair.left.control_path, "/dev/input/event2")
        self.assertEqual(pair.left.imu_path, "/dev/input/event31")
        self.assertEqual(pair.right.control_path, "/dev/input/event8")
        self.assertEqual(pair.right.imu_path, "/dev/input/event19")


class TestJoyConInput(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.source = JoyConInput(
            make_binding("left"),
            ecodes=LinuxEventCodes,
            clock=self.clock,
        )

    def _feed_stationary_imu(self):
        for code, value in (
            (LinuxEventCodes.ABS_X, 0),
            (LinuxEventCodes.ABS_Y, 0),
            (LinuxEventCodes.ABS_Z, 4096),
            (LinuxEventCodes.ABS_RX, 0),
            (LinuxEventCodes.ABS_RY, 0),
            (LinuxEventCodes.ABS_RZ, 0),
        ):
            self.source.handle_imu_event(LinuxEventCodes.EV_ABS, code, value)
        self.source.handle_imu_event(
            LinuxEventCodes.EV_SYN,
            LinuxEventCodes.SYN_REPORT,
            0,
            self.clock(),
        )

    def test_reads_normalized_stick_buttons_and_imu(self):
        self._feed_stationary_imu()
        profile = JOYCON_PROFILES["left"]
        self.source.handle_control_event(
            LinuxEventCodes.EV_ABS,
            getattr(LinuxEventCodes, profile.stick_x),
            16384,
        )
        self.source.handle_control_event(
            LinuxEventCodes.EV_ABS,
            getattr(LinuxEventCodes, profile.stick_y),
            -16384,
        )
        self.source.handle_control_event(
            LinuxEventCodes.EV_KEY,
            getattr(LinuxEventCodes, profile.clutch),
            1,
        )

        state = self.source.snapshot()

        self.assertTrue(state.valid)
        self.assertTrue(state.clutch)
        self.assertAlmostEqual(state.stick_x, 0.5, places=2)
        self.assertAlmostEqual(state.stick_y, -0.5, places=2)
        np.testing.assert_allclose(state.rotation, np.eye(3), atol=1e-8)

        # ZL/ZR 是按住式姿态离合，松开立即释放。
        self.source.handle_control_event(
            LinuxEventCodes.EV_KEY,
            getattr(LinuxEventCodes, profile.clutch),
            0,
        )
        self.assertFalse(self.source.snapshot().clutch)
        self.source.handle_control_event(
            LinuxEventCodes.EV_KEY,
            getattr(LinuxEventCodes, profile.clutch),
            1,
        )
        self.assertTrue(self.source.snapshot().clutch)

    def test_stale_imu_releases_validity(self):
        self._feed_stationary_imu()
        profile = JOYCON_PROFILES["left"]
        self.source.handle_control_event(
            LinuxEventCodes.EV_KEY,
            getattr(LinuxEventCodes, profile.clutch),
            1,
        )
        self.assertTrue(self.source.snapshot().clutch)
        self.clock.advance(0.6)
        state = self.source.snapshot()
        self.assertFalse(state.valid)
        self.assertFalse(state.clutch)

    def test_right_profile_uses_right_stick_and_zr_clutch(self):
        source = JoyConInput(
            make_binding("right"),
            ecodes=LinuxEventCodes,
            clock=self.clock,
        )
        profile = JOYCON_PROFILES["right"]
        for code, value in (
            (LinuxEventCodes.ABS_X, 0),
            (LinuxEventCodes.ABS_Y, 0),
            (LinuxEventCodes.ABS_Z, 4096),
            (LinuxEventCodes.ABS_RX, 0),
            (LinuxEventCodes.ABS_RY, 0),
            (LinuxEventCodes.ABS_RZ, 0),
        ):
            source.handle_imu_event(LinuxEventCodes.EV_ABS, code, value)
        source.handle_imu_event(
            LinuxEventCodes.EV_SYN,
            LinuxEventCodes.SYN_REPORT,
            0,
            self.clock(),
        )
        source.handle_control_event(
            LinuxEventCodes.EV_ABS,
            getattr(LinuxEventCodes, profile.stick_x),
            -20000,
        )
        source.handle_control_event(
            LinuxEventCodes.EV_KEY,
            getattr(LinuxEventCodes, profile.clutch),
            1,
        )
        source.handle_control_event(
            LinuxEventCodes.EV_KEY,
            getattr(LinuxEventCodes, profile.gripper_close),
            1,
        )

        state = source.snapshot()

        self.assertLess(state.stick_x, -0.5)
        self.assertTrue(state.clutch)
        self.assertTrue(state.gripper_close)

    def test_profiles_do_not_assign_side_rail_sl_sr_buttons(self):
        side_rail_codes = {
            "left": {"BTN_TR", "BTN_TR2"},
            "right": {"BTN_TL", "BTN_TL2"},
        }
        for side, profile in JOYCON_PROFILES.items():
            assigned = {
                profile.clutch,
                profile.z_positive,
                profile.z_negative,
                profile.gripper_close,
                profile.gripper_open,
                profile.zero,
                profile.home,
                profile.exit,
            }
            self.assertTrue(assigned.isdisjoint(side_rail_codes[side]))


class TestImuOrientationFilter(unittest.TestCase):
    def test_stationary_filter_remains_at_relative_identity(self):
        orientation = ImuOrientationFilter()
        orientation.update([0, 0, 1], [0, 0, 0], 0.0)
        for index in range(1, 101):
            rotation = orientation.update([0, 0, 1], [0, 0, 0], index * 0.01)

        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-8)

    def test_integrates_gyro_to_relative_yaw_and_keeps_rotation_orthonormal(self):
        orientation = ImuOrientationFilter()
        orientation.update([0, 0, 1], [0, 0, 0], 0.0)
        for index in range(1, 21):
            rotation = orientation.update(
                [0, 0, 1], [0, 0, np.pi / 2], index * 0.05
            )

        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        self.assertAlmostEqual(yaw, np.pi / 2, places=2)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)


class FakeSource:
    def __init__(self, state):
        self.state = state

    def snapshot(self):
        return self.state

    def start(self):
        pass

    def stop(self):
        pass

    def is_connected(self):
        return True


def snapshot(clutch=False, stick_y=0.0):
    return JoyConSnapshot(
        side="left",
        valid=True,
        stick_x=0.0,
        stick_y=stick_y,
        rotation=np.eye(3),
        clutch=clutch,
        z_positive=False,
        z_negative=False,
        gripper_close=False,
        gripper_open=False,
        zero=False,
        home=False,
        exit=False,
    )


class TestJoyConPoseAdapter(unittest.TestCase):
    def test_translation_does_not_require_rotation_clutch(self):
        clock = FakeClock()
        source = FakeSource(snapshot(clutch=False, stick_y=-1.0))
        adapter = JoyConPoseAdapter(source, translation_speed=1.0, clock=clock)

        clock.advance(0.05)
        released = adapter.get_state()
        source.state = snapshot(clutch=True, stick_y=-1.0)
        clock.advance(0.05)
        engaged = adapter.get_state()

        np.testing.assert_allclose(released.position, [0.05, 0, 0])
        np.testing.assert_allclose(engaged.position, [0.10, 0, 0])
        self.assertEqual(released.grip, 0.0)
        self.assertEqual(engaged.grip, 1.0)
        self.assertTrue(released.translation_without_clutch)

    def test_controller_gates_rotation_but_not_xyz_translation(self):
        class FakeRpy:
            @staticmethod
            def matrixToRpy(rotation):
                return np.array([0.0, 0.0, np.arctan2(rotation[1, 0], rotation[0, 0])])

            @staticmethod
            def rpyToMatrix(_roll, _pitch, yaw):
                return rotation_z(yaw)

        class FakeKinematics:
            def forward_kinematics_se3(self, _joints):
                return SimpleNamespace(translation=np.zeros(3), rotation=np.eye(3))

            def ik_step(self, _pose, _seed, **_kwargs):
                return [0.0] * 6, 0.0

        class FakeArm:
            def __init__(self):
                self.kinematics = FakeKinematics()

            def _get_kinematics(self):
                return self.kinematics

            def GetArmJointMsgs(self):
                return SimpleNamespace(
                    to_list=lambda include_gripper=True: [0.0] * (7 if include_gripper else 6)
                )

            def GetArmJointEfforts(self):
                return SimpleNamespace(
                    to_list=lambda include_gripper=True: [0.0] * (7 if include_gripper else 6)
                )

            def JointCtrl(self, *_positions, **_kwargs):
                pass

            def GripperCtrl(self, **_kwargs):
                pass

        class FakeReceiver:
            def __init__(self):
                self.state = relative_state([0, 0, 0], np.eye(3), grip=0.0)

            def get_state(self):
                return self.state

            def is_connected(self):
                return True

        def rotation_z(angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

        def relative_state(position, rotation, grip):
            return RelativePoseInput(
                valid=True,
                position=np.asarray(position, dtype=float),
                rotation=rotation,
                grip=grip,
                joystick_y=0.0,
                button_a=False,
                button_b=False,
                translation_without_clutch=True,
            )

        receiver = FakeReceiver()
        fake_pinocchio = SimpleNamespace(rpy=FakeRpy())
        with patch.dict(sys.modules, {"pinocchio": fake_pinocchio}):
            controller = VRArmController(
                FakeArm(),
                receiver,
                state_getter=receiver.get_state,
                position_fix=np.ones(3),
                rotation_fix=np.eye(3),
                remap_rotation_rpy=False,
                input_name="Joy-Con (L)",
            )
        controller._initialize()
        controller._tick()  # establish the free-translation anchor

        receiver.state = relative_state([0.1, 0, 0], rotation_z(np.pi / 2), grip=0.0)
        controller._tick()
        np.testing.assert_allclose(controller._target_pos, [0.1, 0, 0], atol=1e-8)
        np.testing.assert_allclose(controller._target_rot, np.eye(3), atol=1e-8)

        receiver.state = relative_state([0.1, 0, 0], rotation_z(np.pi / 2), grip=1.0)
        controller._tick()  # clutch press reanchors rotation without a jump
        np.testing.assert_allclose(controller._target_rot, np.eye(3), atol=1e-8)
        receiver.state = relative_state([0.1, 0, 0], rotation_z(np.pi), grip=1.0)
        controller._tick()
        held_rotation = controller._target_rot.copy()
        np.testing.assert_allclose(held_rotation, rotation_z(np.pi / 2), atol=1e-8)

        receiver.state = relative_state([0.2, 0, 0], np.eye(3), grip=0.0)
        controller._tick()
        np.testing.assert_allclose(controller._target_pos, [0.2, 0, 0], atol=1e-8)
        np.testing.assert_allclose(controller._target_rot, held_rotation, atol=1e-8)


class RecordingController:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestDualArmBinding(unittest.TestCase):
    def test_left_and_right_sources_bind_to_matching_arm_instances(self):
        left_arm = object()
        right_arm = object()
        left_adapter = FakeSource(snapshot())
        right_adapter = FakeSource(snapshot())
        left_adapter.get_state = left_adapter.snapshot
        right_adapter.get_state = right_adapter.snapshot

        controllers = build_dual_joycon_controllers(
            {"left_arm": left_arm, "right_arm": right_arm},
            {"left": left_adapter, "right": right_adapter},
            controller_factory=RecordingController,
        )

        self.assertIs(controllers["left"].kwargs["arm"], left_arm)
        self.assertIs(controllers["right"].kwargs["arm"], right_arm)
        self.assertIs(controllers["left"].kwargs["receiver"], left_adapter)
        self.assertIs(controllers["right"].kwargs["receiver"], right_adapter)
        self.assertFalse(controllers["left"].kwargs["remap_rotation_rpy"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

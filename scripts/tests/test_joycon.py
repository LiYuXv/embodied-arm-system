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
    JOYCON_ROTATION_VECTOR_SIGN,
    JOYCON_MODE_IMU_MOTION,
    JOYCON_MODE_STANDARD,
    JOYCON_ROTATION_FIX,
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
        np.testing.assert_allclose(state.linear_acceleration, np.zeros(3), atol=1e-8)

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

    def test_acceleration_is_rotated_to_fixed_frame_and_gravity_removed(self):
        # 即使传感器的重力轴不是本体系 Z，初始化姿态补偿后静止线加速度也应为零。
        for code, value in (
            (LinuxEventCodes.ABS_X, 4096),
            (LinuxEventCodes.ABS_Y, 0),
            (LinuxEventCodes.ABS_Z, 0),
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

        np.testing.assert_allclose(
            self.source.snapshot().linear_acceleration, np.zeros(3), atol=1e-8
        )

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


def snapshot(
    clutch=False,
    stick_y=0.0,
    stick_x=0.0,
    *,
    valid=True,
    acceleration=None,
    angular_velocity=None,
    rotation=None,
):
    return JoyConSnapshot(
        side="left",
        valid=valid,
        stick_x=stick_x,
        stick_y=stick_y,
        rotation=np.eye(3) if rotation is None else np.asarray(rotation, dtype=float),
        clutch=clutch,
        z_positive=False,
        z_negative=False,
        gripper_close=False,
        gripper_open=False,
        zero=False,
        home=False,
        exit=False,
        linear_acceleration=np.zeros(3) if acceleration is None else np.asarray(acceleration, dtype=float),
        angular_velocity=np.zeros(3) if angular_velocity is None else np.asarray(angular_velocity, dtype=float),
    )


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


class TestJoyConRotationCalibration(unittest.TestCase):
    def test_rotation_vector_sign_keeps_pitch_yaw_and_only_flips_z(self):
        np.testing.assert_allclose(JOYCON_ROTATION_VECTOR_SIGN, [-1.0, 1.0, -1.0])

    def test_mixed_rotation_vector_components_apply_signs_independently(self):
        import pinocchio as pin

        rotvec = np.array([0.20, -0.13, 0.17], dtype=float)
        dR = pin.exp3(rotvec)
        measured = np.asarray(pin.log3(dR), dtype=float)
        corrected = pin.exp3(measured * JOYCON_ROTATION_VECTOR_SIGN)
        corrected_rotvec = np.asarray(pin.log3(corrected), dtype=float)
        np.testing.assert_allclose(
            corrected_rotvec,
            rotvec * JOYCON_ROTATION_VECTOR_SIGN,
            atol=1e-10,
        )

    def test_rotation_fix_only_permutates_axes_without_direction_signs(self):
        # 第一次真机校正已经确认轴对应：raw Y→Roll/X, raw Z→Pitch/Y, raw X→Yaw/Z。
        observed_to_target = (
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
            ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        )
        for raw_axis, target_axis in observed_to_target:
            raw_rotation = axis_rotation(raw_axis, 0.2)
            corrected = JOYCON_ROTATION_FIX.T @ raw_rotation @ JOYCON_ROTATION_FIX
            expected = axis_rotation(target_axis, 0.2)
            np.testing.assert_allclose(corrected, expected, atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(JOYCON_ROTATION_FIX), 1.0)


class TestJoyConPoseAdapter(unittest.TestCase):
    def test_translation_does_not_require_rotation_clutch(self):
        clock = FakeClock()
        source = FakeSource(snapshot(clutch=False, stick_y=1.0))
        adapter = JoyConPoseAdapter(source, translation_speed=1.0, clock=clock)

        clock.advance(0.05)
        released = adapter.get_state()
        source.state = snapshot(clutch=True, stick_y=1.0)
        clock.advance(0.05)
        engaged = adapter.get_state()

        np.testing.assert_allclose(released.position, [0.05, 0, 0])
        np.testing.assert_allclose(engaged.position, [0.10, 0, 0])
        self.assertEqual(released.grip, 0.0)
        self.assertEqual(engaged.grip, 1.0)
        self.assertTrue(released.translation_without_clutch)

    def test_standard_mode_positive_stick_x_moves_positive_y(self):
        clock = FakeClock()
        source = FakeSource(snapshot(stick_x=1.0))
        adapter = JoyConPoseAdapter(source, translation_speed=1.0, clock=clock)
        clock.advance(0.05)
        state = adapter.get_state()
        np.testing.assert_allclose(state.position, [0.0, 0.05, 0.0])

    def test_rejects_unknown_control_mode(self):
        with self.assertRaises(ValueError):
            JoyConPoseAdapter(FakeSource(snapshot()), mode="space_position")

    def test_imu_motion_total_clutch_freezes_and_restarts_from_zero_velocity(self):
        clock = FakeClock()
        source = FakeSource(snapshot(clutch=False, acceleration=[1.0, 0.0, 0.0]))
        adapter = JoyConPoseAdapter(
            source,
            mode=JOYCON_MODE_IMU_MOTION,
            translation_speed=10.0,
            accel_lowpass_tau=0.0,
            accel_deadzone=0.0,
            velocity_damping=0.0,
            stationary_time=1.0,
            clock=clock,
        )

        clock.advance(0.04)
        released = adapter.get_state()
        source.state = snapshot(clutch=True, acceleration=[1.0, 0.0, 0.0])
        clock.advance(0.04)
        engaged = adapter.get_state()  # 按下边沿只清零并重新锚定。
        clock.advance(0.04)
        moving = adapter.get_state()

        np.testing.assert_allclose(released.position, np.zeros(3))
        np.testing.assert_allclose(engaged.position, np.zeros(3))
        np.testing.assert_allclose(adapter.virtual_velocity, [0.04, 0.0, 0.0])
        np.testing.assert_allclose(moving.position, [0.0016, 0.0, 0.0])

        source.state = snapshot(clutch=False, acceleration=[5.0, 0.0, 0.0])
        clock.advance(0.04)
        frozen = adapter.get_state()
        np.testing.assert_allclose(frozen.position, moving.position)
        np.testing.assert_allclose(adapter.virtual_velocity, np.zeros(3))
        self.assertEqual(frozen.grip, 0.0)

        source.state = snapshot(clutch=True, acceleration=[5.0, 0.0, 0.0])
        clock.advance(0.04)
        adapter.get_state()
        np.testing.assert_allclose(adapter.virtual_velocity, np.zeros(3))

    def test_imu_motion_lowpass_deadzone_damping_and_speed_limit(self):
        clock = FakeClock()
        source = FakeSource(snapshot(clutch=True, acceleration=[1.0, 0.0, 0.0]))
        adapter = JoyConPoseAdapter(
            source,
            mode=JOYCON_MODE_IMU_MOTION,
            translation_speed=0.2,
            accel_lowpass_tau=0.04,
            accel_deadzone=0.5,
            velocity_damping=4.0,
            stationary_time=1.0,
            clock=clock,
        )

        clock.advance(0.04)
        adapter.get_state()  # 离合按下边沿。
        clock.advance(0.02)
        adapter.get_state()  # 低通后仍在死区内。
        np.testing.assert_allclose(adapter.virtual_velocity, np.zeros(3))
        clock.advance(0.04)
        adapter.get_state()
        speed_before_damping = float(np.linalg.norm(adapter.virtual_velocity))
        self.assertGreater(speed_before_damping, 0.0)

        source.state = snapshot(
            clutch=True,
            acceleration=[0.0, 0.0, 0.0],
            angular_velocity=[1.0, 0.0, 0.0],
        )
        adapter.accel_lowpass_tau = 0.0
        adapter.accel_deadzone = 0.0
        clock.advance(0.04)
        adapter.get_state()
        self.assertLess(np.linalg.norm(adapter.virtual_velocity), speed_before_damping)

        source.state = snapshot(clutch=True, acceleration=[100.0, 0.0, 0.0])
        clock.advance(0.04)
        adapter.get_state()
        self.assertLessEqual(np.linalg.norm(adapter.virtual_velocity), 0.2 + 1e-12)

    def test_imu_motion_stationary_update_and_abnormal_dt_zero_velocity(self):
        clock = FakeClock()
        source = FakeSource(snapshot(clutch=True, acceleration=[2.0, 0.0, 0.0]))
        adapter = JoyConPoseAdapter(
            source,
            mode=JOYCON_MODE_IMU_MOTION,
            translation_speed=1.0,
            accel_lowpass_tau=0.0,
            accel_deadzone=0.0,
            velocity_damping=0.0,
            stationary_time=0.08,
            max_dt=0.05,
            clock=clock,
        )

        clock.advance(0.04)
        adapter.get_state()
        clock.advance(0.04)
        moving = adapter.get_state()
        self.assertGreater(np.linalg.norm(adapter.virtual_velocity), 0.0)

        source.state = snapshot(clutch=True)
        clock.advance(0.04)
        adapter.get_state()
        clock.advance(0.04)
        stopped = adapter.get_state()
        np.testing.assert_allclose(adapter.virtual_velocity, np.zeros(3))

        source.state = snapshot(clutch=True, acceleration=[10.0, 0.0, 0.0])
        clock.advance(1.0)
        protected = adapter.get_state()
        np.testing.assert_allclose(protected.position, stopped.position)
        np.testing.assert_allclose(adapter.virtual_velocity, np.zeros(3))

    def test_switching_modes_resets_imu_motion_state(self):
        clock = FakeClock()
        source = FakeSource(snapshot(clutch=True, acceleration=[1.0, 0.0, 0.0]))
        adapter = JoyConPoseAdapter(
            source,
            mode=JOYCON_MODE_IMU_MOTION,
            translation_speed=1.0,
            accel_lowpass_tau=0.0,
            accel_deadzone=0.0,
            velocity_damping=0.0,
            stationary_time=1.0,
            clock=clock,
        )
        clock.advance(0.04)
        adapter.get_state()
        clock.advance(0.04)
        adapter.get_state()
        self.assertGreater(np.linalg.norm(adapter.virtual_velocity), 0.0)

        adapter.set_mode(JOYCON_MODE_STANDARD)
        self.assertEqual(adapter.mode, JOYCON_MODE_STANDARD)
        np.testing.assert_allclose(adapter.virtual_velocity, np.zeros(3))

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
        fake_pinocchio = SimpleNamespace(
            rpy=FakeRpy(),
            log3=lambda rotation: np.array([
                0.0,
                0.0,
                np.arctan2(rotation[1, 0], rotation[0, 0]),
            ]),
            exp3=lambda vector: rotation_z(float(np.asarray(vector)[2])),
        )
        with patch.dict(sys.modules, {"pinocchio": fake_pinocchio}):
            controller = VRArmController(
                FakeArm(),
                receiver,
                state_getter=receiver.get_state,
                position_fix=np.ones(3),
                rotation_fix=np.eye(3),
                remap_rotation_rpy=False,
                rotation_vector_sign=JOYCON_ROTATION_VECTOR_SIGN,
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
        np.testing.assert_allclose(
            held_rotation,
            rotation_z(np.pi / 2 * JOYCON_ROTATION_VECTOR_SIGN[2]),
            atol=1e-8,
        )

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
        np.testing.assert_allclose(
            controllers["left"].kwargs["rotation_fix"], JOYCON_ROTATION_FIX
        )
        np.testing.assert_allclose(
            controllers["right"].kwargs["rotation_fix"], JOYCON_ROTATION_FIX
        )
        np.testing.assert_allclose(
            controllers["left"].kwargs["rotation_vector_sign"], JOYCON_ROTATION_VECTOR_SIGN
        )
        np.testing.assert_allclose(
            controllers["right"].kwargs["rotation_vector_sign"], JOYCON_ROTATION_VECTOR_SIGN
        )
        self.assertFalse(controllers["left"].kwargs["remap_rotation_rpy"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

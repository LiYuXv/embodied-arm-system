"""Offline regression checks for zero-torque gravity selection in the 200 Hz loop."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from el_a3_sdk.interface import ELA3Interface
from el_a3_sdk.protocol import ArmState, ModeState, RunMode


class TestZeroTorqueControlLoop(unittest.TestCase):
    def setUp(self):
        self.arm = ELA3Interface(can_name="vcan0")
        self.arm._driver.get_feedback = Mock(return_value=SimpleNamespace(
            is_valid=True, position=0.25, velocity=0.0,
        ))
        self.arm._driver.send_motion_control = Mock()
        self.arm._read_feedback_positions = Mock(return_value=[0.0] * 6)
        self.arm._compute_gravity_vector = Mock(return_value=[1.0] * 6)
        self.arm._zero_torque_mode = True
        self.arm._first_command = False

    def test_control_loop_sends_gravity_only_when_selected(self):
        self.arm.SetGravityFeedforwardRatio(0.5)
        self.arm._zero_torque_gravity_enabled = False
        self.arm._control_loop_tick(0.005)
        plain = self.arm._driver.send_motion_control.call_args_list
        self.assertEqual([call.args[-1] for call in plain], [0.0] * 7)
        self.arm._compute_gravity_vector.assert_not_called()

        self.arm._driver.send_motion_control.reset_mock()
        self.arm._zero_torque_gravity_enabled = True
        self.arm._control_loop_tick(0.005)
        gravity = self.arm._driver.send_motion_control.call_args_list
        self.assertEqual(
            [call.args[-1] for call in gravity],
            [0.5 * self.arm._joint_directions[i] for i in range(1, 7)] + [0.0],
        )
        self.assertEqual([call.args[3] for call in gravity], [0.0] * 7)
        self.arm._compute_gravity_vector.assert_called_once()

    def test_background_gravity_loop_applies_ratio(self):
        self.arm.SetGravityFeedforwardRatio(0.4)
        self.arm._connected = True
        self.arm._zt_gravity_running = True
        self.arm._get_kinematics = Mock(return_value=SimpleNamespace(
            compute_gravity=Mock(return_value=[2.0] * 6),
        ))
        self.arm.GetArmJointMsgs = Mock(return_value=SimpleNamespace(
            to_list=lambda: [0.0] * 7,
        ))
        with patch("el_a3_sdk.interface.time.sleep", side_effect=lambda _: setattr(
            self.arm, "_zt_gravity_running", False,
        )):
            self.arm._zero_torque_gravity_loop([0.05] * 7, 0.005)

        calls = self.arm._driver.send_motion_control.call_args_list
        self.assertEqual(
            [call.args[-1] for call in calls],
            [0.8 * self.arm._joint_directions[i] for i in range(1, 7)] + [0.0],
        )

    def test_joint_override_only_applies_to_zero_torque_gravity(self):
        self.arm.SetGravityFeedforwardRatio(0.8)
        self.arm._per_joint_zero_torque_gravity_ratio = {2: 0.7}
        self.arm._zero_torque_gravity_enabled = True
        self.arm._control_loop_tick(0.005)
        calls = self.arm._driver.send_motion_control.call_args_list
        self.assertEqual(
            [call.args[-1] for call in calls],
            [0.8 * self.arm._joint_directions[1],
             0.7 * self.arm._joint_directions[2]]
            + [0.8 * self.arm._joint_directions[i] for i in range(3, 7)]
            + [0.0],
        )

        self.arm._driver.send_motion_control.reset_mock()
        self.arm._zero_torque_mode = False
        self.arm._control_loop_tick(0.005)
        self.assertEqual(
            [call.args[-1] for call in self.arm._driver.send_motion_control.call_args_list],
            [0.8 * self.arm._joint_directions[i] for i in range(1, 7)],
        )

    def test_joint_ratio_defaults_to_global_and_is_clamped(self):
        arm = ELA3Interface(
            can_name="vcan0", gravity_feedforward_ratio=0.8,
            per_joint_zero_torque_gravity_ratio={2: 0.7, 3: 2.0},
        )
        self.assertEqual(arm._zero_torque_gravity_ratio(1), 0.8)
        self.assertEqual(arm._zero_torque_gravity_ratio(2), 0.7)
        self.assertEqual(arm._zero_torque_gravity_ratio(3), 1.0)

    def test_background_gravity_loop_uses_joint_override(self):
        self.arm.SetGravityFeedforwardRatio(0.8)
        self.arm._per_joint_zero_torque_gravity_ratio = {2: 0.7}
        self.arm._connected = True
        self.arm._zt_gravity_running = True
        self.arm._get_kinematics = Mock(return_value=SimpleNamespace(
            compute_gravity=Mock(return_value=[2.0] * 6),
        ))
        self.arm.GetArmJointMsgs = Mock(return_value=SimpleNamespace(
            to_list=lambda: [0.0] * 7,
        ))
        with patch("el_a3_sdk.interface.time.sleep", side_effect=lambda _: setattr(
            self.arm, "_zt_gravity_running", False,
        )):
            self.arm._zero_torque_gravity_loop([0.05] * 7, 0.005)
        self.assertEqual(
            [call.args[-1] for call in self.arm._driver.send_motion_control.call_args_list],
            [1.6 * self.arm._joint_directions[1],
             1.4 * self.arm._joint_directions[2]]
            + [1.6 * self.arm._joint_directions[i] for i in range(3, 7)]
            + [0.0],
        )

    def test_gravity_api_selects_gravity_in_running_control_loop(self):
        self.arm._connected = True
        self.arm._control_running = True
        self.arm.ZeroTorqueMode = Mock(return_value=True)
        self.assertTrue(self.arm.ZeroTorqueModeWithGravity(True, kd=[0.05] * 7))
        self.arm.ZeroTorqueMode.assert_called_once_with(
            True, kd=0.05, gravity_compensation=True,
        )

    def test_mode_entry_records_gravity_choice(self):
        self.arm._connected = True
        self.arm._control_running = True
        self.arm._state = ArmState.ENABLED
        self.arm._driver.get_feedback.return_value.mode_state = ModeState.MOTOR
        self.arm._driver.get_run_mode = Mock(return_value=int(RunMode.MOTION_CONTROL))
        self.arm._driver.write_parameter = Mock()

        self.assertTrue(self.arm.ZeroTorqueMode(True, kd=0.05))
        self.assertFalse(self.arm._zero_torque_gravity_enabled)
        self.assertTrue(self.arm.ZeroTorqueMode(
            True, kd=0.05, gravity_compensation=True,
        ))
        self.assertTrue(self.arm._zero_torque_gravity_enabled)

    def test_initial_gravity_command_applies_ratio_without_control_loop(self):
        self.arm.SetGravityFeedforwardRatio(0.4)
        self.arm._per_joint_zero_torque_gravity_ratio = {2: 0.7}
        self.arm._connected = True
        self.arm._control_running = False
        self.arm._state = ArmState.ENABLED
        self.arm._driver.get_feedback.return_value.mode_state = ModeState.MOTOR
        self.arm._driver.get_run_mode = Mock(return_value=int(RunMode.MOTION_CONTROL))
        self.arm._driver.write_parameter = Mock()

        self.assertTrue(self.arm.ZeroTorqueMode(
            True, kd=0.05, gravity_torques=[2.0] * 6,
            gravity_compensation=True,
        ))
        calls = self.arm._driver.send_motion_control.call_args_list
        self.assertEqual(
            [call.args[-1] for call in calls],
            [0.8 * self.arm._joint_directions[1],
             1.4 * self.arm._joint_directions[2]]
            + [0.8 * self.arm._joint_directions[i] for i in range(3, 7)]
            + [0.0],
        )


if __name__ == "__main__":
    unittest.main()

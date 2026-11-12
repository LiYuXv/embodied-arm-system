#!/usr/bin/env python3
"""Headless VLA safety and dispatch tests; no SDK instance or hardware I/O."""

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from debugger.backend.vla_controller import DualArmVLAController
from debugger.backend.vla_runtime import (
    LocalSmolVLABackend,
    VLABackendInfo,
    VLAInferenceResult,
    VLAObservationSnapshot,
)
from debugger.vla_features import (
    DualArmActionDispatcher,
    as_action_chunk,
    dual_joint_limit_arrays,
    observation_to_torch,
    sanitize_dual_arm_action,
)


class FakePanel(QObject):
    load_requested = pyqtSignal()
    single_requested = pyqtSignal()
    continuous_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    mode_changed = pyqtSignal(bool)
    checkpoint_changed = pyqtSignal(str)
    checkpoint_path = "unused-local-test-checkpoint"
    task = "pick up the object one by one"
    max_delta_rad = 0.05

    def set_runtime_state(self, **state):
        self.runtime_state = state

    def set_status(self, message, *, error=False):
        self.status = (message, error)


class FakeArm(QObject):
    control_loop_changed = pyqtSignal(bool)
    is_connected = True
    is_enabled = True

    def __init__(self):
        super().__init__()
        self.arm = self
        self.control_loop_running = False
        self.auto_ack_starts = True
        self.commands = []

    def submit_command(self, command, *args):
        self.commands.append((command, args))
        if command == "start_control_loop" and self.auto_ack_starts:
            self.ack_start()

    def ack_start(self):
        self.control_loop_running = True
        self.control_loop_changed.emit(True)


class FakeBackend:
    def load(self, checkpoint):
        return VLABackendInfo(checkpoint, "smolvla_torque", "cpu", 50, 20)

    def infer(self, snapshot):
        return VLAInferenceResult(np.full((50, 14), 0.2, np.float32), 1.0)

    def close(self):
        pass


class OnlineVLATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = FakePanel()
        self.left, self.right = FakeArm(), FakeArm()
        self.logs = []
        self.controller = DualArmVLAController(
            self.panel, self.left, self.right,
            lambda: {name: np.zeros((4, 4, 3), np.uint8) for name in ("cam1", "cam2", "cam3")},
            lambda: np.zeros(14, np.float32),
            lambda: np.zeros(2, np.float32), self.logs.append,
            backend_factory=FakeBackend,
        )

    def tearDown(self):
        self.controller.shutdown()

    def pump_until(self, predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertTrue(predicate())

    def load(self):
        self.controller.load_model()
        self.pump_until(lambda: self.controller._loaded)

    def test_safety_limits_shape_nan_and_delta(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            as_action_chunk(np.zeros((2, 7)))
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            as_action_chunk(np.full((2, 14), np.nan))
        with self.assertRaisesRegex(ValueError, "shape"):
            sanitize_dual_arm_action(np.zeros(7), np.zeros(14), .05)
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            sanitize_dual_arm_action(np.full(14, np.nan), np.zeros(14), .05)
        clamped = sanitize_dual_arm_action(np.full(14, 9), np.zeros(14), 10.0)
        _, upper = dual_joint_limit_arrays()
        self.assertEqual(len(clamped.joint_limit_indices), 14)
        np.testing.assert_allclose(clamped.action, upper, atol=1e-6)

        delta_limited = sanitize_dual_arm_action(
            np.full(14, 0.2), np.zeros(14), 0.05
        )
        self.assertEqual(len(delta_limited.delta_limit_indices), 14)
        np.testing.assert_allclose(delta_limited.action, 0.05, atol=1e-6)

    def test_snapshot_is_validated_copied_and_read_only(self):
        frames = {
            name: np.zeros((4, 6, 3), np.uint8)
            for name in ("cam1", "cam2", "cam3")
        }
        state = np.arange(14, dtype=np.float32)
        torque = np.array([0.1, -0.2], dtype=np.float32)
        snapshot = VLAObservationSnapshot.capture(
            frames, state, torque, "pick the object"
        )

        frames["cam1"].fill(255)
        state.fill(99)
        torque.fill(99)
        self.assertEqual(int(snapshot.frames[0].max()), 0)
        np.testing.assert_array_equal(snapshot.state, np.arange(14))
        np.testing.assert_allclose(snapshot.gripper_torque, [0.1, -0.2])
        self.assertFalse(snapshot.frames[0].flags.writeable)
        self.assertFalse(snapshot.state.flags.writeable)

        with self.assertRaisesRegex(ValueError, "missing camera frame"):
            VLAObservationSnapshot.capture(
                {"cam1": frames["cam1"], "cam2": frames["cam2"]},
                np.zeros(14), np.zeros(2), "task",
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            VLAObservationSnapshot.capture(
                frames, np.zeros((2, 7)), np.zeros(2), "task"
            )
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            VLAObservationSnapshot.capture(
                frames, np.zeros(14), np.array([np.nan, 0]), "task"
            )

    def test_online_bgr_is_converted_to_training_rgb(self):
        self.assertTrue(LocalSmolVLABackend().bgr_images)
        frame = np.array([[[10, 20, 30]]], dtype=np.uint8)
        batch = observation_to_torch({"observation.images.cam1": frame}, bgr_images=True)
        np.testing.assert_allclose(
            batch["observation.images.cam1"][0, :, 0, 0].numpy(),
            np.array([30, 20, 10]) / 255,
        )

    def test_checkpoint_edit_requires_explicit_reload(self):
        self.load()
        self.panel.checkpoint_changed.emit("different-checkpoint")
        self.assertFalse(self.controller._loaded)
        self.assertFalse(self.panel.runtime_state["model_loaded"])
        self.assertIn("请重新加载模型", self.panel.status[0])

    def test_dispatcher_blocks_when_one_arm_not_enabled(self):
        self.right.is_enabled = False
        dispatcher = DualArmActionDispatcher(
            self.left, self.right, execute=True, print_fn=lambda _message: None
        )
        with self.assertRaisesRegex(RuntimeError, "enabled"):
            dispatcher.dispatch(np.zeros(14))
        self.assertEqual(self.left.commands, [])

    def test_dispatcher_blocks_when_one_arm_not_connected(self):
        self.right.is_connected = False
        dispatcher = DualArmActionDispatcher(
            self.left, self.right, execute=True, print_fn=lambda _message: None
        )
        with self.assertRaisesRegex(RuntimeError, "connected"):
            dispatcher.dispatch(np.zeros(14))
        self.assertEqual(self.left.commands, [])
        self.assertEqual(self.right.commands, [])

    def test_dry_run_finishes_twenty_steps_without_hardware(self):
        self.load()
        self.left.control_loop_running = True
        self.right.control_loop_running = True
        self.controller.start()
        self.pump_until(lambda: self.panel.status[0] == "完成 20 步")
        self.assertEqual(self.left.commands, [])
        self.assertEqual(self.right.commands, [])
        self.assertEqual(sum("dry-run" in line for line in self.logs), 20)

    def test_stop_cancels_both_arms_and_prevents_more_commands(self):
        self.load()
        self.controller.dispatcher.execute = True
        self.controller.start(continuous=True)
        self.pump_until(lambda: any(
            command == "joint_ctrl" for command, _ in self.left.commands
        ))
        self.controller.stop()
        self.assertEqual(self.left.commands[-1][0], "cancel_motion")
        self.assertEqual(self.right.commands[-1][0], "cancel_motion")
        count = len(self.left.commands)
        for _ in range(20):
            self.app.processEvents()
            time.sleep(.005)
        self.assertEqual(len(self.left.commands), count)
        left_joint = next(
            args for command, args in self.left.commands if command == "joint_ctrl"
        )
        right_joint = next(
            args for command, args in self.right.commands if command == "joint_ctrl"
        )
        self.assertLessEqual(left_joint[0][0], .05 + 1e-6)
        self.assertEqual(len(left_joint[0]), 7)
        self.assertEqual(len(right_joint[0]), 7)

    def test_execution_requires_both_arms_ready(self):
        self.load()
        self.controller.dispatcher.execute = True
        self.right.is_enabled = False
        self.controller.start()
        self.assertTrue(self.panel.status[1])
        self.assertEqual(self.left.commands, [])

    def test_execution_waits_for_both_control_loops_to_start(self):
        self.load()
        self.left.auto_ack_starts = False
        self.right.auto_ack_starts = False
        self.controller.dispatcher.execute = True
        self.controller.start()
        self.assertEqual(self.controller._inference_count, 0)
        self.assertEqual(self.left.commands[-1][0], "start_control_loop")
        self.assertEqual(self.right.commands[-1][0], "start_control_loop")
        self.assertTrue(self.controller._waiting_for_loop_starts)

        self.left.ack_start()
        self.app.processEvents()
        self.assertEqual(self.controller._inference_count, 0)
        self.assertTrue(self.controller._waiting_for_loop_starts)

        self.right.ack_start()
        self.pump_until(lambda: self.controller._inference_count == 1)
        self.assertFalse(self.controller._waiting_for_loop_starts)
        starting = next(i for i, line in enumerate(self.logs)
                        if "正在启动双臂 control loop" in line)
        ready = next(i for i, line in enumerate(self.logs)
                     if i > starting and "双臂 control loop 已就绪" in line)
        self.assertLess(starting, ready)

    def test_continuous_inference_keeps_control_loops_running(self):
        self.load()
        self.controller.dispatcher.execute = True
        self.controller.start(continuous=True)
        self.pump_until(lambda: self.controller._inference_count == 2, timeout=3)
        self.assertEqual(
            sum(command == "start_control_loop" for command, _ in self.left.commands),
            1,
        )
        self.assertFalse(any(
            command == "stop_control_loop" for command, _ in self.left.commands
        ))
        self.controller.stop()

    def test_stop_during_loop_startup_discards_late_acknowledgements(self):
        self.load()
        for arm in (self.left, self.right):
            arm.auto_ack_starts = False
        self.controller.dispatcher.execute = True
        self.controller.start()
        self.assertTrue(self.controller._waiting_for_loop_starts)
        self.assertEqual(self.controller._inference_count, 0)

        self.controller.stop()
        self.assertEqual(self.left.commands[-1][0], "cancel_motion")
        self.assertEqual(self.right.commands[-1][0], "cancel_motion")
        self.left.ack_start()
        self.right.ack_start()
        self.app.processEvents()
        self.assertEqual(self.controller._inference_count, 0)


if __name__ == "__main__":
    unittest.main()

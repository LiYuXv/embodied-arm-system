#!/usr/bin/env python3
"""Offline MoveJ to streaming joint-control handoff tests."""

import unittest

from debugger.backend.arm_worker import ArmWorker


class FakeArm:
    def __init__(self, *, move_j_result=True):
        self.events = []
        self.moving = False
        self.move_j_result = move_j_result

    def MoveJ(self, positions, *, duration, block):
        self.events.append(("MoveJ", list(positions), block))
        self.moving = bool(self.move_j_result) and not block
        return self.move_j_result

    def GripperCtrl(self, *, gripper_angle):
        self.events.append(("GripperCtrl", gripper_angle))

    def is_moving(self):
        self.events.append(("is_moving",))
        return self.moving

    def cancel_motion(self):
        self.events.append(("cancel_motion",))
        self.moving = False

    def JointCtrlList(self, positions):
        self.events.append(("JointCtrlList", list(positions)))


class MotionHandoffTests(unittest.TestCase):
    def make_worker(self, arm):
        worker = ArmWorker(arm=arm)
        worker._enabled = True
        worker._ensure_control_loop = lambda: None
        return worker

    def test_active_move_j_is_cancelled_before_streaming_joint_command(self):
        arm = FakeArm()
        worker = self.make_worker(arm)
        messages = []
        worker.log_message.connect(messages.append)
        worker._do_move_j([.1] * 7, block=False)
        self.assertTrue(worker._trajectory_playing)

        worker._do_joint_ctrl([.2] * 7)

        self.assertEqual(
            [event[0] for event in arm.events],
            ["MoveJ", "GripperCtrl", "is_moving", "cancel_motion", "JointCtrlList"],
        )
        self.assertFalse(worker._trajectory_playing)
        self.assertIn("切换到流式 joint_ctrl，已取消上一轨迹", messages)

    def test_idle_joint_command_does_not_cancel(self):
        arm = FakeArm()
        worker = self.make_worker(arm)
        worker._do_joint_ctrl([.2] * 7)
        self.assertEqual(
            [event[0] for event in arm.events], ["is_moving", "JointCtrlList"]
        )

    def test_move_j_state_tracks_successful_nonblocking_submission(self):
        for success, block, expected in ((True, False, True), (False, False, False), (True, True, False)):
            with self.subTest(success=success, block=block):
                worker = self.make_worker(FakeArm(move_j_result=success))
                worker._do_move_j([.1] * 7, block=block)
                self.assertEqual(worker._trajectory_playing, expected)


if __name__ == "__main__":
    unittest.main()

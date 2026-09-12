#!/usr/bin/env python3
"""Offline schema, frame and recording tests for dual-arm LeRobot data."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from debugger.backend.dual_arm_lerobot import (
    DUAL_ARM_CAMERA_KEYS,
    DUAL_ARM_FPS,
    DUAL_ARM_STATE_NAMES,
    DualArmLeRobotRecordThread,
    build_dual_arm_frame,
    create_or_resume_dual_arm_dataset,
    make_dual_arm_features,
)
from debugger.widgets.camera_pyqt import LeRobotRecordThread


class FakeDataset:
    def __init__(self):
        self.frames = []
        self.save_calls = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self, **kwargs):
        self.save_calls.append(kwargs)


class FakeDatasetClass:
    create_kwargs = None

    @classmethod
    def create(cls, **kwargs):
        cls.create_kwargs = kwargs
        return FakeDataset()


def sample_images(value=0, shape=(4, 6, 3)):
    return {
        name: np.full(shape, value + index, dtype=np.uint8)
        for index, name in enumerate(DUAL_ARM_CAMERA_KEYS)
    }


class TestDualArmSchema(unittest.TestCase):
    def test_schema_has_exact_keys_shapes_and_joint_order(self):
        features = make_dual_arm_features((480, 640))

        self.assertEqual(
            tuple(features),
            (
                "observation.images.cam1",
                "observation.images.cam2",
                "observation.images.cam3",
                "observation.state",
                "observation.gripper_torque",
                "action",
            ),
        )
        for camera in DUAL_ARM_CAMERA_KEYS:
            feature = features[f"observation.images.{camera}"]
            self.assertEqual(feature["dtype"], "video")
            self.assertEqual(feature["shape"], (480, 640, 3))
        self.assertEqual(features["observation.state"]["shape"], (14,))
        self.assertEqual(features["action"]["shape"], (14,))
        self.assertEqual(features["observation.gripper_torque"]["shape"], (2,))
        self.assertEqual(
            tuple(features["observation.state"]["names"]),
            DUAL_ARM_STATE_NAMES,
        )
        self.assertEqual(
            DUAL_ARM_STATE_NAMES,
            (
                "left_j1", "left_j2", "left_j3", "left_j4",
                "left_j5", "left_j6", "left_gripper",
                "right_j1", "right_j2", "right_j3", "right_j4",
                "right_j5", "right_j6", "right_gripper",
            ),
        )

    def test_dataset_factory_uses_20_hz_and_canonical_features(self):
        FakeDatasetClass.create_kwargs = None
        with tempfile.TemporaryDirectory() as root:
            dataset = create_or_resume_dual_arm_dataset(
                root,
                "offline_dual",
                image_size=(4, 6),
                fps=20,
                dataset_cls=FakeDatasetClass,
            )

        self.assertIsInstance(dataset, FakeDataset)
        kwargs = FakeDatasetClass.create_kwargs
        self.assertEqual(kwargs["fps"], DUAL_ARM_FPS)
        self.assertEqual(kwargs["robot_type"], "ela3_dual_arm")
        self.assertEqual(
            kwargs["features"]["observation.state"]["shape"], (14,)
        )
        self.assertIn("observation.images.cam3", kwargs["features"])

    def test_resume_rejects_non_20_hz_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            info_path = Path(root) / "offline_dual" / "meta" / "info.json"
            info_path.parent.mkdir(parents=True)
            info_path.write_text(
                json.dumps({
                    "fps": 30,
                    "features": make_dual_arm_features((4, 6)),
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fps must be 20"):
                create_or_resume_dual_arm_dataset(
                    root,
                    "offline_dual",
                    image_size=(4, 6),
                    dataset_cls=FakeDatasetClass,
                )


class TestDualArmFrame(unittest.TestCase):
    def test_frame_constructor_preserves_fixed_order_and_copies_vectors(self):
        state = np.arange(14, dtype=np.float64)
        action = state + 100
        torque = [1.25, -2.5]
        images = sample_images()
        frame = build_dual_arm_frame(
            images, state, action, torque, "pick object"
        )

        self.assertEqual(
            tuple(frame),
            (
                "observation.images.cam1",
                "observation.images.cam2",
                "observation.images.cam3",
                "observation.state",
                "observation.gripper_torque",
                "action",
                "task",
            ),
        )
        np.testing.assert_array_equal(
            frame["observation.state"], np.arange(14, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            frame["action"], np.arange(14, dtype=np.float32) + 100
        )
        np.testing.assert_array_equal(
            frame["observation.gripper_torque"], np.array(torque, dtype=np.float32)
        )
        self.assertEqual(frame["task"], "pick object")
        state[:] = -1
        self.assertEqual(frame["observation.state"][0], 0.0)


class TestOfflineRecording(unittest.TestCase):
    def test_dual_arm_writer_uses_future_joint_state_as_action(self):
        dataset = FakeDataset()
        writer = DualArmLeRobotRecordThread(
            dataset, fps=20, action_lookahead=2, max_queue=10
        )
        writer.set_task_name("dual task")
        for index in range(5):
            state = np.arange(14, dtype=np.float32) + index * 100
            writer.push_frame(
                sample_images(index),
                state=state,
                gripper_torque=[index + 0.1, index + 0.2],
            )
        writer.stop(flush=True, max_remaining=10)
        writer.run()

        self.assertEqual(len(dataset.frames), 5)
        self.assertEqual(dataset.save_calls, [{"parallel_encoding": True}])
        expected_actions = [200, 300, 400, 400, 400]
        for index, expected_first in enumerate(expected_actions):
            frame = dataset.frames[index]
            self.assertEqual(frame["action"][0], expected_first)
            self.assertEqual(frame["observation.state"][0], index * 100)
            self.assertEqual(frame["observation.gripper_torque"].shape, (2,))
            self.assertEqual(frame["task"], "dual task")
            self.assertIn("observation.images.cam3", frame)

    def test_legacy_single_arm_writer_remains_pose7_and_two_camera(self):
        dataset = FakeDataset()
        end_pose = SimpleNamespace(x=1, y=2, z=3, rx=4, ry=5, rz=6)
        writer = LeRobotRecordThread(
            dataset,
            pose_callback=lambda: (end_pose, 0.7),
            torque_callback=lambda: 0.8,
            fps=20,
            action_lookahead=1,
        )
        writer.set_task_name("single task")
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        writer.push_frame(image, image)
        writer.push_frame(image, image)
        writer.stop(flush=True, max_remaining=10)
        writer.run()

        self.assertEqual(len(dataset.frames), 2)
        frame = dataset.frames[0]
        self.assertEqual(frame["observation.state"].shape, (7,))
        self.assertEqual(frame["action"].shape, (7,))
        self.assertIn("observation.images.cam1", frame)
        self.assertIn("observation.images.cam2", frame)
        self.assertNotIn("observation.images.cam3", frame)
        self.assertIn("observation.gripper_torque", frame)


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""No-hardware regression for repeated VLA inference on one Python thread."""

import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PyQt6.QtCore import QCoreApplication, QObject, QTimer, pyqtSlot

from debugger.backend.vla_inference_worker import VLAInferenceWorker
from debugger.backend.vla_runtime import (
    LocalSmolVLABackend,
    VLABackendInfo,
    VLAObservationSnapshot,
)
from debugger.vla_features import observation_to_torch


class TinyPolicy:
    def reset(self):
        pass

    def predict_action_chunk(self, batch):
        assert batch["observation.images.cam1"].shape == (1, 3, 4, 4)
        np.testing.assert_allclose(
            batch["observation.images.cam1"][0, :, 0, 0].numpy(),
            np.array([30, 20, 10]) / 255,
        )
        return torch.full((1, 50, 14), 0.1, dtype=torch.float32)


class RecordingBackend(LocalSmolVLABackend):
    def __init__(self, calls):
        super().__init__(device="cpu")
        self.calls = calls
        self.calls.append(("construct", threading.get_ident(), id(self)))

    def load(self, checkpoint):
        self.calls.append(("load", threading.get_ident(), id(self)))
        inputs = {
            f"observation.images.cam{i}": {"shape": [3, 4, 4]}
            for i in range(1, 4)
        }
        inputs["observation.state"] = {"shape": [14]}
        inputs["observation.gripper_torque"] = {"shape": [2]}
        self.config = SimpleNamespace(
            type="smolvla_torque", input_features=inputs,
            output_features={"action": {"shape": [14]}},
        )
        self.policy = TinyPolicy()
        self.preprocessor = lambda batch: batch
        self.postprocessor = lambda action: action
        self._torch = torch
        self.info = VLABackendInfo(checkpoint, "smolvla_torque", "cpu", 50, 20)
        return self.info

    def infer(self, snapshot):
        self.calls.append(("infer", threading.get_ident(), id(self)))
        time.sleep(0.02)
        return super().infer(snapshot)

    def close(self):
        self.calls.append(("close", threading.get_ident(), id(self)))
        super().close()


class Receiver(QObject):
    def __init__(self):
        super().__init__()
        self.loaded = []
        self.results = []
        self.errors = []
        self.shutdown_count = 0
        self.delivery_threads = []

    @pyqtSlot(object)
    def on_loaded(self, info):
        self.delivery_threads.append(threading.get_ident())
        self.loaded.append(info)

    @pyqtSlot(int, object)
    def on_result(self, request_id, result):
        self.delivery_threads.append(threading.get_ident())
        self.results.append((request_id, result))

    @pyqtSlot(str)
    def on_error(self, message):
        self.errors.append(message)

    @pyqtSlot(int, str)
    def on_inference_error(self, request_id, message):
        self.errors.append(f"{request_id}: {message}")

    @pyqtSlot()
    def on_shutdown(self):
        self.shutdown_count += 1


class ExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def pump_until(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertTrue(predicate())

    def test_load_and_twelve_inferences_share_python_thread(self):
        calls = []
        receiver = Receiver()
        worker = VLAInferenceWorker(backend_factory=lambda: RecordingBackend(calls))
        worker.model_loaded.connect(receiver.on_loaded)
        worker.model_load_failed.connect(receiver.on_error)
        worker.inference_completed.connect(receiver.on_result)
        worker.inference_failed.connect(receiver.on_inference_error)
        worker.shutdown_completed.connect(receiver.on_shutdown)
        main_thread = threading.get_ident()
        frames = {name: np.full((4, 4, 3), [10, 20, 30], np.uint8)
                  for name in ("cam1", "cam2", "cam3")}
        snapshot = VLAObservationSnapshot.capture(
            frames, np.zeros(14, np.float32), np.zeros(2, np.float32), "task"
        )

        try:
            worker.load_model("fake-checkpoint")
            self.pump_until(lambda: receiver.loaded or receiver.errors)
            self.assertEqual(receiver.errors, [])
            for request_id in range(12):
                worker.run_inference(request_id, snapshot)
            gui_event_processed = []
            QTimer.singleShot(0, lambda: gui_event_processed.append(True))
            self.pump_until(lambda: gui_event_processed)
            self.assertLess(len(receiver.results), 12)
            self.pump_until(lambda: len(receiver.results) == 12 or receiver.errors)
            self.assertEqual(receiver.errors, [])
            self.assertEqual(
                [request_id for request_id, _ in receiver.results], list(range(12))
            )
            for _, result in receiver.results:
                self.assertEqual(result.action_chunk.shape, (50, 14))
                self.assertTrue(np.isfinite(result.action_chunk).all())
        finally:
            worker.shutdown()
            self.pump_until(lambda: receiver.shutdown_count == 1)

        self.assertEqual([name for name, _, _ in calls].count("construct"), 1)
        self.assertEqual([name for name, _, _ in calls].count("load"), 1)
        self.assertEqual([name for name, _, _ in calls].count("infer"), 12)
        self.assertEqual([name for name, _, _ in calls].count("close"), 1)
        self.assertEqual(len({thread_id for _, thread_id, _ in calls}), 1)
        self.assertNotEqual(calls[0][1], main_thread)
        self.assertEqual(set(receiver.delivery_threads), {main_thread})
        self.assertEqual(len({backend_id for _, _, backend_id in calls}), 1)

    def test_observation_tensors_do_not_share_numpy_storage(self):
        image = np.array([[[10, 20, 30]]], dtype=np.uint8)
        state = np.arange(14, dtype=np.float32)
        torque = np.array([.2, -.2], dtype=np.float32)
        batch = observation_to_torch(
            {
                "observation.images.cam1": image,
                "observation.state": state,
                "observation.gripper_torque": torque,
            },
            bgr_images=True,
        )
        image[:] = 0
        state[:] = 0
        torque[:] = 0
        np.testing.assert_allclose(
            batch["observation.images.cam1"][0, :, 0, 0].numpy(),
            np.array([30, 20, 10]) / 255,
        )
        np.testing.assert_allclose(
            batch["observation.state"][0].numpy(), np.arange(14)
        )
        np.testing.assert_allclose(
            batch["observation.gripper_torque"][0].numpy(), [.2, -.2]
        )


if __name__ == "__main__":
    unittest.main()

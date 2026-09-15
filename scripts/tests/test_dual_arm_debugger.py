#!/usr/bin/env python3
"""Offline tests for the dual-arm GUI, bound workers and three cameras."""

import os
import sys
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from PyQt6.QtCore import QObject, pyqtSignal
    from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
except ModuleNotFoundError:
    # CI images used by the SDK's offline suite provide PyQt5.  Its scoped
    # enum/API surface is sufficient for this smoke test, so expose a local
    # import shim without changing production modules or installing packages.
    from PyQt5 import QtCore, QtGui, QtWidgets

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = QtCore
    pyqt6.QtGui = QtGui
    pyqt6.QtWidgets = QtWidgets
    sys.modules.update({
        "PyQt6": pyqt6,
        "PyQt6.QtCore": QtCore,
        "PyQt6.QtGui": QtGui,
        "PyQt6.QtWidgets": QtWidgets,
    })
    QObject = QtCore.QObject
    pyqtSignal = QtCore.pyqtSignal
    QApplication = QtWidgets.QApplication
    QMainWindow = QtWidgets.QMainWindow
    QWidget = QtWidgets.QWidget

from debugger.backend.arm_worker import ArmWorker
from debugger.main_window import MainWindow
from debugger.window_coordinator import WindowCoordinator
from debugger.widgets.joycon_panel import JoyConPanel
from debugger.widgets.triple_camera_panel import CameraStreamWidget, TripleCameraPanel
from el_a3_sdk.joycon import JOYCON_ROTATION_FIX, JOYCON_ROTATION_VECTOR_SIGN


class FakeArm:
    def __init__(self, can_name):
        self.can_name = can_name
        self.connect_calls = 0
        self.disconnect_calls = 0

    def ConnectPort(self):
        self.connect_calls += 1
        return True

    def DisconnectPort(self):
        self.disconnect_calls += 1


class FakeJointVector:
    def __init__(self, values):
        self.values = list(values)

    def to_list(self, include_gripper=True):
        return list(self.values if include_gripper else self.values[:6])


class FakeManager:
    def __init__(self):
        self.arms = {
            "left_arm": FakeArm("can0"),
            "right_arm": FakeArm("can1"),
        }

    def has_arm(self, name):
        return name in self.arms

    def __getitem__(self, name):
        return self.arms[name]


class FakeCapture:
    def __init__(self, _index):
        self.released = False
        self.properties = []

    def isOpened(self):
        return True

    def set(self, key, value):
        self.properties.append((key, value))
        return True

    def release(self):
        self.released = True


class FakeCaptureThread(QObject):
    frame_ready = pyqtSignal(int, object)
    error_occurred = pyqtSignal(int, str)

    def __init__(self, index, capture):
        super().__init__()
        self.index = index
        self.capture = capture
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class QtTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])


class TestBoundArmWorker(QtTestCase):
    def test_bound_worker_connects_and_retains_supplied_instance(self):
        arm = FakeArm("can0")
        worker = ArmWorker(arm=arm, arm_name="left_arm", can_name="can0")

        with patch("debugger.utils.can_utils.get_can_state", return_value="UP"):
            worker._do_connect("can0")
        worker._do_disconnect()

        self.assertIs(worker.arm, arm)
        self.assertEqual(arm.connect_calls, 1)
        self.assertEqual(arm.disconnect_calls, 1)
        self.assertFalse(worker.is_connected)

    def test_simulation_supports_connect_enable_motion_refresh_and_disconnect(self):
        worker = ArmWorker(
            arm=FakeArm("can1"), arm_name="right_arm", can_name="can1"
        )
        target = [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.7]

        worker._do_connect("can1", sim_mode=True)
        worker._do_enable()
        worker._do_joint_ctrl(target)
        worker._do_refresh_feedback()

        self.assertTrue(worker.is_connected)
        self.assertTrue(worker.is_enabled)
        self.assertEqual(worker._sim_target, target)
        worker._do_disconnect()
        self.assertFalse(worker.is_connected)
        self.assertFalse(worker.is_enabled)

    def test_two_workers_keep_commands_and_instances_independent(self):
        left, right = FakeArm("can0"), FakeArm("can1")
        left_worker = ArmWorker(arm=left, arm_name="left_arm", can_name="can0")
        right_worker = ArmWorker(arm=right, arm_name="right_arm", can_name="can1")
        left_worker.submit_command("cancel_motion")

        self.assertIs(left_worker.arm, left)
        self.assertIs(right_worker.arm, right)
        self.assertEqual(left_worker._cmd_queue.qsize(), 1)
        self.assertEqual(right_worker._cmd_queue.qsize(), 0)


class TestDualArmWindow(QtTestCase):
    def setUp(self):
        self.manager = FakeManager()
        self.window = MainWindow(
            arm_manager=self.manager,
            sim_mode=True,
            start_workers=False,
        )

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def test_layout_has_two_arms_no_3d_viewer_and_three_named_cameras(self):
        self.assertEqual(set(self.window.workers), {"left_arm", "right_arm"})
        self.assertIs(self.window.left_worker.arm, self.manager["left_arm"])
        self.assertIs(self.window.right_worker.arm, self.manager["right_arm"])
        self.assertFalse(hasattr(self.window, "viewer_3d"))
        self.assertEqual(
            list(self.window.camera_panel.streams), ["cam1", "cam2", "cam3"]
        )
        for panel in self.window.arm_panels.values():
            self.assertEqual(len(panel.joint_panel._spinboxes), 7)
            self.assertEqual(panel.joint_panel._name_labels[-1].text(), "夹爪")
            self.assertEqual(panel.tabs.count(), 9)
            self.assertIs(panel.tabs.widget(0), panel.joint_panel)
            self.assertIs(panel.tabs.widget(1), panel.trajectory_panel)
            self.assertIs(panel.tabs.widget(2), panel.teaching_panel)
            self.assertIs(panel.tabs.widget(3), panel.diagnostics_panel)
            self.assertIs(panel.tabs.widget(4), panel.gripper_panel)
            self.assertIs(panel.tabs.widget(5), panel.gamepad_panel)
            self.assertIs(panel.tabs.widget(6), panel.joycon_panel)
            self.assertIs(panel.tabs.widget(7), panel.vr_panel)
            self.assertIs(panel.tabs.widget(8), panel.kinematics_display)
            self.assertTrue(panel.trajectory_panel.drag_btn.isHidden())
            self.assertTrue(panel.trajectory_panel.sync_btn.isHidden())
            self.assertIsNotNone(panel.monitoring_window)
            self.assertIsNotNone(panel.toolbar.can_combo)
            self.assertIsNotNone(panel.toolbar.can_toggle_btn)
            self.assertEqual(
                panel.toolbar._get_selected_can_name(), panel.can_name
            )

    def test_per_arm_and_global_commands_route_to_the_correct_workers(self):
        left_submit = Mock()
        right_submit = Mock()
        self.window.left_worker.submit_command = left_submit
        self.window.right_worker.submit_command = right_submit

        self.window.left_arm_panel.toolbar.connect_btn.click()
        left_submit.assert_called_once_with("connect", "can0", sim_mode=True)
        right_submit.assert_not_called()

        left_submit.reset_mock()
        self.window._stop_all()
        left_submit.assert_called_once_with("cancel_motion")
        right_submit.assert_called_once_with("cancel_motion")

        left_submit.reset_mock()
        right_submit.reset_mock()
        self.window._emergency_stop_all()
        left_submit.assert_called_once_with("emergency_stop")
        right_submit.assert_called_once_with("emergency_stop")

    def test_full_left_arm_modules_route_only_to_left_worker(self):
        left_submit = Mock()
        right_submit = Mock()
        self.window.left_worker.submit_command = left_submit
        self.window.right_worker.submit_command = right_submit
        panel = self.window.left_arm_panel
        target = [0.1] * 7

        panel.trajectory_panel.move_j_requested.emit(target, 1.5)
        left_submit.assert_called_once_with("move_j", target, 1.5)
        right_submit.assert_not_called()

        left_submit.reset_mock()
        panel.teaching_panel.zero_torque_requested.emit(True)
        left_submit.assert_called_once_with("zero_torque", True)

        left_submit.reset_mock()
        panel.gripper_panel.gripper_command.emit(0.2, 1.0, -1.0)
        left_submit.assert_called_once_with("gripper_ctrl", 0.2, 1.0, -1.0)

        left_submit.reset_mock()
        panel.diagnostics_panel.read_param_requested.emit(3, 7)
        left_submit.assert_called_once_with("read_motor_param", 3, 7)

        left_submit.reset_mock()
        panel.refresh_requested.emit()
        left_submit.assert_called_once_with("refresh_feedback")
        right_submit.assert_not_called()

        left_submit.reset_mock()
        right_panel = self.window.right_arm_panel
        right_panel.trajectory_panel.cancel_requested.emit()
        right_submit.assert_called_once_with("cancel_motion")
        left_submit.assert_not_called()

    def test_dual_record_callbacks_use_left_then_right_and_three_cameras(self):
        self.window.left_arm_panel._last_joint_states = FakeJointVector(range(7))
        self.window.right_arm_panel._last_joint_states = FakeJointVector(range(7, 14))
        self.window.left_arm_panel._last_effort_states = FakeJointVector(
            [0, 0, 0, 0, 0, 0, 1.25]
        )
        self.window.right_arm_panel._last_effort_states = FakeJointVector(
            [0, 0, 0, 0, 0, 0, -2.5]
        )
        for index, stream in enumerate(self.window.camera_panel.streams.values()):
            stream._last_frame = np.full((4, 6, 3), index, dtype=np.uint8)

        np.testing.assert_array_equal(
            self.window._get_dual_arm_state(),
            np.arange(14, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            self.window._get_dual_gripper_torque(),
            np.array([1.25, -2.5], dtype=np.float32),
        )
        self.assertEqual(
            tuple(self.window._get_camera_frames()), ("cam1", "cam2", "cam3")
        )
        self.assertEqual(self.window.lerobot_panel.fps, 20)

    def test_mode_button_requests_single_arm_window(self):
        requested = []
        self.window.mode_switch_requested.connect(requested.append)
        self.window.mode_switch_btn.click()
        self.assertEqual(requested, ["single"])

    def test_dual_vla_preview_splits_action_without_hardware_execution(self):
        left_submit = Mock()
        right_submit = Mock()
        self.window.left_worker.submit_command = left_submit
        self.window.right_worker.submit_command = right_submit

        result = self.window.preview_vla_action_chunk(
            np.arange(28, dtype=np.float32).reshape(2, 14), step=1
        )

        np.testing.assert_array_equal(result["left_action"][1], np.arange(14, 21))
        np.testing.assert_array_equal(result["right_action"][1], np.arange(21, 28))
        self.assertFalse(result["executed"])
        left_submit.assert_not_called()
        right_submit.assert_not_called()


class FakeLegacyCamera(QWidget):
    camera_log = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._lerobot_recording = False

    def set_end_pose_callback(self, callback):
        self.end_pose_callback = callback

    def set_torque_callback(self, callback):
        self.torque_callback = callback

    def set_motion_callback(self, callback):
        self.motion_callback = callback

    def set_vla_takeover_callback(self, callback):
        self.vla_takeover_callback = callback

    def pause_vla_control(self):
        pass

    def resume_vla_control(self):
        pass

    def toggle_lerobot_record(self):
        self._lerobot_recording = not self._lerobot_recording

    def cleanup(self):
        pass


class TestSingleArmWindow(QtTestCase):
    def test_layout_has_no_3d_keeps_joycon_and_switches_to_dual(self):
        # Replace only the camera hardware edge; the window itself now has no
        # VTK/OpenGL dependency and can be constructed directly offscreen.
        with patch(
            "debugger.single_arm_window.CameraWidget", FakeLegacyCamera
        ):
            from debugger.single_arm_window import SingleArmWindow

            window = SingleArmWindow(
                sim_mode=True,
                start_worker=False,
            )
            requested = []
            window.mode_switch_requested.connect(requested.append)
            window.mode_switch_btn.click()

            self.assertEqual(requested, ["dual"])
            self.assertFalse(hasattr(window, "viewer_3d"))
            self.assertFalse(hasattr(window, "viewer_dock"))
            self.assertTrue(window.trajectory_panel.drag_btn.isHidden())
            self.assertTrue(window.trajectory_panel.sync_btn.isHidden())
            self.assertEqual(window.tabs.count(), 8)
            self.assertIs(window.tabs.widget(6), window.joycon_panel)
            self.assertEqual(window.tabs.tabText(6), "Joy-Con")
            window.close()
            self.app.processEvents()


class FakeModeWindow(QMainWindow):
    mode_switch_requested = pyqtSignal(str)

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.closed = False

    def closeEvent(self, event):
        self.closed = True
        event.accept()


class TestWindowCoordinator(QtTestCase):
    def test_switches_windows_in_both_directions(self):
        created = {"single": [], "dual": []}

        def factory(name):
            def create():
                window = FakeModeWindow(name)
                created[name].append(window)
                return window
            return create

        coordinator = WindowCoordinator(
            FakeManager(),
            window_factories={
                "single": factory("single"),
                "dual": factory("dual"),
            },
        )
        coordinator.show("dual")
        dual = coordinator.window
        dual.mode_switch_requested.emit("single")
        self.app.processEvents()

        self.assertTrue(dual.closed)
        self.assertEqual(coordinator.mode, "single")
        single = coordinator.window
        single.mode_switch_requested.emit("dual")
        self.app.processEvents()
        self.assertTrue(single.closed)
        self.assertEqual(coordinator.mode, "dual")
        coordinator.window.close()


class FakeJoyConSource:
    def __init__(self, binding):
        self.binding = binding
        self.running = False

    def snapshot(self):
        return SimpleNamespace(
            valid=True, clutch=False, stick_x=0.0, stick_y=0.0
        )

    def recenter_orientation(self):
        pass


class FakeJoyConAdapter:
    def __init__(self, source):
        self.source = source
        self.exit_requested = False
        self.mode = "standard"

    def set_mode(self, mode):
        self.mode = mode

    def start(self):
        self.source.running = True

    def stop(self):
        self.source.running = False

    def is_connected(self):
        return self.source.running

    def get_state(self):
        return object()


class FakeTeleopArm:
    def __init__(self):
        self.control_loop_rates = []

    def start_control_loop(self, rate_hz):
        self.control_loop_rates.append(rate_hz)


class FakeJoyConController:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scale = kwargs["scale"]
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.__class__.instances.append(self)

    def start(self):
        self.started.set()
        self.stopped.wait(2.0)

    def stop(self):
        self.stopped.set()


class TestSingleArmJoyConPanel(QtTestCase):
    def test_selected_joycon_controls_the_current_single_arm(self):
        discovered = []

        def discover(side):
            discovered.append(side)
            return SimpleNamespace(side=side)

        FakeJoyConController.instances.clear()
        panel = JoyConPanel(
            discovery_func=discover,
            source_factory=FakeJoyConSource,
            adapter_factory=FakeJoyConAdapter,
            controller_factory=FakeJoyConController,
        )
        arm = FakeTeleopArm()
        panel.set_arm(arm)
        panel.set_arm_enabled(True)
        panel.mode_combo.setCurrentIndex(1)
        panel._connect_input()
        self.assertEqual(panel._adapter.mode, "imu_motion")
        panel._start_control()
        controller = FakeJoyConController.instances[-1]
        self.assertTrue(controller.started.wait(1.0))
        self.assertFalse(panel.mode_combo.isEnabled())

        self.assertEqual(discovered, ["left"])
        self.assertIs(controller.kwargs["arm"], arm)
        self.assertIs(controller.kwargs["receiver"], panel._adapter)
        self.assertFalse(controller.kwargs["remap_rotation_rpy"])
        np.testing.assert_allclose(
            controller.kwargs["rotation_fix"], JOYCON_ROTATION_FIX
        )
        np.testing.assert_allclose(
            controller.kwargs["rotation_vector_sign"],
            JOYCON_ROTATION_VECTOR_SIGN,
        )
        self.assertEqual(arm.control_loop_rates, [200.0])
        panel._stop_control()
        self.assertTrue(panel.mode_combo.isEnabled())
        panel.cleanup()
        self.assertTrue(controller.stopped.is_set())


class TestTripleCameraPanel(QtTestCase):
    def test_streams_start_and_stop_independently_without_real_cameras(self):
        captures = []

        def capture_factory(index):
            capture = FakeCapture(index)
            captures.append(capture)
            return capture

        streams = {}

        def stream_factory(name, index):
            stream = CameraStreamWidget(
                name,
                index,
                capture_factory=capture_factory,
                thread_factory=FakeCaptureThread,
            )
            streams[name] = stream
            return stream

        panel = TripleCameraPanel(stream_factory=stream_factory)
        self.assertTrue(streams["cam1"].start_stream())
        self.assertTrue(streams["cam2"].start_stream())

        self.assertTrue(streams["cam1"].is_running)
        self.assertTrue(streams["cam2"].is_running)
        self.assertFalse(streams["cam3"].is_running)
        frames = []
        streams["cam1"].frame_captured.connect(
            lambda name, index, _frame: frames.append((name, index))
        )
        streams["cam1"]._thread.frame_ready.emit(
            0, np.zeros((24, 32, 3), dtype=np.uint8)
        )
        self.app.processEvents()
        self.assertEqual(frames, [("cam1", 0)])
        self.assertIsNotNone(streams["cam1"]._last_pixmap)
        first_thread = streams["cam1"]._thread
        streams["cam1"].stop_stream()
        self.assertTrue(first_thread.stopped)
        self.assertTrue(captures[0].released)
        self.assertTrue(streams["cam2"].is_running)
        panel.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)

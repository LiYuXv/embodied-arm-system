"""Offline regression checks; load only the audited methods, without Qt or CAN."""
import ast
import logging
import threading
import time
import unittest
from queue import Queue, Empty
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def load_methods(path, class_name, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=[ast.ClassDef(name=class_name, bases=[], keywords=[],
                                           body=methods, decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[class_name]


class Signal:
    def __init__(self):
        self.values = []

    def emit(self, *args):
        self.values.append(args)


class Arm:
    def __init__(self, enable_result=True):
        self.enable_result = enable_result
        self.calls = []

    def EnableArm(self):
        self.calls.append("enable")
        return self.enable_result

    def DisableArm(self):
        self.calls.append("disable")
        return True

    def GetArmJointMsgs(self):
        return SimpleNamespace(to_list=lambda: [0.0] * 7)

    def GetArmJointEfforts(self):
        return SimpleNamespace(to_list=lambda: [0.0] * 7)

    def GetMotorStates(self):
        return {}


class DualArmOfflineTest(unittest.TestCase):
    def test_zero_torque_requires_each_arm_enabled(self):
        cls = load_methods("debugger/backend/arm_worker.py", "ArmWorker",
                           {"_do_zero_torque", "_do_zero_torque_gravity"}, {})
        worker = cls()
        calls = []
        arm = SimpleNamespace(
            ZeroTorqueMode=lambda *a, **kw: calls.append("plain") or True,
            ZeroTorqueModeWithGravity=lambda *a, **kw: calls.append("gravity") or True,
        )
        worker.arm = arm
        worker.arm2 = arm
        worker._sim_mode = False
        worker._ZERO_TORQUE_KD = [0.05] * 7
        worker._enabled = False
        worker._arm2_enabled = True
        worker.log_message = Signal()
        worker.error_occurred = Signal()
        worker._do_zero_torque(True)
        worker._do_zero_torque_gravity(True)
        self.assertEqual(calls, ["plain", "gravity"])

    def test_gravity_loop_not_started_after_mode_rejection(self):
        cls = load_methods("el_a3_sdk/interface.py", "ELA3Interface",
                           {"ZeroTorqueModeWithGravity"}, {"threading": SimpleNamespace(
                               Thread=lambda **kw: self.fail("gravity thread started"))})
        sdk = cls()
        sdk._connected = True
        sdk._control_running = False
        sdk.NUM_JOINTS = 7
        sdk._get_kinematics = lambda: object()
        sdk.ZeroTorqueMode = lambda *a, **kw: False
        self.assertFalse(sdk.ZeroTorqueModeWithGravity(True, kd=[0.05] * 7))
        self.assertFalse(getattr(sdk, "_zt_gravity_running", False))

    def test_safety_failure_on_arm1_does_not_skip_arm2(self):
        cls = load_methods("debugger/backend/arm_worker.py", "ArmWorker",
                           {"_do_emergency_stop", "_do_disconnect"}, {})
        worker = cls()
        calls = []
        class FailingArm:
            def EmergencyStop(self):
                calls.append("arm1 stop")
                raise RuntimeError("stop failed")
            def DisableArm(self):
                calls.append("arm1 disable")
                raise RuntimeError("disable failed")
        class WorkingArm:
            def EmergencyStop(self):
                calls.append("arm2 stop")
                return True
            def DisableArm(self):
                calls.append("arm2 disable")
                return True
            def DisconnectPort(self):
                calls.append("arm2 disconnect")
        worker.arm = FailingArm()
        worker.arm2 = WorkingArm()
        worker._sim_mode = False
        worker._connected = worker._arm2_connected = True
        worker._enabled = worker._arm2_enabled = True
        worker.is_enabled = True
        worker.is_connected = True
        worker.error_occurred = Signal()
        worker.enabled_changed = Signal()
        worker.connected_changed = Signal()
        worker.log_message = Signal()
        worker._cancel_gripper_move = lambda: None
        worker._do_emergency_stop()
        worker._do_disconnect()
        self.assertIn("arm2 stop", calls)
        self.assertIn("arm2 disconnect", calls)
        self.assertFalse(worker._arm2_connected)
        self.assertEqual(len(worker.error_occurred.values), 2)

    def test_stop_drains_safety_commands_before_thread_exit(self):
        cls = load_methods("debugger/backend/arm_worker.py", "ArmWorker",
                           {"stop", "_process_commands", "_execute_command"},
                           {"deque": __import__("collections").deque, "Empty": Empty,
                            "logger": logging.getLogger(__name__)})
        worker = cls()
        worker._cmd_queue = Queue()
        worker._cmd_queue.put(("move_j", ([],), {}))
        worker._stopping = False
        worker.is_connected = True
        worker.is_enabled = False
        worker._running = True
        worker.error_occurred = Signal()
        worker._cancel_gripper_move = lambda: None
        calls = []
        worker._do_emergency_stop = lambda: calls.append("emergency_stop")
        def disconnect():
            calls.append("disconnect")
            worker.is_connected = False
        worker._do_disconnect = disconnect
        worker._do_move_j = lambda *args: self.fail("pending motion executed during stop")
        def wait(timeout):
            self.assertEqual(timeout, 3000)
            worker._process_commands()
            return True
        worker.wait = wait
        self.assertTrue(worker.stop())
        self.assertEqual(calls, ["emergency_stop", "disconnect"])
        self.assertFalse(worker._running)

    def test_stop_timeout_is_bounded_and_does_not_claim_exit(self):
        cls = load_methods("debugger/backend/arm_worker.py", "ArmWorker",
                           {"stop", "submit_command"}, {"deque": __import__("collections").deque})
        worker = cls()
        worker._cmd_queue = Queue()
        worker._stopping = False
        worker.is_connected = True
        worker.is_enabled = True
        worker._cancel_gripper_move = lambda: None
        worker.error_occurred = Signal()
        waits = []
        worker.wait = lambda timeout: waits.append(timeout) or False
        self.assertFalse(worker.stop())
        self.assertEqual(waits, [3000])
        self.assertEqual([item[0] for item in worker._cmd_queue.queue],
                         ["emergency_stop", "disconnect", "shutdown"])
        self.assertIn("停机超时", worker.error_occurred.values[0][0])
        self.assertFalse(worker.submit_command("move_j", [1] * 7))
        self.assertEqual([item[0] for item in worker._cmd_queue.queue],
                         ["emergency_stop", "disconnect", "shutdown"])

    def test_shutdown_stops_processing_late_motion(self):
        cls = load_methods("debugger/backend/arm_worker.py", "ArmWorker",
                           {"_process_commands", "_execute_command"},
                           {"Empty": Empty, "logger": logging.getLogger(__name__)})
        worker = cls()
        worker._cmd_queue = Queue()
        worker._cmd_queue.put(("shutdown", (), {}))
        worker._cmd_queue.put(("move_j", ([1] * 7,), {}))
        worker._running = True
        worker.error_occurred = Signal()
        worker._do_move_j = lambda *args: self.fail("motion after shutdown")
        worker._process_commands()
        self.assertFalse(worker._running)
        self.assertEqual(worker._cmd_queue.qsize(), 1)

    def test_concurrent_submit_during_stop_is_rejected(self):
        cls = load_methods("debugger/backend/arm_worker.py", "ArmWorker",
                           {"stop", "submit_command"}, {"deque": __import__("collections").deque})
        worker = cls()
        worker._cmd_queue = Queue()
        worker._stopping = False
        worker.is_connected = False
        worker.is_enabled = False
        worker._cancel_gripper_move = lambda: None
        worker.error_occurred = Signal()
        waiting = threading.Event()
        release = threading.Event()
        def wait(timeout):
            waiting.set()
            self.assertTrue(release.wait(1))
            return False
        worker.wait = wait
        thread = threading.Thread(target=worker.stop)
        thread.start()
        try:
            self.assertTrue(waiting.wait(1))
            self.assertFalse(worker.submit_command("joint_ctrl", [1] * 7))
            self.assertEqual([item[0] for item in worker._cmd_queue.queue], ["shutdown"])
        finally:
            release.set()
            thread.join(1)
        self.assertFalse(thread.is_alive())

    def test_close_event_keeps_window_on_stop_failure(self):
        cls = load_methods("debugger/main_window.py", "MainWindow",
                           {"closeEvent"}, {})
        window = cls()
        logs = []
        window._append_log = logs.append
        window._cleanup_before_close = lambda: False
        window._report_stop_failure = lambda: logs.append("报警")
        event = SimpleNamespace(ignore=lambda: logs.append("拒绝关闭"))
        namespace = cls.closeEvent.__globals__
        namespace["tr"] = lambda key: key
        window.closeEvent(event)
        self.assertIn("拒绝关闭", logs)
        self.assertIn("报警", logs)

    def test_disconnect_false_keeps_failed_arm_connected_and_checks_other_arm(self):
        cls = load_methods("debugger/backend/arm_worker.py", "ArmWorker",
                           {"_do_disconnect"}, {})
        worker = cls()
        calls = []
        worker.arm = SimpleNamespace(
            DisableArm=lambda: calls.append("arm1 disable") or False,
            DisconnectPort=lambda: self.fail("failed arm disconnected"))
        worker.arm2 = SimpleNamespace(
            DisableArm=lambda: calls.append("arm2 disable") or True,
            DisconnectPort=lambda: calls.append("arm2 disconnect"))
        worker._sim_mode = False
        worker._connected = worker._arm2_connected = True
        worker._enabled = worker._arm2_enabled = True
        worker.error_occurred = Signal()
        worker.connected_changed = Signal()
        worker.enabled_changed = Signal()
        worker.log_message = Signal()
        worker._cancel_gripper_move = lambda: None
        worker.is_connected = worker.is_enabled = True
        worker._do_disconnect()
        self.assertEqual(calls, ["arm1 disable", "arm2 disable", "arm2 disconnect"])
        self.assertIsNotNone(worker.arm)
        self.assertTrue(worker._connected)
        self.assertTrue(worker._enabled)
        self.assertFalse(worker._arm2_connected)
        self.assertEqual(worker.connected_changed.values, [(True,)])
        self.assertIn("失能失败", worker.error_occurred.values[0][0])

    def test_failed_partial_enable_with_false_worker_flag_stays_connected(self):
        cls = load_methods("debugger/backend/arm_worker.py", "ArmWorker",
                           {"_do_enable", "_do_disconnect"}, {})
        worker = cls()
        calls = []
        test = self
        class PartlyEnabledArm:
            def EnableArm(self):
                calls.append("arm1 enable partial")
                return False
            def DisableArm(self):
                calls.append("arm1 disable failed")
                return False
            def DisconnectPort(self):
                test.fail("unsafe arm disconnected")
        worker.arm = PartlyEnabledArm()
        worker.arm2 = SimpleNamespace(
            EnableArm=lambda: calls.append("arm2 enable") or True,
            DisableArm=lambda: calls.append("arm2 disable") or True,
            DisconnectPort=lambda: calls.append("arm2 disconnect"))
        worker._connected = worker._arm2_connected = True
        worker._enabled = worker._arm2_enabled = worker._sim_mode = False
        worker._read_joint_positions_for_enable = lambda arm: [0.0] * 7
        worker._check_positions_in_range = lambda positions: []
        worker.enabled_changed = Signal()
        worker.connected_changed = Signal()
        worker.enable_rejected = Signal()
        worker.error_occurred = Signal()
        worker.log_message = Signal()
        worker._cancel_gripper_move = lambda: None
        worker._do_enable()
        self.assertFalse(worker._enabled)
        self.assertTrue(worker._arm2_enabled)
        worker.is_connected = True
        worker.is_enabled = True
        worker._do_disconnect()
        self.assertEqual(calls, ["arm1 enable partial", "arm1 disable failed",
                                 "arm2 enable", "arm1 disable failed",
                                 "arm2 disable", "arm2 disconnect"])
        self.assertIsNotNone(worker.arm)
        self.assertTrue(worker._connected)
        self.assertFalse(worker._enabled)
        self.assertFalse(worker._arm2_connected)
        self.assertEqual(worker.connected_changed.values, [(True,)])
        self.assertEqual(len(worker.error_occurred.values), 2)
        self.assertIn("未确认电机安全失能", worker.error_occurred.values[-1][0])
        self.assertNotIn(("已断开连接",), worker.log_message.values)

    def test_stop_blocks_motion_sources_before_safety_sequence(self):
        cls = load_methods("debugger/backend/arm_worker.py", "ArmWorker",
                           {"stop", "submit_command"}, {"deque": __import__("collections").deque})
        worker = cls()
        worker._cmd_queue = Queue()
        worker._stopping = False
        worker.is_connected = True
        worker.is_enabled = False
        worker._cancel_gripper_move = lambda: None
        worker.error_occurred = Signal()
        observed = []
        def stop_sources():
            observed.append(worker.submit_command("joint_ctrl", [1] * 7))
            observed.append(list(item[0] for item in worker._cmd_queue.queue))
        def wait(timeout):
            observed.append([item[0] for item in worker._cmd_queue.queue])
            worker.is_connected = False
            return True
        worker.wait = wait
        self.assertTrue(worker.stop(before_shutdown=stop_sources))
        self.assertEqual(observed, [False, [],
                                    ["emergency_stop", "disconnect", "shutdown"]])

    def test_window_stops_all_motion_sources_on_cleanup_error(self):
        cls = load_methods("debugger/main_window.py", "MainWindow",
                           {"_stop_motion_sources"}, {})
        window = cls()
        calls = []
        window.joycon_left = SimpleNamespace(cleanup=lambda: calls.append("Joy-Con L"))
        window.joycon_right = SimpleNamespace(cleanup=lambda: calls.append("Joy-Con R"))
        def failed_gamepad():
            calls.append("Gamepad")
            raise RuntimeError("cleanup failed")
        window.gamepad_panel = SimpleNamespace(cleanup=failed_gamepad)
        window.vr_panel = SimpleNamespace(cleanup=lambda: calls.append("VR"))
        window.camera_widget = SimpleNamespace(cleanup=lambda: calls.append("VLA"))
        with self.assertRaisesRegex(RuntimeError, "Gamepad"):
            window._stop_motion_sources()
        self.assertEqual(calls, ["Joy-Con L", "Joy-Con R", "Gamepad", "VR", "VLA"])

    def test_gamepad_cleanup_does_not_resume_vla_during_stop(self):
        cls = load_methods("debugger/main_window.py", "MainWindow",
                           {"_on_gamepad_disconnected_for_vla"}, {})
        window = cls()
        window._vla_interlock_guard = False
        window.worker = SimpleNamespace(_stopping=True)
        window.camera_widget = SimpleNamespace(
            resume_vla_control=lambda: self.fail("VLA resumed during stop"))
        window._on_gamepad_disconnected_for_vla()

    def test_source_cleanup_failure_retains_connection_for_review(self):
        cls = load_methods("debugger/backend/arm_worker.py", "ArmWorker",
                           {"stop"}, {})
        worker = cls()
        worker._cmd_queue = Queue()
        worker._stopping = False
        worker.is_connected = True
        worker.is_enabled = False
        worker._cancel_gripper_move = lambda: None
        worker.error_occurred = Signal()
        queued = []
        def wait(timeout):
            queued.extend(item[0] for item in worker._cmd_queue.queue)
            return True
        worker.wait = wait
        def failed_cleanup():
            raise RuntimeError("source still running")
        self.assertFalse(worker.stop(before_shutdown=failed_cleanup))
        self.assertEqual(queued, ["emergency_stop", "shutdown"])
        self.assertTrue(worker.is_connected)
        self.assertIn("停止运动源失败", worker.error_occurred.values[0][0])

    def test_arm2_only_feedback(self):
        cls = load_methods("debugger/backend/arm_worker.py", "ArmWorker",
                           {"_poll_feedback"}, {"time": time, "logger": logging.getLogger(__name__)})
        worker = cls()
        worker.arm = None
        worker.arm2 = Arm()
        worker._sim_mode = False
        worker._arm2_enabled = True
        worker._slow_poll_counter = 0
        worker.joints_updated2 = Signal()
        worker.efforts_updated2 = Signal()
        worker.motor_feedback_updated2 = Signal()
        worker._trigger_feedback_poll = lambda arm: self.fail("enabled arm should not be polled by command")
        worker._poll_feedback()
        self.assertEqual(len(worker.joints_updated2.values), 1)
        self.assertEqual(len(worker.efforts_updated2.values), 1)

    def test_failed_enable_does_not_mark_arm_enabled(self):
        cls = load_methods("debugger/backend/arm_worker.py", "ArmWorker",
                           {"_do_enable"}, {})
        worker = cls()
        worker.arm = Arm(enable_result=False)
        worker.arm2 = Arm(enable_result=True)
        worker._connected = worker._arm2_connected = True
        worker._enabled = worker._arm2_enabled = worker._sim_mode = False
        worker._read_joint_positions_for_enable = lambda arm: [0.0] * 7
        worker._check_positions_in_range = lambda positions: []
        worker.enabled_changed = Signal()
        worker.enable_rejected = Signal()
        worker.log_message = Signal()
        worker._do_enable()
        self.assertEqual(worker.arm.calls, ["enable", "disable"])
        self.assertFalse(worker._enabled)
        self.assertTrue(worker._arm2_enabled)
        self.assertEqual(worker.enabled_changed.values, [(True,)])
        self.assertEqual(len(worker.enable_rejected.values), 1)

    def test_enable_exception_rolls_back_failing_arm(self):
        cls = load_methods("debugger/backend/arm_worker.py", "ArmWorker",
                           {"_do_enable"}, {})
        worker = cls()
        class RaisingArm(Arm):
            def EnableArm(self):
                self.calls.append("enable")
                raise RuntimeError("partial failure")
        worker.arm = RaisingArm()
        worker.arm2 = Arm()
        worker._connected = worker._arm2_connected = True
        worker._enabled = worker._arm2_enabled = worker._sim_mode = False
        worker._read_joint_positions_for_enable = lambda arm: [0.0] * 7
        worker._check_positions_in_range = lambda positions: []
        worker.enabled_changed = Signal()
        worker.enable_rejected = Signal()
        worker.error_occurred = Signal()
        worker.log_message = Signal()
        worker._do_enable()
        self.assertEqual(worker.arm.calls, ["enable", "disable"])
        self.assertTrue(worker._arm2_enabled)
        self.assertFalse(worker._enabled)

    def test_zero_torque_exit_uses_entry_mode(self):
        theme = SimpleNamespace(theme="test")
        cls = load_methods("debugger/widgets/teaching_panel.py", "TeachingPanel",
                           {"_toggle_zero_torque"}, {
                               "tr": lambda key: key,
                               "ThemeManager": SimpleNamespace(instance=lambda: theme),
                               "SCENE_COLORS": {"test": {"zt_active_bg": "blue"}},
                               "logger": logging.getLogger(__name__),
                           })
        panel = cls()
        panel._zero_torque_active = False
        panel._zero_torque_gravity_active = False
        panel._current_positions = panel._current_positions2 = [0.0] * 7
        panel._has_arm2 = False
        check = SimpleNamespace(checked=True, enabled=True)
        check.isChecked = lambda: check.checked
        check.setEnabled = lambda value: setattr(check, "enabled", value)
        panel.gravity_check = check
        button = SimpleNamespace(setText=lambda value: None, setStyleSheet=lambda value: None,
                                 setObjectName=lambda value: None)
        panel.zt_toggle_btn = button
        panel.zero_torque_requested = Signal()
        panel.zero_torque_gravity_requested = Signal()
        panel.joint_ctrl_requested = Signal()
        panel._toggle_zero_torque()
        check.checked = False  # programmatic change while active
        panel._toggle_zero_torque()
        self.assertEqual(panel.zero_torque_gravity_requested.values, [(True,), (False,)])
        self.assertEqual(panel.zero_torque_requested.values, [])
        self.assertTrue(check.enabled)


if __name__ == "__main__":
    unittest.main()

"""Joy-Con GUI control policy tests without Qt, evdev, CAN or devices."""
import ast
import builtins
import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from el_a3_sdk.joycon import JOYCON_ROTATION_FIX, JOYCON_ROTATION_VECTOR_SIGN


def load_methods(path, class_name, names, namespace=None):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=[ast.ClassDef(name=class_name, bases=[], keywords=[],
                                   body=methods, decorator_list=[])], type_ignores=[])
    ns = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
    return ns[class_name]


class Label:
    def __init__(self):
        self.text = ''
    def setText(self, text):
        self.text = text
    def setEnabled(self, value):
        self.enabled = value
    def currentData(self):
        return 'standard'
    def value(self):
        return 1.0


class Signal:
    def __init__(self):
        self.messages = []
    def emit(self, *values):
        self.messages.append(values)


class Controller:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.stopped = False
    def start(self):
        while not self.stopped:
            threading.Event().wait(0.001)
    def stop(self):
        self.stopped = True


class JoyConGuiOffline(unittest.TestCase):
    def test_panel_module_import_does_not_load_evdev_or_pinocchio(self):
        class StubWidget:
            pass
        qt_core = SimpleNamespace(QTimer=StubWidget, pyqtSignal=lambda *args: object())
        qt_widgets = SimpleNamespace(**{name: StubWidget for name in (
            'QComboBox', 'QDoubleSpinBox', 'QGroupBox', 'QHBoxLayout',
            'QLabel', 'QPushButton', 'QVBoxLayout', 'QWidget')})
        real_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name in ('evdev', 'pinocchio'):
                raise AssertionError(f'eager optional import: {name}')
            return real_import(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location(
            'joycon_panel_import_check', ROOT / 'debugger/widgets/joycon_panel.py')
        with patch.dict(sys.modules, {'PyQt6': SimpleNamespace(),
                                      'PyQt6.QtCore': qt_core, 'PyQt6.QtWidgets': qt_widgets}), \
             patch('builtins.__import__', side_effect=guarded_import):
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'JoyConPanel'))

    def panel(self, side='left'):
        cls = load_methods('debugger/widgets/joycon_panel.py', 'JoyConPanel',
            {'is_connected', 'is_controlling', 'set_arm_state', 'start_control',
             'stop_control', '_run_controller', '_update_buttons', 'connect_input'},
            {'threading': threading, 'np': np, 'JOYCON_MODE_IMU_MOTION': 'imu_motion',
             'JOYCON_ROTATION_FIX': JOYCON_ROTATION_FIX,
             'JOYCON_ROTATION_VECTOR_SIGN': JOYCON_ROTATION_VECTOR_SIGN})
        p = cls()
        p.side = side
        p._adapter = SimpleNamespace(is_connected=lambda: True, get_state=lambda: None)
        p._source = object()
        p._arm = None
        p._arm_connected = p._arm_enabled = False
        p._ready_check = lambda _side: False
        p._claim_check = lambda _side: True
        p._controller_factory = Controller
        p._controller = p._thread = None
        p._running = False
        p.arm_label = p.status_label = p.control_btn = p.connect_btn = Label()
        p.recenter_btn = p.mode_combo = p.scale_spin = p.rotation_scale_spin = Label()
        p.joycon_log = p._finished = Signal()
        return p

    def test_fixed_arm_independent_enable_and_simulation_gate(self):
        left, right = self.panel('left'), self.panel('right')
        arms = [SimpleNamespace(control_loop_running=True), SimpleNamespace(control_loop_running=True)]
        ready = {'left': True, 'right': False}
        for p in (left, right):
            p._ready_check = lambda side: ready[side]
        left.set_arm_state(arms[0], True, True)
        right.set_arm_state(arms[1], True, False)
        self.assertTrue(left.start_control())
        self.assertFalse(right.start_control())
        self.assertIs(left._controller.kwargs['arm'], arms[0])
        self.assertEqual(left._controller.kwargs['input_name'], 'Joy-Con (L)')
        self.assertFalse(left.start_control())  # no duplicate control loop
        self.assertTrue(left.stop_control())
        ready['left'] = False  # simulation or disconnected real arm
        self.assertFalse(left.start_control())
        ready['right'] = True
        right.set_arm_state(arms[1], True, True)
        self.assertTrue(right.start_control())
        self.assertIs(right._controller.kwargs['arm'], arms[1])
        self.assertTrue(right.stop_control())

    def test_claim_then_starts_only_own_sdk_loop(self):
        p = self.panel()
        calls = []
        arm = SimpleNamespace(control_loop_running=False)
        def start_control_loop(*, rate_hz):
            calls.append(rate_hz)
            arm.control_loop_running = True
        arm.start_control_loop = start_control_loop
        p._ready_check = lambda _: True
        p.set_arm_state(arm, True, True)
        p._claim_check = lambda _: False
        self.assertFalse(p.start_control())
        self.assertEqual(calls, [])
        self.assertIsNone(p._thread)
        p._claim_check = lambda _: True
        self.assertTrue(p.start_control())
        self.assertEqual(calls, [200.0])
        self.assertIs(p._controller.kwargs['arm'], arm)
        self.assertTrue(p.stop_control())
        self.assertTrue(p.start_control())
        self.assertEqual(calls, [200.0])
        self.assertTrue(p.stop_control())

    def test_sdk_loop_start_failure_leaves_no_controller_thread(self):
        p = self.panel('right')
        p._ready_check = lambda _: True
        arm = SimpleNamespace(control_loop_running=False,
                              start_control_loop=lambda **_: (_ for _ in ()).throw(RuntimeError('loop failed')))
        p.set_arm_state(arm, True, True)
        self.assertFalse(p.start_control())
        self.assertIsNone(p._thread)
        self.assertFalse(p.is_controlling)
        self.assertIn('loop failed', p.joycon_log.messages[-1][0])
        arm.start_control_loop = lambda **_: None
        self.assertFalse(p.start_control())
        self.assertIsNone(p._thread)
        self.assertIn('仍未运行', p.joycon_log.messages[-1][0])

    def test_right_panel_starts_right_arm_loop_only(self):
        left, right = self.panel('left'), self.panel('right')
        calls = []
        arms = []
        for side in ('left', 'right'):
            arm = SimpleNamespace(control_loop_running=False)
            def start_control_loop(*, rate_hz, arm=arm, side=side):
                calls.append((side, rate_hz))
                arm.control_loop_running = True
            arm.start_control_loop = start_control_loop
            arms.append(arm)
        for panel, arm in ((left, arms[0]), (right, arms[1])):
            panel._ready_check = lambda _: True
            panel.set_arm_state(arm, True, True)
        self.assertTrue(right.start_control())
        self.assertEqual(calls, [('right', 200.0)])
        self.assertFalse(arms[0].control_loop_running)
        self.assertIs(right._controller.kwargs['arm'], arms[1])
        self.assertTrue(right.stop_control())

    def test_join_timeout_keeps_running_state(self):
        p = self.panel()
        p._controller = SimpleNamespace(stop=lambda: None)
        p._thread = SimpleNamespace(join=lambda timeout: None, is_alive=lambda: True)
        p._running = True
        self.assertFalse(p.stop_control())
        self.assertTrue(p.is_controlling)
        self.assertIsNotNone(p._controller)

    def test_connection_failure_or_wrong_side_never_starts_input(self):
        p = self.panel()
        p._adapter = None
        p._discovery = lambda _side: (_ for _ in ()).throw(ImportError('evdev unavailable'))
        self.assertFalse(p.connect_input())
        self.assertIsNone(p._adapter)
        calls = []
        p._discovery = lambda _side: SimpleNamespace(side='right')
        p._source_factory = lambda binding: calls.append(binding)
        self.assertFalse(p.connect_input())
        self.assertEqual(calls, [])

    def test_worker_ready_and_other_sources_are_per_arm(self):
        cls = load_methods('debugger/main_window.py', 'MainWindow',
            {'_joycon_arm_ready', '_joycon_can_claim', '_other_source_can_start',
             '_stop_joycon_inputs', '_on_disconnect_requested', '_on_emergency_stop_requested'}, {})
        w = cls()
        w.worker = SimpleNamespace(_stopping=False, _sim_mode=False, _connected=True,
            _enabled=True, arm=object(), _arm2_connected=True, _arm2_enabled=False,
            arm2=object(), submit_command=lambda cmd: commands.append(cmd))
        commands = []
        w.vr_panel = SimpleNamespace(_ctrl_running=False, _ctrl_thread=None, _ctrl_thread2=None)
        w.gamepad_panel = SimpleNamespace(_ctrl_running=False, _ctrl_thread=None)
        w.camera_widget = SimpleNamespace(_vla_running=False, _vla_control_paused=False)
        w.joycon_left = SimpleNamespace(is_controlling=False, disconnect_input=lambda: True)
        w.joycon_right = SimpleNamespace(is_controlling=False, disconnect_input=lambda: True)
        w._append_log = lambda msg: None
        self.assertTrue(w._joycon_arm_ready('left'))
        self.assertFalse(w._joycon_arm_ready('right'))
        w.worker._arm2_enabled = True
        self.assertTrue(w._joycon_arm_ready('right'))
        w.gamepad_panel._ctrl_running = True
        self.assertFalse(w._joycon_can_claim('left'))
        self.assertTrue(w._joycon_can_claim('right'))
        w.gamepad_panel._ctrl_running = False
        w.camera_widget._vla_running = True
        w.camera_widget._vla_control_paused = True
        self.assertFalse(w._joycon_can_claim('left'))
        self.assertTrue(w._joycon_can_claim('right'))
        w.camera_widget._vla_running = False
        w.joycon_right.is_controlling = True
        self.assertFalse(w._other_source_can_start('vr'))
        self.assertTrue(w._other_source_can_start('gamepad'))
        w.joycon_right.is_controlling = False
        w.worker._sim_mode = True
        self.assertFalse(w._joycon_arm_ready('left'))
        w.worker._sim_mode = False
        w._on_disconnect_requested()
        w._on_emergency_stop_requested()
        self.assertEqual(commands, ['disconnect', 'emergency_stop'])
        w.joycon_right.disconnect_input = lambda: False
        w._on_disconnect_requested()
        w._on_emergency_stop_requested()
        self.assertEqual(commands, ['disconnect', 'emergency_stop', 'emergency_stop'])

    def test_other_motion_sources_reject_joycon_claim_before_side_effects(self):
        for path, class_name, method_name, log_name in (
            ('debugger/widgets/gamepad_panel.py', 'GamepadPanel', '_start_control', 'gamepad_log'),
            ('debugger/widgets/vr_panel.py', 'VrPanel', '_start_control', 'vr_log'),
            ('debugger/widgets/camera_pyqt.py', 'CameraWidget', '_start_vla_inference', 'camera_log'),
        ):
            with self.subTest(source=class_name):
                cls = load_methods(path, class_name, {method_name})
                source = cls()
                source._motion_start_guard = lambda: False
                setattr(source, log_name, Signal())
                getattr(source, method_name)()
                self.assertTrue(getattr(source, log_name).messages)

    def test_vla_single_request_and_chunk_recheck_joycon_claim(self):
        cls = load_methods('debugger/widgets/camera_pyqt.py', 'CameraWidget',
                           {'_on_vla_once_clicked', '_on_single_chunk'})
        source = cls()
        calls = []
        source._motion_start_guard = lambda: False
        source.camera_log = Signal()
        source._vla_running = True
        source._vla_thread = SimpleNamespace(request_single_chunk=lambda: calls.append('request'))
        source._motion_cb = lambda *args: calls.append('motion')
        source._single_active = False
        source._vla_control_paused = True
        source._valid_action_chunk = lambda chunk: calls.append('validate') or True
        source._on_vla_once_clicked()
        source._on_single_chunk(np.zeros((1, 7)), {})
        self.assertEqual(calls, [])
        self.assertEqual(len(source.camera_log.messages), 2)
        source._motion_start_guard = lambda: True
        source.status_label = Label()
        source._on_vla_once_clicked()
        self.assertEqual(calls, ['request'])


if __name__ == '__main__':
    unittest.main()

"""Pure offline checks: AST extraction avoids importing Qt, cameras or hardware."""
import ast
import json
import logging
import math
import os
import time
import sys
import tempfile
import types
import unittest
from collections import deque
from pathlib import Path
from queue import Queue, Empty, Full
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def method(path, cls_name, name, ns):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), 'exec'), ns)
    return ns[name]


def schema_ns():
    path = ROOT / 'debugger/backend/dual_arm_lerobot.py'
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef))
                and (isinstance(n, ast.Assign) or n.name in
                     {'make_dual_arm_features', 'build_dual_arm_frame',
                      'features_are_compatible', '_float_vector'})]
    ns = {'np': np, 'logging': logging}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), 'exec'), ns)
    return ns


class OfflineTest(unittest.TestCase):
    def test_default_base_off_without_reading_private_config(self):
        tree = ast.parse((ROOT / 'debugger/utils/system_config.py').read_text())
        cfg = next(n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_CONFIG' for t in n.targets))
        value = ast.literal_eval(cfg.value)
        self.assertIs(value['mobile_base_enabled'], False)

    def test_features_joint14_and_mobile18(self):
        ns = schema_ns()
        expected14 = tuple([f'left_j{i}' for i in range(1, 7)] + ['left_gripper']
                           + [f'right_j{i}' for i in range(1, 7)] + ['right_gripper'])
        for enabled, names in ((False, expected14),
                               (True, expected14 + ('lift_m', 'vx', 'vy', 'omega'))):
            features = ns['make_dual_arm_features']((480, 640), include_mobile=enabled)
            self.assertEqual(tuple(features), tuple(f'observation.images.cam{i}' for i in (1, 2, 3))
                             + ('observation.state', 'observation.gripper_torque', 'action'))
            self.assertEqual(tuple(features['observation.state']['names']), names)
            self.assertEqual(features['action']['shape'], (len(names),))
            self.assertEqual(features['observation.gripper_torque']['names'],
                             ['left_gripper', 'right_gripper'])
            images = {f'cam{i}': np.zeros((2, 2, 3), np.uint8) for i in (1, 2, 3)}
            frame = ns['build_dual_arm_frame'](images, np.arange(len(names)),
                                               np.arange(len(names)) + 5, [1, 2], 'task',
                                               include_mobile=enabled)
            self.assertEqual(frame['observation.state'].shape, (len(names),))
            self.assertEqual(frame['action'].shape, (len(names),))
            self.assertEqual(frame['observation.gripper_torque'].shape, (2,))

    def test_old_pose14_rejected_without_writes(self):
        ns = schema_ns()
        current = ns['make_dual_arm_features']()
        old = json.loads(json.dumps(current))
        old['observation.state']['names'] = ['state']
        old['action']['names'] = ['action']
        before = json.dumps(old)
        self.assertFalse(ns['features_are_compatible'](old, current))
        self.assertEqual(json.dumps(old), before)
        self.assertFalse(ns['features_are_compatible'](current,
                         ns['make_dual_arm_features'](include_mobile=True)))

    def test_existing_pose14_dataset_is_not_opened_or_changed(self):
        ns = schema_ns()
        class Dataset:
            @classmethod
            def create(cls, **_):
                raise AssertionError('create must not run')
            @classmethod
            def resume(cls, *_args, **_kwargs):
                raise AssertionError('resume must not run')
        packages = {'lerobot': types.ModuleType('lerobot'),
                    'lerobot.datasets': types.ModuleType('lerobot.datasets'),
                    'lerobot.datasets.lerobot_dataset': types.ModuleType('lerobot.datasets.lerobot_dataset')}
        packages['lerobot.datasets.lerobot_dataset'].LeRobotDataset = Dataset
        scope = {'os': os, 'logger': logging.getLogger('test'),
                 'make_dual_arm_features': ns['make_dual_arm_features'],
                 'features_are_compatible': ns['features_are_compatible'],
                 '_resolve_vcodec_for_recording': lambda: 'h264'}
        ensure = method('debugger/widgets/camera_pyqt.py', 'CameraWidget',
                        '_ensure_lerobot_dataset', scope)
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, packages):
            widget = types.SimpleNamespace(dataset_root=root, _lerobot_repo_name='old',
                _lerobot_image_size=(480, 640), _lerobot_fps=20, _pose_dim=lambda: 14)
            path = Path(root) / 'old/meta/info.json'
            path.parent.mkdir(parents=True)
            features = ns['make_dual_arm_features']((480, 640))
            features['observation.state']['names'] = ['state']
            features['action']['names'] = ['action']
            original = json.dumps({'fps': 20, 'robot_type': 'ela3_dual_arm',
                                   'features': features}).encode()
            path.write_bytes(original)
            with self.assertLogs('test', 'ERROR'):
                self.assertEqual(ensure(widget), 'incompatible')
            self.assertEqual(path.read_bytes(), original)

    def test_base_disabled_does_not_construct_or_set_callback(self):
        fn = method('debugger/main_window.py', 'MainWindow', '_init_optional_mobile_base',
                    {'logger': logging.getLogger('test')})
        camera = types.SimpleNamespace(set_mobile_base_callbacks=lambda *_: self.fail('callback'))
        main = types.SimpleNamespace(_mobile_enabled=False, camera_widget=camera)
        fn(main)
        self.assertFalse(hasattr(main, '_chassis_win'))

        tree = ast.parse((ROOT / 'debugger/main_window.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MainWindow')
        connect = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_connect_signals')
        guard = next(n for n in ast.walk(connect) if isinstance(n, ast.If)
                     and any(isinstance(c, ast.Attribute) and c.attr == 'chassis_cmd'
                             for c in ast.walk(n)))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])),
                     '<offline VR signals>', 'exec'), {'self': main})

    def test_base_enabled_uses_panel_and_mobile_callback(self):
        fn = method('debugger/main_window.py', 'MainWindow', '_init_optional_mobile_base',
                    {'logger': logging.getLogger('test')})
        calls = []
        lift = lambda _: None
        velocity = lambda *_: None
        win = types.SimpleNamespace(get_mobile_state=lambda: [1, 2, 3, 4],
                                    set_lift_position_mm=lift, set_external_velocity_si=velocity,
                                    close=lambda: None)
        fake = types.ModuleType('chassis_lift_status')
        fake.create_embedded_panel = lambda: (calls.append('create') or (win, object()))
        main = types.SimpleNamespace(_mobile_enabled=True, _chassis_win=None,
            chassis_panel=None, tabs=types.SimpleNamespace(insertTab=lambda *_: calls.append('tab')),
            camera_widget=types.SimpleNamespace(set_mobile_base_callbacks=lambda *args, **kw: calls.append((args, kw))),
            _get_mobile_state=lambda: [1, 2, 3, 4])
        with patch.dict(sys.modules, {'chassis_lift_status': fake}):
            fn(main)
        self.assertEqual(calls[:2], ['create', 'tab'])
        self.assertEqual(calls[2], ((main._get_mobile_state, lift, velocity), {'required': True}))

    def test_original_vr_components_and_conditional_signal_connections(self):
        tree = ast.parse((ROOT / 'debugger/widgets/vr_panel.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VrPanel')
        names = {target.id for node in cls.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name)}
        self.assertTrue({'chassis_cmd', 'lift_vel_cmd', 'hmd_cmd', 'hmd_released'} <= names)
        attrs = {n.attr for n in ast.walk(cls) if isinstance(n, ast.Attribute)}
        self.assertTrue({'_hpos_scale_slider', '_hspeed_slider', '_lift_speed_spin',
                         '_poll_hmd_chassis'} <= attrs)
        self.assertTrue({'set_lift_provider', 'set_chassis_pose_provider', '_poll_hmd_chassis'} <=
                        {n.name for n in cls.body if isinstance(n, ast.FunctionDef)})
        main_tree = ast.parse((ROOT / 'debugger/main_window.py').read_text())
        main_cls = next(n for n in main_tree.body if isinstance(n, ast.ClassDef) and n.name == 'MainWindow')
        connect = next(n for n in main_cls.body if isinstance(n, ast.FunctionDef) and n.name == '_connect_signals')
        guard = next(n for n in ast.walk(connect) if isinstance(n, ast.If)
                     and any(isinstance(c, ast.Attribute) and c.attr == 'chassis_cmd'
                             for c in ast.walk(n)))
        signals = {}
        for name in ('chassis_cmd', 'lift_vel_cmd', 'hmd_cmd', 'hmd_released'):
            signals[name] = types.SimpleNamespace(connect=lambda fn, name=name: calls.append((name, fn.__name__)))
        calls = []
        main = types.SimpleNamespace(_mobile_enabled=True, _chassis_win=object(),
                                     vr_panel=types.SimpleNamespace(**signals))
        for name in ('_on_vr_chassis_cmd', '_on_vr_lift_vel_cmd', '_on_vr_hmd_cmd', '_on_vr_hmd_released'):
            setattr(main, name, lambda *_, _name=name: None)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])),
                     '<offline VR signals>', 'exec'), {'self': main})
        self.assertEqual([name for name, _ in calls], list(signals))
        calls.clear()
        main._mobile_enabled = False
        exec(compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])),
                     '<offline VR signals>', 'exec'), {'self': main})
        main._mobile_enabled = True
        main._chassis_win = None
        exec(compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])),
                     '<offline VR signals>', 'exec'), {'self': main})
        self.assertEqual(calls, [])

    def test_vr_dispatch_and_ready_use_existing_base_only(self):
        path = 'debugger/main_window.py'
        logger = logging.getLogger('test')
        fns = {name: method(path, 'MainWindow', name, {'math': math, 'logger': logger})
               for name in ('_on_vr_chassis_cmd', '_on_vr_lift_vel_cmd', '_on_vr_hmd_cmd',
                            '_on_vr_hmd_released', '_on_chassis_ready')}
        calls = []
        class Base:
            def __getattr__(self, name):
                return lambda *args, **kw: calls.append((name, args, kw))
        base = Base()
        vr = types.SimpleNamespace(set_lift_provider=lambda fn: calls.append(('lift_provider', fn)),
                                   set_chassis_pose_provider=lambda fn: calls.append(('pose_provider', fn)))
        main = types.SimpleNamespace(_mobile_enabled=True, _chassis_win=base, vr_panel=vr,
                                     _append_log=lambda msg: calls.append(('log', msg)))
        fns['_on_vr_chassis_cmd'](main, 1, 2, 3, 4, 5, 6)
        fns['_on_vr_lift_vel_cmd'](main, -1, 200)
        fns['_on_vr_hmd_cmd'](main, 1, 2, 3, .3, 10)
        fns['_on_vr_hmd_released'](main)
        fns['_on_chassis_ready'](main)
        self.assertEqual([entry[0] for entry in calls],
                         ['set_external_direction_pads', 'set_lift_position_mm',
                          'set_lift_velocity', 'set_external_pose_target',
                          'set_lift_position_mm', 'clear_external_pose_target',
                          'clear_lift_target', 'lift_provider', 'pose_provider', 'log'])
        calls.clear()
        main._mobile_enabled = False
        for name, args in (('_on_vr_chassis_cmd', (1, 2, 3, 4, 5, 6)),
                           ('_on_vr_lift_vel_cmd', (-1, 200)),
                           ('_on_vr_hmd_cmd', (1, 2, 3, .3, 10)),
                           ('_on_vr_hmd_released', ()), ('_on_chassis_ready', ())):
            fns[name](main, *args)
        self.assertEqual(calls, [])

    def test_camera_three_callbacks_and_safe_ready_timer(self):
        setter = method('debugger/widgets/camera_pyqt.py', 'CameraWidget',
                        'set_mobile_base_callbacks', {})
        camera = types.SimpleNamespace()
        callbacks = (lambda: None, lambda _: None, lambda *_: None)
        setter(camera, *callbacks, required=True)
        self.assertEqual((camera._mobile_get_state, camera._mobile_lift_cmd,
                          camera._mobile_chassis_vel, camera._mobile_required),
                         (*callbacks, True))
        setter(camera, None, None, None, required=False)
        self.assertEqual((camera._mobile_get_state, camera._mobile_lift_cmd,
                          camera._mobile_chassis_vel, camera._mobile_required),
                         (None, None, None, False))
        tree = ast.parse((ROOT / 'debugger/main_window.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MainWindow')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        init_ui = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Expr)
                       and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                       and n.value.func.attr == '_init_ui')
        connect = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Expr)
                       and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                       and n.value.func.attr == '_connect_signals')
        ready = next(i for i, n in enumerate(init.body) if isinstance(n, ast.If)
                     and any(isinstance(c, ast.Attribute) and c.attr == '_on_chassis_ready'
                             for c in ast.walk(n)))
        self.assertLess(init_ui, connect)
        self.assertLess(connect, ready)
        scheduled = []
        timer = types.SimpleNamespace(singleShot=lambda delay, fn: scheduled.append((delay, fn)))
        for enabled, base in ((False, None), (True, None), (True, object())):
            main = types.SimpleNamespace(_mobile_enabled=enabled, _chassis_win=base,
                                         _on_chassis_ready=lambda: None)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[init.body[ready]], type_ignores=[])),
                         '<offline ready timer>', 'exec'), {'self': main, 'QTimer': timer})
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0][0], 0)

    def test_joint_feedback_and_mobile_order(self):
        cls = types.SimpleNamespace()
        fn = method('debugger/widgets/camera_pyqt.py', 'CameraWidget', '_get_record_state', {'np': np})
        cls._joint_state_callback = lambda: list(range(14))
        cls._mobile_get_state = None
        self.assertEqual(fn(cls).tolist(), list(range(14)))
        cls._mobile_get_state = lambda: [.2, .3, .4, .5]
        np.testing.assert_allclose(fn(cls)[14:], [.2, .3, .4, .5])
        cls._joint_state_callback = lambda: None
        self.assertIsNone(fn(cls))

    def test_disconnected_base_has_no_record_state(self):
        fn = method('debugger/main_window.py', 'MainWindow', '_get_mobile_state', {})
        main = types.SimpleNamespace(_chassis_win=None)
        with self.assertRaisesRegex(ValueError, '尚未连接'):
            fn(main)
        main._chassis_win = types.SimpleNamespace(connected=False)
        with self.assertRaisesRegex(ValueError, '尚未连接'):
            fn(main)

    def test_vla_rejects_joint14_18_before_motion(self):
        fn = method('debugger/widgets/camera_pyqt.py', 'CameraWidget', '_valid_action_chunk',
                    {'np': np, 'logger': logging.getLogger('test')})
        obj = types.SimpleNamespace(status_label=types.SimpleNamespace(setText=lambda _: None))
        self.assertTrue(fn(obj, np.zeros((2, 7))))
        for dim in (14, 18):
            with self.assertLogs('test', 'ERROR'):
                self.assertFalse(fn(obj, np.zeros((2, dim))))
        source = (ROOT / 'debugger/widgets/camera_pyqt.py').read_text()
        self.assertIn('int(reply.action_dim) != 7', source)
        self.assertIn('Cartesian pose7', source)

    def test_writer_20hz_future_joint_state(self):
        ns = schema_ns()
        class Signal:
            def emit(self, *_):
                pass
        class Thread:
            def __init__(self, *_):
                pass
        tree = ast.parse((ROOT / 'debugger/backend/dual_arm_lerobot.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'DualArmLeRobotRecordThread')
        cls.decorator_list = []
        ns.update(QThread=Thread, pyqtSignal=lambda *_: Signal(), Queue=Queue,
                  Empty=Empty, Full=Full, deque=deque, time=time,
                  logger=logging.getLogger('test'))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                     '<offline writer>', 'exec'), ns)
        class Dataset:
            def __init__(self):
                self.frames = []
            def add_frame(self, frame):
                self.frames.append(frame)
            def save_episode(self, **_):
                pass
        dataset = Dataset()
        writer = ns['DualArmLeRobotRecordThread'](dataset, action_lookahead=5, max_queue=10)
        images = {f'cam{i}': np.zeros((2, 2, 3), np.uint8) for i in (1, 2, 3)}
        for index in range(7):
            writer.push_frame(images, np.full(14, index), [1, 2])
        writer.stop(flush=True, max_remaining=7)
        writer.run()
        self.assertEqual(len(dataset.frames), 7)
        self.assertEqual(dataset.frames[0]['action'][0], 5)
        self.assertEqual(dataset.frames[1]['action'][0], 6)


if __name__ == '__main__':
    unittest.main()

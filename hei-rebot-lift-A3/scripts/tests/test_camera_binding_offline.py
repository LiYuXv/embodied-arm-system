"""Camera role binding checks without Qt, /dev/video, or the real settings file."""

import ast
import logging
import os
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def extracted(path, class_name, names, globals_=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    source = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in source.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name in names]
    cls = ast.ClassDef(name=class_name, bases=[], keywords=[], body=methods,
                       decorator_list=[])
    ns = dict(globals_ or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(path), "exec"), ns)
    return ns[class_name]


class Combo:
    def __init__(self, value=None):
        self.items = []
        self.value_ = value

    def currentData(self):
        return self.value_

    def blockSignals(self, _):
        pass

    def clear(self):
        self.items = []

    def addItem(self, text, value):
        self.items.append((text, value))

    def findData(self, value):
        return next((i for i, (_, v) in enumerate(self.items) if v == value), -1)

    def setCurrentIndex(self, i):
        self.value_ = self.items[i][1]


class FakeDetect:
    def __init__(self):
        self.indices_found = SimpleNamespace(connect=Mock())
        self.start = Mock()

    def isRunning(self):
        return False


class CameraBindingOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = {'_assign_camera_roles', '_set_recording_indices',
                 '_display_order_indices', '_sync_camera_role_combos',
                 'apply_selected_camera_roles', 'apply_camera_roles',
                 '_camera_binding_busy', '_update_port_labels', '_required_cameras_ready',
                 '_stop_capture_for_binding', '_lerobot_push_frame',
                 'refresh_camera_devices', '_on_frame_if_current',
                 'restore_camera_binding_capture', 'prepare_camera_binding_change'}
        cls.Camera = extracted('debugger/widgets/camera_pyqt.py', 'CameraWidget',
                               names, {'logger': logging.getLogger('camera-test'),
                                       'time': __import__('time'),
                                       'CameraDetectThread': FakeDetect})
        cls.Camera.CAMERA_ROLES = ['arm1', 'head', 'arm2']
        cls.Camera.CAMERA_DISPLAY_ROLES = ['arm2', 'head', 'arm1']
        cls.Camera.CAMERA_ROLE_NAMES = {'arm1': '臂1', 'head': '头部', 'arm2': '臂2'}

    def setUp(self):
        self.cfg = {'camera_roles': {}}
        self.c = self.Camera()
        self.c._detected_indices = [2, 4, 6]
        self.c._role_to_idx = {}
        self.c.camera_indices = []
        self.c.capture_res = (640, 480)
        self.c.caps = {}
        self.c.camera_role_combos = {r: Combo() for r in self.c.CAMERA_ROLES}
        self.c._usb_id_of_index = lambda i: {2: '1-1:1.0', 4: '1-2:1.0',
                                            6: '1-3:1.0', 8: '1-4:1.0'}.get(i)
        self.c._usb_port_text = lambda i: f'/dev/video{i} (usb {self.c._usb_id_of_index(i)})'
        self.c._load_camera_settings = lambda: {'roles': self.cfg['camera_roles']}
        self.c._binding_warning = Mock()
        self.c._camera_binding_busy = lambda: False
        self.c.status_label = SimpleNamespace(setText=Mock())
        self.c.port_labels = [SimpleNamespace(setText=Mock()) for _ in range(3)]
        self.c._role_map_text = lambda: str(self.c._role_to_idx)
        self.c._stop_capture_for_binding = Mock(return_value=True)
        self.c.open_camera = Mock()
        self.c.on_indices_found = Mock()
        self.c.latest_frames = {}
        self.c._pending_frames = {}
        self.c._capture_tokens = {}
        self.c._on_frame_received = Mock()

    def config_patches(self, save_ok=True):
        from debugger.utils import system_config
        return (patch.object(system_config, 'load_config', side_effect=lambda: dict(self.cfg)),
                patch.object(system_config, 'save_config',
                             side_effect=lambda cfg: (self.cfg.update(cfg) or True)
                             if save_ok else False))

    def apply(self, selections, save_ok=True):
        for role, idx in selections.items():
            self.c.camera_role_combos[role].value_ = idx
        a, b = self.config_patches(save_ok)
        with a, b:
            return self.c.apply_selected_camera_roles()

    def test_undetected_devices_and_no_implicit_roles(self):
        self.c._detected_indices = []
        self.c._assign_camera_roles([])
        self.c._set_recording_indices()
        self.c._sync_camera_role_combos()
        self.assertEqual(self.c.camera_indices, [])
        self.assertTrue(all(len(x.items) == 1 for x in self.c.camera_role_combos.values()))

    def test_three_selection_persists_usb_and_preserves_recording_order(self):
        self.assertTrue(self.apply({'arm1': 6, 'head': 2, 'arm2': 4}))
        self.assertEqual(self.c.camera_indices, [6, 2, 4])
        self.assertEqual(self.c._display_order_indices(), [4, 2, 6])
        self.assertEqual(self.cfg['camera_roles']['arm1'], {'usb': '1-3:1.0', 'index': 6})
        self.assertEqual(self.c.open_camera.call_count, 3)

    def test_duplicate_device_rejected_before_save(self):
        self.assertFalse(self.apply({'arm1': 2, 'head': 2, 'arm2': 4}))
        self.assertEqual(self.cfg['camera_roles'], {})
        self.c._binding_warning.assert_called_once()

    def test_two_video_nodes_on_one_usb_cannot_fill_two_roles(self):
        self.c._usb_id_of_index = lambda i: {2: 'same-port', 4: 'same-port',
                                            6: 'other-port'}.get(i)
        self.assertFalse(self.apply({'arm1': 2, 'head': 4, 'arm2': 6}))
        self.cfg['camera_roles'] = {'arm1': {'usb': 'same-port', 'index': 2},
                                    'head': {'usb': 'same-port', 'index': 4}}
        self.c._assign_camera_roles([2, 4])
        self.assertEqual(len(self.c._role_to_idx), 1)

    def test_usb_index_drift_and_missing_usb_do_not_steal_role(self):
        self.cfg['camera_roles'] = {'arm1': {'usb': '1-3:1.0', 'index': 6},
                                    'head': {'usb': '1-2:1.0', 'index': 4},
                                    'arm2': {'usb': '1-1:1.0', 'index': 2}}
        self.c._usb_id_of_index = lambda i: {2: '1-3:1.0', 4: '1-2:1.0',
                                            6: '1-1:1.0'}.get(i)
        self.c._assign_camera_roles([2, 4, 6])
        self.c._set_recording_indices()
        self.assertEqual(self.c.camera_indices, [2, 4, 6])
        self.c._assign_camera_roles([4, 6])
        self.c._set_recording_indices()
        self.assertEqual(self.c._role_to_idx, {'head': 4, 'arm2': 6})
        self.assertEqual(self.c.camera_indices, [])
        self.assertEqual(self.cfg['camera_roles']['arm1']['usb'], '1-3:1.0')

    def test_missing_role_defaults_to_keep_but_can_be_unassigned(self):
        self.cfg['camera_roles'] = {'arm1': {'usb': 'absent', 'index': 8}}
        self.c._sync_camera_role_combos()
        self.assertEqual(self.c.camera_role_combos['arm1'].currentData(), 'keep')
        self.assertTrue(self.apply({'arm1': 'keep'}))
        self.assertIn('arm1', self.cfg['camera_roles'])
        self.assertTrue(self.apply({'arm1': None}))
        self.assertNotIn('arm1', self.cfg['camera_roles'])

    def test_unidentified_usb_and_save_failure(self):
        self.c._usb_id_of_index = lambda i: None if i == 2 else 'usb' + str(i)
        self.assertFalse(self.apply({'arm1': 2}))
        self.assertEqual(self.cfg['camera_roles'], {})
        self.c._usb_id_of_index = lambda i: 'usb' + str(i)
        self.assertFalse(self.apply({'arm1': 2}, save_ok=False))
        self.assertEqual(self.c._role_to_idx, {})

    def test_sysfs_identity_requires_usb_interface_path(self):
        parser = extracted('debugger/widgets/camera_pyqt.py', 'CameraWidget',
                           {'_usb_id_of_index'}, {'os': os, 're': re})
        with patch.object(os.path, 'islink', return_value=True), \
             patch.object(os.path, 'realpath', return_value=(
                 '/sys/devices/pci/usb1/1-2/1-2.3:1.0/video4linux/video6')):
            self.assertEqual(parser._usb_id_of_index(6), '1-2.3:1.0')
        with patch.object(os.path, 'islink', return_value=True), \
             patch.object(os.path, 'realpath', return_value=(
                 '/sys/devices/platform/video4linux/video6')):
            self.assertIsNone(parser._usb_id_of_index(6))

    def test_busy_refuses_change(self):
        self.c._camera_binding_busy = lambda: True
        self.assertFalse(self.apply({'arm1': 2}))
        self.assertEqual(self.cfg['camera_roles'], {})

    def test_each_runtime_mode_blocks_binding(self):
        self.c._camera_binding_busy = self.Camera._camera_binding_busy.__get__(self.c)
        fields = ('is_recording', '_lerobot_recording', '_lerobot_record_thread',
                  '_rollout_recording', '_rollout_record_thread', '_vla_running',
                  '_vla_thread', '_single_active')
        for field in fields:
            for name in fields:
                setattr(self.c, name, False if name.endswith('recording')
                        or name in ('is_recording', '_vla_running', '_single_active') else None)
            setattr(self.c, field, True if field.endswith('recording')
                    or field in ('is_recording', '_vla_running', '_single_active') else object())
            self.assertFalse(self.apply({'arm1': 2}), field)

    def test_settings_dialog_reads_same_roles_and_preserves_absent_usb(self):
        dialog_cls = extracted('debugger/widgets/settings_dialog.py', 'SettingsDialog',
                               {'_load_camera_tab', '_collect_camera_roles'},
                               {'load_config': lambda: self.cfg,
                                'CAMERA_ROLE_NAMES': self.Camera.CAMERA_ROLE_NAMES})
        self.cfg['camera_roles'] = {'arm1': {'usb': 'absent', 'index': 8},
                                    'head': {'usb': '1-2:1.0', 'index': 4}}
        combo = Combo()
        combo.items = [('未分配', None), ('臂1', 'arm1'), ('头部', 'head'),
                       ('臂2', 'arm2')]
        dialog = dialog_cls()
        dialog.cam_rows = [('1-2:1.0', 4, combo)]
        dialog._load_camera_tab(self.cfg)
        self.assertEqual(combo.currentData(), 'head')
        combo.setCurrentIndex(3)
        roles, err = dialog._collect_camera_roles()
        self.assertIsNone(err)
        self.assertEqual(roles['arm1']['usb'], 'absent')
        self.assertEqual(roles['arm2']['usb'], '1-2:1.0')
        self.cfg['camera_roles'] = roles
        self.c._assign_camera_roles([2, 4, 6])
        self.assertEqual(self.c._role_to_idx, {'arm2': 4})

    def test_old_thread_exits_before_capture_release(self):
        events = []
        self.c._stop_capture_for_binding = self.Camera._stop_capture_for_binding.__get__(self.c)
        self.c.capture_threads = {2: SimpleNamespace(stop=lambda: events.append('stop') or True,
                                                          deleteLater=lambda: events.append('delete'))}
        self.c.caps = {2: SimpleNamespace(release=lambda: events.append('release'))}
        self.assertTrue(self.c._stop_capture_for_binding())
        self.assertEqual(events, ['stop', 'delete', 'release'])

    def test_stop_timeout_keeps_capture_and_refresh_does_not_detect(self):
        self.c._stop_capture_for_binding = self.Camera._stop_capture_for_binding.__get__(self.c)
        self.c.capture_threads = {2: SimpleNamespace(stop=lambda: False)}
        cap = SimpleNamespace(release=Mock())
        self.c.caps = {2: cap}
        self.c.detect_thread = FakeDetect()
        self.c.btn_camera_refresh = SimpleNamespace(setEnabled=Mock())
        self.assertFalse(self.c.refresh_camera_devices())
        cap.release.assert_not_called()
        self.assertEqual(list(self.c.capture_threads), [2])

    def test_idle_refresh_uses_mock_detector_after_capture_stop(self):
        self.c.detect_thread = FakeDetect()
        self.c.btn_camera_refresh = SimpleNamespace(setEnabled=Mock())
        self.assertTrue(self.c.refresh_camera_devices())
        self.c._stop_capture_for_binding.assert_called_once()
        self.c.detect_thread.start.assert_called_once()

    def test_queued_old_frames_are_dropped_after_reopen(self):
        old, new = object(), object()
        self.c._capture_tokens[2] = new
        self.c._on_frame_if_current(old, 2, 'old-frame')
        self.c._on_frame_if_current(new, 2, 'new-frame')
        self.c._on_frame_received.assert_called_once_with(2, 'new-frame')

    def test_real_frame_semantics_after_remap(self):
        self.c.camera_indices = [6, 2, 4]
        self.c._role_to_idx = {'arm1': 6, 'head': 2, 'arm2': 4}
        self.c.latest_frames = {2: 'head-frame', 4: 'arm2-frame', 6: 'arm1-frame'}
        sink = SimpleNamespace(push_frame=Mock())
        self.c._lerobot_recording = True
        self.c._lerobot_record_thread = sink
        self.c._rollout_recording = False
        self.c._rollout_record_thread = None
        self.c._lerobot_image_size = (1, 1)
        self.c._lerobot_crop_side = None
        self.c._resize_frame = lambda frame, *_args, **_kw: frame
        self.c._lerobot_frame_count = self.c._rollout_frame_count = 0
        self.c._lerobot_push_frame()
        sink.push_frame.assert_called_once_with(
            {'cam1': 'arm1-frame', 'cam2': 'head-frame', 'cam3': 'arm2-frame'})

    def test_video_files_follow_role_order_after_remap(self):
        writers = []
        cv2 = SimpleNamespace(
            CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
            VideoWriter_fourcc=lambda *_: 1,
            VideoWriter=lambda path, *_: writers.append(path) or object())
        recorder = extracted('debugger/widgets/camera_pyqt.py', 'CameraWidget',
                             {'toggle_record'},
                             {'cv2': cv2, 'os': os, 'logger': logging.getLogger('camera-test'),
                              'datetime': SimpleNamespace(now=lambda: SimpleNamespace(
                                  strftime=lambda _: 'timestamp')),
                              'QMessageBox': SimpleNamespace(warning=Mock())})()
        recorder.camera_indices = [6, 2, 4]
        recorder.caps = {i: SimpleNamespace(get=lambda _: 640) for i in (2, 4, 6)}
        recorder.recorders = {}
        recorder.is_recording = False
        recorder.save_dir = '/tmp'
        recorder.btn_record = SimpleNamespace(setText=Mock(), setStyleSheet=Mock())
        recorder.status_label = SimpleNamespace(setText=Mock())
        recorder.toggle_record()
        self.assertEqual([Path(p).name for p in writers],
                         [f'Video_cam{i}_timestamp.mp4' for i in (1, 2, 3)])
        self.assertEqual(list(recorder.recorders), [6, 2, 4])


if __name__ == '__main__':
    unittest.main()

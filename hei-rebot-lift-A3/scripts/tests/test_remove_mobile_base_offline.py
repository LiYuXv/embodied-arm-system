"""Offline schema and dispatch regressions; never import Qt, CAN, or cameras."""
import ast
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAMERA = ROOT / 'debugger/widgets/camera_pyqt.py'
MAIN = ROOT / 'debugger/main_window.py'
VR = ROOT / 'debugger/widgets/vr_panel.py'


def methods(path, name, selected, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    code = ast.Module(body=[ast.ClassDef(name=name, bases=[], keywords=[],
                                         body=body, decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), 'exec'), namespace)
    return namespace[name]


class RemovedMobileBaseOfflineTest(unittest.TestCase):
    def test_dual_pose14_and_single_pose7(self):
        cls = methods(CAMERA, 'LeRobotRecordThread', {'_base_pose7', '_get_pose_dual'},
                      {'np': np, 'logger': logging.getLogger('test')})
        worker = cls()
        pose = types.SimpleNamespace(x=1, y=2, z=3, rx=4, ry=5, rz=6)
        worker.pose_callback = lambda: (pose, 0.4)
        worker.pose2_callback = lambda: (pose, 0.8)
        out = worker._get_pose_dual()
        self.assertEqual(out.shape, (14,))
        np.testing.assert_allclose(out[[6, 13]], [0.4, 0.8])
        worker.pose2_callback = None
        np.testing.assert_array_equal(worker._get_pose_dual()[7:], np.zeros(7))

    def test_fixed_schema_and_legacy_dataset_rejection_without_mutation(self):
        class FakeDataset:
            create = resume = lambda *a, **kw: self.fail('dataset writer called')
        packages = {'lerobot': types.ModuleType('lerobot'),
                    'lerobot.datasets': types.ModuleType('lerobot.datasets'),
                    'lerobot.datasets.lerobot_dataset': types.ModuleType('lerobot.datasets.lerobot_dataset')}
        packages['lerobot.datasets.lerobot_dataset'].LeRobotDataset = FakeDataset
        cls = methods(CAMERA, 'CameraWidget', {'_pose_dim', '_ensure_lerobot_dataset'},
                      {'os': os, 'logger': logging.getLogger('test'),
                       '_resolve_vcodec_for_recording': lambda: 'libx264',
                       'make_dual_arm_features': lambda *_args, **_kw: {},
                       'features_are_compatible': lambda *_args: False})
        widget = cls()
        widget._lerobot_image_size = (480, 480)
        widget._lerobot_repo_name = 'old'
        widget._lerobot_fps = 20
        self.assertEqual(widget._pose_dim(), 14)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, packages):
            widget.dataset_root = tmp
            path = Path(tmp) / 'old/meta/info.json'
            path.parent.mkdir(parents=True)
            features = {'observation.state': {'shape': [18]},
                        'action': {'shape': [18]},
                        'observation.gripper_torque': {'shape': [2]}}
            for i in (1, 2, 3):
                features[f'observation.images.cam{i}'] = {
                    'shape': [480, 480, 3], 'dtype': 'video',
                    'info': {'video.codec': 'h264'}}
            original = json.dumps({'features': features}).encode()
            path.write_bytes(original)
            with self.assertLogs('test', level='ERROR') as logs:
                self.assertEqual(widget._ensure_lerobot_dataset(), 'incompatible')
            self.assertIn('14', ''.join(logs.output))
            self.assertEqual(path.read_bytes(), original)

    def test_historical_actions_never_dispatched(self):
        cls = methods(CAMERA, 'CameraWidget', {'_valid_action_chunk'},
                      {'np': np, 'logger': logging.getLogger('test')})
        widget = cls()
        self.assertTrue(widget._valid_action_chunk(np.zeros((2, 7))))
        for dim in (11, 14, 18):
            with self.assertLogs('test', level='ERROR'):
                self.assertFalse(widget._valid_action_chunk(np.zeros((2, dim))))

    def test_mobile_hardware_is_only_created_in_optional_method(self):
        tree = ast.parse(MAIN.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MainWindow')
        calls = [(n.name, c.func.id) for n in cls.body if isinstance(n, ast.FunctionDef)
                 for c in ast.walk(n) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                 and c.func.id == 'create_embedded_panel']
        self.assertEqual(calls, [('_init_optional_mobile_base', 'create_embedded_panel')])
        self.assertNotIn('from chassis_lift_status import', '\n'.join(
            MAIN.read_text(encoding='utf-8').splitlines()[:40]))

    def test_recording_and_rollout_keep_three_cameras_two_torques(self):
        src = (ROOT / 'debugger/backend/dual_arm_lerobot.py').read_text(encoding='utf-8')
        for i in (1, 2, 3):
            self.assertIn(f'"cam{i}"', src)
        self.assertIn('"shape": (2,)', src)
        self.assertNotIn('extra_state_callback=', src)

    def test_recap_dual_grippers_and_legacy_rejection(self):
        path = ROOT / 'debugger/recap_sidecar.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'gripper_dim_scale')
        module = ast.Module(body=[fn], type_ignores=[])
        ns = {'np': np, 'RecapWeightConfig': object}
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
        cfg = types.SimpleNamespace(gripper_index=-1, negative_weight=0.25)
        scaled = ns['gripper_dim_scale'](np.array([True, False]), 14, cfg)
        self.assertEqual(np.flatnonzero(scaled[0] != 1).tolist(), [6, 13])
        np.testing.assert_array_equal(scaled[1], np.ones(14))
        with self.assertRaises(ValueError):
            ns['gripper_dim_scale'](np.array([True]), 18, cfg)
        sidecar_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                           and n.name == 'SidecarDataset')
        init = next(n for n in sidecar_cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == '__init__')
        self.assertEqual(init.args.defaults[-1].value, 14)


if __name__ == '__main__':
    unittest.main()

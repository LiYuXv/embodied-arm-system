"""No hardware/network: pure contract checks plus Qt with every device entry mocked.

Run the Qt cases with the lerobot Python and QT_QPA_PLATFORM=offscreen.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from debugger.utils.work_mode import camera_roles, normalize_mode, validate_model_contract, load_model_contract
from test_dual_arm_control_offline import load_methods, Signal


def contract(dim=7, cameras=2, state_dim=7, policy='smolvla'):
    return {'type': policy, 'input_features': {
        **{f'observation.images.cam{i}': {'shape': [3, 32, 32]} for i in range(1, cameras+1)},
        'observation.state': {'shape': [state_dim]}},
        'output_features': {'action': {'shape': [dim]}}}


class ModeContractTests(unittest.TestCase):
    def test_mode_defaults_and_camera_roles(self):
        self.assertEqual(normalize_mode(None), 'single')
        self.assertEqual(normalize_mode('bad'), 'single')
        self.assertEqual(camera_roles('single'), ('arm1', 'head'))
        self.assertEqual(camera_roles('dual'), ('arm1', 'head', 'arm2'))

    def test_known_pose7_and_act_pose_plus_torque(self):
        for cfg in (contract(), contract(state_dim=8, policy='act')):
            self.assertIs(validate_model_contract(cfg, 'single'), cfg)

    def test_incompatible_models_fail_before_observation(self):
        invalid = [contract(14, 3, 14), contract(18, 3, 18), contract(7, 3),
                   contract(7, 1), contract(state_dim=8), {},
                   contract() | {'action_space': 'joint7'},
                   contract() | {'work_mode': 'dual'},
                   contract() | {'robot_type': 'ela3_dual_arm'}]
        named = contract()
        named['output_features']['action']['names'] = ['left_j1']
        invalid.append(named)
        torque = contract()
        torque['input_features']['observation.gripper_torque'] = {'shape': [2]}
        invalid.append(torque)
        for cfg in invalid:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                validate_model_contract(cfg, 'single')
        with self.assertRaises(ValueError):
            validate_model_contract(contract(), 'dual')

    def test_checkpoint_directory_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'pretrained_model'
            checkpoint.mkdir()
            path = checkpoint / 'config.json'
            path.write_text(json.dumps(contract()))
            before = path.read_bytes()
            validate_model_contract(load_model_contract(folder), 'single')
            self.assertEqual(path.read_bytes(), before)

    def test_single_backend_never_creates_arm2_even_with_stale_kwargs(self):
        cls = load_methods('debugger/backend/arm_worker.py', 'ArmWorker',
                           {'_do_connect'}, {})
        for mode, expected in (('single', [1]), ('dual', [1, 2])):
            worker = cls()
            worker.work_mode = mode
            worker._connected = worker._arm2_connected = False
            worker.arm = worker.arm2 = None
            ids = []
            worker._create_arm = lambda **kw: ids.append(kw['arm_id']) or object()
            worker._request_initial_feedback = Mock()
            worker.connected_changed = Signal()
            worker.error_occurred = Signal()
            worker.log_message = Signal()
            worker._do_connect('can0', arm2={'can_name': 'can1'})
            self.assertEqual(ids, expected)
            self.assertEqual(worker.error_occurred.values, [])
            if mode == 'single':
                self.assertIsNone(worker.arm2)
                self.assertFalse(worker._arm2_connected)

    def test_single_wrong_can_rejected_without_sdk_creation(self):
        cls = load_methods('debugger/backend/arm_worker.py', 'ArmWorker', {'_do_connect'}, {})
        worker = cls()
        worker.work_mode = 'single'
        worker._connected = worker._arm2_connected = False
        worker._create_arm = Mock(side_effect=AssertionError('SDK must not open'))
        worker.error_occurred = Signal()
        worker._do_connect('can1')
        worker._create_arm.assert_not_called()
        self.assertIn('can0', worker.error_occurred.values[0][0])


QT_AVAILABLE = importlib.util.find_spec('PyQt6') is not None


@unittest.skipUnless(QT_AVAILABLE, 'Qt integration requires the lerobot environment')
class QtModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
        from debugger.main_window import MainWindow
        from debugger.widgets.camera_pyqt import CameraWidget, CameraDetectThread, CloudInferenceThread
        from debugger.backend.arm_worker import ArmWorker
        cls.Main, cls.Camera, cls.Detect, cls.Worker, cls.Cloud = MainWindow, CameraWidget, CameraDetectThread, ArmWorker, CloudInferenceThread

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg_path = Path(self.temp.name) / 'config.json'
        self.roles = {'arm1': {'usb': 'usb-wrist', 'index': 9},
                      'head': {'usb': 'usb-head', 'index': 5},
                      'arm2': {'usb': 'usb-other', 'index': 1}}
        self.cfg_path.write_text(json.dumps({'auto_connect_on_startup': False,
            'mobile_base_enabled': False, 'work_mode': 'single', 'camera_roles': self.roles}))
        patches = [patch('debugger.utils.system_config.CONFIG_PATH', self.cfg_path),
            patch.object(self.Detect, 'start'),
            patch.object(self.Camera, 'open_camera'),
            patch.object(self.Worker, '_create_arm', side_effect=AssertionError('real SDK forbidden')),
            patch('debugger.widgets.toolbar_panel.get_can_state', side_effect=self.can_state),
            patch('debugger.widgets.toolbar_panel.get_can_bitrate', side_effect=self.can_bitrate),
            patch('debugger.widgets.toolbar_panel.setup_can_interface', side_effect=AssertionError('CAN setup forbidden')),
            patch('debugger.widgets.toolbar_panel.shutdown_can_interface', side_effect=AssertionError('CAN shutdown forbidden')),
            patch('PyQt6.QtWidgets.QMessageBox.warning'),
            patch('PyQt6.QtWidgets.QMessageBox.information')]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.allowed_can1 = False
        self.window = self.Main(auto_connect=False)
        self.camera = self.window.camera_widget
        self.addCleanup(self.window.close)
        self.app.processEvents()

    def can_state(self, name):
        if name == 'can1' and not self.allowed_can1:
            self.fail('single-arm UI touched can1')
        return 'DOWN'

    def can_bitrate(self, name):
        self.can_state(name)
        return None

    def bind(self, dual=False):
        if dual:
            self.switch('dual')
        self.camera._usb_id_of_index = lambda idx: {9: 'usb-wrist', 5: 'usb-head', 1: 'usb-other'}.get(idx)
        self.camera.on_indices_found([1, 5, 9])

    def feedback(self):
        self.camera._single_feedback_ready = lambda: True
        pose = SimpleNamespace(x=.1, y=.2, z=.3, rx=.4, ry=.5, rz=.6)
        self.camera._lerobot_get_end_pose = lambda: (pose, .7)
        self.camera._lerobot_get_torque = lambda: .15
        self.camera._lerobot_get_torque2 = Mock(side_effect=AssertionError('single read arm2'))

    def switch(self, mode):
        self.allowed_can1 = mode == 'dual'
        self.window._on_work_mode_requested(mode)
        self.app.processEvents()
        self.assertEqual(self.window.work_mode, mode)

    def test_default_single_layout_and_dual_restore_persisted(self):
        w = self.window
        self.assertEqual(w.worker.work_mode, 'single')
        self.assertTrue(w.toolbar.arm2_row.isHidden())
        self.assertTrue(w.trajectory_panel.movej_group2.isHidden())
        self.assertTrue(w.joint_panel._arm_title_labels[1].isHidden())
        self.assertTrue(w.joycon_right.isHidden())
        self.assertTrue(w.camera_widget.camera_role_combos['arm2'].isHidden())
        self.assertTrue(w.diagnostics_panel.arm_combo.isHidden())
        self.assertTrue(w.vr_panel._target_label2.isHidden())
        self.assertFalse(w.joycon_left.isHidden())
        self.assertEqual(w.main_splitter.count(), 2)
        self.assertEqual(w.vertical_splitter.count(), 2)
        self.switch('dual')
        for widget in (w.toolbar.arm2_row, w.trajectory_panel.movej_group2,
                       w.joycon_right, w.camera_widget.camera_role_combos['arm2'],
                       w.diagnostics_panel.arm_combo, w.vr_panel._target_label2):
            self.assertFalse(widget.isHidden())
        self.assertEqual(w.diagnostics_panel.arm_combo.itemData(1), 1)
        cfg = json.loads(self.cfg_path.read_text())
        self.assertEqual(cfg['work_mode'], 'dual')
        self.assertFalse(cfg['auto_connect_on_startup'])
        self.assertEqual(cfg['camera_roles'], self.roles)
        self.assertEqual(self.camera._rollout_repo_name, 'ela3_rollout')
        self.switch('single')
        self.assertEqual(self.camera._rollout_repo_name, 'ela3_rollout_single')
        self.assertEqual(json.loads(self.cfg_path.read_text())['work_mode'], 'single')

    def test_toolbar_single_connect_does_not_query_arm2(self):
        got = []
        self.window.toolbar.connect_requested.disconnect()
        self.window.toolbar.connect_requested.connect(lambda name, kw: got.append((name, kw)))
        self.window.toolbar.arm2_row.get_config = Mock(side_effect=AssertionError('arm2 queried'))
        self.window.toolbar._on_connect_clicked()
        self.assertEqual(got, [('can0', {'backend': 'socketcan', 'arm2': None})])
        self.assertFalse(self.window.worker.is_connected)

    def test_worker_switch_interlocks(self):
        worker = self.Worker()  # deliberately not started
        self.assertTrue(worker.try_set_work_mode('single'))
        for attr, active, idle in (
            ('_connected', True, False), ('_arm2_connected', True, False),
            ('_enabled', True, False), ('_arm2_enabled', True, False),
            ('_trajectory_playing', True, False), ('_executing_command', 'connect', None),
            ('arm', object(), None), ('arm2', object(), None)):
            with self.subTest(attr=attr):
                setattr(worker, attr, active)
                self.assertFalse(worker.try_set_work_mode('dual'))
                self.assertEqual(worker.work_mode, 'single')
                setattr(worker, attr, idle)
        worker.submit_command('connect', 'can0')
        self.assertFalse(worker.try_set_work_mode('dual'))
        worker._cmd_queue.queue.clear()
        self.assertTrue(worker.try_set_work_mode('dual'))

    def test_ui_interlocks_recording_save_vla_teaching_and_teleop(self):
        cases = [(self.camera, '_lerobot_recording', True, False),
                 (self.camera, '_rollout_recording', True, False),
                 (self.camera, 'is_recording', True, False),
                 (self.camera, '_vla_running', True, False),
                 (self.camera, '_lerobot_record_thread', object(), None),
                 (self.camera, '_rollout_record_thread', object(), None),
                 (self.window.teaching_panel, '_zero_torque_active', True, False),
                 (self.window.gamepad_panel, '_ctrl_running', True, False)]
        for obj, attr, active, idle in cases:
            with self.subTest(attr=attr):
                setattr(obj, attr, active)
                self.window._on_work_mode_requested('dual')
                self.assertEqual(self.window.work_mode, 'single')
                self.assertEqual(self.window.worker.work_mode, 'single')
                self.assertEqual(self.window.toolbar.mode_combo.currentData(), 'single')
                setattr(obj, attr, idle)

    def test_two_camera_usb_roles_and_vla_order(self):
        self.bind()
        self.assertEqual(self.camera._role_to_idx, {'arm1': 9, 'head': 5})
        self.assertEqual(self.camera.camera_indices, [9, 5])
        self.assertEqual(self.camera.open_camera.call_args_list[-2:][0].args[0], 9)
        self.assertNotIn(1, [c.args[0] for c in self.camera.open_camera.call_args_list])
        self.camera.latest_frames = {9: 'wrist', 5: 'head', 1: 'other'}
        self.camera._resize_frame = lambda f, *_a, **_k: f
        self.feedback()
        wrist, head, state = self.camera._vla_get_obs()
        self.assertEqual((wrist, head), ('wrist', 'head'))
        np.testing.assert_allclose(state, [.1, .2, .3, .4, .5, .6, .7])
        np.testing.assert_allclose(self.camera._get_record_torque(), [.15])
        del self.camera._role_to_idx['head']
        self.camera._set_recording_indices()
        self.assertEqual(self.camera.camera_indices, [])
        self.assertIsNone(self.camera._vla_get_obs())

    def test_single_real_feedback_required(self):
        self.feedback()
        self.camera._single_feedback_ready = lambda: False
        self.assertIsNone(self.camera._get_record_state())
        self.camera._single_feedback_ready = lambda: True
        self.camera._lerobot_get_end_pose = lambda: (None, .7)
        self.assertIsNone(self.camera._get_record_state())
        self.camera._lerobot_get_torque = lambda: None
        with self.assertRaises(ValueError):
            self.camera._get_record_torque()

    def test_record_and_rollout_push_use_two_or_three_roles(self):
        for dual in (False, True):
            self.bind(dual)
            self.camera.latest_frames = {9: 'wrist', 5: 'head', 1: 'other'}
            self.camera._resize_frame = lambda f, *_a, **_k: f
            for rollout in (False, True):
                sink = SimpleNamespace(push_frame=Mock())
                self.camera._lerobot_recording = not rollout
                self.camera._rollout_recording = rollout
                self.camera._lerobot_record_thread = None if rollout else sink
                self.camera._rollout_record_thread = sink if rollout else None
                self.camera._lerobot_push_frame()
                expected = {'cam1': 'wrist', 'cam2': 'head'}
                if dual:
                    expected['cam3'] = 'other'
                sink.push_frame.assert_called_once_with(expected)
                self.camera._lerobot_recording = self.camera._rollout_recording = False
                self.camera._lerobot_record_thread = self.camera._rollout_record_thread = None

    def test_record_and_rollout_start_select_single_schema(self):
        self.bind()
        self.feedback()
        self.camera._ensure_capture_resolution = Mock()
        self.camera._ensure_lerobot_dataset = Mock(return_value=object())
        writer = Mock()
        with patch('debugger.widgets.camera_pyqt.DualArmLeRobotRecordThread', return_value=writer) as factory:
            self.camera._start_lerobot_record()
            self.assertTrue(self.camera._lerobot_recording)
            self.assertTrue(factory.call_args.kwargs['single_arm'])
            self.assertFalse(factory.call_args.kwargs['include_mobile'])
            self.camera._lerobot_recording = False
            self.camera._lerobot_record_thread = None
            self.camera._vla_running = True
            self.camera._start_rollout_record()
            self.assertTrue(self.camera._rollout_recording)
            self.assertTrue(factory.call_args.kwargs['single_arm'])
        self.camera._vla_running = False
        self.camera._rollout_recording = False
        self.camera._rollout_record_thread = None
        self.camera._lerobot_dataset = self.camera._rollout_dataset = None
        self.camera._lerobot_timer.stop()

    def test_invalid_model_does_not_start_local_or_cloud(self):
        self.bind()
        self.camera._prepare_cloud_connection = Mock(side_effect=AssertionError('network forbidden'))
        bad = Path(self.temp.name) / 'bad'
        bad.mkdir()
        (bad / 'config.json').write_text(json.dumps(contract(14, 3, 14)))
        self.camera.vla_model_edit.setText(str(bad))
        self.camera.chk_cloud.setChecked(False)
        with patch('debugger.widgets.camera_pyqt.VLAInferenceThread') as thread:
            self.camera._start_vla_inference()
            thread.assert_not_called()
        self.switch('dual')
        self.camera.chk_cloud.setChecked(True)
        self.camera._start_vla_inference()
        self.camera._prepare_cloud_connection.assert_not_called()
        self.assertFalse(self.camera._vla_running)

    def cloud_fixture(self, **kwargs):
        """Real protobufs, fake RPCs only: never open a channel or start a server."""
        from debugger.cloud_inference import vla_inference_pb2 as pb
        codes = SimpleNamespace(**{name: SimpleNamespace(name=name) for name in
                                  ('UNAUTHENTICATED', 'UNAVAILABLE')})

        class RpcError(Exception):
            def code(self):
                return self.args[0]

            def details(self):
                return 'offline injected failure'

        channel = SimpleNamespace(close=Mock())
        grpc = SimpleNamespace(insecure_channel=Mock(return_value=channel),
                               RpcError=RpcError, StatusCode=codes)
        ping = pb.PingReply(policy_type='pi05', model_path='legacy-checkpoint',
                           device='cuda', trained_max_delay=6,
                           image_height=224, image_width=320)
        self.assertNotIn('model_config_json', ping.DESCRIPTOR.fields_by_name)
        stub = SimpleNamespace(Ping=Mock(return_value=ping), InferChunk=Mock())
        obs = Mock(return_value=(np.zeros((8, 8, 3), np.uint8),
                                 np.ones((8, 8, 3), np.uint8), np.arange(7)))
        thread = self.Cloud('offline.invalid:1', 'offline-token', 'task', obs,
                            get_torque=lambda: .25, **kwargs)
        modules = {'grpc': grpc,
                   'cloud_inference': SimpleNamespace(vla_inference_pb2=pb,
                       vla_inference_pb2_grpc=SimpleNamespace(VLAInferenceStub=lambda _: stub))}
        return pb, thread, stub, channel, modules, grpc

    def test_cloud_legacy_ping_accepts_7d_and_rejects_invalid_replies(self):
        valid = np.arange(14, dtype=float).reshape(2, 7) / 100
        cases = [('pose7', 7, 2, valid.ravel()),
                 ('joint14', 14, 2, np.zeros(28)),
                 ('14d_mislabeled_as_7d', 7, 2, np.zeros(28)),
                 ('short', 7, 2, np.zeros(13)),
                 ('long', 7, 2, np.zeros(15)),
                 ('empty', 7, 2, []),
                 ('zero_steps', 7, 0, []),
                 ('negative_steps', 7, -1, []),
                 ('nan_later_frame', 7, 2, [0.] * 13 + [np.nan]),
                 ('inf_later_frame', 7, 2, [0.] * 13 + [np.inf]),
                 ('negative_inf', 7, 2, [0.] * 13 + [-np.inf])]
        for single in (False, True):
            for name, dim, steps, actions in cases:
                with self.subTest(single=single, reply=name):
                    pb, thread, stub, channel, modules, _ = self.cloud_fixture()
                    stub.InferChunk.side_effect = lambda req, **kw: pb.InferenceReply(
                        request_id=req.request_id, action_dim=dim, num_steps=steps,
                        actions=actions)
                    chunks, singles, errors, infos, loaded, statuses = [], [], [], [], [], []
                    thread.chunk_ready.connect(lambda c, dt: chunks.append(c))
                    thread.single_chunk_ready.connect(lambda c, info: singles.append((c, info)))
                    thread.error.connect(errors.append)
                    thread.server_info.connect(infos.append)
                    thread.model_loaded.connect(lambda: loaded.append(True))
                    thread.status.connect(statuses.append)
                    infer = thread._do_infer

                    def infer_and_stop(*args, **kwargs):
                        infer(*args, **kwargs)
                        thread.stop()

                    (thread.request_single_chunk if single else thread.request_chunk)()
                    with patch.dict(sys.modules, modules), \
                         patch.object(thread, '_do_infer', side_effect=infer_and_stop):
                        thread.run()
                    self.assertEqual(errors, [])
                    self.assertEqual(loaded, [True])
                    self.assertEqual(stub.Ping.call_count, 2)  # handshake + heartbeat
                    for call in stub.Ping.call_args_list:
                        self.assertEqual(call.args[0].token, 'offline-token')
                    self.assertEqual((infos[0]['image_height'], infos[0]['image_width']), (224, 320))
                    self.assertTrue(any('动作语义未预先校验' in s for s in statuses))
                    thread.get_obs.assert_called_once()
                    stub.InferChunk.assert_called_once()
                    req = stub.InferChunk.call_args.args[0]
                    self.assertEqual(list(req.state), list(range(7)))
                    self.assertEqual(req.token, 'offline-token')
                    self.assertTrue(req.reset)
                    self.assertEqual(req.is_single, single)
                    self.assertTrue(req.cam1_jpeg and req.cam2_jpeg)
                    expected = int(name == 'pose7')
                    self.assertEqual(len(chunks), expected if not single else 0)
                    self.assertEqual(len(singles), expected if single else 0)
                    if expected:
                        consumed = singles[0][0] if single else chunks[0]
                        np.testing.assert_allclose(consumed, valid)
                        self.assertTrue(self.camera._valid_action_chunk(consumed))
                    else:
                        self.assertTrue(any('动作空间不匹配' in s or '形状异常' in s
                                            or 'NaN/Inf' in s for s in statuses))
                    channel.close.assert_called_once()

    def test_schema_and_writer_future_action_for_both_modes(self):
        from debugger.backend.dual_arm_lerobot import (make_single_arm_features,
            make_dual_arm_features, features_are_compatible, DualArmLeRobotRecordThread)
        for single in (True, False):
            dim, ncam, ntorque = (7, 2, 1) if single else (14, 3, 2)
            schema = make_single_arm_features((32, 32)) if single else make_dual_arm_features((32, 32))
            self.assertEqual(schema['observation.state']['shape'], (dim,))
            self.assertEqual(schema['action']['shape'], (dim,))
            self.assertEqual(schema['observation.gripper_torque']['shape'], (ntorque,))
            self.assertEqual(sum(k.startswith('observation.images.') for k in schema), ncam)
            frames = []
            dataset = SimpleNamespace(add_frame=frames.append, save_episode=Mock())
            writer = DualArmLeRobotRecordThread(dataset, single_arm=single, max_queue=10)
            images = {f'cam{i}': np.zeros((32, 32, 3), np.uint8) for i in range(1, ncam+1)}
            for i in range(7):
                writer.push_frame(images, np.full(dim, i), np.arange(ntorque))
            writer._running = False
            with patch('debugger.backend.dual_arm_lerobot.time.sleep'):
                writer.run()
            self.assertEqual(len(frames), 7)
            np.testing.assert_array_equal(frames[0]['action'], np.full(dim, 5))
            np.testing.assert_array_equal(frames[-1]['action'], np.full(dim, 6))
            self.assertEqual(frames[0]['observation.gripper_torque'].shape, (ntorque,))
            dataset.save_episode.assert_called_once()
            wrong = make_dual_arm_features() if single else make_single_arm_features()
            self.assertFalse(features_are_compatible(wrong, schema))
            with self.assertRaises(ValueError):
                writer.push_frame(images, np.zeros(14 if single else 7), np.zeros(ntorque))
            with self.assertRaises(ValueError):
                writer.push_frame({}, np.zeros(dim), np.zeros(ntorque))

    def test_failed_config_save_rolls_back_mode_and_binding(self):
        before = self.cfg_path.read_bytes()
        with patch('debugger.main_window.save_config', return_value=False):
            self.window._on_work_mode_requested('dual')
        self.assertEqual(self.window.work_mode, 'single')
        self.assertEqual(self.window.worker.work_mode, 'single')
        self.assertEqual(self.camera.work_mode, 'single')
        self.assertEqual(self.camera.CAMERA_ROLES, ['arm1', 'head'])
        self.assertTrue(self.window.toolbar.arm2_row.isHidden())
        self.assertEqual(self.cfg_path.read_bytes(), before)

    def test_missing_head_cannot_be_replaced_by_arm2_for_recording(self):
        self.bind()
        self.camera._role_to_idx = {'arm1': 9, 'arm2': 1}
        self.camera._set_recording_indices()
        self.camera._ensure_lerobot_dataset = Mock()
        self.camera._start_lerobot_record()
        self.camera._ensure_lerobot_dataset.assert_not_called()
        self.assertFalse(self.camera._lerobot_recording)

    def test_local_rtc_and_advantage_flags_are_preserved(self):
        self.bind()
        path = Path(self.temp.name) / 'model'
        path.mkdir()
        (path / 'config.json').write_text(json.dumps(contract()))
        self.camera.vla_model_edit.setText(str(path))
        self.camera.chk_cloud.setChecked(False)
        self.camera.chk_rtc.setChecked(True)
        self.camera.chk_advantage_condition.setChecked(True)
        self.camera._ensure_capture_resolution = Mock()
        self.camera._on_vla_takeover_requested = None
        self.camera._motion_cb = Mock()
        with patch('debugger.widgets.camera_pyqt.VLAInferenceThread') as factory:
            self.camera._start_vla_inference()
            self.assertTrue(self.camera._vla_running)
            self.assertTrue(factory.call_args.kwargs['rtc_mode'])
            self.assertEqual(factory.return_value.advantage_prefix, 'advantage: positive, ')
            factory.return_value.start.assert_called_once()
        self.camera._vla_running = False
        self.camera._vla_thread = None
        self.camera._seq_grip_timer.stop()
        self.camera._rtc_grip_timer.stop()

    def test_real_lerobot_disk_roundtrip_and_wrong_mode_append_rejected(self):
        from debugger.backend.dual_arm_lerobot import DualArmLeRobotRecordThread
        import pyarrow.parquet as pq
        real_import = __import__
        def offline_import(name, *args, **kwargs):
            if name == 'lerobot.datasets.lerobot_dataset':
                raise ImportError('test explicitly uses the offline v3 writer')
            return real_import(name, *args, **kwargs)
        self.camera.dataset_root = self.temp.name
        self.camera._lerobot_image_size = (32, 32)
        for single in (True, False):
            self.camera.work_mode = 'single' if single else 'dual'
            self.camera._lerobot_repo_name = 'single-data' if single else 'dual-data'
            dim, ncam, ntorque = (7, 2, 1) if single else (14, 3, 2)
            # Force the existing offline v3 writer; CPU codec only, all files in /tmp.
            with patch('builtins.__import__', side_effect=offline_import), \
                 patch('debugger.widgets.camera_pyqt._resolve_vcodec_for_recording', return_value='h264'):
                dataset = self.camera._ensure_lerobot_dataset()
                self.assertNotIsInstance(dataset, str)
                errors = []
                writer = DualArmLeRobotRecordThread(dataset, single_arm=single, max_queue=10)
                writer.error.connect(errors.append)
                images = {f'cam{i}': np.full((32, 32, 3), i*20, np.uint8) for i in range(1, ncam+1)}
                for i in range(7):
                    writer.push_frame(images, np.full(dim, i), np.arange(ntorque))
                writer._running = False
                writer.run()
                self.assertEqual(errors, [])
                dataset_dir = Path(self.temp.name) / self.camera._lerobot_repo_name
                info_path = dataset_dir / 'meta/info.json'
                info = json.loads(info_path.read_text())
                self.assertEqual(info['features']['action']['shape'], [dim])
                self.assertEqual(info['features']['observation.gripper_torque']['shape'], [ntorque])
                self.assertEqual(len(list(dataset_dir.glob('videos/**/*.mp4'))), ncam)
                table = pq.read_table(next(dataset_dir.glob('data/**/*.parquet'))).to_pydict()
                np.testing.assert_array_equal(table['action'][0], np.full(dim, 5))
                np.testing.assert_array_equal(table['observation.state'][0], np.zeros(dim))
                resumed = self.camera._ensure_lerobot_dataset()
                self.assertEqual(resumed.num_episodes, 1)
                resumed.finalize()
                self.camera._write_recap_annotations(str(dataset_dir), 0,
                    [{'start': 0, 'end': 6, 'label': 'good', 'is_human_intervention': True}],
                    True, source='rollout', policy_checkpoint='offline-policy', group_id='offline-group')
                annotation = json.loads((dataset_dir / 'meta/reward_annotations.json').read_text())
                episode = annotation['episodes']['0']
                self.assertTrue(episode['episode_success'])
                self.assertEqual(episode['source'], 'rollout')
                self.assertEqual(episode['policy_checkpoint'], 'offline-policy')
                before = info_path.read_bytes()
                self.camera.work_mode = 'dual' if single else 'single'
                self.assertEqual(self.camera._ensure_lerobot_dataset(), 'incompatible')
                self.assertEqual(info_path.read_bytes(), before)
        self.camera.work_mode = 'single'

    def test_incompatible_action_chunks_cannot_reach_execution(self):
        self.camera._motion_cb = Mock()
        for chunk in (np.zeros((2, 14)), np.zeros((2, 18)),
                      np.full((2, 7), np.nan), np.zeros(7)):
            self.assertFalse(self.camera._valid_action_chunk(chunk))
        self.assertTrue(self.camera._valid_action_chunk(np.zeros((2, 7))))
        self.camera._motion_cb.assert_not_called()
        self.switch('dual')
        self.assertFalse(self.camera._valid_action_chunk(np.zeros((2, 7))))

    def test_explicit_single_role_can_reassign_reserved_arm2_usb(self):
        self.bind()
        combo = self.camera.camera_role_combos['arm1']
        combo.setCurrentIndex(combo.findData(1))
        self.assertTrue(self.camera.apply_selected_camera_roles())
        roles = json.loads(self.cfg_path.read_text())['camera_roles']
        self.assertEqual(roles['arm1']['usb'], 'usb-other')
        self.assertEqual(roles['head']['usb'], 'usb-head')
        self.assertNotIn('arm2', roles)
        self.assertEqual(self.camera.camera_indices, [1, 5])


    def test_cloud_legacy_reconnect_resets_and_validates_every_reply(self):
        pb, thread, stub, channel, modules, grpc = self.cloud_fixture()
        valid = np.arange(14, dtype=float).reshape(2, 7)
        stub.InferChunk.side_effect = [
            grpc.RpcError(grpc.StatusCode.UNAVAILABLE),
            pb.InferenceReply(request_id=2, action_dim=14, num_steps=2, actions=np.zeros(28)),
            pb.InferenceReply(request_id=3, action_dim=7, num_steps=2, actions=valid.ravel())]
        lost, restored, chunks, errors = [], [], [], []
        thread.link_lost.connect(lost.append)
        thread.link_restored.connect(lambda: (restored.append(True), thread.request_chunk()))
        thread.chunk_ready.connect(lambda c, dt: chunks.append(c))
        thread.error.connect(errors.append)
        infer = thread._do_infer

        def infer_next(*args, **kwargs):
            infer(*args, **kwargs)
            if stub.InferChunk.call_count == 2:
                self.assertEqual(chunks, [])  # 14D never reaches execution after restart
                thread.request_chunk()
            elif stub.InferChunk.call_count == 3:
                thread.stop()

        thread.request_chunk()
        with patch.dict(sys.modules, modules), \
             patch('debugger.widgets.camera_pyqt.time.monotonic', side_effect=[0., 2., 4.]), \
             patch.object(thread, '_do_infer', side_effect=infer_next):
            thread.run()
        self.assertEqual(errors, [])
        self.assertEqual(len(lost), 1)
        self.assertEqual(restored, [True])
        self.assertFalse(thread._link_down)
        self.assertEqual(stub.Ping.call_count, 4)
        self.assertEqual([c.args[0].reset for c in stub.InferChunk.call_args_list], [True, True, False])
        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_equal(chunks[0], valid)
        channel.close.assert_called_once()

    def test_cloud_ping_authentication_and_error_replies_still_block_requests(self):
        for failure in ('token', 'unavailable', 'reply_error', 'heartbeat_error'):
            with self.subTest(failure=failure):
                pb, thread, stub, channel, modules, grpc = self.cloud_fixture()
                if failure in ('token', 'unavailable'):
                    code = (grpc.StatusCode.UNAUTHENTICATED if failure == 'token'
                            else grpc.StatusCode.UNAVAILABLE)
                    stub.Ping.side_effect = grpc.RpcError(code)
                elif failure == 'reply_error':
                    stub.Ping.return_value = pb.PingReply(error='denied')
                else:
                    stub.Ping.side_effect = [stub.Ping.return_value, pb.PingReply(error='denied')]
                errors = []
                thread.error.connect(errors.append)
                thread.request_chunk()
                with patch.dict(sys.modules, modules):
                    thread.run()
                self.assertEqual(len(errors), 1)
                thread.get_obs.assert_not_called()
                stub.InferChunk.assert_not_called()
                if failure == 'token':
                    self.assertIn('token', errors[0])

    def test_cloud_legacy_rtc_reset_and_single_request_parameters(self):
        pb, thread, stub, _, modules, _ = self.cloud_fixture(rtc_trained=True)
        thread._running = False  # capability handshake only
        with patch.dict(sys.modules, modules):
            thread.run()
        self.assertTrue(thread._trained_rtc_active)
        self.assertEqual(thread._trained_max_delay, 6)
        thread._running = True
        thread.avg_n = 3
        thread.advantage_prefix = 'advantage: positive, '
        stub.InferChunk.side_effect = lambda req, **kw: pb.InferenceReply(
            request_id=req.request_id, action_dim=7, num_steps=2, actions=np.zeros(14),
            used_delay=req.rtc_d)
        chunks, singles = [], []
        thread.chunk_ready.connect(lambda c, dt: chunks.append(c))
        thread.single_chunk_ready.connect(lambda c, info: singles.append((c, info)))
        with patch.dict(sys.modules, modules):
            for single, pending, reset, expected_d in (
                    (False, (5, 99), True, 6), (False, (8, -1), False, 0),
                    (False, None, True, 0), (True, (5, 3), True, 0)):
                thread._pending_rtc = pending
                if pending is None:
                    thread._last_raw_chunk = None  # pause/resume keeps server reset
                thread._do_infer(stub, single)
                call = stub.InferChunk.call_args
                req = call.args[0]
                self.assertEqual(req.reset, reset)
                self.assertEqual(req.rtc_d, expected_d)
                self.assertEqual(req.rtc_k, 0 if single or pending is None else pending[0])
                self.assertEqual(req.avg_n, 3)
                self.assertEqual(req.task, 'advantage: positive, task')
                self.assertEqual(req.torque, .25)
                self.assertEqual(call.kwargs['timeout'], 30. if single else thread.rpc_timeout * 3)
                self.assertEqual(thread._last_used_delay, expected_d)
                self.assertIsNone(thread._pending_rtc)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(singles), 1)
        np.testing.assert_array_equal(singles[0][1]['state'], np.arange(7))

    def test_cloud_rpc_errors_and_stale_replies_never_emit_actions(self):
        for failure in ('token', 'unavailable', 'stale', 'reply_error'):
            with self.subTest(failure=failure):
                pb, thread, stub, _, modules, grpc = self.cloud_fixture()
                thread._pb2 = pb
                if failure in ('token', 'unavailable'):
                    code = (grpc.StatusCode.UNAUTHENTICATED if failure == 'token'
                            else grpc.StatusCode.UNAVAILABLE)
                    stub.InferChunk.side_effect = grpc.RpcError(code)
                else:
                    stub.InferChunk.return_value = pb.InferenceReply(
                        request_id=0 if failure == 'stale' else 1, action_dim=7,
                        num_steps=2, actions=np.zeros(14), error='failed' if failure == 'reply_error' else '')
                chunks, errors, lost = [], [], []
                thread.chunk_ready.connect(lambda c, dt: chunks.append(c))
                thread.single_chunk_ready.connect(lambda c, info: chunks.append(c))
                thread.error.connect(errors.append)
                thread.link_lost.connect(lost.append)
                with patch.dict(sys.modules, modules):
                    thread._do_infer(stub, single=False)
                self.assertEqual(chunks, [])
                self.assertEqual(bool(errors), failure == 'token')
                self.assertEqual(bool(lost), failure == 'unavailable')
                if failure == 'token':
                    self.assertFalse(thread._running)
                elif failure == 'unavailable':
                    self.assertTrue(thread._need_reset)
                    self.assertTrue(thread._link_down)



if __name__ == '__main__':
    unittest.main()

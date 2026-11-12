"""GUI-thread orchestration for asynchronous dual-arm VLA inference and 20 Hz dispatch."""

from PyQt6.QtCore import QObject, QTimer, Qt

from debugger.backend.vla_inference_worker import VLAInferenceWorker
from debugger.backend.vla_runtime import VLAObservationSnapshot
from debugger.vla_features import (
    DUAL_JOINT_NAMES,
    DualArmActionDispatcher,
    as_action_chunk,
    sanitize_dual_arm_action,
    split_dual_action,
)


class DualArmVLAController(QObject):
    """Own inference lifecycle; all camera reads and hardware submissions stay on GUI thread."""

    def __init__(self, panel, left_worker, right_worker, frames, state, torque,
                 log, parent=None, backend_factory=None):
        super().__init__(parent)
        self.panel = panel
        self.frames = frames
        self.state = state
        self.torque = torque
        self.log = log
        self.dispatcher = DualArmActionDispatcher(left_worker, right_worker,
                                                   print_fn=log)
        worker_kwargs = {"backend_factory": backend_factory} if backend_factory else {}
        self._worker = VLAInferenceWorker(**worker_kwargs)
        self._worker.model_loaded.connect(self._on_loaded)
        self._worker.model_load_failed.connect(self._on_load_failed)
        self._worker.inference_completed.connect(self._on_inferred)
        self._worker.inference_failed.connect(self._on_infer_failed)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._send_step)
        self._loop_start_timer = QTimer(self)
        self._loop_start_timer.setSingleShot(True)
        self._loop_start_timer.setInterval(5000)
        self._loop_start_timer.timeout.connect(
            lambda: self._fail("等待双臂 control loop 启动超时")
        )
        self._waiting_for_loop_starts = False
        self._pending_loop_starts = set()
        for name, worker in (("left", left_worker), ("right", right_worker)):
            worker.control_loop_changed.connect(
                lambda running, side=name: self._on_control_loop_changed(side, running)
            )
        self._loaded = False
        self._busy = False
        self._loading = False
        self._continuous = False
        self._active = False
        self._request_id = 0
        self._inference_count = 0
        self._chunk = None
        self._index = 0
        self._configured_steps = 20
        self._steps = 20
        panel.load_requested.connect(self.load_model)
        panel.single_requested.connect(lambda: self.start(False))
        panel.continuous_requested.connect(lambda: self.start(True))
        panel.stop_requested.connect(self.stop)
        panel.mode_changed.connect(self._on_mode_changed)
        panel.checkpoint_changed.connect(self._on_checkpoint_changed)
        self._update_controls()

    def _update_controls(self):
        self.panel.set_runtime_state(model_loaded=self._loaded, busy=self._busy,
                                     active=self._active, loading=self._loading)

    def _note(self, message, *, error=False):
        self.log(f"[VLA] {message}")
        self.panel.set_status(message, error=error)

    def load_model(self):
        if self._busy or self._active or self._loading:
            return
        path = self.panel.checkpoint_path
        if not path:
            self._note("请填写 checkpoint 路径", error=True)
            return
        self._loaded = False
        self._loading = True
        self._update_controls()
        self._note(f"正在加载模型: {path}")
        self._worker.load_model(path)

    def _on_loaded(self, info):
        self._loading = False
        self._loaded = True
        self._configured_steps = int(info.n_action_steps)
        self._steps = self._configured_steps
        self._update_controls()
        self._note(
            f"模型已加载: checkpoint={info.checkpoint}, "
            f"policy={info.policy_type}, device={info.device}, "
            f"chunk={info.chunk_size}, 执行步数={self._steps}"
        )

    def _on_load_failed(self, message):
        self._loading = False
        self._loaded = False
        self._update_controls()
        self._note(f"加载失败: {message}", error=True)

    def _on_checkpoint_changed(self, _path):
        if not self._loaded:
            return
        self._loaded = False
        self._update_controls()
        self._note("checkpoint 路径已修改，请重新加载模型")

    def _on_mode_changed(self, execute):
        self.dispatcher.execute = execute
        self._note("真机执行模式" if execute else "Dry-run 模式")

    def start(self, continuous=False):
        if not self._loaded or self._busy or self._active or self._loading:
            return
        try:
            if self.dispatcher.execute:
                self.dispatcher.require_ready()
            if not self.panel.task:
                raise ValueError("task 不能为空")
            self._continuous = bool(continuous)
            self._active = True
            self._note(
                f"启动 {'continuous' if self._continuous else 'single'} 推理, "
                f"mode={'EXECUTE' if self.dispatcher.execute else 'DRY-RUN'}, "
                f"task={self.panel.task!r}, "
                f"max_delta={self.panel.max_delta_rad:.3f} rad/command"
            )
            self._request_inference()
        except Exception as exc:
            self._fail(str(exc))

    def _request_inference(self):
        if self.dispatcher.execute:
            self.dispatcher.require_ready()
            workers = {
                "left": self.dispatcher.left_worker,
                "right": self.dispatcher.right_worker,
            }
            pending = {
                side for side, worker in workers.items()
                if not self._control_loop_running(worker)
            }
            if pending:
                self._waiting_for_loop_starts = True
                self._pending_loop_starts = pending
                self._busy = True
                self._update_controls()
                self._note("正在启动双臂 control loop")
                self._loop_start_timer.start()
                # Fake/simulation workers may acknowledge synchronously, which
                # mutates the pending set from the signal handler.
                for side in tuple(pending):
                    workers[side].submit_command("start_control_loop")
                return
        self._capture_and_infer()

    @staticmethod
    def _control_loop_running(worker):
        worker_status = getattr(worker, "is_control_loop_running", None)
        if worker_status is not None:
            return bool(worker_status)
        return bool(
            getattr(getattr(worker, "arm", None), "control_loop_running", False)
        )

    def _on_control_loop_changed(self, side, running):
        if not self._waiting_for_loop_starts:
            if self._active and self.dispatcher.execute and not running:
                self._fail(f"{side} arm control loop 意外停止")
            return
        if not running or side not in self._pending_loop_starts:
            return
        self._pending_loop_starts.remove(side)
        if self._pending_loop_starts:
            return
        self._loop_start_timer.stop()
        self._waiting_for_loop_starts = False
        if not self._active:
            return
        try:
            self.dispatcher.require_ready()
            if not all(
                self._control_loop_running(worker)
                for worker in (
                    self.dispatcher.left_worker,
                    self.dispatcher.right_worker,
                )
            ):
                raise RuntimeError("双臂 control loop 启动信号与实际状态不一致")
            self._note("双臂 control loop 已就绪，开始推理")
            self._capture_and_infer()
        except Exception as exc:
            self._fail(str(exc))

    def _capture_and_infer(self):
        snapshot = VLAObservationSnapshot.capture(
            self.frames(), self.state(), self.torque(), self.panel.task
        )
        self._request_id += 1
        self._inference_count += 1
        self._busy = True
        self._update_controls()
        self._note(
            f"快照就绪 / 推理 #{self._inference_count}: "
            f"cameras={snapshot.frame_shapes}, state=(14,), torque=(2,), "
            f"task={snapshot.task!r}"
        )
        self._worker.run_inference(self._request_id, snapshot)

    def _on_inferred(self, request_id, result):
        if request_id != self._request_id or not self._active:
            return
        self._busy = False
        try:
            chunk = as_action_chunk(result.action_chunk)
            playback_steps = min(self._configured_steps, len(chunk))
            if playback_steps <= 0:
                raise ValueError("模型未返回可执行的动作")
            if self.dispatcher.execute:
                self.dispatcher.require_ready()
            self._chunk = chunk
            self._index = 0
            self._steps = playback_steps
            left, right = split_dual_action(chunk)
            self._note(
                f"推理 #{self._inference_count} 完成: "
                f"latency={result.latency_ms:.1f} ms, "
                f"chunk_shape={tuple(chunk.shape)}, playback_steps={self._steps}, "
                f"first_left={left[0].tolist()}, first_right={right[0].tolist()}"
            )
            self._update_controls()
            self._send_step()
            if self._active and self._chunk is not None:
                self._timer.start()
        except Exception as exc:
            self._fail(str(exc))

    def _on_infer_failed(self, request_id, message):
        if request_id == self._request_id and self._active:
            self._fail(f"推理失败: {message}")

    def _send_step(self):
        if not self._active or self._chunk is None:
            return
        try:
            if self.dispatcher.execute:
                self.dispatcher.require_ready()
                if not all(
                    self._control_loop_running(worker)
                    for worker in (
                        self.dispatcher.left_worker,
                        self.dispatcher.right_worker,
                    )
                ):
                    raise RuntimeError("both arm control loops must be running")
            current = self.state()
            result = sanitize_dual_arm_action(
                self._chunk[self._index], current, self.panel.max_delta_rad
            )
            if result.joint_limit_indices:
                names = ", ".join(
                    DUAL_JOINT_NAMES[i] for i in result.joint_limit_indices
                )
                self.log(
                    f"[VLA] step {self._index + 1}: "
                    f"SDK 关节限位 clipping: {names}"
                )
            if result.delta_limit_indices:
                names = ", ".join(
                    DUAL_JOINT_NAMES[i] for i in result.delta_limit_indices
                )
                self.log(
                    f"[VLA] step {self._index + 1}: 单步变化 clipping "
                    f"({self.panel.max_delta_rad:.3f} rad): {names}"
                )
            command = result.action
            self.dispatcher.dispatch(command, step=0, log=False)
            self.log(
                f"[VLA] step {self._index + 1}/{self._steps} "
                f"{'execute' if self.dispatcher.execute else 'dry-run'}: "
                f"left={command[:7].tolist()}, right={command[7:].tolist()}"
            )
            self._index += 1
            if self._index >= self._steps:
                self._timer.stop()
                self._chunk = None
                if self._continuous:
                    self._request_inference()
                else:
                    self._active = False
                    self._update_controls()
                    self._note(f"完成 {self._steps} 步")
        except Exception as exc:
            self._fail(f"执行中止: {exc}")

    def _fail(self, message):
        self.stop(cancel=True)
        self._note(message, error=True)

    def stop(self, *, cancel=True):
        was_active = self._active or self._busy
        was_executing = self.dispatcher.execute and was_active
        self._timer.stop()
        self._loop_start_timer.stop()
        self._waiting_for_loop_starts = False
        self._pending_loop_starts.clear()
        self._request_id += 1  # Ignore a running, non-interruptible inference result.
        self._active = False
        self._busy = False
        self._continuous = False
        self._chunk = None
        self._update_controls()
        if cancel and was_executing:
            self.dispatcher.left_worker.submit_command("cancel_motion")
            self.dispatcher.right_worker.submit_command("cancel_motion")
            self.log("[VLA] 已向双臂发送 cancel_motion")
        if was_active:
            self._note("已停止；后台推理结果将丢弃")

    def shutdown(self):
        self.stop()
        self._worker.shutdown()

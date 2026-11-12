"""Single Python-thread VLA backend with Qt signals for GUI delivery."""

from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtCore import QObject, pyqtSignal

from debugger.backend.vla_runtime import LocalSmolVLABackend


_CLOSING_WORKERS = set()


class _BackendSession:
    """All backend ownership and calls are confined to one executor thread."""

    def __init__(self, backend_factory):
        self.backend_factory = backend_factory
        self.backend = None

    def load(self, checkpoint_path):
        self.close()
        backend = self.backend_factory()
        try:
            info = backend.load(checkpoint_path)
        except BaseException:
            backend.close()
            raise
        self.backend = backend
        return info

    def infer(self, snapshot):
        if self.backend is None:
            raise RuntimeError("VLA model is not loaded")
        return self.backend.infer(snapshot)

    def close(self):
        backend, self.backend = self.backend, None
        if backend is not None:
            backend.close()


class VLAInferenceWorker(QObject):
    """Submit model work to one Python thread; emit results to Qt receivers."""

    model_loaded = pyqtSignal(object)
    model_load_failed = pyqtSignal(str)
    inference_completed = pyqtSignal(int, object)
    inference_failed = pyqtSignal(int, str)
    shutdown_completed = pyqtSignal()

    def __init__(self, backend_factory=LocalSmolVLABackend, parent=None):
        super().__init__(parent)
        self._session = _BackendSession(backend_factory)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vla-model")
        self._closed = False
        self.shutdown_completed.connect(self._release_after_shutdown)

    def load_model(self, checkpoint_path):
        if self._closed:
            return
        future = self._executor.submit(self._session.load, checkpoint_path)

        def completed(done):
            if self._closed:
                return
            try:
                info = done.result()
            except Exception as exc:
                self.model_load_failed.emit(f"{type(exc).__name__}: {exc}")
            else:
                self.model_loaded.emit(info)

        future.add_done_callback(completed)

    def run_inference(self, request_id, snapshot):
        if self._closed:
            return
        future = self._executor.submit(self._session.infer, snapshot)

        def completed(done):
            if self._closed:
                return
            try:
                result = done.result()
            except Exception as exc:
                self.inference_failed.emit(
                    int(request_id), f"{type(exc).__name__}: {exc}"
                )
            else:
                self.inference_completed.emit(int(request_id), result)

        future.add_done_callback(completed)

    def shutdown(self):
        """Queue backend cleanup on its owning thread without waiting in the GUI."""
        if self._closed:
            return
        self._closed = True
        _CLOSING_WORKERS.add(self)
        future = self._executor.submit(self._session.close)
        future.add_done_callback(lambda _done: self.shutdown_completed.emit())
        self._executor.shutdown(wait=False)

    def _release_after_shutdown(self):
        _CLOSING_WORKERS.discard(self)

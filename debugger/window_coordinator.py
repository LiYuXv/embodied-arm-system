"""Own the active debugger window and switch safely between GUI modes."""

from PyQt6.QtCore import QObject, pyqtSignal


class WindowCoordinator(QObject):
    """Runtime single/dual window switcher.

    The replacement window is shown before the old one is closed, preventing
    QApplication's ``quitOnLastWindowClosed`` behavior from terminating the
    process during a mode change.
    """

    mode_changed = pyqtSignal(str)

    def __init__(self, arm_manager, *, urdf_path=None,
                 single_can_name="can0", sim_mode=False,
                 window_factories=None, parent=None):
        super().__init__(parent)
        self.arm_manager = arm_manager
        self.urdf_path = urdf_path
        self.single_can_name = single_can_name
        self.sim_mode = bool(sim_mode)
        self.window_factories = window_factories or {}
        self.window = None
        self.mode = None
        self._switching = False

    def show(self, mode="dual"):
        self.switch_mode(mode)
        return self.window

    def switch_mode(self, mode):
        if mode not in {"single", "dual"}:
            raise ValueError("mode must be 'single' or 'dual'")
        if self._switching or mode == self.mode:
            return

        self._switching = True
        try:
            replacement = self._create_window(mode)
            replacement.mode_switch_requested.connect(self.switch_mode)
            replacement.show()

            previous = self.window
            self.window = replacement
            self.mode = mode
            if previous is not None:
                previous.close()
                previous.deleteLater()
            self.mode_changed.emit(mode)
        finally:
            self._switching = False

    def _create_window(self, mode):
        factory = self.window_factories.get(mode)
        if factory is not None:
            return factory()
        if mode == "dual":
            from debugger.main_window import MainWindow
            return MainWindow(
                arm_manager=self.arm_manager,
                urdf_path=self.urdf_path,
                sim_mode=self.sim_mode,
            )

        from debugger.single_arm_window import SingleArmWindow
        return SingleArmWindow(
            urdf_path=self.urdf_path,
            can_name=self.single_can_name,
            sim_mode=self.sim_mode,
        )

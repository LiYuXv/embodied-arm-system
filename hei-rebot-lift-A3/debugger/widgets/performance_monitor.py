"""系统性能监测（独立程序 + 可嵌入对话框）

独立运行:
    python3 debugger/widgets/performance_monitor.py

嵌入使用（主界面「监控」按钮）:
    from debugger.widgets.performance_monitor import PerformanceMonitorDialog
"""

import os
import sys
import threading

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6 import QtWidgets

from system_analyzer.thread_analyzer import SystemThreadAnalyzer
from system_analyzer.performance_panel import PerformancePanel


class PerformanceMonitorDialog(QtWidgets.QDialog):
    """系统性能监测对话框（与原底盘程序中的「性能监测」一致）"""

    def __init__(self, parent=None, update_interval: float = 1.0):
        super().__init__(parent)
        self.setWindowTitle("系统性能监测")
        self.analyzer = SystemThreadAnalyzer(update_interval=update_interval)
        self.analyzer.register_thread("MainThread", threading.current_thread())
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(PerformancePanel(self.analyzer))
        self.resize(640, 520)

    def showEvent(self, event):
        self.analyzer.start_monitoring()
        super().showEvent(event)

    def closeEvent(self, event):
        self.analyzer.stop_monitoring()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    dlg = PerformanceMonitorDialog()
    dlg.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

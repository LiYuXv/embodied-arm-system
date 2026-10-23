# performance_panel.py
"""
简化的性能监控面板 - 兼容Windows
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QTextEdit, QPushButton, QScrollArea)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

class PerformancePanel(QWidget):
    """性能监控面板"""
    
    def __init__(self, thread_analyzer, parent=None):
        super().__init__(parent)
        self.thread_analyzer = thread_analyzer
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        
        # 控制按钮组
        control_group = QGroupBox("控制")
        control_layout = QHBoxLayout(control_group)
        
        self.start_btn = QPushButton("开始监控")
        self.stop_btn = QPushButton("停止监控")
        self.refresh_btn = QPushButton("刷新报告")
        self.print_report_btn = QPushButton("打印报告")
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.refresh_btn)
        control_layout.addWidget(self.print_report_btn)
        control_layout.addStretch()
        
        # 系统状态组
        system_group = QGroupBox("系统状态")
        system_layout = QVBoxLayout(system_group)
        
        self.system_status_label = QLabel("系统状态: 未监控")
        system_layout.addWidget(self.system_status_label)
        
        # 线程状态组
        thread_group = QGroupBox("线程状态")
        thread_layout = QVBoxLayout(thread_group)
        
        self.thread_display = QTextEdit()
        self.thread_display.setReadOnly(True)
        self.thread_display.setFont(QFont("Consolas", 9))
        thread_layout.addWidget(self.thread_display)
        
        # 警告信息组
        warning_group = QGroupBox("警告信息")
        warning_layout = QVBoxLayout(warning_group)
        
        self.warning_display = QTextEdit()
        self.warning_display.setReadOnly(True)
        self.warning_display.setMaximumHeight(100)
        warning_layout.addWidget(self.warning_display)
        
        # 添加到主布局
        layout.addWidget(control_group)
        layout.addWidget(system_group)
        layout.addWidget(thread_group)
        layout.addWidget(warning_group)
        
    def setup_connections(self):
        """设置信号连接"""
        self.start_btn.clicked.connect(self.thread_analyzer.start_monitoring)
        self.stop_btn.clicked.connect(self.thread_analyzer.stop_monitoring)
        self.refresh_btn.clicked.connect(self.update_display)
        self.print_report_btn.clicked.connect(self.thread_analyzer.print_detailed_report)
        
        # 连接分析器信号
        self.thread_analyzer.performance_metrics_updated.connect(self.on_metrics_updated)
        self.thread_analyzer.thread_warning.connect(self.on_thread_warning)
        
        # 设置自动刷新定时器
        self.auto_refresh_timer = QTimer()
        self.auto_refresh_timer.timeout.connect(self.update_display)
        self.auto_refresh_timer.start(2000)  # 2秒刷新一次
        
    def on_metrics_updated(self, metrics: dict):
        """性能指标更新时的回调"""
        self.current_metrics = metrics
        self.update_display()
        
    def on_thread_warning(self, thread_name: str, warning_msg: str):
        """线程警告回调"""
        timestamp = self.current_metrics.get('timestamp', '')[:19] if hasattr(self, 'current_metrics') else ''
        warning_text = f"[{timestamp}] {thread_name}: {warning_msg}"
        
        # 添加到警告显示区
        current_warnings = self.warning_display.toPlainText()
        lines = current_warnings.split('\n')
        lines.insert(0, warning_text)
        if len(lines) > 20:  # 保持最近20条警告
            lines = lines[:20]
        self.warning_display.setPlainText('\n'.join(lines))
        
    def update_display(self):
        """更新显示"""
        if not hasattr(self, 'current_metrics'):
            return
            
        metrics = self.current_metrics
        
        # 更新系统状态
        system = metrics.get('system', {})
        process = metrics.get('process', {})
        
        system_text = (
            f"系统CPU: {system.get('cpu_percent', 0):.1f}% | "
            f"内存: {system.get('memory_percent', 0):.1f}% | "
            f"进程CPU: {process.get('cpu_percent', 0):.1f}% | "
            f"进程内存: {process.get('memory_mb', 0):.1f}MB | "
            f"线程数: {process.get('thread_count', 0)}"
        )
        self.system_status_label.setText(system_text)
        
        # 更新线程状态
        threads = metrics.get('threads', {})
        thread_text = "线程名称                CPU使用率   状态    注册\n"
        thread_text += "-" * 50 + "\n"
        
        for name, info in sorted(threads.items()):
            cpu_percent = info.get('cpu_percent', 0)
            is_alive = "活跃" if info.get('is_alive', False) else "停止"
            registered = "是" if info.get('registered', False) else "否"
            
            thread_text += f"{name:20} {cpu_percent:8.1f}% {is_alive:6} {registered:8}\n"
            
        self.thread_display.setPlainText(thread_text)


# # performance_panel.py
# """
# 性能监控面板 - 在GUI中显示线程和性能信息
# """

# from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
#                              QLabel, QTextEdit, QPushButton, QScrollArea)
# from PyQt6.QtCore import Qt, QTimer
# from PyQt6.QtGui import QFont

# class PerformancePanel(QWidget):
#     """性能监控面板"""
    
#     def __init__(self, thread_analyzer, parent=None):
#         super().__init__(parent)
#         self.thread_analyzer = thread_analyzer
#         self.setup_ui()
#         self.setup_connections()
        
#     def setup_ui(self):
#         """设置UI"""
#         layout = QVBoxLayout(self)
        
#         # 控制按钮组
#         control_group = QGroupBox("控制")
#         control_layout = QHBoxLayout(control_group)
        
#         self.start_btn = QPushButton("开始监控")
#         self.stop_btn = QPushButton("停止监控")
#         self.refresh_btn = QPushButton("刷新报告")
        
#         control_layout.addWidget(self.start_btn)
#         control_layout.addWidget(self.stop_btn)
#         control_layout.addWidget(self.refresh_btn)
#         control_layout.addStretch()
        
#         # 系统状态组
#         system_group = QGroupBox("系统状态")
#         system_layout = QVBoxLayout(system_group)
        
#         self.system_status_label = QLabel("系统状态: 未监控")
#         system_layout.addWidget(self.system_status_label)
        
#         # 线程状态组
#         thread_group = QGroupBox("线程状态")
#         thread_layout = QVBoxLayout(thread_group)
        
#         self.thread_display = QTextEdit()
#         self.thread_display.setReadOnly(True)
#         self.thread_display.setFont(QFont("Consolas", 9))
#         thread_layout.addWidget(self.thread_display)
        
#         # 警告信息组
#         warning_group = QGroupBox("警告信息")
#         warning_layout = QVBoxLayout(warning_group)
        
#         self.warning_display = QTextEdit()
#         self.warning_display.setReadOnly(True)
#         self.warning_display.setMaximumHeight(100)
#         warning_layout.addWidget(self.warning_display)
        
#         # 添加到主布局
#         layout.addWidget(control_group)
#         layout.addWidget(system_group)
#         layout.addWidget(thread_group)
#         layout.addWidget(warning_group)
        
#     def setup_connections(self):
#         """设置信号连接"""
#         self.start_btn.clicked.connect(self.thread_analyzer.start_monitoring)
#         self.stop_btn.clicked.connect(self.thread_analyzer.stop_monitoring)
#         self.refresh_btn.clicked.connect(self.update_display)
        
#         # 连接分析器信号
#         self.thread_analyzer.performance_metrics_updated.connect(self.on_metrics_updated)
#         self.thread_analyzer.thread_warning.connect(self.on_thread_warning)
        
#         # 设置自动刷新定时器
#         self.auto_refresh_timer = QTimer()
#         self.auto_refresh_timer.timeout.connect(self.update_display)
#         self.auto_refresh_timer.start(2000)  # 2秒刷新一次
        
#     def on_metrics_updated(self, metrics: dict):
#         """性能指标更新时的回调"""
#         self.current_metrics = metrics
#         self.update_display()
        
#     def on_thread_warning(self, thread_name: str, warning_msg: str):
#         """线程警告回调"""
#         timestamp = metrics.get('timestamp', '')[:19] if hasattr(self, 'current_metrics') else ''
#         warning_text = f"[{timestamp}] {thread_name}: {warning_msg}"
        
#         # 添加到警告显示区
#         current_warnings = self.warning_display.toPlainText()
#         lines = current_warnings.split('\n')
#         lines.insert(0, warning_text)
#         if len(lines) > 20:  # 保持最近20条警告
#             lines = lines[:20]
#         self.warning_display.setPlainText('\n'.join(lines))
        
#     def update_display(self):
#         """更新显示"""
#         if not hasattr(self, 'current_metrics'):
#             return
            
#         metrics = self.current_metrics
        
#         # 更新系统状态
#         system = metrics.get('system', {})
#         process = metrics.get('process', {})
        
#         system_text = (
#             f"系统CPU: {system.get('cpu_percent', 0):.1f}% | "
#             f"内存: {system.get('memory_percent', 0):.1f}% | "
#             f"进程CPU: {process.get('cpu_percent', 0):.1f}% | "
#             f"进程内存: {process.get('memory_mb', 0):.1f}MB | "
#             f"线程数: {process.get('thread_count', 0)}"
#         )
#         self.system_status_label.setText(system_text)
        
#         # 更新线程状态
#         threads = metrics.get('threads', {})
#         thread_text = "线程名称                CPU使用率   状态    注册\n"
#         thread_text += "-" * 50 + "\n"
        
#         for name, info in sorted(threads.items()):
#             cpu_percent = info.get('cpu_percent', 0)
#             is_alive = "活跃" if info.get('is_alive', False) else "停止"
#             registered = "是" if info.get('registered', False) else "否"
            
#             thread_text += f"{name:20} {cpu_percent:8.1f}% {is_alive:6} {registered:8}\n"
            
#         self.thread_display.setPlainText(thread_text)
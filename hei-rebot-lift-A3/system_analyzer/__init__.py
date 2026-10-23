# -*- coding: utf-8 -*-
"""
system_analyzer 包 - 系统监控 / 性能面板 / 系统设置

默认只强依赖轻量的设置模块（settings_manager），这样即使缺少 psutil/numpy
也不会影响系统设置功能；线程分析器（依赖 psutil/numpy）在可用时额外导出。
"""

from .settings_manager import DEFAULT_SETTINGS, load_settings, save_settings, SETTINGS_FILE, settings_path

__all__ = ["DEFAULT_SETTINGS", "load_settings", "save_settings", "SETTINGS_FILE", "settings_path"]

# 可选导出：线程/CPU 监控（依赖 psutil/numpy，缺失时静默跳过）
try:
    from .thread_analyzer import SystemThreadAnalyzer, ThreadInfo
    from .performance_panel import PerformancePanel

    __all__ += ["SystemThreadAnalyzer", "ThreadInfo", "PerformancePanel"]
except Exception:  # noqa: BLE001
    pass

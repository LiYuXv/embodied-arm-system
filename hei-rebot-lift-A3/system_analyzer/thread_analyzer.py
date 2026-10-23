# thread_analyzer.py
"""
系统线程分析器 - 独立模块
修复Windows兼容性问题
"""

import threading  # 添加这个导入
import time
import psutil
import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from PyQt6.QtCore import QTimer, QObject, pyqtSignal

class ThreadInfo:
    """线程信息类"""
    def __init__(self, name: str, thread_obj):
        self.name = name
        self.thread_obj = thread_obj
        self.thread_id = thread_obj.ident if hasattr(thread_obj, 'ident') else None
        self.start_time = time.time()
        self.cpu_usage_history = []
        self.update_count = 0
        self.is_alive = True
        self.peak_cpu_usage = 0.0
        self.last_update_time = self.start_time

class SystemThreadAnalyzer(QObject):
    """
    系统线程分析器
    监控所有线程的状态、CPU占用和性能指标
    """
    
    # 信号定义
    performance_metrics_updated = pyqtSignal(dict)  # 性能指标更新
    thread_warning = pyqtSignal(str, str)  # 线程警告 (线程名, 警告信息)
    
    def __init__(self, update_interval: float = 1.0, max_history: int = 100):
        super().__init__()
        self.update_interval = update_interval
        self.max_history = max_history
        self.threads_info: Dict[str, ThreadInfo] = {}
        self.process = psutil.Process(os.getpid())
        self.running = False
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self._update_all_metrics)
        
        # 性能阈值
        self.cpu_warning_threshold = 80.0  # CPU使用率警告阈值(%)
        self.memory_warning_threshold = 85.0  # 内存使用率警告阈值(%)
        self.thread_cpu_warning = 30.0  # 单个线程CPU使用率警告阈值(%)
        
        # 统计信息
        self.start_time = time.time()
        self.total_updates = 0
        
        # 平台特定设置
        self.is_windows = sys.platform.startswith('win')
        
        print("线程分析器初始化成功")
        
    def start_monitoring(self):
        """开始监控"""
        if self.running:
            return
            
        self.running = True
        self.monitor_timer.start(int(self.update_interval * 1000))
        print(f"线程分析器已启动，更新间隔: {self.update_interval}秒")
        
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        self.monitor_timer.stop()
        print("线程分析器已停止")
        
    def register_thread(self, name: str, thread_obj):
        """注册线程到监控系统"""
        if name in self.threads_info:
            print(f"线程 '{name}' 已存在，更新信息")
            
        thread_info = ThreadInfo(name, thread_obj)
        self.threads_info[name] = thread_info
        print(f"已注册线程: {name} (ID: {thread_info.thread_id})")
        
    def unregister_thread(self, name: str):
        """取消线程注册"""
        if name in self.threads_info:
            del self.threads_info[name]
            print(f"已取消注册线程: {name}")
            
    def _update_all_metrics(self):
        """更新所有性能指标"""
        if not self.running:
            return
            
        self.total_updates += 1
        
        # 获取系统级指标
        system_metrics = self._get_system_metrics()
        
        # 获取进程级指标
        process_metrics = self._get_process_metrics()
        
        # 更新线程级指标
        thread_metrics = self._update_thread_metrics()
        
        # 合并所有指标
        all_metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': system_metrics,
            'process': process_metrics,
            'threads': thread_metrics,
            'uptime': time.time() - self.start_time,
            'total_updates': self.total_updates
        }
        
        # 检查警告条件
        self._check_warnings(all_metrics)
        
        # 发送更新信号
        self.performance_metrics_updated.emit(all_metrics)
        
    def _get_system_metrics(self) -> Dict:
        """获取系统级性能指标"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') and not self.is_windows else [0, 0, 0]
            }
        except Exception as e:
            print(f"获取系统指标失败: {e}")
            return {}
            
    def _get_process_metrics(self) -> Dict:
        """获取进程级性能指标"""
        try:
            with self.process.oneshot():
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                thread_count = self.process.num_threads()
                
                return {
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_info.rss / (1024**2),
                    'thread_count': thread_count,
                    'open_files': len(self.process.open_files()) if not self.is_windows else 0,
                    'connections': len(self.process.connections()) if not self.is_windows else 0
                }
        except Exception as e:
            print(f"获取进程指标失败: {e}")
            return {}
            
    def _update_thread_metrics(self) -> Dict:
        """更新线程级性能指标 - Windows兼容版本"""
        thread_metrics = {}
        
        # Windows平台下使用简化的线程监控
        if self.is_windows:
            return self._update_thread_metrics_windows()
        
        # 非Windows平台的原有逻辑
        try:
            # 获取所有线程的CPU时间
            thread_cpu_times = {}
            for thread in self.process.threads():
                try:
                    thread_cpu_times[thread.id] = thread.cpu_times().user + thread.cpu_times().system
                except AttributeError:
                    # 如果线程没有cpu_times属性，使用0
                    thread_cpu_times[thread.id] = 0
        except Exception as e:
            print(f"获取线程CPU时间失败: {e}")
            # 回退到简化模式
            return self._update_thread_metrics_simple()
            
        # 短暂等待后再次采样计算CPU使用率
        time.sleep(0.05)
        
        try:
            for thread in self.process.threads():
                try:
                    new_cpu_time = thread.cpu_times().user + thread.cpu_times().system
                    old_cpu_time = thread_cpu_times.get(thread.id, 0)
                    
                    # 计算CPU使用率
                    cpu_delta = new_cpu_time - old_cpu_time
                    cpu_percent = (cpu_delta / 0.05) * 100  # 基于采样间隔
                except AttributeError:
                    cpu_percent = 0.0
                
                # 找到对应的注册线程
                thread_name = f"Thread-{thread.id}"
                for name, info in self.threads_info.items():
                    if info.thread_id == thread.id:
                        thread_name = name
                        break
                
                # 更新线程信息
                if thread_name in self.threads_info:
                    info = self.threads_info[thread_name]
                    info.update_count += 1
                    info.last_update_time = time.time()
                    info.is_alive = True  # 简化处理
                    
                    # 记录CPU使用率历史
                    info.cpu_usage_history.append(cpu_percent)
                    if len(info.cpu_usage_history) > self.max_history:
                        info.cpu_usage_history.pop(0)
                    
                    # 更新峰值CPU使用率
                    if cpu_percent > info.peak_cpu_usage:
                        info.peak_cpu_usage = cpu_percent
                
                thread_metrics[thread_name] = {
                    'thread_id': thread.id,
                    'cpu_percent': cpu_percent,
                    'user_time': 0.0,
                    'system_time': 0.0,
                    'is_alive': True,
                    'registered': thread_name in self.threads_info
                }
                
        except Exception as e:
            print(f"更新线程指标失败: {e}")
            return self._update_thread_metrics_simple()
            
        return thread_metrics
        
    def _update_thread_metrics_windows(self) -> Dict:
        """Windows平台的线程指标更新"""
        thread_metrics = {}
        
        # Windows下使用简化的线程信息
        for name, info in self.threads_info.items():
            # 检查线程是否存活
            is_alive = False
            if hasattr(info.thread_obj, 'is_alive'):
                is_alive = info.thread_obj.is_alive()
            elif hasattr(info.thread_obj, 'isRunning'):  # QThread
                is_alive = info.thread_obj.isRunning()
            else:
                is_alive = True  # 默认认为存活
                
            info.is_alive = is_alive
            info.update_count += 1
            info.last_update_time = time.time()
            
            thread_metrics[name] = {
                'thread_id': info.thread_id or 0,
                'cpu_percent': 0.0,  # Windows下难以获取准确值
                'user_time': 0.0,
                'system_time': 0.0,
                'is_alive': is_alive,
                'registered': True
            }
            
        return thread_metrics
        
    def _update_thread_metrics_simple(self) -> Dict:
        """简化版的线程指标更新"""
        thread_metrics = {}
        
        for name, info in self.threads_info.items():
            # 简化处理：只更新基本状态
            info.update_count += 1
            info.last_update_time = time.time()
            
            thread_metrics[name] = {
                'thread_id': info.thread_id or 0,
                'cpu_percent': 0.0,
                'user_time': 0.0,
                'system_time': 0.0,
                'is_alive': True,  # 简化处理
                'registered': True
            }
            
        return thread_metrics
        
    def _check_warnings(self, metrics: Dict):
        """检查性能警告条件"""
        # 系统级警告
        system = metrics.get('system', {})
        if system.get('cpu_percent', 0) > self.cpu_warning_threshold:
            self.thread_warning.emit(
                "SYSTEM", 
                f"系统CPU使用率过高: {system['cpu_percent']:.1f}%"
            )
            
        if system.get('memory_percent', 0) > self.memory_warning_threshold:
            self.thread_warning.emit(
                "SYSTEM", 
                f"系统内存使用率过高: {system['memory_percent']:.1f}%"
            )
            
        # 线程级警告 - 在Windows下简化处理
        if not self.is_windows:
            threads = metrics.get('threads', {})
            for thread_name, thread_info in threads.items():
                if thread_info.get('cpu_percent', 0) > self.thread_cpu_warning:
                    self.thread_warning.emit(
                        thread_name,
                        f"线程CPU使用率过高: {thread_info['cpu_percent']:.1f}%"
                    )
                
    def get_summary_report(self) -> Dict:
        """获取汇总报告"""
        total_threads = len(self.threads_info)
        alive_threads = sum(1 for info in self.threads_info.values() if info.is_alive)
        
        # 计算平均CPU使用率
        avg_cpu_usage = {}
        for name, info in self.threads_info.items():
            if info.cpu_usage_history:
                avg_cpu_usage[name] = np.mean(info.cpu_usage_history)
            else:
                avg_cpu_usage[name] = 0.0
        
        return {
            'total_registered_threads': total_threads,
            'alive_threads': alive_threads,
            'uptime_hours': (time.time() - self.start_time) / 3600,
            'total_updates': self.total_updates,
            'average_cpu_usage': avg_cpu_usage,
            'high_cpu_threads': [
                name for name, avg_cpu in avg_cpu_usage.items() 
                if avg_cpu > self.thread_cpu_warning
            ]
        }
        
    def print_detailed_report(self):
        """打印详细报告到控制台"""
        print("\n" + "="*60)
        print("系统线程分析报告")
        print("="*60)
        
        summary = self.get_summary_report()
        print(f"运行时间: {summary['uptime_hours']:.2f} 小时")
        print(f"注册线程数: {summary['total_registered_threads']}")
        print(f"活跃线程数: {summary['alive_threads']}")
        print(f"监控更新次数: {summary['total_updates']}")
        
        print("\n线程详情:")
        print("-" * 40)
        for name, info in self.threads_info.items():
            avg_cpu = np.mean(info.cpu_usage_history) if info.cpu_usage_history else 0
            status = "活跃" if info.is_alive else "停止"
            print(f"{name:20} | CPU: {avg_cpu:5.1f}% | 状态: {status:6} | 更新: {info.update_count:6}")
            
        if summary['high_cpu_threads']:
            print(f"\n警告: 以下线程CPU使用率较高: {summary['high_cpu_threads']}")
            
        print("="*60)


# # thread_analyzer.py
# """
# 系统线程分析器 - 独立模块
# 用于监控和分析多线程程序的线程状态、CPU占用和性能指标
# """

# import threading
# import time
# import psutil
# import os
# import numpy as np
# from datetime import datetime
# from typing import Dict, List, Optional
# from PyQt6.QtCore import QTimer, QObject, pyqtSignal

# class ThreadInfo:
#     """线程信息类"""
#     def __init__(self, name: str, thread_obj):
#         self.name = name
#         self.thread_obj = thread_obj
#         self.thread_id = thread_obj.ident if hasattr(thread_obj, 'ident') else None
#         self.start_time = time.time()
#         self.cpu_usage_history = []
#         self.update_count = 0
#         self.is_alive = True
#         self.peak_cpu_usage = 0.0
#         self.last_update_time = self.start_time

# class SystemThreadAnalyzer(QObject):
#     """
#     系统线程分析器
#     监控所有线程的状态、CPU占用和性能指标
#     """
    
#     # 信号定义
#     performance_metrics_updated = pyqtSignal(dict)  # 性能指标更新
#     thread_warning = pyqtSignal(str, str)  # 线程警告 (线程名, 警告信息)
    
#     def __init__(self, update_interval: float = 1.0, max_history: int = 100):
#         super().__init__()
#         self.update_interval = update_interval
#         self.max_history = max_history
#         self.threads_info: Dict[str, ThreadInfo] = {}
#         self.process = psutil.Process(os.getpid())
#         self.running = False
#         self.monitor_timer = QTimer()
#         self.monitor_timer.timeout.connect(self._update_all_metrics)
        
#         # 性能阈值
#         self.cpu_warning_threshold = 80.0  # CPU使用率警告阈值(%)
#         self.memory_warning_threshold = 85.0  # 内存使用率警告阈值(%)
#         self.thread_cpu_warning = 30.0  # 单个线程CPU使用率警告阈值(%)
        
#         # 统计信息
#         self.start_time = time.time()
#         self.total_updates = 0
        
#     def start_monitoring(self):
#         """开始监控"""
#         if self.running:
#             return
            
#         self.running = True
#         self.monitor_timer.start(int(self.update_interval * 1000))
#         print(f"线程分析器已启动，更新间隔: {self.update_interval}秒")
        
#     def stop_monitoring(self):
#         """停止监控"""
#         self.running = False
#         self.monitor_timer.stop()
#         print("线程分析器已停止")
        
#     def register_thread(self, name: str, thread_obj):
#         """注册线程到监控系统"""
#         if name in self.threads_info:
#             print(f"线程 '{name}' 已存在，更新信息")
            
#         thread_info = ThreadInfo(name, thread_obj)
#         self.threads_info[name] = thread_info
#         print(f"已注册线程: {name} (ID: {thread_info.thread_id})")
        
#     def unregister_thread(self, name: str):
#         """取消线程注册"""
#         if name in self.threads_info:
#             del self.threads_info[name]
#             print(f"已取消注册线程: {name}")
            
#     def _update_all_metrics(self):
#         """更新所有性能指标"""
#         if not self.running:
#             return
            
#         self.total_updates += 1
        
#         # 获取系统级指标
#         system_metrics = self._get_system_metrics()
        
#         # 获取进程级指标
#         process_metrics = self._get_process_metrics()
        
#         # 更新线程级指标
#         thread_metrics = self._update_thread_metrics()
        
#         # 合并所有指标
#         all_metrics = {
#             'timestamp': datetime.now().isoformat(),
#             'system': system_metrics,
#             'process': process_metrics,
#             'threads': thread_metrics,
#             'uptime': time.time() - self.start_time,
#             'total_updates': self.total_updates
#         }
        
#         # 检查警告条件
#         self._check_warnings(all_metrics)
        
#         # 发送更新信号
#         self.performance_metrics_updated.emit(all_metrics)
        
#     def _get_system_metrics(self) -> Dict:
#         """获取系统级性能指标"""
#         try:
#             return {
#                 'cpu_percent': psutil.cpu_percent(interval=None),
#                 'memory_percent': psutil.virtual_memory().percent,
#                 'memory_used_gb': psutil.virtual_memory().used / (1024**3),
#                 'memory_total_gb': psutil.virtual_memory().total / (1024**3),
#                 'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
#             }
#         except Exception as e:
#             print(f"获取系统指标失败: {e}")
#             return {}
            
#     def _get_process_metrics(self) -> Dict:
#         """获取进程级性能指标"""
#         try:
#             with self.process.oneshot():
#                 cpu_percent = self.process.cpu_percent()
#                 memory_info = self.process.memory_info()
#                 thread_count = self.process.num_threads()
                
#                 return {
#                     'cpu_percent': cpu_percent,
#                     'memory_mb': memory_info.rss / (1024**2),
#                     'thread_count': thread_count,
#                     'open_files': len(self.process.open_files()),
#                     'connections': len(self.process.connections())
#                 }
#         except Exception as e:
#             print(f"获取进程指标失败: {e}")
#             return {}
            
#     def _update_thread_metrics(self) -> Dict:
#         """更新线程级性能指标"""
#         thread_metrics = {}
        
#         # 获取所有线程的CPU时间
#         try:
#             thread_cpu_times = {}
#             for thread in self.process.threads():
#                 thread_cpu_times[thread.id] = thread.cpu_times().user + thread.cpu_times().system
#         except Exception as e:
#             print(f"获取线程CPU时间失败: {e}")
#             return {}
            
#         # 短暂等待后再次采样计算CPU使用率
#         time.sleep(0.05)
        
#         try:
#             for thread in self.process.threads():
#                 new_cpu_time = thread.cpu_times().user + thread.cpu_times().system
#                 old_cpu_time = thread_cpu_times.get(thread.id, 0)
                
#                 # 计算CPU使用率
#                 cpu_delta = new_cpu_time - old_cpu_time
#                 cpu_percent = (cpu_delta / 0.05) * 100  # 基于采样间隔
                
#                 # 找到对应的注册线程
#                 thread_name = f"Thread-{thread.id}"
#                 for name, info in self.threads_info.items():
#                     if info.thread_id == thread.id:
#                         thread_name = name
#                         break
                
#                 # 更新线程信息
#                 if thread_name in self.threads_info:
#                     info = self.threads_info[thread_name]
#                     info.update_count += 1
#                     info.last_update_time = time.time()
#                     info.is_alive = thread.is_alive()
                    
#                     # 记录CPU使用率历史
#                     info.cpu_usage_history.append(cpu_percent)
#                     if len(info.cpu_usage_history) > self.max_history:
#                         info.cpu_usage_history.pop(0)
                    
#                     # 更新峰值CPU使用率
#                     if cpu_percent > info.peak_cpu_usage:
#                         info.peak_cpu_usage = cpu_percent
                
#                 thread_metrics[thread_name] = {
#                     'thread_id': thread.id,
#                     'cpu_percent': cpu_percent,
#                     'user_time': thread.cpu_times().user,
#                     'system_time': thread.cpu_times().system,
#                     'is_alive': thread.is_alive(),
#                     'registered': thread_name in self.threads_info
#                 }
                
#         except Exception as e:
#             print(f"更新线程指标失败: {e}")
            
#         return thread_metrics
        
#     def _check_warnings(self, metrics: Dict):
#         """检查性能警告条件"""
#         # 系统级警告
#         system = metrics.get('system', {})
#         if system.get('cpu_percent', 0) > self.cpu_warning_threshold:
#             self.thread_warning.emit(
#                 "SYSTEM", 
#                 f"系统CPU使用率过高: {system['cpu_percent']:.1f}%"
#             )
            
#         if system.get('memory_percent', 0) > self.memory_warning_threshold:
#             self.thread_warning.emit(
#                 "SYSTEM", 
#                 f"系统内存使用率过高: {system['memory_percent']:.1f}%"
#             )
            
#         # 线程级警告
#         threads = metrics.get('threads', {})
#         for thread_name, thread_info in threads.items():
#             if thread_info.get('cpu_percent', 0) > self.thread_cpu_warning:
#                 self.thread_warning.emit(
#                     thread_name,
#                     f"线程CPU使用率过高: {thread_info['cpu_percent']:.1f}%"
#                 )
                
#     def get_summary_report(self) -> Dict:
#         """获取汇总报告"""
#         total_threads = len(self.threads_info)
#         alive_threads = sum(1 for info in self.threads_info.values() if info.is_alive)
        
#         # 计算平均CPU使用率
#         avg_cpu_usage = {}
#         for name, info in self.threads_info.items():
#             if info.cpu_usage_history:
#                 avg_cpu_usage[name] = np.mean(info.cpu_usage_history)
        
#         return {
#             'total_registered_threads': total_threads,
#             'alive_threads': alive_threads,
#             'uptime_hours': (time.time() - self.start_time) / 3600,
#             'total_updates': self.total_updates,
#             'average_cpu_usage': avg_cpu_usage,
#             'high_cpu_threads': [
#                 name for name, avg_cpu in avg_cpu_usage.items() 
#                 if avg_cpu > self.thread_cpu_warning
#             ]
#         }
        
#     def print_detailed_report(self):
#         """打印详细报告到控制台"""
#         print("\n" + "="*60)
#         print("系统线程分析报告")
#         print("="*60)
        
#         summary = self.get_summary_report()
#         print(f"运行时间: {summary['uptime_hours']:.2f} 小时")
#         print(f"注册线程数: {summary['total_registered_threads']}")
#         print(f"活跃线程数: {summary['alive_threads']}")
#         print(f"监控更新次数: {summary['total_updates']}")
        
#         print("\n线程详情:")
#         print("-" * 40)
#         for name, info in self.threads_info.items():
#             avg_cpu = np.mean(info.cpu_usage_history) if info.cpu_usage_history else 0
#             status = "活跃" if info.is_alive else "停止"
#             print(f"{name:20} | CPU: {avg_cpu:5.1f}% | 状态: {status:6} | 更新: {info.update_count:6}")
            
#         if summary['high_cpu_threads']:
#             print(f"\n警告: 以下线程CPU使用率较高: {summary['high_cpu_threads']}")
            
#         print("="*60)
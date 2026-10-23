# -*- coding: utf-8 -*-
"""
系统设置管理 - 默认配置 + 从 JSON 文件加载/保存。

配置文件位于主程序文件夹下的 settings/system_config.json。
如果文件不存在，则使用 DEFAULT_SETTINGS 作为内置默认值。
"""

import json
import os

# 程序自带默认设置
DEFAULT_SETTINGS = {
    "refresh_ms": 200,           # 状态刷新周期(ms)
    "mit_interval_ms": 20,       # MIT 控制循环周期(ms)
    "analyzer_interval_s": 1.0,  # 系统监测更新间隔(s)
    "serial_timeout_s": 0.2,     # 串口超时(s)
    "default_port": "/dev/ttyACM1",   # 默认主串口(U2CAN)
    "default_io_port": "/dev/ttyUSB0",  # 默认升降限位IO串口
    "baud": 921600,              # 波特率
    "kp_default": 10.0,          # MIT kp 默认
    "kd_default": 1.0,           # MIT kd 默认
    "auto_connect_enable": True,  # 启动后自动连接并使能
}

# 配置文件相对于主程序文件夹的路径
SETTINGS_FILE = "settings/system_config.json"


def _main_dir():
    """主程序文件夹 = 本包(system_analyzer)的上一级目录。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def settings_path():
    """返回配置文件的绝对路径。"""
    return os.path.join(_main_dir(), SETTINGS_FILE)


def load_settings():
    """加载配置；文件不存在或字段缺失时用默认值补全。"""
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(settings_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in DEFAULT_SETTINGS:
                if key in data:
                    settings[key] = data[key]
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        pass
    return settings


def save_settings(settings):
    """保存配置到 JSON 文件（自动创建目录）。"""
    path = settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)

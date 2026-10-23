"""主系统配置加载/保存：debugger/settings/main_system_config.json"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("debugger")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "settings" / "main_system_config.json"

DEFAULT_CONFIG = {
    # 启动 main.py 后自动连接串口（连接成功后自动使能）
    "auto_connect_on_startup": True,
    # CAN 串口控制波特率 (bit/s)
    "can_bitrate": 500000,
    # 摄像头角色绑定：{角色: {"usb": 端口拓扑路径, "index": /dev/videoN}}
    # 角色固定为 arm1(臂1腕部) / head(头部) / arm2(臂2腕部)，与 USB 端口绑定，
    # /dev/videoN 索引重启后会漂移，仅作显示；插拔后按 usb 自动匹配
    "camera_roles": {},
    # VR 控制面板默认设置（启动时应用到 VR 面板，运行时仍可在面板内临时调整）
    "vr_defaults": {
        "lift_speed_rpm": 200.0,      # 升降转速 (RPM)
        "gripper_speed_rad_s": 2.0,   # 摇杆 Y → 夹爪速度 (rad/s)
        "scale": 1.0,                 # 运动缩放 (x)
        "hmd_pos_scale": 1.0,         # 头显位置缩放 (x)
        "hmd_max_lin": 0.30,          # 头显速度跟随 (m/s)
    },
}

# 摄像头角色顺序与中文名（与 widgets/camera_pyqt.py 保持一致）
CAMERA_ROLES = ["arm1", "head", "arm2"]
CAMERA_ROLE_NAMES = {"arm1": "臂1", "head": "头部", "arm2": "臂2"}


def load_config() -> dict:
    """加载配置文件；文件缺失或损坏时返回默认配置。"""
    cfg = dict(DEFAULT_CONFIG)
    try:
        if CONFIG_PATH.is_file():
            text = CONFIG_PATH.read_text(encoding="utf-8").strip()
            if text:
                data = json.loads(text)
                if isinstance(data, dict):
                    cfg.update(data)
    except Exception as e:
        logger.warning(f"[Config] 加载 {CONFIG_PATH} 失败，使用默认配置: {e}")
    return cfg


def save_config(cfg: dict) -> bool:
    """保存配置到 settings/main_system_config.json。"""
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        logger.info(f"[Config] 配置已保存到 {CONFIG_PATH}")
        return True
    except Exception as e:
        logger.warning(f"[Config] 保存 {CONFIG_PATH} 失败: {e}")
        return False


def load_camera_roles() -> dict:
    """读取摄像头角色绑定（settings/main_system_config.json 的 camera_roles）。"""
    roles = load_config().get("camera_roles")
    return dict(roles) if isinstance(roles, dict) else {}


def save_camera_roles(roles: dict) -> bool:
    """保存摄像头角色绑定到 settings/main_system_config.json。"""
    cfg = load_config()
    cfg["camera_roles"] = dict(roles or {})
    return save_config(cfg)

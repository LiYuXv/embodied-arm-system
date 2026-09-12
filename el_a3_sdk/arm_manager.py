"""
多臂管理器 — 统一管理不同 CAN 接口的机械臂实例

提供 Singleton ArmManager，支持：
  - 按名注册 CAN 直连臂
  - 按名获取、遍历、批量断开
  - 从 multi_arm_config.yaml 批量创建
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

logger = logging.getLogger("el_a3_sdk.arm_manager")

DEFAULT_MULTI_ARM_CONFIG = (
    Path(__file__).resolve().parent.parent
    / "resources"
    / "config"
    / "multi_arm_config.yaml"
)
DUAL_ARM_CAN_INTERFACES = {
    "left_arm": "can0",
    "right_arm": "can1",
}


class ArmManager:
    """多臂管理器 — 统一管理不同 CAN 接口的机械臂实例"""

    _instance: Optional["ArmManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ArmManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._arms: Dict[str, object] = {}
                cls._instance._arm_configs: Dict[str, dict] = {}
            return cls._instance

    @classmethod
    def get_instance(cls) -> "ArmManager":
        return cls()

    @classmethod
    def reset(cls):
        """销毁 Singleton（仅用于测试）"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.disconnect_all()
                cls._instance = None

    # ------------------------------------------------------------------
    # 注册 / 获取
    # ------------------------------------------------------------------

    def register_can_arm(
        self,
        name: str,
        can_name: str = "can0",
        **kwargs,
    ):
        """注册 CAN 直连模式臂，返回 ELA3Interface 实例

        Args:
            name: 臂的唯一标识（如 "master"）
            can_name: CAN 接口名（如 "can0"）
            **kwargs: 传给 ELA3Interface 的其他参数
        """
        if name in self._arms:
            requested_config = {"can_name": can_name, **kwargs}
            existing_config = self._arm_configs.get(name)
            if existing_config is not None and existing_config != requested_config:
                raise ValueError(
                    f"臂 '{name}' 已使用不同配置注册: "
                    f"existing={existing_config}, requested={requested_config}"
                )
            logger.warning("臂 '%s' 已注册，返回已有实例", name)
            return self._arms[name]

        from el_a3_sdk.interface import ELA3Interface

        arm = ELA3Interface(can_name=can_name, **kwargs)
        self._arms[name] = arm
        self._arm_configs[name] = {"can_name": can_name, **kwargs}
        logger.info("注册 CAN 臂 '%s' (can=%s)", name, can_name)
        return arm

    def get_arm(self, name: str):
        """按名获取已注册的臂实例"""
        if name not in self._arms:
            raise KeyError(f"臂 '{name}' 未注册")
        return self._arms[name]

    def get_all_arms(self) -> Dict[str, object]:
        return dict(self._arms)

    def has_arm(self, name: str) -> bool:
        return name in self._arms

    def unregister(self, name: str):
        """注销并断开一个臂"""
        arm = self._arms.pop(name, None)
        self._arm_configs.pop(name, None)
        if arm is not None:
            try:
                arm.DisconnectPort()
            except Exception as e:
                logger.warning("断开 '%s' 时出错: %s", name, e)

    def disconnect_all(self):
        """断开所有臂"""
        for name in list(self._arms.keys()):
            self.unregister(name)

    # ------------------------------------------------------------------
    # 工厂方法：从 config 文件批量创建
    # ------------------------------------------------------------------

    @staticmethod
    def from_config(
        config_path: Union[str, Path],
        auto_connect: bool = False,
    ) -> "ArmManager":
        """从 multi_arm_config.yaml 批量创建臂

        Args:
            config_path: YAML 配置文件路径
            auto_connect: 创建后是否自动调用 ConnectPort
        """
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError(f"配置文件顶层必须是映射: {path}")

        mgr = ArmManager.get_instance()
        arms_cfg = cfg.get("arms")
        if not isinstance(arms_cfg, dict) or not arms_cfg:
            raise ValueError(f"配置文件必须包含非空的 'arms' 映射: {path}")

        normalized_configs = []

        for arm_name, arm_params in arms_cfg.items():
            if not isinstance(arm_name, str) or not arm_name:
                raise ValueError("机械臂名称必须是非空字符串")
            if not isinstance(arm_params, dict):
                raise ValueError(f"臂 '{arm_name}' 的配置必须是映射")
            if not arm_params.get("enabled", True):
                logger.info("跳过已禁用的臂: %s", arm_name)
                continue

            can_iface = arm_params.get("can_interface", "can0")
            if not isinstance(can_iface, str) or not can_iface:
                raise ValueError(f"臂 '{arm_name}' 的 can_interface 必须是非空字符串")

            extra = {
                key: value
                for key, value in arm_params.items()
                if key not in {"enabled", "can_interface"}
            }
            if "can_name" in extra:
                raise ValueError(
                    f"臂 '{arm_name}' 请使用 can_interface，不要同时配置 can_name"
                )
            extra.setdefault("host_can_id", 0xFD)

            for path_key in ("inertia_config_path", "urdf_path"):
                configured_path = extra.get(path_key)
                if configured_path and not Path(configured_path).is_absolute():
                    extra[path_key] = str((path.parent / configured_path).resolve())

            normalized_configs.append((arm_name, can_iface, extra))

        for arm_name, can_iface, extra in normalized_configs:
            arm = mgr.register_can_arm(
                name=arm_name,
                can_name=can_iface,
                **extra,
            )

            if auto_connect:
                arm.ConnectPort()

        return mgr

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def arm_names(self):
        return list(self._arms.keys())

    def __len__(self):
        return len(self._arms)

    def __contains__(self, name: str):
        return name in self._arms

    def __getitem__(self, name: str):
        return self.get_arm(name)

    def __repr__(self):
        return f"ArmManager(arms={self.arm_names})"


def initialize_dual_arms(
    config_path: Optional[Union[str, Path]] = None,
    auto_connect: bool = False,
) -> ArmManager:
    """使用统一配置初始化左右臂，供控制端和数据链路共同复用。

    默认读取 ``resources/config/multi_arm_config.yaml``，并保证存在两个
    独立实例且总线映射为 ``left_arm -> can0``、``right_arm -> can1``。
    """
    path = config_path or DEFAULT_MULTI_ARM_CONFIG
    mgr = ArmManager.from_config(path, auto_connect=auto_connect)

    missing = [name for name in DUAL_ARM_CAN_INTERFACES if not mgr.has_arm(name)]
    if missing:
        raise ValueError(f"双臂配置缺少已启用的机械臂: {', '.join(missing)}")

    left_arm = mgr.get_arm("left_arm")
    right_arm = mgr.get_arm("right_arm")
    if left_arm is right_arm:
        raise RuntimeError("left_arm 和 right_arm 必须是两个独立实例")

    for arm_name, expected_can in DUAL_ARM_CAN_INTERFACES.items():
        actual_can = mgr._arm_configs[arm_name]["can_name"]
        if actual_can != expected_can:
            raise ValueError(
                f"{arm_name} 必须使用 {expected_can}，当前配置为 {actual_can}"
            )

    return mgr

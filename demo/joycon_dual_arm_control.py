#!/usr/bin/env python3
"""Joy-Con (L/R) 分别遥操 EL-A3 左右机械臂。"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from typing import Callable, Dict, Mapping

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from demo.vr_control import VRArmController
except ImportError:  # 直接执行 demo/joycon_dual_arm_control.py
    from vr_control import VRArmController
from el_a3_sdk import initialize_dual_arms
from el_a3_sdk.joycon import (
    JOYCON_CONTROL_MODES,
    JOYCON_ROTATION_VECTOR_SIGN,
    JOYCON_MODE_IMU_MOTION,
    JOYCON_ROTATION_FIX,
    JoyConInput,
    JoyConPoseAdapter,
    JoyConUnavailableError,
    discover_joycons,
)

logger = logging.getLogger("joycon_dual_arm_control")

JOYCON_ARM_BINDINGS = {
    "left": "left_arm",
    "right": "right_arm",
}


def build_dual_joycon_controllers(
    arms: Mapping[str, object],
    adapters: Mapping[str, JoyConPoseAdapter],
    update_rate: float = 100.0,
    scale: float = 1.0,
    controller_factory: Callable = VRArmController,
) -> Dict[str, object]:
    """按固定左右关系创建控制器，核心运动链路复用 VRArmController。"""
    controllers = {}
    for side, arm_name in JOYCON_ARM_BINDINGS.items():
        adapter = adapters[side]
        controllers[side] = controller_factory(
            arm=arms[arm_name],
            receiver=adapter,
            state_getter=adapter.get_state,
            update_rate=update_rate,
            scale=scale,
            position_fix=np.ones(3),
            rotation_fix=JOYCON_ROTATION_FIX,
            remap_rotation_rpy=False,
            rotation_vector_sign=JOYCON_ROTATION_VECTOR_SIGN,
            input_name=f"Joy-Con ({'L' if side == 'left' else 'R'})",
        )
    return controllers


def _print_mapping(mode: str):
    print("\nJoy-Con 双臂控制映射")
    print("  Joy-Con (L) -> left_arm  -> can0")
    print("  Joy-Con (R) -> right_arm -> can1")
    if mode == JOYCON_MODE_IMU_MOTION:
        print("  模式           -> IMU 动作模式（实验）")
        print("  ZL / ZR      -> 总离合；按住控制 XYZ + 姿态，松开立即冻结")
        print("  平移动作       -> 姿态补偿后的加速度生成末端 XYZ 速度")
    else:
        print("  模式           -> 标准模式")
        print("  ZL / ZR      -> 按住启用 IMU 姿态旋转，松开冻结旋转")
        print("  摇杆          -> 末端 X/Y 平移（无需离合）")
        print("  L: ↑/↓       -> 末端 Z+/Z- 平移（无需离合）")
        print("  R: X/B       -> 末端 Z+/Z- 平移（无需离合）")
    print("  L: →/←       -> 夹爪张开/闭合")
    print("  R: A/Y       -> 夹爪张开/闭合")
    print("  摇杆按下      -> 回零位")
    print("  L / R        -> 回工作位")
    print("  Minus / Plus -> 退出")


def main() -> int:
    parser = argparse.ArgumentParser(description="EL-A3 Joy-Con 双臂遥操")
    parser.add_argument("--config", default=None, help="双臂 YAML；默认使用 SDK 配置")
    parser.add_argument("--rate", type=float, default=100.0, help="控制频率 Hz")
    parser.add_argument("--scale", type=float, default=1.0, help="相对运动缩放")
    parser.add_argument(
        "--mode", choices=JOYCON_CONTROL_MODES, default="standard",
        help="Joy-Con 控制模式：standard 或 imu_motion（实验）",
    )
    parser.add_argument(
        "--translation-speed", type=float, default=0.12,
        help="标准模式摇杆速度或 IMU 动作模式 XYZ 速度上限，单位 m/s",
    )
    parser.add_argument("--deadzone", type=float, default=0.15, help="摇杆死区")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(name)s][%(levelname)s] %(message)s",
    )
    logging.getLogger("el_a3_sdk").propagate = False

    exit_code = 0
    try:
        pair = discover_joycons()
    except JoyConUnavailableError as exc:
        print(f"\nJoy-Con 发现失败: {exc}")
        print("请确认 hid_nintendo 已加载、两只手柄均已连接且 event 节点可读。")
        return 1

    logger.info(
        "Joy-Con (L): control=%s imu=%s",
        pair.left.control_path,
        pair.left.imu_path,
    )
    logger.info(
        "Joy-Con (R): control=%s imu=%s",
        pair.right.control_path,
        pair.right.imu_path,
    )

    sources = {
        side: JoyConInput(pair.for_side(side)) for side in JOYCON_ARM_BINDINGS
    }
    adapters = {
        side: JoyConPoseAdapter(
            source,
            mode=args.mode,
            translation_speed=args.translation_speed,
            deadzone=args.deadzone,
        )
        for side, source in sources.items()
    }

    arms = initialize_dual_arms(config_path=args.config, auto_connect=False)
    connected_arms = []
    controllers: Dict[str, object] = {}
    controller_threads = []
    shutdown = threading.Event()

    def on_signal(_signal, _frame):
        shutdown.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        for side, arm_name in JOYCON_ARM_BINDINGS.items():
            arm = arms[arm_name]
            if not arm.ConnectPort():
                raise RuntimeError(f"{arm_name} CAN 连接失败")
            connected_arms.append(arm)
            arm.EnableArm()
            arm.start_control_loop(rate_hz=200.0)
            adapters[side].start()

        controllers = build_dual_joycon_controllers(
            arms,
            adapters,
            update_rate=args.rate,
            scale=args.scale,
        )
        _print_mapping(args.mode)
        for side, controller in controllers.items():
            thread = threading.Thread(
                target=controller.start,
                daemon=True,
                name=f"joycon-{side}-arm-controller",
            )
            thread.start()
            controller_threads.append(thread)

        while not shutdown.wait(0.2):
            if any(adapter.exit_requested for adapter in adapters.values()):
                break
            if any(controller.exit_requested for controller in controllers.values()):
                break
    except (KeyboardInterrupt, RuntimeError) as exc:
        if not isinstance(exc, KeyboardInterrupt):
            logger.error("双臂遥操停止: %s", exc)
            exit_code = 1
    finally:
        for controller in controllers.values():
            controller.stop()
        for adapter in adapters.values():
            adapter.stop()
        for thread in controller_threads:
            thread.join(timeout=1.0)
        for arm in connected_arms:
            try:
                arm.ZeroTorqueMode(False)
            except Exception:
                pass
            arm.stop_control_loop()
            arm.DisableArm()
        arms.disconnect_all()

    print("Joy-Con 双臂遥操已退出")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

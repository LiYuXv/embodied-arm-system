"""Fixed Joy-Con L/R controller construction for the GUI and offline checks."""

from typing import Callable, Mapping

import numpy as np

from el_a3_sdk.joycon import (JOYCON_MODE_IMU_MOTION, JOYCON_ROTATION_FIX,
                             JOYCON_ROTATION_VECTOR_SIGN)

JOYCON_ARM_BINDINGS = {"left": "left_arm", "right": "right_arm"}


def build_dual_joycon_controllers(arms: Mapping[str, object], adapters: Mapping[str, object],
                                  update_rate: float = 100.0, scale: float = 1.0,
                                  controller_factory: Callable = None,
                                  rotation_scale: float = 1.0):
    if controller_factory is None:
        from demo.joycon_arm_control import VRArmController
        controller_factory = VRArmController
    controllers = {}
    for side, arm_name in JOYCON_ARM_BINDINGS.items():
        adapter = adapters[side]
        controllers[side] = controller_factory(
            arm=arms[arm_name], receiver=adapter, state_getter=adapter.get_state,
            update_rate=update_rate, scale=scale,
            rotation_scale=rotation_scale if getattr(adapter, "mode", "standard") == JOYCON_MODE_IMU_MOTION else 1.0,
            position_fix=np.ones(3), rotation_fix=JOYCON_ROTATION_FIX,
            remap_rotation_rpy=False, rotation_vector_sign=JOYCON_ROTATION_VECTOR_SIGN,
            input_name=f"Joy-Con ({'L' if side == 'left' else 'R'})")
    return controllers

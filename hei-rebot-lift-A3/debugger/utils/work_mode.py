"""Hardware-free work-mode and checkpoint contracts."""

import json
from pathlib import Path

SINGLE = "single"
DUAL = "dual"


def normalize_mode(mode):
    return mode if mode in (SINGLE, DUAL) else SINGLE


def camera_roles(mode):
    return ("arm1", "head") if mode == SINGLE else ("arm1", "head", "arm2")


def load_model_contract(path):
    checkpoint = Path(path).expanduser().resolve()
    if (checkpoint / "pretrained_model").is_dir():
        checkpoint /= "pretrained_model"
    with (checkpoint / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    return config


def validate_model_contract(config, mode):
    """The existing executor consumes Cartesian pose7 and exactly two cameras.

    Legacy single-arm checkpoints have generic feature names; preserve that
    convention, but reject explicit joint-space or dual-arm metadata.
    ACT's existing pose7+torque input convention is supported as well.
    """
    if mode != SINGLE:
        raise ValueError("当前 VLA 执行器为单臂 Cartesian pose7；请断开后切换单臂模式。双臂 joint14/18 模型尚无执行器。")
    if not isinstance(config, dict):
        raise ValueError("缺少模型配置，不能确认动作空间")
    declared_mode = config.get("work_mode", config.get("arm_mode"))
    if declared_mode is not None and declared_mode != SINGLE:
        raise ValueError("checkpoint 工作模式与单臂模式不匹配")
    space = config.get("action_space")
    if space is not None and space not in ("pose7", "cartesian", "cartesian_pose7"):
        raise ValueError("checkpoint 动作空间不是 Cartesian pose7")
    if "dual" in str(config.get("robot_type", "")).lower():
        raise ValueError("双臂 checkpoint 不可用于单臂执行器")
    inputs = config.get("input_features") or {}
    outputs = config.get("output_features") or {}
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError("模型 input/output_features 必须是配置对象")
    if any(not isinstance(feature, dict) for feature in (*inputs.values(), *outputs.values())):
        raise ValueError("模型 feature 配置无效")
    expected_images = {"observation.images.cam1", "observation.images.cam2"}
    images = {key for key in inputs if key.startswith("observation.images.")}
    if images != expected_images:
        raise ValueError("模型必须且只能使用 cam1=臂1腕部、cam2=头部 两路图像")
    for key in images:
        shape = inputs[key].get("shape", [])
        if len(shape) != 3 or shape[0] != 3 or any(int(x) <= 0 for x in shape):
            raise ValueError("模型图像配置必须为 RGB CHW")
    state = inputs.get("observation.state", {})
    action = outputs.get("action", {})
    allowed_state = ([7], [8]) if config.get("type") == "act" else ([7],)
    if list(state.get("shape", [])) not in allowed_state or list(action.get("shape", [])) != [7]:
        raise ValueError("模型 state/action 维度不匹配：单臂 pose7，ACT 可额外输入单夹爪力矩；禁止 14/18 维")
    torque = inputs.get("observation.gripper_torque")
    if torque is not None and list(torque.get("shape", [])) != [1]:
        raise ValueError("单臂模型必须使用单夹爪力矩")
    for feature in (state, action):
        names = feature.get("names") or []
        if any("joint" in str(n).lower() or str(n).lower().startswith(("left_j", "right_j", "j1")) for n in names):
            raise ValueError("关节动作与末端 pose7 语义不兼容")
    return config

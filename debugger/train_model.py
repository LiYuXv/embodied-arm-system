#!/usr/bin/env python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 本脚本为 lerobot 官方训练入口（lerobot.scripts.lerobot_train）的薄封装，
# 仅额外导入自定义策略模块以完成 draccus 注册：
#   - smolvla_torque：SmolVLA + 独立夹爪力矩观测
#   - pi05_torque：Pi0.5 + 独立夹爪力矩观测
# 用法与官方一致：
#   python train_model.py --config_path smolvla_omy.yaml
try:
    from . import configuration_pi05_torque  # noqa: F401
    from . import configuration_smolvla_torque  # noqa: F401
except ImportError:
    import configuration_pi05_torque  # noqa: F401
    import configuration_smolvla_torque  # noqa: F401

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()

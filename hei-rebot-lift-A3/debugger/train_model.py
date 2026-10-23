#!/usr/bin/env python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 本脚本为 lerobot 官方训练入口（lerobot.scripts.lerobot_train）的薄封装，
# 仅额外导入自定义策略模块以完成 draccus 注册：
#   - smolvla_torque：SmolVLA + 夹爪力矩观测输入
#     （configuration_smolvla_torque / modeling_smolvla_torque / processor_smolvla_torque）
# 用法与官方一致：
#   python train_model.py --config_path smolvla_omy.yaml
import configuration_smolvla_torque  # noqa: F401

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()

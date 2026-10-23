#!/usr/bin/env python
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# pi0.5 专用运行环境约定（见 TRAIN_README.md pi0.5 章节 §2.2 / §5 / §6）：
# - HF_HUB_OFFLINE=1：paligemma-3b-pt-224 tokenizer 走本地缓存（HF 门控仓库，
#   设 hf-mirror 镜像会 403 直接抛异常）；必须在 import lerobot/transformers 之前
#   设置才生效
# - PYTORCH_ALLOC_CONF=expandable_segments:True：抑制 16GB 显存碎片（OOM 对策之一）
# 用 setdefault：外部已显式设置时不覆盖
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# 与 train_model.py 相同的薄封装（导入自定义策略模块完成 draccus 注册，对 pi05 无害），
# 区别：内置上述两个环境变量，且未指定 --config_path 时默认加载 pi05_omy.yaml。
# 用法：
#   python train_model_pi05.py                                    # = --config_path pi05_omy.yaml
#   python train_model_pi05.py --config_path pi05_omy_fullft.yaml # 全量微调（96G 服务器）
#   python train_model_pi05.py --steps=50000 --batch_size=8       # 其余 CLI 覆盖参数照旧
#   python train_model_pi05.py --resume=true                      # 断点续训
import configuration_smolvla_torque  # noqa: F401

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    if not any(a.startswith("--config_path") for a in sys.argv[1:]):
        sys.argv += ["--config_path", "pi05_omy.yaml"]
    main()

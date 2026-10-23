"""ACT 训练/验证共享工具：力矩并入 state、等比分辨率缩放、合并 collate。

力矩适配原理：原生 ACT 只读取 observation.state 作为机器人状态输入，
observation.gripper_torque 会被忽略。因此把 torque 拼接到 state 末尾
(7 -> 8 维)，特征形状与归一化统计量同步合并，模型侧零改动。
"""
import numpy as np
import torch
import torchvision
from torch.utils.data._utils.collate import default_collate

from lerobot.configs.types import FeatureType, PolicyFeature

TORQUE_KEY = "observation.gripper_torque"
STATE_KEY = "observation.state"


def get_cam_hw(metadata, cam_key="observation.images.cam1"):
    h, w, _ = metadata.features[cam_key]["shape"]
    return int(h), int(w)


def get_image_transforms(image_size, cam_hw=None):
    """构建图像缩放变换。

    image_size:
      - "auto" : 按数据集实际分辨率等比缩放，最长边 256（不拉伸失真），需 cam_hw
      - -1     : 保持原始分辨率
      - "H,W"  : 缩放到指定尺寸
      - int    : 缩放到正方形 size x size
    """
    if image_size == -1 or image_size == "-1":
        print("[INFO] Using original image resolution (no resize)")
        return torchvision.transforms.Lambda(lambda x: x)

    if isinstance(image_size, str) and image_size.lower() == "auto":
        if cam_hw is None:
            raise ValueError("image_size='auto' 需要 cam_hw（数据集相机分辨率）")
        h, w = cam_hw
        scale = 256.0 / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        print(f"[INFO] 相机分辨率 {w}x{h}，等比缩放为 {nw}x{nh}（保持长宽比）")
        return torchvision.transforms.Resize((nh, nw))

    if isinstance(image_size, str) and "," in image_size:
        parts = [p.strip() for p in image_size.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid image_size string: {image_size!r}. Expected 'H,W'.")
        h, w = int(parts[0]), int(parts[1])
        print(f"[INFO] Resizing images to {h}x{w}")
        return torchvision.transforms.Resize((h, w))

    size = int(image_size)
    print(f"[INFO] Resizing images to {size}x{size}")
    return torchvision.transforms.Resize((size, size))


def merge_torque_into_state_features(input_features):
    """把 observation.gripper_torque 的特征维度并入 observation.state。"""
    if TORQUE_KEY not in input_features or STATE_KEY not in input_features:
        return input_features
    state_ft = input_features[STATE_KEY]
    torque_ft = input_features.pop(TORQUE_KEY)
    input_features[STATE_KEY] = PolicyFeature(
        type=FeatureType.STATE,
        shape=(state_ft.shape[0] + torque_ft.shape[0],),
    )
    return input_features


def merge_torque_stats(stats):
    """把 torque 的归一化统计量拼到 state 的末尾（mean/std/min/max/qXX 等）。"""
    if TORQUE_KEY not in stats or STATE_KEY not in stats:
        return stats
    stats = dict(stats)
    s_stat = stats[STATE_KEY]
    t_stat = stats.pop(TORQUE_KEY)
    merged = {}
    for name, arr in s_stat.items():
        if name in t_stat and name != "count":
            a = np.asarray(arr, dtype=np.float64).reshape(-1)
            b = np.asarray(t_stat[name], dtype=np.float64).reshape(-1)
            merged[name] = np.concatenate([a, b])
        else:
            merged[name] = arr
    stats[STATE_KEY] = merged
    return stats


def make_collate_fn(merge_torque=True):
    """默认 collate + 把 torque 拼到 state 最后一维（batch 中无 torque 时自动跳过）。"""
    def collate(batch):
        b = default_collate(batch)
        if merge_torque and TORQUE_KEY in b and STATE_KEY in b:
            s = b[STATE_KEY].float()
            t = b[TORQUE_KEY].float()
            while t.dim() < s.dim():
                t = t.unsqueeze(-1)
            b[STATE_KEY] = torch.cat([s, t], dim=-1)
            del b[TORQUE_KEY]
        return b
    return collate

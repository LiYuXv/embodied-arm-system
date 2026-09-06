#!/usr/bin/env python3
"""
Train ACT policy on real robot data (ela3_demo).

This script is adapted from 3.train.ipynb to work with the real robot dataset
located at debugger/lerobot_data/ela3_demo.

Usage:
    # quick test run (200 steps, small batch)
    python3 train_act_real.py --test

    # full training
    python3 train_act_real.py --training_steps 20000 --batch_size 64

    # use only cam1
    python3 train_act_real.py --use_cam1 --use_cam2 False
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
try:
    from lerobot.datasets.utils import dataset_to_policy_features
except ImportError:  # lerobot >= 0.6 移动了该函数
    from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from act_data_utils import (
    TORQUE_KEY,
    get_cam_hw,
    get_image_transforms,
    make_collate_fn,
    merge_torque_into_state_features,
    merge_torque_stats,
)


# Default paths: this script lives in the real robot project folder.
DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_ROOT = os.path.join(
    DEFAULT_PROJECT_ROOT, "lerobot_data", "ela3_demo"
)
DEFAULT_CKPT_DIR = os.path.join(DEFAULT_PROJECT_ROOT, "ckpt", "act_ela3_real")


def build_policy_and_dataset(
    dataset_root: str,
    device: torch.device,
    chunk_size: int = 10,
    n_action_steps: int = 10,
    kl_weight: float = 1.0,
    vision_backbone: str = "resnet18",
    dim_model: int = 512,
    n_heads: int = 8,
    dim_feedforward: int = 3200,
    n_encoder_layers: int = 4,
    n_decoder_layers: int = 1,
    dropout: float = 0.1,
    optimizer_lr: float = 1e-4,
    optimizer_lr_backbone: float = 1e-5,
    use_cam1: bool = True,
    use_cam2: bool = True,
    use_torque: bool = True,
    image_size: int = 256,
    val_split: float = 0.1,
):
    """Load metadata, build policy config, and instantiate dataset."""
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    print(f"[INFO] Loading dataset metadata from: {dataset_root}")
    dataset_metadata = LeRobotDatasetMetadata("ela3_demo", root=dataset_root)
    print(f"[INFO] Dataset: {dataset_metadata.total_episodes} episodes, "
          f"{dataset_metadata.total_frames} frames")
    n_tasks = getattr(dataset_metadata, "total_tasks", None)
    if n_tasks == 1:
        print("[INFO] 单任务数据集，ACT 无任务条件输入，直接训练即可")

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    # Drop unused camera streams if requested.
    if "observation.images.cam1" in input_features and not use_cam1:
        input_features.pop("observation.images.cam1")
        print("[INFO] Dropped observation.images.cam1")
    if "observation.images.cam2" in input_features and not use_cam2:
        input_features.pop("observation.images.cam2")
        print("[INFO] Dropped observation.images.cam2")

    print("[INFO] Input features:", list(input_features.keys()))
    print("[INFO] Output features:", list(output_features.keys()))

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        kl_weight=kl_weight,
        vision_backbone=vision_backbone,
        dim_model=dim_model,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        dropout=dropout,
        optimizer_lr=optimizer_lr,
        optimizer_lr_backbone=optimizer_lr_backbone,
    )

    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
    print(f"[INFO] chunk_size={chunk_size}, n_action_steps={n_action_steps}, "
          f"kl_weight={kl_weight}")
    print(f"[INFO] action delta_timestamps: {delta_timestamps.get('action')}")

    # 力矩适配：原生 ACT 只用 observation.state，把 torque 并入 state（7->8 维），
    # 特征形状与归一化统计量同步合并；数据集仍返回独立的 torque 键，由 collate 拼接
    stats = dataset_metadata.stats
    torque_merged = False
    if use_torque and TORQUE_KEY in input_features:
        merge_torque_into_state_features(input_features)
        cfg.input_features = input_features
        stats = merge_torque_stats(stats)
        torque_merged = True
        print(f"[INFO] 力矩已并入 observation.state，新维度: "
              f"{input_features['observation.state'].shape}")
    elif TORQUE_KEY in input_features:
        input_features.pop(TORQUE_KEY)
        cfg.input_features = input_features
        print("[INFO] 已禁用力矩输入（--no_torque）")
    collate_fn = make_collate_fn(merge_torque=torque_merged)

    print("[INFO] Building policy...")
    policy = ACTPolicy(cfg, dataset_stats=stats)
    policy.train()
    policy.to(device)

    cam_keys = [k for k in ("observation.images.cam1", "observation.images.cam2")
                if k in input_features]
    cam_hw = get_cam_hw(dataset_metadata, cam_keys[0]) if cam_keys else None

    print("[INFO] Loading dataset...")
    # 按 episode 划分训练/验证集（固定种子，保证多次运行划分一致）
    n_ep = dataset_metadata.total_episodes
    val_episodes = []
    train_episodes = list(range(n_ep))
    if val_split > 0 and n_ep >= 5:
        shuffled = np.random.RandomState(42).permutation(n_ep).tolist()
        n_val = max(1, int(round(n_ep * val_split)))
        val_episodes = sorted(shuffled[:n_val])
        train_episodes = sorted(shuffled[n_val:])
        print(f"[INFO] 验证集划分: {len(train_episodes)} 训练 / {len(val_episodes)} 验证 "
              f"episodes（验证: {val_episodes}）")

    dataset = LeRobotDataset(
        "ela3_demo",
        delta_timestamps=delta_timestamps,
        root=dataset_root,
        image_transforms=get_image_transforms(image_size, cam_hw),
        episodes=train_episodes,
    )
    print(f"[INFO] Dataset length: {len(dataset)}")

    val_dataset = None
    if val_episodes:
        val_dataset = LeRobotDataset(
            "ela3_demo",
            delta_timestamps=delta_timestamps,
            root=dataset_root,
            image_transforms=get_image_transforms(image_size, cam_hw),
            episodes=val_episodes,
        )
        print(f"[INFO] Val dataset length: {len(val_dataset)}")

    return policy, dataset, val_dataset, cfg, collate_fn


def train(
    policy: ACTPolicy,
    dataset: LeRobotDataset,
    device: torch.device,
    training_steps: int = 20000,
    batch_size: int = 64,
    num_workers: int = 4,
    log_freq: int = 100,
    save_freq: int = 2000,
    ckpt_dir: str = DEFAULT_CKPT_DIR,
    optimizer_lr: float = 1e-4,
    optimizer_weight_decay: float = 0.0001,
    collate_fn=None,
    val_dataset: LeRobotDataset = None,
    eval_freq: int = 500,
):
    """Standard offline training loop."""
    os.makedirs(ckpt_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=optimizer_lr,
        weight_decay=optimizer_weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_steps
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=2,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=device.type != "cpu",
            drop_last=False,
            collate_fn=collate_fn,
        )

    def evaluate():
        """在验证集上计算平均 L1 loss（eval 模式 ACT 用 prior 采样，kld 不计算）。"""
        policy.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for vb in val_dataloader:
                vb = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in vb.items()
                }
                vloss, _ = policy.forward(vb)
                bs = vb["action"].shape[0]
                tot += vloss.item() * bs
                n += bs
        policy.train()
        return tot / max(n, 1)

    if len(dataloader) == 0:
        raise ValueError(
            f"Dataloader is empty. Dataset has {len(dataset)} samples, "
            f"batch_size={batch_size}."
        )

    print(f"[INFO] Starting training for {training_steps} steps "
          f"(batch_size={batch_size}, workers={num_workers})")

    step = 0
    done = False
    start_time = time.time()
    while not done:
        for batch in dataloader:
            inp_batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            loss, loss_dict = policy.forward(inp_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            if step % log_freq == 0:
                elapsed = time.time() - start_time
                lr = optimizer.param_groups[0]["lr"]
                msg = f"step: {step:>5}  loss: {loss.item():.4f}"
                if loss_dict:
                    parts = [f"{k}: {v:.4f}" for k, v in loss_dict.items()]
                    msg += f"  ({', '.join(parts)})"
                msg += f"  lr: {lr:.2e}  elapsed: {elapsed:.1f}s"
                print(msg, flush=True)

            if val_dataloader is not None and step % eval_freq == 0:
                val_loss = evaluate()
                print(f"step: {step:>5}  [VAL] l1_loss: {val_loss:.4f}", flush=True)

            if step > 0 and step % save_freq == 0:
                save_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}")
                policy.save_pretrained(save_path)
                print(f"[INFO] Saved checkpoint to {save_path}")

            step += 1
            if step >= training_steps:
                done = True
                break

    if val_dataloader is not None:
        val_loss = evaluate()
        print(f"step: {step:>5}  [VAL] l1_loss: {val_loss:.4f} (final)", flush=True)

    # Save final policy.
    policy.save_pretrained(ckpt_dir)
    print(f"[INFO] Final policy saved to {ckpt_dir}")


def quick_test(
    policy: ACTPolicy,
    dataset: LeRobotDataset,
    device: torch.device,
    steps: int = 200,
    batch_size: int = 8,
    collate_fn=None,
):
    """Run a short training loop to verify the loss decreases."""
    print(f"[INFO] Running quick test training: {steps} steps, batch_size={batch_size}")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=2,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    losses = []
    step = 0
    for batch in dataloader:
        if step >= steps:
            break
        inp_batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        loss, _ = policy.forward(inp_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if step % 20 == 0:
            print(f"step: {step:>3}  loss: {loss.item():.4f}")
        step += 1

    if len(losses) < 2:
        print("[WARNING] Not enough steps to judge loss trend.")
        return

    first = sum(losses[:10]) / min(10, len(losses))
    last = sum(losses[-10:]) / min(10, len(losses))
    print(f"[INFO] Average loss of first 10 steps: {first:.4f}")
    print(f"[INFO] Average loss of last 10 steps:  {last:.4f}")
    if last < first:
        print("[SUCCESS] Loss is decreasing. Training setup looks good.")
    else:
        print("[WARNING] Loss did not decrease. Check data/config.")


def main():
    parser = argparse.ArgumentParser(
        description="Train ACT on real robot data (ela3_demo)"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.path.normpath(DEFAULT_DATASET_ROOT),
        help="Path to ela3_demo dataset folder",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=DEFAULT_CKPT_DIR,
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--training_steps",
        type=int,
        default=20000,
        help="Total training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="DataLoader num_workers",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="ACT action chunk size",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=10,
        help="Number of action steps to execute from each chunk",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1.0,
        help="KL weight for ACT VAE",
    )
    parser.add_argument(
        "--image_size",
        type=str,
        default="auto",
        help="Image resize: 'auto' (等比缩放最长边256, 默认), int (square), 'H,W', or -1 for no resize",
    )
    parser.add_argument(
        "--no_torque",
        action="store_true",
        help="禁用夹爪力矩输入（默认并入 observation.state 第8维）",
    )
    parser.add_argument(
        "--use_cam1",
        action="store_true",
        default=True,
        help="Use observation.images.cam1",
    )
    parser.add_argument(
        "--no_cam1",
        action="store_true",
        help="Disable observation.images.cam1",
    )
    parser.add_argument(
        "--use_cam2",
        action="store_true",
        default=True,
        help="Use observation.images.cam2",
    )
    parser.add_argument(
        "--no_cam2",
        action="store_true",
        help="Disable observation.images.cam2",
    )
    parser.add_argument(
        "--optimizer_lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="留出做验证的 episode 比例（监控过拟合），0 表示不划分",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=500,
        help="每 N 步在验证集上计算一次 loss",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=100,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=2000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a short test training to verify loss decreases",
    )
    parser.add_argument(
        "--test_steps",
        type=int,
        default=200,
        help="Number of steps for --test mode",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=8,
        help="Batch size for --test mode",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA unavailable, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    use_cam1 = args.use_cam1 and not args.no_cam1
    use_cam2 = args.use_cam2 and not args.no_cam2
    if not use_cam1 and not use_cam2:
        print("[ERROR] At least one camera must be enabled")
        sys.exit(1)

    policy, dataset, val_dataset, cfg, collate_fn = build_policy_and_dataset(
        dataset_root=args.dataset_root,
        device=device,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        kl_weight=args.kl_weight,
        optimizer_lr=args.optimizer_lr,
        use_cam1=use_cam1,
        use_cam2=use_cam2,
        use_torque=not args.no_torque,
        image_size=args.image_size,
        val_split=args.val_split,
    )

    if args.test:
        quick_test(
            policy,
            dataset,
            device,
            steps=args.test_steps,
            batch_size=args.test_batch_size,
            collate_fn=collate_fn,
        )
        # Also save a test checkpoint for reference.
        test_ckpt = os.path.join(args.ckpt_dir, "test_checkpoint")
        policy.save_pretrained(test_ckpt)
        print(f"[INFO] Test checkpoint saved to {test_ckpt}")
    else:
        train(
            policy,
            dataset,
            device,
            training_steps=args.training_steps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            log_freq=args.log_freq,
            save_freq=args.save_freq,
            ckpt_dir=args.ckpt_dir,
            optimizer_lr=args.optimizer_lr,
            collate_fn=collate_fn,
            val_dataset=val_dataset,
            eval_freq=args.eval_freq,
        )


if __name__ == "__main__":
    main()

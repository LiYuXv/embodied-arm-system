#!/usr/bin/env python3
"""Episode-based offline inference test for a trained ACT checkpoint.

This is an ACT adaptation of `test_inference_episode.py` (which was written for
SmolVLA). It loads a checkpoint produced by `train_act_real.py` and rolls it out
frame-by-frame over whole episodes, comparing predicted actions with ground
truth and plotting every action dimension.

Unlike SmolVLA, ACT performs normalization inside the policy, so we do NOT need
to call an external preprocessor/postprocessor. The policy handles image
normalization and state normalization via the transforms saved in the checkpoint.

Usage:
    python3 test_inference_act.py --checkpoint ./ckpt/act_ela3_real --episode 0

    # evaluate all episodes
    python3 test_inference_act.py --checkpoint ./ckpt/act_ela3_real --episode all

    # specify a different dataset root or image size
    python3 test_inference_act.py --checkpoint ./ckpt/act_ela3_real \
        --dataset_root ./lerobot_data/ela3_demo --image_size 256 --episode 0,1,2
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; we save PNGs
import matplotlib.pyplot as plt
import torch
from termcolor import colored

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy

from act_data_utils import (
    TORQUE_KEY,
    camera_keys_for_features,
    get_cam_hw,
    get_image_transforms,
    make_collate_fn,
)
from vla_features import (
    as_action_chunk,
    split_dual_action,
    validate_policy_checkpoint,
    validate_vla_features,
)


# Default paths kept consistent with train_act_real.py.
DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_ROOT = os.path.join(
    DEFAULT_PROJECT_ROOT, "lerobot_data", "ela3_demo"
)
DEFAULT_CKPT_DIR = os.path.join(DEFAULT_PROJECT_ROOT, "ckpt", "act_ela3_real")


def load_policy(checkpoint_path: str, device: str = "cuda"):
    """Load an ACT policy checkpoint and move it to the requested device."""
    ckpt = Path(checkpoint_path).resolve()
    # Handle checkpoints saved in a `pretrained_model` subfolder (e.g. some
    # HF-format exports). train_act_real.py saves directly to the root, so this
    # is only for compatibility.
    if (ckpt / "pretrained_model").is_dir():
        ckpt = ckpt / "pretrained_model"

    if not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")

    print(f"[INFO] Loading ACT policy from: {ckpt}")
    policy = ACTPolicy.from_pretrained(ckpt)
    policy.to(device)
    policy.eval()
    return policy


def build_dataset(
    dataset_root: str,
    repo_id: str,
    policy: ACTPolicy,
    image_size=256,
):
    """Build a LeRobotDataset that matches the policy's expected input features."""
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    print(f"[INFO] Loading dataset metadata from: {dataset_root}")
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    print(f"[INFO] Dataset: {dataset_metadata.total_episodes} episodes, "
          f"{dataset_metadata.total_frames} frames")

    # Resolve delta_timestamps using the policy's config (same as training).
    delta_timestamps = resolve_delta_timestamps(policy.config, dataset_metadata)

    cam_keys = camera_keys_for_features(policy.config.input_features)
    cam_hw = get_cam_hw(dataset_metadata, cam_keys[0]) if cam_keys else None

    print("[INFO] Building dataset...")
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps=delta_timestamps,
        root=dataset_root,
        image_transforms=get_image_transforms(image_size, cam_hw),
    )
    print(f"[INFO] Dataset length: {len(dataset)}")

    # 力矩适配：训练时 torque 并入 state（7->8 维），此处检测模型 state 维度
    # 是否等于数据集 state+torque 维度，一致则启用 collate 拼接
    merge_torque = False
    state_ft = policy.config.input_features.get("observation.state")
    if (state_ft is not None and TORQUE_KEY in dataset_metadata.features
            and "observation.state" in dataset_metadata.features):
        ds_state_dim = dataset_metadata.features["observation.state"]["shape"][0]
        ds_torque_dim = dataset_metadata.features[TORQUE_KEY]["shape"][0]
        if state_ft.shape[0] == ds_state_dim + ds_torque_dim:
            merge_torque = True
            print(f"[INFO] 模型 state 维度 {state_ft.shape[0]} = 数据集 state {ds_state_dim}"
                  f" + torque {ds_torque_dim}，启用力矩拼接")
    collate_fn = make_collate_fn(merge_torque=merge_torque)
    return dataset, collate_fn


class EpisodeSampler(torch.utils.data.Sampler):
    """Iterate frame ids of a single episode (same as test_inference_episode.py)."""

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        ep = dataset.meta.episodes[episode_index]
        self.frame_ids = range(ep["dataset_from_index"], ep["dataset_to_index"])

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def build_observation(batch: dict, policy: ACTPolicy, device: torch.device):
    """Select only the input features the policy expects and move them to device."""
    input_keys = set(policy.config.input_features.keys())
    observation = {}
    for k in input_keys:
        if k not in batch:
            raise KeyError(
                f"Policy expects input feature '{k}' but the dataset batch does not "
                f"contain it. Available keys: {list(batch.keys())}"
            )
        v = batch[k]
        observation[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return observation


def run_episode(policy, dataset, episode_index, device, collate_fn=None, mode="single"):
    """Roll out the policy over one episode; return (pred_actions, gt_actions)."""
    episode_sampler = EpisodeSampler(dataset, episode_index)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type != "cpu",
        sampler=episode_sampler,
        collate_fn=collate_fn,
    )

    actions, gt_actions = [], []
    policy.reset()  # clear the action-chunk queue once per episode
    for batch in test_dataloader:
        gt = batch["action"]
        # ACT 的 dataloader 返回的是 action chunk (1, chunk_size, action_dim)。
        # 跟 pred 只取当前帧动作对应，GT 也取 chunk 的第一个动作。
        if gt.dim() == 3:
            gt = gt[:, 0, :]
        elif gt.dim() == 1:
            gt = gt.unsqueeze(0)

        observation = build_observation(batch, policy, device)

        with torch.no_grad():
            action = policy.select_action(observation)

        # select_action may return the full chunk (1, n_action_steps, dim) or a
        # single step (1, dim). Take the first step of the chunk either way.
        if action.dim() == 3:
            pred = action[:, 0, :]
        else:
            pred = action
        as_action_chunk(pred, mode=mode)

        actions.append(pred)
        gt_actions.append(gt)

    actions = torch.cat(actions, dim=0)        # (T, action_dim)
    gt_actions = torch.cat(gt_actions, dim=0)  # (T, action_dim)
    return actions, gt_actions


def plot_actions(actions, gt_actions, action_dim, title, save_path):
    fig, axs = plt.subplots(action_dim, 1, figsize=(10, 2 * action_dim))
    if action_dim == 1:
        axs = [axs]
    labels = (
        [f"left_J{i}" for i in range(1, 7)] + ["left_gripper"]
        + [f"right_J{i}" for i in range(1, 7)] + ["right_gripper"]
        if action_dim == 14 else
        ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    )
    for i in range(action_dim):
        axs[i].plot(actions[:, i].cpu().detach().numpy(), label="pred")
        axs[i].plot(gt_actions[:, i].cpu().detach().numpy(), label="gt")
        name = labels[i] if i < len(labels) else f"dim{i}"
        axs[i].set_ylabel(name)
        axs[i].legend(loc="upper right")
    axs[-1].set_xlabel("frame")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Episode-based offline inference for an ACT policy"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CKPT_DIR,
        help="Path to the ACT checkpoint directory produced by train_act_real.py",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.path.normpath(DEFAULT_DATASET_ROOT),
        help="Path to the dataset folder (e.g. ./lerobot_data/ela3_demo)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="ela3_demo",
        help="Dataset repo_id (folder name)",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default="0",
        help="Episode index, or 'all' to evaluate every episode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./inference_plots_act",
        help="Directory to save prediction plots",
    )
    parser.add_argument(
        "--image_size",
        type=str,
        default="auto",
        help="Image resize: 'auto' (等比缩放最长边256, 默认), int (square), 'H,W', or -1 for no resize",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers for episode rollout",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA unavailable, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    policy = load_policy(args.checkpoint, args.device)
    print(colored(
        f"Loaded ACT policy: chunk_size={policy.config.chunk_size}, "
        f"n_action_steps={policy.config.n_action_steps}", "green"))
    print(f"[INFO] Policy input features: {list(policy.config.input_features.keys())}")
    print(f"[INFO] Policy output features: {list(policy.config.output_features.keys())}")

    dataset, collate_fn = build_dataset(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        policy=policy,
        image_size=args.image_size,
    )
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames, "
          f"fps={dataset.fps}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.episode.lower() == "all":
        episodes = list(range(dataset.num_episodes))
    else:
        episodes = [int(e) for e in args.episode.split(",")]

    action_dim = dataset.meta.features["action"]["shape"][0]
    mode = "dual" if action_dim == 14 else "single" if action_dim == 7 else None
    if mode is None:
        raise ValueError(f"unsupported dataset action dimension: {action_dim}")
    validate_vla_features(dataset.meta.features, mode)
    validate_policy_checkpoint(policy.config, mode)

    total_err, total_frames = 0.0, 0
    for ep in episodes:
        actions, gt_actions = run_episode(
            policy, dataset, ep, device, collate_fn, mode=mode
        )
        if mode == "dual":
            left, right = split_dual_action(actions)
            print(f"  dual split: left={tuple(left.shape)}, right={tuple(right.shape)}")
        mean_err = torch.mean(torch.abs(actions - gt_actions.to(actions.device))).item()
        n = actions.shape[0]
        total_err += mean_err * n
        total_frames += n
        print(f"episode {ep:03d}: frames={n}, mean abs action error={mean_err:.4f}")

        plot_actions(
            actions.cpu(), gt_actions.cpu(), action_dim,
            title=f"episode {ep}  (mean abs err={mean_err:.4f})",
            save_path=save_dir / f"episode_{ep:03d}.png",
        )
        print(colored(f"  saved plot -> {save_dir / f'episode_{ep:03d}.png'}", "cyan"))

    if total_frames:
        print(colored(
            f"\nOverall mean abs action error = {total_err / total_frames:.4f} "
            f"over {total_frames} frames", "yellow", attrs=["bold"]))
    print(colored("ACT episode inference test done.", "green", attrs=["bold"]))


if __name__ == "__main__":
    main()

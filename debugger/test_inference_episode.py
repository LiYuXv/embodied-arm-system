#!/usr/bin/env python
"""Episode-based offline inference test for a trained SmolVLA checkpoint.

This faithfully adapts the "Test Inference" section of the ACT tutorial notebook
(`3.train.ipynb`, cells 12-14) to SmolVLA:

    - run the policy frame-by-frame over a WHOLE episode (EpisodeSampler)
    - collect predicted actions vs ground-truth actions
    - plot every action dimension (pred vs gt) with matplotlib

Differences vs the ACT notebook:
    - SmolVLA does NOT normalize inside the policy, so we explicitly apply the
      checkpoint's `preprocessor` (tokenizer + state normalizer) before
      `select_action` and the `postprocessor` (action unnormalizer) afterwards.
    - action_dim 由 checkpoint/dataset 决定：单臂 7D，双臂 14D。

Usage:
    conda run -n yolo --no-capture-output python test_inference_episode.py \
        --checkpoint ./ckpt/smolvla_ela3/checkpoints/last --episode 0

    # 摄像头消融（测试哪路摄像头对模型准确率影响最大）：
    #   默认输入 checkpoint 声明的所有相机；--camN 只保留指定视角
    # 随机种子：默认 --seed 1000（每个 episode 推理前重置，保证各 cam 模式可对比）
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; we also save PNGs
import matplotlib.pyplot as plt
import torch
from termcolor import colored

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class
from lerobot.processor import DataProcessorPipeline

import configuration_smolvla_torque  # noqa: F401 注册 "smolvla_torque" 策略（含夹爪力矩输入）
import configuration_pi05_torque  # noqa: F401 注册 "pi05_torque"
import processor_pi05_torque  # noqa: F401 注册自定义 processor step

from vla_features import (
    ACTION_KEY,
    CAMERA_PREFIX,
    TORQUE_KEY,
    as_action_chunk,
    get_mode_spec,
    split_dual_action,
    validate_policy_checkpoint,
    validate_vla_features,
)


def load_policy(checkpoint_path: str, device: str = "cuda"):
    ckpt = Path(checkpoint_path).resolve()
    if (ckpt / "pretrained_model").is_dir():
        ckpt = ckpt / "pretrained_model"
    config = PreTrainedConfig.from_pretrained(ckpt)
    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(ckpt)
    policy.to(device)
    policy.eval()

    preprocessor = DataProcessorPipeline.from_pretrained(
        ckpt, config_filename="policy_preprocessor.json"
    )
    postprocessor = DataProcessorPipeline.from_pretrained(
        ckpt, config_filename="policy_postprocessor.json"
    )
    return policy, preprocessor, postprocessor, config


class EpisodeSampler(torch.utils.data.Sampler):
    """Iterate frame ids of a single episode (same as the notebook)."""

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        # LeRobotDatasetMetadata stores the global frame range of each episode.
        ep = dataset.meta.episodes[episode_index]
        self.frame_ids = range(ep["dataset_from_index"], ep["dataset_to_index"])

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def camera_keys_for_inference(config, dataset):
    configured = getattr(config, "input_features", None) or {}
    available = dataset.meta.features
    return [
        key for key in configured
        if key.startswith(CAMERA_PREFIX) and key in available
    ]


def build_episode_observation(
    batch, device, camera_keys, *, torque_key=None, torque_dim=1, cam_mode="all"
):
    """Build a dynamic two/three-camera single-frame observation."""
    observation = {key: batch[key].to(device) for key in camera_keys}
    observation["observation.state"] = batch["observation.state"].to(device)
    observation["task"] = batch["task"]

    mode = "blind" if cam_mode == "cam0" else cam_mode
    if mode not in {"all", "both"}:
        keep_key = f"{CAMERA_PREFIX}{mode}" if mode.startswith("cam") else None
        for key in camera_keys:
            if mode == "blind" or key != keep_key:
                observation[key] = torch.zeros_like(observation[key])
    if torque_key is not None:
        if torque_key in batch:
            observation[torque_key] = batch[torque_key].reshape(
                batch[torque_key].shape[0], -1
            ).to(device)
        else:
            observation[torque_key] = torch.zeros(
                observation["observation.state"].shape[0], torque_dim, device=device
            )
    return observation


def run_episode(policy, preprocessor, postprocessor, dataset, episode_index, device,
                torque_key=None, cam_mode="all", camera_keys=None, torque_dim=1,
                mode="single"):
    """Roll out one episode with dynamic cameras and 7D/14D action validation."""
    episode_sampler = EpisodeSampler(dataset, episode_index)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type != "cpu",
        sampler=episode_sampler,
    )

    # 模型需要力矩输入时，逐帧把 observation.gripper_torque 传入观测
    need_torque = torque_key is not None
    if need_torque and torque_key not in dataset.meta.features:
        print(colored(
            f"警告: 模型需要力矩输入 '{torque_key}'，但数据集中不存在该特征，将以 0 填充。",
            "yellow", attrs=["bold"]))
    warned_missing = False

    camera_keys = camera_keys or [
        key for key in dataset.meta.features if key.startswith(CAMERA_PREFIX)
    ]
    actions, gt_actions = [], []
    policy.reset()  # clear the action-chunk queue once per episode
    for batch in test_dataloader:
        # ground-truth single-step action at this frame: (1, action_dim)
        gt = batch["action"]

        observation = build_episode_observation(
            batch,
            device,
            camera_keys,
            torque_key=torque_key if need_torque else None,
            torque_dim=torque_dim,
            cam_mode=cam_mode,
        )
        if need_torque and torque_key not in batch and not warned_missing:
            warned_missing = True
        observation = preprocessor(observation)

        with torch.no_grad():
            action = policy.select_action(observation)

        # unnormalize prediction back to real units (gt is already real)
        action = postprocessor(action if isinstance(action, dict) else {"action": action})
        action = action["action"] if isinstance(action, dict) else action

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./ckpt/smolvla_ela3_torque/checkpoints/last")
    parser.add_argument("--dataset_root", type=str, default="./lerobot_data/ela3_demo_20260808_185706")
    parser.add_argument("--repo_id", type=str, default="ela3_demo_20260808_185706")
    parser.add_argument(
        "--episode", type=str, default="0",
        help="episode index, or 'all' to evaluate every episode",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./inference_plots")
    cam_group = parser.add_mutually_exclusive_group()
    cam_group.add_argument("--cam1", action="store_true",
                           help="只输入 cam1，其余摄像头置黑")
    cam_group.add_argument("--cam2", action="store_true",
                           help="只输入 cam2，其余摄像头置黑")
    cam_group.add_argument("--cam0", action="store_true",
                           help="所有摄像头均置黑（盲测）")
    cam_group.add_argument("--cam3", action="store_true",
                           help="只输入 cam3，其余摄像头置黑")
    parser.add_argument("--seed", type=int, default=1000,
                        help="flow-matching 采样随机种子（默认 1000），每个 episode 推理前重置")
    args = parser.parse_args()

    cam_mode = "cam1" if args.cam1 else ("cam2" if args.cam2 else (
        "cam3" if args.cam3 else ("blind" if args.cam0 else "all")))
    cam_suffix = "" if cam_mode == "all" else f"_{cam_mode}"

    device = torch.device(args.device)
    policy, preprocessor, postprocessor, config = load_policy(args.checkpoint, args.device)
    print(colored(
        f"Loaded policy type={config.type}, chunk_size={config.chunk_size}, "
        f"n_action_steps={config.n_action_steps}", "green"))
    print(colored(f"摄像头模式: {cam_mode} | 随机种子: {args.seed}", "green"))

    # 力矩输入：仅当模型（checkpoint config）声明了该输入特征时才传入，
    # 旧版 smolvla checkpoint 行为保持不变。
    torque_key = getattr(config, "torque_key", "observation.gripper_torque")
    input_features = getattr(config, "input_features", None) or {}
    torque_key = torque_key if torque_key in input_features else None
    if torque_key:
        print(colored(f"力矩输入已启用: {torque_key}", "green"))

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames, fps={dataset.fps}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.episode.lower() == "all":
        episodes = list(range(dataset.num_episodes))
    else:
        episodes = [int(e) for e in args.episode.split(",")]

    action_dim = dataset.meta.features[ACTION_KEY]["shape"][0]
    mode = "dual" if action_dim == 14 else "single" if action_dim == 7 else None
    if mode is None:
        raise ValueError(f"unsupported dataset action dimension: {action_dim}")
    validate_vla_features(dataset.meta.features, mode)
    validate_policy_checkpoint(config, mode)
    camera_keys = camera_keys_for_inference(config, dataset)
    torque_dim = dataset.meta.features.get(
        TORQUE_KEY, {"shape": [get_mode_spec(mode).torque_dim]}
    )["shape"][0]
    print(colored(
        f"VLA schema: mode={mode}, cameras={camera_keys}, "
        f"state/action={action_dim}D, torque={torque_dim}D",
        "green",
    ))

    total_err, total_frames = 0.0, 0
    for ep in episodes:
        # 每个 episode 推理前重置种子：flow-matching 采样噪声序列一致，
        # 保证 both/cam1/cam2 不同模式之间可公平对比
        torch.manual_seed(args.seed)
        actions, gt_actions = run_episode(
            policy, preprocessor, postprocessor, dataset, ep, device,
            torque_key=torque_key, cam_mode=cam_mode,
            camera_keys=camera_keys, torque_dim=torque_dim, mode=mode)
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
            title=f"episode {ep} [{cam_mode}]  (mean abs err={mean_err:.4f})",
            save_path=save_dir / f"episode_{ep:03d}{cam_suffix}.png",
        )
        print(colored(f"  saved plot -> {save_dir / f'episode_{ep:03d}{cam_suffix}.png'}", "cyan"))

    if total_frames:
        print(colored(
            f"\nOverall mean abs action error = {total_err / total_frames:.4f} "
            f"over {total_frames} frames", "yellow", attrs=["bold"]))
    print(colored("Episode inference test done.", "green", attrs=["bold"]))


if __name__ == "__main__":
    main()

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
    - action_dim = 7 (x, y, z, rx, ry, rz, gripper_rad).

Usage:
    conda run -n yolo --no-capture-output python test_inference_episode.py \
        --checkpoint ./ckpt/smolvla_ela3/checkpoints/last --episode 0

    # 摄像头消融（测试哪路摄像头对模型准确率影响最大）：
    #   默认双路输入；--cam1 只喂 cam1 画面（cam2 置黑）；--cam2 反之
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


def run_episode(policy, preprocessor, postprocessor, dataset, episode_index, device,
                torque_key=None, cam_mode="both"):
    """Roll out the policy over one episode; return (pred_actions, gt_actions).

    cam_mode: "both" 双路输入；"cam1" 只喂 cam1（cam2 置黑）；"cam2" 反之；
    "cam0" 双路置黑（盲测，验证模型对视觉的依赖程度）。
    """
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

    actions, gt_actions = [], []
    policy.reset()  # clear the action-chunk queue once per episode
    for batch in test_dataloader:
        # ground-truth single-step action at this frame: (1, action_dim)
        gt = batch["action"]

        observation = {
            "observation.images.cam1": batch["observation.images.cam1"].to(device),
            "observation.images.cam2": batch["observation.images.cam2"].to(device),
            "observation.state": batch["observation.state"].to(device),
            "task": batch["task"],
        }
        # 摄像头消融：cam1 只保留 cam1（cam2 置黑）；cam2 反之；cam0 双路置黑
        if cam_mode == "cam1":
            observation["observation.images.cam2"] = torch.zeros_like(
                observation["observation.images.cam2"])
        elif cam_mode == "cam2":
            observation["observation.images.cam1"] = torch.zeros_like(
                observation["observation.images.cam1"])
        elif cam_mode == "cam0":
            observation["observation.images.cam1"] = torch.zeros_like(
                observation["observation.images.cam1"])
            observation["observation.images.cam2"] = torch.zeros_like(
                observation["observation.images.cam2"])
        if need_torque:
            if torque_key in batch:
                # (1, 1)，已带 batch 维（add_batch_dim 步骤不处理该自定义键）
                observation[torque_key] = batch[torque_key].reshape(1, -1).to(device)
            else:
                if not warned_missing:
                    warned_missing = True
                observation[torque_key] = torch.zeros(1, 1, device=device)
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

        actions.append(pred)
        gt_actions.append(gt)

    actions = torch.cat(actions, dim=0)        # (T, action_dim)
    gt_actions = torch.cat(gt_actions, dim=0)  # (T, action_dim)
    return actions, gt_actions


def plot_actions(actions, gt_actions, action_dim, title, save_path):
    fig, axs = plt.subplots(action_dim, 1, figsize=(10, 2 * action_dim))
    if action_dim == 1:
        axs = [axs]
    labels = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
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
                           help="只输入 cam1 画面（cam2 置黑），用于摄像头消融")
    cam_group.add_argument("--cam2", action="store_true",
                           help="只输入 cam2 画面（cam1 置黑），用于摄像头消融")
    cam_group.add_argument("--cam0", action="store_true",
                           help="双路均置黑（盲测），验证模型对视觉的依赖程度")
    parser.add_argument("--seed", type=int, default=1000,
                        help="flow-matching 采样随机种子（默认 1000），每个 episode 推理前重置")
    args = parser.parse_args()

    cam_mode = "cam1" if args.cam1 else ("cam2" if args.cam2 else ("cam0" if args.cam0 else "both"))
    cam_suffix = "" if cam_mode == "both" else f"_{cam_mode}"

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

    action_dim = dataset.meta.features["action"]["shape"][0]

    total_err, total_frames = 0.0, 0
    for ep in episodes:
        # 每个 episode 推理前重置种子：flow-matching 采样噪声序列一致，
        # 保证 both/cam1/cam2 不同模式之间可公平对比
        torch.manual_seed(args.seed)
        actions, gt_actions = run_episode(
            policy, preprocessor, postprocessor, dataset, ep, device,
            torque_key=torque_key, cam_mode=cam_mode)
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

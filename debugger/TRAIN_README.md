# SmolVLA 真机数据训练说明

## 1. 环境

使用 **VLA** conda 环境（lerobot 0.6.1 + torch + CUDA）。

> 注意：`lerobot` 环境装的是旧版 lerobot 0.1.0，没有 smolvla 模块，**不能使用**。
> `train_model.py` 是官方 `lerobot_train` 入口的薄封装，仅额外注册自定义策略
> `smolvla_torque`（夹爪力矩输入），用法与官方一致。

```bash
conda activate VLA
# 或全程用 conda run -n VLA ...
```

## 2. 数据集

- 路径：`./lerobot_data/ela3_demo_20260808_185706`（LeRobot v3.0 格式）
- 规模：13 episodes / 10086 帧 / 20 fps
- 特征：
  - `observation.images.cam1`：720×1280 摄像头 1
  - `observation.images.cam2`：720×1280 摄像头 2
  - `observation.state`：(7,) = `[x, y, z, rx, ry, rz, gripper_rad]`
  - `observation.gripper_torque`：(1,) = 夹爪力矩观测 (Nm)，**作为模型输入**
  - `action`：(7,) 同 state 维度
- 任务文本：见 `meta/tasks.parquet`

> 力矩特征以 `observation.` 开头，lerobot 自动识别为 STATE 类型，进入
> `input_features` 并按 STATE -> MEAN_STD 归一化（stats 已在数据集
> `meta/stats.json` 中），无需额外配置。

## 3. 训练

```bash
cd debugger
conda run -n VLA --no-capture-output python train_model.py --config_path smolvla_omy.yaml 2>&1 | tee train.log
```

### 关键配置（`smolvla_omy.yaml`）

| 参数 | 值 | 说明 |
|---|---|---|
| `policy.type` | smolvla_torque | 策略类型（SmolVLA + 力矩输入） |
| `policy.torque_key` | observation.gripper_torque | 力矩观测键名（默认即此，可省） |
| `policy.chunk_size` / `n_action_steps` | 50 / 25 | 动作块长度 / 每次执行步数 |
| `policy.vlm_model_name` | 本地路径 | SmolVLM2-500M 权重（无需联网） |
| `batch_size` | 16 | 720p 双摄很占内存，**不要超过 16~32**，OOM 就降到 8 |
| `num_workers` | 4 | dataloader 进程数，内存不足时降到 2 |
| `steps` | 20_000 | 训练步数（数据仅 10k 帧，注意过拟合） |
| `save_freq` | 4_000 | 每 4000 步保存 checkpoint |
| `log_freq` | 50 | 每 50 步打印 loss |
| `env_eval_freq` | -1 | 真机数据无仿真环境，训练中不评估 |

### 力矩输入的实现原理

策略 `smolvla_torque`（`modeling_smolvla_torque.SmolVLATorquePolicy`）继承
`SmolVLAPolicy`，仅重写 `prepare_state`：将归一化后的力矩拼接到
`observation.state` 向量末尾（7+1=8 维），再 pad 到 `max_state_dim=32`，
经原有 `state_proj` 投影为 state token 进入 VLM。训练与推理共用同一
`prepare_state`，行为一致；力矩缺失时退化为原版 SmolVLA 行为。

### 断点续训

lerobot 0.6.1 的续训**不读 `smolvla_omy.yaml`**：`resume: true` 时 `--config_path`
必须指向 checkpoint 内保存的 `train_config.json`（`checkpoint_path` 是
`field(init=False)`，不能从命令行/yaml 传入，传了报 `fields not valid`）：

```bash
conda run -n VLA --no-capture-output python train_model.py \
    --config_path ./ckpt/smolvla_ela3_torque/checkpoints/last/pretrained_model/train_config.json \
    --resume true 2>&1 | tee -a train.log
```

- 配置（steps/output_dir/optimizer 等）全部取自 checkpoint 里的 `train_config.json`，
  训练步数、优化器与 scheduler 状态从 `training_state` 恢复；
- 若从 `smolvla_omy.yaml` + `resume: true` 启动，lerobot 会把 yaml 所在目录当
  checkpoint 根，报 `FileNotFoundError: ./model.safetensors`；
- checkpoint 损坏时（如磁盘写满中断）：删掉不完整的步数目录，确认
  `checkpoints/last` 软链指向最后一个完整 checkpoint 再续训；
- 每个 checkpoint 约 1.5G，训练前确保磁盘剩余 ≥ 2 个 checkpoint 的空间；
- HF datasets 会把 parquet 缓存到 `~/.cache/huggingface/datasets/parquet`，
  数据集每次被编辑都会生成新副本（曾累积 148G 导致磁盘写满），空间紧张时可
  整体删除该目录（下次加载自动重建）。

## 4. 查看训练过程

```bash
# 实时查看日志
tail -f train.log

# 只看 loss 曲线（应持续下降）
grep "loss:" train.log

# 查看已保存的 checkpoint
ls ckpt/smolvla_ela3_torque/checkpoints/     # 004000/008000/... + last(最新)
```

需要可视化曲线可将 yaml 中 `wandb.enable` 改为 `true`（需 `wandb login`）。

## 5. 推理测试

离线验证（用真实数据集样本跑前向，对比预测动作与真值）：

```bash
conda run -n VLA python test_inference_episode.py \
    --checkpoint ./ckpt/smolvla_ela3_torque/checkpoints/last --episode 5 \
    --dataset_root ./lerobot_data/ela3_demo --repo_id ela3_demo
# 或全部 episode：
conda run -n VLA python test_inference_episode.py \
    --checkpoint ./ckpt/smolvla_ela3_torque/checkpoints/last --episode all \
    --dataset_root ./lerobot_data/ela3_demo --repo_id ela3_demo
```

> 注意：脚本的 `--dataset_root` **默认值是旧快照** `ela3_demo_20260808_185706`
> （带 `language_persistent`/`language_events` 语言标注列，新版 lerobot 强转
> messages 结构会报 `Couldn't cast array of type string`）。务必显式指定
> 当前训练数据集 `ela3_demo`。

参考结果（18K 步 checkpoint，episode 5 / 867 帧，seed=1000）：mean abs action error ≈ 0.0291。

### 摄像头消融与随机种子

```bash
# 默认双路输入；--cam1 只喂 cam1 画面（cam2 置黑），--cam2 反之（互斥）
conda run -n VLA python test_inference_episode.py \
    --checkpoint ./ckpt/smolvla_ela3_torque/checkpoints/last --episode 5 \
    --dataset_root ./lerobot_data/ela3_demo --repo_id ela3_demo --cam1
    
    
conda run -n VLA python test_inference_episode.py \
    --checkpoint ./ckpt/smolvla_ela3_torque_la5/checkpoints/last --episode 5 \
    --dataset_root ./lerobot_data/ela3_demo --repo_id ela3_demo
# 随机种子：默认 --seed 1000，每个 episode 推理前重置，
# 保证 both/cam1/cam2 三种模式使用完全相同的 flow-matching 采样噪声，公平可比；
# 换种子如 --seed 42 可验证结论稳健性
```

判读方法：**缺哪路误差升得越多，哪路越重要**。参考结果（18K checkpoint，
episode 5，seed=1000）：双路 0.0291 / 仅 cam1 0.0522 / 仅 cam2 0.0422，
即 cam2 贡献更大但两路均有贡献。建议多 episode 或 `--episode all` 再下结论。
消融模式输出图带后缀（`episode_005_cam1.png`），不与双路结果互相覆盖。

输出每个 episode 的 mean abs action error 及逐维 pred/gt 对比图（PNG）。
脚本读取 checkpoint config：若声明了 `observation.gripper_torque` 输入，
会自动从数据集读取力矩一并传入模型；旧版 smolvla checkpoint 行为不变。
观测输入格式：

```python
observation = {
    "observation.images.cam1": img1,            # (1,3,H,W) float [0,1]
    "observation.images.cam2": img2,
    "observation.state": state,                 # (1,7) [x,y,z,rx,ry,rz,gripper_rad]
    "observation.gripper_torque": torque,       # (1,1) 夹爪力矩 Nm（smolvla_torque 模型必传）
    "task": ["put the black square"],
}
observation = preprocessor(observation)
action = policy.select_action(observation)   # 内部自动管理动作块队列
action = postprocessor({"action": action})["action"]
```

真机部署（`camera_pyqt.py` 的 VLA 推理）已同步支持：加载 checkpoint 后检测
模型是否声明力矩输入，若是则每帧通过 `set_torque_callback` 注册的回调读取
夹爪力矩并传入（读取失败兜底 0.0，与录制端一致）；旧模型不受影响。

## 6. 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| `DataLoader worker ... Killed` 或进程`已杀死` | 系统内存 OOM（720p 双摄解码占内存大，或有残留训练进程占内存） | 先执行下方清理命令；仍不足则降 `batch_size` / `num_workers` |
| `CUDA out of memory` | 显存不足 | 降 `batch_size` 至 8 |
| 提示安装 lerobot | 用错 conda 环境 | 必须用 `VLA` 环境 |
| `error: The fields ... are not valid` | yaml 字段与 lerobot 0.6.1 不符 | 参照官方 `lerobot_train` 的 `TrainPipelineConfig` 调整 |

### 清理残留训练进程（OOM 首选）

训练异常中断（如直接关终端）后，挂起的 python/worker 进程可能仍占用大量内存，导致下次训练被系统杀掉。清理方法：

```bash
# 查看残留训练进程（STAT 为 T 的是挂起进程）
ps aux | grep train_model | grep -v grep

# 一键清除所有残留训练进程
pkill -9 -f "train_model.py"

# 确认内存已释放（available 应恢复到 25G 以上）
free -g
```

## 7. 文件清单

| 文件 | 说明 |
|---|---|
| `train_model.py` | 训练脚本（官方 lerobot_train 入口 + 注册自定义策略） |
| `smolvla_omy.yaml` | 训练配置 |
| `configuration_smolvla_torque.py` | `smolvla_torque` 策略配置（新增 `torque_key`） |
| `modeling_smolvla_torque.py` | `SmolVLATorquePolicy`：`prepare_state` 拼接力矩到 state |
| `processor_smolvla_torque.py` | 预/后处理器工厂（复用 SmolVLA 流程） |
| `test_inference_episode.py` | 离线推理测试（自动检测并传入力矩） |
| `train_act_real.py` | ACT 训练脚本（独立于 SmolVLA，见第 8 节） |
| `test_inference_act.py` | ACT 离线推理验证（episode rollout + pred/gt 对比图） |
| `act_data_utils.py` | ACT 共享工具：力矩并入 state、auto 等比缩放、collate |
| `ckpt/smolvla_ela3_torque/` | 训练输出（checkpoints） |
| `train.log` | 训练日志（tee 生成） |


## 8. ACT 模型训练与推理

ACT 与 SmolVLA 相互独立：不走 `train_model.py`/yaml，而是两个独立脚本 +
共享工具模块 `act_data_utils.py`（力矩拼接、分辨率适配、collate）。

### 8.1 数据适配（自动完成，无需配置）

- **力矩输入**：原生 ACT 只读 `observation.state`，独立的
  `observation.gripper_torque` 特征会被模型静默忽略。脚本把 torque 拼到 state
  末尾（7+1=8 维），`input_features` 形状与归一化统计（mean/std/min/max/qXX）
  同步合并，DataLoader 通过 collate_fn 在 batch 级拼接。**默认启用**，
  `--no_torque` 可禁用。
- **分辨率适配**：`--image_size` 默认 `auto`——按数据集实际分辨率等比缩放至
  最长边 256，保持长宽比不拉伸（720x720→256x256；旧 1280x720→256x144）。
  也可显式指定：`256`（方形）、`"360,640"`（H,W）、`-1`（原尺寸不缩放）。
- 单任务数据集直接训练即可（ACT 无任务条件输入）；多任务数据会混在一起学。

### 8.2 训练

```bash
cd debugger
conda activate VLA

# 快速自检（200 步小批量，验证 loss 下降，并保存 test_checkpoint）
python3 train_act_real.py --test

# 正式训练（默认 20000 步 / batch 32，输出到 ckpt/act_ela3_real）
python3 train_act_real.py --training_steps 20000 --batch_size 32 2>&1 | tee train_act.log
```

训练日志示例：

```
step:     0  loss: 7.5256  (l1_loss: 0.7704, kld_loss: 6.7552)  lr: 9.83e-05  elapsed: 1.1s
step:     0  [VAL] l1_loss: 1.7605
step:   500  [VAL] l1_loss: 0.3215
```

- `loss` = l1_loss + kl_weight × kld_loss；**l1_loss 是动作预测误差**（最重要），
  kld_loss 是 VAE 隐变量正则。
- `[VAL] l1_loss` 为留出验证集上的动作误差，每 `--eval_freq` 步打印一次。
- 监控命令：`grep -E "step:|VAL" train_act.log`
- **过拟合判读**：训练 loss 持续下降但 VAL loss 走平或回升 = 开始过拟合，
  选 VAL 最低附近的 `checkpoint_step_N` 部署，而不是最后一步。

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset_root` | ./lerobot_data/ela3_demo | 数据集路径 |
| `--ckpt_dir` | ./ckpt/act_ela3_real | checkpoint 输出目录（每 `--save_freq` 步存 `checkpoint_step_N`，训练结束存最终版到根目录） |
| `--chunk_size` / `--n_action_steps` | 10 / 10 | 动作块长度 / 每块执行步数（**部署时必须与训练一致**） |
| `--kl_weight` | 1.0 | VAE KL 权重，ACT 较敏感，可对比 0.1 / 1.0 / 10.0 |
| `--batch_size` / `--num_workers` | 32 / 6 | OOM 降 batch；视频解码吃 CPU，workers 4~6 合适，0=单进程调试 |
| `--val_split` | 0.1 | 留出做验证的 episode 比例（按 episode 划分，固定种子可复现），0 关闭 |
| `--eval_freq` | 500 | 每 N 步计算一次验证 loss |
| `--image_size` | auto | 见 8.1 |
| `--no_torque` | 关 | 禁用力矩输入（state 回到 7 维） |
| `--no_cam1` / `--no_cam2` | 关 | 相机消融用，禁用某一路 |
| `--optimizer_lr` | 1e-4 | 主干网络学习率固定为其 1/10（脚本内 1e-5），cosine 衰减 |
| `--save_freq` / `--log_freq` | 2000 / 100 | 保存 / 打印间隔（步） |

> 模型结构参数（vision_backbone=resnet18、dim_model=512、n_encoder_layers=4 等）
> 未暴露到命令行，需要时改 `build_policy_and_dataset()` 的默认参数。
> 脚本兼容 lerobot 0.4.4（yolo 环境）与 0.6.1（VLA 环境），训练请用 VLA。

### 8.3 离线推理验证

用数据集整段 episode 逐帧 rollout，输出每个 episode 的 mean abs action error
及逐维 pred/gt 对比图（PNG 保存到 `--save_dir`，默认 `./inference_plots_act`）：

```bash
# 单个 episode
python3 test_inference_act.py --checkpoint ./ckpt/act_ela3_real --episode 0

# 多个 / 全部 episode
python3 test_inference_act.py --checkpoint ./ckpt/act_ela3_real --episode 0,1,2
python3 test_inference_act.py --checkpoint ./ckpt/act_ela3_real --episode all
```

- `--image_size` 默认同样为 auto，**必须与训练时一致**（两边都用 auto 即可）。
- 力矩自动适配：检测到模型 state 维度 = 数据集 state+torque（8=7+1）时自动
  启用拼接；旧的无 torque checkpoint（state 7 维）行为不变，无需任何参数。
- 判读：误差越小越好，多跑几个 episode（或 `--episode all`）再下结论。
  建议对比多个 `checkpoint_step_N`，选验证误差最低的而不是最后一步（防过拟合）。

### 8.4 ACT 调参建议（真机）

一次只改一个变量，用 held-out episode 的 Mean Action Error 对比：

1. **提高重规划频率**（推荐先试）：`--n_action_steps 5 --chunk_size 10`——
   每块只执行前 5 步就重新观测预测，误差累积更少。
2. **kl_weight**：对比 0.1 / 1.0 / 10.0。太小 VAE latent 不受约束训练不稳，
   太大动作重建被压制、预测保守。
3. **训练步数**：loss 仍在降就加到 40000~60000 步（脚本已带 cosine 衰减）。
4. **提高输入分辨率**：显存有余量时 `--image_size 512`，同时降 batch_size。
5. **模型容量**：改 `build_policy_and_dataset()` 默认值：vision_backbone 换
   resnet34/50、dim_model 512→1024、n_encoder_layers 4→6（显存占用显著上升，
   batch 相应降到 16~24）。
6. **数据增强**：只加轻微 ColorJitter / 高斯噪声；不要 random crop / flip
   （破坏任务语义）。
7. **用验证误差选模型**，不要看训练 loss：训练中看 `[VAL] l1_loss`（8.2），
   部署前再用 `test_inference_act.py --episode <验证集编号>` 对比多个 checkpoint。
8. **输入一致性**：训练与部署的图像预处理、state/action 定义必须完全一致，
   任何不一致都会表现为训练 loss 很好但真机执行很差。

### 附：主程序控制与 VLA 推理链路（速查）

- 控制频率：ArmWorker 轮询 50 Hz；SDK 底层控制循环 200 Hz（EMA 平滑、速度
  前馈、重力补偿、限位保护，直接向 CAN 发指令）；UI 刷新上限 20 Hz。
- 关节界面"发送"只是更新目标位置并立即返回，实际运动由 200 Hz 控制循环
  持续跟踪目标完成，并非阻塞等待。
- 电机为 MIT-like 混合 PD 力矩模式：τ = Kp·(θ_tgt−θ_act) + Kd·(ω_tgt−ω_act) + τ_ff，
  位置目标、速度前馈、Kp/Kd、前馈力矩一并下发，固件闭环执行。
- VLA 推理链路：`CameraWidget.toggle_vla_inference()` 启动 VLAInferenceThread
  （20 Hz，与录制 fps 一致）→ 模型输出 7 维动作 → `_on_vla_action()` 做每步
  位移限幅（0.05 m）与夹爪限幅（−π/2 ~ π/60 rad）→ `motion_cb("end_pose_ctrl",
  ..., duration=1/fps, block=False)` + `motion_cb("gripper_ctrl", rad)` →
  最终由 `arm.EndPoseCtrl` 做笛卡尔逆解（非 JointCtrl/MoveJ/MoveL），
  block=False 伺服模式推入 200 Hz 控制循环，避免命令积压。
- 手柄连接时自动暂停 VLA 下发（安全互锁），断开后自动恢复。




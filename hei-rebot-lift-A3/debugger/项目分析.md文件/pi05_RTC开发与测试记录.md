# PI0.5 + 训练时 RTC 开发与测试记录

> 对应论文：[Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964)
> （arXiv:2512.05964, Kevin Black 等 / Physical Intelligence, 2025-12, 论文实验基于 π0.6）
> 本文档记录将论文方法回传（backport）到 lerobot PI0.5 训练端的全部改动、验证结果与调参指南，
> 用于与论文方法逐点比对，并支撑后续调参训练。

---

## 1. 背景与方法对应关系

RTC（Real-Time Chunking）让 VLA 在异步预测动作块时，以上一块已提交动作（prefix）为条件，
消除块间抖动。两种实现路线：

| 路线 | 论文称呼 | 本实现对应 | 特点 |
|---|---|---|---|
| 推理时 inpainting | inference-time RTC | `rtc_config.mode="guided"`（lerobot 原有） | 推理端额外计算，增加延迟 |
| **训练时动作条件** | **training-time RTC** | `rtc_training_max_delay > 0` + `mode="trained"` | 推理零开销；论文证明高延迟下优于 inpainting |

论文核心做法（本实现与其一致）：
1. 训练时逐样本随机模拟推理延迟 `d ~ Uniform{0, ..., max_delay}`；
2. 前 `d` 步动作作为**干净 prefix** 输入（flow matching 的 time 置 0，不加噪）；
3. loss 只计算 postfix（第 d 步之后）部分；
4. 推理时把上一块剩余动作 hard clamp 到 prefix 位置（非 inpainting、无反传、无额外去噪开销）。

论文声明的待验证结论（供实机比对）：
- 训练时 RTC 在**较大推理延迟**下优于推理时 inpainting；
- 与推理时 RTC 相比任务成功率与速度持平，但计算更便宜；
- 不改模型结构、不改机器人运行时，属于 drop-in 替换。

## 2. 代码改动清单（site-packages 补丁）

改动前原文件均备份为同目录 `.pre_rtc.bak`，回滚 = 复制回原文件名。
环境路径：`/root/miniconda3/lib/python3.12/site-packages/lerobot/`

| 文件 | 改动 |
|---|---|
| `policies/rtc/configuration_rtc.py` | `RTCConfig` 新增 `mode` 字段（`guided`/`trained`）及校验 |
| `policies/rtc/modeling_rtc.py` | `RTCProcessor.__init__` 新增 `trained_mode_supported` 关键字；未按 RTC 训练的 checkpoint 开 trained 模式直接报错 |
| `policies/common/vla_utils.py` | `create_sinusoidal_pos_embedding` 支持 `(B,)` 或 `(B, T)` 逐 token 时间 |
| `policies/pi_gemma.py` | `PiGemmaRMSNorm` 的 cond 支持 `(B, T, cond_dim)` 逐 token 调制（prefix 时间置 0 需要） |
| `policies/common/flow_matching.py` | `euler_integrate` 新增 `hard_prefix`/`hard_prefix_mask`，每步去噪前后各 clamp 一次 prefix（无反传） |
| `policies/pi05/configuration_pi05.py` | 新增 `rtc_training_max_delay` 字段；校验 `0 <= delay < chunk_size` |
| `policies/pi05/modeling_pi05.py` | 新增 4 个辅助函数；`forward` 与 `sample_actions` 接入 trained RTC |

`modeling_pi05.py` 的 4 个核心函数：

| 函数 | 作用 |
|---|---|
| `_sample_training_rtc_prefix_mask(B, T, max_delay, device)` | 逐样本采 `d ~ Uniform{0..max_delay}`，生成 `(B, T)` 连续前导 True 的 bool mask；`max_delay=0` 返回 `None`（关闭） |
| `_build_flow_matching_inputs(actions, noise, time, prefix_mask)` | prefix 位置 time→0、`x_t=actions`（干净）；postfix 按 `x_t = t·noise + (1-t)·actions` 正常加噪 |
| `_reduce_training_rtc_loss(losses, prefix_mask, reduction)` | postfix-only 平均；支持 `mean`（标量）与 `none`（逐样本，供 RA-BC 加权） |
| `_prepare_trained_rtc_prefix(x_t, prev_chunk_left_over, inference_delay, training_max_delay)` | 推理端 prefix 填充与全部合法性校验 |

关键设计：`rtc_training_max_delay: 0`（默认）时所有路径与原行为**逐位等价**（已用固定种子测试验证，
见 §4 C2），对非 RTC 训练零影响。

## 3. 配置项

`pi05_omy_fullft.yaml` 中的相关配置：

```yaml
policy:
  chunk_size: 50
  n_action_steps: 50
  rtc_training_max_delay: 6   # 0 = 关闭；fps=20 时 6 步 = 0~300ms 推理延迟覆盖
```

推理端启用方式（checkpoint 需带 `rtc_training_max_delay > 0` 训练）：

```python
rtc_config.enabled = True
rtc_config.mode = "trained"
# 调用 predict_action_chunk / select_action 时传：
#   inference_delay=<实测延迟步数>, prev_chunk_left_over=<上一块剩余动作>
```

约束（均在代码中强制校验）：
- `0 <= rtc_training_max_delay < chunk_size`
- 推理 `inference_delay <= rtc_training_max_delay`，否则 `_prepare_trained_rtc_prefix` 报错
- `prev_chunk_left_over` 含 NaN/Inf、batch/动作维度不匹配、长度小于 delay 均报错

## 4. 验证结果（2026-09-05，vGPU-32GB，torch 2.8.0+cu128）

测试脚本：`test_rtc_pi05.py`（`python test_rtc_pi05.py`，10 项，全部通过）

| # | 测试 | 结果 |
|---|---|---|
| A1 | delay 采样：4096 样本落 [0,6]、连续前导 mask、`max_delay=0` 返回 None | PASS |
| A2 | flow 输入：prefix time=0 且 x_t=干净动作；postfix 正常加噪；mask=None 与旧路径逐位一致 | PASS |
| A3 | postfix-only loss 与手算一致；逐样本 reduction 正确；非法 reduction 报错 | PASS |
| A4 | 推理 prefix 填充正确；7 类非法输入（超限/NaN/形状）全部按预期抛错 | PASS |
| B1 | 配置边界：-1、50、100 均被拒绝 | PASS |
| B2 | `mode` 只接受 guided/trained；未训练 checkpoint 开 trained 被 RTCProcessor 拦截 | PASS |
| C1 | 端到端前向+反向（真实数据 ela3_demo + pi05_base）：loss=0.1236 有限，805 个参数梯度有限 | PASS |
| C2 | **回归**：固定种子下 `delay=0` 与强制空 mask 的 loss 逐位相等 | PASS |
| C3 | `reduction="none"` 逐样本 loss 形状/数值正确 | PASS |
| C4 | 推理 trained 模式：输出前 3 步被 hard clamp 为 `prev_chunk_left_over`（逐位相等）；`inference_delay=7>6` 报错 | PASS |

100 步真实训练验证（`rtc_probe_train.py` 探针包装，日志 `train_rtc100.log`）：

- 探针确认 **100 次训练前向全部经过 RTC 路径**，1600 个样本 delay 分布近似均匀：
  `{0: 233, 1: 207, 2: 209, 3: 241, 4: 246, 5: 228, 6: 236}`
- loss 0.158 → 0.106 平稳下降，grad norm 0.8~1.2，无 NaN/Inf
- 权重加载正常（`Remapped 812 keys, All keys loaded`）

## 5. 硬件与显存（重要约束）

| 配置 | 可训参数 | 实测显存 | 结论 |
|---|---|---|---|
| 全量微调（VLM+视觉塔+专家，bs=16） | 4.14B | OOM：前向/反向后 AdamW 动量分配失败（31.18/31.47 GiB） | **32GB 卡不可行**，需 96GB 级（RTX 6000 Pro 实测可跑，固定占用 ~35GB + ~2GB/样本激活） |
| `train_expert_only=true`（仅动作专家，bs=16） | 693M | 15.0 GB 稳定，~3.1 s/step | 32GB 卡可用 |

注意：全量微调的 OOM 发生在**优化器状态分配**（约 25~33GB，与 batch size 无关），
降 bs 无法解决；可选方案：96GB 卡 / expert-only / PEFT / 8-bit 优化器（需装 bitsandbytes）。

## 6. 训练与验证命令

正式训练（96GB 机器）：

```bash
cd /root/autodl-tmp/debugger
OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n base --no-capture-output \
  python train_model.py --config_path pi05_omy_fullft.yaml 2>&1 | tee train.log
```

带 RTC 探针观察 delay 分布（把入口换成 `rtc_probe_train.py` 即可，不改 site-packages）：

```bash
python rtc_probe_train.py --config_path pi05_omy_fullft.yaml ...
# 日志中 [RTC-PROBE] 行打印每批 delay 分布，退出时打印累计分布
```

小步数验证（例：100 步、不存 checkpoint、仅训专家）：

```bash
python rtc_probe_train.py --config_path pi05_omy_fullft.yaml \
  --steps 100 --log_freq 10 --save_checkpoint false --policy.train_expert_only true
```

## 7. 后续调参指南

- **`rtc_training_max_delay` 选取**：`delay 步数 = 预期最大推理延迟(秒) × fps`。
  当前 fps=20、delay=6 覆盖 0~300ms。实机测得更大延迟（如 400ms）→ 调大到 8 并重新训练。
  代价：prefix 占比期望上升，有效监督信号（postfix 长度）略减；delay 应远小于 chunk_size(50)。
- **推理端 `inference_delay`**：必须按实机实测延迟（步数）设置，且 ≤ 训练值。
  论文指出训练时 RTC 在**高延迟**区间相对 inpainting 优势更明显——实机比对实验
  建议在 delay 接近上限（5~6 步）处对比 guided vs trained 两种模式的平滑性与成功率。
- **超参无需随 RTC 改动**：lr 2.5e-5 / warmup / cosine 预设沿用官方全量微调预设；
  RTC 只改变 loss 的掩码与 time 条件，不改变优化动力学。
- **checkpoint 兼容性**：`rtc_training_max_delay > 0` 训出的 checkpoint 仍可用于
  非 RTC 推理（`rtc_config=None`）和 guided 模式；trained 模式会被
  `RTCProcessor(trained_mode_supported=...)` 守护，防止误用于未训练的 checkpoint。
- **监控**：`loss_per_dim`（postfix-only）与 eval_loss（eval_split=0.1，每 500 步）；
  数据量小（136K 帧），按 eval_loss 走势可提前停。
- **回滚**：所有 site-packages 改动均有 `.pre_rtc.bak` 备份；配置置
  `rtc_training_max_delay: 0` 即逐位回到原行为，无需回滚代码。

## 8. 与论文的已对齐点 / 待实机验证点

已对齐（代码级）：
- 逐样本均匀随机延迟采样，prefix 干净动作 + time=0 条件，postfix-only loss；
- 推理端 hard clamp 替代 inpainting，零额外去噪开销；
- 无模型结构改动，drop-in。

待实机验证（需机器人实验）：
- trained vs guided 在不同实测延迟下的任务成功率/平滑性对比（论文：高延迟下 trained 更优）；
- 与论文 π0.6 实验（box building / espresso）对应任务的延迟-成功率曲线；
- 训练时 max_delay 与实测延迟错配（ underestimating / overestimating ）的敏感度。

---

## 9. 推理端集成：`widgets/camera_pyqt.py`（2026-09-06）

### 9.1 原程序是否需要修改？

**直接加载新 checkpoint 不需要修改即可运行。** `camera_pyqt.py` 通过
`get_policy_class(...).from_pretrained(ckpt)` 加载；checkpoint 中 `rtc_config`
默认为 `None`，`_rtc_enabled()` 为 False，推理走普通采样路径，行为与旧模型
完全一致。训练时 `rtc_training_max_delay=6` 只是让模型在训练中见过 prefix
条件（其中 1/7 样本 delay=0 即无条件情形），不强制推理端使用。

**但要获得论文 trained RTC 的收益则必须改。** 关键事实：程序里原有的
"RTC异步"**不是论文的 RTC**——它是应用层自实现的"异步推理 + 时滞补偿"：
新 chunk 到达时按 `d_stale = latency × fps` 丢弃前 d 步过期动作
（原 `camera_pyqt.py:2438`），对应论文的 discard 基线。它从不设置
`rtc_config`，也不传 `inference_delay` / `prev_chunk_left_over`。两种方案
解决同一个时滞问题，**不能混用**（否则时滞被补偿两遍）。

### 9.2 本次改动内容（已实现）

推理线程 `VLAInferenceThread`：

| 位置 | 改动 |
|---|---|
| `__init__`（约 :434） | 新增 `rtc_trained` 参数与状态：`_trained_rtc_active` / `_trained_max_delay` / `_last_raw_chunk` / `_last_used_delay` / `_pending_rtc` |
| 模型加载后（约 :653） | 自动检测：checkpoint 带 `rtc_training_max_delay>0` 且开关打开时，注入 `RTCConfig(enabled=True, mode="trained")` 并 `init_rtc_processor()`；不支持则提示并退回 d 步丢弃 |
| `_rtc_inference_loop`（约 :753） | 用主线程传入的 `(k, d)` 从上一段**归一化空间** chunk（postprocessor 之前保存）第 k 步起取剩余动作作 `prev_chunk_left_over`（不足 d 步重复末行补齐），连同 `inference_delay` 传给 `predict_action_chunk`；保存本次输出供下一段使用 |

主线程 `CameraWidget`：

| 位置 | 改动 |
|---|---|
| UI（约 :1166） | 新增"训练RTC"复选框（默认勾选，与"RTC异步"并列，运行期间锁定） |
| `_request_next_rtc_chunk`（约 :2652） | 请求下一段时推算已执行步索引 `k = 段偏移 + 已耗时×fps`、预测延迟 `d = round(实测延迟×fps)` 钳制到 `[1, max_delay]`，交给推理线程 |
| `_on_rtc_chunk`（约 :2510） | trained 模式跳过新 chunk 前 d 步（clamp 重放，机械臂已在执行），**不再做 d_stale 丢弃**；状态栏显示 `trained prefix=d` |
| 暂停恢复（约 :2004） | 清空上一段 chunk 条件（机械臂可能被手柄移动，旧 prefix 作废，恢复后首段退回普通推理） |

### 9.3 使用方式与注意事项

- 用法：加载带 RTC 训练的 checkpoint，勾"RTC异步"+"训练RTC"即可；
  模型加载状态栏显示 `trained RTC 已启用(max_delay=N)`。
- 老 checkpoint（smolvla、ACT、未开 RTC 训练的 pi05）行为完全不变；
  "推理一次"与普通顺序块模式不受影响（无前缀条件，delay=0）。
- 对比实验：同一 checkpoint 下勾选/取消"训练RTC"即可在 trained 与
  discard 基线间切换——对应 §8 待验证的实机对比设置。
- d 用上一次实测延迟预测；若实机推理延迟经常超过训练上限
  （6 步 = 300ms@20fps），d 被钳到 6，应按 §7 调大
  `rtc_training_max_delay` 重新训练。
- prefix 必须在**归一化动作空间**（postprocessor 之前的原始输出），
  代码中 `_last_raw_chunk` 即此约定，改动相关路径时勿破坏。
- 已通过 `py_compile` 与接线一致性检查；段调度时序（k/d 推算）需真机
  验证，建议先在低速度（步长mm 调小）下试跑，观察状态栏
  `trained prefix=d` 是否符合实测延迟。

### 9.3 在其他电脑上部署修改后的lerobot0.6.1（加入了pi0.5的RTC）
  推荐做法：装原版 + 打补丁包                                                                              
   我已在项目目录打好补丁包 rtc_patch_lerobot_0.6.1.tar.gz（29KB，含 7 个补丁文件、SHA256 清单、应用脚本，本机
   自测通过）。目标机流程：                                                                                   
                                                                                                              
   ```bash                                                                                                    
     # 1. 按 服务器环境部署.md 装环境，确保 lerobot 版本一致                                                  
     pip install lerobot==0.6.1                                                                               
                                                                                                              
     # 2. 解包并应用（脚本会自动备份 .pre_rtc.bak、校验哈希、检查版本）                                       
     tar xzf rtc_patch_lerobot_0.6.1.tar.gz                                                                   
     bash rtc_patch_lerobot_0.6.1/apply_rtc_patch.sh                                                          
     # 输出"补丁校验通过"即完成；回滚 = 把各 .pre_rtc.bak 复制回原文件名                                      
   ```                                                                                                        
                                                                                                    
   除了代码补丁，目标机还需要这些                                                                             
                                                                                                    
   • pretrained_models/pi05_base/（模型权重，约 8GB）                                                         
   • pi05_omy_fullft.yaml、train_model.py（+ rtc_probe_train.py 探针、test_rtc_pi05.py 测试脚本）             
   • 训练数据 lerobot_data/ela3_demo/（若要训练）                                                             
   • paligemma tokenizer 离线缓存（手册 §2.2，用 HF_HUB_OFFLINE=1 启动时必须；缓存通常在                      
     ~/.cache/huggingface，可整目录复制或按手册重建）                                                         
   • 推理用：widgets/camera_pyqt.py（本次已加入 trained RTC 支持）                                            

### 9.4 已验证的部署故障：未打补丁导致 checkpoint 加载失败（2026-09-10）

**现象**：在本地电脑（`~/anaconda3/envs/VLA`，lerobot 0.6.1 原版）运行
`camera_pyqt.py` 加载 `ckpt/pi05_ela3_fullft/checkpoints/008000/pretrained_model`，
模型选择后约 16 秒报错：

```
draccus.utils.DecodingError: The fields `rtc_training_max_delay` are not valid for PI05Config
```

**原因**：该 checkpoint 的 `config.json` 带 `rtc_training_max_delay: 6`（训练端新字段），
未打补丁的原版 `PI05Config` 没有此字段，draccus 在 `from_pretrained` 的配置解析阶段
直接拒绝。**与是否启用 trained RTC 推理无关**——只要加载带该字段的 checkpoint 就必现，
属于纯配置解析失败，不是权重问题。

**解决**：必须先按 §9.3 打补丁再加载。本地实际操作要点：

```bash
cd /home/chen/yolo/lingzu电机/EDULITE_A3/el_a3_sdk_RTC_VR/debugger
tar xzf rtc_patch_lerobot_0.6.1.tar.gz
conda activate VLA          # 必须！脚本用当前 python 检测/校验 site-packages
bash rtc_patch_lerobot_0.6.1/apply_rtc_patch.sh
# 或显式传路径：
# bash rtc_patch_lerobot_0.6.1/apply_rtc_patch.sh \
#   ~/anaconda3/envs/VLA/lib/python3.12/site-packages
```

**教训 / 检查清单**：往新机器部署时，"下载权重 + 界面程序"之外，**补丁包是必需品**
（除非只用未开 RTC 训练的旧 checkpoint）。部署后可用一行命令快速自检：

```bash
python -c "from lerobot.policies.pi05.configuration_pi05 import PI05Config; \
print('rtc_training_max_delay' in PI05Config.__dataclass_fields__)"
# 输出 True 即补丁生效
```

### 9.5 本机部署排障实录（2026-09-10 深夜，本机 VLA 环境）

**故障 A：模型加载后卡在"VLA正在加载"假死（6~10 分钟级）**

- 现象：008000/012000 都能打印 `All keys loaded successfully!`，之后界面停在
  "VLA正在加载"，机械臂不动，无任何报错。
- 原因：**与 RTC 补丁无关**。新旧两版 `camera_pyqt.py` 对比发现新版文件头丢了
  `os.environ.setdefault("HF_HUB_OFFLINE", "1")`（复件旧版有）。缺它时，加载
  `policy_preprocessor.json` pipeline 内的 paligemma tokenizer 会触发 transformers
  缓存新鲜度检查（HEAD huggingface.co），本机访问不通，每次 TCP 超时 ~75s × 5 次
  重试，白卡 6~10 分钟才回退本地缓存——恰好处在权重加载完成之后，与日志位置吻合。
- 补丁无辜的佐证：复件旧版在**同一套打过补丁的 site-packages** 上运行正常；且对
  补丁前后 7 个文件逐一 diff，非 RTC 推理路径（forward/sample_actions/euler_integrate/
  AdaRMS/sinusoidal embedding）在 `rtc_config=None` 时与原实现逐位等价。
- 次生故障：卡死进程不退出会残留显存（实测 `python main.py` 残留 9.4GB），导致
  下一次加载 OOM（`Process ... has 9.22 GiB memory in use`）。排障先查
  `nvidia-smi --query-compute-apps` 清残留。
- 修复：新版 `camera_pyqt.py` 头部补回 `HF_HUB_OFFLINE=1`（必须在 lerobot/transformers
  首次导入前）。修复后验证：012000 完整加载 15.4s + 首次推理 0.59s，正常出块。

**故障 B：008000 + 勾"训练RTC"仍一卡一卡**

- 现象：日志全是 `顺序块 #N: steps=10, window=500ms, latency=~390ms`，段间明显停顿。
- 原因：**只勾了"训练RTC"，没勾"RTC异步"**。trained RTC 生效需两个开关同开：
  `_rtc_trained_active() = self._rtc_mode and 线程侧 _trained_rtc_active`
  （camera_pyqt.py:2653）。顺序块模式下 `_pending_rtc` 从不写入（只有 RTC 管道调度
  `_request_next_rtc_chunk` 才写），推理时 `rtc_kwargs={}`，**prefix 条件化根本没
  启用**；而顺序块节奏本身就是"执行 500ms → 等推理 ~390ms 原地保持"，必然卡顿。
- 原理性不兼容：顺序块模式观测时滞 ≈ 完整推理延迟 390ms ≈ 8 步 >
  `rtc_training_max_delay=6`，即使传 prefix 也会被 `_prepare_trained_rtc_prefix`
  校验拒绝。trained RTC 的设计前提就是管道模式：上一段未执行完就提前推理，推理期间
  机械臂正在执行上一段剩余动作（即 prefix）。
- 正确用法：停止后**同时勾"RTC异步"+"训练RTC"**，加载 008000；状态栏应显示
  `trained RTC 已启用(max_delay=6)`，日志变为 `[VLA-RTC] 段 #N ...`。
  对照实验组：012000 + 只勾"RTC异步"（应用层 d 步丢弃 discard 基线）。

### 9.6 补丁包修复 v2（2026-09-11）

排障中发现 v1 补丁包（已留存为 `rtc_patch_lerobot_0.6.1.tar.gz.pre_recap_fix.bak`）
的三个问题，v2 已全部修复：

| # | 问题 | 影响 | v2 修复 |
|---|---|---|---|
| 1 | `modeling_pi05.py` 取自训练服务器，**丢了本机 [RECAP P1.2 patch]**（`forward(reduction="none")` 的 `recap_dim_scale` 逐维缩放，APO 负样本夹爪维豁免） | 本地 APO/Recap 训练会静默丢失夹爪维豁免；纯推理无影响 | 已合并：先逐维缩放再走 `_reduce_training_rtc_loss` postfix 归约；无 recap 键且 `prefix_mask=None` 时与原实现逐位一致 |
| 2 | tar 包**无顶层目录**，解开直接把 `files/`、`apply_rtc_patch.sh`、`SHA256SUMS.txt` 散落当前目录 | 与 §9.3 命令 `bash rtc_patch_lerobot_0.6.1/apply_rtc_patch.sh` 不符，照文档操作必失败 | 已加 `rtc_patch_lerobot_0.6.1/` 顶层目录重打包 |
| 3 | `apply_rtc_patch.sh` 校验路径 bug：显式传 site-packages 参数时，SHA256 校验仍用当前 python 的 purelib（sysconfig） | 两者不一致时误检/漏检 | 校验改用传入的 `$SP`（环境变量 `SP_DIR` 传入） |

- `SHA256SUMS.txt` 已重算：`modeling_pi05.py` 哈希 `887c62…` → `d29a3d…`，
  其余 6 个文件不变。
- 本机 site-packages 已同步 v2 版并复验：全新解包 → apply → `补丁校验通过`；
  012000 加载+推理 17.6s 正常；`forward` 源码中 RECAP 与 RTC prefix 采样共存确认。
- **后续部署到下一台机器务必用 v2 包。** 若某台机器已用 v1 打过补丁，用 v2 重新
  apply 一次即可（`.pre_rtc.bak` 已存在不会被覆盖，仍指向原版备份，回滚不受影响）。
  
  
## 10. 云端推理（方案 A：全量云端推理，2026-09-06）

模型部署在云服务器，本地机器人 PC 只采集观测（2×720×720 JPEG + 位姿 7 维 +
夹爪力矩 + 任务文本），经 gRPC 上行，云端返回完整动作 chunk，本地执行层
（限幅/夹爪斜坡/投影剥离/RTC 调度）完全不变。

### 10.1 文件清单

| 文件 | 位置 | 说明 |
|---|---|---|
| `cloud_inference/vla_inference.proto` | 双端 | gRPC 协议定义（Ping 握手 + InferChunk 推理） |
| `cloud_inference/vla_inference_pb2{,_grpc}.py` | 双端 | protoc 生成代码 |
| `cloud_inference/server.py` | 云端 | 常驻推理服务：加载 ckpt、预热、trained RTC 会话状态、token 校验 |
| `widgets/camera_pyqt.py` | 本地 | 新增 `CloudInferenceThread`（与 `VLAInferenceThread` 接口完全对齐）+ 界面"云端推理"开关 |
| `cloud_inference/smoke_test.py` | 云端 | 服务端冒烟测试（7 项） |
| `cloud_inference/test_client_thread.py` | 云端/本地 | 客户端线程端到端测试（11 项，含故障注入） |

### 10.2 架构与安全机制

- **通信**：每次 chunk 一次 unary RPC（与 request_chunk→chunk_ready 模式 1:1）；
  上行 ~240KB/请求（2×JPEG q90），2 请求/s ≈ 4Mbps；下行 chunk 50×7 float32 ≈1.4KB。
- **trained RTC**：`_last_raw_chunk`（归一化空间）保存在云端会话内，客户端只发
  (rtc_k, rtc_d)；reset 标志清空云端 prefix（首段/暂停恢复/重连后自动带）。
- **看门狗**：单次推理 RPC 超时（1.5s×avg_n）→ `link_lost` → 主线程安全停：
  cancel_motion + 停夹爪斜坡 + 自动"暂停控制"（保位），恢复需人工点"继续控制"。
- **心跳**：1s 一次 Ping，连续 2 次失败提示链路劣化；gRPC 通道指数退避自动重连，
  重连成功发 `link_restored`（仅提示，不自动恢复运动）。
- **过期响应丢弃**：request_id 单调递增校验；响应含 NaN/Inf 直接丢弃。
- **token 校验**：UNAUTHENTICATED 走 error 信号（配置错误），不误判为断链。
- **本地最后防线不变**：`_clamp_pose_jump` 限幅限速、夹爪力矩保护、投影剥离。

### 10.3 云端启动

> 注意：AutoDL 容器可能没有 tmux（报 `command not found`），先
> `apt install -y tmux`，否则 server 跑在 SSH 前台、断连即死（见 §10.7 坑 3）。

```bash
cd /root/autodl-tmp/debugger
tmux new -s vla_server
python cloud_inference/server.py \
    --host 127.0.0.1 --port 50051 \
    --model_path ckpt/pi05_ela3_fullft/checkpoints/last \
    --token ela3-vla-2026
# 日志出现"预热完成，服务就绪"后即可接受连接；GPU 占用 ~9.7GB
```

### 10.4 本地（机器人 PC）部署

> **2026-09-11 起推荐界面自动化流程（下方"方式一"）**：界面新增 SSH 地址/密码
> 输入框，"开始推理"时自动完成拉取代码+建隧道+保存配置；原手动命令保留为
> 方式二（自动化失效时的兜底）。

**方式一：界面自动化（推荐，多机器人部署）**

```bash
# 每台机器人仅一次性准备：
# 1) 拷贝整个 debugger/ 文件夹（已含 cloud_inference/；没有也不影响，见下）
# 2) 装依赖（无需 torch/lerobot）
pip install grpcio protobuf
sudo apt install -y sshpass autossh rsync   # 自动隧道/自动拉取所需
```

界面操作：勾"云端推理" → 填 **SSH 地址**（`root@主机:端口`）和 **SSH 密码**、
勾"**记住**" → 点"开始推理"。程序自动完成：

1. `cloud_inference/vla_inference_pb2.py` 缺失 → 按 SSH 信息自动 rsync 拉取
   （云端目录取常量 `CLOUD_REMOTE_DIR=/root/autodl-tmp/debugger/cloud_inference`）；
2. gRPC 地址为回环（127.0.0.1:50051）且端口无监听 → 自动用
   sshpass+autossh（无 autossh 退化为 ssh）建立隧道，首次连接自动接受主机密钥；
   已有隧道（外部建立或上次拉起）则直接复用；gRPC 填公网 IP:端口 时跳过隧道；
3. 勾"记住"后 SSH 地址/密码/gRPC 地址/令牌保存到 `debugger/cloud_settings.json`
   （明文、权限 600），下次启动自动填入——**每台机器人只需填一次**；
4. 更换服务器：停止推理 → 改 SSH 地址 → 开始推理（旧托管隧道自动清理重建）。

**方式二：手动命令（兜底）**

```bash
# 1) 同步代码（在本地 debugger/ 目录执行）
S="ssh -p 43394"; H="root@connect.westd.seetacloud.com"; D="/root/autodl-tmp/debugger"
rsync -avz -e "$S" $D/widgets/camera_pyqt.py widgets/
rsync -avz -e "$S" $D/cloud_inference/ cloud_inference/ \
    --exclude '__pycache__' --exclude '*.log'

# 2) 本地依赖（无需 torch/lerobot）
pip install grpcio protobuf

# 3) SSH 隧道（建议先配 SSH 密钥免密）：把云端 50051 映射到本地
autossh -M 0 -N -L 50051:localhost:50051 -p 43394 root@connect.westd.seetacloud.com
# 无 autossh 可用 ssh 代替（断线不自动重连）：
# ssh -N -L 50051:localhost:50051 -p 43394 root@connect.westd.seetacloud.com

# 4) 界面：勾选"云端推理"，地址保持 127.0.0.1:50051，令牌与服务端一致，
#    点"开始推理"即可（本地/云端双路径可随时切换，模型路径框仅本地模式生效）
```

> 首次在一台新机器上执行本节流程的常见坑（rsync 失败、autossh 缺失、
> 误导性报错），以及一行 Ping 自检命令，见 §10.7。

### 10.5 实测数据（云端 127.0.0.1 直连，4080 32GB）

| 项目 | 结果 |
|---|---|
| 冒烟测试 smoke_test.py | 7/7 通过（握手/错误token/reset/trained RTC k=5,d=3/超限钳制/avg_n=2/连续推理） |
| 端到端 test_client_thread.py | 11/11 通过（含杀服务→link_lost、重启→link_restored、重连恢复推理） |
| 服务端推理耗时 | **~480~550ms**（pi05 chunk=50，含 pre/post processor） |
| JPEG 解码 | ~35ms（2×720×720） |
| 客户端端到端（本机回环） | ~546ms |

### 10.6 ⚠️ 关键风险：推理延迟超过 trained RTC 训练上限

当前 checkpoint `rtc_training_max_delay=6`（300ms@20fps），而云端纯推理已达
~500ms，加网络 RTT 后总延迟 550~650ms，d 会被长期钳到 6（欠补偿约一半）。
对策（按优先级）：

1. **先用 discard 基线跑**：取消勾选"训练RTC"（保留"RTC异步"），应用层
   d 步丢弃补偿无上限，可吸收任意延迟——云端的 discard 正好对应论文基线；
2. **重训更大 max_delay**：按 §7，`max_delay = 预期延迟×fps` = 0.65s×20 ≈ 13，
   建议 `rtc_training_max_delay: 12`（仍远小于 chunk_size=50）重训后恢复 trained RTC；
3. 可选优化：云端推理加速（bf16/torch.compile/减少去噪步数）若能把推理压到
   <250ms，现有 checkpoint 即可直接用 trained RTC。

另注意：延迟(~550ms) > 段窗口(0.5s) 时 RTC 管道退化为段间短停顿（每段
~50~100ms 空隙）；可把"目标步N"调到 14~20 提速吸收，或接受轻微停顿。

### 10.7 首次本机联调实录（2026-09-11，本机 VLA 环境）

云端 server 按 §10.3 启动（ckpt=`008000`，trained RTC 已启用 max_delay=6，
预热稳态推理 ~408ms，GPU ~9.7GB）。本机侧按 §10.4 联调踩到四个坑，按出现顺序：

**坑 1：rsync 拉取失败 → `cloud_inference/` 空目录（最坑）**

首次连接云主机有两道坎：① 主机密钥未确认（`Host key verification failed`，
非交互模式需 `-o StrictHostKeyChecking=accept-new`，或先手动 ssh 一次写入
known_hosts）；② 未配 SSH 密钥时非交互 rsync 无法输密码
（`Permission denied (publickey,password)`）。结果 §10.4 第 1 步实际没执行，
`cloud_inference/` 是**空目录**，Python 把它当命名空间包，界面报
`cannot import name 'vla_inference_pb2' from 'cloud_inference' (unknown location)`
并提示"请 pip install grpcio protobuf"——但依赖早已装好，**该提示具有误导性**。

- 检查清单：`ls cloud_inference/` 必须有 7 个文件（`__init__.py / server.py /
  vla_inference.proto / vla_inference_pb2.py / vla_inference_pb2_grpc.py /
  smoke_test.py / test_client_thread.py`），非空才算同步成功；
- 未配密钥时用 sshpass 完成拉取：

```bash
sshpass -p '<密码>' rsync -avz -e "ssh -p 43394" \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/debugger/cloud_inference/ \
  cloud_inference/ --exclude '__pycache__' --exclude '*.log'
```

**坑 2：autossh 未安装**

`sudo apt install -y autossh`（本机已装 1.4g）。装好后把 plain ssh 隧道换成
autossh 托管（断线自动重连，进程 detached 后台常驻）：

```bash
SSHPASS='<密码>' setsid sshpass -e autossh -M 0 -N \
  -L 50051:localhost:50051 -p 43394 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  root@connect.westd.seetacloud.com </dev/null >>/tmp/autossh_vla.log 2>&1 &
```

注意：手工启动的隧道**重启电脑后需重跑**；长期建议 `ssh-copy-id` 配免密后
去掉 sshpass（密码不再出现在进程环境），并可做成 systemd 用户服务开机自启。

**坑 3：云端 tmux 缺失，server 跑在 SSH 前台**

§10.3 的 `tmux new -s vla_server` 报 `command not found` 后 server 直接跑在
SSH 会话里——**SSH 一断服务就死**。云端 `apt install -y tmux` 后按 §10.3 重开，
或改用 `nohup python cloud_inference/server.py ... &`。
（云端另有 `libgomp: Invalid value for OMP_NUM_THREADS` 警告，无害，
系环境变量值非法，顺手修正即可。）

**坑 4（验证方法）：一行 Ping 做端到端自检**

隧道建立后、开界面前先自检（token 错误时返回的 error 字段非空）：

```bash
python -c "
import grpc
from cloud_inference import vla_inference_pb2 as pb2, vla_inference_pb2_grpc as pb2_grpc
rep = pb2_grpc.VLAInferenceStub(grpc.insecure_channel('127.0.0.1:50051')).Ping(
    pb2.PingRequest(token='ela3-vla-2026'), timeout=10)
print(rep.policy_type, rep.device, rep.trained_max_delay, repr(rep.error))"
# 期望输出: pi05 cuda 6 ''
```

实测（2026-09-11）：Ping 通过，服务端回报 `policy_type=pi05, device=cuda,
ckpt=.../008000/pretrained_model, trained_max_delay=6, image=720x720`，
与云端启动日志一致；界面勾选"云端推理"即可使用。

                                                                                                        

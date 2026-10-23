# π\*0.6 RECAP 在线 RL 迁移实施方案（ela3 真机项目）

> 日期：2026-08-30　｜　前置文档：`RECAP_MEM_实施路线.md`（P0~P2 已完成、P1.2 已交付）
> **进度（2026-08-30 深夜）**：R1a ✅ + **R1b 组优势管线 ✅ + R1c 条件化训练（P4）✅**
> ——训练/部署双侧代码全部就位（见 §4.4/§4.5 实现记录与 §6 验收）；
> 下一步 R2 飞轮：真实成组采集 → 组优势 → 加权+条件化训练 → 干预率对比。
> 目标：在现有 pi0.5 / SmolVLA + 部署录制 + RECAP 标注工具链上，迁移 π\*0.6
> （arXiv:2511.14759）的 RECAP 核心机制，进入**在线 RL 迭代**阶段。
> 本文回答四件事：**现状、必要性、具体实现（采集/编辑/训练/推理四端）、分阶段计划**。

---

## 1. π\*0.6 论文核心机制拆解（只取我们可迁移的部分）

π\*0.6 = π0.5 基座 + **RECAP**（REinforcement learning with Contrastive Advantage
Preconditioning）。其配方可拆为四个环节：

| 环节 | 论文做法 | 一句话本质 |
|---|---|---|
| ① 在线数据采集 | 当前策略大规模 rollout（仿真+真机），**人工干预即时纠正** | 数据由"现在的模型"产生，错误即奖励信号 |
| ② 优势估计 | 同一任务实例多次 rollout **互为基线**（组相对，GRPO 式，无需独立 critic） | 消掉任务难度，成败差异归因于策略行为 |
| ③ 优势条件化训练 | 策略以优势为条件训练（contrastive advantage preconditioning）：正优势数据吸引、负优势数据对照，**部署时恒设最高优势** | "训练时告诉模型动作好坏，推理时命令它拿最好的" |
| ④ 迭代飞轮 | 采集→优势标注→重训→部署 循环，干预率逐轮下降 | 终身学习节奏（APO 同源：每 ~20 条 rollout 更新一次） |

**关键设计约束（论文明确）**：flow-matching 策略无动作对数似然 → 放弃
PPO/DPO/KTO 类目标，改走优势条件化——这与本项目路线图附录 A 的决策一致。

---

## 2. 现状盘点与差距分析

### 2.1 已就绪（可直接复用）

| 能力 | 现状 | 位置 |
|---|---|---|
| 部署 rollout 录制（干预段/bad 窗口/成败/部署日志） | ✅ | `camera_pyqt.py` 推理录制 |
| 片段级标注 + sidecar 展开 + 8 项检查 | ✅ | `visualize_data` |
| filtered BC 样本级加权（P1.2，含 APO 夹爪豁免） | ✅ 2026-08-30 交付 | `RecapWeighter` + 两个 recap yaml |
| 数据闭环防遗忘（剪辑/合并/删除标注重映射） | ✅ | `vd_data_ops.py` |
| 北极星指标雏形（deployment_log.csv 干预率） | ✅ | 每集一行 |

### 2.2 与 π\*0.6 的差距（按必要性排序，2026-08-30 晚更新）

| # | 缺失 | π\*0.6 对应环节 | 必要性 | 状态 |
|---|---|---|---|---|
| G1 | **策略版本标记**：不知道每集是哪个 checkpoint 跑的 | ① 在线采集（on-policy 前提） | **硬前提**：多轮迭代中混入旧策略数据=自我污染；干预率无法按轮归因 | ✅ **已实现**（采集端自动标记 + 表格/总信息显示 + 检查） |
| G2 | **任务组标记**：无"同任务实例 N 次"的分组 | ② 组相对优势 | **硬前提**：没有组就没有基线，优势只能靠人标/启发式 | ✅ **已实现**（可编辑下拉框 + 单集/批量改名 + 检查） |
| G3 | 组优势计算 + 写回 sidecar | ② | 训练前预处理脚本（毫秒级） | ✅ **已实现**（`debugger/recap_group_advantage.py` + weighter 调制 + sidecar 透传） |
| G4 | **优势条件化 prompt**（P4） | ③ preconditioning | π\*0.6 的核心创新 | ✅ **已实现**（factory.py 数据集包装注入 task 前缀，默认关） |
| G5 | 部署端恒 positive 条件 | ③ 部署形态 | camera_pyqt 一处拼接 | ✅ **已实现**（"advantage条件"复选框 → 线程前缀，实时切换） |
| G6 | 真策略梯度（policy gradient on flow） | 论文的 RL 更新 | **可选终局**（见 §5 R3） | ❌ 可选 |

### 2.3 已验证可行的"准 RL"路径

本项目 flow-matching 限制下，π\*0.6 配方的可行迁移路径是
**"组相对优势 + 条件化加权 SFT"的迭代飞轮**（论文 ①②③ 环节的完整复刻，
仅 ③ 的损失形式用加权条件监督而非其原始 contrastive 损失，见 §8 诚实声明）。
**该路径不需要策略梯度**，是工程上最现实的 π\*0.6 复刻方式。

---

## 3. 必要性论证（为什么 G1/G2 是下一步的全部采集侧工作）

1. **没有 G1，迭代即污染**：第 2 轮训练应"重点用 m2 的 rollout + 恒定混入专家
   数据"。现在所有 rollout 只标 `source=rollout`，三轮之后 m1/m3 的数据与 m2
   不可区分——旧策略的错误被当成当前策略的错误去纠正，训练信号互相抵消。
2. **没有 G1，北极星失效**：干预率必须按 policy 版本分组统计才能回答
   "m2 比 m1 好了吗"；混合统计无法归因到任何一轮改动。
3. **没有 G2，优势只能靠人**：现有 advantage 来自人工片段标注 + 回传/终端
   启发式。组相对优势把"成败归因"自动化：同组 20 轮里 10 成 5 败 5 干预，
   基线自动可算，**且天然免疫场景难度偏差**（难组成功加分多、失败宽容）。
4. **G1/G2 都是纯元数据**：不进 parquet、不进模型输入（sidecar 外挂原则），
   采集侧各 ~20 行改动，风险趋近于零。

---

## 4. 具体实现设计（四端）

### 4.1 数据 schema 扩展（标注 v2.1）✅ 已实现（2026-08-30）

```jsonc
// reward_annotations.json 的 episode 级新增字段（与 source 同级，剪辑/合并自动保留）
{
  "policy_checkpoint": "ckpt/pi05_xxx/checkpoints/030000",  // 完整路径（血缘）
                                                              // teleop 集显式 null（schema 统一）
  "group_id": "做饭-开火 | pi05_xxx"     // 组名 | 策略run名（策略段=checkpoints 上一级目录）
                                          // 未分组为 null；同名+同策略跨会话自动归并（有意延续）
}
```

实际实现与原设计的差异：**去掉了时间戳后缀与 group_size 冗余字段**——组会话
语义简化为"同名+同策略即同组"（下拉框选历史组名=显式延续，跨会话重选视为
有意合并；换策略同组名自动成新组，天然防 m1/m3 混组）。

```jsonc
// recap_sidecar.jsonl 逐帧新增列（expand_dataset 透传）——R1b 时实现
{"episode_index": 42, "frame_index": 0, ..., "policy_checkpoint": "...", "group_id": "...",
 "group_advantage": 0.3125}   // 预处理脚本写入（见 4.4），无组/未算时缺省 0
```

**三条文本红线**：`task`（任务指令，唯一进模型）｜`group_id`（分组键+人看）｜
`policy_checkpoint`（血缘）——后两者永不进模型输入。

### 4.2 采集端（`widgets/camera_pyqt.py`）✅ 已实现（2026-08-30）

1. **策略版本自动标记**（G1）：
   - 开录即捕获 `_rollout_policy_tag`：优先推理线程实际加载的
     `VLAInferenceThread.model_path`，回退界面模型输入框（血缘与推理严格一致）；
   - `_finalize_rollout_stop` 写入标注 episode 字段 + `deployment_log.csv`
     新列 `policy_checkpoint`/`group_id`（**旧日志表头自动迁移**：缺新列时
     整表重写、旧行补空，杜绝 CSV 列错位）；
   - 遥操作录制**显式写 null**（`_write_recap_annotations` 无条件写两字段）——
     两类录制 schema 完全统一，数据集融合零差异；
   - 顺带修复：停止录制 flush 丢帧的越界干预区间按实际保存帧数**钳位**
     （此前 ela3_demo5 出现的 3 个越界区间根因，见 RECAP_MEM P0.4）。
2. **任务组可编辑下拉框**（G2）：
   - 推理录制按钮旁 `rollout_group_input`（editable QComboBox）：下拉选历史
     组名 = 继续该组，手动输入新名称 = 新建组，留空 = 不分组（无"新组"按钮
     ——同名即延续，语义即分组）；
   - `group_id = f"{组名} | {策略run名}"`（run 名 = checkpoints 上一级目录，
     如 `ckpt/pi05_ela3_fullft20260830/checkpoints/012000` →
     `pi05_ela3_fullft20260830`）：同组名+同策略的所有录制自动归并一组
     （跨会话选同名组是有意延续；换策略同组名自动成新组，防 m1/m3 混组）；
   - 历史组名从 ela3_rollout 标注自动加载去重（启动 + 每次开录刷新），
     新组名录入后自动入候选；
   - **命名规范**：自然语言短描述（`搬运-左上到右下`、`做饭-开火`）；
     禁止写进任务指令框（两框物理分离防误用）。
3. **组采集操作纪律**（操作规程，程序不强制）：
   - 每组前重摆相同场景；从"目标步骤前一刻"起录（长任务拆子步骤组）；
   - 每组 N=4~10 轮；同组只换策略随机性，不换场景。

### 4.3 编辑端（`visualize_data`）✅ 已实现（2026-08-30）

- **Episode 表格扩为 7 列**（编号/帧数/时长/分辨率/任务/**策略**/**组**）：
  策略显示 run 名、组显示标签段；列宽 Interactive（可拖动），列总宽超出
  可视区自动出**横向滚动条**，长文本列不再挤压；
- **总信息面板**新增策略/组统计（`_policy_group_stats_html`）：
  `策略版本: 专家/未标 ×18 | pi05_xxx ×8`、`任务组: 做饭-开火 ×4 | ...`；
- **组名编辑**：任务框下方"组名"框（单集改，回车确认弹窗防误改；**策略段
  自动保留**，置空=取消分组）+ **批量修改组名**对话框（原组名下拉带影响
  集数实时预览 → 新组名，同组全部替换、各自策略段保留）；
- **数据集检查问题8 新增元数据一致性检测**（实测输出）：
  - 同组混策略：`✗ 组 "测试Z" 混入 2 个不同策略（…）——组相对优势要求
    同组同策略，非数据损坏；改组名拆分即可修复`；
  - 策略段矛盾：`✗ ep7 group_id 策略段(t_wrong) 与 policy 字段(…) 不符`；
  - 统计行：`· 任务组 5 个（10 集）；策略版本 5 种`；
- **安全性结论（已验证）**：批量改组名只写标注元数据（自动备份），parquet/
  视频/info 零接触；**合并数据集 deepcopy 整个 episode 字典**，policy/group
  与 `source` 同机制自动保留（`merge_recap_annotations`），deployment_log
  同步按 episode 重映射拼接——剪辑/合并/删除全路径零破坏。

### 4.4 训练端（分四步，全在数据/预处理/配置层）

**Step-1 组优势预处理脚本**（新 `debugger/recap_group_advantage.py`，~120 行）：

```python
# 伪代码（真实实现含边界处理）
R_i = success(1/0) − 0.25 × takeover_count        # 每集回报，截断 [-1,1]
for group in 按 group_id 分桶 (n>=4):
    A_i = (R_i − mean(R_group)) / (std(R_group)+ε)  # GRPO 式标准化
    写回 sidecar 列 group_advantage
未分组/小组(n<4) → group_advantage 缺省 0（不参与组调制）
```

实例（2026-08-30 会话已推演）：开火组 20 轮（10 成/5 败/5 干预后成）→
R̄=0.6875，干净成功 A=+0.31、失败 A=−0.69、干预后成 A=+0.06；切菜组自成基线；
50 轮未分组走现有 P1.2 规则。

**Step-2 RecapWeighter 消费 group_advantage**（扩展，~15 行）：

| 帧类型 | 权重 |
|---|---|
| 干预段 | ×2.0（恒定，**不受 A 影响**） |
| 接管前 bad 窗口 | ×0.1（恒定，不受 A 影响） |
| 策略自主帧 | `w = clamp(1 + β·A, 0.3, 1.5)`（β 默认 1.0，仅当该帧所属集有 group_advantage） |
| 无组数据自主帧 | 1.0（现有行为） |

**Step-3 优势条件化 prompt（P4，π\*0.6 核心机制的落地）**：
- 训练时：sidecar 的 advantage（人标 indicator 与 group_advantage 取并集符号）
  注入 prompt——pi05 在 preprocessor 文本组装处 patch：
  `Task: xxx, Advantage: positive, State: ...`（negative/neutral 同理）；
  SmolVLA 直接前缀 `"advantage: positive, " + task`；
- 消融开关 yaml 独立（`recap.advantage_conditioning: true/false`），与加权
  正交，支持 2×2 对照；
- patch 点登记 TRAIN_README §4。

**Step-4 迭代飞轮（R2，操作流程）**：

```
每轮（建议 1~2 周；APO 节奏：每 ~20 条 rollout 更新一次）：
  1. 部署 checkpoint mN（推理录制开启，组标记就绪）
  2. 采集：常规多样性录（不分组）+ 失败位形/子步骤定向成组（4~10 轮/组）
  3. visualize_data 标注复核 → 生成 sidecar → recap_group_advantage.py
  4. 数据集检查 8 项全绿 + action≠state 抽查
  5. 重训：专家数据恒定混入（ela3_demo 全量）+ mN rollout 按 A 加权
     + 优势条件化（P4 全开）
  6. 留出集离线评估（test_inference_episode.py）→ 真机 10 轮
  7. 记录：按 policy 版本统计成功率/干预率（deployment_log 新列直查）
```

### 4.5 推理端（`camera_pyqt.py`，两处）

1. **优势条件恒 positive**（G5，~5 行）：`VLAInferenceThread` 组装 task 文本时
   追加条件前缀（与训练侧 P4 格式严格一致，`advantage: positive`）——
   开关 `chk_advantage_condition`（默认关，模型未条件化训练时不得开）；
2. **语言输入隔离校验**：推理任务框输入仅作 task；组名/策略版本只在录制
   元数据通道，UI 上二者分离显示。

### 4.6 R3 数据准备链（P3 critic → 优势合并）

R3 的密集优势依赖 P3 视觉 critic（`recap_critic.py`，见 RECAP_MEM P3 路线 A）。
`recap_critic.py --backend siglip train --apply` 一次跑完四步：

| 步骤 | 日志特征 | 作用 |
|---|---|---|
| ① 特征提取 | `epN 特征完成（X 帧）` | 逐集解码双相机视频 → SigLIP 视觉塔嵌入(均值) ⊕ state，缓存 `meta/critic_feats_siglip.npz` |
| ② 训价值头 | `留出集: N 帧 AUC=... loss=...` | 在特征上训 `LayerNorm→256→256→1(sigmoid)` 小头：干预/good 帧 V=1、bad 帧 V=0、episode 成败弱监督 V=success（w=0.3） |
| ③ 验收 | `验收: V(bad)=... V(good)=... 分离度=...` | **关键**：V(bad) 应显著低于 V(good)（分离度 > 0） |
| ④ 写回 | `sidecar 已更新：补正优势 X / 补负优势 Y / 人标保留 Z` | V(s) 写入 `critic_value` 列 + 未标注帧按 V 水平相对本集均值补标（人标/干预帧不动） |

**产物（均在 `<dataset>/meta/`）**：

| 产物 | 说明 | 下游 |
|---|---|---|
| `critic_feats_siglip.npz` | 视觉特征缓存（N×~1159 维，ela3_demo5≈700MB） | 重训头/重跑命中缓存，不再重算 |
| `critic_head.pt` | 价值头权重 | 复现 / 重应用 |
| sidecar `critic_value` 列 | 每帧价值 V(s) | `recap_pg.py` 读它算 V 水平优势 |
| sidecar `advantage_indicator` 补标 | 未标注帧按 V 水平补 ±1 | R2 的 RecapWeighter 也受益（密集化） |

**critic 验收口径（成不成的唯一判据）**：`V(bad)` 应显著低于 `V(good)`——分离度
`V(good)−V(bad) > guard_frac×σ(V)`（guard_frac 默认 1.0）。state 后端 V(s) 近平坦
必判废；siglip 视觉后端才有分离度。`recap_pg.py` 内置同一守卫（`--guard-frac`，
`--no-critic-guard` 可关）——critic 不合格则丢弃 critic 来源、退回 indicator。

**组相对优势（π\*0.6 方式）**：R1b 的 `group_advantage`（GRPO 组内标准化，来自独立
标注的 episode 成败）作为 **episode 级加性调制项**：`A(t) += group_beta × group_adv(episode)`，
作用于成组集（≥4 轮同任务）的**所有帧**。`--group-beta` 默认 0.5（0=关）。未分组 / 小组
A=0 自动无害；`episode_success=None`（不确定）维持"从帧标签派生兜底"（R1b 现行为）。
⚠️ 组优势是 episode 级信号，不能当"逐帧兜底"用（否则被 critic/人标覆盖、永远 0 帧）。

**后续衔接**：

```
recap_critic (写 critic_value，V 水平)
      ↓
recap_group_advantage (写 group_advantage，episode 级)
      ↓
recap_pg.py --write → 合并：干预+1 / 人标±1 / critic V水平(未标注帧) ⊕ β_g×组优势调制
      ↓
训练时 RecapPGWeighter 读 pg_advantage → AWR 权重 exp(A/τ)
```

**siglip 后端两个实现要点**（16GB 卡已实测）：
1. 整机(3.3B)直接上 GPU 会 OOM → 改「整机 CPU 载入 → 取 `vision_tower` →
   `del` 整机 → 塔上 GPU」；
2. transformers 5.14 的 `SiglipVisionModel` 在 pi05 里 `use_head=False`，
   `pooler_output` 为 None → 改用 `last_hidden_state.mean(dim=1)` 均值池化。

**踩坑记录（2026-08-30，ela3_demo2 实测）**：

1. **progress 特征陷阱**：曾给 state/siglip 两后端各拼 `progress = 帧号/(总帧数-1)`（0→1 单调），
   结果 V(s)≈f(帧号) 单调递增、`ΔV=V(t+1)−V(t)` 恒为正、`AUC=1.0`（模型在记时间进度而非任务
   好坏）。**已去掉**（state 后端 8→7 维、siglip 1160→1159 维）。⚠️ 改维度后旧缓存
   `critic_feats_*.npz` / `critic_head.pt` 受污染必须删除再重跑。
2. **ΔV 口径方向测反**：原验收/补标用 `ΔV=V(t+1)−V(t)` 对 bad 帧求均值，捕获的是坏段**结束回升(+1)**
   而非**进入下降(−1)**（进入下降被归到"上一帧好帧"名下），导致合格 critic 被误判"失败"（`bad 段
   ΔV=+0.0003`）。**已改为 V 水平口径**：验收看 `V(bad) < V(good)` 分离度；补标用 `V−本集均值`。
3. **"复用缓存"会重训头**：`recap_critic.py` 的"复用缓存"只复用特征缓存、仍按 seed=42 重训价值头；
   而 `--apply` 补标会改变下一轮训练标签，反复"复用缓存"重训会让分离度递减（实测 0.45→0.38→0.29，
   AUC 0.96→0.93→0.90）。**训好一次就停**；要重来就「重新计算」（删缓存从头）。

**完整命令（顺序不能乱）**：

```bash
cd debugger

# ① 生成 sidecar（R3 建议关掉启发式，把中性帧密集化交给 critic）
conda run -n VLA --no-capture-output python visualize_data/make_recap_sidecar.py \
  --dataset lerobot_data/<数据集> --bad-backprop 0 --terminal-fail 0

# ② 组优势（R1b；组 <min_group 时仍全 0）
conda run -n VLA --no-capture-output python recap_group_advantage.py --dataset lerobot_data/<数据集>

# ③ critic（siglip；旧缓存删除后重提特征，~10 分钟；48GB 服务器更快）
rm -f lerobot_data/<数据集>/meta/critic_feats_*.npz lerobot_data/<数据集>/meta/critic_head.pt
conda run -n VLA --no-capture-output python recap_critic.py \
  --dataset lerobot_data/<数据集> train --backend siglip --apply
# 看"验收: V(bad)=... V(good)=... 分离度=..."：应显著为正（V bad < V good）

# ④ R3 优势合并（写 pg_advantage；group-beta 调组优势强度）
conda run -n VLA --no-capture-output python recap_pg.py \
  --dataset lerobot_data/<数据集> --operator awr --group-beta 0.5 --write

# 预览权重分布（不写文件）
conda run -n VLA --no-capture-output python recap_pg.py \
  --dataset lerobot_data/<数据集> --operator awr --preview

# 结束卡住的进程（后台占显卡时）
pkill -9 -f "recap_critic.py"
```

**训练启动**（数据准备完成后，cd debugger）：

```bash
# SmolVLA 全量（16GB 4080S 本机）
HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python train_model.py --config_path smolvla_omy_fullft_recap_pg.yaml 2>&1 | tee train_recap_pg_smolvla.log

# pi0.5 全量（48GB 服务器，freeze_vision_encoder=true + bs=8 已调好）
HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python train_model.py --config_path pi05_omy_fullft_recap_pg.yaml 2>&1 | tee train_recap_pg_pi05.log

# 对照实验（判断 R3 是否优于 R2）：同 seed 分别跑
#   recap_pg（R3）/ recap（R2）/ 删除 sample_weighting 段（纯 SFT），
#   test_inference_episode.py 留出集比 mean action error + 真机干预率。
```

### 4.7 RECAP 价值函数管线（RLinf Step1/2/3 真机复刻，`recap_value.py`）✅ 已实现（2026-09-11）

> 对标 RLinf RECAP（`RLinf-main`：`compute_returns.py` / `modeling_critic.py` /
> `compute_advantages.py`，对比分析见 `RLinf与我的项目对比.md`），在 ela3 真机
> 两轮数据（ela3_demo 遥操作 253 集 + ela3_rollout 推理录制 16 集）上实现
> "价值函数判断照这样下去还有多久完成"的完整训练管线。与 §4.6 的
> recap_critic（V 水平启发式）并存：**recap_value 是 RLinf faithful 复刻
> （DynamicReturn 真奖励 + N 步 TD 自举），信用分配更准，优先于 V 水平口径**。

**奖励结构（DynamicReturn，逐条对应设计规格）**：
- 每个非终止步 `r_t = -1`（多耽误一步扣一分，"尽快做完分数越高"——效率压力，
  这是 RLinf 有而原组回报 `success−0.25×takeover` 缺的维度）；
- 终止步：成功 `r_T = 0`，失败 `r_T = -C_fail`（`--cfail`）；
- `G_t = Σ γ^k·r_{t+k}`（γ 默认 1.0），按 |全局最小回报| 归一化到 [-1, 0]
  （与 RLinf `return_min` 归一化逐位一致）；
- 未评集（"跳过"）：默认按片段标注派生（good 帧占比>50% 且无未恢复 bad 段），
  无法派生则跳过不进入训练。

**Step 2 价值头（201-bin 分布式，two-hot 软标签交叉熵）**：
- `atoms = linspace(-1, 0, 201)`，目标值 Dirac 投影到相邻两 bin（与 RLinf
  `_compute_categorical_loss` 同式），解码 `V = Σ softmax(logits)·atoms`；
- 网络 LayerNorm→256→256→201，特征复用 recap_critic 的 state/siglip 后端
  （npz 缓存共享，ela3_demo 提 siglip 特征一次性 ~10 分钟）；
- 留出集按 episode 划分，监控 CE / MAE(V) / Spearman / bad-good 分离度。

**Step 3 N 步 TD 优势 + 分位数二值化**：
- `A_t = norm(r_{t:t+N}) + γ^N·V(o_{t+N}) − V(o_t)`（越界 V=0：回报已结清；
  RLinf Step 3 同式）——语义验证：顺利推进 A>0，**卡住不动 A=−N/分母<0**
  （白扣 N 步，效率压力的体现），走向失败 A 深负；
- 分位数二值化（top-q% 为正，默认 q=0.3），写回 sidecar
  `td_value/td_advantage/td_label` 三列 + 独立 parquet
  `td_advantage_<tag>_N<>_q<>.parquet`；
- `--fill-unlabeled`：未标注帧按 td_label 补 advantage_indicator=±1
  （人标/干预帧不动）。⚠ 语义警示：这是 CFG 式标注（供 P4 条件化 prompt
  取符号）；若与 R2 RecapWeighter 降权（bad×0.1）同开，(1-q)% 帧标负会
  大面积降权中性数据——加权训练应走 recap_pg 的连续 td_advantage 通道。

**训练侧接入（recap_pg.py 已扩展）**：TD 优势成为 critic 来源的**优先项**
（带 bad/good 分离度守卫，方向反了自动丢弃退回 V 水平），z-score 后进
AWR/CRR/linear 算子——优先级：干预 > 人标 > TD 优势 > critic V 水平 >
组优势兜底。

**实测记录（2026-09-11，ela3_demo + ela3_rollout）**：

| 步骤 | 结果 |
|---|---|
| returns | demo 253 集全成功（分母 1515）；rollout 10 成功/6 失败/0 跳过（分母 1845） |
| train（state 后端冒烟） | 159787 帧，留出 CE=4.38，MAE(V)=0.107，Spearman=0.36（姿态特征弱属预期，siglip 后端提升） |
| advantage（rollout） | N=10 q=0.3 阈值=-0.0038，正样本 2864/9545（30.0%）；sidecar 三列 + 5619 帧中性帧补标，人标/干预帧零改动 |
| 数学验证 | 归一化上下界（失败 G_0=-1.0 / 成功终止 0.0）、two-hot 投影、TD 优势三场景符号全部手算一致 |

**调参结论（实测暴露，正式训练前必看）**：
1. **C_fail=100 偏小**：rollout 最长集 1846 帧，失败惩罚在归一化空间仅占
   0.054（≈11/201 bins），成败区分度弱。RLinf 用 -300 对 ~300 帧集；
   建议 `--cfail 500~1000`（与典型集长同量级，失败≈"白干一整段"）；
2. 两数据集分母各自独立（1515 vs 1845，per-dataset 归一化与 RLinf 行为一致），
   合并训练时 V 尺度差 ~20%，可接受；后续可按混合物统一分母消融。

**命令序列**：

```bash
cd debugger
# Step 1：回报（--cfail 建议 500~1000，见上调参结论）
python3 recap_value.py returns --dataset lerobot_data/ela3_demo \
    --dataset lerobot_data/ela3_rollout --cfail 500
# Step 2：价值头（siglip 后端；state 后端秒级冒烟用）
python3 recap_value.py train --dataset lerobot_data/ela3_demo \
    --dataset lerobot_data/ela3_rollout --backend siglip --tag cfail500
# Step 3：TD 优势 + 二值化（每个参训数据集各一次；sidecar 需先生成）
python3 recap_value.py advantage --dataset lerobot_data/ela3_rollout \
    --tag cfail500 --lookahead 10 --quantile 0.3
# 策略训练消费：recap_pg.py 自动优先读取 td_advantage（无需改 yaml）
```

**部署侧衔接 ✅ 已实现（2026-09-11，`camera_pyqt.py`）**：推理中勾选"价值判断"
即可 2Hz 实时计算 V(s)（预期剩余代价 → 显示预计剩余秒数）与**回顾式 TD 优势**
（A = −窗口步数代价 + ΔV，训练侧 N 步前瞻的时间镜像，因果可用）——界面标签
实时显示 ▲正向 / ▼负向 / ◆中性（死区=窗口代价一半防闪烁），符号翻转动日志；
加载 `value_head.pt`（config 已带 denominator/fps），siglip 后端后台线程加载
视觉塔不卡界面；仅作指示不影响控制。合成场景验证：快速接近 ▲ / 卡住不动 ▼ /
死区内 ◆ 全部正确。

---

## 5. 分阶段实施计划

| 阶段 | 内容 | 依赖 | 工作量 | 状态 / 退出标准 |
|---|---|---|---|---|
| **R1a 元数据基建** | G1 策略版本标记 + G2 任务组标记（采集端+编辑端+检查） | 无 | 1~2 天 | ✅ **全部完成（2026-08-30）**：采集端（自动标记/可编辑下拉框/旧日志表头迁移/遥操作显式 null/flush 钳位）+ 编辑端（7 列表格+横向滚动/总信息统计/单集+批量改组名/检查混策略与矛盾）+ 合并 deepcopy 保留验证；无头测试全过，真机成组录制待用户实际采集 |
| **R1b 优势管线** | Step-1 组优势脚本 + Step-2 weighter 扩展 + sidecar 透传 policy/group | R1a ✅ | 1 天 | ✅ **完成（2026-08-30）**：合成组数学验证（开火组 A=+0.665/−1.700/+0.074 与手算一致）；weighter 四态（成功组帧钳 1.5 / 失败组帧钳 0.3 / 干预帧 2.0 恒定 / bad 帧 0.1 恒定）；真实训练 loss 验证待 R2 首轮 |
| **R1c 条件化训练（P4）** | Step-3 prompt patch + 消融开关 + 部署恒 positive（G5） | R1b ✅ | 2~3 天 | ✅ **代码完成（2026-08-30）**：task 前缀注入/关闭两态验证、部署端格式一致；条件化模型对照实验待 R2 首轮训练 |
| **R2 飞轮迭代（准在线 RL）** | Step-4 完整循环 ×2~3 轮 | R1 全部 | 持续 | ⬜ **北极星：按轮统计的干预率下降**；成功率升 |
| **R3 真策略梯度（可选）** | flow-matching 上的 REINFORCE/flow-QL 类 on-policy 更新 | R2 验证收益 | 研究+2 周+ | 🟡 **代码完成（2026-08-30，ela3_demo2 全链跑通）**：`recap_pg.py`（优势合并：critic **V 水平** ∪ 人标 → 逐帧 `pg_advantage`，再叠加 **组相对优势 episode 级调制** `A += β_g×group_adv`，β_g 默认 0.5）+ `RecapPGWeighter`（site-packages `type: "recap_pg"`，算子 `awr=exp(A/τ)` / `crr` / `linear`）+ `pi05/smolvla_omy_fullft_recap_pg.yaml`；VLA 环境数学/解析/roundtrip 测试通过，真机数据三信号源（人标/critic/组优势）均已进优势。**2026-09-11 增补**：RLinf RECAP Step1/2/3 真机复刻 `recap_value.py`（§4.7）——DynamicReturn 回报（r=-1/步、终止 0/−C_fail、[-1,0] 归一化）+ 201-bin two-hot 价值头 + N 步 TD 优势分位数二值化，已接入 recap_pg 为 critic 优先来源（分离度守卫）；ela3_demo+ela3_rollout 实测跑通。**诚实声明**：flow-matching 无动作似然，用 AWR/CRR 优势加权策略改进（offline RL 标准形式，等价 flow-QL/Diffusion-QL 策略提取），非 REINFORCE 似然比；真 REINFORCE 需仿真器或可微 Q(s,a)，仍属研究项。真实训练对照待启动 |

---

## 6. 验收与北极星指标

- **R1a 验收（✅ 2026-08-30 无头测试全过）**：
  - 组逻辑四态：同组连续复用 / 同名换策略自动新组 / 空名不分组 / 改名自动新组；
  - 端到端停止保存：标注两字段写入、部署日志旧表头迁移（旧行补空列）、
    AnnotationStore 重载往返字段保留、flush 钳位（`[415,460]→[415,457]`）；
  - 编辑端：表格策略/组取值（含 teleop 空值）、总信息统计、单集改名保留
    策略段、置空取消分组、批量同组替换（策略段各自保留）、检查混策略/
    策略段矛盾三场景检出与误报为零；
  - ⬜ 真机验收（待实际采集）：新录 3 组（每组 ≥6 轮）数据完整带两字段，
    8 项检查全绿。
- **R1b/R1c 验收（✅ 2026-08-30 无头测试全过）**：
  - sidecar 透传：policy_checkpoint/group_id/group_advantage 三列随帧平铺；
  - 组优势数学（合成开火组 8 集：5 干净成/2 败/1 干预后成）：
    R̄=0.7188 σ=0.4227，A=+0.665/−1.700/+0.074 与手算逐位一致；未分组 A=0；
    小组(n<min_group)与全同结局组自动 A=0；写回原子替换；
  - RecapWeighter 调制四态：成功组自主帧 1.5（钳）/失败组自主帧 0.3（钳）/
    干预帧 2.0（恒定不受 A）/bad 回传帧 0.1（恒定）；未分组帧 1.0 不变；
  - 条件化：sidecar 符号（indicator ∪ group_advantage）→ task 前缀
    `advantage: positive/negative, `（正/负/未标三态验证）；关闭开关原样返回；
  - 部署端：复选框实时切换 `VLAInferenceThread` 前缀，格式与训练侧逐字一致；
  - yaml：group_beta/group_weight_min/max/advantage_conditioning draccus 解析通过；
  - ⬜ 真机验收（R2 首轮）：条件化训练 loss 正常、positive 条件下无退化。
- **R2 北极星**（deployment_log.csv 按 policy 分组直查）：

```
policy        episodes  success_rate  intervention_ratio
m1               29         55%             0.25
m2(+flywheel)    40         70%             0.12      ← 干预率下降 = RECAP 起效
```

- **防遗忘红线**：每轮重训后在留出集回归原任务，成功率下降 >5% 即回滚上轮
  checkpoint（APO 教训：BC 法遗忘 9 个百分点）。

---

## 7. 风险与纪律

1. **默认关闭原则**：G3/G4 全部走 yaml 开关，一次只开一个变量；
2. **组质量纪律**：组内场景必须一致（程序不强制，操作规程约束）；n<4 自动
   降级为无组数据；全同结局组（全成功）A 全 0，自动无害；
3. **回报系数保守**：干预惩罚 0.25/次起步，β=1.0 起步，按北极星趋势调；
4. **sidecar 一致性**：剪辑/标注/分组变更后必须重跑展开 + 组优势脚本
   （P0.4 门禁第 3 条扩展）；
5. **site-packages 补丁登记**：P4 的 preprocessor patch 进 TRAIN_README §4 表。

---

## 8. 与论文的差异与诚实声明

| 维度 | π\*0.6 论文 | 本方案 | 差距性质 |
|---|---|---|---|
| 数据闭环（①） | 大规模在线 + 干预回流 | 同构，小规模（组 4~10 轮） | 规模差，机制同 |
| 优势估计（②） | 组相对（真实回报） | 组相对（成败+干预拼标量） | 回报函数更粗糙，可迭代 |
| 条件化训练（③） | contrastive advantage preconditioning（论文原始损失） | **加权条件监督**（P1.2 权重 + P4 prompt 条件） | **损失形式不同**：保条件化思想，未复刻其 contrastive 损失细节（论文细节以原文为准，实现前应核对） |
| 策略更新（③） | 含在线 RL 更新 | R1/R2 无策略梯度；R3 可选 | 本方案 R2 = "准在线"；真 RL 属研究项 |
| 部署（④） | 恒最高优势条件，零额外开销 | 同（positive 前缀） | 一致 |

**结论**：R1+R2 完整复刻 π\*0.6 的**数据飞轮与优势条件化部署形态**，
训练更新用可行的加权条件监督替代其 RL 损失；R3 评估后再决定是否追真策略梯度。

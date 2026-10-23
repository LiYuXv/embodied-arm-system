# RLinf RECAP 与本项目 RECAP 实现对比分析

> 日期：2026-09-11
> 参考来源：
> - RLinf 侧：`debugger/RLinf-main/README.md`（算法表登记 "RECAP (CFG)"）、
>   `RLinf-main/docs/source-zh/rst_source/examples/embodied/recap.rst`、
>   `RLinf-main/rlinf/{algorithms/offline/process,data/datasets/recap,models/embodiment/value_model/recap,models/embodiment/openpi_cfg}`、
>   `RLinf-main/examples/offline_rl/advantage_labeling/recap/`
> - 本项目侧：`debugger/visualize_data/PI06_RL_迁移实施方案.md`、
>   `debugger/visualize_data/RECAP_MEM_实施路线.md`

---

## 一、RLinf 中的 RECAP 算法

**RECAP = RL with Experience and Corrections via Advantage-conditioned Policies**。
RLinf README 定位为 *offline advantage-based policy optimization*（2026/04 支持）。
核心卖点：**无需在线交互**，纯离线给已有数据打分后优化策略——适合真机难大规模
采样的场景。官方实测 LIBERO-10 Task0 一轮迭代 **48.8% → 66.5%（+17.7%）**。

### 四阶段流水线（tag 机制串联，sidecar 不落 parquet）

```
Step1 Compute Returns → Step2 Value Model SFT → Step3 Compute Advantages → Step4 CFG Training
 (returns_tag)            (读 returns)            (advantage_tag)          (读 advantage)
 meta/returns_{tag}.parquet              meta/advantages_{tag}.parquet
```

### Step 1 回报计算（`compute_returns.py`，DynamicReturn 奖励结构）

- 每步 `r_t = -1`（**隐含"越快完成越好"的效率偏好**）；
- 成功集终止步 `r_T = 0`；失败集终止步 `r_T = failure_reward`（如 -300，拉开成败区分度）；
- `G_t = r_t + γ·G_{t+1}` 逆序折扣回报，γ 默认 1.0；
- 输出 sidecar（`meta/returns_{tag}.parquet`），**不改原始数据**。

### Step 2 价值模型（`modeling_critic.py`）

- SigLIP2-so400m 视觉 + Gemma3-270M 语言 + 可学习 Critic Expert；
- **分布式价值预测**（Categorical Value Distribution，201 bins，归一化到 [-1,0]）；
- 监控 `eval/spearman_correlation`（预测值与真实回报的排序一致性）防过拟合。

### Step 3 优势估计（`compute_advantages.py` + 共享 `advantage.py`）

- **N 步 TD 优势**：`A_t = normalize(r_{t:t+N}) + γ^N·V(o_{t+N}) − V(o_t)`（N=10）
  ——真奖励 + 价值自举，不是纯 V 差分；
- **分位数阈值**：top 30% 标正样本（`positive_quantile`，`>=` 阈值，RECAP 约定）；
- `relabel_advantages.py` 支持只改阈值免 GPU 重算（连续优势值与布尔标签分离）。

### Step 4 CFG 训练（`openpi_cfg_action_model.py`）

- 正样本（advantage=True）→ **条件输入**；负样本 → **无条件输入**；
- `positive_only_conditional=true` 时正样本以 `unconditional_prob=0.1` 随机
  dropout 成无条件（正则化）；
- **推理时双分支引导**：每个去噪步跑两次前向（条件/无条件），
  `v_t = (1−w)·v_uncond + w·v_cond`，`w = cfgrl_guidance_scale` 控制
  "往高优势方向推"的强度。

---

## 二、与本项目已实现工作的逐项对照

| RECAP 组件 | RLinf 实现 | 本项目实现（PI06/RECAP_MEM 文档） | 评价 |
|---|---|---|---|
| **奖励结构** | r_t=−1/步 + 终止 0/−300 | 组回报 `R_i = success − 0.25×接管次数`；sidecar 有 bad 回传/终端失败启发式 | ⚠️ **缺每步 −1 的效率压力**（见 §三.2） |
| **价值模型** | VLM 级分布式 critic（SigLIP2+Gemma3+201 bins） | `recap_critic.py`：冻结 SigLIP 特征 + 小 MLP 头（256→256→1），AUC/分离度验收 | ✅ 轻量等价物；验收口径（V(bad)<V(good) 分离度）≈ Spearman 作用 |
| **优势计算** | N 步 TD：`normalize(r) + γ^N·V' − V` | V 水平（V−集均值）+ 人标 ±1 + **组相对优势**（GRPO 式标准化，RLinf 没有！） | 🔶 各有取舍：组相对优势更适合真机小样本；TD 形式信用分配更准 |
| **正负标注** | 分位数 top-30% 自动阈值 | 人工片段标注 + 接管前 20 帧 bad（APO 规则）+ 干预段恒正 | ✅ 本项目自动标注源更丰富（无确认制采集端） |
| **策略优化** | **CFG 双分支对比** + 推理 guidance scale 可调 | **文本前缀恒 positive**（`"advantage: positive, " + task`）+ AWR 权重 `exp(A/τ)`（R3） | ⚠️ 关键差异，见 §三.1 |
| **迭代管理** | tag 版本链（returns_tag/advantage_tag） | policy_checkpoint + group_id 元数据 + deployment_log 北极星 | ✅ 已等价甚至更细 |
| **sidecar 原则** | meta/returns_*.parquet 外挂 | meta/recap_sidecar.jsonl 外挂 | ✅ 完全一致 |

---

## 三、关键差距分析（继续完善时可借鉴的点）

### 1. CFG vs 文本前缀——最值得决策的一条

RLinf 的 CFG 在**推理时**做条件/无条件双前向对比
（`v = (1−w)·v_uncond + w·v_cond`），有一个**可调旋钮 w**：w 越大越激进地偏向
高优势动作，部署者可按真机表现连续调节。

本项目的文本前缀方案 = **正条件单前向**（等价 w=1 固定、且无条件端从未参与
训练后的对比）。代价与收益：

- ✅ 零额外推理开销（CFG 每去噪步 ×2 前向，20Hz 真机伺服可能跟不上）；
- ❌ 没有对比放大机制，"高优势方向"只靠训练时正负前缀的语义差异隐式拉开；
- 🔧 **可选升级**：pi05 flow-matching 推理时跑两次（positive 前缀 / 空前缀），
  对动作 chunk 做 `a = a_base + w·(a_pos − a_base)`——纯部署侧 ~20 行即可拥有
  guidance scale 旋钮，不用重训。建议 R2 验证条件化有效后再做。

### 2. 奖励结构缺"效率压力"

RLinf 每步 −1 意味着**拖延本身就是负回报**，价值模型自然学到"离完成越近 V 越高"。
本项目组回报 `success − 0.25×takeover` 只看结局和接管次数，100 帧完成和 400 帧
完成的 R 相同。建议：组回报可加一项步数惩罚
（如 `R_i = success − 0.25×takeover − λ·len/max_len`，λ 小取 0.1），
或等 critic 上线后用 N 步 TD 替代 V 水平口径。

### 3. 可直接搬的两个工程件

- **`relabel_advantages.py` 思路**：连续优势值与布尔标签分离存储，调阈值不用
  重算——本项目已有 `recap_pg.py --preview` 雏形，把 `advantage_continuous`
  落盘即可拥有；
- **分位数自动标注**（top-X% 为正）：作为"人标 + APO 窗口"之外的第三自动来源，
  对**完全未标注的中性帧**（当前权重 1.0 直通）做分位切分，比全 1.0 更精细。

### 4. 本项目领先 RLinf 的地方（保持）

- **组相对优势（GRPO 式组内标准化）**：RLinf RECAP 纯离线无此机制；本项目
  "同任务实例 N 轮互为基线"天然免疫场景难度偏差，这是 π*0.6 的核心创新点，
  RLinf 版本反而没有；
- **无确认制采集端**（接管自动标注 + bad 窗口 + 部署日志）：RLinf 数据靠外部
  采集，本项目采集-标注-训练闭环更完整。

---

## 四、建议的下一步（按投入产出排序）

1. **R2 飞轮先跑起来**（PI06 文档中 ⬜）：真实成组采集 → 组优势 → 加权+条件化
   训练 → 看干预率是否下降——北极星没动之前不加新机制；
2. **部署侧 guidance scale**（~20 行）：双前向前缀对比，`w` 实时可调，
   替代"硬恒 positive"；
3. **组回报加步数项**：吸收 RLinf 的 r_t=−1 效率压力；
4. **优势连续值落盘 + 阈值重标**：对照实验调参成本从"重跑 critic"降到毫秒；
5. （远期）若换 pi0-FAST 载体或仿真器上线，再评估 RLinf 式真 CFG 训练 /
   真策略梯度（PI06 文档 §8 已诚实声明的边界）。

# RECAP + pi0.5 模型训练的数据流

> 日期：2026-09-12 ｜ 数据集：`lerobot_data/ela3_demo6`（298 集 / 172499 帧 = 253 遥操作 + 45 rollout）
> 关联文档：`PI06_RL_迁移实施方案.md`（§4.6/§4.7）、`RLinf与我的项目对比.md`

---

## 1. 数据流总图

```
原始采集（录制时无需价值模型）
  遥操作录制（ela3_demo，默认成功 + source=teleop）
  推理录制（ela3_rollout：干预段恒正 + 接管前20帧bad + 弹窗成败 + 组/策略标记）
      ↓ 物理合并（标注/日志/策略/组随索引重映射保留）
ela3_demo6 = data/*.parquet + videos/*.mp4 + meta/{info,tasks,episodes,stats}
      ↓ 人工标注（visualize_data）
meta/reward_annotations.json（区间 good/bad/干预 + episode_success + source
                              + policy_checkpoint + group_id）—— 唯一人工来源
      ↓ ①~⑧ 预处理链（recap预处理qt6.py）
meta/recap_sidecar.jsonl（逐帧 sidecar，训练唯一读取的派生文件）
      ↓
pi0.5 全量训练（train_model.py + pi05_omy_fullft_recap_pg.yaml）
```

## 2. 训练实际读取的（唯二通道）

| 数据 | 位置 | 训练侧消费者 | 读取字段 |
|---|---|---|---|
| **原始采集数据** | `data/chunk-*/file-*.parquet`（每帧 state/action/gripper_torque）<br>`videos/`（双相机 mp4）<br>`meta/{info.json, tasks.parquet, episodes/}` | lerobot 标准训练管线 | 图像 + observation.state + action + task 文本 |
| **逐帧优势 sidecar** ★ | **`meta/recap_sidecar.jsonl`**（172499 行，每行一帧） | `RecapPGWeighter`（sample_weighting 钩子，按 `batch["index"]` 全局帧号对齐） | **`pg_advantage`**（→ AWR 权重 exp(A/τ)）、`is_human_intervention`（×2 乘子）、`advantage_indicator`（负帧夹爪维豁免的符号依据） |

> `reward_annotations.json` **不直接进训练**——经 ① 展开成 sidecar 后才被消费。

## 3. 8 步链派生产物清单（/home/chen/yolo/lingzu电机/EDULITE_A3/el_a3_sdk_RTC_VR_recap/debugger/recap预处理qt6.py）

| 产物 | 位置 | 产生步骤 | 本轮是否进训练 |
|---|---|---|---|
| sidecar 基础列（indicator/score/干预/成败/source/策略/组） | `meta/recap_sidecar.jsonl` | ① | ✅（经 ⑥ 合并后） |
| 组优势列 `group_advantage` | 同上 | ② | ✅（⑥ 以 β_g=0.5 调制） |
| 逐帧回报 G_t | `meta/returns_cfail500.parquet` + `.json`（分母 1845） | ③ | ❌ 仅供 ④⑤ |
| **价值头模型** | **`lerobot_data/recap_value_cfail500/value_head.pt`** | ④ | ❌ 供 ⑤ 离线打分 + 部署侧实时显示 |
| **逐帧价值值 V(s)** | sidecar 的 **`td_value`** 列；另存 `meta/td_advantage_cfail500_N10_q30.parquet`（td_value/td_advantage/td_label） | ⑤ | ❌ 守卫拒绝（分离度 0.06σ < 1.0σ） |
| 特征缓存 | `meta/critic_feats_siglip.npz`（742MB，172499×1159） | ④ | ❌ 中间产物（重训提速用） |
| **pg_advantage 列** | sidecar | ⑥ | ✅ **训练唯一优势来源** |
| 部署日志 | `meta/deployment_log.csv` | 录制时 | ❌ 北极星指标（干预率）用 |

## 4. 本轮 pg_advantage 的实际成分（ela3_demo6 实测）

- 构成 = **人标 ±1 + 干预恒 +1 + 组优势调制**（价值模型 TD 通道被 ⑥ 守卫拒绝：
  分离度 0.06σ ≪ 1.0σ，18 失败集/998 bad 帧信号太稀薄——守卫在保护训练）；
- 分布验证：正 13496 / 负 5811 / 中性 153192；干预帧 A∈[+0.13,+1.0] 全正、
  bad 帧 A∈[−1.0,−0.42] 全负；AWR 权重 mean=1.135，有效样本量 113.5%；
- 中性无组帧 154521 帧 A 全 0（价值模型一帧未参与的实测证据）。

## 5. 关键结论

1. **历史数据无需"录制时的价值值"**：离线 RECAP 的设计就是"先采集、后打分"——
   ④ 在历史数据上训价值头，⑤ 对每帧离线补算 V/TD 优势写回 sidecar。
2. **价值进入训练的形式不是裸 V 值**，而是"V → TD 优势 → ⑥ 合并 → 样本权重"
   的派生链，且带分离度守卫（弱信号自动拦截）。
3. **camera_pyqt.py 的实时价值显示是部署侧监控工具**，不写训练数据。
4. **价值通道的启用是自动的**：后续每轮重跑 ⑥ 时重新校验，分离度随数据积累
   过 1.0σ 即自动并入，无需改配置、无需重采数据。
5. **增强建议**：给价值头特征加 `gripper_torque`（抓空/滑脱的力矩前兆是最强
   失败先兆信号）；cfail 用 500~1000（与典型集长同量级）。

---

## 6. 第二轮及以后迭代（拿到新模型如 20000 后）

```
部署 20000 → 推理录制（同第一轮：接管自动标注/组标记/策略标记，无需额外操作）
  → 合并进新数据集（如 ela3_demo7）
  → 重跑 ①~⑧（价值头在更大数据集上重训，tag 建议按轮区分如 cfail500_demo7）
  → 重训 pi0.5
北极星：deployment_log.csv 按策略版本统计干预率逐轮下降
```

### 6.1 常见问题：第二轮采集时需要同步记录旧价值模型的实时价值值吗？

**不需要。** 用新模型（如 20000）自主运行采集 rollout 时，无需同步记录
旧价值模型（`recap_value_cfail500`）实时计算的价值值。原因有三：

1. **管线设计就是"先采集、后离线打分"**：价值值从来不是在线采集的——
   新数据合并后，价值头在**更大的数据集上重训**（信号更强），再对每帧
   离线重算 V 和 TD 优势（④⑤）。离线重算的准确性严格优于录制时用旧模型
   在线打的分。
2. **在线记录的旧价值值是"过时的弱信号"**：价值函数本质是"在当前行为分布
   下的预期剩余代价"——20000 模型的行为已不同于 008000，旧价值头对第二轮
   数据是离策略（off-policy）的；且 ⑥ 读的是 sidecar 里重算的
   `td_advantage`，不读任何录制期日志。
3. **录制端需要的标注已全自动**：接管干预段、接管前 bad 窗口、弹窗成败、
   组名、策略版本标记均由 camera_pyqt.py 自动完成——采集操作与第一轮
   完全相同，零新增负担。

**唯一建议保留的用法**：推理时可勾选"价值判断"当**仪表盘**用——V 停滞/
下滑说明策略卡住或在恶化，是操作员决定"何时接管救场"的辅助依据；
它只显示、不落盘，对数据集无任何影响。

### 6.2 常见问题：visualize_data_qwen.py的recap界面中“优势查看”发现专家遥操作集的episode片段的 pg_advantage 全 0 是否正常？

> 现象（2026-09-12 实测，"显示优势"弹窗）：遥操作集 pg 恒 0、td_advantage
> 在 0 附近波动；rollout 无接管+成功集 pg 恒正；接管集干预段 pg=1。
> 疑问：专家遥操作数据的 pg 值是否应为 1？

**正常且符合设计，不是 bug。** 关键概念：**pg_advantage=0 ≠ 不训练**——
AWR 权重 = `exp(A/τ)`，A=0 → 权重 exp(0)=**1.0（正常模仿）**。遥操作数据
（253 集 / 88%）以 1.0 权重稳定训练，是防遗忘的主体。

| 观察 | 原因 | 判定 |
|---|---|---|
| 遥操作集 pg 恒 0 | 只有 episode_success、无片段区间 → indicator=0（中性）；无组名 → 无组调制 → ⑥ 合并后 A=0 | ✅ 设计内 |
| td_advantage 0 附近波动 | TD 优势 = 瞬时进展 − 步数代价：平稳进行时 V 缓升恰好抵消 −N/分母，均值≈0 | ✅ 正常 |
| rollout 无接管+成功 → pg 恒正 | 成组且组回报为正 → 组优势调制（A += 0.5×group_adv）抬高整集 | ✅ R1b 生效 |
| 接管+成功 → 干预段 pg=1 | 干预恒正（RECAP 规则） | ✅ 正确 |

**为什么专家数据不该 pg=+1**：论文中专家数据的"正优势"是 CFG 的 prompt
条件语义，不是损失权重；在本项目的 AWR 加权翻译里对应"权重 1.0 正常模仿"
（RECAP_MEM P1.2 明确：中性=1.0 是防遗忘混训的主体）。若提到 +1，权重将变为
exp(1)≈2.7×，253 集遥操作会压倒性主导，rollout 纠错信号（干预 ×2~5、
bad ×0.4）的相对对比度反而被压缩——破坏"专家做稳定 backbone、rollout 提供
增量信号"的分工。如需温和抬升可用 recap_pg.py 的 `--expert-bonus` 类开关
（如 +0.3≈权重 1.35），默认建议保持现状。

---

## 7. 训练：价值函数模型value_head.pt`的数据构成（value_head.pt checkpoint 实测）

> 来源：`lerobot_data/recap_value_cfail500/value_head.pt` 的 config 与权重形状
> （2026-09-12 读取核实）。

### 7.1 输入特征：1159 维/帧（摄像头画面 ✅ + 机械臂状态 ✅）

| 成分 | 维度 | 来源 | 说明 |
|---|---|---|---|
| **双相机画面** ✅ | 1152 | `videos/` cam1+cam2 解码帧 | pi0.5 冻结视觉塔（SigLIP2）提取嵌入：224×224 缩放 → [-1,1] 归一化 → patch 均值池化 → 两相机取平均 |
| **机械臂状态** ✅ | 7 | parquet 的 `observation.state` | 末端位姿 x,y,z,rx,ry,rz + 夹爪角度（弧度） |
| 夹爪力矩 | — | — | ❌ 没有用（增强点：力矩是抓空/滑脱的最强失败先兆，后续可加入） |

> 1159 = 1152 + 7，与 checkpoint 的 `dim_in: 1159`、`backend: siglip` 完全吻合。

### 7.2 监督标签：③ 从标注派生的回报（不是画面/动作）

- 每帧目标值 = **G_t（折扣回报）**：r=-1/步、成功终止 0、失败终止 −500，
  按分母 1845 归一化到 [-1,0]；
- 来源链：`reward_annotations.json` 的 episode_success（弹窗成败标注）
  → ③ returns parquet → two-hot 投影到 201 个 bin → 交叉熵；
- 即价值头学的是"**从当前局面（画面+位姿）预期还要扣多少分才能完成**"
  ——"照这样下去还有多久完成"。

### 7.3 数据范围与网络结构

- **数据**：ela3_demo6 全部 298 集 / 172499 帧（253 遥操作 + 45 rollout，无跳过）；
- **结构**：LayerNorm → 256 → 256 → **201 bins**（W1: 1159→256，W3: 256→201），
  two-hot 软标签交叉熵，解码 V = Σ softmax·atoms（atoms = linspace(-1, 0, 201)）；
- **留出集**：29 集 / 17639 帧（Spearman=0.769，排序一致性良好）；
- checkpoint config 全量：`{dim_in: 1159, backend: siglip, num_bins: 201,
  v_min: -1.0, v_max: 0.0, tag: cfail500, datasets: [ela3_demo6],
  denominator: 1845.0, fps: 20}`（denominator/fps 供部署侧实时价值判断使用）。

一句话总结：**价值头用双相机画面 + 机械臂位姿作输入，用"每步 -1、失败 -500"
的回报作标签，学"当前局面离完成还有多远"；动作不用是价值函数的定义使然，
力矩没用是后续可加力的增强点。**

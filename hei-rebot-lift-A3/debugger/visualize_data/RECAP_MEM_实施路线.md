# RECAP / MEM 分阶段实施路线（ela3 真机项目）

> **后续阶段（2026-08-30 起）**：π\*0.6 在线 RL 迁移（策略版本标记 / 任务组标记 /
> 组相对优势 / 优势条件化 P4）见 **`PI06_RL_迁移实施方案.md`**。

> 目标：在现有 lerobot 0.6.1 + pi0.5/SmolVLA/ACT 链路上，逐步实现 π\*0.6 风格的
> RECAP（优势条件 RL）与 MEM（记忆增强）。
> 设计参考：本仓库 `数据清洗.md`（片段级奖励标注方案）、
> `CBC_Pi0.7_Openpi-main`（sidecar 数据流 / RL token 加权 / MEM prompt 追加的工程模式）、
> π\*0.6 论文（arXiv:2511.14759）。
>
> **总原则**：数据侧改动一律走 **sidecar（外挂文件），不修改原始 parquet**；
> 训练侧改动以最小 patch 实现，默认行为不变（所有新功能默认关闭）。
>
> **路线决策（2026-08-29）**：训练侧选定 **路线 A**——数据规则借鉴 APO 论文，
> 训练侧沿用本路线 P1（损失加权）+ P4（优势条件化），不引入偏好优化目标
> （pi0.5 为 flow-matching，无动作对数似然，APO/KTO/DPO 不可用，详见附录 A）。
>
> **标注粒度决策（2026-08-29）**：以**片段级标注**为唯一逐段标注形态
> （good/bad/neutral 小段，纪律见 `项目分析.md文件/数据清洗.md` §8.2）；
> episode 级成败为**一键直标**（部署/保存时点一下，几乎零成本，
> 是 critic 的 r_T 主监督信号），未直标时由片段标注派生兜底（P0.1）。
>
> **部署采集决策（2026-08-29 定稿，同日二次修订）**：**无确认制**——部署录制
> 自动标注直接生效（事后 GUI 双击可改）；接管期间自动标干预段（恒正，支持多次
> 接管）；**接管前 20 帧自动标 bad**（恢复 APO 规则：P1 阶段负优势的唯一自动
> 来源——否则失败帧按 0 权重被正常模仿；P3 阶段作 critic 监督样本；窗口可配
> `_rollout_pre_bad_frames`，与上一段干预重叠时自动裁剪）；遥操作示范保存默认
> `episode_success=True`；rollout 独立数据集 `ela3_rollout`，两路录制互斥。
>
> **数据质量工具链（2026-08-29 晚定稿，全部已实现）**：
> - **后期处理分页**（智能分析右侧）：抖动检测（ZigZag 方向反转计数 + 滑窗幅度
>   阈值，默认 15 帧/7mm/5°/2 次反转）+ 紫色标记（播放不消失）+ **剪辑晃动**
>   （复用剪辑静止安全管线，剪辑后**自动重算 action 标签** `refix_action_labels`
>   ——运动段剪辑必需，自动估计前瞻 k；干预段被覆盖时确认框红字警告）；
> - **数据集检查升级为 8 项**：新增问题8（RECAP 标注一致性：键越界/区间越界/
>   重叠/字段非法/sidecar 过期），**sidecar 过期与数据损坏明确区分**（仅过期时
>   给"⚠ 数据结构完好"温和结论，真结构问题才红 ❌）；
> - **总信息面板 RECAP 概览**：标注分布 + sidecar 三态（未生成/已生成/过期红警），
>   内存计算毫秒级（`summarize_from_frames`）；
> - **单条任务改名孤儿回收**（`_gc_orphan_tasks_df` 接入，修复 ela3_demo2 实例）；
>   合并时标注/部署日志/source 自动随 episode_index 重映射保留；
>   剪辑后内存 stats 自动刷新（消除检查误报）；
> - **存量转换** `convert_to_recap_format.py`：旧数据集一键补齐 RECAP 标注
>   （只补缺不覆盖）+ 重新生成 sidecar（ela3_demo 253 集已转换）。

---

## 0. 现状盘点（2026-08-29 晚核实，数据侧全部就绪）

| 能力 | 状态 | 位置 |
|---|---|---|
| 示范数据录制 / 训练 / 部署全链路 | ✅ 已通 | `widgets/camera_pyqt.py`、`train_model.py` |
| 片段级标注 GUI（双击编辑/干预列/快捷键/派生成败/source） | ✅ | `vd_trajectory.py` RECAP 分页 + `annotation_store.py` |
| 区间 → 逐帧展开（sidecar，不改 parquet） | ✅ `make_recap_sidecar.py`（CLI + RECAP 面板"生成训练 sidecar"按钮 + 总信息概览） | 本目录 |
| 遥操作/推理双路录制（默认成功/source/干预自动标注/接管前 bad/部署日志） | ✅ | `camera_pyqt.py` 推理录制按钮 |
| 剪辑/合并/删除的标注重映射与防护 | ✅ 方案乙（剪辑随帧重映射不丢失） | `remap_intervals_after_trim` 等 |
| 抖动检测 + 剪辑晃动（含 action 重算） | ✅ 后期处理分页 | `detect_jitter_intervals`、`refix_action_labels` |
| 数据集检查 8 项（含 RECAP 一致性 + sidecar 过期区分） | ✅ | `vd_dialogs.py` |
| 存量数据集 RECAP 格式转换 | ✅ `convert_to_recap_format.py` | 本目录 |
| **样本级损失加权（filtered BC）** | ✅ **已实现（2026-08-30，P1.2）** | `debugger/recap_sidecar.py` + site-packages `RecapWeighter`（见 TRAIN_README §4）+ `pi05_omy_fullft_recap.yaml` / `smolvla_omy_fullft_recap.yaml` |
| advantage 条件注入 prompt | ❌ P4 | pi05 preprocessor patch |
| 价值函数 critic | ❌ P3 | — |
| MEM（memory 字段 → prompt） | ❌ M1 | — |

**数据集清单（全部已统一为 RECAP 格式，含 source 溯源）**：

| 数据集 | 内容 | 状态 |
|---|---|---|
| `ela3_demo` | 253 集 / 150242 帧，存量示范（批量转换 source=teleop success=True） | sidecar 已生成 |
| `ela3_demo1` / `ela3_demo3` | 新录遥操作示范（自动默认成功） | 就绪 |
| `ela3_rollout` | 4 集部署实录（含干预段/接管前 bad/成败/部署日志），已剪辑抖动 | sidecar 已重新生成 |
| `ela3_demo2` | demo3 + rollout 物理合并产物（标注/日志/来源全保留） | 8 项检查全绿 |
| `ela3_demo5` | **285 集 / 166335 帧 = 256 遥操作 + 29 rollout**（16 成功/13 失败，97 干预段/25 bad 段，干预帧 4974）——P1.2 加权训练主数据集 | 2026-08-30 就绪：teleop 批量补标（success=True）、3 个越界干预区间截断、sidecar 已生成（bp=50/tf=10，166335 行），action≠state 抽查通过 |

**数据规模现实性**：示范 253 集 / rollout 4 集。RECAP 论文的机上经验量级远超于此，
按"先 filtered BC 获益，再逐步堆 on-policy 数据"的节奏推进。

---

## 总体数据流（目标形态）

```
录制/部署 (camera_pyqt.py)
  ├─ 遥操作示范（保存默认 success=True + source=teleop）
  ├─ 推理录制（ela3_rollout）：干预段恒正 + 接管前20帧bad + 弹窗成败 + 部署日志
  └─ （后期处理：静止/抖动检测与剪辑，标注随帧重映射、action 自动重算）
                 ↓
标注/复核 (visualize_data)：片段级 good/bad/neutral + 干预列 + reviewed；
  episode_success 人工直标优先、片段派生兜底；source 溯源
                 ↓  meta/reward_annotations.json（区间稿，唯一人工来源）
展开 (make_recap_sidecar.py / RECAP 面板按钮)：→ meta/recap_sidecar.jsonl
  （逐帧 advantage_indicator/score + 干预 + 成败 + source，不改 parquet，可重跑覆盖）
                 ↓
训练 (train_model.py)：RecapWeighter（lerobot sample_weighting 钩子，type=recap）
  按 batch["index"] 查 sidecar 做样本级 loss 加权 + APO 夹爪维豁免
  （good×1.0 / bad×0.1 / 中性=1.0 / 干预×2.0）[+ P4 advantage prompt 注入]
                 ↓
部署 (camera_pyqt.py)：advantage 恒 positive；接管即触发干预采集
                 ↺ 回到录制，形成闭环（P5 迭代，北极星指标=干预率下降）
```

---

## P0：标注工具链加固（1~2 天，纯本目录内改动） ✅ 已完成（2026-08-29）

> 目标：标注数据可信、可追踪、不怕剪辑/合并。这是后续一切训练实验的地基。
>
> **完成内容**（遵标注粒度决策：episode 成败不人工标注，由片段标注派生）：
> schema v2.0（episode_success 派生字段/reviewed/is_human_intervention + v1 自动迁移）、
> `derive_episode_success()` 派生规则（正优势帧占比>50% 且无未恢复 bad 段；干预段恒为正优势）、
> 区间重叠拒绝、保存自动备份（10 份滚动）、面板 episode 行**只读**派生成败显示 + 已审核勾选、
> 区间表干预列、轨道蓝色干预段、G/B/N 快捷键（最近 2 秒区间）、TrimSlider 右端 off-by-one 修复、
> 剪辑三入口（选中段/静止段/批量静止）防护（警告 + 备份 + 失效区间自动清除）、
> 合并目标标注防护（备份并移除）、**删除防护（备份 + 标注键自动重映射 `remap_after_delete`）**。
> 验证：`AnnotationStore` 40 项单元测试全过；GUI 冒烟（minimal 平台 + 3D 面板打桩）通过。

### P0.1 schema 补全（`annotation_store.py`）

在现有 `{"start","end","label","note"}` 基础上扩展：

```json
{
  "version": "2.0",
  "episodes": {
    "0": {
      "episode_success": true,          // 派生字段（非人工标注）：good 帧占比>阈值且无未恢复 bad 段=成功
      "reviewed": true,                 // 新增：是否已完整审核（区分"未标注"与"确认中性"）
      "intervals": [
        {"start": 10, "end": 50, "label": "good", "note": "",
         "is_human_intervention": false} // 新增：干预段标志（Recap 中干预段恒为正优势）
      ]
    }
  }
}
```

- `episode_success` **优先级：人工直标 > 片段派生**（2026-08-29 修正）：部署/保存时
  一键标记的成败是 critic 的 r_T 主监督信号，必须准确，展开时优先采用；
  未直标的 episode 由片段标注派生兜底（good 帧占比 > 50% 且不存在未接恢复段的
  bad 段 = 成功）。GUI 面板只读显示（标注来源标明"人工/派生"），人工只需标片段。

实现要点：
- `load()` 对旧版（1.0，episodes 值为区间列表）自动迁移：列表 → `{"intervals": [...]}`；
- `add_interval()` 增加**重叠校验**：同帧已有其他标签区间时拒绝或提示拆分（当前只排序不查重，展开时会互相覆盖）；
- `use_advantage` 不在标注层存，放训练配置（见 P1），数据只存事实标签。

### P0.2 标注面板补全（`vd_trajectory.py` RECAP 分页 + `vd_main_window.py`）

- **不做**人工整段标注按钮（遵标注粒度决策）；episode 行信息显示派生的
  `episode_success`（只读）+ "标记为已审核"复选框；
- 区间表格加"干预"列（勾选框），干预段在 RecapTrackWidget 用**蓝色**显示（现有 good绿/bad红/neutral橙 之外新增）；
- 键盘快捷键：播放中 `G`/`B`/`N` 直接以最近 2 秒为区间打标（当前只能鼠标点按钮）；
- 修复 `TrimSlider._frame_from_x` 右端 off-by-one（`round(ratio*total)` 应为 `total-1`）。

### P0.3 剪辑/合并/删除的安全防护（`vd_data_ops.py`、`vd_main_window.py`）

标注以 `episode_index` 为键，**剪辑静止段、合并、删除 episode 都会重排索引**
（删除后的重排见 `项目分析.md文件/数据问题分析.md` 问题 6 修复流程）导致标注错位。
防护与保留措施：
- **剪辑（方案乙，2026-08-29 已实现）**：剪辑前自动备份标注文件；剪辑启动时登记
  实际删除的帧区间，剪辑成功后标注区间**按保留帧重映射**——原样前移 / 截断 /
  拆分（完全落在删除段内的丢弃），标签/备注/干预标志继承，episode 级字段
  （成败/已审核/source）保留（`remap_intervals_after_trim` + `_recap_after_trim_annotations`）；
- **删除 episode**：`remap_after_delete` 自动重映射标注键，后续 episode 标注不错位；
- **合并**：目标数据集已有标注时弹窗警告并备份；源标注按合并后索引重映射保留
  （P0.3 合并增强，见 `merge_lerobot_datasets.merge_recap_annotations`）；
- **单条任务改名孤儿回收（2026-08-29 修复）**：`_save_task_name` 接入
  `_gc_orphan_tasks_df`——改名使旧任务名失去最后一处引用时自动删除任务行并把
  task_index 压实为 0 基连续（此前单条改名留孤儿行，总信息报"0 帧引用"告警；
  批量改名/删除路径早有此逻辑，唯单条改名遗漏）；
- `AnnotationStore.save()` 每次保存自动留 `reward_annotations.<时间戳>.bak` 一份。

### P0.4 数据质量门禁（`项目分析.md文件/数据问题分析.md` 的血泪教训，标注/训练前必查）

1. **action==state 防线（问题 5，曾致真机全失败）**：P2 的部署录制是**新录制入口**，
   必须复用 `LeRobotRecordThread` 的 lookahead 写入（`action[t]=pose[t+N]`），
   **禁止新造录制路径**；每个新数据集（含 ela3_rollout）训练前抽查
   `mean|action−state| > 0`（一行 pandas），为 0 即整集报废，不得训练。
2. **先剪静止段，再标注，再训练**（`数据清洗.md` §8.3）：未标注帧是训练主体，
   静止段不剪则模型学会"站着不动"。固定顺序：
   `采集 → 剪辑静止 → 合并 → 标注 → 展开 sidecar → 训练`（`数据清洗.md` §8.4）。
3. **frame_index 对齐**：标注的 start/end 基于 episode 内帧索引，任何
   剪辑/删除/降噪/合并后必须重跑 `make_recap_sidecar.py`（P1.1），旧 sidecar 直接作废。
4. **训练前过"数据集检查"（8 项，2026-08-29 升级）**：问题 6（三方一致）、
   问题 7（task_index 0 基连续）、**问题 8（RECAP 标注一致性：键/区间越界、
   重叠、字段非法、sidecar 行数/时效）**必须全绿。**仅 sidecar 过期**（剪辑/删除/
   标注更新后的正常现象）给"⚠ 数据结构完好"温和结论——重新点"生成训练
   sidecar"即可，不要误读为数据损坏；真结构问题才显示红色 ❌。
   ela3_rollout 与 ela3_demo 混合训练前同样过检。
5. sidecar 路线的额外收益：标注流程不改 parquet，天然免疫问题 6/7 那类
   "先改数据后写元数据"的中断损坏。
6. **总信息面板即战况板**：RECAP 概览（标注分布/成败计数）+ sidecar 三态
   （未生成提示 / 已生成行数+时间 / 过期红警）实时可见，内存计算毫秒级。

**验收**：旧标注文件能加载迁移；重叠区间被拒绝；剪辑/合并/删除已标注数据集时弹窗；
episode_success 派生结果与人工抽查一致；数据集检查 8 项全绿。

### P0.5 后期处理：抖动检测与剪辑晃动（2026-08-29 实现，`vd_trajectory.py` 后期处理分页）

> 背景：推理录制（rollout）数据机械臂抖动明显（策略输出振荡/chunk 边界跳变），
> 不处理则模型以正常权重模仿自己的振荡行为。

- **检测算法**（`detect_jitter_intervals`）：滑窗内**带迟滞死区的方向反转计数**
  （ZigZag 指标，死区=幅度阈值/2，抗数值噪声）+ 窗口极差 ≥ 幅度阈值；
  任一轴（xyz 毫米 / rx,ry,rz 度）命中即标记。默认参数：**窗口 15 帧（0.75s@20fps）、
  位置 7mm、姿态 5°、反转 ≥2 次**（= 完整一个来回；设 1 可捕捉单次大折返，
  但可能是正常任务语义——先升后降，慎用）；
- **紫色半透明标记**三幅子图，与灰色静止条并存；**播放视频时标记不消失**
  （`set_data` 仅在切换 Episode 时清空，播放重绘路径保留）；
- **剪辑晃动**：与剪辑静止同一安全管线（备份/一致性校验/RECAP 重映射）+ 两项专属：
  - **action 标签自动重算**（`refix_action_labels`：自动估计前瞻 k，
    `action[t]=state[t+k]`，原子重写 + stats 重算）——运动段剪辑的安全前提
    （剪辑静止因位姿恒定可容忍，运动段不重算则 action 指向已删除状态）；
  - **干预段保护警告**：抖动窗覆盖人工干预段时确认框红字列出
    （干预段是 RECAP 最珍贵数据，被抖动窗吞掉会随剪辑删除）；
- 检测边界含约半窗的保守膨胀（滑窗固有，利于剪辑安全）；
- **方法论文献**：轻度抖动首选已有"降噪"（零相位平滑，不改时间结构）；
  根因治理在推理侧（RTC/低通/多次采样平均，部署端已备）。

---

## P1：sidecar 展开 + 训练侧损失加权（filtered BC，2~3 天）

> 对应 RECAP 论文 offline RL+SFT 基线一档：good 正常学、bad 降权/过滤、干预段加权。
> **这是全路线投入产出比最高的一步，先做它。**

### P1.1 `make_recap_sidecar.py`（本目录，**已实现并通过测试 2026-08-29**）

把 `reward_annotations.json` 展开为逐帧 sidecar，**不写 parquet、不改 info.json**（弃用 `备份/expand_annotations.py` 的注入路线）：

```bash
python3 make_recap_sidecar.py --dataset ../lerobot_data/ela3_demo            # 平坦展开
python3 make_recap_sidecar.py --dataset ../lerobot_data/ela3_demo --dry-run  # 预览分布
# RECAP 打分规则（可选项）：bad 段负优势回传 50 帧 + 失败 episode 末 10 帧自动标 bad
python3 make_recap_sidecar.py --dataset ../lerobot_data/ela3_demo --bad-backprop 50 --terminal-fail 10
# 输出 ../lerobot_data/ela3_demo/meta/recap_sidecar.jsonl
```

三种等价入口：CLI / RECAP 面板"🚀 生成训练 sidecar"按钮（参数=回传N帧/失败末K帧
两个旋钮，注意与 CLI 保持一致）/ 总信息面板只读概览（`summarize_from_frames`）。

**重跑覆盖语义**：sidecar 是编译产物——生成后随时可继续编辑标注/剪辑/删除/重标，
重跑同一命令即**原子覆盖**（tmp+replace），新旧无缝切换，无需手动删旧文件。

**存量转换**：`convert_to_recap_format.py` 把旧数据集补齐 RECAP 标注
（`--source teleop --success true` 默认；只补缺不覆盖已有标注）+ 可选重生成 sidecar。
ela3_demo（253 集）已转换。

逐帧记录格式（对齐 CBC schema）：

```json
{"episode_index": 0, "frame_index": 10, "advantage_indicator": 1,
 "advantage_score": 1.0, "is_human_intervention": 0, "episode_success": 1,
 "rl_token_weight": 1.0}
```

展开规则：
- 区间内每帧取区间标签的 advantage（1/-1/0），未标注帧 = 0；
- 干预段帧：`is_human_intervention=1` 且 `advantage_indicator` 强制为 1；
- `episode_success` 由片段标注派生（P0.1），逐帧平铺（1/0/-1 = 成功/失败/未评，
  供 critic 训练用，P3）；
- `advantage_score`：强度幅值 [-1,1]，平坦模式等于 indicator；开启 RECAP 规则后
  承载回传衰减值，供训练侧调制权重；
- `rl_token_weight` 默认 1.0，真正的权重映射放训练侧配置；
- `pending=true` 的预标区间（P2.2 自动预标）跳过不展开——未确认不进训练；
- RECAP 打分规则（可选开关，`--bad-backprop N` / `--terminal-fail K`，默认关）：
  bad 段向前回传 N 帧线性衰减（n-step 前瞻的无 critic 近似，只填未被人标覆盖的帧）；
  派生失败 episode 末 K 帧自动标 bad（r_T=-C_fail 的近似）。critic（P3）上线后
  由 V(s) 替代这两个启发式。

### P1.2 训练侧：数据集包装 + 损失加权 ✅ 已实现（2026-08-30，实现方式有更新）

> **实际实现**（比原方案更省 patch）：lerobot 0.6.1 已内置 sample_weighting
> 训练钩子（`lerobot_train.py` 的 `update_policy` 原生支持逐样本加权 +
> pi05/smolvla forward 原生支持 `reduction="none"`），因此**不需要**自造
> `SidecarDataset` 包装数据集，也不需要 patch `train_model.py` 入口——
> 权重器直接按 `batch["index"]` 全局帧号查 sidecar（sidecar 行序与数据集
> 全局帧序一致），零数据集包装。

落地组件：
1. **`RecapWeighter`**（site-packages `lerobot/utils/sample_weighting.py`，
   工厂分支 `type: "recap"`，重装 lerobot 后按 TRAIN_README §4 重打）：
   - 读 `<dataset_root>/meta/recap_sidecar.jsonl` → 权重映射：
     good×positive(1.0) / bad×negative(0.1) / **中性/未标注=1.0**（与 CBC 的
     "非正即负"不同——中性正常模仿，是防遗忘混训的主体）/ 干预×intervention(2.0)，
     clamp [0,3] 后每 batch 归一化 Σw=B（梯度尺度与不加权一致）；
   - **APO 夹爪维豁免**：向 batch 注入 `recap_dim_scale` (B,D)，负样本夹爪维
     scale=1/negative_weight → 与样本级标量权重相乘后有效权重恒 1.0；
2. **模型 forward 补丁**（pi05 / smolvla 各一处，带 shape 守卫，键不存在时
   与原实现逐位一致）：reduction="none" 路径应用 `recap_dim_scale`；
3. **yaml**（`sample_weighting` 是 TrainPipelineConfig 原生字段，draccus 直接
   解析，无需 patch config）：`pi05_omy_fullft_recap.yaml` /
   `smolvla_omy_fullft_recap.yaml`（ela3_demo5 全量微调 + recap 加权）；
4. **本地 `debugger/recap_sidecar.py`**：权重预览 CLI（训练前一键查看分布）+
   `SidecarDataset` 包装类（调试/未来扩展）+ 与 weighter 同源的权重映射。

**数据准备记录（ela3_demo5 → P1.2 主训练集，2026-08-30）**：
1. teleop 批量补标：`convert_to_recap_format.py --source teleop --success true`
   （只补缺不覆盖）→ 新建 256 集标注，rollout 29 集已有标注完整保留
   （16 成功/13 失败，97 干预段 + 25 bad 段）；
2. 修复 3 个越界干预区间（ep273 [415,460]→457 / ep276 [539,609]→606 /
   ep282 [35,124]→123，均超 2~3 帧——rollout 停录时队列 flush 丢帧所致，
   非合并错位；按保存后实际帧数截断，AnnotationStore 自动备份）；
3. sidecar 生成：`make_recap_sidecar.py --bad-backprop 50 --terminal-fail 10`
   → 166335 行 == 总帧数 == info.json；分布：good 4974 / bad 1537
   （含回传 979 + 终端失败 130）/ 中性 159824 / 干预 4974；
4. 门禁抽查：sidecar 行序 == 数据集全局帧序（index 列 == 行号，episode 升序，
   RecapWeighter 查找前提）；`mean|action−state| = 0.0144`（问题 5 防线通过）。

**启动训练**（cd debugger；删除 yaml 中 `sample_weighting` 段即普通 SFT 对照组）：

```bash
# 权重分布预览（训练前检查）
python3 recap_sidecar.py --dataset lerobot_data/ela3_demo5
# pi0.5 全量微调 + RECAP 加权（96GB 服务器；16GB 卡请改用 expert_only 配置并复制 sample_weighting 段）
HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python train_model.py --config_path pi05_omy_fullft_recap.yaml 2>&1 | tee train_recap_pi05.log
# SmolVLA(smolvla_torque) 全量微调 + RECAP 加权
HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python train_model.py --config_path smolvla_omy_fullft_recap.yaml 2>&1 | tee train_recap_smolvla.log
```

验收（已过）：recap 关闭时逐位一致（补丁带键守卫）；RecapWeighter 单元测试
（映射/归一化/豁免/钳位/越界回退）+ 集成测试（真实 ela3_demo5 batch 经 pi05
preprocessor 后 index 存活、权重/维度缩放正确）；draccus 解析两个 recap yaml
通过。**对照实验（SFT vs 加权 SFT 同 seed）待训练后记录到附录 B**。

**APO 论文的三条实战修正**（已并入实现，依据见附录 A）：
- **负权重不作用于夹爪维** ✅（recap_dim_scale 豁免，见上）；
- **每轮训练必须混入原示范数据** ✅（ela3_demo5 本身= 256 遥操作集 + 29 rollout
  集物理合并，遥操作帧占 96%+，中性 1.0 权重正常模仿）；
- 权重保守 ✅（negative 0.1 / intervention 2.0 / max 3.0 为默认）。

---

## P2：部署侧数据采集闭环（ela3_rollout，**已定稿并实现 2026-08-29**，`widgets/camera_pyqt.py`）

> Recap 超越 IL 的最主要数据来源：**策略执行中的失败 + 人工接管纠正**。
> 钩子链路：手柄分页"连接手柄" → `pause_vla_control()` → `toggle_vla_control_pause()`
> （与"暂停控制"按钮同一汇聚点），部署录制的接管钩子即挂于此，一处覆盖全部路径。

### P2.0 数据流总览

```
遥操作示范录制（btn_lerobot，ela3_demo）
  保存成功 → 静默写 episode_success=True（P2.1）

VLA 推理 + 推理录制（btn_rollout_record，ela3_rollout）
  接管开始 T → 记起始帧；接管结束 R → 暂存区间对：
    [T-N, T-1] = bad（接管前 20 帧，策略失败动作；P1 降权/P3 critic 监督）
    [T, R-1]   = 干预段（good + is_human_intervention=True，恒正）
    （N 可配 _rollout_pre_bad_frames；与上一段干预重叠自动裁剪）
  多次接管 → 各自追加区间对
  停止录制 → save_episode → 弹窗【✓成功 / ✗失败 / 跳过】
            → AnnotationStore 写入区间 + episode_success（pending=false 直接生效）
            → 追加 meta/deployment_log.csv（干预次数/占比/成败）
            ↓
  visualize_data 打开 ela3_rollout：轨道可见（bad红/干预蓝），双击可改可删
            ↓ “生成训练 sidecar”
  recap_sidecar.jsonl → 训练（与 ela3_demo 在 dataloader 层混合，不物理合并）
```

### P2.1 遥操作录制：保存默认成功 + 来源溯源（已实现）

- `toggle_lerobot_record` 保存完成后：用 `AnnotationStore` 对刚保存的 episode
  写 `episode_success=True` + **`source="teleop"`**（静默，不弹窗，不拖慢采集）；
  推理录制对应写 `source="rollout"`——合并后训练按来源配权重（防遗忘混训，
  对应 P1.2 的"每轮必须混入 ela3_demo"）；空区间列表即"无干预"的正确编码，
  **不加任何占位区间**（干预帧在 P1.2 有 ×2 权重，占位会污染权重体系）；
- 录砸的示范在 visualize_data 用三态控件改判（人工 > 派生优先级已就绪）；
- 存量数据已回填：ela3_demo1→teleop、ela3_rollout→rollout（2026-08-29）。

### P2.2 "推理录制"按钮（已实现）

- **位置**：`btn_lerobot` 右侧 `btn_rollout_record`，toggle 行为同 LeRobot 录制按钮；
- **启动守卫**：仅当 `self._vla_running`（推理状态）才可开始；与遥操作录制**互斥**；
- **数据集**：固定 `ela3_rollout`（复用 `_ensure_lerobot_dataset` 创建/追加，同特征）；
- **录制线程**：复用 `LeRobotRecordThread`（lookahead 写入，action[t]=pose[t+N]，
  满足 P0.4 防线）；帧时钟 = 线程推帧计数（与数据集 frame_index 同序）；
- **联动**：停止 VLA 推理时若仍在推理录制 → 自动走停止保存流程。

### P2.3 接管自动标注（钩子：`toggle_vla_control_pause`，已实现）

- 推理录制中，控制暂停状态翻转时：
  - **转暂停（接管开始）**：记 `T = 当前推帧计数`；
  - **转继续（接管结束）**：记 `R`，追加区间对：
    - bad 窗口 `[max(0, T−20), T−1]`（label=bad，note="接管前20帧内"）——
      **决策二次修订恢复**：P1 阶段负优势的唯一自动来源（失败帧不标则按 0
      权重被正常模仿）；P3 阶段作 critic 监督样本。APO 实证（真实接管负样本
      是其性能提升关键）；窗口 `_rollout_pre_bad_frames` 可配（APO 原文 10 帧，
      默认 20，按效果调整）；与上一段干预重叠自动裁剪防冲突；
    - 干预区间 `[T, R−1]`（label=good，is_human_intervention=True，恒正——RECAP 规则）；
  - 录制开始时已处于接管 → T 从 0 起算；停止录制时仍在接管 → 按停止帧收尾；
- 区间暂存内存，保存时统一写入（**pending=false 直接生效，无确认制**）。

### P2.4 停止推理录制：成败评价对话框（已实现）

- 弹窗"本段推理录制任务最终成功了吗？"【✓ 成功 / ✗ 失败 / 跳过（不评）】；
- 成功/失败 → `episode_success=True/False`；跳过 → None（片段派生兜底）；
- 全部暂存区间 + 成败写入 `ela3_rollout/meta/reward_annotations.json`；
- 追加 `meta/deployment_log.csv`：时间戳、episode_index、帧数、干预次数、
  干预帧占比、成败——P5 迭代的北极星指标。

### P2.5 visualize_data 配套（基本完成）

- ✅ 自动写入的区间是普通区间：轨道显示（bad红/干预蓝）、双击编辑、删除、
  补标全部可用；`make_recap_sidecar.py` 直接展开；
- ✅ 存量批量标记：由 `convert_to_recap_format.py` 承担（ela3_demo 253 集
  已批量 success=True + source=teleop）；
- ⬜ RECAP 面板 episode 成败**三态覆盖控件**（未评/成功/失败）——误标纠正入口
  （低优先：当前可用录制端直标 + convert 脚本 + 手编 JSON 兜底）。

### P2.6 测试

- 无头单测：区间暂存（多次接管、接管中停止收尾）、弹窗三分支、
  遥操作默认成功写入——**已通过**；
- 真机联调（待做）：启动推理 → 开推理录制 → 失败 → 连接手柄救场 → 断开恢复 →
  二次接管 → 停止推理录制 → 标成功；检查标注 json（干预段+success，无 bad 区间）→
  GUI 改区间 → sidecar 分布验证；
- 对照：接管后仍失败 → 标失败 → 验证末 K 帧终端惩罚触发、干预段仍 +1。

---

## P3：价值函数 critic（1 周，可延后）

> 作用：把稀疏的人标（episode 成败 + 片段好坏）密集化为逐帧 advantage，
> 未标注帧不再"留 0 浪费"。完整 Recap 的关键组件。

两条路线（先 A 后 B）：

- **路线 A（轻量，先落地）**：冻结 pi0.5 主干提特征（图像+state 过 VLM 取末尾 hidden），
  训练一个小 MLP 价值头 V(s)∈[0,1]，监督信号 = episode_success（成败逐帧平铺）+
  片段标注加权。advantage = `V(s_{t+1}) − V(s_t)`，按阈值二值化回写 sidecar 的
  `advantage_indicator`（自动标注未标注帧）。
- **路线 B（借力 lerobot 内置）**：lerobot 0.6.1 已内置 SARM / TOPReward / Robometer
  奖励模型。SARM（子任务阶段感知）与本项目 `qwen3vl_subtask_annotate.py` 的子任务标注
  流程天然衔接（子任务进度 → 价值估计）。评估后择一接入。

**验收**：critic 在留出 episode 上的 V 曲线与人工标注的 good/bad 段一致性好
（bad 段 ΔV 显著为负）；自动标注结果抽 10% 人工复核，准确率可接受再用于训练。

---

## P4：advantage 条件化注入（2~3 天）

> Recap 的策略提取方式：训练时告诉模型每个动作的好坏，推理时恒命令"出高优势动作"。

- **pi0.5**：prompt 通道天然支持——lerobot pi05 预处理器把 state 离散化后拼文本
  （`Task: xxx, State: ...;\nAction: `）。训练时按样本的 `advantage_indicator` 在 prompt 注入
  条件文本（如 `Task: xxx, Advantage: positive, State: ...`；negative 同理）；
  推理时（`camera_pyqt.py`）恒用 positive。patch 点：pi05 preprocessor 的文本组装步骤
  （site-packages 补丁，登记进 TRAIN_README §4）。
- **SmolVLA**：task 文本前加前缀即可（`"advantage: positive, " + task`），smolvla_torque
  的自定义注册机制可直接复用做开关。
- 消融开关：训练 yaml 加 `recap.advantage_conditioning: false/true`，与 P1 的损失加权独立，
  便于做 2×2 对照（加权 × 条件化）。

**验收**：开启条件化训出的模型，在 positive 条件下推理行为与不注入时分布一致、
无退化；对照实验记录到附录。

---

## P5：完整 Recap 循环（长期节奏，非一次性任务）

```
每轮迭代（建议 1~2 周一轮；APO 的节奏参考：每 20 条 rollout 更新一次模型）：
  1. 用当前最优 checkpoint 部署（P2 录制开启）
  2. 收集 rollout + 干预段 → visualize_data 片段级标注（P0 工具 + P2.2 自动预标人工确认）
  3. 展开 sidecar（P1.1，含 episode_success 派生）→ [critic 自动补标未标注帧（P3，可选）]
  4. 数据质量门禁（P0.4）：action≠state 抽查 + 数据集检查第 6/7 项全绿
  5. 加权 SFT + advantage 条件化重训（P1+P4）
     —— 必须混入 ela3_demo 原示范数据（防遗忘，APO 教训，见附录 A）
  6. 离线评估（test_inference_episode.py 留出集）选 checkpoint → 上机
  7. 记录每轮成功率/干预率：干预率下降 = Recap 在起作用的直接证据
```

---

## MEM 路线（独立于 RECAP，P1 完成后即可并行启动）

> 对标 PI 2026-03 的 MEM（多尺度具身记忆，长时+短时记忆支撑 10 分钟以上任务）。
> 本路线先落地 MEM-lite（prompt 级），验证价值后再考虑真·记忆检索。

### M1：MEM-lite 接口（1~2 天）

- sidecar schema 加可选文本字段 `memory` / `next_memory`；
- 训练侧加可选 prompt 追加（参照 CBC `AppendMemoryToPrompt`，~20 行：
  `prompt = f"{prompt}\nMemory: {memory}"`），默认关闭；
- 部署侧（`camera_pyqt.py`）加 memory 构建钩子：初始实现从最近执行的子任务状态
  拼一句话（如 `completed: grasp box 1; last_failure: place slipped at frame 230`）。

### M2：memory 内容生成（2~3 天）

离线生成器：用 `qwen3vl_subtask_annotate.py` 的子任务分段结果 + RECAP 标注的失败段，
生成每段的 memory 文本（"已完成哪些子任务 / 哪里失败过 / 恢复方式"），写回 sidecar。
多任务长程数据集（如"清理桌面"= 多个子任务串联）是 MEM 的试验场。

### M3：对照评估（1 天）

同数据集训练 有/无 memory 两版，`test_inference_episode.py` 同 seed 对比误差 +
真机各跑 10 轮记成功率。**无提升则砍掉**（MEM-lite 对短任务大概率无增益，
它面向的是长程/需恢复的任务）。

### M4：远期真·MEM（暂不动工）

多尺度记忆（长期：历史 episode  embedding 检索；短期：当前任务状态机）需要记忆编码器
与检索机制，远超 prompt 追加。等 M3 证明记忆对本项目任务有价值、且任务复杂度
（>10 分钟长程）真的出现时再投入。届时重新调研 PI MEM 论文与 lerobot 后续版本
是否已有官方支持。

---

## 训练就绪清单（进入 P1.2 加权训练前的最后检查，2026-08-29 数据侧全部达标）

> 数据侧工具链已全部就绪，以下清单确认后即可启动 P1.2（训练侧加权）。

| # | 检查项 | 当前状态 | 操作 |
|---|---|---|---|
| 1 | 数据集统一 RECAP 格式（source/成败/区间） | ✅ ela3_demo/1/2/3 + rollout 全部就绪 | — |
| 2 | sidecar 已生成且不过期 | ✅ ela3_demo（150242 行）/ ela3_rollout（3721 行） | 每次剪辑/标注后重点"生成训练 sidecar" |
| 3 | "数据集检查" 8 项全绿 | ✅ ela3_demo2 / ela3_rollout 已验证 | 训练前对目标数据集跑一次 |
| 4 | action≠state + 前瞻健康 | ✅ 抖动剪辑后自动重算（k=5 实测吻合） | 检查问题5 |
| 5 | 训练数据选择 | ✅ **ela3_demo5 已就绪**（285 集/166335 帧 = 256 遥操作 + 29 rollout 含干预；sidecar 166335 行，bp=50/tf=10） | 全量微调用 `pi05_omy_fullft_recap.yaml` / `smolvla_omy_fullft_recap.yaml` |
| 6 | P1.2 代码 | ✅ **已实现（2026-08-30）** | 实现=lerobot 内置 sample_weighting 钩子（非自造 SidecarDataset 包装）：`RecapWeighter`（site-packages，type=recap）按 batch["index"] 全局帧号查 sidecar，零数据集包装、默认关闭；夹爪维豁免经 pi05/smolvla forward 补丁（recap_dim_scale）；单元+集成测试通过（权重映射/归一化/豁免/越界回退/preprocessor index 存活） |

**P1.2 启动规格回顾**（详见 P1.2 节 + 附录 A.3）：
- 权重映射：`positive_weight=1.0` / `negative_weight=0.1~0.3` / `intervention_weight=2~3`；
- 按 sidecar 的 `source` 字段区分专家/rollout 帧配权（防遗忘：每轮必混 ela3_demo）；
- `recap.enabled=false` 时训练行为与现状逐位一致（默认关闭原则）；
- 验收：SFT vs 加权 SFT 同 seed 对照，`test_inference_episode.py` 留出集比 mean action error。

---

## 风险与纪律

1. **默认关闭原则**：所有新训练开关（sidecar、加权、条件化、memory）默认 off，
   每次只开一个变量做对照（沿用 ACT 调参纪律）。
2. **sidecar 与数据一致性**：sidecar 以 `(episode_index, frame_index)` 为键，
   剪辑/合并后必须重新运行 `make_recap_sidecar.py`（P0.3 的警告会提醒）；
   原始 parquet 永远不被标注流程修改。
3. **与现有训练的兼容**：sidecar 字段不进 parquet、不进 `input_features`，
   不开启 recap 开关时训练输入与现状完全一致。
4. **数据规模**：每轮部署能净增的 on-policy 数据有限，RECAP 收益随轮数累积，
   不要用单轮结果下结论；以"干预率/成功率随轮数的趋势"为北极星指标。
5. **site-packages 补丁登记**：P1/P4 涉及 lerobot 补丁时，同步更新
   `TRAIN_README.md` §4 表格，服务器新环境按表重打。
6. **防遗忘**：每轮重训必须混入原示范数据（ela3_demo），权重/学习率保持保守；
   每轮重训后在留出集上回归测试原任务，成功率下降 >5% 即回滚上一轮 checkpoint
   （APO 对照实验中 BC 方法遗忘 9 个百分点的教训，附录 A.2）。

## 参考索引

| 资料 | 位置 |
|---|---|
| 本项目标注方案设计（片段级标注纪律） | `项目分析.md文件/数据清洗.md`（§8.2/§8.4） |
| 数据质量教训（问题 1~7，训练前门禁依据） | `项目分析.md文件/数据问题分析.md` |
| pi0.5 训练/推理/补丁 | `TRAIN_README.md` §4/§7 |
| CBC 复刻项目（sidecar/加权/MEM 模式来源） | `CBC_Pi0.7_Openpi-main/src/openpi/training/cbc_training.py`、`scripts/cbc_make_recap_sidecar.py`、`docs/cbc/training_time_rtc.md` |
| RECAP 论文 | arXiv:2511.14759（π\*0.6） |
| APO 论文（标注规则/防遗忘/终身学习节奏来源） | Human-assisted Robotic Policy Refinement via Action Preference Optimization，分析见附录 A |
| PI MEM 方向 | physicalintelligence.company/research/memory |
| lerobot 内置奖励模型 | VLA 环境 `lerobot/policies`（SARM/TOPReward/Robometer） |
| 存量转换 / 任务表巡检 | 本目录 `convert_to_recap_format.py`、`check_tasks_parquet.py`（--fix 清孤儿任务） |

## 附录 A：APO 论文分析结论（2026-08-29 录入）

> 文献：*Human-assisted Robotic Policy Refinement via Action Preference Optimization*（APO）。
> **本项目决策：路线 A**——数据规则借鉴 APO，训练侧沿用 P1（损失加权）+ P4（优势条件化），
> 不引入偏好优化目标。

### A.1 方法要点

- **标注规则**（附录 B 原文）：接管前**最后 10 个动作**（≈0.5s@20Hz，**不是 10 秒**）标
  undesirable；其余轨迹（含人工纠正段与接管前的策略帧）全部 desirable；
- **优化目标**：KTO 变体 v(o,â) = λ·σ(±(r_θ(o,â) − z₀))，z₀ = KL(π_θ‖π_ref)
  锚定基座防遗忘 + 自适应权重 λ_D/λ_U（需解码出连续动作计算）；
- **夹爪豁免**：reject reward 不作用于 gripper token（防止误惩接管前后相同的夹爪状态）。

### A.2 关键实验数字

| 实验 | Base | APO | 备注 |
|---|---|---|---|
| RoboMimic 4 任务均值 | 40.5% | **48.0%** | 优于 DAgger 39.0 / Sirius 39.5 / DPO 37.0 / TPO 41.5 / KTO 43.5 |
| 扰动泛化（位置/背景/纹理） | 21.3% | **28.0%** | 仅 20 条交互 + 20 条专家数据 |
| 原任务保持（扰动微调后） | 39.3% | **45.3%** | BC 方法遗忘至 30.7%，APO 反升 |
| π0-FAST 真机 Square | 85% | **95%** | 100 专家 + 20 干预轨迹 |
| 终身学习 | — | 成功率持续上升 | 每 20 条 rollout 更新一次，干预率同步下降 |

涌现行为：抓取失败自动重抓、插入受阻迭代微调位姿（与 π\*0.6 恢复行为同源，均来自干预纠正数据）。

### A.3 对本项目的适用性

- **根本障碍**：APO 仅在**自回归** VLA（OpenVLA、π0-FAST）上验证（论文 Discussion 明确承认），
  其隐含奖励 r = β·log(π_θ/π_ref) 需要动作对数似然；pi0.5 为 flow-matching 不可得
  （π\*0.6 同因放弃 PPO/DPO 类方法，改走优势条件化）。
- **已吸收的三条规则**：① 接管前 10 动作窗口（→ P2.2）；② 负样本夹爪维豁免（→ P1.2）；
  ③ 每轮混专家数据防遗忘（→ P1.2 / P5 / 风险纪律 6）。
- **不采用**：KTO/DPO 偏好目标。若未来换 π0-FAST 做 refinement 载体（lerobot 0.6.1 内置
  `pi0_fast`，log-prob 可得），再重新评估 APO 原版目标。

## 附录 B：实验记录（逐次追加）

| 日期 | 阶段 | 配置 | 留出集误差 / 真机成功率 | 结论 |
|---|---|---|---|---|
| 2026-08-30 | P1.2 实现（非训练） | RecapWeighter(type=recap) + pi05/smolvla forward 夹爪豁免 patch + ela3_demo5 sidecar(166335 行, bp=50/tf=10) | —（单元+集成测试全过：权重映射/归一化/豁免/钳位/越界回退/preprocessor index 存活/draccus 解析） | filtered BC 管线就绪；训练侧权重预览 mean=1.022（有效样本量 102.2%），负样本 1537 帧×0.1、干预 4974 帧×2.0 |
| ⬜ 待做 | P1.2 对照实验 | pi05_omy_fullft_recap.yaml vs 去掉 sample_weighting 段（同 seed=42） | `test_inference_episode.py` 留出集 mean action error + 真机各 10 轮 | 训练完成后填写；关注 train.log 无 NaN、loss 正常下降、真机干预率对比 |

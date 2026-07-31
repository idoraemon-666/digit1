# Protocol3 digit0 / digit8 闭环失真联合诊断设计

## 0. 文档状态与授权边界

本文件只属于 `digit_writing_original_protocol3`，用于把 digit0 和 digit8 的现有失败拆解为可复核的时序、空间、RNN 相位、正则梯度和动力学机制。

本文件已完成 **Stage A 实现、服务器执行与人工诊断**，并已形成 **Stage B 四臂同步训练实现草案**。Stage B 尚未运行本地 Python/测试或服务器训练，正式实现身份以提交后的 Protocol3 Git HEAD 冻结。当前不自动授权：

- 修改正式几何、时间表、loss、网络、batch 或工程阈值；
- 运行本地 Python、测试或训练；
- 代替用户执行服务器命令，或启动第二种子、自动改 geometry/loss、formal full10；
- 复用旧 `o3_digit8` LR 消融；
- 给起点、终点、闭环、自交点或其他特殊点增加权重。

Stage A 实现必须先提交到正式 Protocol3 分支并通过服务器测试；所有服务器命令仍只由用户执行。Stage A 是 0 optimizer-step 只读诊断；Stage B 包含 digit0/digit8 × 双 LR 四个独立同步训练臂，已在 Stage A 完成并人工复核后由用户单独授权设计。

## 1. 当前事实与实验目标

### 1.1 digit0

现有 v3 结果：

```text
source best@4800: mean=0.057405 endpoint=0.038148 path=0.855478
source final@6000: mean=0.043329 endpoint=0.118674 path=0.921021
lr1e3 final@8000: mean=0.038893 endpoint=0.091946 path=0.922522
lr3e4 final@8000: mean=0.046806 endpoint=0.090518 path=0.923684
```

6000 之前只有 3100、3700、4000、4800、4900 五个离散通过点；6000→8000 两臂均为 0 个通过点。整体 mean 和 path 可以改善，但 endpoint 同时退化。

### 1.2 digit8

现有 v3 结果：

```text
best@5800: mean=0.045602 endpoint=0.106399 path=0.809261
final@6000: mean=0.047249 endpoint=0.115620 path=0.843315
```

60 个训练后验证点中，mean 达标 32 次、endpoint 达标 1 次、path 达标 5 次，三项同时达标 0 次。轨迹两环均存在，但整体压缩且闭环终点偏离。

### 1.3 已排除项

现有证据已经排除：

- source geometry、derived path 或 temporal sampling 身份损坏；
- 缺段、切断、错误笔顺或错误方向；
- NaN、Inf、工作空间或关节触限；
- 明显的 excitation 上端饱和；
- digit0 仅仅因为 6000 updates 不足或 LR 0.001 过高；
- corner-ease 对 digit0/8 目标产生意外修改。两者均为单 primitive、0 sharp corners，v3 对其目标数组不做重参数化。

### 1.4 本实验只回答

1. movement 结束时是否只是尚未到达正确闭环相位；
2. 相位对齐后是否仍存在真实空间压缩；
3. RNN hidden state 是否能区分相同空间位置上的不同闭环相位；
4. position、hold 和 regularization 梯度是否形成持续冲突；
5. digit8 的高加速度/jerk 与自交相位负担是否构成主要困难；
6. digit8 是否能被与 digit0 完全匹配的 6000→8000 双 LR 续跑解释。

## 2. 总体结构

```text
Stage A：7 个现有 checkpoint 的联合 0-step deterministic 诊断
    ↓ 强制暂停并人工复核
Stage B：digit0 / digit8 的当前 v3 双 LR 四臂同步 6000→8000 匹配续跑
    ↓ 强制暂停并人工复核
可选 Plant Gate：只有 Stage A/B 仍无法区分 plant 可行性时另行设计和授权
```

Stage A 和 Stage B 使用独立配置、入口、输出目录、日志和归档。Stage A 完成不得自动触发 Stage B。

## 3. 冻结身份

### 3.1 Git 与模型族

```text
Protocol3 source repository HEAD: ee5a1900a33f8a8b7921fd9fc5b09aeed6600925
six-digit LR implementation HEAD: 76349d51222eed62bdb26d78d251b7d57760f2e8
mRNNTorch recorded/worktree HEAD: ac0c4f589eae37bbde63968912925de99232e306
scale_multiplier: 2.5
selected_reference_steps: 100
direction_index: 0
delay_steps: 50
seed: 42
batch_size: 8
network_noise: false during audit
environment_deterministic: true during audit
```

Stage A/B 的新实现 HEAD 必须在部署前另行提交并冻结；不得把未提交代码用于服务器正式诊断。

### 3.2 Stage A checkpoint 矩阵

| label | digit | checkpoint | source HEAD | SHA-256 |
|---|---:|---|---|---|
| `d0_source_best4800` | 0 | source best@4800 | `ee5a190` | `629e353d161281265e40d221cf5bcd30d2906db05b459f2a7be2b62efe81cff7` |
| `d0_source_final6000` | 0 | source final@6000 | `ee5a190` | `fd4c42c7a5ee546ccdc65615df2648379fdc1b36f36dbe0d918de5ed476ae908` |
| `d0_lr1e3_final8000` | 0 | LR 0.001 final@8000 | `76349d5` | `f075309b342c1ecd88c2fa65131ab03e029144dce24712fdf8f2fd505a6bdffc` |
| `d0_lr3e4_final8000` | 0 | LR 0.0003 final@8000 | `76349d5` | `3ec9099d25519add5bad3077b3f10c77c45f43ca4dc2052e0b0a427068823808` |
| `d8_source_best5800` | 8 | source best@5800 | `ee5a190` | `f70e0ba8e8585414f499f8ae2cc54e9061923983ea444ca62117e5faa5822a39` |
| `d8_source_final6000` | 8 | source final@6000 | `ee5a190` | `2812c1c336c14d8cb58bb02638592499d0bc018fb6628287d73e10a5dfa445ce` |
| `d9_positive_control_best4800` | 9 | source best@4800 | `ee5a190` | `4ad0674a167077a6a2935e6143bec9f723e6d9e07335e4a31edae5f2ed384bb9` |

digit9 只作为已稳定、路径长度相近且包含闭环片段的正对照，不训练、不参与任何选胜。

对应 source/continuation `run_summary.json` SHA 也必须冻结：

```text
digit0 source: 0df65f909782fab0fcd8a1fbcda90511720a451ce9112a18aebbd63070efc65a
digit8 source: 15df42d78db01d52a802f7dcf10c6a8661c342d04c6ade5e30e32fe61817a5b1
digit9 control: 8957fee9d878aa2d8099f953aa782b4102db64b726aa4e2a244cfed467602cfb
digit0 lr1e3: 136dfb8e8958f85d42b6bc5d3f0174de3c1806d1fa0d262d773487cd7c54008b
digit0 lr3e4: 6bfde34728b35757363860dc6762bd15c1536b92b6306a387a1626705145bf50
```

任何 SHA、case、Git、config、update 或 checkpoint metadata 不一致都属于 engineering failure，停止而不是替换来源。

## 4. Stage A：联合 0-step deterministic 诊断

### 4.1 实现边界

Stage A 当前实现文件：

```text
digit_writing/protocol3_closed_loop_diagnostic.py
configurations/digit_writing_original_protocol3_digit0_digit8_closed_loop_diagnostic.json
server/run_digit_writing_original_protocol3_digit0_digit8_closed_loop_diagnostic_parallel.sh
tests/test_protocol3_closed_loop_diagnostic.py
```

不得修改训练主循环。新模块复用：

- `train.load_digit_policy_checkpoint` 与 `_build_policy`；
- `validate_protocol3_resume_checkpoint` 与 `state_dict_sha256`；
- `protocol3_gate2.py` 的 deterministic rollout、安全、loss 和梯度实现语义；
- 当前正式 geometry/environment/config 分派。

现有 `protocol3_gate2.py` 已在 rollout 中产生 position、target、action、muscle、hidden 和 joint；新模块只扩展只读导出与分析，不复制训练循环。

### 4.2 rollout 数据

每个 checkpoint 使用 `testing=False`、batch 8、direction index 0、delay 50、speed condition 0、固定 `validation_seed` 和无噪声 deterministic observation。该口径与现有正式单条件 candidate audit 一致。所有指标先按 batch sample 独立计算，再报告 mean/min/max；不得先平均轨迹后再计算唯一指标。

逐时间点保存：

```text
episode_index
phase
movement_index
observation_pre_action
actual_x_m / actual_y_m
target_x_m / target_y_m
actual_speed_m_s / target_speed_m_s
actual_acceleration_m_s2 / target_acceleration_m_s2
actual_jerk_m_s3 / target_jerk_m_s3
actual_cumulative_path_m / target_cumulative_path_m
distance_to_closed_endpoint_m
six muscle excitations
muscle activations
joint positions
joint velocities
x recurrent state
h recurrent state
```

必须冻结两套时间轴，避免把现有训练 loss 的初始 hidden 错配到第一步位置：

- `aligned_step[0..T-1]`：每步送入 policy 的 `observation_pre_action`，policy 输出的 post-update `x/h/action`，以及该 action 后 environment 返回的 position/target/muscle/joint；这些数组长度均为 `T`，用于相位和空间诊断；
- `loss_hidden[0..T]` 与 `loss_muscle[0..T]`：保留训练语义中的全零初始 `h`/reset muscle，再追加每步值，长度为 `T+1`，仅用于精确复现 `l1_rate`、`simple_dynamics` 和 `l1_muscle_act`。

原始向量数据使用 `.npz`；摘要使用 CSV/JSON。输出不得包含无法由配置和 checkpoint 重建的隐式状态。测试必须证明上述两条时间轴不存在 off-by-one。

### 4.3 闭环到达诊断

目标 bbox diagonal 记为 `D`，沿用正式 endpoint 门槛：

```text
closure tolerance = 0.05 * D
```

主闭环标签只用于 digit0/8。设 movement 含 `N` 个 samples，只在

```text
late_movement_start = movement_start + floor(0.8 * N)
```

到 movement end 以及完整 hold 中搜索闭环。必须排除 stable、delay 和 movement 前 80%，因为 digit0/8 从闭合端点起笔；否则起始静止会把所有 case 误报为已经闭环。

计算：

```text
movement_endpoint_distance
minimum_late_movement_closure_distance
minimum_hold_closure_distance
first_closure_entry_episode_index
first_closure_entry_phase
hold_path_length_m
hold_final_displacement_from_movement_endpoint_m
hold_max_displacement_from_movement_endpoint_m
closure_reached_during_movement
closure_reached_only_during_hold
closure_never_reached
```

判定定义：

- late movement 内至少一次进入：`closure_reached_during_movement=true`；
- late movement 从未进入、hold 至少一次进入：`closure_reached_only_during_hold=true`，并标记 `late_closure`；
- late movement 与 hold 均从未进入：`closure_never_reached=true`，并标记 `persistent_nonclosure`。

上述三个到达类别必须恰有一个为真。`closure_hold_instability` 是独立稳定性标签：只要 late movement 已进入 tolerance，但 movement endpoint 已在 tolerance 外，或随后任一 hold sample 离开 tolerance，就标记为真；它可以与到达类别同时存在。

`first_closure_entry_*` 也只在上述 late-movement + hold 窗口内定义；没有进入时必须为 `null`。digit9 的正对照闭环是其 ellipse primitive 的结束 boundary，而不是整条 digit9 的 movement endpoint；它只报告以正式 primitive boundary 为中心的局部到达/状态指标，不获得 digit0/8 的闭环行为标签。

### 4.4 受限单调相位对齐

movement actual 为 `x[0..N-1]`，target 为 `y[0..N-1]`。使用动态规划求单调映射：

```text
j(0) = 0
j(t+1) - j(t) in {0, 1, 2}
abs(j(t) - t) <= 25
cost = sum(||x[t] - y[j(t)]||_2 / D)
movement end 不强制 j(N-1)=N-1
```

所有 DP 距离和累计 cost 使用 float64。主 cost 相等时依次以累计 `sum(abs(t-j(t)))` 更小、完整 `j` 序列字典序更小作确定性 tie-break。25 来自现有 hold_steps，不新增扫描参数。digit8 两个中心交叉相隔约 100 intervals，该 band 禁止从第一次交叉跳到第二次交叉以人为降低误差。

必须报告：

```text
time_aligned_normalized_mean_error
phase_aligned_normalized_mean_error
phase_alignment_error_reduction
phase_lag_steps_mean / p95 / final
target_phase_completion_at_movement_end
```

冻结符号：`phase_lag_steps(t) = t - j(t)`，正值表示 actual 落后于正式 target phase，负值表示超前。`target_phase_completion_at_movement_end = j(N-1)/(N-1)`。相位对齐是诊断，不参与 checkpoint 选择或行为 PASS/FAIL。

### 4.5 digit0 四象限分解

以 target bbox 中心为固定中心，不使用 actual 轨迹重新拟合中心。把 `atan2` 角归一到 `[-pi, pi)`，使用固定半开区间：

```text
right = [-pi/4, pi/4)
upper = [pi/4, 3*pi/4)
left  = [3*pi/4, pi) union [-pi, -3*pi/4)
lower = [-3*pi/4, -pi/4)
```

每个 movement sample 必须且只能落入一区。

每区报告：

```text
time_aligned_error
phase_aligned_error
actual/target path ratio
radial_extent_ratio
maximum_inward_deviation_m
mean_phase_lag_steps
```

同时报告 x/y bbox ratio，用于区分整体缩放和右半环方向性内缩。

### 4.6 digit8 两环和交叉点分解

对于 200 intervals / 201 samples，冻结 landmark indices：

```text
[0, 50, 100, 150, 200]
= [top start, center crossing 1, bottom, center crossing 2, top end]
```

训练前测试必须确认这些索引对应 authoritative Gerono target 的五个几何地标；不一致立即停止，不得改索引适配输出。

四个区间分别报告：

```text
segment path ratio
width/height extent ratio
time/phase-aligned error
entry/exit phase lag
```

路径量以四组 displacement intervals `[0,50)`、`[50,100)`、`[100,150)`、`[150,200)` 计算；逐点误差以 `[0,50)`、`[50,100)`、`[100,150)`、`[150,200]` 计算，确保 201 个 target samples 不重不漏。

两次中心交叉另外报告：

```text
crossing_position_error_m
crossing_arrival_lag_steps
actual_velocity_vs_target_tangent_cosine
outgoing_10_step_progress_ratio
wrong_branch_departure
```

对 target crossing `c in {50,150}`，在互不重叠的固定窗口 `[c-25,c+25]` 内选择使 `||actual[t]-target[c]||` 最小的 actual sample 作为 `actual_crossing_time`，相等时选更早时刻；因此两个 crossing 不可能互换。`crossing_arrival_lag_steps = actual_crossing_time-c`，正值表示迟到。`wrong_branch_departure` 仅在

```text
dot(actual[t+10]-actual[t], target[c+10]-target[c]) < 0
```

时为真，其中 `t=actual_crossing_time`；索引越界属于 engineering failure，不截断或改窗口。

### 4.7 hidden phase 诊断

同时保存和比较 policy 的 `x`、`h` 两套 recurrent state，不得只选其中一套。

hidden/action 与 position 使用 4.2 的 `aligned_step` 口径。digit8 两个 crossing 使用 4.6 的 `actual_crossing_time`；其他内部 target landmark `c` 在 `[max(0,c-25), min(N-1,c+25)]` 内按最小空间距离、再按更早时刻映射到 actual arrival time；movement start/end 和 first hold 使用其正式 episode index，不做最近点重映射。然后读取对应的 pre-action observation、post-update `x/h/action` 和 motor state。关键配对：

```text
digit0: movement start vs half-loop vs movement end; movement end vs first hold
digit8: crossing 1 vs crossing 2; top start vs top end; movement end vs first hold
digit9 control: ellipse start vs 正式 ellipse boundary; ellipse boundary vs tail end
```

每对报告：

```text
x/h cosine similarity
x/h normalized L2 distance
action cosine similarity
action L2 distance
pre-action observation cosine/L2 distance
joint position/velocity L2 distance
target tangent cosine
actual tangent cosine
```

本轮不拟合额外 phase decoder，也不设任意 hidden-distance 自动阈值。只有当目标切线明显不同，且在检查 pre-action observation 与 motor state 后，recurrent state 和 action 仍在多个 batch sample 与多个 checkpoint 中一致地近似重合，才在人工报告中标记 `phase_state_collision_supported`。digit0 起点/终点切线近似相同，本身不能单独支持 phase collision；该配对主要检验“启动”与“闭环后转入 hold”的状态区分。

### 4.8 分项梯度诊断

在同一 deterministic rollout 上分别计算：

```text
0.1 * stable position gradient
0.1 * delay position gradient
0.6 * movement position gradient
0.2 * hold position gradient
total position gradient
weighted l1_rate gradient
weighted l1_weight gradient
weighted l1_muscle_act gradient
weighted simple_dynamics gradient
total regularization gradient
total objective gradient
```

每项记录原始 loss、gradient norm，以及 movement/hold/total-position 与每个 regularizer/total-objective 的 cosine。

固定规则：

- phase 子项直接使用 `PHASE_WEIGHTS * *_mean_l1`，regularizer 使用正式已加权函数，所有 gradient 都是 clip 前原始参数梯度；
- `l1_rate` 与 `simple_dynamics` 必须使用 4.2 的 `loss_hidden[0..T]`，不得误用对齐后少一个初始状态的 hidden；
- norm `<1e-12` 时相关 cosine 记为 `undefined`；
- 记录原始分子、分母和 small-denominator 标记；
- 单个 checkpoint 的负 cosine 不是硬判据；
- 只有同一方向冲突在 best/final/continuation 多个 checkpoint 中重复出现，才支持“持续梯度冲突”；
- 审计前后 model state SHA 必须一致；
- `optimizer_steps_during_audit=0`。

### 4.9 目标与实际运动学

除全局 peak/p95 speed、acceleration、jerk 外，还必须按 digit0 象限和 digit8 四区间报告。与现有 Protocol3 Gate1 完全一致地使用一阶前向差分：`velocity=diff(position)/dt`、`acceleration=diff(velocity)/dt`、`jerk=diff(acceleration)/dt`，向量取 L2 norm 后计算 NumPy p95；不得改用中心差分或平滑。已知 target 参考值：

```text
digit0: path=0.425041m, duration=1.7s, p95 accel=1.425484, p95 jerk=8.234062
digit8: path=0.515537m, duration=2.0s, p95 accel=4.417187, p95 jerk=82.388109
```

digit8 是十数字中 target p95 acceleration 最高者。高 jerk 不能单独判定 plant 不可行，因为 digit4 也曾出现相近 jerk 并最终获得稳定候选；必须与 phase、hidden 和 actual tracking 共同解释。

### 4.10 Stage A 诊断标签

程序只计算事实，不自动宣布唯一根因。人工报告允许以下多标签：

```text
late_closure
persistent_nonclosure
closure_hold_instability
global_path_compression
directional_spatial_compression
phase_tracking_lag
phase_state_collision_supported
regularization_conflict_supported
dynamic_tracking_limit_supported
mixed_mechanism
```

以下使用已有工程阈值直接确定：closure tolerance、mean/endpoint/path PASS、workspace/joint safety。其余新机制不新增未经数据支持的硬数值阈值。

## 5. Stage B：当前 v3 digit0 / digit8 四臂同步匹配续跑

### 5.1 目的

Stage B 同步检验：digit0 和 digit8 各自是否能由训练长度或 LR 解释。它不修改目标，也不作为修复方案。

两个数字都从当前 v3 state-complete final@6000 恢复。四个臂在同一 runner、同一实现 HEAD 和同一服务器环境中并行启动：

```text
digit0 best checkpoint SHA: 629e353d161281265e40d221cf5bcd30d2906db05b459f2a7be2b62efe81cff7
digit0 final checkpoint SHA: fd4c42c7a5ee546ccdc65615df2648379fdc1b36f36dbe0d918de5ed476ae908
digit0 source status: PASS_UNSTABLE
digit8 best checkpoint SHA: f70e0ba8e8585414f499f8ae2cc54e9061923983ea444ca62117e5faa5822a39
digit8 final checkpoint SHA: 2812c1c336c14d8cb58bb02638592499d0bc018fb6628287d73e10a5dfa445ce
digit8 source status: FAIL
source repository HEAD: ee5a1900a33f8a8b7921fd9fc5b09aeed6600925
```

### 5.2 四臂

| digit | arm | learning rate | source | target |
|---:|---|---:|---:|---:|
| 0 | `lr1e3` | 0.001 | 6000 | 8000 |
| 0 | `lr3e4` | 0.0003 | 6000 | 8000 |
| 8 | `lr1e3` | 0.001 | 6000 | 8000 |
| 8 | `lr3e4` | 0.0003 | 6000 | 8000 |

冻结：

```text
additional_updates=2000
validation_interval=100
batch_size=8
seed=42
direction=0
delay=50
fixed_target_no_early_stop=true
synchronized_parallel_arms=4
```

每个数字的两臂必须从该数字同一 final@6000 恢复 model、Adam moments、condition counts、validation history 及 Python/NumPy/Torch RNG；恢复 optimizer 后只覆盖 param-group LR。source best 只用于维持 0→8000 三连选择历史，绝不作为恢复状态。

不得加入旧消融的 `1e-4`，不得停止在 7000，不得读取 `o3_digit8` checkpoint。

### 5.3 选择与解释

沿用完整 0→8000 历史的三连续通过规则。每个数字的两臂并列报告，四臂之间不自动选胜。

| 结果 | 允许的诊断解释 |
|---|---|
| 某数字两臂都 `STABLE_PASS` | 该数字的 6000 训练边界不足 |
| 仅 `lr1e3` 稳定 | 主要是训练长度，不支持降 LR |
| 仅 `lr3e4` 稳定 | 学习率/优化稳定性是重要因素 |
| 两臂继续压缩且 endpoint 失败 | 简单训练长度/LR解释被削弱，支持结构性 phase/loss/dynamics 机制 |
| 任一 engineering failure | 不得作行为结论 |

Stage B 完成后不得自动再续跑、增加第三 LR、第二种子、改变 reference 或启动 formal full10。

### 5.4 实现隔离

Stage B 当前实现：

```text
digit_writing/protocol3_corner_ease_lr_continuation.py
configurations/digit_writing_original_protocol3_digit0_digit8_matched_lr_continuation_6000_to8000.json
server/run_digit_writing_original_protocol3_digit0_digit8_matched_lr_continuation_parallel.sh
tests/test_protocol3_digit0_digit8_matched_lr_continuation.py
```

不得修改或复用旧 `protocol3_digit8_lr_ablation.py` 的 6000→7000/三臂身份。现有 continuation 引擎只增加严格配置驱动的 0/8 四臂路径，原六数字配置、数字范围、输出和决策边界保持不变；复用 checkpoint validation、正式训练 continuation、`select_validation_history` 和 best audit，不复制训练循环。

## 6. 输出隔离

### 6.1 Stage A

```text
runs/digit_writing_original_protocol3/
  scale2p50/ref100/corner_ease_v3/
    closed_loop_digit0_digit8_diagnostic/
      checkpoint_identity.json
      resolved_config.json
      per_checkpoint/<label>/
      temporal_alignment.csv
      closure_summary.csv
      shape_region_summary.csv
      hidden_phase_summary.json
      gradient_decomposition.json
      checkpoint_comparison.csv
      digit0_time_colored_overlay.png
      digit8_time_colored_overlay.png
      closure_distance_vs_time.png
      CLOSED_LOOP_DIAGNOSTIC_REPORT.md
      SHA256SUMS
```

每个 `per_checkpoint/<label>/` 至少包含 raw rollout `.npz`、逐时间 CSV、metrics JSON 和 overlay。

### 6.2 Stage B

```text
runs/digit_writing_original_protocol3/
  scale2p50/ref100/corner_ease_v3/
    digit0_digit8_matched_lr_continuation_from6000_to8000/
      digit0/lr1e3/
      digit0/lr3e4/
      digit8/lr1e3/
      digit8/lr3e4/
      digit0_digit8_lr_comparison.csv
      DIGIT0_DIGIT8_MATCHED_LR_CONTINUATION_REPORT.md
      SHA256SUMS
```

两个 stage 都必须有独立 `_server_evidence`、runner log、process exit codes 和外层归档 SHA，不写入或追加 source 目录。

## 7. 最小实现与测试

### 7.1 Stage A 必要测试

1. 7 个 checkpoint SHA、Git、case、update、config 精确匹配；
2. deterministic fixed seed 重复 rollout 逐元素一致；
3. exact-match synthetic trajectory 的相位对齐误差为 0；
4. 固定 10-step lag synthetic trajectory 恢复相同 lag；
5. 压缩但无 lag 的 synthetic trajectory不得被相位对齐伪装成 exact match；
6. digit8 两个交叉不得因 DP 跳转互换；
7. DP 约束严格为增量 `{0,1,2}` 和 band `25`；
8. digit8 landmark indices 与 authoritative target 一致；
9. digit0 四象限完整覆盖且不重不漏；
10. closure 搜索排除 stable/delay/前80% movement，三种标签边界正确；
11. digit9 使用正式 ellipse primitive boundary，不套用 digit0/8 闭环标签；
12. aligned step 为 `T`、loss hidden/muscle 为 `T+1`，x/h/action/observation/position 无 off-by-one；
13. 两个 crossing 的 local arrival window 不重叠、arrival lag 符号正确；
14. 分项 gradient 与原 total loss 加和语义一致；
15. 小梯度 cosine 为 `undefined`；
16. optimizer steps 为 0，审计前后 model SHA 相同；
17. 现有 Protocol3 Gate2、corner-ease 和 LR continuation 测试不变；
18. output 已存在时拒绝覆盖。

### 7.2 Stage B 必要测试

1. source 必须为当前 v3 digit0 和 digit8，不接受旧 continuation 或 `o3_digit8`；
2. best/final/summary SHA 精确匹配；
3. final@6000 恢复 model、optimizer、RNG、counts、history；
4. 每个数字的两臂只允许 LR 不同，四臂在同一 runner 并行启动；
5. validation updates 严格为 6100..8000；
6. 0..6000 history 逐元素保持；
7. target 精确停止 8000，无 early stop；
8. selector 覆盖 0..8000 三连语义；
9. source best 跨 6000 被正确保留但不作恢复状态；
10. 不启动 1e-4、第二种子、自动改 geometry/loss 或 formal full10；
11. output/归档已存在时拒绝覆盖；
12. 旧 digit8 消融与六数字 continuation 测试保持不变。

## 8. 服务器并行与门禁

Stage A 的 7 个 checkpoint audit 可在 7 个独立单线程 CPU 进程中并行，条件是：

```text
nproc >= 7
CPU-equivalent cores >= 7
每进程 OMP/MKL/OpenBLAS/NumExpr = 1
```

Stage B 四臂使用四个独立单线程进程并行启动，要求 `nproc >= 4` 且 CPU-equivalent cores >= 4。Stage A 与 Stage B 不并行，也不与其他训练共享同一输出或修改中的仓库。

统一启动前门禁：

- repository/submodule HEAD 正确且 clean；
- `device=cpu` compatibility marker 正确；
- source 和 source evidence 两套 `SHA256SUMS` 全通过；
- 输出、证据、日志和归档均不存在；
- 无重复进程；
- 指定测试全部通过、skipped 0；
- Stage A runner 不包含任何训练入口；
- Stage B runner 必须要求独立显式授权环境变量。

任一 worker 失败时保留现场、整体停止汇总，不覆盖重跑。

## 9. 可选 Plant Gate

Plant Gate 不属于当前最小实验。只有同时满足以下条件才值得另行设计：

1. Stage A 显示 hidden phase 可区分且正则梯度不构成主要冲突；
2. Stage B 中 digit0/digit8 各自两臂均继续以相同形态失败；
3. 错误集中在 digit8 高加速度/jerk区域或 digit0固定方向区间；
4. 用户明确授权独立 MotorNet 可行性诊断。

不得把 Plant Gate 隐藏在 Stage A 测试或 Stage B 训练中。

## 10. 完成判据与人工结论

Stage A 完成必须回答：

1. digit0/8 的闭环是在 movement 内完成、hold 中迟到完成，还是始终未完成；
2. 相位对齐能解释多少 time-aligned error；
3. digit0 是整体缩放还是右半环方向性内缩；
4. digit8 哪个半环/交叉点贡献主要误差；
5. recurrent state 是否区分重复空间位置；
6. position、hold 与 regularization 是否存在跨checkpoint一致冲突；
7. 与 digit9 正对照相比，闭环状态和路径完成差异在哪里；
8. 是否有充分依据进入 Stage B。

Stage B 完成必须回答：

1. digit0 和 digit8 四臂的完整状态、best update和三项行为指标；
2. 6000→8000 每臂 20 个验证点及稳定区间；
3. 四臂 final checkpoint/metrics 和 best overlay；
4. 对每个数字，训练长度/LR解释是否成立；
5. 是否仍需 Plant Gate。

允许的最终结论仅为：

```text
temporal/phase dominant
spatial compression dominant
regularization conflict supported
dynamic tracking limit supported
mixed mechanism
evidence insufficient
```

不得在本实验中自动提出或执行正式修复。任何 endpoint/path/phase cue、时间表、loss 或结构修改都必须基于上述证据另行设计和授权。

## 11. Codex 后续实施顺序

```text
1. Stage A 独立只读模块、配置、测试和服务器执行已完成
2. Stage A 归档已完成本地只读验收和人工复核
3. 用户已明确授权设计 digit0/digit8 Stage B 同步训练
4. 实现并静态审查 Stage B 四臂配置、复用引擎、测试和并行 runner
5. 提交并推送正式 Protocol3 Stage B implementation HEAD
6. 用户通过 GitHub 在服务器启动四臂训练
7. 下载并人工比较 digit0/8 的匹配 LR 证据
8. 强制暂停，不自动进入修复、第二种子、formal full10 或 Plant Gate
```

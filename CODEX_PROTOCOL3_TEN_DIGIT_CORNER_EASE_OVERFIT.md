# Codex 指导文档：Protocol3 十数字局部减速单任务过拟合

## 0. 任务目标

本次只完成一项工作：

> 在当前 Protocol3 `scale=2.5`、medium 固定片段时间表基础上，对尖角连接处实施“连续减速—近零速度转向—连续加速”的局部时间重参数化，然后对数字 0–9 分别进行单数字、单方向、单 reference、单 delay 过拟合训练，尽快获得可清楚辨识的数字轨迹。

本次不是正式 full10，不验证多任务联合学习或组合机制。

本次结果只用于证明：

1. 新目标轨迹在 RNN–MotorNet 闭环中可以被单任务模型拟合；
2. 哪些数字已经得到稳定、清楚、可展示的轨迹；
3. 局部减速是否比原 medium baseline 更适合尖角数字。

---

# 1. 严格边界

## 1.1 允许做的工作

只允许：

1. 新增一个可选的尖角局部时间重参数化模式；
2. 保持空间几何不变，只改变目标点沿原路径前进的时间密度；
3. 对数字 0–9 分别训练 10 个独立单任务模型；
4. 生成必要的目标审计、行为指标、轨迹图、checkpoint 和汇总报告；
5. 对已有 medium baseline 结果做只读比较。

## 1.2 禁止事项

不得：

- 修改当前正在运行或已经完成的 Gate 2、corner-settle 或旧 full10；
- 使用任何旧 checkpoint 初始化本次模型；
- 使用 corner settle、重复连接点或物理停顿；
- 圆角化、裁剪、移动或重新设计空间轨迹；
- 修改数字尺度；
- 修改非尖角任务的时间步数；
- 全局增加所有数字的 movement intervals；
- 修改 RNN、MotorNet、28 维 observation 或 6 维输出；
- 修改 phase-normalized position loss 或四个正式正则项；
- 增加 endpoint、corner、velocity、acceleration 或 jerk loss；
- 修改 Adam、学习率或梯度裁剪；
- 训练多方向、多 delay 或多任务模型；
- 自动延长至 10,000 updates；
- 自动补第二个 seed；
- 运行 MotorNet 直接控制优化、Stage E、Stage F；
- 启动 formal full10、heldout5、composition 或 transfer5；
- 产生与本任务无关的图表、消融或分析。

完成本文件规定的 10 个模型和报告后必须停止。

---

# 2. 实验身份与隔离

## 2.1 基础身份

必须读取当前已完成的 Protocol3 medium Gate 配置和代码身份，并冻结：

```text
protocol                  = digit_writing_original_protocol3
scale_multiplier          = 2.5
timing_mode_base          = fixed_segment_timing
selected_reference_steps  = 100
delay_steps               = 50
direction                 = 0
seed                      = 42
batch_size                = 8
max_updates               = 6000
evaluation_interval       = 100
dt                        = 0.01 s
```

mRNNTorch 必须仍为：

```text
ac0c4f589eae37bbde63968912925de99232e306
```

除本文件明确允许的局部时间规则外，必须与当前 medium 单任务诊断配置一致。

## 2.2 独立运行身份

建议 timing mode 名称：

```text
fixed_segment_timing_corner_ease_v3
```

建议输出根目录：

```text
runs/digit_writing_original_protocol3/
scale2p50/ref100/corner_ease_v3/single_digit_overfit/
```

每个数字独立目录：

```text
digit0/seed42/
...
digit9/seed42/
```

不得写入任何 baseline、settle、Gate 2 或 full10 目录。

每个 run label 至少包含：

```text
protocol3
scale2p50
ref100
corner-ease-v3
digit
direction0
delay50
seed42
Git short SHA
```

本地只允许代码修改、静态审查、Git 核对和可审计提交；不得在本地运行 Python、
测试、smoke 或训练。全部测试、预检和训练只能由用户在服务器执行。

---

# 3. 冻结的 medium 基础时间表

本次不重新讨论或修改基础时间表。

| timing key | medium 基础 intervals |
|---|---:|
| `line` | 60 |
| `curve_A` | 80 |
| `curve_B` | 90 |
| `ellipse_5_4` | 140 |
| `ellipse_4_3` | 170 |
| `independent_uji8` | 200 |

不含局部减速时，完整数字的 medium movement intervals：

| 数字 | 基础 intervals |
|---|---:|
| 0 | 170 |
| 1 | 60 |
| 2 | 200 |
| 3 | 170 |
| 4 | 180 |
| 5 | 210 |
| 6 | 200 |
| 7 | 120 |
| 8 | 200 |
| 9 | 200 |

本次没有依据认为所有平滑任务仍普遍时间不足，因此：

> 不对全部任务统一加时，只对大角度连接处局部增加时间。

---

# 4. 需要局部减速的尖角

## 4.1 自动识别规则

在每个内部 segment boundary 计算有序目标轨迹的转向角：

```python
turn_angle_deg = degrees(
    arccos(
        clip(
            dot(v_pre_unit, v_post_unit),
            -1.0,
            1.0,
        )
    )
)
```

其中：

- `v_pre` 是前一片段末端最后一个有效非零切向量；
- `v_post` 是后一片段起始第一个有效非零切向量；
- 必须跳过零长度差分；
- 只处理内部连接点；
- 阈值固定为：

```text
turn_angle_deg >= 60°
```

## 4.2 预期尖角清单

代码审计结果必须与以下清单一致：

| 数字 | 连接关系 | 预期转向角 | 尖角数 |
|---|---|---:|---:|
| 0 | 无 | — | 0 |
| 1 | 无 | — | 0 |
| 2 | diagonal → horizontal | 约 125° | 1 |
| 3 | curve_A → curve_B | 约 179.24° | 1 |
| 4 | segment 0 → 1；segment 1 → 2 | 约 125°；145° | 2 |
| 5 | segment 0 → 1；segment 1 → 2 | 约 90°；89.69° | 2 |
| 6 | 无合格尖角 | — | 0 |
| 7 | segment 0 → 1 | 约 116.57° | 1 |
| 8 | 无 | — | 0 |
| 9 | 无合格尖角 | — | 0 |

digit 2 的 `curve_A → diagonal` 连接角约 0.45°，不得错误加入局部减速。

若自动审计得到不同尖角数量或连接关系，停止，不得开始训练。

---

# 5. 局部时间重参数化的唯一实现

## 5.1 核心原则

每个尖角：

- 前一片段末端逐渐减速；
- 尖角处离散目标速度接近零；
- 后一片段起点逐渐加速；
- 连接点只出现一次；
- 没有重复点；
- 没有停顿；
- 没有倒退；
- 空间轨迹、端点和连接点完全不变。

每个尖角总共增加：

```text
10 movement intervals
```

分配为：

```text
前一片段末端 +5 intervals
后一片段起点 +5 intervals
```

## 5.2 固定空间窗口

对每个尖角相邻片段，使用原基础采样中的：

```text
10 个基础 intervals
```

作为局部过渡空间窗口。

因此：

```text
BASE_WINDOW_INTERVALS = 10
EXTRA_INTERVALS_PER_SIDE = 5
RESAMPLED_WINDOW_INTERVALS = 15
```

前一片段使用最后 10 个基础 intervals 对应的弧长范围。

后一片段使用最前 10 个基础 intervals 对应的弧长范围。

不允许根据数字或转向角单独调整窗口和额外步数。

## 5.3 v3 minimax 固定离散时间映射

v1 五次映射已在服务器通过 92 项代码测试，但在训练前运动学硬门禁中使以下局部
`p95 acceleration` 上升：digit 4 boundary 0 为 4.98%，digit 5 两个 boundary
分别为 23.39% 和 22.70%，digit 7 boundary 0 为 19.29%。所有训练均未启动。

v2 固定有理数映射随后在服务器运行 93 项测试，其中只有运动学硬门禁的 digit 5
两个边界失败：boundary 0 为 `1.71869193608 > 1.70354900845`（高 0.8889%），
boundary 1 为 `2.56713751180 > 2.55800743737`（高 0.3569%）。训练仍未启动。

经用户明确授权，v3 只替换时间步长分配，不改变第 5.2 节空间窗口、总 intervals、
尖角阈值、几何、hash、训练条件或运动学门禁。不得把 v1 或 v2 结果伪装成 v3
结果。

终点减速窗口的 15 个正弧长步长，以一个基础 interval 为单位，固定为：

```text
denominator = 20
numerators  = [
  20, 20, 20, 20, 20, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1
]
```

必须满足：

```text
sum(step_weights) = 10
step_weights > 0
step_weights 单调不增
last_step / baseline_step = 1 / 20 = 0.05
max(abs(diff([1.0] + step_weights))) = 0.10
```

在上述冻结约束下，`0.10` 是最大相邻步长变化的最小可达值：从末步 0.05
反向按每步最多 0.10 增长并在 1.0 截断时，15 步总和最多恰为 10；任何更小
上限都会使总和小于 10。v3 采用达到该下界的固定表，不进行参数扫描。

起点加速窗口严格使用上述序列的时间反向。累计弧长比例由步长序列直接累加并除以
10 得到，必须严格单调。不得按数字、转向角或服务器结果调整此表。

## 5.4 单片段采样规则

设一个片段原基础 intervals 为 `N`。

### 无尖角端点

使用原来的均匀弧长采样，不改变目标。

### 只有起点是尖角

- 起点窗口空间比例：
  ```text
  r_start = 10 / N
  ```
- 起点窗口用 15 intervals 采样：
  ```python
  u = r_start * cumulative(reverse(step_weights))[k] / 10
  ```
- 剩余空间 `[r_start, 1]` 使用原剩余 `N-10` intervals 均匀采样；
- 新总 intervals：
  ```text
  N + 5
  ```

### 只有终点是尖角

- 终点窗口空间比例：
  ```text
  r_end = 10 / N
  ```
- 前段 `[0, 1-r_end]` 使用原 `N-10` intervals 均匀采样；
- 终点窗口用 15 intervals：
  ```python
  u = (1-r_end) + r_end * cumulative(step_weights)[k] / 10
  ```
- 新总 intervals：
  ```text
  N + 5
  ```

### 起点和终点都是尖角

- 起点窗口：原 10 intervals → 15 intervals；
- 中间空间：原 `N-20` intervals，保持均匀；
- 终点窗口：原 10 intervals → 15 intervals；
- 新总 intervals：
  ```text
  N + 10
  ```

必须满足：

```text
N >= 2*BASE_WINDOW_INTERVALS + 2
```

若不满足，停止并报告，不得缩小窗口或自行修改规则。

## 5.5 拼接规则

片段内部三个子区间拼接时：

- 子区间公共边界只保留一次。

完整数字拼接时：

- 相邻片段公共连接点只保留一次；
- 下一片段的首点不得再次加入；
- 尖角处不得出现两个或更多连续相同坐标。

完整轨迹始终满足：

```text
movement_samples = movement_intervals + 1
```

---

# 6. 新的预期完整时长

每个尖角增加 10 intervals，因此预期结果：

| 数字 | 基础 intervals | 尖角数 | 新 intervals |
|---|---:|---:|---:|
| 0 | 170 | 0 | 170 |
| 1 | 60 | 0 | 60 |
| 2 | 200 | 1 | 210 |
| 3 | 170 | 1 | 180 |
| 4 | 180 | 2 | 200 |
| 5 | 210 | 2 | 230 |
| 6 | 200 | 0 | 200 |
| 7 | 120 | 1 | 130 |
| 8 | 200 | 0 | 200 |
| 9 | 200 | 0 | 200 |

如果代码生成结果与此表不一致，停止，不得训练。

---

# 7. 空间几何与共享片段边界

## 7.1 空间几何不变

局部减速模式不得改变：

- design-unit 几何模板；
- 米制全局尺度；
- 起点、终点和连接点；
- 轨迹笔顺；
- 包围盒；
- 工作空间方向；
- segment 的空间支持路径。

每个新采样点必须位于对应原始高分辨率 segment polyline 上，允许的数值距离：

```text
<= 1e-10 m
```

## 7.2 共享片段解释

几何身份拆分为三个不可混用的层级：

```text
source_geometry_sha256
derived_path_geometry_sha256
temporal_sampling_sha256
```

- `source_geometry_sha256`：原始设计 primitive、锚点或控制参数的规范化身份；原始
  设计未变时必须保持不变；
- `derived_path_geometry_sha256`：完成圆润化或其他允许的空间派生后、时间采样前的
  连续有序路径身份；若未来横折、竖钩等发生空间圆润化，本项必须变化，而 source
  hash 保持不变；
- `temporal_sampling_sha256`：最终 intervals 和时间映射作用后的离散有序点数组身份；
  intervals 或采样密度变化后本项必须变化。

当前 corner-ease 不做空间圆润化，因此相对 medium baseline：

- `source_geometry_sha256` 必须相同；
- `derived_path_geometry_sha256` 必须相同；
- 只有被局部重参数化的 segment，其 `temporal_sampling_sha256` 必须变化；
- 非尖角数字三层身份和目标数组均必须保持一致。

旧 `canonical_template_sha256` 已包含按 intervals 重采样后的点数组，必须保留原有
语义，不得重新定义以制造一致。legacy timing 路径继续核对旧值；新 corner-ease
协议不要求 `canonical_template_sha256` 与 baseline 相同。

局部减速会改变某些共享片段端部的有序时间采样，因此：

- 不得要求完整 `ordered_instance_sha256` 与 baseline 相同；
- 后续共享活动比较应排除局部减速窗口；
- 共享片段可比较部分定义为：

```text
shared_core = 原共享片段去除所有与尖角相邻的 10 个基础 interval 空间窗口
```

本次只做行为过拟合，不开展网络活动比较。

---

# 8. 训练前必须通过的最小审计

只执行以下必要审计，不开展额外实验。

## 8.1 目标时序审计

生成：

```text
corner_ease_manifest.csv
target_timing_summary.csv
target_kinematics_comparison.csv
```

`corner_ease_manifest.csv` 至少包含：

```text
digit
boundary_index
previous_segment
next_segment
turn_angle_deg
base_window_intervals
extra_intervals_before
extra_intervals_after
corner_sample_index
incoming_step_m
outgoing_step_m
regular_step_median_m
incoming_speed_ratio
outgoing_speed_ratio
```

## 8.2 硬性检查

全部满足才可训练：

1. 尖角清单与第 4.2 节一致；
2. 完整 intervals 与第 6 节一致；
3. 非尖角数字 0、1、6、8、9 的目标数组与当前 medium baseline 逐点一致；
4. 所有端点和连接点与 baseline 一致；
5. 连接点只出现一次；
6. 所有相邻目标点距离严格大于 0；
7. 累计弧长严格单调，无倒退；
8. 所有新点位于原 segment polyline 上；
9. 新旧目标包围盒一致，误差不超过 `1e-10 m`；
10. `source_geometry_sha256` 和 `derived_path_geometry_sha256` 与 baseline 一致；
    `temporal_sampling_sha256` 只在实际重参数化的 segment 变化；legacy
    `canonical_template_sha256` 保留旧语义但新协议不要求与 baseline 相同；
11. 28 维 observation 不变；
12. phase-normalized loss 和正则项不变；
13. 无 NaN/Inf；
14. `movement_samples = movement_intervals + 1`。

## 8.3 近零速度判据

对每个尖角，定义：

```text
incoming_speed_ratio =
尖角前最后一步位移 / 邻近正常区域步长中位数

outgoing_speed_ratio =
尖角后第一步位移 / 邻近正常区域步长中位数
```

必须满足：

```text
incoming_speed_ratio <= 0.10
outgoing_speed_ratio <= 0.10
```

同时禁止：

```text
incoming_step_m == 0
outgoing_step_m == 0
```

即速度接近零，但不是停顿。

## 8.4 运动学改善判据

对每个尖角使用相同空间范围，比较新规则与 medium baseline 的：

```text
local_p95_speed
local_p95_acceleration
local_p95_jerk
local_peak_acceleration
local_peak_jerk
```

必须至少满足：

```text
local_p95_acceleration_new <= local_p95_acceleration_baseline
local_p95_jerk_new         <= local_p95_jerk_baseline
```

若任一尖角不满足，停止，不得训练或调整多组参数。

---

# 9. 十个独立过拟合模型

## 9.1 固定训练条件

分别训练：

```text
digit 0
digit 1
...
digit 9
```

每个模型：

```text
单数字
direction = 0
selected_reference_steps = 100
delay = 50
seed = 42
batch_size = 8
max_updates = 6000
evaluation_interval = 100
```

所有模型从零初始化。

不得共享：

- model state；
- optimizer state；
- hidden state；
- checkpoint；
- RNG continuation state。

允许并行运行，但每个模型必须有独立进程和输出目录。

## 9.2 训练和评估规则

训练：

- 保持当前 medium Gate 的 network/environment noise 规则；
- 完成整个 trial 后计算一次 loss；
- 每 update 只进行一次 optimizer step。

确定性评估：

```text
network_noise = false
environment deterministic = true
direction = 0
delay = 50
```

每 100 updates 评估一次。

固定运行至 6000 updates：

- 不提前停止；
- 不自动延长；
- 不重新初始化；
- 不补 seed。

---

# 10. 指标与 checkpoint 选择

## 10.1 必须记录的行为指标

仅记录必要指标：

```text
normalized_mean_movement_error
normalized_endpoint_error
target_path_length_m
actual_path_length_m
path_length_ratio
target_bbox_diagonal_m
```

安全指标：

```text
min_inner_radius_margin_m
min_outer_radius_margin_m
min_joint_margin_rad
muscle_activation_max
muscle_activation_fraction_ge_0_99
muscle_excitation_fraction_le_0_01
muscle_excitation_fraction_ge_0_99
has_nan_or_inf
```

## 10.2 通过阈值

沿用当前单任务工程门槛：

```text
normalized_mean_movement_error <= 0.08
normalized_endpoint_error      <= 0.05
0.85 <= path_length_ratio <= 1.15
```

这些是工程展示门槛，不宣称来自原论文。

## 10.3 稳定状态定义

连续 3 个评估点全部通过，定义为：

```text
STABLE_PASS
```

有通过点但从未连续 3 次通过：

```text
PASS_UNSTABLE
```

从未通过：

```text
FAIL
```

## 10.4 best checkpoint 的唯一选择规则

按以下顺序：

1. 若存在 `STABLE_PASS` 区间：
   - 仅在所有长度至少为 3 的连续通过区间中选择；
   - 选择 `normalized_mean_movement_error` 最低的 checkpoint；
   - 精确同分时选择较早 update。
2. 若无稳定区间，但存在单点通过：
   - 在所有通过点中选择 mean error 最低者；
   - 标记 `PASS_UNSTABLE`。
3. 若无任何通过点：
   - 选择 mean error 最低者用于诊断图；
   - 状态必须为 `FAIL`，不得宣称成功。

不得根据人工观看图形事后更换 checkpoint。

---

# 11. 必须产生的最小结果

每个数字只保存：

```text
resolved_config.json
metrics.jsonl
best_checkpoint.pt
final_checkpoint.pt
best_movement_overlay.png
run_summary.json
run.log
```

全局只生成：

```text
corner_ease_manifest.csv
target_timing_summary.csv
target_kinematics_comparison.csv
ten_digit_metrics.csv
ten_digit_best_overlays.png
displayable_digits_overlays.png
TEN_DIGIT_CORNER_EASE_OVERFIT_REPORT.md
SHA256SUMS
```

## 11.1 `ten_digit_best_overlays.png`

固定为 2×5 布局，展示数字 0–9 的 best target/actual movement-only overlay。

每个子图标明：

```text
digit
best update
STABLE_PASS / PASS_UNSTABLE / FAIL
mean
endpoint
path ratio
```

坐标轴比例必须相等，不得自动拉伸使失真轨迹看起来正常。

## 11.2 `displayable_digits_overlays.png`

只纳入：

```text
STABLE_PASS
```

的数字。

不得把 `PASS_UNSTABLE` 或 `FAIL` 混入“可展示结果”。

如果没有稳定通过的数字，仍生成空白说明图并明确报告，不得挑选失败模型冒充成功。

## 11.3 报告必须回答

只回答：

1. 十个数字各自是否 `STABLE_PASS / PASS_UNSTABLE / FAIL`；
2. 哪些数字已经具有清楚、稳定、可展示的轨迹；
3. 尖角数字 2、3、4、5、7 相比已有 medium baseline：
   - mean error 是否改善；
   - endpoint 是否改善；
   - path ratio 是否更接近 1；
   - 是否减少切角、漏段、整体压缩或过冲；
4. 是否出现新的安全、饱和或数值问题；
5. 本目标是否值得进入后续 full10 候选。

报告不得：

- 宣称多任务学习成功；
- 宣称组合机制成功；
- 建议自动启动 full10；
- 建议新的几何、loss、停顿或参数扫描。

---

# 12. 复用已有 baseline 的规则

对于 digit 2、3、4、5、7，可只读引用已完成的 medium baseline 6000-update 结果，前提是以下身份完全一致：

```text
scale = 2.5
selected_reference_steps = 100
direction = 0
delay = 50
seed = 42
batch_size = 8
max_updates = 6000
network/loss/optimizer 相同
```

不得重新训练 baseline。

若任何身份不一致，只在报告中注明“不可严格比较”，也不得补跑 baseline。

数字 0、1、6、8、9 不需要额外 baseline 对照，因为局部减速模式不会改变它们的目标数组；第 8.2 节必须证明逐点一致。

---

# 13. 最小代码改动原则

优先复用当前 Protocol3 Gate 的：

- geometry configuration；
- segment arc-length sampling；
- 单任务训练 runner；
- deterministic evaluator；
- checkpoint 和指标输出。

建议只新增一个小型纯函数模块，例如：

```text
digit_writing/corner_time_reparameterization.py
```

其中只包含：

```text
corner_ease_step_weights_v3
detect_sharp_boundaries
resample_segment_with_corner_easing
```

共享 geometry 入口只增加一个明确的 timing-mode dispatch：

```text
fixed_segment_timing
fixed_segment_timing_corner_ease_v3
```

默认模式和所有旧配置行为必须保持不变。

不得复制一套新的训练循环。

---

# 14. 必要测试

训练前至少通过：

1. v3 固定分子、分母、严格正步长、总弧长、末步 0.05 和 minimax 0.10；
2. 减速与时间反向加速映射严格单调；
3. `corner_ease` 关闭时目标与 baseline 逐点一致；
4. 非尖角数字目标逐点不变；
5. 尖角清单与冻结清单一致；
6. 每角恰好增加 10 intervals；
7. 新完整时长与冻结表一致；
8. 无重复点和零位移；
9. 无弧长倒退；
10. 尖角点只出现一次；
11. 近零速度比例不超过 0.10；
12. 空间端点、连接点和包围盒不变；
13. 三层 geometry hash 按第 7.2 节分离；legacy canonical template hash 语义不变；
14. 28 维 observation 不变；
15. loss、regularization、optimizer 不变；
16. 固定 seed 下目标和评估完全复现；
17. smoke rollout 无 NaN/Inf 或安全失败；
18. protocol2、现有 Protocol3 baseline 和 settle 模式回归测试不变。

不得通过放宽断言来使测试通过。

---

# 15. Codex 执行顺序

严格按以下顺序：

```text
1. 读取当前 Protocol3 medium 单任务 resolved config 和 Git 身份
2. 确认现有运行与输出目录只读
3. 新增局部时间重参数化纯函数
4. 增加 timing-mode dispatch，不改变默认行为
5. 添加最小单元和回归测试
6. 生成尖角、时长和运动学审计
7. 审计全部通过后，训练 digit 0–9 十个独立模型
8. 固定运行至 6000 updates
9. 按预定义规则选择 best checkpoint
10. 生成 10 张独立 overlay、两个汇总面板和一份报告
11. 生成 SHA256SUMS
12. 停止，等待用户决定
```

任何审计硬条件失败时：

```text
立即停止
保留证据
不得自动修改窗口、额外步数、阈值或时间表
```

---

# 16. Codex 最终回复格式

只汇报：

```text
A. Git / mRNNTorch / 基础配置身份
B. 实际修改文件
C. 回归测试结果
D. 尖角清单与新完整时长
E. 近零速度和运动学审计结果
F. 10 个 run 是否全部正常结束
G. 每个数字的状态、best update 和三项行为指标
H. STABLE_PASS 数字列表
I. 两张汇总图路径
J. 与已有 medium baseline 的尖角数字对比结论
K. 安全和数值结果
L. 结果目录和 SHA-256 清单
```

不得在最终回复中扩大工作范围或启动其他实验。

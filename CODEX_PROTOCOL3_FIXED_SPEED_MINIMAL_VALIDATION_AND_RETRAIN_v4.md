# Codex 指导文档：Protocol3 固定片段时间版的最小验证与正式重训（独立随机采样）

## 0. 任务目标

在不改写既有正式提交和结果的前提下，为数字书写任务建立新的科学协议版本：

1. 所有数字几何统一放大 `2.5×`；
2. 不再按“路径长度 ÷ 固定物理速度”自动确定时间；
3. 改为按**片段类型**固定 movement intervals；
4. 同一严格共享片段在所有数字中使用相同空间模板、相同采样顺序和相同步数；
5. 训练保留 8 个空间方向，正式验证保留 32 个方向，并取消多时间条件联合训练；
6. 每个 update 独立等概率抽取数字，并独立随机抽取 delay；
7. 先验证 fast 固定片段时间条件；3,000 updates 尚未通过时先检查并按次人工授权续跑；fast 最终仍未通过且已排除工程、数据流和安全问题时，只允许整体回退一次到 medium；
8. Gate 2 后先冻结最终工程验收阈值，再从零启动 full10；训练必须在 5,000 updates 强制暂停并等待用户人工批准。

核心目标：

> 尽快得到能够高质量拟合数字轨迹、且适合后续共享片段活动比较和组合机制验证的基础模型。

---

# 1. 不可违反的边界

## 1.1 固定身份只读

不得改写：

- 原始基线：`105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33`
- 已完成正式科学代码：`f76067232fac8add756b04ea75a8825cca156330`
- 固定 mRNNTorch：`ac0c4f589eae37bbde63968912925de99232e306`

科学基点固定为：

```text
f76067232fac8add756b04ea75a8825cca156330
```

当前唯一工作目录和分支固定为：

```text
D:\digit_writing_original_protocol2
codex/digit-writing-original-protocol3
```

旧 `full10-dev42` 的输出、checkpoint、JSONL 和证据全部只读：

- 不覆盖；
- 不续训；
- 不作为新模型初始化；
- 不写回旧输出目录。

## 1.2 本次允许改变

只允许改变：

1. 数字几何的统一全局尺度；
2. 各片段的 movement interval 分配；
3. 正式训练由三个 reference 条件改为一个固定片段时间条件；
4. batch 内方向采样改为严格均衡；
5. reference 条件固定为 Gate 2 选出的单一时间倍率；
6. 支持上述变化所必需的配置、验证、测试和运行入口；
7. 已确认的 `geometry_audit.py` 辅助入口字段名 bug。

## 1.3 本次禁止改变

不得改变：

- 数字的归一化形状、相对坐标、长宽比、角度、笔顺和连接关系；
- MotorNet plant；
- 28 维 observation 的结构；
- RNN 架构、隐藏单元数、激活函数和网络噪声；
- 6 维肌肉输出；
- stable、delay、movement、hold 四阶段定义；
- phase-normalized position loss；
- 固定阶段权重 `stable=0.1, delay=0.1, movement=0.6, hold=0.2`；
- 四个正式正则项；
- Adam、学习率和梯度裁剪；
- 正式 batch size 32；
- 75,000-update 正式训练长度；
- heldout5、composition、transfer5；
- Stage E、Stage F；
- minimum-jerk、暂停点、连接点平滑或新增 loss。

如果本方案规定的最小验证失败，停止并报告，不得自行扩大改造范围。

## 1.4 旧诊断结论不能直接继承

protocol2 中“不是主要原因”的结论仅适用于旧几何和旧时间表。新几何与新时间分配可能重新引入：

- movement 时间边界或绘图窗口错位；
- 工作空间、关节限位或 IK/FK 问题；
- muscle activation 饱和；
- excitation 长时间贴近 0 或 1；
- 正则项压过位置目标；
- 跨数字或 digit × delay 梯度冲突；
- validation loss 改善但轨迹形状没有改善；
- NaN、Inf 或梯度异常。

这些因素必须重新监测，但不为此新增训练目标或大规模消融。

---

# 2. 新协议的科学边界

新协议继续验证：

- 10 个数字是否由同一 recurrent network 联合学习；
- 严格共享片段在不同数字上下文中是否诱发相似网络活动；
- 冻结网络后能否对已学计算进行组合或重新调用。

新协议不再验证：

- 跨速度泛化；
- 跨速度共享计算；
- 速度条件之间的组合性。

后续结论必须限定为：

> 固定片段时间条件、多个空间方向下的严格共享片段复用和组合机制。

取消多速度不会直接破坏组合性验证，但会缩小科学结论的适用范围。

---

# 3. 新协议身份与输出隔离

协议名继续使用：

```text
digit_writing_original_protocol3
```

但配置中必须明确记录：

```json
{
  "speed_mode": "fixed",
  "selected_reference_steps": 50,
  "direction_mode": "balanced_8",
  "condition_schedule": "independent_random_digit_delay_v1"
}
```

`selected_reference_steps` 最终只能是：

```text
50  → fast
100 → medium
```

建议新增：

```text
configurations/
  digit_writing_original_protocol3_geometry_scale<实际尺度>_ref<50或100>.json
  digit_writing_original_protocol3_gate2_scale<实际尺度>_ref<50或100>.json

server/
  run_digit_writing_original_protocol3_gate1.sh
  run_digit_writing_original_protocol3_gate2.sh
```

新输出根目录：

```text
runs/digit_writing_original_protocol3/
```

不得写入：

```text
runs/digit_writing_original_protocol2/
```

新 checkpoint 必须保存：

- 新协议完整 JSON；
- 当前 Git HEAD；
- parent commit；
- mRNNTorch commit；
- 几何与片段清单 SHA-256；
- `selected_reference_steps`；
- `speed_mode`；
- `condition_schedule`；
- seed、update、validation loss；
- optimizer state 和全部 RNG state。

---

# 4. 冻结的空间尺度

## 4.1 统一缩放

旧尺度：

```python
GLOBAL_SCALE_M_PER_UNIT = 0.06410256410256411
```

新候选尺度：

```python
SCALE_MULTIPLIER = 2.5
GLOBAL_SCALE_M_PER_UNIT_PROTOCOL3 = (
    0.06410256410256411 * 2.5
)
# 0.16025641025641027 m / design unit
```

所有数字和片段只能使用同一个各向同性缩放系数。

禁止：

- 按数字分别缩放；
- 按片段分别缩放；
- 分别改变 x/y 比例；
- 为适应工作空间而手工移动单个数字内部的点。

如果 `2.5×` 未通过工作空间安全警戒线，只允许统一回退至：

```text
2.25×
```

然后重新执行完整 Gate 1。不得继续扫描其他尺度。

## 4.2 缩放后的关键长度

以下值应由代码重新计算并写入审计文件，预期约为：

| 片段 | 2.5× 后长度 |
|---|---:|
| digit 5 顶部横线 | 6.7308 cm |
| digit 5 竖线 | 7.6923 cm |
| 普通直线 | 10.2564–17.2005 cm |
| curve A | 15.9374 cm |
| curve B | 17.2261 cm |
| ellipse 5:4 | 34.0885 cm |
| digit 0 ellipse 4:3 | 42.5067 cm |
| digit 8 independent path | 51.5644 cm |

这些值仅用于审计，不允许手工写死替代实际计算。

---

# 5. 冻结的片段时间表

## 5.1 基本定义

时间统一以 **movement intervals** 表示。

单片段：

```text
samples = intervals + 1
```

多片段完整数字：

```text
digit movement intervals = 各片段 intervals 之和
digit movement samples   = digit movement intervals + 1
```

相邻片段的公共连接点只保留一次。

严格共享片段必须具有：

- 相同归一化几何；
- 相同全局尺度；
- 相同 intervals；
- 相同 canonical sample hash；
- 相同采样方向和顺序。

## 5.2 fast 基础 interval 表

固定：

| timing key | fast intervals | 说明 |
|---|---:|---|
| `line` | 30 | 所有直线统一，不按长度比例细分 |
| `curve_A` | 40 | digit 2 / digit 3 严格共享 |
| `curve_B` | 45 | digit 3 / digit 5 严格共享 |
| `ellipse_5_4` | 70 | digit 6 / digit 9 各自独立轨迹使用同一 timing key |
| `ellipse_4_3` | 85 | digit 0 独立椭圆 |
| `independent_uji8` | 100 | digit 8 独立完整轨迹 |

不得为某个数字单独增加时间。

## 5.3 fast 与 medium 候选

片段 intervals 使用唯一公式：

```python
intervals = floor(
    fast_base_intervals[timing_key]
    * selected_reference_steps
    / 50.0
    + 0.5
)
```

必须使用 `floor(x + 0.5)`，不得使用 Python 默认 `round()`。

候选时间表：

| timing key | fast：50 | medium：100 |
|---|---:|---:|
| `line` | 30 | 60 |
| `curve_A` | 40 | 80 |
| `curve_B` | 45 | 90 |
| `ellipse_5_4` | 70 | 140 |
| `ellipse_4_3` | 85 | 170 |
| `independent_uji8` | 100 | 200 |

这里的 fast/medium 是原仓库的 reference condition 标签，不表示不同片段具有严格相同的物理速度。

## 5.4 完整数字的候选时长

| 数字 | 片段组成 | fast | medium |
|---|---|---:|---:|
| 0 | ellipse 4:3 | 85 | 170 |
| 1 | line | 30 | 60 |
| 2 | curve A + line + line | 100 | 200 |
| 3 | curve A + curve B | 85 | 170 |
| 4 | line + line + line | 90 | 180 |
| 5 | line + line + curve B | 105 | 210 |
| 6 | line + ellipse 5:4 | 100 | 200 |
| 7 | line + line | 60 | 120 |
| 8 | independent path | 100 | 200 |
| 9 | ellipse 5:4 + line | 100 | 200 |

以上均为 intervals。

## 5.5 共享关系的解释边界

严格共享片段：

- `curve_A`：digit 2 / digit 3；
- `curve_B`：digit 3 / digit 5。

`ellipse_5_4` 在 digit 6 / digit 9 中：

- 尺寸相同；
- 必须使用相同 intervals；
- 但起始相位、运动方向和上下文不同。

两者按各自已冻结的设计轨迹独立生成，不属于本项目的严格共享片段，不要求共享
片段 hash 相同，也不据此讨论或验证计算复用机制。

所有直线使用相同 timing key，只表示时间规则一致，不表示所有直线都是严格共享片段。

---

# 6. 固定片段时间条件决策规则

## 6.1 首选 fast

先以：

```text
selected_reference_steps = 50
timing_mode = fixed_segment_timing
```

执行 Gate 1 和 Gate 2。

只有 O1、O2、O3 全部通过，才正式固定 fast。

## 6.2 唯一回退：medium

如果 fast 下任一 O1/O2/O3 未通过，但不存在：

- 数据流错误；
- loss 实现错误；
- IK/FK 或工作空间失败；
- NaN/Inf；
- 其他明显工程 bug；

则只允许将整个协议统一回退到：

```text
selected_reference_steps = 100
timing_mode = fixed_segment_timing
```

然后对 O1、O2、O3 全部重新验证。

禁止：

- 不同数字使用不同 reference 条件；
- 不同数字或片段另行改变已冻结的 interval 表；
- 同时训练 fast 和 medium；
- 再回退到 slow；
- 为单个数字增加额外时间。

若 medium 仍未通过，停止并报告，不启动正式训练。

## 6.3 speed 输入通道仅作为兼容标签保留

不得删除 speed 输入维度。

28 维 observation 保持不变，只把 speed scalar 固定为所选 reference 条件的兼容标签。该标签不表示不同片段具有严格相同的实际物理速度。

这样能够：

- 不改变网络架构；
- 保持与原仓库接口兼容；
- 避免重新定义 checkpoint 结构；
- 明确表明模型只在一个固定片段时间条件下训练。

---

# 7. 正式训练的条件采样

## 7.1 保留方向条件

保留原有 8 个训练方向。

每个正式 batch：

```text
batch_size = 32
8 个方向 × 每个方向 4 个样本
```

要求 8 个方向严格均衡出现，不再对 batch 内每个样本独立随机抽取方向。

batch 内：

- 数字相同；
- 固定片段时间条件相同；
- delay 相同；
- 方向严格均衡；
- 网络噪声仍按正式配置启用。

取消方向不会缩短 trial 时间，且会降低任务多样性，因此本协议不得取消方向条件。

## 7.2 保留三个 delay

保留原始仓库使用的：

```text
delay_steps = [25, 50, 75]
```

原始基线在 `original_baseline_105cd0c/envs.py::choose_delay()` 中从这三个 delay 随机选择。

保留理由：

- 与原始仓库一致；
- 三个 delay 的平均值仍为 50 steps，不会比始终固定 50 增加平均 delay 计算量；
- 它只改变准备阶段长度，不改变共享 movement 片段的空间轨迹和 movement intervals；
- 可以避免模型只适应单一准备时长。

## 7.3 每个 update 独立随机抽取数字和 delay

正式训练保留原仓库式的独立随机采样：

1. 从 10 个数字中等概率随机选择一个数字；
2. 从 `[25, 50, 75]` 中等概率随机选择一个 delay；
3. 片段时间条件固定为 Gate 2 选出的唯一 reference；
4. batch 内 32 个样本使用同一个数字、同一个 reference 和同一个 delay；
5. batch 内 8 个方向各出现 4 次；
6. 完成整个 trial 后计算一次梯度并更新；
7. 下一次 update 重新独立抽取数字和 delay。

不得实现：

- 30-update 分块轮换；
- 固定 `0→9` 顺序；
- 额外 scheduler state；
- 强制每个短窗口内各任务出现次数完全相同。

75,000 updates 下，各数字和各 delay 的抽样次数在期望上均衡。训练日志必须累计记录：

```text
digit_update_counts[0..9]
delay_update_counts[25,50,75]
digit_delay_update_counts[10 × 3]
```

这些计数只用于审计实际采样分布，不参与训练控制。

## 7.4 一个 update 的完整过程

每个 update：

1. 使用训练 RNG 独立等概率抽取一个数字；
2. 使用训练 RNG 独立等概率抽取一个 delay；
3. 使用固定的唯一片段时间条件；
4. 构造 32 个样本：8 个方向各 4 次；
5. 完成 stable + delay + movement + hold 整个 trial；
6. 计算 phase-normalized position loss 和四个正式正则项；
7. 反向传播一次；
8. 梯度裁剪；
9. Adam 更新一次。

checkpoint 必须保存 Python、NumPy 和 Torch RNG state。每个 update 的 MotorNet 环境 seed 必须由已保存的训练 RNG 明确派生；resume 后恢复全部 RNG，不得重新播种后重新抽样。

# 8. 位置损失保持不变

## 8.1 固定阶段权重

继续使用现有 phase-normalized position loss：

```text
stable   = 0.1
delay    = 0.1
movement = 0.6
hold     = 0.2
```

即每个阶段先独立求平均位置误差，再按固定权重组合。

不得把阶段权重改为：

```text
阶段时间步数 / trial 总时间步数
```

原因：

- 这种设置在数学上会退化为整段 trial 的普通逐时间点平均位置损失；
- 不同数字的 movement 长度差异很大，会导致短数字的 movement 权重明显低于长数字；
- 当前首要目标是高质量拟合 movement 轨迹，固定 `0.6` 能避免 stable、delay 和 hold 因步数较多而压低 movement 的相对监督；
- 本轮已经修改尺度、片段时间和速度条件，不应同时再修改 loss，避免无法归因。

允许记录“按时间占比计算的普通 trial mean loss”作为只读诊断指标，但不得用于反向传播、validation 选 best 或 checkpoint 排序。

---

# 9. 最小代码改动


## 9.1 `digit_writing/digit_geometry_final.py`

只做：

1. 新协议使用 `2.5×` 全局尺度；
2. 增加不可变 `FAST_BASE_INTERVALS`；
3. 增加 `timing_key(segment)`；
4. 增加 `segment_intervals(segment, selected_reference_steps)`；
5. 将 `sample_digit()` 改为使用 timing key 和固定 reference；
6. 保留 linear arc-length resampling；
7. 不改变归一化轨迹点。

审计同时记录：

```text
selected_reference_steps
segment_actual_mean_speed_m_s
digit_actual_mean_speed_m_s
```

## 9.2 `digit_writing/geometry.py`

最小适配：

1. `_sample_final()` 接收 `selected_reference_steps`；
2. `build_digit_trajectory()` 保持现有主要外部接口；
3. `PrimitiveBoundary` 保存：
   - timing/template key；
   - start/end index；
   - intervals；
   - arc length；
   - canonical sample SHA-256；
4. time audit 输出片段和完整数字的 interval、samples 和实际平均速度。

不要重构 observation、环境或训练主循环。

## 9.3 `envs.py`

只做必要改动：

1. protocol3 环境只接受一个固定 `selected_reference_steps`；
2. speed 输入通道保留并写入所选 reference 的固定兼容标签；
3. batch 方向由训练器显式传入并严格均衡；
4. 不改变 28 维拼接顺序；
5. 不改变四阶段边界。

## 9.4 `train.py`

只做：

1. 每个 update 独立等概率抽取 digit；
2. 每个 update 独立等概率抽取 delay；
3. 每个 batch 生成 8 方向 × 4 repetitions；
4. reference 固定，不再随机抽 speed/reference；
5. 累计记录 digit、delay 和 digit×delay 的实际 update 计数；
6. checkpoint 保存 reference、采样模式、环境 seed 派生规则和全部 RNG state；
7. validation 改为固定片段时间条件下的：
   - 10 digits；
   - 3 delays；
   - 每个 batch 32 个互不相同的 validation 方向。

不得改变 optimizer、loss 或网络架构。

## 9.5 配置验证

protocol3 必须验证：

```text
speed_mode == fixed
selected_reference_steps ∈ {50, 100}
training_reference_steps == [selected_reference_steps]
validation_reference_steps == [selected_reference_steps]
direction_mode == balanced_8
condition_schedule == independent_random_digit_delay_v1
delay_steps == [25, 50, 75]
digit_sampling == uniform_independent
delay_sampling == uniform_independent
```

protocol2 配置和行为必须保持不变。

composition 和 transfer5 不得自动接受 protocol3。

## 9.6 必要测试

至少覆盖：

1. 2.5× 尺度正确；
2. design-unit 轨迹点未改变；
3. fast/medium 两套总 intervals 与表格一致；
4. 选定速度以外的 speed 不进入 protocol3 正式训练；
5. speed 输入维度保留，28 维 observation 不变；
6. `curve_A` 在 digit 2/3 中 intervals 和 canonical hash 相等；
7. `curve_B` 在 digit 3/5 中满足同样条件；
8. digit 6/9 的椭圆分别按各自设计轨迹生成；intervals 相等，但不做共享 hash
   或计算复用断言；
9. 总 samples = 总 intervals + 1；
10. 四阶段无 off-by-one；
11. 每个 batch 的 8 个方向各出现 4 次；
12. digit 抽样只来自 `0..9` 且为独立均匀抽样；
13. delay 抽样只来自 `[25,50,75]` 且为独立均匀抽样；
14. 相同 seed 和初始 RNG state 生成相同条件序列；
15. 保存并恢复 RNG state 后，后续条件序列与不中断运行一致；
16. 统计计数不反向影响训练采样；
17. 固定阶段权重严格为 `0.1/0.1/0.6/0.2`；
18. 反向传播使用 phase-normalized loss，而不是时间占比 loss；
19. protocol2 旧测试不被改写；
20. 修复并测试：

```text
geometry_audit.py
primitives_with_fewer_than_two_intervals
```

与：

```text
segments_with_fewer_than_two_intervals
```

字段名不一致问题。

---

# 10. 正式训练前 Gate 1：静态与安全预检

## 10.1 必须执行

对 fast 候选和 medium 候选分别生成目标运动学记录，但先只以 fast 进入 Gate 2。

执行：

1. 全部单元测试；
2. geometry self-test；
3. 生成：
   - `primitive_manifest.csv`
   - `digit_timing_fast.csv`
   - `digit_timing_medium.csv`
   - `target_kinematics_fast.csv`
   - `target_kinematics_medium.csv`
   - `workspace_audit.json`
   - `preflight_summary.json`
4. 10 digits × 32 directions 工作空间审计；
5. 对 fast 和 medium 记录：
   - 路径长度；
   - intervals；
   - 实际平均速度；
   - peak/p95 speed；
   - peak/p95 acceleration；
   - peak/p95 jerk；
   - 最小有限曲率半径；
   - 片段连接处方向变化角；
6. 高风险 closed-loop smoke：
   - digit 1 fast；
   - digit 4 fast；
   - digit 5 fast；
   - digit 6 fast；
   - digit 8 fast；
   - digit 9 fast；
   - digit 8 medium。

## 10.2 `primitive_manifest.csv`

至少包含：

```text
digit
segment_name
shared_id
timing_key
reference_steps
arc_length_m
intervals
samples
actual_mean_speed_m_s
canonical_sample_sha256
```

## 10.3 Gate 1 硬性通过条件

全部满足：

- 测试全部通过；
- 服务器正式预检 0 skipped；
- 10 × 32 全部轨迹有限且可达；
- 无 IK/FK 失败；
- `touches_limit = false`；
- 内径、外径和关节余量严格大于 0；
- smoke 无 NaN/Inf；
- activation 和 excitation 边界占比被正确记录；
- 共享片段 intervals/hash 断言通过；
- fast/medium 总 intervals 与表格一致；
- protocol2 文件和旧输出无修改。

安全警戒线：

```text
任一径向余量 < 0.02 m
或任一关节余量 < 0.10 rad
```

若触发，统一回退至 `2.25×`，重新执行 Gate 1。

---

# 11. 正式训练前 Gate 2：固定片段时间条件选择

## 11.1 三个代表性任务

三个独立模型均从零初始化：

| 诊断 | digit | 代表问题 | direction | delay |
|---|---:|---|---:|---:|
| O1 | 1 | 简单直线 | 0 | 50 |
| O2 | 5 | 多段连接 + curve B | 0 | 50 |
| O3 | 8 | 最长独立曲线 | 0 | 50 |

要求：

- batch size 可用 8；
- 网络、MotorNet、loss、optimizer 与正式协议一致；
- 每个模型只训练一个固定条件；
- 训练保留网络噪声；
- deterministic evaluation 关闭网络和环境噪声；
- 每 100 updates 评估；
- 首段最多 3,000 updates，并在该边界保存完整 continuation checkpoint；
- 连续两次通过后可提前停止。

## 11.2 第一轮：fast

O1/O2/O3 全部使用：

```text
selected_reference_steps = 50
```

若全部通过：

```text
正式速度 = fast
```

不再运行 medium 过拟合。

若任一案例在 3,000 updates 时尚未通过，不得只凭 update 数直接认定 fast 不可行，
也不得自动启动 medium。必须先完成第 11.3 节的只读检查和人工判断。

## 11.3 3,000 updates 后的受控续跑

3,000 updates 不是“充分训练”的预设上限。只有同时满足以下条件，案例才有资格
申请续跑：

- source summary 明确分类为 `behavior_failure`；
- 工程、数据流和安全检查全部通过；
- 尚未连续两次达到行为标准；
- source checkpoint、source summary 和 source archive 的 SHA/Git/config/case
  身份全部一致；
- checkpoint 包含完整 model、optimizer、Python/NumPy/Torch RNG、条件计数、
  validation history 和连续通过计数；
- 用户检查完整学习曲线后，为本次续跑明确批准新的
  `target_completed_updates`。

续跑固定规则：

1. 只续跑未通过的单案例，不重新训练已经通过的案例；
2. 从同一 `final_continuation_checkpoint.pt` 恢复，不得改用 best checkpoint；
3. 恢复 model、optimizer、全部 RNG、累计 updates、条件计数、验证历史和连续通过
   计数；
4. 新目标必须严格大于 source updates，并落在 100-update 验证边界；
5. 每次只运行到本次人工批准的目标或提前连续两次通过；
6. 续跑结果写入独立 `gate2_continuations` 目录，不追加、覆盖或修改 source 结果；
7. 每次续跑结束后再次强制暂停，不得自动继续下一段，也不得自动启动 medium；
8. 不预设 6,000 或其他无直接依据的“充分训练”硬上限。是否再续跑必须结合
   normalized error、endpoint error、path ratio、loss/gradient 关系和平台趋势重新
   人工决定。

只读检查必须同时报告首值、末值、最近 10 个验证点、前一组 10 个验证点及两组均值
之差；脚本不得把该趋势统计自动转换成“充分训练”或“必须续跑”的结论。

续跑若通过，仍须执行最终 candidate audit 和轨迹图人工审查。梯度指标继续只作
诊断，不得单独触发续跑、失败或 medium 回退。

## 11.4 第二轮：medium 回退

若 fast 任一模型在获准的检查/续跑后仍由用户判定为行为失败，并且失败不是工程
bug、数据流或安全问题，则：

- 将 O1/O2/O3 全部重新从零训练；
- 统一使用 `selected_reference_steps = 100`；
- 不允许复用 fast 模型。

若三者全部通过：

```text
正式速度 = medium
```

若 medium 任一失败：

```text
Gate 2 = FAIL
禁止启动正式 full10
```

不得再修改片段时长或回退到 slow。

medium 的 3,000-update 边界同样适用第 11.3 节；medium 续跑仍须逐次人工批准，
不得自动执行。medium 最终仍失败后必须停止，不得扫描第三种速度。

## 11.5 必须计算的指标

movement 阶段：

```text
mean_euclidean_error_m
normalized_mean_error
endpoint_error_m
normalized_endpoint_error
actual_path_length_m
target_path_length_m
path_length_ratio

min_inner_radius_margin_m
min_outer_radius_margin_m
min_joint_margin_rad

muscle_activation_max
muscle_activation_fraction_ge_0_99
muscle_excitation_fraction_le_0_01
muscle_excitation_fraction_ge_0_99

position_loss
weighted_l1_rate
weighted_l1_weight
weighted_l1_muscle_act
weighted_simple_dynamics
weighted_regularization_to_position_ratio

position_gradient_norm
regularization_gradient_norm
regularization_to_position_gradient_ratio
position_total_gradient_cosine
```

归一化分母：

```text
target bounding-box diagonal
```

同时保存：

- movement-only overlay；
- x(t)、y(t)；
- loss/指标曲线；
- 最终 checkpoint；
- 完整配置和 seed。

梯度分解只在最终候选 checkpoint 上执行一次，0 optimizer steps；审计前后模型 SHA-256 必须一致。

## 11.6 Gate 2 通过条件

O1/O2/O3 均满足：

```text
normalized_mean_error     <= 0.08
normalized_endpoint_error <= 0.05
0.85 <= path_length_ratio <= 1.15
```

并且：

- 无 NaN/Inf；
- 无工作空间或关节触限；
- 余量高于 Gate 1 安全警戒线；
- 完整记录梯度范数、ratio 和 cosine；当任一相关范数低于 `1e-12` 时 cosine 记为 `undefined`，不得据此判失败；
- 梯度 ratio、cosine 和单次负 cosine 均只作为诊断，不作为独立硬停止条件；
- 无明显只完成前半段、整体缩小、长距离切角或错误闭合；
- 连续两次评估满足。

这些是启动正式训练的工程阈值，不声称来自原论文。

---

# 12. Gate 通过后的正式 full10

## 12.1 启动条件

仅当：

```text
Gate 1 = PASS
Gate 2 = PASS
selected_reference_steps 已冻结
最终工程 PASS/FAIL 阈值已根据 Gate 2 与旧 protocol2 无量纲指标一次性冻结
用户已明确批准只启动前 5,000 updates
```

才允许生成正式启动命令。

未获得用户明确启动指令前，不得开始正式 full10。首次启动只允许运行前 5,000 updates，不是一次性自动运行 75,000 updates。

## 12.2 正式配置

```text
protocol             = digit_writing_original_protocol3
run_kind             = base_training
variant              = full10
seed                 = 42
train_digits         = 0..9
updates              = 75,000
batch_size           = 32
save_iter            = 500
hidden_size          = 256
input_size           = 28
output_size          = 6
optimizer            = Adam
learning_rate        = 0.001
grad_clip_norm       = 1.0

timing_mode          = fixed_segment_timing
selected_reference_steps = Gate 2 选出的 50 或 100
direction_mode       = balanced_8
delay_steps          = [25, 50, 75]
condition_schedule   = independent_random_digit_delay_v1
digit_sampling         = uniform_independent
delay_sampling         = uniform_independent
```

其余模型、反馈、loss 和正则与已验收 full10 保持一致。

必须从零初始化：

```text
不得读取旧 full10 权重
不得读取旧 optimizer state
不得 resume 旧运行
```

建议输出：

```text
runs/digit_writing_original_protocol3/scale<实际尺度>/ref<50或100>/full10/dev42/
```

## 12.3 validation

在 update 0 以及每 500 completed updates 执行固定片段时间条件 validation：

```text
10 digits × 3 delays
```

每个条件使用一个 batch：

```text
32 个不同 validation 方向（direction 0..31）
```

validation 保持原正式验证噪声规则和固定 RNG，可复现。

另外在同一条件网格记录无噪声 deterministic sentinel：

- 10 digits；
- delay = [25, 50, 75]；
- 32 个不同 validation 方向；
- movement-only 行为指标。

不再执行多速度 validation。

## 12.4 运行时最低监控

持续记录：

```text
total loss
position
l1_rate
l1_weight
l1_muscle_act
simple_dynamics
```

每 500 updates 记录：

- normalized mean/endpoint error；
- path-length ratio；
- 最小工作空间和关节余量；
- activation/excitation 边界占比；
- NaN/Inf。

---

# 13. update 5,000 早期只读安全审计

到 update 5,000：

1. 保存完整 continuation checkpoint；
2. 不初始化新模型；
3. 不改变 optimizer 和 RNG；
4. 执行只读审计。

审计条件：

```text
10 digits × 3 delays = 30 conditions
固定正式片段时间条件
每个 batch：32 个不同 validation 方向
delay 与正式训练一致
network/environment noise = false
```

执行：

1. deterministic movement-only 行为评估；
2. update 0 与 update 5,000 使用完全相同的 3 delays × 32 directions 网格；
3. 汇总总体与逐数字 normalized movement error、endpoint error 和 path ratio；
4. 保存十数字 movement-only overlay；
5. 梯度比例和 cosine 如执行，只作为诊断，不参与自动硬停止；
6. 0 optimizer steps；
7. 审计前后模型 SHA-256 一致。

自动停止硬条件：

- NaN/Inf；
- IK/FK、工作空间或关节安全失败；
- continuation checkpoint、optimizer 或 RNG state 不完整。

学习进展报告条件固定为：总体 mean normalized movement error 低于 update 0；至少 8/10 个数字下降；任何数字相对 update 0 的恶化不超过 15%。若未满足，只能暂停并报告，由用户人工决定，不得自动判定正式训练失败。

无论学习进展条件是否满足，update 5,000 后都必须保持暂停。只有用户明确人工批准，才可使用同一 continuation state 恢复至 75,000 updates；不得自动 resume。

---

# 14. 正式训练结束后的最低验收

只做基础行为、安全和训练诊断，不启动 heldout5、composition 或 transfer5。

必须输出：

1. `best_checkpoint.pt`
2. `final_continuation_checkpoint.pt`
3. 完整训练/验证 JSONL
4. 固定片段时间条件下：
   - 10 digits × 3 delays 行为指标；
   - delay=50 的 10 digits × 8 directions movement-only overlay；
5. 每个 digit × delay 的：
   - normalized mean error；
   - normalized endpoint error；
   - path-length ratio；
   - 最小空间/关节余量；
   - activation/excitation 边界占比；
6. best checkpoint 上的 weighted loss 分量；
7. 30 条件只读梯度报告；
8. `formal_run_summary.json`
9. SHA-256 清单。

必须回答：

- 是否仍大量只完成目标前半段；
- digit 5 和 digit 8 是否保持清晰、可识别形状；
- 不同 delay 下是否稳定；
- validation loss 与行为质量是否一致改善；
- 新几何是否重新引入工作空间或关节限制；
- 是否出现 activation 饱和或 excitation 持续贴边；
- 正则项是否重新成为主导；
- 是否存在严重跨数字或 digit × delay 冲突；
- best checkpoint 是否明显优于旧失败结果。

行为验收前不得启动后续实验。

---

# 15. Codex 执行顺序

严格执行：

```text
1. 从 f760672 建立新分支/工作树
2. 确认 protocol2 和旧 full10 只读
3. 实现 2.5× 统一尺度
4. 实现 fast/medium 候选片段时间表
5. 保留 8 方向并实现 batch 内严格均衡
6. 实现固定片段时间模式
7. 保留每个 update 独立随机抽取 digit 和 delay
8. 保留固定阶段权重 `0.1/0.1/0.6/0.2`
9. 做最少量配置适配和测试
10. 修复 geometry_audit 字段名 bug
11. 运行 Gate 1
12. 运行 fast Gate 2
13. fast 全通过则冻结 fast
14. fast 到达 3,000 尚未通过时先检查；仍在改善且用户批准时，从完整 checkpoint 续跑失败案例并再次暂停
15. fast 最终未通过且无工程、数据流或安全问题，则运行 medium Gate 2
16. medium 适用同一人工分段续跑边界；全通过则冻结 medium，否则停止
17. 结合 Gate 2 可达到水平和旧 protocol2 无量纲指标，一次性冻结最终工程 PASS/FAIL 阈值
18. 输出 preflight 报告和唯一正式启动命令
19. 等待用户明确启动指令
20. 正式 full10 只运行到 update 5,000 并强制暂停
21. 完成只读审计并等待用户人工批准
22. 获得人工批准后才从同一 continuation checkpoint 续训至 75,000
23. 按训练前冻结的阈值完成最终行为和安全验收
```

---

# 16. Codex 最终回复格式

预检完成后只汇报：

```text
A. 当前分支与 commit
B. 实际修改文件
C. protocol2 和旧 full10 是否保持只读
D. 最终尺度：2.5× 或 2.25×
E. Gate 1：PASS/FAIL
F. fast O1/O2/O3 结果
G. 是否回退 medium
H. medium O1/O2/O3 结果（若执行）
I. 最终固定片段时间条件及 selected_reference_steps
J. 方向均衡与独立随机 digit/delay 采样测试结果
K. 阶段损失权重与反向传播目标核对结果
L. 旧诊断非主因的重新监测结果
M. 是否允许启动 75,000-update 正式训练
N. 唯一正式启动命令
O. 结果和证据目录
```

不得建议额外实验或扩大项目范围。

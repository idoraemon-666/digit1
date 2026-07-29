# Codex 指导文档：Protocol3 尖角短暂停顿并行对照实验

## 0. 实验定位

当前 Protocol3 Gate 2 已在另一张 GPU 上运行。本文件只定义一组**完全独立的并行尖角实验**，用于回答：

> 数字轨迹在片段连接处发生大角度转向时，加入短暂的定点稳定时间，能否明显减少切角、漏段和轨迹变形？

本实验只验证一个候选补救方法：

```text
Corner Settle：到达尖角连接点后，目标位置短暂停留，再开始下一片段
```

本实验不得：

- 修改、暂停或覆盖当前正在运行的 Gate 2；
- 读取当前 Gate 2 的训练中间 checkpoint 继续训练；
- 修改正式 Protocol3 的已冻结几何、网络、loss 或 optimizer；
- 自动把 corner settle 合并进正式 full10；
- 扩展到 Stage E、Stage F、heldout5、composition 或 transfer5。

本实验的结果只能用于决定：

```text
是否值得把 corner settle 作为 Protocol3 的候选轨迹时间规则
```

正式采用前必须由用户另行批准。

---

# 1. 为什么选择短暂停顿

当前尖角困难的本质是：

- 前一片段在连接点仍有非零速度；
- 下一片段要求运动方向突然变化；
- MotorNet 和 RNN 可能通过圆滑切角、缩短后续片段或绕开连接点来降低平均误差。

单纯增加整段时间只能降低平均速度，不能消除连接点处的瞬时转向。

本实验选择短暂停顿，因为它：

- 不改变数字的空间几何；
- 不改变原片段内部采样点；
- 不改变严格共享片段的 canonical template；
- 不增加新输入维度；
- 不改变 RNN、MotorNet、loss、正则项或 optimizer；
- 只增加尖角处的时间预算；
- 相比圆角化、minimum-jerk、速度曲线重参数化或新增转角 loss，改动更小且更容易做因果解释。

必须同时承认其科学影响：

> 加入暂停会把连续书写变成带短暂稳定点的分段书写，并可能使片段调用和组合变得更容易。

因此本实验是候选补救方案，不是默认正式方案。

---

# 2. 代码和运行隔离

## 2.1 不得接触当前 Gate 2

并行实验必须使用：

- 独立代码副本、分支或 worktree；
- 独立配置文件；
- 独立 run label；
- 独立输出目录；
- 独立 checkpoint；
- 另一张型号相同的 GPU。

不得写入当前 Gate 2 的任何目录。

不得修改当前 Gate 2 正在使用的：

- Git commit；
- Python 文件；
- JSON 配置；
- checkpoint；
- 日志；
- RNG state。

## 2.2 基础身份必须匹配

除 corner settle 外，并行实验必须与当前 Gate 2 使用完全相同的：

- Protocol3 代码基线；
- Git commit 或明确的 parent commit；
- mRNNTorch commit；
- 几何尺度；
- `timing_mode=fixed_segment_timing`；
- `selected_reference_steps`；
- speed scalar；
- 方向；
- delay；
- batch size；
- seed；
- 网络、MotorNet、loss、正则项；
- Adam、学习率和梯度裁剪；
- 最大 updates 和评估频率。

Codex 必须先读取当前 Gate 2 的 resolved config，并将其复制为并行实验基础配置。除本文件允许的字段外，不得自行改变其他字段。

---

# 3. 实验对象

只测试最典型的尖角数字：

```text
digit 4
digit 7
```

固定条件：

```text
direction = 0
delay = 50
selected_reference_steps = 当前 Gate 2 正在验证的 reference 条件
```

若当前 Gate 2 正在验证 fast：

```text
selected_reference_steps = 50
corner_settle_intervals = 10
```

若当前 Gate 2 正在验证 medium：

```text
selected_reference_steps = 100
corner_settle_intervals = 10
```

因为：

```text
dt = 10 ms
```

两者都对应约：

```text
100 ms 的尖角停顿
```

`corner_settle_intervals` 是固定物理时间，不随 reference 条件成比例放大。
因此 medium 也保持 10 intervals；20 intervals 实际是 200 ms，不属于本实验。

不得同时扫描 5、10、15、20 等多组停顿长度。

本轮只验证一个统一的 100 ms 候选值。

---

# 4. 对照条件

## 4.1 必须有匹配基线

每个数字需要：

```text
A：baseline，无 corner settle
B：corner-settle，尖角处加入 100 ms 停顿
```

因此完整实验最多为：

```text
digit 4 baseline
digit 4 corner-settle
digit 7 baseline
digit 7 corner-settle
```

## 4.2 允许复用当前 Gate 2 的 digit 4 baseline

只有同时满足以下条件时，才允许复用当前 Gate 2 的 digit 4 baseline，而不在并行卡上重复训练：

- 当前 Gate 2 确实包含 digit 4；
- digit、direction、delay 完全一致；
- `selected_reference_steps` 一致；
- 几何尺度和时间表一致；
- seed、batch size、最大 updates 一致；
- 网络、loss、optimizer 一致；
- deterministic evaluation 条件一致；
- 有完整 resolved config、checkpoint 和指标文件；
- baseline 未使用 corner settle。

只要任一项不一致，就必须在并行卡上重新训练匹配 baseline。

digit 7 若当前 Gate 2 不包含，则必须同时训练 baseline 和 corner-settle 两个模型。

不得用旧 protocol2 的结果作为本实验 baseline。

---

# 5. 尖角识别规则

不得手工凭视觉随意选择停顿位置。

对 digit 4 和 digit 7 的每个内部片段连接点，计算转向角。

设：

- `v_pre`：前一片段末端最后一个有效非零切向量；
- `v_post`：后一片段起始第一个有效非零切向量。

归一化后：

```python
turn_angle_deg = degrees(
    arccos(
        clip(
            dot(v_pre_unit, v_post_unit),
            -1.0,
            1.0
        )
    )
)
```

仅当：

```text
turn_angle_deg >= 60°
```

时插入 corner settle。

要求：

- 使用实际有序目标采样点计算；
- 跳过重复点和零长度差分；
- 将每个连接点的转向角写入审计 CSV；
- 不在轨迹起点或终点增加 settle；
- 不在平滑、近切线连接处增加 settle；
- 本轮只处理 digit 4 和 digit 7。

输出：

```text
corner_manifest.csv
```

至少包含：

```text
digit
boundary_index
previous_segment
next_segment
turn_angle_deg
settle_applied
settle_intervals
corner_point_x_m
corner_point_y_m
```

---

# 6. Corner Settle 的唯一实现定义

## 6.1 空间轨迹不变

在被选中的尖角连接点：

1. 前一片段完整执行至连接点；
2. 插入一个独立的重复点序列；
3. 每个 settle step 的目标位置都等于该连接点；
4. settle 完成后，下一片段从同一个连接点开始。

因此空间路径仍经过完全相同的连接点，不允许：

- 圆角化；
- 移动连接点；
- 裁剪前后片段；
- 增加过渡弧；
- 改变任一原片段的设计单位坐标。

## 6.2 独立记录为时间片段

插入的停顿必须记录为：

```text
segment_type = corner_settle
```

建议命名：

```text
corner_settle_0
corner_settle_1
...
```

它必须：

- 拥有独立 segment boundary；
- 记录 intervals 和 samples；
- 不属于前一片段；
- 不属于后一片段；
- 不进入原共享片段的 canonical template hash；
- 不改变原共享片段的 ordered samples；
- 在完整数字 movement epoch 内执行。

## 6.3 时间点约定

若 `corner_settle_intervals = K`：

- 连接点在前一片段末端已经出现一次；
- 新增 K 个保持 interval；
- 目标在 K 个后续时间步继续保持连接点；
- 下一片段的第一个非重复目标点在停顿后出现；
- 公共连接点不得因拼接错误重复 K+2 次以上。

必须通过单元测试确认：

```text
新增总 movement intervals
=
所有 settle intervals 之和
```

且：

```text
movement samples
=
movement intervals + 1
```

## 6.4 observation 和 cue 不变

不得新增：

- corner cue；
- settle cue；
- 新 rule 输入；
- 新 observation 维度。

RNN只能根据：

- digit rule；
-固定 reference-condition scalar；
- go cue；
- 空间 cue；
- 视觉和本体感觉反馈；
- 自身内部动力学；

学习停顿和转向。

28 维 observation 必须保持不变。

## 6.5 loss 不变

继续使用正式 phase-normalized position loss：

```text
stable   = 0.1
delay    = 0.1
movement = 0.6
hold     = 0.2
```

corner settle 属于 movement epoch。

不得：

- 增加 corner loss；
- 单独提高连接点权重；
- 删除 settle 时间点的误差；
- 修改四个正则项；
- 改变 validation 选 best 的 loss。

需要在报告中承认：

> 重复连接点不仅增加了减速时间，也增加了连接点在 movement 平均损失中的时间权重。

因此本实验验证的是“corner settle 作为整体时间规则是否有效”，不能把效果完全归因于纯动力学减速。

---

# 7. 训练设置

每个模型：

```text
从零随机初始化
不得读取旧 full10 或其他 Gate checkpoint
```

除 corner settle 外，与当前 Gate 2 完全一致。

除训练审查边界外，保持当前 Gate 2 的：

```text
batch size
seed
evaluation interval = 100 updates
network noise
environment noise
```

训练审查规则固定为：

```text
首轮只运行 fast（selected_reference_steps = 50）
每个模型从零训练至 completed_updates = 6000
每增加 6000 updates 保存完整 continuation checkpoint 并强制暂停
不得因早期行为指标通过而提前结束 6000-update 分段
不得自动续跑下一分段
不得自动切换 medium
```

每个 continuation checkpoint 必须包含完整 model、optimizer、Python RNG、NumPy RNG、
Torch RNG、训练计数和验证历史。每次暂停后只报告确定性评估、训练曲线和完整指标，
是否已经收敛由用户人工审查，不设置未经证据支持的自动收敛阈值。

只有用户判断 fast 已训练收敛但轨迹拟合仍然不佳时，才允许另行启动 medium。
medium 仍使用固定 10 intervals（100 ms），并沿用相同的 6000-update 人工审查边界。

训练时：

- 使用当前 Gate 2 的噪声规则；
- 完整 trial 后计算一次 loss 和 optimizer update。

评估时：

```text
network_noise = false
environment deterministic = true
```

baseline 和 intervention 必须使用相同随机种子和初始化规则。

---

# 8. 必须输出的指标

## 8.1 全局 movement 指标

```text
normalized_mean_movement_error
normalized_endpoint_error
actual_path_length_m
target_path_length_m
path_length_ratio
```

归一化分母与当前 Gate 2 完全一致。

## 8.2 尖角局部指标

对每个 corner 定义目标时间窗口：

```text
前一片段末端最后 10 个非-settle movement steps
+
全部 settle steps
+
下一片段开始后前 10 个非-settle movement steps
```

baseline 没有 settle 时，使用：

```text
连接点前 10 steps + 连接点后 10 steps
```

记录：

```text
corner_local_mean_error_m
corner_local_max_error_m
corner_miss_distance_m
post_corner_progress_ratio
```

定义：

### `corner_miss_distance_m`

在该局部窗口内，实际 fingertip 到目标连接点的最小欧氏距离。

它用于判断实际轨迹是否真正接近尖角，而不是提前切过去。

### `post_corner_progress_ratio`

下一片段开始后的局部实际路径进度除以目标路径进度。

它用于判断模型是否在尖角后漏掉或缩短下一片段。

若该指标的具体计算在当前实现中存在歧义，Codex必须先给出唯一数学定义和单元测试，不得自行生成无法复核的数值。

## 8.3 安全和控制指标

记录：

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

## 8.4 图形

每个数字、每个条件保存：

1. movement-only target/actual overlay；
2. 尖角局部放大图；
3. x(t)、y(t)；
4. fingertip speed 随时间曲线；
5. 连接点位置和 settle 区间标记；
6. 训练/评估指标曲线。

---

# 9. 结果判定

本实验不采用单一 validation loss 判定。

## 9.1 人工候选审查

本轮不冻结缺少直接依据的自动采用阈值，也不由程序自动宣称训练已经收敛。
每个 6000-update 审查点必须完整报告：

1. digit 4 和 digit 7 的所有目标尖角是否实际插入 settle；
2. 两个数字是否存在数值或安全失败；
3. `corner_local_mean_error` 相对各自 baseline 的实际变化量和变化比例；
4. `corner_miss_distance` 相对各自 baseline 的实际变化量和变化比例；
5. `normalized_mean_movement_error`、`normalized_endpoint_error` 和
   `path_length_ratio` 的完整值与成对变化；
6. 最近 1000 和 2000 updates 的确定性评估变化；
7. 最佳 checkpoint 所在 update；
8. overlay 中切角、漏段、连接点绕行、振荡和过冲的实际表现；
9. 停顿后数字整体是否仍明显可辨识。

是否继续 fast、是否认为 fast 已收敛、是否切换 medium、以及 corner settle 是否值得
进入正式协议候选，全部由用户在审查完整证据后决定。

## 9.2 不建议采用

出现以下任一情况，不建议把 corner settle 纳入正式协议：

- 只改善一个数字，另一个无改善或变差；
- 局部误差改善，但整体轨迹明显缩短或变形；
- endpoint 或 path-length ratio 明显恶化；
- 出现新的 activation/excitation 长时间贴边；
- 停顿造成明显振荡、回摆或过冲；
- baseline 本身已经清晰拟合，settle 没有实际增益；
- 只有 loss 改善，但尖角视觉上仍被圆滑切过。

## 9.3 两组都失败

若 baseline 和 corner-settle 都失败：

- 不得自动增加停顿到 200 ms 或更长；
- 不得自动改成圆角、minimum-jerk 或新增 loss；
- 停止并报告。

这说明：

- 尖角可能不是唯一问题；
- 或短暂停顿不足以解决；
- 需要用户决定是否进一步做 MotorNet 直接控制可行性诊断。

---

# 10. 输出目录和 run label

使用独立根目录，例如：

```text
runs/digit_writing_original_protocol3_corner_settle_gate/
```

建议结构：

```text
scale2p50/
  ref50/
    digit4/
      baseline/
      settle100ms/
    digit7/
      baseline/
      settle100ms/
```

若当前 reference 为 medium：

```text
ref100/
```

run label 至少包含：

```text
protocol3
corner-settle-gate
scale multiplier
selected reference steps
digit
baseline 或 settle100ms
seed
Git short SHA
```

不得使用：

```text
full10
stageE
stageF
```

作为本实验标签。

---

# 11. 必要测试

在开始训练前，至少通过：

1. `corner_settle_intervals=0` 时，目标轨迹与当前 Gate 2 baseline 逐点一致；
2. baseline movement intervals 与当前 Gate 2 完全一致；
3. settle 版本只新增重复的连接点，不改变原空间点；
4. 新增 intervals 等于全部 settle intervals 之和；
5. movement samples = movement intervals + 1；
6. digit 4/7 的转向角计算可复现；
7. 只有 `turn_angle_deg >= 60°` 的内部边界插入 settle；
8. 原片段 canonical template hash 不变；
9. 28 维 observation 不变；
10. phase-normalized loss 权重不变；
11. protocol2 和当前 Protocol3 Gate 2 baseline 行为不变；
12. baseline 与 settle 输出目录完全隔离；
13. fixed seed 下两次目标生成完全一致；
14. smoke rollout 无 NaN/Inf、IK/FK 或边界错误。
15. 初始训练严格停止在 6000 updates，并保存完整 continuation state；
16. 续跑目标只能比 source 多 6000 updates，且每段结束后再次强制停止；
17. fast 不得自动触发 medium。

---

# 12. Codex 执行顺序

严格执行：

```text
1. 读取当前 Gate 2 resolved config
2. 建立完全隔离的实验代码和输出身份
3. 实现可选 corner_settle，不改变默认 baseline
4. 添加转向角和时间边界测试
5. 生成 digit 4/7 corner manifest
6. 运行无训练 smoke
7. 确认是否可复用当前 Gate 2 的 digit 4 baseline
8. 在四个独立单线程 CPU 进程中并行运行两个 baseline 和两个 corner-settle 模型至 6000 updates
9. 为四个模型保存完整 continuation checkpoint
10. 生成全局和局部指标、overlay 和对比报告
11. 强制停止并等待用户决定是否续跑下一个 6000-update 分段
12. 只有用户确认 fast 已收敛但拟合仍不佳时，才建立独立 medium 实验
```

不得自动修改正式 Protocol3 文档或启动 full10。

---

# 13. Codex 最终回复格式

Codex 只需汇报：

```text
A. 实验代码身份和 Git commit
B. 读取的当前 Gate 2 resolved config
C. selected_reference_steps
D. 实际插入 settle 的 digit 4/7 边界和转向角
E. baseline 是否复用，复用依据
F. 实际运行的模型列表
G. 每个模型的关键全局指标
H. 每个 corner 的局部指标
I. 安全和 activation/excitation 结果
J. baseline 与 settle overlay 路径
K. 是否满足本文件的候选采用条件
L. 结果和证据目录
```

结论只能是：

```text
推荐作为正式候选
不推荐
证据不足
```

不得直接宣称已授权修改正式协议。

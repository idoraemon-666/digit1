# Protocol3 medium/ref100 五数字 corner-settle 并行诊断设计

## 1. 范围与目的

本设计只属于 `digit_writing_original_protocol3`。它把已经完成的 fast/ref50 digit4、digit7 corner-settle 诊断扩展到一个隔离的 medium/ref100 五数字诊断矩阵：digit2、digit3、digit4、digit5、digit7。

本轮只回答一个问题：在不改变数字空间轨迹、笔画方向、统一尺寸或原始分段时间的前提下，在符合条件的内部尖角插入固定 100 ms 原地目标，是否比 matched baseline 改善局部过角和全轨迹拟合。

本轮不是 formal full10，不得自动续跑，不得自动切换速度，不得自动把任何 settle 方案纳入正式候选。

## 2. 权威几何与不可修改项

唯一空间几何源仍为：

`digit_writing/digit_geometry_final.py`

medium 几何配置固定为：

`configurations/digit_writing_original_protocol3_geometry_scale2p50_ref100.json`

冻结值：

- `scale_multiplier = 2.5`
- `global_scale_m_per_unit = 0.16025641025641027`
- `timing_mode = fixed_segment_timing`
- `selected_reference_steps = 100`
- `dt_seconds = 0.01`
- `stable_steps = 25`
- `delay_steps = 50`（本诊断固定使用中间 delay）
- `hold_steps = 25`
- `direction_index = 0`

settle 只能在原始相邻 primitive 的共享连接点之后插入重复坐标。不得改变、平滑、旋转、翻转、重采样或重新缩放任何原始 primitive；baseline 必须与正式 ref100 轨迹逐元素完全相同。

## 3. 五个数字的冻结方向和连接点

下列坐标先用权威设计单位表示，运行时统一乘以上述 `global_scale_m_per_unit`。

### digit2

方向与分段：

1. rotated curve A：从 canonical A 的上端开始，初始切向为 55°右上，随后沿曲线运行至位移 `(0.48 sin55°, -0.48 cos55°)` 的下端；
2. `diagonal_55_2`：沿 235°方向左下运行；
3. `horizontal_2`：向右运行至终点。

curve A 与 diagonal 在解析几何上相切，不插入 settle。diagonal 转向底部右向横线的内部连接为约 125°尖角，插入一个 settle。

### digit3

方向与分段：

1. `curve_A_3`：从上方起点经过右侧上弧，到达中部连接点；
2. `curve_B_3`：从同一中部连接点向右折返，经过右侧下弧，到达下方终点。

中部连接是接近 180°的方向反转点，插入一个 settle。不得把 digit3 的 curve B 反向或镜像。

设计单位连接点：`(0, 0) -> (0, -0.48) -> (0, -0.96)`；两条曲线均向右侧鼓出。

### digit4

方向与分段：

1. `horizontal_4`：从 `(0, 0)` 向左至 `(-0.72, 0)`；
2. `diagonal_55_4`：从左端向右上运行，高度增量为 `0.78`；
3. `vertical_4`：从斜线顶端向下运行 `1.00`。

两个内部连接分别约为 125°和 145°，均插入 settle。不得把最后竖线改为向上。

### digit5

方向与分段：

1. `horizontal_5`：从 `(0, 0)` 向左至 `(-0.42, 0)`；
2. `vertical_5`：向下至 `(-0.42, -0.48)`；
3. `curve_B_5`：从该点向右进入下弧，最终回到 `(-0.42, -0.96)`。

两个内部连接均约为 90°，均插入 settle。不得把最后的 curve B 置于竖线左侧或反向运行。

### digit7

方向与分段：

1. `horizontal_7`：从 `(0, 0)` 向右至 `(0.64, 0)`；
2. `diagonal_63_435_7`：从右端向左下运行至 `(0.16, -0.96)`。

内部连接约为 116.565°，插入一个 settle。不得把斜线改为右下方向。

## 4. 角点判据与插入规则

- 转角由连接前最后一个非零采样切向量和连接后第一个非零采样切向量计算。
- `turn_angle_deg >= 60.0` 才 qualifies。
- 每个 qualifying 连接在原共享连接点之后插入 10 个完全相同的目标坐标。
- 10 intervals × 10 ms = 100 ms。
- 不 qualifying 的连接仍记录在 manifest，但插入 0 个点。
- baseline 使用 `settle_intervals = 0`，必须保持正式 ref100 轨迹逐元素不变。

冻结的 qualifying 数量：

| digit | 内部边界数 | qualifying 边界索引 | settle 点数 |
|---|---:|---|---:|
| 2 | 2 | `[1]` | 1 |
| 3 | 1 | `[0]` | 1 |
| 4 | 2 | `[0, 1]` | 2 |
| 5 | 2 | `[0, 1]` | 2 |
| 7 | 1 | `[0]` | 1 |

## 5. medium/ref100 时间步冻结表

| digit | primitive intervals | baseline movement intervals | settle movement intervals | baseline movement duration | settle movement duration | baseline episode steps | settle episode steps |
|---|---|---:|---:|---:|---:|---:|---:|
| 2 | `80 + 60 + 60` | 200 | 210 | 2.00 s | 2.10 s | 301 | 311 |
| 3 | `80 + 90` | 170 | 180 | 1.70 s | 1.80 s | 271 | 281 |
| 4 | `60 + 60 + 60` | 180 | 200 | 1.80 s | 2.00 s | 281 | 301 |
| 5 | `60 + 60 + 90` | 210 | 230 | 2.10 s | 2.30 s | 311 | 331 |
| 7 | `60 + 60` | 120 | 130 | 1.20 s | 1.30 s | 221 | 231 |

`episode_steps = stable_steps + delay_steps + movement_intervals + 1 + hold_steps`。

## 6. 冻结 case 矩阵

严格按以下十个独立 case 运行：

1. `digit4_medium_baseline`
2. `digit4_medium_settle100ms`
3. `digit7_medium_baseline`
4. `digit7_medium_settle100ms`
5. `digit2_medium_baseline`
6. `digit2_medium_settle100ms`
7. `digit5_medium_baseline`
8. `digit5_medium_settle100ms`
9. `digit3_medium_baseline`
10. `digit3_medium_settle100ms`

每个数字的 baseline 和 settle 必须使用相同 seed、模型、优化器、batch size、direction、delay 和初始参数状态；汇总必须用初始 `state_dict` SHA256 验证 matched initialization。

## 7. 训练与并行边界

- CPU-only。
- 每个 case 独立单线程进程。
- 十个 case 只在至少 10 个 CPU-equivalent cores 且 `nproc >= 10` 时同时启动；不足则拒绝启动，不静默超卖 CPU。
- `batch_size = 8`。
- 初始训练精确停止在 6000 updates。
- 每 100 updates 验证一次。
- 不允许提前行为通过后自动停止，也不允许越过 6000 自动续跑。
- 不允许自动 medium fallback、其他速度、reference fallback 或 formal full10。
- 任何一个 case 工程或安全失败都保留结果并阻止整体进入行为结论。

## 8. 输出隔离

输出根固定为：

`runs/digit_writing_original_protocol3_corner_settle_gate/scale2p50/ref100/five_digit_review6000`

不得读取或写入 ref50 corner-settle 结果、正式 Gate2 checkpoint、其他项目或其他协议的 checkpoint/results。

## 9. 运行前强制验证

服务器启动训练前必须完成：

1. Git HEAD、记录的 mRNNTorch HEAD 与实际 submodule HEAD 精确匹配；
2. 主仓库和 submodule 均 clean；
3. CPU compatibility 环境标记存在；
4. ref100 Gate1 summary 已通过、scale 为 2p50、repository HEAD 匹配；
5. medium 配置逐字段匹配正式 ref100 Gate2 的模型、优化器、损失、正则和几何；
6. baseline 与权威 ref100 轨迹逐元素相同；
7. 五个数字的 primitive 名称、intervals、方向终点、qualifying 索引和最终 movement intervals 符合本设计；
8. 全部指定测试通过且不得有 skipped tests；
9. 输出根、归档和日志均不存在，禁止覆盖。

## 10. 人工审查

6000 updates 完成后必须同时报告 final checkpoint 与 best checkpoint 的：

- phase-normalized position loss；
- normalized mean error；
- normalized endpoint error；
- path-length ratio；
- 每个角点的 local mean/max error、corner miss distance、post-corner progress ratio；
- movement overlay、corner-local overlay、fingertip speed；
- dynamic safety、activation/excitation boundary metrics；
- 每对 baseline/settle 的 matched-initialization 证明。

完成归档后强制暂停，由用户决定下一步。

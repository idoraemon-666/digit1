# Protocol3 digit8 等权点损失学习率消融

## 目的

旧卡 medium/ref100 的 `o3_digit8` 从 3000 续跑到 6000 后仍为行为失败：
最终 normalized mean error 为约 0.04725，normalized endpoint error 为约
0.11562，path ratio 为约 0.84332。逐点复核显示起点、终点和若干中段同时存在
误差，且轨迹整体横向压缩，因此本实验不通过给起点、终点或闭环增加特殊权重来
追逐 endpoint 审计阈值。

## 冻结原则

- movement 的 201 个位置样本继续在 `L_movement` 中等权；
- position loss 保持 `phase_normalized_l1`，阶段权重保持
  `stable=0.1, delay=0.1, movement=0.6, hold=0.2`；
- 起点额外权重、终点额外权重、闭环额外权重全部固定为 0；
- 模型、正则、batch、训练噪声、digit、direction、delay、几何、scale、ref 和
  时间步数全部不变；
- 三臂从同一个 O3@6000 state-complete checkpoint 恢复相同模型、Adam moments、
  condition counts、validation history 和 Python/NumPy/Torch RNG；
- 恢复 Adam 后只允许覆盖各 param group 的 `lr`；
- 三臂都固定执行到 7000，不因两次指标通过提前停止；
- 不自动选择胜者，不自动再续跑，不自动切换 reference，不启动 formal full10。

## 三个并行臂

| label | Adam learning rate | source | target |
|---|---:|---:|---:|
| `digit8_medium_lr1e3` | 0.001 | 6000 | 7000 |
| `digit8_medium_lr3e4` | 0.0003 | 6000 | 7000 |
| `digit8_medium_lr1e4` | 0.0001 | 6000 | 7000 |

`lr=0.001` 是延续原优化过程的 matched control。其余两臂仅改变学习率。

## 冻结的 digit8 身份

- scale multiplier：2.5；
- fixed-segment timing reference：100；
- digit：8；direction index：0；delay：50 steps；
- 起笔：顶端；初始方向：向左下；目标首尾闭合；
- movement：200 intervals / 201 samples；
- stable/delay/movement/hold：25/50/201/25 samples；
- episode：301 steps；dt：0.01 s。

## 来源身份

- source repository HEAD：`792c4d3d20ae6ec0c5deef4f06bd41a0859730ba`；
- mRNNTorch：`ac0c4f589eae37bbde63968912925de99232e306`；
- source checkpoint SHA256：
  `45373d9db856afa08652ed700081d23a375f5d772499e2ae56efc89423088c64`；
- source summary SHA256：
  `991385a6862af4ddff99101c8704fb07da31c239e1b5c6711e7b9d41fcaf0112`。

## 复核边界

每 100 updates 记录原始等权 position loss、normalized mean error、normalized
endpoint error 和 path ratio。7000 时对每臂执行无 optimizer step 的完整 Gate2
candidate audit，并保存 movement overlay/CSV。汇总另外从 CSV 计算：

- 201 点平均欧氏误差；
- 最大单点误差及其 index；
- 起点和终点误差；
- 前 10 点、中间 181 点、后 10 点平均误差。

任何一个指标都不自动决定后续。人工判断必须优先看全部 201 点、最大误差、path
ratio 和 overlay，不能只依据 endpoint error。即使某臂达到原 Gate2 数值门槛，仍只
是诊断结果，不自动进入正式 Protocol3。

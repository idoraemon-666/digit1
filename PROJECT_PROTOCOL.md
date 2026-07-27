# digit_writing_original_protocol2 科学与实施规范

## 1. 权威来源与项目边界

本项目直接复用已完成的原仓库改造框架，但形成独立的项目 2 结果谱系。

权威顺序：

1. `digit_writing/digit_geometry_final.py`：数字几何、物理尺度、速度与分段采样的唯一权威来源；
2. 本文件：模型、输入、损失、训练、组合、迁移和审计规范；
3. `CODEX_MINIMAL_FINAL_DIGIT_MODIFICATION.md`：本轮决定的实施来源；
4. 原仓库代码：仅决定本文明确要求保持不变的部分。

项目名称固定为 `digit_writing_original_protocol2`。服务器继续使用：

```text
/root/autodl-tmp/digit_writing_original_protocol_4b8b1db
```

但配置、运行目录、归档和日志必须使用项目 2 命名空间，不得覆盖或读取上一项目的训练结果。

禁止导入历史数字项目或上一项目的 checkpoint、optimizer state 和训练结果。

## 2. 最终数字几何

### 2.1 唯一实现

`envs.py` 通过 `digit_writing/geometry.py` 的薄适配接口直接调用
`digit_writing/digit_geometry_final.py`。仓库中不得存在第二套数字坐标公式。

最终统一尺度：

```text
digit 0 physical height = 0.06153846153846154 m
global scale            = 0.06410256410256411 m / design unit
```

所有数字使用同一全局尺度，禁止数字特异缩放。

### 2.2 曲线共享关系

```text
A：数字 2 上部、数字 3 上部
B：数字 3 下部、数字 5 下部
独立 8：仅用于数字 8
4:3 椭圆：数字 0
5:4 椭圆：数字 6、9
```

平移和旋转不改变共享身份，镜像不算共享。共享 A/B 在相同速度下必须具有相同弧长、interval 数和刚性对齐后的采样形状。旧曲线 C 不属于项目 2。

### 2.3 数字结构摘要

```text
0：4:3 竖直椭圆，上端起笔，逆时针一周
1：竖直向下
2：A 旋转 55° → 55°向左下斜线 → 向右横线
3：上 A → 下 B
4：向左横线 0.72 → 55°向右上斜线 → 向下竖线
5：向左横线 → 向下竖线 → B
6：63.435°切向斜线 → 5:4 椭圆逆时针一周
7：向右横线 → 63.435°向左下斜线
8：独立轴对称 Gerono 型轨迹
9：5:4 椭圆从右端顺时针一周 → 向下竖线
```

具体控制点和构造公式只以 `digit_geometry_final.py` 为准。

## 3. 速度与时间

固定：

```yaml
dt_s: 0.01
train_speed_mps:
  fast: 0.5
  medium: 0.25
  slow: 0.16666666666666667
train_speed_scalar:
  fast: 0.6666666666666666
  medium: 0.3333333333333333
  slow: 0.0
```

验证参考步数：

```text
[50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
```

速度和 speed scalar 分别为：

```text
v = 0.25 / (reference_steps * 0.01)
speed_scalar = 1 - reference_steps / 150
```

每个 segment 独立计算：

```text
intervals = ceil(segment_length / (speed * dt))
```

使用线性弧长重采样，连接点只保留一次。禁止 minimum-jerk、插入停顿、转角速度归零和统一总 movement 时长。

movement 阶段监督 `movement_intervals + 1` 个轨迹样本，必须同时覆盖首末目标点。hold 从最后一个 movement 样本的下一步开始。

## 4. 模型、输入和 trial

以下保持不变：

```yaml
network: rnn
input_size: 28
hidden_size: 256
activation: softplus
output_size: 6
recurrent_noise_std: 0.1
input_noise_std: 0.01
constrained: false
rnn_dt_ms: 10
rnn_tau_ms: 20
batch_first: true
```

保持 `RigidTendonArm26`、`MujocoHillMuscle`、sigmoid 肌肉输出、视觉反馈、本体反馈和 `action_frame_stacking=0`。

28 维输入顺序：

```text
[0:10]   digit rule one-hot
[10:11]  speed scalar
[11:12]  go cue
[12:14]  2-D spatial direction cue
[14:16]  fingertip visual feedback
[16:28]  muscle proprioceptive feedback
```

时序：

```yaml
stable_steps: 25
delay_steps: [25, 50, 75]
hold_steps: 25
training_angles: 8
validation_angles: 32
```

rule 全 trial 恒定；speed 与空间提示在 stable 为 0、从 delay 开始保持；go cue 从 movement 开始；hidden target 在 stable/delay 保持起点、movement 逐点跟踪、hold 保持终点。

同一 batch 的 digit、speed、delay 相同，每个样本空间方向可不同。

## 5. 阶段归一化位置损失

每步位置误差：

```text
e_t = |x_t - x*_t| + |y_t - y*_t|
```

四阶段分别内部求 mean，主位置项固定为：

```text
L_position = 0.1 L_stable
           + 0.1 L_delay
           + 0.6 L_movement
           + 0.2 L_hold
```

不得按照阶段实际步数再次加权。

总损失保持：

```text
L = L_position
  + L_rate
  + L_weight
  + L_muscle
  + L_simple_dynamics
```

基础训练的其他正则项、系数、作用对象和完整 trial 语义不变。迁移和组合只使用阶段归一化位置项。

训练、checkpoint 验证、组合优化和迁移必须统一使用该位置目标。full-trial L1、四阶段 mean、endpoint error 和 hold drift 只记录为诊断。

`hold drift` 定义为 hold 阶段实际位置相对 movement 最后实际位置的平均二维 L1 距离；不参与优化或 checkpoint。

## 6. 基础训练与模型隔离

固定优化设置：

```yaml
optimizer: Adam
learning_rate: 0.001
batch_size: 32
max_updates: 75000
validation_interval: 500
grad_clip_norm: 1.0
l1_rate: 0.001
l1_weight: 0.001
l1_muscle_act: 0.01
simple_dynamics_weight: 0.001
```

模型 A `full10`：训练数字 0–9，用于基础行为和组合。

模型 B `heldout5`：训练 `{0,1,2,3,4,6,7,8,9}`，仅用于数字 5 迁移。

两套模型必须独立实例化、独立 optimizer、独立配置、独立目录、独立 checkpoint，不得相互加载状态。

## 7. 组合

固定任务组：

```text
G = {0, 4, 6, 9, 8}
```

对每个 `d ∈ G` 执行一次 leave-one-out，只优化其余四个 rule 的外部系数：

```text
q_d = sum(alpha_j * e_j), j ∈ G, j != d
```

目标和组外 rule 系数均为 0。网络、输出层和 MotorNet 全部冻结。rollout 使用目标数字自己的完整 trial 和 movement 长度，不执行来源数字轨迹。

固定组合参数：

```yaml
optimizer: Adam
coefficient_learning_rate: 0.1
iterations: 250
network_noise: false
initial_coefficients: 1.0
batch_size: 8
validation_direction_indices: [0,4,8,12,16,20,24,28]
validation_speed_index: 9
custom_delay: 150
```

必须报告五个目标，禁止根据结果更换任务组。

## 8. 数字 5 迁移

迁移只允许读取项目 2 的 heldout5 best checkpoint。开始前重新初始化数字 5 rule 列，只训练该列；其他网络参数、已有九个 rule、输出层和 MotorNet 必须逐位不变。

训练数据只包含数字 5；位置目标使用阶段归一化位置损失，不加基础训练正则；不得用组合系数初始化。

## 9. 必要审计

正式训练前必须通过：

1. `digit_geometry_final.py --self-test` 及几何/时间输出；
2. 阶段损失权重、mask、可变 movement 和端点测试；
3. 28 维输入、batch 条件和 custom delay 环境测试；
4. `10 digits × 32 directions` MotorNet 工作空间审计；
5. 数字 1/4/6/8/9 高风险短闭环 smoke test；
6. 组合网络 bitwise 冻结测试；
7. 迁移只有数字 5 rule 列变化的冻结测试。

若工作空间失败，只报告，不得自动缩放。

不重复旧 Git provenance 全审计、历史项目安全审计、无关旧分析和正式多 seed 训练。

## 10. 配置、运行入口与停止线

配置位于：

```text
configurations/digit_writing_original_protocol2_geometry.json
configurations/digit_writing_original_protocol2_full10_dev42.json
configurations/digit_writing_original_protocol2_heldout5_dev42.json
configurations/digit_writing_original_protocol2_composition.json
configurations/digit_writing_original_protocol2_transfer5.json
```

服务器入口位于：

```text
server/run_digit_writing_original_protocol2_audit.sh
server/run_digit_writing_original_protocol2_full10_dev42.sh
server/run_digit_writing_original_protocol2_heldout5_dev42.sh
server/run_digit_writing_original_protocol2_composition.sh
server/run_digit_writing_original_protocol2_transfer5.sh
```

正式实验仍要求用户逐项明确授权 exact run label。完成代码和审计不自动授权任何 75,000-update 训练、组合或迁移。

用户审查前禁止自动修改尺度、速度、损失权重或网络规模，禁止启动正式训练，禁止根据 smoke test 或组合结果修改数字外形。

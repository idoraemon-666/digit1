# Codex 实施任务：按原仓库协议重新改造数字书写项目

> 本文是本轮重构的唯一执行指令。你已经熟悉当前项目目录、`reference/original_repo`、原论文及现有代码，因此不要重新泛读无关材料，也不要自行扩展研究问题。
>
> 本轮允许推翻当前数字项目的旧实验协议和旧实现，但必须保留 Git 历史、只读参考仓库及已有审计证据。旧实验结果只能作为历史开发记录，不能继续作为新协议的训练依据或正式结果。

---

## 1. 本轮重构的核心原则

新项目只做一项主要科学改动：

> 将原仓库的 10 种运动任务替换为 0–9 的 10 种参数化数字轨迹。

除数字目标轨迹以及由“不同数字总弧长不同”直接导致的最小必要适配外，以下内容尽量与 `reference/original_repo` 的实际实现保持一致：

- MotorNet 闭环肌肉骨骼系统；
- mRNNTorch RNN；
- 256 个 Softplus 隐藏单元；
- 10 维静态 rule one-hot；
- 8 个训练空间方向、32 个验证空间方向；
- 3 个训练速度条件、10 个验证速度条件；
- stable、delay、movement、hold 时序；
- 28 维输入合同；
- 原仓库实际使用的 L1 位置损失和全部原有正则项；
- 原优化器、梯度裁剪、模型选择逻辑；
- 代数组合通过闭环轨迹误差优化 rule 系数；
- 留出迁移通过冻结动力学、只学习新任务 rule 输入完成。

禁止把当前旧数字项目中的下列设计带入新协议：

- 正向/反向笔顺条件；
- 2 维“正笔顺/反笔顺”方向 one-hot；
- 每个基元独立 minimum-jerk 起停；
- 基元边界强制速度归零；
- 显式速度损失；
- movement、pre、hold 的重新加权损失；
- 预设任何 rule 代数公式，例如 `r5 = r2 + r9 - r0`；
- 让同一个九数字模型同时承担代数组合和留出迁移；
- 使用旧 seed-42 checkpoint、旧继续训练边界或旧行为分数继续训练。

---

## 2. 参考边界和代码处理原则

### 2.1 只读参考

- `reference/original_repo` 是原实验行为基准，禁止修改。
- 当前 `mRNNTorch` 子模块保持既有已验证版本，除非兼容性确实要求，否则不修改。
- MotorNet 版本兼容适配可以保留，但不得借兼容性名义改变科学协议。

### 2.2 当前数字项目代码

允许重写或替换：

- `digit_writing/geometry.py`
- `digit_writing/config.py`
- `digit_writing/environment.py`
- `digit_writing/training.py`
- `digit_writing/model.py`
- 数字项目相关测试、配置、报告和运行入口

旧协议专属的配置、测试和 runner 不得继续伪装成当前有效实现。处理方式二选一：

1. 删除无历史保留价值的旧入口；或
2. 明确移入 `legacy/` 并在文件名和文档中标注 `OBSOLETE_OLD_PROTOCOL`。

不得留下会被误执行的旧默认入口。

### 2.3 文档优先级

完成重构后：

- 本文内容写入项目内新的权威规范；
- 更新 `NEXT_SESSION_HANDOFF.md`；
- 将旧 `codex_digit_writing_project_brief.md` 和旧 frozen spec 标为已被新协议取代，或重写为与本规范一致；
- 更新 `轨迹.md` 为本文第 5 节的最终定义；
- 不再保留“正反笔顺、minimum-jerk、预设代数公式、3 是 8 前缀、0 是 9 前缀”等过时结论。

---

## 3. 原仓库实验设置：必须保持的部分

以 `reference/original_repo/train.py`、`envs.py`、`losses.py`、`model.py` 和对应实验入口的实际代码为基准，而不是根据论文文字自行改写。

### 3.1 模型

正式基础模型固定为：

```yaml
network: rnn
input_size: 28
hidden_size: 256
activation: softplus
output: 6 muscle activations
recurrent_noise_std: 0.1
input_noise_std: 0.01
constrained: false
rnn_dt_ms: 10
rnn_tau_ms: 20
batch_first: true
```

保留：

- `RigidTendonArm26`
- `MujocoHillMuscle`
- sigmoid 肌肉输出层
- 视觉反馈和本体感觉反馈
- `action_frame_stacking = 0`

不得增加：

- 新的隐藏层；
- 新的轨迹编码器；
- primitive ID、笔画 ID、基元时刻、目标终点序列等额外输入；
- 数字特异的速度输入或方向输入。

### 3.2 优化与正则

基础训练沿用原仓库：

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

如果当前兼容实现对参数命名不同，应保持数值和作用对象一致，并写出映射表。

### 3.3 Trial 时序

```yaml
stable_steps: 25
delay_steps: [25, 50, 75]
hold_steps: 25
```

- digit rule：全 trial 恒定；
- speed scalar：stable 阶段为 0，从 delay 开始保持到 hold 结束；
- 2 维空间方向提示：stable 阶段为 0，从 delay 开始保持到 hold 结束；
- go cue：movement 开始时由 0 变为 1；
- hidden target：stable/delay 保持起点，movement 逐点跟踪数字轨迹，hold 保持终点。

### 3.4 Batch 采样方式

必须恢复原仓库的采样语义：

- 每个训练 batch 只选择一个 digit；
- 每个 batch 只选择一个 speed condition；
- 每个 batch 只选择一个 delay；
- batch 内每个样本独立随机选择 8 个训练空间方向之一；
- 10 个数字等概率采样；九数字基础模型中其余 9 个数字等概率采样；
- 同一 batch 内 episode 长度一致，因为 digit、speed、delay 一致。

不要沿用旧数字项目“一个 batch 所有样本只有一个固定空间方向”的实现。

---

## 4. 输入合同：保持 28 维，不再使用正反笔顺

输入顺序必须与原仓库语义一致：

```text
[0:10]   digit rule one-hot
[10:11]  speed scalar
[11:12]  go cue
[12:14]  2-D spatial direction cue / vis_inp
[14:16]  fingertip visual feedback
[16:28]  muscle proprioceptive feedback
```

### 4.1 数字 rule

- 10 维 one-hot 对应数字 0–9；
- 完整十数字模型中正常使用全部 10 列；
- 九数字留出模型中数字 5 的 rule 通道从不激活；
- rule 在 stable、delay、movement、hold 全阶段恒定。

### 4.2 空间方向

只保留一种规定笔顺。所谓 8 个方向，是将完整规定方向数字轨迹围绕共同起笔点做刚性旋转：

```text
training angles = 0°, 45°, 90°, ..., 315°
validation angles = 32 个等间隔角度，即 0°, 11.25°, ..., 348.75°
```

设规定方向局部轨迹为 `p(t)`，且 `p(0)=0`，空间方向为 `theta`：

\[
p_\theta(t)=p_{anchor}+R(\theta)p(t).
\]

2 维 `vis_inp` 不再表示正反笔顺，而恢复为原仓库的空间目标提示：

\[
c_\theta=p_{anchor}+0.25
\begin{bmatrix}
\cos\theta\\
\sin\theta
\end{bmatrix}.
\]

- stable 阶段输入 `[0,0]`；
- delay 开始后持续输入 `c_theta`，与原仓库的目标提示幅值和时间规则一致；
- `c_theta` 是空间方向提示，不要求等于数字轨迹终点；
- 不允许把完整目标轨迹、数字边界框或 primitive 信息输入网络。

---

## 5. 数字几何：唯一最终规范

## 5.1 坐标、尺寸和统一缩放

规定方向坐标：

```yaml
x_positive: right
y_positive: up
unit: meter
```

参数化尺寸表和本文公式是唯一几何规范；用户提供的手绘图只用于说明笔顺和拓扑，不作为像素级参考，也不得引入外部字体或参考图拟合。沿用当前新项目轨迹脚本中的几何数值：

```yaml
a: 0.03500          # 标准椭圆半长轴
b: 0.02625          # 标准椭圆半短轴，b = 0.75*a
H: 0.03500          # 所有标准横线长度
V: 0.06125          # 所有标准竖线长度
D_dx: 0.02625       # 标准斜线水平分量
D_dy: 0.05250       # 标准斜线竖直分量
global_scale: 0.8791208791208792
```

由此：

```text
标准斜线长度 = sqrt(D_dx^2 + D_dy^2)
              = 0.05869678440936948 m（缩放前）
标准斜线与 +x 轴夹角 = atan2(D_dy, D_dx)
                      = 63.43494882292201°
```

所有数字完成规定方向局部几何后，统一乘 `global_scale`。禁止：

- 单独缩放某个数字；
- 单独改变某条横线、竖线或斜线；
- 单独改变某个椭圆的长短轴；
- 恢复 `lobe_rx/lobe_ry`。

删除 `GeometryConfig` 和 JSON 中的：

```text
lobe_rx
lobe_ry
```

所有完整椭圆和半椭圆均使用同一个标准椭圆 `a,b`，仅允许平移、旋转、选择弧段和改变运行方向。

## 5.2 统一基元库

至少建立以下唯一基元或通用生成函数：

```text
H_right, H_left
V_down, V_up
D_up, D_down
standard_ellipse(a, b, orientation, start_phase, end_phase, direction)
standard_half_ellipse(...)
```

其中：

```text
D_up   = (+D_dx, +D_dy)
D_down = (-D_dx, -D_dy)
```

所有数字中的同类直线必须直接复用同一个基元定义。椭圆必须调用同一个通用生成函数，不得复制十份近似公式。

## 5.3 参数化公式

水平长轴标准椭圆：

\[
E_h(\phi)=
\begin{bmatrix}
a\cos\phi\\
b\sin\phi
\end{bmatrix}.
\]

竖直长轴标准椭圆：

\[
E_v(\phi)=
\begin{bmatrix}
b\cos\phi\\
a\sin\phi
\end{bmatrix}.
\]

旋转椭圆：设长轴单位向量为 `u`，与其顺时针垂直的短轴单位向量为：

\[
v=(u_y,-u_x),
\]

则：

\[
E_{u}(\phi)=a\cos\phi\,u+b\sin\phi\,v.
\]

## 5.4 0–9 的规定方向轨迹

### 数字 0

- 使用竖直长轴标准椭圆；
- 从最上顶点出发；
- 逆时针完整一周；
- 回到起点。

\[
\phi:\frac{\pi}{2}\rightarrow\frac{5\pi}{2}.
\]

### 数字 1

直接复用：

```text
V_down: (0,0) -> (0,-V)
```

### 数字 2

由半椭圆、标准斜线、标准横线三段组成。

定义标准斜线长轴方向：

\[
u=\frac{(D_{dx},D_{dy})}{\sqrt{D_{dx}^2+D_{dy}^2}},
\qquad
v=(u_y,-u_x).
\]

半椭圆弧：

\[
E_2(\phi)=a\cos\phi\,u+b\sin\phi\,v,
\qquad
\phi:-\frac{\pi}{2}\rightarrow\frac{\pi}{2}.
\]

三个关键点：

\[
A=-bv,\qquad B=au,\qquad C=bv.
\]

解释：

- `A、C` 是两个短轴端点；
- `B` 是弧线经过的长轴端点；
- 椭圆长轴平行于标准斜线；
- 在 `C` 点，弧线切向为 `-u`。

随后：

```text
C -> D：执行 D_down，即位移 (-D_dx, -D_dy)
D -> E：执行 H_right，即位移 (+H, 0)
```

因此 `CD` 与半椭圆在 `C` 点几何相切。不得在连接点插入停顿或 minimum-jerk 起停。

### 数字 3

由上下两个形状和大小完全相同的标准水平椭圆右半周组成：

- 上椭圆中心：`(0,+b)`；
- 下椭圆中心：`(0,-b)`；
- 两者在中间颈点 `(0,0)` 相接；
- 从最上点开始，经中间颈点，到最下点。

上半段：

\[
(0,b)+E_h(\phi),
\quad
\phi:\frac{\pi}{2}\rightarrow-\frac{\pi}{2}.
\]

下半段：

\[
(0,-b)+E_h(\phi),
\quad
\phi:\frac{\pi}{2}\rightarrow-\frac{\pi}{2}.
\]

删除所有专用小叶瓣参数和函数。数字 3 不再要求是数字 8 的连续前缀。

### 数字 4

控制点：

\[
P_0=(H,0),\quad
P_1=(0,0),\quad
P_2=(D_{dx},D_{dy}),\quad
P_3=(D_{dx},D_{dy}-V).
\]

笔顺：

```text
P0 -> P1：H_left
P1 -> P2：D_up
P2 -> P3：V_down
```

### 数字 5

笔顺：

```text
H_left -> V_down -> 水平长轴右半椭圆（从上到下）
```

右半椭圆：

\[
E_h(\phi),
\qquad
\phi:\frac{\pi}{2}\rightarrow-\frac{\pi}{2}.
\]

它与数字 2 的半椭圆形状和大小完全相同，均来自标准 `a,b` 椭圆；数字 2 只是在空间中对同一模板做了刚性旋转和平移。数字 5 的横线必须复用标准 `H`，竖线必须复用标准 `V`。

### 数字 6

笔顺修改为：

1. 从标准斜线的自由端点开始；
2. 沿 `D_up` 到达竖直长轴椭圆的切点；
3. 从切点开始逆时针绕完整椭圆一周；
4. 回到同一切点结束。

竖直椭圆：

\[
E_v(\phi)=
\begin{bmatrix}
b\cos\phi\\
a\sin\phi
\end{bmatrix}.
\]

逆时针切向：

\[
t_{ccw}(\phi)=
\begin{bmatrix}
-b\sin\phi\\
a\cos\phi
\end{bmatrix}.
\]

令切向与 `D_up` 同向，切点相位为：

\[
\phi_6=
\operatorname{atan2}
\left(
-\frac{D_{dx}}{b},
\frac{D_{dy}}{a}
\right)
\approx-0.58800260355\ \mathrm{rad}
\approx-33.690067526^\circ.
\]

令切点 `C6 = E_v(phi6)`，自由起点为：

\[
S_6=C_6-(D_{dx},D_{dy}).
\]

轨迹：

```text
S6 -> C6：D_up
C6 开始：phi6 -> phi6 + 2*pi，逆时针完整椭圆
```

斜线到椭圆必须几何相切；不插入停顿。

### 数字 7

```text
H_right -> D_down
```

控制点：

\[
P_0=(0,0),\quad
P_1=(H,0),\quad
P_2=(H-D_{dx},-D_{dy}).
\]

### 数字 8

使用两个形状和大小完全相同的标准水平长轴椭圆：

- 上椭圆中心：`(0,+b)`；
- 下椭圆中心：`(0,-b)`；
- 两者在中间颈点 `(0,0)` 相接；
- 起点 `A=(0,2b)`，即上椭圆最上顶点。

笔顺严格为：

1. 从 `A` 出发，沿上椭圆左半周逆时针到中间颈点；
2. 从颈点出发，沿下椭圆顺时针完整一周并回到颈点；
3. 从颈点出发，沿上椭圆右半周逆时针回到 `A`。

参数范围：

```text
上椭圆第一半：phi =  pi/2 ->  3*pi/2   （逆时针）
下椭圆完整环：phi =  pi/2 -> -3*pi/2   （顺时针）
上椭圆第二半：phi = -pi/2 ->  pi/2     （逆时针）
```

删除旧的四个小叶瓣函数。数字 3 不再是数字 8 的连续前缀。

### 数字 9

- 使用竖直长轴标准椭圆；
- 从最右顶点出发；
- 顺时针完整一周；
- 回到最右点后执行标准 `V_down`。

\[
\phi:0\rightarrow-2\pi,
\]

随后：

```text
V_down: (0,0) -> (0,-V)
```

数字 9 与数字 0 只共享椭圆的形状和大小，不共享起点、方向或逐采样点前缀。删除任何 `9[:T0] == 0` 断言。

---

## 6. 时间参数化和速度条件

## 6.1 不再数学消除速度突变

目标生成器只要求位置连续 `C0`：

- 相邻基元首尾坐标必须一致；
- 不做圆角化；
- 不插入停顿；
- 不强制边界速度归零；
- 不使用 minimum-jerk；
- 锐角和换向点允许目标速度方向突变；
- 2 和 6 的相切连接应自然保持几何切向一致。

保留高分辨率弧长查表，但将旧：

```python
target_arc = minimum_jerk(u) * arc_length
```

替换为线性弧长推进：

```python
target_arc = np.linspace(0.0, arc_length, n_intervals + 1)
```

每个共享基元在相同速度条件下使用同一采样数组。拼接时连接点只保留一次。

## 6.2 三个训练速度：按原仓库速度条件解释

原仓库的 0.25 m 单程 reach 在三个训练 movement time 下分别为 50、100、150 步，`dt=0.01 s`。据此定义三个全局物理速度：

```text
fast   = 0.25 / (50  * 0.01) = 0.5000000000 m/s
medium = 0.25 / (100 * 0.01) = 0.2500000000 m/s
slow   = 0.25 / (150 * 0.01) = 0.1666666667 m/s
```

对任意共享基元 `p`：

\[
n_{p,k}=\left\lceil\frac{L_p}{v_k\,dt}\right\rceil.
\]

对任意数字：

```text
movement_intervals = 所有基元 intervals 之和
movement_samples   = movement_intervals + 1
```

因此同一速度条件下，数字总时间由总弧长决定；共享基元的物理长度、速度、采样点数和局部数组一致。

速度输入不根据某个数字自己的总时长重新计算，而沿用原仓库的三个条件编码：

```text
fast   -> 2/3
medium -> 1/3
slow   -> 0
```

这样相同速度条件对所有数字输入相同标量。

## 6.3 十个验证速度

沿用原仓库测试参考 movement time：

```text
reference_steps = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
```

对应物理速度：

\[
v_j=\frac{0.25}{\text{reference\_steps}_j\times0.01}.
\]

对应 speed scalar：

\[
s_j=1-\frac{\text{reference\_steps}_j}{150}.
\]

每个数字仍按自身弧长计算实际 movement intervals；不得让 speed scalar 随数字改变。

## 6.4 必须输出的时间审计

训练前输出：

- 每个数字的总弧长；
- 每个基元的弧长；
- 10×3 训练 movement intervals 和 duration；
- 10×10 验证 movement intervals 和 duration；
- 最短和最长 episode；
- 每个共享基元在不同数字中的 sample hash；
- 是否存在少于 2 个 interval 的基元。

如原仓库速度映射导致数值或 MotorNet 稳定性问题，先报告，不得自行改速度、缩放或插入 minimum-jerk。

---

## 7. 训练损失与 checkpoint：回到原仓库

基础训练损失严格使用：

\[
L=
L_{position,L1}
+L_{rate}
+L_{weight}
+L_{muscle}
+L_{simple\ dynamics}.
\]

其中位置项沿用原仓库 `losses.py::l1_dist`，对完整 trial 求平均。不得增加：

- velocity loss；
- acceleration loss；
- primitive boundary loss；
- movement-only mask；
- digit 等权二次汇总；
- pre/hold 单独权重；
- 终点特殊权重。

可以额外记录 movement error、corner error、endpoint error 作为诊断指标，但它们不得进入主损失或 checkpoint 选择。

checkpoint 选择必须移植 `reference/original_repo/train.py::do_eval` 的行为验证逻辑：

- 使用全部训练任务；
- 使用 32 个验证空间方向；
- 使用 10 个验证速度；
- 以平均位置 L1 最小为唯一选择依据；
- 不使用迁移、组合、隐藏活动相似性或几何指标选 checkpoint；
- 不再使用当前旧项目的自定义 behavior score。

为保证可复核，可固定验证 RNG 状态，但不得改变原验证条件分布或损失定义。

---

## 8. 两套基础模型必须完全分开

## 8.1 模型 A：完整十数字模型

训练任务：

```text
{0,1,2,3,4,5,6,7,8,9}
```

用途：

- 十数字基础行为；
- 代数组合实验；
- 后续共享计算和内部动力学分析。

不得用于留出迁移。

## 8.2 模型 B：九数字留出模型

训练任务：

```text
{0,1,2,3,4,6,7,8,9}
held out = 5
```

用途：

- 冻结动力学迁移到数字 5。

不得用于代数组合主实验。

两套模型必须：

- 独立实例化并独立训练；同一正式 seed 可以作为两套模型的配对 seed，但不得共享模型状态或 optimizer 状态；
- 使用相同的基础超参数和验证规则；
- 使用独立目录、配置、checkpoint 和日志；
- 不相互加载 checkpoint；
- 不使用组合系数初始化迁移 rule。

---

## 9. 代数组合：完全按照轨迹误差优化，不预设关系

## 9.1 基础模型

只使用完整十数字模型的已选 checkpoint。

冻结：

- recurrent weights；
- recurrent bias；
- 所有普通输入权重；
- 输出层；
- MotorNet；
- 现有 10 个 rule 输入权重。

只优化外部 rule 组合系数。

## 9.2 组合形式

对目标数字 `d`：

\[
q_d=\sum_{j\neq d}\alpha_j e_j,
\qquad \alpha_d=0.
\]

将 `q_d` 替换原 one-hot rule，其他 speed、go cue、空间提示和反馈均保持目标数字条件本身。

禁止：

- 预设使用 `{2,9,0}` 或任何其他数字子集；
- 预设系数符号；
- 预设平行四边形公式；
- 用正式结果反向修改几何；
- 同时优化网络参数。

## 9.3 按原仓库方式实现系数优化

移植 `experiments/exp_utils.py::composite_input_optimization` 的核心逻辑：

```yaml
optimizer: Adam
coefficient_learning_rate: 0.1
iterations: 250
network_noise: false
loss: full-trial position L1 only
initial_coefficients: 1.0
target_digit_coefficient: fixed 0.0
```

保持原仓库的重要性质：

- batch 中不同空间方向可以具有各自独立的系数行；
- 不强制所有方向共享同一组系数；
- 不加 L1 稀疏、范数约束或符号约束；
- 每次迭代从相同网络 checkpoint 和零 RNN 初态做闭环 rollout；
- 保存最小轨迹误差对应的系数和 rollout。

原仓库组合分析使用的评估条件应作为第一版默认复现：

```text
batch_size = 8
从 32 个验证方向中等间隔取 8 个方向：indices [0,4,8,12,16,20,24,28]
validation speed index = 9
custom delay = 150
```

环境必须支持组合实验的 `custom_delay=150`，但基础训练 delay 仍只有 `[25,50,75]`。

实现应支持对 0–9 任意目标数字运行；正式批处理默认对全部 10 个目标数字运行，避免事后只挑选成功数字。

## 9.4 结果解释边界

组合成功只说明：

> 其他已学习 rule 输入的线性组合能够重新调用冻结网络完成目标数字。

系数中是否自然出现稳定的代数关系，只能在优化完成后跨方向、跨种子做事后分析；不得事前把某条公式写入优化目标。

---

## 10. 留出迁移：独立九数字模型，只学习数字 5 的新 rule

## 10.1 起点

只使用九数字留出模型的已选 checkpoint。

## 10.2 冻结规则

冻结：

- recurrent weights 和 bias；
- 输出层；
- speed、go cue、空间提示、视觉反馈和本体反馈输入权重；
- 已有 9 个数字的 rule 向量；
- MotorNet。

只允许训练数字 5 对应的一个新 rule 输入向量。

不要直接沿用原仓库会替换整块 rule 输入矩阵的旧 `add_new_rule_inputs` 实现。应实现列级冻结，二选一：

1. 单独的 `new_rule_5` 参数，并在前向时只替代第 5 列；或
2. 对原输入矩阵第 5 列使用严格梯度 mask，并验证其他列逐位不变。

迁移开始前显式重新初始化数字 5 rule，不使用旧基础训练中从未激活列的残留值。

## 10.3 迁移训练

- 训练数据只包含数字 5；
- 覆盖 8 个训练空间方向、3 个训练速度和 3 个训练 delay；
- batch 采样方式与基础训练一致；
- 优化器、学习率和梯度裁剪沿用原仓库迁移入口；
- 迁移损失只使用完整 trial 的位置 L1，不加入基础训练正则；
- checkpoint 只按数字 5 的行为验证 L1 选择；
- 不使用任何代数组合系数初始化或正则化新 rule。

迁移后同时评估：

- 数字 5 的 32 方向×10 验证速度；
- 原九个数字的完整行为；
- 参数差异审计。

除 `new_rule_5` 外，任何参数发生变化均视为实现错误。

---

## 11. 必须删除或更新的旧断言

删除：

```text
forward/reverse 轨迹反转断言
3 是 8 的连续前缀
8[:T3] == 3
0 是 9 的逐点前缀
9[:T0] == 0
2 与 5 在世界坐标中逐点相同
minimum-jerk 端点速度为 0
每个 primitive 边界速度为 0
预设 r5-r2 ≈ r9-r0
```

替换为：

```text
所有 H 使用同一 canonical H_right；H_left 是其严格时间反向，长度和采样数一致
所有 V 使用同一 canonical V_down；V_up 是其严格时间反向，长度和采样数一致
所有 D 的几何长度和角度一致；D_down 是 D_up 的时间反向/符号反向
所有椭圆和半椭圆使用相同 a,b
2 与 5 的半椭圆在去除刚性旋转和平移后相同
0、6、9 的完整竖直椭圆周长一致
3 和 8 使用相同标准水平椭圆尺寸，但无前缀关系
0 和 9 使用相同标准竖直椭圆尺寸，但无前缀关系
2 和 6 的指定连接满足数值切向条件
```

---

## 12. 测试和审计要求

## 12.1 几何单元测试

至少覆盖：

1. 每个数字规定方向起点为 `(0,0)`；
2. 所有数组有限，无 NaN/Inf；
3. 每个基元连接位置连续；
4. 连接点只保留一个样本；
5. H、V、D 的长度在严格容差内一致；
6. D 的角度为 `63.43494882292201°`，允许方向反转；
7. 所有椭圆半轴均为同一 `a,b`；
8. 数字 0：上顶点起笔、逆时针完整环；
9. 数字 2：A/B/C 点关系正确，`C` 点切向与 `D_down` 同向；
10. 数字 3：两个标准右半椭圆，经颈点连接；
11. 数字 5：标准 H、V、水平右半椭圆；
12. 数字 6：先 D_up，后逆时针完整椭圆，切向连续；
13. 数字 8：半上环逆时针→完整下环顺时针→半上环逆时针；
14. 数字 9：右顶点顺时针完整环→V_down；
15. 明确验证 `3` 不是 `8` 前缀；
16. 明确验证 `0` 不是 `9` 前缀；
17. 8/32 个空间旋转保持弧长不变；
18. 所有空间方向均从同一 MotorNet 起笔锚点开始。

## 12.2 时间和速度测试

- 相同共享基元在相同速度下 sample count 相同；
- speed scalar 只由 speed condition 决定，不由 digit 决定；
- movement duration 随总弧长增长；
- 快速条件 duration < 中速 < 慢速；
- 无 minimum-jerk 函数调用；
- hard corner 附近允许离散速度方向突变；
- 2、6 相切点的前后离散切向近似一致。

## 12.3 环境测试

- observation 维度始终为 28；
- 输入切片顺序正确；
- rule 全 trial 恒定；
- speed/vis 在 stable 为 0；
- go cue 从 movement 开始；
- 不含 primitive、endpoint sequence 或 action history；
- batch 内 digit/speed/delay 一致，方向可不同；
- 基础训练、验证和 custom delay=150 均无 off-by-one；
- movement 第一和最后目标点均被实际监督。

## 12.4 模型冻结测试

组合实验：

- 所有网络参数 before/after bitwise 相同；
- 只有外部系数有梯度。

迁移实验：

- 只有数字 5 rule 向量变化；
- 其他参数逐位相同；
- 原九数字在迁移前后输出一致到确定性数值容差。

## 12.5 工作空间审计

重新对至少以下全集做 MotorNet 可达性和关节限位检查：

```text
10 digits × 32 spatial directions
```

速度不改变几何，但需对最长/最快、最长/最慢 episode 做动态 smoke test。输出：

- Cartesian bbox；
- radial reach margin；
- joint angle margin；
- inverse/forward kinematics consistency；
- 最危险数字和方向；
- 是否触碰限位。

不得为了通过审计单独修某个数字。若失败，只能报告并等待统一全局缩放决策。

---

## 13. 新配置与运行入口

创建清晰分离的配置，建议：

```text
configurations/digit_original_protocol_geometry.json
configurations/digit_original_protocol_full10_dev42.json
configurations/digit_original_protocol_heldout5_dev42.json
configurations/digit_original_protocol_composition.json
configurations/digit_original_protocol_transfer5.json
```

创建独立 runner，建议：

```text
server/run_digit_original_protocol_geometry_audit.sh
server/run_digit_original_protocol_full10_dev42.sh
server/run_digit_original_protocol_heldout5_dev42.sh
server/run_digit_original_protocol_composition.sh
server/run_digit_original_protocol_transfer5.sh
```

命名可按现有工程规范调整，但不得把两套基础模型写进同一个模糊入口。

每次运行必须保存：

- Git HEAD；
- submodule HEAD；
- 原仓库 reference commit；
- 配置全文和 SHA-256；
- 随机种子；
- 环境版本；
- 完整训练/验证日志；
- best checkpoint 和 final continuation checkpoint；
- optimizer state；
- 文件清单及 hash。

---

## 14. 实施顺序和停止线

按以下顺序执行，不得跳步：

### 阶段 A：协议清理

- 标记旧协议失效；
- 创建新配置 schema；
- 移除 reverse/minimum-jerk/速度损失/预设代数关系；
- 不运行训练。

### 阶段 B：几何与时间层

- 实现第 5、6 节；
- 生成 10 个规定方向图；
- 生成 10×8 训练方向图；
- 输出带起点、箭头、primitive 边界的审查图；
- 完成几何、时间和工作空间审计。

### 阶段 C：环境与 28 维输入

- 恢复原空间方向条件；
- 通过环境单元测试和短闭环 smoke test；
- 对照原仓库列出逐项一致/必要差异表。

### 阶段 D：训练、组合、迁移代码

- 实现完整十数字基础训练；
- 实现独立九数字基础训练；
- 实现通用组合系数优化；
- 实现只训练新数字 5 rule 的迁移；
- 完成冻结测试。

### 阶段 E：交付审查

在任何昂贵训练前停止，提交：

1. 修改文件清单；
2. 与原仓库的逐项一致性表；
3. 所有必要差异及理由；
4. 10 个规定方向轨迹图；
5. 10×8 训练方向轨迹图；
6. 几何/时间/工作空间审计报告；
7. 全部测试结果；
8. 两套模型与组合/迁移入口说明；
9. 更新后的 `NEXT_SESSION_HANDOFF.md`；
10. 明确声明尚未开始 seed-42 新协议训练。

未经用户审查，不得：

- 启动 75,000 更新训练；
- 启动正式种子；
- 自动调整速度、缩放、损失或网络规模；
- 根据生成图自行“美化”某个数字；
- 添加新的科学分析或指标作为硬门禁。

---

## 15. 完成标准

本轮代码改造只有在以下条件全部满足时才算完成：

- 数字几何严格符合第 5 节；
- 所有直线和椭圆共享约束通过测试；
- 只有一种规定笔顺；
- 8/32 空间方向恢复；
- 3/10 速度条件恢复并按弧长决定实际时长；
- 28 维输入语义恢复；
- 原仓库基础损失与正则恢复；
- 完整十数字模型和九数字留出模型彻底分离；
- 组合实验无预设代数关系，只优化轨迹误差；
- 迁移实验只训练数字 5 新 rule；
- 旧 checkpoint 和旧协议不会被误用；
- 所有测试、审计和文档更新完成；
- 尚未启动昂贵训练。

# Codex 实施任务：直接基于原仓库改造数字书写项目

> 本文是本项目唯一的科学与实施规范。当前仓库的 Git 历史直接继承自原论文仓库：项目初始提交 `2c2fe1e021ab37c75db71ca8a8adce2d4b3618b0` 的直接父提交固定为原仓库基线 `105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33`。当前根目录中的 `config.py`、`envs.py`、`losses.py`、`model.py`、`train.py` 和 `utils.py` 就是该基线的原始训练核心，不存在需要继续维护或迁移的旧数字书写实现。
>
> `D:\digit_writing_project` 仅作为历史数字项目和审计证据的外部归档，不属于本项目代码基线，也不得从中复制数字几何、训练逻辑、配置、测试、checkpoint 或实验结果。实现时只读取本规范、当前根目录原始代码、原论文，以及 Git 基线提交中明确需要迁移的原始实验函数；不要泛读或引入无关分析。

---

## 1. 直接改造的核心原则

新项目只做一项主要科学改动：

> 将原仓库的 10 种运动任务替换为 0–9 的 10 种参数化数字轨迹。

除数字目标轨迹以及由“不同数字总弧长不同”直接导致的最小必要适配外，以下内容必须以当前根目录原始代码及基线提交 `105cd0c...` 的实际实现为基准：

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

禁止从外部历史数字项目引入下列设计，也禁止在本项目中重新实现：

- 正向/反向笔顺条件；
- 2 维“正笔顺/反笔顺”方向 one-hot；
- 每个基元独立 minimum-jerk 起停；
- 基元边界强制速度归零；
- 显式速度损失；
- movement、pre、hold 的重新加权损失；
- 预设任何 rule 代数公式，例如 `r5 = r2 + r9 - r0`；
- 让同一个九数字模型同时承担代数组合和留出迁移；
- 导入或使用任何历史数字项目的 seed-42 checkpoint、继续训练边界、optimizer state 或自定义行为分数。

---

## 2. 仓库基线和代码处理原则

### 2.1 当前仓库就是原始代码工作副本

- 当前活动分支必须以原仓库提交 `105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33` 为祖先，且项目初始提交 `2c2fe1e...` 必须直接建立在该提交之上；不得把历史数字项目分支合并进来。
- 当前根目录的六个原始训练核心文件在首次功能修改前必须与基线提交无差异。后续每一处改动都应能追溯到本规范中的明确要求。
- `mRNNTorch` 保持原仓库 gitlink `ac0c4f589eae37bbde63968912925de99232e306`，除非出现无法通过外层适配解决的确定性兼容问题，否则不修改子模块。
- 原始 `analysis/`、`experiments/` 和 `configurations/` 已从活动树移除以减少干扰，但仍完整存在于 Git 基线提交中。需要原始实现时使用 `git show 105cd0c:<path>` 精确读取单个文件，不得把整套无关分析恢复到活动树。
- MotorNet 版本兼容适配必须以最小补丁实现，并单独记录“API 兼容变化”和“科学协议变化”；兼容性不得改变任务、输入、损失、采样或模型语义。
- `D:\digit_writing_project\reference\original_repo` 只是外部不可变核对副本，禁止修改；真正的活动实现始终位于当前仓库根目录。

### 2.2 活动代码修改面

优先直接、最小化修改原仓库结构：

- `envs.py`：数字任务环境、28 维输入、trial 时序、空间方向和速度条件；
- `train.py`：batch 采样、基础训练、验证、checkpoint、两套基础模型及迁移入口；
- `losses.py`：保留原始 L1 位置损失与正则定义，仅在确有必要时做接口适配；
- `model.py`：保留原始 RNN/输出结构，只为数字 5 的列级冻结迁移增加最小接口；
- `config.py`：命令行和配置入口；
- `utils.py`：仅保留运行记录与复核所需的通用工具。

允许在当前原仓库中按阶段新增职责单一的 `digit_writing/` 包；它必须是本项目从零编写的实现，不能从外部历史数字项目复制。模块只在对应功能开始实现时创建：

```text
digit_writing/geometry.py     # 阶段 B：纯数字几何、弧长重采样、旋转和时间审计
digit_writing/experiments.py  # 阶段 D：组合系数优化与数字 5 迁移的薄入口
tests/                        # 各阶段只添加对应功能的测试
```

不要创建覆盖所有阶段的预置 schema 或与原仓库功能重复的大型框架。配置加载器只在实际代码需要时添加，并且只验证当前阶段真正消费的字段。`digit_writing/` 只能承载原仓库中不存在的数字专属逻辑；环境、模型、损失和训练主流程仍直接修改根目录原始文件。不要在对应新协议代码完成前创建可运行的训练 runner。

### 2.3 规范和文档优先级

- `PROJECT_PROTOCOL.md` 是唯一权威规范；本文中的科学定义优先于论文叙述，原始代码只决定本文明确要求“保持原仓库”的部分。
- `NEXT_SESSION_HANDOFF.md` 只记录当前实现状态、Git 身份和下一步，不得另行定义科学协议。
- `README.md` 只说明项目边界和运行状态。
- 轨迹审查材料在阶段 B 由代码生成；如需单独轨迹说明，只能从本文第 5、6 节派生，不得形成第二套几何定义。
- 历史数字项目的 brief、frozen spec、项目决定和旧测试不进入本仓库，也不得作为实现依据。

---

## 3. 原仓库实验设置：必须保持的部分

以当前根目录 `train.py`、`envs.py`、`losses.py`、`model.py` 的实际代码为基准；已从活动树移除的原始实验入口通过 `git show 105cd0c:<path>` 读取。不得根据论文文字或历史数字项目自行重写这些保持项。

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

如因当前 MotorNet/PyTorch API 需要兼容性适配，必须保持数值、作用对象和优化语义一致，并写出原始参数到兼容参数的逐项映射表。

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

必须保留原仓库的采样语义：

- 每个训练 batch 只选择一个 digit；
- 每个 batch 只选择一个 speed condition；
- 每个 batch 只选择一个 delay；
- batch 内每个样本独立随机选择 8 个训练空间方向之一；
- 10 个数字等概率采样；九数字基础模型中其余 9 个数字等概率采样；
- 同一 batch 内 episode 长度一致，因为 digit、speed、delay 一致。

数字环境实现不得把同一 batch 的所有样本锁定为一个空间方向。

---

## 4. 输入合同：保持原始 28 维，只使用空间方向

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

每个数字只定义一种规定笔顺。所谓 8 个方向，是将完整规定方向数字轨迹围绕共同起笔点做刚性旋转：

```text
training angles = 0°, 45°, 90°, ..., 315°
validation angles = 32 个等间隔角度，即 0°, 11.25°, ..., 348.75°
```

设规定方向局部轨迹为 `p(t)`，且 `p(0)=0`，空间方向为 `theta`：

\[
p_\theta(t)=p_{anchor}+R(\theta)p(t).
\]

2 维 `vis_inp` 保持原仓库的空间目标提示语义：

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

参数化尺寸表和本文公式是唯一几何规范；用户提供的手绘图只用于说明笔顺和拓扑，不作为像素级参考，也不得引入外部字体或参考图拟合。当前原始仓库不存在数字轨迹实现，以下数值由本文直接冻结，必须从零实现：

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
- 引入 `lobe_rx/lobe_ry` 或其他数字特异叶瓣尺寸。

新建的几何配置和 JSON 不得包含：

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

不得创建专用小叶瓣参数或函数。数字 3 与数字 8 不具有连续前缀关系。

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
2. 沿 `D_down` 到达竖直长轴椭圆左上侧的切点；
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

令切向与 `D_down` 同向，切点相位为：

\[
\phi_6=
\operatorname{atan2}
\left(
\frac{D_{dx}}{b},
-\frac{D_{dy}}{a}
\right)
\approx2.55359005004\ \mathrm{rad}
\approx146.309932474^\circ.
\]

令切点 `C6 = E_v(phi6)`，自由起点为：

\[
S_6=C_6+(D_{dx},D_{dy}).
\]

轨迹：

```text
S6 -> C6：D_down
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

不得创建四个数字特异小叶瓣函数。数字 3 不是数字 8 的连续前缀。

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

数字 9 与数字 0 只共享椭圆的形状和大小，不共享起点、方向或逐采样点前缀。不得创建 `9[:T0] == 0` 一类断言。

---

## 6. 时间参数化和速度条件

## 6.1 线性弧长推进，不数学消除速度突变

目标生成器只要求位置连续 `C0`：

- 相邻基元首尾坐标必须一致；
- 不做圆角化；
- 不插入停顿；
- 不强制边界速度归零；
- 不使用 minimum-jerk；
- 锐角和换向点允许目标速度方向突变；
- 2 和 6 的相切连接应自然保持几何切向一致。

数字轨迹模块必须建立高分辨率弧长查表，并直接使用线性弧长推进：

```python
target_arc = np.linspace(0.0, arc_length, n_intervals + 1)
```

本仓库没有需要保留的历史数字时间参数化代码；不得从外部历史项目复制 `minimum_jerk` 实现。

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

## 7. 训练损失与 checkpoint：严格保持原仓库语义

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

checkpoint 选择必须保持当前根目录 `train.py::do_eval` 的行为验证逻辑：

- 使用全部训练任务；
- 使用 32 个验证空间方向；
- 使用 10 个验证速度；
- 以平均位置 L1 最小为唯一选择依据；
- 不使用迁移、组合、隐藏活动相似性或几何指标选 checkpoint；
- 不得导入历史数字项目的自定义 behavior score。

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

通过 `git show 105cd0c:experiments/exp_utils.py` 读取并移植 `composite_input_optimization` 的核心逻辑；不要恢复整个原始 `experiments/` 目录：

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

不要直接复用基线提交中会替换整块 rule 输入矩阵的 `add_new_rule_inputs` 实现。应在当前 `model.py` 中实现列级冻结，二选一：

1. 单独的 `new_rule_5` 参数，并在前向时只替代第 5 列；或
2. 对原输入矩阵第 5 列使用严格梯度 mask，并验证其他列逐位不变。

迁移开始前显式重新初始化数字 5 rule，不使用九数字基础训练期间从未激活列的残留值。

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

## 11. 新项目禁止引入的断言与必须建立的断言

本项目从原始运动任务代码直接起步，不存在需要清理的历史数字断言。实现和测试中禁止引入：

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

必须新建以下断言：

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
12. 数字 6：先 D_down 到达左上切点，后逆时针完整椭圆，切向连续；
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

配置必须随对应实现逐步创建，不得在阶段 A 一次性预建和锁死全部结构：

- 阶段 B 只创建并冻结几何/时间配置；
- 阶段 C 在环境实际消费字段确定后补充环境相关配置；
- 阶段 D 才创建完整十数字、九数字留出、组合和迁移配置；
- 每份配置只包含对应代码实际读取的字段，不为未来功能预留抽象字段。

计划文件名为：

```text
configurations/digit_original_protocol_geometry.json
configurations/digit_original_protocol_full10_dev42.json
configurations/digit_original_protocol_heldout5_dev42.json
configurations/digit_original_protocol_composition.json
configurations/digit_original_protocol_transfer5.json
```

`full10` 与 `heldout5` 的配置和目录必须分离。用户已将 CPU 冻结为本项目的基准设备后端；从阶段 B 开始，项目实现、测试、审计和后续基准运行均在新的 AutoDL 实例上使用固定 CPU 环境执行并如实记录。CUDA 不属于基准结果谱系；如后续需要使用 CUDA，必须通过新的协议决定建立独立实验，不得与 CPU 基准混合。

仅当对应代码、配置和服务器测试均已完成后，才创建独立 runner：

```text
server/run_digit_original_protocol_geometry_audit.sh
server/run_digit_original_protocol_full10_dev42.sh
server/run_digit_original_protocol_heldout5_dev42.sh
server/run_digit_original_protocol_composition.sh
server/run_digit_original_protocol_transfer5.sh
```

命名可按原仓库简洁风格调整，但不得把两套基础模型写进同一个模糊入口。阶段 A 不创建任何 runner。

每次运行必须保存：

- Git HEAD；
- submodule HEAD；
- 原仓库基线提交 `105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33`；
- 配置全文和 SHA-256；
- 随机种子；
- 环境版本；
- 完整训练/验证日志；
- best checkpoint 和 final continuation checkpoint；
- optimizer state；
- 文件清单及 hash。

人工维护的全项目哈希清单不属于阶段 A 门禁；阶段 E 交付和每次实际运行归档时再从最终文件自动生成。

---

## 14. 实施顺序和停止线

按以下顺序执行，不得跳步：

### 阶段 A：确认原始基线和项目边界

- 断言 `105cd0c...` 是当前分支祖先、`2c2fe1e...` 的直接父提交确为该基线；
- 确认 `mRNNTorch` gitlink 为 `ac0c4f...`，六个原始训练核心文件与基线提交无差异；
- 明确历史数字项目仅为外部归档，活动树未导入其代码、配置、测试、checkpoint 或结果；
- 记录 full10 与 heldout5 必须独立训练、组合与迁移不得共享基础模型的边界；
- 不创建配置 schema、训练配置、runner 或阶段性人工哈希清单；
- 不运行训练。

阶段 A 是一次轻量身份检查，不是独立软件子项目。上述事实写入 README 和接续说明即可，无需为它创建运行时代码或单元测试。

### 阶段 B：几何与时间层

- 创建 `digit_writing/geometry.py`、对应的单一几何/时间配置和几何测试，从零实现第 5、6 节，不复制历史数字项目几何代码；
- 生成 10 个规定方向图；
- 按训练空间方向生成 8 张方向审查图，每张固定一个角度，并用 `2×5` 独立子图分别展示数字 0–9；
- 同一张方向图中每个数字只能出现一次，禁止把同一数字的 8 个方向叠加或并列在同一张图中；
- 每个数字子图均标出起点、运动箭头和 primitive 边界，并使用一致的等比例坐标轴；
- 完成几何、时间和工作空间审计。

### 阶段 C：环境与 28 维输入

- 在当前原始 `envs.py` 结构中保留空间方向条件并替换十种任务轨迹；
- 通过环境单元测试和短闭环 smoke test；
- 对照基线提交 `105cd0c...` 列出逐项一致/必要差异表。

### 阶段 D：训练、组合、迁移代码

- 在当前原始 `train.py` 和 `model.py` 上实现完整十数字基础训练；
- 实现与其完全分离的九数字基础训练；
- 从基线提交的 `experiments/exp_utils.py` 只迁移通用组合系数优化核心；
- 实现只训练新数字 5 rule 的迁移接口；
- 在各代码路径实际确定后创建 full10、heldout5、composition 和 transfer 配置，不预留未消费字段；
- 完成冻结测试。

### 阶段 E：交付审查

在任何昂贵训练前停止，提交：

1. 修改文件清单；
2. 与基线提交 `105cd0c...` 的逐项一致性表；
3. 所有必要差异及理由；
4. 10 个规定方向轨迹图；
5. 8 张按训练方向分开的 `2×5` 数字轨迹图；同一张图中每个数字只出现一次；
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
- 8/32 空间方向语义保持不变；
- 3/10 速度条件语义保持不变，并按数字弧长决定实际时长；
- 28 维输入语义保持不变；
- 原仓库基础损失与正则保持不变；
- 完整十数字模型和九数字留出模型彻底分离；
- 组合实验无预设代数关系，只优化轨迹误差；
- 迁移实验只训练数字 5 新 rule；
- 不存在历史数字 checkpoint、配置或协议的导入路径；
- 六个原始训练核心文件的所有差异都能追溯到本规范；
- 所有测试、审计和文档更新完成；
- 尚未启动昂贵训练。

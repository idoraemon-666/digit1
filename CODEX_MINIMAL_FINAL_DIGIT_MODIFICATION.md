# Codex 最简实施指令：将当前项目更新为最终数字轨迹协议

> 本文件与随附的 `digit_geometry_final.py` 共同构成本轮修改的唯一实施依据。  
> `digit_geometry_final.py` 是数字几何、物理尺度、速度与时间采样的唯一权威来源。  
> 本文件只覆盖当前 `PROJECT_PROTOCOL.md` 中与本轮最终决定冲突的部分；其余原仓库模型、环境、训练、迁移及记录要求继续保持不变。

---

## 0. 工作目标与边界

当前仓库已经完成过原有流程和大部分审查。本轮不要重建项目、不要恢复无关目录、不要重复历史来源审查，也不要重新设计网络。

只做以下必要修改：

1. 将最终 0–9 数字轨迹写入当前项目；
2. 保留原仓库参考速度及可变 movement 时长；
3. 将位置损失改成分阶段归一化位置损失；
4. 将代数组合限定为当前最匹配的五任务子集；
5. 保持完整十数字模型和留出数字 5 模型完全独立；
6. 只执行与本次改动直接相关的测试、审计和短闭环 smoke test；
7. 在用户审查前，不启动 75,000 updates 正式训练。

禁止：

- 新增网络层、轨迹编码器、primitive ID 或笔画 ID；
- 改变 28 维输入合同；
- 改变 RNN、MotorNet、优化器、基础正则项及其系数；
- 为了通过测试单独缩放某个数字；
- 根据组合结果反向挑选数字或修改轨迹；
- 导入历史数字项目 checkpoint、optimizer state 或训练结果；
- 重做已经完成且不受本轮改动影响的全仓库审计。

---

## 1. 先检查当前实现，再做最小补丁

开始前只做轻量检查：

1. 阅读当前 `PROJECT_PROTOCOL.md`；
2. 查看当前 `git status` 和 `git diff`；
3. 找到当前数字几何、环境、位置损失、组合和迁移入口；
4. 复用已经存在且通过测试的代码，不重写可复用模块；
5. 输出一份“计划修改文件清单”，然后直接实施，不等待额外确认。

不要重复：

- Git 祖先和原仓库 provenance 全量审计；
- 历史数字项目安全审计；
- 与本次最终几何无关的旧图和旧测试；
- 已经完成的训练结果复核。

---

## 2. 最终几何：以随附脚本为唯一来源

### 2.1 文件处理

将随附的：

```text
digit_geometry_final.py
```

整合到当前仓库中。推荐做法：

```text
digit_writing/geometry.py
```

要求：

- 保留脚本中的全部几何常量、构造公式和采样语义；
- 可以把 CLI 绘图和审计入口留在原脚本，或拆成薄 wrapper；
- 仓库中不得同时存在第二套数字公式；
- `envs.py` 必须直接调用这一实现，而不是复制参数；
- 若移动文件，保证原始脚本仍可作为审计入口运行。

### 2.2 全局物理尺度

最终使用：

```text
digit 0 physical height = 0.06153846153846154 m
global scale            = 0.06410256410256411 m / design unit
```

所有数字统一乘同一个全局尺度。

禁止：

- 数字特异缩放；
- 为工作空间单独缩小某个数字；
- 再次根据人类真实书写速度缩放整体轨迹。

### 2.3 最终曲线与共享关系

当前实际曲线家族：

```text
A：数字 2 上部、数字 3 上部
B：数字 3 下部、数字 5 下部
独立 8：只用于数字 8，不与其他数字共享
4:3 椭圆：数字 0
5:4 椭圆：数字 6、9
```

旧曲线 C 不再使用，应从活动配置、测试和文档中移除。

共享定义：

- 平移和旋转后仍属于同一共享曲线；
- 镜像不算同一共享曲线；
- 同一共享曲线不得在不同数字中单独拉伸或修改控制点；
- 共享 A/B 在相同速度条件下必须具有相同弧长、interval 数和刚性对齐后的采样形状。

### 2.4 最终 0–9 结构

必须逐点遵守脚本，不再凭文字重新推导。摘要如下：

```text
0：4:3 竖直椭圆，上端起笔，逆时针一周
1：竖直向下
2：A 旋转 55° → 55°向左下斜线 → 向右横线
3：上 A → 下 B
4：向左横线 0.72 → 55°向右上斜线 → 向下竖线
5：向左横线 → 向下竖线 → B
6：63.435°切向斜线 → 5:4 椭圆逆时针一周
7：向右横线 → 63.435°向左下斜线
8：UJI 宽高比统计约束的独立轴对称 Gerono 型轨迹
9：5:4 椭圆从右端顺时针一周 → 向下竖线
```

---

## 3. 速度与时间参数化

### 3.1 继续保留当前三个训练速度

这些速度是原仓库直线 reach 派生的名义路径速度，不解释成人类真实书写速度：

```yaml
dt_s: 0.01

train_speed_mps:
  fast:   0.5000000000
  medium: 0.2500000000
  slow:   0.1666666667

train_speed_scalar:
  fast:   0.6666666667
  medium: 0.3333333333
  slow:   0.0
```

验证速度继续由：

```text
reference_steps = [50,60,70,80,90,100,110,120,130,140]
v = 0.25 / (reference_steps * 0.01)
speed_scalar = 1 - reference_steps / 150
```

得到。

### 3.2 每个数字时长由总弧长决定

每个 segment：

\[
n_{p,k}
=
\left\lceil
\frac{L_p}{v_k\Delta t}
\right\rceil .
\]

要求：

- 每段独立计算 interval；
- 使用线性弧长重采样；
- 连接点只保留一次；
- 不使用 minimum-jerk；
- 不插入停顿；
- 不强制转角处速度归零；
- 同一共享 segment 在同一速度下 interval 数相同。

数字总时长允许不同。这是最终决定，不再把所有数字重采样到统一总时长。

### 3.3 最终弧长参考值

脚本当前输出的总弧长约为：

| 数字 | 总弧长/m |
|---:|---:|
| 0 | 0.170026832 |
| 1 | 0.061538462 |
| 2 | 0.145800867 |
| 3 | 0.132654179 |
| 4 | 0.171295140 |
| 5 | 0.126596902 |
| 6 | 0.185072096 |
| 7 | 0.109827733 |
| 8 | 0.206257428 |
| 9 | 0.194046455 |

若实现输出与脚本超过数值容差，视为实现错误，不得自行接受。

---

## 4. 28 维输入和 trial 时序保持不变

输入顺序继续为：

```text
[0:10]   digit rule one-hot
[10:11]  speed scalar
[11:12]  go cue
[12:14]  2-D spatial direction cue
[14:16]  fingertip visual feedback
[16:28]  muscle proprioceptive feedback
```

保持：

```yaml
stable_steps: 25
delay_steps: [25, 50, 75]
hold_steps: 25
training_angles: 8
validation_angles: 32
```

时序：

- rule：全 trial 恒定；
- speed、spatial cue：stable 为 0，从 delay 开始保持；
- go cue：movement 起始变为 1；
- hidden target：
  - stable/delay 保持起点；
  - movement 逐点跟踪完整数字；
  - hold 保持终点。

同一 batch：

- digit、speed、delay 相同；
- 每个样本方向可不同；
- episode 长度一致。

---

## 5. 位置损失：改为分阶段归一化

### 5.1 只修改位置项

保持原仓库坐标绝对误差形式，不改为空间平方误差。

先计算每个时间步的位置误差：

\[
e_t
=
|x_t-x_t^\*|+|y_t-y_t^\*|.
\]

分别在四个阶段内部求平均：

\[
L_{\mathrm{stable}}
=
\operatorname{mean}_{t\in stable} e_t,
\]

\[
L_{\mathrm{delay}}
=
\operatorname{mean}_{t\in delay} e_t,
\]

\[
L_{\mathrm{movement}}
=
\operatorname{mean}_{t\in movement} e_t,
\]

\[
L_{\mathrm{hold}}
=
\operatorname{mean}_{t\in hold} e_t.
\]

最终位置项固定为：

\[
\boxed{
L_{\mathrm{position}}
=
0.1L_{\mathrm{stable}}
+
0.1L_{\mathrm{delay}}
+
0.6L_{\mathrm{movement}}
+
0.2L_{\mathrm{hold}}
}
\]

等价写法：

\[
L_{\mathrm{position}}
=
0.2L_{\mathrm{pre}}
+
0.6L_{\mathrm{movement}}
+
0.2L_{\mathrm{hold}},
\]

其中：

\[
L_{\mathrm{pre}}
=
0.5L_{\mathrm{stable}}+0.5L_{\mathrm{delay}}.
\]

不得按照各阶段实际时间步数再次加权。

### 5.2 其他损失完全不变

总损失继续为：

\[
L=
L_{\mathrm{position}}
+L_{\mathrm{rate}}
+L_{\mathrm{weight}}
+L_{\mathrm{muscle}}
+L_{\mathrm{simple\ dynamics}}.
\]

要求：

- 原有系数不变；
- 原有作用对象不变；
- 原有时间范围不变；
- 神经活动、肌肉活动和 simple dynamics 仍按原仓库完整 trial 语义计算；
- 权重正则不变。

### 5.3 统一用于训练、验证、组合和迁移

为避免优化目标不一致：

- 基础训练：使用上述 phase-normalized position loss；
- checkpoint：使用验证集平均 phase-normalized position loss 最小；
- 代数组合系数优化：只使用目标 trial 的 phase-normalized position loss；
- 数字 5 迁移：只使用 phase-normalized position loss，不加基础训练正则；
- 同时记录：
  - full-trial position L1；
  - stable mean L1；
  - delay mean L1；
  - movement mean L1；
  - hold mean L1；
  - endpoint error；
  - hold drift。

checkpoint 不得使用隐藏活动、组合结果或迁移结果。

### 5.4 必须防止 off-by-one

测试必须明确验证：

- stable、delay、movement、hold 切片互斥且无遗漏；
- movement 第一个和最后一个目标点都被监督；
- hold 从 movement 结束后的下一步开始；
- 不把连接点重复计入两个阶段；
- 可变 movement 长度下 mask 正确。

---

## 6. 两套基础模型继续完全分离

### 模型 A：full10

```text
digits = {0,1,2,3,4,5,6,7,8,9}
```

用途：

- 基础行为；
- 共享结构分析；
- 代数组合。

### 模型 B：heldout5

```text
digits = {0,1,2,3,4,6,7,8,9}
held_out = 5
```

用途：

- 冻结动力学后只学习数字 5 的新 rule。

要求继续保持：

- 独立模型；
- 独立 optimizer；
- 独立 checkpoint；
- 独立日志目录；
- 不相互加载状态；
- 组合系数不用于迁移初始化。

迁移时只有数字 5 新 rule 向量可变化，其他参数必须逐位不变。

---

## 7. 代数组合：限定为五任务子集

### 7.1 固定任务组

根据最终弧长，当前最匹配的五任务子集固定为：

```text
G = {0, 4, 6, 9, 8}
```

在原组合默认验证速度 index 9 下，movement intervals 约为：

```text
0:  96
4:  97
6: 105
9: 110
8: 116
```

该集合必须在看到组合结果前固定，禁止事后更换。

### 7.2 运行五次 leave-one-out

对每个目标：

```text
target d ∈ G
sources = G \ {d}
```

组合形式：

\[
q_d
=
\sum_{j\in G,\ j\neq d}\alpha_j e_j.
\]

要求：

- 只使用另外四个 rule；
- 目标 rule 系数固定为 0；
- composite rule 全 trial 恒定；
- rollout 使用目标数字自己的完整 trial；
- movement 使用目标数字自己的完整 movement 时长；
- 不执行、截取或拼接来源数字的轨迹；
- 不要求来源数字的总时长与目标完全相同；
- 网络、输出层和 MotorNet 全部冻结；
- 不加稀疏、符号、范数约束。

保留原仓库组合参数：

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

保存每个目标、每个方向的最佳系数和最佳 rollout。

结果必须完整报告五个目标，不能只展示成功数字。

---

## 8. 只做必要审查

### 8.1 必须执行

#### A. 几何与时间

运行：

```bash
python digit_geometry_final.py \
  --self-test \
  --output-dir <repo>/artifacts/final_digit_geometry
```

保存：

- `digit_geometry_audit.json`
- `digit_arc_lengths.csv`
- `digits_0_to_9.png`
- `curves_A_B.png`

并验证：

- 0–9 起点均为 `(0,0)`；
- 所有数组有限；
- 连接点连续；
- A、B 轴对称；
- A、B 共享关系通过刚性对齐；
- 2/4 为 55°；
- 6/7 为 63.435°；
- 0 为 4:3 椭圆；
- 6/9 为 5:4 椭圆；
- 8 左右和上下轴对称；
- 同一共享片段在相同速度下 interval 数一致。

#### B. 位置损失

新增针对性单元测试：

- 四阶段 mask；
- 四阶段内部 mean；
- 固定权重 `0.1/0.1/0.6/0.2`；
- 可变 movement 时长不改变 movement 总权重；
- 完整 trial L1 只记录、不参与主 checkpoint。

#### C. 环境

仅重跑与新几何/可变长度相关的测试：

- observation 始终 28 维；
- batch 内 digit/speed/delay 相同；
- batch 内方向可不同；
- variable movement length 无 off-by-one；
- custom delay 150 可运行；
- 10 个数字均可 rollout。

#### D. 工作空间

因为几何已改变，必须重新运行：

```text
10 digits × 32 directions
```

检查：

- Cartesian bbox；
- radial reach margin；
- joint angle margin；
- inverse/forward kinematics consistency；
- 是否触碰关节限位。

若失败，只报告，不得自行缩放。

#### E. 短闭环 smoke test

不训练 75,000 updates。只覆盖高风险条件：

```text
digit 1 fast       # 最短任务
digit 4 fast       # 长直线与尖角
digit 6 fast       # 长斜线 + 小闭环
digit 8 fast/slow  # 最长闭环与最长时序
digit 9 fast       # 闭环 + 长尾
```

检查：

- 无 NaN/Inf；
- episode 正常结束；
- MotorNet 不越界；
- loss 各项数量级可记录；
- 四阶段位置误差可正确输出。

#### F. 冻结测试

组合：

- 网络参数 before/after bitwise 相同；
- 只有外部组合系数有梯度。

迁移：

- 只有数字 5 新 rule 变化；
- 其他参数逐位不变；
- 原九数字确定性输出保持一致。

### 8.2 不需要重复

除非新修改直接破坏，不要重做：

- 原始 Git provenance 全审计；
- 历史项目目录安全审计；
- 原仓库所有旧分析脚本；
- 全量多 seed 训练；
- 75,000 updates 正式训练；
- 已经完成且与本次改动无关的旧结果复核。

---

## 9. 最小配置更新

在现有配置上只修改实际消费的字段：

```yaml
geometry_source: digit_geometry_final.py
global_scale_m_per_unit: 0.06410256410256411

position_loss:
  type: phase_normalized_l1
  stable_weight: 0.1
  delay_weight: 0.1
  movement_weight: 0.6
  hold_weight: 0.2

composition:
  digit_group: [0, 4, 6, 9, 8]
  leave_one_out: true
```

保留：

- 网络结构；
- 28 维输入；
- optimizer；
- learning rate；
- batch size；
- max updates；
- gradient clipping；
- 原有正则项及系数；
- 3 个训练速度；
- 10 个验证速度；
- 8/32 个方向；
- stable/delay/hold；
- full10/heldout5 分离。

删除或停用与最终协议冲突的字段：

- 旧曲线 C；
- 旧椭圆和旧数字参数；
- `minimum_jerk`；
- 统一总 movement steps；
- 人类真实书写速度；
- 全十数字组合默认批处理；
- full-trial 位置损失作为主位置项。

---

## 10. 交付内容和停止线

完成代码后提交：

1. 修改文件清单；
2. 与本文件逐项对应的差异说明；
3. 最终几何和时间审计；
4. 0–9 最终轨迹图；
5. 四阶段位置损失测试结果；
6. 环境及短闭环 smoke test；
7. 10×32 工作空间审计；
8. 组合和迁移冻结测试；
9. full10、heldout5、composition、transfer 的运行入口；
10. 更新后的 `PROJECT_PROTOCOL.md` 和接续说明。

明确声明：

```text
尚未启动 75,000 updates 正式训练。
```

用户审查前禁止：

- 自动修改全局尺度；
- 自动修改速度；
- 自动修改损失权重；
- 自动修改网络规模；
- 启动正式训练；
- 根据 smoke test 结果擅自美化某个数字。

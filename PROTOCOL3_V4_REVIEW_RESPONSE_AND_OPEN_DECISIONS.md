# Protocol3 v4 审查问题答复与待确认事项

审查来源：`CODEX_PROTOCOL3_V4_REVIEW_ISSUES.md`
答复对象：`CODEX_PROTOCOL3_FIXED_SPEED_MINIMAL_VALIDATION_AND_RETRAIN_v4.md`

## 0. 本文件的用途和边界

本文件用于逐条回答 Codex 的审查问题，并区分：

- **已确认**：可直接写入下一版实施规范；
- **有依据的建议**：技术上合理，但属于新的工程判据，需要在正式规范中明确标注；
- **暂时不能确定**：附件和现有证据不足，必须由用户确认或通过后续预检决定。

用户后续已明确授权：

- 在当前目录内创建并切换 `codex/digit-writing-original-protocol3` 分支；
- 为 Protocol3 修改当前项目代码，同时保持 protocol2 行为隔离；
- 完成本地验证后准备 Gate 1 / Gate 2 服务器包；
- Gate 1 通过后直接执行 Gate 2，无需再次请示；
- Gate 2 先验证 fast，满足规定条件时只允许整体回退一次到 medium。

仍未授权：

- 在 Gate 2 后尚未冻结最终工程阈值时启动正式 full10；
- full10 到达 5,000 updates 后自动续训；
- heldout5、composition、transfer5、旧 Stage E 或 Stage F。

除特别说明外，文件路径均按**实际仓库根目录**书写；在最小接管包中，对应源码位于 `formal_f760672/` 下。

---

# 1. 可以立即冻结的总体结论

## 1.1 协议不是“固定物理速度”，而是“固定片段时间表”

**状态：已确认。**

Protocol3 不应再称为 `fixed_speed`。

当前设计给同一类片段固定 movement intervals，例如所有直线在 fast reference 下统一为 30 intervals，但直线实际长度不同，因此它们的实际平均速度必然不同。

这与原始仓库的基本做法并不冲突：原始仓库的直线、半圆和 sinusoid 路径长度不同，但在同一 reference condition 下可以使用相同 movement time。因此该条件更准确的含义是：

```text
fixed_segment_timing
```

建议正式配置统一使用：

```json
{
  "timing_mode": "fixed_segment_timing",
  "selected_reference_steps": 50,
  "speed_scalar_semantics": "frozen_reference_condition_label"
}
```

其中：

- `selected_reference_steps=50` 表示选用 fast reference 时间倍率；
- `selected_reference_steps=100` 表示选用 medium reference 时间倍率；
- observation 中的 speed scalar 保留，以维持 28 维接口和原仓库输入结构；
- speed scalar 是一个**冻结的 reference-condition 标签**，不代表每个片段具有相同的真实物理速度；
- 审计和报告必须另外输出每个片段的真实平均速度、峰值速度、加速度和 jerk。

下一版文档标题建议改为：

```text
Protocol3 固定片段时间表版
```

不得继续用“固定速度版”描述科学含义。

---

## 1.2 protocol2 和 protocol3 必须通过配置显式分派

**状态：已确认。**

Codex 指出的冲突成立。

当前：

- `digit_writing/digit_geometry_final.py::Segment.points_m()` 直接使用唯一的 `GLOBAL_SCALE_M_PER_UNIT`；
- `digit_writing/digit_geometry_final.py::sample_digit()` 只支持按物理速度采样；
- `digit_writing/geometry.py::load_geometry_config()` 强制配置尺度等于该全局常量；
- protocol2 和未来 protocol3 会共同经过这些入口。

因此不得直接修改全局常量或把 `sample_digit()` 全局替换成新时间规则。

下一版实施规范应明确：

1. 设计单位几何保持唯一且不变；
2. 物理尺度由 `GeometryConfig.global_scale_m_per_unit` 显式传入；
3. 时间模式由 `GeometryConfig.timing_mode` 显式传入；
4. protocol2 继续走旧路径：
   ```text
   timing_mode = physical_speed_arclength
   scale = 0.06410256410256411
   training_reference_steps = [50,100,150]
   validation_reference_steps = [50,60,...,140]
   ```
5. protocol3 走新路径：
   ```text
   timing_mode = fixed_segment_timing
   scale_multiplier = 2.5，必要时回退 2.25
   selected_reference_steps = 50 或 100
   ```
6. protocol2 的配置、目标轨迹、时间步数、observation、训练、验证、checkpoint 和测试行为保持不变。

“protocol2 只读”的准确含义应改为：

> 可以为实现协议分派而修改共享源码，但 protocol2 的配置和所有可观察行为必须保持不变，并由回归测试证明。

建议在改造前先生成 protocol2 的冻结回归清单，至少覆盖：

分别生成两个独立清单，不把 training 和 validation reference conditions
相互做笛卡尔积：

```text
training：10 digits × 3 training reference conditions = 30 conditions
validation：10 digits × 10 validation reference conditions = 100 conditions
```

记录：

- movement intervals；
- segment boundaries；
- path 数组 SHA-256；
- canonical template SHA-256；
- speed scalar；
- 28 维 observation 结构；
- phase bounds。

改造后必须逐项一致。

---

## 1.3 Gate 2 只能选择时间条件，不能单独证明 full10 可行

**状态：已确认。**

Gate 2 的三个单条件短过拟合模型只能证明：

- 简单直线在该时间条件下可以学习；
- 一个多段数字可以学习；
- 一个长曲线数字可以学习；
- fast 或 medium 哪个更适合作为候选 reference condition。

它不能证明：

- 十个数字可以在一个网络内联合学习；
- 8 个训练方向都可学习；
- 3 个 delay 下稳定；
- digit 4/7 尖角条件可学习；
- digit 6/9 的连接结构可学习；
- 多任务干扰不会重新导致失败。

因此下一版规范必须写明：

```text
Gate 2 的功能 = 时间条件选择和基本可学习性检查
Gate 2 ≠ full10 联合训练成功的充分证据
```

为了避免额外训练一个无用模型，建议把 full10 的前 5,000 updates 定义为：

```text
Protocol3 full10 early joint-training gate
```

它是最终 75,000-update 模型的前半段，不是独立模型：

1. 从零启动 full10；
2. 完成 5,000 次 optimizer update；
3. 保存完整 continuation checkpoint；
4. 暂停；
5. 对十数字、方向和 delay 做联合行为审计；
6. 用户确认后，从同一模型、optimizer 和 RNG state 继续到 75,000。

这样能够检查联合训练风险，又不会浪费已经完成的 5,000 updates。

但需要明确：

> Gate 2 通过后只说明“允许进入 full10 的 5,000-update 联合试运行”，不能直接声明整个 full10 已经被证实可行。

---

## 1.4 正式 validation 应继续使用 32 个不同方向

**状态：已确认。**

不应把正式 validation 缩减成 8 个方向各重复 4 次。

当前环境：

- `testing=False`：使用 8-direction grid；
- `testing=True`：使用 32-direction grid；
- `train.py::do_eval()` 当前显式传入：
  ```python
  reach_conds=np.arange(0, 32)
  ```

Protocol3 建议保持：

### 训练

```text
8 个训练方向 × 每方向 4 个样本 = batch 32
```

### 正式 validation

```text
32 个不同 validation 方向
reach_conds = np.arange(32)
```

这不会增加 validation 的 batch 数，因为一个 batch 本来就是 32。

若需要从 32-direction grid 中指出训练方向，对应索引为：

```text
[0,4,8,12,16,20,24,28]
```

但正式 validation 应使用完整的 `0..31`，用于评估训练方向之间的角度泛化。

---

## 1.5 梯度比例和 cosine 不应作为独立硬停止条件

**状态：已确认。**

以下条件应从 hard stop 中删除：

```text
regularization_to_position_gradient_ratio >= 1
position_total_gradient_cosine <= 0
```

原因：

- 位置拟合变好后，位置梯度可能自然变小；
- 位置梯度接近 0 时，比值可能被极小分母放大；
- 任一梯度范数接近 0 时 cosine 不稳定或没有意义；
- 单个 checkpoint 的梯度快照不能证明某种状态“持续存在”；
- 行为良好的模型可能被错误停止。

梯度指标只作为诊断信息：

```text
position_gradient_norm
regularization_gradient_norm
regularization_to_position_gradient_ratio
position_total_gradient_cosine
```

建议的数值规则：

- 若任一参与 cosine 的梯度范数低于 `1e-12`，将 cosine 记为 `undefined`；
- 比值计算时必须同时记录原始分子、分母和是否触发小分母标志；
- 不允许把 `undefined` 自动解释为失败；
- 不允许仅凭一次负 cosine 停止训练。

可独立触发硬停止的条件保留为：

- NaN/Inf；
- IK/FK 失败；
- 工作空间或关节安全失败；
- checkpoint、optimizer state 或 RNG state 不完整；
- 代码、配置或 Git 身份不匹配；
- 输出目录冲突或证据链损坏。

是否因“行为长期没有改善”停止，应由预先冻结的行为判据决定，而不是梯度比例决定。

---

## 1.6 Protocol3 Gate 2 不是旧 Stage E，但仍属于新的诊断训练

**状态：已确认。**

下一版文档必须明确区分：

```text
protocol2 Stage E E1–E5：不执行
Protocol3 Gate 2：新定义的时间条件选择诊断
```

Gate 2 仍会训练临时模型，因此必须使用独立 run label，并得到明确授权。

建议标签：

```text
protocol3-gate2-scale2p50-ref50-o1-digit1
protocol3-gate2-scale2p50-ref50-o2-digit5
protocol3-gate2-scale2p50-ref50-o3-digit8
```

medium 回退时：

```text
protocol3-gate2-scale2p50-ref100-o1-digit1
...
```

Gate 2 结果和临时 checkpoint 不得写入旧 Stage E、旧 full10 或正式 full10 目录。

---

# 2. 需要在下一版规范中补齐的确定性定义

## 2.1 canonical hash 应拆成两种

**状态：定义可以冻结。**

建议记录：

```text
canonical_template_sha256
ordered_instance_sha256
```

### A. `canonical_template_sha256`

用途：证明不同任务调用的是同一个设计模板。

定义：

1. 基于**设计单位坐标**；
2. 不乘全局米制尺度；
3. 不包含任务的整体平移；
4. 不包含任务的整体旋转；
5. 不包含 trial 的空间方向旋转和 fingertip anchor；
6. 保留采样方向和点顺序；
7. 根据该片段在当前 reference condition 下的 intervals 做线性弧长采样；
8. 数组转换为：
   ```text
   dtype = little-endian float64 (`<f8`)
   layout = C contiguous
   shape = [N,2]
   ```
9. SHA-256 输入应包含 shape 和 dtype 头，再拼接数组 bytes。

例如：

```python
header = f"shape={points.shape};dtype=<f8;".encode("ascii")
payload = header + np.ascontiguousarray(points, dtype="<f8").tobytes()
sha256(payload)
```

要求相同：

- digit 2 / digit 3 的 `curve_A`；
- digit 3 / digit 5 的 `curve_B`。

digit 6 / digit 9 的椭圆按各自已冻结的设计轨迹独立生成。它们不属于本项目的
严格共享片段，不要求 `canonical_template_sha256` 相同，也不据此讨论计算复用。

该 hash 与 `2.5×` 或 `2.25×` 无关。

### B. `ordered_instance_sha256`

用途：记录某个数字内实际有序片段实例。

定义：

1. 基于任务局部设计单位坐标；
2. 已应用任务内部的旋转、平移、起始相位和运动方向；
3. 尚未应用 trial 的空间方向旋转和 fingertip anchor；
4. 保留点顺序；
5. 使用同样的 `<f8`、C contiguous 和 shape header。

该 hash 不要求以下实例相同：

- digit 2 与 digit 3 的 curve A，因为 digit 2 内部旋转了 55°；
- digit 3 与 digit 5 的 curve B，因为任务内位置不同。

digit 6 和 digit 9 的椭圆只分别记录各自的 `ordered_instance_sha256`，用于证明
各自实际执行序列的身份；两者之间不做共享性断言。

因此：

- `canonical_template_sha256` 证明模板复用；
- `ordered_instance_sha256` 证明实际执行序列的身份；
- 不得混用两者解释共享性。

---

## 2.2 配置字段统一为 `selected_reference_steps`

**状态：已确认。**

全部位置统一使用：

```text
selected_reference_steps
```

不得再出现：

```text
selected_reference
```

该字段必须同时出现在：

- geometry config；
- training config；
- resolved config；
- hp；
- checkpoint metadata；
- run summary；
-服务器脚本参数；
- 输出目录身份。

---

## 2.3 目录结构使用实际仓库的根目录结构

**状态：已确认。**

最小接管包中的 `formal_f760672/` 是打包时用于区分代码身份的目录，不应写入实际仓库实施路径。

实际仓库统一使用：

```text
configurations/
server/
runs/
```

建议 protocol3 唯一路径：

```text
configurations/
  digit_writing_original_protocol3_geometry_scale2p50.json
  digit_writing_original_protocol3_preflight_scale2p50_ref50.json
  digit_writing_original_protocol3_preflight_scale2p50_ref100.json
  digit_writing_original_protocol3_full10_scale2p50_ref50_dev42.json
  digit_writing_original_protocol3_full10_scale2p50_ref100_dev42.json

server/
  run_digit_writing_original_protocol3_preflight.sh
  run_digit_writing_original_protocol3_full10.sh

runs/digit_writing_original_protocol3/
  scale2p50/ref50/gate1/
  scale2p50/ref50/gate2/
  scale2p50/ref50/full10/dev42/
```

回退到 2.25× 时必须使用另一套目录：

```text
runs/digit_writing_original_protocol3/
  scale2p25/ref50/...
```

不得混用 `outputs/` 和 `runs/`。

---

## 2.4 validation 的噪声、RNG 和 best checkpoint 规则

**状态：可以冻结，建议保持 protocol2 主验证语义。**

建议正式 best-checkpoint validation 使用：

### 条件枚举

固定顺序：

```text
digit = 0..9
delay = 25,50,75
direction = 0..31
selected_reference_steps = 固定值
```

每个 digit×delay 使用一个 batch，包含 32 个不同方向。

全部 30 个 digit×delay 条件等权平均。

### 噪声

保持 protocol2 当前主验证语义：

- recurrent/input network noise：开启；
- environment observation noise：开启，即 `deterministic=False`；
- action noise：当前 `envs.py::DigitWritingEnv.step()` 中 `noisy_action=action`，不额外加入 action noise；
- 每次 validation 开始时使用同一个：
  ```text
  validation_seed = 1042
  ```
- `_fixed_rng(1042)` 在 validation 后恢复训练 RNG，不改变后续训练随机序列。

由于每次 validation 都恢复同一个固定初态，所以噪声实现虽开启，但跨 update 可复现。

### deterministic sentinel

另外执行无噪声评估：

- policy `network_noise=False`；
- env `deterministic=True`，并且该状态必须从 reset 持续到整个 trial 结束，所有
  后续 `step()` observation 都不得重新加入噪声；
- 相同 30 条件和 32 个方向；
- 只写行为指标；
- 不参与 best checkpoint 选择。

### best checkpoint 同分规则

当前 `train.py::train_subsets_base_model()` 使用：

```python
if test_loss <= best_test_loss:
```

因此精确同分时保留**较晚** checkpoint。

为减少非必要改动，建议 protocol3 明确保留这一规则；若要改为较早 checkpoint，必须作为单独科学/工程决策记录，不能静默改变。

---

## 2.5 update 5,000 必须采用明确的暂停—审计—恢复流程

**状态：流程可以冻结，但需要实现新的可靠 resume。**

推荐唯一流程：

1. full10 从零训练；
2. 完成恰好 5,000 次 optimizer update；
3. 保存：
   ```text
   early_continuation_checkpoint.pt
   ```
4. checkpoint 至少包含：
   - model state；
   - optimizer state；
   - Python/NumPy/Torch RNG state；
   - 当前 Git/config identity；
   - `completed_updates=5000`；
   - `next_update=5000`；
   - validation history；
   - best checkpoint 状态；
5. 训练 runner 正常退出，并写：
   ```text
   EARLY_AUDIT_PENDING=1
   ```
6. 独立只读审计脚本加载 checkpoint；
7. 审计脚本不得调用 `optimizer.step()`；
8. 审计前后模型 state SHA-256 相同；
9. 审计报告交给用户；
10. 用户明确同意后，resume runner 加载 model、optimizer 和全部 RNG；
11. 从 `next_update` 继续至 75,000；
12. 必须有自动测试证明：
    ```text
    中断后恢复的条件序列
    ==
    同 seed 不中断运行的条件序列
    ```

不建议在训练进程继续运行时异步做审计，因为这样会越过停止线。

不得直接使用旧的：

```text
train.py::load_prev_training()
```

作为 Protocol3 resume 入口。该函数是遗留路径，循环、loss 和环境调用语义不符合当前正式基础训练入口。Protocol3 需要针对 `_checkpoint_payload()` 和 `train_subsets_base_model()` 的正式 resume 实现。

---

## 2.6 旧 full10 与 protocol3 的比较口径

**状态：已确认。**

不能直接比较：

- raw validation loss；
- raw position error（米）；
- training loss；
- checkpoint update；
- best loss 数值。

原因是 protocol3 改变了：

- 全局尺度；
- movement intervals；
- reference condition；
- validation 条件；
- 训练条件分布。

允许比较的只是各自目标下的无量纲行为完成质量，例如：

```text
normalized movement error
normalized endpoint error
path-length ratio
path completion ratio
```

其中归一化尺度必须在两个协议中一致，例如：

```text
目标轨迹 bounding-box diagonal
```

这种比较只能支持：

> 新协议下的轨迹完成质量是否比旧失败结果更好。

不能解释为：

> 两个模型在同一实验协议下的严格性能高低。

---

## 2.7 尺度回退必须拥有独立身份

**状态：已确认。**

若 2.5× Gate 1 触发安全警戒线：

1. 保留全部 2.5× 失败结果；
2. 不覆盖任何 JSON、CSV、日志或 hash；
3. 2.25× 使用新配置；
4. 使用独立输出目录；
5. run label、服务器包名和 checkpoint metadata 写明：
   ```text
   scale_multiplier = 2.25
   global_scale_m_per_unit = 实际值
   ```
6. 2.5× 与 2.25× 不共享：
   - checkpoint；
   - optimizer state；
   - RNG state；
   - Gate 2 临时模型；
   - continuation state。

最终 protocol3 结果必须同时记录协议族名称和实际尺度身份，不能只写 `protocol3`。

---

## 2.8 正式代码必须先提交再生成服务器包

**状态：已确认。**

正式训练 checkpoint 中的 Git HEAD 必须对应实际 protocol3 代码。

服务器包生成前必须满足：

- protocol3 代码、配置、测试和最终实施文档已提交；
- Git HEAD 是实际 protocol3 commit；
- parent/base commit 记录为：
  ```text
  f76067232fac8add756b04ea75a8825cca156330
  ```
- mRNNTorch 记录为：
  ```text
  ac0c4f589eae37bbde63968912925de99232e306
  ```
- 正式工作树为 clean，或明确记录唯一允许的非科学未跟踪文件；
- 外层和包内 SHA-256 通过；
- server run label 包含：
  - protocol；
  - scale；
  - selected reference；
  - seed；
  - Git short SHA。

未提交代码不能用于正式训练，因为 checkpoint 中的旧 HEAD 无法证明实际执行内容。

---

# 3. 用户已经确认的实施决策

## 3.1 Git 分支或 worktree 操作

**状态：已确认并已执行。**

- 唯一工作目录继续为 `D:\digit_writing_original_protocol2`；
- 当前分支固定为 `codex/digit-writing-original-protocol3`；
- 科学基点固定为 `f76067232fac8add756b04ea75a8825cca156330`；
- 不创建第二 worktree 或新的本地项目文件夹；
- protocol3、protocol2、旧数字项目和汉字项目必须保持代码、配置、checkpoint、
  结果与服务器任务隔离；
- 保留当前未跟踪文档、压缩包和 artifacts，不清理或混入正式训练结果。

---

## 3.2 Protocol3 Gate 2 是否已经获得执行授权

**状态：Gate 1 和 Gate 2 均已授权。**

执行顺序固定为：

1. 先执行 Gate 1；
2. Gate 1 通过后可直接执行 Gate 2，无需再次请示；
3. Gate 2 先验证 fast；
4. fast 未通过，并且已经排除代码、数据流和安全问题时，只允许整体回退一次到
   medium；
5. medium 仍失败则停止并报告，不再扫描其他速度或修改协议。

服务器仍由用户执行逐条短命令，助手不得直接连接或接管服务器。

---

## 3.3 update 5,000 后由谁决定是否继续

**状态：已确认，必须由用户人工批准。**

固定流程：

```text
训练在 5,000 updates 明确暂停
→ Codex/脚本生成审计报告
→ 用户人工确认
→ 再 resume
```

到达 5,000 updates 时必须保存完整 model、optimizer 和全部 RNG state，并正常退出。
未经用户明确批准，不得从 continuation checkpoint 自动续训至 75,000 updates。
不得根据梯度 ratio、cosine 或单个数字的早期表现自动继续或自动判定失败。

---

## 3.4 full10 的最终 PASS/FAIL 阈值

**状态：已确认采用工程 PASS/FAIL，但具体数值尚未冻结。**

此前提出的候选数值缺少直接依据，当前不采用。固定决策为：

1. Gate 2 完成后，汇总 Gate 2 的可达到水平；
2. 使用同一定义重新整理旧 protocol2 的无量纲行为指标；
3. 在正式 full10 启动前一次性提出并由用户确认最终工程阈值；
4. 阈值必须覆盖总体、逐数字和路径完成质量，并完整报告全部指标；
5. 阈值冻结后才能启动 full10；
6. 正式训练启动后不得修改阈值。

这些阈值是预先冻结的工程验收标准，不得描述为论文原有标准。

---

## 3.5 5,000-update 联合训练门槛

**状态：已确认采用人工决策，不设置学习进展自动硬停止。**

安全与证据完整性硬条件仍包括：

- 无 NaN/Inf；
- 无工作空间、关节或 IK/FK 失败；
- checkpoint、optimizer 和 RNG 完整；
- Git、配置和输出身份一致。

学习进展统一使用无噪声、同网格比较：

```text
network noise = false
environment deterministic = true
delay = [25, 50, 75]
direction = 0..31
update 0 与 update 5000 使用完全相同的评估网格
```

报告条件固定为：

1. overall mean normalized movement error 低于 update 0；
2. 至少 8/10 个数字的该指标低于 update 0；
3. 任一数字相对 update 0 的恶化不超过 15%。

若未满足，只能暂停并报告具体结果，由用户人工决定是否继续；不得自动判定正式
训练失败，也不得自动 resume。

推荐由用户查看：

- update 0、500、1000、...、5000 的曲线；
- 十数字 movement-only overlay；
- 每个数字的 normalized error 和 path ratio；

然后决定是否续训。

---

# 4. 计算量的有依据估计

## 4.1 已知基准

旧 protocol2 full10：

```text
75,000 updates
CPU
约 24 小时 15 分
```

旧 protocol2 在三训练 reference 条件下，当前十数字的平均 movement intervals 约为：

```text
61.2
```

平均 trial 时间步数近似：

```text
stable 25
+ average delay 50
+ movement samples 62.2
+ hold 25
≈ 162.2 steps
```

## 4.2 Protocol3 fast

候选时间表下十数字平均 movement intervals：

```text
85.5
```

平均 trial 约：

```text
25 + 50 + 86.5 + 25
≈ 186.5 steps
```

单个训练 batch 的时间序列比旧 protocol2 平均约长：

```text
186.5 / 162.2 ≈ 1.15
```

但旧 validation 每次包含：

```text
10 digits × 10 reference conditions = 100 batches
```

新固定 reference validation 只需：

```text
10 digits × 3 delays = 30 batches
```

按每 500 updates 粗略计算，训练和验证合计的序列工作量：

```text
旧 protocol2 ≈ 97,000 相对 step-batch 单位
protocol3 fast ≈ 99,000
```

因此 protocol3 fast 的总耗时很可能与旧 full10接近，而不是显著增加。

合理估计：

```text
约 24–30 小时 CPU
```

这是估计，不是保证。

## 4.3 Protocol3 medium

medium 的平均 movement intervals 约为 fast 的两倍：

```text
171
```

平均 trial 约：

```text
272 steps
```

考虑 validation batch 数减少后，合理估计：

```text
约 34–45 小时 CPU
```

## 4.4 Gate 1、Gate 2 和 5,000-update 审计

粗略估计：

- Gate 1：分钟级到 1 小时以内，取决于 MotorNet 环境和审计图；
- Gate 2 fast 三个模型：约 1–2 小时 CPU；
- 若重新运行 medium：总诊断时间可能增加到约 2–4 小时；
- full10 前 5,000 updates：约为总训练的 1/15，通常约 1.5–2.5 小时；
- 只读审计时间取决于 30 条件、32 方向 rollout 和梯度计算，预计低于完整训练，但附件不足以给出精确值。

正式申请服务器前，Codex 应根据本机 100-update smoke 的实测速率重新给出：

```text
seconds_per_update
seconds_per_validation
estimated_full_runtime
estimated_archive_size
```

归档空间不能从当前轻量附件精确推断，应以实际预检产物测量。

---

# 5. 对审查问题的逐项结论表

| 审查问题 | 结论 |
|---|---|
| 2.1 固定速度含义冲突 | 同意；改为 `fixed_segment_timing`，speed scalar 仅为冻结 reference 标签 |
| 2.2 protocol2 只读与共享入口冲突 | 同意；使用配置显式分派，保护行为而非禁止修改共享文件 |
| 2.3 Gate 2 证据不足 | 同意；Gate 2 只选择时间，5,000-update full10 作为联合训练门 |
| 2.4 validation 缩减方向 | 同意；validation 继续使用 32 个不同方向 |
| 2.5 梯度 hard stop | 同意；梯度只做诊断，不单独停止 |
| 2.6 Gate 2 与 Stage E | 同意；明确是新的独立诊断并单独授权 |
| 3.1 canonical hash | 拆为 canonical template 和 ordered instance 两种 |
| 3.2 配置字段 | 统一为 `selected_reference_steps` |
| 3.3 目录 | 使用根目录 `configurations/`、`server/`、`runs/` |
| 3.4 Git 操作 | 未确定，等待用户授权 |
| 3.5 validation RNG | 可冻结：固定条件顺序、固定 seed、主验证有噪声、sentinel 无噪声 |
| 3.6 update 5000 流程 | 明确暂停、独立审计、用户确认、可靠 resume |
| 3.7 最终 PASS/FAIL | 无权威标准；候选工程阈值需用户批准 |
| 3.8 旧 full10 比较 | 只能比较无量纲行为质量，不能比较 raw loss |
| 3.9 尺度回退身份 | 分配置、分目录、分 hash、分 checkpoint |
| 4.1 计算量 | fast 预计接近旧 full10；medium 明显更慢；需实测校准 |
| 4.2 Git 身份 | 正式代码必须提交后再打包和训练 |

---

# 6. 当前剩余停止线

用户已经确认本文件此前列出的实施决策。当前剩余停止线只有：

1. Gate 2 完成后、正式 full10 启动前，必须先根据 Gate 2 和旧 protocol2 的
   无量纲行为指标冻结最终工程 PASS/FAIL 阈值；
2. full10 到达 5,000 updates 后必须强制暂停，未经用户人工批准不得续训；
3. heldout5、composition、transfer5、旧 Stage E 和 Stage F 仍未授权；
4. 不得创建第二 worktree、本地项目文件夹或混用其他项目资源。

---

# 7. 2026-07-29 本地实施状态

以下三项审查修正已经落实到本地 Protocol3 分支：

1. deterministic evaluation 的状态从 reset 持续到整个 trial；最终 observation、proprioception 和 vision 均不重新注入环境噪声；
2. 协议和代码统一使用 `fixed_segment_timing` / “固定片段时间条件”表述，fast/medium 仅表示 reference 时间倍率，不表示不同片段具有严格相同物理速度；
3. digit 6/9 椭圆按各自设计轨迹独立生成，只记录各自 ordered instance 身份，不做共享 hash 或计算复用断言。

同时已经实现但尚未在服务器执行：

- Gate 1 静态、安全和 0-update closed-loop 预检；
- Gate 2 三个独立模型、工程/安全/行为失败分类、fast→medium 唯一回退约束；
- update 0、500、…、5000 的验证时间点；
- 完整 model、optimizer、Python/NumPy/Torch RNG、配置/Git 身份和采样计数 continuation checkpoint；
- 每个 update 从已保存训练 RNG 派生 MotorNet 环境 seed；
- update 5,000 强制暂停、只读同网格审计和显式人工授权 resume。

当前没有创建正式 full10 配置，也没有冻结最终工程阈值。Gate 1、Gate 2 和任何正式训练均尚未在服务器启动。

# Protocol3 v4 改造方案审查问题

审查对象：`CODEX_PROTOCOL3_FIXED_SPEED_MINIMAL_VALIDATION_AND_RETRAIN_v4.md`

审查日期：2026-07-29（Asia/Shanghai）

## 1. 总体结论

当前 v4 文档不适合直接作为实施规范。

方案的主要技术路径可以实现，但仍存在科学定义冲突、验证证据不足、停止条件不稳健以及实施身份不唯一等问题。如果直接编码，即使新增测试全部通过，也可能出现以下结果：

- 协议名称与真实物理含义不一致；
- protocol3 的改动破坏 protocol2；
- 最小 Gate 通过，但 full10 联合训练仍系统性失败；
- 良好模型因梯度比例规则被错误停止；
- 不同实现者依据同一文档得到不同配置、目录或验证结果；
- 正式训练结束后仍无法客观判定是否达到“高质量拟合”。

实施前至少应先解决第 2 节列出的阻断问题。

## 2. 必须先解决的阻断问题

### 2.1 “固定速度”与实际时间规则不一致

文档规定所有直线统一使用 30 个 fast movement intervals，不按直线长度分配时间，同时明确承认不同片段不具有严格相同的物理速度。

按文档给出的缩放后长度计算，fast 条件下：

```text
最短直线：0.067308 m / 0.30 s ≈ 0.224 m/s
最长直线：0.172005 m / 0.30 s ≈ 0.573 m/s
```

两者实际均速相差约 2.56 倍。因此当前方案实际定义的是“固定片段类型时间表”，不是“固定物理速度”。所有片段仍使用同一个 fast 或 medium speed scalar，也会造成输入标签与真实片段速度不一致。

实施前必须二选一：

1. 如果目标是固定物理速度，intervals 必须随片段弧长变化；
2. 如果目标是固定片段类型时长，应将协议表述改为 `fixed_segment_timing` 或等价名称，并明确 speed scalar 只是冻结的 reference-condition 标签，不代表真实片段速度。

在该决策完成前，不能唯一确定协议的科学含义。

### 2.2 protocol2 只读要求与共享几何入口改造相冲突

文档要求修改 `digit_geometry_final.py` 的全局尺度、`sample_digit()` 的时间参数化以及共享的 `geometry.py`、`envs.py` 和 `train.py`。

但当前实现中：

- `Segment.points_m()` 直接乘唯一的 `GLOBAL_SCALE_M_PER_UNIT`；
- `sample_digit()` 没有 protocol 或 scale 参数；
- protocol2 几何配置强制检查配置尺度等于该全局常量；
- protocol2 和 protocol3 将共同经过上述共享入口。

直接把全局常量改成 `2.5×` 会破坏 protocol2；保留原常量又无法按当前文档直接实现 protocol3。

文档必须明确采用协议分派：

- 设计单位几何保持唯一且不变；
- 物理尺度和时间模式由协议配置显式传入；
- protocol2 继续走原尺度和按物理速度采样路径；
- protocol3 走候选统一尺度和固定片段时间表；
- 两种协议有独立配置校验和回归测试。

保护目标应改为“protocol2 的配置、行为、输出、checkpoint 和测试保持不变”，不能要求所有共享代码文件在字面上完全不修改。

### 2.3 Gate 2 的证据不足以授权 full10

Gate 2 只训练三个互相独立的单任务模型：

```text
digit 1，direction 0，delay 50
digit 5，direction 0，delay 50
digit 8，direction 0，delay 50
```

这最多能证明三个单条件可以短时过拟合，不能证明：

- 十数字能在同一网络中联合学习；
- 8 个空间方向均可学习；
- 3 个 delay 下表现稳定；
- digit 4/7 的尖角条件可学习；
- digit 6/9 的闭环加尾段可学习；
- 已知的局部任务干扰不会导致 full10 失败。

Gate 1 smoke 只检查有限性、安全性和运行完整性，不能替代轨迹可学习性验证。

因此必须明确：Gate 2 是只用于选择 fast/medium，还是被视为授权 full10 的充分条件。如果坚持当前最小 Gate，正式启动结论必须明确保留尚未排除的联合训练和方向泛化风险。

### 2.4 正式验证从 32 个方向缩减为 8 个方向

protocol2 正式验证覆盖 32 个不同方向。v4 改为：

```text
8 个方向 × 每方向 4 个样本
```

该变更会失去训练方向之间的角度泛化评估。确定性验证中，把同一个方向重复 4 次也不会增加方向覆盖。

当前环境中 `testing=True` 使用 32-direction 网格。如果直接传入方向索引 `0..7`，得到的是 32-direction 网格中的前 8 个相邻角度，而不是原来的 8 个训练方向。

如果验证确实只保留 8 个训练方向，必须明确使用：

```text
[0, 4, 8, 12, 16, 20, 24, 28]
```

更稳妥的方案是继续使用一个 batch 内的 32 个不同验证方向。batch size 原本就是 32，不需要增加 batch 数。

无论采用哪种方案，都必须把验证角度覆盖范围的变化列入 protocol3 的科学边界，不能把它作为未说明的实现细节。

### 2.5 梯度比例和 cosine 不适合作为当前形式的硬停止条件

文档要求：

```text
regularization_to_position_gradient_ratio < 1.0
position_total_gradient_cosine > 0
```

存在以下问题：

1. 模型拟合变好后，位置梯度可能自然接近 0，正则/位置梯度比例会被动升高；
2. 当任一梯度范数接近 0 时，cosine 没有稳定意义；
3. 极小分母会造成比例数值爆炸；
4. 一个行为表现良好的模型可能因此被错误判定为失败；
5. update 5000 只进行一次梯度审计，无法仅凭一个快照证明正则项“持续”高于位置项。

必须补充梯度范数有效下限、除零规则、cosine undefined 处理、趋势窗口，以及行为指标与梯度指标的联合判定规则。

NaN/Inf、checkpoint 不完整和工作空间失败可以继续作为独立硬停止条件；梯度比例和 cosine 不宜在没有上述定义时单独触发硬停止。

### 2.6 “禁止 Stage E”与 Gate 2 的实质相矛盾

文档一方面禁止 Stage E，另一方面要求 O1/O2/O3 三个短过拟合模型；如果 fast 失败，还要重新训练三个 medium 模型，最多可能执行：

```text
6 models × 3000 updates = 18000 diagnostic updates
```

这在实验性质上仍属于新的短诊断训练。

文档应明确：

> 不执行 protocol2 的 Stage E E1–E5；Protocol3 Gate 2 是新定义的独立诊断训练，必须按 exact run label 单独获得授权。

按照当前接管状态，Protocol3 Gate 2、Stage E、Stage F 和任何新的正式训练都尚未获得执行授权。

## 3. 必须进一步明确的问题

### 3.1 canonical sample SHA-256 的定义不完整

curve A/B 在不同数字中可能发生平移和旋转，直接对实际米制坐标逐字节求 hash 不会相等。

文档必须定义：

- 哪些 `shared_id` 必须具有相同 hash；
- hash 基于设计单位坐标还是米制坐标；
- 是否在 hash 前消除平移和旋转；
- 是否保留采样方向和顺序；
- 浮点 dtype、endianness 和序列化方式；
- 尺度 2.5× 与 2.25× 是否产生不同 hash；
- digit 6/9 的椭圆比较模板 hash、实际有序样本 hash，还是只比较 intervals。

建议分开记录：

```text
canonical_template_sha256
ordered_sample_sha256
```

前者证明模板共享，后者证明实际采样顺序完全相同。digit 6/9 因起始相位和方向不同，不应要求 ordered sample hash 相等。

### 3.2 配置字段名不统一

文档多数位置使用 `selected_reference_steps`，正式配置示例却使用 `selected_reference`。必须统一为一个字段名，否则配置验证、checkpoint 元数据和运行脚本可能记录不同字段。

### 3.3 文件和输出目录不统一

文档同时出现：

```text
formal_f760672/configurations/
formal_f760672/server/
outputs/digit_writing_original_protocol3/
```

当前仓库实际使用根目录下的：

```text
configurations/
server/
runs/digit_writing_original_protocol2/
```

必须明确 protocol3 的唯一目录结构，并说明是继续使用 `runs/`，还是有意迁移到 `outputs/`。不得在代码、配置、服务器脚本和交付文档中混用两套输出根。

### 3.4 Git 分支/worktree 要求与当前接管边界冲突

当前接管指令固定工作目录为：

```text
D:\digit_writing_original_protocol2
```

并明确要求不创建或切换 worktree/分支。v4 则要求从 `f760672` 新建分支或 worktree。

实施前必须由用户明确覆盖旧边界，并确定：

- 是否允许在当前目录切换到新分支；
- 是否允许创建新 worktree；
- 新分支的准确名称；
- 当前未跟踪文档和 artifacts 如何只读保留；
- 是否以 `f760672` 为新分支基点。

在该授权明确前，不应执行任何分支或 worktree 操作。

### 3.5 validation 噪声和 RNG 规则不够精确

“保持原正式验证噪声规则和固定 RNG”仍存在多种解释。必须明确：

- validation 的 recurrent/input noise 是否开启；
- environment noise 是否开启；
- 每次 validation 是否恢复同一个固定 RNG 初态；
- 30 个 digit×delay 条件是否显式枚举；
- 每个条件的方向顺序；
- 所有条件是否等权平均；
- deterministic sentinel 是否完全不参与 best checkpoint 选择；
- best checkpoint 同分时保留较早还是较晚 checkpoint。

否则 validation loss 不能保证跨 update 可比。

### 3.6 update 5000 审计的“原位继续”流程不完整

文档要求在 update 5000 保存 continuation checkpoint、执行只读审计，然后原位继续至 75000 updates，但没有定义：

- 训练进程是否在 5000 明确暂停；
- 审计由同一个 runner 内部执行，还是由用户重新启动独立脚本；
- 审计失败时如何阻止后续 update；
- 审计通过后如何恢复 optimizer 和全部 RNG state；
- 如何证明恢复后的条件序列与不中断训练一致；
- 审计结果是否需要用户人工验收后再继续。

必须选定唯一流程，避免训练在审计尚未完成时已经越过停止线。

### 3.7 正式训练结束没有预先冻结的 PASS/FAIL 标准

Gate 2 有数值阈值，但 75000-update full10 结束后的最低验收主要是定性问题，没有给出正式模型的最低通过标准。

需要预先定义：

- 30 个 digit×delay 条件的聚合阈值；
- 是否检查每个条件、每个数字的最差值；
- normalized mean/endpoint error 的正式阈值；
- path-length ratio 的正式阈值；
- 定性判断的固定人工审查规则或客观替代指标；
- 正式结果是 PASS、FAIL，还是仅报告不做二元结论。

没有预先冻结的标准，就无法客观判断核心目标“高质量拟合”是否实现。

### 3.8 与旧 full10 的比较口径不成立

protocol3 改变了全局尺度、movement intervals、速度条件、validation 条件、训练条件分布，并可能改变方向覆盖范围。因此不能直接比较 raw validation loss，也不能让旧 checkpoint 执行新目标后据此判断训练优劣。

如果需要回答“是否明显优于旧失败结果”，必须预先定义基于各自目标的无量纲行为指标，例如 normalized movement error、normalized endpoint error 和 path-length ratio，并明确这种比较只能说明行为完成质量，不能视为同协议下的严格模型优劣比较。

### 3.9 尺度回退的身份和证据保留规则不完整

文档允许 `2.5×` 在触发安全警戒线时统一回退到 `2.25×`，但还应规定：

- 2.5× 失败的 Gate 1 结果不得覆盖；
- 2.25× 使用新的配置 SHA 和结果目录；
- 最终协议身份记录实际 multiplier；
- 2.5× 与 2.25× 不共用 checkpoint 或 continuation state；
- 服务器包名和 exact run label 包含最终尺度身份。

否则 protocol3 名称相同但实际几何不同，结果谱系会产生歧义。

## 4. 其他实施风险

### 4.1 训练和验证计算量尚未预算

统一放大后，movement intervals 整体增加。medium 回退的最长 movement 达到 210 intervals，正式 validation 还会显式覆盖 30 个条件，并增加运行时行为与安全指标。

在申请正式服务器执行授权前，应说明：

- Gate 1、Gate 2 和 update 5000 审计的预计 CPU 时间；
- full10 fast/medium 相对 protocol2 的预计耗时；
- 中间归档空间；
- 服务器运行根和日志的唯一命名。

### 4.2 正式代码身份需要先提交再打包

checkpoint 要记录 Git HEAD、parent commit 和 mRNNTorch commit。若 protocol3 代码处于未提交状态，Git HEAD 只能指向旧代码，无法唯一证明实际训练内容。

正式服务器包生成前应满足：

- protocol3 文档和代码已经提交；
- 工作树状态已记录；
- parent/base commit 明确为 `f76067232fac8add756b04ea75a8825cca156330`；
- 服务器训练 HEAD 指向实际 protocol3 提交；
- 外层和包内 SHA-256 均通过。

当前被审查的 v4 文档仍是未跟踪文件，尚未形成正式 Git 身份。

## 5. 建议的实施前决策顺序

在修改代码前，应按以下顺序冻结决策：

1. 决定协议固定的是物理速度，还是片段类型时长；
2. 明确 protocol2/protocol3 在共享几何和训练入口中的分派方案；
3. 明确训练方向与验证方向的准确集合；
4. 明确 Gate 2 的证明边界，以及它是否足以授权 full10；
5. 修订梯度 hard-stop 数学规则；
6. 将 Protocol3 Gate 2 与旧 Stage E 明确区分，并单独授权；
7. 统一配置字段、目录、分支和输出身份；
8. 冻结 validation、best checkpoint 和 update 5000 审计流程；
9. 冻结正式 full10 的最终 PASS/FAIL 标准；
10. 修订 v4 文档后，再输出精准修改文件清单并开始编码。

## 6. 当前停止线

本文件仅整理审查问题，不构成以下操作的授权：

- 修改 protocol2 或 protocol3 代码；
- 创建或切换分支/worktree；
- 运行本地测试、Gate 1 或 Gate 2；
- 生成服务器包；
- 连接或接管服务器；
- 启动 Stage E、Stage F 或任何正式训练；
- 启动 heldout5、composition 或 transfer5。

在用户审查并明确上述关键决策前，应保持当前代码、旧 full10 结果和所有现有 artifacts 不变。

# digit_writing_original_protocol2 接续说明

## 1. 接手顺序与边界

工作目录：`D:\digit_writing_original_protocol`

接续时依次完整阅读：

1. `AGENTS.md`；
2. 本文件；
3. `PROJECT_PROTOCOL.md`；
4. `CODEX_MINIMAL_FINAL_DIGIT_MODIFICATION.md`；
5. 只有涉及 AutoDL 操作时再完整阅读 `AUTODL_SERVER_USAGE_GUIDE.md`。

当前项目固定为 `digit_writing_original_protocol2`，与上一项目完全分离。服务器仍
复用 `/root/autodl-tmp/digit_writing_original_protocol_4b8b1db` 这个 Git 工作目录，
但不得读取、覆盖或续训上一项目的 checkpoint、optimizer state、日志和结果。

## 2. Git 与环境身份

- 本地分支：`codex/digit-writing-original-protocol2`
- 项目 2 实现提交：`a18b172`
- 原仓库基线：`105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33`
- 固定 `mRNNTorch`：`ac0c4f589eae37bbde63968912925de99232e306`
- 服务器仓库：`/root/autodl-tmp/digit_writing_original_protocol_4b8b1db`
- CPU Python：`/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python`
- 固定环境：Python 3.10.20、PyTorch 2.6.0+cpu、MotorNet 0.2.0、
  NumPy 2.2.6、`torch.cuda.is_available() == False`

本地仅有一个与本次实现无关的既有未跟踪文件 `digit_writing.zip`；不要修改、删除或
提交它。不要修改 `artifacts/` 下上一项目留下的本地归档。

## 3. 已完成的项目 2 实现

- `digit_writing/digit_geometry_final.py` 是唯一数字几何、尺度、速度和分段采样权威；
- `digit_writing/geometry.py` 只保留环境所需的薄适配与审计接口；
- movement 监督 `movement_intervals + 1` 个样本，首末目标都属于 movement，hold
  从下一步开始；
- 新增固定权重 `0.1/0.1/0.6/0.2` 的阶段归一化二维位置 L1；
- full10、heldout5、组合、迁移统一使用新位置目标；基础训练原有四个正则项及
  系数保持不变，组合与迁移仍只使用位置项；
- 组合固定为 `[0, 4, 6, 9, 8]` 五任务 leave-one-out，每次只优化其余四个 rule
  系数；
- full10 与 heldout5 的模型、optimizer、checkpoint 和输出目录保持独立；
- 配置、运行目录、归档、授权变量和 runner 均已改为项目 2 命名空间；
- 新增正式训练前的几何、阶段损失、10×32 工作空间及高风险闭环审计入口。

活动配置：

```text
configurations/digit_writing_original_protocol2_geometry.json
configurations/digit_writing_original_protocol2_full10_dev42.json
configurations/digit_writing_original_protocol2_heldout5_dev42.json
configurations/digit_writing_original_protocol2_composition.json
configurations/digit_writing_original_protocol2_transfer5.json
```

## 4. 已完成的本地验证

本地只执行了不依赖 MotorNet 的针对性验证，没有执行训练：

```text
tests.test_digit_geometry                           8 passed
tests.test_phase_normalized_loss                    4 passed
tests.test_phase_d_protocol.PhaseDConfigurationTests 4 passed
tests.test_phase_d_protocol.PhaseDModelFreezingTests 4 passed
合计                                               20 passed
```

另已通过：

- 16 个当前 Python 文件的只读语法编译检查；
- 5 个项目 2 JSON 配置解析；
- `git diff --check`；
- 最终几何 `self_test()`；
- 本地生成并目视检查 `artifacts/final_digit_geometry/` 下的 0–9 与 A/B 图。

Windows 本地没有 MotorNet，因此以下证据仍未取得，不能写成已通过：

- MotorNet 环境测试；
- 10 digits × 32 directions 工作空间审计；
- 高风险闭环 smoke；
- 服务器上的组合梯度与迁移冻结整合测试。

## 5. 唯一下一步：AutoDL CPU 必要审计

先把当前分支上传到服务器仓库并确认父仓库、submodule 都干净，然后只运行：

```bash
cd /root/autodl-tmp/digit_writing_original_protocol_4b8b1db
bash server/run_digit_writing_original_protocol2_audit.sh \
  "$PWD" \
  /root/autodl-tmp/digit-writing-original-protocol2-audit-run1
```

审计 runner 会记录 Git/submodule、CPU 环境、配置和 tracked-file SHA-256，并运行
本轮要求的定向测试、最终几何输出、10×32 工作空间审计和高风险无训练闭环；最后
生成 `.tar.gz` 与 `.sha256`。若任何工作空间或闭环检查失败，只报告证据，不得修改
几何尺度、速度、损失、网络或审计阈值来通过。

审计通过后仍应停止，等待用户审查结果并明确授权 exact run label。

## 6. 正式运行状态与停止线

项目 2 尚未启动任何 75,000-update 正式训练。上一项目的任何授权都不延伸到项目 2。

| Exact run label | 当前状态 | 依赖 |
|---|---|---|
| `full10-dev42` | 未授权 | 项目 2 服务器审计通过并经用户审查 |
| `heldout5-dev42` | 未授权 | 项目 2 服务器审计通过并经用户审查 |
| `composition-dev42` | 未授权 | 项目 2 full10 best checkpoint 已验收 |
| `transfer5-dev42` | 未授权 | 项目 2 heldout5 best checkpoint 已验收 |

硬停止线：

- 不启动未被用户逐字授权的 exact run label；
- 不导入或覆盖上一项目结果；
- 不混用 full10 与 heldout5 checkpoint；
- 不修改最终几何、全局尺度、速度、阶段损失权重、网络规模或 75,000-update 协议；
- 不根据 smoke、组合或单次 seed 结果反向美化数字或更换组合任务组。

明确状态：`尚未启动 75,000 updates 正式训练。`

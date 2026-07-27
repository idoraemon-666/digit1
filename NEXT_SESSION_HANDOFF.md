# Next-session handoff

## 1. 接手顺序与权威边界

工作目录：`D:\digit_writing_original_protocol`

新任务开始后依次完整阅读：

1. `AGENTS.md`；
2. 本文件；
3. `PROJECT_PROTOCOL.md`；
4. 只有涉及 AutoDL 操作时再读 `AUTODL_SERVER_USAGE_GUIDE.md`。

`PROJECT_PROTOCOL.md` 是唯一科学与实施规范。不要从
`D:\digit_writing_project` 导入代码、配置、checkpoint、测试或实验结果；该目录
只是历史归档。项目基准设备固定为 AutoDL CPU，CUDA 不属于当前结果谱系。

## 2. 固定身份与当前实现

- 本地分支：`codex/original-protocol-rebuild`
- 原仓库基线：`105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33`
- 固定 `mRNNTorch`：`ac0c4f589eae37bbde63968912925de99232e306`
- 当前接受的服务器 HEAD：
  `b7b9b28189e33010a1ee5da28a0a27b0bd101a59`
- 服务器仓库：
  `/root/autodl-tmp/digit_writing_original_protocol_4b8b1db`
- CPU 环境：`/root/autodl-tmp/conda/envs/motor-compat-cpu`
- 固定环境：Python 3.10.20、PyTorch 2.6.0+cpu、MotorNet 0.2.0、
  NumPy 2.2.6、`torch.cuda.is_available() == False`

本地工作树是由各次已接受服务器结果组合形成的审查工作树，当前非干净状态是
已知事实。不要执行 `git reset --hard`、`git checkout --`，也不要删除未跟踪的
项目文件。服务器上的正式运行仓库在运行期间必须保持干净，不要修改任何文件。

## 3. 已完成工作

- Phase A：原仓库身份、Git 祖先、训练核心和 submodule 边界已确认。
- Phase B：数字几何、时间规则、18 张正式轨迹图及工作空间审计已通过。
- Phase C：28 维输入、trial 时序、采样和 MotorNet CPU 闭环已通过。
- 无训练协议审计：几何、时间、工作空间和 rollout 全集无阻断项。
- Phase D v2/v2.1：full10、heldout5、composition、transfer5 四条独立路径已
  实施；51 项 AutoDL CPU 测试全部通过，0 skipped。
- Phase E：交付审查已完成，详见 `PHASE_E_DELIVERY_REVIEW.md`；未发现实现
  阻断项。

关键实现边界：

- full10 和 heldout5 复用原仓库基础训练循环，但模型、optimizer、配置、目录和
  checkpoint 完全独立；
- composition 只允许读取 full10 best checkpoint，并冻结网络、只优化外部系数；
- transfer5 只允许读取 heldout5 best checkpoint，并只训练重新初始化的数字 5
  rule 输入列；
- 固定 submodule 的 RNN 调用兼容补丁只修正参数映射，不改变网络、loss 或协议；
- 每个正式 runner 都要求用户明确授权与其完全一致的 exact run label。

已知但非阻断的科学风险：数字最长/最短 movement steps 比为 `6.318`，大于原任务
的 `2.0`。当前证据不支持修改几何、速度、loss、时间尺度或网络规模。

## 4. 当前正在运行的正式实验

用户只授权了：`full10-dev42`。

当前状态是该实验正在 AutoDL CPU 上运行，尚未完成归档。最后一次用户报告为：

```text
验证记录数：128
最近完成的验证 batch：63500/75000
当前验证损失：0.01426648113
历史最佳损失：0.01421837435（batch 62000）
当前比最佳高：0.338%
最近 10000 updates 首尾改善：2.558%
最近 10 次验证均值相对前 10 次：-2.921%
相对斜率：-0.628% / 1000 updates
```

判断：训练健康、整体收敛、仍在缓慢改善，尚未完全平台化；62500–63000 的反弹
已在 63500 基本恢复，没有发散或持续退化证据。必须继续跑满协议规定的 75000
updates，不提前停止、不调整学习率、不修改代码。

运行位置：

```text
启动日志：/root/autodl-tmp/full10-dev42-launch.log
训练输出：/root/autodl-tmp/digit_writing_original_protocol_4b8b1db/runs/digit_original_protocol/full10/dev42
验证损失：上述目录/test_losses.txt
最佳模型：上述目录/best_checkpoint.pt
```

`nohup` 日志只出现 preflight 标记而暂时没有 batch 行，是 Python 通过管道输出时的
缓冲现象，不能据此认定卡住。进度以 Python 进程、`test_losses.txt`、checkpoint
及其更新时间共同判断。

## 5. 下一步工作

### 5.1 先完成并验收 `full10-dev42`

1. 等待进程自然完成，不重复启动同一 label。
2. 收集 runner 最终标记：`PARENT_HEAD`、`RUN_ROOT`、`ARCHIVE`、
   `RUN_EXIT=0`、`TEE_EXIT=0`、`FORMAL_RUN_COMPLETE=1`。
3. 核对预期归档及校验文件：

```text
/root/autodl-tmp/digit-writing-full10-dev42-b7b9b28-seed42.tar.gz
/root/autodl-tmp/digit-writing-full10-dev42-b7b9b28-seed42.tar.gz.sha256
```

4. 用户下载归档后，由助手在本地独立核对外层 SHA-256、包内
   `SHA256SUMS`、Git/submodule HEAD、CPU 环境、退出码、配置、日志、best/final
   checkpoint 和实际执行范围。
5. 只有上述验收全部通过，才把 full10 标记为正式完成并更新项目文档。

如果新任务开始时训练仍在运行，先让用户返回以下只读信息：Python 进程、
`test_losses.txt` 行数与末尾、`best_checkpoint.pt` 时间。不要修改服务器仓库。

### 5.2 后续实验仍需逐项明确授权

当前授权表：

| Exact run label | 状态 | 依赖 |
|---|---|---|
| `full10-dev42` | 已授权，正在运行 | 无 |
| `heldout5-dev42` | 未授权 | 无；必须新建独立九数字模型 |
| `composition-dev42` | 未授权 | full10 best checkpoint 已验收 |
| `transfer5-dev42` | 未授权 | heldout5 best checkpoint 已验收 |

`full10-dev42` 的授权不自动延伸到其他三个 label。完成 full10 验收后，下一项应由
用户再次明确指定 exact run label；在两套基础模型都完成前，不进入最终组合与迁移
结果阶段。

## 6. 硬停止线

- 不修改冻结的数字几何、全局缩放、速度、loss、网络规模或 75000-update 协议。
- 不导入历史数字项目的代码、配置、checkpoint 或结果。
- 不混用 full10 与 heldout5 checkpoint。
- 不预设数字 rule 的代数组合关系。
- 不根据单次 seed 或短期 loss 波动反向修改协议。
- 不运行未被用户逐字授权的 exact run label。

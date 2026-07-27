# AutoDL 服务器使用说明

## 1. 范围

本文件只说明服务器、终端、环境、上传、后台运行和结果归档方法，不定义科学
协议。数字几何、训练、组合和迁移的唯一依据是 `PROJECT_PROTOCOL.md`。

当前项目的基准设备是 CPU。服务器存在 RTX 4090 不代表可以把 CUDA 结果并入
当前基准谱系。

当前实例已经验证的路径：

```text
工作盘：/root/autodl-tmp
仓库：/root/autodl-tmp/digit_writing_original_protocol_4b8b1db
CPU 环境：/root/autodl-tmp/conda/envs/motor-compat-cpu
Python：/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python
```

当前固定软件：

```text
Python 3.10.20
PyTorch 2.6.0+cpu
MotorNet 0.2.0
NumPy 2.2.6
torch.cuda.is_available() == False
```

这些路径和环境只对当前实例成立。新开一张相同型号的卡，仍然必须按全新服务器
处理。

## 2. 实例与本地终端

关闭本地电脑、浏览器、网页终端或 SSH，不会停止已经正确放入后台的任务。
关闭、释放、重建实例或手动结束目标进程会停止任务，并可能使未持久化的文件和
环境消失。

显卡型号相同不意味着仓库、Conda 环境、日志或 checkpoint 已经复制。进入新
实例后必须重新上传、校验、恢复仓库、建立环境和执行 preflight。

大型环境、仓库、日志和结果都放在 `/root/autodl-tmp`，不要放在空间较小的
`/root` 系统盘。

## 3. 网页终端安全输入规则

### 3.1 一次执行一条短命令

粘贴时不要带 `root@autodl-container-...#` 终端提示符。本项目默认由助手把
严格多步逻辑写进上传包，用户只执行短的单行命令：

```bash
cd /root/autodl-tmp
sha256sum -c package.tar.gz.sha256
tar -xzf package.tar.gz
nohup bash /root/autodl-tmp/stage/RUN.sh >/root/autodl-tmp/stage.log 2>&1 </dev/null &
```

每条执行完先看输出，再执行下一条。不要把多段命令、终端输出和下一条命令一起
粘贴。

### 3.2 不在交互终端启用严格模式

不要直接执行：

```bash
set -euo pipefail
```

一旦后续路径不存在、变量未定义或管道返回非零，当前交互 Shell 可能直接退出，
看起来像网页终端“闪退”。严格模式应只存在于已经上传并审计过的脚本内部。

如果当前终端尚未退出，可恢复普通模式：

```bash
set +e
set +u
set +o pipefail
```

如果终端已经退出，重新打开后不要立即重跑；先检查文件、环境、Git HEAD、进程
和日志，判断前一任务究竟是否执行过。

### 3.3 避免粘贴复杂多行脚本

实际使用中出现过多行内容被终端交错、截断或粘贴为错误文本的情况。不要直接
粘贴长 heredoc、内嵌 Python、复杂反斜杠续行或含大量变量的命令块。

正确做法是把逻辑放入上传包的 `.sh` 文件并执行该文件。如果终端显示 `>`，
说明引号或多行结构没有闭合，应按 `Ctrl+C` 取消，不要继续盲目输入。

`Ctrl+C` 只中断当前前台命令；`Ctrl+D` 或 `exit` 会关闭当前 Shell。

## 4. 每次开始前的只读检查

```bash
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
test -d /root/autodl-tmp/digit_writing_original_protocol_4b8b1db/.git && echo REPO_PRESENT=1
test -x /root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python && echo CPU_ENV_PRESENT=1
```

只有看到预期路径存在，才继续运行项目脚本。环境目录曾经存在并不代表现在仍然
存在。

进一步检查：

```bash
git -C /root/autodl-tmp/digit_writing_original_protocol_4b8b1db rev-parse HEAD
git -C /root/autodl-tmp/digit_writing_original_protocol_4b8b1db/mRNNTorch rev-parse HEAD
git -C /root/autodl-tmp/digit_writing_original_protocol_4b8b1db status --short
/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python -m pip check
/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python -c "import motornet,numpy,torch; print(torch.__version__,motornet.__version__,numpy.__version__,torch.cuda.is_available())"
```

`git status --short` 没有输出表示工作区干净，不是卡住。

## 5. CPU 环境

环境固定放在 `/root/autodl-tmp/conda/envs/motor-compat-cpu`。安装普通依赖时
使用临时国内镜像，不永久修改 pip 全局配置：

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple PIP_DEFAULT_TIMEOUT=120 /root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python -m pip install 包名
```

PyTorch CPU wheel 必须使用项目固定来源和版本，不能因为服务器有 GPU 就安装
CUDA 构建。

后台脚本应直接调用绝对 Python 路径，不应依赖 `source .../bin/activate`。
本轮曾出现 `bin/activate: No such file or directory`，说明目标环境在当时实例中
不存在，不能通过重跑任务解决。正确顺序是：

1. 停止重跑；
2. 检查绝对 Python 是否存在；
3. 如不存在，重新创建并验证 CPU 环境；
4. 再使用新的运行目录启动任务。

## 6. 上传包与 SHA-256

文件统一上传到 `/root/autodl-tmp`。先校验外层压缩包：

```bash
cd /root/autodl-tmp
sha256sum -c package.tar.gz.sha256
```

只有看到 `package.tar.gz: OK` 才能解压。上传完成不等于内容完整。

### 6.1 Windows CRLF 问题

如果内部校验出现：

```text
sha256sum: 'MANIFEST.txt'$'\r': No such file or directory
```

通常不是文件真的缺失，而是校验清单使用了 Windows CRLF 换行。不要据此删除或
重传 payload。应停止应用，使用 LF 规范化后的新包，或者仅在明确指令下规范化
校验清单后重新验证。

新的上传包交付前必须确认：外层 SHA-256 正确、内部 hash 全部通过、脚本和
checksum 清单为 LF、CRLF 行数为 0，并在独立解压目录完成二次验证。

## 7. Git bundle 与仓库恢复

离线 bundle 可以避免 GitHub 或 submodule 下载卡住。父仓库和 submodule 应
各自使用 bundle，并显式核对目标 commit。

`git bundle verify` 需要 Git 仓库上下文。直接在普通目录运行会得到：

```text
error: need a repository to verify a bundle
```

正确形式是在已有仓库中验证：

```bash
git -C /root/autodl-tmp/digit_writing_original_protocol_4b8b1db bundle verify /root/autodl-tmp/repository.bundle
git -C /root/autodl-tmp/digit_writing_original_protocol_4b8b1db/mRNNTorch bundle verify /root/autodl-tmp/mRNNTorch.bundle
```

恢复脚本应先克隆，再在对应仓库上下文验证；不要把验证放在仓库创建之前。

## 8. 后台运行

短期审计和安装任务可使用：

```bash
nohup bash /root/autodl-tmp/stage/RUN.sh >/root/autodl-tmp/stage.log 2>&1 </dev/null &
```

终端返回 `[1] 5114` 只表示后台进程被 Shell 接收，不表示脚本已经通过。任务
可能在一秒内失败。

检查指定 PID：

```bash
ps -p 5114 -o pid=,etime=,stat=,cmd= || echo STATUS=FINISHED
```

如果已经结束，先检查日志，不要直接重跑。后台任务返回 `Exit 1` 时同样要保留
现场并读取第一个错误。启动前用 `pgrep -af 'RUN.sh' || true` 确认没有重复任务。

长期训练可使用 `screen`，但仍必须同时检查 Python 进程、日志和完成标记；
`screen` 存在本身不代表训练正常。

## 9. 正式训练的进度与收敛监控

### 9.1 启动日志没有 batch 行不等于训练卡住

正式 runner 通过 `tee` 和 `nohup` 保存日志。Python 标准输出进入管道后可能按块
缓冲，因此启动日志有时只显示 preflight 标记，长时间看不到 `Batch ...`。不要因
日志暂时不刷新而终止或重复启动训练，应同时检查真实 Python 进程和输出文件：

```bash
pgrep -af 'config.py --protocol_config'
ps -p PID -o pid=,etime=,etimes=,%cpu=,%mem=,rss=,stat=,cmd=
```

`%CPU` 可以高于 100；例如约 1500% 表示进程正在使用约 15 个逻辑 CPU，并不表示
异常。`R`、`Rl` 等运行状态、高 CPU 占用以及持续更新的验证文件共同说明任务仍在
工作。只看到进程存在仍不足以判定训练健康。

### 9.2 用验证文件判断真实进度

基础训练每 500 updates 验证一次，并在 batch 0 写入第一条记录。因此
`test_losses.txt` 有 `N` 行时，最近完成的验证 batch 为 `(N-1)*500`：

```bash
FULL10_DIR=/root/autodl-tmp/digit_writing_original_protocol_4b8b1db/runs/digit_writing_original_protocol2/full10/dev42
wc -l "$FULL10_DIR/test_losses.txt"
tail -n 12 "$FULL10_DIR/test_losses.txt"
stat -c '%y %s %n' "$FULL10_DIR/test_losses.txt" "$FULL10_DIR/best_checkpoint.pt"
```

验证文件更新时间推进、记录数增长且损失为有限值，才是训练进度的直接证据。
`best_checkpoint.pt` 只在当前验证损失不高于历史最佳值时覆盖；它没有在每次验证
后更新时间是正常现象。训练自然结束后才会生成 final continuation checkpoint。

### 9.3 收敛判断与剩余时间

不要用一次验证值或单次反弹判断收敛。至少比较最近 10–20 次验证：

- 最近窗口线性斜率为负、后半窗口均值低于前半窗口，表示仍在改善；
- 斜率接近零且窗口改善长期很小，表示接近平台；
- 连续多个窗口斜率为正、近期均值持续高于此前均值，才需要怀疑退化；
- 当前值略高于最近的历史最佳值通常只是正常波动，best checkpoint 会保留最佳点。

早期 ETA 容易受首次验证开销和不同数字 episode 长度影响，应在多个验证周期后再
按“已完成 batch / 进程 elapsed time”线性估计，并明确它只是粗略值。

协议指定 75000 updates 时，即使已接近平台也不能自行提前停止、调学习率或修改
协议。正式结果优先使用 best checkpoint；final continuation checkpoint 用于审计
和续跑状态保存。训练期间不要修改服务器仓库，否则 runner 结束时的仓库洁净门禁
会失败。

### 9.4 完成判定

Python 进程消失后，先检查启动日志末尾和正式标记，不要立刻重跑。有效完成必须
同时看到：

```text
RUN_EXIT=0
TEE_EXIT=0
FORMAL_RUN_COMPLETE=1
```

并确认结果归档及对应 `.sha256` 已生成。若训练本身完成但归档或洁净门禁失败，
应保留现有 checkpoint、日志和运行目录，先定位失败阶段，不得覆盖式重跑训练。

## 10. 查看日志而不影响终端

优先读取短状态标记，而不是输出整个日志：

```bash
grep -aE 'COMPLETE=|TEST_COUNT=|SKIPPED_TESTS=|EXIT=|Traceback|FAILED|ERROR' /root/autodl-tmp/stage.log
```

需要日志尾部时，先确认文件存在，再限制行数：

```bash
test -f /root/autodl-tmp/stage.log && tail -n 40 /root/autodl-tmp/stage.log
```

单独的 `tail` 不会关闭终端。此前看似由 `tail` 引发的“闪退”，更可能是交互
Shell 已启用 `set -e`，而日志路径不存在或命令返回非零。新终端中应先关闭严格
模式并检查路径。

`tail -f` 中按 `Ctrl+C` 只停止日志跟踪，不会停止 `nohup` 后台任务；前提是
交互 Shell 没有被错误地置于严格模式。

## 11. 成功与失败判定

不能仅凭后台 PID、进程消失、`screen` 状态、`tee` 返回 0 或压缩包生成判断
成功。

成功至少要求：

- 目标程序与日志管道退出码均为 0；
- 预期测试数完整且 skipped 为 0；
- 明确的阶段完成标记存在；
- Git 与 submodule HEAD 正确，工作区干净；
- 日志中没有 Traceback、FAILED、ERROR、Killed、NaN 或 Inf；
- 预期输出、manifest 和归档存在；
- 外层与内部 SHA-256 均通过。

失败后：

1. 不覆盖或删除失败目录；
2. 不立即重复同一命令；
3. 保存第一个错误、Git HEAD、环境路径和进程状态；
4. 修复后使用新的 `run2`、`retry2` 或新版本上传包；
5. 不使用 `rm -rf`、`git reset --hard` 等破坏性命令清理现场。

## 12. 结果归档和下载

每次有效运行至少保存 Git 与 submodule HEAD、原仓库基线、配置与哈希、软件和
设备信息、完整日志、真实退出码、逐文件 `SHA256SUMS` 及外层归档哈希。

服务器校验：

```bash
cd /root/autodl-tmp
sha256sum -c result-run1.tar.gz.sha256
```

下载到本地后再次计算 SHA-256，并独立解压验证内部 `SHA256SUMS`。只有本地备份
和校验完成后，才能考虑关闭或释放实例。

## 13. 当前项目协作方式

- 用户负责上传文件并在 AutoDL 终端执行命令。
- 助手负责生成可审计上传包、SHA-256 和逐条短命令。
- 助手不直接接管服务器，也不假设具有 SSH 权限。
- 每轮先验证上传包，再解压，再后台运行，再用短标记验收。
- 只有失败时才读取有限日志定位第一个错误。
- 实现和测试可以合并在一个审计脚本中，但不得把昂贵训练隐藏在安装或测试包
  中。
- 未经当前项目规定的人工审查和明确授权，不启动开发 seed、75000-update
  训练、组合优化、迁移或正式多种子实验。

## 14. 日常检查清单

启动前：

- [ ] 当前实例和数据盘空间正确；
- [ ] 仓库、submodule 和 CPU Python 路径存在；
- [ ] Git HEAD 与预期一致，工作区干净；
- [ ] PyTorch 为 `+cpu` 且 CUDA 可用性为 `False`；
- [ ] 上传包外层 SHA-256 通过；
- [ ] 目标 stage、运行目录和日志名不会覆盖已有证据；
- [ ] 没有同一任务的后台进程。

启动后：

- [ ] 记录后台 PID；
- [ ] 日志文件开始产生；
- [ ] preflight 标记与配置正确；
- [ ] 没有立即出现 Traceback 或环境路径错误。

完成后：

- [ ] 目标程序和日志管道退出码均为 0；
- [ ] 测试数量、skipped 和完成标记正确；
- [ ] 没有训练越过本阶段授权边界；
- [ ] 仓库干净；
- [ ] 结果归档和 SHA-256 通过；
- [ ] 已下载并在本地独立复核。

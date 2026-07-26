# AutoDL 服务器正确使用指南

## 1. 适用服务器

本指南适用于当前使用的 AutoDL 实例，以及配置完全相同的新卡：

- 系统：Ubuntu 22.04
- 基础镜像：Miniconda conda3 / Python 3.10（Ubuntu 22.04）
- GPU：NVIDIA GeForce RTX 4090 24 GB × 1
- CUDA：11.8
- CPU：20 vCPU，Intel Xeon Platinum 8470Q
- 内存：90 GB
- 系统盘：30 GB
- 数据盘：50 GB

CPU 型号以 AutoDL 当前实例页面和实例内命令为准。本实例不是旧资料中出现过的 Xeon Gold 6430；如果后续更换实例，也不能只根据旧指南推断硬件。

登录实例后可用以下命令快速核对实际资源：

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
lscpu | grep -E 'Model name|^CPU\(s\)'
free -h
df -h / /root/autodl-tmp
```

预期 GPU 型号应包含 `NVIDIA GeForce RTX 4090`，显存约为 24 GB。CPU 型号应包含 `Intel(R) Xeon(R) Platinum 8470Q`。平台可能对显示文字做轻微缩写，因此应核对型号和资源量，不要求标点完全一致。

训练文件、代码仓库、Conda 环境和结果均应优先放在数据盘：

```text
/root/autodl-tmp
```

不要把大型环境、模型或日志放在 `/root` 系统盘。系统盘只有 30 GB，写满后可能导致安装失败、程序异常或终端不可用。

---

## 2. AutoDL 实例、电脑和终端之间的关系

以下操作不会停止服务器后台任务：

- 关闭本地电脑；
- 关闭浏览器；
- 关闭 AutoDL 网页终端；
- SSH 连接断开；
- 从 `screen` 会话中分离。

前提是任务已放入 `screen` 等后台会话，并且 AutoDL 实例仍然保持开机。

以下操作会停止任务：

- 关闭或释放 AutoDL 实例；
- 实例发生重启、故障或被平台回收；
- 手动结束目标 Python 进程；
- 手动关闭对应的 `screen` 会话。

`screen` 只能防止 SSH 或网页终端断开，不能让任务跨实例关机继续运行。

如果历史训练入口没有保存完整的模型、优化器、迭代位置和随机数状态，实例停止后通常不能精确断点续训。

---

## 3. 网页终端中如何正确输入命令

### 3.1 不要复制终端提示符

终端中显示：

```text
root@autodl-container-xxxxxxxxxx:~/autodl-tmp#
```

这里只是提示符。粘贴命令时，不要把 `root@...#` 一起复制进去。

正确：

```bash
pwd
```

错误：

```text
root@autodl-container-xxxxxxxxxx:~/autodl-tmp# pwd
```

### 3.2 普通命令可以逐行输入

```bash
cd /root/autodl-tmp
pwd
df -h
```

每条命令执行结束后，先看输出和退出状态，再执行下一阶段。

### 3.3 示例中的中文目录名不能原样使用

指南中的下列写法只是占位示例：

```bash
REPO=/root/autodl-tmp/新的干净仓库目录
```

执行前必须把它替换为服务器上实际存在的仓库路径。例如：

```bash
REPO=/root/autodl-tmp/digit_writing_project_ea61b00

printf 'REPO=%s\n' "$REPO"
test -d "$REPO/.git" && echo "REPO_OK" || echo "REPO_NOT_FOUND"
test -f "$REPO/server/run_digit_seed42_cpu_30000.sh" && echo "SCRIPT_OK" || echo "SCRIPT_NOT_FOUND"
```

只有看到 `REPO_OK` 和预期的 `SCRIPT_OK` 后才继续。把“新的干净仓库目录”
原样赋给 `REPO` 并不会自动寻找仓库；后续 `bash` 或 `git -C` 只会报告目录或
文件不存在。

### 3.4 多行严格脚本应放入子 Shell

不要在交互终端中直接输入：

```bash
set -euo pipefail
```

原因：

- `-e`：任何命令返回非零状态时，当前 Shell 会立即退出；
- `-u`：引用未定义变量时，当前 Shell 会立即退出；
- `pipefail`：管道中任何子命令失败，整条管道都被视为失败；
- 在交互终端中按 `Ctrl+C` 也可能触发退出。

这就是之前网页终端看起来“闪退”或重新连接的主要原因之一。

正确方式是把严格脚本放进一个子 Shell：

```bash
bash <<'BASH'
set -euo pipefail

echo "在这里执行严格检查"
pwd

BASH
```

即使脚本失败，退出的也只是子 Shell，外层网页终端仍然保留。

结束标记 `BASH` 必须：

- 单独占一行；
- 前后不能有空格；
- 大小写与开头完全一致。

如果已经在交互 Shell 中启用了严格模式，并且终端尚未退出，可先恢复普通
交互模式：

```bash
set +e
set +u
set +o pipefail
set -o | grep -E 'errexit|nounset|pipefail'
```

三个选项都应显示为 `off`。如果网页终端已经关闭，重新打开一个终端即可；
先不要重复刚才的命令，应先检查 `REPO`、`OLD_REPO` 等变量是否为真实路径。

### 3.5 多行命令和 `>` 提示符

反斜杠 `\` 表示命令还没有结束。它必须是该行最后一个字符，后面不能有空格
或注释。以下写法可以作为一条完整命令一次粘贴：

```bash
git -C "$REPO" \
  -c submodule.mRNNTorch.url="$OLD_REPO/mRNNTorch" \
  submodule update --init mRNNTorch
```

不确定网页终端能否正确处理多行粘贴时，使用等价的单行写法更容易排错：

```bash
git -C "$REPO" -c submodule.mRNNTorch.url="$OLD_REPO/mRNNTorch" submodule update --init mRNNTorch
```

如果终端显示：

```text
>
```

通常表示引号、括号或 heredoc 尚未闭合。不要继续盲目粘贴。先检查是否缺少：

- 单引号 `'`；
- 双引号 `"`；
- 右括号；
- 独占一行的 `BASH`。

拿不准时按 `Ctrl+C` 取消当前未完成命令，然后重新粘贴完整命令块。

Git 命令本身正常情况下不会关闭网页终端。如果输入上述命令后终端立即消失，
应优先怀疑此前启用的严格模式与错误路径共同导致 Shell 退出，而不是直接判断
为 Git、submodule 或服务器故障。

### 3.6 `Ctrl+C`、`Ctrl+D` 和 `exit` 的正确理解

`Ctrl+C` 只中断当前前台命令。

常见情况：

- 在 `tail -f` 中按 `Ctrl+C`：只停止日志跟踪，不会停止后台训练；
- 在前台执行 Python 时按 `Ctrl+C`：会中断 Python；
- 在交互终端直接启用 `set -e` 后按 `Ctrl+C`：可能导致当前终端 Shell 退出；
- 在 `screen -r` 中不要用 `Ctrl+C` 退出会话，应使用 `Ctrl+A`，再按 `D` 分离。

`Ctrl+D` 和 `exit` 会关闭当前 Shell。它们不是从 `screen` 安全分离的方法；
如果关闭的是 `screen` 内最后一个 Shell，该 `screen` 会话也会结束。

---

## 4. 新卡初始化的正确顺序

新卡即使型号完全相同，也应视为一台全新的服务器。不要假定旧卡上的以下内容仍然存在：

- 上传文件；
- Git 仓库；
- Conda 环境；
- Python 包；
- `screen` 会话；
- checkpoint；
- 日志和运行目录。

推荐顺序：

1. 检查磁盘、CPU、内存和 GPU；
2. 上传 bundle、配置或必要数据；
3. 校验上传文件的 SHA-256；
4. 在数据盘建立独立仓库；
5. 在数据盘建立独立 Conda 环境；
6. 验证代码 commit、submodule commit 和依赖版本；
7. 使用新的、空的运行目录；
8. 在 `screen` 中启动；
9. 同时验证会话、进程、日志标记和首次评估结果；
10. 训练完成后检查退出码、完成标记和结果文件。

首先运行只读硬件检查：

```bash
cd /root/autodl-tmp
pwd
df -h
free -h
lscpu | head -n 20
nvidia-smi
```

端口映射对于普通终端训练不是必需的。只有运行 Jupyter、TensorBoard 或其他网页服务时，才需要配置并使用对应端口。

---

## 5. 文件上传与完整性校验

### 5.1 上传位置

使用 AutoDL 网页文件管理器时，优先上传到：

```text
/root/autodl-tmp
```

不要覆盖服务器上已有的同名文件。文件名应包含用途、commit 或日期，例如：

```text
project-methods-aligned-914abba.bundle
project-analysis-662c8b2.bundle
```

### 5.2 上传后必须校验 SHA-256

单文件检查：

```bash
sha256sum /root/autodl-tmp/文件名
```

严格比对：

```bash
echo "预期SHA256  /root/autodl-tmp/文件名" | sha256sum -c -
```

只有看到：

```text
/root/autodl-tmp/文件名: OK
```

才能继续。

上传完成不等于文件完整；网络中断、重复上传或选择错误文件都可能留下大小看似正常但内容错误的文件。

---

## 6. 仓库和 Git submodule 的使用

### 6.1 不覆盖非空目录

每次克隆前先确认目标不存在：

```bash
REPO=/root/autodl-tmp/新的仓库目录

if [ -e "$REPO" ]; then
    echo "ABORT: target already exists: $REPO"
else
    echo "TARGET_IS_FREE"
fi
```

如果目录已经存在：

- 不要继续克隆到其中；
- 不要直接删除；
- 不要复用为新的运行；
- 先查看和保留已有内容；
- 使用新的目录名，或在确认后把旧目录改名为带时间戳的保留目录。

### 6.2 GitHub 下载缓慢时使用离线 bundle

本次使用中，父仓库从本地 bundle 克隆成功，但 GitHub submodule 长时间卡在：

```text
Cloning into '.../mRNNTorch'...
```

经验是：

- 先用 `ps` 确认是否仍有 Git 进程；
- 可以按 `Ctrl+C` 终止卡住的前台克隆；
- 保留已经产生的目录，不覆盖；
- 为 submodule 单独生成并上传 Git bundle；
- 使用本地 bundle 完成离线初始化。

不要因为 GitHub 卡住就反复在同一目录中重新执行克隆。

### 6.3 commit 必须显式验证

```bash
git -C "$REPO" rev-parse HEAD
git -C "$REPO/mRNNTorch" rev-parse HEAD
git -C "$REPO" status --short
git -C "$REPO/mRNNTorch" status --short
```

正确状态应满足：

- 父仓库 commit 与预期完全一致；
- submodule commit 与预期完全一致；
- 两个 `status --short` 均没有输出。

`git status --short` 没有输出通常表示工作区干净，不是命令卡住。

---

## 7. Conda 环境和软件安装

### 7.1 环境放在数据盘

推荐使用明确的环境路径：

```bash
conda create -y \
  -p /root/autodl-tmp/conda/envs/环境名 \
  python=3.10
```

以后始终通过这个环境的 Python 执行：

```bash
/root/autodl-tmp/conda/envs/环境名/bin/python --version
/root/autodl-tmp/conda/envs/环境名/bin/python -m pip --version
```

不要混用：

- 系统 Python；
- Conda base 环境；
- 另一个实验的环境；
- 仅名称相似但路径不同的环境。

### 7.2 下载缓慢时使用国内镜像

普通 Python 包可以使用清华等镜像：

```bash
PYTHON=/root/autodl-tmp/conda/envs/环境名/bin/python

"$PYTHON" -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  包名
```

PyTorch 应根据项目需要选择 CPU 或 CUDA 版本及对应来源，不要仅因为服务器有 GPU 就默认安装 CUDA 版。

#### 7.2.1 本项目最快且可复现的安装方式

先检查项目的两个固定环境是否已经存在。版本和设备输出都正确时应直接复用，不要重复安装：

```bash
CPU_PYTHON=/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python
CUDA_PYTHON=/root/autodl-tmp/conda/envs/motor-compat-cu118/bin/python

if [ -x "$CPU_PYTHON" ]; then
  "$CPU_PYTHON" -c "import motornet,torch; print('CPU',torch.__version__,motornet.__version__,torch.cuda.is_available())"
fi

if [ -x "$CUDA_PYTHON" ]; then
  "$CUDA_PYTHON" -c "import motornet,torch; print('CUDA',torch.__version__,motornet.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
fi
```

预期结果：

- CPU 环境：PyTorch `2.6.0+cpu`、MotorNet `0.2.0`、CUDA 可用性为 `False`；
- CUDA 环境：PyTorch `2.6.0+cu118`、MotorNet `0.2.0`、CUDA 可用性为 `True`、GPU 为 RTX 4090。

新实例或环境缺失时，从干净的项目仓库根目录执行规范脚本。清华源只对当前命令临时生效；pip 缓存放在数据盘，可让第二个环境复用已经下载的普通依赖：

```bash
cd /root/autodl-tmp/digit-writing-stageA
mkdir -p /root/autodl-tmp/pip-cache

env \
  PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  PIP_CACHE_DIR=/root/autodl-tmp/pip-cache \
  PIP_DEFAULT_TIMEOUT=120 \
  bash server/create_compat_environment.sh cpu

env \
  PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  PIP_CACHE_DIR=/root/autodl-tmp/pip-cache \
  PIP_DEFAULT_TIMEOUT=120 \
  bash server/create_compat_environment.sh cuda
```

该脚本会：

1. 把两个独立 Conda 环境创建在 `/root/autodl-tmp/conda/envs`；
2. 从临时清华源安装完全固定版本的普通依赖；
3. 从 PyTorch 官方 CPU 或 CUDA 11.8 索引安装对应的固定 wheel；
4. 执行 `pip check`、打印完整依赖和关键版本；
5. 在安装被中断时允许同一命令续装，验证成功后写入完成标记并拒绝覆盖。

不要执行永久性的 `pip config set global.index-url ...`。永久改源可能影响之后的 PyTorch 构建选择，也会让环境来源难以审计。CUDA PyTorch 和 NVIDIA 运行库体积较大，即使普通依赖使用国内镜像，首次 CUDA 安装仍可能需要较长时间；这不代表命令卡死。

### 7.3 RTX 4090 存在不代表程序会使用 GPU

检查：

```bash
PYTHON=/root/autodl-tmp/conda/envs/环境名/bin/python

"$PYTHON" -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果安装的是 `torch+cpu`，输出：

```text
False
```

是正常现象。当前复现实验为了与既有 CPU 基线保持一致，主动使用 CPU；RTX 4090 不会被使用。

只有明确决定进行 GPU 实验时，才安装与驱动和项目兼容的 CUDA 版 PyTorch。CPU 与 GPU 的结果不应在没有说明的情况下混为同一条严格复现线路。

### 7.4 安装后验证

```bash
PYTHON=/root/autodl-tmp/conda/envs/环境名/bin/python

"$PYTHON" -m pip check
"$PYTHON" -m pip freeze
"$PYTHON" -c "import torch,numpy; print('torch',torch.__version__); print('numpy',numpy.__version__); print('cuda',torch.cuda.is_available())"
```

在 Conda 环境中以容器的 `root` 用户运行 pip 时，可能出现 root 用户警告。这不等于安装失败，但必须确认使用的是独立 Conda 环境，而不是系统 Python。

---

## 8. 运行目录和结果保护

每次运行都使用全新的目录，例如：

```text
/root/autodl-tmp/experiment_name_seed105_run1
/root/autodl-tmp/experiment_name_seed105_run2
```

启动前检查：

```bash
RUN_ROOT=/root/autodl-tmp/新的运行目录

if [ -e "$RUN_ROOT" ]; then
    echo "ABORT: run directory already exists: $RUN_ROOT"
else
    echo "RUN_DIRECTORY_IS_FREE"
fi
```

基本原则：

- 不覆盖已有 checkpoint；
- 不清空非空目录；
- 不把失败任务的目录直接用于重跑；
- 失败目录改名保留，记录失败原因；
- 新任务使用新的 `run2`、`retry2` 或时间戳目录；
- 不使用 `rm -rf` 处理拿不准的目录。

建议每个运行目录至少包含：

```text
preflight.txt
train.log 或 eval.log
exit_code.txt
elapsed_seconds.txt
配置文件
checkpoint 或结果文件
```

---

## 9. 使用 screen 在后台运行

### 9.1 推荐的启动方式

假设已有可执行脚本：

```text
/root/autodl-tmp/project/server/run_job.sh
```

使用独立的 `screen` 名称和日志：

```bash
bash <<'BASH'
set -euo pipefail

SCREEN_NAME=my_job_run1
SCREEN_LOG=/root/autodl-tmp/my_job_run1_screen.log
REPO=/root/autodl-tmp/project

if screen -ls 2>&1 | grep -q "\.${SCREEN_NAME}[[:space:]]"; then
    echo "ABORT: screen already exists: $SCREEN_NAME"
    exit 1
fi

screen -dmS "$SCREEN_NAME" \
  -L -Logfile "$SCREEN_LOG" \
  bash -lc "cd '$REPO' && bash server/run_job.sh"

screen -ls

BASH
```

这里的 `set -euo pipefail` 位于子 Shell 中，不会导致外层终端闪退。

### 9.2 常用 screen 命令

列出会话：

```bash
screen -ls
```

进入会话：

```bash
screen -r my_job_run1
```

进入后看到一个新的提示符和一整屏独立输出是正常现象。`screen` 提供的是同一台
AutoDL 实例中的虚拟终端，不是新服务器，也不会自动复制或重启任务。

从会话中安全分离：

```text
先按 Ctrl+A，再按 D
```

分离后任务继续运行。

如果旧网页连接仍把会话标记为 `Attached`，在新终端中执行：

```bash
screen -d -r my_job_run1
```

这会先分离旧连接，再把同一会话接到当前终端，不会启动第二份任务。

若 `screen -ls` 显示 `No Sockets found`，说明该 `screen` 会话已经结束。此时
不要立刻重跑；先按第 11、12 节检查 Python 进程、训练日志和退出码，因为任务
可能已经正常完成，也可能在报错后退出。

不要在任务正常运行时随意执行：

```bash
screen -S my_job_run1 -X quit
```

该命令会关闭指定会话，并可能终止其中的任务。

### 9.3 不要重复启动

每次启动前同时检查：

```bash
screen -ls
pgrep -af '目标脚本名.py' || true
```

如果已有对应会话或 Python 进程，先确认其身份，不要再次运行启动命令。

---

## 10. 日志查看

查看最后 50 行：

```bash
tail -n 50 /root/autodl-tmp/运行日志.log
```

实时跟踪：

```bash
tail -f /root/autodl-tmp/运行日志.log
```

停止实时跟踪：

```text
Ctrl+C
```

这只会停止 `tail -f`，不会停止已经在 `screen` 中运行的任务。

查找关键标记：

```bash
grep -F 'PYTHON_PROCESS_STARTED' /root/autodl-tmp/运行日志.log
grep -F 'Eval Results:' /root/autodl-tmp/运行日志.log | tail
grep -Ein 'traceback|exception|killed|segmentation fault|nan' /root/autodl-tmp/运行日志.log | tail -n 20
```

---

## 11. 如何确认任务真的正在运行

仅看到一个 `screen` 会话，不足以证明训练正常。

至少同时确认：

1. 指定名称的 `screen` 会话存在；
2. 目标 Python 进程存在；
3. 日志中出现 Python 启动标记；
4. 日志中的配置与计划一致；
5. 日志中已经出现首次评估结果。

示例：

```bash
screen -ls
pgrep -af '目标脚本名.py' || true
grep -F 'PYTHON_PROCESS_STARTED' /root/autodl-tmp/运行日志.log
grep -F 'CONFIG' /root/autodl-tmp/运行日志.log | head
grep -F 'Eval Results:' /root/autodl-tmp/运行日志.log | head
```

进程的 CPU 使用情况：

```bash
ps -eo pid,etime,%cpu,%mem,cmd |
grep -E '[p]ython.*目标脚本名.py'
```

在 20 vCPU 服务器上，多线程 CPU 任务的 `%CPU` 可以超过 `100%`。例如 `1285%` 表示大约使用了 12.85 个 CPU 核，并不代表异常。

---

## 12. 如何确认任务已经正常完成

训练结束后，`screen` 会话和 Python 进程消失通常是正常的。此时不要只根据“没有进程”判断成功或失败。

应检查：

```bash
RUN_ROOT=/root/autodl-tmp/运行目录

cat "$RUN_ROOT/exit_code.txt"
cat "$RUN_ROOT/elapsed_seconds.txt"
tail -n 50 "$RUN_ROOT/train.log"
ls -lh "$RUN_ROOT"
```

正常完成通常需要：

- `exit_code.txt` 为 `0`；
- 日志中存在明确的正常完成标记；
- 最后训练进度符合预期；
- 评估次数符合配置；
- checkpoint 和配置文件存在且非空；
- 日志中没有 traceback、Killed 或 NaN；
- checkpoint 可以被 Python 严格加载；
- 记录 checkpoint 的 SHA-256。

计算 checkpoint 哈希：

```bash
sha256sum /root/autodl-tmp/运行目录/model/checkpoint.pth
```

注意：如果训练代码只保存“验证最优 checkpoint”，保存时间早于训练结束时间是正常的。它代表验证最优状态，不一定是最后一次迭代的模型。

---

## 13. 结果下载和归档

训练或评估完成后，优先下载：

- checkpoint；
- `hp.json` 或其他配置文件；
- `train.log` / `eval.log`；
- `preflight.txt`；
- `exit_code.txt`；
- `elapsed_seconds.txt`；
- 原始指标数据；
- 图像和摘要；
- SHA-256 记录。

归档前不要覆盖已有压缩包：

```bash
ARCHIVE=/root/autodl-tmp/job_results_run1.tar.gz
RUN_ROOT=/root/autodl-tmp/job_results_run1

if [ -e "$ARCHIVE" ]; then
    echo "ABORT: archive already exists: $ARCHIVE"
else
    tar -C /root/autodl-tmp \
      -czf "$ARCHIVE" \
      "$(basename "$RUN_ROOT")"
    sha256sum "$ARCHIVE"
fi
```

下载后应在本机再次计算 SHA-256，与服务器记录比对。

---

## 14. 本次使用中最重要的经验教训

### 14.1 网页终端“闪退”通常不是服务器故障

最常见原因是在交互 Shell 中直接启用了：

```bash
set -euo pipefail
```

随后某条检查失败、引用了未定义变量或按下 `Ctrl+C`，可能导致整个交互 Shell
退出。实际出现过的典型组合是：把示例中的中文占位目录原样赋给 `REPO`，在
交互 Shell 已启用严格模式的情况下运行 `git -C "$REPO"` 或仓库脚本；路径检查
失败后，网页终端看起来像“闪退”。

严格命令块应始终放入 `bash <<'BASH'` 子 Shell。重新打开终端后先确认严格模式
为关闭状态，再验证实际路径；不要未经检查就重复整段命令。Git 或 submodule
命令失败本身不等于服务器故障。

### 14.2 `screen` 只解决连接中断，不解决实例关机

关闭电脑和网页不会影响后台任务；关闭 AutoDL 实例一定会停止任务。长任务期间必须保持实例开机。

### 14.3 GitHub 卡住时不要反复覆盖

GitHub 或 submodule 下载缓慢并不说明父仓库有问题。应停止卡住的前台命令、保留现场、检查进程，并改用离线 bundle。

### 14.4 国内镜像能解决大部分 pip 下载问题

普通依赖可以切换清华等镜像。PyTorch 应单独选择准确版本和 CPU/CUDA 构建，不能只执行一个模糊的 `pip install torch`。

### 14.5 相同型号新卡仍然需要完整初始化

硬件型号相同只意味着资源相近，不意味着环境、文件、随机状态和依赖已经复制。每张新卡都必须重新：

- 上传并校验文件；
- 创建仓库；
- 初始化 submodule；
- 创建 Conda 环境；
- 安装并验证依赖；
- 使用新的运行目录；
- 记录完整 preflight。

### 14.6 “程序启动了”与“任务正常运行”不是一回事

必须同时看到 `screen`、Python 进程、启动标记、正确配置和首次评估，才能确认训练进入正常运行状态。

### 14.7 极小的加权 loss 不等于物理误差很小

日志中的损失可能乘有很小的权重。检查结果时必须同时查看未加权的物理指标、轨迹图和逐任务表现，不能仅依据科学计数法中的小数值判断成功。

### 14.8 所有重跑都使用新目录

已有目录无论成功、失败还是只运行了一部分，都不应直接覆盖。保留旧输出是排查问题和维护证据链的基础。

---

## 15. 推荐的日常操作清单

启动前：

- [ ] 当前实例型号和磁盘空间正确；
- [ ] 文件已上传到 `/root/autodl-tmp`；
- [ ] SHA-256 校验通过；
- [ ] 仓库和 submodule commit 正确；
- [ ] 两个 Git 工作区干净；
- [ ] Python 路径和依赖版本正确；
- [ ] 目标运行目录不存在；
- [ ] 没有同名 `screen`；
- [ ] 没有重复的目标 Python 进程。

启动后：

- [ ] `screen` 会话存在；
- [ ] 目标 Python 进程存在；
- [ ] 启动标记存在；
- [ ] 配置日志正确；
- [ ] 首次评估已出现；
- [ ] 日志中没有异常。

完成后：

- [ ] 退出码为 `0`；
- [ ] 正常完成标记存在；
- [ ] 最后进度和评估次数正确；
- [ ] checkpoint 和结果文件存在；
- [ ] 没有 traceback、Killed 或 NaN；
- [ ] 记录所有重要文件的 SHA-256；
- [ ] 下载并校验结果；
- [ ] 确认本地备份完成后，再决定是否关闭或释放实例。

---

## 16. 正式运行的审计与失败保护规则

以下规则适用于兼容性门禁、开发种子和正式种子，是服务器操作步骤之外的
工程审计要求。

### 16.1 代码目录与运行目录必须分离

- 代码目录只保存 Git 工作树和只读输入；
- 每次运行使用全新的唯一目录；
- 配置、日志、checkpoint、结果和审计文件全部写入该运行目录；
- 不在代码目录中保存正式输出，不复用失败或中断过的运行目录。

### 16.2 正式输入必须不可变

运行前记录并在运行后复核：

- 父仓库 commit；
- `mRNNTorch` submodule commit；
- `protocol.yaml`、`protocol.md` 及其他正式配置的 SHA-256；
- Python、PyTorch、MotorNet、CUDA 和 GPU 信息；
- 完整启动命令、seed、开始时间和结束时间。

父仓库、submodule、协议文件或配置哈希发生变化时，当前运行不得继续归入
原实验批次。

### 16.3 启动包装器必须保留真实退出状态

- Python 使用无缓冲输出，例如 `PYTHONUNBUFFERED=1` 或 `python -u`；
- 使用 `tee` 时必须启用 `set -o pipefail`，并在管道结束后立即保存
  `PIPESTATUS`；
- 同时记录 Python 命令退出码和日志管道退出码；
- 只有目标程序退出码为 `0` 才能进入结果验收，不能以 `tee` 成功代替训练成功；
- 正式脚本必须写出明确的启动标记和正常完成标记。

### 16.4 失败后保留现场

失败、NaN、进程被杀或人工中止后：

1. 不覆盖、不删除原运行目录；
2. 保存当前 commit、submodule、配置哈希、进程、GPU、日志尾部和已有输出；
3. 记录失败类别和最后成功阶段；
4. 修复后使用新的运行目录和新的 `run_id`；
5. 若修复影响正式结果，废弃并重新运行全部受影响的正式 seed。

### 16.5 成功必须经过结果验收

退出码为 `0` 仍不等于实验有效。归档前至少确认：

- 预期条件、迭代和验证次数完整；
- checkpoint、逐条件指标、轨迹图和配置快照齐全；
- 日志中没有 traceback、Killed、NaN 或缺失条件；
- 关键物理指标合理，不能只查看加权总 loss；
- 逐文件 SHA-256 和归档文件 SHA-256 已生成；
- 下载后在本地重新校验归档哈希和成员列表。

---

## 17. 数字书写闭环集成门禁

该门禁只用于开发种子 `42` 的环境、梯度、动力学和冻结接口检查，不会启动
长训练、正式种子或正式迁移实验。

在一个全新、干净且 submodule 已初始化的仓库副本中执行：

```bash
REPO=/root/autodl-tmp/你的全新仓库目录
cd "$REPO"

git status --short
git rev-parse HEAD
git -C mRNNTorch rev-parse HEAD

bash server/run_digit_integration_gate.sh "$REPO" cuda
```

CUDA 环境不可用时可把最后一个参数改为 `cpu`。脚本会依次：

1. 验证父仓库、submodule 和工作区；
2. 在固定 CPU 环境中运行完整单元测试，再在所选 CPU/CUDA 环境中运行动态门禁；
3. 枚举 `10 digits × 2 directions × 3 speeds × 3 delays = 180`
   个无噪声闭环条件；
4. 执行 10 次反向传播与参数更新；
5. 检查保存加载、独立 `r5` 冻结范围及三系数接口；
6. 在 `/root/autodl-tmp/digit_writing_integration_runs/` 下保存合并证据，
   并生成归档及 SHA-256。

验收时必须同时满足：

- `tests_exit_code.txt` 和 `gate_exit_code.txt` 均为 `0`；
- `gate/integration_gate.json` 中 `"passed": true`；
- `gate/run_manifest.json` 中 `"formal_seeds_started": false`；
- `SHA256SUMS` 与归档哈希复核通过；
- 日志中没有 traceback、Killed、NaN 或 Inf。

门禁通过后仍不能直接启动正式种子；下一步只是运行从随机初始化开始的
seed `42` 完整条件短程演练，并据此冻结正式协议。

## 18. 开发种子 42 完整条件短程演练

该阶段从随机初始化开始，只运行配置中固定的 500 次更新，并在
`0/250/500` 次更新时完整验证 162 个基础训练条件。它用于检查早期学习
趋势和 loss 量级，不是正式训练。

```bash
REPO=/root/autodl-tmp/新的干净仓库目录
bash "$REPO/server/run_digit_seed42_short.sh" "$REPO" cuda
```

脚本会先校验已通过的 CUDA 集成门禁归档及其内部 PASS 状态，再运行完整
单元测试和短程训练。只有 `TEST_EXIT_CODE=0`、`TRAIN_EXIT_CODE=0` 且
`training/run_manifest.json` 中 `"passed": true` 时，短程演练才通过。
该阶段不得启动正式种子，也不得把生成的 checkpoint 当作正式模型。

## 19. update-10000 CPU/GPU 非正式短基准

该基准只比较当前完整训练更新与 162 条件验证在 CPU、CUDA 上的耗时。它从
同一个已审计 update-10000 checkpoint 分别初始化两个设备，训练计时产生的
参数随后丢弃，并在完整验证前重新加载原 checkpoint。CPU 与 CUDA 的随机数
流不同，因此结果不能作为精确续训或模型选择证据。

```bash
REPO=/root/autodl-tmp/新的干净仓库目录
SOURCE_ARCHIVE=/root/autodl-tmp/digit_writing_seed42_long_horizon_6e0b11d5fa25728f9e3f2fc84f0b39c7e20212cf_cuda.tar.gz

bash "$REPO/server/run_digit_cpu_gpu_benchmark.sh" \
  "$REPO" \
  "$SOURCE_ARCHIVE"
```

脚本会依次：

1. 校验源归档 SHA-256、仓库、submodule 和完整单元测试；
2. 在固定 CPU 环境中预热后计时 36 个 batch-size 32 的完整训练更新；
3. 重新加载 update-10000 checkpoint，计时全部 162 个验证条件；
4. 在固定 CUDA 环境中重复相同流程；
5. 写出 `cpu_result.json`、`cuda_result.json` 和 `comparison.json`；
6. 生成逐文件 `SHA256SUMS`、结果归档及归档 SHA-256。

完成后需要保存终端输出中的：

```text
SOURCE_COMMIT
SOURCE_ARCHIVE_SHA256
RUN_ROOT
ARCHIVE
TEST_EXIT_CODE
CPU_EXIT_CODE
CUDA_EXIT_CODE
COMPARE_EXIT_CODE
FORMAL_SEEDS_STARTED
```

四个退出码都必须为 `0`，且 `FORMAL_SEEDS_STARTED=false`。判断设备速度时
分别读取 `comparison.json` 中的 `training_update` 和 `full_validation`；
不得仅用其中一个耗时推断另一阶段，也不得从该基准启动后续训练。

## 20. seed 42 CPU 从零训练到 update 30000

该阶段建立一条新的非正式 CPU 开发谱系。它从 seed `42` 的随机初始化开始，
不读取任何 CUDA 或 CPU checkpoint。学习率在固定更新边界切换：

```text
updates 1-500       0.001
updates 501-1000    0.0005
updates 1001-30000  0.0001
```

验证点为 `0/250/500/750/1000`，之后每 1000 次更新验证一次，直至
`30000`。每个验证点都保存 checkpoint；即使模型选择最终指向更早的更新，
`step_030000.pt` 也必须保留，供结果审计后选择是否精确接续。

在新的干净仓库副本中运行：

```bash
REPO=/root/autodl-tmp/新的干净仓库目录

bash "$REPO/server/run_digit_seed42_cpu_30000.sh" \
  "$REPO" \
  cpu
```

该任务预计运行约 9–9.5 小时，建议在独立 `screen` 会话中启动。不要在交互
Shell 中执行 `set -euo pipefail`；严格模式已经封装在运行脚本内部。

本次任务使用的 `screen` 名为 `digit_cpu_30000`。进入 `screen` 后看到独立终端
是预期行为。确认训练已启动后，用 `Ctrl+A`、再按 `D` 分离；随后可以关闭网页、
浏览器或本地电脑，但不能关闭、释放或重启 AutoDL 实例。

以后打开新的网页终端，可以用以下命令恢复查看：

```bash
screen -ls
screen -r digit_cpu_30000
```

如果显示会话仍为 `Attached`，使用：

```bash
screen -d -r digit_cpu_30000
```

不进入 `screen` 也可以检查当前进度：

```bash
RUN_ROOT=/root/autodl-tmp/digit_writing_seed42_cpu_30000_runs/ea61b00edeb5527b7f581d8c0d90181a96f93d5e_cpu

grep 'SEED42_CPU_30000_TRAIN' "$RUN_ROOT/training.log" | tail -n 1 || true
grep 'SEED42_CPU_30000_VALIDATION' "$RUN_ROOT/training.log" | tail -n 1 || true
pgrep -af 'digit_writing.seed42_cpu_30000' || true
```

实时查看日志可执行：

```bash
tail -f "$RUN_ROOT/training.log"
```

这里按 `Ctrl+C` 只停止 `tail -f`。训练完成后才会出现：

```bash
cat "$RUN_ROOT/training_exit_code.txt"
```

其中 `0` 表示训练进程正常返回，但仍需按下述验收边界审计完整归档。不要因为
暂时没有新的验证行就重启任务；应同时根据最后一条训练记录和 Python 进程判断。

运行结束后必须保存：

```text
SOURCE_COMMIT
DEVICE
RUN_ROOT
ARCHIVE
TEST_EXIT_CODE
TRAIN_EXIT_CODE
DEVELOPMENT_CPU_FROM_SCRATCH_30000_STARTED
FORMAL_SEEDS_STARTED
```

验收边界：

- `TEST_EXIT_CODE=0`、`TRAIN_EXIT_CODE=0`；
- `DEVICE=cpu`、`FORMAL_SEEDS_STARTED=false`；
- `training/training_metrics.csv` 正好包含 30000 行更新；
- `training/validation_metrics.csv` 正好包含 `34 × 162 = 5508` 行；
- checkpoint 集合与 34 个验证点严格一致；
- `run_manifest.json` 声明 `initialized_from_checkpoint=false`；
- `step_030000.pt` 包含模型、Adam、Python/NumPy/Torch CPU RNG、条件池和
  条件计数状态；
- 外层归档 SHA-256 与内部 `SHA256SUMS` 均通过复核。

该任务完成只表示开发诊断有效，不冻结协议，也不得启动正式种子。

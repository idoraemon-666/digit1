# digit_writing_original_protocol2

本分支是最终数字轨迹与阶段归一化位置损失协议的独立项目。它复用原仓库的 MotorNet、mRNNTorch、28 维输入、基础训练循环和冻结迁移框架，但不复用上一项目的 checkpoint、optimizer state 或训练结果。

权威文件：

- `PROJECT_PROTOCOL.md`：项目 2 科学与实施规范；
- `digit_writing/digit_geometry_final.py`：0–9 几何、尺度、速度和分段采样唯一来源；
- `CODEX_MINIMAL_FINAL_DIGIT_MODIFICATION.md`：本轮决定的原始实施指令。

核心变化：

- 使用最终 A/B 共享曲线、独立数字 8、4:3 与 5:4 椭圆几何；
- 每个 segment 按线性弧长和物理速度独立确定 interval；
- movement 同时监督首末轨迹点，hold 从下一步开始；
- 主位置项采用 `0.1/0.1/0.6/0.2` 四阶段归一化 L1；
- 组合固定为 `{0,4,6,9,8}` 的五次 leave-one-out；
- full10 与 heldout5 仍完全独立。

主要实现：

- `digit_writing/geometry.py`：最终几何到环境接口的缓存薄适配；
- `digit_writing/phase_normalized_loss.py`：阶段损失和诊断指标；
- `digit_writing/final_protocol_audit.py`：工作空间与高风险短闭环审计；
- `envs.py`：28 维环境与最终 movement/hold 时序；
- `train.py`：基础训练、验证和数字 5 迁移；
- `digit_writing/experiments.py`：固定五任务组合入口。

服务器继续使用仓库目录：

```text
/root/autodl-tmp/digit_writing_original_protocol_4b8b1db
```

项目 2 的运行输出固定进入：

```text
runs/digit_writing_original_protocol2/
```

训练前审计入口：

```bash
bash server/run_digit_writing_original_protocol2_audit.sh \
  /root/autodl-tmp/digit_writing_original_protocol_4b8b1db \
  /root/autodl-tmp/digit-writing-original-protocol2-audit-run1
```

当前停止线：尚未启动项目 2 的 75,000 updates 正式训练。每个正式 exact run label 仍需用户单独授权。

# Protocol3 corner-ease v3 八数字共享 RNN 训练设计

## 1. 实验身份

- 仅训练数字 `1, 2, 3, 4, 5, 6, 7, 9`；数字 `0`、`8` 不进入训练、验证和 checkpoint 选择。
- 使用一个从头初始化的共享 RNN 和一个共享 Adam optimizer；不加载任何单数字 checkpoint。
- 保留 10 维数字 rule one-hot，因此模型输入维度仍为 28；排除数字的输入位不删除，只是不被采样。
- 固定 corner-ease v3、scale `2.5`、ref `100`、方向索引 `0`、delay `50`。
- 固定 batch size `8`。一个 optimizer step 的整个 batch 只属于一个数字。

## 2. 更新量与均衡调度

- 每个数字贡献 8,000 个 optimizer step，共 `8 × 8,000 = 64,000` 个共享 optimizer step。
- 每个数字实际训练样本数为 `8,000 × batch 8 = 64,000`，八数字共 512,000 个样本。
- 每连续 8 个 global step 构成一个 block；每个 block 内八个数字恰好各出现一次。
- block 内顺序由 seed `42` 和 block index 通过 `sha256_ranked_permutation_v1` 唯一确定，不依赖 Python hash 或全局随机状态。
- checkpoint 只能落在完整 block 边界；每次验证都核对八数字计数完全相等、0/8 为零、全部 delay 为 50。

## 3. 学习率

- global step `1–48,000`：`1e-3`，等价于每个数字前 6,000 次暴露。
- global step `48,001–64,000`：`3e-4`，等价于每个数字后 2,000 次暴露。
- 在第 48,000 步之后只修改 Adam parameter group 的 `lr`；不重建 optimizer，不清空一阶或二阶矩。
- checkpoint 记录完整 optimizer state。验证日志同时记录刚完成 step 的学习率和下一 step 的学习率，避免边界歧义。

## 4. 数字时长

| 数字 | movement intervals |
|---:|---:|
| 1 | 60 |
| 2 | 210 |
| 3 | 180 |
| 4 | 200 |
| 5 | 230 |
| 6 | 200 |
| 7 | 130 |
| 9 | 200 |

这些数值直接来自当前 corner-ease v3 几何，不做 joint-training 专用重采样。

## 5. 验证与共同 checkpoint

- 在 update `0, 800, 1,600, …, 64,000` 做固定 seed、无网络噪声、方向 0、delay 50、batch 8 的八数字验证。
- 每个数字通过条件：normalized mean error `≤ 0.08`、normalized endpoint error `≤ 0.05`、path-length ratio 位于 `[0.85, 1.15]`。
- 八数字共同通过连续 3 次验证才记为 `STABLE_PASS`；只有孤立共同通过则为 `PASS_UNSTABLE`；从未共同通过则为 `FAIL`。
- 所有候选都必须是同一个共享 checkpoint。候选排序依次为：
  1. 通过数字数最多；
  2. 最坏相对阈值超限最小；
  3. 八数字 macro normalized mean error 最小；
  4. 完全相同时选择更早 update。
- 若存在稳定共同通过区间，只在稳定区间内选择；否则在共同通过候选中选择；仍不存在时在全部验证点中选择。

## 6. 完成边界与结论限制

- 固定跑满 64,000 steps；不早停、不自动延长、不自动启动第二 seed、不自动启动 full10。
- best、final、initial checkpoint 分开保存；best 还要逐数字做零 optimizer step 的确定性轨迹与安全审计。
- 本实验只能回答固定方向 0、固定 delay 50 下八数字共享拟合是否接近单数字结果，不能外推到其他方向、其他 delay、数字 0/8 或完整十数字训练。
- 主要风险是共享容量与梯度干扰；共同 checkpoint 规则能如实暴露该风险，但不能保证八个数字都达到单数字最优值。
- 八数字的 optimizer step 数严格相等，但 movement intervals 不同，因此实际仿真时间步和计算量不相等；phase-normalized position loss 可避免位置损失被长度直接放大，但这仍不是“每个时间步等权”的实验。

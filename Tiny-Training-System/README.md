# tiny-training-system

训练加速与分布式层：kernel 融合 / Triton / FlashAttention / DDP / TP / MoE / profiling。

## 预计来源

- `assignment2-systems/` —— profiling / Triton / FlashAttention / DDP / 优化器 state 分片
- `cse234-w25-PA/pa2/` Part 1（Triton matmul+ReLU+add）+ Part 2（朴素 TP 通信布线 → 换 torch.distributed）
- `cse234-w25-PA/pa3/` Part 1（MoE TP/EP）+ Part 3（speculative decoding）

## 当前状态

空壳，等待迁入。

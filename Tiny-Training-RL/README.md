# tiny-training-rl

RL / alignment 层：SFT / Expert Iteration / GRPO / DPO / rollout。

## 预计来源

- `assignment5-alignment/` —— SFT / EI / GRPO / DPO adapters + Dr.GRPO grader

## 定位（相对 verl）

- 算法层对齐 verl（GRPO loss / advantage / response mask 全做）
- 编排层简化：先解决 policy 权重 ↔ vLLM rollout 同步
- 并行层复用 `tiny-training-system` 的 TP/DP/MoE 产出

## 当前状态

空壳，等待迁入。

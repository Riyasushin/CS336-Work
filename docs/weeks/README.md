# 周任务索引

| 周                 | 主题                                            | 截止       | 主要 stubs                                                                                                       | 主要测试                                                                                        |
| ------------------ | ----------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [Week 1](week1.md) | RMSNorm + RoPE + Attention + GQA                | 2026-04-26 | `tt/nn/` (Linear, Embedding, RMSNorm, RoPE, MHA, MHA+RoPE)                                                       | `tests/basics/test_model.py` 部分, `test_nn_utils.py`                                           |
| [Week 2](week2.md) | SwiGLU + Block + TransformerLM + Loss + AdamW   | 2026-05-03 | `tt/nn/` (SwiGLU, TransformerBlock), `tt/models/`, `tt/optim/`, `tt/utils/`                                      | `tests/basics/test_model.py` 剩余, `test_optimizer.py`, `test_serialization.py`, `test_data.py` |
| [Week 3](week3.md) | Triton Fused RMSNorm + FlashAttention-2 Forward | 2026-05-10 | `tt/kernels/` (fused_rmsnorm, FlashAttention{PyTorch,Triton}.forward)                                            | `tests/systems/test_attention.py::test_flash_forward_*`                                         |
| [Week 4](week4.md) | FlashAttention-2 Backward + Benchmark           | 2026-05-17 | `tt/kernels/FlashAttention*.backward` + `scripts/benchmark_attention.py` (自写)                                  | `tests/systems/test_attention.py::test_flash_backward_*`                                        |
| [Week 5](week5.md) | Bucketed DDP + ZeRO-1                           | 2026-05-24 | `tt/parallel/containers.py` (DDP, ShardedOptimizer)                                                              | `tests/systems/test_ddp.py`, `test_sharded_optimizer.py`                                        |
| [Week 6](week6.md) | FSDP + (stretch) Tensor Parallelism             | 2026-05-31 | `tt/parallel/containers.py` (FSDP); 可选 PA3 ShardedLinear                                                       | `tests/systems/test_fsdp.py`; 可选 `tests/moe/`（预热）                                         |
| [Week 7](week7.md) | MoE + Speculative Decoding                      | 2026-06-07 | `tt/moe/layers.py::ShardedLinear`, `tt/moe/models.py::{MoE_TP,MoE_EP}.forward`, `tt/decoding/speculative_decode` | `tests/moe/` 全绿（现在 4/5 红，实现后变绿），`notebooks/pa3_speculative_decoding.ipynb`        |
| [Week 8](week8.md) | SFT + GRPO + Repo 整合                          | 2026-06-14 | `tt/rl/` 全部                                                                                                    | `tests/alignment/` 全绿（需 `uv sync --extra alignment` + `ALIGNMENT_MODEL_ID`）                |

## 通用节奏

每周的 loop：
1. 读本周 doc 的 "要实现的 stubs" 表，对应打开 `tt/<module>/` 里的 `NotImplementedError`。
2. 读 "参考材料" 列出的 PDF 对应章节。
3. 填 stub → `uv run pytest <本周测试>` → 红变绿。
4. 全绿后跑一遍 `uv run pytest tests/moe/test_moe.py::test_simple_moe`，确认没破坏前面的工作。

## 全局资源

- **PDF 总集**：`docs/cs336/` 有 assignment 1/2/4/5 + supplement；`docs/cse234/` 有 PA2/PA3。
- **路径映射**：`docs/PATHS.md`（旧 `cs336_basics/` ↔ 新 `tt/nn/` 等）。
- **adapter 原理**：`tests/<name>/adapters.py` 是薄代理；测试靠 `from .adapters import run_*` 拿到契约，你实现 `tt/*` 后 adapter 自动生效。
- **分布式测试 harness**：`tests/_spawn.py::run_distributed` — Week 5/6/7 都用这个起进程组。

## 止损建议

| 情况                                    | 对策                                                                                                             |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Week 3 Triton FA2 forward 卡住 > 3 天   | 只交 `fused_rmsnorm` 和 `FlashAttentionPyTorch.forward`；`FlashAttentionTriton.forward` 先留 TODO，直接进 Week 4 |
| Week 4 backward 算不对                  | 只做 PyTorch 版，Triton backward 标 WIP，进 Week 5                                                               |
| Week 6 FSDP mixed-precision hook 太绕   | 先交 `compute_dtype=None` 的 fp32 路径，fp16 参数化测试 skip，进 Week 7                                          |
| Week 7 MoE_EP all-to-all packing 调不出 | 先交 MoE_TP（TP 更简单），EP 标 WIP，尾声再补                                                                    |

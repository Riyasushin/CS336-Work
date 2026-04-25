# tiny-training-system

训练加速与分布式层：kernel 融合 / Triton / FlashAttention / DDP / FSDP / TP / MoE / profiling。

## 来源

- `assignment2-systems/` —— profiling / Triton FlashAttention / DDP / FSDP / 优化器 state 分片（ZeRO-1）
- `cse234-w25-PA/pa2/` Part 1（Triton matmul+ReLU+add）+ Part 2（朴素 TP 通信布线 → 换 `torch.distributed`）
- `cse234-w25-PA/pa3/` Part 1（MoE TP/EP）+ Part 3（speculative decoding）

## 当前状态

已迁入 A2 的 tests + fixtures + adapters（8 个 adapter stub 全部 `raise NotImplementedError`）。
`uv run --directory Tiny-Training-System pytest` → **14 个测试全 FAIL**：

| 文件                              | 用例数 | 失败形式                              |
| --------------------------------- | ------ | ------------------------------------- |
| `tests/test_attention.py`         | 6      | NotImplementedError（直接）           |
| `tests/test_ddp.py`               | 2      | ProcessRaisedException（mp.spawn 内） |
| `tests/test_fsdp.py`              | 4      | ProcessRaisedException（mp.spawn 内） |
| `tests/test_sharded_optimizer.py` | 2      | ProcessRaisedException（mp.spawn 内） |

PA2/PA3 尚未迁入。

## Adapter 覆盖面

`tests/adapters.py`：

- `get_flashattention_autograd_function_pytorch()` / `get_flashattention_autograd_function_triton()`
- `get_ddp(module)` / `ddp_on_after_backward(ddp_model, optimizer)`
- `get_fsdp(module, compute_dtype=None)` / `fsdp_on_after_backward(...)` / `fsdp_gather_full_params(...)`
- `get_sharded_optimizer(params, optimizer_cls, **kwargs)`

每实现一个就把 `raise NotImplementedError` 换成对 `tiny_training_system.<module>` 里实现的调用。

## 依赖说明

- `tiny-training-basic`（workspace dep）—— `test_fsdp.py` 里 `ToyFSDPModel` 用 `RLinear / REmbedding / RRMSNorm`
- `einops` —— `test_attention.py` 用 `einsum`
- `triton`（Linux only）—— Triton FlashAttention

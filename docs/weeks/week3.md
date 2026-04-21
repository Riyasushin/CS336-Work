# Week 3 — Triton Fused RMSNorm + FlashAttention-2 Forward

**截止**：2026-05-10

**一句话目标**：上 Triton。先写一个 fused RMSNorm 练手，再做 FlashAttention-2 的前向（PyTorch 纯版 + Triton 版）。

---

## 要实现的 stubs

### `tt/kernels/__init__.py`

| 符号 | 作用 | 参考 |
|---|---|---|
| `def fused_rmsnorm(x, weight, eps) -> Tensor` | Triton kernel：把 RMSNorm 的 variance 计算 + 归一化 + 仿射融进一个 kernel | `docs/cs336/assignment2_systems.pdf` §2（Triton 入门章） |
| `class FlashAttentionPyTorch(torch.autograd.Function).forward` | 分块版 Attention 的纯 PyTorch 实现（用 `@staticmethod`，按 autograd.Function 的 ctx 约定） | §3.1（FA2 forward） |
| `class FlashAttentionTriton(torch.autograd.Function).forward` | 同上但用 Triton kernel 实现 | §3.2 |

本周**不**碰 `.backward` —— 那是 Week 4 的事。stub 里保留 `NotImplementedError` 不动。

---

## 跑绿

```bash
uv run pytest tests/systems/test_attention.py::test_flash_forward_pass_pytorch -v
uv run pytest tests/systems/test_attention.py::test_flash_forward_pass_triton -v
```

（Triton forward 参数化成两个 case，`[False]` / `[True]` 区分 causal mask。）

### 测试 → adapter → 你的 stub

| 测试 | adapter | stub |
|---|---|---|
| `test_flash_forward_pass_pytorch` | `get_flashattention_autograd_function_pytorch` | `tt.kernels.FlashAttentionPyTorch.forward` |
| `test_flash_forward_pass_triton[*]` | `get_flashattention_autograd_function_triton` | `tt.kernels.FlashAttentionTriton.forward` |

测试会调 `ClassName.apply(Q, K, V, is_causal)` 这样的接口，别忘了 Triton 版本也要走 `autograd.Function.apply` 协议。

---

## 参考材料

- **PDF**：`docs/cs336/assignment2_systems.pdf` §2（Triton 基础）、§3.1（FA2 forward 公式）
- **CSE234 的 Triton 教学笔记本**：`notebooks/pa2_matmul_triton.ipynb` —— 从最简单的 block-wise matmul 开始，比 CS336 的推导更手把手；Triton 生手建议先走一遍这个再开 FA2
- **路径映射**：`docs/PATHS.md`

---

## 验证

```bash
# 本周
uv run pytest tests/systems/test_attention.py -k forward -v

# 回归
uv run pytest tests/basics -q  # Week 1/2 不能破
uv run pytest tests/moe/test_moe.py::test_simple_moe -v
```

---

## Done 的定义

- `tests/systems/test_attention.py` 下所有 `*_forward_*` 测试绿（3 个：pytorch、triton[False]、triton[True]）。
- `test_attention.py` 下 `*_backward_*` 测试仍然报 `NotImplementedError`（预期 —— Week 4 再做）。
- Week 1/2 的回归全绿。

---

## 小贴士

- **硬件要求**：Triton 需要 CUDA GPU；MacBook / 纯 CPU 跑不动 Triton 测试。没卡就只做 `FlashAttentionPyTorch.forward`，Triton 版本标 WIP。
- **数值容差**：FA2 的分块累加和原生 SDPA 的结果 `atol=1e-4` 级别可接受；测试里已经放宽了。
- **causal mask 位置**：mask 要在 score 上（softmax 前），不是结果上。分块版里，block (i, j) 全部 masked 时可以直接跳过 —— 是主要的速度来源。
- **止损**：如果 Triton forward 连 3 天没拿下，交 PyTorch 版 + fused_rmsnorm，Triton forward 标 WIP 直接进 Week 4。

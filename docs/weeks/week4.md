# Week 4 — FlashAttention-2 Backward + Benchmark

**截止**：2026-05-17

**一句话目标**：把 FA2 的反向写完；写 benchmark 跑速度对比（vanilla SDPA vs PyTorch FA vs Triton FA，forward + forward+backward）。

---

## 要实现的 stubs

### `tt/kernels/__init__.py`

| 符号 | 作用 | PDF 参考 |
|---|---|---|
| `FlashAttentionPyTorch.backward` | 分块版 Attention 的 PyTorch backward（dQ / dK / dV） | §3.3（FA2 backward 推导） |
| `FlashAttentionTriton.backward` | Triton kernel 版 backward；Q 一个 kernel、KV 一个 kernel 常见 | §3.3 |

### 新增 `scripts/benchmark_attention.py`（自写）

PDF §3.4 要求一个 benchmark 脚本：对不同 `(seq_len, d_head, batch)` 配置，跑 3 个实现的 forward+backward 时间，输出 CSV / 表格。你自己决定格式。

---

## 跑绿

```bash
uv run pytest tests/systems/test_attention.py::test_flash_backward_pytorch -v
uv run pytest tests/systems/test_attention.py::test_flash_backward_triton -v
```

测试会构造 Q/K/V，跑 `apply()` + `.sum().backward()`，对拍梯度 vs `torch.nn.functional.scaled_dot_product_attention` + autograd。

### 测试 → adapter → 你的 stub

| 测试 | adapter | stub |
|---|---|---|
| `test_flash_backward_pytorch` | `get_flashattention_autograd_function_pytorch` | `tt.kernels.FlashAttentionPyTorch.backward` |
| `test_flash_backward_triton[*]` | `get_flashattention_autograd_function_triton` | `tt.kernels.FlashAttentionTriton.backward` |

---

## 参考材料

- **PDF**：`docs/cs336/assignment2_systems.pdf` §3.3（backward）+ §3.4（benchmark 规范）
- **CHANGELOG**：`docs/cs336/assignment2_CHANGELOG.md` —— 注意 FA2 的一些数值细节在不同版本里有调整（尤其 log-sum-exp 的存储）
- **回望 Week 3**：forward 里你保存到 ctx 的张量就是 backward 要用的（通常是 Q, K, V, O, L 其中的 L 是行方向 log-sum-exp）

---

## 验证

```bash
# 本周
uv run pytest tests/systems/test_attention.py -v

# 你自写的 benchmark 脚本（示例入口，shape 随你定）
uv run python scripts/benchmark_attention.py \
  --seq-len 1024 --d-head 64 --batch 8 --iters 50

# 回归
uv run pytest tests/basics -q
uv run pytest tests/moe/test_moe.py::test_simple_moe -v
```

---

## Done 的定义

- `tests/systems/test_attention.py` 全绿（forward + backward × pytorch/triton）。
- `scripts/benchmark_attention.py` 能跑，产出至少一个速度对比表（PDF §3.4 附表风格）。
- 如果 Triton backward 止损，交 PyTorch backward + benchmark 即可，Triton backward 标 WIP。

---

## 小贴士

- **FA2 backward 的关键不变量**：dQ 只依赖 row，dK/dV 只依赖 col —— 所以常用"两次扫"的 kernel 结构（一个 kernel 扫 Q，一个扫 KV）。
- **log-sum-exp 存储**：forward 里保存 `L = logsumexp(scores, dim=-1)`，backward 要用它算 `P = exp(scores - L)`。别保存整个 (Q, K) 级别的 attention 矩阵，那是 O(N²) 的空间。
- **benchmark 踩坑**：`torch.cuda.synchronize()` 包住测量；别忘了 warmup；不同 shape 下 SDPA 可能走的 backend 不同（Math/Mem-Efficient/Flash），直接对比有迷惑性 —— 可以用 `torch.nn.attention.sdpa_kernel` 锁定。
- **止损**：Week 4 如果反向搞不定，`tests/systems/test_attention.py::test_flash_backward_*` 标 `xfail`，进 Week 5 不回头。FA2 backward 自己手算一遍也要 2 天（正常节奏）。

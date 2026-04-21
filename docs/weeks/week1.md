# Week 1 — RMSNorm + RoPE + Attention + GQA

**截止**：2026-04-26

**一句话目标**：写出 LLaMA-style Transformer 的前四个核心层（含 GQA 决策点），让对应的单元测试绿。

---

## 要实现的 stubs

编辑 `tt/nn/__init__.py`，把下面的 `NotImplementedError` 换成你的实现。

| 符号                                   | 作用                                                                                     | PDF 参考                                 |
| -------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------- |
| `class Linear`                         | 无 bias 的 `y = xW^T`（下游所有层的基础）                                                | `docs/cs336/assignment1_basics.pdf` §3.1 |
| `class Embedding`                      | token id → embedding 查表                                                                | §3.1                                     |
| `class RMSNorm`                        | 权重 × (x / RMS(x))，`.weight` 可从外部赋值                                              | §3.2                                     |
| `class RoPE`                           | 旋转位置编码；输入 `(..., seq, d_k)` + `token_positions`                                 | §3.3                                     |
| `def scaled_dot_product_attention`     | Q/K/V + 可选 mask，返回 `(..., queries, d_v)`                                            | §3.4                                     |
| `class MultiHeadSelfAttention`         | batched 单次 QKV 投影 + 多头 SDPA + 输出投影（**GQA 决策点在此**：KV 头数可少于 Q 头数） | §3.5                                     |
| `class MultiHeadSelfAttentionWithRoPE` | 上面 + 在 Q/K 上套 RoPE                                                                  | §3.5                                     |
| `def softmax`                          | 数值稳定 softmax（减 max）                                                               | §3.4                                     |

---

## 跑绿

```bash
# 本周主战场
uv run pytest tests/basics/test_nn_utils.py::test_softmax_matches_pytorch -v
uv run pytest tests/basics/test_model.py -k "linear or embedding or rmsnorm or rope or attention or scaled_dot_product" -v

# 全体 basics 模型测试（Transformer Block/LM 要到 Week 2 才能过）
uv run pytest tests/basics/test_model.py -v
```

### 测试 → adapter → 你的 stub

| 测试                                                     | 调 adapter                               | 路由到的 stub                          |
| -------------------------------------------------------- | ---------------------------------------- | -------------------------------------- |
| `test_model.py::test_linear`                             | `run_linear`                             | `tt.nn.Linear`                         |
| `test_model.py::test_embedding`                          | `run_embedding`                          | `tt.nn.Embedding`                      |
| `test_model.py::test_rmsnorm`                            | `run_rmsnorm`                            | `tt.nn.RMSNorm`                        |
| `test_model.py::test_rope`                               | `run_rope`                               | `tt.nn.RoPE`                           |
| `test_model.py::test_scaled_dot_product_attention`       | `run_scaled_dot_product_attention`       | `tt.nn.scaled_dot_product_attention`   |
| `test_model.py::test_multihead_self_attention`           | `run_multihead_self_attention`           | `tt.nn.MultiHeadSelfAttention`         |
| `test_model.py::test_multihead_self_attention_with_rope` | `run_multihead_self_attention_with_rope` | `tt.nn.MultiHeadSelfAttentionWithRoPE` |
| `test_nn_utils.py::test_softmax_*`                       | `run_softmax`                            | `tt.nn.softmax`                        |

---

## 参考材料

- `docs/cs336/assignment1_basics.pdf` — 全规范（§3 是本周核心章节）
- `docs/cs336/assignment1_README.md` — 概览
- `docs/cs336/assignment1_CHANGELOG.md` — 版本演进（可跳过）

---

## 验证

```bash
# 本周任务完整验证
uv run pytest tests/basics/test_nn_utils.py::test_softmax_matches_pytorch \
              tests/basics/test_model.py::test_linear \
              tests/basics/test_model.py::test_embedding \
              tests/basics/test_model.py::test_rmsnorm \
              tests/basics/test_model.py::test_rope \
              tests/basics/test_model.py::test_scaled_dot_product_attention \
              tests/basics/test_model.py::test_multihead_self_attention \
              tests/basics/test_model.py::test_multihead_self_attention_with_rope \
              -v

# 回归：Week 0 的 MoE simple 不能破
uv run pytest tests/moe/test_moe.py::test_simple_moe -v
```

---

## Done 的定义

- 上述 8 个 `test_model.py` / `test_nn_utils.py` test 全绿。
- `tests/moe/test_moe.py::test_simple_moe` 依旧 pass（没搞坏下游）。
- `uv run pytest --collect-only -q | tail -1` 仍显示 72 tests。

---

## 小贴士（边界内的）

- **adapter 的 `_set_weight`**：它同时支持 `nn.Parameter` 和纯 Tensor 属性。你选 `nn.Module` 还是 plain class 都行。
- **GQA 的接口决策**：`run_multihead_self_attention` 的签名只传 `num_heads`，不区分 KV 头数。GQA 做不做、参数几名由你在 `__init__` 里决定；adapter 只管 forward 对拍。
- **RMSNorm 权重**：adapter 会执行 `layer.weight = weights`，你的实现需要能接受这样的赋值（`nn.Parameter` 要用 `.data.copy_()` 包住——adapter 已经帮你做了）。

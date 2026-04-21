# Week 2 — SwiGLU + Block + TransformerLM + Loss + AdamW

**截止**：2026-05-03

**一句话目标**：拼完 Transformer LM，加上训练环的优化器、loss、checkpoint、数据批次。

---

## 要实现的 stubs

### `tt/nn/__init__.py`

| 符号 | 作用 | PDF 参考 |
|---|---|---|
| `class SwiGLU` | `W2(SiLU(W1 x) * W3 x)` FFN | §3.6 |
| `class TransformerBlock` | 两次 pre-norm + MHA(+RoPE) + SwiGLU；`load_weights(dict)` 接收 assignment 指定的 key set | §3.7 |
| `def silu` | SiLU 激活 | §3.6 |

### `tt/models/__init__.py`

| `class TransformerLM` | embedding → N 个 block → final RMSNorm → lm_head；`load_weights(dict)` 接收 `token_embeddings.weight` / `layers.{i}.*` / `ln_final.weight` / `lm_head.weight` | §3.8 |

### `tt/optim/__init__.py`

| `get_adamw_cls` | 返回一个 `torch.optim.Optimizer` 子类（类本身，不是实例） | §4.1 |
| `cross_entropy` | batched CE loss，输入 logits + target ids | §4.2 |
| `gradient_clipping` | in-place 按 L2 norm 截断梯度 | §4.3 |
| `get_lr_cosine_schedule` | linear warmup + cosine decay，返回 float | §4.4 |

### `tt/utils/__init__.py`

| `get_batch` | 从 1D token array 里随机采样 `(batch, context)` 对 (inputs, labels) | §4.5 |
| `save_checkpoint` | 把 model / optimizer / iteration 落盘 | §4.6 |
| `load_checkpoint` | 还原 model / optimizer，返回 iteration | §4.6 |

---

## 跑绿

```bash
# 本周所有测试
uv run pytest tests/basics/test_model.py::test_swiglu \
              tests/basics/test_model.py::test_silu \
              tests/basics/test_model.py::test_transformer_block \
              tests/basics/test_model.py::test_transformer_lm \
              tests/basics/test_nn_utils.py::test_cross_entropy \
              tests/basics/test_nn_utils.py::test_cross_entropy_loss \
              tests/basics/test_nn_utils.py::test_gradient_clipping \
              tests/basics/test_optimizer.py \
              tests/basics/test_serialization.py \
              tests/basics/test_data.py \
              -v
```

### 测试 → adapter → 你的 stub

| 测试文件 | adapter | stub |
|---|---|---|
| `test_model.py::test_swiglu` | `run_swiglu` | `tt.nn.SwiGLU` |
| `test_model.py::test_silu` | `run_silu` | `tt.nn.silu` |
| `test_model.py::test_transformer_block` | `run_transformer_block` | `tt.nn.TransformerBlock` |
| `test_model.py::test_transformer_lm` | `run_transformer_lm` | `tt.models.TransformerLM` |
| `test_nn_utils.py::test_cross_entropy*` | `run_cross_entropy` | `tt.optim.cross_entropy` |
| `test_nn_utils.py::test_gradient_clipping` | `run_gradient_clipping` | `tt.optim.gradient_clipping` |
| `test_optimizer.py` | `get_adamw_cls` + `run_get_lr_cosine_schedule` | `tt.optim.{get_adamw_cls, get_lr_cosine_schedule}` |
| `test_serialization.py` | `run_save_checkpoint` + `run_load_checkpoint` + `get_adamw_cls` | `tt.utils.{save_checkpoint, load_checkpoint}` |
| `test_data.py` | `run_get_batch` | `tt.utils.get_batch` |

---

## 参考材料

- `docs/cs336/assignment1_basics.pdf` — §3.6–§3.8（模型剩余部分）、§4（训练环）
- **关于 `TransformerBlock.load_weights` 的 key set**：PDF §3.7 有，adapter 的 `run_transformer_block` docstring 也列了（你能从 adapter 源码反查）
- **关于 `TransformerLM.load_weights` 的 key set**：PDF §3.8，同上

---

## 验证

```bash
# Week 2 全绿
uv run pytest tests/basics/test_model.py tests/basics/test_optimizer.py \
              tests/basics/test_serialization.py tests/basics/test_data.py \
              tests/basics/test_nn_utils.py -v

# 回归
uv run pytest tests/moe/test_moe.py::test_simple_moe \
              tests/basics/test_model.py::test_linear \
              tests/basics/test_model.py::test_rmsnorm -v  # Week 1 样本
```

---

## Done 的定义

- `tests/basics/` 除 `test_tokenizer.py` / `test_train_bpe.py` 外全绿（tokenizer/BPE 是 Week 1 可选副本，可滚到 Week 2 末尾或跳过不影响下游）。
- `uv run pytest tests/basics -q` 通过率 ≥ 95%。
- Week 1 的回归测试仍绿。

---

## 小贴士

- **TransformerLM weights 的 key prefix**：adapter docstring 里有完整列表。不要让 `load_weights` 硬依赖 `torch.nn.Module.load_state_dict`——用 dict + 手动赋值也行，更灵活。
- **optimizer 工厂**：`get_adamw_cls()` 返回的是"类"，不是实例。测试里会 `cls(params, lr=..., betas=..., weight_decay=...)` 构造。
- **BPE / Tokenizer（可选）**：`test_tokenizer.py` 和 `test_train_bpe.py` 属于 Week 1 语义但时间上常滑到 Week 2——需要 `psutil` + `tiktoken`（基础 env 已带）。顶多当收尾，别挡路。

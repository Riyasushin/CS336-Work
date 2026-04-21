# Week 6 — FSDP + (stretch) Tensor Parallelism

**截止**：2026-05-31

**一句话目标**：把 CS336 的 FSDP 做完；有余力再做 PA3 的 ShardedLinear（Column-parallel）提前预热 Week 7 MoE。

> **提示**：你最初的 8 周计划是"Column + Row TP"。CS336 Assignment 2 里**没有**直接的 TP 测试 —— 它测的是 FSDP（参数 + 梯度 + optimizer 全分片 = ZeRO-3），功能上跟 TP 有交集但不是一回事。下面的主线按 CS336 走 FSDP，TP 作为 stretch（PA3 ShardedLinear 提前做一部分）。

---

## 要实现的 stubs

### 主线：`tt/parallel/containers.py`

| 符号 | 作用 | PDF 参考 |
|---|---|---|
| `class FSDP(nn.Module)` | 参数全 shard；forward 前 all-gather，forward 后丢弃；backward reduce-scatter | `docs/cs336/assignment2_systems.pdf` §5 |
| `FSDP.finish_gradient_synchronization()` | backward 末尾 reduce-scatter handle `.wait()` | §5 |
| `FSDP.gather_full_params()` | 返回完整参数 state_dict（调试 / checkpoint 用） | §5 |
| FSDP 的 `compute_dtype` 参数 | 通信时降到 `compute_dtype`（通常 bf16/fp16），master weights 保持 fp32 | §5.2 |

### Stretch：`tt/moe/layers.py`

| 符号 | 作用 | 参考 |
|---|---|---|
| `ShardedLinear.__call__` | column-parallel linear：本 rank 算 partial，all-reduce 聚合 | `docs/cse234/pa3_README.md` §1.1，`tt/moe/layers.py` 里的 TODO 注释 |

做了这个，Week 7 MoE_TP 就只剩"拼装"了。

---

## 跑绿

```bash
# 主线
uv run pytest tests/systems/test_fsdp.py -v

# Stretch（做了 ShardedLinear 之后）
STRICT_MOE=1 uv run pytest tests/moe/test_moe.py -v  # MoE_TP 部分可能已经能对拍 SimpleMoE
```

### 测试 → adapter → 你的 stub

| 测试 | adapter | stub |
|---|---|---|
| `test_fsdp.py::test_fsdp_correctness[fp32]` | `get_fsdp`, `fsdp_on_after_backward` | `tt.parallel.FSDP` + `finish_gradient_synchronization` |
| `test_fsdp.py::test_fsdp_correctness[fp16]` | 同上（`compute_dtype=torch.float16`） | 同上 |
| `test_fsdp.py::test_fsdp_gradient_sync[fp32/fp16]` | 同上 | 同上 |

`test_fsdp.py` 的 ToyFSDPModel 用了 `tt.nn.{Linear, Embedding, RMSNorm}`（我已帮你把 import 从 `cs336_basics.model` 改到 `tt.nn`）。**所以 Week 1/2 的 `tt.nn` 必须先完成**。

---

## 参考材料

- **PDF**：`docs/cs336/assignment2_systems.pdf` §5（FSDP）
- **对照 Week 5 的 ZeRO-1**：FSDP 相当于在 DDP + ZeRO-1 基础上再把参数本身也分片；注意 forward 时的 unshard / reshard 时机
- **stretch 参考**：`docs/cse234/pa3_README.md` §1.1；你之前写的 ShardedLinear stub 里有两种策略提示（all_gather 拼 vs 零填充 + all_reduce）

---

## 验证

```bash
# 本周主线
uv run pytest tests/systems/test_fsdp.py -v

# 如果做了 stretch
uv run pytest tests/moe -v  # MoE_TP 至少 shape test 过

# 回归
uv run pytest tests/basics -q
uv run pytest tests/systems/test_ddp.py tests/systems/test_sharded_optimizer.py -q
```

---

## Done 的定义

- **主线达标**：`test_fsdp.py` 全绿（4 tests：fp32/fp16 × correctness/gradient_sync）。
- **Stretch 达标（可选）**：`STRICT_MOE=1` 下 MoE_TP 的 2 个 world_size 测试绿，说明 ShardedLinear 数值正确。
- Week 1–5 全回归。

---

## 小贴士

- **混精读写**：`test_fsdp_correctness[fp16]` 要求 forward/backward 用 fp16 communicate + compute，optimizer step 用 fp32 master weights。`test_fsdp.py::_apply_mixed_precision_hooks` 有对照组的 hook 实现 —— **那是 reference，不是你的 FSDP 代码**，读它理解预期行为即可。
- **unshard 时机**：forward 是 layer-by-layer all-gather；backward 里也要在使用前 unshard（否则拿不到完整权重算梯度）。
- **reshard 时机**：用完立刻 reshard，否则 FSDP 比 DDP 还耗显存。
- **tied weights**：`ToyModelWithTiedWeights` 的共享参数只能 shard 一次、gather 一次 —— 跟 Week 5 的经验复用。
- **边界**：FSDP 的 forward/backward 要和 nn.Module 的 hook 体系契合。`register_forward_pre_hook` / `register_full_backward_hook` 是朋友；如果绕太大圈也可以考虑直接包一层 `nn.Module` 改写 forward。

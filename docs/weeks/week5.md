# Week 5 — Bucketed DDP + ZeRO-1

**截止**：2026-05-24

**一句话目标**：分布式数据并行 + 优化器状态分片 —— 单机多卡（或 CPU gloo）能跑起来训练的最小集。

---

## 要实现的 stubs

### `tt/parallel/containers.py`

| 符号 | 作用 | PDF 参考 |
|---|---|---|
| `class DDP(nn.Module)` | 包装 `torch.nn.Module`；在 `.forward` 广播参数，在 backward 过程中 bucketed async all-reduce 梯度 | `docs/cs336/assignment2_systems.pdf` §4.1–§4.2 |
| `DDP.finish_gradient_synchronization()` | 所有 bucket 的 all-reduce handle `.wait()`，然后平均梯度 | §4.2 |
| `class ShardedOptimizer(torch.optim.Optimizer)` | ZeRO-Stage-1：每个 rank 只持有一段 optimizer state，local step 后 all-gather 参数回来 | §4.3 |

FSDP 留给 Week 6，**本周别动** `tt/parallel/FSDP`。

---

## 跑绿

```bash
uv run pytest tests/systems/test_ddp.py -v           # 2 tests: ToyModel / ToyModelWithTiedWeights
uv run pytest tests/systems/test_sharded_optimizer.py -v  # 2 tests: 同上两个模型
```

### 测试 → adapter → 你的 stub

| 测试 | adapter | stub |
|---|---|---|
| `test_ddp.py::test_DistributedDataParallel[*]` | `get_ddp` + `ddp_on_after_backward` | `tt.parallel.DDP` + `DDP.finish_gradient_synchronization` |
| `test_sharded_optimizer.py::test_sharded_optimizer[*]` | `get_sharded_optimizer` | `tt.parallel.ShardedOptimizer` |

测试通过 `torch.multiprocessing.spawn` 起 2 个进程（`common.py::_setup_process_group`），gloo backend 走 CPU 就够，不一定要 GPU。

---

## 参考材料

- **PDF**：`docs/cs336/assignment2_systems.pdf` §4（数据并行章）
- **CSE234 PA2**：`docs/cse234/pa2_README.md` + `notebooks/pa2_matmul_triton.ipynb` 周边 —— PA2 也讲 DDP（MPI 版本），对拆解 bucket 和 communication scheduling 有帮助
- **ToyModel 定义**：`tests/systems/common.py` 里 —— 读它能知道测试会给你什么模型，包括带 `tied weights` 的变体（有共享 `fc2.weight`/`fc4.weight`；DDP 不能对共享参数重复 all-reduce）
- **分布式 harness**：`tests/_spawn.py::run_distributed`（其他周用它；CS336 的测试用自己的 `_setup_process_group`，等效）

---

## 验证

```bash
# 本周
uv run pytest tests/systems/test_ddp.py tests/systems/test_sharded_optimizer.py -v

# 回归
uv run pytest tests/basics -q
uv run pytest tests/systems/test_attention.py -q  # Week 3/4
uv run pytest tests/moe/test_moe.py::test_simple_moe -v
```

---

## Done 的定义

- `test_ddp.py` 2/2 绿；`test_sharded_optimizer.py` 2/2 绿。
- ToyModelWithTiedWeights 路径能正确处理共享参数（不二次通信；梯度最终一致）。
- Week 1–4 全回归通过。

---

## 小贴士

- **bucketed all-reduce**：别一个参数一个 all-reduce（太碎），也别整网络一次 all-reduce（backward 没法 overlap）。bucket 大小常见 25MB 上下；`torch.distributed.all_reduce(..., async_op=True)` 拿 handle，backward 结束后统一 `.wait()`。
- **tied weights 处理**：对每个参数 id 只注册一次 hook。测试专门测这个。
- **ZeRO-1 vs ZeRO-2/3**：
  - Stage 1：分片 optimizer state（momentum、variance）。参数和梯度仍复制。
  - Stage 2：加上梯度分片。
  - Stage 3：加上参数分片（= FSDP）。
  
  本周只做 Stage 1。all-gather 在 `step()` 之后把参数对齐到全 rank。
- **在 `tt.parallel.comm` 里你已经有**：`all_reduce`, `all_gather`, `all_to_all_single`, `broadcast` —— 用这些就够，不用直接 `torch.distributed`。

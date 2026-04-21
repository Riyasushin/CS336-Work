# Week 7 — MoE + Speculative Decoding

**截止**：2026-06-07

**一句话目标**：把 PA3 Part 1 的三个 MoE TODO 做完（TP / EP），再把 Part 3 的 speculative decoding 落地成 `tt.decoding.speculative_decode`。

---

## 要实现的 stubs

### MoE（`tt/moe/`）

| 符号 | 作用 | 参考 |
|---|---|---|
| `ShardedLinear.__call__` (`layers.py`) | column-parallel linear 的 forward | `docs/cse234/pa3_README.md` §1.1；文件内已有策略 A/B 注释 |
| `MoE_TP.forward` (`models.py`) | 拼装：router → ShardedExpert 逐个算 → gated 求和 | §1.1 结尾 |
| `MoE_EP.forward` (`models.py`) | all-to-all dispatch → 本 rank expert → all-to-all combine | §1.2（**PA3 最难的一段**） |

每个 stub 文件里已有 hint 注释（PA3 允许协作，算法提示是 PA3 原本 README 级别的指引）。

### Speculative Decoding（`tt/decoding/__init__.py`）

| `speculative_decode(target_model, draft_model, prompt_ids, max_new_tokens, num_draft_tokens=4) -> Tensor` | draft 采样 k 个 token → target 一次 forward 并行校验 → accept/reject → 生成一个新 token → 继续 | PA3 `notebooks/pa3_speculative_decoding.ipynb`，`docs/cse234/pa3_README.md` §3 |

---

## 跑绿

```bash
uv run pytest tests/moe/test_moe.py -v
```

现状（你还没实现时）：
- `test_simple_moe` 绿（SimpleMoE 是参考实现）
- `test_tp_moe_world_size_{2,4}` 和 `test_ep_moe_world_size_{2,4}` 四个都报 `all-zero tensor`，告诉你对应的 TODO 还没填

### MoE 测试 → 对应 stub

| 测试 | 需要的 stub | 断言 |
|---|---|---|
| `test_simple_moe` | 无（已实现） | shape / finite / 非零 |
| `test_tp_moe_world_size_{2,4}` | `ShardedLinear.__call__` + `MoE_TP.forward` | shape / finite / 非零 / 各 rank 输出一致 |
| `test_ep_moe_world_size_{2,4}` | `MoE_EP.forward`（`tt.parallel.comm.all_to_all_single` 已备） | shape / finite / 非零 / 各 rank 输出一致 |

"各 rank 输出一致"的断言用 `dist.all_reduce(out) == world_size * out` 检查——TP 和 EP 都应该让每个 rank 拿到相同的完整输出（router 复制、experts 通过 all_reduce/all_to_all 合并结果）。

### Speculative decoding

PA3 Part 3 是 notebook 形式的验证（没有 pytest 测试）。在 `notebooks/pa3_speculative_decoding.ipynb` 里替换"draft"和"target"函数为你的 `tt.decoding.speculative_decode`，核对加速比与 draft 接受率（PA3 目标：>1× 加速、>75% 接受率；bonus：>1.7×、>85%）。

---

## 参考材料

- **CSE234 PA3**：
  - `docs/cse234/pa3_README.md` —— 主文档（Part 1 MoE、Part 3 speculative）
  - `docs/cse234/pa3_part1_analysis.md` —— 空文件，你做完 MoE 后写性能分析填这里
  - `tt/moe/layers.py` / `tt/moe/models.py` 里的 TODO 注释 —— 策略提示
  - `notebooks/pa3_speculative_decoding.ipynb` —— 投机解码的参考 notebook，你可以直接改里面的实现，或在 `tt/decoding/` 里写完后回来接入
- **配套 PA3 Part 2（已不在本周，但相关）**：`docs/cse234/pa3_part2_moe.md` 和 `scripts/pa3_model_training_cost_analysis.py` —— MoE 成本分析，对理解"为什么做 MoE"有帮助但不 block

---

## 验证

```bash
# MoE 全绿
uv run pytest tests/moe/test_moe.py -v

# Speculative decoding：notebook 里接入 tt.decoding.speculative_decode，跑一遍并核对
uv run jupyter notebook notebooks/pa3_speculative_decoding.ipynb

# 回归
uv run pytest tests/basics -q
uv run pytest tests/systems -q
```

---

## Done 的定义

- `uv run pytest tests/moe/test_moe.py` 全绿（5 tests；simple + TP×2 + EP×2）。
- `docs/cse234/pa3_part1_analysis.md` 写一段短分析（TP vs EP 速度 + 为什么差）。
- `tt.decoding.speculative_decode` 在 notebook 里跑通、接受率记录 ≥ 75%（PA3 基础目标）。

---

## 小贴士

### MoE
- **TP 先于 EP**：ShardedLinear 对了，MoE_TP 就差一个循环；MoE_EP 的 all-to-all packing 是真正耗时间的。按这个顺序做，每步都有绿灯。
- **`all_to_all` packing/unpacking**：`indices.flatten()` 拿到每个 token 的目标 expert，`torch.argsort(indices.flatten())` 给 permutation，`torch.bincount(..., minlength=world_size)` 给 split sizes。combine 是 dispatch 的逆过程。
- **为什么没和 SimpleMoE 数值对拍**：`ShardedLinear` 和 `Linear` 的 RNG 消耗不同（前者 `(in, out/ws)` 权重 + `(out/ws)` 随机 bias；后者 `(in, out)` 权重 + 零 bias），PA3 设计里它们产生不同的权重，所以 TP 和 SimpleMoE 的 output 本来就不相等。测试改用"各 rank 输出一致 + 非零 + 有限"三条组合，已经能把"未实现 / 实现了但各 rank 不一致"这两大类错抓出来；绝对数值正确性留给你自己在 `docs/cse234/pa3_part1_analysis.md` 里用 toy 对照分析。

### Speculative Decoding
- **draft/target 的 KV cache**：要让两个 model 共享 prompt 的 KV cache 前缀；draft 只在 prompt+accepted 之后延伸 k 步。
- **验证步**：target 一次前向看所有 k+1 个位置的 logits；逐位置比较 `p_target(token) / p_draft(token)`，按 rejection sampling 决定 accept 到哪个 token。
- **接受率 vs 加速**：draft 越小越快但接受率低；draft = target 时接受率 100% 但没有加速。甜蜜点通常是 draft 比 target 小 10×+。

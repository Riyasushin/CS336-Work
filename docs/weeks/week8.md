# Week 8 — SFT + GRPO + Repo 整合

**截止**：2026-06-14

**一句话目标**：实现 alignment 的核心——tokenize / log-prob / mask / SFT step / GRPO step / DPO loss；repo 整理收尾。

---

## 要实现的 stubs（全在 `tt/rl/__init__.py`）

### 基础：tokenize + log-prob + entropy + mask utils

| 符号 | 参考 |
|---|---|
| `tokenize_prompt_and_output` | `docs/cs336/assignment5_alignment.pdf` §2.1 |
| `compute_entropy` | §2.2 |
| `get_response_log_probs` | §2.2 |
| `masked_mean` | §2.3 |
| `masked_normalize` | §2.3 |

### GRPO

| 符号 | 参考 |
|---|---|
| `compute_group_normalized_rewards` | §3.2（group advantage + normalize by std） |
| `compute_naive_policy_gradient_loss` | §3.3 |
| `compute_grpo_clip_loss` | §3.4（`loss_type="grpo_clip"`） |
| `compute_policy_gradient_loss` | 分派器，按 `loss_type` 转发 |
| `grpo_microbatch_train_step` | §3.5（梯度累积 + 反向） |

### SFT + 其它

| 符号 | 参考 |
|---|---|
| `sft_microbatch_train_step` | §2.4 |
| `get_packed_sft_dataset` / `iterate_batches` | §2.5 |
| `parse_mmlu_response` / `parse_gsm8k_response` | §4.1（eval parse utils） |

### 可选（RLHF 补充包）

| 符号 | 参考 |
|---|---|
| `compute_per_instance_dpo_loss` | `docs/cs336/assignment5_supplement_safety_rlhf.pdf` |

---

## 环境准备

alignment 测试的依赖默认**不装**（`vllm` / `transformers` 较重）：

```bash
uv sync --extra alignment

# tests/alignment/conftest.py 的 model_id fixture 默认指向 Stanford 集群路径
# 外部环境要用 env var 覆盖：
export ALIGNMENT_MODEL_ID=Qwen/Qwen2.5-Math-1.5B   # 或本地 tests/alignment/fixtures/tiny-gpt2
```

`tiny-gpt2` 已随 fixtures 进仓（26MB），**适合本地快速迭代**：
```bash
export ALIGNMENT_MODEL_ID=$(pwd)/tests/alignment/fixtures/tiny-gpt2
```

---

## 跑绿

```bash
# 本周全集
uv run pytest tests/alignment -v

# 分区跑（依赖模型下载时更实际）
uv run pytest tests/alignment/test_sft.py -v
uv run pytest tests/alignment/test_grpo.py -v
uv run pytest tests/alignment/test_metrics.py -v
uv run pytest tests/alignment/test_data.py -v
uv run pytest tests/alignment/test_dpo.py -v
```

### 测试 → adapter → stub（一对一，adapter 纯代理）

| 测试 | adapter | stub |
|---|---|---|
| `test_sft.py::*tokenize*` | `run_tokenize_prompt_and_output` | `tt.rl.tokenize_prompt_and_output` |
| `test_sft.py::*log_probs*` | `run_get_response_log_probs` | `tt.rl.get_response_log_probs` |
| `test_sft.py::*entropy*` | `run_compute_entropy` | `tt.rl.compute_entropy` |
| `test_sft.py::*masked_normalize*` | `run_masked_normalize` | `tt.rl.masked_normalize` |
| `test_sft.py::*microbatch*` | `run_sft_microbatch_train_step` | `tt.rl.sft_microbatch_train_step` |
| `test_grpo.py::*group_normalized*` | `run_compute_group_normalized_rewards` | `tt.rl.compute_group_normalized_rewards` |
| `test_grpo.py::*naive_policy*` | `run_compute_naive_policy_gradient_loss` | `tt.rl.compute_naive_policy_gradient_loss` |
| `test_grpo.py::*grpo_clip*` | `run_compute_grpo_clip_loss` | `tt.rl.compute_grpo_clip_loss` |
| `test_grpo.py::*policy_gradient_loss` | `run_compute_policy_gradient_loss` | `tt.rl.compute_policy_gradient_loss` |
| `test_grpo.py::*microbatch*` | `run_grpo_microbatch_train_step` | `tt.rl.grpo_microbatch_train_step` |
| `test_grpo.py::*masked_mean*` | `run_masked_mean` | `tt.rl.masked_mean` |
| `test_metrics.py::*mmlu*` | `run_parse_mmlu_response` | `tt.rl.parse_mmlu_response` |
| `test_metrics.py::*gsm8k*` | `run_parse_gsm8k_response` | `tt.rl.parse_gsm8k_response` |
| `test_data.py::*packed*` / `*iterate*` | `get_packed_sft_dataset`, `run_iterate_batches` | `tt.rl.{get_packed_sft_dataset, iterate_batches}` |
| `test_dpo.py::*dpo*` | `run_compute_per_instance_dpo_loss` | `tt.rl.compute_per_instance_dpo_loss` |

---

## 参考材料

- **主 PDF**：`docs/cs336/assignment5_alignment.pdf` —— §2 SFT、§3 GRPO、§4 eval
- **补充（可选）**：`docs/cs336/assignment5_supplement_safety_rlhf.pdf`（DPO、safety）
- **评估脚本**：
  - `scripts/evaluate_safety.py` —— LLaMA 3-70B judge 判响应安全性
  - `scripts/alpaca_eval_vllm_llama3_3_70b_fn/configs.yaml` —— alpaca_eval 的 vLLM 配置
  （两者都含 Stanford 集群路径；本机跑需要改 `--model-name-or-path` 或 `model_name:`）
- **依赖冲突注意**：`vllm==0.7.2` + `flash-attn==2.7.4.post1` 钉 `torch==2.5.1`，和基础环境 `torch~=2.11` 冲突；我把它们从 `alignment` extra 里摘掉了。需要的话在单独的 venv 里装：
  ```bash
  uv venv .venv-alignment --python 3.12
  .venv-alignment/bin/pip install torch==2.5.1 vllm==0.7.2 flash-attn==2.7.4.post1 --no-build-isolation
  ```

---

## Repo 整合（Week 8 的第二件事）

除 tt.rl 之外，还要把整个 repo 打包成"能向别人展示的状态"：

- **README**：根目录目前没 README；写一个对外 README，列出 `tt/` 结构 + 怎么跑。
- **examples**：在 `examples/` 下放一个可跑的 mini 训练 loop（tiny TransformerLM + SGD-based AdamW + Wikitext）——跟 tests 互相印证。
- **CI**（可选）：一个 GitHub Actions workflow，`uv sync && uv run pytest --collect-only`，至少把 collection 卡进去。
- **CHANGELOG**：每周做了什么，一行一行列。

---

## 验证

```bash
# 全项目 smoke
uv run pytest --collect-only -q   # 72 -> 应该还是 72

# alignment 全体
uv sync --extra alignment
ALIGNMENT_MODEL_ID=$(pwd)/tests/alignment/fixtures/tiny-gpt2 \
  uv run pytest tests/alignment -v

# 前 7 周回归
uv run pytest tests/basics tests/systems tests/moe -q
```

---

## Done 的定义

- `tests/alignment/` 在 `ALIGNMENT_MODEL_ID=tiny-gpt2` 下全绿（DPO 可选）。
- 根目录有 README，把项目讲清楚。
- 前 7 周所有测试回归绿。
- 项目在 `git log` 上能看到 8 周的合理进展。

---

## 小贴士

- **tiny-gpt2 vs Qwen-1.5B**：tiny-gpt2 跑得快但数值弱；Qwen-Math-1.5B 数学能力强，GRPO 奖励信号更真实。开发阶段用 tiny-gpt2 反复 iterate，accept PR 前在 Qwen 上跑一遍。
- **reward_fn 的 dummy 实现**：`tests/alignment/conftest.py` 里有 `dummy_reward_fn` fixture，基于 SHA-256 返回 reward —— 稳定、可测但无真实 signal。GRPO 测试用的是它，只测你的数学对不对，不测模型真的 align。
- **`loss_type` 三种**：`no_baseline`、`reinforce_with_baseline`、`grpo_clip`。前两种是 heuristic，`grpo_clip` 是 PPO-style clip。你的 `compute_policy_gradient_loss` 就是这三种的分派器。
- **止损**：DPO（supplement）整个可选，不做不影响 alignment 主线测试绿。

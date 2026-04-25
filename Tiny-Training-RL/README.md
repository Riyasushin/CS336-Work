# tiny-training-rl

CS336 Spring 2025 Assignment 5（**主 PDF + supplement PDF 全部 24 个 Problem**）的完整实装：reasoning RL（SFT / EI / GRPO）+ chat alignment（instruction tuning / DPO）。

## 当前状态

- **31/31 test PASS**：14 GRPO 原语 + 10 SFT 原语 + 4 MMLU/GSM8K parser + 2 packed dataset + 1 DPO loss
- **24 个 Problem 全实装**：算法原语 + 训练脚本 + 评测脚本 + 概念答题 + debug 笔记
- **本机能跑的都跑了**：pytest, MMLU/GSM8K demo eval（Qwen2.5-Math-1.5B 64-256 题）
- **本机跑不动的都写了**：双卡训练（SFT/EI/GRPO/DPO/leaderboard）+ 70B annotator 评测（AlpacaEval/SST），完整 CLI 模板，搬集群直接跑

## 文档入口

→ **[`docs/README.md`](docs/README.md)** —— 21 个讲解文档的完整索引（按 PDF 章节顺序）

每个 PDF Problem 一个 doc，含：任务描述 + 实装位置 + 关键决策 + 实测数字 + debug 故事 + PDF deliverable 答题。

## 实装范围

### 主 PDF（reasoning RL）

| 章节 | 内容 | 状态 |
|---|---|---|
| §3 | zero-shot baseline | scripts/eval_zeroshot.py + 本地跑通 |
| §4.2 | 5 SFT 原语 + log_generations | src/sft.py + 10/10 test |
| §4.3 | SFT 训练循环 | scripts/train_sft.py |
| §5 | Expert Iteration | scripts/train_ei.py |
| §6 | Policy gradient primer | docs/6 primer_pg.md（推导笔记） |
| §7.2 | 6 GRPO 原语 + train loop | src/grpo.py + 14/14 test + scripts/train_grpo.py |
| §8 | 8 个 GRPO 实验 | train_grpo.py 4 个 §8 flag 全支持 + 1 概念答题 |
| §9 | 4h leaderboard | scripts/leaderboard.sh + filter_train_curriculum.py |

### supplement PDF（chat alignment）

| 章节 | 内容 | 状态 |
|---|---|---|
| §2.1 | MMLU eval | parse_mmlu_response (test) + scripts/eval_mmlu.py (跑通 demo) |
| §2.2 | GSM8K eval | parse_gsm8k_response (test) + scripts/eval_gsm8k_supp.py (跑通 demo) |
| §2.3 | AlpacaEval | scripts/eval_alpaca.py + staff annotator 拷过来 |
| §2.4 | SST | scripts/eval_sst.py + staff evaluate_safety.py |
| §3.1 | look_at_sft | docs 数据探索答题 |
| §3.2.1 | PackedSFTDataset + iterate_batches | src/data.py + 2/2 test |
| §3.2.2 | sft_script | scripts/train_sft_packed.py |
| §3.3 + §4 | SFT + 4 评测 | docs CLI 模板 |
| §4.5 | red_teaming | docs 答题 |
| §5.2 | look_at_hh | scripts/load_hh.py（HH 单轮过滤 + parse 单测通过） |
| §5.3 | dpo_loss | src/dpo.py + 1/1 test (loss=0.5785) |
| §5.4 | dpo_training | scripts/train_dpo.py |

## 跑测试

```bash
# 全 31 test（注意 SOCKS proxy 要 unset）
env -u all_proxy -u http_proxy -u https_proxy -u ALL_PROXY -u HTTPS_PROXY -u HTTP_PROXY \
    HF_ENDPOINT=https://hf-mirror.com \
    uv run python -m pytest tests/

# 离线 test（不需要 HF download，27/27）
uv run python -m pytest tests/test_sft.py tests/test_grpo.py tests/test_metrics.py
```

## 安装依赖

```bash
# 核心（必装）
uv sync --package tiny-training-rl

# 评测要 vllm（仅集群 CUDA 12.x；本机 CUDA 13 装不上）
uv sync --package tiny-training-rl --extra eval

# r1_zero_reward_fn 要 sympy / math-verify / latex2sympy 栈
uv sync --package tiny-training-rl --extra grader

# 训练要 wandb / typer
uv sync --package tiny-training-rl --extra train
```

## 跨 PDF 对比

| | 主 PDF | supplement PDF |
|---|---|---|
| 模型 | Qwen 2.5 Math 1.5B | Llama 3.1 8B Base |
| 数据 | MATH (本仓库用 GSM8K 替代) | UltraChat-200K-safety + Anthropic HH |
| Prompt | r1_zero (`<think>...</think> <answer>...</answer>`) | Alpaca + system prompt |
| 关键算法 | SFT → EI → GRPO | SFT (packed) → DPO |
| Eval | MATH val acc | MMLU + GSM8K + AlpacaEval (70B) + SST (70B) |

→ 两条路径在 `src/` 共用 6 个 SFT 原语 + Alpaca 模板 + sft_train 工具集。

## 已知限制

1. **vLLM**: 本机 CUDA 13.0 与 vLLM 编译版 12.8 mismatch，所有 vLLM 路径标记"集群跑"
2. **MATH 数据**: 因版权不可得；GSM8K 替代（PDF §3 明确允许 "Tip for Open-Source Auditors"）
3. **70B annotator**: AlpacaEval / SST 评测需要 Llama 3.3 70B Instruct (2×80GB)，本机不可跑
4. **train_sft.py / sft_train.py 重复**: 待统一重构（避免双份代码）

## 参考

- 主 PDF: `docs/cs336_spring2025_assignment5_alignment.pdf`（38 页）
- supplement PDF: `docs/cs336_spring2025_assignment5_supplement_safety_rlhf.pdf`（19 页）
- staff 仓库: https://github.com/stanford-cs336/assignment5-alignment

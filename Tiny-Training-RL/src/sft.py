"""SFT 监督微调原语（assignment5_alignment §4.2）。

PDF §4 概念回顾：SFT 让 base 模型学会 "<think>...</think> <answer>...</answer>"
模板下的解题轨迹，先解决 §3 暴露的 80% format=0 问题。每个原语单独承担
SFT loop 的一个步骤，便于 §5 (EI) 和 §7 (GRPO) 复用：

    数据预处理 ──► tokenize_prompt_and_output                # PDF Problem
        ↓
    前向        ──► get_response_log_probs（含 compute_entropy）
        ↓
    构造 loss  ──► masked_normalize（按 response_mask 求和归一）
        ↓
    backward    ──► sft_microbatch_train_step（含 grad accum 缩放）

设计动机（顶部说明，函数内不再重复）：
    - response_mask 在数据预处理就构造出来：训练时只对 response 段算 loss，
      prompt / padding 不参与；mask 形状 (B, T-1) 与 labels 对齐。
    - log-prob / entropy 一次前向同时取出（return_token_entropy=True）：
      避免重复跑 model；entropy 用来做 §7 GRPO 的 logging。
    - microbatch_train_step 内部直接 backward()：grad accum 时如果让外面
      backward，多个 microbatch 的计算图同时驻留显存（policy 1.5B + grad
      = 几 GB），必须立刻 backward 释放图。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase


# ============================================================================
# Problem (tokenize_prompt_and_output)：prompt + response 拼接分词 + mask
# ============================================================================
def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """对每对 (prompt, output) 分别分词后拼接，构造 input_ids/labels/response_mask。

    返回三个 (B, max_len-1) 的张量：
        - input_ids : 拼接序列去掉最后一个 token（作为模型输入）
        - labels    : 拼接序列去掉第一个 token（next-token 目标）
        - response_mask : 在 labels 上标记哪些位置属于 response（1）

    为什么要这样切：
        训练时一次前向得到所有位置的 next-token logits，然后用 labels 取每
        个位置目标 token 的 log-prob，再用 response_mask 屏蔽掉 prompt 段
        和 padding 段。三个张量的最后一维都是 max_len-1，索引一一对齐。

    padding 策略：
        prompt+output 拼接后长度不一，用 tokenizer.pad_token_id 右 padding 到
        batch 内最长。pad_token_id 缺失时回退到 eos。padding 位置在
        response_mask 上为 0，不参与 loss。
    """
    # 1) 分别分词，记录每条 prompt / output 的真实 token 数（不带特殊 token，
    #    因为 r1_zero / Alpaca 模板已经把所有结构写到字符串里了）
    prompt_ids = [tokenizer.encode(s, add_special_tokens=False) for s in prompt_strs]
    output_ids = [tokenizer.encode(s, add_special_tokens=False) for s in output_strs]
    full_ids = [p + o for p, o in zip(prompt_ids, output_ids)]
    full_lens = [len(x) for x in full_ids]
    max_len = max(full_lens)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    B = len(prompt_strs)
    full_padded = torch.full((B, max_len), pad_id, dtype=torch.long)
    # response_mask_full 在 max_len 全长度上记录 response 位置，先构造再 shift
    resp_mask_full = torch.zeros((B, max_len), dtype=torch.long)
    for i, (p, o) in enumerate(zip(prompt_ids, output_ids)):
        full_padded[i, : len(p) + len(o)] = torch.tensor(p + o, dtype=torch.long)
        # response 占据 [len(p), len(p)+len(o)) 这一段
        resp_mask_full[i, len(p) : len(p) + len(o)] = 1

    # 2) shift：input = [:, :-1]，label = [:, 1:]，mask = [:, 1:]
    #    mask 跟 labels 对齐 —— labels[t] 的 reward 由 mask[t] 决定参不参与 loss
    input_ids = full_padded[:, :-1].contiguous()
    labels = full_padded[:, 1:].contiguous()
    response_mask = resp_mask_full[:, 1:].contiguous()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


# ============================================================================
# Problem (compute_entropy)：每位置 next-token 分布的熵
# ============================================================================
def compute_entropy(logits: Tensor) -> Tensor:
    """对最后一维（vocab）算 H = -Σ p log p，数值稳定写法。

    H = log Z - <p, logits>
      = logsumexp(logits) - sum(softmax(logits) * logits)

    为什么要这样写：
        直接 -sum(softmax * log_softmax) 也对；但 logsumexp 形式让 vocab
        很大时不会先指数化大数（避免 overflow）。Qwen 的 vocab=151936，
        softmax(logits) 在 fp32 下也可能下溢；先 logsumexp 一下减出来更稳。

    bf16 的处理：
        若调用方传进来的 logits 是 bf16（HF 模型常见输出），先转 fp32 再算。
        bf16 的尾数只有 7 bit，logsumexp 这种大 V 加和会丢精度；HF 的
        F.cross_entropy 内部也是这么做（loss 在 fp32 上算）。
    """
    if logits.dtype in (torch.bfloat16, torch.float16):
        logits = logits.float()
    log_z = torch.logsumexp(logits, dim=-1)               # (B, T)
    probs = torch.softmax(logits, dim=-1)                  # (B, T, V)
    expected_logits = (probs * logits).sum(dim=-1)         # (B, T)
    return log_z - expected_logits


# ============================================================================
# Problem (get_response_log_probs)：一次前向取 log p(label_t | x_<t) (+ entropy)
# ============================================================================
def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """跑一次前向，gather 出每个位置 labels 对应 token 的 log-prob。

    返回：
        - "log_probs": (B, T) 张量，log p(labels[t] | input_ids[:t+1])
        - "token_entropy": 可选 (B, T)；下游 §7 GRPO 训练时记录熵看模型自信度

    实现要点：
        - model(input_ids).logits 形状 (B, T, V)
        - log_softmax 的稳定形式 = logits - logsumexp(logits, -1, keepdim=True)
        - gather labels：必须先 unsqueeze(-1) 让 labels 形状变成 (B, T, 1)
        - 这里**不**对 prompt / padding 做 mask；外层 train loop 用
          response_mask 过滤。这样 entropy 的统计可以在所有位置上取，loss
          的 mask 由调用方决定。

    bf16 的处理：
        模型在 bf16 下的 logits 拿来直接 log_softmax 会丢精度（vocab 大、
        概率小的位置 log p 抖动大）。统一转 fp32 再算 —— 这跟 HF
        F.cross_entropy 内部把 logits 升到 fp32 是同一个理由。返回 fp32 的
        log_probs / entropy，下游 numpy snapshot 也才能 .numpy() 通过
        （bf16 不能直接转 numpy）。
    """
    out = model(input_ids)
    logits: Tensor = out.logits
    if logits.dtype in (torch.bfloat16, torch.float16):
        logits = logits.float()
    log_probs_full = F.log_softmax(logits, dim=-1)         # (B, T, V)
    # gather 取 labels 位置
    log_probs = log_probs_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T)

    result: dict[str, Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


# ============================================================================
# Problem (masked_normalize)：sum(tensor*mask, dim) / normalize_constant
# ============================================================================
def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float = 1.0,
    dim: int | None = None,
) -> Tensor:
    """带 mask 求和后除以一个**外部给的常数**（不是 mask 元素个数）。

    sum_mask(x, m) = sum(x * m, dim=dim)
    masked_normalize = sum_mask(x, m) / normalize_constant

    与 masked_mean 的区别：分母不是 mask.sum(dim) 而是给定常数。Dr.GRPO 的
    长度归一化用这个形式，分母固定为 max_new_tokens，避免短序列被不成比例
    地放大（详见 docs/grpo_experiments.md 的 think_about_length_normalization）。

    dim=None 时对整个 tensor 求和。
    """
    masked = tensor * mask
    if dim is None:
        s = masked.sum()
    else:
        s = masked.sum(dim=dim)
    return s / normalize_constant


# ============================================================================
# Problem (sft_microbatch_train_step)：一个 microbatch 的 fwd + bwd
# ============================================================================
def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float | None = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """单个 microbatch 的 SFT 损失 + backward。

    SFT 损失定义：
        loss_per_example = -masked_normalize(log_probs, mask, dim=-1, /K)   # 沿序列累加
        loss = mean over batch of (loss_per_example) / gradient_accumulation_steps

    其中 K = normalize_constant，PDF §4.2 默认 1.0；放置 grad accum 系数
    在最外层除，使得 K 个 microbatch 的梯度之和 ≈ 单 batch 一次性算的梯度。

    返回：
        - loss: 已经做过 grad-accum 缩放的标量；用来 logging
        - metadata: 内部统计（这里返回 sum-mask 之类，便于训练循环算平均损失）
    """
    # 1) 沿序列方向用 masked_normalize 求和，shape -> (B,)
    per_example = masked_normalize(
        policy_log_probs, response_mask.float(), normalize_constant=normalize_constant or 1.0, dim=-1
    )
    # 2) 取 batch 平均后取负号（最大化 log-prob 等价于 minimize 负 log-prob）
    #    再除以 grad_accum 让多个 microbatch 累加后等价于一个大 batch 的平均
    loss = -per_example.mean() / gradient_accumulation_steps
    loss.backward()

    metadata: dict[str, Tensor] = {
        "sum_log_probs_masked": per_example.detach(),
    }
    return loss, metadata


# ============================================================================
# log_generations：训练循环里抽样若干 (prompt, generation) + 统计指标
# 用 utility（无 adapter test，PDF §4.2 末尾 Problem (log_generations) 1 point）
# ============================================================================
def log_generations(
    prompts: list[str],
    generations: list[str],
    ground_truths: list[str],
    rewards: list[dict[str, float]],
    token_entropies: list[float] | None = None,
) -> dict[str, Any]:
    """汇总训练 / 验证步骤里 in-the-loop 生成的统计量。

    PDF §4.2 (log_generations) 列了 6 项必须 log：
        1. prompt
        2. response（generation）
        3. ground truth
        4. reward 信息（format / answer / total）
        5. response 的平均 token 熵
        6. 平均长度（总体 / correct / incorrect）

    实现给一个纯 Python 字典而非 wandb 调用，让 train_loop 自己决定上报通道。
    """
    n = len(prompts)
    correct_lens, incorrect_lens, all_lens = [], [], []
    for gen, r in zip(generations, rewards):
        L = len(gen.split())  # 这里用空白切；正式调用方应传 token 长度
        all_lens.append(L)
        (correct_lens if r.get("answer_reward", 0) == 1.0 else incorrect_lens).append(L)

    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n": n,
        "samples": [
            {
                "prompt": p,
                "generation": g,
                "ground_truth": gt,
                "reward": r,
                "token_entropy_mean": (token_entropies[i] if token_entropies else None),
            }
            for i, (p, g, gt, r) in enumerate(zip(prompts, generations, ground_truths, rewards))
        ],
        "avg_response_len": _avg(all_lens),
        "avg_response_len_correct": _avg(correct_lens),
        "avg_response_len_incorrect": _avg(incorrect_lens),
        "avg_token_entropy": _avg(token_entropies) if token_entropies else None,
        "format_reward_mean": _avg([r.get("format_reward", 0.0) for r in rewards]),
        "answer_reward_mean": _avg([r.get("answer_reward", 0.0) for r in rewards]),
        "reward_mean": _avg([r.get("reward", 0.0) for r in rewards]),
    }

"""GRPO 原语（PDF §7.2，6 个 Problem）。

数学背景见 docs/primer_pg.md。本模块只承担"组装层 + 数学公式落地"，所有
原语对接 tests/test_grpo.py 的 12 个 snapshot test。

原语依赖关系（一图）
====================

    rollouts + ground_truths
            │
            ▼
    compute_group_normalized_rewards     ← Problem #1
        ├ for each group of G rollouts:
        ├   r^(i) = reward_fn(o^(i), gt^(i))
        ├   A^(i) = (r - mean) / (std + ε)        # 或减 mean 不除 std
        └ return (advantages, raw_rewards, metadata)
            │
            ▼
    policy_log_probs (B, T)
            │
            ▼
    compute_policy_gradient_loss         ← Problem #4 (dispatcher)
        ├─ "no_baseline"           → compute_naive_policy_gradient_loss(raw_rewards)   ← Problem #2
        ├─ "reinforce_with_baseline" → compute_naive_policy_gradient_loss(advantages)  ← Problem #2
        └─ "grpo_clip"             → compute_grpo_clip_loss(advantages, old, ε)        ← Problem #3
            │
            ▼
    per-token loss (B, T)
            │
            ▼
    grpo_microbatch_train_step           ← Problem #6
        ├ masked_mean(per_token_loss, response_mask, dim=-1)    ← Problem #5
        ├ mean over batch / gradient_accumulation_steps
        ├ loss.backward()                                        ← 让计算图当场释放
        └ return (loss_scalar, metadata)

数学要点回顾（详细推导见 primer_pg.md）
======================================

REINFORCE per-token loss：
    L_t = -A · log π_θ(a_t | s_t)
其中 A 是 advantage（在 verified-reward 设定下，A 是 raw reward 或 group-normalized reward）。

GRPO-Clip per-token loss（off-policy + clip 稳定）：
    r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)            # per-token IS ratio
    L_t = -min(r_t · A, clip(r_t, 1-ε, 1+ε) · A)
    A > 0：clip 上界封顶 1+ε —— 不让 policy 走太远
    A < 0：clip 下界封底 1-ε —— 不让 prob 压到 0

Group-normalized advantage（GRPO 的 critic-free baseline）：
    A^(i) = (r^(i) - mean(r^(1..G))) / (std(r^(1..G)) + advantage_eps)
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import Tensor


# =============================================================================
# Problem #1: compute_group_normalized_rewards
# =============================================================================
def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """逐 rollout 算 reward，按 group 归一化得 advantage。

    输入：
        reward_fn: 输入 (response, ground_truth)，返回 dict 含 "reward" key
                   （format_reward / answer_reward 也常见，但这里只用 reward）
        rollout_responses: list[str]，长度 = n_prompts × group_size
        repeated_ground_truths: list[str]，同长度，每条 gt 重复 group_size 次
        group_size: int，每题 G 个 rollout
        advantage_eps: 防 std=0 除零
        normalize_by_std: True → (r-mean)/(std+ε)；False → r-mean（Dr.GRPO 选项）

    返回：
        advantages: (rollout_batch_size,) 张量
        raw_rewards: (rollout_batch_size,) 张量
        metadata: {"reward_mean", "reward_std", "reward_max", "reward_min"}

    实现要点：
    - reshape 成 (n_prompts, G) 让组内统计量沿 dim=1 聚合
    - 用 unbiased=False 算 std —— 与 numpy 默认一致；snapshot 也是这个口径
    - normalize_by_std=False 是 Dr.GRPO (Liu et al 2025) 的发现：
      group std 归一化在低答对率题上把权重放大，可能引入 unwanted bias；
      不除 std 是 §8 grpo_group_standard_deviation 的对照组
    """
    assert len(rollout_responses) == len(repeated_ground_truths), \
        f"len mismatch: {len(rollout_responses)} vs {len(repeated_ground_truths)}"
    assert len(rollout_responses) % group_size == 0, \
        f"rollout_batch_size {len(rollout_responses)} not divisible by group_size {group_size}"

    n_prompts = len(rollout_responses) // group_size
    raw = torch.tensor(
        [reward_fn(r, gt)["reward"] for r, gt in zip(rollout_responses, repeated_ground_truths)],
        dtype=torch.float32,
    )  # (rollout_batch_size,)

    # reshape 成 (n_prompts, G) 让组内 mean/std 沿 dim=1 算
    grouped = raw.view(n_prompts, group_size)
    group_mean = grouped.mean(dim=1, keepdim=True)                  # (n_prompts, 1)
    if normalize_by_std:
        # unbiased=True：用 1/(G-1) Bessel correction。snapshot 是这个口径
        # （PyTorch tensor.std() 默认 unbiased=True，与 numpy std() 默认 ddof=0
        # 不同；snapshot 跟 PyTorch 默认走）
        group_std = grouped.std(dim=1, keepdim=True, unbiased=True)
        adv_grouped = (grouped - group_mean) / (group_std + advantage_eps)
    else:
        adv_grouped = grouped - group_mean
    advantages = adv_grouped.reshape(-1).contiguous()                # (rollout_batch_size,)

    metadata = {
        "reward_mean": raw.mean().item(),
        "reward_std": raw.std(unbiased=False).item(),
        "reward_max": raw.max().item(),
        "reward_min": raw.min().item(),
    }
    return advantages, raw, metadata


# =============================================================================
# Problem #5: masked_mean —— 先实现，后面 #6 要用
# =============================================================================
def masked_mean(tensor: Tensor, mask: Tensor, dim: int | None = None) -> Tensor:
    """`sum(x*m, dim) / sum(m, dim)`，对 dim=None 时所有 mask 元素取平均。

    与 sft.masked_normalize 区别：
        masked_mean      分母 = mask.sum(dim)        # 每条 example 等权重
        masked_normalize 分母 = 给定常数 K           # Dr.GRPO 的固定长度归一化

    详细对比见 docs/grpo_experiments.md::think_about_length_normalization。

    实现要点：
    - tensor 与 mask 形状要一致；调用方负责
    - mask 全 0 时返回 NaN（数学上 0/0 = NaN）。这是数学纯函数行为；snapshot
      也期望 NaN。训练循环要自己保证 response_mask 至少有一个 1（实践上
      tokenize_prompt_and_output 已经保证 response 段非空）。
    """
    masked = tensor * mask
    if dim is None:
        s = masked.sum()
        m = mask.sum()
    else:
        s = masked.sum(dim=dim)
        m = mask.sum(dim=dim)
    return s / m


# =============================================================================
# Problem #2: compute_naive_policy_gradient_loss
# =============================================================================
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
) -> Tensor:
    """每位置朴素 PG loss：L_t = -A · log π_θ(a_t | s_t)。

    Args:
        raw_rewards_or_advantages: (B, 1)，每条 rollout 一个标量
        policy_log_probs: (B, T)，每位置一个 log p

    Returns:
        (B, T) per-token loss

    Broadcast：(B,1) × (B,T) 自动沿 dim=1 复制 advantage —— 同一条 rollout
    内每个 token 共享 advantage（verified-reward 设定下，整条轨迹只有终止
    那步 reward，所以每个 token 上 advantage 相同）。
    """
    return -raw_rewards_or_advantages * policy_log_probs


# =============================================================================
# Problem #3: compute_grpo_clip_loss
# =============================================================================
def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """GRPO-Clip per-token loss + clip 元数据。

    L_t = -min(r_t · A, clip(r_t, 1-ε, 1+ε) · A)
        其中 r_t = exp(log π_θ(a_t|s_t) - log π_θ_old(a_t|s_t))

    Args:
        advantages: (B, 1)
        policy_log_probs: (B, T)
        old_log_probs: (B, T)
        cliprange: float ε（PDF 默认 0.2）

    Returns:
        loss: (B, T) per-token clipped loss
        metadata: {"clipped": (B, T) bool tensor —— True 表示该位置被 clip 救场}

    实现要点：
    - ratio = exp(policy - old)；用减法在 log 空间做避免 exp(large) 爆掉
    - clip(ratio, 1-ε, 1+ε)：torch.clamp 同款
    - min(unclipped, clipped) 取每位置较小那个 —— 相当于"surrogate 取 conservative
      估计"。对 A>0 取小 = 限制增长；对 A<0 取小（更负）= 限制下降。两情况一致
    - clipped indicator：当 unclipped > clipped（即 min 选了 clipped）说明被
      clip 救了；记录率给 wandb 监控 off-policy 程度
    """
    # log-space ratio 防 exp 爆掉
    log_ratio = policy_log_probs - old_log_probs        # (B, T)
    ratio = torch.exp(log_ratio)                         # (B, T)

    unclipped = ratio * advantages                       # (B, T) —— advantages 自动广播
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped = clipped_ratio * advantages                 # (B, T)

    # PDF 公式：surrogate = min(unclipped, clipped)，再取负号当 loss
    surrogate = torch.minimum(unclipped, clipped)
    loss = -surrogate                                    # (B, T)

    # clipped indicator：当 min 取了 clipped 那一项，说明 clip 在起作用
    # 实务上等价于 ratio 飘到 [1-ε, 1+ε] 区间外
    clipped_mask = (ratio < (1.0 - cliprange)) | (ratio > (1.0 + cliprange))
    metadata = {"clipped": clipped_mask}
    return loss, metadata


# =============================================================================
# Problem #4: compute_policy_gradient_loss —— dispatcher
# =============================================================================
def compute_policy_gradient_loss(
    policy_log_probs: Tensor,
    loss_type: str,
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """三种 loss_type 的 dispatcher。

    loss_type 选项（PDF §7.2）：
        - "no_baseline":            朴素 PG，A = raw_rewards（"R 直接当 advantage"）
        - "reinforce_with_baseline": 朴素 PG，A = group-normalized advantage
        - "grpo_clip":              GRPO-Clip with per-token IS

    返回：(per_token_loss (B,T), metadata dict)

    metadata 设计：
        - 三个分支都返回 dict，方便上层统一收集
        - "no_baseline" / "reinforce_with_baseline" 没特殊 metadata，返回空 dict
        - "grpo_clip" 返回 {"clipped": (B,T) bool}
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "no_baseline needs raw_rewards"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "reinforce_with_baseline needs advantages"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    elif loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None, \
            "grpo_clip needs advantages, old_log_probs, cliprange"
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    elif loss_type == "grpo_no_clip":
        # PDF §8 grpo_off_policy_clip_ablation：用 unclipped per-token surrogate
        # L_t = -ratio_t · A_t = -exp(log π - log π_old) · A
        # 与 grpo_clip 比较看 clip 的稳定作用
        assert advantages is not None and old_log_probs is not None, \
            "grpo_no_clip needs advantages and old_log_probs"
        log_ratio = policy_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        loss = -(ratio * advantages)
        # 仍记录 "would-be-clipped" 位置作为 metadata，看 ratio 飘到哪
        # （这里 clip 不生效但 ratio 飘的位置就是 grpo_clip 会救的那些）
        if cliprange is not None:
            clipped_mask = (ratio < (1.0 - cliprange)) | (ratio > (1.0 + cliprange))
            return loss, {"clipped": clipped_mask}
        return loss, {}
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")


# =============================================================================
# Problem #6: grpo_microbatch_train_step
# =============================================================================
def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    length_normalize_constant: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """单个 microbatch 的 GRPO 前向 + 反向。

    流程：
        1. compute_policy_gradient_loss → per-token loss (B, T)
        2. 沿 seq 聚合：
            - length_normalize_constant=None → masked_mean(dim=-1)（默认，每条 example 等权重）
            - length_normalize_constant=K     → masked_normalize(dim=-1, K)（Dr.GRPO 长度公平）
           这是 PDF §8 think_about_length_normalization 的开关。
        3. mean over batch → 标量；除 grad_accum
        4. loss.backward()
        5. 返回 (loss_scalar, metadata)
    """
    # 1) 算 per-token loss
    per_token_loss, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )

    # 2) 沿 seq 聚合
    response_mask_f = response_mask.float()
    if length_normalize_constant is None:
        # GRPO 主流：每条 example 等权重
        per_example_loss = masked_mean(per_token_loss, response_mask_f, dim=-1)        # (B,)
    else:
        # Dr.GRPO 长度公平：sum(loss·mask) / K，K 通常 = max_gen_len
        # 与 src/sft.py::masked_normalize 同语义；放这里避免循环依赖
        per_example_loss = (per_token_loss * response_mask_f).sum(dim=-1) / length_normalize_constant

    # 3) batch 平均 + grad_accum 缩放
    loss = per_example_loss.mean() / gradient_accumulation_steps

    # 4) 当场反传
    loss.backward()

    # 5) metadata：把 grpo_clip 的 clipped indicator 抽出来，供训练循环算 clip_fraction
    metadata: dict[str, Tensor] = {"per_example_loss": per_example_loss.detach()}
    if "clipped" in loss_metadata:
        # clip_fraction 只在 response 区域算（pad 区域 mask=0）
        clipped = loss_metadata["clipped"].float() * response_mask_f
        n_resp = response_mask_f.sum().clamp(min=1)
        metadata["clip_fraction"] = (clipped.sum() / n_resp).detach()

    return loss, metadata

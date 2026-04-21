"""Adapter layer for cs336_alignment tests (Week 8).

Routes run_*/get_* signatures to tt.rl. Heavy deps (transformers, vllm,
flash-attn) are only needed for the alignment path — install with:

    uv sync --extra alignment
"""

from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset


def run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer) -> dict[str, Tensor]:
    from tt.rl import tokenize_prompt_and_output
    return tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, dict[str, float]]:
    from tt.rl import compute_group_normalized_rewards
    return compute_group_normalized_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
        group_size,
        advantage_eps,
        normalize_by_std,
    )


def run_compute_entropy(logits: Tensor) -> Tensor:
    from tt.rl import compute_entropy
    return compute_entropy(logits)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool,
) -> dict[str, Tensor]:
    from tt.rl import get_response_log_probs
    return get_response_log_probs(model, input_ids, labels, return_token_entropy)


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor, policy_log_probs: Tensor
) -> Tensor:
    from tt.rl import compute_naive_policy_gradient_loss
    return compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs)


def run_compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    from tt.rl import compute_grpo_clip_loss
    return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)


def run_compute_policy_gradient_loss(
    policy_log_probs: Tensor,
    loss_type: str,
    raw_rewards: Tensor,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    from tt.rl import compute_policy_gradient_loss
    return compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )


def run_masked_mean(tensor: Tensor, mask: Tensor, dim: int | None = None) -> Tensor:
    from tt.rl import masked_mean
    return masked_mean(tensor, mask, dim)


def run_masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> Tensor:
    from tt.rl import masked_normalize
    return masked_normalize(tensor, mask, dim, normalize_constant)


def run_sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    from tt.rl import sft_microbatch_train_step
    return sft_microbatch_train_step(
        policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant
    )


def run_grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    from tt.rl import grpo_microbatch_train_step
    return grpo_microbatch_train_step(
        policy_log_probs,
        response_mask,
        gradient_accumulation_steps,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )


# --- optional RLHF / safety part ---

def get_packed_sft_dataset(
    tokenizer, dataset_path: str | os.PathLike, seq_length: int, shuffle: bool
) -> Dataset:
    from tt.rl import get_packed_sft_dataset
    return get_packed_sft_dataset(tokenizer, dataset_path, seq_length, shuffle)


def run_iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool):
    from tt.rl import iterate_batches
    return iterate_batches(dataset, batch_size, shuffle)


def run_parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    from tt.rl import parse_mmlu_response
    return parse_mmlu_response(mmlu_example, model_output)


def run_parse_gsm8k_response(model_output: str) -> str | None:
    from tt.rl import parse_gsm8k_response
    return parse_gsm8k_response(model_output)


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    from tt.rl import compute_per_instance_dpo_loss
    return compute_per_instance_dpo_loss(
        lm, lm_ref, tokenizer, beta, prompt, response_chosen, response_rejected
    )

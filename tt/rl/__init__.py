"""RL / alignment: SFT, DPO, GRPO. Stubs only — Week 8."""

from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: Any,
) -> dict[str, Tensor]:
    raise NotImplementedError("tt.rl.tokenize_prompt_and_output (Week 8)")


def compute_entropy(logits: Tensor) -> Tensor:
    raise NotImplementedError("tt.rl.compute_entropy (Week 8)")


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool,
) -> dict[str, Tensor]:
    raise NotImplementedError("tt.rl.get_response_log_probs (Week 8)")


def masked_mean(
    tensor: Tensor, mask: Tensor, dim: int | None = None
) -> Tensor:
    raise NotImplementedError("tt.rl.masked_mean (Week 8)")


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> Tensor:
    raise NotImplementedError("tt.rl.masked_normalize (Week 8)")


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, dict[str, float]]:
    raise NotImplementedError("tt.rl.compute_group_normalized_rewards (Week 8)")


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
) -> Tensor:
    raise NotImplementedError("tt.rl.compute_naive_policy_gradient_loss (Week 8)")


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    raise NotImplementedError("tt.rl.compute_grpo_clip_loss (Week 8)")


def compute_policy_gradient_loss(
    policy_log_probs: Tensor,
    loss_type: str,
    raw_rewards: Tensor,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    raise NotImplementedError("tt.rl.compute_policy_gradient_loss (Week 8)")


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    raise NotImplementedError("tt.rl.sft_microbatch_train_step (Week 8)")


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    raise NotImplementedError("tt.rl.grpo_microbatch_train_step (Week 8)")


def get_packed_sft_dataset(
    tokenizer: Any,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    raise NotImplementedError("tt.rl.get_packed_sft_dataset (Week 8)")


def iterate_batches(
    dataset: Dataset, batch_size: int, shuffle: bool
):
    raise NotImplementedError("tt.rl.iterate_batches (Week 8)")


def parse_mmlu_response(
    mmlu_example: dict[str, Any], model_output: str
) -> str | None:
    raise NotImplementedError("tt.rl.parse_mmlu_response (Week 8)")


def parse_gsm8k_response(model_output: str) -> str | None:
    raise NotImplementedError("tt.rl.parse_gsm8k_response (Week 8)")


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: Any,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    raise NotImplementedError("tt.rl.compute_per_instance_dpo_loss (Week 8)")


__all__ = [
    "tokenize_prompt_and_output",
    "compute_entropy",
    "get_response_log_probs",
    "masked_mean",
    "masked_normalize",
    "compute_group_normalized_rewards",
    "compute_naive_policy_gradient_loss",
    "compute_grpo_clip_loss",
    "compute_policy_gradient_loss",
    "sft_microbatch_train_step",
    "grpo_microbatch_train_step",
    "get_packed_sft_dataset",
    "iterate_batches",
    "parse_mmlu_response",
    "parse_gsm8k_response",
    "compute_per_instance_dpo_loss",
]
